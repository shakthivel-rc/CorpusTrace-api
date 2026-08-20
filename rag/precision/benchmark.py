"""Benchmark runner for the High-Precision mode — `python -m rag.precision.benchmark`.

    python -m rag.precision.benchmark --demo
    python -m rag.precision.benchmark --cases cases.json --corpus corpus.json
    python -m rag.precision.benchmark --cases cases.json --resource-id <id> --user-id <id>

THIS MODULE IS THE ONE PART OF `rag/precision/` THAT REACHES OUTSIDE THE PACKAGE. It imports
`rag.service` for the tokenizer (and, in `--resource-id` mode, for the database and the
embedding call). That is deliberate and it does not weaken the package's isolation, because
nothing imports it: `rag/precision/__init__.py` does not, so `import rag.precision` still
pulls in no ORM, no FastAPI and no `rag.service`. It is a tool that happens to live beside
the code it measures.

Why it borrows the tokenizer rather than defining one: the index is built from `terms_json`,
which ingestion writes with `rag.service._term_counts`. A benchmark that tokenized queries
its own way would measure a retriever nobody runs, and the discrepancy would look like a
scoring bug rather than a harness bug.

Ablation is the point. `--variants` runs the same cases through BM25 alone, dense alone,
both fused, both fused + reranked, and the full pipeline, so every stage has to justify
itself with a number rather than with an argument.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from rag.precision import evaluation
from rag.precision.config import PrecisionConfig, get_precision_config
from rag.precision.index import clear_cache
from rag.precision.pipeline import PipelineInputs, retrieve


DEMO_PATH = Path(__file__).parent / "benchmarks" / "demo.json"


@dataclass
class StubChunk:
    """A chunk with no database behind it, for `--corpus` and `--demo` runs.

    Satisfies `types.ChunkLike` structurally, which is the whole reason that protocol is
    structural: the same pipeline code runs against these and against real `DocumentChunk`
    rows with no branch anywhere deciding which it is looking at.
    """

    id: str
    file_id: str | None
    chunk_index: int
    source_name: str
    modality: str
    title: str | None
    content: str
    contextual_content: str
    page_start: int | None = None
    page_end: int | None = None
    char_start: int | None = None
    char_end: int | None = None


def _prepare_offline_environment() -> None:
    """Let `rag.service` import without a configured deployment.

    `db/session.py` and `utils/token.py` read the environment at IMPORT time, so a `--demo`
    run on a machine with no `.env` would abort before reaching any retrieval code. These
    are `setdefault` calls: a real `.env` always wins, and nothing here touches a database.
    """
    import base64

    os.environ.setdefault("SECRET_KEY", base64.urlsafe_b64encode(b"0" * 32).decode())
    os.environ.setdefault("DATABASE_URL", "sqlite://")
    os.environ.setdefault("ENVIRONMENT", "test")


def _contextualize(resource_name: str, chunk: dict, index: int) -> str:
    """The same six-line header ingestion prepends, so term statistics match production."""
    return (
        f"Resource: {resource_name}\n"
        f"Source file: {chunk.get('source_name', 'document')}\n"
        f"Title: {chunk.get('title') or chunk.get('source_name', 'document')}\n"
        f"Modality: {chunk.get('modality', 'text')}\n"
        f"Chunk: {index}\n"
        f"Content: {chunk['content']}"
    )


def load_corpus(payload: dict) -> list[StubChunk]:
    resource_name = payload.get("resource_name", "Benchmark")
    chunks: list[StubChunk] = []
    for index, raw in enumerate(payload.get("chunks", [])):
        content = str(raw.get("content", "")).strip()
        if not content:
            continue
        chunks.append(
            StubChunk(
                id=str(raw.get("id") or f"chunk-{index}"),
                file_id=str(raw.get("file_id") or raw.get("document_id") or "doc-1"),
                chunk_index=int(raw.get("chunk_index", index)),
                source_name=str(raw.get("source_name", "document.txt")),
                modality=str(raw.get("modality", "text")),
                title=raw.get("title"),
                content=content,
                contextual_content=_contextualize(resource_name, raw, index),
                page_start=raw.get("page_start"),
                page_end=raw.get("page_end"),
            )
        )
    return chunks


def _offline_runner(chunks, tokenize, term_counts, extract_entities, base_config: PrecisionConfig):
    """A `run_variant(variant, overrides, case)` closure over an in-memory corpus."""
    terms_by_id = {chunk.id: term_counts(chunk.contextual_content) for chunk in chunks}

    def run_variant(variant: str, overrides: dict, case) -> tuple[list[str], list[str], float]:
        # The index is keyed by resource id and the variants differ in how the corpus is
        # *derived* (entity extraction can be off), so each variant gets its own key and a
        # cold cache. Sharing one would let variant A's index answer variant B's query.
        clear_cache()
        started = time.perf_counter()
        outcome = retrieve(
            PipelineInputs(
                resource_id=f"benchmark::{variant}",
                query=case.query,
                chunks=chunks,
                tokenize=tokenize,
                terms_of=lambda chunk: terms_by_id[chunk.id],
                extract_entities=extract_entities,
                dense_candidates=[],
                config=base_config.with_overrides(overrides),
                # No operator dictionary: a benchmark that silently picked up whatever
                # PRECISION_RAG_DICTIONARY_PATH happened to point at would not be comparable
                # between two machines.
                dictionary={},
            )
        )
        elapsed = (time.perf_counter() - started) * 1000
        return (
            [result.chunk.id for result in outcome.results],
            [result.chunk.file_id or "" for result in outcome.results],
            elapsed,
        )

    return run_variant


def _database_runner(user_id: str, resource_id: str, base_config: PrecisionConfig):
    """A `run_variant` closure over a real knowledge base, dense side included."""
    from db.session import SessionLocal
    from models.rag import DocumentChunk
    from rag import service

    def run_variant(variant: str, overrides: dict, case) -> tuple[list[str], list[str], float]:
        clear_cache()
        db = SessionLocal()
        try:
            chunks = (
                db.query(DocumentChunk)
                .options(*service._DEFER_VECTORS)
                .filter(DocumentChunk.resource_id == (case.resource_id or resource_id))
                .order_by(DocumentChunk.chunk_index.asc())
                .all()
            )
            config = base_config.with_overrides(overrides)
            started = time.perf_counter()
            semantic = (
                service._semantic_retrieval(
                    db, user_id, case.resource_id or resource_id, case.query, chunks,
                    candidates=config.candidate_k,
                )
                if config.dense_enabled
                else service._SemanticRetrieval()
            )
            outcome = service._high_precision(
                case.resource_id or resource_id, case.query, chunks, semantic, overrides=overrides
            )
            elapsed = (time.perf_counter() - started) * 1000
            return (
                [result.chunk.id for result in outcome.results],
                [result.chunk.file_id or "" for result in outcome.results],
                elapsed,
            )
        finally:
            db.close()

    return run_variant


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark the High-Precision RAG mode")
    parser.add_argument("--demo", action="store_true", help="Run the bundled synthetic corpus and cases")
    parser.add_argument("--cases", help="Benchmark file: a JSON list, or {'cases': [...]}")
    parser.add_argument("--corpus", help="Offline corpus JSON: {'chunks': [{'id','content',...}]}")
    parser.add_argument("--resource-id", help="Run against a real knowledge base instead")
    parser.add_argument("--user-id", help="Owner of --resource-id (needed for the embedding call)")
    parser.add_argument(
        "--variants",
        default=",".join(evaluation.VARIANTS),
        help="Comma-separated subset of: " + ", ".join(evaluation.VARIANTS),
    )
    parser.add_argument("--json", help="Write the full report to this path")
    parser.add_argument("--ks", default="1,5,10", help="Cut-offs for @K metrics")
    args = parser.parse_args(argv)

    if not args.demo and not args.cases:
        parser.error("pass --demo, or --cases with either --corpus or --resource-id")
    if args.cases and not (args.corpus or args.resource_id):
        parser.error("--cases needs --corpus (offline) or --resource-id (a real base)")
    if args.resource_id and not args.user_id:
        parser.error("--resource-id needs --user-id: the dense side embeds with that user's credentials")

    if not args.resource_id:
        _prepare_offline_environment()

    # Imported here, not at module scope, so `--demo` can seed the environment first.
    from rag.service import _extract_entities, _term_counts, _tokenize

    ks = tuple(int(value) for value in args.ks.split(",") if value.strip())
    # Strip THEN test THEN keep the stripped name. Keeping the raw token meant
    # `--variants "bm25, dense"` passed the membership check and then looked up " dense",
    # which `VARIANTS.get` answers with {} — the full pipeline, silently mislabelled.
    variants = [
        name.strip() for name in args.variants.split(",") if name.strip() in evaluation.VARIANTS
    ]
    base_config = get_precision_config()

    if args.demo:
        payload = json.loads(DEMO_PATH.read_text(encoding="utf-8"))
        cases = [evaluation.BenchmarkCase.from_dict(row) for row in payload.get("cases", [])]
        cases = [case for case in cases if case.is_usable]
        runner = _offline_runner(load_corpus(payload), _tokenize, _term_counts, _extract_entities, base_config)
        title = f"demo corpus ({len(payload.get('chunks', []))} chunks)"
    elif args.corpus:
        cases = evaluation.load_cases(args.cases)
        payload = json.loads(Path(args.corpus).read_text(encoding="utf-8"))
        runner = _offline_runner(load_corpus(payload), _tokenize, _term_counts, _extract_entities, base_config)
        title = f"{args.corpus} ({len(payload.get('chunks', []))} chunks)"
    else:
        cases = evaluation.load_cases(args.cases)
        runner = _database_runner(args.user_id, args.resource_id, base_config)
        title = f"resource {args.resource_id}"

    if not cases:
        print("No usable cases: every case needs a query and at least one expected chunk or document.")
        return 1

    print(f"High-Precision RAG benchmark — {title}, {len(cases)} cases, variants: {', '.join(variants)}\n")
    reports = evaluation.compare(cases, runner, variants=variants, ks=ks)
    print(evaluation.format_table(reports))

    # An all-zero row is almost always "this retriever had nothing to work with", not "this
    # retriever is bad". Saying so beside the table stops a reader diagnosing a corpus
    # property as a code defect — the offline runner supplies no vectors at all, so `dense`
    # is expected to be empty there.
    empty = [name for name, report in reports.items() if not any(report.metrics.values())]
    if empty:
        print(
            "\nnote: "
            + ", ".join(empty)
            + " returned nothing on every case. For `dense` this is expected unless the corpus "
              "was indexed with an embedding model — the offline runner supplies no vectors."
        )

    if args.json:
        Path(args.json).write_text(
            json.dumps({name: report.to_dict() for name, report in reports.items()}, indent=2),
            encoding="utf-8",
        )
        print(f"\nFull report written to {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
