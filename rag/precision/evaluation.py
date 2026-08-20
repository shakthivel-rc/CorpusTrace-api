"""Retrieval evaluation for the High-Precision mode — metrics, and the ablation harness.

This exists because the brief asks for an "evaluation-driven" pipeline, and a pipeline is
only evaluation-driven if the evaluation can say which STAGE earned its place. Every knob
in `PrecisionConfig` is a hypothesis; `compare()` turns each into a measurement by running
the same benchmark through variants that differ in exactly one stage.

    bm25                 lexical only
    dense                embeddings only
    bm25_dense           both, fused
    bm25_dense_rerank    both, fused, cross-encoder reranked
    full                 everything: expansion, PRF, filtering, rerank, dedup, MMR, parents

The metrics are the standard IR set (Recall@K, Precision@K, MRR, nDCG@K, Hit Rate). Nothing
here is specific to this application except the shape of a result, and nothing here touches
the existing RAG modes — the harness runs the precision pipeline directly, so a benchmark
run cannot change what any other mode returns and a change to another mode cannot move these
numbers.

Ground truth is chunk ids and/or document ids. Both are supported because they answer
different questions: chunk-level tells you whether the *passage* was found, document-level
whether the right *source* was surfaced at all, and a pipeline can be good at one and bad at
the other.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from typing import Callable, Iterable, Sequence


@dataclass
class BenchmarkCase:
    """One labelled query.

    `expected_chunks` is the strict ground truth; `expected_documents` the lenient one. A
    case may supply either or both — a benchmark written before chunk ids were stable can
    still measure document-level retrieval, which is most of the value.

    `expected_chunk_groups` expresses INTERCHANGEABLE ground truth: a list of groups where
    finding any one member counts as finding that group. It exists because this application's
    chunker overlaps by design, so a real corpus reliably contains passages that say the same
    thing, and a pipeline with a deduplication stage is SUPPOSED to return only one of them.
    Scored as a flat set, that correct behaviour reads as a recall miss — measured on the
    bundled benchmark, deduplication appeared to cost 1.6 points of Recall@5 while doing
    exactly its job. A benchmark that punishes the behaviour it was built to verify is
    measuring the harness, not the retriever.

    Every entry in `expected_chunks` is its own singleton group, so a case that does not use
    the feature scores identically to before it existed.
    """

    query: str
    expected_documents: tuple[str, ...] = ()
    expected_chunks: tuple[str, ...] = ()
    expected_chunk_groups: tuple[tuple[str, ...], ...] = ()
    resource_id: str | None = None
    notes: str = ""

    @classmethod
    def from_dict(cls, raw: dict) -> "BenchmarkCase":
        groups = []
        for group in raw.get("expected_chunk_groups", []) or ():
            if isinstance(group, (list, tuple)):
                members = tuple(str(item) for item in group if str(item).strip())
                if members:
                    groups.append(members)
            elif str(group).strip():
                groups.append((str(group),))
        return cls(
            query=str(raw.get("query", "")).strip(),
            expected_documents=tuple(str(item) for item in raw.get("expected_documents", []) or ()),
            expected_chunks=tuple(str(item) for item in raw.get("expected_chunks", []) or ()),
            expected_chunk_groups=tuple(groups),
            resource_id=raw.get("resource_id"),
            notes=str(raw.get("notes", "")),
        )

    @property
    def chunk_groups(self) -> tuple[frozenset[str], ...]:
        """The chunk ground truth as interchangeability groups."""
        return tuple(
            [frozenset({chunk_id}) for chunk_id in self.expected_chunks]
            + [frozenset(group) for group in self.expected_chunk_groups]
        )

    @property
    def is_usable(self) -> bool:
        return bool(self.query) and bool(
            self.expected_documents or self.expected_chunks or self.expected_chunk_groups
        )


def load_cases(path: str) -> list[BenchmarkCase]:
    """Read a benchmark file: a JSON list, or `{"cases": [...]}`."""
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    rows = raw.get("cases", []) if isinstance(raw, dict) else raw
    cases = [BenchmarkCase.from_dict(row) for row in rows if isinstance(row, dict)]
    return [case for case in cases if case.is_usable]


# --------------------------------------------------------------------------------------
# Metrics. Pure functions of (ranked_ids, relevant_ids) — no I/O, no configuration.
# --------------------------------------------------------------------------------------

def recall_at_k(ranked: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Share of the relevant items that appear in the top K.

    Denominator is the number of RELEVANT items, not K. With one relevant chunk this is
    identical to hit rate; with several it is the metric that notices a pipeline finding one
    of five and calling it a success.
    """
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    found = sum(1 for item in ranked[:k] if item in relevant_set)
    return found / len(relevant_set)


def precision_at_k(ranked: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Share of the top K that is relevant. Denominator is a flat K, as in the IR literature.

    Dividing by `min(k, len(ranked))` was tried first and is wrong for THIS harness even
    though it is kinder to a system that returns a short list: it makes two variants
    incomparable. Measured on the bundled demo, the `bm25` ablation returned fewer than five
    candidates on most queries and scored 0.4367 at K=5, while `full` — which retrieves
    strictly more and ranks it better on every other metric — scored 0.3017 purely because
    MMR filled the window. A metric that penalises a variant for finding more is worse than
    useless in an ablation, which is the only thing this function is used for.

    The consequence to remember when reading the numbers: with one relevant chunk per query,
    P@5 is capped at 0.2 and P@10 at 0.1. Precision@K here measures ranking density, not
    correctness — Recall@1, MRR and nDCG are what say whether the answer was found.
    """
    if k <= 0 or not ranked:
        return 0.0
    relevant_set = set(relevant)
    return sum(1 for item in ranked[:k] if item in relevant_set) / k


def reciprocal_rank(ranked: Sequence[str], relevant: Iterable[str]) -> float:
    relevant_set = set(relevant)
    for position, item in enumerate(ranked, start=1):
        if item in relevant_set:
            return 1.0 / position
    return 0.0


def hit_rate(ranked: Sequence[str], relevant: Iterable[str], k: int) -> float:
    relevant_set = set(relevant)
    return 1.0 if any(item in relevant_set for item in ranked[:k]) else 0.0


def ndcg_at_k(ranked: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Binary-gain nDCG@K.

    Binary rather than graded because the ground truth here is binary — a chunk either does
    or does not contain the answer. Inventing graded relevance would put a judgement into
    the metric that the benchmark file does not carry.
    """
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    gain = sum(
        1.0 / math.log2(position + 1)
        for position, item in enumerate(ranked[:k], start=1)
        if item in relevant_set
    )
    ideal = sum(1.0 / math.log2(position + 1) for position in range(1, min(len(relevant_set), k) + 1))
    return (gain / ideal) if ideal > 0 else 0.0


# --------------------------------------------------------------------------------------
# Group-aware variants. Identical to the flat metrics above when every group is a singleton,
# which is what `BenchmarkCase.chunk_groups` produces for a plain `expected_chunks` list — so
# nothing about an existing benchmark file's numbers changes by these existing.
# --------------------------------------------------------------------------------------


def _first_hits(ranked: Sequence[str], groups: Sequence[frozenset[str]]) -> dict[int, int]:
    """`{rank (1-based): group index}` for the FIRST time each group is hit.

    Only the first hit of a group counts. A second near-duplicate from the same group adds
    no information to the reader, so crediting it would reward exactly the redundancy the
    deduplication stage exists to remove.
    """
    hits: dict[int, int] = {}
    seen: set[int] = set()
    for position, item in enumerate(ranked, start=1):
        for group_index, group in enumerate(groups):
            if group_index in seen or item not in group:
                continue
            hits[position] = group_index
            seen.add(group_index)
            break
    return hits


def group_recall_at_k(ranked: Sequence[str], groups: Sequence[frozenset[str]], k: int) -> float:
    if not groups:
        return 0.0
    return sum(1 for rank in _first_hits(ranked, groups) if rank <= k) / len(groups)


def group_precision_at_k(ranked: Sequence[str], groups: Sequence[frozenset[str]], k: int) -> float:
    if k <= 0 or not ranked:
        return 0.0
    return sum(1 for rank in _first_hits(ranked, groups) if rank <= k) / k


def group_reciprocal_rank(ranked: Sequence[str], groups: Sequence[frozenset[str]]) -> float:
    hits = _first_hits(ranked, groups)
    return 1.0 / min(hits) if hits else 0.0


def group_hit_rate(ranked: Sequence[str], groups: Sequence[frozenset[str]], k: int) -> float:
    return 1.0 if any(rank <= k for rank in _first_hits(ranked, groups)) else 0.0


def group_ndcg_at_k(ranked: Sequence[str], groups: Sequence[frozenset[str]], k: int) -> float:
    if not groups:
        return 0.0
    gain = sum(1.0 / math.log2(rank + 1) for rank in _first_hits(ranked, groups) if rank <= k)
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(groups), k) + 1))
    return (gain / ideal) if ideal > 0 else 0.0


DEFAULT_KS = (1, 5, 10)


@dataclass
class EvaluationReport:
    variant: str
    cases: int = 0
    metrics: dict[str, float] = field(default_factory=dict)
    per_case: list[dict] = field(default_factory=list)
    latency_ms: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "variant": self.variant,
            "cases": self.cases,
            "metrics": {key: round(value, 4) for key, value in self.metrics.items()},
            "latency_ms": {key: round(value, 3) for key, value in self.latency_ms.items()},
            "per_case": self.per_case,
        }


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = max(0, min(len(ordered) - 1, int(round(fraction * (len(ordered) - 1)))))
    return ordered[position]


def evaluate(
    cases: Sequence[BenchmarkCase],
    run: Callable[[BenchmarkCase], tuple[list[str], list[str], float]],
    *,
    variant: str = "full",
    ks: Sequence[int] = DEFAULT_KS,
) -> EvaluationReport:
    """Score one variant over a set of cases.

    `run(case)` returns `(ranked_chunk_ids, ranked_document_ids, elapsed_ms)`. Keeping the
    runner injected is what lets the same metrics score an in-memory fixture in a unit test
    and a real knowledge base from the CLI, with no branch inside the scoring code deciding
    which it is looking at.

    Document ids are DEDUPLICATED IN RANK ORDER before scoring: ten chunks from one file are
    one document retrieved at rank 1, and counting them as ten would make document-level
    precision a measure of how repetitive the chunker is.
    """
    report = EvaluationReport(variant=variant)
    totals: dict[str, float] = {}
    latencies: list[float] = []

    for case in cases:
        ranked_chunks, ranked_documents, elapsed = run(case)
        ranked_documents = list(dict.fromkeys(ranked_documents))
        latencies.append(elapsed)
        row: dict = {"query": case.query, "latency_ms": round(elapsed, 3)}

        groups = case.chunk_groups
        if groups:
            for k in ks:
                row[f"chunk_recall@{k}"] = group_recall_at_k(ranked_chunks, groups, k)
                row[f"chunk_precision@{k}"] = group_precision_at_k(ranked_chunks, groups, k)
                row[f"chunk_ndcg@{k}"] = group_ndcg_at_k(ranked_chunks, groups, k)
                row[f"chunk_hit@{k}"] = group_hit_rate(ranked_chunks, groups, k)
            row["chunk_mrr"] = group_reciprocal_rank(ranked_chunks, groups)

        if case.expected_documents:
            for k in ks:
                row[f"doc_recall@{k}"] = recall_at_k(ranked_documents, case.expected_documents, k)
                row[f"doc_precision@{k}"] = precision_at_k(ranked_documents, case.expected_documents, k)
                row[f"doc_ndcg@{k}"] = ndcg_at_k(ranked_documents, case.expected_documents, k)
                row[f"doc_hit@{k}"] = hit_rate(ranked_documents, case.expected_documents, k)
            row["doc_mrr"] = reciprocal_rank(ranked_documents, case.expected_documents)

        for key, value in row.items():
            if isinstance(value, (int, float)) and key not in {"latency_ms"}:
                totals[key] = totals.get(key, 0.0) + float(value)
        report.per_case.append(row)

    report.cases = len(cases)
    if cases:
        # Averaged over the cases that CARRY that ground truth, not over every case: a
        # benchmark mixing chunk-labelled and document-labelled cases would otherwise report
        # chunk recall diluted by the cases that never had a chunk label to find.
        chunk_cases = sum(1 for case in cases if case.chunk_groups) or 1
        doc_cases = sum(1 for case in cases if case.expected_documents) or 1
        for key, total in totals.items():
            divisor = chunk_cases if key.startswith("chunk_") else doc_cases
            report.metrics[key] = total / divisor
    report.latency_ms = {
        "mean": (sum(latencies) / len(latencies)) if latencies else 0.0,
        "p50": _percentile(latencies, 0.5),
        "p95": _percentile(latencies, 0.95),
        "max": max(latencies) if latencies else 0.0,
    }
    return report


# Ablation variants. Each is a set of `PrecisionConfig` overrides; the pipeline is otherwise
# identical, so any difference in the metrics is attributable to the stage that was turned
# off. `full` is deliberately an empty override set — it must be whatever the deployment's
# configuration actually is, or the benchmark measures a pipeline nobody runs.
VARIANTS: dict[str, dict] = {
    "bm25": {
        "dense_enabled": False,
        "query_expansion_enabled": False,
        "prf_enabled": False,
        "metadata_filtering_enabled": False,
        "reranker_enabled": False,
        "dedup_enabled": False,
        "mmr_enabled": False,
        "keyword_weight": 0.0,
        "metadata_weight": 0.0,
    },
    "dense": {
        "bm25_enabled": False,
        "query_expansion_enabled": False,
        "prf_enabled": False,
        "metadata_filtering_enabled": False,
        "reranker_enabled": False,
        "dedup_enabled": False,
        "mmr_enabled": False,
        "keyword_weight": 0.0,
        "metadata_weight": 0.0,
    },
    "bm25_dense": {
        "query_expansion_enabled": False,
        "prf_enabled": False,
        "metadata_filtering_enabled": False,
        "reranker_enabled": False,
        "dedup_enabled": False,
        "mmr_enabled": False,
        "keyword_weight": 0.0,
        "metadata_weight": 0.0,
    },
    "bm25_dense_rerank": {
        "query_expansion_enabled": False,
        "prf_enabled": False,
        "metadata_filtering_enabled": False,
        "dedup_enabled": False,
        "mmr_enabled": False,
        "keyword_weight": 0.0,
        "metadata_weight": 0.0,
    },
    "full": {},
}


def compare(
    cases: Sequence[BenchmarkCase],
    run_variant: Callable[[str, dict, BenchmarkCase], tuple[list[str], list[str], float]],
    *,
    variants: Sequence[str] = tuple(VARIANTS),
    ks: Sequence[int] = DEFAULT_KS,
) -> dict[str, EvaluationReport]:
    """Run every variant over the same cases and return one report each."""
    reports: dict[str, EvaluationReport] = {}
    for variant in variants:
        # A copy: the runner receives this dict, and a runner that mutated it would corrupt
        # the module-level table for every later variant in the same process.
        overrides = dict(VARIANTS.get(variant, {}))
        reports[variant] = evaluate(
            cases,
            lambda case, _variant=variant, _overrides=overrides: run_variant(_variant, _overrides, case),
            variant=variant,
            ks=ks,
        )
    return reports


def format_table(reports: dict[str, EvaluationReport], keys: Sequence[str] | None = None) -> str:
    """A fixed-width comparison table, for a terminal and for pasting into a report."""
    if not reports:
        return "(no results)"
    default_keys = [
        "chunk_recall@1", "chunk_recall@5", "chunk_recall@10",
        "chunk_precision@5", "chunk_mrr", "chunk_ndcg@10",
        "doc_recall@5", "doc_mrr",
    ]
    available = {key for report in reports.values() for key in report.metrics}
    columns = [key for key in (keys or default_keys) if key in available]
    header = "variant".ljust(20) + "".join(column.rjust(18) for column in columns) + "latency p50".rjust(14)
    lines = [header, "-" * len(header)]
    for name, report in reports.items():
        row = name.ljust(20)
        row += "".join(f"{report.metrics.get(column, 0.0):.4f}".rjust(18) for column in columns)
        row += f"{report.latency_ms.get('p50', 0.0):.1f}ms".rjust(14)
        lines.append(row)
    return "\n".join(lines)
