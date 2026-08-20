"""High-Precision Non-LLM RAG — an isolated, evaluation-driven retrieval mode.

    User query
       -> query normalization            (normalize.py)
       -> deterministic query expansion   (expansion.py)
       -> entity / keyword extraction     (pipeline.py + the host's extractor)
       -> metadata filtering              (metadata.py)
       -> BM25 + dense retrieval          (bm25.py + the host's embeddings)
       -> candidate pool, top 50-100      (pipeline.py)
       -> cross-encoder rerank            (rerank.py)
       -> deduplication                   (diversity.py)
       -> MMR / diversity                 (diversity.py)
       -> parent-chunk recovery           (parents.py)
       -> final top-K

WHAT MAKES IT ISOLATED. This package imports nothing from `rag.service`, `models`,
`sqlalchemy` or FastAPI. The host injects a tokenizer, a term-map accessor, an optional
entity extractor and an optional list of dense candidates; everything else is arithmetic
over data structures. `rag.service` imports this package — never the reverse — so there is
one integration point (a single `elif` in `_plan_answer`), no import cycle, and no path by
which a change in here can alter what any existing mode retrieves.

WHAT MAKES IT NON-LLM. Expansion is dictionary- and corpus-driven; the default reranker is
a pure-Python interaction scorer; the optional HTTP reranker backend talks to a
cross-encoder service that emits relevance scores and cannot produce text. No prompt is
constructed and no completion is requested anywhere in this package.
"""
from rag.precision.config import (
    PrecisionConfig,
    RERANKER_HTTP,
    RERANKER_LEXICAL,
    RERANKER_NONE,
    get_precision_config,
)
from rag.precision.index import clear_cache as clear_index_cache, invalidate as invalidate_index
from rag.precision.pipeline import PipelineInputs, retrieve
from rag.precision.types import (
    Candidate,
    ChunkLike,
    ChunkMetadata,
    PrecisionOutcome,
    PrecisionResult,
    PrecisionTrace,
)


# The mode id, defined HERE rather than in `rag/service.py`, so the string and the pipeline
# that implements it cannot drift apart. `rag.service` imports it.
RAG_MODE_HIGH_PRECISION = "high_precision"

# What the mode is called in prose.
#
# "High Precision", NOT "High-Precision", and the space is load-bearing: `_compose_answer`
# in `rag/service.py` derives the label it prints as `mode.replace("_", " ").title()`, which
# for this id yields exactly this string. Spelling it with a hyphen anywhere else would put
# two names for one mode in front of the same user — the answer body saying one thing and
# the picker, the badge and the indexing catalogue saying another.
RAG_MODE_HIGH_PRECISION_LABEL = "High Precision"


def evidence_for_synthesis(outcome: PrecisionOutcome, limit: int = 5, max_chars: int = 1800) -> list[dict]:
    """The retrieved passages in the shape `rag.service` hands to an LLM for synthesis.

    Separate from `rag.service._llm_evidence` on purpose — it is not a copy, it is the one
    place the PARENT CONTEXT is allowed to matter. Parent recovery exists so a child chunk
    selected for a precise match arrives with the surrounding text that makes it mean
    something; sending only the child to a synthesiser would throw that away at the last
    step, and sending it through the shared helper would mean editing a function five other
    modes depend on.

    The citation anchor is still the CHILD (`_citations` in `rag.service` reads
    `result.chunk`), so the evidence panel highlights the passage that actually matched.
    """
    evidence: list[dict] = []
    for result in outcome.results[:limit]:
        text = result.parent_context or result.chunk.content or ""
        evidence.append(
            {
                "source_name": result.chunk.source_name,
                "chunk_index": (result.chunk.chunk_index or 0) + 1,
                "modality": result.chunk.modality,
                "content": text[:max_chars],
            }
        )
    return evidence


__all__ = [
    "Candidate",
    "ChunkLike",
    "ChunkMetadata",
    "PipelineInputs",
    "PrecisionConfig",
    "PrecisionOutcome",
    "PrecisionResult",
    "PrecisionTrace",
    "RAG_MODE_HIGH_PRECISION",
    "RAG_MODE_HIGH_PRECISION_LABEL",
    "RERANKER_HTTP",
    "RERANKER_LEXICAL",
    "RERANKER_NONE",
    "clear_index_cache",
    "invalidate_index",
    "evidence_for_synthesis",
    "get_precision_config",
    "retrieve",
]
