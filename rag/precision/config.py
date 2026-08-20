"""Configuration for the High-Precision Non-LLM RAG mode.

Follows the project's existing configuration pattern exactly (`core/config.py`): a frozen
dataclass built once behind `lru_cache`, every field read from an environment variable with
a coerced, documented default. No YAML file and no second configuration mechanism — a
second mechanism would be a second place for the truth to live, and this repository already
decided that question.

Every knob the pipeline exposes is here, and *nothing* here is read at import time by any
other module: `get_precision_config()` is called per question, so a deployment can change a
weight without a code change, and the benchmark can sweep variants with `replace()` without
touching the process-wide value.

The prefix is `PRECISION_RAG_` so an operator grepping their `.env` can see the whole mode
at once, and so no name here can ever collide with an existing setting.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, replace
from functools import lru_cache
import math
import os


def _flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _int(name: str, default: int, low: int, high: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return max(low, min(high, int(raw.strip())))
    except ValueError:
        # Clamp-and-continue, never raise. This mirrors `chunking.normalize_config`: a
        # malformed retrieval weight must not take the whole chat endpoint down, and the
        # documented default is a defensible answer where the typo is not.
        return default


def _float(name: str, default: float, low: float, high: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = float(raw.strip())
    except ValueError:
        return default
    # `float("nan")` and `float("inf")` PARSE, so the except branch never sees them, and
    # clamping does not help: every comparison against nan is False, so `min(high, nan)`
    # returns nan and it reaches a retrieval weight. A nan weight makes every score nan and
    # every ordering arbitrary — a failure with no error message anywhere.
    if not math.isfinite(value):
        return default
    return max(low, min(high, value))


def _csv(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return tuple(item.strip().lower() for item in raw.split(",") if item.strip())


# Reranker backends. `lexical` is the built-in interaction scorer — a cross-encoder in the
# architectural sense (it scores the (query, passage) PAIR jointly and its output cannot be
# precomputed per passage) implemented in pure Python with no model download and no new
# dependency. `http` posts to a local rerank service (text-embeddings-inference, Infinity,
# Xinference and BGE-reranker servers all expose the same shape) over the stdlib `urllib`
# seam this codebase already uses for every outbound call. `none` disables reranking, which
# is what the benchmark's ablation variants need.
RERANKER_LEXICAL = "lexical"
RERANKER_HTTP = "http"
RERANKER_NONE = "none"
RERANKERS = (RERANKER_LEXICAL, RERANKER_HTTP, RERANKER_NONE)


@dataclass(frozen=True)
class PrecisionConfig:
    """Every tunable of the high-precision pipeline, in pipeline order."""

    enabled: bool = True

    # --- retrieval -------------------------------------------------------------------
    bm25_enabled: bool = True
    dense_enabled: bool = True
    candidate_k: int = 100          # the pool that reaches the reranker
    final_k: int = 10               # what the mode returns

    # Okapi BM25. k1 controls term-frequency saturation, b the length normalisation;
    # 1.2/0.75 are the values the original TREC experiments settled on and are what every
    # comparable implementation defaults to.
    bm25_k1: float = 1.2
    bm25_b: float = 0.75

    # --- fusion weights --------------------------------------------------------------
    # Applied to MIN-MAX NORMALISED per-signal scores, which is the only reason a weighted
    # sum is defensible here at all: a raw BM25 score and a cosine are on incomparable
    # scales, and weighting them directly would be a number chosen to look reasonable.
    dense_weight: float = 1.0
    bm25_weight: float = 1.0
    metadata_weight: float = 0.35
    keyword_weight: float = 0.5

    # --- query understanding ---------------------------------------------------------
    normalization_enabled: bool = True
    query_expansion_enabled: bool = True
    expansion_max_terms: int = 12
    # Weight applied to a term that the user did not type. Below 1.0 on purpose: an
    # expansion is a guess about intent, and a guess must never outweigh the words someone
    # actually chose.
    expansion_term_weight: float = 0.45
    # The hard ceiling on what expansion may contribute, as a FRACTION of the strongest
    # typed-term score this query produced. Per-term down-weighting alone was not enough:
    # twelve terms at 0.45 outweigh a four-word question, and a passage matching many guesses
    # about the question beat one matching the question. Because the fraction is below 1.0, a
    # chunk found only through an expansion can be surfaced but can never outrank the best
    # literal match — the same inequality `_graph_rag` uses to bound its entity boost.
    expansion_max_fraction: float = 0.35
    # Pseudo-relevance feedback: take the top-N chunks of a first BM25 pass and add their
    # most distinctive terms to the query. Deterministic (no sampling, no model).
    prf_enabled: bool = True
    prf_documents: int = 5
    prf_terms: int = 6
    prf_term_weight: float = 0.25

    entity_extraction_enabled: bool = True

    # --- metadata filtering ----------------------------------------------------------
    metadata_filtering_enabled: bool = True
    # A filter only applies when it still leaves at least this many candidates. Filtering
    # is an optimisation, not a promise: a version or document-type constraint inferred
    # from the question is a guess, and a guess that empties the candidate set has turned a
    # ranking problem into a refusal. See `metadata.apply_filters`.
    metadata_filter_min_survivors: int = 12

    # --- reranking -------------------------------------------------------------------
    reranker_enabled: bool = True
    reranker_backend: str = RERANKER_LEXICAL
    reranker_top_k: int = 20        # how many survive the reranker into dedup/MMR
    reranker_model: str = "BAAI/bge-reranker-base"
    reranker_endpoint: str = ""     # e.g. http://localhost:8080/rerank
    reranker_timeout_seconds: int = 10
    # How much the reranker is allowed to move the retrieval ranking. 1.0 = the reranker
    # decides outright; 0.0 = it is ignored. Kept below 1.0 by default so a reranker that
    # is misconfigured or scoring badly degrades the ordering rather than replacing it.
    reranker_weight: float = 0.75

    # --- negative signals ------------------------------------------------------------
    negative_signals_enabled: bool = True
    # Deliberately EMPTY by default. Domain terms belong in a deployment's configuration,
    # never in the engine — a shipped list of "bad" words is a silent, unauditable opinion
    # about someone else's documents.
    negative_terms: tuple[str, ...] = ()
    negative_penalty: float = 0.35

    # --- diversity -------------------------------------------------------------------
    dedup_enabled: bool = True
    dedup_threshold: float = 0.90
    mmr_enabled: bool = True
    mmr_lambda: float = 0.7

    # --- structure -------------------------------------------------------------------
    parent_child_enabled: bool = True
    # How many consecutive sibling chunks form one parent window. Chunks ARE the children:
    # nothing is re-chunked, because re-chunking would change what every other mode
    # retrieves from the same knowledge base.
    parent_group_size: int = 3
    parent_max_chars: int = 2400

    # --- observability ---------------------------------------------------------------
    trace_enabled: bool = True
    # Trace payloads name chunk ids, counts and scores. Never passage text, never a
    # credential — the same rule `services/llm_provider.py` applies to prompts.
    trace_max_candidates_logged: int = 25

    def with_overrides(self, overrides: dict | None) -> "PrecisionConfig":
        """A copy with `overrides` applied, ignoring keys that are not fields.

        Used by the benchmark to sweep variants and by tests to pin one stage, so neither
        has to mutate the process-wide configuration to ask a question about one knob.
        """
        if not overrides:
            return self
        known = {field.name for field in fields(self)}
        clean = {key: value for key, value in overrides.items() if key in known}
        return replace(self, **clean) if clean else self


@lru_cache
def get_precision_config() -> PrecisionConfig:
    backend = os.getenv("PRECISION_RAG_RERANKER_BACKEND", RERANKER_LEXICAL).strip().lower()
    return PrecisionConfig(
        enabled=_flag("PRECISION_RAG_ENABLED", True),
        bm25_enabled=_flag("PRECISION_RAG_BM25", True),
        dense_enabled=_flag("PRECISION_RAG_DENSE", True),
        candidate_k=_int("PRECISION_RAG_CANDIDATE_K", 100, 10, 1000),
        final_k=_int("PRECISION_RAG_FINAL_K", 10, 1, 50),
        bm25_k1=_float("PRECISION_RAG_BM25_K1", 1.2, 0.0, 5.0),
        bm25_b=_float("PRECISION_RAG_BM25_B", 0.75, 0.0, 1.0),
        dense_weight=_float("PRECISION_RAG_DENSE_WEIGHT", 1.0, 0.0, 10.0),
        bm25_weight=_float("PRECISION_RAG_BM25_WEIGHT", 1.0, 0.0, 10.0),
        metadata_weight=_float("PRECISION_RAG_METADATA_WEIGHT", 0.35, 0.0, 10.0),
        keyword_weight=_float("PRECISION_RAG_KEYWORD_WEIGHT", 0.5, 0.0, 10.0),
        normalization_enabled=_flag("PRECISION_RAG_NORMALIZATION", True),
        query_expansion_enabled=_flag("PRECISION_RAG_QUERY_EXPANSION", True),
        expansion_max_terms=_int("PRECISION_RAG_EXPANSION_MAX_TERMS", 12, 0, 64),
        expansion_term_weight=_float("PRECISION_RAG_EXPANSION_WEIGHT", 0.45, 0.0, 1.0),
        expansion_max_fraction=_float("PRECISION_RAG_EXPANSION_MAX_FRACTION", 0.35, 0.0, 1.0),
        prf_enabled=_flag("PRECISION_RAG_PRF", True),
        prf_documents=_int("PRECISION_RAG_PRF_DOCUMENTS", 5, 1, 50),
        prf_terms=_int("PRECISION_RAG_PRF_TERMS", 6, 0, 32),
        prf_term_weight=_float("PRECISION_RAG_PRF_WEIGHT", 0.25, 0.0, 1.0),
        entity_extraction_enabled=_flag("PRECISION_RAG_ENTITY_EXTRACTION", True),
        metadata_filtering_enabled=_flag("PRECISION_RAG_METADATA_FILTERING", True),
        metadata_filter_min_survivors=_int("PRECISION_RAG_FILTER_MIN_SURVIVORS", 12, 1, 500),
        reranker_enabled=_flag("PRECISION_RAG_RERANKER", True),
        reranker_backend=backend if backend in RERANKERS else RERANKER_LEXICAL,
        reranker_top_k=_int("PRECISION_RAG_RERANKER_TOP_K", 20, 1, 200),
        # `.strip() or default`, not `getenv(name, default).strip()`: a variable that is
        # PRESENT but whitespace-only otherwise yields "" instead of the default, which is
        # the opposite of what every other helper in this file does with a blank value.
        reranker_model=os.getenv("PRECISION_RAG_RERANKER_MODEL", "").strip() or "BAAI/bge-reranker-base",
        reranker_endpoint=os.getenv("PRECISION_RAG_RERANKER_ENDPOINT", "").strip(),
        reranker_timeout_seconds=_int("PRECISION_RAG_RERANKER_TIMEOUT_SECONDS", 10, 1, 120),
        reranker_weight=_float("PRECISION_RAG_RERANKER_WEIGHT", 0.75, 0.0, 1.0),
        negative_signals_enabled=_flag("PRECISION_RAG_NEGATIVE_SIGNALS", True),
        negative_terms=_csv("PRECISION_RAG_NEGATIVE_TERMS", ()),
        negative_penalty=_float("PRECISION_RAG_NEGATIVE_PENALTY", 0.35, 0.0, 1.0),
        dedup_enabled=_flag("PRECISION_RAG_DEDUP", True),
        dedup_threshold=_float("PRECISION_RAG_DEDUP_THRESHOLD", 0.90, 0.0, 1.0),
        mmr_enabled=_flag("PRECISION_RAG_MMR", True),
        mmr_lambda=_float("PRECISION_RAG_MMR_LAMBDA", 0.7, 0.0, 1.0),
        parent_child_enabled=_flag("PRECISION_RAG_PARENT_CHILD", True),
        parent_group_size=_int("PRECISION_RAG_PARENT_GROUP_SIZE", 3, 1, 20),
        parent_max_chars=_int("PRECISION_RAG_PARENT_MAX_CHARS", 2400, 200, 20000),
        trace_enabled=_flag("PRECISION_RAG_TRACE", True),
        trace_max_candidates_logged=_int("PRECISION_RAG_TRACE_MAX_CANDIDATES", 25, 0, 200),
    )
