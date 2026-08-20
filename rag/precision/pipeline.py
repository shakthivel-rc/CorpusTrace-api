"""The High-Precision Non-LLM RAG pipeline.

    query -> normalize -> expand -> entities -> metadata filter
          -> [BM25 || dense] -> candidate pool -> cross-encoder rerank
          -> dedup -> MMR -> parent recovery -> top-K

Nothing in this module imports `rag.service`, `models` or `sqlalchemy`. Everything that
needs the database or the network arrives as a callable on `PipelineInputs`, which is what
makes the whole pipeline unit-testable against plain objects and what stops the import cycle
that would otherwise exist (`rag.service` imports this package to register the mode).

NO LANGUAGE MODEL RUNS ANYWHERE IN HERE. Query expansion is dictionary- and corpus-driven,
the reranker is an interaction scorer or a non-generative cross-encoder service, and every
threshold is a configured number. That is a deliberate property, not an accident of what was
available: it makes the mode reproducible, free to run, and benchmarkable — an identical
corpus and an identical query produce an identical ranking, which is the only condition
under which `benchmark.py`'s numbers attribute a change to a change.
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Callable, Sequence

from rag.precision import bm25, diversity, expansion, index as index_module, metadata as metadata_module, parents, rerank
from rag.precision.config import PrecisionConfig, get_precision_config
from rag.precision.normalize import correct_spelling, normalize_query
from rag.precision.types import Candidate, ChunkLike, PrecisionOutcome, PrecisionResult, PrecisionTrace


logger = logging.getLogger("corpustrace.rag.precision")


@dataclass
class PipelineInputs:
    """Everything one run needs, with the database and network held at arm's length."""

    resource_id: str
    query: str
    chunks: list[ChunkLike]
    # The SAME tokenizer the index was built with. Injected rather than reimplemented: a
    # second tokenizer would mean the query and the corpus disagreed about what a word is,
    # and the symptom would be a term that "matches nothing" while sitting in the document.
    tokenize: Callable[[str], list[str]]
    terms_of: Callable[[ChunkLike], dict[str, int]]
    extract_entities: Callable[[str], list[str]] | None = None
    # (chunk_id, similarity) pairs, already ranked, from whatever dense retriever the host
    # application has. Empty or None means this knowledge base has no embeddings — the
    # normal case — and the pipeline runs lexically, which is the whole reason the dense
    # side is optional rather than required.
    dense_candidates: Sequence[tuple[str, float]] | None = None
    # Optional: a chunk's vector, for MMR redundancy. Falls back to term-set overlap.
    vector_of: Callable[[str], Sequence[float] | None] | None = None
    config: PrecisionConfig | None = None
    dictionary: dict[str, tuple[str, ...]] | None = None


# The floor a candidate that DID score on a signal is normalised to. Min-max maps a pool's
# minimum to exactly 0.0 by construction, which is indistinguishable from "this signal did
# not fire at all" — and for a chunk the dense retriever ranked at 0.93 but that shares no
# word with the query, every other signal is genuinely 0.0 too, so it reached the user with
# a reported score of 0.0000 beside the reason "dense retrieval". Ordering is untouched: the
# map is affine, so it is the same ranking read on a scale that distinguishes "weakest" from
# "absent".
MIN_NORMALIZED = 0.05


def _normalize_scores(values: dict[str, float]) -> dict[str, float]:
    """Min-max onto 0..1, so weights in the fusion mean what they say.

    A BM25 score is an unbounded sum over query terms; a cosine is bounded at 1.0; a
    metadata score is a hand-built 0..1. Weighting those directly would be picking numbers
    that look reasonable — the weights would silently encode the corpus's score magnitudes
    rather than a preference between signals.

    A value of exactly 0.0 means the signal did not fire for that candidate and is mapped to
    0.0; everything above it lands in [MIN_NORMALIZED, 1.0].
    """
    if not values:
        return {}
    low = min(values.values())
    high = max(values.values())
    span = high - low
    if span <= 0:
        # Every candidate scored the same. `1.0` for a positive score (they are all equally
        # good), `0.0` for none — never 0.5, which would give a signal that fired for
        # nobody the same weight as one that fired for everybody.
        return {key: (1.0 if value > 0 else 0.0) for key, value in values.items()}
    scale = 1.0 - MIN_NORMALIZED
    return {
        key: (MIN_NORMALIZED + scale * (value - low) / span) if value > 0 else 0.0
        for key, value in values.items()
    }


# `math.sumprod` is a C-level fused dot product (3.12+). CLAUDE.md §9 records it as a
# measured win on exactly this loop, and it is stdlib — the isolation rule keeps
# `rag.vectors` out of this package, not the standard library. The fallback keeps the
# module importable on the 3.10/3.11 interpreters `scripts/lib/deps.sh` accepts with a note.
_sumprod = getattr(math, "sumprod", None) or (lambda a, b: sum(x * y for x, y in zip(a, b)))


def _cosine(left: Sequence[float] | None, right: Sequence[float] | None) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    left_norm = math.sqrt(_sumprod(left, left))
    right_norm = math.sqrt(_sumprod(right, right))
    if not left_norm or not right_norm:
        return 0.0
    similarity = _sumprod(left, right) / (left_norm * right_norm)
    return max(0.0, min(1.0, similarity)) if math.isfinite(similarity) else 0.0


def retrieve(inputs: PipelineInputs) -> PrecisionOutcome:
    """Run the pipeline. Returns results plus the trace that explains them."""
    config = inputs.config or get_precision_config()
    trace = PrecisionTrace(original_query=inputs.query)
    outcome = PrecisionOutcome(trace=trace)
    started = time.perf_counter()

    if not inputs.chunks:
        trace.record("input", 0, 0, reason="resource has no indexed chunks")
        return _finish(outcome, started, config)

    # ---- stage 1: normalization ------------------------------------------------------
    stage = time.perf_counter()
    normalized = normalize_query(inputs.query) if config.normalization_enabled else (inputs.query or "").strip()
    trace.normalized_query = normalized
    base_terms = inputs.tokenize(normalized)
    trace.timings_ms["normalize"] = round((time.perf_counter() - stage) * 1000, 3)
    if not base_terms:
        # Nothing to search for. Returning empty lets the caller's existing sufficiency
        # gate produce the refusal it already produces for every other mode, rather than
        # inventing a second wording for the same situation.
        trace.record("normalize", 0, 0, reason="query has no searchable terms")
        return _finish(outcome, started, config)

    # ---- index (cached per resource) --------------------------------------------------
    stage = time.perf_counter()
    corpus = index_module.get_index(
        inputs.resource_id,
        inputs.chunks,
        terms_of=inputs.terms_of,
        extract_entities=inputs.extract_entities if config.entity_extraction_enabled else None,
        parent_group_size=config.parent_group_size,
        tokenize=inputs.tokenize,
    )
    trace.timings_ms["index"] = round((time.perf_counter() - stage) * 1000, 3)

    # ---- stage 2: deterministic expansion --------------------------------------------
    stage = time.perf_counter()
    corrections = correct_spelling(base_terms, corpus.vocabulary) if config.normalization_enabled else {}
    expanded = expansion.expand_query(
        inputs.query,
        normalized,
        base_terms,
        config=config,
        dictionary=inputs.dictionary if inputs.dictionary is not None else expansion.load_dictionary(),
        entity_aliases=corpus.entity_aliases,
        corrections=corrections,
        tokenize=inputs.tokenize,
        vocabulary=corpus.vocabulary,
    )
    trace.timings_ms["expand"] = round((time.perf_counter() - stage) * 1000, 3)

    # ---- stage 3: entity / keyword extraction ----------------------------------------
    query_entities: list[str] = []
    if config.entity_extraction_enabled and inputs.extract_entities is not None:
        try:
            query_entities = [entity for entity in inputs.extract_entities(inputs.query)[:8]]
        except Exception:  # noqa: BLE001 - a nicety, never a failure
            query_entities = []
    trace.entities = query_entities
    entity_terms = {
        term
        for entity in query_entities
        for term in inputs.tokenize(entity)
    }

    # ---- stage 4: metadata filtering --------------------------------------------------
    stage = time.perf_counter()
    filters = (
        metadata_module.infer_filters(normalized, base_terms, corpus.categories)
        if config.metadata_filtering_enabled
        else {}
    )
    allowed_ids: set[str] | None = None
    applied_filters: dict = {}
    if filters:
        universe = [Candidate(chunk=chunk) for chunk in corpus.chunks]
        survivors, applied_filters = metadata_module.apply_filters(
            universe, corpus.metadata, filters, config.metadata_filter_min_survivors
        )
        if any(value is not None for value in applied_filters.values()):
            allowed_ids = {candidate.chunk_id for candidate in survivors}
    trace.metadata_filters = {"inferred": filters, "applied": applied_filters}
    trace.record(
        "metadata_filter",
        len(corpus.chunks),
        len(allowed_ids) if allowed_ids is not None else len(corpus.chunks),
        **{"filters": applied_filters},
    )
    trace.timings_ms["metadata_filter"] = round((time.perf_counter() - stage) * 1000, 3)

    # ---- stage 5a: BM25 ---------------------------------------------------------------
    # BM25 and the dense pass are INDEPENDENT of one another and both read only immutable
    # state, so a host that wants them concurrent can run them concurrently. They are
    # sequential here because the dense side has already been computed by the caller (one
    # embedding call, made before the mode dispatched) — there is nothing left to overlap,
    # and a thread per question to overlap nothing is pure cost.
    stage = time.perf_counter()
    bm25_scores: dict[str, float] = {}
    if config.bm25_enabled:
        typed_term_set = set(expanded.original_terms)
        bm25_scores = _bm25_pass(expanded.terms, typed_term_set, corpus, allowed_ids, config)

        # Pseudo-relevance feedback: a second pass whose query has absorbed the most
        # distinctive terms of the first pass's best passages. Deterministic, corpus-driven,
        # and bounded — no model is asked what the query "really means".
        if config.prf_enabled and config.prf_terms > 0 and bm25_scores:
            top_ids = sorted(bm25_scores, key=lambda cid: (-bm25_scores[cid], cid))[: config.prf_documents]
            feedback = bm25.feedback_terms(
                [corpus.terms.get(cid, {}) for cid in top_ids],
                document_count=corpus.document_count,
                document_frequencies=corpus.document_frequencies,
                exclude=set(expanded.terms),
                limit=config.prf_terms,
            )
            if feedback:
                expansion.apply_pseudo_relevance_feedback(expanded, feedback, config=config)
                # PRF terms are expansions too, so they land under the same ceiling. A
                # feedback term is a corpus-derived guess about the question, and a guess
                # that could outrank the question is exactly what the cap exists to stop.
                bm25_scores = _bm25_pass(expanded.terms, typed_term_set, corpus, allowed_ids, config)
    trace.expanded_terms = list(expanded.added_terms)
    trace.timings_ms["bm25"] = round((time.perf_counter() - stage) * 1000, 3)
    trace.record("bm25", len(corpus.chunks), len(bm25_scores))

    # ---- stage 5b: dense --------------------------------------------------------------
    stage = time.perf_counter()
    dense_scores: dict[str, float] = {}
    if config.dense_enabled and inputs.dense_candidates:
        for chunk_id, similarity in inputs.dense_candidates:
            if similarity <= 0 or chunk_id not in corpus.by_id:
                continue
            if allowed_ids is not None and chunk_id not in allowed_ids:
                continue
            dense_scores[chunk_id] = max(dense_scores.get(chunk_id, 0.0), float(similarity))
    trace.timings_ms["dense"] = round((time.perf_counter() - stage) * 1000, 3)
    trace.record("dense", len(inputs.dense_candidates or ()), len(dense_scores))

    if not bm25_scores and not dense_scores:
        trace.record("candidates", 0, 0, reason="no chunk contained any query term")
        return _finish(outcome, started, config)

    # ---- candidate pool ---------------------------------------------------------------
    stage = time.perf_counter()
    typed_terms = set(expanded.original_terms)
    pool: dict[str, Candidate] = {}
    for chunk_id in set(bm25_scores) | set(dense_scores):
        chunk = corpus.by_id.get(chunk_id)
        if chunk is None:
            continue
        candidate = Candidate(chunk=chunk)
        candidate.bm25_score = bm25_scores.get(chunk_id, 0.0)
        candidate.dense_score = dense_scores.get(chunk_id, 0.0)
        if candidate.bm25_score > 0:
            candidate.sources.add("bm25")
        if candidate.dense_score > 0:
            candidate.sources.add("dense")
        chunk_terms = corpus.terms.get(chunk_id, {})
        candidate.keyword_score = _keyword_score(typed_terms, entity_terms, chunk_terms)
        candidate.metadata_score = metadata_module.metadata_score(
            corpus.metadata.get(chunk_id), filters, typed_terms
        )
        candidate.negative_penalty = _negative_penalty(chunk_terms, config)
        pool[chunk_id] = candidate

    normalized_bm25 = _normalize_scores({cid: c.bm25_score for cid, c in pool.items()})
    normalized_dense = _normalize_scores({cid: c.dense_score for cid, c in pool.items()})
    for chunk_id, candidate in pool.items():
        combined = (
            config.dense_weight * normalized_dense.get(chunk_id, 0.0)
            + config.bm25_weight * normalized_bm25.get(chunk_id, 0.0)
            + config.metadata_weight * candidate.metadata_score
            + config.keyword_weight * candidate.keyword_score
        )
        # The penalty is MULTIPLICATIVE, never subtractive. Subtracting a constant makes a
        # weak match negative and a strong one merely weaker, which inverts the ordering it
        # was meant to adjust; scaling preserves it and cannot push a score below zero.
        candidate.retrieval_score = combined * (1.0 - candidate.negative_penalty)
        candidate.final_score = candidate.retrieval_score

    ranked = sorted(pool.values(), key=lambda c: (-c.final_score, c.chunk_id))[: config.candidate_k]
    trace.timings_ms["fuse"] = round((time.perf_counter() - stage) * 1000, 3)
    trace.record("candidate_pool", len(pool), len(ranked), candidate_k=config.candidate_k)

    # ---- stage 6: cross-encoder rerank -------------------------------------------------
    stage = time.perf_counter()
    rerank_scores, backend, rerank_error = rerank.rerank(
        expanded, ranked, corpus, tokenize=inputs.tokenize, config=config
    )
    trace.reranker_backend = backend
    trace.reranker_error = rerank_error
    if rerank_scores:
        normalized_retrieval = _normalize_scores({c.chunk_id: c.retrieval_score for c in ranked})
        weight = config.reranker_weight
        for candidate in ranked:
            score = rerank_scores.get(candidate.chunk_id)
            if score is None:
                continue
            candidate.rerank_score = score
            candidate.final_score = (
                weight * score + (1.0 - weight) * normalized_retrieval.get(candidate.chunk_id, 0.0)
            ) * (1.0 - candidate.negative_penalty)
        ranked.sort(key=lambda c: (-c.final_score, c.chunk_id))
    # Narrowed whether or not a reranker ran. `reranker_top_k` reads as "how many survive
    # the reranker", but what it actually controls is how many candidates reach dedup and
    # MMR — and tying it to `reranker_enabled` meant switching the reranker off also widened
    # the pool those two stages see, from 20 to 100. An ablation that changes two things is
    # not an ablation, and this harness exists to attribute a metric change to one change.
    reranked = ranked[: config.reranker_top_k]
    trace.timings_ms["rerank"] = round((time.perf_counter() - stage) * 1000, 3)
    trace.record("rerank", len(ranked), len(reranked), backend=backend, error=rerank_error)

    # ---- stage 7: deduplication --------------------------------------------------------
    stage = time.perf_counter()
    term_sets = {c.chunk_id: set(corpus.terms.get(c.chunk_id, {})) for c in reranked}
    if config.dedup_enabled:
        deduped, dedup_decisions = diversity.deduplicate(reranked, term_sets, config.dedup_threshold)
    else:
        deduped, dedup_decisions = list(reranked), []
    trace.timings_ms["dedup"] = round((time.perf_counter() - stage) * 1000, 3)
    trace.record("dedup", len(reranked), len(deduped), dropped=dedup_decisions[: config.trace_max_candidates_logged])

    # ---- stage 8: MMR ------------------------------------------------------------------
    stage = time.perf_counter()
    if config.mmr_enabled and len(deduped) > 1:
        similarity = _similarity_function(term_sets, inputs.vector_of)
        selected, mmr_decisions = diversity.maximal_marginal_relevance(
            deduped, similarity, lambda_=config.mmr_lambda, limit=config.final_k
        )
    else:
        selected, mmr_decisions = deduped[: config.final_k], []
    trace.timings_ms["mmr"] = round((time.perf_counter() - stage) * 1000, 3)
    trace.record("mmr", len(deduped), len(selected), selection=mmr_decisions[: config.trace_max_candidates_logged])

    # ---- stage 9: parent recovery ------------------------------------------------------
    stage = time.perf_counter()
    results: list[PrecisionResult] = []
    for candidate in selected:
        meta = corpus.metadata.get(candidate.chunk_id)
        parent_context: str | None = None
        sibling_ids: tuple[str, ...] = ()
        if config.parent_child_enabled and meta is not None:
            parent_context, sibling_ids = parents.recover_parent(
                candidate.chunk,
                corpus.families.get(meta.parent_chunk_id or "", []),
                corpus.by_id,
                max_chars=config.parent_max_chars,
            )
        results.append(
            PrecisionResult(
                chunk=candidate.chunk,
                score=candidate.final_score,
                metadata=meta,
                retrieval_score=candidate.retrieval_score,
                bm25_score=candidate.bm25_score,
                dense_score=candidate.dense_score,
                rerank_score=candidate.rerank_score,
                retrieval_method="+".join(sorted(candidate.sources)) or "bm25",
                parent_context=parent_context,
                parent_chunk_ids=sibling_ids,
            )
        )
    trace.timings_ms["parents"] = round((time.perf_counter() - stage) * 1000, 3)
    trace.record("parent_recovery", len(selected), len(results))

    outcome.results = results
    trace.final_chunk_ids = [result.chunk.id for result in results]
    return _finish(outcome, started, config)


def _finish(outcome: PrecisionOutcome, started: float, config: PrecisionConfig) -> PrecisionOutcome:
    """Stamp the total time and emit the trace, on EVERY exit path.

    The three empty outcomes — no chunks, no searchable terms, no chunk matching a term —
    are the three a person is most likely to be reading the log to explain, and two of them
    used to return before the logging line. A question that came back empty for the one
    reason that logged looked like the only reason a question can come back empty.
    """
    outcome.trace.timings_ms["total"] = round((time.perf_counter() - started) * 1000, 3)
    _log_trace(outcome.trace, config)
    return outcome


def _bm25_pass(
    query_weights: dict[str, float],
    typed_terms: set[str],
    corpus,
    allowed_ids: set[str] | None,
    config: PrecisionConfig,
) -> dict[str, float]:
    """BM25 over the chunks containing at least one query term, with expansion bounded.

    The postings list keeps this sub-linear in the corpus for a typical question: a chunk
    sharing no term with the query scores exactly 0.0 by construction, so visiting it to
    discover that is work with a known answer.

    THE CAP IS THE PART WORTH READING. Each chunk's score arrives as two numbers — what the
    words the user typed earned, and what the expansions earned — and the second is clamped
    to `expansion_max_fraction` of the BEST TYPED SCORE THIS QUERY PRODUCED. Two properties
    follow, and both were measured rather than assumed (see `bm25.split_score`):

      * a passage matching the literal question always outranks one matching only guesses
        about it, because the guesses' total is strictly below the best literal total;
      * a passage found ONLY through an expansion — the synonym case, the entire reason the
        stage exists — still surfaces, because the cap is relative to the query rather than
        to that chunk's own (zero) typed score.

    The cap is a fraction rather than a constant because BM25 scores are corpus- and
    query-dependent: any absolute number would be right for one document and wrong for the
    next. `_graph_rag` in `rag/service.py` bounds its entity boost the same way, for the same
    reason and after the same kind of measurement.
    """
    split: dict[str, tuple[float, float]] = {}
    best_typed = 0.0
    for chunk_id in corpus.candidates_for(list(query_weights)):
        if allowed_ids is not None and chunk_id not in allowed_ids:
            continue
        typed, expanded = bm25.split_score(
            query_weights,
            typed_terms,
            corpus.terms.get(chunk_id, {}),
            corpus.lengths.get(chunk_id, 0),
            document_count=corpus.document_count,
            document_frequencies=corpus.document_frequencies,
            average_length=corpus.average_length,
            k1=config.bm25_k1,
            b=config.bm25_b,
        )
        if typed > 0 or expanded > 0:
            split[chunk_id] = (typed, expanded)
            best_typed = max(best_typed, typed)

    ceiling = config.expansion_max_fraction * best_typed
    scores: dict[str, float] = {}
    for chunk_id, (typed, expanded) in split.items():
        total = typed + (min(expanded, ceiling) if ceiling > 0 else expanded)
        if total > 0:
            scores[chunk_id] = total
    return scores


def _keyword_score(typed_terms: set[str], entity_terms: set[str], chunk_terms: dict[str, int]) -> float:
    """Plain presence of the user's own words, and of the entities in their question.

    Deliberately unweighted by frequency, so it says something BM25 does not: BM25 answers
    "how strongly does this passage discuss these terms", this answers "does this passage
    contain the words that were actually typed". A passage mentioning every one of them once
    is often the specification line the user wanted; BM25 alone ranks it below a passage that
    repeats one of them.
    """
    if not typed_terms:
        return 0.0
    present = sum(1 for term in typed_terms if term in chunk_terms)
    score = present / len(typed_terms)
    if entity_terms:
        entity_present = sum(1 for term in entity_terms if term in chunk_terms)
        score = min(1.0, score + 0.25 * (entity_present / len(entity_terms)))
    return score


def _negative_penalty(chunk_terms: dict[str, int], config: PrecisionConfig) -> float:
    """A configurable penalty for candidates matching a deployment's negative terms.

    Empty by default. A shipped list of "bad" words would be an unauditable opinion about
    someone else's documents applied silently to every query they ask.
    """
    if not config.negative_signals_enabled or not config.negative_terms:
        return 0.0
    hits = sum(1 for term in config.negative_terms if term in chunk_terms)
    if not hits:
        return 0.0
    share = hits / len(config.negative_terms)
    return min(0.95, config.negative_penalty * share)


def _similarity_function(term_sets: dict[str, set[str]], vector_of):
    """Redundancy measure for MMR: dense cosine when vectors exist, term overlap otherwise.

    Falling back rather than requiring vectors matters — the default knowledge base in this
    application has no embeddings at all, and an MMR that silently measured every pair as
    0.0 similar would select the top-K by relevance alone while reporting that it diversified.
    """
    def similarity(left_id: str, right_id: str) -> float:
        if vector_of is not None:
            left_vector = vector_of(left_id)
            right_vector = vector_of(right_id)
            if left_vector and right_vector:
                return _cosine(left_vector, right_vector)
        return diversity.jaccard(term_sets.get(left_id, set()), term_sets.get(right_id, set()))

    return similarity


def _log_trace(trace: PrecisionTrace, config: PrecisionConfig) -> None:
    """One structured log line per question.

    Ids, counts, scores and the QUERY's own terms only. No passage text — the log is the
    thing that gets pasted into a bug report, and a chunk of somebody's contract in it is a
    disclosure, not a diagnostic.
    """
    if not config.trace_enabled:
        return
    logger.info(
        "precision retrieval",
        extra={
            "extra_fields": {
                "normalized_query": trace.normalized_query,
                "expanded_terms": trace.expanded_terms[:20],
                "entities": trace.entities[:10],
                "filters": trace.metadata_filters,
                "reranker": trace.reranker_backend,
                "reranker_error": trace.reranker_error,
                "stages": [
                    {"stage": s.stage, "in": s.count_in, "out": s.count_out} for s in trace.stages
                ],
                "final_chunk_ids": trace.final_chunk_ids,
                "timings_ms": trace.timings_ms,
            }
        },
    )
