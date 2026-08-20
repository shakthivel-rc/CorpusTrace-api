"""Okapi BM25 over the term maps ingestion already wrote — the lexical half of stage 5.

This is a genuine BM25, not the existing `_score_chunk` renamed. The two differ in the ways
that matter for precision:

  * `_score_chunk` weights a term by its raw frequency product, so a term appearing in
    every chunk of the corpus contributes as much as one appearing in three. BM25 weights
    by INVERSE DOCUMENT FREQUENCY, so the rare term — the one that actually discriminates —
    dominates.
  * `_score_chunk` saturates nothing: a chunk repeating a query term twenty times scores
    twenty times a chunk mentioning it once. BM25's `k1` saturation says the twentieth
    mention is worth almost nothing, which is the empirically correct answer.
  * length normalisation is `1/sqrt(len)` there and BM25's tunable `b` here.

Both are correct for what they were built for. The existing modes keep `_score_chunk`
untouched; this mode uses BM25 because its whole purpose is ranking a large candidate pool
precisely rather than picking five passages cheaply.

No dependency: `rank_bm25` would be one more pin for forty lines of arithmetic over data
this repository already has in memory.
"""
from __future__ import annotations

import math


def idf(document_count: int, document_frequency: int) -> float:
    """Robertson/Sparck-Jones IDF with the +0.5 smoothing, floored at zero.

    The floor matters because of what sits above it, not because this form often goes
    negative: `log(1 + x)` is negative only once `df` exceeds roughly `N + 0.5`, which cannot
    happen. What the floor guarantees is that IDF is never negative for any input, including
    a `document_frequencies` map that disagrees with `document_count` — a stale cached index,
    or a caller passing statistics from a different corpus. A negative IDF would PENALISE a
    passage for containing one of the user's words, and it would do so silently.
    """
    numerator = document_count - document_frequency + 0.5
    denominator = document_frequency + 0.5
    return max(0.0, math.log(1.0 + (numerator / denominator)))


def split_score(
    query_weights: dict[str, float],
    typed_terms: set[str],
    chunk_terms: dict[str, int],
    chunk_length: int,
    *,
    document_count: int,
    document_frequencies: dict[str, int],
    average_length: float,
    k1: float,
    b: float,
) -> tuple[float, float]:
    """BM25 for one (query, chunk) pair, split into `(typed, expanded)` contributions.

    Two numbers rather than one because per-term down-weighting turned out NOT to be enough
    to keep expansion additive, and the caller needs the two halves to bound the second
    against the first. Measured on the bundled 46-chunk benchmark: at twelve expansion terms
    weighted 0.45 each, the expanded half carries up to ~5.4 term-equivalents of mass against
    a typical typed query's 3-4, so a passage matching many *guesses* about the question
    outranked one matching the question. Recall@1 fell from 0.8594 to 0.8125 — expansion, the
    stage meant to help, was the only stage in the pipeline making things worse.

    `pipeline._bm25_pass` caps the expanded half at a fraction of the BEST TYPED SCORE FOR
    THIS QUERY. Bounding it per chunk instead would have been wrong in the other direction:
    a chunk that matches only a synonym has a typed score of zero, and a per-chunk cap would
    multiply it away — killing precisely the case expansion exists for.
    """
    if not query_weights or not chunk_terms:
        return 0.0, 0.0
    if average_length <= 0:
        average_length = 1.0
    denominator_base = k1 * (1.0 - b + b * (chunk_length / average_length))

    typed_total = 0.0
    expanded_total = 0.0
    for term, weight in query_weights.items():
        frequency = chunk_terms.get(term)
        if not frequency:
            continue
        term_idf = idf(document_count, document_frequencies.get(term, 0))
        if term_idf <= 0.0:
            continue
        contribution = weight * term_idf * (frequency * (k1 + 1.0)) / (frequency + denominator_base)
        if term in typed_terms:
            typed_total += contribution
        else:
            expanded_total += contribution
    return typed_total, expanded_total


def score(
    query_weights: dict[str, float],
    chunk_terms: dict[str, int],
    chunk_length: int,
    *,
    document_count: int,
    document_frequencies: dict[str, int],
    average_length: float,
    k1: float,
    b: float,
) -> float:
    """Plain Okapi BM25 for one (query, chunk) pair, with per-term query weights.

    The undivided form. `split_score` is what the pipeline uses; this stays because it is
    the definition the tests check the arithmetic against, and because a caller with no
    expansion has nothing to split.
    """
    typed, expanded = split_score(
        query_weights,
        set(query_weights),
        chunk_terms,
        chunk_length,
        document_count=document_count,
        document_frequencies=document_frequencies,
        average_length=average_length,
        k1=k1,
        b=b,
    )
    return typed + expanded


def feedback_terms(
    chunk_term_maps: list[dict[str, int]],
    *,
    document_count: int,
    document_frequencies: dict[str, int],
    exclude: set[str],
    limit: int,
) -> list[tuple[str, float]]:
    """The most distinctive terms of a set of passages, for pseudo-relevance feedback.

    Distinctive means frequent HERE and rare in the corpus — the same IDF weighting the
    ranking uses, so a feedback term is one that would itself have discriminated. Ordered
    deterministically (score, then the term itself) so two runs over the same corpus
    produce the same expansion; ties broken by anything else would make the benchmark
    unreproducible.
    """
    if limit <= 0 or not chunk_term_maps:
        return []
    scores: dict[str, float] = {}
    for terms in chunk_term_maps:
        total = sum(terms.values()) or 1
        for term, frequency in terms.items():
            if term in exclude or len(term) < 3:
                continue
            term_idf = idf(document_count, document_frequencies.get(term, 0))
            if term_idf <= 0.0:
                continue
            scores[term] = scores.get(term, 0.0) + (frequency / total) * term_idf
    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return ranked[:limit]
