"""Deduplication and MMR — stages 7 and 8 of the high-precision pipeline.

Both stages answer the same question from different ends: the final Top-K is a small,
expensive budget, and spending it on the same sentence five times is the most common way a
technically-correct retriever produces a useless result set.

Deduplication removes near-copies. It has to exist here specifically because the chunker
OVERLAPS by design — 180 characters by default (`rag/chunking.py`) — so adjacent chunks
genuinely share text, and a passage sitting on a chunk boundary reliably matches twice.

MMR then trades a little relevance for complementary evidence. It runs AFTER dedup, not
instead of it: MMR's penalty is proportional, so two 97%-identical chunks both survive it
when they are far and away the best matches, which is exactly the case dedup exists for.
"""
from __future__ import annotations

from typing import Callable


def jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    intersection = len(left & right)
    if not intersection:
        return 0.0
    return intersection / len(left | right)


def containment(left: set[str], right: set[str]) -> float:
    """Share of the SMALLER term set contained in the larger.

    Jaccard alone misses the chunker's own failure mode: a 400-term chunk that wholly
    contains a 120-term one scores 0.3 by Jaccard and is 100% redundant. Containment sees
    that; Jaccard sees two chunks of similar size saying the same thing. Dedup takes the
    max of the two, because either is a reason not to spend a citation slot twice.
    """
    if not left or not right:
        return 0.0
    intersection = len(left & right)
    return intersection / min(len(left), len(right))


def deduplicate(
    candidates: list,
    term_sets: dict[str, set[str]],
    threshold: float,
) -> tuple[list, list[dict]]:
    """Drop candidates too similar to a better-ranked one already kept.

    Input order is the ranking, so the survivor of any pair is always the higher-ranked
    one. Returns `(kept, decisions)`; the decisions carry the pair and the similarity, so
    the trace can answer "why is chunk X missing" without re-running anything.
    """
    kept: list = []
    decisions: list[dict] = []
    if threshold <= 0:
        return list(candidates), decisions

    for candidate in candidates:
        terms = term_sets.get(candidate.chunk_id, set())
        duplicate_of = None
        similarity = 0.0
        for survivor in kept:
            other = term_sets.get(survivor.chunk_id, set())
            score = max(jaccard(terms, other), containment(terms, other))
            if score >= threshold:
                duplicate_of = survivor.chunk_id
                similarity = score
                break
        if duplicate_of is None:
            kept.append(candidate)
        else:
            decisions.append(
                {"dropped": candidate.chunk_id, "duplicate_of": duplicate_of, "similarity": round(similarity, 4)}
            )
    return kept, decisions


def maximal_marginal_relevance(
    candidates: list,
    similarity: Callable[[str, str], float],
    *,
    lambda_: float,
    limit: int,
) -> tuple[list, list[dict]]:
    """Classic MMR: argmax λ·rel(c) − (1−λ)·max_{s∈selected} sim(c, s).

    Relevance is taken from `final_score` and MIN-MAX NORMALISED across the pool first, so
    the two halves of the subtraction are on the same 0..1 scale. Without that the trade-off
    is between an unbounded relevance and a bounded similarity, and λ stops meaning anything
    — it becomes "ignore diversity" for any corpus whose scores happen to run large.
    """
    if limit <= 0 or not candidates:
        return [], []

    scores = [candidate.final_score for candidate in candidates]
    low, high = min(scores), max(scores)
    span = high - low
    relevance = {
        candidate.chunk_id: (0.5 if span <= 0 else (candidate.final_score - low) / span)
        for candidate in candidates
    }

    remaining = list(candidates)
    selected: list = []
    decisions: list[dict] = []

    # The first pick is pure relevance: with nothing selected there is nothing to be diverse
    # from, and starting anywhere else would discard the best passage on principle.
    #
    # Chosen by MAXIMUM SCORE, not by taking the head of the list. The pipeline happens to
    # pass a sorted list, so the two agree there — but a function whose documented behaviour
    # depends on an undocumented precondition is a trap for the next caller, and this one is
    # also called directly by tests and by the benchmark.
    first_index = max(range(len(remaining)), key=lambda i: remaining[i].final_score)
    first = remaining.pop(first_index)
    selected.append(first)
    decisions.append({"selected": first.chunk_id, "mmr": round(relevance[first.chunk_id], 4), "redundancy": 0.0})

    if lambda_ >= 1.0:
        # Pure relevance: no redundancy term to compute. Still emitted one decision per
        # selected item, because the trace reader must not have to know which lambda was in
        # force to interpret a missing row.
        for candidate in sorted(remaining, key=lambda c: (-c.final_score, c.chunk_id)):
            if len(selected) >= limit:
                break
            selected.append(candidate)
            decisions.append(
                {"selected": candidate.chunk_id, "mmr": round(relevance[candidate.chunk_id], 4), "redundancy": 0.0}
            )
        return selected, decisions

    while remaining and len(selected) < limit:
        best_index = 0
        best_value = None
        best_redundancy = 0.0
        for position, candidate in enumerate(remaining):
            redundancy = max(
                (similarity(candidate.chunk_id, chosen.chunk_id) for chosen in selected),
                default=0.0,
            )
            value = lambda_ * relevance[candidate.chunk_id] - (1.0 - lambda_) * redundancy
            if best_value is None or value > best_value:
                best_value = value
                best_index = position
                best_redundancy = redundancy
        chosen = remaining.pop(best_index)
        selected.append(chosen)
        decisions.append(
            {
                "selected": chosen.chunk_id,
                "mmr": round(best_value or 0.0, 4),
                "redundancy": round(best_redundancy, 4),
            }
        )

    return selected, decisions
