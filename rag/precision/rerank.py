"""Cross-encoder re-ranking — stage 6 of the high-precision pipeline. No LLM.

A cross-encoder is defined by WHERE the query and the document meet: a bi-encoder embeds
each independently and compares vectors, a cross-encoder scores the PAIR jointly, so its
output cannot be precomputed and stored per passage. That architectural property — not the
presence of a transformer — is what makes reranking work, and it is what both backends here
implement.

`lexical` (the default) is a pure-Python interaction scorer. It reads features that only
exist for a (query, passage) pair — where the query's terms land in the passage, how close
together, in what order, how rare they are in this corpus — and none of them can be derived
from the passage alone. It needs no model download, no GPU, no new dependency, and it is
deterministic, which is what lets `benchmark.py` attribute a metric change to a code change.

`http` posts to a local reranking service — text-embeddings-inference, Infinity, Xinference
and the BGE-reranker servers all speak one of the two shapes parsed below — over the stdlib
`urllib` seam this codebase uses for every outbound call. Off by default and configured by
endpoint, so a deployment can put `BAAI/bge-reranker-base` behind it and measure the
difference with the same benchmark. These are rerankers, not generators: they emit one
relevance score per pair and cannot produce text.

EVERY FAILURE HERE IS INERT. A reranker that times out, 500s or returns nonsense costs the
ordering it would have improved, never the answer — the retrieval ranking is already a
complete result. The reason is recorded in the trace so a silently-degraded deployment is
visible rather than merely slow.
"""
from __future__ import annotations

import json
import logging
import math
import re
import urllib.error
import urllib.request

from rag.precision import bm25
from rag.precision.config import RERANKER_HTTP, RERANKER_LEXICAL, RERANKER_NONE


logger = logging.getLogger("corpustrace.rag.precision")

_WS = re.compile(r"\s+")

# The scale of the proximity decay, in tokens — NOT a cut-off. A window as tight as the terms
# it holds scores 1.0 and every wider one decays smoothly from there, so a slightly-too-wide
# match still outranks a far worse one. Nothing is ever rejected for exceeding this.
PROXIMITY_WINDOW = 30
# Weights of the interaction features. They sum to 1.0, so a rerank score is directly
# comparable to a normalised retrieval score and `reranker_weight` blends two numbers that
# mean the same kind of thing.
FEATURE_WEIGHTS = {
    "coverage": 0.34,   # how much of the question's *information* the passage contains
    "proximity": 0.22,  # how tightly those terms sit together
    "order": 0.16,      # whether the question's word order survives in the passage
    "exact": 0.14,      # the whole normalized question appears verbatim
    "heading": 0.08,    # the passage's own heading names the question's words
    "position": 0.06,   # the match is near the top of the passage, not trailing off it
}


def _normalized_text(text: str) -> str:
    return _WS.sub(" ", (text or "").casefold()).strip()


def _coverage(query_weights: dict[str, float], tokens_present: set[str], index) -> float:
    """IDF-weighted share of the query's information that this passage carries.

    Weighted by IDF, not by term count: a passage containing four of five query terms has
    covered almost nothing if the one it missed was the only rare word. That is precisely
    the failure a plain overlap ratio cannot see, and it is the common one.
    """
    total = 0.0
    matched = 0.0
    for term, weight in query_weights.items():
        term_idf = bm25.idf(index.document_count, index.document_frequencies.get(term, 0))
        contribution = weight * max(term_idf, 0.05)
        total += contribution
        if term in tokens_present:
            matched += contribution
    return (matched / total) if total > 0 else 0.0


def _proximity_and_position(query_terms: set[str], tokens: list[str]) -> tuple[float, float]:
    """`(proximity, position)` from where the query's terms actually land.

    Proximity is computed from the smallest window containing the largest number of
    DISTINCT query terms — the classic minimum-covering-window, restricted to the terms
    that are present so a missing term cannot make a perfect passage score zero. It is the
    feature that separates "this passage discusses the connection pool timing out" from
    "this passage mentions connections, and forty lines later mentions a timeout".
    """
    positions: dict[str, list[int]] = {}
    for offset, token in enumerate(tokens):
        if token in query_terms:
            positions.setdefault(token, []).append(offset)
    if not positions:
        return 0.0, 0.0

    present = list(positions)
    if len(present) == 1:
        first = positions[present[0]][0]
        return 0.35, _position_score(first, len(tokens))

    # Sweep the merged, sorted occurrence stream and keep the tightest window that holds
    # every present term. O(occurrences), not O(occurrences^2).
    stream = sorted(
        (offset, token) for token, offsets in positions.items() for offset in offsets
    )
    needed = len(present)
    counts: dict[str, int] = {}
    distinct = 0
    left = 0
    best_window = None
    for right, (offset, token) in enumerate(stream):
        counts[token] = counts.get(token, 0) + 1
        if counts[token] == 1:
            distinct += 1
        while distinct == needed:
            width = offset - stream[left][0]
            if best_window is None or width < best_window:
                best_window = width
            left_token = stream[left][1]
            counts[left_token] -= 1
            if counts[left_token] == 0:
                distinct -= 1
            left += 1
    if best_window is None:  # pragma: no cover - the sweep always closes at least one window
        return 0.35, _position_score(stream[0][0], len(tokens))

    # A window as wide as the terms it holds is perfect; wider windows decay smoothly on a
    # scale set by PROXIMITY_WINDOW rather than cutting off, so a slightly-too-wide match is
    # still ranked above one that is far worse.
    slack = max(best_window - (needed - 1), 0)
    proximity = 1.0 / (1.0 + math.log1p(slack / max(PROXIMITY_WINDOW / 10.0, 1.0)))
    return min(proximity, 1.0), _position_score(stream[0][0], len(tokens))


def _position_score(first_match: int, token_count: int) -> float:
    if token_count <= 0:
        return 0.0
    return max(0.0, 1.0 - (first_match / token_count))


def _order(phrases: list[tuple[str, str]], tokens: list[str]) -> float:
    """The share of the question's adjacent word pairs that survive in the passage.

    "database connection timeout" and "timeout connection database" are the same bag of
    words and different questions. Nothing upstream of here can tell them apart.
    """
    if not phrases:
        return 0.0
    positions: dict[str, list[int]] = {}
    for offset, token in enumerate(tokens):
        positions.setdefault(token, []).append(offset)
    hits = 0
    for left, right in phrases:
        left_positions = positions.get(left)
        right_positions = positions.get(right)
        if not left_positions or not right_positions:
            continue
        # Adjacent, or separated by at most two tokens — "connection pool timeout" should
        # still credit the pair ("connection", "timeout").
        if any(0 < rp - lp <= 3 for lp in left_positions for rp in right_positions):
            hits += 1
    return hits / len(phrases)


def _heading_match(heading: str | None, query_terms: set[str]) -> float:
    if not heading or not query_terms:
        return 0.0
    heading_terms = {word.lower() for word in re.findall(r"[A-Za-z0-9]+", heading)}
    if not heading_terms:
        return 0.0
    return min(1.0, len(heading_terms.intersection(query_terms)) / max(len(query_terms), 1) * 2.0)


def lexical_cross_encode(
    expanded,
    candidates: list,
    index,
    *,
    tokenize,
    config=None,
) -> dict[str, float]:
    """Score every candidate against the query jointly. Returns `{chunk_id: 0..1}`.

    `config` is accepted and unused: it keeps this callable interchangeable with
    `http_cross_encode` at the `rerank()` dispatch, and it is where a knob for this scorer
    would land. Defaulted so a caller that has no configuration — a test, a notebook — can
    score a pair without constructing one.
    """
    typed_terms = set(expanded.original_terms)
    normalized_query = expanded.normalized
    scores: dict[str, float] = {}

    for candidate in candidates:
        chunk = candidate.chunk
        content = chunk.content or ""
        tokens = tokenize(content)
        tokens_present = set(tokens)

        # Coverage reads the weighted map MINUS the pseudo-relevance-feedback terms, so a
        # passage reached only through a synonym is not scored zero (typed terms dominate it
        # anyway, at 1.0 against an expansion's 0.45) while the corpus's own guesses about
        # the topic get no vote on which candidate wins. See `ExpandedQuery.rerank_terms`.
        coverage = _coverage(expanded.rerank_terms, tokens_present, index)
        # Proximity and order read the TYPED terms only, and this is the single most
        # important line in the file. Retrieval and reranking have opposite jobs: expansion
        # exists to widen the candidate pool (recall), the cross-encoder exists to decide
        # which candidate is actually the answer (precision). Measuring "how tightly do the
        # query's words sit together" over twelve words the user never typed measures the
        # tightness of a guess. It was worth 4.7 points of Recall@1 on the bundled 46-chunk
        # benchmark (0.8594 -> 0.8125): enabling expansion made the pipeline WORSE, and it
        # was not the BM25 mass — capping that changed nothing — it was here.
        proximity, position = _proximity_and_position(typed_terms, tokens)
        order = _order(expanded.phrases, tokens)
        exact = 1.0 if normalized_query and len(normalized_query) > 6 and normalized_query in _normalized_text(content) else 0.0
        meta = index.metadata.get(chunk.id)
        heading = _heading_match(meta.heading if meta else None, typed_terms)

        raw = (
            FEATURE_WEIGHTS["coverage"] * coverage
            + FEATURE_WEIGHTS["proximity"] * proximity
            + FEATURE_WEIGHTS["order"] * order
            + FEATURE_WEIGHTS["exact"] * exact
            + FEATURE_WEIGHTS["heading"] * heading
            + FEATURE_WEIGHTS["position"] * position
        )
        scores[chunk.id] = max(0.0, min(1.0, raw))

    return scores


def _http_payload(query: str, texts: list[str], model: str) -> bytes:
    # One body carrying both spellings. TEI reads `texts` and ignores the rest; the
    # Cohere-compatible servers read `documents` and `model`. Sending both is what lets one
    # endpoint setting work against either without the operator having to know which they
    # are running.
    return json.dumps(
        {"query": query, "texts": texts, "documents": texts, "model": model, "return_documents": False}
    ).encode("utf-8")


def _parse_http_scores(payload) -> dict[int, float] | None:
    """Both response shapes, or None when neither is recognisable."""
    rows = None
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        rows = payload.get("results") or payload.get("data")
    if not isinstance(rows, list):
        return None
    parsed: dict[int, float] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        position = row.get("index")
        value = row.get("score")
        if value is None:
            value = row.get("relevance_score")
        if isinstance(position, int) and isinstance(value, (int, float)):
            parsed[position] = float(value)
    return parsed or None


def http_cross_encode(expanded, candidates: list, *, config) -> tuple[dict[str, float] | None, str | None]:
    """Score via a local reranking service. Returns `(scores, error)`; never raises."""
    if not config.reranker_endpoint:
        return None, "no endpoint configured"
    texts = [(candidate.chunk.content or "")[:2000] for candidate in candidates]
    if not texts:
        return {}, None

    request = urllib.request.Request(
        config.reranker_endpoint,
        data=_http_payload(expanded.normalized or expanded.original, texts, config.reranker_model),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=config.reranker_timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8", errors="replace"))
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError, ValueError) as exc:
        # Never the exception text: a URL with an embedded token is a plausible endpoint
        # configuration and this string is written to the application log.
        return None, type(exc).__name__

    by_position = _parse_http_scores(payload)
    if by_position is None:
        return None, "unrecognised response shape"

    # Drop out-of-range positions BEFORE normalising, not after. A service returning an
    # index this request never sent would otherwise stretch the min-max range with a value
    # no candidate can receive, compressing every real score toward the middle.
    usable = {
        position: value for position, value in by_position.items() if 0 <= position < len(candidates)
    }
    if not usable:
        # Parsed, but about no candidate we asked for. That is a failed rerank, not a rerank
        # that scored everything zero — and the difference decides whether the caller falls
        # back to the lexical scorer or silently ships an unreranked ordering.
        return None, "no scores matched the candidates"

    # Min-max onto 0..1. Cross-encoder services return raw logits, sigmoids or cosines
    # depending on the model, and `reranker_weight` blends this against a normalised
    # retrieval score — mixing a logit into that would let one candidate's -8.2 dominate
    # the whole ranking.
    values = list(usable.values())
    low, high = min(values), max(values)
    span = high - low
    scores: dict[str, float] = {}
    for position, value in usable.items():
        scores[candidates[position].chunk.id] = 0.5 if span <= 0 else (value - low) / span
    return scores, None


def rerank(expanded, candidates: list, index, *, tokenize, config) -> tuple[dict[str, float], str, str | None]:
    """Dispatch to the configured backend. Returns `(scores, backend_used, error)`."""
    if not config.reranker_enabled or config.reranker_backend == RERANKER_NONE or not candidates:
        return {}, RERANKER_NONE, None

    if config.reranker_backend == RERANKER_HTTP:
        scores, error = http_cross_encode(expanded, candidates, config=config)
        # `scores` must be non-empty as well as non-None: an empty dict means the service
        # answered about nothing we asked for, which reranks nothing while looking like
        # success.
        if scores:
            return scores, RERANKER_HTTP, None
        logger.info(
            "precision reranker unavailable, falling back to the lexical cross-encoder",
            extra={"extra_fields": {"backend": RERANKER_HTTP, "error": error}},
        )
        # Fall through rather than returning nothing: the lexical scorer is always
        # available and is a better ordering than none.
        return (
            lexical_cross_encode(expanded, candidates, index, tokenize=tokenize, config=config),
            RERANKER_LEXICAL,
            error,
        )

    return (
        lexical_cross_encode(expanded, candidates, index, tokenize=tokenize, config=config),
        RERANKER_LEXICAL,
        None,
    )
