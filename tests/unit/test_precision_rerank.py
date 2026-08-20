"""Unit tests for the high-precision mode's cross-encoder re-ranking stage.

`rag/precision/rerank.py` is the stage that decides the FINAL ordering of an answer's
evidence, and every one of its failure modes is silent:

* **The lexical backend is the product claim.** It is called a cross-encoder because it
  reads features that exist only for a (query, passage) PAIR — where the query's terms
  land, how close together, in what order, how rare they are in *this* corpus. If any of
  those degenerates into something derivable from the passage alone (a bag-of-words
  overlap, say), retrieval still returns five passages and the mode still answers; it just
  stops being more precise than the lexical mode that already existed. Nothing raises.
  So the tests below isolate one feature at a time and pin the exact arithmetic difference
  it contributes, rather than asserting a vague "better passage wins".
* **The HTTP backend must never raise and must never leak.** A rerank service that is down
  costs the ordering it would have improved, never the answer — so a URLError has to come
  back as a value and fall through to the lexical scorer. And the error string is written
  to the application log, while a rerank endpoint is exactly the kind of URL that carries a
  token in its query string, so the message may name the exception type and nothing else.
* **Normalisation is what makes `reranker_weight` meaningful.** A cross-encoder service
  returns raw logits; the pipeline blends its score against a min-max normalised retrieval
  score. Mixing an unnormalised -8.2 into that blend would let one candidate dominate the
  whole ranking, and the ranking would still look plausible from outside.

Everything here is pure: local structural stand-ins for a chunk and an index, so this file
imports no database, no session and no `rag.service`.
"""
import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field, replace

import pytest

from rag.precision.config import (
    RERANKER_HTTP,
    RERANKER_LEXICAL,
    RERANKER_NONE,
    PrecisionConfig,
)
from rag.precision.expansion import ExpandedQuery
from rag.precision.rerank import (
    FEATURE_WEIGHTS,
    PROXIMITY_WINDOW,
    _order,
    _parse_http_scores,
    _proximity_and_position,
    http_cross_encode,
    lexical_cross_encode,
    rerank,
)
from rag.precision.types import Candidate

pytestmark = pytest.mark.unit


# --- local stand-ins -------------------------------------------------------------------
# The pipeline is structurally typed (`types.ChunkLike`), which is the whole reason this
# file can test the ranking arithmetic with no database. Duplicated rather than imported
# from a sibling test module: tests/ is not a package.

_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{1,}")


def _tokenize(text: str) -> list[str]:
    """The corpus tokenizer's shape. Fixtures below deliberately contain no stopwords, so
    the production tokenizer's stopword filter cannot shift a token position out from
    under an assertion about proximity or order."""
    return [token.lower() for token in _TOKEN.findall(text)]


@dataclass
class _Chunk:
    id: str
    content: str


@dataclass
class _Meta:
    heading: str | None = None


@dataclass
class _Index:
    """Only the three attributes the reranker reads off an index."""

    document_count: int = 10
    document_frequencies: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)


def _query(text: str, terms: list[str], *, normalized: str | None = None) -> ExpandedQuery:
    """An `ExpandedQuery` as the pipeline hands it to the reranker: every typed term at
    weight 1.0, and the adjacent pairs that carry the user's word order."""
    return ExpandedQuery(
        original=text,
        normalized=normalized if normalized is not None else text,
        terms={term: 1.0 for term in terms},
        original_terms=list(terms),
        phrases=[(terms[i], terms[i + 1]) for i in range(len(terms) - 1)],
    )


def _candidates(*contents: str) -> list[Candidate]:
    return [Candidate(chunk=_Chunk(id=f"c{i}", content=text)) for i, text in enumerate(contents)]


def _score(expanded, candidates, index, config=None) -> dict[str, float]:
    return lexical_cross_encode(
        expanded,
        candidates,
        index,
        tokenize=_tokenize,
        config=config or PrecisionConfig(),
    )


class TestLexicalCrossEncode:
    """The pure-Python interaction scorer — the default backend, and the mode's claim."""

    def test_scores_stay_inside_the_unit_interval(self):
        # The weights sum to 1.0 precisely so a rerank score is comparable to a normalised
        # retrieval score; `reranker_weight` blends the two as if they mean the same thing.
        # A score outside 0..1 would silently break that blend rather than raise.
        expanded = _query("coolant flow rate", ["coolant", "flow", "rate"])
        index = _Index(
            document_count=10,
            document_frequencies={"coolant": 2, "flow": 3, "rate": 2},
            metadata={"c0": _Meta(heading="Coolant Flow Rate"), "c1": _Meta(heading=None)},
        )
        candidates = _candidates("coolant flow rate", "bracket torque values recorded")
        scores = _score(expanded, candidates, index)

        assert all(0.0 <= value <= 1.0 for value in scores.values())
        # Every feature maxed: full IDF coverage, adjacent terms, order preserved, the whole
        # query verbatim, a heading naming it, match at token zero.
        assert scores["c0"] == pytest.approx(1.0)
        # And a passage sharing nothing with the query is exactly zero, not a small
        # positive floor — a floor would put unrelated passages above nothing at all.
        assert scores["c1"] == 0.0

    def test_adjacent_query_terms_outrank_the_same_terms_forty_tokens_apart(self):
        """The core claim: this is an interaction score, not an overlap ratio.

        Both passages contain both query terms exactly once, so every bag-of-words scorer
        in the codebase ranks them identically. The gap between them here has to exceed the
        order feature's whole weight, which is what proves PROXIMITY contributed rather
        than order alone doing the work.
        """
        expanded = _query("coolant pressure", ["coolant", "pressure"])
        index = _Index(document_count=10, document_frequencies={"coolant": 2, "pressure": 2})
        padding = " ".join(f"padding{n}" for n in range(40))
        # The comma keeps the exact-substring feature out of the comparison: the tokens are
        # still adjacent, but "coolant, pressure" is not the normalized query verbatim.
        near, far = _candidates(
            "coolant, pressure regulator readings recorded",
            f"coolant {padding} pressure regulator readings recorded",
        )
        scores = _score(expanded, [near, far], index)

        assert scores["c0"] > scores["c1"]
        assert scores["c0"] - scores["c1"] > FEATURE_WEIGHTS["order"]

    def test_preserving_the_query_word_order_beats_the_reversed_passage(self):
        """"database connection timeout" and "timeout connection database" are the same bag
        of words and different questions. Nothing upstream of the reranker can tell them
        apart, so if this feature stops working nothing else catches it.

        The two passages are mirror images: identical terms, identical spacing, identical
        first-match offset, so coverage, proximity, position and exact are all equal and the
        difference is exactly the order weight.
        """
        expanded = _query("database connection timeout", ["database", "connection", "timeout"])
        index = _Index(
            document_count=10,
            document_frequencies={"database": 3, "connection": 3, "timeout": 2},
        )
        forward, backward = _candidates(
            "database primary connection request timeout observed",
            "timeout primary connection request database observed",
        )
        scores = _score(expanded, [forward, backward], index)

        assert scores["c0"] > scores["c1"]
        assert scores["c0"] - scores["c1"] == pytest.approx(FEATURE_WEIGHTS["order"])

    def test_the_whole_query_appearing_verbatim_scores_higher(self):
        # The only difference is the comma, which leaves the token stream — and therefore
        # coverage, proximity, order and position — byte-identical while breaking the
        # verbatim substring. So the gap is exactly the exact-match weight.
        expanded = _query("coolant flow rate", ["coolant", "flow", "rate"])
        index = _Index(
            document_count=10,
            document_frequencies={"coolant": 2, "flow": 3, "rate": 2},
        )
        verbatim, punctuated = _candidates(
            "coolant flow rate stays constant",
            "coolant flow, rate stays constant",
        )
        scores = _score(expanded, [verbatim, punctuated], index)

        assert scores["c0"] - scores["c1"] == pytest.approx(FEATURE_WEIGHTS["exact"])

    def test_a_short_query_never_earns_the_exact_match_credit(self):
        # `len(normalized_query) > 6` guards this: a two-letter query occurring verbatim is
        # a coincidence, and crediting it would hand full confidence to whichever passage
        # happened to contain the substring first.
        expanded = _query("valve", ["valve"], normalized="valve")
        index = _Index(document_count=10, document_frequencies={"valve": 2})
        candidate, = _candidates("valve seat inspection")
        scores = _score(expanded, [candidate], index)

        # coverage(1.0) + proximity(single term, 0.35) + position(1.0), and no exact credit.
        expected = (
            FEATURE_WEIGHTS["coverage"] + FEATURE_WEIGHTS["proximity"] * 0.35 + FEATURE_WEIGHTS["position"]
        )
        assert scores["c0"] == pytest.approx(expected)

    def test_a_heading_naming_the_query_words_lifts_the_passage(self):
        # The heading is derived structure the index already holds; two passages with the
        # same body but different headings are genuinely different evidence, and this is
        # the only feature that can see it.
        expanded = _query("torque specification", ["torque", "specification"])
        index = _Index(
            document_count=10,
            document_frequencies={"torque": 2, "specification": 2},
            metadata={"c0": _Meta(heading="Torque Specification"), "c1": _Meta(heading=None)},
        )
        candidates = _candidates(
            "torque specification values listed",
            "torque specification values listed",
        )
        scores = _score(expanded, candidates, index)

        assert scores["c0"] - scores["c1"] == pytest.approx(FEATURE_WEIGHTS["heading"])

    def test_a_chunk_absent_from_the_index_metadata_scores_without_a_heading(self):
        # `index.metadata.get()` returning None means "no derived structure for this chunk",
        # which must read as "no heading" and not raise — an index rebuilt while a request
        # is in flight is the ordinary way this happens.
        expanded = _query("torque specification", ["torque", "specification"])
        index = _Index(document_count=10, document_frequencies={"torque": 2, "specification": 2})
        candidate, = _candidates("torque specification values listed")

        scores = _score(expanded, [candidate], index)
        assert 0.0 < scores["c0"] < 1.0

    def test_a_rare_shared_term_beats_a_corpus_common_one(self):
        """IDF coverage: the feature a plain overlap ratio cannot see.

        Both passages share exactly one of the two query terms, in the same position, in
        the same sentence shape. Counting terms makes them equal. Weighting by IDF says the
        passage carrying the word that actually discriminates has covered almost all of the
        question's information and the other has covered almost none.
        """
        expanded = _query("system vaporizer", ["system", "vaporizer"])
        index = _Index(
            document_count=10,
            # "system" is in every chunk of the corpus; "vaporizer" in one.
            document_frequencies={"system": 10, "vaporizer": 1},
        )
        common, rare = _candidates(
            "maintenance report describes system operation clearly",
            "maintenance report describes vaporizer operation clearly",
        )
        scores = _score(expanded, [common, rare], index)

        assert scores["c1"] > scores["c0"]
        # Not marginally: the common term is floored at an IDF of 0.05 against the rare
        # term's ~2.0, so it carries about 2% of the query's information.
        assert scores["c0"] < scores["c1"] / 2


class TestProximityAndPosition:
    """The minimum-covering-window, isolated from the rest of the score."""

    def test_no_present_terms_scores_zero_on_both_features(self):
        proximity, position = _proximity_and_position({"valve"}, ["bracket", "torque", "cover"])
        assert (proximity, position) == (0.0, 0.0)

    def test_a_single_present_term_gets_the_neutral_proximity(self):
        # One term cannot be near or far from anything, so it takes a fixed middling value
        # rather than 0.0 (which would read as "the terms are scattered") or 1.0 (which
        # would let a one-word match outrank a genuinely tight multi-word one).
        proximity, position = _proximity_and_position({"beta"}, ["alpha", "beta", "gamma"])
        assert proximity == 0.35
        assert position == pytest.approx(1.0 - 1 / 3)

    def test_the_window_is_the_smallest_one_containing_every_present_term(self):
        # "alpha" and "beta" both occur twice: once far apart at the front, once adjacent at
        # the back. A sweep that stopped at the first covering window would score this as a
        # scattered match; the minimum window is what makes it a perfect one.
        tokens = ["alpha", "pad", "pad", "pad", "pad", "beta", "pad", "alpha", "beta"]
        proximity, position = _proximity_and_position({"alpha", "beta"}, tokens)

        assert proximity == pytest.approx(1.0)
        # Position still reports the FIRST occurrence in the passage, not the best window's
        # — the match starts at token zero, so nothing is trailing off the end.
        assert position == pytest.approx(1.0)

    def test_a_window_as_tight_as_the_terms_it_holds_is_perfect(self):
        proximity, _ = _proximity_and_position({"alpha", "beta"}, ["alpha", "beta", "pad"])
        assert proximity == pytest.approx(1.0)

    def test_proximity_decays_smoothly_and_monotonically_with_distance(self):
        # Smooth decay rather than a cutoff, so a slightly-too-wide match still outranks one
        # that is far worse. Note the decay begins at the FIRST slack token, well before
        # PROXIMITY_WINDOW — that constant sets the rate of the decay, not its onset.
        widths = [1, 4, 10, PROXIMITY_WINDOW, PROXIMITY_WINDOW * 3]
        values = []
        for width in widths:
            tokens = ["alpha"] + [f"pad{n}" for n in range(width - 1)] + ["beta"]
            proximity, _ = _proximity_and_position({"alpha", "beta"}, tokens)
            values.append(proximity)

        assert values[0] == pytest.approx(1.0)
        assert values == sorted(values, reverse=True)
        assert all(0.0 < value <= 1.0 for value in values)
        # Halfway to the nominal window the score has already fallen well below half.
        assert values[2] < 0.5

    def test_position_falls_as_the_first_match_moves_down_the_passage(self):
        early = ["alpha", "beta"] + [f"pad{n}" for n in range(20)]
        late = [f"pad{n}" for n in range(20)] + ["alpha", "beta"]

        _, early_position = _proximity_and_position({"alpha", "beta"}, early)
        _, late_position = _proximity_and_position({"alpha", "beta"}, late)

        assert early_position == pytest.approx(1.0)
        assert late_position < early_position


class TestOrder:
    """The share of the question's adjacent word pairs that survive in the passage."""

    def test_no_phrases_scores_zero(self):
        # A single-word query has no pairs. 0.0 (not 1.0) is right: there is no order
        # evidence to credit, and crediting it would hand every passage a free 0.16.
        assert _order([], ["alpha", "beta"]) == 0.0

    def test_a_gap_of_three_still_counts_and_a_gap_of_four_does_not(self):
        # "connection pool timeout" should still credit ("connection", "timeout"), so the
        # window is deliberately wider than strict adjacency — but bounded, or the feature
        # degenerates into "both words appear somewhere", which coverage already says.
        assert _order([("alpha", "beta")], ["alpha", "pad", "pad", "beta"]) == 1.0
        assert _order([("alpha", "beta")], ["alpha", "pad", "pad", "pad", "beta"]) == 0.0

    def test_direction_matters(self):
        assert _order([("alpha", "beta")], ["beta", "alpha"]) == 0.0

    def test_a_missing_term_drops_only_its_own_pair(self):
        phrases = [("alpha", "beta"), ("beta", "gamma")]
        assert _order(phrases, ["alpha", "beta", "delta"]) == pytest.approx(0.5)


class TestParseHttpScores:
    """Both wire shapes the local rerank services speak, and everything else rejected."""

    def test_a_bare_list_of_index_and_score(self):
        # text-embeddings-inference returns the array directly.
        assert _parse_http_scores([{"index": 0, "score": 1.5}, {"index": 1, "score": -0.5}]) == {
            0: 1.5,
            1: -0.5,
        }

    def test_a_results_array_of_index_and_relevance_score(self):
        # The Cohere-compatible servers wrap it and rename the field.
        payload = {"results": [{"index": 2, "relevance_score": 0.9}, {"index": 0, "relevance_score": 0.1}]}
        assert _parse_http_scores(payload) == {2: 0.9, 0: 0.1}

    def test_a_data_array_is_accepted_too(self):
        assert _parse_http_scores({"data": [{"index": 0, "score": 0.4}]}) == {0: 0.4}

    @pytest.mark.parametrize(
        "payload",
        [
            None,
            "not json at all",
            42,
            {},
            {"results": "nope"},
            [],
            ["a string", 7],
            [{"no_index": 1, "no_score": 2}],
        ],
    )
    def test_garbage_returns_none_rather_than_an_empty_ranking(self, payload):
        # None is what makes the caller fall back to the lexical scorer. An empty dict would
        # be indistinguishable from "the service ranked nothing", and the ordering would be
        # silently left as retrieval produced it with the trace claiming http succeeded.
        assert _parse_http_scores(payload) is None

    def test_rows_with_the_wrong_types_are_skipped_not_fatal(self):
        # One malformed row from a service under load must not discard the rows around it.
        payload = [
            {"index": "0", "score": 1.0},      # index is a string
            {"index": 1, "score": "high"},     # score is a string
            {"index": 2, "score": 0.5},        # the only usable row
            "not a row",
        ]
        assert _parse_http_scores(payload) == {2: 0.5}

    def test_an_integer_score_is_accepted_as_a_float(self):
        assert _parse_http_scores([{"index": 0, "score": 1}]) == {0: 1.0}


# --- HTTP backend ----------------------------------------------------------------------

_ENDPOINT = "http://localhost:9999/rerank?token=hunter2"


class _FakeResponse:
    """The context-manager shape `urllib.request.urlopen` returns."""

    def __init__(self, body: bytes):
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _http_config(**overrides) -> PrecisionConfig:
    return replace(
        PrecisionConfig(),
        reranker_backend=RERANKER_HTTP,
        reranker_endpoint=_ENDPOINT,
        **overrides,
    )


class TestHttpCrossEncode:
    def test_no_endpoint_configured_is_reported_not_attempted(self, monkeypatch):
        def explode(*args, **kwargs):  # pragma: no cover - proves no call is made
            raise AssertionError("urlopen must not be called without an endpoint")

        monkeypatch.setattr(urllib.request, "urlopen", explode)
        config = replace(PrecisionConfig(), reranker_backend=RERANKER_HTTP, reranker_endpoint="")

        scores, error = http_cross_encode(_query("valve", ["valve"]), _candidates("valve seat"), config=config)

        assert scores is None
        assert error == "no endpoint configured"

    def test_raw_logits_are_min_max_normalised_onto_zero_to_one_by_chunk_id(self, monkeypatch):
        """The property `reranker_weight` depends on.

        A cross-encoder service returns logits, sigmoids or cosines depending on the model.
        The pipeline blends this against a min-max normalised retrieval score, so a raw
        -8.2 reaching that blend would let one candidate dominate the whole ranking.
        """
        payload = [
            {"index": 0, "score": -8.2},
            {"index": 1, "score": 3.1},
            {"index": 2, "score": -2.55},
        ]
        monkeypatch.setattr(
            urllib.request,
            "urlopen",
            lambda request, timeout=None: _FakeResponse(json.dumps(payload).encode("utf-8")),
        )

        scores, error = http_cross_encode(
            _query("valve clearance", ["valve", "clearance"]),
            _candidates("first passage", "second passage", "third passage"),
            config=_http_config(),
        )

        assert error is None
        # Keyed by chunk id, not by the service's positional index — the pipeline looks up
        # by `candidate.chunk_id` and a positional key would silently score nothing.
        assert set(scores) == {"c0", "c1", "c2"}
        assert scores["c0"] == pytest.approx(0.0)
        assert scores["c1"] == pytest.approx(1.0)
        assert scores["c2"] == pytest.approx(0.5)

    def test_identical_scores_collapse_to_a_neutral_half_rather_than_dividing_by_zero(self, monkeypatch):
        payload = [{"index": 0, "score": 2.0}, {"index": 1, "score": 2.0}]
        monkeypatch.setattr(
            urllib.request,
            "urlopen",
            lambda request, timeout=None: _FakeResponse(json.dumps(payload).encode("utf-8")),
        )

        scores, error = http_cross_encode(
            _query("valve clearance", ["valve", "clearance"]),
            _candidates("first passage", "second passage"),
            config=_http_config(),
        )

        assert error is None
        assert scores == {"c0": 0.5, "c1": 0.5}

    def test_a_position_outside_the_candidate_list_is_dropped_before_normalising(self, monkeypatch):
        # A service that echoes an index we never sent must not raise an IndexError inside a
        # chat response — and must not distort the ranking either. The stray row is removed
        # BEFORE the min-max, so the two real candidates still span the full 0..1 range;
        # dropping it afterwards left them compressed into the bottom fifth of a range set by
        # a score no candidate could receive.
        payload = [{"index": 0, "score": 0.0}, {"index": 1, "score": 1.0}, {"index": 9, "score": 5.0}]
        monkeypatch.setattr(
            urllib.request,
            "urlopen",
            lambda request, timeout=None: _FakeResponse(json.dumps(payload).encode("utf-8")),
        )

        scores, error = http_cross_encode(
            _query("valve clearance", ["valve", "clearance"]),
            _candidates("first passage", "second passage"),
            config=_http_config(),
        )

        assert error is None
        assert set(scores) == {"c0", "c1"}
        assert scores["c0"] == pytest.approx(0.0)
        assert scores["c1"] == pytest.approx(1.0)

    def test_the_request_carries_both_field_spellings_and_truncates_long_passages(self, monkeypatch):
        # One body serves both server families, so an operator does not have to know which
        # one they are running. The 2000-char cap is what stops a 40-chunk pool of full
        # passages becoming a multi-megabyte POST.
        captured = {}

        def capture(request, timeout=None):
            captured["request"] = request
            captured["timeout"] = timeout
            return _FakeResponse(json.dumps([{"index": 0, "score": 1.0}]).encode("utf-8"))

        monkeypatch.setattr(urllib.request, "urlopen", capture)
        expanded = _query("valve clearance", ["valve", "clearance"])
        candidate, = _candidates("x" * 3000)

        http_cross_encode(expanded, [candidate], config=_http_config(reranker_timeout_seconds=7))

        body = json.loads(captured["request"].data.decode("utf-8"))
        assert body["query"] == expanded.normalized
        assert body["texts"] == body["documents"] == ["x" * 2000]
        assert body["model"] == PrecisionConfig().reranker_model
        assert body["return_documents"] is False
        assert captured["request"].get_method() == "POST"
        assert captured["timeout"] == 7

    def test_a_transport_failure_is_returned_as_a_value_and_never_raised(self, monkeypatch):
        def refuse(request, timeout=None):
            raise urllib.error.URLError(f"connection refused talking to {_ENDPOINT}")

        monkeypatch.setattr(urllib.request, "urlopen", refuse)

        scores, error = http_cross_encode(
            _query("valve clearance", ["valve", "clearance"]),
            _candidates("first passage"),
            config=_http_config(),
        )

        assert scores is None
        assert error == "URLError"

    def test_the_error_string_never_carries_the_endpoint_url(self, monkeypatch):
        """This string is written to the application log and pasted into bug reports.

        A rerank endpoint with an embedded token is a plausible configuration, and the
        exception text quotes the URL back — which is why only the exception TYPE is
        reported. Asserting the absence of the whole URL is not enough on its own, so the
        credential substring is checked separately.
        """
        def refuse(request, timeout=None):
            raise urllib.error.URLError(f"failed to reach {_ENDPOINT}")

        monkeypatch.setattr(urllib.request, "urlopen", refuse)

        _, error = http_cross_encode(
            _query("valve clearance", ["valve", "clearance"]),
            _candidates("first passage"),
            config=_http_config(),
        )

        assert _ENDPOINT not in error
        assert "hunter2" not in error
        assert "localhost" not in error

    @pytest.mark.parametrize(
        "exception, expected",
        [
            (TimeoutError("slow"), "TimeoutError"),
            (OSError("socket gone"), "OSError"),
            (urllib.error.HTTPError(_ENDPOINT, 503, "Service Unavailable", {}, None), "HTTPError"),
        ],
    )
    def test_every_transport_failure_mode_degrades_to_a_named_type(self, monkeypatch, exception, expected):
        def fail(request, timeout=None):
            raise exception

        monkeypatch.setattr(urllib.request, "urlopen", fail)

        scores, error = http_cross_encode(
            _query("valve clearance", ["valve", "clearance"]),
            _candidates("first passage"),
            config=_http_config(),
        )

        assert scores is None
        assert error == expected

    def test_a_body_that_is_not_json_is_a_reported_failure_not_an_exception(self, monkeypatch):
        monkeypatch.setattr(
            urllib.request,
            "urlopen",
            lambda request, timeout=None: _FakeResponse(b"<html>502 Bad Gateway</html>"),
        )

        scores, error = http_cross_encode(
            _query("valve clearance", ["valve", "clearance"]),
            _candidates("first passage"),
            config=_http_config(),
        )

        assert scores is None
        assert error == "JSONDecodeError"

    def test_valid_json_in_an_unknown_shape_is_reported_as_such(self, monkeypatch):
        monkeypatch.setattr(
            urllib.request,
            "urlopen",
            lambda request, timeout=None: _FakeResponse(json.dumps({"scores": [0.4]}).encode("utf-8")),
        )

        scores, error = http_cross_encode(
            _query("valve clearance", ["valve", "clearance"]),
            _candidates("first passage"),
            config=_http_config(),
        )

        assert scores is None
        assert error == "unrecognised response shape"


class TestRerankDispatch:
    """Which backend runs, and what happens when the configured one cannot."""

    def _fixture(self):
        expanded = _query("coolant flow rate", ["coolant", "flow", "rate"])
        index = _Index(
            document_count=10,
            document_frequencies={"coolant": 2, "flow": 3, "rate": 2},
        )
        candidates = _candidates("coolant flow rate stays constant", "bracket torque values")
        return expanded, candidates, index

    def test_the_none_backend_scores_nothing_and_says_so(self):
        expanded, candidates, index = self._fixture()
        config = replace(PrecisionConfig(), reranker_backend=RERANKER_NONE)

        scores, backend, error = rerank(expanded, candidates, index, tokenize=_tokenize, config=config)

        # An empty dict is what the pipeline reads as "leave the retrieval ranking alone";
        # the ablation variants in benchmark.py depend on this being a no-op, not a zero.
        assert scores == {}
        assert backend == RERANKER_NONE
        assert error is None

    def test_the_disable_flag_wins_over_the_configured_backend(self):
        expanded, candidates, index = self._fixture()
        config = replace(PrecisionConfig(), reranker_enabled=False, reranker_backend=RERANKER_LEXICAL)

        assert rerank(expanded, candidates, index, tokenize=_tokenize, config=config) == (
            {},
            RERANKER_NONE,
            None,
        )

    def test_an_empty_candidate_pool_short_circuits(self):
        expanded, _, index = self._fixture()

        assert rerank(expanded, [], index, tokenize=_tokenize, config=PrecisionConfig()) == (
            {},
            RERANKER_NONE,
            None,
        )

    def test_the_default_backend_is_the_lexical_scorer(self):
        expanded, candidates, index = self._fixture()

        scores, backend, error = rerank(
            expanded, candidates, index, tokenize=_tokenize, config=PrecisionConfig()
        )

        assert backend == RERANKER_LEXICAL
        assert error is None
        assert set(scores) == {"c0", "c1"}
        assert scores["c0"] > scores["c1"]

    def test_an_unreachable_http_endpoint_falls_back_to_the_lexical_scorer(self, monkeypatch):
        """Every failure here is inert: a reranker that is down costs the ordering it would
        have improved, never the answer. The error still surfaces, so a silently-degraded
        deployment is visible in the trace rather than merely slower."""
        def refuse(request, timeout=None):
            raise urllib.error.URLError("connection refused")

        monkeypatch.setattr(urllib.request, "urlopen", refuse)
        expanded, candidates, index = self._fixture()

        scores, backend, error = rerank(
            expanded, candidates, index, tokenize=_tokenize, config=_http_config()
        )

        assert backend == RERANKER_LEXICAL
        assert error == "URLError"
        # The fallback is a real ranking, identical to what the lexical backend would have
        # produced on its own — not an empty dict wearing a backend name.
        assert scores == lexical_cross_encode(
            expanded, candidates, index, tokenize=_tokenize, config=PrecisionConfig()
        )

    def test_a_missing_endpoint_falls_back_and_reports_the_configuration_error(self):
        expanded, candidates, index = self._fixture()
        config = replace(PrecisionConfig(), reranker_backend=RERANKER_HTTP, reranker_endpoint="")

        scores, backend, error = rerank(expanded, candidates, index, tokenize=_tokenize, config=config)

        assert backend == RERANKER_LEXICAL
        assert error == "no endpoint configured"
        assert scores["c0"] > scores["c1"]

    def test_a_working_http_endpoint_is_used_and_reported_as_http(self, monkeypatch):
        payload = {"results": [{"index": 0, "relevance_score": -1.0}, {"index": 1, "relevance_score": 4.0}]}
        monkeypatch.setattr(
            urllib.request,
            "urlopen",
            lambda request, timeout=None: _FakeResponse(json.dumps(payload).encode("utf-8")),
        )
        expanded, candidates, index = self._fixture()

        scores, backend, error = rerank(
            expanded, candidates, index, tokenize=_tokenize, config=_http_config()
        )

        assert backend == RERANKER_HTTP
        assert error is None
        # The service's verdict, normalised — and it is allowed to disagree with the lexical
        # ordering, which is the entire reason a deployment would configure one.
        assert scores == {"c0": 0.0, "c1": 1.0}
