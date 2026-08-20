"""End-to-end unit tests for the high-precision pipeline — `rag/precision/pipeline.py`.

`pipeline.py` is the only module in the package that nothing else can stand in for: every
other file is one stage, and every stage's own test can pass while the orchestration between
them is wrong. Four properties are pinned here because each one fails *silently* — the mode
still returns passages, so nothing raises and nothing logs an error:

* **An empty result must carry its reason.** Three distinct situations produce zero
  results — no indexed chunks, a query that tokenizes to nothing, and a query no chunk
  contains — and the caller's refusal wording is chosen from the trace. Collapsing them into
  a bare empty list is how "you have not uploaded anything" becomes "your documents do not
  cover that".
* **The dense side is strictly additive.** A chunk with zero lexical score must still surface
  when the dense retriever ranked it (the synonym case — the entire reason the dense input
  exists), and supplying dense candidates must be incapable of removing a chunk that lexical
  retrieval found. Both halves are asserted, because a fusion that only ever *adds* is the
  property that makes enabling embeddings safe per document (CLAUDE.md §22).
* **The negative penalty is multiplicative.** Subtracting a constant would make a weak match
  negative and a strong one merely weaker, inverting the ordering it was meant to adjust.
  Scaling cannot produce a negative score, and the penalty is capped at 0.95, so the test
  drives it to its ceiling and asserts the floor holds.
* **The trace must narrate every stage, in order, with counts that chain.** "Why is this
  passage missing" is answerable only if stage N's `count_in` is stage N-1's `count_out`, and
  the trace is written to the application log, so it must also survive `json.dumps`.

Everything here is a pure unit: a local `FakeChunk` satisfies `types.ChunkLike` structurally
and a local tokenizer mirrors `rag.service._tokenize`, so this file imports no model, no
session and no `rag.service` — which is the isolation property the package was built for.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass

import pytest

from rag.precision import index as index_module
from rag.precision.config import PrecisionConfig
from rag.precision.pipeline import (
    MIN_NORMALIZED,
    PipelineInputs,
    _cosine,
    _keyword_score,
    _negative_penalty,
    _normalize_scores,
    _similarity_function,
    retrieve,
)

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _clear_index_cache():
    """The index cache is process-wide and keyed by resource id.

    Two tests using the same resource id with corpora of the same shape would otherwise
    share derived data, and the second one would be testing the first one's index.
    """
    index_module.clear_cache()
    yield
    index_module.clear_cache()


# --------------------------------------------------------------------------------------
# The tokenizer. Deliberately a local copy of `rag.service._tokenize` rather than an import:
# the pipeline takes the tokenizer as an injected callable precisely so it can be exercised
# without `rag.service`, and importing the real one here would quietly give that up. What
# matters is that ONE tokenizer builds the index and reads the query — see `PipelineInputs`.
# --------------------------------------------------------------------------------------
_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{1,}")
_STOPWORDS = {
    "a", "all", "an", "and", "any", "are", "as", "at", "be", "by", "do", "does", "each",
    "for", "from", "how", "in", "is", "it", "its", "not", "of", "on", "one", "or", "so",
    "than", "that", "the", "this", "to", "what", "which", "with",
}


def _tokenize(text: str) -> list[str]:
    return [
        token.lower()
        for token in _TOKEN.findall(text)
        if token.lower() not in _STOPWORDS and len(token) > 1
    ]


@dataclass
class FakeChunk:
    """The attributes the precision package reads off a chunk.

    A local dataclass rather than `models.rag.DocumentChunk`: `types.ChunkLike` is a
    structural protocol, and a test that imported the model would need a database import to
    prove something that is pure arithmetic.
    """

    id: str
    content: str
    file_id: str | None = "file-1"
    chunk_index: int = 0
    source_name: str = "pooling.txt"
    modality: str = "text"
    title: str | None = None
    contextual_content: str = ""
    page_start: int | None = None
    page_end: int | None = None
    char_start: int | None = None
    char_end: int | None = None


# A ten-chunk corpus across four documents. The shape is chosen, not incidental:
#
#   file-1 (4 chunks) — the topic the query asks about. chunk_index 0..2 form one parent
#                       window at the default group size of 3, so parent recovery has
#                       siblings to find; chunk_index 3 opens a window of its own and so
#                       has none, which is the "no parent" case.
#   file-2 (3 chunks) — one chunk shares the query's vocabulary from a different document,
#                       so the candidate pool is not a single file.
#   file-3 (2 chunks) — shares NO term with the query. This is the dense-only chunk: the
#                       synonym case the dense input exists for.
#   file-4 (1 chunk)  — a single-chunk document, i.e. a family of one.
QUERY = "database connection pool timeout"

CORPUS: list[FakeChunk] = [
    FakeChunk(
        "c1",
        "3.1 Connection Pooling The database connection pool keeps twenty idle connections "
        "ready so a request never waits for a fresh socket handshake.",
        "file-1", 0, "pooling.txt", page_start=4,
    ),
    FakeChunk(
        "c2",
        "3.2 Timeout Behaviour A borrower blocked longer than the configured timeout receives "
        "an error instead of waiting indefinitely for the pool.",
        "file-1", 1, "pooling.txt", page_start=4,
    ),
    FakeChunk(
        "c3",
        "3.3 Sizing Guidance the pool ceiling should follow measured concurrency; oversizing "
        "wastes database memory and undersizing serialises traffic.",
        "file-1", 2, "pooling.txt", page_start=5,
    ),
    FakeChunk(
        "c4",
        "Leak detection prints a stack trace for any borrower holding a pool lease beyond the "
        "abandonment threshold.",
        "file-1", 3, "pooling.txt", page_start=5,
    ),
    FakeChunk(
        "c5",
        "Every invoice run opens one database connection and holds it until the ledger export "
        "finishes writing.",
        "file-2", 0, "billing.txt",
    ),
    FakeChunk(
        "c6",
        "A refund reverses one invoice line and never rewrites the original document, which "
        "auditors require.",
        "file-2", 1, "billing.txt",
    ),
    FakeChunk(
        "c7",
        "Dunning reminders escalate after fourteen, thirty and sixty days before collection is "
        "handed onward.",
        "file-2", 2, "billing.txt",
    ),
    FakeChunk(
        "c8",
        "Lubricate every chain link, torque each spoke nipple and replace worn brake shoes "
        "before winter riding.",
        "file-3", 0, "cycling.txt",
    ),
    FakeChunk(
        "c9",
        "Saddle height follows inseam measurement; a rider whose knee locks out is sitting far "
        "too high.",
        "file-3", 1, "cycling.txt",
    ),
    FakeChunk(
        "c10",
        "Jumbo frames raise throughput but a stalled connection still shows up as a timeout at "
        "the far end.",
        "file-4", 0, "network.txt",
    ),
]

# The chunks `QUERY` reaches lexically, in no particular order. c8 and c9 are absent by
# construction and c6/c7 share nothing with the question either.
LEXICAL_MATCHES = {"c1", "c2", "c3", "c4", "c5", "c10"}


def _terms_of(chunk: FakeChunk) -> dict[str, int]:
    return dict(Counter(_tokenize(chunk.content)))


def _retrieve(query: str = QUERY, *, chunks=None, resource_id="resource-1", dense=None,
              vector_of=None, **overrides):
    """One pipeline run.

    `dictionary={}` is passed explicitly rather than left to default: `None` makes the
    pipeline call `expansion.load_dictionary()`, which reads `PRECISION_RAG_DICTIONARY_PATH`
    from the environment — a test whose expansion depends on the machine it runs on.
    Likewise `PrecisionConfig()` is constructed rather than fetched through the cached
    `get_precision_config()`, so no `PRECISION_RAG_*` variable can move a threshold.
    """
    inputs = PipelineInputs(
        resource_id=resource_id,
        query=query,
        chunks=list(CORPUS if chunks is None else chunks),
        tokenize=_tokenize,
        terms_of=_terms_of,
        dense_candidates=dense,
        vector_of=vector_of,
        config=PrecisionConfig().with_overrides(overrides),
        dictionary={},
    )
    return retrieve(inputs)


def _stages(outcome) -> dict[str, object]:
    return {record.stage: record for record in outcome.trace.stages}


class TestEmptyOutcomes:
    """Zero results is an answer, and the trace has to say WHICH answer."""

    def test_no_chunks_returns_an_empty_outcome_naming_the_empty_resource(self):
        outcome = _retrieve(chunks=[], resource_id="empty-resource")

        assert outcome.results == []
        assert outcome.chunk_ids == []
        # A single stage, and it is the input gate — nothing downstream ran, so nothing
        # downstream may claim to have looked.
        assert [record.stage for record in outcome.trace.stages] == ["input"]
        assert outcome.trace.stages[0].detail["reason"] == "resource has no indexed chunks"

    def test_a_query_of_only_stopwords_returns_empty_before_touching_the_corpus(self):
        # "what is it" tokenizes to nothing, so there is nothing to search for. Returning
        # empty lets the caller's existing sufficiency gate produce the refusal it already
        # produces for every other mode instead of inventing a second wording for it.
        outcome = _retrieve("what is it")

        assert outcome.results == []
        assert [record.stage for record in outcome.trace.stages] == ["normalize"]
        assert outcome.trace.stages[0].detail["reason"] == "query has no searchable terms"

    def test_a_query_no_chunk_contains_reports_that_it_reached_retrieval(self):
        # Distinct from the two above: retrieval genuinely ran and found nothing, which is
        # the only one of the three that means "your documents do not cover this".
        outcome = _retrieve("zzzqqqwww vvvxxxyyy")

        assert outcome.results == []
        assert [record.stage for record in outcome.trace.stages] == [
            "metadata_filter", "bm25", "dense", "candidates",
        ]
        assert _stages(outcome)["candidates"].detail["reason"] == "no chunk contained any query term"
        # BM25 looked at the whole corpus and scored none of it.
        assert _stages(outcome)["bm25"].count_in == len(CORPUS)
        assert _stages(outcome)["bm25"].count_out == 0


class TestRanking:
    def test_only_chunks_sharing_a_query_term_are_retrieved(self):
        outcome = _retrieve()

        assert set(outcome.chunk_ids) == LEXICAL_MATCHES

    def test_results_are_ordered_by_final_score_descending(self):
        # Asserted with MMR off, because MMR is the one stage that deliberately trades
        # relevance for complementary evidence and so may legitimately reorder — see the
        # next test for what survives when it is on.
        outcome = _retrieve(mmr_enabled=False)
        scores = [result.score for result in outcome.results]

        assert len(scores) > 1
        assert scores == sorted(scores, reverse=True)

    def test_mmr_still_puts_the_best_passage_first(self):
        # MMR's first pick is pure relevance: with nothing selected there is nothing to be
        # diverse from, so discarding the best passage on principle would be a bug.
        outcome = _retrieve()

        assert outcome.results[0].score == max(result.score for result in outcome.results)
        assert outcome.results[0].chunk.id == "c1"

    def test_final_k_caps_the_result_count(self):
        capped = _retrieve(final_k=3)

        assert len(capped.results) == 3
        # ...and it is a ceiling, not a target: a corpus with fewer matches than final_k
        # returns the matches it has rather than padding the list with irrelevance.
        assert len(_retrieve(final_k=50).results) == len(LEXICAL_MATCHES)

    def test_every_score_carried_on_a_result_is_reported_separately(self):
        # Scores are kept apart rather than collapsed as they are produced, because a single
        # running total cannot answer "why did this rank here".
        top = _retrieve().results[0]

        assert top.bm25_score > 0
        assert top.dense_score == 0.0  # no dense candidates were supplied
        assert top.rerank_score is not None  # the lexical cross-encoder is on by default
        assert top.retrieval_method == "bm25"


class TestTrace:
    EXPECTED_STAGES = [
        "metadata_filter", "bm25", "dense", "candidate_pool",
        "rerank", "dedup", "mmr", "parent_recovery",
    ]
    EXPECTED_TIMINGS = {
        "normalize", "index", "expand", "metadata_filter", "bm25", "dense",
        "fuse", "rerank", "dedup", "mmr", "parents", "total",
    }

    def test_every_stage_is_recorded_in_pipeline_order(self):
        outcome = _retrieve()

        assert [record.stage for record in outcome.trace.stages] == self.EXPECTED_STAGES

    def test_stage_counts_chain_from_one_stage_to_the_next(self):
        # This is what makes "why is chunk X missing" answerable: the stage where count_out
        # first drops below count_in is the stage that dropped it. A gap in the chain means
        # a stage silently reported on a pool it did not receive.
        outcome = _retrieve()
        stages = _stages(outcome)

        assert stages["metadata_filter"].count_in == len(CORPUS)
        assert stages["bm25"].count_in == len(CORPUS)
        assert stages["bm25"].count_out == len(LEXICAL_MATCHES)
        assert stages["candidate_pool"].count_in == len(LEXICAL_MATCHES)
        for earlier, later in zip(self.EXPECTED_STAGES[3:], self.EXPECTED_STAGES[4:]):
            assert stages[later].count_in == stages[earlier].count_out
        assert stages["parent_recovery"].count_out == len(outcome.results)

    def test_no_stage_reports_more_out_than_in(self):
        # Every stage from BM25 onward narrows or holds. A stage that grew its pool would be
        # inventing candidates the stage before it rejected.
        for record in _retrieve().trace.stages:
            assert record.count_out <= record.count_in, record.stage

    def test_every_stage_is_timed(self):
        outcome = _retrieve()

        assert set(outcome.trace.timings_ms) == self.EXPECTED_TIMINGS
        assert all(value >= 0.0 for value in outcome.trace.timings_ms.values())
        # `total` spans the whole run, so no individual stage may exceed it.
        assert all(
            value <= outcome.trace.timings_ms["total"] + 1e-6
            for key, value in outcome.trace.timings_ms.items()
            if key != "total"
        )

    def test_the_trace_records_the_query_side_of_the_run(self):
        outcome = _retrieve()

        assert outcome.trace.original_query == QUERY
        assert outcome.trace.normalized_query == QUERY  # already lowercase and unpunctuated
        assert outcome.trace.final_chunk_ids == outcome.chunk_ids
        assert outcome.trace.reranker_backend == "lexical"
        assert outcome.trace.reranker_error is None
        # Expansion added terms the user did not type; the typed ones are not "added".
        assert set(outcome.trace.expanded_terms).isdisjoint(_tokenize(QUERY))

    def test_to_dict_is_json_serialisable(self):
        # The trace is written to the application log through the JSON formatter, so a value
        # `json.dumps` cannot encode is a log line that never gets written.
        outcome = _retrieve(dense=[("c8", 0.93)])

        encoded = json.dumps(outcome.trace.to_dict())
        restored = json.loads(encoded)

        assert [stage["stage"] for stage in restored["stages"]] == self.EXPECTED_STAGES
        assert restored["final_chunk_ids"] == outcome.chunk_ids
        # Per-stage detail rides along, not just the counts — the dedup/MMR decisions are
        # the part that explains a missing passage.
        assert all({"stage", "in", "out"} <= set(stage) for stage in restored["stages"])

    def test_an_empty_trace_is_also_json_serialisable(self):
        # The failure paths return early, so their traces are the ones least likely to have
        # been exercised — and they are the traces a bug report will actually contain.
        for query, chunks in (("what is it", None), (QUERY, []), ("zzzqqqwww vvvxxxyyy", None)):
            outcome = _retrieve(query, chunks=chunks, resource_id=f"r-{len(chunks or CORPUS)}-{query}")
            assert json.loads(json.dumps(outcome.trace.to_dict()))["original_query"] == query


class TestDenseFusion:
    """The dense side is additive: it may surface a passage, never remove one."""

    def test_a_chunk_with_zero_lexical_score_surfaces_on_its_dense_score(self):
        # c8 is about bicycle maintenance and shares no term with a question about database
        # connection pools — this is precisely the synonym/vocabulary case that lexical
        # retrieval cannot bridge and that the dense input exists to cover.
        assert "c8" not in _retrieve().chunk_ids

        outcome = _retrieve(dense=[("c8", 0.93)])
        surfaced = {result.chunk.id: result for result in outcome.results}

        assert "c8" in surfaced
        assert surfaced["c8"].bm25_score == 0.0
        assert surfaced["c8"].dense_score == pytest.approx(0.93)
        assert surfaced["c8"].retrieval_method == "dense"

    def test_dense_candidates_never_displace_a_lexical_match(self):
        # The whole safety argument for enabling embeddings per document is that fusion can
        # only widen what the system will answer.
        outcome = _retrieve(dense=[("c8", 0.93), ("c9", 0.71)])

        assert LEXICAL_MATCHES <= set(outcome.chunk_ids)

    def test_a_chunk_matched_by_both_reports_both_sources(self):
        outcome = _retrieve(dense=[("c1", 0.88)])
        top = {result.chunk.id: result for result in outcome.results}["c1"]

        assert top.retrieval_method == "bm25+dense"
        assert top.bm25_score > 0 and top.dense_score == pytest.approx(0.88)

    def test_a_dense_candidate_for_an_unknown_chunk_is_ignored(self):
        # A stale dense index can name a chunk this resource no longer has. Scoring it would
        # mean a `KeyError` or, worse, a result with no passage behind it.
        outcome = _retrieve(dense=[("c8", 0.93), ("no-such-chunk", 0.99)])

        assert "no-such-chunk" not in outcome.chunk_ids
        # Reported honestly: two candidates arrived, one was usable.
        assert _stages(outcome)["dense"].count_in == 2
        assert _stages(outcome)["dense"].count_out == 1

    def test_a_non_positive_dense_similarity_is_ignored(self):
        outcome = _retrieve(dense=[("c8", 0.0), ("c9", -0.4)])

        assert set(outcome.chunk_ids) == LEXICAL_MATCHES
        assert _stages(outcome)["dense"].count_out == 0

    def test_dense_disabled_ignores_the_candidates_entirely(self):
        outcome = _retrieve(dense=[("c8", 0.93), ("c9", 0.71)], dense_enabled=False)

        assert "c8" not in outcome.chunk_ids
        assert set(outcome.chunk_ids) == LEXICAL_MATCHES
        # The candidates that were offered are still counted in, so a deployment that turned
        # the dense side off can see from the trace that it was the switch and not the data.
        assert _stages(outcome)["dense"].count_in == 2
        assert _stages(outcome)["dense"].count_out == 0

    def test_dense_disabled_leaves_a_dense_only_question_with_nothing(self):
        outcome = _retrieve("zzzqqqwww", dense=[("c8", 0.93)], dense_enabled=False)

        assert outcome.results == []
        assert _stages(outcome)["candidates"].detail["reason"] == "no chunk contained any query term"


class TestNormalizeScores:
    """Min-max onto 0..1, so the fusion weights mean what they say."""

    def test_an_empty_pool_normalizes_to_an_empty_pool(self):
        assert _normalize_scores({}) == {}

    def test_an_all_equal_positive_pool_maps_to_one(self):
        # Every candidate scored the same, so they are all equally good. 0.5 would give a
        # signal that fired for everybody the same weight as one that fired for nobody.
        assert _normalize_scores({"a": 3.0, "b": 3.0, "c": 3.0}) == {"a": 1.0, "b": 1.0, "c": 1.0}

    def test_an_all_zero_pool_maps_to_zero(self):
        # The everyday case: no dense candidates were supplied, so the dense signal fired for
        # nobody and must contribute nothing to any candidate's combined score.
        assert _normalize_scores({"a": 0.0, "b": 0.0}) == {"a": 0.0, "b": 0.0}

    def test_a_spread_pool_maps_the_extremes_to_the_floor_and_one(self):
        # The weakest candidate lands on MIN_NORMALIZED, not on 0.0, and the map stays
        # affine so the ORDERING and the relative spacing are untouched.
        #
        # Min-max maps a pool's minimum to exactly 0.0 by construction, which is
        # indistinguishable from "this signal never fired". That is not a cosmetic
        # distinction: a chunk the dense retriever ranked at 0.93 but that shares no word
        # with the query scores 0.0 on every other signal too, so it reached the user with a
        # reported score of 0.0000 beside the reason "dense retrieval".
        normalized = _normalize_scores({"low": 2.0, "mid": 4.0, "high": 6.0})

        assert normalized["low"] == pytest.approx(MIN_NORMALIZED)
        assert normalized["high"] == pytest.approx(1.0)
        assert normalized["mid"] == pytest.approx((MIN_NORMALIZED + 1.0) / 2)
        assert normalized["low"] < normalized["mid"] < normalized["high"]

    def test_a_signal_that_did_not_fire_still_maps_to_zero(self):
        # Zero keeps meaning zero. Only a candidate that actually scored is lifted off the
        # bottom — otherwise the floor would hand a signal that fired for nobody the same
        # weight as one that fired weakly, which is the distinction it exists to make.
        normalized = _normalize_scores({"absent": 0.0, "weak": 1.0, "strong": 9.0})

        assert normalized["absent"] == 0.0
        assert normalized["strong"] == pytest.approx(1.0)
        # The pool minimum here is the ABSENT candidate, so "weak" sits above the floor
        # rather than on it: MIN + (1 - MIN) * (1 - 0) / (9 - 0).
        assert normalized["weak"] == pytest.approx(MIN_NORMALIZED + (1 - MIN_NORMALIZED) / 9)
        assert normalized["absent"] < normalized["weak"] < normalized["strong"]


class TestKeywordScore:
    """Plain presence of the words the user actually typed."""

    def test_no_typed_terms_scores_zero(self):
        assert _keyword_score(set(), set(), {"pool": 3}) == 0.0

    def test_the_score_is_the_share_of_typed_terms_present(self):
        # Unweighted by frequency on purpose: this answers "does this passage contain the
        # words that were typed", which is a different question from BM25's.
        assert _keyword_score({"pool", "timeout"}, set(), {"pool": 9}) == pytest.approx(0.5)
        assert _keyword_score({"pool", "timeout"}, set(), {"pool": 1, "timeout": 1}) == 1.0

    def test_entities_add_a_bounded_bonus(self):
        score = _keyword_score({"pool", "timeout"}, {"database"}, {"pool": 1, "database": 1})

        assert score == pytest.approx(0.75)

    def test_the_bonus_cannot_push_the_score_above_one(self):
        # `keyword_weight` is applied to this number as if it were a 0..1 signal, so a value
        # above 1.0 would silently re-weight the whole fusion for one candidate.
        score = _keyword_score({"pool"}, {"database"}, {"pool": 1, "database": 1})

        assert score == 1.0


class TestNegativePenalty:
    """Multiplicative, capped, and off unless a deployment configures it."""

    CHUNK = {"pool": 2, "database": 1, "timeout": 1}

    def test_empty_negative_terms_is_a_no_op(self):
        # The default. A shipped list of "bad" words would be an unauditable opinion about
        # someone else's documents applied silently to every query they ask.
        config = PrecisionConfig().with_overrides({"negative_terms": (), "negative_penalty": 1.0})

        assert _negative_penalty(self.CHUNK, config) == 0.0

    def test_disabling_negative_signals_is_a_no_op(self):
        config = PrecisionConfig().with_overrides(
            {"negative_signals_enabled": False, "negative_terms": ("pool",), "negative_penalty": 1.0}
        )

        assert _negative_penalty(self.CHUNK, config) == 0.0

    def test_a_chunk_hitting_no_negative_term_is_unpenalised(self):
        config = PrecisionConfig().with_overrides({"negative_terms": ("bicycle", "saddle")})

        assert _negative_penalty(self.CHUNK, config) == 0.0

    def test_the_penalty_scales_with_the_share_of_terms_hit(self):
        config = PrecisionConfig().with_overrides(
            {"negative_terms": ("pool", "bicycle"), "negative_penalty": 0.4}
        )

        assert _negative_penalty(self.CHUNK, config) == pytest.approx(0.2)

    def test_the_penalty_is_capped_below_one(self):
        # The cap is what guarantees the multiplier `1 - penalty` stays positive, so a
        # penalised candidate is demoted rather than annihilated.
        config = PrecisionConfig().with_overrides(
            {"negative_terms": ("pool", "database"), "negative_penalty": 1.0}
        )

        assert _negative_penalty(self.CHUNK, config) == 0.95

    def test_the_worst_possible_penalty_cannot_produce_a_negative_score(self):
        # Subtracting a constant instead would make a weak match negative and a strong one
        # merely weaker, inverting the ordering the penalty was meant to adjust.
        outcome = _retrieve(negative_terms=("pool", "database"), negative_penalty=1.0)

        assert outcome.results
        assert all(result.score > 0 for result in outcome.results)
        assert all(result.retrieval_score > 0 for result in outcome.results)

    def test_negative_signals_off_reproduces_the_unpenalised_ranking_exactly(self):
        baseline = _retrieve()
        disabled = _retrieve(
            negative_signals_enabled=False, negative_terms=("pool", "database"), negative_penalty=1.0
        )
        no_terms = _retrieve(negative_terms=(), negative_penalty=1.0)

        assert disabled.chunk_ids == baseline.chunk_ids
        assert no_terms.chunk_ids == baseline.chunk_ids
        assert [r.score for r in disabled.results] == [r.score for r in baseline.results]
        assert [r.score for r in no_terms.results] == [r.score for r in baseline.results]


class TestCosine:
    def test_mismatched_lengths_score_zero(self):
        # Two embedding models produce vectors of different widths, and a cosine between
        # them is a number with no meaning that would nonetheless rank confidently.
        assert _cosine([1.0, 0.0, 0.0], [1.0, 0.0]) == 0.0

    def test_a_zero_vector_scores_zero(self):
        # Not a division by zero: the norm is the denominator.
        assert _cosine([0.0, 0.0], [1.0, 1.0]) == 0.0
        assert _cosine([1.0, 1.0], [0.0, 0.0]) == 0.0

    def test_missing_vectors_score_zero(self):
        assert _cosine(None, [1.0, 1.0]) == 0.0
        assert _cosine([1.0, 1.0], None) == 0.0
        assert _cosine([], []) == 0.0

    def test_identical_vectors_score_one(self):
        assert _cosine([0.3, 0.9, -0.2], [0.3, 0.9, -0.2]) == pytest.approx(1.0)

    def test_orthogonal_vectors_score_zero(self):
        assert _cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_opposed_vectors_are_clamped_to_zero(self):
        # MMR subtracts this as a redundancy, so a negative similarity would *reward*
        # opposition rather than merely not penalising it.
        assert _cosine([1.0, 0.0], [-1.0, 0.0]) == 0.0


class TestSimilarityFunction:
    TERM_SETS = {"a": {"pool", "timeout"}, "b": {"timeout", "socket"}, "c": {"chain", "spoke"}}

    def test_falls_back_to_jaccard_when_no_vector_accessor_is_supplied(self):
        # The default knowledge base in this application has no embeddings at all, and an
        # MMR that measured every pair as 0.0 similar would select by relevance alone while
        # reporting that it diversified.
        similarity = _similarity_function(self.TERM_SETS, None)

        assert similarity("a", "b") == pytest.approx(1 / 3)
        assert similarity("a", "c") == 0.0

    def test_falls_back_to_jaccard_when_the_accessor_returns_nothing(self):
        # A resource whose chunks were indexed lexically returns None per chunk rather than
        # raising, and that must land on the same fallback as having no accessor at all.
        similarity = _similarity_function(self.TERM_SETS, lambda chunk_id: None)

        assert similarity("a", "b") == pytest.approx(1 / 3)

    def test_uses_the_vectors_when_they_exist(self):
        # Proven by disagreement: "a" and "c" share no term, so any jaccard fallback scores
        # them 0.0 — only the cosine path can return 1.0 here.
        vectors = {"a": [1.0, 0.0], "c": [1.0, 0.0]}
        similarity = _similarity_function(self.TERM_SETS, vectors.get)

        assert similarity("a", "c") == pytest.approx(1.0)

    def test_an_unknown_chunk_id_is_simply_dissimilar(self):
        similarity = _similarity_function(self.TERM_SETS, None)

        assert similarity("a", "missing") == 0.0


class TestParentRecovery:
    def test_a_chunk_with_siblings_comes_back_with_its_parent_window(self):
        # The point of the stage: a child selected for a precise match arrives with the
        # surrounding text that makes it mean something.
        result = {r.chunk.id: r for r in _retrieve().results}["c1"]

        assert result.parent_chunk_ids == ("c2", "c3")
        assert result.parent_context is not None
        # The child's own text is in the window, and so is a sibling's.
        assert result.chunk.content.strip() in result.parent_context
        assert "Sizing Guidance" in result.parent_context

    def test_the_window_stops_at_the_document_boundary(self):
        # `parent_key` is derived from file_id, so stitching the tail of one document onto
        # the head of the next would fabricate context that exists in neither.
        result = {r.chunk.id: r for r in _retrieve().results}["c1"]

        assert "invoice" not in (result.parent_context or "").lower()

    def test_a_chunk_alone_in_its_window_reports_no_parent(self):
        # None says the child stands alone. An empty string would read downstream as "there
        # is a parent and it is blank".
        results = {r.chunk.id: r for r in _retrieve().results}

        assert results["c10"].parent_context is None   # the only chunk of its document
        assert results["c10"].parent_chunk_ids == ()
        assert results["c4"].parent_context is None    # opens a window of its own
        assert results["c4"].parent_chunk_ids == ()

    def test_disabling_parent_recovery_leaves_the_ranking_untouched(self):
        baseline = _retrieve()
        without = _retrieve(parent_child_enabled=False)

        assert without.chunk_ids == baseline.chunk_ids
        assert all(result.parent_context is None for result in without.results)
        assert all(result.parent_chunk_ids == () for result in without.results)

    def test_a_tight_budget_truncates_the_window_and_never_the_child(self):
        # The budget is characters, not siblings, because the point is "enough surrounding
        # text to preserve meaning" and three 200-character chunks are not the same amount
        # of context as three 1200-character ones. What it spends the budget on is the
        # siblings — the child's own text is always whole.
        generous = {r.chunk.id: r for r in _retrieve().results}["c1"]
        tight = {r.chunk.id: r for r in _retrieve(parent_max_chars=200).results}["c1"]

        assert tight.parent_context is not None
        assert CORPUS[0].content.strip() in tight.parent_context
        assert len(tight.parent_context) < len(generous.parent_context)
        assert tight.parent_chunk_ids == ("c2",)  # the budget ran out after one sibling

    def test_a_child_longer_than_the_budget_gets_no_parent(self):
        # `max(max_chars - len(child), 0)` is the guard. Without it a child that already
        # fills the budget would be handed a negative window, and the natural "fix" — pad
        # it anyway — truncates the passage that actually matched.
        outcome = _retrieve(parent_max_chars=100)  # below every chunk's own length

        assert outcome.results
        assert all(result.parent_context is None for result in outcome.results)
        assert all(result.parent_chunk_ids == () for result in outcome.results)


class TestPrecisionResultPayload:
    EXPECTED_KEYS = {
        "chunk_id", "document_id", "parent_chunk_id", "text", "parent_context",
        "parent_chunk_ids", "section", "heading", "page", "source_name",
        "retrieval_method", "scores", "metadata",
    }

    def test_to_dict_carries_the_full_documented_key_set(self):
        payload = _retrieve().results[0].to_dict()

        assert set(payload) == self.EXPECTED_KEYS
        assert set(payload["scores"]) == {"final", "retrieval", "bm25", "dense", "rerank"}
        assert set(payload["metadata"]) == {
            "document_id", "chunk_id", "parent_chunk_id", "section", "heading", "page",
            "document_type", "category", "version", "entities",
        }

    def test_the_payload_describes_the_child_chunk_that_matched(self):
        # The child stays the citation anchor even though a parent window travels with it,
        # so the evidence panel highlights the passage that actually matched (CLAUDE.md §20).
        payload = {r.chunk.id: r.to_dict() for r in _retrieve().results}["c1"]

        assert payload["chunk_id"] == "c1"
        assert payload["document_id"] == "file-1"
        assert payload["source_name"] == "pooling.txt"
        assert payload["text"] == CORPUS[0].content
        assert payload["page"] == 4
        assert payload["section"] == "3.1"
        assert payload["heading"] == "Connection Pooling"
        assert payload["parent_chunk_id"] == "file-1:0"  # the window, not a stored pointer
        assert payload["parent_chunk_ids"] == ["c2", "c3"]
        assert payload["retrieval_method"] == "bm25"

    def test_an_unresolved_structural_field_is_reported_as_null(self):
        # None means unknown and must stay unknown: inventing a section name is the same
        # class of mistake as defaulting an unknown page to page 1.
        payload = {r.chunk.id: r.to_dict() for r in _retrieve().results}["c5"]

        assert payload["page"] is None
        assert payload["section"] is None
        assert payload["heading"] is None

    def test_the_rerank_score_is_null_rather_than_zero_when_no_reranker_ran(self):
        # "The reranker did not see this candidate" and "the reranker scored it 0.0" are
        # different facts, and only the first is a reason to fall back to retrieval order.
        payload = _retrieve(reranker_enabled=False).results[0].to_dict()

        assert payload["scores"]["rerank"] is None

    def test_the_payload_is_json_serialisable(self):
        outcome = _retrieve(dense=[("c8", 0.93)])

        encoded = json.dumps([result.to_dict() for result in outcome.results])

        assert json.loads(encoded)[0]["chunk_id"] == outcome.results[0].chunk.id
