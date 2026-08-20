"""Unit tests for `rag/precision/bm25.py` — the ranking arithmetic the high-precision mode
stands on, and the one place where it differs *in kind* from the retrieval every other mode
in this repository uses.

Three properties here are worth the whole file, and every one of them fails silently:

* **IDF is never negative.** The unsmoothed Robertson/Sparck-Jones form goes below zero for
  a term appearing in more than half the corpus, which means a passage would be *penalised*
  for containing one of the user's own words. On a single-document knowledge base — the
  common case in this installation — that is most of the vocabulary, so the sign of this
  one number decides whether the mode ranks or anti-ranks.
* **Rare beats common.** `rag.service._score_chunk` weights a term by its raw frequency
  product, so a word in every chunk counts as much as one in three. BM25 weighting by
  inverse document frequency is the entire reason this module exists rather than the
  existing scorer being reused, and nothing else in the pipeline re-checks it.
* **Saturation and length normalisation.** `k1` says the tenth mention of a word is nearly
  worthless and `b` says a long passage is not a better one; both are pure arithmetic with
  no exception path, so a wrong constant produces a plausible ranking that is quietly worse.

Everything below is exercised against plain dicts. `bm25.py` imports nothing but `math`,
and this file keeps it that way: no database, no ORM, no `rag.service`.
"""
import math

import pytest

from rag.precision import bm25

pytestmark = pytest.mark.unit


def _score(
    query_weights: dict[str, float],
    chunk_terms: dict[str, int],
    chunk_length: int = 100,
    *,
    document_frequencies: dict[str, int],
    document_count: int = 100,
    average_length: float = 100.0,
    k1: float = 1.2,
    b: float = 0.75,
) -> float:
    """`bm25.score` with the corpus held still, so a test varies only what it names.

    Defaults describe a 100-chunk corpus averaging 100 terms per chunk, at the shipped
    `PrecisionConfig` values for k1 and b.
    """
    return bm25.score(
        query_weights,
        chunk_terms,
        chunk_length,
        document_count=document_count,
        document_frequencies=document_frequencies,
        average_length=average_length,
        k1=k1,
        b=b,
    )


class TestIdf:
    def test_a_term_in_every_document_is_not_negative(self):
        # A negative IDF would subtract from a passage for containing a word the user typed.
        # df == document_count is the worst case the smoothing has to survive.
        assert bm25.idf(10, 10) > 0.0
        assert bm25.idf(1, 1) > 0.0

    def test_a_single_document_corpus_still_produces_a_usable_weight(self):
        # Most bases here hold one document. The classic unsmoothed form is log(0.5/1.5)
        # at N=1, df=1 — negative — so this asserts the +1.0 form is the one in use.
        assert bm25.idf(1, 1) == pytest.approx(math.log(2.0 / 1.5))

    def test_never_negative_when_document_frequency_exceeds_the_corpus(self):
        # Inconsistent statistics (a stale frequency table, a corpus that shrank) must be
        # floored rather than inverting the ranking of whatever term carries them.
        assert bm25.idf(3, 10) == 0.0
        assert bm25.idf(0, 5) == 0.0

    def test_strictly_decreasing_as_document_frequency_rises(self):
        # The ordering IS the discrimination: if two document frequencies ever tie, two
        # terms of different rarity become interchangeable in the ranking.
        weights = [bm25.idf(20, df) for df in range(21)]
        assert all(earlier > later for earlier, later in zip(weights, weights[1:]))

    def test_an_unseen_term_gets_the_largest_weight(self):
        # df == 0 must not divide by zero, and must sit above every real frequency.
        assert bm25.idf(100, 0) == pytest.approx(math.log(101 / 0.5))
        assert bm25.idf(100, 0) > bm25.idf(100, 1)

    def test_zero_document_count_is_handled(self):
        # An empty index is asked for IDF before anything has been retrieved from it.
        assert bm25.idf(0, 0) == pytest.approx(math.log(2.0))

    def test_matches_the_smoothed_closed_form(self):
        # Pins the smoothing constants themselves: log((N + 1) / (df + 0.5)). Changing
        # either would move every score in the mode without failing anything else.
        for count, frequency in ((100, 1), (100, 37), (10, 3), (5, 5)):
            assert bm25.idf(count, frequency) == pytest.approx(
                math.log((count + 1) / (frequency + 0.5))
            )


class TestScore:
    def test_zero_for_an_empty_query(self):
        assert _score({}, {"apple": 3}, document_frequencies={"apple": 4}) == 0.0

    def test_zero_for_a_chunk_with_no_terms(self):
        # A chunk whose term map failed to load is scoreless, not an exception.
        assert _score({"apple": 1.0}, {}, document_frequencies={"apple": 4}) == 0.0

    def test_zero_when_nothing_overlaps(self):
        assert _score({"banana": 1.0}, {"apple": 3}, document_frequencies={"banana": 4}) == 0.0

    def test_a_term_recorded_with_zero_frequency_contributes_nothing(self):
        # `terms_json` can carry a 0 count; treating it as presence would credit a passage
        # for a word it does not contain.
        assert _score({"apple": 1.0}, {"apple": 0}, document_frequencies={"apple": 4}) == 0.0

    def test_a_zero_idf_term_contributes_nothing(self):
        # The floor in `idf` is only useful if `score` also declines to add a zero term.
        frequencies = {"stale": 50}
        assert _score({"stale": 1.0}, {"stale": 5}, document_count=3, document_frequencies=frequencies) == 0.0

    def test_a_rare_term_outranks_a_common_one_at_equal_frequency(self):
        # THE property that separates this module from `rag.service._score_chunk`, which
        # weights by raw frequency product and would score these two chunks identically.
        # Both chunks hold one term three times and are the same length; the only thing
        # telling them apart is how much of the corpus already contains that term.
        frequencies = {"valve": 1, "document": 100}
        rare = _score({"valve": 1.0}, {"valve": 3}, document_frequencies=frequencies)
        common = _score({"document": 1.0}, {"document": 3}, document_frequencies=frequencies)
        assert rare > common
        assert rare > 100 * common  # measured ~848x — this is a difference in kind, not a tilt

    def test_two_terms_of_equal_rarity_score_equally(self):
        # The companion to the test above: with document frequency held equal the scores
        # collapse together, which proves the gap there came from IDF and nothing else.
        frequencies = {"valve": 7, "document": 7}
        rare = _score({"valve": 1.0}, {"valve": 3}, document_frequencies=frequencies)
        common = _score({"document": 1.0}, {"document": 3}, document_frequencies=frequencies)
        assert rare == pytest.approx(common)

    def test_term_frequency_saturates(self):
        # Ten mentions is not ten times the evidence of one. At k1 = 1.2 with the chunk at
        # average length the ceiling for ANY frequency is k1 + 1 = 2.2x the single-mention
        # score, so a bound of 2.0 is a real assertion rather than a restatement.
        frequencies = {"valve": 5}
        once = _score({"valve": 1.0}, {"valve": 1}, document_frequencies=frequencies)
        ten_times = _score({"valve": 1.0}, {"valve": 10}, document_frequencies=frequencies)
        thousand_times = _score({"valve": 1.0}, {"valve": 1000}, document_frequencies=frequencies)
        assert once < ten_times < 2.0 * once
        assert thousand_times < 2.2 * once  # the k1 + 1 asymptote, never reached

    def test_b_zero_disables_length_normalisation(self):
        # b = 0 must make the length term vanish exactly, not merely shrink: this is the
        # knob a deployment turns when its chunks are all one size.
        frequencies = {"valve": 5}
        short = _score({"valve": 1.0}, {"valve": 2}, 10, document_frequencies=frequencies, b=0.0)
        long = _score({"valve": 1.0}, {"valve": 2}, 1000, document_frequencies=frequencies, b=0.0)
        assert short == long

    def test_b_one_applies_length_normalisation_fully(self):
        frequencies = {"valve": 5}
        short = _score({"valve": 1.0}, {"valve": 2}, 10, document_frequencies=frequencies, b=1.0)
        long = _score({"valve": 1.0}, {"valve": 2}, 1000, document_frequencies=frequencies, b=1.0)
        assert short > long

    def test_the_default_b_normalises_partially(self):
        # 0.75 is a blend, so a long chunk must land strictly between the two extremes.
        # Anything else means `b` is being read as a flag rather than as a coefficient.
        frequencies = {"valve": 5}
        kwargs = {"document_frequencies": frequencies}
        off = _score({"valve": 1.0}, {"valve": 2}, 1000, b=0.0, **kwargs)
        default = _score({"valve": 1.0}, {"valve": 2}, 1000, b=0.75, **kwargs)
        full = _score({"valve": 1.0}, {"valve": 2}, 1000, b=1.0, **kwargs)
        assert full < default < off

    def test_query_weights_scale_the_contribution_linearly(self):
        # Expansion weighting is a multiplier on the per-term contribution; if it were not
        # linear, `expansion_term_weight` would not mean what its docstring says it means.
        frequencies = {"valve": 5}
        full = _score({"valve": 1.0}, {"valve": 2}, document_frequencies=frequencies)
        weighted = _score({"valve": 0.45}, {"valve": 2}, document_frequencies=frequencies)
        assert weighted == pytest.approx(0.45 * full)

    def test_an_expansion_term_cannot_outweigh_a_typed_term(self):
        # Same document frequency, same frequency in the chunk, same length — the ONLY
        # difference is that one word was typed and the other was guessed. The guess must
        # lose, which is what keeps query expansion additive rather than substitutive.
        frequencies = {"typed": 5, "expanded": 5}
        chunk = {"typed": 2, "expanded": 2}
        typed = _score({"typed": 1.0}, chunk, document_frequencies=frequencies)
        expanded = _score({"expanded": 0.45}, chunk, document_frequencies=frequencies)
        assert expanded < typed
        assert expanded == pytest.approx(0.45 * typed)

    def test_contributions_are_additive_across_terms(self):
        # A query is the sum of its terms, so adding an expansion term can only ever raise
        # a passage that contains it — never rearrange the ones that do not.
        frequencies = {"typed": 5, "expanded": 40}
        chunk = {"typed": 2, "expanded": 6}
        typed = _score({"typed": 1.0}, chunk, document_frequencies=frequencies)
        expanded = _score({"expanded": 0.45}, chunk, document_frequencies=frequencies)
        both = _score({"typed": 1.0, "expanded": 0.45}, chunk, document_frequencies=frequencies)
        assert both == pytest.approx(typed + expanded)

    def test_average_length_of_zero_does_not_divide_by_zero(self):
        # An index built from an empty corpus reports average_length 0.0, and that reaches
        # here before any refusal does.
        frequencies = {"valve": 5}
        zero = _score({"valve": 1.0}, {"valve": 2}, document_frequencies=frequencies, average_length=0.0)
        one = _score({"valve": 1.0}, {"valve": 2}, document_frequencies=frequencies, average_length=1.0)
        assert zero > 0.0
        assert zero == pytest.approx(one)

    def test_negative_average_length_is_treated_as_one(self):
        frequencies = {"valve": 5}
        negative = _score({"valve": 1.0}, {"valve": 2}, document_frequencies=frequencies, average_length=-5.0)
        one = _score({"valve": 1.0}, {"valve": 2}, document_frequencies=frequencies, average_length=1.0)
        assert negative == pytest.approx(one)

    def test_a_chunk_of_zero_length_is_scored_not_skipped(self):
        # `lengths.get(chunk_id, 0)` is the pipeline's default for a chunk whose length was
        # never recorded; it must still rank rather than silently drop out of the results.
        assert _score({"valve": 1.0}, {"valve": 2}, 0, document_frequencies={"valve": 5}) > 0.0


class TestFeedbackTerms:
    def test_empty_when_the_limit_is_zero(self):
        # The ablation variants set prf_terms to 0; that has to mean "no feedback", not
        # "all of it".
        assert bm25.feedback_terms(
            [{"valve": 3, "pressure": 2}],
            document_count=100,
            document_frequencies={"valve": 4, "pressure": 4},
            exclude=set(),
            limit=0,
        ) == []

    def test_empty_when_there_are_no_passages(self):
        assert bm25.feedback_terms(
            [],
            document_count=100,
            document_frequencies={},
            exclude=set(),
            limit=5,
        ) == []

    def test_orders_by_score_descending(self):
        # "Distinctive" means frequent here and rare in the corpus: the rare term must lead
        # even though both appear the same number of times in the passage.
        ranked = bm25.feedback_terms(
            [{"valve": 3, "document": 3}],
            document_count=100,
            document_frequencies={"valve": 2, "document": 95},
            exclude=set(),
            limit=5,
        )
        assert [term for term, _ in ranked] == ["valve", "document"]
        assert ranked[0][1] > ranked[1][1]

    def test_a_tie_is_broken_by_the_term_itself(self):
        # Identical statistics, so the scores are equal to the last bit. Ordering by
        # anything else (insertion order, hash) would make two benchmark runs over the same
        # corpus expand the query differently and disagree on their own results.
        ranked = bm25.feedback_terms(
            [{"zebra": 2, "alpha": 2}],
            document_count=100,
            document_frequencies={"zebra": 5, "alpha": 5},
            exclude=set(),
            limit=5,
        )
        assert [term for term, _ in ranked] == ["alpha", "zebra"]
        assert ranked[0][1] == pytest.approx(ranked[1][1])

    def test_ordering_does_not_depend_on_dictionary_insertion_order(self):
        # The same corpus described in two orders is the same corpus. This is the property
        # `index.candidates_for` is careful about upstream, asserted here too because the
        # expansion it feeds is what the second BM25 pass ranks on.
        frequencies = {"valve": 2, "alpha": 5, "zebra": 5, "document": 95}
        forward = bm25.feedback_terms(
            [{"valve": 3, "alpha": 2, "zebra": 2, "document": 3}],
            document_count=100,
            document_frequencies=frequencies,
            exclude=set(),
            limit=5,
        )
        reversed_map = bm25.feedback_terms(
            [{"document": 3, "zebra": 2, "alpha": 2, "valve": 3}],
            document_count=100,
            document_frequencies=frequencies,
            exclude=set(),
            limit=5,
        )
        assert forward == reversed_map

    def test_honours_exclude(self):
        # The caller passes the terms already in the query: re-adding one would spend a
        # feedback slot restating something the ranking already weighted.
        ranked = bm25.feedback_terms(
            [{"valve": 3, "pressure": 3}],
            document_count=100,
            document_frequencies={"valve": 2, "pressure": 2},
            exclude={"valve"},
            limit=5,
        )
        assert [term for term, _ in ranked] == ["pressure"]

    def test_honours_the_limit(self):
        # `prf_terms` bounds how far the query can drift from what the user typed.
        ranked = bm25.feedback_terms(
            [{"alpha": 3, "bravo": 3, "charlie": 3, "delta": 3}],
            document_count=100,
            document_frequencies={"alpha": 2, "bravo": 3, "charlie": 4, "delta": 5},
            exclude=set(),
            limit=2,
        )
        assert len(ranked) == 2
        assert [term for term, _ in ranked] == ["alpha", "bravo"]

    def test_skips_terms_shorter_than_three_characters(self):
        # Two-character tokens survive tokenization but carry no meaning to expand with,
        # and a short token is exactly the kind that is frequent everywhere.
        ranked = bm25.feedback_terms(
            [{"ab": 9, "abc": 1}],
            document_count=100,
            document_frequencies={"ab": 2, "abc": 2},
            exclude=set(),
            limit=5,
        )
        assert [term for term, _ in ranked] == ["abc"]

    def test_skips_zero_idf_terms(self):
        # A term the floor in `idf` zeroed carries no discrimination, so expanding with it
        # would add cost to every later scoring pass for no change in the ranking.
        ranked = bm25.feedback_terms(
            [{"stale": 5, "good": 1}],
            document_count=3,
            document_frequencies={"stale": 10, "good": 1},
            exclude=set(),
            limit=5,
        )
        assert [term for term, _ in ranked] == ["good"]

    def test_accumulates_across_passages(self):
        # A term running through all of the top passages is more characteristic of the
        # result set than one concentrated in a single passage, even a denser one.
        maps = [
            {"shared": 1, "filler": 9},
            {"shared": 1, "filler": 9},
            {"shared": 1, "solo": 2, "filler": 7},
        ]
        ranked = bm25.feedback_terms(
            maps,
            document_count=100,
            document_frequencies={"shared": 5, "solo": 5, "filler": 5},
            exclude={"filler"},
            limit=5,
        )
        assert [term for term, _ in ranked] == ["shared", "solo"]

    def test_returns_the_score_alongside_the_term(self):
        # The caller weights each feedback term by `prf_term_weight`, so the pair shape is
        # part of the contract, not a debugging convenience.
        ranked = bm25.feedback_terms(
            [{"valve": 3}],
            document_count=100,
            document_frequencies={"valve": 2},
            exclude=set(),
            limit=5,
        )
        term, weight = ranked[0]
        assert term == "valve"
        assert weight == pytest.approx(bm25.idf(100, 2))  # frequency/total is 1.0 here
