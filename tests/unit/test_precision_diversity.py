"""Unit tests for `rag/precision/diversity.py` — stages 7 and 8, the only two stages of the
high-precision pipeline that *remove* things from an already-correct result set.

Three properties carry this file, and every one of them fails silently: the mode still
returns its five citations, they are just the wrong five.

* **The survivor of a duplicate pair is the better-ranked one.** Input order IS the ranking
  — `pipeline` hands dedup a list sorted by `-final_score` and nothing downstream re-sorts —
  so keeping the wrong member of a pair swaps a top passage for a near-copy that ranked
  lower, and the trace records a perfectly sensible-looking drop either way.
* **Containment, not Jaccard alone.** The chunker OVERLAPS by design (180 characters by
  default), so "a small chunk wholly inside a larger one" is the redundancy this stage
  actually meets — and Jaccard scores that pair around 0.3, nowhere near the shipped 0.90
  threshold. Containment is the only thing that sees it; lose it and dedup silently stops
  doing the job it exists for while still reporting a dedup stage in the trace.
* **MMR's first pick is the head of the list, whatever λ is.** With nothing selected there is
  nothing to be diverse from, so λ must never be able to trade away the best passage. λ is
  operator-tunable through `PRECISION_RAG_MMR_LAMBDA` across the whole 0..1 range, so this
  has to hold at both ends of it, not just at the default 0.7.

Everything below runs against a two-field stub candidate. `diversity.py` imports nothing but
`typing.Callable`, and this file keeps it that way: no database, no ORM, no `rag.service`,
not even `rag.precision.types`.
"""
from dataclasses import dataclass
from typing import Callable

import pytest

from rag.precision import diversity

pytestmark = pytest.mark.unit


@dataclass
class _Candidate:
    """The only two attributes `diversity.py` reads off a candidate.

    The real `types.Candidate` carries eleven more and requires a chunk object; the module
    under test is structurally typed, so duplicating this stub keeps these tests independent
    of every other stage's data shape.
    """

    chunk_id: str
    final_score: float = 0.0


def _terms(*words: str) -> set[str]:
    return set(words)


def _numbered(prefix: str, count: int, start: int = 0) -> set[str]:
    """A term set of a known size, for the size-asymmetry cases containment exists for."""
    return {f"{prefix}{n}" for n in range(start, start + count)}


def _jaccard_similarity(term_sets: dict[str, set[str]]) -> Callable[[str, str], float]:
    """The same fallback `pipeline._similarity_function` uses when a base has no vectors.

    MMR is handed a callable rather than term sets precisely so a vector-backed similarity
    can be substituted; testing through a callable keeps that seam honest.
    """

    def similarity(left_id: str, right_id: str) -> float:
        return diversity.jaccard(term_sets.get(left_id, set()), term_sets.get(right_id, set()))

    return similarity


class TestJaccard:
    def test_two_empty_sets_score_zero(self):
        # A chunk whose terms never made it into the corpus map arrives here as an empty
        # set; scoring it 1.0 against another empty one would drop it as a "duplicate".
        assert diversity.jaccard(set(), set()) == 0.0

    def test_one_empty_side_scores_zero(self):
        assert diversity.jaccard(set(), _terms("torque")) == 0.0
        assert diversity.jaccard(_terms("torque"), set()) == 0.0

    def test_disjoint_sets_score_zero(self):
        assert diversity.jaccard(_terms("valve", "clearance"), _terms("coolant", "gasket")) == 0.0

    def test_identical_sets_score_one(self):
        terms = _terms("valve", "clearance", "torque")
        assert diversity.jaccard(terms, set(terms)) == 1.0

    def test_partial_overlap_is_intersection_over_union(self):
        left = _terms("a", "b", "c")
        right = _terms("b", "c", "d")
        assert diversity.jaccard(left, right) == pytest.approx(2 / 4)

    def test_is_symmetric(self):
        left = _terms("a", "b", "c")
        right = _terms("c", "d")
        assert diversity.jaccard(left, right) == diversity.jaccard(right, left)

    def test_a_small_set_inside_a_much_larger_one_scores_low(self):
        # The motivating case, from Jaccard's side: 120 terms wholly inside 400 is 100%
        # redundant evidence, and 0.3 is far below the 0.90 threshold dedup ships with — so
        # Jaccard alone would spend two of five citation slots on the same passage.
        small = _numbered("t", 120)
        large = _numbered("t", 400)
        assert small < large
        assert diversity.jaccard(small, large) == pytest.approx(0.3)


class TestContainment:
    def test_two_empty_sets_score_zero(self):
        # min(0, 0) would be a ZeroDivisionError; the guard is what keeps an unindexed chunk
        # from raising out of the middle of a retrieval.
        assert diversity.containment(set(), set()) == 0.0

    def test_one_empty_side_scores_zero(self):
        assert diversity.containment(set(), _terms("torque")) == 0.0
        assert diversity.containment(_terms("torque"), set()) == 0.0

    def test_disjoint_sets_score_zero(self):
        assert diversity.containment(_terms("valve", "clearance"), _terms("coolant")) == 0.0

    def test_identical_sets_score_one(self):
        terms = _terms("valve", "clearance", "torque")
        assert diversity.containment(terms, set(terms)) == 1.0

    def test_a_small_set_fully_inside_a_much_larger_one_scores_one(self):
        # The same pair as the Jaccard case above, and the whole reason this function
        # exists: 0.3 by Jaccard, 1.0 here. Dedup takes the max of the two, so this is the
        # number that decides the pair.
        small = _numbered("t", 120)
        large = _numbered("t", 400)
        assert diversity.containment(small, large) == 1.0
        assert diversity.jaccard(small, large) < diversity.containment(small, large)

    def test_is_symmetric(self):
        # It divides by the SMALLER set, not by the left one, so argument order cannot
        # change the verdict — dedup calls it with (candidate, survivor) only.
        small = _numbered("t", 3)
        large = _numbered("t", 9)
        assert diversity.containment(small, large) == diversity.containment(large, small)

    def test_partial_containment_is_share_of_the_smaller_set(self):
        small = _terms("a", "b", "c", "d")
        large = _terms("a", "b", "z1", "z2", "z3", "z4", "z5", "z6")
        assert diversity.containment(small, large) == pytest.approx(0.5)


class TestDeduplicate:
    def test_higher_ranked_member_of_a_duplicate_pair_survives(self):
        top = _Candidate("top", 0.90)
        near_copy = _Candidate("near_copy", 0.80)
        term_sets = {"top": _terms("a", "b", "c", "d"), "near_copy": _terms("a", "b", "c", "d")}

        kept, decisions = diversity.deduplicate([top, near_copy], term_sets, 0.90)

        assert [c.chunk_id for c in kept] == ["top"]
        assert [d["dropped"] for d in decisions] == ["near_copy"]

    def test_reversing_the_input_reverses_which_one_survives(self):
        # Input order is the ONLY ranking signal this stage has — it reads no score at all —
        # so the pair's verdict has to flip with the order, and the pipeline is responsible
        # for handing it a list sorted by -final_score.
        first = _Candidate("first", 0.10)
        second = _Candidate("second", 0.99)
        term_sets = {"first": _terms("a", "b", "c"), "second": _terms("a", "b", "c")}

        kept, _ = diversity.deduplicate([first, second], term_sets, 0.90)
        assert [c.chunk_id for c in kept] == ["first"]

        reversed_kept, _ = diversity.deduplicate([second, first], term_sets, 0.90)
        assert [c.chunk_id for c in reversed_kept] == ["second"]

    def test_decision_names_dropped_duplicate_of_and_similarity(self):
        # The trace has to answer "why is chunk X missing" without re-running retrieval, so
        # a decision that names only the victim is not enough — it needs the pair and the
        # number that condemned it.
        keeper = _Candidate("keeper", 0.90)
        dropped = _Candidate("dropped", 0.80)
        term_sets = {"keeper": _terms("a", "b", "c", "d"), "dropped": _terms("a", "b", "c", "d")}

        _, decisions = diversity.deduplicate([keeper, dropped], term_sets, 0.90)

        assert decisions == [{"dropped": "dropped", "duplicate_of": "keeper", "similarity": 1.0}]

    def test_similarity_in_the_decision_is_rounded_to_four_places(self):
        # Traces are logged and pasted into bug reports; a raw float here is 17 digits of
        # noise around a number nobody reads past the second decimal.
        keeper = _Candidate("keeper", 0.90)
        dropped = _Candidate("dropped", 0.80)
        # 1/5 by Jaccard, 1/3 by containment — dedup takes the max, so 0.3333...
        term_sets = {"keeper": _terms("p", "q", "r"), "dropped": _terms("p", "x", "y")}

        _, decisions = diversity.deduplicate([keeper, dropped], term_sets, 0.30)

        assert decisions[0]["similarity"] == 0.3333

    def test_a_threshold_of_zero_disables_dropping_entirely(self):
        # `dedup_threshold` is operator-tunable down to 0.0, and at 0.0 every pair sharing a
        # single stopword would score above it — so 0 has to mean "off", not "drop almost
        # everything".
        candidates = [_Candidate("a", 0.9), _Candidate("b", 0.8), _Candidate("c", 0.7)]
        identical = _terms("a", "b", "c")
        term_sets = {"a": set(identical), "b": set(identical), "c": set(identical)}

        kept, decisions = diversity.deduplicate(candidates, term_sets, 0.0)

        assert [c.chunk_id for c in kept] == ["a", "b", "c"]
        assert decisions == []

    def test_a_disabled_dedup_returns_a_copy_not_the_caller_s_list(self):
        # The pipeline goes on to slice and reorder this list; aliasing the caller's input
        # would let stage 8 mutate stage 6's record of what it produced.
        candidates = [_Candidate("a", 0.9), _Candidate("b", 0.8)]

        kept, _ = diversity.deduplicate(candidates, {}, 0.0)

        assert kept == candidates
        assert kept is not candidates

    def test_nothing_similar_means_nothing_dropped(self):
        candidates = [_Candidate("a", 0.9), _Candidate("b", 0.8), _Candidate("c", 0.7)]
        term_sets = {
            "a": _terms("valve", "clearance"),
            "b": _terms("coolant", "hose"),
            "c": _terms("brake", "pad"),
        }

        kept, decisions = diversity.deduplicate(candidates, term_sets, 0.90)

        assert [c.chunk_id for c in kept] == ["a", "b", "c"]
        assert decisions == []

    def test_a_chunk_swallowed_by_a_larger_one_is_dropped_on_containment(self):
        # The chunker's own failure mode, end to end: this pair is 0.3 by Jaccard and would
        # survive a Jaccard-only threshold, which is the regression this asserts against.
        large = _Candidate("large", 0.90)
        small = _Candidate("small", 0.80)
        term_sets = {"large": _numbered("t", 400), "small": _numbered("t", 120)}
        assert diversity.jaccard(term_sets["large"], term_sets["small"]) < 0.90

        kept, decisions = diversity.deduplicate([large, small], term_sets, 0.90)

        assert [c.chunk_id for c in kept] == ["large"]
        assert decisions[0]["similarity"] == 1.0

    def test_a_candidate_with_no_term_set_is_never_dropped(self):
        # A missing entry reads as an empty set, and both metrics floor an empty side at
        # 0.0 — so a chunk the corpus map lost is kept and stays inspectable, rather than
        # being silently deduped against everything.
        known = _Candidate("known", 0.90)
        unmapped = _Candidate("unmapped", 0.80)
        term_sets = {"known": _terms("a", "b", "c")}

        kept, decisions = diversity.deduplicate([known, unmapped], term_sets, 0.90)

        assert [c.chunk_id for c in kept] == ["known", "unmapped"]
        assert decisions == []

    def test_similarity_is_measured_against_survivors_only(self):
        # `b` is dropped against `a`; `c` overlaps `b` by exactly as much but shares nothing
        # with `a`, so it has to survive. Comparing against already-dropped candidates would
        # cascade one duplicate into removing an independent passage — and the trace would
        # blame a chunk that is itself missing from the results.
        candidates = [_Candidate("a", 0.9), _Candidate("b", 0.8), _Candidate("c", 0.7)]
        term_sets = {
            "a": _terms("p1", "p2", "p3", "p4"),
            "b": _terms("p1", "p2", "q1", "q2"),
            "c": _terms("q1", "q2", "q3", "q4"),
        }
        assert diversity.containment(term_sets["b"], term_sets["c"]) == 0.5
        assert diversity.containment(term_sets["a"], term_sets["c"]) == 0.0

        kept, decisions = diversity.deduplicate(candidates, term_sets, 0.5)

        assert [c.chunk_id for c in kept] == ["a", "c"]
        assert [d["duplicate_of"] for d in decisions] == ["a"]

    def test_an_empty_candidate_list_produces_nothing(self):
        kept, decisions = diversity.deduplicate([], {}, 0.90)
        assert kept == []
        assert decisions == []

    def test_similarity_exactly_at_the_threshold_drops(self):
        # The comparison is `>=`, so the configured threshold is inclusive. A pair landing
        # exactly on 0.5 is a coin-flip that has to be decided the same way every run.
        keeper = _Candidate("keeper", 0.9)
        other = _Candidate("other", 0.8)
        term_sets = {"keeper": _terms("a", "b", "c", "d"), "other": _terms("a", "b", "z", "w")}
        assert diversity.containment(term_sets["keeper"], term_sets["other"]) == 0.5

        kept, decisions = diversity.deduplicate([keeper, other], term_sets, 0.5)

        assert [c.chunk_id for c in kept] == ["keeper"]
        assert decisions[0]["similarity"] == 0.5


class TestMaximalMarginalRelevance:
    def test_a_limit_of_zero_gives_two_empty_lists(self):
        candidates = [_Candidate("a", 1.0), _Candidate("b", 0.5)]

        selected, decisions = diversity.maximal_marginal_relevance(
            candidates, _jaccard_similarity({}), lambda_=0.7, limit=0
        )

        assert selected == []
        assert decisions == []

    def test_an_empty_candidate_list_gives_two_empty_lists(self):
        selected, decisions = diversity.maximal_marginal_relevance(
            [], _jaccard_similarity({}), lambda_=0.7, limit=5
        )

        assert selected == []
        assert decisions == []

    def test_lambda_one_is_pure_relevance_and_reports_every_selection(self):
        # λ=1.0 is the escape hatch for an operator who wants MMR configured on but neutral:
        # no redundancy term, just the highest scores in order.
        candidates = [_Candidate("a", 1.0), _Candidate("b", 0.9), _Candidate("c", 0.5)]
        term_sets = {"a": _terms("x"), "b": _terms("x"), "c": _terms("y")}

        selected, decisions = diversity.maximal_marginal_relevance(
            candidates, _jaccard_similarity(term_sets), lambda_=1.0, limit=2
        )

        assert [c.chunk_id for c in selected] == ["a", "b"]
        # One decision per selected item, exactly as every other λ produces. It used to
        # return none: the λ>=1.0 path short-circuited before the loop that records them, so
        # a trace read at λ=1.0 showed selections that no decision explained — and the reader
        # had to know which λ was in force to know the gap was expected rather than a bug.
        assert [d["selected"] for d in decisions] == ["a", "b"]
        assert all(d["redundancy"] == 0.0 for d in decisions)

    def test_lambda_one_never_calls_the_similarity_function(self):
        def exploding_similarity(left_id: str, right_id: str) -> float:
            raise AssertionError("λ=1.0 must not measure redundancy it is going to ignore")

        candidates = [_Candidate("a", 1.0), _Candidate("b", 0.9)]

        selected, _ = diversity.maximal_marginal_relevance(
            candidates, exploding_similarity, lambda_=1.0, limit=2
        )

        assert [c.chunk_id for c in selected] == ["a", "b"]

    def test_lambda_zero_maximises_diversity(self):
        # `b` is the second-best passage and a near-copy of the best; `c` is much weaker but
        # says something else. At λ=0 relevance is worth nothing after the first pick, so
        # the complementary passage has to win.
        candidates = [_Candidate("a", 1.0), _Candidate("b", 0.9), _Candidate("c", 0.5)]
        term_sets = {
            "a": _numbered("t", 10),
            "b": _numbered("t", 9) | {"x"},
            "c": _numbered("z", 10),
        }

        selected, _ = diversity.maximal_marginal_relevance(
            candidates, _jaccard_similarity(term_sets), lambda_=0.0, limit=2
        )

        assert [c.chunk_id for c in selected] == ["a", "c"]

    def test_a_high_lambda_keeps_the_near_duplicate(self):
        # The contrast case for the one above, on the identical pool: at the shipped λ=0.7
        # relevance still outweighs an 0.82 redundancy, which is exactly why dedup runs
        # first rather than MMR being asked to do both jobs.
        candidates = [_Candidate("a", 1.0), _Candidate("b", 0.9), _Candidate("c", 0.5)]
        term_sets = {
            "a": _numbered("t", 10),
            "b": _numbered("t", 9) | {"x"},
            "c": _numbered("z", 10),
        }

        selected, _ = diversity.maximal_marginal_relevance(
            candidates, _jaccard_similarity(term_sets), lambda_=0.7, limit=2
        )

        assert [c.chunk_id for c in selected] == ["a", "b"]

    @pytest.mark.parametrize("lambda_", [0.0, 0.25, 0.5, 0.75, 1.0])
    def test_the_first_pick_is_the_top_relevance_item_at_every_lambda(self, lambda_):
        # With nothing selected there is nothing to be diverse from, so no value of λ may
        # trade away the best passage. λ is env-tunable across this whole range.
        candidates = [_Candidate("best", 1.0), _Candidate("mid", 0.6), _Candidate("low", 0.2)]
        term_sets = {
            "best": _numbered("t", 10),
            "mid": _numbered("t", 10),
            "low": _numbered("z", 10),
        }

        selected, _ = diversity.maximal_marginal_relevance(
            candidates, _jaccard_similarity(term_sets), lambda_=lambda_, limit=3
        )

        assert selected[0].chunk_id == "best"
        assert selected[0].final_score == max(c.final_score for c in candidates)

    def test_an_all_equal_score_pool_does_not_crash(self):
        # Min-max normalisation divides by the span; an identically-scored pool spans zero,
        # which is not exotic — it is what a one-chunk-per-page document looks like after
        # the reranker flattens everything.
        candidates = [_Candidate("a", 0.42), _Candidate("b", 0.42), _Candidate("c", 0.42)]
        term_sets = {"a": _terms("p"), "b": _terms("q"), "c": _terms("r")}

        selected, decisions = diversity.maximal_marginal_relevance(
            candidates, _jaccard_similarity(term_sets), lambda_=0.7, limit=3
        )

        assert [c.chunk_id for c in selected] == ["a", "b", "c"]
        # Every relevance collapses to the same 0.5, so the first pick's recorded MMR is it.
        assert decisions[0]["mmr"] == 0.5

    def test_selection_is_invariant_to_the_scale_of_the_scores(self):
        # The point of normalising: λ has to mean the same thing on a corpus whose scores
        # run 0..1 and one whose scores run 0..1000, or the default 0.7 quietly becomes
        # "ignore diversity" on half the knowledge bases in the installation.
        term_sets = {
            "a": _numbered("t", 10),
            "b": _numbered("t", 9) | {"x"},
            "c": _numbered("z", 10),
        }
        small = [_Candidate("a", 1.0), _Candidate("b", 0.9), _Candidate("c", 0.5)]
        large = [_Candidate("a", 1000.0), _Candidate("b", 900.0), _Candidate("c", 500.0)]

        small_selected, small_decisions = diversity.maximal_marginal_relevance(
            small, _jaccard_similarity(term_sets), lambda_=0.7, limit=3
        )
        large_selected, large_decisions = diversity.maximal_marginal_relevance(
            large, _jaccard_similarity(term_sets), lambda_=0.7, limit=3
        )

        assert [c.chunk_id for c in small_selected] == [c.chunk_id for c in large_selected]
        assert small_decisions == large_decisions

    def test_one_decision_per_selected_item_in_selection_order(self):
        # The trace reads these positionally against the final chunk ids; a decision list
        # that skipped the first pick would misattribute every score by one.
        candidates = [
            _Candidate("a", 1.0),
            _Candidate("b", 0.8),
            _Candidate("c", 0.6),
            _Candidate("d", 0.4),
        ]
        term_sets = {cid: _terms(cid) for cid in ("a", "b", "c", "d")}

        selected, decisions = diversity.maximal_marginal_relevance(
            candidates, _jaccard_similarity(term_sets), lambda_=0.5, limit=3
        )

        assert len(selected) == 3
        assert [d["selected"] for d in decisions] == [c.chunk_id for c in selected]
        assert all(set(d) == {"selected", "mmr", "redundancy"} for d in decisions)

    def test_the_first_decision_records_zero_redundancy(self):
        # There is nothing selected to be redundant against, and reporting a non-zero here
        # would read as "the best passage duplicated something" in the trace.
        candidates = [_Candidate("a", 1.0), _Candidate("b", 0.5)]
        term_sets = {"a": _terms("x"), "b": _terms("x")}

        _, decisions = diversity.maximal_marginal_relevance(
            candidates, _jaccard_similarity(term_sets), lambda_=0.7, limit=2
        )

        assert decisions[0] == {"selected": "a", "mmr": 1.0, "redundancy": 0.0}

    def test_recorded_redundancy_is_the_max_similarity_to_the_already_selected(self):
        # MMR penalises by the WORST overlap, not the average: a passage that duplicates one
        # already-chosen chunk is redundant however unlike the rest of the set it is.
        candidates = [_Candidate("a", 1.0), _Candidate("b", 0.9), _Candidate("c", 0.8)]
        term_sets = {
            "a": _terms("p1", "p2", "p3", "p4"),
            "b": _terms("q1", "q2", "q3", "q4"),
            "c": _terms("q1", "q2", "q3", "q4"),
        }

        selected, decisions = diversity.maximal_marginal_relevance(
            candidates, _jaccard_similarity(term_sets), lambda_=0.7, limit=3
        )

        assert [c.chunk_id for c in selected] == ["a", "b", "c"]
        # c is identical to b and disjoint from a — max(1.0, 0.0), never the mean.
        assert decisions[2]["redundancy"] == 1.0

    def test_the_limit_caps_the_selection(self):
        candidates = [_Candidate(chr(ord("a") + n), 1.0 - n / 10) for n in range(6)]
        term_sets = {c.chunk_id: _terms(c.chunk_id) for c in candidates}

        selected, decisions = diversity.maximal_marginal_relevance(
            candidates, _jaccard_similarity(term_sets), lambda_=0.7, limit=3
        )

        assert len(selected) == 3
        assert len(decisions) == 3

    def test_a_limit_beyond_the_pool_returns_the_whole_pool_exactly_once(self):
        # `final_k` is a fixed 5 while the pool after dedup can be smaller; selecting a
        # chunk twice would put the same citation in two slots.
        candidates = [_Candidate("a", 1.0), _Candidate("b", 0.7), _Candidate("c", 0.3)]
        term_sets = {"a": _terms("x"), "b": _terms("y"), "c": _terms("z")}

        selected, _ = diversity.maximal_marginal_relevance(
            candidates, _jaccard_similarity(term_sets), lambda_=0.7, limit=99
        )

        assert sorted(c.chunk_id for c in selected) == ["a", "b", "c"]
        assert len({c.chunk_id for c in selected}) == len(selected)

    def test_a_single_candidate_is_selected_without_consulting_similarity(self):
        def exploding_similarity(left_id: str, right_id: str) -> float:
            raise AssertionError("a lone candidate has nothing to be diverse from")

        selected, decisions = diversity.maximal_marginal_relevance(
            [_Candidate("only", 0.4)], exploding_similarity, lambda_=0.0, limit=5
        )

        assert [c.chunk_id for c in selected] == ["only"]
        assert decisions == [{"selected": "only", "mmr": 0.5, "redundancy": 0.0}]

    def test_the_caller_s_list_is_not_mutated(self):
        # `pipeline` records `count_in` from the same list it passes; MMR pops from a copy,
        # so a stage that reported "8 in, 5 out" cannot become "5 in, 5 out".
        candidates = [_Candidate("a", 1.0), _Candidate("b", 0.8), _Candidate("c", 0.6)]
        term_sets = {"a": _terms("x"), "b": _terms("y"), "c": _terms("z")}

        diversity.maximal_marginal_relevance(
            candidates, _jaccard_similarity(term_sets), lambda_=0.7, limit=2
        )

        assert [c.chunk_id for c in candidates] == ["a", "b", "c"]

    def test_ties_are_broken_by_relevance_order(self):
        # Three mutually-unrelated passages score an identical MMR value; `>` keeps the
        # earliest, so the pipeline's existing ranking survives instead of the result set
        # reordering itself differently on every run.
        candidates = [_Candidate("a", 1.0), _Candidate("b", 0.5), _Candidate("c", 0.5)]
        term_sets = {"a": _terms("p"), "b": _terms("q"), "c": _terms("r")}

        selected, _ = diversity.maximal_marginal_relevance(
            candidates, _jaccard_similarity(term_sets), lambda_=0.7, limit=3
        )

        assert [c.chunk_id for c in selected] == ["a", "b", "c"]

    def test_the_mmr_value_in_a_decision_is_rounded_to_four_places(self):
        candidates = [_Candidate("a", 1.0), _Candidate("b", 0.9), _Candidate("c", 0.5)]
        term_sets = {
            "a": _numbered("t", 10),
            "b": _numbered("t", 9) | {"x"},
            "c": _numbered("z", 10),
        }

        _, decisions = diversity.maximal_marginal_relevance(
            candidates, _jaccard_similarity(term_sets), lambda_=0.7, limit=2
        )

        # 0.7 * 0.8 - 0.3 * (9/11) = 0.3145454..., which must not reach a log line at
        # seventeen significant figures.
        assert decisions[1]["mmr"] == 0.3145
