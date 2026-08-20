"""Unit tests for `rag/precision/evaluation.py` — the harness that decides which stage of
the high-precision pipeline keeps its place.

This file matters more than a metrics module usually would, because these numbers are the
only evidence the mode's design is built on: every entry in `VARIANTS` is a hypothesis, and
`compare()` is what turns it into a measurement. A metric that is quietly wrong does not
raise, does not look wrong, and produces an ablation table that argues confidently for
deleting the stage that was actually helping.

Four properties are worth the whole file, and every one of them fails silently:

* **The denominators.** `recall_at_k` divides by the count of RELEVANT items and
  `precision_at_k` by a flat K. Those are different choices with different reasons — the
  flat K is deliberate and documented (a `min(k, len(ranked))` denominator scored the `bm25`
  ablation *above* `full` purely for returning a shorter list) — and swapping either one
  changes the ranking of variants without changing anything that reads as a metric bug.
* **Document ids are deduplicated in rank order.** Ten chunks from one file are one document
  at rank 1. Counting them as ten makes document-level precision a measurement of how
  repetitive the chunker is, which is a number that would rise every time retrieval got
  worse at spreading across sources.
* **Averaging is per ground-truth type.** A benchmark mixing chunk-labelled and
  document-labelled cases must not divide chunk recall by the cases that never carried a
  chunk label — that reports a pipeline as half as good as it is, in proportion to how many
  lenient cases someone added.
* **Each variant is run with its OWN overrides.** `compare` builds one lambda per variant in
  a loop; binding the loop variable late instead of by default argument would run every
  variant with the last one's configuration and produce five identical rows that look like a
  pipeline with no working stages.

Everything here is arithmetic over plain lists. `evaluation.py` imports nothing but
`dataclasses`, `json`, `math` and `typing`, and this file keeps it that way: no database, no
ORM, no `rag.service`, and the runner is always a local stub.
"""
import json
import math

import pytest

from rag.precision import evaluation
from rag.precision.evaluation import (
    VARIANTS,
    BenchmarkCase,
    compare,
    evaluate,
    format_table,
    hit_rate,
    load_cases,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

pytestmark = pytest.mark.unit


def _case(query="q", chunks=(), documents=()) -> BenchmarkCase:
    return BenchmarkCase(query=query, expected_chunks=tuple(chunks), expected_documents=tuple(documents))


def _runner(table):
    """A stub `run(case)` answering from a {query: (chunks, documents, elapsed_ms)} table."""

    def run(case):
        return table[case.query]

    return run


class TestRecallAtK:
    def test_denominator_is_the_relevant_count_not_k(self):
        # Two relevant, one of them inside the window: 1 found / 2 relevant = 0.5.
        # A K denominator would say 1/5 = 0.2 and call a half-solved query nearly a miss.
        assert recall_at_k(["a", "x", "y"], ["a", "b"], 5) == 0.5

    def test_full_recall_when_every_relevant_item_is_inside_the_window(self):
        assert recall_at_k(["a", "z", "b"], ["a", "b"], 5) == 1.0

    def test_only_the_top_k_counts(self):
        # "b" sits at rank 3, outside K=2, so it must not be credited: the cut-off is the
        # entire point of an @K metric.
        assert recall_at_k(["a", "z", "b"], ["a", "b"], 2) == 0.5

    def test_zero_when_nothing_relevant_was_retrieved(self):
        assert recall_at_k(["x", "y"], ["a"], 5) == 0.0

    def test_zero_when_the_case_has_no_ground_truth(self):
        # Guards the division: an unlabelled case is not a perfect score.
        assert recall_at_k(["a"], [], 5) == 0.0

    def test_duplicate_ground_truth_labels_do_not_inflate_the_denominator(self):
        # `relevant` is set-ified, so a benchmark file listing the same chunk twice by
        # accident cannot halve every recall number it appears in.
        assert recall_at_k(["a"], ["a", "a"], 5) == 1.0


class TestPrecisionAtK:
    def test_flat_k_denominator(self):
        # One relevant item found at rank 1, K=5: 1/5 = 0.2 exactly. This pins the choice
        # of denominator — `min(k, len(ranked))` would return 1.0 here.
        assert precision_at_k(["a"], ["a"], 5) == pytest.approx(0.2)

    def test_a_short_list_is_not_rewarded_for_being_short(self):
        # The property the docstring was written for: a variant returning one correct hit
        # must not out-score a variant that returns the same hit plus four more candidates.
        # Both score 0.2, so the ablation compares ranking rather than list length.
        short = precision_at_k(["a"], ["a"], 5)
        long = precision_at_k(["a", "w", "x", "y", "z"], ["a"], 5)
        assert short == long == pytest.approx(0.2)

    def test_counts_every_relevant_item_in_the_window(self):
        # 2 of the top 5 are relevant: 2/5 = 0.4.
        assert precision_at_k(["a", "x", "b", "y", "z"], ["a", "b"], 5) == pytest.approx(0.4)

    def test_zero_for_an_empty_ranking(self):
        assert precision_at_k([], ["a"], 5) == 0.0

    def test_zero_for_a_non_positive_k(self):
        # Guards the division by K.
        assert precision_at_k(["a"], ["a"], 0) == 0.0


class TestReciprocalRank:
    def test_first_hit_only(self):
        # Rank 3 -> 1/3. The second relevant item at rank 4 must not change the answer;
        # MRR is a statement about how far the reader has to look, not about coverage.
        assert reciprocal_rank(["x", "y", "a", "b"], ["a", "b"]) == pytest.approx(1 / 3)

    def test_rank_one_is_one(self):
        assert reciprocal_rank(["a", "b"], ["a"]) == 1.0

    def test_zero_when_the_relevant_item_is_absent(self):
        assert reciprocal_rank(["x", "y"], ["a"]) == 0.0

    def test_zero_when_the_case_has_no_ground_truth(self):
        assert reciprocal_rank(["a"], []) == 0.0

    def test_is_not_truncated_at_any_k(self):
        # `reciprocal_rank` takes no K on purpose: a hit at rank 20 is a poor result, not
        # the same result as no hit at all, and the @K metrics already report the cut-off.
        assert reciprocal_rank(["x"] * 19 + ["a"], ["a"]) == pytest.approx(1 / 20)


class TestHitRate:
    def test_one_when_anything_relevant_is_inside_the_window(self):
        assert hit_rate(["x", "a"], ["a"], 5) == 1.0

    def test_zero_when_the_only_hit_is_outside_the_window(self):
        # Same ranking as above, K=1: the hit at rank 2 is not visible to the reader.
        assert hit_rate(["x", "a"], ["a"], 1) == 0.0

    def test_zero_when_nothing_relevant_was_retrieved(self):
        assert hit_rate(["x", "y"], ["a"], 5) == 0.0

    def test_is_binary_regardless_of_how_many_hits_land(self):
        # Two hits are still one hit rate; the metric answers "did the user get anything?".
        assert hit_rate(["a", "b"], ["a", "b"], 5) == 1.0


class TestNdcgAtK:
    def test_single_relevant_item_at_rank_two(self):
        # gain  = 1/log2(2+1) = 1/log2(3) = 0.630929...
        # ideal = 1/log2(1+1) = 1.0        (one relevant item, so the ideal is rank 1)
        # nDCG  = 0.630929...
        assert ndcg_at_k(["x", "a"], ["a"], 5) == pytest.approx(1 / math.log2(3))
        assert ndcg_at_k(["x", "a"], ["a"], 5) == pytest.approx(0.6309, abs=1e-4)

    def test_perfect_ranking_is_one(self):
        # gain = 1/log2(2) + 1/log2(3), ideal = the same two positions.
        assert ndcg_at_k(["a", "b", "x"], ["a", "b"], 5) == pytest.approx(1.0)

    def test_discounts_by_position(self):
        # Two relevant at ranks 1 and 3: gain = 1 + 1/log2(4) = 1.5,
        # ideal = 1 + 1/log2(3) = 1.630929..., nDCG = 0.919721...
        expected = (1.0 + 1 / math.log2(4)) / (1.0 + 1 / math.log2(3))
        assert ndcg_at_k(["a", "x", "b"], ["a", "b"], 5) == pytest.approx(expected)
        assert ndcg_at_k(["a", "x", "b"], ["a", "b"], 5) == pytest.approx(0.9197, abs=1e-4)

    def test_ideal_is_capped_at_k(self):
        # Three relevant but K=1, so the best achievable is one hit at rank 1. Without the
        # min(len(relevant), k) cap the ideal would be unreachable and a perfect top-1
        # would score 0.39 — a variant punished for the size of the window it was measured in.
        assert ndcg_at_k(["a", "b", "c"], ["a", "b", "c"], 1) == pytest.approx(1.0)

    def test_zero_when_the_case_has_no_ground_truth(self):
        assert ndcg_at_k(["a"], [], 5) == 0.0

    def test_zero_when_nothing_relevant_was_retrieved(self):
        assert ndcg_at_k(["x", "y"], ["a"], 5) == 0.0


class TestBenchmarkCase:
    def test_from_dict_reads_every_field(self):
        case = BenchmarkCase.from_dict(
            {
                "query": "what is the warranty period",
                "expected_documents": ["file-1"],
                "expected_chunks": ["chunk-9"],
                "resource_id": "res-1",
                "notes": "from the FAQ",
            }
        )
        assert case.query == "what is the warranty period"
        assert case.expected_documents == ("file-1",)
        assert case.expected_chunks == ("chunk-9",)
        assert case.resource_id == "res-1"
        assert case.notes == "from the FAQ"

    def test_ids_are_coerced_to_strings(self):
        # A hand-written benchmark file writes numeric ids without thinking about it; the
        # ranked ids coming back from the pipeline are always strings, and an int label
        # would silently match nothing while looking present.
        case = BenchmarkCase.from_dict({"query": "q", "expected_chunks": [7], "expected_documents": [3]})
        assert case.expected_chunks == ("7",)
        assert case.expected_documents == ("3",)

    def test_query_is_stripped(self):
        assert BenchmarkCase.from_dict({"query": "  spaced  ", "expected_chunks": ["c"]}).query == "spaced"

    def test_missing_and_null_ground_truth_become_empty_tuples(self):
        # `or ()` is what makes an explicit null survive; without it the comprehension
        # would raise on `None` and take the whole benchmark file down over one row.
        case = BenchmarkCase.from_dict({"query": "q", "expected_documents": None})
        assert case.expected_documents == ()
        assert case.expected_chunks == ()
        assert case.notes == ""
        assert case.resource_id is None

    def test_a_query_without_ground_truth_is_unusable(self):
        # Nothing to score against: including it would drag every average toward zero and
        # look like a retrieval failure rather than a missing label.
        assert BenchmarkCase.from_dict({"query": "q"}).is_usable is False

    def test_ground_truth_without_a_query_is_unusable(self):
        assert BenchmarkCase.from_dict({"expected_chunks": ["c"]}).is_usable is False

    def test_either_kind_of_ground_truth_is_enough(self):
        # Document-only labelling is explicitly supported: a benchmark written before chunk
        # ids were stable still measures whether the right source was surfaced.
        assert _case(chunks=["c"]).is_usable is True
        assert _case(documents=["d"]).is_usable is True


class TestLoadCases:
    def test_reads_a_bare_json_list(self, tmp_path):
        path = tmp_path / "cases.json"
        path.write_text(json.dumps([{"query": "a", "expected_chunks": ["c1"]}]), encoding="utf-8")
        cases = load_cases(str(path))
        assert [case.query for case in cases] == ["a"]

    def test_reads_an_object_with_a_cases_key(self, tmp_path):
        # The bundled demo file uses this shape because it carries a corpus alongside.
        path = tmp_path / "cases.json"
        path.write_text(
            json.dumps({"chunks": [{"id": "x"}], "cases": [{"query": "a", "expected_documents": ["d1"]}]}),
            encoding="utf-8",
        )
        cases = load_cases(str(path))
        assert [case.expected_documents for case in cases] == [("d1",)]

    def test_unusable_rows_are_filtered_out(self, tmp_path):
        path = tmp_path / "cases.json"
        path.write_text(
            json.dumps(
                [
                    {"query": "keep", "expected_chunks": ["c1"]},
                    {"query": "no ground truth"},
                    {"expected_chunks": ["c2"]},
                    {"query": "   ", "expected_chunks": ["c3"]},
                ]
            ),
            encoding="utf-8",
        )
        assert [case.query for case in load_cases(str(path))] == ["keep"]

    def test_non_dict_rows_are_skipped(self, tmp_path):
        # A file with a stray string in the list must not abort the run: the usable cases
        # are still worth measuring, and the alternative is an AttributeError from `.get`.
        path = tmp_path / "cases.json"
        path.write_text(json.dumps(["oops", 3, {"query": "a", "expected_chunks": ["c1"]}]), encoding="utf-8")
        assert len(load_cases(str(path))) == 1

    def test_an_object_without_a_cases_key_yields_nothing(self, tmp_path):
        path = tmp_path / "cases.json"
        path.write_text(json.dumps({"chunks": []}), encoding="utf-8")
        assert load_cases(str(path)) == []


class TestEvaluate:
    def test_scores_chunk_and_document_ground_truth_separately(self):
        case = _case(query="q", chunks=["c1"], documents=["d1"])
        report = evaluate([case], _runner({"q": (["c1"], ["d1"], 4.0)}), variant="full")
        assert report.variant == "full"
        assert report.cases == 1
        assert report.metrics["chunk_recall@1"] == 1.0
        assert report.metrics["doc_recall@1"] == 1.0
        assert report.metrics["chunk_mrr"] == 1.0
        assert report.metrics["doc_mrr"] == 1.0

    def test_a_case_only_contributes_metrics_for_the_labels_it_carries(self):
        # A chunk-only case must not emit doc_* keys at all; emitting them as zeros is what
        # dilution looks like from the inside.
        case = _case(query="q", chunks=["c1"])
        report = evaluate([case], _runner({"q": (["c1"], ["d1"], 1.0)}))
        assert "chunk_recall@1" in report.metrics
        assert not any(key.startswith("doc_") for key in report.metrics)

    def test_mixed_ground_truth_types_do_not_dilute_each_other(self):
        # One chunk-labelled case and one document-labelled case, both answered perfectly.
        # chunk_recall@1 must be 1.0 / 1 chunk-labelled case = 1.0, NOT 1.0 / 2 cases = 0.5,
        # and likewise for the document side. This is the averaging rule the docstring
        # promises, and getting it wrong makes a benchmark look worse the more lenient
        # cases someone adds to it.
        chunk_case = _case(query="chunky", chunks=["c1"])
        doc_case = _case(query="docky", documents=["d1"])
        report = evaluate(
            [chunk_case, doc_case],
            _runner({"chunky": (["c1"], ["dX"], 1.0), "docky": (["cX"], ["d1"], 1.0)}),
        )
        assert report.cases == 2
        assert report.metrics["chunk_recall@1"] == 1.0
        assert report.metrics["doc_recall@1"] == 1.0
        assert report.metrics["chunk_mrr"] == 1.0
        assert report.metrics["doc_mrr"] == 1.0

    def test_averages_over_the_cases_that_carry_that_label(self):
        # Two chunk-labelled cases, one answered and one missed: 1.0 + 0.0 over 2 = 0.5.
        hit = _case(query="hit", chunks=["c1"])
        miss = _case(query="miss", chunks=["c2"])
        report = evaluate(
            [hit, miss],
            _runner({"hit": (["c1"], [], 1.0), "miss": (["zzz"], [], 1.0)}),
        )
        assert report.metrics["chunk_recall@1"] == pytest.approx(0.5)

    def test_document_ids_are_deduplicated_before_scoring(self):
        # Ten chunks from one file are ONE document at rank 1. Deduplicated, the top 5 holds
        # a single document: doc_precision@5 = 1/5 = 0.2. Without the dedup the same
        # retrieval would score 5/5 = 1.0 — a perfect precision earned by the chunker
        # repeating itself, which is exactly backwards.
        case = _case(query="q", documents=["f1"])
        report = evaluate([case], _runner({"q": ([], ["f1"] * 10, 1.0)}))
        assert report.metrics["doc_precision@5"] == pytest.approx(0.2)
        assert report.metrics["doc_recall@5"] == 1.0

    def test_deduplication_keeps_rank_order(self):
        # `dict.fromkeys` preserves first appearance: f2 was seen first, so the relevant f1
        # sits at rank 2 and MRR is 0.5. Sorting or set-ifying here would invent a ranking.
        case = _case(query="q", documents=["f1"])
        report = evaluate([case], _runner({"q": ([], ["f2", "f1", "f2", "f1"], 1.0)}))
        assert report.metrics["doc_mrr"] == pytest.approx(0.5)

    def test_per_case_rows_carry_the_query_and_its_latency(self):
        # The per-case rows are what a reader opens when an average looks wrong, so they
        # have to identify the query and not just the score.
        cases = [_case(query="one", chunks=["c1"]), _case(query="two", chunks=["c2"])]
        report = evaluate(cases, _runner({"one": (["c1"], [], 12.3456), "two": ([], [], 7.0)}))
        assert [row["query"] for row in report.per_case] == ["one", "two"]
        assert report.per_case[0]["latency_ms"] == pytest.approx(12.346)
        assert report.per_case[0]["chunk_hit@1"] == 1.0
        assert report.per_case[1]["chunk_hit@1"] == 0.0

    def test_latency_is_summarised_but_never_averaged_into_the_metrics(self):
        # Latency is an order statistic (nearest rank, no interpolation) and must stay out
        # of `metrics`, where it would be rounded to 4 decimals and read as a score.
        cases = [_case(query=str(index), chunks=["c"]) for index in range(5)]
        table = {str(index): ([], [], value) for index, value in enumerate([30.0, 10.0, 50.0, 20.0, 40.0])}
        report = evaluate(cases, _runner(table))
        assert report.latency_ms["mean"] == pytest.approx(30.0)
        assert report.latency_ms["p50"] == pytest.approx(30.0)   # sorted[round(0.5*4)] = sorted[2]
        assert report.latency_ms["p95"] == pytest.approx(50.0)   # sorted[round(0.95*4)] = sorted[4]
        assert report.latency_ms["max"] == pytest.approx(50.0)
        assert "latency_ms" not in report.metrics

    def test_percentiles_are_nearest_rank_not_interpolated(self):
        # Four samples, p50 = sorted[round(0.5*3)] = sorted[2] = 30.0, not the 25.0 an
        # interpolating median would report. Pinned because the difference only shows up
        # on an even number of cases.
        cases = [_case(query=str(index), chunks=["c"]) for index in range(4)]
        table = {str(index): ([], [], value) for index, value in enumerate([10.0, 20.0, 30.0, 40.0])}
        report = evaluate(cases, _runner(table))
        assert report.latency_ms["p50"] == pytest.approx(30.0)

    def test_respects_the_requested_cut_offs(self):
        case = _case(query="q", chunks=["c1"])
        report = evaluate([case], _runner({"q": (["c1"], [], 1.0)}), ks=(3,))
        assert "chunk_recall@3" in report.metrics
        assert "chunk_recall@1" not in report.metrics
        assert "chunk_mrr" in report.metrics  # MRR carries no cut-off

    def test_no_cases_produces_an_empty_report_rather_than_a_division_error(self):
        # `--cases` pointing at a file whose rows were all filtered out reaches here.
        report = evaluate([], _runner({}))
        assert report.cases == 0
        assert report.metrics == {}
        assert report.latency_ms == {"mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}

    def test_the_runner_is_called_once_per_case(self):
        seen = []

        def run(case):
            seen.append(case.query)
            return ([], [], 1.0)

        evaluate([_case(query="a", chunks=["c"]), _case(query="b", chunks=["c"])], run)
        assert seen == ["a", "b"]

    def test_to_dict_rounds_for_reporting_only(self):
        # The JSON report is read by a human; the in-memory metrics stay full precision so
        # a later comparison is not made against a rounded number.
        case = _case(query="q", chunks=["c1"])
        report = evaluate([case], _runner({"q": (["zz", "c1"], [], 1.23456)}))
        payload = report.to_dict()
        assert payload["variant"] == "full"
        assert payload["cases"] == 1
        assert payload["metrics"]["chunk_ndcg@5"] == pytest.approx(0.6309, abs=1e-4)
        assert report.metrics["chunk_ndcg@5"] == pytest.approx(1 / math.log2(3))
        assert payload["latency_ms"]["mean"] == pytest.approx(1.235)
        assert payload["per_case"][0]["query"] == "q"


class TestCompare:
    def test_runs_every_variant_over_every_case(self):
        calls = []

        def run_variant(variant, overrides, case):
            calls.append((variant, case.query))
            return ([], [], 1.0)

        cases = [_case(query="a", chunks=["c"]), _case(query="b", chunks=["c"])]
        reports = compare(cases, run_variant)
        assert set(reports) == set(VARIANTS)
        assert len(calls) == len(VARIANTS) * len(cases)
        for name, report in reports.items():
            assert report.variant == name
            assert report.cases == 2

    def test_each_variant_receives_its_own_override_set(self):
        # The regression this guards: `compare` builds one lambda per variant inside a loop.
        # Binding `variant`/`overrides` late instead of by default argument would hand every
        # run the LAST variant's configuration, and the table would show five identical rows
        # that read as "no stage does anything" rather than as a harness bug.
        received = {}

        def run_variant(variant, overrides, case):
            received[variant] = dict(overrides)
            return ([], [], 1.0)

        compare([_case(chunks=["c"])], run_variant)
        assert received == {name: dict(overrides) for name, overrides in VARIANTS.items()}

    def test_the_bm25_variant_turns_off_every_other_stage(self):
        # The ablation is only attributable if the overrides really are the single-stage
        # ones; asserting the exact set is what stops a knob being added to `PrecisionConfig`
        # and silently staying on in the "lexical only" row.
        received = {}

        def run_variant(variant, overrides, case):
            received[variant] = dict(overrides)
            return ([], [], 1.0)

        compare([_case(chunks=["c"])], run_variant, variants=("bm25",))
        assert received["bm25"] == VARIANTS["bm25"]
        assert received["bm25"]["dense_enabled"] is False
        assert received["bm25"]["reranker_enabled"] is False
        assert received["bm25"]["query_expansion_enabled"] is False
        assert received["bm25"]["mmr_enabled"] is False
        # bm25 itself is NOT disabled — that is the whole point of the row.
        assert "bm25_enabled" not in received["bm25"]

    def test_the_full_variant_overrides_nothing(self):
        # `full` must measure whatever the deployment's configuration actually is, or the
        # benchmark reports on a pipeline nobody runs.
        received = {}

        def run_variant(variant, overrides, case):
            received[variant] = dict(overrides)
            return ([], [], 1.0)

        compare([_case(chunks=["c"])], run_variant, variants=("full",))
        assert received["full"] == {}

    def test_only_the_requested_variants_run(self):
        ran = []

        def run_variant(variant, overrides, case):
            ran.append(variant)
            return ([], [], 1.0)

        reports = compare([_case(chunks=["c"])], run_variant, variants=("dense", "full"))
        assert ran == ["dense", "full"]
        assert list(reports) == ["dense", "full"]

    def test_an_unknown_variant_name_runs_with_no_overrides(self):
        # Documenting the actual behaviour rather than endorsing it: `VARIANTS.get(name, {})`
        # makes a typo behave exactly like `full`. It cannot reach here from the CLI, which
        # filters `--variants` against `VARIANTS` before calling, so the empty-override
        # fallback is only reachable from code that already knows the names.
        received = {}

        def run_variant(variant, overrides, case):
            received[variant] = dict(overrides)
            return ([], [], 1.0)

        compare([_case(chunks=["c"])], run_variant, variants=("bm52",))
        assert received == {"bm52": {}}

    def test_scores_each_variant_from_its_own_results(self):
        # Two variants, different rankings for the same case: the reports must not share
        # state, which is the other way a per-variant closure bug shows up.
        case = _case(query="q", chunks=["c1"])

        def run_variant(variant, overrides, case_in):
            return ((["c1"], [], 1.0) if variant == "full" else (["zz", "c1"], [], 1.0))

        reports = compare([case], run_variant, variants=("bm25", "full"))
        assert reports["full"].metrics["chunk_recall@1"] == 1.0
        assert reports["bm25"].metrics["chunk_recall@1"] == 0.0
        assert reports["bm25"].metrics["chunk_recall@5"] == 1.0

    def test_passes_the_cut_offs_through(self):
        reports = compare(
            [_case(query="q", chunks=["c1"])],
            lambda variant, overrides, case: (["c1"], [], 1.0),
            variants=("full",),
            ks=(2,),
        )
        assert "chunk_recall@2" in reports["full"].metrics
        assert "chunk_recall@1" not in reports["full"].metrics


class TestFormatTable:
    def _reports(self):
        return compare(
            [_case(query="q", chunks=["c1"], documents=["d1"])],
            lambda variant, overrides, case: (["c1"], ["d1"], 5.0),
        )

    def test_names_every_variant_it_was_given(self):
        # The table is the artefact people actually read; a variant missing from it is a
        # measurement silently dropped.
        table = format_table(self._reports())
        for name in VARIANTS:
            assert name in table

    def test_shows_the_headline_metrics_and_latency(self):
        table = format_table(self._reports())
        assert "chunk_recall@1" in table
        assert "chunk_mrr" in table
        assert "latency p50" in table
        assert "1.0000" in table  # a perfect recall, formatted to four decimals
        assert "5.0ms" in table

    def test_an_empty_report_dict_says_so_instead_of_crashing(self):
        assert format_table({}) == "(no results)"

    def test_reports_with_no_metrics_still_render(self):
        # `evaluate([])` produces exactly this: no metrics, so no columns. The table must
        # still name the variants rather than raising while a benchmark run is being
        # summarised.
        reports = compare([], lambda variant, overrides, case: ([], [], 0.0), variants=("bm25", "full"))
        table = format_table(reports)
        assert "bm25" in table and "full" in table
        assert "0.0ms" in table

    def test_unavailable_columns_are_dropped_rather_than_zero_filled(self):
        # A document-only benchmark has no chunk_* metrics; printing them as 0.0000 would
        # read as a pipeline that found nothing, not as a benchmark that never asked.
        reports = compare(
            [_case(query="q", documents=["d1"])],
            lambda variant, overrides, case: ([], ["d1"], 1.0),
            variants=("full",),
        )
        table = format_table(reports)
        assert "doc_mrr" in table
        assert "chunk_recall@1" not in table

    def test_explicit_columns_are_honoured(self):
        table = format_table(self._reports(), keys=["doc_recall@5"])
        assert "doc_recall@5" in table
        assert "chunk_mrr" not in table


class TestVariantRegistry:
    def test_every_override_key_is_a_real_configuration_field(self):
        # An override naming a field that does not exist is dropped by
        # `PrecisionConfig.with_overrides`, so the ablation row would silently run with the
        # stage still switched ON and the table would argue the stage is worthless.
        from dataclasses import fields

        from rag.precision.config import PrecisionConfig

        known = {field.name for field in fields(PrecisionConfig)}
        for name, overrides in VARIANTS.items():
            unknown = sorted(set(overrides) - known)
            assert unknown == [], f"{name} overrides unknown config fields: {unknown}"

    def test_full_is_the_deployment_configuration(self):
        assert VARIANTS["full"] == {}

    def test_default_variant_order_walks_stages_back_on(self):
        # The table reads as an argument: lexical alone, dense alone, both, both reranked,
        # everything. Reordering it does not break a test elsewhere, but it does break the
        # only reason the output is legible.
        assert list(VARIANTS) == ["bm25", "dense", "bm25_dense", "bm25_dense_rerank", "full"]

    def test_the_module_exposes_the_default_cut_offs(self):
        assert evaluation.DEFAULT_KS == (1, 5, 10)
