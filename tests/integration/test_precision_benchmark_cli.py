"""The benchmark CLI is a documented command, so the suite has to be able to run it.

`python -m rag.precision.benchmark --demo` is named in CLAUDE.md §26 and in the module's own
docstring as the way to reproduce the numbers reported there. Left untested it was the one
part of the mode at 0% coverage — and the one part whose breakage would be discovered by
somebody following the documentation.

Marked `integration` rather than `unit` because `main()` imports `rag.service`, which pulls
in the ORM and the settings — no database is touched, but the import chain is not a unit's.
"""
import json

import pytest

from rag.precision import evaluation
from rag.precision.benchmark import DEMO_PATH, load_corpus, main

pytestmark = pytest.mark.integration


class TestTheBundledDemo:
    def test_the_demo_dataset_is_well_formed(self):
        payload = json.loads(DEMO_PATH.read_text(encoding="utf-8"))
        chunks = load_corpus(payload)
        cases = [evaluation.BenchmarkCase.from_dict(row) for row in payload["cases"]]

        assert len(chunks) >= 40
        assert all(case.is_usable for case in cases), "an unusable case is silently skipped"

        # Every expected id must exist in the corpus. A typo in ground truth reads as a
        # retrieval regression — the metric drops and nothing points at the benchmark.
        known = {chunk.id for chunk in chunks}
        documents = {chunk.file_id for chunk in chunks}
        for case in cases:
            for group in case.chunk_groups:
                assert group <= known, f"unknown chunk id in ground truth for {case.query!r}"
            assert set(case.expected_documents) <= documents, case.query

    def test_chunk_ids_are_unique(self):
        # Two chunks sharing an id would make the postings list and the metadata map
        # disagree about which passage they describe.
        chunks = load_corpus(json.loads(DEMO_PATH.read_text(encoding="utf-8")))
        assert len({chunk.id for chunk in chunks}) == len(chunks)


class TestTheCli:
    def test_demo_runs_and_writes_a_report(self, tmp_path, capsys):
        out = tmp_path / "report.json"
        assert main(["--demo", "--json", str(out)]) == 0

        printed = capsys.readouterr().out
        assert "High-Precision RAG benchmark" in printed
        for variant in evaluation.VARIANTS:
            assert variant in printed

        report = json.loads(out.read_text(encoding="utf-8"))
        assert set(report) == set(evaluation.VARIANTS)
        for variant, entry in report.items():
            assert entry["cases"] > 0, variant
            assert "latency_ms" in entry

    def test_the_full_pipeline_beats_bm25_alone_on_the_demo(self, tmp_path):
        # The claim CLAUDE.md §26 makes, asserted rather than trusted. Deliberately an
        # INEQUALITY and not the published figures: pinning 0.8906 would turn any future
        # tuning into a test failure, while a full pipeline that stopped beating its own
        # first stage is a real regression that nothing else in the suite would catch.
        out = tmp_path / "report.json"
        assert main(["--demo", "--json", str(out), "--variants", "bm25,full"]) == 0
        report = json.loads(out.read_text(encoding="utf-8"))

        bm25 = report["bm25"]["metrics"]
        full = report["full"]["metrics"]
        assert full["chunk_recall@1"] > bm25["chunk_recall@1"]
        assert full["chunk_mrr"] > bm25["chunk_mrr"]
        assert full["chunk_ndcg@10"] > bm25["chunk_ndcg@10"]

    def test_variant_names_survive_surrounding_whitespace(self, tmp_path):
        # `--variants "bm25, full"` is what a person types. The membership test used to strip
        # and the result used to not, so " full" reached `VARIANTS.get` as an unknown key and
        # silently ran the default configuration under the wrong label.
        out = tmp_path / "report.json"
        assert main(["--demo", "--json", str(out), "--variants", "bm25, full"]) == 0
        assert set(json.loads(out.read_text(encoding="utf-8"))) == {"bm25", "full"}

    def test_bad_argument_combinations_are_refused(self):
        with pytest.raises(SystemExit):
            main([])                                   # neither --demo nor --cases
        with pytest.raises(SystemExit):
            main(["--cases", "nowhere.json"])          # --cases with no corpus or resource
        with pytest.raises(SystemExit):
            main(["--cases", "c.json", "--resource-id", "r"])  # a resource with no user
