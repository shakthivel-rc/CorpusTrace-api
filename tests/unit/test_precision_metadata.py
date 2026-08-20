"""Unit tests for `rag/precision/metadata.py` — the derived structure the high-precision
mode ranks and filters on.

Nothing in this module reads a column that exists. `section`, `heading`, `document_type`,
`category`, `version` and the parent window are all *inferred* from the text and the
filename a chunk already carries, which is what lets the mode work on knowledge bases
indexed long before it existed. That is also what makes it worth testing hard: an inference
that quietly over-reaches produces a confident wrong answer rather than an error.

Three properties here fail silently in production and are the reason this file exists:

* **A heading must stop where the heading stops.** Ingestion normalizes whitespace away, so
  a heading and the sentence beneath it arrive as one line. Running on into the prose gives
  the chunk a heading the document never had — displayed as provenance and scored as a
  retrieval signal (CLAUDE.md §20's failure mode, in text rather than in a PDF highlight).
  The determiner break in `_title_run` was a review fix, and the test named for it pins the
  exact input that motivated it.
* **A parent window must never span two documents.** `parent_key` is keyed on `file_id`, and
  the whole point is that the tail of one document cannot be stitched onto the head of the
  next — that would fabricate context present in neither.
* **A filter must never be the reason the answer was not ranked.** `apply_filters` keeps a
  filter only when enough candidates survive it, and records the ones it dropped so the
  trace can say which. A filter that empties the pool turns a ranking problem into a
  refusal about a document set the user never asked to restrict.

Pure functions only: no database, no session, no network, and no `models` import.
"""
from dataclasses import dataclass

import pytest

from rag.precision.metadata import (
    CATEGORY_HINTS,
    apply_filters,
    derive_metadata,
    detect_category,
    detect_document_type,
    detect_heading,
    detect_version,
    infer_filters,
    metadata_score,
    parent_key,
)
from rag.precision.types import ChunkMetadata

pytestmark = pytest.mark.unit


@dataclass
class _Chunk:
    """The subset of `types.ChunkLike` this module actually reads.

    A local dataclass rather than `models.rag.DocumentChunk` on purpose: the precision
    package is structurally typed and imports no ORM, so its unit tests should not drag a
    database import in either.
    """

    id: str = "chunk-1"
    file_id: str | None = "file-1"
    chunk_index: int = 0
    source_name: str = "doc.pdf"
    modality: str = "text"
    content: str = ""
    page_start: int | None = None


@dataclass
class _StubCandidate:
    """`apply_filters` reads exactly one attribute off a candidate."""

    chunk_id: str


def _meta(chunk_id: str = "c1", **overrides) -> ChunkMetadata:
    fields = dict(
        document_id="file-1",
        chunk_id=chunk_id,
        parent_chunk_id="file-1:0",
        section=None,
        heading=None,
        page=None,
        document_type="pdf",
        category=None,
        version=None,
        entities=(),
    )
    fields.update(overrides)
    return ChunkMetadata(**fields)


class TestDetectHeading:
    def test_numbered_heading_keeps_every_word_of_a_multi_word_title(self):
        # The regex-only version of this stopped at the second word ("Connection"), because
        # a lazy pattern terminated by "the next capitalised word" cannot tell a heading's
        # own words from the sentence that follows it.
        assert detect_heading("3.2 Connection Pooling The database maintains a pool.") == (
            "Connection Pooling",
            "3.2",
        )

    def test_determiner_after_a_capitalised_word_ends_the_run(self):
        # The bug this pins: trimming only the LAST word of the run is not enough, because
        # "An" is followed by "API" — another capitalised word — so the run ran on to
        # "Authentication Tokens An API". A determiner has to break the run wherever it
        # appears, not merely be trimmed off the end.
        assert detect_heading("4.1 Authentication Tokens An API token is issued.") == (
            "Authentication Tokens",
            "4.1",
        )

    def test_lowercase_connector_survives_inside_a_heading(self):
        # "of" is legitimately part of the title. Breaking on every lowercase word would
        # truncate a large share of real headings to their first word.
        assert detect_heading("12. Terms of Service This agreement governs your use.") == (
            "Terms of Service",
            "12",
        )

    def test_all_caps_heading_is_title_cased_and_has_no_section(self):
        # A caps run carries no number, and None means unknown — never a fabricated "1".
        assert detect_heading("SAFETY PRECAUTIONS Always disconnect the battery.") == (
            "Safety Precautions",
            None,
        )

    def test_colon_terminated_title_case_phrase_is_a_heading(self):
        assert detect_heading("Connection Pooling: The pool is sized at startup.") == (
            "Connection Pooling",
            None,
        )

    def test_prose_containing_a_colon_is_not_a_heading(self):
        # A colon in ordinary prose is common; the shape test (short, mostly capitalised)
        # is what keeps this from labelling half the corpus.
        assert detect_heading(
            "There are three reasons for this decision: cost, speed and reliability."
        ) == (None, None)

    def test_ordinary_prose_has_no_heading(self):
        assert detect_heading("The database maintains a pool of connections.") == (None, None)

    def test_capitalised_proper_noun_opener_is_not_a_heading(self):
        # A Title Case run on its own is not evidence of structure — this is why
        # `_title_run` is only ever consulted after a section number has matched.
        assert detect_heading("John Smith reported the outage at 3am.") == (None, None)

    def test_a_leading_year_is_not_a_section_number(self):
        # "In 2024 ..." and "3.2 million" both look numeric; neither opens a heading, and
        # labelling them would put an invented section on an ordinary sentence.
        assert detect_heading("In 2024 the company shipped 3.2 million units.") == (None, None)

    def test_empty_text_is_unknown_not_an_error(self):
        assert detect_heading("") == (None, None)

    def test_a_run_longer_than_a_heading_is_abandoned_not_truncated(self):
        # Past MAX_HEADING_WORDS the run is dropped entirely rather than cut short: a
        # twelve-word "heading" is a sentence someone capitalised, and the first ten words
        # of it would be a heading the document never had.
        assert detect_heading(
            "3.2 Alpha Bravo Charlie Delta Echo Foxtrot Golf Hotel India Juliet Kilo Lima are codewords."
        ) == (None, None)


class TestDetectDocumentType:
    def test_extension_refines_a_coarser_modality(self):
        # `modality` on the row is coarse ("text"/"table"/"pdf"); the extension is the more
        # specific of the two, so it wins.
        assert detect_document_type("manual.pdf", "table") == "pdf"

    def test_extension_matching_is_case_insensitive(self):
        assert detect_document_type("Report.PDF", "text") == "pdf"

    def test_unknown_extension_falls_back_to_modality(self):
        assert detect_document_type("notes.rtf", "table") == "table"

    def test_no_extension_falls_back_to_modality(self):
        assert detect_document_type("README", "text") == "text"

    def test_nothing_known_at_all_still_returns_a_type(self):
        # The field is non-optional on ChunkMetadata, so there is always an answer.
        assert detect_document_type("", "") == "text"


class TestDetectVersion:
    def test_v_prefixed_token(self):
        assert detect_version("Release v4.2 introduces pooling.") == "4.2"

    def test_spelled_out_version_word(self):
        assert detect_version("This is version 4.2 of the guide.") == "4.2"

    def test_three_part_version(self):
        assert detect_version("Applies to 4.2.1 and later.") == "4.2.1"

    def test_a_bare_integer_is_never_a_version(self):
        # Every document is full of bare integers. Treating one as a version would attach a
        # constraint to a chunk that nothing in it stated.
        assert detect_version("There are 42 connections and 7 retries.") is None
        assert detect_version("Chapter 12 of the manual.") is None

    def test_only_the_first_match_is_reported(self):
        # A document naming six versions has no single version; reporting the last one seen
        # would be a filter that silently excludes on a coin flip.
        assert detect_version("Covers 1.0, 2.0 and 3.0 releases.") == "1.0"

    def test_nothing_found_is_none(self):
        assert detect_version("no numbers here at all") is None
        assert detect_version("") is None


class TestDetectCategory:
    def test_filename_beats_heading(self):
        # A document called troubleshooting-guide.pdf is about troubleshooting whatever any
        # one of its passages happens to be headed. The heading only breaks a tie the
        # filename did not answer — and "api" is *earlier* in CATEGORY_HINTS than
        # "troubleshooting", so this cannot pass by dict-ordering accident.
        assert detect_category("troubleshooting-guide.pdf", "API Endpoints") == "troubleshooting"

    def test_heading_is_consulted_when_the_filename_says_nothing(self):
        assert detect_category("document.pdf", "Token Rotation") == "authentication"

    def test_underscores_in_a_filename_are_treated_as_separators(self):
        assert detect_category("api_reference.pdf", None) == "api"

    def test_unknown_gives_none(self):
        # None means "no category", and infer_filters will therefore never propose one —
        # which is the whole safety property of the category filter.
        assert detect_category("scan-0001.bin", None) is None
        assert detect_category("scan-0001.bin", "Nothing Relevant Here") is None


class TestParentKey:
    def test_adjacent_indices_in_one_file_share_a_key(self):
        keys = [parent_key(_Chunk(chunk_index=i), 3) for i in range(3)]
        assert keys == ["file-1:0"] * 3

    def test_the_window_advances_at_the_group_boundary(self):
        assert parent_key(_Chunk(chunk_index=2), 3) != parent_key(_Chunk(chunk_index=3), 3)

    def test_two_files_never_share_a_key_at_the_same_index(self):
        # This is the property the whole parent stage rests on: stitching the tail of one
        # document onto the head of the next fabricates context that exists in neither.
        assert parent_key(_Chunk(file_id="file-1", chunk_index=1), 3) != parent_key(
            _Chunk(file_id="file-2", chunk_index=1), 3
        )

    def test_group_size_of_one_makes_every_chunk_its_own_parent(self):
        keys = [parent_key(_Chunk(chunk_index=i), 1) for i in range(3)]
        assert len(set(keys)) == 3

    def test_missing_file_id_falls_back_to_source_name(self):
        # file_id is nullable on the row; the filename still separates two documents.
        assert parent_key(_Chunk(file_id=None, source_name="doc.pdf", chunk_index=4), 3) == "doc.pdf:1"

    def test_no_identity_at_all_still_produces_a_key(self):
        assert parent_key(_Chunk(file_id=None, source_name="", chunk_index=4), 3) == "unknown:1"

    def test_a_negative_index_does_not_crash_and_lands_in_the_first_window(self):
        # Indices come from stored rows, so this is untrusted input in the same sense the
        # ingestion parsers treat an upload as untrusted.
        assert parent_key(_Chunk(chunk_index=-5), 3) == "file-1:0"

    def test_a_non_integer_index_is_treated_as_zero(self):
        assert parent_key(_Chunk(chunk_index=None), 3) == "file-1:0"

    def test_a_zero_group_size_does_not_divide_by_zero(self):
        assert parent_key(_Chunk(chunk_index=2), 0) == "file-1:2"


class TestMetadataScore:
    def test_missing_metadata_scores_zero(self):
        # The pipeline passes `corpus.metadata.get(chunk_id)`, which can be None.
        assert metadata_score(None, {}, set()) == 0.0

    def test_no_filters_and_no_heading_scores_zero(self):
        assert metadata_score(_meta(), {}, set()) == 0.0

    def test_a_matching_version_contributes(self):
        assert metadata_score(_meta(version="4.2"), {"version": "4.2"}, set()) == pytest.approx(0.4)

    def test_a_mismatched_version_contributes_nothing(self):
        assert metadata_score(_meta(version="3.1"), {"version": "4.2"}, set()) == 0.0

    def test_document_type_and_category_matches_contribute(self):
        score = metadata_score(
            _meta(document_type="spreadsheet", category="api"),
            {"document_type": "spreadsheet", "category": "api"},
            set(),
        )
        assert score == pytest.approx(0.4)

    def test_heading_overlap_contributes_and_is_capped(self):
        # A heading naming the question's words is the strongest structural signal here, but
        # it is capped so a long heading cannot swamp the retrieval scores it is nudging.
        one = metadata_score(_meta(heading="Connection Pooling"), {}, {"connection"})
        many = metadata_score(
            _meta(heading="Connection Pooling Timeout Settings"),
            {},
            {"connection", "pooling", "timeout", "settings"},
        )
        assert one == pytest.approx(0.2)
        assert many == pytest.approx(0.4)

    def test_a_heading_with_no_overlap_contributes_nothing(self):
        assert metadata_score(_meta(heading="Connection Pooling"), {}, {"invoice"}) == 0.0

    def test_the_score_is_bounded_at_one(self):
        # The parts sum to 1.2 at maximum; the fusion weights assume a 0..1 signal, so an
        # unclamped score would silently re-weight the whole ranking.
        score = metadata_score(
            _meta(
                version="4.2",
                document_type="pdf",
                category="api",
                heading="Api Version Pdf Connection",
            ),
            {"version": "4.2", "document_type": "pdf", "category": "api"},
            {"api", "version", "pdf", "connection"},
        )
        assert score == 1.0

    def test_the_score_is_never_negative(self):
        assert metadata_score(_meta(version="3.1", category="policy"), {"version": "4.2"}, {"x"}) >= 0.0


class TestInferFilters:
    def test_a_version_in_the_question_is_inferred(self):
        assert infer_filters("what changed in 4.2", ["changed", "4.2"], set()) == {"version": "4.2"}

    def test_a_bare_integer_in_the_question_is_not_a_version(self):
        assert infer_filters("what changed in release 4", ["changed", "release"], set()) == {}

    def test_a_named_format_infers_a_document_type(self):
        assert infer_filters("show the spreadsheet totals", ["spreadsheet", "totals"], set()) == {
            "document_type": "spreadsheet"
        }

    def test_csv_maps_onto_the_spreadsheet_type(self):
        # The filter is compared against `detect_document_type`'s vocabulary, not the user's,
        # so the synonym has to be resolved here or the filter matches nothing.
        assert infer_filters("totals in the csv", ["totals", "csv"], set())["document_type"] == "spreadsheet"

    def test_a_category_is_proposed_only_when_the_corpus_has_it(self):
        terms = ["how", "token", "rotation"]
        assert infer_filters("how does token rotation work", terms, {"authentication"}) == {
            "category": "authentication"
        }

    def test_a_category_the_corpus_lacks_is_never_proposed(self):
        # Inferring category=authentication in a base with no auth document filters
        # everything out, and the user gets a refusal about a restriction they never asked
        # for. This negative is the entire reason `available_categories` is a parameter.
        terms = ["how", "token", "rotation"]
        assert infer_filters("how does token rotation work", terms, set()) == {}
        assert infer_filters("how does token rotation work", terms, {"policy"}) == {}

    def test_a_question_with_no_signals_yields_no_filters(self):
        assert infer_filters("what is the weather", ["weather"], set(CATEGORY_HINTS)) == {}


class TestApplyFilters:
    def _pool(self, versions: list[str]) -> tuple[list, dict]:
        candidates = [_StubCandidate(f"c{i}") for i in range(len(versions))]
        metadata = {
            f"c{i}": _meta(f"c{i}", version=version) for i, version in enumerate(versions)
        }
        return candidates, metadata

    def test_a_filter_leaving_enough_candidates_is_applied(self):
        candidates, metadata = self._pool(["4.2"] * 4 + ["3.1"] * 2)
        survivors, applied = apply_filters(candidates, metadata, {"version": "4.2"}, 3)
        assert [c.chunk_id for c in survivors] == ["c0", "c1", "c2", "c3"]
        assert applied == {"version": "4.2"}

    def test_min_survivors_is_inclusive(self):
        candidates, metadata = self._pool(["4.2"] * 3 + ["3.1"] * 3)
        _, applied = apply_filters(candidates, metadata, {"version": "4.2"}, 3)
        assert applied == {"version": "4.2"}

    def test_a_filter_leaving_too_few_is_dropped_and_recorded_as_none(self):
        # None is "proposed, not applied" — distinct from the key being absent, which would
        # be "never considered". The trace has to be able to say which happened.
        candidates, metadata = self._pool(["4.2"] * 3 + ["3.1"] * 3)
        survivors, applied = apply_filters(candidates, metadata, {"version": "4.2"}, 4)
        assert applied == {"version": None}
        assert [c.chunk_id for c in survivors] == [c.chunk_id for c in candidates]

    def test_a_filter_matching_nothing_leaves_the_pool_untouched(self):
        # The dangerous case: a guessed constraint that no chunk satisfies must not be the
        # reason the answer was never ranked.
        candidates, metadata = self._pool(["4.2"] * 6)
        survivors, applied = apply_filters(candidates, metadata, {"version": "9.9"}, 1)
        assert applied == {"version": None}
        assert survivors == candidates

    def test_metadata_missing_for_a_candidate_is_a_non_match_not_a_crash(self):
        candidates, metadata = self._pool(["4.2"] * 6)
        del metadata["c5"]
        survivors, applied = apply_filters(candidates, metadata, {"version": "4.2"}, 3)
        assert applied == {"version": "4.2"}
        assert "c5" not in [c.chunk_id for c in survivors]

    def test_each_filter_is_judged_separately_so_one_drop_does_not_lose_the_other(self):
        # Two filters, one safe and one not: the safe one still narrows the pool and the
        # unsafe one is reported dropped, rather than either being all-or-nothing.
        candidates = [_StubCandidate(f"c{i}") for i in range(6)]
        metadata = {
            f"c{i}": _meta(f"c{i}", version="4.2" if i < 4 else "3.1", document_type="pdf")
            for i in range(6)
        }
        survivors, applied = apply_filters(
            candidates, metadata, {"version": "4.2", "document_type": "spreadsheet"}, 3
        )
        assert applied == {"version": "4.2", "document_type": None}
        assert [c.chunk_id for c in survivors] == ["c0", "c1", "c2", "c3"]

    def test_a_later_filter_is_judged_against_what_survived_the_earlier_one(self):
        # Narrowing compounds: `document_type` is tested on the four chunks `version` left,
        # not on the original six. That is what keeps min_survivors a promise about the pool
        # the reranker will actually see.
        candidates = [_StubCandidate(f"c{i}") for i in range(6)]
        metadata = {
            f"c{i}": _meta(
                f"c{i}",
                version="4.2" if i < 4 else "3.1",
                document_type="pdf" if i in (0, 1, 4, 5) else "text",
            )
            for i in range(6)
        }
        survivors, applied = apply_filters(
            candidates, metadata, {"version": "4.2", "document_type": "pdf"}, 3
        )
        # pdf alone would leave four (c0, c1, c4, c5); after version it leaves only two, so
        # it is dropped and the version-filtered pool stands.
        assert applied == {"version": "4.2", "document_type": None}
        assert [c.chunk_id for c in survivors] == ["c0", "c1", "c2", "c3"]

    def test_no_filters_returns_the_pool_unchanged(self):
        candidates, metadata = self._pool(["4.2"] * 3)
        survivors, applied = apply_filters(candidates, metadata, {}, 2)
        assert survivors == candidates
        assert applied == {}


class TestDeriveMetadata:
    def test_it_composes_the_per_chunk_structure_from_the_row_alone(self):
        # `derive_metadata` is the only producer of ChunkMetadata, which is what makes
        # persisting these as columns later a purely additive change.
        chunk = _Chunk(
            id="chunk-9",
            file_id="file-7",
            chunk_index=4,
            source_name="maintenance-manual.pdf",
            modality="text",
            content="SAFETY PRECAUTIONS Always disconnect the battery before servicing.",
            page_start=12,
        )
        meta = derive_metadata(chunk, ("battery",), 3)
        assert meta.chunk_id == "chunk-9"
        assert meta.document_id == "file-7"
        assert meta.parent_chunk_id == "file-7:1"
        assert (meta.heading, meta.section) == ("Safety Precautions", None)
        assert meta.document_type == "pdf"
        assert meta.category == "manual"
        assert meta.page == 12
        assert meta.entities == ("battery",)

    def test_an_unstructured_chunk_reports_unknown_rather_than_guessing(self):
        # A neutral filename on purpose: `detect_category` reads the *filename* first, so
        # any word from CATEGORY_HINTS in it would answer for the whole document.
        chunk = _Chunk(content="The database maintains a pool of connections.", source_name="scan-0001.bin")
        meta = derive_metadata(chunk, (), 3)
        assert meta.heading is None
        assert meta.section is None
        assert meta.category is None
        assert meta.version is None
