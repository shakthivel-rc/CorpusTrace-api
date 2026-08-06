"""Unit tests for chunk provenance — the page and character span recorded at ingestion.

These offsets are what lets an answer point at the page it came from, and they are the one
part of the pipeline whose failure mode is *silent and confidently wrong*: a mis-tracked
page number does not raise, it just highlights the wrong paragraph. So the properties
asserted here are exact — the span must slice the chunk back out of the text, byte for
byte, and a page whose text extraction fails must not shift its neighbours' numbers.
"""
import pytest

from rag.service import (
    _extract_document,
    _extract_pdf_pages,
    _extract_pdf_text,
    _extract_text,
    _derive_title,
    _join_pages,
    _normalize_whitespace,
    _pages_for_span,
    _split_into_chunk_spans,
    _split_into_chunks,
)

pytestmark = pytest.mark.unit


class TestChunkSpans:
    def test_every_span_slices_its_own_chunk_back_out(self):
        """The property the whole feature rests on. If this drifts by one character the
        highlight silently lands on the wrong text."""
        text = ". ".join(f"Sentence number {index} about compliance controls" for index in range(200))
        cleaned = _normalize_whitespace(text)

        spans = list(_split_into_chunk_spans(text))

        assert len(spans) > 1, "the fixture must be long enough to actually split"
        for chunk, start, end in spans:
            assert cleaned[start:end] == chunk

    def test_spans_advance_and_stay_inside_the_text(self):
        text = "word " * 500
        cleaned = _normalize_whitespace(text)
        spans = list(_split_into_chunk_spans(text))

        assert spans[0][1] == 0
        assert spans[-1][2] == len(cleaned)
        for _, start, end in spans:
            assert 0 <= start < end <= len(cleaned)
        # Overlap means starts advance monotonically but do not jump past the previous end.
        for (_, _, previous_end), (_, start, _) in zip(spans, spans[1:]):
            assert start < previous_end

    def test_a_short_document_is_one_span_covering_everything(self):
        chunk, start, end = next(iter(_split_into_chunk_spans("  hello world  ")))
        assert (chunk, start, end) == ("hello world", 0, 11)

    def test_the_string_only_chunker_is_unchanged(self):
        """`_split_into_chunks` is the older API and several call sites still use it — it
        must keep yielding exactly what the span version yields."""
        text = "A" * 700 + ". " + "B" * 2000
        assert list(_split_into_chunks(text)) == [chunk for chunk, _, _ in _split_into_chunk_spans(text)]


class TestPageJoining:
    def test_spans_locate_each_page_in_the_joined_text(self):
        text, spans = _join_pages([(1, "alpha beta"), (2, "gamma"), (3, "delta epsilon")])

        assert text == "alpha beta gamma delta epsilon"
        assert spans == [(1, 0, 10), (2, 11, 16), (3, 17, 30)]
        for number, start, end in spans:
            assert text[start:end] == {1: "alpha beta", 2: "gamma", 3: "delta epsilon"}[number]

    def test_blank_pages_contribute_no_span_and_no_separator(self):
        text, spans = _join_pages([(1, "alpha"), (2, ""), (3, "gamma")])

        assert text == "alpha gamma"
        assert [number for number, _, _ in spans] == [1, 3]
        assert text[spans[1][1] : spans[1][2]] == "gamma"

    def test_joining_normalized_pages_matches_normalizing_the_joined_pages(self):
        """The equivalence the offsets depend on: `_extract_pdf_text` joins raw pages with
        newlines and normalizes afterwards, so per-page normalization must land on the same
        string or every offset is measured against text that was never chunked."""
        raw = [(1, "alpha\n  beta "), (2, "   "), (3, "\ngamma\t\tdelta")]
        joined_then_normalized = _normalize_whitespace("\n".join(text for _, text in raw))
        normalized_then_joined, _ = _join_pages([(n, _normalize_whitespace(t)) for n, t in raw])

        assert normalized_then_joined == joined_then_normalized


class TestPagesForSpan:
    SPANS = [(1, 0, 10), (2, 11, 20), (3, 21, 30)]

    def test_a_span_inside_one_page_reports_that_page_twice(self):
        assert _pages_for_span(self.SPANS, 2, 8) == (1, 1)

    def test_a_span_across_a_page_break_reports_both_ends(self):
        assert _pages_for_span(self.SPANS, 8, 25) == (1, 3)

    def test_no_page_map_means_unknown_not_page_one(self):
        # Guessing "1" here would point every non-PDF citation at a page that does not exist.
        assert _pages_for_span([], 0, 100) == (None, None)

    def test_a_span_landing_only_in_the_separator_matches_nothing(self):
        assert _pages_for_span(self.SPANS, 10, 11) == (None, None)

    def test_boundaries_are_half_open(self):
        # Ending exactly where page 2 begins must not drag page 2 in.
        assert _pages_for_span(self.SPANS, 0, 11) == (1, 1)
        assert _pages_for_span(self.SPANS, 11, 12) == (2, 2)


class TestExtractedDocument:
    def test_a_multi_page_pdf_reports_a_span_per_page(self, make_pdf):
        pdf = make_pdf([["Scope and purpose"], ["Normative references"], ["Terms and definitions"]])

        document = _extract_document("standard.pdf", "application/pdf", pdf, "pdf")

        assert [number for number, _, _ in document.page_spans] == [1, 2, 3]
        for number, start, end in document.page_spans:
            excerpt = document.text[start:end]
            assert excerpt == {
                1: "Scope and purpose",
                2: "Normative references",
                3: "Terms and definitions",
            }[number]

    def test_the_text_matches_what_the_plain_extractor_produces(self, make_pdf):
        """`_extract_text` and `_extract_document` must not drift: the chunks are built from
        one and the offsets from the other."""
        pdf = make_pdf([["Alpha beta gamma"], ["Delta epsilon"]])

        document = _extract_document("standard.pdf", "application/pdf", pdf, "pdf")
        plain, failure = _extract_pdf_text(pdf)

        assert failure is None
        assert document.text == _normalize_whitespace(plain)

    def test_a_chunk_can_be_traced_from_its_span_to_a_page(self, make_pdf):
        pdf = make_pdf([["Alpha " * 120], ["Beta " * 120], ["Gamma " * 120]])
        document = _extract_document("standard.pdf", "application/pdf", pdf, "pdf")

        located = [
            (chunk, _pages_for_span(document.page_spans, start, end))
            for chunk, start, end in _split_into_chunk_spans(document.text)
        ]

        assert len(located) > 1
        for chunk, (page_start, page_end) in located:
            assert page_start is not None and page_end is not None
            assert 1 <= page_start <= page_end <= 3
            # The words on the reported pages must actually be in the chunk.
            words = {1: "Alpha", 2: "Beta", 3: "Gamma"}
            assert any(words[page] in chunk for page in range(page_start, page_end + 1))

    def test_an_unreadable_pdf_yields_the_note_and_no_spans(self):
        document = _extract_document("broken.pdf", "application/pdf", b"not a pdf", "pdf")

        assert "could not be read" in document.text
        assert document.page_spans == []

    def test_a_pdf_with_no_text_layer_yields_the_scanned_note_and_no_spans(self, make_pdf):
        # A page with an empty content stream: readable structure, nothing to extract.
        document = _extract_document("scanned.pdf", "application/pdf", make_pdf([[]]), "pdf")

        assert "No embedded text layer" in document.text
        assert document.page_spans == []

    def test_a_spreadsheet_keeps_its_line_structure_for_title_derivation(self):
        """`_derive_title` reads lines and a spreadsheet's title is its first row.

        Normalizing before deriving the title fuses every row into one over-long line, so
        no line falls in the 4–120 char window and the title silently becomes the filename
        — which also changes `contextual_content`, and therefore what retrieval scores.
        """
        csv_bytes = b"Region,Revenue\nNorth,120000\nSouth,98000\n" + b"Filler,1\n" * 40

        document = _extract_document("sales.csv", "text/csv", csv_bytes, "table")

        title = _derive_title("sales.csv", document.title_text)
        assert title.startswith("Row 1:"), "the title should still come from the first row"
        # The regression this guards, stated directly: deriving from the normalized text
        # finds no line inside the 4–120 char window and silently falls back to the filename.
        assert _derive_title("sales.csv", document.text) == "Sales"
        # The chunked text is still the normalized form the offsets index.
        assert "\n" not in document.text

    def test_title_text_matches_the_plain_extractor_for_every_modality(self):
        cases = [
            ("notes.txt", "text/plain", b"Alpha beta", "text"),
            ("sales.csv", "text/csv", b"Region,Revenue\nNorth,1", "table"),
            ("photo.png", "image/png", b"\x89PNG", "image"),
            ("clip.mp3", "audio/mpeg", b"ID3", "audio"),
            ("legacy.doc", "application/msword", b"\xd0\xcf", "text"),
        ]
        for filename, content_type, payload, modality in cases:
            document = _extract_document(filename, content_type, payload, modality)
            assert document.title_text == _extract_text(filename, content_type, payload, modality), filename

    def test_non_pdf_formats_get_offsets_but_no_pages(self):
        document = _extract_document("notes.txt", "text/plain", b"Alpha beta  gamma", "text")

        assert document.text == "Alpha beta gamma"
        assert document.page_spans == []

    def test_pages_carry_their_own_number_so_a_skipped_page_shifts_nothing(self, make_pdf):
        """pypdf can fail on a single page; that page is dropped. If numbering came from
        list position, every later page would be reported one page early."""
        pages, failure = _extract_pdf_pages(make_pdf([["One"], ["Two"], ["Three"]]))

        assert failure is None
        assert [number for number, _ in pages] == [1, 2, 3]
        # Simulating the drop: page 2 unreadable, page 3 must still say 3.
        survived = [(number, text) for number, text in pages if number != 2]
        _, spans = _join_pages([(number, _normalize_whitespace(text)) for number, text in survived])
        assert [number for number, _, _ in spans] == [1, 3]
