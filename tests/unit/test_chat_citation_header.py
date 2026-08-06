"""Unit tests for the citation response header.

The header is how retrieval provenance reaches the browser at all, and it shares a size
budget with the answer itself: a header a proxy rejects does not lose the citations, it
loses the whole response. So the two things asserted here are that it decodes back to what
went in, and that it can never grow without bound.
"""
import base64
import json

import pytest

from controllers.chat_controller import (
    CITATION_HEADER_FIELDS,
    MAX_CITATION_HEADER_CHARS,
    MAX_CITATION_LABEL_CHARS,
    _encode_citations,
)

pytestmark = pytest.mark.unit


def _decode(encoded: str) -> list[dict]:
    return json.loads(base64.urlsafe_b64decode(encoded.encode("ascii")).decode("utf-8"))


def _citation(index: int, **overrides) -> dict:
    citation = {
        "index": index,
        "chunk_id": f"chunk-{index}",
        "file_id": f"file-{index}",
        "source_name": "handbook.pdf",
        "chunk_index": index - 1,
        "modality": "pdf",
        "title": "Vacation policy",
        "score": 0.42,
        "page_start": 4,
        "page_end": 5,
        "char_start": 1200,
        "char_end": 2400,
        "snippet": "Employees receive twenty vacation days per year.",
    }
    citation.update(overrides)
    return citation


class TestEncoding:
    def test_round_trips_the_fields_the_client_needs(self):
        decoded = _decode(_encode_citations([_citation(1)]))

        assert decoded == [
            {
                "index": 1,
                "chunk_id": "chunk-1",
                "file_id": "file-1",
                "source_name": "handbook.pdf",
                "chunk_index": 0,
                "modality": "pdf",
                "page_start": 4,
                "page_end": 5,
                "score": 0.42,
            }
        ]

    def test_document_text_never_reaches_the_header(self):
        """The snippet is quoted document content. It belongs in a response body the user
        asked for, not in a header attached to every answer and logged by every proxy."""
        decoded = _decode(_encode_citations([_citation(1)]))

        assert "snippet" not in decoded[0]
        assert all(key in CITATION_HEADER_FIELDS for key in decoded[0])

    def test_no_citations_means_no_header(self):
        # Small talk and refusals retrieve nothing; sending an empty array would have the
        # client render an empty evidence affordance for an answer with no evidence.
        assert _encode_citations([]) is None

    def test_the_value_is_ascii_even_for_a_non_ascii_filename(self):
        encoded = _encode_citations([_citation(1, source_name="rapport-financiér-2026.pdf")])

        encoded.encode("ascii")  # a header value that raises here breaks the response
        assert _decode(encoded)[0]["source_name"].startswith("rapport-financi")

    def test_a_long_filename_is_truncated_not_dropped(self):
        encoded = _encode_citations([_citation(1, source_name="x" * 500)])

        assert len(_decode(encoded)[0]["source_name"]) == MAX_CITATION_LABEL_CHARS


class TestSizeBudget:
    def test_five_realistic_citations_stay_well_inside_the_cap(self):
        encoded = _encode_citations([_citation(index) for index in range(1, 6)])

        assert len(encoded) <= MAX_CITATION_HEADER_CHARS
        assert len(_decode(encoded)) == 5

    def test_an_oversized_set_sheds_citations_until_it_fits(self):
        oversized = [_citation(index, source_name="n" * MAX_CITATION_LABEL_CHARS) for index in range(1, 200)]

        encoded = _encode_citations(oversized)

        assert len(encoded) <= MAX_CITATION_HEADER_CHARS
        decoded = _decode(encoded)
        assert 0 < len(decoded) < len(oversized)
        # The ones kept are the highest-ranked, in order — dropping from the tail.
        assert [item["index"] for item in decoded] == list(range(1, len(decoded) + 1))
