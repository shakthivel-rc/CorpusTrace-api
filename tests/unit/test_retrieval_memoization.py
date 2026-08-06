"""The per-request scoring memo.

Retrieval used to re-parse every chunk's `terms_json` and re-lowercase every passage on
each scoring pass, and the modes make many passes: `rag_fusion` scores the corpus once per
query variant, `corrective` may run hybrid and then fusion, and `agentic_rag` runs four of
those as tools. `_ChunkScoringData` derives that work once per chunk and hangs it off the
chunk instance.

What these tests protect is not the speed — it is the two properties that make the speed
safe: the memo must produce byte-identical scores, and it must never survive a change to
the row it was derived from.
"""
import json
import math
from collections import Counter

import pytest

from models.rag import DocumentChunk
from rag.service import (
    _chunk_term_set,
    _contextual_hybrid,
    _load_terms,
    _score_chunk,
    _scoring_data,
    _term_counts,
    _tokenize,
    _unmatched_terms,
)

pytestmark = pytest.mark.unit


def _chunk(text: str, title: str | None = "Valve Timing", terms: str | None = None) -> DocumentChunk:
    contextual = (
        f"Resource: Manual\nSource file: manual.pdf\nTitle: {title}\n"
        f"Modality: pdf\nChunk: 0\nContent:\n{text}"
    )
    return DocumentChunk(
        id="chunk-1",
        resource_id="res-1",
        file_id="file-1",
        chunk_index=0,
        source_name="manual.pdf",
        modality="pdf",
        title=title,
        content=text,
        contextual_content=contextual,
        terms_json=terms if terms is not None else json.dumps(_term_counts(contextual)),
    )


class TestMemoCorrectness:
    def test_scoring_matches_the_unmemoized_formula(self):
        chunk = _chunk("valve clearance is checked at every service interval")
        query = "valve clearance"
        query_terms = Counter(_tokenize(query))

        chunk_terms = Counter(json.loads(chunk.terms_json))
        overlap = set(query_terms).intersection(chunk_terms)
        lexical = sum(chunk_terms[t] * query_terms[t] for t in overlap)
        coverage = len(overlap) / len(query_terms)
        length_norm = 1 / math.sqrt(max(sum(chunk_terms.values()), 1))
        exact = 2.0 if query.lower() in chunk.contextual_content.lower() else 0.0
        title = 0.5 if any(t in chunk.title.lower() for t in overlap) else 0.0
        expected = (lexical * length_norm) + (coverage * 3.0) + exact + title

        assert _score_chunk(query, query_terms, chunk) == expected

    def test_repeated_scoring_returns_the_same_score(self):
        """The second pass reads the memo; it must not drift from the first."""
        chunk = _chunk("valve clearance and torque specification")
        query_terms = Counter(_tokenize("valve torque"))
        first = _score_chunk("valve torque", query_terms, chunk)
        assert all(_score_chunk("valve torque", query_terms, chunk) == first for _ in range(5))

    def test_precomputed_query_lower_matches_deriving_it(self):
        """`_contextual_hybrid` hoists `query.lower()`; passing it must change nothing."""
        chunk = _chunk("Valve Clearance Specification")
        query = "Valve Clearance"
        terms = Counter(_tokenize(query))
        assert _score_chunk(query, terms, chunk, query_lower=query.lower()) == _score_chunk(
            query, terms, chunk
        )

    def test_falls_back_to_tokenizing_when_terms_json_is_unreadable(self):
        chunk = _chunk("valve clearance measured cold", terms="{not json at all")
        assert _load_terms(chunk) == Counter(_tokenize(chunk.contextual_content))
        assert _score_chunk("valve", Counter(_tokenize("valve")), chunk) > 0

    def test_falls_back_when_terms_json_is_valid_json_but_not_an_object(self):
        """`json.loads` succeeds on a bare list and raises nothing — the type check catches it."""
        chunk = _chunk("valve clearance measured cold", terms="[1, 2, 3]")
        assert _load_terms(chunk) == Counter(_tokenize(chunk.contextual_content))
        assert _score_chunk("valve", Counter(_tokenize("valve")), chunk) > 0

    def test_chunk_term_set_equals_tokenizing_the_passage(self):
        """The substitution made in `_has_sufficient_evidence` and `_grade_evidence`."""
        chunk = _chunk("valve clearance torque specification interval")
        assert _chunk_term_set(chunk) == set(_tokenize(chunk.contextual_content))

    def test_chunk_term_set_still_equals_tokenizing_when_terms_json_is_broken(self):
        chunk = _chunk("valve clearance torque", terms=None)
        chunk.terms_json = "not json"
        assert _chunk_term_set(chunk) == set(_tokenize(chunk.contextual_content))

    def test_load_terms_returns_a_counter_not_the_stored_dict(self):
        """Callers get a Counter, and mutating it must not corrupt the memo."""
        chunk = _chunk("valve clearance")
        terms = _load_terms(chunk)
        assert isinstance(terms, Counter)
        terms["valve"] = 9999
        assert _load_terms(chunk)["valve"] != 9999


class TestMemoCannotGoStale:
    def test_memo_is_rebuilt_when_terms_json_changes(self):
        """The identity guard. A session refresh replaces `terms_json` with a new object;
        the memo derived from the old one must not answer for the new."""
        chunk = _chunk("valve clearance")
        assert "valve" in _load_terms(chunk)

        chunk.terms_json = json.dumps({"motorcycle": 3})
        assert _load_terms(chunk) == Counter({"motorcycle": 3})
        assert "valve" not in _load_terms(chunk)

    def test_rebuilt_memo_recomputes_the_length_norm(self):
        chunk = _chunk("valve clearance")
        before = _scoring_data(chunk).length_norm
        chunk.terms_json = json.dumps({"a": 1, "b": 1, "c": 1, "d": 1, "e": 1, "f": 1, "g": 1, "h": 1})
        assert _scoring_data(chunk).length_norm != before
        assert _scoring_data(chunk).length_norm == 1 / math.sqrt(8)

    def test_rebuilt_memo_drops_the_cached_lowered_text(self):
        chunk = _chunk("valve clearance")
        _score_chunk("valve", Counter(_tokenize("valve")), chunk)  # populates lowered_content
        chunk.contextual_content = "Content:\nMOTORCYCLE CHAIN TENSION"
        chunk.terms_json = json.dumps(_term_counts(chunk.contextual_content))
        data = _scoring_data(chunk)
        assert data.lowered_content(chunk) == "content:\nmotorcycle chain tension"

    def test_memo_is_reused_while_the_row_is_unchanged(self):
        chunk = _chunk("valve clearance")
        assert _scoring_data(chunk) is _scoring_data(chunk)


class TestMemoIsInvisibleToCallers:
    def test_hybrid_ranking_is_unaffected_by_a_warm_memo(self):
        chunks = [
            _chunk("valve clearance specification", title="A"),
            _chunk("unrelated chain tension", title="B"),
        ]
        chunks[1].id = "chunk-2"
        cold = [(r.chunk.id, r.score) for r in _contextual_hybrid("valve clearance", chunks)]
        warm = [(r.chunk.id, r.score) for r in _contextual_hybrid("valve clearance", chunks)]
        assert cold == warm
        assert cold[0][0] == "chunk-1"

    def test_unmatched_terms_still_reports_words_absent_from_every_chunk(self):
        chunks = [_chunk("valve clearance specification")]
        assert _unmatched_terms("valve motorcycle", chunks) == ["motorcycle"]
        # Second call reads the memo and must agree.
        assert _unmatched_terms("valve motorcycle", chunks) == ["motorcycle"]

    def test_lowered_forms_are_not_built_for_a_chunk_with_no_overlap(self):
        """The memo defers lowering precisely so a non-matching corpus costs nothing."""
        chunk = _chunk("chain tension adjustment")
        assert _score_chunk("motorcycle", Counter(_tokenize("motorcycle")), chunk) == 0.0
        assert _scoring_data(chunk)._lowered_content is None
