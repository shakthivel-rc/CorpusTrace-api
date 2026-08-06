"""A lexical index cannot bridge a typo or a synonym, so the refusal has to name the
words that matched nothing — otherwise "try rephrasing" gives the user no way to know
WHICH word let them down, and a one-letter slip ("value" for "valve") reads as "the
document does not cover this".
"""
import json

import pytest

import rag.service as rag_service
from models.rag import DocumentChunk

pytestmark = pytest.mark.unit


def _chunk(terms: dict[str, int]) -> DocumentChunk:
    return DocumentChunk(
        resource_id="r-1",
        chunk_index=0,
        source_name="daytona.pdf",
        modality="pdf",
        title="Service",
        content="text",
        contextual_content="text",
        terms_json=json.dumps(terms),
    )


class TestUnmatchedTerms:
    CHUNKS = [_chunk({"valve": 3, "clearance": 1}), _chunk({"motorcycle": 9, "tire": 4})]

    def test_names_a_typo_and_a_synonym_the_document_does_not_use(self):
        # The real failing question from the Daytona manual.
        out = rag_service._unmatched_terms("what is the value clearence of the bike", self.CHUNKS)
        assert out == ["value", "clearence", "bike"]

    def test_is_silent_when_every_word_matched(self):
        # Here the failure was about how the words were distributed, not which were used;
        # naming them would send the user off to fix something that is not broken.
        assert rag_service._unmatched_terms("valve clearance motorcycle", self.CHUNKS) == []

    def test_ignores_stopwords_so_the_hint_stays_about_content_words(self):
        out = rag_service._unmatched_terms("what is the valve", self.CHUNKS)
        assert out == []

    def test_reports_in_the_order_asked_without_duplicates(self):
        out = rag_service._unmatched_terms("zeta alpha zeta beta", self.CHUNKS)
        assert out == ["zeta", "alpha", "beta"]

    def test_bounds_the_list(self):
        query = " ".join(f"word{i}" for i in range(20))
        out = rag_service._unmatched_terms(query, self.CHUNKS)
        assert len(out) == rag_service.MAX_UNMATCHED_TERMS_REPORTED

    def test_handles_a_query_of_pure_stopwords(self):
        assert rag_service._unmatched_terms("what is the", self.CHUNKS) == []

    def test_handles_an_empty_corpus(self):
        assert rag_service._unmatched_terms("valve", []) == ["valve"]


class TestVocabularyHint:
    def test_singular_and_plural_agree(self):
        assert rag_service._vocabulary_hint(["bike"]).strip().startswith("**bike** does not appear")
        assert "**a**, **b** do not appear" in rag_service._vocabulary_hint(["a", "b"])

    def test_empty_when_nothing_is_unmatched(self):
        assert rag_service._vocabulary_hint([]) == ""
        assert rag_service._vocabulary_hint(None) == ""

    def test_refusal_carries_the_hint(self):
        answer = rag_service._compose_answer(
            "value clearence", [], "contextual_hybrid", "Daytona", unmatched=["value", "clearence"]
        )
        assert 'I could not find anything matching that in "Daytona".' in answer
        assert "**value**, **clearence** do not appear anywhere in this document." in answer
        # The original guidance is kept, not replaced.
        assert "try rephrasing" in answer

    def test_refusal_without_a_hint_is_unchanged_prose(self):
        answer = rag_service._compose_answer("x", [], "contextual_hybrid", "Daytona")
        assert "do not appear" not in answer
        assert "  " not in answer  # no double space where the hint would have gone


class TestRetrievalSituation:
    """Configuring a provider must not downgrade the reply: the LLM is told the same fact."""

    def test_passes_the_words_to_the_conversational_model(self):
        out = rag_service._retrieval_situation(["value", "bike"])
        assert "appear nowhere in it: value, bike" in out

    def test_falls_back_to_the_plain_situation(self):
        assert rag_service._retrieval_situation([]) == (
            "retrieval found no sufficiently relevant passage in the documents"
        )
