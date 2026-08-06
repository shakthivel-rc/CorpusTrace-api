"""Unit tests for lexical tokenization and chunk scoring — the ONLY relevance
mechanism in the system (there is no vector store / no embeddings), so this
arithmetic is the sole defence against silent retrieval-quality regressions."""
import json
from collections import Counter

import pytest

from models.rag import DocumentChunk
from rag.service import _score_chunk, _tokenize

pytestmark = pytest.mark.unit


def _chunk(terms: dict[str, int], content: str, title: str = "") -> DocumentChunk:
    return DocumentChunk(terms_json=json.dumps(terms), contextual_content=content, title=title)


class TestTokenize:
    def test_lowercases_and_keeps_content_words(self):
        tokens = _tokenize("Machine Learning Basics")
        assert tokens == ["machine", "learning", "basics"]

    def test_drops_stopwords_and_single_characters(self):
        tokens = _tokenize("Cats and dogs a X")
        assert "and" not in tokens  # stopword
        assert "cats" in tokens and "dogs" in tokens
        assert all(len(t) > 1 for t in tokens)  # single chars filtered by the regex


class TestScoreChunk:
    def test_zero_when_no_terms_overlap(self):
        chunk = _chunk({"apple": 3}, "apple pie recipe")
        score = _score_chunk("banana", Counter(_tokenize("banana")), chunk)
        assert score == 0.0

    def test_zero_for_an_empty_query(self):
        chunk = _chunk({"apple": 3}, "apple pie recipe")
        assert _score_chunk("", Counter(), chunk) == 0.0

    def test_positive_when_terms_overlap(self):
        chunk = _chunk({"machine": 2, "learning": 1}, "machine learning basics")
        score = _score_chunk("machine learning", Counter(_tokenize("machine learning")), chunk)
        assert score > 0

    def test_exact_phrase_match_scores_higher(self):
        query = "machine learning"
        query_terms = Counter(_tokenize(query))
        exact = _chunk({"machine": 1, "learning": 1}, "an intro to machine learning today")
        scattered = _chunk({"machine": 1, "learning": 1}, "machine foo bar learning baz")
        assert _score_chunk(query, query_terms, exact) > _score_chunk(query, query_terms, scattered)

    def test_falls_back_to_tokenizing_content_on_bad_terms_json(self):
        # _load_terms recovers by tokenizing contextual_content when terms_json is invalid.
        chunk = DocumentChunk(terms_json="not-json", contextual_content="machine learning", title="")
        score = _score_chunk("machine learning", Counter(_tokenize("machine learning")), chunk)
        assert score > 0
