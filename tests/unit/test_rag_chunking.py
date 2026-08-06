"""Unit tests for the character-based chunker and whitespace normalizer.
Boundary-condition-dense and dependency-free — the ideal unit-test target."""
import pytest

from rag.service import _normalize_whitespace, _split_into_chunks

pytestmark = pytest.mark.unit


def test_normalize_collapses_all_whitespace_runs():
    assert _normalize_whitespace("a\n\n  b\t c ") == "a b c"


def test_text_within_limit_yields_a_single_chunk():
    assert list(_split_into_chunks("hello world")) == ["hello world"]


def test_long_text_splits_into_bounded_overlapping_chunks():
    text = "word " * 500  # ~2500 chars after normalization
    chunks = list(_split_into_chunks(text, max_chars=1200, overlap=180))
    assert len(chunks) >= 2
    assert all(len(c) <= 1200 for c in chunks)


def test_prefers_a_sentence_boundary_past_the_halfway_point():
    # The ". " sits at index 700 — beyond start + max_chars//2 (600) — so the first
    # chunk should end at that sentence boundary rather than the hard 1200 cut.
    text = "A" * 700 + ". " + "B" * 700
    chunks = list(_split_into_chunks(text, max_chars=1200, overlap=50))
    assert chunks[0].endswith(".")
    assert "B" not in chunks[0]


def test_ignores_a_boundary_before_the_halfway_point():
    # ". " at index 100 is before the halfway point (600), so it is NOT used as the cut.
    text = "A" * 100 + ". " + "B" * 1500
    chunks = list(_split_into_chunks(text, max_chars=1200, overlap=50))
    assert len(chunks[0]) > 200  # did not cut early at index ~101
