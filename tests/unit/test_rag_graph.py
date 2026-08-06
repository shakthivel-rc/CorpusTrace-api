"""Unit tests for the RAG graph helpers.

entity_chunks_by_key is correct *in isolation*; the historical upload-500 was a
CALLER bug in _rebuild_graph (it passed the already-inverted dict). The bug is now
FIXED and TestRebuildGraph below is the regression test that keeps it that way."""
import pytest

from models.rag import DocumentChunk, RagGraphEntity
from rag.service import (
    MAX_ENTITY_NAME_CHARS,
    MAX_FILENAME_CHARS,
    MAX_VARCHAR_CHARS,
    _extract_entities,
    _rebuild_graph,
    _safe_filename,
    entity_chunks_by_key,
)

pytestmark = pytest.mark.unit


class _FakeSession:
    """Records added rows; no real database (the bug fires before any flush)."""

    def __init__(self):
        self.added = []

    def add(self, obj):
        self.added.append(obj)

    def flush(self):
        pass


class TestEntityChunksByKey:
    def test_inverts_chunk_to_entities_into_entity_to_chunks(self):
        result = entity_chunks_by_key({"c1": ["Apple", "Banana"], "c2": ["Apple"]})
        assert result["apple"] == {"c1", "c2"}
        assert result["banana"] == {"c1"}

    def test_lowercases_entity_keys(self):
        result = entity_chunks_by_key({"c1": ["OpenAI"]})
        assert "openai" in result
        assert "OpenAI" not in result

    def test_empty_input_yields_empty_mapping(self):
        assert entity_chunks_by_key({}) == {}


class TestExtractEntities:
    def test_finds_capitalized_entity_phrases(self):
        entities = _extract_entities("Apple Inc released a phone. Microsoft did too.")
        joined = " ".join(entities).lower()
        assert "apple" in joined
        assert "microsoft" in joined

    def test_falls_back_to_top_terms_when_no_capitalized_entities(self):
        # all-lowercase text → no capitalized candidates → title-cased top terms
        entities = _extract_entities("machine learning machine learning models")
        assert len(entities) > 0

    def test_discards_binary_garbage_runs(self):
        """A PDF's hex-encoded UTF-16 text (no PDF parser exists — raw bytes are indexed
        as text) matches the entity regex as one enormous token. Left unbounded it
        overflowed rag_graph_entities.name VARCHAR(255) → MySQL 1406 → HTTP 500 upload."""
        hex_run = "FEFF0034002E00320020200B0055006E006400650072007300740061006E00640069006E00670020" * 3
        entities = _extract_entities(f"Intro text. {hex_run} More text.")
        assert all(len(entity) <= MAX_ENTITY_NAME_CHARS for entity in entities)
        assert hex_run not in entities

    def test_keeps_realistic_entity_names(self):
        entities = _extract_entities("International Organization For Standardization publishes ISO 9001.")
        assert any("International" in entity for entity in entities)


class TestRebuildGraph:
    """Regression test for the upload-500. _rebuild_graph must pass chunk→entities
    (not the already-inverted entity→chunks map) into entity_chunks_by_key, or the
    returned keys become chunk UUIDs and entity_display[key] raises KeyError on every
    document upload. Fails against the buggy call; passes once the argument is fixed."""

    def test_builds_named_graph_entities_without_raising(self):
        chunks = [
            DocumentChunk(id="c1", content="Apple and Microsoft ship software."),
            DocumentChunk(id="c2", content="Microsoft and Google compete in search."),
        ]
        session = _FakeSession()

        _rebuild_graph(session, "resource-1", chunks)

        entities = [obj for obj in session.added if isinstance(obj, RagGraphEntity)]
        assert entities, "expected graph entities to be created"

        names = {entity.name for entity in entities}
        # The entity NAMES must be the human display names, never the chunk ids —
        # that is exactly what the double-inversion bug got wrong.
        assert "c1" not in names and "c2" not in names
        assert "Microsoft" in names

    def test_entity_names_always_fit_the_varchar_column(self):
        """Second upload-500 (MySQL 1406): a binary document produced an entity name far
        longer than rag_graph_entities.name VARCHAR(255). Nothing catches DataError."""
        hex_run = "FEFF0034002E00320020200B0055006E00640065007200730074" * 8
        chunks = [DocumentChunk(id="c1", content=f"Policy document. {hex_run}")]
        session = _FakeSession()

        _rebuild_graph(session, "resource-1", chunks)

        for entity in [obj for obj in session.added if isinstance(obj, RagGraphEntity)]:
            assert len(entity.name) <= MAX_VARCHAR_CHARS


class TestSafeFilename:
    def test_bounds_long_names_to_fit_varchar_columns(self):
        # files.file_url embeds this path, and file_name / source_name are VARCHAR(255).
        safe = _safe_filename("a" * 400 + ".pdf")
        assert len(safe) <= MAX_FILENAME_CHARS

    def test_preserves_the_extension_so_modality_detection_still_works(self):
        assert _safe_filename("b" * 400 + ".pdf").endswith(".pdf")

    def test_leaves_ordinary_names_untouched(self):
        assert _safe_filename("quarterly-report.pdf") == "quarterly-report.pdf"
