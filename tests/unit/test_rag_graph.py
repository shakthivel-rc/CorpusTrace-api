"""Unit tests for the RAG graph helpers.

entity_chunks_by_key is correct *in isolation*; the historical upload-500 was a
CALLER bug in _rebuild_graph (it passed the already-inverted dict). The bug is now
FIXED and TestRebuildGraph below is the regression test that keeps it that way."""
import json

import pytest

from models.rag import DocumentChunk, RagGraphEntity
from rag.service import (
    GRAPH_BOOST_MAX_FRACTION,
    MAX_ENTITY_NAME_CHARS,
    MAX_FILENAME_CHARS,
    MAX_VARCHAR_CHARS,
    _contextual_hybrid,
    _extract_entities,
    _graph_rag,
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


class _EntityQuerySession:
    """Just enough Session for `_graph_rag`, which issues exactly one entity query."""

    def __init__(self, entities):
        self._entities = entities

    def query(self, model):
        assert model is RagGraphEntity, "graph retrieval should query entities only"
        return self

    def filter(self, *_args, **_kwargs):
        return self

    def all(self):
        return list(self._entities)


def _scored_chunk(chunk_id: str, text: str) -> DocumentChunk:
    """A chunk `_contextual_hybrid` can score. `terms_json` is left unset on purpose —
    `_ChunkScoringData` then recomputes the term map from `contextual_content`, which is
    the same map ingestion would have stored."""
    return DocumentChunk(id=chunk_id, content=text, contextual_content=text, title="")


def _graph_entity(name: str, chunk_refs: list[str], weight: int = 1) -> RagGraphEntity:
    return RagGraphEntity(name=name, weight=weight, chunk_refs_json=json.dumps(chunk_refs))


QUERY = "front tyre pressure daytona"
# Shares no term with QUERY, so its lexical score is exactly 0 and whatever score it ends
# up with IS the graph boost — which is what makes the boost directly observable.
UNRELATED_TEXT = "Appendix listing assorted maintenance intervals and workshop notes."
ANSWER_TEXT = "The front tyre pressure for the daytona: front tyre pressure daytona values."


class TestGraphBoostIsBoundedAndRelative:
    """The entity boost must REFINE the lexical ranking, never replace it.

    It used to be `1.5 + min(weight, 5) * 0.1` summed per matched entity with no ceiling,
    on the same numeric scale as a whole lexical score. Measured on a real 783-chunk
    manual, one question matched 38 of 1,207 entities and the boost reached 18.9x the
    entire spread of the lexical top-8: four of five citations arrived on boost alone and
    the passage that answered the question was evicted. Every test here fails against that
    version."""

    def test_a_graph_only_chunk_never_outranks_the_best_lexical_passage(self):
        chunks = [_scored_chunk("answer", ANSWER_TEXT), _scored_chunk("toc", UNRELATED_TEXT)]
        # Twenty entities all pointing at the same unrelated chunk — the table-of-contents
        # shape, where one page mentions every heading in the book.
        session = _EntityQuerySession(
            [_graph_entity(f"Front Fork Inspection {i}", ["toc"], weight=5) for i in range(20)]
        )

        results, notes = _graph_rag(session, "resource-1", QUERY, chunks)

        assert results[0].chunk.id == "answer"
        assert notes.startswith("Graph signals:")

    def test_the_boost_is_capped_at_a_fraction_of_the_top_lexical_score(self):
        chunks = [_scored_chunk("answer", ANSWER_TEXT), _scored_chunk("toc", UNRELATED_TEXT)]
        session = _EntityQuerySession(
            [_graph_entity(f"Daytona Section {i}", ["toc"], weight=5) for i in range(50)]
        )

        results, _ = _graph_rag(session, "resource-1", QUERY, chunks)
        by_id = {result.chunk.id: result.score for result in results}
        top_lexical = _contextual_hybrid(QUERY, chunks, limit=8)[0].score

        # The cap is expressed against the score THIS query produced, so the same code
        # behaves identically on a corpus whose scores run an order of magnitude higher.
        assert by_id["toc"] < GRAPH_BOOST_MAX_FRACTION * top_lexical
        assert by_id["toc"] < by_id["answer"]

    def test_the_boost_saturates_instead_of_accumulating(self):
        chunks = [_scored_chunk("answer", ANSWER_TEXT), _scored_chunk("toc", UNRELATED_TEXT)]

        def boost_with(entity_count: int) -> float:
            session = _EntityQuerySession(
                [_graph_entity(f"Daytona Part {i}", ["toc"]) for i in range(entity_count)]
            )
            results, _ = _graph_rag(session, "resource-1", QUERY, chunks)
            return {result.chunk.id: result.score for result in results}["toc"]

        one, forty = boost_with(1), boost_with(40)

        assert forty > one, "more matched entities must still rank higher"
        # Forty times the entities is nowhere near forty times the evidence: on a document
        # about the Daytona, the subject noun alone appears in 18 entity names, so a raw
        # sum mostly measures how often the document says its own name.
        assert forty < 4 * one

    def test_no_entity_match_leaves_the_lexical_ranking_untouched(self):
        chunks = [_scored_chunk("answer", ANSWER_TEXT), _scored_chunk("other", UNRELATED_TEXT)]
        session = _EntityQuerySession([_graph_entity("Carburettor Overhaul", ["other"])])

        results, notes = _graph_rag(session, "resource-1", QUERY, chunks)
        lexical = _contextual_hybrid(QUERY, chunks, limit=5)

        assert [r.chunk.id for r in results] == [r.chunk.id for r in lexical]
        assert [r.score for r in results] == [r.score for r in lexical]
        assert notes == ""

    def test_the_graph_can_still_surface_a_chunk_lexical_retrieval_missed(self):
        """The capability the boost exists for. Bounding it must not remove it."""
        chunks = [_scored_chunk("answer", ANSWER_TEXT), _scored_chunk("hidden", UNRELATED_TEXT)]
        session = _EntityQuerySession([_graph_entity("Daytona Tyre Chart", ["hidden"], weight=4)])

        results, _ = _graph_rag(session, "resource-1", QUERY, chunks)
        surfaced = {result.chunk.id: result for result in results}

        assert "hidden" in surfaced, "a chunk with no lexical score must still be reachable"
        assert surfaced["hidden"].reason == "graph entity expansion"
        assert surfaced["hidden"].score > 0
