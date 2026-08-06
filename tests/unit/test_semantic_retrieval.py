"""Semantic retrieval — the embedding side of `rag/service.py`.

Lexical scoring cannot bridge a synonym: a question about a "bike" scores 0.0 against a
document that says "motorcycle", and that is indistinguishable from the document not
covering the topic. This layer adds cosine similarity over per-chunk vectors to close that
gap, and it is bolted on to a system that has answered questions without it for its whole
life. So the properties worth pinning are mostly about what it must NOT do:

* a knowledge base with no embeddings must come out of `_blend_semantic` byte-identical to
  what the mode returned — that is what makes this safe to enable per document;
* the sufficiency gate must only ever WIDEN — enabling embeddings on one document may never
  make the system refuse a question it used to answer;
* vectors from two different models must never be compared — cosine between them is a
  number with no meaning that ranks with complete confidence;
* a provider outage must cost the synonym bridging, not the answer;
* one question must cost exactly one embedding call, whatever the corpus looks like.

Nothing here talks to a provider: `rag.service.embed_texts` is the single seam and every
test replaces it with vectors it chose itself.
"""
import json
import math
from collections import Counter
from itertools import count

import pytest

import rag.service as rag_service
from models.file import File
from models.rag import DocumentChunk
from models.resource import Resource
from rag.service import (
    MIN_SEMANTIC_SIMILARITY,
    SEMANTIC_CANDIDATES,
    RetrievalResult,
    _blend_semantic,
    _contextual_hybrid,
    _cosine,
    _has_sufficient_evidence,
    _score_chunk,
    _semantic_retrieval,
    _SemanticRetrieval,
    _tokenize,
)
from services.llm_provider import LlmProviderError

pytestmark = pytest.mark.unit

OWNER = "user-1"
MODEL = "text-embedding-3-small"

# Every vector below lives in the same 3-dimensional space and is a unit vector, so the
# cosine against QUERY_VECTOR is exactly the number asked for — the tests can name a
# similarity instead of reverse-engineering one.
QUERY_VECTOR = [1.0, 0.0, 0.0]


def _vector(similarity: float) -> list[float]:
    """A unit vector whose cosine against QUERY_VECTOR equals `similarity`."""
    return [similarity, math.sqrt(max(0.0, 1.0 - similarity * similarity)), 0.0]


# ---------------------------------------------------------------------------
# Seeding helpers — rows are built directly rather than through upload so the vectors
# are exactly the ones the assertion talks about.
# ---------------------------------------------------------------------------


def _resource(db, name: str = "Daytona", owner: str = OWNER) -> Resource:
    resource = Resource(resource_name=name, user_id=owner, upload_status=True)
    db.add(resource)
    db.flush()
    return resource


def _file(db, resource: Resource, name: str, provider: str | None = None, model: str | None = None) -> File:
    file_row = File(
        file_name=name,
        file_type="text/plain",
        file_url=f"/tmp/{resource.id}/{name}",
        resource_id=resource.id,
        embedding_provider=provider,
        embedding_model=model,
    )
    db.add(file_row)
    db.flush()
    return file_row


def _chunk(
    db,
    resource: Resource,
    file_row: File,
    text: str,
    index: int = 0,
    vector: list[float] | None = None,
    model: str | None = None,
) -> DocumentChunk:
    chunk = DocumentChunk(
        resource_id=resource.id,
        file_id=file_row.id,
        chunk_index=index,
        source_name=file_row.file_name,
        modality="text",
        title=None,
        content=text,
        contextual_content=text,
        terms_json=json.dumps(dict.fromkeys(_tokenize(text), 1)),
    )
    if vector is not None:
        chunk.embedding_json = json.dumps(vector)
        chunk.embedding_model = model if model is not None else file_row.embedding_model
        chunk.embedding_dim = len(vector)
    db.add(chunk)
    db.flush()
    return chunk


_detached_ids = count()


def _detached_chunk(text: str, vector: list[float] | None = None, model: str | None = MODEL) -> DocumentChunk:
    """A chunk with no DB row — enough for the pure scoring functions.

    The id comes from a counter, not from the text: fusion keys on `chunk.id`, so two
    chunks sharing one would silently merge into a single fused result.
    """
    chunk = DocumentChunk(
        content=text,
        contextual_content=text,
        terms_json=json.dumps(dict.fromkeys(_tokenize(text), 1)),
        title=None,
    )
    chunk.id = f"chunk-{next(_detached_ids)}"
    if vector is not None:
        chunk.embedding_json = json.dumps(vector)
        chunk.embedding_model = model
        chunk.embedding_dim = len(vector)
    return chunk


def _stub_embed(monkeypatch, vectors=None, error: Exception | None = None) -> list[list[str]]:
    """Replace the one seam and record what it was asked to embed."""
    calls: list[list[str]] = []

    def fake_embed_texts(db, user_id, provider, model_id, texts):
        calls.append(list(texts))
        if error is not None:
            raise error
        return [QUERY_VECTOR] * len(texts) if vectors is None else vectors

    monkeypatch.setattr(rag_service, "embed_texts", fake_embed_texts)
    return calls


# ---------------------------------------------------------------------------


class TestBlendingIsInertWithoutEmbeddings:
    """The central safety property: no semantic side means no change at all."""

    def test_a_base_with_no_embeddings_gets_the_very_same_list_back(self):
        """The guarantee the whole feature rests on. If this ever returns a rebuilt list,
        every knowledge base that predates embeddings — which is all of them — silently
        gets a different ranking, different citations and different answers on an upgrade
        that changed nothing about their documents."""
        results = [
            RetrievalResult(chunk=_detached_chunk("valve clearance is checked cold"), score=3.0, reason="lexical"),
            RetrievalResult(chunk=_detached_chunk("torque the cover to 10 Nm"), score=1.0, reason="lexical"),
        ]

        blended = _blend_semantic(results, _SemanticRetrieval())

        assert blended is results
        assert [item.reason for item in blended] == ["lexical", "lexical"]
        assert [item.score for item in blended] == [3.0, 1.0]

    def test_an_empty_ranking_stays_empty(self):
        # A refusal must stay a refusal; fusion must not invent a result out of an empty side.
        assert _blend_semantic([], _SemanticRetrieval()) == []

    def test_a_similarity_with_no_passages_behind_it_is_still_inert(self):
        """`active` is "are there results", not "is there a number". A stray top_similarity
        with nothing to rank must not send the ranking through fusion and reorder it."""
        results = [RetrievalResult(chunk=_detached_chunk("valve clearance"), score=3.0, reason="lexical")]

        assert _blend_semantic(results, _SemanticRetrieval(results=[], top_similarity=0.99)) is results


class TestBlending:
    def test_a_passage_found_by_both_sides_is_cited_once_and_ranks_first(self):
        """RRF sums the two ranks. A chunk listed twice would be shown to the user as two
        separate sources saying the same thing, and would burn two of the five citations."""
        both = _detached_chunk("motorcycle valve clearance is checked cold")
        lexical_only = _detached_chunk("torque the engine cover to ten newton metres")
        semantic_only = _detached_chunk("two-wheeler drivetrain service interval")

        results = [
            RetrievalResult(chunk=both, score=3.0, reason="hybrid lexical/contextual match"),
            RetrievalResult(chunk=lexical_only, score=1.0, reason="hybrid lexical/contextual match"),
        ]
        semantic = _SemanticRetrieval(
            results=[
                RetrievalResult(chunk=both, score=0.91, reason="semantic similarity 0.91"),
                RetrievalResult(chunk=semantic_only, score=0.80, reason="semantic similarity 0.80"),
            ],
            top_similarity=0.91,
            model=MODEL,
        )

        blended = _blend_semantic(results, semantic)

        ids = [item.chunk.id for item in blended]
        assert ids.count(both.id) == 1
        assert ids[0] == both.id
        assert set(ids) == {both.id, lexical_only.id, semantic_only.id}

    def test_the_fused_reason_names_both_signals(self):
        """A chunk cited with no term overlap looks like a bug in the logs unless the
        reason says a vector put it there."""
        chunk = _detached_chunk("motorcycle service schedule")
        blended = _blend_semantic(
            [RetrievalResult(chunk=chunk, score=2.0, reason="hybrid lexical/contextual match")],
            _SemanticRetrieval(
                results=[RetrievalResult(chunk=chunk, score=0.9, reason="semantic similarity 0.90")],
                top_similarity=0.9,
                model=MODEL,
            ),
        )

        assert "hybrid" in blended[0].reason and "semantic" in blended[0].reason

    def test_the_fused_list_is_capped_so_it_still_fits_five_citations(self):
        """Citations are `results[:5]`; a longer fused list would be silently truncated
        somewhere else instead of being ranked here."""
        lexical = [
            RetrievalResult(chunk=_detached_chunk(f"lexical passage {index}"), score=5.0 - index, reason="lex")
            for index in range(5)
        ]
        semantic = _SemanticRetrieval(
            results=[
                RetrievalResult(chunk=_detached_chunk(f"semantic passage {index}"), score=0.9, reason="sem")
                for index in range(5)
            ],
            top_similarity=0.9,
            model=MODEL,
        )

        assert len(_blend_semantic(lexical, semantic)) == 5


class TestSufficiencyOnlyWidens:
    """`_has_sufficient_evidence` gained an OR branch, never a new requirement.

    This is the asymmetry that makes the feature safe to turn on for one document in one
    knowledge base: it can license an answer that lexical coverage would have refused, and
    it can never withdraw one that lexical coverage already allowed. Enabling embeddings
    must never turn a working answer into "I do not have enough information".
    """

    QUERY = "how often should the bike be serviced"

    def _low_coverage_case(self):
        """A passage a lexical reader would refuse: it answers the question and shares
        none of its words."""
        chunk = _detached_chunk("Motorcycle drivetrain maintenance intervals are listed per model year")
        results = [RetrievalResult(chunk=chunk, score=0.05, reason="semantic similarity 0.90")]
        # Stated rather than assumed: this is genuinely below the 0.35 lexical floor.
        query_terms = set(_tokenize(self.QUERY))
        coverage = len(query_terms & set(_tokenize(chunk.contextual_content))) / len(query_terms)
        assert coverage < 0.35
        return chunk, results

    def test_a_passage_with_no_shared_words_is_refused_on_lexical_coverage_alone(self):
        _, results = self._low_coverage_case()
        assert _has_sufficient_evidence(self.QUERY, results) is False

    def test_a_strong_vector_licenses_the_same_passage(self):
        # The synonym case, at the gate: without this branch the answer the user wanted
        # is retrieved, ranked first, and then thrown away by a word-overlap test.
        chunk, results = self._low_coverage_case()
        semantic = _SemanticRetrieval(results=results, top_similarity=0.90, model=MODEL)

        assert _has_sufficient_evidence(self.QUERY, results, semantic) is True

    def test_a_weak_vector_cannot_license_an_answer_on_its_own(self):
        """Below the floor a similarity still helps ranking but must not stand in for
        evidence — otherwise every question gets an answer, because some chunk is always
        the nearest one."""
        _, results = self._low_coverage_case()
        weak = _SemanticRetrieval(
            results=results, top_similarity=MIN_SEMANTIC_SIMILARITY - 0.01, model=MODEL
        )

        assert _has_sufficient_evidence(self.QUERY, results, weak) is False

    def test_the_floor_itself_is_enough(self):
        # The comparison is `>=`; a passage sitting exactly on the documented floor is in.
        _, results = self._low_coverage_case()
        at_floor = _SemanticRetrieval(results=results, top_similarity=MIN_SEMANTIC_SIMILARITY, model=MODEL)

        assert _has_sufficient_evidence(self.QUERY, results, at_floor) is True

    def test_a_question_that_already_passed_lexically_still_passes_with_an_inert_semantic(self):
        """The additive property from the other direction. Every existing base passes
        `semantic=_SemanticRetrieval()` on every question; if that argument could ever
        subtract, turning the feature on would break knowledge bases that never used it."""
        chunk = _detached_chunk("Valve clearance is checked cold on this engine")
        results = _contextual_hybrid("valve clearance", [chunk])
        assert results and results[0].score > 0

        assert _has_sufficient_evidence("valve clearance", results) is True
        assert _has_sufficient_evidence("valve clearance", results, _SemanticRetrieval()) is True
        assert _has_sufficient_evidence("valve clearance", results, None) is True

    def test_nothing_retrieved_is_still_a_refusal_however_strong_the_vector(self):
        """`if not results` comes first on purpose. A similarity score is a property of
        passages that were found; with none found there is nothing to cite and nothing to
        ground an answer in."""
        semantic = _SemanticRetrieval(results=[], top_similarity=0.99, model=MODEL)
        assert _has_sufficient_evidence(self.QUERY, [], semantic) is False


class TestCosine:
    """Every malformed input must score 0.0, not "a number".

    A cosine is a confident, well-behaved-looking float in [-1, 1]. Computing one from
    mismatched or corrupt data does not raise and does not look wrong — it ranks a random
    passage first and the answer is then written from it, with a citation.
    """

    def test_a_matching_vector_scores_its_similarity(self):
        # The baseline the MIN_SEMANTIC_SIMILARITY floor is calibrated against.
        chunk = _detached_chunk("motorcycle service", vector=_vector(0.9))
        assert _cosine(QUERY_VECTOR, 1.0, chunk) == pytest.approx(0.9)

    def test_a_dimension_mismatch_scores_zero_rather_than_a_plausible_number(self):
        """Two models with the same name in different generations produce different
        dimensions. Scoring the overlap would rank on a prefix of a vector."""
        chunk = _detached_chunk("motorcycle service", vector=[1.0, 0.0])
        assert _cosine(QUERY_VECTOR, 1.0, chunk) == 0.0

    def test_malformed_json_scores_zero(self):
        chunk = _detached_chunk("motorcycle service")
        chunk.embedding_json = "not-json"
        assert _cosine(QUERY_VECTOR, 1.0, chunk) == 0.0

    def test_a_chunk_with_no_embedding_scores_zero(self):
        # The overwhelmingly common case: an unembedded chunk in a partially embedded base.
        chunk = _detached_chunk("motorcycle service")
        assert chunk.embedding_json is None
        assert _cosine(QUERY_VECTOR, 1.0, chunk) == 0.0

    def test_a_json_object_instead_of_a_list_scores_zero(self):
        chunk = _detached_chunk("motorcycle service")
        chunk.embedding_json = json.dumps({"vector": [1.0, 0.0, 0.0]})
        assert _cosine(QUERY_VECTOR, 1.0, chunk) == 0.0

    def test_a_zero_magnitude_vector_scores_zero_instead_of_dividing_by_zero(self):
        """A provider that returns an all-zero vector for an empty or unsupported passage
        would otherwise raise ZeroDivisionError mid-retrieval — an unhandled 500 on a chat
        request, since there is no global exception handler."""
        chunk = _detached_chunk("motorcycle service", vector=[0.0, 0.0, 0.0])
        assert _cosine(QUERY_VECTOR, 1.0, chunk) == 0.0

    def test_non_numeric_values_score_zero(self):
        chunk = _detached_chunk("motorcycle service")
        chunk.embedding_json = json.dumps(["a", "b", "c"])
        assert _cosine(QUERY_VECTOR, 1.0, chunk) == 0.0


class TestSemanticRetrieval:
    def test_returns_passages_ranked_by_similarity(self, db, monkeypatch):
        resource = _resource(db)
        file_row = _file(db, resource, "manual.txt", provider="openai", model=MODEL)
        near = _chunk(db, resource, file_row, "motorcycle service", 0, _vector(0.95))
        far = _chunk(db, resource, file_row, "paint colour options", 1, _vector(0.30))
        _stub_embed(monkeypatch)

        semantic = _semantic_retrieval(db, OWNER, resource.id, "bike servicing", [near, far])

        assert [item.chunk.id for item in semantic.results] == [near.id, far.id]
        assert semantic.top_similarity == pytest.approx(0.95)
        assert semantic.model == MODEL
        assert semantic.active is True

    def test_a_base_with_no_embedded_documents_never_calls_the_provider(self, db, monkeypatch):
        """The default state of every knowledge base. Embedding the question anyway would
        bill an API call per question for a feature nobody enabled."""
        resource = _resource(db)
        file_row = _file(db, resource, "manual.txt")
        chunk = _chunk(db, resource, file_row, "valve clearance is checked cold")
        calls = _stub_embed(monkeypatch)

        semantic = _semantic_retrieval(db, OWNER, resource.id, "valve clearance", [chunk])

        assert calls == []
        assert semantic.active is False
        assert semantic.results == [] and semantic.top_similarity == 0.0

    def test_chunks_embedded_with_another_model_are_skipped(self, db, monkeypatch):
        """Cosine between two models' vectors is a number with no meaning that ranks with
        full confidence. The stranger here is deliberately the *most* similar chunk in the
        base — if the model check regresses, it goes straight to the top and gets cited."""
        resource = _resource(db)
        dominant_file = _file(db, resource, "manual.txt", provider="openai", model=MODEL)
        dominant = [
            _chunk(db, resource, dominant_file, f"motorcycle service step {index}", index, _vector(0.70))
            for index in range(3)
        ]
        other_file = _file(db, resource, "legacy.txt", provider="openai", model="ancient-embedding-v1")
        stranger = _chunk(db, resource, other_file, "unrelated legacy passage", 9, _vector(0.99), model="ancient-embedding-v1")
        _stub_embed(monkeypatch)

        semantic = _semantic_retrieval(db, OWNER, resource.id, "bike servicing", dominant + [stranger])

        assert stranger.id not in {item.chunk.id for item in semantic.results}
        assert {item.chunk.id for item in semantic.results} == {chunk.id for chunk in dominant}
        assert semantic.top_similarity == pytest.approx(0.70)

    def test_the_model_covering_the_most_chunks_wins_when_a_base_mixes_two(self, db, monkeypatch):
        """One embedding call per question means one model per question. Picking by chunk
        count keeps the majority of the base searchable semantically; the minority still
        participates lexically, exactly as it would with no embeddings at all."""
        resource = _resource(db)
        minority_file = _file(db, resource, "small.txt", provider="openai", model="minority-model")
        minority = [_chunk(db, resource, minority_file, "minority passage", 0, _vector(0.9))]
        majority_file = _file(db, resource, "big.txt", provider="ollama", model="majority-model")
        majority = [
            _chunk(db, resource, majority_file, f"majority passage {index}", index + 1, _vector(0.8))
            for index in range(4)
        ]
        recorded: list[tuple[str, str]] = []

        def fake_embed_texts(db_, user_id, provider, model_id, texts):
            recorded.append((provider, model_id))
            return [QUERY_VECTOR]

        monkeypatch.setattr(rag_service, "embed_texts", fake_embed_texts)

        semantic = _semantic_retrieval(db, OWNER, resource.id, "bike servicing", minority + majority)

        assert recorded == [("ollama", "majority-model")]
        assert semantic.model == "majority-model"
        assert {item.chunk.id for item in semantic.results} == {chunk.id for chunk in majority}

    def test_a_provider_outage_costs_the_bridging_not_the_answer(self, db, monkeypatch):
        """A revoked key, a rate limit or a dead endpoint must degrade to the lexical
        behaviour the system had before embeddings existed — never a 500 on a question the
        documents could answer."""
        resource = _resource(db)
        file_row = _file(db, resource, "manual.txt", provider="openai", model=MODEL)
        chunk = _chunk(db, resource, file_row, "motorcycle service", 0, _vector(0.95))
        _stub_embed(monkeypatch, error=LlmProviderError("429 rate limited", 429))

        semantic = _semantic_retrieval(db, OWNER, resource.id, "bike servicing", [chunk])

        assert semantic.active is False
        assert semantic.results == []
        assert semantic.top_similarity == 0.0
        # And the gate is then exactly the pre-feature gate.
        assert _blend_semantic([], semantic) == []

    def test_a_provider_returning_nothing_is_inert(self, db, monkeypatch):
        resource = _resource(db)
        file_row = _file(db, resource, "manual.txt", provider="openai", model=MODEL)
        chunk = _chunk(db, resource, file_row, "motorcycle service", 0, _vector(0.95))
        _stub_embed(monkeypatch, vectors=[])

        assert _semantic_retrieval(db, OWNER, resource.id, "bike servicing", [chunk]).active is False

    def test_a_zero_magnitude_query_vector_is_inert(self, db, monkeypatch):
        """Guarding here rather than in `_cosine` alone: `query_norm` is the divisor for
        every chunk in the resource, so one bad question vector would be a ZeroDivisionError
        per chunk, i.e. an unhandled 500."""
        resource = _resource(db)
        file_row = _file(db, resource, "manual.txt", provider="openai", model=MODEL)
        chunk = _chunk(db, resource, file_row, "motorcycle service", 0, _vector(0.95))
        _stub_embed(monkeypatch, vectors=[[0.0, 0.0, 0.0]])

        assert _semantic_retrieval(db, OWNER, resource.id, "bike servicing", [chunk]).active is False

    def test_passages_with_no_similarity_are_dropped(self, db, monkeypatch):
        """An orthogonal or opposed vector is not weak evidence, it is no evidence — and
        carrying it into the fusion would give it a reciprocal rank anyway."""
        resource = _resource(db)
        file_row = _file(db, resource, "manual.txt", provider="openai", model=MODEL)
        hit = _chunk(db, resource, file_row, "motorcycle service", 0, _vector(0.9))
        orthogonal = _chunk(db, resource, file_row, "unrelated", 1, [0.0, 1.0, 0.0])
        opposed = _chunk(db, resource, file_row, "opposite", 2, [-1.0, 0.0, 0.0])
        _stub_embed(monkeypatch)

        semantic = _semantic_retrieval(db, OWNER, resource.id, "bike servicing", [hit, orthogonal, opposed])

        assert [item.chunk.id for item in semantic.results] == [hit.id]

    def test_only_the_top_candidates_enter_the_fusion(self, db, monkeypatch):
        """Reciprocal-rank fusion weighs the two lists by their length as much as their
        quality; an unbounded semantic list would swamp whatever the chosen mode found."""
        resource = _resource(db)
        file_row = _file(db, resource, "manual.txt", provider="openai", model=MODEL)
        chunks = [
            _chunk(db, resource, file_row, f"motorcycle service step {index}", index, _vector(0.9 - index * 0.01))
            for index in range(SEMANTIC_CANDIDATES + 5)
        ]
        _stub_embed(monkeypatch)

        semantic = _semantic_retrieval(db, OWNER, resource.id, "bike servicing", chunks)

        assert len(semantic.results) == SEMANTIC_CANDIDATES
        assert [item.chunk.id for item in semantic.results] == [chunk.id for chunk in chunks[:SEMANTIC_CANDIDATES]]


class TestOneEmbeddingCallPerQuestion:
    def test_the_whole_corpus_costs_a_single_call_embedding_only_the_question(self, db, monkeypatch):
        """Per-chunk or per-model embedding of the *question* would be a per-question API
        bill that scales with the size of the knowledge base — and the retrieval walk
        already loads every chunk, so it is an easy mistake to make inside the loop."""
        resource = _resource(db)
        chunks = []
        for file_index in range(3):
            file_row = _file(db, resource, f"doc-{file_index}.txt", provider="openai", model=MODEL)
            chunks += [
                _chunk(db, resource, file_row, f"motorcycle service {file_index}-{index}", file_index * 10 + index, _vector(0.8))
                for index in range(10)
            ]
        calls = _stub_embed(monkeypatch)

        _semantic_retrieval(db, OWNER, resource.id, "bike servicing", chunks)

        assert calls == [["bike servicing"]]

    def test_answering_a_question_embeds_it_once_however_many_gates_run(self, db, monkeypatch):
        """`corrective` mode runs the sufficiency gate twice and re-retrieves in between.
        The semantic side is computed once for the whole question and shared; recomputing
        it per gate would double the bill on exactly the mode that retries."""
        resource = _resource(db)
        file_row = _file(db, resource, "manual.txt", provider="openai", model=MODEL)
        for index in range(3):
            _chunk(db, resource, file_row, f"motorcycle valve clearance step {index}", index, _vector(0.9))
        calls = _stub_embed(monkeypatch)

        plan = rag_service._plan_answer(db, OWNER, resource.id, "valve clearance", "corrective", None, None)

        assert len(calls) == 1
        assert plan.answer


class TestTheSynonymBridge:
    """The case the feature exists for, end to end at the unit level."""

    QUERY = "how often should the bike be serviced"

    def test_a_bike_question_scores_zero_against_a_motorcycle_passage(self, db):
        # The starting point: lexically this passage does not exist. `_score_chunk`
        # returning 0.0 is indistinguishable from "the document does not cover this".
        resource = _resource(db)
        file_row = _file(db, resource, "manual.txt", provider="openai", model=MODEL)
        chunk = _chunk(
            db, resource, file_row,
            "Motorcycle drivetrain maintenance intervals are listed per model year",
            0, _vector(0.93),
        )

        assert _score_chunk(self.QUERY, Counter(_tokenize(self.QUERY)), chunk) == 0.0
        assert _contextual_hybrid(self.QUERY, [chunk]) == []

    def test_the_vector_surfaces_it_once_and_the_gate_lets_the_answer_through(self, db, monkeypatch):
        resource = _resource(db)
        file_row = _file(db, resource, "manual.txt", provider="openai", model=MODEL)
        chunk = _chunk(
            db, resource, file_row,
            "Motorcycle drivetrain maintenance intervals are listed per model year",
            0, _vector(0.93),
        )
        _stub_embed(monkeypatch)

        lexical = _contextual_hybrid(self.QUERY, [chunk])
        semantic = _semantic_retrieval(db, OWNER, resource.id, self.QUERY, [chunk])
        blended = _blend_semantic(lexical, semantic)

        # Retrieved exactly once, credited to the similarity, and allowed as evidence.
        assert [item.chunk.id for item in blended] == [chunk.id]
        assert "semantic" in blended[0].reason
        assert _has_sufficient_evidence(self.QUERY, blended, semantic) is True
        # ...and without the vector, the very same question is refused. This pair is the
        # feature: the difference between the two lines is the whole value of embedding.
        assert _has_sufficient_evidence(self.QUERY, blended) is False
