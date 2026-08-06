"""stream_answer_question — the streaming twin of answer_question.

Retrieval is local and fast; what streams is the LLM's synthesis. The properties that
matter are that a user never loses an answer retrieval already produced (a provider
failure degrades to the extractive text), that a failure *after* text is on the wire is
explained rather than swallowed, and that both entry points end up saying the same thing.
"""
import json

import pytest

import rag.service as rag_service
from models.rag import DocumentChunk
from models.resource import Resource
from services.llm_provider import LlmProviderError

pytestmark = pytest.mark.unit


def _seed_resource_with_chunk(db, user_id: str) -> Resource:
    resource = Resource(resource_name="Handbook", user_id=user_id, upload_status=True)
    db.add(resource)
    db.flush()
    db.add(
        DocumentChunk(
            resource_id=resource.id,
            chunk_index=0,
            source_name="handbook.txt",
            modality="text",
            title="Vacation policy",
            content="Employees receive twenty vacation days per year.",
            contextual_content="Vacation policy. Employees receive twenty vacation days per year.",
            terms_json=json.dumps({"vacation": 3, "days": 2, "employees": 1}),
        )
    )
    db.commit()
    return resource


def _collect(stream) -> str:
    return "".join(stream.chunks)


class TestWithoutAnLlm:
    def test_the_extractive_answer_arrives_as_one_chunk(self, db):
        resource = _seed_resource_with_chunk(db, "s-1")
        stream = rag_service.stream_answer_question(db, "s-1", resource.id, "vacation days", None)
        chunks = list(stream.chunks)
        assert len(chunks) == 1, "there is nothing to stream — the answer is already complete"
        assert "vacation" in chunks[0].lower()

    def test_mode_and_citations_are_known_before_the_first_chunk(self, db):
        resource = _seed_resource_with_chunk(db, "s-2")
        stream = rag_service.stream_answer_question(db, "s-2", resource.id, "vacation days", None)
        assert stream.mode == "contextual_hybrid"
        assert stream.citations, "retrieval already ran, so citations are available up front"

    def test_small_talk_never_reaches_retrieval(self, db):
        resource = _seed_resource_with_chunk(db, "s-3")
        stream = rag_service.stream_answer_question(db, "s-3", resource.id, "hi", None)
        assert "Hello!" in _collect(stream)
        assert stream.citations == []

    def test_an_unknown_resource_raises_before_any_chunk(self, db):
        with pytest.raises(ValueError):
            rag_service.stream_answer_question(db, "s-4", "missing-id", "anything", None)

    def test_provider_without_model_raises_before_any_chunk(self, db):
        resource = _seed_resource_with_chunk(db, "s-5")
        with pytest.raises(LlmProviderError):
            rag_service.stream_answer_question(db, "s-5", resource.id, "vacation days", None, "groq", None)


class TestWithAnLlm:
    def test_deltas_stream_through_and_the_body_is_verbatim_answer_text(self, db, monkeypatch):
        """Invariant 24: citations ride the header, the body is the model's own words.

        The provenance footer this used to assert on is gone — provider, model, source
        name, chunk index, modality and score are all structured fields on `citations`,
        and the client renders them as a panel.
        """
        resource = _seed_resource_with_chunk(db, "s-6")
        monkeypatch.setattr(rag_service, "stream_grounded_answer", lambda *a, **k: iter(["Twenty ", "days. [1]"]))
        stream = rag_service.stream_answer_question(
            db, "s-6", resource.id, "vacation days", None, "groq", "llama-3.3-70b-versatile"
        )
        citations = stream.citations
        chunks = list(stream.chunks)
        assembled = "".join(chunks)
        assert len(chunks) > 1, "deltas must arrive separately, not as one buffered answer"
        assert assembled == "Twenty days. [1]"
        for leaked in ("LLM: groq/llama-3.3-70b-versatile", "Sources:", "modality=", "score="):
            assert leaked not in assembled
        # The same information, structurally — nothing was lost by dropping the block.
        assert citations and citations[0]["source_name"] == "handbook.txt"
        assert "chunk_index" in citations[0] and "modality" in citations[0] and "score" in citations[0]

    def test_leading_and_trailing_whitespace_is_stripped_like_the_blocking_path(self, db, monkeypatch):
        """Models often open with a newline and close with one. The blocking path emits
        llm_answer.strip(), so the streamed text has to match exactly.

        Nothing follows the last delta now, so the deferred trailing whitespace must be
        dropped rather than flushed — an equality assertion is the only thing that pins it.
        """
        resource = _seed_resource_with_chunk(db, "s-6b")
        monkeypatch.setattr(rag_service, "stream_grounded_answer", lambda *a, **k: iter(["\n\n Twenty ", "days. \n\n"]))
        stream = rag_service.stream_answer_question(
            db, "s-6b", resource.id, "vacation days", None, "groq", "llama-3.3-70b-versatile"
        )
        answer = _collect(stream)
        assert answer == "Twenty days.", "leading and trailing whitespace must both be gone"

    def test_a_failure_before_the_first_token_degrades_to_the_extractive_answer(self, db, monkeypatch):
        resource = _seed_resource_with_chunk(db, "s-7")

        def boom(*args, **kwargs):
            raise LlmProviderError("Provider API returned HTTP 429: rate limited", 502)

        monkeypatch.setattr(rag_service, "stream_grounded_answer", boom)
        stream = rag_service.stream_answer_question(
            db, "s-7", resource.id, "vacation days", None, "groq", "llama-3.3-70b-versatile"
        )
        answer = _collect(stream)
        assert "vacation" in answer.lower(), "the extractive answer must survive"
        assert "Could not write an answer with groq/llama-3.3-70b-versatile" in answer
        assert stream.citations, "citations from the extractive pass must survive too"

    def test_a_failure_raised_lazily_on_the_first_delta_still_degrades_cleanly(self, db, monkeypatch):
        """Generators do not run until first next(); priming is what turns this into a
        clean fallback instead of a half-written response."""
        resource = _seed_resource_with_chunk(db, "s-8")

        def lazy_boom(*args, **kwargs):
            def gen():
                raise LlmProviderError("Provider API returned HTTP 401: bad key", 502)
                yield  # pragma: no cover - unreachable, marks this a generator

            return gen()

        monkeypatch.setattr(rag_service, "stream_grounded_answer", lazy_boom)
        stream = rag_service.stream_answer_question(
            db, "s-8", resource.id, "vacation days", None, "groq", "llama-3.3-70b-versatile"
        )
        answer = _collect(stream)
        assert "vacation" in answer.lower()
        assert "Could not write an answer with" in answer
        assert "401" in answer

    def test_an_empty_stream_degrades_to_the_extractive_answer(self, db, monkeypatch):
        resource = _seed_resource_with_chunk(db, "s-9")
        monkeypatch.setattr(rag_service, "stream_grounded_answer", lambda *a, **k: iter([]))
        stream = rag_service.stream_answer_question(
            db, "s-9", resource.id, "vacation days", None, "groq", "llama-3.3-70b-versatile"
        )
        answer = _collect(stream)
        assert "vacation" in answer.lower()
        assert "empty response" in answer

    def test_a_mid_stream_failure_keeps_what_was_streamed_and_explains_the_cut(self, db, monkeypatch):
        resource = _seed_resource_with_chunk(db, "s-10")

        def half_broken(*args, **kwargs):
            def gen():
                yield "Twenty days are granted"
                raise LlmProviderError("connection reset", 502)

            return gen()

        monkeypatch.setattr(rag_service, "stream_grounded_answer", half_broken)
        stream = rag_service.stream_answer_question(
            db, "s-10", resource.id, "vacation days", None, "groq", "llama-3.3-70b-versatile"
        )
        answer = _collect(stream)
        assert answer.startswith("Twenty days are granted"), "already-streamed text must be kept"
        assert "cut short" in answer
        assert "Sources:" not in answer, "an interrupted answer must not claim complete provenance"

    def test_no_evidence_means_nothing_is_streamed_from_the_model(self, db, monkeypatch):
        resource = _seed_resource_with_chunk(db, "s-11")

        def must_not_be_called(*args, **kwargs):
            raise AssertionError("the evidence gate must run before any synthesis call")

        monkeypatch.setattr(rag_service, "stream_grounded_answer", must_not_be_called)
        monkeypatch.setattr(rag_service, "generate_conversational_reply", lambda *a, **k: "")
        stream = rag_service.stream_answer_question(
            db, "s-11", resource.id, "who is the current president of France", None, "groq", "m"
        )
        assert stream.citations == []
        assert _collect(stream)


class TestStreamingMatchesBlocking:
    """The two entry points must not drift: same retrieval, same gates, same wording."""

    def test_the_assembled_stream_equals_the_blocking_answer(self, db, monkeypatch):
        resource = _seed_resource_with_chunk(db, "s-12")
        monkeypatch.setattr(rag_service, "generate_grounded_answer", lambda *a, **k: "Twenty days. [1]")
        monkeypatch.setattr(rag_service, "stream_grounded_answer", lambda *a, **k: iter(["Twenty ", "days. [1]"]))

        blocking = rag_service.answer_question(db, "s-12", resource.id, "vacation days", None, "groq", "m")
        streamed = rag_service.stream_answer_question(db, "s-12", resource.id, "vacation days", None, "groq", "m")

        assert _collect(streamed) == blocking.answer
        assert streamed.mode == blocking.mode
        assert streamed.citations == blocking.citations

    def test_both_agree_without_an_llm(self, db):
        resource = _seed_resource_with_chunk(db, "s-13")
        blocking = rag_service.answer_question(db, "s-13", resource.id, "vacation days", None)
        streamed = rag_service.stream_answer_question(db, "s-13", resource.id, "vacation days", None)
        assert _collect(streamed) == blocking.answer

    def test_both_agree_on_a_provider_failure(self, db, monkeypatch):
        resource = _seed_resource_with_chunk(db, "s-14")
        failure = LlmProviderError("Provider API returned HTTP 429: rate limited", 502)

        def boom(*args, **kwargs):
            raise failure

        monkeypatch.setattr(rag_service, "generate_grounded_answer", boom)
        monkeypatch.setattr(rag_service, "stream_grounded_answer", boom)

        blocking = rag_service.answer_question(db, "s-14", resource.id, "vacation days", None, "groq", "m")
        streamed = rag_service.stream_answer_question(db, "s-14", resource.id, "vacation days", None, "groq", "m")
        assert _collect(streamed) == blocking.answer
