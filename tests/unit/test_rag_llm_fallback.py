"""answer_question must degrade to the extractive answer when the selected LLM
provider fails (bad key, rate limit, outage) — the pre-computed extractive answer
was previously discarded and the whole ask returned an error."""
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


class TestLlmFailureFallback:
    def test_provider_failure_returns_the_extractive_answer_with_a_note(self, db, monkeypatch):
        resource = _seed_resource_with_chunk(db, "user-a")

        def boom(*args, **kwargs):
            raise LlmProviderError("Provider API returned HTTP 429: rate limited", 502)

        monkeypatch.setattr(rag_service, "generate_grounded_answer", boom)
        result = rag_service.answer_question(db, "user-a", resource.id, "vacation days", None, "groq", "llama-3.3-70b-versatile")

        assert "vacation" in result.answer.lower()
        assert "Could not write an answer with groq/llama-3.3-70b-versatile" in result.answer
        assert result.citations, "citations from the extractive pass must survive the fallback"

    def test_successful_llm_answer_is_used_verbatim(self, db, monkeypatch):
        resource = _seed_resource_with_chunk(db, "user-b")
        monkeypatch.setattr(rag_service, "generate_grounded_answer", lambda *a, **k: "Twenty days. [1]")
        result = rag_service.answer_question(db, "user-b", resource.id, "vacation days", None, "groq", "llama-3.3-70b-versatile")
        assert result.answer.startswith("Twenty days. [1]")
        assert "failed" not in result.answer

    def test_provider_and_model_must_be_supplied_together(self, db):
        resource = _seed_resource_with_chunk(db, "user-c")
        with pytest.raises(LlmProviderError):
            rag_service.answer_question(db, "user-c", resource.id, "vacation days", None, "groq", None)


class TestNoRawProvenanceFooter:
    """Invariant 24, stated at full strength: the answer body is verbatim answer text.

    Provider, model, source name, chunk index, modality and score all reach the client as
    structured fields on `citations` (the X-Nexarag-Citations header and citations_json),
    where the UI renders them as a provenance panel. None of them may be printed into the
    prose, in either the synthesized or the extractive path.
    """

    FOOTER_MARKERS = ("Sources:", "LLM: ", "modality=", "score=", ", chunk ")

    def test_a_synthesized_answer_is_exactly_what_the_model_wrote(self, db, monkeypatch):
        resource = _seed_resource_with_chunk(db, "user-d")
        monkeypatch.setattr(rag_service, "generate_grounded_answer", lambda *a, **k: "  Twenty days. [1]\n\n")
        result = rag_service.answer_question(
            db, "user-d", resource.id, "vacation days", None, "groq", "llama-3.3-70b-versatile"
        )
        assert result.answer == "Twenty days. [1]"
        for marker in self.FOOTER_MARKERS:
            assert marker not in result.answer

    def test_the_extractive_answer_keeps_its_bullets_but_carries_no_footer(self, db):
        """The no-provider path is the one a user without an API key sees every time."""
        resource = _seed_resource_with_chunk(db, "user-e")
        result = rag_service.answer_question(db, "user-e", resource.id, "vacation days", None)

        assert "- " in result.answer, "the extractive bullets ARE the answer and must stay"
        assert "[1]" in result.answer, "the [N] markers still resolve against the panel"
        for marker in self.FOOTER_MARKERS:
            assert marker not in result.answer

    def test_the_dropped_fields_are_all_present_on_the_citations(self, db, monkeypatch):
        resource = _seed_resource_with_chunk(db, "user-f")
        monkeypatch.setattr(rag_service, "generate_grounded_answer", lambda *a, **k: "Twenty days. [1]")
        result = rag_service.answer_question(
            db, "user-f", resource.id, "vacation days", None, "groq", "llama-3.3-70b-versatile"
        )
        citation = result.citations[0]
        assert citation["index"] == 1
        assert citation["source_name"] == "handbook.txt"
        assert citation["chunk_index"] == 0
        assert citation["modality"] == "text"
        assert isinstance(citation["score"], float)

    def test_the_synthesis_failure_note_carries_no_footer_either(self, db, monkeypatch):
        resource = _seed_resource_with_chunk(db, "user-g")

        def boom(*args, **kwargs):
            raise LlmProviderError("Provider API returned HTTP 429: rate limited", 502)

        monkeypatch.setattr(rag_service, "generate_grounded_answer", boom)
        result = rag_service.answer_question(
            db, "user-g", resource.id, "vacation days", None, "groq", "llama-3.3-70b-versatile"
        )
        for marker in self.FOOTER_MARKERS:
            assert marker not in result.answer


class TestReadableProviderError:
    """The note is user-facing prose, so the provider's JSON envelope must not reach it.

    Regression: a real OpenRouter 402 rendered as a wall of raw JSON in the chat answer,
    and because `_error_detail` bounds the body, its trailing URL arrived cut in half —
    GFM then autolinked the fragment into a dead `https://o` link.
    """

    # Exactly what the provider returned, truncated by _error_detail's 300-char bound.
    TRUNCATED_402 = (
        'Provider API returned HTTP 402: {"error":{"message":"Insufficient credits. This '
        "account never purchased credits. Make sure your key is on the correct account or "
        'org, and if so, purchase more at https://openrouter.ai/settings/credits","code":402,'
        '"metadata":{"limit_source":"openrouter_credits","remedy_hint":"Add credits at https://o'
    )

    def test_keeps_the_human_message_and_drops_the_envelope(self):
        out = rag_service._readable_provider_error(LlmProviderError(self.TRUNCATED_402, 502))
        assert "Insufficient credits." in out
        assert "Provider API returned HTTP 402" in out
        # The machine envelope is gone.
        assert '"metadata"' not in out
        assert '"code":402' not in out
        assert "limit_source" not in out

    def test_never_emits_a_url_cut_mid_token(self):
        out = rag_service._readable_provider_error(LlmProviderError(self.TRUNCATED_402, 502))
        assert "https://o " not in out + " "
        assert not out.rstrip("…").endswith("https://o")

    def test_parses_a_complete_envelope_too(self):
        body = 'Provider API returned HTTP 401: {"error": {"message": "Incorrect API key provided."}}'
        out = rag_service._readable_provider_error(LlmProviderError(body, 502))
        assert out == "Provider API returned HTTP 401: Incorrect API key provided."

    def test_passes_through_a_plain_message(self):
        out = rag_service._readable_provider_error(LlmProviderError("Provider API request failed: timed out", 502))
        assert out == "Provider API request failed: timed out"

    def test_bounds_the_length(self):
        out = rag_service._readable_provider_error(LlmProviderError("word " * 400, 502))
        assert len(out) <= rag_service.MAX_PROVIDER_ERROR_CHARS + 1

    def test_note_reads_as_prose_not_json(self):
        answer = rag_service._synthesis_failed_answer(
            "Twenty days.", "openrouter", "anthropic/claude-x", LlmProviderError(self.TRUNCATED_402, 502)
        )
        assert answer.startswith("Twenty days.")
        assert "Insufficient credits." in answer
        assert '"metadata"' not in answer
        # The user keeps the retrieved answer regardless.
        assert "keyword match" in answer
