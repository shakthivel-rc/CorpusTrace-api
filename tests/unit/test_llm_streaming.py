"""Streaming parsers for every wire protocol.

All nine providers can stream, but in three different framings: OpenAI-style SSE
(6 providers), Anthropic's named SSE events, Gemini's alt=sse, and Ollama's NDJSON.
Each parser has one job — yield text deltas — and these tests pin the framing details
that are easy to get subtly wrong: keep-alive comment lines, the [DONE] sentinel,
non-text delta types, and mid-stream error events.
"""
import json

import pytest

import services.llm_provider as llm
from models.llm import LlmProviderCredential
from services.llm_provider import (
    GROUNDED_SYSTEM_PROMPT,
    LlmProviderError,
    encrypt_secret,
    stream_anthropic_messages,
    stream_gemini_generate,
    stream_grounded_answer,
    stream_ollama_generate,
    stream_openai_chat,
)

pytestmark = pytest.mark.unit


class _FakeStream:
    """Stands in for the urlopen response: iterates lines and closes like a context manager."""

    def __init__(self, lines: list[str]):
        self._lines = [line.encode("utf-8") for line in lines]
        self.closed = False

    def __iter__(self):
        return iter(self._lines)

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.closed = True
        return False


def _capture_stream(monkeypatch, lines: list[str]) -> dict:
    """Replace the transport with a canned stream and record what was requested."""
    captured: dict = {}
    stream = _FakeStream(lines)

    def fake_open(method, url, headers, payload):
        captured.update(method=method, url=url, headers=headers, payload=payload, stream=stream)
        return stream

    monkeypatch.setattr(llm, "_open_with_retry", fake_open)
    return captured


def _sse(chunk: dict) -> str:
    return f"data: {json.dumps(chunk)}\n"


def _openai_delta(text: str) -> str:
    return _sse({"choices": [{"index": 0, "delta": {"content": text}}]})


class TestOpenAiStream:
    def test_deltas_are_yielded_in_order(self, monkeypatch):
        _capture_stream(monkeypatch, [_openai_delta("Hello"), _openai_delta(" world"), "data: [DONE]\n"])
        assert list(stream_openai_chat("k", "https://x/v1", "m", "p")) == ["Hello", " world"]

    def test_the_request_asks_for_streaming(self, monkeypatch):
        captured = _capture_stream(monkeypatch, ["data: [DONE]\n"])
        list(stream_openai_chat("k", "https://x/v1", "m", "p", {"temperature": 0.5}, "SYS"))
        assert captured["payload"]["stream"] is True
        assert captured["payload"]["temperature"] == 0.5
        assert captured["payload"]["messages"][0] == {"role": "system", "content": "SYS"}
        assert captured["url"] == "https://x/v1/chat/completions"

    def test_openrouter_keepalive_comments_are_skipped(self, monkeypatch):
        """OpenRouter injects ': OPENROUTER PROCESSING' to defeat proxy timeouts; feeding
        one to json.loads would raise and lose the answer."""
        _capture_stream(
            monkeypatch,
            [": OPENROUTER PROCESSING\n", "\n", _openai_delta("ok"), ": OPENROUTER PROCESSING\n", "data: [DONE]\n"],
        )
        assert list(stream_openai_chat("k", "https://openrouter.ai/api/v1", "m", "p")) == ["ok"]

    def test_the_done_sentinel_stops_the_stream(self, monkeypatch):
        _capture_stream(monkeypatch, [_openai_delta("a"), "data: [DONE]\n", _openai_delta("never")])
        assert list(stream_openai_chat("k", "https://x/v1", "m", "p")) == ["a"]

    def test_the_role_only_first_chunk_yields_nothing(self, monkeypatch):
        _capture_stream(
            monkeypatch,
            [_sse({"choices": [{"delta": {"role": "assistant", "content": ""}}]}), _openai_delta("hi"), "data: [DONE]\n"],
        )
        assert list(stream_openai_chat("k", "https://x/v1", "m", "p")) == ["hi"]

    def test_a_malformed_line_is_skipped_rather_than_losing_the_answer(self, monkeypatch):
        _capture_stream(monkeypatch, [_openai_delta("a"), "data: {not json\n", _openai_delta("b"), "data: [DONE]\n"])
        assert list(stream_openai_chat("k", "https://x/v1", "m", "p")) == ["a", "b"]

    def test_a_mid_stream_error_event_raises(self, monkeypatch):
        _capture_stream(monkeypatch, [_openai_delta("a"), _sse({"error": {"message": "rate limit exceeded"}})])
        with pytest.raises(LlmProviderError) as excinfo:
            list(stream_openai_chat("k", "https://x/v1", "m", "p"))
        assert "rate limit exceeded" in str(excinfo.value)

    def test_the_response_is_closed(self, monkeypatch):
        captured = _capture_stream(monkeypatch, [_openai_delta("a"), "data: [DONE]\n"])
        list(stream_openai_chat("k", "https://x/v1", "m", "p"))
        assert captured["stream"].closed


class TestAnthropicStream:
    def test_text_deltas_are_accumulated_and_message_stop_ends_it(self, monkeypatch):
        _capture_stream(
            monkeypatch,
            [
                _sse({"type": "message_start", "message": {"content": []}}),
                _sse({"type": "content_block_start", "index": 0}),
                _sse({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hel"}}),
                _sse({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "lo"}}),
                _sse({"type": "content_block_stop", "index": 0}),
                _sse({"type": "message_stop"}),
            ],
        )
        assert list(stream_anthropic_messages("k", "https://x/v1", "m", "p")) == ["Hel", "lo"]

    def test_ping_events_are_ignored(self, monkeypatch):
        _capture_stream(
            monkeypatch,
            [
                "event: ping\n",
                _sse({"type": "ping"}),
                _sse({"type": "content_block_delta", "delta": {"type": "text_delta", "text": "x"}}),
                _sse({"type": "message_stop"}),
            ],
        )
        assert list(stream_anthropic_messages("k", "https://x/v1", "m", "p")) == ["x"]

    def test_non_text_delta_types_are_ignored(self, monkeypatch):
        """thinking_delta and input_json_delta are not answer text."""
        _capture_stream(
            monkeypatch,
            [
                _sse({"type": "content_block_delta", "delta": {"type": "thinking_delta", "thinking": "hmm"}}),
                _sse({"type": "content_block_delta", "delta": {"type": "input_json_delta", "partial_json": "{}"}}),
                _sse({"type": "content_block_delta", "delta": {"type": "text_delta", "text": "real"}}),
                _sse({"type": "message_stop"}),
            ],
        )
        assert list(stream_anthropic_messages("k", "https://x/v1", "m", "p")) == ["real"]

    def test_a_mid_stream_error_event_raises(self, monkeypatch):
        _capture_stream(monkeypatch, [_sse({"type": "error", "error": {"message": "overloaded_error"}})])
        with pytest.raises(LlmProviderError) as excinfo:
            list(stream_anthropic_messages("k", "https://x/v1", "m", "p"))
        assert "overloaded" in str(excinfo.value)

    def test_the_request_carries_stream_and_the_system_prompt(self, monkeypatch):
        captured = _capture_stream(monkeypatch, [_sse({"type": "message_stop"})])
        list(stream_anthropic_messages("k", "https://x/v1", "m", "p", None, "SYS"))
        assert captured["payload"]["stream"] is True
        assert captured["payload"]["system"] == "SYS"
        assert captured["payload"]["max_tokens"], "max_tokens is required by the Messages API"
        assert captured["headers"]["anthropic-version"] == "2023-06-01"


class TestGeminiStream:
    def _chunk(self, text: str) -> str:
        return _sse({"candidates": [{"content": {"parts": [{"text": text}]}}]})

    def test_parts_are_yielded(self, monkeypatch):
        _capture_stream(monkeypatch, [self._chunk("Hello"), self._chunk(" there")])
        assert list(stream_gemini_generate("k", "https://x/v1beta", "gemini-2.5-flash", "p")) == ["Hello", " there"]

    def test_alt_sse_is_requested_on_the_streaming_verb(self, monkeypatch):
        """Without alt=sse the endpoint returns one big JSON array, which cannot be
        parsed incrementally."""
        captured = _capture_stream(monkeypatch, [self._chunk("x")])
        list(stream_gemini_generate("key123", "https://x/v1beta", "gemini-2.5-flash", "p"))
        assert ":streamGenerateContent" in captured["url"]
        assert "alt=sse" in captured["url"]
        assert "key=key123" in captured["url"]

    def test_a_final_chunk_without_parts_is_harmless(self, monkeypatch):
        _capture_stream(monkeypatch, [self._chunk("a"), _sse({"candidates": [{"finishReason": "STOP"}]})])
        assert list(stream_gemini_generate("k", "https://x/v1beta", "m", "p")) == ["a"]

    def test_the_stream_simply_ends_at_eof(self, monkeypatch):
        """Gemini has no [DONE] sentinel."""
        _capture_stream(monkeypatch, [self._chunk("only")])
        assert list(stream_gemini_generate("k", "https://x/v1beta", "m", "p")) == ["only"]

    def test_the_system_instruction_is_sent(self, monkeypatch):
        captured = _capture_stream(monkeypatch, [self._chunk("x")])
        list(stream_gemini_generate("k", "https://x/v1beta", "m", "p", None, "SYS"))
        assert captured["payload"]["systemInstruction"] == {"parts": [{"text": "SYS"}]}


class TestOllamaStream:
    def test_ndjson_lines_are_accumulated_until_done(self, monkeypatch):
        _capture_stream(
            monkeypatch,
            [
                json.dumps({"response": "Hel", "done": False}) + "\n",
                json.dumps({"response": "lo", "done": False}) + "\n",
                json.dumps({"response": "", "done": True, "eval_count": 12}) + "\n",
            ],
        )
        assert list(stream_ollama_generate("http://localhost:11434", "llama3.3", "p")) == ["Hel", "lo"]

    def test_done_stops_the_stream(self, monkeypatch):
        _capture_stream(
            monkeypatch,
            [
                json.dumps({"response": "a", "done": True}) + "\n",
                json.dumps({"response": "never", "done": False}) + "\n",
            ],
        )
        assert list(stream_ollama_generate("http://x", "m", "p")) == ["a"]

    def test_streaming_is_requested_explicitly(self, monkeypatch):
        """Ollama streams by default, so the flag is sent on both paths for clarity."""
        captured = _capture_stream(monkeypatch, [json.dumps({"response": "", "done": True}) + "\n"])
        list(stream_ollama_generate("http://x", "m", "p", None, "SYS"))
        assert captured["payload"]["stream"] is True
        assert captured["payload"]["system"] == "SYS"
        assert captured["url"] == "http://x/api/generate"


class TestStreamDispatch:
    """`_stream` must pick the parser from api_style, exactly as `_complete` does."""

    def _credential(self, db, user_id: str, provider: str, base_url: str):
        db.add(
            LlmProviderCredential(
                user_id=user_id, provider=provider, encrypted_api_key=encrypt_secret("sk-test"), base_url=base_url
            )
        )
        db.commit()

    def test_openai_style_provider_uses_the_openai_parser(self, db, monkeypatch):
        self._credential(db, "stream-1", "groq", "https://api.groq.com/openai/v1")
        _capture_stream(monkeypatch, [_openai_delta("hi"), "data: [DONE]\n"])
        chunks = llm._stream(db, "stream-1", "groq", "llama-3.3-70b-versatile", "prompt", GROUNDED_SYSTEM_PROMPT)
        assert list(chunks) == ["hi"]

    def test_gemini_provider_uses_the_gemini_parser(self, db, monkeypatch):
        self._credential(db, "stream-2", "gemini", "https://generativelanguage.googleapis.com/v1beta")
        captured = _capture_stream(monkeypatch, [_sse({"candidates": [{"content": {"parts": [{"text": "g"}]}}]})])
        assert list(llm._stream(db, "stream-2", "gemini", "gemini-2.5-flash", "p", GROUNDED_SYSTEM_PROMPT)) == ["g"]
        assert "streamGenerateContent" in captured["url"]

    def test_ollama_streams_without_a_credential(self, db, monkeypatch):
        _capture_stream(monkeypatch, [json.dumps({"response": "local", "done": True}) + "\n"])
        assert list(llm._stream(db, "stream-3", "ollama", "llama3.3", "p", GROUNDED_SYSTEM_PROMPT)) == ["local"]

    def test_an_unconfigured_provider_fails_before_iteration_begins(self, db):
        """The caller must learn this while it can still choose a non-streaming fallback,
        not on its first next() after the response body is already committed."""
        with pytest.raises(LlmProviderError) as excinfo:
            llm._stream(db, "stream-4", "groq", "m", "p", GROUNDED_SYSTEM_PROMPT)
        assert excinfo.value.status_code == 400

    def test_a_placeholder_base_url_fails_before_iteration_begins(self, db):
        self._credential(db, "stream-5", "cloudflare", llm.SUPPORTED_PROVIDERS["cloudflare"]["default_base_url"])
        with pytest.raises(LlmProviderError) as excinfo:
            llm._stream(db, "stream-5", "cloudflare", "@cf/openai/gpt-oss-20b", "p", GROUNDED_SYSTEM_PROMPT)
        assert excinfo.value.status_code == 400


class TestStreamGroundedAnswer:
    def test_it_streams_under_the_grounded_system_prompt(self, db, monkeypatch):
        db.add(
            LlmProviderCredential(
                user_id="grounded-1",
                provider="groq",
                encrypted_api_key=encrypt_secret("sk-test"),
                base_url="https://api.groq.com/openai/v1",
            )
        )
        db.commit()
        captured = _capture_stream(monkeypatch, [_openai_delta("Twenty days."), "data: [DONE]\n"])
        chunks = stream_grounded_answer(
            db,
            "grounded-1",
            "groq",
            "llama-3.3-70b-versatile",
            "how many vacation days",
            [{"source_name": "hr.pdf", "chunk_index": 1, "modality": "text", "content": "Twenty days."}],
            "contextual_hybrid",
            "Handbook",
        )
        assert list(chunks) == ["Twenty days."]
        assert captured["payload"]["messages"][0]["content"] == GROUNDED_SYSTEM_PROMPT
        assert "hr.pdf" in captured["payload"]["messages"][1]["content"], "evidence must reach the prompt"
