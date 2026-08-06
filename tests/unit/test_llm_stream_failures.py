"""Streaming failures must arrive as LlmProviderError, never as a raw exception.

`_open_with_retry` can only classify failures up to the moment response headers arrive.
Everything after that is a body read, and a stall, a reset or a structurally odd chunk
used to escape every caller — all of which catch `LlmProviderError` — and, with no global
exception handler, became a plain-text HTTP 500 that also discarded the extractive answer
retrieval had already produced. These tests pin the conversion.
"""
import json

import pytest

import services.llm_provider as llm
from services.llm_provider import (
    LlmProviderError,
    stream_anthropic_messages,
    stream_gemini_generate,
    stream_ollama_generate,
    stream_openai_chat,
)

pytestmark = pytest.mark.unit


class _ExplodingStream:
    """Yields some lines, then fails the way a dropped socket really does."""

    def __init__(self, lines: list[str], error: BaseException | None = None):
        self._lines = lines
        self._error = error
        self.closed = False

    def __iter__(self):
        for line in self._lines:
            yield line.encode("utf-8")
        if self._error is not None:
            raise self._error

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.closed = True
        return False


def _install(monkeypatch, stream):
    monkeypatch.setattr(llm, "_open_with_retry", lambda method, url, headers, payload: stream)
    return stream


def _openai_delta(text: str) -> str:
    return "data: " + json.dumps({"choices": [{"delta": {"content": text}}]}) + "\n"


class TestTransportFailuresDuringTheBodyRead:
    @pytest.mark.parametrize(
        "error",
        [
            TimeoutError("timed out"),
            ConnectionResetError(104, "Connection reset by peer"),
            OSError("socket closed"),
        ],
    )
    def test_a_socket_failure_before_the_first_token_becomes_a_provider_error(self, monkeypatch, error):
        _install(monkeypatch, _ExplodingStream([], error))
        with pytest.raises(LlmProviderError) as excinfo:
            list(stream_openai_chat("k", "https://x/v1", "m", "p"))
        assert excinfo.value.status_code == 502

    def test_a_socket_failure_after_some_tokens_becomes_a_provider_error(self, monkeypatch):
        _install(monkeypatch, _ExplodingStream([_openai_delta("Twenty ")], TimeoutError("timed out")))
        chunks = []
        with pytest.raises(LlmProviderError):
            for chunk in stream_openai_chat("k", "https://x/v1", "m", "p"):
                chunks.append(chunk)
        assert chunks == ["Twenty "], "text received before the failure is still delivered"

    @pytest.mark.parametrize(
        "parser,args",
        [
            (stream_anthropic_messages, ("k", "https://x/v1", "m", "p")),
            (stream_gemini_generate, ("k", "https://x/v1beta", "m", "p")),
            (stream_ollama_generate, ("http://x", "m", "p")),
        ],
    )
    def test_every_parser_converts_a_stalled_read(self, monkeypatch, parser, args):
        _install(monkeypatch, _ExplodingStream([], TimeoutError("timed out")))
        with pytest.raises(LlmProviderError):
            list(parser(*args))


class TestStructurallyOddChunks:
    """Valid JSON in an unexpected shape must not raise AttributeError/KeyError.

    The blocking calls all wrap the same nested access in `except (KeyError, IndexError,
    TypeError)`; the streaming path has to match that contract.
    """

    @pytest.mark.parametrize(
        "payload",
        [
            {"choices": ["oops"]},
            {"choices": {"0": {"delta": {"content": "x"}}}},
            {"choices": [{"delta": "oops"}]},
        ],
    )
    def test_openai_shape_errors_become_provider_errors(self, monkeypatch, payload):
        _install(monkeypatch, _ExplodingStream(["data: " + json.dumps(payload) + "\n"]))
        with pytest.raises(LlmProviderError) as excinfo:
            list(stream_openai_chat("k", "https://x/v1", "m", "p"))
        assert excinfo.value.status_code == 502

    def test_gemini_shape_errors_become_provider_errors(self, monkeypatch):
        _install(
            monkeypatch,
            _ExplodingStream(['data: {"candidates": [{"content": "oops"}]}\n'], StopIteration()),
        )
        with pytest.raises(LlmProviderError):
            list(stream_gemini_generate("k", "https://x/v1beta", "m", "p"))

    def test_anthropic_shape_errors_become_provider_errors(self, monkeypatch):
        _install(
            monkeypatch,
            _ExplodingStream(['data: {"type": "content_block_delta", "delta": "oops"}\n'], StopIteration()),
        )
        with pytest.raises(LlmProviderError):
            list(stream_anthropic_messages("k", "https://x/v1", "m", "p"))


class TestDeltaContentParts:
    """delta.content is typed string | ContentPart[] | null. A list yielded into the text
    pipeline only fails much later, inside Starlette's encode()."""

    def test_a_content_parts_array_is_flattened_to_text(self, monkeypatch):
        payload = {"choices": [{"delta": {"content": [{"type": "text", "text": "hi"}]}}]}
        _install(monkeypatch, _ExplodingStream(["data: " + json.dumps(payload) + "\n", "data: [DONE]\n"]))
        assert list(stream_openai_chat("k", "https://x/v1", "m", "p")) == ["hi"]

    def test_a_non_string_delta_is_never_yielded(self, monkeypatch):
        payload = {"choices": [{"delta": {"content": 42}}]}
        _install(monkeypatch, _ExplodingStream(["data: " + json.dumps(payload) + "\n", "data: [DONE]\n"]))
        assert list(stream_openai_chat("k", "https://x/v1", "m", "p")) == []


class TestMultiBlockJoining:
    def test_anthropic_separates_text_blocks_the_way_the_blocking_path_does(self, monkeypatch):
        """call_anthropic_messages joins content blocks with "\\n"; streaming must match."""
        lines = [
            'data: {"type": "content_block_start", "index": 0}\n',
            'data: {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "first"}}\n',
            'data: {"type": "content_block_stop", "index": 0}\n',
            'data: {"type": "content_block_start", "index": 1}\n',
            'data: {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "second"}}\n',
            'data: {"type": "message_stop"}\n',
        ]
        _install(monkeypatch, _ExplodingStream(lines))
        assert "".join(stream_anthropic_messages("k", "https://x/v1", "m", "p")) == "first\nsecond"

    def test_gemini_joins_the_parts_of_one_chunk_with_a_newline(self, monkeypatch):
        payload = {"candidates": [{"content": {"parts": [{"text": "a"}, {"text": "b"}]}}]}
        _install(monkeypatch, _ExplodingStream(["data: " + json.dumps(payload) + "\n"]))
        assert list(stream_gemini_generate("k", "https://x/v1beta", "m", "p")) == ["a\nb"]


class TestRetryDelayHardening:
    """Retry-After is provider-controlled input reachable from an untrusted network path."""

    def test_a_nan_retry_after_does_not_reach_time_sleep(self):
        # time.sleep(NaN) raises ValueError from outside the retry loop's guards.
        assert llm._retry_delay(0, "nan") == llm._retry_delay(0, None)

    def test_an_infinite_retry_after_is_capped(self):
        assert llm._retry_delay(0, "inf") == llm.MAX_RETRY_SLEEP_SECONDS

    def test_a_huge_attempt_number_does_not_overflow(self):
        assert llm._retry_delay(5000, None) == llm.MAX_RETRY_SLEEP_SECONDS


class TestErrorDetailReading:
    def test_a_failing_error_body_read_does_not_escape(self):
        """An exception raised inside an `except` block is not offered to the sibling
        `except` clauses of the same `try`, so this would bypass retry entirely."""

        class Unreadable:
            def read(self, *args):
                raise ConnectionResetError("gone")

        assert llm._error_detail(Unreadable()) == "<no response body>"

    def test_the_error_body_read_is_bounded(self):
        captured = {}

        class Huge:
            def read(self, limit=None):
                captured["limit"] = limit
                return b"x" * 10_000

        detail = llm._error_detail(Huge())
        assert captured["limit"] == llm.MAX_ERROR_DETAIL_BYTES
        assert len(detail) <= 300
