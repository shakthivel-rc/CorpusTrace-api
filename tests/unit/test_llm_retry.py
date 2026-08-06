"""Retry and backoff for provider HTTP calls.

Every free tier throttles with HTTP 429, so a single retry is often the difference
between a working answer and a visible failure. The rules being pinned here are which
statuses are worth retrying (transient only), that a permanent verdict like 401 is
surfaced immediately rather than after N pointless round trips, and that the provider's
own Retry-After hint wins over the computed backoff.
"""
import io
from urllib import error

import pytest

import services.llm_provider as llm
from core.config import get_settings
from services.llm_provider import LlmProviderError, _open_with_retry, _redact_url, _retry_delay, http_json

pytestmark = pytest.mark.unit


class _FakeResponse(io.BytesIO):
    """Minimal stand-in for the urlopen response: readable and a context manager."""

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()
        return False


def _http_error(code: int, body: bytes = b"nope", headers: dict | None = None) -> error.HTTPError:
    return error.HTTPError("https://api.groq.com/openai/v1/models", code, "err", headers or {}, io.BytesIO(body))


@pytest.fixture(autouse=True)
def _no_real_sleeping(monkeypatch):
    """Backoff is real seconds — replace sleep so the tests stay fast, and record it."""
    slept: list[float] = []
    monkeypatch.setattr(llm.time, "sleep", slept.append)
    return slept


class TestRetryableStatuses:
    def test_a_429_is_retried_and_then_succeeds(self, monkeypatch, _no_real_sleeping):
        attempts = []

        def flaky(req, timeout=None):
            attempts.append(req.full_url)
            if len(attempts) == 1:
                raise _http_error(429)
            return _FakeResponse(b'{"data": []}')

        monkeypatch.setattr(llm.request, "urlopen", flaky)
        assert http_json("GET", "https://x/v1/models", {}) == {"data": []}
        assert len(attempts) == 2
        assert _no_real_sleeping, "a retry must back off before trying again"

    @pytest.mark.parametrize("status", [408, 425, 429, 500, 502, 503, 504])
    def test_every_transient_status_is_retried(self, monkeypatch, status):
        attempts = []

        def flaky(req, timeout=None):
            attempts.append(status)
            if len(attempts) == 1:
                raise _http_error(status)
            return _FakeResponse(b"{}")

        monkeypatch.setattr(llm.request, "urlopen", flaky)
        http_json("GET", "https://x/v1/models", {})
        assert len(attempts) == 2

    @pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
    def test_permanent_statuses_are_not_retried(self, monkeypatch, status):
        attempts = []

        def always_fails(req, timeout=None):
            attempts.append(status)
            raise _http_error(status, b"bad key")

        monkeypatch.setattr(llm.request, "urlopen", always_fails)
        with pytest.raises(LlmProviderError) as excinfo:
            http_json("GET", "https://x/v1/models", {})
        assert len(attempts) == 1, "retrying a permanent failure only delays the real reason"
        assert str(status) in str(excinfo.value)
        assert "bad key" in str(excinfo.value), "the provider's own explanation must survive"

    def test_transport_errors_are_retried(self, monkeypatch):
        attempts = []

        def flaky(req, timeout=None):
            attempts.append(1)
            if len(attempts) < 3:
                raise error.URLError("connection refused")
            return _FakeResponse(b"{}")

        monkeypatch.setattr(llm.request, "urlopen", flaky)
        http_json("POST", "https://x/v1/chat/completions", {}, {"model": "m"})
        assert len(attempts) == 3

    def test_a_timeout_is_retried(self, monkeypatch):
        """A stall raises a bare TimeoutError (OSError), not URLError."""
        attempts = []

        def flaky(req, timeout=None):
            attempts.append(1)
            if len(attempts) == 1:
                raise TimeoutError("timed out")
            return _FakeResponse(b"{}")

        monkeypatch.setattr(llm.request, "urlopen", flaky)
        http_json("GET", "https://x/v1/models", {})
        assert len(attempts) == 2

    def test_retries_are_bounded_by_the_configured_maximum(self, monkeypatch):
        attempts = []

        def always_fails(req, timeout=None):
            attempts.append(1)
            raise _http_error(503)

        monkeypatch.setattr(llm.request, "urlopen", always_fails)
        with pytest.raises(LlmProviderError):
            http_json("GET", "https://x/v1/models", {})
        assert len(attempts) == get_settings().llm_max_retries + 1

    def test_the_last_failure_is_what_the_caller_sees(self, monkeypatch):
        def always_fails(req, timeout=None):
            raise _http_error(503, b"upstream is down")

        monkeypatch.setattr(llm.request, "urlopen", always_fails)
        with pytest.raises(LlmProviderError) as excinfo:
            http_json("GET", "https://x/v1/models", {})
        assert "503" in str(excinfo.value)
        assert excinfo.value.status_code == 502


class TestRetryDelay:
    def test_backoff_grows_exponentially(self):
        assert _retry_delay(1, None) > _retry_delay(0, None)
        assert _retry_delay(2, None) > _retry_delay(1, None)

    def test_backoff_is_capped(self):
        assert _retry_delay(50, None) == llm.MAX_RETRY_SLEEP_SECONDS

    def test_retry_after_seconds_wins_over_the_computed_backoff(self):
        assert _retry_delay(0, "7") == 7.0

    def test_an_absurd_retry_after_is_still_capped(self):
        assert _retry_delay(0, "9999") == llm.MAX_RETRY_SLEEP_SECONDS

    def test_an_http_date_retry_after_falls_back_to_backoff(self):
        """Retry-After may be an HTTP-date; it must not crash the retry path."""
        assert _retry_delay(0, "Wed, 21 Oct 2026 07:28:00 GMT") == _retry_delay(0, None)

    def test_a_429_honours_retry_after(self, monkeypatch, _no_real_sleeping):
        attempts = []

        def flaky(req, timeout=None):
            attempts.append(1)
            if len(attempts) == 1:
                raise _http_error(429, headers={"Retry-After": "3"})
            return _FakeResponse(b"{}")

        monkeypatch.setattr(llm.request, "urlopen", flaky)
        http_json("GET", "https://x/v1/models", {})
        assert _no_real_sleeping == [3.0]


class TestResponseShape:
    def test_a_non_object_body_is_rejected_rather_than_crashing_later(self, monkeypatch):
        """A JSON array would otherwise fail much later as AttributeError on .get()."""
        monkeypatch.setattr(llm.request, "urlopen", lambda req, timeout=None: _FakeResponse(b"[1, 2]"))
        with pytest.raises(LlmProviderError) as excinfo:
            http_json("GET", "https://x/v1/models", {})
        assert excinfo.value.status_code == 502

    def test_invalid_json_is_reported_as_a_provider_error(self, monkeypatch):
        monkeypatch.setattr(llm.request, "urlopen", lambda req, timeout=None: _FakeResponse(b"<html>502</html>"))
        with pytest.raises(LlmProviderError) as excinfo:
            http_json("GET", "https://x/v1/models", {})
        assert "invalid JSON" in str(excinfo.value)


class TestUrlRedaction:
    def test_a_gemini_key_never_reaches_the_logs(self):
        redacted = _redact_url("https://generativelanguage.googleapis.com/v1beta/models/x:generateContent?key=SECRET123")
        assert "SECRET123" not in redacted
        assert "REDACTED" in redacted

    def test_other_query_parameters_survive(self):
        redacted = _redact_url("https://x/v1/models?key=SECRET&alt=sse")
        assert "alt=sse" in redacted
        assert "SECRET" not in redacted


class TestOpenWithRetryContract:
    def test_the_response_is_returned_unread_for_streaming_callers(self, monkeypatch):
        """Streaming parsers need the body still unconsumed."""
        monkeypatch.setattr(llm.request, "urlopen", lambda req, timeout=None: _FakeResponse(b"data: hello\n"))
        response = _open_with_retry("POST", "https://x/v1/chat/completions", {}, {"model": "m"})
        assert response.read() == b"data: hello\n"
