"""A provider may not hold a request open indefinitely without answering.

`LLM_REQUEST_TIMEOUT_SECONDS` is a per-socket-read timeout, so it bounds *silence* and
nothing else. A provider that emits keep-alive comment lines resets it on every one —
which is exactly what they are for. OpenRouter's `: OPENROUTER PROCESSING` held a real
request open for 124 seconds under a 45-second timeout while a queued free-tier model
produced nothing, and the user watched a typing indicator for two minutes.

These pin the wall-clock budget that bounds the wait for the FIRST token, and pin equally
hard that it does not bound generation: a long answer that streams steadily must never be
cut off, no matter how long it takes.
"""
import json

import pytest

import services.llm_provider as llm
from services.llm_provider import LlmProviderError, stream_openai_chat

pytestmark = pytest.mark.unit


class _Clock:
    """A monotonic clock the test drives, standing in for the module's `time`."""

    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:  # pragma: no cover - retry path is not exercised
        self.now += seconds


class _KeepAliveStream:
    """Emits comment lines — successful reads that carry no answer — then optionally data.

    Each line advances the clock, which is how a real stall accumulates: the socket keeps
    delivering, so the read timeout never fires.
    """

    def __init__(self, clock: _Clock, keep_alives: int, seconds_each: float, tail: list[str] | None = None):
        self._clock = clock
        self._keep_alives = keep_alives
        self._seconds_each = seconds_each
        self._tail = tail or []
        self.closed = False

    def __iter__(self):
        for _ in range(self._keep_alives):
            self._clock.now += self._seconds_each
            yield b": OPENROUTER PROCESSING\n"
        for line in self._tail:
            self._clock.now += self._seconds_each
            yield line.encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.closed = True
        return False


def _delta(text: str) -> str:
    return "data: " + json.dumps({"choices": [{"delta": {"content": text}}]}) + "\n"


def _install(monkeypatch, clock, stream, budget=60):
    monkeypatch.setattr(llm, "time", clock)
    monkeypatch.setattr(llm, "_open_with_retry", lambda method, url, headers, payload: stream)
    settings = llm.get_settings()
    monkeypatch.setattr(
        llm,
        "get_settings",
        lambda: type("S", (), {**{k: getattr(settings, k) for k in dir(settings) if not k.startswith("_")},
                               "llm_first_token_timeout_seconds": budget})(),
    )


class TestFirstTokenDeadline:
    def test_gives_up_on_a_provider_that_only_sends_keep_alives(self, monkeypatch):
        clock = _Clock()
        # 20 keep-alives, 10 s apart: 200 s of "still working" and not one token.
        _install(monkeypatch, clock, _KeepAliveStream(clock, keep_alives=20, seconds_each=10.0))

        with pytest.raises(LlmProviderError) as excinfo:
            list(stream_openai_chat("k", "https://x/v1", "m", "prompt"))

        assert excinfo.value.status_code == 504
        assert "no output" in str(excinfo.value)

    def test_gives_up_near_the_budget_rather_than_after_the_whole_stall(self, monkeypatch):
        clock = _Clock()
        _install(monkeypatch, clock, _KeepAliveStream(clock, keep_alives=60, seconds_each=5.0), budget=60)

        with pytest.raises(LlmProviderError):
            list(stream_openai_chat("k", "https://x/v1", "m", "prompt"))

        # The check runs per line, so it fires within one keep-alive interval of the budget —
        # not after the provider finally gives up 240 s later.
        assert clock.now <= 70.0

    def test_a_slow_first_token_that_arrives_in_time_is_kept(self, monkeypatch):
        clock = _Clock()
        stream = _KeepAliveStream(clock, keep_alives=4, seconds_each=5.0, tail=[_delta("Hello")])
        _install(monkeypatch, clock, stream, budget=60)

        assert list(stream_openai_chat("k", "https://x/v1", "m", "prompt")) == ["Hello"]

    def test_generation_itself_is_never_bounded(self, monkeypatch):
        """The budget covers the wait for the first token, not the answer.

        A model that streams steadily for ten minutes is working. Cutting it off would
        turn the fix for a dead provider into a truncation bug on a healthy one.
        """
        clock = _Clock()
        # First token lands at t=2s, comfortably inside the 5s budget; the answer then keeps
        # streaming until t=9s, well past it.
        letters = ["a", "b", "c", "d", "e", "f", "g", "h"]
        stream = _KeepAliveStream(
            clock, keep_alives=1, seconds_each=1.0, tail=[_delta(letter) for letter in letters]
        )
        _install(monkeypatch, clock, stream, budget=5)

        assert list(stream_openai_chat("k", "https://x/v1", "m", "prompt")) == letters
        assert clock.now > 5.0

    def test_a_zero_budget_disables_the_deadline(self, monkeypatch):
        """An operator must be able to turn this off without editing code."""
        clock = _Clock()
        stream = _KeepAliveStream(clock, keep_alives=50, seconds_each=10.0, tail=[_delta("late")])
        _install(monkeypatch, clock, stream, budget=0)

        assert list(stream_openai_chat("k", "https://x/v1", "m", "prompt")) == ["late"]
