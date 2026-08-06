"""Unit tests for the config helpers and the production CORS guard."""
import pytest

from core.config import _optional_int, _required, _split_csv, get_settings

pytestmark = pytest.mark.unit


class TestRequired:
    def test_returns_stripped_value(self, monkeypatch):
        monkeypatch.setenv("X_REQ", "  hello  ")
        assert _required("X_REQ") == "hello"

    def test_raises_when_missing(self, monkeypatch):
        monkeypatch.delenv("X_REQ_MISSING", raising=False)
        with pytest.raises(RuntimeError, match="Missing required"):
            _required("X_REQ_MISSING")

    def test_raises_when_blank(self, monkeypatch):
        monkeypatch.setenv("X_REQ_BLANK", "   ")
        with pytest.raises(RuntimeError):
            _required("X_REQ_BLANK")


class TestOptionalInt:
    def test_default_when_unset(self, monkeypatch):
        monkeypatch.delenv("X_INT", raising=False)
        assert _optional_int("X_INT", 7) == 7

    def test_parses_a_value(self, monkeypatch):
        monkeypatch.setenv("X_INT", "42")
        assert _optional_int("X_INT", 7) == 42

    def test_raises_on_non_integer(self, monkeypatch):
        monkeypatch.setenv("X_INT", "not-a-number")
        with pytest.raises(RuntimeError, match="must be an integer"):
            _optional_int("X_INT", 7)


class TestSplitCsv:
    def test_splits_and_strips(self, monkeypatch):
        monkeypatch.setenv("X_CSV", "a, b ,c")
        assert _split_csv("X_CSV", "") == ("a", "b", "c")

    def test_raises_when_empty(self, monkeypatch):
        monkeypatch.setenv("X_CSV", " , ,")
        with pytest.raises(RuntimeError):
            _split_csv("X_CSV", "")


class TestProductionCorsGuard:
    def test_wildcard_origin_is_rejected_in_production(self, monkeypatch):
        monkeypatch.setenv("ENVIRONMENT", "production")
        monkeypatch.setenv("CORS_ORIGINS", "*")
        get_settings.cache_clear()
        try:
            with pytest.raises(RuntimeError, match="cannot contain"):
                get_settings()
        finally:
            # Restore the cache so later tests rebuild from the (restored) test env.
            get_settings.cache_clear()
