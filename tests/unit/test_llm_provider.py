"""Unit tests for the LLM provider registry and dispatch layer.

The registry (SUPPORTED_PROVIDERS) is metadata-driven: api_style selects the wire
protocol, so an OpenAI-compatible provider is a data-only addition. These tests pin
the registry contract, the style dispatch, the Cloudflare model-listing shim, the
generation-parameter plumbing and the extractive fallback when a provider fails."""
import json

import pytest

import services.llm_provider as llm
from models.llm import LlmProviderCredential, LlmUserPreference
# NOTE: llm.test_provider_connection is referenced via the module — importing the
# name directly would make pytest collect the service function itself as a test.
from services.llm_provider import (
    LlmProviderError,
    SEED_MODELS,
    SUPPORTED_PROVIDERS,
    _require_resolved_base_url,
    call_openai_chat,
    encrypt_secret,
    ensure_seed_models,
    fetch_provider_models,
    generation_params,
    list_provider_options,
    normalize_provider,
    set_user_preference,
)

pytestmark = pytest.mark.unit

DISPATCHABLE_STYLES = {"openai", "anthropic", "gemini", "ollama"}

# api_style -> the shared suffix of its call_*/stream_* pair.
_STYLE_FUNCTION_SUFFIX = {
    "openai": "openai_chat",
    "anthropic": "anthropic_messages",
    "gemini": "gemini_generate",
    "ollama": "ollama_generate",
}


class TestRegistryIntegrity:
    """A malformed registry entry fails at request time with a KeyError → bare 500,
    so the shape is enforced here instead."""

    def test_every_provider_has_the_required_metadata_keys(self):
        required = {"display_name", "credential_required", "api_style", "default_base_url", "docs_url", "console_url", "free_tier"}
        for provider, metadata in SUPPORTED_PROVIDERS.items():
            missing = required - metadata.keys()
            assert not missing, f"{provider} is missing {missing}"

    def test_every_api_style_is_dispatchable(self):
        for provider, metadata in SUPPORTED_PROVIDERS.items():
            assert metadata["api_style"] in DISPATCHABLE_STYLES, provider

    def test_every_provider_has_seed_models(self):
        for provider in SUPPORTED_PROVIDERS:
            assert SEED_MODELS.get(provider), f"{provider} has no seed models"

    def test_free_tier_is_none_or_the_documented_shape(self):
        for provider, metadata in SUPPORTED_PROVIDERS.items():
            free_tier = metadata["free_tier"]
            if free_tier is None:
                continue
            assert set(free_tier.keys()) == {
                "summary",
                "limits",
                "data_policy",
                "notes",
                "data_training",
            }, provider
            assert all(isinstance(value, str) and value for value in free_tier.values()), provider

    def test_data_training_is_a_known_enum_value(self):
        """The UI renders this as a privacy badge, so an unrecognised value would silently
        show nothing on exactly the providers a user most needs warning about."""
        for provider, metadata in SUPPORTED_PROVIDERS.items():
            free_tier = metadata["free_tier"]
            if free_tier is None:
                continue
            assert free_tier["data_training"] in {"no", "yes", "opt_out", "unknown"}, provider

    def test_the_data_training_flag_agrees_with_the_prose(self):
        """The badge and the paragraph beneath it must not contradict each other."""
        assert SUPPORTED_PROVIDERS["groq"]["free_tier"]["data_training"] == "no"
        assert SUPPORTED_PROVIDERS["gemini"]["free_tier"]["data_training"] == "yes"
        assert SUPPORTED_PROVIDERS["mistral"]["free_tier"]["data_training"] == "opt_out"
        assert SUPPORTED_PROVIDERS["sambanova"]["free_tier"]["data_training"] == "unknown"
        assert SUPPORTED_PROVIDERS["ollama"]["free_tier"]["data_training"] == "no"

    def test_only_cloudflare_ships_a_placeholder_base_url(self):
        for provider, metadata in SUPPORTED_PROVIDERS.items():
            if provider == "cloudflare":
                assert "{account_id}" in metadata["default_base_url"]
            else:
                assert "{" not in metadata["default_base_url"], provider

    def test_the_free_provider_lineup_is_registered(self):
        for provider in ("groq", "cerebras", "mistral", "cloudflare", "sambanova", "zai"):
            assert provider in SUPPORTED_PROVIDERS
            assert SUPPORTED_PROVIDERS[provider]["free_tier"] is not None

    def test_every_api_style_can_both_complete_and_stream(self):
        """A style with no streaming parser would fall through to "Unsupported LLM
        provider" the moment a user picked that provider for a chat."""
        for provider, metadata in SUPPORTED_PROVIDERS.items():
            style = metadata["api_style"]
            assert callable(getattr(llm, f"call_{_STYLE_FUNCTION_SUFFIX[style]}", None)), provider
            assert callable(getattr(llm, f"stream_{_STYLE_FUNCTION_SUFFIX[style]}", None)), provider


class TestNormalizeProvider:
    def test_accepts_the_new_providers(self):
        for provider in ("groq", "cerebras", "mistral", "cloudflare", "sambanova", "zai"):
            assert normalize_provider(provider) == provider

    def test_normalizes_case_and_hyphens(self):
        assert normalize_provider(" Groq ") == "groq"

    def test_rejects_unknown_providers(self):
        with pytest.raises(LlmProviderError):
            normalize_provider("togetherai")


class TestRequireResolvedBaseUrl:
    def test_placeholder_raises_a_400(self):
        with pytest.raises(LlmProviderError) as excinfo:
            _require_resolved_base_url("https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1")
        assert excinfo.value.status_code == 400

    def test_resolved_url_passes(self):
        _require_resolved_base_url("https://api.groq.com/openai/v1")


class TestFetchProviderModels:
    def test_openai_style_maps_the_openai_list_shape(self, monkeypatch):
        monkeypatch.setattr(
            llm,
            "http_json",
            lambda method, url, headers, payload=None: {"data": [{"id": "llama-3.3-70b-versatile"}, {"id": "x", "name": "X Pretty"}]},
        )
        models = fetch_provider_models("groq", "key", "https://api.groq.com/openai/v1")
        assert [m["model_id"] for m in models] == ["llama-3.3-70b-versatile", "x"]
        assert models[1]["display_name"] == "X Pretty"

    def test_cloudflare_uses_the_native_catalog_and_keeps_only_text_generation(self, monkeypatch):
        calls = {}

        def fake_http(method, url, headers, payload=None):
            calls["url"] = url
            return {
                "result": [
                    {"name": "@cf/openai/gpt-oss-120b", "task": {"name": "Text Generation"}},
                    {"name": "@cf/baai/bge-m3", "task": {"name": "Text Embeddings"}},
                ]
            }

        monkeypatch.setattr(llm, "http_json", fake_http)
        base = "https://api.cloudflare.com/client/v4/accounts/abc123/ai/v1"
        models = fetch_provider_models("cloudflare", "token", base)
        assert calls["url"] == "https://api.cloudflare.com/client/v4/accounts/abc123/ai/models/search?per_page=100"
        assert [m["model_id"] for m in models] == ["@cf/openai/gpt-oss-120b"]

    def test_cloudflare_placeholder_base_url_fails_before_any_http(self, monkeypatch):
        def explode(*args, **kwargs):
            raise AssertionError("http_json must not be called")

        monkeypatch.setattr(llm, "http_json", explode)
        with pytest.raises(LlmProviderError):
            fetch_provider_models("cloudflare", "token", SUPPORTED_PROVIDERS["cloudflare"]["default_base_url"])


class TestGenerationParams:
    def test_defaults_match_the_previous_hardcoded_values(self):
        params = generation_params(None)
        assert params == {"temperature": llm.DEFAULT_TEMPERATURE, "max_tokens": None, "top_p": None}

    def test_preference_values_override_the_defaults(self):
        preference = LlmUserPreference(user_id="u", provider="groq", model_id="m", temperature=0.7, max_tokens=800, top_p=0.9)
        assert generation_params(preference) == {"temperature": 0.7, "max_tokens": 800, "top_p": 0.9}


class TestCallOpenAiChatPayload:
    def _capture(self, monkeypatch):
        captured = {}

        def fake_http(method, url, headers, payload=None):
            captured["payload"] = payload
            return {"choices": [{"message": {"content": "answer"}}]}

        monkeypatch.setattr(llm, "http_json", fake_http)
        return captured

    def test_default_payload_keeps_the_original_wire_format(self, monkeypatch):
        captured = self._capture(monkeypatch)
        call_openai_chat("key", "https://api.groq.com/openai/v1", "m", "prompt")
        assert captured["payload"]["temperature"] == llm.DEFAULT_TEMPERATURE
        assert "max_tokens" not in captured["payload"]
        assert "top_p" not in captured["payload"]

    def test_user_params_are_forwarded(self, monkeypatch):
        captured = self._capture(monkeypatch)
        call_openai_chat("key", "https://x/v1", "m", "prompt", {"temperature": 1.0, "max_tokens": 512, "top_p": 0.5})
        assert captured["payload"]["temperature"] == 1.0
        assert captured["payload"]["max_tokens"] == 512
        assert captured["payload"]["top_p"] == 0.5


class TestSystemPromptDelivery:
    """Every wire protocol must actually carry the system prompt.

    The conversational path relies on it for its guardrails — a provider that silently
    dropped the system prompt would answer from model knowledge with no restrictions at
    all, which is exactly what the feature exists to prevent. Each API has a different
    field for it, so this is checked per style rather than assumed."""

    SENTINEL = "GUARDRAIL-SENTINEL"

    def _capture(self, monkeypatch):
        captured = {}

        def fake_http(method, url, headers, payload=None):
            captured["payload"] = payload
            return {
                "choices": [{"message": {"content": "ok"}}],
                "content": [{"type": "text", "text": "ok"}],
                "candidates": [{"content": {"parts": [{"text": "ok"}]}}],
                "response": "ok",
            }

        monkeypatch.setattr(llm, "http_json", fake_http)
        return captured

    def test_openai_style_sends_a_system_message(self, monkeypatch):
        captured = self._capture(monkeypatch)
        llm.call_openai_chat("k", "https://x/v1", "m", "p", None, self.SENTINEL)
        assert captured["payload"]["messages"][0] == {"role": "system", "content": self.SENTINEL}

    def test_anthropic_uses_the_top_level_system_field(self, monkeypatch):
        captured = self._capture(monkeypatch)
        llm.call_anthropic_messages("k", "https://x/v1", "m", "p", None, self.SENTINEL)
        assert captured["payload"]["system"] == self.SENTINEL

    def test_gemini_uses_system_instruction(self, monkeypatch):
        captured = self._capture(monkeypatch)
        llm.call_gemini_generate("k", "https://x/v1beta", "m", "p", None, self.SENTINEL)
        assert captured["payload"]["systemInstruction"] == {"parts": [{"text": self.SENTINEL}]}

    def test_ollama_uses_the_system_field(self, monkeypatch):
        captured = self._capture(monkeypatch)
        llm.call_ollama_generate("http://x", "m", "p", None, self.SENTINEL)
        assert captured["payload"]["system"] == self.SENTINEL

    def test_grounded_answers_still_get_the_grounded_system_prompt(self, monkeypatch):
        captured = self._capture(monkeypatch)
        llm.call_openai_chat("k", "https://x/v1", "m", "p")
        assert captured["payload"]["messages"][0]["content"] == llm.GROUNDED_SYSTEM_PROMPT


class TestSetUserPreference:
    @pytest.fixture(autouse=True)
    def _seed_catalog(self, db):
        # The test sessionmaker runs autoflush=False, so seeds must be committed
        # before set_user_preference can find the model in the catalog.
        ensure_seed_models(db, "groq")
        db.commit()

    def test_model_only_save_does_not_clobber_tuned_params(self, db):
        set_user_preference(db, "user-1", "groq", "llama-3.3-70b-versatile", temperature=0.9, max_tokens=700, top_p=0.8, update_generation_params=True)
        # The plain dropdown path: provider/model only, params not sent.
        result = set_user_preference(db, "user-1", "groq", "llama-3.1-8b-instant")
        assert result["model_id"] == "llama-3.1-8b-instant"
        assert result["temperature"] == 0.9
        assert result["max_tokens"] == 700
        assert result["top_p"] == 0.8

    def test_explicit_params_update_replaces_them(self, db):
        set_user_preference(db, "user-2", "groq", "llama-3.3-70b-versatile", temperature=0.9, update_generation_params=True)
        result = set_user_preference(db, "user-2", "groq", "llama-3.3-70b-versatile", temperature=None, update_generation_params=True)
        assert result["temperature"] is None


class TestProviderOptions:
    def test_serializes_free_tier_and_console_metadata(self, db):
        options = list_provider_options(db, "user-3")
        by_provider = {entry["provider"]: entry for entry in options["providers"]}
        assert by_provider["groq"]["free_tier"]["summary"].startswith("Permanent free tier")
        assert by_provider["groq"]["console_url"]
        assert by_provider["groq"]["api_style"] == "openai"
        assert by_provider["openai"]["free_tier"] is None


class TestConnectionTest:
    def test_success_marks_the_credential_valid(self, db, monkeypatch):
        credential = LlmProviderCredential(
            user_id="user-4",
            provider="groq",
            encrypted_api_key=encrypt_secret("sk-test"),
            base_url="https://api.groq.com/openai/v1",
        )
        db.add(credential)
        db.commit()
        monkeypatch.setattr(llm, "fetch_provider_models", lambda *a, **k: [{"model_id": "m"}])
        result = llm.test_provider_connection(db, "user-4", "groq")
        assert result["ok"] is True
        assert result["validation_status"] == "valid"
        db.refresh(credential)
        assert credential.validation_status == "valid"

    def test_failure_marks_the_credential_invalid_with_the_reason(self, db, monkeypatch):
        credential = LlmProviderCredential(
            user_id="user-5",
            provider="groq",
            encrypted_api_key=encrypt_secret("sk-bad"),
            base_url="https://api.groq.com/openai/v1",
        )
        db.add(credential)
        db.commit()

        def boom(*args, **kwargs):
            raise LlmProviderError("Provider API returned HTTP 401: bad key", 502)

        monkeypatch.setattr(llm, "fetch_provider_models", boom)
        result = llm.test_provider_connection(db, "user-5", "groq")
        assert result["ok"] is False
        assert "401" in result["error"]
        db.refresh(credential)
        assert credential.validation_status == "invalid"

    def test_credential_required_provider_without_credential_is_a_400(self, db):
        with pytest.raises(LlmProviderError) as excinfo:
            llm.test_provider_connection(db, "user-6", "groq")
        assert excinfo.value.status_code == 400

    def test_ollama_can_be_tested_without_a_credential(self, db, monkeypatch):
        monkeypatch.setattr(llm, "fetch_provider_models", lambda *a, **k: [{"model_id": "llama3.3"}])
        result = llm.test_provider_connection(db, "user-7", "ollama")
        assert result["ok"] is True
