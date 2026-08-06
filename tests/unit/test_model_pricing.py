"""Whether a model is free, as reported rather than as guessed.

`model_is_free` is tri-state on purpose. Most providers' free tier covers the account, not
the model, and their `/models` payloads carry no pricing — so the only honest answer there
is "unknown", and the UI draws no badge. The two-state version of this function would put
"Free" on every Groq model (true today, and silently wrong the moment a paid tier appears)
or "not free" on all of them (wrong right now).
"""
import pytest

from services.llm_provider import model_is_free

pytestmark = pytest.mark.unit


class TestPricingIsRead:
    def test_zero_priced_openrouter_model_is_free(self):
        raw = {"pricing": {"prompt": "0", "completion": "0", "request": "0"}}
        assert model_is_free("openrouter", "meta/llama-3-8b:free", raw) is True

    def test_priced_model_is_not_free(self):
        raw = {"pricing": {"prompt": "0.0000005", "completion": "0.0000015"}}
        assert model_is_free("openrouter", "anthropic/claude-sonnet-latest", raw) is False

    def test_a_zero_prompt_but_priced_completion_is_not_free(self):
        """Both rates must be zero — a free prompt with a billed completion still bills."""
        raw = {"pricing": {"prompt": "0", "completion": "0.0000015"}}
        assert model_is_free("openrouter", "some/model", raw) is False

    def test_float_rates_are_accepted_as_well_as_strings(self):
        assert model_is_free("openrouter", "some/model", {"pricing": {"prompt": 0, "completion": 0}}) is True
        assert model_is_free("openrouter", "some/model", {"pricing": {"prompt": 0.5, "completion": 0}}) is False


class TestUnknownStaysUnknown:
    @pytest.mark.parametrize("provider", ["groq", "cerebras", "cloudflare", "sambanova", "zai", "gemini"])
    def test_providers_with_an_account_level_free_tier_report_unknown(self, provider):
        """Their payloads carry no per-model price, so no per-model claim is made."""
        assert model_is_free(provider, "some-model", {}) is None

    @pytest.mark.parametrize("provider", ["openai", "anthropic"])
    def test_paid_providers_without_pricing_also_report_unknown(self, provider):
        # Absence of evidence, not evidence of a price. The provider row already says the
        # account has no free tier.
        assert model_is_free(provider, "gpt-5", {}) is None

    def test_malformed_pricing_is_unknown_not_free(self):
        """The dangerous direction: junk must never round down to "free"."""
        for pricing in ({"prompt": "abc", "completion": "0"}, {"prompt": None}, {"completion": "0"}, "free"):
            assert model_is_free("openai", "gpt-5", {"pricing": pricing}) is None

    def test_a_seeded_row_with_no_raw_payload_is_unknown(self):
        assert model_is_free("groq", "llama-3.3-70b-versatile", {}) is None


class TestProviderSpecificRules:
    def test_ollama_is_always_free_because_it_runs_locally(self):
        assert model_is_free("ollama", "gemma3", {}) is True

    def test_openrouter_free_suffix_is_honoured_without_pricing(self):
        """A catalogue cached before pricing was read still labels the `:free` variants."""
        assert model_is_free("openrouter", "nvidia/nemotron-nano-9b-v2:free", {}) is True

    def test_the_free_suffix_does_not_override_a_real_price(self):
        """Pricing wins. If OpenRouter ever bills a `:free` id, the badge must follow the money."""
        raw = {"pricing": {"prompt": "0.0000005", "completion": "0.0000015"}}
        assert model_is_free("openrouter", "vendor/model:free", raw) is False

    def test_the_free_suffix_is_not_honoured_for_other_providers(self):
        """`:free` is OpenRouter's id convention, not a universal one."""
        assert model_is_free("openai", "gpt-5:free", {}) is None
