"""Unit tests for `rag/precision/config.py` — the whole tunable surface of the
High-Precision Non-LLM RAG mode.

Three reasons this file earns its place, and all three fail *silently* in production:

* **A malformed environment value must never take chat down.** `get_precision_config()`
  runs per question, inside the answer path, and there is no global exception handler
  (CLAUDE.md §3) — a `ValueError` raised while parsing a weight would surface as a bare
  text/plain 500 on every question a deployment asks. Every helper here is written to
  clamp-and-continue, and that is a promise only a test can hold.
* **The documented defaults are the mode's behaviour.** Nothing outside this module reads
  the environment, so these constants *are* what a knowledge base gets. Two of them are
  safety properties rather than tuning: `negative_terms` ships EMPTY (a built-in list of
  "bad" words would be an unauditable opinion about someone else's documents), and the
  expansion / PRF / reranker weights all sit below 1.0 so a guess can never outweigh the
  words the user actually typed or replace the retrieval ranking outright.
* **`get_precision_config` is `lru_cache`d.** A test that sets `PRECISION_RAG_*` without
  clearing the cache on the way *out* poisons every later test in the session with a
  configuration none of them asked for — `monkeypatch` restores the environment but cannot
  reach inside the cache. Hence the `precision_env` fixture below, which clears in both
  directions, and the test that pins the trap so nobody re-discovers it by accident.
"""
import dataclasses
import math
import os

import pytest

from rag.precision.config import (
    RERANKER_HTTP,
    RERANKER_LEXICAL,
    RERANKER_NONE,
    RERANKERS,
    PrecisionConfig,
    _csv,
    _flag,
    _float,
    _int,
    get_precision_config,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def precision_env(monkeypatch):
    """A clean `PRECISION_RAG_*` environment and an empty config cache, both restored.

    The cache is cleared on the way IN so a value left behind by an earlier test cannot be
    mistaken for a default, and on the way OUT so a value set by this test cannot outlive
    the environment `monkeypatch` is about to put back.
    """
    for name in [key for key in os.environ if key.startswith("PRECISION_RAG_")]:
        monkeypatch.delenv(name, raising=False)
    get_precision_config.cache_clear()
    yield monkeypatch
    get_precision_config.cache_clear()


class TestDocumentedDefaults:
    def test_retrieval_and_fusion_defaults(self):
        config = PrecisionConfig()
        assert config.enabled is True
        assert config.bm25_enabled is True and config.dense_enabled is True
        assert config.candidate_k == 100 and config.final_k == 10
        # The TREC-settled Okapi constants; every comparable implementation uses these, so
        # a drift here silently makes this mode's BM25 incomparable with published numbers.
        assert config.bm25_k1 == 1.2 and config.bm25_b == 0.75
        assert config.dense_weight == 1.0 and config.bm25_weight == 1.0
        assert config.metadata_weight == 0.35 and config.keyword_weight == 0.5

    def test_query_understanding_defaults(self):
        config = PrecisionConfig()
        assert config.normalization_enabled is True
        assert config.query_expansion_enabled is True
        assert config.expansion_max_terms == 12
        assert config.expansion_term_weight == 0.45
        assert config.prf_enabled is True
        assert config.prf_documents == 5 and config.prf_terms == 6
        assert config.prf_term_weight == 0.25
        assert config.entity_extraction_enabled is True

    def test_terms_the_user_did_not_type_are_weighted_below_one(self):
        # An expansion and a PRF term are both guesses about intent. Weighting either at or
        # above 1.0 would let a word nobody typed outrank one that was, which is exactly the
        # failure a "high precision" mode exists to avoid.
        config = PrecisionConfig()
        assert 0.0 < config.expansion_term_weight < 1.0
        assert 0.0 < config.prf_term_weight < 1.0

    def test_filtering_and_reranking_defaults(self):
        config = PrecisionConfig()
        assert config.metadata_filtering_enabled is True
        assert config.metadata_filter_min_survivors == 12
        assert config.reranker_enabled is True
        assert config.reranker_backend == RERANKER_LEXICAL
        assert config.reranker_top_k == 20
        assert config.reranker_model == "BAAI/bge-reranker-base"
        assert config.reranker_endpoint == ""
        assert config.reranker_timeout_seconds == 10
        # Below 1.0 on purpose: a misconfigured reranker then degrades the retrieval
        # ordering instead of replacing it outright.
        assert config.reranker_weight == 0.75
        assert config.reranker_weight < 1.0

    def test_negative_terms_ship_empty(self):
        # A shipped list of "bad" words is a silent opinion about documents this engine has
        # never seen. Domain terms belong in a deployment's own configuration.
        config = PrecisionConfig()
        assert config.negative_signals_enabled is True
        assert config.negative_terms == ()
        assert config.negative_penalty == 0.35

    def test_diversity_and_structure_defaults(self):
        config = PrecisionConfig()
        assert config.dedup_enabled is True and config.dedup_threshold == 0.90
        assert config.mmr_enabled is True and config.mmr_lambda == 0.7
        assert config.parent_child_enabled is True
        assert config.parent_group_size == 3
        assert config.parent_max_chars == 2400

    def test_observability_defaults(self):
        config = PrecisionConfig()
        assert config.trace_enabled is True
        assert config.trace_max_candidates_logged == 25

    def test_reranker_backend_constants(self):
        assert RERANKERS == (RERANKER_LEXICAL, RERANKER_HTTP, RERANKER_NONE)
        assert (RERANKER_LEXICAL, RERANKER_HTTP, RERANKER_NONE) == ("lexical", "http", "none")

    def test_environment_defaults_equal_the_dataclass_defaults(self, precision_env):
        # The dataclass declares a default AND `get_precision_config` passes a default to
        # every `_int`/`_float`/`_flag` call. Two copies of the same number is exactly the
        # shape that drifts, and the drift is invisible: whichever copy is wrong only shows
        # up in the half of the codebase that reads it.
        from_env = get_precision_config()
        declared = PrecisionConfig()
        for field in dataclasses.fields(PrecisionConfig):
            assert getattr(from_env, field.name) == getattr(declared, field.name), field.name


class TestFlagHelper:
    @pytest.mark.parametrize("raw", ["1", "true", "yes", "on", "TRUE", "  On  "])
    def test_truthy_spellings(self, monkeypatch, raw):
        monkeypatch.setenv("X_FLAG", raw)
        assert _flag("X_FLAG", False) is True

    @pytest.mark.parametrize("raw", ["0", "false", "no", "off", "FALSE"])
    def test_falsey_spellings(self, monkeypatch, raw):
        monkeypatch.setenv("X_FLAG", raw)
        assert _flag("X_FLAG", True) is False

    def test_unrecognised_value_reads_as_false_not_as_the_default(self, monkeypatch):
        # Worth pinning because it is asymmetric with `_int`/`_float`, which fall back to
        # their default on garbage. Anything outside the truthy set switches a feature OFF.
        monkeypatch.setenv("X_FLAG", "maybe")
        assert _flag("X_FLAG", True) is False

    def test_empty_falls_back_to_the_default(self, monkeypatch):
        monkeypatch.setenv("X_FLAG", "   ")
        assert _flag("X_FLAG", True) is True
        assert _flag("X_FLAG", False) is False

    def test_unset_falls_back_to_the_default(self, monkeypatch):
        monkeypatch.delenv("X_FLAG", raising=False)
        assert _flag("X_FLAG", True) is True


class TestIntHelper:
    def test_parses_a_value_within_range(self, monkeypatch):
        monkeypatch.setenv("X_INT", " 42 ")
        assert _int("X_INT", 7, 0, 100) == 42

    def test_clamps_below_the_floor_and_above_the_ceiling(self, monkeypatch):
        monkeypatch.setenv("X_INT", "-5")
        assert _int("X_INT", 7, 1, 10) == 1
        monkeypatch.setenv("X_INT", "9999")
        assert _int("X_INT", 7, 1, 10) == 10

    @pytest.mark.parametrize("raw", ["not-a-number", "3.7", "12abc", "--"])
    def test_non_integer_falls_back_rather_than_raising(self, monkeypatch, raw):
        # Clamp-and-continue: a typo in one weight must not 500 every question. Note "3.7"
        # is in here deliberately — `int()` refuses a float string, so a value that looks
        # perfectly reasonable in a `.env` silently reverts to the default.
        monkeypatch.setenv("X_INT", raw)
        assert _int("X_INT", 7, 0, 100) == 7

    def test_empty_and_unset_fall_back_to_the_default(self, monkeypatch):
        monkeypatch.setenv("X_INT", "  ")
        assert _int("X_INT", 7, 0, 100) == 7
        monkeypatch.delenv("X_INT", raising=False)
        assert _int("X_INT", 7, 0, 100) == 7


class TestFloatHelper:
    def test_parses_a_value_within_range(self, monkeypatch):
        monkeypatch.setenv("X_FLOAT", " 0.42 ")
        assert _float("X_FLOAT", 0.7, 0.0, 1.0) == pytest.approx(0.42)

    def test_accepts_an_integer_spelling(self, monkeypatch):
        monkeypatch.setenv("X_FLOAT", "1")
        assert _float("X_FLOAT", 0.7, 0.0, 1.0) == pytest.approx(1.0)

    def test_clamps_below_the_floor_and_above_the_ceiling(self, monkeypatch):
        monkeypatch.setenv("X_FLOAT", "-2.5")
        assert _float("X_FLOAT", 0.7, 0.0, 1.0) == 0.0
        monkeypatch.setenv("X_FLOAT", "5")
        assert _float("X_FLOAT", 0.7, 0.0, 1.0) == 1.0

    @pytest.mark.parametrize("raw", ["not-a-number", "0.5.1", "1,5"])
    def test_garbage_falls_back_rather_than_raising(self, monkeypatch, raw):
        # "1,5" is the comma decimal separator half the world writes; `float()` rejects it,
        # and the default is a defensible answer where the typo is not.
        monkeypatch.setenv("X_FLOAT", raw)
        assert _float("X_FLOAT", 0.7, 0.0, 1.0) == pytest.approx(0.7)

    def test_empty_falls_back_to_the_default(self, monkeypatch):
        monkeypatch.setenv("X_FLOAT", "  ")
        assert _float("X_FLOAT", 0.7, 0.0, 1.0) == pytest.approx(0.7)

    @pytest.mark.parametrize("raw", ["nan", "inf", "-inf"])
    def test_ieee_specials_are_clamped_to_a_real_number(self, monkeypatch, raw):
        # `float()` happily parses these, so the `except ValueError` never sees them — the
        # clamp is what stops them through. It matters: a NaN weight compares False against
        # everything, so it would not error, it would quietly flatten the ranking it feeds.
        monkeypatch.setenv("X_FLOAT", raw)
        value = _float("X_FLOAT", 0.7, 0.0, 1.0)
        assert math.isfinite(value)
        assert 0.0 <= value <= 1.0

    def test_unset_falls_back_to_the_default(self, monkeypatch):
        monkeypatch.delenv("X_FLOAT", raising=False)
        assert _float("X_FLOAT", 0.7, 0.0, 1.0) == pytest.approx(0.7)


class TestCsvHelper:
    def test_lowercases_strips_and_drops_blanks(self, monkeypatch):
        monkeypatch.setenv("X_CSV", " Draft , ,  DEPRECATED ,archived")
        assert _csv("X_CSV", ()) == ("draft", "deprecated", "archived")

    def test_lowercasing_is_what_makes_the_list_usable(self, monkeypatch):
        # Terms are matched against tokenized text, which is lowercase everywhere in this
        # codebase (`_tokenize`). A capitalised entry that kept its case would match nothing
        # and report no error — the whole list would just quietly do nothing.
        monkeypatch.setenv("X_CSV", "Obsolete")
        assert _csv("X_CSV", ()) == ("obsolete",)

    def test_empty_and_unset_fall_back_to_the_default(self, monkeypatch):
        monkeypatch.setenv("X_CSV", "   ")
        assert _csv("X_CSV", ("fallback",)) == ("fallback",)
        monkeypatch.delenv("X_CSV", raising=False)
        assert _csv("X_CSV", ("fallback",)) == ("fallback",)

    def test_separators_only_yields_an_empty_tuple_not_the_default(self, monkeypatch):
        # `,,,` is not blank, so the emptiness check does not catch it and every item is
        # dropped by the comprehension. The result is a deliberate "no terms" rather than a
        # revert — which is the right reading of someone writing a list of nothing.
        monkeypatch.setenv("X_CSV", " , , ")
        assert _csv("X_CSV", ("fallback",)) == ()


class TestGetPrecisionConfig:
    def test_reads_flags_ints_floats_and_csv_from_the_environment(self, precision_env):
        precision_env.setenv("PRECISION_RAG_MMR", "off")
        precision_env.setenv("PRECISION_RAG_CANDIDATE_K", "250")
        precision_env.setenv("PRECISION_RAG_RERANKER_WEIGHT", "0.5")
        precision_env.setenv("PRECISION_RAG_NEGATIVE_TERMS", "Draft, superseded ")
        get_precision_config.cache_clear()

        config = get_precision_config()

        assert config.mmr_enabled is False
        assert config.candidate_k == 250
        assert config.reranker_weight == pytest.approx(0.5)
        assert config.negative_terms == ("draft", "superseded")

    def test_reads_the_string_settings(self, precision_env):
        precision_env.setenv("PRECISION_RAG_RERANKER_MODEL", "  BAAI/bge-reranker-large  ")
        precision_env.setenv("PRECISION_RAG_RERANKER_ENDPOINT", " http://localhost:8080/rerank ")
        get_precision_config.cache_clear()

        config = get_precision_config()

        assert config.reranker_model == "BAAI/bge-reranker-large"
        assert config.reranker_endpoint == "http://localhost:8080/rerank"

    @pytest.mark.parametrize("raw", [RERANKER_HTTP, RERANKER_NONE, "  HTTP  "])
    def test_recognised_backends_are_accepted_case_and_space_insensitively(self, precision_env, raw):
        precision_env.setenv("PRECISION_RAG_RERANKER_BACKEND", raw)
        get_precision_config.cache_clear()
        assert get_precision_config().reranker_backend == raw.strip().lower()

    @pytest.mark.parametrize("raw", ["cohere", "", "   ", "lexical-v2"])
    def test_unknown_backend_falls_back_to_lexical(self, precision_env, raw):
        # The alternative is an "Unsupported reranker" explosion the first time anyone asks
        # a question. The lexical scorer needs no service and no download, so falling back
        # to it always leaves the mode able to answer.
        precision_env.setenv("PRECISION_RAG_RERANKER_BACKEND", raw)
        get_precision_config.cache_clear()
        assert get_precision_config().reranker_backend == RERANKER_LEXICAL

    def test_garbage_numeric_values_do_not_raise(self, precision_env):
        # The whole point of the clamp-and-continue helpers: this call happens inside the
        # answer path, and an exception here is a bare 500 on every question.
        precision_env.setenv("PRECISION_RAG_CANDIDATE_K", "lots")
        precision_env.setenv("PRECISION_RAG_BM25_K1", "high")
        precision_env.setenv("PRECISION_RAG_FINAL_K", "10.5")
        precision_env.setenv("PRECISION_RAG_PARENT_MAX_CHARS", "")
        get_precision_config.cache_clear()

        config = get_precision_config()

        assert config.candidate_k == 100
        assert config.bm25_k1 == 1.2
        assert config.final_k == 10
        assert config.parent_max_chars == 2400

    def test_out_of_range_values_are_clamped_to_the_documented_bounds(self, precision_env):
        precision_env.setenv("PRECISION_RAG_CANDIDATE_K", "5")          # floor 10
        precision_env.setenv("PRECISION_RAG_FINAL_K", "500")            # ceiling 50
        precision_env.setenv("PRECISION_RAG_MMR_LAMBDA", "4")           # ceiling 1.0
        precision_env.setenv("PRECISION_RAG_BM25_B", "-3")              # floor 0.0
        precision_env.setenv("PRECISION_RAG_EXPANSION_WEIGHT", "9")     # ceiling 1.0
        get_precision_config.cache_clear()

        config = get_precision_config()

        assert config.candidate_k == 10
        assert config.final_k == 50
        assert config.mmr_lambda == 1.0
        assert config.bm25_b == 0.0
        assert config.expansion_term_weight == 1.0

    def test_the_whole_mode_can_be_switched_off_from_the_environment(self, precision_env):
        precision_env.setenv("PRECISION_RAG_ENABLED", "0")
        get_precision_config.cache_clear()
        assert get_precision_config().enabled is False

    def test_result_is_cached_until_the_cache_is_cleared(self, precision_env):
        # Documenting the trap rather than working around it: the cache is what makes a
        # per-question call free, and it is also why every test touching PRECISION_RAG_*
        # has to clear it on the way out or the next test inherits this environment.
        first = get_precision_config()
        precision_env.setenv("PRECISION_RAG_FINAL_K", "3")
        assert get_precision_config() is first
        assert get_precision_config().final_k == 10

        get_precision_config.cache_clear()
        assert get_precision_config().final_k == 3


class TestWithOverrides:
    def test_applies_known_keys(self):
        config = PrecisionConfig()
        tuned = config.with_overrides({"final_k": 3, "reranker_enabled": False})
        assert tuned.final_k == 3
        assert tuned.reranker_enabled is False

    def test_ignores_unknown_keys(self):
        # The benchmark sweeps variants from data; a typo'd knob name must narrow the sweep,
        # not raise `TypeError` halfway through a run that has already cost minutes.
        config = PrecisionConfig()
        tuned = config.with_overrides({"final_k": 4, "no_such_knob": 99})
        assert tuned.final_k == 4
        assert not hasattr(tuned, "no_such_knob")

    def test_only_unknown_keys_returns_self_unchanged(self):
        config = PrecisionConfig()
        assert config.with_overrides({"no_such_knob": 99}) is config

    def test_none_and_empty_return_self(self):
        # Identity, not equality: `rag.service` calls this on every high-precision question
        # with overrides that are usually empty, and a copy per question would be pure waste.
        config = PrecisionConfig()
        assert config.with_overrides(None) is config
        assert config.with_overrides({}) is config

    def test_the_original_is_never_mutated(self):
        # The base object is the process-wide `lru_cache`d instance. If an override leaked
        # into it, one benchmark variant would silently reconfigure every later question.
        config = PrecisionConfig()
        tuned = config.with_overrides({"final_k": 3, "mmr_lambda": 0.1})
        assert config.final_k == 10
        assert config.mmr_lambda == 0.7
        assert tuned is not config

    def test_unrelated_fields_are_carried_across(self):
        config = PrecisionConfig().with_overrides({"negative_terms": ("draft",)})
        tuned = config.with_overrides({"final_k": 2})
        assert tuned.negative_terms == ("draft",)
        assert tuned.final_k == 2

    def test_the_config_is_frozen(self):
        # `with_overrides` is only safe to hand out because the instance cannot be edited in
        # place — that is what makes sharing one cached object between questions correct.
        config = PrecisionConfig()
        with pytest.raises(dataclasses.FrozenInstanceError):
            config.final_k = 99
