"""Unit tests for deterministic query expansion — stage 2 of the high-precision mode.

Expansion is the one stage of that pipeline that puts words into the query the user never
typed, so it is also the one stage that can move a question away from its own answer. The
module's whole safety argument is a single arithmetic invariant — **a typed term is 1.0 and
everything added is below it** — plus a hard budget on how many guesses may join. Neither
property fails loudly: an expansion weighted too highly, or a budget that leaks, produces a
plausible-looking answer sourced from a passage that matched a synonym nobody asked for.

So the tests here pin the numbers and the ordering, not just the shape:

* typed terms survive every expansion path at 1.0, whichever path rediscovers them;
* `expansion_max_terms` bounds the additions absolutely, and the sources are consulted in
  confidence order (corrections, canonical forms, operator dictionary, built-in synonyms,
  entity aliases, morphology) so the budget is spent on the best guesses first;
* `_morphological_variants` only fires where its inverse is unambiguous — the
  `MIN_MORPH_STEM` floor is what stops "ring" becoming "r";
* `load_dictionary` never raises. It reads an operator's tuning file, and a syntax error in
  a tuning file must not stop the deployment answering questions;
* PRF mutates the query **in place** and is bounded — `pipeline.py` discards the return
  value and reads `expanded.terms` afterwards, so in-place is load-bearing, not incidental.

Pure unit tests: no database, no network, and configuration is built with
`dataclasses.replace()` so nothing here depends on the process environment.
"""
import json
from dataclasses import replace

import pytest

from rag.precision.config import PrecisionConfig
from rag.precision.expansion import (
    BUILTIN_SYNONYMS,
    MIN_MORPH_STEM,
    ExpandedQuery,
    _morphological_variants,
    apply_pseudo_relevance_feedback,
    expand_query,
    load_dictionary,
)

pytestmark = pytest.mark.unit


def _config(**overrides) -> PrecisionConfig:
    """A config differing from the documented defaults only where a test says so.

    `replace()` rather than environment variables: `get_precision_config` is `lru_cache`d,
    so a test that set an env var would leak into every later test in the process.
    """
    return replace(PrecisionConfig(), **overrides)


def _expand(terms, *, config=None, **kwargs) -> ExpandedQuery:
    """Call `expand_query` the way the pipeline does — already-tokenized terms in, weighted
    map out. The original/normalized strings are pass-through metadata here."""
    query = " ".join(terms)
    return expand_query(query, query, list(terms), config=config or _config(), **kwargs)


class TestTypedTermsAreNeverDemoted:
    def test_every_typed_term_carries_weight_one(self):
        expanded = _expand(["engine", "valve", "clearance"])
        for term in ["engine", "valve", "clearance"]:
            assert expanded.terms[term] == 1.0

    def test_a_synonym_path_cannot_lower_a_term_the_user_typed(self):
        # "fault" is a built-in synonym of "error" AND a word the user typed. If `offer`
        # overwrote instead of comparing, typing both words would score the query lower
        # than typing one of them.
        expanded = _expand(["error", "fault"])
        assert expanded.terms["fault"] == 1.0

    def test_a_correction_onto_a_typed_term_neither_demotes_nor_counts_as_added(self):
        # A correction proposes a term the user *might* have meant; if they already typed
        # it, nothing has been added and no budget should be spent.
        expanded = _expand(["error", "eror"], corrections={"eror": "error"})
        assert expanded.terms["error"] == 1.0
        assert "error" not in expanded.added_terms

    def test_no_added_term_ever_reaches_the_weight_of_a_typed_one(self):
        expanded = _expand(
            ["error", "install", "db"],
            dictionary={"error": ("regression",)},
            entity_aliases={"install": ("installer",)},
        )
        assert expanded.added_terms  # the assertion below is vacuous if nothing was added
        for term in expanded.added_terms:
            assert expanded.terms[term] < 1.0

    def test_repeating_a_word_does_not_duplicate_it(self):
        expanded = _expand(["valve", "valve"])
        assert expanded.original_terms == ["valve"]
        assert expanded.terms["valve"] == 1.0


class TestExpansionBudget:
    def test_added_terms_never_exceed_expansion_max_terms(self):
        # Three synonym-rich words against a budget of three: the budget, not the number
        # of available candidates, is what decides.
        expanded = _expand(["error", "problem", "install"], config=_config(expansion_max_terms=3))
        assert len(expanded.added_terms) == 3
        assert len(expanded.terms) == len(expanded.original_terms) + 3

    def test_the_budget_is_spent_in_source_order(self):
        # Built-in synonyms of "error" come first because "error" is the first term, so a
        # budget of one buys exactly the first candidate offered and nothing later.
        expanded = _expand(["error", "problem"], config=_config(expansion_max_terms=1))
        assert expanded.added_terms == ["failure"]

    def test_a_budget_of_zero_is_the_same_as_disabling_expansion(self):
        expanded = _expand(["error"], config=_config(expansion_max_terms=0))
        assert expanded.added_terms == []
        assert expanded.terms == {"error": 1.0}


class TestExpansionDisabled:
    def test_returns_only_the_typed_terms(self):
        expanded = _expand(["error", "timeout"], config=_config(query_expansion_enabled=False))
        assert expanded.terms == {"error": 1.0, "timeout": 1.0}
        assert expanded.added_terms == []

    def test_phrases_and_corrections_still_survive(self):
        # Phrases are the user's own word order and corrections are recorded evidence, not
        # expansions — switching expansion off must not take either away, because the
        # reranker and the vocabulary hint read them.
        expanded = _expand(
            ["valve", "clearance"],
            config=_config(query_expansion_enabled=False),
            corrections={"valv": "valve"},
        )
        assert expanded.phrases == [("valve", "clearance")]
        assert expanded.corrections == {"valv": "valve"}


class TestPhrases:
    def test_phrases_are_adjacent_typed_pairs_in_order(self):
        expanded = _expand(["cold", "engine", "valve"])
        assert expanded.phrases == [("cold", "engine"), ("engine", "valve")]

    def test_a_single_term_has_no_phrases(self):
        assert _expand(["valve"]).phrases == []

    def test_phrases_are_built_only_from_terms_the_user_typed(self):
        # An expansion is a guess about a word, never a guess about word order — a phrase
        # made of a synonym would let the reranker reward an ordering nobody wrote.
        expanded = _expand(["error", "timeout"])
        typed = {"error", "timeout"}
        assert expanded.added_terms  # the test is vacuous if nothing was added
        assert all(left in typed and right in typed for left, right in expanded.phrases)


class TestSourceOrderAndWeights:
    def test_a_correction_weighs_at_least_the_expansion_weight(self):
        expanded = _expand(["xyzzy"], corrections={"xyzzy": "clearance"})
        config = _config()
        assert expanded.terms["clearance"] >= config.expansion_term_weight
        assert expanded.terms["clearance"] == pytest.approx(0.8)

    def test_a_correction_outweighs_a_synonym(self):
        # Corrections are only proposed when exactly one corpus term could have been
        # meant, so they are the highest-confidence thing expansion adds.
        expanded = _expand(["error", "xyzzy"], corrections={"xyzzy": "clearance"})
        assert expanded.terms["clearance"] > expanded.terms["failure"]

    def test_a_high_expansion_weight_raises_the_correction_with_it(self):
        expanded = _expand(
            ["xyzzy"],
            config=_config(expansion_term_weight=0.9),
            corrections={"xyzzy": "clearance"},
        )
        assert expanded.terms["clearance"] == pytest.approx(0.9)

    def test_canonical_forms_of_abbreviations_join_at_the_expansion_weight(self):
        expanded = _expand(["db", "config"])
        assert expanded.terms["database"] == pytest.approx(0.45)
        assert expanded.terms["configuration"] == pytest.approx(0.45)
        # The abbreviation itself is never removed — documents that use it stay reachable.
        assert expanded.terms["db"] == 1.0

    def test_a_correction_is_itself_canonicalized(self):
        # Canonicalization runs over the whole weighted map, not just the typed terms, so
        # correcting "cnfig" to "config" also reaches "configuration".
        expanded = _expand(["cnfig"], corrections={"cnfig": "config"})
        assert expanded.terms["config"] == pytest.approx(0.8)
        assert expanded.terms["configuration"] == pytest.approx(0.45)

    def test_operator_dictionary_is_offered_before_builtin_synonyms(self):
        # A deployment that wrote a dictionary knows more about its documents than this
        # engine does, so a scarce budget must buy the operator's term, not ours.
        expanded = _expand(
            ["error"],
            config=_config(expansion_max_terms=1),
            dictionary={"error": ("regression",)},
        )
        assert expanded.added_terms == ["regression"]
        assert "failure" not in expanded.terms

    def test_operator_dictionary_also_outweighs_builtin_synonyms(self):
        expanded = _expand(["error"], dictionary={"error": ("regression",)})
        assert expanded.terms["regression"] == pytest.approx(0.45)
        assert expanded.terms["failure"] == pytest.approx(0.45 * 0.9)
        assert expanded.terms["regression"] > expanded.terms["failure"]

    def test_entity_aliases_from_the_corpus_are_used(self):
        expanded = _expand(["turbo"], entity_aliases={"turbo": ("turbocharger",)})
        assert expanded.terms["turbocharger"] == pytest.approx(0.45)
        assert "turbocharger" in expanded.added_terms

    def test_morphological_variants_are_the_lowest_weighted_addition(self):
        # Morphology is the cheapest signal and the likeliest to be noise, so it must lose
        # every tie against a source that consulted actual vocabulary.
        expanded = _expand(["error"], entity_aliases={"error": ("misfire",)})
        assert expanded.terms["errors"] == pytest.approx(0.45 * 0.7)
        others = [expanded.terms[term] for term in expanded.added_terms if term != "errors"]
        assert others and all(weight > expanded.terms["errors"] for weight in others)

    def test_a_term_reached_by_two_paths_keeps_the_higher_weight(self):
        # "failure" arrives first as a built-in synonym (0.9x) and again as an entity alias
        # (1.0x). The second offer must raise it and must not buy a second slot.
        expanded = _expand(["error"], entity_aliases={"error": ("failure",)})
        assert expanded.terms["failure"] == pytest.approx(0.45)
        assert expanded.added_terms.count("failure") == 1

    def test_empty_and_single_character_candidates_are_refused(self):
        # A one-character term matches almost every passage; the index's own tokenizer
        # drops them, so admitting one here would be a term with no possible match that
        # still costs budget.
        expanded = _expand(["error"], entity_aliases={"error": ("", "x", "xy")})
        assert "" not in expanded.terms
        assert "x" not in expanded.terms
        assert expanded.terms["xy"] == pytest.approx(0.45)

    def test_all_terms_lists_typed_terms_before_added_ones(self):
        expanded = _expand(["error", "timeout"])
        assert expanded.all_terms[:2] == ["error", "timeout"]
        assert set(expanded.all_terms) == set(expanded.terms)


class TestMorphologicalVariants:
    def test_ies_becomes_y(self):
        assert _morphological_variants("policies") == ["policy"]

    def test_ing_is_dropped(self):
        assert _morphological_variants("starting") == ["start"]

    def test_ed_is_dropped(self):
        assert _morphological_variants("started") == ["start"]

    def test_trailing_s_is_dropped(self):
        assert _morphological_variants("errors") == ["error"]

    def test_only_the_first_matching_rule_applies(self):
        # "policies" ends in both "ies" and "s". Applying both would offer "policie",
        # which is not a word in any document, at the cost of a budget slot.
        assert "policie" not in _morphological_variants("policies")

    def test_a_stem_shorter_than_the_floor_yields_nothing(self):
        # "ring" would become "r" under the -ing rule: a one-character term that matches
        # nothing. MIN_MORPH_STEM stops it — and the term then yields NOTHING rather than
        # falling through to the plural rule. Falling through is what produced "catss" from
        # "cats"; a term that already carries an inflection must not be inflected again.
        assert _morphological_variants("ring") == []
        assert len("ring") - len("ing") < MIN_MORPH_STEM

    def test_a_stem_exactly_at_the_floor_is_allowed(self):
        # "runn" is four characters — the floor is inclusive, so the rule fires. The
        # undoubled "run" comes with it: English doubles a final consonant before -ing, so
        # the naive stem is a word no corpus contains and the real verb is one character
        # shorter. Both are offered; neither is chosen.
        assert _morphological_variants("running") == ["runn", "run"]

    def test_a_term_with_no_recognised_suffix_gets_its_plural(self):
        assert _morphological_variants("valve") == ["valves"]

    def test_a_term_shorter_than_the_floor_gets_nothing(self):
        # Three characters is too little to guess from in either direction.
        assert _morphological_variants("cat") == []

    def test_a_short_plural_is_not_pluralised_again(self):
        # "cats" ends in "s" but its stem ("cat") is below the floor, so the suffix rule
        # declines. It must NOT then fall through to the plural fallback: that produced
        # "catss", a term no document contains, at the cost of a slot in a deliberately small
        # budget. The for/else that allowed it was the defect.
        assert _morphological_variants("cats") == []


class TestLoadDictionary:
    def test_a_missing_file_is_not_an_error(self, tmp_path):
        # A deployment that has not written a dictionary yet must still be able to use the
        # mode — the built-in list is a working default, not a fallback for a broken state.
        missing = tmp_path / "never-written.json"
        assert not missing.exists()
        assert load_dictionary(str(missing)) == {}

    def test_no_path_and_no_environment_variable_yields_an_empty_dictionary(self, monkeypatch):
        monkeypatch.delenv("PRECISION_RAG_DICTIONARY_PATH", raising=False)
        assert load_dictionary() == {}

    def test_the_environment_variable_is_the_fallback_path(self, tmp_path, monkeypatch):
        path = tmp_path / "dictionary.json"
        path.write_text(json.dumps({"valve": ["gate"]}), encoding="utf-8")
        monkeypatch.setenv("PRECISION_RAG_DICTIONARY_PATH", str(path))
        assert load_dictionary() == {"valve": ("gate",)}

    def test_malformed_json_is_ignored_rather_than_raised(self, tmp_path):
        # A syntax error in a tuning file is not a reason to stop answering questions.
        path = tmp_path / "broken.json"
        path.write_text("{\"valve\": [", encoding="utf-8")
        assert load_dictionary(str(path)) == {}

    def test_an_unreadable_path_is_ignored_rather_than_raised(self, tmp_path):
        # A directory where a file was expected is the commonest form of "unreadable", and
        # it raises OSError rather than a JSON error — both paths must be caught.
        assert load_dictionary(str(tmp_path)) == {}

    def test_a_non_dict_top_level_yields_an_empty_dictionary(self, tmp_path):
        path = tmp_path / "list.json"
        path.write_text(json.dumps(["valve", "gate"]), encoding="utf-8")
        assert load_dictionary(str(path)) == {}

    def test_keys_and_values_are_lowercased_and_stripped(self, tmp_path):
        # The index is lowercase, so an operator's capitalisation would produce entries no
        # query term could ever match — silently, and only for the words they capitalised.
        path = tmp_path / "case.json"
        path.write_text(json.dumps({" Valve ": ["Clearance", "  GAP  "]}), encoding="utf-8")
        assert load_dictionary(str(path)) == {"valve": ("clearance", "gap")}

    def test_non_list_values_are_skipped(self, tmp_path):
        path = tmp_path / "mixed.json"
        path.write_text(
            json.dumps({"valve": "gate", "torque": ["moment"], "gasket": 7}), encoding="utf-8"
        )
        assert load_dictionary(str(path)) == {"torque": ("moment",)}

    def test_an_entry_with_no_usable_aliases_is_dropped(self, tmp_path):
        path = tmp_path / "empty.json"
        path.write_text(json.dumps({"valve": [], "torque": ["   ", ""]}), encoding="utf-8")
        assert load_dictionary(str(path)) == {}

    def test_non_string_aliases_are_coerced(self, tmp_path):
        path = tmp_path / "numbers.json"
        path.write_text(json.dumps({"version": [12, "twelve"]}), encoding="utf-8")
        assert load_dictionary(str(path)) == {"version": ("12", "twelve")}

    def test_a_loaded_dictionary_feeds_straight_into_expansion(self, tmp_path):
        # The loader's output shape is only interesting because `expand_query` consumes it
        # directly; pinning the round trip stops the two drifting apart.
        path = tmp_path / "domain.json"
        path.write_text(json.dumps({"valve": ["Poppet"]}), encoding="utf-8")
        expanded = _expand(["valve"], dictionary=load_dictionary(str(path)))
        assert expanded.terms["poppet"] == pytest.approx(0.45)


class TestPseudoRelevanceFeedback:
    def test_feedback_terms_join_at_the_prf_weight(self):
        expanded = _expand(["valve"], config=_config(query_expansion_enabled=False))
        config = _config(query_expansion_enabled=False)
        apply_pseudo_relevance_feedback(expanded, [("gasket", 4.2)], config=config)
        assert expanded.terms["gasket"] == pytest.approx(config.prf_term_weight)
        assert expanded.added_terms == ["gasket"]

    def test_it_is_bounded_by_prf_terms(self):
        # A query that has absorbed the whole top passage is no longer the question that
        # was asked, which is the failure PRF is famous for.
        expanded = _expand(["valve"], config=_config(query_expansion_enabled=False))
        feedback = [("gasket", 4.0), ("torque", 3.0), ("bracket", 2.0), ("coolant", 1.0)]
        apply_pseudo_relevance_feedback(expanded, feedback, config=_config(prf_terms=2))
        assert expanded.added_terms == ["gasket", "torque"]
        assert "bracket" not in expanded.terms

    def test_terms_already_in_the_query_are_skipped_without_spending_budget(self):
        # The top passage's most distinctive term is very often a word the user typed.
        # Counting it would silently halve the feedback the user actually gets.
        expanded = _expand(["valve"], config=_config(query_expansion_enabled=False))
        feedback = [("valve", 9.0), ("gasket", 4.0), ("torque", 3.0)]
        apply_pseudo_relevance_feedback(expanded, feedback, config=_config(prf_terms=2))
        assert expanded.terms["valve"] == 1.0
        assert expanded.added_terms == ["gasket", "torque"]

    def test_a_term_an_earlier_expansion_added_is_also_skipped(self):
        expanded = _expand(["error"])
        assert expanded.terms["failure"] == pytest.approx(0.45 * 0.9)
        apply_pseudo_relevance_feedback(expanded, [("failure", 9.0)], config=_config())
        # PRF's weight (0.25) is lower than the synonym weight it already has; re-adding it
        # would demote a term on the strength of a weaker signal.
        assert expanded.terms["failure"] == pytest.approx(0.45 * 0.9)
        assert expanded.added_terms.count("failure") == 1

    def test_disabled_feedback_is_a_no_op(self):
        expanded = _expand(["valve"], config=_config(query_expansion_enabled=False))
        before = dict(expanded.terms)
        apply_pseudo_relevance_feedback(
            expanded, [("gasket", 4.0)], config=_config(prf_enabled=False)
        )
        assert expanded.terms == before
        assert expanded.added_terms == []

    def test_a_prf_budget_of_zero_is_a_no_op(self):
        expanded = _expand(["valve"], config=_config(query_expansion_enabled=False))
        before = dict(expanded.terms)
        apply_pseudo_relevance_feedback(expanded, [("gasket", 4.0)], config=_config(prf_terms=0))
        assert expanded.terms == before

    def test_it_mutates_the_query_in_place_and_returns_it(self):
        # `pipeline.py` discards the return value and re-reads `expanded.terms` for its
        # second BM25 pass, so returning a copy would make PRF do nothing at all there.
        expanded = _expand(["valve"], config=_config(query_expansion_enabled=False))
        returned = apply_pseudo_relevance_feedback(expanded, [("gasket", 4.0)], config=_config())
        assert returned is expanded
        assert "gasket" in expanded.terms


class TestBuiltinSynonyms:
    def test_every_alias_is_a_lowercase_multi_character_term(self):
        # `offer` refuses anything under two characters and the index is lowercase, so an
        # entry failing either test would be a shipped synonym that can never match.
        for term, aliases in BUILTIN_SYNONYMS.items():
            assert term == term.lower() and len(term) >= 2
            for alias in aliases:
                assert alias == alias.lower() and len(alias) >= 2

    def test_no_entry_lists_itself(self):
        # A self-alias would only re-offer a term already at 1.0 — harmless, but it means
        # the entry was written without checking, which is how a wrong one gets in.
        for term, aliases in BUILTIN_SYNONYMS.items():
            assert term not in aliases
