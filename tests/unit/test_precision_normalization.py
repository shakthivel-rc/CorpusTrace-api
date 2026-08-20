"""Unit tests for `rag/precision/normalize.py` — stage 1 of the high-precision pipeline.

Normalization is the only stage whose mistakes are *invisible*. Everything downstream —
BM25, the reranker, MMR — reports scores someone can look at; a query that lost a term to a
curly apostrophe simply retrieves less, and the user reads that as "the document does not
cover this". So the contract worth pinning here is narrow and absolute:

* **Determinism.** Every function is pure. `benchmark.py` compares pipeline variants against
  each other, and that comparison means nothing if the same query normalizes two ways.
* **Nothing is dropped.** Normalization removes *notation* (NBSP, ligatures, a sentence-final
  period glued to the last word), never a word. `canonicalize_terms` returns only ADDITIONS
  and `correct_spelling` returns only a mapping the caller adds alongside the typed term, so
  neither can move a question away from its own answer.
* **The four spelling guards are load-bearing individually.** Each one exists to stop a
  different wrong correction, so each is exercised on its own with the other three
  deliberately satisfied — a test that trips two guards at once proves nothing about either.

Self-contained on purpose: no DB import, no fixtures, no cross-file helpers (tests/ is not a
package). Everything here is strings and dicts.
"""
import pytest

from rag.precision.normalize import (
    CANONICAL_FORMS,
    MIN_SPELLING_LENGTH,
    _edit_distance_within_one,
    canonicalize_terms,
    correct_spelling,
    normalize_query,
)

pytestmark = pytest.mark.unit


class TestNormalizeQueryUnicode:
    def test_full_width_characters_fold_to_ascii(self):
        # NFKC runs first because the index was built from ASCII. Without it a full-width
        # query matches nothing at all while looking identical on screen.
        assert normalize_query("Ｃｏｎｆｉｇ") == "config"
        assert normalize_query("ＡＢＣ １２３") == "abc 123"

    def test_ligatures_decompose(self):
        # "ﬁle" is one codepoint. `_tokenize` would keep it as a token that no document
        # containing the word "file" can ever match.
        assert normalize_query("ﬁle") == "file"
        assert normalize_query("ﬂow") == "flow"

    def test_non_breaking_space_becomes_an_ordinary_separator(self):
        # NBSP survives a naive `.split()` and glues two terms into one.
        # Written as an escape: an invisible NBSP in a source file is one editor-save
        # away from becoming an ordinary space, which would silently gut this test.
        assert normalize_query("a\u00a0b") == "a b"

    def test_casefold_not_lower(self):
        # casefold is the reason this is not `.lower()`: German ß only compares equal to
        # "ss" under casefolding, and the corpus spells it whichever way the author did.
        assert normalize_query("Straße") == "strasse"

    def test_accented_characters_are_preserved(self):
        # NFKC is a compatibility fold, not a transliteration. Stripping the diaeresis
        # would be deciding what the user meant, which is explicitly not this stage's job.
        assert normalize_query("naïve") == "naïve"


class TestNormalizeQueryPunctuation:
    def test_curly_apostrophe_is_removed_like_the_ascii_one(self):
        # A typed ’ and a typed ' must reach the tokenizer as the same string, or the same
        # question asked on two keyboards retrieves two different sets of chunks.
        assert normalize_query("don’t") == "dont"
        assert normalize_query("don't") == "dont"

    def test_possessive_s_is_stripped_to_the_bare_noun(self):
        # "the engine's capacity" has to reach "engine", not "engines" and not a bare "s":
        # the possessive is removed BEFORE the punctuation map for exactly that reason.
        assert normalize_query("the engine's capacity") == "the engine capacity"
        assert normalize_query("the engine’s capacity") == "the engine capacity"

    def test_possessive_stripping_is_case_insensitive(self):
        # The rule runs BEFORE casefold, so a case-sensitive `s` made it fire on
        # "engine's" and not on "ENGINE'S" — a shouted query then tokenized to the
        # plural-looking "engines" and matched different chunks from the same words typed
        # normally. `re.IGNORECASE` is the whole fix, and this is why it is there.
        assert normalize_query("ENGINE'S CAPACITY") == "engine capacity"
        assert normalize_query("Engine'S Capacity") == "engine capacity"

    def test_en_and_em_dashes_become_spaces(self):
        # U+2010..U+2015 plus the maths minus are token boundaries the tokenizer cannot
        # see through — it keeps "-" inside a word but not these.
        assert normalize_query("co–operate") == "co operate"
        assert normalize_query("em—dash") == "em dash"
        assert normalize_query("minus−sign") == "minus sign"
        assert normalize_query("non‑breaking") == "non breaking"
        assert normalize_query("A—B–C‒D―E−F") == "a b c d e f"

    def test_ascii_hyphen_is_deliberately_left_alone(self):
        # DELIBERATE, and the opposite of what it looks like. `_tokenize` in rag/service.py
        # matches [A-Za-z0-9][A-Za-z0-9_-]{1,}, so the corpus was indexed with the hyphen
        # INSIDE the token: "real-time system" is ["real-time", "system"]. Splitting it here
        # would turn the query into ["real", "time", "system"] and leave it sharing one term
        # with the passage it is looking for — the query and the corpus disagreeing about
        # what a word is, which is the failure this module exists to prevent.
        #
        # The two spellings genuinely do need bridging, and that happens where a bridge can
        # be additive instead of destructive: `expansion.hyphen_variants` offers the parts
        # and the joined form of a hyphenated term, and `expand_query` offers the hyphenated
        # form of an adjacent pair whenever the CORPUS VOCABULARY actually contains it. Both
        # spellings stay reachable and neither is chosen on the user's behalf.
        assert normalize_query("state-of-the-art") == "state-of-the-art"
        assert normalize_query("state of the art") == "state of the art"

        # The Unicode dashes are split, and that is the same rule rather than an exception:
        # the tokenizer does not accept them either, so it already treats them as boundaries.
        assert normalize_query("state\u2010of\u2010the\u2010art") == "state of the art"

    def test_structural_punctuation_becomes_a_separator(self):
        assert normalize_query("read/write") == "read write"
        assert normalize_query("a|b") == "a b"
        assert normalize_query("x^2") == "x 2"
        assert normalize_query("**bold**") == "bold"
        assert normalize_query("What is the DB spec?") == "what is the db spec"

    def test_whitespace_is_collapsed_and_trimmed(self):
        assert normalize_query("  Hello   World  ") == "hello world"
        assert normalize_query("a\t\nb") == "a b"


class TestNormalizeQueryPeriods:
    def test_trailing_period_after_a_letter_is_dropped(self):
        # A sentence-final period glued to the last term is the single most common way a
        # real question loses its most specific word.
        assert normalize_query("config.") == "config"
        assert normalize_query("the valve's seal.") == "the valve seal"

    def test_decimals_survive(self):
        # The lookbehind is [A-Za-z] precisely so a number keeps its point. "4.2" and "42"
        # are different measurements and the corpus spells one of them.
        assert normalize_query("4.2") == "4.2"
        assert normalize_query("Is it 4.2 or v1.0.3?") == "is it 4.2 or v1.0.3"

    def test_version_strings_survive_in_full(self):
        assert normalize_query("v1.0.3") == "v1.0.3"
        assert normalize_query("TLS1.3") == "tls1.3"

    def test_only_the_period_at_the_end_of_a_word_is_stripped(self):
        # "u.s.a." loses the final period (letter + end) and keeps the interior ones
        # (letter + letter), which is what makes the rule safe for "e.g." and "v1.0.3".
        assert normalize_query("U.S.A.") == "u.s.a"
        assert normalize_query("a.b") == "a.b"

    def test_a_period_after_a_digit_is_never_stripped(self):
        # Falls out of the same lookbehind: the rule cannot tell a sentence end from a
        # decimal point after a number, so it leaves it. Pinned because the alternative
        # (stripping it) would silently truncate "3.14".
        assert normalize_query("3.") == "3."


class TestNormalizeQueryDegenerateInput:
    def test_empty_string_returns_empty(self):
        assert normalize_query("") == ""

    def test_whitespace_only_returns_empty(self):
        assert normalize_query("   ") == ""
        assert normalize_query("\t\n ") == ""

    def test_punctuation_only_returns_empty(self):
        # The pipeline's "query has no searchable terms" branch depends on this: an empty
        # normalized query must produce zero terms, not a token made of brackets.
        assert normalize_query("?!,;:()") == ""
        assert normalize_query("—") == ""

    def test_a_bare_period_is_not_punctuation_here(self):
        # Consequence of protecting decimals: "." is not in `_PUNCT_TO_SPACE`, so a
        # dots-only query survives normalization and is discarded by the tokenizer instead.
        assert normalize_query("...") == "..."

    def test_is_idempotent(self):
        # Stage 1 output feeds the trace and the benchmark; running it twice must not
        # produce a third string.
        for query in ["The Engine's Max Temp.", "Ｃｏｎｆｉｇ 4.2", "co–operate  now"]:
            once = normalize_query(query)
            assert normalize_query(once) == once


class TestCanonicalizeTerms:
    def test_returns_only_additions_never_the_input(self):
        # The caller keeps the typed terms at FULL weight and joins these at the expansion
        # weight. Returning the originals here would double-count them.
        assert canonicalize_terms(["db", "config"]) == ["database", "configuration"]

    def test_preserves_first_seen_order(self):
        assert canonicalize_terms(["spec", "db"]) == ["specification", "database"]
        assert canonicalize_terms(["db", "spec"]) == ["database", "specification"]

    def test_multi_word_canonical_form_is_split_into_words(self):
        # "ie" -> "that is". The consumer weights individual TERMS, so a two-word string
        # would arrive as one unmatchable token.
        assert canonicalize_terms(["ie"]) == ["that", "is"]

    def test_never_repeats_an_addition(self):
        # Two abbreviations of the same word ("auth", "authn") must contribute one term,
        # or the expansion budget is spent twice on the same signal.
        assert canonicalize_terms(["auth", "authn"]) == ["authentication"]

    def test_does_not_add_a_term_the_user_already_typed(self):
        # `seen` is seeded from the whole input, so spelling it out and abbreviating it in
        # the same question adds nothing.
        assert canonicalize_terms(["db", "database"]) == []

    def test_repeated_input_term_yields_one_addition(self):
        assert canonicalize_terms(["db", "db"]) == ["database"]

    def test_unknown_terms_yield_nothing(self):
        # The map is deliberately small and domain-neutral; an unrecognised word is left
        # entirely alone rather than guessed at.
        assert canonicalize_terms(["widget", "gizmo"]) == []

    def test_empty_input_yields_nothing(self):
        assert canonicalize_terms([]) == []

    def test_matching_is_exact_and_case_sensitive(self):
        # Terms reach this function from `_tokenize`, which lowercases. An upper-case key
        # would be a lookup that can never fire, so the absence is the correct behaviour.
        assert canonicalize_terms(["DB"]) == []

    def test_every_canonical_form_is_lowercase_and_non_empty(self):
        # These are matched against tokenizer output, which is lowercase. A capitalised or
        # blank value in the table would be a silently dead entry.
        for written, canonical in CANONICAL_FORMS.items():
            assert written == written.lower(), written
            assert canonical == canonical.lower(), written
            assert canonical.split(), written


class TestEditDistanceWithinOne:
    def test_identical_strings(self):
        assert _edit_distance_within_one("valve", "valve") is True
        assert _edit_distance_within_one("", "") is True

    def test_single_substitution(self):
        assert _edit_distance_within_one("valve", "value") is True

    def test_single_insertion(self):
        # Both argument orders, because the function has to pick the longer side itself.
        assert _edit_distance_within_one("vale", "valve") is True
        assert _edit_distance_within_one("bcd", "abcd") is True
        assert _edit_distance_within_one("ab", "abx") is True

    def test_single_deletion(self):
        assert _edit_distance_within_one("valve", "vale") is True
        assert _edit_distance_within_one("abcd", "abc") is True

    def test_two_substitutions_rejected(self):
        # "cache"/"cause" differ in two places. This is the whole point of the bound: at
        # distance 2 the honest answer is "do not guess".
        assert _edit_distance_within_one("cache", "cause") is False

    def test_transposition_is_distance_two_here(self):
        # Damerau would call this distance 1; this function deliberately does not, because
        # a transposition of a short term is usually a different word.
        assert _edit_distance_within_one("ab", "ba") is False

    def test_substitution_plus_insertion_rejected(self):
        # Exercises the walk's second-mismatch exit rather than the length prefilter.
        assert _edit_distance_within_one("abcx", "abd") is False

    def test_length_gap_greater_than_one_rejected(self):
        assert _edit_distance_within_one("port", "portion") is False
        assert _edit_distance_within_one("abc", "abcde") is False
        assert _edit_distance_within_one("", "ab") is False

    def test_empty_against_one_character(self):
        assert _edit_distance_within_one("", "a") is True


class TestCorrectSpelling:
    def test_min_spelling_length_is_what_the_fixtures_assume(self):
        # The word lengths chosen below only isolate the guards while this is 5.
        assert MIN_SPELLING_LENGTH == 5

    def test_happy_path_corrects_a_typo(self):
        # One absent term, one unique distance-1 neighbour, seen twice in the corpus.
        assert correct_spelling(["presure"], {"pressure": 9}) == {"presure": "pressure"}

    def test_returns_a_mapping_only_for_the_terms_it_changed(self):
        vocabulary = {"pressure": 9, "temperature": 4}
        assert correct_spelling(["presure", "temperature"], vocabulary) == {
            "presure": "pressure"
        }

    def test_guard_one_a_term_in_the_vocabulary_is_never_corrected(self):
        # "value" is a real word the documents use; "valve" is one edit away. Correcting
        # here would move the question off the passage that answers it.
        assert correct_spelling(["value"], {"value": 6, "valve": 6}) == {}
        # ...and the ONLY reason is the guard: drop "value" from the corpus and the same
        # term corrects immediately.
        assert correct_spelling(["value"], {"valve": 6}) == {"value": "valve"}

    def test_guard_two_short_terms_are_never_corrected(self):
        # At four characters and below a distance-1 neighbour is usually a different word.
        assert correct_spelling(["vale"], {"valve": 6}) == {}
        # Distance is not what rejected it — the length is.
        assert _edit_distance_within_one("vale", "valve") is True
        # One character longer, everything else identical, and it corrects.
        assert correct_spelling(["valse"], {"valve": 6}) == {"valse": "valve"}

    def test_guard_three_two_candidates_yields_nothing(self):
        # "cater" is one edit from both "water" and "later". Picking the more frequent one
        # would be a coin flip presented to the user as a correction.
        assert correct_spelling(["cater"], {"water": 4, "later": 4}) == {}
        # Each candidate on its own is a perfectly good correction, so ambiguity is the
        # only thing the pair test is measuring.
        assert correct_spelling(["cater"], {"water": 4}) == {"cater": "water"}
        assert correct_spelling(["cater"], {"later": 4}) == {"cater": "later"}

    def test_guard_four_a_candidate_seen_once_is_not_offered(self):
        # A typo INSIDE a document must not recruit the query into matching it.
        assert correct_spelling(["presure"], {"pressure": 1}) == {}
        # The same candidate at count 2 clears the bar, so frequency is the only variable.
        assert correct_spelling(["presure"], {"pressure": 2}) == {"presure": "pressure"}

    def test_empty_vocabulary_returns_nothing(self):
        # A resource with no indexed chunks must not raise on the way to the refusal.
        assert correct_spelling(["presure"], {}) == {}

    def test_empty_term_list_returns_nothing(self):
        assert correct_spelling([], {"pressure": 9}) == {}

    def test_correction_never_replaces_the_typed_term(self):
        # The contract the docstring rests on: this returns a mapping, and the caller ADDS
        # the value while keeping the key, so `_unmatched_terms` still reports the typed
        # word and the vocabulary hint keeps working (CLAUDE.md §17).
        corrections = correct_spelling(["presure"], {"pressure": 9})
        assert list(corrections) == ["presure"]
        assert corrections["presure"] == "pressure"

    def test_terms_are_independent_of_each_other(self):
        # Each term is judged against the corpus alone; one term's ambiguity must not
        # suppress another term's correction.
        vocabulary = {"pressure": 9, "water": 4, "later": 4}
        assert correct_spelling(["cater", "presure"], vocabulary) == {
            "presure": "pressure"
        }
