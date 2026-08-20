"""Deterministic query expansion — stage 2 of the high-precision pipeline.

Five sources, all deterministic and all configurable, and none of them a language model:

  * abbreviation mappings          (`normalize.CANONICAL_FORMS`)
  * synonym / domain terminology   (`BUILTIN_SYNONYMS` + an operator-supplied dictionary)
  * morphological variants         (conservative, suffix-level: plural / -ing / -ed)
  * hyphenation variants           (both spellings of a hyphenated term, offered not chosen)
  * entity aliases                 (supplied by the caller from the corpus's own entities)
  * pseudo-relevance feedback      (the distinctive terms of the top first-pass hits)

THE INVARIANT THAT MAKES THIS SAFE: expansion is additive and weighted. A term the user
typed always carries weight 1.0; everything added carries less. So an expansion can only
raise a passage the user's own words already reached, or surface one they nearly reached —
it can never outrank a passage that matched what they actually asked. That is the same
property `_blend_semantic` relies on in `rag/service.py`, applied one stage earlier.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
import os

from rag.precision.normalize import CANONICAL_FORMS, canonicalize_terms


logger = logging.getLogger("corpustrace.rag.precision")


# General-purpose, domain-neutral equivalences. The list is short on purpose: every entry
# is a claim that two words mean the same thing in every document this software will ever
# index, and very few words survive that test. Domain vocabulary belongs in a deployment's
# own dictionary (PRECISION_RAG_DICTIONARY_PATH), where it can be reviewed by someone who
# knows the domain.
BUILTIN_SYNONYMS: dict[str, tuple[str, ...]] = {
    "error": ("failure", "fault", "exception"),
    "failure": ("error", "fault"),
    "timeout": ("timed", "expiry", "expired", "deadline"),
    "delete": ("remove", "erase", "purge"),
    "remove": ("delete", "erase"),
    "create": ("add", "new", "insert"),
    "update": ("modify", "change", "edit"),
    "start": ("begin", "launch", "boot"),
    "stop": ("halt", "terminate", "shutdown"),
    "fix": ("repair", "resolve", "correct"),
    "problem": ("issue", "fault", "defect"),
    "issue": ("problem", "defect"),
    "limit": ("maximum", "cap", "threshold"),
    "cost": ("price", "fee", "charge"),
    "speed": ("velocity", "rate"),
    "size": ("dimension", "capacity"),
    "requirement": ("required", "must", "mandatory"),
    "install": ("installation", "setup", "deploy"),
    "connect": ("connection", "connectivity"),
    "permission": ("access", "privilege", "authorization"),
    "user": ("account", "member"),
    "document": ("file", "record"),
}

# Suffix rules used to reach a word's other surface forms. Only rules whose inverse is
# unambiguous appear here — "-s" is included, "-es" is not, because "buses" -> "bus" and
# "cases" -> "cas" are the same rule with different answers.
_SUFFIX_RULES: tuple[tuple[str, str], ...] = (
    ("ies", "y"),
    ("ing", ""),
    ("ed", ""),
    ("s", ""),
)

MIN_MORPH_STEM = 4


@dataclass
class ExpandedQuery:
    """The query as the retrievers see it: weighted terms plus ordered phrases."""

    original: str
    normalized: str
    # term -> weight. Typed terms are 1.0; everything else is below it.
    terms: dict[str, float] = field(default_factory=dict)
    original_terms: list[str] = field(default_factory=list)
    added_terms: list[str] = field(default_factory=list)
    # Adjacent pairs of the user's own terms, in the order they typed them. Used by the
    # reranker to reward a passage that preserves the order, which no bag of words can see.
    phrases: list[tuple[str, str]] = field(default_factory=list)
    corrections: dict[str, str] = field(default_factory=dict)
    # Terms that came from pseudo-relevance feedback rather than from the user or a
    # dictionary. Tracked separately because the reranker must NOT read them — see
    # `rerank.lexical_cross_encode`. They are the least question-like terms in the map: the
    # corpus's opinion of what the top hits were about, not anything anyone asked.
    feedback_terms: set[str] = field(default_factory=set)

    @property
    def all_terms(self) -> list[str]:
        return list(self.terms)

    @property
    def rerank_terms(self) -> dict[str, float]:
        """The weighted terms the cross-encoder is allowed to score against."""
        if not self.feedback_terms:
            return self.terms
        return {term: weight for term, weight in self.terms.items() if term not in self.feedback_terms}


def load_dictionary(path: str | None = None) -> dict[str, tuple[str, ...]]:
    """An operator-supplied synonym / domain-terminology / entity-alias dictionary.

    A JSON object of `{"term": ["alias", ...]}`. Read from `PRECISION_RAG_DICTIONARY_PATH`
    when no path is given. A missing file is not an error — the built-in list is a working
    default and a deployment that has not written a dictionary yet must still be able to
    use the mode. A malformed file is logged and ignored for the same reason: a syntax
    error in a tuning file is not a reason to stop answering questions.
    """
    resolved = path or os.getenv("PRECISION_RAG_DICTIONARY_PATH", "").strip()
    if not resolved:
        return {}
    try:
        with open(resolved, "r", encoding="utf-8") as handle:
            raw = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning(
            "precision rag dictionary unreadable, continuing with built-ins",
            extra={"extra_fields": {"path": resolved, "error": type(exc).__name__}},
        )
        return {}
    if not isinstance(raw, dict):
        return {}
    dictionary: dict[str, tuple[str, ...]] = {}
    for key, values in raw.items():
        if not isinstance(key, str) or not isinstance(values, (list, tuple)):
            continue
        aliases = tuple(str(value).strip().lower() for value in values if str(value).strip())
        if aliases:
            dictionary[key.strip().lower()] = aliases
    return dictionary


def _morphological_variants(term: str) -> list[str]:
    """Other surface forms of one term, by unambiguous suffix rules only."""
    variants: list[str] = []
    if "-" in term:
        # A compound is handled by `hyphen_variants`, which knows how to take it apart.
        # Inflecting it whole yields "real-times" — a token no corpus contains, spending a
        # slot of a budget the module treats as scarce.
        return variants
    for suffix, replacement in _SUFFIX_RULES:
        if not term.endswith(suffix):
            continue
        # The term already carries an inflection. Whether or not the stem is long enough to
        # offer, STOP HERE — falling through to the plural rule below would pluralise a
        # plural ("cats" -> "catss", "runs" -> "runss"), which is a term no corpus contains
        # and a wasted slot in a budget that is deliberately small.
        if len(term) - len(suffix) >= MIN_MORPH_STEM:
            stem = term[: len(term) - len(suffix)] + replacement
            if stem and stem != term:
                variants.append(stem)
                # English doubles a final consonant before -ing/-ed ("running" -> "runn"),
                # so the naive stem is a word no corpus contains. Offering the undoubled
                # form as well costs one slot and recovers the actual verb.
                if (
                    len(stem) > 2
                    and stem[-1] == stem[-2]
                    and stem[-1] not in "aeiou"
                ):
                    variants.append(stem[:-1])
        return variants
    # No suffix matched, so the term is probably a stem: offer the plural.
    if len(term) >= MIN_MORPH_STEM:
        variants.append(term + "s")
    return variants


def hyphen_variants(term: str, tokenize=None) -> list[str]:
    """Both other spellings of a hyphenated term: joined, and split into its parts.

    `normalize_query` deliberately leaves the ASCII hyphen alone so the query agrees with the
    tokenizer the corpus was indexed by. That leaves a real gap — a document writing
    "state of the art" is unreachable from a query typing "state-of-the-art" and vice versa —
    and this is where it is closed, ADDITIVELY: both spellings are offered as expansions at
    the expansion weight, so neither is chosen on the user's behalf.
    """
    if "-" not in term:
        return []
    parts = term.split("-")
    if tokenize is not None:
        # Run the parts through the CORPUS tokenizer, which drops stopwords. The hyphenated
        # term arrived as a single token, so its interior words were never filtered — without
        # this, "state-of-the-art" offers "of" and "the" as expansions and spends two of a
        # twelve-term budget on words no chunk's term map even contains.
        parts = [word for part in parts for word in tokenize(part)]
    else:
        parts = [part for part in parts if len(part) > 2]
    joined = term.replace("-", "")
    variants = [joined] if len(joined) > 1 else []
    variants.extend(parts)
    return [variant for variant in dict.fromkeys(variants) if variant != term and len(variant) > 1]


def expand_query(
    original: str,
    normalized: str,
    terms: list[str],
    *,
    config,
    dictionary: dict[str, tuple[str, ...]] | None = None,
    entity_aliases: dict[str, tuple[str, ...]] | None = None,
    corrections: dict[str, str] | None = None,
    tokenize=None,
    vocabulary: dict[str, int] | None = None,
) -> ExpandedQuery:
    """Build the weighted term map every downstream retriever scores against.

    `terms` are the tokenized, stopword-filtered terms of the normalized query, produced by
    the SAME tokenizer the index was built with (injected by the caller) — using a second
    tokenizer here would mean the query and the corpus disagreed about what a word is.
    """
    weighted: dict[str, float] = {}
    for term in terms:
        weighted[term] = 1.0

    expanded = ExpandedQuery(
        original=original,
        normalized=normalized,
        original_terms=list(dict.fromkeys(terms)),
        corrections=dict(corrections or {}),
    )
    expanded.phrases = [
        (terms[index], terms[index + 1]) for index in range(len(terms) - 1)
    ]

    if not config.query_expansion_enabled or config.expansion_max_terms <= 0:
        expanded.terms = weighted
        return expanded

    weight = config.expansion_term_weight
    budget = config.expansion_max_terms
    added: list[str] = []

    def offer(candidate: str, candidate_weight: float) -> None:
        nonlocal budget
        if not candidate or len(candidate) < 2:
            return
        existing = weighted.get(candidate)
        if existing is not None:
            # Never lower a weight already assigned. A term the user typed stays at 1.0 even
            # if some expansion path rediscovers it. Checked BEFORE the budget on purpose:
            # raising an existing term's weight adds no term, so it spends nothing — gating
            # it on the budget meant a term's final weight depended on how many unrelated
            # expansions happened to be offered before it.
            if candidate_weight > existing:
                weighted[candidate] = candidate_weight
            return
        if budget <= 0:
            return
        weighted[candidate] = candidate_weight
        added.append(candidate)
        budget -= 1

    # 1. Spelling corrections first — they are the highest-confidence addition, because
    #    they were only proposed when exactly one corpus term could have been meant.
    for typo, correction in (corrections or {}).items():
        offer(correction, max(weight, 0.8))

    # 2. Abbreviation / canonical forms.
    for addition in canonicalize_terms(list(weighted)):
        offer(addition, weight)

    # 3. Operator dictionary, then built-in synonyms. The operator's entries win by being
    #    consulted first while budget is plentiful — a deployment that wrote a dictionary
    #    knows more about its documents than this file does.
    for term in list(expanded.original_terms):
        for alias in (dictionary or {}).get(term, ()):
            offer(alias, weight)
    for term in list(expanded.original_terms):
        for alias in BUILTIN_SYNONYMS.get(term, ()):
            offer(alias, weight * 0.9)

    # 4. Entity aliases drawn from the corpus's own extracted entities.
    for term in list(expanded.original_terms):
        for alias in (entity_aliases or {}).get(term, ()):
            offer(alias, weight)

    # 5. Hyphenation, both directions, then morphology.
    #
    # HYPHENATION IS A REAL AND INVISIBLE RETRIEVAL LOSS HERE, not a nicety. The corpus
    # tokenizer treats "-" as word-interior, so "real-time system" indexes as
    # ["real-time", "system"] and "real time system" indexes as ["real", "time", "system"] —
    # two spellings of one phrase with ONE term in common. Neither normalization nor BM25 can
    # see through that; the only honest fix is to offer the other spelling as well.
    #
    # Outward: a hyphenated term the user typed offers its parts and its joined form.
    for term in list(expanded.original_terms):
        for variant in hyphen_variants(term, tokenize):
            offer(variant, weight * 0.8)
    # Inward: adjacent typed terms offer their hyphenated and joined forms — but ONLY when
    # the corpus actually contains that token. Anchoring on the vocabulary is what keeps this
    # from being a guess: an offered term that no chunk contains would score nothing while
    # spending a slot of a deliberately small budget, and the weight is high precisely
    # because the corpus has already vouched for the term.
    if vocabulary:
        for left, right in expanded.phrases:
            for joined in (f"{left}-{right}", f"{left}{right}"):
                if joined in vocabulary:
                    offer(joined, max(weight, 0.85))
    for term in list(expanded.original_terms):
        for variant in _morphological_variants(term):
            offer(variant, weight * 0.7)

    expanded.terms = weighted
    expanded.added_terms = added
    return expanded


def apply_pseudo_relevance_feedback(
    expanded: ExpandedQuery,
    feedback_terms: list[tuple[str, float]],
    *,
    config,
) -> ExpandedQuery:
    """Fold the distinctive terms of the first-pass top hits back into the query.

    Classic Rocchio-style PRF with the generative half removed: the terms come from the
    corpus's own top-ranked passages, weighted by how distinctive they are there (the
    caller supplies an IDF-weighted score), never from a model's idea of what is related.

    Deterministic by construction — the same first pass yields the same feedback terms —
    and bounded by `prf_terms`, because a query that has absorbed thirty new words is no
    longer the question that was asked.

    PRF IS A RECALL DEVICE AND IS TREATED AS ONE. The terms it adds are recorded in
    `feedback_terms` and excluded from `rerank_terms`, so they can pull a passage INTO the
    candidate pool but have no say in which candidate wins. Letting them rerank was measured
    at -6.3 points of Recall@1 on the bundled benchmark (0.8750 -> 0.8125) while raising
    Recall@5 to 1.0 — textbook query drift, and precisely the split of responsibilities that
    fixes it.
    """
    if not config.prf_enabled or config.prf_terms <= 0:
        return expanded
    added = 0
    for term, _weight in feedback_terms:
        if added >= config.prf_terms:
            break
        if term in expanded.terms:
            continue
        expanded.terms[term] = config.prf_term_weight
        expanded.added_terms.append(term)
        expanded.feedback_terms.add(term)
        added += 1
    return expanded
