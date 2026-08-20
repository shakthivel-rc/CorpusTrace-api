"""Deterministic query normalization — stage 1 of the high-precision pipeline.

Everything here is a pure function of its input. No model, no LLM, no network, no
randomness: the same query normalizes to the same string on every machine and in every
process, which is the property that makes the benchmark in `benchmark.py` mean anything.

The one judgement call worth stating: normalization NEVER drops a word it does not
recognise. Its job is to remove notation the retriever cannot see through (curly quotes,
NBSP, a trailing question mark glued to the last term), not to decide what the user meant.
Deciding what they meant is `expansion.py`'s job, and it does it additively — the original
terms always survive at full weight.
"""
from __future__ import annotations

import re
import unicodedata


# Characters that carry no retrieval signal but do split a token, mapped to a space.
#
# THE ASCII HYPHEN IS DELIBERATELY ABSENT, and so is the underscore. This class has to agree
# with the tokenizer the corpus was indexed by (`_tokenize` in `rag/service.py`, regex
# `[A-Za-z0-9][A-Za-z0-9_-]{1,}`), which treats `-` and `_` as WORD-INTERIOR. Splitting them
# here would turn "state-of-the-art" into "state art" while the index holds the single token
# "state-of-the-art", so the query and the corpus would disagree about what a word is — the
# exact failure this module exists to prevent, introduced by the fix for it.
#
# The Unicode dashes U+2010-U+2015 and the minus sign U+2212 ARE split, and that is the same
# rule rather than an exception: the tokenizer does not accept them either, so it already
# treats them as boundaries.
#
# Bridging "state-of-the-art" and "state of the art" is real and is handled where it belongs
# — additively, in `expansion.hyphen_variants`, which offers both forms instead of choosing.
_PUNCT_TO_SPACE = re.compile(r"[‐-―−/\\|,;:()\[\]{}<>\"'`~!?*^]+")
_COLLAPSE_WS = re.compile(r"\s+")
# A period that ends a sentence rather than sitting inside "4.2" or "v1.0.3".
_TRAILING_DOT = re.compile(r"(?<=[A-Za-z])\.(?=\s|$)")
# Possessives: "the engine's capacity" -> "the engine capacity". Applied before the
# punctuation map so the apostrophe does not become a token boundary that leaves a bare "s".
#
# Case-insensitive because `casefold()` does not run until the end of `normalize_query`: with
# a case-sensitive `s` this rule simply never fired on "THE ENGINE'S CAPACITY", and the
# symptom was a shouted query tokenizing differently from the same words typed normally.
#
# It cannot distinguish a possessive from a contraction of "is"/"has", so "what's" becomes
# "what". That is harmless HERE and only here: both "is" and "has" are in STOPWORDS, so the
# tokenizer would have discarded them anyway, and the surviving word is the same either way.
_POSSESSIVE = re.compile(r"(\w)['’]s\b", re.IGNORECASE)
_APOSTROPHE = re.compile(r"['’](?=\w)")


# Abbreviations and terminology that are unambiguous in general technical prose. Kept
# deliberately small and domain-NEUTRAL: a domain dictionary belongs in a deployment's
# configuration (see `expansion.load_dictionary`), never compiled into the engine, because
# a wrong expansion is indistinguishable from a retrieval bug from the outside.
#
# These map a written form to its CANONICAL form. Both sides survive into the query — the
# canonical form is added, the typed form is never removed — so a document that uses the
# abbreviation and a document that spells it out both stay reachable.
CANONICAL_FORMS: dict[str, str] = {
    "db": "database",
    "dbs": "databases",
    "auth": "authentication",
    "authn": "authentication",
    "authz": "authorization",
    "config": "configuration",
    "configs": "configurations",
    "env": "environment",
    "api": "api",
    "apis": "api",
    "spec": "specification",
    "specs": "specifications",
    "docs": "documentation",
    "doc": "document",
    "info": "information",
    "admin": "administrator",
    "repo": "repository",
    "req": "request",
    "reqs": "requirements",
    "resp": "response",
    "cert": "certificate",
    "certs": "certificates",
    "perf": "performance",
    "vuln": "vulnerability",
    "creds": "credentials",
    "cred": "credential",
    "params": "parameters",
    "param": "parameter",
    "args": "arguments",
    "arg": "argument",
    "vars": "variables",
    "var": "variable",
    "func": "function",
    "funcs": "functions",
    "impl": "implementation",
    "init": "initialization",
    "sync": "synchronization",
    "async": "asynchronous",
    "temp": "temperature",
    "max": "maximum",
    "min": "minimum",
    "avg": "average",
    "num": "number",
    "qty": "quantity",
    "approx": "approximately",
    "vs": "versus",
    "eg": "example",
    "ie": "that is",
}


def normalize_query(query: str) -> str:
    """Whitespace, punctuation, case and Unicode, in that order.

    NFKC first, so a full-width or ligatured character becomes the ASCII the index was
    built from. `_tokenize` in `rag/service.py` matches `[A-Za-z0-9][A-Za-z0-9_-]{1,}`, so
    a curly apostrophe or an en-dash silently splits a term that the document spells with
    an ASCII one — which reads to the user as the document not containing their word.
    """
    if not query:
        return ""
    text = unicodedata.normalize("NFKC", query)
    text = _POSSESSIVE.sub(r"\1", text)
    text = _APOSTROPHE.sub("", text)
    text = _TRAILING_DOT.sub(" ", text)
    text = _PUNCT_TO_SPACE.sub(" ", text)
    # Case is folded, not lowered: `casefold` handles ß/İ correctly, and the downstream
    # tokenizer lowercases anyway — doing it here means every stage sees one form.
    text = text.casefold()
    return _COLLAPSE_WS.sub(" ", text).strip()


def canonicalize_terms(terms: list[str]) -> list[str]:
    """Canonical forms for any recognised abbreviation, in first-seen order.

    Returns only the ADDITIONS. The caller keeps the original terms at full weight; these
    join at the expansion weight, because "db" meaning "decibel" is rare but real.
    """
    additions: list[str] = []
    seen = set(terms)
    for term in terms:
        canonical = CANONICAL_FORMS.get(term)
        if not canonical:
            continue
        for word in canonical.split():
            if word not in seen:
                seen.add(word)
                additions.append(word)
    return additions


def _edit_distance_within_one(left: str, right: str) -> bool:
    """True when `left` and `right` differ by at most one insert, delete or substitute.

    A bounded check rather than a full Levenshtein: the only question asked here is "is
    this a typo of that", and the answer for distance 2+ is "do not guess".
    """
    if left == right:
        return True
    len_left, len_right = len(left), len(right)
    if abs(len_left - len_right) > 1:
        return False
    if len_left == len_right:
        differences = sum(1 for a, b in zip(left, right) if a != b)
        return differences <= 1
    # One is exactly one character longer: it must be the other plus one insertion.
    longer, shorter = (left, right) if len_left > len_right else (right, left)
    index_long = index_short = 0
    skipped = False
    while index_long < len(longer) and index_short < len(shorter):
        if longer[index_long] != shorter[index_short]:
            if skipped:
                return False
            skipped = True
            index_long += 1
            continue
        index_long += 1
        index_short += 1
    return True


MIN_SPELLING_LENGTH = 5


def correct_spelling(terms: list[str], vocabulary: dict[str, int]) -> dict[str, str]:
    """Corpus-anchored spelling correction, applied only where it is provably safe.

    Four conditions, and every one of them exists to stop a "correction" from moving a
    question away from its own answer:

      1. the typed term is ABSENT from the corpus vocabulary — a word the documents
         actually use is never a typo, whatever a dictionary thinks of it;
      2. the term is at least `MIN_SPELLING_LENGTH` characters — at four characters and
         below, distance-1 neighbours are usually different words ("port"/"part");
      3. exactly ONE vocabulary term is within edit distance 1 — two candidates means the
         correct answer is unknown, and picking the more frequent one is a coin flip
         dressed as a decision;
      4. that term occurs at least twice, so a typo inside the document cannot recruit the
         query into matching it.

    Returns `{typed_term: corrected_term}`. The caller ADDS the correction and keeps the
    typed term, so even a wrong correction can only widen the search, never redirect it —
    and the typed term still reaches `_unmatched_terms`, so the vocabulary hint that tells
    the user "value" matched nothing keeps working.
    """
    corrections: dict[str, str] = {}
    if not vocabulary:
        return corrections
    for term in terms:
        if len(term) < MIN_SPELLING_LENGTH or term in vocabulary:
            continue
        matches = [
            candidate
            for candidate, count in vocabulary.items()
            if count >= 2
            and abs(len(candidate) - len(term)) <= 1
            and _edit_distance_within_one(term, candidate)
        ]
        if len(matches) == 1:
            corrections[term] = matches[0]
    return corrections
