"""Unit tests for the high-precision mode's per-resource index — `rag/precision/index.py`.

Three properties in this module fail *silently* in production, which is why they are pinned
here rather than left to the pipeline's own tests:

* **The corpus statistics are the ranking.** `document_frequencies`, `lengths` and
  `average_length` are the entire input to BM25's IDF and length normalisation. A term
  counted once per *occurrence* instead of once per *chunk* does not raise — it quietly
  flattens IDF and changes which passage every question against that base gets back.
* **Candidate order must be reproducible.** `candidates_for` deduplicates by first
  appearance precisely because a set's iteration order is not stable, and an unstable
  candidate order makes two identical benchmark runs disagree on ties. Only an exact list
  assertion catches a regression to `set()`.
* **A cache hit must hand back THIS request's chunk objects.** The cache deliberately keeps
  derived data only; `_rebind` re-attaches the caller's rows. Returning the cached ORM
  instances instead would pass every test that builds and reads in one breath, and fail in
  production as a `DetachedInstanceError` inside a chat response — `get_db` detaches on
  teardown and this application defers the embedding columns (CLAUDE.md §9). The identity
  assertions below (`is`, not `==`) are the only way to state that difference.

Everything here is a pure unit: a local `FakeChunk` satisfies `types.ChunkLike`
structurally, so this file imports no model, no session and no `rag.service`.
"""
from __future__ import annotations

from dataclasses import dataclass
import re

import pytest

from rag.precision import index as index_module
from rag.precision.index import (
    MAX_CACHED_INDEXES,
    build_index,
    fingerprint_of,
    get_index,
    invalidate,
)

pytestmark = pytest.mark.unit


@dataclass
class FakeChunk:
    """The attributes `index.py` and `metadata.derive_metadata` read off a chunk.

    A local dataclass rather than `models.rag.DocumentChunk` on purpose: the precision
    package is structurally typed so it can be exercised with no database at all, and a test
    that imports the model would quietly give that property up.
    """

    id: str
    content: str = "plain prose with nothing that looks like a heading"
    file_id: str | None = "file-1"
    chunk_index: int = 0
    source_name: str = "corpus.txt"
    modality: str = "text"
    title: str | None = None
    contextual_content: str = ""
    page_start: int | None = None
    page_end: int | None = None
    char_start: int | None = None
    char_end: int | None = None


# A hand-computable 3-chunk corpus. Every expectation below is arithmetic over this table
# and is written out literally, so a change to how a statistic is accumulated has to be
# argued for rather than absorbed.
#
#   term    c1  c2  c3   df   total
#   alpha    2   .   1    2     3
#   beta     1   3   .    2     4
#   gamma    .   1   2    2     3
#   delta    .   .   5    1     5
#   length   3   4   8         15  -> average 5.0
_TERMS: dict[str, dict[str, int]] = {
    "c1": {"alpha": 2, "beta": 1},
    "c2": {"beta": 3, "gamma": 1},
    "c3": {"alpha": 1, "gamma": 2, "delta": 5},
}

_STOPWORDS = {"the", "a", "an", "and", "of"}
_WORD = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    """A stand-in for `rag.service._tokenize`: lowercased words, minus stopwords and singles.

    Duplicated rather than imported because the real one drags in the database session, and
    because what these tests care about is the *contract* the index relies on — that the
    corpus tokenizer drops stopwords.
    """
    return [word for word in _WORD.findall((text or "").lower()) if len(word) > 1 and word not in _STOPWORDS]


def _corpus() -> list[FakeChunk]:
    """A fresh set of chunk objects carrying the same ids every time.

    Freshness is the point: the cache-identity tests need two lists that are equal by id and
    distinct by identity, exactly as two requests loading the same rows are.
    """
    return [
        FakeChunk(id="c1", chunk_index=0, content="alpha beta prose"),
        FakeChunk(id="c2", chunk_index=1, content="beta gamma prose"),
        FakeChunk(id="c3", chunk_index=2, content="alpha gamma delta prose"),
    ]


def _terms_of(chunk) -> dict[str, int]:
    return dict(_TERMS.get(chunk.id, {"filler": 1}))


def _counting_terms_of() -> tuple:
    """`(terms_of, calls)` — `calls` records one entry per chunk the builder derived.

    This is how "did it rebuild?" is asked without reaching into the cache: `terms_of` is
    called exactly once per chunk per build and never on a cache hit.
    """
    calls: list[str] = []

    def terms_of(chunk):
        calls.append(chunk.id)
        return _terms_of(chunk)

    return terms_of, calls


def _build(resource_id, chunks, *, terms_of=_terms_of, extract_entities=None, parent_group_size=3, tokenize=None):
    return build_index(
        resource_id,
        chunks,
        terms_of=terms_of,
        extract_entities=extract_entities,
        parent_group_size=parent_group_size,
        tokenize=tokenize,
    )


def _get(resource_id, chunks, *, terms_of=_terms_of, extract_entities=None, parent_group_size=3, tokenize=None):
    return get_index(
        resource_id,
        chunks,
        terms_of=terms_of,
        extract_entities=extract_entities,
        parent_group_size=parent_group_size,
        tokenize=tokenize,
    )


@pytest.fixture(autouse=True)
def _isolated_cache():
    """The index cache is process-global, so a leak between tests is a false pass.

    Cleared on both sides: before, so a previous module's resource id cannot answer a build
    here; after, so nothing this file cached outlives it.
    """
    index_module.clear_cache()
    yield
    index_module.clear_cache()


class TestBuildIndexStatistics:
    def test_corpus_statistics_match_the_hand_computed_table(self):
        index = _build("r", _corpus())

        # df counts CHUNKS containing a term; vocabulary counts OCCURRENCES. Conflating the
        # two is the single most damaging mistake available here: df feeds IDF, so counting
        # occurrences there would make a term that appears five times in one chunk look as
        # common as one spread across five chunks.
        assert index.document_frequencies == {"alpha": 2, "beta": 2, "gamma": 2, "delta": 1}
        assert index.vocabulary == {"alpha": 3, "beta": 4, "gamma": 3, "delta": 5}
        assert index.lengths == {"c1": 3, "c2": 4, "c3": 8}
        assert index.average_length == 5.0  # 15 / 3
        assert index.terms["c3"] == {"alpha": 1, "gamma": 2, "delta": 5}

    def test_postings_list_every_chunk_holding_a_term_in_corpus_order(self):
        index = _build("r", _corpus())

        assert index.postings == {
            "alpha": ["c1", "c3"],
            "beta": ["c1", "c2"],
            "gamma": ["c2", "c3"],
            "delta": ["c3"],
        }

    def test_by_id_holds_the_objects_that_were_passed_in(self):
        chunks = _corpus()
        index = _build("r", chunks)

        assert index.by_id == {"c1": chunks[0], "c2": chunks[1], "c3": chunks[2]}
        assert index.chunks == chunks
        # A copy of the list, so a caller mutating its own list afterwards cannot reorder
        # what the index walks.
        assert index.chunks is not chunks

    def test_a_chunk_with_no_terms_still_counts_as_a_document(self):
        # `terms_of` returning None is `or {}` in the builder. The chunk contributes no
        # postings but IS a document: dropping it from `terms` would change N for every IDF
        # in the corpus, and dropping it from `lengths` would raise later in BM25.
        chunks = [FakeChunk(id="c1", chunk_index=0), FakeChunk(id="empty", chunk_index=1)]
        index = _build("r", chunks, terms_of=lambda chunk: None if chunk.id == "empty" else {"alpha": 3})

        assert index.terms["empty"] == {}
        assert index.lengths["empty"] == 0
        assert index.document_count == 2
        assert index.average_length == 1.5  # 3 / 2 — the empty chunk drags the average down

    def test_an_empty_corpus_produces_a_zero_average_rather_than_dividing_by_zero(self):
        index = _build("r", [])

        assert index.average_length == 0.0
        assert index.document_count == 0
        assert index.candidates_for(["alpha"]) == []

    def test_fingerprint_is_the_corpus_plus_the_derivation_parameters(self):
        # Deliberately cheap on the corpus side — see `TestCacheValidity` for what that does
        # and does not catch. The last two entries are the DERIVATION: `parent_group_size`
        # and whether entities were extracted. Without them, two requests against the same
        # unchanged corpus that legitimately asked for different settings — which
        # `POST /resources/{id}/precision-trace` allows, since it accepts a config patch —
        # would share one cached index and the second would silently get the first's answer.
        assert fingerprint_of("r", _corpus()) == ("r", 3, "c1", "c3", 0, False)
        assert fingerprint_of("r", _corpus(), 3, True) == ("r", 3, "c1", "c3", 3, True)
        assert fingerprint_of("r", []) == ("r", 0, "", "", 0, False)
        assert fingerprint_of("r", _corpus(), 3, True) != fingerprint_of("r", _corpus(), 5, True)


class TestCandidatesFor:
    def test_returns_only_chunks_containing_a_term_in_a_stable_order(self):
        index = _build("r", _corpus())

        # The exact list, not a set: postings are walked query-term by query-term and
        # deduplicated by first appearance. c3 holds both terms and must appear once, at the
        # position its FIRST posting put it.
        assert index.candidates_for(["gamma", "alpha"]) == ["c2", "c3", "c1"]
        # Same terms, other order — a different, equally reproducible answer.
        assert index.candidates_for(["alpha", "gamma"]) == ["c1", "c3", "c2"]

    def test_a_repeated_query_term_does_not_duplicate_a_candidate(self):
        index = _build("r", _corpus())

        assert index.candidates_for(["alpha", "alpha"]) == ["c1", "c3"]

    def test_a_term_the_corpus_does_not_have_contributes_nothing(self):
        index = _build("r", _corpus())

        assert index.candidates_for(["epsilon"]) == []
        # An unknown term must not suppress the known ones beside it.
        assert index.candidates_for(["epsilon", "delta"]) == ["c3"]

    def test_no_query_terms_means_no_candidates(self):
        index = _build("r", _corpus())

        assert index.candidates_for([]) == []


class TestFamilies:
    def test_siblings_group_by_parent_window_and_sort_by_chunk_index(self):
        # Shuffled on the way in, because chunks for a base whose files were indexed
        # concurrently do not necessarily load in `chunk_index` order — and parent recovery
        # walks this list to stitch a window together, so a wrong order reads back as
        # scrambled prose rather than as an error.
        ordered = [FakeChunk(id=f"c{i}", chunk_index=i) for i in range(4)]
        shuffled = [ordered[2], ordered[0], ordered[3], ordered[1]]
        index = _build("r", shuffled, parent_group_size=2)

        assert index.families == {"file-1:0": ["c0", "c1"], "file-1:1": ["c2", "c3"]}

    def test_chunks_from_two_files_are_never_one_family(self):
        # Stitching the tail of one document onto the head of the next would fabricate
        # context that exists in neither.
        chunks = [
            FakeChunk(id="a0", file_id="file-a", chunk_index=0),
            FakeChunk(id="b0", file_id="file-b", chunk_index=0),
        ]
        index = _build("r", chunks, parent_group_size=3)

        assert index.families == {"file-a:0": ["a0"], "file-b:0": ["b0"]}

    def test_a_missing_chunk_index_sorts_as_zero_rather_than_raising(self):
        # `chunk_index` is typed int but arrives from a row; None must not take the build
        # down, because one malformed row would cost the whole knowledge base.
        chunks = [FakeChunk(id="c1", chunk_index=1), FakeChunk(id="c0", chunk_index=None)]
        index = _build("r", chunks, parent_group_size=5)

        assert index.families == {"file-1:0": ["c0", "c1"]}

    def test_categories_collects_what_the_corpus_actually_has(self):
        # The set is what `metadata.infer_filters` consults before proposing a category
        # filter, so a category the corpus does not hold must never appear in it.
        chunks = [
            FakeChunk(id="c1", source_name="api-reference.pdf"),
            FakeChunk(id="c2", source_name="corpus.txt"),
        ]
        index = _build("r", chunks)

        assert index.categories == {"api"}


class TestEntityAliases:
    def test_multi_word_entities_alias_each_other(self):
        # "Gemera" reaching passages that only say "Koenigsegg" is an alias the corpus
        # itself asserts, which is the whole reason this is derived rather than looked up.
        index = _build(
            "r",
            [FakeChunk(id="c1")],
            extract_entities=lambda text: ["Koenigsegg Gemera", "Koenigsegg Jesko"],
            tokenize=_tokenize,
        )

        assert index.entity_aliases["gemera"] == ("koenigsegg",)
        assert index.entity_aliases["jesko"] == ("koenigsegg",)
        # Merged across both entities, first appearance first.
        assert index.entity_aliases["koenigsegg"] == ("gemera", "jesko")

    def test_the_supplied_tokenizer_drops_stopwords_from_aliases(self):
        # A real bug this guards: entity extraction returns capitalised runs, and a run like
        # "Connection Pooling The" ends on the first word of the NEXT sentence. Splitting on
        # whitespace made "the" an alias for "connection", which then matched every chunk in
        # the corpus. The corpus tokenizer already drops stopwords, so it is the filter.
        index = _build(
            "r",
            [FakeChunk(id="c1")],
            extract_entities=lambda text: ["Connection Pooling The"],
            tokenize=_tokenize,
        )

        assert index.entity_aliases == {"connection": ("pooling",), "pooling": ("connection",)}
        assert "the" not in index.entity_aliases
        assert all("the" not in aliases for aliases in index.entity_aliases.values())

    def test_without_a_tokenizer_no_aliases_are_built_at_all(self):
        # There used to be a whitespace-split fallback here, and it reproduced exactly the
        # bug the tokenizer was introduced to fix: an entity run ends on the first word of
        # the following sentence, so "Connection Pooling The" made "the" an alias for
        # "connection" — a term in every chunk of the corpus, offered as an expansion.
        # A caller with no tokenizer now gets NO aliases, which is a much smaller wrong
        # answer than a corpus-wide one.
        index = _build(
            "r",
            [FakeChunk(id="c1")],
            extract_entities=lambda text: ["Connection Pooling The"],
        )

        assert index.entity_aliases == {}

    def test_a_single_word_entity_produces_no_alias(self):
        # One word cannot assert a relationship, and aliasing it to nothing would still cost
        # a dictionary entry every expansion has to look through.
        index = _build(
            "r",
            [FakeChunk(id="c1")],
            extract_entities=lambda text: ["Koenigsegg", "A B"],
            tokenize=_tokenize,
        )

        assert index.entity_aliases == {}

    def test_aliases_are_capped_at_six_per_word(self):
        # An expansion is a guess; a long capitalised run must not be able to add a dozen
        # guessed terms to every query that touches one of its words.
        entity = "One Two Three Four Five Six Seven Eight"
        index = _build("r", [FakeChunk(id="c1")], extract_entities=lambda text: [entity], tokenize=_tokenize)

        assert index.entity_aliases["one"] == ("two", "three", "four", "five", "six", "seven")

    def test_only_the_first_twelve_entities_of_a_chunk_are_kept(self):
        many = [f"Alpha{n} Beta{n}" for n in range(15)]
        index = _build("r", [FakeChunk(id="c1")], extract_entities=lambda text: many, tokenize=_tokenize)

        assert len(index.metadata["c1"].entities) == 12
        assert "alpha11" in index.entity_aliases
        assert "alpha12" not in index.entity_aliases

    def test_an_extractor_that_raises_is_swallowed(self):
        # Entity extraction is a nicety layered on top of retrieval. A regex that blows up
        # on one pathological passage must cost that passage its entities, not cost the user
        # their knowledge base.
        def explode(text):
            raise ValueError("bad passage")

        index = _build("r", _corpus(), extract_entities=explode, tokenize=_tokenize)

        assert index.entity_aliases == {}
        assert index.metadata["c1"].entities == ()
        # ...and everything else about the build is unaffected.
        assert index.document_frequencies == {"alpha": 2, "beta": 2, "gamma": 2, "delta": 1}


class TestDocumentCount:
    def test_document_count_reads_the_derived_terms_not_the_chunk_list(self):
        index = _build("r", _corpus())
        # `chunks` is rebound per call and `terms` is not. If N came off the chunk list, a
        # cached index stored with an empty one would report N=0 and turn every IDF into a
        # constant.
        index.chunks = []
        index.by_id = {}

        assert index.document_count == 3

    def test_document_count_survives_the_cache_round_trip(self):
        _get("r", _corpus())
        cached = index_module._CACHE["r"]

        # The cache entry itself holds no chunks at all, and still knows N.
        assert cached.chunks == []
        assert cached.by_id == {}
        assert cached.document_count == 3
        assert _get("r", _corpus()).document_count == 3


class TestCacheValidity:
    def test_a_second_call_with_the_same_chunks_does_not_rebuild(self):
        terms_of, calls = _counting_terms_of()
        _get("r", _corpus(), terms_of=terms_of)
        assert calls == ["c1", "c2", "c3"]

        second = _get("r", _corpus(), terms_of=terms_of)

        assert calls == ["c1", "c2", "c3"]  # nothing re-derived
        assert second.document_frequencies == {"alpha": 2, "beta": 2, "gamma": 2, "delta": 1}
        # A cache hit is a shallow VIEW: the heavy dictionaries are shared by reference, not
        # copied, which is what makes rebinding cost one dict comprehension.
        assert second.terms is index_module._CACHE["r"].terms
        assert second.families is index_module._CACHE["r"].families

    def test_a_different_chunk_count_rebuilds(self):
        terms_of, calls = _counting_terms_of()
        _get("r", _corpus(), terms_of=terms_of)

        index = _get("r", _corpus()[:2], terms_of=terms_of)

        assert calls == ["c1", "c2", "c3", "c1", "c2"]
        assert index.document_count == 2
        assert "delta" not in index.document_frequencies

    def test_a_different_first_id_rebuilds(self):
        terms_of, calls = _counting_terms_of()
        _get("r", _corpus(), terms_of=terms_of)

        chunks = _corpus()
        chunks[0] = FakeChunk(id="new-head", chunk_index=0)
        _get("r", chunks, terms_of=terms_of)

        assert calls == ["c1", "c2", "c3", "new-head", "c2", "c3"]

    def test_a_different_last_id_rebuilds(self):
        terms_of, calls = _counting_terms_of()
        _get("r", _corpus(), terms_of=terms_of)

        chunks = _corpus()
        chunks[-1] = FakeChunk(id="new-tail", chunk_index=2)
        _get("r", chunks, terms_of=terms_of)

        assert calls == ["c1", "c2", "c3", "c1", "c2", "new-tail"]

    def test_a_change_the_fingerprint_cannot_see_is_the_reason_invalidate_exists(self):
        # Count and endpoints unchanged, contents reordered: the fingerprint says "same
        # corpus" and the cached derivation is served. This is a documented limitation, not
        # an oversight — chunk ids are UUIDs so a re-upload almost always moves an endpoint,
        # and `invalidate()` is what the ingestion path calls for everything else.
        terms_of, calls = _counting_terms_of()
        chunks = _corpus() + [FakeChunk(id="c4", chunk_index=3)]
        _get("r", chunks, terms_of=terms_of)

        # The two middle chunks swap places; head, tail and count are untouched.
        view = _get("r", [chunks[0], chunks[2], chunks[1], chunks[3]], terms_of=terms_of)

        assert calls == ["c1", "c2", "c3", "c4"]
        # The caller's order is what `chunks` reports back, but the derived postings still
        # describe the order the index was built in.
        assert [chunk.id for chunk in view.chunks] == ["c1", "c3", "c2", "c4"]
        assert view.postings["alpha"] == ["c1", "c3"]

    def test_invalidate_forces_a_rebuild(self):
        terms_of, calls = _counting_terms_of()
        _get("r", _corpus(), terms_of=terms_of)

        invalidate("r")
        _get("r", _corpus(), terms_of=terms_of)

        assert calls == ["c1", "c2", "c3", "c1", "c2", "c3"]

    def test_invalidate_leaves_other_resources_alone_and_ignores_unknown_ones(self):
        terms_of, calls = _counting_terms_of()
        _get("r1", _corpus(), terms_of=terms_of)
        _get("r2", _corpus(), terms_of=terms_of)

        invalidate("never-cached")  # callers do not know whether this mode was ever used
        invalidate("r1")
        _get("r2", _corpus(), terms_of=terms_of)

        assert calls == ["c1", "c2", "c3"] * 2  # r2 still cached, only r1 was dropped

    def test_clear_cache_empties_everything(self):
        terms_of, calls = _counting_terms_of()
        _get("r1", _corpus(), terms_of=terms_of)
        _get("r2", _corpus(), terms_of=terms_of)

        index_module.clear_cache()

        assert index_module._CACHE == {}
        _get("r1", _corpus(), terms_of=terms_of)
        _get("r2", _corpus(), terms_of=terms_of)
        assert calls == ["c1", "c2", "c3"] * 4  # both had to be rebuilt

    def test_the_oldest_index_is_evicted_past_the_cache_bound(self):
        # Each entry is proportional to its corpus, so an unbounded cache would hold every
        # knowledge base anyone has asked about for the process lifetime. Eviction costs one
        # rebuild, which is the trade being asserted here.
        terms_of, calls = _counting_terms_of()
        for n in range(MAX_CACHED_INDEXES + 1):
            _get(f"r{n}", _corpus(), terms_of=terms_of)
        built = len(calls)

        _get(f"r{MAX_CACHED_INDEXES}", _corpus(), terms_of=terms_of)  # newest: still cached
        assert len(calls) == built

        _get("r0", _corpus(), terms_of=terms_of)  # oldest: evicted, so rebuilt
        assert len(calls) == built + 3


class TestCacheHandsBackThisRequestsChunks:
    def test_a_cache_hit_returns_the_objects_passed_on_this_call(self):
        # THE one that matters. Cached ORM instances belong to a session that has since
        # closed: `get_db` detaches on teardown and this application defers the embedding
        # columns, so touching a stale row raises `DetachedInstanceError` inside a chat
        # response rather than issuing a query. `==` cannot see the difference — only `is`.
        terms_of, calls = _counting_terms_of()
        first_chunks = _corpus()
        first = _get("r", first_chunks, terms_of=terms_of)
        assert first.by_id["c1"] is first_chunks[0]

        second_chunks = _corpus()  # same ids, brand-new objects, as a second request loads
        second = _get("r", second_chunks, terms_of=terms_of)

        assert calls == ["c1", "c2", "c3"]  # served from cache, so this is the risky path
        for stale, fresh in zip(first_chunks, second_chunks):
            assert second.by_id[fresh.id] is fresh
            assert second.by_id[fresh.id] is not stale
        assert [chunk.id for chunk in second.chunks] == ["c1", "c2", "c3"]
        assert all(cached is fresh for cached, fresh in zip(second.chunks, second_chunks))

    def test_rebinding_does_not_disturb_the_cached_entry(self):
        # The view is a copy; mutating what a request got back must not reach the next one.
        first = _get("r", _corpus())
        first.chunks.append(FakeChunk(id="c4", chunk_index=3))

        second = _get("r", _corpus())

        assert index_module._CACHE["r"].chunks == []
        assert [chunk.id for chunk in second.chunks] == ["c1", "c2", "c3"]

    def test_a_rebuild_also_returns_this_calls_chunks(self):
        chunks = _corpus()
        index = _get("r", chunks)

        assert all(index.by_id[chunk.id] is chunk for chunk in chunks)
