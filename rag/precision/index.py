"""The per-resource retrieval index for the high-precision mode — built once, then cached.

WHAT IT HOLDS, and why each piece cannot be computed per question:

  * BM25 corpus statistics (document frequencies, average length). IDF is a statement about
    the whole corpus; recomputing it per question is an O(N) pass whose answer never
    changes between two questions about the same documents.
  * A postings list (term -> chunk ids). This is what turns "score every chunk" into "score
    the chunks that contain at least one query term", which on a real corpus is a small
    fraction of it.
  * The derived structure of every chunk (`metadata.derive_metadata`) and the sibling
    ordering parent recovery walks.
  * The corpus vocabulary, which corpus-anchored spelling correction needs.

CACHE VALIDITY, and the two rules that make caching safe here.

FIRST: the cache holds DERIVED DATA ONLY — term maps, frequencies, postings, metadata,
family membership — all keyed by chunk id. It never retains a `DocumentChunk`. Every call
rebinds `chunks`/`by_id` to the list the caller just loaded (`_rebind`, a shallow copy that
shares the heavy dictionaries). Retaining the ORM instances would hand a later request rows
belonging to a session that has since closed: `get_db` detaches on teardown, and this
application defers the embedding columns (`_DEFER_VECTORS` in `rag/service.py`), so touching
one on a stale instance raises `DetachedInstanceError` inside a chat response rather than
issuing a query. It also keeps every cached passage's text alive for the process lifetime.

SECOND: validity is a fingerprint — count, first id, last id — AND an explicit
`invalidate(resource_id)` called by the ingestion path. The fingerprint alone is not enough
and is not claimed to be: chunk ids are UUIDs so a re-upload almost always changes it, but
re-embedding in place changes neither the count nor the endpoints. `rag/service.py` calls
`invalidate` when a document finishes indexing and when a resource is deleted or purged;
the fingerprint is the backstop for anything that ever writes chunks without going through
those paths.

This module imports nothing from `rag.service`: the tokenizer, the term maps and the entity
extractor all arrive as callables. That is what lets the whole pipeline be tested with plain
objects and no database.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field, replace
import threading
from typing import Callable

from rag.precision import metadata as metadata_module
from rag.precision.types import ChunkLike, ChunkMetadata


# How many resources' indexes are held at once. Small on purpose: each entry is
# proportional to the corpus, and a chat process serving many users would otherwise hold
# every knowledge base anyone has asked about. Eviction costs one rebuild.
MAX_CACHED_INDEXES = 8

_CACHE: "OrderedDict[str, PrecisionIndex]" = OrderedDict()
_CACHE_LOCK = threading.Lock()


@dataclass
class PrecisionIndex:
    """Everything the pipeline needs to know about one resource's chunks."""

    resource_id: str
    fingerprint: tuple
    chunks: list[ChunkLike] = field(default_factory=list)
    by_id: dict[str, ChunkLike] = field(default_factory=dict)
    terms: dict[str, dict[str, int]] = field(default_factory=dict)
    lengths: dict[str, int] = field(default_factory=dict)
    document_frequencies: dict[str, int] = field(default_factory=dict)
    vocabulary: dict[str, int] = field(default_factory=dict)
    postings: dict[str, list[str]] = field(default_factory=dict)
    metadata: dict[str, ChunkMetadata] = field(default_factory=dict)
    entity_aliases: dict[str, tuple[str, ...]] = field(default_factory=dict)
    categories: set[str] = field(default_factory=set)
    # parent key -> chunk ids, ordered by chunk_index. The parent window's membership.
    families: dict[str, list[str]] = field(default_factory=dict)
    average_length: float = 0.0

    @property
    def document_count(self) -> int:
        """N for IDF. Read off the DERIVED data, not `chunks`.

        `chunks` is rebound per call and `terms` is not, so reading the chunk list here
        would make IDF depend on which request asked — and on a cached index stored with an
        empty chunk list it would be zero, turning every IDF into a constant.
        """
        return len(self.terms)

    def candidates_for(self, query_terms: list[str]) -> list[str]:
        """Chunk ids containing at least one of `query_terms`, in a stable order.

        Deduplicated by first appearance rather than by a set, so the candidate order is
        reproducible across runs — a set's iteration order is not, and an unstable
        candidate order makes two identical benchmark runs disagree on ties.
        """
        seen: dict[str, None] = {}
        for term in query_terms:
            for chunk_id in self.postings.get(term, ()):
                if chunk_id not in seen:
                    seen[chunk_id] = None
        return list(seen)


def fingerprint_of(
    resource_id: str,
    chunks: list[ChunkLike],
    parent_group_size: int = 0,
    entities: bool = False,
) -> tuple:
    """What the cached index was built FROM — the corpus and the derivation parameters.

    The derivation half is not decoration. `POST /resources/{id}/precision-trace` accepts a
    sparse config patch, so two requests against the same unchanged corpus can legitimately
    ask for different `parent_group_size` values or ask with entity extraction off; keyed on
    the corpus alone, the second would silently receive the first one's families and aliases.
    """
    if not chunks:
        return (resource_id, 0, "", "", parent_group_size, entities)
    return (resource_id, len(chunks), chunks[0].id, chunks[-1].id, parent_group_size, entities)


def build_index(
    resource_id: str,
    chunks: list[ChunkLike],
    *,
    terms_of: Callable[[ChunkLike], dict[str, int]],
    extract_entities: Callable[[str], list[str]] | None,
    parent_group_size: int,
    tokenize: Callable[[str], list[str]] | None = None,
) -> PrecisionIndex:
    """One pass over the corpus. Everything derived here is derived exactly once."""
    index = PrecisionIndex(
        resource_id=resource_id,
        fingerprint=fingerprint_of(
            resource_id, chunks, parent_group_size, extract_entities is not None
        ),
    )
    index.chunks = list(chunks)

    total_length = 0
    for chunk in chunks:
        chunk_id = chunk.id
        index.by_id[chunk_id] = chunk
        term_map = terms_of(chunk) or {}
        index.terms[chunk_id] = term_map
        length = sum(term_map.values())
        index.lengths[chunk_id] = length
        total_length += length

        for term, count in term_map.items():
            index.document_frequencies[term] = index.document_frequencies.get(term, 0) + 1
            index.vocabulary[term] = index.vocabulary.get(term, 0) + count
            index.postings.setdefault(term, []).append(chunk_id)

        entities: tuple[str, ...] = ()
        if extract_entities is not None:
            try:
                entities = tuple(extract_entities(chunk.content or "")[:12])
            except Exception:  # noqa: BLE001 - entity extraction is a nicety, never a failure
                entities = ()
        meta = metadata_module.derive_metadata(chunk, entities, parent_group_size)
        index.metadata[chunk_id] = meta
        if meta.category:
            index.categories.add(meta.category)
        index.families.setdefault(meta.parent_chunk_id or chunk_id, []).append(chunk_id)

        for entity in entities:
            # Tokenized with the CORPUS tokenizer, not `str.split`. Entity extraction
            # returns capitalised runs, and a run like "Connection Pooling The" ends on the
            # first word of the following sentence — splitting on whitespace turned "the"
            # into a query-expansion alias for "connection", which then matched every chunk
            # in the corpus. The tokenizer drops stopwords, which is exactly the filter
            # needed and one that already agrees with the index.
            # No whitespace-split fallback. It was the original implementation and it is
            # exactly the bug the tokenizer was introduced to fix — "Connection Pooling The"
            # made "the" an alias for "connection". A caller with no tokenizer gets no
            # aliases, which is a smaller wrong answer than a corpus-wide one.
            words = tokenize(entity) if tokenize is not None else []
            words = [word for word in dict.fromkeys(words) if len(word) > 1]
            if len(words) < 2:
                continue
            # Each word of a multi-word entity aliases the entity's OTHER words. "Gemera"
            # in "Koenigsegg Gemera" therefore reaches passages that only say
            # "Koenigsegg" — an alias the corpus itself asserts, rather than one a
            # dictionary guessed.
            for word in words:
                existing = index.entity_aliases.get(word, ())
                merged = tuple(dict.fromkeys((*existing, *(other for other in words if other != word))))
                index.entity_aliases[word] = merged[:6]

    index.average_length = (total_length / len(chunks)) if chunks else 0.0

    # Sibling order is what parent recovery walks; `chunk_index` is authoritative and the
    # load order is not guaranteed to match it for a base whose files were indexed
    # concurrently.
    for key, members in index.families.items():
        members.sort(key=lambda cid: getattr(index.by_id[cid], "chunk_index", 0) or 0)

    return index


def _rebind(index: "PrecisionIndex", chunks: list[ChunkLike]) -> "PrecisionIndex":
    """A view of `index` bound to the caller's chunk objects.

    Shallow: every derived dictionary is shared by reference, so this costs one dict
    comprehension over the chunk list and no re-derivation. What it buys is that the
    `DocumentChunk` instances reaching the response are always the ones this request loaded.
    """
    view = replace(index)
    view.chunks = list(chunks)
    view.by_id = {chunk.id: chunk for chunk in chunks}
    return view


def get_index(
    resource_id: str,
    chunks: list[ChunkLike],
    *,
    terms_of: Callable[[ChunkLike], dict[str, int]],
    extract_entities: Callable[[str], list[str]] | None,
    parent_group_size: int,
    tokenize: Callable[[str], list[str]] | None = None,
) -> PrecisionIndex:
    """The cached index for this resource, rebuilt if the chunk set changed."""
    wanted = fingerprint_of(resource_id, chunks, parent_group_size, extract_entities is not None)
    with _CACHE_LOCK:
        cached = _CACHE.get(resource_id)
        if cached is not None and cached.fingerprint == wanted:
            _CACHE.move_to_end(resource_id)
            return _rebind(cached, chunks)

    # Built OUTSIDE the lock. The build is the expensive part and it is idempotent, so two
    # threads racing on a cold cache do the work twice and store the same thing — which is
    # strictly better than every other request for every other resource queueing behind one
    # 2000-chunk build.
    index = build_index(
        resource_id,
        chunks,
        terms_of=terms_of,
        extract_entities=extract_entities,
        parent_group_size=parent_group_size,
        tokenize=tokenize,
    )

    # Stored WITHOUT the chunk objects. `_rebind` puts a live list back on every read, and
    # what stays in the cache is the derived data that was expensive to compute.
    stored = replace(index)
    stored.chunks = []
    stored.by_id = {}
    with _CACHE_LOCK:
        _CACHE[resource_id] = stored
        _CACHE.move_to_end(resource_id)
        while len(_CACHE) > MAX_CACHED_INDEXES:
            _CACHE.popitem(last=False)
    return index


def invalidate(resource_id: str) -> None:
    """Drop one resource's cached index.

    Called by `rag/service.py` when a document finishes indexing and when a resource is
    deleted or purged. Cheap and idempotent: a resource that was never cached is a no-op,
    which is why the callers do not need to know whether this mode has ever been used.
    """
    with _CACHE_LOCK:
        _CACHE.pop(resource_id, None)


def clear_cache() -> None:
    """Drop every cached index. Used by tests, and by anything that invalidates a corpus."""
    with _CACHE_LOCK:
        _CACHE.clear()
