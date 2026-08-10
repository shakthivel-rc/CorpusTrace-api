"""Chunking strategies and the per-document indexing configuration.

Until now every document in every knowledge base was chunked identically: a 1200-character
window with 180 characters of overlap, preferring the last sentence boundary past the
midpoint. That is a reasonable default and a poor universal rule — a contract whose clauses
are one paragraph each and a scanned-to-text manual whose pages are self-contained want
different slicing, and the person who uploaded them knows which is which.

This module holds the strategies, the bounds, and the per-RAG-mode recommendations. It is
deliberately free of any dependency on `rag.service` (the import goes the other way) and of
any database import, so it can be exercised as pure text arithmetic.

**The offset contract is the constraint every strategy is written around.** A chunk's
`char_start`/`char_end` index the whitespace-normalized document text — the same string
`content` is sliced from — and the source-evidence view uses them to name the page a
citation came from. Every strategy here therefore yields `(text, start, end)` into *that*
string. A strategy that produced text it could not locate would take the evidence panel
back to guessing, which CLAUDE.md §20 exists to prevent.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# --------------------------------------------------------------------------------------
# Strategies
# --------------------------------------------------------------------------------------

STRATEGY_CHARACTER = "character"
STRATEGY_SENTENCE = "sentence"
STRATEGY_PARAGRAPH = "paragraph"
STRATEGY_PAGE = "page"

DEFAULT_STRATEGY = STRATEGY_CHARACTER
DEFAULT_CHUNK_SIZE = 1200
DEFAULT_OVERLAP = 180

# Bounds, not preferences. Below ~200 characters a chunk carries too few terms for the
# lexical scorer's coverage component to mean anything; above ~4000 a single chunk starts
# to dominate an answer's context window on the small free-tier models this app targets.
MIN_CHUNK_SIZE = 200
MAX_CHUNK_SIZE = 4000
MIN_OVERLAP = 0
# Overlap at or beyond half the chunk size makes the window advance slower than it grows,
# which is how a large document turns into an unbounded number of chunks.
MAX_OVERLAP_RATIO = 0.5

# These two land in VARCHAR(50) and VARCHAR(255) on `files`. Both arrive from the client,
# and MySQL rejects an over-long value outright (error 1406) — with no global exception
# handler that is an unhandled HTTP 500 on upload, which is precisely the failure CLAUDE.md
# §14 records from the last time ingestion wrote an unbounded string into a VARCHAR.
MAX_PROVIDER_CHARS = 50
MAX_MODEL_CHARS = 255


@dataclass(frozen=True)
class IndexingConfig:
    """How one document should be indexed. Per document, never per knowledge base.

    `embedding_provider`/`embedding_model` are None by default and that default is load
    bearing: indexing with embeddings sends the document's text to a third party and, on
    most providers, costs money. Neither may happen because someone accepted a form they
    did not read.
    """

    strategy: str = DEFAULT_STRATEGY
    chunk_size: int = DEFAULT_CHUNK_SIZE
    overlap: int = DEFAULT_OVERLAP
    embedding_provider: str | None = None
    embedding_model: str | None = None

    @property
    def embeds(self) -> bool:
        return bool(self.embedding_provider and self.embedding_model)

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy,
            "chunk_size": self.chunk_size,
            "overlap": self.overlap,
            "embedding_provider": self.embedding_provider,
            "embedding_model": self.embedding_model,
        }


DEFAULT_CONFIG = IndexingConfig()


def normalize_config(raw: dict | None) -> IndexingConfig:
    """Coerce client input into a config that cannot break the chunker.

    Clamps rather than rejects. This is called on an upload carrying several documents, and
    failing the whole batch because one slider arrived as 1201 would be a worse outcome than
    indexing it at 1201. An unknown strategy falls back to the default — the same
    forgiveness `normalize_rag_mode` already applies to `rag_type`.
    """
    raw = raw or {}

    strategy = str(raw.get("strategy") or DEFAULT_STRATEGY).strip().lower()
    if strategy not in STRATEGIES:
        strategy = DEFAULT_STRATEGY

    chunk_size = _clamp_int(raw.get("chunk_size"), DEFAULT_CHUNK_SIZE, MIN_CHUNK_SIZE, MAX_CHUNK_SIZE)
    max_overlap = int(chunk_size * MAX_OVERLAP_RATIO)
    overlap = _clamp_int(raw.get("overlap"), min(DEFAULT_OVERLAP, max_overlap), MIN_OVERLAP, max_overlap)

    provider = _clean_str(raw.get("embedding_provider"), MAX_PROVIDER_CHARS)
    model = _clean_str(raw.get("embedding_model"), MAX_MODEL_CHARS)
    # Half a configuration is not a configuration. Both or neither, so `embeds` can never be
    # true with nothing to call.
    if not (provider and model):
        provider, model = None, None

    return IndexingConfig(
        strategy=strategy,
        chunk_size=chunk_size,
        overlap=overlap,
        embedding_provider=provider,
        embedding_model=model,
    )


def _clamp_int(value, fallback: int, low: int, high: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return max(low, min(high, fallback))
    return max(low, min(high, number))


def _clean_str(value, limit: int) -> str | None:
    """Trim, bound, or reject. The bound is what keeps a client string out of MySQL 1406.

    A truncated provider or model id simply will not match the registry, so the document
    indexes lexically and the ingestion report names the failure — a far better outcome
    than a 500 that loses the whole upload.
    """
    if not isinstance(value, str):
        return None
    cleaned = value.strip()[:limit]
    return cleaned or None


# --------------------------------------------------------------------------------------
# The strategies themselves
# --------------------------------------------------------------------------------------


def split_spans(
    text: str,
    config: IndexingConfig,
    page_spans: list[tuple[int, int, int]] | None = None,
    paragraph_breaks: list[int] | None = None,
) -> list[tuple[str, int, int]]:
    """Slice already-normalized `text` into chunks, each with its half-open span.

    `page_spans` and `paragraph_breaks` are what the extractor knew about the document's
    structure. When a strategy needs structure the extractor could not supply, it degrades
    to a strategy that does not — it never invents the boundary. `describe_effective` says
    which one actually ran, so the ingestion report can tell the user.
    """
    if not text:
        return []

    size, overlap = config.chunk_size, config.overlap

    if config.strategy == STRATEGY_PAGE and page_spans:
        return _split_by_pages(text, page_spans, size, overlap)

    if config.strategy == STRATEGY_PARAGRAPH:
        units = _paragraph_units(text, paragraph_breaks, page_spans)
        if units:
            return _pack(text, units, size, overlap)
        return _pack(text, _sentence_units(text), size, overlap)

    if config.strategy == STRATEGY_SENTENCE:
        return _pack(text, _sentence_units(text), size, overlap)

    return _split_by_characters(text, size, overlap)


def describe_effective(config: IndexingConfig, has_pages: bool, has_paragraphs: bool) -> str:
    """The strategy that will really run, for the per-document report.

    A user who picked "One chunk per page" for a .txt file is owed the information that it
    was indexed by sentence instead. Silently substituting is how a setting becomes
    decorative.
    """
    if config.strategy == STRATEGY_PAGE and not has_pages:
        return STRATEGY_CHARACTER
    if config.strategy == STRATEGY_PARAGRAPH and not has_paragraphs:
        return STRATEGY_SENTENCE
    return config.strategy


def _split_by_characters(text: str, max_chars: int, overlap: int) -> list[tuple[str, int, int]]:
    """The original chunker, unchanged in behaviour.

    Kept byte-identical at the default size/overlap on purpose: it is what every existing
    chunk in every existing knowledge base was produced by, and the regression tests around
    chunk positions assert against it.
    """
    spans: list[tuple[str, int, int]] = []
    if len(text) <= max_chars:
        return [(text, 0, len(text))]

    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            boundary = text.rfind(". ", start, end)
            if boundary > start + max_chars // 2:
                end = boundary + 1
        raw = text[start:end]
        chunk = raw.strip()
        if chunk:
            offset = start + (len(raw) - len(raw.lstrip()))
            spans.append((chunk, offset, offset + len(chunk)))
        if end >= len(text):
            break
        start = max(0, end - overlap)
    return spans


def _split_by_pages(
    text: str, page_spans: list[tuple[int, int, int]], max_chars: int, overlap: int
) -> list[tuple[str, int, int]]:
    """One chunk per page, sub-splitting any page that exceeds the size budget.

    The spans are exact rather than derived: `page_spans` is built by the PDF extractor as
    it joins the pages, so a page chunk's `char_start` is the page's own first character.
    That makes this the one strategy whose citations can never straddle a page break.
    """
    spans: list[tuple[str, int, int]] = []
    for _number, start, end in page_spans:
        page_text = text[start:end].strip()
        if not page_text:
            continue
        offset = start + (len(text[start:end]) - len(text[start:end].lstrip()))
        if len(page_text) <= max_chars:
            spans.append((page_text, offset, offset + len(page_text)))
            continue
        for chunk, sub_start, sub_end in _split_by_characters(page_text, max_chars, overlap):
            spans.append((chunk, offset + sub_start, offset + sub_end))
    return spans


_SENTENCE_END = re.compile(r"[.!?]+(?=\s|$)")


def _sentence_units(text: str) -> list[tuple[int, int]]:
    """Sentence spans over normalized text.

    Abbreviations ("Fig. 4", "Dr. Chen") split early here. That is a deliberate
    simplification rather than an oversight: the cost is a slightly short chunk, the
    alternative is an abbreviation list that is wrong for every document it was not written
    for, and the packer immediately re-joins short units up to the size budget anyway.
    """
    units: list[tuple[int, int]] = []
    cursor = 0
    for match in _SENTENCE_END.finditer(text):
        end = match.end()
        if end <= cursor:
            continue
        units.append((cursor, end))
        cursor = end
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
    if cursor < len(text):
        units.append((cursor, len(text)))
    return units


def _paragraph_units(
    text: str,
    paragraph_breaks: list[int] | None,
    page_spans: list[tuple[int, int, int]] | None,
) -> list[tuple[int, int]]:
    """Paragraph spans, from whatever boundaries the extractor could actually record.

    Whitespace normalization collapses the blank line that separates two paragraphs, so the
    boundary has to be captured during extraction or not at all — this function cannot
    recover it from `text`. For PDFs only page boundaries survive (pypdf gives no reliable
    paragraph structure), so a PDF paragraph split is a page split refined by the packer.
    Returning [] means "no structure known", and `split_spans` falls back to sentences.
    """
    boundaries: set[int] = set()
    for offset in paragraph_breaks or []:
        if 0 < offset < len(text):
            boundaries.add(offset)
    for _number, start, _end in page_spans or []:
        if 0 < start < len(text):
            boundaries.add(start)
    if not boundaries:
        return []

    units: list[tuple[int, int]] = []
    cursor = 0
    for offset in sorted(boundaries):
        if offset > cursor:
            units.append((cursor, offset))
            cursor = offset
    if cursor < len(text):
        units.append((cursor, len(text)))
    return units


def _pack(
    text: str, units: list[tuple[int, int]], max_chars: int, overlap: int
) -> list[tuple[str, int, int]]:
    """Greedily pack whole units into chunks, overlapping by whole units.

    A unit longer than the budget on its own is handed to the character splitter — refusing
    to split it would emit a chunk of unbounded size, and the point of the size setting is
    that it is respected.

    The `next_index > index` guarantee at the bottom is what makes this terminate. Without
    it, a chunk consisting of exactly one over-budget unit would compute an overlap start
    equal to its own start and pack the same unit forever.
    """
    if not units:
        return []

    spans: list[tuple[str, int, int]] = []
    index = 0
    while index < len(units):
        start = units[index][0]
        end = start
        last = index
        for cursor in range(index, len(units)):
            unit_end = units[cursor][1]
            if cursor > index and unit_end - start > max_chars:
                break
            end = unit_end
            last = cursor

        piece = text[start:end].strip()
        if piece:
            offset = start + (len(text[start:end]) - len(text[start:end].lstrip()))
            if len(piece) > max_chars:
                for chunk, sub_start, sub_end in _split_by_characters(piece, max_chars, overlap):
                    spans.append((chunk, offset + sub_start, offset + sub_end))
            else:
                spans.append((piece, offset, offset + len(piece)))

        if last >= len(units) - 1:
            break

        next_index = last + 1
        if overlap > 0:
            covered = 0
            candidate = next_index
            while candidate > index + 1 and covered < overlap:
                candidate -= 1
                covered = end - units[candidate][0]
            next_index = candidate
        index = max(next_index, index + 1)

    return spans


# --------------------------------------------------------------------------------------
# The option catalogue the UI renders
# --------------------------------------------------------------------------------------

STRATEGIES: dict[str, dict] = {
    STRATEGY_CHARACTER: {
        "label": "Fixed size",
        "summary": "Slices a fixed number of characters, backing up to the last sentence end so a chunk rarely stops mid-sentence.",
        "best_for": "Anything, and the safe answer when you are not sure. This is how every document in this app was indexed before per-document settings existed.",
        "caveat": "Ignores the document's own structure, so a chunk can span the end of one section and the start of the next.",
    },
    STRATEGY_SENTENCE: {
        "label": "By sentence",
        "summary": "Packs whole sentences up to the size budget and never splits one in half.",
        "best_for": "Dense prose where a truncated sentence would change its meaning — policies, contracts, medical or legal text.",
        "caveat": "Chunks come out slightly under the size budget, so a document produces a few more of them.",
    },
    STRATEGY_PARAGRAPH: {
        "label": "By paragraph",
        "summary": "Keeps a paragraph whole where the document's paragraph breaks survived text extraction.",
        "best_for": "Documents whose paragraphs are self-contained: FAQs, release notes, clause-per-paragraph agreements.",
        "caveat": "PDFs lose paragraph breaks during extraction — in a PDF this only knows page boundaries and packs by sentence within a page.",
    },
    STRATEGY_PAGE: {
        "label": "One chunk per page",
        "summary": "Each page becomes one chunk, split further only if a page exceeds the size budget.",
        "best_for": "PDFs where a page is a unit of meaning — forms, datasheets, slide decks, scanned-and-OCRd manuals.",
        "caveat": "PDF only. Any other format falls back to Fixed size, and the ingestion report will say so.",
    },
}

CHUNK_SIZE_PRESETS: list[dict] = [
    {
        "value": 600,
        "label": "Small (600)",
        "hint": "More, tighter chunks. Sharper retrieval on specific facts; more likely to cut an explanation in half.",
    },
    {
        "value": 1200,
        "label": "Medium (1200)",
        "hint": "The long-standing default. Roughly a page of prose.",
    },
    {
        "value": 2000,
        "label": "Large (2000)",
        "hint": "Fewer, richer chunks. Better for questions whose answer is spread over several paragraphs.",
    },
    {
        "value": 3000,
        "label": "Extra large (3000)",
        "hint": "Long-form context. Watch the token limits on free-tier models — a few of these fill a small context window.",
    },
]

OVERLAP_PRESETS: list[dict] = [
    {"value": 0, "label": "None", "hint": "No repeated text. An answer that straddles a boundary can be missed entirely."},
    {"value": 180, "label": "Standard (180)", "hint": "The long-standing default. Enough to carry a sentence across a boundary."},
    {"value": 300, "label": "Generous (300)", "hint": "Safer across boundaries, at the cost of more duplicated text in the index."},
]


# Per-RAG-mode guidance. Every entry below is derived from what the retrieval code
# mechanically does, not from general RAG folklore — and where a mode is genuinely
# indifferent to chunk size, it says so instead of inventing a preference.
RAG_MODE_RECOMMENDATIONS: dict[str, dict] = {
    "contextual_hybrid": {
        "label": "Contextual Hybrid",
        "strategy": STRATEGY_CHARACTER,
        "chunk_size": 1200,
        "overlap": 180,
        "why": (
            "Scores each chunk once, and a third of that score is how many of your question's "
            "distinct terms appear in it. A medium chunk holds enough vocabulary to score well "
            "without diluting the length normalisation."
        ),
    },
    "rag_fusion": {
        "label": "RAG-Fusion",
        "strategy": STRATEGY_SENTENCE,
        "chunk_size": 600,
        "overlap": 180,
        "why": (
            "Runs several rewordings of your question and fuses the rankings. Fusion can only "
            "discriminate between chunks it can rank separately, so smaller chunks give it more "
            "to work with."
        ),
    },
    "graph_rag": {
        "label": "GraphRAG",
        "strategy": STRATEGY_SENTENCE,
        "chunk_size": 600,
        "overlap": 180,
        # Justified by what retrieval reads, not by what ingestion writes. This string used
        # to cite entity CO-OCCURRENCE, which is computed into `rag_graph_edges` — a table
        # no query-time code has ever read. Recommending a chunk size on the strength of a
        # mechanism that never runs is advice a user cannot act on.
        "why": (
            "Boosts passages whose extracted entity names share a word with your question, "
            "and each entity points back at the chunks it was found in. Smaller chunks make "
            "that pointer more precise, so the boost lands on the passage that actually "
            "discusses the entity rather than on everything around it."
        ),
    },
    "corrective": {
        "label": "Corrective",
        "strategy": STRATEGY_CHARACTER,
        "chunk_size": 2000,
        "overlap": 300,
        "why": (
            "The only mode that refuses to answer on weak evidence, and the test it applies is "
            "how much of your question the top chunks cover. Larger chunks cover more of it, so "
            "they reduce refusals on questions the document genuinely does answer."
        ),
    },
    "multi_modal": {
        "label": "Multi-Modal",
        "strategy": STRATEGY_CHARACTER,
        "chunk_size": 1200,
        "overlap": 180,
        "why": (
            "Boosts chunks whose modality matches your question — a table for 'how many', an "
            "image caption for 'what does the diagram show'. That is independent of chunk size, "
            "so there is no mechanical reason to move off the default."
        ),
    },
    "agentic_rag": {
        "label": "Agentic RAG",
        "strategy": STRATEGY_CHARACTER,
        "chunk_size": 1200,
        "overlap": 180,
        "why": (
            "Runs the other strategies as tools and merges their results, deliberately spreading "
            "the citations across source files. It inherits whatever the others prefer, so the "
            "balanced default serves it best."
        ),
    },
}


def recommendation_for(rag_mode: str) -> dict:
    return RAG_MODE_RECOMMENDATIONS.get(rag_mode, RAG_MODE_RECOMMENDATIONS["contextual_hybrid"])
