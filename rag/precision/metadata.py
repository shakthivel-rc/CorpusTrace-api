"""Structure and metadata — stages 3 and 4 of the high-precision pipeline.

WHY THIS DERIVES RATHER THAN READS. The brief asks for `section`, `heading`,
`parent_chunk_id`, `document_type`, `category`, `version` and `entities` per chunk. None of
those columns exist on `document_chunks`, and adding them would mean a migration, a change
to the ingestion path every other mode shares, and — the part that actually decides it — a
knowledge base that only gains the metadata if it is re-uploaded. Every base indexed before
today would have NULL in all of them, which is the same as the mode not working.

So the structure is derived, per resource, from data the row already carries, and cached
(`index.py`). Cost: one pass over the chunks the first time a resource is queried in this
mode. Benefit: the mode works on 100% of existing knowledge bases with no migration, no
re-upload, and — this is the important one — no edit to `ingest_file`, so no existing mode
can possibly regress. If profiling ever shows the derivation is the bottleneck, persisting
these as nullable columns is a purely additive follow-up that this module's shape already
anticipates: `derive_metadata` is the only producer.

Everything here is a pure function. No DB, no session, no network.
"""
from __future__ import annotations

import re

from rag.precision.types import ChunkLike, ChunkMetadata


# A section number opening the chunk: "3.2 ", "4) ", "12. ". The heading itself is then
# read as a WORD RUN rather than by regex — ingestion normalizes whitespace away, so a
# heading and the sentence following it are one line, and a lazy regex terminated by "the
# next capitalised word" truncates every multi-word heading at its second word
# ("3.2 Connection Pooling The database..." -> "Connection"). See `_title_run`.
_LEAD_NUMBER = re.compile(r"^\s*(\d+(?:\.\d+){0,3})[.)]?\s+(?=[A-Z])")
# A run of capitals at the start, followed by ordinary prose: "SAFETY PRECAUTIONS The ...".
_CAPS_HEADING = re.compile(r"^\s*([A-Z][A-Z0-9][A-Z0-9 &/\'()-]{3,68}[A-Z0-9)])(?=\s+[A-Z][a-z])")
# A short Title Case phrase terminated by a colon: "Connection pooling: the pool is ...".
_COLON_HEADING = re.compile(r"^\s*([A-Z][A-Za-z0-9][^.!?:]{2,68}):\s+\S")

MAX_HEADING_WORDS = 10
MAX_HEADING_CHARS = 90

# Lowercase words that legitimately sit INSIDE a heading ("Terms of Service", "Setup and
# Configuration") and so must not end the run.
_HEADING_CONNECTORS = {"of", "and", "or", "for", "in", "on", "to", "with", "by", "from"}
# Determiners END the run wherever they appear after the first word. Trimming only the LAST
# word is not enough: "4.1 Authentication Tokens An API token is issued..." keeps a
# capitalised word AFTER the determiner, so the run ran on to "Authentication Tokens An API".
# A determiner mid-heading is rare; a determiner opening the sentence that FOLLOWS a heading
# is the overwhelmingly common case, and over-extending into prose is the worse error —
# a heading is used as a retrieval signal and as displayed provenance.
_HEADING_BREAKERS = {"the", "a", "an", "this", "that", "these", "those", "it", "its", "there"}
# Words that end a heading when they are the LAST thing in the run.
_HEADING_TRAILERS = _HEADING_BREAKERS | {
    "all", "each", "every", "you", "your", "we", "our", "they", "he", "she", "if", "when",
    "after", "before", "and", "or", "for", "in", "on", "to", "with", "by", "from", "of",
    "use", "using", "do", "does",
}


def _title_run(text: str) -> str | None:
    """The leading run of Title Case words, or None when there is not one.

    Only ever called once a section number has already been matched. A Title Case run on its
    own is not evidence of a heading — "John Smith reported the outage" opens with one — so
    this is deliberately not a shape `detect_heading` recognises by itself.
    """
    run: list[str] = []
    for raw in text.split()[: MAX_HEADING_WORDS + 2]:
        word = raw.strip(",;:\u2013\u2014")
        if not word:
            break
        if run and word.lower() in _HEADING_BREAKERS:
            break
        if word[0].isupper() or (run and word.lower() in _HEADING_CONNECTORS):
            run.append(word)
            if len(run) > MAX_HEADING_WORDS:
                return None  # a whole capitalised sentence is prose, not a heading
        else:
            break
    while run and run[-1].lower() in _HEADING_TRAILERS:
        run.pop()
    if not run:
        return None
    heading = " ".join(run)
    return heading if len(heading) <= MAX_HEADING_CHARS else None

# A version-looking token: "v4.2", "version 4.2", or a bare dotted number like "4.2.1".
#
# The bare-dotted form is the dangerous one and it is why `detect_version` skips the leading
# section number: a chunk opening "3.2 Connection Pooling ..." would otherwise report its own
# SECTION as the document's VERSION, and that version then feeds the metadata filter — so
# asking about "version 4.2" would silently exclude every chunk whose section number was not
# 4.2. Found by the test suite, not by reading the regex.
_VERSION = re.compile(r"\bv(?:ersion)?[\s.]?(\d+(?:\.\d+){0,3})\b|\b(\d+\.\d+(?:\.\d+){0,2})\b", re.IGNORECASE)

# Extension -> the document type a user would name. `modality` on the row is a coarser
# label ("text", "table", "pdf"), so this refines it rather than replacing it.
_TYPE_BY_SUFFIX: dict[str, str] = {
    "pdf": "pdf",
    "docx": "document",
    "doc": "document",
    "txt": "text",
    "md": "markdown",
    "markdown": "markdown",
    "csv": "spreadsheet",
    "xlsx": "spreadsheet",
    "xls": "spreadsheet",
    "json": "data",
    "xml": "data",
    "yaml": "data",
    "yml": "data",
    "html": "web",
    "htm": "web",
    "pptx": "slides",
}

# Advisory only. These feed the metadata SCORE, and gate a filter only when the filter
# still leaves enough candidates — see `apply_filters`. A category is a guess about a
# document from its name and its headings, and a guess must never be the reason a passage
# that answers the question was never ranked.
CATEGORY_HINTS: dict[str, tuple[str, ...]] = {
    "api": ("api", "endpoint", "endpoints", "rest", "graphql", "sdk", "reference"),
    "authentication": ("auth", "authentication", "login", "signin", "oauth", "token", "sso"),
    "authorization": ("authorization", "permission", "permissions", "rbac", "role", "roles"),
    "installation": ("install", "installation", "setup", "getting-started", "onboarding", "deploy", "deployment"),
    "configuration": ("config", "configuration", "settings", "options", "env"),
    "security": ("security", "vulnerability", "hardening", "compliance", "privacy"),
    "troubleshooting": ("troubleshoot", "troubleshooting", "faq", "errors", "known-issues", "diagnostics"),
    # "notes" is deliberately absent: it is one of the commonest words in a filename and
    # almost never means release notes, so "meeting-notes.docx" was being categorised as a
    # release. A hint that fires on the wrong document is worse than a missing category,
    # because a category feeds a filter.
    "release": ("changelog", "release", "releases", "whatsnew", "migration"),
    "policy": ("policy", "policies", "terms", "agreement", "licence", "license", "sla"),
    "manual": ("manual", "guide", "handbook", "tutorial", "howto", "instructions"),
    "specification": ("spec", "specification", "datasheet", "requirements", "standard"),
}

_WORD = re.compile(r"[A-Za-z0-9]+")


def detect_heading(content: str) -> tuple[str | None, str | None]:
    """`(heading, section)` for one chunk, or `(None, None)` when nothing is confident.

    Ingestion normalizes whitespace, so the blank lines and line breaks that mark a heading
    in the source are gone by the time a chunk is stored. What survives is the *shape* of
    the leading text, and only three shapes are reliable enough to act on. Everything else
    returns None — an invented section name is exactly the class of confident-wrong answer
    the evidence view was built to avoid (CLAUDE.md §20).
    """
    if not content:
        return None, None
    head = content[:220]

    match = _LEAD_NUMBER.match(head)
    if match:
        heading = _title_run(head[match.end():])
        if heading:
            return heading, match.group(1)

    match = _CAPS_HEADING.match(head)
    if match:
        return match.group(1).strip().title(), None

    match = _COLON_HEADING.match(head)
    if match:
        candidate = match.group(1).strip()
        # A colon inside ordinary prose is common; a heading is short and mostly
        # capitalised words. Require both.
        words = candidate.split()
        if len(words) <= 8 and sum(1 for word in words if word[:1].isupper()) >= max(1, len(words) // 2):
            return candidate, None

    return None, None


def detect_document_type(source_name: str, modality: str) -> str:
    suffix = source_name.rsplit(".", 1)[-1].lower() if "." in source_name else ""
    return _TYPE_BY_SUFFIX.get(suffix) or (modality or "text")


def detect_version(text: str) -> str | None:
    """The first version-looking token, or None.

    Deliberately conservative: a bare integer is never a version (every document is full of
    them), and only the first match is reported, because a document mentioning six versions
    has no single version and saying otherwise would be a filter that silently excludes.
    """
    if not text:
        return None
    # Skip a leading section number before searching. `detect_heading` reads exactly the same
    # prefix as a SECTION, and one string cannot honestly be both.
    body = text
    lead = _LEAD_NUMBER.match(body)
    if lead:
        body = body[lead.end():]
    match = _VERSION.search(body[:400])
    if not match:
        return None
    return match.group(1) or match.group(2)


def detect_category(source_name: str, heading: str | None) -> str | None:
    """A coarse category from the filename and heading, or None.

    Filename first: a document called `api-reference.pdf` is about the API whatever any one
    of its passages says. The heading only breaks a tie the filename did not answer.
    """
    haystacks = [source_name.lower().replace("_", "-")]
    if heading:
        haystacks.append(heading.lower())
    for haystack in haystacks:
        tokens = set(_WORD.findall(haystack))
        for category, hints in CATEGORY_HINTS.items():
            if tokens.intersection(hints):
                return category
    return None


def parent_key(chunk: ChunkLike, group_size: int) -> str:
    """The identity of the parent window this chunk belongs to.

    Children are the chunks that already exist — NOTHING is re-chunked, because re-chunking
    would change what every other mode retrieves from the same knowledge base. A parent is a
    window of `group_size` consecutive siblings from the same file, so the relationship is
    derived from `file_id` + `chunk_index` and needs no stored pointer.

    Keyed on `file_id` rather than the resource: two documents in one base are two
    documents, and stitching the tail of one onto the head of the next would fabricate
    context that does not exist in either.
    """
    document = chunk.file_id or chunk.source_name or "unknown"
    index = chunk.chunk_index if isinstance(chunk.chunk_index, int) else 0
    return f"{document}:{max(index, 0) // max(group_size, 1)}"


def derive_metadata(chunk: ChunkLike, entities: tuple[str, ...], group_size: int) -> ChunkMetadata:
    heading, section = detect_heading(chunk.content or "")
    return ChunkMetadata(
        document_id=chunk.file_id,
        chunk_id=chunk.id,
        parent_chunk_id=parent_key(chunk, group_size),
        section=section,
        heading=heading,
        page=chunk.page_start,
        document_type=detect_document_type(chunk.source_name or "", chunk.modality or "text"),
        category=detect_category(chunk.source_name or "", heading),
        version=detect_version(chunk.content or ""),
        entities=entities,
    )


# --------------------------------------------------------------------------------------
# Query-side filters
# --------------------------------------------------------------------------------------

_QUERY_VERSION = re.compile(r"\bv(?:ersion)?[\s.]?(\d+(?:\.\d+){1,3})\b|\b(\d+\.\d+(?:\.\d+){0,2})\b", re.IGNORECASE)


def infer_filters(normalized_query: str, query_terms: list[str], available_categories: set[str]) -> dict:
    """Metadata constraints the question itself states, and nothing else.

    Only three are inferred, and each has an unambiguous surface form:

      * `version`  — the question names one ("what changed in 4.2")
      * `document_type` — the question names a format ("in the spreadsheet")
      * `category` — the question's words hit a category's hint list AND that category is
        actually present in this knowledge base

    The last condition matters. Inferring `category=api` in a base with no API document
    filters everything out, and the user gets a refusal about a document set they never
    asked to restrict. A filter is only ever proposed for a category the corpus HAS.
    """
    filters: dict = {}

    match = _QUERY_VERSION.search(normalized_query)
    if match:
        filters["version"] = match.group(1) or match.group(2)

    token_set = set(query_terms)
    for word, doc_type in (
        ("spreadsheet", "spreadsheet"), ("csv", "spreadsheet"), ("excel", "spreadsheet"),
        ("pdf", "pdf"), ("slide", "slides"), ("slides", "slides"),
    ):
        if word in token_set:
            filters["document_type"] = doc_type
            break

    for category, hints in CATEGORY_HINTS.items():
        if category not in available_categories:
            continue
        if token_set.intersection(hints):
            filters["category"] = category
            break

    return filters


def metadata_score(meta: ChunkMetadata, filters: dict, query_terms: set[str]) -> float:
    """A 0..1 signal for how well one chunk's metadata matches the question.

    Separate from the filter on purpose: scoring is always safe (it nudges a ranking),
    filtering is not (it removes a passage from consideration entirely). Everything the
    filter is too blunt to enforce is expressed here instead.
    """
    if not meta:
        return 0.0
    score = 0.0
    if filters.get("version") and meta.version == filters["version"]:
        score += 0.4
    if filters.get("document_type") and meta.document_type == filters["document_type"]:
        score += 0.2
    if filters.get("category") and meta.category == filters["category"]:
        score += 0.2
    if meta.heading and query_terms:
        heading_terms = {word.lower() for word in _WORD.findall(meta.heading)}
        overlap = len(heading_terms.intersection(query_terms))
        if overlap:
            # A heading naming the question's words is the single strongest structural
            # signal available here — it is the document's own statement of what the
            # passage below it is about.
            score += min(0.4, 0.2 * overlap)
    return min(score, 1.0)


def apply_filters(
    candidates: list,
    metadata_by_chunk: dict[str, ChunkMetadata],
    filters: dict,
    min_survivors: int,
) -> tuple[list, dict]:
    """Narrow the candidate pool, but only where narrowing is safe.

    Returns `(candidates, applied)`. Filters are applied CUMULATIVELY, in iteration order:
    each one is tested against the pool its predecessors left, and kept only if it still
    leaves at least `min_survivors`. Cumulative rather than independent because the pool that
    matters is the one that will actually reach the reranker — testing each filter against
    the untouched universe could keep two filters that are individually safe and jointly
    empty it.

    A filter that would take the pool below the floor is dropped and recorded with a value of
    `None`, so the trace distinguishes "inferred and applied" from "inferred and declined".
    Filtering exists to save the reranker work on passages that cannot be the answer — the
    moment it starts deciding what the answer is, it has exceeded its remit, and a question
    with no reliable metadata must not be restricted on a guess.
    """
    applied: dict = {}
    surviving = candidates
    for key, value in filters.items():
        kept = [
            candidate
            for candidate in surviving
            if getattr(metadata_by_chunk.get(candidate.chunk_id), key, None) == value
        ]
        if len(kept) >= min_survivors:
            surviving = kept
            applied[key] = value
        else:
            applied[key] = None  # proposed, not applied — the trace says which
    return surviving, applied
