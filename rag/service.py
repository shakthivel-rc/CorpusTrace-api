from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from itertools import chain
import csv
import io
import json
import logging
import math
import operator
import re
import shutil
import uuid
import zipfile
from pathlib import Path
from typing import Callable, Iterable, Iterator
from xml.etree import ElementTree

from fastapi import UploadFile
from pypdf import PdfReader
from sqlalchemy import func
from sqlalchemy.orm import Session

from core.config import get_settings
from models.file import File
from models.ingestion import IngestionJob, JOB_CANCELLED, JOB_QUEUED, JOB_RUNNING
from models.rag import DocumentChunk, RagGraphEdge, RagGraphEntity
from models.resource import Resource
from rag import chunking
from rag.chunking import IndexingConfig
from services.llm_provider import (
    LlmProviderError,
    embed_texts,
    generate_conversational_reply,
    generate_grounded_answer,
    llm_synthesis_enabled,
    stream_grounded_answer,
)


logger = logging.getLogger("nexarag.rag")

# Ingestion writes user- and document-derived strings into VARCHAR columns. MySQL rejects
# an over-long value outright (error 1406), and with no global exception handler that
# surfaces as an unhandled HTTP 500 on upload — so every such value is bounded at the source.
MAX_ENTITY_NAME_CHARS = 80
MAX_FILENAME_CHARS = 120
MAX_VARCHAR_CHARS = 255

# Ingestion is synchronous and holds everything in memory, so document parsing is capped.
MAX_PDF_PAGES = 500
MAX_SPREADSHEET_ROWS = 300
# OOXML parts are zip members: a few KB can declare gigabytes of decompressed XML.
MAX_ARCHIVE_MEMBER_BYTES = 32 * 1024 * 1024
MAX_ARCHIVE_EXPANSION_RATIO = 200

RAG_MODE_CONTEXTUAL_HYBRID = "contextual_hybrid"
RAG_MODE_FUSION = "rag_fusion"
RAG_MODE_GRAPH = "graph_rag"
RAG_MODE_CORRECTIVE = "corrective"
RAG_MODE_MULTIMODAL = "multi_modal"
RAG_MODE_AGENTIC = "agentic_rag"

SUPPORTED_RAG_MODES = {
    RAG_MODE_CONTEXTUAL_HYBRID,
    RAG_MODE_FUSION,
    RAG_MODE_GRAPH,
    RAG_MODE_CORRECTIVE,
    RAG_MODE_MULTIMODAL,
    RAG_MODE_AGENTIC,
}

STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any",
    "are", "as", "at", "be", "because", "been", "before", "being", "below", "between",
    "both", "but", "by", "can", "did", "do", "does", "doing", "down", "during", "each",
    "few", "for", "from", "further", "had", "has", "have", "having", "he", "her", "here",
    "hers", "herself", "him", "himself", "his", "how", "i", "if", "in", "into", "is", "it",
    "its", "itself", "just", "me", "more", "most", "my", "myself", "no", "nor", "not", "now",
    "of", "off", "on", "once", "only", "or", "other", "our", "ours", "ourselves", "out",
    "over", "own", "same", "she", "should", "so", "some", "such", "than", "that", "the",
    "their", "theirs", "them", "themselves", "then", "there", "these", "they", "this",
    "those", "through", "to", "too", "under", "until", "up", "very", "was", "we", "were",
    "what", "when", "where", "which", "while", "who", "whom", "why", "will", "with", "you",
    "your", "yours", "yourself", "yourselves",
}

TABLE_TERMS = {"table", "row", "column", "csv", "data", "dataset", "spreadsheet", "total", "average", "count"}
IMAGE_TERMS = {"image", "photo", "picture", "screenshot", "diagram", "chart", "figure"}
AUDIO_TERMS = {"audio", "voice", "recording", "transcript", "timestamp", "video"}


@dataclass
class RetrievalResult:
    chunk: DocumentChunk
    score: float
    reason: str


@dataclass
class ExtractedDocument:
    """A document's searchable text, plus where each page starts and ends inside it.

    `text` is already whitespace-normalized, so it is exactly the string the chunker
    slices — which is what makes a chunk's `char_start`/`char_end` meaningful and lets a
    chunk be traced back to a page.

    `page_spans` is empty for every format without pages (and for a PDF that yielded no
    text at all). Empty means "unknown", never "page 1": a wrong page number sends the
    reader to the wrong part of the document with full confidence.

    `title_text` is what `_extract_text` returns, line breaks intact. `_derive_title`
    reads lines, and a spreadsheet's title is its first row — normalizing first would fuse
    every row into one over-long line and silently fall back to the filename, changing both
    the displayed title and the `contextual_content` that retrieval scores against.

    `notice` is set when the "text" above is an explanation rather than the document —
    an unreadable PDF, or a scanned one with no text layer. Those still index (the note is
    honest and searchable), but they index nothing the user actually uploaded, and the
    ingestion report has to say so. Defaulted so existing positional construction is
    unchanged.
    """

    text: str
    page_spans: list[tuple[int, int, int]]  # (page_number, start, end) — end exclusive
    title_text: str
    notice: str | None = None
    # Offsets into `text` where a paragraph began in the original document. Whitespace
    # normalization destroys the blank line that marks one, so it has to be recorded during
    # extraction or it is gone — the paragraph chunking strategy cannot recover it later.
    # Empty means "no paragraph structure known", which is honest for PDFs: pypdf gives no
    # reliable paragraph boundaries, only page ones. Defaulted so positional construction
    # elsewhere is unchanged.
    paragraph_breaks: list[int] = field(default_factory=list)


@dataclass
class RagAnswer:
    answer: str
    citations: list[dict]
    mode: str


@dataclass
class RagStream:
    """A RagAnswer whose text arrives incrementally.

    `mode` and `citations` are known before the first chunk (retrieval already ran), so a
    caller can persist or send metadata immediately; the full answer text is whatever the
    consumer accumulates from `chunks`.
    """

    mode: str
    citations: list[dict]
    chunks: Iterator[str]


def normalize_rag_mode(mode: str | None) -> str:
    if not mode:
        return RAG_MODE_CONTEXTUAL_HYBRID
    normalized = mode.strip().lower().replace("-", "_")
    return normalized if normalized in SUPPORTED_RAG_MODES else RAG_MODE_CONTEXTUAL_HYBRID


GREETINGS = {
    "hi", "hii", "hiii", "hey", "heya", "hello", "helo", "hallo", "yo", "howdy", "greetings",
    "hi there", "hey there", "hello there", "good morning", "good afternoon", "good evening",
    "good day", "morning", "afternoon", "evening", "hola", "namaste", "vanakkam", "sup",
    "whats up", "what s up", "how are you", "how are you doing", "how r u", "test", "testing",
}

THANKS = {"thanks", "thank you", "thanks a lot", "thank you so much", "thx", "ty", "cheers", "nice", "great", "cool", "ok", "okay", "got it"}

FAREWELLS = {"bye", "goodbye", "good bye", "see you", "see ya", "cya", "good night", "gn"}

CAPABILITY_QUESTIONS = {
    "help", "what can you do", "what can you do for me", "who are you", "what are you",
    "how do you work", "how does this work", "what is this", "how to use this", "how do i use this",
}


REALTIME_MARKERS = {"now", "today", "todays", "tonight", "current", "currently", "moment"}
DATETIME_SUBJECTS = {"date", "time", "day", "clock", "oclock"}
LIVE_WORLD_SUBJECTS = {"weather", "temperature", "forecast", "news", "headlines", "stock", "score"}


def _guarded_reply(
    db: Session,
    user_id: str,
    query: str,
    resource_name: str,
    llm_provider: str | None,
    llm_model: str | None,
    deterministic: str,
    situation: str,
    unmatched: list[str] | None = None,
) -> str:
    """Answer a non-document message with the LLM under strict guardrails.

    Falls back to the deterministic wording whenever no provider is configured or the
    call fails, so this path can never leave the user without a reply.
    """
    if not (llm_provider and llm_model):
        return deterministic
    try:
        reply = generate_conversational_reply(
            db, user_id, llm_provider, llm_model, query, resource_name, situation
        )
        if not reply:
            return deterministic
        return f"{reply.rstrip()}{_vocabulary_hint(unmatched)}"
    except Exception as exc:
        # Deliberately broad: saying "hi" now makes an outbound network call, and NOTHING
        # downstream would turn an unexpected provider/transport error into a response —
        # the request would 500 on a greeting. The deterministic wording is always valid.
        logger.warning(
            "guarded conversational reply failed; using the built-in wording",
            extra={"extra_fields": {"provider": llm_provider, "model": llm_model, "error": str(exc)}},
        )
        return deterministic


QUESTION_FILLERS = {"whats", "tell", "know", "please", "give", "show", "say", "right", "u", "s", "its"}


def _out_of_scope_reply(query: str) -> str | None:
    """Reply for questions no document can answer — the live date, time or weather.

    Decided by what is LEFT once stopwords, question filler and time markers are removed:
    the question is out of scope only when its entire remaining subject is a clock/calendar
    or live-world word. That is what separates "what time is it" (subject = {time}) from
    "what is the maximum operating temperature" (subject = {maximum, operating, temperature}),
    which a datasheet answers perfectly. Matching on mere word PRESENCE got both wrong.

    This subject test is also what protects real document questions — "what is the
    publication date of this standard?" keeps {publication, standard} in its subject — so
    no keyword allow-list is needed. An earlier version had one, but since the message is
    attacker-controlled it only served as an escape hatch: appending any listed word
    disabled the check.
    """
    words = re.sub(r"[^a-z\s]", " ", (query or "").lower()).split()
    if not words:
        return None

    subject = set(words) - STOPWORDS - QUESTION_FILLERS - REALTIME_MARKERS
    if not subject:
        return None

    grounding = (
        "I can only answer from the documents you have uploaded, and I have no access to "
        "real-time information."
    )
    if subject <= DATETIME_SUBJECTS:
        return f"I do not know the current date or time — {grounding} Ask me about the contents of your documents instead."
    if subject <= LIVE_WORLD_SUBJECTS:
        return f"I cannot look that up — {grounding} Ask me about the contents of your documents instead."
    return None


def _conversational_reply(query: str, resource_name: str) -> str | None:
    """A friendly reply for greetings and small talk, or None to run normal retrieval.

    Matches only when the WHOLE message is small talk — "hi, what does clause 5 require?"
    must still be treated as a document question.
    """
    if any(character.isdigit() for character in (query or "")):
        # Stripping digits turned "What is this 4.2?" into "what is this", which matched
        # CAPABILITY_QUESTIONS — a real clause reference answered with a canned blurb.
        return None
    normalized = re.sub(r"[^a-z\s]", " ", (query or "").lower())
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if not normalized or len(normalized) > 40:
        return None

    usage_hint = (
        f'Ask me anything about the documents in "{resource_name}" — for example a specific '
        "term, section or requirement, and I will answer with citations back to the source."
    )
    if normalized in GREETINGS:
        return f"Hello! {usage_hint}"
    if normalized in THANKS:
        return f"You're welcome! {usage_hint}"
    if normalized in FAREWELLS:
        return "Goodbye! Come back any time you need something from your documents."
    if normalized in CAPABILITY_QUESTIONS:
        return (
            f'I am NexaRAG. I answer questions using only the documents you have uploaded into '
            f'"{resource_name}", and I cite the exact source chunk for every claim so you can verify it. '
            "I do not answer from outside knowledge. Ask about a term, section or requirement to begin."
        )
    return None


class IngestionCancelled(Exception):
    """Raised out of `ingest_file` when the job was cancelled part-way through it."""


# How often the chunk loop asks whether the job was cancelled. Cancellation cannot
# interrupt a single document's text extraction — pypdf reads the file in one call — so a
# large PDF stays responsive only from this loop onward. 50 chunks is roughly 60 KB of
# text, well under a second, and costs one cheap check per batch.
CANCEL_CHECK_CHUNKS = 50


@dataclass
class IngestedFile:
    """What indexing one document produced, for the job report the user reads."""

    chunks: list[DocumentChunk]
    page_count: int | None
    notice: str | None  # set when the file yielded an explanation instead of content
    # The settings this document was actually indexed with. `effective_strategy` can differ
    # from `config.strategy` — "one chunk per page" on a .txt has no pages to chunk by — and
    # the report says so rather than substituting silently.
    config: IndexingConfig = chunking.DEFAULT_CONFIG
    effective_strategy: str = chunking.DEFAULT_STRATEGY
    embedded_chunks: int = 0
    # Set when embeddings were requested and did not happen. The document is still indexed
    # and still searchable by keyword; this is what stops that being a silent downgrade.
    embedding_error: str | None = None


# An embedding failure is reported to the user, so it is bounded like every other
# ingestion-written string (see MAX_VARCHAR_CHARS above for why that matters).
MAX_EMBEDDING_ERROR_CHARS = 300


def config_for_file(file_record: File) -> IndexingConfig:
    """The indexing settings stored on a document row, as a validated config.

    Reads through `normalize_config` rather than trusting the columns, because a row can
    predate the columns' defaults or have been written by an older revision.
    """
    return chunking.normalize_config(
        {
            "strategy": file_record.chunk_strategy,
            "chunk_size": file_record.chunk_size,
            "overlap": file_record.chunk_overlap,
            "embedding_provider": file_record.embedding_provider,
            "embedding_model": file_record.embedding_model,
        }
    )


def stage_upload(
    db: Session,
    user_id: str,
    resource_name: str,
    uploads: list[UploadFile],
    configs: list[IndexingConfig] | None = None,
) -> tuple[Resource, list[File]]:
    """Persist the uploaded bytes and their `File` rows. No extraction, no chunking.

    This is deliberately the only part of ingestion that stays inside the request, and it
    has to be: an `UploadFile`'s underlying spooled file is closed when the request ends,
    so the bytes cannot be handed to a worker thread — only a path on disk can. Everything
    after this point (parsing, chunking, graph construction) is what actually takes the
    time, and that is what moves to the background.

    Leaves `upload_status` False. The resource exists but is not yet answerable, and the
    job row created alongside it is what says so.

    Returns the file rows in upload order rather than leaving the caller to re-query them:
    `resource.files` is an unordered lazy load, and the batch's `chunk_index` sequence has
    to be reproducible.

    `configs` is positional against `uploads` — the Nth config belongs to the Nth document.
    A short list (or none at all) leaves the remaining documents on the defaults, so a
    caller that sends no configuration gets exactly the behaviour it got before per-document
    settings existed.
    """
    if not uploads:
        raise ValueError("At least one document is required")

    configs = configs or []

    settings = get_settings()
    resource = Resource(
        resource_name=(resource_name.strip() or f"Knowledge base {uuid.uuid4()}")[:MAX_VARCHAR_CHARS],
        resource_type="knowledge_base",
        upload_status=False,
        user_id=user_id,
    )
    db.add(resource)
    db.flush()

    resource_dir = Path(settings.rag_upload_dir) / resource.id
    resource_dir.mkdir(parents=True, exist_ok=True)

    staged: list[File] = []
    for position, upload in enumerate(uploads):
        safe_name = _safe_filename(upload.filename or f"document-{uuid.uuid4()}.txt")
        destination = resource_dir / safe_name
        with destination.open("wb") as buffer:
            shutil.copyfileobj(upload.file, buffer)

        config = configs[position] if position < len(configs) else chunking.DEFAULT_CONFIG
        file_record = File(
            file_name=safe_name,
            # content_type comes straight off the client's multipart header.
            file_type=(upload.content_type or "application/octet-stream")[:MAX_VARCHAR_CHARS],
            file_url=str(destination)[:MAX_VARCHAR_CHARS],
            resource_id=resource.id,
            chunk_strategy=config.strategy,
            chunk_size=config.chunk_size,
            chunk_overlap=config.overlap,
            embedding_provider=config.embedding_provider,
            embedding_model=config.embedding_model,
        )
        db.add(file_record)
        db.flush()
        staged.append(file_record)

    return resource, staged


def ingest_file(
    db: Session,
    resource: Resource,
    file_record: File,
    start_index: int,
    should_cancel: Callable[[], bool] | None = None,
) -> IngestedFile:
    """Index one already-staged document into `DocumentChunk` rows.

    `start_index` continues `chunk_index` across the batch — it is the position within the
    resource, not within the file, and retrieval orders by it.

    `should_cancel` is polled every `CANCEL_CHECK_CHUNKS` chunks and raises
    `IngestionCancelled` when it returns True. Without it a single 500-page PDF would make
    the Cancel button do nothing for the whole of that document.

    The chunking strategy, size and overlap come from the document's own `File` row, so two
    documents in the same knowledge base can be sliced differently. Embedding is opt-in per
    document and happens after chunking, because a failed embedding call must not cost the
    user the lexical index that already succeeded.
    """
    config = config_for_file(file_record)
    raw_bytes = Path(file_record.file_url).read_bytes()
    modality = _detect_modality(file_record.file_name, file_record.file_type)
    document = _extract_document(file_record.file_name, file_record.file_type, raw_bytes, modality)
    title = _derive_title(file_record.file_name, document.title_text)
    page_count = max((page for page, _, _ in document.page_spans), default=None)

    spans = chunking.split_spans(
        document.text, config, document.page_spans, document.paragraph_breaks
    )
    effective_strategy = chunking.describe_effective(
        config, bool(document.page_spans), bool(document.paragraph_breaks)
    )

    chunks: list[DocumentChunk] = []
    for index, (content, char_start, char_end) in enumerate(spans):
        if should_cancel and index and index % CANCEL_CHECK_CHUNKS == 0 and should_cancel():
            raise IngestionCancelled()
        contextual_content = _contextualize(
            resource.resource_name, file_record.file_name, title, modality, index, content
        )
        terms = _term_counts(contextual_content)
        page_start, page_end = _pages_for_span(document.page_spans, char_start, char_end)
        chunk = DocumentChunk(
            resource_id=resource.id,
            file_id=file_record.id,
            chunk_index=start_index + len(chunks),
            source_name=file_record.file_name,
            modality=modality,
            title=title,
            content=content,
            contextual_content=contextual_content,
            terms_json=json.dumps(terms),
            page_start=page_start,
            page_end=page_end,
            char_start=char_start,
            char_end=char_end,
        )
        db.add(chunk)
        chunks.append(chunk)

    db.flush()

    embedded_chunks, embedding_error = 0, None
    if config.embeds and chunks:
        embedded_chunks, embedding_error = _embed_chunks(
            db, resource.user_id, config, chunks, should_cancel
        )
        db.flush()

    return IngestedFile(
        chunks=chunks,
        page_count=page_count,
        notice=document.notice,
        config=config,
        effective_strategy=effective_strategy,
        embedded_chunks=embedded_chunks,
        embedding_error=embedding_error,
    )


def _embed_chunks(
    db: Session,
    user_id: str,
    config: IndexingConfig,
    chunks: list[DocumentChunk],
    should_cancel: Callable[[], bool] | None = None,
) -> tuple[int, str | None]:
    """Attach a vector to each chunk. Returns (how many succeeded, why it stopped).

    **A failure here is reported, not raised.** By the time this runs the chunks are written
    and the document is searchable by keyword; turning a provider's 429 into a failed
    document would throw away work that succeeded and leave the user with nothing. The
    caller records the reason on the job item instead, and `embedded_chunks` vs
    `chunk_count` is what tells the UI this document is keyword-only.

    Batching is done here rather than left to `embed_texts` so cancellation can land between
    batches. Embedding a large PDF is minutes of network round-trips, and a Cancel that only
    took effect at the end of it would be a button that does nothing.

    The vector is computed over `content`, not `contextual_content`. The contextual header
    ("Resource: … / Source file: … / Modality: …") is identical for every chunk of a
    document, so embedding it would pull all of them toward each other and compress exactly
    the differences retrieval needs to see.
    """
    embedded = 0
    for start in range(0, len(chunks), llm_embedding_batch_size()):
        if should_cancel and should_cancel():
            raise IngestionCancelled()
        batch = chunks[start : start + llm_embedding_batch_size()]
        try:
            vectors = embed_texts(
                db, user_id, config.embedding_provider, config.embedding_model,
                [chunk.content for chunk in batch],
            )
        except LlmProviderError as exc:
            logger.warning(
                "embedding failed during ingestion",
                extra={
                    "extra_fields": {
                        "provider": config.embedding_provider,
                        "model": config.embedding_model,
                        "embedded": embedded,
                        "total": len(chunks),
                        "error": str(exc)[:MAX_EMBEDDING_ERROR_CHARS],
                    }
                },
            )
            return embedded, str(exc)[:MAX_EMBEDDING_ERROR_CHARS]

        for chunk, vector in zip(batch, vectors):
            chunk.embedding_json = json.dumps([round(value, 6) for value in vector])
            chunk.embedding_model = (config.embedding_model or "")[:MAX_VARCHAR_CHARS]
            chunk.embedding_dim = len(vector)
            embedded += 1

    return embedded, None


def llm_embedding_batch_size() -> int:
    """Indirection so a test can shrink the batch without reaching into the provider module."""
    from services.llm_provider import EMBEDDING_BATCH_SIZE

    return EMBEDDING_BATCH_SIZE


def upload_resource(
    db: Session,
    user_id: str,
    resource_name: str,
    uploads: list[UploadFile],
    configs: list[IndexingConfig] | None = None,
) -> Resource:
    """Stage and index in one blocking call.

    The API no longer uses this — uploads go through the job worker so they can report
    progress and be cancelled. It stays because it is the same pipeline expressed without
    the bookkeeping, which makes it the honest thing for tests to exercise.
    """
    resource, staged = stage_upload(db, user_id, resource_name, uploads, configs)

    all_chunks: list[DocumentChunk] = []
    for file_record in staged:
        result = ingest_file(db, resource, file_record, len(all_chunks))
        all_chunks.extend(result.chunks)

    db.flush()
    _rebuild_graph(db, resource.id, all_chunks)
    resource.upload_status = True
    db.commit()
    db.refresh(resource)
    return resource


def list_user_resources(db: Session, user_id: str, deleted: bool = False) -> list[Resource]:
    """Active knowledge bases, or the deleted ones awaiting restore or purge.

    The default is unchanged and is what chat and retrieval call, so a soft-deleted base
    disappears from both the moment `deleted_at` is set — no other query needs touching.
    """
    query = db.query(Resource).filter(Resource.user_id == user_id)
    query = query.filter(Resource.deleted_at.isnot(None)) if deleted else query.filter(Resource.deleted_at.is_(None))
    return query.order_by(Resource.created_at.desc()).all()


def get_user_resource(db: Session, user_id: str, resource_id: str) -> Resource | None:
    return (
        db.query(Resource)
        .filter(Resource.id == resource_id, Resource.user_id == user_id, Resource.deleted_at.is_(None))
        .first()
    )


def get_owned_resource(db: Session, user_id: str, resource_id: str) -> Resource | None:
    """Ownership without the deleted filter — for restore and purge, which act on a
    resource precisely because it is already deleted."""
    return (
        db.query(Resource)
        .filter(Resource.id == resource_id, Resource.user_id == user_id)
        .first()
    )


def list_resource_documents(db: Session, user_id: str, resource_id: str) -> list[dict] | None:
    """Every document in a knowledge base with the settings it was indexed under.

    Returns None when the resource is not the caller's — the route turns that into a 404
    rather than a 403, matching how an unknown ingestion job is already handled: telling a
    stranger that an id exists but is not theirs is itself a disclosure.

    Chunk and embedded-chunk counts come from one grouped query rather than a per-file
    lookup. This backs a page that lists every document in a base, and a per-row count is
    the N+1 CLAUDE.md §9 warns about.
    """
    resource = get_user_resource(db, user_id, resource_id)
    if not resource:
        return None

    files = (
        db.query(File)
        .filter(File.resource_id == resource_id, File.deleted_at.is_(None))
        .order_by(File.created_at.asc())
        .all()
    )
    counts = dict(
        db.query(
            DocumentChunk.file_id,
            func.count(DocumentChunk.id),
        )
        .filter(DocumentChunk.resource_id == resource_id)
        .group_by(DocumentChunk.file_id)
        .all()
    )
    embedded = dict(
        db.query(
            DocumentChunk.file_id,
            func.count(DocumentChunk.id),
        )
        .filter(
            DocumentChunk.resource_id == resource_id,
            DocumentChunk.embedding_json.isnot(None),
        )
        .group_by(DocumentChunk.file_id)
        .all()
    )

    return [
        {
            "id": file_record.id,
            "file_name": file_record.file_name,
            "file_type": file_record.file_type,
            "created_at": file_record.created_at.isoformat() if file_record.created_at else None,
            "chunk_count": counts.get(file_record.id, 0),
            "embedded_chunks": embedded.get(file_record.id, 0),
            "config": config_for_file(file_record).to_dict(),
        }
        for file_record in files
    ]


def resource_counts(db: Session, resource_ids: list[str]) -> dict[str, dict[str, int]]:
    """File and chunk counts for a whole page of resources in two queries.

    Grouped rather than counted per resource: the management screen lists every base a
    user owns, and a per-row count is the N+1 that CLAUDE.md §9 warns about.
    """
    counts: dict[str, dict[str, int]] = {rid: {"file_count": 0, "chunk_count": 0} for rid in resource_ids}
    if not resource_ids:
        return counts
    for rid, total in (
        db.query(File.resource_id, func.count(File.id))
        .filter(File.resource_id.in_(resource_ids))
        .group_by(File.resource_id)
        .all()
    ):
        counts[rid]["file_count"] = total
    for rid, total in (
        db.query(DocumentChunk.resource_id, func.count(DocumentChunk.id))
        .filter(DocumentChunk.resource_id.in_(resource_ids))
        .group_by(DocumentChunk.resource_id)
        .all()
    ):
        counts[rid]["chunk_count"] = total
    return counts


def rename_resource(db: Session, user_id: str, resource_id: str, name: str) -> Resource:
    resource = get_user_resource(db, user_id, resource_id)
    if not resource:
        raise LookupError("Resource not found or not accessible")
    cleaned = name.strip()
    if not cleaned:
        raise ValueError("A name is required")
    # Same bound as ingestion: the column is VARCHAR(255) and MySQL raises 1406 rather
    # than truncating, which would be an unhandled 500.
    resource.resource_name = cleaned[:MAX_VARCHAR_CHARS]
    db.flush()
    return resource


def soft_delete_resource(db: Session, user_id: str, resource_id: str) -> Resource:
    resource = get_user_resource(db, user_id, resource_id)
    if not resource:
        raise LookupError("Resource not found or not accessible")
    resource.deleted_at = datetime.now(timezone.utc)
    db.flush()
    return resource


def restore_resource(db: Session, user_id: str, resource_id: str) -> Resource:
    resource = get_owned_resource(db, user_id, resource_id)
    if not resource:
        raise LookupError("Resource not found or not accessible")
    resource.deleted_at = None
    db.flush()
    return resource


def _resource_upload_dir(resource_id: str) -> Path:
    """The upload directory for a resource, proven to sit inside the configured root.

    `shutil.rmtree` is not a function to point at an unvalidated path. `resource_id` comes
    off the wire, so a traversal attempt must not be able to walk the tree upwards — and
    resolving to the root itself would delete every user's documents, which is why that
    case is rejected explicitly rather than relying on the containment check alone.
    """
    root = Path(get_settings().rag_upload_dir).resolve()
    target = (root / resource_id).resolve()
    if target == root or not target.is_relative_to(root):
        raise ValueError("Refusing to remove a path outside the upload directory")
    return target


def purge_resource(db: Session, user_id: str, resource_id: str) -> dict:
    """Permanently remove a resource, everything indexed from it, and its uploaded files.

    Children are deleted before the parent because every one of them carries a foreign key
    to `resources`, and edges before entities for the same reason. The `deleted` counts are
    returned so the caller can tell the user what actually went.
    """
    resource = get_owned_resource(db, user_id, resource_id)
    if not resource:
        raise LookupError("Resource not found or not accessible")

    edges = db.query(RagGraphEdge).filter(RagGraphEdge.resource_id == resource_id).delete(synchronize_session=False)
    entities = (
        db.query(RagGraphEntity).filter(RagGraphEntity.resource_id == resource_id).delete(synchronize_session=False)
    )
    chunks = (
        db.query(DocumentChunk).filter(DocumentChunk.resource_id == resource_id).delete(synchronize_session=False)
    )
    files = db.query(File).filter(File.resource_id == resource_id).delete(synchronize_session=False)

    removed_files = False
    try:
        directory = _resource_upload_dir(resource_id)
        if directory.exists():
            shutil.rmtree(directory)
            removed_files = True
    except (OSError, ValueError) as exc:
        # A leftover directory is wasted disk; a failed request here would leave the rows
        # already deleted and no way for the user to retry. Report it and continue.
        logger.warning(
            "could not remove uploaded files for a purged resource",
            extra={"extra_fields": {"resource_id": resource_id, "error": str(exc)}},
        )

    db.delete(resource)
    db.flush()
    return {
        "resource_id": resource_id,
        "chunks_deleted": chunks,
        "files_deleted": files,
        "graph_entities_deleted": entities,
        "graph_edges_deleted": edges,
        "uploaded_files_removed": removed_files,
    }


def get_user_file(db: Session, user_id: str, file_id: str) -> File | None:
    """The file behind a citation, but only if the caller owns the resource holding it.

    Ownership lives on Resource, not File, so this join is the only thing standing between
    a guessed uuid and someone else's uploaded document.
    """
    return (
        db.query(File)
        .join(Resource, File.resource_id == Resource.id)
        .filter(
            File.id == file_id,
            File.deleted_at.is_(None),
            Resource.user_id == user_id,
            Resource.deleted_at.is_(None),
        )
        .first()
    )


def get_user_chunk(db: Session, user_id: str, chunk_id: str) -> DocumentChunk | None:
    """The chunk behind a citation, scoped to the caller's own resources."""
    return (
        db.query(DocumentChunk)
        .join(Resource, DocumentChunk.resource_id == Resource.id)
        .filter(
            DocumentChunk.id == chunk_id,
            Resource.user_id == user_id,
            Resource.deleted_at.is_(None),
        )
        .first()
    )


def serialize_chunk(chunk: DocumentChunk) -> dict:
    """The full text and position of one retrieved chunk.

    The citation delivered with an answer carries only a short snippet — the evidence view
    fetches this to get the whole passage, which is what the in-page matcher needs to draw
    a highlight that covers the real extent of the quoted text.
    """
    return {
        "chunk_id": chunk.id,
        "resource_id": chunk.resource_id,
        "file_id": chunk.file_id,
        "source_name": chunk.source_name,
        "chunk_index": chunk.chunk_index,
        "modality": chunk.modality,
        "title": chunk.title,
        "content": chunk.content,
        "page_start": chunk.page_start,
        "page_end": chunk.page_end,
        "char_start": chunk.char_start,
        "char_end": chunk.char_end,
    }


@dataclass
class AnswerPlan:
    """The result of retrieval: an answer that is always valid on its own, plus
    everything needed to improve it with LLM synthesis.

    Both the blocking and the streaming entry points run this same planning phase, so
    retrieval, the small-talk gate, the evidence gate and citation building can never
    diverge between them — only the final synthesis step differs.
    """

    mode: str
    answer: str
    citations: list[dict]
    resource_name: str
    evidence: list[dict]
    synthesize: bool


def _terminal_plan(mode: str, answer: str, resource_name: str) -> AnswerPlan:
    """A plan that is already final — no retrieval evidence to synthesize from."""
    return AnswerPlan(
        mode=mode,
        answer=answer,
        citations=[],
        resource_name=resource_name,
        evidence=[],
        synthesize=False,
    )


def _not_ready_note(db: Session, resource: Resource) -> str | None:
    """Why this knowledge base cannot answer yet, or None if it can.

    `upload_status` alone would only let us say "not ready". The job row says whether that
    means still working, failed, or cancelled — three situations with three different next
    actions for the user, and guessing between them is worse than not saying.
    """
    if resource.upload_status:
        return None

    job = (
        db.query(IngestionJob)
        .filter(IngestionJob.resource_id == resource.id)
        .order_by(IngestionJob.created_at.desc())
        .first()
    )
    if job is None:
        return (
            f"“{resource.resource_name}” has not finished indexing yet, so it cannot be "
            "searched. Please upload its documents again."
        )
    if job.status in (JOB_QUEUED, JOB_RUNNING):
        done, total = job.processed_documents, job.total_documents
        progress = f" {done} of {total} documents indexed so far." if total else ""
        return (
            f"“{resource.resource_name}” is still being indexed.{progress} Answering now "
            "would search only part of it, so please wait for indexing to finish and ask again."
        )
    if job.status == JOB_CANCELLED:
        return (
            f"Indexing of “{resource.resource_name}” was cancelled, so there is nothing to "
            "search. Upload the documents again to use it."
        )
    return (
        f"Indexing of “{resource.resource_name}” did not finish: {job.message or 'the job failed'}. "
        "Nothing in it can be searched until it is uploaded again."
    )


def _plan_answer(
    db: Session,
    user_id: str,
    resource_id: str,
    query: str,
    rag_mode: str | None,
    llm_provider: str | None,
    llm_model: str | None,
) -> AnswerPlan:
    resource = get_user_resource(db, user_id, resource_id)
    if not resource:
        raise ValueError("Resource not found or not accessible")
    if bool(llm_provider) != bool(llm_model):
        raise LlmProviderError("Both LLM provider and model must be selected", 400)

    # A base whose indexing has not finished must not answer. Retrieval over the documents
    # that happen to have landed so far returns a confident answer drawn from part of the
    # evidence, and nothing in the reply would reveal that the rest was still being read.
    not_ready = _not_ready_note(db, resource)
    if not_ready:
        return _terminal_plan(normalize_rag_mode(rag_mode), not_ready, resource.resource_name)

    # The off switch is enforced here, not by the client omitting the fields. `llm_provider`
    # and `llm_model` are query parameters, so a UI toggle alone would be a suggestion —
    # anyone who kept sending them would keep getting synthesis, and a user who turned the
    # LLM off to stop their documents reaching a third party would be wrong about that.
    if llm_provider and llm_model and not llm_synthesis_enabled(db, user_id):
        llm_provider = None
        llm_model = None

    mode = normalize_rag_mode(rag_mode)

    # "Hi" is not a document question. Running it through retrieval matches nothing and
    # produces the "not enough information" refusal, which reads as a broken assistant.
    # Answer conversationally instead — and before any LLM call, so it costs no tokens.
    small_talk = _conversational_reply(query, resource.resource_name)
    out_of_scope = None if small_talk else _out_of_scope_reply(query)
    if small_talk or out_of_scope:
        situation = (
            "the user sent a greeting or other small talk"
            if small_talk
            else "the user asked for real-time or general-world information the documents cannot contain"
        )
        answer = _guarded_reply(
            db, user_id, query, resource.resource_name, llm_provider, llm_model,
            deterministic=small_talk or out_of_scope,
            situation=situation,
        )
        return _terminal_plan(mode, answer, resource.resource_name)

    chunks = (
        db.query(DocumentChunk)
        .filter(DocumentChunk.resource_id == resource_id)
        .order_by(DocumentChunk.chunk_index.asc())
        .all()
    )
    if not chunks:
        return _terminal_plan(
            mode,
            "This resource has no indexed document chunks yet. Please upload a document and try again.",
            resource.resource_name,
        )

    # Computed once for the whole question: at most one embedding call per question, shared
    # by whichever mode runs and by both sufficiency gates. Inert when the base has no
    # embedded documents, which is the default and the state of every base uploaded before
    # per-document indexing settings existed.
    semantic = _semantic_retrieval(db, user_id, resource_id, query, chunks)

    if mode == RAG_MODE_FUSION:
        results = _blend_semantic(_rag_fusion(query, chunks), semantic)
        answer = _compose_answer(query, results, mode, resource.resource_name)
    elif mode == RAG_MODE_GRAPH:
        results, graph_notes = _graph_rag(db, resource_id, query, chunks)
        results = _blend_semantic(results, semantic)
        answer = _compose_answer(query, results, mode, resource.resource_name, graph_notes=graph_notes)
    elif mode == RAG_MODE_CORRECTIVE:
        results = _blend_semantic(_corrective_rag(query, chunks), semantic)
        if not _has_sufficient_evidence(query, results, semantic):
            unmatched = _unmatched_terms(query, chunks)
            return _terminal_plan(
                mode,
                _guarded_reply(
                    db, user_id, query, resource.resource_name, llm_provider, llm_model,
                    deterministic=(
                        "I do not have enough information in the selected uploaded documents to answer "
                        f"this question.{_vocabulary_hint(unmatched)} Try another RAG mode, upload more "
                        "source material, or ask a more specific question."
                    ),
                    situation=_retrieval_situation(unmatched),
                    unmatched=unmatched,
                ),
                resource.resource_name,
            )
        answer = _compose_answer(query, results, mode, resource.resource_name)
    elif mode == RAG_MODE_MULTIMODAL:
        results = _blend_semantic(_multimodal_rag(query, chunks), semantic)
        answer = _compose_answer(query, results, mode, resource.resource_name)
    elif mode == RAG_MODE_AGENTIC:
        results, agent_notes = _agentic_rag(db, resource_id, query, chunks)
        results = _blend_semantic(results, semantic)
        answer = _compose_answer(query, results, mode, resource.resource_name, graph_notes=agent_notes)
    else:
        results = _blend_semantic(_contextual_hybrid(query, chunks), semantic)
        answer = _compose_answer(query, results, mode, resource.resource_name)

    # Gate on EVIDENCE, not merely on a non-empty result list. Lexical scoring returns
    # something for almost any query sharing one non-stopword with the corpus, so
    # `if not results` let unanswerable questions ("who is the president of France") reach
    # the permissive grounded prompt and come back decorated with real-looking citations.
    if not _has_sufficient_evidence(query, results, semantic):
        unmatched = _unmatched_terms(query, chunks)
        return _terminal_plan(
            mode,
            _guarded_reply(
                db, user_id, query, resource.resource_name, llm_provider, llm_model,
                deterministic=_compose_answer(query, [], mode, resource.resource_name, unmatched=unmatched),
                situation=_retrieval_situation(unmatched),
                unmatched=unmatched,
            ),
            resource.resource_name,
        )

    synthesize = bool(llm_provider and llm_model and results)
    return AnswerPlan(
        mode=mode,
        answer=answer,
        citations=_citations(results),
        resource_name=resource.resource_name,
        evidence=_llm_evidence(results) if synthesize else [],
        synthesize=synthesize,
    )


MAX_PROVIDER_ERROR_CHARS = 220
_JSON_MESSAGE_RE = re.compile(r'"message"\s*:\s*"((?:[^"\\]|\\.)*)"')


def _readable_provider_error(exc: Exception) -> str:
    """Reduce a provider failure to the one sentence its author wrote for a human.

    Providers report errors as a JSON envelope, and interpolating that verbatim put the
    raw blob into the answer, where GFM autolinked every URL inside it — including one the
    bounded error read had already cut in half, rendering a dead `https://o` link. So take
    the human `message` and drop the machine envelope around it.

    The regex fallback is the load-bearing half: `_error_detail` truncates the body, so the
    JSON that reaches here is usually incomplete and `json.loads` cannot parse it at all.
    """
    text = " ".join(str(exc).split())
    start = text.find("{")
    if start != -1:
        message = None
        try:
            payload = json.loads(text[start:])
        except ValueError:
            payload = None
        if isinstance(payload, dict):
            field = payload.get("error")
            if isinstance(field, dict):
                message = field.get("message")
            elif isinstance(field, str):
                message = field
            message = message or payload.get("message")
        if not isinstance(message, str) or not message.strip():
            match = _JSON_MESSAGE_RE.search(text[start:])
            message = match.group(1) if match else None
        if isinstance(message, str) and message.strip():
            text = f"{text[:start].rstrip()} {message.strip()}".strip()

    if len(text) > MAX_PROVIDER_ERROR_CHARS:
        # Break on whitespace so a truncated URL never becomes a live, broken link.
        text = text[:MAX_PROVIDER_ERROR_CHARS].rsplit(" ", 1)[0] + "…"
    return text


def _synthesis_failed_answer(extractive_answer: str, provider: str, model: str, exc: Exception) -> str:
    """Degrade to the extractive answer that was already computed, and say why.

    A provider failure (bad key, rate limit, outage) must never cost the user the answer
    retrieval had already produced.
    """
    logger.warning(
        "llm synthesis failed; falling back to extractive answer",
        extra={"extra_fields": {"provider": provider, "model": model, "error": str(exc)}},
    )
    return (
        f"{extractive_answer}\n\n"
        f"> **Note:** Could not write an answer with {provider}/{model} — "
        f"{_readable_provider_error(exc)}\n>\n"
        "> The passages above come from your documents by keyword match. "
        "Check the provider configuration in chat settings."
    )


def plan_answer(
    db: Session,
    user_id: str,
    resource_id: str,
    query: str,
    rag_mode: str | None,
    llm_provider: str | None = None,
    llm_model: str | None = None,
) -> AnswerPlan:
    """Run retrieval and stop, before any provider call.

    `answer_question` does both halves in one go, which is all an HTTP handler needs. The
    chat WebSocket needs them apart: citations are fully resolved here, and a socket can
    deliver them the moment they exist rather than holding them until the LLM has finished
    writing. Raises exactly what `answer_question` raises — `ValueError` for a resource the
    caller cannot see, `LlmProviderError` for a provider sent without a model.
    """
    return _plan_answer(db, user_id, resource_id, query, rag_mode, llm_provider, llm_model)


def complete_answer(
    db: Session,
    user_id: str,
    plan: AnswerPlan,
    query: str,
    llm_provider: str | None = None,
    llm_model: str | None = None,
) -> str:
    """The synthesis half: the finished answer text for an already-planned question.

    Never raises on a provider failure — the extractive answer retrieval already produced
    is returned with a note saying why the model did not write one. Losing an answer the
    system had in hand to a rate limit is the failure mode this guards.
    """
    if not plan.synthesize:
        return plan.answer
    try:
        llm_answer = generate_grounded_answer(
            db, user_id, llm_provider, llm_model, query, plan.evidence, plan.mode, plan.resource_name
        )
        # Verbatim model output, nothing appended. Provenance (source, chunk, modality,
        # score, page range) travels as structured data on `plan.citations` — the
        # X-Nexarag-Citations header, the WebSocket `citations` frame and citations_json —
        # and the client renders it as a panel. Re-printing the same fields as prose here
        # made the answer body its own duplicate footer, unstyled and unparseable.
        return llm_answer.strip()
    except LlmProviderError as exc:
        return _synthesis_failed_answer(plan.answer, llm_provider, llm_model, exc)


def answer_question(
    db: Session,
    user_id: str,
    resource_id: str,
    query: str,
    rag_mode: str | None,
    llm_provider: str | None = None,
    llm_model: str | None = None,
) -> RagAnswer:
    plan = plan_answer(db, user_id, resource_id, query, rag_mode, llm_provider, llm_model)
    answer = complete_answer(db, user_id, plan, query, llm_provider, llm_model)
    return RagAnswer(answer=answer, citations=plan.citations, mode=plan.mode)


def stream_answer_question(
    db: Session,
    user_id: str,
    resource_id: str,
    query: str,
    rag_mode: str | None,
    llm_provider: str | None = None,
    llm_model: str | None = None,
) -> RagStream:
    """Same answer as `answer_question`, delivered as text deltas.

    Retrieval still happens up front (it is a local database + scoring pass), so what
    streams is the LLM's synthesis. Without a configured provider there is nothing to
    stream — the extractive answer is already complete — so it is handed back as a single
    chunk and the caller's code path stays identical.
    """
    plan = _plan_answer(db, user_id, resource_id, query, rag_mode, llm_provider, llm_model)
    if not plan.synthesize:
        return RagStream(mode=plan.mode, citations=plan.citations, chunks=iter([plan.answer]))

    try:
        deltas = stream_grounded_answer(
            db, user_id, llm_provider, llm_model, query, plan.evidence, plan.mode, plan.resource_name
        )
        # Pull the first delta here, before the caller has committed to a response body:
        # a bad key or a rate limit surfaces now, while falling back is still seamless.
        first = next(deltas, None)
    except Exception as exc:
        # Deliberately broad. Nothing downstream can turn an unexpected transport or
        # parsing error into a response — the request would 500 and discard the extractive
        # answer retrieval had already produced, which is the one thing this path exists
        # to protect.
        return RagStream(
            mode=plan.mode,
            citations=plan.citations,
            chunks=iter([_synthesis_failed_answer(plan.answer, llm_provider, llm_model, exc)]),
        )

    if first is None:
        return RagStream(
            mode=plan.mode,
            citations=plan.citations,
            chunks=iter(
                [
                    _synthesis_failed_answer(
                        plan.answer,
                        llm_provider,
                        llm_model,
                        LlmProviderError("the provider streamed an empty response", 502),
                    )
                ]
            ),
        )

    def body() -> Iterator[str]:
        # The blocking path emits llm_answer.strip(), so the streamed text has to end up
        # byte-identical: leading whitespace is dropped, and trailing whitespace is held
        # back until more text proves it was interior. Only whitespace is ever deferred,
        # so this costs no perceptible latency.
        #
        # Nothing follows the last delta any more, so whatever is still in `pending` when
        # the loop ends is trailing whitespace and is simply never emitted — which is
        # exactly what .strip() does on the blocking side. Do not "flush" it at the end.
        pending = ""
        started = False
        try:
            for delta in chain([first], deltas):
                if not started:
                    delta = delta.lstrip()
                    if not delta:
                        continue
                    started = True
                body_text = delta.rstrip()
                if body_text:
                    yield pending + body_text
                    pending = delta[len(body_text) :]
                else:
                    pending += delta
        except LlmProviderError as exc:
            # Text is already on the wire, so the extractive fallback is no longer an
            # option — the honest move is to keep what was streamed and explain the cut.
            logger.warning(
                "llm stream interrupted after partial output",
                extra={"extra_fields": {"provider": llm_provider, "model": llm_model, "error": str(exc)}},
            )
            yield (
                f"\n\n> **Note:** this answer was cut short — streaming via "
                f"{llm_provider}/{llm_model} failed ({exc})."
            )
            return

    return RagStream(mode=plan.mode, citations=plan.citations, chunks=body())


def serialize_resource(resource: Resource, chunk_count: int | None = None) -> dict:
    payload = {
        "id": resource.id,
        "resource_id": resource.id,
        "resource_name": resource.resource_name,
        "resource_type": resource.resource_type,
        "upload_status": bool(resource.upload_status),
        "created_at": resource.created_at.isoformat() if resource.created_at else None,
        "updated_at": resource.updated_at.isoformat() if resource.updated_at else None,
    }
    if chunk_count is not None:
        payload["chunk_count"] = chunk_count
    return payload


def _safe_filename(filename: str) -> str:
    """Sanitize an uploaded filename, always preserving the extension.

    The extension is separated FIRST: sanitizing the whole name would turn a non-ASCII
    stem such as "文档.pdf" into "_.pdf" and then strip it to "pdf" — losing the suffix
    that _detect_modality and _extract_text dispatch on, so the file would never be parsed.
    Length is bounded because files.file_name, files.file_url (which embeds this path) and
    document_chunks.source_name are all VARCHAR(255).
    """
    raw_suffix = Path(filename).suffix
    suffix = re.sub(r"[^A-Za-z0-9.]+", "", raw_suffix)[:16]
    if suffix in {"", "."}:
        suffix = ""
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(filename).stem).strip("._")
    if not stem:
        stem = f"document-{uuid.uuid4()}"
    return stem[: MAX_FILENAME_CHARS - len(suffix)] + suffix


def _detect_modality(filename: str, content_type: str) -> str:
    suffix = Path(filename).suffix.lower()
    if content_type.startswith("image/") or suffix in {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}:
        return "image"
    if content_type.startswith("audio/") or content_type.startswith("video/") or suffix in {".mp3", ".wav", ".m4a", ".mp4", ".mov"}:
        return "audio"
    if suffix in {".csv", ".tsv", ".xls", ".xlsx"}:
        return "table"
    if suffix == ".pdf":
        return "pdf"
    return "text"


def _extract_text(filename: str, content_type: str, raw_bytes: bytes, modality: str) -> str:
    suffix = Path(filename).suffix.lower()
    if modality == "table":
        return _extract_table_text(raw_bytes, suffix)
    if modality == "image":
        return f"Image file {filename}. OCR is not configured, so only image metadata is searchable."
    if modality == "audio":
        return f"Audio or video file {filename}. Transcription is not configured, so only media metadata is searchable."

    if suffix == ".pdf":
        text, failure = _extract_pdf_text(raw_bytes)
        if failure:
            return _pdf_unreadable_note(filename, failure)
        if not text.strip():
            return _pdf_no_text_note(filename)
        return _normalize_whitespace(text)

    if suffix == ".docx":
        text = _extract_docx_text(raw_bytes)
        if not text.strip():
            return f"Document file {filename}. No readable text was found in the document body."
        return _normalize_whitespace(text)

    if suffix == ".doc":
        # The legacy binary .doc format is not OOXML and has no parser here.
        return (
            f"Document file {filename}. Legacy .doc parsing is not configured; "
            "re-save as .docx, PDF or TXT for answers."
        )

    text = raw_bytes.decode("utf-8", errors="ignore")
    return _normalize_whitespace(text or f"Uploaded file {filename}. No extractable text was found.")


def _pdf_unreadable_note(filename: str, failure: str) -> str:
    """An unreadable PDF must not be reported as "scanned" — the user would go looking
    for an OCR problem that does not exist."""
    return (
        f"PDF file {filename}. The file could not be read ({failure}). "
        "If it is password-protected, upload an unlocked copy."
    )


def _pdf_no_text_note(filename: str) -> str:
    """Scanned/image-only PDFs carry no text layer and there is no OCR here."""
    return (
        f"PDF file {filename}. No embedded text layer was found (the pages are most likely "
        "scanned images). OCR is not configured, so upload a text-based version for answers."
    )


def _extract_pdf_pages(raw_bytes: bytes) -> tuple[list[tuple[int, str]], str | None]:
    """Extract the embedded text layer from a PDF, one entry per readable page.

    Returns ([(page_number, text), ...], failure_reason). A failure_reason means the file
    itself could not be read (encrypted/corrupt) — distinct from a readable PDF that simply
    has no text layer, which returns ([], None) or a list of empty strings and is reported
    to the user as a scanned document.

    The page NUMBER is carried alongside the text rather than implied by list position: a
    page whose text extraction fails is skipped, which would otherwise shift every later
    page's number by one and send the evidence view to the wrong page.

    Without this, raw PDF bytes were decoded as text and the *file structure*
    (`%PDF-1.4`, `/Annots`, `obj`, compressed streams) was indexed instead of the
    document, so every question scored 0.0 and retrieval returned nothing.
    """
    try:
        reader = PdfReader(io.BytesIO(raw_bytes))
        if reader.is_encrypted:
            # An empty user password is common (owner-password-only PDFs) and legitimate.
            try:
                if not reader.decrypt(""):
                    return [], "the PDF is password-protected"
            except Exception:
                return [], "the PDF is encrypted with an unsupported scheme"
        pages: list[tuple[int, str]] = []
        for page_number, page in enumerate(reader.pages, start=1):
            if page_number > MAX_PDF_PAGES:
                logger.info(
                    "pdf truncated at the page cap",
                    extra={"extra_fields": {"page_cap": MAX_PDF_PAGES, "total_pages": len(reader.pages)}},
                )
                break
            try:
                pages.append((page_number, page.extract_text() or ""))
            except Exception:  # a single malformed page must not lose the whole document
                logger.warning("pdf page text extraction failed", extra={"extra_fields": {"page": page_number}})
        return pages, None
    except Exception as exc:
        # Corrupt/unsupported PDF — degrade to an explanatory note rather than a 500.
        logger.warning("pdf parsing failed", extra={"extra_fields": {"error": str(exc)}})
        return [], "the file is not a readable PDF"


def _extract_pdf_text(raw_bytes: bytes) -> tuple[str, str | None]:
    """The whole text layer as one string — `_extract_pdf_pages` with the pages joined."""
    pages, failure = _extract_pdf_pages(raw_bytes)
    return "\n".join(text for _, text in pages), failure


def _join_pages(pages: list[tuple[int, str]]) -> tuple[str, list[tuple[int, int, int]]]:
    """Join already-normalized page texts with a single space, recording each page's span.

    Joining normalized pages with one space is byte-identical to normalizing the
    newline-joined pages, which is what `_extract_pdf_text` produces — so the offsets
    computed here index the exact string the chunker sees. Blank pages contribute no span:
    they hold no text to point at.
    """
    parts: list[str] = []
    spans: list[tuple[int, int, int]] = []
    cursor = 0
    for page_number, page_text in pages:
        if not page_text:
            continue
        if parts:
            cursor += 1  # the single space `" ".join` will insert before this page
        spans.append((page_number, cursor, cursor + len(page_text)))
        parts.append(page_text)
        cursor += len(page_text)
    return " ".join(parts), spans


def _pages_for_span(
    page_spans: list[tuple[int, int, int]], char_start: int, char_end: int
) -> tuple[int | None, int | None]:
    """The first and last page a character span touches, or (None, None) if unknown.

    A chunk is 1200 characters of a document that was flattened before chunking, so it
    routinely straddles a page break — returning a single page would hide half the evidence.
    """
    touched = [number for number, start, end in page_spans if start < char_end and end > char_start]
    if not touched:
        return None, None
    return min(touched), max(touched)


def _extract_document(filename: str, content_type: str, raw_bytes: bytes, modality: str) -> ExtractedDocument:
    """`_extract_text` plus the page map, for callers that need to locate a chunk later.

    Only PDFs produce spans. Everything else returns the same text `_extract_text` gives,
    normalized so that chunk offsets index it directly — the chunker normalizes internally
    anyway, so the chunks themselves are unchanged either way.
    """
    if modality == "pdf":
        pages, failure = _extract_pdf_pages(raw_bytes)
        if failure:
            note = _pdf_unreadable_note(filename, failure)
            return ExtractedDocument(text=note, page_spans=[], title_text=note, notice=note)
        normalized = [(number, _normalize_whitespace(text)) for number, text in pages]
        if not any(text for _, text in normalized):
            note = _pdf_no_text_note(filename)
            return ExtractedDocument(text=note, page_spans=[], title_text=note, notice=note)
        text, spans = _join_pages(normalized)
        # For a PDF `_extract_text` already returns this same normalized string, so title
        # derivation sees exactly what it always did.
        return ExtractedDocument(text=text, page_spans=spans, title_text=text)

    raw_text = _extract_text(filename, content_type, raw_bytes, modality)
    normalized, breaks = _normalize_with_breaks(raw_text)
    return ExtractedDocument(
        text=normalized,
        page_spans=[],
        title_text=raw_text,
        paragraph_breaks=breaks,
    )


_PARAGRAPH_BREAK = re.compile(r"\n[ \t]*\n\s*")


def _normalize_with_breaks(text: str) -> tuple[str, list[int]]:
    """`_normalize_whitespace`, plus where each paragraph starts in the result.

    Normalization collapses every run of whitespace to one space, so the blank line that
    separated two paragraphs becomes indistinguishable from the space between two words.
    This walks the original once, splitting on blank lines, and records the offset each
    following paragraph lands on in the normalized string.

    The returned text is identical to `_normalize_whitespace(text)` — asserted by a test,
    because two normalizers that drift apart would put every chunk offset in this codebase
    one character out and the evidence view would highlight the wrong words.
    """
    pieces = [_normalize_whitespace(piece) for piece in _PARAGRAPH_BREAK.split(text)]
    kept = [piece for piece in pieces if piece]
    if not kept:
        return "", []

    breaks: list[int] = []
    cursor = 0
    for index, piece in enumerate(kept):
        if index:
            cursor += 1  # the single space the join inserts before this paragraph
            breaks.append(cursor)
        cursor += len(piece)
    return " ".join(kept), breaks


def _local_name(tag: str) -> str:
    """Element name without its namespace, so both Transitional and Strict OOXML parse."""
    return tag.rsplit("}", 1)[-1] if isinstance(tag, str) else ""


def _read_archive_member(archive: zipfile.ZipFile, name: str) -> bytes:
    """Read one OOXML part with zip-bomb guards; b"" when absent, oversized or unreadable.

    Callers must still wrap this in `except Exception` — zipfile raises RuntimeError for
    encrypted members, NotImplementedError for unsupported compression and zlib.error for
    a damaged payload, none of which share a useful base class.
    """
    try:
        info = archive.getinfo(name)
    except KeyError:
        return b""
    if info.file_size > MAX_ARCHIVE_MEMBER_BYTES:
        logger.warning(
            "ooxml part exceeds the size cap",
            extra={"extra_fields": {"part": name, "declared_bytes": info.file_size}},
        )
        return b""
    if info.compress_size and info.file_size / info.compress_size > MAX_ARCHIVE_EXPANSION_RATIO:
        logger.warning(
            "ooxml part rejected as a suspected zip bomb",
            extra={"extra_fields": {"part": name, "ratio": round(info.file_size / info.compress_size)}},
        )
        return b""
    # file_size is attacker-controlled header metadata, so bound the actual read too.
    with archive.open(name) as stream:
        data = stream.read(MAX_ARCHIVE_MEMBER_BYTES + 1)
    if len(data) > MAX_ARCHIVE_MEMBER_BYTES:
        logger.warning("ooxml part exceeded the size cap while reading", extra={"extra_fields": {"part": name}})
        return b""
    return data


def _extract_docx_text(raw_bytes: bytes) -> str:
    """Extract paragraph text from a .docx.

    A .docx is a ZIP of XML parts, so stdlib zipfile + ElementTree is enough — no extra
    dependency. Reads only the main document part (headers, footers and footnotes are
    out of scope). Table cell text is included: cells contain ordinary w:p paragraphs.
    """
    try:
        with zipfile.ZipFile(io.BytesIO(raw_bytes)) as archive:
            data = _read_archive_member(archive, "word/document.xml")
            if not data:
                return ""
            root = ElementTree.fromstring(data)
    except Exception as exc:
        logger.warning("docx parsing failed", extra={"extra_fields": {"error": str(exc)}})
        return ""

    # A paragraph nested inside another (text boxes, mc:AlternateContent) is already
    # covered by its ancestor's recursive walk; emitting it again duplicates the text.
    parents = {child: parent for parent in root.iter() for child in parent}

    def _nested_in_paragraph(node) -> bool:
        current = parents.get(node)
        while current is not None:
            if _local_name(current.tag) == "p":
                return True
            current = parents.get(current)
        return False

    lines = []
    for paragraph in root.iter():
        if _local_name(paragraph.tag) != "p" or _nested_in_paragraph(paragraph):
            continue
        parts = []
        for node in paragraph.iter():
            name = _local_name(node.tag)
            if name == "t":
                parts.append(node.text or "")
            elif name == "tab":
                # Without these, runs either side of a tab/break merge into one
                # non-word ("NameJohn") that no query can ever match.
                parts.append("\t")
            elif name in {"br", "cr"}:
                parts.append("\n")
        text = "".join(parts).strip()
        if text:
            lines.append(text)
    return "\n".join(lines)


def _extract_table_text(raw_bytes: bytes, suffix: str) -> str:
    if suffix == ".xlsx":
        text = _extract_xlsx_text(raw_bytes)
        return _normalize_whitespace(text) if text.strip() else "Spreadsheet uploaded but no cell text was readable."
    if suffix not in {".csv", ".tsv"}:
        # .xls is the legacy OLE2 binary format, not OOXML. Decoding it would index
        # binary garbage as if it were content, which is what broke PDFs before.
        return (
            "Legacy .xls parsing is not configured; re-save the spreadsheet as .xlsx or CSV "
            "for row-level extraction."
        )

    decoded = raw_bytes.decode("utf-8", errors="ignore")
    delimiter = "\t" if suffix == ".tsv" else ","
    rows = []
    try:
        reader = csv.DictReader(io.StringIO(decoded), delimiter=delimiter)
        for row_number, row in enumerate(reader, start=1):
            if row_number > MAX_SPREADSHEET_ROWS:
                break
            values = [f"{key}: {value}" for key, value in row.items() if key and value]
            if values:
                rows.append(f"Row {row_number}: " + "; ".join(values))
    except csv.Error as exc:
        # e.g. a field longer than csv's 128 KB limit — do not let it become a 500.
        logger.warning("csv parsing failed", extra={"extra_fields": {"error": str(exc)}})
        return _normalize_whitespace(decoded)
    return "\n".join(rows) or _normalize_whitespace(decoded)


def _ordered_sheets(archive: zipfile.ZipFile, names: list[str]) -> list[tuple[str, str]]:
    """Worksheet parts as (part_name, display_name) in workbook tab order.

    Falls back to numeric part order — plain sorted() puts sheet10 before sheet2.
    """
    numeric = sorted(
        (name for name in names if re.fullmatch(r"xl/worksheets/sheet\d+\.xml", name)),
        key=lambda name: int(re.search(r"(\d+)", name.rsplit("/", 1)[-1]).group(1)),
    )
    try:
        workbook = _read_archive_member(archive, "xl/workbook.xml")
        rels = _read_archive_member(archive, "xl/_rels/workbook.xml.rels")
        if not workbook or not rels:
            return [(name, "") for name in numeric]
        targets = {
            node.get("Id"): node.get("Target", "")
            for node in ElementTree.fromstring(rels)
            if node.get("Id")
        }
        ordered = []
        for node in ElementTree.fromstring(workbook).iter():
            if _local_name(node.tag) != "sheet":
                continue
            rel_id = next((value for key, value in node.attrib.items() if _local_name(key) == "id"), None)
            target = targets.get(rel_id, "")
            part = target if target.startswith("xl/") else f"xl/{target.lstrip('/')}"
            if part in names:
                ordered.append((part, node.get("name") or ""))
        return ordered or [(name, "") for name in numeric]
    except Exception:
        return [(name, "") for name in numeric]


def _cell_text(cell, shared: list[str]) -> str:
    cell_type = cell.get("t")
    if cell_type == "inlineStr":
        return "".join(node.text or "" for node in cell.iter() if _local_name(node.tag) == "t")
    value_node = next((node for node in cell if _local_name(node.tag) == "v"), None)
    if value_node is None:
        return ""
    raw = (value_node.text or "").strip()
    if not raw:
        return ""
    if cell_type == "s":
        try:
            index = int(raw)
        except ValueError:
            # One malformed index must cost one cell, not the whole workbook.
            return ""
        return shared[index] if 0 <= index < len(shared) else ""
    if cell_type == "b":
        return "TRUE" if raw == "1" else "FALSE"
    if cell_type == "e":  # #REF!, #DIV/0! and friends carry no meaning for retrieval
        return ""
    return raw


def _shared_string_text(item) -> str:
    """Concatenate a shared string's runs, skipping phonetic (rPh) guide text."""
    parts: list[str] = []

    def walk(node) -> None:
        for child in node:
            name = _local_name(child.tag)
            if name == "rPh":
                continue
            if name == "t":
                parts.append(child.text or "")
            else:
                walk(child)

    walk(item)
    return "".join(parts)


def _extract_xlsx_text(raw_bytes: bytes) -> str:
    """Extract cell values from a .xlsx via stdlib zipfile + ElementTree.

    Inline cell values live in the sheet XML; everything else is an index into the
    shared-strings table, so both parts are needed to recover the text.
    """
    try:
        with zipfile.ZipFile(io.BytesIO(raw_bytes)) as archive:
            names = archive.namelist()
            shared: list[str] = []
            shared_data = _read_archive_member(archive, "xl/sharedStrings.xml")
            if shared_data:
                for item in ElementTree.fromstring(shared_data).iter():
                    if _local_name(item.tag) == "si":
                        shared.append(_shared_string_text(item))

            lines: list[str] = []
            rows_seen = 0
            for sheet_part, display_name in _ordered_sheets(archive, names):
                sheet_data = _read_archive_member(archive, sheet_part)
                if not sheet_data:
                    continue
                sheet_root = ElementTree.fromstring(sheet_data)
                if display_name:
                    lines.append(f"Sheet: {display_name}")
                for row in sheet_root.iter():
                    if _local_name(row.tag) != "row":
                        continue
                    # Bound rows EXAMINED, not rows emitted: a sheet of blank rows would
                    # otherwise never trip a cap that only counts output lines.
                    rows_seen += 1
                    if rows_seen > MAX_SPREADSHEET_ROWS:
                        lines.append("Row limit reached; remaining rows were not indexed.")
                        logger.info(
                            "xlsx truncated at the row cap",
                            extra={"extra_fields": {"row_cap": MAX_SPREADSHEET_ROWS}},
                        )
                        return "\n".join(lines)
                    cells = [
                        value
                        for value in (
                            _cell_text(cell, shared)
                            for cell in row.iter()
                            if _local_name(cell.tag) == "c"
                        )
                        if value.strip()
                    ]
                    if cells:
                        # The row's own index, not a running output counter.
                        lines.append(f"Row {row.get('r') or rows_seen}: " + "; ".join(cells))
            return "\n".join(lines)
    except Exception as exc:
        logger.warning("xlsx parsing failed", extra={"extra_fields": {"error": str(exc)}})
        return ""


def _derive_title(filename: str, text: str) -> str:
    for line in text.splitlines():
        cleaned = line.strip()
        if 4 <= len(cleaned) <= 120:
            return cleaned[:120]
    return Path(filename).stem.replace("_", " ").replace("-", " ").title()


def _split_into_chunk_spans(
    text: str, max_chars: int = 1200, overlap: int = 180
) -> Iterable[tuple[str, int, int]]:
    """The original chunker, yielding each chunk with its half-open span in the normalized text.

    The span is the whole point: it is what lets a retrieved chunk be traced back to a page
    of the original document. It is computed after the `.strip()`, so it covers the chunk's
    real characters rather than the whitespace the slice happened to start on.

    **Do not delete this as dead code.** Ingestion no longer calls it — that goes through
    `chunking.split_spans` now — but it is the reference implementation that every chunk in
    every existing knowledge base was produced by, and `tests/unit/test_chunking_strategies.py`
    asserts the new default path is byte-identical to it over randomized documents. Removing
    it removes the only evidence that per-document settings did not silently re-slice
    everything that came before them.
    """
    cleaned = _normalize_whitespace(text)
    if len(cleaned) <= max_chars:
        yield cleaned, 0, len(cleaned)
        return

    start = 0
    while start < len(cleaned):
        end = min(start + max_chars, len(cleaned))
        if end < len(cleaned):
            boundary = cleaned.rfind(". ", start, end)
            if boundary > start + max_chars // 2:
                end = boundary + 1
        raw = cleaned[start:end]
        chunk = raw.strip()
        if chunk:
            offset = start + (len(raw) - len(raw.lstrip()))
            yield chunk, offset, offset + len(chunk)
        if end >= len(cleaned):
            break
        start = max(0, end - overlap)


def _split_into_chunks(text: str, max_chars: int = 1200, overlap: int = 180) -> Iterable[str]:
    for chunk, _, _ in _split_into_chunk_spans(text, max_chars, overlap):
        yield chunk


def _contextualize(resource_name: str, filename: str, title: str, modality: str, index: int, content: str) -> str:
    return (
        f"Resource: {resource_name}\n"
        f"Source file: {filename}\n"
        f"Title: {title}\n"
        f"Modality: {modality}\n"
        f"Chunk: {index + 1}\n"
        f"Content: {content}"
    )


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _tokenize(text: str) -> list[str]:
    return [
        token.lower()
        for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9_-]{1,}", text)
        if token.lower() not in STOPWORDS and len(token) > 1
    ]


def _term_counts(text: str) -> dict[str, int]:
    return dict(Counter(_tokenize(text)))


# Attribute holding the per-chunk scoring memo. Named, rather than inlined, because it is
# an UNMAPPED attribute on a mapped class: SQLAlchemy walks its mapper's columns when it
# flushes, not the instance `__dict__`, so a key it does not know about is inert.
_SCORING_MEMO_ATTR = "_nexarag_scoring_memo"


class _ChunkScoringData:
    """Everything lexical scoring needs from one chunk, derived once instead of per pass.

    Scoring a chunk used to re-parse `terms_json` and re-lowercase the whole passage on
    every call. That is affordable once; the modes call it far more than once. `rag_fusion`
    scores the corpus once per query variant, `corrective` runs hybrid and then possibly
    fusion, and `agentic_rag` runs four of those as "tools" and may add repair passes — so
    a single agentic question re-parsed the same JSON blob roughly 15 times per chunk, and
    that parse was measurably a fifth of all retrieval CPU (CLAUDE.md §9).

    The memo hangs off the chunk instance and is validated against the identity of the
    `terms_json` string it was derived from, so it lives exactly as long as the loaded row
    is unchanged. A session expiry or refresh replaces that string with a new object, the
    identity check fails, and the data is rebuilt — which is why this cannot go stale.
    """

    __slots__ = ("source", "terms", "_length_norm", "_lowered_content", "_lowered_title")

    def __init__(self, chunk: DocumentChunk) -> None:
        self.source = chunk.terms_json
        try:
            # The parsed dict is kept as-is. `Counter(json.loads(...))` copied every term
            # map a second time on the way in, and nothing on the scoring path needs a
            # Counter: it indexes, takes .keys() and sums .values(), all of which a plain
            # dict does. `_load_terms` still hands out a Counter for its callers.
            terms = json.loads(chunk.terms_json)
            self.terms: dict[str, int] = terms if isinstance(terms, dict) else _term_counts(chunk.contextual_content)
        except (TypeError, json.JSONDecodeError):
            self.terms = _term_counts(chunk.contextual_content)
        # Everything below is derived on first use, never in __init__. A chunk sharing no
        # term with the query returns before any of it is needed, and on a real corpus that
        # is most of the corpus — eager lowering in particular would hold a second copy of
        # every passage in memory to serve the few that actually score.
        self._length_norm: float | None = None
        self._lowered_content: str | None = None
        self._lowered_title: str | None = None

    @property
    def length_norm(self) -> float:
        if self._length_norm is None:
            self._length_norm = 1 / math.sqrt(max(sum(self.terms.values()), 1))
        return self._length_norm

    # These take the chunk rather than storing it: a memo that held a reference to the row
    # it hangs off would make every chunk a reference cycle for the GC to collect.
    def lowered_content(self, chunk: DocumentChunk) -> str:
        if self._lowered_content is None:
            self._lowered_content = (chunk.contextual_content or "").lower()
        return self._lowered_content

    def lowered_title(self, chunk: DocumentChunk) -> str:
        if self._lowered_title is None:
            self._lowered_title = (chunk.title or "").lower()
        return self._lowered_title


def _scoring_data(chunk: DocumentChunk) -> _ChunkScoringData:
    memo = getattr(chunk, _SCORING_MEMO_ATTR, None)
    if memo is not None and memo.source is chunk.terms_json:
        return memo
    memo = _ChunkScoringData(chunk)
    setattr(chunk, _SCORING_MEMO_ATTR, memo)
    return memo


def _load_terms(chunk: DocumentChunk) -> Counter:
    return Counter(_scoring_data(chunk).terms)


def _chunk_term_set(chunk: DocumentChunk) -> set[str]:
    """The distinct terms of a chunk.

    Equivalent to `set(_tokenize(chunk.contextual_content))` by construction — ingestion
    stores `terms_json` as `_term_counts(contextual_content)`, and `_ChunkScoringData`
    falls back to tokenizing that same text when the JSON is unreadable — but it reads the
    memo instead of re-tokenizing the passage.
    """
    return set(_scoring_data(chunk).terms)


def _contextual_hybrid(query: str, chunks: list[DocumentChunk], limit: int = 5) -> list[RetrievalResult]:
    query_terms = Counter(_tokenize(query))
    query_lower = query.lower()
    results = []
    for chunk in chunks:
        score = _score_chunk(query, query_terms, chunk, query_lower=query_lower)
        if score > 0:
            results.append(RetrievalResult(chunk=chunk, score=score, reason="hybrid lexical/contextual match"))
    return sorted(results, key=lambda item: item.score, reverse=True)[:limit]


def _score_chunk(
    query: str,
    query_terms: Counter,
    chunk: DocumentChunk,
    modality_boost: float = 1.0,
    query_lower: str | None = None,
) -> float:
    if not query_terms:
        return 0.0
    data = _scoring_data(chunk)
    chunk_terms = data.terms
    overlap = query_terms.keys() & chunk_terms.keys()
    if not overlap:
        return 0.0

    lexical = sum(chunk_terms[term] * query_terms[term] for term in overlap)
    coverage = len(overlap) / max(len(query_terms), 1)
    if query_lower is None:
        query_lower = query.lower()
    exact_bonus = 2.0 if query_lower in data.lowered_content(chunk) else 0.0
    lowered_title = data.lowered_title(chunk)
    title_bonus = 0.5 if lowered_title and any(term in lowered_title for term in overlap) else 0.0
    return ((lexical * data.length_norm) + (coverage * 3.0) + exact_bonus + title_bonus) * modality_boost


# ---------------------------------------------------------------------------
# Semantic retrieval
#
# Lexical scoring cannot bridge a synonym: ask about a "bike" where the document says
# "motorcycle" and `_score_chunk` returns 0.0, which is indistinguishable from the
# document not covering the topic at all. When a document was indexed with an embedding
# model, its chunks carry a vector and this layer scores them by cosine similarity as well.
#
# It is strictly ADDITIVE. Semantic results are fused with whatever the chosen mode
# returned, and the sufficiency gate gains an OR branch rather than a new condition. A
# knowledge base with no embeddings, or a question asked while the embedding provider is
# down, behaves exactly as it did before — that property is what makes this safe to enable
# per document rather than per installation.
# ---------------------------------------------------------------------------


# Cosine floor for treating a passage as evidence on similarity alone. This number is
# model-dependent — there is no universal scale — and it is set conservatively on purpose:
# crossing it lets an answer be built from a passage with no lexical overlap at all, which
# is exactly the case this feature exists for and exactly the case that is hardest to
# verify by eye. Below it, a semantic hit still helps ranking; it just cannot by itself
# license an answer.
# Dot product in C rather than a Python-level generator. An embedding is 768–3072 floats and
# every chunk of the resource is scored against the question, so the naive `sum(a * b for
# a, b in zip(...))` was ~3000 interpreted float operations per chunk — measurably a quarter
# of semantic retrieval on a 500-chunk base. `math.sumprod` landed in Python 3.12 (which is
# what the Dockerfile pins); the fallback keeps this importable on 3.11 rather than turning a
# performance detail into a hard version requirement.
try:
    from math import sumprod as _sumprod
except ImportError:  # pragma: no cover - Python < 3.12
    def _sumprod(left, right):
        if len(left) != len(right):
            raise ValueError("inputs are not the same length")
        return sum(map(operator.mul, left, right))


MIN_SEMANTIC_SIMILARITY = 0.62

# How many semantic candidates enter the fusion. Matches the per-variant depth `_rag_fusion`
# already uses, so the two lists arrive at the fusion with comparable weight.
SEMANTIC_CANDIDATES = 8

RRF_K = 60


@dataclass
class _SemanticRetrieval:
    """The semantic side of one question, or the reason there wasn't one."""

    results: list[RetrievalResult] = field(default_factory=list)
    top_similarity: float = 0.0
    model: str | None = None

    @property
    def active(self) -> bool:
        return bool(self.results)


def _embedded_model_for_resource(db: Session, resource_id: str) -> tuple[str, str] | None:
    """The (provider, model) pair covering the most embedded chunks in this resource.

    A knowledge base can hold documents embedded with different models, and vectors from
    two different models are not comparable — cosine between them is a number with no
    meaning, and it would rank confidently. Rather than embed the question once per model
    (a per-question API call per model), this picks the dominant one; documents embedded
    with anything else still participate lexically, exactly as they would with no
    embeddings at all.
    """
    rows = (
        db.query(File.embedding_provider, File.embedding_model, func.count(DocumentChunk.id))
        .join(DocumentChunk, DocumentChunk.file_id == File.id)
        .filter(
            File.resource_id == resource_id,
            File.embedding_provider.isnot(None),
            File.embedding_model.isnot(None),
            DocumentChunk.embedding_json.isnot(None),
        )
        .group_by(File.embedding_provider, File.embedding_model)
        .order_by(func.count(DocumentChunk.id).desc())
        .all()
    )
    for provider, model, _count in rows:
        if provider and model:
            return provider, model
    return None


def _semantic_retrieval(
    db: Session, user_id: str, resource_id: str, query: str, chunks: list[DocumentChunk]
) -> _SemanticRetrieval:
    """Rank chunks by similarity to the question, or return an inert result.

    Every failure path here is inert rather than raised. The user asked a question about
    their documents; a provider outage, a revoked key or a rate limit should cost them the
    synonym bridging, not the answer. The lexical index is always there underneath.
    """
    pair = _embedded_model_for_resource(db, resource_id)
    if not pair:
        return _SemanticRetrieval()
    provider, model = pair

    try:
        vectors = embed_texts(db, user_id, provider, model, [query])
    except LlmProviderError as exc:
        logger.info(
            "semantic retrieval unavailable, continuing lexically",
            extra={"extra_fields": {"provider": provider, "model": model, "error": str(exc)[:200]}},
        )
        return _SemanticRetrieval()
    if not vectors:
        return _SemanticRetrieval()

    query_vector = vectors[0]
    query_norm = math.sqrt(sum(value * value for value in query_vector))
    if not query_norm:
        return _SemanticRetrieval()

    scored: list[tuple[float, DocumentChunk]] = []
    for chunk in chunks:
        # Comparing across models is the one thing that must not happen silently.
        if not chunk.embedding_json or chunk.embedding_model != model:
            continue
        similarity = _cosine(query_vector, query_norm, chunk)
        if similarity > 0:
            scored.append((similarity, chunk))

    scored.sort(key=lambda item: item[0], reverse=True)
    top = scored[:SEMANTIC_CANDIDATES]
    return _SemanticRetrieval(
        results=[
            RetrievalResult(chunk=chunk, score=similarity, reason=f"semantic similarity {similarity:.2f}")
            for similarity, chunk in top
        ],
        top_similarity=top[0][0] if top else 0.0,
        model=model,
    )


def _cosine(query_vector: list[float], query_norm: float, chunk: DocumentChunk) -> float:
    try:
        chunk_vector = json.loads(chunk.embedding_json)
    except (TypeError, ValueError, json.JSONDecodeError):
        return 0.0
    if not isinstance(chunk_vector, list) or len(chunk_vector) != len(query_vector):
        # A dimension mismatch means these vectors came from different models despite the
        # name matching. Scoring them anyway would produce a plausible number from nothing.
        return 0.0
    try:
        dot = _sumprod(query_vector, chunk_vector)
        norm = math.sqrt(_sumprod(chunk_vector, chunk_vector))
    except (TypeError, ValueError):
        return 0.0
    if not norm:
        return 0.0
    return dot / (query_norm * norm)


def _blend_semantic(
    results: list[RetrievalResult], semantic: _SemanticRetrieval, limit: int = 5
) -> list[RetrievalResult]:
    """Fuse the lexical and semantic rankings by reciprocal rank.

    RRF rather than a weighted sum because the two scores are on incomparable scales — a
    lexical score is an unbounded sum of term products, a cosine is bounded at 1.0 — and
    any fixed weighting between them would be a number chosen to look reasonable rather
    than one that means anything.

    Returns `results` untouched when there is no semantic side, so every existing test of
    every existing mode describes behaviour that is still exactly reachable.
    """
    if not semantic.active:
        return results

    fused: dict[str, float] = defaultdict(float)
    reasons: dict[str, list[str]] = defaultdict(list)
    by_id: dict[str, DocumentChunk] = {}

    for rank, result in enumerate(results, start=1):
        fused[result.chunk.id] += 1 / (RRF_K + rank)
        reasons[result.chunk.id].append(result.reason)
        by_id[result.chunk.id] = result.chunk

    for rank, result in enumerate(semantic.results, start=1):
        fused[result.chunk.id] += 1 / (RRF_K + rank)
        reasons[result.chunk.id].append(result.reason)
        by_id.setdefault(result.chunk.id, result.chunk)

    ordered = sorted(fused.items(), key=lambda item: item[1], reverse=True)[:limit]
    return [
        RetrievalResult(chunk=by_id[chunk_id], score=score, reason="; ".join(reasons[chunk_id][:3]))
        for chunk_id, score in ordered
    ]


def _rag_fusion(query: str, chunks: list[DocumentChunk]) -> list[RetrievalResult]:
    variants = _query_variants(query)
    fused_scores: dict[str, float] = defaultdict(float)
    reasons: dict[str, list[str]] = defaultdict(list)
    chunks_by_id = {chunk.id: chunk for chunk in chunks}

    for variant in variants:
        ranked = _contextual_hybrid(variant, chunks, limit=8)
        for rank, result in enumerate(ranked, start=1):
            fused_scores[result.chunk.id] += 1 / (60 + rank)
            reasons[result.chunk.id].append(f"variant='{variant}' rank={rank}")

    return [
        RetrievalResult(chunks_by_id[chunk_id], score, "; ".join(reasons[chunk_id]))
        for chunk_id, score in sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)[:5]
    ]


def _query_variants(query: str) -> list[str]:
    terms = _tokenize(query)
    unique_terms = list(dict.fromkeys(terms))
    variants = [query]
    if unique_terms:
        variants.append(" ".join(unique_terms))
    if len(unique_terms) > 2:
        variants.append(" ".join(reversed(unique_terms)))
        variants.append(" ".join(unique_terms[:3]))
    return list(dict.fromkeys(variants))


# How far the entity graph is allowed to move a lexical ranking.
#
# The boost used to be an absolute `1.5 + min(weight, 5) * 0.1`, summed once per matched
# entity with no ceiling — which put it on the same numeric scale as an entire lexical
# score while being unbounded. Measured on a 783-chunk motorcycle manual, one question
# matched 38 of 1,207 entities and the accumulated boost came to 18.9x the whole spread
# between the best and eighth-best lexical candidate. The result was not a refinement of
# the ranking but a replacement of it: four of the five citations arrived on boost alone,
# and the passage that actually answered the question was pushed out of the list entirely.
# A signal that can always override the thing it is meant to refine is not a signal.
#
# Two properties fix that, and both are load-bearing:
#
#   * The boost is a FRACTION of the strongest lexical score this query produced, never an
#     absolute number. Scores are corpus- and query-dependent (`_score_chunk` sums term
#     frequencies), so any constant is right for one document and wrong for the next.
#     Because the cap is below 1.0, a chunk carrying no lexical score at all — the "graph
#     entity expansion" case — can be surfaced into the shortlist but can never outrank the
#     best lexical passage. That single inequality is the whole point of the change.
#
#   * Affinity SATURATES instead of accumulating. Matching 38 entities is not 38 times the
#     evidence of matching one: on a document about the Daytona, "daytona" alone appears in
#     18 entity names, so a raw sum mostly measures how often the document says its own
#     subject. `w / (w + SATURATION)` keeps the ordering (more entities still ranks higher)
#     while flattening the tail.
GRAPH_BOOST_MAX_FRACTION = 0.35
GRAPH_BOOST_SATURATION = 3.0


def _graph_rag(db: Session, resource_id: str, query: str, chunks: list[DocumentChunk]) -> tuple[list[RetrievalResult], str]:
    query_terms = set(_tokenize(query))
    entities = db.query(RagGraphEntity).filter(RagGraphEntity.resource_id == resource_id).all()
    affinity: dict[str, float] = defaultdict(float)
    matched_entities = []

    for entity in entities:
        entity_terms = set(_tokenize(entity.name))
        if query_terms.intersection(entity_terms):
            matched_entities.append(entity.name)
            try:
                chunk_refs = json.loads(entity.chunk_refs_json)
            except json.JSONDecodeError:
                chunk_refs = []
            for chunk_id in chunk_refs:
                # Deliberately unitless. It becomes a score below, against the lexical
                # scale this particular query produced — accumulating points here is what
                # made the old boost mean something different on every document.
                affinity[chunk_id] += 1.0 + min(entity.weight, 5) * 0.1

    base_results = _contextual_hybrid(query, chunks, limit=8)
    top_lexical = base_results[0].score if base_results else 0.0
    # No lexical match anywhere means there is no scale to be a fraction of, and any
    # positive score ranks the same as any other. In practice this branch is unreachable —
    # entity names are extracted from chunk text, so an entity token matching the query
    # implies the chunk holding it also scored — but a retrieval path must not depend on
    # that being true for every future extractor.
    ceiling = top_lexical * GRAPH_BOOST_MAX_FRACTION if top_lexical > 0 else GRAPH_BOOST_MAX_FRACTION

    def graph_boost(chunk_id: str) -> float:
        weight = affinity.get(chunk_id, 0.0)
        if weight <= 0:
            return 0.0
        return ceiling * weight / (weight + GRAPH_BOOST_SATURATION)

    chunks_by_id = {chunk.id: chunk for chunk in chunks}
    scores: dict[str, RetrievalResult] = {
        result.chunk.id: RetrievalResult(result.chunk, result.score + graph_boost(result.chunk.id), result.reason)
        for result in base_results
    }
    for chunk_id in affinity:
        if chunk_id in chunks_by_id and chunk_id not in scores:
            scores[chunk_id] = RetrievalResult(chunks_by_id[chunk_id], graph_boost(chunk_id), "graph entity expansion")

    graph_notes = ""
    if matched_entities:
        graph_notes = "Graph signals: " + ", ".join(matched_entities[:8])
    return sorted(scores.values(), key=lambda item: item.score, reverse=True)[:5], graph_notes


def _corrective_rag(query: str, chunks: list[DocumentChunk]) -> list[RetrievalResult]:
    primary = _contextual_hybrid(query, chunks, limit=5)
    if _has_sufficient_evidence(query, primary):
        return primary
    return _rag_fusion(query, chunks)


def _has_sufficient_evidence(
    query: str, results: list[RetrievalResult], semantic: "_SemanticRetrieval | None" = None
) -> bool:
    """Whether the retrieved passages actually support answering the query.

    The score term is `> 0`, not `> 0.2`: scores are NOT on a common scale across modes.
    `_rag_fusion` returns reciprocal-rank fusion scores that top out around 0.07, so a 0.2
    floor rejected every fusion result — which silently made corrective mode's whole
    self-correction retry dead code. A zero score already means "no term overlap at all",
    because `_score_chunk` returns 0.0 in exactly that case.

    `semantic` adds an OR branch, never a new requirement. Term coverage is the wrong test
    for a passage found by similarity — a chunk saying "motorcycle" answers a question about
    a "bike" while covering none of its words, and the lexical branch would refuse it. The
    branch only fires above `MIN_SEMANTIC_SIMILARITY`, so a weak semantic hit cannot license
    an answer on its own. Keeping this additive is what guarantees enabling embeddings on
    one document can never make the system refuse something it used to answer.
    """
    if not results:
        return False
    if semantic and semantic.top_similarity >= MIN_SEMANTIC_SIMILARITY:
        return True
    query_terms = set(_tokenize(query))
    if not query_terms:
        return False
    evidence_terms = set()
    for result in results[:3]:
        evidence_terms.update(_chunk_term_set(result.chunk))
    coverage = len(query_terms.intersection(evidence_terms)) / len(query_terms)
    return coverage >= 0.35 and results[0].score > 0


def _multimodal_rag(query: str, chunks: list[DocumentChunk]) -> list[RetrievalResult]:
    query_terms = set(_tokenize(query))
    # Counter over the SET, so every query term weighs 1 regardless of repetition. That
    # differs from `_contextual_hybrid`, which counts duplicates, and the difference is
    # preserved deliberately — this is hoisted out of the loop, not changed.
    scoring_terms = Counter(query_terms)
    query_lower = query.lower()
    wants_table = bool(query_terms.intersection(TABLE_TERMS))
    wants_image = bool(query_terms.intersection(IMAGE_TERMS))
    wants_audio = bool(query_terms.intersection(AUDIO_TERMS))
    results = []
    for chunk in chunks:
        boost = 1.0
        if wants_table and chunk.modality == "table":
            boost = 1.8
        elif wants_image and chunk.modality == "image":
            boost = 1.8
        elif wants_audio and chunk.modality == "audio":
            boost = 1.8
        elif chunk.modality in {"pdf", "text"}:
            boost = 1.1
        score = _score_chunk(query, scoring_terms, chunk, modality_boost=boost, query_lower=query_lower)
        if score > 0:
            results.append(RetrievalResult(chunk=chunk, score=score, reason=f"multi-modal {chunk.modality} match"))
    return sorted(results, key=lambda item: item.score, reverse=True)[:5]


def _agentic_rag(db: Session, resource_id: str, query: str, chunks: list[DocumentChunk]) -> tuple[list[RetrievalResult], str]:
    tool_plan = _agentic_tool_plan(query)
    tool_runs: list[tuple[str, list[RetrievalResult], float]] = []

    for tool_name, weight in tool_plan:
        if tool_name == RAG_MODE_CONTEXTUAL_HYBRID:
            tool_runs.append((tool_name, _contextual_hybrid(query, chunks, limit=8), weight))
        elif tool_name == RAG_MODE_FUSION:
            tool_runs.append((tool_name, _rag_fusion(query, chunks), weight))
        elif tool_name == RAG_MODE_GRAPH:
            graph_results, _ = _graph_rag(db, resource_id, query, chunks)
            tool_runs.append((tool_name, graph_results, weight))
        elif tool_name == RAG_MODE_CORRECTIVE:
            tool_runs.append((tool_name, _corrective_rag(query, chunks), weight))
        elif tool_name == RAG_MODE_MULTIMODAL:
            tool_runs.append((tool_name, _multimodal_rag(query, chunks), weight))

    merged = _merge_agentic_tool_results(tool_runs)
    if not _has_sufficient_evidence(query, merged):
        for variant in _query_variants(query)[1:]:
            tool_runs.append((f"repair:{variant}", _contextual_hybrid(variant, chunks, limit=5), 0.8))
        merged = _merge_agentic_tool_results(tool_runs)

    grade, coverage = _grade_evidence(query, merged)
    tools_used = ", ".join(tool_name for tool_name, results, _ in tool_runs if results)
    notes = (
        "Agentic plan: decomposed the question, selected retrieval tools, merged tool results, "
        f"and graded evidence. Tools used: {tools_used or 'none'}. "
        f"Evidence grade: {grade} ({coverage:.0%} query coverage)."
    )
    return merged[:7], notes


def _agentic_tool_plan(query: str) -> list[tuple[str, float]]:
    query_terms = set(_tokenize(query))
    has_modality_intent = bool(query_terms.intersection(TABLE_TERMS | IMAGE_TERMS | AUDIO_TERMS))
    has_graph_intent = bool(re.search(r"\b(compare|relationship|related|between|across|impact|depends|connect|connection)\b", query, re.I))

    plan = [
        (RAG_MODE_CONTEXTUAL_HYBRID, 1.0),
        (RAG_MODE_FUSION, 1.25),
        (RAG_MODE_CORRECTIVE, 1.15),
        (RAG_MODE_GRAPH, 1.35 if has_graph_intent else 0.85),
        (RAG_MODE_MULTIMODAL, 1.35 if has_modality_intent else 0.75),
    ]
    return sorted(plan, key=lambda item: item[1], reverse=True)


def _merge_agentic_tool_results(tool_runs: list[tuple[str, list[RetrievalResult], float]]) -> list[RetrievalResult]:
    chunks_by_id: dict[str, DocumentChunk] = {}
    scores: dict[str, float] = defaultdict(float)
    reasons: dict[str, list[str]] = defaultdict(list)

    for tool_name, results, weight in tool_runs:
        max_score = max((result.score for result in results), default=0.0) or 1.0
        for rank, result in enumerate(results, start=1):
            chunk_id = result.chunk.id
            chunks_by_id[chunk_id] = result.chunk
            normalized_score = (result.score / max_score) * weight
            rank_bonus = 1 / (rank + 4)
            scores[chunk_id] += normalized_score + rank_bonus
            reasons[chunk_id].append(f"{tool_name}: {result.reason}")

    merged = [
        RetrievalResult(chunks_by_id[chunk_id], score, "; ".join(reasons[chunk_id]))
        for chunk_id, score in scores.items()
    ]
    return _source_diverse_results(sorted(merged, key=lambda item: item.score, reverse=True), limit=7)


def _source_diverse_results(results: list[RetrievalResult], limit: int) -> list[RetrievalResult]:
    selected: list[RetrievalResult] = []
    source_counts: Counter[str] = Counter()
    pending = results[:]

    while pending and len(selected) < limit:
        pending.sort(
            key=lambda result: result.score - (source_counts[result.chunk.source_name] * 0.15),
            reverse=True,
        )
        result = pending.pop(0)
        selected.append(result)
        source_counts[result.chunk.source_name] += 1
    return selected


def _grade_evidence(query: str, results: list[RetrievalResult]) -> tuple[str, float]:
    query_terms = set(_tokenize(query))
    if not query_terms or not results:
        return "weak", 0.0

    evidence_terms = set()
    for result in results[:5]:
        evidence_terms.update(_chunk_term_set(result.chunk))
    coverage = len(query_terms.intersection(evidence_terms)) / len(query_terms)
    if coverage >= 0.65 and results[0].score >= 1.0:
        return "strong", coverage
    if coverage >= 0.35:
        return "moderate", coverage
    return "weak", coverage


MAX_UNMATCHED_TERMS_REPORTED = 6


def _unmatched_terms(query: str, chunks: list[DocumentChunk]) -> list[str]:
    """The words in the question that appear in no chunk of this resource.

    Retrieval scores by term overlap, so these are precisely the words that contributed
    nothing — the one fact that turns "try rephrasing" from advice into instruction. A
    typo ("value" for "valve") or a synonym the document does not use ("bike" for
    "motorcycle") is invisible to the user otherwise, because a lexical index cannot
    bridge either and never says which word let them down.

    Runs only on the refusal path, where a second pass over the chunk terms is affordable,
    and stops early once every query term has been accounted for.
    """
    wanted = list(dict.fromkeys(_tokenize(query)))
    if not wanted:
        return []
    missing = set(wanted)
    for chunk in chunks:
        missing -= _scoring_data(chunk).terms.keys()
        if not missing:
            return []
    # Report in the order they were asked, so the sentence reads like the question.
    return [term for term in wanted if term in missing][:MAX_UNMATCHED_TERMS_REPORTED]


def _retrieval_situation(unmatched: list[str] | None) -> str:
    """The `situation` handed to the conversational LLM on a retrieval miss.

    The unmatched words go here too, not only into the deterministic wording — otherwise
    configuring a provider would silently downgrade the reply, dropping the one detail
    that lets the user fix their own question.
    """
    base = "retrieval found no sufficiently relevant passage in the documents"
    if not unmatched:
        return base
    return f"{base}; these words from the question appear nowhere in it: {', '.join(unmatched)}"


def _vocabulary_hint(unmatched: list[str] | None) -> str:
    """One sentence naming the dead words, or nothing at all.

    Silent when every word matched: in that case the failure was about how the words were
    distributed, not which were used, and naming them would misdirect the user.

    Appended to the conversational reply in code, not merely fed to the model through
    `_retrieval_situation`. A prompt is a request; this is the one detail that lets a user
    repair their own question, and it was being lost. Asked why "piston ring size" returned
    nothing, a model answered "the term 'piston ring size' isn't covered" — echoing the whole
    query and dropping the fact that only **piston** matched nothing while "ring" and "size"
    both did. That is the difference between "this document has no pistons in it" and "try
    rephrasing". Same reasoning as `sanitize_conversational_reply`: what must reach the reader
    is enforced here, not requested in a system prompt. Restating it after a model that did
    name the word is a trivial redundancy; losing it is not.
    """
    if not unmatched:
        return ""
    words = ", ".join(f"**{term}**" for term in unmatched)
    verb = "does not appear" if len(unmatched) == 1 else "do not appear"
    return f" {words} {verb} anywhere in this document."


def _compose_answer(
    query: str,
    results: list[RetrievalResult],
    mode: str,
    resource_name: str,
    graph_notes: str = "",
    unmatched: list[str] | None = None,
) -> str:
    if not results:
        # Retrieval is lexical term overlap, so "no match" usually means different
        # vocabulary rather than missing content — say so instead of a flat refusal.
        return (
            f'I could not find anything matching that in "{resource_name}".'
            f"{_vocabulary_hint(unmatched)} "
            "Matching is based on the words in your question, so try rephrasing it using terms "
            "that appear in the document, or ask about a specific section or requirement."
        )

    mode_label = mode.replace("_", " ").title()
    lines = [f"Using {mode_label} on \"{resource_name}\", I found the following grounded evidence:"]
    if graph_notes:
        lines.append(graph_notes)

    query_terms = set(_tokenize(query))
    used_sentences: set[str] = set()
    for citation_index, result in enumerate(results[:5], start=1):
        sentence = _best_sentence(result.chunk.content, query_terms)
        if sentence in used_sentences:
            sentence = result.chunk.content[:260].strip()
        used_sentences.add(sentence)
        lines.append(f"- {sentence} [{citation_index}]")

    # No "Sources:" footer. The bullets above ARE the extractive answer and stay; the
    # footer that used to follow them repeated the citation fields the client already
    # receives structurally, so a user with no provider configured saw the raw block on
    # every single answer. The [N] markers still resolve — against the provenance panel.
    return "\n".join(lines)


def _best_sentence(content: str, query_terms: set[str]) -> str:
    sentences = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", content) if sentence.strip()]
    if not sentences:
        return content[:280].strip()
    ranked = []
    for sentence in sentences:
        sentence_terms = set(_tokenize(sentence))
        ranked.append((len(query_terms.intersection(sentence_terms)), len(sentence), sentence))
    ranked.sort(key=lambda item: (item[0], -item[1]), reverse=True)
    selected = ranked[0][2]
    return selected[:500]


def _citations(results: list[RetrievalResult]) -> list[dict]:
    citations = []
    for citation_index, result in enumerate(results[:5], start=1):
        citations.append(
            {
                # The same number `_compose_answer` prints as "[N]" and the client renders
                # as a chip and a provenance row. Derived from one shared `results[:5]`
                # slice so they can never disagree about which source they mean.
                "index": citation_index,
                "chunk_id": result.chunk.id,
                "file_id": result.chunk.file_id,
                "source_name": result.chunk.source_name,
                "chunk_index": result.chunk.chunk_index,
                "modality": result.chunk.modality,
                "title": result.chunk.title,
                "score": round(result.score, 4),
                # None for anything indexed before positions were recorded, and for every
                # format without pages. The evidence view must report that as unknown.
                "page_start": result.chunk.page_start,
                "page_end": result.chunk.page_end,
                "char_start": result.chunk.char_start,
                "char_end": result.chunk.char_end,
                "snippet": result.chunk.content[:300],
            }
        )
    return citations


def _llm_evidence(results: list[RetrievalResult]) -> list[dict]:
    evidence = []
    for result in results[:5]:
        evidence.append(
            {
                "source_name": result.chunk.source_name,
                "chunk_index": result.chunk.chunk_index + 1,
                "modality": result.chunk.modality,
                "content": result.chunk.content[:1800],
            }
        )
    return evidence


# `_sources_block` was removed here. It printed "LLM: {provider}/{model}" and one
# "[N] {source}, chunk {n}, modality={m}, score={s}" line per citation onto the end of
# every synthesized answer — every one of those fields already rides back as structured
# data on `_citations()` (the X-Nexarag-Citations header and citations_json), so the block
# was pure duplication that rendered as unstyled prose inside the chat bubble. The client
# renders the same information as a designed provenance panel. Do not reintroduce it: the
# answer body is verbatim answer text.


def _rebuild_graph(db: Session, resource_id: str, chunks: list[DocumentChunk]) -> None:
    entity_display: dict[str, str] = {}
    chunk_entities: dict[str, list[str]] = {}

    for chunk in chunks:
        entities = _extract_entities(chunk.content)
        chunk_entities[chunk.id] = entities
        for entity in entities:
            entity_display[entity.lower()] = entity

    graph_entities: dict[str, RagGraphEntity] = {}
    # Pass chunk->entities so the helper inverts it to entity->chunks. Passing an
    # already-inverted map here made the returned keys chunk UUIDs, so the
    # entity_display[key] lookup below raised KeyError → an unhandled 500 on every
    # document upload.
    for key, chunk_refs in entity_chunks_by_key(chunk_entities).items():
        entity = RagGraphEntity(
            resource_id=resource_id,
            name=entity_display[key][:MAX_VARCHAR_CHARS],
            entity_type="concept",
            weight=len(chunk_refs),
            chunk_refs_json=json.dumps(sorted(chunk_refs)),
        )
        db.add(entity)
        graph_entities[key] = entity
    db.flush()

    edge_weights: Counter[tuple[str, str]] = Counter()
    for entities in chunk_entities.values():
        keys = sorted({entity.lower() for entity in entities})
        for index, source in enumerate(keys):
            for target in keys[index + 1:index + 4]:
                edge_weights[(source, target)] += 1

    for (source, target), weight in edge_weights.items():
        if source in graph_entities and target in graph_entities:
            db.add(
                RagGraphEdge(
                    resource_id=resource_id,
                    source_entity_id=graph_entities[source].id,
                    target_entity_id=graph_entities[target].id,
                    relationship="co_occurs_with",
                    weight=weight,
                )
            )


def entity_chunks_by_key(chunk_entities: dict[str, list[str]]) -> dict[str, set[str]]:
    grouped: dict[str, set[str]] = defaultdict(set)
    for chunk_id, entities in chunk_entities.items():
        for entity in entities:
            grouped[entity.lower()].add(chunk_id)
    return grouped


def _extract_entities(text: str) -> list[str]:
    candidates = re.findall(r"\b[A-Z][A-Za-z0-9&.-]*(?:\s+[A-Z][A-Za-z0-9&.-]*){0,3}\b", text)
    entities = []
    for candidate in candidates:
        normalized = candidate.strip()
        if len(normalized) < 3 or len(normalized) > MAX_ENTITY_NAME_CHARS:
            # Over-long matches are never real entities. Binary files have no parser here,
            # so their bytes are indexed as text and a run like a hex-encoded UTF-16 string
            # matches this regex as one huge token — which then overflowed
            # rag_graph_entities.name (VARCHAR 255) and 500'd the whole upload.
            continue
        if normalized.lower() in STOPWORDS:
            continue
        if normalized not in entities:
            entities.append(normalized)
        if len(entities) >= 20:
            break
    if entities:
        return entities
    top_terms = [term for term, _ in Counter(_tokenize(text)).most_common(8)]
    return [term.title() for term in top_terms]
