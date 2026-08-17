"""The background document-indexing worker.

Indexing used to happen inside the upload request. That made a large upload a request the
browser held open for minutes, gave the user a spinner and no information, and offered no
way to stop it. This module moves the work behind a job row.

**Why an in-process worker and not a queue.** This repository has no Celery, no Redis and
no broker of any kind, and CLAUDE.md is explicit that the queue described in
SYSTEM_DOCUMENTATION.md does not exist. Adding one to ship a progress bar would be a new
piece of infrastructure to run, deploy and monitor. Instead the jobs table *is* the queue:
a poll loop in each API process claims work with a conditional UPDATE, exactly the pattern
the LLM catalog refresh loop already uses for its periodic work. The honest limits of that
choice are stated where they bite — see `reclaim_stranded_jobs`.

**Why every transition commits.** The progress a user watches is read by a *different*
request, on a different connection. An uncommitted update is invisible to it, so a worker
that batched its writes would show a frozen bar and then jump to 100%.
"""
from __future__ import annotations

import json
import logging
import os
import socket
from datetime import datetime, timezone
from pathlib import Path

from fastapi import UploadFile
from sqlalchemy import func
from sqlalchemy.orm import Session, selectinload

from models.file import File
from models.ingestion import (
    ITEM_CANCELLED,
    ITEM_FAILED,
    ITEM_PARTIAL,
    ITEM_PENDING,
    ITEM_RUNNING,
    ITEM_SUCCEEDED,
    JOB_CANCELLED,
    JOB_FAILED,
    JOB_QUEUED,
    JOB_RUNNING,
    JOB_SUCCEEDED,
    STAGE_DONE,
    STAGE_INDEXING,
    STAGE_LINKING,
    STAGE_QUEUED,
    STAGE_READING,
    TERMINAL_JOB_STATUSES,
    IngestionJob,
    IngestionJobItem,
)
from models.rag import DocumentChunk
from models.resource import Resource
from rag import chunking
from rag.chunking import IndexingConfig
from rag.service import (
    IngestedFile,
    IngestionCancelled,
    MAX_VARCHAR_CHARS,
    _rebuild_graph,
    config_for_file,
    get_owned_resource,
    ingest_file,
    purge_resource,
    stage_upload,
)
from services.activity_log import log_activity


logger = logging.getLogger("corpustrace.ingestion")

# Identifies which process claimed a job. Two uvicorn workers poll the same table, so this
# is what distinguishes "someone else is working on it" from "nobody is".
WORKER_ID = f"{socket.gethostname()}:{os.getpid()}"

# How long the loop sleeps when it finds nothing. Short enough that an upload starts
# indexing immediately from the user's point of view; long enough to be one trivial
# indexed query per second per process at idle.
POLL_INTERVAL_SECONDS = 1.0


def _now() -> datetime:
    return datetime.now(timezone.utc)


# --------------------------------------------------------------------------------------
# Enqueue
# --------------------------------------------------------------------------------------


def enqueue_ingestion(
    db: Session,
    user_id: str,
    resource_name: str,
    uploads: list[UploadFile],
    configs: list[IndexingConfig] | None = None,
) -> IngestionJob:
    """Save the uploaded bytes and queue the indexing. Does not commit — the caller owns
    the transaction, per the layering rule in CLAUDE.md §3.

    The bytes must be written here, in the request: an `UploadFile` is backed by a spooled
    temporary file that FastAPI closes when the response is sent, so a worker thread could
    never read it. What moves to the background is everything after that — parsing,
    chunking and graph construction, which is where the time actually goes.
    """
    resource, staged = stage_upload(db, user_id, resource_name, uploads, configs)

    job = IngestionJob(
        resource_id=resource.id,
        resource_name=resource.resource_name,
        user_id=user_id,
        status=JOB_QUEUED,
        stage=STAGE_QUEUED,
        total_documents=len(staged),
        total_bytes=_total_bytes(staged),
    )
    db.add(job)
    db.flush()

    for position, file_record in enumerate(staged):
        db.add(
            IngestionJobItem(
                job_id=job.id,
                file_id=file_record.id,
                position=position,
                file_name=file_record.file_name,
                file_type=file_record.file_type,
                byte_size=_file_size(file_record),
                status=ITEM_PENDING,
                stage=STAGE_QUEUED,
                # Snapshotted here rather than joined at read time: cancelling purges the
                # resource and its `files` rows, and the record of what was attempted has
                # to survive that.
                config_json=json.dumps(config_for_file(file_record).to_dict()),
            )
        )
    db.flush()
    return job


def _file_size(file_record: File) -> int:
    try:
        return Path(file_record.file_url).stat().st_size
    except OSError:
        # A size is a nicety on a progress display; failing an upload over one would not be.
        return 0


def _total_bytes(files: list[File]) -> int:
    return sum(_file_size(f) for f in files)


# --------------------------------------------------------------------------------------
# Claiming
# --------------------------------------------------------------------------------------


def claim_next_job(db: Session, worker_id: str = WORKER_ID) -> IngestionJob | None:
    """Take ownership of the oldest queued job, or return None.

    The UPDATE carries `status == queued` in its WHERE clause and the claim is confirmed by
    the row count, so two processes racing for the same job cannot both win — the second
    one's UPDATE matches zero rows. Doing this as SELECT-then-UPDATE would leave exactly
    that race, and the symptom would be a document indexed twice.
    """
    candidate = (
        db.query(IngestionJob.id)
        .filter(IngestionJob.status == JOB_QUEUED)
        .order_by(IngestionJob.created_at.asc())
        .first()
    )
    if candidate is None:
        return None

    claimed = (
        db.query(IngestionJob)
        .filter(IngestionJob.id == candidate.id, IngestionJob.status == JOB_QUEUED)
        .update(
            {
                IngestionJob.status: JOB_RUNNING,
                IngestionJob.stage: STAGE_READING,
                IngestionJob.claimed_by: worker_id,
                IngestionJob.started_at: _now(),
            },
            synchronize_session=False,
        )
    )
    db.commit()
    if not claimed:
        return None
    return db.query(IngestionJob).filter(IngestionJob.id == candidate.id).first()


def reclaim_stranded_jobs(db: Session) -> int:
    """Fail any job left `running` by a process that died, at startup.

    An in-process worker cannot survive a restart, so a `running` row after a fresh boot
    means the work stopped mid-way and nothing will resume it. Marking it failed is the
    only honest option — leaving it `running` shows the user a progress bar that will never
    move again.

    The caveat this cannot handle: with more than one API process, a restart of one marks
    jobs another is actively running as failed. That is the price of not having a broker,
    and it is recorded here rather than hidden — the resource is left intact and the job
    says to upload again.
    """
    stranded = (
        db.query(IngestionJob)
        .filter(IngestionJob.status == JOB_RUNNING)
        .update(
            {
                IngestionJob.status: JOB_FAILED,
                IngestionJob.finished_at: _now(),
                IngestionJob.message: "Indexing stopped when the server restarted. Please upload again.",
            },
            synchronize_session=False,
        )
    )
    if stranded:
        db.query(IngestionJobItem).filter(
            IngestionJobItem.status == ITEM_RUNNING
        ).update(
            {IngestionJobItem.status: ITEM_FAILED, IngestionJobItem.detail: "Interrupted by a server restart."},
            synchronize_session=False,
        )
        logger.warning(
            "failed ingestion jobs stranded by a restart",
            extra={"extra_fields": {"count": stranded, "worker": WORKER_ID}},
        )
    db.commit()
    return stranded


# --------------------------------------------------------------------------------------
# Running
# --------------------------------------------------------------------------------------


def _cancel_requested(db: Session, job_id: str) -> bool:
    """Re-read the flag from the database, bypassing the identity map.

    The cancel was written by another session on another connection, so the worker's copy
    of the row is stale by construction — `job.cancel_requested` would be whatever it was
    when this session loaded it, which is always False.
    """
    value = (
        db.query(IngestionJob.cancel_requested)
        .filter(IngestionJob.id == job_id)
        .scalar()
    )
    return bool(value)


def run_job(db: Session, job: IngestionJob) -> IngestionJob:
    """Index every document in a claimed job, writing progress as it goes.

    Never raises for an ingestion failure: a job that dies has to end up in a terminal
    state with an explanation, because there is no other channel through which the user
    would learn what happened.
    """
    resource = db.query(Resource).filter(Resource.id == job.resource_id).first()
    if resource is None:
        return _finish(db, job, JOB_FAILED, "The knowledge base was removed before indexing started.")

    items = (
        db.query(IngestionJobItem)
        .filter(IngestionJobItem.job_id == job.id)
        .order_by(IngestionJobItem.position.asc())
        .all()
    )

    indexed_chunks = 0
    try:
        for item in items:
            if _cancel_requested(db, job.id):
                return _cancel(db, job, items)

            file_record = db.query(File).filter(File.id == item.file_id).first()
            if file_record is None:
                _fail_item(db, job, item, "The uploaded file was no longer available.")
                continue

            item.status = ITEM_RUNNING
            item.stage = STAGE_INDEXING
            item.started_at = _now()
            job.stage = STAGE_INDEXING
            job.current_document = item.file_name
            db.commit()

            try:
                result = ingest_file(
                    db,
                    resource,
                    file_record,
                    indexed_chunks,
                    should_cancel=lambda: _cancel_requested(db, job.id),
                )
            except IngestionCancelled:
                db.rollback()
                return _cancel(db, job, items)
            except Exception as exc:  # noqa: BLE001 - one bad document must not sink the batch
                db.rollback()
                logger.exception(
                    "document ingestion failed",
                    extra={"extra_fields": {"job_id": job.id, "file_name": item.file_name}},
                )
                _fail_item(db, job, item, _readable_failure(exc))
                continue

            indexed_chunks += len(result.chunks)
            item.status = ITEM_PARTIAL if result.notice else ITEM_SUCCEEDED
            item.stage = STAGE_DONE
            item.chunk_count = len(result.chunks)
            item.embedded_chunks = result.embedded_chunks
            item.page_count = result.page_count
            item.detail = result.notice or _describe(result)
            item.finished_at = _now()

            job.processed_documents += 1
            job.indexed_chunks = indexed_chunks
            db.commit()

        if _cancel_requested(db, job.id):
            return _cancel(db, job, items)

        job.stage = STAGE_LINKING
        job.current_document = None
        db.commit()

        chunks = (
            db.query(DocumentChunk)
            .filter(DocumentChunk.resource_id == resource.id)
            .order_by(DocumentChunk.chunk_index.asc())
            .all()
        )
        _rebuild_graph(db, resource.id, chunks)

        # Only now is the base answerable. `_not_ready_note` in rag/service.py reads this,
        # so setting it earlier would let chat search a partially indexed resource.
        resource.upload_status = True
        db.commit()
    except Exception as exc:  # noqa: BLE001 - the job must reach a terminal state regardless
        db.rollback()
        logger.exception("ingestion job failed", extra={"extra_fields": {"job_id": job.id}})
        return _finish(db, job, JOB_FAILED, _readable_failure(exc))

    return _finish(db, job, JOB_SUCCEEDED, _summarize(items, indexed_chunks))


def _describe(result: IngestedFile) -> str:
    """The one line a user reads about a document that indexed.

    Names the strategy that actually ran, not the one that was asked for. Picking "one chunk
    per page" for a .txt file is a reasonable thing to do and a silent substitution is not a
    reasonable response to it.

    An embedding failure is stated in the same sentence rather than logged and forgotten:
    the document is searchable, just not the way the user paid for, and nothing else in the
    UI would ever say so.
    """
    chunks = len(result.chunks)
    pages = f"{result.page_count} page{'s' if result.page_count != 1 else ''} · " if result.page_count else ""
    parts = [f"{pages}{chunks} passage{'s' if chunks != 1 else ''} indexed"]

    strategy = chunking.STRATEGIES.get(result.effective_strategy, {}).get("label")
    if strategy:
        asked = result.config.strategy
        if result.effective_strategy != asked:
            requested = chunking.STRATEGIES.get(asked, {}).get("label", asked)
            parts.append(f"{strategy} ({requested} does not apply to this format)")
        else:
            parts.append(strategy)

    if result.embedding_error:
        parts.append(f"embeddings failed — searchable by keyword only ({result.embedding_error})")
    elif result.embedded_chunks:
        parts.append(f"{result.embedded_chunks} embedded with {result.config.embedding_model}")

    return " · ".join(parts)


def _summarize(items: list[IngestionJobItem], chunks: int) -> str:
    failed = sum(1 for i in items if i.status == ITEM_FAILED)
    partial = sum(1 for i in items if i.status == ITEM_PARTIAL)
    parts = [f"{chunks} passage{'s' if chunks != 1 else ''} indexed from {len(items)} document{'s' if len(items) != 1 else ''}"]
    if partial:
        parts.append(f"{partial} yielded no readable text")
    if failed:
        parts.append(f"{failed} could not be read")
    return ". ".join(parts) + "."


def _readable_failure(exc: Exception) -> str:
    text = " ".join(str(exc).split()) or exc.__class__.__name__
    return text[:MAX_VARCHAR_CHARS]


def _fail_item(db: Session, job: IngestionJob, item: IngestionJobItem, detail: str) -> None:
    item.status = ITEM_FAILED
    item.stage = STAGE_DONE
    item.detail = detail
    item.finished_at = _now()
    job.processed_documents += 1
    db.commit()


def _cancel(db: Session, job: IngestionJob, items: list[IngestionJobItem]) -> IngestionJob:
    """Stop, and leave nothing behind.

    A cancelled upload discards the resource rather than keeping the documents that had
    already landed. A half-indexed base is the dangerous outcome: it answers questions
    confidently from part of the evidence, and nothing in the answer says which part is
    missing. "Cancel" therefore means what it says everywhere else — as if it never ran.
    """
    for item in items:
        if item.status in (ITEM_PENDING, ITEM_RUNNING):
            item.status = ITEM_CANCELLED
            item.stage = STAGE_DONE
            item.detail = "Cancelled before this document was indexed."
            item.finished_at = _now()

    resource_id = job.resource_id
    # Released before the purge: `ingestion_jobs.resource_id` is a foreign key, and the
    # resource row is about to go.
    job.resource_id = None
    db.flush()
    if resource_id:
        try:
            purge_resource(db, job.user_id, resource_id)
        except (LookupError, ValueError, OSError) as exc:
            logger.warning(
                "could not discard the resource for a cancelled job",
                extra={"extra_fields": {"job_id": job.id, "error": str(exc)}},
            )

    return _finish(db, job, JOB_CANCELLED, "Cancelled. Nothing was added to your knowledge bases.")


def _finish(db: Session, job: IngestionJob, status: str, message: str) -> IngestionJob:
    job.status = status
    job.stage = STAGE_DONE
    job.current_document = None
    job.message = message
    job.finished_at = _now()

    action = {JOB_SUCCEEDED: "INDEXED", JOB_CANCELLED: "CANCELLED", JOB_FAILED: "FAILED"}[status]
    log_activity(
        db,
        job.user_id,
        "",
        action,
        entity_type="ingestion_job",
        entity_id=job.id,
        details=f"Indexing “{job.resource_name}”: {message}",
    )
    db.commit()
    logger.info(
        "ingestion job finished",
        extra={
            "extra_fields": {
                "job_id": job.id,
                "status": status,
                "documents": job.total_documents,
                "chunks": job.indexed_chunks,
            }
        },
    )
    return job


# --------------------------------------------------------------------------------------
# Reads and cancellation, for the API
# --------------------------------------------------------------------------------------


def list_jobs(db: Session, user_id: str, limit: int = 25, active_only: bool = False) -> list[IngestionJob]:
    query = db.query(IngestionJob).options(selectinload(IngestionJob.items)).filter(
        IngestionJob.user_id == user_id
    )
    if active_only:
        query = query.filter(IngestionJob.status.in_([JOB_QUEUED, JOB_RUNNING]))
    return query.order_by(IngestionJob.created_at.desc()).limit(max(1, min(limit, 100))).all()


def get_job(db: Session, user_id: str, job_id: str) -> IngestionJob | None:
    return (
        db.query(IngestionJob)
        .options(selectinload(IngestionJob.items))
        .filter(IngestionJob.id == job_id, IngestionJob.user_id == user_id)
        .first()
    )


def request_cancel(db: Session, user_id: str, job_id: str) -> IngestionJob:
    """Ask a job to stop.

    A job that has not been claimed yet is cancelled here and now — no worker will ever
    look at it otherwise — and that path commits, because discarding the resource is not
    something to leave half-applied in an open transaction. A running job only gets the
    flag set; the worker sees it at its next checkpoint and does the discarding itself,
    since it is the one holding the half-written rows.
    """
    job = get_job(db, user_id, job_id)
    if job is None:
        raise LookupError("Indexing job not found")
    if job.status in TERMINAL_JOB_STATUSES:
        raise ValueError(f"This job already finished ({job.status}) and cannot be cancelled")

    job.cancel_requested = True
    db.flush()

    if job.status == JOB_QUEUED:
        # Claim it away from the pollers first, so a worker cannot pick it up between this
        # check and the commit and start indexing a resource we are about to delete.
        claimed = (
            db.query(IngestionJob)
            .filter(IngestionJob.id == job.id, IngestionJob.status == JOB_QUEUED)
            .update({IngestionJob.status: JOB_RUNNING, IngestionJob.claimed_by: "cancel"}, synchronize_session=False)
        )
        if claimed:
            db.refresh(job)
            items = list(job.items)
            return _cancel(db, job, items)

    return job


def serialize_job(job: IngestionJob, include_items: bool = True) -> dict:
    payload = {
        "id": job.id,
        "resource_id": job.resource_id,
        "resource_name": job.resource_name,
        "status": job.status,
        "stage": job.stage,
        "cancel_requested": bool(job.cancel_requested),
        "total_documents": job.total_documents,
        "processed_documents": job.processed_documents,
        "total_bytes": job.total_bytes,
        "indexed_chunks": job.indexed_chunks,
        "current_document": job.current_document,
        "progress": _progress(job),
        "message": job.message,
        "created_at": job.created_at.isoformat() if job.created_at else None,
        "started_at": job.started_at.isoformat() if job.started_at else None,
        "finished_at": job.finished_at.isoformat() if job.finished_at else None,
    }
    if include_items:
        payload["documents"] = [serialize_job_item(item) for item in job.items]
    return payload


def _progress(job: IngestionJob) -> int:
    """Percent complete, reserving the last slice for graph construction.

    Documents cap at 95 rather than 100 because linking runs after the last one and is not
    instant. A bar that sits at 100% while work continues is the thing that teaches people
    progress bars lie.
    """
    if job.status == JOB_SUCCEEDED:
        return 100
    if not job.total_documents:
        return 0 if job.status == JOB_QUEUED else 95
    return min(95, int(job.processed_documents * 95 / job.total_documents))


def serialize_job_item(item: IngestionJobItem) -> dict:
    return {
        "id": item.id,
        "file_id": item.file_id,
        "position": item.position,
        "file_name": item.file_name,
        "file_type": item.file_type,
        "byte_size": item.byte_size,
        "status": item.status,
        "stage": item.stage,
        "chunk_count": item.chunk_count,
        # Sent alongside `chunk_count` so the UI can state "keyword only" from data rather
        # than by matching strings in `detail`: embeddings were asked for (the config says
        # so) and none landed.
        "embedded_chunks": item.embedded_chunks or 0,
        "page_count": item.page_count,
        "detail": item.detail,
        "config": _item_config(item),
        "started_at": item.started_at.isoformat() if item.started_at else None,
        "finished_at": item.finished_at.isoformat() if item.finished_at else None,
    }


def _item_config(item: IngestionJobItem) -> dict:
    """The settings this document was queued with.

    Falls back to the defaults for items written before `config_json` existed, so the UI
    never has to render a missing settings block — those documents really were indexed with
    exactly these settings.
    """
    try:
        stored = json.loads(item.config_json) if item.config_json else {}
    except (TypeError, ValueError):
        stored = {}
    return chunking.normalize_config(stored).to_dict()


def latest_jobs_for_resources(db: Session, resource_ids: list[str]) -> dict[str, IngestionJob]:
    """The most recent job per resource, in one query rather than one per row.

    The Knowledge Bases screen lists every base a user owns and each needs its indexing
    state; a per-row lookup is the N+1 CLAUDE.md §9 warns against.
    """
    if not resource_ids:
        return {}
    newest = (
        db.query(IngestionJob.resource_id, func.max(IngestionJob.created_at).label("created_at"))
        .filter(IngestionJob.resource_id.in_(resource_ids))
        .group_by(IngestionJob.resource_id)
        .subquery()
    )
    rows = (
        db.query(IngestionJob)
        .join(
            newest,
            (IngestionJob.resource_id == newest.c.resource_id)
            & (IngestionJob.created_at == newest.c.created_at),
        )
        .all()
    )
    return {job.resource_id: job for job in rows if job.resource_id}
