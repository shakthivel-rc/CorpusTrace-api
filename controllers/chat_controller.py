import base64
import json
import logging
from pathlib import Path
from typing import Iterator

from fastapi import UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from sqlalchemy.orm import Session

from core.config import get_settings
from db.session import SessionLocal
from models.chat_history import ChatHistory
from models.ingestion import JOB_CANCELLED
from models.resource import Resource
from rag.jobs import (
    enqueue_ingestion,
    get_job,
    latest_jobs_for_resources,
    list_jobs,
    request_cancel,
    serialize_job,
)
from rag import chunking
from rag.chunking import IndexingConfig
from rag.service import (
    SUPPORTED_RAG_MODES,
    get_user_chunk,
    get_user_file,
    list_resource_documents,
    list_user_resources,
    purge_resource,
    rename_resource,
    resource_counts,
    restore_resource,
    serialize_chunk,
    serialize_resource,
    soft_delete_resource,
    stream_answer_question,
)
from schemas.response import ErrorResponse, SuccessResponse
from services.activity_log import log_activity
from services.llm_provider import LlmProviderError, embedding_provider_options


logger = logging.getLogger("nexarag.chat")

# Streaming only reaches the user token-by-token if nothing in front of the app buffers
# the body. nginx buffers proxied responses by default, which would re-assemble the whole
# answer and deliver it in one burst — exactly the behaviour this endpoint replaced.
STREAM_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}

# Retrieval provenance rides back on a response header rather than in the body.
#
# Citations are fully resolved before the first token (see `stream_answer_question`), and
# the body is raw model text that the client appends verbatim to the visible answer — so
# there is nowhere in it to put structured data without inventing a framing the whole
# streaming path would have to learn. A header carries it without touching a single byte
# of what already works.
CITATION_HEADER = "X-Nexarag-Citations"
# Fields the client needs to render a chip and open the panel. `snippet`, `char_start` and
# `char_end` are deliberately absent: they are fetched per citation from
# GET /resources/chunks/{id} when the panel actually opens, which keeps document text out
# of headers and the header itself comfortably inside every proxy's limit.
CITATION_HEADER_FIELDS = (
    "index",
    "chunk_id",
    "file_id",
    "source_name",
    "chunk_index",
    "modality",
    "page_start",
    "page_end",
    "score",
)
# Proxies routinely cap a single header at 8 KB and drop the whole response when it is
# exceeded — losing the answer, not just the citations. Stay well under it.
MAX_CITATION_HEADER_CHARS = 6144
MAX_CITATION_LABEL_CHARS = 120

# User-uploaded bytes are served back to a browser, so the content type is derived from the
# extension against this allow-list — never from `files.file_type`, which is whatever the
# client put in its multipart header. Serving an attacker-chosen `text/html` (or
# `image/svg+xml`) inline from the API origin is stored XSS.
INLINE_MEDIA_TYPES = {
    ".pdf": "application/pdf",
    ".txt": "text/plain; charset=utf-8",
    ".md": "text/plain; charset=utf-8",
    ".csv": "text/csv; charset=utf-8",
    ".tsv": "text/tab-separated-values; charset=utf-8",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
}
SOURCE_FILE_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "Content-Security-Policy": "default-src 'none'; sandbox",
    "Cache-Control": "private, max-age=300",
}


def _encode_citations(citations: list[dict]) -> str | None:
    """Compact citations into a base64url header value, or None if there are none.

    base64 because a header value must be ASCII and a source filename need not be. The
    payload is trimmed until it fits rather than sent oversized: a dropped citation costs
    one chip, a rejected header costs the entire answer.
    """
    if not citations:
        return None

    payload = [
        {
            key: (value[:MAX_CITATION_LABEL_CHARS] if isinstance(value, str) else value)
            for key, value in citation.items()
            if key in CITATION_HEADER_FIELDS
        }
        for citation in citations
    ]

    while payload:
        encoded = base64.urlsafe_b64encode(
            json.dumps(payload, separators=(",", ":")).encode("utf-8")
        ).decode("ascii")
        if len(encoded) <= MAX_CITATION_HEADER_CHARS:
            if len(payload) < len(citations):
                logger.warning(
                    "citation header truncated to fit",
                    extra={"extra_fields": {"kept": len(payload), "total": len(citations)}},
                )
            return encoded
        payload.pop()

    logger.warning("citation header dropped: a single citation exceeded the size cap")
    return None


def fetch_chatbot_reply(query, chat_history_name, brain_id, rag_type, request, db: Session, llm_provider=None, llm_model=None):
    user_payload = getattr(request.state, "user", None)
    if not user_payload:
        return JSONResponse(
            status_code=401,
            content=ErrorResponse(status_code=401, message="Not authenticated").model_dump(),
        )

    user_id = user_payload.get("sub")
    # Retrieval, the evidence gate and the first LLM delta all resolve here, while the
    # request still holds `db` and can still answer with a JSON error envelope. Once the
    # StreamingResponse is returned the status line is fixed, so every failure that can be
    # reported properly must be reported before this point.
    try:
        rag_stream = stream_answer_question(db, user_id, brain_id, query, rag_type, llm_provider, llm_model)
    except LlmProviderError as exc:
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(status_code=exc.status_code, message=str(exc)).model_dump(),
        )
    except ValueError as exc:
        return JSONResponse(
            status_code=404,
            content=ErrorResponse(status_code=404, message=str(exc)).model_dump(),
        )

    session_name = chat_history_name or f"session_{user_id}"
    client_ip = request.client.host if request.client else None
    brain_label = f"{brain_id}:{rag_stream.mode}" + (f":{llm_provider}/{llm_model}" if llm_provider and llm_model else "")
    details = f"Asked question with {rag_stream.mode}" + (
        f" using {llm_provider}/{llm_model}" if llm_provider and llm_model else ""
    )

    def stream_body() -> Iterator[str]:
        # A plain (non-async) generator: Starlette runs it via iterate_in_threadpool, so
        # the blocking provider reads never touch the event loop. As an `async def` this
        # would stall every other request in the worker for the length of the answer.
        collected: list[str] = []
        try:
            for chunk in rag_stream.chunks:
                collected.append(chunk)
                yield chunk
        except Exception as exc:
            # Headers are long gone, so there is no status code left to set — the only
            # honest option is to keep what was streamed and say it ended early.
            logger.exception(
                "chat stream failed",
                extra={"extra_fields": {"brain_id": brain_id, "mode": rag_stream.mode, "error": str(exc)}},
            )
            note = "\n\n> **Note:** this answer ended early because of an unexpected server error."
            collected.append(note)
            yield note
        finally:
            # On normal completion this runs on the threadpool thread driving the
            # generator. On a client disconnect Starlette abandons the iterator rather
            # than closing it, so the turn is instead recorded when the generator is
            # collected and GeneratorExit unwinds to here — promptly under CPython
            # refcounting, but on the collecting thread and not at a guaranteed moment.
            # Recording a partial answer late still beats losing a turn the user saw.
            persist_chat_turn(
                user_id=user_id,
                session_name=session_name,
                query=query,
                answer="".join(collected),
                brain_id=brain_id,
                brain_label=brain_label,
                details=details,
                client_ip=client_ip,
                citations=rag_stream.citations,
            )

    headers = dict(STREAM_HEADERS)
    encoded_citations = _encode_citations(rag_stream.citations)
    if encoded_citations:
        headers[CITATION_HEADER] = encoded_citations

    return StreamingResponse(stream_body(), media_type="text/event-stream", headers=headers)


def persist_chat_turn(
    user_id: str,
    session_name: str,
    query: str,
    answer: str,
    brain_id: str,
    brain_label: str,
    details: str,
    client_ip: str | None,
    citations: list[dict] | None = None,
) -> None:
    """Write the completed turn on a dedicated session.

    The request-scoped `db` dependency is already torn down by the time a streaming body
    is consumed, so this cannot reuse it. Nothing here may raise: the user has already
    received the answer, and failing to record it must not break the response that is
    still being written.

    Public because the chat WebSocket records its turns through this same function. A
    socket has no request-scoped session at all, so it needs exactly the properties this
    already has — its own `SessionLocal()`, and no exception that can escape.
    """
    if not answer.strip():
        return

    db = SessionLocal()
    try:
        db.add(
            ChatHistory(
                chat_id=session_name,
                user_query=query,
                user_id=user_id,
                bot_response=answer,
                brain=brain_label,
                # Stored with the snippet the header omits, so a restored conversation can
                # still show something for a citation whose chunk has since been deleted.
                citations_json=json.dumps(citations) if citations else None,
            )
        )
        log_activity(
            db,
            user_id,
            "",
            "CHAT",
            entity_type="resource",
            entity_id=brain_id,
            details=details,
            ip_address=client_ip,
        )
        db.commit()
    except Exception:
        db.rollback()
        logger.exception("failed to persist chat turn", extra={"extra_fields": {"brain_id": brain_id}})
    finally:
        db.close()


def get_brain_list(request, db: Session, deleted: bool = False):
    user_payload = getattr(request.state, "user", None)
    if not user_payload:
        return JSONResponse(
            status_code=401,
            content=ErrorResponse(status_code=401, message="Not authenticated").model_dump(),
        )

    user_id = user_payload.get("sub")
    resources = list_user_resources(db, user_id, deleted=deleted)
    resource_ids = [resource.id for resource in resources]
    counts = resource_counts(db, resource_ids)
    # One query for the whole page, not one per row: this endpoint backs both the chat
    # picker and the management screen, and every base needs its indexing state.
    jobs = latest_jobs_for_resources(db, resource_ids)
    records = []
    for resource in resources:
        payload = serialize_resource(resource)
        payload.update(counts.get(resource.id, {"file_count": 0, "chunk_count": 0}))
        payload["deleted_at"] = resource.deleted_at.isoformat() if resource.deleted_at else None
        # `job` is None for every base uploaded before background indexing existed. The
        # client must read that as "nothing in flight", never as "unknown, so block it" —
        # those bases are indexed and answerable, which `upload_status` already says.
        job = jobs.get(resource.id)
        payload["job"] = serialize_job(job, include_items=False) if job else None
        records.append(payload)
    return SuccessResponse(
        status_code=200,
        message="Resources fetched",
        data={"records": records, "total": len(records)}
    )


def _resource_action(request, db: Session, resource_id: str, action, *, log_action: str, detail: str, message: str):
    """Shared shell for rename/delete/restore/purge.

    They differ only in the service call and the wording, and every one of them needs the
    same four things: an authenticated caller, ownership enforced inside the service, an
    activity-log row, and a commit — with a rollback if anything raises, since the
    controller owns the transaction boundary.
    """
    user_payload = getattr(request.state, "user", None)
    if not user_payload:
        return JSONResponse(
            status_code=401,
            content=ErrorResponse(status_code=401, message="Not authenticated").model_dump(),
        )
    user_id = user_payload.get("sub")
    try:
        result = action(db, user_id, resource_id)
    except LookupError as exc:
        return JSONResponse(status_code=404, content=ErrorResponse(status_code=404, message=str(exc)).model_dump())
    except ValueError as exc:
        return JSONResponse(status_code=400, content=ErrorResponse(status_code=400, message=str(exc)).model_dump())
    except Exception:
        db.rollback()
        logger.exception("resource action failed", extra={"extra_fields": {"resource_id": resource_id}})
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(status_code=500, message="Could not complete the request").model_dump(),
        )

    log_activity(
        db,
        user_id,
        "",
        log_action,
        entity_type="resource",
        entity_id=resource_id,
        details=detail,
        ip_address=request.client.host if request.client else None,
    )
    db.commit()
    data = result if isinstance(result, dict) else serialize_resource(result)
    return SuccessResponse(status_code=200, message=message, data=data)


def rename_brain(request, db: Session, resource_id: str, name: str):
    return _resource_action(
        request,
        db,
        resource_id,
        lambda db_, user_id, rid: rename_resource(db_, user_id, rid, name),
        log_action="UPDATE",
        detail=f"Renamed knowledge base to {name.strip()[:120]}",
        message="Knowledge base renamed",
    )


def delete_brain(request, db: Session, resource_id: str):
    return _resource_action(
        request,
        db,
        resource_id,
        soft_delete_resource,
        log_action="DELETE",
        detail="Moved knowledge base to deleted",
        message="Knowledge base deleted",
    )


def restore_brain(request, db: Session, resource_id: str):
    return _resource_action(
        request,
        db,
        resource_id,
        restore_resource,
        log_action="UPDATE",
        detail="Restored knowledge base",
        message="Knowledge base restored",
    )


def purge_brain(request, db: Session, resource_id: str):
    return _resource_action(
        request,
        db,
        resource_id,
        purge_resource,
        log_action="DELETE",
        detail="Permanently deleted knowledge base, its passages and its uploaded files",
        message="Knowledge base permanently deleted",
    )


def get_indexing_options(request, db: Session):
    """Everything the upload UI needs to render the per-document settings.

    Served from the server rather than duplicated in TypeScript on purpose. The strategy
    list, the bounds and the per-mode recommendations are all statements about what the
    chunker and the retrieval code actually do; a second copy in the frontend would be a
    second thing to keep true, and the first symptom of it going stale would be a
    recommendation that no longer matches the code that acts on it.

    Which providers are *connected* is deliberately not merged in here — this endpoint is
    the catalogue, and the chat settings panel already owns credential state.
    """
    user_payload = getattr(request.state, "user", None)
    if not user_payload:
        return JSONResponse(
            status_code=401,
            content=ErrorResponse(status_code=401, message="Not authenticated").model_dump(),
        )

    strategies = [
        {"id": key, **spec, "recommended": key == chunking.DEFAULT_STRATEGY}
        for key, spec in chunking.STRATEGIES.items()
    ]
    # SUPPORTED_RAG_MODES is a set, not a mapping — the display label lives on the
    # recommendation itself, which is also the only place that knows what to call a mode.
    recommendations = {mode: chunking.recommendation_for(mode) for mode in sorted(SUPPORTED_RAG_MODES)}

    return SuccessResponse(
        status_code=200,
        message="Indexing options retrieved",
        data={
            "strategies": strategies,
            "chunk_sizes": chunking.CHUNK_SIZE_PRESETS,
            "overlaps": chunking.OVERLAP_PRESETS,
            "bounds": {
                "min_chunk_size": chunking.MIN_CHUNK_SIZE,
                "max_chunk_size": chunking.MAX_CHUNK_SIZE,
                "min_overlap": chunking.MIN_OVERLAP,
                "max_overlap_ratio": chunking.MAX_OVERLAP_RATIO,
            },
            "defaults": chunking.DEFAULT_CONFIG.to_dict(),
            "embedding_providers": embedding_provider_options(),
            "recommendations": recommendations,
        },
    )


def get_resource_documents(request, db: Session, resource_id: str):
    """The documents in one knowledge base, each with the settings it was indexed under."""
    user_payload = getattr(request.state, "user", None)
    if not user_payload:
        return JSONResponse(
            status_code=401,
            content=ErrorResponse(status_code=401, message="Not authenticated").model_dump(),
        )

    documents = list_resource_documents(db, user_payload.get("sub"), resource_id)
    if documents is None:
        return JSONResponse(
            status_code=404,
            content=ErrorResponse(status_code=404, message="Knowledge base not found").model_dump(),
        )
    return SuccessResponse(
        status_code=200,
        message="Documents retrieved",
        data={"records": documents, "total": len(documents)},
    )


def parse_document_configs(raw: str | None, count: int) -> list[IndexingConfig]:
    """Decode the per-document indexing settings that came with an upload.

    The wire format is one JSON array, positional against `files` — the Nth entry configures
    the Nth document. A multipart form cannot nest, and repeating a scalar field per file
    would make the pairing depend on field ordering the client cannot control.

    Malformed input yields defaults rather than a 400. Every value inside is clamped by
    `normalize_config` anyway, and the failure mode of being strict here is rejecting a
    perfectly good batch of documents over a settings blob the user never typed.
    """
    if not raw:
        return [chunking.DEFAULT_CONFIG] * count
    try:
        decoded = json.loads(raw)
    except (TypeError, ValueError):
        logger.warning("ignoring unparseable document_configs on upload")
        decoded = []
    if not isinstance(decoded, list):
        decoded = []
    return [
        chunking.normalize_config(decoded[index] if index < len(decoded) and isinstance(decoded[index], dict) else None)
        for index in range(count)
    ]


def upload_documents(
    request,
    db: Session,
    resource_name: str,
    files: list[UploadFile],
    document_configs: str | None = None,
):
    """Accept the documents and queue the indexing.

    Returns 202, not 201: the knowledge base is not usable when this returns. The response
    carries the job so the client has something to poll and something to cancel — the
    previous 201 was only ever sent after minutes of blocking work, with no way to observe
    or stop it.

    `document_configs` is optional throughout. An upload that sends none is indexed exactly
    the way every upload was before per-document settings existed.
    """
    user_payload = getattr(request.state, "user", None)
    if not user_payload:
        return JSONResponse(
            status_code=401,
            content=ErrorResponse(status_code=401, message="Not authenticated").model_dump(),
        )

    user_id = user_payload.get("sub")
    configs = parse_document_configs(document_configs, len(files or []))
    try:
        job = enqueue_ingestion(db, user_id, resource_name, files, configs)
    except ValueError as exc:
        return JSONResponse(
            status_code=400,
            content=ErrorResponse(status_code=400, message=str(exc)).model_dump(),
        )
    except Exception:
        db.rollback()
        logger.exception("could not queue document indexing")
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(status_code=500, message="Could not accept the upload").model_dump(),
        )

    resource = db.query(Resource).filter(Resource.id == job.resource_id).first()
    log_activity(
        db,
        user_id,
        "",
        "CREATE",
        entity_type="resource",
        entity_id=job.resource_id,
        details=f"Uploaded {job.total_documents} document(s) to “{job.resource_name}”; indexing queued",
        ip_address=request.client.host if request.client else None,
    )
    db.commit()

    payload = serialize_resource(resource) if resource else {}
    payload["job"] = serialize_job(job)
    # Wrapped, so the wire status really is 202. Returning the bare envelope would send
    # HTTP 200 with a body claiming 202 — the same mismatch CLAUDE.md §5 calls out for
    # errors, and here it would tell a client the work is finished when it has not started.
    return JSONResponse(
        status_code=202,
        content=SuccessResponse(
            status_code=202,
            message="Documents uploaded. Indexing has been queued.",
            data=payload,
        ).model_dump(),
    )


def get_ingestion_jobs(request, db: Session, active_only: bool = False, limit: int = 25):
    user_payload = getattr(request.state, "user", None)
    if not user_payload:
        return JSONResponse(
            status_code=401,
            content=ErrorResponse(status_code=401, message="Not authenticated").model_dump(),
        )

    jobs = list_jobs(db, user_payload.get("sub"), limit=limit, active_only=active_only)
    records = [serialize_job(job) for job in jobs]
    return SuccessResponse(
        status_code=200,
        message="Indexing jobs fetched",
        data={"records": records, "total": len(records)},
    )


def get_ingestion_job(request, db: Session, job_id: str):
    user_payload = getattr(request.state, "user", None)
    if not user_payload:
        return JSONResponse(
            status_code=401,
            content=ErrorResponse(status_code=401, message="Not authenticated").model_dump(),
        )

    job = get_job(db, user_payload.get("sub"), job_id)
    if job is None:
        # Someone else's job is reported as missing rather than forbidden, for the same
        # reason a chunk is: a 403 confirms the id exists.
        return JSONResponse(
            status_code=404,
            content=ErrorResponse(status_code=404, message="Indexing job not found").model_dump(),
        )
    return SuccessResponse(status_code=200, message="Indexing job fetched", data=serialize_job(job))


def cancel_ingestion_job(request, db: Session, job_id: str):
    user_payload = getattr(request.state, "user", None)
    if not user_payload:
        return JSONResponse(
            status_code=401,
            content=ErrorResponse(status_code=401, message="Not authenticated").model_dump(),
        )

    user_id = user_payload.get("sub")
    try:
        job = request_cancel(db, user_id, job_id)
    except LookupError as exc:
        return JSONResponse(status_code=404, content=ErrorResponse(status_code=404, message=str(exc)).model_dump())
    except ValueError as exc:
        return JSONResponse(status_code=409, content=ErrorResponse(status_code=409, message=str(exc)).model_dump())
    except Exception:
        db.rollback()
        logger.exception("could not cancel indexing job", extra={"extra_fields": {"job_id": job_id}})
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(status_code=500, message="Could not cancel the job").model_dump(),
        )

    log_activity(
        db,
        user_id,
        "",
        "CANCEL",
        entity_type="ingestion_job",
        entity_id=job_id,
        details=f"Requested cancellation of indexing for “{job.resource_name}”",
        ip_address=request.client.host if request.client else None,
    )
    db.commit()
    # Still `running` here means the worker is mid-document and will stop at its next
    # checkpoint — reporting "cancelled" now would be a claim we cannot yet make.
    message = (
        "Indexing cancelled"
        if job.status == JOB_CANCELLED
        else "Cancellation requested — the current document will finish stopping shortly"
    )
    return SuccessResponse(status_code=200, message=message, data=serialize_job(job))


def fetch_source_chunk(request, db: Session, chunk_id: str):
    """The full text and position of one retrieved chunk, for the evidence view."""
    user_payload = getattr(request.state, "user", None)
    if not user_payload:
        return JSONResponse(
            status_code=401,
            content=ErrorResponse(status_code=401, message="Not authenticated").model_dump(),
        )

    chunk = get_user_chunk(db, user_payload.get("sub"), chunk_id)
    if not chunk:
        # A chunk belonging to someone else is reported as missing, not forbidden: a 403
        # would confirm the id exists and name a document the caller cannot see.
        return JSONResponse(
            status_code=404,
            content=ErrorResponse(status_code=404, message="Source passage not found").model_dump(),
        )

    return SuccessResponse(status_code=200, message="Source passage fetched", data=serialize_chunk(chunk))


def fetch_source_file(request, db: Session, file_id: str):
    """Serve an uploaded document's bytes back to its owner.

    This is the only path by which a browser can see an uploaded file, so the two checks
    below are the whole security boundary: ownership (via the resource), and that the
    stored path really does live under the upload directory.
    """
    user_payload = getattr(request.state, "user", None)
    if not user_payload:
        return JSONResponse(
            status_code=401,
            content=ErrorResponse(status_code=401, message="Not authenticated").model_dump(),
        )

    file_record = get_user_file(db, user_payload.get("sub"), file_id)
    if not file_record:
        return JSONResponse(
            status_code=404,
            content=ErrorResponse(status_code=404, message="Source file not found").model_dump(),
        )

    settings = get_settings()
    try:
        # `file_url` is stored data, and stored data is untrusted input: resolve it and
        # prove it is inside the upload directory before opening anything. resolve() first
        # so a symlink cannot point out of the tree after the check.
        base = Path(settings.rag_upload_dir).resolve()
        target = Path(file_record.file_url).resolve()
        target.relative_to(base)
    except (ValueError, OSError):
        logger.warning(
            "source file path is outside the upload directory",
            extra={"extra_fields": {"file_id": file_id}},
        )
        return JSONResponse(
            status_code=404,
            content=ErrorResponse(status_code=404, message="Source file not found").model_dump(),
        )

    if not target.is_file():
        # Indexed but no longer on disk — a real state after a redeploy onto fresh storage.
        logger.info("source file missing from disk", extra={"extra_fields": {"file_id": file_id}})
        return JSONResponse(
            status_code=404,
            content=ErrorResponse(status_code=404, message="Source file is no longer available").model_dump(),
        )

    media_type = INLINE_MEDIA_TYPES.get(target.suffix.lower(), "application/octet-stream")
    headers = dict(SOURCE_FILE_HEADERS)
    # `file_name` was sanitized to ASCII at upload, so it is safe in a header; the quotes
    # are escaped anyway rather than trusting that to stay true.
    safe_name = file_record.file_name.replace('"', "")
    headers["Content-Disposition"] = f'inline; filename="{safe_name}"'
    return FileResponse(path=str(target), media_type=media_type, headers=headers)
