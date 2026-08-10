import json
import logging

from fastapi import APIRouter, Depends, File, Form, Query, Request, UploadFile, WebSocket
from controllers.chat_ws_controller import chat_socket
from controllers.chat_controller import (
    cancel_ingestion_job,
    delete_brain,
    fetch_chatbot_reply,
    fetch_source_chunk,
    fetch_source_file,
    get_brain_list,
    get_indexing_options,
    get_ingestion_job,
    get_ingestion_jobs,
    get_resource_documents,
    purge_brain,
    rename_brain,
    restore_brain,
    upload_documents,
)
from schemas.chat import ResourceRenameRequest
from schemas.response import SuccessResponse, ErrorResponse
from dependencies.auth import check_permissions
from sqlalchemy.orm import Session
from sqlalchemy import text
from db.session import get_db


logger = logging.getLogger("nexarag.chat")


router = APIRouter(
    prefix="/chat",
    tags=["Chats"]
)


@router.get("/history", dependencies=[Depends(check_permissions(["ai_access"]))])
def get_chat_history(request: Request, chat_history_name: str = Query(None), db: Session = Depends(get_db)):
    user_payload = getattr(request.state, "user", None)
    if not user_payload:
        return ErrorResponse(status_code=401, message="Not authenticated")

    user_id = user_payload.get("sub")
    session_name = chat_history_name or f"session_{user_id}"

    # `chat_id` is client-supplied, so it alone never scoped this query to the caller — the
    # `session_{user_id}` default merely made that invisible. Now that a row also carries
    # citations (source filenames and quoted passages), an unscoped read would hand another
    # user's documents to anyone who guessed their chat id, so the owner filter is no longer
    # optional. Still the only raw SQL in a handler body — do not copy the pattern.
    result = db.execute(
        text(
            "SELECT id, user_query, bot_response, brain, citations_json, timestamp "
            "FROM chat_history WHERE chat_id = :chat_id AND user_id = :user_id "
            "ORDER BY timestamp ASC"
        ),
        {"chat_id": session_name, "user_id": user_id},
    )
    rows = result.mappings().all()

    records = []
    for row in rows:
        records.append({
            "id": row["id"],
            "user_query": row["user_query"],
            "bot_response": row["bot_response"],
            "brain": row["brain"],
            "citations": _decode_citations(row["citations_json"], row["id"]),
            "timestamp": str(row["timestamp"]),
        })

    return SuccessResponse(status_code=200, message="Chat history fetched", data={"records": records})


def _decode_citations(raw: str | None, row_id: str) -> list[dict]:
    """Stored citations, or an empty list if the column is empty or unparseable.

    A turn recorded before this column existed has NULL here, and a truncated write would
    have invalid JSON. Neither is worth failing a whole conversation load over — the answer
    text is the part the user came for.
    """
    if not raw:
        return []
    try:
        decoded = json.loads(raw)
    except (TypeError, ValueError):
        logger.warning("unparseable citations_json", extra={"extra_fields": {"chat_history_id": str(row_id)}})
        return []
    return decoded if isinstance(decoded, list) else []


@router.post("/asks", dependencies=[Depends(check_permissions(["ai_access"]))])
def ask_question(
    request: Request,
    query: str,
    chat_history_name: str,
    brain_id: str,
    rag_type: str = Query("contextual_hybrid"),
    llm_provider: str | None = Query(None),
    llm_model: str | None = Query(None),
    db: Session = Depends(get_db),
):
    return fetch_chatbot_reply(query, chat_history_name, brain_id, rag_type, request, db, llm_provider, llm_model)


@router.websocket("/ws")
async def chat_websocket(websocket: WebSocket):
    """The chat transport the SPA prefers; `POST /chat/asks` remains its fallback.

    No `Depends(check_permissions(["ai_access"]))` and no `Depends(get_db)`, and neither is
    an oversight. A route dependency resolves against an HTTP request, `JWTAuthMiddleware`
    is a `BaseHTTPMiddleware` that never sees a websocket scope, and a browser cannot put
    an `Authorization` header on a WebSocket handshake — so `request.state.user` is never
    populated here. Authentication and the `ai_access` check both happen inside
    `chat_socket`, against the token in the connection's first frame, using the same
    functions the HTTP path uses.

    `async def` is required (WebSocket endpoints have no threadpool dispatch), which is
    exactly why every blocking step inside runs under `asyncio.to_thread`.
    """
    await chat_socket(websocket)


# Minimal resources router to keep /resources/brain path for the frontend
resources_router = APIRouter(
    prefix="/resources",
    tags=["Resources"]
)


@resources_router.get("/brain", dependencies=[Depends(check_permissions(["ai_access"]))])
def brain(request: Request, deleted: bool = Query(False), db: Session = Depends(get_db)):
    """Knowledge bases the caller owns. `deleted=true` lists the ones awaiting restore
    or permanent removal — the default is unchanged, so chat still sees only active ones.

    `def`, not `async def`: every call below is blocking SQLAlchemy (CLAUDE.md §9).
    """
    return get_brain_list(request, db, deleted=deleted)


@resources_router.patch("/brain/{resource_id}", dependencies=[Depends(check_permissions(["ai_access"]))])
def rename_resource_route(
    resource_id: str, payload: ResourceRenameRequest, request: Request, db: Session = Depends(get_db)
):
    return rename_brain(request, db, resource_id, payload.resource_name)


@resources_router.delete("/brain/{resource_id}", dependencies=[Depends(check_permissions(["ai_access"]))])
def delete_resource_route(resource_id: str, request: Request, db: Session = Depends(get_db)):
    """Soft delete. Both resource accessors already filter `deleted_at`, so this removes
    the base from chat and from retrieval without touching a single chunk."""
    return delete_brain(request, db, resource_id)


@resources_router.post("/brain/{resource_id}/restore", dependencies=[Depends(check_permissions(["ai_access"]))])
def restore_resource_route(resource_id: str, request: Request, db: Session = Depends(get_db)):
    return restore_brain(request, db, resource_id)


@resources_router.delete("/brain/{resource_id}/permanent", dependencies=[Depends(check_permissions(["ai_access"]))])
def purge_resource_route(resource_id: str, request: Request, db: Session = Depends(get_db)):
    """Irreversible: drops the chunks, graph rows and file rows, and removes the uploaded
    documents from disk. Separate from DELETE so a misclick cannot reach it."""
    return purge_brain(request, db, resource_id)


@resources_router.get("/chunks/{chunk_id}", dependencies=[Depends(check_permissions(["ai_access"]))])
def get_source_chunk(chunk_id: str, request: Request, db: Session = Depends(get_db)):
    """The full passage behind a citation — what the evidence view highlights."""
    return fetch_source_chunk(request, db, chunk_id)


@resources_router.get("/files/{file_id}", dependencies=[Depends(check_permissions(["ai_access"]))])
def get_source_file(file_id: str, request: Request, db: Session = Depends(get_db)):
    # `def`, not `async def`: reading a file off disk and the ownership query are both
    # blocking, so this belongs on the threadpool.
    return fetch_source_file(request, db, file_id)


@resources_router.get("/indexing-options", dependencies=[Depends(check_permissions(["ai_access"]))])
def read_indexing_options(request: Request, db: Session = Depends(get_db)):
    # Declared before /upload_files only for readability; FastAPI matches on the literal
    # path, so ordering against the {file_id} routes above is not load-bearing here.
    return get_indexing_options(request, db)


@resources_router.post("/upload_files", dependencies=[Depends(check_permissions(["ai_access"]))])
def upload_files(
    request: Request,
    resource_name: str = Form(...),
    files: list[UploadFile] = File(...),
    # Optional so an older client — or a curl that just wants to upload something — keeps
    # working unchanged and gets the long-standing defaults.
    document_configs: str | None = Form(None),
    db: Session = Depends(get_db),
):
    # Deliberately `def`, not `async def`: writing the uploaded bytes to disk and the
    # SQLAlchemy inserts are both blocking, so FastAPI must dispatch this to the
    # threadpool. Parsing and indexing no longer happen here — they run in the background
    # worker — but the request still does enough IO to stall an event loop.
    return upload_documents(request, db, resource_name, files, document_configs)


@resources_router.get("/jobs", dependencies=[Depends(check_permissions(["ai_access"]))])
def list_ingestion_jobs(
    request: Request,
    active: bool = Query(False, description="Only jobs still queued or running"),
    limit: int = Query(25, ge=1, le=100),
    db: Session = Depends(get_db),
):
    return get_ingestion_jobs(request, db, active_only=active, limit=limit)


@resources_router.get("/jobs/{job_id}", dependencies=[Depends(check_permissions(["ai_access"]))])
def read_ingestion_job(job_id: str, request: Request, db: Session = Depends(get_db)):
    # The polled endpoint: one indexed lookup plus its items. Kept separate from the list
    # so a client watching one upload does not re-fetch every job it has ever run.
    return get_ingestion_job(request, db, job_id)


@resources_router.post("/jobs/{job_id}/cancel", dependencies=[Depends(check_permissions(["ai_access"]))])
def stop_ingestion_job(job_id: str, request: Request, db: Session = Depends(get_db)):
    return cancel_ingestion_job(request, db, job_id)


# Declared last on purpose. Every literal route above ("/brain", "/jobs", "/chunks/{id}"…)
# is matched before this two-segment pattern, so a base whose id happened to be "chunks"
# cannot shadow them.
@resources_router.get("/{resource_id}/documents", dependencies=[Depends(check_permissions(["ai_access"]))])
def read_resource_documents(resource_id: str, request: Request, db: Session = Depends(get_db)):
    return get_resource_documents(request, db, resource_id)
