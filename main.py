import asyncio
import logging

from fastapi import FastAPI
from routes import user
from routes import auth
from routes import permissions
from routes import roles
from routes import chats
from routes import activity_logs
from routes import health
from routes import llm
from routes.auth_middleware import JWTAuthMiddleware
from fastapi.middleware.cors import CORSMiddleware

from core.config import get_settings
from core.logging import configure_logging
from core.middleware import RequestContextMiddleware
from db.session import SessionLocal
from rag.jobs import POLL_INTERVAL_SECONDS, claim_next_job, reclaim_stranded_jobs, run_job
from services.llm_provider import refresh_due_model_catalogs


settings = get_settings()
configure_logging(settings.log_level)
logger = logging.getLogger("corpustrace.llm_catalog")
ingestion_logger = logging.getLogger("corpustrace.ingestion")

app = FastAPI(
    title=settings.app_name,
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    # A cross-origin response's headers are invisible to JS unless they are exposed here.
    # /chat/asks returns its retrieval provenance on X-CorpusTrace-Citations, so without this
    # the SPA reads null for it in every deployment where the API is on another origin —
    # and works fine behind the dev proxy, which is the worst way for it to fail.
    expose_headers=["X-CorpusTrace-Citations"],
)

app.add_middleware(JWTAuthMiddleware)
app.add_middleware(RequestContextMiddleware)

app.include_router(auth.router)
app.include_router(user.router)
app.include_router(roles.router)
app.include_router(permissions.router)
app.include_router(chats.router)
app.include_router(chats.resources_router)
app.include_router(activity_logs.router)
app.include_router(health.router)
app.include_router(llm.router)


@app.on_event("startup")
async def start_llm_model_catalog_refresh_loop():
    asyncio.create_task(_llm_model_catalog_refresh_loop())


def _refresh_model_catalogs_blocking():
    db = SessionLocal()
    try:
        return refresh_due_model_catalogs(db)
    finally:
        db.close()


async def _llm_model_catalog_refresh_loop():
    while True:
        try:
            # Off the event loop: this walks every stored credential and calls each
            # provider over the network, and a throttled provider now costs retries and
            # backoff sleeps on top of the request timeout. Run inline it would freeze
            # every other request in the worker for the whole cycle.
            result = await asyncio.to_thread(_refresh_model_catalogs_blocking)
            if result["refreshed"] or result["failed"]:
                # The JSON formatter reads only record.extra_fields; a plain `extra=`
                # sets bare LogRecord attributes that never reach the output.
                logger.info("LLM model catalog refresh completed", extra={"extra_fields": result})
        except Exception:
            logger.exception("LLM model catalog refresh failed")
        await asyncio.sleep(max(settings.llm_model_cache_ttl_hours, 1) * 60 * 60)


@app.on_event("startup")
async def start_document_ingestion_worker():
    # Synchronous and before the loop starts: a job left `running` by the process that
    # just died would otherwise be claimed by nobody and show a progress bar that never
    # moves. Failing it here is the only state that tells the truth.
    await asyncio.to_thread(_reclaim_stranded_jobs_blocking)
    asyncio.create_task(_document_ingestion_worker_loop())


def _reclaim_stranded_jobs_blocking() -> int:
    db = SessionLocal()
    try:
        return reclaim_stranded_jobs(db)
    finally:
        db.close()


def _run_next_ingestion_job_blocking() -> bool:
    """Claim and run one job. Returns whether there was anything to do."""
    db = SessionLocal()
    try:
        job = claim_next_job(db)
        if job is None:
            return False
        run_job(db, job)
        return True
    finally:
        db.close()


async def _document_ingestion_worker_loop():
    while True:
        worked = False
        try:
            # `asyncio.to_thread` uses the loop's default executor, which is *not* the
            # anyio threadpool FastAPI dispatches `def` handlers to — so a long indexing
            # run cannot starve request handling. Running it on the event loop instead
            # would freeze every request in this worker for the length of the upload,
            # which is precisely the problem this whole change exists to remove.
            worked = await asyncio.to_thread(_run_next_ingestion_job_blocking)
        except Exception:
            ingestion_logger.exception("ingestion worker iteration failed")
        # Straight back for the next job when one was just finished; otherwise idle at one
        # cheap indexed query per second.
        await asyncio.sleep(0 if worked else POLL_INTERVAL_SECONDS)


@app.get("/")
def read_root():
    return {"message": "CorpusTrace API"}
