"""Chat over a WebSocket: structured frames instead of a raw HTTP body.

`POST /chat/asks` says everything it has to say through one status line, one header and a
byte stream. That is enough for "here is the answer, a piece at a time" and nothing else:
the status code is fixed the moment the first byte leaves, retrieval and generation are
indistinguishable to the client, and stopping a generation means aborting the request. A
socket carries typed frames in both directions, so the same turn can report which stage it
is in, hand over citations the moment retrieval resolves them, accept a cancel mid-answer
without tearing the connection down, and still report an error *after* work has started.

Two things about this path differ deliberately from the HTTP one:

**The provider call does not stream.** `complete_answer` makes one blocking request and
returns the finished text, which the socket then delivers as `delta` frames. Streaming a
provider response and streaming to the browser are separable, and unpicking them buys
real properties: every provider behaves identically (a model with no streaming endpoint is
no longer a special case), a provider failure is still recoverable when it happens —
`complete_answer` degrades to the extractive answer rather than cutting an answer already
half-written — and the answer is complete in memory before the first frame, so what gets
persisted can never disagree with what was sent. The cost is latency: nothing appears
until the whole answer is written. That is the trade this endpoint makes.

**Authentication happens in the first frame, not in middleware.** `JWTAuthMiddleware` is a
`BaseHTTPMiddleware` and never sees a websocket scope, and a browser cannot set an
`Authorization` header on a WebSocket handshake in any case. So the client sends its token
as the first frame, over the already-established connection — which also keeps the token
out of the URL, where it would land in every access log and proxy trace.
"""
from __future__ import annotations

import asyncio
import json
import logging
import uuid
from math import ceil

from fastapi import WebSocket, WebSocketDisconnect

from controllers.chat_controller import persist_chat_turn
from core.permissions import permission_is_satisfied
from db.session import SessionLocal
from dependencies.auth import permission_slugs
from rag.service import AnswerPlan, complete_answer, plan_answer
from routes.auth_middleware import TokenRejected, decode_access_token, assert_session_is_live
from services.llm_provider import LlmProviderError
from services.user import get_user_by_id


logger = logging.getLogger("corpustrace.chat")

# The one slug that gates chat, matching `Depends(check_permissions(["ai_access"]))` on
# every sibling HTTP route.
CHAT_PERMISSION = "ai_access"

# Close codes. 1000/1008 are the standard ones; 4401/4403 are in the application range
# (4000-4999) and mirror the HTTP statuses they stand in for, because a client cannot read
# a response body from a handshake it never completed.
WS_NORMAL_CLOSE = 1000
WS_UNAUTHENTICATED = 4401
WS_FORBIDDEN = 4403

# A socket that connects and then says nothing is either a scanner or a client that
# crashed mid-handshake. Neither should hold a connection open indefinitely.
AUTH_TIMEOUT_SECONDS = 10

# How the finished answer is cut into `delta` frames.
#
# The text is complete before the first frame, so this pacing is presentation, not
# transport: it exists so an answer appears the way a person reads rather than materialising
# all at once, and the constants together are a promise about the worst case — never more
# than MAX_DELTAS frames, so no answer takes longer than MAX_DELTAS *
# DELTA_INTERVAL_SECONDS (~1.2 s) to finish painting however long it is. Short answers run
# at the natural DELTA_TARGET_CHARS rate and finish far sooner; long ones use bigger frames
# rather than more of them, because every frame is a send, a decode and a store dispatch.
DELTA_TARGET_CHARS = 24
MAX_DELTAS = 80
DELTA_INTERVAL_SECONDS = 0.015

STAGE_RETRIEVING = "retrieving"
STAGE_GENERATING = "generating"


def split_into_deltas(text: str) -> list[str]:
    """Cut a finished answer into the frames it will be delivered in.

    Joining the result must reproduce `text` exactly — the client appends frames verbatim
    and the server persists what it sent, so any character invented or dropped here shows
    up as a transcript that disagrees with the stored history.

    Breaks are preferred at a space so a frame never ends mid-word, which is visible: the
    client renders the partial answer as plain text while it arrives, and a split word
    reads as a typo for as long as the next frame takes to land.
    """
    if not text:
        return []

    # The frame count is decided up front and the size recomputed from what is actually
    # left, rather than fixing a size and letting the count fall out. Word-boundary breaks
    # shorten frames unpredictably, so a fixed size overshoots MAX_DELTAS by however much
    # the breaks happened to trim — turning the ceiling into an approximation. Dividing the
    # remainder by the frames remaining absorbs every short frame into the next one, and
    # the last frame is by construction the whole remainder.
    remaining_frames = min(MAX_DELTAS, max(1, ceil(len(text) / DELTA_TARGET_CHARS)))
    deltas: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + ceil((len(text) - start) / remaining_frames), len(text))
        if end < len(text):
            # Search one past `start` so a boundary can never land on the cut itself and
            # produce a zero-length frame — that would loop forever.
            boundary = text.rfind(" ", start + 1, end + 1)
            if boundary > start:
                end = boundary + 1
        deltas.append(text[start:end])
        start = end
        remaining_frames -= 1
    return deltas


async def _receive_frame(websocket: WebSocket) -> dict | None:
    """The next client frame as a mapping, or None once the socket is closed.

    Anything that is not a JSON object comes back as `{}` rather than closing the
    connection: a malformed frame is a client bug, and dropping a conversation over one is
    a worse answer than replying that the frame was not understood.
    """
    try:
        message = await websocket.receive()
    except (WebSocketDisconnect, RuntimeError):
        return None
    if message.get("type") != "websocket.receive":
        return None

    raw = message.get("text")
    if raw is None:
        blob = message.get("bytes") or b""
        raw = blob.decode("utf-8", errors="replace")
    try:
        frame = json.loads(raw)
    except ValueError:
        return {}
    return frame if isinstance(frame, dict) else {}


async def _send(websocket: WebSocket, lock: asyncio.Lock, payload: dict) -> bool:
    """Send one frame. False means the socket is gone.

    Under a lock because the answer task and the receive loop both write to this socket,
    and interleaved `send` calls corrupt the frame stream. Deliberately broad on the
    exception: a client that closed mid-answer is ordinary, and every caller here already
    treats a failed send as "stop, and record what did get through" — turning it into a
    raised exception would only convert that into an unhandled task error.
    """
    async with lock:
        try:
            await websocket.send_json(payload)
            return True
        except Exception:
            logger.debug("chat websocket send failed; treating the client as gone")
            return False


async def _close(websocket: WebSocket, lock: asyncio.Lock, code: int, reason: str) -> None:
    async with lock:
        try:
            await websocket.close(code=code, reason=reason)
        except Exception:
            logger.debug("chat websocket close failed; the client had already gone")


def _authorize(token: str) -> tuple[str | None, int, str]:
    """`(user_id, close_code, message)` — the whole auth decision, on a worker thread.

    JWT verification, the `user_sessions` lookup and the role→permission walk are all
    blocking, and a WebSocket handler must be `async def`, so none of it may run on the
    event loop (CLAUDE.md §9). Every rule applied here is the same function the HTTP path
    uses, not a re-implementation.
    """
    try:
        payload = decode_access_token(token)
    except TokenRejected as exc:
        return None, WS_UNAUTHENTICATED, str(exc)

    db = SessionLocal()
    try:
        try:
            assert_session_is_live(db, token, payload)
        except TokenRejected as exc:
            return None, WS_UNAUTHENTICATED, str(exc)

        user_id = payload.get("sub")
        user = get_user_by_id(db, user_id)
        if not user_id or not user:
            return None, WS_UNAUTHENTICATED, "Not authenticated"
        if not permission_is_satisfied(permission_slugs(user), CHAT_PERMISSION):
            return None, WS_FORBIDDEN, "Insufficient permissions"
        return user_id, WS_NORMAL_CLOSE, ""
    finally:
        db.close()


def _plan(user_id: str, brain_id: str, query: str, rag_type: str, provider: str | None, model: str | None) -> AnswerPlan:
    db = SessionLocal()
    try:
        return plan_answer(db, user_id, brain_id, query, rag_type, provider, model)
    finally:
        db.close()


def _complete(user_id: str, plan: AnswerPlan, query: str, provider: str | None, model: str | None) -> str:
    db = SessionLocal()
    try:
        return complete_answer(db, user_id, plan, query, provider, model)
    finally:
        db.close()


async def _authenticate(websocket: WebSocket, lock: asyncio.Lock) -> str | None:
    """Consume the opening `auth` frame. Returns the user id, or None having closed."""
    try:
        frame = await asyncio.wait_for(_receive_frame(websocket), AUTH_TIMEOUT_SECONDS)
    except asyncio.TimeoutError:
        await _close(websocket, lock, WS_UNAUTHENTICATED, "Authentication timed out")
        return None
    if frame is None:
        return None

    token = frame.get("token")
    if frame.get("type") != "auth" or not isinstance(token, str) or not token:
        await _send(websocket, lock, {"type": "error", "status_code": 401, "message": "Missing or invalid token"})
        await _close(websocket, lock, WS_UNAUTHENTICATED, "Missing or invalid token")
        return None

    user_id, code, message = await asyncio.to_thread(_authorize, token)
    if user_id is None:
        # The frame goes first: a client can read it and decide whether to refresh its
        # token, whereas a close code alone cannot carry why.
        status = 403 if code == WS_FORBIDDEN else 401
        await _send(websocket, lock, {"type": "error", "status_code": status, "message": message})
        await _close(websocket, lock, code, message)
        return None

    await _send(websocket, lock, {"type": "ready"})
    return user_id


def _string(frame: dict, key: str) -> str | None:
    """A non-empty string field, or None. Frames are client-supplied JSON: a number or a
    nested object where a string belongs must not reach the query layer."""
    value = frame.get(key)
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value or None


async def _handle_ask(
    websocket: WebSocket,
    lock: asyncio.Lock,
    user_id: str,
    frame: dict,
    cancel: asyncio.Event,
    client_ip: str | None,
) -> None:
    """Answer one question, reporting each stage as it happens."""
    request_id = _string(frame, "request_id") or str(uuid.uuid4())

    async def error(status_code: int, message: str) -> None:
        await _send(
            websocket,
            lock,
            {"type": "error", "request_id": request_id, "status_code": status_code, "message": message},
        )

    query = _string(frame, "query")
    brain_id = _string(frame, "brain_id")
    if not query:
        await error(400, "A question is required")
        return
    if not brain_id:
        await error(400, "A knowledge base is required")
        return

    rag_type = _string(frame, "rag_type") or "contextual_hybrid"
    provider = _string(frame, "llm_provider")
    model = _string(frame, "llm_model")
    session_name = _string(frame, "chat_history_name") or f"session_{user_id}"

    await _send(websocket, lock, {"type": "accepted", "request_id": request_id})
    await _send(websocket, lock, {"type": "status", "request_id": request_id, "stage": STAGE_RETRIEVING})

    try:
        plan = await asyncio.to_thread(_plan, user_id, brain_id, query, rag_type, provider, model)
    except LlmProviderError as exc:
        await error(exc.status_code, str(exc))
        return
    except ValueError as exc:
        await error(404, str(exc))
        return
    except Exception:
        # There is no global exception handler behind a socket either, and an unhandled
        # error here would silently leave the client waiting on a turn that never lands.
        logger.exception("chat websocket planning failed", extra={"extra_fields": {"brain_id": brain_id}})
        await error(500, "Could not answer this question")
        return

    # Citations resolve during retrieval, so they are sent the moment they exist rather
    # than waiting for the answer they annotate. Unlike the HTTP header these carry the
    # full citation — snippet and character offsets included — because a frame has no
    # 8 KB proxy limit to stay under.
    if plan.citations:
        await _send(
            websocket,
            lock,
            {"type": "citations", "request_id": request_id, "citations": plan.citations},
        )

    if cancel.is_set():
        # Stopped before a provider call was made: nothing was generated, nothing was
        # shown, so there is no turn to record.
        await _send(websocket, lock, {"type": "cancelled", "request_id": request_id})
        return

    if plan.synthesize:
        await _send(websocket, lock, {"type": "status", "request_id": request_id, "stage": STAGE_GENERATING})

    try:
        answer = await asyncio.to_thread(_complete, user_id, plan, query, provider, model)
    except Exception:
        # `complete_answer` already turns a provider failure into the extractive answer,
        # so reaching here means something genuinely unexpected broke.
        logger.exception("chat websocket generation failed", extra={"extra_fields": {"brain_id": brain_id}})
        await error(502, "Could not write an answer")
        return

    delivered: list[str] = []
    cancelled = False
    for position, delta in enumerate(split_into_deltas(answer)):
        if cancel.is_set():
            cancelled = True
            break
        if position:
            await asyncio.sleep(DELTA_INTERVAL_SECONDS)
        if not await _send(websocket, lock, {"type": "delta", "request_id": request_id, "text": delta}):
            # The client is gone. Everything before this frame was seen, so it is still
            # recorded — the same choice the HTTP path makes on a disconnect.
            cancelled = True
            break
        delivered.append(delta)

    brain_label = f"{brain_id}:{plan.mode}" + (f":{provider}/{model}" if provider and model else "")
    details = f"Asked question with {plan.mode}" + (f" using {provider}/{model}" if provider and model else "")
    # Before the terminal frame, not after: a client that reloads its history on `done`
    # must not race the write that produced it.
    await asyncio.to_thread(
        persist_chat_turn,
        user_id=user_id,
        session_name=session_name,
        query=query,
        answer="".join(delivered),
        brain_id=brain_id,
        brain_label=brain_label,
        details=details,
        client_ip=client_ip,
        citations=plan.citations,
    )

    await _send(
        websocket,
        lock,
        {"type": "cancelled" if cancelled else "done", "request_id": request_id},
    )


async def chat_socket(websocket: WebSocket) -> None:
    """One connection: authenticate, then answer questions until the client leaves."""
    await websocket.accept()
    lock = asyncio.Lock()

    user_id = await _authenticate(websocket, lock)
    if user_id is None:
        return

    client_ip = websocket.client.host if websocket.client else None
    pending: asyncio.Task | None = None
    cancel = asyncio.Event()

    try:
        while True:
            frame = await _receive_frame(websocket)
            if frame is None:
                break

            kind = frame.get("type")
            if kind == "ping":
                # Idle sockets die quietly behind proxies (nginx closes at
                # `proxy_read_timeout`), and a client that only discovers this when it
                # sends a question loses that question.
                await _send(websocket, lock, {"type": "pong"})
            elif kind == "cancel":
                # Setting the flag is all this does. The provider call is a blocking call
                # on a worker thread and cannot be interrupted, so cancelling stops the
                # *delivery* — which is what the user asked for — and the answer already
                # in flight is discarded when it returns.
                cancel.set()
            elif kind == "ask":
                if pending is not None and not pending.done():
                    await _send(
                        websocket,
                        lock,
                        {
                            "type": "error",
                            "request_id": _string(frame, "request_id"),
                            "status_code": 409,
                            "message": "A question is already being answered on this connection",
                        },
                    )
                    continue
                # A fresh Event per question: reusing one would leave a cancel from the
                # previous turn latched and kill the next answer before it started.
                cancel = asyncio.Event()
                pending = asyncio.create_task(
                    _handle_ask(websocket, lock, user_id, frame, cancel, client_ip)
                )
            else:
                await _send(
                    websocket,
                    lock,
                    {"type": "error", "status_code": 400, "message": f"Unsupported message type: {kind!r}"},
                )
    finally:
        if pending is not None and not pending.done():
            # The client has gone; nothing is left to deliver the rest of the answer to.
            # `_handle_ask` persists as it goes, so what was already sent is safe.
            pending.cancel()
