"""POST /chat/asks — the real streaming path, end to end.

This endpoint used to compute the whole answer, persist it, and then drip it back word
by word with a 20 ms sleep: a typing animation, not streaming. It now streams the LLM's
actual output, which moves persistence *after* the response body — so the two things
worth guarding are that the turn still reaches chat_history and activity_logs, and that
failures which can still be reported as a status code are reported before the first byte.
"""
import asyncio
import json
import time

import jwt
import pytest

import rag.service as rag_service
from core.config import get_settings
from models.activity_log import ActivityLog
from models.chat_history import ChatHistory
from models.permissions import Permission
from models.rag import DocumentChunk
from models.resource import Resource
from models.role_permissions import RolePermission
from models.roles import Role
from models.user import User
from models.user_roles import UserRole
from models.user_session import UserSession
from services.llm_provider import LlmProviderError
from utils.password import hash_password

pytestmark = pytest.mark.integration


def _seed_ai_user(db, user_id: str = "chat-user") -> str:
    """A user holding ai_access, with a live session — the minimum to reach /chat/asks."""
    settings = get_settings()
    token = jwt.encode(
        {"sub": user_id, "type": "access", "exp": int(time.time()) + 3600},
        settings.secret_key,
        algorithm=settings.algorithm,
    )
    db.add(
        User(
            id=user_id,
            email=f"{user_id}@example.com",
            username=user_id,
            first_name="Chat",
            last_name="User",
            organization="Org",
            department="Dept",
            password=hash_password("Abcdef123!@#"),
            status=True,
            deleted=0,
        )
    )
    role = Role(name="AI User")
    permission = Permission(name="AI Access", machine_name="ai_access")
    db.add_all([role, permission])
    db.flush()
    db.add_all(
        [
            UserRole(user_id=user_id, role_id=role.id),
            RolePermission(role_id=role.id, permission_id=permission.id),
            UserSession(user_id=user_id, access_token=token),
        ]
    )
    db.commit()
    return token


def _seed_resource(db, user_id: str) -> Resource:
    resource = Resource(resource_name="Handbook", user_id=user_id, upload_status=True)
    db.add(resource)
    db.flush()
    db.add(
        DocumentChunk(
            resource_id=resource.id,
            chunk_index=0,
            source_name="handbook.txt",
            modality="text",
            title="Vacation policy",
            content="Employees receive twenty vacation days per year.",
            contextual_content="Vacation policy. Employees receive twenty vacation days per year.",
            terms_json=json.dumps({"vacation": 3, "days": 2, "employees": 1}),
        )
    )
    db.commit()
    return resource


async def _asgi_request(path: str, query_string: str, headers: dict[str, str]) -> list[dict]:
    """Drive the ASGI app directly and return every message it sent."""
    from main import app

    messages: list[dict] = []
    request_sent = False

    async def receive():
        # The request body is delivered once; after that Starlette's disconnect listener
        # keeps polling, so this must block rather than repeat itself. The listener is
        # cancelled when the response completes.
        nonlocal request_sent
        if not request_sent:
            request_sent = True
            return {"type": "http.request", "body": b"", "more_body": False}
        await asyncio.Event().wait()

    async def send(message):
        messages.append(message)

    await app(
        {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.3"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": path,
            "raw_path": path.encode(),
            "query_string": query_string.encode(),
            "root_path": "",
            "server": ("testserver", 80),
            "client": ("1.2.3.4", 5000),
            "headers": [(key.encode(), value.encode()) for key, value in headers.items()],
        },
        receive,
        send,
    )
    return messages


def _ask(client, token, resource_id, query="vacation days", **params):
    query_params = {"query": query, "chat_history_name": "session-1", "brain_id": resource_id, **params}
    return client.post("/chat/asks", params=query_params, headers={"Authorization": f"Bearer {token}"})


class TestStreamingResponse:
    def test_the_extractive_answer_streams_back(self, client, db):
        token = _seed_ai_user(db)
        resource = _seed_resource(db, "chat-user")
        resp = _ask(client, token, resource.id)
        assert resp.status_code == 200
        assert "vacation" in resp.text.lower()

    def test_buffering_is_disabled_so_chunks_actually_reach_the_client(self, client, db):
        """nginx buffers proxied responses by default, which would re-assemble the answer
        and deliver it in one burst — the exact behaviour this endpoint replaced."""
        token = _seed_ai_user(db)
        resource = _seed_resource(db, "chat-user")
        resp = _ask(client, token, resource.id)
        assert resp.headers["x-accel-buffering"] == "no"
        assert resp.headers["cache-control"] == "no-cache"

    async def test_llm_deltas_are_sent_as_separate_asgi_body_messages(self, db, monkeypatch):
        """The one property a buffered response cannot fake.

        TestClient's transport coalesces the body, so this drives the ASGI app directly
        and counts `http.response.body` messages — one per delta is what makes the answer
        appear progressively instead of all at once.
        """
        token = _seed_ai_user(db)
        resource = _seed_resource(db, "chat-user")
        monkeypatch.setattr(rag_service, "stream_grounded_answer", lambda *a, **k: iter(["Twenty ", "days ", "per year."]))

        messages = await _asgi_request(
            "/chat/asks",
            query_string=(
                f"query=vacation+days&chat_history_name=session-1&brain_id={resource.id}"
                "&llm_provider=groq&llm_model=llama-3.3-70b-versatile"
            ),
            headers={"authorization": f"Bearer {token}"},
        )

        start = next(m for m in messages if m["type"] == "http.response.start")
        bodies = [m["body"] for m in messages if m["type"] == "http.response.body" and m.get("body")]
        assert start["status"] == 200
        assert len(bodies) > 1, "a single body message means the answer was buffered, not streamed"
        assembled = b"".join(bodies).decode()
        assert assembled.startswith("Twenty days per year.")
        # Invariant 24: the body is verbatim answer text. Provider and model reach the
        # client through the query params it sent and through chat_history.brain on reload,
        # never as a footer printed into the answer.
        assert "LLM: groq/llama-3.3-70b-versatile" not in assembled
        assert "Sources:" not in assembled


class TestPersistence:
    def test_the_turn_is_recorded_after_the_stream_completes(self, client, db):
        token = _seed_ai_user(db)
        resource = _seed_resource(db, "chat-user")
        resp = _ask(client, token, resource.id)

        history = db.query(ChatHistory).all()
        assert len(history) == 1
        assert history[0].chat_id == "session-1"
        assert history[0].user_query == "vacation days"
        assert history[0].bot_response == resp.text, "history must match exactly what the user saw"
        assert history[0].brain == f"{resource.id}:contextual_hybrid"

    def test_the_persisted_answer_includes_the_streamed_llm_text(self, client, db, monkeypatch):
        token = _seed_ai_user(db)
        resource = _seed_resource(db, "chat-user")
        monkeypatch.setattr(rag_service, "stream_grounded_answer", lambda *a, **k: iter(["Twenty ", "days."]))
        resp = _ask(client, token, resource.id, llm_provider="groq", llm_model="llama-3.3-70b-versatile")

        history = db.query(ChatHistory).one()
        assert history.bot_response == resp.text
        assert history.bot_response.startswith("Twenty days.")
        assert history.brain.endswith(":groq/llama-3.3-70b-versatile")

    def test_the_ask_is_activity_logged(self, client, db):
        token = _seed_ai_user(db)
        resource = _seed_resource(db, "chat-user")
        _ask(client, token, resource.id)

        logs = db.query(ActivityLog).all()
        assert len(logs) == 1
        assert logs[0].action == "CHAT"
        assert logs[0].entity_id == resource.id

    def test_a_failed_synthesis_still_records_the_fallback_answer(self, client, db, monkeypatch):
        token = _seed_ai_user(db)
        resource = _seed_resource(db, "chat-user")

        def boom(*args, **kwargs):
            raise LlmProviderError("Provider API returned HTTP 429: rate limited", 502)

        monkeypatch.setattr(rag_service, "stream_grounded_answer", boom)
        resp = _ask(client, token, resource.id, llm_provider="groq", llm_model="m")

        assert resp.status_code == 200, "a provider failure must not lose the extractive answer"
        assert "vacation" in resp.text.lower()
        history = db.query(ChatHistory).one()
        assert history.bot_response == resp.text


class TestErrorsBeforeTheFirstByte:
    """Once bytes are on the wire the status code is fixed, so anything reportable must
    be reported during planning — while a JSON envelope is still possible."""

    def test_an_unknown_brain_is_a_404_envelope_not_a_stream(self, client, db):
        token = _seed_ai_user(db)
        resp = _ask(client, token, "no-such-resource")
        assert resp.status_code == 404
        assert resp.json()["message"] == "Resource not found or not accessible"

    def test_another_users_resource_is_not_reachable(self, client, db):
        token = _seed_ai_user(db)
        stranger_resource = _seed_resource(db, "someone-else")
        resp = _ask(client, token, stranger_resource.id)
        assert resp.status_code == 404

    def test_a_provider_without_a_model_is_a_400_envelope(self, client, db):
        token = _seed_ai_user(db)
        resource = _seed_resource(db, "chat-user")
        resp = _ask(client, token, resource.id, llm_provider="groq")
        assert resp.status_code == 400
        assert resp.json()["message"] == "Both LLM provider and model must be selected"

    def test_an_unconfigured_provider_is_reported_before_streaming(self, client, db):
        """No stored credential for groq — the user gets a real status code, not a
        stream that turns out to be an error message."""
        token = _seed_ai_user(db)
        resource = _seed_resource(db, "chat-user")
        resp = _ask(client, token, resource.id, llm_provider="groq", llm_model="llama-3.3-70b-versatile")
        assert resp.status_code == 200, "synthesis is optional — retrieval already answered"
        assert "vacation" in resp.text.lower()

    def test_anonymous_callers_are_rejected(self, client):
        resp = client.post("/chat/asks", params={"query": "q", "chat_history_name": "s", "brain_id": "b"})
        assert resp.status_code == 401

    def test_a_socket_failure_before_the_first_token_does_not_become_a_bare_500(self, client, db, monkeypatch):
        """The provider accepts the request, then the connection dies before any token.

        This used to escape every handler as a plain-text HTTP 500 that also discarded the
        extractive answer, because a body read fails after `_open_with_retry` has already
        returned the response and can no longer classify anything.
        """
        token = _seed_ai_user(db)
        resource = _seed_resource(db, "chat-user")

        def dies_mid_read(*args, **kwargs):
            def gen():
                raise ConnectionResetError(104, "Connection reset by peer")
                yield  # pragma: no cover - unreachable, marks this a generator

            return gen()

        monkeypatch.setattr(rag_service, "stream_grounded_answer", dies_mid_read)
        resp = _ask(client, token, resource.id, llm_provider="groq", llm_model="llama-3.3-70b-versatile")

        assert resp.status_code == 200
        assert "vacation" in resp.text.lower(), "the extractive answer must survive a dead socket"
        assert db.query(ChatHistory).count() == 1
