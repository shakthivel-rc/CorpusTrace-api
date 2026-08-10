"""WS /chat/ws — the chat transport the SPA prefers.

Three things separate this from `POST /chat/asks` and they are what these tests guard:
the connection authenticates itself from its first frame (no middleware ever sees a
websocket scope), the provider call does **not** stream so the answer is complete before
the first `delta` leaves, and an error can be reported *after* work has started because a
frame is not a status line.
"""
import json
import threading
import time
from contextlib import contextmanager

import jwt
import pytest

import controllers.chat_ws_controller as chat_ws
import rag.service as rag_service
from controllers.chat_ws_controller import (
    WS_FORBIDDEN,
    WS_UNAUTHENTICATED,
    split_into_deltas,
)
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


def _seed_user(db, user_id: str = "ws-user", *, permission: str | None = "ai_access") -> str:
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
            first_name="Socket",
            last_name="User",
            organization="Org",
            department="Dept",
            password=hash_password("Abcdef123!@#"),
            status=True,
            deleted=0,
        )
    )
    role = Role(name=f"Role {user_id}")
    db.add(role)
    db.flush()
    rows = [UserRole(user_id=user_id, role_id=role.id), UserSession(user_id=user_id, access_token=token)]
    if permission:
        slug = Permission(name=permission, machine_name=permission)
        db.add(slug)
        db.flush()
        rows.append(RolePermission(role_id=role.id, permission_id=slug.id))
    db.add_all(rows)
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


@contextmanager
def _authenticated(client, token):
    """An open, authenticated socket. Fails loudly rather than yielding a dead one."""
    with client.websocket_connect("/chat/ws") as socket:
        socket.send_json({"type": "auth", "token": token})
        assert socket.receive_json() == {"type": "ready"}
        yield socket


def _drain(socket) -> list[dict]:
    """Every frame of one answer, up to and including its terminal frame."""
    frames = []
    while True:
        frame = socket.receive_json()
        frames.append(frame)
        if frame["type"] in {"done", "cancelled", "error"}:
            return frames


def _answer(frames: list[dict]) -> str:
    return "".join(frame["text"] for frame in frames if frame["type"] == "delta")


def _ask(socket, resource_id, query="vacation days", **extra):
    socket.send_json({"type": "ask", "query": query, "brain_id": resource_id, "chat_history_name": "session-1", **extra})
    return _drain(socket)


class TestAuthentication:
    def test_a_valid_token_in_the_first_frame_is_accepted(self, client, db):
        token = _seed_user(db)
        with client.websocket_connect("/chat/ws") as socket:
            socket.send_json({"type": "auth", "token": token})
            assert socket.receive_json() == {"type": "ready"}

    def test_a_garbage_token_closes_the_socket(self, client, db):
        _seed_user(db)
        with client.websocket_connect("/chat/ws") as socket:
            socket.send_json({"type": "auth", "token": "not-a-jwt"})
            assert socket.receive_json() == {"type": "error", "status_code": 401, "message": "Invalid token"}
            with pytest.raises(Exception):
                socket.receive_json()

    def test_a_revoked_session_is_rejected_even_with_a_valid_signature(self, client, db):
        """The whole point of the `user_sessions` lookup: signing out must take effect
        immediately, and a socket that skipped this check would outlive a logout."""
        token = _seed_user(db)
        db.query(UserSession).delete()
        db.commit()
        with client.websocket_connect("/chat/ws") as socket:
            socket.send_json({"type": "auth", "token": token})
            assert socket.receive_json()["message"] == "Invalid session or token revoked"

    def test_a_user_without_ai_access_is_refused(self, client, db):
        token = _seed_user(db, permission=None)
        with client.websocket_connect("/chat/ws") as socket:
            socket.send_json({"type": "auth", "token": token})
            frame = socket.receive_json()
            assert frame == {"type": "error", "status_code": 403, "message": "Insufficient permissions"}

    def test_a_first_frame_that_is_not_auth_is_refused(self, client, db):
        token = _seed_user(db)
        resource = _seed_resource(db, "ws-user")
        with client.websocket_connect("/chat/ws") as socket:
            socket.send_json({"type": "ask", "query": "vacation days", "brain_id": resource.id})
            assert socket.receive_json()["status_code"] == 401
            # And nothing was answered.
            with pytest.raises(Exception):
                socket.receive_json()

    def test_close_codes_distinguish_unauthenticated_from_forbidden(self):
        """4401/4403 are what a client reads to decide whether refreshing its token could
        possibly help; collapsing them to one code makes that undecidable."""
        assert WS_UNAUTHENTICATED != WS_FORBIDDEN


class TestAnswering:
    def test_the_extractive_answer_arrives_as_deltas(self, client, db):
        token = _seed_user(db)
        resource = _seed_resource(db, "ws-user")
        with _authenticated(client, token) as socket:
            frames = _ask(socket, resource.id)

        assert [frame["type"] for frame in frames[:2]] == ["accepted", "status"]
        assert frames[1]["stage"] == "retrieving"
        assert frames[-1]["type"] == "done"
        assert "vacation" in _answer(frames).lower()

    def test_the_answer_is_delivered_in_more_than_one_frame(self, client, db, monkeypatch):
        """A single `delta` would mean the socket is doing nothing the old endpoint did
        not, which is the one property that cannot be faked by a buffered response."""
        token = _seed_user(db)
        resource = _seed_resource(db, "ws-user")
        monkeypatch.setattr(
            rag_service,
            "generate_grounded_answer",
            lambda *a, **k: "Twenty vacation days per year, accrued monthly from the start date.",
        )
        with _authenticated(client, token) as socket:
            frames = _ask(socket, resource.id, llm_provider="groq", llm_model="llama-3.3-70b-versatile")

        deltas = [frame for frame in frames if frame["type"] == "delta"]
        assert len(deltas) > 1
        assert _answer(frames) == "Twenty vacation days per year, accrued monthly from the start date."

    def test_the_provider_is_called_without_streaming(self, client, db, monkeypatch):
        """The behavioural half of "the LLM response is not streamed": this path must go
        through the blocking completion call, never the streaming one."""
        token = _seed_user(db)
        resource = _seed_resource(db, "ws-user")
        calls: list[str] = []

        def blocking(*args, **kwargs):
            calls.append("blocking")
            return "Twenty days."

        def streaming(*args, **kwargs):  # pragma: no cover - reaching this is the failure
            calls.append("streaming")
            return iter(["Twenty ", "days."])

        monkeypatch.setattr(rag_service, "generate_grounded_answer", blocking)
        monkeypatch.setattr(rag_service, "stream_grounded_answer", streaming)
        with _authenticated(client, token) as socket:
            _ask(socket, resource.id, llm_provider="groq", llm_model="m")

        assert calls == ["blocking"]

    def test_citations_arrive_before_the_answer_they_annotate(self, client, db):
        token = _seed_user(db)
        resource = _seed_resource(db, "ws-user")
        with _authenticated(client, token) as socket:
            frames = _ask(socket, resource.id)

        types = [frame["type"] for frame in frames]
        assert "citations" in types
        assert types.index("citations") < types.index("delta")
        citations = frames[types.index("citations")]["citations"]
        assert citations[0]["source_name"] == "handbook.txt"
        # Unlike the header, a frame has no size cap to stay under, so the snippet the
        # evidence panel would otherwise re-fetch is already here.
        assert "snippet" in citations[0]

    def test_a_generating_stage_is_reported_only_when_a_model_writes_the_answer(self, client, db, monkeypatch):
        token = _seed_user(db)
        resource = _seed_resource(db, "ws-user")

        with _authenticated(client, token) as socket:
            extractive = _ask(socket, resource.id)
        assert "generating" not in [frame.get("stage") for frame in extractive]

        monkeypatch.setattr(rag_service, "generate_grounded_answer", lambda *a, **k: "Twenty days.")
        with _authenticated(client, token) as socket:
            synthesized = _ask(socket, resource.id, llm_provider="groq", llm_model="m")
        assert "generating" in [frame.get("stage") for frame in synthesized]

    def test_the_socket_answers_more_than_one_question(self, client, db):
        """The reason this is a socket and not a request: the connection outlives a turn."""
        token = _seed_user(db)
        resource = _seed_resource(db, "ws-user")
        with _authenticated(client, token) as socket:
            first = _ask(socket, resource.id, query="vacation days")
            second = _ask(socket, resource.id, query="how many days")

        assert first[-1]["type"] == "done"
        assert second[-1]["type"] == "done"
        assert db.query(ChatHistory).count() == 2

    def test_a_ping_is_answered_with_a_pong(self, client, db):
        token = _seed_user(db)
        with _authenticated(client, token) as socket:
            socket.send_json({"type": "ping"})
            assert socket.receive_json() == {"type": "pong"}


class TestCancellation:
    """Stopping is the thing an HTTP stream can only express by aborting the request.

    Both tests hold the provider call open on a real threading.Event so the socket is
    genuinely mid-answer when the second frame arrives — the state that matters here does
    not exist for any length of time otherwise.
    """

    @staticmethod
    def _block_generation(monkeypatch) -> threading.Event:
        released = threading.Event()

        def slow(*args, **kwargs):
            released.wait(5)
            return "Twenty vacation days per year, accrued monthly from the start date."

        monkeypatch.setattr(rag_service, "generate_grounded_answer", slow)
        return released

    @staticmethod
    def _ask_and_reach_generating(socket, resource_id, **extra):
        socket.send_json(
            {
                "type": "ask",
                "query": "vacation days",
                "brain_id": resource_id,
                "llm_provider": "groq",
                "llm_model": "m",
                **extra,
            }
        )
        frames = [socket.receive_json() for _ in range(4)]
        assert frames[-1]["stage"] == "generating"
        return frames

    def test_a_cancel_mid_answer_stops_delivery_and_records_nothing(self, client, db, monkeypatch):
        token = _seed_user(db)
        resource = _seed_resource(db, "ws-user")
        released = self._block_generation(monkeypatch)
        try:
            with _authenticated(client, token) as socket:
                accepted = self._ask_and_reach_generating(socket, resource.id)[0]
                socket.send_json({"type": "cancel"})
                # Let the receive loop see the cancel before generation returns: the flag
                # is read between deltas, so it must be set before the first one.
                time.sleep(0.2)
                released.set()
                terminal = socket.receive_json()
        finally:
            released.set()

        assert terminal == {"type": "cancelled", "request_id": accepted["request_id"]}
        # Nothing was delivered, so there is no turn the user saw and nothing to record.
        assert db.query(ChatHistory).count() == 0

    def test_a_second_question_while_one_is_in_flight_is_refused(self, client, db, monkeypatch):
        """One answer per connection at a time. Interleaving two would put both answers'
        deltas on the same wire with nothing but `request_id` to tell them apart."""
        token = _seed_user(db)
        resource = _seed_resource(db, "ws-user")
        released = self._block_generation(monkeypatch)
        try:
            with _authenticated(client, token) as socket:
                self._ask_and_reach_generating(socket, resource.id)
                socket.send_json({"type": "ask", "query": "again", "brain_id": resource.id})
                frame = socket.receive_json()
                assert frame["type"] == "error"
                assert frame["status_code"] == 409
        finally:
            released.set()


class TestErrors:
    def test_an_unknown_brain_is_an_error_frame_not_a_closed_socket(self, client, db):
        """The property the HTTP path cannot have: the failure is reported *and* the
        connection survives, so the next question does not pay for a new handshake."""
        token = _seed_user(db)
        resource = _seed_resource(db, "ws-user")
        with _authenticated(client, token) as socket:
            frames = _ask(socket, "no-such-resource")
            assert frames[-1] == {
                **frames[-1],
                "type": "error",
                "status_code": 404,
                "message": "Resource not found or not accessible",
            }
            assert _ask(socket, resource.id)[-1]["type"] == "done"

    def test_another_users_resource_is_not_reachable(self, client, db):
        token = _seed_user(db)
        stranger = _seed_resource(db, "someone-else")
        with _authenticated(client, token) as socket:
            assert _ask(socket, stranger.id)[-1]["status_code"] == 404

    def test_a_provider_without_a_model_is_a_400(self, client, db):
        token = _seed_user(db)
        resource = _seed_resource(db, "ws-user")
        with _authenticated(client, token) as socket:
            frames = _ask(socket, resource.id, llm_provider="groq")
        assert frames[-1]["status_code"] == 400
        assert frames[-1]["message"] == "Both LLM provider and model must be selected"

    def test_an_empty_question_is_rejected_before_any_work(self, client, db):
        token = _seed_user(db)
        resource = _seed_resource(db, "ws-user")
        with _authenticated(client, token) as socket:
            socket.send_json({"type": "ask", "query": "   ", "brain_id": resource.id})
            assert socket.receive_json()["status_code"] == 400

    def test_a_malformed_frame_does_not_kill_the_conversation(self, client, db):
        token = _seed_user(db)
        resource = _seed_resource(db, "ws-user")
        with _authenticated(client, token) as socket:
            socket.send_text("{ not json")
            assert socket.receive_json()["status_code"] == 400
            assert _ask(socket, resource.id)[-1]["type"] == "done"

    def test_a_provider_failure_still_delivers_the_extractive_answer(self, client, db, monkeypatch):
        """A rate limit must not cost the answer retrieval already produced — the same
        guarantee the HTTP path makes, reached through the blocking call instead."""
        token = _seed_user(db)
        resource = _seed_resource(db, "ws-user")

        def boom(*args, **kwargs):
            raise LlmProviderError("Provider API returned HTTP 429: rate limited", 502)

        monkeypatch.setattr(rag_service, "generate_grounded_answer", boom)
        with _authenticated(client, token) as socket:
            frames = _ask(socket, resource.id, llm_provider="groq", llm_model="m")

        assert frames[-1]["type"] == "done"
        assert "vacation" in _answer(frames).lower()
        assert db.query(ChatHistory).one().bot_response == _answer(frames)


class TestPersistence:
    def test_the_turn_is_recorded_with_exactly_what_was_delivered(self, client, db):
        token = _seed_user(db)
        resource = _seed_resource(db, "ws-user")
        with _authenticated(client, token) as socket:
            frames = _ask(socket, resource.id)

        history = db.query(ChatHistory).one()
        assert history.chat_id == "session-1"
        assert history.user_query == "vacation days"
        assert history.bot_response == _answer(frames), "history must match what the user saw"
        assert history.brain == f"{resource.id}:contextual_hybrid"

    def test_the_provider_and_model_are_recorded_on_the_brain_label(self, client, db, monkeypatch):
        token = _seed_user(db)
        resource = _seed_resource(db, "ws-user")
        monkeypatch.setattr(rag_service, "generate_grounded_answer", lambda *a, **k: "Twenty days.")
        with _authenticated(client, token) as socket:
            _ask(socket, resource.id, llm_provider="groq", llm_model="llama-3.3-70b-versatile")

        assert db.query(ChatHistory).one().brain.endswith(":groq/llama-3.3-70b-versatile")

    def test_the_ask_is_activity_logged(self, client, db):
        token = _seed_user(db)
        resource = _seed_resource(db, "ws-user")
        with _authenticated(client, token) as socket:
            _ask(socket, resource.id)

        logs = db.query(ActivityLog).all()
        assert len(logs) == 1
        assert logs[0].action == "CHAT"
        assert logs[0].entity_id == resource.id

    def test_the_turn_lands_before_the_done_frame(self, client, db, monkeypatch):
        """A client that reloads its history on `done` must not race the write."""
        token = _seed_user(db)
        resource = _seed_resource(db, "ws-user")
        seen: list[int] = []
        original = chat_ws.persist_chat_turn

        def spy(**kwargs):
            original(**kwargs)
            seen.append(1)

        monkeypatch.setattr(chat_ws, "persist_chat_turn", spy)
        with _authenticated(client, token) as socket:
            frames = _ask(socket, resource.id)
            assert frames[-1]["type"] == "done"
            assert seen == [1]


class TestDeltaSplitting:
    """`split_into_deltas` is the only place the delivered text can diverge from the
    answer that was generated and persisted."""

    @pytest.mark.parametrize(
        "text",
        [
            "",
            "short",
            "Twenty vacation days per year.",
            "word " * 500,
            "nospacesatallinthisentirestring" * 40,
            "  leading and trailing  ",
            "多字节字符也必须完整保留" * 30,
        ],
    )
    def test_joining_the_deltas_reproduces_the_answer(self, text):
        assert "".join(split_into_deltas(text)) == text

    def test_an_empty_answer_produces_no_frames(self):
        assert split_into_deltas("") == []

    def test_a_long_answer_is_capped_so_it_still_finishes_painting(self):
        deltas = split_into_deltas("word " * 5000)
        assert len(deltas) <= chat_ws.MAX_DELTAS

    def test_every_delta_is_non_empty(self):
        """A zero-length frame would spin the delivery loop without progress."""
        assert all(split_into_deltas("word " * 400))
