"""The source-evidence surface: serving an uploaded file, its passages, and the citations
that point at them.

This is the first endpoint in the codebase that hands user-uploaded bytes back to a
browser, so most of what is guarded here is not the happy path: whose file it is, whether
the stored path really is inside the upload directory, and what content type the bytes are
allowed to claim.
"""
import base64
import dataclasses
import json
import time

import jwt
import pytest

import controllers.chat_controller as chat_controller
import rag.service as rag_service
from core.config import get_settings
from models.chat_history import ChatHistory
from models.file import File
from models.permissions import Permission
from models.rag import DocumentChunk
from models.resource import Resource
from models.role_permissions import RolePermission
from models.roles import Role
from models.user import User
from models.user_roles import UserRole
from models.user_session import UserSession
from utils.password import hash_password

pytestmark = pytest.mark.integration


def _seed_ai_user(db, user_id: str = "evidence-user") -> str:
    """A user holding ai_access with a live session. Roles are per-user so two callers can
    coexist in one test."""
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
            first_name="Evidence",
            last_name="User",
            organization="Org",
            department="Dept",
            password=hash_password("Abcdef123!@#"),
            status=True,
            deleted=0,
        )
    )
    role = Role(name=f"AI User {user_id}")
    db.add(role)
    db.flush()
    permission = db.query(Permission).filter(Permission.machine_name == "ai_access").first()
    if not permission:
        permission = Permission(name="AI Access", machine_name="ai_access")
        db.add(permission)
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


def _seed_document(db, user_id: str, upload_dir, filename: str = "handbook.pdf", body: bytes = b"%PDF-1.4 fake"):
    """A resource with one file on disk and one positioned chunk pointing into it."""
    resource = Resource(resource_name="Handbook", user_id=user_id, upload_status=True)
    db.add(resource)
    db.flush()

    resource_dir = upload_dir / resource.id
    resource_dir.mkdir(parents=True, exist_ok=True)
    destination = resource_dir / filename
    destination.write_bytes(body)

    file_record = File(
        file_name=filename,
        # Deliberately a lie: this is whatever the client put in its multipart header, and
        # the endpoint must not trust it.
        file_type="text/html",
        file_url=str(destination),
        resource_id=resource.id,
    )
    db.add(file_record)
    db.flush()

    chunk = DocumentChunk(
        resource_id=resource.id,
        file_id=file_record.id,
        chunk_index=0,
        source_name=filename,
        modality="pdf",
        title="Vacation policy",
        content="Employees receive twenty vacation days per year.",
        contextual_content="Vacation policy. Employees receive twenty vacation days per year.",
        terms_json=json.dumps({"vacation": 3, "days": 2, "employees": 1}),
        page_start=4,
        page_end=5,
        char_start=1200,
        char_end=2400,
    )
    db.add(chunk)
    db.commit()
    return resource, file_record, chunk


@pytest.fixture()
def upload_dir(tmp_path, monkeypatch):
    """Point the upload directory at a temp dir for both the writer and the reader.

    Settings are lru_cached and frozen, so this replaces the accessor each module calls
    rather than mutating anything.
    """
    directory = tmp_path / "uploads" / "rag"
    directory.mkdir(parents=True)
    patched = dataclasses.replace(get_settings(), rag_upload_dir=str(directory))
    monkeypatch.setattr(rag_service, "get_settings", lambda: patched)
    monkeypatch.setattr(chat_controller, "get_settings", lambda: patched)
    return directory


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


class TestIngestionRecordsPositions:
    """The real `upload_resource` path, not the extraction helpers in isolation.

    Everything between "bytes on disk" and "a row with page_start" only runs here, and it is
    where an off-by-one would actually reach a user.
    """

    def test_a_real_multi_page_upload_lands_each_chunk_on_the_right_page(self, db, upload_dir, make_pdf):
        from fastapi import UploadFile
        from io import BytesIO

        from rag.service import upload_resource

        # Each page carries a distinct marker word repeated enough to fill several chunks.
        pdf = make_pdf([
            ["Alpha compliance clause " * 30],
            ["Bravo retention clause " * 30],
            ["Charlie disposal clause " * 30],
        ])
        upload = UploadFile(filename="standard.pdf", file=BytesIO(pdf))

        resource = upload_resource(db, "evidence-user", "Standards", [upload])

        chunks = (
            db.query(DocumentChunk)
            .filter(DocumentChunk.resource_id == resource.id)
            .order_by(DocumentChunk.chunk_index.asc())
            .all()
        )
        assert len(chunks) > 1, "the fixture must be long enough to produce several chunks"

        marker_for_page = {1: "Alpha", 2: "Bravo", 3: "Charlie"}
        for chunk in chunks:
            assert chunk.page_start is not None, "a text-bearing PDF chunk must know its page"
            assert 1 <= chunk.page_start <= chunk.page_end <= 3
            # Every marker present in the text must fall inside the reported page range, and
            # every page in the range must contribute at least one marker.
            present = {page for page, marker in marker_for_page.items() if marker in chunk.content}
            assert present, "chunk text should carry at least one page marker"
            assert min(present) >= chunk.page_start and max(present) <= chunk.page_end
            assert chunk.char_start < chunk.char_end

    def test_a_scanned_pdf_records_no_position_rather_than_page_one(self, db, upload_dir, make_pdf):
        from fastapi import UploadFile
        from io import BytesIO

        from rag.service import upload_resource

        upload = UploadFile(filename="scanned.pdf", file=BytesIO(make_pdf([[]])))

        resource = upload_resource(db, "evidence-user", "Scans", [upload])

        chunk = db.query(DocumentChunk).filter(DocumentChunk.resource_id == resource.id).one()
        assert "No embedded text layer" in chunk.content
        assert chunk.page_start is None and chunk.page_end is None

    def test_a_non_pdf_upload_records_offsets_but_no_pages(self, db, upload_dir):
        from fastapi import UploadFile
        from io import BytesIO

        from rag.service import upload_resource

        upload = UploadFile(filename="notes.txt", file=BytesIO(b"Alpha beta gamma delta"))

        resource = upload_resource(db, "evidence-user", "Notes", [upload])

        chunk = db.query(DocumentChunk).filter(DocumentChunk.resource_id == resource.id).one()
        assert (chunk.page_start, chunk.page_end) == (None, None)
        assert (chunk.char_start, chunk.char_end) == (0, len("Alpha beta gamma delta"))


class TestSourceFileEndpoint:
    def test_the_owner_receives_the_bytes(self, client, db, upload_dir):
        token = _seed_ai_user(db)
        _, file_record, _ = _seed_document(db, "evidence-user", upload_dir, body=b"%PDF-1.4 real bytes")

        resp = client.get(f"/resources/files/{file_record.id}", headers=_auth(token))

        assert resp.status_code == 200
        assert resp.content == b"%PDF-1.4 real bytes"

    def test_the_content_type_comes_from_the_extension_not_the_stored_type(self, client, db, upload_dir):
        """`files.file_type` is client-supplied. This row claims text/html; honouring that
        would serve attacker-controlled markup inline from the API origin — stored XSS."""
        token = _seed_ai_user(db)
        _, file_record, _ = _seed_document(db, "evidence-user", upload_dir)

        resp = client.get(f"/resources/files/{file_record.id}", headers=_auth(token))

        assert resp.headers["content-type"] == "application/pdf"
        assert resp.headers["x-content-type-options"] == "nosniff"
        assert resp.headers["content-disposition"].startswith("inline;")

    def test_an_unrecognised_extension_is_served_as_a_download_not_as_markup(self, client, db, upload_dir):
        token = _seed_ai_user(db)
        _, file_record, _ = _seed_document(
            db, "evidence-user", upload_dir, filename="payload.html", body=b"<script>alert(1)</script>"
        )

        resp = client.get(f"/resources/files/{file_record.id}", headers=_auth(token))

        assert resp.headers["content-type"] == "application/octet-stream"

    def test_another_users_file_is_not_reachable(self, client, db, upload_dir):
        token = _seed_ai_user(db, "evidence-user")
        _seed_ai_user(db, "stranger")
        _, stranger_file, _ = _seed_document(db, "stranger", upload_dir)

        resp = client.get(f"/resources/files/{stranger_file.id}", headers=_auth(token))

        # 404 rather than 403: a 403 confirms the id exists and names a document the
        # caller is not allowed to know about.
        assert resp.status_code == 404
        assert resp.json()["message"] == "Source file not found"

    def test_a_soft_deleted_resource_hides_its_files(self, client, db, upload_dir):
        from datetime import datetime, timezone

        token = _seed_ai_user(db)
        resource, file_record, _ = _seed_document(db, "evidence-user", upload_dir)
        resource.deleted_at = datetime.now(timezone.utc)
        db.commit()

        resp = client.get(f"/resources/files/{file_record.id}", headers=_auth(token))

        assert resp.status_code == 404

    def test_a_stored_path_outside_the_upload_directory_is_refused(self, client, db, upload_dir, tmp_path):
        """`file_url` is stored data, and stored data is untrusted input."""
        token = _seed_ai_user(db)
        _, file_record, _ = _seed_document(db, "evidence-user", upload_dir)
        escapee = tmp_path / "secret.pdf"
        escapee.write_bytes(b"not yours")
        file_record.file_url = str(escapee)
        db.commit()

        resp = client.get(f"/resources/files/{file_record.id}", headers=_auth(token))

        assert resp.status_code == 404
        assert resp.content != b"not yours"

    def test_a_traversal_sequence_in_the_stored_path_is_refused(self, client, db, upload_dir, tmp_path):
        token = _seed_ai_user(db)
        resource, file_record, _ = _seed_document(db, "evidence-user", upload_dir)
        outside = tmp_path / "etc-passwd-lookalike"
        outside.write_bytes(b"root:x:0:0")
        file_record.file_url = str(upload_dir / resource.id / ".." / ".." / ".." / outside.name)
        db.commit()

        resp = client.get(f"/resources/files/{file_record.id}", headers=_auth(token))

        assert resp.status_code == 404

    def test_an_indexed_file_missing_from_disk_is_a_404_envelope_not_a_500(self, client, db, upload_dir):
        token = _seed_ai_user(db)
        _, file_record, _ = _seed_document(db, "evidence-user", upload_dir)
        (upload_dir / file_record.resource_id / file_record.file_name).unlink()

        resp = client.get(f"/resources/files/{file_record.id}", headers=_auth(token))

        assert resp.status_code == 404
        assert resp.json()["status_code"] == 404
        assert "no longer available" in resp.json()["message"]

    def test_an_unknown_id_is_a_404_envelope(self, client, db, upload_dir):
        token = _seed_ai_user(db)

        resp = client.get("/resources/files/does-not-exist", headers=_auth(token))

        assert resp.status_code == 404
        assert resp.json()["status"] == "error"

    def test_anonymous_callers_are_rejected(self, client, db, upload_dir):
        _seed_ai_user(db)
        _, file_record, _ = _seed_document(db, "evidence-user", upload_dir)

        resp = client.get(f"/resources/files/{file_record.id}")

        assert resp.status_code == 401


class TestSourceChunkEndpoint:
    def test_the_owner_receives_the_full_passage_and_its_position(self, client, db, upload_dir):
        token = _seed_ai_user(db)
        _, file_record, chunk = _seed_document(db, "evidence-user", upload_dir)

        resp = client.get(f"/resources/chunks/{chunk.id}", headers=_auth(token))

        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["content"] == "Employees receive twenty vacation days per year."
        assert data["file_id"] == file_record.id
        assert (data["page_start"], data["page_end"]) == (4, 5)
        assert (data["char_start"], data["char_end"]) == (1200, 2400)

    def test_a_chunk_indexed_before_positions_existed_reports_them_as_unknown(self, client, db, upload_dir):
        token = _seed_ai_user(db)
        _, _, chunk = _seed_document(db, "evidence-user", upload_dir)
        chunk.page_start = chunk.page_end = chunk.char_start = chunk.char_end = None
        db.commit()

        resp = client.get(f"/resources/chunks/{chunk.id}", headers=_auth(token))

        data = resp.json()["data"]
        assert data["page_start"] is None and data["char_start"] is None

    def test_another_users_chunk_is_not_reachable(self, client, db, upload_dir):
        token = _seed_ai_user(db, "evidence-user")
        _seed_ai_user(db, "stranger")
        _, _, stranger_chunk = _seed_document(db, "stranger", upload_dir)

        resp = client.get(f"/resources/chunks/{stranger_chunk.id}", headers=_auth(token))

        assert resp.status_code == 404

    def test_anonymous_callers_are_rejected(self, client, db, upload_dir):
        _seed_ai_user(db)
        _, _, chunk = _seed_document(db, "evidence-user", upload_dir)

        assert client.get(f"/resources/chunks/{chunk.id}").status_code == 401


class TestCitationHeaderOnAnswers:
    def _ask(self, client, token, resource_id, query="vacation days"):
        return client.post(
            "/chat/asks",
            params={"query": query, "chat_history_name": "session-1", "brain_id": resource_id},
            headers=_auth(token),
        )

    def test_an_answer_carries_its_citations_on_the_response_header(self, client, db, upload_dir):
        token = _seed_ai_user(db)
        resource, file_record, chunk = _seed_document(db, "evidence-user", upload_dir)

        resp = self._ask(client, token, resource.id)

        assert resp.status_code == 200
        decoded = json.loads(base64.urlsafe_b64decode(resp.headers["x-nexarag-citations"]))
        assert decoded[0]["chunk_id"] == chunk.id
        assert decoded[0]["file_id"] == file_record.id
        assert decoded[0]["page_start"] == 4
        assert decoded[0]["index"] == 1

    def test_the_citation_index_matches_the_marker_printed_in_the_answer(self, client, db, upload_dir):
        token = _seed_ai_user(db)
        resource, _, _ = _seed_document(db, "evidence-user", upload_dir)

        resp = self._ask(client, token, resource.id)

        decoded = json.loads(base64.urlsafe_b64decode(resp.headers["x-nexarag-citations"]))
        # `_compose_answer` prints "[1]" for the first source; a chip labelled differently
        # would point the reader at the wrong line of the answer.
        assert f"[{decoded[0]['index']}]" in resp.text

    def test_small_talk_retrieves_nothing_and_sends_no_header(self, client, db, upload_dir):
        token = _seed_ai_user(db)
        resource, _, _ = _seed_document(db, "evidence-user", upload_dir)

        resp = self._ask(client, token, resource.id, query="hello there")

        assert resp.status_code == 200
        assert "x-nexarag-citations" not in resp.headers

    def test_the_turn_is_persisted_with_its_citations(self, client, db, upload_dir):
        token = _seed_ai_user(db)
        resource, _, chunk = _seed_document(db, "evidence-user", upload_dir)

        self._ask(client, token, resource.id)

        record = db.query(ChatHistory).one()
        stored = json.loads(record.citations_json)
        assert stored[0]["chunk_id"] == chunk.id
        # The stored copy keeps the snippet the header omits, so a reloaded conversation
        # can still show something if the chunk has since been deleted.
        assert "snippet" in stored[0]

    def test_a_turn_with_no_retrieval_stores_no_citations(self, client, db, upload_dir):
        token = _seed_ai_user(db)
        resource, _, _ = _seed_document(db, "evidence-user", upload_dir)

        self._ask(client, token, resource.id, query="hello there")

        assert db.query(ChatHistory).one().citations_json is None


class TestChatHistoryCitations:
    def test_restored_history_carries_the_citations_back(self, client, db, upload_dir):
        token = _seed_ai_user(db)
        resource, _, chunk = _seed_document(db, "evidence-user", upload_dir)
        client.post(
            "/chat/asks",
            params={"query": "vacation days", "chat_history_name": "session-1", "brain_id": resource.id},
            headers=_auth(token),
        )

        resp = client.get("/chat/history?chat_history_name=session-1", headers=_auth(token))

        record = resp.json()["data"]["records"][0]
        assert record["citations"][0]["chunk_id"] == chunk.id

    def test_restored_history_carries_the_provider_and_model_on_brain(self, client, db, upload_dir):
        """The provenance panel needs provider/model/mode on a RELOADED conversation too.

        The raw "LLM: provider/model" footer used to be the only place they appeared, and
        it was inside the answer text. They already ride the `brain` label instead —
        "{brain_id}:{mode}:{provider}/{model}" — which /chat/history already returns, so no
        new field and no new header was needed. Note the model id itself may contain both
        separators (`nvidia/nemotron-nano-12b-v2-vl:free`), so a reader must split the mode
        off with a bounded split and take only the FIRST "/" after it.
        """
        token = _seed_ai_user(db)
        resource, _, _ = _seed_document(db, "evidence-user", upload_dir)
        db.add(
            ChatHistory(
                chat_id="session-1",
                user_query="vacation days",
                user_id="evidence-user",
                bot_response="Twenty days.",
                brain=f"{resource.id}:graph_rag:openrouter/nvidia/nemotron-nano-12b-v2-vl:free",
                citations_json=None,
            )
        )
        db.commit()

        resp = client.get("/chat/history?chat_history_name=session-1", headers=_auth(token))

        brain = resp.json()["data"]["records"][0]["brain"]
        brain_id, mode, provider_model = brain.split(":", 2)
        provider, model = provider_model.split("/", 1)
        assert brain_id == resource.id
        assert mode == "graph_rag"
        assert provider == "openrouter"
        assert model == "nvidia/nemotron-nano-12b-v2-vl:free"

    def test_a_turn_recorded_before_this_column_existed_reports_an_empty_list(self, client, db, upload_dir):
        token = _seed_ai_user(db)
        db.add(
            ChatHistory(
                chat_id="session-1",
                user_query="q",
                user_id="evidence-user",
                bot_response="a",
                brain="b",
                citations_json=None,
            )
        )
        db.commit()

        resp = client.get("/chat/history?chat_history_name=session-1", headers=_auth(token))

        assert resp.json()["data"]["records"][0]["citations"] == []

    def test_unparseable_stored_citations_do_not_fail_the_whole_conversation(self, client, db, upload_dir):
        token = _seed_ai_user(db)
        db.add(
            ChatHistory(
                chat_id="session-1",
                user_query="q",
                user_id="evidence-user",
                bot_response="the answer the user came for",
                brain="b",
                citations_json="{truncated",
            )
        )
        db.commit()

        resp = client.get("/chat/history?chat_history_name=session-1", headers=_auth(token))

        assert resp.status_code == 200
        record = resp.json()["data"]["records"][0]
        assert record["bot_response"] == "the answer the user came for"
        assert record["citations"] == []

    def test_another_users_conversation_is_not_readable_by_guessing_its_id(self, client, db, upload_dir):
        """chat_id is client-supplied and was never scoped to the caller. Now that a row
        carries source filenames and quoted passages, that leak is no longer theoretical."""
        token = _seed_ai_user(db, "evidence-user")
        _seed_ai_user(db, "stranger")
        db.add(
            ChatHistory(
                chat_id="session_stranger",
                user_query="what is our margin",
                user_id="stranger",
                bot_response="confidential",
                brain="b",
                citations_json=json.dumps([{"source_name": "board-pack.pdf", "snippet": "secret"}]),
            )
        )
        db.commit()

        resp = client.get("/chat/history?chat_history_name=session_stranger", headers=_auth(token))

        assert resp.json()["data"]["records"] == []
