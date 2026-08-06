"""The HTTP surface of background indexing: upload, poll, cancel.

The contract these pin is what the SPA is written against. Two parts of it are easy to
break without noticing: that upload returns 202 rather than 201 (the base is not usable
when the call returns, and a 201 says it is), and that a job belonging to someone else is
a 404 rather than a 403 — a 403 confirms the id exists and that its owner uploaded
something.
"""
import dataclasses
import io
import time

import jwt
import pytest

import rag.jobs as jobs
import rag.service as rag_service
from core.config import get_settings
from models.ingestion import JOB_CANCELLED, JOB_QUEUED, IngestionJob
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


def _seed_ai_user(db, user_id: str = "job-user") -> str:
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
            first_name="Job",
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


@pytest.fixture()
def upload_dir(tmp_path, monkeypatch):
    root = tmp_path / "uploads" / "rag"
    root.mkdir(parents=True)
    patched = dataclasses.replace(get_settings(), rag_upload_dir=str(root))
    monkeypatch.setattr(rag_service, "get_settings", lambda: patched)
    return root


def _post_upload(client, token, name="Daytona", files=None):
    files = files or [("files", ("manual.txt", io.BytesIO(b"valve clearance " * 40), "text/plain"))]
    return client.post(
        "/resources/upload_files",
        data={"resource_name": name},
        files=files,
        headers={"Authorization": f"Bearer {token}"},
    )


class TestUpload:
    def test_accepts_the_upload_and_returns_the_queued_job(self, client, db, upload_dir):
        token = _seed_ai_user(db)
        response = _post_upload(client, token)

        # 202, not 201: nothing is indexed yet and the base cannot answer a question.
        assert response.status_code == 202
        body = response.json()["data"]
        assert body["job"]["status"] == JOB_QUEUED
        assert body["job"]["total_documents"] == 1
        assert body["job"]["progress"] == 0
        assert body["upload_status"] is False
        assert db.query(DocumentChunk).count() == 0

    def test_the_response_names_every_document_in_the_batch(self, client, db, upload_dir):
        token = _seed_ai_user(db)
        response = _post_upload(
            client,
            token,
            files=[
                ("files", ("a.txt", io.BytesIO(b"valve " * 40), "text/plain")),
                ("files", ("b.txt", io.BytesIO(b"brake " * 40), "text/plain")),
            ],
        )
        names = [doc["file_name"] for doc in response.json()["data"]["job"]["documents"]]
        assert names == ["a.txt", "b.txt"]

    def test_requires_authentication(self, client, db, upload_dir):
        assert _post_upload(client, "not-a-token").status_code == 401


class TestPolling:
    def test_reports_progress_while_the_job_runs(self, client, db, upload_dir):
        token = _seed_ai_user(db)
        job_id = _post_upload(client, token).json()["data"]["job"]["id"]

        # Run the worker inline — the loop that would normally do this is a startup event
        # the TestClient never fires.
        jobs.run_job(db, jobs.claim_next_job(db))

        response = client.get(f"/resources/jobs/{job_id}", headers={"Authorization": f"Bearer {token}"})
        body = response.json()["data"]
        assert response.status_code == 200
        assert body["status"] == "succeeded"
        assert body["progress"] == 100
        assert body["documents"][0]["chunk_count"] > 0
        assert body["documents"][0]["detail"]

    def test_lists_jobs_newest_first_and_can_filter_to_active(self, client, db, upload_dir):
        token = _seed_ai_user(db)
        _post_upload(client, token, name="First")
        jobs.run_job(db, jobs.claim_next_job(db))
        _post_upload(client, token, name="Second")

        headers = {"Authorization": f"Bearer {token}"}
        everything = client.get("/resources/jobs", headers=headers).json()["data"]
        assert [r["resource_name"] for r in everything["records"]] == ["Second", "First"]

        active = client.get("/resources/jobs?active=true", headers=headers).json()["data"]
        assert [r["resource_name"] for r in active["records"]] == ["Second"]

    def test_another_users_job_is_reported_as_missing(self, client, db, upload_dir):
        owner_token = _seed_ai_user(db, "owner-user")
        stranger_token = _seed_ai_user(db, "stranger-user")
        job_id = _post_upload(client, owner_token).json()["data"]["job"]["id"]

        response = client.get(
            f"/resources/jobs/{job_id}", headers={"Authorization": f"Bearer {stranger_token}"}
        )
        assert response.status_code == 404


class TestCancel:
    def test_cancelling_a_queued_job_discards_the_upload(self, client, db, upload_dir):
        token = _seed_ai_user(db)
        data = _post_upload(client, token).json()["data"]
        job_id, resource_id = data["job"]["id"], data["id"]

        response = client.post(
            f"/resources/jobs/{job_id}/cancel", headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code == 200
        assert response.json()["data"]["status"] == JOB_CANCELLED
        assert db.query(Resource).filter(Resource.id == resource_id).first() is None
        assert not (upload_dir / resource_id).exists()

    def test_cancelling_a_finished_job_is_a_conflict_not_a_silent_success(self, client, db, upload_dir):
        token = _seed_ai_user(db)
        job_id = _post_upload(client, token).json()["data"]["job"]["id"]
        jobs.run_job(db, jobs.claim_next_job(db))

        response = client.post(
            f"/resources/jobs/{job_id}/cancel", headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code == 409

    def test_an_unknown_job_is_a_404(self, client, db, upload_dir):
        token = _seed_ai_user(db)
        response = client.post(
            "/resources/jobs/does-not-exist/cancel", headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code == 404


class TestBrainListCarriesIndexingState:
    def test_each_base_reports_its_latest_job(self, client, db, upload_dir):
        token = _seed_ai_user(db)
        _post_upload(client, token, name="Daytona")

        records = client.get(
            "/resources/brain", headers={"Authorization": f"Bearer {token}"}
        ).json()["data"]["records"]
        assert records[0]["job"]["status"] == JOB_QUEUED
        # The list payload stays small: per-document detail is only on the job endpoint.
        assert "documents" not in records[0]["job"]

    def test_a_base_predating_background_indexing_reports_no_job(self, client, db, upload_dir):
        token = _seed_ai_user(db)
        db.add(Resource(resource_name="Legacy", user_id="job-user", upload_status=True))
        db.commit()

        records = client.get(
            "/resources/brain", headers={"Authorization": f"Bearer {token}"}
        ).json()["data"]["records"]
        # None means "nothing in flight", not "unknown" — these bases are answerable.
        assert records[0]["job"] is None
        assert records[0]["upload_status"] is True


class TestChatRefusesAnUnfinishedBase:
    def test_asking_a_still_indexing_base_explains_rather_than_answering(self, client, db, upload_dir):
        token = _seed_ai_user(db)
        resource_id = _post_upload(client, token).json()["data"]["id"]

        response = client.post(
            f"/chat/asks?brain_id={resource_id}&query=valve%20clearance&chat_history_name=t1",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 200
        assert "still being indexed" in response.text
