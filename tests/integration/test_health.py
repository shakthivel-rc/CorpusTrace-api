"""Integration tests for the health endpoints — the only machine-checkable
assertions the app exposes. They also validate the TestClient harness + the
{status, status_code, message, data} envelope end-to-end."""
import pytest

pytestmark = pytest.mark.integration


def test_live_returns_the_success_envelope(client):
    resp = client.get("/api/v1/health/live")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "success"
    assert body["data"] == {"status": "live"}


def test_ready_checks_the_database(client):
    resp = client.get("/api/v1/health/ready")
    assert resp.status_code == 200
    assert resp.json()["data"] == {"database": "ok"}


def test_dependencies_reports_database_ok(client):
    resp = client.get("/api/v1/health/dependencies")
    assert resp.status_code == 200
    assert resp.json()["data"] == {"database": "ok"}


def test_root_is_liveness_only(client):
    # "/" is public in FastAPI but still passes through the middleware stack.
    resp = client.get("/")
    # It requires auth (it is NOT in UNAUTHENTICATED_ROUTES), so an anonymous call is 401.
    assert resp.status_code == 401
