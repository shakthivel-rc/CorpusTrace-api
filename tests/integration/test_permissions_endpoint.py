"""Integration test for route-level authorization (check_permissions). An
authenticated user who lacks the required slug must be refused with 403 — a
regression here is a silent privilege escalation, not a crash."""
import time

import jwt
import pytest

from core.config import get_settings
from models.user import User
from models.user_session import UserSession
from utils.password import hash_password

pytestmark = pytest.mark.integration


def _access_token(sub: str) -> str:
    settings = get_settings()
    payload = {"sub": sub, "type": "access", "exp": int(time.time()) + 3600}
    return jwt.encode(payload, settings.secret_key, algorithm=settings.algorithm)


def _seed_authenticated_user_without_roles(db, user_id="perm-user"):
    db.add(
        User(
            id=user_id,
            email="noperm@example.com",
            username="noperm",
            first_name="No",
            last_name="Perm",
            organization="Org",
            department="Dept",
            password=hash_password("Abcdef123!@#"),
            status=True,
            deleted=0,
        )
    )
    token = _access_token(user_id)
    db.add(UserSession(user_id=user_id, access_token=token))
    db.commit()
    return token


def test_manage_users_route_requires_the_permission(client, db):
    token = _seed_authenticated_user_without_roles(db)
    # DELETE /users/{id} is gated by check_permissions(["manage_users"]); the user has no roles.
    resp = client.delete("/users/some-id", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 403
    assert resp.json()["detail"]["message"] == "Insufficient permissions"


def test_manage_users_route_rejects_anonymous_callers(client):
    # No token → stopped at the middleware before authorization even runs.
    resp = client.delete("/users/some-id")
    assert resp.status_code == 401
