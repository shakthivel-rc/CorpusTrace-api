"""Integration tests for JWTAuthMiddleware — the allow-list, the JWT decode, the
`type == access` check, and the live user_sessions lookup that gates every
non-public request."""
import time

import jwt
import pytest

from core.config import get_settings
from models.user_session import UserSession

pytestmark = pytest.mark.integration


def _make_access_token(sub: str = "user-1", ttl_seconds: int = 3600, token_type: str = "access") -> str:
    settings = get_settings()
    payload = {"sub": sub, "type": token_type, "exp": int(time.time()) + ttl_seconds}
    return jwt.encode(payload, settings.secret_key, algorithm=settings.algorithm)


def test_missing_authorization_header_is_rejected(client):
    resp = client.get("/")
    assert resp.status_code == 401
    assert resp.json()["message"] == "Missing or invalid token"


def test_malformed_token_is_rejected(client):
    resp = client.get("/", headers={"Authorization": "Bearer not-a-real-jwt"})
    assert resp.status_code == 401
    assert resp.json()["message"] == "Invalid token"


def test_non_access_token_type_is_rejected(client):
    token = _make_access_token(token_type="refresh")
    resp = client.get("/", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 401
    assert resp.json()["message"] == "Invalid token type"


def test_valid_jwt_without_a_live_session_is_rejected(client):
    # Correctly signed + typed, but no matching (unrevoked) user_sessions row.
    token = _make_access_token()
    resp = client.get("/", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 401
    assert resp.json()["message"] == "Invalid session or token revoked"


def test_valid_jwt_with_a_live_session_passes_through(client, db):
    token = _make_access_token()
    db.add(UserSession(user_id="user-1", access_token=token))
    db.commit()

    resp = client.get("/", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200
    assert resp.json() == {"message": "CorpusTrace API"}


def test_revoked_session_is_rejected(client, db):
    from datetime import datetime, timezone

    token = _make_access_token()
    db.add(UserSession(user_id="user-1", access_token=token, revoked_at=datetime.now(timezone.utc)))
    db.commit()

    resp = client.get("/", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 401
    assert resp.json()["message"] == "Invalid session or token revoked"
