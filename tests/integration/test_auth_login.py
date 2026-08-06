"""Integration tests for the login flow and lockout arithmetic — security-relevant
business rules where an off-by-one or a missing counter-reset is silent.

Runs the real POST /user/login endpoint through the middleware + controller against
the test database."""
import pytest

from models.user import User
from utils.password import hash_password

pytestmark = pytest.mark.integration

VALID_PASSWORD = "Abcdef123!@#"


def _seed_user(db, password=VALID_PASSWORD, **overrides):
    user = User(
        email="ada@example.com",
        username="ada",
        first_name="Ada",
        last_name="Lovelace",
        organization="Analytical Engines",
        department="R&D",
        password=hash_password(password),
        status=True,
        deleted=0,
        failed_login_attempts=0,
    )
    for key, value in overrides.items():
        setattr(user, key, value)
    db.add(user)
    db.commit()
    return user


def _login(client, password, username="ada@example.com"):
    return client.post("/user/login", json={"username": username, "password": password})


def test_login_succeeds_and_returns_a_token_pair(client, db):
    _seed_user(db)
    resp = _login(client, VALID_PASSWORD)
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["access_token"]
    assert data["refresh_token"]


def test_login_with_wrong_password_is_rejected(client, db):
    _seed_user(db)
    resp = _login(client, "WrongPassword1!")
    assert resp.status_code == 401


def test_unknown_user_is_rejected(client, db):
    resp = _login(client, VALID_PASSWORD, username="nobody@example.com")
    assert resp.status_code == 401


def test_disabled_account_cannot_log_in(client, db):
    _seed_user(db, status=False)
    resp = _login(client, VALID_PASSWORD)
    assert resp.status_code == 401


def test_account_locks_after_the_max_failed_attempts(client, db):
    _seed_user(db)  # default AUTH_MAX_FAILED_LOGIN_ATTEMPTS = 5

    # The first four bad attempts are rejected as invalid credentials (401)...
    for _ in range(4):
        assert _login(client, "WrongPassword1!").status_code == 401
    # ...and the fifth trips the lockout, which answers 423 Locked.
    assert _login(client, "WrongPassword1!").status_code == 423

    db.expire_all()
    user = db.query(User).filter(User.email == "ada@example.com").first()
    assert user.failed_login_attempts >= 5
    assert user.locked_until is not None

    # While locked, even the correct password is refused with 423 Locked.
    assert _login(client, VALID_PASSWORD).status_code == 423


def test_successful_login_resets_the_failure_counter(client, db):
    _seed_user(db, failed_login_attempts=3)
    assert _login(client, VALID_PASSWORD).status_code == 200

    db.expire_all()
    user = db.query(User).filter(User.email == "ada@example.com").first()
    assert user.failed_login_attempts == 0
    assert user.locked_until is None
