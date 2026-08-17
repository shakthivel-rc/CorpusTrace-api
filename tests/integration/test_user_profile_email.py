"""GET /users/profile must not 500 on an address the database already accepted.

`UserProfileResponse` used to declare `email: EmailStr`, which re-validates a value that is
already stored. email-validator refuses the special-use TLDs of RFC 6761/6762 — .local,
.localhost, .test, .invalid — so the seeded superadmin at superadmin@corpustrace.local could sign
in (login takes a plain str) and then got an uncaught ValidationError, which with no global
exception handler is a bare HTTP 500, on every profile load.

Validation belongs on the way in. These tests pin both halves of that: a stored address is
served back whatever its TLD, and the request schemas still refuse a bad one.
"""
import time

import jwt
import pytest

from core.config import get_settings
from models.user import User
from models.user_session import UserSession
from utils.password import hash_password

pytestmark = pytest.mark.integration


def _seed_user(db, email: str, user_id: str = "profile-user") -> str:
    settings = get_settings()
    token = jwt.encode(
        {"sub": user_id, "type": "access", "exp": int(time.time()) + 3600},
        settings.secret_key,
        algorithm=settings.algorithm,
    )
    db.add_all(
        [
            User(
                id=user_id,
                username=user_id,
                email=email,
                first_name="Profile",
                last_name="User",
                organization="Acme",
                department="Ops",
                password=hash_password("Sup3r!Secret!Pass"),
                status=True,
                deleted=0,
            ),
            UserSession(user_id=user_id, access_token=token, refresh_token="r-" + user_id),
        ]
    )
    db.commit()
    return token


@pytest.mark.parametrize(
    "email",
    [
        "superadmin@corpustrace.local",  # what the seeder actually creates
        "someone@corpustrace.localhost",
        "someone@corpustrace.test",
        "someone@corpustrace.invalid",
        "someone@example.com",  # the ordinary case, to prove nothing else regressed
    ],
)
def test_profile_is_served_for_any_stored_address(client, db, email):
    token = _seed_user(db, email)

    response = client.get("/users/profile", headers={"Authorization": f"Bearer {token}"})

    assert response.status_code == 200, response.text
    assert response.json()["data"]["email"] == email


def test_request_schemas_still_reject_a_bad_address():
    """The fix must not have loosened validation on the way in."""
    from pydantic import ValidationError

    from schemas.user import UserRegisterRequest

    with pytest.raises(ValidationError):
        UserRegisterRequest(
            email="not-an-address",
            first_name="A",
            last_name="B",
            username="ab",
        )
