"""Creating a user from the admin console.

This is the console's primary write path — the only supported way to onboard someone into a
named role without them self-registering — and it could not succeed at all: `create_new_user`
returns a dict (`vars(new_user)`) while `add_user_controller` read `new_user.id` when writing
the activity log, raising `AttributeError: 'dict' object has no attribute 'id'`.

The reason that survived a 854-test suite is worth keeping in view. The failing line sits
behind `if request:` / `if current_user:`, so it is only reached when the request carries an
authenticated actor — and nothing outside the live suite (which is skipped without a running
stack and real credentials) had ever driven this route with a live session row. So the tests
here seed one, which is the same shape every real call has: the route requires `manage_users`,
so `request.state.user` is *always* populated in production and the branch is *always* taken.
"""
import time

import jwt
import pytest

from core.config import get_settings
from models.activity_log import ActivityLog
from models.permissions import Permission
from models.role_permissions import RolePermission
from models.roles import Role
from models.user import User
from models.user_roles import UserRole
from models.user_session import UserSession
from utils.password import hash_password

pytestmark = pytest.mark.integration

ADMIN_ID = "console-admin"


@pytest.fixture()
def admin_token(db):
    """An admin holding `manage_users`, with a live session row, plus a role to assign."""
    settings = get_settings()
    token = jwt.encode(
        {"sub": ADMIN_ID, "type": "access", "exp": int(time.time()) + 3600},
        settings.secret_key,
        algorithm=settings.algorithm,
    )
    permission = Permission(name="Manage Users", machine_name="manage_users")
    admin_role = Role(name="Console Admin")
    target_role = Role(name="Customer")
    db.add_all(
        [
            User(
                id=ADMIN_ID,
                username=ADMIN_ID,
                email=f"{ADMIN_ID}@example.com",
                first_name="Console",
                last_name="Admin",
                organization="Acme",
                department="Ops",
                password=hash_password("Sup3r!Secret!Pass"),
                status=True,
                deleted=0,
            ),
            permission,
            admin_role,
            target_role,
        ]
    )
    db.flush()
    db.add_all(
        [
            UserRole(user_id=ADMIN_ID, role_id=admin_role.id),
            RolePermission(role_id=admin_role.id, permission_id=permission.id),
            UserSession(user_id=ADMIN_ID, access_token=token, refresh_token=f"r-{ADMIN_ID}"),
        ]
    )
    db.commit()
    return token, target_role.id


def _payload(role_id: str, suffix: str = "1") -> dict:
    return {
        "email": f"created-{suffix}@example.com",
        "first_name": "Created",
        "last_name": f"User{suffix}",
        "username": f"created-{suffix}",
        "organization": "Acme",
        "department": "QA",
        "role_id": role_id,
    }


def test_creating_a_user_returns_201_and_persists_the_row(client, db, admin_token):
    token, role_id = admin_token

    response = client.post(
        "/users",
        json=_payload(role_id),
        headers={"Authorization": f"Bearer {token}"},
    )

    # The full round trip, not just the status code: a fix that returned 201 without
    # persisting the row would be just as broken from the operator's side.
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["status"] == "success"
    assert body["data"]["email"] == "created-1@example.com"

    created = db.query(User).filter(User.email == "created-1@example.com").first()
    assert created is not None
    assert db.query(UserRole).filter(UserRole.user_id == created.id).count() == 1


def test_the_activity_log_records_the_created_user(client, db, admin_token):
    """The line that raised. `entity_id` must be the new user's id, not the actor's."""
    token, role_id = admin_token

    client.post("/users", json=_payload(role_id, "2"), headers={"Authorization": f"Bearer {token}"})

    created = db.query(User).filter(User.email == "created-2@example.com").first()
    entry = (
        db.query(ActivityLog)
        .filter(ActivityLog.entity_type == "user", ActivityLog.action == "CREATE")
        .first()
    )
    assert entry is not None, "creating a user wrote no activity row"
    assert entry.entity_id == created.id


def test_the_response_never_carries_the_activation_token_or_password(client, db, admin_token):
    """`vars(new_user)` carries every mapped column, and two of them are secrets.

    `verify_token` is the account-activation secret that the invite email delivers — echoing
    it back to whoever created the account hands them the ability to activate it without ever
    seeing that mailbox. `password` is the hash. Both were in the 201 body.
    """
    token, role_id = admin_token

    response = client.post(
        "/users",
        json=_payload(role_id, "3"),
        headers={"Authorization": f"Bearer {token}"},
    )

    data = response.json()["data"]
    for secret in ("verify_token", "password", "fp_token", "otp_code"):
        assert secret not in data, f"{secret} must never be returned"
    # Still useful to the caller.
    assert data["id"] and data["username"] == "created-3"


def test_a_duplicate_username_is_a_400_with_an_envelope(client, db, admin_token):
    token, role_id = admin_token
    client.post("/users", json=_payload(role_id, "4"), headers={"Authorization": f"Bearer {token}"})

    response = client.post(
        "/users",
        json=_payload(role_id, "4"),
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 400
    assert response.json()["message"] == "Username already exists"
