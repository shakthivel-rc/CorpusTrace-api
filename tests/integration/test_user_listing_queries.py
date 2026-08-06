"""How many queries listing users actually costs.

`GET /users` renders each user's roles and each role's permissions. Walking those as lazy
loads is one query per user plus one per role — 2N+1 for a page of N — and this endpoint
defaults to `limit=0`, meaning no limit at all, so N is the whole table.

The assertion here is on the *shape* of the cost, not a golden number: the query count must
not grow with the number of users.
"""
import time

import jwt
import pytest
from sqlalchemy import event

import db.session as db_session
from core.config import get_settings
from models.permissions import Permission
from models.role_permissions import RolePermission
from models.roles import Role
from models.user import User
from models.user_roles import UserRole
from models.user_session import UserSession
from utils.password import hash_password

pytestmark = pytest.mark.integration


def _seed_admin(db, user_id: str = "query-admin") -> str:
    settings = get_settings()
    token = jwt.encode(
        {"sub": user_id, "type": "access", "exp": int(time.time()) + 3600},
        settings.secret_key,
        algorithm=settings.algorithm,
    )
    user = User(
        id=user_id,
        username=f"{user_id}",
        email=f"{user_id}@example.com",
        first_name="Query",
        last_name="Admin",
        organization="Acme",
        department="Ops",
        password=hash_password("Sup3r!Secret!Pass"),
        status=True,
        deleted=0,
    )
    role = Role(name=f"Admin {user_id}")
    permission = db.query(Permission).filter(Permission.machine_name == "manage_users").first()
    if not permission:
        permission = Permission(name="Manage Users", machine_name="manage_users")
        db.add(permission)
        db.flush()
    db.add_all([user, role])
    db.flush()
    db.add_all(
        [
            UserRole(user_id=user_id, role_id=role.id),
            RolePermission(role_id=role.id, permission_id=permission.id),
            UserSession(user_id=user_id, access_token=token, refresh_token="r-" + user_id),
        ]
    )
    db.commit()
    return token


def _seed_plain_users(db, count: int, prefix: str) -> None:
    """Users each holding one role, and that role holding two permissions."""
    perms = []
    for index in range(2):
        permission = Permission(name=f"{prefix} perm {index}", machine_name=f"{prefix}_perm_{index}")
        db.add(permission)
        perms.append(permission)
    db.flush()

    for index in range(count):
        user_id = f"{prefix}-user-{index}"
        db.add(
            User(
                id=user_id,
                username=user_id,
                email=f"{user_id}@example.com",
                first_name="Listed",
                last_name=str(index),
                organization="Acme",
                department="Ops",
                password=hash_password("Sup3r!Secret!Pass"),
                status=True,
                deleted=0,
            )
        )
        role = Role(name=f"{prefix} role {index}")
        db.add(role)
        db.flush()
        db.add(UserRole(user_id=user_id, role_id=role.id))
        for permission in perms:
            db.add(RolePermission(role_id=role.id, permission_id=permission.id))
    db.commit()


class _QueryCounter:
    def __init__(self):
        self.statements = []

    def __enter__(self):
        self._listener = lambda conn, cursor, statement, *a: self.statements.append(statement)
        event.listen(db_session.engine, "before_cursor_execute", self._listener)
        return self

    def __exit__(self, *exc):
        event.remove(db_session.engine, "before_cursor_execute", self._listener)

    @property
    def count(self) -> int:
        return len(self.statements)


def _list_users(client, token):
    return client.get("/users", headers={"Authorization": f"Bearer {token}"})


def test_listing_users_does_not_issue_a_query_per_user(client, db):
    """The N+1 guard: tripling the number of users must not triple the query count."""
    token = _seed_admin(db)

    _seed_plain_users(db, 3, "small")
    with _QueryCounter() as small:
        response = _list_users(client, token)
    assert response.status_code == 200
    assert len(response.json()["data"]["records"]) == 4  # 3 listed + the admin

    _seed_plain_users(db, 9, "large")
    with _QueryCounter() as large:
        response = _list_users(client, token)
    assert response.status_code == 200
    assert len(response.json()["data"]["records"]) == 13

    # selectinload issues a fixed number of extra SELECTs (one for roles, one for
    # permissions) regardless of page size. Lazy loading would have grown by ~24 here.
    assert large.count <= small.count + 1, (
        f"query count grew with row count: {small.count} for 4 users, "
        f"{large.count} for 13 — the roles/permissions walk is lazy-loading again"
    )


def test_listing_users_still_returns_roles_and_permissions(client, db):
    """Eager loading must not change the payload."""
    token = _seed_admin(db, "payload-admin")
    _seed_plain_users(db, 2, "payload")

    response = _list_users(client, token)
    assert response.status_code == 200
    records = {record["username"]: record for record in response.json()["data"]["records"]}

    listed = records["payload-user-0"]
    assert [role["role_name"] for role in listed["roles"]] == ["payload role 0"]
    assert sorted(p["machine_name"] for p in listed["permissions"]) == [
        "payload_perm_0",
        "payload_perm_1",
    ]
