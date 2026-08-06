from __future__ import annotations


TARGET_PERMISSION_CATALOG: tuple[tuple[str, str], ...] = (
    ("Sessions Read", "sessions:read"),
    ("Sessions Revoke", "sessions:revoke"),
    ("Users Read", "users:read"),
    ("Users Create", "users:create"),
    ("Users Update", "users:update"),
    ("Users Disable", "users:disable"),
    ("Users Delete", "users:delete"),
    ("Roles Read", "roles:read"),
    ("Roles Create", "roles:create"),
    ("Roles Update", "roles:update"),
    ("Roles Delete", "roles:delete"),
    ("Permissions Read", "permissions:read"),
    ("Permissions Manage", "permissions:manage"),
    ("Organizations Read", "organizations:read"),
    ("Organizations Create", "organizations:create"),
    ("Organizations Update", "organizations:update"),
    ("Organizations Disable", "organizations:disable"),
    ("Resources Read", "resources:read"),
    ("Resources Create", "resources:create"),
    ("Resources Update", "resources:update"),
    ("Resources Delete", "resources:delete"),
    ("Resources Share", "resources:share"),
    ("Chat Ask", "chat:ask"),
    ("Chat History Read", "chat:history:read"),
    ("Chat Feedback Create", "chat:feedback:create"),
    ("RAG Evaluate", "rag:evaluate"),
    ("Tasks Read", "tasks:read"),
    ("Tasks Retry", "tasks:retry"),
    ("Tasks Cancel", "tasks:cancel"),
    ("Audit Read", "audit:read"),
    ("Audit Export", "audit:export"),
    ("Admin Read", "admin:read"),
    ("Admin Manage Settings", "admin:manage_settings"),
    ("Admin Manage Models", "admin:manage_models"),
    ("Compliance Read", "compliance:read"),
    ("Compliance Export", "compliance:export"),
)

LEGACY_PERMISSION_IMPLICATIONS: dict[str, set[str]] = {
    "manage_users": {
        "admin:read",
        "users:read",
        "users:create",
        "users:update",
        "users:disable",
        "users:delete",
        "roles:read",
        "roles:create",
        "roles:update",
        "roles:delete",
        "permissions:read",
        "permissions:manage",
        "audit:read",
    },
    "ai_access": {
        "chat:ask",
        "chat:history:read",
        "chat:feedback:create",
        "rag:evaluate",
        "resources:read",
    },
}


def permission_is_satisfied(user_permissions: set[str], required_permission: str) -> bool:
    if required_permission in user_permissions:
        return True

    for legacy_permission, implied_permissions in LEGACY_PERMISSION_IMPLICATIONS.items():
        if legacy_permission in user_permissions and required_permission in implied_permissions:
            return True

    implied_permissions = LEGACY_PERMISSION_IMPLICATIONS.get(required_permission)
    if implied_permissions and implied_permissions.issubset(user_permissions):
        return True

    return False
