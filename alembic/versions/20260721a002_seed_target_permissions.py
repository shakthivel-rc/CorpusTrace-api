"""seed_target_permissions

Revision ID: 20260721a002
Revises: 20260721a001
Create Date: 2026-07-21 17:50:00.000000

"""
from typing import Sequence, Union
import uuid

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "20260721a002"
down_revision: Union[str, None] = "20260721a001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


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


def upgrade() -> None:
    permissions_table = sa.table(
        "permissions",
        sa.column("id", sa.String),
        sa.column("name", sa.String),
        sa.column("machine_name", sa.String),
        sa.column("deleted", sa.Integer),
    )
    connection = op.get_bind()

    for name, machine_name in TARGET_PERMISSION_CATALOG:
        existing = connection.execute(
            sa.text("SELECT id FROM permissions WHERE machine_name = :machine_name"),
            {"machine_name": machine_name},
        ).first()
        if existing:
            continue

        op.bulk_insert(
            permissions_table,
            [
                {
                    "id": str(uuid.uuid4()),
                    "name": name,
                    "machine_name": machine_name,
                    "deleted": 0,
                }
            ],
        )


def downgrade() -> None:
    permissions_table = sa.table(
        "permissions",
        sa.column("machine_name", sa.String),
    )
    machine_names = [machine_name for _, machine_name in TARGET_PERMISSION_CATALOG]
    op.execute(permissions_table.delete().where(permissions_table.c.machine_name.in_(machine_names)))
