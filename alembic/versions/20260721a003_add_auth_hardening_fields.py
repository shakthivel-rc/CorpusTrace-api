"""add_auth_hardening_fields

Revision ID: 20260721a003
Revises: 20260721a002
Create Date: 2026-07-21 18:05:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "20260721a003"
down_revision: Union[str, None] = "20260721a002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "users",
        sa.Column("failed_login_attempts", sa.Integer(), nullable=False, server_default="0"),
    )
    op.add_column("users", sa.Column("locked_until", sa.DateTime(), nullable=True))
    op.add_column("users", sa.Column("last_failed_login_at", sa.DateTime(), nullable=True))
    op.create_index("ix_users_locked_until", "users", ["locked_until"])
    op.alter_column("users", "failed_login_attempts", server_default=None)


def downgrade() -> None:
    op.drop_index("ix_users_locked_until", table_name="users")
    op.drop_column("users", "last_failed_login_at")
    op.drop_column("users", "locked_until")
    op.drop_column("users", "failed_login_attempts")
