"""harden_user_sessions

Revision ID: 20260721a001
Revises: a5c0883785b5
Create Date: 2026-07-21 17:45:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "20260721a001"
down_revision: Union[str, None] = "a5c0883785b5"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.alter_column(
        "user_sessions",
        "access_token",
        existing_type=sa.String(length=255),
        type_=sa.String(length=1024),
        existing_nullable=False,
    )
    op.alter_column(
        "user_sessions",
        "refresh_token",
        existing_type=sa.String(length=255),
        type_=sa.String(length=1024),
        nullable=True,
    )
    op.add_column("user_sessions", sa.Column("refresh_token_hash", sa.String(length=128), nullable=True))
    op.add_column("user_sessions", sa.Column("refresh_token_expires_at", sa.DateTime(), nullable=True))
    op.add_column("user_sessions", sa.Column("revoked_at", sa.DateTime(), nullable=True))
    op.add_column("user_sessions", sa.Column("user_agent", sa.String(length=512), nullable=True))
    op.add_column("user_sessions", sa.Column("ip_address", sa.String(length=45), nullable=True))
    op.create_index("ix_user_sessions_refresh_token_hash", "user_sessions", ["refresh_token_hash"])


def downgrade() -> None:
    op.drop_index("ix_user_sessions_refresh_token_hash", table_name="user_sessions")
    op.drop_column("user_sessions", "ip_address")
    op.drop_column("user_sessions", "user_agent")
    op.drop_column("user_sessions", "revoked_at")
    op.drop_column("user_sessions", "refresh_token_expires_at")
    op.drop_column("user_sessions", "refresh_token_hash")
    op.alter_column(
        "user_sessions",
        "refresh_token",
        existing_type=sa.String(length=1024),
        type_=sa.String(length=255),
        nullable=False,
    )
    op.alter_column(
        "user_sessions",
        "access_token",
        existing_type=sa.String(length=1024),
        type_=sa.String(length=255),
        existing_nullable=False,
    )
