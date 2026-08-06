"""add_activity_logs_table

Revision ID: c1a2b3d4e5f6
Revises: 5f2f0c13c12e
Create Date: 2026-02-17 10:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import uuid
from sqlalchemy.dialects.mysql import CHAR


# revision identifiers, used by Alembic.
revision: str = 'c1a2b3d4e5f6'
down_revision: Union[str, None] = '5f2f0c13c12e'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create activity_logs table."""
    op.create_table(
        "activity_logs",
        sa.Column("id", CHAR(36), primary_key=True, default=lambda: str(uuid.uuid4())),
        sa.Column("user_id", CHAR(36), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("user_name", sa.String(200), nullable=True),
        sa.Column("action", sa.String(50), nullable=False),
        sa.Column("entity_type", sa.String(50), nullable=True),
        sa.Column("entity_id", CHAR(36), nullable=True),
        sa.Column("details", sa.Text, nullable=True),
        sa.Column("ip_address", sa.String(45), nullable=True),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_activity_logs_user_id", "activity_logs", ["user_id"])
    op.create_index("ix_activity_logs_action", "activity_logs", ["action"])
    op.create_index("ix_activity_logs_created_at", "activity_logs", ["created_at"])


def downgrade() -> None:
    """Drop activity_logs table."""
    op.drop_index("ix_activity_logs_created_at", "activity_logs")
    op.drop_index("ix_activity_logs_action", "activity_logs")
    op.drop_index("ix_activity_logs_user_id", "activity_logs")
    op.drop_table("activity_logs")
