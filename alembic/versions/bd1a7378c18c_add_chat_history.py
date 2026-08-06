"""add_chat_history

Revision ID: bd1a7378c18c
Revises: 8b8ce50e7ffc
Create Date: 2025-03-25 17:05:01.469985

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import uuid
from sqlalchemy.dialects.mysql import CHAR



# revision identifiers, used by Alembic.
revision: str = 'bd1a7378c18c'
down_revision: Union[str, None] = '8b8ce50e7ffc'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

def upgrade() -> None:
    """Create chat_history table."""
    op.create_table(
        "chat_history",
        sa.Column("id", CHAR(36), primary_key=True, default=lambda: str(uuid.uuid4())),
        sa.Column("chat_id", sa.String(100), nullable=False),
        sa.Column("user_query", sa.Text, nullable=False),
        sa.Column("user_id", sa.String(36), nullable=False),
        sa.Column("bot_response", sa.Text, nullable=False),
        sa.Column("brain", sa.Text, nullable=True),
        sa.Column("timestamp", sa.TIMESTAMP, nullable=False, server_default=sa.func.current_timestamp()),
    )


def downgrade() -> None:
    """Drop chat_history table if rolled back."""
    op.drop_table("chat_history")