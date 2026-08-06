"""add_user_session_table

Revision ID: 8b8ce50e7ffc
Revises: d70b165c5f06
Create Date: 2025-03-13 14:24:45.728503

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import uuid


# revision identifiers, used by Alembic.
revision: str = '8b8ce50e7ffc'
down_revision: Union[str, None] = 'd70b165c5f06'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create user_sessions table"""
    op.create_table(
        'user_sessions',
        sa.Column('id', sa.String(36), primary_key=True, default=lambda: str(uuid.uuid4()), nullable=False),
        sa.Column('user_id', sa.String(36), nullable=False),
        sa.Column('access_token', sa.String(255), nullable=False), 
        sa.Column('refresh_token', sa.String(255), nullable=False), 
        sa.Column('created_at', sa.DateTime, server_default=sa.func.now(), nullable=False)
    )


def downgrade() -> None:
    """Drop user_sessions table"""
    op.drop_table('user_sessions')