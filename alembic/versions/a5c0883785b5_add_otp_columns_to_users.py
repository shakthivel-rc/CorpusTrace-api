"""add_otp_columns_to_users

Revision ID: a5c0883785b5
Revises: c1a2b3d4e5f6
Create Date: 2026-02-28 22:58:24.783618

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = 'a5c0883785b5'
down_revision: Union[str, None] = 'c1a2b3d4e5f6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('users', sa.Column('otp_code', sa.String(length=255), nullable=True))
    op.add_column('users', sa.Column('otp_created_at', sa.DateTime(), nullable=True))


def downgrade() -> None:
    op.drop_column('users', 'otp_created_at')
    op.drop_column('users', 'otp_code')
