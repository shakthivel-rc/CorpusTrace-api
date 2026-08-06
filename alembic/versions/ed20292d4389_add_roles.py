"""add roles

Revision ID: ed20292d4389
Revises: 1471b67e4a25
Create Date: 2025-03-04 21:17:45.158366

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import uuid

# revision identifiers, used by Alembic.
revision: str = 'ed20292d4389'
down_revision: Union[str, None] = '1471b67e4a25'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
    'roles',
        sa.Column('id', sa.String(36), primary_key=True, default=lambda: str(uuid.uuid4()), nullable=False),
        sa.Column('name', sa.String(length=50), nullable=False),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.Column('deleted', sa.Integer(), nullable=False, server_default='0'),  # 0 for active, 1 for deleted
        sa.Column('deleted_at', sa.DateTime(), nullable=True)  # Allow NULL values

    )

def downgrade() -> None:
    op.drop_table('roles')
