"""Add permission

Revision ID: 5a20ffac5e45
Revises: a57fbcdf6fd6
Create Date: 2025-03-05 11:23:45.304739

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import mysql
import uuid

# revision identifiers, used by Alembic.
revision: str = '5a20ffac5e45'
down_revision: Union[str, None] = 'a57fbcdf6fd6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
    'permissions',
        sa.Column('id', sa.String(36), primary_key=True, default=lambda: str(uuid.uuid4()), nullable=False),
        sa.Column('name', sa.String(length=512), nullable=False),
        sa.Column('machine_name', sa.String(length=512), nullable=False),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.Column('deleted', sa.Integer(), nullable=False, server_default='0'),  # 0 for active, 1 for deleted
        sa.Column('deleted_at', sa.DateTime(), nullable=True)  # Allow NULL values

    )
def downgrade() -> None:
    op.drop_table('permissions')