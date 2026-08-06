"""add_users_table

Revision ID: 1471b67e4a25
Revises: 
Create Date: 2025-03-04 16:59:37.634082

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import uuid

# revision identifiers, used by Alembic.
revision: str = '1471b67e4a25'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
    'users',
        sa.Column('id', sa.String(36), primary_key=True, default=lambda: str(uuid.uuid4()), nullable=False),
        sa.Column('username', sa.String(length=100), nullable=False),
        sa.Column('first_name', sa.String(length=100), nullable=False),
        sa.Column('last_name', sa.String(length=100), nullable=False),
        sa.Column('organization', sa.String(length=100), nullable=False),
        sa.Column('password', sa.String(length=100), nullable=True),
        sa.Column('department', sa.String(length=100), nullable=False),
        sa.Column('email', sa.String(length=512), unique=True, nullable=False),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.Column('deleted', sa.Integer(), nullable=False, server_default='0'),  # 0 for active, 1 for deleted
        sa.Column('deleted_at', sa.DateTime(), nullable=True)  # Allow NULL values


    )



def downgrade():
    op.drop_table('users')
 
