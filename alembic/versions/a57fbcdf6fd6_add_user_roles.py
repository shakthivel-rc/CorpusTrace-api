"""add user-roles

Revision ID: a57fbcdf6fd6
Revises: ed20292d4389
Create Date: 2025-03-04 21:20:58.380759

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import uuid

# revision identifiers, used by Alembic.
revision: str = 'a57fbcdf6fd6'
down_revision: Union[str, None] = 'ed20292d4389'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'user_roles',
        sa.Column('id', sa.String(36), primary_key=True, default=lambda: str(uuid.uuid4()), nullable=False),
        sa.Column('user_id', sa.String(36), sa.ForeignKey('users.id', ondelete="CASCADE"), nullable=False),
        sa.Column('role_id', sa.String(36), sa.ForeignKey('roles.id', ondelete="CASCADE"), nullable=False),
    )


def downgrade() -> None:
    op.drop_table('user_roles')
