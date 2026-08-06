"""Add role_permissions

Revision ID: 5b6856e43cbd
Revises: 5a20ffac5e45
Create Date: 2025-03-05 11:27:21.866464

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.mysql import CHAR
import uuid

# revision identifiers, used by Alembic.
revision: str = '5b6856e43cbd'
down_revision: Union[str, None] = '5a20ffac5e45'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'role_permissions',
        sa.Column('id', CHAR(36), primary_key=True, default=lambda: str(uuid.uuid4()), index=True),
        sa.Column('role_id', CHAR(36), sa.ForeignKey("roles.id"), nullable=False),
        sa.Column('permission_id', CHAR(36), sa.ForeignKey("permissions.id"), nullable=False),
    )


def downgrade() -> None:
    op.drop_table('role_permissions')
