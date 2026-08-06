"""add_fp_token in users

Revision ID: 55550e96921f
Revises: 0beee1dd689e
Create Date: 2025-03-06 17:51:52.193685

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '55550e96921f'
down_revision: Union[str, None] = '0beee1dd689e'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('users', sa.Column('fp_token', sa.String(length=255), nullable=True))

def downgrade() -> None:
    op.drop_column('users', 'fp_token')
