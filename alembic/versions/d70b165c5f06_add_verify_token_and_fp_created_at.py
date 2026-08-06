"""add_verify_token_and_fp_created_at

Revision ID: d70b165c5f06
Revises: 55550e96921f
Create Date: 2025-03-09 19:00:03.755408

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'd70b165c5f06'
down_revision: Union[str, None] = '55550e96921f'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('users', sa.Column('verify_token', sa.String(length=255), nullable=True)) # to store verify token
    op.add_column('users', sa.Column('fp_token_created_at', sa.DateTime(), nullable=True)) # to store created date for fp token. using created_at for verify token


def downgrade() -> None:
    op.drop_column('users', 'verify_token')
    op.drop_column('users', 'fp_token_created_at')
