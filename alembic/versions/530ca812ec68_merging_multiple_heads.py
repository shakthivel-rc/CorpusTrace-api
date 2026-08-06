"""Merging multiple heads

Revision ID: 530ca812ec68
Revises: bd1a7378c18c, f99bb99a261c
Create Date: 2025-03-29 16:19:23.699873

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '530ca812ec68'
down_revision: Union[str, None] = ('bd1a7378c18c', 'f99bb99a261c')
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
