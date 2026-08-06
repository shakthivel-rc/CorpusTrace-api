"""status_column in user

Revision ID: 0beee1dd689e
Revises: 5b6856e43cbd
Create Date: 2025-03-06 15:16:43.428598

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '0beee1dd689e'
down_revision: Union[str, None] = '5b6856e43cbd'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('users', sa.Column('status', sa.Boolean(), nullable=False, server_default=sa.false()))



def downgrade() -> None:
    op.drop_column('users', 'status')
