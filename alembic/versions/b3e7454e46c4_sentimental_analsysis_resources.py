"""Sentimental analsysis resources

Revision ID: b3e7454e46c4
Revises: 530ca812ec68
Create Date: 2025-03-29 16:19:49.661205

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'b3e7454e46c4'
down_revision: Union[str, None] = '530ca812ec68'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'sentimental_resources',
        sa.Column('id', sa.String(36), primary_key=True, default=lambda: str(uuid.uuid4()), nullable=False),
        sa.Column('resource_id', sa.String(36), sa.ForeignKey('resources.id', ondelete="CASCADE"), nullable=False),
        sa.Column('output', sa.Text(), nullable=False),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.Column('deleted_at', sa.DateTime(), nullable=True)
    )



def downgrade() -> None:
    op.drop_table('sentimental_resources')
