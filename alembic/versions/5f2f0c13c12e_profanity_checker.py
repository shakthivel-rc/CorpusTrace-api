"""profanity_checker

Revision ID: 5f2f0c13c12e
Revises: b3e7454e46c4
Create Date: 2025-04-02 14:01:42.033874

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '5f2f0c13c12e'
down_revision: Union[str, None] = 'b3e7454e46c4'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() ->None:
    op.create_table(
        'song_resources',
        sa.Column('id', sa.String(36), primary_key=True, default=lambda: str(uuid.uuid4()), nullable=False),
        sa.Column('resource_id', sa.String(36), sa.ForeignKey('resources.id', ondelete="CASCADE"), nullable=False),
        sa.Column('profanity_results', sa.Text(), nullable=False),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.Column('deleted_at', sa.DateTime(), nullable=True)
    )


def downgrade() -> None:
    op.drop_table('song_resources')
