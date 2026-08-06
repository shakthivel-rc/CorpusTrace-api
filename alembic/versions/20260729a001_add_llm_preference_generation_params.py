"""add_llm_preference_generation_params

Revision ID: 20260729a001
Revises: 20260722a001
Create Date: 2026-07-29 10:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "20260729a001"
down_revision: Union[str, None] = "20260722a001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("llm_user_preferences", sa.Column("temperature", sa.Float(), nullable=True))
    op.add_column("llm_user_preferences", sa.Column("max_tokens", sa.Integer(), nullable=True))
    op.add_column("llm_user_preferences", sa.Column("top_p", sa.Float(), nullable=True))


def downgrade() -> None:
    op.drop_column("llm_user_preferences", "top_p")
    op.drop_column("llm_user_preferences", "max_tokens")
    op.drop_column("llm_user_preferences", "temperature")
