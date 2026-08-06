"""add_llm_enabled_preference

Adds the off switch for LLM synthesis. NOT NULL with a server default of 1 so every
existing preference row keeps its current behaviour — a nullable column would read back
as None for anyone who saved a model before this shipped, and the enforcement in
`_plan_answer` would then have to guess what None meant.

Revision ID: 20260805a001
Revises: 20260804a001
Create Date: 2026-08-05 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "20260805a001"
down_revision: Union[str, None] = "20260804a001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "llm_user_preferences",
        sa.Column("llm_enabled", sa.Boolean(), nullable=False, server_default=sa.text("1")),
    )


def downgrade() -> None:
    op.drop_column("llm_user_preferences", "llm_enabled")
