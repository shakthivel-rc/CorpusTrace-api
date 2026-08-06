"""add_llm_provider_tables

Revision ID: 20260722a001
Revises: 20260721a004
Create Date: 2026-07-22 13:10:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.mysql import CHAR


revision: str = "20260722a001"
down_revision: Union[str, None] = "20260721a004"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "llm_provider_credentials",
        sa.Column("id", CHAR(36), primary_key=True),
        sa.Column("user_id", CHAR(36), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("provider", sa.String(50), nullable=False),
        sa.Column("encrypted_api_key", sa.Text(), nullable=True),
        sa.Column("api_key_hint", sa.String(64), nullable=True),
        sa.Column("base_url", sa.String(512), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("validation_status", sa.String(50), nullable=False, server_default="stored"),
        sa.Column("validation_error", sa.String(512), nullable=True),
        sa.Column("last_validated_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("deleted_at", sa.DateTime(), nullable=True),
    )
    op.create_index("ix_llm_provider_credentials_user_id", "llm_provider_credentials", ["user_id"])
    op.create_index("ix_llm_provider_credentials_provider", "llm_provider_credentials", ["provider"])

    op.create_table(
        "llm_model_catalogs",
        sa.Column("id", CHAR(36), primary_key=True),
        sa.Column("provider", sa.String(50), nullable=False),
        sa.Column("model_id", sa.String(255), nullable=False),
        sa.Column("display_name", sa.String(255), nullable=False),
        sa.Column("capabilities_json", sa.Text(), nullable=True),
        sa.Column("raw_json", sa.Text(), nullable=True),
        sa.Column("source", sa.String(50), nullable=False, server_default="seed"),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("fetched_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("expires_at", sa.DateTime(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("provider", "model_id", name="uq_llm_model_provider_model"),
    )
    op.create_index("ix_llm_model_catalogs_provider", "llm_model_catalogs", ["provider"])

    op.create_table(
        "llm_user_preferences",
        sa.Column("id", CHAR(36), primary_key=True),
        sa.Column("user_id", CHAR(36), sa.ForeignKey("users.id"), nullable=False, unique=True),
        sa.Column("provider", sa.String(50), nullable=False),
        sa.Column("model_id", sa.String(255), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_llm_user_preferences_user_id", "llm_user_preferences", ["user_id"])


def downgrade() -> None:
    op.drop_index("ix_llm_user_preferences_user_id", table_name="llm_user_preferences")
    op.drop_table("llm_user_preferences")
    op.drop_index("ix_llm_model_catalogs_provider", table_name="llm_model_catalogs")
    op.drop_table("llm_model_catalogs")
    op.drop_index("ix_llm_provider_credentials_provider", table_name="llm_provider_credentials")
    op.drop_index("ix_llm_provider_credentials_user_id", table_name="llm_provider_credentials")
    op.drop_table("llm_provider_credentials")
