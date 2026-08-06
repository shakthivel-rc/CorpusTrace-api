"""add_knowledge_base_tables

Revision ID: f99bb99a261c
Revises: 8b8ce50e7ffc
Create Date: 2025-03-17 17:56:52.612623

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.mysql import CHAR
import uuid


# revision identifiers, used by Alembic.
revision: str = 'f99bb99a261c'
down_revision: Union[str, None] = '8b8ce50e7ffc'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "resources",
        sa.Column('id', sa.String(36), primary_key=True, default=lambda: str(uuid.uuid4()), nullable=False),
        sa.Column('resource_name', sa.String(255), nullable=False),
        sa.Column('resource_type', sa.String(50), nullable=False),
        sa.Column('upload_status', sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column('user_id', CHAR(36), sa.ForeignKey("users.id"), nullable=False),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.Column('deleted_at', sa.DateTime(), nullable=True)  # Allow NULL values
    )
    op.create_table(
        "files",
        sa.Column('id', sa.String(36), primary_key=True, default=lambda: str(uuid.uuid4()), nullable=False),
        sa.Column('file_name', sa.String(255), nullable=False),
        sa.Column('file_type', sa.String(255), nullable=False),
        sa.Column('file_url', sa.String(255), nullable=False),
        sa.Column('resource_id', CHAR(36), sa.ForeignKey("resources.id"), nullable=False),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.Column('deleted_at', sa.DateTime(), nullable=True)  # Allow NULL values
    )


def downgrade() -> None:
    op.drop_table("resources")
    op.drop_table("files")
