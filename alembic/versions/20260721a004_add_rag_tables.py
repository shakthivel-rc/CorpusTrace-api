"""add_rag_tables

Revision ID: 20260721a004
Revises: 20260721a003
Create Date: 2026-07-21 18:15:00.000000

"""
from typing import Sequence, Union
import uuid

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.mysql import CHAR


# revision identifiers, used by Alembic.
revision: str = "20260721a004"
down_revision: Union[str, None] = "20260721a003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "document_chunks",
        sa.Column("id", CHAR(36), primary_key=True, default=lambda: str(uuid.uuid4())),
        sa.Column("resource_id", CHAR(36), sa.ForeignKey("resources.id"), nullable=False),
        sa.Column("file_id", CHAR(36), sa.ForeignKey("files.id"), nullable=True),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("source_name", sa.String(255), nullable=False),
        sa.Column("modality", sa.String(50), nullable=False, server_default="text"),
        sa.Column("title", sa.String(255), nullable=True),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("contextual_content", sa.Text(), nullable=False),
        sa.Column("terms_json", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_document_chunks_resource_id", "document_chunks", ["resource_id"])
    op.create_index("ix_document_chunks_file_id", "document_chunks", ["file_id"])

    op.create_table(
        "rag_graph_entities",
        sa.Column("id", CHAR(36), primary_key=True, default=lambda: str(uuid.uuid4())),
        sa.Column("resource_id", CHAR(36), sa.ForeignKey("resources.id"), nullable=False),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("entity_type", sa.String(50), nullable=False, server_default="concept"),
        sa.Column("weight", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("chunk_refs_json", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_rag_graph_entities_resource_id", "rag_graph_entities", ["resource_id"])

    op.create_table(
        "rag_graph_edges",
        sa.Column("id", CHAR(36), primary_key=True, default=lambda: str(uuid.uuid4())),
        sa.Column("resource_id", CHAR(36), sa.ForeignKey("resources.id"), nullable=False),
        sa.Column("source_entity_id", CHAR(36), sa.ForeignKey("rag_graph_entities.id"), nullable=False),
        sa.Column("target_entity_id", CHAR(36), sa.ForeignKey("rag_graph_entities.id"), nullable=False),
        sa.Column("relationship", sa.String(100), nullable=False, server_default="co_occurs_with"),
        sa.Column("weight", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_rag_graph_edges_resource_id", "rag_graph_edges", ["resource_id"])


def downgrade() -> None:
    op.drop_index("ix_rag_graph_edges_resource_id", table_name="rag_graph_edges")
    op.drop_table("rag_graph_edges")
    op.drop_index("ix_rag_graph_entities_resource_id", table_name="rag_graph_entities")
    op.drop_table("rag_graph_entities")
    op.drop_index("ix_document_chunks_file_id", table_name="document_chunks")
    op.drop_index("ix_document_chunks_resource_id", table_name="document_chunks")
    op.drop_table("document_chunks")
