"""add_per_document_indexing_config

Per-document chunking settings and optional embeddings.

Every document in every knowledge base was previously chunked identically — a 1200/180
character window — and retrieval was lexical only. These columns let each uploaded document
carry its own chunking strategy and, if the uploader explicitly asks for it, an embedding
model.

Two shapes of default are used here on purpose:

* The chunking columns on `files` are NOT NULL with server defaults that reproduce exactly
  what every existing document already got. A backfilled row and a fresh row then describe
  the same indexing, so nothing has to distinguish "configured as default" from "predates
  the column".
* The embedding columns are NULLABLE with no default, because NULL is a real and different
  state: this document was indexed lexically only. Giving them a default would assert that
  every existing chunk has a vector, and none of them do.

`embedding_json` is a JSON array of floats in a TEXT column, matching `terms_json` next to
it. There is no vector type in this MySQL schema and no vector database in this project;
retrieval already loads and scores every chunk of a resource in Python, so the similarity
walk costs what the lexical walk already cost.

Revision ID: 20260805a003
Revises: 20260805a002
Create Date: 2026-08-05 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "20260805a003"
down_revision: Union[str, None] = "20260805a002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # --- per-document indexing settings -------------------------------------------------
    op.add_column(
        "files",
        sa.Column("chunk_strategy", sa.String(length=30), nullable=False, server_default="character"),
    )
    op.add_column(
        "files",
        sa.Column("chunk_size", sa.Integer(), nullable=False, server_default="1200"),
    )
    op.add_column(
        "files",
        sa.Column("chunk_overlap", sa.Integer(), nullable=False, server_default="180"),
    )
    op.add_column("files", sa.Column("embedding_provider", sa.String(length=50), nullable=True))
    op.add_column("files", sa.Column("embedding_model", sa.String(length=255), nullable=True))

    # --- the vectors themselves ---------------------------------------------------------
    op.add_column("document_chunks", sa.Column("embedding_json", sa.Text(), nullable=True))
    op.add_column("document_chunks", sa.Column("embedding_model", sa.String(length=255), nullable=True))
    op.add_column("document_chunks", sa.Column("embedding_dim", sa.Integer(), nullable=True))

    # --- what the ingestion report says about a run -------------------------------------
    op.add_column(
        "ingestion_job_items",
        sa.Column("embedded_chunks", sa.Integer(), nullable=False, server_default="0"),
    )
    op.add_column("ingestion_job_items", sa.Column("config_json", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("ingestion_job_items", "config_json")
    op.drop_column("ingestion_job_items", "embedded_chunks")

    op.drop_column("document_chunks", "embedding_dim")
    op.drop_column("document_chunks", "embedding_model")
    op.drop_column("document_chunks", "embedding_json")

    op.drop_column("files", "embedding_model")
    op.drop_column("files", "embedding_provider")
    op.drop_column("files", "chunk_overlap")
    op.drop_column("files", "chunk_size")
    op.drop_column("files", "chunk_strategy")
