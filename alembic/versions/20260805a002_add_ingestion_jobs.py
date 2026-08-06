"""add_ingestion_jobs

Backs the background document-indexing worker: one `ingestion_jobs` row per upload batch,
one `ingestion_job_items` row per document in it.

The composite index on (status, created_at) is the only index the worker's poll uses —
it claims the oldest queued job, and without it that query table-scans a table that only
ever grows. The (user_id, created_at) index serves the job list the Tasks screen polls.

Revision ID: 20260805a002
Revises: 20260805a001
Create Date: 2026-08-05 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "20260805a002"
down_revision: Union[str, None] = "20260805a001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "ingestion_jobs",
        sa.Column("id", sa.CHAR(36), nullable=False),
        # Nullable: cancelling an upload purges the resource, and the job survives it as
        # the record that the upload happened and was cancelled.
        sa.Column("resource_id", sa.CHAR(36), nullable=True),
        sa.Column("resource_name", sa.String(255), nullable=False, server_default=""),
        sa.Column("user_id", sa.CHAR(36), nullable=False),
        sa.Column("status", sa.String(20), nullable=False, server_default="queued"),
        sa.Column("stage", sa.String(30), nullable=False, server_default="queued"),
        sa.Column("claimed_by", sa.String(100), nullable=True),
        sa.Column("total_documents", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("processed_documents", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("total_bytes", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("indexed_chunks", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("current_document", sa.String(255), nullable=True),
        sa.Column("cancel_requested", sa.Boolean(), nullable=False, server_default=sa.text("0")),
        sa.Column("message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("started_at", sa.DateTime(), nullable=True),
        sa.Column("finished_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["resource_id"], ["resources.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_ingestion_jobs_resource_id", "ingestion_jobs", ["resource_id"])
    op.create_index("ix_ingestion_jobs_user_id", "ingestion_jobs", ["user_id"])
    op.create_index("ix_ingestion_jobs_status", "ingestion_jobs", ["status"])
    op.create_index("ix_ingestion_jobs_claim", "ingestion_jobs", ["status", "created_at"])
    op.create_index("ix_ingestion_jobs_user_recent", "ingestion_jobs", ["user_id", "created_at"])

    op.create_table(
        "ingestion_job_items",
        sa.Column("id", sa.CHAR(36), nullable=False),
        sa.Column("job_id", sa.CHAR(36), nullable=False),
        sa.Column("file_id", sa.CHAR(36), nullable=True),
        sa.Column("position", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("file_name", sa.String(255), nullable=False),
        sa.Column("file_type", sa.String(255), nullable=False, server_default="application/octet-stream"),
        sa.Column("byte_size", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("status", sa.String(20), nullable=False, server_default="pending"),
        sa.Column("stage", sa.String(30), nullable=False, server_default="queued"),
        sa.Column("chunk_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("page_count", sa.Integer(), nullable=True),
        sa.Column("detail", sa.Text(), nullable=True),
        sa.Column("started_at", sa.DateTime(), nullable=True),
        sa.Column("finished_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["job_id"], ["ingestion_jobs.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_ingestion_job_items_job_id", "ingestion_job_items", ["job_id"])


def downgrade() -> None:
    op.drop_index("ix_ingestion_job_items_job_id", table_name="ingestion_job_items")
    op.drop_table("ingestion_job_items")
    op.drop_index("ix_ingestion_jobs_user_recent", table_name="ingestion_jobs")
    op.drop_index("ix_ingestion_jobs_claim", table_name="ingestion_jobs")
    op.drop_index("ix_ingestion_jobs_status", table_name="ingestion_jobs")
    op.drop_index("ix_ingestion_jobs_user_id", table_name="ingestion_jobs")
    op.drop_index("ix_ingestion_jobs_resource_id", table_name="ingestion_jobs")
    op.drop_table("ingestion_jobs")
