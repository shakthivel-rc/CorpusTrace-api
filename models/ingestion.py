"""Ingestion jobs — the durable record of what happened to an uploaded document.

Indexing used to run inside the upload request: the browser held a connection open
through PDF parsing, chunking and graph construction, and the only thing it learned was
whether the whole batch eventually returned 201. A ten-document upload was a spinner,
and a document that yielded no text (a scanned PDF, a legacy `.doc`) was indistinguishable
from one that indexed perfectly.

These two tables move that work into a background worker and make it observable. The job
row is the batch; an item row is one document. Both are written by the worker as it goes,
so progress survives a page refresh — the state lives in MySQL, not in a browser tab or
in worker memory.

`cancel_requested` is the whole cancellation mechanism: a cancel arrives on a different
request, on a different thread, holding a different Session, so it cannot reach into the
worker. It sets a flag; the worker re-reads it at each safe point and stops itself.
"""
import uuid
from datetime import datetime, timezone

from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.mysql import CHAR
from sqlalchemy.orm import relationship

from db.session import Base

# Job lifecycle. `queued` is claimable, `running` is owned by a worker, and the three
# terminal states are final — nothing transitions out of them.
JOB_QUEUED = "queued"
JOB_RUNNING = "running"
JOB_SUCCEEDED = "succeeded"
JOB_FAILED = "failed"
JOB_CANCELLED = "cancelled"
TERMINAL_JOB_STATUSES = frozenset({JOB_SUCCEEDED, JOB_FAILED, JOB_CANCELLED})

# Per-document outcome. `partial` is the one that earns its keep: extraction succeeded but
# produced no usable text, which is a real and common outcome for scanned PDFs and which
# the old boolean `upload_status` could not express.
ITEM_PENDING = "pending"
ITEM_RUNNING = "running"
ITEM_SUCCEEDED = "succeeded"
ITEM_PARTIAL = "partial"
ITEM_FAILED = "failed"
ITEM_CANCELLED = "cancelled"

# Stages, in the order the worker passes through them. Each one is a phase the worker can
# actually be observed in — extraction and chunking are a single call, so splitting them
# here would invent a distinction the progress display could never truthfully show.
STAGE_QUEUED = "queued"
STAGE_READING = "reading"
STAGE_INDEXING = "indexing"
STAGE_LINKING = "linking"
STAGE_DONE = "done"


class IngestionJob(Base):
    __tablename__ = "ingestion_jobs"

    id = Column(CHAR(36), primary_key=True, default=lambda: str(uuid.uuid4()), nullable=False)
    # Nullable because cancelling discards the resource entirely — a cancelled upload must
    # leave nothing half-indexed behind. The job outlives it as the history record, which
    # is why the name is copied here rather than only joined.
    resource_id = Column(CHAR(36), ForeignKey("resources.id"), nullable=True, index=True)
    resource_name = Column(String(255), nullable=False, default="")
    # Denormalized from the resource on purpose: every read of this table is scoped to the
    # caller, and joining through `resources` to authorize a progress poll that fires once
    # a second is a cost with no benefit.
    user_id = Column(CHAR(36), nullable=False, index=True)

    status = Column(String(20), nullable=False, default=JOB_QUEUED, index=True)
    stage = Column(String(30), nullable=False, default=STAGE_QUEUED)
    # Set when a worker claims the job. Two uvicorn workers each run their own loop and
    # both poll this table, so the claim is a conditional UPDATE and this records who won.
    claimed_by = Column(String(100), nullable=True)

    total_documents = Column(Integer, nullable=False, default=0)
    processed_documents = Column(Integer, nullable=False, default=0)
    total_bytes = Column(Integer, nullable=False, default=0)
    indexed_chunks = Column(Integer, nullable=False, default=0)
    current_document = Column(String(255), nullable=True)

    # Cooperative cancellation. Not a column the worker owns — it is written by the cancel
    # request and only ever read by the worker.
    cancel_requested = Column(Boolean, nullable=False, default=False)

    message = Column(Text, nullable=True)

    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    started_at = Column(DateTime, nullable=True)
    finished_at = Column(DateTime, nullable=True)

    items = relationship(
        "IngestionJobItem",
        back_populates="job",
        order_by="IngestionJobItem.position",
        cascade="all, delete-orphan",
    )


class IngestionJobItem(Base):
    __tablename__ = "ingestion_job_items"

    id = Column(CHAR(36), primary_key=True, default=lambda: str(uuid.uuid4()), nullable=False)
    job_id = Column(CHAR(36), ForeignKey("ingestion_jobs.id"), nullable=False, index=True)
    # Nullable because a cancelled job purges its resource, and the file row goes with it.
    # The item survives so the history still says which documents were in the batch.
    file_id = Column(CHAR(36), nullable=True)

    position = Column(Integer, nullable=False, default=0)
    file_name = Column(String(255), nullable=False)
    file_type = Column(String(255), nullable=False, default="application/octet-stream")
    byte_size = Column(Integer, nullable=False, default=0)

    status = Column(String(20), nullable=False, default=ITEM_PENDING)
    stage = Column(String(30), nullable=False, default=STAGE_QUEUED)
    chunk_count = Column(Integer, nullable=False, default=0)
    embedded_chunks = Column(Integer, nullable=False, default=0, server_default="0")
    page_count = Column(Integer, nullable=True)

    # A snapshot of the indexing settings this run used, denormalized for the same reason
    # `file_name` above is: cancelling purges the resource and its `files` rows, and the
    # report of what was attempted has to outlive them. It records what was *requested*;
    # `detail` records what actually happened, including a strategy that had to degrade.
    config_json = Column(Text, nullable=True)
    # The one line a user reads to understand this document: "14 pages · 47 passages", or
    # "No text layer — this looks like a scanned PDF, so nothing could be indexed."
    detail = Column(Text, nullable=True)

    started_at = Column(DateTime, nullable=True)
    finished_at = Column(DateTime, nullable=True)

    job = relationship("IngestionJob", back_populates="items")
