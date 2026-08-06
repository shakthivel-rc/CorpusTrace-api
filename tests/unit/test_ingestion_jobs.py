"""Background document indexing: staging, progress, per-document reporting, cancellation.

Indexing moved out of the upload request so it could be watched and stopped. That buys
three properties worth pinning here, because each one fails silently if it regresses:

* the request must not index — if staging ever starts parsing again, uploads go back to
  being a multi-minute blocking call and nobody notices until a large PDF arrives;
* a cancelled job must leave nothing behind — a half-indexed base answers questions
  confidently from part of the evidence;
* a document that yielded no text must say so — that is the whole point of per-document
  status, and it is exactly the case the old boolean `upload_status` could not express.
"""
import dataclasses
import io
from pathlib import Path

import pytest
from fastapi import UploadFile

import rag.jobs as jobs
import rag.service as rag_service
from core.config import get_settings
from models.activity_log import ActivityLog
from models.file import File
from models.ingestion import (
    ITEM_CANCELLED,
    ITEM_FAILED,
    ITEM_PARTIAL,
    ITEM_SUCCEEDED,
    JOB_CANCELLED,
    JOB_FAILED,
    JOB_QUEUED,
    JOB_RUNNING,
    JOB_SUCCEEDED,
    IngestionJob,
    IngestionJobItem,
)
from models.rag import DocumentChunk
from models.resource import Resource

pytestmark = pytest.mark.unit

OWNER = "user-1"
STRANGER = "user-2"


@pytest.fixture()
def upload_root(tmp_path, monkeypatch):
    """Point ingestion at a tmp directory — staging writes real files and cancel deletes
    them. Settings are a frozen lru_cached dataclass, so replace the accessor."""
    root = tmp_path / "uploads" / "rag"
    root.mkdir(parents=True)
    patched = dataclasses.replace(get_settings(), rag_upload_dir=str(root))
    monkeypatch.setattr(rag_service, "get_settings", lambda: patched)
    return root


def _upload(name: str, body: bytes, content_type: str = "text/plain") -> UploadFile:
    return UploadFile(
        filename=name,
        file=io.BytesIO(body),
        headers={"content-type": content_type},
    )


def _text(words: str = "valve clearance", repeat: int = 40) -> bytes:
    return (f"{words} inspection procedure. " * repeat).encode()


def _enqueue(db, uploads, name="Daytona") -> IngestionJob:
    job = jobs.enqueue_ingestion(db, OWNER, name, uploads)
    db.commit()
    return job


class TestStaging:
    def test_the_request_stores_bytes_but_indexes_nothing(self, db, upload_root):
        """The point of the split. If staging indexes, the upload blocks again."""
        job = _enqueue(db, [_upload("manual.txt", _text())])

        assert job.status == JOB_QUEUED
        assert job.total_documents == 1
        assert db.query(File).count() == 1
        assert db.query(DocumentChunk).count() == 0

        resource = db.query(Resource).filter(Resource.id == job.resource_id).one()
        # Not answerable yet, and `_not_ready_note` reads exactly this.
        assert resource.upload_status is False

    def test_records_one_item_per_document_in_upload_order(self, db, upload_root):
        job = _enqueue(db, [_upload("a.txt", _text()), _upload("b.txt", _text()), _upload("c.txt", _text())])
        items = db.query(IngestionJobItem).filter(IngestionJobItem.job_id == job.id).order_by(
            IngestionJobItem.position
        ).all()
        assert [item.file_name for item in items] == ["a.txt", "b.txt", "c.txt"]
        assert [item.position for item in items] == [0, 1, 2]

    def test_records_the_size_of_each_document(self, db, upload_root):
        body = _text()
        job = _enqueue(db, [_upload("manual.txt", body)])
        item = db.query(IngestionJobItem).filter(IngestionJobItem.job_id == job.id).one()
        assert item.byte_size == len(body)
        assert job.total_bytes == len(body)

    def test_rejects_an_empty_upload(self, db, upload_root):
        with pytest.raises(ValueError):
            jobs.enqueue_ingestion(db, OWNER, "Empty", [])

    def test_copies_the_name_so_the_job_survives_the_resource(self, db, upload_root):
        job = _enqueue(db, [_upload("manual.txt", _text())], name="Daytona 675")
        assert job.resource_name == "Daytona 675"


class TestClaiming:
    def test_claims_the_oldest_queued_job(self, db, upload_root):
        first = _enqueue(db, [_upload("a.txt", _text())], name="First")
        _enqueue(db, [_upload("b.txt", _text())], name="Second")

        claimed = jobs.claim_next_job(db, worker_id="worker-a")
        assert claimed.id == first.id
        assert claimed.status == JOB_RUNNING
        assert claimed.claimed_by == "worker-a"
        assert claimed.started_at is not None

    def test_a_claimed_job_is_not_claimed_again(self, db, upload_root):
        """Two API processes poll the same table. Double-claiming means double-indexing."""
        _enqueue(db, [_upload("a.txt", _text())])
        assert jobs.claim_next_job(db, worker_id="worker-a") is not None
        assert jobs.claim_next_job(db, worker_id="worker-b") is None

    def test_returns_none_when_the_queue_is_empty(self, db):
        assert jobs.claim_next_job(db) is None


class TestRunning:
    def test_indexes_every_document_and_makes_the_base_answerable(self, db, upload_root):
        job = _enqueue(db, [_upload("a.txt", _text("valve")), _upload("b.txt", _text("brake"))])
        resource_id = job.resource_id

        jobs.run_job(db, jobs.claim_next_job(db))

        assert job.status == JOB_SUCCEEDED
        assert job.processed_documents == 2
        assert job.indexed_chunks > 0
        assert db.query(DocumentChunk).filter(DocumentChunk.resource_id == resource_id).count() == job.indexed_chunks
        assert db.query(Resource).filter(Resource.id == resource_id).one().upload_status is True

    def test_chunk_index_continues_across_documents(self, db, upload_root):
        """`chunk_index` is the position within the resource; retrieval orders by it."""
        job = _enqueue(db, [_upload("a.txt", _text(repeat=200)), _upload("b.txt", _text(repeat=200))])
        jobs.run_job(db, jobs.claim_next_job(db))

        indexes = [
            row.chunk_index
            for row in db.query(DocumentChunk)
            .filter(DocumentChunk.resource_id == job.resource_id)
            .order_by(DocumentChunk.chunk_index)
            .all()
        ]
        assert indexes == list(range(len(indexes)))

    def test_reports_pages_and_passages_per_document(self, db, upload_root, make_pdf):
        pdf = make_pdf([["Valve clearance is checked cold."], ["Torque the cover to 10 Nm."]])
        job = _enqueue(db, [_upload("manual.pdf", pdf, "application/pdf")])
        jobs.run_job(db, jobs.claim_next_job(db))

        item = db.query(IngestionJobItem).filter(IngestionJobItem.job_id == job.id).one()
        assert item.status == ITEM_SUCCEEDED
        assert item.page_count == 2
        assert item.chunk_count > 0
        assert "2 pages" in item.detail

    def test_a_pdf_with_no_text_layer_is_reported_as_partial(self, db, upload_root, make_pdf):
        """A scanned PDF indexes an explanation, not the document. Calling that success is
        how a user ends up asking questions of a knowledge base that contains nothing."""
        blank = make_pdf([[], []])
        job = _enqueue(db, [_upload("scan.pdf", blank, "application/pdf")])
        jobs.run_job(db, jobs.claim_next_job(db))

        item = db.query(IngestionJobItem).filter(IngestionJobItem.job_id == job.id).one()
        assert item.status == ITEM_PARTIAL
        assert "no" in item.detail.lower() and "text" in item.detail.lower()
        # The job still succeeds — the other documents in a batch are unaffected.
        assert job.status == JOB_SUCCEEDED
        assert "no readable text" in job.message

    def test_one_unreadable_document_does_not_sink_the_batch(self, db, upload_root, monkeypatch):
        job = _enqueue(db, [_upload("good.txt", _text()), _upload("bad.txt", _text())])

        real = rag_service.ingest_file

        def explode(db_, resource, file_record, start_index, should_cancel=None):
            if file_record.file_name == "bad.txt":
                raise RuntimeError("disk read failed")
            return real(db_, resource, file_record, start_index, should_cancel)

        monkeypatch.setattr(jobs, "ingest_file", explode)
        jobs.run_job(db, jobs.claim_next_job(db))

        statuses = {
            item.file_name: item.status
            for item in db.query(IngestionJobItem).filter(IngestionJobItem.job_id == job.id).all()
        }
        assert statuses == {"good.txt": ITEM_SUCCEEDED, "bad.txt": ITEM_FAILED}
        assert job.status == JOB_SUCCEEDED
        assert "could not be read" in job.message
        assert job.processed_documents == 2

    def test_a_missing_resource_fails_the_job_rather_than_raising(self, db, upload_root):
        job = _enqueue(db, [_upload("a.txt", _text())])
        claimed = jobs.claim_next_job(db)
        db.query(IngestionJobItem).filter(IngestionJobItem.job_id == job.id).delete()
        db.query(File).delete()
        db.query(Resource).filter(Resource.id == job.resource_id).delete()
        db.commit()

        jobs.run_job(db, claimed)
        assert job.status == JOB_FAILED
        assert job.message


class TestCancellation:
    def test_cancelling_a_queued_job_discards_everything(self, db, upload_root):
        """Cancel means "as if it never ran". A partial base is the dangerous outcome."""
        job = _enqueue(db, [_upload("a.txt", _text())])
        resource_id = job.resource_id
        directory = upload_root / resource_id
        assert directory.exists()

        jobs.request_cancel(db, OWNER, job.id)

        assert job.status == JOB_CANCELLED
        assert job.resource_id is None
        assert db.query(Resource).filter(Resource.id == resource_id).first() is None
        assert db.query(File).count() == 0
        assert db.query(DocumentChunk).count() == 0
        assert not directory.exists()

    def test_the_job_survives_as_the_record_of_what_happened(self, db, upload_root):
        job = _enqueue(db, [_upload("a.txt", _text())], name="Daytona 675")
        jobs.request_cancel(db, OWNER, job.id)

        kept = db.query(IngestionJob).filter(IngestionJob.id == job.id).one()
        assert kept.resource_name == "Daytona 675"
        assert kept.status == JOB_CANCELLED
        items = db.query(IngestionJobItem).filter(IngestionJobItem.job_id == job.id).all()
        assert [item.status for item in items] == [ITEM_CANCELLED]

    def test_a_running_job_is_only_flagged_and_stops_at_its_next_checkpoint(self, db, upload_root):
        job = _enqueue(db, [_upload("a.txt", _text()), _upload("b.txt", _text())])
        claimed = jobs.claim_next_job(db)

        flagged = jobs.request_cancel(db, OWNER, job.id)
        db.commit()
        # Still running: the worker holds the half-written rows, so it does the discarding.
        assert flagged.status == JOB_RUNNING
        assert flagged.cancel_requested is True

        jobs.run_job(db, claimed)
        assert job.status == JOB_CANCELLED
        assert db.query(DocumentChunk).count() == 0
        assert db.query(Resource).count() == 0

    def test_cancellation_is_checked_inside_a_long_document(self, db, upload_root, monkeypatch):
        """One big PDF is the common case. Checking only between documents would make
        Cancel do nothing at all for the entire upload."""
        job = _enqueue(db, [_upload("big.txt", _text(repeat=4000))])
        claimed = jobs.claim_next_job(db)

        calls = {"n": 0}
        real = jobs._cancel_requested

        def cancel_after_first_batch(db_, job_id):
            calls["n"] += 1
            return calls["n"] > 1

        monkeypatch.setattr(jobs, "_cancel_requested", cancel_after_first_batch)
        jobs.run_job(db, claimed)

        assert job.status == JOB_CANCELLED
        assert calls["n"] > 1  # the in-document poll fired, not just the between-documents one
        assert real is not jobs._cancel_requested

    def test_a_finished_job_cannot_be_cancelled(self, db, upload_root):
        job = _enqueue(db, [_upload("a.txt", _text())])
        jobs.run_job(db, jobs.claim_next_job(db))

        with pytest.raises(ValueError):
            jobs.request_cancel(db, OWNER, job.id)

    def test_someone_elses_job_is_reported_as_missing(self, db, upload_root):
        job = _enqueue(db, [_upload("a.txt", _text())])
        with pytest.raises(LookupError):
            jobs.request_cancel(db, STRANGER, job.id)


class TestReclaim:
    def test_a_job_left_running_by_a_dead_process_is_failed_at_startup(self, db, upload_root):
        """An in-process worker cannot survive a restart. Leaving the row `running` shows
        a progress bar that will never move again."""
        job = _enqueue(db, [_upload("a.txt", _text())])
        jobs.claim_next_job(db)

        assert jobs.reclaim_stranded_jobs(db) == 1
        db.refresh(job)
        assert job.status == JOB_FAILED
        assert "restart" in job.message.lower()

    def test_leaves_queued_and_finished_jobs_alone(self, db, upload_root):
        queued = _enqueue(db, [_upload("a.txt", _text())])
        assert jobs.reclaim_stranded_jobs(db) == 0
        db.refresh(queued)
        assert queued.status == JOB_QUEUED


class TestActivityLog:
    def test_every_terminal_outcome_is_logged(self, db, upload_root):
        done = _enqueue(db, [_upload("a.txt", _text())], name="Indexed one")
        jobs.run_job(db, jobs.claim_next_job(db))
        cancelled = _enqueue(db, [_upload("b.txt", _text())], name="Cancelled one")
        jobs.request_cancel(db, OWNER, cancelled.id)

        entries = {
            row.entity_id: row
            for row in db.query(ActivityLog).filter(ActivityLog.entity_type == "ingestion_job").all()
        }
        assert entries[done.id].action == "INDEXED"
        assert "Indexed one" in entries[done.id].details
        assert entries[cancelled.id].action == "CANCELLED"


class TestSerialization:
    def test_progress_holds_back_the_last_slice_for_linking(self, db, upload_root):
        """A bar that reads 100% while work continues is how progress bars lose trust."""
        job = _enqueue(db, [_upload("a.txt", _text()), _upload("b.txt", _text())])
        assert jobs._progress(job) == 0

        job.status = JOB_RUNNING
        job.processed_documents = 2
        assert jobs._progress(job) == 95

        job.status = JOB_SUCCEEDED
        assert jobs._progress(job) == 100

    def test_serializes_the_per_document_breakdown(self, db, upload_root):
        job = _enqueue(db, [_upload("a.txt", _text())])
        jobs.run_job(db, jobs.claim_next_job(db))

        payload = jobs.serialize_job(job)
        assert payload["status"] == JOB_SUCCEEDED
        assert payload["progress"] == 100
        assert len(payload["documents"]) == 1
        assert payload["documents"][0]["file_name"] == "a.txt"
        assert payload["documents"][0]["chunk_count"] > 0

    def test_the_list_payload_omits_documents(self, db, upload_root):
        job = _enqueue(db, [_upload("a.txt", _text())])
        assert "documents" not in jobs.serialize_job(job, include_items=False)

    def test_latest_job_per_resource_in_one_pass(self, db, upload_root):
        first = _enqueue(db, [_upload("a.txt", _text())], name="A")
        second = _enqueue(db, [_upload("b.txt", _text())], name="B")

        found = jobs.latest_jobs_for_resources(db, [first.resource_id, second.resource_id])
        assert found[first.resource_id].id == first.id
        assert found[second.resource_id].id == second.id
        assert jobs.latest_jobs_for_resources(db, []) == {}


class TestChatIsBlockedUntilIndexingFinishes:
    def test_a_still_indexing_base_refuses_rather_than_answering_from_part_of_it(self, db, upload_root):
        job = _enqueue(db, [_upload("a.txt", _text()), _upload("b.txt", _text())])
        resource = db.query(Resource).filter(Resource.id == job.resource_id).one()

        note = rag_service._not_ready_note(db, resource)
        assert note is not None
        assert "still being indexed" in note

    def test_a_cancelled_base_says_so(self, db, upload_root):
        job = _enqueue(db, [_upload("a.txt", _text())])
        resource = db.query(Resource).filter(Resource.id == job.resource_id).one()
        resource.upload_status = False
        # The resource is gone after a real cancel; simulate the window where it is not.
        job.status = JOB_CANCELLED
        db.flush()

        assert "cancelled" in rag_service._not_ready_note(db, resource).lower()

    def test_a_finished_base_answers_normally(self, db, upload_root):
        job = _enqueue(db, [_upload("a.txt", _text())])
        jobs.run_job(db, jobs.claim_next_job(db))
        resource = db.query(Resource).filter(Resource.id == job.resource_id).one()

        assert rag_service._not_ready_note(db, resource) is None
