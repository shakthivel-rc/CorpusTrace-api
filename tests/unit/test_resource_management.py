"""Renaming, deleting, restoring and purging a knowledge base.

Uploading used to be one-way: there was no endpoint that modified or removed a resource,
so a mistyped name or an unwanted document stayed forever. The rules that matter here are
that a soft delete must disappear from retrieval without touching a chunk, that a purge
must not leave orphans scoring against future questions, and that `shutil.rmtree` is
never pointed at a path derived from the wire without proof it sits under the upload root.
"""
import dataclasses
import json
import shutil
from pathlib import Path

import pytest

import rag.service as rag_service
from core.config import get_settings
from models.file import File
from models.rag import DocumentChunk, RagGraphEdge, RagGraphEntity
from models.resource import Resource

pytestmark = pytest.mark.unit

OWNER = "user-1"
STRANGER = "user-2"


def _seed(db, upload_root: Path, name: str = "Daytona") -> Resource:
    resource = Resource(resource_name=name, user_id=OWNER, upload_status=True)
    db.add(resource)
    db.flush()

    directory = upload_root / resource.id
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "daytona.pdf").write_bytes(b"%PDF-1.4 fake")

    file_row = File(
        file_name="daytona.pdf",
        file_type="application/pdf",
        file_url=str(directory / "daytona.pdf"),
        resource_id=resource.id,
    )
    db.add(file_row)
    db.flush()
    db.add(
        DocumentChunk(
            resource_id=resource.id,
            file_id=file_row.id,
            chunk_index=0,
            source_name="daytona.pdf",
            modality="pdf",
            content="Valve clearance check.",
            contextual_content="Valve clearance check.",
            terms_json=json.dumps({"valve": 1, "clearance": 1}),
        )
    )
    entity = RagGraphEntity(resource_id=resource.id, name="valve", chunk_refs_json="[]")
    db.add(entity)
    db.flush()
    db.add(
        RagGraphEdge(resource_id=resource.id, source_entity_id=entity.id, target_entity_id=entity.id)
    )
    db.flush()
    return resource


@pytest.fixture()
def upload_root(tmp_path, monkeypatch):
    """Point the upload directory at a tmp path — purge deletes real files.

    Settings are a frozen, lru_cached dataclass, so this replaces the accessor rather
    than mutating it (the same pattern as tests/integration/test_source_evidence.py).
    """
    root = tmp_path / "uploads" / "rag"
    root.mkdir(parents=True)
    patched = dataclasses.replace(get_settings(), rag_upload_dir=str(root))
    monkeypatch.setattr(rag_service, "get_settings", lambda: patched)
    return root


class TestRename:
    def test_renames_and_persists(self, db, upload_root):
        resource = _seed(db, upload_root)
        rag_service.rename_resource(db, OWNER, resource.id, "  Triumph manual  ")
        assert resource.resource_name == "Triumph manual"

    def test_rejects_a_blank_name(self, db, upload_root):
        resource = _seed(db, upload_root)
        with pytest.raises(ValueError):
            rag_service.rename_resource(db, OWNER, resource.id, "   ")

    def test_bounds_the_name_to_the_column_width(self, db, upload_root):
        # SQLite ignores VARCHAR limits, so assert the length directly — CLAUDE.md §14.
        resource = _seed(db, upload_root)
        rag_service.rename_resource(db, OWNER, resource.id, "x" * 900)
        assert len(resource.resource_name) == rag_service.MAX_VARCHAR_CHARS

    def test_a_stranger_cannot_rename(self, db, upload_root):
        resource = _seed(db, upload_root)
        with pytest.raises(LookupError):
            rag_service.rename_resource(db, STRANGER, resource.id, "mine now")
        assert resource.resource_name == "Daytona"


class TestSoftDelete:
    def test_disappears_from_listing_and_from_chat(self, db, upload_root):
        resource = _seed(db, upload_root)
        rag_service.soft_delete_resource(db, OWNER, resource.id)

        assert rag_service.list_user_resources(db, OWNER) == []
        # The accessor chat and retrieval use — the whole reason a soft delete is enough.
        assert rag_service.get_user_resource(db, OWNER, resource.id) is None

    def test_keeps_every_chunk_and_file(self, db, upload_root):
        resource = _seed(db, upload_root)
        rag_service.soft_delete_resource(db, OWNER, resource.id)

        assert db.query(DocumentChunk).filter_by(resource_id=resource.id).count() == 1
        assert (upload_root / resource.id / "daytona.pdf").exists()

    def test_shows_up_in_the_deleted_listing_and_restores(self, db, upload_root):
        resource = _seed(db, upload_root)
        rag_service.soft_delete_resource(db, OWNER, resource.id)
        assert [r.id for r in rag_service.list_user_resources(db, OWNER, deleted=True)] == [resource.id]

        rag_service.restore_resource(db, OWNER, resource.id)
        assert [r.id for r in rag_service.list_user_resources(db, OWNER)] == [resource.id]
        assert rag_service.list_user_resources(db, OWNER, deleted=True) == []

    def test_a_stranger_cannot_delete(self, db, upload_root):
        resource = _seed(db, upload_root)
        with pytest.raises(LookupError):
            rag_service.soft_delete_resource(db, STRANGER, resource.id)
        assert resource.deleted_at is None


class TestPurge:
    def test_removes_every_row_and_the_uploaded_files(self, db, upload_root):
        resource = _seed(db, upload_root)
        rag_service.soft_delete_resource(db, OWNER, resource.id)

        result = rag_service.purge_resource(db, OWNER, resource.id)

        assert result["chunks_deleted"] == 1
        assert result["files_deleted"] == 1
        assert result["graph_entities_deleted"] == 1
        assert result["graph_edges_deleted"] == 1
        assert result["uploaded_files_removed"] is True

        assert db.query(Resource).filter_by(id=resource.id).count() == 0
        # Orphaned chunks would keep scoring against every future question.
        assert db.query(DocumentChunk).filter_by(resource_id=resource.id).count() == 0
        assert db.query(File).filter_by(resource_id=resource.id).count() == 0
        assert db.query(RagGraphEntity).filter_by(resource_id=resource.id).count() == 0
        assert db.query(RagGraphEdge).filter_by(resource_id=resource.id).count() == 0
        assert not (upload_root / resource.id).exists()

    def test_purges_an_active_resource_too(self, db, upload_root):
        # The UI routes through delete first, but the endpoint must not depend on that.
        resource = _seed(db, upload_root)
        rag_service.purge_resource(db, OWNER, resource.id)
        assert db.query(Resource).filter_by(id=resource.id).count() == 0

    def test_a_stranger_cannot_purge(self, db, upload_root):
        resource = _seed(db, upload_root)
        with pytest.raises(LookupError):
            rag_service.purge_resource(db, STRANGER, resource.id)
        assert db.query(Resource).filter_by(id=resource.id).count() == 1
        assert (upload_root / resource.id / "daytona.pdf").exists()

    def test_survives_a_resource_whose_files_are_already_gone(self, db, upload_root):
        resource = _seed(db, upload_root)
        shutil.rmtree(upload_root / resource.id)
        result = rag_service.purge_resource(db, OWNER, resource.id)
        assert result["uploaded_files_removed"] is False
        assert db.query(Resource).filter_by(id=resource.id).count() == 0


class TestUploadDirectorySafety:
    """`shutil.rmtree` on an unvalidated path is the one mistake here that cannot be undone."""

    def test_rejects_traversal_out_of_the_upload_root(self, upload_root):
        for attempt in ["..", "../..", "../evil", "a/../..", "/etc"]:
            with pytest.raises(ValueError):
                rag_service._resource_upload_dir(attempt)

    def test_rejects_the_root_itself(self, upload_root):
        # Resolving to the root would delete every user's documents in one call.
        with pytest.raises(ValueError):
            rag_service._resource_upload_dir(".")

    def test_accepts_a_normal_resource_id(self, upload_root):
        target = rag_service._resource_upload_dir("2b0f6a1e-0000-4000-8000-000000000000")
        assert target.parent == upload_root.resolve()


class TestResourceCounts:
    def test_counts_files_and_chunks_per_resource(self, db, upload_root):
        first = _seed(db, upload_root, "Daytona")
        second = _seed(db, upload_root, "Street Triple")
        counts = rag_service.resource_counts(db, [first.id, second.id])
        assert counts[first.id] == {"file_count": 1, "chunk_count": 1}
        assert counts[second.id] == {"file_count": 1, "chunk_count": 1}

    def test_reports_zero_rather_than_omitting_an_empty_resource(self, db):
        resource = Resource(resource_name="Empty", user_id=OWNER, upload_status=False)
        db.add(resource)
        db.flush()
        assert rag_service.resource_counts(db, [resource.id])[resource.id] == {
            "file_count": 0,
            "chunk_count": 0,
        }

    def test_handles_an_empty_list(self, db):
        assert rag_service.resource_counts(db, []) == {}
