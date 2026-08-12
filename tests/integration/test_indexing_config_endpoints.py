"""The HTTP surface of per-document indexing settings: the catalogue, the upload, the list.

Per-document settings replaced a single global chunking rule, and the whole point is that
the Nth entry of `document_configs` configures the Nth file. Three properties are worth
pinning here because each one fails quietly rather than loudly:

* an upload that sends no settings must still be indexed exactly as it was before this
  feature existed — every existing client sends none, and a changed default would silently
  re-slice every future knowledge base;
* a settings blob the user never typed must never cost them the upload — malformed input
  falls back to defaults rather than 400-ing a good batch of documents;
* embeddings must be off unless explicitly asked for — turning them on sends document text
  to a third party and, on most providers, spends money.

Auth setup is copied from test_ingestion_endpoints.py rather than imported: `tests/` is not
a package (see the note in conftest.py's make_pdf), so there is no import path between test
modules and duplication is the established convention here.
"""
import dataclasses
import io
import json
import time

import jwt
import pytest

import rag.jobs as jobs
import rag.service as rag_service
from services.llm_provider import EMBED_TASK_DOCUMENT
from core.config import get_settings
from models.file import File
from models.permissions import Permission
from models.rag import DocumentChunk
from models.resource import Resource
from models.role_permissions import RolePermission
from models.roles import Role
from models.user import User
from models.user_roles import UserRole
from models.user_session import UserSession
from rag import chunking
from rag.service import SUPPORTED_RAG_MODES
from utils.password import hash_password

pytestmark = pytest.mark.integration

# --------------------------------------------------------------------------------------
# Two bugs were found while writing this file and have since been fixed:
#
#   1. get_indexing_options() called .get() on SUPPORTED_RAG_MODES, which is a set — that
#      made GET /resources/indexing-options an unconditional HTTP 500, and it is the
#      endpoint the entire upload form is built from.
#   2. stage_upload stored embedding_provider/embedding_model unbounded into VARCHAR(50)
#      and VARCHAR(255), so a long value from client JSON was MySQL error 1406 and, with
#      no global exception handler, a 500 that lost the whole upload.
#
# Both are now covered by ordinary passing tests below. Note SQLite ignores VARCHAR limits,
# so the second one is asserted on the stored LENGTH rather than by expecting a DB error —
# see CLAUDE.md §10.



def _seed_ai_user(db, user_id: str = "config-user") -> str:
    settings = get_settings()
    token = jwt.encode(
        {"sub": user_id, "type": "access", "exp": int(time.time()) + 3600},
        settings.secret_key,
        algorithm=settings.algorithm,
    )
    db.add(
        User(
            id=user_id,
            email=f"{user_id}@example.com",
            username=user_id,
            first_name="Config",
            last_name="User",
            organization="Org",
            department="Dept",
            password=hash_password("Abcdef123!@#"),
            status=True,
            deleted=0,
        )
    )
    role = Role(name=f"AI User {user_id}")
    db.add(role)
    db.flush()
    permission = db.query(Permission).filter(Permission.machine_name == "ai_access").first()
    if not permission:
        permission = Permission(name="AI Access", machine_name="ai_access")
        db.add(permission)
        db.flush()
    db.add_all(
        [
            UserRole(user_id=user_id, role_id=role.id),
            RolePermission(role_id=role.id, permission_id=permission.id),
            UserSession(user_id=user_id, access_token=token),
        ]
    )
    db.commit()
    return token


@pytest.fixture()
def upload_dir(tmp_path, monkeypatch):
    """Point staging at a tmp directory — the upload really writes bytes to disk."""
    root = tmp_path / "uploads" / "rag"
    root.mkdir(parents=True)
    patched = dataclasses.replace(get_settings(), rag_upload_dir=str(root))
    monkeypatch.setattr(rag_service, "get_settings", lambda: patched)
    return root


@pytest.fixture()
def no_embedding_calls(monkeypatch):
    """Fail loudly if anything reaches the embedding provider.

    Installed by default for every upload test in this module: an upload that quietly
    embedded would send the user's document text to a third party and bill them for it,
    and the only visible symptom would be a slower upload.
    """
    calls: list[tuple] = []

    def _forbidden(db, user_id, provider, model_id, texts):
        calls.append((provider, model_id, len(texts)))
        raise AssertionError(
            f"ingestion called the embedding provider ({provider}/{model_id}) without being asked to"
        )

    monkeypatch.setattr(rag_service, "embed_texts", _forbidden)
    return calls


def _text(words: str = "valve clearance", repeat: int = 60) -> bytes:
    return (f"{words} inspection procedure. " * repeat).encode()


def _post_upload(client, token, name="Daytona", files=None, document_configs=None):
    data = {"resource_name": name}
    if document_configs is not None:
        data["document_configs"] = document_configs
    return client.post(
        "/resources/upload_files",
        data=data,
        files=files or [("files", ("manual.txt", io.BytesIO(_text()), "text/plain"))],
        headers={"Authorization": f"Bearer {token}"},
    )


def _files_in_upload_order(db, resource_id: str) -> list[File]:
    return (
        db.query(File)
        .filter(File.resource_id == resource_id)
        .order_by(File.created_at.asc(), File.file_name.asc())
        .all()
    )


class TestIndexingOptionsCatalogue:
    """The catalogue is served rather than duplicated in TypeScript precisely so it cannot
    drift from the chunker. These assert it actually arrives.

    Every test here needs a 200 from the catalogue endpoint: it is the single request the
    entire upload form is built from, so a failure anywhere in it leaves the settings UI
    with nothing to render.
    """

    def test_the_catalogue_carries_everything_the_upload_form_renders(self, client, db):
        # A missing key here is a settings form that renders an empty dropdown — the user
        # cannot choose a strategy at all, and nothing in the UI says why.
        token = _seed_ai_user(db)
        response = client.get(
            "/resources/indexing-options", headers={"Authorization": f"Bearer {token}"}
        )

        assert response.status_code == 200
        body = response.json()
        assert set(body) == {"status", "status_code", "message", "data"}
        assert body["status"] == "success"
        assert body["status_code"] == 200

        data = body["data"]
        assert set(data) >= {
            "strategies",
            "chunk_sizes",
            "overlaps",
            "bounds",
            "defaults",
            "embedding_providers",
            "recommendations",
        }
        assert {s["id"] for s in data["strategies"]} == set(chunking.STRATEGIES)
        assert data["bounds"]["min_chunk_size"] == chunking.MIN_CHUNK_SIZE
        assert data["bounds"]["max_chunk_size"] == chunking.MAX_CHUNK_SIZE

    def test_the_advertised_defaults_are_the_ones_an_unconfigured_upload_gets(self, client, db):
        # The form pre-selects these. If they drift from DEFAULT_CONFIG the UI shows one
        # setting and the server applies another, which is unfalsifiable from the outside.
        token = _seed_ai_user(db)
        data = client.get(
            "/resources/indexing-options", headers={"Authorization": f"Bearer {token}"}
        ).json()["data"]

        assert data["defaults"] == chunking.DEFAULT_CONFIG.to_dict()
        assert data["defaults"]["embedding_provider"] is None
        assert data["defaults"]["embedding_model"] is None

    def test_every_rag_mode_the_app_supports_has_a_recommendation(self, client, db):
        # The UI reads recommendations[current_mode]. A mode with no entry is a KeyError in
        # the SPA — or, worse, a silently missing suggestion for the one mode a user picked.
        token = _seed_ai_user(db)
        data = client.get(
            "/resources/indexing-options", headers={"Authorization": f"Bearer {token}"}
        ).json()["data"]

        assert set(data["recommendations"]) == set(SUPPORTED_RAG_MODES)
        for mode, entry in data["recommendations"].items():
            assert entry["strategy"] in chunking.STRATEGIES, mode
            assert entry["why"]

    def test_every_offered_embedding_provider_offers_at_least_one_model(self, client, db):
        # `embeds` needs a provider *and* a model. A provider row with no models is a choice
        # the user can make that can never take effect.
        token = _seed_ai_user(db)
        data = client.get(
            "/resources/indexing-options", headers={"Authorization": f"Bearer {token}"}
        ).json()["data"]

        assert data["embedding_providers"]
        for option in data["embedding_providers"]:
            assert option["models"], option["provider"]

    def test_requires_authentication(self, client, db):
        # The catalogue is behind ai_access like every other resources route; an
        # unauthenticated 200 here would be the first crack in that. Not xfail: the 401 is
        # decided by JWTAuthMiddleware, so it never reaches the broken handler.
        assert client.get("/resources/indexing-options").status_code == 401

    def test_the_catalogue_data_itself_covers_every_rag_mode(self):
        """The same guarantee as the endpoint test above, asserted against the source data
        so it is genuinely covered while that endpoint is broken. A mode with no
        recommendation silently gives the user no guidance for the mode they picked."""
        assert set(chunking.RAG_MODE_RECOMMENDATIONS) == set(SUPPORTED_RAG_MODES)
        for mode in SUPPORTED_RAG_MODES:
            entry = chunking.recommendation_for(mode)
            assert entry["strategy"] in chunking.STRATEGIES, mode
            assert chunking.MIN_CHUNK_SIZE <= entry["chunk_size"] <= chunking.MAX_CHUNK_SIZE, mode
            # A recommendation the UI cannot explain is a number the user has no reason to
            # trust — the copy is the point of recommending anything at all.
            assert entry["why"], mode


class TestUploadWithoutSettings:
    def test_an_upload_that_sends_no_settings_is_indexed_the_way_it_always_was(
        self, client, db, upload_dir, no_embedding_calls
    ):
        """The backwards-compatibility guarantee. Every client written before this feature
        sends no document_configs, and a changed default silently re-slices their bases."""
        token = _seed_ai_user(db)
        response = _post_upload(client, token)

        assert response.status_code == 202
        resource_id = response.json()["data"]["id"]

        file_record = _files_in_upload_order(db, resource_id)[0]
        assert file_record.chunk_strategy == "character"
        assert file_record.chunk_size == 1200
        assert file_record.chunk_overlap == 180
        # NULL, not "" — NULL is what `config_for_file` reads as "lexical only", and it is
        # what stops ingestion calling a provider at all.
        assert file_record.embedding_provider is None
        assert file_record.embedding_model is None

    def test_an_empty_settings_field_is_the_same_as_sending_none(
        self, client, db, upload_dir, no_embedding_calls
    ):
        # A form that serializes an untouched settings panel to "" must not be treated as
        # a configuration — this is the shape a browser sends for a disabled control.
        token = _seed_ai_user(db)
        resource_id = _post_upload(client, token, document_configs="").json()["data"]["id"]

        file_record = _files_in_upload_order(db, resource_id)[0]
        assert (file_record.chunk_strategy, file_record.chunk_size, file_record.chunk_overlap) == (
            "character",
            1200,
            180,
        )


class TestUploadWithPerDocumentSettings:
    def test_each_document_is_indexed_under_its_own_settings(
        self, client, db, upload_dir, no_embedding_calls
    ):
        """The actual feature. One global setting is what this replaced, so two documents in
        one batch disagreeing is the thing that has to work."""
        token = _seed_ai_user(db)
        configs = json.dumps(
            [
                {"strategy": "sentence", "chunk_size": 600, "overlap": 0},
                {"strategy": "paragraph", "chunk_size": 2000, "overlap": 300},
            ]
        )
        response = _post_upload(
            client,
            token,
            files=[
                ("files", ("policy.txt", io.BytesIO(_text("clause obligation")), "text/plain")),
                ("files", ("faq.txt", io.BytesIO(_text("brake fluid")), "text/plain")),
            ],
            document_configs=configs,
        )
        assert response.status_code == 202

        first, second = _files_in_upload_order(db, response.json()["data"]["id"])
        assert first.file_name == "policy.txt"
        assert (first.chunk_strategy, first.chunk_size, first.chunk_overlap) == ("sentence", 600, 0)
        assert (second.chunk_strategy, second.chunk_size, second.chunk_overlap) == (
            "paragraph",
            2000,
            300,
        )

    def test_the_settings_actually_reach_the_chunker(
        self, client, db, upload_dir, no_embedding_calls
    ):
        # Storing the settings on the File row and never consulting them would pass every
        # assertion above while leaving the feature entirely decorative. A 200-character
        # budget over the same bytes must produce strictly more chunks than 4000.
        token = _seed_ai_user(db)
        body = _text(repeat=120)
        response = _post_upload(
            client,
            token,
            files=[
                ("files", ("small.txt", io.BytesIO(body), "text/plain")),
                ("files", ("large.txt", io.BytesIO(body), "text/plain")),
            ],
            document_configs=json.dumps([{"chunk_size": 200}, {"chunk_size": 4000}]),
        )
        jobs.run_job(db, jobs.claim_next_job(db))

        small, large = _files_in_upload_order(db, response.json()["data"]["id"])
        counted = {
            file_record.file_name: db.query(DocumentChunk)
            .filter(DocumentChunk.file_id == file_record.id)
            .count()
            for file_record in (small, large)
        }
        assert counted["small.txt"] > counted["large.txt"] > 0

    def test_the_job_item_snapshots_the_settings_each_document_was_queued_with(
        self, client, db, upload_dir, no_embedding_calls
    ):
        # Cancelling purges the resource and its File rows; the job item is the only record
        # left of what was attempted, so the snapshot has to be taken at enqueue time.
        token = _seed_ai_user(db)
        job = _post_upload(
            client,
            token,
            document_configs=json.dumps([{"strategy": "sentence", "chunk_size": 600}]),
        ).json()["data"]["job"]

        detail = client.get(
            f"/resources/jobs/{job['id']}", headers={"Authorization": f"Bearer {token}"}
        ).json()["data"]
        assert detail["documents"][0]["config"]["strategy"] == "sentence"
        assert detail["documents"][0]["config"]["chunk_size"] == 600


class TestMalformedSettingsAreForgiven:
    @pytest.mark.parametrize(
        "raw",
        [
            pytest.param("this is not json", id="not-json"),
            pytest.param('{"strategy": "sentence"}', id="object-not-list"),
            pytest.param('["sentence", 600]', id="list-of-scalars"),
            pytest.param("[]", id="empty-list"),
            pytest.param("null", id="json-null"),
        ],
    )
    def test_a_settings_blob_the_user_never_typed_does_not_cost_them_the_upload(
        self, client, db, upload_dir, no_embedding_calls, raw
    ):
        """Rejecting a good batch of documents because a settings blob the user never typed
        arrived malformed is the worse failure — the documents are fine and the defaults are
        a perfectly reasonable way to index them."""
        token = _seed_ai_user(db)
        response = _post_upload(client, token, document_configs=raw)

        assert response.status_code == 202
        file_record = _files_in_upload_order(db, response.json()["data"]["id"])[0]
        assert (file_record.chunk_strategy, file_record.chunk_size, file_record.chunk_overlap) == (
            "character",
            1200,
            180,
        )

    def test_a_short_settings_list_leaves_the_remaining_documents_on_the_defaults(
        self, client, db, upload_dir, no_embedding_calls
    ):
        # Positional pairing means a client that appends a file without appending a config
        # is the ordinary case, not an error. The unconfigured document still has to index.
        token = _seed_ai_user(db)
        response = _post_upload(
            client,
            token,
            files=[
                ("files", ("a.txt", io.BytesIO(_text()), "text/plain")),
                ("files", ("b.txt", io.BytesIO(_text()), "text/plain")),
            ],
            document_configs=json.dumps([{"strategy": "sentence"}]),
        )
        assert response.status_code == 202

        first, second = _files_in_upload_order(db, response.json()["data"]["id"])
        assert first.chunk_strategy == "sentence"
        assert second.chunk_strategy == "character"

    def test_an_unknown_strategy_falls_back_rather_than_failing_the_batch(
        self, client, db, upload_dir, no_embedding_calls
    ):
        # A frontend that ships a new strategy id before the backend does must degrade to
        # the default, not 400. `normalize_rag_mode` already forgives the same mistake.
        token = _seed_ai_user(db)
        response = _post_upload(
            client, token, document_configs=json.dumps([{"strategy": "semantic-magic"}])
        )

        assert response.status_code == 202
        assert _files_in_upload_order(db, response.json()["data"]["id"])[0].chunk_strategy == "character"


class TestOutOfRangeValuesAreClamped:
    @pytest.mark.parametrize(
        "sent,expected_size,expected_overlap",
        [
            pytest.param(999999, chunking.MAX_CHUNK_SIZE, 180, id="above-max"),
            pytest.param(1, chunking.MIN_CHUNK_SIZE, 100, id="below-min"),
            pytest.param("not a number", chunking.DEFAULT_CHUNK_SIZE, 180, id="not-a-number"),
        ],
    )
    def test_a_chunk_size_outside_the_bounds_lands_on_the_bound(
        self, client, db, upload_dir, no_embedding_calls, sent, expected_size, expected_overlap
    ):
        # Clamping rather than rejecting: a slider that arrived one past its own maximum
        # must not fail an upload of documents that are perfectly fine. The stored value has
        # to be inside the bounds the catalogue advertises, or the chunker gets input it was
        # never written for.
        token = _seed_ai_user(db)
        response = _post_upload(
            client, token, document_configs=json.dumps([{"chunk_size": sent, "overlap": 180}])
        )

        assert response.status_code == 202
        file_record = _files_in_upload_order(db, response.json()["data"]["id"])[0]
        assert file_record.chunk_size == expected_size
        # Overlap is bounded by the *clamped* size, not the requested one — at the 200
        # minimum, 180 would be 90% overlap and the window would advance slower than it
        # grows, which is how one document becomes an unbounded number of chunks.
        assert file_record.chunk_overlap == expected_overlap

    def test_an_overlap_at_or_beyond_half_the_chunk_size_is_pulled_back(
        self, client, db, upload_dir, no_embedding_calls
    ):
        token = _seed_ai_user(db)
        response = _post_upload(
            client, token, document_configs=json.dumps([{"chunk_size": 1000, "overlap": 5000}])
        )

        file_record = _files_in_upload_order(db, response.json()["data"]["id"])[0]
        assert file_record.chunk_overlap == int(1000 * chunking.MAX_OVERLAP_RATIO)
        assert file_record.chunk_overlap < file_record.chunk_size


class TestResourceDocuments:
    def test_lists_each_document_with_its_settings_and_chunk_count(
        self, client, db, upload_dir, no_embedding_calls
    ):
        # This backs the screen that answers "why did this document not answer my question".
        # Without the per-document config and count it can only say the base exists.
        token = _seed_ai_user(db)
        resource_id = _post_upload(
            client,
            token,
            files=[
                ("files", ("policy.txt", io.BytesIO(_text("clause obligation")), "text/plain")),
                ("files", ("faq.txt", io.BytesIO(_text("brake fluid")), "text/plain")),
            ],
            document_configs=json.dumps([{"strategy": "sentence", "chunk_size": 600}, {}]),
        ).json()["data"]["id"]
        jobs.run_job(db, jobs.claim_next_job(db))

        response = client.get(
            f"/resources/{resource_id}/documents", headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code == 200
        body = response.json()
        assert set(body) == {"status", "status_code", "message", "data"}

        records = body["data"]["records"]
        assert body["data"]["total"] == 2
        assert [r["file_name"] for r in records] == ["policy.txt", "faq.txt"]
        assert records[0]["config"]["strategy"] == "sentence"
        assert records[0]["config"]["chunk_size"] == 600
        assert records[1]["config"] == chunking.DEFAULT_CONFIG.to_dict()
        assert all(r["chunk_count"] > 0 for r in records)
        # Nothing was embedded, and the count is what lets the UI say "keyword only" from
        # data rather than by matching strings in a detail message.
        assert all(r["embedded_chunks"] == 0 for r in records)

    def test_a_base_whose_indexing_has_not_run_lists_its_documents_with_no_chunks(
        self, client, db, upload_dir, no_embedding_calls
    ):
        # Queued is not the same as empty. Hiding the documents until indexing finishes
        # reads as an upload that vanished.
        token = _seed_ai_user(db)
        resource_id = _post_upload(client, token).json()["data"]["id"]

        records = client.get(
            f"/resources/{resource_id}/documents", headers={"Authorization": f"Bearer {token}"}
        ).json()["data"]["records"]
        assert [r["file_name"] for r in records] == ["manual.txt"]
        assert records[0]["chunk_count"] == 0

    def test_another_users_knowledge_base_is_reported_as_missing(
        self, client, db, upload_dir, no_embedding_calls
    ):
        # 404, not 403 — matching how an unknown ingestion job is already handled. A 403
        # confirms the id exists and that its owner uploaded something to it.
        owner_token = _seed_ai_user(db, "owner-user")
        stranger_token = _seed_ai_user(db, "stranger-user")
        resource_id = _post_upload(client, owner_token).json()["data"]["id"]

        response = client.get(
            f"/resources/{resource_id}/documents",
            headers={"Authorization": f"Bearer {stranger_token}"},
        )
        assert response.status_code == 404
        assert response.json()["status"] == "error"

    def test_an_unknown_knowledge_base_is_a_404(self, client, db, upload_dir):
        token = _seed_ai_user(db)
        response = client.get(
            "/resources/does-not-exist/documents", headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code == 404

    def test_requires_authentication(self, client, db, upload_dir):
        assert client.get("/resources/anything/documents").status_code == 401


class TestEmbeddingsAreOptIn:
    def test_an_upload_with_no_embedding_settings_never_reaches_a_provider(
        self, client, db, upload_dir, no_embedding_calls
    ):
        """The guarantee that matters most in this module: no upload silently ships document
        text to a third party or spends the user's money. `no_embedding_calls` fails the test
        from inside ingestion if the embedding path is entered at all."""
        token = _seed_ai_user(db)
        resource_id = _post_upload(client, token).json()["data"]["id"]

        job = jobs.claim_next_job(db)
        jobs.run_job(db, job)

        assert no_embedding_calls == []
        assert db.query(DocumentChunk).filter(DocumentChunk.resource_id == resource_id).count() > 0
        # Not merely "no call was made" — no vector was written either, so retrieval cannot
        # later believe this base has embeddings to compare against.
        assert (
            db.query(DocumentChunk)
            .filter(
                DocumentChunk.resource_id == resource_id,
                DocumentChunk.embedding_json.isnot(None),
            )
            .count()
            == 0
        )

    def test_half_an_embedding_configuration_is_not_a_configuration(
        self, client, db, upload_dir, no_embedding_calls
    ):
        # A provider with no model (a half-filled form) must not be stored as an intent to
        # embed — `embeds` would be false but the row would claim otherwise, and the
        # documents screen would report a state that never existed.
        token = _seed_ai_user(db)
        resource_id = _post_upload(
            client, token, document_configs=json.dumps([{"embedding_provider": "ollama"}])
        ).json()["data"]["id"]

        jobs.run_job(db, jobs.claim_next_job(db))

        file_record = _files_in_upload_order(db, resource_id)[0]
        assert file_record.embedding_provider is None
        assert file_record.embedding_model is None
        assert no_embedding_calls == []

    def test_asking_for_embeddings_does_reach_the_provider(
        self, client, db, upload_dir, monkeypatch
    ):
        """The control for the two tests above. If the embedding seam were dead code — or if
        `rag.service.embed_texts` were the wrong name to patch — they would pass while
        proving nothing."""
        calls: list[int] = []

        # `task` is accepted, not swallowed by **kwargs: a double that quietly absorbs a new
        # argument is how a call site stops being exercised with nothing going red. Note
        # this control test caught exactly that — run_job's per-document `except Exception`
        # turned the TypeError into a failed document, so `calls` stayed empty.
        def _fake_embed(db_session, user_id, provider, model_id, texts, task=EMBED_TASK_DOCUMENT):
            calls.append(len(texts))
            assert task == EMBED_TASK_DOCUMENT, "ingestion embeds documents, not queries"
            return [[0.1, 0.2, 0.3] for _ in texts]

        monkeypatch.setattr(rag_service, "embed_texts", _fake_embed)

        token = _seed_ai_user(db)
        resource_id = _post_upload(
            client,
            token,
            document_configs=json.dumps(
                [{"embedding_provider": "ollama", "embedding_model": "nomic-embed-text"}]
            ),
        ).json()["data"]["id"]
        jobs.run_job(db, jobs.claim_next_job(db))

        assert sum(calls) > 0
        embedded = (
            db.query(DocumentChunk)
            .filter(
                DocumentChunk.resource_id == resource_id,
                DocumentChunk.embedding_json.isnot(None),
            )
            .count()
        )
        assert embedded == sum(calls)

        records = client.get(
            f"/resources/{resource_id}/documents", headers={"Authorization": f"Bearer {token}"}
        ).json()["data"]["records"]
        assert records[0]["embedded_chunks"] == embedded
        assert records[0]["config"]["embedding_provider"] == "ollama"


class TestStoredSettingsStayInsideTheColumn:
    def test_a_stored_strategy_fits_the_column_it_is_written_to(
        self, client, db, upload_dir, no_embedding_calls
    ):
        """SQLite ignores VARCHAR limits, so a value that would be MySQL error 1406 in
        production passes here unless the length is asserted directly. `files.chunk_strategy`
        is VARCHAR(30) and the value comes from client input."""
        token = _seed_ai_user(db)
        resource_id = _post_upload(
            client,
            token,
            document_configs=json.dumps([{"strategy": "x" * 500}]),
        ).json()["data"]["id"]

        file_record = _files_in_upload_order(db, resource_id)[0]
        assert len(file_record.chunk_strategy) <= 30
        assert file_record.chunk_strategy in chunking.STRATEGIES

    def test_stored_embedding_settings_fit_the_columns_they_are_written_to(
        self, client, db, upload_dir, no_embedding_calls
    ):
        """`files.embedding_provider` is VARCHAR(50) and `files.embedding_model` VARCHAR(255);
        both arrive verbatim from client JSON. On MySQL an over-length value is error 1406 —
        an uncaught DataError and a 500 on an upload whose documents were perfectly fine.
        This passes on SQLite for the wrong reason, which is why the lengths are asserted
        directly rather than left to the insert."""
        token = _seed_ai_user(db)
        resource_id = _post_upload(
            client,
            token,
            document_configs=json.dumps(
                [{"embedding_provider": "p" * 400, "embedding_model": "m" * 4000}]
            ),
        ).json()["data"]["id"]

        file_record = _files_in_upload_order(db, resource_id)[0]
        assert len(file_record.embedding_provider or "") <= 50
        assert len(file_record.embedding_model or "") <= 255
