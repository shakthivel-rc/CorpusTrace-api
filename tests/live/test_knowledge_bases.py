"""The knowledge-base, ingestion and RAG surface, exercised against the running stack.

Everything under `/resources` is covered hermetically already, and covered well — but that
coverage runs against in-memory SQLite, and three things this half of the API depends on
simply do not exist there:

  * **Foreign keys.** SQLite does not enforce them unless asked; MySQL always does. The
    purge path in `rag/service.py` deletes a resource's children and then the resource
    itself, and `ingestion_jobs.resource_id` points at that row — see `TestPermanentDelete`
    below, which is a real defect that only a real database can show you.
  * **The background worker.** `tests/integration/test_ingestion_endpoints.py` drives it by
    calling `jobs.run_job(db, jobs.claim_next_job(db))` inline, because a TestClient never
    fires the startup event that owns the poll loop. Nothing hermetic proves that the loop
    is actually running in a deployed process, that it claims work, or that a browser
    polling `GET /resources/jobs/{id}` ever sees the job leave `queued`.
  * **nginx.** Upload is `multipart/form-data` through a proxy, and the answer streams back
    with its citations on a response header. Both are exactly the kind of thing a proxy
    silently mangles while every unit test stays green.

So these tests upload a real document over HTTP, wait for a real worker to index it, ask a
real question of it and follow the citation back to its passage.

Re-runnability: the live database persists, and other agents write to it concurrently. Every
resource created here is namespaced and soft-deleted in a `finally`, and no assertion depends
on a global count — only on rows this module created, or on contracts that hold whatever else
is in the table.
"""
import base64
import io
import json
import time
import uuid

import httpx
import pytest

from conftest import assert_envelope


# A few sentences, deliberately. The point of this suite is that indexing really ran, not
# that it can chew through a large PDF, and a tiny document keeps the poll below a second.
DOCUMENT_TEXT = (
    "The valve clearance on the Daytona 675 is checked every 12000 miles. "
    "Torque the cylinder head bolts to 25 Nm. "
    "The coolant is a 50/50 mix of distilled water and antifreeze. "
) * 3

# Generous, because this waits on a real worker that is also serving other agents' uploads.
# Long enough that a busy stack does not produce a flake; short enough that a worker which
# is genuinely not running fails the suite rather than hanging it.
INDEXING_TIMEOUT_SECONDS = 90
POLL_INTERVAL_SECONDS = 0.4

# The four chunking strategies in rag/chunking.py. Named here rather than imported so this
# asserts the *wire* catalogue: the frontend renders whatever this endpoint sends, and an
# import would make the test agree with a rename that broke every saved document config.
EXPECTED_STRATEGIES = {"character", "sentence", "paragraph", "page"}

# The only providers with an embeddings endpoint this code was written against. Listing one
# without it would be a dropdown that 404s, which is why the set is closed.
EMBEDDING_PROVIDERS = {"openai", "gemini", "ollama", "mistral"}

TERMINAL_JOB_STATUSES = {"succeeded", "failed", "cancelled"}
ACTIVE_JOB_STATUSES = {"queued", "running"}


def _upload(http: httpx.Client, name: str, filename: str = "manual.txt") -> httpx.Response:
    """POST one small text document as a new knowledge base."""
    return http.post(
        "/resources/upload_files",
        data={"resource_name": name},
        files=[("files", (filename, io.BytesIO(DOCUMENT_TEXT.encode()), "text/plain"))],
    )


def _wait_for_job(http: httpx.Client, job_id: str) -> dict:
    """Poll until the job reaches a terminal state, exactly as the SPA's poller does.

    Chains one request off the previous response rather than firing on an interval, for the
    same reason `useIngestionJobs` does: overlapping polls can land out of order and show
    progress going backwards.
    """
    deadline = time.monotonic() + INDEXING_TIMEOUT_SECONDS
    payload: dict = {}
    while time.monotonic() < deadline:
        payload = assert_envelope(http.get(f"/resources/jobs/{job_id}"))
        if payload["status"] in TERMINAL_JOB_STATUSES:
            return payload
        time.sleep(POLL_INTERVAL_SECONDS)
    pytest.fail(
        f"job {job_id} never left {payload.get('status')!r} within "
        f"{INDEXING_TIMEOUT_SECONDS}s — is the background indexing worker running?"
    )


@pytest.fixture(scope="module")
def indexed_base(api_base: str, admin_token: str):
    """One knowledge base, uploaded and indexed for real, shared by the tests below.

    Module-scoped on purpose: uploading and indexing is the slow, side-effecting part, and
    repeating it per test would multiply both the wall clock and the rows left in a database
    other agents are using. It cannot take the `unique` fixture (function-scoped), so it
    namespaces itself the same way.

    Yields the raw upload response *and* the final job payload rather than asserting on
    them, so each contract below is stated by a named test instead of dying inside a fixture
    where the failure would read as "setup error".
    """
    suffix = uuid.uuid4().hex[:10]
    with httpx.Client(
        base_url=api_base,
        timeout=90,
        follow_redirects=False,
        headers={"Authorization": f"Bearer {admin_token}"},
    ) as http:
        accepted = _upload(http, f"live-rag-{suffix}")
        assert accepted.status_code == 202, accepted.text[:400]
        created = accepted.json()["data"]
        resource_id = created["id"]
        try:
            final_job = _wait_for_job(http, created["job"]["id"])
            yield {
                "http": http,
                "suffix": suffix,
                "resource_id": resource_id,
                "job_id": created["job"]["id"],
                "accepted": accepted,
                "upload_payload": created,
                "job": final_job,
            }
        finally:
            # Soft delete, not purge: purge is currently a 500 on MySQL (see
            # TestPermanentDelete). A soft delete removes the base from every picker and
            # from retrieval, which is what "cleaned up" has to mean until that is fixed.
            http.delete(f"/resources/brain/{resource_id}")


class TestKnowledgeBaseList:
    def test_brain_answers_with_the_envelope_and_a_records_list(self, admin):
        """`GET /resources/brain` is the first authenticated call the chat screen makes.

        It is also the one endpoint here that every other feature depends on: the chat
        picker, the Knowledge Bases screen and the upload form all read it. Assert the
        envelope and the record shape rather than the contents — the account is shared, so
        what is in the list is whatever the stack happens to hold.
        """
        data = assert_envelope(admin.get("/resources/brain"))
        assert isinstance(data["records"], list)
        assert data["total"] == len(data["records"])
        for record in data["records"]:
            # `job` is nullable and null means "nothing in flight", never "unknown" — a
            # client that treated a missing key as unknown would block bases that predate
            # background indexing and are perfectly answerable.
            for key in ("id", "resource_name", "upload_status", "file_count", "chunk_count", "job"):
                assert key in record, f"brain record missing {key!r}: {record}"

    def test_deleted_flag_selects_the_other_list(self, admin):
        """`?deleted=true` is a separate list, not a filter layered onto the default one.

        The two must not overlap: a base in both would appear in the chat picker while the
        management screen offers to restore it.
        """
        active = assert_envelope(admin.get("/resources/brain"))["records"]
        removed = assert_envelope(admin.get("/resources/brain?deleted=true"))["records"]
        assert not ({r["id"] for r in active} & {r["id"] for r in removed})
        assert all(r["deleted_at"] is None for r in active)
        assert all(r["deleted_at"] for r in removed)

    def test_brain_requires_a_token(self, client):
        """`ai_access` gates every route in this module, and the middleware runs first.

        Worth pinning live because CORS is the innermost middleware here (CLAUDE.md §13), so
        a 401 from `JWTAuthMiddleware` returns without CORS headers — and through nginx that
        is the one response most likely to arrive as something other than a clean 401.
        """
        assert client.get("/resources/brain").status_code == 401


class TestIndexingOptionsCatalogue:
    """`GET /resources/indexing-options` is served rather than duplicated in TypeScript.

    That decision is only worth anything if the payload really does carry everything the
    upload form renders. These tests are the check that it still does — if a strategy is
    added to `rag/chunking.py` and this endpoint stops matching, the form silently offers a
    stale menu and nothing else in either project notices.
    """

    def test_offers_the_four_chunking_strategies(self, admin):
        data = assert_envelope(admin.get("/resources/indexing-options"))
        strategies = {s["id"]: s for s in data["strategies"]}
        assert set(strategies) == EXPECTED_STRATEGIES

        # `character` is the default and the safe answer, and it is the only one flagged —
        # two recommendations is a form with no recommendation.
        recommended = [s["id"] for s in data["strategies"] if s["recommended"]]
        assert recommended == ["character"]

        for strategy in data["strategies"]:
            # The caveat is not decoration: "page" on a .txt degrades to Fixed size, and the
            # user has to be able to read that *before* uploading rather than in the report.
            for key in ("label", "summary", "best_for", "caveat"):
                assert strategy.get(key), f"{strategy['id']} has no {key}"

    def test_publishes_the_bounds_the_chunker_actually_clamps_to(self, admin):
        """`normalize_config` clamps, it never rejects — so these bounds are the only
        warning a user gets that a value they typed will be quietly moved."""
        data = assert_envelope(admin.get("/resources/indexing-options"))
        bounds = data["bounds"]
        assert bounds["min_chunk_size"] < bounds["max_chunk_size"]
        assert bounds["min_overlap"] == 0
        # At or beyond half the chunk size the window advances slower than it grows, which
        # turns a large document into an unbounded number of chunks.
        assert 0 < bounds["max_overlap_ratio"] <= 0.5

        for preset in data["chunk_sizes"]:
            assert bounds["min_chunk_size"] <= preset["value"] <= bounds["max_chunk_size"]
            assert preset["label"] and preset["hint"]
        for preset in data["overlaps"]:
            assert preset["value"] >= bounds["min_overlap"]
            assert preset["label"] and preset["hint"]

    def test_defaults_are_the_historical_chunker_with_embeddings_off(self, admin):
        """The default path has to stay byte-identical to the pre-2026-08-05 chunker, and
        embeddings have to stay off.

        Both halves matter. A changed default would silently re-slice nothing already
        indexed but would make every *new* base behave differently from every old one; a
        default embedding model would send documents to a third party and, on three of the
        four providers, spend money because a form was skimmed.
        """
        defaults = assert_envelope(admin.get("/resources/indexing-options"))["defaults"]
        assert defaults["strategy"] == "character"
        assert defaults["chunk_size"] == 1200
        assert defaults["overlap"] == 180
        assert defaults["embedding_provider"] is None
        assert defaults["embedding_model"] is None

    def test_recommends_a_configuration_for_every_rag_mode(self, admin):
        """Every mode the chat screen offers has to have advice, or the form shows a blank
        panel for whichever one nobody remembered.

        The recommendation is also asserted to name a strategy that exists and a size inside
        the bounds — a recommendation the chunker would clamp away is worse than none.
        """
        data = assert_envelope(admin.get("/resources/indexing-options"))
        bounds, strategies = data["bounds"], {s["id"] for s in data["strategies"]}
        recommendations = data["recommendations"]

        # The default mode and the only mode that refuses to answer both have to be here;
        # asserting the full set would make this test a copy of SUPPORTED_RAG_MODES.
        assert {"contextual_hybrid", "corrective"} <= set(recommendations)
        for mode, rec in recommendations.items():
            assert rec["strategy"] in strategies, f"{mode} recommends an unknown strategy"
            assert bounds["min_chunk_size"] <= rec["chunk_size"] <= bounds["max_chunk_size"]
            assert 0 <= rec["overlap"] <= rec["chunk_size"] * bounds["max_overlap_ratio"]
            # "why" is derived from what the retrieval code mechanically does. An empty one
            # means the mode was added to the picker and the reasoning was never written.
            assert rec["label"] and rec["why"]

    def test_only_providers_with_an_embeddings_endpoint_are_offered(self, admin):
        """Seven of the eleven chat providers serve completions only.

        Listing one of those here would be a dropdown entry that 404s at upload time — after
        the user chose it, and after the document was already on disk.
        """
        providers = assert_envelope(admin.get("/resources/indexing-options"))["embedding_providers"]
        assert {p["provider"] for p in providers} <= EMBEDDING_PROVIDERS
        for provider in providers:
            assert provider["models"], f"{provider['provider']} offers no embedding model"
            for model in provider["models"]:
                # Dimensions are what make "vectors are only ever compared within one model"
                # legible to a user choosing between two of them.
                assert model["id"] and isinstance(model["dimensions"], int)


class TestIngestionJobsList:
    def test_jobs_answers_with_the_envelope_and_a_records_list(self, admin):
        """The list behind the Background Tasks screen, which replaced a hard-coded array of
        fictional jobs. Its records must at least carry what that screen renders."""
        data = assert_envelope(admin.get("/resources/jobs"))
        assert isinstance(data["records"], list)
        assert isinstance(data["total"], int)
        for record in data["records"]:
            for key in ("id", "status", "stage", "progress", "total_documents", "documents"):
                assert key in record, f"job record missing {key!r}"
            assert 0 <= record["progress"] <= 100

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "BUG: /resources/jobs reports `total` as the length of the page, not a count of "
            "the matching rows. controllers/chat_controller.py:577 builds "
            "{'records': records, 'total': len(records)} over a query rag/jobs.py:484 caps at "
            "100, so a caller asking for one row is told there is one job. GET /activity-logs "
            "does this correctly with query.count() — 639 against a page of 1 on this stack — "
            "which is what makes this a defect rather than a house style. The Background Tasks "
            "screen cannot paginate or show a job count while it holds."
        ),
    )
    def test_jobs_total_counts_matching_rows_not_the_page(self, admin):
        """`total` must survive being asked for a smaller page.

        Asserted as a relationship between two responses rather than a fixed number, because
        other agents upload against this stack and the job count moves under the test.
        """
        everything = assert_envelope(admin.get("/resources/jobs"))
        if len(everything["records"]) < 2:
            pytest.skip("needs at least two ingestion jobs on the stack to be meaningful")

        page = assert_envelope(admin.get("/resources/jobs?limit=1"))

        assert len(page["records"]) == 1
        assert page["total"] > 1, (
            f"total collapsed to the page size: {page['total']} with "
            f"{len(everything['records'])} jobs visible unpaginated"
        )

    def test_active_filter_returns_only_unfinished_jobs(self, admin):
        """`active=true` is what stops the SPA polling forever.

        Stated as a property of every returned row rather than as a list of ids, because
        other agents are uploading against this same stack and their jobs legitimately show
        up here.
        """
        data = assert_envelope(admin.get("/resources/jobs?active=true"))
        assert all(r["status"] in ACTIVE_JOB_STATUSES for r in data["records"]), [
            r["status"] for r in data["records"]
        ]

    def test_a_cancel_requested_job_still_counts_as_active(self, admin):
        """Cancellation is cooperative: a running job with `cancel_requested` set is
        *stopping*, not stopped, and the poller has to keep watching it until the worker
        reaches its next checkpoint. Any row the filter returns must therefore still be
        queued or running regardless of its cancel flag."""
        data = assert_envelope(admin.get("/resources/jobs?active=true"))
        for record in data["records"]:
            assert isinstance(record["cancel_requested"], bool)
            assert record["status"] in ACTIVE_JOB_STATUSES

    def test_limit_is_bounded_and_rejects_out_of_range_rather_than_500(self, admin):
        """`limit` is declared `ge=1, le=100`. There is no global exception handler, so an
        unvalidated bound would reach SQLAlchemy and come back as a bare 500."""
        assert admin.get("/resources/jobs?limit=0").status_code == 422
        assert admin.get("/resources/jobs?limit=101").status_code == 422
        capped = assert_envelope(admin.get("/resources/jobs?limit=1"))
        assert len(capped["records"]) <= 1

    def test_an_unknown_job_is_a_404(self, admin):
        """Deliberately 404, not 403 — and the same code another user's job returns.

        A 403 would confirm that the id exists and that its owner uploaded something, which
        is exactly the disclosure the 404 exists to avoid. This is also the shape that
        proves an unparseable id is handled rather than exploding in a UUID cast.
        """
        response = admin.get(f"/resources/jobs/not-a-real-job-{uuid.uuid4().hex}")
        assert_envelope(response, expected_status=404)

    def test_cancelling_an_unknown_job_is_a_404(self, admin):
        response = admin.post(f"/resources/jobs/not-a-real-job-{uuid.uuid4().hex}/cancel")
        assert_envelope(response, expected_status=404)


class TestUploadAndIndex:
    """The upload lifecycle end to end: accepted, queued, indexed by the real worker."""

    def test_upload_is_accepted_with_202_and_a_queued_job(self, indexed_base):
        """202, not 201.

        A 201 says the base exists and is usable. It is not: nothing has been parsed,
        chunked or indexed when this returns, and asking it a question at this moment gets
        an explanation rather than an answer. The whole background-indexing design is
        readable from this one status code, which is why it is pinned rather than accepted
        as an implementation detail.
        """
        accepted, payload = indexed_base["accepted"], indexed_base["upload_payload"]
        assert accepted.status_code == 202
        assert payload["upload_status"] is False
        job = payload["job"]
        assert job["status"] == "queued"
        assert job["progress"] == 0
        assert job["total_documents"] == 1
        # The client needs something to name in the progress list before anything has run.
        assert [d["file_name"] for d in job["documents"]] == ["manual.txt"]

    def test_upload_returns_fast_enough_to_be_a_request_and_not_a_wait(self, indexed_base):
        """The point of the 202 is that the request no longer holds the browser open through
        parsing. Bounded loosely — this is a shared stack — but a regression that moved
        indexing back into the request would blow well past it on any real document."""
        assert indexed_base["accepted"].elapsed.total_seconds() < 10

    def test_the_background_worker_indexes_the_document(self, indexed_base):
        """The claim this whole suite exists for: a worker is genuinely running in the
        deployed process, it claimed the job, and it finished it.

        Hermetically this step is a direct call to `jobs.run_job`. Nothing but a live stack
        can tell you the poll loop was ever started.
        """
        job = indexed_base["job"]
        assert job["status"] == "succeeded", job.get("message")
        assert job["progress"] == 100
        assert job["processed_documents"] == job["total_documents"] == 1
        assert job["indexed_chunks"] >= 1
        assert job["started_at"] and job["finished_at"]
        assert job["current_document"] is None

    def test_the_job_reports_per_document_detail(self, indexed_base):
        """The per-document line is the most valuable thing the report says.

        It is how a user learns that a scanned PDF yielded no text at all, and it names the
        strategy that *actually* ran rather than the one that was requested. A job that
        succeeded but reported nothing per document would look identical to one that worked.
        """
        documents = indexed_base["job"]["documents"]
        assert len(documents) == 1
        document = documents[0]
        assert document["status"] == "succeeded"
        assert document["chunk_count"] >= 1
        assert document["detail"], "a document that indexed must say what happened to it"
        # Embeddings are opt-in and this upload did not ask for any, so zero here is the
        # data the UI reads to say "searchable by keyword only".
        assert document["embedded_chunks"] == 0
        assert document["config"]["embedding_provider"] is None
        assert document["config"]["embedding_model"] is None

    def test_the_indexed_base_appears_in_the_brain_list_and_is_answerable(self, indexed_base):
        """Indexing succeeding is not the same as the base being usable.

        `upload_status` is set at the very end of `run_job`, after graph construction, and
        `_not_ready_note` reads it to decide whether chat may search the base at all. A base
        whose job says "succeeded" while its resource still says False would be listed and
        disabled forever.
        """
        records = assert_envelope(indexed_base["http"].get("/resources/brain"))["records"]
        mine = next((r for r in records if r["id"] == indexed_base["resource_id"]), None)
        assert mine is not None, "an indexed base did not appear in the brain list"
        assert mine["upload_status"] is True
        assert mine["file_count"] == 1
        assert mine["chunk_count"] >= 1
        assert mine["job"]["status"] == "succeeded"
        # The list payload stays small — per-document detail lives on the job endpoint only.
        assert "documents" not in mine["job"]

    def test_the_documents_endpoint_reports_the_settings_it_was_indexed_under(self, indexed_base):
        """Chunking settings are per document, so "how was this indexed" is a question only
        this endpoint can answer. An upload that sent no `document_configs` must read back as
        the long-standing defaults, because that is genuinely what it got."""
        data = assert_envelope(
            indexed_base["http"].get(f"/resources/{indexed_base['resource_id']}/documents")
        )
        assert data["total"] == 1
        document = data["records"][0]
        assert document["file_name"] == "manual.txt"
        assert document["chunk_count"] >= 1
        assert document["embedded_chunks"] == 0
        assert document["config"] == {
            "strategy": "character",
            "chunk_size": 1200,
            "overlap": 180,
            "embedding_provider": None,
            "embedding_model": None,
        }

    def test_another_users_resource_documents_are_a_404(self, indexed_base):
        """Same disclosure rule as the job endpoint: not found, never forbidden."""
        response = indexed_base["http"].get(f"/resources/{uuid.uuid4()}/documents")
        assert_envelope(response, expected_status=404)

    def test_the_finished_job_is_no_longer_active(self, indexed_base):
        """The poller stops when nothing is active. A terminal job still reported as active
        would keep every open tab polling this endpoint forever."""
        active = assert_envelope(indexed_base["http"].get("/resources/jobs?active=true"))
        assert indexed_base["job_id"] not in {r["id"] for r in active["records"]}

    def test_the_finished_job_is_still_listed(self, indexed_base):
        """It leaves the active filter but not the history — the Background Tasks screen is
        the record that the upload happened and what came of it."""
        listed = assert_envelope(indexed_base["http"].get("/resources/jobs?limit=100"))
        assert indexed_base["job_id"] in {r["id"] for r in listed["records"]}

    def test_cancelling_a_finished_job_is_a_conflict_not_a_silent_success(self, indexed_base):
        """409, because there is nothing left to cancel and the resource must survive.

        A 200 here would be actively dangerous: cancelling discards the whole knowledge base
        including the documents that already indexed, so "cancel" arriving late must refuse
        rather than delete a base the user is using.
        """
        response = indexed_base["http"].post(f"/resources/jobs/{indexed_base['job_id']}/cancel")
        assert_envelope(response, expected_status=409)

        # And the base is still there afterwards.
        records = assert_envelope(indexed_base["http"].get("/resources/brain"))["records"]
        assert indexed_base["resource_id"] in {r["id"] for r in records}


class TestSourcePassages:
    def test_an_unknown_chunk_is_a_404_not_a_500(self, admin):
        """The evidence panel fetches this by id the moment a citation chip is clicked.

        An id that no longer resolves — a purged base, a stale transcript — has to be a
        clean 404. With no global exception handler an unhandled lookup would be a bare
        plain-text 500, which the SPA's envelope-shaped error handling cannot even read.
        """
        response = admin.get(f"/resources/chunks/not-a-real-chunk-{uuid.uuid4().hex}")
        assert_envelope(response, expected_status=404)

    def test_a_citation_from_a_real_answer_resolves_back_to_its_passage(self, indexed_base):
        """The full round trip: ask, get citations off the response header, open one.

        Citations ride on `X-CorpusTrace-Citations` rather than in the body precisely so the
        streamed body stays verbatim answer text — which makes them the single most
        proxy-fragile part of the API. nginx re-writing or dropping a response header is
        invisible to every hermetic test and breaks the entire evidence view.
        """
        http = indexed_base["http"]
        response = http.post(
            "/chat/asks",
            params={
                "query": "What is the valve clearance on the Daytona checked at?",
                "chat_history_name": f"live-rag-{indexed_base['suffix']}",
                "brain_id": indexed_base["resource_id"],
                "rag_type": "contextual_hybrid",
            },
        )
        assert response.status_code == 200, response.text[:400]

        raw = response.headers.get("X-CorpusTrace-Citations")
        assert raw, "an answer grounded in the uploaded document carried no citations header"
        citations = json.loads(base64.urlsafe_b64decode(raw + "=" * (-len(raw) % 4)))
        assert citations, "the citations header decoded to nothing"

        # The chip number and the [N] marker in the prose come from the same slice, so they
        # can never disagree — 1-based, contiguous, capped at five.
        assert [c["index"] for c in citations] == list(range(1, len(citations) + 1))
        assert len(citations) <= 5

        first = citations[0]
        assert first["source_name"] == "manual.txt"
        # The header carries no document text: the panel fetches the passage separately, so
        # a long quote can never push the header past a proxy's limit.
        assert "content" not in first and "snippet" not in first

        chunk = assert_envelope(http.get(f"/resources/chunks/{first['chunk_id']}"))
        assert chunk["chunk_id"] == first["chunk_id"]
        assert chunk["resource_id"] == indexed_base["resource_id"]
        assert "valve clearance" in chunk["content"]
        # A .txt has no pages, and NULL means unknown — never page 1. The evidence view says
        # so rather than sending the reader confidently to the wrong place.
        assert chunk["page_start"] is None and chunk["page_end"] is None

    def test_the_uploaded_document_is_served_back_with_a_type_from_its_extension(
        self, indexed_base
    ):
        """The only endpoint that hands user-uploaded bytes back to a browser.

        `Content-Type` is derived from the file extension against an allow-list, never from
        `files.file_type` — that column is whatever the client put in its own multipart
        header, and serving an attacker-chosen `text/html` inline from the API origin is
        stored XSS. Asserted live because a proxy is perfectly capable of rewriting this.
        """
        http = indexed_base["http"]
        documents = assert_envelope(
            http.get(f"/resources/{indexed_base['resource_id']}/documents")
        )["records"]
        response = http.get(f"/resources/files/{documents[0]['id']}")
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/plain")
        assert response.headers["x-content-type-options"] == "nosniff"
        assert b"valve clearance" in response.content

    def test_an_unknown_source_file_is_a_404(self, admin):
        response = admin.get(f"/resources/files/not-a-real-file-{uuid.uuid4().hex}")
        assert_envelope(response, expected_status=404)


class TestPermanentDelete:
    """Purging a knowledge base — currently broken on MySQL. See the xfail below."""

    @pytest.fixture()
    def disposable_base(self, admin, unique):
        """A base of its own, because this test may (once fixed) really delete it."""
        accepted = _upload(admin, f"live-purge-{unique}")
        assert accepted.status_code == 202
        created = accepted.json()["data"]
        try:
            _wait_for_job(admin, created["job"]["id"])
            yield created["id"]
        finally:
            # Soft delete is the fallback that always works, and is a no-op (404) if the
            # purge under test actually succeeded.
            admin.delete(f"/resources/brain/{created['id']}")

    @pytest.mark.xfail(
        strict=False,
        reason=(
            "PRODUCT BUG: DELETE /resources/brain/{id}/permanent is a 500 on MySQL for any "
            "base that has an ingestion job. purge_resource() deletes the graph rows, chunks "
            "and files but never releases ingestion_jobs.resource_id, so the final DELETE "
            "FROM resources violates FK ingestion_jobs_ibfk_1 (MySQL 1451). rag/jobs.py "
            "_cancel() nulls job.resource_id before purging for exactly this reason; the HTTP "
            "purge path does not. Invisible hermetically because SQLite does not enforce "
            "foreign keys."
        ),
    )
    def test_permanently_deleting_a_base_succeeds(self, admin, disposable_base):
        """Purge is the only way to reclaim the disk a knowledge base occupies.

        Every base created since background indexing shipped has an `ingestion_jobs` row, so
        this is not an edge case — it is the normal path, and it currently fails for every
        base a user could actually have. The transaction does roll back cleanly (nothing is
        half-deleted), so the damage is that the base can never be removed, not that it is
        left corrupt.
        """
        response = admin.delete(f"/resources/brain/{disposable_base}/permanent")
        assert response.status_code != 500, response.text[:400]
        assert_envelope(response, expected_status=200)

        records = assert_envelope(admin.get("/resources/brain"))["records"]
        removed = assert_envelope(admin.get("/resources/brain?deleted=true"))["records"]
        assert disposable_base not in {r["id"] for r in records + removed}

    def test_a_failed_purge_leaves_the_base_intact_rather_than_half_deleted(
        self, admin, disposable_base
    ):
        """Whatever purge does, it must not lose half a knowledge base.

        `_resource_action` rolls back on any exception, so the 500 above costs the user
        nothing but the ability to delete. This test states the invariant that actually
        matters and stays true either way — it passes today because of the rollback, and
        will keep passing once purge works, because a base that purged is simply gone.
        """
        admin.delete(f"/resources/brain/{disposable_base}/permanent")

        documents = admin.get(f"/resources/{disposable_base}/documents")
        if documents.status_code == 404:
            # The purge succeeded — nothing left to check.
            return
        data = assert_envelope(documents)
        assert data["total"] == 1
        assert data["records"][0]["chunk_count"] >= 1
