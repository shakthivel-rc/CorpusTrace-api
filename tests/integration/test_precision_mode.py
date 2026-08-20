"""The High-Precision Non-LLM RAG mode, end to end and — above all — in isolation.

The mode was added as a seventh `rag_type` on a chat endpoint six other modes already
answer through. Two failures are therefore worth more than any assertion about its
ranking quality, and both are silent:

  * a mode id that is not in `SUPPORTED_RAG_MODES` does not 400 — `normalize_rag_mode`
    forgives it into `contextual_hybrid`, the user gets a perfectly plausible answer from
    the wrong retriever, and `chat_history.brain` then records a mode that never ran;
  * `rag/precision/` shares this application's tokenizer, its per-chunk scoring memo and
    its dense candidates by injection, and it caches a derived index per resource. Any of
    those is a route by which running the new mode could change what an old one retrieves
    from the same knowledge base — a regression nobody would attribute to the feature,
    because the feature was not the mode they were using.

So `TestExistingModesAreUnaffected` is the centrepiece: it pins every citation of all six
pre-existing modes, runs High Precision against the same resource and the same query, and
demands the six come back byte-identical.

The remaining classes cover the surface the mode adds: its registration, its indexing
recommendation, the answer and citations it produces over `POST /chat/asks`, the mode name
persisted with the turn, and `POST /resources/{id}/precision-trace` — including the
log-safety property that module claims, that a trace carries ids, counts and scores and
never a line of somebody's document.
"""
import base64
import json
import re
import time

import jwt
import pytest

import rag.service as rag_service
from core.config import get_settings
from models.chat_history import ChatHistory
from models.file import File
from models.permissions import Permission
from models.rag import DocumentChunk, RagGraphEntity
from models.resource import Resource
from models.role_permissions import RolePermission
from models.roles import Role
from models.user import User
from models.user_roles import UserRole
from models.user_session import UserSession
from rag import chunking, precision
from utils.password import hash_password

pytestmark = pytest.mark.integration


# The six modes that existed before High Precision. Named here rather than derived from
# SUPPORTED_RAG_MODES so that adding an eighth mode cannot quietly shrink what the
# isolation test protects.
PRE_EXISTING_MODES = (
    "contextual_hybrid",
    "rag_fusion",
    "graph_rag",
    "corrective",
    "multi_modal",
    "agentic_rag",
)

# One query, asked of every mode. Every term appears in the corpus below, so no mode has
# to refuse — a refusal from both the "before" and "after" run would compare equal while
# testing nothing.
QUERY = "vacation days accrual"

# The only multi-word strings `pipeline.retrieve` writes into a trace that it did not read
# off the caller's question. Listed so the log-safety test below can tell "the engine
# explaining itself" apart from "the engine quoting somebody's document".
ENGINE_PHRASES = {
    "resource has no indexed chunks",
    "query has no searchable terms",
    "no chunk contained any query term",
}


# `tests/` is not a package, so this duplicates ingestion's bag-of-words rather than
# importing one. It must agree with `rag.service._tokenize`, because `terms_json` is what
# every mode scores against and a term map that disagrees with the tokenizer produces a
# corpus whose words the query can never match.
_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{1,}")


def _terms(text: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for token in _TOKEN_RE.findall(text):
        lowered = token.lower()
        if lowered in rag_service.STOPWORDS:
            continue
        counts[lowered] = counts.get(lowered, 0) + 1
    return counts


# Numbered, headed passages: `rag/precision/metadata.py` derives section, heading and
# version from exactly this shape, so a corpus of bare sentences would leave the metadata
# stage with nothing to do and quietly stop testing it.
PASSAGES = [
    ("Vacation policy", "3.1 Vacation Accrual Employees accrue twenty vacation days per calendar year, credited monthly."),
    ("Vacation carryover", "3.2 Vacation Carryover Unused vacation days carry over to the next year up to a maximum of five days."),
    ("Sick leave", "4.1 Sick Leave Employees receive ten sick days per year, separate from vacation days."),
    ("Expense claims", "5.1 Expense Claims Travel expenses must be submitted within thirty days with a receipt."),
    ("Remote work", "6.1 Remote Work Employees may work remotely three days per week with manager approval."),
    ("Parental leave", "7.1 Parental Leave Parental leave is sixteen weeks and does not reduce vacation days."),
]


@pytest.fixture(autouse=True)
def _cold_precision_index():
    """Start every test with no cached index, and leave none behind.

    `rag/precision/index.py` caches derived corpus data per resource id in a process-global
    OrderedDict that no table truncation touches. Resource ids are UUIDs so a stale entry
    could not be read back anyway, but a test that asserts a mode's behaviour should not
    depend on that — and leaving entries behind would let this file evict another file's.
    """
    precision.clear_index_cache()
    yield
    precision.clear_index_cache()


def _seed_ai_user(db, user_id: str = "precision-user") -> str:
    """A user holding ai_access with a live session — the minimum to reach /chat/asks."""
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
            first_name="Precision",
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
    # One permission row shared by every user this file seeds: `machine_name` is the
    # identity, and inserting a second "ai_access" would make which row a role points at
    # arbitrary.
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


def _seed_resource(db, user_id: str = "precision-user") -> tuple[Resource, File, list[DocumentChunk]]:
    """A ready knowledge base: one file, six positioned chunks, and a graph entity.

    `upload_status=True` is what makes it answerable — `_not_ready_note` refuses on a base
    whose indexing has not succeeded, and that refusal is terminal for every mode.
    """
    resource = Resource(resource_name="Handbook", user_id=user_id, upload_status=True)
    db.add(resource)
    db.flush()
    record = File(
        file_name="handbook.pdf",
        file_type="application/pdf",
        file_url="uploads/rag/handbook.pdf",
        resource_id=resource.id,
    )
    db.add(record)
    db.flush()

    chunks = []
    for index, (title, content) in enumerate(PASSAGES):
        # The six-line contextual header ingestion writes. Term frequencies are computed
        # over this, not over `content` — see the glossary entry for contextual_content.
        contextual = (
            f"Resource: Handbook. Source file: handbook.pdf. Title: {title}. "
            f"Modality: text. Chunk {index}. Content: {content}"
        )
        chunk = DocumentChunk(
            resource_id=resource.id,
            file_id=record.id,
            chunk_index=index,
            source_name="handbook.pdf",
            modality="text",
            title=title,
            content=content,
            contextual_content=contextual,
            terms_json=json.dumps(_terms(contextual)),
            page_start=index + 1,
            page_end=index + 1,
            char_start=index * 200,
            char_end=index * 200 + len(content),
        )
        db.add(chunk)
        chunks.append(chunk)
    db.flush()

    # graph_rag and agentic_rag read `rag_graph_entities` at query time. Without a row here
    # both fall back to pure lexical scoring, and the isolation test would be comparing two
    # runs of a code path the new mode could not have touched anyway.
    db.add(
        RagGraphEntity(
            resource_id=resource.id,
            name="Vacation Accrual",
            entity_type="concept",
            weight=3,
            chunk_refs_json=json.dumps([chunks[0].id, chunks[1].id]),
        )
    )
    db.commit()
    return resource, record, chunks


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _ask(client, token, resource_id, rag_type, query=QUERY, chat_history_name="session-1"):
    return client.post(
        "/chat/asks",
        params={
            "query": query,
            "chat_history_name": chat_history_name,
            "brain_id": resource_id,
            "rag_type": rag_type,
        },
        headers=_auth(token),
    )


def _citations(response) -> list[dict]:
    """The decoded X-CorpusTrace-Citations header, or [] when the answer cited nothing.

    Invariant 24: citations ride on the header as base64url JSON, never in the streamed
    body — so this is the only place a caller can see what an answer was grounded in.
    """
    raw = response.headers.get("x-corpustrace-citations")
    if not raw:
        return []
    return json.loads(base64.urlsafe_b64decode(raw))


def _cited_chunks(response) -> list[tuple[str, float]]:
    """The ordered (chunk id, score) pairs an answer cited — one mode's retrieval, pinned."""
    return [(citation["chunk_id"], citation["score"]) for citation in _citations(response)]


def _trace(client, token, resource_id, query=QUERY, overrides=None):
    payload = {"query": query}
    if overrides is not None:
        payload["overrides"] = overrides
    return client.post(
        f"/resources/{resource_id}/precision-trace", json=payload, headers=_auth(token)
    )


def _strings_in(value) -> list[str]:
    """Every string leaf of a nested JSON structure, keys included."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [
            text
            for key, item in value.items()
            for text in ([key] + _strings_in(item))
        ]
    if isinstance(value, list):
        return [text for item in value for text in _strings_in(item)]
    return []


class TestModeRegistration:
    """A mode the dispatcher does not know about is not rejected — it is substituted."""

    def test_high_precision_is_a_supported_mode(self):
        assert precision.RAG_MODE_HIGH_PRECISION == "high_precision"
        assert "high_precision" in rag_service.SUPPORTED_RAG_MODES
        # The seven modes the chat picker can ask for. Pinned as a whole so a mode removed
        # from the set is as visible as one added.
        assert rag_service.SUPPORTED_RAG_MODES == {
            *PRE_EXISTING_MODES,
            "high_precision",
        }

    def test_the_mode_id_round_trips_through_normalize_rag_mode(self):
        # Not a formality: `normalize_rag_mode` is the only validation `rag_type` gets, and
        # it answers an unrecognised mode with `contextual_hybrid` rather than an error. If
        # the id ever left SUPPORTED_RAG_MODES, every High Precision question would be
        # answered by hybrid retrieval and nothing anywhere would say so.
        assert rag_service.normalize_rag_mode("high_precision") == "high_precision"
        # The client may send the hyphenated spelling; `-` is folded to `_`.
        assert rag_service.normalize_rag_mode("High-Precision") == "high_precision"
        assert rag_service.normalize_rag_mode("  HIGH_PRECISION  ") == "high_precision"

    def test_an_unknown_mode_still_falls_back_to_contextual_hybrid(self):
        # The forgiveness that makes the test above necessary is itself the documented
        # behaviour (invariant 16) and must survive the seventh mode being added.
        assert rag_service.normalize_rag_mode("high_precisian") == "contextual_hybrid"
        assert rag_service.normalize_rag_mode(None) == "contextual_hybrid"
        assert rag_service.normalize_rag_mode("") == "contextual_hybrid"


class TestIndexingRecommendation:
    def test_the_mode_has_a_chunking_recommendation(self):
        assert "high_precision" in chunking.RAG_MODE_RECOMMENDATIONS
        recommendation = chunking.RAG_MODE_RECOMMENDATIONS["high_precision"]
        assert recommendation["strategy"] == chunking.STRATEGY_SENTENCE
        assert recommendation["chunk_size"] == 600
        # Zero, and the only mode that recommends it: parent-chunk recovery restores the
        # neighbouring passages at the end, so overlap would only feed duplicated text to
        # the dedup stage that then has to spend candidate slots removing it.
        assert recommendation["overlap"] == 0
        assert recommendation["why"].strip()
        assert chunking.recommendation_for("high_precision") is recommendation

    def test_the_label_is_exactly_what_the_answer_prints(self):
        # `_compose_answer` derives the name it prints as mode.replace("_", " ").title().
        # A label spelled differently here ("High-Precision") would put two names for one
        # mode in front of the same user — the answer body saying one, the upload form the
        # other.
        assert chunking.RAG_MODE_RECOMMENDATIONS["high_precision"]["label"] == "High Precision"
        assert "high_precision".replace("_", " ").title() == "High Precision"
        assert precision.RAG_MODE_HIGH_PRECISION_LABEL == "High Precision"

    def test_the_recommended_values_survive_normalize_config_unchanged(self):
        # `normalize_config` clamps rather than rejects, so a recommendation outside the
        # bounds would be silently rewritten and the form would offer a setting the
        # uploader can never actually get.
        recommendation = chunking.RAG_MODE_RECOMMENDATIONS["high_precision"]
        config = chunking.normalize_config(recommendation)

        assert config.strategy == recommendation["strategy"]
        assert config.chunk_size == recommendation["chunk_size"]
        assert config.overlap == recommendation["overlap"]
        # A recommendation may never turn embeddings on: that sends the document to a third
        # party and, on three of the four providers, costs money.
        assert config.embedding_provider is None and config.embedding_model is None
        assert config.embeds is False

    def test_the_catalogue_endpoint_serves_the_recommendation(self, client, db):
        # The upload form renders this payload. A mode missing from `recommendations` is a
        # mode the form cannot offer settings for, with nothing in the UI saying why.
        token = _seed_ai_user(db)

        response = client.get("/resources/indexing-options", headers=_auth(token))

        assert response.status_code == 200
        recommendations = response.json()["data"]["recommendations"]
        assert set(recommendations) == rag_service.SUPPORTED_RAG_MODES
        served = recommendations["high_precision"]
        assert served["label"] == "High Precision"
        assert served["strategy"] == chunking.STRATEGY_SENTENCE
        assert served["chunk_size"] == 600
        assert served["overlap"] == 0


class TestAskingWithHighPrecision:
    def test_the_answer_names_the_mode_and_cites_real_chunks(self, client, db):
        token = _seed_ai_user(db)
        resource, record, chunks = _seed_resource(db)

        response = _ask(client, token, resource.id, "high_precision")

        assert response.status_code == 200
        # The mode has to be identifiable from its output, or a user cannot tell which
        # retriever answered them.
        assert 'Using High Precision on "Handbook"' in response.text
        citations = _citations(response)
        assert citations, "a base whose every term the query matches must cite something"
        known_ids = {chunk.id for chunk in chunks}
        for position, citation in enumerate(citations, start=1):
            assert citation["chunk_id"] in known_ids
            assert citation["file_id"] == record.id
            assert citation["source_name"] == "handbook.pdf"
            # The chip number and the "[N]" marker come from one `results[:5]` slice.
            assert citation["index"] == position
            assert f"[{position}]" in response.text

    def test_the_answer_reports_which_retrievers_ran(self, client, db):
        # `_precision_notes`. The pipeline has stages a user can act on — whether the dense
        # side participated, whether a reranker ran — and none of them are visible in a
        # ranking alone.
        token = _seed_ai_user(db)
        resource, _, _ = _seed_resource(db)

        response = _ask(client, token, resource.id, "high_precision")

        assert "Retrieval:" in response.text
        # This base has no embeddings, which is the default state of every knowledge base
        # in the installation. Saying "lexical + embedding retrieval" here would be a claim
        # about a retriever that did not run.
        assert "lexical retrieval" in response.text
        assert "lexical + embedding retrieval" not in response.text
        assert "cross-encoder rerank" in response.text

    def test_the_citation_chunks_belong_to_the_resource_that_was_asked(self, client, db):
        token = _seed_ai_user(db)
        resource, _, _ = _seed_resource(db)
        other, _, other_chunks = _seed_resource(db)

        response = _ask(client, token, resource.id, "high_precision")

        cited = {citation["chunk_id"] for citation in _citations(response)}
        assert cited.isdisjoint({chunk.id for chunk in other_chunks})

    def test_the_persisted_turn_records_high_precision_not_the_fallback(self, client, db):
        """The mode that answered is stored, and the stored mode is what reloads.

        A mode missing from SUPPORTED_RAG_MODES normalises away silently, so this row would
        say `contextual_hybrid` while the user believed — and the picker showed — something
        else. History would then be a record of a question nobody asked.
        """
        token = _seed_ai_user(db)
        resource, _, _ = _seed_resource(db)

        response = _ask(client, token, resource.id, "high_precision")

        record = db.query(ChatHistory).one()
        assert record.brain == f"{resource.id}:high_precision"
        assert record.brain != f"{resource.id}:contextual_hybrid"
        assert record.user_query == QUERY
        assert record.bot_response == response.text

    def test_a_resource_with_no_chunks_is_the_existing_refusal_not_an_error(self, client, db):
        """An empty corpus is a first stage that returns nothing, not an exception.

        `retrieve` records "resource has no indexed chunks" and returns an empty outcome, so
        `_plan_answer` reaches the same terminal note every other mode produces. There is no
        global exception handler here: anything raised would be a bare text/plain 500.
        """
        token = _seed_ai_user(db)
        empty = Resource(resource_name="Empty", user_id="precision-user", upload_status=True)
        db.add(empty)
        db.commit()

        response = _ask(client, token, empty.id, "high_precision")

        assert response.status_code == 200
        assert "no indexed document chunks" in response.text
        assert _citations(response) == []
        # Still recorded as the mode that was asked for — the refusal came from this mode.
        assert db.query(ChatHistory).one().brain == f"{empty.id}:high_precision"

    def test_another_users_resource_is_a_404_envelope(self, client, db):
        token = _seed_ai_user(db)
        stranger, _, _ = _seed_resource(db, "stranger")

        response = _ask(client, token, stranger.id, "high_precision")

        assert response.status_code == 404
        assert response.json()["message"] == "Resource not found or not accessible"


class TestExistingModesAreUnaffected:
    """The centrepiece: High Precision must be invisible to the six modes that predate it.

    `rag/precision/` shares this module's tokenizer, its per-chunk scoring memo and the
    dense candidates one embedding call produced, and it caches a derived index keyed by
    resource id. Each of those is a plausible route by which running the new mode could
    change what an old one retrieves from the same corpus — and the symptom would be a
    ranking that shifted for users who never selected the new mode at all.
    """

    def test_every_mode_answers_identically_before_and_after_precision_runs(self, client, db):
        token = _seed_ai_user(db)
        resource, _, _ = _seed_resource(db)

        before = {}
        for mode in PRE_EXISTING_MODES:
            response = _ask(client, token, resource.id, mode)
            assert response.status_code == 200, f"{mode} did not answer before the precision run"
            before[mode] = _cited_chunks(response)
            assert before[mode], f"{mode} cited nothing, so comparing it would test nothing"

        precision_response = _ask(client, token, resource.id, "high_precision")
        assert precision_response.status_code == 200
        precision_cited = _cited_chunks(precision_response)
        assert precision_cited, "the precision run must actually have retrieved"
        # "Identical before and after" would be trivially true if every mode returned the
        # same thing anyway. It does not: High Precision ranks this corpus differently from
        # all six, which is what makes the equality below a statement about isolation.
        assert all(precision_cited != cited for cited in before.values())

        after = {}
        for mode in PRE_EXISTING_MODES:
            response = _ask(client, token, resource.id, mode)
            assert response.status_code == 200, f"{mode} stopped answering after the precision run"
            after[mode] = _cited_chunks(response)

        # Ids AND scores: an ordering that survives while the scores move means a signal
        # changed and only happened not to reorder this corpus.
        assert after == before

    def test_every_mode_still_produces_the_same_answer_text(self, client, db):
        # The citations are the retrieval; the body is what the user reads. Both are
        # deterministic for a fixed corpus, so a difference in either is a real change.
        token = _seed_ai_user(db)
        resource, _, _ = _seed_resource(db)

        before = {mode: _ask(client, token, resource.id, mode).text for mode in PRE_EXISTING_MODES}
        _ask(client, token, resource.id, "high_precision")
        after = {mode: _ask(client, token, resource.id, mode).text for mode in PRE_EXISTING_MODES}

        assert after == before

    def test_the_precision_trace_endpoint_does_not_disturb_the_other_modes(self, client, db):
        """The trace runs the pipeline for real, including its index build and cache write.

        It is the one surface that runs precision retrieval without going through
        `_plan_answer`, so it is where a shared-state mistake would show up first.
        """
        token = _seed_ai_user(db)
        resource, _, _ = _seed_resource(db)

        before = {mode: _cited_chunks(_ask(client, token, resource.id, mode)) for mode in PRE_EXISTING_MODES}
        assert _trace(client, token, resource.id).status_code == 200
        after = {mode: _cited_chunks(_ask(client, token, resource.id, mode)) for mode in PRE_EXISTING_MODES}

        assert after == before

    def test_asking_the_six_modes_after_precision_still_records_their_own_mode(self, client, db):
        # `chat_history.brain` is what the transcript is scoped and labelled by. A mode
        # answering under another mode's name is a history that lies about itself.
        token = _seed_ai_user(db)
        resource, _, _ = _seed_resource(db)

        _ask(client, token, resource.id, "high_precision")
        for mode in PRE_EXISTING_MODES:
            assert _ask(client, token, resource.id, mode).status_code == 200

        stored = {record.brain for record in db.query(ChatHistory).all()}
        assert stored == {f"{resource.id}:{mode}" for mode in (*PRE_EXISTING_MODES, "high_precision")}


class TestPrecisionTraceEndpoint:
    def test_it_returns_the_mode_its_trace_and_its_results(self, client, db):
        token = _seed_ai_user(db)
        resource, _, chunks = _seed_resource(db)

        response = _trace(client, token, resource.id)

        assert response.status_code == 200
        body = response.json()
        assert set(body) == {"status", "status_code", "message", "data"}
        data = body["data"]
        assert data["mode"] == "high_precision"
        assert data["resource_id"] == resource.id
        assert data["resource_name"] == "Handbook"
        assert data["chunks_indexed"] == len(chunks)
        # No embeddings on this base, and the trace must say which retrievers were
        # available rather than leaving a reader to assume the dense side ran.
        assert data["embeddings_available"] is False
        assert data["embedding_model"] is None
        assert data["trace"]["final_chunk_ids"] == [result["chunk_id"] for result in data["results"]]
        assert {result["chunk_id"] for result in data["results"]} <= {chunk.id for chunk in chunks}

    def test_the_trace_names_the_normalized_query_and_every_stage(self, client, db):
        token = _seed_ai_user(db)
        resource, _, _ = _seed_resource(db)

        response = _trace(client, token, resource.id, query="  How many VACATION days accrue?  ")

        trace = response.json()["data"]["trace"]
        assert trace["original_query"] == "  How many VACATION days accrue?  "
        # Case folded, the question mark and surrounding whitespace gone. This is the
        # string the corpus is actually searched with, and it is the only thing that
        # explains why a term the user typed matched nothing.
        assert trace["normalized_query"] == "how many vacation days accrue"
        stages = [record["stage"] for record in trace["stages"]]
        assert stages == [
            "metadata_filter",
            "bm25",
            "dense",
            "candidate_pool",
            "rerank",
            "dedup",
            "mmr",
            "parent_recovery",
        ]
        for record in trace["stages"]:
            assert isinstance(record["in"], int) and isinstance(record["out"], int)
        assert trace["reranker_backend"] == precision.RERANKER_LEXICAL
        assert trace["reranker_error"] is None
        assert set(trace["timings_ms"]) >= {"normalize", "bm25", "rerank", "total"}

    def test_the_trace_carries_ids_counts_and_scores_but_no_passage_text(self, client, db):
        """The log-safety property the module claims, asserted against the wire.

        `_log_trace` writes this same structure to the application log, and the log is what
        gets pasted into a bug report. A trace that quoted the passages it ranked would
        turn every diagnostic into a disclosure of somebody's documents.
        """
        token = _seed_ai_user(db)
        resource, _, chunks = _seed_resource(db)

        data = _trace(client, token, resource.id).json()["data"]
        serialized = json.dumps(data["trace"]).lower()

        for chunk in chunks:
            assert chunk.content.lower() not in serialized
            # Single words are legitimate — expansion and pseudo-relevance feedback add
            # corpus terms to the query, and the trace has to name them to be explicable.
            # A four-word run is not a term; it is the document.
            words = chunk.content.lower().split()
            for start in range(len(words) - 3):
                window = " ".join(words[start : start + 4])
                assert window not in serialized, f"trace quoted the corpus: {window!r}"

        # Not an accident of the corpus: the trace does identify what it ranked, by id.
        assert data["trace"]["final_chunk_ids"]
        assert set(data["trace"]["final_chunk_ids"]) <= {chunk.id for chunk in chunks}
        # …and the passage text lives on `results`, which is the response body only.
        assert any(result["text"] for result in data["results"])
        assert chunks[0].content in json.dumps(data["results"])

    def test_no_multi_word_string_in_the_trace_comes_from_the_corpus(self, client, db):
        """A tighter statement of the same rule, and a deliberate ratchet.

        Every multi-word string in a trace is either the question the caller sent or one of
        the engine's own fixed phrases. A new stage detail carrying prose fails here, which
        is the point: the next person to add one has to look at where its text came from.
        """
        token = _seed_ai_user(db)
        resource, _, _ = _seed_resource(db)

        trace = _trace(client, token, resource.id).json()["data"]["trace"]

        for text in _strings_in(trace):
            if " " not in text:
                # Single terms are legitimate and necessary: expansion and pseudo-relevance
                # feedback pull corpus terms into the query, and a trace that hid them
                # could not explain why a passage was retrieved.
                continue
            assert (
                text.lower() in QUERY.lower() or text in ENGINE_PHRASES
            ), f"multi-word trace string {text!r} is neither the query nor a known engine phrase"

    def test_overrides_change_the_run_without_changing_the_deployment(self, client, db):
        # The diagnostic exists so one knob can be tried against a real base. A sparse
        # patch must apply to this run only — the process-wide config is `lru_cache`d and
        # shared by every question every other user is asking.
        token = _seed_ai_user(db)
        resource, _, _ = _seed_resource(db)

        response = _trace(client, token, resource.id, overrides={"final_k": 2, "not_a_field": 1})

        assert response.status_code == 200
        data = response.json()["data"]
        assert len(data["results"]) == 2
        assert precision.get_precision_config().final_k == 10

    def test_it_persists_nothing(self, client, db):
        # It is a question about the retriever, not a question the user asked their
        # documents. A trace in the transcript would appear as a turn nobody sent.
        token = _seed_ai_user(db)
        resource, _, _ = _seed_resource(db)

        assert _trace(client, token, resource.id).status_code == 200

        assert db.query(ChatHistory).count() == 0

    def test_another_users_resource_is_a_404_not_a_403(self, client, db):
        # A 403 would confirm the id exists, which is the fact being protected.
        token = _seed_ai_user(db)
        _seed_ai_user(db, "stranger")
        stranger_resource, _, _ = _seed_resource(db, "stranger")

        response = _trace(client, token, stranger_resource.id)

        assert response.status_code == 404
        assert response.json()["message"] == "Knowledge base not found"

    def test_an_unknown_resource_is_a_404(self, client, db):
        token = _seed_ai_user(db)

        assert _trace(client, token, "no-such-resource").status_code == 404

    def test_anonymous_callers_are_rejected(self, client, db):
        _seed_ai_user(db)
        resource, _, _ = _seed_resource(db)

        response = client.post(
            f"/resources/{resource.id}/precision-trace", json={"query": QUERY}
        )

        assert response.status_code == 401

    def test_an_empty_query_is_rejected_by_the_schema(self, client, db):
        # `min_length=1`: a blank query would run the whole pipeline to return nothing and
        # read as the mode being broken.
        token = _seed_ai_user(db)
        resource, _, _ = _seed_resource(db)

        response = client.post(
            f"/resources/{resource.id}/precision-trace", json={"query": ""}, headers=_auth(token)
        )

        assert response.status_code == 422

