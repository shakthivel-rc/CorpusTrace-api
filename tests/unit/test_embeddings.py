"""Unit tests for the embedding capability in services/llm_provider.py.

Embeddings exist to close the one gap lexical retrieval cannot: a question that shares no
vocabulary with the document scores zero, which is indistinguishable from the document not
covering the topic. That makes the failure modes here quiet rather than loud — a vector
paired with the wrong passage does not raise, it just retrieves the wrong paragraph
forever, and re-indexing is the only fix. So the properties pinned below are the ones
whose breakage is invisible:

* order — three providers, three wire shapes, and OpenAI documents that results may come
  back out of order. Trusting arrival order pairs every vector with the wrong chunk;
* count — a batch that returns fewer vectors than passages must raise, not truncate.
  Accepting it shifts every following vector by one;
* type — a malformed vector must become an `LlmProviderError`, never a stray TypeError.
  There is no global exception handler in this app, so an unhandled one is a bare 500;
* secrecy — a key must never reach the logs, on the success path or the failure path.

HTTP is mocked at the single seam, `llm.http_json`. Nothing here touches the network.
"""
import json
import logging
import math

import pytest

import services.llm_provider as llm
from core.logging import JsonFormatter
from models.llm import LlmProviderCredential
from services.llm_provider import (
    SUPPORTED_EMBEDDING_PROVIDERS,
    SUPPORTED_PROVIDERS,
    LlmProviderError,
    call_gemini_embeddings,
    call_ollama_embeddings,
    call_openai_embeddings,
    embed_texts,
    embedding_model_spec,
    embedding_provider_options,
    encrypt_secret,
)

pytestmark = pytest.mark.unit

USER = "user-embed-1"

# Distinctive enough that a substring search for it cannot match anything incidental.
SECRET_KEY_VALUE = "sk-live-EMBEDKEY-9f3c1d2b4a"

# The api_styles embed_texts() can actually dispatch. Anything else falls through to
# "Unsupported embedding provider" at request time.
EMBEDDABLE_STYLES = {"openai", "gemini", "ollama"}


def _seed_credential(db, provider: str, api_key: str = SECRET_KEY_VALUE, base_url: str | None = None):
    """A stored, active credential — the same shape upsert_provider_credential writes."""
    credential = LlmProviderCredential(
        user_id=USER,
        provider=provider,
        encrypted_api_key=encrypt_secret(api_key),
        base_url=base_url or SUPPORTED_PROVIDERS[provider]["default_base_url"],
    )
    db.add(credential)
    db.commit()
    return credential


def _capture_http(monkeypatch, responder):
    """Replace the one HTTP seam and record every call made through it."""
    calls: list[dict] = []

    def fake_http(method, url, headers, payload=None):
        calls.append({"method": method, "url": url, "headers": headers, "payload": payload})
        return responder(payload)

    monkeypatch.setattr(llm, "http_json", fake_http)
    return calls


class TestOpenAiWireShape:
    def test_vectors_come_back_sorted_by_index_not_by_arrival(self, monkeypatch):
        """OpenAI documents that `data` may arrive in any order, keyed by `index`.

        If this regresses, every vector is stored against the wrong passage and retrieval
        is quietly wrong for the whole document — nothing raises, nothing logs.
        """
        _capture_http(
            monkeypatch,
            lambda payload: {
                "data": [
                    {"index": 2, "embedding": [3.0, 3.0]},
                    {"index": 0, "embedding": [1.0, 1.0]},
                    {"index": 1, "embedding": [2.0, 2.0]},
                ]
            },
        )

        vectors = call_openai_embeddings("k", "https://api.openai.com/v1", "text-embedding-3-small", ["a", "b", "c"])

        assert vectors == [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]

    def test_the_request_carries_the_model_the_passages_and_the_bearer_key(self, monkeypatch):
        calls = _capture_http(monkeypatch, lambda payload: {"data": [{"index": 0, "embedding": [0.5]}]})

        call_openai_embeddings("k", "https://api.openai.com/v1", "text-embedding-3-small", ["alpha"])

        assert calls[0]["url"] == "https://api.openai.com/v1/embeddings"
        assert calls[0]["payload"] == {"model": "text-embedding-3-small", "input": ["alpha"]}
        assert calls[0]["headers"]["Authorization"] == "Bearer k"

    def test_integers_on_the_wire_become_floats(self, monkeypatch):
        """JSON has one number type; a provider may serialise 0 as `0`. Cosine similarity
        divides by the norm, so an int-typed vector is a latent integer-division trap."""
        _capture_http(monkeypatch, lambda payload: {"data": [{"index": 0, "embedding": [1, 0, -2]}]})

        vector = call_openai_embeddings("k", "https://x/v1", "m", ["alpha"])[0]

        assert vector == [1.0, 0.0, -2.0]
        assert all(isinstance(value, float) for value in vector)

    def test_a_response_with_no_data_array_is_a_provider_error(self, monkeypatch):
        # An OpenAI-compatible clone that answers `{"error": ...}` with HTTP 200 would
        # otherwise surface as an AttributeError deep inside the indexing worker.
        _capture_http(monkeypatch, lambda payload: {"error": {"message": "nope"}})

        with pytest.raises(LlmProviderError) as excinfo:
            call_openai_embeddings("k", "https://x/v1", "m", ["alpha"])
        assert excinfo.value.status_code == 502


class TestGeminiWireShape:
    def test_it_parses_the_values_key_and_hits_batch_embed_contents(self, monkeypatch):
        """Gemini's batch endpoint is a different verb on the model path — `:embedContent`
        embeds exactly one passage, so hitting it with a batch silently returns one vector
        for N chunks."""
        calls = _capture_http(
            monkeypatch,
            lambda payload: {"embeddings": [{"values": [0.1, 0.2]}, {"values": [0.3, 0.4]}]},
        )

        vectors = call_gemini_embeddings("k", "https://generativelanguage.googleapis.com/v1beta", "text-embedding-004", ["a", "b"])

        assert vectors == [[0.1, 0.2], [0.3, 0.4]]
        assert ":batchEmbedContents" in calls[0]["url"]

    def test_every_sub_request_names_the_model_with_the_models_prefix(self, monkeypatch):
        """Gemini rejects a batch whose sub-requests omit `model`, and the value must be
        the fully-qualified `models/<id>` path even though the URL already carries it."""
        calls = _capture_http(monkeypatch, lambda payload: {"embeddings": [{"values": [0.1]}]})

        call_gemini_embeddings("k", "https://x/v1beta", "text-embedding-004", ["alpha"])

        requests = calls[0]["payload"]["requests"]
        assert requests == [{"model": "models/text-embedding-004", "content": {"parts": [{"text": "alpha"}]}}]

    def test_an_unexpected_body_is_a_provider_error(self, monkeypatch):
        _capture_http(monkeypatch, lambda payload: {"embeddings": "not-a-list"})

        with pytest.raises(LlmProviderError) as excinfo:
            call_gemini_embeddings("k", "https://x/v1beta", "text-embedding-004", ["alpha"])
        assert excinfo.value.status_code == 502


class TestOllamaWireShape:
    def test_it_parses_bare_vectors_and_sends_no_authorization_header(self, monkeypatch):
        """Ollama runs on the user's own machine and has no accounts. Sending an
        Authorization header would mean the code believes a key exists for it — the path
        that then demands one and makes the only private, keyless provider unusable."""
        calls = _capture_http(monkeypatch, lambda payload: {"embeddings": [[0.1, 0.2], [0.3, 0.4]]})

        vectors = call_ollama_embeddings("http://localhost:11434", "nomic-embed-text", ["a", "b"])

        assert vectors == [[0.1, 0.2], [0.3, 0.4]]
        assert calls[0]["url"] == "http://localhost:11434/api/embed"
        assert "Authorization" not in calls[0]["headers"]
        assert calls[0]["payload"] == {"model": "nomic-embed-text", "input": ["a", "b"]}

    def test_an_unexpected_body_is_a_provider_error(self, monkeypatch):
        _capture_http(monkeypatch, lambda payload: {"error": "model not found"})

        with pytest.raises(LlmProviderError) as excinfo:
            call_ollama_embeddings("http://localhost:11434", "nomic-embed-text", ["a"])
        assert excinfo.value.status_code == 502


class TestMalformedVectors:
    """Every one of these arrives as a well-formed HTTP 200 JSON body. Without the guard
    they escape as TypeError/ValueError from inside the indexing worker — and this app has
    no global exception handler, so an unhandled one is a bare plain-text 500 with no
    per-document explanation of what went wrong."""

    @pytest.mark.parametrize(
        "embedding, reason",
        [
            ("0.1,0.2", "a string instead of a list"),
            (None, "a null where a vector was promised"),
            ({"values": [0.1]}, "a nested object instead of a list"),
            ([], "an empty vector — a zero-dimension vector has no cosine similarity"),
            (["alpha", "beta"], "a list of non-numeric strings"),
        ],
    )
    def test_a_malformed_vector_raises_a_provider_error(self, monkeypatch, embedding, reason):
        _capture_http(monkeypatch, lambda payload: {"embeddings": [embedding]})

        with pytest.raises(LlmProviderError) as excinfo:
            call_ollama_embeddings("http://localhost:11434", "nomic-embed-text", ["alpha"])
        assert excinfo.value.status_code == 502, reason

    def test_numeric_strings_are_still_accepted(self, monkeypatch):
        """`float("0.1")` succeeds, so a provider that serialises floats as strings is not
        malformed — rejecting it would be a false alarm on a working provider."""
        _capture_http(monkeypatch, lambda payload: {"embeddings": [["0.1", "0.2"]]})

        assert call_ollama_embeddings("http://x", "m", ["alpha"]) == [[0.1, 0.2]]

    def test_a_malformed_vector_inside_embed_texts_does_not_escape_as_a_type_error(self, db, monkeypatch):
        """The same guard, exercised through the public entry point the worker calls."""
        _capture_http(monkeypatch, lambda payload: {"embeddings": [None]})

        with pytest.raises(LlmProviderError):
            embed_texts(db, USER, "ollama", "nomic-embed-text", ["alpha"])


class TestEmbedTextsBatching:
    def test_n_texts_are_split_into_ceil_n_over_batch_requests_and_return_in_input_order(self, db, monkeypatch):
        """Batching is internal, so the caller hands over every chunk of a document and
        must get every vector back, aligned. A boundary slip here (an off-by-one in the
        range step, or extend() on the wrong list) silently drops or reorders vectors."""
        monkeypatch.setattr(llm, "EMBEDDING_BATCH_SIZE", 2)
        texts = [f"passage-{index}" for index in range(5)]

        # Each vector encodes its own passage, so order is checkable rather than assumed.
        calls = _capture_http(
            monkeypatch,
            lambda payload: {"embeddings": [[float(int(text.split("-")[1]))] for text in payload["input"]]},
        )

        vectors = embed_texts(db, USER, "ollama", "nomic-embed-text", texts)

        assert len(calls) == math.ceil(len(texts) / 2) == 3
        assert [len(call["payload"]["input"]) for call in calls] == [2, 2, 1]
        assert vectors == [[0.0], [1.0], [2.0], [3.0], [4.0]]

    def test_a_single_batch_is_a_single_request(self, db, monkeypatch):
        calls = _capture_http(monkeypatch, lambda payload: {"embeddings": [[0.1] for _ in payload["input"]]})

        vectors = embed_texts(db, USER, "ollama", "nomic-embed-text", ["a", "b", "c"])

        assert len(calls) == 1
        assert len(vectors) == 3

    def test_no_passages_means_no_request_at_all(self, db, monkeypatch):
        # A document that extracted nothing must not bill the user for an empty call —
        # and several providers reject an empty `input` array with a 400.
        calls = _capture_http(monkeypatch, lambda payload: {"embeddings": []})

        assert embed_texts(db, USER, "ollama", "nomic-embed-text", []) == []
        assert calls == []

    def test_an_over_long_passage_is_truncated_before_it_is_sent(self, db, monkeypatch):
        """The cap is the guard against a caller passing something that is not a chunk.
        Without it an oversized payload is either a provider 400 mid-job or a real bill."""
        calls = _capture_http(monkeypatch, lambda payload: {"embeddings": [[0.1] for _ in payload["input"]]})

        embed_texts(db, USER, "ollama", "nomic-embed-text", ["x" * (llm.MAX_EMBEDDING_INPUT_CHARS + 5000)])

        assert len(calls[0]["payload"]["input"][0]) == llm.MAX_EMBEDDING_INPUT_CHARS


class TestEmbedTextsContract:
    def test_a_short_batch_raises_instead_of_being_accepted(self, db, monkeypatch):
        """The single most dangerous provider misbehaviour. Vectors are zipped back onto
        chunks by position, so accepting 2 vectors for 3 passages shifts every following
        vector onto the wrong chunk — for the rest of the document and for every document
        after it in the same job. Nothing raises later; retrieval is just wrong."""
        _capture_http(monkeypatch, lambda payload: {"embeddings": [[0.1], [0.2]]})

        with pytest.raises(LlmProviderError) as excinfo:
            embed_texts(db, USER, "ollama", "nomic-embed-text", ["a", "b", "c"])

        assert excinfo.value.status_code == 502
        assert "2 vectors for 3 passages" in str(excinfo.value)

    def test_more_vectors_than_passages_is_also_refused(self, db, monkeypatch):
        # The mirror case: an extra vector means the alignment assumption is already broken.
        _capture_http(monkeypatch, lambda payload: {"embeddings": [[0.1], [0.2], [0.3]]})

        with pytest.raises(LlmProviderError):
            embed_texts(db, USER, "ollama", "nomic-embed-text", ["a", "b"])

    def test_a_short_batch_is_caught_even_when_it_is_not_the_first_batch(self, db, monkeypatch):
        """A 200-chunk PDF fails on batch seven, not batch one — the count check has to run
        per batch, not once on the assembled total."""
        monkeypatch.setattr(llm, "EMBEDDING_BATCH_SIZE", 2)
        seen = {"batches": 0}

        def responder(payload):
            seen["batches"] += 1
            if seen["batches"] == 3:
                return {"embeddings": [[0.1]]}
            return {"embeddings": [[0.1] for _ in payload["input"]]}

        _capture_http(monkeypatch, responder)

        with pytest.raises(LlmProviderError):
            embed_texts(db, USER, "ollama", "nomic-embed-text", ["a", "b", "c", "d", "e", "f"])

    @pytest.mark.parametrize("provider", ["anthropic", "groq"])
    def test_a_chat_only_provider_is_refused_before_any_http(self, db, monkeypatch, provider):
        """Anthropic, Groq, Cerebras and OpenRouter serve chat completions only. Dispatching
        to them would POST /embeddings and get a 404 — a 502 blaming the provider for a
        choice this code made. It must be a 400 that names the real problem."""

        def explode(*args, **kwargs):
            raise AssertionError("no HTTP call may be made for a chat-only provider")

        monkeypatch.setattr(llm, "http_json", explode)
        _seed_credential(db, provider)

        with pytest.raises(LlmProviderError) as excinfo:
            embed_texts(db, USER, provider, "whatever-model", ["alpha"])

        assert excinfo.value.status_code == 400
        assert "does not offer an embeddings API" in str(excinfo.value)
        assert provider in str(excinfo.value)

    def test_an_unknown_provider_is_a_400_too(self, db):
        with pytest.raises(LlmProviderError) as excinfo:
            embed_texts(db, USER, "not-a-provider", "m", ["alpha"])
        assert excinfo.value.status_code == 400

    def test_a_key_required_provider_without_a_credential_is_a_400(self, db, monkeypatch):
        """Reported as a configuration problem the user can fix, not a provider fault."""

        def explode(*args, **kwargs):
            raise AssertionError("no HTTP call may be made without a credential")

        monkeypatch.setattr(llm, "http_json", explode)

        with pytest.raises(LlmProviderError) as excinfo:
            embed_texts(db, USER, "openai", "text-embedding-3-small", ["alpha"])
        assert excinfo.value.status_code == 400

    def test_the_stored_credential_is_decrypted_and_sent(self, db, monkeypatch):
        """Credentials are Fernet-encrypted at rest; embedding must resolve them through
        the same path a chat call does rather than growing its own."""
        _seed_credential(db, "openai")
        calls = _capture_http(
            monkeypatch,
            lambda payload: {"data": [{"index": i, "embedding": [0.1]} for i in range(len(payload["input"]))]},
        )

        embed_texts(db, USER, "openai", "text-embedding-3-small", ["alpha"])

        assert calls[0]["headers"]["Authorization"] == f"Bearer {SECRET_KEY_VALUE}"

    def test_ollama_needs_no_credential_at_all(self, db, monkeypatch):
        # The keyless, private default. Requiring a credential would make it unreachable.
        _capture_http(monkeypatch, lambda payload: {"embeddings": [[0.1]]})

        assert embed_texts(db, USER, "ollama", "nomic-embed-text", ["alpha"]) == [[0.1]]


class TestEmbeddingCatalogue:
    def test_it_lists_only_providers_with_an_embeddings_endpoint(self):
        """This is the dropdown the indexing UI renders. A provider listed here that has no
        embeddings API is an option that 404s at index time, after the upload succeeded."""
        listed = {entry["provider"] for entry in embedding_provider_options()}

        assert listed == set(SUPPORTED_EMBEDDING_PROVIDERS)
        for chat_only in ("anthropic", "groq", "cerebras", "openrouter", "cloudflare", "sambanova", "zai"):
            assert chat_only not in listed, f"{chat_only} serves chat completions only"

    def test_every_listed_provider_exists_in_the_main_registry(self):
        """`embed_texts` resolves credentials through `_resolve_call`, which indexes
        SUPPORTED_PROVIDERS directly — an id present here and missing there is a KeyError
        at request time, and therefore a bare 500."""
        for provider in SUPPORTED_EMBEDDING_PROVIDERS:
            assert provider in SUPPORTED_PROVIDERS, provider

    def test_each_entry_carries_the_display_name_from_the_main_registry(self):
        """Joined rather than duplicated: two copies of a provider's name drift, and the
        indexing dropdown then disagrees with the chat settings drawer about what to call
        the same provider."""
        for entry in embedding_provider_options():
            assert entry["display_name"] == SUPPORTED_PROVIDERS[entry["provider"]]["display_name"]
            assert entry["display_name"] != entry["provider"], "the fallback must not be what ships"
            assert entry["credential_required"] == SUPPORTED_PROVIDERS[entry["provider"]]["credential_required"]
            assert entry["free_tier"] == SUPPORTED_PROVIDERS[entry["provider"]]["free_tier"]

    def test_exactly_one_model_per_provider_is_recommended(self):
        """The UI preselects the recommended model. Zero means nothing is preselected and
        the user must choose blind; two means whichever the iteration reaches first wins,
        which is a silently order-dependent default."""
        for provider, spec in SUPPORTED_EMBEDDING_PROVIDERS.items():
            recommended = [model for model in spec["models"] if model.get("recommended")]
            assert len(recommended) == 1, f"{provider} has {len(recommended)} recommended models"

    def test_every_model_declares_the_fields_the_ui_renders(self):
        for provider, spec in SUPPORTED_EMBEDDING_PROVIDERS.items():
            assert spec["models"], f"{provider} has no embedding models"
            for model in spec["models"]:
                missing = {"id", "label", "dimensions", "recommended", "notes"} - model.keys()
                assert not missing, f"{provider}/{model.get('id')} is missing {missing}"
                # Dimensions drive the per-chunk storage cost, so a non-positive value is
                # not a display bug — it is a vector nothing can score against.
                assert isinstance(model["dimensions"], int) and model["dimensions"] > 0

    def test_every_embedding_api_style_can_actually_be_dispatched(self):
        """An api_style outside this set reaches the `else` in embed_texts and raises
        "Unsupported embedding provider" — after the upload was accepted."""
        for provider, spec in SUPPORTED_EMBEDDING_PROVIDERS.items():
            assert spec["api_style"] in EMBEDDABLE_STYLES, provider

    def test_model_ids_are_unique_within_a_provider(self):
        # `embedding_model_spec` returns the first match, so a duplicate id silently
        # resolves to whichever copy is listed first.
        for provider, spec in SUPPORTED_EMBEDDING_PROVIDERS.items():
            ids = [model["id"] for model in spec["models"]]
            assert len(ids) == len(set(ids)), provider

    def test_the_model_lookup_resolves_a_known_model_and_rejects_the_rest(self):
        spec = embedding_model_spec("openai", "text-embedding-3-small")
        assert spec is not None and spec["dimensions"] == 1536
        # A model id from the wrong provider must not resolve — dimensions would be wrong.
        assert embedding_model_spec("ollama", "text-embedding-3-small") is None
        assert embedding_model_spec("groq", "text-embedding-3-small") is None


class _RecordingHandler(logging.Handler):
    """Captures records off the `nexarag.llm` logger directly.

    Attached to the logger rather than relying on caplog because `configure_logging()`
    clears the root handlers at import time — and because rendering each record through
    the real JsonFormatter is what proves the line that would reach stdout is clean,
    extra_fields included.
    """

    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record):
        self.records.append(record)


@pytest.fixture()
def llm_log():
    logger = logging.getLogger("nexarag.llm")
    handler = _RecordingHandler()
    previous_level = logger.level
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    try:
        yield handler
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous_level)


def _rendered(handler: _RecordingHandler) -> list[str]:
    """Every captured record as the JSON line the app would actually emit."""
    formatter = JsonFormatter()
    return [formatter.format(record) for record in handler.records]


class TestKeysAreNeverLogged:
    """A leaked key in an application log is a credential disclosure to anyone with log
    access, and it is permanent — logs are shipped, aggregated and retained. Gemini makes
    this concrete: it authenticates with `?key=<secret>` in the URL, so any code path that
    logs a raw URL leaks. Both the success and the failure branch are checked, because the
    error path is the one that is tempted to log "everything we know" for debugging."""

    def test_a_successful_embed_logs_nothing_secret(self, db, monkeypatch, llm_log):
        _seed_credential(db, "openai")
        _capture_http(
            monkeypatch,
            lambda payload: {"data": [{"index": i, "embedding": [0.1, 0.2]} for i in range(len(payload["input"]))]},
        )

        embed_texts(db, USER, "openai", "text-embedding-3-small", ["alpha", "beta"])

        assert llm_log.records, "the success path must still log something observable"
        lines = _rendered(llm_log)
        assert all(SECRET_KEY_VALUE not in line for line in lines)
        # The passage text is not a secret in the same sense, but it is the user's document
        # and has no business in an operational log either.
        assert all("alpha" not in line for line in lines)
        # What it *should* say, so the assertions above cannot be satisfied by logging nothing.
        payload = json.loads(lines[-1])
        assert payload["provider"] == "openai"
        assert payload["model"] == "text-embedding-3-small"
        assert payload["passages"] == 2
        assert payload["dimensions"] == 2

    def test_a_failed_embed_logs_nothing_secret_even_for_url_authenticated_gemini(self, db, monkeypatch, llm_log):
        _seed_credential(db, "gemini")

        def responder(payload):
            # A realistic provider rejection: the body says nothing about the key itself.
            raise LlmProviderError("Provider API returned HTTP 401: API key not valid", 502)

        _capture_http(monkeypatch, responder)

        with pytest.raises(LlmProviderError):
            embed_texts(db, USER, "gemini", "text-embedding-004", ["alpha"])

        assert llm_log.records, "a failure must be logged, or an operator cannot see it"
        lines = _rendered(llm_log)
        assert all(SECRET_KEY_VALUE not in line for line in lines)
        assert all("key=" not in line for line in lines), "the Gemini URL carries the secret as a query parameter"
        payload = json.loads(lines[-1])
        assert payload["provider"] == "gemini"
        assert payload["status_code"] == 502
        assert "401" in payload["error"]

    def test_the_logged_error_is_bounded(self, db, monkeypatch, llm_log):
        """A provider that echoes a whole document back in its error message would
        otherwise write it, in full, into the application log."""
        _capture_http(monkeypatch, lambda payload: (_ for _ in ()).throw(LlmProviderError("E" * 5000, 502)))

        with pytest.raises(LlmProviderError):
            embed_texts(db, USER, "ollama", "nomic-embed-text", ["alpha"])

        # SQLite would not catch a length overflow, and neither would a log — assert directly.
        assert len(llm_log.records[-1].extra_fields["error"]) <= 300
