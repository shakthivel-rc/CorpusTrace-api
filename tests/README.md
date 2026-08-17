# CorpusTrace API test suite

Automated tests for the FastAPI backend. Built on **pytest + httpx (TestClient) +
pytest-cov + freezegun**, configured in [`../pyproject.toml`](../pyproject.toml) and
[`conftest.py`](conftest.py).

## How to run

```bash
cd CorpusTrace-api
python -m venv .venv && source .venv/bin/activate      # or use the repo-root env/
pip install -r requirements.txt -r requirements-dev.txt

pytest -m unit          # pure/unit tests only — no database, runs anywhere
pytest                  # full suite (unit + integration)
pytest --cov            # with coverage (term + configured reports)
```

By default the suite runs against a throwaway **in-memory SQLite** database, so it
needs zero infrastructure. Set `TEST_DATABASE_URL` to run the integration suite against
real MySQL (the CI workflow does exactly this):

```bash
TEST_DATABASE_URL=mysql+pymysql://root:root@127.0.0.1:3306/corpustrace_test pytest
```

## The harness ([conftest.py](conftest.py))

The one non-obvious piece. Environment variables are set **before any app import**,
because `core/config.py`, `db/session.py` and `utils/token.py` all read `os.environ` at
import time (`utils/token.py` even builds `Fernet(SECRET_KEY)` at import — so the test
`SECRET_KEY` is a valid Fernet key). The test engine (SQLite `StaticPool`, or MySQL from
`TEST_DATABASE_URL`) is then swapped into `db.session`, the JWT middleware and `main`, and
`Base.metadata.create_all` builds the schema. The `TestClient` is used **without** a
`with` block so the startup event — which would spawn the LLM catalog refresh loop —
never fires.

## What is covered

| Area | File | Why it's a priority |
|---|---|---|
| Password policy | `unit/test_password.py` | Security rule; mirrors the frontend validator |
| RBAC resolution | `unit/test_permissions.py` | 3-branch resolver; a bug = silent privilege change |
| Config helpers + prod CORS guard | `unit/test_config.py` | Startup-critical parsing |
| Fernet token crypto | `unit/test_token.py` | Credential + reset-link encryption |
| RAG chunker | `unit/test_rag_chunking.py` | Boundary-dense, dependency-free |
| RAG scoring / tokenizer | `unit/test_rag_scoring.py` | The **only** relevance mechanism (no vector store) |
| RAG graph helper | `unit/test_rag_graph.py` | Pins `entity_chunks_by_key`'s real contract |
| Health endpoints | `integration/test_health.py` | The only machine-checkable assertions |
| JWT middleware | `integration/test_auth_middleware.py` | Allow-list, decode, session lookup, revocation |
| Login + lockout | `integration/test_auth_login.py` | Lockout arithmetic (4×401 then 423), counter reset |
| Route authorization | `integration/test_permissions_endpoint.py` | 403 when the slug is missing |
| Migration chain | `integration/test_migrations.py` | up → down → up (MySQL only, see below) |

The tested surface — the pure business logic plus the auth/health request paths — sits at
**~100% coverage**. The global coverage number is lower (see the ratchet floor in
`pyproject.toml`) because of the intentionally-untested subsystems below.

## What is intentionally NOT covered (and why)

- **The RAG engine internals** (`rag/service.py` ingestion, the six retrieval modes,
  graph build): large. Both historical upload-500s **have been fixed** and are regression-tested
  in `test_rag_graph.py` (the `_rebuild_graph` argument bug, and unbounded strings overflowing
  `VARCHAR(255)`); document text extraction is covered by `test_rag_extraction.py`; the
  LLM-failure fallback in `answer_question` by `test_rag_llm_fallback.py`; small talk and
  out-of-scope routing by `test_rag_conversation.py`; and the guard-railed conversational
  LLM path — prompt rules, reply sanitisation, evidence gating and provider-response
  robustness — by `test_rag_guarded_llm.py`. The six retrieval modes themselves remain
  untested beyond the chunking/scoring units.
  **Caveat: the suite runs on SQLite, which ignores VARCHAR limits** — a length overflow that
  would be a MySQL `DataError` (and an unhandled 500) cannot be caught through the database
  here, so those tests assert lengths directly.
- **The LLM provider layer** (`services/llm_provider.py`): live HTTP calls are not made in
  tests; `test_llm_provider.py` covers the registry contract, `api_style` dispatch, the
  Cloudflare model-listing shim, generation-parameter plumbing and the connectivity test by
  monkeypatching `http_json`. The real wire behaviour of each provider is only exercised
  manually.
- **SMTP / email** (`integrations/aws_ses.py`): no fake SMTP server wired up yet.
- **The migration round-trip on SQLite**: the Alembic revisions contain MySQL-specific
  operations, so `test_migrations.py` is **skipped on SQLite**. It used to run in CI against
  a MySQL service container; with CI removed (CLAUDE.md §25) it runs **nowhere** unless you
  point `TEST_DATABASE_URL` at a real MySQL yourself:

  ```bash
  TEST_DATABASE_URL=mysql+pymysql://root:root@127.0.0.1:3306/corpustrace_test pytest
  ```

  This is the one gap that matters, because SQLite also ignores `VARCHAR` limits — the
  overflow MySQL raises as error 1406 passes here silently.

## Reports

`pytest --cov` writes `coverage.xml`, `htmlcov/` and `reports/pytest-junit.xml`. Nothing
collects them automatically any more; read them locally.
