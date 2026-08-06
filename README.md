# Nexarag-api

FastAPI backend for **NexaRAG** — an authenticated service for uploading documents into a
knowledge base and asking questions against them, plus a full identity and access-control
console.

Pairs with the [`nexarag-app`](https://github.com/SHAKTHI-HACKER/nexarag-app) React SPA.

---

## What it does

| Area | Detail |
|---|---|
| **Identity** | Login by email *or* username, per-account lockout, revocable server-side sessions, rotating refresh tokens, OTP email verification, invite and password-reset links |
| **Authorization** | Role-based. Permissions attach to roles, resolved through a three-branch check that also honours legacy permission implications |
| **Documents** | Upload PDF, DOCX and XLSX into a "brain" (knowledge base), then ask questions and get cited answers |
| **LLM providers** | 11 providers behind one metadata-driven abstraction; credentials encrypted at rest and only ever returned masked |
| **Auditing** | Activity log of user and admin actions |

## Stack

Python 3.12 · FastAPI 0.115 · SQLAlchemy 2.0 · Alembic · MySQL · PyJWT · passlib/bcrypt ·
cryptography (Fernet) · pypdf

Outbound HTTP uses the standard library (`urllib.request`) and email uses `smtplib` — there
are no vendor SDKs and no HTTP client dependency.

## How retrieval actually works

Stated plainly, because the name invites the wrong assumption: **there is no vector store and no
vector database.** The baseline is lexical. Documents are split into chunks — by default
overlapping 1200-character windows preferring sentence boundaries — and each chunk stores a
bag-of-words term-frequency map. Retrieval scores **lexical term overlap** against that map across
six selectable strategies.

Since 2026-08-05 each document also chooses **how** it is indexed (four chunking strategies, its own
size and overlap) and may **optionally** carry embeddings from OpenAI, Gemini, Ollama or Mistral.
Those are stored as a JSON array of floats beside the term map, cosine-scored in Python and fused
into the lexical ranking. They are opt-in per document and off by default, because turning them on
sends that document's text to a third party and usually costs money.

The practical consequence, unless you turn embeddings on: a question sharing no vocabulary with your
documents returns nothing. There is also **no OCR**, so a scanned image-only PDF has no text layer
to extract.

Every mode is gated on whether the retrieved evidence actually supports an answer. When it
doesn't — or when the message is small talk or asks about real-time information — the reply
comes from a guard-railed conversational path that may greet and explain but must never
answer from model knowledge or invent document contents. Citation markers are stripped in
code, not merely forbidden by prompt.

## Quick start

**One command, from a clean machine with Docker:**

```bash
git clone https://github.com/SHAKTHI-HACKER/Nexarag-api.git && cd Nexarag-api && make setup
```

That clones the SPA beside this repo, writes a `.env` with freshly generated secrets, brings up
MySQL, runs the migrations, seeds a superadmin, and starts everything. When it finishes it prints
the URLs — the app on <http://localhost:8080>, the API on <http://localhost:8000>.

Docker is the only prerequisite. It is used because this half of the project needs a MySQL server,
a schema, a seeded account and a secret key before it will start at all, and installing MySQL is
the single largest obstacle between a clone and a running app.

Re-running `make setup` is safe: it fills in only what is missing and never overwrites a value you
already have in `.env`.

### Without Docker

```bash
make setup-native     # venv + pip + npm + migrations + seed, against your own MySQL
make dev              # both dev servers with hot reload
```

Native mode needs Python 3.12, Node, and a MySQL server you point `DATABASE_URL` at.
`mysqlclient` builds from source, so it also needs `libmysqlclient-dev` and `pkg-config`.

### Everything else

`make` on its own lists the rest — `make logs`, `make db-shell`, `make test`, `make reset`.

> Always start the API from this directory. `load_dotenv()` reads the *current working directory*
> and settings are read at **import** time, so a missing required value raises before uvicorn
> binds. The container sets `WORKDIR` for exactly this reason.

See [`.env.example`](.env.example) for every variable. Note that `SECRET_KEY` does three jobs — JWT
signing, reset-link encryption, and the fallback LLM-credential key — so set
`LLM_CREDENTIAL_ENCRYPTION_KEY` explicitly if you ever intend to rotate it. Rotating `SECRET_KEY`
without it makes every stored provider credential undecryptable.

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest              # 808 passed, 1 skipped
python -m pytest -m unit      # no database required
```

Runs against in-memory SQLite by default, so it needs no infrastructure. Set
`TEST_DATABASE_URL` to point the integration suite at MySQL, as CI does.

Two caveats worth knowing: use `python -m pytest`, not bare `pytest` — the project root isn't
on `sys.path` otherwise. And SQLite ignores `VARCHAR` limits, so a length overflow that would
be a MySQL `DataError` can't be caught through the database here; those tests assert lengths
directly.

## Layout

```
main.py          app assembly, middleware order, router registration
core/            settings, JSON logging, middleware, permission catalog
routes/          9 APIRouters, 55 endpoints
controllers/     business logic; owns the transaction boundary
services/        query helpers, LLM provider layer, activity log
rag/chunking.py  the four chunking strategies and the per-mode recommendations
rag/service.py   ingestion, retrieval, the 6 modes
rag/jobs.py      the background indexing worker
models/          SQLAlchemy declarative models
schemas/         Pydantic request bodies + the response envelope
alembic/         migrations
tests/           pytest suite
```

Layering runs one direction only: `routes → controllers → services/rag → models`. Controllers
own `commit()`/`rollback()`; services `flush()` but do not commit.

Every endpoint returns the same envelope:

```json
{ "status": "...", "status_code": 200, "message": "...", "data": {} }
```

There is no global exception handler, so each route handles its own error paths.

## API docs

OpenAPI, Swagger and ReDoc are disabled in all environments (`docs_url=None`,
`redoc_url=None`, `openapi_url=None`), so there is no machine-readable contract — read the
route modules.
