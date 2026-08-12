# Third-party notices — Nexarag-api

NexaRAG API is licensed under Apache-2.0 (see [LICENSE](LICENSE)).

**This repository redistributes no third-party code.** Dependencies are pinned in
`requirements.txt` and resolved from PyPI at install time; none is vendored into
the source tree. The Docker image built from [Dockerfile](Dockerfile) *does* embed
them, so this catalogue is what accompanies a redistributed image.

Verified against the on-disk virtualenv metadata on 2026-08-12.

---

## Runtime dependencies (`requirements.txt`)

| License | Packages |
|---|---|
| **MIT** | fastapi, SQLAlchemy, alembic, PyJWT, PyMySQL, passlib, python-dotenv, uvicorn, httptools, h11, anyio, sniffio, idna, Mako, click, annotated-types, dnspython, email-validator |
| **BSD-3-Clause** | starlette, pydantic, pydantic-core, cffi, watchfiles, MarkupSafe |
| **BSD (3-clause, modified wording)** | pypdf — Copyright (c) 2006-2008, Mathieu Fenniak. The third clause reads "the name of the author" rather than the standard organisation wording; quote it verbatim rather than substituting a canned BSD-3-Clause template. |
| **Apache-2.0** | bcrypt, python-multipart |
| **Apache-2.0 OR BSD-3-Clause** (dual — licensee's choice) | cryptography |
| **MIT OR Apache-2.0** (dual — licensee's choice) | uvloop |
| **MIT, with some files under PSF-2.0** | greenlet |
| **PSF-2.0** | typing_extensions — conditional on `python_version < "3.13"` |
| **GPL-2.0-or-later** | **mysqlclient** — see below |

## mysqlclient

`mysqlclient==2.2.7` ([requirements.txt:12](requirements.txt#L12)) is
**GPL-2.0-or-later**. Verified on disk: no linking exception, no FOSS exception,
not dual-licensed at the package level. (`MySQLdb/_mysql.c` alone carries a
permissive Comstar.net alternative; the Python wrapper modules do not, so the
package as a whole is GPL.)

**No obligation is currently triggered, for three independent reasons:**

1. **GPLv2 is distribution-triggered.** §0: *"Activities other than copying,
   distribution and modification are not covered by this License."* The API runs
   server-side and is never distributed to users. mysqlclient is GPL, **not**
   AGPL — there is no §13 network-use clause, so serving HTTP responses is not a
   triggering act.
2. **It cannot reach the distributed artifact.** The only thing this project
   distributes to end users is the SPA bundle, which is JavaScript. A compiled
   CPython extension has no path into it.
3. **It is not used.** No first-party module imports `MySQLdb`. Every
   `DATABASE_URL` this project writes uses the `mysql+pymysql` driver
   ([.env.example](.env.example), [docker-compose.yml](docker-compose.yml),
   [alembic.ini](alembic.ini), [scripts/lib/run_native.sh](scripts/lib/run_native.sh)),
   and `run_native.sh` already installs without mysqlclient when it fails to build.

**Recommended:** drop the pin. Apache-2.0 and GPLv2 are one-way incompatible —
GPLv2 cannot absorb Apache-2.0 code — so an Apache-2.0 project shipping a GPLv2
pin is a signal reviewers will stop on even though nothing here actually triggers.
`PyMySQL==1.1.1` (MIT) is already pinned and is what every configuration uses.

## Development dependencies (`requirements-dev.txt`)

Never installed into the Docker image — [Dockerfile](Dockerfile) installs only
`requirements.txt` and copies forward only that virtualenv. All obligations in
these licenses are conditioned on distribution, so a dev-only dependency imposes
nothing on the distributed product.

pytest, pytest-asyncio, pytest-cov, httpx, freezegun and their 13 transitive
dependencies are MIT, BSD-3-Clause, Apache-2.0 or PSF-2.0, with one weak
file-scoped copyleft (`certifi`, MPL-2.0). No GPL, LGPL, AGPL or SSPL.

---

## Google EmbeddingGemma (optional, operator-supplied)

Not a dependency and not redistributed. Reached through the operator's own Ollama
instance when a user opts into embeddings, per
[CLAUDE.md §22](../CLAUDE.md#22-per-document-indexing-settings). Two obligations
bind the **operator**, not this codebase:

- The **Gemma Prohibited Use Policy** binds *use* (Terms §3.2), and is an external
  document Google may change without notice.
- **"Distribution"** is defined to include making Gemma's functionality available
  as a hosted service. Running NexaRAG against your own documents redistributes
  nothing. Running it as a multi-tenant service that embeds other people's uploads
  is a question for a lawyer.

Gemma Outputs — the vectors stored in `document_chunks.embedding_json` — are
explicitly not Model Derivatives and Google claims no rights in them (§3.3).
