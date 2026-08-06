"""
Shared pytest configuration and fixtures for Nexarag-api.

CRITICAL ORDERING: environment variables are set at the very top, BEFORE importing any
application module. core/config.py, db/session.py and utils/token.py all read os.environ
at *import* time (utils/token.py even builds Fernet(SECRET_KEY) at import), so the values
must exist first.

By default the whole suite runs against a throwaway in-memory SQLite database, so unit
AND integration tests run anywhere with zero infrastructure. Set TEST_DATABASE_URL to
point the integration suite at a real MySQL (as the CI workflow does).
"""
import base64
import os

# A valid 32-byte Fernet key (also serves as the HS256 JWT secret).
os.environ.setdefault("SECRET_KEY", base64.urlsafe_b64encode(b"0" * 32).decode())
os.environ.setdefault("ENVIRONMENT", "test")
os.environ.setdefault("CORS_ORIGINS", "http://localhost:3000")

_TEST_DB_URL = os.environ.get("TEST_DATABASE_URL", "sqlite://")
os.environ["DATABASE_URL"] = _TEST_DB_URL

from core.config import get_settings  # noqa: E402

get_settings.cache_clear()

import pytest  # noqa: E402
from sqlalchemy import create_engine  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402

# Importing main registers every model on Base.metadata (via the router import chain)
# and builds the FastAPI app. It does NOT start the background loop — that only fires
# on the startup event, which we never trigger (TestClient is used without `with`).
import db.session as db_session  # noqa: E402
import main as app_main  # noqa: E402
from main import app  # noqa: E402

_IS_SQLITE = _TEST_DB_URL.startswith("sqlite")

if _IS_SQLITE:
    # One shared in-memory connection, reachable from the TestClient's portal thread.
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
else:
    engine = create_engine(_TEST_DB_URL, pool_pre_ping=True)

TestingSessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

# Re-point every module that captured the original engine/SessionLocal at import time
# so the app, the JWT middleware and get_db all talk to the test database.
db_session.engine = engine
db_session.SessionLocal = TestingSessionLocal
import controllers.chat_controller as _chat_controller  # noqa: E402
import routes.auth_middleware as _auth_mw  # noqa: E402

_auth_mw.SessionLocal = TestingSessionLocal
app_main.SessionLocal = TestingSessionLocal
# The streaming chat body outlives the request-scoped `db` dependency, so it persists the
# turn on its own session — which must also be the test session.
_chat_controller.SessionLocal = TestingSessionLocal

db_session.Base.metadata.create_all(bind=engine)


@pytest.fixture(scope="session")
def make_pdf():
    """Builds a minimal, valid, uncompressed multi-page PDF with a real text layer.

    A fixture rather than a shared module because `tests/` is not a package — there is no
    import path from one test directory to another, and the unit tests for page mapping and
    the integration tests for ingestion both need real PDF bytes.

    Object numbering: 1 catalog, 2 page tree, 3 font, then two objects per page (the page
    and its content stream). The page must declare a /Font resource with a standard
    encoding, or a reader cannot map character codes back to text.
    """

    def _build(pages: list[list[str]]) -> bytes:
        first_page_obj = 4
        page_numbers = [first_page_obj + index * 2 for index in range(len(pages))]
        kids = " ".join(f"{number} 0 R" for number in page_numbers)

        objects: list[bytes] = [
            b"<< /Type /Catalog /Pages 2 0 R >>",
            f"<< /Type /Pages /Kids [{kids}] /Count {len(pages)} >>".encode(),
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>",
        ]
        for index, lines in enumerate(pages):
            content = (
                "BT /F1 12 Tf 72 720 Td "
                + " ".join(f"({line}) Tj 0 -14 Td" for line in lines)
                + " ET"
            ).encode()
            content_obj = first_page_obj + index * 2 + 1
            objects.append(
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                f"/Resources << /Font << /F1 3 0 R >> >> /Contents {content_obj} 0 R >>".encode()
            )
            objects.append(
                b"<< /Length " + str(len(content)).encode() + b" >>\nstream\n" + content + b"\nendstream"
            )

        out = bytearray(b"%PDF-1.4\n")
        offsets = []
        for number, body in enumerate(objects, start=1):
            offsets.append(len(out))
            out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"

        xref_offset = len(out)
        out += f"xref\n0 {len(objects) + 1}\n".encode()
        out += b"0000000000 65535 f \n"
        for offset in offsets:
            out += f"{offset:010d} 00000 n \n".encode()
        out += (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF\n"
        ).encode()
        return bytes(out)

    return _build


@pytest.fixture()
def db():
    """A session for seeding and inspecting rows inside a test."""
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.rollback()
        session.close()


@pytest.fixture()
def client():
    """
    A TestClient with NO lifespan. Instantiating directly (not via `with`) means the
    startup event — which would spawn the LLM catalog refresh loop — never fires.
    raise_server_exceptions=False mirrors production: there is no global exception
    handler, so an uncaught error surfaces as a bare 500 response, not a test crash.
    """
    from fastapi.testclient import TestClient

    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture(autouse=True)
def _clean_tables():
    """Isolate tests: clear every table after each one."""
    yield
    with engine.begin() as conn:
        for table in reversed(db_session.Base.metadata.sorted_tables):
            conn.execute(table.delete())
