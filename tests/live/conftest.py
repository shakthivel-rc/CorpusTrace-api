"""Fixtures for tests that run against a *running* stack, not an in-process app.

Everything else in tests/ is hermetic: the integration suite drives a TestClient against
in-memory SQLite, and the SPA's Playwright specs intercept the API at the browser boundary.
That is the right default — it runs anywhere with no infrastructure — but it means nothing
exercises the system as it is actually deployed, and three real defects reached a browser
through that gap in a single day:

  * GET /users/profile returned 500 for the seeded superadmin, because a response model
    re-validated a stored address whose TLD email-validator rejects.
  * nginx served the pdf.js worker as application/octet-stream, so no citation could open
    its source document. The file was present and returned 200 the whole time.
  * The SPA container reported unhealthy forever while serving every request correctly.

None of those are reachable from SQLite and a mocked fetch. They live in the wiring — MySQL,
nginx, the container healthchecks, the /api prefix strip — so this suite talks to that wiring
over HTTP exactly as a browser does.

These tests are SKIPPED unless NEXARAG_LIVE_URL is set, so `pytest` stays hermetic by default:

    NEXARAG_LIVE_URL=http://localhost:8080 pytest -m live

Point it at the SPA's origin, not the API's — that routes through nginx and therefore covers
the proxy and its prefix strip. The API's own origin works too if you only want the backend.
"""
import os
import uuid

import httpx
import pytest

# The seeder's default. Overridable for a stack seeded with a different account.
SUPERADMIN_EMAIL = os.getenv("NEXARAG_LIVE_EMAIL", "superadmin@nexarag.local")


def pytest_collection_modifyitems(items):
    """Mark everything under tests/live as `live`, rather than trusting each module to.

    A `pytestmark` in this file would not propagate to the test modules beside it, and a
    module that forgets the marker would run during a plain `pytest -m "not live"` and fail
    against a stack that is not there. Applying it at collection removes the chance.
    """
    here = os.path.dirname(__file__)
    for item in items:
        if str(item.fspath).startswith(here):
            item.add_marker(pytest.mark.live)


def _require_live_url() -> str:
    url = os.getenv("NEXARAG_LIVE_URL", "").strip().rstrip("/")
    if not url:
        pytest.skip("set NEXARAG_LIVE_URL to run the live-stack suite")
    return url


@pytest.fixture(scope="session")
def live_root() -> str:
    """The origin a browser would use — the SPA behind nginx."""
    return _require_live_url()


@pytest.fixture(scope="session")
def api_base(live_root: str) -> str:
    """Where API calls go.

    Against the SPA origin every call is prefixed `/api`, which nginx strips before
    proxying. Against the API's own origin there is no prefix. Detect rather than
    configure: ask the health endpoint, which exists on both and is unauthenticated.
    """
    for candidate in (f"{live_root}/api", live_root):
        try:
            probe = httpx.get(f"{candidate}/api/v1/health/ready", timeout=10)
        except httpx.HTTPError:
            continue
        if probe.status_code == 200:
            return candidate
    pytest.skip(f"no NexaRAG API reachable at {live_root}")


@pytest.fixture(scope="session")
def superadmin_password() -> str:
    password = os.getenv("SUPERADMIN_PASSWORD", "").strip()
    if not password:
        pytest.skip(
            "set SUPERADMIN_PASSWORD (the value in the stack's .env) to run authenticated "
            "live tests"
        )
    return password


@pytest.fixture(scope="session")
def admin_token(api_base: str, superadmin_password: str) -> str:
    """A real access token from a real login against a real database."""
    response = httpx.post(
        f"{api_base}/user/login",
        json={"username": SUPERADMIN_EMAIL, "password": superadmin_password},
        timeout=30,
    )
    if response.status_code != 200:
        pytest.skip(f"could not sign in as {SUPERADMIN_EMAIL}: HTTP {response.status_code}")
    token = (response.json().get("data") or {}).get("access_token")
    if not token:
        pytest.skip("login succeeded but returned no access_token")
    return token


@pytest.fixture()
def client(api_base: str) -> httpx.Client:
    """Unauthenticated client."""
    with httpx.Client(base_url=api_base, timeout=30, follow_redirects=False) as http:
        yield http


@pytest.fixture()
def admin(api_base: str, admin_token: str) -> httpx.Client:
    """Client carrying the superadmin bearer token."""
    with httpx.Client(
        base_url=api_base,
        timeout=60,
        follow_redirects=False,
        headers={"Authorization": f"Bearer {admin_token}"},
    ) as http:
        yield http


@pytest.fixture()
def unique() -> str:
    """A short suffix so a test that creates data cannot collide with another run.

    The live stack keeps its database between runs, unlike the hermetic suite whose tables
    are truncated per test. Anything created here has to be namespaced or cleaned up.
    """
    return uuid.uuid4().hex[:10]


def assert_envelope(response: httpx.Response, expected_status: int = 200) -> dict:
    """Every endpoint answers {status, status_code, message, data}. Assert it, and return data.

    The envelope is the contract the whole SPA is written against and there is no OpenAPI
    schema to catch a break, so checking the shape is as valuable as checking the payload.
    """
    assert response.status_code == expected_status, (
        f"{response.request.method} {response.request.url} -> {response.status_code}\n"
        f"{response.text[:400]}"
    )
    body = response.json()
    for key in ("status", "status_code", "message"):
        assert key in body, f"envelope missing {key!r}: {body}"
    assert body["status_code"] == expected_status
    return body.get("data")
