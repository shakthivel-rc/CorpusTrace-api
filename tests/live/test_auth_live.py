"""Authentication and session handling, exercised over real HTTP against real MySQL.

The hermetic suite already covers the login *arithmetic* — lockout counting, the counter
reset, the middleware's four rejection branches — against in-memory SQLite. What it cannot
cover is the half of authentication that only exists once the system is deployed:

  * `user_sessions` is a real MySQL table with a real `VARCHAR(1024)` access-token column,
    and every authenticated request matches a token against it literally. A token that is
    truncated or whitespace-mangled anywhere between the SPA, nginx and the API produces a
    401 that looks exactly like a wrong password, and SQLite's indifference to column
    lengths means the hermetic suite will never notice.
  * The `Authorization` header has to survive nginx's proxy pass. A proxy that drops or
    rewrites it turns every authenticated screen into a login loop while the API itself is
    perfectly healthy.
  * `/api/v1/health/*` is allow-listed by *prefix* in the middleware, and nginx *strips* an
    `/api` prefix of its own. Those two facts interact, and getting the interaction wrong
    means the readiness probe either 401s (container restarts forever) or the allow-list
    accidentally opens more than health.
  * Refresh rotates the access token **in place on the same row**. Whether the old token is
    genuinely dead afterwards is a property of a committed UPDATE on a shared connection —
    exactly the kind of thing that works in a single-session test and not in production.

Concurrency notes, because this stack is shared:

  * Nothing here calls logout on the shared `admin` fixture token. Every session this module
    revokes is one it opened itself, and logout deletes only the row matching the presented
    access token (`logout_controller`), so revoking ours cannot touch anyone else's.
  * The one test that submits a wrong password is bracketed by a successful login on both
    sides. `failed_login_attempts` resets *only* on a successful login (never by the window
    lapsing), so an unbracketed failure would be left on the account for the next agent to
    trip over, five of them in a row locking the shared superadmin for 15 minutes.
  * No test asserts a global count, and everything created is namespaced or revoked.
"""
import time
import uuid

import httpx
import jwt
import pytest
from conftest import SUPERADMIN_EMAIL, assert_envelope

# Deliberately a literal, never `superadmin_password + "x"`. A derived value would put the
# real secret one string operation away from a pytest assertion dump.
WRONG_PASSWORD = "Definitely-Not-The-Password-9!"

# Used wherever a request is expected to fail *schema* validation. FastAPI's 422 body echoes
# the submitted input verbatim (see the module summary of this suite's findings), so the only
# safe password to send into a 422 is one that is not a password.
PLACEHOLDER_PASSWORD = "placeholder-not-a-real-secret"

PROTECTED_PATH = "/users/profile"
HEALTH_PATHS = ("/api/v1/health/live", "/api/v1/health/ready", "/api/v1/health/dependencies")


def _login(client: httpx.Client, password: str, username: str = SUPERADMIN_EMAIL):
    return client.post("/user/login", json={"username": username, "password": password})


def _profile(client: httpx.Client, token: str):
    return client.get(PROTECTED_PATH, headers={"Authorization": f"Bearer {token}"})


@pytest.fixture()
def revoke_later(api_base: str):
    """Collect access tokens this test opened, and revoke each one at teardown.

    Sessions are rows, and rows on a live stack outlive the test that made them. Logging out
    hard-deletes the row (`db.delete(session)`), so this keeps `user_sessions` from growing a
    little every time the suite runs — which matters more than usual here, because
    `user_sessions.access_token` is unindexed and every authenticated request scans it.

    Failures are swallowed on purpose: a token that a test already revoked, or that a refresh
    rotated out from under us, answers 400/401 and that is a successful cleanup, not an error.
    """
    tokens: list[str] = []
    yield tokens.append
    for token in tokens:
        try:
            httpx.post(
                f"{api_base}/user/logout",
                headers={"Authorization": f"Bearer {token}"},
                timeout=30,
            )
        except httpx.HTTPError:
            pass


@pytest.fixture()
def open_session(client: httpx.Client, superadmin_password: str, revoke_later):
    """Open a *separate* login session and guarantee it is revoked afterwards.

    Every destructive session test (refresh rotation, logout) has to run against a session it
    owns. The shared `admin` fixture's token is in use by the rest of this run and, on this
    stack, by other agents' runs; rotating or revoking it would fail their tests with a 401
    that looks like a product defect.
    """

    def _open() -> dict:
        data = assert_envelope(_login(client, superadmin_password))
        revoke_later(data["access_token"])
        return data

    return _open


# --------------------------------------------------------------------------------------
# Login
# --------------------------------------------------------------------------------------


def test_login_returns_the_envelope_and_a_usable_token_pair(
    client: httpx.Client, superadmin_password: str, revoke_later
):
    """A correct login is the one call the whole SPA cannot recover from getting wrong.

    Asserts the full contract rather than just the status: the `{status, status_code,
    message, data}` envelope, both tokens present and non-empty, and — the part only a live
    stack can prove — that the access token is immediately accepted by a protected endpoint.
    A token that is issued but not persisted to `user_sessions`, or persisted truncated by
    the VARCHAR(1024) column, returns 200 here and 401 on the very next request.
    """
    response = _login(client, superadmin_password)
    data = assert_envelope(response)
    assert response.json()["status"] == "success"

    access_token = data["access_token"]
    refresh_token = data["refresh_token"]
    revoke_later(access_token)
    assert isinstance(access_token, str) and access_token
    assert isinstance(refresh_token, str) and refresh_token
    assert access_token != refresh_token

    assert_envelope(_profile(client, access_token))


def test_login_does_not_return_the_stored_password_hash(
    client: httpx.Client, superadmin_password: str, revoke_later
):
    """The login payload is exactly two tokens, and must never grow into a user record.

    `login_controller` builds its `data` dict by hand today, but the neighbouring controllers
    return ORM objects straight into `SuccessResponse`, so widening this one to "return the
    user too" is a one-line change somebody will eventually make. The bcrypt hash is the
    thing that must not ride along.
    """
    data = assert_envelope(_login(client, superadmin_password))
    revoke_later(data["access_token"])

    assert set(data) == {"access_token", "refresh_token"}
    serialized = repr(data)
    assert "$2b$" not in serialized and "$2a$" not in serialized


def test_the_issued_tokens_are_distinctly_typed_access_and_refresh(
    client: httpx.Client, superadmin_password: str, revoke_later
):
    """`type` is what stops a refresh token being replayed as an access token.

    The middleware rejects anything whose `type` is not `access`, and the refresh controller
    rejects anything whose `type` is not `refresh`. Both checks are worthless if the two
    tokens are minted with the same claims, and since they are signed with the same key and
    have the same `sub`, the *only* thing separating them is this claim. Decoded without
    verification on purpose — the test must not need the signing secret.
    """
    data = assert_envelope(_login(client, superadmin_password))
    revoke_later(data["access_token"])

    access = jwt.decode(data["access_token"], options={"verify_signature": False})
    refresh = jwt.decode(data["refresh_token"], options={"verify_signature": False})

    assert access["type"] == "access"
    assert refresh["type"] == "refresh"
    assert access["sub"] == refresh["sub"]
    assert access["jti"] != refresh["jti"]
    # The refresh token has to outlive the access token or rotation is pointless.
    assert refresh["exp"] > access["exp"]


def test_a_wrong_password_is_a_401_envelope_not_a_bare_500(
    client: httpx.Client, superadmin_password: str, revoke_later
):
    """A rejected login must come back as the envelope, because there is no fallback.

    This backend installs no global exception handler, so anything that raises inside a
    controller reaches the client as bare plain-text HTTP 500 that the SPA's `api` wrapper
    cannot parse into a message. The failure path of login is the single most-hit error path
    in the product, and it is also the one that writes to the database (incrementing
    `failed_login_attempts`) before responding — so it is a genuine candidate for a 500 that
    a read-only error path would never hit.

    The wrong attempt is deliberately bracketed by successful logins. The counter resets only
    on success, so leaving one failure behind would hand the next agent a superadmin that is
    one step closer to a 15-minute lockout; the trailing login in the `finally` puts it back
    to zero even when the assertions blow up.
    """
    revoke_later(assert_envelope(_login(client, superadmin_password))["access_token"])
    try:
        rejected = _login(client, WRONG_PASSWORD)
        assert rejected.status_code == 401, (
            "a wrong password must be a 401, not "
            f"{rejected.status_code} (423 would mean the account is locked)"
        )
        assert_envelope(rejected, 401)
        assert rejected.json()["status"] == "error"
        assert rejected.json()["message"] == "Invalid username or password"
    finally:
        recovery = _login(client, superadmin_password)
        assert recovery.status_code == 200, (
            "the bracketing login that resets failed_login_attempts did not succeed; the "
            "shared superadmin may be left one failed attempt closer to a lockout"
        )
        revoke_later(recovery.json()["data"]["access_token"])


def test_an_unknown_username_is_rejected_without_revealing_whether_it_exists(
    client: httpx.Client
):
    """Login must not be a user-enumeration oracle.

    An unknown account and a known account with the wrong password take completely different
    code paths — one returns before any password work, the other runs bcrypt and writes a
    counter — and it costs one careless edit to make them say different things. The response
    a client can see must be indistinguishable: same status, same message. Asserted against
    the literal wording of the wrong-password branch above, so that changing either message
    in isolation fails here.

    Uses a random address so it can never collide with a real account, on this stack or on a
    developer's.
    """
    unknown = f"no-such-user-{uuid.uuid4().hex[:12]}@nexarag.invalid"
    response = _login(client, WRONG_PASSWORD, username=unknown)

    assert response.status_code == 401
    assert_envelope(response, 401)
    body = response.json()
    assert body["status"] == "error"
    assert body["message"] == "Invalid username or password"
    assert unknown not in response.text


def test_the_login_field_is_username_and_it_accepts_an_email_value(
    client: httpx.Client, superadmin_password: str, revoke_later
):
    """`username` accepts an email; there is no `email` field, and sending one is a 422.

    The seeded superadmin's username *is* its email address, and `login_controller` looks up
    by email first and username second — so the field name says one thing and the accepted
    values are broader. Any client written against the obvious-looking `{"email": ...}` shape
    fails schema validation before reaching a controller, which is a very confusing 422 to
    debug from the browser. Pinning both halves keeps the request schema honest.

    The invalid request carries a placeholder password, never the real one: FastAPI's 422 body
    echoes the submitted input.
    """
    accepted = _login(client, superadmin_password, username=SUPERADMIN_EMAIL)
    revoke_later(assert_envelope(accepted)["access_token"])

    wrong_shape = client.post(
        "/user/login",
        json={"email": SUPERADMIN_EMAIL, "password": PLACEHOLDER_PASSWORD},
    )
    assert wrong_shape.status_code == 422


def test_login_without_a_password_is_a_422_and_not_a_500(client: httpx.Client):
    """A body missing a required field must fail validation, not reach bcrypt with a None.

    `verify_password(None, hash)` raises inside passlib, and with no global exception handler
    that would be a bare 500 on an unauthenticated endpoint — reachable by anyone who can
    reach the login form.
    """
    response = client.post("/user/login", json={"username": SUPERADMIN_EMAIL})
    assert response.status_code == 422


# --------------------------------------------------------------------------------------
# The middleware: what a protected endpoint does with a bad or missing token
# --------------------------------------------------------------------------------------


def test_a_protected_endpoint_without_a_token_is_401(client: httpx.Client):
    """No credentials is an authentication failure, and must be the envelope.

    `GET /users/profile` is the first call the SPA makes after a page load, so this is the
    response every expired-session user meets. If it were a 500 the SPA could not tell
    "you are logged out" from "the backend is broken", and the 401-refresh-retry path in
    `api/client.ts` — which keys off exactly this status — would never fire.
    """
    response = client.get(PROTECTED_PATH)
    assert response.status_code == 401
    assert_envelope(response, 401)
    assert response.json()["message"] == "Missing or invalid token"


def test_a_garbage_bearer_token_is_401(client: httpx.Client):
    """A string that is not a JWT must be refused by the decode, not crash it.

    `jwt.decode` raises `DecodeError` on anything unparseable; the middleware catches
    `InvalidTokenError` (its base class). A stray token in `localStorage`, a truncated copy
    or a half-written value has to land on the 401 branch, because the alternative is an
    unhandled exception on every request the browser makes until storage is cleared by hand.
    """
    response = _profile(client, "not-a-real-jwt")
    assert response.status_code == 401
    assert_envelope(response, 401)
    assert response.json()["message"] == "Invalid token"


def test_a_token_signed_with_the_wrong_secret_is_401(client: httpx.Client):
    """A well-formed JWT the server did not sign must be refused on the signature.

    This is the forgery case, and it is the one that is *not* obviously wrong on inspection:
    the token has the right algorithm, the right claim names, an unexpired `exp` and a `sub`
    that could name a real user. Only the signature separates it from a real one. If
    verification were ever relaxed — `options={"verify_signature": False}`, or an `algorithms`
    list that admitted `none` — every other check in the middleware would still pass and this
    request would be authenticated as whoever `sub` names.
    """
    forged = jwt.encode(
        {
            "sub": "00000000-0000-0000-0000-000000000000",
            "type": "access",
            "exp": int(time.time()) + 3600,
        },
        f"not-the-signing-key-{uuid.uuid4().hex}",
        algorithm="HS256",
    )
    response = _profile(client, forged)
    assert response.status_code == 401
    assert_envelope(response, 401)
    assert response.json()["message"] == "Invalid token"


def test_a_non_bearer_authorization_scheme_is_401(client: httpx.Client):
    """Only `Bearer` authenticates. Another scheme is missing credentials, not a crash.

    The middleware splits the header on a space and indexes `[1]`, so a header that is not
    `Bearer <token>` has to be rejected by the `startswith` guard *before* the split — a
    scheme with no value would otherwise be an `IndexError` and a bare 500 on an
    unauthenticated-reachable path.
    """
    response = client.get(PROTECTED_PATH, headers={"Authorization": "Basic dXNlcjpwYXNz"})
    assert response.status_code == 401
    assert_envelope(response, 401)
    assert response.json()["message"] == "Missing or invalid token"


def test_a_refresh_token_cannot_be_used_as_an_access_token(
    client: httpx.Client, open_session
):
    """`type` is checked, not assumed — a real, correctly signed refresh token is still a 401.

    The refresh token is signed with the same key as the access token and names the same
    user, so signature verification alone accepts it happily. It also lives for 30 days
    instead of one hour, so treating it as an access token would silently hand out a
    month-long session. The claim check is the only thing standing in the way, and it is one
    line (`payload.get("type", "access") != "access"`) whose default makes it easy to
    neutralise by accident.
    """
    session = open_session()
    response = _profile(client, session["refresh_token"])
    assert response.status_code == 401
    assert_envelope(response, 401)
    assert response.json()["message"] == "Invalid token type"


# --------------------------------------------------------------------------------------
# Refresh and revocation
# --------------------------------------------------------------------------------------


def test_refresh_returns_a_new_pair_and_the_new_access_token_works(
    client: httpx.Client, open_session, revoke_later
):
    """Rotation has to produce a token that is actually usable, not merely well-formed.

    Refresh rewrites `user_sessions.access_token` in place and commits. The new token is only
    good if that UPDATE reached MySQL *and* the row is then found by the middleware's literal
    `WHERE access_token = ?` on a different connection. Nothing about that is provable in
    process, and the failure mode — a fresh token that 401s — is indistinguishable from
    "your session expired", so users would just be logged out at random.

    Runs on its own session so the shared `admin` token is not rotated out from under the
    rest of the run.
    """
    session = open_session()

    response = client.post("/user/refresh", json={"refresh_token": session["refresh_token"]})
    rotated = assert_envelope(response)
    revoke_later(rotated["access_token"])

    assert rotated["access_token"] and rotated["refresh_token"]
    assert rotated["access_token"] != session["access_token"]
    assert rotated["refresh_token"] != session["refresh_token"]

    profile = assert_envelope(_profile(client, rotated["access_token"]))
    assert profile["email"] == SUPERADMIN_EMAIL


def test_refresh_retires_the_previous_access_and_refresh_tokens(
    client: httpx.Client, open_session, revoke_later
):
    """Rotation is only a security property if the old tokens die.

    "Rotation" that leaves the previous access token working is just token minting: a leaked
    token stays valid for its full hour no matter how often the legitimate client refreshes.
    And because the session row is updated *in place*, the old refresh token's hash no longer
    matches anything, which is what makes a replayed refresh token fail. Both halves are
    asserted here because each is a separate line in the controller.
    """
    session = open_session()
    rotated = assert_envelope(
        client.post("/user/refresh", json={"refresh_token": session["refresh_token"]})
    )
    revoke_later(rotated["access_token"])

    stale_access = _profile(client, session["access_token"])
    assert stale_access.status_code == 401
    assert_envelope(stale_access, 401)
    assert stale_access.json()["message"] == "Invalid session or token revoked"

    replayed = client.post("/user/refresh", json={"refresh_token": session["refresh_token"]})
    assert replayed.status_code == 401
    assert_envelope(replayed, 401)


@pytest.mark.parametrize(
    "token, expected_message",
    [
        ("not-a-jwt-at-all", "Invalid refresh token"),
        ("", "Invalid refresh token"),
    ],
    ids=["garbage", "empty"],
)
def test_refresh_rejects_an_unparseable_token_with_the_envelope(
    client: httpx.Client, token: str, expected_message: str
):
    """`POST /user/refresh` is unauthenticated, so its error paths are public attack surface.

    It is in `UNAUTHENTICATED_ROUTES`, which means the middleware never looks at the body and
    the controller is the first thing to touch attacker-supplied text — it calls `jwt.decode`
    on it directly. Anything that escapes the `InvalidTokenError` catch is a bare 500 that
    anyone on the internet can trigger with a POST.
    """
    response = client.post("/user/refresh", json={"refresh_token": token})
    assert response.status_code == 401
    assert_envelope(response, 401)
    assert response.json()["message"] == expected_message


def test_refresh_rejects_a_forged_token_and_an_access_token(
    client: httpx.Client, open_session
):
    """The two ways to hold a plausible refresh token you should not be able to spend.

    A forged token has the right shape and the right `type` but the wrong signature; an
    access token has the right signature but the wrong `type`. They fail at different lines
    (`jwt.decode` versus the explicit `type` check) and both must be 401s carrying the
    envelope. The second case matters most: an access token is the credential most likely to
    be pasted into the wrong call, and accepting it would let an hour-long token be traded
    for a fresh 30-day one indefinitely.
    """
    forged = jwt.encode(
        {"sub": "nobody", "type": "refresh", "exp": int(time.time()) + 3600},
        f"not-the-signing-key-{uuid.uuid4().hex}",
        algorithm="HS256",
    )
    response = client.post("/user/refresh", json={"refresh_token": forged})
    assert response.status_code == 401
    assert_envelope(response, 401)
    assert response.json()["message"] == "Invalid refresh token"

    session = open_session()
    wrong_type = client.post("/user/refresh", json={"refresh_token": session["access_token"]})
    assert wrong_type.status_code == 401
    assert_envelope(wrong_type, 401)
    assert wrong_type.json()["message"] == "Invalid token type"


def test_logout_revokes_only_the_session_that_presented_the_token(
    client: httpx.Client, open_session
):
    """Signing out of one device must not sign the account out everywhere.

    `logout_controller` deletes the single `user_sessions` row whose `access_token` matches
    the presented one. Widening that filter to `user_id` — an entirely reasonable-looking
    edit — would log the user out of every browser they own, and on this shared stack would
    break every other agent's superadmin session at once. Two independent sessions are opened
    here precisely so the blast radius is measured rather than assumed.

    Both sessions are ones this test created; the shared `admin` fixture token is never
    revoked by this suite.
    """
    first = open_session()
    second = open_session()
    assert first["access_token"] != second["access_token"]

    assert_envelope(_profile(client, first["access_token"]))
    assert_envelope(_profile(client, second["access_token"]))

    logged_out = client.post(
        "/user/logout", headers={"Authorization": f"Bearer {first['access_token']}"}
    )
    assert_envelope(logged_out)

    revoked = _profile(client, first["access_token"])
    assert revoked.status_code == 401
    assert_envelope(revoked, 401)
    assert revoked.json()["message"] == "Invalid session or token revoked"

    # The other device is untouched.
    assert_envelope(_profile(client, second["access_token"]))


def test_a_revoked_session_cannot_be_refreshed_back_into_existence(
    client: httpx.Client, open_session
):
    """Logout must kill the refresh token too, or signing out achieves nothing.

    The access token and the refresh token are two credentials on one row. Logout deletes the
    row, so a refresh token retained by an attacker — or by a stale tab — has nothing to
    match and cannot mint a new access token. If the row were merely marked (or if refresh
    matched on the JWT rather than the stored hash), "log out" would hold for at most an hour
    and then quietly reverse itself.
    """
    session = open_session()
    assert_envelope(
        client.post(
            "/user/logout", headers={"Authorization": f"Bearer {session['access_token']}"}
        )
    )

    response = client.post("/user/refresh", json={"refresh_token": session["refresh_token"]})
    assert response.status_code == 401
    assert_envelope(response, 401)
    assert response.json()["message"] == "Invalid or revoked refresh token"


# --------------------------------------------------------------------------------------
# The allow-list: health must stay reachable without credentials
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("path", HEALTH_PATHS)
def test_health_endpoints_are_reachable_without_authentication(
    client: httpx.Client, path: str
):
    """Health is allow-listed by prefix, and a container orchestrator carries no bearer token.

    `UNAUTHENTICATED_PREFIXES` is a single-entry tuple, and it is the only prefix rule in a
    list that is otherwise exact-match. If it were dropped, or if the router's `/api/v1/health`
    prefix drifted, every probe would get a 401: Docker would mark the API unhealthy and keep
    restarting a process that is serving traffic perfectly. Going the other way — widening the
    prefix — would silently unauthenticate real endpoints, which is why each path is asserted
    individually rather than by hitting one and assuming the rest.

    Routed through nginx, which strips its own `/api` prefix, so this also pins that the two
    prefixes compose the way the deployment assumes.
    """
    response = client.get(path)
    assert response.status_code == 200, f"{path} should need no credentials"
    assert_envelope(response)
    assert response.json()["status"] == "success"


def test_health_ready_reports_the_database_as_ok(client: httpx.Client):
    """Readiness is only meaningful if it actually touches MySQL.

    `/ready` and `/dependencies` both run `SELECT 1` through a real pooled connection, which
    is the difference between "the process is up" (`/live`) and "the process can serve a
    request". A readiness probe that reported ok without querying would let the container
    take traffic while the database was unreachable, and every request would 500.
    """
    for path in ("/api/v1/health/ready", "/api/v1/health/dependencies"):
        data = assert_envelope(client.get(path))
        assert data["database"] == "ok", f"{path} did not report a healthy database"

    live = assert_envelope(client.get("/api/v1/health/live"))
    assert live["status"] == "live"


def test_health_being_public_does_not_make_the_api_root_public(client: httpx.Client):
    """The allow-list is a list, not a mood — everything outside it still needs a token.

    Worth pinning next to the health tests because the two failures are the same edit. The
    bare `/` route in `main.py` is the cheapest possible probe for "did the allow-list stop
    being a list": it is unguarded by any permission dependency, so if the middleware ever
    lets it through, nothing else will stop it either.
    """
    response = client.get("/")
    assert response.status_code == 401
    assert_envelope(response, 401)


def test_every_response_carries_a_request_id_through_nginx(client: httpx.Client):
    """`X-Request-ID` is the only thread between a browser error and a line in the API log.

    `RequestContextMiddleware` is the outermost middleware specifically so it wraps the 401s
    that `JWTAuthMiddleware` returns without ever reaching a router — the responses hardest to
    diagnose, since there is no handler to log from. Both an authenticated-path 401 and a
    successful public response are checked, because a proxy that dropped the header on the
    way back would look completely fine in every in-process test.
    """
    unauthorized = client.get(PROTECTED_PATH)
    assert unauthorized.status_code == 401
    assert unauthorized.headers.get("x-request-id"), "401s must still be traceable"

    healthy = client.get("/api/v1/health/live")
    assert healthy.headers.get("x-request-id")
    assert healthy.headers["x-request-id"] != unauthorized.headers["x-request-id"]
