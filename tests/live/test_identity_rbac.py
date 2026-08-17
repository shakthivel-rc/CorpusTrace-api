"""Identity, RBAC and the admin console, exercised against the deployed stack.

The hermetic integration suite drives a TestClient against in-memory SQLite, which is the
right default but cannot see the class of defect this file exists for: things that are only
wrong once a real MySQL row, a real nginx hop and a real response model are all in play.
The canonical example is the one this file opens with — GET /users/profile returned a bare
HTTP 500 for the seeded superadmin because `UserProfileResponse` re-validated a stored
address whose TLD email-validator refuses. The row was fine, the query was fine, and the
serializer was the thing that broke. Nothing short of "sign in as the real seeded account
and load the real profile" catches that.

Everything here therefore talks HTTP through the SPA origin, exactly as a browser does.

Two constraints shape how the assertions are written:

  * The live database PERSISTS between runs and other suites write to it CONCURRENTLY.
    So nothing asserts a global count ("there are 3 users"), nothing asserts a snapshot of
    a shared list, and everything this file creates is namespaced with the `unique` fixture
    and torn down in a `finally`.
  * The superadmin session is shared. Nothing here logs out, changes the superadmin's
    password, or deletes an account it did not create.

Users are created through `POST /user/register` rather than the admin console's
`POST /users`, because the console endpoint is currently broken — see
`test_admin_console_can_create_a_user`, which documents that defect as a strict xfail.
Registration produces an ordinary `users` row (unverified, no password), which is all the
listing, fetch, update and soft-delete paths under test here care about.
"""
import httpx
import pytest

from conftest import SUPERADMIN_EMAIL, assert_envelope

# Everything under tests/live is auto-marked `live` by the collection hook in conftest.py.

# The two slugs that gate an endpoint anywhere in the codebase (core/permissions.py).
# The other 36 catalog entries are seeded and enforce nothing directly.
GATING_SLUGS = {"manage_users", "ai_access"}

# Every admin-console list endpoint answers {"records": [...], "total": N}.
CONSOLE_LISTS = ["/users", "/roles", "/permissions", "/activity-logs"]


def _register(client: httpx.Client, unique: str) -> dict:
    """Create a real `users` row over HTTP and return the payload it was created from."""
    payload = {
        "email": f"live-rbac-{unique}@example.com",
        "first_name": "Live",
        "last_name": f"Rbac{unique}",
        "username": f"live-rbac-{unique}",
        "organization": "LiveTests",
        "department": "QA",
    }
    assert_envelope(client.post("/user/register", json=payload), 201)
    return payload


def _find_user(admin: httpx.Client, unique: str) -> dict | None:
    """Look the user up through the console's own search, which is how an admin would."""
    data = assert_envelope(admin.get("/users", params={"search": unique}))
    matches = [r for r in data["records"] if unique in r["username"]]
    assert len(matches) <= 1, f"namespaced search matched more than one user: {matches}"
    return matches[0] if matches else None


def _soft_delete(admin: httpx.Client, user_id: str) -> None:
    """Best-effort teardown. A 400 means the test already deleted it, which is fine."""
    admin.delete(f"/users/{user_id}")


@pytest.fixture()
def throwaway_user(client: httpx.Client, admin: httpx.Client, unique: str):
    """A real user row, namespaced per run, removed afterwards whatever the test did.

    The live database is not truncated between runs, so a test that leaves a row behind
    poisons the next run's search. The teardown is unconditional and tolerant: the test
    body is free to delete the user itself as part of what it is asserting.
    """
    payload = _register(client, unique)
    record = _find_user(admin, unique)
    assert record is not None, "a registered user did not appear in GET /users"
    try:
        yield {**payload, "id": record["id"], "unique": unique}
    finally:
        _soft_delete(admin, record["id"])


# --------------------------------------------------------------------------------------
# GET /users/profile
# --------------------------------------------------------------------------------------


def test_profile_returns_the_stored_superadmin_address_verbatim(admin):
    """REGRESSION GUARD. This endpoint 500'd for the seeded account on every load.

    `UserProfileResponse.email` was `EmailStr`, which re-validates an address that the
    database has already accepted. email-validator refuses the special-use TLDs of
    RFC 6761/6762 — .local, .localhost, .test, .invalid — so superadmin@corpustrace.local
    could sign in (login takes a plain `str`) and then got an uncaught ValidationError on
    the way *out*. With no global exception handler that is a bare, envelope-less 500.

    The hermetic twin of this test (tests/integration/test_user_profile_email.py) seeds its
    own row; this one asserts it against the address the deployed seeder actually created,
    which is the value that broke. Asserting the address comes back *verbatim* is the whole
    point — a fix that normalized or masked the address would leave the console showing the
    wrong account.
    """
    data = assert_envelope(admin.get("/users/profile"))

    assert data["email"] == SUPERADMIN_EMAIL


def test_profile_carries_the_shape_the_console_renders(admin):
    """The profile is the SPA's identity source; a missing key is a blank admin console.

    There is no OpenAPI schema and no API versioning here, so the response shape is only
    ever verified by something like this. Roles and permissions are asserted as *lists of
    the right records*, not by content: which roles the account holds is deployment data,
    but "roles is a list of {role_id, role_name}" is the contract.
    """
    data = assert_envelope(admin.get("/users/profile"))

    for key in ("id", "username", "email", "first_name", "last_name",
                "organization", "department", "created_at", "updated_at"):
        assert key in data, f"profile missing {key!r}: {sorted(data)}"

    assert isinstance(data["roles"], list)
    for role in data["roles"]:
        assert set(role) == {"role_id", "role_name"}

    assert isinstance(data["permissions"], list)
    for permission in data["permissions"]:
        assert set(permission) == {"permission_name", "machine_name"}


def test_profile_grants_the_seeded_admin_the_slug_that_gates_the_console(admin):
    """`manage_users` is what stands between the seeded account and the admin console.

    Permissions attach to roles only, and `check_permissions` requires *all* listed slugs,
    so an Admin role that lost this one entry would 403 every user-management endpoint
    while still looking correctly signed in. Asserting it from the profile — the same
    payload the SPA uses to decide what to render — keeps the UI and the guard in step.
    """
    data = assert_envelope(admin.get("/users/profile"))
    slugs = {p["machine_name"] for p in data["permissions"]}

    assert "manage_users" in slugs, f"seeded admin cannot manage users; has {sorted(slugs)}"


# --------------------------------------------------------------------------------------
# The admin console's list endpoints
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("path", CONSOLE_LISTS)
def test_console_lists_answer_with_records_and_a_total(admin, path):
    """Every console list must return 200 and the {records, total} envelope for an admin.

    Deliberately no count assertions: this database persists and other suites are writing
    to it while this runs, so any "there are exactly N" claim is a scheduled failure. What
    is worth pinning is that the endpoint is reachable, wrapped in the standard envelope,
    and paginated in the shape the SPA destructures.
    """
    data = assert_envelope(admin.get(path))

    assert isinstance(data["records"], list)
    assert isinstance(data["total"], int)
    assert data["total"] >= 0


@pytest.mark.parametrize("path", CONSOLE_LISTS + ["/users/profile"])
def test_console_lists_are_unreachable_without_a_token(client, path):
    """A 401 from JWTAuthMiddleware must still be a readable JSON envelope.

    CORS is currently the *innermost* middleware, so a 401 short-circuits before it and
    comes back without `Access-Control-Allow-Origin`. That is a known wart. What must not
    also be true is the response being unparseable: the SPA's fetch wrapper reads
    `status_code` off the body to decide whether to refresh the token, and a bare-text 401
    sends it down the wrong branch.
    """
    response = client.get(path)

    assert response.status_code == 401
    assert response.headers["content-type"].startswith("application/json")
    body = response.json()
    assert body["status"] == "error"
    assert body["status_code"] == 401


@pytest.mark.parametrize("path", ["/users", "/roles", "/permissions"])
def test_console_lists_reject_an_unknown_sort_field(admin, path):
    """`sort_by` is interpolated into `getattr(Model, sort_by)`, so the allow-list matters.

    Each of these controllers checks the field against a whitelist before the getattr. If
    that check were dropped, an arbitrary attribute name would reach `getattr` and an
    unknown one would raise — an uncaught AttributeError, which here is a bare 500. A 400
    with the envelope is both the correct answer and the evidence the allow-list is intact.
    """
    response = admin.get(path, params={"sort_by": "definitely-not-a-column"})

    assert_envelope(response, 400)


def test_seeded_roles_are_listed_and_expand_into_their_permissions(admin):
    """The RBAC screen lists roles, then opens one to show what it grants.

    Asserted as a relationship rather than a fixed permission set: which permissions the
    Admin role holds is deployment data that a migration may legitimately change, but
    "a role detail response carries an `id`, a `name` and a list of permissions" is the
    contract the screen is written against. The seeded Admin role is required to exist
    because the whole console is unusable without it.
    """
    listing = assert_envelope(admin.get("/roles"))
    by_name = {r["name"]: r for r in listing["records"]}
    assert "Admin" in by_name, f"seeded Admin role is missing; saw {sorted(by_name)}"

    detail = assert_envelope(admin.get(f"/roles/{by_name['Admin']['id']}"))
    assert detail["id"] == by_name["Admin"]["id"]
    assert detail["name"] == "Admin"
    assert isinstance(detail["permissions"], list)
    assert detail["permissions"], "the Admin role grants nothing"
    for permission in detail["permissions"]:
        assert set(permission) == {"id", "permission_name"}


def test_permission_catalogue_exposes_both_slugs_that_gate_an_endpoint(admin):
    """`manage_users` and `ai_access` are persisted data, not constants.

    Slugs live in `permissions.machine_name` and are seeded by migration 20260721a002.
    They are also the only two that any route actually enforces. If a migration or a
    rename dropped one, every endpoint behind it would start refusing everybody, and the
    only signal would be 403s in production — the catalogue itself would still look
    healthy because the other 36 entries are untouched.
    """
    data = assert_envelope(admin.get("/permissions"))
    slugs = {p["machine_name"] for p in data["records"]}

    assert GATING_SLUGS <= slugs, f"missing gating slug(s): {sorted(GATING_SLUGS - slugs)}"


@pytest.mark.parametrize(
    "path",
    ["/roles/no-such-role-id", "/permissions/no-such-permission-id"],
)
def test_unknown_role_and_permission_ids_are_404_with_the_envelope(admin, path):
    """A missing id is a 404 the SPA can render, not an exception it has to guess at.

    Worth pinning because the sibling endpoint for users gets this wrong today (see
    `test_unknown_user_id_is_a_404_not_a_500`) — these two are the proof that 404 is the
    intended house style and the user one is a defect rather than a convention.
    """
    assert_envelope(admin.get(path), 404)


# --------------------------------------------------------------------------------------
# The user lifecycle
# --------------------------------------------------------------------------------------


def test_a_new_user_appears_in_the_console_and_is_fetchable_by_id(admin, throwaway_user):
    """Create -> list -> fetch, over HTTP, against MySQL.

    `GET /users` and `GET /users/{id}` are built by two entirely separate hand-rolled
    projections in `user_controller`, so they can drift: the list carries `username`,
    `status` and a flattened permission set, the detail carries `role` (singular). This
    asserts the two agree about the one thing they must — that it is the same user — and
    that each still carries the fields its screen renders.
    """
    listed = _find_user(admin, throwaway_user["unique"])
    assert listed is not None
    assert listed["email"] == throwaway_user["email"]
    assert listed["username"] == throwaway_user["username"]
    assert isinstance(listed["roles"], list)
    assert isinstance(listed["permissions"], list)

    detail = assert_envelope(admin.get(f"/users/{throwaway_user['id']}"))
    assert detail["id"] == throwaway_user["id"]
    assert detail["email"] == throwaway_user["email"]
    assert detail["first_name"] == throwaway_user["first_name"]
    assert detail["organization"] == throwaway_user["organization"]


def test_updating_a_user_persists_and_is_visible_on_the_next_read(admin, throwaway_user):
    """PUT /users/{id} writes through a bulk `query.update()`, which bypasses the ORM.

    That style skips instance-level events, so the only honest verification is to read the
    row back over HTTP afterwards rather than trusting the 200. `department` is used here
    because it is a plain column with no validator or uniqueness constraint attached — a
    change to it can only fail for the reason under test.
    """
    assert_envelope(admin.put(f"/users/{throwaway_user['id']}", json={"department": "Ops"}))

    detail = assert_envelope(admin.get(f"/users/{throwaway_user['id']}"))
    assert detail["department"] == "Ops"


def test_deleting_a_user_is_soft_and_removes_it_from_the_console(client, admin, unique):
    """Deletion must hide the user from every listing without destroying the row.

    Users are soft-deleted (`deleted = 1`, `deleted_at` set by a `@validates` hook) while
    their sessions and role rows are hard-deleted. The invariant that actually matters to
    an operator is the one asserted here: after a delete the account is gone from
    `GET /users`, because `fetch_user` filters `deleted != 1` centrally. A single missed
    filter would leave deleted accounts on the admin screen looking active.

    A second delete must be a 400 with the envelope, not a 500 — the console offers Delete
    on a row a colleague may have just removed, and a double-click must not look like an
    outage. The whole body is wrapped in try/finally so a failed assertion still cleans up.

    This test manages its own user rather than using the `throwaway_user` fixture, because
    deleting it is the behaviour under test.
    """
    _register(client, unique)
    record = _find_user(admin, unique)
    assert record is not None
    user_id = record["id"]

    try:
        deleted = assert_envelope(admin.delete(f"/users/{user_id}"))
        assert deleted["user_id"] == user_id

        assert _find_user(admin, unique) is None, "a soft-deleted user is still listed"

        # Idempotency: the second attempt is a refusal, not a crash.
        assert_envelope(admin.delete(f"/users/{user_id}"), 400)
    finally:
        _soft_delete(admin, user_id)


def test_deleting_a_user_writes_an_activity_log(admin, client, unique):
    """Activity logging is fire-and-forget: it swallows every exception and only flushes.

    An audit row therefore survives only because the calling controller happens to commit
    afterwards. That coupling is invisible at the call site and trivially broken by moving
    a `db.commit()`, so the only way to know the trail still exists is to perform an action
    and then read the log back through the API. The record is located by the namespaced
    suffix rather than by position, since other suites are writing DELETE entries too.
    """
    _register(client, unique)
    record = _find_user(admin, unique)
    assert record is not None

    try:
        assert_envelope(admin.delete(f"/users/{record['id']}"))

        logs = assert_envelope(
            admin.get("/activity-logs", params={"action": "DELETE", "limit": 100})
        )
        mine = [entry for entry in logs["records"] if unique in (entry["details"] or "")]
        assert mine, "no DELETE activity log recorded for the user this test removed"
        assert mine[0]["action"] == "DELETE"
        assert mine[0]["userName"], "the log did not record who performed the deletion"
    finally:
        _soft_delete(admin, record["id"])


# --------------------------------------------------------------------------------------
# GET /activity-logs — the one correctly paginated list
# --------------------------------------------------------------------------------------


def test_activity_logs_report_a_real_total_not_the_page_length(admin):
    """`/activity-logs` is the only list in the API that paginates correctly.

    `GET /users` and `GET /roles` compute `total` as `len(page)`, so their total is useless
    for a pager. `get_activity_logs` runs `query.count()` *before* applying offset/limit,
    which is what lets the SPA draw page numbers. Asking for one record out of many and
    checking `total` still exceeds the page is exactly the difference between the two
    implementations, and it is what would silently regress if someone "tidied" this
    controller to match its neighbours.
    """
    data = assert_envelope(admin.get("/activity-logs", params={"page": 1, "limit": 1}))

    assert isinstance(data["total"], int)
    assert len(data["records"]) <= 1
    if data["total"] < 2:
        pytest.skip("this stack has fewer than two activity log entries to page through")
    assert data["total"] > len(data["records"]), (
        "total collapsed to the page length — the COUNT was dropped"
    )


def test_activity_logs_limit_and_offset_are_applied(admin):
    """Both halves of the pager, asserted without depending on a stable ordering.

    The list is ordered newest-first and other suites are logging in while this runs, so
    comparing page 1 against page 2 across two requests is inherently racy — a row inserted
    between them shifts everything down by one. These two claims are race-free instead:
    a page is never longer than its limit, and a page far past the end is empty while the
    total stays non-zero, which can only happen if OFFSET is being applied to a real COUNT.
    """
    page = assert_envelope(admin.get("/activity-logs", params={"page": 1, "limit": 3}))
    assert len(page["records"]) <= 3
    ids = [entry["id"] for entry in page["records"]]
    assert len(ids) == len(set(ids)), "one page returned the same log row twice"

    if page["total"] == 0:
        pytest.skip("this stack has no activity log entries")

    beyond = assert_envelope(
        admin.get("/activity-logs", params={"page": 1_000_000, "limit": 3})
    )
    assert beyond["records"] == []
    assert beyond["total"] > 0, "paging past the end also zeroed the total"


def test_activity_logs_filter_by_action_case_insensitively(admin):
    """The filter upper-cases its argument, so the SPA may send either case.

    `action.upper()` in the controller is the only reason a lowercase `?action=login` from
    the UI matches rows stored as `LOGIN`. Dropping it would return an empty list for every
    filter the SPA sends — a bug that reads as "there is no activity" rather than as an
    error. An action nobody has ever performed must come back empty with a total of 0
    rather than falling through to the unfiltered list.
    """
    filtered = assert_envelope(admin.get("/activity-logs", params={"action": "login"}))
    for entry in filtered["records"]:
        assert entry["action"] == "LOGIN"

    nothing = assert_envelope(
        admin.get("/activity-logs", params={"action": "no-such-action-ever"})
    )
    assert nothing["records"] == []
    assert nothing["total"] == 0


# --------------------------------------------------------------------------------------
# Known product defects.
#
# These express the CORRECT behaviour and are marked strict-xfail, so the suite stays green
# today and turns red the moment the defect is fixed and the marker is left behind. Do not
# weaken the assertions to make them pass — the assertion is the specification.
# --------------------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason=(
        "BUG: GET /users/{id} returns 500 for an unknown id. get_user() builds "
        "JSONResponse(status_code='User not found', content=<ErrorResponse instance>) — a "
        "string status code and an unserializable body — which raises inside the enclosing "
        "try, so the generic `except` turns a missing row into 'Error occurred while "
        "fetching users' with status 500. Compare /roles/{id} and /permissions/{id}, which "
        "both answer 404 correctly."
    ),
)
def test_unknown_user_id_is_a_404_not_a_500(admin, unique):
    """A user id that does not exist is a client error, and the console has to tell them so.

    A 500 is indistinguishable from the database being down, so the SPA shows "something
    went wrong" for a link to a user who was simply deleted — which is the single most
    likely way to arrive at a stale user id.
    """
    assert_envelope(admin.get(f"/users/missing-{unique}"), 404)


def test_admin_console_can_create_a_user(admin, unique):
    """Creating a user from the admin console must work end to end.

    This is the console's primary write path — the only supported way to onboard someone
    into a named role without them self-registering. It used to be the one endpoint here
    that could not succeed at all: `create_new_user` returns a dict (`vars(new_user)`) and
    `add_user_controller` read `new_user.id` when writing the activity log, raising
    `AttributeError: 'dict' object has no attribute 'id'` on every call — the route requires
    `manage_users`, so `request.state.user` is always populated and the log branch is always
    taken. With no global exception handler the console got a bare text/plain 500 with no
    envelope to read. Fixed 2026-08-11; covered without a live stack by
    `tests/integration/test_user_creation.py`, since only the live suite drove this route
    with a real session row and the live suite is skipped by default.

    The assertions are deliberately the full round trip rather than just the status code,
    because a fix that returns 201 without persisting the row would be just as broken from
    the operator's side.

    Cleanup runs regardless, so a passing run does not leave a user behind for the next one.
    """
    roles = assert_envelope(admin.get("/roles"))
    by_name = {r["name"]: r["id"] for r in roles["records"]}
    role_id = by_name.get("Customer") or next(iter(by_name.values()))

    payload = {
        "email": f"live-console-{unique}@example.com",
        "first_name": "Console",
        "last_name": f"Created{unique}",
        "username": f"live-console-{unique}",
        "organization": "LiveTests",
        "department": "QA",
        "role_id": role_id,
    }

    try:
        created = assert_envelope(admin.post("/users", json=payload), 201)
        assert created["email"] == payload["email"]

        listed = _find_user(admin, unique)
        assert listed is not None, "the created user is not in GET /users"
        assert listed["roles"], "the created user was not given the role it was created with"
    finally:
        found = _find_user(admin, unique)
        if found:
            _soft_delete(admin, found["id"])


@pytest.mark.xfail(
    strict=True,
    reason=(
        "BUG: re-registering an address belonging to a soft-deleted user is a bare 500. "
        "get_user_by_email() filters `deleted != True`, so register_controller cannot see "
        "the soft-deleted row and inserts a duplicate; users.email carries a UNIQUE index, "
        "so MySQL raises 1062 and — with no try/except in the controller and no global "
        "exception handler — the caller gets text/plain 'Internal Server Error'. The "
        "address is effectively burned forever, and the API never says why."
    ),
)
def test_re_registering_a_soft_deleted_address_is_refused_not_crashed(client, admin, unique):
    """Soft delete must not turn an email address into an unexplained dead end.

    Whether a deleted user's address should be reusable is a product decision; that it must
    not produce an envelope-less 500 is not. Either answer — "already registered" or a
    successful re-registration — is serviceable. A bare 500 tells the person signing up
    nothing and tells support nothing either.
    """
    payload = _register(client, unique)
    record = _find_user(admin, unique)
    assert record is not None

    try:
        assert_envelope(admin.delete(f"/users/{record['id']}"))

        again = client.post("/user/register", json={**payload, "username": f"retry-{unique}"})

        assert again.status_code != 500, (
            f"re-registration crashed: {again.headers.get('content-type')} "
            f"{again.text[:200]!r}"
        )
        assert again.headers["content-type"].startswith("application/json")
        body = again.json()
        assert body["status_code"] == again.status_code
    finally:
        _soft_delete(admin, record["id"])
        leftover = _find_user(admin, unique)
        if leftover:
            _soft_delete(admin, leftover["id"])
