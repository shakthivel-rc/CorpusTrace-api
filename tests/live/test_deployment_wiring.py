"""The deployment wiring: nginx, the /api proxy, MIME types and container health.

This is the one layer no other suite in either project can reach. The hermetic backend suite
drives a TestClient, so nginx is not in the picture at all; the SPA's Vitest and Playwright
specs intercept at the browser boundary, so the dev server's MIME table is used rather than
the container's. Everything below only exists in a built image behind a reverse proxy — and
three separate defects reached a browser through that gap in a single day:

  * nginx's bundled mime.types has no entry for `.mjs`, so the pdf.js worker — the only `.mjs`
    Vite emits — was served as `application/octet-stream`. Browsers enforce the MIME type on
    ES module scripts, so it was fetched (200, right bytes) and then refused, and the entire
    source-evidence view was dead with a console message about a dynamic import.
  * The SPA container reported `unhealthy` forever while serving every request correctly,
    because its healthcheck probed the hostname `localhost` — which resolves to `::1` first —
    while nginx's `listen 80;` binds IPv4 only.
  * `GET /users/profile` 500'd against the real database for the seeded superadmin, which is
    the general shape of the problem: MySQL and SQLite disagree, and only one of them ships.

Each test here is written so it would have FAILED before the corresponding fix. They assert
observable behaviour over an HTTP socket, exactly as a browser sees it, because that is the
only vantage point from which the proxy exists.

Static-asset checks go to `live_root` (the SPA origin, port 8080) with a bare httpx client
rather than through `api_base`: the point is what nginx does with a path, and `api_base` is
specifically the part of the origin that gets proxied away.
"""
import json
import re
import shutil
import subprocess
import uuid
from collections import deque

import httpx
import pytest

from conftest import assert_envelope  # tests/live/conftest.py — see its module docstring

# Vite emits every hashed bundle under /assets/. This finds them inside index.html and inside
# each chunk's own import map, which is how the lazy-loaded pdf.js worker is reachable at all:
# nothing links it from the HTML, only PdfViewer's chunk names it.
_ASSET_REF = re.compile(r"(?:/)?assets/[A-Za-z0-9._\-]+\.(?:mjs|js|css)")

# What a browser will accept as an ES module. `application/octet-stream` — the pre-fix value —
# is deliberately not in here, and neither is `text/plain`; both are refused by the module
# loader's strict MIME check.
_JAVASCRIPT_TYPES = {
    "text/javascript",
    "application/javascript",
    "application/ecmascript",
    "text/ecmascript",
    "module",
}

# Compose names the containers `<project>-<service>-<n>`; the project defaults to the
# directory, which is `Nexarag-api`, lower-cased to `nexarag`.
_CONTAINERS = ("nexarag-api-1", "nexarag-app-1", "nexarag-db-1")


def _essence(content_type: str | None) -> str:
    """`text/plain; charset=utf-8` -> `text/plain`. Parameters are not part of the verdict."""
    return (content_type or "").split(";")[0].strip().lower()


def _cache_control(response: httpx.Response) -> str:
    """nginx emits `expires` and `add_header Cache-Control` as two separate header lines.

    Both are real and a browser honours the union, so join them rather than reading whichever
    one httpx hands back first.
    """
    return ", ".join(response.headers.get_list("cache-control")).lower()


@pytest.fixture(scope="session")
def spa(live_root: str) -> httpx.Client:
    """A plain browser-shaped client against the SPA origin — no bearer token, no /api base."""
    with httpx.Client(base_url=live_root, timeout=30, follow_redirects=False) as http:
        yield http


@pytest.fixture(scope="session")
def built_assets(spa: httpx.Client) -> list[str]:
    """Every /assets/ path reachable from index.html, discovered rather than hard-coded.

    Vite puts a content hash in each filename, so `pdf.worker.min-CHFwMXne.mjs` is a different
    string after every build. A test that pinned it would go green by 404ing on the next
    `make up`, which is precisely the failure this file exists to catch. Crawl instead:
    index.html names the entry chunk, the entry chunk names its lazy children, and four hops
    down that graph is the pdf.js worker.

    `.mjs` files are found but never crawled *into*: the worker is ~1.2 MB of minified code,
    nothing imports out of it, and regexing it would only cost time.
    """
    index = spa.get("/")
    assert index.status_code == 200, f"SPA origin did not serve an index: {index.status_code}"

    found: set[str] = set()
    visited: set[str] = set()
    queue: deque[str] = deque()

    def absorb(text: str) -> None:
        for ref in _ASSET_REF.findall(text):
            path = "/" + ref.lstrip("/")
            found.add(path)
            if path.endswith(".js") and path not in visited:
                queue.append(path)

    absorb(index.text)
    # A bound, so a pathological build cannot turn this fixture into an infinite crawl.
    for _ in range(80):
        if not queue:
            break
        path = queue.popleft()
        if path in visited:
            continue
        visited.add(path)
        chunk = spa.get(path)
        if chunk.status_code == 200:
            absorb(chunk.text)

    assert found, "crawled index.html and found no /assets/ references at all"
    return sorted(found)


@pytest.fixture(scope="session")
def pdf_worker_asset(built_assets: list[str]) -> str:
    """The pdf.js worker, by name pattern rather than by hash."""
    workers = [p for p in built_assets if p.endswith(".mjs") and "pdf.worker" in p]
    assert workers, (
        "no /assets/pdf.worker*.mjs is reachable from index.html. Either the source-evidence "
        f"view was removed or the bundle no longer links its worker. Found: {built_assets}"
    )
    return workers[0]


def _docker_inspect(container: str, template: str) -> str:
    """`docker inspect -f <template> <container>`, or skip if docker is not usable here."""
    if not shutil.which("docker"):
        pytest.skip("docker is not on PATH; container-health checks need it")
    try:
        result = subprocess.run(
            ["docker", "inspect", "-f", template, container],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:  # pragma: no cover - env dependent
        pytest.skip(f"could not run docker inspect: {exc}")
    if result.returncode != 0:
        pytest.skip(
            f"docker inspect {container} failed (not a compose stack, or no daemon access): "
            f"{result.stderr.strip()[:200]}"
        )
    return result.stdout.strip()


@pytest.fixture()
def uploaded_source(admin: httpx.Client, unique: str):
    """A knowledge base holding two files whose declared MIME type is a lie.

    Both are uploaded as `text/html` — the multipart part header is entirely client-controlled,
    so this is what an attacker sends. One has a `.txt` extension and one an extension the
    server has never heard of, which are the two branches of `INLINE_MEDIA_TYPES`.

    Yields `(resource_id, {file_name: file_id})`. Cleanup is a soft delete: the permanent
    delete currently 500s on MySQL (see the xfail at the bottom of this file), and a fixture
    that depended on a broken endpoint would leak an unbounded number of live knowledge bases
    into a database that persists between runs.
    """
    name = f"live-wiring-{unique}"
    payload = [
        # Deliberately contains markup: if this were ever served as text/html from the API
        # origin, this is the script that would run with the app's cookies in scope.
        ("files", (f"wiring-{unique}.txt", b"<script>alert(1)</script> alpha beta", "text/html")),
        ("files", (f"wiring-{unique}.weirdext", b"opaque payload", "text/html")),
    ]
    created = admin.post("/resources/upload_files", data={"resource_name": name}, files=payload)
    if created.status_code != 202:
        pytest.skip(f"could not stage an upload: HTTP {created.status_code} {created.text[:200]}")
    resource_id = created.json()["data"]["resource_id"]

    # The `File` rows and the bytes on disk both exist by the time upload_files returns 202 —
    # only parsing and chunking are deferred to the worker — so the documents listing is
    # populated immediately and there is nothing to poll for.
    listing = admin.get(f"/resources/{resource_id}/documents")
    records = (listing.json().get("data") or {}).get("records") or []
    files = {record["file_name"]: record["id"] for record in records}

    try:
        assert len(files) == 2, f"upload staged {len(files)} documents, expected 2: {files}"
        yield resource_id, files
    finally:
        admin.delete(f"/resources/brain/{resource_id}")
        # Best effort; expected to 500 today, and that is what the xfail below documents.
        admin.delete(f"/resources/brain/{resource_id}/permanent")


# --------------------------------------------------------------------------------------
# The SPA is actually served
# --------------------------------------------------------------------------------------


def test_the_spa_index_is_served_as_html(spa: httpx.Client):
    """The most basic thing the container is for, and the cheapest way to catch a bad build.

    A Dockerfile that copied the wrong directory, or an nginx `root` pointing somewhere empty,
    produces nginx's own 403/404 page — still HTML, still from nginx, and easy to mistake for
    a working deploy in a smoke check that only looks at the status code. So assert the app's
    own markup is present, not merely that something answered.
    """
    response = spa.get("/")

    assert response.status_code == 200
    assert _essence(response.headers.get("content-type")) == "text/html"
    assert '<div id="root"></div>' in response.text, "served HTML is not the SPA shell"
    assert "<title>NexaRAG</title>" in response.text


# --------------------------------------------------------------------------------------
# The /api prefix strip
# --------------------------------------------------------------------------------------


def test_health_through_the_proxy_needs_the_api_prefix_twice(spa: httpx.Client):
    """`/api/api/v1/health/ready` is correct and is not a typo. This is why.

    nginx's `location /api/ { proxy_pass http://nexarag_api/; }` strips the leading `/api`
    before the backend sees the path — the trailing slash on proxy_pass is what does it, and
    it mirrors the rewrite in `vite.config.ts` so one bundle works in both places. Meanwhile
    the health router is the single router that carries its own `/api/v1` prefix; every other
    router (`/user`, `/users`, `/roles`, `/resources`…) is mounted bare.

    So the browser-side path is the proxy's prefix plus the router's prefix: `/api` + `/api/v1
    /health/ready`. The doubling looks like a mistake in every log line it appears in, which
    is exactly why it deserves a test that states out loud that it is intended.
    """
    response = spa.get("/api/api/v1/health/ready")

    data = assert_envelope(response)
    assert data == {"database": "ok"}


def test_a_single_api_prefix_does_not_reach_the_health_router(spa: httpx.Client):
    """The intuitive path is the wrong one, and it fails in a misleading way. Pin that.

    `/api/v1/health/ready` — which is what the backend's own origin answers on — has its `/api`
    eaten by the proxy and arrives at the backend as `/v1/health/ready`. That is not a route,
    and it is not in `UNAUTHENTICATED_ROUTES` (an exact-match list plus the `/api/v1/health`
    prefix), so `JWTAuthMiddleware` rejects it before routing and the caller gets 401 rather
    than 404. Anyone debugging a proxy misconfiguration therefore sees "Missing or invalid
    token" and goes looking at auth.

    The load-bearing assertion is that it reached the *API* at all (a JSON envelope, not
    nginx's HTML), which is what proves the strip happened rather than the path falling
    through to index.html.
    """
    response = spa.get("/api/v1/health/ready")

    assert _essence(response.headers.get("content-type")) == "application/json", (
        "single-prefixed health did not reach the API — the /api location may have stopped "
        "matching, in which case every SPA call is now served index.html with status 200"
    )
    assert response.status_code == 401
    assert response.json()["status_code"] == 401


def test_the_proxy_strips_api_on_an_ordinary_route(spa: httpx.Client, unique: str):
    """A real endpoint, not just health — health is special-cased by its own prefix.

    `/api/user/login` must arrive at `/user/login`. Getting the trailing slash on `proxy_pass`
    wrong is completely silent: nginx keeps returning 200s from index.html and every API call
    in the SPA fails to parse JSON, which reads in the browser as a broken frontend rather
    than a broken proxy. Asserting the envelope proves FastAPI answered.

    Bad credentials are used on purpose: a 401 from the login controller is proof the request
    was routed *and* handled, and it creates nothing that needs cleaning up.
    """
    response = spa.post(
        "/api/user/login",
        json={"username": f"nobody-{unique}@nexarag.invalid", "password": "not-the-password"},
    )

    assert _essence(response.headers.get("content-type")) == "application/json"
    assert response.status_code == 401
    body = response.json()
    assert body["status"] == "error"
    assert body["status_code"] == 401


def test_paths_outside_api_are_never_proxied(spa: httpx.Client):
    """Only `/api/` is proxied; everything else belongs to the SPA.

    If the location block were widened (say to `/` or to a bare `/api` without the trailing
    slash), backend routes would start answering on the SPA's own paths and a client-side
    route named after one of them would break. `POST /user/login` with no `/api` must be
    handled by nginx's static file serving, never by FastAPI.
    """
    response = spa.post("/user/login", json={"username": "x", "password": "y"})

    assert _essence(response.headers.get("content-type")) != "application/json", (
        "an un-prefixed backend path reached the API — the proxy location is too broad"
    )


# --------------------------------------------------------------------------------------
# MIME types — the regression that killed the source-evidence view
# --------------------------------------------------------------------------------------


def test_the_pdf_worker_is_served_as_javascript(spa: httpx.Client, pdf_worker_asset: str):
    """The `.mjs` MIME regression, stated exactly as the browser experiences it.

    nginx's bundled mime.types maps `js` and `wasm` but has no `mjs` entry, so this file fell
    through to `default_type application/octet-stream`. It returned 200 with the correct
    ~1.2 MB of code the entire time — which is why it looked like anything except a MIME
    problem — and the browser refused to execute it because ES module scripts are subject to
    a strict MIME check. The user-visible symptom was that no citation could ever open its
    source document.

    A 200 is therefore not enough to assert. The Content-Type is the whole test.
    """
    response = spa.get(pdf_worker_asset)

    assert response.status_code == 200, f"{pdf_worker_asset} is not being served"
    essence = _essence(response.headers.get("content-type"))
    assert essence in _JAVASCRIPT_TYPES, (
        f"{pdf_worker_asset} is served as {essence!r}. A browser will fetch it and then refuse "
        "to execute it as an ES module, and the source-evidence view will be dead while every "
        "network request looks successful."
    )


def test_sibling_js_and_css_keep_their_types(spa: httpx.Client, built_assets: list[str]):
    """The obvious fix for the `.mjs` bug would have broken every other asset.

    A `types { application/javascript mjs; }` block at server level looks like the natural
    thing to write, and nginx *replaces* the inherited type map wholesale rather than merging
    when `types` is redefined at a lower level. That would have left `.js` and `.css` with no
    mapping at all — both served as `application/octet-stream`, so the app would not boot and
    the page would be unstyled. The fix uses `default_type` inside a `location ~ \\.mjs$`
    precisely to avoid that, and this test is what notices if someone "simplifies" it back.

    Every discovered asset is checked, not a sampled one, since the cost is a handful of
    requests against localhost.
    """
    scripts = [p for p in built_assets if p.endswith(".js")]
    stylesheets = [p for p in built_assets if p.endswith(".css")]
    assert scripts and stylesheets, f"crawl found no js/css to check: {built_assets}"

    for path in scripts:
        response = spa.get(path)
        assert response.status_code == 200, path
        assert _essence(response.headers.get("content-type")) in _JAVASCRIPT_TYPES, (
            f"{path} is served as {response.headers.get('content-type')!r}"
        )

    for path in stylesheets:
        response = spa.get(path)
        assert response.status_code == 200, path
        assert _essence(response.headers.get("content-type")) == "text/css", (
            f"{path} is served as {response.headers.get('content-type')!r}; the page will "
            "render unstyled"
        )


# --------------------------------------------------------------------------------------
# Client-side routing
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("path", ["/chat", "/some/deep/route", "/users/42/edit", "/tasks"])
def test_unknown_paths_fall_through_to_the_spa(spa: httpx.Client, path: str):
    """React Router owns every path that is not a file and not `/api/`.

    Without `try_files $uri $uri/ /index.html`, a hard refresh or a pasted deep link returns
    nginx's 404 page and the app appears to have lost half its screens — while navigating to
    the same route from inside the app works perfectly, because that never touches the server.
    That asymmetry is what makes the bug survive testing.
    """
    response = spa.get(path)

    assert response.status_code == 200, f"{path} should be handled by the SPA, not 404'd"
    assert _essence(response.headers.get("content-type")) == "text/html"
    assert '<div id="root"></div>' in response.text


def test_a_missing_hashed_asset_is_a_404_not_the_spa(spa: httpx.Client):
    """The fall-through must stop at `/assets/`, and it is `try_files ... =404` that stops it.

    If a missing chunk were served index.html with status 200, the browser would parse HTML as
    JavaScript and report a syntax error at `<`, from a URL that returns 200 — one of the most
    confusing failures a stale deploy can produce. `=404` makes a missing chunk say so.
    """
    response = spa.get(f"/assets/does-not-exist-{uuid.uuid4().hex[:8]}.js")

    assert response.status_code == 404
    assert '<div id="root"></div>' not in response.text


# --------------------------------------------------------------------------------------
# Caching
# --------------------------------------------------------------------------------------


def test_hashed_assets_are_immutable(spa: httpx.Client, built_assets: list[str]):
    """A content hash in the filename is a promise; `immutable` is how it is collected."""
    sample = next(p for p in built_assets if p.endswith((".js", ".css")))

    cache_control = _cache_control(spa.get(sample))

    assert "immutable" in cache_control, f"{sample} -> {cache_control!r}"
    assert "max-age=31536000" in cache_control, f"{sample} -> {cache_control!r}"


def test_the_mjs_worker_is_cached_like_every_other_asset(
    spa: httpx.Client, pdf_worker_asset: str
):
    """The MIME fix introduced a regex location, and a regex location wins over `/assets/`.

    `location ~ \\.mjs$` takes precedence over the `location /assets/` prefix, so the caching
    headers do not inherit — they had to be repeated inside it. Forgetting that would leave
    the single largest asset in the app (~1.2 MB) re-downloaded on every open of a citation,
    with nothing in the UI to suggest anything was wrong.
    """
    cache_control = _cache_control(spa.get(pdf_worker_asset))

    assert "immutable" in cache_control, (
        f"{pdf_worker_asset} -> {cache_control!r}; the regex location for .mjs is not "
        "repeating the /assets/ caching headers"
    )


def test_index_html_is_never_cached(spa: httpx.Client):
    """index.html is the only unhashed file, so it is the only one a cache can poison.

    It names the hashed bundles. Cached, a browser keeps asking for chunks the deploy deleted
    and the app is broken for exactly as long as the cache lives — with no way for the server
    to intervene.
    """
    cache_control = _cache_control(spa.get("/"))

    assert "no-cache" in cache_control or "no-store" in cache_control, (
        f"index.html -> {cache_control!r}"
    )
    assert "immutable" not in cache_control


# --------------------------------------------------------------------------------------
# Container health
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("container", _CONTAINERS)
def test_every_container_reports_healthy(live_root: str, container: str):
    """`docker ps` is the first thing anyone looks at, so it has to be trustworthy.

    The SPA container sat at `unhealthy` indefinitely while serving every request correctly.
    That is worse than an outage in one specific way: it trains the team to ignore the health
    column, and the next container that goes unhealthy for a real reason is ignored too. It
    also blocks any orchestrator that gates rollout on health.

    `live_root` is requested but unused: these container names describe the stack
    NEXARAG_LIVE_URL points at, so this must skip with the rest of the suite rather than
    quietly inspecting whatever unrelated containers happen to be on the developer's daemon.
    """
    status = _docker_inspect(container, "{{.State.Health.Status}}")

    assert status == "healthy", (
        f"{container} reports {status!r}. Run `docker inspect -f "
        f"'{{{{json .State.Health}}}}' {container}` for the failing probe's output."
    )


def test_the_spa_healthcheck_probes_an_address_nginx_actually_binds(live_root: str):
    """The specific reason the SPA container was unhealthy: IPv6, not nginx.

    The probe was `wget --spider http://localhost/`. In the nginx image `localhost` resolves to
    `::1` before `127.0.0.1`, and the config says `listen 80;` — IPv4 only. So the probe got
    ECONNREFUSED on every attempt while the identical request over IPv4 succeeded, from the
    same container, the whole time.

    Reading the probe itself rather than only its verdict is deliberate. The status assertion
    above passes today; it would also pass tomorrow on a base image whose resolver happened to
    order A before AAAA, and then fail again on the one after that. The defect is naming a
    hostname at all when the listen directive commits to one family — so that is what is
    asserted.
    """
    probe = _docker_inspect("nexarag-app-1", "{{json .Config.Healthcheck.Test}}")
    command = " ".join(json.loads(probe))

    assert not re.search(r"https?://localhost\b", command), (
        "the SPA healthcheck probes the hostname `localhost`, which resolves to ::1 first "
        f"while nginx binds IPv4 only: {command!r}"
    )
    assert "127.0.0.1" in command or "[::1]" in command, (
        "the SPA healthcheck should name a loopback address literally, so which family it "
        f"reaches is not left to the image's resolver: {command!r}"
    )


# --------------------------------------------------------------------------------------
# Serving uploaded bytes — the stored-XSS boundary
# --------------------------------------------------------------------------------------


def test_an_unknown_file_id_is_a_404_and_not_a_500(admin: httpx.Client, unique: str):
    """There is no global exception handler, so any uncaught error here is a bare 500.

    This endpoint does path arithmetic on stored data (`resolve()`, `relative_to()`) before it
    opens anything, and every one of those can raise on input the database happily holds. A
    500 would also mean the ownership check never completed, so the distinction is not
    cosmetic. An id that belongs to nobody must be indistinguishable from an id that belongs
    to someone else: a 404, in the envelope, with no detail.
    """
    response = admin.get(f"/resources/files/missing-{unique}")

    assert response.status_code != 500, response.text[:400]
    assert_envelope(response, expected_status=404)


def test_an_unauthenticated_file_fetch_is_a_json_401(client: httpx.Client, unique: str):
    """Uploaded bytes are behind auth, and the refusal is JSON even though it comes from
    middleware rather than a route.

    `JWTAuthMiddleware` short-circuits before FastAPI routes the request, which is the point in
    the stack most likely to answer with something the SPA's fetch wrapper cannot parse. The
    401→refresh→retry path in `api/client.ts` reads the envelope, so a non-JSON 401 here would
    break the session-refresh flow rather than the file fetch.
    """
    response = client.get(f"/resources/files/missing-{unique}")

    assert response.status_code == 401
    assert _essence(response.headers.get("content-type")) == "application/json"
    assert response.json()["status_code"] == 401


def test_content_type_comes_from_the_extension_not_the_uploaded_header(
    admin: httpx.Client, uploaded_source
):
    """`files.file_type` is attacker-controlled and must never reach a Content-Type header.

    That column stores the multipart part header the client sent — here, `text/html` on a file
    whose real extension is `.txt`. Echoing it back would serve attacker-authored markup
    *inline from the API's own origin*, which is stored XSS against every user who opens the
    citation: same-origin script with the app's storage and its bearer token in scope.

    The type is therefore derived from the extension via `INLINE_MEDIA_TYPES`, and `nosniff`
    plus a `sandbox` CSP are sent so a browser cannot second-guess it either.
    """
    _resource_id, files = uploaded_source
    file_id = next(fid for name, fid in files.items() if name.endswith(".txt"))

    response = admin.get(f"/resources/files/{file_id}")

    assert response.status_code == 200, response.text[:300]
    essence = _essence(response.headers.get("content-type"))
    assert essence == "text/plain", (
        f"a .txt uploaded as text/html was served as {essence!r}; if that is text/html this is "
        "stored XSS on the API origin"
    )
    assert response.headers.get("x-content-type-options") == "nosniff"
    assert "sandbox" in (response.headers.get("content-security-policy") or "")
    # And the bytes themselves are unchanged — the defence is the header, not sanitisation.
    assert b"<script>" in response.content


def test_an_unknown_extension_is_octet_stream_with_nosniff(
    admin: httpx.Client, uploaded_source
):
    """The default branch has to be the safe one, because the allow-list can never be complete.

    An extension `INLINE_MEDIA_TYPES` has never heard of falls back to
    `application/octet-stream`, which a browser downloads rather than renders. `nosniff` is
    what stops it from ignoring that and content-sniffing its way to `text/html` anyway —
    without the header, an octet-stream body starting with markup is exactly what sniffing is
    designed to "helpfully" reinterpret.
    """
    _resource_id, files = uploaded_source
    file_id = next(fid for name, fid in files.items() if name.endswith(".weirdext"))

    response = admin.get(f"/resources/files/{file_id}")

    assert response.status_code == 200, response.text[:300]
    assert _essence(response.headers.get("content-type")) == "application/octet-stream"
    assert response.headers.get("x-content-type-options") == "nosniff"


def test_serving_a_source_file_survives_the_proxy_intact(
    spa: httpx.Client, admin_token: str, uploaded_source
):
    """nginx must not rewrite the security headers on its way past.

    `X-Content-Type-Options` and the `sandbox` CSP are set by the controller, and the whole
    point of them is that they reach a browser — which talks to nginx, not to uvicorn. A
    `proxy_hide_header`, a compressing filter that drops them, or an `add_header` elsewhere in
    the config (which *replaces* inherited headers at the same level) would remove them
    silently, and every backend test would still pass.
    """
    _resource_id, files = uploaded_source
    file_id = next(fid for name, fid in files.items() if name.endswith(".txt"))

    response = spa.get(
        f"/api/resources/files/{file_id}", headers={"Authorization": f"Bearer {admin_token}"}
    )

    assert response.status_code == 200, response.text[:300]
    assert _essence(response.headers.get("content-type")) == "text/plain"
    assert response.headers.get("x-content-type-options") == "nosniff"


# --------------------------------------------------------------------------------------
# A defect the hermetic suite structurally cannot see
# --------------------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=False,
    reason=(
        "BUG: purge_resource deletes the resources row without first clearing "
        "ingestion_jobs.resource_id, so MySQL's FK ingestion_jobs_ibfk_1 raises 1451 and the "
        "endpoint returns a bare 500. Invisible to the hermetic suite because SQLite does not "
        "enforce foreign keys unless PRAGMA foreign_keys is on."
    ),
)
def test_permanently_deleting_a_resource_that_has_an_ingestion_job(
    admin: httpx.Client, uploaded_source
):
    """`DELETE /resources/brain/{id}/permanent` 500s on every base uploaded since 2026-08-05.

    Every upload since background indexing shipped creates an `ingestion_jobs` row pointing at
    the resource, and `purge_resource` (rag/service.py) deletes the `resources` row without
    clearing it. Under MySQL that is `IntegrityError 1451` from
    `ingestion_jobs_ibfk_1`, which with no global exception handler surfaces as an
    uncaught 500 — so the only way to remove a knowledge base is the soft delete, and the row,
    its chunks and its bytes on disk stay forever.

    This is a *deployment* defect in the sense that matters here: it exists only against the
    database that ships. The hermetic integration suite runs on in-memory SQLite, which
    ignores foreign keys by default, so the same call succeeds there and always will.

    The test asserts the behaviour the endpoint promises, and is marked xfail so the suite
    stays green while the bug is open. `rag/jobs.py::_cancel` already demonstrates the fix —
    it nulls `resource_id` on the job row before discarding the base, precisely so the record
    that the upload happened survives the deletion.
    """
    resource_id, _files = uploaded_source

    response = admin.delete(f"/resources/brain/{resource_id}/permanent")

    assert response.status_code != 500, (
        "permanent delete returned a bare 500 — FK 1451 from ingestion_jobs"
    )
    assert response.status_code in (200, 204, 404)
