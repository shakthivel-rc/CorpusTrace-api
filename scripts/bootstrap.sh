#!/usr/bin/env bash
#
# One command to go from a fresh clone to a running NexaRAG.
#
#   ./scripts/bootstrap.sh              # Docker: brings up MySQL, the API and the SPA
#   ./scripts/bootstrap.sh --native     # No Docker: venv + npm + your own MySQL
#
# Safe to re-run. It never overwrites a value you already have in .env — it only fills in
# what is missing — so running it against a configured checkout is a no-op plus a rebuild.
#
# Why this exists: the project is two repositories with no parent, and the API half needs a
# MySQL server, a schema, a seeded superadmin and a secret key before it will start at all.
# That is four manual steps and a database to install, which is three too many for someone
# who just wants to see it run.
set -euo pipefail

API_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_DIR="$(cd "$API_DIR/.." && pwd)/Nexarag-app"
APP_REPO="${NEXARAG_APP_REPO:-https://github.com/SHAKTHI-HACKER/Nexarag-app.git}"
ENV_FILE="$API_DIR/.env"

MODE="docker"
[[ "${1:-}" == "--native" ]] && MODE="native"
[[ "${1:-}" == "--help" || "${1:-}" == "-h" ]] && { sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'; exit 0; }

bold() { printf '\033[1m%s\033[0m\n' "$*"; }
info() { printf '  %s\n' "$*"; }
warn() { printf '\033[33m  ! %s\033[0m\n' "$*"; }
die()  { printf '\033[31m  ✗ %s\033[0m\n' "$*" >&2; exit 1; }

need() { command -v "$1" >/dev/null 2>&1 || die "$1 is required but not installed."; }

# SECRET_KEY is not an arbitrary random string. `utils/token.py` passes it straight to
# `Fernet(key)` at *import* time, so it must be exactly 32 bytes, urlsafe-base64 encoded —
# anything else raises "Fernet key must be 32 url-safe base64-encoded bytes" and the app
# dies before uvicorn binds. A hex string of the same entropy is not accepted.
generate_fernet_key() {
    if command -v python3 >/dev/null 2>&1; then
        python3 -c 'import base64, os; print(base64.urlsafe_b64encode(os.urandom(32)).decode())'
    else
        openssl rand -base64 32 | tr '+/' '-_'
    fi
}

# Passwords, by contrast, must be hex. They are interpolated into
# `mysql+pymysql://user:PASSWORD@db:3306/name`, and a password containing @ : / or ? would
# silently corrupt that URL into something that parses but points somewhere else.
generate_password() {
    if command -v openssl >/dev/null 2>&1; then
        openssl rand -hex 24
    else
        python3 -c 'import secrets; print(secrets.token_hex(24))'
    fi
}

# Give KEY a value if it does not already have one. A key that already holds a real value is
# never touched — this script must be safe to run against a checkout holding live credentials.
#
# "Has a value" is deliberately not the same question as "the line exists". `.env.example`
# ships documented-but-blank placeholders (`SECRET_KEY=`, `SUPERADMIN_PASSWORD=`), and the
# first version of this function tested only for the key, so on a fresh clone it decided both
# were already configured and generated neither. `docker compose` then failed on
# `${SECRET_KEY:?}` — which errors on an EMPTY value exactly as it does on an unset one — and
# the seeder would have skipped silently, leaving no account to sign in with.
ensure_env() {
    local key="$1" value="$2"

    if grep -qE "^${key}=." "$ENV_FILE" 2>/dev/null; then
        return 0
    fi

    if grep -qE "^${key}=" "$ENV_FILE" 2>/dev/null; then
        # Rewrite the blank placeholder where it stands rather than appending a second
        # assignment. Two lines for one key is ambiguous, and which one wins is not agreed
        # between docker compose and python-dotenv — so the file would mean different things
        # to the container and to a native run.
        #
        # awk via ENVIRON, not sed: these values contain `=`, `!` and `/`, all of which need
        # escaping in a sed replacement and none of which need any in an awk variable.
        local tmp
        tmp="$(mktemp "${ENV_FILE}.XXXXXX")"
        NX_KEY="$key" NX_VALUE="$value" awk '
            BEGIN { key = ENVIRON["NX_KEY"]; value = ENVIRON["NX_VALUE"]; filled = 0 }
            !filled && index($0, key "=") == 1 { print key "=" value; filled = 1; next }
            { print }
        ' "$ENV_FILE" > "$tmp"
        mv "$tmp" "$ENV_FILE"
        info "set ${key} (was blank)"
        return 0
    fi

    printf '%s=%s\n' "$key" "$value" >> "$ENV_FILE"
    info "set ${key} (was missing)"
}

# Fail here, with the name of the thing that is wrong, rather than letting compose report it
# as an interpolation error four steps later.
require_env_value() {
    local key="$1"
    grep -qE "^${key}=." "$ENV_FILE" 2>/dev/null \
        || die "${key} has no value in .env — remove the line and re-run to have one generated."
}

# ---------------------------------------------------------------------------------------
bold "NexaRAG bootstrap (${MODE})"

# 1. The other half of the project ------------------------------------------------------
if [[ ! -d "$APP_DIR" ]]; then
    need git
    bold "1. Fetching the SPA"
    info "Nexarag-app is a separate repository; cloning it beside this one"
    git clone --depth 1 "$APP_REPO" "$APP_DIR" \
        || die "could not clone $APP_REPO — clone it manually to $APP_DIR and re-run."
else
    bold "1. SPA already present at $APP_DIR"
fi

# 2. Configuration ----------------------------------------------------------------------
bold "2. Configuration"
if [[ ! -f "$ENV_FILE" ]]; then
    cp "$API_DIR/.env.example" "$ENV_FILE"
    info "created .env from .env.example"
else
    info ".env already exists — filling in only what is missing"
fi

# SECRET_KEY signs JWTs, encrypts reset links AND is the fallback key for stored LLM
# credentials. Generated once and never regenerated: rotating it would invalidate every
# outstanding session and make every saved provider key undecryptable (CLAUDE.md §13).
ensure_env SECRET_KEY            "$(generate_fernet_key)"
ensure_env MYSQL_DATABASE        "nexarag"
ensure_env MYSQL_USER            "nexarag"
ensure_env MYSQL_PASSWORD        "$(generate_password)"
ensure_env MYSQL_ROOT_PASSWORD   "$(generate_password)"
ensure_env MYSQL_PORT            "3307"
ensure_env API_PORT              "8000"
ensure_env APP_PORT              "8080"
# Must satisfy the password policy the API enforces on every entry point: 12+ characters
# with upper, lower, digit, symbol and no whitespace (CLAUDE.md §7, rule 5). A bare random
# hex string has no uppercase and no symbol, so the seeder would reject it.
ensure_env SUPERADMIN_PASSWORD   "Nx!$(generate_password | cut -c1-12)Aa1"

# The two the stack genuinely cannot start without. SECRET_KEY is fatal at compose
# interpolation; SUPERADMIN_PASSWORD fails quietly, by seeding nothing and leaving no way in.
require_env_value SECRET_KEY
require_env_value SUPERADMIN_PASSWORD

chmod 600 "$ENV_FILE"

# ---------------------------------------------------------------------------------------
if [[ "$MODE" == "native" ]]; then
    bold "3. Python environment"
    need python3
    [[ -d "$API_DIR/.venv" ]] || python3 -m venv "$API_DIR/.venv"
    # shellcheck disable=SC1091
    source "$API_DIR/.venv/bin/activate"
    pip install --quiet --upgrade pip
    pip install --quiet -r "$API_DIR/requirements.txt" -r "$API_DIR/requirements-dev.txt" \
        || die "pip install failed — mysqlclient needs libmysqlclient-dev and pkg-config."
    info "dependencies installed"

    bold "4. Database"
    warn "native mode uses the DATABASE_URL already in your .env — make sure that server is running"
    ( cd "$API_DIR" && alembic upgrade head ) || die "migrations failed; check DATABASE_URL."
    ( cd "$API_DIR" && python -m seeders.user_seeder >/dev/null 2>&1 ) \
        && info "seeded superadmin and roles" \
        || info "seeding skipped (already seeded, or SUPERADMIN_PASSWORD unset)"

    bold "5. SPA dependencies"
    need npm
    ( cd "$APP_DIR" && npm install --silent )
    info "installed"

    bold "Ready"
    info "Start both servers with:  make dev"
    info "  API  →  http://localhost:8000"
    info "  SPA  →  http://localhost:3000"
    exit 0
fi

# 3. Docker -----------------------------------------------------------------------------
need docker
docker compose version >/dev/null 2>&1 || die "the Docker Compose plugin is required (docker compose v2)."
docker info >/dev/null 2>&1 || die "the Docker daemon is not reachable — is it running, and are you in the docker group?"

bold "3. Building and starting"
info "first run pulls MySQL and builds two images; expect a few minutes"
cd "$API_DIR"
docker compose up -d --build

bold "4. Waiting for the API to report ready"
for _ in $(seq 1 60); do
    if [[ "$(docker compose ps --format '{{.Health}}' api 2>/dev/null | head -1)" == "healthy" ]]; then
        break
    fi
    sleep 2
done

if [[ "$(docker compose ps --format '{{.Health}}' api 2>/dev/null | head -1)" != "healthy" ]]; then
    warn "the API has not reported healthy yet. Recent logs:"
    docker compose logs --tail 30 api
    die "startup did not complete — 'docker compose logs -f api' has the detail."
fi

APP_PORT_VALUE="$(grep -E '^APP_PORT=' "$ENV_FILE" | cut -d= -f2)"
bold "Ready"
info "App        →  http://localhost:${APP_PORT_VALUE:-8080}"
info "API health →  http://localhost:$(grep -E '^API_PORT=' "$ENV_FILE" | cut -d= -f2)/api/v1/health/ready"
echo
# Matches DEFAULT_SUPERADMIN_EMAIL in seeders/user_seeder.py. Overridable with
# SUPERADMIN_EMAIL; this line names the default because that is what a fresh setup gets.
info "Sign in as  superadmin@nexarag.local  with the SUPERADMIN_PASSWORD value in .env:"
info "  grep '^SUPERADMIN_PASSWORD=' .env"
echo
info "Logs: make logs    ·    Stop: make down    ·    All tasks: make"
