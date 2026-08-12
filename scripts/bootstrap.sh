#!/usr/bin/env bash
#
# One command to go from a fresh clone to a running NexaRAG, on Linux, macOS or Windows.
#
#   ./scripts/bootstrap.sh              # use Docker if it can, fall back to native if not
#   ./scripts/bootstrap.sh --docker     # Docker only; fail rather than fall back
#   ./scripts/bootstrap.sh --native     # no Docker: venv + npm + a MySQL server
#   ./scripts/bootstrap.sh --check      # diagnose only; change nothing
#
#   --yes           never ask; take the recommended answer (implied with no terminal)
#   --no-install    diagnose missing packages, install none
#
# Safe to re-run. It never overwrites a value you already have in .env — it only fills in
# what is missing — so running it against a configured checkout is a no-op plus a rebuild.
#
# Why this exists: the project is two repositories with no parent, and the API half needs a
# MySQL server, a schema, a seeded superadmin and a secret key before it will start at all.
# That is four manual steps and a database to install, which is three too many for someone
# who just wants to see it run.
#
# WHAT THIS SCRIPT DOES WHEN SOMETHING IS WRONG.
#
# It repairs what it can and says so on the line where it happens, rather than reporting a
# precondition and stopping. Ports in use are moved and every URL that named them follows;
# a Docker daemon that is not running is started; a build dependency that is missing is
# installed; a Python package that will not compile is retried against the pure-Python
# driver the project also pins; a Docker installation that cannot be repaired falls back to
# a native install. What it will not do is repair something by destroying data — the one
# genuinely unrecoverable case, a database volume whose password this checkout no longer
# knows, is resolved by building a *second* stack beside it and leaving the first intact.
#
# Every repair is announced with a → and the exact command. NEXARAG_AUTO_INSTALL=0 turns
# installation off and leaves the diagnosis.
set -euo pipefail

API_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_DIR="$(cd "$API_DIR/.." && pwd)/Nexarag-app"
APP_REPO="${NEXARAG_APP_REPO:-https://github.com/SHAKTHI-HACKER/Nexarag-app.git}"
ENV_FILE="$API_DIR/.env"

# shellcheck source=lib/common.sh
. "$API_DIR/scripts/lib/common.sh"
# shellcheck source=lib/env.sh
. "$API_DIR/scripts/lib/env.sh"
# shellcheck source=lib/ports.sh
. "$API_DIR/scripts/lib/ports.sh"
# shellcheck source=lib/deps.sh
. "$API_DIR/scripts/lib/deps.sh"

MODE="auto"
CHECK_ONLY=0

while [ "$#" -gt 0 ]; do
    case "$1" in
        --native) MODE="native" ;;
        --docker) MODE="docker" ;;
        --check|--doctor) CHECK_ONLY=1; NX_DRY_RUN=1; export NX_DRY_RUN ;;
        --yes|-y) NEXARAG_ASSUME_YES=1; export NEXARAG_ASSUME_YES ;;
        --no-install) NEXARAG_AUTO_INSTALL=0; export NEXARAG_AUTO_INSTALL ;;
        --help|-h) sed -n '2,14p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) nx_die "unknown option: $1 (try --help)" ;;
    esac
    shift
done

# Python's minimum is 3.10 because the code annotates with `X | None` at module import
# time, which is a SyntaxError before then. 3.12 is what the Docker image runs and what
# `math.sumprod` in the retrieval hot path wants, so it is preferred but not required —
# rag/service.py carries a fallback for exactly this.
PYTHON_MINIMUM="3.10"
PYTHON_PREFERRED="3.12"
# Not a preference: package.json declares `engines: >=24.0.0 <25` and .nvmrc says 24.
NODE_MINIMUM="24"

# SECRET_KEY is not an arbitrary random string. `utils/token.py` passes it straight to
# `Fernet(key)` at *import* time, so it must be exactly 32 bytes, urlsafe-base64 encoded —
# anything else raises "Fernet key must be 32 url-safe base64-encoded bytes" and the app
# dies before uvicorn binds. A hex string of the same entropy is not accepted.
generate_fernet_key() {
    if nx_have python3; then
        python3 -c 'import base64, os; print(base64.urlsafe_b64encode(os.urandom(32)).decode())'
    elif nx_have openssl; then
        openssl rand -base64 32 | tr '+/' '-_'
    else
        nx_die "neither python3 nor openssl is available to generate a key. Install either and re-run."
    fi
}

# Passwords, by contrast, must be hex. They are interpolated into
# `mysql+pymysql://user:PASSWORD@db:3306/name`, and a password containing @ : / or ? would
# silently corrupt that URL into something that parses but points somewhere else.
generate_password() {
    if nx_have openssl; then
        openssl rand -hex 24
    elif nx_have python3; then
        python3 -c 'import secrets; print(secrets.token_hex(24))'
    else
        nx_die "neither openssl nor python3 is available to generate a password."
    fi
}

# Is this string something Fernet will accept? `utils/token.py` does `Fernet(os.getenv(
# "SECRET_KEY"))` at module import, so a wrong-shaped value is not a subtle
# misconfiguration — it is an ImportError before uvicorn binds, on every entry point.
valid_fernet_key() {
    local key="$1"
    nx_have python3 || {
        # No interpreter to check with: fall back to the shape Fernet requires — 32 bytes
        # of urlsafe base64 is always 44 characters ending in '='.
        case "$key" in
            ????????????????????????????????????????????=) return 0 ;;
            *) return 1 ;;
        esac
    }
    NX_KEY_CHECK="$key" python3 -c '
import base64, os, sys
try:
    if len(base64.urlsafe_b64decode(os.environ["NX_KEY_CHECK"])) != 32:
        sys.exit(1)
except Exception:
    sys.exit(1)
' 2>/dev/null
}

# Where setup finishes, whichever route it took.
nx_summary() {
    local app_url="$1" health_url="$2" native="${3:-}"

    nx_step "Ready"
    if [ -n "$native" ]; then
        nx_info "Start both servers with:  make dev"
    fi
    nx_info "App        →  ${app_url}"
    nx_info "API health →  ${health_url}"
    printf '\n' >&2
    # Matches DEFAULT_SUPERADMIN_EMAIL in seeders/user_seeder.py. Overridable with
    # SUPERADMIN_EMAIL; this line names the default because that is what a fresh setup gets.
    nx_info "Sign in as  superadmin@nexarag.local  with the SUPERADMIN_PASSWORD value in .env:"
    nx_info "  grep '^SUPERADMIN_PASSWORD=' .env"
    printf '\n' >&2
    if [ -n "$native" ]; then
        nx_dim "Tests: make test    ·    All tasks: make"
    else
        nx_dim "Logs: make logs    ·    Stop: make down    ·    All tasks: make"
    fi
}

nx_bold "NexaRAG setup"
nx_dim "$(nx_os)/$(nx_arch) · package manager: $(nx_pkg_manager) · bash ${BASH_VERSION%%(*}"

# =========================================================================================
nx_step "1. The other half of the project"

# `package.json`, not the directory. An empty or half-cloned Nexarag-app is a directory
# that exists, and treating that as "present" pushed the failure four steps down the
# script, where it arrived as `npm ci` failing on a folder with nothing in it — three
# fallbacks deep, with an error about dependencies rather than about a missing repository.
# An interrupted clone leaves exactly this state.
if [ -f "$APP_DIR/package.json" ]; then
    nx_ok "SPA already present at $APP_DIR"
elif [ -d "$APP_DIR" ] && [ -n "$(ls -A "$APP_DIR" 2>/dev/null || true)" ]; then
    nx_die "$APP_DIR exists but has no package.json, so it is not a usable checkout of the SPA.
       An interrupted clone looks exactly like this. Remove the directory and re-run, or
       clone $APP_REPO into it yourself."
elif [ "$CHECK_ONLY" = "1" ]; then
    nx_warn "SPA is missing; setup would clone it from $APP_REPO"
else
    nx_ensure_command git git || nx_die "git is required to fetch the SPA and could not be installed.
       Install git, or clone $APP_REPO to $APP_DIR yourself, then re-run."
    nx_info "Nexarag-app is a separate repository; cloning it beside this one"
    # Retried because a clone is the one step here that is pure network, and the failure is
    # almost always a proxy or a flaky DNS answer rather than a wrong URL.
    nx_retry 3 git clone --depth 1 "$APP_REPO" "$APP_DIR" \
        || nx_die "could not clone $APP_REPO — clone it manually to $APP_DIR and re-run."
    nx_ok "cloned"
fi

# =========================================================================================
nx_step "2. Configuration"

if [ ! -f "$ENV_FILE" ]; then
    [ "$CHECK_ONLY" = "1" ] && nx_die "no .env yet — run without --check to create one."
    cp "$API_DIR/.env.example" "$ENV_FILE"
    nx_ok "created .env from .env.example"
else
    nx_info ".env already exists — filling in only what is missing"
fi

if [ "$CHECK_ONLY" = "0" ]; then
    # SECRET_KEY signs JWTs, encrypts reset links AND is the fallback key for stored LLM
    # credentials. Generated once and never regenerated: rotating it would invalidate every
    # outstanding session and make every saved provider key undecryptable (CLAUDE.md §13).
    nx_env_ensure SECRET_KEY          "$(generate_fernet_key)"   "$ENV_FILE"
    # The MYSQL_* keys are deliberately NOT settled here. Generating a password is the last
    # resort for them, not the first: if a database already exists for this project, the
    # only password that can open it is the one it was created with, and that one may still
    # be recoverable. See step 4.
    # Must satisfy the password policy the API enforces on every entry point: 12+ characters
    # with upper, lower, digit, symbol and no whitespace (CLAUDE.md §7, rule 5). A bare random
    # hex string has no uppercase and no symbol, so the seeder would reject it.
    nx_env_ensure SUPERADMIN_PASSWORD "Nx!$(generate_password | cut -c1-12)Aa1" "$ENV_FILE"

    # A SECRET_KEY that is present but not a Fernet key is the most common hand-edit
    # failure here — `SECRET_KEY=changeme` looks configured, satisfies every "is it set?"
    # check including this script's own, and then kills the process at import with a
    # message about base64. Replacing it is provably safe *because* it is invalid: Fernet
    # rejects it at import, so the app has never run, so nothing has ever been encrypted
    # with it. A valid key is never touched, for the opposite reason.
    if ! valid_fernet_key "$(nx_env_get SECRET_KEY "$ENV_FILE")"; then
        nx_warn "SECRET_KEY is set but is not a valid Fernet key (32 bytes, urlsafe base64)"
        nx_fix "replacing it — nothing can have been encrypted with a key the app cannot load"
        nx_env_write SECRET_KEY "$(generate_fernet_key)" "$ENV_FILE"
    fi

    # The two the stack genuinely cannot start without. SECRET_KEY is fatal at compose
    # interpolation; SUPERADMIN_PASSWORD fails quietly, by seeding nothing and leaving no way in.
    nx_env_require SECRET_KEY "$ENV_FILE"
    nx_env_require SUPERADMIN_PASSWORD "$ENV_FILE"

    chmod 600 "$ENV_FILE" 2>/dev/null || nx_warn "could not chmod .env (harmless on Windows filesystems)"
fi

PROJECT="$(nx_env_get COMPOSE_PROJECT_NAME "$ENV_FILE")"
[ -n "$PROJECT" ] || PROJECT="nexarag"

# =========================================================================================
nx_step "3. Toolchain"

USE_DOCKER=0
case "$MODE" in
    docker)
        nx_docker_ready || nx_die "Docker was requested with --docker but is not usable here.
       Start Docker and re-run, drop the flag to fall back automatically, or use:
       make setup-native"
        USE_DOCKER=1
        ;;
    native)
        nx_info "native mode requested; not looking for Docker"
        ;;
    auto)
        if nx_docker_ready; then
            USE_DOCKER=1
            nx_ok "Docker is usable ($NX_COMPOSE)"
        else
            # This is the fallback that makes --docker's failure message honest. Native
            # needs more from the machine, not less, so it is the second choice rather
            # than the safe one — but it needs no daemon, no root and no group membership,
            # which is exactly what is missing when Docker is unusable.
            nx_warn "Docker is not usable on this machine"
            nx_fix "falling back to a native install (Python venv + npm + MySQL)"
        fi
        ;;
esac

# =========================================================================================
nx_step "4. Database credentials"

# Recover before generating. A password this script invents is worthless against a data
# directory that already exists — MySQL only reads MYSQL_PASSWORD while initialising an
# empty one — so the order matters more than it looks: generate first and the recovery
# never runs, because the key now has a value.
if [ "$CHECK_ONLY" = "1" ]; then
    if [ -z "$(nx_env_get MYSQL_PASSWORD "$ENV_FILE")" ]; then
        nx_warn "MYSQL_PASSWORD has no value in .env — 'docker compose' cannot interpolate this file,"
        nx_info "  so make up/down/logs/restart all fail. Setup would recover it from the existing"
        nx_info "  container if there is one, and generate a new one if there is not."
    else
        nx_ok "database credentials present"
    fi
else
    if [ -z "$(nx_env_get MYSQL_PASSWORD "$ENV_FILE")" ]; then
        nx_recover_db_credentials "$PROJECT" "$ENV_FILE" || true
    fi
    nx_env_ensure MYSQL_DATABASE      "nexarag"              "$ENV_FILE"
    nx_env_ensure MYSQL_USER          "nexarag"              "$ENV_FILE"
    nx_env_ensure MYSQL_PASSWORD      "$(generate_password)" "$ENV_FILE"
    nx_env_ensure MYSQL_ROOT_PASSWORD "$(generate_password)" "$ENV_FILE"
    nx_ok "database credentials present"
fi

# =========================================================================================
nx_step "5. Ports"

# Ports are settled AFTER the mode is known, because "is this port free?" has a different
# answer when the thing holding it is this project's own container from a previous run.
if [ "$USE_DOCKER" = "1" ]; then
    OLD_APP_PORT="$(nx_env_get APP_PORT "$ENV_FILE")"; [ -n "$OLD_APP_PORT" ] || OLD_APP_PORT=8080
    OLD_API_PORT="$(nx_env_get API_PORT "$ENV_FILE")"; [ -n "$OLD_API_PORT" ] || OLD_API_PORT=8000

    # `nx_settle_port` returns through NX_PORT rather than stdout — see the note on it.
    nx_settle_port MYSQL_PORT 3307 "the database" "$ENV_FILE" "$PROJECT" \
        || nx_die "could not find a free port for MySQL."
    MYSQL_PORT_VALUE="$NX_PORT"
    nx_settle_port API_PORT "$OLD_API_PORT" "the API" "$ENV_FILE" "$PROJECT" \
        || nx_die "could not find a free port for the API."
    API_PORT_VALUE="$NX_PORT"
    nx_settle_port APP_PORT "$OLD_APP_PORT" "the app" "$ENV_FILE" "$PROJECT" \
        || nx_die "could not find a free port for the app."
    APP_PORT_VALUE="$NX_PORT"

    # Both of these name the app's port, and both are read by something that cannot be
    # told to look somewhere else afterwards: APP_BASE_URL goes into emailed invite and
    # reset links, and CORS_ORIGINS decides whether a browser may talk to the API at all.
    nx_env_ensure APP_BASE_URL  "http://localhost:${APP_PORT_VALUE}" "$ENV_FILE"
    nx_env_ensure CORS_ORIGINS  "http://localhost:3000,http://localhost:${APP_PORT_VALUE}" "$ENV_FILE"
    nx_env_retarget_port APP_BASE_URL "$OLD_APP_PORT" "$APP_PORT_VALUE" "$ENV_FILE"
    nx_env_retarget_port CORS_ORIGINS "$OLD_APP_PORT" "$APP_PORT_VALUE" "$ENV_FILE"
else
    OLD_API_PORT="$(nx_env_get API_PORT "$ENV_FILE")"; [ -n "$OLD_API_PORT" ] || OLD_API_PORT=8000
    OLD_DEV_PORT="$(nx_env_get APP_DEV_PORT "$ENV_FILE")"; [ -n "$OLD_DEV_PORT" ] || OLD_DEV_PORT=3000

    nx_settle_port API_PORT "$OLD_API_PORT" "the API" "$ENV_FILE" "" \
        || nx_die "could not find a free port for the API."
    API_PORT_VALUE="$NX_PORT"
    nx_settle_port APP_DEV_PORT "$OLD_DEV_PORT" "the dev server" "$ENV_FILE" "" \
        || nx_die "could not find a free port for the Vite dev server."
    APP_DEV_PORT_VALUE="$NX_PORT"

    nx_env_ensure APP_BASE_URL "http://localhost:${APP_DEV_PORT_VALUE}" "$ENV_FILE"
    nx_env_ensure CORS_ORIGINS "http://localhost:${APP_DEV_PORT_VALUE}" "$ENV_FILE"
    nx_env_retarget_port APP_BASE_URL "$OLD_DEV_PORT" "$APP_DEV_PORT_VALUE" "$ENV_FILE"
    nx_env_retarget_port CORS_ORIGINS "$OLD_DEV_PORT" "$APP_DEV_PORT_VALUE" "$ENV_FILE"
fi

if [ "$CHECK_ONLY" = "1" ]; then
    nx_step "Diagnosis complete"
    nx_info "Nothing was changed. Run without --check to apply."
    exit 0
fi

cd "$API_DIR"

# =========================================================================================
if [ "$USE_DOCKER" = "1" ]; then
    . "$API_DIR/scripts/lib/run_docker.sh"
    nx_run_docker
else
    . "$API_DIR/scripts/lib/run_native.sh"
    nx_run_native
fi
