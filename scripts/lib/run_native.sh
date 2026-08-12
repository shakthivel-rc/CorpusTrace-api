# shellcheck shell=bash
#
# Installing and starting the stack without Docker, and recovering from the ways that goes
# wrong. Sourced by bootstrap.sh once it has settled .env and the ports.
#
# Native is harder than Docker, not easier: it needs a Python of the right vintage, a C
# toolchain for one of the pinned wheels, a Node of the right major, and a MySQL server.
# Every one of those is a rung this file can climb down from.
#
# This file is held to the portability contract at the top of common.sh.

[ -n "${NX_RUN_NATIVE_SH:-}" ] && return 0
NX_RUN_NATIVE_SH=1

NX_VENV="" NX_VENV_PY="" NX_VENV_PIP=""

# --- python -------------------------------------------------------------------------------

nx_native_python() {
    local interpreter="" version=""

    if interpreter="$(nx_find_python "$PYTHON_PREFERRED")"; then
        nx_ok "using $interpreter ($(nx_version_of "$interpreter" -V))"
    elif interpreter="$(nx_find_python "$PYTHON_MINIMUM")"; then
        version="$(nx_version_of "$interpreter" -V)"
        nx_warn "Python ${version} is below the ${PYTHON_PREFERRED} the Docker image runs"
        nx_info "  It is above the ${PYTHON_MINIMUM} floor so everything works; semantic retrieval"
        nx_info "  falls back to a Python loop where 3.12 would use math.sumprod."
    else
        nx_warn "no Python ${PYTHON_MINIMUM} or newer on this machine"
        nx_install python python-venv || nx_die "could not install Python.
       Install Python ${PYTHON_PREFERRED} yourself and re-run."
        interpreter="$(nx_find_python "$PYTHON_MINIMUM")" \
            || nx_die "Python still not found after installing. Install ${PYTHON_PREFERRED} and re-run."
        nx_ok "using $interpreter"
    fi

    NX_VENV="$API_DIR/.venv"
    if [ ! -x "$NX_VENV/bin/python" ] && [ ! -x "$NX_VENV/Scripts/python.exe" ]; then
        if ! "$interpreter" -m venv "$NX_VENV" >/dev/null 2>&1; then
            # Debian and Ubuntu ship the venv module in a separate package, and the error
            # without it — "ensurepip is not available" — names a module nobody has heard
            # of rather than the package to install.
            nx_warn "creating the virtualenv failed (Debian and Ubuntu split python3-venv into its own package)"
            nx_install python-venv || nx_die "could not install the venv module."
            "$interpreter" -m venv "$NX_VENV" \
                || nx_die "could not create a virtualenv at $NX_VENV."
        fi
        nx_ok "virtualenv created"
    else
        nx_ok "virtualenv already present"
    fi

    # Windows layouts a venv as Scripts/, everything else as bin/.
    if [ -x "$NX_VENV/bin/python" ]; then
        NX_VENV_PY="$NX_VENV/bin/python"
    else
        NX_VENV_PY="$NX_VENV/Scripts/python.exe"
    fi
    NX_VENV_PIP="$NX_VENV_PY -m pip"
}

# --- dependencies ---------------------------------------------------------------------------

nx_native_pip() {
    local filtered="" url=""

    # shellcheck disable=SC2086
    $NX_VENV_PIP install --quiet --upgrade pip >/dev/null 2>&1 || true

    # shellcheck disable=SC2086
    if $NX_VENV_PIP install --quiet -r "$API_DIR/requirements.txt" -r "$API_DIR/requirements-dev.txt"; then
        nx_ok "Python dependencies installed"
        return 0
    fi

    # One package in the list is a C extension. `mysqlclient` links against the MySQL client
    # library and needs a compiler, the headers and pkg-config to build — and pip's error
    # for a missing header is a hundred lines of compiler output ending in
    # "Exception: Can not find valid pkg-config name", which does not name a package either.
    nx_warn "installing the pinned packages failed; the usual cause is mysqlclient having nothing to compile against"
    if nx_install build-tools mysql-dev pkg-config python-dev; then
        # shellcheck disable=SC2086
        if $NX_VENV_PIP install --quiet -r "$API_DIR/requirements.txt" -r "$API_DIR/requirements-dev.txt"; then
            nx_ok "Python dependencies installed"
            return 0
        fi
    fi

    # Last rung, and a genuine one rather than a shrug: the project pins *two* MySQL
    # drivers, and every DATABASE_URL it writes uses `mysql+pymysql`. PyMySQL is pure
    # Python, so it needs no compiler at all. Dropping mysqlclient costs some throughput on
    # a driver nothing here selects, and it is the difference between a working install and
    # no install on a machine with no toolchain.
    nx_warn "mysqlclient still will not build"
    nx_fix "installing without it — PyMySQL is also pinned, and is the driver every DATABASE_URL here uses"

    filtered="$(mktemp "${TMPDIR:-/tmp}/nexarag-req.XXXXXX")"
    grep -v -i '^mysqlclient' "$API_DIR/requirements.txt" > "$filtered"
    # shellcheck disable=SC2086
    $NX_VENV_PIP install --quiet -r "$filtered" -r "$API_DIR/requirements-dev.txt" || {
        rm -f "$filtered"
        nx_die "Python dependencies could not be installed. Re-run without --no-install, or
       install them by hand:  .venv/bin/pip install -r requirements.txt"
    }
    rm -f "$filtered"

    # Without mysqlclient, a URL naming it resolves to nothing at import time and the app
    # dies with "No module named 'MySQLdb'" — which reads as a broken install rather than a
    # driver that was deliberately left out.
    url="$(nx_env_get DATABASE_URL "$ENV_FILE")"
    case "$url" in
        mysql://*|mysql+mysqldb://*)
            nx_env_write DATABASE_URL "mysql+pymysql://${url#*://}" "$ENV_FILE"
            nx_fix "DATABASE_URL switched to the mysql+pymysql driver"
            ;;
    esac
    nx_ok "Python dependencies installed (without mysqlclient)"
}

# --- database -------------------------------------------------------------------------------

# Can the configured DATABASE_URL be connected to?
#
# The exception *type* is printed and never the exception, because SQLAlchemy's connection
# errors quote the URL back — password included — and this output goes into whatever the
# operator pastes into a bug report.
nx_database_reachable() {
    local url=""
    url="$(nx_env_get DATABASE_URL "$ENV_FILE")"
    [ -n "$url" ] || return 1
    NX_DB_URL="$url" "$NX_VENV_PY" -c '
import os, sys
from sqlalchemy import create_engine, text
try:
    with create_engine(os.environ["NX_DB_URL"], pool_pre_ping=True).connect() as connection:
        connection.execute(text("SELECT 1"))
except Exception as exc:
    print(type(exc).__name__, file=sys.stderr)
    sys.exit(1)
' 2>/dev/null
}

# Start only the database service in Docker and point .env at it.
#
# A native app against a containerised database is a completely ordinary way to work — it
# is what `make dev` wants anyway — and it removes the hardest part of a native install
# (installing, initialising and securing a MySQL server) on any machine that has Docker but
# could not use it for the whole stack.
nx_database_via_docker() {
    local port="" waited=0 container=""

    nx_docker_ready || return 1

    nx_settle_port MYSQL_PORT 3307 "the database" "$ENV_FILE" "$PROJECT" || return 1
    port="$NX_PORT"

    nx_fix "starting MySQL in Docker on port ${port} (the app itself stays native)"
    export COMPOSE_PROJECT_NAME="$PROJECT"
    # shellcheck disable=SC2086
    $NX_COMPOSE up -d db >/dev/null 2>&1 || return 1

    while [ "$waited" -lt 120 ]; do
        # `|| true` for the same reason as nx_api_container: a plain assignment under
        # `set -e -o pipefail` turns "nothing matched yet" into "exit the script".
        container="$($NX_DOCKER ps -q --filter "label=com.docker.compose.project=${PROJECT}" --filter "label=com.docker.compose.service=db" 2>/dev/null | head -1 || true)"
        if [ -n "$container" ] \
            && [ "$($NX_DOCKER inspect --format '{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}' "$container" 2>/dev/null)" = "healthy" ]; then
            break
        fi
        sleep 3
        waited=$((waited + 3))
    done

    nx_env_write DATABASE_URL \
        "mysql+pymysql://$(nx_env_get MYSQL_USER "$ENV_FILE"):$(nx_env_get MYSQL_PASSWORD "$ENV_FILE")@127.0.0.1:${port}/$(nx_env_get MYSQL_DATABASE "$ENV_FILE")" \
        "$ENV_FILE"
    nx_database_reachable
}

# Install a MySQL server from the package manager and create the database and user.
nx_database_via_package() {
    local manager="" prefix="" sql="" client="mysql"

    manager="$(nx_pkg_manager)"
    case "$manager" in
        apt|dnf|yum|brew) ;;
        *)
            nx_warn "no reliable MySQL server package is known for '${manager}'."
            return 1
            ;;
    esac

    if ! nx_have mysqld && ! nx_have mysql; then
        nx_install mysql-server || return 1
    else
        # A MySQL server that is already here belongs to somebody, and this step adds a
        # database and a user to it. That is small and reversible, but it is a change to
        # shared state outside the checkout, so it is the one repair in this script that
        # asks first when there is anybody to ask.
        nx_confirm "create the '$(nx_env_get MYSQL_DATABASE "$ENV_FILE")' database on the MySQL server already running here?" \
            || return 1
    fi

    if [ "$manager" = "brew" ]; then
        brew services start mysql >/dev/null 2>&1 || true
    else
        prefix="$(nx_sudo_prefix)" || return 1
        if nx_have systemctl; then
            # shellcheck disable=SC2086
            $prefix systemctl start mysql >/dev/null 2>&1 || $prefix systemctl start mysqld >/dev/null 2>&1 || true
        fi
        client="$prefix mysql"
    fi

    # Written to a mode-600 file rather than passed with -e: anything on a command line is
    # readable by every process on the machine through `ps`, and this line contains the
    # database password.
    sql="$(mktemp "${TMPDIR:-/tmp}/nexarag-sql.XXXXXX")"
    chmod 600 "$sql"
    {
        printf 'CREATE DATABASE IF NOT EXISTS `%s` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;\n' \
            "$(nx_env_get MYSQL_DATABASE "$ENV_FILE")"
        printf "CREATE USER IF NOT EXISTS '%s'@'localhost' IDENTIFIED BY '%s';\n" \
            "$(nx_env_get MYSQL_USER "$ENV_FILE")" "$(nx_env_get MYSQL_PASSWORD "$ENV_FILE")"
        printf "CREATE USER IF NOT EXISTS '%s'@'127.0.0.1' IDENTIFIED BY '%s';\n" \
            "$(nx_env_get MYSQL_USER "$ENV_FILE")" "$(nx_env_get MYSQL_PASSWORD "$ENV_FILE")"
        printf "GRANT ALL PRIVILEGES ON \`%s\`.* TO '%s'@'localhost';\n" \
            "$(nx_env_get MYSQL_DATABASE "$ENV_FILE")" "$(nx_env_get MYSQL_USER "$ENV_FILE")"
        printf "GRANT ALL PRIVILEGES ON \`%s\`.* TO '%s'@'127.0.0.1';\n" \
            "$(nx_env_get MYSQL_DATABASE "$ENV_FILE")" "$(nx_env_get MYSQL_USER "$ENV_FILE")"
        printf 'FLUSH PRIVILEGES;\n'
    } > "$sql"

    nx_fix "creating the database and user on the local MySQL server"
    # shellcheck disable=SC2086
    $client < "$sql" >/dev/null 2>&1 || {
        rm -f "$sql"
        nx_warn "could not create the database as the MySQL root user."
        return 1
    }
    rm -f "$sql"

    nx_env_write DATABASE_URL \
        "mysql+pymysql://$(nx_env_get MYSQL_USER "$ENV_FILE"):$(nx_env_get MYSQL_PASSWORD "$ENV_FILE")@127.0.0.1:3306/$(nx_env_get MYSQL_DATABASE "$ENV_FILE")" \
        "$ENV_FILE"
    nx_database_reachable
}

nx_native_database() {
    if nx_database_reachable; then
        nx_ok "the configured DATABASE_URL answers"
        return 0
    fi

    # .env.example ships a placeholder URL, so on a fresh clone this is not a *broken*
    # database — it is one that was never chosen. Say which of the two it is.
    case "$(nx_env_get DATABASE_URL "$ENV_FILE")" in
        *user:password@*) nx_warn "DATABASE_URL is still the example placeholder" ;;
        *) nx_warn "nothing is answering on the configured DATABASE_URL" ;;
    esac

    nx_database_via_docker && { nx_ok "database ready"; return 0; }
    nx_database_via_package && { nx_ok "database ready"; return 0; }

    nx_die "no database could be reached or created.
       Install and start MySQL 8, create a database and a user for it, and put the
       connection string in .env as DATABASE_URL. Then re-run — everything else is done."
}

# --- the rest -------------------------------------------------------------------------------

nx_native_schema() {
    # `alembic upgrade head` is idempotent, and reads DATABASE_URL from .env through
    # load_dotenv() — which reads the *current working directory*, hence the cd.
    ( cd "$API_DIR" && PATH="$(dirname "$NX_VENV_PY"):$PATH" "$NX_VENV_PY" -m alembic upgrade head ) \
        || nx_die "migrations failed. The database answered but the schema could not be applied;
       'alembic current' from ${API_DIR} with the venv active will say why."
    nx_ok "schema up to date"

    if ( cd "$API_DIR" && "$NX_VENV_PY" -m seeders.user_seeder >/dev/null 2>&1 ); then
        nx_ok "seeded the superadmin and default roles"
    else
        nx_dim "seeding skipped (already seeded, or SUPERADMIN_PASSWORD unset)"
    fi
}

nx_native_spa() {
    local node_command=""

    if ! node_command="$(nx_find_node "$NODE_MINIMUM")"; then
        nx_warn "Node ${NODE_MINIMUM} is required by package.json (engines) and .nvmrc"
        nx_install node || nx_die "could not install Node.
       Install Node ${NODE_MINIMUM} — nvm or fnm is the least invasive way — and re-run."
        nx_find_node "$NODE_MINIMUM" >/dev/null \
            || nx_warn "the installed Node is older than ${NODE_MINIMUM}; the build may still work, but this is unsupported"
    fi
    nx_ok "using node $(nx_version_of node -v)"

    # `npm ci` first, not `npm install`. It replays package-lock.json exactly, which makes
    # the install reproducible AND sidesteps .npmrc's `min-release-age` cooldown — that
    # applies to *resolution*, so `npm install` can exit non-zero over a package published
    # in the last three days while `npm ci` is unaffected.
    ( cd "$APP_DIR" && npm ci --silent ) && { nx_ok "SPA dependencies installed"; return 0; }

    nx_warn "npm ci failed"
    nx_fix "clearing node_modules and trying once more"
    rm -rf "$APP_DIR/node_modules"
    ( cd "$APP_DIR" && npm ci --silent ) && { nx_ok "SPA dependencies installed"; return 0; }

    # A lockfile out of step with package.json is the one thing `npm ci` refuses outright
    # and `npm install` fixes.
    nx_fix "falling back to npm install (the lockfile may be out of step with package.json)"
    ( cd "$APP_DIR" && npm install --silent ) && { nx_ok "SPA dependencies installed"; return 0; }

    nx_fix "retrying with --legacy-peer-deps"
    ( cd "$APP_DIR" && npm install --silent --legacy-peer-deps ) \
        || nx_die "npm could not install the SPA's dependencies. Run 'npm install' in ${APP_DIR} to see why."
    nx_ok "SPA dependencies installed (with --legacy-peer-deps)"
}

nx_run_native() {
    nx_step "6. Python environment"
    nx_native_python
    nx_native_pip

    nx_step "7. Database"
    nx_native_database
    nx_native_schema

    nx_step "8. SPA dependencies"
    nx_native_spa

    nx_summary "http://localhost:${APP_DEV_PORT_VALUE}" "http://localhost:${API_PORT_VALUE}/api/v1/health/ready" "native"
}
