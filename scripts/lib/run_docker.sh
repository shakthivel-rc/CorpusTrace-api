# shellcheck shell=bash
#
# Bringing the stack up in Docker, and recovering from the ways that goes wrong.
#
# Sourced by bootstrap.sh, which has already settled .env and the ports and has set
# NX_DOCKER, NX_COMPOSE, PROJECT and the *_PORT_VALUE variables.
#
# This file is held to the portability contract at the top of common.sh.

[ -n "${NX_RUN_DOCKER_SH:-}" ] && return 0
NX_RUN_DOCKER_SH=1

# MySQL 8.0 publishes both amd64 and arm64, so this list exists for the case where a
# specific tag has been yanked or a registry mirror is incomplete — not as a routine
# fallback. It contains only MySQL: swapping in MariaDB would be changing the database the
# schema was written and migrated against, which is a decision for a person, not a repair.
NX_MYSQL_IMAGES="mysql:8.0 mysql:8.4 mysql:9"

# Pick an image tag the local architecture can actually run.
#
# On Apple Silicon a missing arm64 manifest fails at `docker compose up` with "no matching
# manifest for linux/arm64/v8", ~40 seconds in and with the other containers already
# started. `docker manifest inspect` answers the same question in one round trip.
nx_choose_mysql_image() {
    local configured="" candidate="" platform=""

    configured="$(nx_env_get MYSQL_IMAGE "$ENV_FILE")"
    if [ -n "$configured" ]; then
        nx_dim "database image: ${configured} (set in .env)"
        return 0
    fi

    if [ "$(nx_arch)" != "arm64" ]; then
        return 0
    fi

    platform="linux/arm64"
    for candidate in $NX_MYSQL_IMAGES; do
        # A manifest command that fails outright (no network, registry down, an old Docker
        # without the subcommand) tells us nothing about the tag, so it must not be read as
        # "unsupported" — bail out and let compose use the default and report properly.
        $NX_DOCKER manifest inspect "$candidate" >/dev/null 2>&1 || {
            [ "$candidate" = "mysql:8.0" ] && return 0
            continue
        }
        if $NX_DOCKER manifest inspect "$candidate" 2>/dev/null | grep -q 'arm64'; then
            if [ "$candidate" != "mysql:8.0" ]; then
                nx_warn "mysql:8.0 has no ${platform} image on this registry"
                nx_fix "using ${candidate} instead"
                nx_env_write MYSQL_IMAGE "$candidate" "$ENV_FILE"
            fi
            return 0
        fi
    done
    nx_warn "no MySQL tag with an ${platform} image was found; compose will use the default and report what it finds"
}

# A generated MYSQL_PASSWORD cannot work against a database volume that already exists.
#
# MySQL honours MYSQL_USER/MYSQL_PASSWORD *only* while initialising an empty data
# directory; afterwards the credentials live in the volume and the environment is ignored.
# The volume belongs to Docker, not to this checkout, so deleting the clone and cloning
# again leaves it behind — and the new .env then carries a password the database has never
# heard of.
#
# Left to itself that surfaces ~90 seconds later as "container nexarag-api-1 is unhealthy",
# with the real reason (1045, "Access denied for user 'nexarag'") buried in the API's log
# behind a SQLAlchemy traceback. Everything the operator can see says the *database* is
# healthy, because it is: it just does not have this password.
#
# This is the SECOND line of defence, and it only runs when the first one failed. Step 4
# already tried nx_recover_db_credentials, which reads the original password back out of
# the db container's stored environment — that repair is complete and lossless, and it
# covers every case where the container still exists, which is nearly all of them. What is
# left here is the case where the containers are gone but the volume is not: `docker
# compose down --rmi` or a container prune against a checkout whose .env was replaced.
#
# For that, the password is genuinely unrecoverable — the only copies were the .env and the
# container config, and both are gone. So the repair cannot be "use the existing database".
# It is to build a second stack beside it under its own project name. The old volume is not
# touched, not renamed and not deleted, and the message says how to reach it. Deleting data
# to make a setup script succeed would be the wrong trade in every case, including this one.
nx_resolve_volume_conflict() {
    local volume="${PROJECT}_db-data" suffix=2 candidate=""

    nx_env_generated MYSQL_PASSWORD || return 0
    $NX_DOCKER volume inspect "$volume" >/dev/null 2>&1 || return 0

    nx_warn "the database volume '${volume}' already exists and still holds the credentials it was"
    nx_info "  created with. This .env has a freshly generated MYSQL_PASSWORD, which that database has"
    nx_info "  never heard of, so the API would fail with 'Access denied for user' against a database"
    nx_info "  reporting itself perfectly healthy."

    while [ "$suffix" -lt 50 ]; do
        candidate="nexarag-${suffix}"
        $NX_DOCKER volume inspect "${candidate}_db-data" >/dev/null 2>&1 || break
        suffix=$((suffix + 1))
    done

    PROJECT="$candidate"
    nx_env_write COMPOSE_PROJECT_NAME "$PROJECT" "$ENV_FILE"
    nx_fix "starting a separate stack named '${PROJECT}' instead — nothing in '${volume}' is touched"
    nx_info "  To go back to the old database instead: restore the .env that created it (it holds the"
    nx_info "  only copy of that password), remove COMPOSE_PROJECT_NAME from .env, and re-run."
    nx_info "  To discard it:  docker volume rm ${volume}"
}

# The api container id for this project, or "" if it is not up. `|| true` because this is
# called from a plain assignment under `set -e -o pipefail`, where a docker that is briefly
# unreachable mid-startup would otherwise end the script instead of the poll returning "not
# up yet", which is what it means.
nx_api_container() {
    $NX_DOCKER ps -q \
        --filter "label=com.docker.compose.project=${PROJECT}" \
        --filter "label=com.docker.compose.service=api" 2>/dev/null | head -1 || true
}

# `compose up`, once, capturing the output so a known failure can be recognised. The log
# is echoed as it goes: a first run builds two images and pulls MySQL, and several minutes
# of silence is indistinguishable from a hang.
nx_compose_up() {
    local log="$1"
    # shellcheck disable=SC2086
    $NX_COMPOSE up -d --build 2>&1 | tee "$log"
}

nx_run_docker() {
    local log="" attempt=1 container="" status="" waited=0 previous_app_port=""

    nx_choose_mysql_image
    nx_resolve_volume_conflict

    export COMPOSE_PROJECT_NAME="$PROJECT"

    nx_step "6. Building and starting"
    nx_info "first run pulls MySQL and builds two images; expect a few minutes"

    log="$(mktemp "${TMPDIR:-/tmp}/nexarag-compose.XXXXXX")"
    while :; do
        if nx_compose_up "$log"; then
            break
        fi

        if [ "$attempt" -ge 3 ]; then
            nx_warn "the stack did not start after ${attempt} attempts."
            nx_info "The output above is the whole story; ${NX_COMPOSE} logs is the next place to look."
            rm -f "$log"
            exit 1
        fi

        # A port that was free during step 4 and taken by the time compose bound it. The
        # window is small but it is exactly what happens when two people run setup on one
        # shared machine, or when the operator started something else while the images
        # built. Re-settling moves whichever port is now occupied.
        if grep -qiE 'port is already allocated|address already in use|bind: ' "$log"; then
            nx_warn "a port was taken between choosing it and binding it"
            previous_app_port="$APP_PORT_VALUE"
            NX_CLAIMED_PORTS=""
            nx_settle_port MYSQL_PORT "$MYSQL_PORT_VALUE" "the database" "$ENV_FILE" "$PROJECT" \
                && MYSQL_PORT_VALUE="$NX_PORT"
            nx_settle_port API_PORT "$API_PORT_VALUE" "the API" "$ENV_FILE" "$PROJECT" \
                && API_PORT_VALUE="$NX_PORT"
            nx_settle_port APP_PORT "$APP_PORT_VALUE" "the app" "$ENV_FILE" "$PROJECT" \
                && APP_PORT_VALUE="$NX_PORT"
            nx_env_retarget_port APP_BASE_URL "$previous_app_port" "$APP_PORT_VALUE" "$ENV_FILE"
            nx_env_retarget_port CORS_ORIGINS "$previous_app_port" "$APP_PORT_VALUE" "$ENV_FILE"
        elif grep -qiE 'no matching manifest|no match for platform' "$log"; then
            nx_warn "an image has no build for $(nx_arch)"
            nx_env_write MYSQL_IMAGE "mysql:8.4" "$ENV_FILE"
            nx_fix "retrying with mysql:8.4"
        else
            nx_warn "retrying — a build or pull failure at this point is usually the network"
            sleep 5
        fi

        attempt=$((attempt + 1))
    done
    rm -f "$log"

    nx_step "7. Waiting for the API to report ready"
    while [ "$waited" -lt 180 ]; do
        container="$(nx_api_container)"
        if [ -n "$container" ]; then
            # `docker inspect` rather than `compose ps --format`, because the format flag
            # is a compose v2 feature and this may be running through v1.
            status="$($NX_DOCKER inspect --format '{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}' "$container" 2>/dev/null || echo unknown)"
            [ "$status" = "healthy" ] && break
        fi
        sleep 3
        waited=$((waited + 3))
        [ $((waited % 30)) -eq 0 ] && nx_dim "still starting (${waited}s)"
    done

    if [ "$status" != "healthy" ]; then
        nx_warn "the API has not reported healthy after ${waited}s. Recent logs:"
        # shellcheck disable=SC2086
        $NX_COMPOSE logs --tail 40 api || true
        if [ -n "$container" ] && $NX_DOCKER logs "$container" 2>&1 | grep -q "Access denied for user"; then
            nx_warn "the database refused the API's credentials — see the volume note in scripts/lib/run_docker.sh."
            nx_info "Fix:  make reset   (deletes the database)   or restore the .env that created it."
        fi
        nx_die "startup did not complete — '${NX_COMPOSE} logs -f api' has the detail."
    fi
    nx_ok "API is healthy"

    nx_summary "http://localhost:${APP_PORT_VALUE}" "http://localhost:${API_PORT_VALUE}/api/v1/health/ready"
}
