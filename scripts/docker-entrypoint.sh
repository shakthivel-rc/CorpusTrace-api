#!/usr/bin/env sh
# Container entrypoint: bring the schema up to date, then hand over to the CMD.
#
# Migrations run here rather than in a separate one-shot service because `alembic upgrade
# head` is idempotent and the API is unusable against an out-of-date schema anyway — a
# container that starts and then 500s on every request is a worse outcome than one that
# takes three seconds longer to come up.
#
# For a multi-replica production deployment you would move this to a job that runs once,
# rather than racing N replicas through the same migration. With the single API container
# compose starts, this is the right trade.
set -eu

# OLLAMA_BASE_URL is the one setting whose correct value differs between the two ways this
# project runs, and both ways read the same .env. Natively the API reaches Ollama on the
# host at localhost; in here localhost is this container, where nothing serves 11434 — so
# the call fails with "[Errno 111] Connection refused" and the app reports it as a provider
# problem, naming the model rather than the address it could not reach.
#
# docker-compose.yml already defaults this to the service name, but `${OLLAMA_BASE_URL:-...}`
# only applies when the variable is unset, and .env.example ships a localhost line — so any
# .env copied from it silently defeats the default.
#
# The rewrite is conditional on the `ollama` service actually resolving, which is what makes
# it safe: if there is no such service, the operator's value is left exactly as they wrote
# it, whatever it points at.
case "${OLLAMA_BASE_URL:-}" in
    *localhost*|*127.0.0.1*)
        if getent hosts ollama >/dev/null 2>&1; then
            OLLAMA_BASE_URL="http://ollama:11434"
            export OLLAMA_BASE_URL
            echo "[entrypoint] OLLAMA_BASE_URL named localhost, which in here is this container; using http://ollama:11434"
        fi
        ;;
esac

echo "[entrypoint] waiting for the database"
# Alembic's own connection is the honest readiness probe: it is the exact driver, host and
# credentials the app will use, so a success here means more than a TCP port being open.
attempt=1
until alembic current >/dev/null 2>&1; do
    if [ "$attempt" -ge 30 ]; then
        echo "[entrypoint] database did not become reachable; last attempt:" >&2
        alembic current || true
        exit 1
    fi
    attempt=$((attempt + 1))
    sleep 2
done

echo "[entrypoint] applying migrations"
alembic upgrade head

# Seeding is opt-in and idempotent. Without SUPERADMIN_PASSWORD the seeder raises by
# design, so it is only attempted when one was supplied.
if [ -n "${SUPERADMIN_PASSWORD:-}" ]; then
    echo "[entrypoint] seeding superadmin and roles"
    python -m seeders.user_seeder || echo "[entrypoint] seeding skipped (already seeded)"
fi

echo "[entrypoint] starting: $*"
exec "$@"
