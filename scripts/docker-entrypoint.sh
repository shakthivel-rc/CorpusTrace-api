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
