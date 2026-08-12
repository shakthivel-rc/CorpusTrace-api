#!/usr/bin/env bash
#
# Both dev servers, on the ports setup actually chose. Ctrl-C stops both.
#
# This is a script rather than a recipe in the Makefile because the ports live in .env, and
# make cannot read that file safely: `include .env` treats `#` as a comment mid-value, trips
# over spaces and would expose every secret in it to every recipe. Sourcing it in a shell is
# the only way to read it that agrees with how python-dotenv and docker compose read it.
set -euo pipefail

API_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_DIR="$(cd "$API_DIR/.." && pwd)/Nexarag-app"

# shellcheck source=lib/common.sh
. "$API_DIR/scripts/lib/common.sh"
# shellcheck source=lib/env.sh
. "$API_DIR/scripts/lib/env.sh"

ENV_FILE="$API_DIR/.env"
[ -f "$ENV_FILE" ] || nx_die "no .env — run 'make setup' first."

API_PORT="$(nx_env_get API_PORT "$ENV_FILE")";     [ -n "$API_PORT" ] || API_PORT=8000
DEV_PORT="$(nx_env_get APP_DEV_PORT "$ENV_FILE")"; [ -n "$DEV_PORT" ] || DEV_PORT=3000

[ -x "$API_DIR/.venv/bin/uvicorn" ] \
    || nx_die "no virtualenv at $API_DIR/.venv — run 'make setup-native' first."
[ -d "$APP_DIR/node_modules" ] \
    || nx_die "the SPA has no node_modules — run 'make setup-native' first."

nx_info "API → http://localhost:${API_PORT}   SPA → http://localhost:${DEV_PORT}"

# The dev server reads both of these (see vite.config.ts): one to listen on, one to proxy
# /api to. Without the second, moving the API off 8000 gives a SPA that loads and then
# fails every request against a port with nothing behind it.
export NEXARAG_DEV_PORT="$DEV_PORT"
export NEXARAG_API_PORT="$API_PORT"

trap 'kill 0' INT TERM EXIT
( cd "$API_DIR" && .venv/bin/uvicorn main:app --reload --host 0.0.0.0 --port "$API_PORT" ) &
( cd "$APP_DIR" && npm run dev ) &
wait
