# shellcheck shell=bash
#
# Shared plumbing for the setup scripts: output, retries and version comparison.
#
# PORTABILITY CONTRACT FOR EVERY FILE IN THIS DIRECTORY.
#
# macOS still ships bash 3.2 as /bin/bash — it is from 2007, and Apple will not update it
# because bash 4 is GPLv3. `#!/usr/bin/env bash` finds that one unless the operator has
# installed a newer bash *and* put it first on PATH, so assuming bash 4 is assuming the
# setup script only has to work on Linux. Everything here therefore avoids: associative
# arrays (`declare -A`), `${var^^}` / `${var,,}`, `mapfile`/`readarray`, `&>>`, `[[ -v ]]`
# and negative array indices. It also avoids `timeout` and `grep -P`, which are GNU
# coreutils/grep and are absent on a stock macOS, and any `sed -i`, whose in-place flag
# takes a mandatory argument on BSD and none on GNU.
#
# These are not stylistic preferences. Each one is a construct that fails *only* on the
# platform the operator is least likely to be able to debug it on.

[ -n "${NX_COMMON_SH:-}" ] && return 0
NX_COMMON_SH=1

# Colour is opt-out (NO_COLOR, https://no-color.org) and only ever emitted to a terminal.
# `make setup 2>&1 | tee setup.log` is a normal thing to do when something has gone wrong,
# and escape codes in that log are noise in the one artefact that has to stay readable.
# EVERY message goes to stderr, deliberately. Several of these functions are called from
# inside functions whose *stdout is a return value* — nx_env_get, nx_os, nx_find_python —
# and a log line on stdout there does not look like a broken log, it looks like a value
# with `  ! port 8080 is in use` glued to the front of it. Logs are diagnostics; stderr is
# where diagnostics belong.
if [ -t 2 ] && [ -z "${NO_COLOR:-}" ] && [ "${TERM:-dumb}" != "dumb" ]; then
    NX_C_BOLD=$(printf '\033[1m'); NX_C_DIM=$(printf '\033[2m')
    NX_C_RED=$(printf '\033[31m'); NX_C_GREEN=$(printf '\033[32m')
    NX_C_YELLOW=$(printf '\033[33m'); NX_C_CYAN=$(printf '\033[36m')
    NX_C_OFF=$(printf '\033[0m')
else
    NX_C_BOLD=""; NX_C_DIM=""; NX_C_RED=""; NX_C_GREEN=""
    NX_C_YELLOW=""; NX_C_CYAN=""; NX_C_OFF=""
fi

nx_bold() { printf '%s%s%s\n' "$NX_C_BOLD" "$*" "$NX_C_OFF" >&2; }
nx_step() { printf '\n%s%s%s\n' "$NX_C_BOLD" "$*" "$NX_C_OFF" >&2; }
nx_info() { printf '  %s\n' "$*" >&2; }
nx_dim()  { printf '  %s%s%s\n' "$NX_C_DIM" "$*" "$NX_C_OFF" >&2; }
nx_ok()   { printf '  %s✓%s %s\n' "$NX_C_GREEN" "$NX_C_OFF" "$*" >&2; }
nx_warn() { printf '  %s!%s %s\n' "$NX_C_YELLOW" "$NX_C_OFF" "$*" >&2; }

# The distinguishing mark of this script: it says what went wrong AND what it is doing
# about it, on the line where it happens. A setup that silently repairs things is a setup
# nobody can debug when the repair is the thing that was wrong.
nx_fix()  { printf '  %s→%s %s\n' "$NX_C_CYAN" "$NX_C_OFF" "$*" >&2; }

nx_die()  { printf '\n%s  ✗ %s%s\n' "$NX_C_RED" "$*" "$NX_C_OFF" >&2; exit 1; }

nx_have() { command -v "$1" >/dev/null 2>&1; }

# Ask, but never block a pipeline. With no terminal there is nobody to answer, and a setup
# script that hangs forever waiting on stdin inside CI is worse than one that decides.
# CORPUSTRACE_ASSUME_YES=1 (or --yes) is the explicit "decide for me" switch.
nx_confirm() {
    local prompt="$1" answer=""
    if [ "${CORPUSTRACE_ASSUME_YES:-0}" = "1" ] || [ ! -t 0 ]; then
        nx_dim "$prompt — assuming yes"
        return 0
    fi
    printf '  %s [Y/n] ' "$prompt" >&2
    read -r answer || return 0
    case "$answer" in
        [nN]|[nN][oO]) return 1 ;;
        *) return 0 ;;
    esac
}

# Run a command up to N times with 2s/4s/8s backoff, quietly the first time and loudly
# after that. Network-bound steps (git clone, pip, npm, docker pull) fail transiently far
# more often than they fail permanently, and "run it again" is what an operator would do.
nx_retry() {
    local tries="$1"; shift
    local attempt=1 delay=2
    while :; do
        if "$@"; then
            return 0
        fi
        if [ "$attempt" -ge "$tries" ]; then
            return 1
        fi
        nx_warn "'$1' failed (attempt ${attempt}/${tries}); retrying in ${delay}s"
        sleep "$delay"
        attempt=$((attempt + 1))
        delay=$((delay * 2))
    done
}

# Numeric dotted-version comparison: is HAVE >= WANT?
#
# In awk rather than shell because the shell answer people reach for is `sort -V`, which is
# GNU-only — on macOS it silently sorts lexically, and lexical order puts 3.9 after 3.12.
# That is exactly the comparison this function exists to get right.
nx_version_at_least() {
    awk -v have="$1" -v want="$2" '
        BEGIN {
            gsub(/^[vV]/, "", have); gsub(/^[vV]/, "", want)
            # Trim any suffix: "3.12.3rc1", "24.19.0-nightly", "8.0.36-1ubuntu2".
            gsub(/[^0-9.].*$/, "", have); gsub(/[^0-9.].*$/, "", want)
            n = split(have, h, "."); m = split(want, w, ".")
            top = (n > m ? n : m)
            for (i = 1; i <= top; i++) {
                a = (i <= n ? h[i] + 0 : 0)
                b = (i <= m ? w[i] + 0 : 0)
                if (a > b) exit 0
                if (a < b) exit 1
            }
            exit 0
        }'
}

# The numeric version a command reports, or "" if it will not say. Every tool prints its
# version somewhere different in the line, so take the first dotted number anywhere in the
# first line of output and let nx_version_at_least normalise it.
nx_version_of() {
    local out=""
    out="$("$@" 2>&1 | head -1)" || true
    printf '%s' "$out" | awk '{ if (match($0, /[0-9]+\.[0-9]+(\.[0-9]+)?/)) print substr($0, RSTART, RLENGTH) }'
}
