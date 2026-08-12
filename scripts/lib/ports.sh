# shellcheck shell=bash
#
# Finding a port that is actually free, on a machine whose other software this script knows
# nothing about.
#
# 8080 is the single most contested port on a developer machine (Tomcat, Jenkins, another
# compose stack, a colleague's demo). 8000 is the second. Docker's failure when one is
# taken is `Bind for 0.0.0.0:8080 failed: port is already allocated`, emitted *after* the
# images have built — several minutes into a first run — and it leaves the other containers
# up, so the next `docker compose up` hits a different error. Choosing a free port up front
# costs one syscall.
#
# This file is held to the portability contract at the top of common.sh.

[ -n "${NX_PORTS_SH:-}" ] && return 0
NX_PORTS_SH=1

# Ports this run has already handed out. Two services asking for a free port a millisecond
# apart would otherwise both be told 8081, and the second container to start would fail
# with the exact error the scan exists to prevent.
NX_CLAIMED_PORTS=""

nx_port_claim() { NX_CLAIMED_PORTS="${NX_CLAIMED_PORTS} $1"; }

nx_port_claimed() {
    case " ${NX_CLAIMED_PORTS} " in
        *" $1 "*) return 0 ;;
        *) return 1 ;;
    esac
}

# Is anything listening on PORT on this host?
#
# Four implementations because no single tool is present everywhere: `ss` is Linux-only and
# `netstat` has been deprecated out of several distributions' default install; `lsof` is
# usual on macOS and optional on Linux; and a machine with none of them still has bash.
#
# All four answer "is something listening", which is the right question for "will Docker be
# able to publish here" in every case an operator will meet. A socket bound to one specific
# non-loopback interface can still collide with a 0.0.0.0 publish while showing up in none
# of the loopback probes — that is the known gap, and Docker's own error remains the
# backstop for it.
nx_port_in_use() {
    local port="$1"

    if nx_have ss; then
        ss -ltn 2>/dev/null | awk -v p="$port" '
            NR > 1 { n = split($4, a, ":"); if (a[n] == p) { found = 1 } }
            END { exit found ? 0 : 1 }' && return 0
        return 1
    fi

    if nx_have lsof; then
        lsof -nP -iTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1 && return 0
        return 1
    fi

    if nx_have netstat; then
        # Linux prints 0.0.0.0:8080, macOS prints *.8080 — split on both separators and
        # compare the last field either way.
        netstat -an 2>/dev/null | awk -v p="$port" '
            /LISTEN/ { n = split($4, a, /[.:]/); if (a[n] == p) { found = 1 } }
            END { exit found ? 0 : 1 }' && return 0
        return 1
    fi

    # Last resort: bash can open a TCP socket itself. A connect that succeeds proves
    # something is there; connection-refused on loopback returns immediately.
    if (exec 3<>"/dev/tcp/127.0.0.1/$port") 2>/dev/null; then
        exec 3>&- 2>/dev/null || true
        return 0
    fi
    return 1
}

# Is PORT occupied by a container belonging to compose project PROJECT?
#
# This is the difference between "8080 is taken, move to 8081" and "8080 is taken *by the
# stack this script is setting up*, keep it". Without the distinction, every re-run of
# `make setup` against a running stack walks the app one port further along — 8080, 8081,
# 8082 — and each move orphans the previous containers while the operator watches the URL
# change for no reason.
nx_port_used_by_project() {
    local port="$1" project="$2"
    nx_have docker || return 1
    docker ps --filter "label=com.docker.compose.project=${project}" \
        --format '{{.Ports}}' 2>/dev/null \
        | grep -q ":${port}->" || return 1
    return 0
}

# The port the last successful nx_free_port / nx_settle_port chose.
#
# A GLOBAL, NOT STDOUT, AND THAT IS THE WHOLE POINT.
#
# These functions used to echo the port so a caller could write `PORT="$(nx_settle_port
# …)"`. A command substitution runs in a subshell, so every nx_port_claim inside it mutated
# a copy of NX_CLAIMED_PORTS that was discarded the instant the function returned — which
# meant the claim list, whose entire job is stopping two services being handed the same
# port, was empty on every call. It cost nothing and prevented nothing, and it looked
# exactly like it was working. Returning through a global keeps the call in this shell.
NX_PORT=""

# Find the first free port at or after START. Sets NX_PORT; returns 1 having found none.
#
# The scan is bounded: an unbounded one on a machine with an exhausted range walks 60,000
# ports and reports nothing useful. Twenty is far past any plausible run of collisions, and
# stopping there produces a message naming the range that was tried.
nx_free_port() {
    local start="$1" project="${2:-}" range="${3:-20}"
    local port="$start" limit=$((start + range))

    while [ "$port" -lt "$limit" ]; do
        if nx_port_claimed "$port"; then
            port=$((port + 1)); continue
        fi
        if [ -n "$project" ] && nx_port_used_by_project "$port" "$project"; then
            nx_port_claim "$port"; NX_PORT="$port"; return 0
        fi
        if ! nx_port_in_use "$port"; then
            nx_port_claim "$port"; NX_PORT="$port"; return 0
        fi
        port=$((port + 1))
    done
    return 1
}

# Settle one port key in .env: keep what is there if it works, move it if it does not.
# Sets NX_PORT.
#
# LABEL is what the port is for, in words, because "APP_PORT moved to 8081" tells an
# operator nothing about which URL just changed.
nx_settle_port() {
    local key="$1" default="$2" label="$3" file="$4" project="$5"
    local current=""

    current="$(nx_env_get "$key" "$file")"
    [ -n "$current" ] || current="$default"

    if nx_port_claimed "$current"; then
        : # another service in this run took it; fall through to the scan
    elif [ -n "$project" ] && nx_port_used_by_project "$current" "$project"; then
        nx_port_claim "$current"
        nx_env_write "$key" "$current" "$file"
        NX_PORT="$current"
        nx_dim "${label} keeps port ${current} (already published by this stack)"
        return 0
    elif ! nx_port_in_use "$current"; then
        nx_port_claim "$current"
        nx_env_write "$key" "$current" "$file"
        NX_PORT="$current"
        nx_dim "${label} → port ${current}"
        return 0
    fi

    if ! nx_free_port "$((current + 1))" "$project"; then
        nx_warn "port ${current} is in use and nothing from $((current + 1)) to $((current + 20)) is free."
        nx_info "Free one of them, or set ${key} in .env to a port you know is available."
        return 1
    fi

    nx_warn "port ${current} is already in use by something else on this machine"
    nx_fix "${label} → port ${NX_PORT} instead"
    nx_env_write "$key" "$NX_PORT" "$file"
}
