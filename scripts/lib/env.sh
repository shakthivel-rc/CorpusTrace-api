# shellcheck shell=bash
#
# Reading and writing .env without ever destroying a value that is already there.
#
# This file is held to the portability contract at the top of common.sh.

[ -n "${NX_ENV_SH:-}" ] && return 0
NX_ENV_SH=1

# Which keys this run invented a value for, rather than read from an existing .env. Only
# the generated ones can disagree with state that outlives the checkout — see the database
# volume check in bootstrap.sh.
NX_GENERATED_KEYS=""

# --check has to be able to walk the whole decision tree — including choosing ports, which
# is where most of the interesting diagnosis is — without leaving a mark. Every write in
# this file goes through nx_env_write, so one guard here covers all of them.
NX_DRY_RUN="${NX_DRY_RUN:-0}"

nx_env_generated() {
    case " ${NX_GENERATED_KEYS} " in
        *" $1 "*) return 0 ;;
        *) return 1 ;;
    esac
}

# The value of KEY, or "" if it is unset or blank.
#
# The `|| line=""` is not defensive noise. bootstrap.sh runs under `set -o pipefail`, and a
# grep that matches nothing exits 1 — which makes the whole pipeline fail, which makes the
# *assignment* fail, which under `set -e` exits the script. A key simply not being present
# in .env is the single most ordinary thing that can happen here, and without this it ended
# setup silently, mid-step, with status 0.
#
# `${line#*=}` rather than `cut -d= -f2`: DATABASE_URL and APP_BASE_URL contain no '=' today,
# but a password generated elsewhere or a query string in a URL would be truncated at the
# first one, and the corruption is invisible until something tries to connect.
nx_env_get() {
    local key="$1" file="$2" line=""
    [ -f "$file" ] || return 0
    line="$(grep -E "^${key}=" "$file" 2>/dev/null | head -1)" || line=""
    [ -n "$line" ] || return 0
    printf '%s' "${line#*=}"
}

# Give KEY a value if it does not already have one. A key that already holds a real value
# is never touched — this script must be safe to run against a checkout holding live
# credentials.
#
# "Has a value" is deliberately not the same question as "the line exists". `.env.example`
# ships documented-but-blank placeholders (`SECRET_KEY=`, `SUPERADMIN_PASSWORD=`), and the
# first version of this function tested only for the key, so on a fresh clone it decided
# both were already configured and generated neither. `docker compose` then failed on
# `${SECRET_KEY:?}` — which errors on an EMPTY value exactly as it does on an unset one —
# and the seeder would have skipped silently, leaving no account to sign in with.
nx_env_ensure() {
    local key="$1" value="$2" file="$3"

    if grep -qE "^${key}=." "$file" 2>/dev/null; then
        return 0
    fi

    NX_GENERATED_KEYS="${NX_GENERATED_KEYS} ${key}"

    if grep -qE "^${key}=" "$file" 2>/dev/null; then
        nx_env_write "$key" "$value" "$file"
        nx_dim "set ${key} (was blank)"
        return 0
    fi

    nx_env_write "$key" "$value" "$file"
    nx_dim "set ${key} (was missing)"
}

# Set KEY to VALUE, replacing whatever was there. Only for values this script *derives*
# from another one it just chose — never for a secret, and never for anything the operator
# might have set deliberately. nx_env_ensure is the function for those.
nx_env_write() {
    local key="$1" value="$2" file="$3" tmp=""

    if [ "$NX_DRY_RUN" = "1" ]; then
        nx_dim "would set ${key} in $(basename "$file")"
        return 0
    fi

    if ! grep -qE "^${key}=" "$file" 2>/dev/null; then
        printf '%s=%s\n' "$key" "$value" >> "$file"
        return 0
    fi

    # Rewrite the assignment where it stands rather than appending a second one. Two lines
    # for one key is ambiguous, and which one wins is not agreed between docker compose and
    # python-dotenv — so the file would mean different things to the container and to a
    # native run.
    #
    # awk via ENVIRON, not sed: these values contain `=`, `!`, `&` and `/`, all of which
    # need escaping in a sed replacement (`&` silently expands to the whole match) and none
    # of which need any in an awk variable. And `sed -i` itself is not portable — BSD sed
    # requires an argument to -i, GNU sed forbids one.
    tmp="$(mktemp "${file}.XXXXXX")"
    NX_KEY="$key" NX_VALUE="$value" awk '
        BEGIN { key = ENVIRON["NX_KEY"]; value = ENVIRON["NX_VALUE"]; filled = 0 }
        !filled && index($0, key "=") == 1 { print key "=" value; filled = 1; next }
        { print }
    ' "$file" > "$tmp"
    mv "$tmp" "$file"
}

# Fail here, with the name of the thing that is wrong, rather than letting compose report
# it as an interpolation error four steps later.
nx_env_require() {
    local key="$1" file="$2"
    grep -qE "^${key}=." "$file" 2>/dev/null \
        || nx_die "${key} has no value in .env — remove the line and re-run to have one generated."
}

# Follow a port move through a derived URL.
#
# Load-bearing, not cosmetic. APP_BASE_URL is interpolated into every invite and
# password-reset link the API emails, and CORS_ORIGINS decides whether the browser is
# allowed to talk to the API at all. If the SPA lands on 8081 because 8080 was taken and
# these still say 8080, the stack comes up perfectly and then hands out sign-in links to a
# port with nothing behind it — a failure that looks like broken email, not a busy port.
nx_env_retarget_port() {
    local key="$1" old="$2" new="$3" file="$4" current="" updated=""
    [ "$old" = "$new" ] && return 0

    current="$(nx_env_get "$key" "$file")"
    [ -n "$current" ] || return 0
    case "$current" in
        *":${old}"*) ;;
        *) return 0 ;;
    esac

    updated="$(NX_OLD=":$old" NX_NEW=":$new" awk -v s="$current" '
        BEGIN {
            old = ENVIRON["NX_OLD"]; new = ENVIRON["NX_NEW"]
            out = ""
            while ((i = index(s, old)) > 0) {
                # Only a whole port number, so :8080 does not also rewrite :80800.
                tail = substr(s, i + length(old))
                if (tail ~ /^[0-9]/) { out = out substr(s, 1, i + length(old) - 1); s = tail; continue }
                out = out substr(s, 1, i - 1) new
                s = tail
            }
            print out s
        }')"

    [ "$updated" = "$current" ] && return 0
    nx_env_write "$key" "$updated" "$file"
    nx_fix "${key} now points at port ${new}"
}
