"""The setup scripts' pure logic.

`scripts/bootstrap.sh` is the one piece of this repository that runs on a machine where
nothing works yet, which is exactly where a bug is hardest to diagnose — there is no
application to read a stack trace from, and the operator has no reason to suspect the
installer rather than the thing being installed. So the parts that can be tested without a
Docker daemon, a package manager or a network are tested here.

The harness sources the libraries under `set -euo pipefail`, deliberately, because that is
what bootstrap.sh runs under and it is not a neutral choice: `nx_env_get` originally ended
in a `grep` pipeline, and a key simply being absent from .env made grep exit 1, which
pipefail turned into a failed assignment, which `set -e` turned into the script exiting
silently in the middle of step 2 with status 0. Without the same flags here, that passes.

Everything these tests touch writes to a tmp_path .env; nothing runs docker, and nothing
installs anything.
"""
import shutil
import socket
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

API_DIR = Path(__file__).resolve().parents[2]
SCRIPTS = API_DIR / "scripts"
LIB_DIR = SCRIPTS / "lib"
LIBS = ("common.sh", "env.sh", "ports.sh", "deps.sh")

no_bash = pytest.mark.skipif(shutil.which("bash") is None, reason="bash is not installed")


def run(body: str) -> subprocess.CompletedProcess:
    """Source the setup libraries and run `body`, under bootstrap.sh's own shell flags."""
    preamble = "set -euo pipefail\n" + "\n".join(f'. "{LIB_DIR / lib}"' for lib in LIBS)
    return subprocess.run(
        ["bash", "-c", f"{preamble}\n{body}"],
        cwd=str(API_DIR),
        capture_output=True,
        text=True,
        # NO_COLOR keeps escape codes out of the assertions; the scripts honour it.
        env={"PATH": "/usr/bin:/bin:/usr/sbin:/sbin", "NO_COLOR": "1", "HOME": "/tmp"},
    )


def write_env(tmp_path: Path, contents: str) -> Path:
    target = tmp_path / ".env"
    target.write_text(contents)
    return target


@no_bash
class TestEverythingParses:
    def test_every_shell_script_is_syntactically_valid(self):
        """`bash -n` on all of them.

        A syntax error in a lib file is not caught by any other test here — sourcing it
        would abort the harness — and it is caught by nothing at all until an operator on a
        fresh machine runs setup.
        """
        scripts = sorted(SCRIPTS.glob("*.sh")) + sorted((SCRIPTS / "lib").glob("*.sh"))
        assert scripts, "no shell scripts found; has the layout moved?"
        for script in scripts:
            result = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
            assert result.returncode == 0, f"{script.name}: {result.stderr}"


@no_bash
class TestVersionComparison:
    @pytest.mark.parametrize(
        "have,want,expected",
        [
            ("3.12.3", "3.10", True),
            ("3.10", "3.10", True),
            ("3.9.18", "3.10", False),      # the one `sort -V` gets wrong on macOS
            ("24.19.0", "24", True),
            ("v22.1.0", "24", False),       # node -v prints a leading v
            ("3.12.3rc1", "3.12", True),    # a pre-release suffix is not a version component
            ("8.0.36-1ubuntu2", "8.0", True),
            ("", "3.10", False),            # a tool that would not say
        ],
    )
    def test_dotted_versions_compare_numerically(self, have, want, expected):
        result = run(f'nx_version_at_least "{have}" "{want}" && echo YES || echo NO')
        assert result.stdout.strip() == ("YES" if expected else "NO"), result.stderr


@no_bash
class TestEnvFile:
    def test_a_missing_key_is_empty_and_not_an_error(self, tmp_path):
        """The pipefail regression, pinned.

        Reading a key that is not there is the most ordinary thing this code does — every
        fresh clone hits it for every generated value — and it must not end the script.
        """
        env = write_env(tmp_path, "PRESENT=yes\n")
        result = run(f'value="$(nx_env_get ABSENT "{env}")"; echo "[$value]"; echo alive')
        assert result.returncode == 0, result.stderr
        assert "[]" in result.stdout
        assert "alive" in result.stdout, "the script exited early instead of returning empty"

    def test_a_value_containing_an_equals_sign_survives(self, tmp_path):
        # `cut -d= -f2` would truncate this at the first '='. A generated password or a URL
        # with a query string is where that bites, and it is silent.
        env = write_env(tmp_path, "TOKEN=abc=def==\n")
        result = run(f'nx_env_get TOKEN "{env}"')
        assert result.stdout == "abc=def=="

    def test_a_blank_placeholder_is_filled_in_place(self, tmp_path):
        """`.env.example` ships `SECRET_KEY=`, which is set-but-empty.

        Compose's `${SECRET_KEY:?}` errors on an empty value exactly as it does on an unset
        one, so "the line exists" is the wrong question.
        """
        env = write_env(tmp_path, "SECRET_KEY=\nOTHER=keep\n")
        result = run(f'nx_env_ensure SECRET_KEY generated "{env}"')
        assert result.returncode == 0, result.stderr
        text = env.read_text()
        assert "SECRET_KEY=generated" in text
        assert text.count("SECRET_KEY=") == 1, "a second assignment for one key is ambiguous"
        assert "OTHER=keep" in text

    def test_an_existing_value_is_never_overwritten(self, tmp_path):
        # Setup runs against checkouts holding live credentials. Regenerating SECRET_KEY
        # there would invalidate every session and make every stored LLM credential
        # permanently undecryptable.
        env = write_env(tmp_path, "SECRET_KEY=already-real\n")
        run(f'nx_env_ensure SECRET_KEY brand-new "{env}"')
        assert env.read_text() == "SECRET_KEY=already-real\n"

    def test_a_missing_key_is_appended(self, tmp_path):
        env = write_env(tmp_path, "A=1\n")
        run(f'nx_env_ensure B two "{env}"')
        assert env.read_text() == "A=1\nB=two\n"

    def test_only_invented_values_are_marked_generated(self, tmp_path):
        """The volume-conflict check keys off this list, so a false entry is expensive."""
        env = write_env(tmp_path, "KEPT=real\n")
        result = run(
            f'nx_env_ensure KEPT other "{env}"; nx_env_ensure MADE up "{env}"; '
            'nx_env_generated MADE && echo MADE-YES; nx_env_generated KEPT && echo KEPT-YES || echo KEPT-NO'
        )
        assert "MADE-YES" in result.stdout
        assert "KEPT-NO" in result.stdout

    def test_dry_run_changes_nothing(self, tmp_path):
        """`--check` walks the whole decision tree, including choosing ports."""
        env = write_env(tmp_path, "A=1\n")
        result = run(f'NX_DRY_RUN=1 nx_env_write A 2 "{env}"; NX_DRY_RUN=1 nx_env_write NEW x "{env}"')
        assert result.returncode == 0, result.stderr
        assert env.read_text() == "A=1\n"


@no_bash
class TestPortRetargeting:
    def test_a_url_follows_the_port(self, tmp_path):
        # APP_BASE_URL is interpolated into every emailed invite and reset link. Left
        # pointing at the old port, the stack comes up perfectly and hands out sign-in
        # links to a port with nothing behind it.
        env = write_env(tmp_path, "APP_BASE_URL=http://localhost:8080\n")
        run(f'nx_env_retarget_port APP_BASE_URL 8080 8081 "{env}"')
        assert env.read_text().strip() == "APP_BASE_URL=http://localhost:8081"

    def test_every_occurrence_in_a_csv_follows(self, tmp_path):
        env = write_env(tmp_path, "CORS_ORIGINS=http://localhost:3000,http://127.0.0.1:8080,http://localhost:8080\n")
        run(f'nx_env_retarget_port CORS_ORIGINS 8080 8081 "{env}"')
        value = env.read_text().strip()
        assert value.endswith("http://127.0.0.1:8081,http://localhost:8081")
        assert ":3000" in value, "an unrelated port was rewritten"

    def test_a_longer_number_starting_with_the_port_is_left_alone(self, tmp_path):
        # ":8080" is a prefix of ":80800". Substring replacement would corrupt it.
        env = write_env(tmp_path, "X=http://h:80800/a\n")
        run(f'nx_env_retarget_port X 8080 8081 "{env}"')
        assert env.read_text().strip() == "X=http://h:80800/a"

    def test_a_value_that_does_not_name_the_port_is_untouched(self, tmp_path):
        env = write_env(tmp_path, "APP_BASE_URL=https://corpustrace.example.com\n")
        run(f'nx_env_retarget_port APP_BASE_URL 8080 8081 "{env}"')
        assert env.read_text().strip() == "APP_BASE_URL=https://corpustrace.example.com"


@no_bash
class TestPortSelection:
    @pytest.fixture()
    def occupied_port(self):
        """A port with a real listener on it, for the duration of one test."""
        listener = socket.socket()
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        yield listener.getsockname()[1]
        listener.close()

    def test_a_listening_port_is_detected(self, occupied_port):
        result = run(f"nx_port_in_use {occupied_port} && echo IN-USE || echo FREE")
        assert result.stdout.strip() == "IN-USE", result.stderr

    def test_a_quiet_port_is_free(self):
        probe = socket.socket()
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
        probe.close()
        result = run(f"nx_port_in_use {port} && echo IN-USE || echo FREE")
        assert result.stdout.strip() == "FREE", result.stderr

    def test_the_scan_steps_over_an_occupied_port(self, occupied_port):
        result = run(f'nx_free_port {occupied_port} && echo "$NX_PORT"')
        assert result.returncode == 0, result.stderr
        assert int(result.stdout.strip()) > occupied_port

    def test_two_services_never_receive_the_same_port(self):
        """NX_CLAIMED_PORTS, and why the result comes back through a global.

        Both services would otherwise be told the same free port a millisecond apart, and
        the second container to start would fail with the exact error the scan exists to
        prevent. This test is the reason these functions do not echo their result: with
        `port="$(nx_free_port …)"` the claim happened inside a command-substitution
        subshell and was discarded on return, so the list was empty on every call and this
        assertion failed while everything looked right.
        """
        result = run('nx_free_port 41000; a="$NX_PORT"; nx_free_port 41000; echo "$a $NX_PORT"')
        first, second = result.stdout.split()
        assert first != second, result.stderr

    def test_a_claim_survives_the_call(self):
        """The subshell trap itself, stated directly."""
        result = run('nx_free_port 41100; nx_port_claimed "$NX_PORT" && echo CLAIMED || echo LOST')
        assert result.stdout.strip() == "CLAIMED", result.stderr

    def test_an_occupied_port_moves_and_the_env_file_records_it(self, tmp_path, occupied_port):
        env = write_env(tmp_path, f"APP_PORT={occupied_port}\n")
        result = run(f'nx_settle_port APP_PORT {occupied_port} "the app" "{env}" ""; echo "$NX_PORT"')
        chosen = result.stdout.strip()
        assert int(chosen) > occupied_port
        assert env.read_text().strip() == f"APP_PORT={chosen}"

    def test_a_free_port_is_kept(self, tmp_path):
        probe = socket.socket()
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
        probe.close()
        env = write_env(tmp_path, f"APP_PORT={port}\n")
        result = run(f'nx_settle_port APP_PORT {port} "the app" "{env}" ""; echo "$NX_PORT"')
        assert result.stdout.strip() == str(port)

    def test_the_narration_goes_to_stderr(self, tmp_path, occupied_port):
        """Values on stdout, diagnostics on stderr.

        Several helpers here do return through stdout (nx_env_get, nx_find_python), and a
        log line written there is captured as part of the value rather than read as a log.
        """
        env = write_env(tmp_path, f"APP_PORT={occupied_port}\n")
        result = run(f'nx_settle_port APP_PORT {occupied_port} "the app" "{env}" ""')
        assert result.stdout == "", f"a log line reached stdout: {result.stdout!r}"
        assert "in use" in result.stderr, "the narration went missing entirely"


@no_bash
class TestPackageMapping:
    @pytest.mark.parametrize("logical", ["git", "python", "build-tools", "mysql-dev", "pkg-config", "node"])
    def test_apt_has_a_package_for_every_logical_name(self, logical):
        """A typo in a `case` arm falls through to the empty default and installs nothing.

        `nx_install` reads an empty mapping as "this manager bundles it elsewhere" and
        returns success, so the mistake surfaces as a missing dependency several rungs
        later rather than as a failed install.
        """
        result = run(f'nx_package_name {logical} apt')
        assert result.stdout.strip(), f"apt has no package mapped for '{logical}'"

    def test_a_bundled_package_maps_to_nothing_deliberately(self):
        # Homebrew's python formula includes venv and the headers; asking for them
        # separately is an error, not a no-op.
        result = run('nx_package_name python-venv brew')
        assert result.stdout.strip() == ""

    def test_an_unknown_logical_name_is_empty_rather_than_a_crash(self):
        result = run('nx_package_name not-a-real-package apt')
        assert result.returncode == 0
        assert result.stdout.strip() == ""


class TestLineEndings:
    """The Windows failure that has nothing to do with Windows.

    Git for Windows defaults to core.autocrlf=true. A CRLF `docker-entrypoint.sh` makes the
    kernel look for an interpreter named "sh\\r", so the API container exits immediately
    with `/usr/bin/env: 'sh\\r': No such file or directory` — a message naming no file in
    this repository, appearing only on Windows, against an image that built perfectly.
    """

    @pytest.mark.parametrize("repo", ["CorpusTrace-api", "CorpusTrace-app"])
    def test_shell_scripts_are_pinned_to_lf(self, repo):
        attributes = (API_DIR if repo == "CorpusTrace-api" else API_DIR.parent / repo) / ".gitattributes"
        if repo == "CorpusTrace-app" and not attributes.exists():
            pytest.skip("CorpusTrace-app is not checked out beside this repository")
        assert attributes.exists(), f"{repo}/.gitattributes is missing"
        text = attributes.read_text()
        assert "*.sh" in text and "eol=lf" in text

    def test_no_script_in_this_repository_has_crlf(self):
        for script in sorted(SCRIPTS.rglob("*.sh")):
            assert b"\r\n" not in script.read_bytes(), f"{script.name} has CRLF line endings"


class TestOllamaBaseUrlRepair:
    """The one setting whose right value differs between the two ways this project runs.

    Natively the API reaches Ollama on the host at localhost; inside a container localhost
    is that container, where nothing serves 11434. Both defaults are already correct — but
    only while .env leaves the key unset, because compose's `${OLLAMA_BASE_URL:-...}` is a
    default rather than an override. A .env that names it hands the container the host's
    answer and every LLM call fails with "[Errno 111] Connection refused", reported against
    the model rather than the address.

    The block is sliced out and run on its own: the entrypoint around it waits on a database
    and then execs a server.
    """

    MARKER = "RESULT="

    def _block(self) -> str:
        text = (SCRIPTS / "docker-entrypoint.sh").read_text()
        start = text.index('case "${OLLAMA_BASE_URL:-}"')
        return text[start : text.index("esac", start) + len("esac")]

    def _run(self, url: str, *, ollama_resolves: bool, tmp_path: Path):
        stub = tmp_path / "bin"
        stub.mkdir(exist_ok=True)
        getent = stub / "getent"
        getent.write_text(f"#!/bin/sh\nexit {0 if ollama_resolves else 2}\n")
        getent.chmod(0o755)
        return subprocess.run(
            ["sh", "-c", f'{self._block()}\nprintf "{self.MARKER}%s" "${{OLLAMA_BASE_URL:-}}"'],
            capture_output=True,
            text=True,
            env={"PATH": f"{stub}:/usr/bin:/bin", "OLLAMA_BASE_URL": url},
        )

    def _value(self, result) -> str:
        return result.stdout.split(self.MARKER, 1)[1]

    @pytest.mark.parametrize("url", ["http://localhost:11434", "http://127.0.0.1:11434"])
    def test_a_localhost_url_is_repaired_when_the_service_exists(self, url, tmp_path):
        result = self._run(url, ollama_resolves=True, tmp_path=tmp_path)
        assert self._value(result) == "http://ollama:11434"
        # Repairs are announced. A silent one is indistinguishable from the setting having
        # been ignored, which is the next thing someone would suspect.
        assert "OLLAMA_BASE_URL" in result.stdout

    def test_a_localhost_url_is_left_alone_with_no_ollama_service(self, tmp_path):
        # What makes the rewrite safe: with no such service to reach, the operator's value
        # is theirs — it may name a tunnel, or a host-networked container.
        result = self._run("http://localhost:11434", ollama_resolves=False, tmp_path=tmp_path)
        assert self._value(result) == "http://localhost:11434"

    def test_a_remote_url_is_never_rewritten(self, tmp_path):
        # Someone pointing at an Ollama on another machine has said something deliberate.
        result = self._run("http://10.0.0.5:11434", ollama_resolves=True, tmp_path=tmp_path)
        assert self._value(result) == "http://10.0.0.5:11434"


class TestEnvExampleDefaults:
    def test_ollama_base_url_ships_commented_out(self):
        """An active line here is what defeats compose's container-correct default.

        .env.example is copied wholesale into a real .env, so a value that is right for one
        of the two run modes and wrong for the other must not be handed out pre-set.
        """
        for line in (API_DIR / ".env.example").read_text().splitlines():
            assert not line.strip().startswith("OLLAMA_BASE_URL="), (
                "OLLAMA_BASE_URL must ship commented out: an active value overrides the "
                "compose default and points the containerised API at its own port"
            )
