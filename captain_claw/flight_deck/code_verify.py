"""C1 — the ground-truth test gate for Code's build/fix loop.

Code's quality gate is otherwise three LLM reviewers + an LLM triage: opinion,
not proof. This adds the one infallible checker code has — the test runner —
as a deterministic, **zero-LLM-token** signal. After a build/fix commit the
loop runs the repo's tests; a failure becomes a blocking finding fed straight
into the existing triage alongside the reviewer reports, so the fix loop acts
on real breakage instead of a reviewer's guess.

Everything here is subprocess + string work. It costs no model tokens, which is
why the ``balanced`` quality preset can turn it on without moving spend. It is
also fully injectable (the command runner is a parameter) so it unit-tests with
stubs, mirroring ``dubina/coder.py``.

Detection is intentionally conservative: a command is returned only when there
is real signal in the repo. No signal → no command → the gate silently no-ops,
exactly as if the feature were off.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from captain_claw.dubina.coder import CommandRunner, shell_command_runner
from captain_claw.logging import get_logger

log = get_logger(__name__)

_TEST_TIMEOUT_S = 300.0     # a suite that hangs must not stall the build loop
_OUTPUT_CAP = 6000          # chars of failing output threaded into triage


def detect_test_command(repo: Path, override: str = "") -> str:
    """Best-effort discovery of how to run this repo's tests.

    An explicit ``override`` always wins. Otherwise probe for the common
    ecosystems; return ``""`` when there is no clear test signal so the caller
    skips the gate rather than running something meaningless.
    """
    if override and override.strip():
        return override.strip()
    try:
        return _detect(repo)
    except Exception as e:  # noqa: BLE001 — detection must never break the build
        log.warning("test command detection failed", error=str(e))
        return ""


def _detect(repo: Path) -> str:
    # Node — only if package.json declares a real test script (npm's default
    # placeholder exits non-zero on purpose; running it would be a false fail).
    pkg = repo / "package.json"
    if pkg.is_file():
        try:
            scripts = (json.loads(pkg.read_text()).get("scripts") or {})
        except (OSError, ValueError):
            scripts = {}
        test_script = str(scripts.get("test") or "")
        if test_script and "no test specified" not in test_script.lower():
            return "npm test --silent"

    # Python / pytest — a tests dir, test files, or a pytest config.
    if (repo / "pytest.ini").is_file() or (repo / "tox.ini").is_file():
        return "pytest -q"
    if (repo / "tests").is_dir() or (repo / "test").is_dir():
        if any(repo.rglob("test_*.py")) or any(repo.rglob("*_test.py")):
            return "pytest -q"
    if any(repo.glob("test_*.py")) or any(repo.glob("*_test.py")):
        return "pytest -q"
    pyproject = repo / "pyproject.toml"
    if pyproject.is_file() and "pytest" in _safe_read(pyproject).lower():
        return "pytest -q"

    # Go, Rust — cheap unambiguous markers.
    if (repo / "go.mod").is_file() and any(repo.rglob("*_test.go")):
        return "go test ./..."
    if (repo / "Cargo.toml").is_file():
        return "cargo test --quiet"

    # Make — last, only if there is an explicit test target.
    mk = repo / "Makefile"
    if mk.is_file():
        body = _safe_read(mk)
        if any(line.startswith(("test:", "test ")) for line in body.splitlines()):
            return "make test"
    return ""


def _safe_read(p: Path) -> str:
    try:
        return p.read_text()
    except OSError:
        return ""


async def run_tests(repo: Path, command: str, *,
                    runner: CommandRunner | None = None,
                    timeout: float = _TEST_TIMEOUT_S) -> dict:
    """Run ``command`` in ``repo`` and return a structured result.

    Shape: ``{ran, ok, command, output}``. ``ran`` is False only when there is
    no command. A timeout is reported as a failed run (``ok=False``) with a
    timeout note as output, never as a hang.
    """
    if not command:
        return {"ran": False, "ok": True, "command": "", "output": ""}
    run = runner or shell_command_runner
    try:
        ok, output = await asyncio.wait_for(run(command, str(repo)), timeout=timeout)
    except asyncio.TimeoutError:
        return {"ran": True, "ok": False, "command": command,
                "output": f"Test command timed out after {int(timeout)}s: {command}"}
    except Exception as e:  # noqa: BLE001
        log.warning("test run errored", command=command, error=str(e))
        return {"ran": True, "ok": False, "command": command,
                "output": f"Test command errored: {e}"}
    out = output or ""
    if len(out) > _OUTPUT_CAP:
        out = out[:_OUTPUT_CAP] + f"\n… (truncated at {_OUTPUT_CAP} chars)"
    return {"ran": True, "ok": bool(ok), "command": command, "output": out}


def as_review_entry(result: dict) -> dict | None:
    """Turn a test result into a synthetic reviewer report for triage.

    Returns ``None`` when there is nothing to say (no tests ran, or they passed —
    a pass shouldn't manufacture work). A failure becomes a high-signal,
    ground-truth "Test Runner" report the triage LLM weighs against the opinion
    reviews. Because it rides in as one more review entry, no extra agent is
    dispatched — the marginal cost is a few hundred input tokens on the triage
    call that already runs.
    """
    if not result.get("ran"):
        return None
    if result.get("ok"):
        return None
    cmd = result.get("command", "")
    body = (
        f"GROUND TRUTH — the test suite FAILS. This is a deterministic, blocking "
        f"finding: the code does not pass its own tests and must be fixed before "
        f"this task is considered done.\n\nCommand: `{cmd}`\n\nOutput:\n```\n"
        f"{result.get('output', '').strip()}\n```"
    )
    return {"role": "Test Runner", "id": "test-runner", "output": body}
