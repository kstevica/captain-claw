"""Tests for C1 — the ground-truth test gate.

Detection is pure filesystem inspection; the runner is injected, so no real
subprocess or model is needed.
"""

from __future__ import annotations

import json

from captain_claw.flight_deck import code_verify


# ── detection ────────────────────────────────────────────────────────

def test_override_wins(tmp_path):
    assert code_verify.detect_test_command(tmp_path, "  make check ") == "make check"


def test_no_signal_returns_empty(tmp_path):
    assert code_verify.detect_test_command(tmp_path) == ""


def test_pytest_detected_from_test_files(tmp_path):
    (tmp_path / "test_thing.py").write_text("def test_x(): assert True\n")
    assert code_verify.detect_test_command(tmp_path) == "pytest -q"


def test_npm_default_placeholder_is_ignored(tmp_path):
    (tmp_path / "package.json").write_text(json.dumps(
        {"scripts": {"test": 'echo "Error: no test specified" && exit 1'}}))
    assert code_verify.detect_test_command(tmp_path) == ""


def test_npm_real_test_script_detected(tmp_path):
    (tmp_path / "package.json").write_text(json.dumps({"scripts": {"test": "vitest run"}}))
    assert code_verify.detect_test_command(tmp_path) == "npm test --silent"


# ── running ──────────────────────────────────────────────────────────

async def test_run_tests_no_command_does_not_run():
    res = await code_verify.run_tests(_tmp(), "")
    assert res == {"ran": False, "ok": True, "command": "", "output": ""}


async def test_run_tests_pass_injects_no_review():
    async def runner(cmd, cwd):
        return True, "3 passed"
    res = await code_verify.run_tests(_tmp(), "pytest -q", runner=runner)
    assert res["ran"] and res["ok"]
    assert code_verify.as_review_entry(res) is None


async def test_run_tests_fail_becomes_blocking_review_entry():
    async def runner(cmd, cwd):
        return False, "E   assert 1 == 2"
    res = await code_verify.run_tests(_tmp(), "pytest -q", runner=runner)
    assert res["ran"] and not res["ok"]
    entry = code_verify.as_review_entry(res)
    assert entry is not None
    assert entry["id"] == "test-runner"
    assert "GROUND TRUTH" in entry["output"]
    assert "assert 1 == 2" in entry["output"]


async def test_run_tests_timeout_is_reported_not_hung():
    import asyncio

    async def slow(cmd, cwd):
        await asyncio.sleep(10)
        return True, ""
    res = await code_verify.run_tests(_tmp(), "pytest -q", runner=slow, timeout=0.05)
    assert res["ran"] and not res["ok"]
    assert "timed out" in res["output"]


def _tmp():
    from pathlib import Path
    return Path(".")
