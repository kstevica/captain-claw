"""Tests for C3 — the deep-build best-of-N logic.

The agent spawn, git, and test runner are stubbed, so this exercises the
decision path (verify each attempt, reset between failed attempts, keep the first
that passes else the last) without a real repo or model.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from captain_claw.flight_deck import code_routes
from captain_claw.flight_deck.quality_profile import QualityProfile, TokenBudget


@pytest.fixture
def stub(monkeypatch):
    """Stub the externals `_deep_build` calls; return a mutable control dict."""
    ctrl = {"commits": [], "resets": [], "test_results": [], "acted": True}

    async def fake_run_agent(*a, **k):
        return {"ok": True, "output": "built",
                "actions": [{"tool": "write"}] if ctrl["acted"] else []}

    commit_seq = iter(["c1", "c2", "c3", "c4"])

    async def fake_git_log(repo, limit=1):
        return [{"sha": "base0"}]

    async def fake_git_commit(repo, msg):
        sha = next(commit_seq)
        ctrl["commits"].append((sha, msg))
        return sha

    async def fake_git_reset(repo, ref):
        ctrl["resets"].append(ref)

    def fake_detect(repo, override=""):
        return "pytest -q"

    async def fake_run_tests(repo, cmd, **k):
        return {"ran": True, "ok": ctrl["test_results"].pop(0), "command": cmd, "output": ""}

    monkeypatch.setattr(code_routes, "_run_agent", fake_run_agent)
    monkeypatch.setattr(code_routes, "_phase", lambda *a, **k: None)
    monkeypatch.setattr(code_routes, "_progress", lambda *a, **k: None)
    monkeypatch.setattr(code_routes, "_cancelled", lambda pkey: False)
    monkeypatch.setattr(code_routes.code_git, "git_log", fake_git_log)
    monkeypatch.setattr(code_routes.code_git, "git_commit", fake_git_commit)
    monkeypatch.setattr(code_routes.code_git, "git_reset", fake_git_reset)
    monkeypatch.setattr(code_routes.code_verify, "detect_test_command", fake_detect)
    monkeypatch.setattr(code_routes.code_verify, "run_tests", fake_run_tests)
    return ctrl


async def _run(quality, ctrl):
    return await code_routes._deep_build(
        request=None, user={"id": "u"}, pkey="t/t", repo=Path("."), sdir=Path("."),
        intent="build a thing", by_id={}, tiers_map={}, env_vars=[],
        plan_file="plan.md", quality=quality, budget=TokenBudget(quality.token_budget))


async def test_first_passing_attempt_wins(stub):
    stub["test_results"] = [True]  # attempt 1 passes
    q = QualityProfile.from_dict({"deep_build": True, "deep_build_samples": 3})
    d, sha = await _run(q, stub)
    assert sha == "c1"
    assert stub["resets"] == []            # no isolation reset needed
    assert len(stub["commits"]) == 1       # stopped at the first winner


async def test_failed_attempt_is_reset_then_next_wins(stub):
    stub["test_results"] = [False, True]   # attempt 1 fails, attempt 2 passes
    q = QualityProfile.from_dict({"deep_build": True, "deep_build_samples": 3})
    d, sha = await _run(q, stub)
    assert sha == "c2"
    assert stub["resets"] == ["base0"]     # discarded the failed attempt exactly once


async def test_all_fail_keeps_last_without_resetting_it(stub):
    stub["test_results"] = [False, False]  # both fail
    q = QualityProfile.from_dict({"deep_build": True, "deep_build_samples": 2})
    d, sha = await _run(q, stub)
    assert sha == "c2"                      # keep the last attempt for the fix loop
    assert stub["resets"] == ["base0"]      # reset once (after attempt 1), NOT after the last


async def test_budget_stops_extra_attempts(stub):
    stub["test_results"] = [False, False, False]
    # Budget only affords one deep-build attempt's worth beyond the first is blocked.
    q = QualityProfile.from_dict(
        {"deep_build": True, "deep_build_samples": 3, "token_budget": 1})
    d, sha = await _run(q, stub)
    # First attempt runs; its failure can't afford a retry → keep it, no reset.
    assert sha == "c1"
    assert stub["resets"] == []


async def test_no_test_command_keeps_first_build(stub, monkeypatch):
    monkeypatch.setattr(code_routes.code_verify, "detect_test_command", lambda r, o="": "")
    q = QualityProfile.from_dict({"deep_build": True, "deep_build_samples": 3})
    d, sha = await _run(q, stub)
    assert sha == "c1"                      # no verifier → first real build wins
    assert stub["resets"] == []
