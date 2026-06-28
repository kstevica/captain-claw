"""Tests for the Dubina route layer (Phase 3): ladder building + track executors.

The thin POST/GET endpoints wrap these; we exercise the executors directly with a
stub provider factory and (for coder) a stub test runner, so no real model or
subprocess is needed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from captain_claw.flight_deck import dubina_routes as dr
from captain_claw.flight_deck.dubina_store import DubinaStore
from captain_claw.llm import LLMResponse

# ── Stubs ────────────────────────────────────────────────────────────

class StubProvider:
    def __init__(self, content: str):
        self.content = content

    async def complete(self, messages, tools=None, temperature=None, max_tokens=None):
        return LLMResponse(content=self.content, finish_reason="stop")


def factory_const(content: str):
    return lambda tier: StubProvider(content)


def runner_passes_on(token: str):
    async def run(command: str, cwd: str) -> tuple[bool, str]:
        code = (Path(cwd) / "solution.py").read_text()
        return (token in code), ("1 passed" if token in code else "1 failed")
    return run


def factory_raises():
    def f(tier):
        raise RuntimeError("provider unavailable")
    return f


# Library tier map a test "user" would have configured.
TIERS_MAP = {"t1": {"provider": "openai", "model": "m1"},
             "t2": {"provider": "openai", "model": "m2"}}
LADDER = ["t1", "t2"]


@pytest.fixture
async def store(tmp_path):
    s = DubinaStore(tmp_path / "dubina.db")
    await s.init()
    yield s
    await s.close()


# ── build_ladder ─────────────────────────────────────────────────────

def test_build_ladder_explicit_tiers():
    ladder = dr.build_ladder("t1", "t2", ["t1", "t2"],
                             default_ladder=LADDER, allowed={"t1", "t2"})
    assert [t.id for t in ladder] == ["t1", "t2"]
    assert ladder[0].cost < ladder[1].cost   # cost escalates by position


def test_build_ladder_single_tier_when_base_equals_max():
    assert [t.id for t in dr.build_ladder("t1", "t1", None,
                                          default_ladder=LADDER, allowed={"t1"})] == ["t1"]


def test_build_ladder_rejects_unknown_tier():
    from fastapi import HTTPException
    with pytest.raises(HTTPException):
        dr.build_ladder("t1", "ghost", ["t1", "ghost"],
                        default_ladder=LADDER, allowed={"t1"})


def test_final_status_mapping():
    assert dr._final_status(True, "") == "passed"
    assert dr._final_status(False, "budget") == "budget"
    assert dr._final_status(False, "step_failed") == "failed"


# ── execute_coder (end to end through the engine) ────────────────────

async def test_execute_coder_elevates_and_persists(tmp_path, store):
    (tmp_path / "test_add.py").write_text("from solution import add\n")
    req = dr.CoderRequest(
        task="implement add(a, b)", workspace=str(tmp_path),
        test_command="pytest -q", solution_path="solution.py", test_path="test_add.py",
        base_tier="t1", max_tier="t2", tiers=["t1", "t2"],
    )
    run_id = await store.create_run("coder", "u1", req.task, "t1", "t2", 0.0)

    result = await dr.execute_coder(
        store, run_id, req, tiers_map=TIERS_MAP,
        provider_factory=factory_const("```python\ndef add(a, b): return a + b  # CORRECT\n```"),
        runner=runner_passes_on("CORRECT"),
    )
    assert result.passed

    run = await store.get_run("coder", run_id)
    assert run["status"] == "passed" and run["passed"] is True
    assert "CORRECT" in run["result"]["code"]
    assert len(run["steps"]) >= 1            # the ladder log was persisted


async def test_execute_coder_records_error_status(tmp_path, store):
    # A failing provider must be recorded as an error run, not crash the loop.
    req = dr.CoderRequest(task="x", workspace=str(tmp_path),
                          base_tier="t1", max_tier="t1", tiers=["t1"])
    run_id = await store.create_run("coder", "u1", req.task, "t1", "t1", 0.0)
    await dr.execute_coder(store, run_id, req, tiers_map=TIERS_MAP,
                           provider_factory=factory_raises(),
                           runner=runner_passes_on("CORRECT"))
    run = await store.get_run("coder", run_id)
    assert run["status"] == "error" and run["error"]


# ── execute_reason (elevated via self-consistency) ───────────────────

async def test_execute_reason_passes_on_agreement(tmp_path, store):
    req = dr.ReasonRequest(
        task="what is 2+2?", base_tier="t1", max_tier="t2", tiers=["t1", "t2"],
        agreement_threshold=0.6, critic_modes=[],   # high agreement -> critics moot
        max_step_samples=3,
    )
    run_id = await store.create_run("reason", "u1", req.task, "t1", "t2", 0.0)

    result = await dr.execute_reason(
        store, run_id, req, tiers_map=TIERS_MAP,
        provider_factory=factory_const("reasoning...\nAnswer: 42"),
    )
    assert result.passed

    run = await store.get_run("reason", run_id)
    assert run["status"] == "passed"
    assert run["result"]["answer"].endswith("Answer: 42")
    assert run["steps"]                       # vote attempt logged


async def test_execute_reason_target_agreement_only_and_disposes(tmp_path, store):
    # Target generator with no critic model → agreement-only; dispose must be awaited.
    disposed = []

    async def dispose():
        disposed.append(True)

    req = dr.ReasonRequest(
        task="q", base_tier="t1", max_tier="t1", tiers=["t1"],
        agreement_threshold=0.6, max_step_samples=3,
    )
    run_id = await store.create_run("reason", "u1", req.task, "t1", "t1", 0.0)
    result = await dr.execute_reason(
        store, run_id, req, tiers_map=TIERS_MAP,
        provider_factory=factory_const("reasoning...\nAnswer: 7"),
        critic_provider=None, dispose=dispose,   # target path: critics off, cleanup on
    )
    assert result.passed                          # unanimous agreement carries it
    assert disposed == [True]                      # cleanup ran even on success
