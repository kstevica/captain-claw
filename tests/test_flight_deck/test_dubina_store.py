"""Tests for the Dubina run store (Phase 3): split per-track runs + step log."""

from __future__ import annotations

import json
import math

import pytest

from captain_claw.flight_deck.dubina_store import DubinaStore


@pytest.fixture
async def store(tmp_path):
    s = DubinaStore(tmp_path / "dubina.db")
    await s.init()
    yield s
    await s.close()


async def test_create_finish_and_get_with_steps(store):
    run_id = await store.create_run(
        "coder", "u1", "implement add", "gemini-flash", "gpt-5.3-codex", 0.0,
        config={"max_step_samples": 3},
    )
    await store.append_step(run_id, "coder", 0,
                            {"step": "task", "tier": "gemini-flash", "rung": 0,
                             "kind": "single", "samples": 1, "passed": False, "confidence": 0.0})
    await store.append_step(run_id, "coder", 1,
                            {"step": "task", "tier": "gpt-5.3-codex", "rung": 2,
                             "kind": "single", "samples": 1, "passed": True, "confidence": 1.0})
    await store.finish_run("coder", run_id, status="passed", passed=True,
                           stopped_reason="", cost_spent=9.0, result={"code": "def add..."})

    run = await store.get_run("coder", run_id)
    assert run["status"] == "passed" and run["passed"] is True
    assert run["cost_spent"] == 9.0
    assert run["result"] == {"code": "def add..."}
    assert run["config"] == {"max_step_samples": 3}
    assert [st["seq"] for st in run["steps"]] == [0, 1]
    assert run["steps"][1]["tier"] == "gpt-5.3-codex" and run["steps"][1]["passed"] == 1


async def test_runs_are_split_per_track(store):
    await store.create_run("coder", "u1", "code task", "gemini-flash", "gpt-5.3-codex", 0.0)
    await store.create_run("reason", "u1", "reason task", "gemini-flash", "claude-opus", 0.0)

    coder = await store.list_runs("coder", "u1")
    reason = await store.list_runs("reason", "u1")
    assert len(coder) == 1 and coder[0]["task"] == "code task"
    assert len(reason) == 1 and reason[0]["task"] == "reason task"


async def test_list_runs_scoped_by_user(store):
    await store.create_run("coder", "u1", "mine", "gemini-flash", "gpt-5.3-codex", 0.0)
    await store.create_run("coder", "u2", "theirs", "gemini-flash", "gpt-5.3-codex", 0.0)
    assert [r["task"] for r in await store.list_runs("coder", "u1")] == ["mine"]


async def test_unknown_track_rejected(store):
    with pytest.raises(ValueError):
        await store.create_run("nope", "u1", "x", "a", "b", 0.0)


async def test_non_finite_floats_are_json_safe(store):
    """An unbounded (inf) budget or a NaN confidence must not break JSON responses.

    FastAPI renders with allow_nan=False, so inf/nan reaching the endpoint 500s it.
    The store must collapse them to a finite sentinel on store and on read.
    """
    run_id = await store.create_run(
        "reason", "u1", "t", "fast", "reason", math.inf,  # unbounded → inf budget
    )
    await store.finish_run("reason", run_id, status="passed", passed=True,
                           stopped_reason="", cost_spent=math.inf,
                           result={"confidence": math.nan, "answer": "ok"})

    for run in (await store.get_run("reason", run_id), (await store.list_runs("reason", "u1"))[0]):
        # Every float must be finite, and the whole dict must survive strict JSON.
        assert math.isfinite(run["compute_budget"]) and run["compute_budget"] == 0.0
        assert math.isfinite(run["cost_spent"])
        json.dumps(run, allow_nan=False)  # would raise ValueError on inf/nan
