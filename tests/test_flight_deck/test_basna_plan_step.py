"""Tests for the ensemble-per-step runner of the Basna Plan-Horizon.

route/execute are injected stubs, so this exercises the per-step wiring (route the
step → execute the child ensemble → return its merged truth) without spawning agents
or touching a real DB.
"""

from __future__ import annotations

import json

from captain_claw.flight_deck.basna_routes import (
    PlanRequest,
    make_basna_ensemble_step_runner,
    make_vatra_team_step_runner,
)

USER = {"id": "u1"}
BODY = PlanRequest(intent="T", step_mode="ensemble", step_max_agents=4,
                   tiers={"fast": {"provider": "openai", "model": "m-fast"}})


async def test_step_runs_a_child_ensemble_and_returns_its_truth():
    routed: list[str] = []
    executed: list[str] = []
    on_step: list = []

    async def route_fn(rr, user):
        routed.append(rr.intent)
        # The Library fast-tier creds are threaded into the child router.
        assert rr.provider == "openai" and rr.model == "m-fast"
        assert rr.max_agents == 4
        return {"session_id": "child-1"}

    async def execute_fn(er, request, user):
        executed.append(er.session_id)
        return {"truth": "merged answer", "agents": [1, 2, 3], "confidence": 0.87}

    runner = make_basna_ensemble_step_runner(
        "parent-1", BODY, USER, route_fn=route_fn, execute_fn=execute_fn,
        on_step=lambda c, r: on_step.append((c, r.get("confidence"))))
    out = await runner("do the thing", "context: prior verified results")

    assert out == "merged answer"
    # The step goal + accumulated context both reach the child intent.
    assert "do the thing" in routed[0] and "prior verified results" in routed[0]
    assert executed == ["child-1"]
    assert on_step == [("child-1", 0.87)]


async def test_step_returns_empty_when_route_makes_no_session():
    async def route_fn(rr, user):
        return {}

    async def execute_fn(er, request, user):  # pragma: no cover
        raise AssertionError("must not execute without a child session")

    runner = make_basna_ensemble_step_runner(
        "p", BODY, USER, route_fn=route_fn, execute_fn=execute_fn)
    assert await runner("g", "c") == ""


async def test_step_returns_empty_truth_gracefully():
    async def route_fn(rr, user):
        return {"session_id": "child-x"}

    async def execute_fn(er, request, user):
        return {"truth": "", "agents": []}  # ensemble produced nothing usable

    runner = make_basna_ensemble_step_runner(
        "p", BODY, USER, route_fn=route_fn, execute_fn=execute_fn)
    assert await runner("g", "c") == ""


# ── Vatra-team step runner ───────────────────────────────────────────

VBODY = PlanRequest(intent="T", step_mode="vatra", step_max_agents=3,
                    tiers={"fast": {"provider": "openai", "model": "m"}})


async def test_step_runs_a_child_vatra_team_and_returns_its_deliverable():
    created: list = []
    executed: list[str] = []
    on_step: list = []

    async def create_fn(intent, config_json, title):
        created.append((intent, json.loads(config_json), title))
        return {"id": "child-v1"}

    async def execute_fn(er, request, user):
        executed.append(er.session_id)
        return {"truth": "team deliverable", "subtasks": [1, 2], "confidence": 0.8}

    runner = make_vatra_team_step_runner(
        "parent-1", VBODY, USER, create_session_fn=create_fn, execute_fn=execute_fn,
        on_step=lambda c, r: on_step.append((c, len(r.get("subtasks") or []))))
    out = await runner("build the thing", "context: prior results")

    assert out == "team deliverable"
    assert "build the thing" in created[0][0] and "prior results" in created[0][0]
    cfg = created[0][1]
    assert cfg["mode"] == "vatra" and cfg["source"] == "plan-step"
    assert cfg["parent"] == "parent-1" and cfg["max_agents"] == 3
    assert executed == ["child-v1"]
    assert on_step == [("child-v1", 2)]


async def test_vatra_step_empty_when_session_creation_fails():
    async def create_fn(intent, config_json, title):
        return {}

    async def execute_fn(er, request, user):  # pragma: no cover
        raise AssertionError("must not execute without a child session")

    runner = make_vatra_team_step_runner(
        "p", VBODY, USER, create_session_fn=create_fn, execute_fn=execute_fn)
    assert await runner("g", "c") == ""
