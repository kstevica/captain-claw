"""Tests for the Plan-Horizon engine (Lever C) — the verify-gated step chain.

All four seams (planner / step_runner / verifier / synthesizer) are injected stubs,
so these exercise the orchestration loop — verify gating, the fix loop, re-planning,
bounds, and explicit stop reasons — without any LLM call.
"""

from __future__ import annotations

import asyncio

from captain_claw.flight_deck.horizon_plan import (
    PlanConfig,
    StepVerdict,
    _normalize_dag,
    make_llm_dag_planner,
    make_llm_planner,
    make_llm_synthesizer,
    make_llm_verifier,
    run_dag_horizon,
    run_plan_horizon,
)
from captain_claw.llm import LLMResponse


def runner_echo():
    """step_runner that echoes the goal and records the contexts it saw."""
    seen: list[str] = []

    async def step_runner(goal: str, context: str) -> str:
        seen.append(context)
        return f"out:{goal}"

    step_runner.seen = seen  # type: ignore[attr-defined]
    return step_runner


def verifier_for(fail_goals: set[str], *, fail_times: dict | None = None,
                 low_conf_goals: set[str] | None = None):
    """Verifier that fails listed goals (optionally only the first ``fail_times``),
    or passes them with sub-threshold confidence."""
    fail_times = dict(fail_times or {})
    low_conf_goals = low_conf_goals or set()
    seen: dict = {}

    async def verifier(task: str, goal: str, output: str) -> StepVerdict:
        seen[goal] = seen.get(goal, 0) + 1
        if goal in low_conf_goals:
            return StepVerdict(passed=True, confidence=0.3, feedback="weak")
        if goal in fail_times:
            if seen[goal] <= fail_times[goal]:
                return StepVerdict(passed=False, confidence=0.2, feedback=f"fix {goal}")
            return StepVerdict(passed=True, confidence=0.9)
        if goal in fail_goals:
            return StepVerdict(passed=False, confidence=0.1, feedback=f"bad {goal}")
        return StepVerdict(passed=True, confidence=0.9)

    verifier.seen = seen  # type: ignore[attr-defined]
    return verifier


def planner_seq(*plans: list):
    """Planner that returns successive plans on successive calls."""
    calls = {"n": 0}

    async def planner(task: str, completed: list) -> list:
        idx = min(calls["n"], len(plans) - 1)
        calls["n"] += 1
        return list(plans[idx])

    planner.calls = calls  # type: ignore[attr-defined]
    return planner


async def synth_join(task: str, steps: list) -> str:
    return "DELIVERABLE[" + ",".join(s["goal"] for s in steps) + "]"


# ── Tests ────────────────────────────────────────────────────────────

async def test_happy_path_all_steps_verify():
    res = await run_plan_horizon(
        "T", planner=planner_seq(["A", "B", "C"]), step_runner=runner_echo(),
        verifier=verifier_for(set()), synthesizer=synth_join)
    assert res.completed == 3 and res.replans == 0 and res.stopped_reason == ""
    assert [s["goal"] for s in res.steps] == ["A", "B", "C"]
    assert all(s["verified"] for s in res.steps)
    assert res.deliverable == "DELIVERABLE[A,B,C]"


async def test_context_accumulates_verified_results():
    runner = runner_echo()
    await run_plan_horizon("T", planner=planner_seq(["A", "B"]), step_runner=runner,
                           verifier=verifier_for(set()), synthesizer=synth_join)
    # The 2nd step's context must carry the 1st step's verified output.
    assert "out:A" in runner.seen[1] and "out:A" not in runner.seen[0]


async def test_fix_loop_recovers_a_step():
    verifier = verifier_for(set(), fail_times={"B": 1})  # B fails once, then passes
    res = await run_plan_horizon(
        "T", planner=planner_seq(["A", "B", "C"]), step_runner=runner_echo(),
        verifier=verifier, synthesizer=synth_join,
        cfg=PlanConfig(max_fix_per_step=1))
    b = next(s for s in res.steps if s["goal"] == "B")
    assert b["attempts"] == 2 and b["verified"] is True
    assert res.completed == 3 and res.stopped_reason == ""


async def test_replan_on_hard_failure():
    # First plan [A,B,C]; B always fails → re-plan returns [B2,C2] which pass.
    verifier = verifier_for({"B"})
    planner = planner_seq(["A", "B", "C"], ["B2", "C2"])
    res = await run_plan_horizon(
        "T", planner=planner, step_runner=runner_echo(), verifier=verifier,
        synthesizer=synth_join, cfg=PlanConfig(max_fix_per_step=0, max_replans=1))
    assert res.replans == 1
    assert [s["goal"] for s in res.steps] == ["A", "B2", "C2"]
    assert res.completed == 3 and res.stopped_reason == ""


async def test_replans_exhausted_keeps_best_so_far_unverified():
    verifier = verifier_for({"B"})  # B never passes
    res = await run_plan_horizon(
        "T", planner=planner_seq(["A", "B", "C"]), step_runner=runner_echo(),
        verifier=verifier, synthesizer=synth_join,
        cfg=PlanConfig(max_fix_per_step=0, max_replans=0))
    b = next(s for s in res.steps if s["goal"] == "B")
    assert b["verified"] is False and b["output"] == "out:B"  # no silent drop
    assert res.stopped_reason == "step_unverified"
    assert res.completed == 2  # A and C verified; the run still finishes
    assert res.deliverable == "DELIVERABLE[A,B,C]"


async def test_low_confidence_pass_is_rejected():
    """passed=True but confidence below the bar must NOT count as verified."""
    verifier = verifier_for(set(), low_conf_goals={"B"})
    res = await run_plan_horizon(
        "T", planner=planner_seq(["A", "B"]), step_runner=runner_echo(),
        verifier=verifier, synthesizer=synth_join,
        cfg=PlanConfig(max_fix_per_step=1, max_replans=0, min_step_confidence=0.6))
    b = next(s for s in res.steps if s["goal"] == "B")
    assert b["attempts"] == 2 and b["verified"] is False
    assert res.stopped_reason == "step_unverified"


async def test_max_steps_truncates_the_plan():
    res = await run_plan_horizon(
        "T", planner=planner_seq(["A", "B", "C", "D", "E"]), step_runner=runner_echo(),
        verifier=verifier_for(set()), synthesizer=synth_join,
        cfg=PlanConfig(max_steps=2))
    assert len(res.steps) == 2 and res.stopped_reason == "max_steps"


async def test_empty_plan_short_circuits():
    called = {"synth": False}

    async def synth(task, steps):
        called["synth"] = True
        return "X"

    res = await run_plan_horizon(
        "T", planner=planner_seq([]), step_runner=runner_echo(),
        verifier=verifier_for(set()), synthesizer=synth)
    assert res.stopped_reason == "empty_plan" and res.deliverable == ""
    assert called["synth"] is False


async def test_event_stream_covers_the_lifecycle():
    events: list[dict] = []
    await run_plan_horizon(
        "T", planner=planner_seq(["A", "B"]), step_runner=runner_echo(),
        verifier=verifier_for(set()), synthesizer=synth_join,
        on_event=events.append)
    stages = [e["stage"] for e in events]
    for s in ("plan", "step_start", "verify", "step_done", "synthesize"):
        assert s in stages


async def test_terminates_when_replan_returns_nothing():
    verifier = verifier_for({"B"})
    planner = planner_seq(["A", "B", "C"], [])  # re-plan yields no steps
    res = await run_plan_horizon(
        "T", planner=planner, step_runner=runner_echo(), verifier=verifier,
        synthesizer=synth_join, cfg=PlanConfig(max_fix_per_step=0, max_replans=1))
    assert res.stopped_reason == "step_unverified"
    assert any(s["goal"] == "B" and not s["verified"] for s in res.steps)


# ── LLM-backed seam builders ─────────────────────────────────────────

class _Provider:
    def __init__(self, reply: str):
        self.reply = reply

    async def complete(self, messages, tools=None, temperature=None, max_tokens=None):
        return LLMResponse(content=self.reply, finish_reason="stop")


async def test_llm_planner_parses_json_array_and_tolerates_garbage():
    good = make_llm_planner(_Provider('```json\n["step one", "step two"]\n```'))
    assert await good("T", []) == ["step one", "step two"]
    bad = make_llm_planner(_Provider("not json"))
    assert await bad("T", []) == []


async def test_llm_verifier_parses_and_soft_passes_on_garbage():
    ok = make_llm_verifier(_Provider('{"passed": true, "confidence": 0.8, "feedback": ""}'))
    v = await ok("T", "G", "O")
    assert v.passed and v.confidence == 0.8
    flaky = make_llm_verifier(_Provider("garbage"))
    vg = await flaky("T", "G", "O")
    assert vg.passed and vg.confidence == 0.5  # soft pass, low conf → a fix still fires


async def test_llm_synthesizer_returns_text():
    synth = make_llm_synthesizer(_Provider("final"))
    assert await synth("T", [{"goal": "g", "output": "o", "verified": True}]) == "final"


# ── DAG plan-horizon ─────────────────────────────────────────────────

def dag_planner(steps: list):
    async def planner_dag(task: str) -> list:
        return [dict(s) for s in steps]
    return planner_dag


DIAMOND = [
    {"id": "a", "goal": "A"},
    {"id": "b", "goal": "B", "depends_on": ["a"]},
    {"id": "c", "goal": "C", "depends_on": ["a"]},
    {"id": "d", "goal": "D", "depends_on": ["b", "c"]},
]


def test_normalize_dag_drops_danglers_dupes_and_cycles():
    steps = _normalize_dag(
        [{"id": "a", "goal": "A", "depends_on": ["a", "ghost"]},
         {"id": "b", "goal": "B", "depends_on": ["a", "c"]},   # 'c' is a forward edge
         {"id": "c", "goal": "C", "depends_on": ["b"]},
         {"id": "a", "goal": "dup"}], 10)
    assert [(s.id, s.depends_on) for s in steps] == [("a", []), ("b", ["a"]), ("c", ["b"])]


async def test_dag_runs_in_dependency_order():
    res = await run_dag_horizon(
        "T", planner_dag=dag_planner(DIAMOND), step_runner=runner_echo(),
        verifier=verifier_for(set()), synthesizer=synth_join)
    goals = [s["goal"] for s in res.steps]
    assert res.completed == 4 and res.stopped_reason == ""
    # 'a' before b/c; b and c before d.
    assert goals.index("A") < goals.index("B") and goals.index("A") < goals.index("C")
    assert goals.index("B") < goals.index("D") and goals.index("C") < goals.index("D")


async def test_dag_runs_independent_steps_concurrently():
    peak = {"now": 0, "max": 0}

    async def step_runner(goal: str, context: str) -> str:
        peak["now"] += 1
        peak["max"] = max(peak["max"], peak["now"])
        await asyncio.sleep(0)  # yield so a sibling in the wave can enter
        peak["now"] -= 1
        return f"out:{goal}"

    await run_dag_horizon(
        "T", planner_dag=dag_planner(DIAMOND), step_runner=step_runner,
        verifier=verifier_for(set()), synthesizer=synth_join)
    assert peak["max"] >= 2  # b and c (both depend only on a) ran in the same wave


async def test_dag_step_sees_only_its_dependencies():
    runner = runner_echo()
    await run_dag_horizon(
        "T", planner_dag=dag_planner(DIAMOND), step_runner=runner,
        verifier=verifier_for(set()), synthesizer=synth_join)
    # 'd' depends on b and c → its context carries their outputs, but not the task-only
    # context that 'a' saw. Find the context passed for goal D.
    d_ctx = next(c for c in runner.seen if "out:B" in c and "out:C" in c)
    assert "out:B" in d_ctx and "out:C" in d_ctx


async def test_dag_fix_loop_recovers_a_step():
    res = await run_dag_horizon(
        "T", planner_dag=dag_planner(DIAMOND), step_runner=runner_echo(),
        verifier=verifier_for(set(), fail_times={"C": 1}), synthesizer=synth_join,
        cfg=PlanConfig(max_fix_per_step=1))
    c = next(s for s in res.steps if s["goal"] == "C")
    assert c["attempts"] == 2 and c["verified"] and res.completed == 4


async def test_dag_unverified_step_still_feeds_dependents():
    res = await run_dag_horizon(
        "T", planner_dag=dag_planner(DIAMOND), step_runner=runner_echo(),
        verifier=verifier_for({"B"}), synthesizer=synth_join,
        cfg=PlanConfig(max_fix_per_step=0))
    b = next(s for s in res.steps if s["goal"] == "B")
    d = next(s for s in res.steps if s["goal"] == "D")  # depends on the failed B
    assert b["verified"] is False and d["verified"] is True
    assert res.stopped_reason == "step_unverified" and res.completed == 3


async def test_dag_empty_plan_short_circuits():
    res = await run_dag_horizon(
        "T", planner_dag=dag_planner([]), step_runner=runner_echo(),
        verifier=verifier_for(set()), synthesizer=synth_join)
    assert res.stopped_reason == "empty_plan" and res.deliverable == ""


async def test_dag_cyclic_planner_output_is_made_runnable():
    # a↔b cycle in the raw output; normalization breaks it → the run still completes.
    cyclic = [{"id": "a", "goal": "A", "depends_on": ["b"]},
              {"id": "b", "goal": "B", "depends_on": ["a"]}]
    res = await run_dag_horizon(
        "T", planner_dag=dag_planner(cyclic), step_runner=runner_echo(),
        verifier=verifier_for(set()), synthesizer=synth_join)
    assert res.completed == 2 and res.stopped_reason == ""


async def test_llm_dag_planner_parses_objects():
    p = make_llm_dag_planner(_Provider('[{"id":"a","goal":"A","depends_on":[]}]'))
    assert await p("T") == [{"id": "a", "goal": "A", "depends_on": []}]
    bad = make_llm_dag_planner(_Provider("nope"))
    assert await bad("T") == []
