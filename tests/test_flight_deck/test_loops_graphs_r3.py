"""R3 tests — kill false sequential edges; validate crossing artifacts.

Covers: the two new opt-in flags, the pure data-edge lint, the runtime
parallel-first directive + gated edge-drop in the plan generator, and the
opt-in Flows {{steps.<id>}} reference validation.
"""

from __future__ import annotations

import json

import pytest

from captain_claw import plan_mode, task_graph
from captain_claw.flight_deck import flow_dsl
from captain_claw.flight_deck.quality_profile import (
    PARALLEL_EDGES_DIRECTIVE,
    QualityProfile,
)


# ── flags ────────────────────────────────────────────────────────────────
def test_r3_flags_default_off_and_explicit_on():
    off = QualityProfile.from_dict(None)
    assert off.parallel_edges is False
    assert off.flow_ref_lint is False
    on = QualityProfile.from_dict({"parallel_edges": True, "flow_ref_lint": True})
    assert on.parallel_edges is True and on.flow_ref_lint is True
    assert on.any_enabled is True


def test_r3_flags_not_enabled_by_presets():
    for preset in ("off", "balanced", "thorough"):
        q = QualityProfile.from_dict({"profile": preset})
        assert q.parallel_edges is False, preset
        assert q.flow_ref_lint is False, preset


# ── data-edge lint ─────────────────────────────────────────────────────────
def _task(tid, deps=None, ins=None, outs=None):
    return task_graph.OrchestratorTask(
        id=tid, title=tid, depends_on=list(deps or []),
        workspace_inputs=list(ins or []), workspace_outputs=list(outs or []),
    )


def test_lint_reports_unnamed_edges_but_they_are_advisory():
    # An edge where neither side names an artifact is REPORTED (reason set), but
    # the report only drives a debug log by default — it is acted on (dropped)
    # solely when parallel_edges is enabled (see the plan-generator tests, where
    # quality=None keeps the edge and parallel_edges drops it). So the default
    # path is a functional no-op even though findings are non-empty.
    tasks = [_task("a"), _task("b", deps=["a"])]
    f = task_graph.lint_data_edges(tasks)
    assert len(f) == 1 and f[0]["kind"] == "no_crossing_edge"
    assert f[0].get("reason")


def test_lint_crossing_edge_is_clean():
    tasks = [_task("a", outs=["x"]), _task("b", deps=["a"], ins=["x"])]
    assert task_graph.lint_data_edges(tasks) == []


def test_lint_flags_no_crossing_and_unproduced():
    tasks = [_task("a", outs=["x"]), _task("b", deps=["a"], ins=["y"])]
    f = task_graph.lint_data_edges(tasks)
    kinds = {x["kind"] for x in f}
    assert "no_crossing_edge" in kinds       # a→b: produces x, consumes y
    assert "unproduced_input" in kinds       # nobody produces y


def test_lint_unnamed_edge_and_dangling_dep():
    # Neither side names anything → reported with a reason; dangling dep skipped.
    tasks = [_task("a"), _task("b", deps=["a", "ghost"])]
    f = task_graph.lint_data_edges(tasks)
    assert any(x.get("reason") for x in f if x["kind"] == "no_crossing_edge")
    assert all(x.get("depends_on") != "ghost" for x in f)  # dangling not asserted


# ── Flows strict refs ──────────────────────────────────────────────────────
def _flow(prompt_b):
    return {
        "steps": [
            {"id": "a", "type": "agent", "prompt": "do a"},
            {"id": "b", "type": "agent", "prompt": prompt_b},
        ],
        "output": {"channel": "same"},
    }


def test_flow_ref_lint_off_is_tolerant():
    flow = _flow("use {{steps.MISSING.output}}")
    errs = flow_dsl.validate_flow(flow, strict_refs=False)
    assert not any("names no step id" in e for e in errs)


def test_flow_ref_lint_on_rejects_typo_only():
    bad = flow_dsl.validate_flow(_flow("use {{steps.MISSING.output}}"), strict_refs=True)
    assert any("names no step id" in e for e in bad)
    good = flow_dsl.validate_flow(_flow("use {{steps.a.output}}"), strict_refs=True)
    assert not any("names no step id" in e for e in good)
    # Non-step refs are never flagged.
    other = flow_dsl.validate_flow(_flow("{{input}} {{vars.x}} {{trigger.y}}"), strict_refs=True)
    assert not any("names no step id" in e for e in other)


# ── plan generator: directive + gated edge-drop ────────────────────────────
class _StubProvider:
    def __init__(self, content: str):
        self._content = content
        self.calls: list = []

    async def complete(self, messages=None, tools=None, max_tokens=None, **kw):
        self.calls.append({"messages": messages})
        return type("R", (), {"content": self._content})()


_PLAN_JSON = json.dumps({
    "summary": "s",
    "tasks": [
        {"id": "a", "title": "A", "workspace_outputs": ["x"]},
        {"id": "b", "title": "B", "depends_on": ["a"], "workspace_inputs": ["y"]},
    ],
})


@pytest.mark.asyncio
async def test_plan_default_keeps_edge_no_directive():
    prov = _StubProvider(_PLAN_JSON)
    gen = plan_mode.PlanGenerator(provider=prov)
    plan = await gen.generate("build a thing")
    assert plan is not None
    b = next(t for t in plan.tasks if t.id == "b")
    assert b.depends_on == ["a"]  # not dropped by default
    sys_msg = prov.calls[-1]["messages"][0].content
    assert PARALLEL_EDGES_DIRECTIVE.strip()[:30] not in sys_msg  # directive absent


@pytest.mark.asyncio
async def test_plan_parallel_edges_appends_directive_and_drops_edge():
    prov = _StubProvider(_PLAN_JSON)
    gen = plan_mode.PlanGenerator(provider=prov)
    plan = await gen.generate("build a thing",
                              quality=QualityProfile.from_dict({"parallel_edges": True}))
    assert plan is not None
    sys_msg = prov.calls[-1]["messages"][0].content
    assert "DEPENDENCIES ARE DATA EDGES" in sys_msg      # directive appended
    b = next(t for t in plan.tasks if t.id == "b")
    assert b.depends_on == []  # no-crossing edge a→b dropped
