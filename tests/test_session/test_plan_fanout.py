"""Tests for orchestrate fan-out expansion in PlanExecutor (step 5)."""

from __future__ import annotations

from typing import Any

import pytest

from captain_claw.plan_mode import (
    PlanExecutor,
    orchestrate_expander_from_orchestrator,
)
from captain_claw.task_graph import COMPLETED, OrchestratorTask, TaskGraph


def _build_graph(tasks: list[OrchestratorTask]) -> TaskGraph:
    g = TaskGraph(max_parallel=5)
    for t in tasks:
        g.add_task(t)
    return g


class _StubOrchestrator:
    """Stand-in for SessionOrchestrator with optional ``_decompose`` hook."""

    def __init__(
        self,
        graph: TaskGraph | None,
        execute_impl: Any = None,
        decompose_impl: Any = None,
    ):
        self._graph = graph
        self._execute_impl = execute_impl
        self._decompose_impl = decompose_impl
        self.execute_calls = 0
        self.decompose_calls: list[str] = []

    async def execute(self) -> str:
        self.execute_calls += 1
        if self._execute_impl is None:
            return ""
        return await self._execute_impl(self._graph)

    async def _decompose(self, prompt: str) -> dict[str, Any] | None:
        self.decompose_calls.append(prompt)
        if self._decompose_impl is None:
            return None
        return await self._decompose_impl(prompt)


# ---------------------------------------------------------------------------
# PlanExecutor._expand_orchestrate_steps — graph mutation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_expand_replaces_orchestrate_step_with_subtasks_and_join():
    """An orchestrate step P with deps=[a] is replaced by [s1, s2] || join P.

    After expansion:
      - s1, s2 are new atomic tasks with depends_on=[a]
      - P remains, becomes atomic, depends_on=[s1, s2]
      - downstream task c keeps depends_on=[P] (no rewire needed)
    """
    a = OrchestratorTask(id="a", title="A", description="x")
    p = OrchestratorTask(
        id="p", title="Process files", description="Process N files",
        depends_on=["a"], step_kind="orchestrate",
        acceptance_criteria="all files processed",
    )
    c = OrchestratorTask(id="c", title="C", description="z", depends_on=["p"])
    graph = _build_graph([a, p, c])

    async def expand(task: OrchestratorTask) -> list[OrchestratorTask]:
        return [
            OrchestratorTask(id="s1", title="proc 1", description="d1"),
            OrchestratorTask(id="s2", title="proc 2", description="d2"),
        ]

    orch = _StubOrchestrator(graph=graph)
    executor = PlanExecutor(orch, expander=expand)

    expansions = await executor._expand_orchestrate_steps(graph)

    assert len(expansions) == 1
    assert expansions[0]["expanded"] is True
    assert expansions[0]["task_id"] == "p"
    sub_ids = expansions[0]["sub_task_ids"]
    assert len(sub_ids) == 2

    # Sub-tasks are namespaced under the parent and inherit parent's deps.
    s1 = graph.tasks[sub_ids[0]]
    s2 = graph.tasks[sub_ids[1]]
    assert s1.id.startswith("p__")
    assert s2.id.startswith("p__")
    assert s1.depends_on == ["a"]
    assert s2.depends_on == ["a"]
    assert s1.step_kind == "atomic"
    assert s2.step_kind == "atomic"

    # Parent becomes a join.
    parent = graph.tasks["p"]
    assert parent.step_kind == "atomic"
    assert sorted(parent.depends_on) == sorted(sub_ids)
    assert "Synthesize" in parent.description
    assert "Process N files" in parent.description  # original goal preserved
    # Acceptance criteria preserved on the join step → verifier still runs.
    assert parent.acceptance_criteria == "all files processed"

    # Downstream c still depends on p (the join), no rewiring needed.
    assert graph.tasks["c"].depends_on == ["p"]


@pytest.mark.asyncio
async def test_expand_falls_back_to_atomic_when_expander_returns_empty():
    p = OrchestratorTask(id="p", title="P", description="d",
                         step_kind="orchestrate")
    graph = _build_graph([p])

    async def expand(_t: OrchestratorTask) -> list[OrchestratorTask]:
        return []

    executor = PlanExecutor(_StubOrchestrator(graph=graph), expander=expand)
    expansions = await executor._expand_orchestrate_steps(graph)

    assert expansions[0]["expanded"] is False
    assert "no sub-tasks" in expansions[0]["error"]
    assert graph.tasks["p"].step_kind == "atomic"
    assert graph.task_count == 1  # nothing added


@pytest.mark.asyncio
async def test_expand_falls_back_to_atomic_when_expander_raises():
    p = OrchestratorTask(id="p", title="P", description="d",
                         step_kind="orchestrate")
    graph = _build_graph([p])

    async def expand(_t: OrchestratorTask) -> list[OrchestratorTask]:
        raise RuntimeError("decomposer down")

    executor = PlanExecutor(_StubOrchestrator(graph=graph), expander=expand)
    expansions = await executor._expand_orchestrate_steps(graph)

    assert expansions[0]["expanded"] is False
    assert "decomposer down" in expansions[0]["error"]
    assert graph.tasks["p"].step_kind == "atomic"


@pytest.mark.asyncio
async def test_expand_disambiguates_subtask_id_collisions():
    """If a sub-task id collides with an existing plan step, suffix-disambiguate."""
    # Pre-existing step with id "p__sub_1" — sub-task should become "p__sub_1_2".
    existing = OrchestratorTask(id="p__sub_1", title="legacy", description="x")
    p = OrchestratorTask(id="p", title="P", description="d",
                         step_kind="orchestrate")
    graph = _build_graph([existing, p])

    async def expand(_t: OrchestratorTask) -> list[OrchestratorTask]:
        return [OrchestratorTask(id="", title="auto1", description="d")]

    executor = PlanExecutor(_StubOrchestrator(graph=graph), expander=expand)
    expansions = await executor._expand_orchestrate_steps(graph)

    # Default-generated id is "p__sub_1" → collides → bumped to "p__sub_1_2".
    new_id = expansions[0]["sub_task_ids"][0]
    assert new_id == "p__sub_1_2"
    assert new_id in graph.tasks
    assert "p__sub_1" in graph.tasks  # original preserved


# ---------------------------------------------------------------------------
# PlanExecutor.run() integration — end-to-end with expander
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_expands_orchestrate_before_executing():
    """Full path: expand → execute → verify (no verifier here)."""
    p = OrchestratorTask(id="p", title="P", description="fan it out",
                         step_kind="orchestrate")
    graph = _build_graph([p])

    async def expand(_t: OrchestratorTask) -> list[OrchestratorTask]:
        return [
            OrchestratorTask(id="x1", title="x1", description="d"),
            OrchestratorTask(id="x2", title="x2", description="d"),
        ]

    async def execute_impl(g: TaskGraph) -> str:
        for t in g.tasks.values():
            t.status = COMPLETED
            t.result = {"output": f"out-{t.id}"}
        return "all done"

    orch = _StubOrchestrator(graph=graph, execute_impl=execute_impl)
    events: list[dict[str, Any]] = []
    executor = PlanExecutor(orch, broadcast=events.append, expander=expand)

    result = await executor.run()

    assert result.ok is True
    # Plan step "p" plus 2 expanded sub-tasks were all completed.
    assert "p" in result.completed_steps
    assert len([s for s in result.completed_steps if s.startswith("p__")]) == 2

    types = [e["type"] for e in events]
    assert "plan_orchestrate_expanded" in types
    expand_event = next(e for e in events if e["type"] == "plan_orchestrate_expanded")
    assert expand_event["expansions"][0]["expanded"] is True


@pytest.mark.asyncio
async def test_run_without_expander_falls_back_to_coercion():
    """No expander → orchestrate step is coerced to atomic (legacy step-3 path)."""
    p = OrchestratorTask(id="p", title="P", description="d",
                         step_kind="orchestrate")
    graph = _build_graph([p])

    async def execute_impl(g: TaskGraph) -> str:
        for t in g.tasks.values():
            t.status = COMPLETED
            t.result = {"output": "ok"}
        return ""

    orch = _StubOrchestrator(graph=graph, execute_impl=execute_impl)
    executor = PlanExecutor(orch, expander=None)

    result = await executor.run()

    assert result.ok is True
    assert graph.tasks["p"].step_kind == "atomic"
    assert graph.task_count == 1


# ---------------------------------------------------------------------------
# orchestrate_expander_from_orchestrator factory
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_factory_calls_orchestrator_decompose():
    async def decompose_impl(prompt: str) -> dict[str, Any]:
        return {
            "summary": "split",
            "tasks": [
                {"id": "step_a", "title": "A", "description": "do a",
                 "depends_on": []},
                {"id": "step_b", "title": "B", "description": "do b",
                 "depends_on": ["step_a"]},  # sibling dep — should be dropped
            ],
        }

    orch = _StubOrchestrator(graph=None, decompose_impl=decompose_impl)
    expand = orchestrate_expander_from_orchestrator(orch)

    parent = OrchestratorTask(
        id="p", title="parent", description="fan-out goal",
        step_kind="orchestrate",
    )
    subs = await expand(parent)

    assert orch.decompose_calls == ["fan-out goal"]
    assert [s.title for s in subs] == ["A", "B"]
    # All sub-tasks are atomic regardless of decomposer output.
    assert all(s.step_kind == "atomic" for s in subs)
    # Sibling deps are stripped — fan-out is parallel.
    assert all(s.depends_on == [] for s in subs)


@pytest.mark.asyncio
async def test_factory_returns_empty_when_decompose_none():
    async def decompose_impl(_: str) -> dict[str, Any] | None:
        return None

    orch = _StubOrchestrator(graph=None, decompose_impl=decompose_impl)
    expand = orchestrate_expander_from_orchestrator(orch)

    subs = await expand(OrchestratorTask(id="p", title="t", description="d"))
    assert subs == []


@pytest.mark.asyncio
async def test_factory_returns_empty_when_orchestrator_lacks_decompose():
    class NoDecomp:
        pass

    expand = orchestrate_expander_from_orchestrator(NoDecomp())
    subs = await expand(OrchestratorTask(id="p", title="t", description="d"))
    assert subs == []
