"""Tests for PlanExecutor (step 3 of plan-mode)."""

from __future__ import annotations

from typing import Any

import pytest

from captain_claw.plan_mode import PlanExecutor, PlanExecutionResult
from captain_claw.task_graph import COMPLETED, FAILED, OrchestratorTask, TaskGraph


def _build_graph(tasks: list[OrchestratorTask]) -> TaskGraph:
    graph = TaskGraph(max_parallel=5)
    for t in tasks:
        graph.add_task(t)
    return graph


class _StubOrchestrator:
    """Stand-in for SessionOrchestrator.

    ``execute()`` is a pluggable async function so each test can simulate
    success, failure, or exception without spinning up a real DAG runner.
    """

    def __init__(
        self,
        graph: TaskGraph | None,
        execute_impl: Any,
    ):
        self._graph = graph
        self._execute_impl = execute_impl
        self.execute_calls = 0

    async def execute(self) -> str:
        self.execute_calls += 1
        return await self._execute_impl(self._graph)


@pytest.mark.asyncio
async def test_run_returns_error_when_no_graph_loaded():
    orch = _StubOrchestrator(graph=None, execute_impl=None)
    events: list[dict[str, Any]] = []
    executor = PlanExecutor(orch, broadcast=events.append)

    result = await executor.run()

    assert isinstance(result, PlanExecutionResult)
    assert result.ok is False
    assert "No plan loaded" in result.error
    assert orch.execute_calls == 0
    assert events == []  # nothing to broadcast — short-circuit


@pytest.mark.asyncio
async def test_run_executes_and_marks_steps_completed():
    tasks = [
        OrchestratorTask(id="a", title="Step A", description="x",
                         acceptance_criteria="ok"),
        OrchestratorTask(id="b", title="Step B", description="y",
                         depends_on=["a"], acceptance_criteria="ok"),
    ]
    graph = _build_graph(tasks)

    async def execute_impl(g: TaskGraph) -> str:
        for t in g.tasks.values():
            t.status = COMPLETED
        return "all good"

    orch = _StubOrchestrator(graph=graph, execute_impl=execute_impl)
    events: list[dict[str, Any]] = []
    executor = PlanExecutor(orch, broadcast=events.append)

    result = await executor.run()

    assert result.ok is True
    assert result.final_output == "all good"
    assert result.completed_steps == ["a", "b"]
    assert orch.execute_calls == 1

    types = [e["type"] for e in events]
    # No verifier configured → verification is a no-op pass.
    assert types == ["plan_execution_started", "plan_execution_verified"]
    assert events[0]["step_count"] == 2
    assert events[0]["steps"][0]["acceptance_criteria"] == "ok"
    assert events[-1]["verified"] == ["a", "b"]


@pytest.mark.asyncio
async def test_run_reports_failed_step_with_error_message():
    tasks = [
        OrchestratorTask(id="a", title="A", description="x"),
        OrchestratorTask(id="b", title="B", description="y", depends_on=["a"]),
    ]
    graph = _build_graph(tasks)

    async def execute_impl(g: TaskGraph) -> str:
        g.tasks["a"].status = COMPLETED
        g.tasks["b"].status = FAILED
        g.tasks["b"].error = "boom"
        return "partial output"

    orch = _StubOrchestrator(graph=graph, execute_impl=execute_impl)
    events: list[dict[str, Any]] = []
    executor = PlanExecutor(orch, broadcast=events.append)

    result = await executor.run()

    assert result.ok is False
    assert result.failed_step == "b"
    assert result.error == "boom"
    assert result.completed_steps == ["a"]
    assert result.final_output == "partial output"
    # Execution-level failure short-circuits before verification.
    assert events[-1]["type"] == "plan_execution_completed"
    assert events[-1]["has_failures"] is True


@pytest.mark.asyncio
async def test_run_reports_orchestrator_exception():
    tasks = [OrchestratorTask(id="a", title="A", description="x")]
    graph = _build_graph(tasks)

    async def execute_impl(_g: TaskGraph) -> str:
        raise RuntimeError("provider down")

    orch = _StubOrchestrator(graph=graph, execute_impl=execute_impl)
    events: list[dict[str, Any]] = []
    executor = PlanExecutor(orch, broadcast=events.append)

    result = await executor.run()

    assert result.ok is False
    assert result.error == "provider down"
    assert result.completed_steps == []
    assert "plan_execution_failed" in [e["type"] for e in events]


@pytest.mark.asyncio
async def test_run_coerces_orchestrate_steps_to_atomic():
    """Step 5 will replace this — for v3, fan-out steps run as atomic."""
    tasks = [
        OrchestratorTask(id="a", title="A", description="x", step_kind="atomic"),
        OrchestratorTask(id="b", title="B", description="y", step_kind="orchestrate"),
    ]
    graph = _build_graph(tasks)

    async def execute_impl(g: TaskGraph) -> str:
        for t in g.tasks.values():
            t.status = COMPLETED
        return ""

    orch = _StubOrchestrator(graph=graph, execute_impl=execute_impl)
    executor = PlanExecutor(orch)

    await executor.run()

    assert graph.tasks["a"].step_kind == "atomic"
    assert graph.tasks["b"].step_kind == "atomic"  # coerced


@pytest.mark.asyncio
async def test_run_works_without_broadcast_callback():
    """Broadcast is optional — non-web entry points (CLI, tests) skip it."""
    tasks = [OrchestratorTask(id="a", title="A", description="x")]
    graph = _build_graph(tasks)

    async def execute_impl(g: TaskGraph) -> str:
        g.tasks["a"].status = COMPLETED
        return "done"

    orch = _StubOrchestrator(graph=graph, execute_impl=execute_impl)
    executor = PlanExecutor(orch, broadcast=None)

    result = await executor.run()
    assert result.ok is True


@pytest.mark.asyncio
async def test_run_swallows_broadcast_errors():
    """A broken UI broadcast must not break plan execution."""
    tasks = [OrchestratorTask(id="a", title="A", description="x")]
    graph = _build_graph(tasks)

    async def execute_impl(g: TaskGraph) -> str:
        g.tasks["a"].status = COMPLETED
        return "done"

    def angry_broadcast(_payload: dict[str, Any]) -> None:
        raise RuntimeError("ws closed")

    orch = _StubOrchestrator(graph=graph, execute_impl=execute_impl)
    executor = PlanExecutor(orch, broadcast=angry_broadcast)

    result = await executor.run()
    assert result.ok is True  # broadcast errors must not propagate
