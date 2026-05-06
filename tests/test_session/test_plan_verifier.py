"""Tests for PlanVerifier and PlanExecutor's verification gate (step 4)."""

from __future__ import annotations

import json
from typing import Any

import pytest

from captain_claw.llm import LLMResponse, Message
from captain_claw.plan_mode import (
    PlanExecutor,
    PlanVerifier,
    VerificationOutcome,
)
from captain_claw.task_graph import COMPLETED, OrchestratorTask, TaskGraph


class _StubProvider:
    """Records calls and returns a canned response."""

    def __init__(self, content: str):
        self._content = content
        self.calls: list[dict[str, Any]] = []

    async def complete(
        self,
        messages: list[Message],
        tools: Any = None,
        temperature: Any = None,
        max_tokens: Any = None,
    ) -> LLMResponse:
        self.calls.append({"messages": messages, "max_tokens": max_tokens})
        return LLMResponse(content=self._content, finish_reason="stop")


class _StubOrchestrator:
    """Stand-in for SessionOrchestrator."""

    def __init__(self, graph: TaskGraph | None, execute_impl: Any):
        self._graph = graph
        self._execute_impl = execute_impl
        self.execute_calls = 0

    async def execute(self) -> str:
        self.execute_calls += 1
        return await self._execute_impl(self._graph)


def _build_graph(tasks: list[OrchestratorTask]) -> TaskGraph:
    g = TaskGraph(max_parallel=5)
    for t in tasks:
        g.add_task(t)
    return g


# ---------------------------------------------------------------------------
# PlanVerifier.verify() — direct unit tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_verify_passes_via_llm_judge():
    provider = _StubProvider(json.dumps({
        "passed": True,
        "notes": "Output contains 3 sections as required.",
    }))
    verifier = PlanVerifier(provider=provider)
    task = OrchestratorTask(
        id="step",
        title="Write digest",
        description="Make sections",
        acceptance_criteria="Output contains at least 3 sections.",
    )

    outcome = await verifier.verify(task, "## A\n## B\n## C")

    assert isinstance(outcome, VerificationOutcome)
    assert outcome.passed is True
    assert "3 sections" in outcome.notes
    assert len(provider.calls) == 1
    # verifier prompts should have been used.
    assert provider.calls[0]["messages"][0].role == "system"


@pytest.mark.asyncio
async def test_verify_fails_when_judge_returns_false():
    provider = _StubProvider(json.dumps({
        "passed": False,
        "notes": "Only 1 section present, criteria requires 3.",
    }))
    task = OrchestratorTask(
        id="x", title="t", description="d",
        acceptance_criteria="Output contains at least 3 sections.",
    )

    outcome = await PlanVerifier(provider=provider).verify(task, "## only one")

    assert outcome.passed is False
    assert "Only 1 section" in outcome.notes


@pytest.mark.asyncio
async def test_verify_short_circuits_when_no_criteria_or_schema():
    provider = _StubProvider("UNUSED")
    task = OrchestratorTask(id="x", title="t", description="d")  # no criteria

    outcome = await PlanVerifier(provider=provider).verify(task, "anything")

    assert outcome.passed is True
    assert "auto-passed" in outcome.notes.lower()
    assert provider.calls == []  # judge never called


@pytest.mark.asyncio
async def test_verify_fails_on_empty_output_with_criteria():
    provider = _StubProvider("UNUSED")
    task = OrchestratorTask(
        id="x", title="t", description="d",
        acceptance_criteria="Output must list at least one PDF.",
    )

    outcome = await PlanVerifier(provider=provider).verify(task, "")

    assert outcome.passed is False
    assert "no output" in outcome.notes.lower()
    assert provider.calls == []  # short-circuited before LLM


@pytest.mark.asyncio
async def test_verify_runs_schema_gate_first_and_fails_fast():
    """If output_schema is set and validation fails, judge is never called."""
    provider = _StubProvider(json.dumps({"passed": True, "notes": "would-pass"}))
    schema = {
        "type": "object",
        "properties": {"summary": {"type": "string"}},
        "required": ["summary"],
    }
    task = OrchestratorTask(
        id="x", title="t", description="d",
        acceptance_criteria="Has a summary.",
        output_schema=schema,
    )

    outcome = await PlanVerifier(provider=provider).verify(task, "not even json")

    assert outcome.passed is False
    assert "schema" in outcome.notes.lower()
    assert outcome.schema_error
    assert provider.calls == []  # judge skipped on schema failure


@pytest.mark.asyncio
async def test_verify_schema_passes_then_runs_judge():
    """Valid schema → judge still runs against acceptance_criteria."""
    provider = _StubProvider(json.dumps({"passed": True, "notes": "looks good"}))
    schema = {"type": "object", "properties": {"summary": {"type": "string"}},
              "required": ["summary"]}
    task = OrchestratorTask(
        id="x", title="t", description="d",
        acceptance_criteria="Summary mentions widgets.",
        output_schema=schema,
    )

    output = json.dumps({"summary": "the widgets are fine"})
    outcome = await PlanVerifier(provider=provider).verify(task, output)

    assert outcome.passed is True
    assert len(provider.calls) == 1


@pytest.mark.asyncio
async def test_verify_handles_unparseable_judge_response():
    provider = _StubProvider("not even json — but explanatory prose")
    task = OrchestratorTask(
        id="x", title="t", description="d",
        acceptance_criteria="Anything.",
    )

    outcome = await PlanVerifier(provider=provider).verify(task, "some output")

    assert outcome.passed is False
    assert "could not be parsed" in outcome.notes.lower()


@pytest.mark.asyncio
async def test_verify_handles_provider_exception():
    class BoomProvider:
        async def complete(self, **_kw: Any) -> LLMResponse:
            raise RuntimeError("provider down")

    task = OrchestratorTask(
        id="x", title="t", description="d",
        acceptance_criteria="Anything.",
    )

    outcome = await PlanVerifier(provider=BoomProvider()).verify(task, "out")

    assert outcome.passed is False
    assert "provider down" in outcome.notes


# ---------------------------------------------------------------------------
# PlanExecutor + verifier integration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_executor_marks_steps_verified_on_pass():
    tasks = [
        OrchestratorTask(id="a", title="A", description="x",
                         acceptance_criteria="ok"),
        OrchestratorTask(id="b", title="B", description="y",
                         depends_on=["a"], acceptance_criteria="ok"),
    ]
    graph = _build_graph(tasks)

    async def execute_impl(g: TaskGraph) -> str:
        for t in g.tasks.values():
            t.status = COMPLETED
            t.result = {"output": "step done"}
        return "all good"

    orch = _StubOrchestrator(graph=graph, execute_impl=execute_impl)
    provider = _StubProvider(json.dumps({"passed": True, "notes": "yes"}))
    verifier = PlanVerifier(provider=provider)

    events: list[dict[str, Any]] = []
    executor = PlanExecutor(orch, broadcast=events.append, verifier=verifier)
    result = await executor.run()

    assert result.ok is True
    assert result.verified_steps == ["a", "b"]
    assert graph.tasks["a"].verification_status == "passed"
    assert graph.tasks["b"].verification_status == "passed"
    assert "yes" in graph.tasks["a"].verification_notes

    types = [e["type"] for e in events]
    assert types.count("plan_step_verified") == 2
    assert types[-1] == "plan_execution_verified"


@pytest.mark.asyncio
async def test_executor_stops_at_first_verification_failure():
    tasks = [
        OrchestratorTask(id="a", title="A", description="x",
                         acceptance_criteria="ok"),
        OrchestratorTask(id="b", title="B", description="y",
                         depends_on=["a"], acceptance_criteria="ok"),
    ]
    graph = _build_graph(tasks)

    async def execute_impl(g: TaskGraph) -> str:
        for t in g.tasks.values():
            t.status = COMPLETED
            t.result = {"output": "step done"}
        return "execution ok"

    orch = _StubOrchestrator(graph=graph, execute_impl=execute_impl)

    # First call (a) passes; second call (b) fails.
    class SequencedProvider:
        def __init__(self) -> None:
            self.idx = 0
            self.responses = [
                json.dumps({"passed": True, "notes": "a is fine"}),
                json.dumps({"passed": False, "notes": "b missed criteria"}),
            ]

        async def complete(self, **_kw: Any) -> LLMResponse:
            content = self.responses[self.idx]
            self.idx += 1
            return LLMResponse(content=content, finish_reason="stop")

    verifier = PlanVerifier(provider=SequencedProvider())
    events: list[dict[str, Any]] = []
    executor = PlanExecutor(orch, broadcast=events.append, verifier=verifier)
    result = await executor.run()

    assert result.ok is False
    assert result.verification_failed_step == "b"
    assert "missed criteria" in result.verification_notes
    assert result.verified_steps == ["a"]
    assert result.completed_steps == ["a", "b"]
    assert graph.tasks["a"].verification_status == "passed"
    assert graph.tasks["b"].verification_status == "failed"

    final = events[-1]
    assert final["type"] == "plan_execution_completed"
    assert final["verification_failed_step"] == "b"
    assert final["has_failures"] is True


@pytest.mark.asyncio
async def test_executor_without_verifier_passes_through():
    """No verifier → all completed steps marked verified, no LLM calls."""
    tasks = [
        OrchestratorTask(id="a", title="A", description="x",
                         acceptance_criteria="ok"),
    ]
    graph = _build_graph(tasks)

    async def execute_impl(g: TaskGraph) -> str:
        g.tasks["a"].status = COMPLETED
        g.tasks["a"].result = {"output": "ok"}
        return "done"

    orch = _StubOrchestrator(graph=graph, execute_impl=execute_impl)
    executor = PlanExecutor(orch, verifier=None)
    result = await executor.run()

    assert result.ok is True
    assert result.verified_steps == ["a"]
    assert graph.tasks["a"].verification_status == "passed"


@pytest.mark.asyncio
async def test_executor_skips_verification_on_execution_failure():
    """An execution-level failure must NOT trigger the verifier."""
    from captain_claw.task_graph import FAILED

    tasks = [
        OrchestratorTask(id="a", title="A", description="x",
                         acceptance_criteria="ok"),
    ]
    graph = _build_graph(tasks)

    async def execute_impl(g: TaskGraph) -> str:
        g.tasks["a"].status = FAILED
        g.tasks["a"].error = "boom"
        return ""

    orch = _StubOrchestrator(graph=graph, execute_impl=execute_impl)
    provider = _StubProvider("would crash if called")  # any non-json
    verifier = PlanVerifier(provider=provider)

    result = await PlanExecutor(orch, verifier=verifier).run()

    assert result.ok is False
    assert result.failed_step == "a"
    assert result.verification_failed_step is None
    # Verifier was never called because execution failed first.
    assert provider.calls == []
