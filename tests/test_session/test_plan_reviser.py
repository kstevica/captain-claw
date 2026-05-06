"""Tests for PlanReviser and PlanExecutor's revision loop (step 6)."""

from __future__ import annotations

import json
from typing import Any

import pytest

from captain_claw.llm import LLMResponse, Message
from captain_claw.plan_mode import (
    PlanExecutor,
    PlanReviser,
    PlanVerifier,
    RevisionProposal,
)
from captain_claw.task_graph import (
    COMPLETED,
    PENDING,
    OrchestratorTask,
    TaskGraph,
)


class _StubProvider:
    """LLM stub returning a list of canned responses in order."""

    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.calls: list[list[Message]] = []

    async def complete(
        self,
        messages: list[Message],
        tools: Any = None,
        temperature: Any = None,
        max_tokens: Any = None,
    ) -> LLMResponse:
        self.calls.append(messages)
        idx = min(len(self.calls) - 1, len(self._responses) - 1)
        return LLMResponse(content=self._responses[idx], finish_reason="stop")


class _StubOrchestrator:
    def __init__(self, graph: TaskGraph, execute_impl: Any):
        self._graph = graph
        self._execute_impl = execute_impl
        self.execute_calls = 0

    async def execute(self) -> str:
        self.execute_calls += 1
        return await self._execute_impl(self._graph, self.execute_calls)


def _build_graph(tasks: list[OrchestratorTask]) -> TaskGraph:
    g = TaskGraph(max_parallel=5)
    for t in tasks:
        g.add_task(t)
    return g


# ---------------------------------------------------------------------------
# PlanReviser.revise — direct unit tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reviser_returns_proposal_on_well_formed_response():
    provider = _StubProvider([json.dumps({
        "revised_description": "Read pdfs/*.pdf and write summaries.md with one section per file.",
        "revised_acceptance_criteria": "summaries.md exists and has N sections.",
        "rationale": "Original was vague — added explicit file paths and section count.",
    })])
    reviser = PlanReviser(provider=provider)
    task = OrchestratorTask(
        id="x", title="Summarize PDFs", description="Make summaries",
        acceptance_criteria="Summaries written.",
    )

    proposal = await reviser.revise(task, "I tried but didn't write anything.",
                                    "no file produced")

    assert isinstance(proposal, RevisionProposal)
    assert "summaries.md" in proposal.revised_description
    assert proposal.revised_acceptance_criteria.startswith("summaries.md exists")
    assert "explicit file paths" in proposal.rationale


@pytest.mark.asyncio
async def test_reviser_returns_none_on_empty_revised_description():
    provider = _StubProvider([json.dumps({"revised_description": ""})])
    proposal = await PlanReviser(provider=provider).revise(
        OrchestratorTask(id="x", title="t", description="d"),
        "out", "notes",
    )
    assert proposal is None


@pytest.mark.asyncio
async def test_reviser_returns_none_on_unparseable_response():
    provider = _StubProvider(["not even json"])
    proposal = await PlanReviser(provider=provider).revise(
        OrchestratorTask(id="x", title="t", description="d"),
        "out", "notes",
    )
    assert proposal is None


@pytest.mark.asyncio
async def test_reviser_returns_none_on_provider_exception():
    class BoomProvider:
        async def complete(self, **_kw: Any) -> LLMResponse:
            raise RuntimeError("provider down")

    proposal = await PlanReviser(provider=BoomProvider()).revise(
        OrchestratorTask(id="x", title="t", description="d"),
        "out", "notes",
    )
    assert proposal is None


# ---------------------------------------------------------------------------
# PlanExecutor revision loop integration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_executor_reruns_failed_step_after_revision_and_passes():
    """Verifier fails on attempt 1, reviser proposes new desc, attempt 2 passes."""
    a = OrchestratorTask(id="a", title="A", description="orig",
                         acceptance_criteria="ok")
    graph = _build_graph([a])

    async def execute_impl(g: TaskGraph, call_n: int) -> str:
        # On every call, mark the (re-)PENDING task COMPLETED with output
        # tagged by attempt number.
        for t in g.tasks.values():
            if t.status == PENDING:
                t.status = COMPLETED
                t.result = {"output": f"output-attempt-{call_n}"}
        return f"run-{call_n}"

    orch = _StubOrchestrator(graph=graph, execute_impl=execute_impl)

    # Verifier: fail on first call, pass on second.
    verifier_provider = _StubProvider([
        json.dumps({"passed": False, "notes": "not concrete enough"}),
        json.dumps({"passed": True, "notes": "now concrete"}),
    ])
    reviser_provider = _StubProvider([json.dumps({
        "revised_description": "Do X with file Y and produce Z.",
        "rationale": "added concrete details",
    })])

    events: list[dict[str, Any]] = []
    executor = PlanExecutor(
        orch,
        broadcast=events.append,
        verifier=PlanVerifier(provider=verifier_provider),
        reviser=PlanReviser(provider=reviser_provider),
        max_revisions=2,
    )

    result = await executor.run()

    assert result.ok is True
    assert result.verified_steps == ["a"]
    assert len(result.revisions) == 1
    rev = result.revisions[0]
    assert rev["task_id"] == "a"
    assert rev["revision_count"] == 1
    assert "concrete details" in rev["rationale"]

    # Task carries the revision metadata.
    assert graph.tasks["a"].revision_count == 1
    assert graph.tasks["a"].original_description == "orig"
    assert "Do X with file Y" in graph.tasks["a"].description
    assert graph.tasks["a"].verification_status == "passed"

    # Two execute calls: initial + after revision.
    assert orch.execute_calls == 2

    types = [e["type"] for e in events]
    assert "plan_step_revised" in types
    assert types[-1] == "plan_execution_verified"


@pytest.mark.asyncio
async def test_executor_gives_up_after_max_revisions_exhausted():
    a = OrchestratorTask(id="a", title="A", description="orig",
                         acceptance_criteria="ok")
    graph = _build_graph([a])

    async def execute_impl(g: TaskGraph, call_n: int) -> str:
        for t in g.tasks.values():
            if t.status == PENDING:
                t.status = COMPLETED
                t.result = {"output": f"out-{call_n}"}
        return ""

    orch = _StubOrchestrator(graph=graph, execute_impl=execute_impl)

    # Always fail verification.
    verifier_provider = _StubProvider([
        json.dumps({"passed": False, "notes": "still not right"}),
    ])
    # Reviser proposes a revision every time.
    reviser_provider = _StubProvider([json.dumps({
        "revised_description": "Try harder",
        "rationale": "tighten",
    })])

    executor = PlanExecutor(
        orch,
        verifier=PlanVerifier(provider=verifier_provider),
        reviser=PlanReviser(provider=reviser_provider),
        max_revisions=2,
    )
    result = await executor.run()

    assert result.ok is False
    assert result.verification_failed_step == "a"
    # 2 revisions attempted (the budget), then give up.
    assert len(result.revisions) == 2
    assert graph.tasks["a"].revision_count == 2
    # 1 initial run + 2 retries = 3 execute() calls.
    assert orch.execute_calls == 3
    assert "after 2 revision(s)" in result.error


@pytest.mark.asyncio
async def test_executor_aborts_when_reviser_returns_none():
    a = OrchestratorTask(id="a", title="A", description="orig",
                         acceptance_criteria="ok")
    graph = _build_graph([a])

    async def execute_impl(g: TaskGraph, _n: int) -> str:
        for t in g.tasks.values():
            if t.status == PENDING:
                t.status = COMPLETED
                t.result = {"output": "out"}
        return ""

    orch = _StubOrchestrator(graph=graph, execute_impl=execute_impl)

    verifier_provider = _StubProvider([
        json.dumps({"passed": False, "notes": "nope"}),
    ])
    # Reviser returns unparseable → revise() returns None.
    reviser_provider = _StubProvider(["not json"])

    events: list[dict[str, Any]] = []
    executor = PlanExecutor(
        orch,
        broadcast=events.append,
        verifier=PlanVerifier(provider=verifier_provider),
        reviser=PlanReviser(provider=reviser_provider),
        max_revisions=2,
    )
    result = await executor.run()

    assert result.ok is False
    assert result.verification_failed_step == "a"
    assert result.revisions == []  # nothing applied
    assert "could not produce a revision" in result.error
    final = events[-1]
    assert final.get("revision_aborted") is True


@pytest.mark.asyncio
async def test_executor_does_nothing_extra_when_no_reviser_configured():
    a = OrchestratorTask(id="a", title="A", description="orig",
                         acceptance_criteria="ok")
    graph = _build_graph([a])

    async def execute_impl(g: TaskGraph, _n: int) -> str:
        for t in g.tasks.values():
            if t.status == PENDING:
                t.status = COMPLETED
                t.result = {"output": "out"}
        return ""

    verifier_provider = _StubProvider([
        json.dumps({"passed": False, "notes": "nope"}),
    ])
    orch = _StubOrchestrator(graph=graph, execute_impl=execute_impl)

    executor = PlanExecutor(
        orch,
        verifier=PlanVerifier(provider=verifier_provider),
        reviser=None,
        max_revisions=2,
    )
    result = await executor.run()

    assert result.ok is False
    assert result.verification_failed_step == "a"
    assert result.revisions == []
    assert orch.execute_calls == 1  # no rerun without a reviser


@pytest.mark.asyncio
async def test_revision_resets_failed_step_and_downstream():
    """Revising step a must reset b too (transitive downstream)."""
    a = OrchestratorTask(id="a", title="A", description="d",
                         acceptance_criteria="ok")
    b = OrchestratorTask(id="b", title="B", description="d",
                         depends_on=["a"], acceptance_criteria="ok")
    graph = _build_graph([a, b])

    state = {"a_attempts": 0}

    async def execute_impl(g: TaskGraph, _n: int) -> str:
        for t in g.tasks.values():
            if t.status == PENDING:
                t.status = COMPLETED
                t.result = {"output": f"out-{t.id}"}
                if t.id == "a":
                    state["a_attempts"] += 1
        return ""

    orch = _StubOrchestrator(graph=graph, execute_impl=execute_impl)

    # Pass A on attempt 1, fail B; after revision of B, pass both.
    # Verifier sees a, b in order: pass, fail, then on rerun: pass (b only,
    # because a already passed and is not re-verified).
    verifier_provider = _StubProvider([
        json.dumps({"passed": True, "notes": "a ok"}),
        json.dumps({"passed": False, "notes": "b broken"}),
        json.dumps({"passed": True, "notes": "b ok now"}),
    ])
    reviser_provider = _StubProvider([json.dumps({
        "revised_description": "Try b harder",
        "rationale": "tighten",
    })])

    executor = PlanExecutor(
        orch,
        verifier=PlanVerifier(provider=verifier_provider),
        reviser=PlanReviser(provider=reviser_provider),
        max_revisions=2,
    )
    result = await executor.run()

    assert result.ok is True
    assert state["a_attempts"] == 1  # a was NOT re-run
    # B was re-run after revision.
    assert graph.tasks["b"].revision_count == 1
    assert graph.tasks["a"].revision_count == 0


@pytest.mark.asyncio
async def test_runtime_failure_is_not_revised():
    """Worker errors / FAILED status skip the revision loop."""
    from captain_claw.task_graph import FAILED

    a = OrchestratorTask(id="a", title="A", description="d",
                         acceptance_criteria="ok")
    graph = _build_graph([a])

    async def execute_impl(g: TaskGraph, _n: int) -> str:
        g.tasks["a"].status = FAILED
        g.tasks["a"].error = "boom"
        return ""

    orch = _StubOrchestrator(graph=graph, execute_impl=execute_impl)

    reviser_provider = _StubProvider([json.dumps({
        "revised_description": "should not be called",
    })])
    reviser = PlanReviser(provider=reviser_provider)

    executor = PlanExecutor(orch, reviser=reviser, max_revisions=2)
    result = await executor.run()

    assert result.ok is False
    assert result.failed_step == "a"
    assert result.revisions == []
    assert reviser_provider.calls == []
    assert orch.execute_calls == 1
