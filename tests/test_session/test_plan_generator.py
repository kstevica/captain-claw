"""Tests for PlanGenerator (step 2 of plan-mode)."""

from __future__ import annotations

import json
from typing import Any

import pytest

from captain_claw.llm import LLMResponse, Message
from captain_claw.plan_mode import Plan, PlanGenerator, parse_json_response


class _StubProvider:
    """Minimal LLMProvider stub: returns a canned response for ``complete``."""

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
        self.calls.append(
            {"messages": messages, "tools": tools, "max_tokens": max_tokens}
        )
        return LLMResponse(content=self._content, finish_reason="stop")


@pytest.mark.asyncio
async def test_generate_returns_plan_with_acceptance_criteria():
    canned = json.dumps({
        "summary": "Summarize three PDFs into a Markdown digest.",
        "tasks": [
            {
                "id": "find_pdfs",
                "title": "Find PDFs",
                "description": "Use glob to list pdfs/*.pdf",
                "depends_on": [],
                "step_kind": "atomic",
                "acceptance_criteria": "At least one PDF path is returned.",
            },
            {
                "id": "summarize_each",
                "title": "Summarize each PDF",
                "description": "For each PDF, extract text and append a section.",
                "depends_on": ["find_pdfs"],
                "step_kind": "orchestrate",
                "acceptance_criteria": "summaries.md contains a section per PDF.",
            },
        ],
    })
    provider = _StubProvider(canned)
    gen = PlanGenerator(provider=provider)

    plan = await gen.generate("summarize the pdfs in pdfs/")

    assert plan is not None
    assert plan.summary.startswith("Summarize three")
    assert len(plan.tasks) == 2
    assert plan.tasks[0].id == "find_pdfs"
    assert plan.tasks[0].step_kind == "atomic"
    assert plan.tasks[0].acceptance_criteria
    assert plan.tasks[1].step_kind == "orchestrate"
    assert plan.tasks[1].depends_on == ["find_pdfs"]


@pytest.mark.asyncio
async def test_generate_handles_code_fenced_response():
    canned = (
        "Here is the plan:\n```json\n"
        + json.dumps({
            "summary": "ok",
            "tasks": [{
                "id": "step",
                "title": "Do it",
                "description": "x",
                "depends_on": [],
                "acceptance_criteria": "done",
            }],
        })
        + "\n```\nLet me know!"
    )
    plan = await PlanGenerator(provider=_StubProvider(canned)).generate("x")
    assert plan is not None
    assert plan.tasks[0].id == "step"


@pytest.mark.asyncio
async def test_generate_drops_dangling_dependencies():
    canned = json.dumps({
        "summary": "s",
        "tasks": [
            {"id": "a", "title": "A", "description": "x",
             "depends_on": [], "acceptance_criteria": "ok"},
            {"id": "b", "title": "B", "description": "y",
             "depends_on": ["a", "ghost"], "acceptance_criteria": "ok"},
        ],
    })
    plan = await PlanGenerator(provider=_StubProvider(canned)).generate("x")
    assert plan is not None
    assert plan.tasks[1].depends_on == ["a"]


@pytest.mark.asyncio
async def test_generate_coerces_unknown_step_kind():
    canned = json.dumps({
        "summary": "s",
        "tasks": [{
            "id": "a", "title": "A", "description": "x",
            "depends_on": [], "step_kind": "wibble",
            "acceptance_criteria": "ok",
        }],
    })
    plan = await PlanGenerator(provider=_StubProvider(canned)).generate("x")
    assert plan is not None
    assert plan.tasks[0].step_kind == "atomic"


@pytest.mark.asyncio
async def test_generate_returns_none_on_empty_user_input():
    plan = await PlanGenerator(provider=_StubProvider("")).generate("")
    assert plan is None


@pytest.mark.asyncio
async def test_generate_returns_none_on_unparseable_response():
    plan = await PlanGenerator(provider=_StubProvider("not json at all")).generate("x")
    assert plan is None


@pytest.mark.asyncio
async def test_generate_returns_none_when_tasks_missing():
    canned = json.dumps({"summary": "s", "tasks": []})
    plan = await PlanGenerator(provider=_StubProvider(canned)).generate("x")
    assert plan is None


def test_plan_to_workflow_dict_round_trips_through_orchestrator_loader():
    """A generated plan must be loadable as a workflow file."""
    from captain_claw.task_graph import OrchestratorTask

    plan = Plan(
        summary="s",
        user_input="do the thing",
        tasks=[
            OrchestratorTask(
                id="step1", title="Step one", description="d",
                step_kind="orchestrate",
                acceptance_criteria="output.md exists",
            ),
        ],
    )
    payload = plan.to_workflow_dict("plan-do-the-thing")

    assert payload["workflow_name"] == "plan-do-the-thing"
    assert payload["user_input"] == "do the thing"
    assert payload["tasks"][0]["id"] == "step1"
    assert payload["tasks"][0]["step_kind"] == "orchestrate"
    assert payload["tasks"][0]["acceptance_criteria"] == "output.md exists"

    # Re-serialize → deserialize via the same helpers used by load_workflow.
    restored = OrchestratorTask.from_dict(payload["tasks"][0])
    assert restored.step_kind == "orchestrate"
    assert restored.acceptance_criteria == "output.md exists"


def test_plan_render_markdown_includes_step_metadata():
    from captain_claw.task_graph import OrchestratorTask

    plan = Plan(
        summary="Build the widget",
        user_input="build me a widget",
        tasks=[
            OrchestratorTask(
                id="design", title="Design widget", description="Sketch the API",
                acceptance_criteria="api.md exists",
            ),
            OrchestratorTask(
                id="build", title="Build widget", description="Write code",
                depends_on=["design"], step_kind="orchestrate",
                acceptance_criteria="tests pass",
            ),
        ],
    )
    md = plan.render_markdown()
    assert "Build the widget" in md
    assert "Design widget" in md
    assert "depends on: design" in md
    assert "[orchestrate]" in md
    assert "tests pass" in md


def test_parse_json_response_handles_prose_wrapping():
    raw = "Sure, here's the plan: {\"summary\": \"s\", \"tasks\": []}\nThanks!"
    parsed = parse_json_response(raw)
    assert parsed == {"summary": "s", "tasks": []}
