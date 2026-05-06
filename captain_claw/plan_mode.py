"""Plan-mode planner: generates a reviewable, step-by-step plan from a user request.

Step 2 of plan-mode (read-only generation). The planner calls an LLM with the
``plan_mode_*`` instruction templates and returns a workflow-shaped dict ready to
be persisted via ``SessionOrchestrator.load_workflow`` and rendered to the user.

Execution, verification, and revision live in later modules.
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from typing import Any

from captain_claw.instructions import InstructionLoader
from captain_claw.llm import LLMProvider, Message
from captain_claw.logging import get_logger
from captain_claw.task_graph import OrchestratorTask

log = get_logger(__name__)

_PLAN_TIMEOUT_SECONDS = 120.0
_PLAN_MAX_TOKENS = 16000

_VALID_STEP_KINDS = {"atomic", "orchestrate", "verify", "revise"}


@dataclass
class Plan:
    """A generated plan, ready to persist as a workflow file."""

    summary: str
    user_input: str
    tasks: list[OrchestratorTask]

    def to_workflow_dict(self, name: str) -> dict[str, Any]:
        """Shape the plan as the workflow JSON ``load_workflow`` expects."""
        return {
            "workflow_name": name,
            "user_input": self.user_input,
            "synthesis_instruction": "",
            "tasks": [t.to_dict() for t in self.tasks],
        }

    def render_markdown(self) -> str:
        """Human-readable plan rendering for the chat UI."""
        lines: list[str] = []
        if self.summary:
            lines.append(f"**Plan summary:** {self.summary}")
            lines.append("")
        for i, task in enumerate(self.tasks, 1):
            kind_tag = "" if task.step_kind == "atomic" else f" *[{task.step_kind}]*"
            lines.append(f"**{i}. {task.title}**{kind_tag}  (`{task.id}`)")
            if task.depends_on:
                lines.append(f"   - depends on: {', '.join(task.depends_on)}")
            if task.description:
                lines.append(f"   - {task.description}")
            if task.acceptance_criteria:
                lines.append(f"   - ✓ {task.acceptance_criteria}")
            lines.append("")
        return "\n".join(lines).rstrip()


class PlanGenerator:
    """Calls the planner LLM and parses the response into a Plan.

    The planner uses the same provider as the main agent so users don't have
    to configure a separate planning model.
    """

    def __init__(
        self,
        provider: LLMProvider,
        instructions: InstructionLoader | None = None,
        *,
        timeout_seconds: float = _PLAN_TIMEOUT_SECONDS,
        max_tokens: int = _PLAN_MAX_TOKENS,
    ):
        self._provider = provider
        self._instructions = instructions or InstructionLoader()
        self._timeout = timeout_seconds
        self._max_tokens = max_tokens

    async def generate(
        self,
        user_input: str,
        *,
        workspace_tree: str = "",
    ) -> Plan | None:
        """Generate a plan for ``user_input``. Returns None on failure."""
        if not user_input or not user_input.strip():
            log.warning("PlanGenerator.generate called with empty user_input")
            return None

        system_prompt = self._instructions.load("plan_mode_system_prompt.md")
        user_prompt = self._instructions.render(
            "plan_mode_user_prompt.md",
            user_input=user_input,
            workspace_tree=workspace_tree or "",
        )
        if not system_prompt or not user_prompt:
            log.error("Plan-mode prompts missing or empty")
            return None

        messages = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=user_prompt),
        ]

        try:
            response = await asyncio.wait_for(
                self._provider.complete(
                    messages=messages, tools=None, max_tokens=self._max_tokens,
                ),
                timeout=self._timeout,
            )
        except asyncio.TimeoutError:
            log.error("Plan generation timed out", timeout=self._timeout)
            return None
        except Exception as e:
            log.error("Plan generation LLM call failed",
                      error=str(e), error_type=type(e).__name__)
            return None

        raw = str(getattr(response, "content", "") or "").strip()
        if not raw:
            log.error("Planner returned empty content")
            return None

        parsed = parse_json_response(raw)
        if parsed is None:
            log.error("Failed to parse planner JSON response", raw_preview=raw[:500])
            return None

        return self._build_plan(user_input, parsed)

    def _build_plan(self, user_input: str, parsed: dict[str, Any]) -> Plan | None:
        """Validate the parsed JSON and construct a Plan."""
        raw_tasks = parsed.get("tasks", [])
        if not isinstance(raw_tasks, list) or not raw_tasks:
            log.error("Plan JSON missing 'tasks' or empty",
                      keys=list(parsed.keys()))
            return None

        tasks: list[OrchestratorTask] = []
        seen_ids: set[str] = set()
        for i, raw in enumerate(raw_tasks):
            if not isinstance(raw, dict):
                log.warning("Skipping non-dict task entry", index=i)
                continue
            task = OrchestratorTask.from_dict(raw)
            if not task.id:
                log.warning("Skipping plan step with empty id", index=i)
                continue
            if task.id in seen_ids:
                log.warning("Duplicate plan step id, skipping", task_id=task.id)
                continue
            if task.step_kind not in _VALID_STEP_KINDS:
                log.warning("Unknown step_kind, coercing to atomic",
                            task_id=task.id, step_kind=task.step_kind)
                task.step_kind = "atomic"
            seen_ids.add(task.id)
            tasks.append(task)

        if not tasks:
            log.error("Plan had no valid tasks after filtering")
            return None

        # Drop dangling dependencies — keep only deps that point to known steps.
        for task in tasks:
            task.depends_on = [d for d in task.depends_on if d in seen_ids]

        summary = str(parsed.get("summary", "")).strip()
        return Plan(summary=summary, user_input=user_input, tasks=tasks)


def parse_json_response(raw: str) -> dict[str, Any] | None:
    """Parse JSON from an LLM response, tolerating code fences and prose wrap.

    Mirrors the behavior of ``SessionOrchestrator._parse_json_response`` so
    plan-mode doesn't take a dependency on the orchestrator's internals.
    """
    text = raw.strip()
    if not text:
        return None

    try:
        value = json.loads(text)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass

    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if fence_match:
        try:
            value = json.loads(fence_match.group(1).strip())
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            pass

    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        try:
            value = json.loads(brace_match.group(0))
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            pass

    return None
