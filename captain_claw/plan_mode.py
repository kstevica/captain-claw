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
from dataclasses import dataclass, field
from typing import Any, Callable

from captain_claw.instructions import InstructionLoader
from captain_claw.llm import LLMProvider, Message
from captain_claw.logging import get_logger
from captain_claw.output_validation import validate_task_output
from captain_claw.task_graph import COMPLETED, FAILED, OrchestratorTask, TaskGraph

log = get_logger(__name__)

_PLAN_TIMEOUT_SECONDS = 120.0
_PLAN_MAX_TOKENS = 16000
_VERIFY_TIMEOUT_SECONDS = 60.0
_VERIFY_MAX_TOKENS = 1000
_VERIFY_OUTPUT_TRUNCATE = 8000  # chars of step output sent to the verifier

_VALID_STEP_KINDS = {"atomic", "orchestrate", "verify", "revise"}

_PASSED = "passed"
_FAILED = "failed"


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


@dataclass
class PlanExecutionResult:
    """Outcome of running a plan."""

    ok: bool
    final_output: str
    completed_steps: list[str]
    failed_step: str | None = None
    error: str = ""
    verified_steps: list[str] = field(default_factory=list)
    verification_failed_step: str | None = None
    verification_notes: str = ""


@dataclass
class VerificationOutcome:
    """Result of verifying a single step's output."""

    passed: bool
    notes: str
    schema_error: str = ""  # populated when output_schema validation fails


class PlanVerifier:
    """Verify a step's output against its acceptance criteria + optional schema.

    Step 4 of plan-mode. Two-stage gate:
      1. If ``task.output_schema`` is set, run schema validation first
         (fast, deterministic). On failure, the step fails immediately.
      2. Otherwise (or after schema passes), call an LLM judge against
         ``task.acceptance_criteria``. The judge returns ``{passed, notes}``.

    Steps with no ``acceptance_criteria`` and no ``output_schema`` auto-pass
    with a note explaining why — the planner is encouraged to set criteria
    but a missing one shouldn't stall execution.
    """

    def __init__(
        self,
        provider: LLMProvider,
        instructions: InstructionLoader | None = None,
        *,
        timeout_seconds: float = _VERIFY_TIMEOUT_SECONDS,
        max_tokens: int = _VERIFY_MAX_TOKENS,
    ):
        self._provider = provider
        self._instructions = instructions or InstructionLoader()
        self._timeout = timeout_seconds
        self._max_tokens = max_tokens

    async def verify(
        self,
        task: OrchestratorTask,
        output_text: str,
    ) -> VerificationOutcome:
        """Verify ``task``'s ``output_text``. Never raises — failures return passed=False."""
        if task.output_schema:
            valid, error, _parsed = validate_task_output(output_text, task.output_schema)
            if not valid:
                return VerificationOutcome(
                    passed=False,
                    notes=f"Schema validation failed: {error}",
                    schema_error=error or "schema validation failed",
                )

        criteria = (task.acceptance_criteria or "").strip()
        if not criteria:
            return VerificationOutcome(
                passed=True,
                notes="No acceptance criteria set — auto-passed.",
            )

        if not (output_text or "").strip():
            return VerificationOutcome(
                passed=False,
                notes="Step produced no output to verify against the acceptance criteria.",
            )

        return await self._llm_judge(task, output_text, criteria)

    async def _llm_judge(
        self,
        task: OrchestratorTask,
        output_text: str,
        criteria: str,
    ) -> VerificationOutcome:
        system_prompt = self._instructions.load("plan_mode_verifier_system_prompt.md")
        user_prompt = self._instructions.render(
            "plan_mode_verifier_user_prompt.md",
            title=task.title,
            description=task.description,
            acceptance_criteria=criteria,
            output=output_text[:_VERIFY_OUTPUT_TRUNCATE],
        )
        if not system_prompt or not user_prompt:
            log.error("Verifier prompts missing — auto-passing")
            return VerificationOutcome(
                passed=True,
                notes="Verifier prompts unavailable — defaulted to pass.",
            )

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
            log.error("Verifier call timed out", task_id=task.id, timeout=self._timeout)
            return VerificationOutcome(
                passed=False,
                notes=f"Verifier timed out after {self._timeout:.0f}s.",
            )
        except Exception as e:
            log.error("Verifier LLM call failed",
                      task_id=task.id,
                      error=str(e),
                      error_type=type(e).__name__)
            return VerificationOutcome(
                passed=False,
                notes=f"Verifier failed: {e}",
            )

        raw = str(getattr(response, "content", "") or "").strip()
        parsed = parse_json_response(raw)
        if parsed is None:
            log.warning("Verifier returned unparseable JSON",
                        task_id=task.id, raw_preview=raw[:300])
            return VerificationOutcome(
                passed=False,
                notes="Verifier response could not be parsed as JSON.",
            )

        passed = bool(parsed.get("passed", False))
        notes = str(parsed.get("notes", "")).strip() or (
            "Verifier returned no notes."
        )
        return VerificationOutcome(passed=passed, notes=notes)


class PlanExecutor:
    """Run a plan whose graph is already loaded into an orchestrator.

    Step 3 of plan-mode: sequential execution of ``atomic`` steps. The DAG
    runner inside ``SessionOrchestrator`` already serializes when each step
    declares the previous one as a dependency (which is the planner's default
    output), so we lean on it instead of re-implementing worker plumbing.

    Step 4 adds post-execution verification: after the graph runs, completed
    steps are walked in topological order and each is verified against its
    ``acceptance_criteria`` (and optional ``output_schema``). The first
    verification failure stops the walk and returns a failed result so the
    revision loop in step 6 can pick it up.

    ``orchestrate``-kind steps are coerced to ``atomic`` with a warning. Step 5
    will replace this coercion with real fan-out via ``SessionOrchestrator``.
    """

    def __init__(
        self,
        orchestrator: Any,
        *,
        broadcast: Callable[[dict[str, Any]], None] | None = None,
        verifier: PlanVerifier | None = None,
    ):
        self._orchestrator = orchestrator
        self._broadcast = broadcast
        self._verifier = verifier

    async def run(self) -> PlanExecutionResult:
        graph: TaskGraph | None = getattr(self._orchestrator, "_graph", None)
        if graph is None or graph.task_count == 0:
            return PlanExecutionResult(
                ok=False,
                final_output="",
                completed_steps=[],
                error="No plan loaded. Run /plan first.",
            )

        coerced = self._coerce_orchestrate_to_atomic(graph)
        if coerced:
            log.warning(
                "PlanExecutor coerced orchestrate steps to atomic "
                "(step 5 will add real fan-out)",
                step_ids=coerced,
            )

        self._emit("plan_execution_started", {
            "step_count": graph.task_count,
            "steps": [
                {
                    "id": t.id,
                    "title": t.title,
                    "step_kind": t.step_kind,
                    "acceptance_criteria": t.acceptance_criteria,
                    "depends_on": t.depends_on,
                }
                for t in graph.tasks.values()
            ],
        })

        try:
            output = await self._orchestrator.execute()
        except Exception as e:
            log.error("Plan execution raised", error=str(e),
                      error_type=type(e).__name__)
            self._emit("plan_execution_failed", {"error": str(e)})
            return PlanExecutionResult(
                ok=False, final_output="", completed_steps=[], error=str(e),
            )

        completed = [tid for tid, t in graph.tasks.items() if t.status == COMPLETED]
        failed = next(
            (tid for tid, t in graph.tasks.items() if t.status == FAILED), None,
        )

        if failed:
            self._emit("plan_execution_completed", {
                "completed": completed,
                "failed_step": failed,
                "has_failures": True,
            })
            failed_task = graph.tasks[failed]
            return PlanExecutionResult(
                ok=False,
                final_output=output or "",
                completed_steps=completed,
                failed_step=failed,
                error=failed_task.error or "step failed",
            )

        verified, verify_failed, verify_notes = await self._verify_completed(
            graph, completed,
        )

        if verify_failed is not None:
            self._emit("plan_execution_completed", {
                "completed": completed,
                "verified": verified,
                "verification_failed_step": verify_failed,
                "verification_notes": verify_notes,
                "has_failures": True,
            })
            return PlanExecutionResult(
                ok=False,
                final_output=output or "",
                completed_steps=completed,
                verified_steps=verified,
                verification_failed_step=verify_failed,
                verification_notes=verify_notes,
                error=f"verification failed at step '{verify_failed}': {verify_notes}",
            )

        self._emit("plan_execution_verified", {
            "completed": completed,
            "verified": verified,
        })

        return PlanExecutionResult(
            ok=True,
            final_output=output or "",
            completed_steps=completed,
            verified_steps=verified,
        )

    async def _verify_completed(
        self,
        graph: TaskGraph,
        completed: list[str],
    ) -> tuple[list[str], str | None, str]:
        """Walk completed steps in order; verify each. Stop at first failure.

        Returns ``(verified_ids, failed_id_or_None, notes)``. When the verifier
        is not configured, all completed steps are marked verified=passed and
        the gate is a no-op.
        """
        verified: list[str] = []
        for tid in completed:
            task = graph.tasks.get(tid)
            if task is None:
                continue

            if self._verifier is None:
                task.verification_status = _PASSED
                task.verification_notes = "No verifier configured."
                verified.append(tid)
                continue

            output_text = self._extract_output(task)
            outcome = await self._verifier.verify(task, output_text)
            task.verification_status = _PASSED if outcome.passed else _FAILED
            task.verification_notes = outcome.notes

            self._emit("plan_step_verified", {
                "task_id": tid,
                "passed": outcome.passed,
                "notes": outcome.notes,
            })

            if not outcome.passed:
                return verified, tid, outcome.notes
            verified.append(tid)

        return verified, None, ""

    @staticmethod
    def _extract_output(task: OrchestratorTask) -> str:
        """Pull the executor's text output off ``task.result`` for verification."""
        result = task.result
        if isinstance(result, dict):
            return str(result.get("output", "") or "")
        return ""

    @staticmethod
    def _coerce_orchestrate_to_atomic(graph: TaskGraph) -> list[str]:
        """Coerce orchestrate-kind steps to atomic for v3 (no fan-out yet).

        Returns the IDs that were coerced so callers can warn the user.
        """
        coerced: list[str] = []
        for tid, task in graph.tasks.items():
            if task.step_kind == "orchestrate":
                task.step_kind = "atomic"
                coerced.append(tid)
        return coerced

    def _emit(self, event_name: str, data: dict[str, Any]) -> None:
        if self._broadcast is None:
            return
        try:
            self._broadcast({"type": event_name, **data})
        except Exception as e:
            log.debug("Plan-mode event broadcast failed",
                      event_name=event_name, error=str(e))


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
