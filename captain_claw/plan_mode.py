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
from typing import Any, Awaitable, Callable

from captain_claw.instructions import InstructionLoader
from captain_claw.llm import LLMProvider, Message
from captain_claw.logging import get_logger
from captain_claw.output_validation import validate_task_output
from captain_claw.task_graph import (
    COMPLETED,
    FAILED,
    PENDING,
    OrchestratorTask,
    TaskGraph,
)

log = get_logger(__name__)

_PLAN_TIMEOUT_SECONDS = 120.0
_PLAN_MAX_TOKENS = 16000
_VERIFY_TIMEOUT_SECONDS = 60.0
_VERIFY_MAX_TOKENS = 1000
_VERIFY_OUTPUT_TRUNCATE = 8000  # chars of step output sent to the verifier
_REVISE_TIMEOUT_SECONDS = 120.0
_REVISE_MAX_TOKENS = 4000
_REVISE_OUTPUT_TRUNCATE = 8000

_VALID_STEP_KINDS = {"atomic", "orchestrate", "verify", "revise"}

_PASSED = "passed"
_FAILED = "failed"

DEFAULT_MAX_REVISIONS = 2

# Plan-mode enrichment levels — cumulative.
#
#   plain      → today's behavior: planner sees only the user request +
#                a workspace tree scan.
#   enriched   → + the latest reflection block, so the plan honors the
#                "what I learned about this user" personality layer.
#   insightful → enriched + top-N matching insights (curated facts), so
#                the plan can lean on known preferences/decisions/deadlines.
#   complete   → insightful + the agent's persona block (cognitive-mode-
#                aware), so the plan also reflects the agent's role bias
#                (research / coding / writing / etc).
#
# Levels are strings rather than an Enum so the same value flows through
# slash commands, websocket events, JSON storage and the FD UI unchanged.
PLAN_LEVELS: tuple[str, ...] = ("plain", "enriched", "insightful", "complete")
DEFAULT_PLAN_LEVEL = "plain"


def normalize_plan_level(value: str | None) -> str:
    """Coerce *value* to a known plan level, falling back to ``DEFAULT_PLAN_LEVEL``."""
    if not value:
        return DEFAULT_PLAN_LEVEL
    v = str(value).strip().lower()
    return v if v in PLAN_LEVELS else DEFAULT_PLAN_LEVEL


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
        reflection_block: str = "",
        insights_block: str = "",
        personality_block: str = "",
        system_prompt_name: str = "plan_mode_system_prompt.md",
    ) -> Plan | None:
        """Generate a plan for ``user_input``. Returns None on failure.

        The four optional ``*_block`` / ``system_prompt_name`` parameters are
        the plan-level enrichment knobs — see :data:`PLAN_LEVELS`. The caller
        (``handle_plan_command`` in ``web/plan_commands.py``) is responsible
        for rendering each block from its source layer (reflections store,
        insights DB, personality.py) and choosing the appropriate planner
        template name. Empty strings collapse cleanly via ``_SafeFormatDict``.
        """
        if not user_input or not user_input.strip():
            log.warning("PlanGenerator.generate called with empty user_input")
            return None

        system_prompt = self._instructions.load(system_prompt_name)
        user_prompt = self._instructions.render(
            "plan_mode_user_prompt.md",
            user_input=user_input,
            workspace_tree=workspace_tree or "",
            reflection_block=reflection_block or "",
            insights_block=insights_block or "",
            personality_block=personality_block or "",
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
    revisions: list[dict[str, Any]] = field(default_factory=list)


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


@dataclass
class RevisionProposal:
    """The reviser's proposed update for a failed step."""

    revised_description: str
    revised_acceptance_criteria: str = ""
    rationale: str = ""


class PlanReviser:
    """Generate a revised description for a step that failed verification.

    Step 6 of plan-mode. When the verifier reports a step failed against its
    ``acceptance_criteria``, the reviser is asked for a sharper description
    so the next executor attempt has a better chance of passing. Returns
    ``None`` on hard failures (timeout, unparseable response) so the caller
    can give up cleanly.
    """

    def __init__(
        self,
        provider: LLMProvider,
        instructions: InstructionLoader | None = None,
        *,
        timeout_seconds: float = _REVISE_TIMEOUT_SECONDS,
        max_tokens: int = _REVISE_MAX_TOKENS,
    ):
        self._provider = provider
        self._instructions = instructions or InstructionLoader()
        self._timeout = timeout_seconds
        self._max_tokens = max_tokens

    async def revise(
        self,
        task: OrchestratorTask,
        output_text: str,
        verifier_notes: str,
    ) -> RevisionProposal | None:
        system_prompt = self._instructions.load("plan_mode_reviser_system_prompt.md")
        user_prompt = self._instructions.render(
            "plan_mode_reviser_user_prompt.md",
            title=task.title,
            description=task.description,
            acceptance_criteria=task.acceptance_criteria or "(none)",
            revision_count=task.revision_count + 1,
            output=(output_text or "")[:_REVISE_OUTPUT_TRUNCATE],
            verifier_notes=verifier_notes or "(no notes provided)",
        )
        if not system_prompt or not user_prompt:
            log.error("Reviser prompts missing")
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
            log.error("Reviser call timed out", task_id=task.id, timeout=self._timeout)
            return None
        except Exception as e:
            log.error("Reviser LLM call failed",
                      task_id=task.id,
                      error=str(e),
                      error_type=type(e).__name__)
            return None

        raw = str(getattr(response, "content", "") or "").strip()
        parsed = parse_json_response(raw)
        if parsed is None:
            log.warning("Reviser returned unparseable JSON",
                        task_id=task.id, raw_preview=raw[:300])
            return None

        revised_desc = str(parsed.get("revised_description", "")).strip()
        if not revised_desc:
            log.warning("Reviser returned empty revised_description",
                        task_id=task.id)
            return None

        return RevisionProposal(
            revised_description=revised_desc,
            revised_acceptance_criteria=str(
                parsed.get("revised_acceptance_criteria", ""),
            ).strip(),
            rationale=str(parsed.get("rationale", "")).strip(),
        )


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
        expander: "Callable[[OrchestratorTask], Awaitable[list[OrchestratorTask]]] | None" = None,
        reviser: PlanReviser | None = None,
        max_revisions: int = DEFAULT_MAX_REVISIONS,
    ):
        self._orchestrator = orchestrator
        self._broadcast = broadcast
        self._verifier = verifier
        # If no expander is provided, orchestrate steps fall back to atomic
        # execution (legacy step-3 behavior). Tests inject a stub; production
        # wires ``orchestrate_expander_from_orchestrator`` against the
        # SessionOrchestrator's decomposer.
        self._expander = expander
        self._reviser = reviser
        self._max_revisions = max(0, int(max_revisions))

    async def run(self) -> PlanExecutionResult:
        graph: TaskGraph | None = getattr(self._orchestrator, "_graph", None)
        if graph is None or graph.task_count == 0:
            return PlanExecutionResult(
                ok=False,
                final_output="",
                completed_steps=[],
                error="No plan loaded. Run /plan first.",
            )

        if self._expander is not None:
            try:
                expanded = await self._expand_orchestrate_steps(graph)
            except Exception as e:
                log.error("Orchestrate expansion failed",
                          error=str(e), error_type=type(e).__name__)
                self._emit("plan_execution_failed", {
                    "error": f"orchestrate expansion failed: {e}",
                })
                return PlanExecutionResult(
                    ok=False, final_output="", completed_steps=[],
                    error=f"orchestrate expansion failed: {e}",
                )
            if expanded:
                self._emit("plan_orchestrate_expanded", {"expansions": expanded})
                graph.refresh()
        else:
            coerced = self._coerce_orchestrate_to_atomic(graph)
            if coerced:
                log.warning(
                    "PlanExecutor coerced orchestrate steps to atomic "
                    "(no expander configured)",
                    step_ids=coerced,
                )

        self._emit("plan_execution_started", {
            "step_count": graph.task_count,
            "max_revisions": self._max_revisions,
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

        revisions: list[dict[str, Any]] = []
        last_output = ""
        # The cycle budget is `max_revisions + 1`: one initial run plus up to
        # ``max_revisions`` retry cycles after revision proposals.
        for _cycle in range(self._max_revisions + 1):
            try:
                # skip_synthesize: plan-mode owns its own verification +
                # per-step result rendering, so the orchestrator's final
                # synthesis pass is redundant work that makes the chat sit
                # at "Orchestrator: synthesizing results..." after the plan
                # card already says VERIFIED.
                output = await self._orchestrator.execute(skip_synthesize=True)
            except Exception as e:
                log.error("Plan execution raised", error=str(e),
                          error_type=type(e).__name__)
                self._emit("plan_execution_failed", {"error": str(e)})
                return PlanExecutionResult(
                    ok=False,
                    final_output="",
                    completed_steps=[],
                    error=str(e),
                    revisions=revisions,
                )

            last_output = output or ""
            completed = [
                tid for tid, t in graph.tasks.items() if t.status == COMPLETED
            ]
            failed = next(
                (tid for tid, t in graph.tasks.items() if t.status == FAILED),
                None,
            )

            if failed:
                # Run-time failures (worker error / timeout) skip the revision
                # loop — those need orchestrator-level recovery, not a
                # description rewrite.
                self._emit("plan_execution_completed", {
                    "completed": completed,
                    "failed_step": failed,
                    "has_failures": True,
                })
                failed_task = graph.tasks[failed]
                return PlanExecutionResult(
                    ok=False,
                    final_output=last_output,
                    completed_steps=completed,
                    failed_step=failed,
                    error=failed_task.error or "step failed",
                    revisions=revisions,
                )

            verified, verify_failed, verify_notes = await self._verify_completed(
                graph, completed,
            )

            if verify_failed is None:
                self._emit("plan_execution_verified", {
                    "completed": completed,
                    "verified": verified,
                    "revision_count": len(revisions),
                })
                return PlanExecutionResult(
                    ok=True,
                    final_output=last_output,
                    completed_steps=completed,
                    verified_steps=verified,
                    revisions=revisions,
                )

            # Verification failed — attempt revision if budget remains.
            failed_task = graph.tasks[verify_failed]
            should_revise = (
                self._reviser is not None
                and failed_task.revision_count < self._max_revisions
            )

            if not should_revise:
                self._emit("plan_execution_completed", {
                    "completed": completed,
                    "verified": verified,
                    "verification_failed_step": verify_failed,
                    "verification_notes": verify_notes,
                    "has_failures": True,
                    "revision_count": failed_task.revision_count,
                })
                return PlanExecutionResult(
                    ok=False,
                    final_output=last_output,
                    completed_steps=completed,
                    verified_steps=verified,
                    verification_failed_step=verify_failed,
                    verification_notes=verify_notes,
                    error=(
                        f"verification failed at step '{verify_failed}'"
                        + (
                            f" after {failed_task.revision_count} revision(s)"
                            if failed_task.revision_count
                            else ""
                        )
                        + f": {verify_notes}"
                    ),
                    revisions=revisions,
                )

            proposal = await self._reviser.revise(
                failed_task,
                self._extract_output(failed_task),
                verify_notes,
            )
            if proposal is None:
                self._emit("plan_execution_completed", {
                    "completed": completed,
                    "verified": verified,
                    "verification_failed_step": verify_failed,
                    "verification_notes": verify_notes,
                    "has_failures": True,
                    "revision_count": failed_task.revision_count,
                    "revision_aborted": True,
                })
                return PlanExecutionResult(
                    ok=False,
                    final_output=last_output,
                    completed_steps=completed,
                    verified_steps=verified,
                    verification_failed_step=verify_failed,
                    verification_notes=verify_notes,
                    error=(
                        f"verification failed at step '{verify_failed}' and "
                        f"reviser could not produce a revision: {verify_notes}"
                    ),
                    revisions=revisions,
                )

            self._apply_revision(graph, failed_task, proposal)
            revisions.append({
                "task_id": verify_failed,
                "revision_count": failed_task.revision_count,
                "rationale": proposal.rationale,
                "revised_description": proposal.revised_description,
                "previous_verification_notes": verify_notes,
            })
            self._emit("plan_step_revised", revisions[-1])
            # Loop continues — re-execute the graph; only PENDING tasks
            # (the revised step + downstream) will run.

        # Defensive: budget exhausted without resolution. Surface the last
        # known verification failure if available.
        log.error("Plan revision loop exhausted without resolution",
                  max_revisions=self._max_revisions)
        return PlanExecutionResult(
            ok=False,
            final_output=last_output,
            completed_steps=[],
            error="revision budget exhausted",
            revisions=revisions,
        )

    def _apply_revision(
        self,
        graph: TaskGraph,
        task: OrchestratorTask,
        proposal: RevisionProposal,
    ) -> None:
        """Apply ``proposal`` to ``task`` and reset it + downstream to PENDING."""
        if not task.original_description:
            task.original_description = task.description
        task.description = proposal.revised_description
        if proposal.revised_acceptance_criteria:
            task.acceptance_criteria = proposal.revised_acceptance_criteria
        task.revision_count += 1
        # revision_of points at the *original* step id for traceability;
        # since we update in place, we just record the same id.
        task.revision_of = task.id

        affected = {task.id} | self._collect_downstream(graph, task.id)
        for tid in affected:
            t = graph.tasks.get(tid)
            if t is None:
                continue
            t.status = PENDING
            t.result = None
            t.error = ""
            t.retries = 0
            t.started_at = 0.0
            t.completed_at = 0.0
            t.verification_status = "unverified"
            t.verification_notes = ""
        graph.refresh()

    @staticmethod
    def _collect_downstream(graph: TaskGraph, root_id: str) -> set[str]:
        """Return all transitive dependents of ``root_id`` (excluding root)."""
        visited: set[str] = set()
        stack = [root_id]
        while stack:
            cur = stack.pop()
            for tid, task in graph.tasks.items():
                if cur in task.depends_on and tid not in visited:
                    visited.add(tid)
                    stack.append(tid)
        return visited

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
        """Fallback when no expander is wired — flatten orchestrate to atomic.

        Returns the IDs that were coerced so callers can warn the user.
        """
        coerced: list[str] = []
        for tid, task in graph.tasks.items():
            if task.step_kind == "orchestrate":
                task.step_kind = "atomic"
                coerced.append(tid)
        return coerced

    async def _expand_orchestrate_steps(
        self, graph: TaskGraph,
    ) -> list[dict[str, Any]]:
        """Replace each orchestrate step with parallel sub-tasks + a join.

        For an orchestrate step ``P`` with depends_on=[D] and sub-tasks
        [S1, S2, ...] returned by the expander:
          * Each Si is added with ``depends_on=[D]`` so they fan out from
            where P would have started.
          * P stays in the graph but becomes ``step_kind="atomic"`` with
            ``depends_on=[S1, S2, ...]`` — it joins the sub-task outputs.
            P's description is rewritten as a synthesis instruction.
          * Tasks that previously depended on P keep doing so — the join
            preserves the original dependency edges, so no rewiring needed.

        Returns a list of expansion summaries (one per orchestrate step) for
        broadcast to the UI.
        """
        if self._expander is None:
            return []

        # Snapshot orchestrate steps up front — expansion mutates the graph.
        orchestrate_ids = [
            tid for tid, t in graph.tasks.items() if t.step_kind == "orchestrate"
        ]
        if not orchestrate_ids:
            return []

        expansions: list[dict[str, Any]] = []
        existing_ids: set[str] = set(graph.tasks.keys())

        for parent_id in orchestrate_ids:
            parent = graph.tasks.get(parent_id)
            if parent is None:
                continue

            try:
                subtasks = await self._expander(parent)
            except Exception as e:
                log.error("Expander raised for orchestrate step",
                          task_id=parent_id, error=str(e))
                # Coerce this single step to atomic and continue — don't abort
                # the whole plan because one fan-out couldn't be decomposed.
                parent.step_kind = "atomic"
                expansions.append({
                    "task_id": parent_id,
                    "expanded": False,
                    "error": str(e),
                })
                continue

            if not subtasks:
                log.warning("Expander returned no sub-tasks; coercing to atomic",
                            task_id=parent_id)
                parent.step_kind = "atomic"
                expansions.append({
                    "task_id": parent_id,
                    "expanded": False,
                    "error": "expander returned no sub-tasks",
                })
                continue

            parent_deps = list(parent.depends_on)
            sub_ids: list[str] = []
            for sub in subtasks:
                # Namespace sub-task ids so they don't collide with the rest
                # of the plan (or with sub-tasks of another orchestrate step).
                if not sub.id:
                    sub.id = f"{parent_id}__sub_{len(sub_ids) + 1}"
                else:
                    sub.id = f"{parent_id}__{sub.id}"
                # Defensive: preserve unique ids even if the expander returned
                # duplicates or one collides with a pre-existing plan step.
                base_id = sub.id
                suffix = 2
                while sub.id in existing_ids:
                    sub.id = f"{base_id}_{suffix}"
                    suffix += 1
                existing_ids.add(sub.id)

                sub.depends_on = list(parent_deps)
                sub.step_kind = "atomic"
                # Inherit timeout/retries from the parent so worker config is
                # consistent — the parent values came from worker_timeout.
                if not sub.timeout_seconds:
                    sub.timeout_seconds = parent.timeout_seconds
                if not sub.max_retries:
                    sub.max_retries = parent.max_retries
                graph.add_task(sub)
                sub_ids.append(sub.id)

            # Convert parent into a join/synthesis atomic step.
            original_description = parent.description
            parent.step_kind = "atomic"
            parent.depends_on = sub_ids
            parent.description = (
                f"Synthesize the outputs of the upstream sub-tasks "
                f"({', '.join(sub_ids)}) into a single result that satisfies "
                f"the original step goal:\n\n{original_description}"
            )

            expansions.append({
                "task_id": parent_id,
                "expanded": True,
                "sub_task_ids": sub_ids,
            })

        return expansions

    def _emit(self, event_name: str, data: dict[str, Any]) -> None:
        if self._broadcast is None:
            return
        try:
            self._broadcast({"type": event_name, **data})
        except Exception as e:
            log.debug("Plan-mode event broadcast failed",
                      event_name=event_name, error=str(e))


def orchestrate_expander_from_orchestrator(
    orchestrator: Any,
) -> Callable[[OrchestratorTask], Awaitable[list[OrchestratorTask]]]:
    """Build an expander that calls a SessionOrchestrator's ``_decompose``.

    The returned coroutine takes one orchestrate-kind task and returns a list
    of atomic sub-tasks (each an ``OrchestratorTask``) ready to splice into
    the plan graph.
    """
    async def expand(task: OrchestratorTask) -> list[OrchestratorTask]:
        decompose = getattr(orchestrator, "_decompose", None)
        if decompose is None:
            log.warning("Orchestrator has no _decompose method")
            return []

        prompt = task.description.strip() or task.title.strip()
        if not prompt:
            log.warning("Orchestrate step has empty description and title",
                        task_id=task.id)
            return []

        plan = await decompose(prompt)
        if not isinstance(plan, dict):
            return []

        raw_tasks = plan.get("tasks") or []
        if not isinstance(raw_tasks, list):
            return []

        sub_tasks: list[OrchestratorTask] = []
        for raw in raw_tasks:
            if not isinstance(raw, dict):
                continue
            sub = OrchestratorTask.from_dict(raw)
            if not sub.id and not sub.title:
                continue
            # Sub-tasks always run as atomic; the planner's orchestrate
            # semantics are about fan-out, not nested fan-out.
            sub.step_kind = "atomic"
            # Sub-tasks have no acceptance_criteria from the orchestrator
            # decomposer — verification happens at the join step. Clear any
            # leftover so the verifier's no-criteria short-circuit applies.
            sub.acceptance_criteria = ""
            # Drop deps that would refer to siblings — fan-out is parallel.
            sub.depends_on = []
            sub_tasks.append(sub)
        return sub_tasks

    return expand


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
