"""Plan-mode slash command handlers (`/plan`, `/plan-show`, ...).

Step 2 ships ``/plan <request>``: generate a plan, save it as a workflow file,
load it into the orchestrator for preview, and render it to the chat.
Execution comes in step 3 (``/plan-execute``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from captain_claw.config import get_config
from captain_claw.logging import get_logger
from captain_claw.plan_mode import Plan, PlanExecutor, PlanGenerator, PlanVerifier
from captain_claw.session_orchestrator import SessionOrchestrator, _scan_workspace_tree

if TYPE_CHECKING:
    from captain_claw.web_server import WebServer

log = get_logger(__name__)


async def handle_plan_command(server: "WebServer", request: str) -> str:
    """Generate a plan for ``request`` and persist it as a workflow file."""
    if not request.strip():
        return "Usage: `/plan <request>`"

    if not server.agent:
        return "Agent not available."

    provider = getattr(server.agent, "provider", None)
    if provider is None:
        return "No LLM provider available — cannot generate a plan."

    # Reuse the orchestrator's instruction loader when present so plan-mode
    # honors the same nano/micro/personal-override resolution as /orchestrate.
    instructions = None
    if server._orchestrator is not None:
        instructions = getattr(server._orchestrator, "_instructions", None)
    generator = PlanGenerator(provider=provider, instructions=instructions)

    workspace_tree = _safe_workspace_tree()

    plan = await generator.generate(request, workspace_tree=workspace_tree)
    if plan is None:
        return "Plan generation failed — see server logs for details."

    name = _plan_name(plan)
    path = _save_plan_file(name, plan)
    log.info("Plan generated", name=name, path=str(path), step_count=len(plan.tasks))

    # Load into orchestrator so /plan-execute (step 3) and the workflow UI
    # can pick it up. Best-effort — failure here is not fatal for display.
    if server._orchestrator is not None:
        try:
            await server._orchestrator.load_workflow(name)
        except Exception as e:
            log.warning("Loaded plan but orchestrator load_workflow failed",
                        error=str(e))

    return _render_plan_response(plan, name, path)


def _safe_workspace_tree() -> str:
    try:
        ws = get_config().resolved_workspace_path()
        return _scan_workspace_tree(ws)
    except Exception as e:
        log.debug("Workspace scan for plan failed (non-fatal)", error=str(e))
        return ""


def _plan_name(plan: Plan) -> str:
    """Derive a workflow filename from the plan summary or the user input."""
    seed = plan.summary or plan.user_input
    base = SessionOrchestrator._generate_workflow_name(seed)
    # Prefix so plan-mode workflows are distinguishable from /orchestrate ones.
    return f"plan-{base}" if not base.startswith("plan-") else base


def _save_plan_file(name: str, plan: Plan) -> Path:
    cfg = get_config()
    workflows_dir = cfg.resolved_workspace_path() / "workflows"
    workflows_dir.mkdir(parents=True, exist_ok=True)
    safe = SessionOrchestrator._safe_filename(name)
    path = workflows_dir / f"{safe}.json"
    payload = plan.to_workflow_dict(name)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


async def handle_plan_execute_command(server: "WebServer", arg: str) -> str:
    """Run the currently loaded plan, or load by name first if ``arg`` is given."""
    if server._orchestrator is None:
        return "Orchestrator not available."

    if arg.strip():
        load_result = await server._orchestrator.load_workflow(arg.strip())
        if not load_result.get("ok"):
            return load_result.get("error", f"Failed to load plan '{arg.strip()}'.")

    if getattr(server._orchestrator, "_graph", None) is None:
        return "No plan loaded. Run `/plan <request>` first or pass a plan name."

    verifier = _build_verifier(server)
    executor = PlanExecutor(
        server._orchestrator,
        broadcast=getattr(server, "_broadcast", None),
        verifier=verifier,
    )
    outcome = await executor.run()

    if not outcome.ok:
        head = "❌ Plan execution stopped"
        details = []
        if outcome.failed_step:
            details.append(f"failed at step `{outcome.failed_step}`")
        if outcome.verification_failed_step:
            details.append(
                f"verification failed at step `{outcome.verification_failed_step}`"
            )
        if outcome.error:
            details.append(f"error: {outcome.error}")
        if outcome.completed_steps:
            details.append(
                f"completed: {', '.join(outcome.completed_steps)}"
            )
        if outcome.verified_steps:
            details.append(f"verified: {', '.join(outcome.verified_steps)}")
        body = " — ".join(details) if details else ""
        tail = f"\n\n{outcome.final_output}" if outcome.final_output else ""
        return f"{head}{(' — ' + body) if body else ''}{tail}"

    completed = ", ".join(outcome.completed_steps) if outcome.completed_steps else "—"
    verified_line = ""
    if outcome.verified_steps:
        verified_line = f"\nVerified: {', '.join(outcome.verified_steps)}"
    return (
        f"✅ Plan executed ({len(outcome.completed_steps)} steps: {completed})"
        f"{verified_line}\n\n"
        f"{outcome.final_output}"
    )


def _build_verifier(server: "WebServer") -> PlanVerifier | None:
    """Construct a PlanVerifier using the agent's provider and orchestrator instructions."""
    if not server.agent:
        return None
    provider = getattr(server.agent, "provider", None)
    if provider is None:
        return None
    instructions = None
    if server._orchestrator is not None:
        instructions = getattr(server._orchestrator, "_instructions", None)
    return PlanVerifier(provider=provider, instructions=instructions)


def _render_plan_response(plan: Plan, name: str, path: Path) -> str:
    header = f"📋 Plan saved as **{name}** (`{path.name}`)\n"
    body = plan.render_markdown()
    footer = (
        "\n\n_Run_ `/plan-execute` _to run this plan, or_ "
        "`/orchestrate-execute` _for the existing fan-out path._"
    )
    return f"{header}\n{body}{footer}"
