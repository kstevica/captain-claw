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
from captain_claw.plan_mode import (
    DEFAULT_MAX_REVISIONS,
    DEFAULT_PLAN_LEVEL,
    Plan,
    PlanExecutor,
    PlanGenerator,
    PlanReviser,
    PlanVerifier,
    normalize_plan_level,
    orchestrate_expander_from_orchestrator,
)
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

    # Resolve enrichment per the agent's current plan-mode level. Each level
    # is cumulative; see ``captain_claw.plan_mode.PLAN_LEVELS``.
    level = normalize_plan_level(getattr(server.agent, "plan_mode_level", DEFAULT_PLAN_LEVEL))
    enrichment = await _build_enrichment(level, request)

    plan = await generator.generate(
        request,
        workspace_tree=workspace_tree,
        reflection_block=enrichment["reflection_block"],
        insights_block=enrichment["insights_block"],
        personality_block=enrichment["personality_block"],
        system_prompt_name=enrichment["system_prompt_name"],
    )
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


# Number of insights pulled from the curated store at the "insightful" level.
# Mirrors the per-turn budget the main agent uses in its system prompt — eight
# is enough to surface preferences/decisions/deadlines without flooding the
# planner with low-relevance facts.
_PLAN_INSIGHTS_LIMIT = 8


async def _build_enrichment(level: str, user_input: str) -> dict[str, str]:
    """Resolve the four enrichment knobs for the given plan-mode level.

    Returns a dict with keys ``reflection_block``, ``insights_block``,
    ``personality_block`` and ``system_prompt_name`` — exactly what
    ``PlanGenerator.generate`` expects. Each block is empty when the level
    doesn't include it, and the keys are always present so the caller can
    splat unconditionally.

    Levels are cumulative:
        plain      → all blocks empty, default planner template.
        enriched   → reflection only.
        insightful → reflection + insights.
        complete   → reflection + insights + persona + planner template
                     biased toward this agent's role.

    Failures in any single layer (reflections dir missing, insights DB not
    initialised, personality module raising) are downgraded to empty strings
    so a misconfigured layer never blocks a plan from being generated.
    """
    out: dict[str, str] = {
        "reflection_block": "",
        "insights_block": "",
        "personality_block": "",
        "system_prompt_name": "plan_mode_system_prompt.md",
    }

    if level == "plain":
        return out

    # ── enriched / insightful / complete: reflection ──
    try:
        from captain_claw.reflections import (
            load_latest_reflection,
            reflection_to_prompt_block,
        )
        block = reflection_to_prompt_block(load_latest_reflection())
        if block.strip():
            out["reflection_block"] = block
    except Exception as e:
        log.debug("Plan reflection enrichment failed (non-fatal)", error=str(e))

    if level == "enriched":
        return out

    # ── insightful / complete: top-N insights ──
    try:
        from captain_claw.insights import get_insights_manager
        rows = await get_insights_manager().search(
            user_input, limit=_PLAN_INSIGHTS_LIMIT,
        )
        out["insights_block"] = _format_insights_rows(rows)
    except Exception as e:
        log.debug("Plan insights enrichment failed (non-fatal)", error=str(e))

    if level == "insightful":
        return out

    # ── complete: persona + cognitive-mode-aware template ──
    try:
        from captain_claw.personality import (
            load_effective_personality,
            personality_to_prompt_block,
        )
        persona = load_effective_personality()
        block = personality_to_prompt_block(persona) if persona is not None else ""
        if block.strip():
            out["personality_block"] = (
                "\nAgent persona (you are planning AS this agent):\n"
                + block
                + "\n"
            )
    except Exception as e:
        log.debug("Plan persona enrichment failed (non-fatal)", error=str(e))

    out["system_prompt_name"] = "plan_mode_complete_system_prompt.md"
    return out


def _format_insights_rows(rows: list[dict]) -> str:
    """Render the top-N insights rows as a planner prompt block."""
    if not rows:
        return ""
    lines: list[str] = [
        "\nRelevant facts about the user (curated insight store; treat as authoritative):",
    ]
    for r in rows:
        category = (r.get("category") or "fact").strip()
        content = (r.get("content") or "").strip()
        if not content:
            continue
        importance = r.get("importance") or 5
        lines.append(f"- [{category}, importance {importance}] {content}")
    if len(lines) == 1:
        return ""
    return "\n".join(lines) + "\n"


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
    reviser = _build_reviser(server)
    expander = orchestrate_expander_from_orchestrator(server._orchestrator)
    executor = PlanExecutor(
        server._orchestrator,
        broadcast=getattr(server, "_broadcast", None),
        verifier=verifier,
        expander=expander,
        reviser=reviser,
        max_revisions=DEFAULT_MAX_REVISIONS,
        cancel_event=getattr(server.agent, "cancel_event", None) if server.agent else None,
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
        if outcome.revisions:
            details.append(f"revisions attempted: {len(outcome.revisions)}")
        body = " — ".join(details) if details else ""
        tail = f"\n\n{outcome.final_output}" if outcome.final_output else ""
        return f"{head}{(' — ' + body) if body else ''}{tail}"

    completed = ", ".join(outcome.completed_steps) if outcome.completed_steps else "—"
    verified_line = ""
    if outcome.verified_steps:
        verified_line = f"\nVerified: {', '.join(outcome.verified_steps)}"
    revisions_line = ""
    if outcome.revisions:
        revisions_line = (
            f"\nRevisions: {len(outcome.revisions)} "
            f"(steps: {', '.join(r['task_id'] for r in outcome.revisions)})"
        )
    return (
        f"✅ Plan executed ({len(outcome.completed_steps)} steps: {completed})"
        f"{verified_line}{revisions_line}\n\n"
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


def _build_reviser(server: "WebServer") -> PlanReviser | None:
    if not server.agent:
        return None
    provider = getattr(server.agent, "provider", None)
    if provider is None:
        return None
    instructions = None
    if server._orchestrator is not None:
        instructions = getattr(server._orchestrator, "_instructions", None)
    return PlanReviser(provider=provider, instructions=instructions)


def _render_plan_response(plan: Plan, name: str, path: Path) -> str:
    header = f"📋 Plan saved as **{name}** (`{path.name}`)\n"
    body = plan.render_markdown()
    footer = (
        "\n\n_Run_ `/plan-execute` _to run this plan, or_ "
        "`/orchestrate-execute` _for the existing fan-out path._"
    )
    return f"{header}\n{body}{footer}"
