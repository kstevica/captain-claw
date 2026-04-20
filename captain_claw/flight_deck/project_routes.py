"""Project management REST endpoints for Flight Deck."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from captain_claw.logging import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/fd/projects", tags=["projects"])


# ── Helpers ──────────────────────────────────────────────────────────

async def _get_pm():
    """Get ProjectManager, initializing if needed."""
    from captain_claw.projects import get_project_manager, ProjectManager, set_project_manager

    pm = get_project_manager()
    if pm is not None:
        return pm

    # Try to initialize from session manager.
    try:
        from captain_claw.session import get_session_manager
        sm = get_session_manager()
        await sm._ensure_db()
        db = getattr(sm, "_db", None)
        if db is None:
            raise HTTPException(503, "Session database not available")
        pm = ProjectManager(db)
        await pm.ensure_tables()
        set_project_manager(pm)
        return pm
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(503, f"Project system init failed: {exc}")


# ── Request / response models ────────────────────────────────────────

class CreateProjectRequest(BaseModel):
    name: str
    description: str = ""
    goals: list[dict[str, Any]] = Field(default_factory=list)
    config: dict[str, Any] = Field(default_factory=dict)
    lead_agent_id: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class UpdateProjectRequest(BaseModel):
    name: str | None = None
    description: str | None = None
    status: str | None = None
    goals: list[dict[str, Any]] | None = None
    config: dict[str, Any] | None = None
    lead_agent_id: str | None = None
    completed_at: str | None = None
    metadata: dict[str, Any] | None = None


class JoinProjectRequest(BaseModel):
    agent_id: str
    agent_name: str = ""
    role: str = "contributor"
    expertise_tags: list[str] = Field(default_factory=list)


class LeaveProjectRequest(BaseModel):
    agent_id: str
    contribution_summary: str = ""


class LinkSessionRequest(BaseModel):
    session_id: str
    session_name: str = ""
    agent_id: str = ""
    purpose: str = ""


class AddArtifactRequest(BaseModel):
    kind: str  # decision | milestone | deliverable | note | blocker
    title: str
    content: str = ""
    created_by: str = ""
    session_id: str = ""
    status: str = "active"


class UpdateArtifactRequest(BaseModel):
    title: str | None = None
    content: str | None = None
    status: str | None = None
    kind: str | None = None


class AddGoalRequest(BaseModel):
    goal: str
    success_criteria: str = ""
    priority: str = "normal"


class UpdateGoalRequest(BaseModel):
    goal_index: int
    goal: str | None = None
    success_criteria: str | None = None
    priority: str | None = None
    status: str | None = None
    notes: str | None = None


# ── Project CRUD ─────────────────────────────────────────────────────

@router.get("")
async def list_projects(status: str | None = None, limit: int = 50):
    pm = await _get_pm()
    return await pm.list_projects(status=status, limit=limit)


@router.post("")
async def create_project(body: CreateProjectRequest):
    pm = await _get_pm()
    try:
        return await pm.create(
            body.name,
            description=body.description,
            goals=body.goals,
            config=body.config,
            lead_agent_id=body.lead_agent_id,
            metadata=body.metadata,
        )
    except Exception as exc:
        if "UNIQUE" in str(exc):
            raise HTTPException(409, f"Project name already exists: {body.name}")
        raise HTTPException(500, str(exc))


@router.get("/{project_id}")
async def get_project(project_id: str):
    pm = await _get_pm()
    project = await pm.get(project_id)
    if not project:
        project = await pm.get_by_name(project_id)
    if not project:
        raise HTTPException(404, "Project not found")
    return project


@router.put("/{project_id}")
async def update_project(project_id: str, body: UpdateProjectRequest):
    pm = await _get_pm()
    fields = body.model_dump(exclude_none=True)
    result = await pm.update(project_id, **fields)
    if not result:
        raise HTTPException(404, "Project not found")
    return result


@router.delete("/{project_id}")
async def delete_project(project_id: str):
    pm = await _get_pm()
    ok = await pm.delete(project_id)
    if not ok:
        raise HTTPException(404, "Project not found")
    return {"ok": True}


# ── Context (rich summary) ───────────────────────────────────────────

@router.get("/{project_id}/context")
async def get_project_context(project_id: str):
    pm = await _get_pm()
    ctx = await pm.get_project_context(project_id)
    if not ctx:
        raise HTTPException(404, "Project not found")
    return ctx


# ── Members ──────────────────────────────────────────────────────────

@router.get("/{project_id}/members")
async def list_members(project_id: str, active_only: bool = True):
    pm = await _get_pm()
    if active_only:
        return await pm.active_members(project_id)
    return await pm.all_members(project_id)


@router.post("/{project_id}/members")
async def join_project(project_id: str, body: JoinProjectRequest):
    pm = await _get_pm()
    project = await pm.get(project_id)
    if not project:
        raise HTTPException(404, "Project not found")
    return await pm.join(
        project_id,
        body.agent_id,
        agent_name=body.agent_name,
        role=body.role,
        expertise_tags=body.expertise_tags,
    )


@router.delete("/{project_id}/members/{agent_id}")
async def leave_project(
    project_id: str,
    agent_id: str,
    contribution_summary: str = "",
):
    pm = await _get_pm()
    ok = await pm.leave(project_id, agent_id, contribution_summary=contribution_summary)
    if not ok:
        raise HTTPException(404, "Active membership not found")
    return {"ok": True}


# ── Sessions ─────────────────────────────────────────────────────────

@router.get("/{project_id}/sessions")
async def list_sessions(project_id: str):
    pm = await _get_pm()
    return await pm.project_sessions(project_id)


@router.post("/{project_id}/sessions")
async def link_session(project_id: str, body: LinkSessionRequest):
    pm = await _get_pm()
    project = await pm.get(project_id)
    if not project:
        raise HTTPException(404, "Project not found")
    return await pm.link_session(
        project_id,
        body.session_id,
        session_name=body.session_name,
        agent_id=body.agent_id,
        purpose=body.purpose,
    )


@router.post("/{project_id}/sessions/{session_id}/end")
async def end_session(project_id: str, session_id: str):
    pm = await _get_pm()
    ok = await pm.end_session(project_id, session_id)
    if not ok:
        raise HTTPException(404, "Active session link not found")
    return {"ok": True}


# ── Artifacts ────────────────────────────────────────────────────────

@router.get("/{project_id}/artifacts")
async def list_artifacts(
    project_id: str,
    kind: str | None = None,
    status: str | None = None,
    limit: int = 50,
):
    pm = await _get_pm()
    return await pm.list_artifacts(project_id, kind=kind, status=status, limit=limit)


@router.post("/{project_id}/artifacts")
async def add_artifact(project_id: str, body: AddArtifactRequest):
    pm = await _get_pm()
    project = await pm.get(project_id)
    if not project:
        raise HTTPException(404, "Project not found")
    return await pm.add_artifact(
        project_id,
        body.kind,
        body.title,
        content=body.content,
        created_by=body.created_by,
        session_id=body.session_id,
        status=body.status,
    )


@router.put("/{project_id}/artifacts/{artifact_id}")
async def update_artifact(
    project_id: str, artifact_id: str, body: UpdateArtifactRequest,
):
    pm = await _get_pm()
    fields = body.model_dump(exclude_none=True)
    result = await pm.update_artifact(artifact_id, **fields)
    if not result:
        raise HTTPException(404, "Artifact not found")
    return result


# ── Goals ────────────────────────────────────────────────────────────

@router.post("/{project_id}/goals")
async def add_goal(project_id: str, body: AddGoalRequest):
    pm = await _get_pm()
    result = await pm.add_goal(
        project_id,
        body.goal,
        success_criteria=body.success_criteria,
        priority=body.priority,
    )
    if not result:
        raise HTTPException(404, "Project not found")
    return result


@router.put("/{project_id}/goals")
async def update_goal(project_id: str, body: UpdateGoalRequest):
    pm = await _get_pm()
    fields = body.model_dump(exclude_none=True, exclude={"goal_index"})
    result = await pm.update_goal(project_id, body.goal_index, **fields)
    if not result:
        raise HTTPException(404, "Project or goal not found")
    return result


# ── Activity ─────────────────────────────────────────────────────────

@router.get("/{project_id}/activity")
async def get_activity(
    project_id: str,
    limit: int = 20,
    event_type: str | None = None,
):
    pm = await _get_pm()
    return await pm.recent_activity(project_id, limit=limit, event_type=event_type)


# ── Memory search (proxied to semantic + insights) ───────────────────

@router.get("/{project_id}/memory")
async def search_project_memory(project_id: str, q: str = "", limit: int = 10):
    if not q.strip():
        raise HTTPException(400, "Query parameter 'q' is required")
    pm = await _get_pm()
    project = await pm.get(project_id)
    if not project:
        raise HTTPException(404, "Project not found")

    sessions = await pm.project_sessions(project_id)
    session_ids = [s["session_id"] for s in sessions]
    if not session_ids:
        return {"results": [], "message": "No sessions linked to this project"}

    try:
        from captain_claw.config import get_config
        from pathlib import Path
        import sqlite3
        cfg = get_config()
        db_path = Path(cfg.memory.path).expanduser()
        if not db_path.exists():
            return {"results": [], "message": "Semantic memory database not found"}
        conn = sqlite3.connect(str(db_path), check_same_thread=False)
        placeholders = ",".join("?" for _ in session_ids)
        rows = conn.execute(
            f"""SELECT chunk_id, source, reference, path, start_line, end_line,
                       text, updated_at, text_l1, text_l2
                FROM memory_chunks
                WHERE (source = 'workspace' OR reference IN ({placeholders}))
                AND text LIKE ?
                ORDER BY updated_at DESC
                LIMIT ?""",
            (*session_ids, f"%{q.strip()}%", limit),
        ).fetchall()
        conn.close()
        results = []
        for r in rows:
            results.append({
                "chunk_id": r[0], "source": r[1], "reference": r[2],
                "path": r[3], "start_line": r[4], "end_line": r[5],
                "text": r[6][:500], "updated_at": r[7],
            })
        return {"results": results}
    except Exception as exc:
        return {"results": [], "error": str(exc)}


@router.get("/{project_id}/insights")
async def search_project_insights(
    project_id: str, q: str = "", limit: int = 10,
):
    pm = await _get_pm()
    project = await pm.get(project_id)
    if not project:
        raise HTTPException(404, "Project not found")

    from captain_claw.insights import get_insights_manager
    mgr = get_insights_manager()
    if q.strip():
        results = await mgr.search_in_project(q.strip(), project_id, limit=limit)
    else:
        results = await mgr.list_project_insights(project_id, limit=limit)
    return results


# ── Goal checking: probe agents for progress ────────────────────────


class GoalCheckRequest(BaseModel):
    """Request to check goal progress by analyzing agent activity."""
    provider: str = "anthropic"
    model: str = "claude-sonnet-4-20250514"
    api_key: str = ""


GOAL_CHECK_PROMPT = """You are a project manager analyzing agent activity logs to assess goal progress.

Given a project's goals and the recent activity from team agents, determine the status of each goal.

Respond with JSON only — no markdown fences, no explanation:
{
  "goals": [
    {
      "goal": "exact goal text",
      "status": "not_started | in_progress | blocked | done",
      "progress_pct": 0-100,
      "evidence": "what the agent(s) actually did toward this goal",
      "remaining": "what still needs to be done (empty if done)",
      "agents_involved": ["agent names that worked on this"]
    }
  ],
  "summary": "1-2 sentence overall project progress summary"
}

Rules:
- Assess EVERY goal listed, even if no agent worked on it
- Base your assessment only on concrete evidence from the activity logs
- If no activity relates to a goal, mark it as not_started with 0% progress
- Be honest — don't inflate progress. If an agent attempted something but failed, note that
- Include the agent name(s) that contributed to each goal"""


@router.post("/{project_id}/goals/check")
async def check_goals(project_id: str, body: GoalCheckRequest, request: Request):
    """Analyze agent activity to assess goal progress using LLM."""
    import httpx

    pm = await _get_pm()
    project = await pm.get(project_id)
    if not project:
        raise HTTPException(404, "Project not found")

    goals = project.get("goals", [])
    if not goals:
        raise HTTPException(400, "Project has no goals")

    # Gather agent activity (same logic as agent-activity endpoint)
    sessions = await pm.project_sessions(project_id)
    members = await pm.active_members(project_id)
    agent_name_map: dict[str, str] = {}
    for m in members:
        agent_name_map[m.get("agent_id", "")] = m.get("agent_name") or m.get("agent_id", "")

    port_map = _resolve_fleet_ports()
    from captain_claw.flight_deck.server import _resolve_agent_auth

    all_activity: list[str] = []

    async with httpx.AsyncClient(timeout=10) as client:
        for sess in sessions:
            session_id = sess.get("session_id", "")
            agent_id = sess.get("agent_id", "")
            agent_name = agent_name_map.get(agent_id, agent_id)
            port = port_map.get(agent_name)
            if not port:
                continue

            auth = _resolve_agent_auth(port)
            headers = {"Authorization": f"Bearer {auth}"} if auth else {}

            try:
                resp = await client.get(
                    f"http://localhost:{port}/api/sessions/{session_id}",
                    headers=headers,
                )
                if resp.status_code != 200:
                    continue
                data = resp.json()
                for msg in data.get("messages", []):
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    if not content or role == "system":
                        continue
                    tool_name = msg.get("tool_name", "")
                    prefix = f"[{agent_name}] ({role}"
                    if tool_name:
                        prefix += f" / {tool_name}"
                    prefix += ")"
                    # Truncate very long messages
                    text = content[:1000] if len(content) > 1000 else content
                    all_activity.append(f"{prefix}: {text}")
            except Exception as exc:
                log.debug("goal-check: failed to fetch from %s:%s: %s", agent_name, port, exc)

    # Build LLM prompt
    goals_text = "\n".join(
        f"- [{g.get('priority', 'normal')}] {g['goal']}"
        + (f" (criteria: {g['success_criteria']})" if g.get("success_criteria") else "")
        + (f" (current status: {g.get('status', 'pending')})" if g.get("status") else "")
        for g in goals
    )

    if all_activity:
        activity_text = "\n".join(all_activity[-200:])  # last 200 entries max
    else:
        activity_text = "(No agent activity recorded yet)"

    user_prompt = (
        f"## Project: {project['name']}\n"
        f"{project.get('description', '')}\n\n"
        f"## Goals\n{goals_text}\n\n"
        f"## Agent Activity Log\n{activity_text}"
    )

    from captain_claw.llm import create_provider, Message
    llm = create_provider(
        provider=body.provider,
        model=body.model,
        api_key=body.api_key or None,
        temperature=0.3,
        max_tokens=4096,
    )
    response = await llm.complete(
        messages=[
            Message(role="system", content=GOAL_CHECK_PROMPT),
            Message(role="user", content=user_prompt),
        ],
        temperature=0.3,
        max_tokens=4096,
    )
    content = response.content.strip()
    if content.startswith("```"):
        lines = content.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        content = "\n".join(lines)

    try:
        result = json.loads(content)
    except json.JSONDecodeError:
        result = {"goals": [], "summary": content, "raw": True}

    await pm.log_activity(project_id, "goal_check", detail={
        "goals_checked": len(goals),
        "activity_entries": len(all_activity),
    })

    return result


# ── Dispatch: send goals to agents (hidden, kept for API compat) ─────


class DispatchPlanRequest(BaseModel):
    """Request a dispatch plan (step 1)."""
    goal_indices: list[int] | None = None  # None = all pending goals
    agent_names: list[str] | None = None  # None = all active members
    provider: str = "anthropic"
    model: str = "claude-sonnet-4-20250514"
    api_key: str = ""


class DispatchExecuteRequest(BaseModel):
    """Execute a previously generated plan (step 2)."""
    plan: dict[str, Any]  # the plan object from /dispatch/plan


DISPATCH_PLANNER_PROMPT = """You are a project dispatch planner. Given a project's goals and a team of available agents, create an optimal task assignment plan.

Analyze each goal and each agent's name, role, and expertise. Then assign goals to agents based on best fit. If a goal is complex, decompose it into concrete subtasks.

Respond with JSON only — no markdown fences, no explanation:
{
  "assignments": [
    {
      "agent_name": "exact agent name",
      "goals": ["goal text assigned to this agent"],
      "subtasks": ["specific actionable subtask 1", "specific actionable subtask 2"],
      "rationale": "why this agent is best suited",
      "priority_order": "which subtask to tackle first and why"
    }
  ],
  "coordination_notes": "any dependencies between agents or sequencing advice"
}

Rules:
- Every goal must be assigned to at least one agent
- Prefer assigning goals to agents whose role/expertise aligns
- If there are more goals than agents, some agents get multiple goals
- If there are more agents than goals, assign the most suitable agents and give others a supporting/review role
- Decompose vague goals into 2-4 concrete subtasks with clear deliverables
- Include coordination notes if agents need to share results or sequence work"""


def _resolve_fleet_ports() -> dict[str, int]:
    """Build name→port map from Docker containers and process agents."""
    from captain_claw.flight_deck.server import _load_process_registry, _process_is_alive
    port_map: dict[str, int] = {}

    try:
        from captain_claw.flight_deck.server import get_docker, CONTAINER_LABEL
        client = get_docker()
        for c in client.containers.list(all=False, filters={"label": CONTAINER_LABEL}):
            labels = c.labels or {}
            wp = labels.get("flight-deck.web-port", "")
            name = labels.get("flight-deck.agent-name", c.name)
            if wp and c.status == "running":
                port_map[name] = int(wp)
    except Exception:
        pass

    registry = _load_process_registry()
    for slug, entry in registry.items():
        if _process_is_alive(slug) and entry.get("web_port"):
            port_map[entry.get("name", slug)] = entry["web_port"]

    return port_map


async def _plan_dispatch(
    project: dict, selected_goals: list[dict], members: list[dict],
    provider_name: str, model: str, api_key: str,
) -> dict:
    """Use LLM to create an intelligent goal→agent assignment plan."""
    goals_text = "\n".join(
        f"- [{g.get('priority', 'normal')}] {g['goal']}"
        + (f" (success criteria: {g['success_criteria']})" if g.get("success_criteria") else "")
        for g in selected_goals
    )
    team_text = "\n".join(
        f"- {m.get('agent_name') or m.get('agent_id', '?')} "
        f"(role: {m.get('role', 'contributor')}"
        + (f", expertise: {', '.join(m.get('expertise_tags', []))}" if m.get("expertise_tags") else "")
        + ")"
        for m in members
    )
    user_prompt = (
        f"## Project: {project['name']}\n"
        f"{project.get('description', '')}\n\n"
        f"## Goals to assign\n{goals_text}\n\n"
        f"## Available agents\n{team_text}"
    )

    from captain_claw.llm import create_provider, Message
    llm = create_provider(
        provider=provider_name,
        model=model,
        api_key=api_key or None,
        temperature=0.4,
        max_tokens=4096,
    )
    response = await llm.complete(
        messages=[
            Message(role="system", content=DISPATCH_PLANNER_PROMPT),
            Message(role="user", content=user_prompt),
        ],
        temperature=0.4,
        max_tokens=4096,
    )
    content = response.content.strip()
    if content.startswith("```"):
        lines = content.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        content = "\n".join(lines)
    return json.loads(content)


def _build_agent_message(
    project: dict, assignment: dict, team_names: list[str], coordination_notes: str,
) -> str:
    """Build a personalized task message for one agent based on the plan."""
    subtasks = assignment.get("subtasks", [])
    goals = assignment.get("goals", [])
    rationale = assignment.get("rationale", "")
    priority = assignment.get("priority_order", "")

    goals_section = "\n".join(f"  - {g}" for g in goals)
    subtasks_section = "\n".join(f"  {i+1}. {t}" for i, t in enumerate(subtasks))

    teammates = [n for n in team_names if n != assignment["agent_name"]]
    team_line = ", ".join(teammates) if teammates else "you are working solo"

    parts = [
        f"## Project: {project['name']}",
        f"{project.get('description', '')}",
        f"\n### Your Assignment",
        f"**Why you:** {rationale}" if rationale else "",
        f"\n**Goals assigned to you:**\n{goals_section}",
    ]
    if subtasks:
        parts.append(f"\n**Your task plan:**\n{subtasks_section}")
    if priority:
        parts.append(f"\n**Priority:** {priority}")
    parts += [
        f"\n### Team",
        f"Working with: {team_line}",
    ]
    if coordination_notes:
        parts.append(f"**Coordination:** {coordination_notes}")
    parts += [
        f"\n### Instructions",
        "Work through your assigned subtasks in order.",
        "",
        "**Collaborating with teammates:** Use the `flight_deck` tool to communicate with other agents:",
        "- `consult` action: ask a teammate a question and get their response",
        "- `delegate` action: hand off a subtask to a teammate",
        "Do NOT use shell commands (curl, wget, etc.) to contact other agents. Always use the `flight_deck` tool.",
        "",
        "**Tracking progress:** Use the `project_memory` tool to:",
        "- `contribute` your findings and results back to the project",
        "- `search` for what teammates have already found",
        "- `status` to check overall project progress",
        "When you complete a goal, contribute a summary of your results as an artifact.",
    ]
    return "\n".join(p for p in parts if p)


async def _send_to_agent(
    pm: Any, project_id: str, member: dict, port: int, message: str,
) -> dict:
    """Connect to an agent via WebSocket, link session, and send task."""
    import websockets
    from captain_claw.flight_deck.server import _resolve_agent_auth

    agent_name = member.get("agent_name") or member.get("agent_id", "")
    try:
        auth = _resolve_agent_auth(port)
        token_param = f"?token={auth}" if auth else ""
        uri = f"ws://localhost:{port}/ws{token_param}"

        async with websockets.connect(uri, close_timeout=5) as ws:
            raw_welcome = await asyncio.wait_for(ws.recv(), timeout=10)
            session_id = ""
            try:
                welcome_data = json.loads(raw_welcome)
                session_id = welcome_data.get("session", {}).get("id", "")
            except Exception:
                pass

            log.info("dispatch: connected", agent=agent_name, port=port, session_id=session_id)

            if session_id:
                try:
                    await pm.link_session(
                        project_id, session_id,
                        session_name=f"{agent_name} dispatch",
                        agent_id=member.get("agent_id", ""),
                        purpose="dispatched",
                    )
                except Exception as link_err:
                    log.warning("dispatch: session link failed", error=str(link_err))

            await ws.send(json.dumps({"type": "chat", "content": message}))
            return {"agent": agent_name, "port": port, "status": "dispatched", "session_id": session_id}

    except Exception as exc:
        log.error("dispatch: send failed", agent=agent_name, error=str(exc))
        return {"agent": agent_name, "port": port, "status": "error", "error": str(exc)}


@router.post("/{project_id}/dispatch/plan")
async def dispatch_plan(project_id: str, body: DispatchPlanRequest, request: Request):
    """Step 1: Generate an LLM-powered dispatch plan for review.

    Returns the plan with goal→agent assignments, subtasks, and coordination
    notes. The user can review/edit before executing.
    """
    pm = await _get_pm()
    project = await pm.get(project_id)
    if not project:
        raise HTTPException(404, "Project not found")

    goals = project.get("goals", [])
    if not goals:
        raise HTTPException(400, "Project has no goals")

    if body.goal_indices is not None:
        selected_goals = [goals[i] for i in body.goal_indices if 0 <= i < len(goals)]
    else:
        selected_goals = [g for g in goals if g.get("status") in ("pending", "in_progress")]
    if not selected_goals:
        raise HTTPException(400, "No actionable goals to dispatch")

    members = await pm.active_members(project_id)
    if body.agent_names:
        name_set = set(body.agent_names)
        members = [m for m in members if (m.get("agent_name") or m.get("agent_id")) in name_set]
    if not members:
        raise HTTPException(400, "No active members to dispatch to")

    # Check fleet availability
    port_map = _resolve_fleet_ports()
    matched_names: list[str] = []
    unmatched: list[str] = []
    for m in members:
        name = m.get("agent_name") or m.get("agent_id", "")
        if port_map.get(name):
            matched_names.append(name)
        else:
            unmatched.append(name)
    if not matched_names:
        raise HTTPException(400, f"No members found in fleet. Unmatched: {unmatched}")

    try:
        plan = await _plan_dispatch(
            project, selected_goals, members,
            body.provider, body.model, body.api_key,
        )
    except Exception as exc:
        log.error("dispatch: planning failed", error=str(exc))
        raise HTTPException(502, f"Planning failed: {exc}")

    return {
        "plan": plan,
        "goals_count": len(selected_goals),
        "matched_agents": matched_names,
        "unmatched_agents": unmatched,
    }


@router.post("/{project_id}/dispatch/execute")
async def dispatch_execute(project_id: str, body: DispatchExecuteRequest, request: Request):
    """Step 2: Execute a reviewed dispatch plan — send tasks to agents."""
    pm = await _get_pm()
    project = await pm.get(project_id)
    if not project:
        raise HTTPException(404, "Project not found")

    plan = body.plan
    assignments = plan.get("assignments", [])
    coordination_notes = plan.get("coordination_notes", "")

    if not assignments:
        raise HTTPException(400, "Plan has no assignments")

    # Resolve fleet ports
    port_map = _resolve_fleet_ports()
    members = await pm.active_members(project_id)
    member_lookup: dict[str, tuple[dict, int]] = {}
    for m in members:
        name = m.get("agent_name") or m.get("agent_id", "")
        port = port_map.get(name)
        if port:
            member_lookup[name] = (m, port)

    team_names = list(member_lookup.keys())

    # Send personalized tasks to each assigned agent
    send_tasks = []
    for assignment in assignments:
        agent_name = assignment.get("agent_name", "")
        entry = member_lookup.get(agent_name)
        if not entry:
            continue
        member, port = entry
        message = _build_agent_message(project, assignment, team_names, coordination_notes)
        send_tasks.append(_send_to_agent(pm, project_id, member, port, message))

    results = await asyncio.gather(*send_tasks) if send_tasks else []

    dispatched_count = sum(1 for r in results if r["status"] == "dispatched")
    await pm.log_activity(project_id, "dispatch", detail={
        "agents_dispatched": dispatched_count,
        "agents_failed": len(results) - dispatched_count,
        "plan": plan,
    })

    # Mark dispatched goals as in_progress
    goals = project.get("goals", [])
    assigned_goal_texts = set()
    for a in assignments:
        for g in a.get("goals", []):
            assigned_goal_texts.add(g)
    for i, g in enumerate(goals):
        if g.get("status") == "pending" and g.get("goal") in assigned_goal_texts:
            await pm.update_goal(project_id, i, status="in_progress")

    return {
        "dispatched": dispatched_count,
        "total_agents": len(send_tasks),
        "results": results,
    }


# ── Agent activity: proxy chat history from project agents ────────────


@router.get("/{project_id}/agent-activity")
async def get_agent_activity(project_id: str, limit: int = 50):
    """Fetch recent chat/tool activity from agents linked to this project.

    For each project session, fetches the agent's session messages via their
    REST API, merges and sorts by timestamp.
    """
    import httpx

    pm = await _get_pm()
    project = await pm.get(project_id)
    if not project:
        raise HTTPException(404, "Project not found")

    # Get project sessions (linked during dispatch)
    sessions = await pm.project_sessions(project_id)
    if not sessions:
        return []

    # Get members to map agent_id → name
    members = await pm.active_members(project_id)
    agent_name_map: dict[str, str] = {}
    for m in members:
        agent_name_map[m.get("agent_id", "")] = m.get("agent_name") or m.get("agent_id", "")

    # Resolve ports
    port_map = _resolve_fleet_ports()
    from captain_claw.flight_deck.server import _resolve_agent_auth

    all_events: list[dict] = []

    async with httpx.AsyncClient(timeout=10) as client:
        for sess in sessions:
            session_id = sess.get("session_id", "")
            agent_id = sess.get("agent_id", "")
            agent_name = agent_name_map.get(agent_id, agent_id)

            # Find port for this agent
            port = port_map.get(agent_name)
            if not port:
                continue

            auth = _resolve_agent_auth(port)
            headers = {"Authorization": f"Bearer {auth}"} if auth else {}

            try:
                resp = await client.get(
                    f"http://localhost:{port}/api/sessions/{session_id}",
                    headers=headers,
                )
                if resp.status_code != 200:
                    continue
                data = resp.json()
                for msg in data.get("messages", []):
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    if not content or role == "system":
                        continue
                    # Truncate long content for the activity feed
                    if len(content) > 500:
                        content = content[:500] + "..."
                    all_events.append({
                        "agent_name": agent_name,
                        "role": role,
                        "content": content,
                        "timestamp": msg.get("timestamp", ""),
                        "tool_name": msg.get("tool_name", ""),
                        "session_id": session_id,
                    })
            except Exception as exc:
                log.debug("agent-activity: failed to fetch from %s:%s: %s", agent_name, port, exc)

    # Sort by timestamp descending, limit
    all_events.sort(key=lambda e: e.get("timestamp", ""), reverse=True)
    return all_events[:limit]
