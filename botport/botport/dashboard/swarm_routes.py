"""Swarm orchestration REST API routes."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

from aiohttp import web

from botport.swarm.dag import DAGError, auto_layout, validate_dag
from botport.swarm.models import (
    Swarm,
    SwarmAuditEntry,
    SwarmEdge,
    SwarmProject,
    SwarmTask,
    SwarmTemplate,
    _utcnow_iso,
)

if TYPE_CHECKING:
    from botport.server import BotPortServer


def setup_swarm_routes(app: web.Application, server: BotPortServer) -> None:
    """Register swarm REST API routes."""

    async def _swarm_store():
        return server.swarm_store

    # ── Projects ──────────────────────────────────────────────

    async def list_projects(request: web.Request) -> web.Response:
        store = await _swarm_store()
        projects = await store.list_projects()
        return web.json_response([p.to_dict() for p in projects])

    async def create_project(request: web.Request) -> web.Response:
        data = await request.json()
        name = str(data.get("name", "")).strip()
        if not name:
            return web.json_response({"error": "Name is required"}, status=400)

        now = _utcnow_iso()
        project = SwarmProject(
            id=str(uuid.uuid4()),
            name=name,
            description=str(data.get("description", "")),
            created_at=now,
            updated_at=now,
        )
        store = await _swarm_store()
        await store.save_project(project)
        return web.json_response(project.to_dict(), status=201)

    async def get_project(request: web.Request) -> web.Response:
        project_id = request.match_info["id"]
        store = await _swarm_store()
        project = await store.get_project(project_id)
        if not project:
            return web.json_response({"error": "Not found"}, status=404)

        swarms = await store.list_swarms(project_id=project_id)
        result = project.to_dict()
        result["swarms"] = [s.to_dict() for s in swarms]
        return web.json_response(result)

    async def update_project(request: web.Request) -> web.Response:
        project_id = request.match_info["id"]
        store = await _swarm_store()
        project = await store.get_project(project_id)
        if not project:
            return web.json_response({"error": "Not found"}, status=404)

        data = await request.json()
        if "name" in data:
            project.name = str(data["name"]).strip()
        if "description" in data:
            project.description = str(data["description"])
        project.updated_at = _utcnow_iso()

        await store.save_project(project)
        return web.json_response(project.to_dict())

    async def delete_project(request: web.Request) -> web.Response:
        project_id = request.match_info["id"]
        store = await _swarm_store()
        ok = await store.delete_project(project_id)
        if not ok:
            return web.json_response(
                {"error": "Cannot delete project with running swarms"}, status=409,
            )
        return web.json_response({"ok": True})

    # ── Swarms ────────────────────────────────────────────────

    async def list_swarms(request: web.Request) -> web.Response:
        project_id = request.query.get("project_id")
        store = await _swarm_store()
        swarms = await store.list_swarms(project_id=project_id)
        return web.json_response([s.to_dict() for s in swarms])

    async def create_swarm(request: web.Request) -> web.Response:
        data = await request.json()
        project_id = str(data.get("project_id", "")).strip()
        if not project_id:
            return web.json_response({"error": "project_id is required"}, status=400)

        store = await _swarm_store()
        project = await store.get_project(project_id)
        if not project:
            return web.json_response({"error": "Project not found"}, status=404)

        now = _utcnow_iso()
        swarm = Swarm(
            id=str(uuid.uuid4()),
            project_id=project_id,
            name=str(data.get("name", "")),
            original_task=str(data.get("task", "")),
            created_at=now,
            updated_at=now,
        )
        if "concurrency_limit" in data:
            swarm.concurrency_limit = int(data["concurrency_limit"])
        if "priority" in data:
            swarm.priority = int(data["priority"])
        if "error_policy" in data:
            swarm.error_policy = str(data["error_policy"])
        if "agent_mode" in data:
            swarm.agent_mode = str(data["agent_mode"])

        await store.save_swarm(swarm)

        await store.add_audit_entry(SwarmAuditEntry(
            swarm_id=swarm.id,
            event_type="swarm_created",
            details={"name": swarm.name, "task": swarm.original_task},
            actor="user",
            created_at=now,
        ))

        return web.json_response(swarm.to_dict(), status=201)

    async def get_swarm(request: web.Request) -> web.Response:
        swarm_id = request.match_info["id"]
        store = await _swarm_store()
        swarm = await store.get_swarm(swarm_id)
        if not swarm:
            return web.json_response({"error": "Not found"}, status=404)

        tasks = await store.list_tasks(swarm_id)
        edges = await store.list_edges(swarm_id)

        result = swarm.to_dict()
        result["tasks"] = [t.to_dict() for t in tasks]
        result["edges"] = [e.to_dict() for e in edges]
        return web.json_response(result)

    async def update_swarm(request: web.Request) -> web.Response:
        swarm_id = request.match_info["id"]
        store = await _swarm_store()
        swarm = await store.get_swarm(swarm_id)
        if not swarm:
            return web.json_response({"error": "Not found"}, status=404)

        data = await request.json()
        if "name" in data:
            swarm.name = str(data["name"])
        if "original_task" in data:
            swarm.original_task = str(data["original_task"])
        if "concurrency_limit" in data:
            swarm.concurrency_limit = int(data["concurrency_limit"])
        if "priority" in data:
            swarm.priority = int(data["priority"])
        if "error_policy" in data:
            swarm.error_policy = str(data["error_policy"])
        if "agent_mode" in data:
            swarm.agent_mode = str(data["agent_mode"])
        swarm.touch()

        await store.save_swarm(swarm)
        return web.json_response(swarm.to_dict())

    async def delete_swarm(request: web.Request) -> web.Response:
        swarm_id = request.match_info["id"]
        store = await _swarm_store()
        ok = await store.delete_swarm(swarm_id)
        if not ok:
            return web.json_response(
                {"error": "Cannot delete running swarm"}, status=409,
            )
        return web.json_response({"ok": True})

    # ── Pipeline actions ──────────────────────────────────────

    async def swarm_start(request: web.Request) -> web.Response:
        swarm_id = request.match_info["id"]
        store = await _swarm_store()
        swarm = await store.get_swarm(swarm_id)
        if not swarm:
            return web.json_response({"error": "Not found"}, status=404)
        if swarm.status not in ("draft", "ready", "paused"):
            return web.json_response(
                {"error": f"Cannot start swarm in '{swarm.status}' state"}, status=409,
            )

        # Validate DAG before starting.
        tasks = await store.list_tasks(swarm_id)
        edges = await store.list_edges(swarm_id)
        if not tasks:
            return web.json_response({"error": "Swarm has no tasks"}, status=400)

        try:
            validate_dag(tasks, edges)
        except DAGError as e:
            return web.json_response({"error": str(e)}, status=400)

        swarm.status = "running"
        swarm.started_at = swarm.started_at or _utcnow_iso()
        swarm.touch()
        await store.save_swarm(swarm)

        await store.add_audit_entry(SwarmAuditEntry(
            swarm_id=swarm.id,
            event_type="swarm_started",
            actor="user",
            created_at=_utcnow_iso(),
        ))

        # Create initial checkpoint.
        if server.swarm_engine:
            await server.swarm_engine.create_checkpoint(swarm_id, label="Pre-execution")

        return web.json_response(swarm.to_dict())

    async def swarm_pause(request: web.Request) -> web.Response:
        swarm_id = request.match_info["id"]
        store = await _swarm_store()
        swarm = await store.get_swarm(swarm_id)
        if not swarm:
            return web.json_response({"error": "Not found"}, status=404)
        if swarm.status != "running":
            return web.json_response({"error": "Swarm is not running"}, status=409)

        swarm.status = "paused"
        swarm.touch()
        await store.save_swarm(swarm)

        await store.add_audit_entry(SwarmAuditEntry(
            swarm_id=swarm.id,
            event_type="swarm_paused",
            created_at=_utcnow_iso(),
        ))

        return web.json_response(swarm.to_dict())

    async def swarm_resume(request: web.Request) -> web.Response:
        swarm_id = request.match_info["id"]
        store = await _swarm_store()
        swarm = await store.get_swarm(swarm_id)
        if not swarm:
            return web.json_response({"error": "Not found"}, status=404)
        if swarm.status != "paused":
            return web.json_response({"error": "Swarm is not paused"}, status=409)

        swarm.status = "running"
        swarm.touch()
        await store.save_swarm(swarm)

        await store.add_audit_entry(SwarmAuditEntry(
            swarm_id=swarm.id,
            event_type="swarm_resumed",
            created_at=_utcnow_iso(),
        ))

        return web.json_response(swarm.to_dict())

    async def swarm_cancel(request: web.Request) -> web.Response:
        swarm_id = request.match_info["id"]
        store = await _swarm_store()
        swarm = await store.get_swarm(swarm_id)
        if not swarm:
            return web.json_response({"error": "Not found"}, status=404)
        if swarm.is_terminal:
            return web.json_response({"error": "Swarm is already terminal"}, status=409)

        swarm.status = "cancelled"
        swarm.completed_at = _utcnow_iso()
        swarm.touch()
        await store.save_swarm(swarm)

        await store.add_audit_entry(SwarmAuditEntry(
            swarm_id=swarm.id,
            event_type="swarm_cancelled",
            created_at=_utcnow_iso(),
        ))

        return web.json_response(swarm.to_dict())

    # ── Rephrase & Decompose ──────────────────────────────────

    async def swarm_rephrase(request: web.Request) -> web.Response:
        swarm_id = request.match_info["id"]
        store = await _swarm_store()
        swarm = await store.get_swarm(swarm_id)
        if not swarm:
            return web.json_response({"error": "Not found"}, status=404)
        if not swarm.original_task:
            return web.json_response({"error": "Swarm has no task to rephrase"}, status=400)

        data = await request.json() if request.can_read_body else {}
        model = str(data.get("model", ""))

        from botport.swarm.decomposer import TaskDecomposer
        decomposer = TaskDecomposer()

        try:
            rephrased = await decomposer.rephrase(swarm.original_task, model=model)
        except Exception as e:
            return web.json_response({"error": f"Rephrase failed: {e}"}, status=500)

        swarm.rephrased_task = rephrased
        swarm.touch()
        await store.save_swarm(swarm)

        await store.add_audit_entry(SwarmAuditEntry(
            swarm_id=swarm.id,
            event_type="task_rephrased",
            details={
                "original_length": len(swarm.original_task),
                "rephrased_length": len(rephrased),
            },
            created_at=_utcnow_iso(),
        ))

        return web.json_response(swarm.to_dict())

    async def swarm_decompose(request: web.Request) -> web.Response:
        swarm_id = request.match_info["id"]
        store = await _swarm_store()
        swarm = await store.get_swarm(swarm_id)
        if not swarm:
            return web.json_response({"error": "Not found"}, status=404)

        task_text = swarm.rephrased_task or swarm.original_task
        if not task_text:
            return web.json_response({"error": "Swarm has no task to decompose"}, status=400)

        data = await request.json() if request.can_read_body else {}
        model = str(data.get("model", ""))

        # Set status to decomposing.
        swarm.status = "decomposing"
        swarm.touch()
        await store.save_swarm(swarm)

        # Gather available agents for persona suggestions.
        available_agents = []
        instances = server.connections.list_instances()
        for inst in instances:
            available_agents.append({
                "name": inst.name,
                "personas": [p.to_dict() for p in inst.personas],
            })

        from botport.swarm.decomposer import TaskDecomposer
        decomposer = TaskDecomposer()

        try:
            result = await decomposer.decompose(
                task_text,
                available_agents=available_agents,
                swarm_id=swarm_id,
                model=model,
            )
        except Exception as e:
            swarm.status = "draft"
            swarm.touch()
            await store.save_swarm(swarm)
            return web.json_response({"error": f"Decomposition failed: {e}"}, status=500)

        # Clear existing tasks/edges if re-decomposing.
        existing_tasks = await store.list_tasks(swarm_id)
        for t in existing_tasks:
            await store.delete_task(t.id)

        # Save new tasks and edges.
        for task in result.tasks:
            await store.save_task(task)

        for edge in result.edges:
            await store.save_edge(edge)

        # Auto-layout the new DAG.
        positions = auto_layout(result.tasks, result.edges)
        for task in result.tasks:
            if task.id in positions:
                task.position_x, task.position_y = positions[task.id]
                await store.save_task(task)

        # Update swarm status.
        swarm.status = "ready"
        swarm.metadata["decomposition_reasoning"] = result.reasoning
        swarm.touch()
        await store.save_swarm(swarm)

        await store.add_audit_entry(SwarmAuditEntry(
            swarm_id=swarm.id,
            event_type="task_decomposed",
            details={
                "task_count": len(result.tasks),
                "edge_count": len(result.edges),
                "reasoning": result.reasoning[:500],
            },
            created_at=_utcnow_iso(),
        ))

        # Return full swarm with tasks.
        tasks = await store.list_tasks(swarm_id)
        edges = await store.list_edges(swarm_id)
        resp = swarm.to_dict()
        resp["tasks"] = [t.to_dict() for t in tasks]
        resp["edges"] = [e.to_dict() for e in edges]
        return web.json_response(resp)

    async def swarm_select_agents(request: web.Request) -> web.Response:
        """Re-run agent selection on existing tasks."""
        swarm_id = request.match_info["id"]
        store = await _swarm_store()
        swarm = await store.get_swarm(swarm_id)
        if not swarm:
            return web.json_response({"error": "Not found"}, status=404)

        data = await request.json() if request.can_read_body else {}
        model = str(data.get("model", ""))

        tasks = await store.list_tasks(swarm_id)
        if not tasks:
            return web.json_response({"error": "No tasks to assign"}, status=400)

        available_agents = []
        instances = server.connections.list_instances()
        for inst in instances:
            available_agents.append({
                "name": inst.name,
                "personas": [p.to_dict() for p in inst.personas],
            })

        if not available_agents:
            return web.json_response({"error": "No agents available"}, status=409)

        from botport.swarm.decomposer import TaskDecomposer
        decomposer = TaskDecomposer()

        try:
            assignments = await decomposer.select_agents(tasks, available_agents, model=model)
        except Exception as e:
            return web.json_response({"error": f"Agent selection failed: {e}"}, status=500)

        # Apply assignments.
        for task in tasks:
            if task.id in assignments and assignments[task.id]:
                task.assigned_persona = assignments[task.id]
                task.touch()
                await store.save_task(task)

        await store.add_audit_entry(SwarmAuditEntry(
            swarm_id=swarm.id,
            event_type="agents_selected",
            details={"assignments": assignments},
            created_at=_utcnow_iso(),
        ))

        tasks = await store.list_tasks(swarm_id)
        return web.json_response([t.to_dict() for t in tasks])

    async def swarm_design_agents(request: web.Request) -> web.Response:
        """Generate task-optimized agent specs (persona + model) via LLM."""
        swarm_id = request.match_info["id"]
        store = await _swarm_store()
        swarm = await store.get_swarm(swarm_id)
        if not swarm:
            return web.json_response({"error": "Not found"}, status=404)

        data = await request.json() if request.can_read_body else {}
        model = str(data.get("model", ""))

        tasks = await store.list_tasks(swarm_id)
        if not tasks:
            return web.json_response({"error": "No tasks to design agents for"}, status=400)

        from botport.swarm.agent_designer import AgentDesigner
        designer = AgentDesigner()

        try:
            specs = await designer.design_agents(tasks, model=model)
        except Exception as e:
            return web.json_response({"error": f"Agent design failed: {e}"}, status=500)

        # Apply specs to tasks.
        for task in tasks:
            if task.id in specs:
                task.metadata["agent_spec"] = specs[task.id]
                task.assigned_persona = specs[task.id].get("persona_name", "")
                task.touch()
                await store.save_task(task)

        await store.add_audit_entry(SwarmAuditEntry(
            swarm_id=swarm.id,
            event_type="agents_designed",
            details={
                "task_count": len(tasks),
                "specs_generated": len(specs),
                "unique_personas": len({s["persona_name"] for s in specs.values()}),
            },
            actor="user",
            created_at=_utcnow_iso(),
        ))

        tasks = await store.list_tasks(swarm_id)
        return web.json_response([t.to_dict() for t in tasks])

    # ── Tasks ─────────────────────────────────────────────────

    async def list_tasks(request: web.Request) -> web.Response:
        swarm_id = request.match_info["id"]
        store = await _swarm_store()
        tasks = await store.list_tasks(swarm_id)
        edges = await store.list_edges(swarm_id)
        return web.json_response({
            "tasks": [t.to_dict() for t in tasks],
            "edges": [e.to_dict() for e in edges],
        })

    async def create_task(request: web.Request) -> web.Response:
        swarm_id = request.match_info["id"]
        store = await _swarm_store()
        swarm = await store.get_swarm(swarm_id)
        if not swarm:
            return web.json_response({"error": "Swarm not found"}, status=404)

        data = await request.json()
        description = str(data.get("description", "")).strip()
        if not description:
            return web.json_response({"error": "Description is required"}, status=400)

        now = _utcnow_iso()
        task = SwarmTask(
            id=str(uuid.uuid4()),
            swarm_id=swarm_id,
            name=str(data.get("name", "")),
            description=description,
            priority=int(data.get("priority", 0)),
            assigned_persona=str(data.get("assigned_persona", "")),
            max_retries=int(data.get("max_retries", 3)),
            timeout_seconds=int(data.get("timeout_seconds", 600)),
            position_x=float(data.get("position_x", 0)),
            position_y=float(data.get("position_y", 0)),
            created_at=now,
            updated_at=now,
        )

        if data.get("requires_approval"):
            task.requires_approval = True
        if "timeout_warn_seconds" in data:
            task.timeout_warn_seconds = int(data["timeout_warn_seconds"])
        if "timeout_extend_seconds" in data:
            task.timeout_extend_seconds = int(data["timeout_extend_seconds"])

        if data.get("is_periodic"):
            task.is_periodic = True
            task.cron_expression = str(data.get("cron_expression", ""))

        await store.save_task(task)

        # Create edges if depends_on is provided.
        depends_on = data.get("depends_on", [])
        for dep_id in depends_on:
            edge = SwarmEdge(
                swarm_id=swarm_id,
                from_task_id=str(dep_id),
                to_task_id=task.id,
            )
            await store.save_edge(edge)

        await store.add_audit_entry(SwarmAuditEntry(
            swarm_id=swarm_id,
            task_id=task.id,
            event_type="task_created",
            details={"name": task.name, "description": task.description},
            created_at=now,
        ))

        return web.json_response(task.to_dict(), status=201)

    async def update_task(request: web.Request) -> web.Response:
        task_id = request.match_info["task_id"]
        store = await _swarm_store()
        task = await store.get_task(task_id)
        if not task:
            return web.json_response({"error": "Task not found"}, status=404)

        data = await request.json()
        for field in ("name", "description", "assigned_persona", "fallback_persona",
                      "error_message"):
            if field in data:
                setattr(task, field, str(data[field]))
        for field in ("priority", "max_retries", "retry_backoff_seconds", "timeout_seconds",
                      "timeout_warn_seconds", "timeout_extend_seconds"):
            if field in data:
                setattr(task, field, int(data[field]))
        if "requires_approval" in data:
            task.requires_approval = bool(data["requires_approval"])
        for field in ("position_x", "position_y"):
            if field in data:
                setattr(task, field, float(data[field]))
        task.touch()

        await store.save_task(task)
        return web.json_response(task.to_dict())

    async def delete_task(request: web.Request) -> web.Response:
        task_id = request.match_info["task_id"]
        store = await _swarm_store()
        task = await store.get_task(task_id)
        if not task:
            return web.json_response({"error": "Task not found"}, status=404)
        if task.status == "running":
            return web.json_response({"error": "Cannot delete running task"}, status=409)

        await store.delete_task(task_id)
        return web.json_response({"ok": True})

    async def task_retry(request: web.Request) -> web.Response:
        task_id = request.match_info["task_id"]
        store = await _swarm_store()
        task = await store.get_task(task_id)
        if not task:
            return web.json_response({"error": "Task not found"}, status=404)
        if task.status not in ("failed", "skipped"):
            return web.json_response({"error": "Task is not in a retryable state"}, status=409)

        task.status = "queued"
        task.retry_count = 0
        task.error_message = ""
        task.concern_id = ""
        task.touch()
        await store.save_task(task)

        await store.add_audit_entry(SwarmAuditEntry(
            swarm_id=task.swarm_id,
            task_id=task.id,
            event_type="task_manual_retry",
            created_at=_utcnow_iso(),
        ))

        return web.json_response(task.to_dict())

    async def task_skip(request: web.Request) -> web.Response:
        task_id = request.match_info["task_id"]
        store = await _swarm_store()
        task = await store.get_task(task_id)
        if not task:
            return web.json_response({"error": "Task not found"}, status=404)
        if task.status in ("completed", "skipped"):
            return web.json_response({"error": "Task is already terminal"}, status=409)

        task.status = "skipped"
        task.completed_at = _utcnow_iso()
        task.touch()
        await store.save_task(task)

        await store.add_audit_entry(SwarmAuditEntry(
            swarm_id=task.swarm_id,
            task_id=task.id,
            event_type="task_skipped",
            created_at=_utcnow_iso(),
        ))

        return web.json_response(task.to_dict())

    # ── Task approval & intervention ─────────────────────────

    async def task_approve(request: web.Request) -> web.Response:
        task_id = request.match_info["task_id"]
        store = await _swarm_store()
        task = await store.get_task(task_id)
        if not task:
            return web.json_response({"error": "Task not found"}, status=404)
        if task.status != "pending_approval":
            return web.json_response({"error": "Task is not pending approval"}, status=409)

        data = await request.json() if request.can_read_body else {}
        task.approval_status = "approved"
        task.approved_by = str(data.get("approved_by", "user"))
        task.status = "queued"
        task.touch()
        await store.save_task(task)

        await store.add_audit_entry(SwarmAuditEntry(
            swarm_id=task.swarm_id,
            task_id=task.id,
            event_type="task_approved",
            details={"approved_by": task.approved_by},
            actor="user",
            created_at=_utcnow_iso(),
        ))

        return web.json_response(task.to_dict())

    async def task_reject(request: web.Request) -> web.Response:
        task_id = request.match_info["task_id"]
        store = await _swarm_store()
        task = await store.get_task(task_id)
        if not task:
            return web.json_response({"error": "Task not found"}, status=404)
        if task.status != "pending_approval":
            return web.json_response({"error": "Task is not pending approval"}, status=409)

        data = await request.json() if request.can_read_body else {}
        task.approval_status = "rejected"
        task.approved_by = str(data.get("rejected_by", "user"))
        task.status = "skipped"
        task.completed_at = _utcnow_iso()
        task.error_message = str(data.get("reason", "Rejected by user"))
        task.touch()
        await store.save_task(task)

        await store.add_audit_entry(SwarmAuditEntry(
            swarm_id=task.swarm_id,
            task_id=task.id,
            event_type="task_rejected",
            details={"rejected_by": task.approved_by, "reason": task.error_message},
            actor="user",
            created_at=_utcnow_iso(),
        ))

        return web.json_response(task.to_dict())

    async def task_override_output(request: web.Request) -> web.Response:
        """Manually set a task's output and mark it completed."""
        task_id = request.match_info["task_id"]
        store = await _swarm_store()
        task = await store.get_task(task_id)
        if not task:
            return web.json_response({"error": "Task not found"}, status=404)
        if task.status in ("running", "completed"):
            return web.json_response({"error": f"Cannot override task in '{task.status}' state"}, status=409)

        data = await request.json()
        output = data.get("output_data", {})
        if isinstance(output, str):
            output = {"response": output}

        task.status = "completed"
        task.completed_at = _utcnow_iso()
        task.output_data = output
        task.error_message = ""
        task.touch()
        await store.save_task(task)

        await store.add_audit_entry(SwarmAuditEntry(
            swarm_id=task.swarm_id,
            task_id=task.id,
            event_type="task_output_overridden",
            details={"output_keys": list(output.keys()) if isinstance(output, dict) else []},
            actor="user",
            created_at=_utcnow_iso(),
        ))

        return web.json_response(task.to_dict())

    async def task_pause(request: web.Request) -> web.Response:
        task_id = request.match_info["task_id"]
        store = await _swarm_store()
        task = await store.get_task(task_id)
        if not task:
            return web.json_response({"error": "Task not found"}, status=404)
        if task.status not in ("queued", "pending_approval"):
            return web.json_response({"error": f"Cannot pause task in '{task.status}' state"}, status=409)

        task.status = "paused"
        task.touch()
        await store.save_task(task)

        await store.add_audit_entry(SwarmAuditEntry(
            swarm_id=task.swarm_id,
            task_id=task.id,
            event_type="task_paused",
            actor="user",
            created_at=_utcnow_iso(),
        ))

        return web.json_response(task.to_dict())

    async def task_unpause(request: web.Request) -> web.Response:
        task_id = request.match_info["task_id"]
        store = await _swarm_store()
        task = await store.get_task(task_id)
        if not task:
            return web.json_response({"error": "Task not found"}, status=404)
        if task.status != "paused":
            return web.json_response({"error": "Task is not paused"}, status=409)

        task.status = "queued"
        task.touch()
        await store.save_task(task)

        await store.add_audit_entry(SwarmAuditEntry(
            swarm_id=task.swarm_id,
            task_id=task.id,
            event_type="task_unpaused",
            actor="user",
            created_at=_utcnow_iso(),
        ))

        return web.json_response(task.to_dict())

    # ── Checkpoints ──────────────────────────────────────────

    async def list_checkpoints(request: web.Request) -> web.Response:
        swarm_id = request.match_info["id"]
        store = await _swarm_store()
        checkpoints = await store.list_checkpoints(swarm_id)
        return web.json_response([c.to_dict() for c in checkpoints])

    async def create_checkpoint(request: web.Request) -> web.Response:
        swarm_id = request.match_info["id"]
        data = await request.json() if request.can_read_body else {}
        label = str(data.get("label", "Manual checkpoint"))

        if not server.swarm_engine:
            return web.json_response({"error": "Engine not available"}, status=503)

        checkpoint = await server.swarm_engine.create_checkpoint(swarm_id, label=label)
        if not checkpoint:
            return web.json_response({"error": "Swarm not found"}, status=404)

        return web.json_response(checkpoint.to_dict(), status=201)

    async def restore_checkpoint(request: web.Request) -> web.Response:
        checkpoint_id = request.match_info["checkpoint_id"]

        if not server.swarm_engine:
            return web.json_response({"error": "Engine not available"}, status=503)

        ok = await server.swarm_engine.restore_checkpoint(checkpoint_id)
        if not ok:
            return web.json_response(
                {"error": "Cannot restore: checkpoint not found or swarm is still running"},
                status=409,
            )

        return web.json_response({"ok": True, "checkpoint_id": checkpoint_id})

    async def delete_checkpoint(request: web.Request) -> web.Response:
        checkpoint_id = request.match_info["checkpoint_id"]
        store = await _swarm_store()
        await store.delete_checkpoint(checkpoint_id)
        return web.json_response({"ok": True})

    # ── Edges ─────────────────────────────────────────────────

    async def create_edge(request: web.Request) -> web.Response:
        swarm_id = request.match_info["id"]
        data = await request.json()

        from_task_id = str(data.get("from_task_id", ""))
        to_task_id = str(data.get("to_task_id", ""))
        if not from_task_id or not to_task_id:
            return web.json_response({"error": "from_task_id and to_task_id required"}, status=400)

        store = await _swarm_store()

        # Validate: both tasks exist and belong to this swarm.
        tasks = await store.list_tasks(swarm_id)
        task_ids = {t.id for t in tasks}
        if from_task_id not in task_ids or to_task_id not in task_ids:
            return web.json_response({"error": "Task not found in this swarm"}, status=404)

        edge = SwarmEdge(
            swarm_id=swarm_id,
            from_task_id=from_task_id,
            to_task_id=to_task_id,
            edge_type=str(data.get("edge_type", "dependency")),
        )

        # Validate DAG would still be valid.
        edges = await store.list_edges(swarm_id)
        edges.append(edge)
        try:
            validate_dag(tasks, edges)
        except DAGError as e:
            return web.json_response({"error": str(e)}, status=400)

        edge_id = await store.save_edge(edge)
        edge.id = edge_id
        return web.json_response(edge.to_dict(), status=201)

    async def delete_edge(request: web.Request) -> web.Response:
        edge_id = int(request.match_info["edge_id"])
        store = await _swarm_store()
        await store.delete_edge(edge_id)
        return web.json_response({"ok": True})

    # ── Auto-layout ───────────────────────────────────────────

    async def swarm_auto_layout(request: web.Request) -> web.Response:
        swarm_id = request.match_info["id"]
        store = await _swarm_store()
        tasks = await store.list_tasks(swarm_id)
        edges = await store.list_edges(swarm_id)

        positions = auto_layout(tasks, edges)
        for task in tasks:
            if task.id in positions:
                task.position_x, task.position_y = positions[task.id]
                await store.save_task(task)

        return web.json_response(positions)

    # ── Artifacts ─────────────────────────────────────────────

    async def list_artifacts(request: web.Request) -> web.Response:
        swarm_id = request.match_info["id"]
        task_id = request.query.get("task_id")
        store = await _swarm_store()
        artifacts = await store.list_artifacts(swarm_id, task_id=task_id)
        return web.json_response([a.to_dict() for a in artifacts])

    # ── Observability ─────────────────────────────────────────

    async def swarm_timeline(request: web.Request) -> web.Response:
        swarm_id = request.match_info["id"]
        store = await _swarm_store()
        swarm = await store.get_swarm(swarm_id)
        if not swarm:
            return web.json_response({"error": "Not found"}, status=404)

        tasks = await store.list_tasks(swarm_id)
        edges = await store.list_edges(swarm_id)

        from botport.swarm.dag import get_task_depth
        timeline_tasks = []
        for task in tasks:
            timeline_tasks.append({
                "id": task.id,
                "name": task.name,
                "status": task.status,
                "depth": get_task_depth(task.id, edges),
                "started_at": task.started_at,
                "completed_at": task.completed_at,
            })

        return web.json_response({
            "tasks": timeline_tasks,
            "swarm_started_at": swarm.started_at,
            "now": _utcnow_iso(),
        })

    async def swarm_costs(request: web.Request) -> web.Response:
        swarm_id = request.match_info["id"]
        store = await _swarm_store()
        summary = await store.get_cost_summary(swarm_id)
        return web.json_response(summary)

    async def swarm_audit(request: web.Request) -> web.Response:
        swarm_id = request.match_info["id"]
        store = await _swarm_store()
        entries = await store.list_audit_log(swarm_id)
        return web.json_response([e.to_dict() for e in entries])

    # ── Swarm files ────────────────────────────────────────────

    async def swarm_list_files(request: web.Request) -> web.Response:
        """List all files in a swarm workspace."""
        swarm_id = request.match_info["id"]
        fm = server.file_manager
        files = fm.list_files(swarm_id)
        # Strip internal float modified_at for JSON.
        for f in files:
            f.pop("modified_at", None)
        total_size = fm.workspace_size(swarm_id)
        return web.json_response({
            "swarm_id": swarm_id,
            "files": files,
            "total_size": total_size,
        })

    async def swarm_download_file(request: web.Request) -> web.Response:
        """Download a file from the swarm workspace."""
        swarm_id = request.match_info["id"]
        file_path = request.match_info["file_path"]
        fm = server.file_manager
        data = fm.get_file(swarm_id, file_path)
        if data is None:
            return web.json_response({"error": "File not found"}, status=404)

        from botport.swarm.file_manager import guess_mime_type
        mime = guess_mime_type(file_path)
        filename = Path(file_path).name

        return web.Response(
            body=data,
            content_type=mime,
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
                "Content-Length": str(len(data)),
            },
        )

    # ── LLM config ────────────────────────────────────────────

    async def llm_config(request: web.Request) -> web.Response:
        """Return LLM model list from Captain Claw's config for the UI model selector."""
        from botport.swarm.decomposer import get_cc_default_model, get_cc_models

        default = get_cc_default_model()
        models = get_cc_models()
        return web.json_response({
            "default_provider": default["provider"],
            "default_model": default["model"],
            "models": models,
        })

    async def llm_reload_config(request: web.Request) -> web.Response:
        """Reload CC config (e.g. after settings change)."""
        from botport.swarm.decomposer import reload_cc_config
        reload_cc_config()
        return web.json_response({"ok": True})

    # ── Templates ────────────────────────────────────────────

    async def list_templates(request: web.Request) -> web.Response:
        store = await _swarm_store()
        templates = await store.list_templates()
        return web.json_response([t.to_dict() for t in templates])

    async def get_template(request: web.Request) -> web.Response:
        template_id = request.match_info["template_id"]
        store = await _swarm_store()
        template = await store.get_template(template_id)
        if not template:
            return web.json_response({"error": "Not found"}, status=404)
        return web.json_response(template.to_dict())

    async def create_template(request: web.Request) -> web.Response:
        data = await request.json()
        name = str(data.get("name", "")).strip()
        if not name:
            return web.json_response({"error": "Name is required"}, status=400)

        template = SwarmTemplate(
            id=str(uuid.uuid4()),
            name=name,
            description=str(data.get("description", "")),
            dag_definition=data.get("dag_definition", {}),
            created_at=_utcnow_iso(),
            metadata=data.get("metadata", {}),
        )
        store = await _swarm_store()
        await store.save_template(template)
        return web.json_response(template.to_dict(), status=201)

    async def save_swarm_as_template(request: web.Request) -> web.Response:
        """Save an existing swarm's DAG as a reusable template."""
        swarm_id = request.match_info["id"]
        store = await _swarm_store()
        swarm = await store.get_swarm(swarm_id)
        if not swarm:
            return web.json_response({"error": "Not found"}, status=404)

        tasks = await store.list_tasks(swarm_id)
        edges = await store.list_edges(swarm_id)

        data = await request.json() if request.can_read_body else {}
        name = str(data.get("name", swarm.name or "Untitled Template")).strip()

        dag_def = {
            "tasks": [
                {
                    "name": t.name,
                    "description": t.description,
                    "priority": t.priority,
                    "assigned_persona": t.assigned_persona,
                    "max_retries": t.max_retries,
                    "timeout_seconds": t.timeout_seconds,
                    "requires_approval": t.requires_approval,
                    "is_periodic": t.is_periodic,
                    "cron_expression": t.cron_expression,
                    "position_x": t.position_x,
                    "position_y": t.position_y,
                    "_original_id": t.id,
                }
                for t in tasks
            ],
            "edges": [
                {
                    "from_task_id": e.from_task_id,
                    "to_task_id": e.to_task_id,
                    "edge_type": e.edge_type,
                }
                for e in edges
            ],
        }

        template = SwarmTemplate(
            id=str(uuid.uuid4()),
            name=name,
            description=str(data.get("description", swarm.original_task or "")),
            dag_definition=dag_def,
            created_at=_utcnow_iso(),
            metadata={"source_swarm_id": swarm_id},
        )
        await store.save_template(template)

        await store.add_audit_entry(SwarmAuditEntry(
            swarm_id=swarm_id,
            event_type="template_saved",
            details={"template_id": template.id, "name": name},
            actor="user",
            created_at=_utcnow_iso(),
        ))

        return web.json_response(template.to_dict(), status=201)

    async def instantiate_template(request: web.Request) -> web.Response:
        """Create a new swarm from a template."""
        template_id = request.match_info["template_id"]
        store = await _swarm_store()
        template = await store.get_template(template_id)
        if not template:
            return web.json_response({"error": "Template not found"}, status=404)

        data = await request.json()
        project_id = str(data.get("project_id", "")).strip()
        if not project_id:
            return web.json_response({"error": "project_id is required"}, status=400)

        project = await store.get_project(project_id)
        if not project:
            return web.json_response({"error": "Project not found"}, status=404)

        now = _utcnow_iso()
        swarm = Swarm(
            id=str(uuid.uuid4()),
            project_id=project_id,
            name=str(data.get("name", template.name)),
            original_task=str(data.get("task", template.description)),
            status="ready",
            template_id=template_id,
            created_at=now,
            updated_at=now,
        )
        if "concurrency_limit" in data:
            swarm.concurrency_limit = int(data["concurrency_limit"])
        if "error_policy" in data:
            swarm.error_policy = str(data["error_policy"])
        await store.save_swarm(swarm)

        # Create tasks from template.
        dag_def = template.dag_definition
        template_tasks = dag_def.get("tasks", [])
        template_edges = dag_def.get("edges", [])

        # Map original task IDs to new ones.
        id_map: dict[str, str] = {}
        from botport.swarm.scheduler import next_cron_time

        for task_def in template_tasks:
            new_id = str(uuid.uuid4())
            original_id = task_def.get("_original_id", "")
            if original_id:
                id_map[original_id] = new_id

            task = SwarmTask(
                id=new_id,
                swarm_id=swarm.id,
                name=str(task_def.get("name", "")),
                description=str(task_def.get("description", "")),
                priority=int(task_def.get("priority", 0)),
                assigned_persona=str(task_def.get("assigned_persona", "")),
                max_retries=int(task_def.get("max_retries", 3)),
                timeout_seconds=int(task_def.get("timeout_seconds", 600)),
                requires_approval=bool(task_def.get("requires_approval", False)),
                is_periodic=bool(task_def.get("is_periodic", False)),
                cron_expression=str(task_def.get("cron_expression", "")),
                position_x=float(task_def.get("position_x", 0)),
                position_y=float(task_def.get("position_y", 0)),
                created_at=now,
                updated_at=now,
            )
            if task.is_periodic and task.cron_expression:
                task.next_run_at = next_cron_time(task.cron_expression)
            await store.save_task(task)

        # Create edges using the ID map.
        for edge_def in template_edges:
            from_id = id_map.get(edge_def.get("from_task_id", ""), "")
            to_id = id_map.get(edge_def.get("to_task_id", ""), "")
            if from_id and to_id:
                edge = SwarmEdge(
                    swarm_id=swarm.id,
                    from_task_id=from_id,
                    to_task_id=to_id,
                    edge_type=str(edge_def.get("edge_type", "dependency")),
                )
                await store.save_edge(edge)

        await store.add_audit_entry(SwarmAuditEntry(
            swarm_id=swarm.id,
            event_type="swarm_from_template",
            details={"template_id": template_id, "template_name": template.name},
            actor="user",
            created_at=now,
        ))

        # Return full swarm with tasks.
        tasks = await store.list_tasks(swarm.id)
        edges = await store.list_edges(swarm.id)
        resp = swarm.to_dict()
        resp["tasks"] = [t.to_dict() for t in tasks]
        resp["edges"] = [e.to_dict() for e in edges]
        return web.json_response(resp, status=201)

    async def update_template(request: web.Request) -> web.Response:
        template_id = request.match_info["template_id"]
        store = await _swarm_store()
        template = await store.get_template(template_id)
        if not template:
            return web.json_response({"error": "Not found"}, status=404)

        data = await request.json()
        if "name" in data:
            template.name = str(data["name"]).strip()
        if "description" in data:
            template.description = str(data["description"])
        if "dag_definition" in data:
            template.dag_definition = data["dag_definition"]
        if "metadata" in data:
            template.metadata = data["metadata"]
        await store.save_template(template)
        return web.json_response(template.to_dict())

    async def delete_template(request: web.Request) -> web.Response:
        template_id = request.match_info["template_id"]
        store = await _swarm_store()
        await store.delete_template(template_id)
        return web.json_response({"ok": True})

    # ── Cost details ─────────────────────────────────────────

    async def swarm_cost_details(request: web.Request) -> web.Response:
        swarm_id = request.match_info["id"]
        store = await _swarm_store()
        entries = await store.list_cost_log(swarm_id)
        return web.json_response(entries)

    # ── Register routes ───────────────────────────────────────

    # Projects.
    app.router.add_get("/api/swarm/projects", list_projects)
    app.router.add_post("/api/swarm/projects", create_project)
    app.router.add_get("/api/swarm/projects/{id}", get_project)
    app.router.add_put("/api/swarm/projects/{id}", update_project)
    app.router.add_delete("/api/swarm/projects/{id}", delete_project)

    # Swarms.
    app.router.add_get("/api/swarm/swarms", list_swarms)
    app.router.add_post("/api/swarm/swarms", create_swarm)
    app.router.add_get("/api/swarm/swarms/{id}", get_swarm)
    app.router.add_put("/api/swarm/swarms/{id}", update_swarm)
    app.router.add_delete("/api/swarm/swarms/{id}", delete_swarm)

    # Pipeline actions.
    app.router.add_post("/api/swarm/swarms/{id}/start", swarm_start)
    app.router.add_post("/api/swarm/swarms/{id}/pause", swarm_pause)
    app.router.add_post("/api/swarm/swarms/{id}/resume", swarm_resume)
    app.router.add_post("/api/swarm/swarms/{id}/cancel", swarm_cancel)
    app.router.add_post("/api/swarm/swarms/{id}/auto-layout", swarm_auto_layout)
    app.router.add_post("/api/swarm/swarms/{id}/rephrase", swarm_rephrase)
    app.router.add_post("/api/swarm/swarms/{id}/decompose", swarm_decompose)
    app.router.add_post("/api/swarm/swarms/{id}/select-agents", swarm_select_agents)
    app.router.add_post("/api/swarm/swarms/{id}/design-agents", swarm_design_agents)

    # Tasks.
    app.router.add_get("/api/swarm/swarms/{id}/tasks", list_tasks)
    app.router.add_post("/api/swarm/swarms/{id}/tasks", create_task)
    app.router.add_put("/api/swarm/tasks/{task_id}", update_task)
    app.router.add_delete("/api/swarm/tasks/{task_id}", delete_task)
    app.router.add_post("/api/swarm/tasks/{task_id}/retry", task_retry)
    app.router.add_post("/api/swarm/tasks/{task_id}/skip", task_skip)
    app.router.add_post("/api/swarm/tasks/{task_id}/approve", task_approve)
    app.router.add_post("/api/swarm/tasks/{task_id}/reject", task_reject)
    app.router.add_post("/api/swarm/tasks/{task_id}/override-output", task_override_output)
    app.router.add_post("/api/swarm/tasks/{task_id}/pause", task_pause)
    app.router.add_post("/api/swarm/tasks/{task_id}/unpause", task_unpause)

    # Edges.
    app.router.add_post("/api/swarm/swarms/{id}/edges", create_edge)
    app.router.add_delete("/api/swarm/edges/{edge_id}", delete_edge)

    # Checkpoints.
    app.router.add_get("/api/swarm/swarms/{id}/checkpoints", list_checkpoints)
    app.router.add_post("/api/swarm/swarms/{id}/checkpoints", create_checkpoint)
    app.router.add_post("/api/swarm/checkpoints/{checkpoint_id}/restore", restore_checkpoint)
    app.router.add_delete("/api/swarm/checkpoints/{checkpoint_id}", delete_checkpoint)

    # LLM config.
    app.router.add_get("/api/swarm/llm-config", llm_config)
    app.router.add_post("/api/swarm/llm-config/reload", llm_reload_config)

    # Templates.
    app.router.add_get("/api/swarm/templates", list_templates)
    app.router.add_post("/api/swarm/templates", create_template)
    app.router.add_get("/api/swarm/templates/{template_id}", get_template)
    app.router.add_put("/api/swarm/templates/{template_id}", update_template)
    app.router.add_delete("/api/swarm/templates/{template_id}", delete_template)
    app.router.add_post("/api/swarm/swarms/{id}/save-as-template", save_swarm_as_template)
    app.router.add_post("/api/swarm/templates/{template_id}/instantiate", instantiate_template)

    # Artifacts & Observability.
    app.router.add_get("/api/swarm/swarms/{id}/artifacts", list_artifacts)
    app.router.add_get("/api/swarm/swarms/{id}/timeline", swarm_timeline)
    app.router.add_get("/api/swarm/swarms/{id}/costs", swarm_costs)
    app.router.add_get("/api/swarm/swarms/{id}/cost-details", swarm_cost_details)
    app.router.add_get("/api/swarm/swarms/{id}/audit", swarm_audit)

    # Files.
    app.router.add_get("/api/swarm/swarms/{id}/files", swarm_list_files)
    app.router.add_get("/api/swarm/swarms/{id}/files/{file_path:.*}", swarm_download_file)
