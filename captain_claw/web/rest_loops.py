"""REST handlers and execution logic for the loop runner."""

from __future__ import annotations

import asyncio
import json
import time as _time
import uuid as _uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Callable

from aiohttp import web

from captain_claw.logging import get_logger

if TYPE_CHECKING:
    from captain_claw.web_server import WebServer

log = get_logger(__name__)


async def start_loop(server: WebServer, request: web.Request) -> web.Response:
    """POST /api/loops/start — start a loop execution."""
    if not server.agent:
        return web.json_response({"error": "Agent not initialized"}, status=503)

    if server._loop_runner_task and not server._loop_runner_task.done():
        return web.json_response(
            {"error": "A loop is already running"}, status=409,
        )

    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON"}, status=400)

    workflow_name = str(body.get("workflow", "")).strip()
    if not workflow_name:
        return web.json_response({"error": "Missing workflow name"}, status=400)

    iterations_data = body.get("iterations", [])
    if not iterations_data or not isinstance(iterations_data, list):
        return web.json_response({"error": "Missing iterations"}, status=400)

    loop_id = str(_uuid.uuid4())[:8]

    iterations: list[dict[str, Any]] = []
    for i, it in enumerate(iterations_data):
        iterations.append({
            "iteration_index": i,
            "variable_values": it.get("variable_values", {}),
            "status": "pending",
            "duration": None,
            "result_preview": None,
            "error": None,
        })

    server._loop_runner_stop = False
    server._loop_runner_state = {
        "loop_id": loop_id,
        "workflow": workflow_name,
        "state": "running",
        "iterations": iterations,
        "started_at": datetime.now(UTC).isoformat(),
    }

    server._loop_runner_task = asyncio.create_task(
        run_loop(server, loop_id, workflow_name, iterations),
    )

    log.info("loop_started", loop_id=loop_id, workflow=workflow_name,
             iteration_count=len(iterations))
    return web.json_response({"loop_id": loop_id})


async def get_loop_status(server: WebServer, request: web.Request) -> web.Response:
    """GET /api/loops/status — return current loop state."""
    if not server._loop_runner_state:
        return web.json_response({"state": "idle"})
    return web.json_response(
        server._loop_runner_state,
        dumps=lambda obj: json.dumps(obj, default=str),
    )


async def stop_loop(server: WebServer, request: web.Request) -> web.Response:
    """POST /api/loops/stop — signal the running loop to stop."""
    if not server._loop_runner_task or server._loop_runner_task.done():
        return web.json_response({"error": "No loop running"}, status=400)

    server._loop_runner_stop = True
    log.info("loop_stop_requested")
    return web.json_response({"ok": True})


async def run_loop(
    server: WebServer,
    loop_id: str,
    workflow_name: str,
    iterations: list[dict[str, Any]],
) -> None:
    """Execute a workflow N times sequentially with different variable values."""
    from captain_claw.config import get_config
    from captain_claw.session_orchestrator import SessionOrchestrator

    cfg = get_config()

    for i, iteration in enumerate(iterations):
        if server._loop_runner_stop:
            for j in range(i, len(iterations)):
                iterations[j]["status"] = "cancelled"
            server._loop_runner_state["state"] = "stopped"
            server._broadcast({
                "type": "loop_event",
                "event": "loop_stopped",
                "loop_id": loop_id,
                "stopped_at_index": i,
            })
            log.info("loop_stopped", loop_id=loop_id, stopped_at=i)
            return

        iteration["status"] = "running"
        server._broadcast({
            "type": "loop_event",
            "event": "iteration_started",
            "loop_id": loop_id,
            "iteration_index": i,
        })
        log.info("loop_iteration_started", loop_id=loop_id, iteration=i)

        started = _time.time()
        orchestrator: SessionOrchestrator | None = None
        try:
            def _make_loop_broadcast(idx: int) -> Callable[[dict[str, Any]], None]:
                def _cb(payload: dict[str, Any]) -> None:
                    if payload.get("type") == "orchestrator_event":
                        payload = dict(payload)
                        payload["type"] = "loop_orchestrator_event"
                        payload["loop_id"] = loop_id
                        payload["iteration_index"] = idx
                    server._broadcast(payload)
                return _cb

            orchestrator = SessionOrchestrator(
                main_agent=server.agent,
                max_parallel=cfg.orchestrator.max_parallel,
                max_agents=cfg.orchestrator.max_agents,
                provider=server.agent.provider,
                broadcast_callback=_make_loop_broadcast(i),
            )

            load_result = await orchestrator.load_workflow(workflow_name)
            if not load_result.get("ok"):
                raise ValueError(
                    load_result.get("error", f"Failed to load workflow: {workflow_name}"),
                )

            wf_variables = load_result.get("variables") or []
            effective_vars: dict[str, str] = {}
            for v in wf_variables:
                default_val = v.get("default", "")
                if default_val:
                    effective_vars[v["name"]] = default_val
            user_vars = iteration.get("variable_values") or {}
            effective_vars.update(user_vars)

            synthesis = await orchestrator.execute(
                variable_values=effective_vars if effective_vars else None,
            )

            elapsed = _time.time() - started
            iteration["status"] = "completed"
            iteration["duration"] = round(elapsed, 1)
            iteration["result_preview"] = (synthesis[:500] if synthesis else "")

            server._broadcast({
                "type": "loop_event",
                "event": "iteration_completed",
                "loop_id": loop_id,
                "iteration_index": i,
                "duration": iteration["duration"],
                "result_preview": iteration["result_preview"],
            })
            log.info("loop_iteration_completed", loop_id=loop_id,
                     iteration=i, duration=iteration["duration"])

        except Exception as e:
            elapsed = _time.time() - started
            iteration["status"] = "failed"
            iteration["duration"] = round(elapsed, 1)
            iteration["error"] = str(e)

            server._broadcast({
                "type": "loop_event",
                "event": "iteration_failed",
                "loop_id": loop_id,
                "iteration_index": i,
                "duration": iteration["duration"],
                "error": str(e),
            })
            log.error("loop_iteration_failed", loop_id=loop_id,
                      iteration=i, error=str(e))

        finally:
            if orchestrator:
                try:
                    await orchestrator.shutdown()
                except Exception:
                    pass

    server._loop_runner_state["state"] = "completed"
    server._broadcast({
        "type": "loop_event",
        "event": "loop_completed",
        "loop_id": loop_id,
    })
    log.info("loop_completed", loop_id=loop_id,
             total=len(iterations))
