"""REST handlers for cron job management."""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING

from aiohttp import web

from captain_claw.logging import get_logger

if TYPE_CHECKING:
    from captain_claw.web_server import WebServer

log = get_logger(__name__)


async def list_cron_jobs(server: WebServer, request: web.Request) -> web.Response:
    """GET /api/cron/jobs — list all cron jobs."""
    if not server.agent:
        return web.json_response({"error": "Agent not initialized"}, status=503)
    sm = server.agent.session_manager
    jobs = await sm.list_cron_jobs(limit=200, active_only=False)
    result = []
    for j in jobs:
        result.append({
            "id": j.id,
            "kind": j.kind,
            "payload": j.payload,
            "schedule": j.schedule,
            "session_id": j.session_id,
            "enabled": j.enabled,
            "created_at": j.created_at,
            "updated_at": j.updated_at,
            "last_run_at": j.last_run_at,
            "next_run_at": j.next_run_at,
            "last_status": j.last_status,
            "last_error": j.last_error,
        })
    return web.json_response(result)


async def create_cron_job(server: WebServer, request: web.Request) -> web.Response:
    """POST /api/cron/jobs — create a new cron job."""
    if not server.agent:
        return web.json_response({"error": "Agent not initialized"}, status=503)
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON body"}, status=400)

    kind = str(body.get("kind", "")).strip().lower()
    if kind not in ("prompt", "script", "tool", "orchestrate"):
        return web.json_response({"error": "Invalid kind"}, status=400)

    schedule = body.get("schedule")
    if not isinstance(schedule, dict) or "type" not in schedule:
        return web.json_response({"error": "Invalid schedule"}, status=400)

    payload = body.get("payload")
    if not isinstance(payload, dict):
        return web.json_response({"error": "Invalid payload"}, status=400)

    session_id = str(body.get("session_id", "")).strip()
    if not session_id:
        return web.json_response({"error": "session_id is required"}, status=400)

    from captain_claw.cron import compute_next_run, schedule_to_text, to_utc_iso

    schedule["_text"] = schedule_to_text(schedule)

    try:
        next_run = to_utc_iso(compute_next_run(schedule))
    except Exception as e:
        return web.json_response({"error": f"Bad schedule: {e}"}, status=400)

    sm = server.agent.session_manager
    job = await sm.create_cron_job(
        kind=kind,
        payload=payload,
        schedule=schedule,
        session_id=session_id,
        next_run_at=next_run,
    )
    return web.json_response({
        "ok": True,
        "id": job.id,
        "next_run_at": job.next_run_at,
    })


async def run_cron_job(server: WebServer, request: web.Request) -> web.Response:
    """POST /api/cron/jobs/{id}/run — execute a job immediately."""
    if not server.agent:
        return web.json_response({"error": "Agent not initialized"}, status=503)

    job_id = request.match_info.get("id", "")
    sm = server.agent.session_manager
    job = await sm.load_cron_job(job_id)
    if not job:
        return web.json_response({"error": "Job not found"}, status=404)

    from captain_claw.cron_dispatch import execute_cron_job

    try:
        ctx = server._get_web_runtime_context()
        asyncio.create_task(execute_cron_job(ctx, job, trigger="manual"))
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)

    return web.json_response({"ok": True, "status": "started"})


async def pause_cron_job(server: WebServer, request: web.Request) -> web.Response:
    """POST /api/cron/jobs/{id}/pause — disable a job."""
    if not server.agent:
        return web.json_response({"error": "Agent not initialized"}, status=503)

    job_id = request.match_info.get("id", "")
    sm = server.agent.session_manager
    ok = await sm.update_cron_job(job_id, enabled=False, last_status="paused")
    if not ok:
        return web.json_response({"error": "Job not found"}, status=404)
    return web.json_response({"ok": True})


async def resume_cron_job(server: WebServer, request: web.Request) -> web.Response:
    """POST /api/cron/jobs/{id}/resume — re-enable a paused job."""
    if not server.agent:
        return web.json_response({"error": "Agent not initialized"}, status=503)

    job_id = request.match_info.get("id", "")
    sm = server.agent.session_manager
    job = await sm.load_cron_job(job_id)
    if not job:
        return web.json_response({"error": "Job not found"}, status=404)

    from captain_claw.cron import compute_next_run, to_utc_iso

    next_run = to_utc_iso(compute_next_run(job.schedule))
    await sm.update_cron_job(
        job_id, enabled=True, last_status="pending", next_run_at=next_run,
    )
    return web.json_response({"ok": True, "next_run_at": next_run})


async def update_cron_job_payload(server: WebServer, request: web.Request) -> web.Response:
    """PATCH /api/cron/jobs/{id} — update job payload."""
    if not server.agent:
        return web.json_response({"error": "Agent not initialized"}, status=503)

    job_id = request.match_info.get("id", "")
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON"}, status=400)

    sm = server.agent.session_manager
    new_payload = body.get("payload")
    if new_payload is None:
        return web.json_response({"error": "Missing payload"}, status=400)

    ok = await sm.update_cron_job(job_id, payload=new_payload)
    if not ok:
        return web.json_response({"error": "Job not found"}, status=404)
    return web.json_response({"ok": True})


async def delete_cron_job(server: WebServer, request: web.Request) -> web.Response:
    """DELETE /api/cron/jobs/{id} — remove a job."""
    if not server.agent:
        return web.json_response({"error": "Agent not initialized"}, status=503)

    job_id = request.match_info.get("id", "")
    sm = server.agent.session_manager
    ok = await sm.delete_cron_job(job_id)
    if not ok:
        return web.json_response({"error": "Job not found"}, status=404)
    return web.json_response({"ok": True})


async def get_cron_job_history(server: WebServer, request: web.Request) -> web.Response:
    """GET /api/cron/jobs/{id}/history — get job execution history."""
    if not server.agent:
        return web.json_response({"error": "Agent not initialized"}, status=503)

    job_id = request.match_info.get("id", "")
    sm = server.agent.session_manager
    job = await sm.load_cron_job(job_id)
    if not job:
        return web.json_response({"error": "Job not found"}, status=404)
    return web.json_response(
        {"chat_history": job.chat_history, "monitor_history": job.monitor_history},
        dumps=lambda obj: json.dumps(obj, default=str),
    )
