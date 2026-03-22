"""Cron job management tool — schedule, list, pause, resume, and remove recurring tasks.

Frozen tool (always available). Allows the agent to create and manage
scheduled tasks from natural conversation.
"""

from __future__ import annotations

import json
from typing import Any

from captain_claw.cron import (
    compute_next_run,
    parse_schedule_tokens,
    schedule_to_text,
    to_utc_iso,
)
from captain_claw.logging import get_logger
from captain_claw.tools.registry import Tool, ToolResult

log = get_logger(__name__)


class CronTool(Tool):
    """Create, list, pause, resume, and remove scheduled (cron) jobs.

    Use this tool whenever the user asks you to do something on a schedule,
    periodically, or at a specific recurring time.  Examples:

    • "Check my email every 30 minutes"
    • "Run a morning briefing daily at 09:00"
    • "Every Friday at 17:00, summarise the week"
    • "Stop the morning-briefing job"
    • "Show me all scheduled jobs"

    Schedule formats:
      every <N>m            — every N minutes   (e.g. every 15m)
      every <N>h            — every N hours     (e.g. every 2h)
      daily <HH:MM>         — once a day        (e.g. daily 09:00)
      weekly <day> <HH:MM>  — once a week       (e.g. weekly fri 17:00)
      in <N>d               — one-shot in N days  (e.g. in 3d)
      in <N>h               — one-shot in N hours (e.g. in 2h)
      in <N>m               — one-shot in N min   (e.g. in 30m)
      once <ISO-datetime>   — one-shot at exact time

    Days: mon, tue, wed, thu, fri, sat, sun (full names also accepted).
    One-shot jobs run once and then auto-disable.
    """

    name = "cron"
    description = (
        "Manage scheduled/recurring tasks (cron jobs).  Actions: "
        "create — schedule a new recurring task; "
        "list — show all scheduled jobs; "
        "pause — temporarily disable a job; "
        "resume — re-enable a paused job; "
        "remove — permanently delete a job; "
        "run — execute a job immediately (one-off trigger)."
    )
    timeout_seconds = 15.0
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["create", "list", "pause", "resume", "remove", "run"],
                "description": (
                    "Action to perform: create, list, pause, resume, remove, or run."
                ),
            },
            "schedule": {
                "type": "string",
                "description": (
                    "Schedule string (required for 'create').  "
                    "Recurring: 'every 15m', 'every 2h', 'daily 09:00', "
                    "'weekly fri 17:00'.  "
                    "One-shot: 'in 3d', 'in 2h', 'in 30m', "
                    "'once 2026-03-25T09:00:00'."
                ),
            },
            "task": {
                "type": "string",
                "description": (
                    "The prompt/instruction the agent should execute on each run "
                    "(required for 'create').  Write it as a complete instruction."
                ),
            },
            "job_id": {
                "type": "string",
                "description": (
                    "Job ID or #index (required for pause/resume/remove/run).  "
                    "Use 'list' first to find the ID."
                ),
            },
        },
        "required": ["action"],
    }

    # Injected by the agent during registration.
    _agent: Any = None

    async def execute(
        self,
        action: str,
        schedule: str | None = None,
        task: str | None = None,
        job_id: str | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        """Route to the appropriate cron action."""
        action = (action or "").strip().lower()
        try:
            if action == "create":
                return await self._create(schedule, task)
            if action == "list":
                return await self._list()
            if action == "pause":
                return await self._pause(job_id)
            if action == "resume":
                return await self._resume(job_id)
            if action == "remove":
                return await self._remove(job_id)
            if action == "run":
                return await self._run(job_id)
            return ToolResult(
                success=False,
                error=f"Unknown action '{action}'. Use: create, list, pause, resume, remove, run.",
            )
        except Exception as exc:
            log.error("Cron tool error", action=action, error=str(exc))
            return ToolResult(success=False, error=str(exc))

    # ── helpers ────────────────────────────────────────────────────

    def _get_session_manager(self):
        from captain_claw.session import get_session_manager
        return get_session_manager()

    def _get_session_id(self) -> str:
        """Return the current session ID from the agent."""
        if self._agent and self._agent.session:
            return str(self._agent.session.id)
        raise RuntimeError("No active session — cannot create cron job.")

    # ── actions ────────────────────────────────────────────────────

    async def _create(self, schedule_str: str | None, task: str | None) -> ToolResult:
        if not schedule_str:
            return ToolResult(
                success=False,
                error="Missing 'schedule'. Examples: 'every 15m', 'daily 09:00', 'weekly mon 10:00'.",
            )
        if not task:
            return ToolResult(
                success=False,
                error="Missing 'task' — the prompt/instruction to execute on each run.",
            )

        # Parse schedule string into tokens.
        tokens = schedule_str.strip().split()
        try:
            schedule_dict, _ = parse_schedule_tokens(tokens)
        except ValueError as exc:
            return ToolResult(success=False, error=f"Bad schedule: {exc}")

        next_run = compute_next_run(schedule_dict)
        session_id = self._get_session_id()

        sm = self._get_session_manager()
        job = await sm.create_cron_job(
            kind="prompt",
            payload={"text": task.strip()},
            schedule=schedule_dict,
            session_id=session_id,
            next_run_at=to_utc_iso(next_run),
            enabled=True,
        )

        human_sched = schedule_to_text(schedule_dict)
        return ToolResult(
            success=True,
            content=(
                f"Cron job created.\n"
                f"  ID:       {job.id}\n"
                f"  Schedule: {human_sched}\n"
                f"  Task:     {task.strip()}\n"
                f"  Next run: {job.next_run_at}\n"
                f"  Session:  {session_id}"
            ),
        )

    async def _list(self) -> ToolResult:
        sm = self._get_session_manager()
        jobs = await sm.list_cron_jobs(limit=50)

        if not jobs:
            return ToolResult(success=True, content="No scheduled jobs.")

        lines: list[str] = [f"Scheduled jobs ({len(jobs)}):"]
        for i, job in enumerate(jobs, 1):
            status = "✅ active" if job.enabled else "⏸ paused"
            sched_text = schedule_to_text(job.schedule)
            task_text = (job.payload.get("text") or json.dumps(job.payload))[:80]
            lines.append(
                f"\n  #{i}  {job.id[:12]}\n"
                f"      Schedule: {sched_text}  |  {status}\n"
                f"      Task:     {task_text}\n"
                f"      Last run: {job.last_run_at or 'never'}  |  Next: {job.next_run_at}\n"
                f"      Status:   {job.last_status}"
            )
        return ToolResult(success=True, content="\n".join(lines))

    async def _pause(self, job_id: str | None) -> ToolResult:
        if not job_id:
            return ToolResult(success=False, error="Missing 'job_id'. Use 'list' to find it.")

        sm = self._get_session_manager()
        job = await sm.select_cron_job(job_id)
        if not job:
            return ToolResult(success=False, error=f"Job '{job_id}' not found.")

        await sm.update_cron_job(job.id, enabled=False)
        return ToolResult(success=True, content=f"Job {job.id[:12]} paused.")

    async def _resume(self, job_id: str | None) -> ToolResult:
        if not job_id:
            return ToolResult(success=False, error="Missing 'job_id'. Use 'list' to find it.")

        sm = self._get_session_manager()
        job = await sm.select_cron_job(job_id)
        if not job:
            return ToolResult(success=False, error=f"Job '{job_id}' not found.")

        # Recompute next_run so it doesn't fire immediately for stale jobs.
        next_run = compute_next_run(job.schedule)
        await sm.update_cron_job(job.id, enabled=True, next_run_at=to_utc_iso(next_run))
        return ToolResult(
            success=True,
            content=f"Job {job.id[:12]} resumed. Next run: {to_utc_iso(next_run)}",
        )

    async def _remove(self, job_id: str | None) -> ToolResult:
        if not job_id:
            return ToolResult(success=False, error="Missing 'job_id'. Use 'list' to find it.")

        sm = self._get_session_manager()
        job = await sm.select_cron_job(job_id)
        if not job:
            return ToolResult(success=False, error=f"Job '{job_id}' not found.")

        deleted = await sm.delete_cron_job(job.id)
        if deleted:
            return ToolResult(success=True, content=f"Job {job.id[:12]} removed.")
        return ToolResult(success=False, error=f"Failed to remove job {job_id}.")

    async def _run(self, job_id: str | None) -> ToolResult:
        if not job_id:
            return ToolResult(success=False, error="Missing 'job_id'. Use 'list' to find it.")

        sm = self._get_session_manager()
        job = await sm.select_cron_job(job_id)
        if not job:
            return ToolResult(success=False, error=f"Job '{job_id}' not found.")

        # Trigger execution via the dispatch system.
        import asyncio
        from captain_claw.cron_dispatch import execute_cron_job

        # We need a RuntimeContext — get it from the agent.
        ctx = getattr(self._agent, "_runtime_context", None)
        if not ctx:
            return ToolResult(
                success=False,
                error="No runtime context available — cannot trigger job.",
            )

        asyncio.create_task(execute_cron_job(ctx, job, trigger="manual"))
        return ToolResult(
            success=True,
            content=f"Job {job.id[:12]} triggered (running in background).",
        )
