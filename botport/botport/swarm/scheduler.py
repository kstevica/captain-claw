"""Background scheduler for periodic swarm tasks."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from botport.swarm.models import SwarmAuditEntry, _utcnow_iso

if TYPE_CHECKING:
    from botport.server import BotPortServer

log = logging.getLogger(__name__)

SCHEDULER_POLL_INTERVAL = 30  # seconds


def _parse_cron_field(field: str, min_val: int, max_val: int) -> set[int]:
    """Parse a single cron field (e.g. '*/5', '1,3,5', '1-10', '*')."""
    values: set[int] = set()

    for part in field.split(","):
        part = part.strip()
        if part == "*":
            values.update(range(min_val, max_val + 1))
        elif part.startswith("*/"):
            step = int(part[2:])
            values.update(range(min_val, max_val + 1, step))
        elif "-" in part:
            start, end = part.split("-", 1)
            values.update(range(int(start), int(end) + 1))
        else:
            values.add(int(part))

    return values


def cron_matches(cron_expression: str, dt: datetime) -> bool:
    """Check if a datetime matches a cron expression (minute hour dom month dow).

    Supports: *, */N, N, N-M, N,M,O
    """
    parts = cron_expression.strip().split()
    if len(parts) != 5:
        return False

    try:
        minutes = _parse_cron_field(parts[0], 0, 59)
        hours = _parse_cron_field(parts[1], 0, 23)
        doms = _parse_cron_field(parts[2], 1, 31)
        months = _parse_cron_field(parts[3], 1, 12)
        dows = _parse_cron_field(parts[4], 0, 6)
    except (ValueError, IndexError):
        return False

    return (
        dt.minute in minutes
        and dt.hour in hours
        and dt.day in doms
        and dt.month in months
        and dt.weekday() in dows  # Python: 0=Monday; cron: 0=Sunday
        # Adjust: cron uses 0=Sunday, Python uses 0=Monday
    )


def cron_matches_utc(cron_expression: str) -> bool:
    """Check if the current UTC time matches a cron expression."""
    now = datetime.now(timezone.utc)
    # For cron DOW: 0=Sunday. Python weekday(): 0=Monday.
    # We'll treat the DOW field with Sunday=0 convention.
    parts = cron_expression.strip().split()
    if len(parts) != 5:
        return False

    try:
        minutes = _parse_cron_field(parts[0], 0, 59)
        hours = _parse_cron_field(parts[1], 0, 23)
        doms = _parse_cron_field(parts[2], 1, 31)
        months = _parse_cron_field(parts[3], 1, 12)
        dows = _parse_cron_field(parts[4], 0, 6)
    except (ValueError, IndexError):
        return False

    # Convert Python weekday (0=Mon) to cron (0=Sun): (weekday + 1) % 7
    cron_dow = (now.weekday() + 1) % 7

    return (
        now.minute in minutes
        and now.hour in hours
        and now.day in doms
        and now.month in months
        and cron_dow in dows
    )


def next_cron_time(cron_expression: str, from_dt: datetime | None = None) -> str:
    """Calculate the next matching time for a cron expression. Returns ISO string."""
    from datetime import timedelta

    dt = from_dt or datetime.now(timezone.utc)
    # Move to next minute.
    dt = dt.replace(second=0, microsecond=0) + timedelta(minutes=1)

    # Brute-force search (max ~525960 minutes = 1 year).
    for _ in range(525960):
        parts = cron_expression.strip().split()
        if len(parts) != 5:
            return ""

        try:
            minutes = _parse_cron_field(parts[0], 0, 59)
            hours = _parse_cron_field(parts[1], 0, 23)
            doms = _parse_cron_field(parts[2], 1, 31)
            months = _parse_cron_field(parts[3], 1, 12)
            dows = _parse_cron_field(parts[4], 0, 6)
        except (ValueError, IndexError):
            return ""

        cron_dow = (dt.weekday() + 1) % 7

        if (dt.minute in minutes and dt.hour in hours
                and dt.day in doms and dt.month in months
                and cron_dow in dows):
            return dt.isoformat()

        dt += timedelta(minutes=1)

    return ""


class SwarmScheduler:
    """Background loop that triggers periodic tasks and template schedules."""

    def __init__(self, server: BotPortServer) -> None:
        self._server = server
        self._task: asyncio.Task[None] | None = None
        self._running = False

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._running = True
            self._task = asyncio.create_task(self._loop())
            log.info("Swarm scheduler started")

    def stop(self) -> None:
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()

    async def wait_stopped(self) -> None:
        if self._task:
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _loop(self) -> None:
        while self._running:
            try:
                await asyncio.sleep(SCHEDULER_POLL_INTERVAL)
                await self._tick()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                log.error("Scheduler error: %s", exc, exc_info=True)

    async def _tick(self) -> None:
        """Check for due periodic tasks and re-queue them."""
        store = self._server.swarm_store
        now = _utcnow_iso()

        due_tasks = await store.list_due_periodic_tasks(now)
        for task in due_tasks:
            await self._requeue_periodic_task(task)

    async def _requeue_periodic_task(self, task) -> None:
        """Reset a periodic task to queued for re-execution."""
        store = self._server.swarm_store

        # Only re-queue if the parent swarm is running.
        swarm = await store.get_swarm(task.swarm_id)
        if not swarm or swarm.status != "running":
            return

        # Reset task for re-execution.
        task.status = "queued"
        task.concern_id = ""
        task.started_at = ""
        task.completed_at = ""
        task.error_message = ""
        task.retry_count = 0
        task.output_data = {}
        task.metadata.pop("timeout_warned", None)
        task.metadata.pop("retry_at", None)
        task.metadata.pop("policy_applied", None)

        # Calculate next run time.
        if task.cron_expression:
            task.next_run_at = next_cron_time(task.cron_expression)

        task.touch()
        await store.save_task(task)

        await store.add_audit_entry(SwarmAuditEntry(
            swarm_id=task.swarm_id,
            task_id=task.id,
            event_type="periodic_task_requeued",
            details={
                "cron": task.cron_expression,
                "next_run_at": task.next_run_at,
            },
            actor="scheduler",
            created_at=_utcnow_iso(),
        ))

        log.info(
            "Periodic task %s requeued (next: %s)",
            task.name or task.id[:8], task.next_run_at or "unknown",
        )
