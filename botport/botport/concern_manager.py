"""Concern lifecycle management for BotPort."""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from botport.config import get_config
from botport.models import Concern, ConcernExchange, TERMINAL_STATES, _utcnow_iso
from botport.store import BotPortStore

log = logging.getLogger(__name__)


class ConcernManager:
    """Manages concern lifecycle: creation, assignment, results, timeouts."""

    def __init__(self, store: BotPortStore) -> None:
        self._concerns: dict[str, Concern] = {}  # active concerns by ID
        self._store = store
        self._timeout_task: asyncio.Task[None] | None = None

    def start_timeout_checker(self) -> None:
        """Start the background timeout checking loop."""
        if self._timeout_task is None or self._timeout_task.done():
            self._timeout_task = asyncio.create_task(self._timeout_loop())

    def stop_timeout_checker(self) -> None:
        if self._timeout_task and not self._timeout_task.done():
            self._timeout_task.cancel()

    async def create_concern(
        self,
        from_instance: str,
        task: str,
        context: dict[str, Any] | None = None,
        expertise_tags: list[str] | None = None,
        from_session: str = "",
    ) -> Concern:
        """Create a new concern in pending state."""
        cfg = get_config()
        timeout_seconds = cfg.concerns.idle_timeout_seconds

        now = _utcnow_iso()
        timeout_at = (
            datetime.now(timezone.utc) + timedelta(seconds=timeout_seconds)
        ).isoformat()

        concern = Concern(
            id=str(uuid.uuid4()),
            from_instance=from_instance,
            from_session=from_session,
            task=task,
            context=context or {},
            expertise_tags=expertise_tags or [],
            status="pending",
            created_at=now,
            updated_at=now,
            timeout_at=timeout_at,
        )

        # Record the initial request in the exchange.
        exchange = ConcernExchange(
            direction="request",
            content=task,
            from_instance=from_instance,
        )
        concern.messages.append(exchange)

        self._concerns[concern.id] = concern
        await self._store.save_concern(concern)
        await self._store.save_exchange(concern.id, exchange)

        log.debug("Concern created: %s from %s", concern.id[:8], from_instance)
        return concern

    async def assign_concern(
        self,
        concern_id: str,
        instance_id: str,
        instance_name: str = "",
    ) -> bool:
        """Assign a concern to a specific instance."""
        concern = self._concerns.get(concern_id)
        if not concern or concern.is_terminal:
            return False

        concern.assigned_instance = instance_id
        concern.status = "assigned"
        concern.touch()
        self._refresh_timeout(concern)

        await self._store.save_concern(concern)
        log.debug(
            "Concern %s assigned to %s (%s)",
            concern_id[:8], instance_name or instance_id, instance_id[:8],
        )
        return True

    async def mark_in_progress(self, concern_id: str, session_id: str = "") -> bool:
        """Mark concern as in_progress (CC-B started working)."""
        concern = self._concerns.get(concern_id)
        if not concern or concern.is_terminal:
            return False

        concern.status = "in_progress"
        if session_id:
            concern.assigned_session = session_id
        concern.touch()
        self._refresh_timeout(concern)

        await self._store.save_concern(concern)
        return True

    async def record_result(
        self,
        concern_id: str,
        response: str,
        metadata: dict[str, Any] | None = None,
        from_instance: str = "",
    ) -> bool:
        """Record a result from CC-B."""
        concern = self._concerns.get(concern_id)
        if not concern or concern.is_terminal:
            return False

        concern.status = "responded"
        concern.touch()
        self._refresh_timeout(concern)
        if metadata:
            concern.metadata.update(metadata)

        exchange = ConcernExchange(
            direction="response",
            content=response,
            from_instance=from_instance,
            metadata=metadata or {},
        )
        concern.messages.append(exchange)

        await self._store.save_concern(concern)
        await self._store.save_exchange(concern_id, exchange)

        log.debug("Concern %s got result (%d chars)", concern_id[:8], len(response))
        return True

    async def add_follow_up(
        self,
        concern_id: str,
        message: str,
        from_instance: str = "",
        additional_context: dict[str, Any] | None = None,
    ) -> bool:
        """Add a follow-up message to a concern."""
        concern = self._concerns.get(concern_id)
        if not concern or concern.is_terminal:
            return False

        cfg = get_config()
        follow_up_count = sum(1 for m in concern.messages if m.direction == "follow_up")
        if follow_up_count >= cfg.concerns.max_follow_ups:
            log.warning("Concern %s hit max follow-ups (%d)", concern_id[:8], cfg.concerns.max_follow_ups)
            return False

        concern.status = "in_progress"
        concern.touch()
        self._refresh_timeout(concern)

        exchange = ConcernExchange(
            direction="follow_up",
            content=message,
            from_instance=from_instance,
            metadata=additional_context or {},
        )
        concern.messages.append(exchange)

        await self._store.save_concern(concern)
        await self._store.save_exchange(concern_id, exchange)

        log.debug("Follow-up added to concern %s", concern_id[:8])
        return True

    async def add_context_request(
        self,
        concern_id: str,
        questions: list[str],
        from_instance: str = "",
    ) -> bool:
        """Record a context request from CC-B."""
        concern = self._concerns.get(concern_id)
        if not concern or concern.is_terminal:
            return False

        concern.touch()
        self._refresh_timeout(concern)

        exchange = ConcernExchange(
            direction="context_request",
            content="\n".join(questions),
            from_instance=from_instance,
        )
        concern.messages.append(exchange)

        await self._store.save_concern(concern)
        await self._store.save_exchange(concern_id, exchange)
        return True

    async def add_context_reply(
        self,
        concern_id: str,
        answers: dict[str, Any],
        from_instance: str = "",
    ) -> bool:
        """Record a context reply from CC-A."""
        concern = self._concerns.get(concern_id)
        if not concern or concern.is_terminal:
            return False

        concern.touch()
        self._refresh_timeout(concern)

        import json
        exchange = ConcernExchange(
            direction="context_reply",
            content=json.dumps(answers),
            from_instance=from_instance,
        )
        concern.messages.append(exchange)

        await self._store.save_concern(concern)
        await self._store.save_exchange(concern_id, exchange)
        return True

    async def close_concern(self, concern_id: str, reason: str = "closed") -> bool:
        """Close a concern (terminal state)."""
        concern = self._concerns.get(concern_id)
        if not concern or concern.is_terminal:
            return False

        concern.status = "closed"
        concern.touch()
        concern.metadata["close_reason"] = reason

        await self._store.save_concern(concern)
        log.debug("Concern %s closed: %s", concern_id[:8], reason)
        return True

    async def fail_concern(self, concern_id: str, reason: str = "failed") -> bool:
        """Mark a concern as failed (terminal state)."""
        concern = self._concerns.get(concern_id)
        if not concern or concern.is_terminal:
            return False

        concern.status = "failed"
        concern.touch()
        concern.metadata["fail_reason"] = reason

        await self._store.save_concern(concern)
        log.info("Concern %s failed: %s", concern_id[:8], reason)
        return True

    async def timeout_concern(self, concern_id: str) -> bool:
        """Mark a concern as timed out (terminal state)."""
        concern = self._concerns.get(concern_id)
        if not concern or concern.is_terminal:
            return False

        concern.status = "timeout"
        concern.touch()

        await self._store.save_concern(concern)
        log.warning("Concern %s timed out", concern_id[:8])
        return True

    def get_concern(self, concern_id: str) -> Concern | None:
        return self._concerns.get(concern_id)

    def get_active_concerns(self) -> list[Concern]:
        return [c for c in self._concerns.values() if c.is_active]

    def get_concerns_for_instance(self, instance_id: str) -> list[Concern]:
        """Get all active concerns assigned to an instance."""
        return [
            c for c in self._concerns.values()
            if c.is_active and c.assigned_instance == instance_id
        ]

    async def get_stats(self) -> dict[str, Any]:
        """Stats for the dashboard."""
        active = self.get_active_concerns()
        stored_stats = await self._store.get_stats()

        return {
            "active_count": len(active),
            "active_by_status": _count_by_status(active),
            **stored_stats,
        }

    def _refresh_timeout(self, concern: Concern) -> None:
        """Reset the idle timeout for a concern."""
        cfg = get_config()
        concern.timeout_at = (
            datetime.now(timezone.utc) + timedelta(seconds=cfg.concerns.idle_timeout_seconds)
        ).isoformat()

    async def _timeout_loop(self) -> None:
        """Background loop checking for timed-out concerns."""
        cfg = get_config()
        interval = cfg.concerns.timeout_check_interval_seconds
        while True:
            try:
                await asyncio.sleep(interval)
                await self._check_timeouts()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                log.error("Timeout checker error: %s", exc)

    async def _check_timeouts(self) -> list[str]:
        """Check and timeout idle concerns. Returns list of timed-out IDs."""
        now = datetime.now(timezone.utc)
        timed_out: list[str] = []

        for concern in list(self._concerns.values()):
            if concern.is_terminal or not concern.timeout_at:
                continue
            try:
                timeout_dt = datetime.fromisoformat(
                    concern.timeout_at.replace("Z", "+00:00")
                )
                if timeout_dt <= now:
                    await self.timeout_concern(concern.id)
                    timed_out.append(concern.id)
            except (ValueError, TypeError):
                continue

        return timed_out

    async def fail_concerns_for_instance(self, instance_id: str) -> list[str]:
        """Fail all active concerns assigned to a disconnected instance."""
        failed: list[str] = []
        for concern in self.get_concerns_for_instance(instance_id):
            await self.fail_concern(concern.id, reason=f"instance {instance_id} disconnected")
            failed.append(concern.id)
        return failed


def _count_by_status(concerns: list[Concern]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for c in concerns:
        counts[c.status] = counts.get(c.status, 0) + 1
    return counts
