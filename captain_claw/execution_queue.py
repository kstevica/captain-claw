"""Lane-based async execution queue primitives."""

import asyncio
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Literal, TypeVar

from captain_claw.logging import log

T = TypeVar("T")

QueueMode = Literal["steer", "followup", "collect", "steer-backlog", "interrupt", "queue"]
QueueDropPolicy = Literal["old", "new", "summarize"]
QueueDedupeMode = Literal["message-id", "prompt", "none"]


class CommandLane:
    """Built-in global command lanes."""

    MAIN = "main"
    CRON = "cron"
    SUBAGENT = "subagent"
    NESTED = "nested"
    AGENT_RUNTIME = "agent_runtime"


class CommandLaneClearedError(RuntimeError):
    """Raised when queued tasks are rejected after a lane clear."""

    def __init__(self, lane: str | None = None):
        message = f'Command lane "{lane}" cleared' if lane else "Command lane cleared"
        super().__init__(message)
        self.lane = lane or ""


@dataclass
class QueueEntry:
    task: Callable[[], Awaitable[object]]
    future: asyncio.Future[object]
    enqueued_at_ms: int
    warn_after_ms: int
    on_wait: Callable[[int, int], None] | None = None


@dataclass
class LaneState:
    lane: str
    queue: deque[QueueEntry] = field(default_factory=deque)
    active_task_ids: set[int] = field(default_factory=set)
    max_concurrent: int = 1
    draining: bool = False
    generation: int = 0


class CommandQueueManager:
    """In-process async lane queue with per-lane concurrency limits."""

    def __init__(self):
        self._lanes: dict[str, LaneState] = {}
        self._next_task_id = 1

    def _get_lane_state(self, lane: str) -> LaneState:
        existing = self._lanes.get(lane)
        if existing:
            return existing
        created = LaneState(lane=lane)
        self._lanes[lane] = created
        return created

    @staticmethod
    def _complete_task(state: LaneState, task_id: int, task_generation: int) -> bool:
        if task_generation != state.generation:
            return False
        state.active_task_ids.discard(task_id)
        return True

    def _schedule_drain(self, lane: str) -> None:
        state = self._get_lane_state(lane)
        if state.draining:
            return
        state.draining = True
        asyncio.create_task(self._drain_lane(lane))

    async def _run_entry(
        self,
        lane: str,
        state: LaneState,
        entry: QueueEntry,
        *,
        task_id: int,
        task_generation: int,
    ) -> None:
        started_ms = int(asyncio.get_running_loop().time() * 1000)
        try:
            result = await entry.task()
            completed_current_generation = self._complete_task(state, task_id, task_generation)
            if completed_current_generation:
                elapsed = int(asyncio.get_running_loop().time() * 1000) - started_ms
                log.debug(
                    "lane task complete",
                    lane=lane,
                    duration_ms=elapsed,
                    active=len(state.active_task_ids),
                    queued=len(state.queue),
                )
                self._schedule_drain(lane)
            if not entry.future.done():
                entry.future.set_result(result)
        except Exception as e:
            completed_current_generation = self._complete_task(state, task_id, task_generation)
            if completed_current_generation:
                elapsed = int(asyncio.get_running_loop().time() * 1000) - started_ms
                log.error("lane task failed", lane=lane, duration_ms=elapsed, error=str(e))
                self._schedule_drain(lane)
            if not entry.future.done():
                entry.future.set_exception(e)

    async def _drain_lane(self, lane: str) -> None:
        state = self._get_lane_state(lane)
        while state.queue and len(state.active_task_ids) < state.max_concurrent:
            entry = state.queue.popleft()
            waited_ms = int(asyncio.get_running_loop().time() * 1000) - entry.enqueued_at_ms
            if waited_ms >= entry.warn_after_ms:
                queued_ahead = len(state.queue)
                if entry.on_wait:
                    entry.on_wait(waited_ms, queued_ahead)
                log.warning(
                    "lane wait exceeded",
                    lane=lane,
                    waited_ms=waited_ms,
                    queued_ahead=queued_ahead,
                )

            task_id = self._next_task_id
            self._next_task_id += 1
            task_generation = state.generation
            state.active_task_ids.add(task_id)
            asyncio.create_task(
                self._run_entry(
                    lane,
                    state,
                    entry,
                    task_id=task_id,
                    task_generation=task_generation,
                )
            )
        state.draining = False

    async def enqueue_in_lane(
        self,
        lane: str,
        task: Callable[[], Awaitable[T]],
        *,
        warn_after_ms: int = 2_000,
        on_wait: Callable[[int, int], None] | None = None,
    ) -> T:
        cleaned = lane.strip() or CommandLane.MAIN
        state = self._get_lane_state(cleaned)
        loop = asyncio.get_running_loop()
        future: asyncio.Future[object] = loop.create_future()
        state.queue.append(
            QueueEntry(
                task=task,
                future=future,
                enqueued_at_ms=int(loop.time() * 1000),
                warn_after_ms=max(0, int(warn_after_ms)),
                on_wait=on_wait,
            )
        )
        self._schedule_drain(cleaned)
        result = await future
        return result  # type: ignore[return-value]

    async def enqueue(
        self,
        task: Callable[[], Awaitable[T]],
        *,
        warn_after_ms: int = 2_000,
        on_wait: Callable[[int, int], None] | None = None,
    ) -> T:
        return await self.enqueue_in_lane(
            CommandLane.MAIN,
            task,
            warn_after_ms=warn_after_ms,
            on_wait=on_wait,
        )

    def set_lane_concurrency(self, lane: str, max_concurrent: int) -> None:
        cleaned = lane.strip() or CommandLane.MAIN
        state = self._get_lane_state(cleaned)
        state.max_concurrent = max(1, int(max_concurrent))
        self._schedule_drain(cleaned)

    def get_queue_size(self, lane: str = CommandLane.MAIN) -> int:
        cleaned = lane.strip() or CommandLane.MAIN
        state = self._lanes.get(cleaned)
        if not state:
            return 0
        return len(state.queue) + len(state.active_task_ids)

    def get_total_queue_size(self) -> int:
        total = 0
        for state in self._lanes.values():
            total += len(state.queue) + len(state.active_task_ids)
        return total

    def clear_lane(self, lane: str = CommandLane.MAIN) -> int:
        cleaned = lane.strip() or CommandLane.MAIN
        state = self._lanes.get(cleaned)
        if not state:
            return 0
        removed = len(state.queue)
        while state.queue:
            entry = state.queue.popleft()
            if not entry.future.done():
                entry.future.set_exception(CommandLaneClearedError(cleaned))
        return removed

    def reset_all_lanes(self) -> None:
        lanes_to_drain: list[str] = []
        for state in self._lanes.values():
            state.generation += 1
            state.active_task_ids.clear()
            state.draining = False
            if state.queue:
                lanes_to_drain.append(state.lane)
        for lane in lanes_to_drain:
            self._schedule_drain(lane)

    def get_active_task_count(self) -> int:
        total = 0
        for state in self._lanes.values():
            total += len(state.active_task_ids)
        return total

    async def wait_for_active_tasks(self, timeout_ms: int) -> dict[str, bool]:
        poll_interval = 0.05
        deadline = asyncio.get_running_loop().time() + (max(0, timeout_ms) / 1000.0)
        active_at_start: set[int] = set()
        for state in self._lanes.values():
            active_at_start.update(state.active_task_ids)

        while True:
            if not active_at_start:
                return {"drained": True}

            has_pending = False
            for state in self._lanes.values():
                for task_id in state.active_task_ids:
                    if task_id in active_at_start:
                        has_pending = True
                        break
                if has_pending:
                    break

            if not has_pending:
                return {"drained": True}
            if asyncio.get_running_loop().time() >= deadline:
                return {"drained": False}
            await asyncio.sleep(poll_interval)


@dataclass
class QueueSettings:
    """Queue behavior settings."""

    mode: QueueMode = "collect"
    debounce_ms: int = 1_000
    cap: int = 20
    drop_policy: QueueDropPolicy = "summarize"


@dataclass
class FollowupRun:
    """Follow-up run payload."""

    prompt: str
    enqueued_at_ms: int
    message_id: str = ""
    summary_line: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class FollowupQueueState:
    """Per-key follow-up queue state."""

    items: deque[FollowupRun] = field(default_factory=deque)
    draining: bool = False
    last_enqueued_at_ms: int = 0
    mode: QueueMode = "collect"
    debounce_ms: int = 1_000
    cap: int = 20
    drop_policy: QueueDropPolicy = "summarize"
    dropped_count: int = 0
    summary_lines: list[str] = field(default_factory=list)
    last_run: FollowupRun | None = None


class FollowupQueueManager:
    """Queue manager for deferred follow-up prompts while a session is busy."""

    def __init__(self):
        self._queues: dict[str, FollowupQueueState] = {}

    def _get_queue(self, key: str, settings: QueueSettings) -> FollowupQueueState:
        existing = self._queues.get(key)
        if existing:
            existing.mode = settings.mode
            existing.debounce_ms = max(0, int(settings.debounce_ms))
            existing.cap = max(1, int(settings.cap))
            existing.drop_policy = settings.drop_policy
            return existing
        created = FollowupQueueState(
            mode=settings.mode,
            debounce_ms=max(0, int(settings.debounce_ms)),
            cap=max(1, int(settings.cap)),
            drop_policy=settings.drop_policy,
        )
        self._queues[key] = created
        return created

    @staticmethod
    def _dedupe_run(run: FollowupRun, queue: FollowupQueueState, dedupe_mode: QueueDedupeMode) -> bool:
        if dedupe_mode == "none":
            return False
        if dedupe_mode == "message-id" and run.message_id:
            return any(item.message_id and item.message_id == run.message_id for item in queue.items)
        if dedupe_mode == "prompt":
            return any(item.prompt.strip() == run.prompt.strip() for item in queue.items)
        return False

    @staticmethod
    def _apply_drop_policy(queue: FollowupQueueState, incoming: FollowupRun) -> bool:
        if len(queue.items) < queue.cap:
            return True
        if queue.drop_policy == "new":
            return False
        dropped = queue.items.popleft()
        if queue.drop_policy == "summarize":
            queue.dropped_count += 1
            summary = dropped.summary_line.strip() or dropped.prompt.strip()
            if summary:
                queue.summary_lines.append(summary)
        return True

    def enqueue_followup(
        self,
        key: str,
        run: FollowupRun,
        settings: QueueSettings,
        *,
        dedupe_mode: QueueDedupeMode = "message-id",
    ) -> bool:
        cleaned = key.strip()
        if not cleaned:
            return False
        queue = self._get_queue(cleaned, settings)
        if self._dedupe_run(run, queue, dedupe_mode):
            return False

        queue.last_enqueued_at_ms = int(asyncio.get_running_loop().time() * 1000)
        queue.last_run = run

        # Steer mode keeps only the freshest queued prompt while the session is busy.
        if queue.mode == "steer":
            queue.items.clear()
            queue.items.append(run)
            return True

        # Interrupt mode drops queued backlog and prioritizes the latest follow-up.
        if queue.mode == "interrupt":
            queue.items.clear()
            queue.items.append(run)
            return True

        if not self._apply_drop_policy(queue, run):
            return False
        queue.items.append(run)
        return True

    def get_queue_depth(self, key: str) -> int:
        cleaned = key.strip()
        if not cleaned:
            return 0
        queue = self._queues.get(cleaned)
        if not queue:
            return 0
        return len(queue.items)

    def clear_queue(self, key: str) -> int:
        cleaned = key.strip()
        if not cleaned:
            return 0
        queue = self._queues.get(cleaned)
        if not queue:
            return 0
        cleared = len(queue.items) + queue.dropped_count
        queue.items.clear()
        queue.dropped_count = 0
        queue.summary_lines = []
        queue.last_run = None
        queue.last_enqueued_at_ms = 0
        self._queues.pop(cleaned, None)
        return cleared

    @staticmethod
    async def _wait_for_debounce(queue: FollowupQueueState) -> None:
        if queue.debounce_ms <= 0:
            return
        loop = asyncio.get_running_loop()
        while True:
            now_ms = int(loop.time() * 1000)
            elapsed = now_ms - queue.last_enqueued_at_ms
            remaining = queue.debounce_ms - elapsed
            if remaining <= 0:
                return
            await asyncio.sleep(min(0.25, remaining / 1000.0))

    @staticmethod
    def _build_summary_prompt(queue: FollowupQueueState) -> str:
        if queue.dropped_count <= 0 or not queue.summary_lines:
            return ""
        lines = [
            "[Queue summary]",
            f"{queue.dropped_count} queued follow-ups were summarized due to queue cap.",
        ]
        for idx, item in enumerate(queue.summary_lines[-10:], start=1):
            lines.append(f"{idx}. {item}")
        return "\n".join(lines)

    @staticmethod
    def _build_collect_prompt(queue: FollowupQueueState, items: list[FollowupRun]) -> str:
        lines = ["[Queued follow-ups while session was busy]"]
        summary = FollowupQueueManager._build_summary_prompt(queue)
        if summary:
            lines.append(summary)
        for idx, item in enumerate(items, start=1):
            lines.append(f"---\nQueued #{idx}\n{item.prompt}")
        return "\n".join(lines)

    def schedule_drain(
        self,
        key: str,
        run_followup: Callable[[FollowupRun], Awaitable[None]],
        *,
        wait_until_ready: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        cleaned = key.strip()
        queue = self._queues.get(cleaned)
        if not cleaned or not queue or queue.draining:
            return
        queue.draining = True

        async def _drain() -> None:
            try:
                while queue.items or queue.dropped_count > 0:
                    await self._wait_for_debounce(queue)
                    if wait_until_ready is not None:
                        await wait_until_ready()
                    if queue.mode == "collect":
                        items = list(queue.items)
                        queue.items.clear()
                        if not items:
                            break
                        template = items[-1]
                        collected = FollowupRun(
                            prompt=self._build_collect_prompt(queue, items),
                            enqueued_at_ms=int(asyncio.get_running_loop().time() * 1000),
                            message_id="",
                            summary_line=template.summary_line,
                            metadata=dict(template.metadata),
                        )
                        await run_followup(collected)
                        queue.dropped_count = 0
                        queue.summary_lines = []
                        continue

                    summary_prompt = self._build_summary_prompt(queue)
                    if summary_prompt:
                        template = queue.last_run or (queue.items[0] if queue.items else None)
                        if template is not None:
                            summary_run = FollowupRun(
                                prompt=summary_prompt,
                                enqueued_at_ms=int(asyncio.get_running_loop().time() * 1000),
                                message_id="",
                                summary_line="",
                                metadata=dict(template.metadata),
                            )
                            await run_followup(summary_run)
                        queue.dropped_count = 0
                        queue.summary_lines = []

                    if not queue.items:
                        break
                    next_run = queue.items.popleft()
                    await run_followup(next_run)
            except Exception as e:
                queue.last_enqueued_at_ms = int(asyncio.get_running_loop().time() * 1000)
                log.error("followup queue drain failed", key=cleaned, error=str(e))
            finally:
                queue.draining = False
                if not queue.items and queue.dropped_count <= 0:
                    self._queues.pop(cleaned, None)
                else:
                    self.schedule_drain(cleaned, run_followup, wait_until_ready=wait_until_ready)

        asyncio.create_task(_drain())


def resolve_session_lane(key: str) -> str:
    cleaned = key.strip() if key else ""
    if not cleaned:
        return "session:default"
    if cleaned.startswith("session:"):
        return cleaned
    return f"session:{cleaned}"


def resolve_global_lane(lane: str | None = None) -> str:
    cleaned = lane.strip() if lane else ""
    return cleaned or CommandLane.MAIN


def normalize_queue_mode(raw: str | None) -> QueueMode | None:
    if not raw:
        return None
    cleaned = raw.strip().lower()
    if cleaned in {"queue", "queued", "steer", "steering"}:
        return "steer"
    if cleaned in {"interrupt", "interrupts", "abort"}:
        return "interrupt"
    if cleaned in {"followup", "follow-ups", "followups"}:
        return "followup"
    if cleaned in {"collect", "coalesce"}:
        return "collect"
    if cleaned in {"steer+backlog", "steer-backlog", "steer_backlog"}:
        return "steer-backlog"
    return None


def normalize_queue_drop_policy(raw: str | None) -> QueueDropPolicy | None:
    if not raw:
        return None
    cleaned = raw.strip().lower()
    if cleaned in {"old", "oldest"}:
        return "old"
    if cleaned in {"new", "newest"}:
        return "new"
    if cleaned in {"summarize", "summary"}:
        return "summarize"
    return None
