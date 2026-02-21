"""Standalone DAG scheduler for parallel session orchestration.

Provides a task graph with topological ordering, concurrency control
(traffic lights), timeout/retry management, and dependency tracking.
Algorithms adapted from AgentPipelineMixin but decoupled for use by
SessionOrchestrator.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Task statuses
# ---------------------------------------------------------------------------

PENDING = "pending"
QUEUED = "queued"
RUNNING = "running"
COMPLETED = "completed"
FAILED = "failed"
PAUSED = "paused"
EDITING = "editing"
TIMEOUT_WARNING = "timeout_warning"

_TERMINAL_STATES = {COMPLETED, FAILED}
_ACTIVATABLE_STATES = {PENDING, QUEUED}
_HOLD_STATES = {PAUSED, EDITING}

# Grace period (seconds) between timeout warning and automatic restart.
TIMEOUT_GRACE_SECONDS = 60.0


# ---------------------------------------------------------------------------
# OrchestratorTask dataclass
# ---------------------------------------------------------------------------


@dataclass
class OrchestratorTask:
    """Single unit of work in the orchestration DAG."""

    id: str
    title: str
    description: str = ""
    depends_on: list[str] = field(default_factory=list)
    session_id: str = ""
    session_name: str = ""
    status: str = PENDING
    result: dict[str, Any] | None = None
    timeout_seconds: float = 300.0
    max_retries: int = 2
    retries: int = 0
    started_at: float = 0.0
    completed_at: float = 0.0
    error: str = ""
    editing: bool = False
    original_description: str = ""
    # Timeout warning grace period tracking.
    timeout_warning_at: float = 0.0  # monotonic timestamp when warning started
    # Per-task configuration (set during preview / workflow load).
    model_id: str = ""
    skills: list[str] = field(default_factory=list)
    use_existing_session: bool = False

    def is_terminal(self) -> bool:
        return self.status in _TERMINAL_STATES


# ---------------------------------------------------------------------------
# TaskGraph
# ---------------------------------------------------------------------------


class TaskGraph:
    """DAG of OrchestratorTask nodes with concurrency control.

    Manages topological ordering, activation gates (traffic lights),
    timeout monitoring, and retry logic.
    """

    def __init__(self, max_parallel: int = 5):
        self.max_parallel = max(1, int(max_parallel))
        self._tasks: dict[str, OrchestratorTask] = {}
        self._order: list[str] = []
        # Pre-computed sets refreshed after mutations.
        self._ready: set[str] = set()
        self._blocked: set[str] = set()
        self._running: set[str] = set()
        self._completed: set[str] = set()
        self._failed: set[str] = set()

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def add_task(self, task: OrchestratorTask) -> None:
        """Add a task to the graph (idempotent by id)."""
        self._tasks[task.id] = task

    def add_tasks(self, tasks: list[OrchestratorTask]) -> None:
        for task in tasks:
            self.add_task(task)

    def get_task(self, task_id: str) -> OrchestratorTask | None:
        return self._tasks.get(task_id)

    @property
    def tasks(self) -> dict[str, OrchestratorTask]:
        return dict(self._tasks)

    @property
    def task_count(self) -> int:
        return len(self._tasks)

    # ------------------------------------------------------------------
    # Topological sort (Kahn's algorithm)
    # ------------------------------------------------------------------

    @staticmethod
    def _topological_sort(
        task_ids: list[str],
        dependencies: dict[str, list[str]],
    ) -> list[str]:
        """Return stable topological order via Kahn's algorithm."""
        dep_map: dict[str, set[str]] = {tid: set() for tid in task_ids}
        reverse_map: dict[str, set[str]] = {tid: set() for tid in task_ids}
        id_set = set(task_ids)
        for tid in task_ids:
            for dep_id in dependencies.get(tid, []):
                if dep_id not in id_set:
                    continue
                dep_map[tid].add(dep_id)
                reverse_map[dep_id].add(tid)

        # Start with zero-dependency tasks, sorted for stability.
        ready = sorted(tid for tid, deps in dep_map.items() if not deps)
        order: list[str] = []
        while ready:
            current = ready.pop(0)
            order.append(current)
            for dependent in sorted(reverse_map.get(current, set())):
                pending = dep_map.get(dependent)
                if not pending:
                    continue
                pending.discard(current)
                if not pending and dependent not in order and dependent not in ready:
                    ready.append(dependent)
            ready.sort()

        # Append unreachable nodes (cyclic leftovers) for robustness.
        if len(order) < len(task_ids):
            remaining = sorted(tid for tid in task_ids if tid not in order)
            order.extend(remaining)
        return order

    # ------------------------------------------------------------------
    # State refresh
    # ------------------------------------------------------------------

    def refresh(self) -> None:
        """Recompute topological order and ready/blocked/active sets."""
        task_ids = list(self._tasks.keys())
        dep_map: dict[str, list[str]] = {
            tid: list(task.depends_on) for tid, task in self._tasks.items()
        }
        self._order = self._topological_sort(task_ids, dep_map)

        self._ready.clear()
        self._blocked.clear()
        self._running.clear()
        self._completed.clear()
        self._failed.clear()
        self._held: set[str] = set()

        for tid, task in self._tasks.items():
            if task.status == COMPLETED:
                self._completed.add(tid)
            elif task.status == FAILED:
                self._failed.add(tid)
            elif task.status == RUNNING or task.status == TIMEOUT_WARNING:
                # TIMEOUT_WARNING tasks are still running; they count
                # against the parallel slot limit.
                self._running.add(tid)
            elif task.status in _HOLD_STATES:
                self._held.add(tid)
            elif task.status in _ACTIVATABLE_STATES:
                deps_met = all(
                    self._tasks.get(dep_id, OrchestratorTask(id="", title="")).status == COMPLETED
                    for dep_id in task.depends_on
                    if dep_id in self._tasks
                )
                deps_failed = any(
                    self._tasks.get(dep_id, OrchestratorTask(id="", title="")).status == FAILED
                    for dep_id in task.depends_on
                    if dep_id in self._tasks
                )
                if deps_failed:
                    # Cascade failure: dependency failed, mark this task failed too.
                    task.status = FAILED
                    task.error = "dependency_failed"
                    task.completed_at = time.monotonic()
                    self._failed.add(tid)
                elif deps_met:
                    self._ready.add(tid)
                else:
                    self._blocked.add(tid)

    # ------------------------------------------------------------------
    # Traffic light: activation gate
    # ------------------------------------------------------------------

    def activate_next(self) -> list[OrchestratorTask]:
        """Move ready tasks to running state up to max_parallel limit.

        Returns the list of newly activated tasks.
        """
        self.refresh()
        available_slots = max(0, self.max_parallel - len(self._running))
        if available_slots <= 0:
            return []

        # Activate in topological order for deterministic behavior.
        ready_ordered = [tid for tid in self._order if tid in self._ready]
        activated: list[OrchestratorTask] = []
        for tid in ready_ordered:
            if available_slots <= 0:
                break
            task = self._tasks.get(tid)
            if task is None:
                continue
            if task.editing:
                continue  # Skip tasks in edit mode
            task.status = RUNNING
            task.started_at = time.monotonic()
            self._running.add(tid)
            self._ready.discard(tid)
            activated.append(task)
            available_slots -= 1

        return activated

    # ------------------------------------------------------------------
    # Task completion / failure
    # ------------------------------------------------------------------

    def complete_task(
        self,
        task_id: str,
        result: dict[str, Any] | None = None,
    ) -> None:
        """Mark task as completed.

        Does NOT auto-activate dependents — the execution loop in
        SessionOrchestrator calls activate_next() after each poll cycle
        to avoid race conditions with concurrent workers.
        """
        task = self._tasks.get(task_id)
        if task is None:
            return
        task.status = COMPLETED
        task.result = result or {}
        task.completed_at = time.monotonic()
        task.error = ""

    def fail_task(
        self,
        task_id: str,
        error: str = "",
    ) -> None:
        """Mark task as failed (or schedule retry).

        Does NOT auto-activate — the execution loop handles activation.
        """
        task = self._tasks.get(task_id)
        if task is None:
            return

        if task.retries < task.max_retries:
            # Retry: reset to pending for re-activation.
            task.retries += 1
            task.status = PENDING
            task.started_at = 0.0
            task.error = f"retry_{task.retries}: {error}"
            return

        task.status = FAILED
        task.error = error or "max_retries_exceeded"
        task.completed_at = time.monotonic()

    # ------------------------------------------------------------------
    # Task control: pause / resume / edit / restart
    # ------------------------------------------------------------------

    def pause_task(self, task_id: str) -> bool:
        """Pause a running task.  Returns True if status was changed."""
        task = self._tasks.get(task_id)
        if task is None or task.status != RUNNING:
            return False
        task.status = PAUSED
        return True

    def resume_task(self, task_id: str) -> bool:
        """Resume a paused or editing task back to PENDING."""
        task = self._tasks.get(task_id)
        if task is None or task.status not in _HOLD_STATES:
            return False
        task.status = PENDING
        task.editing = False
        task.started_at = 0.0
        task.error = ""
        return True

    def edit_task(self, task_id: str) -> bool:
        """Put a task into editing mode.

        Works for PENDING, PAUSED, FAILED, COMPLETED, QUEUED states.
        """
        task = self._tasks.get(task_id)
        if task is None:
            return False
        if task.status not in {PENDING, PAUSED, FAILED, COMPLETED, QUEUED}:
            return False
        task.editing = True
        task.status = EDITING
        task.original_description = task.description
        return True

    def update_task_description(self, task_id: str, new_description: str) -> bool:
        """Update a task's instructions (description)."""
        task = self._tasks.get(task_id)
        if task is None:
            return False
        task.description = new_description
        return True

    def restart_task(self, task_id: str) -> bool:
        """Reset a terminal / paused / editing task to PENDING."""
        task = self._tasks.get(task_id)
        if task is None:
            return False
        if task.status not in {FAILED, COMPLETED, PAUSED, EDITING}:
            return False
        task.status = PENDING
        task.result = None
        task.error = ""
        task.started_at = 0.0
        task.completed_at = 0.0
        task.editing = False
        return True

    # ------------------------------------------------------------------
    # Timeout tick
    # ------------------------------------------------------------------

    def tick_timeouts(self) -> dict[str, Any]:
        """Check running tasks for timeout, apply warning → grace → restart flow.

        Instead of immediately retrying/failing on timeout, tasks enter a
        60-second grace period (TIMEOUT_WARNING). The user can postpone the
        restart (resetting the timer for another timeout_seconds). If the
        grace period expires without postponement, the task is restarted.

        Returns dict with keys:
            - ``warned``: tasks that just entered the warning phase
            - ``restarted``: tasks whose grace period expired → restarted
            - ``failed``: tasks that exhausted retries
            - ``countdown``: list of ``{task_id, remaining_seconds}`` for
              tasks currently in the warning phase
        """
        now = time.monotonic()
        warned: list[str] = []
        restarted: list[str] = []
        failed: list[str] = []
        countdown: list[dict[str, Any]] = []

        for tid in list(self._running):
            task = self._tasks.get(tid)
            if task is None:
                continue

            # ── Phase 2: task already in warning → check grace period ──
            if task.status == TIMEOUT_WARNING:
                if task.timeout_warning_at <= 0:
                    task.timeout_warning_at = now
                grace_elapsed = now - task.timeout_warning_at
                remaining = max(0.0, TIMEOUT_GRACE_SECONDS - grace_elapsed)

                if remaining > 0:
                    # Still within grace period; report countdown.
                    countdown.append({
                        "task_id": tid,
                        "remaining_seconds": round(remaining),
                    })
                    continue

                # Grace period expired → restart or fail.
                if task.retries < task.max_retries:
                    task.retries += 1
                    task.status = PENDING
                    task.started_at = 0.0
                    task.timeout_warning_at = 0.0
                    task.error = f"timeout_restart_{task.retries}"
                    restarted.append(tid)
                else:
                    task.status = FAILED
                    task.error = "timeout_exhausted"
                    task.completed_at = now
                    task.timeout_warning_at = 0.0
                    failed.append(tid)
                continue

            # ── Phase 1: running task → check if timeout reached ──
            if task.status != RUNNING:
                continue
            if task.started_at <= 0:
                continue
            elapsed = now - task.started_at
            if elapsed < task.timeout_seconds:
                continue

            # Timeout reached → enter warning phase.
            task.status = TIMEOUT_WARNING
            task.timeout_warning_at = now
            warned.append(tid)
            countdown.append({
                "task_id": tid,
                "remaining_seconds": round(TIMEOUT_GRACE_SECONDS),
            })

        if warned or restarted or failed:
            self.refresh()

        return {
            "warned": warned,
            "restarted": restarted,
            "failed": failed,
            "countdown": countdown,
        }

    def postpone_task(self, task_id: str) -> bool:
        """Postpone a timeout warning, granting another full timeout period.

        Resets the task back to RUNNING with a fresh started_at timestamp,
        so the next timeout check starts from zero.

        Returns True if the task was successfully postponed.
        """
        task = self._tasks.get(task_id)
        if task is None:
            return False
        if task.status != TIMEOUT_WARNING:
            return False
        task.status = RUNNING
        task.started_at = time.monotonic()
        task.timeout_warning_at = 0.0
        task.error = ""
        return True

    # ------------------------------------------------------------------
    # Graph-level queries
    # ------------------------------------------------------------------

    @property
    def is_complete(self) -> bool:
        """All tasks are in a terminal state."""
        return all(task.is_terminal() for task in self._tasks.values()) if self._tasks else True

    @property
    def has_failures(self) -> bool:
        return bool(self._failed)

    @property
    def running_count(self) -> int:
        return len(self._running)

    @property
    def ready_count(self) -> int:
        return len(self._ready)

    def get_results(self) -> dict[str, dict[str, Any]]:
        """Collect results from all tasks."""
        results: dict[str, dict[str, Any]] = {}
        for tid, task in self._tasks.items():
            results[tid] = {
                "title": task.title,
                "description": task.description,
                "status": task.status,
                "result": task.result,
                "error": task.error,
                "retries": task.retries,
                "editing": task.editing,
            }
        return results

    def get_summary(self) -> dict[str, Any]:
        """Compact graph status summary."""
        self.refresh()
        held = getattr(self, "_held", set())
        return {
            "total": len(self._tasks),
            "completed": len(self._completed),
            "failed": len(self._failed),
            "running": len(self._running),
            "ready": len(self._ready),
            "blocked": len(self._blocked),
            "paused": len(held),
            "is_complete": self.is_complete,
            "has_failures": self.has_failures,
        }
