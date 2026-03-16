"""Swarm execution engine — orchestrates DAG tasks as BotPort concerns."""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any

from botport.swarm.dag import get_all_dependents, get_predecessors, get_ready_tasks
from botport.swarm.models import (
    SWARM_TERMINAL_STATES,
    TASK_TERMINAL_STATES,
    SwarmAuditEntry,
    SwarmCheckpoint,
    SwarmTask,
    _utcnow_iso,
)

if TYPE_CHECKING:
    from botport.server import BotPortServer

log = logging.getLogger(__name__)

ENGINE_POLL_INTERVAL = 2  # seconds
CHECKPOINT_THROTTLE_SECONDS = 30  # min interval between auto-checkpoints


class SwarmEngine:
    """Orchestration loop that advances swarm DAGs by creating concerns."""

    def __init__(self, server: BotPortServer) -> None:
        self._server = server
        self._engine_task: asyncio.Task[None] | None = None
        self._running = False
        self._last_checkpoint: dict[str, float] = {}  # swarm_id -> timestamp

    def start(self) -> None:
        """Start the engine background loop."""
        if self._engine_task is None or self._engine_task.done():
            self._running = True
            self._engine_task = asyncio.create_task(self._engine_loop())
            log.info("Swarm engine started")

    def stop(self) -> None:
        """Stop the engine loop."""
        self._running = False
        if self._engine_task and not self._engine_task.done():
            self._engine_task.cancel()

    async def wait_stopped(self) -> None:
        """Wait for the engine loop to finish."""
        if self._engine_task:
            try:
                await self._engine_task
            except asyncio.CancelledError:
                pass

    # ── Main loop ─────────────────────────────────────────────

    async def _engine_loop(self) -> None:
        """Main loop: poll running swarms, advance tasks."""
        while self._running:
            try:
                await asyncio.sleep(ENGINE_POLL_INTERVAL)
                await self._tick()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                log.error("Swarm engine error: %s", exc, exc_info=True)

    async def _tick(self) -> None:
        """Single engine tick: process all running swarms."""
        store = self._server.swarm_store
        swarms = await store.list_running_swarms()

        for swarm in swarms:
            try:
                await self._advance_swarm(swarm.id)
            except Exception as exc:
                log.error("Error advancing swarm %s: %s", swarm.id[:8], exc, exc_info=True)

    # ── Swarm advancement ─────────────────────────────────────

    async def _advance_swarm(self, swarm_id: str) -> None:
        """Check running tasks, launch ready ones, update swarm status."""
        store = self._server.swarm_store
        swarm = await store.get_swarm(swarm_id)
        if not swarm or swarm.status != "running":
            return

        tasks = await store.list_tasks(swarm_id)
        edges = await store.list_edges(swarm_id)

        # Step 1: Check running tasks for completion.
        for task in tasks:
            if task.status == "running" and task.concern_id:
                await self._check_task_concern(task)
            elif task.status == "retrying":
                await self._check_retry_timer(task)

        # Re-fetch tasks after potential status changes.
        tasks = await store.list_tasks(swarm_id)

        # Step 2: Apply error policy — check for newly failed tasks.
        failed_count = sum(1 for t in tasks if t.status == "failed")
        running_count = sum(1 for t in tasks if t.status == "running")
        terminal_count = sum(1 for t in tasks if t.status in TASK_TERMINAL_STATES)
        pending_approval_count = sum(1 for t in tasks if t.status == "pending_approval")

        if failed_count > 0:
            policy_result = await self._apply_error_policy(swarm, tasks, edges)
            if policy_result == "stop":
                return
            # Re-fetch after potential skips.
            tasks = await store.list_tasks(swarm_id)
            terminal_count = sum(1 for t in tasks if t.status in TASK_TERMINAL_STATES)
            running_count = sum(1 for t in tasks if t.status == "running")

        # Step 3: Check if swarm is complete.
        if terminal_count == len(tasks):
            failed_final = sum(1 for t in tasks if t.status == "failed")
            swarm.status = "failed" if failed_final > 0 else "completed"
            swarm.completed_at = _utcnow_iso()
            swarm.touch()
            await store.save_swarm(swarm)
            await store.add_audit_entry(SwarmAuditEntry(
                swarm_id=swarm_id,
                event_type=f"swarm_{swarm.status}",
                details={
                    "total_tasks": len(tasks),
                    "failed": failed_final,
                    "completed": sum(1 for t in tasks if t.status == "completed"),
                    "skipped": sum(1 for t in tasks if t.status == "skipped"),
                },
                actor="engine",
                created_at=_utcnow_iso(),
            ))
            log.info("Swarm %s %s (%d tasks)", swarm_id[:8], swarm.status, len(tasks))
            return

        # Step 4: Launch ready tasks (respecting concurrency limit).
        ready = get_ready_tasks(tasks, edges)
        available_slots = swarm.concurrency_limit - running_count

        for task in ready[:available_slots]:
            # Approval gate: tasks requiring approval go to pending_approval.
            if task.requires_approval and task.approval_status != "approved":
                if task.approval_status == "rejected":
                    await self._skip_task(task, "Rejected by user")
                else:
                    await self._set_pending_approval(task)
                continue
            await self._launch_task(swarm, task, tasks, edges)

        # Step 5: Auto-checkpoint (throttled).
        await self._maybe_auto_checkpoint(swarm_id)

    # ── Error policy ──────────────────────────────────────────

    async def _apply_error_policy(self, swarm: Any, tasks: list[SwarmTask], edges: list) -> str:
        """Apply the swarm's error policy. Returns 'stop' if swarm should not continue."""
        store = self._server.swarm_store
        policy = swarm.error_policy

        # Check if there are newly failed tasks (not yet handled by policy).
        newly_failed = [t for t in tasks if t.status == "failed"
                        and not t.metadata.get("policy_applied")]

        if not newly_failed:
            return "continue"

        for task in newly_failed:
            task.metadata["policy_applied"] = True
            task.touch()
            await store.save_task(task)

        if policy == "fail_fast":
            # Stop the swarm immediately.
            swarm.status = "failed"
            swarm.completed_at = _utcnow_iso()
            swarm.touch()
            await store.save_swarm(swarm)
            await store.add_audit_entry(SwarmAuditEntry(
                swarm_id=swarm.id,
                event_type="swarm_failed",
                details={
                    "reason": "fail_fast_policy",
                    "failed_tasks": [t.name or t.id[:8] for t in newly_failed],
                },
                actor="engine",
                severity="error",
                created_at=_utcnow_iso(),
            ))
            log.warning("Swarm %s failed (fail_fast policy)", swarm.id[:8])
            return "stop"

        elif policy == "continue_on_error":
            # Skip all dependents of failed tasks.
            for task in newly_failed:
                dependents = get_all_dependents(task.id, tasks, edges)
                for dep_id in dependents:
                    dep_task = next((t for t in tasks if t.id == dep_id), None)
                    if dep_task and dep_task.status not in TASK_TERMINAL_STATES:
                        await self._skip_task(dep_task, f"Skipped: upstream task '{task.name or task.id[:8]}' failed")

            await store.add_audit_entry(SwarmAuditEntry(
                swarm_id=swarm.id,
                event_type="error_policy_continue",
                details={
                    "failed_tasks": [t.name or t.id[:8] for t in newly_failed],
                    "policy": "continue_on_error",
                },
                actor="engine",
                severity="warn",
                created_at=_utcnow_iso(),
            ))
            return "continue"

        elif policy == "manual_review":
            # Pause the swarm for human intervention.
            swarm.status = "paused"
            swarm.touch()
            await store.save_swarm(swarm)
            await store.add_audit_entry(SwarmAuditEntry(
                swarm_id=swarm.id,
                event_type="swarm_paused_for_review",
                details={
                    "reason": "manual_review_policy",
                    "failed_tasks": [t.name or t.id[:8] for t in newly_failed],
                },
                actor="engine",
                severity="warn",
                created_at=_utcnow_iso(),
            ))
            log.warning("Swarm %s paused for manual review", swarm.id[:8])
            return "stop"

        return "continue"

    async def _skip_task(self, task: SwarmTask, reason: str) -> None:
        """Mark a task as skipped."""
        store = self._server.swarm_store
        task.status = "skipped"
        task.completed_at = _utcnow_iso()
        task.error_message = reason
        task.touch()
        await store.save_task(task)

        await store.add_audit_entry(SwarmAuditEntry(
            swarm_id=task.swarm_id,
            task_id=task.id,
            event_type="task_skipped",
            details={"reason": reason},
            actor="engine",
            created_at=_utcnow_iso(),
        ))

    # ── Approval gates ────────────────────────────────────────

    async def _set_pending_approval(self, task: SwarmTask) -> None:
        """Transition task to pending_approval state."""
        store = self._server.swarm_store
        task.status = "pending_approval"
        task.touch()
        await store.save_task(task)

        await store.add_audit_entry(SwarmAuditEntry(
            swarm_id=task.swarm_id,
            task_id=task.id,
            event_type="task_pending_approval",
            details={"task_name": task.name},
            actor="engine",
            severity="warn",
            created_at=_utcnow_iso(),
        ))

        log.info("Swarm task %s awaiting approval", task.name or task.id[:8])

    # ── Task launching ────────────────────────────────────────

    async def _launch_task(
        self,
        swarm: Any,
        task: SwarmTask,
        all_tasks: list[SwarmTask],
        edges: list,
    ) -> None:
        """Create a BotPort concern for a task and dispatch it."""
        store = self._server.swarm_store

        # Build input data from predecessor outputs.
        input_data = self._gather_inputs(task, all_tasks, edges)
        task.input_data = input_data

        # Build the task prompt.
        prompt = self._build_task_prompt(task, input_data)

        # Build file manifest for agent context.
        file_manifest = []
        try:
            fm = self._server.file_manager
            file_manifest = fm.list_files(swarm.id)
            # Strip internal fields, keep what agents need.
            file_manifest = [
                {
                    "filename": f["filename"],
                    "path": f["path"],
                    "size": f["size"],
                    "mime_type": f["mime_type"],
                    "agent": f.get("agent", ""),
                }
                for f in file_manifest
            ]
        except Exception as exc:
            log.debug("Could not build file manifest: %s", exc)

        # Build concern context.
        concern_context: dict[str, Any] = {
            "swarm_id": swarm.id,
            "swarm_task_id": task.id,
            "swarm_task_name": task.name,
            "swarm_files": file_manifest,
        }

        # Inject designed agent spec if present (agent_mode == "designed").
        agent_spec = task.metadata.get("agent_spec")
        if agent_spec:
            concern_context["agent_spec"] = agent_spec

        # Create concern through the concern manager.
        concern = await self._server.concerns.create_concern(
            from_instance="__swarm__",
            task=prompt,
            context=concern_context,
            expertise_tags=[],
            from_session="",
        )
        concern.from_instance_name = f"Swarm: {swarm.name or swarm.id[:8]}"

        # Route to best instance, preferring the assigned persona.
        route_result = await self._route_task(concern, task)

        if route_result is None:
            # No available instance — mark task as retrying.
            task.status = "retrying"
            task.error_message = "No available instance"
            task.touch()
            await store.save_task(task)
            # Clean up the concern.
            await self._server.concerns.fail_concern(
                concern.id, reason="no_available_instance",
            )
            log.warning("Swarm task %s: no available instance", task.id[:8])
            return

        target = route_result.instance
        persona_hint = route_result.persona_name

        # Link concern to task.
        task.concern_id = concern.id
        task.status = "running"
        task.started_at = _utcnow_iso()
        task.assigned_instance = target.name
        if persona_hint:
            task.assigned_persona = persona_hint
        task.touch()
        await store.save_task(task)

        # Assign concern and dispatch.
        concern.assigned_instance_name = target.name
        await self._server.concerns.assign_concern(concern.id, target.id, target.name)
        self._server.connections.increment_active(target.id)

        from botport.protocol import DispatchMessage
        await self._server.connections.send_to(target.id, DispatchMessage(
            concern_id=concern.id,
            from_instance_name=concern.from_instance_name,
            task=concern.task,
            context=concern.context,
            persona_hint=persona_hint,
        ))

        await store.add_audit_entry(SwarmAuditEntry(
            swarm_id=swarm.id,
            task_id=task.id,
            event_type="task_started",
            details={
                "concern_id": concern.id,
                "instance": target.name,
                "persona": persona_hint,
            },
            actor="engine",
            created_at=_utcnow_iso(),
        ))

        log.info(
            "Swarm task %s launched: %s -> %s (persona: %s)",
            task.name or task.id[:8], swarm.id[:8], target.name, persona_hint or "auto",
        )

    async def _route_task(self, concern: Any, task: SwarmTask) -> Any:
        """Route a task, preferring the pre-assigned persona."""
        from botport.router import RouteResult

        # If a specific persona is assigned, try to find it.
        if task.assigned_persona:
            instances = self._server.connections.list_available()
            for inst in instances:
                for persona in inst.personas:
                    if persona.name.lower() == task.assigned_persona.lower():
                        return RouteResult(
                            instance=inst,
                            persona_name=persona.name,
                            reason="swarm_persona_match",
                        )

        # Fall back to normal routing.
        return await self._server.router.route(concern, exclude_instance="__swarm__")

    def _gather_inputs(
        self,
        task: SwarmTask,
        all_tasks: list[SwarmTask],
        edges: list,
    ) -> dict[str, Any]:
        """Gather output data from predecessor tasks as input for this task."""
        pred_ids = get_predecessors(task.id, edges)
        task_map = {t.id: t for t in all_tasks}
        inputs: dict[str, Any] = {}

        for pred_id in pred_ids:
            pred = task_map.get(pred_id)
            if pred and pred.output_data:
                key = pred.name or pred.id[:8]
                inputs[key] = pred.output_data

        return inputs

    def _build_task_prompt(self, task: SwarmTask, input_data: dict) -> str:
        """Build the full prompt for an agent from task description + inputs."""
        prompt = task.description

        if input_data:
            prompt += "\n\n--- Input from previous tasks ---\n"
            for key, data in input_data.items():
                if isinstance(data, dict) and "response" in data:
                    prompt += f"\n[{key}]:\n{data['response']}\n"
                elif isinstance(data, str):
                    prompt += f"\n[{key}]:\n{data}\n"
                else:
                    import json
                    prompt += f"\n[{key}]:\n{json.dumps(data, indent=2)}\n"

        return prompt

    # ── Task monitoring ───────────────────────────────────────

    async def _check_task_concern(self, task: SwarmTask) -> None:
        """Check if a running task's concern has completed."""
        store = self._server.swarm_store
        concern = self._server.concerns.get_concern(task.concern_id)

        if concern is None:
            # Concern not in memory — may have been cleaned up. Load from store.
            concern = await self._server.store.load_concern(task.concern_id)
            if concern is None:
                # Concern completely gone — fail the task.
                await self._fail_task(task, "Concern not found")
                return

        if concern.status == "responded" or concern.status == "closed":
            # Task completed successfully.
            await self._complete_task(task, concern)

        elif concern.status == "failed" or concern.status == "timeout":
            # Task failed — apply retry policy.
            error = concern.metadata.get("fail_reason", concern.status)
            await self._handle_task_failure(task, str(error))

        # If still running (assigned/in_progress), check for timeout escalation.
        elif task.started_at:
            await self._check_timeout_escalation(task)

    async def _check_timeout_escalation(self, task: SwarmTask) -> None:
        """Multi-level timeout check: warn → extend → fail."""
        if task.timeout_seconds <= 0:
            return

        try:
            started = datetime.fromisoformat(task.started_at.replace("Z", "+00:00"))
            elapsed = (datetime.now(timezone.utc) - started).total_seconds()
        except (ValueError, TypeError):
            return

        store = self._server.swarm_store

        # Level 1: Warning threshold.
        if (task.timeout_warn_seconds > 0
                and elapsed > task.timeout_warn_seconds
                and not task.metadata.get("timeout_warned")):
            task.metadata["timeout_warned"] = True
            task.touch()
            await store.save_task(task)

            await store.add_audit_entry(SwarmAuditEntry(
                swarm_id=task.swarm_id,
                task_id=task.id,
                event_type="task_timeout_warning",
                details={
                    "elapsed_seconds": int(elapsed),
                    "warn_threshold": task.timeout_warn_seconds,
                    "hard_timeout": task.timeout_seconds,
                    "extension": task.timeout_extend_seconds,
                },
                actor="engine",
                severity="warn",
                created_at=_utcnow_iso(),
            ))

            log.warning(
                "Swarm task %s timeout warning (%.0fs > %ds)",
                task.name or task.id[:8], elapsed, task.timeout_warn_seconds,
            )

        # Level 2: Hard timeout (with possible extension).
        effective_timeout = task.timeout_seconds
        if task.timeout_extend_seconds > 0 and task.metadata.get("timeout_warned"):
            effective_timeout += task.timeout_extend_seconds

        if elapsed > effective_timeout:
            log.warning(
                "Swarm task %s timed out (%.0fs > %ds)",
                task.name or task.id[:8], elapsed, effective_timeout,
            )
            await self._handle_task_failure(task, "timeout")

    async def _complete_task(self, task: SwarmTask, concern: Any) -> None:
        """Mark task as completed and store its output."""
        store = self._server.swarm_store

        # Extract response from concern messages.
        response_text = ""
        for msg in concern.messages:
            if msg.direction == "response":
                response_text = msg.content
                break

        # Collect file list produced by this task's agent.
        task_files = []
        try:
            fm = self._server.file_manager
            agent_name = task.assigned_instance or ""
            if agent_name:
                task_files = [
                    {"filename": f["filename"], "path": f["path"], "size": f["size"]}
                    for f in fm.list_files(task.swarm_id, agent_name=agent_name)
                ]
        except Exception:
            pass

        task.status = "completed"
        task.completed_at = _utcnow_iso()
        task.output_data = {
            "response": response_text,
            "concern_id": concern.id,
            "persona": concern.metadata.get("persona_name", ""),
            "files": task_files,
        }
        task.touch()
        await store.save_task(task)

        # Record cost if available in concern metadata.
        tokens_in = int(concern.metadata.get("tokens_in", 0))
        tokens_out = int(concern.metadata.get("tokens_out", 0))
        cost_usd = float(concern.metadata.get("cost_usd", 0))
        if tokens_in or tokens_out or cost_usd:
            await store.add_cost_entry(
                swarm_id=task.swarm_id,
                task_id=task.id,
                instance_name=task.assigned_instance,
                persona_name=task.assigned_persona,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                cost_usd=cost_usd,
            )

        # Close the concern if it's in responded state.
        if concern.status == "responded":
            await self._server.concerns.close_concern(
                concern.id, reason="swarm_task_completed",
            )

        await store.add_audit_entry(SwarmAuditEntry(
            swarm_id=task.swarm_id,
            task_id=task.id,
            event_type="task_completed",
            details={
                "response_length": len(response_text),
                "persona": concern.metadata.get("persona_name", ""),
            },
            actor="engine",
            created_at=_utcnow_iso(),
        ))

        log.info("Swarm task %s completed (%d chars)", task.name or task.id[:8], len(response_text))

    async def _handle_task_failure(self, task: SwarmTask, error: str) -> None:
        """Handle task failure with retry policy."""
        store = self._server.swarm_store

        task.retry_count += 1

        if task.retry_count <= task.max_retries:
            # Schedule retry.
            task.status = "retrying"
            task.error_message = f"Attempt {task.retry_count}/{task.max_retries}: {error}"
            task.concern_id = ""
            task.started_at = ""
            task.metadata.pop("timeout_warned", None)
            # Set retry timer: backoff * retry_count.
            wait_seconds = task.retry_backoff_seconds * task.retry_count
            task.metadata["retry_at"] = (
                datetime.now(timezone.utc) + timedelta(seconds=wait_seconds)
            ).isoformat()
            task.touch()
            await store.save_task(task)

            # Try fallback persona on retry.
            if task.fallback_persona and task.retry_count > 1:
                task.assigned_persona = task.fallback_persona

            await store.add_audit_entry(SwarmAuditEntry(
                swarm_id=task.swarm_id,
                task_id=task.id,
                event_type="task_retry_scheduled",
                details={
                    "attempt": task.retry_count,
                    "max_retries": task.max_retries,
                    "wait_seconds": wait_seconds,
                    "error": error,
                },
                actor="engine",
                severity="warn",
                created_at=_utcnow_iso(),
            ))

            log.info(
                "Swarm task %s retry %d/%d in %ds",
                task.name or task.id[:8], task.retry_count, task.max_retries, wait_seconds,
            )
        else:
            await self._fail_task(task, error)

    async def _fail_task(self, task: SwarmTask, error: str) -> None:
        """Permanently fail a task (retries exhausted)."""
        store = self._server.swarm_store

        task.status = "failed"
        task.completed_at = _utcnow_iso()
        task.error_message = error
        task.touch()
        await store.save_task(task)

        await store.add_audit_entry(SwarmAuditEntry(
            swarm_id=task.swarm_id,
            task_id=task.id,
            event_type="task_failed",
            details={"error": error, "retries_used": task.retry_count},
            actor="engine",
            severity="error",
            created_at=_utcnow_iso(),
        ))

        log.warning(
            "Swarm task %s failed after %d retries: %s",
            task.name or task.id[:8], task.retry_count, error,
        )

    async def _check_retry_timer(self, task: SwarmTask) -> None:
        """Check if a retrying task is ready to be re-launched."""
        retry_at = task.metadata.get("retry_at", "")
        if not retry_at:
            # No timer set — reset to queued immediately.
            task.status = "queued"
            task.touch()
            store = self._server.swarm_store
            await store.save_task(task)
            return

        try:
            retry_dt = datetime.fromisoformat(retry_at.replace("Z", "+00:00"))
            if datetime.now(timezone.utc) >= retry_dt:
                task.status = "queued"
                task.metadata.pop("retry_at", None)
                task.touch()
                store = self._server.swarm_store
                await store.save_task(task)
        except (ValueError, TypeError):
            task.status = "queued"
            task.touch()
            store = self._server.swarm_store
            await store.save_task(task)

    # ── Checkpointing ─────────────────────────────────────────

    async def create_checkpoint(self, swarm_id: str, label: str = "") -> SwarmCheckpoint | None:
        """Create a checkpoint snapshot of the current swarm state."""
        store = self._server.swarm_store
        swarm = await store.get_swarm(swarm_id)
        if not swarm:
            return None

        tasks = await store.list_tasks(swarm_id)
        edges = await store.list_edges(swarm_id)

        checkpoint = SwarmCheckpoint(
            id=str(uuid.uuid4()),
            swarm_id=swarm_id,
            label=label or f"Auto checkpoint",
            swarm_state=swarm.to_dict(),
            task_states=[t.to_dict() for t in tasks],
            edge_states=[e.to_dict() for e in edges],
            created_at=_utcnow_iso(),
        )
        await store.save_checkpoint(checkpoint)

        await store.add_audit_entry(SwarmAuditEntry(
            swarm_id=swarm_id,
            event_type="checkpoint_created",
            details={"checkpoint_id": checkpoint.id, "label": checkpoint.label},
            actor="engine",
            created_at=_utcnow_iso(),
        ))

        log.info("Checkpoint created for swarm %s: %s", swarm_id[:8], checkpoint.label)
        return checkpoint

    async def restore_checkpoint(self, checkpoint_id: str) -> bool:
        """Restore a swarm from a checkpoint. Swarm must be paused/terminal."""
        store = self._server.swarm_store
        checkpoint = await store.get_checkpoint(checkpoint_id)
        if not checkpoint:
            return False

        swarm = await store.get_swarm(checkpoint.swarm_id)
        if not swarm:
            return False

        if swarm.status not in ("paused", "failed", "cancelled", "completed"):
            return False

        # Restore swarm state.
        from botport.swarm.models import Swarm, SwarmEdge
        restored_swarm = Swarm.from_dict(checkpoint.swarm_state)
        restored_swarm.status = "ready"  # Set to ready so user can re-start.
        restored_swarm.completed_at = ""
        restored_swarm.touch()
        await store.save_swarm(restored_swarm)

        # Clear current tasks and edges.
        current_tasks = await store.list_tasks(checkpoint.swarm_id)
        for t in current_tasks:
            await store.delete_task(t.id)

        # Restore tasks from checkpoint.
        for task_dict in checkpoint.task_states:
            task = SwarmTask.from_dict(task_dict)
            # Reset non-terminal tasks to queued.
            if task.status not in TASK_TERMINAL_STATES:
                task.status = "queued"
                task.concern_id = ""
                task.started_at = ""
                task.assigned_instance = ""
                task.error_message = ""
                task.metadata.pop("timeout_warned", None)
                task.metadata.pop("retry_at", None)
                task.metadata.pop("policy_applied", None)
            task.touch()
            await store.save_task(task)

        # Restore edges.
        current_edges = await store.list_edges(checkpoint.swarm_id)
        for e in current_edges:
            await store.delete_edge(e.id)
        for edge_dict in checkpoint.edge_states:
            edge = SwarmEdge.from_dict(edge_dict)
            edge.id = 0  # Let DB assign new ID.
            await store.save_edge(edge)

        await store.add_audit_entry(SwarmAuditEntry(
            swarm_id=checkpoint.swarm_id,
            event_type="checkpoint_restored",
            details={
                "checkpoint_id": checkpoint_id,
                "label": checkpoint.label,
                "checkpoint_created_at": checkpoint.created_at,
            },
            actor="user",
            created_at=_utcnow_iso(),
        ))

        log.info("Swarm %s restored from checkpoint %s", checkpoint.swarm_id[:8], checkpoint.label)
        return True

    async def _maybe_auto_checkpoint(self, swarm_id: str) -> None:
        """Create auto-checkpoint if enough time has passed since the last one."""
        now = datetime.now(timezone.utc).timestamp()
        last = self._last_checkpoint.get(swarm_id, 0)

        if now - last >= CHECKPOINT_THROTTLE_SECONDS:
            self._last_checkpoint[swarm_id] = now
            await self.create_checkpoint(swarm_id, label="Auto checkpoint")
