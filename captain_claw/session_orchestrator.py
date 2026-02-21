"""Parallel multi-session orchestrator for Captain Claw.

Decomposes complex requests into a DAG of tasks, runs each task in its
own session via a worker Agent, and synthesizes the final result.

Intra-session pipeline execution runs exactly as today (tools, contracts,
critic). Parallelism happens *across* sessions.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from captain_claw.agent_pool import AgentPool
from captain_claw.config import get_config
from captain_claw.file_registry import FileRegistry
from captain_claw.instructions import InstructionLoader
from captain_claw.llm import LLMProvider, Message, get_provider
from captain_claw.logging import get_logger
from captain_claw.session import get_session_manager
from captain_claw.task_graph import (
    COMPLETED,
    FAILED,
    RUNNING,
    OrchestratorTask,
    TaskGraph,
)

log = get_logger(__name__)

# Timeout for the decomposition and synthesis LLM calls.
_PLANNER_TIMEOUT_SECONDS = 120.0
# Max output tokens for the decompose / synthesize LLM calls.
_DECOMPOSE_MAX_TOKENS = 16000
_SYNTHESIZE_MAX_TOKENS = 16000
# How often to poll for task graph changes during execution.
_POLL_INTERVAL_SECONDS = 1.0


class SessionOrchestrator:
    """Orchestrates parallel session execution.

    Flow:
        1. DECOMPOSE  — Main agent's LLM decomposes request into task plan (JSON)
        2. BUILD GRAPH — Create TaskGraph from plan
        3. ASSIGN SESSIONS — Match existing sessions by name or create new ones
        4. EXECUTE GRAPH — Parallel dispatch loop with traffic light gating
        5. SYNTHESIZE — Feed all results back to main agent for final answer
    """

    def __init__(
        self,
        main_agent: Any | None = None,
        max_parallel: int = 5,
        max_agents: int = 50,
        provider: LLMProvider | None = None,
        status_callback: Callable[[str], None] | None = None,
        tool_output_callback: Callable[[str, dict[str, Any], str], None] | None = None,
        broadcast_callback: Callable[[dict[str, Any]], None] | None = None,
    ):
        cfg = get_config()
        orch_cfg = cfg.orchestrator

        self._main_agent = main_agent
        self._provider = provider or get_provider()
        self._status_callback = status_callback
        self._tool_output_callback = tool_output_callback
        self._broadcast_callback = broadcast_callback
        self._instructions = InstructionLoader()
        self._session_manager = get_session_manager()

        self._pool = AgentPool(
            max_agents=max_agents or orch_cfg.max_agents,
            idle_evict_seconds=orch_cfg.idle_evict_seconds,
            provider=self._provider,
            status_callback=status_callback,
            tool_output_callback=tool_output_callback,
        )
        self._max_parallel = max_parallel or orch_cfg.max_parallel
        self._worker_timeout = orch_cfg.worker_timeout_seconds
        self._worker_max_retries = orch_cfg.worker_max_retries
        self._graph: TaskGraph | None = None
        self._pending_futures: dict[str, asyncio.Task[None]] = {}
        self._execution_done: bool = False
        self._execution_task: asyncio.Task[None] | None = None
        self._resume_event = asyncio.Event()
        # Shared file registry for cross-task file resolution within a run.
        self._file_registry: FileRegistry | None = None
        # Workflow metadata persisted across prepare() → execute().
        self._workflow_name: str = ""
        self._workflow_model: str = ""          # workflow-level model override
        self._workflow_saved_filename: str = "" # safe filename of last save/load
        self._user_input: str = ""
        self._synthesis_instruction: str = ""
        self._workflow_variables: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Workflow naming helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_filename(name: str) -> str:
        """Convert a string to a filesystem-safe slug."""
        slug = re.sub(r"[^\w\s-]", "", name.lower().strip())
        slug = re.sub(r"[\s_-]+", "-", slug).strip("-")
        return slug[:80] or "workflow"

    @staticmethod
    def _generate_workflow_name(summary: str) -> str:
        """Generate a short workflow name from the summary text."""
        words = summary.split()[:5]
        base = " ".join(words) if words else "workflow"
        return SessionOrchestrator._safe_filename(base)

    # ------------------------------------------------------------------
    # Workflow variable helpers
    # ------------------------------------------------------------------

    _VAR_RE = re.compile(r"\{\{(\w+)\}\}")

    @staticmethod
    def _extract_variables(
        texts: list[str],
        existing_vars: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """Scan *texts* for ``{{variable_name}}`` and build a variables list.

        Preserves label/default from *existing_vars* for names that still
        appear; adds new entries for newly detected names; drops entries
        whose name no longer appears in any text.
        """
        seen: set[str] = set()
        ordered: list[str] = []
        for text in texts:
            for m in SessionOrchestrator._VAR_RE.finditer(text or ""):
                name = m.group(1)
                if name not in seen:
                    seen.add(name)
                    ordered.append(name)

        lookup: dict[str, dict[str, Any]] = {}
        if existing_vars:
            for v in existing_vars:
                lookup[v.get("name", "")] = v

        result: list[dict[str, Any]] = []
        for name in ordered:
            if name in lookup:
                result.append(lookup[name])
            else:
                result.append({
                    "name": name,
                    "label": name.replace("_", " ").title(),
                    "default": "",
                })
        return result

    @staticmethod
    def _substitute_variables(text: str, values: dict[str, str]) -> str:
        """Replace ``{{name}}`` placeholders with corresponding *values*.

        Unmatched placeholders (no value provided) are left as-is.
        """
        def _repl(m: re.Match[str]) -> str:
            name = m.group(1)
            return values.get(name, m.group(0))
        return SessionOrchestrator._VAR_RE.sub(_repl, text or "")

    # ------------------------------------------------------------------
    # Status helpers
    # ------------------------------------------------------------------

    def _set_status(self, status: str) -> None:
        if self._status_callback:
            try:
                self._status_callback(status)
            except Exception:
                pass

    def _emit_output(self, tool_name: str, arguments: dict[str, Any], output: str) -> None:
        if self._tool_output_callback:
            try:
                self._tool_output_callback(tool_name, arguments, output)
            except Exception:
                pass

    def _broadcast_event(self, event: str, data: dict[str, Any] | None = None) -> None:
        """Emit a structured orchestrator_event to the broadcast callback."""
        if not self._broadcast_callback:
            return
        payload: dict[str, Any] = {"type": "orchestrator_event", "event": event}
        if data:
            payload.update(data)
        # Attach current graph summary when available.
        if self._graph:
            payload["graph"] = self._graph.get_summary()
        try:
            self._broadcast_callback(payload)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def prepare(self, user_input: str) -> dict[str, Any]:
        """Decompose a request and build the task graph for preview.

        Runs stages 1 (DECOMPOSE) and 2 (BUILD GRAPH) only.  The graph
        is stored internally so :meth:`execute` can run it later.

        Returns:
            Dict with ``ok``, ``workflow_name``, ``summary``, ``tasks``,
            and ``synthesis_instruction``.
        """
        log.info("prepare() called", input_len=len(user_input),
                 input_preview=user_input[:200])
        self._set_status("Orchestrator: decomposing request...")
        self._broadcast_event("decomposing", {"input": user_input[:500]})

        # Create a shared file registry for this orchestration run.
        import uuid
        orch_run_id = str(uuid.uuid4())
        self._file_registry = FileRegistry(orchestration_id=orch_run_id)

        # 1. DECOMPOSE
        plan = await self._decompose(user_input)
        if plan is None:
            log.error("prepare() failed: _decompose returned None")
            self._broadcast_event("error", {"message": "Could not decompose the request into tasks."})
            return {"ok": False, "error": "Could not decompose the request into tasks."}

        tasks_data = plan.get("tasks", [])
        if not tasks_data:
            log.error("prepare() failed: plan has no tasks", plan_keys=list(plan.keys()))
            self._broadcast_event("error", {"message": "Decomposition produced no tasks."})
            return {"ok": False, "error": "Decomposition produced no tasks."}

        synthesis_instruction = str(plan.get("synthesis_instruction", "")).strip()
        summary = str(plan.get("summary", "")).strip()

        # 2. BUILD GRAPH
        log.info("Building task graph", raw_task_count=len(tasks_data),
                 summary=summary[:200])
        self._broadcast_event("building_graph", {"task_count": len(tasks_data)})
        graph = TaskGraph(max_parallel=self._max_parallel)
        skipped_tasks = []
        for i, task_data in enumerate(tasks_data):
            task = OrchestratorTask(
                id=str(task_data.get("id", "")).strip(),
                title=str(task_data.get("title", "")).strip(),
                description=str(task_data.get("description", "")).strip(),
                depends_on=list(task_data.get("depends_on", [])),
                session_name=str(task_data.get("session_name", "")).strip(),
                timeout_seconds=self._worker_timeout,
                max_retries=self._worker_max_retries,
            )
            if not task.id:
                log.warning("Skipping task with empty id",
                            index=i, raw_data=str(task_data)[:300])
                skipped_tasks.append({"index": i, "reason": "empty_id",
                                      "data": str(task_data)[:200]})
                continue
            log.debug("Adding task to graph", task_id=task.id,
                      title=task.title, depends_on=task.depends_on)
            graph.add_task(task)

        if skipped_tasks:
            log.warning("Some tasks were skipped during graph build",
                        skipped_count=len(skipped_tasks), skipped=skipped_tasks)

        if graph.task_count == 0:
            log.error("Graph has 0 valid tasks after build",
                      raw_count=len(tasks_data), skipped=skipped_tasks)
            self._broadcast_event("error", {"message": "Decomposition produced no valid tasks."})
            return {"ok": False, "error": "Decomposition produced no valid tasks."}

        # Store state for execute().
        self._graph = graph
        self._user_input = user_input
        self._synthesis_instruction = synthesis_instruction
        self._workflow_name = self._generate_workflow_name(summary)

        self._emit_output(
            "orchestrator",
            {"event": "decomposed", "task_count": len(tasks_data), "summary": summary},
            json.dumps(plan, ensure_ascii=False, indent=2),
        )

        tasks_out = []
        for tid, t in graph.tasks.items():
            tasks_out.append({
                "id": t.id, "title": t.title, "description": t.description,
                "depends_on": t.depends_on, "session_name": t.session_name,
                "session_id": t.session_id, "model_id": t.model_id,
            })

        # Detect {{variable}} placeholders in the decomposed content.
        all_texts = [user_input, synthesis_instruction]
        for t in tasks_out:
            all_texts.append(t.get("title", ""))
            all_texts.append(t.get("description", ""))
        self._workflow_variables = self._extract_variables(all_texts, self._workflow_variables)

        self._broadcast_event("decomposed", {
            "summary": summary,
            "tasks": tasks_out,
            "workflow_name": self._workflow_name,
            "model": self._workflow_model,
            "variables": self._workflow_variables,
        })

        return {
            "ok": True,
            "workflow_name": self._workflow_name,
            "model": self._workflow_model,
            "summary": summary,
            "tasks": tasks_out,
            "synthesis_instruction": synthesis_instruction,
            "variables": self._workflow_variables,
        }

    async def execute(
        self,
        task_overrides: dict[str, dict[str, Any]] | None = None,
        variable_values: dict[str, str] | None = None,
    ) -> str:
        """Execute a previously prepared graph.

        Runs stages 3 (ASSIGN SESSIONS), 4 (EXECUTE), and 5 (SYNTHESIZE).

        Args:
            task_overrides: Optional per-task overrides keyed by task ID.
                Each value may contain ``title``, ``description``,
                ``session_id``, ``model_id``, and/or ``skills``.
            variable_values: Optional mapping of ``{{name}}`` →  value for
                workflow template variables.  Applied to user_input,
                synthesis_instruction, and all task titles/descriptions.

        Returns:
            Final synthesized response string.
        """
        if self._graph is None:
            return "No prepared graph to execute.  Call prepare() first."

        graph = self._graph

        # Substitute {{variable}} placeholders if values provided.
        if variable_values:
            self._user_input = self._substitute_variables(self._user_input, variable_values)
            self._synthesis_instruction = self._substitute_variables(
                self._synthesis_instruction, variable_values,
            )
            for _tid, task in graph.tasks.items():
                task.title = self._substitute_variables(task.title, variable_values)
                task.description = self._substitute_variables(task.description, variable_values)

        # Apply per-task overrides from the preview editor.
        if task_overrides:
            for tid, overrides in task_overrides.items():
                task = graph.get_task(tid)
                if task is None:
                    continue
                if "title" in overrides:
                    task.title = str(overrides["title"]).strip()
                if "description" in overrides:
                    task.description = str(overrides["description"]).strip()
                if "session_id" in overrides and overrides["session_id"]:
                    task.session_id = str(overrides["session_id"]).strip()
                if "model_id" in overrides and overrides["model_id"]:
                    task.model_id = str(overrides["model_id"]).strip()
                if "skills" in overrides:
                    task.skills = list(overrides["skills"])

        # Apply workflow-level model to tasks that lack their own override.
        if self._workflow_model:
            for _tid, task in graph.tasks.items():
                if not task.model_id:
                    task.model_id = self._workflow_model

        # 3. ASSIGN SESSIONS
        self._broadcast_event("assigning_sessions", {"task_count": graph.task_count})
        await self._assign_sessions(graph)

        # 4. EXECUTE GRAPH
        self._set_status(f"Orchestrator: executing {graph.task_count} tasks...")
        assigned_tasks = []
        for tid, t in graph.tasks.items():
            assigned_tasks.append({
                "id": t.id, "title": t.title, "description": t.description,
                "depends_on": t.depends_on, "session_id": t.session_id,
                "status": t.status,
            })
        self._broadcast_event("assigned", {"tasks": assigned_tasks})
        self._broadcast_event("executing", {"task_count": graph.task_count})
        await self._execute_graph(graph)

        # 5. SYNTHESIZE
        self._set_status("Orchestrator: synthesizing results...")
        self._broadcast_event("synthesizing")
        result = await self._synthesize(self._user_input, graph, self._synthesis_instruction)

        # Save run output to workspace/workflows/.
        output_path = await self._save_run_output(result)

        self._broadcast_event("completed", {
            "result_preview": result or "",
            "has_failures": graph.has_failures,
        })

        # Cleanup
        await self._pool.evict_idle()

        return result

    async def orchestrate(self, user_input: str) -> str:
        """Convenience wrapper: prepare + execute in one call.

        Used by Telegram, CLI, and other non-web paths that do not need
        the preview phase.
        """
        prep = await self.prepare(user_input)
        if not prep.get("ok"):
            return prep.get("error", "Preparation failed.")
        return await self.execute()

    def get_status(self) -> dict[str, Any] | None:
        """Return current orchestration status for the REST API."""
        if self._graph is None:
            return None
        graph = self._graph
        tasks_list = []
        for tid, task in graph.tasks.items():
            tasks_list.append({
                "id": task.id,
                "title": task.title,
                "description": task.description,
                "depends_on": task.depends_on,
                "session_id": task.session_id,
                "status": task.status,
                "error": task.error,
                "retries": task.retries,
                "editing": task.editing,
                "result_preview": (
                    str((task.result or {}).get("output", ""))
                    if task.result else ""
                ),
            })
        return {
            "summary": graph.get_summary(),
            "tasks": tasks_list,
            "workflow_name": self._workflow_name,
            "model": self._workflow_model,
            "user_input": self._user_input,
        }

    async def reset(self) -> None:
        """Cancel any running work and reset to a clean idle state."""
        # Cancel all pending worker futures.
        for tid, fut in list(self._pending_futures.items()):
            if not fut.done():
                fut.cancel()
        for tid, fut in list(self._pending_futures.items()):
            if not fut.done():
                try:
                    await fut
                except (asyncio.CancelledError, Exception):
                    pass
        self._pending_futures.clear()

        # Cancel the execution loop task.
        if self._execution_task and not self._execution_task.done():
            self._execution_task.cancel()
            try:
                await self._execution_task
            except (asyncio.CancelledError, Exception):
                pass
            self._execution_task = None

        # Clear graph and metadata.
        self._graph = None
        self._workflow_name = ""
        self._workflow_model = ""
        self._workflow_saved_filename = ""
        self._user_input = ""
        self._synthesis_instruction = ""
        self._workflow_variables = []
        self._execution_done = False
        self._file_registry = None
        self._resume_event = asyncio.Event()

        # Evict idle agents from the pool.
        await self._pool.evict_idle()

    async def shutdown(self) -> None:
        """Cancel running tasks and release all pool resources."""
        # Cancel all pending worker futures.
        for tid, fut in list(self._pending_futures.items()):
            if not fut.done():
                fut.cancel()
        for tid, fut in list(self._pending_futures.items()):
            if not fut.done():
                try:
                    await fut
                except (asyncio.CancelledError, Exception):
                    pass
        self._pending_futures.clear()

        # Cancel the execution loop task.
        if self._execution_task and not self._execution_task.done():
            self._execution_task.cancel()
            try:
                await self._execution_task
            except (asyncio.CancelledError, Exception):
                pass
            self._execution_task = None

        await self._pool.shutdown()

    # ------------------------------------------------------------------
    # 1. DECOMPOSE
    # ------------------------------------------------------------------

    async def _decompose(self, user_input: str) -> dict[str, Any] | None:
        """Use LLM to decompose user_input into a task plan (JSON)."""
        log.info("Decompose started", input_len=len(user_input),
                 input_preview=user_input[:200])

        available_sessions = await self._list_available_sessions()
        log.debug("Available sessions for decompose", sessions=available_sessions[:300])

        system_prompt = self._instructions.load("orchestrator_decompose_system_prompt.md")
        user_prompt = self._instructions.render(
            "orchestrator_decompose_user_prompt.md",
            user_input=user_input,
            available_sessions=available_sessions,
        )
        log.debug("Decompose prompts ready",
                  system_prompt_len=len(system_prompt) if system_prompt else 0,
                  user_prompt_len=len(user_prompt) if user_prompt else 0)

        if not system_prompt:
            log.error("Decompose system prompt is empty — instruction file missing?")
        if not user_prompt:
            log.error("Decompose user prompt is empty — template render failed?")

        messages = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=user_prompt),
        ]

        try:
            log.info("Calling LLM for decomposition...",
                     provider=type(self._provider).__name__,
                     max_tokens=_DECOMPOSE_MAX_TOKENS,
                     timeout=_PLANNER_TIMEOUT_SECONDS)
            response = await asyncio.wait_for(
                self._provider.complete(messages=messages, tools=None, max_tokens=_DECOMPOSE_MAX_TOKENS),
                timeout=_PLANNER_TIMEOUT_SECONDS,
            )
            log.info("LLM decompose response received",
                     response_type=type(response).__name__,
                     has_content=bool(getattr(response, "content", None)),
                     content_len=len(getattr(response, "content", "") or ""),
                     model=getattr(response, "model", "?"),
                     usage=getattr(response, "usage", None),
                     finish_reason=getattr(response, "finish_reason", None))
        except asyncio.TimeoutError:
            log.error("Orchestrator decomposition timed out",
                      timeout=_PLANNER_TIMEOUT_SECONDS)
            return None
        except Exception as e:
            log.error("Orchestrator decomposition failed",
                      error=str(e), error_type=type(e).__name__)
            return None

        raw = str(getattr(response, "content", "") or "").strip()
        finish = getattr(response, "finish_reason", "")
        log.debug("LLM raw response", raw_len=len(raw), raw_preview=raw[:500],
                  finish_reason=finish)

        if not raw:
            log.error("LLM returned empty content for decompose",
                      finish_reason=finish,
                      model=getattr(response, "model", "?"),
                      usage=getattr(response, "usage", None),
                      user_prompt_preview=user_prompt[:500])
            return None

        parsed = self._parse_json_response(raw)
        if parsed is None:
            log.error("Failed to parse decompose response as JSON",
                      raw_len=len(raw), raw_full=raw[:2000],
                      finish_reason=finish)
        else:
            task_count = len(parsed.get("tasks", []))
            log.info("Decompose JSON parsed OK",
                     task_count=task_count,
                     has_summary=bool(parsed.get("summary")),
                     has_synthesis=bool(parsed.get("synthesis_instruction")))
        return parsed

    async def _list_available_sessions(self) -> str:
        """Build a compact list of existing sessions for the decompose prompt."""
        try:
            sessions = await self._session_manager.list_sessions(limit=30)
        except Exception:
            return "(none)"

        if not sessions:
            return "(none)"

        lines: list[str] = []
        for s in sessions:
            name = str(getattr(s, "name", "")).strip()
            sid = str(getattr(s, "id", "")).strip()
            if name and sid:
                lines.append(f"- {name} (id: {sid})")
        return "\n".join(lines) if lines else "(none)"

    # ------------------------------------------------------------------
    # 3. ASSIGN SESSIONS
    # ------------------------------------------------------------------

    async def _assign_sessions(self, graph: TaskGraph) -> None:
        """Assign a unique session_id to each task.

        If a task already has a session_id (set via preview overrides),
        it is left as-is and ``use_existing_session`` is set.  Otherwise
        a fresh session is created per task.
        """
        for tid, task in graph.tasks.items():
            if task.session_id:
                # User pre-selected an existing session; mark it.
                task.use_existing_session = True
                continue

            # Create a fresh session per task.
            label = task.session_name or task.title or tid
            try:
                session = await self._session_manager.create_session(
                    name=f"orchestrator::{label}",
                )
                task.session_id = session.id
            except Exception as e:
                log.error("Failed to create session for task", task_id=tid, error=str(e))
                task.session_id = f"fallback-{tid}"

    # ------------------------------------------------------------------
    # 4. EXECUTE GRAPH
    # ------------------------------------------------------------------

    async def _execute_graph(self, graph: TaskGraph) -> None:
        """Drive the task graph to completion with parallel workers."""
        self._execution_done = False
        self._resume_event.clear()

        try:
            # Initial activation.
            activated = graph.activate_next()

            for task in activated:
                future = asyncio.create_task(self._run_worker(graph, task))
                self._pending_futures[task.id] = future

            while not graph.is_complete:
                if not self._pending_futures:
                    # Nothing running and graph not complete.
                    timeout_result = graph.tick_timeouts()
                    await self._broadcast_timeout_events(timeout_result)
                    graph.refresh()
                    if graph.is_complete:
                        break
                    newly_active = graph.activate_next()
                    if newly_active:
                        for task in newly_active:
                            future = asyncio.create_task(self._run_worker(graph, task))
                            self._pending_futures[task.id] = future
                        continue
                    # No tasks activatable — could be waiting for user edits
                    # or timeout postponements.
                    # Wait for resume signal instead of breaking.
                    try:
                        await asyncio.wait_for(
                            self._resume_event.wait(), timeout=2.0,
                        )
                        self._resume_event.clear()
                    except asyncio.TimeoutError:
                        # Check again — user might have restarted a task.
                        graph.refresh()
                        if graph.is_complete:
                            break
                    continue

                # Wait for at least one worker to finish.
                done, _ = await asyncio.wait(
                    self._pending_futures.values(),
                    timeout=_POLL_INTERVAL_SECONDS,
                    return_when=asyncio.FIRST_COMPLETED,
                )

                # Remove completed futures.
                completed_ids = [
                    tid for tid, fut in self._pending_futures.items() if fut.done()
                ]
                for tid in completed_ids:
                    self._pending_futures.pop(tid, None)

                # Tick timeouts and broadcast warning/countdown events.
                timeout_result = graph.tick_timeouts()
                await self._broadcast_timeout_events(timeout_result)

                # Activate newly ready tasks.
                newly_active = graph.activate_next()
                for task in newly_active:
                    self._set_status(f"Orchestrator: starting '{task.title}'...")
                    future = asyncio.create_task(self._run_worker(graph, task))
                    self._pending_futures[task.id] = future

                # Emit progress.
                self._emit_output(
                    "orchestrator",
                    {"event": "progress", **graph.get_summary()},
                    f"Graph: {graph.get_summary()}",
                )
                self._broadcast_event("progress")

        except asyncio.CancelledError:
            log.info("Execution graph cancelled (shutdown)")
            await self._cancel_pending_futures()
            raise
        finally:
            await self._cancel_pending_futures()
            self._execution_done = True

    async def _broadcast_timeout_events(self, timeout_result: dict[str, Any]) -> None:
        """Broadcast timeout warning, countdown, restart, and failure events."""
        # Newly warned tasks.
        for tid in timeout_result.get("warned", []):
            task = self._graph.get_task(tid) if self._graph else None
            title = task.title if task else tid
            self._broadcast_event("timeout_warning", {
                "task_id": tid,
                "title": title,
                "remaining_seconds": 60,
            })
            log.info("Task timeout warning", task_id=tid, title=title)

        # Tasks whose grace period expired and were restarted — cancel their
        # worker futures so the agent stops working on the old attempt.
        for tid in timeout_result.get("restarted", []):
            task = self._graph.get_task(tid) if self._graph else None
            title = task.title if task else tid
            await self._cancel_worker_future(tid)
            self._broadcast_event("task_restarted", {
                "task_id": tid,
                "title": title,
                "reason": "timeout",
            })
            log.info("Task restarted after timeout", task_id=tid, title=title)

        # Tasks that exhausted retries after timeout — cancel their workers.
        for tid in timeout_result.get("failed", []):
            task = self._graph.get_task(tid) if self._graph else None
            title = task.title if task else tid
            await self._cancel_worker_future(tid)
            self._broadcast_event("task_failed", {
                "task_id": tid,
                "title": title,
                "error": "timeout_exhausted",
            })
            log.info("Task failed (timeout exhausted)", task_id=tid, title=title)

        # Active countdown updates for tasks in warning phase.
        countdown = timeout_result.get("countdown", [])
        if countdown:
            self._broadcast_event("timeout_countdown", {
                "tasks": countdown,
            })

    async def _cancel_worker_future(self, task_id: str) -> None:
        """Cancel and clean up a single worker future."""
        fut = self._pending_futures.pop(task_id, None)
        if fut and not fut.done():
            fut.cancel()
            try:
                await fut
            except (asyncio.CancelledError, Exception):
                pass

    async def _cancel_pending_futures(self) -> None:
        """Cancel and await all pending worker futures."""
        for tid, fut in list(self._pending_futures.items()):
            if not fut.done():
                fut.cancel()
        for tid, fut in list(self._pending_futures.items()):
            if not fut.done():
                try:
                    await fut
                except (asyncio.CancelledError, Exception):
                    pass
        self._pending_futures.clear()

    async def _run_worker(self, graph: TaskGraph, task: OrchestratorTask) -> None:
        """Execute a single task via a worker agent."""
        self._set_status(f"Worker: {task.title}...")
        self._emit_output(
            "orchestrator",
            {"event": "worker_start", "task_id": task.id, "title": task.title},
            f"Starting worker for: {task.title}",
        )
        self._broadcast_event("task_started", {
            "task_id": task.id, "title": task.title,
            "session_id": task.session_id,
        })

        try:
            agent = await self._pool.get_or_create(
                task.session_id,
                file_registry=self._file_registry,
            )

            # Apply per-task model override if specified.
            if task.model_id:
                try:
                    await agent.set_session_model(task.model_id, persist=False)
                except Exception as e:
                    log.warning("Failed to set task model", task_id=task.id, model=task.model_id, error=str(e))

            # Build file manifest from prior completed tasks so the worker
            # knows which files are available from upstream dependencies.
            file_manifest = ""
            if self._file_registry and len(self._file_registry) > 0:
                manifest_text = self._file_registry.build_manifest()
                if manifest_text:
                    file_manifest = f"\n\n{manifest_text}\n"

            worker_prompt = self._instructions.render(
                "orchestrator_worker_prompt.md",
                task_title=task.title,
                task_description=task.description,
                file_manifest=file_manifest,
            )
            # No asyncio.wait_for timeout — timeout management is handled
            # by tick_timeouts() in the execution loop, which provides
            # a warning phase and user-postpone flow before restarting.
            response = await agent.complete(worker_prompt)
            result = {
                "success": True,
                "output": str(response or "").strip(),
            }
            graph.complete_task(task.id, result)

            # Collect usage metrics from the worker agent.
            usage = getattr(agent, "last_usage", {}) or {}
            ctx = getattr(agent, "last_context_window", {}) or {}

            self._emit_output(
                "orchestrator",
                {"event": "worker_done", "task_id": task.id, "title": task.title},
                f"Completed: {task.title}",
            )
            self._broadcast_event("task_completed", {
                "task_id": task.id, "title": task.title,
                "output": str(response or "").strip(),
                "usage": usage,
                "context": {
                    "prompt_tokens": ctx.get("prompt_tokens", 0),
                    "budget": ctx.get("context_budget_tokens", 0),
                    "utilization": round(ctx.get("utilization", 0) * 100, 1),
                    "messages": ctx.get("included_messages", 0),
                },
            })
        except asyncio.CancelledError:
            log.info("Worker cancelled", task_id=task.id, title=task.title)
            if task.status == RUNNING:
                graph.fail_task(task.id, error="cancelled")
            raise  # Re-raise so the caller knows it was cancelled.
        except asyncio.TimeoutError:
            log.warning("Worker timed out", task_id=task.id, title=task.title)
            graph.fail_task(task.id, error="timeout")
            self._emit_output(
                "orchestrator",
                {"event": "worker_timeout", "task_id": task.id},
                f"Timed out: {task.title}",
            )
            self._broadcast_event("task_failed", {
                "task_id": task.id, "title": task.title, "error": "timeout",
            })
        except Exception as e:
            log.error("Worker failed", task_id=task.id, error=str(e))
            graph.fail_task(task.id, error=str(e))
            self._emit_output(
                "orchestrator",
                {"event": "worker_error", "task_id": task.id, "error": str(e)},
                f"Failed: {task.title} — {e}",
            )
            self._broadcast_event("task_failed", {
                "task_id": task.id, "title": task.title, "error": str(e),
            })
        finally:
            await self._pool.release(task.session_id)

    # ------------------------------------------------------------------
    # Task control: pause / edit / update / restart / resume
    # ------------------------------------------------------------------

    async def pause_task(self, task_id: str) -> dict[str, Any]:
        """Pause a running task.  Cancels its worker."""
        if not self._graph:
            return {"ok": False, "error": "No active graph"}

        task = self._graph.get_task(task_id)
        if task is None:
            return {"ok": False, "error": "Task not found"}
        if task.status != RUNNING:
            return {"ok": False, "error": f"Task is {task.status}, not running"}

        # Cancel the worker future.
        fut = self._pending_futures.get(task_id)
        if fut and not fut.done():
            fut.cancel()
            try:
                await fut
            except (asyncio.CancelledError, Exception):
                pass
            self._pending_futures.pop(task_id, None)

        self._graph.pause_task(task_id)
        await self._pool.release(task.session_id)

        self._broadcast_event("task_paused", {
            "task_id": task_id, "title": task.title,
        })
        return {"ok": True}

    async def edit_task(self, task_id: str) -> dict[str, Any]:
        """Put a task into edit mode."""
        if not self._graph:
            return {"ok": False, "error": "No active graph"}

        task = self._graph.get_task(task_id)
        if task is None:
            return {"ok": False, "error": "Task not found"}

        # If task is running, pause it first.
        if task.status == RUNNING:
            pause_result = await self.pause_task(task_id)
            if not pause_result.get("ok"):
                return pause_result

        if not self._graph.edit_task(task_id):
            return {"ok": False, "error": f"Cannot edit task in {task.status} state"}

        self._broadcast_event("task_editing", {
            "task_id": task_id, "title": task.title,
            "description": task.description,
        })
        return {"ok": True, "description": task.description}

    async def update_task(self, task_id: str, description: str) -> dict[str, Any]:
        """Update a task's instructions (description)."""
        if not self._graph:
            return {"ok": False, "error": "No active graph"}

        if not self._graph.update_task_description(task_id, description):
            return {"ok": False, "error": "Task not found"}

        task = self._graph.get_task(task_id)
        self._broadcast_event("task_updated", {
            "task_id": task_id,
            "title": task.title if task else task_id,
            "description": description,
        })
        return {"ok": True}

    async def restart_task(self, task_id: str) -> dict[str, Any]:
        """Restart a failed/completed/paused/editing task."""
        if not self._graph:
            return {"ok": False, "error": "No active graph"}

        task = self._graph.get_task(task_id)
        if task is None:
            return {"ok": False, "error": "Task not found"}

        # Evict the cached agent so the task starts fresh on next run.
        if task.session_id:
            await self._pool.evict(task.session_id)

        if not self._graph.restart_task(task_id):
            return {"ok": False, "error": f"Cannot restart task in {task.status} state"}

        # Un-cascade dependents that failed because this task failed.
        self._uncascade_dependents(task_id)

        self._broadcast_event("task_restarted", {
            "task_id": task_id, "title": task.title,
        })

        # Re-enter the execution loop if it has exited.
        self._reenter_execution_if_needed()
        return {"ok": True}

    async def resume_task(self, task_id: str) -> dict[str, Any]:
        """Resume a paused or editing task back to PENDING."""
        if not self._graph:
            return {"ok": False, "error": "No active graph"}

        task = self._graph.get_task(task_id)
        if task is None:
            return {"ok": False, "error": "Task not found"}

        # If resuming from edit mode and description changed, evict cached
        # agent so the task re-runs with a fresh session (no stale context).
        was_editing = task.editing
        desc_changed = (
            was_editing
            and task.original_description
            and task.description != task.original_description
        )

        if not self._graph.resume_task(task_id):
            return {"ok": False, "error": "Cannot resume task"}

        if desc_changed and task.session_id:
            await self._pool.evict(task.session_id)

        self._broadcast_event("task_resumed", {
            "task_id": task_id,
            "title": task.title if task else task_id,
        })

        # Re-enter the execution loop if it has exited.
        self._reenter_execution_if_needed()
        return {"ok": True}

    async def postpone_task(self, task_id: str) -> dict[str, Any]:
        """Postpone a timeout warning, granting another full timeout period."""
        if not self._graph:
            return {"ok": False, "error": "No active graph"}

        task = self._graph.get_task(task_id)
        if task is None:
            return {"ok": False, "error": "Task not found"}

        if not self._graph.postpone_task(task_id):
            return {"ok": False, "error": f"Cannot postpone task in {task.status} state"}

        self._broadcast_event("timeout_postponed", {
            "task_id": task_id,
            "title": task.title,
        })
        log.info("Task timeout postponed", task_id=task_id, title=task.title)
        return {"ok": True}

    def _uncascade_dependents(self, task_id: str) -> None:
        """Reset cascade-failed dependents when their dependency is restarted."""
        if not self._graph:
            return
        for tid, task in self._graph.tasks.items():
            if task.status == FAILED and task.error == "dependency_failed":
                if task_id in task.depends_on:
                    task.status = "pending"
                    task.error = ""
                    task.completed_at = 0.0

    def _reenter_execution_if_needed(self) -> None:
        """Re-enter execution loop if it has exited, or signal it."""
        if self._execution_done and self._graph:
            self._execution_done = False
            self._execution_task = asyncio.create_task(
                self._execute_graph(self._graph)
            )
        else:
            # Signal the running loop to check for new activatable tasks.
            self._resume_event.set()

    # ------------------------------------------------------------------
    # 5. SYNTHESIZE
    # ------------------------------------------------------------------

    async def _synthesize(
        self,
        user_input: str,
        graph: TaskGraph,
        synthesis_instruction: str,
    ) -> str:
        """Feed all task results back to the LLM for a final combined answer."""
        results = graph.get_results()
        task_results_text = self._format_results_for_synthesis(results)

        # Append file manifest so synthesis knows about all created files.
        if self._file_registry and len(self._file_registry) > 0:
            manifest = self._file_registry.build_manifest()
            if manifest:
                task_results_text = f"{task_results_text}\n\n{manifest}"

        user_prompt = self._instructions.render(
            "orchestrator_synthesize_user_prompt.md",
            user_input=user_input,
            task_results=task_results_text,
            synthesis_instruction=synthesis_instruction or "Provide a comprehensive answer.",
        )

        # Use the main agent's provider for synthesis (keeps context in main session).
        messages = [
            Message(role="user", content=user_prompt),
        ]

        try:
            response = await asyncio.wait_for(
                self._provider.complete(messages=messages, tools=None, max_tokens=_SYNTHESIZE_MAX_TOKENS),
                timeout=_PLANNER_TIMEOUT_SECONDS,
            )
            return str(getattr(response, "content", "") or "").strip() or "Synthesis returned no content."
        except asyncio.TimeoutError:
            return f"Synthesis timed out. Raw results:\n{task_results_text}"
        except Exception as e:
            return f"Synthesis failed ({e}). Raw results:\n{task_results_text}"

    @staticmethod
    def _format_results_for_synthesis(results: dict[str, dict[str, Any]]) -> str:
        """Format task results into a readable block for the synthesis prompt."""
        lines: list[str] = []
        for tid, info in results.items():
            title = info.get("title", tid)
            status = info.get("status", "unknown")
            output = ""
            result_data = info.get("result")
            if isinstance(result_data, dict):
                output = str(result_data.get("output", "")).strip()
            error = info.get("error", "")

            lines.append(f"### Task: {title} (id: {tid})")
            lines.append(f"Status: {status}")
            if output:
                lines.append(f"Output:\n{output}")
            if error:
                lines.append(f"Error: {error}")
            lines.append("")
        return "\n".join(lines).strip()

    # ------------------------------------------------------------------
    # Workflow save / load / export
    # ------------------------------------------------------------------

    def _workflows_dir(self) -> Path:
        """Return (and create) the workflows directory under workspace."""
        cfg = get_config()
        ws = cfg.resolved_workspace_path()
        d = ws / "workflows"
        d.mkdir(parents=True, exist_ok=True)
        return d

    async def save_workflow(
        self,
        name: str | None = None,
        task_overrides: dict[str, dict[str, Any]] | None = None,
        model: str | None = None,
    ) -> dict[str, Any]:
        """Serialize the current graph as a reusable workflow JSON file.

        Args:
            name: Workflow name (uses generated name if omitted).
            task_overrides: Optional per-task overrides from the preview
                editor (title, description, session_id, model_id, skills).
                Applied to the graph tasks before serializing so that
                user-configured session/model selections are persisted.
            model: Workflow-level model override.  When set, this model is
                applied to every task that doesn't have its own override.
        """
        if self._graph is None:
            return {"ok": False, "error": "No prepared graph to save."}

        # Apply pending preview overrides so they are persisted.
        if task_overrides:
            for tid, overrides in task_overrides.items():
                task = self._graph.get_task(tid)
                if task is None:
                    continue
                if "title" in overrides:
                    task.title = str(overrides["title"]).strip()
                if "description" in overrides:
                    task.description = str(overrides["description"]).strip()
                if "session_id" in overrides and overrides["session_id"]:
                    task.session_id = str(overrides["session_id"]).strip()
                if "model_id" in overrides and overrides["model_id"]:
                    task.model_id = str(overrides["model_id"]).strip()
                if "skills" in overrides:
                    task.skills = list(overrides["skills"])

        # Track workflow-level model.
        if model is not None:
            self._workflow_model = model

        wf_name = name or self._workflow_name or "workflow"
        new_safe = self._safe_filename(wf_name)
        wf_dir = self._workflows_dir()
        path = wf_dir / f"{new_safe}.json"

        # If the name changed, remove the old file to avoid duplicates.
        old_safe = self._workflow_saved_filename
        if old_safe and old_safe != new_safe:
            old_path = wf_dir / f"{old_safe}.json"
            if old_path.is_file():
                try:
                    old_path.unlink()
                except Exception:
                    pass  # best-effort cleanup

        tasks_out: list[dict[str, Any]] = []
        for tid, t in self._graph.tasks.items():
            tasks_out.append({
                "id": t.id,
                "title": t.title,
                "description": t.description,
                "depends_on": t.depends_on,
                "session_name": t.session_name,
                "session_id": t.session_id,
                "model_id": t.model_id,
                "skills": t.skills,
            })

        # Auto-detect {{variable}} placeholders and build metadata.
        all_texts = [self._user_input, self._synthesis_instruction]
        for t in tasks_out:
            all_texts.append(t.get("title", ""))
            all_texts.append(t.get("description", ""))
        variables = self._extract_variables(all_texts, self._workflow_variables)

        payload: dict[str, Any] = {
            "workflow_name": wf_name,
            "user_input": self._user_input,
            "synthesis_instruction": self._synthesis_instruction,
            "tasks": tasks_out,
        }
        if self._workflow_model:
            payload["model"] = self._workflow_model
        if variables:
            payload["variables"] = variables

        try:
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:
            return {"ok": False, "error": str(e)}

        # Update internal tracking so subsequent saves detect renames.
        self._workflow_name = wf_name
        self._workflow_saved_filename = new_safe

        self._broadcast_event("workflow_saved", {"name": wf_name, "path": str(path)})
        return {"ok": True, "name": wf_name, "path": str(path)}

    async def load_workflow(self, name: str) -> dict[str, Any]:
        """Load a workflow JSON file and rebuild the graph for preview."""
        safe = self._safe_filename(name)
        path = self._workflows_dir() / f"{safe}.json"

        if not path.is_file():
            return {"ok": False, "error": f"Workflow '{name}' not found."}

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            return {"ok": False, "error": f"Failed to read workflow: {e}"}

        tasks_data = data.get("tasks", [])
        if not tasks_data:
            return {"ok": False, "error": "Workflow contains no tasks."}

        # Build graph from saved tasks.
        graph = TaskGraph(max_parallel=self._max_parallel)
        for td in tasks_data:
            task = OrchestratorTask(
                id=str(td.get("id", "")).strip(),
                title=str(td.get("title", "")).strip(),
                description=str(td.get("description", "")).strip(),
                depends_on=list(td.get("depends_on", [])),
                session_name=str(td.get("session_name", "")).strip(),
                session_id=str(td.get("session_id", "")).strip(),
                model_id=str(td.get("model_id", "")).strip(),
                skills=list(td.get("skills", [])),
                timeout_seconds=self._worker_timeout,
                max_retries=self._worker_max_retries,
            )
            if task.id:
                graph.add_task(task)

        if graph.task_count == 0:
            return {"ok": False, "error": "No valid tasks in workflow."}

        # Store state for preview/execute.
        self._graph = graph
        self._workflow_name = data.get("workflow_name", name)
        self._workflow_model = data.get("model", "")
        self._workflow_saved_filename = safe
        self._user_input = data.get("user_input", "")
        self._synthesis_instruction = data.get("synthesis_instruction", "")
        self._workflow_variables = data.get("variables", [])

        import uuid
        self._file_registry = FileRegistry(orchestration_id=str(uuid.uuid4()))

        tasks_out: list[dict[str, Any]] = []
        for tid, t in graph.tasks.items():
            tasks_out.append({
                "id": t.id, "title": t.title, "description": t.description,
                "depends_on": t.depends_on, "session_name": t.session_name,
                "session_id": t.session_id, "model_id": t.model_id,
                "skills": t.skills,
            })

        self._broadcast_event("decomposed", {
            "summary": f"Loaded workflow: {self._workflow_name}",
            "tasks": tasks_out,
            "workflow_name": self._workflow_name,
            "model": self._workflow_model,
            "user_input": self._user_input,
            "variables": self._workflow_variables,
        })

        return {
            "ok": True,
            "workflow_name": self._workflow_name,
            "model": self._workflow_model,
            "tasks": tasks_out,
            "synthesis_instruction": self._synthesis_instruction,
            "user_input": self._user_input,
            "variables": self._workflow_variables,
        }

    async def list_workflows(self) -> list[dict[str, Any]]:
        """List saved workflow files."""
        d = self._workflows_dir()
        result: list[dict[str, Any]] = []
        for p in sorted(d.glob("*.json")):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                result.append({
                    "name": data.get("workflow_name", p.stem),
                    "filename": p.stem,
                    "task_count": len(data.get("tasks", [])),
                    "model": data.get("model", ""),
                    "has_variables": bool(data.get("variables")),
                    "variables": data.get("variables", []),
                })
            except Exception:
                result.append({"name": p.stem, "filename": p.stem, "task_count": 0})
        return result

    async def delete_workflow(self, name: str) -> dict[str, Any]:
        """Delete a saved workflow file."""
        safe = self._safe_filename(name)
        path = self._workflows_dir() / f"{safe}.json"
        if not path.is_file():
            return {"ok": False, "error": f"Workflow '{name}' not found."}
        try:
            path.unlink()
        except Exception as e:
            return {"ok": False, "error": str(e)}
        return {"ok": True}

    async def _save_run_output(self, synthesis_result: str) -> str | None:
        """Save a Markdown report of the completed run."""
        if not self._graph:
            return None

        wf_name = self._workflow_name or "workflow"
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        safe = self._safe_filename(wf_name)
        filename = f"{safe}-output-{stamp}.md"
        path = self._workflows_dir() / filename

        lines: list[str] = [
            f"# Workflow: {wf_name}",
            f"**Run**: {datetime.now().isoformat()}",
            f"**Tasks**: {self._graph.task_count}",
            "",
            "---",
            "",
        ]

        for tid, task in self._graph.tasks.items():
            lines.append(f"## Task: {task.title} (`{tid}`)")
            lines.append(f"**Status**: {task.status}")
            lines.append("")
            lines.append("### Instructions")
            lines.append(task.description or "_No instructions._")
            lines.append("")

            output = ""
            if task.result and isinstance(task.result, dict):
                output = str(task.result.get("output", "")).strip()
            if output:
                lines.append("### Output")
                lines.append(output)
                lines.append("")

            if task.error:
                lines.append("### Error")
                lines.append(task.error)
                lines.append("")

            lines.append("---")
            lines.append("")

        lines.append("## Synthesis")
        lines.append(synthesis_result or "_No synthesis result._")
        lines.append("")

        try:
            path.write_text("\n".join(lines), encoding="utf-8")
            self._broadcast_event("output_saved", {"filename": filename, "path": str(path)})
            return str(path)
        except Exception as e:
            log.error("Failed to save run output", error=str(e))
            return None

    # ------------------------------------------------------------------
    # JSON parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_json_response(raw: str) -> dict[str, Any] | None:
        """Parse JSON from LLM response, handling markdown fences."""
        text = raw.strip()
        if not text:
            log.warning("_parse_json_response: empty input")
            return None

        # Try direct parse.
        try:
            value = json.loads(text)
            if isinstance(value, dict):
                log.debug("_parse_json_response: direct parse OK")
                return value
            log.warning("_parse_json_response: direct parse returned non-dict",
                        value_type=type(value).__name__)
        except json.JSONDecodeError as e:
            log.debug("_parse_json_response: direct parse failed",
                      error=str(e), pos=e.pos)

        # Strip markdown code fences.
        fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
        if fence_match:
            extracted = fence_match.group(1).strip()
            log.debug("_parse_json_response: found code fence",
                      extracted_len=len(extracted),
                      extracted_preview=extracted[:300])
            try:
                value = json.loads(extracted)
                if isinstance(value, dict):
                    log.debug("_parse_json_response: fence parse OK")
                    return value
                log.warning("_parse_json_response: fence parse returned non-dict",
                            value_type=type(value).__name__)
            except json.JSONDecodeError as e:
                log.warning("_parse_json_response: fence parse failed",
                            error=str(e), pos=e.pos,
                            extracted_preview=extracted[:500])
        else:
            log.debug("_parse_json_response: no code fence found")

        # Last resort: find first { ... } block.
        brace_match = re.search(r"\{.*\}", text, re.DOTALL)
        if brace_match:
            extracted = brace_match.group(0)
            log.debug("_parse_json_response: found brace block",
                      extracted_len=len(extracted),
                      extracted_preview=extracted[:300])
            try:
                value = json.loads(extracted)
                if isinstance(value, dict):
                    log.debug("_parse_json_response: brace parse OK")
                    return value
                log.warning("_parse_json_response: brace parse returned non-dict",
                            value_type=type(value).__name__)
            except json.JSONDecodeError as e:
                log.warning("_parse_json_response: brace parse failed",
                            error=str(e), pos=e.pos,
                            extracted_preview=extracted[:500])
        else:
            log.warning("_parse_json_response: no brace block found in response",
                        text_preview=text[:500])

        log.error("_parse_json_response: all parse attempts failed",
                  text_len=len(text), text_preview=text[:1000])
        return None
