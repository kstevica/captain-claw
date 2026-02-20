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
from typing import Any, Callable

from captain_claw.agent_pool import AgentPool
from captain_claw.config import get_config
from captain_claw.instructions import InstructionLoader
from captain_claw.llm import LLMProvider, Message, get_provider
from captain_claw.logging import get_logger
from captain_claw.session import get_session_manager
from captain_claw.task_graph import (
    COMPLETED,
    FAILED,
    OrchestratorTask,
    TaskGraph,
)

log = get_logger(__name__)

# Timeout for the decomposition and synthesis LLM calls.
_PLANNER_TIMEOUT_SECONDS = 120.0
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

    async def orchestrate(self, user_input: str) -> str:
        """Decompose, execute in parallel, and synthesize a response.

        Args:
            user_input: The user's complex request.

        Returns:
            Final synthesized response string.
        """
        self._set_status("Orchestrator: decomposing request...")
        self._broadcast_event("decomposing", {"input": user_input[:500]})

        # 1. DECOMPOSE
        plan = await self._decompose(user_input)
        if plan is None:
            self._broadcast_event("error", {"message": "Could not decompose the request into tasks."})
            return "Could not decompose the request into tasks."

        tasks_data = plan.get("tasks", [])
        if not tasks_data:
            self._broadcast_event("error", {"message": "Decomposition produced no tasks."})
            return "Decomposition produced no tasks."

        synthesis_instruction = str(plan.get("synthesis_instruction", "")).strip()
        summary = str(plan.get("summary", "")).strip()

        self._emit_output(
            "orchestrator",
            {"event": "decomposed", "task_count": len(tasks_data), "summary": summary},
            json.dumps(plan, ensure_ascii=False, indent=2),
        )
        self._broadcast_event("decomposed", {
            "summary": summary,
            "tasks": tasks_data,
        })

        # 2. BUILD GRAPH
        self._broadcast_event("building_graph", {"task_count": len(tasks_data)})
        graph = TaskGraph(max_parallel=self._max_parallel)
        for task_data in tasks_data:
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
                continue
            graph.add_task(task)

        if graph.task_count == 0:
            self._broadcast_event("error", {"message": "Decomposition produced no valid tasks."})
            return "Decomposition produced no valid tasks."

        self._graph = graph

        # 3. ASSIGN SESSIONS
        self._broadcast_event("assigning_sessions", {"task_count": graph.task_count})
        await self._assign_sessions(graph)

        # 4. EXECUTE GRAPH
        self._set_status(f"Orchestrator: executing {graph.task_count} tasks...")
        # Broadcast assigned task details for the dashboard.
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
        result = await self._synthesize(user_input, graph, synthesis_instruction)

        self._broadcast_event("completed", {
            "result_preview": result[:500] if result else "",
            "has_failures": graph.has_failures,
        })

        # Cleanup
        await self._pool.evict_idle()

        return result

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
                "result_preview": (
                    str((task.result or {}).get("output", ""))[:300]
                    if task.result else ""
                ),
            })
        return {
            "summary": graph.get_summary(),
            "tasks": tasks_list,
        }

    async def shutdown(self) -> None:
        """Release all pool resources."""
        await self._pool.shutdown()

    # ------------------------------------------------------------------
    # 1. DECOMPOSE
    # ------------------------------------------------------------------

    async def _decompose(self, user_input: str) -> dict[str, Any] | None:
        """Use LLM to decompose user_input into a task plan (JSON)."""
        available_sessions = await self._list_available_sessions()

        system_prompt = self._instructions.load("orchestrator_decompose_system_prompt.md")
        user_prompt = self._instructions.render(
            "orchestrator_decompose_user_prompt.md",
            user_input=user_input,
            available_sessions=available_sessions,
        )

        messages = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=user_prompt),
        ]

        try:
            response = await asyncio.wait_for(
                self._provider.complete(messages=messages, tools=None, max_tokens=4000),
                timeout=_PLANNER_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            log.error("Orchestrator decomposition timed out")
            return None
        except Exception as e:
            log.error("Orchestrator decomposition failed", error=str(e))
            return None

        raw = str(getattr(response, "content", "") or "").strip()
        return self._parse_json_response(raw)

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

        Every task gets its own fresh session to enable true parallel
        execution. Session names from the decomposer are used only as
        naming hints, never for session reuse — sharing a session between
        two concurrent workers would serialize them.
        """
        for tid, task in graph.tasks.items():
            if task.session_id:
                continue

            # Always create a fresh session per task.
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
        # Initial activation.
        activated = graph.activate_next()
        pending_futures: dict[str, asyncio.Task[None]] = {}

        for task in activated:
            future = asyncio.create_task(self._run_worker(graph, task))
            pending_futures[task.id] = future

        while not graph.is_complete:
            if not pending_futures:
                # Nothing running and graph not complete — likely stuck.
                timeout_result = graph.tick_timeouts()
                if not timeout_result.get("timed_out"):
                    # No timeouts either — force complete check.
                    graph.refresh()
                    if graph.is_complete:
                        break
                    # Attempt activation once more.
                    newly_active = graph.activate_next()
                    if not newly_active:
                        log.warning("Orchestrator graph stuck — no tasks runnable")
                        break
                    for task in newly_active:
                        future = asyncio.create_task(self._run_worker(graph, task))
                        pending_futures[task.id] = future
                continue

            # Wait for at least one worker to finish.
            done, _ = await asyncio.wait(
                pending_futures.values(),
                timeout=_POLL_INTERVAL_SECONDS,
                return_when=asyncio.FIRST_COMPLETED,
            )

            # Remove completed futures.
            completed_ids = [
                tid for tid, fut in pending_futures.items() if fut.done()
            ]
            for tid in completed_ids:
                pending_futures.pop(tid, None)

            # Tick timeouts.
            graph.tick_timeouts()

            # Activate newly ready tasks.
            newly_active = graph.activate_next()
            for task in newly_active:
                self._set_status(f"Orchestrator: starting '{task.title}'...")
                future = asyncio.create_task(self._run_worker(graph, task))
                pending_futures[task.id] = future

            # Emit progress.
            self._emit_output(
                "orchestrator",
                {"event": "progress", **graph.get_summary()},
                f"Graph: {graph.get_summary()}",
            )
            self._broadcast_event("progress")

        # Cancel any lingering futures.
        for tid, fut in pending_futures.items():
            if not fut.done():
                fut.cancel()
                try:
                    await fut
                except (asyncio.CancelledError, Exception):
                    pass

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
            agent = await self._pool.get_or_create(task.session_id)
            worker_prompt = self._instructions.render(
                "orchestrator_worker_prompt.md",
                task_title=task.title,
                task_description=task.description,
            )
            response = await asyncio.wait_for(
                agent.complete(worker_prompt),
                timeout=task.timeout_seconds,
            )
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
                "output": str(response or "").strip()[:500],
                "usage": usage,
                "context": {
                    "prompt_tokens": ctx.get("prompt_tokens", 0),
                    "budget": ctx.get("context_budget_tokens", 0),
                    "utilization": round(ctx.get("utilization", 0) * 100, 1),
                    "messages": ctx.get("included_messages", 0),
                },
            })
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
                self._provider.complete(messages=messages, tools=None, max_tokens=8000),
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
    # JSON parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_json_response(raw: str) -> dict[str, Any] | None:
        """Parse JSON from LLM response, handling markdown fences."""
        text = raw.strip()
        if not text:
            return None

        # Try direct parse.
        try:
            value = json.loads(text)
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            pass

        # Strip markdown code fences.
        fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
        if fence_match:
            try:
                value = json.loads(fence_match.group(1).strip())
                if isinstance(value, dict):
                    return value
            except json.JSONDecodeError:
                pass

        # Last resort: find first { ... } block.
        brace_match = re.search(r"\{.*\}", text, re.DOTALL)
        if brace_match:
            try:
                value = json.loads(brace_match.group(0))
                if isinstance(value, dict):
                    return value
            except json.JSONDecodeError:
                pass

        return None
