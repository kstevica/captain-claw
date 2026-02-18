"""Pipeline/progress helpers for Agent."""

from datetime import UTC, datetime
import json
import re
from typing import Any


class AgentPipelineMixin:
    """Pipeline construction, progress tracking, and completion checks."""
    @staticmethod
    def _iter_pipeline_nodes(tasks: list[dict[str, Any]]) -> Any:
        """Yield all task nodes in depth-first order."""
        for task in tasks:
            if not isinstance(task, dict):
                continue
            yield task
            children = task.get("children")
            if isinstance(children, list) and children:
                yield from AgentPipelineMixin._iter_pipeline_nodes(children)

    @staticmethod
    def _iter_pipeline_leaves(tasks: list[dict[str, Any]]) -> Any:
        """Yield leaf task nodes in depth-first order."""
        for task in tasks:
            if not isinstance(task, dict):
                continue
            children = task.get("children")
            if isinstance(children, list) and children:
                yield from AgentPipelineMixin._iter_pipeline_leaves(children)
                continue
            yield task

    @staticmethod
    def _set_all_pipeline_status(tasks: list[dict[str, Any]], status: str) -> None:
        """Set status recursively for all nodes in task tree."""
        for task in tasks:
            if not isinstance(task, dict):
                continue
            task["status"] = status
            children = task.get("children")
            if isinstance(children, list) and children:
                AgentPipelineMixin._set_all_pipeline_status(children, status)

    @staticmethod
    def _rollup_pipeline_status(tasks: list[dict[str, Any]]) -> None:
        """Roll up parent statuses from children."""
        def _node_status(node: dict[str, Any]) -> str:
            children = node.get("children")
            if not isinstance(children, list) or not children:
                return str(node.get("status", "pending")).strip().lower() or "pending"

            child_statuses = [_node_status(child) for child in children if isinstance(child, dict)]
            if not child_statuses:
                node["status"] = "pending"
                return "pending"
            if any(status == "failed" for status in child_statuses):
                node["status"] = "failed"
                return "failed"
            if any(status == "in_progress" for status in child_statuses):
                node["status"] = "in_progress"
                return "in_progress"
            if all(status == "completed" for status in child_statuses):
                node["status"] = "completed"
                return "completed"
            if any(status == "completed" for status in child_statuses):
                node["status"] = "in_progress"
                return "in_progress"
            node["status"] = "pending"
            return "pending"

        for task in tasks:
            if isinstance(task, dict):
                _node_status(task)

    @staticmethod
    def _refresh_pipeline_task_order(pipeline: dict[str, Any]) -> list[str]:
        """Refresh leaf-task execution order for pipeline state."""
        tasks = pipeline.get("tasks", [])
        if not isinstance(tasks, list) or not tasks:
            pipeline["task_order"] = []
            pipeline["current_task_id"] = ""
            pipeline["current_index"] = 0
            return []

        order: list[str] = []
        for leaf in AgentPipelineMixin._iter_pipeline_leaves(tasks):
            leaf_id = str(leaf.get("id", "")).strip()
            if not leaf_id:
                continue
            order.append(leaf_id)

        pipeline["task_order"] = order
        if not order:
            pipeline["current_task_id"] = ""
            pipeline["current_index"] = 0
            return []
        current_index = int(pipeline.get("current_index", 0))
        bounded = max(0, min(current_index, len(order) - 1))
        pipeline["current_index"] = bounded
        pipeline["current_task_id"] = order[bounded]
        return order

    @staticmethod
    def _find_pipeline_leaf_path(
        nodes: list[dict[str, Any]],
        target_leaf_id: str,
        prefix: list[int] | None = None,
    ) -> list[dict[str, Any]] | None:
        """Locate path metadata from root to a specific leaf task id."""
        base_prefix = list(prefix or [])
        for idx, node in enumerate(nodes, start=1):
            if not isinstance(node, dict):
                continue
            path_prefix = [*base_prefix, idx]
            path_label = ".".join(str(part) for part in path_prefix)
            entry = {
                "node": node,
                "scope_nodes": nodes,
                "index": idx,
                "siblings_total": len(nodes),
                "path": path_label,
            }
            children = node.get("children")
            if isinstance(children, list) and children:
                child_path = AgentPipelineMixin._find_pipeline_leaf_path(children, target_leaf_id, path_prefix)
                if child_path:
                    return [entry, *child_path]
                continue

            node_id = str(node.get("id", "")).strip()
            if node_id and node_id == target_leaf_id:
                return [entry]
        return None

    @staticmethod
    def _format_eta_seconds(value: float | int | None) -> str:
        """Format ETA seconds into concise human-readable text."""
        if value is None:
            return "unknown"
        seconds = max(0, int(round(float(value))))
        if seconds < 60:
            return f"{seconds}s"
        minutes, rem_seconds = divmod(seconds, 60)
        if minutes < 60:
            return f"{minutes}m {rem_seconds}s"
        hours, rem_minutes = divmod(minutes, 60)
        if hours < 24:
            return f"{hours}h {rem_minutes}m"
        days, rem_hours = divmod(hours, 24)
        return f"{days}d {rem_hours}h"

    @staticmethod
    def _build_pipeline_progress_details(pipeline: dict[str, Any]) -> dict[str, Any]:
        """Build progress details for monitor visibility across nested scopes."""
        tasks = pipeline.get("tasks", [])
        if not isinstance(tasks, list) or not tasks:
            return {
                "leaf_index": 0,
                "leaf_total": 0,
                "leaf_remaining": 0,
                "current_path": "",
                "scope_progress": [],
                "eta_seconds": None,
                "eta_text": "unknown",
                "elapsed_seconds": 0,
                "completed_leaves": 0,
            }

        task_order = pipeline.get("task_order")
        if not isinstance(task_order, list):
            task_order = AgentPipelineMixin._refresh_pipeline_task_order(pipeline)
        total_leaves = len(task_order)
        if total_leaves <= 0:
            return {
                "leaf_index": 0,
                "leaf_total": 0,
                "leaf_remaining": 0,
                "current_path": "",
                "scope_progress": [],
                "eta_seconds": None,
                "eta_text": "unknown",
                "elapsed_seconds": 0,
                "completed_leaves": 0,
            }

        current_index = int(pipeline.get("current_index", 0))
        bounded = max(0, min(current_index, total_leaves - 1))
        current_task_id = str(pipeline.get("current_task_id", "")).strip() or str(task_order[bounded])
        leaf_index = bounded + 1
        leaf_remaining = max(0, total_leaves - leaf_index)
        completed_leaves = sum(
            1
            for leaf in AgentPipelineMixin._iter_pipeline_leaves(tasks)
            if str(leaf.get("status", "")).strip().lower() == "completed"
        )
        created_at_raw = str(pipeline.get("created_at", "")).strip()
        elapsed_seconds = 0.0
        if created_at_raw:
            try:
                created_at = datetime.fromisoformat(created_at_raw)
                if created_at.tzinfo is None:
                    created_at = created_at.replace(tzinfo=UTC)
                elapsed_seconds = max(0.0, (datetime.now(UTC) - created_at.astimezone(UTC)).total_seconds())
            except Exception:
                elapsed_seconds = 0.0

        avg_seconds_per_leaf: float | None = None
        eta_seconds: float | None
        if leaf_remaining <= 0:
            eta_seconds = 0.0
        elif completed_leaves > 0 and elapsed_seconds > 0:
            avg_seconds_per_leaf = elapsed_seconds / max(1, completed_leaves)
            eta_seconds = avg_seconds_per_leaf * leaf_remaining
        else:
            eta_seconds = None

        path_entries = AgentPipelineMixin._find_pipeline_leaf_path(tasks, current_task_id) or []
        scopes: list[dict[str, Any]] = []
        for level, entry in enumerate(path_entries, start=1):
            scope_nodes = entry.get("scope_nodes")
            if not isinstance(scope_nodes, list):
                continue
            scope_leaf_ids: list[str] = []
            for leaf in AgentPipelineMixin._iter_pipeline_leaves(scope_nodes):
                leaf_id = str(leaf.get("id", "")).strip()
                if leaf_id:
                    scope_leaf_ids.append(leaf_id)
            scope_leaf_total = len(scope_leaf_ids)
            try:
                local_pos = scope_leaf_ids.index(current_task_id)
            except ValueError:
                local_pos = -1
            scope_leaf_remaining = (
                max(0, scope_leaf_total - (local_pos + 1))
                if local_pos >= 0
                else scope_leaf_total
            )

            siblings_total = int(entry.get("siblings_total", 0))
            sibling_index = int(entry.get("index", 0))
            scope_eta_seconds: float | None = None
            if scope_leaf_remaining <= 0:
                scope_eta_seconds = 0.0
            elif avg_seconds_per_leaf is not None:
                scope_eta_seconds = avg_seconds_per_leaf * scope_leaf_remaining
            scopes.append({
                "level": level,
                "path": str(entry.get("path", "")),
                "title": str(entry.get("node", {}).get("title", "")).strip(),
                "index": sibling_index,
                "siblings_total": siblings_total,
                "siblings_remaining": max(0, siblings_total - sibling_index),
                "scope_leaf_total": scope_leaf_total,
                "scope_leaf_remaining": scope_leaf_remaining,
                "eta_seconds": scope_eta_seconds,
                "eta_text": AgentPipelineMixin._format_eta_seconds(scope_eta_seconds),
            })

        current_path = str(path_entries[-1].get("path", "")) if path_entries else ""
        return {
            "leaf_index": leaf_index,
            "leaf_total": total_leaves,
            "leaf_remaining": leaf_remaining,
            "current_path": current_path,
            "scope_progress": scopes,
            "eta_seconds": eta_seconds,
            "eta_text": AgentPipelineMixin._format_eta_seconds(eta_seconds),
            "elapsed_seconds": int(round(elapsed_seconds)),
            "completed_leaves": completed_leaves,
        }

    def _build_task_pipeline(
        self,
        user_input: str,
        max_tasks: int = 6,
        tasks_override: list[dict[str, Any]] | None = None,
        completion_checks: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Build a nested task pipeline from user input or planner contract."""
        cleaned = re.sub(r"\s+", " ", (user_input or "").strip())
        tasks: list[dict[str, Any]] = []
        if tasks_override:
            tasks = self._normalize_contract_tasks(
                tasks_override,
                max_tasks=max_tasks,
                max_depth=4,
                max_total_nodes=max(24, max_tasks * 6),
            )
        else:
            parts = [
                piece.strip(" -")
                for piece in re.split(
                    r"(?:\n+|;|\. |\band then\b|\bthen\b|\bnext\b)",
                    cleaned,
                    flags=re.IGNORECASE,
                )
                if piece.strip(" -")
            ]
            fallback_raw = [{"title": part} for part in parts[:max_tasks]]
            tasks = self._normalize_contract_tasks(
                fallback_raw,
                max_tasks=max_tasks,
                max_depth=1,
                max_total_nodes=max_tasks,
            )

        if not tasks:
            tasks = [
                {"id": "task_1", "title": "Understand the request and constraints"},
                {"id": "task_2", "title": "Execute required actions/tools"},
                {"id": "task_3", "title": "Return concise final answer"},
            ]
        for node in self._iter_pipeline_nodes(tasks):
            node["status"] = "pending"

        if completion_checks:
            next_id = sum(1 for _ in self._iter_pipeline_nodes(tasks)) + 1
            tasks.append({
                "id": f"task_{next_id}",
                "title": "Run completion checks before finalizing the answer",
                "status": "pending",
            })

        normalized_checks: list[dict[str, Any]] = []
        for idx, check in enumerate(completion_checks or [], start=1):
            if not isinstance(check, dict):
                continue
            check_id = str(check.get("id", f"check_{idx}")).strip() or f"check_{idx}"
            title = str(check.get("title", check_id)).strip() or check_id
            normalized_checks.append({
                "id": check_id,
                "title": title,
                "status": "pending",
                "detail": "",
            })

        pipeline = {
            "created_at": datetime.now(UTC).isoformat(),
            "request": cleaned[:500],
            "tasks": tasks,
            "checks": normalized_checks,
            "task_order": [],
            "current_index": 0,
            "current_task_id": "",
            "state": "active",
        }
        self._refresh_pipeline_task_order(pipeline)
        self._set_pipeline_progress(pipeline, current_index=0, current_status="in_progress")
        return pipeline

    @staticmethod
    def _set_pipeline_progress(
        pipeline: dict[str, Any],
        current_index: int,
        current_status: str = "in_progress",
    ) -> None:
        """Set pipeline progress using leaf-task execution order."""
        tasks = pipeline.get("tasks", [])
        if not isinstance(tasks, list) or not tasks:
            return
        task_order = AgentPipelineMixin._refresh_pipeline_task_order(pipeline)
        if not task_order:
            return
        bounded = max(0, min(current_index, len(task_order) - 1))
        pipeline["current_index"] = bounded
        pipeline["current_task_id"] = task_order[bounded]

        leaf_status_map: dict[str, str] = {}
        for idx, task_id in enumerate(task_order):
            if idx < bounded:
                leaf_status_map[task_id] = "completed"
            elif idx == bounded:
                leaf_status_map[task_id] = current_status
            else:
                leaf_status_map[task_id] = "pending"

        for leaf in AgentPipelineMixin._iter_pipeline_leaves(tasks):
            leaf_id = str(leaf.get("id", "")).strip()
            if leaf_id in leaf_status_map:
                leaf["status"] = leaf_status_map[leaf_id]
        AgentPipelineMixin._rollup_pipeline_status(tasks)

    def _build_pipeline_note(self, pipeline: dict[str, Any]) -> str:
        """Build planning note injected into model context."""
        tasks = pipeline.get("tasks", [])
        if not isinstance(tasks, list) or not tasks:
            return ""

        def _render(nodes: list[dict[str, Any]], prefix: list[int]) -> list[str]:
            rendered: list[str] = []
            for idx, node in enumerate(nodes, start=1):
                if not isinstance(node, dict):
                    continue
                current_prefix = [*prefix, idx]
                label = ".".join(str(part) for part in current_prefix)
                title = str(node.get("title", "")).strip()
                status = str(node.get("status", "pending")).strip().upper()
                rendered.append(f"{label}. [{status}] {title}")
                children = node.get("children")
                if isinstance(children, list) and children:
                    rendered.extend(_render(children, current_prefix))
            return rendered

        lines = [self.instructions.load("planning_pipeline_header.md")]
        lines.extend(_render(tasks, []))
        lines.append(self.instructions.load("planning_pipeline_footer.md"))
        return "\n".join(lines)

    @staticmethod
    def _build_list_task_note(list_task_plan: dict[str, Any]) -> str:
        """Build list-memory note injected into model context."""
        if not isinstance(list_task_plan, dict) or not bool(list_task_plan.get("enabled", False)):
            return ""
        members = list_task_plan.get("members")
        if not isinstance(members, list) or not members:
            return ""
        strategy = str(list_task_plan.get("strategy", "direct")).strip().lower() or "direct"
        action = str(list_task_plan.get("per_member_action", "")).strip()
        lines = [
            "List task memory is active. You must process every extracted member before final response.",
            f"Strategy: {strategy}",
        ]
        if action:
            lines.append(f"Per-member action: {action}")
        lines.append("Members:")
        for idx, member in enumerate(members[:60], start=1):
            lines.append(f"{idx}. {member}")
        if len(members) > 60:
            lines.append(f"... (+{len(members) - 60} more members)")
        return "\n".join(lines)

    @staticmethod
    def _format_pipeline_monitor_output(event: str, pipeline: dict[str, Any]) -> str:
        """Format pipeline state for monitor output."""
        lines = [f"Planning event={event}"]
        lines.append(f"state={pipeline.get('state', 'active')}")
        progress = AgentPipelineMixin._build_pipeline_progress_details(pipeline)
        lines.append(
            "progress="
            f"{progress.get('leaf_index', 0)}/{progress.get('leaf_total', 0)} "
            f"remaining={progress.get('leaf_remaining', 0)}"
        )
        lines.append(
            f"eta={progress.get('eta_text', 'unknown')} "
            f"(elapsed={progress.get('elapsed_seconds', 0)}s completed={progress.get('completed_leaves', 0)})"
        )
        current_path = str(progress.get("current_path", "")).strip()
        if current_path:
            lines.append(f"current_path={current_path}")
        scope_progress = progress.get("scope_progress", [])
        if isinstance(scope_progress, list) and scope_progress:
            lines.append("scope_progress:")
            for scope in scope_progress:
                if not isinstance(scope, dict):
                    continue
                lines.append(
                    "- "
                    f"level={scope.get('level', '')} "
                    f"path={scope.get('path', '')} "
                    f"index={scope.get('index', 0)}/{scope.get('siblings_total', 0)} "
                    f"siblings_left={scope.get('siblings_remaining', 0)} "
                    f"leaves_left={scope.get('scope_leaf_remaining', 0)}/{scope.get('scope_leaf_total', 0)} "
                    f"eta={scope.get('eta_text', 'unknown')} "
                    f"title={scope.get('title', '')}"
                )

        def _render(nodes: list[dict[str, Any]], prefix: list[int]) -> list[str]:
            rendered: list[str] = []
            for idx, node in enumerate(nodes, start=1):
                if not isinstance(node, dict):
                    continue
                current_prefix = [*prefix, idx]
                label = ".".join(str(part) for part in current_prefix)
                status = str(node.get("status", "pending")).strip()
                title = str(node.get("title", "")).strip()
                rendered.append(f"- {label}. status={status} title={title}")
                children = node.get("children")
                if isinstance(children, list) and children:
                    rendered.extend(_render(children, current_prefix))
            return rendered

        tasks = pipeline.get("tasks", [])
        if isinstance(tasks, list) and tasks:
            lines.extend(_render(tasks, []))
        checks = pipeline.get("checks", [])
        if isinstance(checks, list) and checks:
            lines.append("checks:")
            for idx, check in enumerate(checks, start=1):
                if not isinstance(check, dict):
                    continue
                detail = str(check.get("detail", "")).strip()
                detail_suffix = f" detail={detail}" if detail else ""
                lines.append(
                    f"- {idx}. status={check.get('status', 'pending')} id={check.get('id', '')} "
                    f"title={check.get('title', '')}{detail_suffix}"
                )
        return "\n".join(lines)

    def _emit_pipeline_update(self, event: str, pipeline: dict[str, Any]) -> None:
        """Emit planning pipeline state into monitor output stream."""
        task_order = self._refresh_pipeline_task_order(pipeline)
        progress = self._build_pipeline_progress_details(pipeline)
        self._emit_tool_output(
            "planning",
            {
                "event": event,
                "enabled": self.planning_enabled,
                "mode": str(pipeline.get("mode", "manual")),
                "current_index": int(pipeline.get("current_index", 0)),
                "current_task_id": str(pipeline.get("current_task_id", "")),
                "leaf_tasks": len(task_order),
                "leaf_index": int(progress.get("leaf_index", 0)),
                "leaf_remaining": int(progress.get("leaf_remaining", 0)),
                "current_path": str(progress.get("current_path", "")),
                "eta_seconds": progress.get("eta_seconds"),
                "eta_text": str(progress.get("eta_text", "unknown")),
                "scope_progress": progress.get("scope_progress", []),
            },
            self._format_pipeline_monitor_output(event, pipeline),
        )

    def _advance_pipeline(self, pipeline: dict[str, Any], event: str = "advance") -> None:
        """Advance pipeline to next leaf task if possible."""
        task_order = self._refresh_pipeline_task_order(pipeline)
        if not task_order:
            return
        current_index = int(pipeline.get("current_index", 0))
        bounded = max(0, min(current_index, len(task_order) - 1))
        if bounded >= len(task_order) - 1:
            self._set_pipeline_progress(pipeline, current_index=bounded, current_status="in_progress")
            self._emit_pipeline_update(event, pipeline)
            return
        self._set_pipeline_progress(pipeline, current_index=bounded + 1, current_status="in_progress")
        self._emit_pipeline_update(event, pipeline)

    def _finalize_pipeline(self, pipeline: dict[str, Any], success: bool = True) -> None:
        """Finalize pipeline statuses at turn completion."""
        tasks = pipeline.get("tasks", [])
        if not isinstance(tasks, list):
            return
        task_order = self._refresh_pipeline_task_order(pipeline)
        if success:
            self._set_all_pipeline_status(tasks, "completed")
            AgentPipelineMixin._rollup_pipeline_status(tasks)
            if task_order:
                pipeline["current_index"] = len(task_order) - 1
                pipeline["current_task_id"] = task_order[-1]
            pipeline["state"] = "completed"
            self._emit_pipeline_update("completed", pipeline)
            return

        if task_order:
            current_index = int(pipeline.get("current_index", 0))
            bounded = max(0, min(current_index, len(task_order) - 1))
            self._set_pipeline_progress(pipeline, current_index=bounded, current_status="failed")
        else:
            self._set_all_pipeline_status(tasks, "failed")
            AgentPipelineMixin._rollup_pipeline_status(tasks)
        pipeline["state"] = "failed"
        self._emit_pipeline_update("failed", pipeline)

    @staticmethod
    def _update_pipeline_checks(
        pipeline: dict[str, Any],
        check_results: list[dict[str, Any]],
    ) -> None:
        """Apply completion-check statuses onto active pipeline."""
        checks = pipeline.get("checks", [])
        if not isinstance(checks, list) or not checks:
            return
        result_map: dict[str, dict[str, Any]] = {}
        for result in check_results:
            if not isinstance(result, dict):
                continue
            key = str(result.get("id", "")).strip()
            if not key:
                continue
            result_map[key] = result

        for check in checks:
            if not isinstance(check, dict):
                continue
            key = str(check.get("id", "")).strip()
            data = result_map.get(key)
            if not data:
                continue
            if bool(data.get("ok", False)):
                check["status"] = "passed"
                check["detail"] = ""
            else:
                check["status"] = "failed"
                check["detail"] = str(data.get("reason", "")).strip()

    def _compute_turn_iteration_budget(
        self,
        base_iterations: int,
        planning_pipeline: dict[str, Any] | None,
        completion_requirements: list[dict[str, Any]] | None,
    ) -> int:
        """Compute adaptive iteration budget from task complexity."""
        budget = max(1, int(base_iterations))
        leaf_tasks = 0
        if isinstance(planning_pipeline, dict):
            task_order = planning_pipeline.get("task_order")
            if isinstance(task_order, list):
                leaf_tasks = len(task_order)
            else:
                tasks = planning_pipeline.get("tasks", [])
                if isinstance(tasks, list):
                    leaf_tasks = sum(1 for _ in self._iter_pipeline_leaves(tasks))
        if leaf_tasks > 0:
            budget = max(budget, min(120, 4 + (leaf_tasks * 3)))

        req_count = len(completion_requirements or [])
        if req_count > 0:
            budget = min(140, budget + min(20, req_count))
        return max(budget, max(1, int(base_iterations)))

    def _capture_turn_progress_snapshot(
        self,
        turn_start_idx: int,
        planning_pipeline: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Capture compact progress snapshot for stuck/progress detection."""
        successful_tool_count = 0
        write_success_count = 0
        tool_signatures: set[str] = set()
        assistant_signatures: set[str] = set()
        if self.session:
            for msg in self.session.messages[turn_start_idx:]:
                role = str(msg.get("role", "")).strip().lower()
                if role == "assistant":
                    content = re.sub(r"\s+", " ", str(msg.get("content", "")).strip())
                    if content:
                        assistant_signatures.add(content[:800])
                    continue
                if str(msg.get("role", "")).strip().lower() != "tool":
                    continue
                tool_name = str(msg.get("tool_name", "")).strip().lower()
                if self._is_monitor_only_tool_name(tool_name):
                    continue
                content = str(msg.get("content", ""))
                if not content.strip().lower().startswith("error:"):
                    successful_tool_count += 1
                    if tool_name == "write":
                        write_success_count += 1
                args = msg.get("tool_arguments")
                try:
                    args_text = json.dumps(args, sort_keys=True, ensure_ascii=True) if isinstance(args, dict) else "{}"
                except Exception:
                    args_text = "{}"
                tool_signatures.add(f"{tool_name}|{args_text}")

        pipeline_index = -1
        pipeline_task_id = ""
        if isinstance(planning_pipeline, dict):
            pipeline_index = int(planning_pipeline.get("current_index", -1))
            pipeline_task_id = str(planning_pipeline.get("current_task_id", "")).strip()

        return {
            "successful_tools": successful_tool_count,
            "write_success": write_success_count,
            "unique_tool_signatures": len(tool_signatures),
            "unique_assistant_signatures": len(assistant_signatures),
            "pipeline_index": pipeline_index,
            "pipeline_task_id": pipeline_task_id,
        }

    @staticmethod
    def _has_turn_progress(previous: dict[str, Any], current: dict[str, Any]) -> bool:
        """Whether turn made meaningful progress between snapshots."""
        if int(current.get("write_success", 0)) > int(previous.get("write_success", 0)):
            return True
        if int(current.get("unique_tool_signatures", 0)) > int(previous.get("unique_tool_signatures", 0)):
            return True
        if int(current.get("unique_assistant_signatures", 0)) > int(previous.get("unique_assistant_signatures", 0)):
            return True
        prev_task = str(previous.get("pipeline_task_id", "")).strip()
        curr_task = str(current.get("pipeline_task_id", "")).strip()
        if curr_task and curr_task != prev_task:
            return True
        if int(current.get("pipeline_index", -1)) > int(previous.get("pipeline_index", -1)):
            return True
        return False

    def _pipeline_has_remaining_work(self, planning_pipeline: dict[str, Any] | None) -> bool:
        """Whether pipeline still has unfinished tasks/checks."""
        if not isinstance(planning_pipeline, dict):
            return False
        tasks = planning_pipeline.get("tasks", [])
        if isinstance(tasks, list):
            for leaf in self._iter_pipeline_leaves(tasks):
                status = str(leaf.get("status", "pending")).strip().lower()
                if status not in {"completed"}:
                    return True
        checks = planning_pipeline.get("checks", [])
        if isinstance(checks, list):
            for check in checks:
                if not isinstance(check, dict):
                    continue
                status = str(check.get("status", "pending")).strip().lower()
                if status not in {"passed"}:
                    return True
        return False

