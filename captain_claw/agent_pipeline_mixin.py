"""Pipeline/progress helpers for Agent."""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
import json
import re
from typing import Any


class TaskStatus(str, Enum):
    """Supported task lifecycle states."""

    PENDING = "pending"
    READY = "ready"
    BLOCKED = "blocked"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"


TERMINAL_TASK_STATUSES = {
    TaskStatus.COMPLETED.value,
    TaskStatus.FAILED.value,
    TaskStatus.CANCELLED.value,
}


@dataclass(slots=True)
class TaskResult:
    """Structured task result payload."""

    success: bool
    output: str = ""
    error: str = ""
    completed_at: str = ""


@dataclass(slots=True)
class ExecutionContext:
    """Isolated context metadata for a single task/subagent run."""

    context_id: str
    session_id: str = ""
    spawned_by: str = ""
    parent_task_id: str = ""
    spawn_depth: int = 0
    allow_agents: list[str] = field(default_factory=lambda: ["*"])
    active_children: int = 0
    max_children: int = 5
    token_usage: dict[str, int] = field(
        default_factory=lambda: {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
    )
    compaction_count: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class TaskNode:
    """DAG-capable task node schema."""

    id: str
    title: str
    status: TaskStatus = TaskStatus.PENDING
    depends_on: list[str] = field(default_factory=list)
    children: list["TaskNode"] = field(default_factory=list)
    execution_context: ExecutionContext = field(default_factory=lambda: ExecutionContext(context_id="context"))
    result: TaskResult | None = None
    retries: int = 0
    max_retries: int = 2
    timeout_seconds: float | None = None
    tool_policy: dict[str, Any] | None = None


class AgentPipelineMixin:
    """Pipeline construction, progress tracking, and completion checks."""

    @staticmethod
    def _normalize_task_status(value: str | TaskStatus | None) -> str:
        text = str(value.value if isinstance(value, TaskStatus) else value or "").strip().lower()
        if text in {status.value for status in TaskStatus}:
            return text
        return TaskStatus.PENDING.value

    @staticmethod
    def _parse_pipeline_timestamp(value: str | None) -> datetime | None:
        raw = str(value or "").strip()
        if not raw:
            return None
        try:
            parsed = datetime.fromisoformat(raw)
        except Exception:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return parsed.astimezone(UTC)

    @staticmethod
    def _coerce_timeout_seconds(raw_value: Any, default_seconds: float | None = 180.0) -> float | None:
        if raw_value in {None, ""}:
            return default_seconds
        try:
            timeout = float(raw_value)
        except Exception:
            return default_seconds
        if timeout <= 0:
            return None
        return timeout

    @staticmethod
    def _coerce_tool_policy(raw_value: Any) -> dict[str, Any] | None:
        """Normalize optional task tool-policy payload."""
        if not isinstance(raw_value, dict):
            return None

        allow_raw = raw_value.get("allow")
        if allow_raw is None:
            allow: list[str] | None = None
        elif isinstance(allow_raw, list):
            allow = [str(item).strip() for item in allow_raw if str(item).strip()]
        else:
            return None

        deny_raw = raw_value.get("deny", [])
        deny = [str(item).strip() for item in deny_raw] if isinstance(deny_raw, list) else []
        deny = [item for item in deny if item]

        also_allow_raw = raw_value.get("also_allow", raw_value.get("alsoAllow", []))
        also_allow = (
            [str(item).strip() for item in also_allow_raw]
            if isinstance(also_allow_raw, list)
            else []
        )
        also_allow = [item for item in also_allow if item]

        if allow is None and not deny and not also_allow:
            return None
        return {
            "allow": allow,
            "deny": deny,
            "also_allow": also_allow,
        }

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
    def _iter_pipeline_nodes_with_parent(
        tasks: list[dict[str, Any]],
        parent_id: str = "",
    ) -> Any:
        """Yield nodes with parent task id metadata."""
        for task in tasks:
            if not isinstance(task, dict):
                continue
            task_id = str(task.get("id", "")).strip()
            yield task, parent_id
            children = task.get("children")
            if isinstance(children, list) and children and task_id:
                yield from AgentPipelineMixin._iter_pipeline_nodes_with_parent(children, parent_id=task_id)

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

    def _build_default_execution_context(
        self,
        task_id: str,
        *,
        parent_task_id: str = "",
        depth: int = 1,
    ) -> dict[str, Any]:
        """Create isolated execution context metadata for a task node."""
        parent_session_id = ""
        parent_spawn_depth = 0
        if self.session and isinstance(self.session.metadata, dict):
            parent_session_id = str(self.session.id)
            subagent_meta = self.session.metadata.get("subagent")
            if isinstance(subagent_meta, dict):
                parent_spawn_depth = max(0, int(subagent_meta.get("spawn_depth", 0)))

        context = ExecutionContext(
            context_id=f"context_{task_id}",
            spawned_by=parent_session_id,
            parent_task_id=parent_task_id,
            spawn_depth=parent_spawn_depth + max(1, depth),
        )
        return {
            "context_id": context.context_id,
            "session_id": context.session_id,
            "spawned_by": context.spawned_by,
            "parent_task_id": context.parent_task_id,
            "spawn_depth": context.spawn_depth,
            "allow_agents": list(context.allow_agents),
            "active_children": context.active_children,
            "max_children": context.max_children,
            "token_usage": dict(context.token_usage),
            "compaction_count": context.compaction_count,
            "history": list(context.history),
        }

    def _sanitize_pipeline_nodes(
        self,
        tasks: list[dict[str, Any]],
        *,
        max_total_nodes: int = 96,
    ) -> list[dict[str, Any]]:
        """Schema-validate + normalize task nodes into a DAG-compatible shape."""
        seen_ids: set[str] = set()
        next_id = 0

        def _unique_task_id(raw_id: str) -> str:
            nonlocal next_id
            candidate = re.sub(r"[^a-zA-Z0-9_]+", "_", raw_id).strip("_")
            if not candidate:
                next_id += 1
                candidate = f"task_{next_id}"
            base = candidate
            suffix = 2
            while candidate in seen_ids:
                candidate = f"{base}_{suffix}"
                suffix += 1
            seen_ids.add(candidate)
            return candidate

        def _visit(node: Any, depth: int, parent_task_id: str) -> dict[str, Any] | None:
            nonlocal next_id
            if len(seen_ids) >= max_total_nodes:
                return None
            if not isinstance(node, dict):
                return None
            title = str(node.get("title", "")).strip()
            if not title:
                return None
            next_id += 1
            task_id = _unique_task_id(str(node.get("id", "")).strip() or f"task_{next_id}")
            depends_on_raw = node.get("depends_on")
            if depends_on_raw is None:
                depends_on_raw = node.get("depends")
            depends_on: list[str] = []
            if isinstance(depends_on_raw, list):
                for dep in depends_on_raw:
                    dep_id = re.sub(r"[^a-zA-Z0-9_]+", "_", str(dep or "").strip()).strip("_")
                    if dep_id and dep_id not in depends_on:
                        depends_on.append(dep_id)
            execution_context = self._build_default_execution_context(
                task_id,
                parent_task_id=parent_task_id,
                depth=depth,
            )
            raw_context = node.get("execution_context")
            if isinstance(raw_context, dict):
                for key in ("session_id", "spawned_by", "parent_task_id", "spawn_depth"):
                    if key in raw_context:
                        execution_context[key] = raw_context[key]
                token_usage = raw_context.get("token_usage")
                if isinstance(token_usage, dict):
                    bucket = execution_context.setdefault(
                        "token_usage",
                        {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                    )
                    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                        bucket[key] = int(token_usage.get(key, bucket.get(key, 0)) or 0)
            prepared: dict[str, Any] = {
                "id": task_id,
                "title": title[:220],
                "status": self._normalize_task_status(node.get("status", TaskStatus.PENDING.value)),
                "depends_on": depends_on,
                "retries": max(0, int(node.get("retries", 0))),
                "max_retries": max(0, int(node.get("max_retries", 2))),
                "timeout_seconds": self._coerce_timeout_seconds(node.get("timeout_seconds")),
                "execution_context": execution_context,
                "result": node.get("result") if isinstance(node.get("result"), dict) else None,
            }
            tool_policy = self._coerce_tool_policy(node.get("tool_policy"))
            if tool_policy is not None:
                prepared["tool_policy"] = tool_policy
            raw_children = node.get("children")
            child_nodes: list[dict[str, Any]] = []
            if isinstance(raw_children, list):
                for child in raw_children:
                    normalized_child = _visit(child, depth + 1, task_id)
                    if normalized_child:
                        child_nodes.append(normalized_child)
            if child_nodes:
                prepared["children"] = child_nodes
            return prepared

        normalized: list[dict[str, Any]] = []
        for task in tasks:
            prepared_task = _visit(task, depth=1, parent_task_id="")
            if prepared_task:
                normalized.append(prepared_task)
        return normalized

    @staticmethod
    def _set_all_pipeline_status(tasks: list[dict[str, Any]], status: str) -> None:
        """Set status recursively for all nodes in task tree."""
        normalized = AgentPipelineMixin._normalize_task_status(status)
        for task in tasks:
            if not isinstance(task, dict):
                continue
            task["status"] = normalized
            children = task.get("children")
            if isinstance(children, list) and children:
                AgentPipelineMixin._set_all_pipeline_status(children, normalized)

    @staticmethod
    def _rollup_pipeline_status(tasks: list[dict[str, Any]]) -> None:
        """Roll up parent statuses from children."""

        def _node_status(node: dict[str, Any]) -> str:
            children = node.get("children")
            if not isinstance(children, list) or not children:
                current = AgentPipelineMixin._normalize_task_status(node.get("status", TaskStatus.PENDING.value))
                node["status"] = current
                return current

            child_statuses = [_node_status(child) for child in children if isinstance(child, dict)]
            if not child_statuses:
                node["status"] = TaskStatus.PENDING.value
                return TaskStatus.PENDING.value
            if any(status in {TaskStatus.FAILED.value, TaskStatus.TIMED_OUT.value} for status in child_statuses):
                node["status"] = TaskStatus.FAILED.value
                return TaskStatus.FAILED.value
            if any(status == TaskStatus.CANCELLED.value for status in child_statuses):
                node["status"] = TaskStatus.CANCELLED.value
                return TaskStatus.CANCELLED.value
            if all(status == TaskStatus.COMPLETED.value for status in child_statuses):
                node["status"] = TaskStatus.COMPLETED.value
                return TaskStatus.COMPLETED.value
            if any(status == TaskStatus.IN_PROGRESS.value for status in child_statuses):
                node["status"] = TaskStatus.IN_PROGRESS.value
                return TaskStatus.IN_PROGRESS.value
            if any(status == TaskStatus.READY.value for status in child_statuses):
                node["status"] = TaskStatus.READY.value
                return TaskStatus.READY.value
            if any(status == TaskStatus.BLOCKED.value for status in child_statuses):
                node["status"] = TaskStatus.BLOCKED.value
                return TaskStatus.BLOCKED.value
            if any(status == TaskStatus.COMPLETED.value for status in child_statuses):
                node["status"] = TaskStatus.IN_PROGRESS.value
                return TaskStatus.IN_PROGRESS.value
            node["status"] = TaskStatus.PENDING.value
            return TaskStatus.PENDING.value

        for task in tasks:
            if isinstance(task, dict):
                _node_status(task)

    @staticmethod
    def _topological_task_order(
        leaf_ids: list[str],
        dependencies: dict[str, list[str]],
        priority_map: dict[str, int],
    ) -> list[str]:
        """Build stable topological order for executable task ids."""
        dep_map: dict[str, set[str]] = {task_id: set() for task_id in leaf_ids}
        reverse_map: dict[str, set[str]] = {task_id: set() for task_id in leaf_ids}
        for task_id in leaf_ids:
            for dep_id in dependencies.get(task_id, []):
                if dep_id not in dep_map:
                    continue
                dep_map[task_id].add(dep_id)
                reverse_map[dep_id].add(task_id)
        ready = sorted(
            [task_id for task_id, deps in dep_map.items() if not deps],
            key=lambda item: priority_map.get(item, 1_000_000),
        )
        order: list[str] = []
        while ready:
            current = ready.pop(0)
            order.append(current)
            for dependent in sorted(reverse_map.get(current, set()), key=lambda item: priority_map.get(item, 1_000_000)):
                pending = dep_map.get(dependent)
                if not pending:
                    continue
                pending.discard(current)
                if not pending and dependent not in order and dependent not in ready:
                    ready.append(dependent)
            ready.sort(key=lambda item: priority_map.get(item, 1_000_000))

        if len(order) < len(leaf_ids):
            remaining = [task_id for task_id in leaf_ids if task_id not in order]
            remaining.sort(key=lambda item: priority_map.get(item, 1_000_000))
            order.extend(remaining)
        return order

    @staticmethod
    def _refresh_pipeline_graph_state(pipeline: dict[str, Any]) -> list[str]:
        """Refresh DAG task indexes, dependency state, and ready/active sets."""
        tasks = pipeline.get("tasks", [])
        if not isinstance(tasks, list) or not tasks:
            pipeline["task_graph"] = {}
            pipeline["task_dependencies"] = {}
            pipeline["task_order"] = []
            pipeline["ready_task_ids"] = []
            pipeline["blocked_task_ids"] = []
            pipeline["active_task_ids"] = []
            pipeline["completed_task_ids"] = []
            pipeline["failed_task_ids"] = []
            pipeline["current_task_id"] = ""
            pipeline["current_index"] = 0
            return []

        graph: dict[str, dict[str, Any]] = {}
        parent_by_id: dict[str, str] = {}
        for node, parent_id in AgentPipelineMixin._iter_pipeline_nodes_with_parent(tasks):
            task_id = str(node.get("id", "")).strip()
            if not task_id:
                continue
            graph[task_id] = node
            if parent_id:
                parent_by_id[task_id] = parent_id

        leaf_priority: dict[str, int] = {}
        leaf_ids: list[str] = []
        for idx, leaf in enumerate(AgentPipelineMixin._iter_pipeline_leaves(tasks), start=1):
            leaf_id = str(leaf.get("id", "")).strip()
            if not leaf_id:
                continue
            leaf_ids.append(leaf_id)
            leaf_priority[leaf_id] = idx
        leaf_id_set = set(leaf_ids)

        descendant_leaf_cache: dict[str, list[str]] = {}

        def _descendant_leaves(task_id: str) -> list[str]:
            cached = descendant_leaf_cache.get(task_id)
            if cached is not None:
                return cached
            node = graph.get(task_id)
            if not isinstance(node, dict):
                descendant_leaf_cache[task_id] = []
                return []
            children = node.get("children")
            if not isinstance(children, list) or not children:
                descendant_leaf_cache[task_id] = [task_id] if task_id in leaf_id_set else []
                return descendant_leaf_cache[task_id]
            leaves: list[str] = []
            for child in children:
                if not isinstance(child, dict):
                    continue
                child_id = str(child.get("id", "")).strip()
                if not child_id:
                    continue
                leaves.extend(_descendant_leaves(child_id))
            deduped: list[str] = []
            seen: set[str] = set()
            for leaf_id in leaves:
                if leaf_id in seen:
                    continue
                seen.add(leaf_id)
                deduped.append(leaf_id)
            descendant_leaf_cache[task_id] = deduped
            return deduped

        dependencies: dict[str, list[str]] = {}
        for leaf_id in leaf_ids:
            dep_ids: set[str] = set()
            cursor = leaf_id
            while cursor:
                node = graph.get(cursor)
                if not isinstance(node, dict):
                    break
                raw_depends = node.get("depends_on")
                if isinstance(raw_depends, list):
                    for dep in raw_depends:
                        dep_id = str(dep or "").strip()
                        if dep_id:
                            dep_ids.add(dep_id)
                cursor = parent_by_id.get(cursor, "")
            expanded: set[str] = set()
            for dep_id in dep_ids:
                if dep_id == leaf_id:
                    continue
                if dep_id in graph:
                    for dep_leaf in _descendant_leaves(dep_id):
                        if dep_leaf and dep_leaf != leaf_id:
                            expanded.add(dep_leaf)
                    continue
                if dep_id in leaf_id_set:
                    expanded.add(dep_id)
            deps_sorted = sorted(expanded, key=lambda item: leaf_priority.get(item, 1_000_000))
            dependencies[leaf_id] = deps_sorted

        task_order = AgentPipelineMixin._topological_task_order(leaf_ids, dependencies, leaf_priority)
        order_pos = {task_id: idx for idx, task_id in enumerate(task_order)}

        completed_ids: list[str] = []
        failed_ids: list[str] = []
        active_ids: list[str] = []
        ready_ids: list[str] = []
        blocked_ids: list[str] = []
        for leaf_id in task_order:
            node = graph.get(leaf_id)
            if not isinstance(node, dict):
                continue
            status = AgentPipelineMixin._normalize_task_status(node.get("status", TaskStatus.PENDING.value))
            deps = dependencies.get(leaf_id, [])
            deps_satisfied = all(
                AgentPipelineMixin._normalize_task_status(
                    graph.get(dep_id, {}).get("status", TaskStatus.PENDING.value)
                )
                == TaskStatus.COMPLETED.value
                for dep_id in deps
            )
            if status in {TaskStatus.FAILED.value, TaskStatus.TIMED_OUT.value}:
                failed_ids.append(leaf_id)
                node["status"] = TaskStatus.FAILED.value
                continue
            if status == TaskStatus.CANCELLED.value:
                failed_ids.append(leaf_id)
                continue
            if status == TaskStatus.COMPLETED.value:
                completed_ids.append(leaf_id)
                continue
            if status == TaskStatus.IN_PROGRESS.value:
                active_ids.append(leaf_id)
                continue
            if deps_satisfied:
                node["status"] = TaskStatus.READY.value
                ready_ids.append(leaf_id)
            else:
                node["status"] = TaskStatus.BLOCKED.value
                blocked_ids.append(leaf_id)

        AgentPipelineMixin._rollup_pipeline_status(tasks)

        pipeline["task_graph"] = graph
        pipeline["task_dependencies"] = dependencies
        pipeline["task_order"] = task_order
        pipeline["ready_task_ids"] = ready_ids
        pipeline["blocked_task_ids"] = blocked_ids
        pipeline["active_task_ids"] = active_ids
        pipeline["completed_task_ids"] = completed_ids
        pipeline["failed_task_ids"] = failed_ids

        current_id = str(pipeline.get("current_task_id", "")).strip()
        if active_ids:
            active_sorted = sorted(active_ids, key=lambda item: order_pos.get(item, 1_000_000))
            current_id = active_sorted[0]
        elif current_id not in order_pos:
            if ready_ids:
                current_id = sorted(ready_ids, key=lambda item: order_pos.get(item, 1_000_000))[0]
            elif task_order:
                current_id = task_order[min(len(task_order) - 1, len(completed_ids))]
            else:
                current_id = ""
        pipeline["current_task_id"] = current_id
        if current_id and current_id in order_pos:
            pipeline["current_index"] = order_pos[current_id]
        else:
            pipeline["current_index"] = 0
        return task_order

    @staticmethod
    def _refresh_pipeline_task_order(pipeline: dict[str, Any]) -> list[str]:
        """Refresh DAG task execution order for pipeline state."""
        return AgentPipelineMixin._refresh_pipeline_graph_state(pipeline)

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
                "ready_leaves": 0,
                "active_leaves": 0,
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
                "ready_leaves": 0,
                "active_leaves": 0,
            }

        order_pos = {task_id: idx for idx, task_id in enumerate(task_order)}
        current_task_id = str(pipeline.get("current_task_id", "")).strip()
        if current_task_id not in order_pos:
            current_task_id = task_order[0]
        bounded = max(0, min(order_pos.get(current_task_id, 0), total_leaves - 1))
        leaf_index = bounded + 1
        completed_leaves = len(list(pipeline.get("completed_task_ids", [])))
        ready_leaves = len(list(pipeline.get("ready_task_ids", [])))
        active_leaves = len(list(pipeline.get("active_task_ids", [])))
        leaf_remaining = max(0, total_leaves - leaf_index)
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
            scopes.append(
                {
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
                }
            )

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
            "ready_leaves": ready_leaves,
            "active_leaves": active_leaves,
        }

    def _activate_pipeline_tasks(self, pipeline: dict[str, Any], *, event: str = "activate") -> list[str]:
        """Activate next ready tasks up to max parallel task limit."""
        task_order = self._refresh_pipeline_graph_state(pipeline)
        if not task_order:
            return []
        active = list(pipeline.get("active_task_ids", []))
        ready = list(pipeline.get("ready_task_ids", []))
        max_parallel = max(1, int(pipeline.get("max_parallel_branches", 2)))
        available_slots = max(0, max_parallel - len(active))
        if available_slots <= 0:
            pipeline["recently_activated_task_ids"] = []
            return []

        graph = pipeline.get("task_graph", {})
        if not isinstance(graph, dict):
            pipeline["recently_activated_task_ids"] = []
            return []
        ready_ordered = [task_id for task_id in task_order if task_id in ready]
        now_iso = datetime.now(UTC).isoformat()
        activated: list[str] = []
        for task_id in ready_ordered:
            if available_slots <= 0:
                break
            node = graph.get(task_id)
            if not isinstance(node, dict):
                continue
            status = self._normalize_task_status(node.get("status", TaskStatus.PENDING.value))
            if status not in {TaskStatus.READY.value, TaskStatus.PENDING.value, TaskStatus.BLOCKED.value}:
                continue
            node["status"] = TaskStatus.IN_PROGRESS.value
            node.setdefault("started_at", now_iso)
            node["last_heartbeat_at"] = now_iso
            active.append(task_id)
            activated.append(task_id)
            available_slots -= 1

        pipeline["recently_activated_task_ids"] = activated
        self._refresh_pipeline_graph_state(pipeline)
        self._rollup_pipeline_status(pipeline.get("tasks", []))
        return activated

    def _record_pipeline_task_usage(
        self,
        pipeline: dict[str, Any] | None,
        usage: dict[str, Any] | None,
    ) -> None:
        """Accumulate token usage counters on active task execution contexts."""
        if not isinstance(pipeline, dict) or not isinstance(usage, dict):
            return
        graph = pipeline.get("task_graph")
        active_ids = pipeline.get("active_task_ids")
        if not isinstance(graph, dict) or not isinstance(active_ids, list):
            return
        prompt = int(usage.get("prompt_tokens", 0) or 0)
        completion = int(usage.get("completion_tokens", 0) or 0)
        total = int(usage.get("total_tokens", prompt + completion) or 0)
        for task_id in active_ids:
            node = graph.get(task_id)
            if not isinstance(node, dict):
                continue
            execution_context = node.get("execution_context")
            if not isinstance(execution_context, dict):
                execution_context = self._build_default_execution_context(task_id)
                node["execution_context"] = execution_context
            token_usage = execution_context.setdefault(
                "token_usage",
                {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            )
            token_usage["prompt_tokens"] = int(token_usage.get("prompt_tokens", 0)) + prompt
            token_usage["completion_tokens"] = int(token_usage.get("completion_tokens", 0)) + completion
            token_usage["total_tokens"] = int(token_usage.get("total_tokens", 0)) + total

    def _build_task_pipeline(
        self,
        user_input: str,
        max_tasks: int = 6,
        tasks_override: list[dict[str, Any]] | None = None,
        completion_checks: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Build a DAG-backed task pipeline from contract tasks."""
        cleaned = re.sub(r"\s+", " ", (user_input or "").strip())
        tasks: list[dict[str, Any]] = []
        if tasks_override:
            tasks = self._normalize_contract_tasks(
                tasks_override,
                max_tasks=max_tasks,
                max_depth=12,
                max_total_nodes=max(36, max_tasks * 8),
            )
        else:
            extractor = getattr(self, "_extract_json_object", None)
            if callable(extractor):
                payload = extractor(cleaned)
                if isinstance(payload, dict):
                    tasks = self._normalize_contract_tasks(
                        payload.get("tasks"),
                        max_tasks=max_tasks,
                        max_depth=12,
                        max_total_nodes=max(36, max_tasks * 8),
                    )
            if not tasks:
                default_contract_builder = getattr(self, "_default_task_contract", None)
                if callable(default_contract_builder):
                    default_contract = default_contract_builder(cleaned)
                    if isinstance(default_contract, dict):
                        tasks = self._normalize_contract_tasks(
                            default_contract.get("tasks"),
                            max_tasks=max_tasks,
                            max_depth=8,
                            max_total_nodes=max(24, max_tasks * 6),
                        )

        if not tasks:
            tasks = [
                {"id": "task_1", "title": "Understand the request and constraints"},
                {"id": "task_2", "title": "Execute required actions/tools"},
                {"id": "task_3", "title": "Return concise final answer"},
            ]
        tasks = self._sanitize_pipeline_nodes(tasks, max_total_nodes=max(96, max_tasks * 10))

        if completion_checks:
            existing_leaf_ids = [
                str(leaf.get("id", "")).strip()
                for leaf in self._iter_pipeline_leaves(tasks)
                if isinstance(leaf, dict) and str(leaf.get("id", "")).strip()
            ]
            next_id = sum(1 for _ in self._iter_pipeline_nodes(tasks)) + 1
            tasks.append(
                {
                    "id": f"task_{next_id}",
                    "title": "Run completion checks before finalizing the answer",
                    "status": TaskStatus.PENDING.value,
                    "depends_on": existing_leaf_ids,
                    "retries": 0,
                    "max_retries": 0,
                    "timeout_seconds": 90.0,
                    "execution_context": self._build_default_execution_context(f"task_{next_id}"),
                    "result": None,
                }
            )

        normalized_checks: list[dict[str, Any]] = []
        for idx, check in enumerate(completion_checks or [], start=1):
            if not isinstance(check, dict):
                continue
            check_id = str(check.get("id", f"check_{idx}")).strip() or f"check_{idx}"
            title = str(check.get("title", check_id)).strip() or check_id
            normalized_checks.append(
                {
                    "id": check_id,
                    "title": title,
                    "status": "pending",
                    "detail": "",
                }
            )

        pipeline = {
            "created_at": datetime.now(UTC).isoformat(),
            "request": cleaned[:500],
            "tasks": tasks,
            "checks": normalized_checks,
            "task_order": [],
            "task_graph": {},
            "task_dependencies": {},
            "ready_task_ids": [],
            "blocked_task_ids": [],
            "active_task_ids": [],
            "completed_task_ids": [],
            "failed_task_ids": [],
            "current_index": 0,
            "current_task_id": "",
            "recently_activated_task_ids": [],
            "state": "active",
            "max_parallel_branches": 2,
            "subagents": {
                "enabled": True,
                "max_spawn_depth": 2,
                "max_active_children": 5,
                "allow_agents": ["*"],
                "active_child_session_ids": [],
            },
        }
        self._refresh_pipeline_graph_state(pipeline)
        self._activate_pipeline_tasks(pipeline, event="created")
        self._refresh_pipeline_graph_state(pipeline)
        return pipeline

    @staticmethod
    def _set_pipeline_progress(
        pipeline: dict[str, Any],
        current_index: int,
        current_status: str = "in_progress",
    ) -> None:
        """Set pipeline progress using execution order for compatibility paths."""
        tasks = pipeline.get("tasks", [])
        if not isinstance(tasks, list) or not tasks:
            return
        task_order = AgentPipelineMixin._refresh_pipeline_task_order(pipeline)
        if not task_order:
            return
        bounded = max(0, min(current_index, len(task_order) - 1))
        pipeline["current_index"] = bounded
        pipeline["current_task_id"] = task_order[bounded]

        target_status = AgentPipelineMixin._normalize_task_status(current_status)
        leaf_status_map: dict[str, str] = {}
        for idx, task_id in enumerate(task_order):
            if idx < bounded:
                leaf_status_map[task_id] = TaskStatus.COMPLETED.value
            elif idx == bounded:
                leaf_status_map[task_id] = target_status
            else:
                leaf_status_map[task_id] = TaskStatus.PENDING.value

        for leaf in AgentPipelineMixin._iter_pipeline_leaves(tasks):
            leaf_id = str(leaf.get("id", "")).strip()
            if leaf_id in leaf_status_map:
                leaf["status"] = leaf_status_map[leaf_id]
                if leaf_status_map[leaf_id] == TaskStatus.IN_PROGRESS.value:
                    leaf.setdefault("started_at", datetime.now(UTC).isoformat())
                if leaf_status_map[leaf_id] == TaskStatus.COMPLETED.value:
                    leaf["completed_at"] = datetime.now(UTC).isoformat()
        AgentPipelineMixin._refresh_pipeline_graph_state(pipeline)
        AgentPipelineMixin._rollup_pipeline_status(tasks)

    def _advance_pipeline(self, pipeline: dict[str, Any], event: str = "advance") -> list[str]:
        """Advance pipeline by completing current task and activating ready dependents."""
        task_order = self._refresh_pipeline_graph_state(pipeline)
        if not task_order:
            return []
        graph = pipeline.get("task_graph", {})
        if not isinstance(graph, dict):
            return []

        active_ids = list(pipeline.get("active_task_ids", []))
        current_task_id = str(pipeline.get("current_task_id", "")).strip()
        to_complete = ""
        if current_task_id and current_task_id in active_ids:
            to_complete = current_task_id
        elif active_ids:
            order_pos = {task_id: idx for idx, task_id in enumerate(task_order)}
            to_complete = sorted(active_ids, key=lambda item: order_pos.get(item, 1_000_000))[0]

        if to_complete:
            node = graph.get(to_complete)
            if isinstance(node, dict):
                node["status"] = TaskStatus.COMPLETED.value
                node["completed_at"] = datetime.now(UTC).isoformat()
                node["result"] = {
                    "success": True,
                    "output": "",
                    "error": "",
                    "completed_at": node["completed_at"],
                    "event": event,
                }

        self._refresh_pipeline_graph_state(pipeline)
        activated = self._activate_pipeline_tasks(pipeline, event=event)
        self._refresh_pipeline_graph_state(pipeline)
        self._emit_pipeline_update(event, pipeline)
        return activated

    def _tick_pipeline_runtime(
        self,
        pipeline: dict[str, Any] | None,
        *,
        event: str = "runtime_tick",
    ) -> dict[str, Any]:
        """Apply task-level timeout + retry policy and activate newly ready tasks."""
        if not isinstance(pipeline, dict):
            return {"changed": False, "activated": [], "timed_out": [], "retried": [], "failed": []}
        task_order = self._refresh_pipeline_graph_state(pipeline)
        if not task_order:
            return {"changed": False, "activated": [], "timed_out": [], "retried": [], "failed": []}

        graph = pipeline.get("task_graph", {})
        active_ids = list(pipeline.get("active_task_ids", []))
        if not isinstance(graph, dict) or not active_ids:
            return {"changed": False, "activated": [], "timed_out": [], "retried": [], "failed": []}

        now_utc = datetime.now(UTC)
        changed = False
        timed_out: list[str] = []
        retried: list[str] = []
        failed: list[str] = []
        for task_id in active_ids:
            node = graph.get(task_id)
            if not isinstance(node, dict):
                continue
            status = self._normalize_task_status(node.get("status"))
            if status != TaskStatus.IN_PROGRESS.value:
                continue
            timeout_seconds = self._coerce_timeout_seconds(node.get("timeout_seconds"), default_seconds=180.0)
            if timeout_seconds is None:
                continue
            started_at = self._parse_pipeline_timestamp(str(node.get("started_at", "")))
            if started_at is None:
                node["started_at"] = now_utc.isoformat()
                continue
            elapsed = max(0.0, (now_utc - started_at).total_seconds())
            node["last_heartbeat_at"] = now_utc.isoformat()
            if elapsed < timeout_seconds:
                continue
            changed = True
            timed_out.append(task_id)
            retries = max(0, int(node.get("retries", 0)))
            max_retries = max(0, int(node.get("max_retries", 2)))
            node["status"] = TaskStatus.TIMED_OUT.value
            node["timed_out_at"] = now_utc.isoformat()
            if retries < max_retries:
                node["retries"] = retries + 1
                node["status"] = TaskStatus.PENDING.value
                node.pop("started_at", None)
                node["result"] = {
                    "success": False,
                    "output": "",
                    "error": "timeout_retried",
                    "completed_at": now_utc.isoformat(),
                }
                retried.append(task_id)
                continue
            node["status"] = TaskStatus.FAILED.value
            node["result"] = {
                "success": False,
                "output": "",
                "error": "timeout_failed",
                "completed_at": now_utc.isoformat(),
            }
            failed.append(task_id)

        if not changed:
            return {"changed": False, "activated": [], "timed_out": [], "retried": [], "failed": []}

        self._refresh_pipeline_graph_state(pipeline)
        activated = self._activate_pipeline_tasks(pipeline, event=event)
        self._refresh_pipeline_graph_state(pipeline)
        if failed and not list(pipeline.get("active_task_ids", [])) and not list(pipeline.get("ready_task_ids", [])):
            pipeline["state"] = "failed"
        return {
            "changed": True,
            "activated": activated,
            "timed_out": timed_out,
            "retried": retried,
            "failed": failed,
        }

    def _cancel_pipeline_tasks(
        self,
        pipeline: dict[str, Any],
        task_ids: list[str] | None = None,
        *,
        reason: str = "cancelled",
    ) -> list[str]:
        """Cancel active or explicit task ids with a structured reason."""
        self._refresh_pipeline_graph_state(pipeline)
        graph = pipeline.get("task_graph", {})
        if not isinstance(graph, dict):
            return []
        target_ids = [str(item).strip() for item in (task_ids or list(pipeline.get("active_task_ids", [])))]
        cancelled: list[str] = []
        for task_id in target_ids:
            node = graph.get(task_id)
            if not isinstance(node, dict):
                continue
            status = self._normalize_task_status(node.get("status"))
            if status in TERMINAL_TASK_STATUSES:
                continue
            node["status"] = TaskStatus.CANCELLED.value
            node["result"] = {
                "success": False,
                "output": "",
                "error": reason,
                "completed_at": datetime.now(UTC).isoformat(),
            }
            cancelled.append(task_id)
        if cancelled:
            self._refresh_pipeline_graph_state(pipeline)
        return cancelled

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
                status = self._normalize_task_status(node.get("status", TaskStatus.PENDING.value)).upper()
                depends = node.get("depends_on", [])
                depends_suffix = ""
                if isinstance(depends, list) and depends:
                    depends_suffix = f" (deps: {', '.join(str(dep) for dep in depends[:4])})"
                rendered.append(f"{label}. [{status}] {title}{depends_suffix}")
                children = node.get("children")
                if isinstance(children, list) and children:
                    rendered.extend(_render(children, current_prefix))
            return rendered

        lines = [self.instructions.load("planning_pipeline_header.md")]
        lines.append(
            f"Task graph mode: DAG | parallel_limit={max(1, int(pipeline.get('max_parallel_branches', 2)))}"
        )
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
            f"active={progress.get('active_leaves', 0)} "
            f"ready={progress.get('ready_leaves', 0)}"
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
                status = AgentPipelineMixin._normalize_task_status(node.get("status", TaskStatus.PENDING.value))
                title = str(node.get("title", "")).strip()
                retries = int(node.get("retries", 0))
                max_retries = int(node.get("max_retries", 2))
                rendered.append(
                    f"- {label}. status={status} retries={retries}/{max_retries} title={title}"
                )
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
                "ready_tasks": int(progress.get("ready_leaves", 0)),
                "active_tasks": int(progress.get("active_leaves", 0)),
                "current_path": str(progress.get("current_path", "")),
                "eta_seconds": progress.get("eta_seconds"),
                "eta_text": str(progress.get("eta_text", "unknown")),
                "scope_progress": progress.get("scope_progress", []),
            },
            self._format_pipeline_monitor_output(event, pipeline),
        )

    def _finalize_pipeline(self, pipeline: dict[str, Any], success: bool = True) -> None:
        """Finalize pipeline statuses at turn completion."""
        tasks = pipeline.get("tasks", [])
        if not isinstance(tasks, list):
            return
        task_order = self._refresh_pipeline_task_order(pipeline)
        if success:
            self._set_all_pipeline_status(tasks, TaskStatus.COMPLETED.value)
            AgentPipelineMixin._rollup_pipeline_status(tasks)
            if task_order:
                pipeline["current_index"] = len(task_order) - 1
                pipeline["current_task_id"] = task_order[-1]
            pipeline["state"] = "completed"
            self._refresh_pipeline_graph_state(pipeline)
            self._emit_pipeline_update("completed", pipeline)
            return

        cancelled = self._cancel_pipeline_tasks(pipeline, reason="finalize_failed")
        if task_order:
            current_index = int(pipeline.get("current_index", 0))
            bounded = max(0, min(current_index, len(task_order) - 1))
            self._set_pipeline_progress(pipeline, current_index=bounded, current_status=TaskStatus.FAILED.value)
        else:
            self._set_all_pipeline_status(tasks, TaskStatus.FAILED.value)
            AgentPipelineMixin._rollup_pipeline_status(tasks)
        if cancelled:
            pipeline["cancelled_task_ids"] = cancelled
        pipeline["state"] = "failed"
        self._refresh_pipeline_graph_state(pipeline)
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
                status = self._normalize_task_status(leaf.get("status", TaskStatus.PENDING.value))
                if status not in TERMINAL_TASK_STATUSES:
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
