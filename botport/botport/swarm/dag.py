"""DAG operations for swarm task graphs."""

from __future__ import annotations

from collections import defaultdict, deque

from botport.swarm.models import SwarmEdge, SwarmTask


class DAGError(Exception):
    """Raised when a DAG operation fails (e.g. cycle detected)."""


def validate_dag(tasks: list[SwarmTask], edges: list[SwarmEdge]) -> None:
    """Validate that the task graph is a valid DAG (no cycles).

    Raises DAGError if a cycle is detected or edges reference missing tasks.
    """
    task_ids = {t.id for t in tasks}

    for edge in edges:
        if edge.from_task_id not in task_ids:
            raise DAGError(f"Edge references unknown source task: {edge.from_task_id}")
        if edge.to_task_id not in task_ids:
            raise DAGError(f"Edge references unknown target task: {edge.to_task_id}")
        if edge.from_task_id == edge.to_task_id:
            raise DAGError(f"Self-loop detected on task: {edge.from_task_id}")

    # Kahn's algorithm for cycle detection.
    in_degree: dict[str, int] = {t.id: 0 for t in tasks}
    adjacency: dict[str, list[str]] = defaultdict(list)

    for edge in edges:
        adjacency[edge.from_task_id].append(edge.to_task_id)
        in_degree[edge.to_task_id] += 1

    queue = deque(tid for tid, deg in in_degree.items() if deg == 0)
    visited = 0

    while queue:
        node = queue.popleft()
        visited += 1
        for neighbor in adjacency[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    if visited != len(tasks):
        raise DAGError("Cycle detected in task graph")


def topological_sort(tasks: list[SwarmTask], edges: list[SwarmEdge]) -> list[str]:
    """Return task IDs in topological order."""
    in_degree: dict[str, int] = {t.id: 0 for t in tasks}
    adjacency: dict[str, list[str]] = defaultdict(list)

    for edge in edges:
        adjacency[edge.from_task_id].append(edge.to_task_id)
        in_degree[edge.to_task_id] += 1

    queue = deque(tid for tid, deg in in_degree.items() if deg == 0)
    result: list[str] = []

    while queue:
        node = queue.popleft()
        result.append(node)
        for neighbor in adjacency[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    return result


def get_ready_tasks(tasks: list[SwarmTask], edges: list[SwarmEdge]) -> list[SwarmTask]:
    """Return tasks that are queued and whose all predecessors are completed/skipped.

    Tasks in ``pending_approval`` are NOT returned — they are waiting for human action.
    """
    completed_ids = {t.id for t in tasks if t.status == "completed"}
    skipped_ids = {t.id for t in tasks if t.status == "skipped"}
    done_ids = completed_ids | skipped_ids

    # Build predecessor map.
    predecessors: dict[str, set[str]] = defaultdict(set)
    for edge in edges:
        predecessors[edge.to_task_id].add(edge.from_task_id)

    ready: list[SwarmTask] = []
    for task in tasks:
        if task.status != "queued":
            continue
        preds = predecessors.get(task.id, set())
        if preds <= done_ids:
            ready.append(task)

    # Sort by priority (higher first), then by name for determinism.
    ready.sort(key=lambda t: (-t.priority, t.name))
    return ready


def get_all_dependents(
    task_id: str, tasks: list[SwarmTask], edges: list[SwarmEdge],
) -> list[str]:
    """Get all transitive dependent task IDs (for cascading skip on error)."""
    successors_map: dict[str, list[str]] = defaultdict(list)
    for edge in edges:
        successors_map[edge.from_task_id].append(edge.to_task_id)

    visited: set[str] = set()
    queue = deque(successors_map.get(task_id, []))
    while queue:
        tid = queue.popleft()
        if tid in visited:
            continue
        visited.add(tid)
        queue.extend(successors_map.get(tid, []))
    return list(visited)


def get_predecessors(task_id: str, edges: list[SwarmEdge]) -> list[str]:
    """Get direct predecessor task IDs."""
    return [e.from_task_id for e in edges if e.to_task_id == task_id]


def get_successors(task_id: str, edges: list[SwarmEdge]) -> list[str]:
    """Get direct successor task IDs."""
    return [e.to_task_id for e in edges if e.from_task_id == task_id]


def get_task_depth(task_id: str, edges: list[SwarmEdge]) -> int:
    """Get the longest path from any root to this task (for Gantt layout)."""
    predecessors_map: dict[str, list[str]] = defaultdict(list)
    for edge in edges:
        predecessors_map[edge.to_task_id].append(edge.from_task_id)

    cache: dict[str, int] = {}

    def _depth(tid: str) -> int:
        if tid in cache:
            return cache[tid]
        preds = predecessors_map.get(tid, [])
        if not preds:
            cache[tid] = 0
            return 0
        d = max(_depth(p) for p in preds) + 1
        cache[tid] = d
        return d

    return _depth(task_id)


def auto_layout(tasks: list[SwarmTask], edges: list[SwarmEdge]) -> dict[str, tuple[float, float]]:
    """Compute automatic layout positions for DAG nodes.

    Returns {task_id: (x, y)} based on topological depth (x) and row within depth (y).
    """
    depths: dict[str, int] = {}
    for task in tasks:
        depths[task.id] = get_task_depth(task.id, edges)

    # Group tasks by depth level.
    levels: dict[int, list[str]] = defaultdict(list)
    for tid, depth in depths.items():
        levels[depth].append(tid)

    positions: dict[str, tuple[float, float]] = {}
    x_spacing = 250.0
    y_spacing = 120.0

    for depth, task_ids in sorted(levels.items()):
        task_ids.sort()  # Deterministic order.
        for i, tid in enumerate(task_ids):
            positions[tid] = (depth * x_spacing + 50, i * y_spacing + 50)

    return positions
