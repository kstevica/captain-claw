# Summary: agent_pipeline_mixin.py

# agent_pipeline_mixin.py Summary

**Summary:** A comprehensive mixin class providing DAG-based task pipeline construction, state management, and execution orchestration for multi-step agent workflows. Handles task scheduling, dependency resolution, timeout/retry policies, progress tracking, and parallel task activation with sophisticated status rollup and topological ordering.

**Purpose:** Solves the problem of managing complex, hierarchical task workflows with dependencies, parallel execution constraints, and real-time progress monitoring. Enables agents to decompose multi-step requests into structured task graphs with isolated execution contexts, token tracking, and completion verification.

---

## Most Important Functions/Classes

### 1. **TaskStatus (Enum) & TERMINAL_TASK_STATUSES**
Defines the complete task lifecycle with 8 states (PENDING, READY, BLOCKED, IN_PROGRESS, COMPLETED, FAILED, CANCELLED, TIMED_OUT). Terminal statuses mark irreversible completion states. Critical for state machine validation throughout the pipeline.

### 2. **ExecutionContext (Dataclass)**
Isolated metadata container for individual task execution tracking. Captures context_id, session lineage (spawned_by, parent_task_id, spawn_depth), token usage buckets (prompt/completion/total), child process limits (max_children=5), and bounded history (max 80 events). Enables subagent spawning with depth constraints and resource accounting.

### 3. **TaskNode (Dataclass)**
DAG-capable task schema with id, title, status, dependency list (depends_on), child nodes, execution context, result payload, and retry policy (max_retries=2). Supports hierarchical task trees with timeout_seconds and optional tool_policy restrictions.

### 4. **_refresh_pipeline_graph_state()**
Core state machine that rebuilds the entire DAG index from task tree. Computes:
- Leaf task identification and topological ordering via `_topological_task_order()`
- Dependency expansion (resolves transitive dependencies through parent hierarchy)
- Status classification: ready (deps satisfied), blocked (waiting), active (in progress), completed, failed
- Ready/active/blocked/completed/failed task ID sets
- Current task pointer and execution index

Handles circular dependency detection implicitly through leaf-only ordering. Returns task_order list for execution sequencing.

### 5. **_sanitize_pipeline_nodes()**
Schema validation + normalization recursive function. Enforces:
- Unique task ID generation (sanitizes invalid chars, auto-generates fallbacks)
- Max node limits (96 total, configurable)
- Title requirement (220 char truncation)
- Status normalization via `_normalize_task_status()`
- Execution context inheritance with depth tracking
- Tool policy coercion via `_coerce_tool_policy()`
- Timeout coercion with 180s default via `_coerce_timeout_seconds()`
- Dependency ID sanitization and deduplication
- Recursive child node processing with depth increment

Returns normalized task tree ready for graph operations.

### 6. **_build_task_pipeline()**
High-level pipeline factory that orchestrates:
- Task extraction from user input (JSON parsing via `_extract_json_object()` or fallback to `_default_task_contract()`)
- Default 3-task fallback if extraction fails
- Node sanitization with configurable max_tasks (default 6)
- Optional completion_checks appended as final task with all-leaf dependencies
- Pipeline initialization with metadata (created_at, request, state="active")
- Subagent configuration (enabled, max_spawn_depth=2, max_active_children=5)
- Initial graph refresh and task activation

Returns fully initialized pipeline dict with task_graph, task_order, and ready_task_ids populated.

### 7. **_activate_pipeline_tasks()**
Selects ready tasks for execution up to max_parallel_branches limit (default 2). Transitions status READY→IN_PROGRESS, sets started_at timestamp, appends to active_task_ids. Respects available_slots calculation (max_parallel - current_active). Returns list of newly activated task IDs.

### 8. **_topological_task_order()**
Kahn's algorithm implementation for stable topological sort. Inputs: leaf_ids, dependencies dict, priority_map. Processes nodes with zero in-degree first, sorted by priority. Handles disconnected components by appending remaining tasks. Returns execution-safe task order respecting all dependencies.

### 9. **_tick_pipeline_runtime()**
Timeout enforcement + retry policy application. For each active task:
- Parses started_at timestamp, calculates elapsed seconds
- Compares against timeout_seconds (180s default)
- On timeout: increments retries counter, checks max_retries (default 2)
- If retries < max_retries: resets to PENDING, clears started_at, records "timeout_retried"
- If retries exhausted: marks FAILED with "timeout_failed" error
- Triggers graph refresh and task activation for newly ready tasks
- Sets pipeline state="failed" if no active/ready tasks remain

Returns change summary with activated/timed_out/retried/failed lists.

### 10. **_build_pipeline_progress_details()**
Comprehensive progress telemetry builder. Computes:
- Leaf index/total/remaining from task_order
- Elapsed seconds from created_at timestamp
- ETA calculation: (elapsed / completed_leaves) × leaf_remaining
- Scope-level progress for nested task hierarchies via `_find_pipeline_leaf_path()`
- Per-scope ETA and sibling tracking
- Formatted ETA text via `_format_eta_seconds()` (e.g., "2h 15m")

Returns dict with leaf_index, leaf_total, leaf_remaining, current_path, scope_progress array, eta_seconds, elapsed_seconds, and leaf counts by status.

### 11. **_rollup_pipeline_status()**
Recursive status aggregation from leaf nodes upward. Propagation rules:
- FAILED/TIMED_OUT in any child → parent FAILED
- CANCELLED in any child → parent CANCELLED
- All children COMPLETED → parent COMPLETED
- Any child IN_PROGRESS → parent IN_PROGRESS
- Any child READY → parent READY
- Any child BLOCKED → parent BLOCKED
- Leaf nodes retain their own status

Ensures parent status accurately reflects subtree state.

### 12. **_advance_pipeline()**
Completes current/first-active task and cascades activation. Marks task COMPLETED with completed_at timestamp and result payload. Calls `_refresh_pipeline_graph_state()` to recompute ready set, then `_activate_pipeline_tasks()` to spawn newly unblocked tasks. Emits pipeline update event. Returns list of newly activated task IDs.

### 13. **_cancel_pipeline_tasks()**
Bulk task cancellation with structured reason. Filters to non-terminal tasks, sets status=CANCELLED, records error reason in result payload. Respects explicit task_ids list or defaults to active_task_ids. Returns list of cancelled task IDs.

### 14. **_record_pipeline_task_usage()**
Token usage accumulation on active tasks. Extracts prompt/completion/total tokens from usage dict, appends to each active task's execution_context.token_usage bucket. Supports distributed token tracking across parallel tasks.

### 15. **_build_pipeline_note()**
Renders task tree into markdown planning note for model context injection. Formats as numbered outline with status badges, dependency annotations, and result data snippets (truncated to 280 chars). Includes header/footer from instructions. Used for in-context task awareness.

---

## Architecture & Dependencies

**Architecture Pattern:** Mixin class providing stateless utility methods + instance methods requiring `self.session`, `self.instructions`, `self.planning_enabled`, `self._emit_tool_output()`, `self._is_monitor_only_tool_name()`.

**Key Dependencies:**
- `dataclasses` (TaskStatus, TaskResult, ExecutionContext, TaskNode)
- `datetime.UTC` (timestamp normalization to UTC)
- `enum.Enum` (TaskStatus lifecycle)
- `json` (result data serialization, tool argument tracking)
- `re` (task ID sanitization, whitespace normalization)
- `typing` (type hints for dict/list/Any payloads)

**Data Flow:**
1. User input → `_build_task_pipeline()` → task extraction/normalization
2. `_sanitize_pipeline_nodes()` → schema validation + execution context injection
3. `_refresh_pipeline_graph_state()` → DAG indexing + dependency resolution
4. `_activate_pipeline_tasks()` → ready→in_progress transition
5. `_tick_pipeline_runtime()` → timeout/retry enforcement
6. `_advance_pipeline()` → completion + cascade activation
7. `_build_pipeline_progress_details()` → telemetry for monitoring

**Role in System:** Provides complete task orchestration layer for multi-step agent workflows. Decouples task definition (schema) from execution (state machine), enabling flexible parallel execution with dependency constraints, timeout policies, and real-time progress visibility. Integrates with session/instruction/monitoring infrastructure for full observability.