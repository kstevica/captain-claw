# Summary: task_graph.py

# task_graph.py Summary

## Summary
Standalone DAG scheduler implementing a task graph with topological ordering, concurrency control via traffic lights, timeout/retry management, and dependency tracking. Decoupled from AgentPipelineMixin for use by SessionOrchestrator, providing deterministic parallel task execution with sophisticated timeout handling (warning → grace period → restart flow).

## Purpose
Solves the problem of orchestrating parallel execution of dependent tasks with:
- **Dependency management**: Topological sorting ensures tasks execute only after dependencies complete
- **Concurrency control**: Traffic light mechanism limits parallel execution to prevent resource exhaustion
- **Timeout handling**: Three-phase timeout system (running → warning → grace period → restart/fail) allows user intervention before automatic restart
- **Retry logic**: Automatic retry on failure with configurable max retries and graceful cascade failure on dependency failure
- **Task control**: Pause/resume, edit, and restart capabilities for interactive workflow management

## Most Important Functions/Classes

### 1. **OrchestratorTask** (dataclass)
Single unit of work in the orchestration DAG. Tracks task identity (id, title, description), execution state (status, started_at, completed_at), dependencies (depends_on), retry/timeout metadata (retries, max_retries, timeout_seconds, timeout_warning_at), and result/error information. Includes editing mode support for interactive workflow modification.

### 2. **TaskGraph.refresh()** 
Core state management method that recomputes topological order and categorizes all tasks into ready/blocked/running/completed/failed/held sets. Implements cascade failure logic: if a dependency fails, dependent tasks are automatically marked FAILED. Called before activation to ensure accurate state representation.

### 3. **TaskGraph.activate_next()**
Traffic light implementation that moves ready tasks to RUNNING state up to max_parallel limit, respecting topological order for deterministic behavior. Returns newly activated tasks. Skips tasks in edit mode. Designed to be called repeatedly by execution loop without race conditions.

### 4. **TaskGraph.tick_timeouts()**
Sophisticated timeout management implementing three-phase flow: (1) RUNNING tasks exceeding timeout_seconds enter TIMEOUT_WARNING phase, (2) tasks in warning phase have timeout_grace_seconds to be postponed before automatic action, (3) grace expiration triggers restart (if retries available) or failure. Returns detailed countdown information for UI display.

### 5. **TaskGraph._topological_sort()** (static)
Kahn's algorithm implementation for stable topological ordering. Handles cyclic/unreachable nodes gracefully by appending them at end. Ensures deterministic execution order by sorting ready tasks at each step, critical for reproducible parallel execution.