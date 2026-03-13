# Summary: session_orchestrator.py

# session_orchestrator.py Summary

## Overview
A sophisticated parallel task orchestration system for Captain Claw that decomposes complex user requests into directed acyclic graphs (DAGs) of independent tasks, executes them in parallel across worker agents in separate sessions, and synthesizes results into final responses. Implements a five-stage pipeline: decompose → build graph → assign sessions → execute with traffic-light gating → synthesize.

## Purpose
Solves the problem of handling complex multi-step requests that require parallel execution, dependency management, and cross-task data sharing. Enables:
- **Request decomposition**: LLM breaks user input into structured task plans with dependencies
- **Parallel execution**: Multiple tasks run simultaneously across a pool of worker agents
- **Timeout management**: Sophisticated warning/countdown/grace-period flow before task restart
- **Scale-loop optimization**: Auto-postpones timeouts for workers actively processing item batches
- **Workflow persistence**: Save/load/template workflows with variable substitution
- **Task control**: Mid-execution pause/edit/restart/resume with cascading failure recovery
- **Cross-task data sharing**: File manifests and text outputs from dependencies injected into downstream workers

## Most Important Functions/Classes

### 1. **SessionOrchestrator (Main Class)**
Central orchestration engine managing the five-stage pipeline. Maintains task graph state, worker pool, file registry, and execution loop. Handles all public API methods (prepare, execute, orchestrate, task control). Key responsibilities:
- Graph lifecycle management (create, store, execute, reset)
- Worker pool coordination via AgentPool
- Timeout tracking and auto-postponement logic
- Workflow serialization/deserialization
- Status/event broadcasting to web UI

### 2. **prepare(user_input, model, auto_select_model) → dict**
Stages 1–2: Decomposes request via LLM into task plan JSON, builds TaskGraph, detects {{variable}} placeholders, applies workflow-level model overrides. Returns preview data (summary, tasks, synthesis instruction, variables) without executing. Enables web UI preview/edit before execution. Injects workspace tree scan and available sessions list into decompose prompt. Supports automatic model selection by injecting model catalog into system prompt.

### 3. **execute(task_overrides, variable_values) → str**
Stages 3–5: Executes previously prepared graph. Applies per-task overrides from preview editor, substitutes {{variable}} placeholders, assigns sessions (shared/per-worker/existing), runs parallel execution loop with traffic-light gating, synthesizes final answer. Returns synthesized response string and saves Markdown run report.

### 4. **_execute_graph(graph) → None**
Core execution loop (Stage 4). Implements parallel dispatch with:
- **Activation**: Identifies ready tasks (dependencies met, max_parallel not exceeded)
- **Worker dispatch**: Creates asyncio tasks for each active task via `_run_worker()`
- **Polling**: Waits for worker completion with 1-second poll interval
- **Timeout management**: Calls `graph.tick_timeouts()` to track warning/countdown/restart/failure states
- **Auto-postponement**: Rescues scale-loop workers from premature restart if they show recent progress
- **Cascading**: Automatically fails downstream tasks when dependencies fail
- Handles cancellation, cleanup, and resume signaling for task control operations

### 5. **_run_worker(graph, task) → None**
Executes a single task via worker agent. Orchestrates:
- Agent pool acquisition with file registry
- Per-task model override application
- Dependency output injection (text + file paths from upstream tasks)
- Workspace tree inclusion for file discovery
- Iteration budget estimation based on task complexity signals
- Deferred scale-init bypass (prevents workers from re-decomposing)
- Worker completion/failure handling with usage metrics broadcast
- Proper cleanup and agent release

### 6. **_decompose(user_input, auto_select_model) → dict | None**
Stage 1: Calls LLM to decompose request into task plan JSON. Builds context:
- Workspace tree scan (pre-existing files/folders)
- Available sessions list
- Model catalog (if auto_select_model=True)
Parses JSON response with fallback logic (direct → markdown fence → brace block). Logs all LLM interactions to session logger. Returns dict with tasks, summary, synthesis_instruction.

### 7. **_synthesize(user_input, graph, synthesis_instruction) → str**
Stage 5: Feeds all task results back to LLM for final answer. Formats results as readable blocks, appends file manifest, renders synthesis prompt, calls LLM with workflow-level model override. Returns synthesized response string or fallback with raw results on timeout/error.

### 8. **_assign_sessions(graph) → None**
Stage 3: Assigns session IDs to tasks supporting three modes:
- `__shared__` / empty: All tasks share one new session (default)
- `__per_worker__`: Each task gets fresh session
- User-selected ID: Reuse existing session
Creates sessions via session_manager, handles creation failures gracefully.

### 9. **_auto_postpone_scale_workers(graph, timeout_result) → None**
Rescues scale-loop workers from timeout restart by checking if they've made recent item-processing progress (< 180 seconds old). Mutates timeout_result in place, removing auto-postponed tasks from warned/countdown/restarted lists. Prevents loss of micro-loop progress on long-running batch operations.

### 10. **Task Control Methods**
- `pause_task(task_id)`: Cancels running worker, moves task to PAUSED
- `edit_task(task_id)`: Pauses if running, enters EDITING mode for description changes
- `update_task(task_id, description)`: Updates task instructions
- `restart_task(task_id)`: Resets failed/completed/paused task to PENDING, evicts cached agent, uncascades dependents
- `resume_task(task_id)`: Resumes from PAUSED/EDITING, evicts agent if description changed
- `postpone_task(task_id)`: Grants another full timeout period to warned task
- `_reenter_execution_if_needed()`: Restarts execution loop if it exited, or signals running loop

### 11. **Workflow Persistence Methods**
- `save_workflow(name, task_overrides, model)`: Serializes graph + metadata to JSON file in `workspace/workflows/`
- `load_workflow(name)`: Loads workflow JSON, rebuilds graph for preview
- `list_workflows()`: Lists saved workflows with metadata
- `delete_workflow(name)`: Removes workflow file
- `_save_run_output(synthesis_result)`: Saves Markdown report of completed run with all task outputs

### 12. **Variable Substitution**
- `_extract_variables(texts, existing_vars)`: Scans for `{{variable_name}}` placeholders, builds variables list with label/default
- `_substitute_variables(text, values)`: Replaces `{{name}}` with values, leaves unmatched placeholders as-is
Enables workflow templating for reusable multi-run scenarios.

### 13. **Helper Functions**
- `_estimate_task_iterations(description)`: Scans task description for complexity signals (multi-file ops, API calls, per-item processing) and returns iteration budget (5–20)
- `_scan_workspace_tree(workspace_path, max_depth, max_entries)`: Builds concise tree listing of workspace, skips hidden/framework dirs, truncates at 200 entries
- `_resolve_model_provider(selector)`: Creates temporary LLM provider for task-specific model override using main agent's allowed-models list
- `_parse_json_response(raw)`: Parses JSON from LLM response with fallback logic (direct → markdown fence → brace block)
- `_format_results_for_synthesis(results)`: Formats task results into readable blocks for synthesis prompt

## Architecture & Dependencies

### Key Dependencies
- **captain_claw.agent_pool.AgentPool**: Manages pool of worker agents with idle eviction
- **captain_claw.task_graph.TaskGraph**: DAG data structure tracking task states, dependencies, timeouts
- **captain_claw.session.SessionManager**: Creates/manages LLM sessions for workers
- **captain_claw.file_registry.FileRegistry**: Tracks files created across tasks, builds manifests for workers
- **captain_claw.instructions.InstructionLoader**: Loads/renders Markdown instruction templates
- **captain_claw.llm.LLMProvider**: Abstracts LLM calls (decompose, synthesize)
- **asyncio**: Async task dispatch, cancellation, timeout management

### State Management
- `_graph`: TaskGraph instance (None until prepare() called)
- `_pending_futures`: Dict mapping task_id → asyncio.Task for running workers
- `_file_registry`: FileRegistry tracking cross-task file outputs
- `_pool`: AgentPool managing worker agent lifecycle
- `_workflow_*`: Metadata (name, model, variables, saved filename) persisted across prepare/execute
- `_resume_event`: asyncio.Event for signaling execution loop to check for new activatable tasks

### Execution Model
- **Async-first**: All I/O (LLM, file, session) is async; execution loop uses asyncio.wait() with 1-second polling
- **Traffic-light gating**: Tasks activate only when dependencies complete AND max_parallel limit not exceeded
- **Timeout management**: Three-phase flow (warning @ 300s, countdown @ 360s, restart @ 360s+grace) with auto-postponement for active scale workers
- **Cascading failures**: Downstream tasks auto-fail if dependency fails (unless restarted)
- **Worker isolation**: Each task runs in its own session with fresh agent (or shared session if configured)

### Prompt Engineering
- **Decompose system prompt**: Instructs LLM to break request into task plan JSON with optional model selection
- **Worker prompt**: Includes task description, file manifest, dependency outputs, workspace tree
- **Synthesize prompt**: Feeds all task results + file manifest back to LLM for final answer
- **Model catalog injection**: When auto_select_model=True, appends model descriptions to decompose prompt so LLM assigns best model_id to each task

### Configuration
- `orchestrator.max_parallel`: Max concurrent tasks (default 5)
- `orchestrator.max_agents`: Max agents in pool (default 50)
- `orchestrator.worker_timeout_seconds`: Timeout before warning (default 300)
- `orchestrator.timeout_grace_seconds`: Grace period before restart (default 60)
- `orchestrator.worker_max_retries`: Max restart attempts (default 3)
- `orchestrator.idle_evict_seconds`: Evict idle agents after N seconds

### Logging & Observability
- Structured logging via `get_logger(__name__)` with detailed context (task_id, title, status, error)
- LLM session logging via `get_llm_session_logger()` (decompose, synthesize calls)
- Status callbacks for UI updates
- Tool output callbacks for execution tracing
- Broadcast callbacks for structured events (decomposing, executing, task_started, task_completed, etc.)
- Thinking callbacks for worker step-by-step progress