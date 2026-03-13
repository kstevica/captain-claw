# Summary: cron_dispatch.py

# Summary: cron_dispatch.py

## Overview
This module implements a complete cron job scheduling, execution, and history management system for an agent-based automation platform. It handles scheduling of recurring tasks (prompts, scripts, tools, and orchestrated workflows), executes them in appropriate sessions, tracks execution history, and manages job state transitions. The architecture uses dependency injection via `RuntimeContext` to avoid mutable global state.

## Purpose
Solves the problem of scheduling and executing recurring automated tasks within user sessions, including:
- Parsing cron schedule expressions and job specifications
- Executing different job types (prompt dispatch, script/tool execution, workflow orchestration)
- Maintaining audit trails of job executions with timestamps and results
- Managing concurrent job execution to prevent duplicate runs
- Providing a background scheduler loop that polls for due jobs at configurable intervals

## Most Important Functions/Classes

### 1. **execute_cron_job(ctx, job, trigger)**
Orchestrates the complete execution lifecycle of a single cron job. Validates job structure, routes to appropriate executor based on job kind (prompt/script/tool/orchestrate), handles errors, updates job state (last_run_at, next_run_at, last_status, last_error), and prevents concurrent execution of the same job. Returns early if job is already running or has no valid ID.

### 2. **cron_scheduler_loop(ctx)**
Background async loop that continuously polls for due cron jobs at intervals defined by `ctx.cron_poll_seconds`. Fetches up to 10 due jobs per cycle, logs execution events, and delegates to `execute_cron_job()`. Handles graceful cancellation and logs exceptions without crashing the loop.

### 3. **run_script_or_tool_in_session(ctx, target_session_id, kind, path_text, trigger, cron_job_id)**
Executes shell scripts or tools within a specific session context. Resolves file paths using `resolve_saved_file_for_kind()`, switches session context if needed, executes via shell tool with guard, captures output, logs execution events to history, and optionally delivers output to external platforms (e.g., Telegram). Restores previous session context in finally block.

### 4. **_run_orchestrate_cron(ctx, workflow_name, trigger, cron_job_id, variable_values)**
Executes a saved orchestrator workflow as a cron job. Creates temporary `SessionOrchestrator`, loads workflow definition, merges workflow default variables with cron-provided overrides, executes workflow with parallelism management, and logs synthesis results. Properly shuts down orchestrator in finally block.

### 5. **resolve_saved_file_for_kind(kind, session_id, path_text, saved_base_path)**
Security-critical path resolution function that validates and resolves file paths for scripts/tools. Enforces that paths must be within saved root directory and properly scoped to session. Handles both absolute and relative paths, normalizes session IDs, validates file existence, and raises descriptive errors for path traversal attempts or missing files.

## Architecture & Dependencies

**Key Dependencies:**
- `captain_claw.cron`: Schedule parsing/computation utilities (parse_schedule_tokens, compute_next_run, schedule_to_text, now_utc, to_utc_iso)
- `captain_claw.execution_queue.CommandLane`: Job queueing mechanism (CRON lane)
- `captain_claw.session_export`: Utility functions (normalize_session_id, truncate_history_text)
- `captain_claw.prompt_execution`: Prompt dispatch and task enqueueing
- `captain_claw.session_orchestrator.SessionOrchestrator`: Workflow execution engine
- `captain_claw.runtime_context.RuntimeContext`: Dependency injection container providing agent, UI, session manager, and configuration

**State Management:**
- Uses `ctx.cron_running_job_ids` (set) to track in-flight jobs and prevent concurrent execution
- Delegates persistent state to `ctx.agent.session_manager` (database layer)
- All functions receive RuntimeContext rather than closing over mutable state

**Execution Flow:**
1. Background loop (`cron_scheduler_loop`) polls database for due jobs
2. `execute_cron_job()` routes to specialized executors based on job kind
3. Executors (prompt dispatch, script execution, orchestration) run in appropriate context
4. History events logged via `append_cron_history()` with chat and monitor event types
5. Job metadata updated with execution results and next scheduled run time

**Security Considerations:**
- Path resolution strictly validates against saved base directory (prevents directory traversal)
- Session context switching with proper restoration in error cases
- Shell command execution guarded via `_execute_tool_with_guard()`
- File existence and type validation before execution