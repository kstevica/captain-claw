# Summary: prompt_execution.py

# Summary: prompt_execution.py

## Overview
This module manages asynchronous prompt execution, command queue dispatch, and session-based task scheduling for an AI agent system. It decouples mutable state from execution logic by passing a `RuntimeContext` object through all operations, enabling safe concurrent execution across multiple sessions and priority lanes.

## Purpose
Solves three critical problems:
1. **Queue Management**: Routes agent tasks through a multi-lane priority queue system (session lanes → global lanes → agent runtime) with configurable debouncing, capacity caps, and drop policies
2. **Prompt Execution**: Executes user prompts with streaming support, cancellation via ESC key, error recovery, and integration with cron jobs and monitoring systems
3. **Followup Dispatch**: Intelligently queues follow-up prompts when sessions are busy, with deduplication and mode-based handling (interrupt, collect, steer, etc.)

## Architecture & Dependencies

**Key Dependencies:**
- `captain_claw.execution_queue`: CommandLane, QueueSettings, lane resolution utilities
- `captain_claw.runtime_context`: RuntimeContext (central state container)
- `captain_claw.cron_dispatch`: Cron event logging and monitoring
- `captain_claw.session_export`: History truncation utilities
- `captain_claw.next_steps`: Suggested action extraction

**System Role:**
Acts as the execution orchestration layer between UI/API endpoints and the agent core. Manages concurrency, session isolation, and graceful degradation when sessions are busy.

---

## Most Important Functions/Classes

### 1. **`enqueue_agent_task(ctx, session_id, task, lane, warn_after_ms)`**
- **Purpose**: Enqueues an async task through a three-tier lane hierarchy (session → global → agent runtime)
- **Behavior**: Resolves session and global lane names, creates nested lambda wrappers to enforce execution order, warns if task waits >2s
- **Critical Role**: Primary entry point for all agent work; ensures session isolation and prevents concurrent execution within a session

### 2. **`run_prompt_in_active_session(ctx, prompt_text, display_prompt, cron_job_id, ...)`**
- **Purpose**: Executes a single user prompt in the currently active session with full lifecycle management
- **Key Features**:
  - Streaming vs. non-streaming response modes
  - ESC-key cancellation support via `run_cancellable()`
  - Cron job integration (logs events, delivers output to platforms)
  - Automatic next-steps extraction and UI rendering
  - Error recovery with partial result extraction from session history
  - Tracks execution time and completion timestamp
- **Critical Role**: Core prompt execution engine; handles both interactive and automated (cron) workflows

### 3. **`dispatch_prompt_in_session(ctx, session_id, prompt_text, source, cron_job_id, ...)`**
- **Purpose**: Intelligently routes prompts to either immediate execution or followup queue based on session state
- **Logic**:
  - If session idle: executes immediately
  - If busy: checks queue settings (mode) and either queues followup or interrupts existing work
  - Supports deduplication modes (prompt, message-id, none)
  - Logs all decisions to cron monitoring system
- **Critical Role**: Smart dispatcher that prevents prompt loss and respects queue policies

### 4. **`resolve_queue_settings_for_session(ctx, session_id)`**
- **Purpose**: Loads and normalizes queue configuration from session metadata with fallback to global config
- **Configuration Hierarchy**: Session metadata → global config → hardcoded defaults
- **Resolves**: mode (collect/steer/followup/etc.), debounce_ms, cap, drop_policy
- **Critical Role**: Enables per-session queue tuning without code changes

### 5. **`run_cancellable(ui, work)`**
- **Purpose**: Wraps async work with ESC-key cancellation capability
- **Behavior**: Races work task against escape detection; cancels work on ESC, cleans up both tasks
- **Critical Role**: Provides user control over long-running operations without blocking UI

### 6. **`update_active_session_queue_settings(ctx, mode, debounce_ms, cap, drop_policy)`**
- **Purpose**: Dynamically updates queue settings for the active session at runtime
- **Validation**: Normalizes and validates all parameters before persistence
- **Persistence**: Saves to session metadata with timestamp, reloads settings for immediate effect
- **Critical Role**: Enables real-time queue tuning without session restart

### 7. **`run_prompt_in_session(ctx, session_id, prompt_text, source, cron_job_id, trigger)`**
- **Purpose**: Executes a prompt in a non-active session (loaded on-demand) with full session context switching
- **Behavior**:
  - Loads target session from storage
  - Saves/restores previous session state
  - Forces "loop" pipeline mode and disables planning (cron-specific)
  - Wraps execution with start/done/failed tool messages for audit trail
  - Handles session switching cleanup on error
- **Critical Role**: Enables cron jobs to run in arbitrary sessions without disrupting interactive sessions

---

## Data Flow

```
User/API Input
    ↓
run_prompt_in_active_session() or dispatch_prompt_in_session()
    ↓
[Session Busy?] → Yes → resolve_queue_settings_for_session()
    ↓                          ↓
   No                    [Mode = interrupt?]
    ↓                      Yes ↓ No
[Execute]              [Clear lanes] [Enqueue]
    ↓                      ↓          ↓
run_cancellable()    cron_monitor  followup_queue
    ↓                              ↓
[Stream/Complete]          schedule_drain()
    ↓                              ↓
extract_next_steps()      run_queued_followup_prompt()
    ↓
UI render + cron logging
```

## Configuration & Extensibility

**Queue Modes** (per-session):
- `collect`: Accumulate prompts, debounce before execution
- `steer`: Interactive steering with human feedback
- `followup`: Queue for later execution
- `interrupt`: Cancel current work, execute immediately
- `steer-backlog`: Steer with backlog preservation

**Drop Policies**:
- `old`: Remove oldest queued item
- `new`: Reject new item
- `summarize`: Summarize dropped items

**Cron Integration Points**:
- `cron_chat_event()`: Logs user/assistant/system messages
- `cron_monitor_event()`: Logs operational events (queued, executed, failed)
- `ctx.on_cron_output`: Platform delivery callback (e.g., Telegram)