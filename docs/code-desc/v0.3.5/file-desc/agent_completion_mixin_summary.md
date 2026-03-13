# Summary: agent_completion_mixin.py

# Summary: agent_completion_mixin.py

## Overview

**Summary:** This mixin implements a sophisticated completion gate system that validates whether an agent's turn output is ready to finalize or requires additional iterations. It enforces multiple quality checks including file output verification, datastore persistence validation, script execution enforcement, list member coverage evaluation, and task contract completion validation before allowing response finalization.

**Purpose:** Solves the critical problem of preventing the agent from returning hallucinated or incomplete results by implementing a multi-stage validation pipeline that verifies actual tool execution, file creation, data persistence, and task contract fulfillment before accepting a response as complete.

## Architecture & Key Components

### Most Important Functions/Classes

1. **`_attempt_finalize_response()` (async, ~600 lines)**
   - Core orchestration method that applies all completion gates sequentially before finalizing output
   - Returns tuple: `(finalized: bool, final_text: str, finish_success: bool, updated_feedback: str, updated_worker_attempted: bool)`
   - When `finalized=False`, injects `updated_feedback` as user message to force retry; when `True`, returns `final_text` to caller
   - Implements 9 distinct validation gates with iteration-aware blocking logic and stuck-loop detection

2. **`_verify_datastore_saves()` (async)**
   - Validates that datastore write operations (insert, import_file, create_table, update) actually persisted
   - Deduplicates saves by table name (only verifies latest per table)
   - Returns list of failures with action, table name, and reason for each failed verification
   - Prevents hallucinated data persistence claims

3. **`_maybe_append_playbook_hint()` (sync)**
   - Appends one-time rating hint after complex tasks to enable playbook learning
   - Checks: task used contract/scale/pipeline, hint not shown this session, ≥3 tool calls this turn
   - Enables feedback loop for pattern capture

## Validation Gates (in execution order)

### 1. **Auto-Write & Scale Completion Bypass**
- Skips auto-write when scale micro-loop already completed (prevents shadowing real output with summary)
- Collects tool output fallback if final response is empty

### 2. **Write-File Enforcement Gate**
- When `list_task_plan.final_action == "write_file"`, verifies actual file production via one of:
  - Datastore export success
  - Write tool success (markdown, text, non-tabular)
  - Python script execution via shell
  - Media tools (browser screenshot, image_gen)
  - Typesense indexing
- Blocks finalization if no file-producing action detected (prevents "I saved the file" hallucinations)

### 3. **Script Execution Enforcement Gate**
- Detects unexecuted Python scripts written to `scripts/` directory
- Forces execution via `shell(command='python3 <path>')` before finalization
- Prevents scripts written but never run (e.g., PDF/XLSX generators)

### 4. **Failed Shell + Pip Install Retry Gate**
- Detects pattern: shell command failed → pip install succeeded → original command never re-run
- Forces re-execution of original command to actually produce output
- Prevents pip-install-only "fixes" without actual execution

### 5. **Datastore Save Verification Gate**
- Calls `_verify_datastore_saves()` to confirm all write operations persisted
- Blocks with detailed feedback if verification fails
- Emits tool output with verification results (passed/failed tables)

### 6. **Python Worker Enforcement Gate**
- When `enforce_python_worker_mode=True`, ensures write+shell tools both succeeded
- Auto-runs `_run_python_worker_for_list_task()` if not attempted
- Includes stuck-loop safety valve: after 3 consecutive blocks without write+shell success, allows finalization to prevent infinite loops
- Resets streak counter on successful write+shell execution

### 7. **List Member Coverage Gate**
- Validates all list members are addressed in response
- **Smart bypasses:**
  - Script execution covers entire task (output on disk, not in conversation)
  - Browser tool with URL-only members (browser already visited them)
  - Write tool with `final_action=write_file` (content on disk, not in messages)
- Includes stuck-loop detection: if same missing count for 3+ consecutive blocks, allows finalization
- Emits coverage metrics (covered/missing/total members, scale progress)

### 8. **Contract Completion Validation Gate**
- Calls `_evaluate_contract_completion()` to validate against task contract
- Implements retry caps:
  - Post-scale synthesis: 1 retry max (scale already did heavy lifting)
  - General tasks: 2 retries max (prevents pipeline timeouts)
- Emits validation results with failed check details
- Updates planning pipeline progress/status if present

### 9. **Finalization & Auto-Capture**
- Persists response to session
- Auto-captures todos, contacts, scripts, APIs from response
- Updates clarification state
- Appends playbook rating hint (one-time, complex tasks only)

## Key Design Patterns

**Iteration-Aware Blocking:** All gates check `iteration < (hard_turn_iterations - 1)` before blocking, allowing final iteration to finalize regardless of gate status (prevents timeout deadlocks).

**Stuck-Loop Detection:** Multiple gates track consecutive blocks with zero progress:
- Python worker: `_pw_enforcement_streak` (max 3 blocks)
- List coverage: `_coverage_gate_streak` + `_coverage_gate_prev_missing` (max 3 blocks)
- Allows finalization when stuck to prevent infinite loops

**Tool Output Emission:** All gates emit structured tool output via `_emit_tool_output()` with step name, pass/fail status, and detailed metadata for observability.

**Feedback Injection:** When gate blocks, returns `(False, "", finish_success, completion_feedback, ...)` to inject feedback as user message for next iteration.

## Dependencies & Integration

- **Datastore Manager:** `get_datastore_manager()` for persistence verification
- **Session Messages:** Scans `self.session.messages[turn_start_idx:]` for tool calls, shell commands, datastore operations
- **Scale Progress Tracking:** Checks `self._scale_progress` dict for item completion status
- **Planning Pipeline:** Updates task order, progress, checks, and emits pipeline events
- **Helper Methods:** Relies on `_turn_has_successful_tool()`, `_turn_has_successful_script_execution()`, `_turn_collect_datastore_saves()`, `_evaluate_contract_completion()`, `_evaluate_list_member_coverage()`, `_build_completion_feedback()`, `_build_list_coverage_feedback()`, `_maybe_auto_write_requested_output()`, `_friendly_tool_output_response()`, `_collect_turn_tool_output()`, `_persist_assistant_response()`, `_auto_capture_*()` methods

## Error Handling

- Datastore verification catches exceptions and reports as failures with reason
- Graceful fallback to tool output when final response is empty
- Logging at INFO level for all gate decisions (passed/blocked) with iteration context
- WARNING level for stuck-loop detection