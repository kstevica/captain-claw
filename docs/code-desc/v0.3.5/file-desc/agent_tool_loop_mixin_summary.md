# Summary: agent_tool_loop_mixin.py

# agent_tool_loop_mixin.py Summary

## Summary
A comprehensive mixin class that orchestrates LLM tool call extraction, execution, and result management. Handles multiple tool call formats (JSON, markdown code blocks, XML), implements duplicate detection and scale-aware guards, tracks progress for large-scale tasks, and manages context optimization through result compression and trimming.

## Purpose
Solves the problem of reliably extracting tool calls from diverse LLM response formats, executing them safely with guards against infinite loops and off-track behavior, and managing the complex lifecycle of tool results in a constrained context window. Enables the agent to handle both simple one-off tool calls and complex multi-step workflows involving dozens of file extractions or API calls.

## Most Important Functions/Classes/Procedures

### 1. **`_handle_tool_calls()`** (async, ~800 lines)
Core execution engine that processes a batch of tool calls. Implements:
- Scale guard interception (hard redirect for off-track calls before execution)
- Duplicate call detection with per-tool-signature counters and context-aware thresholds
- Scale-progress tracking (glob result counting, write/append completion tracking, progress emission)
- Post-execution result enhancement (chunked reduction for oversized content, soft write reminders, pip-install retry hints, small-file warnings)
- Context compaction (write tool call and shell heredoc compression after execution)
- Auto-capture hooks for contacts, scripts, and APIs
- All-blocked streak detection to force finalization when LLM is stuck

### 2. **`_extract_tool_calls_from_content()`** (regex-based, ~400 lines)
Extracts tool calls from LLM response text in 5+ formats:
- Pattern 1: `@tool\ncommand: value` (custom syntax)
- Pattern 2: `{tool => "name", args => { --key "value" }}` (pseudo-object)
- Pattern 3: ` ```tool\ncommand\n``` ` (markdown code blocks, excluding language identifiers like bash/js)
- Pattern 4: `<invoke name="tool"><command>value</command></invoke>` (XML)
- Pattern 5: JSON objects with explicit `tool`/`name` and `args`/`arguments` keys, plus heuristic fallbacks for `query` (→ web_search) and `url` (→ web_fetch)
Deduplicates by signature and enforces max 8 calls per response.

### 3. **`_tool_thinking_summary()`** (static, ~150 lines)
Generates human-readable UI indicators for tool execution. Maps 40+ tool names to concise summaries:
- File tools: `"Reading: filename"`, `"Writing: filename"`
- Web tools: `"Fetching: url[:60]"`, `"Searching: query[:60]"`
- Document extractors: `"Extracting PDF: filename"`
- Planning/pipeline: `"✓ step_title"` or `"▸ step_title"` with scope progress extraction
- Media: `"Text-to-speech"`, `"Generating image: prompt[:50]"`, `"OCR: filename"`
- Device (Termux): `"Taking photo (front camera)"`, `"Torch on"`, `"Getting device location"`
- Guards: `"Guard (type): decision"`
Used to emit progress indicators to the thinking line during execution.

### 4. **Scale-Progress Tracking Block** (within `_handle_tool_calls`, ~200 lines)
Manages large-scale incremental tasks (e.g., extracting 27 PDFs, indexing 100 URLs):
- On `glob`: counts total items, stores full list, sets `_glob_completed` flag
- On `write` with `append=True`: increments completion counter, matches written content against item list (by filename, URL domain, date tokens, normalized text)
- On `typesense` index: tracks indexed items similarly
- Emits progress indicators (`"📄 3 of 27 (11%) — section_title"`)
- Trims processed extracts from session to prevent context bloat
- Injects soft write-reminders when LLM skips writing before reading next item
- Classifies extraction mode (file-based vs. no-file) for micro-loop eligibility

### 5. **Duplicate Call Detection** (~80 lines within `_handle_tool_calls`)
Prevents infinite loops by tracking tool call signatures:
- Path-based tools (read, pdf_extract, etc.): signature = `"tool|path=/absolute/path"`
- URL-based tools (web_fetch, web_get): signature = `"tool|url=https://..."`
- Glob: signature = `"glob|pattern=..."`
- Other tools: signature = `"tool|json_args_sorted"`
- Thresholds: 1–3 calls per signature (higher for stateful CRUD tools, extra headroom during scale tasks)
- On block: returns context-aware message (scale task hint, write-already-saved, or generic duplicate warning)
- On failure: rolls back counter to allow retry with corrected arguments

### 6. **Turn-Level Tool Output Helpers** (~100 lines)
Query session messages for tool results:
- `_collect_turn_tool_output()`: concatenates all non-monitor tool outputs from current turn
- `_turn_has_successful_tool()`: checks for successful result of a specific tool
- `_turn_has_successful_datastore_export()`: checks for successful datastore export
- `_turn_has_unexecuted_script()`: detects .py scripts written but never attempted
- `_turn_has_successful_script_execution()`: stricter check for successful script runs
- `_turn_collect_datastore_saves()`: collects data-writing operations (create_table, insert, update, etc.)
Used by completion gates and planning logic to make decisions about next steps.

---

## Architecture & Dependencies

**Role in System**: Core orchestration layer between LLM response parsing and tool execution. Sits between the agent's main loop (which calls `_handle_tool_calls`) and the underlying tool execution layer (`_execute_tool_with_guard`).

**Key Dependencies**:
- `captain_claw.llm.Message`, `ToolCall`: data structures for LLM communication
- `captain_claw.config.get_config()`: runtime configuration (duplicate thresholds, max chars)
- `captain_claw.logging.get_logger()`: structured logging
- `asyncio`: async execution and event handling
- `json`, `re`, `os`: parsing, regex matching, path resolution
- Assumed parent class methods:
  - `_execute_tool_with_guard()`: actual tool execution with guards
  - `_complete_with_guards()`: LLM completion for rewrite prompts
  - `_scale_guard_intercept()`: scale-aware call blocking
  - `_emit_thinking()`, `_emit_tool_output()`: UI feedback
  - `_add_session_message()`: session state management
  - `_set_runtime_status()`: status updates
  - `get_runtime_model_details()`: model capability checks
  - `_supports_tool_result_followup()`: model compatibility check
  - `_is_monitor_only_tool_name()`: filter monitor-only tools
  - `_classify_item_extraction_mode()`: extraction mode classification
  - `_trim_processed_extracts_in_session()`: context cleanup
  - `_compact_write_tool_call()`, `_compact_shell_tool_call()`: result compression
  - `_chunked_reduce_tool_result()`: oversized result reduction
  - `_auto_capture_contacts_from_tool_call()`, `_auto_capture_scripts_from_tool_call()`, `_auto_capture_apis_from_tool_call()`: memory capture hooks

**Stateful Attributes**:
- `_turn_tool_call_counts`: dict tracking duplicate signatures per turn
- `_scale_progress`: dict tracking large-scale task state (total, completed, items, done_items, _last_action, _glob_completed, _output_file, _extraction_mode)
- `_all_blocked_streak`: counter for consecutive all-blocked batches
- `session`: message history for context queries

**External Interactions**:
- Reads/writes session messages (tool calls, results, user directives)
- Calls LLM for friendly output rewriting via `_complete_with_guards()`
- Emits thinking indicators and tool output events for UI
- Checks model capabilities (tool result followup support)
- Loads instruction templates (tool_output_rewrite_system_prompt.md, tool_output_rewrite_user_prompt.md)