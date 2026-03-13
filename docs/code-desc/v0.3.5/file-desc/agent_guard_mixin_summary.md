# Summary: agent_guard_mixin.py

# agent_guard_mixin.py Summary

**Summary:**
This mixin provides comprehensive guard policy enforcement for LLM interactions within an Agent system. It implements input/output content filtering, tool execution validation, and approval workflows using configurable guard levels (stop_suspicious or ask_for_approval). The module integrates with external guard models to classify suspicious content and maintains detailed tracing/logging of all LLM interactions and guard decisions.

**Purpose:**
Solves the problem of securing LLM-based agent systems by:
- Preventing malicious or suspicious prompts from reaching the LLM (input guard)
- Blocking harmful model outputs before execution (output guard)
- Validating tool/script execution requests against security policies (script_tool guard)
- Supporting both automatic blocking and human-in-the-loop approval workflows
- Maintaining comprehensive audit trails of all guarded interactions

**Most Important Functions/Classes/Procedures:**

1. **`_enforce_guard(guard_type, interaction_label, content, turn_usage)`**
   - Core guard enforcement logic that evaluates content against configured guard policies. Runs guard decision evaluation, applies configured level (stop_suspicious vs ask_for_approval), and either allows, blocks, or requests user approval. Returns tuple of (allowed: bool, error_reason: str).

2. **`_complete_with_guards(messages, tools, interaction_label, turn_usage, max_tokens)`**
   - Orchestrates full guarded LLM completion pipeline: validates input messages via input guard → calls LLM provider → validates output via output guard → emits traces and logs. Raises GuardBlockedError if any guard blocks. Integrates usage tracking and session logging.

3. **`_run_guard_decision(guard_type, interaction_label, content, turn_usage)`**
   - Executes guard evaluation by loading guard-specific system/user prompt templates, calling guard model with truncated content, parsing JSON response, and accumulating token usage. Returns normalized decision dict with allow/reason/raw fields.

4. **`_parse_guard_decision(raw_text)`**
   - Parses guard model output (JSON or natural language) into normalized decision payload. Handles multiple JSON formats (verdict/decision fields), fallback natural language detection, and conservative blocking on ambiguous outputs.

5. **`_execute_tool_with_guard(name, arguments, interaction_label, ...)`**
   - Validates tool execution requests through script_tool guard before delegating to tool executor. Serializes tool metadata to JSON, enforces guard policy, and raises GuardBlockedError if blocked.

6. **`_build_pipeline_trace_payload(source_tool, arguments)`**
   - Compacts trace payloads for different tool sources (planning, completion_gate, task_contract) by extracting only relevant fields while omitting large content bodies. Optimizes logging/monitoring data size.

7. **`_emit_llm_trace(interaction_label, response, messages, tools, max_tokens)`**
   - Exports full intermediate LLM responses for process analysis when monitoring enabled. Extracts tool calls, usage metrics, and response content; formats as human-readable output and JSON; emits via tool output and session messaging.

8. **`_record_usage_to_db(interaction_label, messages, response, ...)`**
   - Fire-and-forget async persistence of LLM usage metrics to SQLite via session_manager. Calculates input/output byte counts, extracts token usage and latency, creates background task to avoid blocking main flow.

**Architecture & Dependencies:**
- **Mixin Pattern:** Designed as mixin to be composed into Agent class, accessing parent methods like `provider.complete()`, `tools.execute()`, `instructions.load/render()`, `_emit_tool_output()`, `_add_session_message()`
- **External Dependencies:** captain_claw.config (guard settings), captain_claw.llm (Message type), captain_claw.exceptions (GuardBlockedError), captain_claw.logging
- **Configuration:** Reads guard settings from config object with per-guard-type enable/level settings
- **Integration Points:** LLM provider, tool executor, instruction templates, session manager, approval callback, file registry
- **Async/Concurrency:** Uses asyncio for guard evaluation and background usage recording; supports abort_event for cancellation
- **Tracing:** Multi-level logging (tool output, session messages, file-based logs, database records) with configurable monitor_trace_llm flag