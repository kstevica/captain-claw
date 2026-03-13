# Summary: agent_session_mixin.py

# agent_session_mixin.py Summary

## Summary
A comprehensive mixin class providing session memory management, context compaction, and runtime configuration for an AI agent system. Implements token-aware message handling, LLM-based conversation summarization, session lifecycle operations (compaction, description generation, procreation), and runtime feature flag synchronization. Handles both persistent session storage and ephemeral memory optimization to maintain context budget constraints in long-running conversations.

## Purpose
Solves the critical problem of managing unbounded conversation history in long-running agent sessions by:
- **Token budgeting**: Accurately counts and tracks token consumption including tool call arguments
- **Context compaction**: Summarizes older messages when token limits approach, preserving recent context
- **Session lifecycle**: Enables session description generation, memory protection, and session "procreation" (merging two parent sessions into a child)
- **Runtime configuration**: Synchronizes feature flags (pipeline mode, tracing, logging) from session metadata
- **Workspace tracking**: Maintains manifest of files created/modified to optimize context injection
- **Tool call optimization**: Post-processes large file writes and shell commands to replace full content with compact references

## Most Important Functions/Classes/Procedures

### 1. **`compact_session(force, trigger)`** (async)
Compacts session by summarizing older messages when token threshold exceeded. Splits messages into "old" (to summarize) and "recent" (to keep), generates LLM-based summary, replaces old messages with summary message, and persists updated session. Tracks compaction metadata (count, trigger type, before/after tokens). Returns tuple of (success: bool, stats: dict).

### 2. **`_session_token_count(messages)` & `_ensure_message_token_count(msg)`**
Dual-layer token accounting system. `_ensure_message_token_count` computes and caches token count per message, crucially including tool_call arguments (which consume LLM tokens but were previously undercounted). `_session_token_count` aggregates across all messages. Uses provider's token counter with fallback to character-based estimation (1 token ≈ 4 chars).

### 3. **`generate_session_description(target_session, max_sentences)` & `_summarize_for_compaction(messages)`** (async)
LLM-powered text generation with deterministic fallbacks. `generate_session_description` creates short summaries of session context for UI display. `_summarize_for_compaction` generates multi-sentence summaries of old messages before compaction. Both format messages for LLM input, call `_complete_with_guards()`, and fall back to rule-based summaries if LLM fails. Prevents token waste by capping output and input sizes.

### 4. **`procreate_sessions(parent_one, parent_two, new_name, persist)` & `_compact_messages_snapshot(messages, trigger)`** (async)
Advanced session merging: creates child session by compacting both parent sessions independently, then merging their compacted message lists. `_compact_messages_snapshot` performs compaction on a deep copy without mutating source. Tracks procreation metadata (parent IDs, stats, merge timestamp). Emits detailed progress updates via `_emit_tool_output()` for UI monitoring.

### 5. **`_sync_runtime_flags_from_session()` & setter methods** (`set_pipeline_mode`, `set_monitor_trace_llm`, etc.)
Bidirectional sync of runtime configuration between session metadata and agent instance attributes. Loads flags on session switch: pipeline mode (loop vs. contracts planning), monitor tracing (LLM/pipeline), logging settings, and model selection. Setter methods persist changes to session metadata. Enables per-session configuration without global state mutation.

### 6. **`_compact_write_tool_call(call_id, arguments)` & `_compact_shell_tool_call(call_id, arguments)`**
Post-execution tool call optimization. Walks backward through session messages to find assistant message containing specified tool_call ID, then replaces large file content arguments with compact references (path + line count + size). Prevents context bloat from multi-file generation tasks. Invalidates cached token counts to force recomputation with smaller payloads.

### 7. **`ensure_pipeline_subagent_contexts(pipeline, task_ids)` & `_record_workspace_write(path, lines, size_kb)`** (async)
Subagent session spawning and workspace manifest tracking. Creates isolated child sessions for pipeline tasks with spawn depth limiting and active child caps. Records file writes in session metadata manifest for context injection hints. `_build_workspace_manifest_note()` generates LLM context note listing created files with rewrite warnings and edit-vs-write guidance.

### 8. **`is_session_memory_protected()` & `set_session_memory_protection(enabled, persist)`** (async)
Memory protection flag management. Prevents accidental session reset when protection enabled. Supports both new dict-based metadata structure and legacy boolean format for backward compatibility. Persists to session metadata with timestamp.

---

## Architecture & Dependencies

**Key Dependencies:**
- `captain_claw.config`: Configuration management (`get_config()`)
- `captain_claw.llm`: Message class for LLM communication
- `captain_claw.logging`: Structured logging
- `captain_claw.session`: Session data model
- `datetime.UTC`: Timestamp generation
- Standard library: `copy`, `re`, `json` (inline imports)

**Expected Instance Attributes** (from parent Agent class):
- `self.session`: Active Session object
- `self.session_manager`: SessionManager for CRUD operations
- `self.provider`: LLM provider with `count_tokens()` method
- `self.instructions`: Template loader with `load()` and `render()` methods
- `self._complete_with_guards()`: Async LLM completion with safety guards
- `self._emit_tool_output()`: Monitoring/tracing output emitter
- `self._set_runtime_status()`: Status flag setter
- `self.memory`: Optional memory subsystem for message recording
- `self._runtime_model_details`: Model metadata dict
- `self._apply_model_option()`, `self._apply_default_config_model_if_needed()`: Model configuration

**Design Patterns:**
- **Mixin composition**: Provides reusable session management without inheritance hierarchy
- **Graceful degradation**: LLM-based operations fall back to rule-based alternatives
- **Lazy evaluation**: Token counts cached on messages, recomputed only when invalidated
- **Metadata-driven configuration**: Session metadata as source of truth for runtime flags
- **Deep copy isolation**: `_compact_messages_snapshot` prevents side effects on source data
- **Backward compatibility**: Supports legacy metadata formats alongside new structures