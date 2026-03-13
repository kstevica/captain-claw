# Summary: agent_orchestration_mixin.py

# agent_orchestration_mixin.py

## Summary

This is the core orchestration engine for an AI agent's turn-level request processing loop. It implements the `complete()` method (main synchronous entry point) and `stream()` method (streaming wrapper), managing the entire lifecycle of a user request from input validation through tool execution, scale detection, micro-loop takeover, and finalization. The file handles complex state management including iteration budgets, progress tracking, deferred initialization, planning pipelines, and multiple fallback/salvage paths for partial results.

## Purpose

Solves the problem of orchestrating multi-turn, multi-tool agent interactions with sophisticated control flow:
- **Request lifecycle management**: Input validation, message building, LLM calls with retries, tool execution
- **Scale detection & micro-loop**: Automatically detects large list-processing tasks and switches to an optimized per-item processing loop
- **Iteration budgeting**: Prevents infinite loops while allowing dynamic budget extension based on progress
- **Completion gating**: Validates that task requirements are met before finalizing responses
- **Graceful degradation**: Salvages partial results when the agent gets stuck, exhausts budget, or encounters errors
- **Planning pipeline integration**: Coordinates with task planning systems and subagent contexts
- **Guard enforcement**: Applies input/output content guards and handles blocked requests

## Most Important Functions/Classes/Procedures

### 1. **`async def complete(user_input: str) -> str`** (lines 29–1,350)
Main entry point for synchronous request processing. Orchestrates the entire turn:
- Initializes agent state, clears per-turn trackers, validates input guards
- Extracts task contract, list task plan, and scale advisory
- Runs main LLM loop with iteration budgeting, progress tracking, and stagnation detection
- Handles tool calls (explicit and embedded), deferred scale initialization, and micro-loop takeover
- Manages completion requirements and finalization gates
- Returns final response text with success flag

### 2. **`async def stream(user_input: str) -> AsyncIterator[str]`** (lines 1,352–1,430)
Streaming wrapper that yields response chunks. Falls back to `complete()` when tools or guards are enabled (since streaming + tool-calling is limited). For tool-free scenarios, streams directly from the LLM provider.

### 3. **`async def _salvage_partial_result(reason: str) -> str`** (lines 195–248)
Graceful degradation helper that attempts to produce a useful partial result when the agent fails:
- Checks for substantial assistant text (>100 chars) and returns it directly
- If tool results exist, makes a final LLM call asking the model to summarize gathered data
- Falls back to short assistant text (>20 chars)
- Returns empty string if nothing useful is available

### 4. **`async def attempt_finalize(...) -> tuple[bool, str, bool]`** (lines 250–277)
Wrapper around `_attempt_finalize_response()` that updates closure variables. Validates completion requirements, checks coverage gates, and determines if the response is ready to return or needs more iterations.

### 5. **`async def _collect_script_credentials() -> str`** (lines 1,432–1,550)
Collects API keys, passwords, and credentials from config and database for injection into force-script mode prompts. Supports Typesense, Deep Memory, Brave Search, Mailgun, SendGrid, SMTP, LLM providers, Telegram, and direct API entries.

## Architecture & Dependencies

**Key State Variables (closure-scoped):**
- `turn_usage`: Tracks token/cost usage for the turn
- `list_task_plan`: Extracted list-processing task metadata (members, strategy, output format)
- `task_contract`: Generated task contract with requirements and prefetch URLs
- `planning_pipeline`: Task planning DAG with runtime state
- `completion_requirements`: List of validation checks before finalization
- `completion_feedback`: Injected user feedback to guide LLM behavior
- `_scale_progress`: Tracks micro-loop progress (items, done_items, output file)
- `soft_turn_iterations` / `hard_turn_iterations`: Dynamic iteration budgets

**Iteration Loop Control:**
- **Soft limit**: Extensible budget based on progress and remaining work
- **Hard limit**: 2x soft limit, capped at 200 (prevents runaway loops)
- **Extension step**: Increments soft limit by 6–24 iterations when progress detected
- **Stagnation detection**: Exits after 6 iterations without progress (if past halfway point)

**Tool Execution Paths:**
1. **Explicit tool calls**: LLM returns structured tool_use blocks
2. **Embedded tool calls**: LLM embeds tool-like patterns in text (fallback extraction)
3. **Inline commands**: LLM returns shell commands (legacy fallback)

**Scale Micro-Loop Triggers:**
- Early takeover: When scale_progress has 2+ non-URL items before main loop
- Deferred takeover: After content-fetch tools (web_fetch, read, pdf_extract, etc.) populate items
- Tool-call path: When scale_loop_ready() returns true during main loop
- Skipped for: Workers, force-script mode, passthrough extraction mode, >100 items

**Guard & Policy Integration:**
- Input guard: Validates user input before adding to session
- Output guard: Validates final response before returning
- Session tool policy: Restricts tools per session
- Task tool policy: Restricts tools per planning task
- Force-script mode: Strips tool definitions to only {shell, write, read, edit, glob}

**Error Handling & Retries:**
- LLM transient errors (408, 429, 502, 503, 529, timeout): Exponential backoff (5, 10, 20, 40s)
- Orphaned tool_result (400 with tool_use_id): Force compact and rebuild messages
- Guard blocks: Immediate return with error message
- Tool execution failures: Collected and passed to LLM for recovery

**Finalization Gates:**
- Completion requirements validation (e.g., "all sources referenced")
- Coverage checks (e.g., "all list items processed")
- Python worker mode enforcement (explicit script requests)
- Post-write finalization hints (prevent duplicate writes)

**Dependencies:**
- `captain_claw.agent_scale_detection_mixin`: `_build_scale_advisory()`
- `captain_claw.config`: `get_config()` for settings
- `captain_claw.exceptions`: `GuardBlockedError`, `LLMAPIError`, `LLMError`
- `captain_claw.llm`: `Message` class
- `captain_claw.logging`: `get_logger()`
- Mixin methods: `_build_messages()`, `_handle_tool_calls()`, `_attempt_finalize_response()`, `_run_micro_loop_and_summarize()`, etc. (delegated to other mixins)