# Summary: agent_chunked_processing_mixin.py

# Summary: agent_chunked_processing_mixin.py

## Overview
A mixin class implementing a map-reduce pipeline for processing large content that exceeds a model's context window. When content is too large, it splits the material into sequential chunks, processes each independently with full instructions, and combines partial results via either concatenation or a synthesis LLM call. The system is transparent to callers and provides comprehensive logging for debugging.

## Purpose
Solves the fundamental constraint of small-context LLMs by enabling processing of arbitrarily large documents without losing instruction context. Automatically detects when content overflows available context and activates chunking. Particularly valuable for models with 4K–8K token limits processing documents that would normally require 20K+ tokens.

## Architecture & Design Patterns

**Map-Reduce Flow:**
- **Map phase**: Sequential (not parallel) LLM calls, each receiving `[system_prompt + task_preamble + chunk_content + task_suffix]`
- **Reduce phase**: Combines partial results via either simple concatenation or a dedicated synthesis LLM call
- **Fallback chain**: If combine call overflows context, automatically falls back to concatenation; if any chunk fails, continues with remaining partials

**Token Budget Management:**
- Computes per-chunk overhead (system tokens + preamble + suffix + output reserve + framing buffer)
- Calibrates chars-per-token ratio from a 2000-char sample to avoid hardcoded assumptions
- Reserves 80-token framing buffer for chunk markers and separators
- Enforces safety floor of 500 chars minimum per chunk

**Content Splitting Strategy:**
1. Paragraph boundaries (double newline) preferred
2. Single newline fallback
3. Hard character split as last resort
4. Configurable overlap (default ~800 chars) between consecutive chunks for continuity

## Most Important Functions/Classes

### 1. `_chunked_process_content()` (async)
**Purpose**: Main entry point orchestrating the entire map-reduce pipeline.
**Behavior**: 
- Validates chunking necessity and computes token budgets
- Splits content via `_split_content_into_chunks()`
- Iterates through chunks, building per-chunk messages with context markers (`[CHUNK N OF M]`)
- Calls `_complete_with_guards()` for each chunk sequentially
- Invokes `_combine_partial_results()` if multiple partials exist
- Returns combined text + comprehensive stats dict (chunk counts, token usage, timing)
- Emits thinking/tool output for observability at every step

### 2. `_split_content_into_chunks()`
**Purpose**: Intelligently partitions text respecting token budgets while maintaining semantic coherence.
**Behavior**:
- Quick-checks if content fits in single chunk
- Calibrates chars-per-token ratio from sample to handle provider variance
- Splits on paragraph boundaries first, then newlines, then hard character boundaries
- Implements overlap mechanism (configurable chars from previous chunk end) for continuity
- Respects `max_chunks` limit (default 12) to prevent runaway splitting
- Returns list of chunk strings with detailed logging of sizes

### 3. `_combine_partial_results()` (async)
**Purpose**: Synthesizes multiple partial results into a single coherent output.
**Behavior**:
- Concatenates partials with labeled separators (`[PARTIAL RESULT N/M]`)
- Checks if combined partials themselves exceed context budget
- If overflow detected, falls back to simple concatenation (prevents infinite recursion)
- Uses dedicated "document processing assistant" system prompt emphasizing deduplication and coherence
- Calls `_complete_with_guards()` with combine-specific interaction label
- Returns combined text + prompt/response token counts
- Graceful degradation: LLM call failure triggers automatic fallback to concatenation

### 4. `_chunked_processing_is_active()` & `_chunked_processing_needed()`
**Purpose**: Activation logic determining when chunking should be triggered.
**Behavior**:
- `_chunked_processing_is_active()`: Returns true if explicitly enabled OR auto-threshold exceeded (context.max_tokens ≤ auto_threshold)
- `_chunked_processing_needed()`: Compares available context (budget - instructions - output_reserve) against content tokens
- Logs detailed overflow metrics when triggered (context_budget, instruction_tokens, overflow_tokens)

### 5. `_chunked_reduce_tool_result()` (async)
**Purpose**: Integration hook for conversation flow—automatically chunks oversized tool results (e.g., from `read`, `pdf_extract`, `web_fetch`).
**Behavior**:
- Estimates instruction overhead (system prompt + conversation messages + 2000-token buffer)
- Compares tool result size against available context
- Returns `None` if content fits (no chunking needed)
- Otherwise, invokes full `_chunked_process_content()` pipeline with tool-specific context
- Wraps result with metadata header indicating chunking occurred
- Enables seamless handling of large tool outputs without conversation disruption

## Dependencies & Integration Points

**Internal Dependencies:**
- `captain_claw.config.get_config()`: Reads chunking configuration (enabled, auto_threshold, output_reserve, chunk_overlap_tokens, max_chunks, combine_strategy)
- `captain_claw.llm.Message`: Message object for LLM calls
- `captain_claw.logging.get_logger()`: Structured logging

**Expected Mixin Methods (called but not defined here):**
- `_count_tokens(text: str) -> int`: Token counting for budget calculations
- `_complete_with_guards(messages, tools, interaction_label, turn_usage) -> response`: LLM call with safety guards
- `_emit_thinking(text, tool, phase)`: Emit internal reasoning for observability
- `_emit_tool_output(tool, metadata, text)`: Emit tool execution results
- `_build_system_prompt() -> str`: Generate system prompt for context estimation

**Configuration Schema Expected:**
```
context:
  max_tokens: int
  chunked_processing:
    enabled: bool
    auto_threshold: int
    output_reserve_tokens: int
    chunk_overlap_tokens: int
    max_chunks: int
    combine_strategy: str  # "concatenate" or "summarize"
```

## Key Design Decisions

1. **Sequential Processing**: Chunks processed one-at-a-time (not parallel) to simplify error handling and token tracking
2. **Graceful Degradation**: Every failure point has a fallback (chunk LLM failure → skip chunk; combine overflow → concatenate; combine LLM failure → concatenate)
3. **Transparency**: Callers see single combined result; chunking is internal implementation detail
4. **Observability**: Extensive logging + thinking/tool output at every phase for debugging and monitoring
5. **Overlap Strategy**: Configurable character overlap between chunks maintains context continuity without token waste
6. **Dynamic Calibration**: Chars-per-token ratio computed per-document rather than hardcoded, accommodating provider variance

## Performance Characteristics

- **Map phase**: O(n) LLM calls where n = number of chunks (typically 2–12)
- **Reduce phase**: O(1) additional LLM call (or O(0) if concatenating)
- **Token efficiency**: Overhead per chunk = system + preamble + suffix + output_reserve + 80-token framing (~200–400 tokens typical)
- **Timing**: Logged per-chunk and aggregate (map_sec, combine_sec, total_sec)