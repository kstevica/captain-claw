# Summary: agent_scale_detection_mixin.py

# Summary: agent_scale_detection_mixin.py

## Overview

This mixin handles pre-flight detection of large-scale list-processing tasks and orchestrates the initialization of scale-progress tracking. It detects when a task involves many items (files, URLs, records), injects mandatory processing advisories into the LLM context to force incremental append-to-file strategies, and manages deferred list extraction after content becomes available. The module prevents context-window overflow by ensuring the LLM processes items one-at-a-time rather than attempting to hold all results in memory.

## Purpose

Solves the problem of LLM agents attempting to process hundreds or thousands of items in a single context window, which causes:
- Context overflow and token exhaustion
- Hallucinated or incomplete results
- Failed task completion

The mixin detects scale early, injects explicit processing constraints, re-extracts item lists after web fetches provide necessary content, and initializes tracking structures for the scale micro-loop that processes items incrementally.

## Most Important Functions/Classes/Procedures

### 1. **`_input_suggests_large_scale(user_input: str) -> bool`** (static)
Detects whether user input indicates a large-scale list-processing task through pattern matching. Returns True on:
- Large-scope phrasing ("all files in folder and subfolders", "process every item")
- Explicit list-providing language ("here are the URLs", "these files")
- List-producing language ("create a CSV", "compile a list")
- Inline item count (3+ URLs or 5+ numbered/bulleted lines)

Critical for early detection before any processing begins.

### 2. **`_preflight_scale_check(effective_user_input: str, list_task_plan: dict) -> str`**
Runs pre-flight detection and returns a scale advisory string to inject into the LLM context. Fires when:
- Member count meets configured threshold (scale_advisory_min_members), OR
- Input explicitly provides/requests a list (detected by `_input_suggests_large_scale`)

Strips dependency-output sections to avoid false positives from upstream task titles. Returns empty string when no advisory needed. The advisory text templates (`_SCALE_ADVISORY_SINGLE_FILE`, `_SCALE_ADVISORY_FILE_PER_ITEM`, `_SCALE_ADVISORY_NO_FILE`) are carefully crafted to force the LLM into strict read-then-write-then-read patterns.

### 3. **`_deferred_scale_init(effective_user_input: str, list_task_plan: dict, turn_usage: dict) -> dict`** (async)
Re-runs list extraction after web_fetch brings article content into session messages. Handles the pattern: "fetch article X, there is a list of Y, research all". Limits re-attempts to 3 to avoid token waste. Validates discovered items (rejects source-URL-only results as degenerate). Stores full source page content for inline extraction mode. Returns updated list_task_plan with newly discovered members.

### 4. **`_init_scale_progress_from_plan(plan: dict, user_input: str) -> dict`**
Centralizes scale-progress initialization from a list_task_plan. Repairs file-path members that the LLM may have shortened by stripping directory prefixes. Builds the `_scale_progress` tracking dict with:
- `total`, `completed`, `items`, `done_items` (progress counters)
- `_output_strategy` (single_file, file_per_item, no_file)
- `_extraction_mode` (inline vs. deferred)
- `_processing_mode` (summarize, extract, etc.)

Used by both preflight and deferred initialization paths to avoid duplication.

### 5. **`_repair_file_path_members(members: list[str]) -> list[str]`**
Fixes file-path members missing directory prefixes (e.g., `subdir/file.pdf` → `pdf-test/subdir/file.pdf`). Uses three strategies:
1. Search prior session tool calls for longer paths ending with the member string
2. Use rglob to find the file and derive the missing prefix
3. Return unchanged if repair fails

Prevents scale-loop failures when the LLM extractor incorrectly shortened paths.

### 6. **`_run_micro_loop_and_summarize(...) -> dict`** (async)
Consolidates the repeated pattern of running the scale micro-loop and building a summary. Orchestrates:
- Strip advisory from user input → derive per-member action
- Emit takeover trace
- Run `_run_scale_micro_loop` (defined elsewhere)
- Build summary via `_build_micro_loop_summary`
- Add session message with results

Returns micro-loop result augmented with `"summary"` key. De-duplicates code that appeared in 4+ takeover paths.

### 7. **`_try_scale_init_from_glob(tool_results: list, effective_user_input: str, list_task_plan: dict) -> list[str]`**
Populates `_scale_progress` directly from glob results without expensive LLM re-extraction. Fires when glob returns multiple files and task is a list-processing request. Validates task actually needs per-item processing (checks skip patterns and positive list-processing signals). Returns list of items if scale progress initialized, empty list otherwise.

## Architecture & Dependencies

**Key Dependencies:**
- `captain_claw.config.get_config()` — retrieves scale thresholds (scale_advisory_min_members, lightweight_progress_min_members)
- `captain_claw.logging.get_logger()` — structured logging
- Session/message management (via `self.session`, `self._add_session_message`)
- Tool output emission (`self._emit_tool_output`)
- LLM plan generation (`self._generate_list_task_plan`)
- Scale micro-loop execution (`self._run_scale_micro_loop`)

**Role in System:**
- **Pre-flight gate** — detects scale before main orchestration loop
- **Advisory injector** — modifies LLM context to enforce incremental processing
- **Deferred extractor** — re-runs list detection after content fetch
- **Progress initializer** — builds tracking structures for micro-loop
- **Repair utility** — fixes LLM extraction artifacts (shortened file paths)

**Key Patterns:**
- **Dependency-output stripping** — all detection checks strip "Results from previous steps:" section to avoid false positives from upstream task titles
- **Source-URL guard** — rejects degenerate extraction results (source URL repeated) via `_items_are_source_urls_only`
- **Skip-pattern matching** — prevents scale loop hijacking non-scalable tasks (send email, combine files, bulk indexing)
- **Positive confirmation** — glob-based init requires both absence of skip patterns AND presence of list-processing signals
- **Token budgeting** — deferred init uses generous context limits (50K chars/message, 80K total) to capture full articles with embedded lists