# Summary: agent_scale_loop_mixin.py

# agent_scale_loop_mixin.py

## Summary

This mixin orchestrates batch processing of item lists through a micro-turn scale loop that isolates per-item LLM calls, preventing context window overflow. It handles context trimming, scale guards to block off-track tool calls, progress tracking, item classification into extraction modes (file, URL, research, inline, passthrough), and implements a constant-context processing pipeline where each item is extracted, processed in isolation, and written independently.

## Purpose

Solves the O(n) context growth problem in batch processing: instead of feeding an ever-growing conversation to the LLM (causing latency creep, token waste, and LLM confusion after 60+ messages), the mixin processes each item with a single isolated LLM call using a minimal prompt. This keeps context size constant regardless of list length while maintaining format consistency and preventing the LLM from "forgetting" the item list or re-extracting already-processed items.

## Most Important Functions/Classes/Procedures

1. **`_run_scale_micro_loop()`** — Main orchestration engine that drives per-item processing. For each unprocessed item: executes extraction tool directly, makes one isolated LLM call with minimal context, writes result via tool execution, updates tracking. Returns summary with success/processed/failed counts. Handles chunked processing for oversized content, research mode with web search, per-item file output, and sink routing (typesense/email/reply).

2. **`_scale_guard_intercept()`** — Hard state-machine guard that blocks off-track tool calls during scale processing. Intercepts and redirects: re-globbing when items are known, re-reading output files, re-extraction of completed items, and batch-fetching of multiple list items. Returns synthetic tool results that tell the LLM what to do next, enforcing the extract→write→next pattern.

3. **`_trim_processed_extracts_in_session()`** — Two-level context trimming: (1) replaces large extractor/fetcher tool results with placeholders, (2) compresses old assistant messages, write results, and guard redirects to single-line placeholders while keeping the last N extract→write pairs in full context. Prevents linear context growth with item count.

4. **`_classify_item_extraction_mode()`** — Classifies list items into extraction modes (url, inline, file, research, passthrough) by sampling items and voting on characteristics. Detects URLs vs. file paths vs. plain-text entities; applies user-level signals (no-external-fetch, save intent, research intent) to override heuristics. Research mode requires explicit opt-in — never auto-enabled.

5. **`_build_scale_item_prompt()`** — Constructs minimal self-contained prompt for processing a single item: focused system message, user message with extracted content and task. Handles output strategy variations (single_file/file_per_item/no_file), extraction mode adjustments (research/inline/file), format references from previous items, and source context for disambiguation.

6. **`_build_scale_progress_note()`** — Injects fresh worklist into every LLM call showing remaining items, completion status, and mode-specific instructions. Keeps note compact (max 15 visible items) so it doesn't consume context budget. Replaces LLM's need to "remember" file list from glob 40+ messages ago.

7. **`_detect_item_extractor()`** — Determines which tool and arguments to use for extracting an item (web_fetch for URLs, pdf_extract/docx_extract for documents, read for files, _passthrough for plain-text entities). Handles URLs with prefixes and Google Drive file overrides.

8. **`_scale_loop_ready()`** — Readiness check determining whether micro-loop can take over from main iteration. Validates: items populated, output file known, at least one item processed, remaining items exist, output strategy compatible, extraction mode not passthrough.

---

## Architecture & Dependencies

**Core Dependencies:**
- `captain_claw.config`, `captain_claw.llm` (Message, ToolCall), `captain_claw.logging`
- Async/await pattern for tool execution and LLM calls
- Session message history for context trimming and format reference extraction
- Tool registry for guard interception and execution

**Key State Variables (in `_scale_progress` dict):**
- `items`: full list of items to process
- `done_items`: set of completed items (path-matched)
- `_output_file`: resolved absolute path for single_file strategy
- `_output_files`: set of per-item output files (file_per_item strategy)
- `_output_strategy`: "single_file" | "file_per_item" | "no_file"
- `_extraction_mode`: "file" | "url" | "research" | "inline" | "passthrough"
- `_processing_mode`: "summarize" | "raw" (tool-to-tool passthrough)
- `_member_context`: dict mapping items to source article context snippets
- `_per_member_action`: task description for per-item processing
- `_glob_completed`, `_batch_extractions`: guards to prevent re-globbing and batch fetching

**Regex Patterns:**
- `_OUTPUT_FILE_RE`: extracts filename from task descriptions ("write results to report.md")
- `_LABEL_ONLY_PREFIX_RE`: distinguishes labelled URLs (date + URL) from named URLs (entity + URL)
- `_FETCH_ONLY_RE`, `_RESEARCH_SIGNAL_RE`: detect per_member_action intent
- `_NO_EXTERNAL_FETCH_RE`, `_SAVE_INTENT_RE`: detect user-level signals that override heuristics

**Tool Integration:**
- Direct tool execution via `_execute_tool_with_guard()` (extraction, writing, sinking)
- Guard interception before tool execution to enforce scale constraints
- Chunked processing pipeline for oversized content (delegates to `_chunked_process_content()`)
- Google Drive file map for GDrive-aware extraction (gws docs_read, drive_download)
- Typesense/email/reply sinks for no_file output strategy

**Message Handling:**
- Session message compression: old pairs replaced with placeholders
- Per-item prompt building with format references from previous items
- No message accumulation — per-item LLM calls discarded after processing
- Token counting for context budgeting and chunked pipeline routing