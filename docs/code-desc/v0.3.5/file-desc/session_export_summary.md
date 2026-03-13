# Summary: session_export.py

## Summary

`session_export.py` is a session history export and rendering utility module that transforms conversation and monitoring data into multiple human-readable and machine-parseable formats (Markdown, JSONL). It provides comprehensive session documentation capabilities by filtering, normalizing, and formatting message logs from AI agent interactions into structured exports for chat transcripts, tool monitoring logs, and pipeline execution traces.

## Purpose

This module solves the problem of exporting and visualizing session history from an AI agent system in multiple formats suitable for different audiences and use cases:
- **Chat exports**: Human-readable conversation transcripts between users and the AI assistant
- **Monitor exports**: Tool/function call logs with arguments and results for debugging
- **Pipeline exports**: Structured execution traces (JSONL format) for programmatic analysis
- **Pipeline summaries**: High-level overview of execution flow with statistics and timeline

The module enables session auditing, debugging, documentation, and analysis by converting raw message data into organized, timestamped records.

## Most Important Functions/Classes/Procedures

1. **`export_session_history(mode, session_id, session_name, messages, saved_base_path) → list[Path]`**
   - Main entry point that orchestrates all export operations. Routes to appropriate renderers based on mode parameter ("chat", "monitor", "pipeline", "pipeline-summary", or "all"). Creates timestamped files in a normalized directory structure and returns list of written file paths.

2. **`render_chat_export_markdown(session_id, session_name, messages) → str`**
   - Filters messages by role (user/assistant/system) and formats them as numbered Markdown sections with timestamps. Includes session metadata header and handles empty message lists gracefully.

3. **`collect_pipeline_trace_entries(session_id, session_name, messages) → list[dict]`**
   - Extracts pipeline trace data from tool messages, with fallback logic to collect planning/task_contract/completion_gate events if primary pipeline_trace messages are absent. Enriches entries with sequence numbers, timestamps, and session context.

4. **`render_pipeline_summary_markdown(session_id, session_name, messages) → str`**
   - Generates comprehensive pipeline execution summary with statistics (source distribution, event types, step counts), timeline visualization, and progress tracking. Aggregates data by source, event, and step with sorted frequency counts.

5. **`normalize_session_id(raw) → str`**
   - Sanitizes session identifiers by removing/replacing unsafe characters, stripping whitespace, and providing "default" fallback. Ensures safe filesystem usage for directory and filename creation.

6. **`truncate_history_text(text, max_chars) → str`**
   - Utility for limiting text length to prevent oversized exports, with graceful truncation indicator appended.

## Architecture & Dependencies

**Dependencies:**
- `captain_claw.cron`: Provides `now_utc()` and `to_utc_iso()` for UTC timestamp generation
- Standard library: `json`, `datetime`, `pathlib`

**System Role:**
- Part of a larger AI agent system (Captain Claw) that manages conversation sessions with planning, task execution, and monitoring capabilities
- Sits at the export/reporting layer, consuming normalized message dictionaries from session storage
- Produces human-readable documentation and machine-parseable traces for downstream analysis tools

**Key Design Patterns:**
- **Multi-format rendering**: Separate render functions for each export type (Markdown for humans, JSONL for machines)
- **Message filtering**: Role-based and tool-name-based filtering to extract relevant subsets from unified message log
- **Fallback logic**: Pipeline trace collection attempts primary source first, then falls back to alternative sources if empty
- **Safe path handling**: Normalization of session IDs before filesystem operations
- **Timestamp enrichment**: All exports include UTC timestamps for traceability and timeline reconstruction