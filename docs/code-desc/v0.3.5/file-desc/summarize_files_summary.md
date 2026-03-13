# Summary: summarize_files.py

# summarize_files.py

## Summary
Batch file summarization tool that reads all files in a folder, summarizes each individually via LLM calls, and combines results into a final output. Implements context-efficient processing by keeping intermediate work out of the main conversation context, returning only the output file path and compact metadata. Supports text files, code, documents (PDF, DOCX, XLSX, PPTX), and handles large files through intelligent chunking.

## Purpose
Solves the problem of efficiently summarizing multiple documents at scale while managing LLM token usage and maintaining personality-aware context. Eliminates the need for users to manually invoke individual extraction tools (pdf_extract, docx_extract, etc.) and provides a unified pipeline that handles file discovery, content extraction, per-file summarization, and intelligent combination into a cohesive final summary.

## Most Important Functions/Classes/Procedures

1. **`SummarizeFilesTool.execute()`** — Main orchestration method that coordinates the entire pipeline: folder resolution, file discovery, LLM provider initialization, personality loading, per-file summarization, summary combination, and result persistence. Implements comprehensive error handling, progress streaming, token tracking, and ETA calculation.

2. **`_summarise_single_file()`** — Summarizes individual file content with automatic chunking for large files (>400k chars). Implements map-reduce pattern: splits large files into chunks, summarizes each chunk independently, then combines chunk summaries into a single coherent file summary. Tracks LLM call count and token usage per file.

3. **`_combine_summaries()`** — Merges individual file summaries into a single final summary targeting a specified word count. Organizes content logically, groups related themes, and ensures natural flow. Respects personality context and additional instructions for tone/style consistency.

4. **`_read_file_content()`** — Polymorphic file reader that delegates to specialized extractors for documents (PDF, DOCX, XLSX, PPTX) and falls back to UTF-8 text reading for code/text files. Handles extraction errors gracefully and returns (content, error) tuples.

5. **`_llm_complete()`** — Low-level LLM API wrapper that makes individual provider calls with system/user message pairs, extracts usage metrics, records interactions to session DB (fire-and-forget), and returns content + usage dict. Implements 120-second timeout and 4096-token default limit.

6. **`_discover_files()`** — Glob-based file discovery that filters by extension (text + document types), validates file size (0 < size ≤ 10MB), and returns sorted list of readable paths. Supports recursive patterns like `**/*.pdf`.

7. **`_load_personality_context()`** — Loads agent personality and optional user persona from the personality module, formats as prompt block for injection into system prompts. Enables personality-aware summarization that respects agent expertise and user preferences.

8. **`_build_summary_system_prompt()`** — Constructs system prompts with style guidance (concise/detailed/bullet_points), personality context, and custom instructions. Prevents preamble and ensures direct content delivery.

## Architecture & Dependencies

**Core Dependencies:**
- `captain_claw.llm` — LLM provider abstraction (Message class, get_provider())
- `captain_claw.tools.document_extract` — Specialized extractors (_extract_pdf_markdown, _extract_docx_markdown, _extract_xlsx_markdown, _extract_pptx_markdown)
- `captain_claw.personality` — Personality loading (load_personality, load_user_personality)
- `captain_claw.session` — Session management and LLM usage recording (get_session_manager)
- `captain_claw.tools.registry` — Tool base class and ToolResult

**Key Design Patterns:**
- **Map-Reduce for Large Files:** Chunks exceed 400k chars, summarizes each chunk independently, then combines
- **Fire-and-Forget Logging:** LLM usage recorded asynchronously via asyncio.create_task() to avoid blocking main flow
- **Personality Injection:** System prompts prepended with personality context for consistent tone across all LLM calls
- **Token Tracking:** Accumulates prompt/completion/cache tokens across all calls; records individual interactions to DB
- **Graceful Degradation:** Falls back to raw concatenation if final combine step fails; skips individual files on error but continues processing

**Configuration Constants:**
- `_TEXT_EXTENSIONS` — 40+ supported text/code file types
- `_DOCUMENT_EXTENSIONS` — PDF, DOCX, XLSX, PPTX
- `_MAX_FILE_BYTES` — 10 MB size limit
- `_MAX_INPUT_CHARS` — ~400k chars (~100k tokens) per LLM call

**Output Structure:**
- `combined_summary.md` — Final merged summary
- `{filename}_summary.md` — Individual file summaries (one per input file)
- Metadata: file count, token usage, timing, error log