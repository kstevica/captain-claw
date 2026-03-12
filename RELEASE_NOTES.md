# Captain Claw v0.3.4 Release Notes

**Release title:** Prompt Caching, LLM Usage Dashboard, Screen Capture & Voice Commands, Clipboard Tool

**Release date:** 2026-03-12

## Highlights

This release introduces Anthropic prompt caching for up to 90% cheaper cache-hit token costs, a full LLM usage analytics dashboard, and the screen capture + voice command system with a global hotkey. The system prompt has been restructured into conditional sections for better cacheability and maintainability, and a new clipboard tool enables cross-app clipboard integration.

## New Features

### Anthropic Prompt Caching
- System prompt templates restructured with a `<!-- CACHE_SPLIT -->` marker separating static (cached) and dynamic (uncached) content
- Two cache breakpoints: (1) static system prompt block, (2) last user/assistant message for tool-loop caching
- Dynamic content (`datetime.now()`, system info, read directories) moved to the end of templates so the static prefix stays byte-identical across calls
- Non-Anthropic providers have the marker stripped transparently — no impact on OpenAI, Gemini, or Ollama
- OpenAI benefits from the static-first layout via its automatic prefix caching
- 14 new tests covering split logic, history breakpoints, mutation safety, and provider integration

### LLM Usage Dashboard
- New `/usage` page with dark-themed analytics UI
- Per-call token tracking: prompt, completion, total, cache read, cache created, input/output bytes, latency, finish reason, error status
- Period filters: Last Hour, Today, Yesterday, This Week, Last Week, This Month, Last Month, All Time
- Provider and model dropdown filters with server-side filtering for accurate totals
- Model dropdown auto-filters by selected provider
- Summary cards: Total Calls, Total Tokens, Prompt/Completion Tokens, Cache Read/Created, Input/Output Size, Avg Latency, Errors
- New `llm_usage` SQLite table with indexed `created_at` and `session_id` columns
- REST API: `GET /api/usage?period=...&provider=...&model=...`
- Homepage card added for navigation

### LLM Usage Tracking
- Every LLM call (conversation, guard, scale loop, orchestrator, etc.) now records token metrics to the session database
- Tracks provider, model, token counts, cache metrics, byte sizes, latency, streaming flag, and error status
- Integrated into the LLM provider layer — works automatically for all providers

### Screen Capture Tool
- New `screen_capture` tool: capture screenshots and optionally analyze with a vision model in one call
- macOS uses native `screencapture` CLI; Linux/Windows use `mss`
- Active display detection on macOS (captures the monitor with the mouse cursor)
- Slash command: `/screenshot [prompt]` for quick capture from the web UI

### Global Hotkey + Voice Commands
- Double-tap Shift (configurable) activates voice command flow
- Hold and speak — audio transcribed in realtime via Soniox, or recorded and transcribed via Whisper/Gemini
- Selected text detection on macOS — if text is selected in any app, it's captured via clipboard round-trip and used as context instead of a screenshot
- Voice responses via TTS so the agent speaks back
- **Hotkey is now opt-in** (disabled by default) — enable in Settings → Voice & Hotkey or set `hotkey_enabled: true` in config
- Hot-reload: toggling the setting starts/stops the daemon without restart
- Web UI settings section added: trigger key, double-tap window, triple-tap wait, max recording duration, STT provider

### STT Tool
- New `stt` tool for speech-to-text transcription
- Provider auto-detection: Soniox (realtime streaming) → OpenAI Whisper → Gemini multimodal
- Supports explicit `stt_provider` and `stt_model` config overrides
- Configurable sample rate, max recording duration, and audio save options

### Clipboard Tool
- New `clipboard` tool: read and write system clipboard (text, images, files)
- macOS-first implementation using native `pbcopy`/`pbpaste`/`osascript`
- Supports text read/write and image/file copy to clipboard

### Conditional System Prompt Assembly
- System prompt split into modular section files: `section_browser_policy.md`, `section_datastore.md`, `section_direct_api.md`, `section_gws.md`, `section_termux_policy.md`
- Micro prompt equivalents: `micro_section_*.md`
- Sections are conditionally included based on enabled tools — unused tool instructions no longer consume context
- Reduces token usage for configurations with fewer tools enabled

### GWS (Google Workspace) Enhancements
- Gmail: base64 attachment content now stripped from LLM context to avoid wasting tokens on binary data
- 700+ lines of new functionality across the GWS tool

### Cron Telegram Integration
- Cron-dispatched tasks now support Telegram delivery
- Results from scheduled tasks can be forwarded to Telegram users
- Runtime context properly propagated through cron execution

### File Export (Markdown → PDF/DOCX)
- REST endpoint for exporting markdown files as styled PDF or DOCX
- Professional CSS styling with proper typography, tables, and code blocks

## Bug Fixes

### File Write Enforcement (Completion Gate)
- The completion gate no longer forces file writes for conversational requests
- Changed default `final_action` from `"write_file"` to `"reply"` across all fallback paths (7 locations in 4 files)
- Updated list task extractor prompts with explicit guidance: "summarize", "tell me", "analyze" → always `"reply"`
- Only triggers `"write_file"` when user explicitly requests file output ("save to", "write to", "export as")

### Shell Tool Timeout System
- Activity-based timeout with automatic extension when process produces output
- Script interpreters (python, node, ruby, etc.) get minimum 120s floor
- Hard wall-time cap at 30 minutes prevents runaway processes
- Quick commands (ls, cat, echo, etc.) get fast 5s timeout

### Orchestrator Improvements
- Multiple fixes to parallel DAG task execution
- Improved session orchestrator reliability
- Better error handling in multi-session workflows

### Other Fixes
- Deep memory: additional error handling for Typesense operations
- Semantic memory: improved indexing reliability
- Agent pool: better idle session cleanup
- Guard system: improved guard evaluation reliability
- Web UI: various CSS and JavaScript fixes
- Orchestrator UI: improved real-time monitoring display
- Typesense tool: minor fix in document operations

## Configuration Changes

### New Settings
- `tools.screen_capture.hotkey_enabled`: `false` (was `true`) — global hotkey is now opt-in
- `tools.screen_capture.hotkey_trigger_key`: `shift` — configurable trigger key
- `tools.screen_capture.hotkey_double_tap_ms`: `400` — double-tap detection window
- `tools.screen_capture.hotkey_triple_tap_wait_ms`: `500` — triple-tap wait window
- `tools.screen_capture.max_recording_seconds`: `30` — voice recording cap
- `tools.screen_capture.stt_provider`: `""` — explicit STT provider override

### Web UI Settings
- New "Voice & Hotkey" section in Settings page with 6 configurable fields
- Hot-reload support: changes apply immediately without server restart

## Internal

- New files: `tools/clipboard.py`, `tools/screen_capture.py`, `tools/stt.py`, `web/hotkey_daemon.py`, `web/static/usage.html`, `tests/test_llm/test_anthropic_cache.py`
- New instruction sections: `section_browser_policy.md`, `section_datastore.md`, `section_direct_api.md`, `section_gws.md`, `section_termux_policy.md` (plus micro equivalents)
- New instruction files: `screenshot_analysis_prompt.md`, `script_rephrase_system_prompt.md`
- `_CACHE_SPLIT_MARKER` constant and `_inject_anthropic_cache_control()` rewrite in `llm/__init__.py`
- `llm_usage` table schema, `record_llm_usage()`, and `query_llm_usage()` in `session/__init__.py`
- `_build_gdrive_file_map()`, `_lookup_gdrive_file()` moved to scale loop
- Nuke confirmation codes in slash commands (safety for `/nuke`)
- 78 files changed, ~8,500 insertions, ~470 deletions
