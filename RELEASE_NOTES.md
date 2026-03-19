# Captain Claw v0.4.3 Release Notes

**Release title:** BYOK (Bring Your Own Key), Visual Generation Usage Tracking

**Release date:** 2026-03-19

## Highlights

This release introduces **BYOK (Bring Your Own Key)** for Public Computer Mode — public users can now provide their own LLM API credentials (OpenAI, Anthropic, Gemini, xAI, OpenRouter) directly in the browser, stored only in localStorage and never persisted on the server. Also fixes a gap where **streaming visual generation** was not tracked in LLM usage analytics, and adds a **BYOK filter** to the Usage dashboard.

## New Features

### BYOK (Bring Your Own Key) for Public Computer Mode

Public users can now use their own LLM API keys instead of relying on the server's shared provider:

- **🔑 BYOK toolbar button** — Visible only in public mode, opens a themed modal for credential entry
- **Provider selection** — OpenAI, Anthropic, Gemini, xAI, OpenRouter (Ollama blocked to prevent SSRF)
- **Browser-only storage** — API keys stored in localStorage, never sent to the server for persistence
- **Per-agent provider** — Each public session gets its own LLM provider instance; BYOK does not affect other users or the server's default provider
- **Auto-reconnect** — Saved credentials are automatically re-applied on WebSocket reconnect
- **Visual generation support** — BYOK provider is used for both chat and visual HTML generation
- **Themed UI** — BYOK modal and button follow the active Computer theme (all 14 themes supported)
- **Security** — Keys transmitted over encrypted WebSocket, held in memory only, never logged; no custom base URLs to prevent SSRF

**WebSocket protocol:**
- New message type: `set_byok` (client → server) — sends provider, model, and API key
- New message type: `clear_byok` (client → server) — reverts to server's default provider
- New message type: `byok_status` (server → client) — confirms activation or reports errors
- `welcome` message now includes `is_public: true/false` flag

### LLM Usage: BYOK Tracking and Filter

- **BYOK column** — New `byok` column in the `llm_usage` database table tracks whether each call used user-provided credentials
- **🔑 indicator** — Usage detail table shows a key emoji for BYOK calls
- **BYOK Calls summary card** — New card in the usage summary showing total BYOK call count
- **BYOK filter dropdown** — Filter usage records by All / BYOK Only / Server Only
- **Database migration** — Existing databases automatically gain the `byok` column via ALTER TABLE

## Bug Fixes

### Streaming Visual Generation Not Tracked

The streaming visual generation endpoint (`POST /api/computer/visualize/stream`) — which is the endpoint actually used by the Computer frontend — was not recording LLM usage. The non-streaming endpoint had tracking, but the streaming one was missing it entirely. Fixed by adding `_record_visual_usage_streaming()` that records input/output byte sizes and latency after successful streaming completion.

## Version Changes

- `captain_claw` — 0.4.2 → 0.4.3
- `botport` — 0.4.2 → 0.4.3
- `desktop` — 0.4.2 → 0.4.3

## Internal

### New Methods
- `AgentModelMixin.set_byok_provider()` — Creates per-agent LLM provider from user-supplied credentials
- `AgentModelMixin.clear_byok_provider()` — Reverts agent to server's default provider
- `_extract_public_session_id()` — Reads public session ID from request cookies for REST endpoints
- `_record_visual_usage_streaming()` — Records LLM usage for streaming visual generation calls

### Modified Files
- `pyproject.toml` — Version 0.4.2 → 0.4.3
- `captain_claw/__init__.py` — Version bump
- `botport/pyproject.toml` — Version bump
- `botport/botport/__init__.py` — Version bump
- `desktop/package.json` — Version bump
- `captain_claw/agent_model_mixin.py` — `set_byok_provider()`, `clear_byok_provider()`, `_BYOK_ALLOWED_PROVIDERS`
- `captain_claw/agent_guard_mixin.py` — Pass `byok` flag to `record_llm_usage()`
- `captain_claw/web/ws_handler.py` — `is_public` in welcome message, `set_byok` and `clear_byok` handlers
- `captain_claw/web_server.py` — Init `_byok_active` on public agents, `byok_calls` in usage totals, BYOK filter in `_get_usage()`
- `captain_claw/web/rest_computer.py` — `_extract_public_session_id()`, BYOK-aware `_resolve_provider()`, `_record_visual_usage_streaming()`, BYOK flag in visual usage tracking
- `captain_claw/reflections.py` — Pass `byok` flag to `record_llm_usage()`
- `captain_claw/session/__init__.py` — `byok` column in `llm_usage` table, ALTER TABLE migration, `byok` param in `record_llm_usage()`, `byok` in `query_llm_usage()` results
- `captain_claw/web/static/computer.html` — BYOK toolbar button, BYOK modal
- `captain_claw/web/static/computer.js` — BYOK state management, modal logic, WebSocket message handling, auto-reconnect
- `captain_claw/web/static/computer.css` — BYOK modal and button styles using theme CSS variables
- `captain_claw/web/static/usage.html` — BYOK column, BYOK summary card, BYOK filter dropdown
- `README.md` — BYOK in feature table and Computer description, BYOK in Usage Dashboard description
- `USAGE.md` — BYOK documentation in Public Mode section, BYOK filter in Usage Dashboard section
