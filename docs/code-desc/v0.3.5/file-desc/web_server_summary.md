# Summary: web_server.py

# web_server.py Summary

**Summary:** Central aiohttp-based web server for Captain Claw that orchestrates the entire web UI, WebSocket communication, REST APIs, and integrations (Telegram, Google OAuth, BotPort). Manages agent lifecycle, session state, callback routing, and delegates handler logic to modular REST/static page modules.

**Purpose:** Provides a unified HTTP/WebSocket server that bridges the CLI-based Agent with a modern web interface, enabling real-time bidirectional communication, multi-session management, orchestration, cron scheduling, and third-party integrations while maintaining separation of concerns through delegated handler architecture.

**Most Important Functions/Classes/Procedures:**

1. **`WebServer.__init__` & `WebServer._init_agent`** — Initializes server state (config, agent, orchestrator, API pool, Telegram bridge, OAuth manager) and sets up callbacks for status/thinking/tool output streaming to connected WebSocket clients. Lazy-initializes orchestrator and API pool based on config.

2. **`WebServer._broadcast` & `WebServer._send`** — Core message distribution to all connected WebSocket clients (broadcast) or single client (send). Handles JSON serialization and stale connection cleanup. Enables real-time UI updates for status, thinking, tool output, and chat messages.

3. **Callback Methods** (`_status_callback`, `_thinking_callback`, `_tool_output_callback`, `_tool_stream_callback`, `_approval_callback`) — Receive events from Agent execution loop and broadcast them to UI. Includes smart filtering (e.g., `_THINKING_SILENT_TOOLS`), auto-extraction of images/audio from tool output, and inline tool output truncation for performance.

4. **`WebServer.create_app`** — Constructs aiohttp Application with 100+ route handlers covering WebSocket, REST APIs (sessions, cron, todos, contacts, scripts, APIs, datastore, playbooks, orchestrator, reflections), static pages, file uploads, and OAuth. Applies auth middleware if configured. Supports 50 MB file uploads.

5. **`_run_server` & `run_web_server`** — Async entry point that initializes agent, starts Telegram/hotkey daemons, BotPort client, cron scheduler, and aiohttp server. Handles graceful shutdown (Ctrl+C) with cleanup of orchestrator, API pool, WebSocket clients, and background tasks. Retries port binding on conflict.

**Key Architecture Patterns:**

- **Delegated Handlers:** 100+ route methods delegate to modular functions in `captain_claw.web.*` subpackages (e.g., `ws_handler`, `rest_instructions`, `rest_sessions`), keeping this file as a thin router/state container.
- **Callback-Driven Updates:** Agent execution feeds events through callbacks that broadcast to all connected WebSocket clients, enabling real-time UI synchronization without polling.
- **Lazy Initialization:** Orchestrator, API pool, OAuth manager, and Telegram bridge initialized only if configured, reducing startup overhead.
- **Multi-Integration Support:** Telegram bridge (with user pairing/session mapping), Google OAuth (with token injection), BotPort client, hotkey daemon, and cron scheduler all managed within single server lifecycle.
- **Runtime Context:** `_get_web_runtime_context` creates lightweight RuntimeContext for cron execution with minimal UI stub, enabling background task output delivery to Telegram.

**Dependencies:**
- `aiohttp` (web framework)
- `captain_claw.agent`, `captain_claw.agent_pool`, `captain_claw.session_orchestrator` (core execution)
- `captain_claw.config`, `captain_claw.instructions`, `captain_claw.logging` (config/logging)
- `captain_claw.google_oauth_manager`, `captain_claw.telegram_bridge` (integrations)
- Modular REST handlers in `captain_claw.web.*` (delegated logic)
- `subprocess` (audio playback), `json`, `asyncio`, `signal` (stdlib)

**Role in System:** Acts as the primary entry point and orchestrator for web-mode operation, managing agent lifecycle, WebSocket connections, REST API routing, and third-party integrations while delegating business logic to specialized modules.