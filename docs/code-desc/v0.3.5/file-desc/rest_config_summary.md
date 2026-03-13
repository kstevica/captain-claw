# Summary: rest_config.py

# rest_config.py Summary

**Summary:** REST API handlers for exposing system configuration, active sessions, and available commands via HTTP endpoints. Provides safe, non-sensitive configuration details to clients while maintaining security by filtering secrets and exposing only relevant runtime state.

**Purpose:** Solves the need for clients to discover system capabilities, monitor active sessions, and retrieve configuration metadata without direct access to internal config objects. Enables web UI and external tools to understand the current state of the Captain Claw agent system (model settings, enabled features, guards, sessions).

**Most Important Functions/Classes:**

1. **`get_config_summary(server, request)`** - Returns a sanitized configuration snapshot including model provider/name/temperature/max_tokens, context window settings, enabled tools, and guard statuses. Prioritizes runtime agent details over static config when available, ensuring clients see actual deployed parameters.

2. **`list_sessions_api(server, request)`** - Retrieves up to 20 active sessions from the session manager and returns a JSON array with session metadata (id, name, message count, timestamps, description). Provides session discovery and monitoring capability for the web interface.

3. **`get_commands_api(server, request)`** - Exposes the COMMANDS constant from web_server module, returning available command definitions to clients. Enables dynamic UI generation and command discovery without hardcoding endpoints.

**Architecture & Dependencies:**
- Depends on `aiohttp.web` for async HTTP request/response handling
- Imports `get_config()` from captain_claw.config for static configuration access
- Lazy imports session manager and COMMANDS to avoid circular dependencies
- Type hints with TYPE_CHECKING guard for WebServer type annotation
- All handlers follow async/await pattern consistent with aiohttp framework
- Returns JSON responses, suitable for REST API consumption