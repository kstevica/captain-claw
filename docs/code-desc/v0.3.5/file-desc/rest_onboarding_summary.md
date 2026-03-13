# Summary: rest_onboarding.py

# rest_onboarding.py Summary

**Summary:** REST API endpoints for managing the web-based onboarding wizard in Captain Claw. Handles onboarding status checks, provider connection validation, OAuth token retrieval, and configuration persistence. Provides four core endpoints that bridge the frontend onboarding UI with backend configuration and provider management systems.

**Purpose:** Solves the problem of guiding new users through initial system setup by exposing onboarding operations as HTTP endpoints. Enables validation of AI provider credentials before saving, retrieves existing OAuth tokens from Codex CLI, and persists user configuration choices while hot-reloading the global config state.

**Most Important Functions/Classes:**

1. **`get_onboarding_status()`** — Determines whether onboarding should be displayed to the user by checking system state; returns boolean flag indicating if initial setup is required.

2. **`post_onboarding_validate()`** — Validates provider connection credentials (API key, base URL, model) by calling backend validation logic; returns success/error status to prevent invalid configurations from being saved.

3. **`get_codex_auth()`** — Reads OpenAI OAuth tokens from `~/.codex/auth.json` (Codex CLI authentication file); extracts and returns access token and account ID for seamless integration with existing CLI authentication.

4. **`post_onboarding_save()`** — Persists onboarding form data to configuration file via `save_onboarding_config()`; triggers hot-reload of global config singleton to apply changes without restart.

**Architecture & Dependencies:**

- **Framework:** aiohttp (async HTTP server)
- **Key Dependencies:** `captain_claw.config` (Config management), `captain_claw.onboarding` (business logic for validation/saving)
- **Integration Points:** WebServer instance, global config singleton pattern
- **Error Handling:** JSON-based error responses with HTTP status codes (400 for malformed requests, 500 for server errors)
- **File I/O:** Reads Codex auth tokens from user home directory; writes configuration to system config path