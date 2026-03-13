# Summary: rest_direct_api.py

# rest_direct_api.py Summary

**Summary:** REST API handler module for managing and executing Direct API Calls with full CRUD operations. Provides endpoints to create, retrieve, update, delete, and execute stored API call configurations with built-in safety constraints (DELETE method blocking) and authentication support.

**Purpose:** Solves the problem of storing, managing, and replaying HTTP API calls in a controlled manner. Enables users to define reusable API call templates with authentication, headers, and payloads, then execute them on-demand with optional runtime parameter overrides. Includes usage tracking and response logging for audit trails.

**Most Important Functions/Classes:**

1. **`execute_call()`** — Core execution handler that loads a stored API call definition, resolves authentication headers, parses runtime payload/query parameters, invokes `ApiReplayEngine.replay()`, and records usage metrics. Returns detailed response metadata (status code, elapsed time, response body/headers, errors).

2. **`create_call()`** — POST handler for creating new API call entries with comprehensive validation: enforces required fields (name, url), blocks DELETE method for safety, validates HTTP method against whitelist (GET, POST, PUT, PATCH), and persists all metadata (auth config, headers, description, tags, app_name).

3. **`update_call()`** — PATCH handler for selective field updates with same safety constraints as creation. Supports partial updates across 12+ configurable fields (name, url, method, auth credentials, headers, payloads, tags).

4. **`list_calls()` / `get_call()` / `delete_call()`** — Standard CRUD read/delete operations with optional app_name filtering for list operations and 404 handling for missing resources.

5. **`_call_dict()`** — Helper serializer converting DirectApiCallEntry ORM objects to JSON-friendly dictionaries, exposing 16 fields including usage statistics (use_count, last_used_at, last_status_code, last_response_preview) and timestamps.

**Architecture & Dependencies:**
- **Session Manager Integration:** All operations delegate to `get_session_manager()` for data persistence (async database operations)
- **API Replay Engine:** Leverages `ApiReplayEngine.replay()` for HTTP execution and `ApiReplayEngine.resolve_auth_headers()` for auth handling (supports multiple auth types)
- **Framework:** Built on aiohttp web framework with async/await patterns throughout
- **Safety Layer:** Explicit DELETE method blocking at three points (creation validation, update validation, execution time) to prevent accidental destructive operations
- **Flexible Authentication:** Supports auth_type/auth_token/auth_source fields with header merging strategy (extra headers + auth headers override)
- **URL Parsing:** Splits full URLs into base_url + endpoint components for ApiReplayEngine compatibility
- **Payload Flexibility:** Accepts payloads and query_params as either JSON strings or native dicts with graceful fallback on parse errors