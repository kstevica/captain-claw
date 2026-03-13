# Summary: apis.py

# apis.py Summary

Implements a persistent cross-session API memory management tool that allows agents to register, retrieve, search, and maintain API configurations (base URLs, endpoints, authentication details) across multiple conversation sessions. The tool provides CRUD operations wrapped in a standardized Tool interface with comprehensive parameter validation and error handling.

## Purpose

Solves the problem of maintaining stateful API context across disconnected sessions. Agents frequently need to interact with multiple APIs, and this tool eliminates the need to re-specify API details in each conversation by persisting configurations in a session manager backend. Supports authentication credential storage, endpoint documentation, usage tracking, and semantic search capabilities.

## Most Important Functions/Classes

1. **ApisTool (class)** – Main Tool subclass implementing the six-action API management interface. Inherits from Tool registry system and delegates operations to session manager. Handles parameter validation, action routing, and error wrapping via ToolResult objects.

2. **execute() (async method)** – Entry point for all API operations. Extracts session context from kwargs, routes to appropriate action handler (_add, _list, _search, _info, _update, _remove), and wraps responses in ToolResult with success/error states. Implements centralized exception handling and logging.

3. **_add() (static async)** – Creates new API record with validation for required fields (name, base_url). Delegates to `sm.create_api()` to persist configuration including endpoints (JSON), auth_type (bearer/api_key/basic/none), credentials, description, purpose, and tags. Returns shortened ID for user reference.

4. **_search() (static async)** – Performs semantic/keyword search across registered APIs via `sm.search_apis()`. Supports finding APIs by name, description, purpose, or tags. Returns paginated results (limit=20) with formatted output showing auth type and base URL.

5. **_info() (static async)** – Retrieves complete API metadata via `sm.select_api()`. Displays all fields including usage statistics (use_count, last_used_at, created_at), authentication details, endpoint definitions, and descriptive metadata. Handles missing optional fields gracefully.

## Architecture & Dependencies

- **Dependencies**: Integrates with `captain_claw.session.get_session_manager()` for backend persistence, `captain_claw.logging` for structured logging, and `captain_claw.tools.registry.Tool/ToolResult` for framework compliance.
- **Session Manager Contract**: Expects session manager to implement async methods: `create_api()`, `list_apis()`, `search_apis()`, `select_api()`, `update_api()`, `delete_api()`.
- **Data Model**: API objects carry fields: id, name, base_url, endpoints (JSON string), auth_type, credentials, description, purpose, tags, use_count, last_used_at, created_at, source_session.
- **Design Pattern**: Stateless tool class with static action handlers; session context passed via kwargs["_session_id"]. All operations are async-first for non-blocking I/O.