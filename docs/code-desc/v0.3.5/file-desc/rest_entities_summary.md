# Summary: rest_entities.py

## Summary

This module provides REST API handlers for CRUD operations on four core entity types (Todos, Contacts, Scripts, APIs) within the Captain Claw agent system. It implements a complete set of endpoints following RESTful conventions with consistent error handling, input validation, and async/await patterns using aiohttp. The module acts as the HTTP interface layer between client applications and the underlying session manager that persists and manages these entities.

## Purpose

Solves the problem of exposing entity management functionality through a standardized HTTP API, enabling external clients and UI applications to create, read, update, and delete todos, contacts, scripts, and API configurations without direct database access. Provides a thin but robust translation layer that validates inputs, handles agent initialization checks, and serializes domain objects to JSON responses.

## Most Important Functions/Classes/Procedures

1. **Todo CRUD Handlers** (`list_todos`, `create_todo`, `update_todo`, `delete_todo`)
   - Manage task/todo items with filtering by status, responsible party, and session. Support priority levels, tagging, and session linking. Create returns 201 status; updates validate required fields and return 404 if not found.

2. **Contact CRUD Handlers** (`list_contacts`, `search_contacts`, `create_contact`, `get_contact`, `update_contact`, `delete_contact`)
   - Manage contact records with rich metadata (position, organization, email, phone, importance ranking 1-10, privacy tiers). Include full-text search capability. Importance updates automatically set `importance_pinned` flag for persistence.

3. **Script CRUD Handlers** (`list_scripts`, `search_scripts`, `create_script`, `get_script`, `update_script`, `delete_script`)
   - Manage executable scripts with metadata (file path, language, purpose, creation reason). Require name and file_path as mandatory fields. Support session-based source tracking and tagging.

4. **API CRUD Handlers** (`list_apis`, `search_apis`, `create_api`, `get_api`, `update_api`, `delete_api`)
   - Manage API integrations with authentication details (base_url, endpoints, auth_type, credentials). Require name and base_url as mandatory fields. Support purpose documentation and credential storage.

5. **Common Patterns & Utilities**
   - All handlers check `server.agent` initialization (return 503 if missing). Use `_JSON_DUMPS` lambda for custom serialization (converts non-JSON types via `str()`). Consistent validation: strip whitespace, convert empty strings to None, validate required fields before delegation to session_manager. All list operations cap at 200 items; searches at 50 items. Update operations return 404 on missing entity; delete operations return boolean success status.

## Architecture & Dependencies

**Dependencies:**
- `aiohttp.web` — HTTP request/response handling
- `captain_claw.web_server.WebServer` — Server context providing agent access
- `json` — Serialization with custom default handler
- TYPE_CHECKING imports for type hints without runtime overhead

**Role in System:**
Acts as the HTTP API gateway layer. Receives requests from web clients, delegates business logic to `server.agent.session_manager` (which handles persistence, validation, and domain logic), and returns standardized JSON responses. Maintains separation of concerns: HTTP protocol handling here, data access and business rules in session_manager. All four entity types follow identical CRUD patterns, making the API predictable and maintainable. The module is stateless—all state lives in the agent's session manager.

**Key Design Decisions:**
- Async/await throughout for non-blocking I/O
- Defensive input validation (strip, type conversion, null checks) before passing to session_manager
- Consistent error responses with appropriate HTTP status codes (400 for bad input, 404 for not found, 503 for service unavailable, 201 for creation)
- Session-aware operations (todos and contacts track source_session for audit/context)
- Search endpoints separate from list endpoints with query parameter validation
- Importance field clamped to 1-10 range with automatic pinning flag