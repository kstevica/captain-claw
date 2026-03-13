# Summary: rest_playbooks.py

# rest_playbooks.py Summary

Complete REST API handler module for playbook CRUD operations, providing endpoints for listing, searching, retrieving, creating, updating, and deleting playbooks with comprehensive input validation and error handling.

## Purpose

Exposes HTTP endpoints for managing playbooks—reusable task execution patterns that capture "do" and "don't" guidelines, ratings, and contextual metadata. Serves as the primary interface between frontend clients and the session manager's playbook persistence layer, handling request parsing, validation, and response serialization.

## Most Important Functions/Classes

1. **list_playbooks()** — GET /api/playbooks handler that retrieves all playbooks with optional task_type filtering; supports pagination (limit=200) and returns serialized playbook dictionaries.

2. **search_playbooks()** — GET /api/playbooks/search handler implementing keyword search across playbooks with optional task_type filtering; validates required "q" parameter and returns up to 50 results.

3. **create_playbook()** — POST /api/playbooks handler that validates required fields (name, task_type, and at least one of do_pattern/dont_pattern), then delegates to session manager for persistence; returns 201 status on success.

4. **update_playbook()** — PATCH /api/playbooks/{id} handler supporting selective field updates (name, task_type, rating, patterns, descriptions, reasoning, tags); validates playbook existence and returns 400 if no fields provided.

5. **_playbook_dict()** — Helper function converting PlaybookEntry objects to JSON-serializable dictionaries, extracting all relevant fields including metadata (use_count, last_used_at, created_at, updated_at).

## Architecture & Dependencies

- **Framework**: aiohttp web framework for async HTTP handling
- **Session Manager**: Integrates with `captain_claw.session.get_session_manager()` for all data operations (list, search, load, create, update, delete)
- **Type Hints**: Uses TYPE_CHECKING pattern to avoid circular imports with WebServer
- **JSON Serialization**: Custom `_JSON_DUMPS` lambda handles non-standard types (dates, objects) via `default=str`
- **Error Handling**: Consistent validation with 400/404/500 status codes; validates JSON parsing, required fields, and operation success

## Key Design Patterns

- **Async/await**: All handlers are coroutines supporting concurrent request handling
- **Stateless handlers**: Each function receives WebServer parameter but primarily uses session manager singleton
- **Validation-first**: Input validation occurs before database operations (fail-fast approach)
- **Consistent response format**: All responses use `web.json_response()` with custom dumps function for date serialization