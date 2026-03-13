# Summary: rest_reflections.py

# rest_reflections.py Summary

## Summary
REST API handler module for managing self-reflections in the Captain Claw system. Provides four endpoints for listing, retrieving, generating, and deleting reflection records with proper error handling and async/await patterns.

## Purpose
Exposes reflection management functionality through HTTP endpoints, allowing clients to query historical reflections, access the active reflection, trigger new reflection generation via the agent, and remove outdated reflections. Acts as the HTTP interface layer between web clients and the core reflection engine.

## Most Important Functions/Classes

1. **list_reflections_api(server, request)**
   - Retrieves all reflections paginated by limit parameter (default 50)
   - Returns JSON array of reflection objects sorted newest-first
   - Converts internal reflection objects to serializable dictionaries

2. **get_latest_reflection(server, request)**
   - Fetches the currently active (most recent) reflection
   - Returns 404 if no reflections exist
   - Single-object response for client's current reflection state

3. **trigger_reflection(server, request)**
   - Asynchronously generates a new reflection using the agent
   - Validates agent availability (returns 503 if unavailable)
   - Includes exception handling with logging and error response formatting
   - Critical integration point with the agent system

4. **delete_reflection_api(server, request)**
   - Removes a reflection by timestamp identifier from URL path parameter
   - Returns 404 if reflection doesn't exist
   - Simple boolean success confirmation response

## Architecture & Dependencies
- **Framework**: aiohttp (async HTTP server)
- **Dependencies**: Imports reflection operations from `captain_claw.reflections` module (lazy imports within handlers)
- **Type hints**: Uses TYPE_CHECKING pattern for WebServer type annotation without circular imports
- **Logging**: Integrated logger for warning-level events during reflection generation failures
- **Error handling**: Consistent HTTP status codes (404 not found, 503 service unavailable, 500 server error)
- **Integration**: Tightly coupled with WebServer instance and agent availability checks