# Summary: rest_sessions.py

# rest_sessions.py Summary

This module provides REST API endpoints for session lifecycle management, including listing, retrieving, updating, deleting, and exporting chat/monitoring sessions. It integrates with the session manager and LLM agent to enable full CRUD operations plus advanced features like auto-description generation and multi-format exports.

## Purpose

Solves the problem of exposing session management capabilities through a RESTful HTTP interface, allowing clients to browse session history, search sessions, modify metadata, generate AI-powered descriptions, and export session data in markdown format for archival or sharing purposes.

## Most Important Functions/Classes

1. **`list_sessions()`** – Retrieves all sessions with optional full-text search filtering on name and description; returns paginated list with metadata (id, name, message count, timestamps, description).

2. **`get_session_detail()`** – Loads and returns complete session object including all messages and metadata; primary endpoint for viewing full conversation history.

3. **`auto_describe_session()`** – Invokes the LLM agent to generate a concise multi-sentence description of session content; persists generated description to session metadata for future reference.

4. **`export_session()`** – Renders session history as downloadable markdown with three modes (chat-only, monitor-only, or combined); handles filename sanitization and HTTP attachment headers for browser downloads.

5. **`update_session()` / `delete_session()` / `bulk_delete_sessions()`** – Core mutation operations for renaming sessions, updating descriptions, and removing sessions individually or in batch; includes validation and error handling.

## Architecture & Dependencies

- **Framework**: aiohttp web framework with async/await patterns
- **Session Management**: Depends on `captain_claw.session.SessionManager` (singleton pattern via `get_session_manager()`)
- **LLM Integration**: Uses `server.agent.generate_session_description()` for AI-powered metadata generation
- **Export Utilities**: Leverages `captain_claw.session_export` module for markdown rendering logic
- **Logging**: Integrated with captain_claw logging system for error tracking
- **Type Safety**: Uses TYPE_CHECKING for forward references to avoid circular imports

**Role in System**: Acts as the HTTP gateway layer between frontend clients and the session persistence/management layer; enables full session lifecycle operations and data export workflows.