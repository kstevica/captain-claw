# Summary: todo.py

# Todo.py Summary

**Summary:**
This module implements a persistent cross-session to-do list management tool integrated into the Captain Claw framework. It provides CRUD operations (add, list, update, remove) for task items that survive across sessions, with support for priority levels, status tracking, responsibility assignment (bot/human), and filtering capabilities.

**Purpose:**
Solves the problem of maintaining task state across multiple conversation sessions without losing context. Enables collaborative task management between AI bot and human users, with persistent storage and rich metadata (priority, status, tags, responsible party) for task organization and tracking.

**Most Important Functions/Classes:**

1. **TodoTool class** - Main tool implementation inheriting from Tool base class. Defines the tool interface with name, description, and parameter schema. Routes incoming requests to appropriate action handlers via the `execute()` method. Manages session context and error handling.

2. **execute() method** - Async entry point that dispatches actions (add/list/update/remove) to corresponding private methods. Extracts session ID and context from kwargs, instantiates session manager, and wraps all operations in exception handling with logging.

3. **_add() method** - Creates new todo items with validation of required content field. Delegates to `sm.create_todo()` with defaults (responsible="bot", priority="normal"). Returns formatted success message with truncated item ID.

4. **_list() method** - Retrieves todos with optional filtering by status and responsible party. Formats output as numbered list with priority/responsible/status/tags metadata. Supports pagination (limit=50) and session-scoped filtering.

5. **_update() method** - Modifies existing todo properties (status, responsible, priority, content, tags). Uses `sm.select_todo()` for ID resolution (supports both full IDs and #index notation), then applies updates via `sm.update_todo()`.

**Architecture Notes:**
- Dependency injection pattern: relies on `get_session_manager()` for data access
- Async/await throughout for non-blocking operations
- Priority ordering defined via `_PRIORITY_ORDER` dict (urgent=0 to low=3)
- Flexible ID resolution supporting both full UUIDs and numeric indices
- Session-aware: todos can be filtered by source session
- Logging integration via captain_claw.logging module