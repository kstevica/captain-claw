# Summary: scripts.py

# scripts.py Summary

Implements a persistent cross-session script/file memory management tool that allows registration, retrieval, searching, and lifecycle management of code scripts with rich metadata. The tool integrates with a session manager backend to provide durable storage of script references, descriptions, and usage tracking across multiple sessions.

## Purpose

Solves the problem of maintaining institutional knowledge about scripts created during development sessions. Enables developers to catalog, discover, and reference previously created scripts without losing context about why they were created, what they do, or where they're located. Supports rapid script reuse and prevents duplication of effort.

## Most Important Functions/Classes

1. **ScriptsTool (class)** - Main tool implementation inheriting from Tool base class. Defines the tool interface with six actions (add, list, search, info, update, remove) and routes requests through the execute() method. Maintains parameter schema for OpenAPI-style tool invocation.

2. **execute() (async method)** - Central dispatcher that routes action requests to appropriate handler methods. Extracts session_id from kwargs, instantiates session manager, and wraps all operations in exception handling with structured logging.

3. **_add() (async static method)** - Registers new scripts with validation of required fields (name, file_path). Creates script record via session manager with optional metadata (description, purpose, language, created_reason, tags) and returns abbreviated ID for reference.

4. **_search() (async static method)** - Full-text search across script catalog with configurable query string. Returns ranked results limited to 20 items, displaying name, language, file path, and ID for quick identification.

5. **_info() (async static method)** - Retrieves complete script metadata including creation timestamp, last usage time, use count, and all descriptive fields. Provides comprehensive view for script inspection and decision-making about reuse.

## Architecture & Dependencies

- **Dependencies**: captain_claw.logging (structured logging), captain_claw.session (session manager for persistence), captain_claw.tools.registry (Tool base class and ToolResult)
- **Pattern**: Async/await throughout with static method handlers for each action, enabling concurrent request handling
- **Data Flow**: User request → execute() dispatcher → action-specific handler → session manager backend → ToolResult response
- **Session Integration**: Captures source_session_id on script creation to track origin; enables cross-session script discovery
- **Error Handling**: Comprehensive validation of required parameters, graceful error returns with descriptive messages, exception wrapping with logging