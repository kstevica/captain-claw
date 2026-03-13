# Summary: slash_commands.py

# slash_commands.py Summary

## Summary
This module implements WebSocket-based slash command handling for the Captain Claw web UI, providing 30+ commands for session management, entity operations (todos, contacts, scripts, APIs), agent control, and workspace operations. It manages command parsing, execution, and response broadcasting to connected clients, with special handling for destructive operations like the `/nuke` command that requires confirmation codes.

## Purpose
Solves the problem of providing a command-line-like interface within a web UI for controlling an AI agent system. Enables users to manage sessions, configure runtime parameters, manipulate persistent entities, and perform administrative tasks without direct API calls. Implements safety mechanisms (confirmation codes, time-limited tokens) for destructive operations.

## Most Important Functions/Classes/Procedures

### 1. **`handle_command(server, ws, raw)`**
Main command dispatcher that parses slash commands and routes them to appropriate handlers. Supports 30+ commands including `/help`, `/clear`, `/nuke`, `/config`, `/session`, `/todo`, `/contacts`, `/scripts`, `/apis`, `/pipeline`, `/planning`, `/models`, `/stop`, `/history`, `/compact`, `/new`, `/sessions`, `/skills`, `/screenshot`, `/reflection`, `/orchestrate`, and `/approve`. Returns formatted results via WebSocket.

### 2. **`_execute_nuke(server)`**
Comprehensive workspace reset function that atomically deletes all user data: workspace files (saved/ folder), deep memory documents (Typesense collection), semantic memory (SQLite FTS), datastore tables, sessions, and entities (todos, contacts, scripts, APIs). Includes fallback mechanisms for Typesense deletion via direct HTTP if collection not initialized. Creates fresh session post-nuke and broadcasts state updates.

### 3. **`handle_session_subcommand(server, args)`**
Handles `/session` subcommands: `list`, `switch`/`load`, `new`, `rename`, `description`, `model`, `protect`, and `export`. Manages session lifecycle (creation, switching, metadata updates), model assignment per-session, memory protection flags, and session history export in multiple formats (all, messages, artifacts, etc.).

### 4. **`handle_todo_command(server, args)` / `handle_contacts_command(server, args)` / `handle_scripts_command(server, args)` / `handle_apis_command(server, args)`**
Entity management handlers providing CRUD operations for persistent entities. Support list/search, add, info, remove, update, and assign operations. Use session manager for persistence and support field-based updates via `field=value` syntax with shell-like parsing (`shlex`). Include importance scoring for contacts and use tracking for scripts/APIs.

### 5. **`_handle_screenshot_command(server, ws, args)`**
Captures screen via `mss` (cross-platform) or native macOS APIs, saves to workspace, and optionally analyzes via vision model using customizable prompts loaded from instruction system. Integrates with chat handler to stream analysis results back to client.

## Architecture & Dependencies

**Key Dependencies:**
- `aiohttp.web`: WebSocket response handling
- `captain_claw.config`: Configuration management
- `captain_claw.web_server`: WebServer instance with agent/session manager
- `captain_claw.datastore`: Datastore manager for table operations
- `captain_claw.session_export`: Session history export
- `captain_claw.reflections`: Reflection generation/loading
- `captain_claw.instructions`: Instruction loader for customizable prompts
- `httpx`: Direct HTTP for Typesense collection deletion
- `sqlite3`: Fallback semantic memory cleanup
- `shlex`: Shell-like argument parsing for entity updates

**Global State:**
- `_pending_nuke`: Dictionary mapping WebSocket IDs to (confirmation_code, timestamp) tuples for nuke operation safety (120-second expiry).

**Design Patterns:**
- **Command dispatcher**: Single entry point (`handle_command`) with if-elif chains routing to specialized handlers
- **Subcommand pattern**: Entity handlers (`/todo`, `/contacts`, etc.) parse subcommands and delegate to specific operations
- **Fallback behavior**: Entity commands default to search/list if subcommand unrecognized
- **Broadcast pattern**: State changes trigger `server._broadcast()` to update all connected clients
- **Safety mechanisms**: Confirmation codes for `/nuke`, time-limited tokens, validation of user inputs
- **Async/await**: All I/O operations (session manager, datastore, HTTP) are async-first

**Role in System:**
Acts as the primary user interaction layer for the web UI, translating human-readable commands into agent/session manager operations. Bridges WebSocket communication with backend services (session manager, datastore, deep memory, semantic memory). Maintains workspace consistency through atomic operations and state synchronization broadcasts.