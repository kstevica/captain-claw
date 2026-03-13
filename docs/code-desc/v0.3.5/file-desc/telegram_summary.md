# Summary: telegram.py

# telegram.py Summary

**Summary:**
Telegram bridge integration module that manages per-user Agent instances, message routing, and command execution for Telegram users. Implements lazy agent creation with per-user asyncio locks for serialized request handling while enabling concurrent execution across different users. Handles message polling, callback queries, session management, and a comprehensive set of slash commands.

**Purpose:**
Solves the problem of integrating Telegram as a chat interface to the Captain Claw agent system while maintaining user isolation, session persistence, and concurrent request handling. Provides a complete Telegram UX including typing indicators, inline keyboards for suggested next steps, image upload/download, and command-driven session/cron job management.

---

## Most Important Functions/Classes/Procedures

### 1. **`_tg_get_or_create_agent(server, message) → Agent`**
Creates or retrieves a cached Agent instance for each Telegram user. Implements lazy initialization pattern where agents are created on first message and bound to user-specific sessions. Shares the main agent's LLM provider and tool registry but maintains independent session state, usage counters, and runtime flags. Ensures per-user personality lookup via `_user_id` attribute and registers default tools for interactive use.

### 2. **`_tg_process_with_typing(server, chat_id, text, ..., user_agent, user_id) → None`**
Orchestrates agent message processing with concurrent request handling and typing indicators. Implements per-user asyncio locks to serialize requests from the same user while allowing different users to run in parallel. Manages typing heartbeat animation, handles `/orchestrate` commands separately, extracts suggested next steps as inline keyboard buttons, collects and sends generated images back to Telegram, and broadcasts usage metrics to connected clients.

### 3. **`execute_telegram_command(server, raw, user_agent) → str | None`**
Dispatcher for slash commands with Telegram-specific restrictions (disables `/new`, `/sessions`, and session switching). Implements 20+ commands including `/clear`, `/history`, `/compact`, `/config`, `/session`, `/models`, `/pipeline`, `/cron`, `/skills`, and `/todo`. Returns `None` only for `/orchestrate` to allow agent fallthrough. Validates command arguments and provides contextual error messages.

### 4. **`_tg_handle_cron_command(server, args, user_agent) → str`**
Manages cron job lifecycle for Telegram users with subcommands: `list`, `add`, `remove`, `pause`, `resume`, `run`. Scopes jobs to user's session, computes next run times, integrates with cron dispatch system, and provides human-readable schedule formatting. Supports one-off prompt execution via cron infrastructure.

### 5. **`_handle_telegram_message(server, message) → None`**
Main message handler that validates user approval status, downloads attached photos to session media folders, strips bot mentions from commands, handles `/start` and `/help` responses, executes slash commands, and routes regular text to agent processing. Broadcasts chat messages to connected web clients and manages image context injection for multimodal requests.

### 6. **`_telegram_poll_loop(server) → None`**
Background long-polling task that continuously fetches Telegram updates with configurable timeout, maintains update offset tracking to prevent duplicates, queues updates for async dispatch, and implements exponential backoff on errors. Runs indefinitely until cancelled.

### 7. **`_telegram_worker(server) → None`**
Async task dispatcher that consumes queued Telegram updates and spawns concurrent handler tasks. Differentiates between callback queries (inline button presses) and regular messages, allowing independent concurrent processing of different update types.

### 8. **`_tg_pair_unknown_user(server, message) → None`**
Implements Telegram user approval workflow. Generates 8-character alphanumeric pairing tokens with configurable TTL, stores pending pairings with expiration timestamps, cleans up expired tokens, and sends pairing instructions to users. Broadcasts pairing requests to web UI for operator approval.

### 9. **`handle_approve_command(server, args) → str`**
Processes `/approve user telegram <token>` commands from operators. Validates token existence and expiration, moves users from pending to approved state, persists state changes, and notifies approved users via Telegram. Returns human-readable confirmation messages.

### 10. **`_handle_telegram_callback_query(server, cbq) → None`**
Handles inline keyboard button presses. Decodes action text from cached next steps using index-based lookup, echoes selected action back to user, creates/retrieves user agent, and processes action as regular message. Cleans up per-user caches after use.

---

## Architecture & Dependencies

**Key Design Patterns:**
- **Per-user agent isolation**: Each Telegram user gets a dedicated Agent instance with independent session, usage tracking, and runtime state
- **Lazy initialization**: Agents created on first message and cached for process lifetime
- **Concurrent request serialization**: Per-user asyncio locks prevent race conditions while enabling cross-user parallelism
- **State persistence**: User approvals, pending pairings, and user-session mappings stored in app_state via session manager
- **Callback caching**: Next steps cached per-user with index-based lookup for inline keyboard handling

**Critical Dependencies:**
- `captain_claw.agent.Agent` - Core agent class for message processing
- `captain_claw.session.Session` - User session storage and message history
- `captain_claw.telegram_bridge.TelegramBridge` - Low-level Telegram API wrapper
- `captain_claw.instructions.InstructionLoader` - System instruction management
- `captain_claw.next_steps` - Suggested action extraction from LLM responses
- `captain_claw.cron_dispatch` - Cron job execution and scheduling
- `asyncio` - Concurrent task management and per-user locking

**State Management:**
- `server._telegram_agents` - User ID → Agent instance cache
- `server._telegram_user_locks` - User ID → asyncio.Lock for serialization
- `server._telegram_user_sessions` - User ID → Session ID mapping (persisted)
- `server._approved_telegram_users` - Approved user metadata (persisted)
- `server._pending_telegram_pairings` - Pending approval tokens with TTL (persisted)
- `server._telegram_queue` - Async queue for update dispatch
- `server._telegram_offset` - Telegram update offset for polling

**Integration Points:**
- Web server broadcasts chat messages, usage metrics, and errors to connected clients
- Session manager handles persistence of user sessions and app state
- Orchestrator handles `/orchestrate` requests separately from agent
- Tool output and thinking callbacks integrated for real-time feedback
- Image generation collection and sending for multimodal responses