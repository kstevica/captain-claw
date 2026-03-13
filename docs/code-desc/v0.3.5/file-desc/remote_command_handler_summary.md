# Summary: remote_command_handler.py

# remote_command_handler.py Summary

**Summary:** This module provides unified command and message handling for remote chat platforms (Telegram, Slack, Discord). It implements a comprehensive command dispatcher that processes slash commands and normal chat messages, routing them through platform-specific adapters while maintaining consistent behavior across all platforms.

**Purpose:** Solves the problem of managing multi-platform chat interactions with a single agent, handling command parsing, session management, skill invocation, contact/todo/script/API CRUD operations, and seamless integration with the agent's prompt execution pipeline. Abstracts platform-specific differences behind a unified interface.

---

## Architecture Overview

The module operates in two primary layers:

1. **Unified Command Handler** (`handle_remote_command`) - Processes all slash commands with consistent logic regardless of platform
2. **Platform Dispatcher** (`handle_platform_message`) - Routes incoming messages from any platform, handles platform-specific quirks, and delegates to the unified handler

---

## Most Important Functions/Classes

### 1. **`handle_remote_command(ctx, *, platform, raw_text, help_label, sender_label, send_text, execute_prompt)`**
   - **Purpose:** Core command processor handling 50+ slash commands across multiple domains
   - **Key Responsibilities:**
     - Parse and validate incoming command text
     - Route to appropriate handler (session, model, skill, todo, contact, script, API, orchestration, pipeline, cron)
     - Execute commands asynchronously with proper error handling
     - Return boolean indicating command was handled
   - **Command Categories Handled:**
     - **Session Management:** `/new`, `/session select`, `/session rename`, `/clear`, `/compact`
     - **Model Control:** `/models`, `/session model info`, `/session model set`
     - **Skills:** `/skill list`, `/skill search`, `/skill install`, `/skill invoke`
     - **Orchestration:** `/orchestrate` (multi-agent coordination)
     - **Pipeline Control:** `/pipeline info`, `/pipeline loop|contracts`
     - **Todo Management:** `/todo list`, `/todo add`, `/todo done`, `/todo remove`, `/todo assign`
     - **Contact Management:** `/contacts list`, `/contacts add`, `/contacts info`, `/contacts search`, `/contacts remove`, `/contacts importance`, `/contacts update`
     - **Script Management:** `/scripts list`, `/scripts add`, `/scripts info`, `/scripts search`, `/scripts remove`, `/scripts update`
     - **API Management:** `/apis list`, `/apis add`, `/apis info`, `/apis search`, `/apis remove`, `/apis update`
     - **Information:** `/help`, `/config`, `/history`, `/session info`, `/sessions`
     - **Cron Tasks:** `/cron` (one-off scheduled execution)
   - **Dependencies:** `RuntimeContext`, `PlatformAdapter`, `SessionOrchestrator`, `prompt_execution` module
   - **Return Value:** `True` if command was handled (regardless of success/failure)

### 2. **`handle_platform_message(ctx, platform, message)`**
   - **Purpose:** Main entry point for all incoming messages from remote platforms
   - **Key Responsibilities:**
     - Platform-specific message validation (e.g., Discord guild mention requirements)
     - User approval/pairing verification
     - Message text extraction and normalization (strip Telegram bot mentions)
     - Distinguish between slash commands and normal chat
     - Coordinate typing indicators and image/audio delivery
     - Error handling with monitoring callbacks
   - **Platform-Specific Logic:**
     - **Discord:** Skip guild messages unless bot is mentioned (configurable)
     - **Telegram:** Strip `@BotName` suffixes from commands
     - **All:** Extract user_id, username, channel_id, message_id for monitoring
   - **Dependencies:** `PlatformAdapter`, `run_prompt_in_active_session`, platform state management
   - **Error Handling:** Catches exceptions, logs to monitoring system, sends user-friendly error messages

### 3. **`format_active_configuration_text(ctx)`**
   - **Purpose:** Generate human-readable active configuration summary for `/config` command
   - **Output Includes:**
     - Active model (provider/name/id/source)
     - Workspace path
     - Pipeline mode (loop vs contracts)
     - Context size (max tokens)
     - Guard settings (input, output, script/tool with enabled/level)
   - **Dependencies:** `get_config()`, agent runtime model details

### 4. **`format_recent_history(ctx, limit=30)`**
   - **Purpose:** Format session message history for `/history` command
   - **Features:**
     - Retrieves last N messages from active session
     - Truncates long content to 220 chars with ellipsis
     - Displays role and content for each message
     - Shows session name and ID
   - **Dependencies:** `RuntimeContext.agent.session`

### 5. **`remote_help_text(ctx, platform_label)`**
   - **Purpose:** Generate platform-specific help text for `/help` command
   - **Output:** Lists all available commands with descriptions from `ctx.telegram_command_specs`
   - **Dependencies:** `RuntimeContext.telegram_command_specs`

---

## Key Design Patterns

### Command Result Parsing
Commands return structured strings that the handler parses:
- `"EXIT"` - Rejected (local-only)
- `"APPROVE_CHAT_USER:"` / `"APPROVE_TELEGRAM_USER:"` - Rejected (operator-only)
- `"ORCHESTRATE:<json>"` - Multi-agent orchestration payload
- `"SKILL_INVOKE:<json>"` / `"SKILL_ALIAS_INVOKE:<json>"` - Skill execution
- `"SESSION_MODEL_SET:<selector>"` - Model switching
- `"TODO_ADD:<text>"`, `"CONTACTS_UPDATE:<selector> <fields>"` - CRUD operations

### Async Callback Pattern
Commands receive two async callbacks:
- `send_text(str)` - Send response message to user
- `execute_prompt(prompt, display_prompt)` - Execute agent prompt with display label

### Platform Abstraction
`PlatformAdapter` handles:
- Message sending with reply-to support
- Typing indicators
- Image/audio delivery coordination
- User pairing/approval workflows
- Event monitoring

---

## Data Flow

```
Incoming Message (Telegram/Slack/Discord)
    ↓
handle_platform_message()
    ├─ Platform validation (Discord guild check)
    ├─ User approval verification
    ├─ Message text extraction & normalization
    ├─ Slash command detection
    │   ↓
    │   handle_remote_command()
    │   ├─ Parse command + arguments
    │   ├─ Route to handler (50+ command types)
    │   ├─ Execute with callbacks (send_text, execute_prompt)
    │   └─ Return True
    │
    └─ Normal chat message
        ↓
        run_prompt_in_active_session()
        ├─ Execute agent prompt
        ├─ Stream responses via adapter.send()
        └─ Deliver images/audio via _after_turn callback
```

---

## Integration Points

- **RuntimeContext:** Access to agent, session manager, UI, platform state
- **PlatformAdapter:** Platform-specific message operations
- **SessionOrchestrator:** Multi-agent coordination for `/orchestrate`
- **prompt_execution:** Core prompt execution and task enqueueing
- **Session Manager:** CRUD for sessions, todos, contacts, scripts, APIs
- **Agent:** Model selection, skill invocation, pipeline mode control
- **Config System:** Active configuration retrieval

---

## Notable Implementation Details

1. **JSON Payload Parsing:** Commands with complex arguments use JSON encoding (e.g., `/skill search`, `/orchestrate`)
2. **Selector Flexibility:** Contact/script/API/todo selection supports ID, index (#N), or name
3. **Field Update Pattern:** `/contacts update`, `/scripts update`, `/apis update` use `field=value` syntax with `shlex` parsing
4. **Note Appending:** Contact notes are appended rather than replaced
5. **Importance Pinning:** Contact importance can be locked with `importance_pinned` flag
6. **Error Resilience:** Graceful degradation with user-friendly error messages for all failure modes
7. **Monitoring Integration:** All message handling includes event monitoring for analytics/debugging