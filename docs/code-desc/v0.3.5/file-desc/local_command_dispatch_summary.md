# Summary: local_command_dispatch.py

# local_command_dispatch.py Summary

## Summary
A comprehensive command dispatcher that handles all local console commands in the Captain Claw agent system. Extracts the giant `elif` chain from the main interactive loop and routes 100+ command types to appropriate handlers. Returns control flow signals ("break", "continue", or None) to manage the main loop execution.

## Purpose
Solves the problem of managing a massive conditional chain in the interactive console loop by centralizing command parsing and dispatch logic. Handles session management, cron jobs, to-do items, contacts, scripts, APIs, monitoring, skills, and configuration commands. Provides a clean separation between command parsing and execution.

## Most Important Functions/Classes/Procedures

1. **`dispatch_local_command(ctx, result, user_input) -> str | None`**
   - Main async dispatcher that routes parsed command results to appropriate handlers
   - Returns "break" (exit loop), "continue" (skip to next prompt), or None (fall through to prompt execution)
   - Handles 100+ command types across 8 major categories

2. **Session Management Commands** (20+ handlers)
   - `NEW`, `CLEAR`, `SESSIONS`, `SESSION_SELECT`, `SESSION_RENAME`
   - `SESSION_PROTECT_ON/OFF`, `SESSION_MODEL_SET`, `SESSION_QUEUE_*` (mode, debounce, cap, drop, clear)
   - `SESSION_DESCRIPTION_*` (info, auto, set), `SESSION_EXPORT`, `SESSION_RUN`, `SESSION_PROCREATE`
   - Manages session lifecycle, protection, model selection, queue settings, descriptions, and cross-session operations

3. **Cron Job Management Commands** (10+ handlers)
   - `CRON_LIST`, `CRON_HISTORY`, `CRON_ONEOFF`, `CRON_ADD`, `CRON_REMOVE`, `CRON_PAUSE`, `CRON_RESUME`, `CRON_RUN`
   - Integrates with `cron_dispatch` module for scheduling, execution, and monitoring
   - Supports one-off prompts, scheduled jobs, and manual execution

4. **Data Management Commands** (40+ handlers across 4 categories)
   - **To-Do**: `TODO_LIST`, `TODO_ADD`, `TODO_DONE`, `TODO_REMOVE`, `TODO_ASSIGN`
   - **Contacts**: `CONTACTS_LIST`, `CONTACTS_ADD`, `CONTACTS_INFO`, `CONTACTS_SEARCH`, `CONTACTS_UPDATE`, `CONTACTS_IMPORTANCE`, `CONTACTS_REMOVE`
   - **Scripts**: `SCRIPTS_LIST`, `SCRIPTS_ADD`, `SCRIPTS_INFO`, `SCRIPTS_SEARCH`, `SCRIPTS_UPDATE`, `SCRIPTS_REMOVE`
   - **APIs**: `APIS_LIST`, `APIS_ADD`, `APIS_INFO`, `APIS_SEARCH`, `APIS_UPDATE`, `APIS_REMOVE`
   - Each category supports CRUD operations, search, and metadata updates

5. **Monitoring & Pipeline Commands** (15+ handlers)
   - `MONITOR_ON/OFF`, `MONITOR_TRACE_ON/OFF`, `MONITOR_PIPELINE_ON/OFF`, `MONITOR_FULL_ON/OFF`, `MONITOR_SCROLL_STATUS`, `MONITOR_SCROLL`
   - `PIPELINE_INFO`, `PIPELINE_MODE`, `PLANNING_ON/OFF`, `ORCHESTRATE`
   - Controls execution pipeline modes (loop vs. contracts), monitoring verbosity, and session orchestration

6. **Skills Management Commands** (5+ handlers)
   - `SKILLS_LIST`, `SKILL_SEARCH`, `SKILL_INSTALL`, `SKILL_INVOKE`, `SKILL_ALIAS_INVOKE`
   - Integrates skill catalog search, GitHub installation, dependency management, and command invocation

7. **Utility Commands**
   - `EXIT`, `APPROVE_CHAT_USER`, `APPROVE_TELEGRAM_USER`, `CONFIG`, `HISTORY`, `COMPACT`
   - `MODELS`, `SESSION_INFO`, `SESSION_MODEL_INFO`, `SESSION_QUEUE_INFO`

## Architecture & Dependencies

**Key Dependencies:**
- `captain_claw.cron_dispatch` - Cron job execution and scheduling
- `captain_claw.prompt_execution` - Prompt dispatch and queue management
- `captain_claw.session_export` - Session history export
- `captain_claw.execution_queue` - Command lane and queue resolution
- `captain_claw.platform_adapter` - Chat platform token approval
- `captain_claw.remote_command_handler` - Configuration formatting
- `captain_claw.session_orchestrator` - Multi-session orchestration (lazy imported)

**Data Flow:**
1. User input → parsed into `result` string by command parser (not shown)
2. `dispatch_local_command()` receives parsed result and RuntimeContext
3. Routes to appropriate handler based on command prefix/type
4. Handler performs operation (DB updates, queue management, etc.)
5. Returns control flow signal to main loop

**Design Patterns:**
- **Command Prefix Routing**: Uses string matching on result prefixes (e.g., "SESSION_SELECT:", "CRON_ADD:")
- **Payload Parsing**: JSON payloads for complex commands with multiple parameters
- **Error Handling**: Consistent error/success messaging via `ui.print_error()` and `ui.print_success()`
- **Async/Await**: All database and agent operations are async
- **Session Context**: Most commands operate on `agent.session` (active session) or accept session selectors
- **Queue Management**: Integrates with `ctx.followup_queue` and `ctx.command_queue` for execution ordering

**Notable Architectural Decisions:**
- Centralized dispatcher avoids massive if-elif chains in main loop
- Consistent return values enable clean loop control flow
- JSON payloads allow complex arguments without shell parsing complexity
- Lazy import of `SessionOrchestrator` to avoid circular dependencies
- Session switching for `/session run` uses try-finally to restore previous session
- Contact/Script/API updates use `shlex.split()` for field=value parsing with proper quoting support