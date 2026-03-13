# Summary: platform_lifecycle.py

## Summary

`platform_lifecycle.py` manages the initialization and teardown of multi-platform chat integrations (Telegram, Slack, Discord) within the Captain Claw system. It orchestrates bridge creation, state loading, command registration, and background polling loops that continuously fetch and dispatch messages from each platform.

## Purpose

This module solves the problem of coordinating the lifecycle of multiple heterogeneous chat platform connections. It provides centralized initialization logic that validates configuration, instantiates platform-specific bridge objects, restores persisted state, and launches independent polling tasks. On shutdown, it gracefully cancels all background tasks and closes connections, preventing resource leaks and orphaned processes.

## Most Important Functions/Classes/Procedures

1. **`init_platforms(ctx: RuntimeContext)`**
   - Asynchronous initialization routine that processes configuration for all three platforms (Telegram, Slack, Discord). For each enabled platform, it instantiates the corresponding bridge, loads persisted state via `PlatformAdapter`, cleans up expired pairings, and spawns a background polling task. Includes validation checks for empty tokens and provides UI feedback on startup status.

2. **`teardown_platforms(ctx: RuntimeContext)`**
   - Graceful shutdown handler that iterates through all active platforms, cancels their polling tasks, and closes bridge connections. Implements exception handling to prevent one platform's failure from blocking cleanup of others.

3. **`_telegram_poll_loop(ctx: RuntimeContext)`**
   - Background coroutine that continuously polls Telegram's `get_updates()` API with configurable timeout. Maintains offset state to track processed updates and dispatches each incoming update to `handle_platform_message()` as a non-blocking task. Implements error recovery with exponential backoff (2-second sleep on failure).

4. **`_slack_poll_loop(ctx: RuntimeContext)` and `_discord_poll_loop(ctx: RuntimeContext)`**
   - Parallel polling implementations for Slack and Discord that follow the same pattern: fetch updates with offset tracking, dispatch messages asynchronously, and sleep between poll cycles. Both handle `CancelledError` for clean shutdown and log errors to the UI.

5. **`_register_telegram_commands(ctx: RuntimeContext)`**
   - Telegram-specific initialization that registers bot commands and menu buttons using the bridge's `set_my_commands()` and `set_chat_menu_button_commands()` methods. Provides user feedback on success/failure through the UI system.

## Architecture & Dependencies

- **Dependencies**: Relies on `RuntimeContext` for state management, platform-specific bridge classes (`TelegramBridge`, `SlackBridge`, `DiscordBridge`), `PlatformAdapter` for state persistence, and `handle_platform_message` from the remote command handler for message processing.
- **Async Pattern**: Heavily uses `asyncio.create_task()` for non-blocking message dispatch and `asyncio.CancelledError` for clean task cancellation.
- **State Management**: Each platform maintains configuration, enabled flag, bridge instance, polling task reference, and platform-specific state (offsets for Slack/Discord, pending pairings).
- **Error Handling**: Implements graceful degradation—missing tokens skip platform startup, and polling loop errors trigger UI notifications and retry delays without crashing the system.