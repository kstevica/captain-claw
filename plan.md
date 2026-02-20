# Refactor main.py

## Problem
`main.py` is 4,154 lines with one giant `run_interactive()` closure containing ~70 nested functions. Three messaging platforms (Telegram/Slack/Discord) have near-identical code tripled across: send, mark-read, typing indicators, monitor events, audio, pairing, message handling, poll loops, and command handling.

## Strategy
Extract logical groups of functions into separate modules using **class-based dependency injection** to replace closure-captured state. The key shared dependencies are: `agent`, `ui`, `command_queue`, `followup_queue`, and per-platform state (bridges, approved users, pending pairings, offsets).

## New Files

### 1. `captain_claw/chat_handler.py` — Generic platform handler (~400 lines)
A base class `ChatPlatformHandler` that unifies all three platforms:

```python
class ChatPlatformHandler:
    def __init__(self, platform, agent, ui, bridge, config, command_queue, followup_queue, ...):
```

**Extracted functions (unified across platforms):**
- `send()` / `send_chat_action()` / `mark_read()` — delegates to bridge
- `monitor_event()` — logs to UI + session (currently `_telegram_monitor_event` / `_slack_monitor_event` / `_discord_monitor_event` — all identical except tool name)
- `send_audio_file()` / `maybe_send_audio_for_turn()` — currently triplicated
- `run_with_typing()` — typing indicator heartbeat (currently triplicated)
- `pair_unknown_user()` / `approve_pairing_token()` — pairing flow (currently triplicated)
- `handle_message()` — incoming message dispatch (currently `_handle_telegram_message` / `_handle_slack_message` / `_handle_discord_message`)
- `handle_command()` — slash command dispatch (wraps the shared `_handle_remote_command`)
- `poll_loop()` — background poll loop (currently triplicated)
- State: `approved_users`, `pending_pairings`, `offsets`
- `load_state()` / `save_state()` — persists approved/pending via app_state
- `start()` / `shutdown()` — lifecycle

Each platform instantiates with its bridge type and minor config differences (Telegram has `reply_to_message_id: int`, Slack has `reply_to_message_ts: str`, Discord has `reply_to_message_id: str` + `guild_id`).

### 2. `captain_claw/prompt_executor.py` — Prompt execution (~200 lines)
Extracts the core prompt execution logic:

**Extracted functions:**
- `run_prompt_in_active_session()` (lines 2034-2178) — the main prompt runner with streaming/non-streaming, cron support, callbacks
- `run_prompt_in_session()` (lines 2180-2279) — session-switching wrapper
- `dispatch_prompt_in_session()` (lines 1384-1476) — followup queue dispatch
- `enqueue_agent_task()` (lines 1302-1323) — lane-based queue wrapper

Receives: `agent`, `ui`, `command_queue`, `followup_queue`, and `cron_monitor_event` / `cron_chat_event` callbacks.

### 3. `captain_claw/cron_handler.py` — Cron scheduler & job execution (~350 lines)
Extracts all cron-related functions:

**Extracted functions:**
- `execute_cron_job()` (lines 2281-2388)
- `cron_scheduler_loop()` (lines 2390-2399)
- `cron_monitor()` / `cron_monitor_event()` / `cron_chat_event()` / `append_cron_history()` (lines 1478-1518)
- `parse_cron_add_args()` (lines 2012-2032)
- `run_script_or_tool_in_session()` (lines 1857-2010)
- `resolve_saved_file_for_kind()` (lines 1820-1855)
- `resolve_queue_settings_for_session()` / `update_active_session_queue_settings()` (lines 1332-1566)

### 4. `captain_claw/export_handler.py` — Session export & formatting (~200 lines)
Extracts export/rendering functions:

**Extracted functions:**
- `render_chat_export_markdown()` (lines 1574-1600)
- `render_monitor_export_markdown()` (lines 1602-1637)
- `render_pipeline_export_jsonl()` / `collect_pipeline_trace_entries()` (lines 1639-1686)
- `render_pipeline_summary_markdown()` (lines 1688-1767)
- `export_active_session_history()` (lines 1769-1818)

### 5. `captain_claw/remote_command_handler.py` — Shared remote command dispatch (~200 lines)
Extracts the `_handle_remote_command()` function (lines 2468-2812) which is the shared dispatcher for all slash commands from remote platforms. Also includes:
- `format_active_configuration_text()` (lines 2416-2439)
- `remote_help_text()` (lines 2441-2454)
- `format_recent_history()` (lines 2401-2414)

### 6. `captain_claw/tui_command_handler.py` — TUI main loop command dispatch (~350 lines)
Extracts the massive `elif` chain in the main loop (lines 3211-4066) into a class:

```python
class TUICommandHandler:
    def __init__(self, agent, ui, command_queue, followup_queue, cron_handler, prompt_executor, ...):
    async def handle(self, result: str) -> str:  # returns "continue", "break", or "prompt"
```

## Modified File

### `captain_claw/main.py` — Slim orchestrator (~350 lines)
Keeps:
- `main()` function (CLI entry, config loading, web/TUI dispatch)
- `run_interactive()` — now only: creates Agent, instantiates handlers, starts poll loops, runs main input loop
- `_run_cancellable()`, `_build_runtime_arg_parser()`, `version()`
- Platform handler startup/teardown
- The `while True` user input loop (now delegates to `TUICommandHandler`)

## Execution Order

1. Create `captain_claw/export_handler.py` (pure functions, no dependencies)
2. Create `captain_claw/remote_command_handler.py` (depends on agent, export_handler)
3. Create `captain_claw/prompt_executor.py` (depends on agent, ui, queues)
4. Create `captain_claw/cron_handler.py` (depends on prompt_executor, agent)
5. Create `captain_claw/chat_handler.py` (depends on prompt_executor, remote_command_handler)
6. Create `captain_claw/tui_command_handler.py` (depends on all above)
7. Rewrite `captain_claw/main.py` to wire everything together

Target: `main.py` drops from ~4,154 lines to ~350 lines.
