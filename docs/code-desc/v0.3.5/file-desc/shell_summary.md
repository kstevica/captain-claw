# Summary: shell.py

# shell.py Summary

This module implements a secure, timeout-intelligent shell command execution tool for an AI agent system. It provides sophisticated command safety validation, activity-based timeout management with hard wall-time caps, and real-time output streaming. The tool intelligently adjusts timeouts based on command type (quick builtins vs. long-running scripts) and prevents abuse of Google Drive downloads when alternative tools are available.

## Purpose

Solves the problem of safely executing arbitrary shell commands within an AI agent framework while preventing:
- Infinite-running or hung processes through intelligent timeout management
- Unsafe command execution via allowlist/blocklist validation
- Misuse of curl/wget for Google Drive access (redirects to proper `gws` tool)
- Premature termination of legitimate long-running scripts
- Output flooding through truncation limits

## Most Important Functions/Classes

1. **`ShellTool` class** – Main Tool implementation inheriting from registry's `Tool` base class. Manages command execution lifecycle, timeout configuration, and safety checks. Exposes `execute()` async method as the primary interface for command execution.

2. **`execute(command, timeout, **kwargs)` method** – Core async execution handler. Orchestrates safety validation, timeout calculation, subprocess creation, real-time output streaming via callbacks, activity-based timeout extension, and graceful abort handling. Returns `ToolResult` with success status and output/error content.

3. **`_is_command_safe(command)` method** – Security validation gate checking three layers: (a) blocks curl/wget to Google Drive/Docs when `gws` tool available, (b) rejects commands matching configured blocklist patterns, (c) enforces allowlist if configured. Returns tuple of (is_safe: bool, reason: str).

4. **`_is_quick_command(command)` method** – Static analyzer detecting trivial, fast-completing commands (ls, mkdir, echo, etc.) by parsing shell operators (&&, |, ;) and checking base command names against `_QUICK_COMMANDS` frozenset. Returns bool to trigger 5-second timeout instead of default 120s.

5. **`_is_script_command(command)` method** – Static analyzer identifying script interpreters (python3, node, ruby, bash) across chained commands. Enforces 120-second minimum timeout floor to prevent premature termination during initial API calls or slow startup phases.

## Architecture & Dependencies

**Key Dependencies:**
- `asyncio` – Async subprocess management and timeout orchestration
- `captain_claw.config` – Runtime configuration (timeouts, blocked/allowed commands)
- `captain_claw.logging` – Structured logging
- `captain_claw.tools.registry` – `Tool` base class, `ToolResult` return type, command validation helpers

**Timeout Strategy:**
- **Quick commands** (ls, pwd, mkdir): 5-second timeout
- **Normal commands**: Config-driven (default 120s)
- **Script interpreters** (python3, node): 120-second minimum floor
- **Hard wall-time cap**: 1800 seconds (30 minutes) absolute maximum regardless of output activity
- **Activity-based extension**: Inactivity deadline resets whenever stdout/stderr receives data, allowing responsive processes to run indefinitely (up to hard cap)

**Safety Layers:**
1. Google Drive download blocking (redirects to `gws` tool)
2. Configured blocklist pattern matching
3. Optional allowlist enforcement (if non-empty)

**Output Handling:**
- Real-time line-by-line streaming via `_stream_callback` for UI updates
- Combines stdout/stderr with `[stderr]` prefix for clarity
- Truncates output exceeding 10,000 characters with total length indicator
- Tracks last activity timestamp to drive timeout extension logic

**Abort Support:**
- Respects `_abort_event` asyncio.Event for graceful cancellation
- Kills subprocess and cleans up tasks on abort or timeout