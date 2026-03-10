"""Shell tool for executing commands."""

import asyncio
import os
import re
import shutil
import time
from typing import Any

from captain_claw.config import get_config
from captain_claw.logging import get_logger
from captain_claw.tools.registry import (
    Tool,
    ToolResult,
    extract_shell_base_commands,
    is_blocked_shell_command,
)

log = get_logger(__name__)

# ── Google Drive download blocking ────────────────────────────────────
# When the gws CLI is available, curl/wget to Google Drive/Docs URLs
# should be blocked — the gws tool handles auth and exports properly.

_GDRIVE_HOST_PATTERNS = (
    "docs.google.com", "drive.google.com",
    "sheets.google.com", "slides.google.com",
    "storage.googleapis.com",
)

_GDRIVE_DOWNLOAD_RE = re.compile(
    r"""(?:curl|wget)\s.*(?:"""
    + "|".join(re.escape(h) for h in _GDRIVE_HOST_PATTERNS)
    + r""")""",
    re.IGNORECASE,
)

_GDRIVE_SHELL_BLOCK_MSG = (
    "Do not use curl/wget to download Google Drive/Docs files — "
    "the gws tool handles authentication and export automatically.\n"
    "Use instead:\n"
    "  - gws(action='docs_read', file_id='...') to read Google Docs content\n"
    "  - gws(action='drive_download', file_id='...') to download files\n"
    "  - gws(action='drive_info', file_id='...') for file metadata"
)

# Commands that complete nearly instantly and should never hang for the
# full config timeout.  We use a short timeout (5 s) for these — they
# finish in <1 s under normal conditions; the 5 s budget only matters
# when the filesystem is extremely slow or the command hangs.
_QUICK_COMMANDS: frozenset[str] = frozenset({
    "cd", "ls", "pwd", "echo", "printf", "cat", "head", "tail",
    "mkdir", "rmdir", "touch", "cp", "mv", "rm", "ln",
    "chmod", "chown", "chgrp",
    "date", "cal", "whoami", "hostname", "uname", "env", "printenv",
    "which", "type", "file", "stat", "wc", "sort", "uniq",
    "basename", "dirname", "realpath", "readlink",
    "true", "false", "test", "[",
    "export", "unset", "set", "alias",
})
_QUICK_TIMEOUT = 5

# Script interpreters that typically run longer than simple commands.
# When the shell config timeout is low (e.g. 30 s), these get a minimum
# floor so that scripts have a reasonable initial window before the
# activity-based timeout kicks in.
_SCRIPT_COMMANDS: frozenset[str] = frozenset({
    "python3", "python", "python3.11", "python3.12", "python3.13",
    "node", "ruby", "perl", "bash", "sh", "zsh",
})
_SCRIPT_MIN_TIMEOUT = 120

# Hard wall-time cap for activity-based timeout extension (30 minutes).
# Even if a process is producing output, it cannot run longer than this.
_HARD_WALL_TIME = 1800


class ShellTool(Tool):
    """Execute shell commands."""

    name = "shell"
    description = "Execute a shell command and return its output."
    timeout_seconds = 120.0
    parameters = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The shell command to execute",
            },
            "timeout": {
                "type": "number",
                "description": "Timeout in seconds (optional, default from config)",
            },
        },
        "required": ["command"],
    }

    def __init__(self):
        self.config = get_config()
        # The shell tool manages its own timeouts internally with an
        # activity-based system + hard wall-time cap (_HARD_WALL_TIME).
        # Set the registry-level timeout to the hard cap so the outer
        # wrapper in tools/registry.py never kills a command before the
        # shell's own timeout handling can act.  Quick commands, normal
        # commands, and scripts all have appropriate internal timeouts
        # that will terminate them long before the hard cap.
        self.timeout_seconds = float(_HARD_WALL_TIME)

    @staticmethod
    def _is_script_command(command: str) -> bool:
        """Return True if *command* runs a script interpreter (python3, node, etc.).

        Checks ALL parts of chained commands (``cd foo && python3 bar.py``)
        so that ``python3`` is detected even if preceded by ``cd``.
        Used to enforce a minimum timeout floor so that scripts have a
        reasonable initial window before the activity-based timeout kicks in.
        """
        stripped = re.sub(r"^\s*(\w+=\S+\s+)*", "", command).strip()
        parts = re.split(r"\s*(?:&&|\|\|?|;)\s*", stripped)
        for part in parts:
            part = part.strip()
            if not part:
                continue
            base = part.split()[0].split("/")[-1] if part.split() else ""
            if base in _SCRIPT_COMMANDS:
                return True
        return False

    @staticmethod
    def _is_quick_command(command: str) -> bool:
        """Return True if *command* consists only of fast, non-blocking builtins.

        Handles pipelines (``ls | head``) and chains (``mkdir a && cd a``).
        If *any* part of the command is not in the quick-list, return False
        so the full timeout is used.
        """
        # Strip leading env vars (VAR=val cmd …)
        stripped = re.sub(r"^\s*(\w+=\S+\s+)*", "", command)
        # Split on shell operators to get individual commands
        parts = re.split(r"\s*(?:&&|\|\|?|;)\s*", stripped)
        for part in parts:
            part = part.strip()
            if not part:
                continue
            # Get the base command name (handle paths like /usr/bin/ls)
            base = part.split()[0].split("/")[-1] if part.split() else ""
            if base not in _QUICK_COMMANDS:
                return False
        return True

    def _is_command_safe(self, command: str) -> tuple[bool, str]:
        """Check if command is safe to execute.

        Args:
            command: Command to check

        Returns:
            Tuple of (is_safe, reason)
        """
        # Block curl/wget targeting Google Drive when gws is available.
        if _GDRIVE_DOWNLOAD_RE.search(command) and shutil.which("gws"):
            return False, _GDRIVE_SHELL_BLOCK_MSG

        # Check blocked patterns
        blocked, matched = is_blocked_shell_command(command, self.config.tools.shell.blocked)
        if blocked:
            if matched == "empty_command":
                return False, "Command is empty"
            if matched == "unparseable_command":
                return False, "Command is not parseable"
            return False, f"Command matches blocked pattern: {matched}"
        
        # Check allowed list (if non-empty)
        if self.config.tools.shell.allowed_commands:
            allowed = {
                str(item).strip()
                for item in self.config.tools.shell.allowed_commands
                if str(item).strip()
            }
            base_commands = extract_shell_base_commands(command)
            if not base_commands:
                return False, "Command is not parseable"
            for base_cmd in base_commands:
                normalized = base_cmd.split("/")[-1]
                if base_cmd not in allowed and normalized not in allowed:
                    return False, f"Command not in allowed list: {base_cmd}"
        
        return True, ""

    async def execute(self, command: str, timeout: int | None = None, **kwargs: Any) -> ToolResult:
        """Execute a shell command.
        
        Args:
            command: Shell command to execute
            timeout: Optional timeout override
        
        Returns:
            ToolResult with command output
        """
        # Check safety
        is_safe, reason = self._is_command_safe(command)
        if not is_safe:
            log.warning("Blocked unsafe command", command=command, reason=reason)
            return ToolResult(
                success=False,
                error=f"Command blocked: {reason}",
            )
        
        # Use config timeout if not provided; auto-shorten for trivial
        # commands so a stalled ``ls`` or ``mkdir`` won't block for 120 s.
        if timeout is None:
            if self._is_quick_command(command):
                timeout = _QUICK_TIMEOUT
            else:
                timeout = self.config.tools.shell.timeout
        timeout = max(1, int(timeout))

        # Script interpreters (python3, node, etc.) need a reasonable
        # initial window — the first API call from within a script may
        # block for many seconds before any output appears.  Enforce a
        # minimum inactivity timeout so the script isn't killed prematurely.
        if self._is_script_command(command):
            timeout = max(timeout, _SCRIPT_MIN_TIMEOUT)

        abort_event = kwargs.get("_abort_event")
        if isinstance(abort_event, asyncio.Event) and abort_event.is_set():
            return ToolResult(success=False, error="Command aborted")
        
        # Set up environment
        env = os.environ.copy()
        env["PATH"] = os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin")

        # Resolve shell CWD to the workspace root so that relative paths
        # in commands (e.g., "ls pdf-test/") behave consistently with other
        # tools (glob, read, write) that also resolve against the workspace.
        runtime_base = kwargs.get("_runtime_base_path")
        shell_cwd: str | None = str(runtime_base) if runtime_base is not None else None

        # Pre-create session-scoped directories under saved/ so that shell
        # commands writing to paths like "saved/tmp/{session_id}/file.png"
        # don't accidentally create a FILE at the session_id path when the
        # intermediate directory is missing.
        session_id = kwargs.get("_session_id")
        if runtime_base and session_id:
            from pathlib import Path

            _saved = Path(runtime_base) / "saved"
            for _cat in ("tmp", "scripts", "showcase", "media", "output", "downloads"):
                _dir = _saved / _cat / str(session_id)
                if not _dir.exists():
                    _dir.mkdir(parents=True, exist_ok=True)

        try:
            log.info("Executing shell command", command=command, timeout=timeout)
            stream_cb = kwargs.get("_stream_callback")

            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                cwd=shell_cwd,
            )

            # Read stdout/stderr line-by-line, streaming each line to the UI.
            # Track last activity time for activity-based timeout extension:
            # as long as the process is producing output, the timeout resets.
            stdout_chunks: list[str] = []
            stderr_chunks: list[str] = []
            last_activity = [time.monotonic()]

            async def _read_stream(
                stream: asyncio.StreamReader, collected: list[str], prefix: str = "",
            ) -> None:
                while True:
                    line = await stream.readline()
                    if not line:
                        break
                    last_activity[0] = time.monotonic()
                    text = line.decode("utf-8", errors="replace")
                    collected.append(text)
                    if stream_cb:
                        try:
                            stream_cb(prefix + text)
                        except Exception:
                            pass

            async def _collect() -> None:
                await asyncio.gather(
                    _read_stream(process.stdout, stdout_chunks),
                    _read_stream(process.stderr, stderr_chunks),
                )
                await process.wait()

            collect_task = asyncio.create_task(_collect())
            abort_wait_task: asyncio.Task[bool] | None = None
            if isinstance(abort_event, asyncio.Event):
                abort_wait_task = asyncio.create_task(abort_event.wait())
            try:
                # Activity-based timeout: instead of a single fixed wait, poll
                # periodically.  Whenever new stdout/stderr output arrives, the
                # inactivity deadline resets.  A hard wall-time cap prevents
                # infinite-running processes even if they keep producing output.
                wall_deadline = time.monotonic() + _HARD_WALL_TIME
                inactivity_deadline = time.monotonic() + timeout
                timed_out = False
                aborted = False

                while True:
                    wait_tasks: set[asyncio.Task[Any]] = {collect_task}
                    if abort_wait_task is not None:
                        wait_tasks.add(abort_wait_task)

                    remaining = max(0.1, min(
                        inactivity_deadline - time.monotonic(),
                        wall_deadline - time.monotonic(),
                    ))
                    check_interval = min(5.0, remaining)

                    done, _ = await asyncio.wait(
                        wait_tasks,
                        timeout=check_interval,
                        return_when=asyncio.FIRST_COMPLETED,
                    )

                    if collect_task in done:
                        await collect_task
                        break

                    if abort_wait_task is not None and abort_wait_task in done:
                        aborted = True
                        break

                    # Extend inactivity deadline if output was received recently
                    now = time.monotonic()
                    time_since_activity = now - last_activity[0]
                    if time_since_activity < timeout:
                        inactivity_deadline = max(
                            inactivity_deadline, last_activity[0] + timeout,
                        )

                    if now >= wall_deadline or now >= inactivity_deadline:
                        timed_out = True
                        break

                if aborted:
                    process.kill()
                    await process.wait()
                    collect_task.cancel()
                    try:
                        await collect_task
                    except asyncio.CancelledError:
                        pass
                    return ToolResult(success=False, error="Command aborted")

                if timed_out:
                    process.kill()
                    await process.wait()
                    collect_task.cancel()
                    try:
                        await collect_task
                    except asyncio.CancelledError:
                        pass
                    return ToolResult(
                        success=False,
                        error=f"Command timed out after {timeout}s of inactivity",
                    )
            except asyncio.CancelledError:
                process.kill()
                await process.wait()
                collect_task.cancel()
                raise
            finally:
                if abort_wait_task is not None and not abort_wait_task.done():
                    abort_wait_task.cancel()
                    try:
                        await abort_wait_task
                    except asyncio.CancelledError:
                        pass

            # Combine collected output
            stdout_text = "".join(stdout_chunks).strip()
            stderr_text = "".join(stderr_chunks).strip()

            output = stdout_text
            if stderr_text:
                output += f"\n[stderr] {stderr_text}"

            # Truncate if too long
            max_length = 10000
            if len(output) > max_length:
                output = output[:max_length] + f"\n... [truncated, {len(output)} total chars]"

            return ToolResult(
                success=process.returncode == 0,
                content=output or "[no output]",
            )
            
        except Exception as e:
            log.error("Shell command failed", command=command, error=str(e))
            return ToolResult(
                success=False,
                error=str(e),
            )
