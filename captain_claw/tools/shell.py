"""Shell tool for executing commands."""

import asyncio
import os
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


class ShellTool(Tool):
    """Execute shell commands."""

    name = "shell"
    description = "Execute a shell command and return its output."
    timeout_seconds = 30.0
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
        self.timeout_seconds = float(getattr(self.config.tools.shell, "timeout", 30) or 30)

    def _is_command_safe(self, command: str) -> tuple[bool, str]:
        """Check if command is safe to execute.
        
        Args:
            command: Command to check
        
        Returns:
            Tuple of (is_safe, reason)
        """
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
        
        # Use config timeout if not provided
        if timeout is None:
            timeout = self.config.tools.shell.timeout
        timeout = max(1, int(timeout))

        abort_event = kwargs.get("_abort_event")
        if isinstance(abort_event, asyncio.Event) and abort_event.is_set():
            return ToolResult(success=False, error="Command aborted")
        
        # Set up environment
        env = os.environ.copy()
        env["PATH"] = os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin")
        
        try:
            log.info("Executing shell command", command=command, timeout=timeout)
            
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
            
            communicate_task = asyncio.create_task(process.communicate())
            abort_wait_task: asyncio.Task[bool] | None = None
            if isinstance(abort_event, asyncio.Event):
                abort_wait_task = asyncio.create_task(abort_event.wait())
            try:
                wait_tasks: set[asyncio.Task[Any]] = {communicate_task}
                if abort_wait_task is not None:
                    wait_tasks.add(abort_wait_task)
                done, _ = await asyncio.wait(
                    wait_tasks,
                    timeout=timeout,
                    return_when=asyncio.FIRST_COMPLETED,
                )

                if communicate_task in done:
                    stdout, stderr = await communicate_task
                elif abort_wait_task is not None and abort_wait_task in done:
                    process.kill()
                    await process.wait()
                    communicate_task.cancel()
                    try:
                        await communicate_task
                    except asyncio.CancelledError:
                        pass
                    return ToolResult(success=False, error="Command aborted")
                else:
                    process.kill()
                    await process.wait()
                    communicate_task.cancel()
                    try:
                        await communicate_task
                    except asyncio.CancelledError:
                        pass
                    return ToolResult(
                        success=False,
                        error=f"Command timed out after {timeout}s",
                    )
            except asyncio.CancelledError:
                process.kill()
                await process.wait()
                communicate_task.cancel()
                raise
            finally:
                if abort_wait_task is not None and not abort_wait_task.done():
                    abort_wait_task.cancel()
                    try:
                        await abort_wait_task
                    except asyncio.CancelledError:
                        pass
            
            # Decode output
            stdout_text = stdout.decode("utf-8", errors="replace").strip()
            stderr_text = stderr.decode("utf-8", errors="replace").strip()
            
            # Combine stdout and stderr
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
