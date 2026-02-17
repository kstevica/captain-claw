"""Shell tool for executing commands."""

import asyncio
import os
import re
from typing import Any

from captain_claw.config import get_config
from captain_claw.logging import get_logger
from captain_claw.tools.registry import Tool, ToolResult

log = get_logger(__name__)


class ShellTool(Tool):
    """Execute shell commands."""

    name = "shell"
    description = "Execute a shell command and return its output."
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

    def _is_command_safe(self, command: str) -> tuple[bool, str]:
        """Check if command is safe to execute.
        
        Args:
            command: Command to check
        
        Returns:
            Tuple of (is_safe, reason)
        """
        # Check blocked patterns
        for blocked in self.config.tools.shell.blocked:
            if blocked in command:
                return False, f"Command matches blocked pattern: {blocked}"
        
        # Check allowed list (if non-empty)
        if self.config.tools.shell.allowed_commands:
            # Extract base command
            base_cmd = command.strip().split()[0] if command.strip() else ""
            if base_cmd not in self.config.tools.shell.allowed_commands:
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
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                return ToolResult(
                    success=False,
                    error=f"Command timed out after {timeout}s",
                )
            
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
