"""Tool registry and base tool class."""

import asyncio
import json
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field

from captain_claw.config import get_config
from captain_claw.exceptions import (
    ToolBlockedError,
    ToolExecutionError,
    ToolNotFoundError,
)
from captain_claw.logging import get_logger

log = get_logger(__name__)


class ToolResult(BaseModel):
    """Result from tool execution."""

    success: bool = True
    content: str = ""
    error: str | None = None


class Tool(ABC):
    """Base class for all tools."""

    name: str = ""
    description: str = ""
    parameters: dict[str, Any] = Field(default_factory=dict)

    @abstractmethod
    async def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the tool.
        
        Args:
            **kwargs: Tool-specific arguments
        
        Returns:
            ToolResult with success status and content
        """
        pass

    def get_definition(self) -> dict[str, Any]:
        """Get the tool definition for LLM.
        
        Returns:
            OpenAI function-style definition
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }

    def validate_arguments(self, arguments: dict[str, Any]) -> None:
        """Validate tool arguments against schema.
        
        Args:
            arguments: Arguments to validate
        
        Raises:
            ValidationError if invalid
        """
        # Basic validation - could use Pydantic for more robust validation
        required = self.parameters.get("required", [])
        for field in required:
            if field not in arguments:
                raise ToolExecutionError(
                    self.name,
                    f"Missing required argument: {field}",
                )


class ToolRegistry:
    """Registry for managing available tools."""

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool.
        
        Args:
            tool: Tool instance to register
        """
        if not tool.name:
            raise ValueError("Tool must have a name")
        
        log.debug("Registering tool", tool=tool.name)
        self._tools[tool.name] = tool

    def unregister(self, name: str) -> None:
        """Unregister a tool.
        
        Args:
            name: Tool name to unregister
        """
        if name in self._tools:
            del self._tools[name]

    def get(self, name: str) -> Tool:
        """Get a tool by name.
        
        Args:
            name: Tool name
        
        Returns:
            Tool instance
        
        Raises:
            ToolNotFoundError if not found
        """
        if name not in self._tools:
            raise ToolNotFoundError(name)
        return self._tools[name]

    def list_tools(self) -> list[str]:
        """List all registered tool names.
        
        Returns:
            List of tool names
        """
        return list(self._tools.keys())

    def get_definitions(self) -> list[dict[str, Any]]:
        """Get all tool definitions for LLM.
        
        Returns:
            List of OpenAI function-style definitions
        """
        return [tool.get_definition() for tool in self._tools.values()]

    async def execute(
        self,
        name: str,
        arguments: dict[str, Any],
        confirm: bool = False,
    ) -> ToolResult:
        """Execute a tool by name.
        
        Args:
            name: Tool name
            arguments: Tool arguments
            confirm: Whether to confirm before execution
        
        Returns:
            ToolResult from execution
        
        Raises:
            ToolNotFoundError if tool not found
            ToolBlockedError if tool is blocked
            ToolExecutionError if execution fails
        """
        # Check if tool exists
        tool = self.get(name)
        
        # Check if tool is blocked
        config = get_config()
        # Check shell-specific blocked commands
        if name == "shell" and hasattr(config.tools.shell, 'blocked'):
            blocked_cmds = config.tools.shell.blocked or []
            cmd = arguments.get('command', '')
            for blocked in blocked_cmds:
                if blocked in cmd:
                    raise ToolBlockedError(name, f"Command contains blocked pattern: {blocked}")
        
        # Check if confirmation required
        if confirm and name in config.tools.require_confirmation:
            # For now, just proceed - UI should handle confirmation
            pass
        
        # Validate arguments
        tool.validate_arguments(arguments)
        
        # Execute with timeout
        try:
            log.info("Executing tool", tool=name, args=arguments)
            result = await asyncio.wait_for(
                tool.execute(**arguments),
                timeout=config.tools.shell.timeout if name == "shell" else 30,
            )
            log.info("Tool executed", tool=name, success=result.success)
            return result
            
        except asyncio.TimeoutError:
            raise ToolExecutionError(name, "Execution timed out")
        except Exception as e:
            log.error("Tool execution failed", tool=name, error=str(e))
            raise ToolExecutionError(name, str(e))


# Global registry
_registry: ToolRegistry | None = None


def get_tool_registry() -> ToolRegistry:
    """Get the global tool registry."""
    global _registry
    if _registry is None:
        _registry = ToolRegistry()
    return _registry


def set_tool_registry(registry: ToolRegistry) -> None:
    """Set the global tool registry."""
    global _registry
    _registry = registry
