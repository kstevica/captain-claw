"""Tool registry and base tool class."""

import asyncio
import fnmatch
import re
import shlex
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, Field, model_validator

from captain_claw.config import get_config
from captain_claw.exceptions import (
    ToolBlockedError,
    ToolExecutionError,
    ToolNotFoundError,
)
from captain_claw.logging import get_logger

log = get_logger(__name__)

_ASSIGNMENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=.*$")
_SHELL_SEPARATOR_TOKENS = {";", "&&", "||", "|", "&"}
_SHELL_WRAPPER_TOKENS = {"sudo", "command", "builtin", "nohup", "time"}


def _normalize_tool_name(value: str) -> str:
    """Normalize tool names for policy comparisons."""
    return str(value or "").strip().lower()


def _compile_shell_pattern(pattern: str) -> re.Pattern[str]:
    """Compile regex pattern with literal fallback for invalid regex input."""
    try:
        return re.compile(pattern)
    except re.error:
        return re.compile(re.escape(pattern))


def _tokenize_shell_command(command: str) -> list[str]:
    """Tokenize shell command while preserving control operators."""
    lexer = shlex.shlex(command, posix=True, punctuation_chars=";&|")
    lexer.whitespace_split = True
    lexer.commenters = ""
    return list(lexer)


def _split_shell_segments(command: str) -> list[list[str]]:
    """Split shell command into tokenized segments separated by control operators."""
    tokens = _tokenize_shell_command(command)
    segments: list[list[str]] = []
    current: list[str] = []
    for token in tokens:
        if token in _SHELL_SEPARATOR_TOKENS:
            if current:
                segments.append(current)
                current = []
            continue
        current.append(token)
    if current:
        segments.append(current)
    return segments


def _extract_segment_base_command(tokens: list[str]) -> str:
    """Extract executable command token from a tokenized shell segment."""
    idx = 0
    while idx < len(tokens):
        token = str(tokens[idx]).strip()
        if not token:
            idx += 1
            continue
        if token in _SHELL_WRAPPER_TOKENS:
            idx += 1
            continue
        if _ASSIGNMENT_RE.match(token) and "/" not in token:
            idx += 1
            continue
        return token
    return ""


def extract_shell_base_commands(command: str) -> list[str]:
    """Extract base command token from each shell segment."""
    cleaned = str(command or "").strip()
    if not cleaned:
        return []
    try:
        segments = _split_shell_segments(cleaned)
    except ValueError:
        return []
    base_commands: list[str] = []
    for segment in segments:
        base = _extract_segment_base_command(segment)
        if base:
            base_commands.append(base)
    return base_commands


def is_blocked_shell_command(command: str, blocked_patterns: list[str]) -> tuple[bool, str]:
    """Evaluate command against blocked patterns using parsed command matching."""
    cleaned = str(command or "").strip()
    if not cleaned:
        return True, "empty_command"

    try:
        segments = _split_shell_segments(cleaned)
    except ValueError:
        return True, "unparseable_command"
    if not segments:
        return True, "unparseable_command"

    segment_texts = [" ".join(tokens) for tokens in segments]
    base_commands = [
        base
        for segment in segments
        if (base := _extract_segment_base_command(segment))
    ]
    if not base_commands:
        return True, "unparseable_command"

    for raw_pattern in blocked_patterns or []:
        pattern = str(raw_pattern or "").strip()
        if not pattern:
            continue
        compiled = _compile_shell_pattern(pattern)
        segment_level_pattern = bool(re.search(r"\s", pattern))
        targets = segment_texts if segment_level_pattern else base_commands
        matcher = compiled.search if segment_level_pattern else compiled.match
        for target in targets:
            if matcher(target):
                return True, pattern
    return False, ""


class ToolResult(BaseModel):
    """Result from tool execution."""

    success: bool = True
    content: str = ""
    error: str | None = None

    @model_validator(mode="after")
    def _normalize_failure_error(self) -> "ToolResult":
        """Ensure failed results always provide an error message."""
        if not self.success and not (self.error or "").strip():
            fallback = (self.content or "").strip()
            self.error = fallback or "Tool execution failed"
        return self


class Tool(ABC):
    """Base class for all tools."""

    name: str = ""
    description: str = ""
    parameters: dict[str, Any] = Field(default_factory=dict)
    timeout_seconds: float = 30.0

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


class ToolPolicy(BaseModel):
    """Policy rule set for filtering available tools."""

    allow: list[str] | None = None
    deny: list[str] = Field(default_factory=list)
    also_allow: list[str] = Field(default_factory=list)


class ToolPolicyChain:
    """Apply policies in cascade: global -> session -> task."""

    def __init__(self, steps: list[tuple[str, ToolPolicy]] | None = None):
        self.steps: list[tuple[str, ToolPolicy]] = list(steps or [])

    @staticmethod
    def _normalize_name_set(items: list[str] | None) -> set[str] | None:
        """Normalize optional tool-name list into comparable set."""
        if items is None:
            return None
        normalized = {
            _normalize_tool_name(item)
            for item in items
            if _normalize_tool_name(item)
        }
        return normalized

    def _apply(
        self,
        current_names: set[str],
        all_names: set[str],
        policy: ToolPolicy,
    ) -> set[str]:
        """Apply one policy step against current allowed tool names."""
        next_names = set(current_names)
        allow_names = self._normalize_name_set(policy.allow)
        if allow_names is not None:
            next_names = {name for name in next_names if name in allow_names}

        deny_names = self._normalize_name_set(policy.deny) or set()
        next_names -= deny_names

        also_allow_names = self._normalize_name_set(policy.also_allow) or set()
        next_names |= also_allow_names & all_names
        return next_names

    def resolve(self, available_tools: list[Tool]) -> list[Tool]:
        """Resolve final tool list after applying policy chain."""
        if not self.steps:
            return list(available_tools)

        tools_by_name: dict[str, Tool] = {
            _normalize_tool_name(tool.name): tool
            for tool in available_tools
            if _normalize_tool_name(tool.name)
        }
        all_names = set(tools_by_name.keys())
        current_names = set(all_names)
        for _, policy in self.steps:
            current_names = self._apply(current_names, all_names, policy)

        return [
            tool
            for tool in available_tools
            if _normalize_tool_name(tool.name) in current_names
        ]


class ToolRegistry:
    """Registry for managing available tools."""

    def __init__(self, base_path: Path | str | None = None, saved_dir_name: str = "saved"):
        self._tools: dict[str, Tool] = {}
        self._tool_metadata: dict[str, dict[str, Any]] = {}
        self._saved_dir_name = (saved_dir_name or "saved").strip() or "saved"
        self._runtime_base_path = Path.cwd()
        self._global_policy: ToolPolicy | None = None
        self._session_policies: dict[str, ToolPolicy] = {}
        self._approval_callback: Callable[[str], bool] | None = None
        self.set_runtime_base_path(base_path or Path.cwd())

    @staticmethod
    def _coerce_policy(policy: ToolPolicy | dict[str, Any] | None) -> ToolPolicy | None:
        """Coerce policy payload into ToolPolicy model."""
        if policy is None:
            return None
        if isinstance(policy, ToolPolicy):
            return policy
        if isinstance(policy, dict):
            return ToolPolicy(**policy)
        raise TypeError(f"Unsupported policy type: {type(policy)!r}")

    def set_runtime_base_path(self, base_path: Path | str) -> None:
        """Set runtime base path used by tools for local file output."""
        self._runtime_base_path = Path(base_path).expanduser().resolve()

    @property
    def runtime_base_path(self) -> Path:
        """Runtime base path from which Captain Claw was launched."""
        return self._runtime_base_path

    def get_saved_base_path(self, create: bool = False) -> Path:
        """Return `<runtime_base>/saved` (or custom save dir name)."""
        saved_root = (self._runtime_base_path / self._saved_dir_name).resolve()
        try:
            saved_root.relative_to(self._runtime_base_path)
        except ValueError:
            saved_root = (self._runtime_base_path / "saved").resolve()
        if create:
            saved_root.mkdir(parents=True, exist_ok=True)
        return saved_root

    def set_approval_callback(self, callback: Callable[[str], bool] | None) -> None:
        """Set approval callback used by ask-mode execution policies."""
        self._approval_callback = callback

    def register(self, tool: Tool, metadata: dict[str, Any] | None = None) -> None:
        """Register a tool.

        Args:
            tool: Tool instance to register
        """
        if not tool.name:
            raise ValueError("Tool must have a name")

        log.debug("Registering tool", tool=tool.name)
        self._tools[tool.name] = tool
        if isinstance(metadata, dict):
            self._tool_metadata[tool.name] = dict(metadata)
        elif tool.name not in self._tool_metadata:
            self._tool_metadata[tool.name] = {}

    def unregister(self, name: str) -> None:
        """Unregister a tool.

        Args:
            name: Tool name to unregister
        """
        if name in self._tools:
            del self._tools[name]
        self._tool_metadata.pop(name, None)

    def has_tool(self, name: str) -> bool:
        """Return whether a tool name is currently registered."""
        return name in self._tools

    def get_tool_metadata(self, name: str) -> dict[str, Any]:
        """Return metadata associated with a registered tool."""
        return dict(self._tool_metadata.get(name, {}))

    def set_global_policy(self, policy: ToolPolicy | dict[str, Any] | None) -> None:
        """Set global tool policy step."""
        self._global_policy = self._coerce_policy(policy)

    def set_session_policy(self, session_id: str, policy: ToolPolicy | dict[str, Any] | None) -> None:
        """Set policy for a specific session id."""
        key = str(session_id or "").strip()
        if not key:
            return
        parsed = self._coerce_policy(policy)
        if parsed is None:
            self._session_policies.pop(key, None)
            return
        self._session_policies[key] = parsed

    def clear_session_policy(self, session_id: str) -> None:
        """Clear policy assigned to a session id."""
        key = str(session_id or "").strip()
        if not key:
            return
        self._session_policies.pop(key, None)

    def _resolve_policy_chain(
        self,
        *,
        session_id: str | None = None,
        session_policy: ToolPolicy | dict[str, Any] | None = None,
        task_policy: ToolPolicy | dict[str, Any] | None = None,
    ) -> ToolPolicyChain:
        """Build tool policy chain in global -> session -> task order."""
        steps: list[tuple[str, ToolPolicy]] = []
        if self._global_policy is not None:
            steps.append(("global", self._global_policy))

        key = str(session_id or "").strip()
        policy_from_session = self._session_policies.get(key)
        policy_from_arg = self._coerce_policy(session_policy)
        effective_session_policy = policy_from_arg or policy_from_session
        if effective_session_policy is not None:
            steps.append(("session", effective_session_policy))

        effective_task_policy = self._coerce_policy(task_policy)
        if effective_task_policy is not None:
            steps.append(("task", effective_task_policy))
        return ToolPolicyChain(steps=steps)

    def _resolve_tools(
        self,
        *,
        session_id: str | None = None,
        session_policy: ToolPolicy | dict[str, Any] | None = None,
        task_policy: ToolPolicy | dict[str, Any] | None = None,
    ) -> list[Tool]:
        """Resolve currently available tools after policy filtering."""
        chain = self._resolve_policy_chain(
            session_id=session_id,
            session_policy=session_policy,
            task_policy=task_policy,
        )
        return chain.resolve(list(self._tools.values()))

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

    def list_tools(
        self,
        *,
        session_id: str | None = None,
        session_policy: ToolPolicy | dict[str, Any] | None = None,
        task_policy: ToolPolicy | dict[str, Any] | None = None,
    ) -> list[str]:
        """List all registered tool names.

        Returns:
            List of tool names
        """
        return [
            tool.name
            for tool in self._resolve_tools(
                session_id=session_id,
                session_policy=session_policy,
                task_policy=task_policy,
            )
        ]

    def get_definitions(
        self,
        *,
        session_id: str | None = None,
        session_policy: ToolPolicy | dict[str, Any] | None = None,
        task_policy: ToolPolicy | dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Get all tool definitions for LLM.

        Returns:
            List of OpenAI function-style definitions
        """
        return [
            tool.get_definition()
            for tool in self._resolve_tools(
                session_id=session_id,
                session_policy=session_policy,
                task_policy=task_policy,
            )
        ]

    @staticmethod
    async def _cancel_task(task: asyncio.Task[Any] | None) -> None:
        """Cancel task and await it to avoid pending task warnings."""
        if task is None or task.done():
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception:
            pass

    @staticmethod
    async def _bridge_abort_event(source: asyncio.Event, target: asyncio.Event) -> None:
        """Mirror external abort event to local tool abort event."""
        await source.wait()
        target.set()

    async def execute(
        self,
        name: str,
        arguments: dict[str, Any],
        confirm: bool = False,
        session_id: str | None = None,
        session_policy: ToolPolicy | dict[str, Any] | None = None,
        task_policy: ToolPolicy | dict[str, Any] | None = None,
        abort_event: asyncio.Event | None = None,
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

        allowed_names = {
            _normalize_tool_name(item)
            for item in self.list_tools(
                session_id=session_id,
                session_policy=session_policy,
                task_policy=task_policy,
            )
        }
        if _normalize_tool_name(name) not in allowed_names:
            raise ToolBlockedError(name, "Blocked by tool policy chain")

        # Check if tool is blocked
        config = get_config()

        if name == "shell":
            command = str(arguments.get("command", "")).strip()
            policy_decision, policy_reason = self._evaluate_shell_exec_policy(command)
            if policy_decision == "deny":
                raise ToolBlockedError(name, policy_reason)
            if policy_decision == "ask":
                question = (
                    "Allow shell command execution?\n"
                    f"Command: {command}\n"
                    f"Reason: {policy_reason}"
                )
                if callable(self._approval_callback):
                    approved = bool(self._approval_callback(question))
                    if not approved:
                        raise ToolBlockedError(name, "Blocked by shell execution approval policy")
                else:
                    log.warning(
                        "Shell execution policy requires approval but no callback is configured; allowing",
                        command=command,
                    )

        # Check shell-specific blocked commands
        if name == "shell" and hasattr(config.tools.shell, "blocked"):
            blocked_cmds = config.tools.shell.blocked or []
            cmd = str(arguments.get("command", ""))
            blocked, matched = is_blocked_shell_command(cmd, blocked_cmds)
            if blocked:
                if matched == "empty_command":
                    raise ToolBlockedError(name, "Command is empty")
                if matched == "unparseable_command":
                    raise ToolBlockedError(name, "Command is not parseable")
                raise ToolBlockedError(name, f"Command matches blocked pattern: {matched}")

        # Check if confirmation required
        if confirm and name in config.tools.require_confirmation:
            # For now, just proceed - UI should handle confirmation
            pass

        # Validate arguments
        tool.validate_arguments(arguments)

        # Execute with timeout / abort propagation
        execute_task: asyncio.Task[ToolResult] | None = None
        abort_wait_task: asyncio.Task[bool] | None = None
        bridge_task: asyncio.Task[None] | None = None
        tool_abort_event = asyncio.Event()
        try:
            log.info("Executing tool", tool=name, args=arguments)
            timeout_seconds = float(getattr(tool, "timeout_seconds", 30.0) or 30.0)
            timeout_override = arguments.get("timeout")
            if timeout_override is not None:
                try:
                    timeout_seconds = float(timeout_override)
                except Exception:
                    pass
            timeout_seconds = max(1.0, timeout_seconds)

            if abort_event is not None:
                bridge_task = asyncio.create_task(
                    self._bridge_abort_event(abort_event, tool_abort_event)
                )

            execute_task = asyncio.create_task(
                tool.execute(
                    **arguments,
                    _runtime_base_path=self.runtime_base_path,
                    _saved_base_path=self.get_saved_base_path(create=False),
                    _session_id=(session_id or "").strip(),
                    _abort_event=tool_abort_event,
                )
            )
            abort_wait_task = asyncio.create_task(tool_abort_event.wait())
            done, _ = await asyncio.wait(
                {execute_task, abort_wait_task},
                timeout=timeout_seconds,
                return_when=asyncio.FIRST_COMPLETED,
            )

            if execute_task in done:
                result = await execute_task
                if not isinstance(result, ToolResult):
                    raise ToolExecutionError(name, "Tool returned invalid result payload")
                log.info("Tool executed", tool=name, success=result.success)
                return result

            if abort_wait_task in done:
                await self._cancel_task(execute_task)
                raise ToolExecutionError(name, "Execution aborted")

            tool_abort_event.set()
            await self._cancel_task(execute_task)
            timeout_label = int(timeout_seconds) if timeout_seconds.is_integer() else timeout_seconds
            raise ToolExecutionError(name, f"Execution timed out after {timeout_label}s")
        except asyncio.CancelledError:
            tool_abort_event.set()
            await self._cancel_task(execute_task)
            raise
        except ToolExecutionError:
            raise
        except Exception as e:
            log.error("Tool execution failed", tool=name, error=str(e))
            raise ToolExecutionError(name, str(e))
        finally:
            await self._cancel_task(abort_wait_task)
            await self._cancel_task(bridge_task)

    def _evaluate_shell_exec_policy(self, command: str) -> tuple[str, str]:
        """Evaluate allow/deny/ask policy for a shell command."""
        cleaned = str(command or "").strip()
        if not cleaned:
            return "deny", "Command is empty"

        cfg = get_config()
        shell_cfg = cfg.tools.shell
        deny_patterns = [str(item).strip() for item in getattr(shell_cfg, "deny_patterns", []) if str(item).strip()]
        allow_patterns = [str(item).strip() for item in getattr(shell_cfg, "allow_patterns", []) if str(item).strip()]
        default_policy = str(getattr(shell_cfg, "default_policy", "ask") or "ask").strip().lower()

        for pattern in deny_patterns:
            if self._shell_pattern_matches(cleaned, pattern):
                return "deny", f"Command matches deny pattern: {pattern}"

        for pattern in allow_patterns:
            if self._shell_pattern_matches(cleaned, pattern):
                return "allow", f"Command matches allow pattern: {pattern}"

        if default_policy == "allow":
            return "allow", "Default shell execution policy allows command"
        if default_policy == "deny":
            return "deny", "Default shell execution policy denies command"
        return "ask", "Default shell execution policy requires approval"

    @staticmethod
    def _shell_pattern_matches(command: str, pattern: str) -> bool:
        """Match shell command against glob-like policy pattern."""
        cleaned_command = str(command or "").strip()
        cleaned_pattern = str(pattern or "").strip()
        if not cleaned_command or not cleaned_pattern:
            return False

        lowered_pattern = cleaned_pattern.lower()
        targets: list[str] = [cleaned_command.lower()]
        try:
            segments = _split_shell_segments(cleaned_command)
            targets.extend(" ".join(segment).strip().lower() for segment in segments if segment)
        except Exception:
            pass
        base_commands = [item.lower() for item in extract_shell_base_commands(cleaned_command)]
        targets.extend(base_commands)
        targets.extend(Path(item).name.lower() for item in base_commands if item)

        return any(fnmatch.fnmatchcase(target, lowered_pattern) for target in targets if target)


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
