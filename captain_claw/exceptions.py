"""Custom exceptions for Captain Claw."""


class CaptainClawError(Exception):
    """Base exception for Captain Claw."""

    pass


class ConfigurationError(CaptainClawError):
    """Configuration-related errors."""

    pass


class LLMError(CaptainClawError):
    """LLM-related errors."""

    pass


class LLMAPIError(LLMError):
    """LLM API errors (rate limit, auth, etc.)."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class ToolError(CaptainClawError):
    """Tool execution errors."""

    pass


class ToolExecutionError(ToolError):
    """Tool execution failed."""

    def __init__(self, tool_name: str, message: str):
        super().__init__(f"Tool '{tool_name}' failed: {message}")
        self.tool_name = tool_name


class ToolNotFoundError(ToolError):
    """Tool not found in registry."""

    def __init__(self, tool_name: str):
        super().__init__(f"Tool not found: {tool_name}")
        self.tool_name = tool_name


class ToolBlockedError(ToolError):
    """Tool execution blocked by policy."""

    def __init__(self, tool_name: str, reason: str):
        super().__init__(f"Tool '{tool_name}' blocked: {reason}")
        self.tool_name = tool_name
        self.reason = reason


class SessionError(CaptainClawError):
    """Session-related errors."""

    pass


class SessionNotFoundError(SessionError):
    """Session not found."""

    def __init__(self, session_id: str):
        super().__init__(f"Session not found: {session_id}")
        self.session_id = session_id


class ContextError(CaptainClawError):
    """Context window errors."""

    pass


class ContextOverflowError(ContextError):
    """Context window overflow."""

    def __init__(self, current_tokens: int, max_tokens: int):
        super().__init__(
            f"Context overflow: {current_tokens} > {max_tokens} tokens"
        )
        self.current_tokens = current_tokens
        self.max_tokens = max_tokens


class ValidationError(CaptainClawError):
    """Validation errors."""

    pass


class GuardBlockedError(CaptainClawError):
    """Guard blocked a request/response/tool action."""

    def __init__(self, guard_type: str, message: str):
        super().__init__(message)
        self.guard_type = guard_type
