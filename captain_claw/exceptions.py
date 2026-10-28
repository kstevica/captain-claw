"""Custom exceptions for Captain Claw."""

import re


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


_BUSY = re.compile(r"rate.?limit|too many requests|overloaded|\b(?:429|529)\b", re.I)


def _chain(exc: BaseException):
    """*exc* and what it was raised from (a provider re-raises LiteLLM's error)."""
    seen: set[int] = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        yield exc
        exc = exc.__cause__ or exc.__context__


def is_busy(err: BaseException | str) -> bool:
    """The model's server saying "not now" — a rate limit, or overloaded —
    which waiting cures, unlike a refusal of the request itself."""
    if isinstance(err, str):
        return bool(_BUSY.search(err))
    return any(getattr(e, "status_code", None) in (429, 529) or bool(_BUSY.search(str(e)))
               for e in _chain(err))


def retry_after(err: BaseException) -> float | None:
    """The seconds the server asked to wait (its ``retry-after`` header), if
    the error carries its response."""
    for e in _chain(err):
        headers = getattr(getattr(e, "response", None), "headers", None) or getattr(
            e, "litellm_response_headers", None)
        try:
            value = float((headers or {}).get("retry-after") or 0)
        except (TypeError, ValueError, AttributeError):
            continue
        if value > 0:
            return value
    return None


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
