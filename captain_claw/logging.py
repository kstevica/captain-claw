"""Logging configuration for Captain Claw."""

import logging
import sys
from typing import Callable
from typing import Any

import structlog

from captain_claw.config import get_config

_system_log_sink: Callable[[str], None] | None = None


class _SinkWriter:
    """File-like sink for structlog that forwards lines to a callback."""

    def __init__(self, sink: Callable[[str], None]):
        self._sink = sink
        self._buffer = ""

    def write(self, text: str) -> int:
        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if line:
                self._sink(line)
        return len(text)

    def flush(self) -> None:
        if self._buffer:
            self._sink(self._buffer)
            self._buffer = ""


def set_system_log_sink(sink: Callable[[str], None] | None) -> None:
    """Set optional sink for system logs (used by fixed UI system panel)."""
    global _system_log_sink
    _system_log_sink = sink


def configure_logging() -> None:
    """Configure structured logging for Captain Claw."""
    config = get_config()
    
    # Get log level from config
    log_level = getattr(logging, config.logging.level.upper(), logging.INFO)
    
    # Configure structlog
    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    
    # Add renderer based on config
    if config.logging.format == "console":
        processors.append(structlog.dev.ConsoleRenderer())
    else:
        processors.append(structlog.processors.JSONRenderer())
    
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(
            file=_SinkWriter(_system_log_sink) if _system_log_sink else sys.stderr
        ),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> structlog.BoundLogger:
    """Get a logger instance.
    
    Args:
        name: Optional logger name (usually __name__)
    
    Returns:
        Configured structlog logger
    """
    if name:
        return structlog.get_logger(name)
    return structlog.get_logger()


# Create module-level logger
log = get_logger(__name__)
