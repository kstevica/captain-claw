"""Logging configuration for Captain Claw."""

import logging
import sys
from typing import Any

import structlog

from captain_claw.config import get_config


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
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
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
