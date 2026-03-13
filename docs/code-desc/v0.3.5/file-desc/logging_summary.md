# Summary: logging.py

# logging.py Summary

**Summary:** This module provides structured logging infrastructure for Captain Claw using structlog, with support for dynamic routing to a TUI system panel or stderr fallback. It implements a sophisticated file-like sink abstraction that allows log output redirection to be configured after structlog initialization, enabling flexible integration with UI components.

**Purpose:** Solves the problem of coordinating log output between a background application and a terminal UI system panel, where the UI component may not be available at initialization time. Provides centralized, configurable logging with support for both human-readable console and machine-readable JSON output formats.

**Most Important Functions/Classes:**

1. **`configure_logging()`** - Initializes structlog with processors (contextvars, timestamps, exception handling, formatting) and renderer selection (console vs JSON) based on configuration. Sets up the logger factory with the dynamic file sink for flexible output routing.

2. **`_DynamicLogFile`** - File-like object that acts as a runtime-switchable sink. Checks the global `_system_log_sink` on every write operation, allowing log destination to change after structlog configuration. Falls back to stderr when no TUI sink is available, enabling graceful degradation.

3. **`set_system_log_sink(sink)`** - Global setter that registers a callback function to receive log lines. Allows the TUI system panel to subscribe to logs after the logging system is initialized, decoupling initialization order concerns.

4. **`_SinkWriter`** - Buffers text output and splits on newlines before forwarding complete lines to a callback sink. Handles the impedance mismatch between structlog's streaming write model and line-oriented log sinks.

5. **`get_logger(name)`** - Factory function returning configured structlog BoundLogger instances, optionally with a name parameter for module-level logger identification.

**Architecture Notes:** The design uses a two-level indirection pattern: `_DynamicLogFile` wraps `_SinkWriter`, which wraps the actual sink callback. This allows the sink to be swapped at runtime without recreating structlog's logger factory. The module-level `_system_log_sink` global variable serves as the dynamic configuration point. Structlog is configured with filtering bound loggers, contextvars support, and ISO timestamp formatting for structured, contextual logging.