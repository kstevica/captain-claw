# Summary: llm_session_logger.py

# LLM Session Logger Summary

## Summary
A file-based append-only markdown logger that comprehensively records all LLM interactions within a session, capturing timestamps, model information, complete message contents, responses, token usage, and tool calls. Designed as a singleton module providing centralized logging infrastructure for debugging and auditing LLM behavior across application sessions.

## Purpose
Solves the problem of tracking and auditing LLM API calls in production systems by creating human-readable markdown logs organized by session. Enables developers and operators to review complete interaction histories, diagnose issues, analyze token consumption, and understand model behavior without relying on external logging services. Particularly valuable for debugging complex multi-turn conversations and tool-use workflows.

## Most Important Functions/Classes

1. **`LLMSessionLogger` class**
   - Core logging engine managing session-specific markdown files. Maintains state for active session slug, log file path, and call counter. Provides thread-safe append-only writes with graceful error handling that logs warnings rather than raising exceptions.

2. **`log_call()` method**
   - Primary public interface accepting interaction metadata (label, model, messages, response, instruction files, tools flag, max tokens). Increments call counter and delegates formatting, handling exceptions to prevent logging failures from crashing application logic.

3. **`_format_entry()` method**
   - Transforms raw LLM interaction data into structured markdown format. Handles polymorphic message objects (dict or object attributes), truncates oversized responses (>8000 chars) and tool arguments (>2000 chars), extracts token usage metrics, and formats tool calls with JSON arguments.

4. **`set_session()` method**
   - Initializes or switches active session context. Creates session directory structure (`logs/<session_slug>/`), resets call counter, and implements idempotent behavior for repeated calls with same slug. Validates and normalizes session slug input.

5. **`get_llm_session_logger()` function**
   - Module-level singleton factory ensuring single logger instance across application. Lazy-initializes with configurable logs directory (defaults to `./logs`), enabling centralized access from multiple code paths without dependency injection complexity.

## Architecture & Dependencies

**Dependencies:** 
- Standard library only (`json`, `datetime`, `pathlib`, `typing`)
- Internal: `captain_claw.logging.get_logger` for warning-level error reporting

**Architecture Pattern:**
- Singleton pattern with module-level state management
- Append-only file I/O (no locking, assumes single-threaded or serialized writes)
- Polymorphic message handling supporting both dict and object-based message representations
- Graceful degradation (silent failures with warnings rather than exceptions)

**Role in System:**
Acts as observability/audit layer for LLM integration, decoupled from core application logic. Sits between LLM API client and business logic, passively recording all interactions without affecting request/response flow. Output serves as source-of-truth for session replay, debugging, and compliance auditing.