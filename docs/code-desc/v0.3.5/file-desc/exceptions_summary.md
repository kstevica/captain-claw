# Summary: exceptions.py

# exceptions.py Summary

**Summary:** Defines a comprehensive exception hierarchy for Captain Claw, an LLM-based agent framework. Implements 14 custom exception classes organized into logical domains (configuration, LLM operations, tool execution, sessions, context management, validation, and safety guards) with structured error information and metadata preservation.

**Purpose:** Provides granular error handling across the Captain Claw system, enabling precise exception catching and recovery strategies. Allows different subsystems (LLM API interactions, tool registry, session management, context windows, and safety policies) to signal specific failure modes with contextual data, improving debugging and operational visibility.

**Most Important Classes:**

1. **CaptainClawError** - Base exception class for all Captain Claw errors; serves as the root of the exception hierarchy for catch-all error handling and framework-wide exception identification.

2. **LLMAPIError** - Handles LLM API failures (rate limiting, authentication, connectivity); stores HTTP status codes for granular API error differentiation and retry logic implementation.

3. **ToolExecutionError** - Captures tool execution failures with tool name and error message; enables tracking which specific tool failed and why for debugging and audit trails.

4. **ToolBlockedError** - Represents policy-enforced tool execution blocks; preserves both tool name and blocking reason for security policy compliance logging and user feedback.

5. **ContextOverflowError** - Signals context window exhaustion with current/max token counts; provides metrics for context management decisions and token budget tracking in long-running conversations.

**Architecture Notes:**
- Three-tier hierarchy: base exception → domain-specific exceptions (LLMError, ToolError, SessionError, ContextError) → specific error types
- Metadata preservation pattern: custom `__init__` methods store structured data (status_code, tool_name, session_id, token counts) as instance attributes for programmatic error handling
- Supports both string messages and structured error information for flexible error reporting and recovery strategies
- No external dependencies; pure Python exception definitions suitable for library distribution