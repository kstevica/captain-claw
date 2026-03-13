# Summary: runtime_context.py

# runtime_context.py Summary

## Summary
This module defines the shared mutable state container for an interactive multi-platform agent runtime session. It centralizes all session-level state that was previously scattered across closure variables, enabling clean separation of concerns across extracted modules while maintaining a single source of truth for runtime configuration and execution state.

## Purpose
Solves the architectural problem of managing complex, interconnected runtime state across multiple independent modules handling agent execution, platform integrations (Telegram, Slack, Discord), command queuing, cron scheduling, and terminal UI interactions. Replaces monolithic closure-captured variables with a structured, injectable context object that can be passed to any module requiring access to shared state.

## Most Important Classes/Functions

1. **RuntimeContext (dataclass)**
   - Central state container holding references to the Agent, TerminalUI, command/followup queues, and all platform-specific state. Manages execution metadata (last execution time, completed timestamp, next steps) and cron job tracking. Provides accessor methods for platform-agnostic state retrieval.

2. **PlatformState (dataclass)**
   - Per-platform bridge state encapsulation storing platform name, active bridge instance, polling task, enabled status, user approval/pairing dictionaries, state persistence keys, message offsets, and platform-specific configuration objects. Enables uniform handling of heterogeneous platform implementations (Telegram, Slack, Discord).

3. **get_platform_state(platform: str) -> PlatformState**
   - Accessor method providing platform-agnostic retrieval of platform-specific state by string identifier. Raises ValueError for unknown platforms, ensuring type safety and explicit error handling.

4. **platform_names() -> list[str]**
   - Utility method returning canonical list of supported platform identifiers, enabling iteration and validation across all integrated platforms.

5. **on_cron_output callback field**
   - Async callback hook (Callable[[str, str], Awaitable[None]]) invoked when cron jobs produce assistant output, enabling decoupled event notification to UI or external handlers without tight coupling between cron scheduler and output consumers.