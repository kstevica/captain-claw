# Summary: screen_capture.py

# screen_capture.py Summary

**Summary:**
A cross-platform screenshot capture tool that abstracts platform-specific implementation details (native macOS `screencapture` CLI vs. `mss` library for Linux/Windows) into a unified interface. Provides both low-level capture functions and a high-level agent-invocable tool that optionally chains into vision analysis for automatic screenshot interpretation.

**Purpose:**
Solves the problem of reliably capturing screenshots across macOS, Linux, and Windows with proper permission handling, display selection, and integration into an AI agent workflow. Handles the complexity of macOS Screen Recording permissions (which require the native `screencapture` tool) while providing fallback mechanisms for other platforms.

**Most Important Functions/Classes:**

1. **`_get_active_display_index_macos()`** — Uses CoreGraphics via ctypes to detect which display contains the mouse cursor, enabling intelligent "active display" capture on macOS without external dependencies. Falls back gracefully to main display on any error.

2. **`_capture_macos(monitor_index: int)`** — Invokes the native macOS `screencapture` CLI with proper permission handling and fallback logic. Returns PNG bytes with dimensions extracted from IHDR chunk. Handles display selection via `-D` flag with automatic retry if unsupported on older macOS versions.

3. **`_capture_mss(monitor_index: int)`** — Fallback implementation using the `mss` library for Linux/Windows. Manages monitor indexing (accounting for mss's 0-based "all monitors" index) and converts captured RGB data to PNG format.

4. **`_capture_screenshot_bytes(monitor_index: int)`** — Unified entry point that dispatches to platform-specific implementation (macOS vs. mss-based). Runs synchronously; callers wrap with `asyncio.to_thread()` for async contexts.

5. **`ScreenCaptureTool` (class)** — Agent-invocable tool implementing the `Tool` interface. Orchestrates screenshot capture, file persistence to workspace, and optional chaining into `ImageVisionTool` for automatic analysis when a prompt is provided. Handles session management and error recovery (vision failures don't prevent returning the screenshot path).

**Architecture & Dependencies:**

- **Platform abstraction:** Conditional imports and platform detection (`sys.platform == "darwin"`) enable graceful degradation across OS targets.
- **Optional dependency guard:** `mss` is optional; absence triggers helpful error messages with installation instructions.
- **Async/sync boundary:** Synchronous capture functions wrapped with `asyncio.to_thread()` for non-blocking execution in async contexts.
- **File management:** Integrates with `WriteTool` for session-aware path normalization and workspace organization (`media/{session_id}/{timestamp}.png`).
- **Tool chaining:** Supports optional downstream analysis via `ImageVisionTool` when a vision prompt is provided, with graceful fallback if vision fails.
- **Logging:** Uses structlog with fallback to stdlib logging for observability across capture operations.

**Role in System:**
Acts as a sensory input mechanism for AI agents, enabling screenshot-based task automation and visual understanding workflows. Bridges the gap between user-facing screen state and agent reasoning capabilities.