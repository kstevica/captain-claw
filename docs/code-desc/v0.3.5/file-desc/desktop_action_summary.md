# Summary: desktop_action.py

# Desktop Action Tool Summary

## Summary
A cross-platform desktop GUI automation tool that provides mouse, keyboard, and application control through a unified interface. Leverages `pyautogui` for input automation on macOS, Linux, and Windows, with platform-specific launchers for opening applications, folders, and URLs. Includes a sophisticated composite action (`screenshot_click`) that integrates vision analysis to locate and interact with UI elements.

## Purpose
Solves the problem of programmatic desktop interaction for AI agents and automation workflows. Enables remote control of GUI applications without native bindings, supports coordinate-based clicking, keyboard input, and application launching. The `screenshot_click` action bridges vision and action by allowing agents to describe UI elements in natural language, have them located via image analysis, and automatically interact with them.

## Most Important Functions/Classes

1. **`DesktopActionTool.execute()`** — Main async dispatcher that routes 12+ different desktop actions (click, type, hotkey, scroll, drag, open, etc.) to their respective handlers. Validates dependencies, normalizes parameters, and wraps all operations in error handling with structured logging.

2. **`_screenshot_click()`** — Composite action orchestrating a three-step workflow: (1) capture screenshot via `ScreenCaptureTool`, (2) use `ImageVisionTool` to locate UI element from natural language description, (3) parse JSON coordinates from vision response and execute click. Enables high-level automation without manual coordinate specification.

3. **`_open_target()`** — Platform-aware application/URL launcher. Detects macOS app names vs paths and uses appropriate `open` command variants; falls back to `xdg-open` on Linux and `os.startfile()` on Windows. Handles subprocess execution with timeout and error reporting.

4. **Mouse/Keyboard Action Methods** (`_click()`, `_type_text()`, `_hotkey()`, `_scroll()`, `_drag()`) — Individual async wrappers around `pyautogui` functions that run blocking operations in thread pool via `asyncio.to_thread()`. Each validates required parameters, executes the action, and returns structured feedback with current state.

5. **`_mouse_position()`** — Utility that reports current mouse coordinates and screen dimensions, useful for debugging and coordinate discovery during automation workflows.

## Architecture & Dependencies

**Core Dependencies:**
- `pyautogui` — Optional but required for all mouse/keyboard actions (graceful degradation if missing)
- `captain_claw.tools.registry.Tool` — Base class providing tool interface contract
- `captain_claw.tools.screen_capture.ScreenCaptureTool` — Dependency for screenshot capture in composite action
- `captain_claw.tools.image_ocr.ImageVisionTool` — Dependency for vision-based element location

**Platform Detection:**
- Runtime checks for macOS (`sys.platform == "darwin"`), Linux, and Windows
- Platform-specific command construction for application launching

**Async Design:**
- All blocking operations (GUI automation, subprocess calls) executed via `asyncio.to_thread()` to prevent event loop blocking
- Timeout enforcement at tool level (30 seconds default)
- Failsafe enabled on `pyautogui` (mouse to corner aborts automation)

**Integration Points:**
- Registers as a `Tool` in captain_claw framework with JSON schema parameter definition
- Composable with other tools (`ScreenCaptureTool`, `ImageVisionTool`) for intelligent automation
- Structured logging via `captain_claw.logging` for debugging and monitoring

**Parameter Schema:**
12 action types with conditional required parameters (e.g., `click` requires x/y; `type` requires text; `open` requires target). Flexible coordinate handling (float inputs converted to int for pyautogui compatibility).