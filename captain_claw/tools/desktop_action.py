"""Desktop automation tool — cross-platform mouse, keyboard, and app control.

Uses ``pyautogui`` for mouse/keyboard interaction (macOS, Linux, Windows).
Platform-specific launchers handle opening apps, folders, and URLs.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import sys
from typing import Any

from captain_claw.logging import get_logger
from captain_claw.tools.registry import Tool, ToolResult

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency guard
# ---------------------------------------------------------------------------

try:
    import pyautogui

    pyautogui.FAILSAFE = True  # move mouse to top-left corner to abort
    pyautogui.PAUSE = 0.1  # small delay between actions
    _HAS_PYAUTOGUI = True
except (ImportError, AttributeError, OSError):
    # AttributeError: rubicon/objc fails on arm64→x64 Rosetta (missing objc_msgSendSuper_stret)
    # OSError: ctypes library loading failures on some platforms
    _HAS_PYAUTOGUI = False

_IS_MACOS = sys.platform == "darwin"
_IS_LINUX = sys.platform.startswith("linux")
_IS_WINDOWS = sys.platform == "win32"


# ---------------------------------------------------------------------------
# Platform launcher
# ---------------------------------------------------------------------------


def _open_target(target: str) -> str:
    """Open an app, folder, or URL using the platform's native launcher.

    Returns a status message.
    """
    target = target.strip()
    if not target:
        raise ValueError("No target specified for 'open' action.")

    if _IS_MACOS:
        # Detect app name vs path/URL.
        # If it looks like an app name (no slashes, no dots except .app),
        # use `open -a`.  Otherwise use plain `open`.
        if (
            not target.startswith(("/", "~", "http://", "https://", "file://"))
            and "/" not in target
            and not os.path.exists(target)
        ):
            cmd = ["open", "-a", target]
        else:
            cmd = ["open", target]
    elif _IS_LINUX:
        cmd = ["xdg-open", target]
    elif _IS_WINDOWS:
        # os.startfile handles URLs, folders, and files on Windows.
        os.startfile(target)  # type: ignore[attr-defined]
        return f"Opened: {target}"
    else:
        raise RuntimeError(f"Unsupported platform: {sys.platform}")

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=10,
    )
    if result.returncode != 0:
        stderr = result.stderr.strip()
        raise RuntimeError(f"Failed to open '{target}': {stderr}")
    return f"Opened: {target}"


# ---------------------------------------------------------------------------
# Tool class
# ---------------------------------------------------------------------------


class DesktopActionTool(Tool):
    """Cross-platform desktop GUI automation."""

    name = "desktop_action"
    description = (
        "Interact with the desktop: click, type, scroll, press keys, "
        "open apps/folders/URLs. Use with screen_capture to see the "
        "screen first, then act on coordinates."
    )
    timeout_seconds = 30.0
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "click",
                    "double_click",
                    "right_click",
                    "move",
                    "type",
                    "press",
                    "hotkey",
                    "scroll",
                    "drag",
                    "open",
                    "mouse_position",
                    "screenshot_click",
                ],
                "description": "Action to perform on the desktop.",
            },
            "x": {
                "type": "number",
                "description": "X coordinate (pixels from left edge of screen).",
            },
            "y": {
                "type": "number",
                "description": "Y coordinate (pixels from top edge of screen).",
            },
            "text": {
                "type": "string",
                "description": (
                    "For 'type': text to type. "
                    "For 'press': key name (enter, tab, escape, f5, space, etc.). "
                    "For 'screenshot_click': description of the UI element to find and click."
                ),
            },
            "keys": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "For 'hotkey': list of keys to press together "
                    "(e.g. ['command', 'c'] on macOS, ['ctrl', 'c'] on Linux/Windows)."
                ),
            },
            "target": {
                "type": "string",
                "description": (
                    "For 'open': application name (e.g. 'Safari', 'Firefox'), "
                    "folder path, or URL to open."
                ),
            },
            "dx": {
                "type": "number",
                "description": "For 'drag': destination X. For 'scroll': horizontal scroll amount.",
            },
            "dy": {
                "type": "number",
                "description": (
                    "For 'drag': destination Y. "
                    "For 'scroll': vertical scroll clicks (positive=up, negative=down)."
                ),
            },
        },
        "required": ["action"],
    }

    # ── execute ────────────────────────────────────────────────

    async def execute(
        self,
        action: str = "mouse_position",
        x: float | None = None,
        y: float | None = None,
        text: str | None = None,
        keys: list[str] | None = None,
        target: str | None = None,
        dx: float | None = None,
        dy: float | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        # The 'open' action doesn't require pyautogui.
        if action != "open" and not _HAS_PYAUTOGUI:
            return ToolResult(
                success=False,
                error=(
                    "Desktop automation requires the 'pyautogui' package. "
                    "Install with: pip install captain-claw[screen]"
                ),
            )

        action = (action or "mouse_position").strip().lower()

        try:
            if action == "click":
                return await self._click(x, y)
            elif action == "double_click":
                return await self._double_click(x, y)
            elif action == "right_click":
                return await self._right_click(x, y)
            elif action == "move":
                return await self._move(x, y)
            elif action == "type":
                return await self._type_text(text)
            elif action == "press":
                return await self._press_key(text)
            elif action == "hotkey":
                return await self._hotkey(keys)
            elif action == "scroll":
                return await self._scroll(dy, x, y)
            elif action == "drag":
                return await self._drag(x, y, dx, dy)
            elif action == "open":
                return await self._open(target)
            elif action == "mouse_position":
                return await self._mouse_position()
            elif action == "screenshot_click":
                return await self._screenshot_click(text, **kwargs)
            else:
                return ToolResult(
                    success=False,
                    error=f"Unknown action: {action!r}",
                )
        except Exception as exc:
            log.error("desktop_action_failed", action=action, error=str(exc))
            return ToolResult(success=False, error=f"Desktop action '{action}' failed: {exc}")

    # ── action implementations ────────────────────────────────

    async def _click(self, x: float | None, y: float | None) -> ToolResult:
        if x is None or y is None:
            return ToolResult(success=False, error="click requires x and y coordinates.")
        await asyncio.to_thread(pyautogui.click, int(x), int(y))
        pos = pyautogui.position()
        return ToolResult(
            success=True,
            content=f"Clicked at ({int(x)}, {int(y)}). Mouse now at ({pos.x}, {pos.y}).",
        )

    async def _double_click(self, x: float | None, y: float | None) -> ToolResult:
        if x is None or y is None:
            return ToolResult(success=False, error="double_click requires x and y coordinates.")
        await asyncio.to_thread(pyautogui.doubleClick, int(x), int(y))
        pos = pyautogui.position()
        return ToolResult(
            success=True,
            content=f"Double-clicked at ({int(x)}, {int(y)}). Mouse now at ({pos.x}, {pos.y}).",
        )

    async def _right_click(self, x: float | None, y: float | None) -> ToolResult:
        if x is None or y is None:
            return ToolResult(success=False, error="right_click requires x and y coordinates.")
        await asyncio.to_thread(pyautogui.rightClick, int(x), int(y))
        pos = pyautogui.position()
        return ToolResult(
            success=True,
            content=f"Right-clicked at ({int(x)}, {int(y)}). Mouse now at ({pos.x}, {pos.y}).",
        )

    async def _move(self, x: float | None, y: float | None) -> ToolResult:
        if x is None or y is None:
            return ToolResult(success=False, error="move requires x and y coordinates.")
        await asyncio.to_thread(pyautogui.moveTo, int(x), int(y))
        return ToolResult(
            success=True,
            content=f"Mouse moved to ({int(x)}, {int(y)}).",
        )

    async def _type_text(self, text: str | None) -> ToolResult:
        if not text:
            return ToolResult(success=False, error="type requires 'text' parameter.")
        await asyncio.to_thread(pyautogui.typewrite, text, interval=0.02)
        return ToolResult(
            success=True,
            content=f"Typed {len(text)} characters.",
        )

    async def _press_key(self, text: str | None) -> ToolResult:
        if not text:
            return ToolResult(success=False, error="press requires 'text' parameter (key name).")
        key = text.strip().lower()
        await asyncio.to_thread(pyautogui.press, key)
        return ToolResult(
            success=True,
            content=f"Pressed key: {key}",
        )

    async def _hotkey(self, keys: list[str] | None) -> ToolResult:
        if not keys:
            return ToolResult(success=False, error="hotkey requires 'keys' parameter (list of keys).")
        normalized = [k.strip().lower() for k in keys]
        await asyncio.to_thread(pyautogui.hotkey, *normalized)
        return ToolResult(
            success=True,
            content=f"Pressed hotkey: {' + '.join(normalized)}",
        )

    async def _scroll(
        self, dy: float | None, x: float | None, y: float | None
    ) -> ToolResult:
        if dy is None:
            return ToolResult(success=False, error="scroll requires 'dy' parameter (scroll clicks).")
        ix = int(x) if x is not None else None
        iy = int(y) if y is not None else None
        if ix is not None and iy is not None:
            await asyncio.to_thread(pyautogui.scroll, int(dy), x=ix, y=iy)
            return ToolResult(
                success=True,
                content=f"Scrolled {int(dy)} clicks at ({ix}, {iy}).",
            )
        await asyncio.to_thread(pyautogui.scroll, int(dy))
        return ToolResult(
            success=True,
            content=f"Scrolled {int(dy)} clicks at current position.",
        )

    async def _drag(
        self,
        x: float | None,
        y: float | None,
        dx: float | None,
        dy: float | None,
    ) -> ToolResult:
        if x is None or y is None or dx is None or dy is None:
            return ToolResult(
                success=False,
                error="drag requires x, y (start) and dx, dy (end) coordinates.",
            )
        # Move to start, then drag to end.
        await asyncio.to_thread(pyautogui.moveTo, int(x), int(y))
        await asyncio.to_thread(
            pyautogui.drag,
            int(dx) - int(x),
            int(dy) - int(y),
            duration=0.5,
        )
        pos = pyautogui.position()
        return ToolResult(
            success=True,
            content=f"Dragged from ({int(x)}, {int(y)}) to ({int(dx)}, {int(dy)}). Mouse now at ({pos.x}, {pos.y}).",
        )

    async def _open(self, target: str | None) -> ToolResult:
        if not target:
            return ToolResult(
                success=False,
                error="open requires 'target' parameter (app name, folder path, or URL).",
            )
        msg = await asyncio.to_thread(_open_target, target)
        return ToolResult(success=True, content=msg)

    async def _mouse_position(self) -> ToolResult:
        if not _HAS_PYAUTOGUI:
            return ToolResult(success=False, error="pyautogui not installed.")
        pos = pyautogui.position()
        size = pyautogui.size()
        return ToolResult(
            success=True,
            content=f"Mouse position: ({pos.x}, {pos.y}). Screen size: {size.width}x{size.height}.",
        )

    # ── screenshot_click composite ────────────────────────────

    async def _screenshot_click(self, text: str | None, **kwargs: Any) -> ToolResult:
        """Take a screenshot, ask vision to find an element, click it."""
        if not text:
            return ToolResult(
                success=False,
                error="screenshot_click requires 'text' describing the UI element to find.",
            )

        # Step 1: capture screenshot
        from captain_claw.tools.screen_capture import ScreenCaptureTool

        capture_tool = ScreenCaptureTool()
        capture_result = await capture_tool.execute(monitor=0, **kwargs)
        if not capture_result.success:
            return ToolResult(
                success=False,
                error=f"Screenshot failed: {capture_result.error}",
            )

        # Extract the screenshot path from the capture result.
        path_match = re.search(r"Path:\s*(\S+)", capture_result.content or "")
        if not path_match:
            return ToolResult(
                success=False,
                error="Could not determine screenshot file path.",
            )
        screenshot_path = path_match.group(1)

        # Step 2: ask vision to locate the element
        from captain_claw.tools.image_ocr import ImageVisionTool

        vision_tool = ImageVisionTool()
        vision_prompt = (
            f"Find the UI element described as: \"{text}\"\n"
            f"Return ONLY a JSON object with the pixel coordinates of the CENTER "
            f"of that element: {{\"x\": <number>, \"y\": <number>}}\n"
            f"If the element is not found, return: {{\"error\": \"not found\"}}"
        )
        vision_result = await vision_tool.execute(
            path=screenshot_path,
            prompt=vision_prompt,
            **kwargs,
        )
        if not vision_result.success:
            return ToolResult(
                success=False,
                error=f"Vision analysis failed: {vision_result.error}",
            )

        # Step 3: parse coordinates from vision response
        response_text = vision_result.content or ""
        # Try to extract JSON from the response.
        json_match = re.search(r"\{[^}]+\}", response_text)
        if not json_match:
            return ToolResult(
                success=False,
                error=f"Could not parse coordinates from vision response: {response_text[:200]}",
            )

        try:
            coords = json.loads(json_match.group(0))
        except json.JSONDecodeError:
            return ToolResult(
                success=False,
                error=f"Invalid JSON in vision response: {json_match.group(0)}",
            )

        if "error" in coords:
            return ToolResult(
                success=False,
                error=f"Element not found: {coords['error']}",
            )

        cx = coords.get("x")
        cy = coords.get("y")
        if cx is None or cy is None:
            return ToolResult(
                success=False,
                error=f"Vision response missing x/y coordinates: {coords}",
            )

        # Step 4: click at the coordinates
        await asyncio.to_thread(pyautogui.click, int(cx), int(cy))
        pos = pyautogui.position()

        return ToolResult(
            success=True,
            content=(
                f"Found \"{text}\" at ({int(cx)}, {int(cy)}) and clicked. "
                f"Mouse now at ({pos.x}, {pos.y})."
            ),
        )
