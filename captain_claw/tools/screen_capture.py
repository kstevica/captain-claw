"""Screen capture tool — take screenshots and optionally analyze them.

On macOS the native ``screencapture`` CLI is used (always available, properly
handles Screen Recording permission).  On Linux / Windows ``mss`` is used
instead.
"""

from __future__ import annotations

import asyncio
import ctypes
import ctypes.util
import os
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from captain_claw.tools.registry import Tool, ToolResult
from captain_claw.tools.write import WriteTool

try:
    from structlog import get_logger

    log = get_logger(__name__)
except Exception:  # pragma: no cover
    import logging

    log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency guard (same pattern as _HAS_PILLOW in image_ocr.py)
# ---------------------------------------------------------------------------

try:
    import mss as _mss
    from mss.tools import to_png as _mss_to_png

    _HAS_MSS = True
except ImportError:
    _HAS_MSS = False

_IS_MACOS = sys.platform == "darwin"


# ---------------------------------------------------------------------------
# macOS: detect the active display (the one with the mouse cursor)
# ---------------------------------------------------------------------------


def _get_active_display_index_macos() -> int:
    """Return the 1-based display index containing the mouse cursor.

    Uses CoreGraphics via ctypes (always available on macOS, no pip deps).
    Falls back to ``1`` (main display) on any error.
    """
    try:
        cg_path = ctypes.util.find_library("CoreGraphics")
        if not cg_path:
            return 1
        cg = ctypes.cdll.LoadLibrary(cg_path)

        # --- get mouse cursor position via a synthetic event ----------------
        class CGPoint(ctypes.Structure):
            _fields_ = [("x", ctypes.c_double), ("y", ctypes.c_double)]

        cg.CGEventCreate.restype = ctypes.c_void_p
        cg.CGEventCreate.argtypes = [ctypes.c_void_p]
        cg.CGEventGetLocation.restype = CGPoint
        cg.CGEventGetLocation.argtypes = [ctypes.c_void_p]

        event = cg.CGEventCreate(None)
        if not event:
            return 1
        cursor = cg.CGEventGetLocation(event)
        # CGEventCreate returns a CFTypeRef — release it.
        _cf = ctypes.cdll.LoadLibrary(ctypes.util.find_library("CoreFoundation"))
        _cf.CFRelease.argtypes = [ctypes.c_void_p]
        _cf.CFRelease(event)

        # --- find which display contains that point -------------------------
        max_displays = 32
        DisplayArray = ctypes.c_uint32 * max_displays
        match_displays = DisplayArray()
        match_count = ctypes.c_uint32(0)

        cg.CGGetDisplaysWithPoint.argtypes = [
            CGPoint,
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint32 * max_displays),
            ctypes.POINTER(ctypes.c_uint32),
        ]
        cg.CGGetDisplaysWithPoint(
            cursor, max_displays, ctypes.byref(match_displays), ctypes.byref(match_count)
        )
        if match_count.value == 0:
            return 1
        target_id = match_displays[0]

        # --- map display-id to a 1-based index (same order screencapture -D uses)
        all_displays = DisplayArray()
        all_count = ctypes.c_uint32(0)
        cg.CGGetActiveDisplayList.argtypes = [
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint32 * max_displays),
            ctypes.POINTER(ctypes.c_uint32),
        ]
        cg.CGGetActiveDisplayList(
            max_displays, ctypes.byref(all_displays), ctypes.byref(all_count)
        )
        for i in range(all_count.value):
            if all_displays[i] == target_id:
                return i + 1  # screencapture -D is 1-based
        return 1
    except Exception:
        return 1


# ---------------------------------------------------------------------------
# macOS-native capture via ``screencapture`` CLI
# ---------------------------------------------------------------------------


def _capture_macos(monitor_index: int = 0) -> tuple[bytes, int, int]:
    """Capture a screenshot on macOS using the native ``screencapture`` CLI.

    This always captures windows correctly because ``screencapture`` is the
    system tool that macOS recognises for Screen Recording permission.  If
    permission has not been granted the OS will prompt the user automatically.

    Parameters
    ----------
    monitor_index:
        0 = active display (the one with the mouse cursor / focused window).
        1-N = specific display by index.

    Returns (png_bytes, width, height).  Runs synchronously.
    """
    # Resolve "active display" when monitor_index is 0.
    if monitor_index == 0:
        display = _get_active_display_index_macos()
    else:
        display = monitor_index

    fd, tmp_path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    try:
        # -x  suppress shutter sound
        # -D  select display (1-based).
        cmd = ["screencapture", "-x", "-D", str(display), tmp_path]

        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=15,
        )

        # If -D failed (not supported on older macOS), retry without it.
        if result.returncode != 0:
            log.warning(
                "screencapture_display_fallback",
                requested=display,
                hint="Retrying without -D flag.",
            )
            cmd = ["screencapture", "-x", tmp_path]
            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=15,
            )

        if result.returncode != 0:
            stderr = result.stderr.decode(errors="replace").strip()
            raise RuntimeError(
                f"screencapture failed (exit {result.returncode}): {stderr}"
            )

        png_bytes = Path(tmp_path).read_bytes()
        if not png_bytes:
            raise RuntimeError("screencapture produced an empty file")

        # Read dimensions from PNG header (IHDR chunk).
        w, h = _png_dimensions(png_bytes)
        log.info(
            "macos_screenshot_captured",
            display=display,
            resolution=f"{w}x{h}",
        )
        return png_bytes, w, h
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def _png_dimensions(data: bytes) -> tuple[int, int]:
    """Extract width and height from the IHDR chunk of PNG data."""
    import struct

    # PNG signature (8 bytes) + IHDR length (4 bytes) + "IHDR" (4 bytes)
    # then 4-byte width + 4-byte height.
    if len(data) < 24 or data[:8] != b"\x89PNG\r\n\x1a\n":
        return 0, 0
    w, h = struct.unpack(">II", data[16:24])
    return w, h


# ---------------------------------------------------------------------------
# mss-based capture (Linux / Windows fallback)
# ---------------------------------------------------------------------------


def _capture_mss(monitor_index: int = 0) -> tuple[bytes, int, int]:
    """Capture a screenshot using *mss* (Linux / Windows).

    Parameters
    ----------
    monitor_index:
        0 = primary display (default).  1 = primary, 2 = secondary, …

    Returns (png_bytes, width, height).  Runs synchronously.
    """
    if not _HAS_MSS:
        raise RuntimeError(
            "Screen capture requires the 'mss' package. "
            "Install with: pip install captain-claw[screen]"
        )

    with _mss.mss() as sct:
        monitors = sct.monitors
        # mss index 0 = all monitors combined.  We default to the primary
        # display (index 1) so the behaviour matches macOS.
        idx = max(1, monitor_index) if len(monitors) > 1 else 0
        idx = min(idx, len(monitors) - 1)
        monitor = monitors[idx]
        shot = sct.grab(monitor)
        png_bytes: bytes = _mss_to_png(shot.rgb, shot.size)
        return png_bytes, shot.width, shot.height


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------


def _capture_screenshot_bytes(monitor_index: int = 0) -> tuple[bytes, int, int]:
    """Capture a screenshot of a single display.

    On macOS: uses native ``screencapture`` and auto-detects the active
    display (the one with the mouse cursor).
    On Linux / Windows: uses ``mss`` targeting the primary display.

    Parameters
    ----------
    monitor_index:
        0 = active / primary display (default).
        1 = primary, 2 = secondary, …

    Returns
    -------
    tuple of (png_bytes, width, height).

    Runs synchronously — callers should wrap with ``asyncio.to_thread``.
    """
    if _IS_MACOS:
        return _capture_macos(monitor_index)
    return _capture_mss(monitor_index)


async def capture_and_save(
    session_id: str,
    saved_root: Path,
    monitor_index: int = 0,
    label: str = "screenshot",
) -> Path:
    """Capture a screenshot and save it to the workspace.

    Returns the saved file path (absolute).
    """
    png_bytes, w, h = await asyncio.to_thread(
        _capture_screenshot_bytes, monitor_index
    )

    sid = WriteTool._normalize_session_id(session_id)
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    filename = f"{label}-{stamp}.png"

    dest = WriteTool._normalize_under_saved(
        f"media/{sid}/{filename}", saved_root, sid
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    await asyncio.to_thread(dest.write_bytes, png_bytes)

    size_kb = len(png_bytes) / 1024.0
    log.info(
        "screenshot_saved",
        path=str(dest),
        size_kb=f"{size_kb:.0f}",
        resolution=f"{w}x{h}",
    )
    return dest


# ---------------------------------------------------------------------------
# Agent-invocable tool
# ---------------------------------------------------------------------------


class ScreenCaptureTool(Tool):
    """Capture a screenshot of the user's screen."""

    name: str = "screen_capture"
    timeout_seconds: float = 30.0
    description: str = (
        "Capture a screenshot of the user's screen (or a specific monitor) "
        "and save it to workspace. Returns the file path for use with "
        "image_vision or image_ocr tools. If a prompt is provided, the "
        "screenshot is automatically analyzed with the vision model."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "monitor": {
                "type": "number",
                "description": (
                    "Monitor index: 0 = main/primary display (default), "
                    "1 = primary, 2 = secondary, etc. "
                    "On macOS 0 maps to the main display."
                ),
            },
            "prompt": {
                "type": "string",
                "description": (
                    "Optional: if provided, automatically analyze the "
                    "screenshot with the vision model using this prompt. "
                    "If omitted, just capture and return the file path."
                ),
            },
        },
        "required": [],
    }

    async def execute(
        self,
        monitor: int = 0,
        prompt: str = "",
        **kwargs: Any,
    ) -> ToolResult:
        # macOS uses the native screencapture CLI — no extra deps needed.
        # On other platforms, mss is required.
        if not _IS_MACOS and not _HAS_MSS:
            return ToolResult(
                success=False,
                error=(
                    "Screen capture requires the 'mss' package. "
                    "Install with: pip install captain-claw[screen]"
                ),
            )

        session_id = str(kwargs.get("_session_id", ""))
        saved_root = WriteTool._resolve_saved_root(kwargs)

        try:
            path = await capture_and_save(
                session_id=session_id,
                saved_root=saved_root,
                monitor_index=int(monitor or 0),
            )
        except Exception as exc:
            return ToolResult(
                success=False,
                error=f"Screenshot capture failed: {exc}",
            )

        # If a prompt was provided, chain into vision analysis.
        effective_prompt = str(prompt or "").strip()
        if effective_prompt:
            try:
                from captain_claw.tools.image_ocr import ImageVisionTool

                vision_tool = ImageVisionTool()
                return await vision_tool.execute(
                    path=str(path),
                    prompt=effective_prompt,
                    **kwargs,
                )
            except Exception as exc:
                # Vision analysis failed — still return the screenshot path.
                return ToolResult(
                    success=True,
                    content=(
                        f"Screenshot captured: {path}\n"
                        f"Vision analysis failed: {exc}\n"
                        f"You can retry with image_vision on the path above."
                    ),
                )

        return ToolResult(
            success=True,
            content=(
                f"Screenshot captured and saved.\n"
                f"Path: {path}\n"
                f"Use image_vision or image_ocr tool on this path to analyze it."
            ),
        )
