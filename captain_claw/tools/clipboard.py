"""Clipboard tool — read and write the system clipboard.

Supports text, images, and generic files.  macOS-first implementation
using native ``pbcopy`` / ``pbpaste`` / ``osascript``.
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
from pathlib import Path
from typing import Any

from captain_claw.logging import get_logger
from captain_claw.tools.registry import Tool, ToolResult

log = get_logger(__name__)

# Image extensions that should be pasted as inline image data rather
# than as a file reference.
_IMAGE_EXTENSIONS = frozenset({
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".tif", ".webp",
})

_SUBPROCESS_TIMEOUT = 5  # seconds


class ClipboardTool(Tool):
    """Read or write the system clipboard."""

    name = "clipboard"
    description = (
        "Interact with the system clipboard. "
        "Use action 'read' to get the current clipboard text. "
        "Use action 'write' with 'text' to copy text, or with 'path' "
        "to copy an image / audio / video / file to the clipboard."
    )
    timeout_seconds = 15.0
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["read", "write"],
                "description": "Action to perform: 'read' to get clipboard contents, 'write' to set them.",
            },
            "text": {
                "type": "string",
                "description": "(write only) Text to copy to the clipboard.",
            },
            "path": {
                "type": "string",
                "description": (
                    "(write only) Path to a file to copy to the clipboard. "
                    "Images are pasted as inline image data. "
                    "Other files (audio, video, documents) are pasted as file references."
                ),
            },
        },
        "required": ["action"],
    }

    # ── execute ────────────────────────────────────────────────

    async def execute(  # type: ignore[override]
        self,
        action: str = "read",
        text: str | None = None,
        path: str | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        if sys.platform != "darwin":
            return ToolResult(
                success=False,
                error="Clipboard tool is currently macOS-only. Linux / Windows support coming soon.",
            )

        action = (action or "read").strip().lower()

        if action == "read":
            return await self._read_text()
        elif action == "write":
            if text:
                return await self._write_text(text)
            if path:
                return await self._write_file(path, kwargs)
            return ToolResult(
                success=False,
                error="Write action requires either 'text' or 'path' parameter.",
            )
        else:
            return ToolResult(success=False, error=f"Unknown action: {action!r}. Use 'read' or 'write'.")

    # ── read ───────────────────────────────────────────────────

    async def _read_text(self) -> ToolResult:
        """Read text from the clipboard via ``pbpaste``."""
        try:
            result = await asyncio.to_thread(
                subprocess.run,
                ["pbpaste"],
                capture_output=True,
                text=True,
                timeout=_SUBPROCESS_TIMEOUT,
            )
            content = result.stdout
            if not content.strip():
                return ToolResult(success=True, content="(clipboard is empty)")
            return ToolResult(success=True, content=content)
        except Exception as exc:
            return ToolResult(success=False, error=f"Failed to read clipboard: {exc}")

    # ── write text ─────────────────────────────────────────────

    async def _write_text(self, text: str) -> ToolResult:
        """Write text to the clipboard via ``pbcopy``."""
        try:
            await asyncio.to_thread(
                subprocess.run,
                ["pbcopy"],
                input=text,
                text=True,
                timeout=_SUBPROCESS_TIMEOUT,
                check=True,
            )
            length = len(text)
            preview = text[:120] + ("…" if length > 120 else "")
            return ToolResult(success=True, content=f"Copied {length} chars to clipboard: {preview}")
        except Exception as exc:
            return ToolResult(success=False, error=f"Failed to write to clipboard: {exc}")

    # ── write file ─────────────────────────────────────────────

    async def _write_file(self, file_path: str, kwargs: dict[str, Any]) -> ToolResult:
        """Copy a file to the clipboard.

        Images are placed as image data (so they paste inline).
        Other file types are placed as Finder file references (paste into
        Finder, Slack, email, etc.).
        """
        resolved = Path(file_path).expanduser()

        # Resolve relative paths against the workspace root.
        if not resolved.is_absolute():
            runtime_base = kwargs.get("_runtime_base_path")
            if runtime_base:
                resolved = (Path(runtime_base) / resolved).resolve()
            else:
                resolved = resolved.resolve()

        if not resolved.is_file():
            return ToolResult(success=False, error=f"File not found: {resolved}")

        suffix = resolved.suffix.lower()
        posix = str(resolved)

        if suffix in _IMAGE_EXTENSIONS:
            return await self._write_image(posix, suffix)
        else:
            return await self._write_file_ref(posix)

    async def _write_image(self, posix_path: str, suffix: str) -> ToolResult:
        """Copy an image file to the clipboard as image data."""
        # Determine the AppleScript clipboard class.
        # PNG and TIFF have dedicated classes; everything else we
        # convert to TIFF via sips first (built into macOS).
        if suffix == ".png":
            as_class = "«class PNGf»"
        elif suffix in (".tiff", ".tif"):
            as_class = "«class TIFF»"
        else:
            # JPEG, GIF, BMP, WEBP — convert to PNG in-place (temp) via sips.
            # Actually it's simpler to just use JPEG class for .jpg and
            # fall back to converting others to PNG.
            if suffix in (".jpg", ".jpeg"):
                as_class = "«class JPEG»"
            else:
                # For GIF, BMP, WEBP: read as PNG via sips conversion to
                # a temp file, then copy that.
                import tempfile
                tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                tmp.close()
                try:
                    await asyncio.to_thread(
                        subprocess.run,
                        ["sips", "-s", "format", "png", posix_path, "--out", tmp.name],
                        capture_output=True,
                        timeout=_SUBPROCESS_TIMEOUT,
                    )
                    posix_path = tmp.name
                    as_class = "«class PNGf»"
                except Exception:
                    # Fall back to file reference if conversion fails.
                    return await self._write_file_ref(posix_path)

        script = f'set the clipboard to (read (POSIX file "{posix_path}") as {as_class})'
        try:
            await asyncio.to_thread(
                subprocess.run,
                ["osascript", "-e", script],
                capture_output=True,
                timeout=_SUBPROCESS_TIMEOUT,
                check=True,
            )
            return ToolResult(
                success=True,
                content=f"Copied image to clipboard: {Path(posix_path).name}",
            )
        except Exception as exc:
            return ToolResult(success=False, error=f"Failed to copy image to clipboard: {exc}")

    async def _write_file_ref(self, posix_path: str) -> ToolResult:
        """Copy a file as a Finder file reference (paste into Finder, Slack, etc.)."""
        script = f'set the clipboard to (POSIX file "{posix_path}")'
        try:
            await asyncio.to_thread(
                subprocess.run,
                ["osascript", "-e", script],
                capture_output=True,
                timeout=_SUBPROCESS_TIMEOUT,
                check=True,
            )
            return ToolResult(
                success=True,
                content=f"Copied file to clipboard: {Path(posix_path).name}",
            )
        except Exception as exc:
            return ToolResult(success=False, error=f"Failed to copy file to clipboard: {exc}")
