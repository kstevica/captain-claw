"""Shared runtime for the Google Workspace (gws) tool mixins.

Hosts the subprocess runner, binary resolution, constants, and shared
helpers used by :mod:`_gws_drive`, :mod:`_gws_docs`, and
:mod:`_gws_calendar`.  The concrete ``GwsTool`` composes these mixins.
"""

from __future__ import annotations

import asyncio
import re
import shutil
from pathlib import Path
from typing import Any

from captain_claw.config import get_config
from captain_claw.logging import get_logger
from captain_claw.tools.registry import ToolResult

log = get_logger(__name__)

# Maximum output length returned to the agent.
_MAX_OUTPUT_CHARS = 60_000

# Default timeout for gws commands (seconds).
_DEFAULT_TIMEOUT = 120

# Pattern matching inline base64-encoded images (can be hundreds of KB in exported markdown).
_BASE64_IMG_RE = re.compile(r"data:image/[^;]+;base64,[A-Za-z0-9+/=\s]+")


def _strip_base64_images(text: str) -> str:
    """Remove inline base64 image data from text to prevent context bloat."""
    cleaned = _BASE64_IMG_RE.sub("[image]", text)
    if len(cleaned) < len(text):
        log.debug(
            "stripped base64 images",
            original_len=len(text),
            cleaned_len=len(cleaned),
        )
    return cleaned


class GwsRuntimeMixin:
    """Base mixin: binary resolution + subprocess runner for gws commands."""

    def __init__(self) -> None:
        self._binary: str | None = None
        self._stream_callback: Any = None

    # ------------------------------------------------------------------
    # Binary resolution
    # ------------------------------------------------------------------

    def _resolve_binary(self) -> str | None:
        """Find the gws binary (config override → PATH)."""
        if self._binary and shutil.which(self._binary):
            return self._binary

        try:
            cfg = get_config()
            custom = getattr(cfg.tools, "gws", None)
            if custom and hasattr(custom, "binary_path") and custom.binary_path:
                p = Path(custom.binary_path).expanduser()
                if p.exists():
                    self._binary = str(p)
                    return self._binary
        except Exception:
            pass

        found = shutil.which("gws")
        if found:
            self._binary = found
            return self._binary

        return None

    # ------------------------------------------------------------------
    # Core runner
    # ------------------------------------------------------------------

    async def _run_gws(
        self,
        binary: str,
        args: list[str],
        timeout: float = _DEFAULT_TIMEOUT,
        json_output: bool = True,
    ) -> ToolResult:
        """Run a gws command, stream stdout/stderr, return captured output."""
        cmd = [binary] + args
        if json_output and "--format" not in args:
            cmd.extend(["--format", "json"])

        log.debug("Running gws command", cmd=" ".join(cmd))
        stream_cb = getattr(self, "_stream_callback", None)

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout_chunks: list[str] = []
        stderr_chunks: list[str] = []

        async def _read_stream(
            stream: asyncio.StreamReader, collected: list[str], prefix: str = "",
        ) -> None:
            while True:
                line = await stream.readline()
                if not line:
                    break
                text = line.decode("utf-8", errors="replace")
                collected.append(text)
                if stream_cb:
                    try:
                        stream_cb(prefix + text)
                    except Exception:
                        pass

        async def _collect() -> None:
            await asyncio.gather(
                _read_stream(proc.stdout, stdout_chunks),
                _read_stream(proc.stderr, stderr_chunks),
            )
            await proc.wait()

        try:
            await asyncio.wait_for(_collect(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return ToolResult(success=False, error="gws command timed out.")

        stdout_str = "".join(stdout_chunks).strip()
        stderr_str = "".join(stderr_chunks).strip()

        if proc.returncode != 0:
            error_msg = stderr_str or stdout_str or f"gws exited with code {proc.returncode}"
            if "no credentials" in error_msg.lower() or "token" in error_msg.lower():
                error_msg += "\n\nHint: Run 'gws auth login' to authenticate."
            return ToolResult(success=False, error=error_msg)

        output = stdout_str
        if len(output) > _MAX_OUTPUT_CHARS:
            output = output[:_MAX_OUTPUT_CHARS] + "\n\n... [output truncated]"

        return ToolResult(success=True, content=output)
