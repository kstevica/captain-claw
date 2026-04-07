"""Small helpers for fire-and-forget WebSocket sends.

Several places in Captain Claw schedule ``ws.send_str(...)`` as a
background task without ever awaiting the result. When the client has
already disconnected, the underlying ``send_frame`` call raises
``ClientConnectionResetError`` (or ``ConnectionResetError``) and Python
prints a noisy "Task exception was never retrieved" traceback for every
queued send. This module wraps the pattern so those benign disconnect
errors are swallowed quietly while real bugs still surface.
"""

from __future__ import annotations

import asyncio
from typing import Any

from captain_claw.logging import get_logger

log = get_logger(__name__)


def _drain(task: asyncio.Task) -> None:
    """Done-callback that retrieves and discards expected disconnect errors."""
    try:
        exc = task.exception()
    except asyncio.CancelledError:
        return
    except Exception:  # pylint: disable=broad-exception-caught
        return
    if exc is None:
        return
    name = type(exc).__name__
    # Benign: client disconnected mid-send.
    if name in (
        "ClientConnectionResetError",
        "ConnectionResetError",
        "ConnectionAbortedError",
        "BrokenPipeError",
    ):
        return
    # Anything else is unexpected — surface it in the log so we don't
    # lose real failures.
    log.warning("ws send task failed", error=str(exc), error_type=name)


def fire_and_forget_send(ws: Any, payload: str) -> None:
    """Schedule ``ws.send_str(payload)`` and silently drop disconnect errors.

    Mirrors ``asyncio.ensure_future(ws.send_str(payload))`` but attaches
    a done-callback that prevents "Task exception was never retrieved"
    spam when the client has already gone away.
    """
    if ws is None:
        return
    if getattr(ws, "closed", False):
        return
    try:
        task = asyncio.ensure_future(ws.send_str(payload))
    except RuntimeError:
        # No running loop — caller is in shutdown. Drop silently.
        return
    task.add_done_callback(_drain)
