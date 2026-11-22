"""No request path climbs out of where it was sent.

The SPA catch-all served ``STATIC_DIR / path`` for any path, so
``GET /../../../../../../../../../../../../proc/self/cwd/.env`` returned the
deck's ``.env`` — to anyone, through the proxy too (a raw ``..`` reaches the
app unchanged). Scanners did exactly that (2026-11-19/20, two addresses):
the deck's secrets, ``config.yaml``, ``.git/config``, ``/etc/passwd``.

:func:`inside` is the containment check for any route that maps a URL path
onto files; :class:`PathTraversalGuard` refuses, before any route runs, a
request whose path has a ``..`` segment (also percent-encoded, also with
backslashes): no page or API of the deck uses one, so a future route that
forgets :func:`inside` stays closed.
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import unquote

from starlette.types import ASGIApp, Receive, Scope, Send


def has_parent_segment(path: str) -> bool:
    """Whether *path* (as it came, or decoded once or twice) has a ``..`` segment."""
    seen = path
    for _ in range(3):  # %252e%252e → %2e%2e → ..
        if any(part == ".." for part in seen.replace("\\", "/").split("/")):
            return True
        decoded = unquote(seen)
        if decoded == seen:
            return False
        seen = decoded
    return False


def inside(base: Path, rel: str) -> Path | None:
    """``base / rel`` when it stays inside *base* once resolved (symlinks and
    ``..`` followed); None otherwise."""
    root = base.resolve()
    try:
        target = (root / rel.lstrip("/\\")).resolve()
    except (OSError, ValueError, RuntimeError):
        return None
    return target if target == root or target.is_relative_to(root) else None


class PathTraversalGuard:
    """400 for an HTTP or WebSocket request whose path climbs with ``..``."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] in ("http", "websocket"):
            raw = scope.get("raw_path")
            raw_text = raw.decode("latin-1") if isinstance(raw, (bytes, bytearray)) else ""
            if has_parent_segment(str(scope.get("path") or "")) or has_parent_segment(raw_text):
                if scope["type"] == "websocket":
                    await send({"type": "websocket.close", "code": 1008})
                    return
                await send({"type": "http.response.start", "status": 400,
                            "headers": [(b"content-type", b"text/plain; charset=utf-8"),
                                        (b"content-length", b"11")]})
                await send({"type": "http.response.body", "body": b"Bad request"})
                return
        await self.app(scope, receive, send)
