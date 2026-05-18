"""SDK for code-apps to talk to Flight Deck + sibling apps.

A "code-app" is an agent-authored ``backend.py`` + ``frontend.html``
pair that runs in its own subprocess under
:mod:`captain_claw.flight_deck.app_runtime`. By default each app's
data is private to that subprocess. This module lets one app read
another's public API in a controlled way.

Two halves
----------
1. **Inside ``backend.py``** — import :func:`sibling` and call another
   app's endpoints::

       from captain_claw.app_sdk import sibling

       async def handle(method, path, headers, body):
           contacts = await sibling("contacts").get_json("/contacts")
           ...

2. **On the target app** — declare a ``data_api`` block in its
   ``manifest.json``. Without that opt-in, FD returns 403 to any
   ``X-FD-Agent-As: app:<slug>`` caller. This is the publish/consume
   contract.

Auth model
----------
Both halves trust the host-local Captain Claw shared secret (see
:mod:`captain_claw.flight_deck.agent_secret`). The FD process injects
``FD_AGENT_SECRET`` + ``FD_INTERNAL_URL`` into every app subprocess
at spawn time, so the only thing user-written ``backend.py`` has to
know is the slug of the sibling it wants to read from.

What this is not
----------------
* Not a security boundary between apps. All code-apps are written by
  the same agent for the same user; the ``data_api`` gate is about
  intent + discoverability, not isolation. If you need real isolation,
  apps need to run as different OS users, which they don't today.
* Not a write surface. v1 is read-friendly: do POSTs at your own risk,
  the contract a publisher declares should be read-only.
"""

from __future__ import annotations

import json as _json
import os
from typing import Any

import httpx


# Env vars injected by app_runtime._build_subprocess_env. Read once at
# import time — subprocess env doesn't change after spawn.
_FD_URL = (os.environ.get("FD_INTERNAL_URL") or "").rstrip("/")
_AGENT_SECRET = os.environ.get("FD_AGENT_SECRET") or ""
_APP_SLUG = os.environ.get("FD_APP_SLUG") or ""

# Sensible default — long enough for cold-spawning a sibling, short
# enough that a stuck call doesn't hang the caller's whole request.
_DEFAULT_TIMEOUT = 15.0


class SiblingError(RuntimeError):
    """A cross-app call returned a non-2xx response.

    Attributes ``slug``, ``path``, ``status``, ``body`` are exposed so
    callers can branch on them without reparsing the message.
    """

    def __init__(self, slug: str, path: str, status: int, body: str):
        super().__init__(
            f"sibling({slug!r}).request({path!r}) → HTTP {status}: {body[:300]}"
        )
        self.slug = slug
        self.path = path
        self.status = status
        self.body = body


class Sibling:
    """Handle for one other code-app, accessed by slug.

    Each method goes through the FD proxy
    (``/fd/code-apps/<slug>/api/<path>``) using the shared agent
    secret. The target app sees a normal request hitting its
    backend's ``handle()`` — it doesn't need any cross-app awareness.
    """

    __slots__ = ("slug", "_timeout")

    def __init__(self, slug: str, *, timeout: float = _DEFAULT_TIMEOUT) -> None:
        if not slug or not isinstance(slug, str):
            raise ValueError("sibling slug must be a non-empty string")
        self.slug = slug
        self._timeout = timeout

    # ── low-level ──────────────────────────────────────────────────

    async def request(
        self,
        path: str,
        *,
        method: str = "GET",
        params: dict[str, Any] | None = None,
        json: Any = None,
        body: bytes | str | None = None,
        headers: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> tuple[int, dict[str, str], bytes]:
        """Issue one request to the sibling app. Returns (status, headers, body).

        Doesn't raise on non-2xx — callers decide how to interpret.
        See :meth:`get_json` / :meth:`post_json` for the common path
        with auto-deserialization + error-raising.
        """
        if not _FD_URL:
            raise RuntimeError(
                "FD_INTERNAL_URL not set in subprocess env — cross-app "
                "calls require Flight Deck to inject it. Check that "
                "this process was spawned by app_runtime, not directly."
            )
        if not path.startswith("/"):
            path = "/" + path
        url = f"{_FD_URL}/fd/code-apps/{self.slug}/api{path}"

        hdrs: dict[str, str] = dict(headers or {})
        if _AGENT_SECRET:
            hdrs["X-FD-Agent-Secret"] = _AGENT_SECRET
            # Identifies us as another app, not the user's chat agent.
            # FD uses this to enforce the target's data_api opt-in.
            if _APP_SLUG:
                hdrs["X-FD-Agent-As"] = f"app:{_APP_SLUG}"
                hdrs["X-FD-Agent-User"] = f"app:{_APP_SLUG}"

        kwargs: dict[str, Any] = {"headers": hdrs, "params": params}
        if json is not None:
            kwargs["json"] = json
        elif body is not None:
            kwargs["content"] = (
                body.encode("utf-8") if isinstance(body, str) else body
            )

        async with httpx.AsyncClient(timeout=timeout or self._timeout) as client:
            resp = await client.request(method, url, **kwargs)
        return resp.status_code, dict(resp.headers), resp.content

    # ── convenience ────────────────────────────────────────────────

    async def get_json(
        self,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> Any:
        """GET + JSON-decode. Raises :class:`SiblingError` on non-2xx."""
        status, _h, body = await self.request(
            path, method="GET", params=params, timeout=timeout,
        )
        self._raise_for_status(path, status, body)
        return _decode_json(body)

    async def post_json(
        self,
        path: str,
        payload: Any,
        *,
        timeout: float | None = None,
    ) -> Any:
        """POST a JSON body + decode response. Raises on non-2xx."""
        status, _h, body = await self.request(
            path, method="POST", json=payload, timeout=timeout,
        )
        self._raise_for_status(path, status, body)
        return _decode_json(body)

    def _raise_for_status(self, path: str, status: int, body: bytes) -> None:
        if status >= 400:
            text = body.decode("utf-8", errors="replace") if body else ""
            raise SiblingError(self.slug, path, status, text)


def sibling(slug: str, *, timeout: float = _DEFAULT_TIMEOUT) -> Sibling:
    """Return a :class:`Sibling` handle for ``slug``.

    Cheap — does no network I/O until you call a method on the handle.
    Safe to call per-request; no need to cache.
    """
    return Sibling(slug, timeout=timeout)


# ── helpers ──────────────────────────────────────────────────────────


def _decode_json(body: bytes) -> Any:
    if not body:
        return None
    try:
        return _json.loads(body.decode("utf-8"))
    except _json.JSONDecodeError as exc:
        raise SiblingError(
            slug="?", path="?", status=200,
            body=f"response was not JSON: {exc}: {body[:200]!r}",
        ) from exc


# Exposed for diagnostics — apps can log "am I configured for
# cross-app calls?" without poking environ directly.
def configured() -> bool:
    """True if both FD URL and agent secret are present in env."""
    return bool(_FD_URL and _AGENT_SECRET)
