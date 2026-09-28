"""Shared helpers for talking to Flight Deck from inside a captain-claw agent.

When captain-claw is launched as a subprocess by Flight Deck, FD injects:

* ``FD_URL`` — the FD HTTP base, e.g. ``http://127.0.0.1:8765``
* ``FD_AGENT_SLUG`` — the agent's slug (used by FD to identify the caller)
* ``FD_AGENT_SHARED_SECRET`` — a per-process secret used for the
  ``X-Agent-Secret`` header on FD-internal endpoints

This module is the single source of truth for those env vars. Anything
that needs to call back into Flight Deck — Codex token resolution, MCP
proxying, etc. — should import the helpers here rather than re-reading
the env directly.

Two layers are provided:

1. The bare functions :func:`flight_deck_base`, :func:`flight_deck_headers`,
   :func:`flight_deck_slug`, :func:`is_under_flight_deck` for callers that
   only need to know the env state.

2. :class:`FDClient` — a thin async wrapper around ``httpx.AsyncClient``
   that pre-populates the base URL + auth headers and exposes
   ``get`` / ``post`` / ``delete`` for typed RPC-style calls. Use this
   when you want to share an HTTP connection pool across many calls
   (e.g. the MCP proxy connector).
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Mapping

import httpx

from captain_claw.logging import get_logger

log = get_logger(__name__)


def flight_deck_base() -> str:
    """Return the Flight Deck base URL, or ``""`` when not running under FD.

    Trailing slashes are stripped so callers can ``f"{base}/fd/..."``
    without worrying about double slashes.

    Also defensively strips a trailing ``/fd`` segment if present —
    every caller in the codebase appends ``/fd/<route>`` itself, so an
    ``FD_URL`` like ``http://host:25080/fd`` would otherwise produce a
    double-prefixed ``/fd/fd/<route>`` URL that hits FD's SPA catchall
    instead of the API. This trap has bitten real users (an agent
    spawned with the wrong inherited ``FD_URL`` silently fell back to
    the SPA index.html, breaking Codex auth + MCP discovery).
    """
    base = (os.environ.get("FD_URL") or "").strip().rstrip("/")
    if base.endswith("/fd"):
        base = base[:-3].rstrip("/")
    return base


def flight_deck_headers() -> dict[str, str]:
    """Return the FD-internal auth headers.

    Always includes ``X-Agent-Slug`` when ``FD_AGENT_SLUG`` is set, so
    FD can identify which agent in the fleet is calling — needed for
    per-agent MCP allowlists and any future per-agent telemetry.

    Includes ``X-Agent-Secret`` when ``FD_AGENT_SHARED_SECRET`` is set;
    that header is what gates FD's agent-facing endpoints when the
    request doesn't come from loopback. When neither env var is set
    this returns an empty dict — callers should treat that as "FD auth
    not available" and fall back to local behaviour.
    """
    out: dict[str, str] = {}
    secret = (os.environ.get("FD_AGENT_SHARED_SECRET") or "").strip()
    if secret:
        out["X-Agent-Secret"] = secret
    slug = (os.environ.get("FD_AGENT_SLUG") or "").strip()
    if slug:
        out["X-Agent-Slug"] = slug
    return out


def flight_deck_slug() -> str:
    """Return the FD-assigned agent slug for this process, or ``""``."""
    return (os.environ.get("FD_AGENT_SLUG") or "").strip()


def is_under_flight_deck() -> bool:
    """``True`` when this process was spawned by Flight Deck.

    The presence of ``FD_URL`` is sufficient — the secret may legitimately
    be missing on public/unauthenticated endpoints.
    """
    return bool(flight_deck_base())


def _norm_url(url: str | None) -> str:
    base = str(url or "").strip().rstrip("/")
    if base.endswith("/fd"):
        base = base[:-3].rstrip("/")
    return base


def pinned_flight_deck_url() -> str:
    """The FD base URL this agent was configured with — ``FD_URL`` (FD pins it
    to itself at spawn), else ``FD_INTERNAL_URL`` — or ``""``."""
    return _norm_url(os.environ.get("FD_URL") or os.environ.get("FD_INTERNAL_URL"))


def _pinned_flight_deck_urls() -> set[str]:
    """Every FD base URL from this agent's environment or config
    (``google_oauth.flight_deck_url``) — never one a websocket client supplied
    (the session ``fd_url`` a ``peer_agents`` message sets)."""
    urls = {os.environ.get("FD_URL", ""), os.environ.get("FD_INTERNAL_URL", "")}
    try:
        from captain_claw.config import get_config

        urls.add(str(get_config().google_oauth.flight_deck_url or ""))
    except Exception:
        pass
    return {u.lower() for u in map(_norm_url, urls) if u}


def agent_identity_headers(fd_url: str) -> dict[str, str]:
    """This agent's identity for FD's peer routes (/fd/consult-peer,
    /fd/delegate-peer) at ``fd_url`` — ``{}`` unless that is a pinned FD URL.

    * ``X-Agent-Auth`` — the agent's own web_auth; FD maps it to the owner it
      recorded at spawn and lets the agent drive only that owner's agents.
    * ``X-Agent-Secret`` — the deck's agent secret (config
      ``google_oauth.flight_deck_secret``, else ``FD_AGENT_SHARED_SECRET`` /
      the per-deck file); needed off-loopback and under FD_LOCKDOWN.

    Both are credentials, so they never go to a URL a websocket client chose.
    Best-effort: a header that can't be resolved is left out.
    """
    if not fd_url or _norm_url(fd_url).lower() not in _pinned_flight_deck_urls():
        return {}
    try:
        from captain_claw.config import get_config

        cfg = get_config()
    except Exception:
        cfg = None
    headers: dict[str, str] = {}
    secret = str(getattr(getattr(cfg, "google_oauth", None), "flight_deck_secret", "") or "").strip()
    if not secret:
        try:
            from captain_claw.flight_deck.agent_secret import get_or_create_agent_secret

            secret = get_or_create_agent_secret()
        except Exception:
            secret = ""
    if secret:
        headers["X-Agent-Secret"] = secret
    token = str(getattr(getattr(cfg, "web", None), "auth_token", "") or "").strip()
    if token:
        headers["X-Agent-Auth"] = token
    return headers


async def refusal_detail(resp: httpx.Response) -> str:
    """``"Flight Deck returned HTTP <code>"`` plus FD's reason (a 4xx
    ``detail``), for surfacing a refusal to the model. Works on streamed
    responses too (reads the body first)."""
    detail = ""
    try:
        await resp.aread()
        detail = str((resp.json() or {}).get("detail") or "")
    except Exception:
        pass
    return f"Flight Deck returned HTTP {resp.status_code}" + (f": {detail}" if detail else "")


class FDClient:
    """Lazy async HTTP client for Flight Deck-internal endpoints.

    Reuses one :class:`httpx.AsyncClient` across calls. Auth headers are
    merged on every request so callers don't need to pass them. When
    ``FD_URL`` is unset every call raises :class:`RuntimeError` — callers
    that want graceful local fallback should check
    :func:`is_under_flight_deck` first.
    """

    def __init__(self, *, timeout: float = 10.0) -> None:
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None

    async def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._timeout)
        return self._client

    async def close(self) -> None:
        if self._client is not None:
            try:
                await self._client.aclose()
            except Exception:
                pass
            self._client = None

    def _url(self, path: str) -> str:
        base = flight_deck_base()
        if not base:
            raise RuntimeError("FD_URL is not set; cannot reach Flight Deck")
        if not path.startswith("/"):
            path = "/" + path
        return f"{base}{path}"

    def _headers(self, extra: Mapping[str, str] | None = None) -> dict[str, str]:
        headers = dict(flight_deck_headers())
        if extra:
            headers.update(dict(extra))
        return headers

    async def get(
        self,
        path: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> httpx.Response:
        client = await self._http()
        return await client.get(
            self._url(path), params=params, headers=self._headers(headers)
        )

    async def post(
        self,
        path: str,
        *,
        json: Any = None,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> httpx.Response:
        client = await self._http()
        return await client.post(
            self._url(path),
            json=json,
            params=params,
            headers=self._headers(headers),
        )

    async def delete(
        self,
        path: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> httpx.Response:
        client = await self._http()
        return await client.delete(
            self._url(path), params=params, headers=self._headers(headers)
        )

    @asynccontextmanager
    async def stream(
        self,
        method: str,
        path: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> AsyncIterator[httpx.Response]:
        """Async context manager around ``httpx.AsyncClient.stream``.

        Used by the MCP SSE subscriber so a long-lived connection
        doesn't tie up the shared client's per-request timeout.
        ``timeout=None`` disables the read timeout (the server is
        expected to keep the socket alive with periodic pings).
        """
        client = await self._http()
        async with client.stream(
            method,
            self._url(path),
            params=params,
            headers=self._headers(headers),
            timeout=timeout,
        ) as response:
            yield response
