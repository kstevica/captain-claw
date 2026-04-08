"""Flight Deck endpoints for OpenAI "Sign in with ChatGPT" (Codex) auth.

Unlike the Google OAuth flow, captain-claw does NOT perform the
ChatGPT OAuth dance itself — that is the Codex CLI's job. Codex CLI
stores the resulting tokens in ``~/.codex/auth.json`` on the host
where it was run, and refreshes them automatically while it is
running.

Flight Deck's role here is just the *distribution* side:

1. Read ``~/.codex/auth.json`` from the FD host when asked.
2. Expose a pretty status view (email, plan, expiry) to the
   Connections page.
3. Expose ``/fd/codex/access_token`` so any captain-claw agent
   (possibly spawned on another host) can pull fresh tokens over
   loopback or via a shared secret — matching the Google OAuth
   auth model.

There is no persistent storage: every call re-reads the file, so
whatever the Codex CLI most recently wrote is what clients see.
"""

from __future__ import annotations

import json
import os
import secrets
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request

from captain_claw.codex_auth_manager import (
    _parse_auth_json,
    _CODEX_AUTH_PATH,
    load_tokens_from_disk,
)
from captain_claw.flight_deck.auth import get_current_user
from captain_claw.logging import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/fd/codex", tags=["codex-oauth"])


# ── agent auth (matches google_oauth_routes) ────────────────────────


def _agent_shared_secret() -> str:
    return os.environ.get("FD_AGENT_SHARED_SECRET", "").strip()


def _authorize_agent_call(request: Request) -> None:
    """Gate ``/access_token`` for captain-claw agents."""
    secret = _agent_shared_secret()
    if secret:
        provided = request.headers.get("X-Agent-Secret", "")
        if provided and secrets.compare_digest(provided, secret):
            return
    client_host = request.client.host if request.client else ""
    if client_host in ("127.0.0.1", "::1", "localhost"):
        return
    raise HTTPException(status_code=401, detail="Unauthorized agent call")


# ── status endpoint (UI) ────────────────────────────────────────────


@router.get("/status")
async def codex_status(_user: dict = Depends(get_current_user)) -> dict[str, Any]:
    """Return the current ChatGPT-OAuth connection status for the UI."""
    path = Path(_CODEX_AUTH_PATH)
    if not path.exists():
        return {
            "configured": False,
            "connected": False,
            "reason": "file_missing",
            "auth_path": str(path),
            "detail": (
                "~/.codex/auth.json not found on the Flight Deck host. "
                "Install the Codex CLI and run `codex login` first."
            ),
        }

    tokens = load_tokens_from_disk()
    if tokens is None:
        return {
            "configured": True,
            "connected": False,
            "reason": "parse_failed",
            "auth_path": str(path),
            "detail": "Found ~/.codex/auth.json but could not parse tokens from it.",
        }

    import time as _time
    now = _time.time()
    seconds_until_expiry = max(0.0, tokens.expires_at - now) if tokens.expires_at else 0.0
    stale = tokens.is_stale(now)

    return {
        "configured": True,
        "connected": True,
        "auth_path": str(path),
        "email": tokens.email,
        "plan": tokens.plan,
        "account_id": tokens.account_id,
        "expires_at": tokens.expires_at,
        "seconds_until_expiry": seconds_until_expiry,
        "stale": stale,
        # Truncated preview of the access_token so users can sanity-check
        # that the file they expect is being read, without leaking it.
        "access_token_preview": (
            tokens.access_token[:12] + "…" + tokens.access_token[-6:]
            if len(tokens.access_token) > 24
            else "…"
        ),
    }


@router.post("/reimport")
async def codex_reimport(_user: dict = Depends(get_current_user)) -> dict[str, Any]:
    """Re-read ``~/.codex/auth.json`` and return the refreshed status.

    There's no persistent state to update — every request already
    re-reads the file on demand. This endpoint exists so the UI has a
    clear "Reimport from Codex CLI" affordance: press it after running
    ``codex login`` again to see the new email / plan / expiry.
    """
    return await codex_status(_user=_user)  # type: ignore[arg-type]


# ── agent-facing endpoint ───────────────────────────────────────────


@router.get("/access_token")
async def codex_access_token(request: Request) -> dict[str, Any]:
    """Return current OAuth tokens for a captain-claw agent.

    Agents call this instead of reading ``~/.codex/auth.json`` directly
    so that a single FD instance can serve many sub-agents — including
    ones running on different hosts. Gated the same way the Google
    equivalent is (loopback OR ``X-Agent-Secret`` header).
    """
    _authorize_agent_call(request)
    tokens = load_tokens_from_disk()
    if tokens is None:
        raise HTTPException(
            status_code=404,
            detail=(
                "Codex OAuth not available — ~/.codex/auth.json is missing "
                "or unreadable on the Flight Deck host. Run `codex login`."
            ),
        )
    return {
        "access_token": tokens.access_token,
        "account_id": tokens.account_id,
        "expires_at": tokens.expires_at,
        "email": tokens.email,
        "plan": tokens.plan,
    }
