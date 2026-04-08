"""Google OAuth endpoints hosted by the Flight Deck backend.

Flight Deck is the single source of truth for Google OAuth credentials and
tokens. It performs the authorization flow, stores ``client_id`` /
``client_secret`` / refresh tokens in its own SQLite ``system_settings``
table, and exposes ``GET /fd/google/access_token`` so any number of
captain-claw agents (potentially running on different ports or hosts) can
pull a fresh access token without duplicating the OAuth dance.

Only one OAuth callback URL needs to be registered with Google — this
one — regardless of how many agents consume the tokens.
"""

from __future__ import annotations

import json
import os
import secrets
import time
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel

from captain_claw.flight_deck.auth import get_current_user, get_db
from captain_claw.flight_deck.db import FlightDeckDB
from captain_claw.google_oauth import (
    DEFAULT_SCOPES,
    SCOPE_CATALOG,
    GoogleOAuthTokens,
    build_authorization_url,
    exchange_code_for_tokens,
    fetch_user_info,
    generate_pkce_pair,
    refresh_access_token,
    revoke_token,
    sanitize_scopes,
)
from captain_claw.logging import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/fd/google", tags=["google-oauth"])


# ── OAuth client: user-supplied only ────────────────────────────────
#
# Captain Claw no longer ships with baked-in Google OAuth credentials.
# Every deployment MUST configure its own ``client_id`` /
# ``client_secret`` in Flight Deck's Connections page (or via the
# ``/fd/google/config`` endpoint). This avoids bundling secrets in the
# source tree / distribution and lets each user own their own Google
# Cloud project, consent screen, and scope verification posture.
#
# Default scopes (granted during the OAuth flow when present on the
# consent screen) are defined in ``captain_claw.google_oauth.DEFAULT_SCOPES``
# and include ``cloud-platform`` so Vertex AI / Gemini is available
# whenever the user also supplies a ``project_id``.


# ── system_settings keys ────────────────────────────────────────────

_K_CLIENT_ID = "google_oauth:client_id"
_K_CLIENT_SECRET = "google_oauth:client_secret"
_K_PROJECT_ID = "google_oauth:project_id"
_K_LOCATION = "google_oauth:location"
_K_SCOPES = "google_oauth:scopes"
_K_TOKENS = "google_oauth:tokens"
_K_USER = "google_oauth:user"
# Legacy / reserved — kept for backwards compatibility when reading
# older databases. Only "custom" is supported going forward.
_K_TOKEN_MODE = "google_oauth:token_mode"


_GMAIL_SCOPE_LABELS = {s["scope"]: s["label"] for s in SCOPE_CATALOG}


def _label_scopes(granted: list[str]) -> list[dict[str, str]]:
    return [{"scope": s, "label": _GMAIL_SCOPE_LABELS.get(s, s)} for s in granted]


# ── in-memory PKCE state ────────────────────────────────────────────
#
# PKCE verifiers are short-lived (≤ 10 min) and only needed between the
# login-redirect and the callback on the same Flight Deck process, so a
# plain dict is fine. No need to persist them.

_pending_oauth: dict[str, dict[str, Any]] = {}


def _purge_stale_pending() -> None:
    cutoff = time.time() - 600
    stale = [k for k, v in _pending_oauth.items() if v.get("ts", 0) < cutoff]
    for k in stale:
        _pending_oauth.pop(k, None)


# ── storage helpers ─────────────────────────────────────────────────


async def _load_user_config(db: FlightDeckDB) -> dict[str, Any]:
    """Return the user-supplied (custom) OAuth config from system_settings."""
    raw_scopes = (await db.get_system_setting(_K_SCOPES)) or ""
    scopes: list[str]
    if raw_scopes:
        try:
            loaded = json.loads(raw_scopes)
            if isinstance(loaded, list):
                scopes = sanitize_scopes([str(s) for s in loaded])
            else:
                scopes = list(DEFAULT_SCOPES)
        except Exception:
            scopes = list(DEFAULT_SCOPES)
    else:
        scopes = list(DEFAULT_SCOPES)
    return {
        "client_id": (await db.get_system_setting(_K_CLIENT_ID)) or "",
        "client_secret": (await db.get_system_setting(_K_CLIENT_SECRET)) or "",
        "project_id": (await db.get_system_setting(_K_PROJECT_ID)) or "",
        "location": (await db.get_system_setting(_K_LOCATION)) or "us-central1",
        "scopes": scopes,
    }


async def _effective_oauth(db: FlightDeckDB) -> dict[str, Any] | None:
    """Resolve the *active* OAuth client + scopes.

    Returns ``None`` when the user hasn't saved a ``client_id`` /
    ``client_secret`` pair yet — there is no bundled fallback. Callers
    must surface a "not configured" error to the user when this
    happens.
    """
    user = await _load_user_config(db)
    if not (user["client_id"] and user["client_secret"]):
        return None
    scopes = sanitize_scopes(user.get("scopes"))
    has_cloud = "https://www.googleapis.com/auth/cloud-platform" in scopes
    return {
        "mode": "custom",
        "client_id": user["client_id"],
        "client_secret": user["client_secret"],
        "project_id": user["project_id"],
        "location": user["location"],
        "scopes": scopes,
        "supports_vertex": bool(user["project_id"]) and has_cloud,
    }


async def _load_tokens(db: FlightDeckDB) -> GoogleOAuthTokens | None:
    raw = await db.get_system_setting(_K_TOKENS)
    if not raw:
        return None
    try:
        return GoogleOAuthTokens.from_dict(json.loads(raw))
    except Exception as exc:
        log.warning("Failed to deserialize Google OAuth tokens: %s", exc)
        return None


async def _store_tokens(db: FlightDeckDB, tokens: GoogleOAuthTokens) -> None:
    await db.set_system_setting(
        _K_TOKENS,
        json.dumps(tokens.to_dict(), ensure_ascii=True),
    )


async def _load_user(db: FlightDeckDB) -> dict[str, Any] | None:
    raw = await db.get_system_setting(_K_USER)
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception:
        return None


async def _store_user(db: FlightDeckDB, user: dict[str, Any]) -> None:
    await db.set_system_setting(_K_USER, json.dumps(user, ensure_ascii=True))


async def _token_client(db: FlightDeckDB) -> dict[str, Any] | None:
    """Resolve the OAuth client that minted the currently-stored tokens.

    Refreshing a token requires the same client_id/secret pair that
    issued it. With bundled credentials gone, the only valid source is
    the user-supplied config.
    """
    user_cfg = await _load_user_config(db)
    if not user_cfg["client_id"] or not user_cfg["client_secret"]:
        return None
    scopes = sanitize_scopes(user_cfg.get("scopes"))
    has_cloud = "https://www.googleapis.com/auth/cloud-platform" in scopes
    return {
        "mode": "custom",
        "client_id": user_cfg["client_id"],
        "client_secret": user_cfg["client_secret"],
        "project_id": user_cfg["project_id"],
        "location": user_cfg["location"] or "us-central1",
        "scopes": scopes,
        "supports_vertex": bool(user_cfg["project_id"]) and has_cloud,
    }


async def _clear_oauth_state(db: FlightDeckDB) -> None:
    # system_settings has no per-row delete helper — overwrite with empty JSON.
    await db.set_system_setting(_K_TOKENS, "")
    await db.set_system_setting(_K_USER, "")
    await db.set_system_setting(_K_TOKEN_MODE, "")


# ── redirect URI ────────────────────────────────────────────────────


def _redirect_uri(request: Request) -> str:
    """Build the redirect URI that Google will bounce the user back to.

    Uses the request's own host so it works for dev (localhost:25080),
    packaged desktop app, or behind a reverse proxy. This must exactly
    match the URI registered in Google Cloud Console.
    """
    scheme = request.url.scheme
    host = request.headers.get("host") or request.url.netloc
    # If FD_PUBLIC_URL is set, prefer it (handles proxied deployments).
    public = os.environ.get("FD_PUBLIC_URL", "").strip().rstrip("/")
    if public:
        return f"{public}/fd/google/callback"
    return f"{scheme}://{host}/fd/google/callback"


# ── models ──────────────────────────────────────────────────────────


class GoogleConfigUpdate(BaseModel):
    # Set ``clear=True`` to wipe all stored credentials + tokens. Other
    # fields: any field that is ``None`` is left unchanged; any non-empty
    # string overwrites the existing value. ``mode`` is accepted for
    # backwards compatibility but ignored.
    mode: str | None = None
    clear: bool | None = None
    client_id: str | None = None
    client_secret: str | None = None
    project_id: str | None = None
    location: str | None = None
    # ``scopes`` is the *full* desired list — pass the current list with
    # items added/removed. Unknown or duplicate entries are silently
    # dropped server-side. ``None`` leaves the stored list unchanged.
    scopes: list[str] | None = None


# ── status / config endpoints ───────────────────────────────────────


@router.get("/status")
async def google_status(
    request: Request,
    _user: dict = Depends(get_current_user),
) -> dict[str, Any]:
    """Return Google OAuth connection status for the UI."""
    db = get_db()
    eff = await _effective_oauth(db)

    tokens = await _load_tokens(db)
    connected = bool(eff and tokens and tokens.refresh_token)
    user = await _load_user(db) if connected else None
    scopes = tokens.scope.split() if tokens and tokens.scope else []

    return {
        "configured": eff is not None,
        "mode": "custom",
        "supports_vertex": bool(eff and eff["supports_vertex"]),
        "connected": connected,
        "user": user,
        "granted_scopes": _label_scopes(scopes),
        "redirect_uri": _redirect_uri(request),
    }


@router.get("/config")
async def google_config_get(
    request: Request,
    _user: dict = Depends(get_current_user),
) -> dict[str, Any]:
    """Return the stored Google OAuth configuration (never returns the secret)."""
    db = get_db()
    user = await _load_user_config(db)
    return {
        "mode": "custom",
        "client_id": user["client_id"],
        "client_id_set": bool(user["client_id"]),
        "client_secret_set": bool(user["client_secret"]),
        "project_id": user["project_id"],
        "location": user["location"] or "us-central1",
        "scopes": user["scopes"],
        "default_scopes": list(DEFAULT_SCOPES),
        "scope_catalog": SCOPE_CATALOG,
        "redirect_uri": _redirect_uri(request),
    }


@router.get("/scope_catalog")
async def google_scope_catalog(
    _user: dict = Depends(get_current_user),
) -> dict[str, Any]:
    """Return the catalogue of selectable OAuth scopes for the UI."""
    return {"scope_catalog": SCOPE_CATALOG, "default_scopes": list(DEFAULT_SCOPES)}


@router.post("/config")
async def google_config_post(
    body: GoogleConfigUpdate,
    _user: dict = Depends(get_current_user),
) -> dict[str, Any]:
    """Save Google OAuth credentials.

    - ``clear=True``: wipe saved credentials AND any stored tokens. The
      Google connection becomes "not configured" until the user enters
      a new ``client_id`` / ``client_secret``.
    - Otherwise: any field that is ``None`` is left unchanged. Any field
      that is a non-empty string overwrites the existing value. If the
      ``client_id`` changes, stored tokens are invalidated (the refresh
      token is bound to the OAuth client that minted it).
    """
    db = get_db()

    if body.clear:
        await db.set_system_setting(_K_CLIENT_ID, "")
        await db.set_system_setting(_K_CLIENT_SECRET, "")
        await db.set_system_setting(_K_PROJECT_ID, "")
        await db.set_system_setting(_K_SCOPES, "")
        await _clear_oauth_state(db)
        return {"ok": True, "cleared": True}

    def _clean(v: str | None) -> str | None:
        if v is None:
            return None
        s = v.strip()
        return s if s else None

    client_id = _clean(body.client_id)
    client_secret = _clean(body.client_secret)
    project_id = _clean(body.project_id)
    location = _clean(body.location)

    # If the client_id rotates, wipe stored tokens — the refresh token
    # was minted by the previous OAuth client and will no longer work.
    eff_before = await _effective_oauth(db)
    prev_client_id = eff_before["client_id"] if eff_before else ""
    will_change_client = client_id is not None and client_id != prev_client_id

    if client_id is not None:
        await db.set_system_setting(_K_CLIENT_ID, client_id)
    if client_secret is not None:
        await db.set_system_setting(_K_CLIENT_SECRET, client_secret)
    if project_id is not None:
        await db.set_system_setting(_K_PROJECT_ID, project_id)
    if location is not None:
        await db.set_system_setting(_K_LOCATION, location)

    scopes_changed = False
    if body.scopes is not None:
        new_scopes = sanitize_scopes(body.scopes)
        prev_scopes = eff_before["scopes"] if eff_before else list(DEFAULT_SCOPES)
        if set(new_scopes) != set(prev_scopes):
            scopes_changed = True
        await db.set_system_setting(_K_SCOPES, json.dumps(new_scopes))

    # Both rotating the client and changing the scope set invalidate
    # the stored tokens — the refresh token is bound to the client that
    # minted it, and scope changes require a fresh consent.
    if will_change_client or scopes_changed:
        await _clear_oauth_state(db)

    return {"ok": True, "mode": "custom", "scopes_changed": scopes_changed}


# ── login / callback / logout ───────────────────────────────────────


@router.get("/login")
async def google_login(request: Request) -> RedirectResponse:
    """Start the OAuth flow by redirecting to Google's consent screen.

    This endpoint is intentionally unauthenticated: it is opened as a
    popup from the Flight Deck UI, which means cookies / Authorization
    headers may not ride along, and the user has already authenticated
    into Flight Deck to click the button. Google's own login then
    gates the actual credential exchange.
    """
    db = get_db()
    eff = await _effective_oauth(db)
    if not eff:
        return HTMLResponse(
            "<h3>Google OAuth not configured</h3>"
            "<p>Enter your Client ID and Client Secret on the Connections "
            "page first, then click Connect Google again.</p>",
            status_code=400,
        )

    _purge_stale_pending()

    state = secrets.token_urlsafe(32)
    verifier, challenge = generate_pkce_pair()
    _pending_oauth[state] = {
        "verifier": verifier,
        "ts": time.time(),
    }

    auth_url = build_authorization_url(
        client_id=eff["client_id"],
        redirect_uri=_redirect_uri(request),
        scopes=eff["scopes"],
        state=state,
        code_challenge=challenge,
    )
    return RedirectResponse(auth_url, status_code=302)


@router.get("/callback")
async def google_callback(request: Request) -> HTMLResponse:
    """Handle Google's redirect, exchange the code, store tokens."""
    error = request.query_params.get("error")
    if error:
        desc = request.query_params.get("error_description", error)
        return _callback_html(ok=False, title="OAuth error", detail=desc)

    code = request.query_params.get("code", "")
    state = request.query_params.get("state", "")
    if not code or not state:
        return _callback_html(ok=False, title="Missing code or state")

    pending = _pending_oauth.pop(state, None)
    if not pending:
        return _callback_html(ok=False, title="Invalid or expired state, please try again.")

    db = get_db()
    user_cfg = await _load_user_config(db)
    if not user_cfg["client_id"] or not user_cfg["client_secret"]:
        return _callback_html(
            ok=False,
            title="Google OAuth credentials were removed mid-flow.",
        )
    client_id = user_cfg["client_id"]
    client_secret = user_cfg["client_secret"]

    try:
        tokens = await exchange_code_for_tokens(
            code=code,
            client_id=client_id,
            client_secret=client_secret,
            redirect_uri=_redirect_uri(request),
            code_verifier=pending["verifier"],
        )
    except Exception as exc:
        log.error("Google OAuth token exchange failed: %s", exc)
        return _callback_html(ok=False, title="Token exchange failed", detail=str(exc))

    try:
        user = await fetch_user_info(tokens.access_token)
    except Exception as exc:
        log.warning("Failed to fetch Google user info: %s", exc)
        user = {}

    await _store_tokens(db, tokens)
    await db.set_system_setting(_K_TOKEN_MODE, "custom")
    if user:
        await _store_user(db, user)

    return _callback_html(
        ok=True,
        title="Connected",
        detail=f"Signed in as {user.get('email', 'your Google account')}",
        email=user.get("email", ""),
    )


@router.post("/logout")
async def google_logout(_user: dict = Depends(get_current_user)) -> dict[str, Any]:
    """Revoke tokens and clear all stored Google OAuth state."""
    db = get_db()
    tokens = await _load_tokens(db)
    if tokens:
        if tokens.refresh_token:
            await revoke_token(tokens.refresh_token)
        elif tokens.access_token:
            await revoke_token(tokens.access_token)
    await _clear_oauth_state(db)
    return {"disconnected": True}


# ── agent-facing endpoint ───────────────────────────────────────────


def _agent_shared_secret() -> str:
    return os.environ.get("FD_AGENT_SHARED_SECRET", "").strip()


def _authorize_agent_call(request: Request) -> None:
    """Gate /access_token and /credentials for captain-claw agents.

    Two accepted authentication modes:

    1. Shared secret via ``X-Agent-Secret`` header (when
       ``FD_AGENT_SHARED_SECRET`` is set in the Flight Deck env).
    2. Request originating from loopback (127.0.0.1 / ::1). Agents
       typically run on the same host as Flight Deck, and the OS-level
       localhost boundary is a reasonable trust zone.
    """
    secret = _agent_shared_secret()
    if secret:
        provided = request.headers.get("X-Agent-Secret", "")
        if provided and secrets.compare_digest(provided, secret):
            return
        # Fall through to loopback check if header missing — easier dev UX.

    client_host = request.client.host if request.client else ""
    if client_host in ("127.0.0.1", "::1", "localhost"):
        return

    raise HTTPException(status_code=401, detail="Unauthorized agent call")


async def _refresh_if_needed(
    db: FlightDeckDB,
    client: dict[str, Any],
    tokens: GoogleOAuthTokens,
) -> GoogleOAuthTokens | None:
    """Refresh *tokens* when near expiry; persist the new pair.

    *client* must be the OAuth client that originally minted the
    refresh token (use :func:`_token_client`).
    """
    if not tokens.is_expired():
        return tokens
    if not tokens.refresh_token:
        return None
    try:
        fresh = await refresh_access_token(
            refresh_token=tokens.refresh_token,
            client_id=client["client_id"],
            client_secret=client["client_secret"],
        )
    except Exception as exc:
        log.warning("Google token refresh failed: %s", exc)
        return None
    await _store_tokens(db, fresh)
    return fresh


@router.get("/access_token")
async def google_access_token(request: Request) -> dict[str, Any]:
    """Return a currently-valid access token for a captain-claw agent."""
    _authorize_agent_call(request)
    db = get_db()
    client = await _token_client(db)
    if not client:
        raise HTTPException(status_code=404, detail="Google OAuth not configured")
    tokens = await _load_tokens(db)
    if not tokens:
        raise HTTPException(status_code=404, detail="No stored Google OAuth tokens")
    tokens = await _refresh_if_needed(db, client, tokens)
    if not tokens:
        raise HTTPException(status_code=401, detail="Could not refresh Google access token")
    return {
        "access_token": tokens.access_token,
        "token_type": tokens.token_type,
        "expires_at": tokens.expires_at,
        "scope": tokens.scope,
    }


@router.get("/credentials")
async def google_credentials(request: Request) -> dict[str, Any]:
    """Return full ``authorized_user`` credentials for LiteLLM / Vertex.

    Intended for captain-claw agents wiring up the Gemini provider via
    LiteLLM's Vertex AI path. Requires that the user has supplied a
    ``project_id`` alongside their ``client_id`` / ``client_secret``.
    """
    _authorize_agent_call(request)
    db = get_db()
    client = await _token_client(db)
    if not client:
        raise HTTPException(status_code=404, detail="Google OAuth not configured")
    if not client["supports_vertex"]:
        raise HTTPException(
            status_code=409,
            detail=(
                "Vertex AI requires a Google Cloud project_id. Add one to "
                "your Google OAuth credentials on the Connections page."
            ),
        )
    tokens = await _load_tokens(db)
    if not tokens:
        raise HTTPException(status_code=404, detail="No stored Google OAuth tokens")
    tokens = await _refresh_if_needed(db, client, tokens)
    if not tokens:
        raise HTTPException(status_code=401, detail="Could not refresh Google access token")
    return {
        "credentials": tokens.to_vertex_credentials_json(
            client_id=client["client_id"],
            client_secret=client["client_secret"],
        ),
        "project_id": client["project_id"],
        "location": client["location"],
    }


# ── callback HTML ───────────────────────────────────────────────────


def _callback_html(
    *,
    ok: bool,
    title: str,
    detail: str = "",
    email: str = "",
) -> HTMLResponse:
    """Render a popup-aware confirmation page.

    Tries to ``postMessage`` the result to ``window.opener`` and close
    itself. Falls back to a meta-refresh redirect to ``/`` for the
    full-tab flow.
    """
    safe_email = email.replace("\\", "\\\\").replace("'", "\\'")
    safe_title = title.replace("\\", "\\\\").replace("'", "\\'")
    safe_detail = detail.replace("\\", "\\\\").replace("'", "\\'")
    status_word = "success" if ok else "error"

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
body{{background:#0d1117;color:#e6edf3;font-family:-apple-system,system-ui,sans-serif;
display:flex;align-items:center;justify-content:center;min-height:100vh;margin:0;}}
.box{{text-align:center;padding:2rem;}}
.box h2{{margin:0 0 0.5rem;font-weight:600;}}
.box p{{color:#8b949e;margin:0;}}
.ok{{color:#3fb950;}}
.err{{color:#f85149;}}
</style>
</head>
<body>
<div class="box">
<h2 class="{'ok' if ok else 'err'}">{title}</h2>
<p>{detail}</p>
<p style="margin-top:1rem;font-size:0.8rem;">You can close this window.</p>
</div>
<script>
(function() {{
  var payload = {{
    type: 'captain-claw-google-oauth',
    status: '{status_word}',
    title: '{safe_title}',
    detail: '{safe_detail}',
    email: '{safe_email}'
  }};
  try {{
    if (window.opener && !window.opener.closed) {{
      window.opener.postMessage(payload, '*');
      setTimeout(function(){{ window.close(); }}, 400);
      return;
    }}
  }} catch (e) {{}}
  setTimeout(function(){{ window.location.href = '/'; }}, 1500);
}})();
</script>
</body>
</html>"""
    return HTMLResponse(content=html)
