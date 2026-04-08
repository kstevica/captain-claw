"""Google OAuth flow handlers."""

from __future__ import annotations

import secrets
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml
from aiohttp import web

from captain_claw.config import DEFAULT_CONFIG_PATH, Config, get_config, set_config
from captain_claw.google_oauth import (
    DEFAULT_SCOPES,
    build_authorization_url,
    exchange_code_for_tokens,
    fetch_user_info,
    generate_pkce_pair,
)
from captain_claw.logging import get_logger

if TYPE_CHECKING:
    from captain_claw.web_server import WebServer

log = get_logger(__name__)


# ── CORS helpers ────────────────────────────────────────────────────
#
# Flight Deck is typically served from a different origin (its own
# FastAPI server on a different port, or Vite dev server on :5173).
# Allow its Google OAuth fetches to cross origins. The status / config
# endpoints expose nothing secret (we never return the client_secret)
# and writes require a POST with a JSON body, so permissive CORS here
# is safe.


def _cors_headers(request: web.Request) -> dict[str, str]:
    origin = request.headers.get("Origin", "*")
    return {
        "Access-Control-Allow-Origin": origin,
        "Access-Control-Allow-Credentials": "true",
        "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type, Authorization",
        "Vary": "Origin",
    }


def _json_cors(request: web.Request, data: Any, status: int = 200) -> web.Response:
    return web.json_response(data, status=status, headers=_cors_headers(request))


async def auth_google_cors_preflight(
    server: WebServer, request: web.Request,
) -> web.Response:
    """Answer any OPTIONS preflight for /auth/google/* routes."""
    return web.Response(status=204, headers=_cors_headers(request))


# ── helpers ─────────────────────────────────────────────────────────

_GMAIL_SCOPE_LABELS = {
    "https://www.googleapis.com/auth/drive": "Drive",
    "https://www.googleapis.com/auth/calendar": "Calendar",
    "https://www.googleapis.com/auth/gmail.readonly": "Gmail (read)",
    "https://www.googleapis.com/auth/gmail.compose": "Gmail (drafts)",
    "https://www.googleapis.com/auth/gmail.modify": "Gmail (read + modify)",
    "https://www.googleapis.com/auth/gmail.send": "Gmail (send)",
    "https://www.googleapis.com/auth/cloud-platform": "Vertex AI / Gemini",
    "openid": "OpenID",
    "email": "Email address",
}


def _label_scopes(granted: list[str]) -> list[dict[str, str]]:
    """Turn a raw scope list into a UI-friendly [{scope, label}] list."""
    out: list[dict[str, str]] = []
    for s in granted:
        out.append({"scope": s, "label": _GMAIL_SCOPE_LABELS.get(s, s)})
    return out


def _save_google_oauth_config(
    *,
    client_id: str | None = None,
    client_secret: str | None = None,
    project_id: str | None = None,
    location: str | None = None,
) -> None:
    """Persist Google OAuth credentials to ~/.captain-claw/config.yaml.

    Re-uses the same YAML-merge path as the settings REST handler so
    the behaviour is identical. Also auto-enables the section when both
    client_id and client_secret become present.
    """
    config_path = DEFAULT_CONFIG_PATH.expanduser()
    if config_path.exists():
        data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    else:
        data = {}

    section = data.setdefault("google_oauth", {})

    if client_id is not None:
        section["client_id"] = client_id
    if client_secret is not None:
        section["client_secret"] = client_secret
    if project_id is not None:
        section["project_id"] = project_id
    if location is not None:
        section["location"] = location

    # Auto-enable once both halves of the credential are present.
    if section.get("client_id") and section.get("client_secret"):
        section["enabled"] = True

    # Validate by loading the merged config through Pydantic.
    Config(**data)

    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)

    # Reload the in-memory singleton.
    set_config(Config.load())


# ── endpoints ───────────────────────────────────────────────────────


async def auth_google_login(server: WebServer, request: web.Request) -> web.Response:
    """Start the Google OAuth2 authorization flow."""
    cfg = get_config()
    oauth = cfg.google_oauth
    if not oauth.client_id or not oauth.client_secret:
        return _json_cors(
            request,
            {"error": "Google OAuth not configured — set client_id and client_secret first."},
            status=400,
        )

    # Purge stale PKCE states (older than 10 minutes).
    cutoff = time.time() - 600
    server._pending_oauth = {
        k: v for k, v in server._pending_oauth.items()
        if v.get("ts", 0) > cutoff
    }

    state = secrets.token_urlsafe(32)
    verifier, challenge = generate_pkce_pair()
    server._pending_oauth[state] = {
        "verifier": verifier,
        "ts": time.time(),
    }

    redirect_uri = (
        f"http://localhost:{cfg.web.port}/auth/google/callback"
    )
    merged_scopes = list(set(DEFAULT_SCOPES) | set(oauth.scopes or []))

    auth_url = build_authorization_url(
        client_id=oauth.client_id,
        redirect_uri=redirect_uri,
        scopes=merged_scopes,
        state=state,
        code_challenge=challenge,
    )
    raise web.HTTPFound(auth_url)


async def auth_google_callback(server: WebServer, request: web.Request) -> web.Response:
    """Handle the OAuth2 callback from Google.

    Rendered HTML is "popup-aware": it tries to postMessage the result
    back to ``window.opener`` and close itself. If there is no opener
    (regular tab flow) it falls back to a meta-refresh to ``/``.
    """
    error = request.query.get("error")
    if error:
        desc = request.query.get("error_description", error)
        return _callback_html(
            ok=False,
            title="OAuth error",
            detail=desc,
        )

    code = request.query.get("code", "")
    state = request.query.get("state", "")
    if not code or not state:
        return _callback_html(ok=False, title="Missing code or state")

    pending = server._pending_oauth.pop(state, None)
    if not pending:
        return _callback_html(ok=False, title="Invalid or expired state, please try again.")

    cfg = get_config()
    oauth = cfg.google_oauth
    redirect_uri = f"http://localhost:{cfg.web.port}/auth/google/callback"

    try:
        tokens = await exchange_code_for_tokens(
            code=code,
            client_id=oauth.client_id,
            client_secret=oauth.client_secret,
            redirect_uri=redirect_uri,
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

    if server._oauth_manager:
        await server._oauth_manager.store_tokens(tokens)
        if user:
            await server._oauth_manager.store_user_info(user)
        await inject_oauth_into_provider(server)

    return _callback_html(
        ok=True,
        title="Connected",
        detail=f"Signed in as {user.get('email', 'your Google account')}",
        email=user.get("email", ""),
    )


def _callback_html(
    *,
    ok: bool,
    title: str,
    detail: str = "",
    email: str = "",
) -> web.Response:
    """Return a popup-aware HTML page for the OAuth callback."""
    # Escape for inline JS string literals.
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
  // Regular tab flow — bounce back home after a moment.
  setTimeout(function(){{ window.location.href = '/'; }}, 1500);
}})();
</script>
</body>
</html>"""
    return web.Response(text=html, content_type="text/html")


async def auth_google_status(server: WebServer, request: web.Request) -> web.Response:
    """Return Google OAuth connection status as JSON.

    Response schema::

        {
          "configured": bool,     # client_id + client_secret present
          "enabled": bool,        # google_oauth.enabled
          "connected": bool,      # refresh_token available
          "user": {...} | null,
          "granted_scopes": [{"scope": "...", "label": "..."}]
        }
    """
    cfg = get_config()
    oauth = cfg.google_oauth
    configured = bool(oauth.client_id and oauth.client_secret)

    if not configured:
        return _json_cors(request, {
            "configured": False,
            "enabled": oauth.enabled,
            "connected": False,
            "user": None,
            "granted_scopes": [],
        })

    if not server._oauth_manager:
        return _json_cors(request, {
            "configured": True,
            "enabled": oauth.enabled,
            "connected": False,
            "user": None,
            "granted_scopes": [],
        })

    connected = await server._oauth_manager.is_connected()
    user = None
    scopes: list[str] = []
    if connected:
        user = await server._oauth_manager.get_user_info()
        tokens = await server._oauth_manager.get_tokens()
        if tokens and tokens.scope:
            scopes = tokens.scope.split()

    return _json_cors(request, {
        "configured": True,
        "enabled": oauth.enabled,
        "connected": connected,
        "user": user,
        "granted_scopes": _label_scopes(scopes),
    })


async def auth_google_logout(server: WebServer, request: web.Request) -> web.Response:
    """Revoke Google OAuth tokens and disconnect."""
    if server._oauth_manager:
        await server._oauth_manager.disconnect()
        clear_oauth_from_provider(server)
    return _json_cors(request, {"disconnected": True})


async def auth_google_config_get(
    server: WebServer, request: web.Request,
) -> web.Response:
    """Return the current Google OAuth configuration (secrets NOT included)."""
    cfg = get_config()
    oauth = cfg.google_oauth
    return _json_cors(request, {
        "client_id": oauth.client_id or "",
        "client_id_set": bool(oauth.client_id),
        "client_secret_set": bool(oauth.client_secret),
        "project_id": oauth.project_id or "",
        "location": oauth.location or "us-central1",
        "enabled": bool(oauth.enabled),
        "redirect_uri": f"http://localhost:{cfg.web.port}/auth/google/callback",
    })


async def auth_google_config_post(
    server: WebServer, request: web.Request,
) -> web.Response:
    """Save Google OAuth credentials from the UI.

    Body: ``{client_id?, client_secret?, project_id?, location?}``.
    Any field omitted is left unchanged.
    """
    try:
        body = await request.json()
    except Exception:
        return _json_cors(request, {"error": "Invalid JSON"}, status=400)

    client_id = body.get("client_id")
    client_secret = body.get("client_secret")
    project_id = body.get("project_id")
    location = body.get("location")

    # Trim surrounding whitespace and treat empty strings as "not provided"
    # so the UI can save only the fields it actually wants to change.
    def _clean(v: Any) -> str | None:
        if v is None:
            return None
        if not isinstance(v, str):
            return None
        s = v.strip()
        return s if s else None

    try:
        _save_google_oauth_config(
            client_id=_clean(client_id),
            client_secret=_clean(client_secret),
            project_id=_clean(project_id),
            location=_clean(location),
        )
    except Exception as exc:
        log.error("Google OAuth config save failed: %s", exc)
        return _json_cors(request, {"error": str(exc)}, status=422)

    return _json_cors(request, {"ok": True})


# ── provider wiring ─────────────────────────────────────────────────


async def inject_oauth_into_provider(server: WebServer) -> None:
    """Inject stored Google OAuth credentials into the Gemini provider."""
    if not server._oauth_manager:
        return
    creds = await server._oauth_manager.get_vertex_credentials()
    if not creds:
        return

    # In Flight Deck client mode the project/location come from Flight
    # Deck; otherwise they come from the local google_oauth config.
    project, location = await server._oauth_manager.get_vertex_project_location()
    if not project or not location:
        oauth = get_config().google_oauth
        project = project or oauth.project_id
        location = location or oauth.location

    from captain_claw.llm import LiteLLMProvider

    if server.agent and isinstance(server.agent.provider, LiteLLMProvider):
        if server.agent.provider.provider == "gemini":
            server.agent.provider.set_vertex_credentials(
                credentials=creds,
                project=project,
                location=location,
            )
            log.info("Google OAuth credentials injected into Gemini provider.")


def clear_oauth_from_provider(server: WebServer) -> None:
    """Remove vertex credentials from the active provider."""
    from captain_claw.llm import LiteLLMProvider

    if server.agent and isinstance(server.agent.provider, LiteLLMProvider):
        server.agent.provider.clear_vertex_credentials()
