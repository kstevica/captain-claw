"""Google OAuth flow handlers."""

from __future__ import annotations

import secrets
import time
from typing import TYPE_CHECKING, Any

from aiohttp import web

from captain_claw.config import get_config
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


async def auth_google_login(server: WebServer, request: web.Request) -> web.Response:
    """Start the Google OAuth2 authorization flow."""
    cfg = get_config()
    oauth = cfg.google_oauth
    if not oauth.enabled or not oauth.client_id:
        return web.json_response(
            {"error": "Google OAuth not configured"}, status=400
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
    # Merge required defaults with user-configured scopes so stale home
    # configs that lack newer scopes (calendar, gmail, etc.) never shrink
    # the scope set.  Order doesn't matter to Google; dedup via set.
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
    """Handle the OAuth2 callback from Google."""
    error = request.query.get("error")
    if error:
        desc = request.query.get("error_description", error)
        return web.Response(
            text=f"<html><body><h2>OAuth Error</h2><p>{desc}</p>"
                 f"<p><a href='/'>Back to home</a></p></body></html>",
            content_type="text/html",
        )

    code = request.query.get("code", "")
    state = request.query.get("state", "")
    if not code or not state:
        return web.Response(text="Missing code or state", status=400)

    pending = server._pending_oauth.pop(state, None)
    if not pending:
        return web.Response(
            text="<html><body><h2>Invalid or expired state</h2>"
                 "<p>Please try again.</p>"
                 "<p><a href='/'>Back to home</a></p></body></html>",
            content_type="text/html",
            status=400,
        )

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
        return web.Response(
            text=f"<html><body><h2>Token Exchange Failed</h2>"
                 f"<p>{exc}</p>"
                 f"<p><a href='/'>Back to home</a></p></body></html>",
            content_type="text/html",
            status=500,
        )

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

    email = user.get("email", "your Google account")
    return web.Response(
        text=(
            "<!DOCTYPE html><html><head>"
            "<meta charset='utf-8'>"
            "<meta http-equiv='refresh' content='2;url=/'>"
            "<title>Connected</title>"
            "<style>"
            "body{background:#0d1117;color:#e6edf3;font-family:sans-serif;"
            "display:flex;align-items:center;justify-content:center;"
            "min-height:100vh;margin:0;}"
            ".box{text-align:center;}"
            ".box h2{margin-bottom:8px;}"
            ".box p{color:#8b949e;}"
            "</style>"
            "</head><body>"
            f"<div class='box'><h2>Connected as {email}</h2>"
            "<p>Redirecting to home page...</p></div>"
            "</body></html>"
        ),
        content_type="text/html",
    )


async def auth_google_status(server: WebServer, request: web.Request) -> web.Response:
    """Return Google OAuth connection status as JSON."""
    cfg = get_config()
    if not cfg.google_oauth.enabled or not cfg.google_oauth.client_id:
        return web.json_response({"connected": False, "enabled": False})

    if not server._oauth_manager:
        return web.json_response({"connected": False, "enabled": True})

    connected = await server._oauth_manager.is_connected()
    user = None
    if connected:
        user = await server._oauth_manager.get_user_info()

    return web.json_response({
        "connected": connected,
        "enabled": True,
        "user": user,
    })


async def auth_google_logout(server: WebServer, request: web.Request) -> web.Response:
    """Revoke Google OAuth tokens and disconnect."""
    if server._oauth_manager:
        await server._oauth_manager.disconnect()
        clear_oauth_from_provider(server)
    return web.json_response({"disconnected": True})


async def inject_oauth_into_provider(server: WebServer) -> None:
    """Inject stored Google OAuth credentials into the Gemini provider."""
    if not server._oauth_manager:
        return
    creds = await server._oauth_manager.get_vertex_credentials()
    if not creds:
        return

    cfg = get_config()
    oauth = cfg.google_oauth

    from captain_claw.llm import LiteLLMProvider

    if server.agent and isinstance(server.agent.provider, LiteLLMProvider):
        if server.agent.provider.provider == "gemini":
            server.agent.provider.set_vertex_credentials(
                credentials=creds,
                project=oauth.project_id,
                location=oauth.location,
            )
            log.info("Google OAuth credentials injected into Gemini provider.")


def clear_oauth_from_provider(server: WebServer) -> None:
    """Remove vertex credentials from the active provider."""
    from captain_claw.llm import LiteLLMProvider

    if server.agent and isinstance(server.agent.provider, LiteLLMProvider):
        server.agent.provider.clear_vertex_credentials()
