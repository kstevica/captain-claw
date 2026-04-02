"""Token-based authentication middleware for the web UI."""

from __future__ import annotations

import hashlib
import hmac
import time
from typing import TYPE_CHECKING, Callable
from urllib.parse import urlencode, urlparse, parse_qs, urlunparse

from aiohttp import web

if TYPE_CHECKING:
    from captain_claw.config import WebConfig

COOKIE_NAME = "claw_session"


def _make_cookie_value(auth_token: str) -> str:
    """Create a signed cookie value: ``timestamp:hmac_hex``."""
    ts = str(int(time.time()))
    sig = hmac.new(
        auth_token.encode(), ts.encode(), hashlib.sha256
    ).hexdigest()
    return f"{ts}:{sig}"


def _validate_cookie(value: str, auth_token: str, max_age_days: int) -> bool:
    """Return *True* if the cookie value is a valid, non-expired HMAC."""
    parts = value.split(":", 1)
    if len(parts) != 2:
        return False
    ts_str, sig = parts
    try:
        ts = int(ts_str)
    except ValueError:
        return False
    # Check expiry
    age_seconds = time.time() - ts
    if age_seconds < 0 or age_seconds > max_age_days * 86400:
        return False
    # Verify HMAC
    expected = hmac.new(
        auth_token.encode(), ts_str.encode(), hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(sig, expected)


def _is_behind_tls(request: web.Request) -> bool:
    """Detect whether the request arrived over TLS (via reverse proxy)."""
    return request.headers.get("X-Forwarded-Proto", "").lower() == "https"


def _strip_token_param(url: str) -> str:
    """Return *url* with the ``token`` query parameter removed."""
    parsed = urlparse(url)
    params = parse_qs(parsed.query, keep_blank_values=True)
    params.pop("token", None)
    new_query = urlencode(params, doseq=True)
    return urlunparse(parsed._replace(query=new_query))


_UNAUTHORIZED_HTML = """\
<!doctype html>
<html>
<head><title>Unauthorized</title>
<style>
  body { font-family: system-ui, sans-serif; display: flex;
         justify-content: center; align-items: center; height: 100vh;
         margin: 0; background: #1a1a2e; color: #e0e0e0; }
  .box { text-align: center; }
  h1 { font-size: 1.5rem; margin-bottom: .5rem; }
  p  { color: #999; }
</style>
</head>
<body><div class="box">
  <h1>Unauthorized</h1>
  <p>A valid access token is required.</p>
</div></body>
</html>
"""


def create_auth_middleware(config: WebConfig) -> Callable:
    """Return an aiohttp middleware that enforces token-based auth.

    When ``config.auth_token`` is non-empty every request must either:
    * carry a valid ``claw_session`` cookie, **or**
    * include ``?token=<secret>`` which will set the cookie and redirect.
    """
    auth_token = config.auth_token
    max_age_days = config.auth_cookie_max_age
    max_age_seconds = max_age_days * 86400

    @web.middleware
    async def middleware(
        request: web.Request,
        handler: Callable,
    ) -> web.StreamResponse:
        # ── 1. Already authenticated via cookie? ─────────────────────
        cookie = request.cookies.get(COOKIE_NAME, "")
        if cookie and _validate_cookie(cookie, auth_token, max_age_days):
            return await handler(request)

        # ── 2. Token in query string? → set cookie + redirect ────────
        token_param = request.query.get("token", "")
        if token_param and hmac.compare_digest(token_param, auth_token):
            # WebSocket and API requests can't follow redirects — pass through directly.
            if request.path == "/ws" or request.headers.get("Upgrade", "").lower() == "websocket" or request.path.startswith("/api/"):
                return await handler(request)

            cookie_val = _make_cookie_value(auth_token)
            redirect_url = _strip_token_param(str(request.url))
            # For relative redirect, keep only path + remaining query
            parsed = urlparse(redirect_url)
            location = parsed.path or "/"
            if parsed.query:
                location += f"?{parsed.query}"

            resp = web.HTTPFound(location=location)
            secure = _is_behind_tls(request)
            resp.set_cookie(
                COOKIE_NAME,
                cookie_val,
                max_age=max_age_seconds,
                httponly=True,
                samesite="Lax",
                path="/",
                secure=secure,
            )
            return resp

        # ── 3. Unauthorized ──────────────────────────────────────────
        # Return JSON for API / WebSocket requests, HTML for browsers
        accept = request.headers.get("Accept", "")
        if (
            request.path.startswith("/api/")
            or request.path == "/ws"
            or "application/json" in accept
        ):
            return web.json_response(
                {"error": "unauthorized", "message": "Valid access token required"},
                status=401,
            )
        return web.Response(
            text=_UNAUTHORIZED_HTML,
            content_type="text/html",
            status=401,
        )

    return middleware
