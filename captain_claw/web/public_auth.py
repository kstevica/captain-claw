"""Route-lockdown middleware for public-run mode.

When ``config.web.public_run`` is set (e.g. ``"computer"``), only the routes
belonging to that section are accessible to anonymous visitors.  Admin users
authenticated via ``auth_token`` bypass the lockdown entirely.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from aiohttp import web

from captain_claw.web.auth import COOKIE_NAME as ADMIN_COOKIE, _validate_cookie as _validate_admin_cookie

if TYPE_CHECKING:
    from captain_claw.config import WebConfig

# Routes (prefixes) allowed per public section.
_SECTION_ROUTES: dict[str, dict[str, list[str]]] = {
    "computer": {
        "pages": ["/computer", "/brain-graph"],
        "api": [
            "/api/computer/",
            "/api/public/",
            "/api/config",
            "/api/orchestrator/models",
            "/api/user-personalities",
            "/api/file/upload",
            "/api/image/upload",
            "/api/files",
            "/api/media",
            "/api/todos",
            "/api/datastore/",
            "/api/insights",
            "/api/nervous-system",
            "/api/sister/",
            "/api/briefings",
            "/api/brain-graph",
        ],
        "ws": ["/ws"],
        "static": ["/static/", "/favicon.ico"],
    },
}

_FORBIDDEN_HTML = """\
<!doctype html>
<html>
<head><title>Forbidden</title>
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
  <h1>Access Denied</h1>
  <p>This instance is running in public mode. Only the allowed section is accessible.</p>
</div></body>
</html>
"""


def _is_admin(request: web.Request, config: "WebConfig") -> bool:
    """Return True if the request carries a valid admin auth_token cookie."""
    if not config.auth_token:
        return False
    cookie = request.cookies.get(ADMIN_COOKIE, "")
    if cookie and _validate_admin_cookie(cookie, config.auth_token, config.auth_cookie_max_age):
        return True
    # Also check query param for initial admin login.
    import hmac as _hmac
    token_param = request.query.get("token", "")
    if token_param and _hmac.compare_digest(token_param, config.auth_token):
        return True
    return False


def _is_allowed_path(path: str, section: str) -> bool:
    """Return True if *path* is allowed for the given public section."""
    routes = _SECTION_ROUTES.get(section)
    if not routes:
        return False
    # Landing page (root redirects to section)
    if path == "/":
        return True
    for prefix in routes.get("pages", []):
        if path == prefix or path.startswith(prefix + "/"):
            return True
    for prefix in routes.get("api", []):
        if path.startswith(prefix):
            return True
    for prefix in routes.get("ws", []):
        if path == prefix:
            return True
    for prefix in routes.get("static", []):
        if path.startswith(prefix):
            return True
    return False


def get_request_session_id(request: web.Request) -> tuple[bool, str | None]:
    """Return ``(is_public, session_id)`` for the current request.

    * Admin users → ``(False, None)`` — no session isolation needed.
    * Public users → ``(True, "<session_id>")`` from their cookie.
    * Non-public mode → ``(False, None)``.
    """
    from captain_claw.config import get_config
    cfg = get_config()
    if not cfg.web.public_run:
        return False, None
    if _is_admin(request, cfg.web):
        return False, None
    # Public user — extract session from cookie.
    from captain_claw.web.public_session import read_public_cookie
    identity = read_public_cookie(request, cfg.web.auth_token)
    if identity is None:
        return True, None  # Public but no valid session
    return True, identity[0]


def create_public_middleware(config: "WebConfig") -> Callable:
    """Return middleware that enforces public-run route lockdown."""
    section = config.public_run.strip().lower()

    @web.middleware
    async def middleware(
        request: web.Request,
        handler: Callable,
    ) -> web.StreamResponse:
        # Admin bypass
        if _is_admin(request, config):
            return await handler(request)

        path = request.path

        if not _is_allowed_path(path, section):
            accept = request.headers.get("Accept", "")
            if (
                path.startswith("/api/")
                or path == "/ws"
                or "application/json" in accept
            ):
                return web.json_response(
                    {"error": "forbidden", "message": "This endpoint is not available in public mode"},
                    status=403,
                )
            return web.Response(text=_FORBIDDEN_HTML, content_type="text/html", status=403)

        return await handler(request)

    return middleware
