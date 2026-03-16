"""Static page serving handlers for the web UI."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from aiohttp import web

if TYPE_CHECKING:
    from captain_claw.web_server import WebServer

STATIC_DIR = Path(__file__).resolve().parent / "static"


def _cache_bust(html_path: Path) -> web.Response:
    """Read an HTML file and append cache-busting query params to static assets."""
    text = html_path.read_text(encoding="utf-8")
    # Use mtime of app.js / style.css as cache buster.
    for fname in ("app.js", "style.css"):
        asset = STATIC_DIR / fname
        if asset.is_file():
            v = int(asset.stat().st_mtime)
            text = text.replace(f"/static/{fname}", f"/static/{fname}?v={v}")
    return web.Response(text=text, content_type="text/html")


async def serve_home(server: WebServer, request: web.Request) -> web.Response:
    return _cache_bust(STATIC_DIR / "home.html")


async def serve_chat(server: WebServer, request: web.Request) -> web.Response:
    return _cache_bust(STATIC_DIR / "index.html")


async def serve_favicon(server: WebServer, request: web.Request) -> web.Response:
    favicon = STATIC_DIR / "favicon.svg"
    if favicon.is_file():
        return web.FileResponse(favicon)
    return web.Response(status=204)


async def serve_orchestrator(server: WebServer, request: web.Request) -> web.FileResponse:
    return web.FileResponse(STATIC_DIR / "orchestrator.html")


async def serve_instructions(server: WebServer, request: web.Request) -> web.FileResponse:
    return web.FileResponse(STATIC_DIR / "instructions.html")


async def serve_cron(server: WebServer, request: web.Request) -> web.FileResponse:
    return web.FileResponse(STATIC_DIR / "cron.html")


async def serve_workflows(server: WebServer, request: web.Request) -> web.FileResponse:
    return web.FileResponse(STATIC_DIR / "workflows.html")


async def serve_loop_runner(server: WebServer, request: web.Request) -> web.FileResponse:
    return web.FileResponse(STATIC_DIR / "loop-runner.html")


async def serve_memory(server: WebServer, request: web.Request) -> web.FileResponse:
    return web.FileResponse(STATIC_DIR / "memory.html")


async def serve_deep_memory(server: WebServer, request: web.Request) -> web.FileResponse:
    return web.FileResponse(STATIC_DIR / "deep-memory.html")


async def serve_settings(server: WebServer, request: web.Request) -> web.FileResponse:
    return web.FileResponse(STATIC_DIR / "settings.html")


async def serve_sessions(server: WebServer, request: web.Request) -> web.FileResponse:
    return web.FileResponse(STATIC_DIR / "sessions.html")


async def serve_files(server: WebServer, request: web.Request) -> web.FileResponse:
    return web.FileResponse(STATIC_DIR / "files.html")


async def serve_onboarding(server: WebServer, request: web.Request) -> web.FileResponse:
    return web.FileResponse(STATIC_DIR / "onboarding.html")


async def serve_datastore(server: WebServer, request: web.Request) -> web.FileResponse:
    return web.FileResponse(STATIC_DIR / "datastore.html")


async def serve_playbooks(server: WebServer, request: web.Request) -> web.FileResponse:
    return web.FileResponse(STATIC_DIR / "playbooks.html")


async def serve_browser_workflows(server: WebServer, request: web.Request) -> web.FileResponse:
    return web.FileResponse(STATIC_DIR / "browser-workflows.html")


async def serve_direct_api_calls(server: WebServer, request: web.Request) -> web.FileResponse:
    return web.FileResponse(STATIC_DIR / "direct-api-calls.html")


async def serve_skills(server: WebServer, request: web.Request) -> web.FileResponse:
    return web.FileResponse(STATIC_DIR / "skills.html")


async def serve_usage(server: WebServer, request: web.Request) -> web.FileResponse:
    return web.FileResponse(STATIC_DIR / "usage.html")


async def serve_reflections(server: WebServer, request: web.Request) -> web.FileResponse:
    return web.FileResponse(STATIC_DIR / "reflections.html")


async def serve_computer(server: WebServer, request: web.Request) -> web.FileResponse | web.Response:
    from captain_claw.config import get_config
    cfg = get_config()
    if cfg.web.public_run:
        # In public mode, require a valid public session cookie.
        from captain_claw.web.public_auth import _is_admin
        if not _is_admin(request, cfg.web):
            from captain_claw.web.public_session import read_public_cookie
            identity = read_public_cookie(request, cfg.web.auth_token)
            if identity is None:
                # Return landing page with no-cache so that redirects
                # from the enter endpoint always re-check the cookie.
                landing = STATIC_DIR / "public_landing.html"
                return web.Response(
                    body=landing.read_bytes(),
                    content_type="text/html",
                    headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
                )
    return web.FileResponse(STATIC_DIR / "computer.html")


async def serve_personality(server: WebServer, request: web.Request) -> web.FileResponse:
    return web.FileResponse(STATIC_DIR / "personality.html")


async def serve_semantic_memory(server: WebServer, request: web.Request) -> web.FileResponse:
    return web.FileResponse(STATIC_DIR / "semantic-memory.html")
