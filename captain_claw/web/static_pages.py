"""Static page serving handlers for the web UI."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from aiohttp import web

if TYPE_CHECKING:
    from captain_claw.web_server import WebServer

STATIC_DIR = Path(__file__).resolve().parent / "static"


async def serve_home(server: WebServer, request: web.Request) -> web.FileResponse:
    return web.FileResponse(STATIC_DIR / "home.html")


async def serve_chat(server: WebServer, request: web.Request) -> web.FileResponse:
    return web.FileResponse(STATIC_DIR / "index.html")


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
