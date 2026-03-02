"""BotPort dashboard REST API and static file serving."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from aiohttp import web

if TYPE_CHECKING:
    from botport.server import BotPortServer

STATIC_DIR = Path(__file__).parent / "static"


def setup_dashboard_routes(app: web.Application, server: BotPortServer) -> None:
    """Register dashboard routes on the aiohttp application."""

    async def index(request: web.Request) -> web.Response:
        index_path = STATIC_DIR / "index.html"
        if index_path.is_file():
            return web.FileResponse(index_path)
        return web.Response(text="Dashboard not found", status=404)

    async def dashboard_css(request: web.Request) -> web.Response:
        css_path = STATIC_DIR / "dashboard.css"
        if css_path.is_file():
            return web.FileResponse(css_path, headers={"Content-Type": "text/css"})
        return web.Response(text="", content_type="text/css")

    async def dashboard_js(request: web.Request) -> web.Response:
        js_path = STATIC_DIR / "dashboard.js"
        if js_path.is_file():
            return web.FileResponse(js_path, headers={"Content-Type": "application/javascript"})
        return web.Response(text="", content_type="application/javascript")

    async def api_instances(request: web.Request) -> web.Response:
        instances = server.connections.list_instances()
        return web.json_response([i.to_dict() for i in instances])

    async def api_concerns(request: web.Request) -> web.Response:
        active_only = request.query.get("active", "").lower() in ("1", "true", "yes")
        if active_only:
            concerns = server.concerns.get_active_concerns()
        else:
            concerns = await server.store.list_concerns(limit=100)
        return web.json_response([c.to_dict() for c in concerns])

    async def api_concern_detail(request: web.Request) -> web.Response:
        concern_id = request.match_info.get("id", "")
        concern = await server.store.load_concern(concern_id)
        if concern is None:
            return web.json_response({"error": "Not found"}, status=404)
        return web.json_response(concern.to_dict())

    async def api_stats(request: web.Request) -> web.Response:
        stats = await server.concerns.get_stats()
        stats["connected_instances"] = server.connections.connected_count
        stats["botport_version"] = server.config.__class__.__module__.split(".")[0]
        return web.json_response(stats)

    async def api_registry(request: web.Request) -> web.Response:
        summary = server.registry.get_summary()
        return web.json_response(summary)

    # Routes.
    app.router.add_get("/", index)
    app.router.add_get("/dashboard.css", dashboard_css)
    app.router.add_get("/dashboard.js", dashboard_js)
    app.router.add_get("/api/instances", api_instances)
    app.router.add_get("/api/concerns", api_concerns)
    app.router.add_get("/api/concerns/{id}", api_concern_detail)
    app.router.add_get("/api/stats", api_stats)
    app.router.add_get("/api/registry", api_registry)
