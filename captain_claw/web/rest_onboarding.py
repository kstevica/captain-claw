"""REST endpoints for the web onboarding wizard."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from aiohttp import web

from captain_claw.config import Config, get_config, set_config
from captain_claw.onboarding import (
    save_onboarding_config,
    should_run_onboarding,
    validate_provider_connection,
)

if TYPE_CHECKING:
    from captain_claw.web_server import WebServer


async def get_onboarding_status(server: WebServer, request: web.Request) -> web.Response:
    """GET /api/onboarding/status — check if onboarding is needed."""
    needed = should_run_onboarding()
    return web.json_response({"needed": needed})


async def post_onboarding_validate(server: WebServer, request: web.Request) -> web.Response:
    """POST /api/onboarding/validate — test a provider connection."""
    try:
        body: dict[str, Any] = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON"}, status=400)

    provider = body.get("provider", "")
    model = body.get("model", "")
    api_key = body.get("api_key", "")
    base_url = body.get("base_url", "")

    if not provider or not model:
        return web.json_response({"ok": False, "error": "Provider and model are required."})

    ok, error = await validate_provider_connection(
        provider=provider,
        model=model,
        api_key=api_key,
        base_url=base_url,
    )
    return web.json_response({"ok": ok, "error": error})


async def post_onboarding_save(server: WebServer, request: web.Request) -> web.Response:
    """POST /api/onboarding/save — save config and mark onboarding completed."""
    try:
        body: dict[str, Any] = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON"}, status=400)

    try:
        config_path = save_onboarding_config(values=body)
    except Exception as exc:
        return web.json_response({"ok": False, "error": str(exc)}, status=500)

    # Hot-reload the global config singleton.
    set_config(Config.load())

    return web.json_response({"ok": True, "config_path": str(config_path)})
