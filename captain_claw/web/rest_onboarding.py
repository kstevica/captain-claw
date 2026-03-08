"""REST endpoints for the web onboarding wizard."""

from __future__ import annotations

import json
from pathlib import Path
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

_CODEX_AUTH_PATH = Path("~/.codex/auth.json").expanduser()


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


async def get_codex_auth(server: WebServer, request: web.Request) -> web.Response:
    """GET /api/onboarding/codex-auth — read OpenAI OAuth tokens from ~/.codex/auth.json."""
    if not _CODEX_AUTH_PATH.exists():
        return web.json_response({"ok": False, "error": "~/.codex/auth.json not found. Install and authenticate with Codex CLI first."})

    try:
        data = json.loads(_CODEX_AUTH_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        return web.json_response({"ok": False, "error": f"Failed to read auth.json: {exc}"})

    tokens = data.get("tokens") or {}
    access_token = tokens.get("access_token", "")
    account_id = tokens.get("account_id", "")

    if not access_token:
        return web.json_response({"ok": False, "error": "No access_token found in ~/.codex/auth.json"})

    return web.json_response({
        "ok": True,
        "access_token": access_token,
        "account_id": account_id,
    })


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
