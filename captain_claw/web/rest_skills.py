"""REST handlers for skills browsing and installation."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml
from aiohttp import web

from captain_claw.logging import get_logger

if TYPE_CHECKING:
    from captain_claw.web_server import WebServer

log = get_logger(__name__)


def _resolve_skill_key(entry: Any) -> str:
    """Return the config key for a skill entry."""
    if entry.metadata and entry.metadata.skill_key:
        return entry.metadata.skill_key
    return entry.name


def _get_enabled_state(cfg: Any, skill_key: str) -> bool | None:
    """Return the explicit enabled state from config, or None if not set."""
    entries = cfg.skills.entries or {}
    for key, val in entries.items():
        if key == skill_key or str(key).strip().lower() == skill_key.lower():
            return val.enabled
    return None


async def list_skills(server: WebServer, request: web.Request) -> web.Response:
    """GET /api/skills — list all workspace skill entries with rich metadata."""
    from captain_claw.config import get_config
    from captain_claw.skills import filter_skill_entries, load_workspace_skill_entries

    cfg = get_config()
    workspace = str(server.agent.workspace_base_path) if server.agent else "."

    try:
        entries = load_workspace_skill_entries(workspace, cfg)
    except Exception as exc:
        log.error("Failed to load skill entries", error=str(exc))
        return web.json_response({"skills": [], "error": str(exc)})

    try:
        filtered = filter_skill_entries(entries, cfg)
    except Exception:
        filtered = entries

    filtered_names = {e.name for e in filtered}

    result = []
    for entry in entries:
        meta = entry.metadata
        requires = meta.requires if meta else None
        skill_key = _resolve_skill_key(entry)
        enabled_state = _get_enabled_state(cfg, skill_key)
        result.append({
            "name": entry.name,
            "description": entry.description,
            "source": entry.source,
            "file_path": entry.file_path,
            "base_dir": entry.base_dir,
            "emoji": meta.emoji if meta else None,
            "homepage": meta.homepage if meta else None,
            "user_invocable": entry.invocation.user_invocable,
            "model_invocation": not entry.invocation.disable_model_invocation,
            "active": entry.name in filtered_names,
            "enabled": enabled_state,
            "skill_key": skill_key,
            "requires": {
                "bins": requires.bins if requires else [],
                "any_bins": requires.any_bins if requires else [],
                "env": requires.env if requires else [],
                "config": requires.config if requires else [],
            },
            "has_install": bool(meta and meta.install),
        })

    return web.json_response({"skills": result})


async def install_skill(server: WebServer, request: web.Request) -> web.Response:
    """POST /api/skills/install — install a skill from a GitHub URL."""
    from captain_claw.config import get_config
    from captain_claw.skills import install_skill_from_github_url

    try:
        body = await request.json()
    except Exception:
        return web.json_response({"ok": False, "error": "Invalid JSON"}, status=400)

    url = str(body.get("url", "")).strip()
    if not url:
        return web.json_response({"ok": False, "error": "Missing URL"}, status=400)

    cfg = get_config()

    try:
        result = install_skill_from_github_url(url, cfg)
    except Exception as exc:
        log.error("Skill install failed", url=url, error=str(exc))
        return web.json_response({"ok": False, "error": str(exc)}, status=500)

    return web.json_response({
        "ok": True,
        "skill_name": result.skill_name,
        "destination": result.destination,
        "repo": result.repo,
    })


async def toggle_skill(server: WebServer, request: web.Request) -> web.Response:
    """POST /api/skills/toggle — enable or disable a skill via config."""
    from captain_claw.config import (
        DEFAULT_CONFIG_PATH,
        LOCAL_CONFIG_FILENAME,
        Config,
        set_config,
    )

    try:
        body = await request.json()
    except Exception:
        return web.json_response({"ok": False, "error": "Invalid JSON"}, status=400)

    skill_key = str(body.get("skill_key", "")).strip()
    if not skill_key:
        return web.json_response({"ok": False, "error": "Missing skill_key"}, status=400)

    enabled = body.get("enabled")
    if enabled is None:
        return web.json_response({"ok": False, "error": "Missing enabled"}, status=400)
    enabled = bool(enabled)

    # ── Read existing YAML ────────────────────────────────
    config_path = DEFAULT_CONFIG_PATH
    if config_path.exists():
        raw = config_path.read_text(encoding="utf-8")
        data: dict = yaml.safe_load(raw) or {}
    else:
        data = {}

    # ── Update skills.entries.<key>.enabled ────────────────
    skills_section = data.setdefault("skills", {})
    entries_section = skills_section.setdefault("entries", {})
    entry = entries_section.setdefault(skill_key, {})

    if enabled:
        # Re-enable: remove the enabled key entirely (None = default enabled)
        entry.pop("enabled", None)
        # Clean up empty entry
        if not entry:
            entries_section.pop(skill_key, None)
        # Clean up empty entries dict
        if not entries_section:
            skills_section.pop("entries", None)
    else:
        entry["enabled"] = False

    # ── Validate ──────────────────────────────────────────
    local_path = Path.cwd() / LOCAL_CONFIG_FILENAME
    local_data = Config._read_yaml_data(local_path)
    merged_data = Config._deep_merge(local_data, data) if local_data else data
    try:
        Config(**merged_data)
    except Exception as e:
        return web.json_response({"ok": False, "error": str(e)}, status=422)

    # ── Save and reload ───────────────────────────────────
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)
    set_config(Config.load())

    log.info("Skill toggled", skill_key=skill_key, enabled=enabled)
    return web.json_response({"ok": True, "skill_key": skill_key, "enabled": enabled})
