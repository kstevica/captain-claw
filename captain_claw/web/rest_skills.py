"""REST handlers for skills browsing and installation."""

from __future__ import annotations

import platform
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml
from aiohttp import web

from captain_claw.logging import get_logger

if TYPE_CHECKING:
    from captain_claw.web_server import WebServer

log = get_logger(__name__)

# ── Blocked system paths ─────────────────────────────────────

_BLOCKED_UNIX = {
    "/bin", "/sbin", "/usr", "/etc", "/var", "/tmp", "/dev", "/proc",
    "/sys", "/boot", "/lib", "/lib64", "/run", "/snap", "/lost+found",
    "/System", "/Library", "/private",
}
_BLOCKED_WIN = {
    "C:\\Windows", "C:\\Program Files", "C:\\Program Files (x86)",
    "C:\\ProgramData", "C:\\$Recycle.Bin", "C:\\Recovery",
}


def _is_drive_root(p: Path) -> bool:
    """Return True if path is a filesystem root (/ or C:\\)."""
    return p == p.anchor or str(p) in ("/", "\\")


def _is_blocked_path(p: Path) -> bool:
    """Return True if path is a drive root or a known system directory."""
    resolved = p.resolve()
    if _is_drive_root(resolved):
        return True
    s = str(resolved)
    blocked = _BLOCKED_WIN if platform.system() == "Windows" else _BLOCKED_UNIX
    for bp in blocked:
        if s == bp or s.startswith(bp + ("/" if "/" in bp else "\\")):
            return True
    return False


# ── Helpers ───────────────────────────────────────────────────

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


def _save_config(config_path: Path, data: dict) -> None:
    """Validate, write, and reload config."""
    from captain_claw.config import Config, LOCAL_CONFIG_FILENAME, set_config

    local_path = Path.cwd() / LOCAL_CONFIG_FILENAME
    local_data = Config._read_yaml_data(local_path)
    merged_data = Config._deep_merge(local_data, data) if local_data else data
    Config(**merged_data)  # validate — raises on error

    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)
    set_config(Config.load())


# ── Skills CRUD ───────────────────────────────────────────────

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
    from captain_claw.config import DEFAULT_CONFIG_PATH

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

    config_path = DEFAULT_CONFIG_PATH
    if config_path.exists():
        data: dict = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    else:
        data = {}

    skills_section = data.setdefault("skills", {})
    entries_section = skills_section.setdefault("entries", {})
    entry = entries_section.setdefault(skill_key, {})

    if enabled:
        entry.pop("enabled", None)
        if not entry:
            entries_section.pop(skill_key, None)
        if not entries_section:
            skills_section.pop("entries", None)
    else:
        entry["enabled"] = False

    try:
        _save_config(config_path, data)
    except Exception as e:
        return web.json_response({"ok": False, "error": str(e)}, status=422)

    log.info("Skill toggled", skill_key=skill_key, enabled=enabled)
    return web.json_response({"ok": True, "skill_key": skill_key, "enabled": enabled})


# ── Directory browsing ────────────────────────────────────────

async def browse_directory(server: WebServer, request: web.Request) -> web.Response:
    """GET /api/browse?path=... — list subdirectories for folder selection."""
    raw_path = request.query.get("path", "").strip()
    if not raw_path:
        raw_path = str(Path.home())

    try:
        target = Path(raw_path).expanduser().resolve()
    except Exception:
        return web.json_response({"error": "Invalid path"}, status=400)

    if not target.exists():
        return web.json_response({"error": "Path does not exist"}, status=404)
    if not target.is_dir():
        return web.json_response({"error": "Not a directory"}, status=400)

    dirs = []
    try:
        for child in sorted(target.iterdir()):
            if not child.is_dir():
                continue
            name = child.name
            # Skip hidden directories.
            if name.startswith("."):
                continue
            dirs.append(name)
    except PermissionError:
        return web.json_response({"error": "Permission denied"}, status=403)

    blocked = _is_blocked_path(target)

    return web.json_response({
        "path": str(target),
        "parent": str(target.parent) if not _is_drive_root(target) else None,
        "dirs": dirs,
        "blocked": blocked,
    })


# ── Read-folder management ────────────────────────────────────

async def list_read_folders(server: WebServer, request: web.Request) -> web.Response:
    """GET /api/read-folders — list configured extra readable directories."""
    from captain_claw.config import get_config

    cfg = get_config()
    dirs = list(cfg.tools.read.extra_dirs)
    resolved = []
    for d in dirs:
        p = Path(d).expanduser().resolve()
        resolved.append({
            "path": d,
            "resolved": str(p),
            "exists": p.exists(),
        })
    return web.json_response({"dirs": resolved})


async def add_read_folder(server: WebServer, request: web.Request) -> web.Response:
    """POST /api/read-folders — add a readable directory."""
    from captain_claw.config import DEFAULT_CONFIG_PATH

    try:
        body = await request.json()
    except Exception:
        return web.json_response({"ok": False, "error": "Invalid JSON"}, status=400)

    raw = str(body.get("path", "")).strip()
    if not raw:
        return web.json_response({"ok": False, "error": "Missing path"}, status=400)

    target = Path(raw).expanduser().resolve()

    if _is_blocked_path(target):
        return web.json_response(
            {"ok": False, "error": "Cannot add root or system directories"},
            status=400,
        )
    if not target.exists():
        return web.json_response({"ok": False, "error": "Directory does not exist"}, status=400)
    if not target.is_dir():
        return web.json_response({"ok": False, "error": "Path is not a directory"}, status=400)

    config_path = DEFAULT_CONFIG_PATH
    if config_path.exists():
        data: dict = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    else:
        data = {}

    read_section = data.setdefault("tools", {}).setdefault("read", {})
    extra = read_section.setdefault("extra_dirs", [])

    # Avoid duplicates (compare resolved paths).
    existing_resolved = {str(Path(e).expanduser().resolve()) for e in extra}
    if str(target) in existing_resolved:
        return web.json_response({"ok": False, "error": "Directory already added"}, status=409)

    extra.append(str(target))

    try:
        _save_config(config_path, data)
    except Exception as e:
        return web.json_response({"ok": False, "error": str(e)}, status=422)

    log.info("Read folder added", path=str(target))
    return web.json_response({"ok": True, "path": str(target)})


async def remove_read_folder(server: WebServer, request: web.Request) -> web.Response:
    """DELETE /api/read-folders — remove a readable directory."""
    from captain_claw.config import DEFAULT_CONFIG_PATH

    try:
        body = await request.json()
    except Exception:
        return web.json_response({"ok": False, "error": "Invalid JSON"}, status=400)

    raw = str(body.get("path", "")).strip()
    if not raw:
        return web.json_response({"ok": False, "error": "Missing path"}, status=400)

    target = Path(raw).expanduser().resolve()

    config_path = DEFAULT_CONFIG_PATH
    if config_path.exists():
        data: dict = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    else:
        data = {}

    read_section = data.get("tools", {}).get("read", {})
    extra = read_section.get("extra_dirs", [])
    if not extra:
        return web.json_response({"ok": False, "error": "No read folders configured"}, status=404)

    new_extra = []
    removed = False
    for e in extra:
        if str(Path(e).expanduser().resolve()) == str(target):
            removed = True
        else:
            new_extra.append(e)

    if not removed:
        return web.json_response({"ok": False, "error": "Directory not found in list"}, status=404)

    read_section["extra_dirs"] = new_extra
    if not new_extra:
        read_section.pop("extra_dirs", None)
    if not read_section:
        data.get("tools", {}).pop("read", None)

    try:
        _save_config(config_path, data)
    except Exception as e:
        return web.json_response({"ok": False, "error": str(e)}, status=422)

    log.info("Read folder removed", path=str(target))
    return web.json_response({"ok": True, "path": str(target)})


# ── Drive enumeration (Windows) ─────────────────────────────

async def list_drives(server: WebServer, request: web.Request) -> web.Response:
    """GET /api/drives — list available drive letters (Windows only)."""
    import string

    os_name = platform.system()
    if os_name != "Windows":
        return web.json_response({"drives": [], "os": os_name})

    drives = []
    for letter in string.ascii_uppercase:
        drive_path = Path(f"{letter}:\\")
        if drive_path.exists():
            drives.append(f"{letter}:")
    return web.json_response({"drives": drives, "os": "Windows"})


# ── GWS status ───────────────────────────────────────────────

async def gws_status(server: WebServer, request: web.Request) -> web.Response:
    """GET /api/gws-status — check whether the gws CLI is available."""
    from captain_claw.file_tree_builder import resolve_gws_binary

    binary = resolve_gws_binary()
    return web.json_response({"available": binary is not None})


# ── Google Drive folder management (via gws) ─────────────────

async def list_gdrive_folders(server: WebServer, request: web.Request) -> web.Response:
    """GET /api/read-folders/gdrive — list configured GDrive folders."""
    from captain_claw.config import get_config

    cfg = get_config()
    folders = [{"id": f.id, "name": f.name} for f in cfg.tools.read.gdrive_folders]
    return web.json_response({"folders": folders})


async def add_gdrive_folder(server: WebServer, request: web.Request) -> web.Response:
    """POST /api/read-folders/gdrive — add a GDrive folder."""
    from captain_claw.config import DEFAULT_CONFIG_PATH

    try:
        body = await request.json()
    except Exception:
        return web.json_response({"ok": False, "error": "Invalid JSON"}, status=400)

    folder_id = str(body.get("id", "")).strip()
    folder_name = str(body.get("name", "")).strip()
    if not folder_id or not folder_name:
        return web.json_response({"ok": False, "error": "Missing id or name"}, status=400)

    config_path = DEFAULT_CONFIG_PATH
    if config_path.exists():
        data: dict = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    else:
        data = {}

    read_section = data.setdefault("tools", {}).setdefault("read", {})
    gdrive = read_section.setdefault("gdrive_folders", [])

    # Avoid duplicates.
    for existing in gdrive:
        if existing.get("id") == folder_id:
            return web.json_response({"ok": False, "error": "Folder already added"}, status=409)

    gdrive.append({"id": folder_id, "name": folder_name})

    try:
        _save_config(config_path, data)
    except Exception as e:
        return web.json_response({"ok": False, "error": str(e)}, status=422)

    log.info("GDrive folder added", folder_id=folder_id, name=folder_name)
    return web.json_response({"ok": True, "id": folder_id, "name": folder_name})


async def remove_gdrive_folder(server: WebServer, request: web.Request) -> web.Response:
    """DELETE /api/read-folders/gdrive — remove a GDrive folder."""
    from captain_claw.config import DEFAULT_CONFIG_PATH

    try:
        body = await request.json()
    except Exception:
        return web.json_response({"ok": False, "error": "Invalid JSON"}, status=400)

    folder_id = str(body.get("id", "")).strip()
    if not folder_id:
        return web.json_response({"ok": False, "error": "Missing id"}, status=400)

    config_path = DEFAULT_CONFIG_PATH
    if config_path.exists():
        data: dict = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    else:
        data = {}

    read_section = data.get("tools", {}).get("read", {})
    gdrive = read_section.get("gdrive_folders", [])
    if not gdrive:
        return web.json_response({"ok": False, "error": "No GDrive folders configured"}, status=404)

    new_gdrive = [f for f in gdrive if f.get("id") != folder_id]
    if len(new_gdrive) == len(gdrive):
        return web.json_response({"ok": False, "error": "Folder not found in list"}, status=404)

    read_section["gdrive_folders"] = new_gdrive
    if not new_gdrive:
        read_section.pop("gdrive_folders", None)

    try:
        _save_config(config_path, data)
    except Exception as e:
        return web.json_response({"ok": False, "error": str(e)}, status=422)

    log.info("GDrive folder removed", folder_id=folder_id)
    return web.json_response({"ok": True, "id": folder_id})


async def browse_gdrive(server: WebServer, request: web.Request) -> web.Response:
    """GET /api/read-folders/gdrive/browse?folder_id=... — browse GDrive subfolders."""
    from captain_claw.file_tree_builder import browse_gdrive_folders

    folder_id = request.query.get("folder_id", "root").strip() or "root"
    result = await browse_gdrive_folders(folder_id)
    if result.get("error"):
        return web.json_response(result, status=502)
    return web.json_response(result)


# ── Folder trees ─────────────────────────────────────────────

async def get_folder_trees(server: WebServer, request: web.Request) -> web.Response:
    """GET /api/folder-trees — return file-tree listings for all configured folders."""
    from captain_claw.config import get_config
    from captain_claw.file_tree_builder import (
        build_gdrive_tree,
        build_local_tree,
        get_cached_tree,
        set_cached_tree,
    )

    cfg = get_config()
    ttl = cfg.tools.read.file_tree_cache_ttl_seconds
    max_entries = cfg.tools.read.file_tree_max_entries
    max_depth = cfg.tools.read.file_tree_max_depth

    trees: list[dict[str, Any]] = []

    # Local folders.
    for d in cfg.tools.read.extra_dirs:
        cache_key = f"local:{d}"
        cached = get_cached_tree(cache_key, ttl)
        if cached:
            trees.append({"type": "local", "path": d, "tree": cached})
            continue
        try:
            tree_str, count = build_local_tree(d, max_entries=max_entries, max_depth=max_depth)
            set_cached_tree(cache_key, tree_str, count)
            trees.append({"type": "local", "path": d, "tree": tree_str})
        except Exception as e:
            trees.append({"type": "local", "path": d, "tree": f"[Error: {e}]"})

    # GDrive folders.
    for gf in cfg.tools.read.gdrive_folders:
        cache_key = f"gdrive:{gf.id}"
        cached = get_cached_tree(cache_key, ttl)
        if cached:
            trees.append({"type": "gdrive", "id": gf.id, "name": gf.name, "tree": cached})
            continue
        try:
            tree_str, count = await build_gdrive_tree(
                gf.id, gf.name, max_entries=max_entries, max_depth=max_depth,
            )
            set_cached_tree(cache_key, tree_str, count)
            trees.append({"type": "gdrive", "id": gf.id, "name": gf.name, "tree": tree_str})
        except Exception as e:
            trees.append({"type": "gdrive", "id": gf.id, "name": gf.name, "tree": f"[Error: {e}]"})

    return web.json_response({"trees": trees})
