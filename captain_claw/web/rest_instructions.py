"""REST handlers for instruction file management."""

from __future__ import annotations

from typing import TYPE_CHECKING

from aiohttp import web

if TYPE_CHECKING:
    from captain_claw.web_server import WebServer


async def list_instructions(server: WebServer, request: web.Request) -> web.Response:
    """List instruction .md files, merging system and personal dirs."""
    seen: dict[str, dict] = {}

    # System (base) directory first
    if server._instructions_dir.is_dir():
        for f in sorted(server._instructions_dir.iterdir()):
            if f.suffix == ".md" and f.is_file():
                seen[f.name] = {
                    "name": f.name,
                    "size": f.stat().st_size,
                    "overridden": False,
                }

    # Personal overrides — update size and mark as overridden
    if server._instructions_personal_dir.is_dir():
        for f in sorted(server._instructions_personal_dir.iterdir()):
            if f.suffix == ".md" and f.is_file():
                seen[f.name] = {
                    "name": f.name,
                    "size": f.stat().st_size,
                    "overridden": True,
                }

    files = sorted(seen.values(), key=lambda x: x["name"])
    return web.json_response(files)


async def get_instruction(server: WebServer, request: web.Request) -> web.Response:
    """Read an instruction file (personal override wins over system)."""
    name = request.match_info["name"]
    if ".." in name or "/" in name or not name.endswith(".md"):
        return web.json_response({"error": "Invalid file name"}, status=400)

    personal_path = server._instructions_personal_dir / name
    system_path = server._instructions_dir / name
    overridden = personal_path.is_file()
    path = personal_path if overridden else system_path

    if not path.is_file():
        return web.json_response({"error": "File not found"}, status=404)

    content = path.read_text(encoding="utf-8")
    return web.json_response({
        "name": name,
        "content": content,
        "overridden": overridden,
    })


async def put_instruction(server: WebServer, request: web.Request) -> web.Response:
    """Save instruction to the personal override directory."""
    name = request.match_info["name"]
    if ".." in name or "/" in name or "\\" in name or not name.endswith(".md"):
        return web.json_response({"error": "Invalid file name"}, status=400)
    path = server._instructions_personal_dir / name
    try:
        path.resolve().relative_to(server._instructions_personal_dir.resolve())
    except ValueError:
        return web.json_response({"error": "Invalid path"}, status=400)
    body = await request.json()
    content = body.get("content", "")
    is_new = not path.exists()
    server._instructions_personal_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    if server.agent:
        server.agent.instructions._cache.pop(name, None)
    status = "created" if is_new else "saved"
    return web.json_response({
        "status": status,
        "name": name,
        "size": path.stat().st_size,
        "overridden": True,
    })


async def revert_instruction(server: WebServer, request: web.Request) -> web.Response:
    """Delete the personal override, reverting to the system default."""
    name = request.match_info["name"]
    if ".." in name or "/" in name or "\\" in name or not name.endswith(".md"):
        return web.json_response({"error": "Invalid file name"}, status=400)
    personal_path = server._instructions_personal_dir / name
    system_path = server._instructions_dir / name
    if not personal_path.is_file():
        return web.json_response({"error": "No personal override to revert"}, status=404)
    if not system_path.is_file():
        return web.json_response(
            {"error": "No system default exists — cannot revert"},
            status=400,
        )
    personal_path.unlink()
    if server.agent:
        server.agent.instructions._cache.pop(name, None)
    content = system_path.read_text(encoding="utf-8")
    return web.json_response({
        "status": "reverted",
        "name": name,
        "content": content,
        "size": system_path.stat().st_size,
        "overridden": False,
    })
