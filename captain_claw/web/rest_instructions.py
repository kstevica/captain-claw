"""REST handlers for instruction file management."""

from __future__ import annotations

from typing import TYPE_CHECKING

from aiohttp import web

if TYPE_CHECKING:
    from captain_claw.web_server import WebServer


async def list_instructions(server: WebServer, request: web.Request) -> web.Response:
    """List instruction .md files, merging system and personal dirs.

    ``micro_*`` files are excluded from the listing; instead each standard
    file gets a ``has_micro`` boolean that the frontend uses to show the
    side-by-side micro editor.
    """
    seen: dict[str, dict] = {}
    micro_names: set[str] = set()

    # Collect all filenames from both directories first to discover micro_ variants.
    for d, overridden in [(server._instructions_dir, False), (server._instructions_personal_dir, True)]:
        if not d.is_dir():
            continue
        for f in sorted(d.iterdir()):
            if f.suffix != ".md" or not f.is_file():
                continue
            if f.name.startswith("micro_"):
                micro_names.add(f.name)
                continue  # Don't list micro files directly
            seen[f.name] = {
                "name": f.name,
                "size": f.stat().st_size,
                "overridden": overridden,
            }

    # Tag standard files that have a micro counterpart.
    for entry in seen.values():
        entry["has_micro"] = f"micro_{entry['name']}" in micro_names

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
