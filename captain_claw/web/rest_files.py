"""REST handlers for the file browser."""

from __future__ import annotations

import mimetypes
from pathlib import Path
from typing import TYPE_CHECKING, Any

from aiohttp import web

if TYPE_CHECKING:
    from captain_claw.web_server import WebServer


# Extensions considered safe to display as text in the browser.
_TEXT_EXTENSIONS: set[str] = {
    ".txt", ".md", ".markdown", ".rst", ".log", ".csv", ".tsv",
    ".json", ".jsonl", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf",
    ".py", ".js", ".ts", ".jsx", ".tsx", ".html", ".htm", ".css", ".scss",
    ".xml", ".svg", ".sql", ".sh", ".bash", ".zsh", ".fish",
    ".c", ".cpp", ".h", ".hpp", ".java", ".go", ".rs", ".rb", ".php",
    ".r", ".m", ".swift", ".kt", ".lua", ".pl", ".pm",
    ".env", ".gitignore", ".dockerignore", ".editorconfig",
    ".makefile", ".cmake", ".gradle",
}

# Filenames without extensions that are text.
_TEXT_FILENAMES: set[str] = {
    "makefile", "dockerfile", "readme", "license", "changelog",
    "authors", "contributing", "todo", "notes",
}

# Maximum size (bytes) for inline text preview.
_MAX_TEXT_PREVIEW_BYTES = 2 * 1024 * 1024  # 2 MB


def _is_text_file(path: Path) -> bool:
    """Determine if a file is likely text based on extension or name."""
    suffix = path.suffix.lower()
    if suffix in _TEXT_EXTENSIONS:
        return True
    if not suffix and path.name.lower() in _TEXT_FILENAMES:
        return True
    return False


def _enrich(logical: str, physical: str, source: str) -> dict[str, Any]:
    """Build a file metadata dict from a logical/physical path pair."""
    p = Path(physical)
    exists = p.is_file()
    size = 0
    modified: float = 0
    mime_type = ""

    if exists:
        try:
            stat = p.stat()
            size = stat.st_size
            modified = stat.st_mtime
        except OSError:
            pass
        mime_type = mimetypes.guess_type(str(p))[0] or ""

    return {
        "logical": logical,
        "physical": physical,
        "filename": p.name,
        "extension": p.suffix.lower(),
        "exists": exists,
        "size": size,
        "modified": modified,
        "mime_type": mime_type or "application/octet-stream",
        "is_text": _is_text_file(p),
        "source": source,
    }


async def _collect_files(server: WebServer) -> list[dict[str, Any]]:
    """Merge files from in-memory registries and persisted SQLite table."""
    seen: dict[str, dict[str, Any]] = {}  # physical_path → entry

    # 1. In-memory registries (current session).
    registries: list[tuple[str, Any]] = []

    if server.agent and getattr(server.agent, "_file_registry", None):
        registries.append(("agent", server.agent._file_registry))

    if getattr(server, "_orchestrator", None) and getattr(
        server._orchestrator, "_file_registry", None
    ):
        registries.append(("orchestrator", server._orchestrator._file_registry))

    for source, registry in registries:
        for entry in registry.list_files():
            logical = entry.get("logical", "")
            physical = entry.get("physical", "")
            if not physical or physical in seen:
                continue
            seen[physical] = _enrich(logical, physical, source)

    # 2. Persisted entries from SQLite (survives restarts).
    try:
        from captain_claw.session import get_session_manager
        sm = get_session_manager()
        persisted = await sm.list_registered_files(limit=1000)
        for row in persisted:
            physical = row.get("physical_path", "")
            if not physical or physical in seen:
                continue
            logical = row.get("logical_path", "")
            source = row.get("source", "persisted")
            seen[physical] = _enrich(logical, physical, source)
    except Exception:
        pass  # Best-effort; in-memory data is still returned.

    return sorted(seen.values(), key=lambda f: f["logical"].lower())


# ------------------------------------------------------------------
# Handlers
# ------------------------------------------------------------------


async def list_files(server: WebServer, request: web.Request) -> web.Response:
    """GET /api/files — list all registered files with metadata."""
    files = await _collect_files(server)
    return web.json_response(files)


async def get_file_content(server: WebServer, request: web.Request) -> web.Response:
    """GET /api/files/content?path=<physical_path> — read text file content."""
    physical = request.query.get("path", "").strip()
    if not physical:
        return web.json_response({"error": "Missing 'path' parameter"}, status=400)

    # Security: verify path is in a current registry.
    known_physicals = {f["physical"] for f in await _collect_files(server)}
    if physical not in known_physicals:
        return web.json_response({"error": "File not in registry"}, status=403)

    p = Path(physical)
    if not p.is_file():
        return web.json_response({"error": "File not found on disk"}, status=404)

    try:
        stat = p.stat()
        if stat.st_size > _MAX_TEXT_PREVIEW_BYTES:
            return web.json_response(
                {"error": "File too large for text preview (>2 MB)"},
                status=413,
            )
    except OSError as exc:
        return web.json_response({"error": str(exc)}, status=500)

    try:
        content = p.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        return web.json_response({"error": f"Read error: {exc}"}, status=500)

    mime_type = mimetypes.guess_type(str(p))[0] or "text/plain"
    return web.json_response({
        "path": physical,
        "filename": p.name,
        "content": content,
        "size": stat.st_size,
        "mime_type": mime_type,
    })


async def download_file(server: WebServer, request: web.Request) -> web.Response:
    """GET /api/files/download?path=<physical_path> — download any registered file."""
    physical = request.query.get("path", "").strip()
    if not physical:
        return web.json_response({"error": "Missing 'path' parameter"}, status=400)

    # Security: verify path is in a current registry.
    known_physicals = {f["physical"] for f in await _collect_files(server)}
    if physical not in known_physicals:
        return web.json_response({"error": "File not in registry"}, status=403)

    p = Path(physical)
    if not p.is_file():
        return web.json_response({"error": "File not found on disk"}, status=404)

    return web.FileResponse(
        p,
        headers={
            "Content-Disposition": f'attachment; filename="{p.name}"',
        },
    )
