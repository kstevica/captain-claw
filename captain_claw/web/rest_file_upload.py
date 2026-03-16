"""REST handler for general file uploads (attach to chat)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from aiohttp import web

from captain_claw.config import get_config
from captain_claw.logging import get_logger

if TYPE_CHECKING:
    from captain_claw.web_server import WebServer

log = get_logger(__name__)

_ALLOWED_EXTENSIONS: set[str] = {
    ".csv", ".xlsx", ".xls",
    ".pdf", ".docx", ".doc",
    ".pptx", ".ppt",
    ".md", ".txt",
}


async def upload_file(server: WebServer, request: web.Request) -> web.Response:
    """POST /api/file/upload — upload a data file and save to workspace saved/downloads/.

    Returns JSON with the absolute path so the frontend can attach it to chat.
    The user decides what to do with the file (datastore import, deep memory, etc.).
    """
    try:
        reader = await request.multipart()
        if reader is None:
            return web.json_response({"error": "Multipart body required"}, status=400)

        file_field = None
        while True:
            field = await reader.next()
            if field is None:
                break
            if field.name == "file":
                file_field = field
                break

        if file_field is None:
            return web.json_response({"error": "No file field in upload"}, status=400)

        original_name = file_field.filename or "data.csv"
        ext = Path(original_name).suffix.lower()

        if ext not in _ALLOWED_EXTENSIONS:
            return web.json_response(
                {"error": f"Unsupported file type '{ext}'. Allowed: {', '.join(sorted(_ALLOWED_EXTENSIONS))}"},
                status=400,
            )

        # Read file data.
        chunks: list[bytes] = []
        while True:
            chunk = await file_field.read_chunk(8192)
            if not chunk:
                break
            chunks.append(chunk)
        file_bytes = b"".join(chunks)

        if not file_bytes:
            return web.json_response({"error": "Empty file"}, status=400)

        # Determine save location: workspace/saved/downloads/<session-id>/
        cfg = get_config()
        workspace = cfg.resolved_workspace_path()

        # For public users, scope uploads to their session.
        from captain_claw.web.public_auth import get_request_session_id
        is_public, pub_session_id = get_request_session_id(request)
        if is_public and pub_session_id:
            session_id = pub_session_id
        else:
            session_id = ""
            if server.agent and server.agent.session:
                session_id = server.agent.session.id or ""
        if not session_id:
            session_id = "uploads"

        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        stem = Path(original_name).stem
        safe_stem = "".join(c if c.isalnum() or c in "-_." else "_" for c in stem)[:60]
        filename = f"{safe_stem}-{stamp}{ext}"

        dest_dir = workspace / "saved" / "downloads" / session_id
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / filename

        dest_path.write_bytes(file_bytes)

        log.info(
            "File uploaded",
            filename=original_name,
            path=str(dest_path),
            size=len(file_bytes),
        )

        return web.json_response({
            "path": str(dest_path),
            "filename": original_name,
            "size": len(file_bytes),
        })

    except web.HTTPException:
        raise
    except Exception as exc:
        log.error("File upload failed", error=str(exc))
        return web.json_response({"error": f"Upload failed: {exc}"}, status=500)
