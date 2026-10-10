"""REST handler for image uploads (attach to chat)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from aiohttp import web

from captain_claw.config import get_config
from captain_claw.logging import get_logger
from captain_claw.web.rest_file_upload import (
    first_file_field,
    sanitize_upload_name,
    save_upload_field,
    upload_session_id,
)

if TYPE_CHECKING:
    from captain_claw.web_server import WebServer

log = get_logger(__name__)

_IMAGE_EXTENSIONS: set[str] = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}


async def upload_image(server: WebServer, request: web.Request) -> web.Response:
    """POST /api/image/upload — upload an image and save to workspace saved/media/.

    Returns JSON with the absolute path so the frontend can attach it to chat.
    Every upload gets its own name (``{stem}-{stamp}-{token}{ext}``, created
    exclusively), so two same-name photos never overwrite each other.
    """
    try:
        # Determine save location: workspace/saved/media/<session-id>/
        # For public users, scope uploads to their session.
        _is_public, session_id = upload_session_id(server, request)
        if session_id is None:
            return web.json_response({"error": "No session — reload the page and try again."},
                                     status=403)

        file_field = await first_file_field(request)
        if file_field is None:
            return web.json_response({"error": "No file field in upload"}, status=400)

        original_name = file_field.filename or "image.png"
        safe_stem, ext = sanitize_upload_name(original_name)

        if ext not in _IMAGE_EXTENSIONS:
            return web.json_response(
                {"error": f"Unsupported image type '{ext}'. Allowed: {', '.join(sorted(_IMAGE_EXTENSIONS))}"},
                status=400,
            )

        cfg = get_config()
        workspace = cfg.resolved_workspace_path()
        dest_dir = workspace / "saved" / "media" / session_id

        dest_path, size = await save_upload_field(file_field, dest_dir, safe_stem, ext)
        if size == 0:
            return web.json_response({"error": "Empty file"}, status=400)

        log.info(
            "Image uploaded",
            filename=original_name,
            path=str(dest_path),
            size=size,
        )

        return web.json_response({
            "path": str(dest_path),
            "filename": original_name,
            "size": size,
        })

    except web.HTTPException:
        raise
    except Exception as exc:
        log.error("Image upload failed", error=str(exc))
        return web.json_response({"error": f"Upload failed: {exc}"}, status=500)
