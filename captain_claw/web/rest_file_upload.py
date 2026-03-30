"""REST handler for general file uploads (attach to chat)."""

from __future__ import annotations

import posixpath
import shutil
import zipfile
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
    ".zip",
}


def _normalize_zip_member_path(raw: str) -> str | None:
    """Validate and normalize a zip member path, rejecting traversal attempts."""
    cleaned = str(raw or "").replace("\\", "/")
    if not cleaned:
        return None
    parts = [part for part in cleaned.split("/") if part and part != "."]
    if not parts:
        return None
    normalized = posixpath.normpath("/".join(parts))
    if not normalized or normalized in {".", ".."}:
        return None
    if normalized.startswith("../") or normalized.startswith("/"):
        raise ValueError(f"Archive member escapes target directory: {raw}")
    if any(part in {"..", ""} for part in normalized.split("/")):
        raise ValueError(f"Archive member escapes target directory: {raw}")
    return normalized


def _extract_zip_upload(archive_path: Path, target_dir: Path) -> list[str]:
    """Extract a zip archive preserving folder structure. Returns list of extracted relative paths."""
    resolved_target = target_dir.resolve()
    extracted: list[str] = []
    with zipfile.ZipFile(archive_path, "r") as archive:
        for member in archive.infolist():
            rel_path = _normalize_zip_member_path(member.filename)
            if not rel_path:
                continue
            destination = (resolved_target / rel_path).resolve()
            # Safety: ensure destination is within target directory.
            destination.relative_to(resolved_target)
            if member.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member, "r") as source_file:
                with destination.open("wb") as out_file:
                    shutil.copyfileobj(source_file, out_file)
            extracted.append(rel_path)
    return extracted


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

        # If zip file, extract contents preserving folder structure.
        if ext == ".zip":
            extract_dir = dest_dir / safe_stem
            extract_dir.mkdir(parents=True, exist_ok=True)
            extracted = _extract_zip_upload(dest_path, extract_dir)
            # Remove the zip file after successful extraction.
            dest_path.unlink()

            log.info(
                "Zip uploaded and extracted",
                filename=original_name,
                extract_dir=str(extract_dir),
                files_extracted=len(extracted),
                size=len(file_bytes),
            )

            return web.json_response({
                "path": str(extract_dir),
                "filename": original_name,
                "size": len(file_bytes),
                "extracted": True,
                "files": extracted,
            })

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
