"""REST handler for general file uploads (attach to chat)."""

from __future__ import annotations

import asyncio
import lzma
import posixpath
import re
import secrets
import shutil
import struct
import unicodedata
import zipfile
import zlib
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, BinaryIO

from aiohttp import web

from captain_claw.config import get_config
from captain_claw.logging import get_logger

if TYPE_CHECKING:
    from captain_claw.web_server import WebServer

log = get_logger(__name__)

# What a public (unauthenticated) visitor may upload. The owner and Flight
# Deck may upload any file: WhatsApp hands over voice notes, contacts,
# calendar invites, JSON, … — they are served back with a sandbox CSP and
# nosniff (rest_files), never as live HTML on the agent's origin.
_ALLOWED_EXTENSIONS: set[str] = {
    ".csv", ".xlsx", ".xls",
    ".pdf", ".docx", ".doc",
    ".pptx", ".ppt",
    ".md", ".txt",
    ".zip",
    # Images: the chat composer uploads attachments here too; accept them so
    # a manually-attached photo isn't rejected (the agent views it via vision).
    ".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp",
    # Video: analyzed by the video_vision tool (frame sampling + transcription).
    ".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v",
}

# Zip extraction limits: past either one the zip stays as a plain file.
_ZIP_MAX_MEMBERS = 5000
_ZIP_MAX_TOTAL_BYTES = 2 * 1024 * 1024 * 1024  # 2 GiB uncompressed

_READ_CHUNK = 64 * 1024
_WRITE_BATCH = 1024 * 1024  # bytes buffered per disk write (one thread hop)
_NAME_ATTEMPTS = 20
_NOT_EXT_CHAR_RE = re.compile(r"[^a-z0-9]")


class _ZipLimitError(ValueError):
    """The archive is over an extraction limit."""


# ── shared upload helpers (also used by rest_image_upload) ─────────────


def sanitize_upload_name(original: str) -> tuple[str, str]:
    """``(safe_stem, ext)`` for an uploaded file name.

    NFC first, so a decomposed accent stays one letter instead of turning
    into ``_``. The extension is lower-cased and keeps only ``[a-z0-9]``;
    one of 1–10 characters is kept, otherwise there is no extension. A bare
    ``.xlsx`` is stem ``""`` + ``.xlsx``. Leading dots are stripped from the
    stem (no hidden files); it may come back empty.
    """
    name = unicodedata.normalize("NFC", str(original or ""))
    # A client-sent path keeps only its last part.
    name = name.replace("\\", "/").rsplit("/", 1)[-1]
    stem, dot, suffix = name.rpartition(".")
    ext = _NOT_EXT_CHAR_RE.sub("", suffix.lower()) if dot else ""
    if not 1 <= len(ext) <= 10:
        stem, ext = name, ""
    safe_stem = "".join(c if c.isalnum() or c in "-_." else "_" for c in stem)[:60]
    return safe_stem.lstrip("."), (f".{ext}" if ext else "")


def open_unique_upload(dest_dir: Path, safe_stem: str, ext: str) -> tuple[Path, BinaryIO]:
    """Create ``{stem}-{stamp}-{token}{ext}`` exclusively and return it open
    for writing. A name that already exists gets a new token — two uploads
    of the same name in the same second (a WhatsApp album) never overwrite
    each other. Blocking: call it through ``asyncio.to_thread``."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    for _ in range(_NAME_ATTEMPTS):
        path = dest_dir / f"{safe_stem or 'file'}-{stamp}-{secrets.token_hex(3)}{ext}"
        try:
            return path, open(path, "xb")  # the caller closes it
        except FileExistsError:
            continue
    raise FileExistsError(f"no free upload name in {dest_dir}")


async def stream_field_to_file(field: Any, fh: BinaryIO) -> int:
    """Copy a multipart field into *fh* without holding it all in memory;
    disk writes run in a thread. Returns the byte count."""
    total = 0
    buf = bytearray()
    while True:
        chunk = await field.read_chunk(_READ_CHUNK)
        if not chunk:
            break
        total += len(chunk)
        buf += chunk
        if len(buf) >= _WRITE_BATCH:
            await asyncio.to_thread(fh.write, bytes(buf))
            buf.clear()
    if buf:
        await asyncio.to_thread(fh.write, bytes(buf))
    return total


async def save_upload_field(field: Any, dest_dir: Path, safe_stem: str, ext: str) -> tuple[Path, int]:
    """Stream *field* into a new unique file under *dest_dir*.

    Returns ``(path, size)``. A partial file is removed when the copy fails;
    an empty one is removed too and reported as size 0.
    """
    path, fh = await asyncio.to_thread(open_unique_upload, dest_dir, safe_stem, ext)
    try:
        size = await stream_field_to_file(field, fh)
    except BaseException:
        fh.close()
        path.unlink(missing_ok=True)
        raise
    await asyncio.to_thread(fh.close)
    if size == 0:
        path.unlink(missing_ok=True)
    return path, size


def upload_session_id(server: WebServer, request: web.Request) -> tuple[bool, str | None]:
    """``(is_public, folder id)`` for an upload; ``None`` = refuse (403).

    A public visitor uploads into their own session's folder. One with no
    session cookie gets nothing — never the owner's main-session folder.
    """
    from captain_claw.web.public_auth import get_request_session_id

    is_public, pub_session_id = get_request_session_id(request)
    if is_public:
        return True, (pub_session_id or None)
    session_id = ""
    if server.agent and server.agent.session:
        session_id = server.agent.session.id or ""
    return False, (session_id or "uploads")


async def first_file_field(request: web.Request) -> Any | None:
    """The multipart ``file`` field, or ``None`` when there is none."""
    reader = await request.multipart()
    if reader is None:
        return None
    while True:
        field = await reader.next()
        if field is None:
            return None
        if field.name == "file":
            return field


# ── zip extraction ─────────────────────────────────────────────────────


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
    """Extract a zip archive preserving folder structure. Returns list of extracted relative paths.

    Refuses more than ``_ZIP_MAX_MEMBERS`` entries or more than
    ``_ZIP_MAX_TOTAL_BYTES`` uncompressed (declared, and counted while
    writing — a zip bomb stops at the cap). Blocking: run it in a thread.
    """
    resolved_target = target_dir.resolve()
    extracted: list[str] = []
    with zipfile.ZipFile(archive_path, "r") as archive:
        members = archive.infolist()
        if len(members) > _ZIP_MAX_MEMBERS:
            raise _ZipLimitError(f"more than {_ZIP_MAX_MEMBERS} entries")
        if sum(m.file_size for m in members) > _ZIP_MAX_TOTAL_BYTES:
            raise _ZipLimitError("over 2 GiB uncompressed")
        written = 0
        for member in members:
            rel_path = _normalize_zip_member_path(member.filename)
            if not rel_path:
                continue
            destination = (resolved_target / rel_path).resolve()
            # Safety: ensure destination is within target directory.
            if not destination.is_relative_to(resolved_target):
                raise ValueError(f"Archive member escapes target directory: {member.filename}")
            if member.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member, "r") as source_file:
                with destination.open("wb") as out_file:
                    while True:
                        block = source_file.read(_READ_CHUNK)
                        if not block:
                            break
                        written += len(block)
                        if written > _ZIP_MAX_TOTAL_BYTES:
                            raise _ZipLimitError("over 2 GiB uncompressed")
                        out_file.write(block)
            extracted.append(rel_path)
    return extracted


def _extract_error_reason(exc: BaseException) -> str:
    """A short, user-facing reason a zip was kept unextracted."""
    if isinstance(exc, _ZipLimitError):
        return str(exc)
    if isinstance(exc, zipfile.BadZipFile):
        return "not a valid zip archive (corrupt or truncated)"
    if isinstance(exc, RuntimeError) and any(w in str(exc).lower() for w in ("encrypt", "password")):
        return "the archive is password-protected"
    if isinstance(exc, NotImplementedError):
        return "the archive uses an unsupported compression method"
    if isinstance(exc, (zlib.error, lzma.LZMAError, struct.error, EOFError)):
        return "not a valid zip archive (corrupt or truncated)"
    if isinstance(exc, ValueError) and "escapes target directory" in str(exc):
        return "an entry points outside the archive folder"
    if isinstance(exc, (FileExistsError, NotADirectoryError, IsADirectoryError)):
        return "two entries collide (a file and a folder with the same name)"
    if isinstance(exc, OSError):
        # Never the message itself: it carries the deck's absolute paths.
        return f"it couldn't be unpacked here ({exc.strerror or type(exc).__name__})"
    return f"it couldn't be unpacked ({type(exc).__name__})"


def _query_flag(request: web.Request, name: str, default: bool = True) -> bool:
    raw = str(request.query.get(name, "") or "").strip().lower()
    if not raw:
        return default
    return raw not in {"0", "false", "no", "off"}


# ── handler ────────────────────────────────────────────────────────────


async def upload_file(server: WebServer, request: web.Request) -> web.Response:
    """POST /api/file/upload — upload a data file and save to workspace saved/downloads/.

    Returns JSON with the absolute path so the frontend can attach it to chat
    (``path``, ``filename``, ``size``). The user decides what to do with the
    file (datastore import, deep memory, etc.).

    ``?extract=0`` keeps a .zip as a plain file; by default it is extracted
    into a folder named after the saved zip and the folder's path comes back
    (``extracted: true``, ``files``). A zip that can't be extracted stays as
    it is: its own path, ``extracted: false`` and ``extract_error``.
    """
    try:
        # Determine save location: workspace/saved/downloads/<session-id>/
        # For public users, scope uploads to their session.
        is_public, session_id = upload_session_id(server, request)
        if session_id is None:
            return web.json_response({"error": "No session — reload the page and try again."},
                                     status=403)

        file_field = await first_file_field(request)
        if file_field is None:
            return web.json_response({"error": "No file field in upload"}, status=400)

        original_name = file_field.filename or "data.csv"
        safe_stem, ext = sanitize_upload_name(original_name)

        if is_public and ext not in _ALLOWED_EXTENSIONS:
            return web.json_response(
                {"error": f"Unsupported file type '{ext}'. Allowed: {', '.join(sorted(_ALLOWED_EXTENSIONS))}"},
                status=400,
            )

        cfg = get_config()
        workspace = cfg.resolved_workspace_path()
        dest_dir = workspace / "saved" / "downloads" / session_id

        dest_path, size = await save_upload_field(file_field, dest_dir, safe_stem, ext)
        if size == 0:
            return web.json_response({"error": "Empty file"}, status=400)

        # If zip file, extract contents preserving folder structure — into a
        # folder named after this upload's unique name, so two zips never merge.
        if ext == ".zip" and _query_flag(request, "extract"):
            extract_dir = dest_path.with_suffix("")
            created = False  # only a folder this upload made is ours to remove
            try:
                await asyncio.to_thread(extract_dir.mkdir, parents=True, exist_ok=False)
                created = True
                extracted = await asyncio.to_thread(_extract_zip_upload, dest_path, extract_dir)
            except (zipfile.BadZipFile, zipfile.LargeZipFile, RuntimeError, NotImplementedError,
                    ValueError, EOFError, OSError, zlib.error, lzma.LZMAError, struct.error) as exc:
                reason = _extract_error_reason(exc)
                if created:
                    await asyncio.to_thread(shutil.rmtree, extract_dir, ignore_errors=True)
                log.warning("Zip upload kept unextracted", filename=original_name,
                            path=str(dest_path), reason=reason)
                return web.json_response({
                    "path": str(dest_path),
                    "filename": original_name,
                    "size": size,
                    "extracted": False,
                    "extract_error": reason,
                })
            # Remove the zip file after successful extraction.
            dest_path.unlink()

            log.info(
                "Zip uploaded and extracted",
                filename=original_name,
                extract_dir=str(extract_dir),
                files_extracted=len(extracted),
                size=size,
            )

            return web.json_response({
                "path": str(extract_dir),
                "filename": original_name,
                "size": size,
                "extracted": True,
                "files": extracted,
            })

        log.info(
            "File uploaded",
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
        log.error("File upload failed", error=str(exc))
        return web.json_response({"error": f"Upload failed: {exc}"}, status=500)
