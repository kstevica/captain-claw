"""Shared-agent member HTTP routes: the saved/ commons and the datastore (PR C).

Flight Deck reaches a member's Files and Datastore panels through
``/api/speaker/*`` — the ONLY HTTP paths that accept ``X-FD-Speaker`` (the
WebSocket handshake verifies its own). Each request carries the owner token
(the auth middleware, which runs first) AND a fresh HTTP assertion bound to
this one request (``aud="http"``, method, path; single-use nonce):
:func:`speaker.verify_http_assertion`. Every response after a successful
verify carries ``X-FD-Speaker-Ack``, so Flight Deck can tell a PR C agent
from an older one.

Payloads name files by their path below ``saved/`` and creators by kind, Flight
Deck user id and name snapshot — never a host path, host, port or token
(contract part 0b §2.2, G-C4). Ownership is decided here, on the agent, from
the verified principal (G-C2): Flight Deck's ``can_delete`` is display only.

Depends only on ``server.config`` and ``await server._speaker_session(p)``,
so tests can pass a stub server.
"""

from __future__ import annotations

import asyncio
import mimetypes
import os
import re
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from aiohttp import web

from captain_claw import saved_attribution, speaker
from captain_claw.logging import get_logger

log = get_logger(__name__)

# ── constants (contract part 0b §4) ──────────────────────────────────

MEMBER_UPLOAD_EXTENSIONS = (".csv", ".xlsx", ".xls", ".pdf", ".docx", ".doc", ".pptx", ".ppt",
                            ".md", ".txt", ".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp",
                            ".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v")
MEMBER_UPLOAD_MAX_BYTES = 25 * 1024 * 1024        # 26214400 — FD and agent
MEMBER_UPLOAD_QUOTA_BYTES = 200 * 1024 * 1024     # sum of valid stamped files of that member
MEMBER_DOWNLOAD_MAX_BYTES = 50 * 1024 * 1024
MEMBER_FILES_LIST_MAX = 2000
MEMBER_FILES_SCAN_MAX = 10_000
TABLE_NAME_RE = r"^[a-z0-9_]{1,128}$"
ORDER_BY_RE = r"^_?[a-z0-9_]{1,128}$"
FILE_ID_MAX = 1024
EXPORT_FORMATS = ("csv", "json", "xlsx")
_ROWS_LIMIT_MAX = 500
_UPLOAD_CHUNK = 64 * 1024

_TABLE_RE = re.compile(TABLE_NAME_RE)
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")

ONLY_MEMBERS_MESSAGE = "This route is only for Flight Deck shared-agent members"
HEADER_ELSEWHERE_MESSAGE = "X-FD-Speaker is only accepted on /api/speaker/ routes"
ASSERTION_REFUSED_MESSAGE = "speaker assertion refused"
PUBLIC_AGENT_MESSAGE = "Not available on a public agent"
NOT_AVAILABLE_MESSAGE = "Files and data aren't available in shared chats on this agent."
INTERNAL_MESSAGE = "internal error"
BAD_FILE_MESSAGE = "Invalid file"
NO_FILE_MESSAGE = "No such file"
DOWNLOAD_TOO_LARGE_MESSAGE = "That file is too large to open here (50 MB at most)"
BAD_TYPE_MESSAGE = "That kind of file can't be uploaded here"
UPLOAD_TOO_LARGE_MESSAGE = "That file is too large (25 MB at most)"
EMPTY_MESSAGE = "That file is empty"
QUOTA_MESSAGE = "You've used your 200 MB for files on this agent — delete some first."
NO_UPLOAD_MESSAGE = "No file in the upload"
BAD_FOLDER_MESSAGE = "Invalid upload folder"
NO_FREE_NAME_MESSAGE = "Try again in a second"
BAD_TABLE_MESSAGE = "Invalid table name"
NO_TABLE_MESSAGE = "No such table"
UNKNOWN_COLUMN_MESSAGE = "Unknown column"
BAD_PAGE_MESSAGE = "Invalid limit, offset or order"
BAD_FORMAT_MESSAGE = "Unsupported format"


def _error(message: str, status: int) -> web.Response:
    return web.json_response({"error": message}, status=status)


# ── middleware ───────────────────────────────────────────────────────


def create_speaker_http_middleware(server: Any):
    """Honour ``X-FD-Speaker`` on ``/api/speaker/*`` only (and pass ``/ws``
    through untouched — its handshake verifies the header itself).

    Appended AFTER the auth / public middleware, so the owner token is still
    required first. It only ever refuses or binds: owner traffic without the
    header reaches its handler unchanged.
    """

    @web.middleware
    async def middleware(request: web.Request, handler) -> web.StreamResponse:
        header = request.headers.get(speaker.SPEAKER_HEADER)
        member_path = request.path.startswith(speaker.SPEAKER_HTTP_PREFIX)
        if not header:
            if member_path:
                return _error(ONLY_MEMBERS_MESSAGE, 403)
            return await handler(request)
        if request.path == "/ws":
            # Verifying here would burn the single-use nonce.
            return await handler(request)
        if not member_path:
            # Never reaches an owner route (/ws/stt included).
            return _error(HEADER_ELSEWHERE_MESSAGE, 403)
        try:
            p = speaker.verify_http_assertion(
                header, str(server.config.web.auth_token or ""), request.method, request.path,
            )
        except speaker.SpeakerAuthError as exc:
            log.info("Speaker HTTP assertion refused", reason=str(exc))
            return _error(ASSERTION_REFUSED_MESSAGE, 401)

        ack = speaker.speaker_ack_for(header)
        resp = await _run_member_handler(server, request, handler, p)
        resp.headers[speaker.SPEAKER_ACK_HEADER] = ack
        return resp

    return middleware


async def _run_member_handler(
    server: Any, request: web.Request, handler, p: speaker.Principal,
) -> web.StreamResponse:
    if server.config.web.public_run:
        return _error(PUBLIC_AGENT_MESSAGE, 403)
    if not p.speaker_id or speaker.runtime_of(p) != "process":
        return _error(NOT_AVAILABLE_MESSAGE, 403)
    request["fd_speaker"] = p
    tok = speaker.bind(p)
    try:
        return await handler(request)
    except web.HTTPException as exc:
        # An unknown /api/speaker/ path (404), a wrong method (405), …
        return _error(exc.reason, exc.status)
    except Exception as exc:
        log.error("Speaker HTTP route failed", route=request.path, error=type(exc).__name__)
        return _error(INTERNAL_MESSAGE, 500)
    finally:
        speaker.reset(tok)


def _principal(request: web.Request) -> speaker.Principal | None:
    """The member the middleware verified (defense in depth: None → 403)."""
    p = request.get("fd_speaker")
    return p if isinstance(p, speaker.Principal) else None


# ── files ────────────────────────────────────────────────────────────


def valid_file_id(raw: object) -> bool:
    """A file id (contract part 0b §4): a relative posix path below saved/,
    1..1024 chars, no backslash or control char, no empty / dot / hidden part."""
    if not isinstance(raw, str) or not 1 <= len(raw) <= FILE_ID_MAX:
        return False
    if "\\" in raw or _CONTROL_RE.search(raw) or raw.startswith("/"):
        return False
    for part in raw.split("/"):
        if not part or part in (".", "..") or part.startswith("."):
            return False
    return True


def _resolve_id(raw: str, p: speaker.Principal) -> Path | None:
    """The file an id names, when this member may see it; else None."""
    base = saved_attribution.saved_base()
    real = (base / raw).resolve()
    try:
        real.relative_to(base)
    except ValueError:
        return None
    if not saved_attribution.visible_to_member(real, p.speaker_id):
        return None
    return real if real.is_file() else None


def _agent_file(path: Path, rel: str, st: os.stat_result, creator: dict[str, str]) -> dict:
    from captain_claw.web.rest_files import _is_text_file

    return {
        "id": rel,
        "filename": path.name,
        "extension": path.suffix.lower(),
        "size": st.st_size,
        "modified": st.st_mtime,
        "mime_type": mimetypes.guess_type(path.name)[0] or "application/octet-stream",
        "is_text": _is_text_file(path),
        "created_by": creator,
    }


def _scan_saved(base: Path) -> tuple[list[tuple[str, str, os.stat_result]], bool]:
    """Regular files under *base* as ``(path, rel, stat)``: no hidden entries,
    no skip dirs, no symlinks; at most MEMBER_FILES_SCAN_MAX (→ truncated)."""
    from captain_claw.web.rest_files import _SCAN_SKIP_DIRS

    found: list[tuple[str, str, os.stat_result]] = []
    stack: list[tuple[str, str]] = [(str(base), "")]
    while stack:
        folder, prefix = stack.pop()
        try:
            with os.scandir(folder) as entries:
                for entry in entries:
                    if entry.name.startswith(".") or entry.is_symlink():
                        continue
                    if entry.is_dir(follow_symlinks=False):
                        if entry.name not in _SCAN_SKIP_DIRS:
                            stack.append((entry.path, prefix + entry.name + "/"))
                        continue
                    if not entry.is_file(follow_symlinks=False):
                        continue
                    if len(found) >= MEMBER_FILES_SCAN_MAX:
                        return found, True
                    try:
                        found.append((entry.path, prefix + entry.name,
                                      entry.stat(follow_symlinks=False)))
                    except OSError:
                        continue
        except OSError:
            continue
    return found, False


async def list_files(request: web.Request) -> web.Response:
    """GET /api/speaker/files — the commons, newest first."""
    p = _principal(request)
    if p is None:
        return _error(NOT_AVAILABLE_MESSAGE, 403)
    await saved_attribution.ensure_member_sessions()
    base = saved_attribution.saved_base()
    if not base.is_dir():
        return web.json_response({"files": [], "truncated": False})
    found, truncated = await asyncio.to_thread(_scan_saved, base)
    found.sort(key=lambda item: item[2].st_mtime, reverse=True)
    if len(found) > MEMBER_FILES_LIST_MAX:
        found = found[:MEMBER_FILES_LIST_MAX]
        truncated = True
    # On the loop, after the scan (the attribution DB is shared with the loop).
    creators = saved_attribution.creators_for([path for path, _rel, _st in found])
    files = []
    for path, rel, st in found:
        c = creators.get(path)
        if c is None:
            continue
        if (c.source == "folder" and c.user_id != p.speaker_id
                and not saved_attribution.LEGACY_MEMBER_FILES_SHARED):
            continue    # J20: another member's file from before PR C
        files.append(_agent_file(Path(path), rel, st, c.as_dict()))
    return web.json_response({"files": files, "truncated": truncated})


async def raw_file(request: web.Request) -> web.Response:
    """GET /api/speaker/files/raw?id= — a file's bytes."""
    p = _principal(request)
    if p is None:
        return _error(NOT_AVAILABLE_MESSAGE, 403)
    raw = request.query.get("id", "")
    if not valid_file_id(raw):
        return _error(BAD_FILE_MESSAGE, 400)
    # J20 needs the member sessions recorded (a raw request can come first
    # after a restart, before any listing or socket).
    await saved_attribution.ensure_member_sessions()
    real = _resolve_id(raw, p)
    if real is None:
        return _error(NO_FILE_MESSAGE, 404)
    try:
        size = real.stat().st_size
    except OSError:
        return _error(NO_FILE_MESSAGE, 404)
    if size > MEMBER_DOWNLOAD_MAX_BYTES:
        return _error(DOWNLOAD_TOO_LARGE_MESSAGE, 413)
    try:
        body = await asyncio.to_thread(real.read_bytes)
    except OSError:
        return _error(NO_FILE_MESSAGE, 404)
    if len(body) > MEMBER_DOWNLOAD_MAX_BYTES:
        return _error(DOWNLOAD_TOO_LARGE_MESSAGE, 413)
    return web.Response(body=body, content_type="application/octet-stream")


def _write_new(dest: Path, data: bytes) -> bool:
    """Create *dest* with O_CREAT|O_EXCL — never through an existing name or
    symlink; False when the name is taken."""
    try:
        with open(dest, "xb") as f:
            f.write(data)
        return True
    except FileExistsError:
        return False


async def _read_upload(request: web.Request) -> tuple[str, bytes] | web.Response:
    """The multipart ``file`` part: ``(name, bytes)`` or an error response."""
    try:
        reader = await request.multipart()
    except Exception:
        return _error(NO_UPLOAD_MESSAGE, 400)
    part = None
    while True:
        try:
            field = await reader.next()
        except Exception:
            return _error(NO_UPLOAD_MESSAGE, 400)
        if field is None:
            break
        if getattr(field, "name", None) == "file":
            part = field
            break
    if part is None:
        return _error(NO_UPLOAD_MESSAGE, 400)
    name = str(part.filename or "").replace("\\", "/").rsplit("/", 1)[-1]
    if Path(name).suffix.lower() not in MEMBER_UPLOAD_EXTENSIONS:
        return _error(BAD_TYPE_MESSAGE, 400)
    data = bytearray()
    while True:
        chunk = await part.read_chunk(_UPLOAD_CHUNK)
        if not chunk:
            break
        data.extend(chunk)
        if len(data) > MEMBER_UPLOAD_MAX_BYTES:
            return _error(UPLOAD_TOO_LARGE_MESSAGE, 413)
    if not data:
        return _error(EMPTY_MESSAGE, 400)
    return name, bytes(data)


def make_upload_handler(server: Any):
    # speaker_id → lock held from the quota check to the stamp, so parallel
    # uploads of one member can't each pass the check and overrun the quota.
    quota_locks: dict[str, asyncio.Lock] = {}

    async def upload_file(request: web.Request) -> web.Response:
        """POST /api/speaker/files/upload — into the member's own lane folder."""
        p = _principal(request)
        if p is None:
            return _error(NOT_AVAILABLE_MESSAGE, 403)
        got = await _read_upload(request)
        if isinstance(got, web.Response):
            return got
        name, data = got
        async with quota_locks.setdefault(p.speaker_id, asyncio.Lock()):
            return await _store_upload(server, p, name, data)

    return upload_file


async def _store_upload(server: Any, p: speaker.Principal, name: str, data: bytes) -> web.Response:
    """Quota check, the member's lane folder, an O_EXCL name, the stamp."""
    if saved_attribution.member_bytes(p.speaker_id) + len(data) > MEMBER_UPLOAD_QUOTA_BYTES:
        return _error(QUOTA_MESSAGE, 413)

    session = await server._speaker_session(p)
    saved_attribution.note_member_session(session.id, p.speaker_id, p.display_name)
    from captain_claw.tools.write import WriteTool

    slug = WriteTool._normalize_session_id(session.id)
    ext = Path(name).suffix.lower()
    stem = "".join(c if c.isalnum() or c in "-_." else "_" for c in Path(name).stem)[:60]
    stem = stem.lstrip(".") or "upload"
    base = saved_attribution.saved_base()
    folder = base / "downloads" / slug
    if (base / "downloads").is_symlink() or folder.is_symlink():
        return _error(BAD_FOLDER_MESSAGE, 400)       # never mkdir through a link
    folder.mkdir(parents=True, exist_ok=True)
    if folder.resolve() != base.resolve() / "downloads" / slug:
        return _error(BAD_FOLDER_MESSAGE, 400)       # a symlinked folder
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    candidates = [f"{stem}-{stamp}{ext}"] + [f"{stem}-{stamp}-{i}{ext}" for i in range(2, 10)]
    dest: Path | None = None
    for candidate in candidates:
        if await asyncio.to_thread(_write_new, folder / candidate, data):
            dest = folder / candidate
            break
    if dest is None:
        return _error(NO_FREE_NAME_MESSAGE, 409)
    saved_attribution.note_write(dest, None)      # the bound member's stamp
    rel = saved_attribution.rel_key(dest) or f"downloads/{slug}/{dest.name}"
    creator = saved_attribution.creator_of(dest).as_dict()
    return web.json_response(_agent_file(dest, rel, dest.stat(), creator))


async def delete_file(request: web.Request) -> web.Response:
    """POST /api/speaker/files/delete {"id"} — only the member's own file."""
    p = _principal(request)
    if p is None:
        return _error(NOT_AVAILABLE_MESSAGE, 403)
    try:
        body = await request.json()
    except Exception:
        return _error(BAD_FILE_MESSAGE, 400)
    raw = body.get("id") if isinstance(body, dict) else None
    if not valid_file_id(raw):
        return _error(BAD_FILE_MESSAGE, 400)
    await saved_attribution.ensure_member_sessions()
    real = _resolve_id(raw, p)
    if real is None:
        return _error(NO_FILE_MESSAGE, 404)
    if not saved_attribution.member_may_change(real, p.speaker_id):
        return _error(speaker.FILE_DELETE_NOT_YOURS, 403)
    rel = saved_attribution.rel_key(real)
    try:
        real.unlink()
    except FileNotFoundError:
        return _error(NO_FILE_MESSAGE, 404)
    saved_attribution.note_delete(real, rel)
    return web.json_response({"ok": True})


# ── datastore (the agent's global store, read-only here) ─────────────


def _table_payload(t: Any) -> dict:
    from captain_claw.datastore import creator_dict

    return {
        "name": t.name,
        "columns": [{"name": c.name, "type": c.col_type, "position": c.position}
                    for c in t.columns],
        "row_count": t.row_count,
        "created_at": t.created_at,
        "updated_at": t.updated_at,
        "created_by": creator_dict(t.created_by, t.created_by_name),
    }


async def list_tables(request: web.Request) -> web.Response:
    """GET /api/speaker/datastore/tables — every table, with its creator."""
    if _principal(request) is None:
        return _error(NOT_AVAILABLE_MESSAGE, 403)
    from captain_claw.datastore import get_datastore_manager

    tables = await get_datastore_manager().list_tables()
    return web.json_response({"tables": [_table_payload(t) for t in tables]})


def _int_param(raw: str, default: int) -> int | None:
    if raw in ("", None):
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


async def table_rows(request: web.Request) -> web.Response:
    """GET /api/speaker/datastore/tables/{name}/rows — a page, each row with ``_creator``."""
    if _principal(request) is None:
        return _error(NOT_AVAILABLE_MESSAGE, 403)
    from captain_claw.datastore import SYSTEM_COLUMNS, get_datastore_manager

    name = request.match_info.get("name", "")
    if not _TABLE_RE.fullmatch(name):
        return _error(BAD_TABLE_MESSAGE, 400)
    limit = _int_param(request.query.get("limit", ""), 100)
    offset = _int_param(request.query.get("offset", ""), 0)
    order_by = request.query.get("order_by", "") or "_id"
    order_dir = (request.query.get("order_dir", "") or "asc").lower()
    if limit is None or offset is None or offset < 0 or order_dir not in ("asc", "desc"):
        return _error(BAD_PAGE_MESSAGE, 400)
    limit = max(1, min(limit, _ROWS_LIMIT_MAX))
    dm = get_datastore_manager()
    try:
        info = await dm.describe_table(name)
    except ValueError:
        return _error(NO_TABLE_MESSAGE, 404)
    if order_by != "_id" and order_by not in SYSTEM_COLUMNS \
            and order_by not in {c.name for c in info.columns}:
        return _error(UNKNOWN_COLUMN_MESSAGE, 400)
    try:
        result = await dm.query(
            name, None, None, [("-" if order_dir == "desc" else "") + order_by],
            limit, offset, include_creator=True,
        )
    except ValueError as exc:
        if str(exc).startswith("Table not found"):
            return _error(NO_TABLE_MESSAGE, 404)
        return _error(str(exc)[:200], 400)
    cols = result["columns"]
    creators = result.get("creators") or []
    rows = []
    for i, row in enumerate(result["rows"]):
        item = {cols[j]: v for j, v in enumerate(row) if j < len(cols)}
        item["_creator"] = creators[i] if i < len(creators) else {"kind": "owner", "user_id": "",
                                                                  "name": ""}
        rows.append(item)
    return web.json_response({
        "columns": cols, "rows": rows, "total": result["total"],
        "offset": result["offset"], "limit": result["limit"],
    }, dumps=_json_dumps)


async def export_table(request: web.Request) -> web.Response:
    """GET /api/speaker/datastore/tables/{name}/export?format= — every cell
    that could run as a spreadsheet formula is defused (J13)."""
    if _principal(request) is None:
        return _error(NOT_AVAILABLE_MESSAGE, 403)
    from captain_claw.datastore import get_datastore_manager

    name = request.match_info.get("name", "")
    fmt = (request.query.get("format", "") or "csv").lower()
    if fmt not in EXPORT_FORMATS:
        return _error(BAD_FORMAT_MESSAGE, 400)
    if not _TABLE_RE.fullmatch(name):
        return _error(BAD_TABLE_MESSAGE, 400)
    dm = get_datastore_manager()
    try:
        await dm.describe_table(name)
    except ValueError:
        return _error(NO_TABLE_MESSAGE, 404)
    exporter = {"csv": dm.export_csv, "json": dm.export_json, "xlsx": dm.export_xlsx}[fmt]
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / f"{name}.{fmt}"
        try:
            await exporter(name, out, neutralize="all")
        except ValueError as exc:
            if str(exc).startswith("Table not found"):
                return _error(NO_TABLE_MESSAGE, 404)
            return _error(str(exc)[:200], 400)
        body = await asyncio.to_thread(out.read_bytes)
    return web.Response(body=body, content_type="application/octet-stream")


def _json_dumps(obj: Any) -> str:
    import json

    return json.dumps(obj, default=str)


# ── wiring ───────────────────────────────────────────────────────────


def register_speaker_routes(app: web.Application, server: Any) -> None:
    """The member routes (contract part 0b §2.2) — reachable only through
    :func:`create_speaker_http_middleware`."""
    app.router.add_get("/api/speaker/files", list_files)
    app.router.add_get("/api/speaker/files/raw", raw_file)
    app.router.add_post("/api/speaker/files/upload", make_upload_handler(server))
    app.router.add_post("/api/speaker/files/delete", delete_file)
    app.router.add_get("/api/speaker/datastore/tables", list_tables)
    app.router.add_get("/api/speaker/datastore/tables/{name}/rows", table_rows)
    app.router.add_get("/api/speaker/datastore/tables/{name}/export", export_table)
