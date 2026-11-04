"""Public file links — publish ONE agent file to anyone holding the link.

Two halves, deliberately kept apart:

* **Owner side** (signed in): ``/fd/agent-file-share/{host}/{port}`` copies one
  file out of the agent into ``<data dir>/public_shares/<token>`` and mints an
  unguessable token. The path is port-addressed (``/fd/agent-…/{host}/{port}``)
  so the server's ownership guard applies exactly as it does to downloads: a
  user can only publish a file of an agent they own, and a shared-agent member
  (speaker grant) is refused before reaching it. ``/fd/public-shares`` lists
  and revokes the caller's links.

* **Public side** (no account): ``/s/{token}`` and its ``/raw`` + ``/download``.
  It reads ONLY the snapshot the token names. It never contacts an agent, takes
  no path or other input besides the token, and its responses name nothing but
  the file itself — not the owner, the agent, the folder it came from, or the
  deck. Markdown is rendered server-side with raw HTML escaped; the wrapper page
  runs no script at all (``default-src 'none'``). Every document that can run
  script (HTML, SVG, anything XML-ish) is served under a CSP ``sandbox``
  without ``allow-same-origin``, so it runs on an opaque origin and can't read
  Flight Deck's cookies, storage or API even though it shares the host.

A snapshot, not a live view: later changes in the agent never reach the link
until the owner refreshes it, so an agent rewriting the file can't leak new
content through an old link. ``FD_PUBLIC_FILE_LINKS=off`` turns the feature
off deck-wide, existing links included.
"""
from __future__ import annotations

import html
import mimetypes
import os
import re
import secrets
import time
import urllib.parse
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, Response
from pydantic import BaseModel

from captain_claw.flight_deck.auth import get_current_user, get_db
from captain_claw.logging import get_logger

log = get_logger(__name__)

owner_router = APIRouter(tags=["public-file-links"])
public_router = APIRouter(tags=["public-file-links-public"])

_TOKEN_RE = re.compile(r"^[A-Za-z0-9_-]{20,64}$")
# Every link expires: new ones after FD_PUBLIC_FILE_LINK_DAYS (7), at most
# FD_PUBLIC_FILE_LINK_MAX_DAYS (30). A max of 0 lets links never expire, as
# they all once did — and test users' client files and medical reports sat on
# links nobody remembered (cohort001). 0 in a request means "never".
_EXPIRY_CHOICES = (1, 7, 30)
_PREVIEW_MAX_BYTES = 2 * 1024 * 1024  # text/markdown rendered inline up to this

_data_dir: Path | None = None
_resolve_auth: Callable[..., str] = lambda _port, host=None: ""  # noqa: E731
_resolve_agent_endpoint: Callable[[str], dict | None] = lambda _token: None  # noqa: E731


def configure(*, data_dir: Path, resolve_auth: Callable[..., str],
              resolve_agent_endpoint: Callable[[str], dict | None] | None = None) -> None:
    """Wire the deck's data dir (snapshots live beside its DB), the agent auth
    resolver the download proxy uses (``resolve_auth(port, host=...)``: "" when
    ``host`` isn't the agent's own address), and — for the agent routes — the
    lookup from an agent's web_auth to ``{"port", "agent_ref", "name"}``."""
    global _data_dir, _resolve_auth, _resolve_agent_endpoint
    _data_dir = Path(data_dir)
    _resolve_auth = resolve_auth
    if resolve_agent_endpoint is not None:
        _resolve_agent_endpoint = resolve_agent_endpoint


def _env_days(name: str, default: int) -> int:
    try:
        return max(0, int(os.environ.get(name, "").strip() or default))
    except ValueError:
        return default


def max_days() -> int:
    """The longest a link may live; 0 = links may never expire."""
    return _env_days("FD_PUBLIC_FILE_LINK_MAX_DAYS", 30)


def default_days() -> int:
    """How long a new link lives unless its owner picks otherwise."""
    days, cap = _env_days("FD_PUBLIC_FILE_LINK_DAYS", 7), max_days()
    if cap and (not days or days > cap):
        return cap
    return days


def expiry_options() -> list[int]:
    """The expiries a link may be given, in days (0 = never, only uncapped)."""
    cap = max_days()
    out = [d for d in _EXPIRY_CHOICES if not cap or d <= cap]
    for extra in (cap, default_days()):
        if extra and extra not in out:
            out.append(extra)
    out.sort()
    return out + ([0] if not cap else [])


def _until(days: int, now: datetime) -> str | None:
    return (now + timedelta(days=days)).isoformat() if days else None


def enabled() -> bool:
    raw = os.environ.get("FD_PUBLIC_FILE_LINKS", "on").strip().lower()
    return raw not in ("0", "off", "false", "no")


def _max_bytes() -> int:
    try:
        mb = float(os.environ.get("FD_PUBLIC_FILE_LINK_MAX_MB", "50"))
    except ValueError:
        mb = 50.0
    return int(max(1.0, mb) * 1024 * 1024)


def _public_base() -> str:
    return os.environ.get("FD_PUBLIC_URL", "").strip().rstrip("/")


def _store_dir() -> Path:
    if _data_dir is None:
        raise HTTPException(503, "Public links are not configured on this deck")
    d = _data_dir / "public_shares"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _snapshot_path(token: str) -> Path:
    return _store_dir() / token


# ── File kinds (mirrors flight-deck's getFileTypeGroup) ──────────────

_IMAGE_TYPES = {
    ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".gif": "image/gif",
    ".webp": "image/webp", ".svg": "image/svg+xml", ".ico": "image/x-icon", ".bmp": "image/bmp",
}
_AUDIO_TYPES = {
    ".mp3": "audio/mpeg", ".wav": "audio/wav", ".ogg": "audio/ogg", ".flac": "audio/flac",
    ".aac": "audio/aac", ".m4a": "audio/mp4", ".opus": "audio/ogg",
}
_VIDEO_TYPES = {
    ".mp4": "video/mp4", ".webm": "video/webm", ".mov": "video/quicktime",
    ".avi": "video/x-msvideo", ".mkv": "video/x-matroska",
}
_TEXT_KINDS = {"markdown", "data", "code", "text"}
_EXT_KIND = {
    **{e: "image" for e in _IMAGE_TYPES},
    **{e: "audio" for e in _AUDIO_TYPES},
    **{e: "video" for e in _VIDEO_TYPES},
    ".pdf": "pdf", ".html": "html", ".htm": "html", ".md": "markdown", ".markdown": "markdown",
    **{e: "data" for e in (".json", ".yaml", ".yml", ".toml", ".xml", ".csv", ".tsv")},
    **{e: "code" for e in (".js", ".ts", ".jsx", ".tsx", ".py", ".rb", ".go", ".rs", ".java",
                           ".c", ".cpp", ".h", ".css", ".scss", ".sh", ".sql")},
    **{e: "text" for e in (".txt", ".log", ".ini", ".cfg", ".conf", ".env")},
}


def _ext(filename: str) -> str:
    dot = filename.rfind(".")
    return filename[dot:].lower() if dot > 0 else ""


def file_kind(filename: str) -> str:
    return _EXT_KIND.get(_ext(filename), "other")


# A document able to run script, on any origin: sandboxed WITHOUT
# allow-same-origin, so it gets an opaque origin of its own.
_HTML_RAW_CSP = ("sandbox allow-scripts allow-popups allow-popups-to-escape-sandbox "
                 "allow-modals allow-downloads allow-forms; frame-ancestors 'self'")
# Everything else that isn't a PDF: no script, no same-origin, ever.
_INERT_CSP = "sandbox; default-src 'none'; img-src data:; media-src 'self'; style-src 'unsafe-inline'"
# The wrapper page: no script of its own, frames only this deck's /raw.
_PAGE_CSP = ("default-src 'none'; style-src 'unsafe-inline'; img-src 'self' data: https:; "
             "media-src 'self'; frame-src 'self'; base-uri 'none'; form-action 'none'; "
             "frame-ancestors 'none'")

_COMMON_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "X-Robots-Tag": "noindex, nofollow, noarchive",
    # Revoking or refreshing a link must take effect at once.
    "Cache-Control": "no-store",
}


def _raw_type(kind: str, filename: str) -> tuple[str, str | None]:
    """(Content-Type, CSP) for the inline /raw view. Text-like kinds are always
    text/plain, so an .xml or .svg-in-disguise can never render as a document."""
    ext = _ext(filename)
    if kind == "html":
        return "text/html; charset=utf-8", _HTML_RAW_CSP
    if kind == "pdf":
        # Chrome's PDF viewer refuses a sandboxed document; a PDF's own script
        # runs inside the viewer, never as this origin.
        return "application/pdf", "frame-ancestors 'self'"
    if kind == "image":
        return _IMAGE_TYPES.get(ext, "application/octet-stream"), _INERT_CSP
    if kind == "audio":
        return _AUDIO_TYPES.get(ext, "application/octet-stream"), _INERT_CSP
    if kind == "video":
        return _VIDEO_TYPES.get(ext, "application/octet-stream"), _INERT_CSP
    if kind in _TEXT_KINDS:
        return "text/plain; charset=utf-8", _INERT_CSP
    return "application/octet-stream", _INERT_CSP


def _download_type(kind: str, filename: str) -> str:
    if kind in ("image", "audio", "video", "pdf"):
        return _raw_type(kind, filename)[0]
    guessed, _ = mimetypes.guess_type(filename)
    return guessed or "application/octet-stream"


def _disposition(kind: str, filename: str) -> str:
    quoted = urllib.parse.quote(filename, safe="")
    ascii_name = re.sub(r'[^A-Za-z0-9._ -]', "_", filename) or "file"
    return f'{kind}; filename="{ascii_name}"; filename*=UTF-8\'\'{quoted}'


def _clean_filename(path: str) -> str:
    name = path.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
    name = re.sub(r"[\x00-\x1f\x7f]", "", name).strip()
    return name[:200] or "file"


# ── Rows ─────────────────────────────────────────────────────────────

def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _is_expired(row: dict) -> bool:
    exp = row.get("expires_at")
    if not exp:
        return False
    try:
        return datetime.fromisoformat(exp) <= _utcnow()
    except ValueError:
        return True


def _owner_view(row: dict) -> dict:
    """What the OWNER sees about one of their links (never sent publicly)."""
    return {
        "token": row["token"],
        "path": f"/s/{row['token']}",
        "filename": row["filename"],
        "kind": row["kind"],
        "size": row["size"],
        "agent_ref": row["agent_ref"],
        "agent_name": row["agent_name"],
        "source_path": row["source_path"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "expires_at": row["expires_at"],
        "expired": _is_expired(row),
        "view_count": row["view_count"],
        "last_viewed_at": row["last_viewed_at"],
    }


async def _drop(db, token: str) -> None:
    await db.delete_public_file_share(token)
    try:
        _snapshot_path(token).unlink(missing_ok=True)
    except OSError:
        pass


async def _sweep(db) -> None:
    """Expire links past their date and delete snapshots no row names (a
    revoked link, a removed user's links, an interrupted copy)."""
    rows = await db.list_public_file_shares()
    for r in rows:
        if _is_expired(r):
            await _drop(db, r["token"])
    live = {r["token"] for r in rows if not _is_expired(r)}
    cutoff = time.time() - 3600
    for f in _store_dir().iterdir():
        try:
            if f.name.startswith("."):
                if f.stat().st_mtime < cutoff:   # a copy that never finished
                    f.unlink(missing_ok=True)
            elif f.name not in live:
                f.unlink(missing_ok=True)
        except OSError:
            pass


async def cap_open_links(db) -> int:
    """Give every link that never expires an end date, once the deck caps them
    (``max_days``): its creation plus the cap, but at least a week from now so
    nobody's link vanishes unannounced. Each owner is told on the Line, with
    the date. Returns how many links got a date."""
    cap = max_days()
    if not cap or not enabled():
        return 0
    now = _utcnow()
    floor = now + timedelta(days=min(7, cap))
    told: dict[str, list[tuple[str, str]]] = {}
    for row in await db.list_public_file_shares():
        if row.get("expires_at"):
            continue
        try:
            created = datetime.fromisoformat(row["created_at"])
        except (TypeError, ValueError):
            created = now
        until = max(created + timedelta(days=cap), floor).isoformat()
        await db.save_public_file_share({**row, "expires_at": until})
        told.setdefault(row["owner_id"], []).append((row["filename"], until))
    for owner, links in told.items():
        try:
            from captain_claw.flight_deck import line

            first = min(u for _, u in links)[:10]
            names = ", ".join(f for f, _ in links[:5]) + (" and more" if len(links) > 5 else "")
            await line.post(
                owner, "deck",
                f"Your {len(links)} public link{'s' if len(links) != 1 else ''} now expire",
                f"Public links on this deck now expire after at most {cap} days. Yours ({names}) "
                f"stop working from {first}. To keep one, open it under Share and pick how long; "
                "to end one now, choose Stop sharing.",
                bell_type="info",
            )
        except Exception as exc:
            log.debug("Public link owner not told", error=type(exc).__name__)
    n = sum(len(v) for v in told.values())
    if n:
        log.info("Public links that never expired got an end date", links=n, cap_days=cap)
    return n


# ── Owner API ────────────────────────────────────────────────────────

class ShareRequest(BaseModel):
    path: str
    agent_auth: str = ""       # the agent's web auth, as the download proxy takes it
    agent_ref: str = ""        # the client's agent id — lets the owner find the link again
    agent_name: str = ""
    expires_days: int | None = None   # None: keep the link's expiry (new: default_days())


def _config() -> dict:
    return {"enabled": enabled(), "max_mb": round(_max_bytes() / (1024 * 1024)),
            "public_base": _public_base(), "expiry_days": expiry_options(),
            "default_days": default_days()}


def _require_enabled() -> None:
    if not enabled():
        raise HTTPException(403, "Public links are turned off on this deck")


@owner_router.get("/fd/public-shares/config")
async def public_share_config(user: dict = Depends(get_current_user)):
    return _config()


@owner_router.post("/fd/public-shares/origin")
async def report_browser_origin(request: Request, user: dict = Depends(get_current_user)):
    """The SPA's "this is where people reach me" — see note_browser_origin."""
    return {"origin": await note_browser_origin(request, user["id"]) or None}


@owner_router.get("/fd/public-shares")
async def list_my_public_shares(user: dict = Depends(get_current_user)):
    db = get_db()
    await _sweep(db)
    rows = await db.list_public_file_shares(user["id"])
    return {"shares": [_owner_view(r) for r in rows], **_config()}


@owner_router.delete("/fd/public-shares/{token}")
async def revoke_public_share(token: str, user: dict = Depends(get_current_user)):
    db = get_db()
    row = await db.get_public_file_share(token) if _TOKEN_RE.match(token) else None
    if not row or (row["owner_id"] != user["id"] and user.get("role") != "admin"):
        raise HTTPException(404, "No such link")
    await _drop(db, token)
    log.info("Public file link revoked", token=token[:6], owner=row["owner_id"])
    return {"ok": True}


@owner_router.get("/fd/agent-file-share/{host}/{port}")
async def get_agent_file_share(host: str, port: int, path: str, agent_ref: str = "",
                               user: dict = Depends(get_current_user)):
    """The caller's existing link to this file, if any (for the Share dialog)."""
    db = get_db()
    row = await db.find_public_file_share(user["id"], agent_ref, path)
    if row and _is_expired(row):
        await _drop(db, row["token"])
        row = None
    return {"share": _owner_view(row) if row else None, **_config()}


@owner_router.post("/fd/agent-file-share/{host}/{port}")
async def create_agent_file_share(host: str, port: int, req: ShareRequest, request: Request,
                                  user: dict = Depends(get_current_user)):
    """Copy the file out of the agent and publish it (or refresh the copy and
    expiry of the caller's existing link to it — the URL stays the same)."""
    await note_browser_origin(request, user["id"])
    # A resolved token only for the agent's own address, never a host the caller named.
    agent_auth = req.agent_auth or _resolve_auth(port, host=host)
    row = await _publish(owner=user["id"], agent_ref=req.agent_ref, agent_name=req.agent_name,
                         host=host, port=port, agent_auth=agent_auth,
                         path=req.path, expires_days=req.expires_days, via="ui")
    return {"share": _owner_view(row), **_config()}


async def _publish(*, owner: str, agent_ref: str, agent_name: str, host: str, port: int,
                   agent_auth: str, path: str, expires_days: int | None, via: str) -> dict:
    """Snapshot ``path`` from the agent and save the owner's link to it — a new
    token, or the existing link's token when this owner already shared this
    file from this agent (the URL stays the same)."""
    _require_enabled()
    path = path.strip()
    if not path:
        raise HTTPException(400, "Missing path")
    if expires_days is not None and expires_days not in expiry_options():
        raise HTTPException(400, f"expires_days must be one of {expiry_options()} "
                                 "(0 = never, where this deck allows it)")
    db = get_db()
    await _sweep(db)

    existing = await db.find_public_file_share(owner, agent_ref, path)
    token = existing["token"] if existing else secrets.token_urlsafe(24)
    filename = _clean_filename(path)
    size = await _snapshot_from_agent(host, port, agent_auth, path, token)

    now = _utcnow()
    if expires_days is None:
        expires = existing["expires_at"] if existing else _until(default_days(), now)
        if existing and not expires and max_days():
            expires = _until(default_days(), now)  # a link from before every link expired
    else:
        expires = _until(expires_days, now)
    row = await db.save_public_file_share({
        "token": token,
        "owner_id": owner,
        "agent_ref": agent_ref[:200],
        "agent_name": agent_name[:200],
        "source_path": path,
        "filename": filename,
        "kind": file_kind(filename),
        "size": size,
        "created_at": existing["created_at"] if existing else now.isoformat(),
        "updated_at": now.isoformat(),
        "expires_at": expires,
    })
    log.info("Public file link saved", token=token[:6], owner=owner, kind=row["kind"],
             size=size, refreshed=bool(existing), via=via)
    return row


# ── Agent API (the `public_link` tool) ───────────────────────────────
#
# The calling agent proves who it is with X-Agent-Auth (its own web_auth, which
# this deck minted): require_agent_caller refuses browsers, unknown tokens and
# callers without the transport secret. The link belongs to that agent's OWNER,
# keyed by the same agent_ref the SPA uses, so it shows up in the owner's Share
# dialog and "All your public links". The file is pulled through the agent's
# own /api/files/download, so the agent can publish only what its Files panel
# could already hand out (saved/, output/, workflows/, or a registered file).
# An agent sees and stops only its own links. A shared-agent member's turn
# never gets here: the tool isn't on the member allowlist, and member markers
# are refused off the grant-aware routes.

class AgentShareRequest(BaseModel):
    path: str = ""
    expires_days: int | None = None
    token: str = ""            # unshare: the link's token (from list), instead of path


def _agent_caller(request: Request) -> tuple[str, str, dict]:
    """(owner, agent web_auth, endpoint) for the agent making this request."""
    from captain_claw.flight_deck.agent_identity import require_agent_caller
    from captain_claw.flight_deck.auth import _fd_auth_enabled

    caller = require_agent_caller(request, what="public file links")
    auth = request.headers.get("X-Agent-Auth", "")
    endpoint = _resolve_agent_endpoint(auth)
    if not endpoint:
        raise HTTPException(403, "Flight Deck can't find this agent's file server")
    owner = caller.owner or ("" if _fd_auth_enabled() else "local")
    if not owner:
        raise HTTPException(403, "This agent has no owner on record, so it can't publish links")
    return owner, auth, endpoint


def _agent_view(row: dict, base: str) -> dict:
    """What the agent gets back: enough to tell the user, with an absolute URL."""
    host = urllib.parse.urlsplit(base).hostname or ""
    return {
        **_owner_view(row),
        "url": f"{base}/s/{row['token']}",
        # Only reachable where this host resolves: nothing better than the
        # address the agent itself used (loopback / private) was known.
        "local_only": _private_host(host),
    }


# ── The deck's public address, for links an agent makes ──────────────
#
# A browser shows a link under its own window.location; an agent only knows the
# deck's internal URL (FD_URL, e.g. http://localhost:8765). So the SPA reports
# the Origin it runs on (POST /fd/public-shares/origin, once it's signed in)
# and agent links use, in order: FD_PUBLIC_URL; that owner's browser origin;
# the origin any user's browser was seen on most recently; a concrete domain
# from FD_ALLOWED_HOSTS (as https); the agent's own base URL (flagged
# local_only). A learned origin must be one this deck answers to
# (origin_guard.host_allowed) and not a loopback / private address.

_origin_cache: dict[str, str] = {}


def _parse_origin(value: str) -> tuple[str, str] | None:
    """(normalised origin, hostname) of an http(s) Origin / URL, else None."""
    try:
        u = urllib.parse.urlsplit(str(value or "").strip())
    except ValueError:
        return None
    if u.scheme not in ("http", "https") or not u.hostname:
        return None
    try:
        port = u.port
    except ValueError:
        return None
    default = 443 if u.scheme == "https" else 80
    host = u.hostname.lower()
    shown = f"[{host}]" if ":" in host else host
    return f"{u.scheme}://{shown}{'' if port in (None, default) else f':{port}'}", host


async def note_browser_origin(request: Request, owner: str) -> str:
    """Remember the public Origin ``owner``'s browser reached the deck at;
    returns it ("" when the request carries none worth keeping)."""
    from captain_claw.flight_deck import origin_guard

    parsed = _parse_origin(request.headers.get("origin", ""))
    if not parsed or not owner:
        return ""
    origin, host = parsed
    if _private_host(host) or not origin_guard.host_allowed(host):
        return ""
    if _origin_cache.get(owner) != origin:
        await get_db().set_public_link_origin(owner, origin)
        _origin_cache[owner] = origin
    return origin


def _allowed_hosts_base() -> str:
    """https://<first concrete domain in FD_ALLOWED_HOSTS>, else ""."""
    for raw in os.environ.get("FD_ALLOWED_HOSTS", "").split(","):
        host = raw.strip().lower()
        if host and "*" not in host and not host.startswith(".") and not _private_host(host):
            return f"https://{host}"
    return ""


async def _agent_link_base(request: Request, owner: str) -> str:
    if _public_base():
        return _public_base()
    db = get_db()
    learned = await db.get_public_link_origin(owner) or await db.get_public_link_origin()
    return learned or _allowed_hosts_base() or str(request.base_url).rstrip("/")


def _private_host(host: str) -> bool:
    import ipaddress

    if host in ("localhost", "") or host.endswith((".localhost", ".local")):
        return True
    try:
        return ipaddress.ip_address(host).is_private
    except ValueError:
        return False


@owner_router.post("/fd/public-shares/agent/share")
async def agent_share(req: AgentShareRequest, request: Request):
    owner, auth, ep = _agent_caller(request)
    row = await _publish(owner=owner, agent_ref=ep["agent_ref"], agent_name=ep["name"],
                         host="localhost", port=ep["port"], agent_auth=auth,
                         path=req.path, expires_days=req.expires_days, via="agent")
    return {"share": _agent_view(row, await _agent_link_base(request, owner))}


@owner_router.post("/fd/public-shares/agent/list")
async def agent_list(request: Request):
    owner, _auth, ep = _agent_caller(request)
    db = get_db()
    await _sweep(db)
    rows = [r for r in await db.list_public_file_shares(owner) if r["agent_ref"] == ep["agent_ref"]]
    base = await _agent_link_base(request, owner)
    return {"shares": [_agent_view(r, base) for r in rows], "enabled": enabled()}


@owner_router.post("/fd/public-shares/agent/unshare")
async def agent_unshare(req: AgentShareRequest, request: Request):
    owner, _auth, ep = _agent_caller(request)
    db = get_db()
    if req.token:
        row = await db.get_public_file_share(req.token) if _TOKEN_RE.match(req.token) else None
    else:
        row = await db.find_public_file_share(owner, ep["agent_ref"], req.path.strip())
    if not row or row["owner_id"] != owner or row["agent_ref"] != ep["agent_ref"]:
        raise HTTPException(404, "No public link from this agent for that file")
    await _drop(db, row["token"])
    log.info("Public file link revoked", token=row["token"][:6], owner=owner, via="agent")
    return {"ok": True, "filename": row["filename"]}


async def _snapshot_from_agent(host: str, port: int, auth: str, path: str, token: str) -> int:
    """Stream the file from the agent into the token's snapshot, capped."""
    limit = _max_bytes()
    params = {"path": path}
    if auth:
        params["token"] = auth
    dest = _snapshot_path(token)
    tmp = dest.with_name(f".{token}.{secrets.token_hex(4)}.part")
    written = 0
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream("GET", f"http://{host}:{port}/api/files/download",
                                     params=params) as resp:
                if resp.status_code != 200:
                    await resp.aread()
                    # Never pass an agent 401 through: to the browser a 401
                    # means ITS session expired, and it would sign the user out.
                    status = resp.status_code if resp.status_code in (403, 404) else 502
                    raise HTTPException(status, f"The agent refused the file ({resp.status_code})")
                declared = resp.headers.get("content-length", "")
                if declared.isdigit() and int(declared) > limit:
                    raise HTTPException(413, f"File is larger than the {limit // (1024 * 1024)} MB limit")
                with tmp.open("wb") as out:
                    async for chunk in resp.aiter_bytes():
                        written += len(chunk)
                        if written > limit:
                            raise HTTPException(
                                413, f"File is larger than the {limit // (1024 * 1024)} MB limit")
                        out.write(chunk)
        os.replace(tmp, dest)
        return written
    except httpx.HTTPError as exc:
        raise HTTPException(502, "Cannot reach the agent") from exc
    finally:
        tmp.unlink(missing_ok=True)


# ── Public side ──────────────────────────────────────────────────────

async def _live_row(token: str) -> dict | None:
    """The row for a token that is valid, unexpired, and has its snapshot."""
    if not enabled() or not _TOKEN_RE.match(token):
        return None
    db = get_db()
    row = await db.get_public_file_share(token)
    if not row:
        return None
    if _is_expired(row):
        await _drop(db, token)
        return None
    if not _snapshot_path(token).is_file():
        return None
    return row


def _gone() -> HTMLResponse:
    # One answer for unknown, revoked and expired alike: it says nothing about
    # whether a link ever existed.
    body = _page_shell(
        "Link unavailable", "gone",
        '<div class="center"><div class="card"><h1>This link isn\'t available</h1>'
        "<p>It may have been turned off or expired.</p></div></div>", bar="")
    return HTMLResponse(body, status_code=404, headers={**_COMMON_HEADERS,
                        "Content-Security-Policy": _PAGE_CSP, "X-Frame-Options": "DENY"})


@public_router.get("/s/{token}")
async def public_share_page(token: str):
    row = await _live_row(token)
    if not row:
        return _gone()
    try:
        await get_db().note_public_file_share_view(token)
    except Exception:  # a view count is never worth failing the page for
        pass
    return HTMLResponse(_render_page(row), headers={
        **_COMMON_HEADERS, "Content-Security-Policy": _PAGE_CSP, "X-Frame-Options": "DENY"})


@public_router.get("/s/{token}/raw")
async def public_share_raw(token: str):
    row = await _live_row(token)
    if not row:
        return _gone()
    ctype, csp = _raw_type(row["kind"], row["filename"])
    headers = {**_COMMON_HEADERS, "Content-Disposition": _disposition("inline", row["filename"])}
    if csp:
        headers["Content-Security-Policy"] = csp
    if row["kind"] == "other":
        headers["Content-Disposition"] = _disposition("attachment", row["filename"])
    return FileResponse(_snapshot_path(token), media_type=ctype, headers=headers)


@public_router.get("/s/{token}/download")
async def public_share_download(token: str):
    row = await _live_row(token)
    if not row:
        return _gone()
    headers = {**_COMMON_HEADERS, "Content-Security-Policy": _INERT_CSP,
               "Content-Disposition": _disposition("attachment", row["filename"])}
    return FileResponse(_snapshot_path(token),
                        media_type=_download_type(row["kind"], row["filename"]), headers=headers)


# ── Rendering ────────────────────────────────────────────────────────

def _fmt_size(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n / (1024 * 1024):.1f} MB"


def _markdown_html(text: str) -> str:
    """CommonMark + tables, raw HTML escaped (never passed through), links open
    in a new tab without a referrer. markdown-it's link validator already drops
    javascript:/vbscript:/file: and non-image data: URLs."""
    from markdown_it import MarkdownIt

    md = MarkdownIt("commonmark", {"html": False, "linkify": False, "typographer": False})
    md.enable(["table", "strikethrough"])

    def _link_open(self, tokens, idx, options, env):
        tokens[idx].attrSet("target", "_blank")
        tokens[idx].attrSet("rel", "noopener noreferrer nofollow")
        return self.renderToken(tokens, idx, options, env)

    md.add_render_rule("link_open", _link_open)
    return md.render(text)


def _read_text(path: Path) -> str:
    return path.read_bytes().decode("utf-8", errors="replace")


def _render_page(row: dict) -> str:
    token, kind, name = row["token"], row["kind"], row["filename"]
    base = f"/s/{token}"
    esc_name = html.escape(name)
    raw, dl = f"{base}/raw", f"{base}/download"
    snap = _snapshot_path(token)
    too_big = row["size"] > _PREVIEW_MAX_BYTES

    open_full = (f'<a class="btn" href="{raw}" target="_blank" rel="noopener noreferrer">'
                 "Open full page</a>") if kind in ("html", "pdf") else ""
    bar = (f'<header class="bar"><div class="title"><span class="name" title="{esc_name}">'
           f'{esc_name}</span><span class="meta">{_fmt_size(row["size"])}</span></div>'
           f'<nav>{open_full}<a class="btn primary" href="{dl}">Download</a></nav></header>')

    def card(msg: str) -> str:
        return (f'<div class="center"><div class="card"><div class="big">{esc_name}</div>'
                f'<p class="muted">{_fmt_size(row["size"])}</p><p>{msg}</p>'
                f'<a class="btn primary" href="{dl}">Download</a></div></div>')

    if kind == "markdown" and not too_big:
        body = f'<main class="doc"><article class="md">{_markdown_html(_read_text(snap))}</article></main>'
    elif kind in _TEXT_KINDS and not too_big:
        body = f'<main class="doc"><pre class="text">{html.escape(_read_text(snap))}</pre></main>'
    elif kind in _TEXT_KINDS:
        body = card("This file is too large to preview here.")
    elif kind == "html":
        # The sandbox comes from /raw's CSP header, which binds the document
        # however it's loaded (framed here, or opened directly). No iframe
        # sandbox attribute on top: some embedded browsers refuse to load any
        # frame that carries one, and the header already gives the same jail.
        body = (f'<main class="fill"><iframe src="{raw}" title="{esc_name}" '
                'referrerpolicy="no-referrer"></iframe></main>')
    elif kind == "pdf":
        body = f'<main class="fill"><iframe src="{raw}" title="{esc_name}"></iframe></main>'
    elif kind == "image":
        body = f'<main class="center"><img src="{raw}" alt="{esc_name}"></main>'
    elif kind == "video":
        body = (f'<main class="center"><video src="{raw}" controls playsinline preload="metadata">'
                "</video></main>")
    elif kind == "audio":
        body = (f'<main class="center"><div class="card"><div class="big">{esc_name}</div>'
                f'<audio src="{raw}" controls preload="metadata"></audio></div></main>')
    else:
        body = card("This file type can't be previewed in the browser.")
    return _page_shell(name, kind, body, bar=bar)


def _page_shell(title: str, kind: str, body: str, *, bar: str) -> str:
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="robots" content="noindex, nofollow, noarchive">
<meta name="referrer" content="no-referrer">
<title>{html.escape(title)}</title>
<style>{_CSS}</style></head>
<body class="k-{kind}">{bar}{body}</body></html>"""


_CSS = """
:root{--bg:#fafafa;--panel:#ffffff;--fg:#18181b;--muted:#71717a;--line:#e4e4e7;
--accent:#7c3aed;--accent-fg:#fff;--code:#f4f4f5}
@media (prefers-color-scheme:dark){:root{--bg:#09090b;--panel:#18181b;--fg:#e4e4e7;
--muted:#a1a1aa;--line:#27272a;--accent:#8b5cf6;--code:#1f1f23}}
*{box-sizing:border-box}
html,body{margin:0;background:var(--bg);color:var(--fg);
font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
body{min-height:100vh;display:flex;flex-direction:column}
.bar{position:sticky;top:0;z-index:1;display:flex;align-items:center;gap:12px;
padding:10px 16px;background:var(--panel);border-bottom:1px solid var(--line)}
.title{min-width:0;flex:1;display:flex;align-items:baseline;gap:10px}
.name{font-weight:600;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.meta{color:var(--muted);font-size:13px;white-space:nowrap}
nav{display:flex;gap:8px;flex-shrink:0}
.btn{display:inline-block;padding:6px 12px;border-radius:8px;border:1px solid var(--line);
color:var(--fg);text-decoration:none;font-size:13px;font-weight:500;white-space:nowrap}
.btn:hover{border-color:var(--muted)}
.btn.primary{background:var(--accent);border-color:var(--accent);color:var(--accent-fg)}
.doc{flex:1;width:100%;max-width:860px;margin:0 auto;padding:28px 16px 64px}
.fill{flex:1;display:flex}
.fill iframe{flex:1;width:100%;min-height:calc(100vh - 54px);border:0;background:#fff}
.center{flex:1;display:flex;align-items:center;justify-content:center;padding:24px 16px}
.center img,.center video{max-width:100%;max-height:calc(100vh - 110px);border-radius:6px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:28px;
text-align:center;max-width:460px;width:100%}
.card h1{font-size:20px;margin:0 0 6px}.card p{margin:8px 0 16px}
.card audio{width:100%;margin-top:16px}
.big{font-weight:600;font-size:17px;word-break:break-all}
.muted{color:var(--muted)}
pre.text{white-space:pre-wrap;word-break:break-word;margin:0;padding:16px;
background:var(--panel);border:1px solid var(--line);border-radius:10px;
font:13px/1.55 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
.md{word-wrap:break-word}
.md h1,.md h2,.md h3,.md h4{line-height:1.25;margin:1.6em 0 .6em}
.md h1{font-size:2em;margin-top:0}.md h2{font-size:1.5em;border-bottom:1px solid var(--line);padding-bottom:.3em}
.md a{color:var(--accent)}
.md img{max-width:100%}
.md code{background:var(--code);padding:.15em .35em;border-radius:4px;
font:.9em ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
.md pre{background:var(--code);padding:14px;border-radius:8px;overflow:auto}
.md pre code{background:none;padding:0}
.md blockquote{margin:0;padding:0 1em;color:var(--muted);border-left:3px solid var(--line)}
.md table{border-collapse:collapse;display:block;overflow:auto;max-width:100%}
.md th,.md td{border:1px solid var(--line);padding:6px 12px}
.md th{background:var(--code)}
.md hr{border:0;border-top:1px solid var(--line);margin:2em 0}
@media (max-width:560px){.meta{display:none}.bar{padding:8px 12px}}
"""
