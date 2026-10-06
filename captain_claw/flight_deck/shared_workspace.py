"""Shared-agent workspace (PR C) — a member's files and data on a shared agent.

A shared PROCESS agent's saved files (everything under its ``saved/`` folder)
and its datastore are a commons for its members: they see everything there,
add their own files, tables and rows, and change only what they added (the
owner changes all of it). Every item shows its creator. Docker agents stay
chat-only.

FD never reaches the agent's owner routes (``/api/files*``, ``/api/datastore*``)
for a member. It calls the agent's member routes (``/api/speaker/*``) with the
token it recorded at spawn plus a FRESH ``X-FD-Speaker`` assertion per request,
bound to the method and path (``aud: "http"``, ``m``, ``p``), and relays a
response only when it carries ``X-FD-Speaker-Ack`` for exactly that header —
an agent too old to verify it (which would answer as if to its owner) gets
nothing relayed and the member sees "needs a restart". The agent enforces
who may change what; FD's ``can_delete`` / ``me`` flags are for display.

This module holds the pure logic behind ``shared_workspace_routes``:
validation, response headers, the per-member rate limit, the membership and
target check, the agent client, creator names (resolved from FD's ``users``
table — never another member's id or email to a member) and the decoration of
the owner proxies' creator badges. Plus the one-time bell telling owners their
agents' ``saved/`` folders and datastores are now open to members.

Off unless agent sharing is active (``FD_AGENT_SHARING`` + auth). Never logs a
token, an assertion, a JWT, a query string or a body.
"""

from __future__ import annotations

import asyncio
import collections
import hmac
import re
import secrets
import time
import unicodedata
import urllib.parse
from datetime import UTC, datetime

import httpx
from fastapi import HTTPException

from captain_claw.flight_deck import agent_sharing as sharing
from captain_claw.flight_deck import context_packs, tenant_profile
from captain_claw.logging import get_logger

log = get_logger(__name__)

# The agent token travels as ``?token=`` and httpx logs request URLs at INFO.
context_packs._install_httpx_redaction()

# ── Constants (contract part 0b §4, part 1 §1) ────────────────────────────

SPEAKER_ACK_HEADER = "X-FD-Speaker-Ack"
HTTP_ASSERTION_AUD = "http"
SPEAKER_HTTP_PREFIX = "/api/speaker/"
MEMBER_UPLOAD_MAX_BYTES = 25 * 1024 * 1024        # 26214400 — FD and agent
MEMBER_DOWNLOAD_MAX_BYTES = 50 * 1024 * 1024      # agent raw; FD view/download
MEMBER_UPLOAD_EXTENSIONS = (".csv", ".xlsx", ".xls", ".pdf", ".docx", ".doc", ".pptx", ".ppt",
                            ".md", ".txt", ".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp",
                            ".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v")
MEMBER_HTTP_RATE_PER_MIN = 120                    # per (agent_ref, member)
TABLE_NAME_RE = r"^[a-z0-9_]{1,128}$"
ORDER_BY_RE = r"^_?[a-z0-9_]{1,128}$"
FILE_ID_MAX = 1024
FORMER_MEMBER = "Former member"

AGENT_TIMEOUT_S = 15.0            # list / tables / rows / delete
AGENT_TRANSFER_TIMEOUT_S = 90.0   # raw / upload / export
RELAYED_STATUSES = frozenset({400, 403, 404, 409, 413})   # = the agent's 4xx set, 0b §2.2
DETAIL_MAX = 300
EXPORT_FORMATS = ("csv", "json", "xlsx")
SHARING_OFF_DETAIL = "Agent sharing is off on this Flight Deck"
BAD_REF_DETAIL     = "Invalid agent reference"
NOT_FOUND_DETAIL   = "Agent not found"
OWNER_DETAIL       = "You own this agent — use your own Files and Datastore panels"
NOT_SHARED_DETAIL  = "This agent isn't shared with you"
DOCKER_DETAIL      = "Shared chats on a Docker agent are chat only — no files or data"
STOPPED_DETAIL     = "This agent is stopped"
OUTDATED_DETAIL    = "This agent needs a restart before its shared files and data can be used — ask its owner"
AGENT_ERROR_DETAIL = "The agent couldn't answer that — try again"
RATE_DETAIL        = "Too many requests — try again in a minute"
TOO_LARGE_DETAIL   = "That file is too large (25 MB at most)"
DOWNLOAD_TOO_LARGE_DETAIL = "That file is too large to open here (50 MB at most)"
EMPTY_DETAIL       = "That file is empty"
BAD_TYPE_DETAIL    = "That kind of file can't be uploaded here"
BAD_ID_DETAIL      = "Invalid file"
BAD_TABLE_DETAIL   = "Invalid table name"
BAD_LANE_DETAIL    = "Invalid lane"
BAD_FORMAT_DETAIL  = "Unsupported format"
LENGTH_DETAIL      = "The upload's size is missing — try again"
UPLOAD_BODY_SLACK  = 64 * 1024   # multipart overhead allowed over MEMBER_UPLOAD_MAX_BYTES

# The one-time owner bell (part 1b §1; the body is part 0c §1, verbatim).
COMMONS_NOTICE_KEY = "shared_workspace_commons_notice_v1"
OWNER_COMMONS_BELL_TITLE = "Members of “{name}” can now open its saved/ folder and datastore"
OWNER_COMMONS_BELL_BODY = (
    "Members can also open everything in this agent’s saved/ folder — including what is "
    "already there: files you uploaded in your own chats, screenshots and browser captures, "
    "script outputs, the scripts and tools it saved for you (check them for passwords or "
    "keys), and what its channels and automations saved, such as WhatsApp or email "
    "attachments — and see every table and row in its datastore. They can add their own "
    "files, tables and rows, shown with their name, and change or delete only what they "
    "added; you can change or delete all of it. Its other files (workspace, output/, "
    "workflows/) stay yours. Your agent reads what members add, so treat it as untrusted "
    "input, especially in automations.")

_TABLE_NAME = re.compile(TABLE_NAME_RE)
_ORDER_BY = re.compile(ORDER_BY_RE)
# The only agent paths FD builds for a member: constants plus validated table names.
_AGENT_PATH = re.compile(r"/api/speaker/[a-z0-9_/]+")
_NAME_MAX = 120          # = agent_sharing_routes / tenant_profile name cap
_SNAPSHOT_MAX = 80
_A_MEMBER = "A member"


# ── Pure helpers ──────────────────────────────────────────────────────────


def valid_file_id(s: object) -> bool:
    """A file id (path relative to ``saved/``, posix) FD will pass on: 1..1024
    chars, no backslash or control character, not absolute, no empty, ``.``,
    ``..`` or dot-leading component."""
    if not isinstance(s, str) or not 1 <= len(s) <= FILE_ID_MAX:
        return False
    if "\\" in s or s.startswith("/"):
        return False
    if any(unicodedata.category(ch) == "Cc" for ch in s):
        return False
    return all(part and not part.startswith(".") for part in s.split("/"))


def valid_table(name: object) -> bool:
    return isinstance(name, str) and _TABLE_NAME.fullmatch(name) is not None


def valid_order_by(name: object) -> bool:
    return isinstance(name, str) and _ORDER_BY.fullmatch(name) is not None


def speaker_display_name(user: dict) -> str:
    """The name a member speaks under: display name, else email local part;
    one line, ≤ 120 chars (as the member chat socket's assertion)."""
    name = str(user.get("display_name") or "").strip() or str(
        user.get("email") or "").split("@")[0]
    return " ".join(name.split())[:_NAME_MAX]


_VIEW_TYPES = {
    ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".gif": "image/gif",
    ".webp": "image/webp", ".bmp": "image/bmp", ".svg": "image/svg+xml",
    ".pdf": "application/pdf",
    ".mp3": "audio/mpeg", ".wav": "audio/wav", ".ogg": "audio/ogg", ".m4a": "audio/mp4",
    ".mp4": "video/mp4", ".m4v": "video/mp4", ".webm": "video/webm", ".mov": "video/quicktime",
}
_TEXT_PLAIN = "text/plain; charset=utf-8"


def _suffix(filename: str) -> str:
    name = str(filename or "").rsplit("/", 1)[-1]
    dot = name.rfind(".")
    return name[dot:].lower() if dot > 0 else ""


def view_headers(filename: str) -> tuple[str, dict[str, str]]:
    """``(media type, headers)`` for viewing a member-visible file inline: only
    images, audio, video, pdf and svg keep their type, everything else is plain
    text; never sniffed, never cached, sandboxed (scripts can't run in FD's
    origin) except pdf, which the browser's own viewer shows."""
    media = _VIEW_TYPES.get(_suffix(filename), _TEXT_PLAIN)
    headers = {"Content-Disposition": "inline", "X-Content-Type-Options": "nosniff",
               "Cache-Control": "no-store"}
    if media != "application/pdf":
        headers["Content-Security-Policy"] = "sandbox"
    return media, headers


def download_headers(filename: str) -> dict[str, str]:
    return {
        "Content-Type": "application/octet-stream",
        "Content-Disposition": "attachment; filename*=UTF-8''"
                               + urllib.parse.quote(str(filename or ""), safe=""),
        "X-Content-Type-Options": "nosniff",
        "Cache-Control": "no-store",
    }


_EXPORT_TYPES = {
    "csv": "text/csv; charset=utf-8",
    "json": "application/json",
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
}


def export_headers(table: str, fmt: str) -> tuple[str, dict[str, str]]:
    """``(media type, headers)`` for a table export (``table`` is ``[a-z0-9_]``)."""
    return _EXPORT_TYPES[fmt], {
        "Content-Disposition": f'attachment; filename="{table}.{fmt}"',
        "X-Content-Type-Options": "nosniff",
        "Cache-Control": "no-store",
    }


# ── Rate limit ────────────────────────────────────────────────────────────

_RATE_WINDOW_S = 60.0
_RATE_MAX_KEYS = 4096
# (agent_ref, member) → monotonic times of their recent workspace requests.
_RATE: dict[tuple[str, str], collections.deque] = {}


def check_rate(ref: str, uid: str) -> None:
    """429 once a member made ``MEMBER_HTTP_RATE_PER_MIN`` workspace requests on
    one agent within the last minute; otherwise counts this one."""
    now = time.monotonic()
    key = (ref, uid)
    window = _RATE.get(key)
    if window is None:
        if len(_RATE) >= _RATE_MAX_KEYS:
            for k, d in list(_RATE.items()):
                while d and now - d[0] >= _RATE_WINDOW_S:
                    d.popleft()
                if not d:
                    _RATE.pop(k, None)
            if len(_RATE) >= _RATE_MAX_KEYS:
                _RATE.clear()
        window = _RATE[key] = collections.deque()
    while window and now - window[0] >= _RATE_WINDOW_S:
        window.popleft()
    if len(window) >= MEMBER_HTTP_RATE_PER_MIN:
        raise HTTPException(429, RATE_DETAIL)
    window.append(now)


def _reset_for_tests() -> None:
    _RATE.clear()


# ── Membership + target ───────────────────────────────────────────────────


async def member_target(db, user: dict, ref: str, *, fresh: bool) -> sharing.AgentRecord:
    """The shared process agent ``ref`` names, for a member of it — or the
    HTTPException saying why not. Every request re-checks membership (``fresh``:
    straight from the DB, for uploads and deletes). Admins get no bypass."""
    if not sharing.sharing_active():
        raise HTTPException(400, SHARING_OFF_DETAIL)
    try:
        sharing.parse_ref(ref)
    except ValueError:
        raise HTTPException(400, BAD_REF_DETAIL) from None
    rec = await asyncio.to_thread(sharing.resolve_agent_record, ref)
    if rec is None or sharing.check_shareable(rec, rec.owner):
        raise HTTPException(404, NOT_FOUND_DETAIL)
    uid = str(user["id"])
    if uid == rec.owner:
        raise HTTPException(400, OWNER_DETAIL)
    max_age = 0 if fresh else sharing.MEMBERSHIP_CACHE_TTL_S
    if not await sharing.member_check(db, ref, rec.owner, uid, max_age=max_age):
        raise HTTPException(404, NOT_SHARED_DETAIL)
    if rec.runtime != "process":
        raise HTTPException(400, DOCKER_DETAIL)
    if not rec.running or not rec.port:
        raise HTTPException(409, STOPPED_DETAIL)
    check_rate(ref, uid)
    return rec


# ── Agent client ──────────────────────────────────────────────────────────


def _agent_client(timeout: float) -> httpx.AsyncClient:
    # Always the local agent, never through an env proxy, never redirected.
    return httpx.AsyncClient(timeout=timeout, trust_env=False, follow_redirects=False)


def _error_detail(resp: httpx.Response) -> str:
    """The agent's ``{"error": str}``, one line and capped; else the generic text."""
    try:
        body = resp.json()
    except Exception:
        return AGENT_ERROR_DETAIL
    err = body.get("error") if isinstance(body, dict) else None
    if not isinstance(err, str):
        return AGENT_ERROR_DETAIL
    return " ".join(err.split())[:DETAIL_MAX] or AGENT_ERROR_DETAIL


async def call_agent(db, user: dict, rec, method: str, path: str, *, lane: str = "A",
                     params: dict | None = None, json_body=None, files=None,
                     timeout: float = AGENT_TIMEOUT_S) -> httpx.Response:
    """One member request to the agent's ``/api/speaker/*`` route ``path``.

    Signs a fresh assertion for this method and path, sends it with the token
    FD recorded (never one the browser sent), and returns the 2xx response —
    only when the agent acknowledged the assertion. An agent that didn't
    (an old one) never has its body read: 426 when it doesn't know the route,
    502 otherwise. The agent's 400/403/404/409/413 are relayed with its own
    ``error`` text; anything else is a 502."""
    method = str(method or "").upper()
    if (method not in ("GET", "POST") or not isinstance(path, str)
            or not path.startswith(SPEAKER_HTTP_PREFIX) or not _AGENT_PATH.fullmatch(path)):
        # FD never calls an owner route (or anything else) for a member.
        raise AssertionError("shared workspace: not a member route")
    if lane not in sharing.MEMBER_LANES:
        raise HTTPException(400, BAD_LANE_DETAIL)
    uid = str(user["id"])
    owner_name = await tenant_profile.owner_name(db, rec.owner)
    payload = sharing.speaker_payload(
        speaker_id=uid, name=speaker_display_name(user), owner=rec.owner,
        owner_name=owner_name, ref=rec.ref, lane=lane, conn="http-" + secrets.token_hex(8))
    payload.update({"aud": HTTP_ASSERTION_AUD, "m": method, "p": path})
    header = sharing.sign_speaker_assertion(rec.web_auth, payload)
    expected_ack = sharing.speaker_ack_for(header)
    # The browser never names the target: host, port and token are FD's records.
    query = {k: v for k, v in (params or {}).items() if k != "token"}
    query["token"] = rec.web_auth
    url = f"http://localhost:{int(rec.port)}{path}"
    try:
        async with _agent_client(timeout) as client:
            request = client.build_request(
                method, url, params=query, headers={sharing.SPEAKER_HEADER: header},
                json=json_body, files=files)
            resp = await client.send(request, stream=True)
            try:
                log.info("Shared-agent workspace call", agent=rec.slug, member=uid, path=path,
                         status=resp.status_code)
                acked = hmac.compare_digest(
                    resp.headers.get(SPEAKER_ACK_HEADER, "").encode("utf-8", "replace"),
                    expected_ack.encode("ascii"))
                if not acked:
                    # Not a PR C agent (or not this assertion): its body is never read.
                    if resp.status_code in (404, 405):
                        raise HTTPException(426, OUTDATED_DETAIL)
                    raise HTTPException(502, AGENT_ERROR_DETAIL)
                await resp.aread()
            finally:
                await resp.aclose()
    except httpx.HTTPError as exc:
        log.info("Shared agent unreachable", agent=rec.slug, error=type(exc).__name__)
        raise HTTPException(502, AGENT_ERROR_DETAIL) from None
    if 200 <= resp.status_code < 300:
        return resp
    if resp.status_code in RELAYED_STATUSES:
        raise HTTPException(resp.status_code, _error_detail(resp))
    raise HTTPException(502, AGENT_ERROR_DETAIL)


# ── Creator names ─────────────────────────────────────────────────────────

_OWNER_KEY = ("owner",)   # the agent owner's name, in the same per-call cache


async def creator_display_name(db, user_id: str, snapshot: str, cache: dict | None = None) -> str:
    """A member creator's name as FD shows it: their CURRENT display name (or
    email local part) while they are a user of this deck; else the name
    snapshot the agent stored, marked "(former member)", or "Former member".
    ``cache`` (per call, keyed by user id) holds what FD's users table said."""
    snap = " ".join(str(snapshot or "").split())[:_SNAPSHOT_MAX]
    uid = str(user_id or "")
    if cache is not None and uid in cache:
        current = cache[uid]
    else:
        row = await db.get_user_by_id(uid) if uid else None
        current = (await tenant_profile.owner_name(db, uid) or "") if row else None
        if cache is not None:
            cache[uid] = current
    if current is not None:
        return current or snap or _A_MEMBER
    return f"{snap} (former member)" if snap else FORMER_MEMBER


async def _owner_display(db, rec, cache: dict) -> str:
    if _OWNER_KEY not in cache:
        cache[_OWNER_KEY] = await tenant_profile.owner_name(db, rec.owner)
    return cache[_OWNER_KEY]


async def member_creator(db, uid: str, rec, creator: object, cache: dict) -> dict:
    """The agent's ``Creator`` as a member sees it (``MemberCreator``): "me",
    the owner by name, or another member by their current name — never a user
    id or an email. Anything malformed reads as the owner's."""
    if not isinstance(creator, dict) or creator.get("kind") not in ("owner", "member"):
        return {"kind": "owner", "name": await _owner_display(db, rec, cache)}
    if creator.get("kind") == "member":
        cid = str(creator.get("user_id") or "")
        if cid and cid == uid:
            return {"kind": "me", "name": ""}
        return {"kind": "member",
                "name": await creator_display_name(db, cid, creator.get("name"), cache)}
    return {"kind": "owner", "name": await _owner_display(db, rec, cache)}


async def _decorated(db, item: dict, key: str, cache: dict) -> dict:
    cb = item.get(key)
    if not isinstance(cb, dict) or cb.get("kind") != "member":
        return item
    name = await creator_display_name(db, str(cb.get("user_id") or ""), cb.get("name"), cache)
    return {**item, key: {**cb, "name": name}}


async def decorate_owner_payload(db, payload):
    """The owner proxies' agent JSON with every member creator's ``name``
    replaced by their current name (the owner sees ids; ids stay). A list:
    each item's ``created_by``; a dict with ``rows``: each row's ``_creator``.
    Never raises — on any error the payload is returned as it came."""
    if db is None:
        return payload
    try:
        cache: dict = {}
        if isinstance(payload, list):
            return [await _decorated(db, it, "created_by", cache) if isinstance(it, dict) else it
                    for it in payload]
        if isinstance(payload, dict) and isinstance(payload.get("rows"), list):
            rows = [await _decorated(db, r, "_creator", cache) if isinstance(r, dict) else r
                    for r in payload["rows"]]
            return {**payload, "rows": rows}
    except Exception as exc:
        log.debug("Could not decorate creator names", error=type(exc).__name__)
    return payload


# ── The one-time owner bell ───────────────────────────────────────────────


def _utcnow_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


async def notify_owners_of_commons_once(db) -> int:
    """Once per deck (sharing active): tell the owner of every shared process
    agent that its ``saved/`` folder and datastore are now open to members —
    including what is already there. One bell per agent; the system setting
    marks it done (a crash mid-way re-sends on the next start). Returns how
    many were sent. Never raises."""
    sent = 0
    try:
        if await db.get_system_setting(COMMONS_NOTICE_KEY):
            return 0
        offset = 0
        while True:
            users = await db.list_users(limit=500, offset=offset)
            if not users:
                break
            offset += len(users)
            for user in users:
                uid = str(user.get("id") or "")
                if not uid:
                    continue
                seen: set[str] = set()
                for row in await db.list_shares_for_owner(uid, sharing.AGENT_RESOURCE):
                    rid = str(row.get("resource_id") or "")
                    if not rid or rid in seen:
                        continue
                    seen.add(rid)
                    try:
                        if sharing.parse_ref(rid)[0] != "process":
                            continue
                    except ValueError:
                        continue
                    rec = await asyncio.to_thread(sharing.resolve_agent_record, rid)
                    if rec is None or rec.owner != uid:
                        continue
                    await db.add_notification(
                        uid, "share", OWNER_COMMONS_BELL_TITLE.format(name=rec.name),
                        body=OWNER_COMMONS_BELL_BODY, ref_type=sharing.AGENT_RESOURCE,
                        ref_id=rid)
                    sent += 1
        await db.set_system_setting(COMMONS_NOTICE_KEY, _utcnow_iso())
    except Exception as exc:
        log.debug("Shared-workspace owner notice failed", error=type(exc).__name__)
    return sent
