"""Browser hands — an agent's ``browser`` tool working in the person's own Chrome.

Captain for Chrome (``browser-extension/chrome``) can lend the browser it runs
in to the person's agents. With **Agents can use this browser** on in its
settings, its service worker keeps a socket open here (``/fd/clip/hands/ws``)
and runs what an agent asks — open a tab, read the page, click, type, take a
screenshot — in the tabs of that agent's own tab group, signed in as the
person, while they watch:

  agent (``browser`` tool, ``target="chrome"``)
      POST /fd/browser-hands/agent/act  {action, args}
  → Flight Deck picks that agent's owner's browser (the one they used last)
      {"type": "act", id, agent: {slug, name}, action, args}   over the socket
  ← {"type": "result", id, ok, content, image?, data?, error?}
  → the agent gets the result (a screenshot as a base64 JPEG)

Files the browser fetches for an agent (``download``, ``save_page``) don't ride
the socket: the extension posts them to ``/fd/clip/hands/files/<act id>`` and
Flight Deck keeps them in the person's **Web clips** folder
(``vfs:web-clips/<day>-<site>-<agent>-browser/``), which every agent of
theirs reads through ``vfs:``; the act's answer lists the paths.

Who may use it: only the person's own agents. The caller is resolved from
Flight Deck's own records (``require_agent_caller``) and its owner is the
person whose browser is used. A shared agent's member turns never reach the
browser tool (the speaker allowlist leaves it out) and these routes aren't
grant-aware (the grant guard refuses member markers here), so a member can't
drive the owner's browser either. Only a browser's own token
(``Browser · …``, still unrevoked) may offer a browser. The extension keeps an
agent to its own tab group and refuses Flight Deck's own address and pages
that aren't on the web; the person can stop any agent from the panel.
"""

from __future__ import annotations

import asyncio
import json
import logging
import secrets
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from captain_claw.flight_deck.auth import get_db, hash_token
from captain_claw.flight_deck.browser_clip import (
    CLIP_PROJECT,
    MAX_FILE_BYTES,
    _clip_root,
    _new_folder,
    _read_part,
    _write,
    clip_folder_name,
    get_clip_user,
    human_size,
    one_line,
    safe_file_name,
    token_kind,
    unique_name,
)

log = logging.getLogger(__name__)

# The extension's side (its origin may reach /fd/clip/: origin_guard.extension_route).
router = APIRouter(prefix="/fd/clip/hands", tags=["browser-hands"])
# The agent's side (require_agent_caller: never a browser).
agent_router = APIRouter(prefix="/fd/browser-hands/agent", tags=["browser-hands"])

PROTOCOL = 1
HELLO_WAIT_S = 10.0          # the extension says who it is this soon, or goes
STALE_S = 75.0               # a browser silent this long is gone (it pings every 20 s)
ACT_WAIT_DEFAULT = 40.0
ACT_WAIT_MAX = 55.0          # under the browser tool's own 60 s
MAX_CONNS_PER_USER = 8
MAX_ACTS_PER_CONN = 24
MAX_ARGS_BYTES = 24 * 1024 * 1024   # an upload_file's bytes ride the args
MAX_CONTENT_CHARS = 400_000
MAX_IMAGE_CHARS = 12_000_000        # base64 JPEG
MAX_ACT_FILES = 40
MAX_ACT_BYTES = 150 * 1024 * 1024

NO_BROWSER = (
    "The person's Chrome isn't connected for agents. Ask them to open Captain for Chrome → "
    "Settings and turn on “Agents can use this browser” (the extension must be connected to "
    "this Flight Deck). Until then use the headless browser (target=\"headless\") or web_fetch."
)


def _now() -> float:
    return time.monotonic()


@dataclass(eq=False)
class _Conn:
    ws: WebSocket
    user_id: str
    token_id: str
    token_hash: str
    browser: str
    version: str
    connected_at: float = field(default_factory=_now)
    last_seen: float = field(default_factory=_now)
    focused_at: float = 0.0     # the person was last at this browser (window focus)
    stopped: tuple[str, ...] = ()   # agents the person stopped in the panel
    closed: bool = False
    send_lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def send(self, msg: dict) -> None:
        async with self.send_lock:
            await self.ws.send_text(json.dumps(msg, ensure_ascii=False))

    def view(self) -> dict:
        return {"browser": self.browser, "version": self.version,
                "connected_s": int(_now() - self.connected_at), "token_id": self.token_id}


@dataclass(eq=False)
class _Act:
    id: str
    user_id: str
    slug: str
    agent_name: str
    action: str
    conn: _Conn
    future: asyncio.Future
    started: float = field(default_factory=_now)
    folder: str = ""            # the Web clips folder its files went to
    files: list[dict] = field(default_factory=list)
    file_bytes: int = 0


_CONNS: dict[str, list[_Conn]] = {}
_ACTS: dict[str, _Act] = {}


def connections(user_id: str) -> list[_Conn]:
    """The person's browsers that are connected for agents, freshest first."""
    now = _now()
    live = [c for c in _CONNS.get(str(user_id), []) if not c.closed and now - c.last_seen < STALE_S]
    return sorted(live, key=lambda c: max(c.focused_at, c.connected_at), reverse=True)


def connected_token_ids(user_id: str) -> set[str]:
    return {c.token_id for c in connections(user_id)}


def _register(conn: _Conn) -> None:
    mine = [c for c in _CONNS.get(conn.user_id, []) if not c.closed]
    # The same browser reconnecting replaces its old socket.
    for old in [c for c in mine if c.token_id == conn.token_id]:
        old.closed = True
        _fail_acts(old, "The browser reconnected; ask again.")
    mine = [c for c in mine if not c.closed]
    while len(mine) >= MAX_CONNS_PER_USER:
        oldest = min(mine, key=lambda c: c.last_seen)
        oldest.closed = True
        _fail_acts(oldest, "The browser was replaced by another.")
        mine.remove(oldest)
    mine.append(conn)
    _CONNS[conn.user_id] = mine


def _unregister(conn: _Conn) -> None:
    conn.closed = True
    left = [c for c in _CONNS.get(conn.user_id, []) if c is not conn and not c.closed]
    if left:
        _CONNS[conn.user_id] = left
    else:
        _CONNS.pop(conn.user_id, None)
    _fail_acts(conn, "The browser disconnected before it answered.")


def _fail_acts(conn: _Conn, why: str) -> None:
    for act in list(_ACTS.values()):
        if act.conn is conn and not act.future.done():
            act.future.set_result({"ok": False, "error": why})


async def drop_token(user_id: str, token_id: str) -> None:
    """A browser's token was revoked: its socket closes now."""
    for conn in [c for c in _CONNS.get(str(user_id), []) if c.token_id == str(token_id)]:
        _unregister(conn)
        await _close(conn.ws, 4401, "Disconnected.")


async def _browser_token(raw: str) -> tuple[dict, dict] | None:
    """``(user, token row)`` for a browser's own unrevoked token, else None."""
    from captain_claw.flight_deck import auth as fd_auth

    raw = str(raw or "").strip()
    if not raw.startswith(fd_auth.PAT_PREFIX):
        return None
    db = get_db()
    row = await db.get_pat_by_hash(hash_token(raw))
    if not row or token_kind(row.get("name")) != "browser":
        return None
    uid = str(row.get("user_id") or "")
    user = await db.get_user_by_id(uid) if uid else None
    if not user:
        if fd_auth._fd_auth_enabled() or not uid:
            return None
        user = {**fd_auth._LOCAL_USER, "id": uid}
    return user, row


async def _still_valid(conn: _Conn) -> bool:
    """The browser's token wasn't revoked since it connected."""
    try:
        row = await get_db().get_pat_by_hash(conn.token_hash)
    except Exception:
        return True   # a database hiccup doesn't cut the person off
    return bool(row) and str(row.get("id")) == conn.token_id


def _clip_text(value: Any, limit: int) -> str:
    s = value if isinstance(value, str) else ("" if value is None else str(value))
    return s if len(s) <= limit else s[: limit - 40] + f"\n… (cut at {limit:,} characters)"


def _clean_result(msg: dict) -> dict:
    """What the agent gets from the browser's answer — bounded, known keys only."""
    out: dict[str, Any] = {"ok": bool(msg.get("ok"))}
    if msg.get("content") is not None:
        out["content"] = _clip_text(msg.get("content"), MAX_CONTENT_CHARS)
    if msg.get("error"):
        out["error"] = _clip_text(msg.get("error"), 4000)
    image = msg.get("image")
    if isinstance(image, str) and image and len(image) <= MAX_IMAGE_CHARS:
        out["image"] = image
        out["image_type"] = "image/png" if msg.get("image_type") == "image/png" else "image/jpeg"
    data = msg.get("data")
    if isinstance(data, dict):
        try:
            text = json.dumps(data, ensure_ascii=False)
        except (TypeError, ValueError):
            text = ""
        if text and len(text) <= MAX_CONTENT_CHARS:
            out["data"] = data
    if not out["ok"] and not out.get("error"):
        out["error"] = "The browser couldn't do that."
    return out


# ── The extension's socket ────────────────────────────────────────────


@router.websocket("/ws")
async def hands_ws(ws: WebSocket) -> None:
    """One browser lending itself to the person's agents. The first message
    says who it is: ``{"type": "hello", "token", "browser", "version"}``."""
    await ws.accept()
    try:
        first = json.loads(await asyncio.wait_for(ws.receive_text(), HELLO_WAIT_S))
    except (asyncio.TimeoutError, ValueError, WebSocketDisconnect, RuntimeError, KeyError):
        await _close(ws, 4400, "Say hello first.")
        return
    if not isinstance(first, dict) or first.get("type") != "hello":
        await _close(ws, 4400, "Say hello first.")
        return
    found = await _browser_token(str(first.get("token") or ""))
    if found is None:
        try:
            await ws.send_text(json.dumps({"type": "denied", "error": "This browser isn't connected to Flight Deck."}))
        except Exception:
            pass
        await _close(ws, 4401, "Not connected.")
        return
    user, row = found
    conn = _Conn(ws=ws, user_id=str(user["id"]), token_id=str(row["id"]),
                 token_hash=str(row.get("token_hash") or ""),
                 browser=one_line(first.get("browser") or row.get("name") or "Browser", 80),
                 version=one_line(first.get("version"), 20))
    conn.stopped = _stopped_list(first.get("stopped"))
    _register(conn)
    log.info("Browser hands connected: user=%s browser=%s", conn.user_id, conn.browser)
    try:
        await conn.send({"type": "welcome", "protocol": PROTOCOL,
                         "user": str(user.get("display_name") or user.get("email") or "")})
        while True:
            raw = await ws.receive_text()
            conn.last_seen = _now()
            try:
                msg = json.loads(raw)
            except ValueError:
                continue
            if not isinstance(msg, dict):
                continue
            kind = msg.get("type")
            if kind == "ping":
                await conn.send({"type": "pong"})
            elif kind == "focus":
                conn.focused_at = _now()
            elif kind == "stopped":
                conn.stopped = _stopped_list(msg.get("agents"))
            elif kind == "result":
                act = _ACTS.get(str(msg.get("id") or ""))
                if act is not None and act.conn is conn and not act.future.done():
                    act.future.set_result(_clean_result(msg))
    except (WebSocketDisconnect, RuntimeError):
        pass
    except Exception:
        log.warning("Browser hands socket failed", exc_info=True)
    finally:
        _unregister(conn)
        log.info("Browser hands disconnected: user=%s browser=%s", conn.user_id, conn.browser)


def _stopped_list(raw: Any) -> tuple[str, ...]:
    if not isinstance(raw, list):
        return ()
    return tuple(str(s)[:80] for s in raw[:200] if isinstance(s, str) and s)


async def _close(ws: WebSocket, code: int, reason: str) -> None:
    try:
        await ws.close(code=code, reason=reason)
    except Exception:
        pass


class _FilesMeta(BaseModel):
    page_url: str = Field("", max_length=4000)
    page_title: str = Field("", max_length=400)


@router.post("/files/{act_id}")
async def hands_files(act_id: str, request: Request, user: dict = Depends(get_clip_user)) -> dict:
    """Files the browser fetched for an agent's act (``download``, ``save_page``):
    kept in the person's Web clips folder, listed in the act's answer."""
    act = _ACTS.get(act_id)
    if act is None or act.user_id != str(user["id"]) or act.future.done():
        raise HTTPException(404, "That request is over.")
    form = await request.form(max_files=MAX_ACT_FILES + 1, max_fields=10)
    try:
        meta = _FilesMeta(**json.loads(str(form.get("meta") or "{}")))
    except Exception:
        meta = _FilesMeta()
    parts = [p for p in form.getlist("file") if hasattr(p, "read")]
    if len(act.files) + len(parts) > MAX_ACT_FILES:
        raise HTTPException(413, f"At most {MAX_ACT_FILES} files at a time.")
    root = _clip_root(act.user_id)
    if not act.folder:
        title = f"{act.agent_name or act.slug} browser"
        act.folder = _new_folder(root, clip_folder_name(title, meta.page_url, datetime.now()))
    taken = {f["name"] for f in act.files}
    saved = []
    for part in parts:
        blob = await _read_part(part, MAX_FILE_BYTES)
        if act.file_bytes + len(blob) > MAX_ACT_BYTES:
            raise HTTPException(413, f"Over {human_size(MAX_ACT_BYTES)} in all.")
        name = unique_name(safe_file_name(getattr(part, "filename", ""), "file"), taken)
        taken.add(name)
        _write(root, act.folder, name, blob, act.user_id)
        act.file_bytes += len(blob)
        entry = {"name": name, "size": len(blob), "path": f"vfs:{CLIP_PROJECT}/{act.folder}/{name}"}
        act.files.append(entry)
        saved.append(entry)
    return {"ok": True, "saved": saved, "folder": f"vfs:{CLIP_PROJECT}/{act.folder}/"}


# ── The agent's calls ─────────────────────────────────────────────────


def _owner_of(request: Request, what: str) -> tuple[str, str]:
    """``(person, agent slug)``: the calling agent's owner, from Flight Deck's records."""
    from captain_claw.flight_deck.agent_identity import require_agent_caller
    from captain_claw.flight_deck.auth import _fd_auth_enabled

    caller = require_agent_caller(request, what=what)
    owner = caller.owner or ("" if _fd_auth_enabled() else "local")
    if not owner:
        raise HTTPException(403, "This agent has no owner on record, so it has no browser to use.")
    return owner, caller.slug


class ActBody(BaseModel):
    action: str = Field(..., pattern=r"^[a-z_]{1,40}$")
    args: dict[str, Any] = Field(default_factory=dict)
    wait: float = Field(ACT_WAIT_DEFAULT, ge=1, le=ACT_WAIT_MAX)


@agent_router.get("/status")
async def agent_status(request: Request) -> dict:
    owner, slug = _owner_of(request, "the person's browser")
    conns = connections(owner)
    return {"connected": bool(conns), "browsers": [c.view() for c in conns],
            "stopped": any(slug in c.stopped for c in conns[:1])}


@agent_router.post("/act")
async def agent_act(body: ActBody, request: Request) -> dict:
    """Do one thing in the person's browser and answer with what happened."""
    owner, slug = _owner_of(request, "the person's browser")
    try:
        size = len(json.dumps(body.args, ensure_ascii=False))
    except (TypeError, ValueError):
        raise HTTPException(422, "The arguments aren't plain data.") from None
    if size > MAX_ARGS_BYTES:
        raise HTTPException(413, "The arguments are too large.")
    conn = None
    for cand in connections(owner):
        if await _still_valid(cand):
            conn = cand
            break
        cand.closed = True
        await _close(cand.ws, 4401, "Disconnected.")
    if conn is None:
        return {"ok": False, "no_browser": True, "error": NO_BROWSER}
    if slug in conn.stopped:
        return {"ok": False, "stopped": True,
                "error": "The person stopped you using their browser (Captain for Chrome). Don't try again "
                         "unless they ask you to; carry on without it or ask them."}
    if sum(1 for a in _ACTS.values() if a.conn is conn) >= MAX_ACTS_PER_CONN:
        return {"ok": False, "error": "The browser is busy with other requests; try again in a moment."}
    from captain_claw.flight_deck.agent_names import display_name

    name = await display_name(slug, owner) or slug or "Agent"
    act = _Act(id="ha_" + secrets.token_hex(10), user_id=owner, slug=slug, agent_name=name,
               action=body.action, conn=conn, future=asyncio.get_running_loop().create_future())
    _ACTS[act.id] = act
    try:
        try:
            await conn.send({"type": "act", "id": act.id, "agent": {"slug": slug, "name": name},
                             "action": body.action, "args": body.args, "wait": body.wait})
        except Exception:
            return {"ok": False, "error": "Couldn't reach the browser; it may have just closed."}
        try:
            result = await asyncio.wait_for(asyncio.shield(act.future), body.wait)
        except asyncio.TimeoutError:
            try:
                await conn.send({"type": "cancel", "id": act.id})
            except Exception:
                pass
            result = {"ok": False, "timeout": True,
                      "error": f"The browser didn't finish in {int(body.wait)} s (the page may still be loading). "
                               "Take a screenshot or read the page to see where it is."}
        if act.files:
            result["files"] = list(act.files)
            result["folder"] = f"vfs:{CLIP_PROJECT}/{act.folder}/"
        result["browser"] = conn.browser
        return result
    finally:
        _ACTS.pop(act.id, None)
