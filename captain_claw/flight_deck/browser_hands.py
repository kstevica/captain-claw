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

**An agent's own Chrome.** An agent can pair Captain for Chrome in its OWN
headless browser (the browser tool's ``extension_connect``): it asks here
(``/connect``), the person approves in Flight Deck, and the agent gets a token
of a kind of its own (``cc_ab_…``, kept in ``agent-browsers.json``, never a
personal access token) that only this socket and the act's file uploads take
— for that agent's acts alone. An agent's ``target="chrome"`` acts go to the
person's own Chrome first, and to its own browser only when theirs isn't
connected; other agents never land there. Connections → Browser lists it and
can disconnect it.

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
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.security import HTTPAuthorizationCredentials
from pydantic import BaseModel, Field

from captain_claw.flight_deck import person_asks as asks
from captain_claw.flight_deck.auth import get_db, get_mcp_user, hash_token
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
    "this Flight Deck). Until then use the headless browser (target=\"headless\") or web_fetch — or "
    "connect your own: browser(action=\"extension_connect\") pairs Captain for Chrome in your headless "
    "browser (the person approves it), and target=\"chrome\" then runs there while theirs is away."
)
AGENT_TOKEN_PREFIX = "cc_ab_"
AGENT_BROWSERS_FILE = "agent-browsers.json"
CONNECT_WAIT_S = 600.0
MINTED_KEEP_S = 600.0


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
    agent_slug: str = ""        # an agent's own (headless) browser: only that agent uses it
    closed: bool = False
    send_lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def send(self, msg: dict) -> None:
        async with self.send_lock:
            await self.ws.send_text(json.dumps(msg, ensure_ascii=False))

    def view(self) -> dict:
        return {"browser": self.browser, "version": self.version, "own_of": self.agent_slug,
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


def usable_by(user_id: str, slug: str) -> list[_Conn]:
    """The browsers *slug* may use, in order: the person's own first, then
    that agent's own headless one. Never another agent's."""
    conns = connections(user_id)
    return [c for c in conns if not c.agent_slug] + [c for c in conns if slug and c.agent_slug == slug]


# ── Agents' own browsers (tokens of their own, not personal access tokens) ──


def _agent_store_path() -> Path:
    from captain_claw.flight_deck import server as srv

    return Path(srv.DATA_DIR) / AGENT_BROWSERS_FILE


def agent_browser_rows(user_id: str | None = None) -> list[dict]:
    try:
        rows = json.loads(_agent_store_path().read_text("utf-8"))
    except (OSError, ValueError):
        return []
    rows = [r for r in rows if isinstance(r, dict) and r.get("id") and r.get("token_hash")] if isinstance(rows, list) else []
    return [r for r in rows if user_id is None or str(r.get("user_id")) == str(user_id)]


def _save_agent_rows(rows: list[dict]) -> None:
    path = _agent_store_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(rows, indent=1), "utf-8")
    tmp.replace(path)


def _agent_row_for(raw: str) -> dict | None:
    if not str(raw or "").startswith(AGENT_TOKEN_PREFIX):
        return None
    digest = hash_token(raw)
    return next((r for r in agent_browser_rows() if r.get("token_hash") == digest), None)


def mint_agent_browser(user_id: str, slug: str, name: str) -> tuple[str, dict]:
    """A new token for *slug*'s own browser (an older one of that agent's goes)."""
    raw = AGENT_TOKEN_PREFIX + secrets.token_urlsafe(32)
    row = {"id": "ab_" + secrets.token_hex(8), "user_id": str(user_id), "slug": slug, "name": name,
           "token_hash": hash_token(raw), "created_at": datetime.now(UTC).isoformat()}
    keep = [r for r in agent_browser_rows() if not (str(r.get("user_id")) == str(user_id) and r.get("slug") == slug)]
    _save_agent_rows(keep + [row])
    for conn in [c for c in _CONNS.get(str(user_id), []) if c.agent_slug == slug]:
        _unregister(conn)
    return raw, row


async def drop_agent_browser(user_id: str, row_id: str = "", slug: str = "") -> bool:
    """Forget an agent's own browser (by its id, or the agent's): its socket closes."""
    rows = agent_browser_rows()
    gone = [r for r in rows if str(r.get("user_id")) == str(user_id)
            and ((row_id and r.get("id") == row_id) or (slug and r.get("slug") == slug))]
    if not gone:
        return False
    _save_agent_rows([r for r in rows if r not in gone])
    for r in gone:
        await drop_token(user_id, str(r["id"]))
    return True


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
    """``(user, token row)`` for a browser's own unrevoked token — the person's
    Browser token, or an agent's own-browser token — else None."""
    from captain_claw.flight_deck import auth as fd_auth

    raw = str(raw or "").strip()
    db = get_db()
    if raw.startswith(AGENT_TOKEN_PREFIX):
        row = _agent_row_for(raw)
        if row is None:
            return None
        row = {**row, "agent_slug": str(row.get("slug") or ""),
               "name": f"{row.get('name') or row.get('slug')}'s own browser"}
    elif raw.startswith(fd_auth.PAT_PREFIX):
        row = await db.get_pat_by_hash(hash_token(raw))
        if not row or token_kind(row.get("name")) != "browser":
            return None
    else:
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
    if conn.agent_slug:
        return any(r.get("id") == conn.token_id and r.get("token_hash") == conn.token_hash
                   for r in agent_browser_rows(conn.user_id))
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
                 browser=one_line((row.get("name") if row.get("agent_slug") else first.get("browser"))
                                  or row.get("name") or "Browser", 80),
                 version=one_line(first.get("version"), 20), agent_slug=str(row.get("agent_slug") or ""))
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


async def _files_caller(request: Request) -> tuple[str, str]:
    """``(person, agent)`` posting an act's files: the person's browser (its
    token, as the extension's other calls), or an agent's own browser — which
    may post for that agent's acts only."""
    raw = (request.headers.get("authorization") or "").partition(" ")[2].strip()
    if raw.startswith(AGENT_TOKEN_PREFIX):
        row = _agent_row_for(raw)
        if row is None:
            raise HTTPException(401, "This browser isn't connected to Flight Deck.")
        return str(row["user_id"]), str(row.get("slug") or "")
    creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials=raw) if raw else None
    user = await get_clip_user(request, await get_mcp_user(request, creds))
    return str(user["id"]), ""


@router.post("/files/{act_id}")
async def hands_files(act_id: str, request: Request, caller: tuple[str, str] = Depends(_files_caller)) -> dict:
    """Files the browser fetched for an agent's act (``download``, ``save_page``):
    kept in the person's Web clips folder, listed in the act's answer."""
    user_id, own_of = caller
    act = _ACTS.get(act_id)
    if act is None or act.user_id != user_id or act.future.done() or (own_of and act.slug != own_of):
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
    conns = usable_by(owner, slug)
    # Whether Captain for Chrome is set up in one of the person's browsers at
    # all (a pairing's token, still unrevoked) — an agent a site blocks asks
    # them to turn it on then, or recommends installing it when it isn't.
    try:
        rows = await get_db().list_pats(owner)
    except Exception:
        rows = []
    return {"connected": bool(conns), "browsers": [c.view() for c in conns],
            "person_connected": any(not c.agent_slug for c in conns),
            "extension_set_up": any(token_kind(r.get("name")) == "browser" for r in rows),
            "own_connected": any(c.agent_slug == slug for c in conns),
            "own_paired": any(r.get("slug") == slug for r in agent_browser_rows(owner)),
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
    for cand in usable_by(owner, slug):
        if await _still_valid(cand):
            conn = cand
            break
        cand.closed = True
        await _close(cand.ws, 4401, "Disconnected.")
    if conn is None:
        return {"ok": False, "no_browser": True, "error": NO_BROWSER}
    if slug in conn.stopped:
        return {"ok": False, "stopped": True,
                "error": "The person stopped you using their browser (Stop in the Captain for Chrome panel). They "
                         "let you go on with “Let it again” there: ask them to, and don't try again until they say so."}
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
        if conn.agent_slug:
            result["own"] = True
        return result
    finally:
        _ACTS.pop(act.id, None)


# ── An agent pairs its own browser (the person approves) ─────────────

_MINTED: dict[str, tuple[float, str, dict]] = {}   # ask id → (when, agent, the answer with its token)


class ConnectBody(BaseModel):
    origin: str = Field("", max_length=60)
    wait: float = Field(20, ge=0, le=30)
    ask_id: str = Field("", pattern=r"^(xa_[0-9a-f]{16})?$")


class ConnectFollow(BaseModel):
    id: str = Field(..., max_length=40)
    wait: float = Field(25, ge=0, le=30)


def _connect_question(name: str) -> list[dict]:
    return asks.check_questions([{
        "id": "connect", "confirm": True,
        "question": f"Let {name} connect Captain for Chrome in its own browser?",
        "help": (f"It's {name}'s headless browser on the server, not your Chrome — your sign-ins aren't in it. "
                 f"{name}'s browser actions for your Chrome go there only while your own Chrome isn't connected; "
                 "no other agent uses it. Disconnect it any time in Connections → Browser."),
    }])


async def _connect_outcome(ask_id: str, done: dict, owner: str, slug: str, name: str) -> dict:
    now = time.time()
    for k in [k for k, (at, _s, _r) in _MINTED.items() if now - at > MINTED_KEEP_S]:
        _MINTED.pop(k, None)
    if done.get("ok") and (done.get("answers") or {}).get("connect") is True:
        kept = _MINTED.get(ask_id)
        if kept and kept[1] == slug:
            return kept[2]   # asked again for the same yes: the same token, never a second one
        raw, row = mint_agent_browser(owner, slug, name)
        result = {"ok": True, "token": raw, "id": row["id"], "name": name}
        _MINTED[ask_id] = (now, slug, result)
        try:
            await get_db().log_usage(owner, "agent_browser_connected", json.dumps({"agent": slug}))
        except Exception:
            pass
        return result
    if done.get("ok"):
        return {"ok": False, "declined": True, "error": "The person said no. Don't connect it unless they ask you to."}
    if done.get("declined"):
        return {"ok": False, "declined": True, "error": "The person chose not to answer. Don't connect it unless they ask."}
    return {"ok": False, "error": str(done.get("error") or "No answer from the person.")}


@agent_router.post("/turn-over")
async def agent_turn_over(request: Request) -> dict:
    """The agent's run is over: the browsers it may use hear it, and go back to
    the tab the person was on if one of its tabs took the front."""
    owner, slug = _owner_of(request, "the person's browser")
    told = 0
    for conn in usable_by(owner, slug):
        try:
            await conn.send({"type": "turn_over", "agent": {"slug": slug}})
            told += 1
        except Exception:
            pass
    return {"ok": True, "browsers": told}


@agent_router.post("/connect")
async def agent_connect(body: ConnectBody, request: Request) -> dict:
    """Ask the person to let this agent pair Captain for Chrome in its own
    headless browser: the token for it once they say yes, or ``{"pending": id}``."""
    owner, slug = _owner_of(request, "connecting a browser")
    if not slug:
        raise HTTPException(403, "Flight Deck can't tell which agent this is.")
    if not asks.attended(body.origin):
        return {"ok": False, "nobody": True,
                "error": "Nobody is here to approve it: this turn started on its own. Ask the person in a chat."}
    from captain_claw.flight_deck.agent_names import display_name

    name = await display_name(slug, owner) or slug
    try:
        a = await asks.open_ask(owner, source="agent", title=name, agent_name=name,
                                questions=_connect_question(name), wait_s=CONNECT_WAIT_S, ask_id=body.ask_id)
    except asks.AskError as exc:
        raise HTTPException(422, str(exc)) from None
    done = await asks.wait_answer(a, body.wait)
    if done is None:
        return {"pending": a.id, "seconds_left": max(0, int(a.deadline - time.time()))}
    return await _connect_outcome(a.id, done, owner, slug, name)


@agent_router.post("/connect/follow")
async def agent_connect_follow(body: ConnectFollow, request: Request) -> dict:
    owner, slug = _owner_of(request, "connecting a browser")
    try:
        done = await asks.follow(body.id, owner, body.wait)
    except KeyError:
        raise HTTPException(404, "There's no such request (it ended a while ago).") from None
    if done.get("pending"):
        return done
    from captain_claw.flight_deck.agent_names import display_name

    return await _connect_outcome(body.id, done, owner, slug, await display_name(slug, owner) or slug)


@agent_router.post("/connect/cancel")
async def agent_connect_cancel(body: ConnectFollow, request: Request) -> dict:
    owner, _slug = _owner_of(request, "connecting a browser")
    return {"cancelled": await asks.cancel(body.id, owner)}


@agent_router.post("/disconnect")
async def agent_disconnect(request: Request) -> dict:
    """The agent lets go of its own browser's connection (Captain removed)."""
    owner, slug = _owner_of(request, "connecting a browser")
    return {"ok": True, "dropped": await drop_agent_browser(owner, slug=slug)}
