"""Iskra §9.1 federation over WebSocket — NAT-friendly visiting beings.

A visiting being lives on a private machine behind NAT, so the HOST can never
reach it directly. Instead the SENDER dials OUT (an outbound WS is NAT-friendly)
and holds the socket open; the host then multiplexes data requests DOWN that
socket and the sender answers from its own store. Same reverse-tunnel shape as
the PTY relay (captain_claw/terminal/relay.py).

Frames (JSON): sender→host ``hello``/``beat``/``res``; host→sender ``req``;
host→sender ``welcome``/``error``. No public URL is needed on the sender — only
the host is publicly reachable, which is the whole point of a public village.

Single FD process → in-memory registries. (Multi-worker would need a shared bus;
the beings loop is single-process, so this holds.)
"""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime, timezone

from fastapi import HTTPException, WebSocket, WebSocketDisconnect

from captain_claw.flight_deck import being_life, being_mind
from captain_claw.flight_deck.beings import BeingError, BeingsStore, get_store
from captain_claw.logging import get_logger

log = get_logger(__name__)

LINK_REQUEST_TIMEOUT = 20.0     # host waits this long for a visitor's answer
BEAT_SECONDS = 15.0             # sender → host snapshot heartbeat (keeps the
                                # roster's cached snapshot live; tiny frame)
_MAX_FRAME = 4_000_000          # a file/journal can be biggish; cap generously


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ══ HOST side: live sender connections, keyed by visitor id ══════════════

class _Link:
    def __init__(self, ws: WebSocket) -> None:
        self.ws = ws
        self.pending: dict[str, asyncio.Future] = {}
        self.send_lock = asyncio.Lock()


_links: dict[str, _Link] = {}


def is_linked(visitor_id: str) -> bool:
    return visitor_id in _links


def _being_by_name(store: BeingsStore, owner: str, ref: str) -> dict | None:
    """Resolve a host-village being by display name or slug (pen-pal target)."""
    ref_l = (ref or "").strip().lower()
    if not ref_l:
        return None
    for row in store.list(owner):
        if row["slug"].lower() == ref_l or row["name"].lower() == ref_l:
            return store.get(owner, row["slug"])
    return None


async def send_letter_to_visitor(store: BeingsStore, visitor_id: str,
                                 frm: str, village: str, body: str) -> None:
    """HOST role of pen-pals (roadmap T2.8): a local being writes to a being
    visiting this village. Rides the existing link; raises BeingError with a
    plain reason when the visitor is offline (loud refusal, never silence)."""
    try:
        await link_request(visitor_id, "letter", frm=frm, village=village,
                           body=body)
    except HTTPException as e:
        raise BeingError(str(e.detail), e.status_code)


async def _close(ws: WebSocket, code: int) -> None:
    try:
        await ws.close(code=code)
    except Exception:  # noqa: BLE001
        pass


async def village_link_ws(ws: WebSocket) -> None:
    """A sender dials in here, presents the village secret + its being, then
    answers the requests we push down for browsing humans."""
    await ws.accept()
    store = get_store()
    try:
        hello = await ws.receive_json()
    except Exception:  # noqa: BLE001 — abandoned handshake
        return
    if not isinstance(hello, dict) or hello.get("t") != "hello":
        await _close(ws, 4001)
        return
    owner = store.owner_by_secret(hello.get("secret") or "")
    if not owner:
        try:
            await ws.send_json({"t": "error", "message": "invalid village secret"})
        except Exception:  # noqa: BLE001
            pass
        await _close(ws, 4003)
        return
    slug = str(hello.get("slug") or "").strip()
    if not slug:
        await _close(ws, 4001)
        return
    name = str(hello.get("name") or slug)[:80]
    home = str(hello.get("home") or "")[:400]
    profile = hello.get("profile") if isinstance(hello.get("profile"), dict) else {}
    try:
        visitor = store.upsert_visitor(owner, home, slug, name, profile)
    except Exception as e:  # noqa: BLE001
        try:
            await ws.send_json({"t": "error", "message": str(e)})
        except Exception:  # noqa: BLE001
            pass
        await _close(ws, 4000)
        return
    vid = visitor["id"]
    # A one-shot PROBE only validates the secret + refreshes the snapshot; it
    # must NOT register in _links (it would clobber the persistent maintainer's
    # live connection under the same visitor id, then delete it on close).
    if hello.get("probe"):
        try:
            await ws.send_json({"t": "welcome", "visitor_id": vid})
        except Exception:  # noqa: BLE001
            pass
        await _close(ws, 1000)
        return
    conn = _Link(ws)
    _links[vid] = conn
    try:
        await ws.send_json({"t": "welcome", "visitor_id": vid})
    except Exception:  # noqa: BLE001
        _links.pop(vid, None)
        return
    log.info("village visitor linked", slug=slug, owner=owner)
    try:
        while True:
            msg = await ws.receive_json()
            if not isinstance(msg, dict):
                continue
            t = msg.get("t")
            if t == "res":
                fut = conn.pending.pop(str(msg.get("id")), None)
                if fut is not None and not fut.done():
                    fut.set_result(msg)
            elif t == "beat":
                try:
                    store.upsert_visitor(owner, home, slug, name,
                                         msg.get("profile") or profile)
                except Exception:  # noqa: BLE001
                    pass
            elif t == "letter":
                # A pen-pal letter (roadmap T2.8) FROM the visiting being TO
                # a being of this village — resolve by name, deliver as an
                # event, and ack so the sender's tick can be honest about
                # whether anything truly arrived.
                ack = {"t": "ack", "id": msg.get("id"), "ok": False}
                try:
                    target = _being_by_name(store, owner,
                                            str(msg.get("to") or ""))
                    if target is None:
                        ack["error"] = "no being of that name lives here"
                    elif target["state"] != "alive":
                        ack["error"] = ("that being cannot receive letters "
                                        "right now")
                    else:
                        store.record_event(
                            target["id"], "penpal_letter",
                            {"from": str(msg.get("frm") or name)[:80],
                             "village": home or "another village",
                             "body": str(msg.get("body") or "")[:2000]})
                        ack["ok"] = True
                except Exception as e:  # noqa: BLE001
                    ack["error"] = str(e)
                try:
                    async with conn.send_lock:
                        await ws.send_json(ack)
                except Exception:  # noqa: BLE001
                    pass
    except WebSocketDisconnect:
        pass
    except Exception as e:  # noqa: BLE001
        log.warning("village link error", slug=slug, error=str(e))
    finally:
        if _links.get(vid) is conn:  # a reconnect may already own the slot
            del _links[vid]
        for fut in conn.pending.values():
            if not fut.done():
                fut.set_exception(RuntimeError("the visitor disconnected"))
        log.info("village visitor unlinked", slug=slug)


async def link_request(visitor_id: str, op: str, **payload) -> dict:
    """Push one data request down a visitor's socket and await its reply. Raises
    HTTPException(502) when the visitor is offline. Used by the host's proxy."""
    conn = _links.get(visitor_id)
    if conn is None:
        raise HTTPException(502, "this visitor is offline right now — its home "
                            "machine isn't connected")
    rid = uuid.uuid4().hex
    fut: asyncio.Future = asyncio.get_running_loop().create_future()
    conn.pending[rid] = fut
    try:
        async with conn.send_lock:
            await conn.ws.send_json({"t": "req", "id": rid, "op": op, **payload})
    except Exception as e:  # noqa: BLE001
        conn.pending.pop(rid, None)
        raise HTTPException(502, f"failed to reach the visitor: {e}")
    try:
        resp = await asyncio.wait_for(fut, timeout=LINK_REQUEST_TIMEOUT)
    except asyncio.TimeoutError:
        conn.pending.pop(rid, None)
        raise HTTPException(504, "the visitor took too long to answer")
    except RuntimeError as e:
        raise HTTPException(502, str(e))
    if not resp.get("ok"):
        raise HTTPException(int(resp.get("status") or 502),
                            resp.get("error") or "the visitor returned an error")
    body = resp.get("body")
    return body if isinstance(body, dict) else {}


# ══ SENDER side: compute a reply to a host's request from our own store ═══

async def handle_link_request(store: BeingsStore, owner: str, slug: str,
                              op: str, req: dict) -> dict:
    """Answer one host request about our visiting being — straight from the
    store/home, no ``public`` flag required (visiting is the consent)."""
    being = store.get(owner, slug)
    if op == "profile":
        return being_life.public_profile(store, being)
    if op == "files":
        return {"files": being_life.list_self_files(being)}
    if op == "file":
        path = str(req.get("path") or "")
        return {"path": path, "text": being_life.read_self_file(being, path)}
    if op == "journal":
        day = str(req.get("date") or "") or _utcnow().strftime("%Y-%m-%d")
        p = being_life._home_path(being, f"journal/{day}.md")
        text = p.read_text(encoding="utf-8") if p.exists() else ""
        return {"date": day, "text": text}
    if op == "graph":
        return being_mind.graph(store, being)
    if op == "message":
        return store.post_message_for(owner, slug, req.get("name") or "",
                                      req.get("body") or "", req.get("thread_id"))
    if op == "thread":
        return store.thread_for(owner, slug, str(req.get("thread_id") or ""))
    if op == "letter":
        # A pen-pal letter (roadmap T2.8) from a being of the village we are
        # visiting, delivered into our being's world: one event, surfaced as
        # a percept on its next tick, exactly once.
        frm = str(req.get("frm") or "a being")[:80]
        village = str(req.get("village") or "")[:80]
        body = str(req.get("body") or "").strip()[:2000]
        if not body:
            raise BeingError("an empty letter says nothing", 400)
        if being["state"] != "alive":
            raise BeingError("this being cannot receive letters right now", 409)
        store.record_event(being["id"], "penpal_letter",
                           {"from": frm, "village": village, "body": body})
        return {"delivered": True}
    raise BeingError(f"unknown op {op!r}", 400)


def _ws_url(url: str) -> str:
    u = (url or "").strip().rstrip("/")
    if u.startswith("https://"):
        u = "wss://" + u[len("https://"):]
    elif u.startswith("http://"):
        u = "ws://" + u[len("http://"):]
    elif not u.startswith(("ws://", "wss://")):
        u = "ws://" + u
    return u + "/fd/public/village/link"


def _home_label(store: BeingsStore, owner: str) -> str:
    """A display label for where a visitor comes from (its own public URL if it
    set one; usually empty for a NAT'd private village)."""
    try:
        return store.get_village_meta(owner).get("public_url") or ""
    except Exception:  # noqa: BLE001
        return ""


class _ClientConn:
    """A live outbound link (sender side), with pending acks for the frames
    WE initiate (pen-pal letters) — the mirror of the host's _Link."""

    def __init__(self, ws) -> None:
        self.ws = ws
        self.pending: dict[str, asyncio.Future] = {}
        self.send_lock = asyncio.Lock()


class VillageClient:
    """Keeps one outbound WS per visiting being alive, reconnecting on drop."""

    def __init__(self) -> None:
        self._tasks: dict[str, asyncio.Task] = {}
        self._cfg: dict[str, tuple[str, str, str]] = {}   # slug -> (owner,url,secret)
        self._conns: dict[str, _ClientConn] = {}          # slug -> live link

    def is_up(self, slug: str) -> bool:
        return slug in self._conns

    async def send_letter(self, slug: str, to: str, frm: str,
                          body: str) -> None:
        """SENDER role of pen-pals (roadmap T2.8): our visiting being writes
        to a being of the village it visits, over its own live link. Waits
        for the host's ack so delivery is real, refusal is loud."""
        conn = self._conns.get(slug)
        if conn is None:
            raise BeingError("your link to the village you visit is down "
                             "right now — nothing can be delivered", 502)
        rid = uuid.uuid4().hex
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        conn.pending[rid] = fut
        try:
            async with conn.send_lock:
                await conn.ws.send(json.dumps(
                    {"t": "letter", "id": rid, "to": to[:80], "frm": frm[:80],
                     "body": body[:2000]}))
            ack = await asyncio.wait_for(fut, timeout=LINK_REQUEST_TIMEOUT)
        except (asyncio.TimeoutError, OSError):
            raise BeingError("the village took too long to answer — the "
                             "letter did not arrive", 504)
        finally:
            conn.pending.pop(rid, None)
        if not ack.get("ok"):
            raise BeingError(str(ack.get("error")
                                 or "the village refused the letter"), 502)

    async def reconcile(self, store: BeingsStore) -> None:
        """Start maintainers for beings that should be visiting, stop the rest."""
        try:
            beings = store.beings_visiting()
        except Exception as e:  # noqa: BLE001
            log.warning("visit reconcile failed", error=str(e))
            return
        want = {b["slug"]: (b["owner_id"], b.get("visit_url") or "",
                            b.get("visit_secret") or "") for b in beings}
        for slug in list(self._tasks):
            if slug not in want or self._cfg.get(slug) != want[slug]:
                self._stop(slug)
        for slug, cfg in want.items():
            if slug not in self._tasks:
                self._cfg[slug] = cfg
                self._tasks[slug] = asyncio.create_task(
                    self._maintain(store, slug))

    def _stop(self, slug: str) -> None:
        self._cfg.pop(slug, None)
        t = self._tasks.pop(slug, None)
        if t is not None and not t.done():
            t.cancel()

    async def stop_all(self) -> None:
        for slug in list(self._tasks):
            self._stop(slug)

    async def _maintain(self, store: BeingsStore, slug: str) -> None:
        from websockets.asyncio.client import connect
        backoff = 1.0
        while True:
            cfg = self._cfg.get(slug)
            if not cfg or not cfg[1]:
                return
            owner, url, secret = cfg
            try:
                async with connect(_ws_url(url), open_timeout=15,
                                   max_size=_MAX_FRAME) as ws:
                    await self._session(store, owner, slug, secret, ws)
                backoff = 1.0
            except asyncio.CancelledError:
                return
            except Exception as e:  # noqa: BLE001
                log.warning("village link down", slug=slug, error=str(e))
            try:
                await asyncio.sleep(min(backoff, 60))
            except asyncio.CancelledError:
                return
            backoff = min(backoff * 2, 60)

    async def _session(self, store: BeingsStore, owner: str, slug: str,
                       secret: str, ws) -> None:
        being = store.get(owner, slug)
        await ws.send(json.dumps({
            "t": "hello", "secret": secret, "slug": slug, "name": being["name"],
            "home": _home_label(store, owner),
            "profile": being_life.public_profile(store, being)}))
        welcome = json.loads(await asyncio.wait_for(ws.recv(), timeout=15))
        if not isinstance(welcome, dict) or welcome.get("t") != "welcome":
            raise RuntimeError(
                (welcome or {}).get("message") if isinstance(welcome, dict)
                else "the village rejected the link")
        store.mark_announced(being["id"])
        log.info("visiting village linked", slug=slug, target=self._cfg.get(slug, ("", "", ""))[1])

        async def beat() -> None:
            while True:
                await asyncio.sleep(BEAT_SECONDS)
                try:
                    b = store.get(owner, slug)
                    await ws.send(json.dumps({
                        "t": "beat",
                        "profile": being_life.public_profile(store, b)}))
                    store.mark_announced(b["id"])
                except Exception:  # noqa: BLE001
                    return

        bt = asyncio.create_task(beat())
        conn = _ClientConn(ws)
        self._conns[slug] = conn
        try:
            async for message in ws:
                try:
                    msg = json.loads(message)
                except Exception:  # noqa: BLE001
                    continue
                if not isinstance(msg, dict):
                    continue
                if msg.get("t") == "req":
                    asyncio.create_task(self._serve(store, owner, slug, ws, msg))
                elif msg.get("t") == "ack":
                    fut = conn.pending.pop(str(msg.get("id")), None)
                    if fut is not None and not fut.done():
                        fut.set_result(msg)
        finally:
            bt.cancel()
            if self._conns.get(slug) is conn:
                del self._conns[slug]
            for fut in conn.pending.values():
                if not fut.done():
                    fut.set_exception(OSError("the village link dropped"))

    async def _serve(self, store: BeingsStore, owner: str, slug: str, ws,
                     req: dict) -> None:
        rid = req.get("id")
        try:
            body = await handle_link_request(store, owner, slug,
                                             str(req.get("op") or ""), req)
            resp = {"t": "res", "id": rid, "ok": True, "body": body}
        except BeingError as e:
            resp = {"t": "res", "id": rid, "ok": False, "status": e.status,
                    "error": str(e)}
        except Exception as e:  # noqa: BLE001
            resp = {"t": "res", "id": rid, "ok": False, "status": 500,
                    "error": str(e)}
        try:
            await ws.send(json.dumps(resp))
        except Exception:  # noqa: BLE001
            pass

    async def probe(self, store: BeingsStore, being: dict) -> dict:
        """One-shot connect (for instant feedback on save): does the link
        establish and the village accept us? Returns {ok, error?}."""
        from websockets.asyncio.client import connect
        url = being.get("visit_url") or ""
        if not url:
            return {"ok": False, "error": "no target village set"}
        secret = being.get("visit_secret") or ""
        try:
            async with connect(_ws_url(url), open_timeout=12,
                               max_size=_MAX_FRAME) as ws:
                await ws.send(json.dumps({
                    "t": "hello", "probe": True, "secret": secret,
                    "slug": being["slug"], "name": being["name"],
                    "home": _home_label(store, being["owner_id"]),
                    "profile": being_life.public_profile(store, being)}))
                welcome = json.loads(await asyncio.wait_for(ws.recv(), timeout=12))
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "error": f"couldn't reach {url}: {e}"}
        if isinstance(welcome, dict) and welcome.get("t") == "welcome":
            store.mark_announced(being["id"])
            return {"ok": True}
        return {"ok": False,
                "error": (welcome or {}).get("message") if isinstance(welcome, dict)
                else "the village said no"}


village_client = VillageClient()
