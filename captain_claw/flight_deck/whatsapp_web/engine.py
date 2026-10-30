"""The service's links: one per Flight Deck user, each with its own client.

Every operation on one user's link runs under that link's lock, so a double
click on Link or an unlink racing a reconnect can't leave two clients on one
session (WhatsApp would answer with "stream replaced").

The client comes from an *adapter factory* (:func:`.neonize_adapter.make_adapter`
in the service, a fake in tests), so everything here runs without WhatsApp.

Bells: :meth:`Engine.events_after` hands Flight Deck the notices
:func:`.link_state.apply_event` raised (unlinked on the phone, stream
replaced, temporary ban, outdated client); Flight Deck rings them.

A link's messages outlive it: every path that ends a link moves them aside
(:mod:`.kept_history`), and only the user's own choice deletes them.
"""

from __future__ import annotations

import asyncio
import logging
import re
import shutil
import sqlite3
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from captain_claw.flight_deck.whatsapp_web import kept_history
from captain_claw.flight_deck.whatsapp_web.link_state import (
    NOT_RESUMED,
    LinkState,
    apply_event,
    normalize_phone,
    now_iso,
)
from captain_claw.flight_deck.whatsapp_web.store import (
    MAX_DAILY_LIMIT,
    MAX_TRIGGER_CHARS,
    MAX_TRIGGERS,
    RETENTION_CHOICES,
    ChatNotFound,
    Store,
    merge_chats,
    merge_contacts,
    merge_messages,
    pick_chat,
)

log = logging.getLogger("whatsapp_web.engine")

# How long a QR / pairing code may wait for the phone before the link is dropped.
PAIRING_TTL = 180.0
# History asked for at link time ("keep everything" → ten years; see the plan).
HISTORY_DAYS = 3650
# Bells kept for Flight Deck to collect.
_MAX_EVENTS = 500
_LOGOUT_TIMEOUT = 15.0
_STOP_TIMEOUT = 10.0
# A folder name under the data dir: no "." / ".." / hidden names, nothing that climbs out.
_USER_ID = re.compile(r"^[A-Za-z0-9_-][A-Za-z0-9_.-]{0,79}$")
_SESSION_FILES = ("session.sqlite3", "session.sqlite3-wal", "session.sqlite3-shm", "session.sqlite3-journal")
# Events that only feed the store: state.json isn't rewritten for each.
_STORE_ONLY = frozenset({"message", "directory"})
_PURGE_EVERY = 3600.0
# Bigger attachments are refused rather than pulled into memory.
MAX_ATTACHMENT_BYTES = 100 * 1024 * 1024
# What WhatsApp's servers say when a file is gone (media expires there).
_GONE_MARKERS = ("404", "410", "not found", "expired", "gone", "no longer")
# Sending (Phase 3).
SEND_PER_CHAT_MINUTE = 6
REPEAT_WINDOW = 86400.0
MAX_SEND_TEXT = 4096
MAX_SIGNATURE = 120
SEND_TIMEOUT = 30.0
# A send error that leaves the outcome open: WhatsApp may have the message.
_UNKNOWN_MARKERS = ("timeout", "timed out", "deadline", "context canceled", "eof", "connection reset")


class Adapter(Protocol):
    async def connect(self) -> None: ...
    async def pair_code(self, phone: str) -> str: ...
    async def download(self, field: str, raw_hex: str) -> bytes: ...
    async def send_text(self, jid: str, text: str) -> str: ...
    async def logout(self) -> None: ...
    async def stop(self) -> None: ...


EventSink = Callable[[str, dict[str, Any]], Awaitable[None]]
AdapterFactory = Callable[..., Adapter]


class LinkError(Exception):
    """A request the engine refuses; ``status`` is the HTTP status to answer with."""

    def __init__(self, status: int, detail: str) -> None:
        super().__init__(detail)
        self.status = status
        self.detail = detail


@dataclass
class _Link:
    user_id: str
    dir: Path
    state: LinkState
    adapter: Adapter | None = None
    store: Store | None = None
    # Bumped for every client started, so a stopped client's late events are ignored.
    gen: int = 0
    watchdog: asyncio.Task | None = None
    pair_code_at: float = 0.0
    started_at: float = 0.0

    @property
    def session(self) -> Path:
        return self.dir / "session.sqlite3"

    @property
    def state_file(self) -> Path:
        return self.dir / "state.json"

    @property
    def store_file(self) -> Path:
        return self.dir / "store.sqlite3"


def _qr_svg(code: str) -> str:
    """The QR as an SVG data URI, or "" when segno (a neonize dependency) is missing."""
    try:
        import segno
    except ImportError:
        return ""
    return segno.make_qr(code).svg_data_uri(scale=6, border=2)


class Engine:
    def __init__(self, root: Path, *, adapter_factory: AdapterFactory | None,
                 unavailable: str = "", clock: Callable[[], float] = time.time,
                 pairing_ttl: float = PAIRING_TTL) -> None:
        self.root = root
        self.unavailable = unavailable
        self._factory = adapter_factory
        self._clock = clock
        self._pairing_ttl = pairing_ttl
        self._links: dict[str, _Link] = {}
        # What's kept from earlier links, opened on first read: (user, entry) → store.
        self._kept_open: dict[tuple[str, str], Store] = {}
        # One lock per user id, created before the link exists: two Link clicks
        # for a new user must not start two clients on one session.
        self._locks: dict[str, asyncio.Lock] = {}
        self._events: list[dict[str, Any]] = []
        self._seq = 0

    # ── helpers ─────────────────────────────────────────────────────

    def _dir(self, user_id: str) -> Path:
        if not _USER_ID.match(user_id or ""):
            raise LinkError(400, "Bad user id")
        return self.root / user_id

    def _lock(self, user_id: str) -> asyncio.Lock:
        return self._locks.setdefault(user_id, asyncio.Lock())

    def _require_runtime(self) -> None:
        if self.unavailable or self._factory is None:
            raise LinkError(503, self.unavailable or "WhatsApp Web isn't available on this deck")

    def _get(self, user_id: str) -> _Link:
        link = self._links.get(user_id)
        if link is None:
            d = self._dir(user_id)
            st = LinkState.load(d / "state.json")
            link = _Link(user_id, d, st or LinkState(state="expired"))
            if st is not None:
                self._links[user_id] = link
        return link

    def _save(self, link: _Link) -> None:
        try:
            link.state.save(link.state_file)
        except OSError as exc:
            log.warning("whatsapp_web: could not save %s: %s", link.state_file, exc)

    def _retire_store(self, link: _Link, reason: str) -> None:
        """The link is ending and its messages outlive it. Raises when they can't
        be moved, so that no caller goes on to delete the folder they're in."""
        self._close_store(link)
        try:
            kept_history.archive(self.root, link.user_id, link.dir, jid=link.state.jid,
                                 linked_at=link.state.linked_at, reason=reason, now=self._clock())
        except OSError as exc:
            log.error("whatsapp_web: keeping %s's messages failed: %s", link.user_id, exc)
            raise LinkError(500, "The kept messages couldn't be moved, so nothing was deleted") from None

    def _wipe_session(self, link: _Link, reason: str) -> None:
        """Drop the session (state.json stays, so the card can say what happened);
        the messages move aside."""
        for name in _SESSION_FILES:
            try:
                (link.dir / name).unlink()
            except FileNotFoundError:
                pass
        try:
            self._retire_store(link, reason)
        except LinkError:
            pass  # they stay in the folder, and the next try is Link again's

    def _remove(self, link: _Link, reason: str, *, keep_history: bool = True) -> None:
        """Forget the link and its folder. Its messages are kept unless the user
        said otherwise."""
        if keep_history:
            self._retire_store(link, reason)
        else:
            self._close_store(link)
        self._links.pop(link.user_id, None)
        shutil.rmtree(link.dir, ignore_errors=True)

    def _kept_stores(self, user_id: str) -> list[tuple[dict[str, Any], Store]]:
        """(entry, its store) for what's kept from the user's earlier links,
        newest first. One that won't open doesn't stop the others being read."""
        out = []
        for entry in kept_history.entries(self.root, user_id):
            key = (user_id, entry["id"])
            store = self._kept_open.get(key)
            if store is None:
                try:
                    store = self._kept_open[key] = Store(kept_history.store_path(self.root, user_id, entry["id"]))
                except sqlite3.Error as exc:
                    log.warning("whatsapp_web: opening %s's kept %s: %s", user_id, entry["id"], exc)
                    continue
            out.append((entry, store))
        return out

    def _close_kept(self, user_id: str | None = None, entry_id: str = "") -> None:
        for key in [k for k in self._kept_open
                    if (user_id is None or k[0] == user_id) and (not entry_id or k[1] == entry_id)]:
            try:
                self._kept_open.pop(key).close()
            except Exception as exc:
                log.warning("whatsapp_web: closing %s's kept %s: %s", key[0], key[1], exc)

    def _open_store(self, link: _Link, *, create: bool = True) -> Store | None:
        """The link's store, opened on first use. With ``create=False`` a link
        whose folder is gone (unlinked meanwhile) gets None, not a new folder."""
        if link.store is None:
            if not create and not link.dir.is_dir():
                return None
            link.dir.mkdir(parents=True, mode=0o700, exist_ok=True)
            link.store = Store(link.store_file)
        return link.store

    def _close_store(self, link: _Link) -> None:
        store, link.store = link.store, None
        if store is not None:
            try:
                store.close()
            except Exception as exc:
                log.warning("whatsapp_web: closing %s's store: %s", link.user_id, exc)

    def _sink(self, link: _Link, gen: int) -> EventSink:
        async def sink(name: str, data: dict[str, Any]) -> None:
            if link.gen == gen and self._links.get(link.user_id) is link:
                await self._on_event(link, name, data)
        return sink

    async def _start_client(self, link: _Link) -> None:
        """Start *link*'s client (a fresh session pairs; an existing one resumes)."""
        assert self._factory is not None
        link.gen += 1
        link.adapter = self._factory(
            f"cc-{link.user_id}", link.session, self._sink(link, link.gen), history_days=HISTORY_DAYS)
        link.started_at = self._clock()
        try:
            await link.adapter.connect()
        except Exception as exc:
            log.error("whatsapp_web: %s did not start: %s", link.user_id, exc)
            link.adapter = None
            apply_event(link.state, "connect_failure", {"reason": str(exc)}, now=self._clock())
            link.state.state = "failed"
            self._save(link)
            raise LinkError(502, f"WhatsApp Web could not start: {exc}") from None

    async def _stop_client(self, link: _Link) -> None:
        adapter, link.adapter = link.adapter, None
        link.gen += 1  # whatever it still reports is ignored
        if link.watchdog is not None and link.watchdog is not asyncio.current_task():
            link.watchdog.cancel()
        link.watchdog = None
        if adapter is None:
            return
        try:
            await asyncio.wait_for(adapter.stop(), _STOP_TIMEOUT)
        except Exception as exc:
            log.warning("whatsapp_web: stopping %s: %s", link.user_id, exc)

    def _watch_pairing(self, link: _Link) -> None:
        async def watch() -> None:
            while link.adapter is not None and link.state.state == "pairing":
                await asyncio.sleep(2.0)
                deadline = max(link.started_at, link.pair_code_at) + self._pairing_ttl
                if link.state.state == "pairing" and self._clock() >= deadline:
                    async with self._lock(link.user_id):
                        if link.state.state != "pairing":
                            return
                        await self._stop_client(link)
                        apply_event(link.state, "expired", now=self._clock())
                        self._wipe_session(link, "expired")
                        self._save(link)
                    return

        link.watchdog = asyncio.get_running_loop().create_task(watch())

    async def _on_event(self, link: _Link, name: str, data: dict[str, Any]) -> None:
        notice = apply_event(link.state, name, data, now=self._clock())
        if name in ("logged_out", "stream_replaced", "outdated"):
            # Let go of the client from outside its own callback.
            asyncio.get_running_loop().create_task(self._after_terminal(link, name))
        if name in ("history", "message", "directory"):
            try:
                incoming = await asyncio.to_thread(self._ingest, link, name, data)
            except Exception as exc:
                log.error("whatsapp_web: storing %s for %s failed: %s", name, link.user_id, exc)
                incoming = None
            if incoming:
                # For Flight Deck's auto-replies (whatsapp_web_autoreply): a new
                # message in a chat the user lets the agent answer by itself.
                self._push_event(link, {"kind": "incoming", **incoming})
        if name not in _STORE_ONLY:
            self._save(link)
        if notice is not None:
            self._seq += 1
            self._events.append({
                "seq": self._seq, "user_id": link.user_id, "title": notice.title,
                "body": notice.body, "at": now_iso(self._clock()),
            })
            del self._events[:-_MAX_EVENTS]

    def _ingest(self, link: _Link, name: str, data: dict[str, Any]) -> dict[str, Any] | None:
        """Write one event's rows (worker thread; the store serialises itself).
        For a live message in an auto-reply chat, what Flight Deck needs to
        decide whether the agent answers."""
        if (not link.state.linked_at or link.state.state == "logged_out"
                or self._links.get(link.user_id) is not link):
            return  # unlinked or logged out while this event was on its way
        store = self._open_store(link, create=False)
        if store is None:
            return
        if name == "history":
            # Mappings first, so this chunk's LID chats land folded already.
            store.put_lid_map(data.get("lid_map") or [])
            store.put_chats(data.get("chats") or [])
            store.put_names(data.get("pushnames") or [])
            store.ingest(data.get("rows") or [], live=False)
        elif name == "message":
            store.ingest([data], live=True)
            return self._autoreply_candidate(link, store, data)
        elif name == "directory":
            store.put_contacts(data.get("contacts") or [])
            store.put_groups(data.get("groups") or [])

    @staticmethod
    def _autoreply_candidate(link: _Link, store: Store, row: dict[str, Any]) -> dict[str, Any] | None:
        """A message written in a chat that's whitelisted, has auto-reply on,
        while sending is on — else None. The user's own count too (they can call
        the agent in), never an agent's send, an edit, reaction or deletion."""
        if row.get("op") != "message" or not row.get("id"):
            return None
        config = store.autoreply_config()
        if not config["chats"] or not config["agent"]:
            return None
        chat = store.canon(str(row.get("chat") or ""))
        policy = store.send_policy()
        if not policy["enabled"] or chat not in policy["allowed"] or chat not in config["chats"]:
            return None
        activity = store.agent_activity(chat)
        if row.get("from_me") and str(row["id"]) in activity["agent_ids"]:
            return None
        title, is_group = store.chat_title(chat)
        sender = store.canon(str(row.get("sender") or ""))
        return {
            "user_id": link.user_id, "chat_jid": chat, "chat": title, "is_group": is_group,
            "id": str(row["id"]), "ts": int(row.get("ts") or 0), "kind_of_message": str(row.get("kind") or "text"),
            "from_owner": bool(row.get("from_me")),
            "sender": "" if row.get("from_me") else store.sender_name(sender, str(row.get("push_name") or "")),
            "text": str(row.get("text") or "")[:2000],
            "reply_to_agent": bool(row.get("reply_to")) and row["reply_to"] in activity["agent_ids"],
            "last_agent_ts": activity["last_agent_ts"], "last_owner_ts": activity["last_owner_ts"],
            "agent": config["agent"], "triggers": config["triggers"],
        }

    def _push_event(self, link: _Link, event: dict[str, Any]) -> None:
        self._seq += 1
        self._events.append({"seq": self._seq, "user_id": link.user_id, "at": now_iso(self._clock()), **event})
        del self._events[:-_MAX_EVENTS]

    async def _after_terminal(self, link: _Link, name: str) -> None:
        async with self._lock(link.user_id):
            if self._links.get(link.user_id) is not link:
                return  # unlinked meanwhile
            await self._stop_client(link)
            if name == "logged_out":
                # The session is worthless now, the messages aren't: this is also
                # how a number WhatsApp banned ends.
                self._wipe_session(link, "logged_out")
                self._save(link)

    # ── what Flight Deck asks ───────────────────────────────────────

    def view(self, user_id: str) -> dict[str, Any]:
        link = self._get(user_id)
        if user_id not in self._links:
            return {"state": "off", "running": False}
        out = link.state.view()
        out["running"] = link.adapter is not None
        out["has_session"] = link.session.exists()
        if link.state.state == "pairing" and link.state.qr:
            out["qr_svg"] = _qr_svg(link.state.qr)
            out["qr_at"] = now_iso(link.state.qr_at)
        expires = max(link.started_at, link.pair_code_at) + self._pairing_ttl
        if link.state.state == "pairing":
            out["pairing_expires_at"] = now_iso(expires)
        if link.state.linked_at and (link.store is not None or link.store_file.exists()):
            store = self._open_store(link)
            out["stats"] = store.stats()
            out["retention_days"] = store.retention_days()
        else:
            out["stats"] = None
            out["retention_days"] = None
        out["retention_choices"] = list(RETENTION_CHOICES)
        return out

    async def link(self, user_id: str, *, consented_at: str) -> dict[str, Any]:
        """Start pairing a fresh link (a QR follows). A link that is already
        pairing or connected is left alone; a dead one is replaced."""
        self._require_runtime()
        self._dir(user_id)
        async with self._lock(user_id):
            link = self._get(user_id)
            if link.adapter is not None and link.state.state in ("pairing", "connecting", "connected", "disconnected"):
                return self.view(user_id)
            await self._stop_client(link)
            self._retire_store(link, "replaced")
            shutil.rmtree(link.dir, ignore_errors=True)
            link.dir.mkdir(parents=True, mode=0o700, exist_ok=True)
            now = self._clock()
            link.state = LinkState(state="pairing", consented_at=consented_at, created_at=now_iso(now))
            link.pair_code_at = 0.0
            self._links[user_id] = link
            self._save(link)
            await self._start_client(link)
            self._watch_pairing(link)
        return self.view(user_id)

    async def reconnect(self, user_id: str) -> dict[str, Any]:
        """Start the client again on the session that's kept (after "stream
        replaced", a failed start or a service stop)."""
        self._require_runtime()
        self._dir(user_id)
        async with self._lock(user_id):
            link = self._get(user_id)
            if user_id not in self._links or not link.session.exists():
                raise LinkError(409, "There's no linked session to reconnect: link again")
            if link.adapter is not None:
                return self.view(user_id)
            link.state.state = "connecting"
            link.state.detail = ""
            self._save(link)
            await self._start_client(link)
        return self.view(user_id)

    async def pair_code(self, user_id: str, phone: str) -> dict[str, Any]:
        """An 8-character code to type on the phone instead of scanning the QR."""
        self._require_runtime()
        try:
            digits = normalize_phone(phone)
        except ValueError as exc:
            raise LinkError(422, str(exc)) from None
        self._dir(user_id)
        async with self._lock(user_id):
            link = self._get(user_id)
            if link.adapter is None or link.state.state != "pairing":
                raise LinkError(409, "Start linking first, then ask for a code")
            try:
                code = await link.adapter.pair_code(digits)
            except Exception as exc:
                raise LinkError(502, f"WhatsApp didn't give a pairing code: {exc}") from None
            link.pair_code_at = self._clock()
            apply_event(link.state, "pair_code", {"code": code}, now=self._clock())
            self._save(link)
        return self.view(user_id)

    async def cancel(self, user_id: str) -> dict[str, Any]:
        """Stop a link that isn't working (pairing, expired, failed, logged out)
        and forget it. Messages it has stay kept."""
        self._dir(user_id)
        async with self._lock(user_id):
            link = self._get(user_id)
            if user_id not in self._links:
                return self.view(user_id)
            if link.state.linked_at and link.state.state not in ("expired", "failed", "logged_out"):
                raise LinkError(409, "This link is paired: disconnect it instead")
            await self._stop_client(link)
            self._remove(link, "dismissed")
        return self.view(user_id)

    async def unlink(self, user_id: str, *, keep_history: bool = True) -> dict[str, Any]:
        """Log the device out (it leaves the phone's Linked devices), stop the
        client and forget the link. Its messages stay readable for the user's
        agents, unless they said to delete them."""
        self._dir(user_id)
        async with self._lock(user_id):
            link = self._get(user_id)
            if user_id not in self._links:
                return {"state": "off", "running": False, "removed_from_phone": False}
            removed = False
            if link.adapter is not None and link.state.state in ("connected", "disconnected", "temporary_ban"):
                # Our own logout may come back as a "logged out" event: not news.
                link.gen += 1
                try:
                    await asyncio.wait_for(link.adapter.logout(), _LOGOUT_TIMEOUT)
                    removed = True
                except Exception as exc:
                    log.warning("whatsapp_web: logout for %s failed: %s", user_id, exc)
            elif link.state.state == "logged_out":
                removed = True  # the phone already did it
            await self._stop_client(link)
            self._remove(link, "disconnected", keep_history=keep_history)
        out = self.view(user_id)
        out["removed_from_phone"] = removed
        return out

    async def delete_kept(self, user_id: str, entry_id: str = "") -> dict[str, Any]:
        """The user's own call: delete what's kept from one earlier link (or all)."""
        self._dir(user_id)
        async with self._lock(user_id):
            self._close_kept(user_id, entry_id)
            removed = kept_history.discard(self.root, user_id, entry_id)
        if entry_id and not removed:
            raise LinkError(404, "Nothing like that is kept")
        return {"removed": removed}

    async def resume_all(self) -> int:
        """After a (re)start: reconnect every paired session that should run.
        A link caught mid-pairing is dropped — its QR is long gone."""
        if self.unavailable or self._factory is None or not self.root.is_dir():
            return 0
        started = 0
        for d in sorted(p for p in self.root.iterdir() if p.is_dir()):
            if not _USER_ID.match(d.name):
                continue
            link = self._get(d.name)
            if d.name not in self._links:
                continue
            st = link.state
            if st.state == "pairing" or (not st.linked_at and st.state not in NOT_RESUMED):
                try:
                    self._remove(link, "never paired")
                except LinkError:
                    pass
                continue
            if st.state in NOT_RESUMED or not link.session.exists():
                continue
            try:
                async with self._lock(link.user_id):
                    st.state, st.detail = "connecting", ""
                    self._save(link)
                    await self._start_client(link)
                started += 1
            except LinkError:
                pass
        return started

    async def shutdown(self) -> None:
        """Stop every client (no logout: the links stay on the phone)."""
        for link in list(self._links.values()):
            async with self._lock(link.user_id):
                await self._stop_client(link)
                if link.state.state in ("connected", "disconnected", "connecting"):
                    link.state.state = "connecting"
                    link.state.detail = "Waiting for the WhatsApp Web service"
                self._save(link)
                self._close_store(link)
        self._close_kept()

    # ── reading (what agents and the card see) ──────────────────────

    def _live(self, user_id: str) -> Store | None:
        """The linked account's store; None when no account is linked."""
        link = self._get(user_id)
        if user_id not in self._links or not link.state.linked_at or link.state.state == "logged_out":
            return None
        return self._open_store(link)

    def _reader(self, user_id: str) -> Store:
        """The linked account's store. Sending and its rules stop here: an earlier
        number's chats are nothing this number may write to."""
        store = self._live(user_id)
        if store is None:
            raise LinkError(409, "WhatsApp isn't linked for this user")
        return store

    def _readers(self, user_id: str, *, with_kept: bool) -> list[Store]:
        """The stores a read covers: the linked account's first, then (*with_kept*)
        what's kept from earlier links, newest first."""
        live = self._live(user_id)
        stores = [live] if live is not None else []
        if with_kept:
            stores += [store for _, store in self._kept_stores(user_id)]
        if not stores:
            raise LinkError(409, "WhatsApp isn't linked for this user")
        return stores

    @staticmethod
    async def _find_chat(stores: list[Store], ref: str) -> dict[str, Any]:
        lists = [await asyncio.to_thread(s.chats) for s in stores]
        try:
            return pick_chat(merge_chats(lists), ref)
        except ChatNotFound as exc:
            options = "; ".join(f"{c['title']} (id {c['jid']})" for c in exc.candidates)
            raise LinkError(404, f"{exc}{': ' + options if options else ''}") from None

    async def chats(self, user_id: str, *, with_kept: bool = False) -> dict[str, Any]:
        stores = self._readers(user_id, with_kept=with_kept)
        live = self._live(user_id)
        lists = [await asyncio.to_thread(s.chats) for s in stores]
        return {"chats": merge_chats(lists, live=live is not None),
                "stats": await asyncio.to_thread(live.stats) if live is not None else None,
                "state": self._links[user_id].state.state if user_id in self._links else "off",
                "kept": kept_history.entries(self.root, user_id) if with_kept else []}

    async def messages(self, user_id: str, *, chat: str = "", query: str = "", since: int | None = None,
                       until: int | None = None, limit: int = 50, agent: str = "",
                       mark_seen: bool = False, with_kept: bool = False) -> dict[str, Any]:
        """Messages newest first, of one chat or all; a look at a chat's (or all
        chats') newest messages by *agent* counts as seen for its news line."""
        stores = self._readers(user_id, with_kept=with_kept)
        found = await self._find_chat(stores, chat) if str(chat or "").strip() else None
        limit = max(1, min(int(limit or 50), 200))
        lists = [await asyncio.to_thread(s.messages, chat_jid=found["jid"] if found else None, since=since,
                                         until=until, query=query, limit=limit) for s in stores]
        live = self._live(user_id)
        if mark_seen and agent and live is not None and lists[0] and until is None and not str(query or "").strip():
            await asyncio.to_thread(live.mark_read, agent, lists[0])  # the cursors are the live store's
        return {"chat": found, "messages": merge_messages(lists, limit)}

    async def contacts(self, user_id: str, *, query: str = "", limit: int = 20) -> dict[str, Any]:
        stores = self._readers(user_id, with_kept=True)
        lists = [await asyncio.to_thread(s.contacts, query, limit) for s in stores]
        return {"contacts": merge_contacts(lists, limit)}

    async def news(self, user_id: str, *, agent: str) -> dict[str, Any]:
        store = self._live(user_id)
        if store is None:
            # Nothing arrives, but what's kept can still be read: the tool stays.
            if kept_history.entries(self.root, user_id):
                return {"connected": True, "chats": [], "total": 0, "first_look": False}
            return {"connected": False}
        return {"connected": True, **await asyncio.to_thread(store.news, agent or "_", now=self._clock())}

    async def attachment(self, user_id: str, *, chat: str, message_id: str) -> tuple[bytes, dict[str, Any]]:
        """A message's file (photo, voice note, document …), downloaded now from
        WhatsApp's servers — the link has to be up. Returns (bytes, details)."""
        stores = self._readers(user_id, with_kept=True)
        link = self._links.get(user_id)
        found = await self._find_chat(stores, chat)
        item = None
        for store in stores:
            item = await asyncio.to_thread(store.media, found["jid"], message_id)
            if item is not None:
                break
        if item is None:
            raise LinkError(404, f"No message #{message_id} in {found['title']}")
        if item["deleted"]:
            raise LinkError(410, "That message was deleted, and its file with it")
        media = item["media"] or {}
        if not media.get("raw") or not media.get("field"):
            raise LinkError(404, f"Message #{message_id} has no file (it's {item['kind']})")
        if int(media.get("size") or 0) > MAX_ATTACHMENT_BYTES:
            raise LinkError(413, f"The file is too large ({int(media['size']) // (1024 * 1024)} MB; "
                                 f"the limit is {MAX_ATTACHMENT_BYTES // (1024 * 1024)} MB)")
        if link is None or link.adapter is None or link.state.state not in ("connected", "disconnected"):
            raise LinkError(409, "WhatsApp isn't connected right now, and files come from WhatsApp's "
                                 "servers: the user can check the link in Flight Deck → Connections")
        try:
            data = await link.adapter.download(str(media["field"]), str(media["raw"]))
        except Exception as exc:
            text = str(exc)
            if any(m in text.lower() for m in _GONE_MARKERS):
                raise LinkError(410, "This file is no longer on WhatsApp's servers (it expired). The "
                                     "phone may still have it.") from None
            raise LinkError(502, f"WhatsApp didn't hand the file over: {text or type(exc).__name__}") from None
        if not data:
            raise LinkError(502, "WhatsApp handed over an empty file")
        if len(data) > MAX_ATTACHMENT_BYTES:
            raise LinkError(413, "The file is too large")
        return data, {
            "id": item["id"], "kind": item["kind"], "chat": found["title"], "chat_jid": found["jid"],
            "ts": item["ts"], "from_me": item["from_me"], "mimetype": str(media.get("mimetype") or ""),
            "file_name": str(media.get("file_name") or ""), "seconds": int(media.get("seconds") or 0),
            "size": len(data),
        }

    # ── sending (Phase 3): automatic, only to whitelisted chats ─────

    async def send_policy(self, user_id: str) -> dict[str, Any]:
        store = self._reader(user_id)
        policy = await asyncio.to_thread(store.send_policy)
        policy["sends"] = await asyncio.to_thread(store.sends, 20)
        counts = await asyncio.to_thread(store.send_counts, "", now=self._clock())
        policy["sent_today"] = counts["day"]
        return policy

    async def set_send_policy(self, user_id: str, changes: dict[str, Any]) -> dict[str, Any]:
        """Change the sending rules. The whitelist keeps only chats that already
        exist here (a group we're in, or a chat with messages): a number typed in
        never becomes a chat Flight Deck may start."""
        store = self._reader(user_id)
        policy = await asyncio.to_thread(store.send_policy)
        dropped: list[str] = []
        if changes.get("enabled") is not None:
            policy["enabled"] = bool(changes["enabled"])
        if changes.get("allowed") is not None:
            wanted = [str(j).strip() for j in changes["allowed"] if str(j).strip()]
            kept = []
            for jid in dict.fromkeys(wanted):
                (kept if await asyncio.to_thread(store.chat_has_history, jid) else dropped).append(jid)
            policy["allowed"] = kept
        if changes.get("daily_limit") is not None:
            try:
                limit = int(changes["daily_limit"])
            except (TypeError, ValueError):
                raise LinkError(422, "The daily limit is a number") from None
            if not 1 <= limit <= MAX_DAILY_LIMIT:
                raise LinkError(422, f"The daily limit is 1 to {MAX_DAILY_LIMIT} messages")
            policy["daily_limit"] = limit
        if changes.get("signature") is not None:
            sig = " ".join(str(changes["signature"]).split())
            if len(sig) > MAX_SIGNATURE:
                raise LinkError(422, f"Keep the signature to {MAX_SIGNATURE} characters")
            policy["signature"] = sig
        if changes.get("signature_on") is not None:
            policy["signature_on"] = bool(changes["signature_on"])
        await asyncio.to_thread(store.set_send_policy, policy)
        out = await self.send_policy(user_id)
        out["dropped"] = dropped
        return out

    async def send(self, user_id: str, *, chat: str, text: str, agent: str = "",
                   allow_repeat: bool = False) -> dict[str, Any]:
        """Send one text message for an agent — automatic once the user turned
        sending on, and only to a chat on their whitelist. Every refusal says why."""
        store = self._reader(user_id)
        link = self._links[user_id]
        policy = await asyncio.to_thread(store.send_policy)
        if not policy["enabled"]:
            raise LinkError(403, "Sending is off: the user can turn it on in Flight Deck → Connections → "
                                 "WhatsApp Web → Sending")
        body = str(text or "").strip()
        if not body:
            raise LinkError(422, "Nothing to send: 'text' is empty")
        if len(body) > MAX_SEND_TEXT:
            raise LinkError(422, f"The message is too long ({len(body)} characters; at most {MAX_SEND_TEXT})")
        try:
            found = await asyncio.to_thread(store.find_chat, chat)
        except ChatNotFound as exc:
            options = "; ".join(f"{c['title']} (id {c['jid']})" for c in exc.candidates)
            if not exc.candidates and (former := await self._kept_chat(user_id, chat)):
                raise LinkError(403, f"{former['title']} is only in the history kept from an earlier number. "
                                     "This number has no conversation with them, and Flight Deck never "
                                     "starts one") from None
            raise LinkError(404, f"{exc}{': ' + options if options else ''}") from None
        jid = found["jid"]
        if jid not in policy["allowed"]:
            raise LinkError(403, f"{found['title']} isn't on the user's sending whitelist: only they can add "
                                 "it, in Flight Deck → Connections → WhatsApp Web → Sending")
        if not await asyncio.to_thread(store.chat_has_history, jid):
            raise LinkError(403, f"There's no conversation with {found['title']} yet, and Flight Deck never "
                                 "starts one")
        if link.state.state == "temporary_ban":
            raise LinkError(409, "WhatsApp has temporarily banned this account: nothing is sent until it lifts")
        if link.adapter is None or link.state.state != "connected":
            raise LinkError(409, "WhatsApp isn't connected right now, so nothing was sent: the user can check "
                                 "the link in Flight Deck → Connections")
        now = self._clock()
        counts = await asyncio.to_thread(store.send_counts, jid, now=now)
        if counts["day"] >= policy["daily_limit"]:
            raise LinkError(429, f"The daily limit is reached ({policy['daily_limit']} messages in 24 hours)")
        if counts["minute"] >= SEND_PER_CHAT_MINUTE:
            raise LinkError(429, f"Too many messages to {found['title']} in the last minute; wait a little")
        composed = body
        if policy["signature_on"] and policy["signature"]:
            composed = f"{body}\n\n{policy['signature']}"
        if not allow_repeat:
            same = await asyncio.to_thread(store.recent_same, jid, composed, since=now - REPEAT_WINDOW)
            if same == "unknown":
                raise LinkError(409, f"This exact message to {found['title']} may already have gone out (WhatsApp "
                                     "didn't confirm the earlier try): check the chat before sending it again; "
                                     "pass allow_repeat to send it anyway")
            if same:
                raise LinkError(409, f"This exact message went to {found['title']} in the last 24 hours; pass "
                                     "allow_repeat if it really should go again")
        rowid = await asyncio.to_thread(store.start_send, jid, agent, composed, now=now)
        try:
            wa_id = await asyncio.wait_for(link.adapter.send_text(jid, composed), SEND_TIMEOUT)
        except Exception as exc:
            text_err = str(exc) or type(exc).__name__
            unknown = isinstance(exc, asyncio.TimeoutError) or any(m in text_err.lower() for m in _UNKNOWN_MARKERS)
            await asyncio.to_thread(store.finish_send, rowid, "unknown" if unknown else "failed", error=text_err)
            if unknown:
                self._bell(link, f"{agent or 'An agent'} may have sent a WhatsApp message",
                           f"WhatsApp didn't confirm it — check the chat. To {found['title']}: {body[:120]}")
                raise LinkError(504, f"WhatsApp didn't confirm the message to {found['title']}; it may have gone "
                                     "out. Don't send it again before the user checks the chat.") from None
            raise LinkError(502, f"WhatsApp refused the message to {found['title']}: {text_err}") from None
        await asyncio.to_thread(store.finish_send, rowid, "sent", wa_id=wa_id)
        # WhatsApp doesn't echo a linked device's own sends back: keep the
        # message, so the chat reads as it is (agents, auto-replies).
        try:
            await asyncio.to_thread(store.ingest, [{
                "op": "message", "chat": jid, "id": wa_id, "sender": link.state.jid or "", "from_me": True,
                "is_group": jid.endswith("@g.us"), "ts": int(now), "push_name": "", "kind": "text",
                "text": composed, "media": None, "reply_to": "",
            }], live=False)
        except Exception as exc:
            log.warning("whatsapp_web: keeping a sent message for %s failed: %s", user_id, type(exc).__name__)
        self._bell(link, f"{agent or 'An agent'} sent a WhatsApp message", f"To {found['title']}: {body[:160]}")
        return {"status": "sent", "id": wa_id, "chat": found["title"], "chat_jid": jid, "text": composed,
                "signed": composed != body}

    async def _kept_chat(self, user_id: str, ref: str) -> dict[str, Any] | None:
        """The chat *ref* names in what's kept from earlier links, if one does."""
        lists = [await asyncio.to_thread(store.chats) for _, store in self._kept_stores(user_id)]
        try:
            return pick_chat(merge_chats(lists), ref)
        except ChatNotFound:
            return None

    # ── auto-replies (Flight Deck decides and writes; see whatsapp_web_autoreply) ──

    async def autoreply_config(self, user_id: str) -> dict[str, Any]:
        store = self._reader(user_id)
        return await asyncio.to_thread(store.autoreply_config)

    async def set_autoreply_config(self, user_id: str, changes: dict[str, Any]) -> dict[str, Any]:
        """Which chats the agent answers by itself (only whitelisted ones), which
        agent (Flight Deck checks it's the user's own), extra trigger words."""
        store = self._reader(user_id)
        config = await asyncio.to_thread(store.autoreply_config)
        policy = await asyncio.to_thread(store.send_policy)
        dropped: list[str] = []
        if changes.get("chats") is not None:
            wanted = list(dict.fromkeys(str(j).strip() for j in changes["chats"] if str(j).strip()))
            config["chats"] = [j for j in wanted if j in policy["allowed"]]
            dropped = [j for j in wanted if j not in policy["allowed"]]
        if changes.get("agent") is not None:
            config["agent"] = str(changes["agent"]).strip()[:200]
        if changes.get("triggers") is not None:
            words: list[str] = []
            for raw in changes["triggers"]:
                word = " ".join(str(raw).split())
                if not word:
                    continue
                if not 2 <= len(word) <= MAX_TRIGGER_CHARS:
                    raise LinkError(422, f"A trigger word is 2 to {MAX_TRIGGER_CHARS} characters")
                if word.casefold() not in {w.casefold() for w in words}:
                    words.append(word)
            if len(words) > MAX_TRIGGERS:
                raise LinkError(422, f"At most {MAX_TRIGGERS} trigger words")
            config["triggers"] = words
        await asyncio.to_thread(store.set_autoreply_config, config)
        return {**config, "dropped": dropped}

    async def add_autoreply_log(self, user_id: str, *, chat: str, outcome: str,
                                details: dict[str, Any]) -> dict[str, Any]:
        """Keep one of Flight Deck's auto-reply decisions for the card."""
        store = self._reader(user_id)
        await asyncio.to_thread(store.add_autoreply_log, store.canon(chat), outcome, details, now=self._clock())
        return {"ok": True}

    async def autoreply_log(self, user_id: str, *, limit: int = 50, chat: str = "",
                            ignored: bool = True) -> dict[str, Any]:
        store = self._reader(user_id)
        return {"entries": await asyncio.to_thread(store.autoreply_log, limit=limit, chat_jid=chat,
                                                   ignored=ignored)}

    def _bell(self, link: _Link, title: str, body: str) -> None:
        self._seq += 1
        self._events.append({"seq": self._seq, "user_id": link.user_id, "title": title[:120],
                             "body": body[:300], "at": now_iso(self._clock())})
        del self._events[:-_MAX_EVENTS]

    async def set_retention(self, user_id: str, days: int | None) -> dict[str, Any]:
        if days is not None and days not in RETENTION_CHOICES:
            raise LinkError(422, "Keep messages for 7, 30, 90 or 365 days, or keep everything")
        store = self._reader(user_id)
        await asyncio.to_thread(store.set_meta, "retention_days", "" if days is None else str(days))
        await asyncio.to_thread(store.purge, now=self._clock())
        return self.view(user_id)

    async def maintain(self, every: float = _PURGE_EVERY) -> None:
        """Forever: drop what's older than each user's retention choice."""
        while True:
            await asyncio.sleep(every)
            for link in list(self._links.values()):
                if link.store is not None or link.store_file.exists():
                    try:
                        await asyncio.to_thread(self._open_store(link).purge, now=self._clock())
                    except Exception as exc:
                        log.warning("whatsapp_web: purge for %s failed: %s", link.user_id, exc)
            await self._purge_kept()

    async def _purge_kept(self) -> None:
        """A retention the user chose goes on counting for what's kept from that link."""
        for user_id in kept_history.users(self.root):
            for entry, store in self._kept_stores(user_id):
                try:
                    if not await asyncio.to_thread(store.purge, now=self._clock()):
                        continue
                    stats = await asyncio.to_thread(store.stats)
                    if stats["messages"]:
                        kept_history.set_stats(self.root, user_id, entry["id"], stats)
                    else:
                        self._close_kept(user_id, entry["id"])
                        kept_history.discard(self.root, user_id, entry["id"])
                except Exception as exc:
                    log.warning("whatsapp_web: purge of %s's kept %s failed: %s", user_id, entry["id"], exc)

    def events_after(self, seq: int) -> list[dict[str, Any]]:
        return [e for e in self._events if e["seq"] > seq]

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for link in self._links.values():
            out[link.state.state] = out.get(link.state.state, 0) + 1
        return out
