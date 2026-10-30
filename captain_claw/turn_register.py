"""Every turn this agent runs, reported to Flight Deck's Taskboard.

The Taskboard (``flight_deck/taskboard_ingest.py``) turns what people ask an
agent into tasks. Only the agent sees every turn — a Telegram turn, a
WhatsApp turn or a scheduled job never passes through one place in Flight
Deck — so :class:`~captain_claw.agent.Agent` reports each one here where every
turn passes (``complete()`` / ``stream()``, right after the turn's origin is
set):

* an **open** record when the turn starts: who asked (the member on a shared
  agent), on which channel, where it came from (``msg_origin``,
  ``turn_source``), the start of what was asked and the ids that tie it to a
  chat, a WhatsApp number, a Telegram chat or the automation that sent it;
* a **close** record when it ends: how (``ok``, ``error``, ``blocked`` by a
  guard, ``cancelled``), the start of the reply, the files saved during the
  turn and how many tool calls it made and how many of those failed.

What an automation runs for comes from where it starts the turn: Flight
Deck's chat frame names an Autonomous Work action or a connected system's
message (``automation.ref``, noted by ``ws_handler``), this agent's own cron
names its job (``prompt_execution``), and a scheduled task's session names
its job. The owner's own Telegram chat (the owner lane) says so too.

A turn's id is in :data:`current_turn_id` while it runs, so each tool call it
makes carries it (``tool_audit.py``). Only the outermost entry is a turn: a
``complete()`` that runs inside one (``stream()`` re-entering it, a worker
called from a turn) opens nothing.

Same transport as the tool-call register (``tool_audit.py``): masked first
(:func:`tool_audit.mask_text`), spooled in ``turn_register_spool.db``,
numbered with the process's ``boot`` and a ``seq`` of its own, shipped in
batches to ``POST /fd/taskboard/agent/turns`` with the agent's own auth.
On only when the tool-call register is, and off with
``CLAW_TURN_REGISTER=off``. It never raises into a turn and never holds one
up: a record that can't be written is logged and dropped.
"""

from __future__ import annotations

import asyncio
import contextvars
import json
import os
import re
import secrets
import sqlite3
import threading
import time
import weakref
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from captain_claw import tool_audit
from captain_claw.logging import get_logger

log = get_logger(__name__)

INGEST_PATH = "/fd/taskboard/agent/turns"

OUTCOMES = ("ok", "error", "blocked", "cancelled")

_HEAD_MAX = 1000          # text_head / reply_head, after masking
_BATCH = 200
_SPOOL_CAP = 20_000
_MAX_DELIVERABLES = 20
_WALK_MAX = 2000          # entries looked at for a turn's saved files
_DETAIL_MAX = 64

# A turn the input or output guard refused (agent_guard_mixin._enforce_guard).
_GUARD_BLOCK_RE = re.compile(r"^Blocked by \w+ guard\b")

# A source (turn_source) → the Taskboard's channel.
_SOURCE_CHANNELS = {
    "web": "web", "app": "web", "glasses": "glasses", "whatsapp": "whatsapp", "telegram": "telegram",
    "messenger": "messenger", "device": "device", "scheduled": "scheduled", "autonomy": "autonomy",
    "api": "api", "link": "link", "member": "member",
}
# An automation kind (mail_authority) → the automation the board knows.
_AUTOMATIONS = {
    "fd_scheduler": "scheduled", "cron": "scheduled",
    "autonomy": "autonomy", "autonomy_tool": "autonomy", "plan": "autonomy",
    "link": "link",
}
# What an automation's own id may look like (an Autonomous Work action's, a
# connected system's message's, a cron job's).
_REF_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")
# The automation field Flight Deck names that id in (fd_dispatch, agent_link).
REF_FIELD = "ref"


@dataclass
class Turn:
    turn_id: str
    started_at: float
    tool_calls: int = 0         # counted by tool_audit as the turn's calls start
    failed_tool_calls: int = 0  # … and end in an error, a timeout or an abort
    closed: bool = False
    _tokens: tuple[contextvars.Token, contextvars.Token] | None = field(default=None, repr=False)


# The running turn: its id (what tool calls carry) and its state.
current_turn_id: ContextVar[str] = ContextVar("claw_turn_id", default="")
_CURRENT: ContextVar[Turn | None] = ContextVar("claw_turn", default=None)
# What the automation that starts the next turn here runs for: ``(its
# mail_authority kind, its id, a name)`` — an Autonomous Work action, a
# connected system's message, one of this agent's own cron jobs.
_AUTOMATION_REF: ContextVar[tuple[str, str, str]] = ContextVar("claw_turn_automation_ref", default=("", "", ""))
# The chat the next turn's frame came from, for a reply to it: ``(kind,
# address)`` — a device's channel on Flight Deck's bus, a person on Messenger.
_FRAME_ORIGIN: ContextVar[tuple[str, str]] = ContextVar("claw_turn_frame_origin", default=("", ""))
_FRAME_ORIGIN_KINDS = ("channel", "messenger")


def bind_automation(kind: str, ref: Any, name: Any = "") -> contextvars.Token:
    """The turns started in this context run for automation *kind*'s *ref*
    (a cron job's id, with its *name*); pair with :func:`reset_automation`.
    A ref that isn't an id binds nothing."""
    ref = str(ref or "").strip()
    first = next((ln.strip() for ln in str(name or "").splitlines() if ln.strip()), "")
    value = (str(kind or "").strip(), ref, first[:160]) if _REF_RE.match(ref) else ("", "", "")
    return _AUTOMATION_REF.set(value)


def reset_automation(token: contextvars.Token | None) -> None:
    if token is None:
        return
    try:
        _AUTOMATION_REF.reset(token)
    except (ValueError, RuntimeError):
        _AUTOMATION_REF.set(("", "", ""))


def note_frame_automation(raw: Any) -> None:
    """An agent socket's chat frame: what its ``automation`` marker names the
    turn it starts for (``ref`` — an Autonomous Work action's id, a connected
    system's message's), or nothing. Set for each chat frame, so one never
    carries the frame before's; the turn's task copies it when it starts.
    Never raises."""
    try:
        if isinstance(raw, dict):
            bind_automation(str(raw.get("kind") or ""), raw.get(REF_FIELD))
        else:
            _AUTOMATION_REF.set(("", "", ""))
    except Exception:
        pass


def note_frame_origin(raw: Any) -> None:
    """An agent socket's chat frame: where its turn came from as the bridge
    names it for a reply (``turn_origin`` — a device's channel on Flight
    Deck's bus, the person on Messenger), or nothing. The turn's own, kept
    on its record only — never the session's durable origin, which later
    results (a cron run) follow. Set for each chat frame, as
    :func:`note_frame_automation`. Never raises."""
    try:
        kind = str(raw.get("kind") or "").strip() if isinstance(raw, dict) else ""
        address = str(raw.get("address") or "").strip() if isinstance(raw, dict) else ""
        ok = kind in _FRAME_ORIGIN_KINDS and bool(_REF_RE.match(address))
        _FRAME_ORIGIN.set((kind, address) if ok else ("", ""))
    except Exception:
        pass


def current() -> Turn | None:
    """The turn running in this context (None outside one, or once it closed)."""
    turn = _CURRENT.get()
    return turn if turn is not None and not turn.closed else None


def enabled() -> bool:
    """On exactly when the tool-call register is, unless switched off."""
    if (os.environ.get("CLAW_TURN_REGISTER") or "").strip().lower() in ("0", "off", "false", "no"):
        return False
    return tool_audit.enabled()


# ── Spool ─────────────────────────────────────────────────────────────

_SPOOL_DDL = """
CREATE TABLE IF NOT EXISTS turns (
    boot  TEXT NOT NULL,
    seq   INTEGER NOT NULL,
    at    REAL NOT NULL,
    body  TEXT NOT NULL,
    PRIMARY KEY (boot, seq)
);
"""


class Spool:
    """The turn records not yet accepted by Flight Deck, in a SQLite file.
    Each record is sent once; a batch holds one boot's records."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        self._conn: sqlite3.Connection | None = None
        self._inserts = 0

    def _db(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = tool_audit.open_spool_db(self.path, _SPOOL_DDL)
        return self._conn

    def add(self, boot: str, seq: int, at: float, record: dict[str, Any]) -> None:
        with self._lock:
            db = self._db()
            db.execute("INSERT OR REPLACE INTO turns (boot, seq, at, body) VALUES (?, ?, ?, ?)",
                       (boot, seq, at, json.dumps(record, ensure_ascii=False, default=str)))
            db.commit()
            self._inserts += 1
            if self._inserts % 500 == 0:
                self._cap(db)

    def _cap(self, db: sqlite3.Connection) -> None:
        (n,) = db.execute("SELECT count(*) FROM turns").fetchone()
        if n > _SPOOL_CAP:
            db.execute("DELETE FROM turns WHERE rowid IN (SELECT rowid FROM turns "
                       "ORDER BY at LIMIT ?)", (n - _SPOOL_CAP,))
            db.commit()
            log.warning("Turn register spool full: dropped the oldest records", dropped=n - _SPOOL_CAP)

    def pending(self) -> bool:
        with self._lock:
            return self._db().execute("SELECT 1 FROM turns LIMIT 1").fetchone() is not None

    def batch(self, boot: str, now: float) -> tuple[list[dict[str, Any]], list[tuple[str, int]], list]:
        """``(records, keys to delete once sent, [])``: the oldest boot's
        records in order, each with its ``boot`` (taken off when posted)."""
        del boot, now   # every record is ready as soon as it is written
        records: list[dict[str, Any]] = []
        done: list[tuple[str, int]] = []
        with self._lock:
            db = self._db()
            first = db.execute("SELECT boot FROM turns ORDER BY at, seq LIMIT 1").fetchone()
            if first is None:
                return [], [], []
            rows = db.execute("SELECT seq, body FROM turns WHERE boot=? ORDER BY seq LIMIT ?",
                              (first[0], _BATCH)).fetchall()
            unreadable: list[tuple[str, int]] = []
            for seq, body in rows:
                try:
                    rec = json.loads(body)
                except ValueError:
                    unreadable.append((first[0], int(seq)))   # dropped (a gap at Flight Deck)
                    continue
                records.append({**rec, "boot": first[0], "seq": int(seq)})
                done.append((first[0], int(seq)))
            if unreadable:
                db.executemany("DELETE FROM turns WHERE boot=? AND seq=?", unreadable)
                db.commit()
        return records, done, []

    def settle(self, done: list[tuple[str, int]], started: list) -> None:
        del started
        with self._lock:
            db = self._db()
            db.executemany("DELETE FROM turns WHERE boot=? AND seq=?", done)
            db.commit()


_SEQ = 0
_SEQ_LOCK = threading.Lock()
_SPOOL: Spool | None = None
_SPOOL_FAILED = False
_SHIPPERS: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def _spool_path() -> Path:
    override = (os.environ.get("CLAW_TURN_REGISTER_SPOOL") or "").strip()
    if override:
        return Path(override).expanduser()
    return tool_audit._spool_path().with_name("turn_register_spool.db")


def _spool() -> Spool | None:
    global _SPOOL, _SPOOL_FAILED
    if _SPOOL is None and not _SPOOL_FAILED:
        try:
            _SPOOL = Spool(_spool_path())
        except Exception as exc:
            _SPOOL_FAILED = True
            log.warning("Turn register disabled: no spool", error=str(exc))
    return _SPOOL


def _next_seq() -> int:
    global _SEQ
    with _SEQ_LOCK:
        _SEQ += 1
        return _SEQ


def _enqueue(record: dict[str, Any]) -> bool:
    """Spool *record* under the next number. False when it couldn't be."""
    spool = _spool()
    if spool is None:
        return False
    seq = _next_seq()
    spool.add(tool_audit._BOOT, seq, time.time(), {"seq": seq, **record})
    _ensure_shipper()
    return True


# ── What the turn is ──────────────────────────────────────────────────


def _head(text: Any) -> str:
    """The start of *text*, secrets masked (masked before it is cut, so a
    key across the cut is still found)."""
    raw = str(text or "")
    return tool_audit.mask_text(raw[: _HEAD_MAX * 2])[:_HEAD_MAX]


def _short(value: Any, limit: int = _DETAIL_MAX) -> str:
    return str(value or "").strip()[:limit]


def channel_for(source: str, raw_channel: str = "") -> str:
    """The Taskboard's channel for a turn's source (``turn_source``) and the
    channel it arrived on; "" when it names none the board knows."""
    if str(raw_channel or "").strip().lower() == "mcp":
        return "mcp"
    source = str(source or "")
    if source.startswith("device:"):
        return source
    from captain_claw import turn_source

    return _SOURCE_CHANNELS.get(turn_source.channel_of(source), "")


def _automation(agent: Any) -> tuple[str, str, str, str, str]:
    """``(automation kind as bound, board automation, its id, the message
    id, a name)``: a Flight Deck scheduled job's id (its session's), this
    agent's own cron job's (and its first line), an Autonomous Work action's,
    a link's (and the connected system's message id); "" when the turn is a
    person's. An id bound for another automation kind is not this turn's."""
    try:
        from captain_claw import mail_authority

        bound = mail_authority.bound_authority()
    except Exception:
        bound = None
    if bound is None or bound.mode != "automated":
        return "", "", "", "", ""
    kind = _AUTOMATIONS.get(bound.kind, "")
    ref_kind, ref, name = _AUTOMATION_REF.get()
    if ref_kind != bound.kind:
        ref = name = ""
    task_id, msg_id = "", ""
    if kind in ("scheduled", "link"):
        meta = getattr(getattr(agent, "session", None), "metadata", None)
        task_id = _short(meta.get("scheduled_task") if isinstance(meta, dict) else "", 128)
        if kind == "link":
            task_id, msg_id = task_id.removeprefix("link-"), ref
    if bound.kind == "cron":
        task_id = ref or task_id   # this agent's own job
    elif kind == "autonomy":
        task_id = ref              # the action
    return bound.kind, kind, task_id, msg_id, name if bound.kind == "cron" else ""


def _open_record(agent: Any, turn: Turn) -> dict[str, Any]:
    from captain_claw import member_privacy, msg_origin, origin, speaker

    turn_origin = getattr(agent, "_turn_origin", None)
    if not (isinstance(turn_origin, tuple) and len(turn_origin) == 3):
        turn_origin = ("human", "", None)
    kind, detail, raw_channel = (str(turn_origin[0] or ""), str(turn_origin[1] or ""),
                                 str(turn_origin[2] or ""))
    src = getattr(agent, "_turn_source", None)
    if not (isinstance(src, tuple) and len(src) == 2):
        src = ("", "")
    source, source_label = str(src[0] or ""), str(src[1] or "")
    text = str(getattr(agent, "_turn_user_text", "") or "")
    session = getattr(agent, "session", None)
    channel = channel_for(source, raw_channel)
    literal = msg_origin.detect_literal(text)
    bound_kind, automation, automation_id, link_msg_id, automation_name = _automation(agent)

    p = speaker.principal_for(agent)
    group = speaker.is_group_principal(p)
    member = p is not None and not group
    speaker_id = str(p.speaker_id or "") if member else ""
    speaker_label = str(p.display_name or "") if p is not None else ""

    refs: dict[str, Any] = {}
    found = origin.get_session_origin(session)
    if found:
        refs["origin"] = {"kind": _short(found["kind"], 32), "address": _short(found["address"], 128)}
    frame_kind, frame_address = _FRAME_ORIGIN.get()
    if frame_kind:
        # The chat this turn came from (a device, a Messenger thread): a reply goes there.
        refs["turn_origin"] = {"kind": frame_kind, "address": frame_address}
    meta = getattr(session, "metadata", None)
    meta = meta if isinstance(meta, dict) else {}
    if channel == "whatsapp" and meta.get("whatsapp_waid"):
        refs["waid"] = _short(meta["whatsapp_waid"], 32)
    if channel == "telegram":
        # A group's chat is in its principal; a private chat's on the agent (web/telegram.py).
        chat_id = (p.speaker_id[len(speaker.GROUP_SPEAKER_PREFIX):] if group
                   else getattr(agent, "_telegram_chat_id", ""))
        if isinstance(chat_id, (int, str)) and str(chat_id).strip():
            refs["telegram_chat_id"] = _short(chat_id, 32)
        # The owner's own private chat (web/telegram.py's owner lane, the only
        # turn that sets _telegram_user_id): the deck's shared bot, or the
        # approved user the owner marked as themselves on a bot of the agent's.
        if not group and _short(getattr(agent, "_telegram_user_id", "")):
            refs["telegram_owner"] = True
    if automation == "link" and automation_id:
        refs["link_id"] = automation_id
        if link_msg_id:
            refs["link_msg_id"] = link_msg_id

    opener = {"role": "user", "origin": kind}
    origin_detail = {k: v for k, v in (
        ("detail", _short(detail)), ("channel", _short(raw_channel)), ("automation", _short(bound_kind)),
        ("literal_detail", _short(literal[1]) if literal else ""),
    ) if v}
    return {
        "phase": "open", "turn_id": turn.turn_id, "at": turn.started_at,
        "session_id": _short(getattr(session, "id", ""), 128),
        "session_name": _short(getattr(session, "name", ""), 160),
        "origin": _short(kind, 32), "origin_detail": origin_detail,
        "channel": channel, "source": _short(source), "source_label": _short(source_label, 80),
        "speaker_id": _short(speaker_id, 128), "speaker_label": _short(speaker_label, 80),
        "member": member,
        # A member's own words, or text relayed with the private header.
        "private": member or bool(member_privacy.header_level(text)),
        "opener": msg_origin.is_turn_opener(opener), "human": msg_origin.is_human_input(opener),
        "literal": literal[0] if literal else "",
        "text_head": _head(text), "ext_refs": refs,
        "automation": {"kind": automation, "id": automation_id,
                       **({"name": _head(automation_name)[:160]} if automation_name else {})},
    }


def _saved_folder(agent: Any) -> tuple[Path | None, str]:
    """``(saved/, the session's folder name)`` — read on the turn's own
    thread (the session may change once the next turn starts)."""
    from captain_claw.turn_deliverables import saved_root_of

    try:
        return saved_root_of(agent), str(agent._current_session_slug() or "")
    except Exception:
        return None, ""


def _deliverables(base: Path | None, slug: str, since: float) -> list[str]:
    """Files this turn saved for the user: under ``saved/<category>/<slug>/``,
    deliverables (``turn_deliverables``), changed since *since*; newest last,
    relative to ``saved/``. Working dirs (downloads, scripts…) and hidden
    folders are passed over by name, and only a file changed since the turn
    began is checked closely; at most ``_WALK_MAX`` entries are looked at."""
    from captain_claw.turn_deliverables import _is_working_dir, is_deliverable

    found: list[tuple[float, str]] = []
    try:
        if base is None or not slug or not base.is_dir():
            return []
        seen = 0
        with os.scandir(base) as categories:
            names = sorted(e.name for e in categories if not e.name.startswith(".") and e.is_dir())
        folders = [base / name / slug for name in names if not _is_working_dir(base, name)]
        stack = [f for f in reversed(folders) if f.is_dir()]
        while stack and seen <= _WALK_MAX:
            try:
                with os.scandir(stack.pop()) as it:
                    entries = list(it)
            except OSError:
                continue
            for entry in entries:
                seen += 1
                if seen > _WALK_MAX:
                    break
                if entry.name.startswith("."):
                    continue
                try:
                    if entry.is_dir(follow_symlinks=False):
                        stack.append(Path(entry.path))
                        continue
                    mtime = entry.stat().st_mtime   # cheap: only a file changed since is checked closely
                    if mtime >= since and is_deliverable(entry.path, base, since=since):
                        found.append((mtime, str(Path(entry.path).resolve().relative_to(base))))
                except (OSError, ValueError):
                    continue
    except Exception as exc:
        log.debug("Turn register: saved files not listed", error=str(exc))
    found.sort()
    return [rel for _m, rel in found[-_MAX_DELIVERABLES:]]


# ── Recording ─────────────────────────────────────────────────────────


def open_turn(agent: Any) -> Turn | None:
    """A turn starts on *agent* (its origin already set): record it and make
    it the current turn. None when recording is off, or inside a turn
    already (that one is the turn). Never raises."""
    try:
        if current() is not None or not enabled():
            return None
        turn = Turn(turn_id=secrets.token_hex(16), started_at=time.time())
        if not _enqueue(_open_record(agent, turn)):
            return None
        turn._tokens = (current_turn_id.set(turn.turn_id), _CURRENT.set(turn))
        return turn
    except Exception as exc:
        log.warning("Turn not recorded", error=str(exc))
        return None


def outcome_of(agent: Any, reply: Any) -> str:
    """How a turn that returned *reply* ended: ``ok``, or — when the agent
    marked it failed — ``cancelled`` (the user pressed Stop: the tool loop
    saw ``cancel_event`` and returned), ``blocked`` (a guard refused it) or
    ``error``."""
    try:
        if getattr(agent, "_last_complete_success", True) is not False:
            return "ok"
        if getattr(agent, "_last_complete_cancelled", False) is True:
            return "cancelled"
        return "blocked" if _GUARD_BLOCK_RE.match(str(reply or "").lstrip()) else "error"
    except Exception:
        return "ok"


def close_turn(turn: Turn | None, outcome: str, reply: Any = "", *, agent: Any = None) -> None:
    """Record how *turn* ended and end it as the current turn. Never raises;
    a second close is ignored. In an event loop the files the turn saved are
    listed on a worker thread — a long-lived session's folders are never
    walked on the loop — and the close is spooled once they are
    (:func:`wait_closes`)."""
    if turn is None or turn.closed:
        return
    turn.closed = True
    try:
        record = {
            "phase": "close", "turn_id": turn.turn_id, "at": time.time(),
            "outcome": outcome if outcome in OUTCOMES else "error",
            "reply_head": _head(reply),
            "tool_calls": int(turn.tool_calls), "failed_tool_calls": int(turn.failed_tool_calls),
        }
        base, slug = _saved_folder(agent) if agent is not None else (None, "")
        pending = None
        if base is not None and slug:
            try:
                pending = asyncio.get_running_loop().run_in_executor(
                    _closer(), _spool_close, record, base, slug, turn.started_at)
            except RuntimeError:
                pending = None   # no loop running (or it is shutting down): list them here
        if pending is not None:
            _CLOSING.add(pending)
            pending.add_done_callback(_closed)
        else:
            _spool_close(record, base, slug, turn.started_at)
    except Exception as exc:
        log.warning("Turn end not recorded", error=str(exc))
    finally:
        tokens, turn._tokens = turn._tokens, None
        if tokens is not None:
            for var, token in zip((current_turn_id, _CURRENT), tokens, strict=True):
                try:
                    var.reset(token)
                except (ValueError, RuntimeError):
                    pass   # closed from another context (a stream closed late): it is marked closed


# Closes whose saved files are being listed on the worker thread (one, so
# they are spooled in the order the turns ended).
_CLOSING: set[asyncio.Future] = set()
_CLOSER: ThreadPoolExecutor | None = None
_CLOSER_LOCK = threading.Lock()


def _closer() -> ThreadPoolExecutor:
    global _CLOSER
    with _CLOSER_LOCK:
        if _CLOSER is None:
            _CLOSER = ThreadPoolExecutor(max_workers=1, thread_name_prefix="turn-register")
        return _CLOSER


def _spool_close(record: dict[str, Any], base: Path | None, slug: str, since: float) -> None:
    try:
        _enqueue({**record, "deliverables": _deliverables(base, slug, since)})
    except Exception as exc:
        log.warning("Turn end not recorded", error=str(exc))


def _closed(pending: asyncio.Future) -> None:
    """Back on the loop once a close is spooled: ship it now (the worker
    thread can't start the shipper)."""
    _CLOSING.discard(pending)
    try:
        _ensure_shipper()
    except Exception:
        pass


async def wait_closes() -> None:
    """Wait for the closes still listing their files (tests, shutdown)."""
    if _CLOSING:
        await asyncio.gather(*list(_CLOSING), return_exceptions=True)


class ReplyHead:
    """The start of a streamed reply, kept as it is yielded (enough for
    ``reply_head``)."""

    def __init__(self) -> None:
        self._parts: list[str] = []
        self._len = 0

    def add(self, chunk: Any) -> None:
        if self._len < _HEAD_MAX * 2 and isinstance(chunk, str):
            self._parts.append(chunk)
            self._len += len(chunk)

    @property
    def text(self) -> str:
        return "".join(self._parts)


# ── Shipping ──────────────────────────────────────────────────────────


def _ensure_shipper() -> None:
    tool_audit.ensure_shipper(_SHIPPERS, _ship_loop)


async def _post(client: Any, records: list[dict[str, Any]]) -> int:
    """One boot's records as Flight Deck reads them: ``{boot, records}``."""
    boot = str(records[0].get("boot") or tool_audit._BOOT) if records else tool_audit._BOOT
    body = {"boot": boot, "records": [{k: v for k, v in r.items() if k != "boot"} for r in records]}
    resp = await client.post(INGEST_PATH, json=body)
    return int(resp.status_code)


async def _ship_loop() -> None:
    await tool_audit.ship_loop("Turn register", _spool, lambda client, records: _post(client, records))


async def flush_now() -> None:
    """Ship whatever is ready (tests, shutdown)."""
    await wait_closes()
    await tool_audit.flush_spool(_spool(), _post)
