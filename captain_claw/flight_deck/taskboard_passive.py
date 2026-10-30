"""Taskboard passive inboxes — the user's own Telegram inbox, WhatsApp chats
and Gmail on their board, each only once they switch it on (v2, decision #1).

  GET  /fd/taskboard/settings   ``passive``: each inbox ``{on, consented_at, available, note}``
  PUT  /fd/taskboard/settings   ``{"passive": {"<inbox>": {"on": true, "consent": true}}}`` switches
                                one on (no consent: 400); ``{"on": false}`` switches it off and
                                deletes what it put on the board (``deleted``)
  GET  /fd/taskboard/status     the Sources sheet marks an inbox tracked while it is on

(the routes are ``taskboard``'s; what they do with the inboxes is here.)

Three inboxes, off until the board user switches one on with their consent
(the UI shows the consent text; the server wants ``consent: true`` and keeps
when it was given): ``telegram_inbox`` (the Telegram inbox Flight Deck keeps
for them, telegram_inbox.py), ``whatsapp_web`` (their linked WhatsApp,
whatsapp_web/store.py) and ``gmail`` (their Google account's mail, read with
their own token — never anyone else's, never through an agent). An inbox
the user hasn't connected is ``available: false``, with a note saying what
to connect; it can't be switched on.

The sweep (``taskboard_sources.ADAPTERS``) reads each switched-on user's own
inbox — and only theirs — from a cursor of its own, never waking an agent:

* a message someone else sent passes a deterministic gate first (no model):
  a group only when it replies to the user or mentions them, never a status
  broadcast or a channel, an email only when ``gmail_inbound.prefilter``
  passes it (no newsletters, no automated or no-reply mail, nothing spoofed
  or sent to a crowd), and the user's own ``task_rules`` (not a task, mute,
  always a task) as they apply to turns. It becomes a task of kind
  ``message`` (direction received, channel ``tg_inbox`` / ``waweb`` /
  ``gmail``) waiting as a *maybe* — Jev's ``is_task`` says whether it is one,
  so a chat's every "ok" never lands in Needs me — flagged
  ``contains_instructions`` until Jev says it asks nothing of the assistant
  (fails closed), queued for triage like a turn (the same caps);
* the user's own message is a ``sent`` event on the conversation's task —
  the next move theirs to wait for — or, with none, a ``sent`` task of its
  own, done (recorded evidence). What an agent sent from the user's account
  is its turn's (the tool call has it), never the user's own words;
* a conversation is one task while it goes on: a chat's messages within 12
  hours of each other, an email thread's while its task is open; after
  that, the next message is a task of its own. An email thread the board
  already knows from another source (an agent mailed it, an auto-reply held
  it) gets the message on that task.

Text is cut (the ask ≤ 600, an event ≤ 500) and kept no longer than the
inbox keeps it: every task and event made from an inbox carries
``retention_until`` (its message's time + 30 days — fewer when the user keeps
that inbox fewer), and the sweep takes the expired ones off (a task with
nothing else on it goes; one with more keeps the rest). Everything an inbox put on the board is known by its source — its
events, the tasks it made, the links and parties it gave a task
(``passive_marks``) — so switching it off deletes exactly that, and an inbox
that was itself switched off or unlinked takes its tasks with it.
"""

from __future__ import annotations

import asyncio
import json
import re
import sqlite3
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import HTTPException

from captain_claw import gmail_inbound
from captain_claw.flight_deck import taskboard_sources, taskboard_store
from captain_claw.flight_deck.taskboard_sources import _addresses, _json, _Pass, _rows
from captain_claw.flight_deck.taskboard_store import TaskboardStore
from captain_claw.logging import get_logger

log = get_logger(__name__)

# The inboxes by the names the settings use, and by what the board calls
# what they make: the ``source`` of their tasks and events (the sweep
# adapter's name too), the channel, the Sources sheet's row.
KEYS = ("telegram_inbox", "whatsapp_web", "gmail")
SOURCE = {"telegram_inbox": "telegram_inbox", "whatsapp_web": "whatsapp_web", "gmail": "gmail_inbox"}
CHANNEL = {"telegram_inbox": "tg_inbox", "whatsapp_web": "waweb", "gmail": "gmail"}
SHEET = {"telegram_inbox": "telegram_inbox", "whatsapp_web": "whatsapp_web", "gmail": "gmail_inbox"}
LABELS = {"telegram_inbox": "Your Telegram inbox", "whatsapp_web": "Your WhatsApp chats", "gmail": "Your Gmail inbox"}
# What each reads, once it is on.
READS = {
    "telegram_inbox": "Messages people send you in the chats your Telegram inbox keeps, and your replies — a group "
                      "only when someone replies to you or mentions you. Their text is kept 30 days.",
    "whatsapp_web": "Messages people send you on your WhatsApp, and your replies — a group only when someone "
                    "replies to you or mentions you. Their text is kept 30 days.",
    "gmail": "Emails people send you in your Gmail inbox — no newsletters or automated mail — and the replies you "
             "send. Their text is kept 30 days.",
}
_OFF_NOTE = "Off — read only once you switch it on."

KEPT_S = 30 * 86400          # an inbox's text on the board: as long as the inbox keeps it
FIRST_LOOK_S = 86400         # switching one on reads back the last day (as an agent's first look does)
CONVERSATION_GAP_S = 12 * 3600  # a chat's messages this close are one conversation, one task
_PASS_MAX = 200              # messages one inbox gives one board per pass
_WA_OWN_WAIT_S = 120.0       # the user's own WhatsApp message waits this long (an agent's send is logged by then)
_GMAIL_MAX = 50              # emails read per pass — the oldest not read yet first; the rest next pass
_GMAIL_LIST_MAX = 500        # … of the newest this many listed (a page of 100 at a time)
_GMAIL_PAGE = 100
_GMAIL_OWN_WAIT_S = 600.0    # … the user's own email this long (an agent's send reaches its records)
_GMAIL_OVERLAP_S = 60        # the next list reaches back this far (what was read is known)
_GMAIL_DONE_KEPT = 2000
_OWN_ADDRESSES_S = 86400.0   # the mailbox's own addresses, read again daily
_LIST_TIMEOUT_S = 60.0
_FETCH_TIMEOUT_S = 240.0
# Prefilter codes that say an email isn't someone writing to the user now —
# no rule makes one a task. The rest (automated, a list, a crowd…) an
# "always a task" rule overrides, as it does a turn's Gmail verdict.
_GMAIL_NEVER = ("labels", "before_enable", "stale", "from_invalid", "own_mail", "empty", gmail_inbound.CHECK_ERROR)
_GMAIL_PASSES = ("", "too_long")  # a long email is still a task
_WA_MEDIA = {"image": "photo", "video": "video", "voice": "voice message", "audio": "audio", "document": "file",
             "location": "location", "contact": "contact", "poll": "poll", "event": "event"}
_WA_NOT_CHATS = ("@broadcast", "@newsletter")  # status updates, broadcast lists, channels
_WA_NEVER_KINDS = ("sticker",)
_WA_USER_ID = re.compile(r"^[A-Za-z0-9_-][A-Za-z0-9_.-]{0,79}$")  # whatsapp_web.engine._USER_ID

# The deck's event loop, for the Gmail reads (async, the user's own token)
# a sweep thread makes: remembered by this module's routes, else triage's.
_LOOP: asyncio.AbstractEventLoop | None = None


def remember_loop() -> None:
    global _LOOP
    try:
        _LOOP = asyncio.get_running_loop()
    except RuntimeError:
        pass


def _main_loop() -> asyncio.AbstractEventLoop | None:
    from captain_claw.flight_deck import taskboard_triage

    for loop in (_LOOP, taskboard_triage._loop):
        if loop is not None and not loop.is_closed() and loop.is_running():
            return loop
    return None


def _on_loop(coro: Any, timeout: float) -> Any:
    """Run *coro* on the deck's loop from a sweep thread and wait for it —
    None when there is no loop to run it on, or this thread is a loop's own
    (it would wait on itself)."""
    loop = _main_loop()
    try:
        asyncio.get_running_loop()
        on_a_loop = True
    except RuntimeError:
        on_a_loop = False
    if loop is None or on_a_loop:
        coro.close()
        return None
    fut = asyncio.run_coroutine_threadsafe(coro, loop)
    try:
        return fut.result(timeout)
    except TimeoutError:
        fut.cancel()
        raise


# ── The switches ──────────────────────────────────────────────────────


def _store() -> TaskboardStore:
    return taskboard_store.get_store()


def switches(store: TaskboardStore, uid: str) -> dict[str, dict]:
    """``{inbox: {on, consented_at}}`` as the user left them (off by default)."""
    raw = store.get_settings(uid).get("passive") or {}
    out: dict[str, dict] = {}
    for key in KEYS:
        s = raw.get(key) if isinstance(raw.get(key), dict) else {}
        at = s.get("consented_at")
        out[key] = {"on": s.get("on") is True,
                    "consented_at": float(at) if isinstance(at, (int, float)) and not isinstance(at, bool) else None}
    return out


def _set(store: TaskboardStore, uid: str, key: str, *, on: bool, at: float | None = None) -> None:
    now = switches(store, uid)
    now[key] = {"on": on, "consented_at": at if on else None}
    store.save_settings(uid, passive=now)


def _cursor_name(key: str, uid: str) -> str:
    return f"passive:{key}:{uid}"


def _cursor(store: TaskboardStore, key: str, uid: str) -> dict:
    """Where the user's read of *key* stands; ``since`` is how far back it
    reads (the last day before they switched it on)."""
    cur = _json(store.get_cursor(_cursor_name(key, uid)), {})
    if not isinstance(cur.get("since"), (int, float)):
        cur["since"] = (switches(store, uid)[key]["consented_at"] or time.time()) - FIRST_LOOK_S
    return cur


def _save_cursor(store: TaskboardStore, key: str, uid: str, cur: dict) -> None:
    store.set_cursor(_cursor_name(key, uid), json.dumps(cur))


def switch_on(uid: str, key: str, store: TaskboardStore | None = None) -> None:
    """The user switched *key* on, consenting now (already on: as it was)."""
    store = store or _store()
    if switches(store, uid)[key]["on"]:
        return
    at = time.time()
    _set(store, uid, key, on=True, at=at)
    _save_cursor(store, key, uid, {"since": at - FIRST_LOOK_S})


def forget(uid: str, key: str, *, store: TaskboardStore | None = None, locked: bool = True) -> int:
    """*key* off for *uid* and everything it put on their board deleted —
    after the sweep reading now, if one is (*locked*; a sweep that finds the
    inbox gone holds the lock already). How many tasks went."""
    store = store or _store()
    _set(store, uid, key, on=False)
    left: set[tuple[str, str]] = set()
    if locked:
        with taskboard_sources._SWEEP_LOCK:
            gone = store.passive_forget(uid, SOURCE[key], left)
    else:
        gone = store.passive_forget(uid, SOURCE[key], left)
    store.set_cursor(_cursor_name(key, uid), "")
    next_steps_left(left)
    return gone


def next_steps_left(left: set[tuple[str, str]]) -> None:
    """The next steps on the tasks an inbox's text just left, and the bell's copy
    of them (taskboard_nextsteps.text_left). The inbox has gone already: this
    failing is logged, never the switch-off's answer (what still waits then
    expires with whom it goes to gone from the task, on the next read)."""
    from captain_claw.flight_deck import owner_notify, taskboard_nextsteps

    if not left:
        return
    try:
        told = taskboard_nextsteps.text_left(left)
    except Exception as exc:  # noqa: BLE001
        log.warning("Taskboard next steps not told plain", error=str(exc)[:200])
        return
    loop = _main_loop()
    for uid, action_ids in told.items():
        if loop is not None:
            asyncio.run_coroutine_threadsafe(owner_notify.bells_told_plain(uid, action_ids), loop)


async def apply(uid: str, changes: Any) -> int:
    """The ``passive`` part of a settings PUT: ``{inbox: {on, consent}}``.
    Every switch is checked before any is moved — an unknown inbox, no
    ``on``, switching on without ``consent: true`` or an inbox the user
    hasn't connected is a 400. How many tasks switching off deleted."""
    remember_loop()
    if not isinstance(changes, dict) or not changes:
        raise HTTPException(400, f"passive names an inbox: {', '.join(KEYS)}")
    plan: list[tuple[str, bool]] = []
    for key, want in changes.items():
        if key not in KEYS:
            raise HTTPException(400, f"passive is one of {', '.join(KEYS)}")
        if not isinstance(want, dict) or not isinstance(want.get("on"), bool):
            raise HTTPException(400, f"passive.{key} needs on: true or false")
        if want["on"] and want.get("consent") is not True:
            raise HTTPException(400, f"Switching on {LABELS[key][0].lower()}{LABELS[key][1:]} needs your consent")
        plan.append((key, want["on"]))
    ready = await availability(uid) if any(on for _, on in plan) else {}
    for key, on in plan:
        if on and not ready[key][0]:
            raise HTTPException(400, ready[key][1])
    deleted = 0
    for key, on in plan:
        if on:
            await asyncio.to_thread(switch_on, uid, key)
        else:
            deleted += await asyncio.to_thread(forget, uid, key)
    return deleted


async def settings_view(uid: str) -> dict[str, dict]:
    """``{inbox: {on, consented_at, available, note}}`` — the settings'
    ``passive``."""
    remember_loop()
    now = await asyncio.to_thread(switches, _store(), uid)
    ready = await availability(uid)
    return {k: {**now[k], "available": ready[k][0], "note": ready[k][1]} for k in KEYS}


def sources(uid: str, base: list[dict]) -> list[dict]:
    """The Sources sheet's rows with the inboxes as the user has them: tracked
    while on."""
    now = switches(_store(), uid)
    by_row = {SHEET[k]: k for k in KEYS}
    out = []
    for row in base:
        key = by_row.get(row.get("key"))
        if key is None:
            out.append(row)
            continue
        on = now[key]["on"]
        out.append({**row, "label": LABELS[key], "tracked": on, "on": on, "note": READS[key] if on else _OFF_NOTE})
    return out


async def sources_view(uid: str, base: list[dict]) -> list[dict]:
    remember_loop()
    return await asyncio.to_thread(sources, uid, base)


# ── Is the inbox there? ───────────────────────────────────────────────


async def availability(uid: str) -> dict[str, tuple[bool, str]]:
    """``{inbox: (available, note)}`` — available once the user connected
    that channel (the note says what to connect, else what it reads)."""
    out: dict[str, tuple[bool, str]] = {}
    for key, check in (("telegram_inbox", _telegram_ready), ("whatsapp_web", _whatsapp_ready),
                       ("gmail", _gmail_ready)):
        try:
            out[key] = await check(uid)
        except Exception as exc:  # noqa: BLE001 — unknown is not connected
            log.debug("Taskboard inbox check failed", source=key, error=type(exc).__name__)
            out[key] = (False, "Couldn't check this connection just now.")
    return out


async def _telegram_ready(uid: str) -> tuple[bool, str]:
    from captain_claw.flight_deck import fd_crypto, telegram_inbox
    from captain_claw.flight_deck.auth import get_db

    if not telegram_inbox.allowed():
        return False, "This deck doesn't offer the Telegram inbox."
    if not fd_crypto.available():
        return False, "This deck can't read the Telegram inbox (its encryption isn't installed)."
    if not await telegram_inbox.collecting(get_db(), uid):
        return False, "Turn your Telegram inbox on first: Connections → Telegram → Inbox."
    return True, READS["telegram_inbox"]


def _wa_dir(uid: str) -> Path | None:
    from captain_claw.flight_deck.whatsapp_web import DIR_NAME

    return taskboard_store.data_dir() / DIR_NAME / uid if _WA_USER_ID.match(uid or "") else None


def _wa_linked(uid: str) -> bool:
    """The user's WhatsApp is linked and its messages kept."""
    from captain_claw.flight_deck.whatsapp_web.link_state import LinkState

    d = _wa_dir(uid)
    if d is None or not (d / "store.sqlite3").is_file():
        return False
    st = LinkState.load(d / "state.json")
    return st is not None and bool(st.linked_at) and st.state not in ("logged_out", "expired", "failed")


async def _whatsapp_ready(uid: str) -> tuple[bool, str]:
    if not await asyncio.to_thread(_wa_linked, uid):
        return False, "Link your WhatsApp first: Connections → WhatsApp Web."
    return True, READS["whatsapp_web"]


async def _gmail_ready(uid: str) -> tuple[bool, str]:
    from captain_claw.flight_deck import gmail_autoreply
    from captain_claw.flight_deck import google_oauth_routes as google
    from captain_claw.flight_deck.auth import get_db

    if not await google.is_google_connected(uid):
        return False, "Connect your Google account first: Connections → Google."
    if "gmail.readonly" in await gmail_autoreply.missing_scopes(get_db(), google._tenant(uid)):
        return False, "Your Google connection can't read mail — connect Google again and allow Gmail."
    return True, READS["gmail"]


# ── One message, through the gate, onto the board ─────────────────────


@dataclass
class _Msg:
    """One inbox message as the board takes it."""

    ref: str                    # its events' source_ref: the board user's, the conversation's, its own id
    conv: str                   # the conversation's key: tg_inbox:<chat>, waweb:<jid>, gmail:thread:<id>
    at: float
    outgoing: bool              # the user's own
    title: str
    text: str                   # ≤ 600, the task's ask
    summary: str                # ≤ 500, its event's
    party: dict | None          # the other side: who wrote it, or who the user wrote to
    others: list[dict] = field(default_factory=list)  # also on an email (To, Cc)
    group: bool = False
    blocked: str = ""           # what the gate keeps it off for ("" passes; "always a task" overrides it)
    refs: dict = field(default_factory=dict)
    extra_keys: list[str] = field(default_factory=list)
    kept: float = KEPT_S        # how long its text is kept: 30 days, less when the inbox keeps it less

    @property
    def until(self) -> float:
        return self.at + self.kept


def _texts(text: str, media: str = "", subject: str = "") -> dict[str, str]:
    """A message's title, ask and event summary — cut, and nothing more."""
    words = " ".join(str(text or "").split())
    shown = f"{words} ({media})" if words and media else words or (f"({media})" if media else "")
    first = next((line.strip() for line in str(text or "").splitlines() if line.strip()), "")
    title = " ".join((subject or first).split())[:160] or media[:1].upper() + media[1:]
    summary = f"{' '.join(subject.split())}: {shown}" if subject and shown else subject or shown
    return {"title": title, "text": shown[:600], "summary": summary[:500]}


def _me(uid: str) -> str:
    """The board user as an event names them."""
    if not taskboard_sources._auth_enabled():
        return "You"
    try:
        rows = _rows(taskboard_sources._fd_db_path(), "SELECT display_name, email FROM users WHERE id = ?", (uid,))
    except sqlite3.Error:
        rows = []
    return str(rows[0]["display_name"] or rows[0]["email"] or "You") if rows else "You"


def _effect(store: TaskboardStore, uid: str, key: str, m: _Msg) -> str:
    """The user's task rule for this message (``taskboard_ingest._rule``, by
    the person on the other side, the channel and the inbox): its effect, or
    ""."""
    from captain_claw.flight_deck import taskboard_ingest

    turn = {"user_id": uid, "agent_slug": "", "channel": CHANNEL[key], "source": SOURCE[key]}
    rule = taskboard_ingest._rule(store, turn, m.party or {})
    if rule is None:
        return ""
    store.rule_hit(uid, rule["id"])
    return str(rule["effect"])


def _recent(task: dict, at: float) -> bool:
    return float(task.get("last_event_at") or task.get("created_at") or 0) >= at - CONVERSATION_GAP_S


def _live(task: dict, key: str, at: float) -> bool:
    """Whether a task holding a conversation takes someone's next message:
    still undecided or open (or judged not a task, which the new message may
    change) — a chat's only while it goes on, whichever task its key is on (a
    merge moves it: a later message, about anything, is no part of that task)."""
    return task["status"] in ("open", "maybe", "not_task") and (key == "gmail" or _recent(task, at))


def _requeue(task: dict, now: float) -> dict | None:
    """The patch that puts a task back in triage's queue for a new message:
    an inbox's (undecided, open or judged not a task), or a person's request
    still open — never one a mute rule keeps quiet, nor one whose status the
    user set (their word stands; there is nothing to ask)."""
    if _json(task.get("flags"), {}).get("rule") == "mute" or "status" in _json(task.get("user_locks"), {}):
        return None
    if task["kind"] == "message" and task["status"] in ("open", "maybe", "not_task"):
        return {"triage_state": "pending", "triage_after": now}
    if task["kind"] == "request" and task["status"] in ("open", "maybe") and task["triage_state"] != "skipped":
        return {"triage_state": "pending", "triage_after": now}
    return None


def _event(key: str, m: _Msg, me: str) -> dict:
    ev = {"source": SOURCE[key], "source_ref": m.ref, "at": m.at, "channel": CHANNEL[key], "summary": m.summary,
          "private": True, "refs": {**m.refs, "retention_until": m.until}}
    if m.outgoing:
        return {**ev, "kind": "sent", "actor_kind": "me", "actor": me, "outcome": "ok", "evidence": "recorded"}
    return {**ev, "kind": "received", "actor_kind": "party", "actor": (m.party or {}).get("display") or ""}


def _message_task(key: str, m: _Msg, effect: str, now: float) -> dict:
    """Someone's message as a task: a maybe until Jev's ``is_task`` decides
    (a rule's "always a task" admits it open; a mute keeps it quiet, nobody's
    move, never triaged), the user's to answer, and taken to ask the
    assistant to act until Jev says otherwise (fails closed)."""
    muted = effect == "mute"
    return {
        "title": m.title or f"Message on {taskboard_store.CHANNEL_LABELS.get(CHANNEL[key], key)}", "ask": m.text,
        "kind": "message", "direction": "received", "channel": CHANNEL[key],
        "status": "open" if effect in ("mute", "always_task") else "maybe", "owes_next": "nobody" if muted else "me",
        "triage_state": "skipped" if muted else "pending", "triage_after": now, "private": 1,
        "source": SOURCE[key], "source_ref": m.ref, "retention_until": m.until,
        "flags": {"contains_instructions": True, **({"rule": effect} if effect else {})},
    }


def _sent_task(key: str, m: _Msg, me: str) -> tuple[dict, dict]:
    """The user's own message with no conversation on the board: a ``sent``
    task, done — Gmail or the chat recorded it."""
    to = (m.party or {}).get("display") or "someone"
    create = {"title": m.title or f"Message to {to}", "ask": m.text, "kind": "sent", "direction": "sent",
              "channel": CHANNEL[key], "owes_next": "nobody", "triage_state": "skipped", "private": 1,
              "source": SOURCE[key], "source_ref": m.ref, "retention_until": m.until}
    patch = {"status": "done", "owes_next": "nobody", "closed_by": "evidence", "closed_by_party": me,
             "evidence": "verified", "result": "success", "result_text": f"You wrote to {to}"}
    return create, patch


def _put(p: _Pass, uid: str, key: str, m: _Msg, *, event: dict, create: dict | None = None,
         patch: dict | None = None, parties: list[dict] | None = None, release: bool = False) -> tuple[str | None, bool]:
    """Write one message the way every sweep adapter writes (``_Pass.ingest``),
    marked as the inbox's."""
    source, keys = SOURCE[key], [m.conv]

    def write() -> tuple[str | None, bool]:
        return p.ingest(source, uid, keys=keys, event=event, create=create, patch=patch,
                        parties=[x for x in parties or [] if x], extra_keys=m.extra_keys)

    return p.store.passive_write(uid, source, keys, write, release=release, retention_until=m.until)


def _not_on(store: TaskboardStore, uid: str, task_id: str, parties: list[dict]) -> list[dict]:
    """*parties* the task doesn't have yet (by kind and address)."""
    have = {(x["kind"], x["address"]) for x in store.parties_for(uid, [task_id]).get(task_id, [])}
    return [x for x in parties if x and (x.get("kind"), x.get("address")) not in have]


def _take(p: _Pass, uid: str, key: str, msgs: list[_Msg]) -> None:
    """The messages one inbox gave *uid*'s board this pass, oldest first,
    each once: past the gate and the user's rules, onto its conversation's
    task — or one of its own."""
    from captain_claw.flight_deck import taskboard_ingest

    store, source = p.store, SOURCE[key]
    seen = store.passive_seen(source, [m.ref for m in msgs])
    me, now = _me(uid), time.time()
    queued = False
    for m in sorted(msgs, key=lambda m: m.at):
        if m.ref in seen or m.until <= now:
            continue  # on the board already — or its text is past its 30 days
        seen.add(m.ref)
        effect = _effect(store, uid, key, m)
        if effect == "not_task" or (m.blocked and effect != "always_task"):
            continue
        held = store.task_for_key(uid, m.conv)
        task = store.get_task(uid, held) if held else None
        ours = task is not None and task["source"] == source
        everyone = [m.party, *m.others]
        if m.outgoing and task is not None and (key == "gmail" or _recent(task, m.at)):
            # The user answered in the conversation: the next move is theirs to wait for.
            patch = _requeue(task, now)
            if ours and task["kind"] == "message" and task["status"] == "open" and task["owes_next"] == "me":
                patch = {**(patch or {}), "owes_next": "them"}
            _, new = _put(p, uid, key, m, event=_event(key, m, me), patch=patch,
                          parties=_not_on(store, uid, task["id"], everyone))
            queued |= bool(new and patch and "triage_state" in patch)
        elif m.outgoing:
            if m.group:
                continue  # the user's own word in a group the board follows no conversation of
            create, patch = _sent_task(key, m, me)
            _put(p, uid, key, m, event=_event(key, m, me), create=create, patch=patch, parties=everyone,
                 release=task is not None)
        elif task is not None and ((not ours and key == "gmail") or _live(task, key, m.at)):
            patch = _requeue(task, now)
            _, new = _put(p, uid, key, m, event=_event(key, m, me), patch=patch,
                          parties=_not_on(store, uid, task["id"], everyone))
            queued |= bool(new and patch)
        else:
            _, new = _put(p, uid, key, m, event=_event(key, m, me), create=_message_task(key, m, effect, now),
                          parties=everyone, release=task is not None)
            queued |= new and effect != "mute"
    if queued:
        taskboard_ingest._kick(uid)


# ── The sweep (taskboard_sources.ADAPTERS) ────────────────────────────


def sweep(p: _Pass, key: str) -> None:
    """One sweep's read of the inbox *key*: what passed its 30 days off every
    board (or the one user's), then each switched-on user's new messages —
    only theirs, only while it is on. A board whose read fails is logged
    and the rest go on; the sweep is told."""
    store, now = p.store, time.time()
    left: set[tuple[str, str]] = set()
    gone = store.passive_purge(SOURCE[key], now=now, kept_s=KEPT_S, user_id=p.user_id, left=left)
    next_steps_left(left)
    if gone:
        log.info("Taskboard inbox text expired", source=key, tasks=gone)
    users = [u for u in store.passive_users(key) if (p.user_id is None or u == p.user_id) and p.board_user(u) == u]
    failed = 0
    for uid in users:
        try:
            _READERS[key](p, uid)
        except sqlite3.OperationalError as exc:
            if "no such table" in str(exc):
                raise  # the inbox isn't on this deck: the sweep says so and goes on
            failed += 1
            log.warning("Taskboard inbox read failed", source=key, user=uid, error=str(exc))
        except Exception as exc:  # noqa: BLE001 — one board never stops the others
            failed += 1
            log.warning("Taskboard inbox read failed", source=key, user=uid, error=str(exc)[:200])
    if failed:
        raise RuntimeError(f"{LABELS[key]}: {failed} board(s) couldn't be read")


def _gone(store: TaskboardStore, uid: str, key: str) -> None:
    """The inbox itself is gone (switched off or unlinked where it lives,
    which deleted what it kept): so is what the board made of it."""
    n = forget(uid, key, store=store, locked=False)
    log.info("Taskboard inbox gone at its source", source=key, user=uid, tasks=n)


def _unsealed(token: Any) -> dict:
    from captain_claw.flight_deck import fd_crypto

    value = fd_crypto.unseal(str(token)) if token else None
    return value if isinstance(value, dict) else {}


def _unsealed_text(token: Any) -> str:
    from captain_claw.flight_deck import fd_crypto

    value = fd_crypto.unseal(str(token)) if token else None
    return value if isinstance(value, str) else ""


# ── Telegram inbox (flight-deck.db: telegram_inbox_*) ─────────────────


def _telegram(p: _Pass, uid: str) -> None:
    key, path = "telegram_inbox", taskboard_sources._fd_db_path()
    on = _rows(path, "SELECT enabled, retention_days FROM telegram_inbox_settings WHERE user_id = ?", (uid,))
    if not on or not on[0]["enabled"]:
        _gone(p.store, uid, key)  # switching the inbox off deleted what it kept
        return
    kept = min(KEPT_S, max(1, int(on[0]["retention_days"] or 30)) * 86400.0)
    cur = _cursor(p.store, key, uid)
    after = int(cur.get("id") or 0)
    rows = _rows(path, "SELECT m.*, c.kind AS chat_kind, c.title AS chat_title FROM telegram_inbox_messages m "
                       "LEFT JOIN telegram_inbox_chats c ON c.user_id = m.user_id AND c.chat_id = m.chat_id "
                       "WHERE m.user_id = ? AND m.id > ? AND m.sent_at >= ? ORDER BY m.id LIMIT ?",
                 (uid, after, int(cur["since"]), _PASS_MAX))
    if not rows:
        return
    link = _rows(path, "SELECT tg_username FROM telegram_links WHERE user_id = ? AND status = 'approved'", (uid,))
    handle = f"@{link[0]['tg_username']}".lower() if link and link[0]["tg_username"] else ""
    msgs = []
    for r in rows:
        if r["deleted"]:
            continue
        body = _unsealed(r["body"])
        text, media = str(body.get("t") or ""), str(body.get("m") or "")
        if not text.strip() and not media:
            continue
        chat_id, outgoing = str(r["chat_id"]), bool(r["outgoing"])
        group = r["chat_kind"] == "group"
        chat = _unsealed_text(r["chat_title"]) or chat_id
        if outgoing:
            party = None if group else {"role": "recipient", "kind": "tg_id", "address": chat_id, "display": chat,
                                        "verified": True}
        else:
            sender = str(r["sender_id"] or "")
            party = {"role": "requester", "kind": "tg_id", "address": sender or chat_id,
                     "display": str(body.get("s") or "") or chat, "verified": bool(sender)}
        addressed = (not group or outgoing or bool(handle and handle in text.lower())
                     or _tg_replies_to_me(path, uid, chat_id, r["reply_to"]))
        msgs.append(_Msg(ref=f"{uid}:{chat_id}:{r['message_id']}", conv=f"tg_inbox:{chat_id}", at=float(r["sent_at"]),
                         outgoing=outgoing, party=party, group=group, blocked="" if addressed else "group",
                         refs={"chat_id": chat_id, "message_id": int(r["message_id"])}, kept=kept,
                         **_texts(text, media)))
    _take(p, uid, key, msgs)
    _save_cursor(p.store, key, uid, {**cur, "id": int(rows[-1]["id"])})


def _tg_replies_to_me(path: Path, uid: str, chat_id: str, reply_to: Any) -> bool:
    if not reply_to:
        return False
    return bool(_rows(path, "SELECT 1 FROM telegram_inbox_messages WHERE user_id = ? AND chat_id = ? "
                            "AND message_id = ? AND outgoing = 1", (uid, chat_id, int(reply_to))))


# ── WhatsApp (<data>/whatsapp_web/<user>/store.sqlite3) ───────────────


def _whatsapp(p: _Pass, uid: str) -> None:
    from captain_claw.flight_deck.whatsapp_web.link_state import LinkState, bare_jid, number_of

    key, d = "whatsapp_web", _wa_dir(uid)
    path = d / "store.sqlite3" if d is not None else None
    if path is None or not path.is_file():
        _gone(p.store, uid, key)  # unlinked, or logged out on the phone: the messages it kept are gone
        return
    st = LinkState.load(d / "state.json")
    me = bare_jid(st.jid) if st and st.jid else ""
    cur = _cursor(p.store, key, uid)
    after = int(cur.get("rowid") or 0)
    rows = _rows(path, "SELECT * FROM messages WHERE rowid > ? AND ts >= ? ORDER BY rowid LIMIT ?",
                 (after, int(cur["since"]), _PASS_MAX))
    if not rows:
        return
    lids = {str(r["lid"]): str(r["pn"]) for r in _rows(path, "SELECT lid, pn FROM lid_map")}
    days = _rows(path, "SELECT value FROM meta WHERE key = 'retention_days'")
    kept = min(KEPT_S, int(days[0]["value"]) * 86400.0) if days and str(days[0]["value"]).isdigit() else KEPT_S

    def canon(jid: Any) -> str:
        jid = str(jid or "")
        return lids.get(jid, jid)

    mine_ids = {me.partition("@")[0], *(lid.partition("@")[0] for lid, pn in lids.items() if pn == me)} - {""}
    jids = sorted(({canon(r["chat_jid"]) for r in rows} | {canon(r["sender_jid"]) for r in rows}) - {""}) or [""]
    marks = ",".join("?" * len(jids))
    names = {r["jid"]: r for r in _rows(path, f"SELECT * FROM contacts WHERE jid IN ({marks})", jids)}
    chats = {r["jid"]: r for r in _rows(path, f"SELECT jid, name, is_group FROM chats WHERE jid IN ({marks})", jids)}
    own = [str(r["id"]) for r in rows if r["from_me"]]
    agents = {str(r["wa_id"]) for r in _rows(
        path, f"SELECT wa_id FROM sends WHERE wa_id IN ({','.join('?' * len(own))})", own)} if own else set()

    def person(jid: str, push: str = "") -> str:
        n = names.get(jid) or {}
        return (_unsealed_text(n.get("name")) or push or _unsealed_text(n.get("push_name"))
                or _unsealed_text(n.get("business_name")) or number_of(jid) or jid)

    now = time.time()
    msgs, last = [], after
    for r in rows:
        mid, from_me = str(r["id"]), bool(r["from_me"])
        if from_me and mid not in agents and now - float(r["ts"] or 0) < _WA_OWN_WAIT_S:
            break  # the user's — or an agent's whose send isn't logged yet: read once it settles
        last = int(r["rowid"])
        chat = canon(r["chat_jid"])
        if chat.endswith(_WA_NOT_CHATS) or r["deleted"] or r["kind"] in _WA_NEVER_KINDS or not r["ts"]:
            continue
        if from_me and mid in agents:
            continue  # an agent sent it from the user's account: its turn's task has it
        body = _unsealed(r["body"])
        text, media = str(body.get("t") or ""), _WA_MEDIA.get(str(r["kind"] or ""), "")
        if not text.strip() and not media:
            continue
        group = chat.endswith("@g.us") or bool((chats.get(chat) or {}).get("is_group"))
        title = _unsealed_text((chats.get(chat) or {}).get("name")) or (chat if group else person(chat))
        if from_me:
            party = None if group else {"role": "recipient", "kind": "wa_jid", "address": chat, "display": title,
                                        "verified": True}
        else:
            sender = canon(r["sender_jid"]) or chat
            party = {"role": "requester", "kind": "wa_jid", "address": sender,
                     "display": person(sender, str(body.get("p") or "")), "verified": True}
        addressed = (not group or from_me or bool(mine_ids & set(re.findall(r"@(\d{5,})", text)))
                     or _wa_replies_to_me(path, str(r["chat_jid"]), str(r["reply_to"] or "")))
        msgs.append(_Msg(ref=f"{uid}:{mid}", conv=f"waweb:{chat}", at=float(r["ts"]), outgoing=from_me, party=party,
                         group=group, blocked="" if addressed else "group", refs={"chat_jid": chat, "message_id": mid},
                         kept=kept, **_texts(text, media)))
    _take(p, uid, key, msgs)
    if last > after:
        _save_cursor(p.store, key, uid, {**cur, "rowid": last})


def _wa_replies_to_me(path: Path, chat_jid: str, reply_to: str) -> bool:
    if not reply_to:
        return False
    return bool(_rows(path, "SELECT 1 FROM messages WHERE chat_jid = ? AND id = ? AND from_me = 1",
                      (chat_jid, reply_to)))


# ── Gmail (the user's own token, on the deck's loop) ──────────────────


async def _gmail_list(uid: str, cur: dict) -> dict | None:
    """The ids of the user's mail since the cursor (``after_ms``, else the
    first look's ``since``) — and before ``before_ms`` while an older part is
    being caught up on —, newest first (inbox and sent, no chats or drafts),
    up to :data:`_GMAIL_LIST_MAX` of them; where the list stopped short of
    the rest (``below_ms``: the oldest listed one's time, None when it
    holds all of it); and the mailbox's own addresses — None when Google
    isn't connected or the mailbox can't be told."""
    import httpx

    from captain_claw.flight_deck import gmail_autoreply as ar
    from captain_claw.flight_deck.google_oauth_routes import get_valid_google_access_token

    token = await get_valid_google_access_token(uid)
    if not token:
        return None
    own = [str(a) for a in cur.get("own") or [] if isinstance(a, str) and a]
    own_at = float(cur.get("own_at") or 0)
    async with httpx.AsyncClient(timeout=ar.GMAIL_TIMEOUT_S) as client:
        if not own or time.time() - own_at > _OWN_ADDRESSES_S:
            mailbox = await ar._mailbox(client, token)
            if mailbox is not None and (mailbox.aliases_ok or not own):
                own, own_at = mailbox.own, time.time()
        if not own:
            return None
        start_ms = int(cur.get("after_ms") or float(cur.get("since") or 0) * 1000)
        q = f"after:{max(0, start_ms // 1000 - _GMAIL_OVERLAP_S)} -in:chats -in:drafts"
        if cur.get("before_ms"):
            q += f" before:{int(cur['before_ms']) // 1000 + 1}"
        ids: list[str] = []
        page = ""
        while len(ids) < _GMAIL_LIST_MAX:
            params: dict[str, Any] = {"q": q, "maxResults": min(_GMAIL_PAGE, _GMAIL_LIST_MAX - len(ids))}
            if page:
                params["pageToken"] = page
            st, data = await ar._gmail_get(client, token, "/messages", params)
            if st != 200:
                raise RuntimeError(f"Gmail answered {st or 'nothing'}")
            ids += [str(m["id"]) for m in data.get("messages") or []
                    if isinstance(m, dict) and ar._ID_RE.match(str(m.get("id") or ""))]
            page = str(data.get("nextPageToken") or "")
            if not page:
                break
        below = None
        if page and ids:
            # More than one list holds: where it stopped, so the rest is read next.
            st, last = await ar._gmail_get(client, token, f"/messages/{ids[-1]}", {"format": "minimal"})
            below = int(last.get("internalDate") or 0) or None if st == 200 and isinstance(last, dict) else None
    return {"ids": ids, "own": own, "own_at": own_at, "truncated": bool(page), "below_ms": below}


async def _gmail_fetch(uid: str, ids: list[str]) -> list[dict]:
    """Those emails, ``format=full`` (what the prefilter reads)."""
    import httpx

    from captain_claw.flight_deck import gmail_autoreply as ar
    from captain_claw.flight_deck.google_oauth_routes import get_valid_google_access_token

    token = await get_valid_google_access_token(uid)
    if not token:
        return []
    out = []
    async with httpx.AsyncClient(timeout=ar.GMAIL_TIMEOUT_S) as client:
        for mid in ids:
            st, msg = await ar._gmail_get(client, token, f"/messages/{mid}", {"format": "full"})
            if st == 200:
                out.append(msg)
    return out


def _gmail_read(uid: str, cur: dict, seen: Callable[[list[str]], set[str]]) -> dict | None:
    """``{own, own_at, messages, more, truncated, below_ms}`` — the mailbox's
    own addresses and the oldest :data:`_GMAIL_MAX` of its new emails (Gmail
    API messages; those the board read already are left out), whether more
    new ones were listed than read (newer than those: next pass), and where
    the list stopped short (:func:`_gmail_list`) — or None when it can't be
    read now."""
    listed = _on_loop(_gmail_list(uid, cur), _LIST_TIMEOUT_S)
    if listed is None:
        return None
    known = seen(listed["ids"])
    fresh = [i for i in listed["ids"] if i not in known]   # newest first
    take = fresh[-_GMAIL_MAX:]
    messages = (_on_loop(_gmail_fetch(uid, take), _FETCH_TIMEOUT_S) or []) if take else []
    return {"own": listed["own"], "own_at": listed["own_at"], "messages": messages, "more": len(fresh) > len(take),
            "truncated": listed.get("truncated", False), "below_ms": listed.get("below_ms")}


def _agent_sent(store: TaskboardStore, uid: str, mid: str) -> bool:
    """An email from the user's account that an agent or the auto-reply
    sent — its own records have it (the deck's sends, the tool-call
    register), and it is never the user's own words."""
    from captain_claw.flight_deck import taskboard_ingest

    if store.task_for_key(uid, f"gmail:msg:{mid}"):
        return True
    mine, args = ("", []) if not taskboard_sources._auth_enabled() else (" AND owner_id = ?", [uid])
    for sql in (f"SELECT 1 FROM gmail_sends WHERE gmail_message_id = ?{mine} LIMIT 1",
                f"SELECT 1 FROM gmail_autoreplies WHERE sent_message_id = ?{mine} LIMIT 1"):
        try:
            if _rows(taskboard_sources._fd_db_path(), sql, (mid, *args)):
                return True
        except sqlite3.OperationalError:
            pass
    try:
        audit = taskboard_ingest._audit_path()
        return bool(audit.is_file() and _rows(
            audit, "SELECT 1 FROM tool_calls WHERE refs LIKE ? AND json_extract(refs, '$.gmail_message_id') = ? "
                   "LIMIT 1", (f"%{mid}%", mid)))
    except sqlite3.Error:
        return False


def _gmail_msg(uid: str, inb: gmail_inbound.Inbound, own_keys: set[str], *, outgoing: bool,
               blocked: str = "") -> _Msg | None:
    """One email as the board takes it: who it is from (the sender —
    verified when Gmail authenticated it) or to, the others on it, its
    subject and what it says without the quoted thread."""
    def people(header: str, role: str) -> list[dict]:
        found = _addresses(", ".join(gmail_inbound.header_values(inb, header)), role)
        return [x for x in found if gmail_inbound.own_address_key(x["address"]) not in own_keys]

    mid = inb.message_id
    if outgoing:
        to = people("To", "recipient") + people("Cc", "cc")
        if not to:
            return None  # mail to themselves
        party, others = {**to[0], "role": "recipient"}, to[1:]
    else:
        sender = gmail_inbound.sender_address(inb)
        party = {"role": "requester", "kind": "email", "address": sender, "display": inb.from_name or sender,
                 "verified": gmail_inbound.sender_authenticated(inb)[0]}
        # Whoever wrote the email wrote these headers too: named, never verified
        # (a next step goes to no one an inbound email put on its task).
        others = [{**x, "role": "cc", "verified": False} for x in people("To", "cc") + people("Cc", "cc")
                  if x["address"] != sender]
    body = gmail_inbound.strip_quoted(inb.text)
    thread = inb.thread_id or mid
    return _Msg(ref=f"{uid}:{mid}", conv=f"gmail:thread:{thread}", at=inb.internal_ms / 1000.0, outgoing=outgoing,
                party=party, others=others, blocked=blocked, refs={"thread_id": thread, "message_id": mid},
                extra_keys=[f"gmail:msg:{mid}"], **_texts(body, "", inb.subject or "(no subject)"))


def _gmail(p: _Pass, uid: str) -> None:
    key, store = "gmail", p.store
    cur = _cursor(store, key, uid)
    since_ms = int(float(cur["since"]) * 1000)
    after_ms = int(cur.get("after_ms") or since_ms)
    done = {str(k): int(v) for k, v in (cur.get("done") or {}).items() if isinstance(v, (int, float))}

    def seen(ids: list[str]) -> set[str]:
        refs = store.passive_seen(SOURCE[key], [f"{uid}:{i}" for i in ids])
        return {i for i in ids if i in done or f"{uid}:{i}" in refs}

    got = _gmail_read(uid, {**cur, "after_ms": after_ms}, seen)
    if got is None:
        return
    own = [str(a) for a in got.get("own") or [] if a]
    own_keys = {gmail_inbound.own_address_key(a) for a in own}
    now_ms = int(time.time() * 1000)
    msgs, put_off, newest = [], None, after_ms
    for inb in sorted((gmail_inbound.parse_message(raw) for raw in got.get("messages") or []),
                      key=lambda i: i.internal_ms):
        mid, at_ms = inb.message_id, inb.internal_ms
        if not mid or at_ms <= 0 or "DRAFT" in inb.label_ids:
            continue
        sender = gmail_inbound.sender_address(inb)
        outgoing = "SENT" in inb.label_ids and bool(sender) and gmail_inbound.own_address_key(sender) in own_keys
        if outgoing and now_ms - at_ms < _GMAIL_OWN_WAIT_S * 1000:
            put_off = at_ms if put_off is None else min(put_off, at_ms)
            continue  # an agent's send may not be on its records yet
        done[mid], newest = at_ms, max(newest, at_ms)
        if outgoing:
            if _agent_sent(store, uid, mid):
                continue
            msg = _gmail_msg(uid, inb, own_keys, outgoing=True)
        else:
            code, _ = gmail_inbound.prefilter(inb, own_addresses=own, enabled_at_ms=since_ms, now_ms=now_ms,
                                              max_age_s=int(KEPT_S))
            if code in _GMAIL_NEVER or store.task_for_key(uid, f"gmail:msg:{mid}"):
                continue  # not someone writing now — or on the board already (an auto-reply held it)
            msg = _gmail_msg(uid, inb, own_keys, outgoing=False,
                             blocked="" if code in _GMAIL_PASSES else f"gmail:{code}")
        if msg is not None:
            msgs.append(msg)
    _take(p, uid, key, msgs)
    # Where the next pass lists from. Gmail lists the newest first, and this
    # pass read the oldest of what it listed: what it listed but didn't read
    # is newer (after: the newest read); what the list didn't reach is older
    # (``before_ms``: caught up on first, from where the list stopped, the
    # cursor kept where it was — every email is read once, none lost).
    before = int(cur.get("before_ms") or 0) or None
    if got.get("truncated"):
        after = after_ms
        before = got.get("below_ms") or before
    elif got.get("more") or not before:
        after = newest
    else:
        after, before = max(newest, before), None   # the older part is caught up on
    if put_off is not None:
        after = min(after, put_off - 1)
    keep = sorted(((k, v) for k, v in done.items() if v >= after - _GMAIL_OVERLAP_S * 1000),
                  key=lambda kv: kv[1])[-_GMAIL_DONE_KEPT:]
    _save_cursor(store, key, uid, {**{k: v for k, v in cur.items() if k != "before_ms"}, "after_ms": after,
                                   **({"before_ms": int(before)} if before else {}), "done": dict(keep),
                                   "own": own, "own_at": got.get("own_at") or 0})


_READERS: dict[str, Callable[[_Pass, str], None]] = {
    "telegram_inbox": _telegram, "whatsapp_web": _whatsapp, "gmail": _gmail}
