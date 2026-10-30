"""Taskboard history — the last 30 days of the caller's agents' chats, filled
in once, and each agent's own to-do list (v2).

  POST /fd/taskboard/backfill   fill the caller's board in from their agents' history (once)

The first time a board opens after v2 (``GET /status``, while the board's
background work runs), or when it asks (``POST /backfill``), a background job
reads each of the caller's agents' ``sessions.db`` — read only (``mode=ro``),
straight from the agent's folder, never waking the agent — and puts the
turns of the last 30 days on the board as the turn register would have: the
same gate (``taskboard_ingest._board`` says whose board and whether it is the
user's own chat, ``_gate`` what the turn is, ``_reply`` its answer), so a
code-made opening, a scheduled run or a turn the user's rules keep off makes
nothing here either. What an agent's register reported already (from the
first of its turns the board keeps) is the register's: the backfill reads
only what came before it.

Whose board (decision #4): each user's job writes their own board only — the
owner's sessions to the owner, a member's sessions on a shared agent to that
member (while they are one), the Telegram account a member linked as v1
routes it (a Telegram user's own session names its account only in the
agent's ``app_state``: one it doesn't is passed over) — never someone else's
turns, and the owner never a member's. A history turn is a task of its own:
never attached to a live one (the person's open request today), and only a
task it made is marked historical.

A backfilled task is ``flags.historical``, kept as long as any task, titled
by its first line. The rules sort it at once (``taskboard_triage.rule_moves``,
logged as a ``rule`` decision) and it never waits in the triage queue: no
extraction, no Jev, none of the day's caps spent — a month of chats would use
up a week of them and hold the live board's queue behind it. A turn that
ended without a reply is owed by nobody (it is long over). No next step is
ever proposed on one, and nothing notifies about it. Each turn is keyed
``turn:<id>``, the id made from its agent, session and message, so a run
again (Flight Deck restarted mid-way) adds nothing.

Not covered: Docker agents (their sessions.db is inside the container), turns
compacted away, and a message from before turns recorded their source in a
session someone else writes in (whose words it is can't be told).

The agents' to-do lists (``todo_items``) are a sweep adapter
(:func:`agent_todos`): every item a task ``todo:<slug>:<id>``, on the board
of the user it belongs to, moved as the agent moves it.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote

from fastapi import APIRouter, Depends

from captain_claw import msg_origin, turn_source
from captain_claw.flight_deck import (
    taskboard_ingest,
    taskboard_sources,
    taskboard_store,
    taskboard_triage,
)
from captain_claw.flight_deck.auth import get_current_user, get_db
from captain_claw.flight_deck.taskboard_sources import _epoch, _head, _iso, _json, _rows
from captain_claw.flight_deck.taskboard_store import TaskboardStore
from captain_claw.logging import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/fd/taskboard", tags=["taskboard"])

WINDOW_S = taskboard_sources.TURNS_KEPT_S   # how far back the history goes: 30 days
STATES = ("idle", "running", "done", "error")
PROGRESS = ("agents_done", "agents_total", "turns", "tasks")
_TURNS_PER_AGENT = 2000   # the newest of one agent's turns a backfill takes
_FRESH_S = 600.0          # a turn without a reply this young may still be running: left out
_OPEN_TIMEOUT_S = 2.0     # an agent writing its sessions.db holds a read back this long at most
_HEAD_MAX = 1000          # text_head / reply_head, as the register keeps them
_HISTORY_WHY = ("It was filled in from your agent's earlier chats (the last 30 days), so rules sorted it "
                "and no next step is suggested for it.")
_NO_REPLY_WHY = "Its turn ended long ago without a reply, so nobody is on it now."

# Each board's backfill while Flight Deck runs: {state, done_at, progress, message, task}.
_JOBS: dict[str, dict] = {}


# ── Reading an agent's history (read only) ────────────────────────────


def sessions_db(slug: str) -> Path:
    """Where a process agent keeps its sessions (``agent_idle`` reads it there too)."""
    return taskboard_store.data_dir() / slug / "data" / "sessions" / "sessions.db"


def _connect(path: Path) -> sqlite3.Connection:
    """An agent's sessions.db, read only: never created, never written
    (``mode=ro``). Each SELECT holds its read lock only while it runs, and an
    agent holding the file fails it soon (``sqlite3.OperationalError``)
    rather than wait."""
    conn = sqlite3.connect(f"file:{quote(str(path))}?mode=ro", uri=True, timeout=_OPEN_TIMEOUT_S)
    conn.row_factory = sqlite3.Row
    return conn


def _is_managed(slug: str) -> bool:
    """A worker Flight Deck runs itself (``agent_sharing.is_managed_agent``):
    by its name — and a Dubina one by its registry description too."""
    from captain_claw.flight_deck.agent_sharing import is_managed_agent

    if is_managed_agent(slug, ""):
        return True
    return slug.startswith("dubina-") and taskboard_ingest._managed(slug)


def _turn_id(slug: str, session_id: str, msg: dict) -> str:
    """A history turn's id (32 hex, as the register's): the same message is
    the same turn on every run."""
    mark = str(msg.get("message_id") or "") or f"{msg.get('timestamp')}:{str(msg.get('content') or '')[:200]}"
    return hashlib.sha256(f"history:{slug}:{session_id}:{mark}".encode()).hexdigest()[:32]


def _masked(text: Any) -> str:
    from captain_claw.tool_audit import mask_text

    return mask_text(str(text or "")[: _HEAD_MAX * 2])[:_HEAD_MAX]


def _is_reply(msg: Any) -> bool:
    """The agent's answer in a turn: an assistant message with text that no
    gate turned down — not a tool call, not an automation lane's note."""
    return (isinstance(msg, dict) and msg.get("role") == "assistant" and not msg.get("tool_calls")
            and isinstance(msg.get("content"), str) and bool(msg["content"].strip())
            and not msg_origin.is_lane_note(msg) and not msg_origin.is_turned_down(msg))


def _session_turns(agent: dict, session: dict, *, since: float, until: float, now: float,
                   telegram_only: bool = False, telegram: str | None = "") -> list[dict]:
    """One session's turns in ``[since, until)``, each as the register's row
    would hold it (``turns``), with ``_rec`` — what ``taskboard_ingest._board``
    reads to say whose board it is. Every turn opener counts (the gate sorts
    out what isn't a person's ask); a message from before turns recorded
    their source, in a session whose stored origin is someone else's address,
    is passed over (whose words it is can't be told). *telegram*: the
    Telegram account whose own session it is (:func:`_telegram_users`) — the
    chat its Telegram turns came from, as the register names it live; None
    for a Telegram user's session nobody can be told the owner of, whose
    Telegram turns are passed over (they may be a member's: never filed as
    the agent owner's)."""
    from captain_claw import member_privacy
    from captain_claw.turn_register import _GUARD_BLOCK_RE, channel_for

    slug, sid, meta, msgs = agent["slug"], session["id"], session["metadata"], session["messages"]
    speaker = str(meta.get("speaker_id") or "")
    origin = meta.get("origin") if isinstance(meta.get("origin"), dict) else {}
    okind, address = str(origin.get("kind") or "").strip(), str(origin.get("address") or "").strip()
    outside = okind.lower() in taskboard_ingest._ORIGIN_CHANNELS
    openers = [i for i, m in enumerate(msgs) if msg_origin.is_turn_opener(m)]
    out: list[dict] = []
    for n, i in enumerate(openers):
        m = msgs[i]
        at = _epoch(m.get("timestamp"))
        if at is None or not since <= at < until:
            continue
        raw_source = str(m.get("source") or "")
        if not raw_source and outside:
            continue
        source = turn_source.normalize(raw_source) or raw_source.strip()
        channel = channel_for(raw_source, str(m.get("channel") or ""))
        if telegram_only and channel != "telegram":
            continue
        rest = msgs[i + 1:openers[n + 1] if n + 1 < len(openers) else len(msgs)]
        reply = next((str(r["content"]) for r in reversed(rest) if _is_reply(r)), "")
        if not reply and now - at < _FRESH_S:
            continue  # it may still be running: the register (or a later read) has it
        refs: dict[str, Any] = {}
        if okind and address:
            refs["origin"] = {"kind": okind[:32], "address": address[:128]}
        if channel == "whatsapp" and meta.get("whatsapp_waid"):
            refs["waid"] = str(meta["whatsapp_waid"])[:32]
        if channel == "telegram":
            chat = address if okind.lower() in ("telegram", "tg") else (telegram or "")
            if chat:
                refs["telegram_chat_id"] = chat[:32]
            elif telegram is None:
                continue  # someone's own Telegram session, whose isn't known: on no board
        refs = taskboard_ingest._small(refs)
        content = str(m.get("content") or "")
        literal = msg_origin.detect_literal(content)
        stamps = [t for t in (_epoch(r.get("timestamp")) for r in rest if isinstance(r, dict)) if t is not None]
        tid = _turn_id(slug, sid, m)
        out.append({
            "turn_id": tid, "agent_owner_id": agent["owner"], "user_id": "", "agent_slug": slug,
            "session_id": sid[:128], "session_name": _head(session.get("name"), 200),
            "origin": msg_origin.origin_of(m),
            "origin_detail": taskboard_ingest._small({k: str(v)[:64] for k, v in (
                ("detail", m.get("origin_detail")), ("channel", m.get("channel"))) if v}),
            "channel": channel, "source": source, "source_label": _head(m.get("source_label"), 80),
            "speaker_id": "", "speaker_label": "", "member": False,
            "private": bool(m.get(member_privacy.FLAG)) or bool(member_privacy.header_level(content)),
            "own_chat": False, "opener": True, "human": msg_origin.is_human_input(m),
            "literal": literal[0] if literal else "",
            "text_head": _masked(msg_origin.model_view_text(m)), "ext_refs": refs, "automation": {},
            "reply_head": _masked(reply), "deliverables": [],
            "outcome": ("blocked" if _GUARD_BLOCK_RE.match(reply.strip()) else "ok") if reply else "cancelled",
            "started_at": at, "ended_at": max([at, *stamps]),
            "tool_calls": sum(1 for r in rest if isinstance(r, dict) and r.get("role") == "tool"
                              and not msg_origin.is_monitor_only(r)),
            "failed_tool_calls": 0, "gate": "", "task_id": "", "task_key": "", "open_ref": "", "close_ref": "",
            "received_at": now, "historical": True,   # never on a live task (taskboard_ingest._admit)
            "_rec": taskboard_ingest.TurnRecord(
                seq=1, phase="open", turn_id=tid, at=at, session_id=sid[:128], channel=channel[:64],
                source=source[:64], speaker_id=speaker[:160], speaker_label=_head(meta.get("speaker_name"), 200),
                member=bool(speaker), ext_refs=refs),
        })
    return out


def _telegram_users(conn: sqlite3.Connection) -> tuple[dict[str, str], dict[str, str]]:
    """Whose own session each Telegram user's is, as the agent keeps them
    (web/telegram.py's ``app_state``) — the session holds no chat id of its
    own: ``({session id: Telegram user id}, {session name: Telegram user
    id})``, a name (``tg-<username | first name | id>``) only when one
    approved user has it. A private chat's id is its person's id. Empty
    when the agent never kept them."""
    def state(key: str) -> dict:
        try:
            row = conn.execute("SELECT value FROM app_state WHERE key = ?", (key,)).fetchone()
        except sqlite3.OperationalError:
            return {}
        return _json(row[0] if row else "", {})

    def person(value: Any) -> str:
        value = str(value or "").strip()
        return value if value.isdigit() and len(value) <= 20 else ""

    by_session = {str(sid): person(tg) for tg, sid in state("telegram_user_sessions").items() if person(tg)}
    names: dict[str, set[str]] = {}
    for tg, info in state("telegram_approved_users").items():
        if person(tg) and isinstance(info, dict):
            label = str(info.get("username") or "").strip() or str(info.get("first_name") or "").strip() or tg
            names.setdefault(f"tg-{label}", set()).add(person(tg))
    return by_session, {name: next(iter(who)) for name, who in names.items() if len(who) == 1}


def _read_agent(store: TaskboardStore, agent: dict, user_id: str, now: float) -> tuple[list[dict], bool]:
    """The turns of one agent's last 30 days the user may have — up to the
    first its register reported — newest :data:`_TURNS_PER_AGENT` of them;
    and whether there were more. A member reads their own sessions on it and,
    of the owner's, only Telegram (an account they linked lands on their
    board); the owner never a member's session. Raises ``sqlite3.Error``
    when the file can't be read now (an agent holding it)."""
    path = sessions_db(agent["slug"])
    if not path.is_file():
        return [], False
    since = now - WINDOW_S
    until = min(now, store.register_since(agent["slug"]) or now)
    if until <= since:
        return [], False
    turns: list[dict] = []
    conn = _connect(path)
    try:
        by_session, by_name = _telegram_users(conn)
        heads = conn.execute("SELECT id, name, metadata FROM sessions WHERE updated_at >= ?",
                             (_iso(since),)).fetchall()
        for head in heads:
            meta = _json(head["metadata"], {})
            speaker = str(meta.get("speaker_id") or "")
            if speaker and (not agent["member"] or speaker != user_id):
                continue  # a member's session: only that member's history
            sid, name = str(head["id"]), str(head["name"] or "")
            tg = by_session.get(sid) or by_name.get(name) or (
                None if name.startswith("tg-") and not name.startswith("tg-group-") else "")
            row = conn.execute("SELECT messages FROM sessions WHERE id = ?", (head["id"],)).fetchone()
            msgs = [m for m in _json(row["messages"] if row else "", []) if isinstance(m, dict)]
            turns += _session_turns(agent, {"id": sid, "name": head["name"], "metadata": meta,
                                            "messages": msgs}, since=since, until=until, now=now,
                                    telegram_only=agent["member"] and not speaker, telegram=tg)
    finally:
        conn.close()
    turns.sort(key=lambda t: (t["started_at"], t["turn_id"]))
    return turns[-_TURNS_PER_AGENT:], len(turns) > _TURNS_PER_AGENT


# ── Onto the board (the v1 gate, then the rules) ──────────────────────


def _put(store: TaskboardStore, turn: dict, agents: dict[str, dict], now: float) -> str:
    """One history turn through v1's gate onto its board: ``made`` (a task,
    historical and sorted), ``had`` (a run before made it), else the gate's
    verdict. The board is held meanwhile, so no triage pass takes the new
    task before it is marked historical."""
    uid, tid = turn["user_id"], turn["turn_id"]
    if store.task_for_key(uid, f"turn:{tid}"):
        return "had"
    with store.holding():
        gate, task_id, key = taskboard_ingest._gate(store, turn, agents, False)
        if gate != "task" or not task_id:
            return gate
        row = store.get_task(uid, task_id) or {}
        if (row.get("source"), row.get("source_ref")) != ("agent_turns", tid):
            return "attached"  # on a task it didn't make (never, for history): that one isn't history
        muted = row.get("triage_state") == "skipped"  # a mute rule's: never sorted
        store.update_fields(uid, task_id, {"flags": {"historical": True}, "triage_state": "skipped",
                                           "triage_after": None})
        turn = {**turn, "gate": gate, "task_id": task_id, "task_key": key}
        taskboard_ingest._reply(store, turn, agents, now)
        if not muted:
            _sort(store, uid, task_id, turn)
    return "made"


def _sort(store: TaskboardStore, uid: str, task_id: str, turn: dict) -> None:
    """The rules' decision on a history task (as triage makes it without
    Jev), logged — read with its turn, which the register's table never
    holds. A turn long over that the rules leave with the agent (no reply
    came) is nobody's now."""
    view = store.triage_view(uid, task_id)
    if view is None:
        return
    view["turns"] = [turn]
    task, f = view["task"], taskboard_triage.facts(view, uid)
    fields, why = taskboard_triage.rule_moves(task, f)
    fields = taskboard_triage._settled(task, fields)
    if fields.get("status", task["status"]) == "open" and fields.get("owes_next", task["owes_next"]) == "agent":
        fields["owes_next"] = "nobody"
        why = [*why, _NO_REPLY_WHY]
    why = [*why, _HISTORY_WHY]
    fields["flags"] = {"why": why[0][:160], **(fields.get("flags") or {})}
    store.triage_write(uid, task_id, task["fingerprint"], fields=fields, decided=True, state="ok", decision={
        "judge": "rule", "qset": taskboard_triage.RULES_QSET, "answers": {},
        "state_hash": taskboard_triage._hash(f"{task_id}:{task['fingerprint']}"),
        "applied": taskboard_triage.applied_text(task, fields), "why": " ".join(why[:3])})


def _put_all(store: TaskboardStore, turns: list[dict], agents: dict[str, dict], now: float) -> int:
    made = 0
    for turn in turns:
        try:
            made += _put(store, turn, agents, now) == "made"
        except Exception as exc:  # noqa: BLE001 — one turn never stops the rest
            log.warning("Taskboard history: a turn couldn't be filled in", agent=turn["agent_slug"],
                        error=str(exc)[:200])
    return made


# ── The job, once per board ───────────────────────────────────────────


async def _caller_agents(uid: str) -> list[dict]:
    """The process agents whose history is the caller's: their own (every
    agent, on a deck without sign-in) and those shared with them while they
    are a member — never a worker Flight Deck runs itself. Docker agents
    aren't in the registry read here (their sessions.db is in the container)."""
    from captain_claw.flight_deck import agent_sharing

    index = await asyncio.to_thread(taskboard_sources._agent_index)
    auth = taskboard_sources._auth_enabled()
    out = [{"slug": slug, **info, "member": False} for slug, info in sorted(index.items())
           if not auth or (info["owner"] and info["owner"] == uid)]
    if auth:
        held = {a["slug"] for a in out}
        try:
            shares = await get_db().list_shares_for_grantee(uid, "agent")
        except Exception as exc:
            log.warning("Taskboard history: shared agents unavailable", error=type(exc).__name__)
            shares = []
        for row in shares:
            ref, owner = str(row.get("resource_id") or ""), str(row.get("owner_id") or "")
            slug = next((s for s, i in sorted(index.items()) if ref and i["ref"] == ref and i["owner"] == owner), "")
            if slug and slug not in held and await agent_sharing.member_check(get_db(), ref, owner, uid):
                out.append({"slug": slug, **index[slug], "member": True})
                held.add(slug)
    return [a for a in out if not await asyncio.to_thread(_is_managed, a["slug"])]


def _done_message(progress: dict, busy: list[str], capped: list[str]) -> str:
    tasks, turns = progress["tasks"], progress["turns"]
    if not progress["agents_total"]:
        text = "None of your agents has chats to fill in."
    elif not turns:
        text = "Your agents had no chats in the last 30 days that the board hasn't got already."
    else:
        text = (f"Filled in {tasks} task{'s' if tasks != 1 else ''} from {turns} turn{'s' if turns != 1 else ''} "
                "of your agents' chats in the last 30 days.")
    if busy:
        text += f" {', '.join(busy)} couldn't be read just then, so {'its' if len(busy) == 1 else 'their'} " \
                "history is left out."
    if capped:
        text += f" Only the latest {_TURNS_PER_AGENT} turns of {', '.join(capped)} were read."
    return text


async def backfill_user(uid: str, job: dict | None = None) -> dict:
    """Fill *uid*'s board in from their agents' last 30 days, and stamp it
    done (``backfill_done_at``) — progress in *job* as it goes. Returns what
    the status says once it ended."""
    job = job if job is not None else _new_job()
    store = taskboard_store.get_store()
    now = time.time()
    try:
        agents = await _caller_agents(uid)
        index = await asyncio.to_thread(taskboard_sources._agent_index)
        progress = job["progress"]
        progress["agents_total"] = len(agents)
        busy: list[str] = []
        capped: list[str] = []
        for n, agent in enumerate(agents, 1):
            job["message"] = f"Reading {agent['name']}'s chats ({n} of {len(agents)})…"
            try:
                found, more = await asyncio.to_thread(_read_agent, store, agent, uid, now)
            except sqlite3.Error as exc:
                log.info("Taskboard history: sessions unreadable", agent=agent["slug"], error=str(exc)[:200])
                found, more = [], False
                busy.append(agent["name"])
            if more:
                capped.append(agent["name"])
            who = {"owner": agent["owner"], "slug": agent["slug"]}
            cache: dict[str, Any] = {"ref": agent["ref"]}  # the agent is known: no token to look it up by
            mine: list[dict] = []
            for turn in found:
                board, speaker, name, own = await taskboard_ingest._board(turn.pop("_rec"), who, "", cache)
                if board != uid:
                    continue  # someone else's turn (a member's, or nobody's): never on this board
                member = board not in (agent["owner"], "local")
                turn.update(user_id=board, speaker_id=speaker, speaker_label=name, own_chat=own, member=member,
                            private=turn["private"] or member)
                mine.append(turn)
            progress["turns"] += len(mine)
            progress["tasks"] += await asyncio.to_thread(_put_all, store, mine, index, now)
            progress["agents_done"] += 1
        message = _done_message(progress, busy, capped)
        saved = await asyncio.to_thread(store.finish_history, uid, progress=dict(progress), message=message)
        job.update(state="done", done_at=saved["done_at"], message=message)
    except asyncio.CancelledError:
        job.update(state="error", message="Stopped before it finished.")
        raise
    except Exception as exc:  # noqa: BLE001 — said on the board; POST /backfill tries again
        log.warning("Taskboard history failed", user=uid, error=str(exc)[:200])
        job.update(state="error", message=f"The history couldn't be filled in: {str(exc)[:200]}")
    return view(job)


def _new_job() -> dict:
    return {"state": "running", "done_at": None, "progress": dict.fromkeys(PROGRESS, 0), "message": "Starting…"}


def _start(uid: str) -> dict:
    """*uid*'s backfill started — or the one running already: two first looks
    at once (the board and its badge) start one job."""
    job = _JOBS.get(uid)
    if job is not None and job["state"] == "running":
        return job
    job = _new_job()
    _JOBS[uid] = job
    job["task"] = asyncio.create_task(backfill_user(uid, job))
    return job


def view(job: dict) -> dict:
    """``status.backfill``: ``{state, done_at, progress, message}``."""
    progress = job.get("progress") or {}
    return {"state": job["state"] if job.get("state") in STATES else "idle", "done_at": job.get("done_at"),
            "progress": {k: int(progress.get(k) or 0) for k in PROGRESS}, "message": str(job.get("message") or "")}


async def status(uid: str, *, start: bool = False) -> dict:
    """Where *uid*'s backfill stands — and, with *start* (the first
    ``GET /status`` while the board's background work runs), the job started
    when it never ran to its end and hasn't failed since Flight Deck
    started (a failure waits for ``POST /backfill``)."""
    job = _JOBS.get(uid)
    if job is not None and job["state"] == "running":
        return view(job)
    saved = await asyncio.to_thread(taskboard_store.get_store().history_state, uid)
    if saved["state"] == "done":
        return view(saved)
    job = _JOBS.get(uid)  # again: another look may have started it meanwhile
    if job is None and start:
        job = _start(uid)
    return view(job or saved)


async def stop() -> None:
    """Flight Deck is stopping: every backfill still reading stops (it is
    started again the next time its board opens; the ``turn:`` keys add
    nothing twice)."""
    tasks = [job["task"] for job in _JOBS.values() if job.get("task") is not None and not job["task"].done()]
    for task in tasks:
        task.cancel()
    if tasks:
        await asyncio.wait(tasks, timeout=5.0)
    _JOBS.clear()


async def wait(uid: str) -> dict | None:
    """The status once *uid*'s running backfill ended (tests, shutdown)."""
    job = _JOBS.get(uid)
    task = (job or {}).get("task")
    if task is not None:
        await asyncio.wait([task])
    return view(job) if job else None


def sources_view(sources: list[dict], backfill: dict) -> list[dict]:
    """The Sources sheet with v2's lines: the history as the backfill stands,
    and the agents' to-do lists (tracked)."""
    state, done_at, progress = backfill["state"], backfill["done_at"], backfill["progress"]
    if state == "done":
        when = time.strftime("%Y-%m-%d", time.localtime(done_at)) if done_at else ""
        n = progress["tasks"]
        note = (f"Filled in{' on ' + when if when else ''}: {n} task{'s' if n != 1 else ''} from your agents' "
                "chats of the last 30 days, sorted by rules — no next step is ever suggested for them.")
    elif state == "running":
        note = f"Filling in your agents' chats of the last 30 days ({progress['agents_done']} of " \
               f"{progress['agents_total']} agents)…"
    elif state == "error":
        note = backfill["message"] or "The history couldn't be filled in."
    else:
        note = "Your agents' chats of the last 30 days, filled in once — the first time the board opens."
    history = {"key": "history", "label": "The last 30 days", "tracked": state in ("running", "done"),
               "on": state in ("running", "done"), "note": note}
    todos = {"key": "agent_todos", "label": "Your agents' to-do lists", "tracked": True, "on": True,
             "note": "What each agent keeps on its own to-do list, and when it ticks an item off."}
    out = [history if s.get("key") == "history" else s for s in sources]
    if not any(s.get("key") == "history" for s in out):
        out.append(history)
    if not any(s.get("key") == "agent_todos" for s in out):
        at = next((i + 1 for i, s in enumerate(out) if s.get("key") == "agent_chats"), len(out))
        out.insert(at, todos)
    return out


@router.post("/backfill")
async def taskboard_backfill(user: dict = Depends(get_current_user)) -> dict:
    """Fill the caller's board in from their agents' last 30 days — once:
    while it runs, or once it is done, this only says how it stands."""
    uid = str(user.get("id") or "") or "local"
    current = await status(uid)
    if current["state"] in ("running", "done"):
        return {"backfill": current}
    return {"backfill": view(_start(uid))}


# ── The agents' own to-do lists (a sweep adapter) ─────────────────────


_TODO_OPEN = ("pending", "in_progress")
_TODO_URGENCY = {"high": 2, "urgent": 3}


def _members(info: dict) -> set[str]:
    """Who is a member of a shared agent now — its owner's grants to users
    still on this deck, from Flight Deck's own records (as
    ``agent_sharing.member_check`` asks them) — nobody when they can't be
    read (an item of a member's session is then on no board)."""
    if not info.get("ref") or not info.get("owner"):
        return set()
    try:
        rows = _rows(taskboard_sources._fd_db_path(),
                     "SELECT s.grantee_id FROM resource_shares s JOIN users u ON u.id = s.grantee_id "
                     "WHERE s.resource_type = 'agent' AND s.resource_id = ? AND s.owner_id = ?",
                     (info["ref"], info["owner"]))
    except sqlite3.Error:
        return set()
    return {str(r["grantee_id"]) for r in rows}


def _read_todos(path: Path, since: str, first: str) -> tuple[list[dict], set[str], dict[str, str | None]]:
    """``(items changed since the cursor — the first read: open ones and
    those touched in the last 30 days —, every item's id, the speaker of each
    session they name: "" for the owner's, ``tg:<id>`` for a Telegram user's
    own (:func:`_telegram_users`), None for one whose it was can't be told —
    a Telegram user's the agent doesn't name, one gone from sessions.db the
    agent's record of its members' sessions can't speak for,
    :func:`_gone_sessions`)``, read only."""
    conn = _connect(path)
    try:
        if since:
            rows = conn.execute("SELECT * FROM todo_items WHERE updated_at >= ? ORDER BY updated_at LIMIT ?",
                                (since, taskboard_sources._BATCH)).fetchall()
        else:
            rows = conn.execute(f"SELECT * FROM todo_items WHERE status IN ({','.join('?' * len(_TODO_OPEN))}) "
                                "OR updated_at >= ? ORDER BY updated_at LIMIT ?",
                                (*_TODO_OPEN, first, taskboard_sources._BATCH)).fetchall()
        items = [dict(r) for r in rows]
        ids = {str(r[0]) for r in conn.execute("SELECT id FROM todo_items")}
        named = sorted({str(s) for r in items for s in (r.get("source_session"), r.get("target_session")) if s})
        speakers: dict[str, str | None] = {}
        by_session, by_name = _telegram_users(conn) if named else ({}, {})
        for i in range(0, len(named), 500):
            chunk = named[i:i + 500]
            for s in conn.execute("SELECT id, name, metadata FROM sessions "
                                  f"WHERE id IN ({','.join('?' * len(chunk))})", chunk):
                sid, name = str(s["id"]), str(s["name"] or "")
                speaker = str(_json(s["metadata"], {}).get("speaker_id") or "")
                tg = by_session.get(sid) or by_name.get(name)
                if not speaker and (tg or (name.startswith("tg-") and not name.startswith("tg-group-"))):
                    # A Telegram user's own session: whose by the account (None: can't be told).
                    speakers[sid] = f"tg:{tg}" if tg else None
                else:
                    speakers[sid] = speaker
    finally:
        conn.close()
    # Sessions gone since: a Telegram user's the agent still names, else its record of members' sessions.
    speakers.update({s: f"tg:{by_session[s]}" for s in named if s not in speakers and s in by_session})
    gone = [s for s in named if s not in speakers]
    if gone:
        speakers.update(_gone_sessions(path, gone))
    return items, ids, speakers


def _session_slug(session_id: str) -> str:
    """A session's id as the agent files its saved folders (and its member
    sessions) under — ``tools.write.WriteTool._normalize_session_id``."""
    value = "".join(c if c.isalnum() or c in "._-" else "-" for c in session_id.strip()).strip("-")
    return value or "default"


def _gone_sessions(path: Path, session_ids: list[str]) -> dict[str, str | None]:
    """Whose each of *session_ids* — sessions no longer in sessions.db (the
    owner deleted them) — was, by the agent's own record of its members'
    sessions, which outlives a session (``saved_attribution.db``
    ``member_sessions``, written when a member's session is made): the
    member's id, or "" for one it never recorded (the owner's); None for
    every one when the record can't be read (whose it was can't be told —
    an item of it is on no board). No record at all: the agent never had a
    member's session."""
    record = path.with_name("saved_attribution.db")
    if not record.is_file():
        return dict.fromkeys(session_ids, "")
    try:
        conn = _connect(record)
        try:
            rows = {str(r[0]): str(r[1] or "") for r in conn.execute("SELECT slug, speaker_id FROM member_sessions")}
        finally:
            conn.close()
    except sqlite3.Error:
        return dict.fromkeys(session_ids, None)
    return {s: rows.get(_session_slug(s), "") for s in session_ids}


def _todo_board(p: taskboard_sources._Pass, info: dict, row: dict, speakers: dict[str, str | None],
                members: set[str]) -> tuple[str | None, bool]:
    """``(board, private)`` of one item: the user it belongs to — the session
    that made it says (else the one it is for): a member's session is that
    member's, while they are one, and never the owner's; a Telegram user's
    session goes by the account, as v1 routes a turn (:func:`_telegram_board`);
    one whose it was can't be told puts the item on no board; an item
    naming no session is the agent's owner's."""
    if not p.auth:
        return "local", False
    for sid in (row.get("source_session"), row.get("target_session")):
        if not sid:
            continue
        who = speakers.get(str(sid))
        if who is None:
            return None, False
        if who.startswith("tg:"):
            return _telegram_board(p, info, who[3:], members)
        if who:
            return (p.board_user(who) if who in members else None), True
        break
    return p.board_user(info["owner"]), False


def _telegram_board(p: taskboard_sources._Pass, info: dict, tg: str,
                    members: set[str]) -> tuple[str | None, bool]:
    """``(board, private)`` of an item from a Telegram user's own session, as
    ``taskboard_ingest._board`` routes their turns: the owner's when the
    owner linked that account to the deck's bot; the one member's of this
    agent who did (their board, private); else someone writing to the
    owner's agent — the owner's."""
    try:
        users = {str(r["user_id"]) for r in _rows(
            taskboard_sources._fd_db_path(),
            "SELECT user_id FROM telegram_links WHERE status = 'approved' AND tg_user_id = ?", (tg,))}
    except sqlite3.Error:
        return None, False  # whose account it is can't be told now
    if info["owner"] in users:
        return p.board_user(info["owner"]), False
    mine = sorted(users & members)
    if len(mine) == 1:
        return p.board_user(mine[0]), True
    return p.board_user(info["owner"]), False


def _todo_move(row: dict, agent_name: str) -> tuple[str, dict, str, float | None]:
    """``(state, patch, summary, when)`` for an item as it stands now. One
    cancelled after it was done (``archive_old_todos`` puts old done items
    away so) is still done."""
    status, mine = str(row.get("status") or "pending"), row.get("responsible") == "human"
    changed = _epoch(row.get("updated_at"))
    if status == "done" or (status == "cancelled" and row.get("completed_at")):
        return "done", {"status": "done", "owes_next": "nobody", "evidence": "claimed", "result": "success",
                        "closed_by": "source", "closed_by_party": "" if mine else agent_name}, \
            "Ticked off the to-do list", _epoch(row.get("completed_at")) or changed
    if status == "cancelled":
        return "cancelled", {"status": "dismissed", "owes_next": "nobody", "closed_by": "source"}, \
            "Cancelled on the to-do list", changed
    owes = "me" if mine else "agent"
    summary = "On the to-do list for you" if mine else (
        f"{agent_name or 'The agent'} is working on it" if status == "in_progress" else "On the agent's to-do list")
    return f"{status}:{owes}", {"status": "open", "owes_next": owes}, summary, changed


def agent_todos(p: taskboard_sources._Pass) -> None:
    """Each agent's own to-do list (``todo_items`` in its sessions.db, read
    only — never waking it) as tasks ``todo:<slug>:<id>``, kind todo: done →
    done, cancelled → dismissed, else open — the user's move when the item is
    the human's, the agent's otherwise; an item taken off the list is
    dismissed. Each on the board of the user it belongs to
    (:func:`_todo_board`). A cursor per agent (its items' ``updated_at``)
    reads only what changed; an agent whose file is busy is read next pass."""
    cur = _json(p.cursor("agent_todos"), {})
    first = _iso(time.time() - WINDOW_S)
    for slug, info in sorted(p.agents.items()):
        path = sessions_db(slug)
        if not path.is_file() or _is_managed(slug):
            continue
        members = _members(info) if p.auth else set()
        if p.user_id is not None and p.auth and p.user_id != info["owner"] and p.user_id not in members:
            continue  # a one-user pass: an agent that is neither theirs nor shared with them
        since = str(cur.get(slug) or "")
        try:
            rows, ids, speakers = _read_todos(path, since, first)
        except sqlite3.Error as exc:
            log.info("Taskboard to-dos unreadable", agent=slug, error=str(exc)[:200])
            continue
        agent = p.agent(slug)
        for row in rows:
            uid, private = _todo_board(p, info, row, speakers, members)
            if uid is None or (p.user_id is not None and p.auth and uid != p.user_id):
                continue
            todo_id, content = str(row["id"]), str(row.get("content") or "")
            key, ref_key = f"todo:{slug}:{todo_id}", f"{slug}:{todo_id}"
            state, patch, summary, at = _todo_move(row, agent["agent_name"])
            create = {"title": _head(content, 160) or "To-do", "ask": _head(content, 600), **agent, "kind": "todo",
                      "direction": "internal", "channel": "", "owes_next": patch.get("owes_next", "agent"),
                      "urgency": _TODO_URGENCY.get(str(row.get("priority") or ""), 0), "private": int(private),
                      "source": "agent_todos", "source_ref": ref_key, "triage_state": "skipped"}
            p.ingest("agent_todos", uid, keys=[key], create=create,
                     touch={k: create[k] for k in ("title", "ask", "urgency")},
                     event={"source": "agent_todos", "source_ref": f"{ref_key}:made",
                            "at": _epoch(row.get("created_at")), "kind": "note", "actor_kind": "agent",
                            "actor": agent["agent_name"], "summary": f"To-do: {_head(content, 200)}",
                            "private": private, "refs": {"todo_id": todo_id}})
            ref = p.store.next_state_ref("agent_todos", ref_key, state)
            if ref:
                p.ingest("agent_todos", uid, keys=[key], create=create, patch=patch,
                         event={"source": "agent_todos", "source_ref": ref, "at": at or time.time(),
                                "kind": "status", "actor_kind": "agent", "actor": agent["agent_name"],
                                "summary": summary, "private": private,
                                "refs": {"todo_id": todo_id, "state": state}})
        _todos_gone(p, slug, ids)
        if rows:
            cur[slug] = max(str(r.get("updated_at") or "") for r in rows) or since
    p.save_cursor("agent_todos", json.dumps({s: v for s, v in sorted(cur.items()) if s in p.agents}))


def _todos_gone(p: taskboard_sources._Pass, slug: str, ids: set[str]) -> None:
    """Open to-do tasks whose item is no longer on the agent's list: dismissed."""
    prefix = f"todo:{slug}:"
    for t in p.store.keyed_tasks(prefix, user_id=p.user_id if p.auth else None, status="open"):
        todo_id = t["key"].removeprefix(prefix)
        if todo_id in ids:
            continue
        ref = p.store.next_state_ref("agent_todos", f"{slug}:{todo_id}", "gone")
        if ref:
            p.ingest("agent_todos", t["user_id"], keys=[t["key"]],
                     patch={"status": "dismissed", "owes_next": "nobody", "closed_by": "source"},
                     event={"source": "agent_todos", "source_ref": ref, "at": time.time(), "kind": "status",
                            "actor_kind": "system", "summary": "Taken off the agent's to-do list",
                            "refs": {"todo_id": todo_id, "state": "gone"}})
