"""The Taskboard's v2 history and to-do lists. The last 30 days of an agent's
chats are filled in once per board from its sessions.db — read only, never
written — through v1's gate (a code-made opening, a scheduled run, an older
turn or one the register reported already makes nothing), a member's turns
only on the member's board, every task historical and sorted by rules (no
Jev, no extraction, no triage queue, no proposal), a run again adding nothing,
a missing or busy file passed over, progress in ``GET /status``. Each agent's
own to-do list is a task per item, moved as the agent moves it, on the board
of the user it belongs to."""

from __future__ import annotations

import asyncio
import hashlib
import json
import secrets
import sqlite3
import threading
import time
from datetime import UTC, datetime
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI

from captain_claw.flight_deck import (
    auth,
    taskboard,
    taskboard_backfill,
    taskboard_sources,
    taskboard_triage,
)
from captain_claw.flight_deck.taskboard_sources import run_sweep
from captain_claw.session import SessionManager

from .test_taskboard_sources import HELPER_REF, env  # noqa: F401

DAY = 86400.0


@pytest.fixture()
async def deck(env, monkeypatch):  # noqa: F811
    """The v0 deck (Alice owns "helper"), its users in the auth DB, no job
    left from another test, a busy file given up on soon — and Jev and the
    extraction model never to be called."""
    auth.set_auth_db(env.db)
    monkeypatch.setattr(taskboard_backfill, "_JOBS", {})
    monkeypatch.setattr(taskboard_backfill, "_OPEN_TIMEOUT_S", 0.2)

    async def never(*a, **kw):
        raise AssertionError("a history task never reaches Jev or the extraction model")

    monkeypatch.setattr(taskboard_triage, "_ask_jev", never)
    monkeypatch.setattr(taskboard_triage, "extract", never)
    carol = (await env.db.create_user("carol@x.co", "h", "Carol"))["id"]
    yield SimpleNamespace(env=env, board=env.board, alice=env.alice, bob=env.bob, carol=carol, tmp=env.tmp)
    auth.set_auth_db(None)  # type: ignore[arg-type]


def _iso(at: float) -> str:
    return datetime.fromtimestamp(at, UTC).isoformat()


def _msg(role: str, content: str, at: float, **kw) -> dict:
    return {"message_id": secrets.token_hex(6), "role": role, "content": content, "timestamp": _iso(at), **kw}


def _ask(text: str, at: float, source: str = "web", **kw) -> dict:
    return _msg("user", text, at, origin="human", channel=source, source=source, **kw)


def _answer(text: str, at: float) -> dict:
    return _msg("assistant", text, at, origin="model")


async def _agent_db(deck, slug: str = "helper") -> object:
    """An agent's sessions.db with the agent's own schema."""
    path = deck.tmp / slug / "data" / "sessions" / "sessions.db"
    sm = SessionManager(path)
    await sm._ensure_db()
    await sm._db.close()
    return path


def _session(path, messages: list[dict], *, sid: str | None = None, meta: dict | None = None,
             name: str = "Chat") -> str:
    sid = sid or secrets.token_hex(8)
    stamps = [m["timestamp"] for m in messages] or [_iso(time.time())]
    with sqlite3.connect(path) as conn:
        conn.execute("INSERT OR REPLACE INTO sessions (id, name, messages, created_at, updated_at, metadata) "
                     "VALUES (?,?,?,?,?,?)", (sid, name, json.dumps(messages), min(stamps), max(stamps),
                                              json.dumps(meta or {})))
    return sid


def _todo(path, content: str, *, status: str = "pending", responsible: str = "bot", priority: str = "normal",
          source_session: str | None = None, completed: bool = False, at: float | None = None) -> str:
    tid = secrets.token_hex(8)
    stamp = _iso(at or time.time())
    with sqlite3.connect(path) as conn:
        conn.execute("INSERT INTO todo_items (id, content, status, responsible, priority, source_session, "
                     "created_at, updated_at, completed_at) VALUES (?,?,?,?,?,?,?,?,?)",
                     (tid, content, status, responsible, priority, source_session, stamp, stamp,
                      stamp if completed else None))
    return tid


def _set_todo(path, tid: str, **fields) -> None:
    fields["updated_at"] = _iso(time.time() + 1)
    with sqlite3.connect(path) as conn:
        conn.execute(f"UPDATE todo_items SET {', '.join(f'{k} = ?' for k in fields)} WHERE id = ?",
                     (*fields.values(), tid))


def _tasks(deck, uid: str) -> list[dict]:
    return [dict(r) for r in deck.board._conn.execute(
        "SELECT * FROM tasks WHERE user_id = ? ORDER BY created_at", (uid,))]


def _by_title(deck, uid: str) -> dict[str, dict]:
    return {t["title"]: t for t in _tasks(deck, uid)}


def _flags(task: dict) -> dict:
    return json.loads(task["flags"])


def _size(deck) -> tuple[int, int, int]:
    conn = deck.board._conn
    return tuple(conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                 for t in ("tasks", "task_events", "task_decisions"))


def _fingerprint(path) -> tuple[str, float, list[str]]:
    return (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime,
            sorted(p.name for p in path.parent.iterdir()))


# ── The 30-day backfill ───────────────────────────────────────────────


async def test_a_month_of_chats_is_filled_in_once_through_the_v1_gate(deck):
    now = time.time()
    path = await _agent_db(deck)
    _session(path, [
        _ask("What's the capital of Croatia?", now - 3 * DAY), _answer("Zagreb.", now - 3 * DAY + 5),
        _ask("Book the usual table", now - 2 * DAY),
        _answer("Blocked by output guard: that reply named a card number.", now - 2 * DAY + 3),
        _ask("Draft the board report", now - DAY),
        # A corrective mid-turn is no turn of its own.
        _msg("user", "You announced intent without acting. Do NOT narrate", now - DAY + 2, origin="corrective"),
        _msg("user", "[SCHEDULED TASK — cron job abc] Check the inbox", now - DAY + 60, origin="cron"),
        _answer("Inbox checked.", now - DAY + 70),
        _msg("user", "[Autonomous nudge] Follow up with Ana", now - DAY + 90, origin="autonomy"),
        _ask("Summarise the run", now - DAY + 120, source="scheduled"),
        _answer("Done.", now - DAY + 130),
    ])
    _session(path, [_ask("Something from long ago", now - 40 * DAY), _answer("Sure.", now - 40 * DAY + 2),
                    _ask("Still there?", now - 4 * DAY), _answer("Yes.", now - 4 * DAY + 1)])

    out = await taskboard_backfill.backfill_user(deck.alice)
    assert out["state"] == "done" and out["done_at"]
    # Seven openers in the window, every one Alice's; four of them a person's ask.
    assert out["progress"] == {"agents_done": 1, "agents_total": 1, "turns": 7, "tasks": 4}
    assert "Filled in 4 tasks from 7 turns" in out["message"]
    tasks = _by_title(deck, deck.alice)
    assert set(tasks) == {"What's the capital of Croatia?", "Book the usual table", "Draft the board report",
                          "Still there?"}
    assert not _tasks(deck, deck.bob)

    asked = tasks["What's the capital of Croatia?"]
    assert (asked["kind"], asked["channel"], asked["status"], asked["owes_next"], asked["closed_by"]) == \
        ("request", "web", "done", "nobody", "rule")
    assert _flags(asked)["historical"] is True and _flags(asked)["answered_in_chat"] is True
    assert _flags(asked)["own_chat"] is True
    assert asked["created_at"] == pytest.approx(now - 3 * DAY, abs=1)
    assert asked["triage_state"] == "ok" and asked["retention_until"] is None
    assert asked["next_steps"] == "[]"
    (party,) = deck.board.parties_for(deck.alice, [asked["id"]])[asked["id"]]
    assert (party["role"], party["kind"], party["address"], party["verified"]) == \
        ("requester", "fd_user", deck.alice, True)
    kinds = [e["kind"] for e in deck.board.events(deck.alice, asked["id"])]
    assert kinds == ["received", "reply"]
    (decision,) = deck.board.decisions(deck.alice, asked["id"])
    assert (decision["judge"], decision["qset"]) == ("rule", taskboard_triage.RULES_QSET)
    assert "earlier chats" in decision["why"]

    blocked = tasks["Book the usual table"]
    assert (blocked["status"], blocked["result"], blocked["closed_by"]) == ("failed", "failed", "evidence")
    # No reply came: long over, nobody is on it (not "agents working" for good).
    silent = tasks["Draft the board report"]
    assert (silent["status"], silent["owes_next"]) == ("open", "nobody")
    assert deck.board.lane_of(deck.alice, silent["id"]) == "other"
    assert "without a reply" in _flags(silent)["why"]

    # Never in the triage queue, so never extracted or asked about.
    assert deck.board.triage_queue(deck.alice) == [] and deck.alice not in deck.board.triage_users()
    assert all(_flags(t)["historical"] for t in _tasks(deck, deck.alice))
    assert deck.env.auto.list_actions(deck.alice) == []  # no next step proposed
    assert deck.board.get_settings(deck.alice)["backfill_done_at"] == out["done_at"]


async def test_a_run_again_adds_nothing(deck):
    now = time.time()
    path = await _agent_db(deck)
    _session(path, [_ask("One", now - 2 * DAY), _answer("1", now - 2 * DAY + 1),
                    _ask("Two", now - DAY), _answer("2", now - DAY + 1)])
    first = await taskboard_backfill.backfill_user(deck.alice)
    size = _size(deck)
    # Flight Deck restarted before it was stamped: the job runs again.
    deck.board.save_settings(deck.alice, backfill_done_at=None)
    again = await taskboard_backfill.backfill_user(deck.alice)
    assert first["progress"]["tasks"] == 2 and again["progress"]["tasks"] == 0
    assert _size(deck) == size


async def test_a_members_turns_reach_only_the_members_board(deck):
    await deck.env.db.create_share("agent", HELPER_REF, deck.alice, deck.bob)
    now = time.time()
    path = await _agent_db(deck)
    _session(path, [_ask("Alice's own question", now - DAY), _answer("Hers.", now - DAY + 1)])
    _session(path, [_ask("Bob's question", now - DAY, source="member"), _answer("His.", now - DAY + 1)],
             meta={"speaker_id": deck.bob, "speaker_name": "Bob", "speaker_lane": "A"})
    # Carol was never a member; her session (left from a share since taken back) is nobody's.
    _session(path, [_ask("Carol's question", now - DAY, source="member"), _answer("Hers.", now - DAY + 1)],
             meta={"speaker_id": deck.carol, "speaker_name": "Carol"})

    await taskboard_backfill.backfill_user(deck.alice)
    assert set(_by_title(deck, deck.alice)) == {"Alice's own question"}
    out = await taskboard_backfill.backfill_user(deck.bob)
    assert out["progress"]["tasks"] == 1
    (bobs,) = _tasks(deck, deck.bob)
    assert bobs["title"] == "Bob's question" and bobs["private"] == 1
    assert _flags(bobs)["historical"] and _flags(bobs)["own_chat"]
    (party,) = deck.board.parties_for(deck.bob, [bobs["id"]])[bobs["id"]]
    assert (party["kind"], party["address"], party["verified"]) == ("fd_user", deck.bob, True)
    out = await taskboard_backfill.backfill_user(deck.carol)
    assert out["progress"] == {"agents_done": 0, "agents_total": 0, "turns": 0, "tasks": 0}
    assert not _tasks(deck, deck.carol)
    # Never the owner's: Alice's board kept only her own.
    assert set(_by_title(deck, deck.alice)) == {"Alice's own question"}


def _app_state(path, **values) -> None:
    """What web/telegram.py keeps in the agent's ``app_state``."""
    with sqlite3.connect(path) as conn:
        for key, value in values.items():
            conn.execute("INSERT OR REPLACE INTO app_state (key, value, updated_at) VALUES (?, ?, ?)",
                         (key, json.dumps(value), _iso(time.time())))


async def test_a_members_linked_telegram_history_is_the_members_never_the_owners(deck):
    """A Telegram user's own session on the owner's agent holds no chat id
    (web/telegram.py keeps whose it is in app_state): Bob linked that account
    to the deck's bot, so his history there is his, private — never Alice's;
    a Telegram session whose account the agent doesn't name is nobody's."""
    await deck.env.db.create_share("agent", HELPER_REF, deck.alice, deck.bob)
    await deck.env.db.create_telegram_link(deck.bob, "", "", "code-b")
    await deck.env.db.update_telegram_link(deck.bob, status="approved", tg_user_id="5552")
    now = time.time()
    path = await _agent_db(deck)
    bobs = _session(path, [_ask("Bob's private telegram ask", now - DAY, source="telegram"),
                           _answer("Done.", now - DAY + 1)], name="tg-bob")
    _session(path, [_ask("Ivo on telegram", now - DAY, source="telegram"), _answer("Hi Ivo.", now - DAY + 1)],
             name="tg-ivo_k")
    _session(path, [_ask("Nobody knows whose", now - DAY, source="telegram"), _answer("?", now - DAY + 1)],
             name="tg-mystery")
    _app_state(path, telegram_user_sessions={"5552": bobs},
               telegram_approved_users={"5552": {"username": "bob"}, "6663": {"username": "ivo_k"}})

    await taskboard_backfill.backfill_user(deck.alice)
    await taskboard_backfill.backfill_user(deck.bob)
    # Ivo (no deck user) writes to Alice's agent: hers, as live; the unknown one: nobody's.
    assert set(_by_title(deck, deck.alice)) == {"Ivo on telegram"}
    (his,) = _tasks(deck, deck.bob)
    assert (his["title"], his["private"]) == ("Bob's private telegram ask", 1)
    assert _flags(his)["historical"] and _flags(his)["own_chat"]


async def test_a_history_turn_never_lands_on_a_live_request(deck):
    """Ana's open WhatsApp request today and her chats of two weeks ago: the
    old ones are tasks of their own, historical — today's stays as it was,
    in triage, never marked history."""
    now = time.time()
    waid = "385911234567"
    live, _ = deck.board.ingest(deck.alice, keys=["turn:" + "a" * 32], create={
        "title": "Live ask", "ask": "Live ask", "agent_slug": "helper", "agent_name": "Helper",
        "agent_owner_id": deck.alice, "kind": "request", "direction": "received", "channel": "whatsapp",
        "status": "open", "owes_next": "agent", "triage_state": "pending", "triage_after": now,
        "source": "agent_turns", "source_ref": "a" * 32},
        parties=[{"role": "requester", "kind": "waid", "address": waid, "display": "+" + waid, "verified": True}],
        event={"source": "agent_turns", "source_ref": "a" * 32 + ":open", "at": now - 60, "kind": "received",
               "channel": "whatsapp", "actor_kind": "party", "actor": "+" + waid, "summary": "Live ask"})
    path = await _agent_db(deck)
    meta = {"origin": {"kind": "whatsapp", "address": waid}, "whatsapp_waid": waid}
    _session(path, [_ask("Old ask from two weeks ago", now - 14 * DAY, source="whatsapp"),
                    _answer("Old answer.", now - 14 * DAY + 4),
                    _ask("And another one", now - 14 * DAY + 600, source="whatsapp")], meta=meta)
    await taskboard_backfill.backfill_user(deck.alice)
    tasks = _by_title(deck, deck.alice)
    assert set(tasks) == {"Live ask", "Old ask from two weeks ago", "And another one"}
    today = tasks["Live ask"]
    assert (today["status"], today["owes_next"], today["triage_state"]) == ("open", "agent", "pending")
    assert not _flags(today).get("historical")
    assert [e["source_ref"] for e in deck.board.events(deck.alice, live)] == ["a" * 32 + ":open"]
    assert all(_flags(tasks[t])["historical"] for t in ("Old ask from two weeks ago", "And another one"))


async def test_without_sign_in_every_agents_history_is_the_one_users(deck, monkeypatch):
    monkeypatch.setenv("FD_AUTH_ENABLED", "false")
    now = time.time()
    path = await _agent_db(deck)
    _session(path, [_ask("Local question", now - DAY), _answer("Local answer.", now - DAY + 1)])
    out = await taskboard_backfill.backfill_user("local")
    assert out["progress"]["tasks"] == 1
    (task,) = _tasks(deck, "local")
    assert task["status"] == "done" and _flags(task)["historical"] and _flags(task)["own_chat"]
    assert not _tasks(deck, deck.alice)


async def test_a_third_partys_message_waits_on_them_and_an_unsourced_one_is_passed_over(deck):
    now = time.time()
    path = await _agent_db(deck)
    meta = {"origin": {"kind": "whatsapp", "address": "385911234567"}, "whatsapp_waid": "385911234567"}
    _session(path, [
        _ask("Can you send me the price list?", now - 2 * DAY, source="whatsapp"),
        _answer("Here it is.", now - 2 * DAY + 4),
        # From before turns recorded their source: whose words? Left out.
        _msg("user", "Old message", now - 3 * DAY, origin="human"), _answer("Ok", now - 3 * DAY + 1),
    ], meta=meta)
    await taskboard_backfill.backfill_user(deck.alice)
    (task,) = _tasks(deck, deck.alice)
    assert (task["channel"], task["status"], task["owes_next"]) == ("whatsapp", "open", "them")
    assert _flags(task)["contains_instructions"] is True and "own_chat" not in _flags(task)
    (party,) = deck.board.parties_for(deck.alice, [task["id"]])[task["id"]]
    assert (party["kind"], party["address"], party["verified"]) == ("waid", "385911234567", True)
    assert deck.board.lane_of(deck.alice, task["id"]) == "waiting"


async def test_what_the_register_reported_is_left_to_it(deck):
    now = time.time()
    path = await _agent_db(deck)
    _session(path, [_ask("Before the register", now - 2 * DAY), _answer("a", now - 2 * DAY + 1),
                    _ask("After it", now - 6 * 3600), _answer("b", now - 6 * 3600 + 1)])
    deck.board.save_turn(secrets.token_hex(16), "open", "boot:1", {
        "agent_slug": "helper", "agent_owner_id": deck.alice, "user_id": deck.alice, "started_at": now - DAY})
    await taskboard_backfill.backfill_user(deck.alice)
    assert set(_by_title(deck, deck.alice)) == {"Before the register"}


async def test_the_sessions_db_is_only_ever_read(deck):
    now = time.time()
    path = await _agent_db(deck)
    sid = _session(path, [_ask("Read me", now - DAY), _answer("Read.", now - DAY + 1)])
    _todo(path, "Call Ana", source_session=sid)
    before = _fingerprint(path)
    time.sleep(0.01)
    await taskboard_backfill.backfill_user(deck.alice)
    run_sweep()
    assert _fingerprint(path) == before
    assert {t["kind"] for t in _tasks(deck, deck.alice)} == {"request", "todo"}


async def test_a_missing_or_busy_sessions_db_is_passed_over(deck, monkeypatch):
    now = time.time()
    agents = {"helper": {"owner": deck.alice, "name": "Helper", "ref": HELPER_REF},
              "notes": {"owner": deck.alice, "name": "Notes", "ref": "process:notes:0"},
              "third": {"owner": deck.alice, "name": "Third", "ref": "process:third:0"}}
    monkeypatch.setattr(taskboard_sources, "_agent_index", lambda: agents)
    busy = await _agent_db(deck, "helper")
    _session(busy, [_ask("Locked away", now - DAY), _answer("x", now - DAY + 1)])
    ok = await _agent_db(deck, "third")
    _session(ok, [_ask("Readable", now - DAY), _answer("y", now - DAY + 1)])
    holder = sqlite3.connect(busy, isolation_level=None)
    holder.execute("BEGIN EXCLUSIVE")
    try:
        out = await taskboard_backfill.backfill_user(deck.alice)
    finally:
        holder.execute("ROLLBACK")
        holder.close()
    assert out["state"] == "done"
    assert out["progress"] == {"agents_done": 3, "agents_total": 3, "turns": 1, "tasks": 1}
    assert "Helper couldn't be read" in out["message"]
    assert set(_by_title(deck, deck.alice)) == {"Readable"}
    # A busy file on the sweep is read on a later pass.
    holder = sqlite3.connect(busy, isolation_level=None)
    holder.execute("BEGIN EXCLUSIVE")
    _todo(ok, "Readable to-do")
    try:
        res = run_sweep()
    finally:
        holder.execute("ROLLBACK")
        holder.close()
    assert res["problems"] == []
    assert {t["title"] for t in _tasks(deck, deck.alice) if t["kind"] == "todo"} == {"Readable to-do"}


# ── GET /status and POST /backfill ────────────────────────────────────


@pytest.fixture()
async def api(deck, monkeypatch):
    app = FastAPI()
    app.include_router(taskboard.router)
    app.include_router(taskboard_backfill.router)
    users = {uid: {"id": uid, "email": f"{name}@x.co", "display_name": name, "role": "user"}
             for uid, name in ((deck.alice, "Alice"), (deck.bob, "Bob"), (deck.carol, "Carol"))}
    current = {"user": users[deck.alice]}
    app.dependency_overrides[auth.get_current_user] = lambda: current["user"]
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://deck") as client:
        yield SimpleNamespace(client=client, deck=deck, be=lambda uid: current.update(user=users[uid]))


async def _status(api) -> dict:
    r = await api.client.get("/fd/taskboard/status")
    assert r.status_code == 200, r.text
    return r.json()


async def test_the_first_look_starts_it_once_and_status_shows_its_progress(api, monkeypatch):
    deck = api.deck
    now = time.time()
    path = await _agent_db(deck)
    _session(path, [_ask("From last week", now - 7 * DAY), _answer("Ok.", now - 7 * DAY + 1)])
    monkeypatch.setattr(taskboard, "_loop_task", object())  # the board's background work runs
    gate = threading.Event()
    runs: list[str] = []
    read = taskboard_backfill._read_agent

    def held_read(store, agent, uid, now):
        runs.append(uid)
        gate.wait(5)
        return read(store, agent, uid, now)

    monkeypatch.setattr(taskboard_backfill, "_read_agent", held_read)
    out = await _status(api)
    assert out["backfill"]["state"] == "running" and out["backfill"]["done_at"] is None
    assert set(out["backfill"]) == {"state", "done_at", "progress", "message"}
    for _ in range(100):
        if runs:
            break
        await asyncio.sleep(0.01)
    running = (await _status(api))["backfill"]
    assert running["state"] == "running" and running["progress"]["agents_total"] == 1
    history = next(s for s in (await _status(api))["sources"] if s["key"] == "history")
    assert history["tracked"] is True and "Filling in" in history["note"]
    gate.set()
    await taskboard_backfill.wait(deck.alice)

    out = await _status(api)
    assert out["backfill"]["state"] == "done" and out["backfill"]["done_at"]
    assert out["backfill"]["progress"] == {"agents_done": 1, "agents_total": 1, "turns": 1, "tasks": 1}
    keys = [s["key"] for s in out["sources"]]
    assert keys.index("agent_todos") == keys.index("agent_chats") + 1
    assert all(set(s) == {"key", "label", "tracked", "on", "note"} for s in out["sources"])
    history = next(s for s in out["sources"] if s["key"] == "history")
    assert history["tracked"] and ": 1 task from" in history["note"]
    r = await api.client.post("/fd/taskboard/backfill")
    assert r.status_code == 200 and r.json()["backfill"]["state"] == "done"
    # Flight Deck restarted: the saved run still says done, and nothing runs again.
    monkeypatch.setattr(taskboard_backfill, "_JOBS", {})
    again = (await _status(api))["backfill"]
    assert again["state"] == "done" and again["progress"]["tasks"] == 1 and runs == [deck.alice]
    assert len(_tasks(deck, deck.alice)) == 1


async def test_looks_at_the_same_time_start_one_job(api, monkeypatch):
    """The board and its badge open together (or two tabs, or a POST with
    them): one job reads the history, and its counts are what it did."""
    deck = api.deck
    now = time.time()
    path = await _agent_db(deck)
    _session(path, [_ask("One", now - 3 * DAY), _answer("1", now - 3 * DAY + 1),
                    _ask("Two", now - 2 * DAY), _answer("2", now - 2 * DAY + 1)])
    monkeypatch.setattr(taskboard, "_loop_task", object())
    runs: list[str] = []
    read = taskboard_backfill._read_agent
    cls = type(deck.board)
    state = cls.history_state

    def counted(store, agent, uid, now):
        runs.append(uid)
        return read(store, agent, uid, now)

    def slow_state(self, uid):
        time.sleep(0.02)  # the read each look awaits before it would start one
        return state(self, uid)

    monkeypatch.setattr(taskboard_backfill, "_read_agent", counted)
    monkeypatch.setattr(cls, "history_state", slow_state)
    got = await asyncio.gather(api.client.get("/fd/taskboard/status"), api.client.get("/fd/taskboard/status"),
                               api.client.post("/fd/taskboard/backfill"))
    assert all(r.status_code == 200 for r in got)
    await taskboard_backfill.wait(deck.alice)
    assert runs == [deck.alice]
    saved = deck.board.history_state(deck.alice)
    assert saved["progress"]["tasks"] == 2 and "Filled in 2 tasks from 2 turns" in saved["message"]


async def test_without_the_background_work_only_a_post_starts_it_and_a_failure_waits_for_one(api, monkeypatch):
    deck = api.deck
    api.be(deck.bob)
    assert (await _status(api))["backfill"]["state"] == "idle"
    assert deck.bob not in taskboard_backfill._JOBS
    r = await api.client.post("/fd/taskboard/backfill")
    assert r.status_code == 200 and r.json()["backfill"]["state"] == "running"
    assert (await taskboard_backfill.wait(deck.bob))["state"] == "done"
    assert (await _status(api))["backfill"]["message"] == "None of your agents has chats to fill in."

    async def broken(uid):
        raise RuntimeError("registry unreadable")

    monkeypatch.setattr(taskboard, "_loop_task", object())
    monkeypatch.setattr(taskboard_backfill, "_caller_agents", broken)
    api.be(deck.carol)
    await _status(api)
    failed = await taskboard_backfill.wait(deck.carol)
    assert failed["state"] == "error" and "registry unreadable" in failed["message"]
    job = taskboard_backfill._JOBS[deck.carol]
    assert (await _status(api))["backfill"]["state"] == "error" and taskboard_backfill._JOBS[deck.carol] is job
    monkeypatch.setattr(taskboard_backfill, "_caller_agents", lambda uid: asyncio.sleep(0, result=[]))
    r = await api.client.post("/fd/taskboard/backfill")
    assert r.json()["backfill"]["state"] == "running"
    assert (await taskboard_backfill.wait(deck.carol))["state"] == "done"


# ── The agents' to-do lists ───────────────────────────────────────────


def _todo_task(deck, uid: str, todo_id: str, slug: str = "helper") -> dict:
    task_id = deck.board.task_for_key(uid, f"todo:{slug}:{todo_id}")
    assert task_id, f"no task for to-do {todo_id}"
    return deck.board.get_task(uid, task_id)


async def test_an_agents_to_do_list_is_tasks_moved_as_the_agent_moves_them(deck):
    path = await _agent_db(deck)
    mine = _todo(path, "Send the contract", priority="urgent")
    yours = _todo(path, "Sign the contract", responsible="human")
    busy = _todo(path, "Research venues", status="in_progress")
    done = _todo(path, "Book flights", status="done", completed=True)
    gone = _todo(path, "Order lunch", status="cancelled")
    archived = _todo(path, "Pay invoice", status="cancelled", completed=True)
    old = _todo(path, "Long finished", status="done", completed=True, at=time.time() - 40 * DAY)
    run_sweep()
    a = deck.alice
    t = _todo_task(deck, a, mine)
    assert (t["kind"], t["direction"], t["status"], t["owes_next"], t["urgency"]) == \
        ("todo", "internal", "open", "agent", 3)
    assert (t["agent_name"], t["triage_state"], t["source"]) == ("Helper", "skipped", "agent_todos")
    assert deck.board.lane_of(a, t["id"]) == "working"
    t = _todo_task(deck, a, yours)
    assert (t["status"], t["owes_next"]) == ("open", "me") and deck.board.lane_of(a, t["id"]) == "needs_me"
    assert _todo_task(deck, a, busy)["owes_next"] == "agent"
    t = _todo_task(deck, a, done)
    assert (t["status"], t["closed_by"], t["closed_by_party"], t["evidence"]) == ("done", "source", "Helper", "claimed")
    assert _todo_task(deck, a, gone)["status"] == "dismissed"
    assert _todo_task(deck, a, archived)["status"] == "done"  # archived after it was done
    assert deck.board.task_for_key(a, f"todo:helper:{old}") is None  # done before the last 30 days
    assert not _tasks(deck, deck.bob)

    size = _size(deck)
    run_sweep()
    assert _size(deck) == size

    _set_todo(path, mine, status="done", completed_at=_iso(time.time()))
    _set_todo(path, busy, responsible="human")
    with sqlite3.connect(path) as conn:
        conn.execute("DELETE FROM todo_items WHERE id = ?", (yours,))
    run_sweep()
    assert _todo_task(deck, a, mine)["status"] == "done"
    assert _todo_task(deck, a, busy)["owes_next"] == "me"
    t = _todo_task(deck, a, yours)
    assert t["status"] == "dismissed"
    assert deck.board.events(a, t["id"])[-1]["summary"] == "Taken off the agent's to-do list"
    # The user's Done holds over the agent opening it again.
    deck.board.correct(a, _todo_task(deck, a, done)["id"], "done", actor="Alice")
    _set_todo(path, done, status="pending")
    run_sweep()
    assert _todo_task(deck, a, done)["status"] == "done"


async def test_a_members_to_do_goes_to_the_member_only(deck):
    await deck.env.db.create_share("agent", HELPER_REF, deck.alice, deck.bob)
    path = await _agent_db(deck)
    bobs_session = _session(path, [], meta={"speaker_id": deck.bob, "speaker_name": "Bob"})
    carols_session = _session(path, [], meta={"speaker_id": deck.carol, "speaker_name": "Carol"})
    owners_session = _session(path, [])
    bobs = _todo(path, "Bob's errand", source_session=bobs_session)
    carols = _todo(path, "Carol's errand", source_session=carols_session)
    owners = _todo(path, "Alice's errand", source_session=owners_session)
    loose = _todo(path, "No session at all")
    # A one-user pass: Bob's board asked, and gets his own item only.
    run_sweep(user_id=deck.bob)
    (only,) = _tasks(deck, deck.bob)
    assert only["title"] == "Bob's errand" and only["private"] == 1
    assert not _tasks(deck, deck.alice)
    run_sweep()
    assert {t["title"] for t in _tasks(deck, deck.alice)} == {"Alice's errand", "No session at all"}
    assert {t["title"] for t in _tasks(deck, deck.bob)} == {"Bob's errand"}
    assert not _tasks(deck, deck.carol)  # not a member: her item is nobody's
    for todo_id, owner in ((bobs, deck.bob), (owners, deck.alice), (loose, deck.alice)):
        assert _todo_task(deck, owner, todo_id)
    assert deck.board.task_for_key(deck.alice, f"todo:helper:{carols}") is None
    assert deck.board.task_for_key(deck.alice, f"todo:helper:{bobs}") is None


def _member_session_record(path, rows: dict[str, str]) -> None:
    """The agent's saved_attribution.db as it notes each member's session when it is made."""
    with sqlite3.connect(path.with_name("saved_attribution.db")) as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS member_sessions (slug TEXT PRIMARY KEY, speaker_id TEXT NOT NULL, "
                     "name TEXT NOT NULL DEFAULT '', created_at TEXT NOT NULL)")
        conn.executemany("INSERT OR REPLACE INTO member_sessions VALUES (?, ?, '', ?)",
                         [(slug, uid, _iso(time.time())) for slug, uid in rows.items()])


async def test_a_members_to_do_never_moves_to_the_owner_once_its_session_is_gone(deck):
    """Alice deleted Bob's session: his item stays his — the agent's record
    of its members' sessions says whose it was —, an item whose session
    nobody can speak for is on no board, and one of Bob's linked Telegram
    goes to him as his turns there do."""
    await deck.env.db.create_share("agent", HELPER_REF, deck.alice, deck.bob)
    await deck.env.db.create_telegram_link(deck.bob, "", "", "code-b")
    await deck.env.db.update_telegram_link(deck.bob, status="approved", tg_user_id="5552")
    path = await _agent_db(deck)
    bobs_session = _session(path, [], meta={"speaker_id": deck.bob, "speaker_name": "Bob"})
    owners_session = _session(path, [])
    _member_session_record(path, {bobs_session: deck.bob})
    bobs = _todo(path, "Bob's private errand", source_session=bobs_session)
    owners = _todo(path, "Alice's errand", source_session=owners_session)
    tg_session = _session(path, [], name="tg-bob")
    _app_state(path, telegram_user_sessions={"5552": tg_session})
    by_telegram = _todo(path, "Bob's errand from Telegram", source_session=tg_session)
    run_sweep()
    assert {t["title"] for t in _tasks(deck, deck.bob)} == {"Bob's private errand", "Bob's errand from Telegram"}
    assert all(t["private"] == 1 for t in _tasks(deck, deck.bob))
    with sqlite3.connect(path) as conn:
        conn.execute("DELETE FROM sessions WHERE id IN (?, ?, ?)", (bobs_session, owners_session, tg_session))
    for todo_id in (bobs, owners, by_telegram):
        _set_todo(path, todo_id, status="done")
    run_sweep()
    assert {t["title"] for t in _tasks(deck, deck.alice)} == {"Alice's errand"}
    assert {t["status"] for t in _tasks(deck, deck.bob)} == {"done"}
    # The record unreadable: an item of a session gone since is nobody's.
    path.with_name("saved_attribution.db").write_bytes(b"not a database")
    lost = _session(path, [], meta={"speaker_id": deck.bob})
    _todo(path, "Whose was it?", source_session=lost)
    with sqlite3.connect(path) as conn:
        conn.execute("DELETE FROM sessions WHERE id = ?", (lost,))
    run_sweep()
    assert "Whose was it?" not in {t["title"] for t in _tasks(deck, deck.alice) + _tasks(deck, deck.bob)}


def test_the_sources_sheet_gets_the_history_and_the_to_do_lists():
    sources = [{"key": "agent_chats", "label": "x", "tracked": True, "on": True, "note": ""},
               {"key": "history", "label": "The last 30 days", "tracked": False, "on": False, "note": "Coming"}]
    idle = taskboard_backfill.view({"state": "idle"})
    out = taskboard_backfill.sources_view(sources, idle)
    assert [s["key"] for s in out] == ["agent_chats", "agent_todos", "history"]
    assert out[2]["tracked"] is False and "filled in once" in out[2]["note"]
    done = taskboard_backfill.view({"state": "done", "done_at": time.time(),
                                    "progress": {"agents_done": 2, "agents_total": 2, "turns": 9, "tasks": 5}})
    history = taskboard_backfill.sources_view(sources, done)[2]
    assert history["tracked"] and history["on"] and "5 tasks" in history["note"]
    assert sources[1]["note"] == "Coming"  # the constant itself is left as it was
