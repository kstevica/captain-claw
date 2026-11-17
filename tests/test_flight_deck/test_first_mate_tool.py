"""The `first_mate` tool and its deck route (P5, piece 1).

Pinned here:

* the route resolves the agent's owner from the deck's records and follows
  First Mate's switch and who may use it;
* list, status, run, pause, resume and the week's report, each on the owner's
  own factories, found by name (exact, or the only one it fits);
* it never approves, carries out or moves trust;
* the tool refuses inside any run, wants a factory where one is needed,
  and is refused in rehearsals like the other run starters;
* no remote caller reaches the route.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from captain_claw.flight_deck import auth as fd_auth
from captain_claw.flight_deck import bat_loop, events, line
from captain_claw.flight_deck import course as c
from captain_claw.flight_deck import first_mate_routes as routes
from captain_claw.flight_deck import first_mate_store as fm
from captain_claw.flight_deck.auth import set_auth_db
from captain_claw.flight_deck.db import FlightDeckDB
from captain_claw.tools.first_mate import FirstMateTool

BOB, CAROL = "user-bob", "user-carol"


class FakeBat:
    def __init__(self) -> None:
        self.runs: dict[str, dict] = {}

    async def create_run(self, *, run_id, owner_id, title, task, config=None, origin=None,
                         source_host="localhost", source_port=0, llm_usd_cap=0.0, status="planning"):
        self.runs[run_id] = {"id": run_id, "owner_id": owner_id, "title": title, "task": task,
                             "config": config or {}, "status": status, "cumulative_tokens": 0,
                             "cumulative_usd": 0.0, "truth": "", "stopped_reason": ""}
        return run_id

    async def append_event(self, run_id, stage, message="", **data):
        return 1

    async def get_run(self, run_id):
        r = self.runs.get(run_id)
        return dict(r) if r else None

    async def list_runs(self, owner_id, limit=50):
        return [dict(r) for r in self.runs.values() if r["owner_id"] == owner_id][:limit]

    async def set_status(self, run_id, status, **kw):
        self.runs[run_id].update(status=status, **kw)


@pytest.fixture(autouse=True)
def _no_register(monkeypatch):
    from captain_claw.flight_deck import tool_audit_routes
    monkeypatch.setattr(tool_audit_routes, "_data_dir", None)
    monkeypatch.setattr(tool_audit_routes, "_store", None)


@pytest.fixture
async def db(tmp_path: Path):
    prev = fd_auth._db
    d = FlightDeckDB(tmp_path / "fd.db")
    await d.init()
    set_auth_db(d)
    for uid, role in ((BOB, "user"), (CAROL, "user")):
        await d._db.execute(
            "INSERT INTO users (id, email, password_hash, display_name, role, created_at, updated_at)"
            " VALUES (?, ?, 'h', ?, ?, '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z')",
            (uid, f"{uid}@x.co", uid, role))
    await d._db.commit()
    try:
        yield d
    finally:
        await d.close()
        fd_auth._db = prev


@pytest.fixture
def store(tmp_path, monkeypatch, db):
    monkeypatch.setenv("FD_AUTH_ENABLED", "true")
    monkeypatch.setattr(events, "_STORE", events.EventsStore(tmp_path / "events.db"))
    monkeypatch.setattr(c, "_STORE", c.CourseStore(tmp_path / "course.db"))
    s = fm.FirstMateStore(tmp_path / "first_mate.db")
    monkeypatch.setattr(fm, "_STORE", s)
    s.set_settings({"enabled": True, "admins_only": False})
    return s


@pytest.fixture
def bat(monkeypatch):
    fake = FakeBat()

    async def kick():
        return 0

    monkeypatch.setattr(bat_loop, "_STORE", fake)
    monkeypatch.setattr(bat_loop, "kick", kick)
    monkeypatch.setattr(bat_loop, "_STOP_CHILDREN", None)
    return fake


@pytest.fixture
def owner_is(monkeypatch):
    who = {"owner": BOB}
    from captain_claw.flight_deck import basna_routes
    monkeypatch.setattr(basna_routes, "_resolve_owner", lambda body: who["owner"])
    return who


@pytest.fixture(autouse=True)
def quiet_line(monkeypatch):
    async def fake_fan_out(user_id, item, to_phones=True, rung=False):
        return {}
    monkeypatch.setattr(line, "fan_out", fake_fan_out)


def _live(store, owner=BOB, **kw):
    kw.setdefault("title", "Inbox replies")
    kw.setdefault("charter", "Reply to clients who wait for an answer.")
    kw.setdefault("grants", {"email": True, "sends_per_run": 3})
    f = store.create(owner, **kw)
    return store.approve(f["id"], 1, f["draft"]["sha"], owner)


async def _ask(action, **kw):
    return await routes.agent_tool(routes.AgentToolReq(action=action, **kw))


# ── the route ──────────────────────────────────────────────────────────

async def test_list_and_status(store, bat, owner_is):
    assert (await _ask("list"))["text"].startswith("You have no factories yet")
    f = _live(store)
    _live(store, title="Calendar tidy", grants={"organize": True})
    _live(store, owner=CAROL, title="Carol's digest")
    got = await _ask("list")
    assert got["ok"] and sorted(x["title"] for x in got["factories"]) == ["Calendar tidy", "Inbox replies"]
    assert "You have 2 factories. Running on their own: off" in got["text"]
    assert "- Inbox replies: running · asks first · Too new to say" in got["text"]
    status = await _ask("status", factory="inbox")
    assert status["ok"] and status["id"] == f["id"]
    assert status["text"].startswith("Inbox replies (version 1, running, asks first)")
    assert "Charter: Reply to clients who wait for an answer." in status["text"]
    assert "Starts: only when you press Run now" in status["text"]
    assert "Health: Too new to say (not enough runs yet to say)." in status["text"]


async def test_finding_a_factory_by_name(store, owner_is):
    _live(store, title="Inbox replies")
    _live(store, title="Inbox digest")
    _live(store, owner=CAROL, title="Weekly invoices")
    assert (await _ask("status", factory="INBOX REPLIES"))["ok"]
    got = await _ask("status", factory="inbox")
    assert not got["ok"] and "more than one factory fits" in got["why"] and "Inbox digest" in got["why"]
    got = await _ask("status", factory="invoices")  # Carol's: never Bob's
    assert not got["ok"] and 'no factory called "invoices"' in got["why"]
    assert not (await _ask("status"))["ok"]


async def test_run_pause_resume(store, bat, owner_is, db):
    f = _live(store)
    got = await _ask("run", factory="inbox replies", note="Only the invoice threads today.")
    assert got["ok"] and "a rehearsal" in got["text"]
    run = store.get_run(got["run_id"])
    assert run["kind"] == "rehearsal" and run["trigger"] == "manual"
    assert "Only the invoice threads today." in bat.runs[run["child_id"]]["task"]
    again = await _ask("run", factory="inbox replies")
    assert not again["ok"] and "already has a run going" in again["why"]
    paused = await _ask("pause", factory="inbox replies", note="on holiday")
    assert paused["ok"] and store.get(f["id"])["state"] == "paused"
    assert store.get(f["id"])["state_reason"] == "on holiday"
    assert (await _ask("resume", factory="inbox replies"))["ok"] and store.get(f["id"])["state"] == "running"
    acts = [(e["event"], e["actor"]) for e in store.events(f["id"]) if e["event"] == "state"]
    assert acts == [("state", "owner"), ("state", "owner")]


async def test_the_report(store, owner_is):
    _live(store)
    got = await _ask("report", week="this")
    assert got["ok"] and got["text"].startswith("First Mate's week: ") and "**Inbox replies**" in got["text"]


async def test_who_may_use_it(store, owner_is, db):
    from fastapi import HTTPException
    store.set_settings({"admins_only": True})
    with pytest.raises(HTTPException) as e:
        await _ask("list")
    assert e.value.status_code == 403 and "admins only" in e.value.detail
    store.set_settings({"admins_only": False, "enabled": False})
    with pytest.raises(HTTPException) as e:
        await _ask("list")
    assert "off on this deck" in e.value.detail
    store.set_settings({"enabled": True})
    with pytest.raises(HTTPException) as e:
        await _ask("approve", factory="x")
    assert e.value.status_code == 400  # nothing that approves or moves trust
    owner_is["owner"] = "user-nobody"
    with pytest.raises(HTTPException) as e:
        await _ask("list")
    assert e.value.status_code == 403


async def test_no_remote_caller_reaches_the_route(monkeypatch):
    from captain_claw.flight_deck import server
    monkeypatch.setenv("FD_ALLOWED_HOSTS", "fd.test")
    monkeypatch.delenv("FD_LOCKDOWN", raising=False)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app, client=("203.0.113.9", 4001)),
                                 base_url="http://fd.test") as cl:
        r = await cl.post("/fd/first-mate/agent/tool", json={"action": "list"})
    assert r.status_code == 403


# ── the tool ───────────────────────────────────────────────────────────

async def test_the_tool_refuses_inside_a_run_and_without_a_factory(monkeypatch):
    tool = FirstMateTool()
    sent: list[dict] = []

    async def post(fd_url, payload):
        sent.append(payload)
        return {"ok": True, "text": "You have 1 factory."}

    monkeypatch.setattr(tool, "_get_fd_url", lambda **kw: "http://127.0.0.1:25080")
    monkeypatch.setattr(tool, "_post", post)
    monkeypatch.setenv("CLAW_BAT_WORKER", "1")
    got = await tool.execute(action="list")
    assert not got.success and "not inside a run" in got.error and sent == []
    monkeypatch.delenv("CLAW_BAT_WORKER")
    assert not (await tool.execute(action="run")).success
    assert not (await tool.execute(action="approve", factory="x")).success
    got = await tool.execute(action="list")
    assert got.success and got.content == "You have 1 factory."
    assert sent == [{"action": "list", "factory": "", "note": "", "week": "last", "words": ""}]
    assert not (await tool.execute(action="draft", words="hm")).success  # the owner's words, please

    async def no(fd_url, payload):
        return {"ok": False, "why": 'no factory called "x"'}

    monkeypatch.setattr(tool, "_post", no)
    got = await tool.execute(action="status", factory="x")
    assert not got.success and got.error == 'no factory called "x"'


def test_a_rehearsal_refuses_the_tool():
    from captain_claw import rehearsal_turn
    verdict, what = rehearsal_turn.classify("first_mate", {"action": "run", "factory": "Inbox"})
    assert verdict == "refuse"
    assert rehearsal_turn._why(what) == "a rehearsal starts no other runs."
