"""First Mate's proposals, accepted from the Line (P5, piece 4).

Pinned here:

* the weekly review proposes pausing a red factory, a bigger share for one
  that used its whole share, and a factory for a stalled goal nothing serves;
  at most three, never a second beside one that waits, nothing when First
  Mate isn't on for the owner or they have no approved factory;
* a proposal is a ledger row of kind ``factory`` waiting on the Line; it never
  fires on its own, whatever the autonomy dials say;
* Approve does its one op (pause, share, draft for the goal), on the owner's
  own factory or goal only; Reject leaves everything as it was;
* the report lists what First Mate proposes;
* the Arbiter may propose a factory for one of the goals it was shown, and
  drops one without a goal or for a goal a factory already serves.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import pytest

from captain_claw.config import AutonomousWorkConfig
from captain_claw.flight_deck import (
    action_catalog,
    arbiter,
    autonomy,
    events,
    fd_dispatch,
    line,
    rate_limiter,
    server,
)
from captain_claw.flight_deck import auth as fd_auth
from captain_claw.flight_deck import course as c
from captain_claw.flight_deck import first_mate_drafting as drafting
from captain_claw.flight_deck import first_mate_report as report
from captain_claw.flight_deck import first_mate_review as review
from captain_claw.flight_deck import first_mate_store as fm
from captain_claw.flight_deck.auth import create_access_token, set_auth_db
from captain_claw.flight_deck.db import FlightDeckDB

BOB, CAROL = "user-bob", "user-carol"
ANSWER = {"title": "Payment chaser", "charter": "Each Monday, list invoices unpaid after 30 days and draft a "
          "friendly reminder for each. Never send anything.", "schedule": "weekly mon 08:00",
          "done_when": ["every overdue invoice has a draft"], "cases": []}


@pytest.fixture(autouse=True)
def _no_register(monkeypatch):
    from captain_claw.flight_deck import tool_audit_routes
    monkeypatch.setattr(tool_audit_routes, "_data_dir", None)
    monkeypatch.setattr(tool_audit_routes, "_store", None)


@pytest.fixture
def ledger(tmp_path, monkeypatch):
    s = autonomy.AutonomyStore(tmp_path / "autonomy.db")
    monkeypatch.setattr(autonomy, "_STORE", s)
    monkeypatch.setattr(autonomy, "global_defaults", lambda: AutonomousWorkConfig().model_dump())
    return s


@pytest.fixture
def stores(tmp_path, monkeypatch, ledger):
    monkeypatch.setattr(events, "_STORE", events.EventsStore(tmp_path / "events.db"))
    monkeypatch.setattr(c, "_STORE", c.CourseStore(tmp_path / "course.db"))
    s = fm.FirstMateStore(tmp_path / "first_mate.db")
    monkeypatch.setattr(fm, "_STORE", s)
    s.set_settings({"enabled": True, "admins_only": False})
    return s


@pytest.fixture
async def db(tmp_path: Path):
    prev = fd_auth._db
    d = FlightDeckDB(tmp_path / "fd.db")
    await d.init()
    set_auth_db(d)
    for uid in (BOB, CAROL):
        await d._db.execute(
            "INSERT INTO users (id, email, password_hash, display_name, role, created_at, updated_at)"
            " VALUES (?, ?, 'h', ?, 'user', '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z')",
            (uid, f"{uid}@x.co", uid))
    await d._db.commit()
    try:
        yield d
    finally:
        await d.close()
        fd_auth._db = prev


@pytest.fixture
def store(stores, db, monkeypatch):
    async def fake_fan_out(user_id, item, to_phones=True, rung=False):
        return {}
    monkeypatch.setattr(line, "fan_out", fake_fan_out)
    return stores


@pytest.fixture
def stalled(monkeypatch):
    """Which of Bob's goals Course calls stalled."""
    found: list[tuple[dict, int]] = []
    from captain_claw.flight_deck import course_progress
    monkeypatch.setattr(course_progress, "stalled_goals", lambda uid, **kw: list(found) if uid == BOB else [])
    return found


def _live(store, owner=BOB, **kw):
    kw.setdefault("title", "Inbox replies")
    kw.setdefault("charter", "Reply to clients who wait for an answer.")
    f = store.create(owner, **kw)
    return store.approve(f["id"], 1, f["draft"]["sha"], owner)


def _line(f, **kw):
    out = {"id": f["id"], "title": f["title"], "health": "green", "health_why": "", "tokens": 0,
           "share": 12_500_000}
    return {**out, **kw}


async def _items(db, who=BOB):
    await asyncio.sleep(0)
    return await db.list_line_items(who, limit=50)


# ── the weekly review ──────────────────────────────────────────────────

async def test_the_weekly_review(store, db, stalled):
    red = _live(store, title="Morning briefing")
    hungry = _live(store, title="Inbox replies")
    calm = _live(store, title="Calendar tidy")
    goal = c.get_store().create(scope="user", owner_id=BOB, horizon="short", title="Get paid on time", success="x")
    served = c.get_store().create(scope="user", owner_id=BOB, horizon="short", title="Calmer weeks", success="x")
    store.update_factory(calm["id"], "owner", goal_id=served["id"])
    stalled += [(goal, 9), (served, 12)]
    rep = {"factories": [_line(red, health="red", health_why="3 failed runs in a row."),
                         _line(hungry, tokens=12_400_000), _line(calm)]}
    made = await review.weekly(store, BOB, rep)
    assert [(r["payload"]["op"], r["title"]) for r in made] == [
        ("pause", "Pause Morning briefing"), ("share", "Give Inbox replies a bigger share"),
        ("draft_for_goal", "Start a factory for: Get paid on time")]
    assert all(r["kind"] == "factory" and r["status"] == "awaiting_approval" for r in made)
    assert made[1]["payload"] == {"factory_id": hungry["id"], "share": 2, "op": "share"}
    assert made[2]["goal_id"] == goal["id"] and "9 days" in made[2]["rationale"]
    items = [i for i in await _items(db) if i["ref_type"] == "autonomy_action"]
    assert len(items) == 3 and all(i["needs_you"] for i in items)
    assert await review.weekly(store, BOB, rep) == []  # the same ones already wait


async def test_the_review_holds_back(store, db, stalled, monkeypatch):
    assert await review.weekly(store, BOB, {"factories": []}) == []  # no approved factory: nothing
    f = _live(store)
    rep = {"factories": [_line(f, health="red", health_why="x")]}
    store.set_settings({"admins_only": True})
    assert await review.weekly(store, BOB, rep) == []  # First Mate isn't on for Bob
    store.set_settings({"admins_only": False})
    monkeypatch.setattr(review, "WEEKLY_MAX", 1)
    many = {"factories": [_line(_live(store, title=f"F{n}"), health="red", health_why="x") for n in range(3)]}
    assert len(await review.weekly(store, BOB, many)) == 1


# ── never on its own; Approve does its one op ──────────────────────────

def test_a_factory_proposal_never_fires_on_its_own():
    for cfg in ({"allow_auto_dispatch": True, "autonomy_level": "act"},
                {"allow_auto_dispatch": True, "autonomy_level": "act_low_risk", "low_risk_kinds": ["factory"]}):
        assert fd_dispatch.should_auto_dispatch(cfg, {"kind": "factory", "risk": "low"}) is False


async def test_approve_does_its_one_op(store, db, stalled, monkeypatch):
    red, hungry = _live(store, title="Morning briefing"), _live(store, title="Inbox replies")
    goal = c.get_store().create(scope="user", owner_id=BOB, horizon="short", title="Get paid on time", success="x")
    stalled.append((goal, 9))
    made = await review.weekly(store, BOB, {"factories": [_line(red, health="red", health_why="x"),
                                                         _line(hungry, tokens=12_500_000)]})
    pause, share, for_goal = made
    assert (await review.dispatch(BOB, pause, approved_by_human=False))["ok"] is False  # nobody approved
    assert store.get(red["id"])["state"] == "running"
    got = await review.dispatch(BOB, pause, approved_by_human=True)
    assert got["note"] == "Paused Morning briefing." and store.get(red["id"])["state"] == "paused"
    assert autonomy.get_store().get_action(pause["id"])["status"] == "done"
    await review.dispatch(BOB, share, approved_by_human=True)
    assert store.get(hungry["id"])["share"] == 2
    asked: list[str] = []

    async def model(owner, system, user, max_tokens=0):
        asked.append(user)
        return json.dumps(ANSWER)

    monkeypatch.setattr(drafting, "_ask_model", model)
    got = await review.dispatch(BOB, for_goal, approved_by_human=True)
    assert got["note"].startswith("Drafted Payment chaser.") and "Get paid on time" in asked[0]
    drafted = [f for f in store.list_factories(BOB) if f["goal_id"] == goal["id"]]
    assert len(drafted) == 1 and drafted[0]["state"] == "awaiting_charter"  # its charter still waits
    again = await review.dispatch(BOB, for_goal, approved_by_human=True)
    assert again["note"] == "a factory already serves that goal"
    assert autonomy.get_store().get_action(for_goal["id"])["status"] == "failed"


async def test_only_the_owners_own(store, db):
    theirs = _live(store, owner=CAROL, title="Carol's digest")
    row = autonomy.get_store().add_action(BOB, kind="factory", title="Pause it", status="awaiting_approval",
                                          payload={"op": "pause", "factory_id": theirs["id"]})
    got = await review.dispatch(BOB, row, approved_by_human=True)
    assert got["note"] == "that factory is gone" and store.get(theirs["id"])["state"] == "running"


# ── the report and the Line's Approve ──────────────────────────────────

async def test_the_report_lists_them(store, db, stalled, monkeypatch):
    from datetime import UTC, datetime, timedelta
    f = _live(store)
    with store._tx() as conn:
        conn.execute("UPDATE fm_factories SET created_at = ? WHERE id = ?", ("2026-10-01T00:00:00+00:00", f["id"]))
    r = store.start_run(f["id"], trigger="manual", now=datetime(2026, 11, 11, 9, tzinfo=UTC))
    store.mark_running(r["id"], "bat_x")
    store.finish_run(r["id"], "done", tokens=100)
    goal = c.get_store().create(scope="user", owner_id=BOB, horizon="short", title="Get paid on time", success="x")
    stalled.append((goal, 9))
    assert await report.maybe_send(store, BOB, datetime(2026, 11, 16, 8, tzinfo=UTC) + timedelta(minutes=1))
    [rep] = [i for i in await _items(db) if i["ref_type"] == "first_mate_report"]
    assert "**First Mate proposes** (approve or reject each on the Line)" in rep["body"]
    assert "- Start a factory for: Get paid on time" in rep["body"]


def _client():
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app, client=("203.0.113.9", 40001)),
                             base_url="http://fd.test")


def _hdr(who):
    return {"Authorization": f"Bearer {create_access_token(who)}"}


async def test_approving_from_the_line(store, db, stalled, monkeypatch):
    monkeypatch.setattr(server, "AUTH_ENABLED", True)
    monkeypatch.setenv("FD_AUTH_ENABLED", "true")
    monkeypatch.setenv("FD_ALLOWED_HOSTS", "fd.test")
    monkeypatch.delenv("FD_LOCKDOWN", raising=False)
    monkeypatch.setattr(rate_limiter, "_api_limiter", rate_limiter._SlidingWindow())
    red = _live(store, title="Morning briefing")
    keep = _live(store, title="Inbox replies")
    pause, other = await review.weekly(store, BOB, {"factories": [_line(red, health="red", health_why="x"),
                                                                   _line(keep, health="red", health_why="x")]})
    async with _client() as cl:
        r = await cl.post(f"/fd/autonomy/actions/{pause['id']}/approve", headers=_hdr(BOB))
        assert r.status_code == 200, r.text
        r = await cl.post(f"/fd/autonomy/actions/{other['id']}/reject", headers=_hdr(BOB))
        assert r.status_code == 200, r.text
        r = await cl.post(f"/fd/autonomy/actions/{pause['id']}/approve", headers=_hdr(CAROL))
        assert r.status_code in (403, 404)
    assert store.get(red["id"])["state"] == "paused" and store.get(keep["id"])["state"] == "running"
    assert autonomy.get_store().get_action(pause["id"])["status"] == "done"
    assert autonomy.get_store().get_action(other["id"])["status"] == "rejected"


# ── the Arbiter ────────────────────────────────────────────────────────

UID = BOB


@pytest.fixture
def ranker(monkeypatch, ledger):
    """The ranking model proposes whatever ``ranker.reply`` holds."""
    import captain_claw.games.remote_provider as rp

    class _Resp:
        def __init__(self, content):
            self.content = content

    class _Provider:
        reply: list = []

        def __init__(self, **kw):
            pass

        async def complete(self, **kw):
            return _Resp(json.dumps(_Provider.reply))

    monkeypatch.setattr(rp, "RemoteLLMProvider", _Provider)

    def _no_events():
        raise RuntimeError("no events store in this test")

    async def _no_runs(uid):
        return []

    async def on(uid):
        return True

    monkeypatch.setattr(events, "get_store", _no_events)
    monkeypatch.setattr(arbiter, "_gather_active_runs", _no_runs)
    monkeypatch.setattr(action_catalog, "list_catalog", lambda **kw: [])
    monkeypatch.setattr(review, "available", on)
    ledger.set_overrides(UID, {"enabled": True, "autonomy_level": "act", "allow_auto_dispatch": True,
                               "quiet_hours_start": 0, "quiet_hours_end": 0})
    return _Provider


def _pass():
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(arbiter.maybe_run_arbiter(
            UID, {"intentions": ["Keep invoices moving"]}, {"host": "localhost", "port": 1, "auth": "", "name": "a"},
            [], trigger="pulse"))
    finally:
        loop.close()


def test_the_arbiter_proposes_a_factory_for_a_goal(tmp_path, monkeypatch, ranker):
    monkeypatch.setattr(c, "_STORE", c.CourseStore(tmp_path / "course.db"))
    s = fm.FirstMateStore(tmp_path / "first_mate.db")
    monkeypatch.setattr(fm, "_STORE", s)
    goal = c.get_store().create(scope="user", owner_id=UID, horizon="short", title="Get paid on time", success="x")
    ranker.reply = [{"kind": "factory", "title": "Start a standing invoice chaser", "rationale": "it recurs",
                     "risk": "normal", "domain": "ops", "score": 0.8, "goal_id": f"G:{goal['id']}"}]
    res = _pass()
    assert res.get("proposed") == 1 and res.get("dispatched") is False, res
    row = autonomy.get_store().get_action(res["action_id"])
    assert row["kind"] == "factory" and row["status"] == "awaiting_approval"
    assert row["payload"] == {"op": "draft_for_goal", "goal_id": goal["id"]} and row["goal_id"] == goal["id"]


def test_the_arbiter_drops_a_factory_it_cant_place(tmp_path, monkeypatch, ranker):
    monkeypatch.setattr(c, "_STORE", c.CourseStore(tmp_path / "course.db"))
    s = fm.FirstMateStore(tmp_path / "first_mate.db")
    monkeypatch.setattr(fm, "_STORE", s)
    goal = c.get_store().create(scope="user", owner_id=UID, horizon="short", title="Get paid on time", success="x")
    ranker.reply = [{"kind": "factory", "title": "A standing job, for no goal", "rationale": "r",
                     "risk": "normal", "domain": "ops", "score": 0.8}]
    assert _pass().get("proposed") == 0
    f = s.create(UID, title="Payment chaser", charter="Chase overdue invoices.", goal_id=goal["id"])
    assert f["goal_id"] == goal["id"]
    ranker.reply = [{"kind": "factory", "title": "Another chaser", "rationale": "r", "risk": "normal",
                     "domain": "ops", "score": 0.8, "goal_id": f"G:{goal['id']}"}]
    assert _pass().get("proposed") == 0  # a factory already serves it
