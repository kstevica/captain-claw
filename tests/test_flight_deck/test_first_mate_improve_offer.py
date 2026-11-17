"""Improvement runs proposed and shown (P6, piece 3).

Pinned here:

* "under target for two weeks": both of the last two weeks under the target,
  two signals or more in each and five in all;
* the weekly review proposes an improvement run for such a factory only for
  an admin (Captain Swarm is theirs), when it's ready and none is going;
* Approve starts it now, as the owner approving First Mate; a factory that
  can't be improved fails with why;
* the drawer's list: the factory's runs, whether each became a draft, and
  whether one can start (and whether improvement runs exist here at all).
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest

from captain_claw.config import AutonomousWorkConfig
from captain_claw.flight_deck import auth as fd_auth
from captain_claw.flight_deck import autonomy, bat_loop, events, line
from captain_claw.flight_deck import captain_swarm_loop as loop
from captain_claw.flight_deck import captain_swarm_routes as csr
from captain_claw.flight_deck import course as c
from captain_claw.flight_deck import first_mate_cases as cases
from captain_claw.flight_deck import first_mate_review as review
from captain_claw.flight_deck import first_mate_routes as routes
from captain_claw.flight_deck import first_mate_store as fm
from captain_claw.flight_deck.auth import set_auth_db
from captain_claw.flight_deck.captain_swarm_store import CaptainSwarmStore
from captain_claw.flight_deck.db import FlightDeckDB

ADMIN, MEMBER = "user-bob", "user-carol"
CHARTER = ("Go through my inbox for client mail about invoices and draft a short reply to each. Never send "
           "anything. Finish with who wrote and whether a draft waits.")


@pytest.fixture(autouse=True)
def _no_register(monkeypatch):
    from captain_claw.flight_deck import tool_audit_routes
    monkeypatch.setattr(tool_audit_routes, "_data_dir", None)
    monkeypatch.setattr(tool_audit_routes, "_store", None)


@pytest.fixture
async def db(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("FD_AUTH_ENABLED", "true")
    prev = fd_auth._db
    d = FlightDeckDB(tmp_path / "fd.db")
    await d.init()
    set_auth_db(d)
    for uid, role in ((ADMIN, "admin"), (MEMBER, "user")):
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
    monkeypatch.setattr(events, "_STORE", events.EventsStore(tmp_path / "events.db"))
    monkeypatch.setattr(c, "_STORE", c.CourseStore(tmp_path / "course.db"))
    s = fm.FirstMateStore(tmp_path / "first_mate.db")
    monkeypatch.setattr(fm, "_STORE", s)
    s.set_settings({"enabled": True, "admins_only": False})
    a = autonomy.AutonomyStore(tmp_path / "autonomy.db")
    monkeypatch.setattr(autonomy, "_STORE", a)
    monkeypatch.setattr(autonomy, "global_defaults", lambda: AutonomousWorkConfig().model_dump())

    async def fake_fan_out(user_id, item, to_phones=True, rung=False):
        return {}
    monkeypatch.setattr(line, "fan_out", fake_fan_out)
    return s


@pytest.fixture
async def cs(tmp_path, monkeypatch):
    s = CaptainSwarmStore(tmp_path / "cs.db")
    loop.set_store(s)
    monkeypatch.setattr(loop, "sealed_dir", lambda rid: tmp_path / "sealed" / rid)

    async def can_run(owner, objective=None, user=None):
        return None

    async def kick():
        return None

    monkeypatch.setattr(csr, "_can_run", can_run)
    monkeypatch.setattr(loop, "kick", kick)
    monkeypatch.setattr(bat_loop, "_STORE", object())
    yield s
    loop.set_store(None)
    await s.close()


def _live(store, owner=ADMIN, ready=True):
    f = store.create(owner, title="Inbox replies", charter=CHARTER, checks={"satisfaction_target": 0.8})
    f = store.approve(f["id"], 1, f["draft"]["sha"], owner)
    for n in range(3 if ready else 1):
        cases.add_case(store, f["id"], title=f"Question {n}", input=f"From: Client {n}\nCheck invoice {n}?",
                       expect=["drafts a reply"])
    return f


def _signals(store, f, *, last_week, week_before):
    now = fm._utcnow()
    for days, values in ((10, week_before), (2, last_week)):
        for v in values:
            sid = store.add_signal(f["id"], "sent_edited", value=v)
            with store._tx() as conn:
                conn.execute("UPDATE fm_signals SET at = ? WHERE id = ?", (fm._iso(now - timedelta(days=days)), sid))


def _report(f):
    return {"factories": [{"id": f["id"], "title": f["title"], "health": "amber", "health_why": "", "tokens": 0,
                           "share": 25_000_000}], "end": fm._iso(fm._utcnow())}


# ── the rule ───────────────────────────────────────────────────────────

def test_under_target_for_two_weeks(store):
    f = _live(store)
    now = fm._utcnow()
    _signals(store, f, last_week=[0.5, 0.6], week_before=[0.4, 0.7, 0.5])
    assert review.below_target(store, store.get(f["id"]), now) == (0.55, 0.8)
    g = _live(store)
    _signals(store, g, last_week=[0.9, 1.0], week_before=[0.4, 0.5, 0.5])  # last week was fine
    assert review.below_target(store, store.get(g["id"]), now) is None
    h = _live(store)
    _signals(store, h, last_week=[0.5], week_before=[0.4, 0.5, 0.5])  # too little last week to say
    assert review.below_target(store, store.get(h["id"]), now) is None


# ── the proposal ───────────────────────────────────────────────────────

async def test_the_review_proposes_an_improvement_run(store, cs):
    f = _live(store)
    _signals(store, f, last_week=[0.5, 0.6], week_before=[0.4, 0.7, 0.5])
    [row] = await review.weekly(store, ADMIN, _report(f))
    assert row["payload"] == {"factory_id": f["id"], "op": "improve"} and row["title"] == "Improve Inbox replies"
    assert "under its target for two weeks (55% last week, target 80%)" in row["rationale"]
    got = await review.dispatch(ADMIN, row, approved_by_human=True)
    assert got["note"].startswith("Started an improvement run of Inbox replies")
    [run] = await cs.list_runs(ADMIN)
    assert run["objective"]["kind"] == "factory" and run["started_by"] == "the owner, approving First Mate"
    assert autonomy.get_store().get_action(row["id"])["status"] == "done"
    assert await review.weekly(store, ADMIN, _report(f)) == []  # one is going now


async def test_only_for_admins_and_ready_factories(store, cs):
    f = _live(store, owner=MEMBER)
    _signals(store, f, last_week=[0.5, 0.6], week_before=[0.4, 0.7, 0.5])
    assert await review.weekly(store, MEMBER, _report(f)) == []  # Captain Swarm is for admins
    g = _live(store, ready=False)
    _signals(store, g, last_week=[0.5, 0.6], week_before=[0.4, 0.7, 0.5])
    assert await review.weekly(store, ADMIN, _report(g)) == []  # too few cases to improve on
    row = autonomy.get_store().add_action(ADMIN, kind="factory", title="Improve it", status="awaiting_approval",
                                          payload={"op": "improve", "factory_id": g["id"]})
    got = await review.dispatch(ADMIN, row, approved_by_human=True)
    assert "at least 3 cases" in got["note"] and autonomy.get_store().get_action(row["id"])["status"] == "failed"


# ── the drawer's list ──────────────────────────────────────────────────

async def test_the_improvements_list(store, cs):
    f = _live(store)
    admin = {"id": ADMIN, "role": "admin"}
    got = await routes.list_improvements(f["id"], user=admin)
    assert got == {"runs": [], "can": {"ok": True, "why": ""}, "available": True}
    started = await routes.improve_factory(f["id"], {}, user=admin)
    got = await routes.list_improvements(f["id"], user=admin)
    assert [r["id"] for r in got["runs"]] == [started["run_id"]] and got["runs"][0]["going"]
    assert got["can"] == {"ok": False, "why": "An improvement run of this factory is going."}
    await cs.update_run(started["run_id"], status="done")
    store.note(f["id"], "first_mate", "improvement_drafted", {"run": started["run_id"], "version": 2})
    got = await routes.list_improvements(f["id"], user=admin)
    assert got["runs"][0]["drafted"] and not got["runs"][0]["going"] and got["can"]["ok"]
    g = _live(store, owner=MEMBER)
    member = await routes.list_improvements(g["id"], user={"id": MEMBER, "role": "user"})
    assert member["available"] is False and member["can"]["ok"] is False and member["runs"] == []
