"""An improvement run's result reaches the factory (P6, piece 2).

Pinned here:

* a run that kept a better charter ends with it as the factory's new draft
  (its grants and starts unchanged), and a card on the Line to approve it;
* approving that draft checks it on the cases against the live version first;
* a run that kept nothing says so; a draft already open is never replaced:
  the owner hears it, and makes the draft later from the run;
* only the factory's owner, only a finished run of that factory.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from captain_claw.flight_deck import auth as fd_auth
from captain_claw.flight_deck import bat_loop, events, line
from captain_claw.flight_deck import captain_swarm_loop as loop
from captain_claw.flight_deck import course as c
from captain_claw.flight_deck import first_mate_cases as cases
from captain_claw.flight_deck import first_mate_improve as improve
from captain_claw.flight_deck import first_mate_routes as routes
from captain_claw.flight_deck import first_mate_store as fm
from captain_claw.flight_deck.auth import set_auth_db
from captain_claw.flight_deck.captain_swarm_store import CaptainSwarmStore
from captain_claw.flight_deck.db import FlightDeckDB

BOB, CAROL = "user-bob", "user-carol"
CHARTER = ("Go through my inbox for client mail about invoices and draft a short reply to each. Never send "
           "anything. Finish with who wrote and whether a draft waits.")
BETTER = CHARTER + " For each one, check the invoice number against my sent mail before drafting."


class FakeBat:
    def __init__(self) -> None:
        self.runs: dict[str, dict] = {}

    async def create_run(self, *, run_id, owner_id, title, task, config=None, origin=None,
                         source_host="localhost", source_port=0, llm_usd_cap=0.0, status="planning"):
        self.runs[run_id] = {"id": run_id, "owner_id": owner_id, "title": title, "task": task, "config": config or {},
                             "status": "done", "cumulative_tokens": 1000, "cumulative_usd": 0.01, "truth": task,
                             "stopped_reason": ""}
        return run_id

    async def get_run(self, run_id):
        r = self.runs.get(run_id)
        return dict(r) if r else None

    async def list_runs(self, owner_id, limit=50):
        return []

    async def append_event(self, run_id, stage, message="", **data):
        return 1


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
    for uid in (BOB, CAROL):
        await d._db.execute(
            "INSERT INTO users (id, email, password_hash, display_name, role, created_at, updated_at)"
            " VALUES (?, ?, 'h', ?, 'admin', '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z')",
            (uid, f"{uid}@x.co", uid))
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

    async def fake_fan_out(user_id, item, to_phones=True, rung=False):
        return {}
    monkeypatch.setattr(line, "fan_out", fake_fan_out)
    return s


@pytest.fixture
async def cs(tmp_path, monkeypatch):
    s = CaptainSwarmStore(tmp_path / "cs.db")
    loop.set_store(s)
    monkeypatch.setattr(loop, "sealed_dir", lambda rid: tmp_path / "sealed" / rid)
    yield s
    loop.set_store(None)
    await s.close()


@pytest.fixture
def bat(monkeypatch):
    fake = FakeBat()

    async def kick():
        return 0

    monkeypatch.setattr(bat_loop, "_STORE", fake)
    monkeypatch.setattr(bat_loop, "kick", kick)
    monkeypatch.setattr(improve, "POLL_S", 0.0)
    return fake


@pytest.fixture
def models(monkeypatch):
    """The reviewer grades a charter that checks invoice numbers higher; the
    proposer offers ``models.ideas`` once."""
    state = {"ideas": [{"hypothesis": "check numbers", "strategy_tag": "check-first", "charter": BETTER}], "asked": 0}

    async def grade(owner, case, result):
        return (1.0 if "check the invoice number" in result.lower() else 0.4), ""

    async def ask_model(run, system, user, *, label, temperature=0.7, panel=False):
        if label == "propose":
            state["asked"] += 1
            return json.dumps(state["ideas"] if state["asked"] == 1 else [])
        return json.dumps({"verdict": "pass", "reason": "a clearer, general charter"})

    monkeypatch.setattr(cases, "grade", grade)
    monkeypatch.setattr(loop, "ask_model", ask_model)
    return state


def _live(store, owner=BOB):
    f = store.create(owner, title="Inbox replies", charter=CHARTER,
                     triggers=[{"kind": "schedule", "schedule": "weekdays 08:00", "tz": ""}],
                     grants={"email": True, "sends_per_run": 3},
                     runner_config={"worker_mode": "plain", "steps": ["read the inbox", "draft replies"]})
    f = store.approve(f["id"], 1, f["draft"]["sha"], owner)
    for n in range(3):
        cases.add_case(store, f["id"], title=f"Sealed question number {n}",
                       input=f"From: Client {n} <client{n}@held.example>\nCheck invoice {1040 + n} for me?",
                       expect=["drafts a reply"])
    cases.add_case(store, f["id"], title="Open total question", sealed=False,
                   input="From: Ana <ana@open.example>\nIs the total right?", expect=["drafts a reply"])
    return f


async def _drive(cs, store, f, run_id="cs_0123456789ab"):
    obj = improve.normalize({**improve.factory_template(store.get(f["id"]), store.live(f["id"])),
                             "stop": {"stall_rounds": 1}})
    await cs.create_run(run_id=run_id, owner_id=BOB, title=obj["title"], objective=obj, status="baseline")
    await loop.Driver(cs, run_id).drive()
    return await cs.get_run(run_id)


async def _items(db):
    await asyncio.sleep(0)
    return await db.list_line_items(BOB, limit=50)


# ── a kept gain becomes a draft ────────────────────────────────────────

async def test_a_kept_gain_becomes_a_draft_with_its_card(store, cs, bat, models, db):
    f = _live(store)
    r = await _drive(cs, store, f)
    view = store.view(f["id"])
    d = view["draft"]
    assert d["status"] == "awaiting" and d["created_by"] == "first_mate" and d["charter"] == BETTER
    assert d["note"].startswith(f"From improvement run {r['id']}: 40% → 100% on its cases")
    assert d["grants"] == view["live"]["grants"] and d["triggers"] == view["live"]["triggers"]
    assert d["runner_config"]["steps"] == ["read the inbox", "draft replies"]
    [card] = [i for i in await _items(db) if i["ref_type"] == "first_mate_charter"]
    assert card["title"] == "Inbox replies: a better charter to approve" and card["type"] == "first_mate_improvement"
    assert card["needs_you"] and card["ref_id"] == f"{f['id']}:{d['sha'][:12]}"
    assert "the best scored 100%, against 40% before" in card["body"] and "- check numbers" in card["body"]
    assert "checks it on its cases against the live version first" in card["body"]
    assert any(e["event"] == "improvement_drafted" for e in store.events(f["id"]))
    # Approving it checks it first: P4's gate, a check on the draft.
    got = await routes.approve(f["id"], {"version": d["version"], "sha": d["sha"]}, user={"id": BOB, "role": "admin"})
    assert got["check"]["purpose"] == "approve" and got["version"] == 1
    assert (await _items(db))[0]["resolved_at"] or any(
        i["resolved_at"] for i in await _items(db) if i["ref_type"] == "first_mate_charter")


async def test_nothing_kept_says_so(store, cs, bat, models, db):
    models["ideas"] = [{"hypothesis": "say it louder", "strategy_tag": "louder", "charter": CHARTER + " LOUDLY."}]
    f = _live(store)
    await _drive(cs, store, f)
    assert store.view(f["id"])["draft"] is None
    [item] = [i for i in await _items(db) if "improvement run" in i["title"]]
    assert item["title"] == "Inbox replies: the improvement run found nothing better" and not item["needs_you"]


async def test_an_open_draft_is_never_replaced(store, cs, bat, models, db):
    f = _live(store)
    store.new_draft(f["id"], BOB, charter=CHARTER + " My own edit.")
    r = await _drive(cs, store, f)
    assert store.view(f["id"])["draft"]["charter"] == CHARTER + " My own edit."
    [item] = [i for i in await _items(db) if i["title"] == "Inbox replies: a better charter waits"]
    assert item["needs_you"] and "already has a draft open" in item["body"]
    owner = {"id": BOB, "role": "admin"}
    with pytest.raises(Exception) as e:
        await routes.improvement_draft(f["id"], r["id"], user=owner)
    assert "already has a draft open" in str(getattr(e.value, "detail", e.value))
    store.decline(f["id"], store.get(f["id"])["draft_version"], BOB)
    view = await routes.improvement_draft(f["id"], r["id"], user=owner)
    assert view["draft"]["charter"] == BETTER
    assert any(i["title"] == "Inbox replies: a better charter to approve" for i in await _items(db))


async def test_only_the_owner_and_a_finished_run(store, cs, bat, models, db):
    from fastapi import HTTPException
    f = _live(store)
    r = await _drive(cs, store, f)
    store.decline(f["id"], store.get(f["id"])["draft_version"], BOB)
    with pytest.raises(HTTPException) as e:
        await routes.improvement_draft(f["id"], r["id"], user={"id": CAROL, "role": "admin"})
    assert e.value.status_code == 404
    with pytest.raises(HTTPException) as e:
        await routes.improvement_draft(f["id"], "cs_nope", user={"id": BOB, "role": "admin"})
    assert e.value.status_code == 404
    await cs.update_run(r["id"], status="running")
    with pytest.raises(HTTPException) as e:
        await routes.improvement_draft(f["id"], r["id"], user={"id": BOB, "role": "admin"})
    assert e.value.status_code == 409


async def test_a_kinds_failing_end_never_breaks_the_run(store, cs, bat, models, db, monkeypatch):
    async def boom(run):
        raise RuntimeError("no Line today")

    monkeypatch.setattr(improve, "finished", boom)
    f = _live(store)
    r = await _drive(cs, store, f)
    assert r["status"] == "stalled" and r["state"]["report"]["improved"]
