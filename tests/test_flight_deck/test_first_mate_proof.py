"""First Mate proof and trust (P3, piece 3).

Pinned here:

* which grant a call needs; what trust ``commit`` may carry out by itself
  (within the mail and organize grants, at most sends_per_run emails, never a
  browser commit point);
* the verdict on a carry-out: confirmed; nothing outward; a doubt for mail
  gone from Google, more sends than approved, something not approved, or a
  claim the record doesn't bear out; the run's own files never count;
* a finished carry-out gets its proof; a doubt lowers trust one level and asks
  the owner; an unreadable register leaves it unknown;
* trust ``commit``: a rehearsal's covered intents are carried out by
  themselves, the rest stays a proposal, and no approval is recorded for the owner;
* setting trust by hand settles the factory's doubts.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from captain_claw.flight_deck import auth as fd_auth
from captain_claw.flight_deck import bat_loop, events, taskboard_proof
from captain_claw.flight_deck import first_mate_proof as proof
from captain_claw.flight_deck import first_mate_runs as runs
from captain_claw.flight_deck import first_mate_store as fm
from captain_claw.flight_deck.auth import set_auth_db
from captain_claw.flight_deck.db import FlightDeckDB

BOB = "user-bob"


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
    """A test never opens a tool-call register."""
    from captain_claw.flight_deck import tool_audit_routes
    monkeypatch.setattr(tool_audit_routes, "_data_dir", None)
    monkeypatch.setattr(tool_audit_routes, "_store", None)


@pytest.fixture
async def db(tmp_path: Path):
    prev = fd_auth._db
    d = FlightDeckDB(tmp_path / "fd.db")
    await d.init()
    set_auth_db(d)
    try:
        yield d
    finally:
        await d.close()
        fd_auth._db = prev


@pytest.fixture
def store(tmp_path, monkeypatch, db):
    monkeypatch.setattr(events, "_STORE", events.EventsStore(tmp_path / "events.db"))
    s = fm.FirstMateStore(tmp_path / "first_mate.db")
    monkeypatch.setattr(fm, "_STORE", s)
    s.set_settings({"enabled": True})
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


def _row(tool, action="", outcome="ok", refs=None, **params):
    return {"tool": tool, "risk": "write", "outcome": outcome, "reason": "",
            "params": json.dumps({"action": action, **params} if action else params),
            "refs": json.dumps(refs or {}), "agent_slug": "bat-bat-1234-worker"}


def _actions(rows):
    return [a for a in (taskboard_proof._action(r) for r in rows) if a]


def _intent(i, tool="google_mail", action="send", kind="intent", status="open"):
    return {"id": i, "tool": tool, "kind": kind, "status": status, "what": f"{tool} {action}", "args": {"action": action}}


# ── pure ───────────────────────────────────────────────────────────────

def test_grant_of():
    assert proof.grant_of("google_mail", "send") == "email"
    assert proof.grant_of("google_mail", "create_draft") == "email"
    assert proof.grant_of("imap_mail", "reply") == "email"
    assert proof.grant_of("send_mail", "") == "email"
    assert proof.grant_of("google_mail", "archive") == "organize"
    assert proof.grant_of("microsoft_mail", "add_labels") == "organize"
    assert proof.grant_of("google_mail", "search") is None
    assert proof.grant_of("google_calendar", "create_event") is None


def test_what_commit_may_carry_out_by_itself():
    intents = [_intent(1), _intent(2), _intent(3), _intent(4, action="archive"),
               _intent(5, tool="google_calendar", action="create_event"),
               _intent(6, tool="browser", action="click", kind="commit"), _intent(7, status="rejected")]
    grants = {"email": True, "organize": True, "sends_per_run": 2}
    assert proof.covered(intents, grants) == [1, 2, 4]
    assert proof.covered(intents, {"email": False, "organize": True, "sends_per_run": 0}) == [4]
    assert proof.covered(intents, {}) == []


def test_a_confirmed_carry_out():
    rows = [_row("google_mail", "send", to="ana@x.co", subject="Re"), _row("google_mail", "send", to="ben@x.co"),
            _row("google_mail", "search", query="from:ana"), _row("write", path="notes.md", content="x")]
    got = proof.judge(rows, _actions(rows), [_intent(1), _intent(2)], {"email": True, "sends_per_run": 3},
                      result="Sent two replies.", status="done")
    assert got["state"] == proof.CONFIRMED and got["sent"] == 2 and got["outward"] == 2 and not got["notes"]


def test_doubts():
    sends = [_row("google_mail", "send", to=f"{n}@x.co") for n in ("a", "b", "c")]
    grants = {"email": True, "sends_per_run": 5}
    more = proof.judge(sends, _actions(sends), [_intent(1), _intent(2)], grants, result="", status="done")
    assert more["state"] == proof.DOUBT and "sent 3 emails; you approved 2" in more["note"]
    capped = proof.judge(sends[:2], _actions(sends[:2]), [_intent(1), _intent(2)], {"email": True, "sends_per_run": 1},
                         result="", status="done")
    assert "you approved 1" in capped["note"]
    extra = [sends[0], _row("google_calendar", "create_event", summary="Lunch")]
    odd = proof.judge(extra, _actions(extra), [_intent(1)], grants, result="", status="done")
    assert odd["state"] == proof.DOUBT and "didn't approve" in odd["note"]
    gone_actions = _actions(sends[:1])
    gone_actions[0]["seen"] = taskboard_proof.GONE
    gone = proof.judge(sends[:1], gone_actions, [_intent(1)], grants, result="", status="done")
    assert "doesn't have it" in gone["note"]
    claim = proof.judge([], [], [_intent(1)], grants, result="I sent the reply to Ana.", status="done")
    assert claim["state"] == proof.DOUBT and "no send is on record" in claim["note"]


def test_nothing_outward():
    rows = [_row("google_mail", "search", query="x"), _row("write", path="a.md", content="y")]
    got = proof.judge(rows, _actions(rows), [_intent(1)], {"email": True, "sends_per_run": 2},
                      result="Nothing needed a reply.", status="done")
    assert got["state"] == proof.NOTHING


# ── proving a carry-out ────────────────────────────────────────────────

def _live(store, trust="propose", **kw):
    kw.setdefault("title", "Client replies")
    kw.setdefault("charter", "Reply to clients who wait for an answer.")
    kw.setdefault("grants", {"email": True, "sends_per_run": 2})
    f = store.create(BOB, **kw)
    f = store.approve(f["id"], 1, f["draft"]["sha"], BOB)
    if trust != "propose":
        store.set_trust(f["id"], trust, "owner")
    return store.get(f["id"])


async def _rehearsal(store, bat, f, intents):
    run = await runs.start(f["id"], BOB, watch=False)
    for tool, action, name in intents:
        store.add_intent(run["id"], kind="intent", tool=tool, what=f"{action} {name}", args={"action": action, "to": f"{name}@x.co"})
    bat.runs[run["child_id"]].update(status="done", truth="Rehearsed.")
    return await runs.refresh(run["id"])


async def _items(db):
    await asyncio.sleep(0)
    return await db.list_line_items(BOB, limit=50)


async def test_a_doubtful_carry_out_lowers_trust_and_asks(store, bat, db, monkeypatch):
    f = _live(store, trust="commit")
    rehearsal = await _rehearsal(store, bat, f, [("google_mail", "send", "ana")])
    [carry] = [r for r in store.list_runs(f["id"]) if r["kind"] == "carry_out"]  # commit carried it out by itself
    sends = [_row("google_mail", "send", to="ana@x.co"), _row("google_mail", "send", to="eve@x.co")]

    async def rows(run):
        return sends

    async def no_google(uid, actions):
        return None

    monkeypatch.setattr(proof, "_rows", rows)
    monkeypatch.setattr(taskboard_proof, "confirm", no_google)
    bat.runs[carry["child_id"]].update(status="done", truth="Sent the reply.")
    done = await runs.refresh(carry["id"])
    assert done["proof"]["state"] == proof.DOUBT and "sent 2 emails; you approved 1" in done["proof"]["note"]
    assert store.get(f["id"])["trust"] == "propose"
    titles = {i["title"]: i for i in await _items(db)}
    doubt = titles["Client replies: check its carry-out"]
    assert doubt["needs_you"] and doubt["ref_type"] == "first_mate_proof" and "asks first" in doubt["body"]
    assert rehearsal["id"] == carry["parent_run"]


async def test_a_confirmed_carry_out_keeps_trust(store, bat, db, monkeypatch):
    f = _live(store, trust="commit")
    await _rehearsal(store, bat, f, [("google_mail", "send", "ana")])
    [carry] = [r for r in store.list_runs(f["id"]) if r["kind"] == "carry_out"]

    async def rows(run):
        return [_row("google_mail", "send", to="ana@x.co")]

    async def no_google(uid, actions):
        return None

    monkeypatch.setattr(proof, "_rows", rows)
    monkeypatch.setattr(taskboard_proof, "confirm", no_google)
    bat.runs[carry["child_id"]].update(status="done", truth="Sent the reply.")
    done = await runs.refresh(carry["id"])
    assert done["proof"]["state"] == proof.CONFIRMED and store.get(f["id"])["trust"] == "commit"


async def test_an_unreadable_register_leaves_it_unknown(store, bat, db):
    f = _live(store)
    rehearsal = await _rehearsal(store, bat, f, [("google_mail", "send", "ana")])
    carry = await runs.carry_out(rehearsal["id"], BOB)
    bat.runs[carry["child_id"]].update(status="done", truth="Sent.")
    done = await runs.refresh(carry["id"])
    assert done["proof"]["state"] == proof.UNKNOWN and store.get(f["id"])["trust"] == "propose"


# ── trust commit ───────────────────────────────────────────────────────

async def test_commit_carries_out_what_its_grants_cover(store, bat, db):
    f = _live(store, trust="commit")
    rehearsal = await _rehearsal(store, bat, f, [("google_mail", "send", "ana"), ("google_mail", "send", "ben"),
                                                 ("google_mail", "send", "cy"), ("google_calendar", "create_event", "lunch")])
    [carry] = [r for r in store.list_runs(f["id"]) if r["kind"] == "carry_out"]
    by_id = {i["id"]: i for i in store.intents(rehearsal["id"])}
    assert sorted(int(x) for x in carry["trigger_ref"]) == sorted(i for i, v in by_id.items() if v["status"] == "approved")
    assert [v["what"] for v in by_id.values() if v["status"] == "approved"] == ["send ana", "send ben"]
    assert sorted(v["what"] for v in by_id.values() if v["status"] == "open") == ["create_event lunch", "send cy"]
    assert store.signals(f["id"]) == []  # the owner didn't approve anything
    items = {i["title"]: i for i in await _items(db)}
    assert items["Client replies would do 2 things"]["needs_you"]  # the rest waits


async def test_propose_carries_out_nothing_by_itself(store, bat, db):
    f = _live(store)
    await _rehearsal(store, bat, f, [("google_mail", "send", "ana")])
    assert [r["kind"] for r in store.list_runs(f["id"])] == ["rehearsal"]


async def test_setting_trust_by_hand_settles_doubts(store, bat, db, monkeypatch):
    from captain_claw.flight_deck import first_mate_routes as routes
    f = _live(store, trust="commit")
    await _rehearsal(store, bat, f, [("google_mail", "send", "ana")])
    [carry] = [r for r in store.list_runs(f["id"]) if r["kind"] == "carry_out"]

    async def rows(run):
        return [_row("google_calendar", "create_event", summary="x")]

    async def no_google(uid, actions):
        return None

    monkeypatch.setattr(proof, "_rows", rows)
    monkeypatch.setattr(taskboard_proof, "confirm", no_google)
    bat.runs[carry["child_id"]].update(status="done", truth="Done.")
    await runs.refresh(carry["id"])
    doubt = next(i for i in await _items(db) if i["ref_type"] == "first_mate_proof")
    assert not doubt["resolved_at"]
    monkeypatch.setattr(routes, "_mine", lambda fid, user: store.view(fid))
    await routes.set_trust(f["id"], {"level": "propose", "pin": True}, user={"id": BOB, "role": "user"})
    doubt = next(i for i in await _items(db) if i["ref_type"] == "first_mate_proof")
    assert doubt["resolved_at"]
