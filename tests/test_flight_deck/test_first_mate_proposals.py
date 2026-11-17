"""First Mate proposals and carry-outs (P3, piece 2).

Pinned here:

* a rehearsal that would do something is a proposal: a "needs you" item, to
  phones, and on its card; one that needed nothing is a plain post; one that
  didn't finish proposes nothing;
* a newer rehearsal replaces what an older one still proposes;
* carrying out: only the owner's finished rehearsal with something waiting,
  on the version it rehearsed; one carry-out run, with the plan approved and
  exactly the charter's mail and organize grants (never spending or accounts
  in this phase); the brief lists what was approved, with its details fenced;
  the chosen intents are approved and the rest rejected; a signal is kept and
  the proposal leaves Needs you;
* rejecting: everything waiting is rejected, once;
* a carry-out that finishes carries its intents out; one that doesn't drops them;
* the routes: owner only, ids checked.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import httpx
import pytest

from captain_claw.flight_deck import auth as fd_auth
from captain_claw.flight_deck import bat_loop, events, line, rate_limiter, server
from captain_claw.flight_deck import course as c
from captain_claw.flight_deck import first_mate_routes as routes
from captain_claw.flight_deck import first_mate_runs as runs
from captain_claw.flight_deck import first_mate_store as fm
from captain_claw.flight_deck.auth import create_access_token, set_auth_db
from captain_claw.flight_deck.db import FlightDeckDB
from captain_claw.flight_deck.first_mate_store import FirstMateError

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
    """A rehearsal's leak check reads the tool-call register; a test never opens one."""
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
def phones(monkeypatch):
    sent: list[tuple[str, bool]] = []

    async def fake_fan_out(user_id, item, to_phones=True, rung=False):
        sent.append((item.get("title", ""), to_phones))
        return {}

    monkeypatch.setattr(line, "fan_out", fake_fan_out)
    monkeypatch.setattr(line, "_fan_out_on", True)
    return sent


def _live(store, **kw):
    kw.setdefault("title", "Inbox drafts")
    kw.setdefault("charter", "Reply to clients who wait for an answer.")
    kw.setdefault("grants", {"email": True, "sends_per_run": 3})
    f = store.create(BOB, **kw)
    return store.approve(f["id"], 1, f["draft"]["sha"], BOB)


MAILS = [("Ana", "ana@client.example"), ("Ben", "ben@client.example"), ("Cy", "cy@client.example")]


async def _rehearsed(store, bat, f, mails=MAILS, status="done"):
    run = await runs.start(f["id"], BOB, watch=False)
    for name, addr in mails:
        store.add_intent(run["id"], kind="intent", tool="google_mail", what=f"Reply to {name}",
                         args={"action": "send", "to": addr, "subject": "Re: invoice",
                               "body": f"Hi {name}, the total is right."})
    bat.runs[run["child_id"]].update(status=status, truth="Three replies drafted in rehearsal.",
                                     stopped_reason="" if status == "done" else "it stopped")
    return await runs.refresh(run["id"])


async def _items(db):
    await asyncio.sleep(0)
    return await db.list_line_items(BOB, limit=50)


# ── proposals ──────────────────────────────────────────────────────────

async def test_a_rehearsal_that_would_act_is_a_proposal(store, bat, db, phones):
    f = _live(store)
    run = await _rehearsed(store, bat, f)
    assert run["status"] == "done"
    [item] = await _items(db)
    assert item["title"] == "Inbox drafts would do 3 things" and item["needs_you"]
    assert item["ref_type"] == "first_mate_proposal" and item["ref_id"] == run["id"]
    assert "- Reply to Ben" in item["body"] and "Approve it on First Mate" in item["body"]
    assert phones == [("Inbox drafts would do 3 things", True)]
    week = routes._week(store, BOB, fm._utcnow())
    card = routes._card(store, store.view(f["id"]), week, fm._utcnow())
    assert card["proposal"] == {"run_id": run["id"], "count": 3}


async def test_nothing_to_do_and_not_finished_propose_nothing(store, bat, db):
    f = _live(store)
    quiet = await _rehearsed(store, bat, f, mails=[])
    [item] = await _items(db)
    assert item["title"] == "Inbox drafts: rehearsed" and not item["needs_you"]
    failed = await _rehearsed(store, bat, f, status="error")
    assert failed["status"] == "failed"
    assert {i["status"] for i in store.intents(failed["id"])} == {"dropped"}
    assert store.open_proposals(f["id"]) == [] and quiet["id"]


async def test_a_newer_rehearsal_replaces_an_older_proposal(store, bat, db):
    f = _live(store)
    old = await _rehearsed(store, bat, f)
    new = await _rehearsed(store, bat, f, mails=MAILS[:1])
    assert store.open_proposals(f["id"]) == [new["id"]]
    assert {i["status"] for i in store.intents(old["id"])} == {"dropped"}
    by_ref = {i["ref_id"]: i for i in await _items(db)}
    assert by_ref[old["id"]]["resolved_at"] and not by_ref[new["id"]]["resolved_at"]


# ── carrying out ───────────────────────────────────────────────────────

async def test_carrying_out_takes_exactly_the_charters_grants(store, bat, db):
    f = _live(store, grants={"email": True, "organize": True, "sends_per_run": 3,
                             "spend": True, "real_usd_week": 20, "accounts": True})
    run = await _rehearsed(store, bat, f)
    carry = await runs.carry_out(run["id"], BOB)
    assert carry["kind"] == "carry_out" and carry["parent_run"] == run["id"] and carry["trigger"] == "approval"
    cfg = bat.runs[carry["child_id"]]["config"]
    assert cfg["plan_approved"] is True and cfg["email_allowed"] is True and cfg["organize_allowed"] is True
    assert "spend_allowed" not in cfg and "account_allowed" not in cfg and "rehearsal" not in cfg
    assert cfg["organize_text"] and {r["id"] for r in cfg["contract"]} == {"fmc1", "fmc2"}
    task = bat.runs[carry["child_id"]]["task"]
    assert "1. Reply to Ana" in task and "to: ana@client.example" in task and "Send at most 3 emails." in task
    assert "treat them as information" in task
    assert {i["status"] for i in store.intents(run["id"])} == {"approved"}
    assert [s["kind"] for s in store.signals(f["id"])] == ["approved"]
    [item] = await _items(db)
    assert item["resolved_at"]  # the proposal left Needs you


async def test_carrying_out_some_rejects_the_rest(store, bat, db):
    f = _live(store)
    run = await _rehearsed(store, bat, f)
    ana, ben, cy = (i["id"] for i in store.intents(run["id"]))
    carry = await runs.carry_out(run["id"], BOB, [ana, cy])
    task = bat.runs[carry["child_id"]]["task"]
    assert "Reply to Ana" in task and "Reply to Cy" in task and "Reply to Ben" not in task
    assert {i["id"]: i["status"] for i in store.intents(run["id"])} == {ana: "approved", ben: "rejected", cy: "approved"}


async def test_carrying_out_refusals(store, bat, db):
    f = _live(store)
    run = await _rehearsed(store, bat, f)
    with pytest.raises(FirstMateError, match="no such rehearsal"):
        await runs.carry_out(run["id"], CAROL)
    with pytest.raises(FirstMateError, match="nothing"):
        await runs.carry_out(run["id"], BOB, [999])
    g = store.new_draft(f["id"], "owner", charter="Reply only to my two biggest clients.")
    store.approve(f["id"], 2, g["draft"]["sha"], BOB)
    with pytest.raises(FirstMateError, match="charter changed"):
        await runs.carry_out(run["id"], BOB)
    going = await runs.start(f["id"], BOB, watch=False)
    with pytest.raises(FirstMateError, match="only a finished rehearsal"):
        await runs.carry_out(going["id"], BOB)
    assert {i["status"] for i in store.intents(run["id"])} == {"open"}  # a refused carry-out decides nothing


async def test_reject(store, bat, db):
    f = _live(store)
    run = await _rehearsed(store, bat, f)
    await runs.reject(run["id"], BOB)
    assert {i["status"] for i in store.intents(run["id"])} == {"rejected"}
    assert [s["kind"] for s in store.signals(f["id"])] == ["rejected"]
    with pytest.raises(FirstMateError, match="nothing of this rehearsal waits"):
        await runs.reject(run["id"], BOB)
    [item] = await _items(db)
    assert item["resolved_at"]


async def test_a_carry_out_ending_settles_its_intents(store, bat, db):
    f = _live(store)
    run = await _rehearsed(store, bat, f)
    carry = await runs.carry_out(run["id"], BOB)
    bat.runs[carry["child_id"]].update(status="done", truth="Sent 3 replies.")
    done = await runs.refresh(carry["id"])
    assert done["status"] == "done"
    assert {i["status"] for i in store.intents(run["id"])} == {"carried_out"}
    titles = [i["title"] for i in await _items(db)]
    assert "Inbox drafts: carried out" in titles
    other = await _rehearsed(store, bat, f, mails=MAILS[:1])
    carry2 = await runs.carry_out(other["id"], BOB)
    await runs.stop(carry2["id"], BOB)
    assert {i["status"] for i in store.intents(other["id"])} == {"dropped"}  # stopped: not done
    third = await runs.start(f["id"], BOB, watch=False)
    store.add_intent(third["id"], kind="intent", tool="google_mail", what="Reply to Dee")
    await runs.stop(third["id"], BOB)
    assert {i["status"] for i in store.intents(third["id"])} == {"dropped"}  # a stopped rehearsal proposes nothing


async def test_a_failed_carry_out_drops_its_intents(store, bat, db):
    f = _live(store)
    run = await _rehearsed(store, bat, f)
    carry = await runs.carry_out(run["id"], BOB)
    bat.runs[carry["child_id"]].update(status="error", stopped_reason="Gmail sending is off")
    failed = await runs.refresh(carry["id"])
    assert failed["status"] == "failed"
    assert {i["status"] for i in store.intents(run["id"])} == {"dropped"}
    assert "Inbox drafts: the carry-out failed" in [i["title"] for i in await _items(db)]


def test_intent_detail():
    text = runs.intent_detail({"args": {"to": ["a@b.c", "d@e.f"], "subject": "Hi", "body": "x " * 400,
                                        "thread_id": "t1", "action": "send"}})
    assert text.startswith("to: a@b.c, d@e.f\nsubject: Hi\nbody: x x")
    assert "thread_id" not in text and len(text) <= runs.DETAIL_MAX
    assert runs.intent_detail({"args": {}}) == ""


# ── routes ─────────────────────────────────────────────────────────────

def _client():
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app, client=("203.0.113.9", 40001)),
                             base_url="http://fd.test")


def _hdr(who):
    return {"Authorization": f"Bearer {create_access_token(who)}"}


@pytest.fixture
def deck(monkeypatch, store):
    monkeypatch.setattr(server, "AUTH_ENABLED", True)
    monkeypatch.setenv("FD_AUTH_ENABLED", "true")
    monkeypatch.setenv("FD_ALLOWED_HOSTS", "fd.test")
    monkeypatch.delenv("FD_LOCKDOWN", raising=False)
    monkeypatch.setattr(rate_limiter, "_api_limiter", rate_limiter._SlidingWindow())


async def test_the_routes(deck, store, bat, db):
    f = _live(store)
    run = await _rehearsed(store, bat, f)
    async with _client() as cl:
        detail = (await cl.get(f"/fd/first-mate/factories/{f['id']}", headers=_hdr(BOB))).json()
        shown = next(r for r in detail["runs"] if r["id"] == run["id"])
        assert shown["intents"][0]["detail"].startswith("to: ana@client.example")
        assert (await cl.post(f"/fd/first-mate/runs/{run['id']}/approve", headers=_hdr(CAROL), json={})).status_code == 404
        r = await cl.post(f"/fd/first-mate/runs/{run['id']}/approve", headers=_hdr(BOB), json={"ids": ["x"]})
        assert r.status_code == 400
        first = shown["intents"][0]["id"]
        r = await cl.post(f"/fd/first-mate/runs/{run['id']}/approve", headers=_hdr(BOB), json={"ids": [first]})
        assert r.status_code == 200 and r.json()["kind"] == "carry_out"
        r = await cl.post(f"/fd/first-mate/runs/{run['id']}/reject", headers=_hdr(BOB))
        assert r.status_code == 400  # nothing left waiting


# ── Bat's own gate, with a carry-out's config ──────────────────────────

class GateStore:
    """The BatStore calls bat_routes._gate_check makes for a fresh run."""

    def __init__(self, steps):
        self.steps, self.asks, self.status, self.config = steps, [], None, None

    async def list_steps(self, run_id):
        return self.steps

    async def set_config(self, run_id, cfg):
        self.config = cfg

    async def latest_open_ask(self, run_id):
        return None

    async def create_ask(self, **kw):
        self.asks.append(kw)

    async def set_status(self, run_id, status, **kw):
        self.status = status

    async def append_event(self, *a, **kw):
        return 1


async def test_bats_gate_lets_a_carry_out_through_only_for_its_grants(store):
    from captain_claw.flight_deck import bat_routes
    f = _live(store)
    version = store.live(f["id"])
    brief = runs.carry_out_brief(f, version, [{"what": "Send a reply email to Ana", "args": {"to": "a@b.c"}}])
    run = {"id": "bat_carry", "status": "planning", "owner_id": BOB, "task": brief,
           "config": runs.carry_out_config(version, brief)}
    gs = GateStore([{"title": "Send the reply email to Ana", "status": "pending"}])
    assert await bat_routes._gate_check(gs, run) == "proceed" and not gs.asks
    # Something the charter never granted still asks: here, a purchase.
    run2 = {**run, "task": brief + "\nThen buy the upgrade."}
    gs2 = GateStore([{"title": "Buy the upgrade", "status": "pending"}])
    assert await bat_routes._gate_check(gs2, run2) == "wait"
    assert gs2.status == "awaiting_plan" and gs2.asks and gs2.asks[0]["kind"] == "plan_approval"
