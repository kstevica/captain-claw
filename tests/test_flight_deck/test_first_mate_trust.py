"""First Mate calibration and trust raised by the record (P4, piece 3).

Pinned here:

* the deck's verdict: failed, a leak or a doubtful proof is bad, finished is
  good; the owner's label: 👍/👎, then approve/reject, then the drafts' fate;
* calibration is label_eval's kappa over the owner's labelled runs across
  their factories: under 30 labels, or every result called good, or kappa
  under 0.6 is not calibrated;
* the record counts carry-outs (the owner's word, else the deck's verdict;
  a part-approved proposal by its share) and outright rejections (0), only
  since trust last moved;
* the factory moves up a level only with 10 items at ≥ 0.9, no doubt, a
  calibrated owner, running and not pinned; never to direct with outward
  grants; the owner hears it on the Line; the supervisor does it on its scan.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import httpx
import pytest

from captain_claw.flight_deck import auth as fd_auth
from captain_claw.flight_deck import course as c
from captain_claw.flight_deck import events, line, rate_limiter, server
from captain_claw.flight_deck import first_mate_loop as loop
from captain_claw.flight_deck import first_mate_store as fm
from captain_claw.flight_deck import first_mate_trust as trust
from captain_claw.flight_deck.auth import create_access_token, set_auth_db
from captain_claw.flight_deck.db import FlightDeckDB

BOB, CAROL = "user-bob", "user-carol"


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
            " VALUES (?, ?, 'h', ?, 'user', '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z')",
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
    s.set_settings({"enabled": True, "admins_only": False, "runs_per_day": 500})
    return s


@pytest.fixture
def phones(monkeypatch):
    sent: list[tuple[str, bool]] = []

    async def fake_fan_out(user_id, item, to_phones=True, rung=False):
        sent.append((item.get("title", ""), to_phones))
        return {}

    monkeypatch.setattr(line, "fan_out", fake_fan_out)
    monkeypatch.setattr(line, "_fan_out_on", True)
    return sent


def _live(store, owner=BOB, **kw):
    kw.setdefault("title", "Inbox replies")
    kw.setdefault("charter", "Reply to clients who wait for an answer.")
    kw.setdefault("grants", {"email": True, "sends_per_run": 3})
    f = store.create(owner, **kw)
    return store.approve(f["id"], 1, f["draft"]["sha"], owner)


def _run(store, f, kind="run", status="done", verdict="pass", parent=""):
    trigger = "approval" if kind == "carry_out" else "manual"
    r = store.start_run(f["id"], trigger=trigger, kind=kind, parent_run=parent)
    store.mark_running(r["id"], "bat_" + r["id"])
    return store.finish_run(r["id"], status, verdict=verdict if status == "done" else "fail")


def _proposal(store, f, *, approve=2, reject=0, outcome=None, doubt=False):
    """A rehearsal with ``approve + reject`` intents decided by the owner, and
    the carry-out of the approved ones (with ``outcome`` signals on it)."""
    reh = _run(store, f, kind="rehearsal")
    ids = [store.add_intent(reh["id"], kind="intent", tool="google_mail", what=f"Reply {n}")
           for n in range(approve + reject)]
    if reject:
        store.decide_intents(reh["id"], "rejected", ids=ids[approve:])
    if not approve:
        store.add_signal(f["id"], "rejected", run_id=reh["id"], source="first_mate")
        return reh, None
    store.decide_intents(reh["id"], "approved", ids=ids[:approve])
    store.add_signal(f["id"], "approved", run_id=reh["id"], source="first_mate")
    carry = _run(store, f, kind="carry_out", parent=reh["id"])
    store.decide_intents(reh["id"], "carried_out", ids=ids[:approve])
    if doubt:
        store.set_proof(carry["id"], {"state": "doubt", "note": "it sent more than approved"})
    for kind, value in outcome or []:
        store.add_signal(f["id"], kind, value=value, run_id=carry["id"], source="owner")
    return reh, store.get_run(carry["id"])


def _labels(store, f, good=25, bad=5, deck_misses=0):
    """Direct runs the owner labelled: ``good`` 👍 on finished runs, ``bad`` 👎
    on failed ones, and ``deck_misses`` 👎 on runs the deck called good."""
    for _ in range(good):
        store.set_thumbs(f["id"], _run(store, f)["id"], "thumbs_up")
    for _ in range(bad):
        store.set_thumbs(f["id"], _run(store, f, status="failed")["id"], "thumbs_down")
    for _ in range(deck_misses):
        store.set_thumbs(f["id"], _run(store, f)["id"], "thumbs_down")


async def _items(db):
    await asyncio.sleep(0)
    return await db.list_line_items(BOB, limit=50)


# ── labels ─────────────────────────────────────────────────────────────

def test_the_decks_verdict_and_the_owners_label():
    run = {"kind": "carry_out", "status": "done", "verdict": "pass", "proof": {}}
    assert trust.deck_verdict(run) == 1
    assert trust.deck_verdict({**run, "proof": {"state": "doubt"}}) == 0
    assert trust.deck_verdict({**run, "status": "failed"}) == 0
    assert trust.deck_verdict({**run, "kind": "rehearsal", "verdict": "leak"}) == 0
    assert trust.deck_verdict({**run, "status": "cancelled"}) is None
    sig = lambda kind, value: {"kind": kind, "value": value}  # noqa: E731
    assert trust.owner_label([sig("draft_deleted", 0.0), sig("thumbs_up", 1.0)]) == 1  # 👍 wins
    assert trust.owner_label([sig("approved", 1.0), sig("thumbs_down", 0.0)]) == 0
    assert trust.owner_label([sig("rejected", 0.0)]) == 0
    assert trust.owner_label([sig("sent_edited", 0.8), sig("sent_as_written", 1.0)]) == 1
    assert trust.owner_label([sig("sent_edited", 0.3)]) == 0
    assert trust.owner_label([]) is None


def test_calibration_needs_30_labels_that_agree(store):
    f = _live(store)
    _labels(store, f, good=10, bad=2)
    cal = trust.calibration(store, BOB)
    assert cal["n"] == 12 and not cal["ready"] and cal["why"] == "12 of the 30 labels it needs so far"

    _labels(store, f, good=15, bad=3)
    cal = trust.calibration(store, BOB)
    assert cal["n"] == 30 and cal["kappa"] == 1.0 and cal["ready"] and cal["why"] == ""
    assert trust.calibration(store, CAROL)["n"] == 0  # one owner's labels only


def test_all_good_is_not_calibrated_nor_disagreement(store):
    f = _live(store)
    _labels(store, f, good=30, bad=0)
    cal = trust.calibration(store, BOB)
    assert cal["kappa"] is None and not cal["ready"] and "can't tell" in cal["why"]
    _labels(store, f, good=0, bad=3, deck_misses=10)  # the deck called 10 of the owner's 👎 good
    cal = trust.calibration(store, BOB)
    assert cal["n"] == 43 and cal["kappa"] < trust.KAPPA_BAR and not cal["ready"] and "kappa" in cal["why"]


def test_calibration_spans_factories(store):
    a, b = _live(store), _live(store, title="Calendar tidy", grants={"organize": True})
    _labels(store, a, good=12, bad=3)
    _labels(store, b, good=12, bad=3)
    assert trust.calibration(store, BOB)["n"] == 30


# ── the record ─────────────────────────────────────────────────────────

def test_the_record_counts_carry_outs_and_rejections(store):
    f = _live(store)
    _proposal(store, f, outcome=[("thumbs_up", None)])  # the owner: good
    _proposal(store, f)  # no word: the deck's verdict, good
    _proposal(store, f, approve=1, reject=1)  # half of it approved
    _proposal(store, f, approve=0, reject=2)  # rejected outright
    _proposal(store, f, outcome=[("draft_deleted", None)])
    rec = trust.record(store, store.get(f["id"]))
    values = [i["value"] for i in rec["items"]]
    assert values == [0.0, 0.0, 0.5, 1.0, 1.0]  # newest first
    assert [i["from"] for i in rec["items"]] == ["you", "you", "its checks", "its checks", "you"]
    assert rec["counted"] == 5 and rec["value"] == 0.5 and not rec["eligible"]
    assert rec["why"] == "5 of the 10 settled results it needs at this level so far."


def test_ten_good_results_and_calibration_raise_it(store, db, phones):
    f = _live(store)
    _labels(store, _live(store, title="Labels", grants={}), good=25, bad=5)
    for _ in range(9):
        _proposal(store, f, outcome=[("sent_as_written", None)])
    cal = trust.calibration(store, BOB)
    assert not trust.record(store, store.get(f["id"]), cal)["eligible"]
    _proposal(store, f, outcome=[("sent_edited", 0.95)])
    rec = trust.record(store, store.get(f["id"]), cal)
    assert rec["eligible"] and rec["next"] == "commit" and rec["value"] >= 0.99


async def test_the_scan_raises_it_and_says_so(store, db, phones):
    f = _live(store)
    _labels(store, _live(store, title="Labels", grants={}), good=25, bad=5)
    for _ in range(10):
        _proposal(store, f, outcome=[("thumbs_up", None)])
    store.set_prefs(BOB, {"on": False})  # the record counts whether or not it runs on its own
    out = await loop.Supervisor().tick()
    assert out["trusted"] == 1
    moved = store.get(f["id"])
    assert moved["trust"] == "commit" and not moved["trust_pinned"]
    [ev] = [e for e in store.events(f["id"]) if e["event"] == "trust"]
    assert ev["actor"] == "record" and ev["detail"]["to"] == "commit"
    [item] = [i for i in await _items(db) if i["title"] == "Inbox replies now acts within its grants"]
    assert "kappa 1.00 on 50" in item["body"] and "pin it" in item["body"]  # 30 + each proposal and carry-out
    assert ("Inbox replies now acts within its grants", True) in phones
    # A new level starts a new record: nothing moves it again yet.
    assert (await loop.Supervisor().tick())["trusted"] == 0
    rec = trust.record(store, store.get(f["id"]), trust.calibration(store, BOB))
    assert rec["counted"] == 0 and rec["next"] is None  # email: never direct
    assert "always rehearses first" in rec["why"]


async def test_what_keeps_it_where_it_is(store, db, phones):
    f = _live(store, grants={"organize": True})
    _labels(store, _live(store, title="Labels", grants={}), good=25, bad=5)
    for _ in range(9):
        _proposal(store, f, outcome=[("thumbs_up", None)])
    _proposal(store, f, doubt=True)
    cal = trust.calibration(store, BOB)
    rec = trust.record(store, store.get(f["id"]), cal)
    assert rec["doubts"] == 1 and not rec["eligible"] and "doubtful" in rec["why"]

    store.set_trust(f["id"], "propose", "owner", pin=True)  # a trust move: a new record
    for _ in range(10):
        _proposal(store, f, outcome=[("thumbs_up", None)])
    rec = trust.record(store, store.get(f["id"]), cal)
    assert rec["counted"] == 10 and rec["doubts"] == 0 and not rec["eligible"] and "pinned" in rec["why"]
    assert await trust.consider_owner(store, BOB) == 0

    store.set_trust(f["id"], "propose", "owner", pin=False)
    store.set_state(f["id"], "paused", "owner")
    for _ in range(10):
        _proposal(store, f, outcome=[("thumbs_up", None)])
    assert "isn't running" in trust.record(store, store.get(f["id"]), cal)["why"]
    store.set_state(f["id"], "running", "owner")
    assert await trust.consider_owner(store, BOB) == 1
    assert store.get(f["id"])["trust"] == "commit"
    # Organize only: commit may go on to direct, on a fresh record.
    for _ in range(10):
        _proposal(store, f, outcome=[("thumbs_up", None)])
    assert await trust.consider_owner(store, BOB) == 1
    assert store.get(f["id"])["trust"] == "direct"


async def test_no_calibration_no_move(store, db, phones):
    f = _live(store)
    for _ in range(10):
        _proposal(store, f, outcome=[("thumbs_up", None)])
    rec = trust.record(store, store.get(f["id"]), trust.calibration(store, BOB))
    assert not rec["eligible"] and "agree with you" in rec["why"]
    assert await trust.consider_owner(store, BOB) == 0
    assert store.get(f["id"])["trust"] == "propose"


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


async def test_status_and_detail_carry_the_ladder(deck, store):
    f = _live(store)
    _labels(store, f, good=5, bad=1)
    for _ in range(3):
        _proposal(store, f, outcome=[("thumbs_up", None)])
    async with _client() as cl:
        status = (await cl.get("/fd/first-mate", headers=_hdr(BOB))).json()
        detail = (await cl.get(f"/fd/first-mate/factories/{f['id']}", headers=_hdr(BOB))).json()
    assert status["calibration"]["n"] == 12 and status["calibration"]["ready"] is False
    ladder = detail["ladder"]
    assert ladder["level"] == "propose" and ladder["next"] == "commit"
    assert ladder["counted"] == 3 and ladder["needed"] == 10 and ladder["calibration"]["n"] == 12
