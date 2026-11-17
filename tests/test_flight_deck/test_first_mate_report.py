"""First Mate's weekly report (P4, piece 4).

Pinned here:

* the week is Monday to Monday UTC; "last" is the week that ended, "this"
  is the week so far;
* totals and a line per factory: runs, done, carry-outs, doubts, tokens
  against the budget and its share, the week's signals, its cases' score,
  what moved (trust, a version, a pause); only what happened in the week;
* worth a look: a paused or red factory, proposals waiting, a held draft,
  more trust waiting on calibration, a budget nearly used;
* it goes out once, on the first scan after 07:00 UTC on Monday, as one Line
  item; a week with nothing in it sends nothing;
* the route shows last week or this week so far, to its owner only.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest

from captain_claw.flight_deck import auth as fd_auth
from captain_claw.flight_deck import course as c
from captain_claw.flight_deck import events, line, rate_limiter, server
from captain_claw.flight_deck import first_mate_cases as cases
from captain_claw.flight_deck import first_mate_loop as loop
from captain_claw.flight_deck import first_mate_report as report
from captain_claw.flight_deck import first_mate_store as fm
from captain_claw.flight_deck.auth import create_access_token, set_auth_db
from captain_claw.flight_deck.db import FlightDeckDB

BOB, CAROL = "user-bob", "user-carol"
TUESDAY = datetime(2026, 11, 17, 10, 0, tzinfo=UTC)  # the week began Monday 16 Nov
LAST_WEEK = datetime(2026, 11, 11, 9, 0, tzinfo=UTC)


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
    f = store.approve(f["id"], 1, f["draft"]["sha"], owner)
    with store._tx() as conn:  # made before the weeks these tests look at
        conn.execute("UPDATE fm_factories SET created_at = ? WHERE id = ?",
                     (fm._iso(LAST_WEEK - timedelta(days=30)), f["id"]))
    return store.get(f["id"])


def _run(store, f, at, kind="run", status="done", tokens=100_000, parent=""):
    r = store.start_run(f["id"], trigger="approval" if kind == "carry_out" else "manual", kind=kind,
                        parent_run=parent, now=at)
    store.mark_running(r["id"], "bat_" + r["id"])
    return store.finish_run(r["id"], status, verdict="pass" if status == "done" else "fail", tokens=tokens)


def _backdate(store, table, ids, at):
    with store._tx() as conn:
        for i in ids:
            conn.execute(f"UPDATE {table} SET at = ? WHERE id = ?", (fm._iso(at), i))


async def _items(db, who=BOB):
    await asyncio.sleep(0)
    return await db.list_line_items(who, limit=50)


# ── the week ───────────────────────────────────────────────────────────

def test_the_weeks():
    assert report.period(TUESDAY, "last") == (datetime(2026, 11, 9, tzinfo=UTC), datetime(2026, 11, 16, tzinfo=UTC))
    assert report.period(TUESDAY, "this") == (datetime(2026, 11, 16, tzinfo=UTC), TUESDAY)
    assert report._dates(datetime(2026, 11, 9, tzinfo=UTC), datetime(2026, 11, 16, tzinfo=UTC)) == "9–15 Nov"
    assert report._dates(datetime(2026, 11, 30, tzinfo=UTC), datetime(2026, 12, 7, tzinfo=UTC)) == "30 Nov – 6 Dec"
    assert report._dates(*report.period(TUESDAY, "this")) == "16–17 Nov"
    assert report.tokens_text(2_150_000) == "2.1M" and report.tokens_text(25_000_000) == "25M"
    assert report.tokens_text(34_200) == "34k" and report.tokens_text(900) == "900"


def test_a_factorys_week_counts_only_that_week(store):
    f = _live(store)
    reh = _run(store, f, LAST_WEEK, kind="rehearsal")
    carry = _run(store, f, LAST_WEEK + timedelta(hours=1), kind="carry_out", parent=reh["id"])
    store.set_proof(carry["id"], {"state": "doubt", "note": "it sent more than approved"})
    _run(store, f, LAST_WEEK + timedelta(hours=2), status="failed")
    _run(store, f, TUESDAY - timedelta(hours=1), tokens=5_000_000)  # this week: not in last week's report
    sigs = [store.add_signal(f["id"], "thumbs_up", run_id=carry["id"]),
            store.add_signal(f["id"], "sent_edited", value=0.8, run_id=carry["id"]),
            store.add_signal(f["id"], "draft_deleted", run_id=carry["id"])]
    _backdate(store, "fm_signals", sigs, LAST_WEEK + timedelta(days=1))
    store.add_signal(f["id"], "thumbs_down", run_id=carry["id"])  # this week
    start, end = report.period(TUESDAY, "last")
    rep = report.build(store, BOB, start, end, TUESDAY)
    [x] = rep["factories"]
    assert (x["runs"], x["done"], x["carry_outs"], x["doubts"]) == (3, 2, 1, 1)
    assert x["tokens"] == 300_000 and x["share"] == 25_000_000
    assert x["signals"] == 3 and x["satisfaction"] == 0.6 and x["drafts"] == {"sent": 0, "edited": 1, "deleted": 1}
    t = rep["totals"]
    assert t["runs"] == 3 and t["tokens"] == 300_000 and t["budget"] == 25_000_000 and t["doubts"] == 1
    this = report.build(store, BOB, *report.period(TUESDAY, "this"), TUESDAY)
    assert this["factories"][0]["runs"] == 1 and this["totals"]["tokens"] == 5_000_000


def test_what_moved_and_whats_worth_a_look(store):
    f = _live(store, title="Inbox replies")
    g = _live(store, title="Calendar tidy", grants={"organize": True})
    store.set_trust(f["id"], "commit", "owner")
    store.set_state(g["id"], "paused", "deck", "3 failed runs in a row")
    reh = _run(store, f, TUESDAY - timedelta(hours=2), kind="rehearsal")
    store.add_intent(reh["id"], kind="intent", tool="google_mail", what="Reply to Ana")
    cases.add_case(store, f["id"], title="Invoice", input="Could you check the total?", expect=["Replies"])
    store.new_draft(f["id"], BOB, charter="Reply to clients briefly.")
    draft = store.submit(f["id"], BOB)["draft"]
    k = store.create_check(f["id"], version=draft["version"], purpose="approve", cases=["case_x"], baseline=0.9)
    store.finish_check(k["id"], "done", score=0.4, note="It scored 40% on its cases; the live version scored 90%. "
                                                       "It stays a draft.")
    rep = report.build(store, BOB, *report.period(fm._utcnow(), "this"), fm._utcnow())
    by = {x["title"]: x for x in rep["factories"]}
    assert "You set its trust to: acts within its grants." in by["Inbox replies"]["moved"]
    assert any("didn't go live" in m for m in by["Inbox replies"]["moved"])
    assert by["Calendar tidy"]["moved"] == ["Paused: 3 failed runs in a row."]
    assert "Calendar tidy is paused: 3 failed runs in a row." in rep["look"]
    assert "Inbox replies has 1 proposal waiting for you." in rep["look"]
    assert "Inbox replies: version 2 waits for you after scoring 40% on its cases." in rep["look"]
    title, body = report.text(rep)
    assert title.startswith("First Mate's week: ")
    assert "**Inbox replies** · " in body and "**Worth a look**" in body and "- Calendar tidy is paused" in body


def test_a_nearly_used_budget_is_worth_a_look(store):
    f = _live(store)
    store.set_prefs(BOB, {"weekly_tokens": 5_000_000})
    _run(store, f, LAST_WEEK, tokens=4_600_000)
    rep = report.build(store, BOB, *report.period(TUESDAY, "last"), TUESDAY)
    assert "Your factories used 92% of the week's 5M tokens." in rep["look"]
    _, body = report.text(rep)
    assert "1 run, 1 done · 4.6M of 5M tokens" in body


# ── sending ────────────────────────────────────────────────────────────

async def test_it_goes_out_once_on_monday_morning(store, db, phones):
    f = _live(store)
    _run(store, f, LAST_WEEK)
    early = datetime(2026, 11, 16, 6, 30, tzinfo=UTC)
    assert await report.maybe_send(store, BOB, early) is False and report.last_sent(store, BOB) == {}
    monday = datetime(2026, 11, 16, 7, 5, tzinfo=UTC)
    assert await report.maybe_send(store, BOB, monday) is True
    assert await report.maybe_send(store, BOB, monday + timedelta(hours=3)) is False
    [item] = await _items(db)
    assert item["title"] == "First Mate's week: 9–15 Nov" and item["ref_type"] == "first_mate_report"
    assert item["ref_id"] == "2026-11-09" and not item["needs_you"] and "1 run, 1 done" in item["body"]
    assert phones == [("First Mate's week: 9–15 Nov", False)]  # nothing worth a ping
    assert report.last_sent(store, BOB)["week"] == "2026-11-16"


async def test_an_empty_week_sends_nothing(store, db, phones):
    _live(store)
    assert await report.maybe_send(store, BOB, TUESDAY) is False
    assert await _items(db) == [] and report.last_sent(store, BOB)["sent"] is False
    assert await report.maybe_send(store, CAROL, TUESDAY) is False  # no factories at all


async def test_the_scan_sends_it(store, db, phones, monkeypatch):
    monkeypatch.setattr(report, "REPORT_HOUR", 0)
    f = _live(store)
    store.set_state(f["id"], "paused", "owner", "on holiday")
    week_ago = loop.week_start(fm._utcnow()) - timedelta(days=3)
    _run(store, f, week_ago)
    await loop.Supervisor().tick()
    [item] = [i for i in await _items(db) if i["ref_type"] == "first_mate_report"]
    assert "Inbox replies is paused: on holiday." in item["body"]
    assert phones[-1][1] is True  # something worth a look may reach the phone
    await loop.Supervisor().tick()
    assert len([i for i in await _items(db) if i["ref_type"] == "first_mate_report"]) == 1


def test_forgetting_the_owner_forgets_the_mark(store):
    _live(store)
    store._put_setting("report:user-bob", {"week": "2026-11-16"})
    store.forget_user(BOB)
    assert report.last_sent(store, BOB) == {}


# ── the route ──────────────────────────────────────────────────────────

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


async def test_the_report_route(deck, store):
    f = _live(store)
    _run(store, f, fm._utcnow() - timedelta(minutes=5))
    async with _client() as cl:
        this = (await cl.get("/fd/first-mate/report?week=this", headers=_hdr(BOB))).json()
        last = (await cl.get("/fd/first-mate/report", headers=_hdr(BOB))).json()
        other = (await cl.get("/fd/first-mate/report?week=this", headers=_hdr(CAROL))).json()
        bad = await cl.get("/fd/first-mate/report?week=next", headers=_hdr(BOB))
    assert this["totals"]["runs"] == 1 and this["title"].startswith("First Mate's week") and "1 run" in this["text"]
    assert last["totals"]["runs"] == 0 and last["sent"] == {}
    assert other["factories"] == [] and bad.status_code == 400
