"""First Mate cases and checks (P4, piece 2).

Pinned here:

* cases live in the factory's sealed folder (owner-only file), need at least
  one line saying what a good result does, refuse secrets, cap at 30;
* a past run with a fixed input (the events that started it) starts a case;
  one with none says so;
* the reviewer's grades: full/partial/none averaged; unreadable or no answer
  counts as nothing, never a pass;
* a check runs one case at a time as a rehearsal of the version it checks,
  grades each, and ends with the average; its runs stay out of the run list,
  the card, the Line and the day's run count;
* approving a draft of a live factory with cases checks it first: it goes
  live only at or near the live version's score, else stays waiting and the
  owner is told; skip_check approves at once; a draft changed during the
  check doesn't go live;
* a weekly check once a week while the switch is on; a drop turns the
  factory amber and posts;
* stopping a check stops its run.
"""

from __future__ import annotations

import asyncio
import json
import stat
from datetime import timedelta
from pathlib import Path

import httpx
import pytest

from captain_claw.flight_deck import auth as fd_auth
from captain_claw.flight_deck import bat_loop, events, line, rate_limiter, server
from captain_claw.flight_deck import course as c
from captain_claw.flight_deck import first_mate_cases as cases
from captain_claw.flight_deck import first_mate_loop as loop
from captain_claw.flight_deck import first_mate_runs as runs
from captain_claw.flight_deck import first_mate_score as score
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


class Reviewer:
    """Stands in for the reviewer model: answers with ``grade`` for every item."""

    def __init__(self, monkeypatch, grade="full"):
        self.grade, self.asked = grade, []

        async def ask(owner, system, user):
            self.asked.append(user)
            if isinstance(self.grade, Exception):
                raise self.grade
            if self.grade == "junk":
                return "I think it went well."
            n = user.count("\n", user.index("The checklist:"), user.index("The result:")) - 2
            return json.dumps({"grades": [{"item": i, "grade": self.grade} for i in range(1, n + 1)]})

        monkeypatch.setattr(cases, "_ask_reviewer", ask)


def _live(store, **kw):
    kw.setdefault("title", "Inbox drafts")
    kw.setdefault("charter", "Draft a reply to each client email.")
    kw.setdefault("grants", {"email": True, "sends_per_run": 3})
    f = store.create(BOB, **kw)
    return store.approve(f["id"], 1, f["draft"]["sha"], BOB)


def _case(store, f, n=1, **kw):
    kw.setdefault("title", f"Invoice question {n}")
    kw.setdefault("input", f"From: Ana <ana@client.example>\nSubject: Invoice {n}\n\nCould you check the total?")
    kw.setdefault("expect", ["Drafts a reply to Ana", "Answers the question about the total"])
    return cases.add_case(store, f["id"], **kw)


async def _items(db):
    await asyncio.sleep(0)
    return await db.list_line_items(BOB, limit=50)


async def _finish(store, bat, check_id, status="done"):
    """Finish the check's run that is going, and let the check take it in."""
    check = store.get_check(check_id)
    going = [r for r in check["results"].values() if r.get("score") is None and r.get("run_id")]
    run = store.get_run(going[-1]["run_id"])
    bat.runs[run["child_id"]].update(status="done" if status == "done" else "error",
                                     truth="Drafted the reply." if status == "done" else "",
                                     stopped_reason="" if status == "done" else "it stopped")
    await runs.refresh(run["id"])
    return run


async def _drive_through(store, bat, check_id, limit=20):
    """Drive a check to its end, finishing each case run as it starts."""
    for _ in range(limit):
        check = store.get_check(check_id)
        if check["status"] != "running":
            return check
        await cases.drive(store, fm._utcnow())
        check = store.get_check(check_id)
        if any(r.get("score") is None and r.get("run_id") and store.get_run(r["run_id"])["status"] in fm.ACTIVE_RUN
               for r in check["results"].values()):
            await _finish(store, bat, check_id)
    return store.get_check(check_id)


# ── cases ──────────────────────────────────────────────────────────────

def test_cases_live_sealed_and_owner_only(store):
    f = _live(store)
    case = _case(store, f)
    assert case["sealed"] is True and case["origin"] == "owner" and case["id"].startswith("case_")
    path = store.sealed_dir(f["id"]) / "cases.json"
    assert path.is_file() and stat.S_IMODE(path.stat().st_mode) == 0o600
    assert [x["id"] for x in cases.cases(store, f["id"])] == [case["id"]]
    open_one = _case(store, f, 2, sealed=False)
    assert open_one["sealed"] is False
    changed = cases.update_case(store, f["id"], case["id"], expect=["Drafts a reply", "Keeps it short"])
    assert changed["expect"] == ["Drafts a reply", "Keeps it short"] and changed["title"] == case["title"]
    cases.delete_case(store, f["id"], open_one["id"])
    assert len(cases.cases(store, f["id"])) == 1
    with pytest.raises(FirstMateError, match="no such case"):
        cases.delete_case(store, f["id"], open_one["id"])
    kinds = [e["event"] for e in store.events(f["id"], limit=20)]
    assert "case_added" in kinds and "case_deleted" in kinds


def test_a_case_says_what_good_looks_like(store):
    f = _live(store)
    with pytest.raises(FirstMateError, match="at least one line"):
        _case(store, f, expect=[])
    with pytest.raises(FirstMateError):
        _case(store, f, input="")
    with pytest.raises(FirstMateError):
        _case(store, f, input="my key is sk-abcdefghijklmnop")
    with pytest.raises(FirstMateError, match="no such factory"):
        cases.add_case(store, "fmf_nope", title="x", input="y", expect=["z"])


def test_cases_cap(store, monkeypatch):
    monkeypatch.setattr(cases, "CASES_MAX", 2)
    f = _live(store)
    _case(store, f, 1)
    _case(store, f, 2)
    with pytest.raises(FirstMateError, match="at most 2"):
        _case(store, f, 3)


def test_a_case_from_a_past_run(store):
    f = _live(store, checks={"contract": ["Drafts one reply per email"]})
    ev = events.get_store().add_event(BOB, source="gmail", event_type="new_email",
                                      summary="Email from Ana: Invoice question",
                                      metadata={"from": "Ana <ana@client.example>", "subject": "Invoice question",
                                                "snippet": "Could you check the total?"}, dedup_key="gmail:1")
    run = store.start_run(f["id"], trigger="events", trigger_ref=[ev["id"]])
    draft = cases.case_draft(store, run)
    assert draft["has_input"] and "Invoice question" in draft["input"] and "check the total" in draft["input"]
    assert draft["expect"] == ["Drafts one reply per email"] and draft["from_run"] == run["id"]
    store.finish_run(run["id"], "done")
    bare = store.start_run(f["id"], trigger="manual")
    assert cases.case_draft(store, bare)["has_input"] is False


# ── grading ────────────────────────────────────────────────────────────

async def test_grades_averaged_and_no_answer_is_nothing(monkeypatch):
    case = {"input": "Mail", "expect": ["One", "Two"]}
    rv = Reviewer(monkeypatch, "partial")
    assert await cases.grade(BOB, case, "a result") == (0.5, "")
    rv.grade = "full"
    assert await cases.grade(BOB, case, "a result") == (1.0, "")
    rv.grade = "junk"
    got, note = await cases.grade(BOB, case, "a result")
    assert got == 0.0 and "couldn't be read" in note and len(rv.asked) == 4  # asked twice
    rv.grade = RuntimeError("no reviewer model")
    got, note = await cases.grade(BOB, case, "a result")
    assert got == 0.0 and "no reviewer model" in note


# ── a check ────────────────────────────────────────────────────────────

async def test_a_check_runs_each_case_as_a_rehearsal(store, bat, db, phones, monkeypatch):
    rv = Reviewer(monkeypatch, "full")
    f = _live(store)
    a, b = _case(store, f, 1), _case(store, f, 2)
    check = cases.start_check(store, f["id"], version=1, purpose="manual")
    assert check["cases"] == [a["id"], b["id"]] and check["status"] == "running"
    with pytest.raises(FirstMateError, match="being checked already"):
        cases.start_check(store, f["id"], version=1, purpose="manual")

    await cases.drive(store, fm._utcnow())
    first = store.get_run(store.get_check(check["id"])["results"][a["id"]]["run_id"])
    assert first["kind"] == "check" and first["trigger"] == "check" and first["trigger_ref"] == [a["id"]]
    child = bat.runs[first["child_id"]]
    assert child["config"]["rehearsal"] == first["id"]  # nothing outside changes
    assert "Invoice 1" in child["task"] and "This is a test case" in child["task"]

    await cases.drive(store, fm._utcnow())  # still going: nothing new
    assert len(bat.runs) == 1
    await _finish(store, bat, check["id"])
    await cases.drive(store, fm._utcnow())  # grades the first
    assert store.get_check(check["id"])["results"][a["id"]]["score"] == 1.0
    rv.grade = "none"
    done = await _drive_through(store, bat, check["id"])
    assert done["status"] == "done" and done["score"] == 0.5
    assert done["results"][b["id"]]["score"] == 0.0

    # Check runs measure; they aren't the factory's work.
    detail_runs = store.list_runs(f["id"], kinds=fm.WORK_KINDS)
    assert detail_runs == [] and len(store.list_runs(f["id"])) == 2
    assert await _items(db) == []  # a manual check that didn't drop posts nothing
    assert score.scorecard(store, store.get(f["id"]), fm._utcnow())["cases"] == 2


async def test_a_failed_case_run_counts_nothing(store, bat, monkeypatch):
    Reviewer(monkeypatch, "full")
    f = _live(store)
    _case(store, f)
    check = cases.start_check(store, f["id"], version=1, purpose="manual")
    await cases.drive(store, fm._utcnow())
    await _finish(store, bat, check["id"], status="failed")
    await cases.drive(store, fm._utcnow())
    await cases.drive(store, fm._utcnow())
    done = store.get_check(check["id"])
    assert done["status"] == "done" and done["score"] == 0.0
    assert "failed" in next(iter(done["results"].values()))["note"]


async def test_check_runs_stay_out_of_the_days_count(store, bat, monkeypatch):
    Reviewer(monkeypatch, "full")
    store.set_settings({"runs_per_day": 1})
    f = _live(store)
    _case(store, f, 1)
    _case(store, f, 2)
    check = cases.start_check(store, f["id"], version=1, purpose="manual")
    assert (await _drive_through(store, bat, check["id"]))["status"] == "done"
    run = store.start_run(f["id"], trigger="manual")  # the day's one run is still free
    assert run["kind"] == "run"


def test_only_a_check_runs_the_waiting_draft(store):
    f = _live(store)
    store.new_draft(f["id"], BOB, charter="Draft a short reply to each client email.")
    store.submit(f["id"], BOB)
    with pytest.raises(FirstMateError, match="only a check"):
        store.start_run(f["id"], trigger="manual", version=2)
    with pytest.raises(FirstMateError, match="only a check"):
        store.start_run(f["id"], trigger="check", kind="check", version=7)
    run = store.start_run(f["id"], trigger="check", kind="check", version=2)
    assert run["version"] == 2


# ── the approve gate ───────────────────────────────────────────────────

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


async def _scored(store, bat, f, grade, monkeypatch):
    """The live version with one finished check at ``grade``."""
    Reviewer(monkeypatch, grade)
    check = cases.start_check(store, f["id"], version=f["version"], purpose="manual")
    return await _drive_through(store, bat, check["id"])


def _waiting_draft(store, f):
    store.new_draft(f["id"], BOB, charter="Draft a short reply to each client email.")
    return store.submit(f["id"], BOB)["draft"]


async def test_a_draft_that_scores_lower_stays_waiting(deck, store, bat, db, phones, monkeypatch):
    f = _live(store)
    _case(store, f)
    assert (await _scored(store, bat, f, "full", monkeypatch))["score"] == 1.0
    draft = _waiting_draft(store, f)
    async with _client() as cl:
        r = await cl.post(f"/fd/first-mate/factories/{f['id']}/approve", headers=_hdr(BOB),
                          json={"version": 2, "sha": draft["sha"]})
    assert r.status_code == 200
    got = r.json()
    assert got["check"]["purpose"] == "approve" and got["check"]["baseline"] == 1.0 and got["version"] == 1
    Reviewer(monkeypatch, "partial")
    done = await _drive_through(store, bat, got["check"]["id"])
    assert done["status"] == "done" and done["score"] == 0.5 and "stays a draft" in done["note"]
    after = store.view(f["id"])
    assert after["version"] == 1 and after["draft"]["version"] == 2  # still waiting
    [item] = [i for i in await _items(db) if "didn't go live" in i["title"]]
    assert item["needs_you"] and item["ref_id"] == f["id"]
    # The owner can still approve it as it is.
    async with _client() as cl:
        r = await cl.post(f"/fd/first-mate/factories/{f['id']}/approve", headers=_hdr(BOB),
                          json={"version": 2, "sha": draft["sha"], "skip_check": True})
    assert r.status_code == 200 and r.json()["version"] == 2 and "check" not in r.json()


async def test_a_draft_that_holds_up_goes_live(deck, store, bat, db, phones, monkeypatch):
    f = _live(store)
    _case(store, f)
    await _scored(store, bat, f, "partial", monkeypatch)
    draft = _waiting_draft(store, f)
    async with _client() as cl:
        got = (await cl.post(f"/fd/first-mate/factories/{f['id']}/approve", headers=_hdr(BOB),
                             json={"version": 2, "sha": draft["sha"]})).json()
    Reviewer(monkeypatch, "full")
    first = store.list_runs(f["id"])
    done = await _drive_through(store, bat, got["check"]["id"])
    assert done["status"] == "done" and "went live" in done["note"]
    assert store.get(f["id"])["version"] == 2
    case_run = next(r for r in store.list_runs(f["id"]) if r not in first)
    assert case_run["version"] == 2  # the draft was what ran
    assert any("version 2 is live" in i["title"] for i in await _items(db))
    assert cases.weekly_due(store, store.get(f["id"]), fm._utcnow()) is False  # just checked


async def test_no_cases_no_gate_and_a_changed_draft_never_goes_live(deck, store, bat, monkeypatch):
    f = _live(store)
    draft = _waiting_draft(store, f)
    async with _client() as cl:
        r = await cl.post(f"/fd/first-mate/factories/{f['id']}/approve", headers=_hdr(BOB),
                          json={"version": 2, "sha": draft["sha"]})
        assert r.status_code == 200 and r.json()["version"] == 2  # nothing to check against

        _case(store, f)
        draft = _waiting_draft(store, f)
        got = (await cl.post(f"/fd/first-mate/factories/{f['id']}/approve", headers=_hdr(BOB),
                             json={"version": 3, "sha": draft["sha"]})).json()
        assert got["check"]["version"] == 3
        r = await cl.post(f"/fd/first-mate/factories/{f['id']}/approve", headers=_hdr(CAROL),
                          json={"version": 3, "sha": draft["sha"]})
        assert r.status_code == 404
    store.edit_draft(f["id"], BOB, charter="Something else entirely.")
    Reviewer(monkeypatch, "full")
    await cases.drive(store, fm._utcnow())
    voided = store.get_check(got["check"]["id"])
    assert voided["status"] == "void" and "changed" in voided["note"]
    assert store.get(f["id"])["version"] == 2


async def test_the_gate_goes_before_a_weekly_check(store, bat, monkeypatch):
    f = _live(store)
    _case(store, f)
    weekly = cases.start_check(store, f["id"], version=1, purpose="weekly")
    draft = _waiting_draft(store, f)
    gate = cases.start_check(store, f["id"], version=2, purpose="approve")
    assert store.get_check(weekly["id"])["status"] == "void"
    assert cases.start_check(store, f["id"], version=2, purpose="approve")["id"] == gate["id"]
    assert draft["version"] == 2


# ── weekly, drift and the routes ───────────────────────────────────────

async def test_weekly_check_and_a_drop(store, bat, db, phones, monkeypatch):
    f = _live(store)
    assert cases.weekly_due(store, f, fm._utcnow()) is False  # no cases
    _case(store, f)
    assert cases.weekly_due(store, store.get(f["id"]), fm._utcnow()) is True
    store.set_prefs(BOB, {"on": True})
    Reviewer(monkeypatch, "full")
    await loop.Supervisor().tick()
    [check] = store.checks(f["id"])
    assert check["purpose"] == "weekly" and check["status"] == "running"
    await _drive_through(store, bat, check["id"])
    assert cases.weekly_due(store, store.get(f["id"]), fm._utcnow()) is False
    assert cases.weekly_due(store, store.get(f["id"]), fm._utcnow() + timedelta(days=8)) is True
    assert score.scorecard(store, store.get(f["id"]), fm._utcnow())["health"] != "amber"

    second = cases.start_check(store, f["id"], version=1, purpose="weekly")
    Reviewer(monkeypatch, "none")
    await _drive_through(store, bat, second["id"])
    card = score.scorecard(store, store.get(f["id"]), fm._utcnow())
    assert card["check"]["score"] == 0.0 and card["check"]["purpose"] == "weekly"
    assert card["health_why"] == "It scored 0% on its cases, down from 100%."
    assert any("scores lower on its cases" in i["title"] for i in await _items(db))


async def test_case_and_check_routes(deck, store, bat, monkeypatch):
    f = _live(store)
    base = f"/fd/first-mate/factories/{f['id']}"
    async with _client() as cl:
        r = await cl.post(f"{base}/check", headers=_hdr(BOB))
        assert r.status_code == 400  # no cases yet
        r = await cl.post(f"{base}/cases", headers=_hdr(BOB),
                          json={"title": "Invoice", "input": "Could you check the total?", "expect": ["Replies"]})
        assert r.status_code == 200
        case = r.json()
        assert (await cl.post(f"{base}/cases", headers=_hdr(CAROL), json={"title": "x"})).status_code == 404
        assert (await cl.post(f"{base}/cases", headers=_hdr(BOB), json={"nope": 1})).status_code == 400
        r = await cl.patch(f"{base}/cases/{case['id']}", headers=_hdr(BOB), json={"sealed": False})
        assert r.status_code == 200 and r.json()["sealed"] is False
        listed = (await cl.get(f"{base}/cases", headers=_hdr(BOB))).json()
        assert [x["id"] for x in listed["cases"]] == [case["id"]] and listed["checks"] == []

        r = await cl.post(f"{base}/check", headers=_hdr(BOB))
        assert r.status_code == 200 and r.json()["purpose"] == "manual"
        await cases.drive(store, fm._utcnow())
        going = store.active_check(f["id"])
        run_id = going["results"][case["id"]]["run_id"]
        assert (await cl.get(base, headers=_hdr(BOB))).json()["check"]["id"] == going["id"]
        r = await cl.get(f"/fd/first-mate/runs/{run_id}/case-draft", headers=_hdr(BOB))
        assert r.status_code == 404  # a check run is no source for a case
        r = await cl.delete(f"{base}/check", headers=_hdr(BOB))
        assert r.status_code == 200 and r.json()["status"] == "void"
        assert store.get_run(run_id)["status"] == "cancelled"
        assert (await cl.delete(f"{base}/check", headers=_hdr(BOB))).status_code == 404

        r = await cl.delete(f"{base}/cases/{case['id']}", headers=_hdr(BOB))
        assert r.status_code == 200 and cases.cases(store, f["id"]) == []


async def test_open_cases_are_examples_sealed_ones_never(store, bat, monkeypatch):
    Reviewer(monkeypatch, "full")
    f = _live(store)
    sealed = _case(store, f, 1, title="Sealed one", input="SEALED-INPUT-TEXT")
    shown = _case(store, f, 2, title="Open one", input="OPEN-INPUT-TEXT", sealed=False)
    run = await runs.start(f["id"], BOB, watch=False)
    task = bat.runs[run["child_id"]]["task"]
    assert "Example: Open one" in task and "OPEN-INPUT-TEXT" in task
    assert "SEALED-INPUT-TEXT" not in task and "Sealed one" not in task
    bat.runs[run["child_id"]].update(status="done", truth="ok")
    await runs.refresh(run["id"])
    # A check of the open case doesn't show it its own answer.
    check = store.create_check(f["id"], version=1, purpose="manual", cases=[shown["id"]])
    await cases.drive(store, fm._utcnow())
    child = bat.runs[store.get_run(store.get_check(check["id"])["results"][shown["id"]]["run_id"])["child_id"]]
    assert "Example: Open one" not in child["task"] and "OPEN-INPUT-TEXT" in child["task"]
    assert sealed["sealed"]
