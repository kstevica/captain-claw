"""The Line (line.py): one timeline per user for everything the deck says on
its own, and the bell as its Needs-you list. The bell's writers keep calling
``db.add_notification``; every item lands on the Line, and the bell shows only
what waits for the user or failed — what still waits on top, however old. A
proposal is answered on the autonomy ledger, anything else by being read.
Bell rows from before the Line come in once (and again after a rollback);
items past the retention go, unless they still wait."""

from __future__ import annotations

import asyncio

import httpx
import pytest
from fastapi import FastAPI

from captain_claw.config import AutonomousWorkConfig
from captain_claw.flight_deck import auth as fd_auth
from captain_claw.flight_deck import autonomy, line, line_routes, notification_routes, owner_notify
from captain_claw.flight_deck.auth import get_current_user, set_auth_db
from captain_claw.flight_deck.db import FlightDeckDB

ALICE, BOB = "user-alice", "user-bob"
OLD = "2026-01-01T00:00:00+00:00"


@pytest.fixture
async def db(tmp_path, monkeypatch):
    store = autonomy.AutonomyStore(tmp_path / "autonomy.db")
    monkeypatch.setattr(autonomy, "_STORE", store)
    base = AutonomousWorkConfig().model_dump()
    monkeypatch.setattr(autonomy, "global_defaults", lambda: dict(base))
    monkeypatch.setattr(line, "_last_prune", 0.0)
    prev = fd_auth._db
    d = FlightDeckDB(tmp_path / "fd.db")
    await d.init()
    set_auth_db(d)
    try:
        yield d
    finally:
        await d.close()
        fd_auth._db = prev


def _client(user_id: str = ALICE) -> httpx.AsyncClient:
    app = FastAPI()
    app.include_router(notification_routes.router)
    app.include_router(line_routes.router)
    app.dependency_overrides[get_current_user] = lambda: {"id": user_id}
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://fd.test")


async def _age(db, item_id: str, when: str = OLD) -> None:
    await db._db.execute("UPDATE line_items SET created_at = ? WHERE id = ?", (when, item_id))
    await db._db.commit()


def test_classify_names_a_kind_and_whether_it_needs_you():
    assert line.classify("autonomy_proposal", "autonomy_action") == ("proposal", True, "agent")
    assert line.classify("autonomy_nudge") == ("nudge", False, "agent")
    assert line.classify("run") == ("run", False, "deck")
    assert line.classify("run_error")[:2] == ("run", True)
    assert line.classify("share")[:2] == ("share", False)
    assert line.classify("payment")[:2] == ("payment", True)
    assert line.classify("email_autoreply_held") == ("mail", True, "untrusted")
    # A member's suggested lesson is an "info", yet it waits for the owner.
    assert line.classify("info", "agent_lessons") == ("agent", True, "deck")
    assert line.classify("success", "agent_exam")[:2] == ("agent", False)
    # A type nobody listed stays in the bell, as before the Line.
    assert line.classify("something_new")[:2] == ("deck", True)


async def test_add_notification_writes_the_line_and_the_old_bell(db):
    nid = await db.add_notification(ALICE, "share", "Bob shared a folder", "docs",
                                    ref_type="folder", ref_id="f1")
    (item,) = await db.list_line_items(ALICE)
    assert item["id"] == nid and item["kind"] == "share" and item["type"] == "share"
    assert item["needs_you"] == 0 and item["read_at"] == "" and item["ref_id"] == "f1"
    # Under the same id in the old table: a deck rolled back still has its bell.
    old = await db._db.execute_fetchall("SELECT id, type FROM notifications")
    assert [(r["id"], r["type"]) for r in old] == [(nid, "share")]
    # Everything it wrote, in the bell's row shape; the bell itself only what needs you.
    (row,) = await db.list_notifications(ALICE)
    assert row["title"] == "Bob shared a folder" and row["read"] == 0 and row["needs_you"] is False
    assert await db.list_notifications(ALICE, needs_you=True) == []


async def test_the_bell_lists_what_needs_you_and_every_one_still_waiting(db):
    store = autonomy.get_store()
    act = store.add_action(ALICE, kind="run_prompt", title="Re-run the search", status="awaiting_approval")
    waiting = await db.add_notification(ALICE, "autonomy_proposal", "Waiting for you", "Re-run",
                                        ref_type="autonomy_action", ref_id=act["id"])
    await _age(db, waiting)
    for i in range(5):                                          # read: answered
        await db.mark_notification_read(await db.add_notification(ALICE, "warning", f"Look at {i}"), ALICE)
    unread = await db.add_notification(ALICE, "warning", "Not looked at yet")
    await _age(db, unread, "2026-01-02T00:00:00+00:00")
    await db.add_notification(ALICE, "email_sent", "Captain sent an email")   # Line only
    await db.add_notification(BOB, "warning", "Bob's")

    async with _client() as c:
        async def bell(limit):
            r = await c.get(f"/fd/notifications?limit={limit}")
            return r.json()

        got = await bell(3)
        ids = [n["id"] for n in got["notifications"]]
        # The newest three, then what still waits however old (newest first).
        assert len(ids) == 5 and ids[-2:] == [unread, waiting]
        assert "Captain sent an email" not in [n["title"] for n in got["notifications"]]
        assert got["unread"] == 2                               # what needs you, unread
        assert {n["id"]: n["resolved"] for n in got["notifications"]}[waiting] is False
        assert [n["id"] for n in (await bell(50))["notifications"]].count(waiting) == 1

        # Read, a proposal still waits; answered on the ledger, it's done.
        await c.post(f"/fd/notifications/{waiting}/read")
        assert waiting in [n["id"] for n in (await bell(3))["notifications"]]
        store.update_status(act["id"], "rejected")
        assert waiting not in [n["id"] for n in (await bell(3))["notifications"]]
        (item,) = [i for i in await db.list_line_items(ALICE) if i["id"] == waiting]
        assert item["resolved_at"] and item["read_at"]


async def test_reading_answers_an_item_unless_it_is_answered_elsewhere(db):
    pay = await db.add_notification(ALICE, "payment", "Paid twice?", ref_type="payment", ref_id="p1")
    prop = await db.add_notification(ALICE, "autonomy_proposal", "Reply to Ana",
                                     ref_type="autonomy_action", ref_id="a1")
    info = await db.add_notification(ALICE, "info", "FYI")
    assert {r["id"] for r in await db.list_unresolved_line_items(ALICE)} == {pay, prop}
    assert await db.mark_all_notifications_read(ALICE, needs_you=True) == 2
    assert [r["id"] for r in await db.list_unresolved_line_items(ALICE)] == [prop]
    # The bell's "mark all read" leaves what isn't in the bell alone.
    (fyi,) = [i for i in await db.list_line_items(ALICE) if i["id"] == info]
    assert fyi["read_at"] == ""
    old = {r["id"]: r["read"] for r in await db._db.execute_fetchall("SELECT id, read FROM notifications")}
    assert old[pay] == 1 and old[prop] == 1


async def test_the_line_page_lists_everything_with_its_counts(db):
    await db.add_notification(ALICE, "run", "Basna run finished: report")
    await db.add_notification(ALICE, "run_error", "Vatra run failed: build")
    fleet = await line.post(ALICE, "fleet", "Ana joined the fleet", "Port 24100.", bell_type="fleet",
                            source="Ana")
    assert fleet
    async with _client() as c:
        r = (await c.get("/fd/line")).json()
        assert [i["title"] for i in r["items"]] == [
            "Ana joined the fleet", "Vatra run failed: build", "Basna run finished: report"]
        assert r["items"][0]["kind"] == "fleet" and r["items"][0]["source"] == "Ana"
        assert r["unread"] == 3 and r["waiting"] == 1 and r["kinds"]["fleet"] == "Fleet"
        runs = (await c.get("/fd/line?kind=run")).json()["items"]
        assert len(runs) == 2
        page = (await c.get(f"/fd/line?limit=1&before={r['items'][0]['created_at']}")).json()["items"]
        assert [i["title"] for i in page] == ["Vatra run failed: build"]
        assert (await c.post(f"/fd/line/{fleet}/read")).json()["ok"] is True
        assert (await c.get("/fd/line/unread")).json() == {"unread": 2, "waiting": 1}
        await c.post("/fd/line/read-all")
        assert (await c.get("/fd/line/unread")).json() == {"unread": 0, "waiting": 0}
    # Only your own.
    async with _client(BOB) as c:
        assert (await c.get("/fd/line")).json()["items"] == []
        assert (await c.post(f"/fd/line/{fleet}/read")).json()["ok"] is False


async def test_post_needs_a_title_and_never_raises(db, monkeypatch):
    assert await line.post(ALICE, "fleet", "  ") == ""
    assert await line.post("", "nonsense", "Deck restarted") != ""
    (item,) = await db.list_line_items("local")                 # no sign-in: the local bucket
    assert item["kind"] == "deck"

    def broken():
        raise RuntimeError("no db")

    monkeypatch.setattr(fd_auth, "get_db", broken)
    assert await line.post(ALICE, "deck", "Lost") == ""


async def test_old_items_go_unless_they_still_wait(db):
    keep = await db.add_notification(ALICE, "payment", "Approve a purchase", ref_type="purchase", ref_id="x")
    gone = await db.add_notification(ALICE, "share", "Old share")
    answered = await db.add_notification(ALICE, "warning", "Old warning")
    await db.mark_notification_read(answered, ALICE)
    fresh = await db.add_notification(ALICE, "share", "New share")
    for item_id in (keep, gone, answered):
        await _age(db, item_id)
    assert await db.prune_line_items(line.retention_cutoff()) == 2
    assert {i["id"] for i in await db.list_line_items(ALICE)} == {keep, fresh}


async def test_bell_rows_from_before_the_line_come_in_once(tmp_path):
    path = tmp_path / "fd.db"
    d = FlightDeckDB(path)
    await d.init()
    rows = [
        ("n1", ALICE, "share", "Shared", 1, "", "", OLD),
        ("n2", ALICE, "payment", "Paid twice?", 1, "payment", "p1", "2026-01-02T00:00:00+00:00"),
        ("n3", ALICE, "autonomy_proposal", "Reply?", 1, "autonomy_action", "a1", "2026-01-03T00:00:00+00:00"),
        ("n4", ALICE, "bat_ask", "Bat needs you", 0, "bat", "r1", "2026-01-04T00:00:00+00:00"),
    ]
    await d._db.execute("DELETE FROM line_items")
    await d._db.execute("DELETE FROM system_settings WHERE key = ?", (d._LINE_COPIED_KEY,))
    await d._db.executemany(
        "INSERT INTO notifications (id, user_id, type, title, body, ref_type, ref_id, read, created_at)"
        " VALUES (?, ?, ?, ?, '', ?, ?, ?, ?)",
        [(i, u, t, ti, rt, ri, rd, at) for i, u, t, ti, rd, rt, ri, at in rows])
    await d._db.commit()
    await d.close()

    d = FlightDeckDB(path)
    await d.init()
    items = {i["id"]: i for i in await d.list_line_items(ALICE)}
    assert set(items) == {"n1", "n2", "n3", "n4"}
    assert items["n1"]["kind"] == "share" and items["n1"]["needs_you"] == 0
    assert items["n2"]["read_at"] == items["n2"]["resolved_at"] == "2026-01-02T00:00:00+00:00"
    # A read proposal still waits until the ledger says it's answered.
    assert items["n3"]["read_at"] and items["n3"]["resolved_at"] == ""
    assert items["n4"]["needs_you"] == 1 and items["n4"]["read_at"] == ""

    # The Line's own state wins; a row an older build wrote meanwhile (a
    # rollback) comes in at the next start.
    await d.mark_notification_read("n4", ALICE)
    await d._db.execute(
        "INSERT INTO notifications (id, user_id, type, title, body, ref_type, ref_id, read, created_at)"
        " VALUES ('n5', ?, 'warning', 'While rolled back', '', '', '', 0, ?)", (ALICE, "2099-01-01T00:00:00+00:00"))
    await d._db.commit()
    await d.close()

    d = FlightDeckDB(path)
    await d.init()
    items = {i["id"]: i for i in await d.list_line_items(ALICE)}
    assert set(items) == {"n1", "n2", "n3", "n4", "n5"}
    assert items["n4"]["read_at"] and items["n5"]["needs_you"] == 1
    await d.close()


async def test_a_proposal_backfilled_into_the_bell_lands_on_the_line(db):
    act = autonomy.get_store().add_action(ALICE, kind="run_prompt", title="Re-run the search",
                                          rationale="Prices moved.", status="awaiting_approval")
    assert await owner_notify.bell_backfill(ALICE) == 1
    (item,) = await db.list_line_items(ALICE)
    assert item["kind"] == "proposal" and item["ref_id"] == act["id"] and item["needs_you"] == 1
    assert await owner_notify.bell_backfill(ALICE) == 0


async def test_a_fleet_change_lands_on_the_owners_line(db, monkeypatch):
    from captain_claw.flight_deck import server

    def no_docker():
        raise RuntimeError("no docker here")

    monkeypatch.setattr(server, "get_docker", no_docker)
    monkeypatch.setattr(server, "_load_process_registry", lambda: {})
    await server._notify_fleet_change("Ana", 24100, owner_id=ALICE)
    await server._notify_fleet_change("Ana", 24133, event="rebound", owner_id=ALICE)
    items = await db.list_line_items(ALICE)
    assert [(i["kind"], i["title"], i["source"], i["needs_you"]) for i in items] == [
        ("fleet", "Ana moved to port 24133", "Ana", 0),
        ("fleet", "Ana joined the fleet", "Ana", 0),
    ]
    assert items[1]["body"] == "Port 24100. No other agent running."
    # Never the bell.
    assert await db.list_notifications(ALICE, needs_you=True) == []


# ── P1: what agents did on their own, heard by the deck ───────────────

async def _settle() -> None:
    """The items the ledger handed to the loop, written."""
    await asyncio.gather(*[t for t in asyncio.all_tasks()
                           if getattr(t.get_coro(), "__name__", "") == "action_done"])


async def test_a_scheduled_run_lands_on_the_line(db, tmp_path, monkeypatch):
    from captain_claw.flight_deck import fd_scheduler as sched

    store = sched.SchedulerStore(db_path=tmp_path / "scheduler.db")
    monkeypatch.setattr(sched, "_STORE", store)
    monkeypatch.setattr(line, "agent_name", lambda slug: {"mira": "Mira"}.get(slug, slug))
    fields = {"schedule": "daily 08:00", "agent_slug": "mira", "prompt": "Brief me",
              "delivery_kind": "agent", "delivery": {"chat": True}}
    brief = store.create(owner_id=ALICE, name="Morning brief", **fields)
    sweep = store.create(owner_id=ALICE, name="Inbox sweep", **fields)
    quiet = store.create(owner_id=ALICE, name="Quiet one", **fields)
    outcomes = {brief["id"]: ("ok", "Three meetings today."),
                sweep["id"]: ("error:no-reply", "agent produced no reply within timeout"),
                quiet["id"]: ("skipped:quiet", "")}

    async def fake(job, force=False):
        return outcomes[job["id"]]

    monkeypatch.setattr(sched, "execute_job", fake)
    for job in (brief, sweep, quiet):
        await sched.run_job(job, force=True)
    items = {i["title"]: i for i in await db.list_line_items(ALICE)}
    assert set(items) == {'Scheduled task "Morning brief"', 'Scheduled task "Inbox sweep" failed'}
    ok, bad = items['Scheduled task "Morning brief"'], items['Scheduled task "Inbox sweep" failed']
    assert ok["kind"] == "result" and ok["body"] == "Three meetings today." and ok["source"] == "Mira"
    assert ok["needs_you"] == 0 and ok["ref_type"] == "scheduled_task" and ok["ref_id"] == brief["id"]
    assert bad["needs_you"] == 1 and bad["body"] == "Problem: no-reply\n\nagent produced no reply within timeout"
    assert [n["title"] for n in await db.list_notifications(ALICE, needs_you=True)] == [bad["title"]]


async def test_an_action_ending_on_the_ledger_lands_on_the_line_once(db, monkeypatch):
    monkeypatch.setattr(line, "agent_name", lambda slug: "Mira" if slug == "mira" else slug)
    store = autonomy.get_store()
    ran = store.add_action(ALICE, kind="run_prompt", title="Re-run the price check", status="dispatched",
                           target="mira")
    broke = store.add_action(ALICE, kind="tool_action", title="Draft a reply to Ana", status="dispatched")
    nudge = store.add_action(ALICE, kind="nudge", title="Invoice due Friday", status="dispatched")
    store.update_status(ran["id"], "done", outcome="success", outcome_note="Two prices moved.")
    store.update_status(broke["id"], "done", outcome="fail", outcome_note="gmail said no")
    store.update_status(nudge["id"], "done", outcome="success", outcome_note="delivered")
    store.update_status(ran["id"], "done", outcome="success", outcome_note="again")   # once
    waiting = store.add_action(ALICE, kind="run_prompt", title="Not yet", status="awaiting_approval")
    store.update_status(waiting["id"], "rejected")                                      # not an outcome
    await _settle()
    rows = await db.list_line_items(ALICE)
    assert sorted(i["title"] for i in rows) == ["Done: Re-run the price check", "Failed: Draft a reply to Ana"]
    items = {i["title"]: i for i in rows}
    done = items["Done: Re-run the price check"]
    assert done["body"] == "Two prices moved." and done["source"] == "Mira" and done["needs_you"] == 0
    assert done["ref_type"] == "autonomy_result" and done["ref_id"] == ran["id"] and done["trust"] == "agent"
    assert items["Failed: Draft a reply to Ana"]["needs_you"] == 1


def test_an_action_ending_off_the_loop_is_not_lost(monkeypatch):
    """The Taskboard writes the ledger from a worker thread: the item goes to
    the deck's loop."""
    import threading

    calls = []

    async def fake_done(row):
        calls.append(row["id"])

    loop = asyncio.new_event_loop()
    monkeypatch.setattr(line, "action_done", fake_done)
    from captain_claw.flight_deck import taskboard_passive

    monkeypatch.setattr(taskboard_passive, "_main_loop", lambda: loop)
    t = threading.Thread(target=line.soon, args=(line.action_done, {"id": "a1"}))
    t.start()
    t.join()
    loop.run_until_complete(asyncio.sleep(0.05))
    loop.close()
    assert calls == ["a1"]


async def test_a_plan_step_lands_on_the_line(db, monkeypatch):
    from captain_claw.flight_deck import plans

    plan = {"id": "plan1", "goal": "Get the Q4 numbers to Ana"}
    await plans._step_to_line(ALICE, plan, {"title": "Export the sheet", "result": "q4.xlsx saved"}, True)
    await plans._step_to_line(ALICE, plan, {"title": "Email Ana", "result": "no Gmail"}, False)
    done, failed = sorted(await db.list_line_items(ALICE), key=lambda i: i["title"])
    assert done["title"] == "Plan step done: Export the sheet" and done["needs_you"] == 0
    assert done["body"] == "q4.xlsx saved\n\nPlan: Get the Q4 numbers to Ana"
    assert failed["title"] == "Plan step failed: Email Ana" and failed["needs_you"] == 1
