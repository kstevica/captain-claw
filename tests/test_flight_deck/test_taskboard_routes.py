"""The Taskboard API: the caller's board only (someone else's task is a 404,
their lists are empty), the shapes the dialog reads, views with counts and
facets, corrections with Undo, folded search — and Approve / Reject of a
waiting proposal moving the row, with Reject a compare-and-set like Approve
(a 409 once the action is no longer waiting)."""

from __future__ import annotations

import secrets
import time
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI, HTTPException

from captain_claw.flight_deck import auth, autonomy_routes, fd_dispatch, taskboard

from .test_taskboard_sources import _seed_everything, env  # noqa: F401

ITEM_FIELDS = {
    "id", "agent_slug", "agent_ref", "agent_name", "title", "ask", "result_text", "kind", "direction",
    "channel", "channels", "status", "owes_next", "evidence", "result", "urgency", "due_at", "snooze_until",
    "created_at", "updated_at", "last_event_at", "closed_at", "closed_by", "closed_by_party", "triage_state",
    "triage_error", "triage_after", "user_locks", "flags", "private", "merged_into", "source", "source_ref",
    "next_steps", "retention_until", "parties", "lane", "proposal", "links", "in_line",
}
LINK_FIELDS = {"scheduled_job_id", "session_id", "session_agent", "autonomy_id", "followup_id",
               "gmail_thread_id", "autoreply_id", "link_id"}


@pytest.fixture()
async def api(env, monkeypatch):  # noqa: F811
    monkeypatch.setattr(taskboard, "_SYNC_GAP_S", 0.0)
    app = FastAPI()
    app.include_router(taskboard.router)
    alice = {"id": env.alice, "email": "alice@x.co", "display_name": "Alice", "role": "user"}
    bob = {"id": env.bob, "email": "bob@x.co", "display_name": "Bob", "role": "user"}
    current = {"user": alice}
    app.dependency_overrides[auth.get_current_user] = lambda: current["user"]
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://deck") as client:
        yield SimpleNamespace(client=client, env=env, alice=alice, bob=bob,
                              be=lambda who: current.update(user=who))


async def _get(api, path, **params):
    r = await api.client.get(path, params=params)
    assert r.status_code == 200, r.text
    return r.json()


async def _tasks(api, **params):
    return await _get(api, "/fd/taskboard/tasks", **params)


async def _sync(api):
    r = await api.client.post("/fd/taskboard/sync")
    assert r.status_code == 200, r.text
    return r.json()


async def _act(api, task_id, action, value=None):
    return await api.client.post(f"/fd/taskboard/tasks/{task_id}/action", json={"action": action, "value": value})


# ── shapes ────────────────────────────────────────────────────────────


async def test_status_lists_the_sources_counts_and_the_sweep(api):
    out = await _get(api, "/fd/taskboard/status")
    # v2: the 30-day history fill (contract 5) and the agents' to-do lists.
    assert set(out) == {"sources", "counts", "sweep", "backfill"}
    tracked = {s["key"] for s in out["sources"] if s["tracked"]}
    assert tracked == {"fd_scheduler", "autonomy", "follow_ups", "gmail_autoreplies", "gmail_sends", "agent_links",
                       "agent_chats", "agent_todos"}
    assert out["backfill"]["state"] == "idle" and set(out["backfill"]) == {"state", "done_at", "progress", "message"}
    assert all(set(s) == {"key", "label", "tracked", "on", "note"} for s in out["sources"])
    assert out["counts"] == {"needs_me": 0, "waiting": 0, "working": 0, "done": 0, "all": 0}
    assert set(out["sweep"]) == {"running", "last_at", "message"}


async def test_the_list_and_the_detail(api):
    await _seed_everything(api.env, api.env.alice)
    synced = await _sync(api)
    assert synced["ok"] is True and synced["new_events"] > 0 and synced["sweep"]["last_at"]
    assert synced["counts"]["all"] == 6
    out = await _tasks(api, view="all")
    assert set(out) == {"items", "total", "facets", "counts", "view"} and out["total"] == 6
    assert set(out["facets"]) == {"status", "channel", "agent", "direction", "kind", "evidence", "result", "party"}
    assert all(set(f) == {"id", "label", "count", *(("me",) if dim == "party" else ())}
               for dim, facet in out["facets"].items() for f in facet)
    for item in out["items"]:
        assert set(item) == ITEM_FIELDS and set(item["links"]) == LINK_FIELDS
        assert isinstance(item["channels"], list) and isinstance(item["flags"], dict)
    by_kind = {i["kind"]: i for i in out["items"]}
    proposal = by_kind["proposal"]
    assert proposal["lane"] == "needs_me"
    # v2: a ProposalView (contract 1) — an Autonomous Work proposal is shown, never edited from the board.
    assert set(proposal["proposal"]) == {"id", "status", "title", "kind", "source", "editable", "payload"}
    assert proposal["proposal"]["status"] == "awaiting_approval" and proposal["proposal"]["editable"] is False
    assert proposal["links"]["autonomy_id"] == proposal["proposal"]["id"]
    scheduled = by_kind["scheduled"]
    assert scheduled["links"]["scheduled_job_id"] and scheduled["proposal"] is None and scheduled["lane"] == "done"
    held = by_kind["held_reply"]
    assert held["links"]["gmail_thread_id"] and held["links"]["autoreply_id"]
    assert held["parties"][0]["address"] == "ana@x.co"

    detail = await _get(api, f"/fd/taskboard/tasks/{proposal['id']}")
    assert set(detail) == {"task", "events", "events_total", "links", "parties", "decisions", "proposals", "merged",
                           "next_step_kinds"}
    assert detail["task"]["id"] == proposal["id"] and detail["decisions"] == [] and detail["merged"] == []
    assert detail["events_total"] == len(detail["events"]) > 0
    assert [p["id"] for p in detail["proposals"]] == [proposal["proposal"]["id"]]
    assert detail["links"][0]["key"].startswith("autonomy:")
    assert all("user_id" not in e and {"at", "kind", "summary", "evidence"} <= set(e) for e in detail["events"])


# ── per user ──────────────────────────────────────────────────────────


async def test_another_user_gets_404s_and_empty_lists(api):
    await _seed_everything(api.env, api.env.alice)
    await _sync(api)
    mine = (await _tasks(api, view="all"))["items"]
    api.be(api.bob)
    synced = await _sync(api)
    assert synced["counts"]["all"] == 0
    for view in ("needs_me", "waiting", "working", "done", "all"):
        out = await _tasks(api, view=view)
        assert out["items"] == [] and out["total"] == 0 and all(not v for v in out["facets"].values())
    for item in mine:
        assert (await api.client.get(f"/fd/taskboard/tasks/{item['id']}")).status_code == 404
        assert (await _act(api, item["id"], "done")).status_code == 404
    assert (await _get(api, "/fd/taskboard/summary")) == {"needs_me": 0}
    api.be(api.alice)
    after = {i["id"]: i["status"] for i in (await _tasks(api, view="all"))["items"]}
    assert after == {i["id"]: i["status"] for i in mine}  # Bob's "done" changed nothing


# ── Approve / Reject ──────────────────────────────────────────────────


def _req(uid):
    return SimpleNamespace(state=SimpleNamespace(user_id=uid))


async def test_approve_and_reject_move_the_row(api, monkeypatch):
    deck = api.env
    one = deck.auto.add_action(deck.alice, kind="nudge", title="Nudge Ana", status="awaiting_approval")
    two = deck.auto.add_action(deck.alice, kind="nudge", title="Nudge Bo", status="awaiting_approval")
    await _sync(api)
    assert (await _get(api, "/fd/taskboard/summary")) == {"needs_me": 2}

    async def dispatched(uid, action, **kw):
        deck.auto.update_status(action["id"], "dispatched", ref_id="helper")
        return {"ok": True, "target": "helper", "note": ""}

    monkeypatch.setattr(fd_dispatch, "dispatch_action", dispatched)
    await autonomy_routes.approve_action_route(one["id"], _req(deck.alice), None)
    await autonomy_routes.reject_action_route(two["id"], _req(deck.alice), None)
    synced = await _sync(api)
    assert synced["counts"]["needs_me"] == 0 and synced["counts"]["working"] == 1
    items = {i["title"]: i for i in (await _tasks(api, view="all"))["items"]}
    assert (items["Nudge Ana"]["lane"], items["Nudge Ana"]["agent_name"]) == ("working", "Helper")
    assert (items["Nudge Bo"]["status"], items["Nudge Bo"]["closed_by"]) == ("dismissed", "user")
    assert items["Nudge Ana"]["proposal"] is None


async def test_reject_is_a_compare_and_set(api):
    deck = api.env
    row = deck.auto.add_action(deck.alice, kind="nudge", title="x", status="awaiting_approval")
    out = await autonomy_routes.reject_action_route(row["id"], _req(deck.alice), None)
    assert out["action"]["status"] == "rejected" and out["action"]["outcome"] == "fail"
    with pytest.raises(HTTPException) as exc:
        await autonomy_routes.reject_action_route(row["id"], _req(deck.alice), None)
    assert exc.value.status_code == 409
    assert deck.auto.reliability_for(deck.alice, "nudge", "general")["fails"] == 1  # counted once

    gone = deck.auto.add_action(deck.alice, kind="nudge", title="y", status="dispatched")
    with pytest.raises(HTTPException) as exc:
        await autonomy_routes.reject_action_route(gone["id"], _req(deck.alice), None)
    assert exc.value.status_code == 409 and "dispatched" in exc.value.detail
    assert deck.auto.get_action(gone["id"])["status"] == "dispatched"
    # Someone else's action is still not found.
    with pytest.raises(HTTPException) as exc:
        await autonomy_routes.reject_action_route(gone["id"], _req(deck.bob), None)
    assert exc.value.status_code == 404


async def test_a_local_action_is_nobodys_to_answer_with_sign_in_on(api, monkeypatch):
    """Rows in the shared ``local`` bucket (from before sign-in, or written
    with no user) are open to every caller only on a deck without sign-in."""
    deck = api.env
    row = deck.auto.add_action("", kind="nudge", title="x", status="awaiting_approval")
    fu = deck.events.add_follow_up("", summary="y", follow_up_at="2099-01-01T00:00:00+00:00")
    assert row["user_id"] == "local"
    for route in (autonomy_routes.approve_action_route, autonomy_routes.reject_action_route,
                  autonomy_routes.undo_action_route):
        with pytest.raises(HTTPException) as exc:
            await route(row["id"], _req(deck.alice), None)
        assert exc.value.status_code == 404
    for route in (autonomy_routes.done_follow_up_route, autonomy_routes.dismiss_follow_up_route):
        with pytest.raises(HTTPException) as exc:
            await route(fu["id"], _req(deck.alice), None)
        assert exc.value.status_code == 404
    assert deck.auto.get_action(row["id"])["status"] == "awaiting_approval"
    monkeypatch.setenv("FD_AUTH_ENABLED", "false")
    await autonomy_routes.reject_action_route(row["id"], _req(""), None)
    assert deck.auto.get_action(row["id"])["status"] == "rejected"


async def test_a_queued_action_can_still_be_rejected(api):
    deck = api.env
    row = deck.auto.add_action(deck.alice, kind="nudge", title="x", status="queued")
    await autonomy_routes.reject_action_route(row["id"], _req(deck.alice), None)
    assert deck.auto.get_action(row["id"])["status"] == "rejected"


# ── corrections ───────────────────────────────────────────────────────


async def test_corrections_lock_and_undo(api):
    deck = api.env
    deck.auto.add_action(deck.alice, kind="nudge", title="Nudge", status="awaiting_approval")
    await _sync(api)
    tid = (await _tasks(api))["items"][0]["id"]
    r = await _act(api, tid, "done")
    assert r.status_code == 200
    task = r.json()["task"]
    assert (task["status"], task["lane"], task["closed_by"], task["closed_by_party"]) == ("done", "done", "user", "Alice")
    assert "status" in task["user_locks"]
    # A later pass leaves the user's word alone.
    await _sync(api)
    assert (await _get(api, f"/fd/taskboard/tasks/{tid}"))["task"]["status"] == "done"
    task = (await _act(api, tid, "undo")).json()["task"]
    assert task["status"] == "open" and task["user_locks"] == {}
    until = time.time() + 3600
    task = (await _act(api, tid, "snooze", until)).json()["task"]
    assert task["snooze_until"] == until and task["lane"] == "snoozed"
    task = (await _act(api, tid, "rename", "Call Ana")).json()["task"]
    assert task["title"] == "Call Ana"
    task = (await _act(api, tid, "owes_next", "them")).json()["task"]
    assert task["owes_next"] == "them"
    events = (await _get(api, f"/fd/taskboard/tasks/{tid}"))["events"]
    assert [e["summary"] for e in events if e["kind"] == "correction"][-1] == "Next move: them"
    for action, value in (("snooze", "soon"), ("snooze", 1e300), ("snooze", 10 ** 400), ("owes_next", "the cat"),
                          ("ack_failed", None), ("teleport", None)):
        assert (await _act(api, tid, action, value)).status_code == 400


async def test_one_users_syncs_share_one_sweep_at_a_time(api, monkeypatch):
    """Calls while a sweep waits join it, calls while it reads get the next
    one, and a user's sweeps are at least the gap apart — so a burst of
    calls neither queues a worker thread each nor re-reads the ledgers each."""
    import asyncio
    import threading

    from captain_claw.flight_deck import taskboard_sources

    gate = threading.Event()
    calls: list[tuple[str, float]] = []
    real = taskboard_sources.run_sweep

    def sweep(store=None, user_id=None):
        calls.append((user_id, time.time()))
        gate.wait(5)
        return real(store, user_id)

    monkeypatch.setattr(taskboard_sources, "run_sweep", sweep)
    first = [asyncio.ensure_future(_sync(api)) for _ in range(5)]
    for _ in range(100):
        if calls:
            break
        await asyncio.sleep(0.01)
    later = [asyncio.ensure_future(_sync(api)) for _ in range(5)]  # while the first one reads
    await asyncio.sleep(0.05)
    assert len(calls) == 1
    gate.set()
    await asyncio.gather(*first, *later)
    assert [uid for uid, _ in calls] == [api.env.alice] * 2
    # One per gap: the next one waits for it, without a thread.
    monkeypatch.setattr(taskboard, "_SYNC_GAP_S", 0.3)
    calls.clear()
    await _sync(api)
    await _sync(api)
    assert len(calls) == 2 and calls[1][1] - calls[0][1] >= 0.25


async def test_the_detail_shows_the_latest_events_and_how_many_there_are(api):
    board, uid = api.env.board, api.env.alice
    tid, _ = board.ingest(uid, keys=["k"], create={"title": "Hourly"}, event={"source": "t", "source_ref": "0", "at": 1.0})
    for i in range(1, 520):
        board.ingest(uid, keys=["k"], event={"source": "t", "source_ref": str(i), "at": 1.0 + i})
    detail = await _get(api, f"/fd/taskboard/tasks/{tid}")
    assert detail["events_total"] == 520 and len(detail["events"]) == 500
    assert [e["source_ref"] for e in detail["events"]][-1] == "519"
    assert detail["events"][0]["source_ref"] == "20"


async def test_failed_needs_me_until_seen(api):
    deck = api.env
    row = deck.auto.add_action(deck.alice, kind="nudge", title="x", status="awaiting_approval")
    deck.auto.update_status(row["id"], "done", outcome="fail", outcome_note="agent unreachable")
    await _sync(api)
    item = (await _tasks(api))["items"][0]
    assert (item["status"], item["lane"]) == ("failed", "needs_me")
    task = (await _act(api, item["id"], "ack_failed")).json()["task"]
    assert task["lane"] == "other" and (await _get(api, "/fd/taskboard/summary"))["needs_me"] == 0


# ── search, facets, views ─────────────────────────────────────────────


async def test_search_is_folded_and_facets_follow_the_other_filters(api):
    board, uid = api.env.board, api.env.alice
    for key, title, channel, owes in (("a", "Ručak s Đurićem", "gmail", "me"), ("b", "Ćevapi for the team", "telegram", "me"),
                                      ("c", "Invoice", "gmail", "them")):
        board.ingest(uid, keys=[key], create={"title": title, "channel": channel, "owes_next": owes},
                     event={"source": "t", "source_ref": key})
    assert [i["title"] for i in (await _tasks(api, view="all", q="duric"))["items"]] == ["Ručak s Đurićem"]
    assert [i["title"] for i in (await _tasks(api, view="all", q="cevapi"))["items"]] == ["Ćevapi for the team"]
    out = await _tasks(api, view="all", channel="gmail")
    assert out["total"] == 2
    channels = {f["id"]: (f["label"], f["count"]) for f in out["facets"]["channel"]}
    assert channels == {"gmail": ("Gmail", 2), "telegram": ("Telegram", 1)}
    assert out["counts"] == {"needs_me": 1, "waiting": 1, "working": 0, "done": 0, "all": 2}
    out = await _tasks(api, view="all", channel="gmail,telegram", status="open")
    assert out["total"] == 3
    needs = await _tasks(api)  # Needs me is the default view
    assert needs["view"] == "needs_me" and needs["total"] == 2
    assert (await api.client.get("/fd/taskboard/tasks", params={"view": "later"})).status_code == 400


async def test_the_query_the_dialog_sends(api):
    """Dates as whole epoch seconds under from/to, people by their facet id,
    a sort, the page the list asks for."""
    board, uid = api.env.board, api.env.alice
    day = 86400.0
    now = time.time()
    for key, title, ago, who in (("a", "Old", 40, "ana@x.co"), ("b", "Last week", 5, "ana@x.co"),
                                 ("c", "Today", 0, "bo@y.org")):
        board.ingest(uid, keys=[key], create={"title": title, "channel": "gmail", "created_at": now - ago * day},
                     parties=[{"role": "requester", "kind": "email", "address": who}],
                     event={"source": "t", "source_ref": key, "at": now - ago * day})
    out = await _tasks(api, view="all", **{"from": int(now - 7 * day), "to": int(now + 60)}, sort="oldest",
                       limit=60, offset=0)
    assert [i["title"] for i in out["items"]] == ["Last week", "Today"]
    out = await _tasks(api, view="all", party="email:ana@x.co", sort="recent", limit=1)
    assert out["total"] == 2 and [i["title"] for i in out["items"]] == ["Last week"]
    assert {f["id"]: f["count"] for f in out["facets"]["party"]} == {"email:ana@x.co": 2, "email:bo@y.org": 1}
    more = await _tasks(api, view="all", party="email:ana@x.co", sort="recent", limit=1, offset=1)
    assert [i["title"] for i in more["items"]] == ["Old"]


# ── settings ──────────────────────────────────────────────────────────


async def test_settings_are_per_user(api):
    out = await _get(api, "/fd/taskboard/settings")
    assert (out["auto_close"], out["jev_per_day"], out["extract_per_day"]) == (True, 200, 100)
    assert set(out["passive"]) == {"telegram_inbox", "whatsapp_web", "gmail"}
    r = await api.client.put("/fd/taskboard/settings", json={"auto_close": False, "jev_per_day": 50})
    assert r.status_code == 200 and (r.json()["auto_close"], r.json()["jev_per_day"]) == (False, 50)
    assert (await api.client.put("/fd/taskboard/settings", json={"extract_per_day": 5000})).status_code == 400
    api.be(api.bob)
    assert (await _get(api, "/fd/taskboard/settings"))["jev_per_day"] == 200


# ── v1: settings, why, confirm / not done, the badge ──────────────────


async def test_settings_carry_todays_counts_jev_and_their_bounds(api, monkeypatch):
    from captain_claw.flight_deck import jev_routes, taskboard_store

    async def status(owner):
        return {"key_set": owner == api.env.alice, "source": "shared" if owner == api.env.alice else None}

    monkeypatch.setattr(jev_routes, "status_for_owner", status)
    api.env.board.take_daily(api.env.alice, "jev", 3)
    out = await _get(api, "/fd/taskboard/settings")
    assert out["counters"] == {"day": taskboard_store.utc_day(), "jev": 3, "extract": 0}
    assert out["jev"] == {"connected": True, "source": "team"}
    for body, code in (({"jev_per_day": 2000}, 200), ({"jev_per_day": 2001}, 400), ({"jev_per_day": -1}, 400),
                       ({"extract_per_day": 1000}, 200), ({"extract_per_day": 1001}, 400)):
        assert (await api.client.put("/fd/taskboard/settings", json=body)).status_code == code, body
    r = await api.client.put("/fd/taskboard/settings", json={"auto_close": False})
    assert set(r.json()) >= {"auto_close", "jev_per_day", "extract_per_day", "counters", "jev"}
    assert (r.json()["jev_per_day"], r.json()["extract_per_day"]) == (2000, 1000)
    api.be(api.bob)
    out = await _get(api, "/fd/taskboard/settings")
    assert out["jev"] == {"connected": False, "source": ""} and out["counters"]["jev"] == 0


def _maybe(board, uid, status="open", owes="me", **flags):
    """A turn's task as triage left it: *status*, who owes it, *flags*."""
    key = secrets.token_hex(8)
    tid, _ = board.ingest(uid, keys=[f"turn:{key}"], event={"source": "t", "source_ref": key},
                          create={"title": "Lunch?", "kind": "request", "status": status, "owes_next": owes,
                                  "flags": flags})
    return tid


async def test_why_lists_the_decisions_latest_first(api):
    board, uid = api.env.board, api.env.alice
    tid = _maybe(board, uid, status="maybe")
    fp = board.get_task(uid, tid)["fingerprint"]
    board.triage_write(uid, tid, fp, decision={"judge": "rule", "qset": "taskboard.rules.v1", "applied": "No change",
                                               "why": "The agent is still working on it."})
    board.triage_write(uid, tid, fp, decided=True, state="ok", decision={
        "judge": "jev", "qset": "taskboard.v1", "applied": "open → maybe",
        "why": "Jev isn't sure this is a task (52%), so it waits under Maybe tasks.",
        "answers": {"is_task": {"p": 0.52}, "owes_next": {"choice": "me", "p": 0.6}, "urgency": {"score": 1.0}}})
    out = await _get(api, f"/fd/taskboard/tasks/{tid}/why")
    assert set(out) == {"items", "summary"}
    assert [i["judge"] for i in out["items"]] == ["jev", "rule"]
    assert set(out["items"][0]) == {"at", "judge", "qset", "applied", "why", "answers"}
    assert out["items"][0]["answers"]["is_task"] == {"p": 0.52}
    assert "Maybe tasks" in out["summary"]
    # The user's own corrections are there too, as the user's.
    await _act(api, tid, "rename", "Lunch with Ana")
    out = await _get(api, f"/fd/taskboard/tasks/{tid}/why")
    assert (out["items"][0]["judge"], out["items"][0]["applied"]) == ("user", "Renamed to “Lunch with Ana”")
    assert [i["judge"] for i in out["items"]] == ["user", "jev", "rule"]
    api.be(api.bob)
    assert (await api.client.get(f"/fd/taskboard/tasks/{tid}/why")).status_code == 404


async def test_confirm_task_and_not_done(api):
    board, uid = api.env.board, api.env.alice
    maybe = _maybe(board, uid, status="maybe")
    task = (await _act(api, maybe, "confirm_task")).json()["task"]
    assert (task["status"], task["owes_next"], task["flags"]["low_confidence"]) == ("open", "agent", False)
    assert {"status", "low_confidence"} <= set(task["user_locks"])
    # Jev's later pass can't move it back.
    fp = board.get_task(uid, maybe)["fingerprint"]
    board.triage_write(uid, maybe, fp, fields={"status": "maybe", "flags": {"low_confidence": True}})
    assert (await _get(api, f"/fd/taskboard/tasks/{maybe}"))["task"]["status"] == "open"
    task = (await _act(api, maybe, "undo")).json()["task"]
    assert task["status"] == "maybe"
    low = _maybe(board, uid, low_confidence=True)
    task = (await _act(api, low, "confirm_task")).json()["task"]
    assert (task["status"], task["owes_next"], task["flags"]["low_confidence"]) == ("open", "me", False)
    assert (await _act(api, low, "confirm_task")).status_code == 400  # sure already

    probably = _maybe(board, uid, probably_done=True)
    task = (await _act(api, probably, "not_done")).json()["task"]
    assert (task["status"], task["flags"]["probably_done"]) == ("open", False)
    assert {"status", "probably_done"} <= set(task["user_locks"])
    closed = _maybe(board, uid, status="done")
    task = (await _act(api, closed, "not_done")).json()["task"]
    assert task["status"] == "open" and task["closed_at"] is None
    assert (await _act(api, _maybe(board, uid, owes="them"), "not_done")).status_code == 400
    api.be(api.bob)
    assert (await _act(api, low, "not_done")).status_code == 404


async def test_the_badge_leaves_out_low_confidence_tasks(api):
    board, uid = api.env.board, api.env.alice
    _maybe(board, uid)
    _maybe(board, uid, low_confidence=True)
    _maybe(board, uid, status="maybe")
    assert (await _get(api, "/fd/taskboard/summary")) == {"needs_me": 1}
    assert (await _get(api, "/fd/taskboard/status"))["counts"]["needs_me"] == 2  # the tab still lists it
    out = await _tasks(api, view="all", status="maybe")
    assert out["total"] == 1 and {f["id"] for f in out["facets"]["status"]} == {"open", "maybe"}


# ── v1.1: parties ─────────────────────────────────────────────────────


async def test_parties_say_which_is_me_and_name_each_person_once(api):
    """The board user's own party says ``me`` (the UI never needs a deck
    user's id to show it); a name read out of the text adds nothing when
    that person is there already — on a board written before extraction kept
    them apart too (an address wins over a bare name). A recorded party is
    always kept, and someone else's deck user is not ``me``."""
    board, uid = api.env.board, api.env.alice
    key = secrets.token_hex(8)
    tid, _ = board.ingest(uid, keys=[f"turn:{key}"], event={"source": "t", "source_ref": key}, create={
        "title": "Capital of Croatia", "kind": "request", "status": "done", "owes_next": "nobody"}, parties=[
        {"role": "requester", "kind": "fd_user", "address": uid, "display": "Ada Admin", "verified": True},
        {"role": "requester", "kind": "name", "address": "ada admin", "display": "Ada Admin"},
        {"role": "mentioned", "kind": "name", "address": "ana", "display": "Ana"},
        {"role": "recipient", "kind": "email", "address": "ana@x.co", "display": "ANA"},
        {"role": "cc", "kind": "fd_user", "address": api.env.bob, "display": "Bob", "verified": True}])
    want = [("fd_user", uid, True), ("email", "ana@x.co", False), ("fd_user", api.env.bob, False)]
    [item] = (await _tasks(api, view="all"))["items"]
    assert [(p["kind"], p["address"], p["me"]) for p in item["parties"]] == want
    detail = await _get(api, f"/fd/taskboard/tasks/{tid}")
    assert [(p["kind"], p["address"], p["me"]) for p in detail["parties"]] == want
    assert detail["task"]["parties"] == detail["parties"]


async def test_the_people_filter_says_which_value_is_me_whatever_the_page_holds(api):
    """Sign-in off (the board user is ``local``): the People filter marks
    their own value ``me`` even on a page none of whose tasks is theirs to
    say — so it reads "You", never "Someone else"."""
    api.be({})
    board = api.env.board
    for n, party in enumerate((
            {"role": "requester", "kind": "fd_user", "address": "local", "display": "You", "verified": True},
            {"role": "requester", "kind": "email", "address": "ana@x.co", "display": "Ana", "verified": True})):
        board.ingest("local", keys=[f"turn:t{n}"], event={"source": "t", "source_ref": f"t{n}"}, parties=[party],
                     create={"title": f"Task {n}", "kind": "request", "status": "open", "owes_next": "agent"})
    out = await _tasks(api, view="all", party="email:ana@x.co")
    assert [i["title"] for i in out["items"]] == ["Task 1"]
    assert not any(p.get("me") for i in out["items"] for p in i["parties"])
    # v2: Ana (a recorded address) is a person the board knows, by her party id.
    ana = f"party:{out['items'][0]['parties'][0]['party_id']}"
    assert {f["id"]: f["me"] for f in out["facets"]["party"]} == {"fd_user:local": True, ana: False}


async def test_why_never_says_the_same_reason_twice(api):
    """The live case through the route: answered in chat by rules."""
    board, uid = api.env.board, api.env.alice
    tid = _maybe(board, uid, status="done", owes="nobody", answered_in_chat=True)
    fp = board.get_task(uid, tid)["fingerprint"]
    board.triage_write(uid, tid, fp, decided=True, state="ok", decision={
        "judge": "rule", "qset": "taskboard.rules.v1", "applied": "open → done",
        "why": "The agent answered it in your chat, with no step failing, so it is done."})
    out = await _get(api, f"/fd/taskboard/tasks/{tid}/why")
    assert out["summary"] == "The agent answered it in your chat, with no step failing, so it is done."
