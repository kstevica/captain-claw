"""The Taskboard's v0 sources: each FD ledger read into the board of the user
its rows belong to, with the plan's status mapping — also for a row that
changed after it was first read (a proposal approved, a run finished, a held
reply sent) — a second sweep adding nothing, the user's corrections winning
over later passes, and a missing store or table skipped."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from captain_claw.config import AutonomousWorkConfig
from captain_claw.flight_deck import (
    autonomy,
    events,
    fd_scheduler,
    taskboard_sources,
    taskboard_store,
)
from captain_claw.flight_deck.db import FlightDeckDB
from captain_claw.flight_deck.taskboard_sources import run_sweep
from captain_claw.flight_deck.taskboard_store import TaskboardStore

HELPER_REF = "process:helper:0123456789abcdef"


@pytest.fixture()
async def env(tmp_path, monkeypatch):
    """A deck with sign-in: every source's store in tmp_path, two users, one
    agent ("helper", Alice's), an empty board."""
    monkeypatch.setenv("FD_AUTH_ENABLED", "true")
    sched = fd_scheduler.SchedulerStore(tmp_path / "scheduler.db")
    auto = autonomy.AutonomyStore(tmp_path / "autonomy.db")
    evs = events.EventsStore(tmp_path / "events.db")
    for mod, s in ((fd_scheduler, sched), (autonomy, auto), (events, evs)):
        monkeypatch.setattr(mod, "_STORE", s)
    monkeypatch.setattr(autonomy, "_status_changed", lambda uid: None)
    monkeypatch.setattr(autonomy, "global_defaults", lambda: AutonomousWorkConfig().model_dump())
    db = FlightDeckDB(tmp_path / "flight-deck.db")
    await db.init()
    alice = (await db.create_user("alice@x.co", "h", "Alice"))["id"]
    bob = (await db.create_user("bob@x.co", "h", "Bob"))["id"]
    board = TaskboardStore(tmp_path / "taskboard.db")
    monkeypatch.setattr(taskboard_store, "_STORE", board)
    monkeypatch.setattr(taskboard_store, "_DATA_DIR", tmp_path)
    monkeypatch.setattr(taskboard_sources, "_agent_index",
                        lambda: {"helper": {"owner": alice, "name": "Helper", "ref": HELPER_REF}})
    monkeypatch.setattr(taskboard_sources, "_STATE", {"running": False, "last_at": None, "message": ""})
    monkeypatch.setattr(taskboard_sources, "_USER_SYNCED", {})
    yield SimpleNamespace(sched=sched, auto=auto, events=evs, db=db, board=board, alice=alice, bob=bob,
                          tmp=tmp_path)
    board.close()
    await db.close()


def _one(env, uid, key):
    tid = env.board.task_for_key(uid, key)
    assert tid, f"no task for {key}"
    return env.board.get_task(uid, tid)


def _events(env, uid, key):
    return env.board.events(uid, env.board.task_for_key(uid, key))


def _lane(env, uid, key, proposals=()):
    return env.board.lane_of(uid, env.board.task_for_key(uid, key), proposal_tasks=list(proposals))


def _iso(**delta) -> str:
    return (datetime.now(UTC) + timedelta(**delta)).isoformat()


def _job(env, owner, **kw):
    fields = {"schedule": "every 1h", "agent_slug": "helper", "prompt": "Summarise my inbox",
              "name": "Inbox digest", "delivery_kind": "agent",
              "delivery": {"chat": True, "email": True, "email_to": "ana@x.co"}, **kw}
    return env.sched.create(owner_id=owner, **fields)


def _board_size(env) -> tuple[int, int]:
    conn = env.board._conn
    return (conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0],
            conn.execute("SELECT COUNT(*) FROM task_events").fetchone()[0])


# ── Scheduled tasks ───────────────────────────────────────────────────


async def test_a_job_is_one_standing_task_its_runs_move(env):
    job = _job(env, env.alice)
    run_sweep()
    key = f"sched:job:{job['id']}"
    task = _one(env, env.alice, key)
    assert (task["kind"], task["direction"], task["channel"]) == ("scheduled", "internal", "scheduled")
    assert (task["status"], task["owes_next"], task["agent_name"], task["title"]) == \
        ("open", "agent", "Helper", "Inbox digest")
    assert task["agent_owner_id"] == env.alice and task["agent_ref"] == HELPER_REF

    run = env.sched.start_run(job, "scheduled")
    run_sweep()
    assert _lane(env, env.alice, key) == "working"
    env.sched.finish_run(run, status="ok", result="Three new emails.", session_id="s-1",
                         deliveries=[{"via": "chat", "ok": True, "note": "posted"},
                                     {"via": "email", "ok": True, "note": "sent", "to": "ana@x.co"}])
    run_sweep()
    task = _one(env, env.alice, key)
    assert (task["status"], task["evidence"], task["closed_by"], task["result"]) == \
        ("done", "verified", "source", "success")
    assert task["result_text"] == "Three new emails." and task["closed_by_party"] == "Helper"
    assert env.board.task_for_key(env.alice, "session:helper:s-1") == task["id"]
    kinds = [e["kind"] for e in _events(env, env.alice, key)]
    assert kinds.count("delivery") == 2
    assert any(p["address"] == "ana@x.co" and p["role"] == "recipient"
               for p in env.board.parties_for(env.alice, [task["id"]])[task["id"]])

    # The next run opens it again; an error fails it.
    run2 = env.sched.start_run(job, "manual")
    run_sweep()
    task = _one(env, env.alice, key)
    assert (task["status"], task["owes_next"], task["evidence"], task["closed_at"]) == ("open", "agent", "none", None)
    env.sched.finish_run(run2, status="error:agent-not-running", result="", deliveries=[])
    run_sweep()
    task = _one(env, env.alice, key)
    assert (task["status"], task["result"]) == ("failed", "failed")
    assert _lane(env, env.alice, key) == "needs_me"


async def test_a_run_without_a_delivery_is_claimed_and_a_partial_one_is_partly_done(env):
    job = _job(env, env.alice)
    run = env.sched.start_run(job, "scheduled")
    env.sched.finish_run(run, status="ok", result="done", deliveries=[])
    run_sweep()
    assert _one(env, env.alice, f"sched:job:{job['id']}")["evidence"] == "claimed"
    run = env.sched.start_run(job, "scheduled")
    env.sched.finish_run(run, status="partial: email failed", result="x",
                         deliveries=[{"via": "chat", "ok": True}, {"via": "email", "ok": False, "note": "no Gmail"}])
    run_sweep()
    task = _one(env, env.alice, f"sched:job:{job['id']}")
    assert (task["status"], task["result"], task["evidence"]) == ("done", "partial", "verified")


async def test_a_run_that_never_began_leaves_the_task_as_it_was(env):
    job = _job(env, env.alice)
    key = f"sched:job:{job['id']}"
    run = env.sched.start_run(job, "scheduled")
    env.sched.finish_run(run, status="ok", result="r", deliveries=[{"via": "chat", "ok": True}])
    run_sweep()
    run = env.sched.start_run(job, "scheduled")
    env.sched.finish_run(run, status="skipped:limit", result="")
    run_sweep()
    task = _one(env, env.alice, key)
    assert (task["status"], task["evidence"], _lane(env, env.alice, key)) == ("done", "verified", "done")
    assert _events(env, env.alice, key)[-1]["summary"] == "Run skipped: the daily allowance is spent"
    # Seen going, then skipped after all: no agent is on it.
    run = env.sched.start_run(job, "scheduled")
    run_sweep()
    assert _lane(env, env.alice, key) == "working"
    env.sched.finish_run(run, status="skipped:quiet", result="")
    run_sweep()
    assert _one(env, env.alice, key)["owes_next"] == "nobody" and _lane(env, env.alice, key) != "working"
    # A muted delivery: the run went through, only its message was held back.
    run = env.sched.start_run(job, "scheduled")
    env.sched.finish_run(run, status="skipped:muted-or-not-allowed", result="r",
                         deliveries=[{"via": "whatsapp", "ok": False, "note": "skipped:muted-or-not-allowed"}])
    run_sweep()
    task = _one(env, env.alice, key)
    assert (task["status"], task["result"], task["evidence"]) == ("done", "partial", "claimed")


async def test_done_on_a_scheduled_task_lasts_until_its_next_run(env):
    job = _job(env, env.alice)
    key = f"sched:job:{job['id']}"
    run = env.sched.start_run(job, "scheduled")
    env.sched.finish_run(run, status="error:x", result="", deliveries=[])
    run_sweep()
    env.board.correct(env.alice, env.board.task_for_key(env.alice, key), "done")
    run_sweep()
    assert _one(env, env.alice, key)["status"] == "done"
    # The next run is the job's again — and its failure needs the user.
    run = env.sched.start_run(job, "scheduled")
    env.sched.finish_run(run, status="error:y", result="", deliveries=[])
    run_sweep()
    task = _one(env, env.alice, key)
    assert (task["status"], task["user_locks"]) == ("failed", "{}")
    assert _lane(env, env.alice, key) == "needs_me"


async def test_a_jobs_next_run_doesnt_close_it_over_live_work_merged_in(env):
    """A merge never hides live work: the run's end waits, the next move the
    user's — and with the work out, the task stands where that run left it."""
    job = _job(env, env.alice)
    key = f"sched:job:{job['id']}"
    env.sched.finish_run(env.sched.start_run(job, "scheduled"), status="ok", result="r", deliveries=[])
    run_sweep()
    assert _one(env, env.alice, key)["status"] == "done"
    request, _ = env.board.ingest(
        env.alice, keys=["turn:r"], create={"title": "The digest, please", "kind": "request", "channel": "whatsapp",
                                            "status": "open", "owes_next": "agent", "triage_state": "ok"},
        event={"source": "agent_turns", "source_ref": "r:open", "kind": "received", "actor_kind": "party"})
    tid = env.board.task_for_key(env.alice, key)
    env.board.merge(env.alice, request, tid)
    assert _lane(env, env.alice, key) == "needs_me"

    run = env.sched.start_run(job, "scheduled")
    run_sweep()
    assert _lane(env, env.alice, key) == "working"
    env.sched.finish_run(run, status="error:x", result="", deliveries=[])
    run_sweep()
    task = _one(env, env.alice, key)
    assert (task["status"], task["owes_next"], task["result"]) == ("open", "me", "") and task["user_locks"] == "{}"
    assert _lane(env, env.alice, key) == "needs_me"

    env.board.unmerge(env.alice, request)
    task = _one(env, env.alice, key)
    assert (task["status"], task["owes_next"], task["result"], task["closed_by"]) == ("failed", "nobody", "failed", "source")


async def test_a_job_switched_off_or_deleted_before_it_ran_leaves_agents_working(env):
    off, gone, busy = _job(env, env.alice), _job(env, env.alice, name="Once"), _job(env, env.alice, name="Busy")
    env.sched.start_run(busy, "manual")
    done = _job(env, env.alice, name="Done")
    run = env.sched.start_run(done, "scheduled")
    env.sched.finish_run(run, status="ok", result="r", deliveries=[])
    run_sweep()
    assert _lane(env, env.alice, f"sched:job:{off['id']}") == "working"
    for job in (off, busy, done):
        env.sched.update(job["id"], enabled=0)
    env.sched.delete(gone["id"])
    run_sweep()
    assert _lane(env, env.alice, f"sched:job:{off['id']}") == "other"
    task = _one(env, env.alice, f"sched:job:{gone['id']}")
    assert (task["status"], task["closed_by"]) == ("dismissed", "source")
    assert _lane(env, env.alice, f"sched:job:{busy['id']}") == "working"  # its run will say
    assert _events(env, env.alice, f"sched:job:{done['id']}")[-1]["source_ref"].endswith(":end")  # untouched
    size = _board_size(env)
    run_sweep()
    assert _board_size(env) == size
    env.sched.update(off["id"], enabled=1)
    run_sweep()
    assert _lane(env, env.alice, f"sched:job:{off['id']}") == "working"
    summaries = [e["summary"] for e in _events(env, env.alice, f"sched:job:{off['id']}")]
    assert summaries[-2:] == ["Switched off", "Switched on again"]


async def test_a_scheduler_file_not_there_is_not_every_job_deleted(env, monkeypatch):
    job = _job(env, env.alice)
    run_sweep()
    monkeypatch.setattr(fd_scheduler, "_STORE", None)
    monkeypatch.setattr(fd_scheduler, "_db_path", lambda: env.tmp / "nowhere" / "scheduler.db")
    run_sweep()
    assert _lane(env, env.alice, f"sched:job:{job['id']}") == "working"


async def test_a_system_job_is_on_nobodys_board(env):
    _job(env, "")
    run_sweep()
    assert _board_size(env) == (0, 0)


# ── Autonomous Work ───────────────────────────────────────────────────


async def test_a_proposal_approved_after_it_was_read_moves_lanes(env):
    row = env.auto.add_action(env.alice, kind="nudge", title="Remind Ana about Q3", rationale="She asked twice",
                              status="awaiting_approval")
    run_sweep()
    key = f"autonomy:{row['id']}"
    task = _one(env, env.alice, key)
    assert (task["kind"], task["status"], task["owes_next"]) == ("proposal", "open", "me")
    assert _lane(env, env.alice, key) == "needs_me"
    # Approved (dispatch hands it to the agent), then done.
    assert env.auto.claim_action(row["id"], ("awaiting_approval",), "dispatched")
    env.auto.update_status(row["id"], "dispatched", ref_id="helper")
    run_sweep()
    task = _one(env, env.alice, key)
    assert (task["status"], task["owes_next"], task["agent_slug"]) == ("open", "agent", "helper")
    assert _lane(env, env.alice, key) == "working"
    env.auto.update_status(row["id"], "done", outcome="success", outcome_note="Nudged Ana")
    run_sweep()
    task = _one(env, env.alice, key)
    assert (task["status"], task["evidence"], task["result"], task["result_text"]) == \
        ("done", "claimed", "success", "Nudged Ana")
    before = _board_size(env)
    run_sweep()
    run_sweep()
    assert _board_size(env) == before


async def test_a_proposal_held_back_for_approval_needs_me_again(env):
    row = env.auto.add_action(env.alice, kind="tool_action", title="Draft a reply", status="awaiting_approval")
    run_sweep()
    env.auto.update_status(row["id"], "dispatched")
    run_sweep()
    env.auto.update_status(row["id"], "awaiting_approval", outcome_note="held: needs a human click")
    run_sweep()
    task = _one(env, env.alice, f"autonomy:{row['id']}")
    assert (task["status"], task["owes_next"]) == ("open", "me")
    states = [e["refs"].get("state") for e in _events(env, env.alice, f"autonomy:{row['id']}")]
    assert states == [None, "awaiting_approval", "dispatched", "awaiting_approval"]


@pytest.mark.parametrize("status,outcome,expect", [
    ("rejected", "fail", ("dismissed", "user")),
    ("expired", None, ("dismissed", "source")),
    ("done", "fail", ("failed", "source")),
])
async def test_settled_proposals(env, status, outcome, expect):
    row = env.auto.add_action(env.alice, kind="nudge", title="x", status="awaiting_approval")
    run_sweep()
    env.auto.update_status(row["id"], status, outcome=outcome)
    run_sweep()
    task = _one(env, env.alice, f"autonomy:{row['id']}")
    assert (task["status"], task["closed_by"]) == expect


async def test_an_action_about_a_follow_up_is_an_event_on_its_task(env):
    fu = env.events.add_follow_up(env.alice, summary="Ana wants the Q3 numbers", detail="Ana, by Friday",
                                  follow_up_at=_iso(days=2))
    row = env.auto.add_action(env.alice, kind="nudge", title="Remind about Q3", status="awaiting_approval",
                              payload={"follow_up_id": fu["id"]})
    run_sweep()
    task = _one(env, env.alice, f"fu:{fu['id']}")
    assert env.board.task_for_key(env.alice, f"autonomy:{row['id']}") == task["id"]
    assert (task["kind"], task["owes_next"]) == ("follow_up", "them")  # its own status, not the action's
    assert any(e["kind"] == "proposal" for e in env.board.events(env.alice, task["id"]))
    assert _lane(env, env.alice, f"fu:{fu['id']}", proposals=[task["id"]]) == "needs_me"
    assert _board_size(env)[0] == 1


async def test_undo_after_the_source_moved_shows_where_the_source_is(env):
    row = env.auto.add_action(env.alice, kind="nudge", title="P", status="awaiting_approval")
    run_sweep()
    key = f"autonomy:{row['id']}"
    tid = env.board.task_for_key(env.alice, key)
    env.board.correct(env.alice, tid, "done")
    env.auto.update_status(row["id"], "rejected")
    run_sweep()
    assert env.board.get_task(env.alice, tid)["status"] == "done"
    env.board.correct(env.alice, tid, "undo")
    run_sweep()
    task = env.board.get_task(env.alice, tid)
    assert (task["status"], task["owes_next"], task["closed_by"]) == ("dismissed", "nobody", "user")
    assert _lane(env, env.alice, key) == "other"


async def test_a_tracked_open_loop_is_one_task_its_follow_up(env):
    fu = env.events.add_follow_up(env.alice, summary="Ana wants Q3", follow_up_at=_iso(days=3))
    env.auto.add_action(env.alice, kind="track", title="Ana wants Q3", status="done",
                        payload={"summary": "Ana wants Q3", "follow_up_days": 3, "follow_up_id": ""})
    again = env.auto.add_action(env.alice, kind="track", title="Ana wants Q3", status="done",
                                payload={"follow_up_id": fu["id"], "follow_up_days": 2})
    run_sweep()
    assert _board_size(env)[0] == 1
    assert env.board.task_for_key(env.alice, f"autonomy:{again['id']}") == \
        env.board.task_for_key(env.alice, f"fu:{fu['id']}")


# ── Follow-ups ────────────────────────────────────────────────────────


async def test_follow_ups_wait_then_fall_due_then_close(env):
    ev = env.events.add_event(env.alice, source="gmail", event_type="new_email", summary="Q3?",
                              metadata={"thread_id": "t-9", "from": '"Ana Kovač" <ana@x.co>'}, dedup_key="m1")
    fu = env.events.add_follow_up(env.alice, summary="Ana wants Q3", origin_event_id=ev["id"],
                                  source="gmail", follow_up_at=_iso(days=1))
    run_sweep()
    key = f"fu:{fu['id']}"
    task = _one(env, env.alice, key)
    assert (task["status"], task["owes_next"], task["channel"], task["direction"]) == \
        ("open", "them", "gmail", "received")
    assert env.board.task_for_key(env.alice, "gmail:thread:t-9") == task["id"]
    party = env.board.parties_for(env.alice, [task["id"]])[task["id"]][0]
    assert (party["role"], party["address"], party["display"], party["verified"]) == \
        ("requester", "ana@x.co", "Ana Kovač", True)
    assert _lane(env, env.alice, key) == "waiting"
    # Due now (its row unchanged otherwise).
    env.events.touch_follow_up(fu["id"], follow_up_at=_iso(minutes=-5))
    run_sweep()
    assert (_one(env, env.alice, key)["owes_next"], _lane(env, env.alice, key)) == ("me", "needs_me")
    env.events.mark_follow_up(fu["id"], "stale")
    run_sweep()
    assert json.loads(_one(env, env.alice, key)["flags"])["stale"] is True
    env.events.mark_follow_up(fu["id"], "done")
    run_sweep()
    assert _one(env, env.alice, key)["status"] == "done"


async def test_a_dismissed_follow_up_is_dismissed(env):
    fu = env.events.add_follow_up(env.alice, summary="x", follow_up_at=_iso(days=1))
    env.events.mark_follow_up(fu["id"], "dismissed")
    run_sweep()
    task = _one(env, env.alice, f"fu:{fu['id']}")
    assert (task["status"], task["channel"], task["direction"]) == ("dismissed", "autonomy", "internal")


# ── Gmail auto-replies and sends ──────────────────────────────────────


async def _held(env, owner, msg="m-1", thread="t-1"):
    rid = await env.db.claim_autoreply(owner, msg, thread)
    await env.db.update_autoreply(rid, owner, from_addr="ana@x.co", subject="Opening hours?",
                                  topic_name="Opening hours", agent_ref=HELPER_REF, agent_name="Helper")
    assert await env.db.cas_autoreply(rid, owner, "new", "held")
    return rid


async def test_a_held_reply_needs_me_until_it_is_sent(env):
    rid = await _held(env, env.alice)
    run_sweep()
    key = "gmail:thread:t-1"
    task = _one(env, env.alice, key)
    assert (task["kind"], task["direction"], task["channel"], task["owes_next"]) == \
        ("held_reply", "received", "gmail", "me")
    assert task["agent_slug"] == "helper" and _lane(env, env.alice, key) == "needs_me"
    assert env.board.task_for_key(env.alice, f"autoreply:{rid}") == task["id"]
    assert await env.db.cas_autoreply(rid, env.alice, "held", "approved", approved_by=env.alice)
    run_sweep()
    assert _lane(env, env.alice, key) == "working"
    # The send fails and the reply is held again: it needs me again.
    assert await env.db.cas_autoreply(rid, env.alice, "approved", "held")
    run_sweep()
    assert _lane(env, env.alice, key) == "needs_me"
    assert await env.db.cas_autoreply(rid, env.alice, "held", "sending")
    assert await env.db.cas_autoreply(rid, env.alice, "sending", "sent", sent_message_id="g-2",
                                      reply_text="We open at 9.")
    run_sweep()
    task = _one(env, env.alice, key)
    assert (task["status"], task["evidence"], task["result_text"]) == \
        ("done", "verified", "Auto-replied (Opening hours)")


async def test_a_failed_auto_reply_fails_and_one_dismissed_unseen_stays_off(env):
    rid = await env.db.claim_autoreply(env.alice, "m-2", "t-2")
    await env.db.update_autoreply(rid, env.alice, topic_id="hours", topic_name="Opening hours")
    assert await env.db.cas_autoreply(rid, env.alice, "new", "failed", reason="writer_error")
    rid3 = await env.db.claim_autoreply(env.alice, "m-3", "t-3")
    assert await env.db.cas_autoreply(rid3, env.alice, "new", "dismissed")
    run_sweep()
    assert _one(env, env.alice, "gmail:thread:t-2")["status"] == "failed"
    assert env.board.task_for_key(env.alice, "gmail:thread:t-3") is None


async def test_an_email_that_never_matched_a_topic_is_not_a_task(env):
    rid = await env.db.claim_autoreply(env.alice, "m-x", "t-x")
    assert await env.db.cas_autoreply(rid, env.alice, "new", "matching")
    assert await env.db.cas_autoreply(rid, env.alice, "matching", "failed", reason="classifier_unavailable")
    rid = await env.db.claim_autoreply(env.alice, "m-y", "t-y")
    assert await env.db.cas_autoreply(rid, env.alice, "new", "failed", reason="fetch")
    run_sweep()
    assert _board_size(env) == (0, 0)


async def test_an_auto_reply_is_sent_once_and_what_it_said_stays_off_the_board(env):
    rid = await _held(env, env.alice)
    assert await env.db.cas_autoreply(rid, env.alice, "held", "approved", approved_by=env.alice)
    assert await env.db.cas_autoreply(rid, env.alice, "approved", "sending")
    # Gmail's send is recorded before the row says sent.
    await env.db.add_gmail_send(env.alice, "Helper", to_addrs="ana@x.co", subject="Re: Opening hours?",
                                gmail_message_id="g-2", thread_id="t-1", origin="autoreply_approved")
    run_sweep()
    assert await env.db.cas_autoreply(rid, env.alice, "sending", "sent", sent_message_id="g-2",
                                      reply_text="We open at 9, Ana.")
    run_sweep()
    events = _events(env, env.alice, "gmail:thread:t-1")
    assert [e["kind"] for e in events].count("sent") == 1
    task = _one(env, env.alice, "gmail:thread:t-1")
    assert task["result_text"] == "Auto-replied (Opening hours)" and "open at 9" not in task["search_text"]
    assert not any("open at 9" in e["summary"] for e in events)


async def test_a_send_on_a_known_thread_is_evidence_and_a_new_one_its_own_task(env, monkeypatch):
    await _held(env, env.alice)
    run_sweep()
    await env.db.add_gmail_send(env.alice, "helper", to_addrs="Ana <ana@x.co>", subject="Re: Opening hours?",
                                gmail_message_id="g-1", thread_id="t-1")
    await env.db.add_gmail_send(env.alice, "helper", to_addrs="bo@y.org", cc_addrs="cy@z.org",
                                subject="Invoice", gmail_message_id="g-5", thread_id="t-5")
    run_sweep()
    held = _one(env, env.alice, "gmail:thread:t-1")
    assert (held["kind"], held["status"], held["evidence"]) == ("held_reply", "open", "verified")
    assert any(e["kind"] == "sent" and e["evidence"] == "recorded" for e in _events(env, env.alice, "gmail:thread:t-1"))
    # An agent's send on a thread the board doesn't have waits a while for
    # the tool call that would put it on its chat's task — then is its own.
    assert env.board.task_for_key(env.alice, "gmail:thread:t-5") is None
    monkeypatch.setattr(taskboard_sources, "_SEND_WAIT_S", 0.0)   # ten minutes on
    run_sweep()
    sent = _one(env, env.alice, "gmail:thread:t-5")
    assert (sent["kind"], sent["direction"], sent["status"], sent["evidence"], sent["closed_by"]) == \
        ("sent", "sent", "done", "verified", "evidence")
    roles = {(p["role"], p["address"]) for p in env.board.parties_for(env.alice, [sent["id"]])[sent["id"]]}
    assert roles == {("recipient", "bo@y.org"), ("cc", "cy@z.org")}
    assert sent["agent_name"] == "Helper"


async def test_an_agents_fresh_send_waits_and_the_cursor_stays_on_it(env, monkeypatch):
    """An agent's send on a thread the board doesn't have waits for its turn's
    tool call (ten minutes at most) — read again every pass, the cursor never
    past it; one with no thread can't be named by a call, so it doesn't wait."""
    old = await env.db.add_gmail_send(env.alice, "helper", to_addrs="a@x.co", subject="Old", thread_id="t-old")
    _sent_ago(env, old, minutes=11)
    fresh = await env.db.add_gmail_send(env.alice, "helper", to_addrs="b@x.co", subject="Fresh", thread_id="t-new")
    await env.db.add_gmail_send(env.alice, "helper", to_addrs="c@x.co", subject="No thread", gmail_message_id="g-9")
    run_sweep()
    assert _one(env, env.alice, "gmail:thread:t-old")["kind"] == "sent"
    assert _one(env, env.alice, "gmail:msg:g-9")["kind"] == "sent"
    assert env.board.task_for_key(env.alice, "gmail:thread:t-new") is None
    conn = sqlite3.connect(env.tmp / "flight-deck.db")
    (fresh_at,) = conn.execute("SELECT created_at FROM gmail_sends WHERE id = ?", (fresh,)).fetchone()
    conn.close()
    assert env.board.get_cursor("gmail_sends") == fresh_at
    before = _board_size(env)
    run_sweep()
    assert _board_size(env) == before   # read again, still waiting, nothing added
    monkeypatch.setattr(taskboard_sources, "_SEND_WAIT_S", 0.0)   # ten minutes on
    run_sweep()
    assert _one(env, env.alice, "gmail:thread:t-new")["kind"] == "sent"


# ── Linked systems ────────────────────────────────────────────────────


async def test_link_messages_are_done_when_they_reach_the_system(env):
    link = await env.db.create_agent_link(env.alice, name="Shop", agent_slug="helper", instructions="",
                                          key_hash="kh", key_hint="abcd")
    reply = await env.db.add_link_event(link["id"], "reply", "m-1", {"text": "Your order shipped", "files": []})
    await env.db.add_link_event(link["id"], "error", "m-2", {"text": "Agent error"})
    push = await env.db.add_link_event(link["id"], "push", "", {"text": "Stock is low"})
    run_sweep()
    key = f"link:{link['id']}:m-1"
    task = _one(env, env.alice, key)
    assert (task["kind"], task["channel"], task["status"], task["owes_next"], task["title"]) == \
        ("link", "link", "open", "agent", "Message from Shop")
    assert _one(env, env.alice, f"link:{link['id']}:m-2")["status"] == "failed"
    pushed = _one(env, env.alice, f"link:{link['id']}:push-{push['seq']}")
    assert (pushed["direction"], pushed["title"]) == ("sent", "Message to Shop")
    await env.db.mark_link_event(reply["seq"], "socket")
    run_sweep()
    task = _one(env, env.alice, key)
    assert (task["status"], task["evidence"], task["closed_by"]) == ("done", "verified", "evidence")


async def test_what_a_link_message_said_stays_off_the_board(env):
    link = await env.db.create_agent_link(env.alice, name="Shop", agent_slug="helper", instructions="",
                                          key_hash="kh", key_hint="abcd")
    await env.db.add_link_event(link["id"], "reply", "m-1", {"text": "Order 4411 for Ana Kovač shipped"})
    await env.db.add_link_event(link["id"], "push", "", {"text": "Ana's card 4411 expired"})
    run_sweep()
    task = _one(env, env.alice, f"link:{link['id']}:m-1")
    assert task["result_text"] == "Answered Shop"
    board = env.board._conn
    assert board.execute("SELECT COUNT(*) FROM tasks WHERE search_text LIKE '%4411%' "
                         "OR result_text LIKE '%4411%'").fetchone()[0] == 0
    assert board.execute("SELECT COUNT(*) FROM task_events WHERE summary LIKE '%4411%' "
                         "OR refs LIKE '%4411%'").fetchone()[0] == 0


# ── The sweep ─────────────────────────────────────────────────────────


def _sent_ago(env, send_id: str, *, minutes: float) -> None:
    """A gmail_sends row as if it went out *minutes* ago."""
    conn = sqlite3.connect(env.tmp / "flight-deck.db")
    conn.execute("UPDATE gmail_sends SET created_at = ? WHERE id = ?", (_iso(minutes=-minutes), send_id))
    conn.commit()
    conn.close()


async def _seed_everything(env, owner):
    job = _job(env, owner)
    run = env.sched.start_run(job, "scheduled")
    env.sched.finish_run(run, status="ok", result="r", deliveries=[{"via": "chat", "ok": True}])
    env.auto.add_action(owner, kind="nudge", title="Nudge", status="awaiting_approval")
    env.events.add_follow_up(owner, summary="Follow up", follow_up_at=_iso(days=1))
    await _held(env, owner, msg=f"m-{owner}", thread=f"t-{owner}")
    sent = await env.db.add_gmail_send(owner, "helper", to_addrs="x@y.z", subject="s", gmail_message_id=f"g-{owner}",
                                       thread_id=f"t2-{owner}")
    _sent_ago(env, sent, minutes=11)  # past the wait for its tool call: a "sent" task of its own
    link = await env.db.create_agent_link(owner, name="Shop", agent_slug="", instructions="",
                                          key_hash=f"kh-{owner}", key_hint="abcd")
    await env.db.add_link_event(link["id"], "reply", "m-1", {"text": "ok"})


async def test_a_second_sweep_adds_nothing(env):
    await _seed_everything(env, env.alice)
    first = run_sweep()
    assert set(first["new_events"]) == set(taskboard_sources.SOURCE_LABELS) and first["problems"] == []
    size = _board_size(env)
    assert size[0] == 6
    again = run_sweep()
    assert again["new_events"] == {} and _board_size(env) == size


async def test_every_row_goes_to_its_own_users_board_only(env):
    await _seed_everything(env, env.alice)
    await _seed_everything(env, env.bob)
    run_sweep()
    a, b = env.board.query(env.alice, view="all"), env.board.query(env.bob, view="all")
    assert a["total"] == b["total"] == 6
    assert not {r["id"] for r in a["rows"]} & {r["id"] for r in b["rows"]}


async def test_one_users_sync_reads_only_their_rows_and_leaves_the_cursors(env):
    await _seed_everything(env, env.alice)
    await _seed_everything(env, env.bob)
    run_sweep(user_id=env.bob)
    assert env.board.query(env.alice, view="all")["total"] == 0
    assert env.board.query(env.bob, view="all")["total"] == 6
    assert env.board.get_cursor("gmail_sends") == ""
    run_sweep()
    assert env.board.query(env.alice, view="all")["total"] == 6
    assert taskboard_sources.sweep_state(env.bob)["last_at"] is not None


async def test_a_deleted_users_board_is_not_made_again(env):
    _job(env, env.bob)
    env.events.add_follow_up(env.bob, summary="Follow up", follow_up_at=_iso(days=1))
    env.auto.add_action(env.bob, kind="nudge", title="Nudge", status="awaiting_approval")
    run_sweep()
    assert env.board.counts(env.bob)["all"] == 3
    await env.db.delete_user(env.bob)
    env.board.delete_user(env.bob)
    run_sweep()
    run_sweep(user_id=env.bob)
    assert _board_size(env) == (0, 0)


async def test_the_users_corrections_win_over_a_later_pass(env):
    job = _job(env, env.alice)
    run_sweep()
    key = f"sched:job:{job['id']}"
    tid = env.board.task_for_key(env.alice, key)
    env.board.correct(env.alice, tid, "rename", "My digest")
    env.board.correct(env.alice, tid, "not_task")
    env.sched.update(job["id"], name="Renamed at the source")
    run = env.sched.start_run(job, "scheduled")
    env.sched.finish_run(run, status="error:x", result="", deliveries=[])
    run_sweep()
    task = env.board.get_task(env.alice, tid)
    assert (task["title"], task["status"], task["closed_by"]) == ("My digest", "not_task", "user")
    assert any(e["source_ref"].endswith(":end") for e in env.board.events(env.alice, tid))  # still recorded


async def test_a_store_or_table_that_is_not_there_is_skipped(env, monkeypatch):
    # No events.db on this deck, and a flight-deck.db without the tables.
    monkeypatch.setattr(events, "_STORE", None)
    monkeypatch.setattr(events, "_db_path", lambda: env.tmp / "nowhere" / "events.db")
    empty = env.tmp / "empty"
    empty.mkdir()
    sqlite3.connect(empty / "flight-deck.db").close()
    monkeypatch.setattr(taskboard_store, "_DATA_DIR", empty)
    job = _job(env, env.alice)
    res = run_sweep()
    assert res["problems"] == [] and res["message"] == ""
    assert env.board.task_for_key(env.alice, f"sched:job:{job['id']}")
    assert not (env.tmp / "nowhere").exists()


async def test_a_failing_source_is_logged_and_the_rest_go_on(env, monkeypatch):
    def boom(p):
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(taskboard_sources, "ADAPTERS",
                        (("fd_scheduler", boom), *taskboard_sources.ADAPTERS[1:]))
    env.auto.add_action(env.alice, kind="nudge", title="x", status="awaiting_approval")
    res = run_sweep()
    assert res["problems"] == ["fd_scheduler"] and "Scheduled tasks" in res["message"]
    assert taskboard_sources.sweep_state()["message"] == res["message"]
    assert env.board.query(env.alice, view="all")["total"] == 1
    # A user's own sync that couldn't read a source says so on their board only.
    monkeypatch.setattr(taskboard_sources, "_STATE", {"running": False, "last_at": None, "message": ""})
    mine = run_sweep(user_id=env.alice)
    assert taskboard_sources.sweep_state(env.alice)["message"] == mine["message"] != ""
    assert taskboard_sources.sweep_state(env.bob)["message"] == ""


async def test_without_sign_in_every_row_is_the_local_users(env, monkeypatch):
    monkeypatch.setenv("FD_AUTH_ENABLED", "false")
    _job(env, "")
    env.auto.add_action("", kind="nudge", title="x", status="awaiting_approval")
    run_sweep()
    assert env.board.query("local", view="all")["total"] == 2


# ── A job merged into another task speaks for itself ──────────────────


def _their_request(env) -> str:
    """Ana's request, in Needs me."""
    tid, _ = env.board.ingest(
        env.alice, keys=["turn:ana"],
        parties=[{"role": "requester", "kind": "waid", "address": "38591", "display": "Ana", "verified": True}],
        create={"title": "Ana: book the table", "ask": "Ana: book the table", "kind": "request", "direction": "received",
                "channel": "whatsapp", "status": "open", "owes_next": "me", "triage_state": "ok",
                "source": "agent_turns", "source_ref": "ana", "agent_slug": "helper",
                "flags": {"contains_instructions": False}},
        event={"source": "agent_turns", "source_ref": "ana:open", "kind": "received", "channel": "whatsapp",
               "actor_kind": "party", "actor": "Ana", "summary": "Book the table"})
    return tid


async def test_a_jobs_runs_never_move_the_live_request_it_was_merged_into(env):
    """Its next run goes on the timeline: the request stays where it stood, its
    title and result its own — and the job comes back out knowing how it ran."""
    job = _job(env, env.alice)
    run = env.sched.start_run(job, "scheduled")
    env.sched.finish_run(run, status="ok", result="Digest 1.", session_id="s-1", deliveries=[])
    run_sweep()
    jid = env.board.task_for_key(env.alice, f"sched:job:{job['id']}")
    tid = _their_request(env)
    env.board.merge(env.alice, jid, tid)

    def still_itself():
        t = env.board.get_task(env.alice, tid)
        assert (t["title"], t["ask"], t["status"], t["owes_next"], t["result_text"]) == (
            "Ana: book the table", "Ana: book the table", "open", "me", "")
        assert env.board.lane_of(env.alice, tid) == "needs_me"

    run_sweep()
    still_itself()
    run = env.sched.start_run(job, "scheduled")
    run_sweep()
    still_itself()
    env.sched.finish_run(run, status="ok", result="Digest 2.", session_id="s-2", deliveries=[])
    run_sweep()
    still_itself()
    assert any("Digest 2." in e["summary"] or e["source"] == "fd_scheduler" for e in env.board.events(env.alice, tid))
    assert env.board.get_task(env.alice, jid)["result_text"] == "Digest 2."
    env.board.unmerge(env.alice, jid)
    still_itself()
    back = env.board.get_task(env.alice, jid)
    assert (back["status"], back["result_text"]) == ("done", "Digest 2.")


@pytest.mark.parametrize("kind,key,patch", [
    ("todo", "todo:helper:1", {"status": "done", "owes_next": "nobody", "evidence": "claimed", "result": "success",
                               "closed_by": "source"}),
    ("follow_up", "fu:1", {"status": "dismissed", "owes_next": "nobody", "closed_by": "source"}),
    ("scheduled", "sched:job:9", {"owes_next": "nobody"}),
])
async def test_whatever_else_a_merged_source_says_is_about_itself(env, kind, key, patch):
    src, _ = env.board.ingest(
        env.alice, keys=[key],
        create={"title": kind, "ask": "", "kind": kind, "direction": "internal", "status": "open", "owes_next": "agent",
                "triage_state": "skipped", "source": kind, "source_ref": key},
        event={"source": kind, "source_ref": f"{key}:1", "kind": "status", "summary": "was"})
    tid = _their_request(env)
    env.board.correct(env.alice, tid, "owes_next", "me")
    env.board.merge(env.alice, src, tid)
    locks = json.loads(env.board.get_task(env.alice, tid)["user_locks"])
    env.board.ingest(env.alice, keys=[key], patch=patch, touch={"title": "Its new name"}, unlock=("owes_next",),
                     event={"source": kind, "source_ref": f"{key}:2", "kind": "status", "summary": "moved"})
    t = env.board.get_task(env.alice, tid)
    assert (t["status"], t["owes_next"], t["title"]) == ("open", "me", "Ana: book the table")
    assert json.loads(t["user_locks"]) == locks
    assert "close_waits" not in json.loads(t["flags"])
    env.board.unmerge(env.alice, src)
    t = env.board.get_task(env.alice, tid)
    assert (t["status"], t["owes_next"]) == ("open", "me")
    freed = env.board.get_task(env.alice, src)
    assert freed["title"] == "Its new name" and freed["owes_next"] == "nobody"


async def test_a_merged_jobs_deletion_and_locks_go_by_its_own_row(env):
    """Finished and merged into a live request, a deleted job stays done; its Dismiss survives a run."""
    job = _job(env, env.alice)
    run = env.sched.start_run(job, "scheduled")
    env.sched.finish_run(run, status="ok", result="Digest 1.", session_id="s-1", deliveries=[])
    run_sweep()
    jid = env.board.task_for_key(env.alice, f"sched:job:{job['id']}")
    tid = _their_request(env)
    env.board.merge(env.alice, jid, tid)
    assert env.board.keyed_tasks("sched:job:", user_id=env.alice, status="open") == []
    assert [t["task_id"] for t in env.board.keyed_tasks("sched:job:", user_id=env.alice, status="done")] == [tid]
    env.board.unmerge(env.alice, jid)
    env.board.correct(env.alice, jid, "dismiss")
    env.board.merge(env.alice, jid, tid)
    run = env.sched.start_run(job, "scheduled")
    env.sched.finish_run(run, status="ok", result="Digest 2.", session_id="s-2", deliveries=[])
    run_sweep()
    assert env.board.get_task(env.alice, jid)["status"] == "dismissed"
    assert env.board.get_task(env.alice, tid)["status"] == "open"


async def test_a_merged_sources_recorded_delivery_is_its_own_evidence(env):
    src, _ = env.board.ingest(
        env.alice, keys=["todo:helper:1"],
        create={"title": "To-do", "ask": "", "kind": "todo", "direction": "internal", "status": "open", "owes_next": "agent",
                "triage_state": "skipped", "source": "todo", "source_ref": "1"},
        event={"source": "todo", "source_ref": "1:1", "kind": "status", "summary": "was"})
    tid = _their_request(env)
    env.board.merge(env.alice, src, tid)
    env.board.ingest(env.alice, keys=["todo:helper:1"], event={
        "source": "todo", "source_ref": "1:2", "kind": "sent", "summary": "Delivered", "evidence": "recorded"})
    assert env.board.get_task(env.alice, tid)["evidence"] == "none"
    assert env.board.get_task(env.alice, src)["evidence"] == "verified"


async def test_a_merged_proposal_hears_from_its_action_and_doesnt_hold_the_task(env):
    action = env.auto.add_action(env.alice, kind="nudge", title="Remind Ana", rationale="", source="arbiter",
                                 status="awaiting_approval", target="helper")
    run_sweep()
    pid = env.board.task_for_key(env.alice, f"autonomy:{action['id']}")
    tid = _their_request(env)
    env.board.merge(env.alice, pid, tid)
    env.auto.mark_status(action["id"], "rejected") if hasattr(env.auto, "mark_status") else env.auto.update_status(action["id"], "rejected")
    run_sweep()
    assert env.board.get_task(env.alice, pid)["status"] != "open"
    assert env.board.get_task(env.alice, tid)["status"] == "open"
    assert not env.board.triage_view(env.alice, tid)["live_in"]
