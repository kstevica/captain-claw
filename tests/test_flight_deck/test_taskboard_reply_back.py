"""A reply brings the same task back to Needs me, as it brings a thread back to
Gmail's inbox: after Done, with the next move someone else's, or snoozed. There,
or there already, it stays until the user acts or answers, whatever triage reads of it. "Not a
task", Dismiss, a mute rule, the user's own mail and old history keep their ways."""

from __future__ import annotations

import json
import time
from types import SimpleNamespace

import pytest

from captain_claw.flight_deck import jev_routes, taskboard_ingest, taskboard_passive, taskboard_sources
from captain_claw.flight_deck import taskboard_triage as tt
from captain_claw.flight_deck.taskboard_sources import run_sweep

from .test_taskboard_nextsteps import jev, model  # noqa: F401
from .test_taskboard_passive import (  # noqa: F401
    ME,
    SENDER,
    _closed_request,
    _connect_all,
    _gmail_box,
    _mail,
    _sent_mail,
    _store,
    board,
    switch_on,
)
from .test_taskboard_sources import env  # noqa: F401


@pytest.fixture()
def inbox(board, jev, model, monkeypatch):  # noqa: F811
    """Alice's Gmail on her board, Jev reading it as a task of hers."""
    uid = board.alice
    monkeypatch.setattr(tt, "_RUNNING", set())
    _connect_all(board, monkeypatch, uid)
    switch_on(uid, "gmail")
    mails: list[dict] = []
    _gmail_box(monkeypatch, {uid: mails})
    jev.answers.update(is_task=.9, owes_next=("me", .9), asks_assistant_to_act=.05, fulfilled=.1, reopen=.9)
    return SimpleNamespace(deck=board, s=board.board, uid=uid, mails=mails, jev=jev)


async def _read(x) -> None:
    x.s._conn.execute("UPDATE tasks SET triage_after = ? WHERE triage_state = 'pending'", (time.time() - 1,))
    x.s._conn.commit()
    await tt.triage_user(x.uid)


async def _task(x, thread="t-1") -> str:
    x.mails.append(_mail(f"{thread}-first", thread=thread, ago=7200))
    run_sweep()
    await _read(x)
    return x.s.task_for_key(x.uid, f"gmail:thread:{thread}")


def _reply(x, mid="r-1", thread="t-1", *, ago=10.0, body="Monday works.") -> None:
    x.mails.append(_mail(mid, thread=thread, subject="Re: Q3 invoice", body=body, ago=ago))
    run_sweep()


def _at(x, tid) -> tuple[str, str, str]:
    t = x.s.get_task(x.uid, tid)
    return t["status"], t["owes_next"], x.s.lane_of(x.uid, tid)


def _flags(x, tid) -> dict:
    return json.loads(x.s.get_task(x.uid, tid)["flags"])


def _tasks(x) -> list[dict]:
    return x.s.query(x.uid, view="all", limit=50)["rows"]


BACK = ("open", "me", "needs_me")


# ── It comes back ─────────────────────────────────────────────────────


async def test_after_done_the_reply_brings_the_same_task_back(inbox):
    tid = await _task(inbox)
    inbox.s.correct(inbox.uid, tid, "done")
    assert _at(inbox, tid)[2] == "done"
    _reply(inbox)
    assert _at(inbox, tid) == BACK
    task = inbox.s.get_task(inbox.uid, tid)
    assert (task["closed_at"], task["closed_by"], task["closed_by_party"], task["result"]) == (None, "", "", "")
    assert "status" not in json.loads(task["user_locks"]) and _flags(inbox, tid)["came_back"] is True
    # One task, with all of it on: what they wrote is what the timeline ends in.
    assert [t["id"] for t in _tasks(inbox)] == [tid]
    assert inbox.s.task_for_key(inbox.uid, "gmail:thread:t-1") == tid
    kinds = [e["kind"] for e in inbox.s.events(inbox.uid, tid)]
    assert kinds.count("received") == 2 and "correction" in kinds
    assert inbox.s.counts(inbox.uid)["needs_me"] == 1 and inbox.deck.kicked


@pytest.mark.parametrize("locked", [True, False], ids=["user_said_someone_else", "the_board_had_it_so"])
async def test_waiting_on_someone_the_reply_makes_it_the_users_move(inbox, locked):
    tid = await _task(inbox)
    if locked:
        inbox.s.correct(inbox.uid, tid, "owes_next", "them")
    else:
        inbox.s.update_fields(inbox.uid, tid, {"owes_next": "them"})
    assert _at(inbox, tid)[2] == "waiting"
    _reply(inbox)
    assert _at(inbox, tid) == BACK and "owes_next" not in json.loads(inbox.s.get_task(inbox.uid, tid)["user_locks"])
    await _read(inbox)
    assert _at(inbox, tid) == BACK


async def test_a_reply_ends_a_snooze(inbox):
    tid = await _task(inbox)
    inbox.s.correct(inbox.uid, tid, "snooze", time.time() + 86400)
    assert _at(inbox, tid)[2] == "snoozed"
    _reply(inbox)
    assert _at(inbox, tid) == BACK and inbox.s.get_task(inbox.uid, tid)["snooze_until"] is None
    assert "snooze_until" not in json.loads(inbox.s.get_task(inbox.uid, tid)["user_locks"])


async def test_a_failed_task_comes_back_too(inbox):
    tid = await _task(inbox)
    inbox.s.update_fields(inbox.uid, tid, {"status": "failed", "result": "failed", "closed_by": "source"})
    _reply(inbox)
    assert _at(inbox, tid) == BACK and len(_tasks(inbox)) == 1


async def test_what_was_said_of_the_last_try_goes(inbox):
    tid = await _task(inbox)
    inbox.s.update_fields(inbox.uid, tid, {"flags": {"agent_declined": True, "blocker": "No access",
                                                     "probably_done": True, "low_confidence": True}})
    inbox.s.correct(inbox.uid, tid, "done")
    _reply(inbox)
    flags = _flags(inbox, tid)
    assert not (flags["agent_declined"] or flags["probably_done"] or flags["low_confidence"])


async def test_a_waiting_follow_up_comes_back_and_its_ledger_keeps_the_rest(inbox):
    tid, _ = inbox.s.ingest(inbox.uid, keys=["fu:1"], extra_keys=["gmail:thread:t-f"], create={
        "title": "Ana's invoice", "kind": "follow_up", "owes_next": "them", "source": "follow_ups",
        "triage_state": "skipped"}, event={"source": "follow_ups", "source_ref": "1:made", "kind": "note"})
    _reply(inbox, thread="t-f")
    task = inbox.s.get_task(inbox.uid, tid)
    assert _at(inbox, tid) == BACK and task["triage_state"] == "skipped"   # triage never reads one
    # Closed, it is its ledger's to say: done is done.
    inbox.s.update_fields(inbox.uid, tid, {"status": "done", "owes_next": "nobody"})
    _reply(inbox, "r-2", thread="t-f")
    assert _at(inbox, tid)[0] == "done"


async def test_on_a_merged_thread_the_task_it_went_into_comes_back(inbox):
    tid = await _task(inbox)
    other = await _task(inbox, "t-2")
    inbox.s.merge(inbox.uid, other, tid)
    inbox.s.correct(inbox.uid, tid, "done")
    _reply(inbox, thread="t-2")
    assert _at(inbox, tid) == BACK


# ── There it stays until the user acts ────────────────────────────────


async def test_what_jev_reads_of_the_reply_doesnt_take_it_away(inbox):
    tid = await _task(inbox)
    inbox.s.correct(inbox.uid, tid, "done")
    _reply(inbox, body="Thanks, all sorted!")
    # Not a task, done for sure, waiting on someone: on any other task those would move it.
    inbox.jev.answers.update(is_task=.1, fulfilled=.97, owes_next=("them", .97))
    await _read(inbox)
    assert _at(inbox, tid) == BACK and _flags(inbox, tid)["came_back"] is True
    assert _flags(inbox, tid)["probably_done"] is True    # it may still say so
    assert "contains_instructions" in _flags(inbox, tid)


@pytest.mark.parametrize("handed", [True, False], ids=["after_an_agent_answered", "still_unanswered"])
async def test_a_reply_on_what_already_waits_on_the_user_isnt_closed_over(inbox, handed):
    tid = await _task(inbox)
    if handed:
        inbox.s.hand_over_started(inbox.uid, tid, "a1", {"slug": "helper", "name": "Helper"}, "Propose Monday.")
        inbox.s.hand_over_ended(inbox.uid, "a1", agent_name="Helper", text="Proposed Monday.")
    assert _at(inbox, tid) == BACK and not _flags(inbox, tid).get("came_back")
    _reply(inbox, body="Monday is a lock, send the invite.")
    assert _at(inbox, tid) == BACK and _flags(inbox, tid)["came_back"] is True
    # What was asked looks done, but what they just wrote hasn't been seen.
    inbox.jev.answers.update(fulfilled=.97)
    await _read(inbox)
    assert _at(inbox, tid) == BACK and _flags(inbox, tid)["came_back"] is True
    assert _flags(inbox, tid)["probably_done"] is True


async def test_without_jev_the_rules_leave_it_there_too(inbox):
    tid = await _task(inbox)
    inbox.s.correct(inbox.uid, tid, "done")
    _reply(inbox)
    inbox.jev.fail = jev_routes.JevNotConnected("no")
    await _read(inbox)
    assert _at(inbox, tid) == BACK and _flags(inbox, tid)["came_back"] is True


async def test_the_board_says_why_it_is_back(inbox):
    tid = await _task(inbox)
    inbox.s.correct(inbox.uid, tid, "done")
    _reply(inbox)
    task = inbox.s.get_task(inbox.uid, tid)
    said = tt.why_summary(task, inbox.s.decisions(inbox.uid, tid))
    assert "came back to you" in said and "You set where it stands" not in said


async def test_marking_it_done_again_holds_until_the_next_reply(inbox):
    tid = await _task(inbox)
    inbox.s.correct(inbox.uid, tid, "done")
    _reply(inbox)
    inbox.s.correct(inbox.uid, tid, "done")
    assert _at(inbox, tid)[2] == "done" and _flags(inbox, tid)["came_back"] is False
    await _read(inbox)
    assert _at(inbox, tid)[2] == "done"
    _reply(inbox, "r-2", body="One more thing.")
    assert _at(inbox, tid) == BACK and _flags(inbox, tid)["came_back"] is True
    assert len(_tasks(inbox)) == 1


async def test_any_of_the_users_moves_ends_the_hold_but_a_rename(inbox):
    tid = await _task(inbox)
    inbox.s.correct(inbox.uid, tid, "done")
    _reply(inbox)
    inbox.s.correct(inbox.uid, tid, "rename", "Ana's invoice")
    assert _flags(inbox, tid)["came_back"] is True
    inbox.s.correct(inbox.uid, tid, "owes_next", "them")
    assert _flags(inbox, tid)["came_back"] is False and _at(inbox, tid)[2] == "waiting"


async def test_undo_has_nothing_left_to_take_back_of_a_close_the_reply_ended(inbox):
    tid = await _task(inbox)
    inbox.s.correct(inbox.uid, tid, "done")
    _reply(inbox)
    with pytest.raises(ValueError):
        inbox.s.correct(inbox.uid, tid, "undo")
    assert _at(inbox, tid) == BACK
    inbox.s.correct(inbox.uid, tid, "done")
    inbox.s.correct(inbox.uid, tid, "undo")
    assert _at(inbox, tid) == BACK and "status" not in json.loads(inbox.s.get_task(inbox.uid, tid)["user_locks"])


async def test_the_users_own_answer_makes_it_wait_again(inbox):
    tid = await _task(inbox)
    inbox.s.correct(inbox.uid, tid, "owes_next", "them")
    _reply(inbox, ago=1800.0)
    assert _at(inbox, tid) == BACK
    inbox.mails.append(_sent_mail("own-1", "t-1", ago=1200.0))
    run_sweep()
    assert _at(inbox, tid)[2] == "waiting" and _flags(inbox, tid)["came_back"] is False
    _reply(inbox, "r-2")
    assert _at(inbox, tid) == BACK


async def test_handing_it_to_an_agent_ends_the_hold(inbox):
    tid = await _task(inbox)
    inbox.s.correct(inbox.uid, tid, "done")
    _reply(inbox)
    inbox.s.hand_over_started(inbox.uid, tid, "a1", {"slug": "helper", "name": "Helper"}, "Answer her.")
    assert _at(inbox, tid)[:2] == ("open", "agent") and _flags(inbox, tid)["came_back"] is False


# ── What keeps its ways ───────────────────────────────────────────────


async def test_the_users_own_mail_on_a_done_thread_opens_nothing(inbox):
    tid = await _task(inbox)
    inbox.s.correct(inbox.uid, tid, "done")
    inbox.mails.append(_sent_mail("own-1", "t-1", ago=1200.0))
    run_sweep()
    assert _at(inbox, tid)[2] == "done" and not _flags(inbox, tid).get("came_back")


async def test_not_a_task_stays_where_the_user_put_it(inbox):
    tid = await _task(inbox)
    inbox.s.correct(inbox.uid, tid, "not_task")
    _reply(inbox)
    assert inbox.s.get_task(inbox.uid, tid)["status"] == "not_task"
    assert inbox.s.task_for_key(inbox.uid, "gmail:thread:t-1") == tid and not _flags(inbox, tid).get("came_back")


async def test_dismissed_stays_and_the_reply_is_a_message_of_its_own(inbox):
    tid = await _task(inbox)
    inbox.s.correct(inbox.uid, tid, "dismiss")
    _reply(inbox)
    assert inbox.s.get_task(inbox.uid, tid)["status"] == "dismissed"
    held = inbox.s.task_for_key(inbox.uid, "gmail:thread:t-1")
    assert held != tid and inbox.s.get_task(inbox.uid, held)["status"] == "maybe"


async def test_a_mute_rule_keeps_a_reply_from_bringing_anything_back(inbox):
    tid = await _task(inbox)
    request, _ = inbox.s.ingest(inbox.uid, keys=["turn:t1"], extra_keys=["gmail:thread:t-r"], create={
        "title": "Ask Ana", "kind": "request", "source": "agent_turns", "owes_next": "them"},
        event={"source": "agent_turns", "source_ref": "t1:open", "kind": "received"})
    inbox.s.correct(inbox.uid, tid, "done")
    inbox.s.add_rule(inbox.uid, {"sender": SENDER}, "mute")
    _reply(inbox)
    _reply(inbox, "r-2", thread="t-r")
    assert _at(inbox, tid)[2] == "done"
    assert _at(inbox, request)[2] == "waiting" and not _flags(inbox, request).get("came_back")


async def test_a_task_a_mute_rule_made_stays_quiet(inbox):
    tid = await _task(inbox)
    inbox.s.update_fields(inbox.uid, tid, {"owes_next": "nobody", "flags": {"rule": "mute"}})
    _reply(inbox)
    assert _at(inbox, tid)[1] == "nobody" and not _flags(inbox, tid).get("came_back")


async def test_an_agent_at_work_on_it_is_left_alone(inbox):
    tid = await _task(inbox)
    inbox.s.update_fields(inbox.uid, tid, {"owes_next": "agent"})
    _reply(inbox)
    assert _at(inbox, tid)[1] == "agent" and not _flags(inbox, tid).get("came_back")


async def test_the_users_own_sent_mail_has_no_task_to_come_back_to(inbox):
    inbox.mails.append(_sent_mail("own-1", "t-s", ago=1200.0))
    run_sweep()
    sent = inbox.s.task_for_key(inbox.uid, "gmail:thread:t-s")
    assert inbox.s.get_task(inbox.uid, sent)["kind"] == "sent"
    _reply(inbox, thread="t-s")
    held = inbox.s.task_for_key(inbox.uid, "gmail:thread:t-s")
    assert held != sent and inbox.s.get_task(inbox.uid, held)["status"] == "maybe"   # Jev reads whether it is one


def _agents_send(x, thread="t-a") -> str:
    tid, _ = x.s.ingest(x.uid, keys=[f"gmail:thread:{thread}"], create={
        "title": "Email to Ana", "kind": "sent", "direction": "sent", "channel": "gmail", "owes_next": "nobody",
        "source": "gmail_sends", "triage_state": "skipped"},
        patch={"status": "done", "owes_next": "nobody", "closed_by": "evidence"},
        event={"source": "gmail_sends", "source_ref": "1", "kind": "sent", "actor_kind": "agent", "summary": "Email sent"})
    return tid


async def test_a_reply_to_what_an_agent_only_sent_is_a_message_for_jev_to_read(inbox):
    sent = _agents_send(inbox)
    _reply(inbox, thread="t-a")
    held = inbox.s.task_for_key(inbox.uid, "gmail:thread:t-a")
    assert held != sent and inbox.s.get_task(inbox.uid, sent)["status"] == "done"
    assert not any(e["kind"] == "received" for e in inbox.s.events(inbox.uid, sent))    # nothing left unseen on it
    task = inbox.s.get_task(inbox.uid, held)
    assert (task["kind"], task["status"], task["owes_next"], task["triage_state"]) == ("message", "maybe", "me", "pending")
    assert json.loads(task["flags"])["contains_instructions"] is True and inbox.deck.kicked
    # Jev's word puts it in Needs me, and the next reply stays on it.
    await _read(inbox)
    assert _at(inbox, held) == BACK
    _reply(inbox, "r-2", thread="t-a")
    assert inbox.s.task_for_key(inbox.uid, "gmail:thread:t-a") == held
    assert sum(e["kind"] == "received" for e in inbox.s.events(inbox.uid, held)) == 2


async def test_jev_can_still_say_a_reply_to_a_send_is_no_task(inbox):
    _agents_send(inbox)
    _reply(inbox, thread="t-a", body="Thanks!")
    inbox.jev.answers.update(is_task=.1)
    await _read(inbox)
    held = inbox.s.task_for_key(inbox.uid, "gmail:thread:t-a")
    assert inbox.s.get_task(inbox.uid, held)["status"] == "not_task"


async def test_a_mute_rule_keeps_a_reply_to_a_send_quiet(inbox):
    sent = _agents_send(inbox)
    inbox.s.add_rule(inbox.uid, {"sender": SENDER}, "mute")
    _reply(inbox, thread="t-a")
    held = inbox.s.task_for_key(inbox.uid, "gmail:thread:t-a")
    assert held != sent and _at(inbox, held)[1] == "nobody"
    assert inbox.s.get_task(inbox.uid, held)["triage_state"] == "skipped"


async def test_a_send_the_user_opened_again_keeps_its_thread(inbox):
    sent = _agents_send(inbox)
    inbox.s.correct(inbox.uid, sent, "reopen")
    _reply(inbox, thread="t-a")
    assert inbox.s.task_for_key(inbox.uid, "gmail:thread:t-a") == sent
    assert _at(inbox, sent) == BACK


async def test_the_agents_next_send_on_the_thread_goes_on_the_reply(inbox):
    _agents_send(inbox)
    _reply(inbox, thread="t-a")
    held = inbox.s.task_for_key(inbox.uid, "gmail:thread:t-a")
    p = taskboard_sources._Pass(inbox.s, None)
    p.ingest("gmail_sends", inbox.uid, keys=["gmail:thread:t-a"], event={
        "source": "gmail_sends", "source_ref": "2", "kind": "sent", "actor_kind": "agent", "summary": "Email sent"})
    assert any(e["kind"] == "sent" for e in inbox.s.events(inbox.uid, held))


def test_history_read_only_now_is_not_a_reply(tmp_path):
    s = _store(tmp_path)
    key = "gmail:thread:t1"
    done = _closed_request(s, key)
    now = time.time()
    assert not taskboard_ingest.reply_brings_back(s, ME, done, [key], now - 3600)
    assert taskboard_ingest.inbound_after_done(s, ME, done, [key], ref="old")    # Jev asks, as before
    task = s.get_task(ME, done)
    assert task["status"] == "done" and json.loads(task["flags"])["new_since_done"] is True
    # Sent just before the close and only read after it: news to the user.
    assert taskboard_ingest.reply_brings_back(s, ME, done, [key], now - 120)
    assert s.get_task(ME, done)["status"] == "open"


def test_a_message_older_than_a_conversation_is_no_answer_to_it():
    now = time.time()
    task = {"kind": "message", "status": "done", "closed_at": now - 60, "created_at": now - 900, "last_event_at": now - 60}
    answers = taskboard_passive._answers_closed
    assert answers(task, "gmail", now, "") and answers(task, "telegram", now, "")
    assert not answers(task, "gmail", now, "mute")
    assert not answers(task, "gmail", now - 3600, "")
    assert not answers({**task, "kind": "sent"}, "gmail", now, "")
    assert not answers({**task, "status": "dismissed"}, "gmail", now, "")
    # A chat's next message half a day on is a conversation of its own; an email thread has no such gap.
    old = {**task, "closed_at": now - 86400, "last_event_at": now - 86400, "created_at": now - 86400}
    assert not answers(old, "telegram", now, "") and answers(old, "gmail", now, "")


def test_a_key_a_task_only_loosely_holds_brings_nothing_back(tmp_path):
    s = _store(tmp_path)
    done = _closed_request(s, "gmail:thread:t1")
    s.add_link(ME, "autoreply:x1", done, "weak")
    assert not taskboard_ingest.reply_brings_back(s, ME, done, ["autoreply:x1"], time.time())
    assert s.get_task(ME, done)["status"] == "done"


def test_a_record_that_makes_a_task_is_no_reply_on_it(tmp_path):
    """A follow-up's "someone wrote this" is how it starts, waiting on them."""
    s = _store(tmp_path)
    p = taskboard_sources._Pass(s, None)
    tid, new = p.ingest("follow_ups", ME, keys=["fu:9"], create={
        "title": "Ana's invoice", "kind": "follow_up", "owes_next": "them", "source": "follow_ups",
        "triage_state": "skipped"}, event={"source": "follow_ups", "source_ref": "9:made", "kind": "received",
                                           "actor_kind": "party", "actor": "Ana", "summary": "Tracked"})
    assert new and s.get_task(ME, tid)["owes_next"] == "them"


def test_a_row_merged_into_another_doesnt_come_back_itself(tmp_path):
    s = _store(tmp_path)
    done = _closed_request(s, "gmail:thread:t1")
    s._conn.execute("UPDATE tasks SET merged_into = 'other' WHERE id = ?", (done,))
    s._conn.commit()
    assert not taskboard_ingest.reply_brings_back(s, ME, done, ["gmail:thread:t1"], time.time())


def test_switching_the_inbox_off_takes_the_reason_with_the_reply(tmp_path):
    s = _store(tmp_path)
    key = "gmail:thread:t1"
    done = _closed_request(s, key)
    now = time.time()

    def write():
        return s.ingest(ME, keys=[key], event={
            "source": "gmail_inbox", "source_ref": "g1", "at": now, "kind": "received", "channel": "gmail",
            "actor_kind": "party", "actor": "Ana", "summary": "SECRETG new", "private": True,
            "refs": {"retention_until": now + taskboard_passive.KEPT_S}})

    s.passive_write(ME, "gmail_inbox", [key], write)
    assert taskboard_ingest.reply_brings_back(s, ME, done, [key], now)
    assert json.loads(s.get_task(ME, done)["flags"])["came_back"] is True
    s.passive_forget(ME, "gmail_inbox")
    task = s.get_task(ME, done)
    assert task["status"] == "open" and "came_back" not in json.loads(task["flags"])
