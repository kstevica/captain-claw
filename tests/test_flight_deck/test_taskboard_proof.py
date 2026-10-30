"""An agent's "done" on what the user handed it is only its word: triage checks it
against the tool-call register and, where it can, the source itself, and a task
that isn't borne out stays with the user, whatever the reply says."""

from __future__ import annotations

import asyncio
import json
import sqlite3
import time
from types import SimpleNamespace

import pytest

from captain_claw import fabrication_guard
from captain_claw.flight_deck import jev_routes, taskboard_ingest, taskboard_nextsteps, taskboard_proof, taskboard_triage
from captain_claw.flight_deck import taskboard_handover as handover

from .test_taskboard_handover import _settle, relay  # noqa: F401
from .test_taskboard_nextsteps import ALICE, _task, board, jev, ledger, model, told  # noqa: F401

WORDS = "Send Ana the times and book the table."
SAYS_SENT = "I sent the email to Ana with the times."


@pytest.fixture()
def audit(tmp_path, monkeypatch):
    """The tool-call register, on its own file as on a deck."""
    path = tmp_path / "tool_audit.db"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE tool_calls (id INTEGER PRIMARY KEY AUTOINCREMENT, turn_id TEXT, agent_slug TEXT, "
                 "tool TEXT, risk TEXT, params TEXT, outcome TEXT, reason TEXT DEFAULT '', refs TEXT DEFAULT '{}')")
    conn.commit()
    monkeypatch.setattr(taskboard_ingest, "_audit_path", lambda: path)

    def add(turn_id, tool, action="", *, risk="write", outcome="ok", reason="", refs=None, **params):
        conn.execute("INSERT INTO tool_calls (turn_id, agent_slug, tool, risk, params, outcome, reason, refs) "
                     "VALUES (?, 'helper', ?, ?, ?, ?, ?, ?)",
                     (turn_id, tool, risk, json.dumps({**({"action": action} if action else {}), **params}),
                      outcome, reason, json.dumps(refs or {})))
        conn.commit()

    return add


@pytest.fixture()
def google(monkeypatch):
    """Google, as the user's own token reads it: no answer until a test gives one."""
    state = SimpleNamespace(asked=[], answers={}, token="tok")

    async def token(uid):
        return state.token

    async def fetch(tok, url, params=None):
        state.asked.append(url)
        return next((answer for part, answer in state.answers.items() if part in url), (0, {}))

    monkeypatch.setattr(taskboard_proof, "_token", token)
    monkeypatch.setattr(taskboard_proof, "_fetch", fetch)
    return state


@pytest.fixture()
def run(board, ledger, relay, audit, google, jev, model, monkeypatch):  # noqa: F811
    """A task handed over, and its run as the register leaves it."""
    monkeypatch.setattr(taskboard_triage, "_RUNNING", set())
    monkeypatch.setattr(handover, "_CHECK_WAIT_S", 3600.0)
    jev.answers.update(is_task=.9, owes_next=("me", .9), asks_assistant_to_act=.05, fulfilled=.1, did_as_asked=.9)

    async def go(reply=SAYS_SENT, calls=(), *, outcome="ok", count=None, words=WORDS, task=None) -> str:
        t = task or _task(board, flags={}, pending=False)
        relay["reply"] = reply
        await handover.hand_over(ALICE, t, "helper", words)
        await _settle()
        turn = f"run-{_handed(ledger, t)}"
        aid = turn.removeprefix("run-")
        for call in calls:
            audit(turn, *call[0], **call[1])
        now = time.time()
        board._conn.execute(
            "INSERT OR REPLACE INTO turns (turn_id, user_id, task_id, task_key, agent_slug, channel, source, started_at, "
            "ended_at, outcome, reply_head, tool_calls, automation) VALUES (?, ?, ?, ?, 'helper', 'web', 'autonomy', ?, ?, "
            "?, ?, ?, ?)",
            (turn, ALICE, t, f"autonomy:{aid}", now - 5, now - 1, outcome, reply,
             len(calls) if count is None else count, json.dumps({"kind": "autonomy", "id": aid})))
        board._conn.execute("UPDATE tasks SET triage_state = 'pending', triage_after = ?, decided_fp = '' WHERE id = ?",
                            (now - 1, t))
        board._conn.commit()
        await taskboard_triage.triage_user(ALICE)
        return t

    return go


def _handed(ledger, t) -> str:  # noqa: F811
    """The hand-over's action on *t*, the newest (next steps are actions too)."""
    return next(a["id"] for a in ledger.list_actions(ALICE) if a["ref_id"] == t and a["title"].startswith("Hand to"))


def _sent(mid="m-1", to="ana@x.co", subject="Times"):
    return (("google_mail", "send"), {"risk": "send", "refs": {"gmail_message_id": mid}, "to": to, "subject": subject})


def _event(eid="e-1", summary="Table for four"):
    return (("google_calendar", "create_event"), {"refs": {"calendar_event_id": eid}, "summary": summary})


def _proof(board, t) -> dict:  # noqa: F811
    return json.loads(board.get_task(ALICE, t)["flags"]).get("proof") or {}


def _where(board, t) -> tuple[str, str, str]:  # noqa: F811
    task = board.get_task(ALICE, t)
    return task["status"], task["owes_next"], board.lane_of(ALICE, t)


NEEDS_ME = ("open", "me", "needs_me")


# ── What it says isn't on record ──────────────────────────────────────


async def test_it_says_it_sent_the_email_and_no_send_is_on_record(board, run, jev):  # noqa: F811
    # Jev is sure it is done and the next move theirs: the reply says so, and that is all it is.
    jev.answers.update(fulfilled=.97, owes_next=("them", .95))
    t = await run()
    proof = _proof(board, t)
    assert proof["state"] == taskboard_proof.NOT_DONE and "no send is on record" in proof["note"]
    assert _where(board, t) == NEEDS_ME
    flags = json.loads(board.get_task(ALICE, t)["flags"])
    assert not flags.get("probably_done") and flags["why"] == proof["note"]
    assert proof["note"] in taskboard_triage.why_summary(board.get_task(ALICE, t), board.decisions(ALICE, t))


async def test_only_a_draft_is_not_a_send(board, run):  # noqa: F811
    t = await run(calls=[(("google_mail", "create_draft"), {"refs": {"gmail_draft_id": "d-1"}, "to": "ana@x.co"})])
    proof = _proof(board, t)
    assert proof["state"] == taskboard_proof.NOT_DONE and "only a draft" in proof["note"]
    assert proof["actions"][0]["label"] == "Draft to ana@x.co"


async def test_it_says_it_booked_the_calendar_with_no_change_on_record(board, run):  # noqa: F811
    t = await run("Created the event for Saturday at 8 pm and sent the invite to Ana.")
    assert _proof(board, t)["state"] == taskboard_proof.NOT_DONE and _where(board, t) == NEEDS_ME


async def test_it_says_it_sent_a_whatsapp_with_none_on_record(board, run):  # noqa: F811
    t = await run("I messaged Ana on WhatsApp with the times.")
    assert "WhatsApp" in _proof(board, t)["note"] and _proof(board, t)["state"] == taskboard_proof.NOT_DONE


async def test_without_jev_the_board_still_catches_it(board, run, jev):  # noqa: F811
    jev.fail = jev_routes.JevNotConnected("no")
    t = await run()
    assert _proof(board, t)["state"] == taskboard_proof.NOT_DONE and _where(board, t) == NEEDS_ME


# ── What it tried ─────────────────────────────────────────────────────


async def test_a_send_that_never_got_through_says_why(board, run):  # noqa: F811
    t = await run("I couldn't send it.", calls=[
        (("google_mail", "send"), {"risk": "send", "outcome": "blocked", "reason": "Email sending is off", "to": "ana@x.co"})])
    proof = _proof(board, t)
    assert proof["state"] == taskboard_proof.FAILED and "Email sending is off" in proof["note"]
    assert _where(board, t) == NEEDS_ME


async def test_a_send_that_worked_the_second_time_is_no_failure(board, run, google):  # noqa: F811
    google.answers["/messages/m-1"] = (200, {"labelIds": ["SENT"]})
    t = await run(calls=[(("google_mail", "send"), {"risk": "send", "outcome": "error", "reason": "timeout"}), _sent()])
    assert _proof(board, t)["state"] == taskboard_proof.PROVEN


async def test_a_run_that_broke_off_says_so_with_what_it_did_before(board, run, google):  # noqa: F811
    google.answers["/messages/m-1"] = (200, {"labelIds": ["SENT"]})
    t = await run("", calls=[_sent()], outcome="error")
    proof = _proof(board, t)
    assert proof["state"] == taskboard_proof.FAILED and proof["actions"][0]["seen"] == "gmail"


# ── What the source says ──────────────────────────────────────────────


async def test_gmail_has_it_and_jev_finds_it_all(board, run, google, jev):  # noqa: F811
    jev.answers.update(fulfilled=.97)
    google.answers["/messages/m-1"] = (200, {"labelIds": ["SENT"], "payload": {"headers": [
        {"name": "To", "value": "Ana Kovač <ana@x.co>"}, {"name": "Subject", "value": "Re: Times"}]}})
    t = await run(calls=[_sent()])
    proof = _proof(board, t)
    assert proof["state"] == taskboard_proof.PROVEN and proof["note"] == ""
    assert proof["actions"][0]["seen"] == "gmail" and proof["actions"][0]["label"] == "Email to Ana Kovač <ana@x.co>: Re: Times"
    assert board.get_task(ALICE, t)["status"] == "done"    # proven, and Jev is sure: closed, as it may


async def test_on_record_but_gmail_doesnt_have_it(board, run, google, jev):  # noqa: F811
    jev.answers.update(fulfilled=.97)
    google.answers["/messages/m-1"] = (404, {})
    t = await run(calls=[_sent()])
    proof = _proof(board, t)
    assert proof["state"] == taskboard_proof.NOT_DONE and "doesn't have it" in proof["note"]
    assert proof["actions"][0]["seen"] == taskboard_proof.GONE and _where(board, t) == NEEDS_ME


async def test_a_message_gmail_has_but_didnt_send_is_not_sent(board, run, google):  # noqa: F811
    google.answers["/messages/m-1"] = (200, {"labelIds": ["DRAFT"]})
    t = await run(calls=[_sent()])
    assert _proof(board, t)["state"] == taskboard_proof.NOT_DONE


@pytest.mark.parametrize("answer", [(0, {}), (401, {}), (403, {}), (500, {})], ids=["down", "401", "403", "500"])
async def test_a_source_that_cant_be_asked_leaves_the_record_standing(board, run, google, answer):  # noqa: F811
    google.answers["/messages/m-1"] = answer
    t = await run(calls=[_sent()])
    proof = _proof(board, t)
    assert proof["state"] == taskboard_proof.PROVEN and proof["actions"][0]["seen"] == ""


async def test_no_google_token_leaves_the_record_standing(board, run, google):  # noqa: F811
    google.token = None
    t = await run(calls=[_sent()])
    assert _proof(board, t)["state"] == taskboard_proof.PROVEN and google.asked == []


async def test_the_calendar_has_the_event(board, run, google):  # noqa: F811
    google.answers["/events/e-1"] = (200, {"status": "confirmed", "summary": "Table for four", "start": {
        "dateTime": "2026-11-07T20:00:00+01:00"}, "attendees": [{"email": "me@x.co", "self": True}, {"email": "ana@x.co"}]})
    t = await run("Created the event for Saturday at 8 pm.", calls=[_event()])
    action = _proof(board, t)["actions"][0]
    assert action["seen"] == "calendar" and _proof(board, t)["state"] == taskboard_proof.PROVEN
    assert action["label"] == "Calendar event created: Table for four, 2026-11-07 20:00, with ana@x.co"


@pytest.mark.parametrize("answer", [(404, {}), (410, {}), (200, {"status": "cancelled"})], ids=["404", "410", "cancelled"])
async def test_an_event_the_calendar_doesnt_have(board, run, google, answer):  # noqa: F811
    google.answers["/events/e-1"] = answer
    t = await run("Created the event.", calls=[_event()])
    assert _proof(board, t)["state"] == taskboard_proof.NOT_DONE


async def test_the_calendar_asked_is_the_one_it_used(board, run, google):  # noqa: F811
    await run("Done.", calls=[(("google_calendar", "create_event"), {"refs": {"calendar_event_id": "e-2"}, "calendar_id": "team"})])
    assert "/calendars/team/events/e-2" in google.asked[0]


async def test_outlook_isnt_read_back_and_its_record_stands(board, run, google):  # noqa: F811
    t = await run(calls=[(("microsoft_mail", "send"), {"risk": "send", "to": "ana@x.co", "subject": "Times"})])
    assert _proof(board, t)["state"] == taskboard_proof.PROVEN and google.asked == []


async def test_the_source_is_read_once_for_a_run(board, run, google, jev):  # noqa: F811
    google.answers["/messages/m-1"] = (200, {"labelIds": ["SENT"]})
    t = await run(calls=[_sent()])
    assert len(google.asked) == 1
    board._conn.execute("UPDATE tasks SET triage_state = 'pending', triage_after = ?, decided_fp = '' WHERE id = ?",
                        (time.time() - 1, t))
    board._conn.commit()
    await taskboard_triage.triage_user(ALICE)
    assert len(google.asked) == 1 and _proof(board, t)["actions"][0]["seen"] == "gmail"


# ── What was asked, in any words, for any work ────────────────────────


async def test_jev_says_what_was_asked_isnt_there(board, run, jev):  # noqa: F811
    """Nothing in the reply a rule could catch, in another language, and no email on record."""
    jev.answers.update(did_as_asked=.1, fulfilled=.97)
    t = await run("Sve je riješeno.", words="Pošalji Ani termine.")
    proof = _proof(board, t)
    assert proof["state"] == taskboard_proof.NOT_DONE and "Jev doesn't find" in proof["note"]
    assert _where(board, t) == NEEDS_ME


async def test_jev_isnt_sure_it_is_all(board, run, jev):  # noqa: F811
    jev.answers.update(did_as_asked=.5, fulfilled=.97)
    t = await run("Booked it.", calls=[_event()])
    assert _proof(board, t)["state"] == taskboard_proof.UNSURE and _where(board, t) == NEEDS_ME


async def test_jev_reads_the_users_words_and_the_record_not_the_reply(board, run, jev):  # noqa: F811
    await run("Booked it.", calls=[_event(summary="Table for four")])
    item = jev.calls[-1].state["items"][0]
    assert WORDS in item["quoted"]["owner_asked"] and "Table for four" in item["quoted"]["action_details"][0]
    assert item["handed_over"]["actions_on_record"][0]["what"] == "calendar"
    assert "t0_did_as_asked" in jev.calls[-1].questions


async def test_a_task_nobody_handed_over_isnt_asked_about(board, jev, model, monkeypatch):  # noqa: F811
    monkeypatch.setattr(taskboard_triage, "_RUNNING", set())
    _task(board, flags={}, pending=True)
    await taskboard_triage.triage_user(ALICE)
    assert jev.calls and not any("did_as_asked" in name for c in jev.calls for name in c.questions)


async def test_an_answer_is_all_some_requests_want(board, run, jev):  # noqa: F811
    jev.answers.update(fulfilled=.97)
    t = await run("Ana asks for the Saturday table: two options, both free.", words="Summarise this for me.")
    assert _proof(board, t)["state"] == taskboard_proof.PROVEN and board.get_task(ALICE, t)["status"] == "done"


async def test_without_jev_nobody_says_it_is_proven(board, run, jev, google):  # noqa: F811
    jev.fail = jev_routes.JevNotConnected("no")
    google.answers["/messages/m-1"] = (200, {"labelIds": ["SENT"]})
    t = await run(calls=[_sent()])
    assert _proof(board, t)["state"] == taskboard_proof.RECORDED and _where(board, t) == NEEDS_ME
    t2 = await run("Here is the summary.", words="Summarise this.")
    assert _proof(board, t2)["state"] == taskboard_proof.ANSWERED


async def test_anything_else_it_did_is_listed_and_its_housekeeping_isnt(board, run):  # noqa: F811
    t = await run("Updated the sheet.", calls=[
        (("datastore", "sql"), {}), (("read",), {"risk": "read"}), (("insights",), {}), (("todo",), {}),
        (("google_mail", "search"), {"risk": "read"})])
    assert [a["label"] for a in _proof(board, t)["actions"]] == ["datastore sql"]


async def test_what_it_passed_to_another_agent_isnt_taken_as_done(board, run, jev):  # noqa: F811
    jev.answers.update(fulfilled=.97)
    t = await run(calls=[(("consult_peer",), {"risk": "send"})])
    proof = _proof(board, t)
    assert proof["state"] == taskboard_proof.UNSURE and "another agent" in proof["note"]
    assert _where(board, t) == NEEDS_ME


# ── Timing ────────────────────────────────────────────────────────────


async def test_it_waits_for_the_calls_still_on_their_way(board, run, audit, ledger):  # noqa: F811
    t = await run(calls=[], count=1)
    assert not _proof(board, t) and board.get_task(ALICE, t)["triage_state"] == "pending"
    assert float(board.get_task(ALICE, t)["triage_after"]) > time.time()
    aid = _handed(ledger, t)
    audit(f"run-{aid}", *_sent()[0], **_sent()[1])
    board._conn.execute("UPDATE tasks SET triage_after = ?, decided_fp = '' WHERE id = ?", (time.time() - 1, t))
    board._conn.commit()
    await taskboard_triage.triage_user(ALICE)
    assert _proof(board, t)["calls"] == 1


async def test_calls_that_never_come_arent_waited_for_for_ever(board, run, ledger):  # noqa: F811
    t = await run(calls=[], count=1)
    aid = _handed(ledger, t)
    board._conn.execute("UPDATE turns SET ended_at = ? WHERE turn_id = ?", (time.time() - 3600, f"run-{aid}"))
    board._conn.execute("UPDATE tasks SET triage_after = ?, decided_fp = '' WHERE id = ?", (time.time() - 1, t))
    board._conn.commit()
    await taskboard_triage.triage_user(ALICE)
    assert _proof(board, t)["state"] == taskboard_proof.NOT_DONE


async def test_a_message_from_the_inbox_is_checked_too(board, run, ledger):  # noqa: F811
    """Nothing but a hand-over's run sends one back to triage: its close has to."""
    t = _task(board, flags={}, pending=False, kind="message")
    await run(task=t)
    assert _proof(board, t)["state"] == taskboard_proof.NOT_DONE
    aid = _handed(ledger, t)
    turn = board.get_turn(f"run-{aid}")
    assert taskboard_ingest._requeue(board, ALICE, t, time.time(), settle=True, worked_on=True)
    assert taskboard_ingest._requeue(board, ALICE, t, time.time(), settle=True) is None
    assert turn["automation"]["kind"] == "autonomy"


async def test_an_agent_that_reports_nothing_cant_be_checked_and_the_task_says_so(board, ledger, relay, monkeypatch):  # noqa: F811
    monkeypatch.setattr(handover, "_CHECK_WAIT_S", 0.0)
    t = _task(board, flags={}, pending=False)
    await handover.hand_over(ALICE, t, "helper", WORDS)
    await _settle()
    await asyncio.gather(*list(handover._CHECKS))
    proof = _proof(board, t)
    assert proof["state"] == taskboard_proof.NO_RECORD and "Restart it" in proof["note"]
    assert proof["note"] in taskboard_triage.why_summary(board.get_task(ALICE, t), board.decisions(ALICE, t))


async def test_a_run_whose_turn_came_isnt_called_unchecked(board, run, ledger):  # noqa: F811
    t = await run()
    assert not board.hand_over_unrecorded(ALICE, _handed(ledger, t), "x")
    assert _proof(board, t)["state"] == taskboard_proof.NOT_DONE


# ── What it leads to ──────────────────────────────────────────────────


def _steps(ledger, t) -> list[dict]:  # noqa: F811
    return [a for a in ledger.list_actions(ALICE) if a["ref_id"] == t and not a["title"].startswith("Hand to")]


async def test_a_false_done_gets_no_next_step_that_would_repeat_it(board, run, ledger):  # noqa: F811
    """"Tell Ana it is booked", from a reply that only says so."""
    t = await run()
    assert _proof(board, t)["state"] == taskboard_proof.NOT_DONE and _steps(ledger, t) == []
    assert taskboard_nextsteps.blocked(board.triage_view(ALICE, t)) == taskboard_nextsteps._NOT_BORNE_OUT
    assert taskboard_nextsteps.offered_kinds(ALICE, t) == []
    row, said = await taskboard_nextsteps.suggest(ALICE, t, "reply_with_result")
    assert row is None and said == taskboard_nextsteps._NOT_BORNE_OUT


async def test_a_proven_one_may_get_its_next_step(board, run, ledger, google):  # noqa: F811
    google.answers["/messages/m-1"] = (200, {"labelIds": ["SENT"]})
    t = await run(calls=[_sent()])
    assert _proof(board, t)["state"] == taskboard_proof.PROVEN and _steps(ledger, t)
    assert taskboard_nextsteps.blocked(board.triage_view(ALICE, t)) == ""


# ── The user's word, and what comes after ─────────────────────────────


async def test_the_users_done_stands_and_the_record_stays_on_it(board, run, ledger):  # noqa: F811
    t = await run()
    board.correct(ALICE, t, "done")
    assert board.get_task(ALICE, t)["status"] == "done"
    await taskboard_triage.triage_user(ALICE)
    assert board.get_task(ALICE, t)["status"] == "done" and _proof(board, t)["state"] == taskboard_proof.NOT_DONE


async def test_handing_it_over_again_starts_the_check_over(board, run, google):  # noqa: F811
    t = await run()
    assert _proof(board, t)["state"] == taskboard_proof.NOT_DONE
    await handover.hand_over(ALICE, t, "helper", "Try again.")
    await _settle()
    assert not _proof(board, t) and board.get_task(ALICE, t)["owes_next"] == "me"


async def test_the_users_own_words_arent_read_as_the_agents_claim(board, run):  # noqa: F811
    t = await run("Done.", words="Tell me if the email was sent to Ana.")
    assert _proof(board, t)["state"] == taskboard_proof.PROVEN


def test_the_whatsapp_claim_reads_the_way_the_guard_does():
    assert fabrication_guard.claims_whatsapp_sent("I messaged Ana on WhatsApp.")
    assert not fabrication_guard.claims_whatsapp_sent("I didn't message Ana on WhatsApp.")
    assert not fabrication_guard.claims_whatsapp_sent("Should I message Ana on WhatsApp?")
    assert not fabrication_guard.claims_whatsapp_sent("The weather is fine.")
