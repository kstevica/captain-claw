"""Taskboard next steps (v2): Jev's next move in triage's pass, a draft on the
agent's own model, an autonomy action awaiting the user's Approve — never for
a task whose message asks the assistant to act, never for history, never
without Jev, never below 0.8, never a second one; ``mail.draft`` only for
email, ``reply.on_origin`` for a chat; recipients only from the task's verified
parties or its turn's origin; the edit-before-approve PATCH and its rules;
expiry when the task closes or after a week (and never by the arbiter's 48 h);
dispatch on the task's own agent; the rejection on the task; the deep link in
the notifications; ``message.send`` gone; the origin stamps the delivery needs."""

from __future__ import annotations

import json
import re
import secrets
import time
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from captain_claw.config import AutonomousWorkConfig
from captain_claw.flight_deck import (
    action_catalog,
    actions,
    autonomy,
    autonomy_routes,
    consciousness,
    delivery_routes,
    fd_dispatch,
    jev_routes,
    owner_notify,
    taskboard_nextsteps,
    taskboard_store,
    taskboard_triage,
)
from captain_claw.flight_deck.taskboard_store import TaskboardStore
from captain_claw.jev_client import JevError

ALICE = "user-alice"
BOB = "user-bob"


@pytest.fixture()
def board(tmp_path, monkeypatch):
    monkeypatch.setenv("FD_AUTH_ENABLED", "true")
    monkeypatch.delenv("FD_PUBLIC_URL", raising=False)
    b = TaskboardStore(tmp_path / "taskboard.db")
    monkeypatch.setattr(taskboard_store, "_STORE", b)
    monkeypatch.setattr(taskboard_triage, "_RUNNING", set())
    monkeypatch.setattr(taskboard_nextsteps, "_BUSY", set())
    yield b
    b.close()


@pytest.fixture()
def ledger(tmp_path, monkeypatch):
    s = autonomy.AutonomyStore(tmp_path / "autonomy.db")
    monkeypatch.setattr(autonomy, "_STORE", s)
    monkeypatch.setattr(autonomy, "global_defaults", lambda: AutonomousWorkConfig().model_dump())
    monkeypatch.setattr(owner_notify, "waiting_may_have_changed", lambda uid: None)
    return s


@pytest.fixture()
def shared_bot(monkeypatch):
    """The deck's shared Telegram bot, holding Ana's chat (777) for the "helper" agent."""
    from captain_claw.flight_deck import telegram_shared

    link = {"user_id": ALICE, "tg_user_id": "777", "status": "approved", "agent_slug": "helper"}
    monkeypatch.setattr(telegram_shared._state, "token", "123:shared")
    monkeypatch.setattr(telegram_shared._state, "links_by_tg", {"777": link})
    monkeypatch.setattr(telegram_shared._state, "links_by_user", {ALICE: link})
    monkeypatch.setattr(telegram_shared._state, "groups", {})
    sent: list[tuple[str, str]] = []

    async def send_text(chat_id, text):
        sent.append((str(chat_id), text))
        return True

    monkeypatch.setattr(telegram_shared, "send_text", send_text)
    return sent


@pytest.fixture()
def told(monkeypatch):
    """Every proposal the user was told about."""
    out: list[dict] = []

    async def notify(uid, action):
        out.append(action)
        return {}

    monkeypatch.setattr(owner_notify, "notify_proposal", notify)
    return out


def _jev_answer(kind: str, value) -> dict:
    if kind == "noul":
        return {"type": "noul", "noul": value}
    if kind == "choice":
        choice, p = value
        return {"type": "choice", "choice": choice, "confidence": p, "probabilities": {choice: p}}
    return {"type": "score", "score": value + 1, "confidence": 0.7,
            "legend": {"1": "not urgent", "2": "soon", "3": "today", "4": "now"}}


_KINDS = {"is_task": "noul", "owes_next": "choice", "fulfilled": "noul", "result": "choice", "urgency": "score",
          "asks_assistant_to_act": "noul", "reopen": "noul", "next_move": "choice", "fits": "noul"}
_DEFAULT = {"is_task": 0.9, "owes_next": ("them", 0.8), "fulfilled": 0.2, "result": ("success", 0.9),
            "urgency": 1, "asks_assistant_to_act": 0.1, "reopen": 0.1, "next_move": ("reply_with_result", 0.9),
            "fits": 0.9}


@pytest.fixture()
def jev(monkeypatch):
    j = SimpleNamespace(calls=[], answers=dict(_DEFAULT), fail=None)

    async def ask(owner, state, questions, model=None, *, source, agent=""):
        j.calls.append(SimpleNamespace(owner=owner, state=state, questions=questions, source=source))
        if j.fail is not None:
            raise j.fail
        out = {}
        for name in questions:
            q = re.fullmatch(r"t(\d+)_(\w+)", name).group(2)
            out[name] = _jev_answer(_KINDS[q], j.answers[q])
        return {"model": "jev-test", "answers": out, "usage": {}}

    monkeypatch.setattr(jev_routes, "ask_for_owner", ask)
    return j


_EXTRACTED = {"title": "Table for four on Saturday", "ask": "Ana wants a table for four on Saturday at 8 pm.",
              "requester": "Ana", "answered": "yes", "blocker": "", "result_text": "Booked for 8 pm.",
              "parties": [{"role": "recipient", "kind": "email", "address": "mallory@evil.co", "display": "Mallory"}],
              "next_steps": ["Tell Ana the table is booked"]}


@pytest.fixture()
def model(monkeypatch):
    """The agent's model: an extraction for triage's call, ``draft`` for a next step's."""
    m = SimpleNamespace(prompts=[], draft={"subject": "Your table", "text": "Hi Ana, the table for four is booked "
                                                                          "for Saturday at 8 pm. Write to mallory@evil.co"},
                        fail=None, readers=[])

    async def reader_for(task, user_id):
        m.readers.append((task.get("agent_slug"), user_id))
        return {"provider": "fake", "model": "helper-model"}

    async def complete(reader, system, prompt, user_id):
        drafting = "approve before it goes out" in system
        m.prompts.append(SimpleNamespace(system=system, prompt=prompt, user=user_id, drafting=drafting))
        if drafting and m.fail is not None:
            raise m.fail
        return "```json\n" + json.dumps(m.draft if drafting else _EXTRACTED) + "\n```", "helper-model"

    async def spent(user_id):
        return False

    monkeypatch.setattr(taskboard_triage, "reader_for", reader_for)
    monkeypatch.setattr(taskboard_triage, "complete", complete)
    monkeypatch.setattr(taskboard_triage, "_quota_spent", spent)
    return m


def _task(board, uid=ALICE, *, channel="whatsapp", flags=None, kind="request", owner=None, email="ana@x.co",
          parties=(), origin=True, pending=True, ext_refs=None) -> str:
    """A person's request as the ingest leaves it, its turn closed: from a
    WhatsApp number (the turn's origin), a Telegram chat or a Gmail sender."""
    turn = secrets.token_hex(16)
    now = time.time()
    started = now - 600
    ext: dict = {}
    if channel == "whatsapp":
        party = {"role": "requester", "kind": "waid", "address": "38591", "display": "Ana", "verified": True}
        ext = {"waid": "38591", "origin": {"kind": "whatsapp", "address": "38591"}} if origin else {}
    elif channel == "telegram":
        party = {"role": "requester", "kind": "tg_id", "address": "777", "display": "Ana", "verified": True}
        ext = {"telegram_chat_id": "777", "origin": {"kind": "telegram", "address": "777"}} if origin else {}
    elif channel.startswith("device"):
        party = {"role": "requester", "kind": "fd_user", "address": uid, "display": "Alice", "verified": True}
        # The frame's own origin (turn_register.note_frame_origin), not the session's.
        ext = {"turn_origin": {"kind": "channel", "address": f"device:{uid}:stick"}} if origin else {}
    elif channel == "gmail":
        party = {"role": "requester", "kind": "email", "address": email, "display": "Ana Đurić", "verified": True}
    else:
        party = {"role": "requester", "kind": "fd_user", "address": uid, "display": "Alice", "verified": True}
    ext = ext_refs if ext_refs is not None else ext
    text = "Can you book a table for four on Saturday?"
    task_id, _ = board.ingest(
        uid, keys=[f"turn:{turn}"], parties=[party, *parties],
        create={"title": text, "ask": text, "kind": kind, "direction": "received", "channel": channel,
                "status": "open", "owes_next": "them", "triage_state": "pending" if pending else "ok",
                "triage_after": now - 1 if pending else None, "agent_slug": "helper", "agent_name": "Helper",
                "agent_owner_id": owner or uid, "source": "agent_turns", "source_ref": turn, "flags": flags or {},
                "next_steps": ["Tell Ana the table is booked"]},
        event={"source": "agent_turns", "source_ref": f"{turn}:open", "at": started, "kind": "received",
               "channel": channel, "actor_kind": "party", "summary": text, "refs": {"turn_id": turn}})
    board._conn.execute(
        "INSERT INTO turns (turn_id, user_id, agent_owner_id, agent_slug, task_id, channel, speaker_label, text_head, "
        "reply_head, outcome, started_at, ended_at, ext_refs) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (turn, uid, owner or uid, "helper", task_id, channel, "Ana", text, "Booked for 8 pm.", "ok", started,
         started + 5, json.dumps(ext)))
    board._conn.commit()
    board.ingest(uid, keys=[f"turn:{turn}"], patch={"triage_state": "pending", "triage_after": time.time() - 1}
                 if pending else None,
                 event={"source": "agent_turns", "source_ref": f"{turn}:close", "at": started + 5, "kind": "reply",
                        "actor_kind": "agent", "summary": "Booked for 8 pm.", "outcome": "ok",
                        "refs": {"turn_id": turn, "outcome": "ok", "failed_tool_calls": 0}})
    return task_id


def _waiting(ledger, uid=ALICE) -> list[dict]:
    return [a for a in ledger.list_actions(uid, status="awaiting_approval") if a["source"] == "taskboard"]


def _flags(board, task_id, uid=ALICE) -> dict:
    return json.loads(board.get_task(uid, task_id)["flags"])


def _req(uid, body=None):
    async def _json():
        return body

    return SimpleNamespace(state=SimpleNamespace(user_id=uid), json=_json)


def _backdate(ledger, action_id, **delta):
    ts = (datetime.now(UTC) - timedelta(**delta)).isoformat()
    with ledger._lock:
        ledger._c().execute("UPDATE autonomous_actions SET created_at = ? WHERE id = ?", (ts, action_id))
        ledger._c().commit()


# ── Jev's next move in triage's pass ──────────────────────────────────


async def test_a_sure_reply_move_proposes_a_reply_on_the_chat(board, ledger, jev, model, told):
    tid = _task(board)
    res = await taskboard_triage.triage_user(ALICE)
    assert res["jev"] == 1
    assert any(n.endswith("_next_move") for n in jev.calls[-1].questions)
    assert jev.calls[-1].state["items"][0]["next_step"]["can_reach"] == ["the WhatsApp chat it came from"]
    (row,) = _waiting(ledger)
    p = row["payload"]
    assert (row["kind"], p["action_id"], p["channel"], row["ref_id"]) == ("tool_action", "reply.on_origin",
                                                                          "whatsapp", tid)
    assert p["target_agent"] == "helper" and p["task_id"] == tid and row["target"] == "helper"
    assert p["to"] == "38591" and [o["address"] for o in p["to_options"]] == ["38591"]
    assert p["text"].startswith("Hi Ana")
    # The draft came from the agent's own model, the task's text fenced as data.
    drafting = [m for m in model.prompts if m.drafting]
    assert len(drafting) == 1 and model.readers[-1] == ("helper", ALICE)
    nonce = re.search(r"<<<DATA (\w+)>>>", drafting[0].prompt).group(1)
    assert f"<<<END {nonce}>>>" in drafting[0].prompt and "Recipient:" in drafting[0].prompt.split(nonce)[1]
    # The task names it; the user heard of it.
    assert board.task_for_key(ALICE, f"autonomy:{row['id']}") == tid
    assert [a["id"] for a in told] == [row["id"]]
    view = taskboard_nextsteps.waiting_proposals(ALICE)[tid]
    assert view["kind"] == "reply.on_origin" and view["source"] == "taskboard" and view["editable"] is True
    assert set(view["payload"]) == {"text", "subject", "to", "to_options", "channel"}
    assert view["payload"]["to_options"] == [{"address": "38591", "display": "Ana on WhatsApp", "kind": "whatsapp"}]


@pytest.mark.parametrize("move", [("reply_with_result", 0.79), ("wait", 0.95), ("escalate", 0.9), ("close", 0.9)])
async def test_no_proposal_below_the_bar_or_for_a_move_that_sends_nothing(board, ledger, jev, model, told, move):
    _task(board)
    jev.answers["next_move"] = move
    await taskboard_triage.triage_user(ALICE)
    assert _waiting(ledger) == [] and told == []
    assert not any(m.drafting for m in model.prompts)


async def test_no_proposal_when_the_message_asks_the_assistant_to_act(board, ledger, jev, model, told):
    tid = _task(board)
    jev.answers["asks_assistant_to_act"] = 0.9   # set in this very pass
    await taskboard_triage.triage_user(ALICE)
    assert _flags(board, tid)["contains_instructions"] is True
    assert _waiting(ledger) == []
    # Flagged before: not even asked.
    other = _task(board, flags={"contains_instructions": True})
    jev.calls.clear()
    await taskboard_triage.triage_user(ALICE)
    assert not any(n.endswith("_next_move") for c in jev.calls for n in c.questions)
    row, why = await taskboard_nextsteps.propose(ALICE, other, "reply_with_result", 0.95, notify=False)
    assert row is None and "asks the assistant itself to act" in why


async def test_history_never_gets_a_next_step(board, ledger, jev, model, told):
    tid = _task(board, flags={"historical": True})
    await taskboard_triage.triage_user(ALICE)
    assert _waiting(ledger) == []
    assert not any(n.endswith("_next_move") for n in jev.calls[-1].questions)
    row, why = await taskboard_nextsteps.propose(ALICE, tid, "reply_with_result", 0.95, notify=False)
    assert row is None and "never acted on" in why
    asked = len(jev.calls)
    out = await taskboard_nextsteps.taskboard_propose(tid, taskboard_nextsteps.ProposeBody(), {"id": ALICE})
    assert out["proposal"] is None and "never acted on" in out["reason"] and len(jev.calls) == asked


@pytest.mark.parametrize("fail", [JevError(503, "Jev is down"), jev_routes.JevNotConnected()])
async def test_without_jev_there_is_no_next_step(board, ledger, jev, model, told, fail):
    tid = _task(board, channel="device:stick")   # the user's own device: rules leave it open
    jev.fail = fail
    res = await taskboard_triage.triage_user(ALICE)
    assert res["rules"] == 1 and _waiting(ledger) == []
    before = board.daily_counts(ALICE)["jev"]
    out = await taskboard_nextsteps.taskboard_propose(tid, taskboard_nextsteps.ProposeBody(), {"id": ALICE})
    assert out["proposal"] is None and "Jev" in out["reason"]
    assert board.daily_counts(ALICE)["jev"] == before   # never answered: not spent


async def test_the_jev_cap_spent_means_no_next_step_on_demand(board, ledger, jev, model):
    tid = _task(board, pending=False)
    board.save_settings(ALICE, jev_per_day=0)
    out = await taskboard_nextsteps.taskboard_propose(tid, taskboard_nextsteps.ProposeBody(), {"id": ALICE})
    assert out["proposal"] is None and "cap" in out["reason"] and jev.calls == []


async def test_not_while_one_waits_on_the_task(board, ledger, jev, model, told):
    tid = _task(board)
    other = ledger.add_action(ALICE, kind="nudge", title="Nudge about Ana", status="awaiting_approval")
    board.add_link(ALICE, f"autonomy:{other['id']}", tid)
    await taskboard_triage.triage_user(ALICE)
    assert not any(n.endswith("_next_move") for n in jev.calls[-1].questions)   # not even asked
    assert _waiting(ledger) == []
    out = await taskboard_nextsteps.taskboard_propose(tid, taskboard_nextsteps.ProposeBody(), {"id": ALICE})
    assert out["proposal"]["id"] == other["id"] and "already waits" in out["reason"]
    # One proposed, then nothing more for the same task.
    ledger.update_status(other["id"], "rejected")
    first, _ = await taskboard_nextsteps.propose(ALICE, tid, "reply_with_result", 0.9, notify=False)
    again, why = await taskboard_nextsteps.propose(ALICE, tid, "remind", 0.9, notify=False)
    assert first is not None and again is None and "already waits" in why


async def test_nobody_to_reach_means_no_question_and_no_proposal(board, ledger, jev, model, told):
    tid = _task(board, origin=False)   # a WhatsApp turn whose number never came
    await taskboard_triage.triage_user(ALICE)
    assert not any(n.endswith("_next_move") for n in jev.calls[-1].questions)
    out = await taskboard_nextsteps.taskboard_propose(tid, taskboard_nextsteps.ProposeBody(), {"id": ALICE})
    assert out["proposal"] is None and "reached" in out["reason"]


async def test_a_draft_that_fails_proposes_nothing(board, ledger, jev, model, told):
    _task(board)
    model.fail = RuntimeError("timeout")
    await taskboard_triage.triage_user(ALICE)
    assert _waiting(ledger) == [] and told == []


# ── Kinds and recipients ──────────────────────────────────────────────


async def test_mail_draft_only_for_email_and_only_to_verified_parties(board, ledger, jev, model):
    tid = _task(board, channel="gmail", parties=[
        {"role": "cc", "kind": "email", "address": "bo@x.co", "display": "Bo", "verified": True},
        {"role": "recipient", "kind": "email", "address": "eve@evil.co", "display": "Eve", "verified": False}])
    board.add_link(ALICE, "gmail:msg:m-42", tid)
    row, why = await taskboard_nextsteps.propose(ALICE, tid, "reply_with_result", 0.9, notify=False)
    p = row["payload"]
    assert why == "" and p["action_id"] == "mail.draft" and p["channel"] == "email"
    assert p["to"] == "ana@x.co" and [o["address"] for o in p["to_options"]] == ["ana@x.co", "bo@x.co"]
    assert p["reply_to_message_id"] == "m-42" and p["subject"] == "Your table"
    # The model wrote an address into its text: it is text, never a recipient.
    args = taskboard_nextsteps.dispatch_args(row)
    assert args["to"] == "ana@x.co" and "mallory@evil.co" in args["body"]
    assert action_catalog.get_action("mail.draft")["base_args"] == {"action": "create_draft"}


async def test_a_chat_task_never_gets_an_email_draft(board, ledger, jev, model):
    tid = _task(board, channel="whatsapp")
    row, why = await taskboard_nextsteps.propose(ALICE, tid, "draft_email", 0.95, notify=False)
    assert row is None and "email" in why
    row, _ = await taskboard_nextsteps.propose(ALICE, tid, "ask_clarifying", 0.95, notify=False)
    assert row["payload"]["action_id"] == "reply.on_origin" and row["payload"]["channel"] == "whatsapp"


async def test_extraction_parties_are_never_recipients(board, ledger, jev, model):
    tid = _task(board, channel="gmail")
    await taskboard_triage.triage_user(ALICE)   # the extraction names mallory@evil.co, unverified
    assert any(p["address"] == "mallory@evil.co" and not p["verified"]
               for p in board.parties_for(ALICE, [tid])[tid])
    (row,) = _waiting(ledger)
    assert all(o["address"] != "mallory@evil.co" for o in row["payload"]["to_options"])
    assert row["payload"]["to"] == "ana@x.co"


@pytest.mark.parametrize(("channel", "ext", "origin"), [
    ("telegram", None, {"kind": "telegram", "address": "777"}),
    ("device:stick", None, {"kind": "channel", "address": f"device:{ALICE}:stick"}),
    # A session origin stamped by an older bridge still counts, when it is the device's.
    ("device:stick", {"origin": {"kind": "channel", "address": f"device:{ALICE}:old"}},
     {"kind": "channel", "address": f"device:{ALICE}:old"})])
async def test_a_reply_goes_to_the_origin_of_the_turn(board, ledger, jev, model, shared_bot, channel, ext, origin):
    tid = _task(board, channel=channel, ext_refs=ext)
    row, _ = await taskboard_nextsteps.propose(ALICE, tid, "reply_with_result", 0.9, notify=False)
    assert taskboard_nextsteps.dispatch_args(row)["origin"] == origin


async def test_a_telegram_chat_the_decks_bot_doesnt_hold_is_never_a_recipient(board, ledger, jev, model, shared_bot):
    from captain_claw.flight_deck import telegram_shared

    # Ana wrote to the agent's own bot: its token never reaches Flight Deck.
    tid = _task(board, channel="telegram", ext_refs={"telegram_chat_id": "888"})
    row, why = await taskboard_nextsteps.propose(ALICE, tid, "reply_with_result", 0.9, notify=False)
    assert row is None and "can be reached" in why
    # The deck's bot holds 777 for another agent only: not this task's either.
    telegram_shared._state.links_by_tg["777"] = {**telegram_shared._state.links_by_tg["777"], "agent_slug": "other"}
    tid = _task(board, channel="telegram")
    assert taskboard_nextsteps.reply_options(board.triage_view(ALICE, tid)) == []
    assert _waiting(ledger) == []


# ── Suggest next step (on demand) ─────────────────────────────────────


async def test_suggest_next_step_auto_and_by_kind(board, ledger, jev, model, told):
    tid = _task(board, pending=False)
    out = await taskboard_nextsteps.taskboard_propose(tid, taskboard_nextsteps.ProposeBody(), {"id": ALICE})
    assert out["proposal"]["kind"] == "reply.on_origin" and out["reason"] == ""
    assert list(jev.calls[-1].questions) == ["t0_next_move"] and board.daily_counts(ALICE)["jev"] == 1
    assert told == []   # asked for on the board: no WhatsApp about it
    assert board.decisions(ALICE, tid)[0]["qset"] == taskboard_nextsteps.NEXT_QSET
    ledger.update_status(out["proposal"]["id"], "rejected")
    jev.answers["fits"] = 0.6
    out = await taskboard_nextsteps.taskboard_propose(
        tid, taskboard_nextsteps.ProposeBody(kind="remind"), {"id": ALICE})
    assert out["proposal"] is None and "not sure enough" in out["reason"]
    assert list(jev.calls[-1].questions) == ["t0_fits"]
    jev.answers["fits"] = 0.85
    out = await taskboard_nextsteps.taskboard_propose(
        tid, taskboard_nextsteps.ProposeBody(kind="remind"), {"id": ALICE})
    assert out["proposal"]["title"].startswith("Remind Ana on WhatsApp")
    with pytest.raises(HTTPException) as e:
        await taskboard_nextsteps.taskboard_propose(tid, taskboard_nextsteps.ProposeBody(kind="send"), {"id": ALICE})
    assert e.value.status_code == 400
    with pytest.raises(HTTPException) as e:
        await taskboard_nextsteps.taskboard_propose(tid, taskboard_nextsteps.ProposeBody(), {"id": BOB})
    assert e.value.status_code == 404


# ── Edit before approve (PATCH) ───────────────────────────────────────


async def _proposed(board, ledger, channel="gmail", **kw) -> tuple[str, dict]:
    tid = _task(board, channel=channel, pending=False, parties=[
        {"role": "cc", "kind": "email", "address": "bo@x.co", "display": "Bo", "verified": True}], **kw)
    row, _ = await taskboard_nextsteps.propose(ALICE, tid, "reply_with_result", 0.9, notify=False)
    return tid, row


async def test_patch_edits_wording_and_recipient_within_to_options(board, ledger, jev, model):
    tid, row = await _proposed(board, ledger)
    out = await autonomy_routes.edit_action_route(
        row["id"], _req(ALICE, {"text": "Hi Ana — booked.", "subject": "Booked", "to": "BO@x.co"}), None)
    p = out["action"]["payload"]
    assert (p["text"], p["subject"], p["to"], p["channel"]) == ("Hi Ana — booked.", "Booked", "bo@x.co", "email")
    assert out["action"]["title"].startswith("Reply to Bo")
    assert taskboard_nextsteps.dispatch_args(out["action"])["to"] == "bo@x.co"


@pytest.mark.parametrize("body", [{"to": "eve@evil.co"}, {"text": "  "}, {"channel": "whatsapp"},
                                  {"to": "bo@x.co", "channel": "telegram"}, {"subject": 7}])
async def test_patch_refuses_what_isnt_the_tasks_own(board, ledger, jev, model, body):
    _tid, row = await _proposed(board, ledger)
    with pytest.raises(HTTPException) as e:
        await autonomy_routes.edit_action_route(row["id"], _req(ALICE, body), None)
    assert e.value.status_code == 400
    assert ledger.get_action(row["id"])["payload"]["to"] == "ana@x.co"


async def test_patch_is_the_owners_while_it_waits_and_only_for_board_proposals(board, ledger, jev, model):
    _tid, row = await _proposed(board, ledger)
    with pytest.raises(HTTPException) as e:
        await autonomy_routes.edit_action_route(row["id"], _req(BOB, {"text": "mine now"}), None)
    assert e.value.status_code == 404
    arbiter_one = ledger.add_action(ALICE, kind="tool_action", title="Draft", status="awaiting_approval",
                                    payload={"action_id": "mail.draft", "args": {"to": "a@x.co"}})
    with pytest.raises(HTTPException) as e:
        await autonomy_routes.edit_action_route(arbiter_one["id"], _req(ALICE, {"text": "x"}), None)
    assert e.value.status_code == 409
    ledger.update_status(row["id"], "rejected")
    with pytest.raises(HTTPException) as e:
        await autonomy_routes.edit_action_route(row["id"], _req(ALICE, {"text": "late"}), None)
    assert e.value.status_code == 409
    # Compare-and-set: an Approve between the read and the write wins.
    _tid, row2 = await _proposed(board, ledger, channel="whatsapp")
    assert ledger.claim_action(row2["id"], ("awaiting_approval",), "dispatched")
    assert ledger.edit_proposal(row2["id"], title="x", payload={}) is False


async def test_patch_moves_a_chat_reply_to_email_and_back(board, ledger, jev, model):
    tid = _task(board, channel="whatsapp", pending=False, parties=[
        {"role": "recipient", "kind": "email", "address": "ana@x.co", "display": "Ana", "verified": True}])
    row, _ = await taskboard_nextsteps.propose(ALICE, tid, "reply_with_result", 0.9, notify=False)
    assert row["payload"]["action_id"] == "reply.on_origin"
    out = await autonomy_routes.edit_action_route(row["id"], _req(ALICE, {"channel": "email"}), None)
    p = out["action"]["payload"]
    assert (p["action_id"], p["to"], p["subject"]) == ("mail.draft", "ana@x.co", "Your table")
    out = await autonomy_routes.edit_action_route(row["id"], _req(ALICE, {"to": "38591"}), None)
    assert out["action"]["payload"]["action_id"] == "reply.on_origin"


# ── Expiry ────────────────────────────────────────────────────────────


async def test_a_next_step_expires_when_its_task_closes(board, ledger, jev, model):
    tid, row = await _proposed(board, ledger, channel="whatsapp")
    assert tid in taskboard_nextsteps.waiting_proposals(ALICE)
    board.correct(ALICE, tid, "done", actor="Alice")
    assert taskboard_nextsteps.waiting_proposals(ALICE) == {}
    got = ledger.get_action(row["id"])
    assert got["status"] == "expired" and "its task closed" in got["outcome_note"]


async def test_a_task_jev_closes_takes_its_next_step_with_it(board, ledger, jev, model):
    tid, row = await _proposed(board, ledger, channel="whatsapp")
    board.update_fields(ALICE, tid, {"triage_state": "pending", "triage_after": time.time() - 1})
    board.ingest(ALICE, keys=[f"autonomy:{row['id']}"], event={
        "source": "test", "source_ref": "new", "at": time.time(), "kind": "note", "summary": "more"})
    jev.answers["fulfilled"] = 0.95
    await taskboard_triage.triage_user(ALICE)
    assert board.get_task(ALICE, tid)["status"] == "done"
    assert ledger.get_action(row["id"])["status"] == "expired"


async def test_approving_a_next_step_whose_task_closed_is_refused(board, ledger, jev, model):
    tid, row = await _proposed(board, ledger, channel="whatsapp")
    board.correct(ALICE, tid, "dismiss", actor="Alice")
    with pytest.raises(HTTPException) as e:
        await autonomy_routes.approve_action_route(row["id"], _req(ALICE), None)
    assert e.value.status_code == 409
    assert ledger.get_action(row["id"])["status"] == "expired"


async def test_a_week_old_next_step_expires_the_48h_ttl_doesnt(board, ledger, jev, model, shared_bot):
    _tid, old = await _proposed(board, ledger, channel="whatsapp")
    _tid2, fresh = await _proposed(board, ledger, channel="telegram")
    _backdate(ledger, old["id"], days=8)
    _backdate(ledger, fresh["id"], hours=60)
    cutoff = (datetime.now(UTC) - timedelta(hours=48)).isoformat()
    assert ledger.expire_stale_proposals(ALICE, cutoff) == 0
    taskboard_nextsteps.expire_closed(ALICE)
    assert ledger.get_action(old["id"])["status"] == "expired"
    assert "7 days" in ledger.get_action(old["id"])["outcome_note"]
    assert ledger.get_action(fresh["id"])["status"] == "awaiting_approval"


# ── Approve, Reject, dispatch ─────────────────────────────────────────


def _agents(monkeypatch, *slugs):
    agents = [{"slug": s, "name": s.title(), "host": "localhost", "port": 9000 + i, "auth": "",
               "tier": "flagship" if s == "strong" else "fast"} for i, s in enumerate(slugs)]
    monkeypatch.setattr(consciousness, "_user_agents", lambda uid: list(agents))
    calls = []

    async def run_tool(agent, tool, args, timeout=60.0, *, automation=None):
        calls.append(SimpleNamespace(agent=agent["slug"], tool=tool, args=args, automation=automation))
        return {"ok": True, "content": "Draft created. ID: d-1", "error": None}

    monkeypatch.setattr(actions, "run_tool_on_agent", run_tool)
    return calls


async def test_an_approved_draft_runs_on_the_tasks_own_agent_as_a_draft(board, ledger, jev, model, monkeypatch):
    calls = _agents(monkeypatch, "strong", "helper")
    _tid, row = await _proposed(board, ledger)
    out = await autonomy_routes.approve_action_route(row["id"], _req(ALICE), None)
    assert out["reliability"] is None   # a board decision teaches Autonomous Work nothing
    (call,) = calls
    assert call.agent == "helper" and call.tool == "google_mail"
    assert call.args["action"] == "create_draft" and call.args["to"] == "ana@x.co"
    assert call.automation["mail_write"] == "allow"
    assert ledger.get_action(row["id"])["status"] == "done"


async def test_dispatch_falls_back_when_the_target_agent_isnt_running_and_says_why(
        board, ledger, jev, model, monkeypatch):
    calls = _agents(monkeypatch, "strong")
    _tid, row = await _proposed(board, ledger)
    await autonomy_routes.approve_action_route(row["id"], _req(ALICE), None)
    assert calls[0].agent == "strong"
    assert "helper isn't running" in ledger.get_action(row["id"])["payload"]["dispatch_note"]
    assert any(r["event"] == "dispatch: target agent unavailable" for r in ledger.list_log(ALICE))


async def test_an_approved_reply_is_delivered_to_the_origin(board, ledger, jev, model, monkeypatch):
    sent = []

    async def deliver(origin, text):
        sent.append((origin, text))
        return True, "ok"

    monkeypatch.setattr(delivery_routes, "deliver_to_origin", deliver)
    _tid, row = await _proposed(board, ledger, channel="whatsapp")
    await autonomy_routes.approve_action_route(row["id"], _req(ALICE), None)
    assert sent == [({"kind": "whatsapp", "address": "38591"}, row["payload"]["text"])]
    got = ledger.get_action(row["id"])
    assert (got["status"], got["outcome"]) == ("done", "success")


async def test_a_tampered_recipient_is_never_sent_to(board, ledger, jev, model, monkeypatch):
    sent = []

    async def deliver(origin, text):
        sent.append(origin)
        return True, "ok"

    monkeypatch.setattr(delivery_routes, "deliver_to_origin", deliver)
    _tid, row = await _proposed(board, ledger, channel="whatsapp")
    ledger.update_payload(row["id"], {"to": "999"})   # not one of the task's own
    await autonomy_routes.approve_action_route(row["id"], _req(ALICE), None)
    assert sent == [] and ledger.get_action(row["id"])["outcome"] == "fail"


async def test_a_board_proposal_never_runs_without_approval(board, ledger, jev, model, monkeypatch):
    sent = []
    monkeypatch.setattr(delivery_routes, "deliver_to_origin", lambda o, t: sent.append(o))
    _tid, row = await _proposed(board, ledger, channel="whatsapp")
    cfg = {"allow_auto_dispatch": True, "autonomy_level": "act", "granted_actions": ["reply.on_origin", "taskboard"]}
    assert fd_dispatch.should_auto_dispatch(cfg, row) is False
    res = await fd_dispatch.dispatch_action(ALICE, row)
    assert res["ok"] is False and sent == []
    assert ledger.get_action(row["id"])["status"] == "awaiting_approval"


async def test_a_rejected_next_step_leaves_its_task_open_and_says_so(board, ledger, jev, model):
    tid, row = await _proposed(board, ledger, channel="whatsapp")
    before = board.get_task(ALICE, tid)
    out = await autonomy_routes.reject_action_route(row["id"], _req(ALICE), None)
    assert out["action"]["status"] == "rejected" and out["reliability"] is None
    after = board.get_task(ALICE, tid)
    assert (after["status"], after["owes_next"]) == ("open", before["owes_next"])
    rejections = [e for e in board.events(ALICE, tid) if e["kind"] == "rejection"]
    assert len(rejections) == 1 and rejections[0]["refs"]["state"] == "rejected"
    # The sweep's own pass over the same row adds no second one.
    assert board.next_state_ref("autonomy", row["id"], "rejected") is None


# ── The catalog and the arbiter ───────────────────────────────────────


def test_message_send_is_gone_and_reply_on_origin_is_never_proposable():
    assert action_catalog.get_action("message.send") is None
    ids = {a["id"] for a in action_catalog.list_catalog()}
    assert "message.send" not in ids and "reply.on_origin" in ids
    spec = action_catalog.get_action("reply.on_origin")
    assert spec["human_only"] is True and action_catalog.may_propose(spec) is False
    assert spec["home"] == "fd" and not action_catalog.is_mail_write(spec)


async def test_the_manual_exerciser_cannot_send_a_free_reply(monkeypatch):
    res = await actions.run_action(ALICE, "reply.on_origin", {"task_id": "t", "to": "1", "text": "hi"},
                                   approved_by_human=True)
    assert res["ok"] is False and "not supported" in res["error"]


def test_board_proposals_stay_out_of_the_arbiters_daily_count(ledger):
    for i in range(3):
        ledger.add_action(ALICE, kind="tool_action", title=f"Board {i}", source="taskboard",
                          status="awaiting_approval")
    ledger.add_action(ALICE, kind="nudge", title="Arbiter's", status="awaiting_approval")
    day = (datetime.now(UTC) - timedelta(hours=24)).isoformat()
    assert ledger.count_since(ALICE, day) == 4
    assert ledger.count_since(ALICE, day, exclude_source="taskboard") == 1


# ── Notifications carry the deep link ─────────────────────────────────


async def test_a_board_proposals_notifications_link_to_its_task(board, ledger, jev, model, monkeypatch):
    tid, row = await _proposed(board, ledger, channel="whatsapp")
    assert f"?taskboard={tid}" not in owner_notify.proposal_text(row, 1)   # no public URL: no link to give
    monkeypatch.setenv("FD_PUBLIC_URL", "https://deck.example")
    text = owner_notify.proposal_text(row, 2)
    assert text.endswith(f"\nhttps://deck.example/?taskboard={tid}") and "(2 waiting)" in text
    _type, _title, body = owner_notify._proposal_bell(row)
    assert body.endswith(f"[Open in Taskboard](?taskboard={tid})")
    seen = {}

    async def fake_owner(uid, kind, text, *, device_text="", cfg=None, bell=None, action_id=""):
        seen.update(text=text, device_text=device_text, bell=bell)
        return {}

    monkeypatch.setattr(owner_notify, "notify_owner", fake_owner)
    await owner_notify.notify_proposal(ALICE, row)
    assert f"taskboard={tid}" in seen["text"] and "taskboard" not in seen["device_text"]   # devices: no link
    assert owner_notify.taskboard_link("tb_1", "https://d.x/app?x=1") == "https://d.x/app?x=1&taskboard=tb_1"


def test_a_nudge_about_a_task_links_to_it(board, monkeypatch):
    tid = _task(board, pending=False)
    board.add_link(ALICE, "fu:fu-1", tid)
    nudge = {"id": "act_1", "user_id": ALICE, "kind": "nudge", "source": "reflection",
             "payload": {"follow_up_id": "fu-1"}}
    said, bell = owner_notify.nudge_texts(nudge, "Ana is waiting for a reply.")
    assert said == "Ana is waiting for a reply." and bell.endswith(f"(?taskboard={tid})")
    monkeypatch.setenv("FD_PUBLIC_URL", "https://deck.example/")
    said, _bell = owner_notify.nudge_texts(nudge, "Ana is waiting for a reply.")
    assert said == f"Ana is waiting for a reply.\nhttps://deck.example/?taskboard={tid}"
    assert owner_notify.nudge_texts({**nudge, "payload": {}}, "x") == ("x", "x")


def test_no_board_no_lookup(monkeypatch):
    monkeypatch.setattr(taskboard_store, "_STORE", None)
    monkeypatch.setattr(taskboard_store, "_DATA_DIR", None)
    assert owner_notify.task_of({"id": "a", "user_id": ALICE, "payload": {}}) == ""


# ── The origin the delivery needs ─────────────────────────────────────


def test_a_device_turn_carries_its_channel_as_its_own_origin_not_the_sessions():
    from captain_claw.flight_deck import device_bridge as db

    class _WS:
        async def send_text(self, text):
            pass

    conn = db._DeviceConn(_WS(), {"id": "u1"}, "stick", lang="", speak=False, screen=(240, 135))
    fields = conn.chat_fields(SimpleNamespace(context_sent=True), "hi", None, None)
    assert fields["turn_origin"] == {"kind": "channel", "address": "device:u1:stick"}
    # The frame's "origin" is the session's durable one (chat_handler stamps it): a
    # device turn never moves where the main conversation's cron results go.
    assert "origin" not in fields


async def test_a_channel_delivery_reaches_a_device_that_isnt_waiting(monkeypatch):
    from captain_claw.flight_deck import glasses_bridge

    seen = []

    async def channel(address):
        return SimpleNamespace(id=address)

    async def broadcast(ch, payload):
        seen.append(payload)

    monkeypatch.setattr(glasses_bridge, "_get_or_create_channel", channel)
    monkeypatch.setattr(glasses_bridge, "_broadcast", broadcast)
    ok, _note = await delivery_routes.deliver_to_origin({"kind": "channel", "address": "device:u1:stick"}, "Booked.")
    assert ok and seen[0]["proactive"] is True and seen[0]["text"] == "Booked."


# ── A next step is its own task's, wherever its key went ──────────────


def _autonomy_pass(board) -> None:
    """The autonomy sweep adapter's pass over the ledger (what a sweep thread runs)."""
    from captain_claw.flight_deck import taskboard_sources

    taskboard_sources._autonomy(taskboard_sources._Pass(board, ALICE, agents={}))


async def test_a_next_step_on_a_task_merged_away_expires_and_its_target_gets_its_own(board, ledger, jev, model,
                                                                                      monkeypatch):
    from captain_claw.flight_deck import taskboard_sources

    monkeypatch.setattr(taskboard_sources, "_user_ids", lambda: None)
    a, b = _task(board, pending=False), _task(board, pending=False)
    first, _ = await taskboard_nextsteps.propose(ALICE, a, "reply_with_result", 0.9, notify=False)
    board.merge(ALICE, a, b)   # A's links, autonomy:<id> among them, move to B
    assert board.task_for_key(ALICE, f"autonomy:{first['id']}") == b
    assert taskboard_nextsteps.waiting_on(ALICE, b) is None
    assert await taskboard_nextsteps.task_gone(first) == "its task was merged into another"
    assert taskboard_nextsteps.expire_closed(ALICE) == 1
    assert ledger.get_action(first["id"])["status"] == "expired"
    second, why = await taskboard_nextsteps.propose(ALICE, b, "remind", 0.9, notify=False)
    assert second is not None, why
    assert taskboard_nextsteps.waiting_proposals(ALICE) == {b: taskboard_nextsteps.proposal_view(
        ledger.get_action(second["id"]))}


async def test_a_sweep_reading_a_next_step_before_it_is_linked_puts_it_on_its_own_task(board, ledger, jev, model,
                                                                                      monkeypatch):
    """propose() writes the ledger row, then links it to the task: a sweep
    between the two finds the row with no task holding its key — it goes on
    its own task (``ref_id``), never a "Proposal" task of its own."""
    import threading

    from captain_claw.flight_deck import taskboard_sources

    monkeypatch.setattr(taskboard_sources, "_user_ids", lambda: None)
    tid = _task(board, pending=False)
    real = board.proposal_made

    def late_link(uid, task_id, action_id, title):
        sweep = threading.Thread(target=_autonomy_pass, args=(board,))
        sweep.start()
        sweep.join()
        return real(uid, task_id, action_id, title)

    monkeypatch.setattr(board, "proposal_made", late_link)
    row, why = await taskboard_nextsteps.propose(ALICE, tid, "reply_with_result", 0.9, notify=False)
    assert row is not None, why
    assert board.task_for_key(ALICE, f"autonomy:{row['id']}") == tid
    assert [r["kind"] for r in board._conn.execute("SELECT kind FROM tasks WHERE user_id = ?", (ALICE,))] == \
        ["request"]
    made = [e for e in board.events(ALICE, tid) if e["source_ref"] == f"{row['id']}:made"]
    assert len(made) == 1 and made[0]["actor"] == "Taskboard"
    assert taskboard_nextsteps.waiting_on(ALICE, tid)["id"] == row["id"]
    again, why = await taskboard_nextsteps.propose(ALICE, tid, "remind", 0.9, notify=False)
    assert again is None and "already waits" in why
    # Its task gone: the row goes on no task at all.
    board.delete_agent(None, "helper")
    _autonomy_pass(board)
    assert board.query(ALICE, view="all")["total"] == 0


async def test_a_week_old_next_step_is_never_approved_listed_or_counted(board, ledger, jev, model, monkeypatch):
    sent = []

    async def deliver(origin, text):
        sent.append(origin)
        return True, "ok"

    monkeypatch.setattr(delivery_routes, "deliver_to_origin", deliver)
    _tid, old = await _proposed(board, ledger, channel="whatsapp")
    _backdate(ledger, old["id"], days=8)
    assert owner_notify.waiting_count(ALICE) == 0   # the channels' "N waiting"
    assert ledger.get_action(old["id"])["status"] == "expired"
    _tid, other = await _proposed(board, ledger, channel="whatsapp")
    _backdate(ledger, other["id"], days=8)
    listed = await autonomy_routes.list_actions_route(_req(ALICE), "awaiting_approval", 100, None)
    assert listed["actions"] == []   # the bell's list
    _tid, last = await _proposed(board, ledger, channel="whatsapp")
    _backdate(ledger, last["id"], days=8)
    with pytest.raises(HTTPException) as e:
        await autonomy_routes.approve_action_route(last["id"], _req(ALICE), None)
    assert e.value.status_code == 409 and "7 days" in e.value.detail
    assert sent == [] and ledger.get_action(last["id"])["status"] == "expired"


async def test_an_approved_telegram_reply_goes_out_through_the_decks_own_bot(board, ledger, jev, model, shared_bot,
                                                                            monkeypatch):
    from captain_claw.flight_deck import telegram_out

    async def env_bot(*a, **kw):
        raise AssertionError("the env token's bot is never the chat's")

    monkeypatch.setattr(telegram_out, "send_telegram_multi", env_bot)
    _tid, row = await _proposed(board, ledger, channel="telegram")
    await autonomy_routes.approve_action_route(row["id"], _req(ALICE), None)
    assert shared_bot == [("777", row["payload"]["text"])]
    assert ledger.get_action(row["id"])["outcome"] == "success"
    # The deck's bot no longer holds the chat for this agent: nothing is sent, and the task says why.
    _tid, row = await _proposed(board, ledger, channel="telegram")
    from captain_claw.flight_deck import telegram_shared
    telegram_shared._state.links_by_tg.clear()
    await autonomy_routes.approve_action_route(row["id"], _req(ALICE), None)
    got = ledger.get_action(row["id"])
    assert len(shared_bot) == 1 and got["outcome"] == "fail" and "bot holds" in got["outcome_note"]


async def test_a_device_reply_goes_out_only_while_someone_is_on_its_channel(board, ledger, jev, model, monkeypatch):
    from captain_claw.flight_deck import glasses_bridge
    from captain_claw.flight_deck.bus_client import BusClient

    channel = f"device:{ALICE}:stick"
    seen: list[dict] = []

    class _Device(BusClient):
        def __init__(self, closed):
            self.closed = closed

        async def on_bus(self, ev):
            seen.append(ev)

    state = glasses_bridge._ChannelState(channel_id=channel)
    monkeypatch.setitem(glasses_bridge._channels, channel, state)
    _tid, off = await _proposed(board, ledger, channel="device:stick")
    state.callback_subscribers.append(_Device(closed=True).on_bus)   # the device went away mid-turn
    await autonomy_routes.approve_action_route(off["id"], _req(ALICE), None)
    got = ledger.get_action(off["id"])
    assert seen == [] and got["outcome"] == "fail" and "nobody is on that channel" in got["outcome_note"]
    _tid, on = await _proposed(board, ledger, channel="device:stick")
    state.callback_subscribers[:] = [_Device(closed=False).on_bus]
    await autonomy_routes.approve_action_route(on["id"], _req(ALICE), None)
    assert [e["text"] for e in seen] == [on["payload"]["text"]] and seen[0]["proactive"] is True
    assert ledger.get_action(on["id"])["outcome"] == "success"
