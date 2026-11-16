"""Only the nudges worth a ping reach the phone (phone_gate.py).

Every nudge still lands in the agent's chat, the Line and the bell; WhatsApp
and Telegram get the ones that can't wait. Rules first (a message that only
says the assistant couldn't check something, the same email thread or calendar
event again within hours, the user chatting right now), then Jev, then rules
when Jev can't answer. Each decision is kept on the action and in the log."""

from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime, timedelta

import pytest

from captain_claw import jev_client
from captain_claw.config import AutonomousWorkConfig
from captain_claw.flight_deck import (
    autonomy,
    basna_routes,
    device_bridge,
    events,
    fd_dispatch,
    fd_scheduler,
    jev_routes,
    phone_gate,
    whatsapp_bridge,
)

ALICE = "user-alice"
WAID = "111"


@pytest.fixture
def store(tmp_path, monkeypatch):
    s = autonomy.AutonomyStore(tmp_path / "autonomy.db")
    monkeypatch.setattr(autonomy, "_STORE", s)
    base = AutonomousWorkConfig().model_dump()
    monkeypatch.setattr(autonomy, "global_defaults", lambda: dict(base))
    monkeypatch.setattr(events, "_STORE", events.EventsStore(tmp_path / "events.db"))
    monkeypatch.setenv("FD_AUTH_ENABLED", "true")
    monkeypatch.setenv("WHATSAPP_ALLOWED_WAIDS", WAID)
    monkeypatch.setattr(phone_gate, "_convo", {})
    monkeypatch.setattr(phone_gate, "_last_in", {})
    monkeypatch.setattr(phone_gate, "_held", {})
    s.set_overrides(ALICE, {"notify_waid": WAID, "quiet_hours_start": 0, "quiet_hours_end": 0})
    return s


@pytest.fixture
def jev(monkeypatch):
    """``jev["values"]`` is what Jev answers (None: not connected);
    ``jev["states"]`` what it was shown."""
    box: dict = {"values": {"needs_user_soon": 0.9, "already_told": 0.1, "couldnt_check": 0.05},
                 "states": [], "raise": None}

    async def fake_ask(owner, state, questions, model=None, *, source, agent=""):
        box["states"].append(state)
        assert source == phone_gate.JEV_SOURCE and set(questions) == set(phone_gate.QUESTIONS)
        if box["raise"] is not None:
            raise box["raise"]
        if box["values"] is None:
            raise jev_routes.JevNotConnected()
        return {"values": dict(box["values"]), "answers": {}, "model": "jev-latest"}

    monkeypatch.setattr(jev_routes, "ask_for_owner", fake_ask)
    return box


@pytest.fixture
def phones(monkeypatch):
    out: dict[str, list] = {"whatsapp": [], "telegram": [], "bell": []}

    async def fake_push(waid, text):
        out["whatsapp"].append((waid, text))
        return True

    async def fake_telegram(owner, text):
        out["telegram"].append((owner, text))
        return {"via": "telegram", "ok": True, "note": "sent"}

    async def no_devices(user_id, count, text="", beep=False):
        return []

    class _DB:
        async def add_notification(self, user_id, type, title, body="", ref_type="", ref_id="", *, source=""):
            out["bell"].append(title)
            return "n1"

    from captain_claw.flight_deck import auth
    monkeypatch.setattr(whatsapp_bridge, "push_to_waid", fake_push)
    monkeypatch.setattr(fd_scheduler, "_deliver_telegram_owner", fake_telegram)
    monkeypatch.setattr(device_bridge, "push_waiting", no_devices)
    monkeypatch.setattr(auth, "get_db", lambda: _DB())
    return out


def _cfg():
    return autonomy.resolve_config(ALICE)


def _action(store, title="Remind about the call", rationale="", payload=None):
    return store.add_action(ALICE, kind="nudge", title=title, rationale=rationale,
                            risk="low", status="dispatched", payload=payload or {})


def _phoned(store, title, subject, minutes_ago=30, text="Earlier ping."):
    a = _action(store, title=title)
    at = (datetime.now(UTC) - timedelta(minutes=minutes_ago)).isoformat()
    store.update_payload(a["id"], {"phone": {"sent": True, "at": at, "subject": subject, "text": text}})
    return a


def _calendar_event(start_in_h=3.0, event_id="cal-1"):
    start = (datetime.now(UTC) + timedelta(hours=start_in_h)).isoformat()
    ev = events.get_store().add_event(
        ALICE, source="calendar", event_type="upcoming", summary="Upcoming: 'Shoot'",
        metadata={"event_id": event_id, "start": start}, dedup_key=f"cal_up:{event_id}:{start}")
    return ev["id"]


# ── rules that need no model ──────────────────────────────────────────

@pytest.mark.parametrize("text", [
    "I can't reconcile these yet. I haven't opened the application in this session.",
    "I can't close the count gap yet. I've found two of them.",
    "**The memo isn't in this session, so I can't check its status.** The meeting is tomorrow.",
    "**I can't see the memo's status from here.** I don't have it.",
    "I haven't been able to open the thread. Try again later.",
    "Sorry, I couldn't find the invite.",
])
def test_a_message_opening_with_couldnt_check_is_spotted(text):
    assert phone_gate.couldnt_check_lead(text)


@pytest.mark.parametrize("text", [
    "**The draft to the client is already in Drafts** (created 15 Nov at 22:16), and nothing "
    "has been sent. I haven't sent it.",
    "**The client is waiting for a reply.** I can't send it without your approval.",
    "Valued partner asks whether the call should still happen.",
    "**IC meeting, Tue 10:00** is confirmed. I checked it by ID just now.",
    "",
])
def test_a_real_finding_is_not_mistaken_for_couldnt_check(text):
    assert not phone_gate.couldnt_check_lead(text)


async def test_couldnt_check_stays_off_the_phone_without_asking_jev(store, jev):
    v = await phone_gate.judge(ALICE, _action(store), "I can't see the memo from here.", _cfg())
    assert (v.send, v.why) == (False, phone_gate.COULDNT_CHECK) and jev["states"] == []


def test_ground_of_names_the_thread_event_or_follow_up(store):
    g = phone_gate.ground_of({"payload": {"thread_key": "gmail:t1", "event_ref": "x"}})
    assert (g.subject, g.kind) == ("gmail:t1", "email")
    g = phone_gate.ground_of({"payload": {"event_ref": _calendar_event(event_id="cal-9")}})
    assert (g.subject, g.kind) == ("cal:cal-9", "event") and g.start is not None
    g = phone_gate.ground_of({"payload": {"follow_up_id": "fu1"}})
    assert (g.subject, g.kind) == ("fu:fu1", "follow_up")
    assert phone_gate.ground_of({"payload": {}}).subject == ""


async def test_the_same_event_reworded_is_a_repeat_within_the_window(store, jev):
    """Two nudges about one calendar event come from two event rows (an
    "upcoming" and a "changed" one) with different titles: the event id ties them."""
    _phoned(store, "Confirm prep for the shoot", "cal:cal-1", minutes_ago=9)
    second = _action(store, title="Settle the location for the shoot",
                     payload={"event_ref": _calendar_event(event_id="cal-1")})
    v = await phone_gate.judge(ALICE, second, "The shoot still has no location.", _cfg())
    assert (v.send, v.why) == (False, phone_gate.REPEAT) and jev["states"] == []


async def test_a_repeat_past_the_window_is_judged_again(store, jev):
    _phoned(store, "Reply to the client", "gmail:t1", minutes_ago=7 * 60)
    v = await phone_gate.judge(ALICE, _action(store, payload={"thread_key": "gmail:t1"}),
                               "The client wrote again.", _cfg())
    assert v.send and v.why == phone_gate.JEV_SEND


async def test_while_the_user_is_chatting_the_nudge_is_held(store, jev):
    phone_gate.note_inbound(phone_gate.wa_key(WAID), "what's next in my calendar?")
    v = await phone_gate.judge(ALICE, _action(store), "Your call is at 13:30.", _cfg())
    assert (v.send, v.held, v.why) == (False, True, phone_gate.CHATTING) and jev["states"] == []


# ── Jev first ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("values,send,why", [
    ({"needs_user_soon": 0.9, "already_told": 0.1, "couldnt_check": 0.0}, True, phone_gate.JEV_SEND),
    ({"needs_user_soon": 0.3, "already_told": 0.1, "couldnt_check": 0.0}, False, phone_gate.JEV_NOT_URGENT),
    ({"needs_user_soon": 0.9, "already_told": 0.8, "couldnt_check": 0.0}, False, phone_gate.JEV_ALREADY_TOLD),
    ({"needs_user_soon": 0.9, "already_told": 0.1, "couldnt_check": 0.7}, False, phone_gate.JEV_COULDNT_CHECK),
])
async def test_jev_decides(store, jev, values, send, why):
    jev["values"] = values
    v = await phone_gate.judge(ALICE, _action(store), "Someone is waiting on you.", _cfg())
    assert (v.send, v.why, v.via) == (send, why, "jev") and v.p == values


async def test_jev_reads_what_reached_the_phone_and_the_conversation(store, jev):
    _phoned(store, "Earlier", "", minutes_ago=60, text="The client is waiting for a reply.")
    key = phone_gate.wa_key(WAID)
    phone_gate.note_inbound(key, "check the applications")
    phone_gate.note_outbound(key, "Five applications came in today.")
    phone_gate._last_in[key] = time.time() - 3600  # quiet again
    await phone_gate.judge(ALICE, _action(store, title="Screen the five applications"),
                           "Five applications are unscreened.", _cfg())
    state = jev["states"][-1]
    assert state["message"] == "Five applications are unscreened."
    assert state["about"] == "Screen the five applications"
    assert [s["message"] for s in state["sent_to_phone_recently"]] == ["The client is waiting for a reply."]
    assert [(c["from"], c["text"]) for c in state["recent_conversation"]] == [
        ("user", "check the applications"), ("assistant", "Five applications came in today.")]


async def test_telegram_edits_replace_the_narration_line():
    key = phone_gate.tg_key("42")
    phone_gate._convo.pop(key, None)
    phone_gate.note_outbound(key, "Thinking…")
    phone_gate.note_outbound(key, "Here is the answer.", replace_last=True)
    assert [c["text"] for c in phone_gate.recent_conversation([key])] == ["Here is the answer."]
    phone_gate._convo.pop(key, None)


# ── rules when Jev can't answer ───────────────────────────────────────

async def test_without_jev_an_email_someone_sent_goes(store, jev):
    jev["values"] = None
    v = await phone_gate.judge(ALICE, _action(store, payload={"thread_key": "gmail:t2"}),
                               "The partner asks whether the call is still on.", _cfg())
    assert (v.send, v.why, v.via) == (True, phone_gate.RULES_EMAIL, "rules")


async def test_without_jev_an_event_within_the_day_goes_and_a_later_one_waits(store, jev):
    jev["values"] = None
    soon = _action(store, payload={"event_ref": _calendar_event(start_in_h=3, event_id="a")})
    later = _action(store, payload={"event_ref": _calendar_event(start_in_h=60, event_id="b")})
    assert (await phone_gate.judge(ALICE, soon, "Prep the meeting.", _cfg())).why == phone_gate.RULES_EVENT_SOON
    assert (await phone_gate.judge(ALICE, later, "Prep the meeting.", _cfg())).why == phone_gate.RULES_CAN_WAIT


async def test_without_jev_a_today_deadline_goes_and_the_rest_waits(store, jev):
    jev["raise"] = jev_client.JevError(500, "down")
    today = _action(store, title="The draft is still unsent",
                    rationale="It needs a send decision from you today.")
    v = await phone_gate.judge(ALICE, today, "The draft is in Drafts.", _cfg())
    assert (v.send, v.why) == (True, phone_gate.RULES_TODAY)
    v = await phone_gate.judge(ALICE, _action(store, title="Deal memo is due before the meeting"),
                               "The memo needs a recommendation.", _cfg())
    assert (v.send, v.why) == (False, phone_gate.RULES_CAN_WAIT)


async def test_filter_off_sends_every_nudge_without_asking(store, jev):
    store.set_overrides(ALICE, {**store.get_overrides(ALICE), "phone_filter": False})
    v = await phone_gate.judge(ALICE, _action(store), "I can't see the memo from here.", _cfg())
    assert (v.send, v.why) == (True, phone_gate.FILTER_OFF) and jev["states"] == []


# ── through the nudge's dispatch ──────────────────────────────────────

async def _dispatch(store, monkeypatch, output, **action_kw):
    async def fake_dispatch(port, auth, instruction, timeout, **kw):
        return {"ok": True, "output": output}

    monkeypatch.setattr(basna_routes, "_dispatch_one", fake_dispatch)
    row = _action(store, **action_kw)
    await fd_dispatch._execute_and_judge(ALICE, row, {"slug": "a", "port": 1, "auth": ""})
    return store.get_action(row["id"])


async def test_a_nudge_jev_says_can_wait_reaches_the_bell_not_the_phone(store, jev, phones, monkeypatch):
    jev["values"] = {"needs_user_soon": 0.2, "already_told": 0.1, "couldnt_check": 0.0}
    done = await _dispatch(store, monkeypatch, "A call was booked for December.")
    assert done["status"] == "done" and done["outcome"] == "success"
    assert phones["whatsapp"] == [] and phones["bell"] == ["Nudge from a"]
    assert done["payload"]["phone"]["sent"] is False
    assert done["payload"]["phone"]["why"] == phone_gate.JEV_NOT_URGENT
    assert any(r["event"] == "nudge kept off the phone" for r in store.list_log(ALICE))


async def test_a_nudge_jev_says_send_rings_the_phone_once(store, jev, phones, monkeypatch):
    store.set_overrides(ALICE, {**store.get_overrides(ALICE), "notify_telegram": True})
    monkeypatch.setattr(fd_scheduler, "_owner_telegram_chat", lambda owner: "42")
    done = await _dispatch(store, monkeypatch, "The partner is waiting for your reply.")
    assert phones["whatsapp"] == [(WAID, "The partner is waiting for your reply.")]
    assert phones["telegram"] == [(ALICE, "The partner is waiting for your reply.")]
    assert done["payload"]["phone"]["sent"] is True
    assert any(r["event"] == "nudge → phone" for r in store.list_log(ALICE))


async def test_a_held_nudge_rings_once_the_user_goes_quiet(store, jev, phones, monkeypatch):
    monkeypatch.setattr(phone_gate, "HOLD_POLL_S", 0.05)
    key = phone_gate.wa_key(WAID)
    phone_gate.note_inbound(key, "hi")
    phone_gate._last_in[key] = time.time() - (10 * 60 - 0.2)   # quiet in ~0.2 s
    done = await _dispatch(store, monkeypatch, "The partner is waiting for your reply.")
    assert phones["whatsapp"] == [] and done["payload"]["phone"]["held"] is True
    await asyncio.wait_for(asyncio.gather(*phone_gate._held.values()), 5)
    assert phones["whatsapp"] == [(WAID, "The partner is waiting for your reply.")]
    assert store.get_action(done["id"])["payload"]["phone"]["sent"] is True


async def test_a_held_nudge_that_only_echoed_the_chat_never_rings(store, jev, phones, monkeypatch):
    monkeypatch.setattr(phone_gate, "HOLD_POLL_S", 0.05)
    key = phone_gate.wa_key(WAID)
    phone_gate.note_inbound(key, "check the applications")
    phone_gate._last_in[key] = time.time() - (10 * 60 - 0.2)
    jev["values"] = {"needs_user_soon": 0.8, "already_told": 0.9, "couldnt_check": 0.0}
    done = await _dispatch(store, monkeypatch, "Five applications are unscreened.")
    await asyncio.wait_for(asyncio.gather(*phone_gate._held.values()), 5)
    assert phones["whatsapp"] == []
    phone = store.get_action(done["id"])["payload"]["phone"]
    assert (phone["sent"], phone["why"]) == (False, phone_gate.JEV_ALREADY_TOLD)


async def test_a_bug_in_the_gate_never_loses_the_nudge(store, jev, phones, monkeypatch):
    async def broken(*a, **kw):
        raise RuntimeError("boom")

    monkeypatch.setattr(phone_gate, "judge", broken)
    await _dispatch(store, monkeypatch, "The partner is waiting for your reply.")
    assert phones["whatsapp"] == [(WAID, "The partner is waiting for your reply.")]


async def test_with_no_phone_reachable_nothing_is_asked(store, jev, phones, monkeypatch):
    store.set_overrides(ALICE, {**store.get_overrides(ALICE), "notify_waid": ""})
    await _dispatch(store, monkeypatch, "The partner is waiting for your reply.")
    assert jev["states"] == [] and phones["whatsapp"] == []
    assert any(r["event"] == "nudge → whatsapp skipped" for r in store.list_log(ALICE))


def test_shipped_defaults():
    cfg = AutonomousWorkConfig()
    assert cfg.phone_filter is True
    assert (cfg.phone_repeat_hours, cfg.phone_hold_while_chatting_min) == (6, 10)
