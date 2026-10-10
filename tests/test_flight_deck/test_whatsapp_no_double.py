"""An autonomous nudge (or a scheduled job) reaches WhatsApp once — the
formatted copy.

The automated turn runs on the automation lane; its result is mirrored into
the main chat, whose channels the WhatsApp bridge relays (markdown stripped),
AND Flight Deck pushes the result itself (formatted). Flight Deck now marks
the turns it delivers itself (``fd_delivers``); the agent's mirror carries the
mark, and the WhatsApp relay skips it — however late the mirror is flushed.
"""

from __future__ import annotations

import types

import pytest

from captain_claw import mail_authority
from captain_claw.flight_deck import basna_routes, fd_dispatch
from captain_claw.flight_deck import glasses_bridge as gb
from captain_claw.flight_deck import meta_webhook_bridge as mwb
from captain_claw.flight_deck import whatsapp_bridge as wb
from captain_claw.session import Session

WAID = "385911111111"
CHANNEL = f"whatsapp:{WAID}"
NUDGE = "**Caught it** — the LoI had a broken date.\n- fixed in place"


class _Resp:
    status_code = 200

    def json(self):
        return {"messages": [{"id": "wamid.out"}]}


class _Client:
    def __init__(self, sent):
        self.sent = sent

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, url, headers=None, json=None):
        if json.get("type") == "text":
            self.sent.append(json["text"]["body"])
        return _Resp()


@pytest.fixture
def wa(monkeypatch):
    """The real relay + push, down to a fake Graph API."""
    sent: list = []
    monkeypatch.setenv("WHATSAPP_ACCESS_TOKEN", "tok")
    monkeypatch.setenv("WHATSAPP_PHONE_NUMBER_ID", "123")
    monkeypatch.setenv("WHATSAPP_ALLOWED_WAIDS", WAID)
    monkeypatch.delenv("WHATSAPP_AUDIO_REPLY", raising=False)
    monkeypatch.setattr(wb.httpx, "AsyncClient", lambda *a, **kw: _Client(sent))
    monkeypatch.setattr(wb, "_RECENT_REPLY_SENDS", {})
    monkeypatch.setattr(wb, "_MUTED_UNTIL", {})
    monkeypatch.setattr(wb, "_WIRED_CHANNELS", set())
    monkeypatch.setattr(wb, "_CHANNEL_WAIDS", {CHANNEL: {WAID}})
    ch = gb._ChannelState(channel_id=CHANNEL)
    monkeypatch.setitem(gb._channels, CHANNEL, ch)
    wb._ensure_whatsapp_forwarding(CHANNEL)

    async def agent_frame(data):
        """An agent WS frame through the real glasses pump and every relay."""
        async def _bc(_ch, payload):
            for cb in list(ch.callback_subscribers):
                await cb(payload)

        real = gb._broadcast
        gb._broadcast = _bc
        try:
            await gb._forward_agent_msg(ch, data)
        finally:
            gb._broadcast = real

    return types.SimpleNamespace(sent=sent, frame=agent_frame)


def _mirror(text, *, fd_delivers=False):
    """The main chat's mirror of an automated result (flush_automation_results)."""
    return {"type": "chat_message", "role": "assistant", "content": text,
            "automation_lane": "AUTO", **({"fd_delivers": True} if fd_delivers else {})}


async def test_a_nudge_arrives_once_formatted(wa):
    await wa.frame(_mirror(NUDGE, fd_delivers=True))         # the mirror, relayed first
    assert await wb.push_to_waid(WAID, NUDGE) is True        # Flight Deck's push
    assert wa.sent == ["*Caught it* — the LoI had a broken date.\n- fixed in place"]


async def test_a_mirror_flushed_hours_later_is_still_not_sent(wa):
    assert await wb.push_to_waid(WAID, NUDGE) is True
    wb._RECENT_REPLY_SENDS.clear()                            # long after any dedup window
    await wa.frame(_mirror(NUDGE, fd_delivers=True))
    assert len(wa.sent) == 1


async def test_same_text_runs_each_arrive_once(wa):
    for _ in range(3):                                        # a job: "No new mail." every run
        await wb.push_to_waid(WAID, "No new mail.")
        await wa.frame(_mirror("No new mail.", fd_delivers=True))
        wb._RECENT_REPLY_SENDS.clear()
    assert wa.sent == ["No new mail."] * 3


async def test_a_nudge_that_ran_on_the_main_lane_arrives_once(wa):
    """Automation lane off (or a Flight Deck worker): the turn's own reply is
    what the bridge sees — marked, so it isn't relayed either."""
    await wa.frame({"type": "chat_message", "role": "assistant", "content": NUDGE, "fd_delivers": True})
    assert await wb.push_to_waid(WAID, NUDGE) is True
    assert wa.sent == ["*Caught it* — the LoI had a broken date.\n- fixed in place"]


async def test_a_result_flight_deck_does_not_deliver_is_still_relayed(wa):
    await wa.frame(_mirror("**Plan finished**: 3 steps done."))   # e.g. a plan or cron result
    assert wa.sent == ["Plan finished: 3 steps done."]


async def test_a_relayed_automated_result_honours_mute(wa):
    wb._MUTED_UNTIL[WAID] = float("inf")
    await wa.frame(_mirror("Plan finished."))
    assert wa.sent == []
    await wa.frame({"type": "chat_message", "role": "assistant", "content": "a direct answer"})
    assert wa.sent == ["a direct answer"]


async def test_an_ordinary_reply_is_unchanged(wa):
    await wa.frame({"type": "chat_message", "role": "assistant", "content": "**Done.**"})
    assert wa.sent == ["Done."]


async def test_messenger_still_gets_every_reply_plain(monkeypatch):
    ch = gb._ChannelState(channel_id="messenger-test")
    monkeypatch.setitem(gb._channels, "messenger-test", ch)
    out = []

    async def _send(rid, text):
        out.append(text)

    mwb.register_channel_callback(channel_id="messenger-test", wired_set=set(),
                                  recipients_for_channel=lambda c: ["psid"], send_one=_send)
    await ch.callback_subscribers[-1]({"type": "agent", "text": "**bold**", "fd_delivers": True,
                                       "automation_lane": "AUTO"})
    assert out == ["bold"]


# ── the mark, end to end ───────────────────────────────────────────────────


class _SessionManager:
    async def save_session(self, session):
        return None


async def test_the_agents_mirror_carries_the_mark_even_when_flushed_late():
    from captain_claw.web import chat_handler

    main = types.SimpleNamespace(session=Session(id="main", name="default"),
                                 session_manager=_SessionManager())
    sent: list = []
    server = types.SimpleNamespace(agent=main, _busy=True,
                                   _broadcast=lambda msg, exclude=None: sent.append(msg))
    ticks = chat_handler._MIRROR_WAIT_TICKS
    chat_handler._MIRROR_WAIT_TICKS = 1                       # lane A stays busy past the wait
    try:
        await chat_handler._mirror_automation_result(
            server, None, mail_authority.Authority(mode="automated", kind="autonomy"),
            NUDGE, "AUTO", fd_delivers=True)
        await chat_handler._mirror_automation_result(
            server, None, mail_authority.Authority(mode="automated", kind="plan"),
            "Plan finished.", "AUTO")
    finally:
        chat_handler._MIRROR_WAIT_TICKS = ticks
    assert sent == []
    server._busy = False                                      # the next lane-A turn, hours later
    assert await chat_handler.flush_automation_results(server) == 2
    assert sent[0]["fd_delivers"] is True and "fd_delivers" not in sent[1]


async def test_ws_handler_reads_the_mark(monkeypatch):
    from captain_claw.web import chat_handler
    from captain_claw.web.ws_handler import handle_ws_message

    got = []

    async def _chat(server, ws, content, **kw):
        got.append(kw)

    monkeypatch.setattr(chat_handler, "handle_chat", _chat)

    async def _send(ws, msg):
        return None

    server = types.SimpleNamespace(agent=types.SimpleNamespace(plan_mode_auto=False), _send=_send,
                                   _broadcast=lambda msg: None)
    for marker in ({"kind": "autonomy", "fd_delivers": True}, {"kind": "autonomy"},
                   {"kind": "autonomy", "fd_delivers": "yes"}):
        await handle_ws_message(server, object(), {"type": "chat", "content": "go", "automation": marker})
    assert [kw["fd_delivers"] for kw in got] == [True, False, False]


@pytest.mark.parametrize("auth,overrides,kind,marked", [
    ("true", {"notify_waid": "111"}, "nudge", True),                 # FD pushes it
    ("false", {}, "nudge", True),                                     # every allowlisted number
    ("true", {"notify_waid": "111", "nudge_to_whatsapp": False}, "nudge", True),  # kept off WhatsApp
    ("true", {}, "nudge", False),                                     # no number: the mirror stays
    ("true", {"notify_waid": "999"}, "nudge", False),                 # not allowlisted: same
    ("true", {"notify_waid": "111"}, "research", False),              # not a nudge
])
async def test_flight_deck_marks_a_nudge_only_when_it_decides_the_delivery(
        tmp_path, monkeypatch, auth, overrides, kind, marked):
    from captain_claw.config import AutonomousWorkConfig
    from captain_claw.flight_deck import autonomy

    store = autonomy.AutonomyStore(tmp_path / "autonomy.db")
    monkeypatch.setattr(autonomy, "_STORE", store)
    monkeypatch.setattr(autonomy, "global_defaults", lambda: AutonomousWorkConfig().model_dump())
    monkeypatch.setenv("FD_AUTH_ENABLED", auth)
    monkeypatch.setenv("WHATSAPP_ALLOWED_WAIDS", "111,222")
    markers = []

    async def fake_dispatch(port, auth_, instruction, timeout, **kw):
        markers.append(kw.get("automation"))
        return {"ok": True, "output": "done"}

    async def fake_push(waid, text):
        return True

    monkeypatch.setattr(basna_routes, "_dispatch_one", fake_dispatch)
    monkeypatch.setattr(wb, "push_to_waid", fake_push)
    store.set_overrides("u1", overrides)
    row = store.add_action("u1", kind=kind, title=f"a {kind}", risk="low", status="dispatched")
    await fd_dispatch._execute_and_judge("u1", row, {"slug": "a", "port": 1, "auth": ""})
    assert markers[0]["kind"] == "autonomy"
    assert bool(markers[0].get("fd_delivers")) is marked


def test_the_mark_holds_only_while_flight_deck_listens():
    from captain_claw.web import chat_handler

    auto = mail_authority.Authority(mode="automated", kind="autonomy")
    open_ws, closed_ws = types.SimpleNamespace(closed=False), types.SimpleNamespace(closed=True)
    assert chat_handler.fd_still_delivers(True, auto, open_ws) is True
    assert chat_handler.fd_still_delivers(True, auto, closed_ws) is False   # FD gave up waiting
    assert chat_handler.fd_still_delivers(True, None, open_ws) is False     # not an automated turn
    assert chat_handler.fd_still_delivers(False, auto, open_ws) is False


async def test_a_scheduled_job_never_takes_another_turns_mirror_as_its_reply(monkeypatch):
    from captain_claw.flight_deck import fd_scheduler as sched

    class _WS:
        async def send(self, raw):
            for cb in list(holder["ch"].callback_subscribers):
                await cb({"type": "agent", "text": "a nudge's result", "automation_lane": "AUTO"})
                await cb({"type": "agent", "text": "the job's own reply"})

    holder = {}

    async def _bind(ch, host, port, auth):
        holder["ch"] = ch
        ch.agent_ws = _WS()

    async def _remove(channel_id):
        return None

    monkeypatch.setattr(sched, "_ensure_agent_binding", _bind)
    monkeypatch.setattr(sched, "_remove_channel", _remove)
    reply = await sched.run_prompt_and_capture(host="h", port=1, auth="", prompt="Check my mail.",
                                               automation={"kind": "fd_scheduler"})
    assert reply == "the job's own reply"


async def test_an_agent_cron_result_flight_deck_delivered_is_not_relayed_again(monkeypatch):
    from captain_claw import delivery
    from captain_claw.web_server import WebServer

    sent: list = []
    server = WebServer.__new__(WebServer)
    server.agent = types.SimpleNamespace()
    server._telegram_bridge = None
    server._broadcast = lambda msg, exclude=None: sent.append(msg)

    async def _delivered(agent, sid, text):
        return True

    async def _not_delivered(agent, sid, text):
        return False

    ctx = server._get_web_runtime_context()
    monkeypatch.setattr(delivery, "deliver_to_origin", _delivered)
    await ctx.on_cron_output("s1", "Weekly report is ready.")
    monkeypatch.setattr(delivery, "deliver_to_origin", _not_delivered)
    await ctx.on_cron_output("s1", "Local only.")
    assert sent[0]["fd_delivers"] is True and sent[0]["proactive"] is True
    assert "fd_delivers" not in sent[1] and sent[1]["proactive"] is True


async def test_a_proactive_message_honours_mute(wa):
    wb._MUTED_UNTIL[WAID] = float("inf")
    await wa.frame({"type": "chat_message", "role": "assistant", "content": "Cron result", "proactive": True})
    assert wa.sent == []


async def test_flight_deck_marks_every_scheduled_job(monkeypatch):
    from captain_claw.flight_deck import fd_scheduler as sched

    seen = {}
    monkeypatch.setattr(sched, "resolve_agent_by_slug", lambda *a, **k: ("127.0.0.1", 1234, "tok"))

    async def _capture(*, host, port, auth, prompt, automation=None, **kw):
        seen["automation"] = automation
        return "No new mail."

    async def _deliver(kind, target, text):
        return (True, "ok")

    monkeypatch.setattr(sched, "run_prompt_and_capture", _capture)
    monkeypatch.setattr(sched, "_deliver", _deliver)
    await sched.execute_job({"agent_slug": "x", "prompt": "Check my mail.", "delivery_kind": "whatsapp",
                             "delivery_target": WAID, "ignore_quiet_hours": True}, force=True)
    assert seen["automation"]["fd_delivers"] is True and seen["automation"]["kind"] == "fd_scheduler"
