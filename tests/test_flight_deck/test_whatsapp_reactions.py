"""WhatsApp emoji reactions on inbound user messages.

Before the user's message reaches the agent, the bridge makes a short side
call to the TARGET agent's own LLM (``POST /api/llm/complete`` — its provider,
model and key, no agent loop) asking for one emoji or NONE, and sends a Cloud
API reaction when one fits. It runs in parallel and never delays the forward;
any failure means no reaction, never an error to the user.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
import types

import httpx
import pytest

from captain_claw.flight_deck import whatsapp_bridge as wb

_REAL_MAYBE_REACT = wb._maybe_react  # the inbound fixture fakes it

GRAPH_PID = "1234567890"
GRAPH_URL = f"https://graph.facebook.com/v18.0/{GRAPH_PID}/messages"
WAID = "385911111111"
WAMID = "wamid.HBgMtest"
HEART = "\u2764\ufe0f"


# ── httpx fake: records every POST, answers per URL ────────────────────────


class _Resp:
    def __init__(self, status_code: int = 200, payload=None, text: str = ""):
        self.status_code = status_code
        self._payload = payload
        self.text = text or (json.dumps(payload) if payload is not None else "")

    def json(self):
        if self._payload is None:
            raise ValueError("not json")
        return self._payload


class _Recorder:
    def __init__(self):
        self.calls: list[dict] = []
        self.llm = _Resp(200, {"ok": True, "content": "🎉"})
        self.llm_exc: Exception | None = None
        self.during_llm = None  # optional async hook, runs mid-classification
        self.graph = _Resp(200, {"messages": [{"id": "wamid.out"}]})

    def client(self, *a, **kw):
        return _FakeClient(self, kw.get("timeout"))

    def llm_calls(self):
        return [c for c in self.calls if c["url"].endswith("/api/llm/complete")]

    def graph_calls(self):
        return [c for c in self.calls if c["url"] == GRAPH_URL]

    def reactions(self):
        return [c for c in self.graph_calls() if c["json"].get("type") == "reaction"]

    def typing(self):
        return [c for c in self.graph_calls() if c["json"].get("status") == "read"]


class _FakeClient:
    """Stands in for httpx.AsyncClient inside the bridge."""

    def __init__(self, rec: _Recorder, timeout):
        self.rec = rec
        self.timeout = timeout

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, url, headers=None, json=None, params=None, **kw):
        self.rec.calls.append({
            "url": url, "headers": headers, "json": json,
            "params": params, "timeout": self.timeout,
        })
        if url.endswith("/api/llm/complete"):
            if self.rec.during_llm is not None:
                await self.rec.during_llm()
            if self.rec.llm_exc is not None:
                raise self.rec.llm_exc
            return self.rec.llm
        return self.rec.graph


@pytest.fixture
def http(monkeypatch):
    rec = _Recorder()
    monkeypatch.setattr(wb.httpx, "AsyncClient", rec.client)
    monkeypatch.setenv("WHATSAPP_ACCESS_TOKEN", "EAAG-test")
    monkeypatch.setenv("WHATSAPP_PHONE_NUMBER_ID", GRAPH_PID)
    monkeypatch.delenv("WHATSAPP_REACTION_TIMEOUT", raising=False)
    monkeypatch.setattr(wb, "_WAID_LAST_SEND_AT", {})
    return rec


# ── _pick_reaction: only ever an allowlisted emoji ─────────────────────────


@pytest.mark.parametrize("reply,expected", [
    ("👍", "👍"),
    ("  🎉\n", "🎉"),
    ("Reaction: 😂", "😂"),
    ("I'd go with 🙏 here.", "🙏"),
    ('"🔥"', "🔥"),
    ("👍 or ✅", "👍"),            # earliest wins
    ("✅ 👍", "✅"),
    ("😂 (NONE would also do)", "😂"),
    ("NONE", None),
    ("none", None),
    ("None.", None),
    ("NONE — maybe 👍", None),     # NONE before any emoji
    ("\u2764", HEART),             # bare heart, no VS16
    (HEART, HEART),
    ("so sweet \u2764", HEART),
    ("👍🏽", "👍"),                 # skin tone → base emoji
    ("🤖", None),                  # not on the allowlist
    ("💩", None),
    ("ok", None),
    ("", None),
    ("   ", None),
])
def test_pick_reaction(reply, expected):
    assert wb._pick_reaction(reply) == expected


def test_pick_reaction_tolerates_none_input():
    assert wb._pick_reaction(None) is None  # type: ignore[arg-type]


def test_every_allowlisted_emoji_maps_to_itself_and_is_offered_in_the_prompt():
    assert len(set(wb._REACTION_EMOJIS)) == len(wb._REACTION_EMOJIS) == 18
    for emoji in wb._REACTION_EMOJIS:
        assert wb._pick_reaction(emoji) == emoji
        assert emoji in wb._REACTION_SYSTEM_PROMPT
    assert HEART in wb._REACTION_EMOJIS


# ── Flag + timeout ─────────────────────────────────────────────────────────


def test_reactions_are_on_by_default(monkeypatch):
    monkeypatch.delenv("WHATSAPP_REACTIONS", raising=False)
    assert wb._reactions_enabled() is True


@pytest.mark.parametrize("value", ["on", "1", "true", "yes", "", "whatever"])
def test_reactions_stay_on_for_anything_but_an_off_value(monkeypatch, value):
    monkeypatch.setenv("WHATSAPP_REACTIONS", value)
    assert wb._reactions_enabled() is True


@pytest.mark.parametrize("value", ["off", "OFF", "0", "false", "False", "no", " No "])
def test_reactions_off_values_disable(monkeypatch, value):
    monkeypatch.setenv("WHATSAPP_REACTIONS", value)
    assert wb._reactions_enabled() is False


@pytest.mark.parametrize("value,expected", [
    (None, 8.0), ("", 8.0), ("abc", 8.0), ("nan", 8.0),
    ("2.5", 2.5), ("0", 1.0), ("-5", 1.0), ("100", 30.0), ("inf", 30.0),
])
def test_reaction_timeout_is_parsed_defensively_and_clamped(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv("WHATSAPP_REACTION_TIMEOUT", raising=False)
    else:
        monkeypatch.setenv("WHATSAPP_REACTION_TIMEOUT", value)
    assert wb._reaction_timeout() == expected


# ── _maybe_react ───────────────────────────────────────────────────────────


async def test_happy_path_asks_the_agents_llm_then_reacts(http):
    await wb._maybe_react(WAID, WAMID, "I got the job!!", "localhost", 24001, "agent-tok")

    [llm] = http.llm_calls()
    assert llm["url"] == "http://localhost:24001/api/llm/complete"
    assert llm["params"] == {"token": "agent-tok"}
    assert llm["timeout"] == 8.0
    body = llm["json"]
    assert body["max_tokens"] == 256
    assert "temperature" not in body
    assert [m["role"] for m in body["messages"]] == ["system", "user"]
    assert body["messages"][0]["content"] == wb._REACTION_SYSTEM_PROMPT
    assert "<<<\nI got the job!!\n>>>" in body["messages"][1]["content"]

    [reaction] = http.reactions()
    assert reaction["headers"]["Authorization"] == "Bearer EAAG-test"
    assert reaction["json"] == {
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": WAID,
        "type": "reaction",
        "reaction": {"message_id": WAMID, "emoji": "🎉"},
    }
    # Nothing was sent since the classifier started → typing dots restored,
    # after the reaction.
    [typing] = http.typing()
    assert typing["json"]["message_id"] == WAMID
    assert http.calls.index(typing) > http.calls.index(reaction)


async def test_long_text_is_truncated_for_the_classifier(http):
    await wb._maybe_react(WAID, WAMID, "x" * 5000, "localhost", 24001, "agent-tok")
    user_msg = http.llm_calls()[0]["json"]["messages"][1]["content"]
    assert len(user_msg) < 1600
    assert "x" * 1500 + "…" in user_msg


async def test_timeout_env_reaches_the_classifier_call(http, monkeypatch):
    monkeypatch.setenv("WHATSAPP_REACTION_TIMEOUT", "3")
    await wb._maybe_react(WAID, WAMID, "thanks!", "localhost", 24001, "agent-tok")
    assert http.llm_calls()[0]["timeout"] == 3.0


async def test_none_means_no_graph_call(http):
    http.llm = _Resp(200, {"ok": True, "content": "NONE"})
    await wb._maybe_react(WAID, WAMID, "what's the weather in Zagreb?", "localhost", 24001, "t")
    assert len(http.llm_calls()) == 1
    assert http.graph_calls() == []


@pytest.mark.parametrize("resp", [
    _Resp(500, {"ok": False, "error": "provider exploded"}),
    _Resp(503, {"ok": False, "error": "no LLM provider"}),
    _Resp(401, None, text="Unauthorized"),
    _Resp(200, {"ok": False, "error": "nope"}),
    _Resp(200, {"ok": True, "content": ""}),
    _Resp(200, {"ok": True, "content": "🤖"}),
    _Resp(200, None, text="<html>not json</html>"),
    _Resp(200, ["not", "a", "dict"]),
])
async def test_classifier_failure_means_no_reaction(http, resp):
    http.llm = resp
    await wb._maybe_react(WAID, WAMID, "thank you so much!", "localhost", 24001, "t")
    assert http.graph_calls() == []


@pytest.mark.parametrize("exc", [
    httpx.ConnectError("refused"),
    httpx.ReadTimeout("slow model"),
    RuntimeError("boom"),
])
async def test_classifier_exception_never_raises(http, exc):
    http.llm_exc = exc
    await wb._maybe_react(WAID, WAMID, "thank you so much!", "localhost", 24001, "t")
    assert http.graph_calls() == []


# A thinking model (Ollama qwen3 / r1 / gpt-oss, DeepSeek reasoner) that spends
# the whole budget reasoning comes back with empty content; the provider then
# surfaces the tail of the cut-off reasoning AS content. It names the emojis it
# is ruling out, so the earliest one is effectively random.
_REASONING_TAIL = (
    "The allowed reactions are 👍 ❤️ 😂 🙏 and so on; for sad news the "
    "guidance says 😢 or ❤️, so I should"
)


@pytest.mark.parametrize("finish_reason", ["length", "max_tokens", "MAX_TOKENS"])
@pytest.mark.parametrize("content", [_REASONING_TAIL, "😢"])
async def test_a_truncated_classifier_reply_means_no_reaction(http, finish_reason, content):
    http.llm = _Resp(200, {"ok": True, "content": content, "finish_reason": finish_reason})
    await wb._maybe_react(WAID, WAMID, "my dog died this morning", "localhost", 24001, "t")
    assert len(http.llm_calls()) == 1
    assert http.graph_calls() == []


@pytest.mark.parametrize("content", [
    _REASONING_TAIL,
    # Finished reasoning recovered as content: the conclusion is NONE, but a
    # ruled-out emoji comes first.
    "Thanks would be 🙏 but this is a routine request, so the answer is NONE",
])
async def test_a_long_classifier_reply_is_never_mined_for_an_emoji(http, content):
    assert len(content) > wb._REACTION_MAX_REPLY
    http.llm = _Resp(200, {"ok": True, "content": content, "finish_reason": "stop"})
    await wb._maybe_react(WAID, WAMID, "my dog died this morning", "localhost", 24001, "t")
    assert http.graph_calls() == []


@pytest.mark.parametrize("content", [
    "I'd go with 🙏 here.",
    "  🙏" + " " * 60,                         # padding is stripped first
    "Reaction: 🙏".ljust(wb._REACTION_MAX_REPLY, "."),  # exactly at the bound
])
async def test_a_terse_finished_reply_still_reacts(http, content):
    http.llm = _Resp(200, {"ok": True, "content": content, "finish_reason": "stop"})
    await wb._maybe_react(WAID, WAMID, "thank you so much!", "localhost", 24001, "t")
    assert [r["json"]["reaction"]["emoji"] for r in http.reactions()] == ["🙏"]


async def test_rejected_reaction_does_not_refire_typing(http):
    http.graph = _Resp(400, {"error": {"message": "bad emoji"}})
    await wb._maybe_react(WAID, WAMID, "haha", "localhost", 24001, "t")
    assert len(http.reactions()) == 1
    assert http.typing() == []


async def test_no_typing_refire_once_the_agent_already_replied(http):
    # The agent's answer lands while the classifier is still thinking.
    async def _agent_replies():
        await wb._send_whatsapp_text(WAID, "Congratulations — here's your plan.")
    http.during_llm = _agent_replies

    await wb._maybe_react(WAID, WAMID, "I got the job!!", "localhost", 24001, "t")

    assert len(http.reactions()) == 1
    assert http.typing() == []  # a false "typing…" would hang for ~25 s


async def test_an_older_send_does_not_block_the_typing_refire(http):
    # e.g. the "🎙 Transcription: …" status text, sent before the classifier.
    wb._WAID_LAST_SEND_AT[WAID] = time.time() - 60
    await wb._maybe_react(WAID, WAMID, "fingers crossed", "localhost", 24001, "t")
    assert len(http.typing()) == 1


async def test_a_newer_send_stamp_blocks_the_typing_refire(http):
    async def _stamp():
        wb._WAID_LAST_SEND_AT[WAID] = time.time() + 1
    http.during_llm = _stamp
    await wb._maybe_react(WAID, WAMID, "fingers crossed", "localhost", 24001, "t")
    assert len(http.reactions()) == 1
    assert http.typing() == []


async def test_no_reaction_when_the_forward_failed(http):
    gate = asyncio.get_running_loop().create_future()
    gate.set_result(False)   # "Agent not ready" / "Send failed"
    await wb._maybe_react(WAID, WAMID, "We closed the round!", "localhost", 24001, "t",
                          forwarded=gate)
    assert len(http.llm_calls()) == 1   # classification still ran in parallel
    assert http.graph_calls() == []      # …but nothing lands on the message


async def test_the_reaction_waits_for_the_forward_to_succeed(http):
    gate = asyncio.get_running_loop().create_future()
    task = asyncio.create_task(wb._maybe_react(
        WAID, WAMID, "We closed the round!", "localhost", 24001, "t", forwarded=gate,
    ))
    for _ in range(5):
        await asyncio.sleep(0)
    assert len(http.llm_calls()) == 1   # classified already
    assert http.reactions() == []        # but held until the agent has it
    gate.set_result(True)
    await task
    assert [r["json"]["reaction"]["emoji"] for r in http.reactions()] == ["🎉"]


async def test_none_needs_no_forward_outcome(http):
    http.llm = _Resp(200, {"ok": True, "content": "NONE"})
    gate = asyncio.get_running_loop().create_future()   # never settled
    await asyncio.wait_for(
        wb._maybe_react(WAID, WAMID, "what's the weather?", "localhost", 24001, "t",
                        forwarded=gate),
        timeout=2,
    )
    assert http.graph_calls() == []


async def test_registry_token_is_used_when_no_explicit_auth(http, monkeypatch):
    fake_server = types.ModuleType("captain_claw.flight_deck.server")
    seen = []

    def _resolve(port):
        seen.append(port)
        return "reg-tok"

    fake_server._resolve_agent_auth = _resolve
    monkeypatch.setitem(sys.modules, "captain_claw.flight_deck.server", fake_server)

    await wb._maybe_react(WAID, WAMID, "thanks!", "localhost", 24001, "")
    assert seen == [24001]
    assert http.llm_calls()[0]["params"] == {"token": "reg-tok"}


async def test_no_token_anywhere_sends_no_token_param(http, monkeypatch):
    fake_server = types.ModuleType("captain_claw.flight_deck.server")

    def _resolve(port):
        raise RuntimeError("docker down")

    fake_server._resolve_agent_auth = _resolve
    monkeypatch.setitem(sys.modules, "captain_claw.flight_deck.server", fake_server)

    await wb._maybe_react(WAID, WAMID, "thanks!", "localhost", 24001, "  ")
    assert "token" not in (http.llm_calls()[0]["params"] or {})


async def test_reaction_send_needs_graph_config_and_allowlisted_emoji(http, monkeypatch):
    assert await wb._send_whatsapp_reaction(WAID, WAMID, "🤖") is False
    assert await wb._send_whatsapp_reaction(WAID, "", "👍") is False
    assert http.calls == []
    assert await wb._send_whatsapp_reaction(WAID, WAMID, "👍") is True
    monkeypatch.delenv("WHATSAPP_ACCESS_TOKEN")
    assert await wb._send_whatsapp_reaction(WAID, WAMID, "👍") is False
    assert len(http.calls) == 1


async def test_reaction_send_survives_a_network_error(http, monkeypatch):
    class _Boom(_FakeClient):
        async def post(self, *a, **kw):
            raise httpx.ConnectError("down")

    monkeypatch.setattr(wb.httpx, "AsyncClient", lambda *a, **kw: _Boom(http, None))
    assert await wb._send_whatsapp_reaction(WAID, WAMID, "👍") is False


# ── Send stamps: only real posts count ─────────────────────────────────────


async def test_text_send_stamps_only_when_it_posts(http, monkeypatch):
    monkeypatch.delenv("WHATSAPP_ACCESS_TOKEN")
    await wb._send_whatsapp_text(WAID, "hello")
    assert WAID not in wb._WAID_LAST_SEND_AT

    monkeypatch.setenv("WHATSAPP_ACCESS_TOKEN", "EAAG-test")
    before = time.time()
    await wb._send_whatsapp_text(WAID, "hello")
    assert wb._WAID_LAST_SEND_AT[WAID] >= before


async def test_audio_send_stamps(http, monkeypatch):
    async def _synth(text):
        return b"mp3"

    async def _upload(blob):
        return "media-1"

    monkeypatch.setattr(wb, "_synth_audio_mp3", _synth)
    monkeypatch.setattr(wb, "_upload_whatsapp_audio", _upload)
    before = time.time()
    await wb._send_whatsapp_audio(WAID, "hello")
    assert wb._WAID_LAST_SEND_AT[WAID] >= before
    assert http.graph_calls()[-1]["json"]["type"] == "audio"


async def test_reaction_itself_does_not_stamp(http):
    await wb._send_whatsapp_reaction(WAID, WAMID, "👍")
    assert WAID not in wb._WAID_LAST_SEND_AT


# ── _spawn_bg ──────────────────────────────────────────────────────────────


async def test_spawn_bg_holds_a_reference_until_done():
    gate = asyncio.Event()

    async def _work():
        await gate.wait()

    task = wb._spawn_bg(_work())
    assert task in wb._BG_TASKS
    gate.set()
    await task
    await asyncio.sleep(0)  # done-callbacks run on the next loop tick
    assert task not in wb._BG_TASKS


# ── _handle_message wiring ─────────────────────────────────────────────────


class _FakeAgentWS:
    def __init__(self):
        self.sent: list[dict] = []

    async def send(self, raw):
        self.sent.append(json.loads(raw))


@pytest.fixture
def inbound(monkeypatch):
    """_handle_message with every network/binding seam faked out."""
    monkeypatch.delenv("WHATSAPP_REACTIONS", raising=False)
    monkeypatch.delenv("WHATSAPP_DEFAULT_CHANNEL", raising=False)
    monkeypatch.setattr(wb, "_WAID_CHANNEL", {})
    monkeypatch.setattr(wb, "_CHANNEL_WAIDS", {})
    monkeypatch.setattr(wb, "_WAID_LAST_MESSAGE_ID", {})
    monkeypatch.setattr(wb, "_MUTED_UNTIL", {})
    monkeypatch.setattr(wb, "_PENDING_IMAGE", {})

    st = types.SimpleNamespace(
        marked=[], texts=[], broadcasts=[], reacted=[], gates=[], tasks=[],
        agent_lookups=0,
        ch=types.SimpleNamespace(
            channel_id=f"whatsapp:{WAID}",
            agent_ws=_FakeAgentWS(),
            send_lock=asyncio.Lock(),
            context_sent=True,
        ),
    )

    async def _mark(message_id):
        st.marked.append(message_id)

    async def _send_text(waid, text, *, mirror=False):
        st.texts.append(text)

    async def _get_channel(channel):
        return st.ch

    def _default_agent():
        st.agent_lookups += 1
        return ("localhost", 24001, "agent-tok")

    async def _bind(ch, host, port, auth):
        return None

    async def _broadcast(ch, event):
        st.broadcasts.append(event)

    async def _react(*args, forwarded=None):
        st.reacted.append(args)
        st.gates.append(forwarded)

    real_spawn = wb._spawn_bg

    def _spawn(coro):
        task = real_spawn(coro)
        st.tasks.append(task)
        return task

    monkeypatch.setattr(wb, "_mark_read_and_typing", _mark)
    monkeypatch.setattr(wb, "_send_whatsapp_text", _send_text)
    monkeypatch.setattr(wb, "_get_or_create_channel", _get_channel)
    monkeypatch.setattr(wb, "_ensure_whatsapp_forwarding", lambda channel_id: None)
    monkeypatch.setattr(wb, "_default_agent", _default_agent)
    monkeypatch.setattr(wb, "_ensure_agent_binding", _bind)
    monkeypatch.setattr(wb, "_broadcast", _broadcast)
    monkeypatch.setattr(wb, "_maybe_react", _react)
    monkeypatch.setattr(wb, "_spawn_bg", _spawn)
    return st


def _text_msg(body: str, wamid: str = WAMID) -> dict:
    return {"from": WAID, "id": wamid, "type": "text", "text": {"body": body}}


async def _settle(st):
    await asyncio.sleep(0)
    if st.tasks:
        await asyncio.gather(*st.tasks)


async def test_inbound_user_reaction_is_dropped_before_read_and_typing(inbound):
    msg = {
        "from": WAID, "id": "wamid.reaction", "type": "reaction",
        "reaction": {"message_id": "wamid.ours", "emoji": "👍"},
    }
    await wb._handle_message(WAID, msg)
    await _settle(inbound)
    assert inbound.marked == []                      # no 25 s phantom "typing…"
    assert inbound.agent_lookups == 0                # no "Bridge offline" reply
    assert inbound.texts == []
    assert inbound.tasks == []
    assert "wamid.reaction" not in wb._WAID_LAST_MESSAGE_ID.values()


async def test_plain_text_spawns_the_reaction_and_still_forwards(inbound):
    await wb._handle_message(WAID, _text_msg("We won the pitch!"))
    await _settle(inbound)
    assert inbound.reacted == [
        (WAID, WAMID, "We won the pitch!", "localhost", 24001, "agent-tok")
    ]
    assert [m["content"] for m in inbound.ch.agent_ws.sent] == ["We won the pitch!"]
    assert inbound.marked == [WAMID]


class _FailingAgentWS:
    async def send(self, raw):
        raise ConnectionError("socket closed")


async def test_the_gate_opens_once_the_message_reached_the_agent(inbound):
    await wb._handle_message(WAID, _text_msg("We won the pitch!"))
    await _settle(inbound)
    [gate] = inbound.gates
    assert gate.done() and gate.result() is True


async def test_send_failed_closes_the_gate(inbound):
    inbound.ch.agent_ws = _FailingAgentWS()
    await wb._handle_message(WAID, _text_msg("We won the pitch!"))
    await _settle(inbound)
    [gate] = inbound.gates
    assert gate.result() is False
    assert any(t.startswith("Send failed") for t in inbound.texts)


async def test_agent_not_ready_closes_the_gate(inbound, monkeypatch):
    inbound.ch.agent_ws = None   # WS pump in reconnect backoff
    real_sleep = asyncio.sleep
    monkeypatch.setattr(asyncio, "sleep", lambda delay, *a, **kw: real_sleep(0))
    await wb._handle_message(WAID, _text_msg("We won the pitch!"))
    await _settle(inbound)
    [gate] = inbound.gates
    assert gate.result() is False
    assert "Agent not ready, try again." in inbound.texts


async def test_an_unexpected_error_still_closes_the_gate(inbound, monkeypatch):
    async def _bus_down(ch, event):
        raise RuntimeError("bus down")

    monkeypatch.setattr(wb, "_broadcast", _bus_down)
    with pytest.raises(RuntimeError):
        await wb._handle_message(WAID, _text_msg("We won the pitch!"))
    await _settle(inbound)
    [gate] = inbound.gates
    assert gate.result() is False   # the reaction task is never left waiting


@pytest.mark.parametrize("agent_ws,reacted", [
    (_FakeAgentWS, ["🎉"]),
    (_FailingAgentWS, []),
])
async def test_end_to_end_reaction_only_after_a_good_forward(inbound, http, monkeypatch,
                                                             agent_ws, reacted):
    monkeypatch.setattr(wb, "_maybe_react", _REAL_MAYBE_REACT)
    inbound.ch.agent_ws = agent_ws()
    await wb._handle_message(WAID, _text_msg("We closed the round!"))
    await _settle(inbound)
    assert len(http.llm_calls()) == 1
    assert [r["json"]["reaction"]["emoji"] for r in http.reactions()] == reacted


async def test_the_forward_never_waits_for_the_classifier(inbound, monkeypatch):
    gate = asyncio.Event()

    async def _slow_react(*args, **kw):
        await gate.wait()  # a model that takes forever

    monkeypatch.setattr(wb, "_maybe_react", _slow_react)
    await asyncio.wait_for(wb._handle_message(WAID, _text_msg("thanks!")), timeout=2)
    assert [m["content"] for m in inbound.ch.agent_ws.sent] == ["thanks!"]
    assert len(inbound.tasks) == 1 and not inbound.tasks[0].done()
    gate.set()
    await _settle(inbound)


@pytest.mark.parametrize("body", ["/c", "/unmute", "/mute"])
async def test_slash_commands_get_no_reaction(inbound, body):
    await wb._handle_message(WAID, _text_msg(body))
    await _settle(inbound)
    assert inbound.tasks == []
    assert inbound.reacted == []
    assert inbound.ch.agent_ws.sent == []


async def test_a_matched_text_flow_gets_no_reaction(inbound, monkeypatch):
    from captain_claw.flight_deck import flow_router

    async def _no_command(payload):
        return False

    async def _flow_ran(payload):
        return True

    monkeypatch.setattr(flow_router, "engine_ready", lambda: True)
    monkeypatch.setattr(flow_router, "maybe_handle_flow_command", _no_command)
    monkeypatch.setattr(flow_router, "deliver_pending_input", lambda **kw: False)
    monkeypatch.setattr(flow_router, "classify_payload", lambda **kw: dict(kw))
    monkeypatch.setattr(flow_router, "try_match_and_run", _flow_ran)

    await wb._handle_message(WAID, _text_msg("daily report please"))
    await _settle(inbound)
    assert inbound.tasks == []
    assert inbound.ch.agent_ws.sent == []


async def test_reactions_disabled_spawns_nothing(inbound, monkeypatch):
    monkeypatch.setenv("WHATSAPP_REACTIONS", "off")
    await wb._handle_message(WAID, _text_msg("thank you!"))
    await _settle(inbound)
    assert inbound.tasks == []
    assert [m["content"] for m in inbound.ch.agent_ws.sent] == ["thank you!"]


async def test_location_fyi_gets_no_reaction(inbound):
    msg = {
        "from": WAID, "id": WAMID, "type": "location",
        "location": {"latitude": 45.81, "longitude": 15.98, "name": "Zagreb"},
    }
    await wb._handle_message(WAID, msg)
    await _settle(inbound)
    assert inbound.tasks == []
    assert len(inbound.ch.agent_ws.sent) == 1


async def test_message_without_an_id_gets_no_reaction(inbound):
    await wb._handle_message(WAID, {"from": WAID, "type": "text", "text": {"body": "thanks!"}})
    await _settle(inbound)
    assert inbound.tasks == []
    assert len(inbound.ch.agent_ws.sent) == 1


async def test_voice_note_reacts_to_the_transcript(inbound, monkeypatch):
    async def _download(media_id):
        return b"ogg"

    async def _upload_audio(blob, host, port, auth, filename):
        return "/agent/files/whatsapp.ogg"

    async def _transcribe(blob, mime):
        return ("We closed the round!", "")

    monkeypatch.setattr(wb, "_download_media", _download)
    monkeypatch.setattr(wb, "_save_fd_local_audio", lambda blob, ext: "")
    monkeypatch.setattr(wb, "_upload_audio_to_agent", _upload_audio)
    monkeypatch.setattr(wb, "_transcribe_soniox", _transcribe)

    msg = {
        "from": WAID, "id": WAMID, "type": "audio",
        "audio": {"id": "media-1", "mime_type": "audio/ogg"},
    }
    await wb._handle_message(WAID, msg)
    await _settle(inbound)
    assert inbound.reacted == [
        (WAID, WAMID, "We closed the round!", "localhost", 24001, "agent-tok")
    ]


# ── Agent side: /api/llm/complete runs on a scoped provider copy ───────────


class _StallingProvider:
    """Mimics a provider whose payload builder consumes the one-shot
    ``_tool_choice_override`` (as ``_request_kwargs`` does)."""

    def __init__(self):
        self._tool_choice_override = None
        self.calls: list[dict] = []   # shared by shallow copies

    async def complete(self, messages, tools=None, temperature=None, max_tokens=None):
        from captain_claw.llm import LLMResponse
        self.calls.append({"messages": messages, "temperature": temperature,
                           "max_tokens": max_tokens})
        self._tool_choice_override = None
        return LLMResponse(content="👍", model="m", usage={"total_tokens": 3},
                           finish_reason="stop")


class _JsonRequest:
    def __init__(self, body):
        self._body = body

    async def json(self):
        return self._body


async def test_llm_complete_cannot_consume_the_agents_tool_choice_override():
    from captain_claw.web_server import WebServer

    shared = _StallingProvider()
    shared._tool_choice_override = "required"   # agent is mid stall-retry
    server = WebServer.__new__(WebServer)        # skip __init__'s heavy wiring
    server.agent = types.SimpleNamespace(provider=shared)

    resp = await server._llm_complete(_JsonRequest({
        "messages": [{"role": "user", "content": "thanks!"}], "max_tokens": 256,
    }))

    assert resp.status == 200
    data = json.loads(resp.text)
    assert data["ok"] is True and data["content"] == "👍"
    assert shared.calls and shared.calls[0]["max_tokens"] == 256
    assert shared.calls[0]["temperature"] is None
    assert shared._tool_choice_override == "required"


async def test_llm_complete_without_an_agent_is_503():
    from captain_claw.web_server import WebServer

    server = WebServer.__new__(WebServer)
    server.agent = None
    resp = await server._llm_complete(_JsonRequest({"messages": []}))
    assert resp.status == 503


async def test_llm_complete_carries_ollamas_think_false_back_to_the_agent(monkeypatch):
    """A model the think heuristic matches but that rejects thinking (e.g.
    qwen3-coder): the copy learns ``think = False`` on the first 400. That
    lesson must reach the shared provider, or every later proxy call repeats
    the doomed think request."""
    from captain_claw.llm import OllamaProvider
    from captain_claw.vastai import wake
    from captain_claw.web_server import WebServer

    async def _no_wake(base_url):
        return None

    monkeypatch.setattr(wake, "maybe_wake_instance", _no_wake)
    monkeypatch.delenv("CLAW_OLLAMA_THINK", raising=False)

    sent: list[bool] = []   # did each HTTP request carry ``think``?

    def _ollama(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        sent.append("think" in body)
        if body.get("think"):
            return httpx.Response(400, json={"error": '"qwen3-coder" does not support thinking'})
        return httpx.Response(200, json={
            "model": body["model"],
            "message": {"role": "assistant", "content": "👍"},
            "done": True, "done_reason": "stop",
            "prompt_eval_count": 5, "eval_count": 1,
        })

    shared = OllamaProvider(model="qwen3-coder:30b", base_url="http://ollama.test:11434",
                            think=True)
    shared.client = httpx.AsyncClient(transport=httpx.MockTransport(_ollama))
    server = WebServer.__new__(WebServer)
    server.agent = types.SimpleNamespace(provider=shared)
    req = {"messages": [{"role": "user", "content": "thanks!"}], "max_tokens": 256}

    for _ in range(3):
        resp = await server._llm_complete(_JsonRequest(req))
        assert resp.status == 200 and json.loads(resp.text)["content"] == "👍"

    assert sent == [True, False, False, False]   # one rejected request, ever
    assert shared.think is False
    await shared.client.aclose()


async def test_llm_complete_leaves_think_alone_after_a_model_switch():
    """The lesson is about the copy's model: if the agent switched models
    mid-call, the base keeps its own think setting."""
    from captain_claw.llm import LLMResponse
    from captain_claw.web_server import WebServer

    class _SwitchingProvider:
        def __init__(self):
            self.model, self.think = "qwen3-coder:30b", True

        async def complete(self, messages, tools=None, temperature=None, max_tokens=None):
            self.think = False                       # this model rejected think…
            server.agent.provider.model = "qwen3:8b"  # …while the agent switched
            server.agent.provider.think = True
            return LLMResponse(content="👍", model=self.model, finish_reason="stop")

    shared = _SwitchingProvider()
    server = WebServer.__new__(WebServer)
    server.agent = types.SimpleNamespace(provider=shared)
    resp = await server._llm_complete(_JsonRequest({"messages": []}))
    assert resp.status == 200
    assert shared.think is True
