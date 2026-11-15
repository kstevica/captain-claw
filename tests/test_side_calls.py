"""Side calls on a self-hosted model wait until the agent's run is over.

A self-hosted server keeps the cache of the prompt it ran last. On a home deck
(a llama.cpp server), Flight Deck's Work board triage borrowed the agent's
model in the middle of a run: a 1,042-token prompt cut in, and the run's next
call read its whole 93k-token context again — 34 s, where its cached calls
took 2–6 s. The task name, sent just before each turn's first call, did the
same to that call.

Borrowed calls that can wait (``defer``) are held while the agent is at work,
and the task name is made once the turn is over, its usage rows named then. A
cloud API caches per prompt: there nothing is held.
"""

from __future__ import annotations

import asyncio
import json
import types

import httpx
import pytest

from captain_claw import side_calls
from captain_claw.llm import LLMResponse

LAN = "http://192.168.1.20:8080/v1"


def _provider(provider: str, base_url: str = "") -> types.SimpleNamespace:
    return types.SimpleNamespace(provider=provider, model="m", base_url=base_url)


# ── which servers lose their cache to a side call ────────────────────


@pytest.mark.parametrize("provider,base_url", [
    ("openai", LAN),
    ("openai", "http://localhost:8000/v1"),
    ("openai", "http://my-box.example.net:8080/v1"),   # a home server behind a DNS name
    ("ollama", "http://127.0.0.1:11434"),
])
def test_self_hosted_servers_evict(provider, base_url):
    assert side_calls.evicts_cache(_provider(provider, base_url))


@pytest.mark.parametrize("provider,base_url", [
    ("anthropic", ""),
    ("openai", ""),
    ("openai", "https://api.openai.com/v1"),
    ("openai", "https://openrouter.ai/api/v1"),
    ("openai", "https://api.deepseek.com"),
    ("ollama", "https://ollama.com"),
    ("gemini", ""),
])
def test_cloud_apis_do_not(provider, base_url):
    assert not side_calls.evicts_cache(_provider(provider, base_url))


def test_no_provider_does_not():
    assert not side_calls.evicts_cache(None)


def test_max_wait_is_clamped():
    assert side_calls.max_wait(None) == side_calls.DEFAULT_MAX_WAIT_S
    assert side_calls.max_wait("120") == 120.0
    assert side_calls.max_wait(10_000) == side_calls.MAX_WAIT_CAP_S
    assert side_calls.max_wait(-5) == 0.0


@pytest.mark.asyncio
async def test_wait_until_quiet_waits_for_the_run_and_caps(monkeypatch):
    monkeypatch.setattr(side_calls, "_POLL_S", 0.01)
    checks = iter([True, True, True, False])
    await side_calls.wait_until_quiet(lambda: next(checks), 5.0)
    assert next(checks, "done") == "done"
    waited = await side_calls.wait_until_quiet(lambda: True, 0.05)
    assert 0.05 <= waited < 1.0


# ── the agent's /api/llm/complete ────────────────────────────────────


class _Req:
    def __init__(self, body):
        self._body = body

    async def json(self):
        return self._body


def _server(monkeypatch, provider_name: str, base_url: str, busy_checks: int):
    """A WebServer at work for *busy_checks* looks, whose model records whether
    the run was still going when the call reached it."""
    from captain_claw import anonymize_gate_client
    from captain_claw.web_server import WebServer

    state = {"checks": 0, "seen_at_work": []}

    def at_work(self) -> bool:
        state["checks"] += 1
        return state["checks"] <= busy_checks

    class Provider:
        provider, model = provider_name, "m"

        def __init__(self):
            self.base_url = base_url

        async def complete(self, messages, tools=None, temperature=None, max_tokens=None):
            state["seen_at_work"].append(state["checks"] <= busy_checks)
            return LLMResponse(content="ok", model="m")

    async def no_gate(agent):
        return None

    server = WebServer.__new__(WebServer)
    server.agent = types.SimpleNamespace(provider=Provider(), session=None)
    monkeypatch.setattr(WebServer, "_scoped_provider", lambda self: self.agent.provider, raising=False)
    monkeypatch.setattr(WebServer, "_at_work", at_work)
    monkeypatch.setattr(anonymize_gate_client, "decision", no_gate)
    monkeypatch.setattr(side_calls, "_POLL_S", 0.01)
    return server, state


_MSGS = [{"role": "user", "content": "Triage this task."}]


@pytest.mark.asyncio
async def test_a_deferred_call_waits_for_the_run_on_a_self_hosted_model(monkeypatch):
    server, state = _server(monkeypatch, "openai", LAN, busy_checks=5)
    resp = await server._llm_complete(_Req({"messages": _MSGS, "defer": True, "defer_max_s": 5}))
    assert json.loads(resp.text)["ok"]
    assert state["seen_at_work"] == [False]


@pytest.mark.asyncio
async def test_a_call_that_cannot_wait_goes_straight_away(monkeypatch):
    server, state = _server(monkeypatch, "openai", LAN, busy_checks=1000)
    await server._llm_complete(_Req({"messages": _MSGS}))
    assert state["seen_at_work"] == [True]


@pytest.mark.asyncio
async def test_a_cloud_model_is_never_held(monkeypatch):
    server, state = _server(monkeypatch, "anthropic", "", busy_checks=1000)
    await server._llm_complete(_Req({"messages": _MSGS, "defer": True}))
    assert state["seen_at_work"] == [True]


@pytest.mark.asyncio
async def test_a_held_call_goes_after_its_cap(monkeypatch):
    server, state = _server(monkeypatch, "openai", LAN, busy_checks=1_000_000)
    resp = await server._llm_complete(_Req({"messages": _MSGS, "defer": True, "defer_max_s": 0.05}))
    assert json.loads(resp.text)["ok"] and state["seen_at_work"] == [True]


# ── the deck's background callers ask for it ─────────────────────────


@pytest.mark.asyncio
async def test_remote_provider_sends_defer(monkeypatch):
    from captain_claw.games import remote_provider
    from captain_claw.llm import Message

    sent: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return httpx.Response(200, json={"ok": True, "content": "{}", "model": "m"})

    real = httpx.AsyncClient
    monkeypatch.setattr(remote_provider.httpx, "AsyncClient",
                        lambda **kw: real(transport=httpx.MockTransport(handler), **kw))
    await remote_provider.RemoteLLMProvider("127.0.0.1", 1, defer_s=300).complete([Message(role="user", content="x")])
    await remote_provider.RemoteLLMProvider("127.0.0.1", 1).complete([Message(role="user", content="x")])
    assert sent[0]["defer"] is True and sent[0]["defer_max_s"] == 300
    assert "defer" not in sent[1]


@pytest.mark.asyncio
async def test_board_triage_defers_and_a_users_draft_does_not(monkeypatch):
    from captain_claw.flight_deck import agent_preset, taskboard_triage

    seen: list[float] = []

    async def fake_complete(reader, system, prompt, owner_id, max_tokens, *, defer_s=0.0):
        seen.append(defer_s)
        return agent_preset.ModelReply("{}", 1, 1, "stop", "")

    monkeypatch.setattr(agent_preset, "_agent_complete", fake_complete)
    await taskboard_triage.complete({"agent": "outreach"}, "sys", "prompt", "u1", defer=True)
    await taskboard_triage.complete({"agent": "outreach"}, "sys", "prompt", "u1")
    assert seen == [taskboard_triage._DEFER_S, 0.0]


# ── the task name, after the turn ────────────────────────────────────


class _Usage:
    def __init__(self):
        self.named: list[tuple] = []

    async def name_unnamed_usage(self, session_ids, since, task_name):
        self.named.append((list(session_ids), since, task_name))
        return 3


def _naming_agent(provider_name: str, base_url: str):
    return types.SimpleNamespace(
        provider=types.SimpleNamespace(provider=provider_name, model="m", base_url=base_url, api_key="x"),
        session_manager=_Usage(),
        _current_session_slug=lambda: "s1",
        _current_task_name="earlier",
    )


def test_naming_waits_for_the_turn_only_on_a_self_hosted_model():
    from captain_claw.web import chat_handler

    assert isinstance(chat_handler._naming_after_turn(_naming_agent("openai", LAN)), asyncio.Event)
    assert chat_handler._naming_after_turn(_naming_agent("anthropic", "")) is None


@pytest.mark.asyncio
async def test_the_task_is_named_after_the_turn_and_its_rows_labelled(monkeypatch):
    from captain_claw.web import chat_handler

    calls: list[str] = []

    async def namer(text, earlier, *args, **kwargs):
        calls.append(text)
        return "Raise the daily cap"

    async def naming_input(agent, content, recent):
        return content, [], None

    monkeypatch.setattr(chat_handler, "_generate_task_name", namer)
    monkeypatch.setattr(chat_handler, "_naming_input", naming_input)
    monkeypatch.delenv("CLAW_BEING_WORKER", raising=False)
    agent = _naming_agent("openai", LAN)
    after = asyncio.Event()

    task = chat_handler._start_task_naming(agent, "raise the daily cap to 80", [], after_turn=after)
    await asyncio.sleep(0.05)
    assert calls == [] and not task.done()          # the turn is still running
    assert agent._current_task_name == ""

    after.set()
    await asyncio.wait_for(task, 1.0)
    assert calls == ["raise the daily cap to 80"]
    (sessions, since, name), = agent.session_manager.named
    assert name == "Raise the daily cap" and "s1" in sessions and since
    assert agent._current_task_name == ""           # never set live: the next turn may be running


@pytest.mark.asyncio
async def test_on_a_cloud_model_the_name_is_made_alongside_the_turn(monkeypatch):
    from captain_claw.web import chat_handler

    async def namer(text, earlier, *args, **kwargs):
        return "Raise the daily cap"

    async def naming_input(agent, content, recent):
        return content, [], None

    monkeypatch.setattr(chat_handler, "_generate_task_name", namer)
    monkeypatch.setattr(chat_handler, "_naming_input", naming_input)
    agent = _naming_agent("anthropic", "")

    await asyncio.wait_for(chat_handler._start_task_naming(agent, "raise the cap", []), 1.0)
    assert agent._current_task_name == "Raise the daily cap"
    assert agent.session_manager.named == []


@pytest.mark.asyncio
async def test_unnamed_usage_rows_get_the_name(tmp_path):
    from captain_claw.session import SessionManager

    mgr = SessionManager(tmp_path / "s.db")
    await mgr.record_llm_usage(session_id="s1", interaction="turn_1", provider="openai", model="m")
    since = __import__("datetime").datetime.now(__import__("datetime").UTC).isoformat()
    await asyncio.sleep(0.01)
    await mgr.record_llm_usage(session_id="s1", interaction="turn_1", provider="openai", model="m")
    await mgr.record_llm_usage(session_id="s1", interaction="turn_1", provider="openai", model="m",
                               task_name="Already named")
    await mgr.record_llm_usage(session_id="other", interaction="turn_1", provider="openai", model="m")

    assert await mgr.name_unnamed_usage(["s1"], since, "Raise the daily cap") == 1
    rows = await mgr.query_llm_usage(limit=10)
    names = sorted((r["session_id"], r["task_name"]) for r in rows)
    assert names == [("other", ""), ("s1", ""), ("s1", "Already named"), ("s1", "Raise the daily cap")]
