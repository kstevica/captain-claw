"""Flight Deck's borrowed background calls run with thinking off.

The deck borrows an agent's model through ``POST /api/llm/complete`` for its own
background work: the consciousness reflection, the arbiter's ranking, a goal's
plan, goal-progress estimates and the dispatch judge. Those calls sent no
reasoning level, so the agent ran them at the model's default — and a model that
thinks by default (a hybrid Qwen on llama.cpp) spent the whole budget thinking.
A reflection on a home deck thought for ~5,900 characters and its JSON stopped
159 characters in, at the 1500-token cap.

They now ask for ``quiet`` (``reasoning.quiet_call``: thinking off where the
model offers "off", nothing sent where it doesn't) and carry roomier caps for a
model that can't switch thinking off.
"""

from __future__ import annotations

import json
import types

import httpx
import pytest

from captain_claw.llm import LLMResponse, Message, reasoning


class _Req:
    def __init__(self, body):
        self._body = body

    async def json(self):
        return self._body


def _server(monkeypatch):
    """An idle WebServer whose model records the reasoning level of each call."""
    from captain_claw import anonymize_gate_client
    from captain_claw.web_server import WebServer

    seen: list = []

    class Provider:
        provider, model, base_url = "openai", "m", ""

        async def complete(self, messages, tools=None, temperature=None, max_tokens=None):
            seen.append((reasoning.requested_effort(), max_tokens))
            return LLMResponse(content="ok", model="m")

    async def no_gate(agent):
        return None

    server = WebServer.__new__(WebServer)
    server.agent = types.SimpleNamespace(provider=Provider(), session=None)
    monkeypatch.setattr(WebServer, "_scoped_provider", lambda self: self.agent.provider, raising=False)
    monkeypatch.setattr(WebServer, "_at_work", lambda self: False)
    monkeypatch.setattr(anonymize_gate_client, "decision", no_gate)
    return server, seen


_MSGS = [{"role": "user", "content": "Reflect."}]


# ── the agent's endpoint ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_quiet_borrowed_call_runs_as_housekeeping(monkeypatch):
    server, seen = _server(monkeypatch)
    resp = await server._llm_complete(_Req({"messages": _MSGS, "reasoning_effort": "quiet",
                                            "max_tokens": 10_000}))
    assert json.loads(resp.text)["ok"]
    assert seen == [(reasoning.QUIET, 10_000)]


@pytest.mark.asyncio
async def test_other_levels_are_unchanged(monkeypatch):
    server, seen = _server(monkeypatch)
    await server._llm_complete(_Req({"messages": _MSGS}))
    await server._llm_complete(_Req({"messages": _MSGS, "reasoning_effort": "off"}))
    await server._llm_complete(_Req({"messages": _MSGS, "reasoning_effort": "high"}))
    assert [lvl for lvl, _ in seen] == [None, "off", "high"]


# ── the deck's provider sends it ─────────────────────────────────────


@pytest.mark.asyncio
async def test_remote_provider_sends_the_level_only_when_set(monkeypatch):
    from captain_claw.games import remote_provider

    sent: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return httpx.Response(200, json={"ok": True, "content": "{}", "model": "m"})

    real = httpx.AsyncClient
    monkeypatch.setattr(remote_provider.httpx, "AsyncClient",
                        lambda **kw: real(transport=httpx.MockTransport(handler), **kw))
    msg = [Message(role="user", content="x")]
    await remote_provider.RemoteLLMProvider("127.0.0.1", 1, reasoning_effort=reasoning.QUIET).complete(msg)
    await remote_provider.RemoteLLMProvider("127.0.0.1", 1).complete(msg)
    assert sent[0]["reasoning_effort"] == "quiet"
    assert "reasoning_effort" not in sent[1]


# ── each caller asks for it ──────────────────────────────────────────

_AGENT = {"slug": "a", "name": "A", "host": "127.0.0.1", "port": 1, "auth": "", "tier": "balanced"}


@pytest.fixture
def borrowed(monkeypatch):
    """A fake RemoteLLMProvider recording how it was built and the cap asked for."""
    import captain_claw.games.remote_provider as rp

    calls: list[dict] = []

    class _Provider:
        reply = "{}"

        def __init__(self, *args, **kw):
            self.kw = kw

        async def complete(self, messages=None, tools=None, temperature=None, max_tokens=None):
            calls.append({"effort": self.kw.get("reasoning_effort"), "max_tokens": max_tokens})
            return LLMResponse(content=_Provider.reply, model="m")

    monkeypatch.setattr(rp, "RemoteLLMProvider", _Provider)
    _Provider.calls = calls
    return _Provider


@pytest.mark.asyncio
async def test_the_reflection_is_quiet_with_a_10k_cap(monkeypatch, borrowed):
    from captain_claw.flight_deck import consciousness, course

    async def no_text(touched):
        return ""

    monkeypatch.setattr(consciousness, "_read_recent_text", no_text)
    monkeypatch.setattr(consciousness, "get_store",
                        lambda: types.SimpleNamespace(list_intentions=lambda uid: []))
    monkeypatch.setattr(course, "live_goals", lambda uid: [])
    monkeypatch.setattr(consciousness, "_build_reflection_user_prompt", lambda *a, **k: "p")
    borrowed.reply = '{"thought": "ok"}'
    out = await consciousness._reflect("u", {"agents": [_AGENT]})
    assert out is not None
    assert borrowed.calls == [{"effort": "quiet", "max_tokens": 10_000}]


@pytest.mark.asyncio
async def test_a_goal_plan_is_quiet_with_a_10k_cap(monkeypatch, borrowed):
    from captain_claw.flight_deck import action_catalog, fd_dispatch, plans

    monkeypatch.setattr(fd_dispatch, "_strongest_agent", lambda uid: _AGENT)
    monkeypatch.setattr(action_catalog, "list_catalog", lambda **kw: [])
    borrowed.reply = "[]"
    await plans.decompose_goal("u", "Ship the report")
    assert borrowed.calls == [{"effort": "quiet", "max_tokens": 10_000}]


@pytest.mark.asyncio
async def test_goal_progress_is_quiet_with_a_2k_cap(borrowed):
    from captain_claw.flight_deck import course_progress

    out = await course_progress.think([_AGENT], "sys", "user")
    assert out is not None
    assert borrowed.calls == [{"effort": "quiet", "max_tokens": 2_000}]


@pytest.mark.asyncio
async def test_the_dispatch_judge_is_quiet_with_a_2k_cap(monkeypatch, borrowed):
    from captain_claw.flight_deck import fd_dispatch

    monkeypatch.setattr(fd_dispatch, "_strongest_agent", lambda uid: _AGENT)
    borrowed.reply = '{"success": true, "why": "done"}'
    res = await fd_dispatch._judge_outcome("u", {"kind": "nudge", "title": "t"}, "out")
    assert res["success"] is True
    assert borrowed.calls == [{"effort": "quiet", "max_tokens": 2_000}]
