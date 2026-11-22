# ruff: noqa: F811 — `data`, `deck` and `as_agent` are fixtures imported from the Custom tools tests
"""An MCP server asks the person before it goes on (MCP elicitation).

The server's form becomes a card for the person the tool call is for; their
answer goes back as accept / decline, and no answer as cancel. The question
goes out only when every call in flight on that server is for one person, who
attends the run. The stdio transport answers the server's requests (ping,
elicitation) and no longer takes them for responses to its own calls. Agents
follow a call in short requests, so it can wait for the person.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI

from captain_claw.flight_deck import (
    agent_identity,
    mcp_calls,
    mcp_elicitation,
    mcp_manager,
    mcp_routes,
    mcp_storage,
    speaker_grants,
)
from captain_claw.flight_deck import person_asks as asks
from captain_claw.flight_deck.auth import get_current_user
from captain_claw.flight_deck.mcp_elicitation import Asker
from captain_claw.flight_deck.mcp_transport import (
    INVALID_PARAMS,
    MCPRequestError,
    StdioTransport,
    _request_key,
)

from .test_external_tools import LOOPBACK, data, deck  # noqa: F401
from .test_external_tools_agent import as_agent  # noqa: F401

FAKE = str(Path(__file__).resolve().parents[1] / "fixtures" / "elicit_mcp_server.py")
SERVER = {"name": "opsfake", "transport": "stdio", "command": sys.executable, "args": ["-I", FAKE],
          "deck_wide": True}
APPROVE_FORM = {
    "type": "object",
    "properties": {
        "approve": {"type": "boolean", "title": "Approve", "description": "Create the server", "default": False},
        "region": {"type": "string", "title": "Region",
                   "oneOf": [{"const": "fsn1", "title": "Falkenstein"}, {"const": "nbg1", "title": "Nuremberg"}]},
    },
    "required": ["approve", "region"],
}


@pytest.fixture(autouse=True)
def fresh(monkeypatch, tmp_path):
    monkeypatch.setenv("CAPTAIN_CLAW_FD_MCP_PATH", str(tmp_path / "mcp_servers.json"))
    asks.reset_for_tests()
    mcp_elicitation.reset_for_tests()
    mcp_calls.reset_for_tests()
    monkeypatch.setattr(asks, "MIN_WAIT_S", 0.3)
    monkeypatch.setattr(asks, "_ring", _no_ring)
    monkeypatch.setattr(asks, "_settle", _no_ring)
    mcp_manager._manager = None  # noqa: SLF001
    yield
    asks.reset_for_tests()
    mcp_elicitation.reset_for_tests()
    mcp_calls.reset_for_tests()


async def _no_ring(*_a, **_k) -> None:
    return None


@pytest.fixture
async def manager():
    await mcp_storage.upsert_server(dict(SERVER))
    m = mcp_manager.get_manager()
    yield m
    await m.close()
    mcp_manager._manager = None  # noqa: SLF001


async def _card_for(user_id: str, tries: int = 100) -> dict:
    for _ in range(tries):
        found = asks.open_for(user_id)
        if found:
            return found[0]
        await asyncio.sleep(0.05)
    raise AssertionError("no card opened")


# ── The form → questions ──────────────────────────────────────────────


def test_a_form_becomes_questions_and_answers_go_back_as_the_servers_values():
    fields, qs = mcp_elicitation.form_questions(APPROVE_FORM)
    assert [q["kind"] for q in qs] == ["confirm", "choice"]
    assert qs[0] == {"id": "f1", "question": "Approve", "help": "Create the server", "optional": False,
                     "kind": "confirm", "default": False}
    assert qs[1]["choices"] == ["Falkenstein", "Nuremberg"]
    content = mcp_elicitation.form_content(fields, asks.check_answers(qs, {"f1": True, "f2": "Nuremberg"}))
    assert content == {"approve": True, "region": "nbg1"}


def test_every_field_type_the_spec_allows():
    schema = {"type": "object", "required": ["n", "mail"], "properties": {
        "mail": {"type": "string", "format": "email", "maxLength": 40},
        "n": {"type": "integer", "minimum": 1, "maximum": 3, "default": 2},
        "ratio": {"type": "number"},
        "size": {"type": "string", "enum": ["s", "m"], "enumNames": ["Small", "Medium"]},
        "tags": {"type": "array", "items": {"anyOf": [{"const": "a", "title": "A"}, {"const": "b", "title": "B"}]},
                 "maxItems": 1},
        "note": {"type": "string"},
    }}
    fields, qs = mcp_elicitation.form_questions(schema)
    kinds = {f.name: q["kind"] for f, q in zip(fields, qs)}
    assert kinds == {"mail": "text", "n": "number", "ratio": "number", "size": "choice", "tags": "multiple",
                     "note": "text"}
    by = {f.name: q for f, q in zip(fields, qs)}
    assert by["mail"]["format"] == "email" and by["mail"]["max_length"] == 40
    assert by["n"]["integer"] and by["n"]["default"] == 2 and by["n"]["min_value"] == 1
    assert by["size"]["choices"] == ["Small", "Medium"] and by["tags"]["max"] == 1
    answers = asks.check_answers(qs, {"f1": "me@example.com", "f2": "3", "f3": 0.5, "f4": "Medium", "f5": ["B"],
                                      "f6": ""})
    assert mcp_elicitation.form_content(fields, answers) == {
        "mail": "me@example.com", "n": 3, "ratio": 0.5, "size": "m", "tags": ["b"]}


@pytest.mark.parametrize("schema", [
    None, {"type": "object", "properties": {}},
    {"type": "object", "properties": {"x": {"type": "object"}}},
    {"type": "object", "properties": {f"f{i}": {"type": "string"} for i in range(11)}},
    {"type": "object", "properties": {"x": {"type": "array", "items": {"type": "string"}}}},
])
def test_a_form_the_card_cant_show_is_refused(schema):
    with pytest.raises(MCPRequestError) as exc:
        mcp_elicitation.form_questions(schema)
    assert exc.value.code == INVALID_PARAMS


def test_number_and_text_answers_are_checked():
    qs = asks.check_questions([
        {"id": "n", "question": "How many?", "number": True, "integer": True, "min_value": 1, "max_value": 5},
        {"id": "m", "question": "Mail", "format": "email", "min_length": 5},
    ])
    assert asks.check_answers(qs, {"n": "4", "m": "a@b.co"}) == {"n": 4, "m": "a@b.co"}
    for bad in ({"n": 9, "m": "a@b.co"}, {"n": 2.5, "m": "a@b.co"}, {"n": True, "m": "a@b.co"},
                {"n": 2, "m": "nope"}, {"n": 2, "m": "a@b"}):
        with pytest.raises(asks.AskError):
            asks.check_answers(qs, bad)


# ── Who is asked ──────────────────────────────────────────────────────

PARAMS = {"message": "Create a server?", "requestedSchema": APPROVE_FORM}


@pytest.mark.parametrize("calls", [
    [],                                                    # nothing in flight
    [None],                                                # a call for nobody we may ask
    [Asker("u1", origin="web"), None],                     # …next to one that is
    [Asker("u1", origin="web"), Asker("u2", origin="web")],  # two people
    [Asker("u1", origin="scheduled")],                     # a run nobody attends
])
async def test_the_question_is_cancelled_when_nobody_single_is_there(calls):
    assert await mcp_elicitation.handle("ops", PARAMS, calls) == {"action": "cancel"}
    assert asks.open_for("u1") == [] and asks.open_for("u2") == []


async def test_url_mode_is_not_offered():
    with pytest.raises(MCPRequestError):
        await mcp_elicitation.handle("ops", {"mode": "url", "url": "https://x", "message": "m"},
                                     [Asker("u1", origin="web")])


async def test_the_person_answers_accept_decline_or_nothing():
    a = Asker("u1", agent="helper", agent_name="Helper", origin="web")
    task = asyncio.create_task(mcp_elicitation.handle("ops", PARAMS, [a]))
    card = await _card_for("u1")
    assert card["source"] == "mcp" and card["title"] == "ops" and card["agent_name"] == "Helper"
    assert card["detail"] == {"server": "ops", "message": "Create a server?"}
    assert a.asking
    asks.answer(card["id"], "u1", answers={"f1": True, "f2": "Falkenstein"})
    assert await task == {"action": "accept", "content": {"approve": True, "region": "fsn1"}}
    assert not a.asking

    task = asyncio.create_task(mcp_elicitation.handle("ops", PARAMS, [a]))
    asks.answer((await _card_for("u1"))["id"], "u1", declined=True)
    assert await task == {"action": "decline"}

    task = asyncio.create_task(mcp_elicitation.handle("ops", PARAMS, [a]))
    await _card_for("u1")
    await mcp_elicitation.call_ended(a)   # the tool call ended meanwhile
    assert await task == {"action": "cancel"}
    assert asks.open_for("u1") == []


async def test_two_calls_of_one_person_share_the_card_until_both_end():
    a, b = Asker("u1", origin="web"), Asker("u1", origin="scheduled")
    task = asyncio.create_task(mcp_elicitation.handle("ops", PARAMS, [a, b]))
    card = await _card_for("u1")
    assert card["agent_name"] == ""
    await mcp_elicitation.call_ended(b)
    assert asks.open_for("u1")             # the other call still waits on it
    await mcp_elicitation.call_ended(a)
    assert await task == {"action": "cancel"}


async def test_a_server_taking_its_question_back_closes_the_card():
    a = Asker("u1", origin="web")
    task = asyncio.create_task(mcp_elicitation.handle("ops", PARAMS, [a]))
    await _card_for("u1")
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert asks.open_for("u1") == [] and not a.asking


# ── The stdio transport, end to end with a fake server ────────────────


async def test_elicitation_is_declared_on_stdio_only():
    stdio = StdioTransport(SERVER)
    assert stdio.client_capabilities() == {}
    stdio.on_request = _no_ring  # anything set
    assert stdio.client_capabilities() == {"elicitation": {"form": {}}}
    assert stdio.protocol_version == "2025-06-18"
    from captain_claw.flight_deck.mcp_transport import HttpTransport

    http = HttpTransport({"name": "r", "url": "https://x"})
    http.on_request = _no_ring
    assert http.client_capabilities() == {} and http.protocol_version == "2025-03-26"


async def test_a_ping_with_the_calls_own_id_is_answered_not_taken_for_the_result(manager):
    out = await manager.call_tool("opsfake", "ping_me", {})
    assert out["content"][0]["text"] == "pong received"


async def test_the_server_asks_the_person_and_deploys_on_yes(manager):
    who = Asker("u1", agent="helper", agent_name="Helper", origin="web")
    call = asyncio.create_task(manager.call_tool("opsfake", "deploy", {"app": "shop"}, asker=who))
    card = await _card_for("u1")
    assert "Deploy shop to a new server?" in card["detail"]["message"]
    assert [q["question"] for q in card["questions"]] == ["Approve", "Region"]
    asks.answer(card["id"], "u1", answers={"f1": True, "f2": "Falkenstein"})
    out = await asyncio.wait_for(call, 10)
    assert out["content"][0]["text"] == "deployed to fsn1"


async def test_no_and_decline_change_nothing(manager):
    who = Asker("u1", origin="web")
    call = asyncio.create_task(manager.call_tool("opsfake", "deploy", {}, asker=who))
    asks.answer((await _card_for("u1"))["id"], "u1", answers={"f1": False, "f2": "Nuremberg"})
    assert (await asyncio.wait_for(call, 10))["content"][0]["text"] == "not approved: nothing changed"
    call = asyncio.create_task(manager.call_tool("opsfake", "deploy", {}, asker=who))
    asks.answer((await _card_for("u1"))["id"], "u1", declined=True)
    assert (await asyncio.wait_for(call, 10))["content"][0]["text"] == "decline: nothing changed"


async def test_a_call_for_nobody_gets_its_question_cancelled(manager):
    out = await asyncio.wait_for(manager.call_tool("opsfake", "deploy", {}), 10)
    assert out["content"][0]["text"] == "cancel: nothing changed"


async def test_stopping_the_call_tells_the_server_and_closes_the_card(manager):
    who = Asker("u1", origin="web")
    call = asyncio.create_task(manager.call_tool("opsfake", "deploy", {}, asker=who))
    await _card_for("u1")
    call.cancel()
    with pytest.raises(asyncio.CancelledError):
        await call
    for _ in range(100):
        if not asks.open_for("u1"):
            break
        await asyncio.sleep(0.05)
    assert asks.open_for("u1") == []
    # The child is still healthy afterwards.
    assert (await manager.call_tool("opsfake", "echo", {"text": "hi"}))["content"][0]["text"] == "hi"


def test_request_keys_tell_string_and_number_ids_apart():
    assert _request_key(1) != _request_key("1")


# ── The agent's routes ────────────────────────────────────────────────


def _app(user: dict | None = None) -> FastAPI:
    from captain_claw.flight_deck import ask_routes

    app = FastAPI()
    app.include_router(mcp_routes.router)
    app.include_router(ask_routes.router)
    if user is not None:
        app.dependency_overrides[get_current_user] = lambda: user
    return app


def _client(user: dict | None = None) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=_app(user), client=LOOPBACK),
                             base_url="http://deck.test")


@pytest.fixture
def agent_caller(as_agent, monkeypatch):
    monkeypatch.setattr(mcp_routes, "require_agent_caller",
                        lambda request, what="": agent_identity.AgentCaller(slug="helper", owner=as_agent["owner"]))
    return as_agent


def _call(tool: str, n: int, **extra) -> dict:
    return {"server": "opsfake", "tool": tool, "arguments": extra.pop("arguments", {}), "origin": "web",
            "session_id": "chat-1", "wait": 0.3, "call_id": f"xm_{n:016x}", **extra}


def test_the_call_route_is_grant_aware():
    assert mcp_routes.GRANT_AWARE <= speaker_grants.GRANT_AWARE_PATHS
    assert "/fd/mcp/agent/calls/follow" not in speaker_grants.GRANT_AWARE_PATHS


async def test_an_agent_follows_a_call_while_its_owner_answers(deck, agent_caller, manager):
    agent_caller["owner"] = deck.user["id"]
    async with _client() as agent, _client(deck.user) as person:
        first = (await agent.post("/fd/mcp/agent/calls", json=_call("deploy", 1, arguments={"app": "shop"}))).json()
        assert first["pending"] == "xm_0000000000000001"
        card = (await _card_for(deck.user["id"]))
        assert card["agent_name"]  # the agent's display name (or slug)
        body = None
        for _ in range(40):
            body = (await agent.post("/fd/mcp/agent/calls/follow", json={"id": first["pending"], "wait": 0.2})).json()
            if body.get("asking"):
                break
        assert body == {"pending": first["pending"], "asking": True, "seconds": body["seconds"]}
        ok = await person.post(f"/fd/asks/{card['id']}", json={"answers": {"f1": True, "f2": "Nuremberg"}})
        assert ok.status_code == 200
        for _ in range(40):
            body = (await agent.post("/fd/mcp/agent/calls/follow", json={"id": first["pending"], "wait": 0.5})).json()
            if "result" in body:
                break
        assert body["result"]["content"][0]["text"] == "deployed to nbg1"
        assert body["server"] == "opsfake" and body["tool"] == "deploy"


async def test_a_member_is_asked_on_their_own_turn_not_the_owner(deck, agent_caller, manager):
    agent_caller["owner"] = deck.user["id"]
    agent_caller["member"] = deck.other["id"]
    async with _client() as agent:
        await agent.post("/fd/mcp/agent/calls", json=_call("deploy", 2))
        await _card_for(deck.other["id"])
        assert asks.open_for(deck.user["id"]) == []
        await agent.post("/fd/mcp/agent/calls/cancel", json={"id": "xm_0000000000000002"})


async def test_cancel_stops_the_call_and_closes_the_card(deck, agent_caller, manager):
    agent_caller["owner"] = deck.user["id"]
    async with _client() as agent:
        await agent.post("/fd/mcp/agent/calls", json=_call("deploy", 3))
        await _card_for(deck.user["id"])
        out = (await agent.post("/fd/mcp/agent/calls/cancel", json={"id": "xm_0000000000000003"})).json()
        assert out == {"cancelled": True}
        for _ in range(100):
            if not asks.open_for(deck.user["id"]):
                break
            await asyncio.sleep(0.05)
        assert asks.open_for(deck.user["id"]) == []
        gone = await agent.post("/fd/mcp/agent/calls/follow", json={"id": "xm_0000000000000003", "wait": 1})
        assert gone.status_code == 502 and "Stopped" in gone.json()["detail"]


async def test_a_scheduled_run_and_a_grantless_member_turn_ask_nobody(deck, agent_caller, manager):
    agent_caller["owner"] = deck.user["id"]
    async with _client() as agent:
        for n, extra in ((4, {"origin": "scheduled"}), (5, {"nobody": True})):
            body = (await agent.post("/fd/mcp/agent/calls", json=_call("deploy", n, **extra))).json()
            for _ in range(40):
                if "result" in body:
                    break
                body = (await agent.post("/fd/mcp/agent/calls/follow",
                                         json={"id": f"xm_{n:016x}", "wait": 0.5})).json()
            assert body["result"]["content"][0]["text"] == "cancel: nothing changed"
        assert asks.open_for(deck.user["id"]) == []


async def test_another_agent_cant_follow_or_cancel_a_call(deck, agent_caller, manager, monkeypatch):
    agent_caller["owner"] = deck.user["id"]
    async with _client() as agent:
        await agent.post("/fd/mcp/agent/calls", json=_call("wait", 6, arguments={"seconds": 2}))
        monkeypatch.setattr(mcp_routes, "require_agent_caller",
                            lambda request, what="": agent_identity.AgentCaller(slug="other", owner=deck.user["id"]))
        assert (await agent.post("/fd/mcp/agent/calls/follow",
                                 json={"id": "xm_0000000000000006", "wait": 0})).status_code == 404
        assert (await agent.post("/fd/mcp/agent/calls/cancel",
                                 json={"id": "xm_0000000000000006"})).json() == {"cancelled": False}


async def test_a_quiet_call_is_stopped(deck, agent_caller, manager, monkeypatch):
    agent_caller["owner"] = deck.user["id"]
    monkeypatch.setattr(mcp_calls, "QUIET_LIMIT_S", 0.2)
    async with _client() as agent:
        await agent.post("/fd/mcp/agent/calls", json=_call("wait", 7, arguments={"seconds": 5}))
        await asyncio.sleep(0.3)
        out = await agent.post("/fd/mcp/agent/calls/follow", json={"id": "xm_0000000000000007", "wait": 0.1})
        assert out.status_code == 502 and "no answer from the server" in out.json()["detail"]


async def test_a_browser_app_call_asks_the_signed_in_person(deck, manager):
    async with _client(deck.user) as person:
        call = asyncio.create_task(person.post("/fd/mcp/opsfake/user_call", json={"tool": "deploy", "arguments": {}}))
        card = await _card_for(deck.user["id"])
        asks.answer(card["id"], deck.user["id"], answers={"f1": True, "f2": "Falkenstein"})
        out = (await asyncio.wait_for(call, 10)).json()
        assert out["result"]["content"][0]["text"] == "deployed to fsn1"


# ── The agent's side ──────────────────────────────────────────────────


class _FD:
    """Stands in for Flight Deck's calls client: scripted answers per path."""

    def __init__(self, answers: dict[str, list[tuple[int, dict]]]):
        self.answers = answers
        self.posts: list[tuple[str, dict]] = []

    async def post(self, path, *, json=None, params=None, headers=None):
        self.posts.append((path, json))
        status, body = self.answers[path].pop(0) if self.answers.get(path) else (200, {})
        return httpx.Response(status, json=body, request=httpx.Request("POST", "http://fd" + path))


async def test_the_agent_follows_and_says_it_waits_for_the_person():
    from captain_claw.tools.mcp_connector import MCPProxyConnector

    result = {"server": "ops", "tool": "deploy", "result": {"content": [{"type": "text", "text": "deployed"}]}}
    fd = _FD({"/fd/mcp/agent/calls": [(200, {"pending": "x", "asking": False})],
              "/fd/mcp/agent/calls/follow": [(200, {"pending": "x", "asking": True}),
                                             (200, {"pending": "x", "asking": True}), (200, result)]})
    said: list[str] = []
    conn = MCPProxyConnector("ops", fd_client=fd, calls_client=fd)
    text = await conn.call_tool("deploy", {"app": "shop"}, stream=said.append)
    assert text == "deployed"
    assert said == ["Waiting for your answer in Flight Deck: ops asks you before it goes on…\n"]
    start = fd.posts[0][1]
    assert start["server"] == "ops" and start["tool"] == "deploy" and start["call_id"].startswith("xm_")
    assert [p for p, _ in fd.posts].count("/fd/mcp/agent/calls/follow") == 3
    assert "/fd/mcp/agent/calls/cancel" not in [p for p, _ in fd.posts]


async def test_the_agent_falls_back_on_an_older_flight_deck():
    from captain_claw.tools.mcp_connector import MCPProxyConnector

    result = {"server": "ops", "tool": "echo", "result": {"content": [{"type": "text", "text": "hi"}]}}
    fd = _FD({"/fd/mcp/agent/calls": [(404, {"detail": "Not Found"})], "/fd/mcp/ops/call": [(200, result)]})
    assert await MCPProxyConnector("ops", fd_client=fd, calls_client=fd).call_tool("echo", {}) == "hi"


async def test_a_stopped_turn_cancels_the_call():
    from captain_claw.tools.mcp_connector import MCPProxyConnector

    class _Slow(_FD):
        async def post(self, path, **kw):
            if path.endswith("/follow"):
                await asyncio.sleep(10)
            return await super().post(path, **kw)

    fd = _Slow({"/fd/mcp/agent/calls": [(200, {"pending": "x", "asking": True})]})
    task = asyncio.create_task(MCPProxyConnector("ops", fd_client=fd, calls_client=fd).call_tool("deploy", {}))
    await asyncio.sleep(0.1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert fd.posts[-1][0] == "/fd/mcp/agent/calls/cancel"
    assert fd.posts[-1][1]["id"] == fd.posts[0][1]["call_id"]
