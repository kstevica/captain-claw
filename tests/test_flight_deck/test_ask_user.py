# ruff: noqa: F811 — `data`, `deck` and `as_agent` are fixtures imported from the Custom tools tests
"""ask_user: an agent asks the person its turn is for, in their Flight Deck, and
waits. The question is the same as a custom tool's (person_asks): one dialog in
every tab, a ring, answered once. Nobody to ask on a turn that started on its
own; elsewhere than Flight Deck the agent asks in its reply."""

from __future__ import annotations

import asyncio

import httpx
import pytest
from fastapi import FastAPI

from captain_claw.flight_deck import ask_routes
from captain_claw.flight_deck import person_asks as asks
from captain_claw.flight_deck.auth import get_current_user
from captain_claw.tools import ask_user
from captain_claw.tools.ask_user import AskUserTool

from .test_external_tools import LOOPBACK, data, deck  # noqa: F401
from .test_external_tools_agent import as_agent  # noqa: F401

ONE = {"questions": [{"id": "answer", "question": "Which colour?", "choices": ["Blue", "Green"]}]}


def _app(user: dict | None = None) -> FastAPI:
    app = FastAPI()
    app.include_router(ask_routes.agent_router)
    app.include_router(ask_routes.router)
    if user is not None:
        app.dependency_overrides[get_current_user] = lambda: user
    return app


def _client(user: dict | None = None) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=_app(user), client=LOOPBACK), base_url="http://deck.test")


@pytest.fixture(autouse=True)
def fresh(monkeypatch):
    asks.reset_for_tests()
    monkeypatch.setattr(asks, "MIN_WAIT_S", 0.3)
    yield
    asks.reset_for_tests()


async def _open_question(c: httpx.AsyncClient) -> dict:
    for _ in range(50):
        found = (await c.get("/fd/asks", params={"wait": 1})).json()["asks"]
        if found:
            return found[0]
    raise AssertionError("no question opened")


async def test_an_agent_asks_its_owner_and_follows_the_answer(deck, as_agent):
    uid = deck.user["id"]
    as_agent["owner"] = uid
    async with _client() as agent, _client(deck.user) as person:
        first = (await agent.post("/fd/asks/agent", json={**ONE, "wait": 0.2, "origin": "web"})).json()
        assert set(first) == {"pending", "seconds_left"}
        q = await _open_question(person)
        assert q["id"] == first["pending"] and q["source"] == "agent" and q["title"] == "helper"
        assert q["questions"][0]["choices"] == ["Blue", "Green"]
        bells = [n for n in await deck.db.list_notifications(uid) if n.get("ref_type") == asks.BELL_REF]
        assert bells and bells[0]["title"] == "helper asks you something"
        # Someone else can't answer it; the person can.
        async with _client(deck.other) as other:
            assert (await other.post(f"/fd/asks/{q['id']}", json={"answers": {"answer": "Blue"}})).status_code == 404
        assert (await person.post(f"/fd/asks/{q['id']}", json={"answers": {"answer": "Green"}})).status_code == 200
        done = (await agent.post("/fd/asks/agent/follow", json={"id": q["id"], "wait": 1})).json()
    assert done == {"ok": True, "answers": {"answer": "Green"}}
    assert all(n.get("resolved") for n in await deck.db.list_notifications(uid) if n.get("ref_type") == asks.BELL_REF)


@pytest.mark.parametrize("origin, key", [("scheduled", "nobody"), ("autonomy:plan", "nobody"), ("exam", "nobody"),
                                         ("telegram", "elsewhere"), ("whatsapp", "elsewhere"), ("public", "elsewhere")])
async def test_nobody_to_ask_or_somewhere_the_dialog_doesnt_reach(deck, as_agent, origin, key):
    as_agent["owner"] = deck.user["id"]
    async with _client() as agent:
        res = (await agent.post("/fd/asks/agent", json={**ONE, "origin": origin})).json()
    assert res["ok"] is False and res[key] is True
    assert asks.open_for(deck.user["id"]) == []


async def test_a_members_question_goes_to_the_member(deck, as_agent):
    as_agent.update(owner=deck.user["id"], member=deck.other["id"])
    async with _client() as agent:
        first = (await agent.post("/fd/asks/agent", json={**ONE, "wait": 0, "origin": "member"})).json()
    assert asks.open_for(deck.user["id"]) == []
    assert [a["id"] for a in asks.open_for(deck.other["id"])] == [first["pending"]]
    # The owner's agent turn can't follow the member's question.
    as_agent["member"] = None
    async with _client() as agent:
        assert (await agent.post("/fd/asks/agent/follow", json={"id": first["pending"], "wait": 0})).status_code == 404


async def test_a_stopped_turn_closes_its_question_even_before_it_opened(deck, as_agent):
    uid = deck.user["id"]
    as_agent["owner"] = uid
    async with _client() as agent:
        early = "xa_" + "1" * 16
        assert (await agent.post("/fd/asks/agent/cancel", json={"id": early})).json() == {"cancelled": False}
        late = await agent.post("/fd/asks/agent", json={**ONE, "wait": 0, "ask_id": early})
        assert late.status_code == 422 and "cancelled" in late.json()["detail"]
        first = (await agent.post("/fd/asks/agent", json={**ONE, "wait": 0, "ask_id": "xa_" + "2" * 16})).json()
        assert (await agent.post("/fd/asks/agent/cancel", json={"id": first["pending"]})).json() == {"cancelled": True}
        after = (await agent.post("/fd/asks/agent/follow", json={"id": first["pending"], "wait": 0})).json()
    assert after["no_answer"] is True and asks.open_for(uid) == []
    async with _client() as agent:
        bad = await agent.post("/fd/asks/agent", json={"questions": [{"id": "a", "question": "x", "choices": []}]})
    assert bad.status_code == 422


def test_the_tools_arguments_and_answers_in_words():
    assert ask_user.questions_from({"question": "Ready?", "confirm": True, "help": "", "_agent": object()}) == [
        {"id": "answer", "question": "Ready?", "confirm": True}]
    form = [{"id": "a", "question": "A?"}, {"id": "b", "question": "B?", "confirm": True}]
    assert ask_user.questions_from({"questions": form, "question": "ignored"}) == form
    assert ask_user.questions_from({}) == []
    assert ask_user.describe_answers([{"id": "answer", "question": "Which?"}], {"answer": "Green"}) == "They answered: Green"
    words = ask_user.describe_answers([*form, {"id": "c", "question": "C?"}], {"a": ["x", "y"], "b": False, "c": ""})
    assert "- A?: x, y" in words and "- B?: no" in words and "- C?: (left empty)" in words and '"b": false' in words


@pytest.fixture
def tool_line(monkeypatch):
    """The tool's requests reach the deck's agent routes in-process."""
    async def call(path, body):
        async with _client() as agent:
            resp = await agent.post(f"/fd/asks/agent{path}", json=body)
        return resp.status_code, resp.json()

    monkeypatch.setattr(ask_user, "_fd_base", lambda: "http://deck.test")
    monkeypatch.setattr(ask_user, "_call", call)


class Turn:
    _turn_source = ("web", "Web chat")


async def test_the_tool_asks_waits_and_says_what_they_answered(deck, as_agent, tool_line):
    as_agent["owner"] = deck.user["id"]
    said: list[str] = []
    run = asyncio.create_task(AskUserTool().execute(question="Which colour?", choices=["Blue", "Green"],
                                                    _agent=Turn(), _stream_callback=said.append))
    async with _client(deck.user) as person:
        q = await _open_question(person)
        await asyncio.sleep(0.3)
        await person.post(f"/fd/asks/{q['id']}", json={"answers": {"answer": "Blue"}})
    result = await asyncio.wait_for(run, 60)
    assert result.success and result.content == "They answered: Blue"
    # "Don't answer" is an answer too.
    run = asyncio.create_task(AskUserTool().execute(question="Send it?", confirm=True, _agent=Turn()))
    async with _client(deck.user) as person:
        q = await _open_question(person)
        await person.post(f"/fd/asks/{q['id']}", json={"declined": True})
    result = await asyncio.wait_for(run, 60)
    assert result.success and result.content.startswith("They chose not to answer")


async def test_the_tool_where_it_cant_reach_them(deck, as_agent, tool_line, monkeypatch):
    as_agent["owner"] = deck.user["id"]

    class Telegram:
        _turn_source = ("telegram", "Telegram")

    away = await AskUserTool().execute(question="Which?", _agent=Telegram())
    assert not away.success and "ask in your reply instead" in away.error
    assert not (await AskUserTool().execute(_agent=Turn())).success
    monkeypatch.setattr(ask_user, "_fd_base", lambda: "")
    assert (await AskUserTool().execute(question="Which?")).error == ask_user.NO_FLIGHT_DECK


async def test_stopping_the_turn_closes_the_question(deck, as_agent, tool_line):
    uid = deck.user["id"]
    as_agent["owner"] = uid
    run = asyncio.create_task(AskUserTool().execute(question="Which colour?", _agent=Turn()))
    async with _client(deck.user) as person:
        await _open_question(person)
    run.cancel()
    with pytest.raises(asyncio.CancelledError):
        await run
    for _ in range(50):
        if not asks.open_for(uid):
            break
        await asyncio.sleep(0.1)
    assert asks.open_for(uid) == []


def test_members_may_ask_and_the_paths_are_grant_aware():
    from captain_claw import speaker
    from captain_claw.flight_deck import speaker_grants, tool_risk

    assert "ask_user" in speaker.SPEAKER_TOOL_ALLOWLIST_MAX and "ask_user" in speaker.SPEAKER_PATH_MAP
    assert ask_routes.GRANT_AWARE <= speaker_grants.GRANT_AWARE_PATHS
    assert tool_risk.risk_for("ask_user") == "read"


async def test_a_question_waits_in_the_bell_until_it_is_answered(deck, as_agent):
    """A Needs-you "ask" on the Line: reading the bell doesn't settle it, answering does."""
    from captain_claw.flight_deck import line

    assert line.classify("person_ask", "person_ask")[:2] == ("ask", True)
    uid = deck.user["id"]
    as_agent["owner"] = uid
    async with _client() as agent, _client(deck.user) as person:
        first = (await agent.post("/fd/asks/agent", json={**ONE, "wait": 0})).json()
        rows = [n for n in await deck.db.list_notifications(uid) if n.get("ref_type") == asks.BELL_REF]
        assert rows and not rows[0].get("resolved")
        await deck.db.mark_line_item_read(rows[0]["id"], uid)
        rows = [n for n in await deck.db.list_notifications(uid) if n.get("ref_type") == asks.BELL_REF]
        assert not rows[0].get("resolved")  # read, still waiting for its answer
        await person.post(f"/fd/asks/{first['pending']}", json={"answers": {"answer": "Blue"}})
        await agent.post("/fd/asks/agent/follow", json={"id": first["pending"], "wait": 1})
    rows = [n for n in await deck.db.list_notifications(uid) if n.get("ref_type") == asks.BELL_REF]
    assert rows[0].get("resolved")
