# ruff: noqa: F811 — `data`, `deck` and `as_agent` are fixtures imported from the Custom tools tests
"""A password or a code the person types for an agent (request_secret): a masked
box in Flight Deck, a reference for the agent, the value released only to that
agent for that person, a few times, for a few minutes — and never in what the
browser tells the model. Also: watching the agent's browser and taking control."""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI

from captain_claw import secrets_client
from captain_claw.flight_deck import ask_routes, secrets_store
from captain_claw.flight_deck import person_asks as asks
from captain_claw.flight_deck.auth import get_current_user
from captain_claw.tools import browser as browser_mod
from captain_claw.tools import browser_live
from captain_claw.tools.registry import ToolResult

from .test_external_tools import LOOPBACK, data, deck  # noqa: F401
from .test_external_tools_agent import as_agent  # noqa: F401

SECRET_Q = {"questions": [{"id": "secret", "question": "Your password for portal.example.test", "secret": True}]}


def _client(user: dict | None = None) -> httpx.AsyncClient:
    app = FastAPI()
    app.include_router(ask_routes.agent_router)
    app.include_router(ask_routes.router)
    if user is not None:
        app.dependency_overrides[get_current_user] = lambda: user
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app, client=LOOPBACK), base_url="http://deck.test")


@pytest.fixture(autouse=True)
def fresh(monkeypatch):
    asks.reset_for_tests()
    secrets_store.reset_for_tests()
    browser_live.reset_for_tests()
    monkeypatch.setattr(asks, "MIN_WAIT_S", 0.3)
    yield
    asks.reset_for_tests()
    secrets_store.reset_for_tests()
    browser_live.reset_for_tests()


async def _typed(agent, person, value="hunter2!") -> str:
    first = (await agent.post("/fd/asks/agent", json={**SECRET_Q, "wait": 0.2, "origin": "web"})).json()
    for _ in range(50):
        found = (await person.get("/fd/asks", params={"wait": 1})).json()["asks"]
        if found:
            break
    q = found[0]
    assert q["questions"][0]["kind"] == "secret"
    assert (await person.post(f"/fd/asks/{q['id']}", json={"answers": {"secret": value}})).status_code == 200
    done = (await agent.post("/fd/asks/agent/follow", json={"id": first["pending"], "wait": 1})).json()
    assert done["ok"] and value not in str(done)
    given = done["answers"]["secret"]
    assert given["length"] == len(value) and given["secret_ref"].startswith("sec_")
    return given["secret_ref"]


# ── Flight Deck holds it ──────────────────────────────────────────────


async def test_the_agent_gets_a_reference_and_only_it_can_have_the_value(deck, as_agent):
    as_agent["owner"] = deck.user["id"]
    async with _client() as agent, _client(deck.user) as person:
        ref = await _typed(agent, person)
        # Nothing about the question keeps the value: not the kept outcome, not the bell.
        kept = asks._DONE[next(iter(asks._DONE))][2]
        assert "hunter2!" not in str(kept)
        assert all("hunter2!" not in str(n) for n in await deck.db.list_notifications(deck.user["id"]))
        got = (await agent.post("/fd/asks/agent/secret", json={"ref": ref})).json()
        assert got == {"value": "hunter2!"}
        # Another agent of the same person can't have it.
        as_agent_slug = secrets_store._SECRETS[ref].agent
        secrets_store._SECRETS[ref].agent = "someone-else"
        assert (await agent.post("/fd/asks/agent/secret", json={"ref": ref})).status_code == 404
        secrets_store._SECRETS[ref].agent = as_agent_slug
        assert (await agent.post("/fd/asks/agent/secret", json={"ref": "sec_0000000000000000"})).status_code == 404
        assert (await agent.post("/fd/asks/agent/secret", json={"ref": "not-a-ref"})).status_code == 422


async def test_three_uses_then_gone(deck, as_agent):
    as_agent["owner"] = deck.user["id"]
    async with _client() as agent, _client(deck.user) as person:
        ref = await _typed(agent, person)
        for _ in range(secrets_store.MAX_USES):
            assert (await agent.post("/fd/asks/agent/secret", json={"ref": ref})).status_code == 200
        assert (await agent.post("/fd/asks/agent/secret", json={"ref": ref})).status_code == 404


def test_old_secrets_are_forgotten(monkeypatch):
    ref = secrets_store.keep("1234", "u1", "helper")
    assert secrets_store.release(ref, "u1", "helper") == "1234"
    monkeypatch.setattr(time, "time", lambda: 10**12)
    with pytest.raises(KeyError):
        secrets_store.release(ref, "u1", "helper")


def test_a_secret_question_needs_a_value_and_has_no_default():
    (q,) = asks.check_questions([{"id": "secret", "question": "Code?", "secret": True, "default": "0000"}])
    assert q["kind"] == "secret" and "default" not in q
    with pytest.raises(asks.AskError):
        asks.check_answers([q], {"secret": ""})


async def test_only_an_agents_question_can_take_a_secret():
    (q,) = asks.check_questions([{"id": "secret", "question": "Code?", "secret": True}])
    a = await asks.open_ask("u1", title="A tool", questions=[q], wait_s=60)
    with pytest.raises(asks.AskError):
        asks.answer(a.id, "u1", answers={"secret": "1234"})


# ── The tool ──────────────────────────────────────────────────────────


async def test_the_tool_hands_back_a_reference_not_the_value(monkeypatch):
    from captain_claw.tools import request_secret as rs

    async def asked(questions, wait_s, agent, stream=None, waiting=""):
        assert questions[0]["secret"] is True and questions[0]["question"].startswith("The code")
        return 200, {"ok": True, "answers": {"secret": {"secret_ref": "sec_0123456789abcdef", "length": 6}}}

    monkeypatch.setattr(rs, "ask_and_follow", asked)
    monkeypatch.setattr(rs, "_fd_base", lambda: "http://deck.test")
    out = await rs.RequestSecretTool().execute(what="The code your bank sent", kind="code")
    assert out.success and "secret_ref='sec_0123456789abcdef'" in out.content and "6 characters" in out.content


async def test_the_tool_never_asks_in_a_message(monkeypatch):
    from captain_claw.tools import request_secret as rs

    async def elsewhere(*a, **k):
        return 200, {"ok": False, "elsewhere": True, "error": "They're talking to you in telegram."}

    monkeypatch.setattr(rs, "ask_and_follow", elsewhere)
    monkeypatch.setattr(rs, "_fd_base", lambda: "http://deck.test")
    out = await rs.RequestSecretTool().execute(what="Your password")
    assert not out.success and "Don't ask for the secret in a message" in out.error


# ── The browser types it and never says it ────────────────────────────


def test_scrub_plain_and_escaped():
    r = ToolResult(success=True, content='Field now reads "pa\\"ss" and pa"ss', error="")
    secrets_client.scrub(r, 'pa"ss')
    assert 'pa"ss' not in r.content and 'pa\\"ss' not in r.content


@pytest.fixture
def browser(monkeypatch):
    browser_mod._SESSION_SECRETS.clear()
    browser_mod._WORKFLOW_RECORDERS.clear()
    seen: list[dict] = []

    async def released(ref):
        if ref != "sec_0123456789abcdef":
            raise secrets_client.SecretError("That secret is gone.")
        return "Tr0ub4dor&3"

    async def run(self, action, **kwargs):
        seen.append({"action": action, **kwargs})
        if action in ("type", "form_input"):
            return ToolResult(success=True, content=f'Password box now reads "{kwargs.get("text")}".')
        return ToolResult(success=True, content="POST /login body: user=ana&pass=Tr0ub4dor%263 pass=Tr0ub4dor&3")

    monkeypatch.setattr(secrets_client, "release", released)
    monkeypatch.setattr(browser_mod.BrowserTool, "_execute", run)
    yield browser_mod.BrowserTool(), seen
    browser_mod._SESSION_SECRETS.clear()


async def test_type_with_a_reference_types_the_value_and_says_nothing_of_it(browser):
    tool, seen = browser
    out = await tool.execute("type", ref="ref_4", secret_ref="sec_0123456789abcdef", _session_id="s1")
    assert seen[0]["text"] == "Tr0ub4dor&3" and "secret_ref" not in seen[0]
    assert out.success and "Tr0ub4dor&3" not in out.content and "[the secret]" in out.content
    assert "typed the secret the person gave" in out.content
    # Later results of the same session are scrubbed too (a captured request).
    later = await tool.execute("network_list", _session_id="s1")
    assert "Tr0ub4dor&3" not in later.content and "Tr0ub4dor%263" not in later.content
    # Another session's aren't touched.
    other = await tool.execute("network_list", _session_id="s2")
    assert "Tr0ub4dor&3" in other.content


async def test_a_reference_only_types_and_never_while_recording(browser):
    tool, seen = browser
    assert "type and form_input only" in (await tool.execute("navigate", secret_ref="sec_0123456789abcdef")).error
    gone = await tool.execute("type", secret_ref="sec_ffffffffffffffff", _session_id="s1")
    assert not gone.success and "gone" in gone.error
    rec = browser_mod.WorkflowRecorder()
    rec._is_recording = True
    browser_mod._WORKFLOW_RECORDERS["s1"] = rec
    out = await tool.execute("type", secret_ref="sec_0123456789abcdef", _session_id="s1")
    assert not out.success and "recording" in out.error
    assert seen == []


# ── Watching the browser and taking control ───────────────────────────


class _Page:
    url = "https://portal.example.test/login"
    viewport_size = {"width": 1280, "height": 800}

    def __init__(self):
        self.did: list[tuple] = []
        self.mouse = SimpleNamespace(click=self._click, wheel=self._wheel)
        self.keyboard = SimpleNamespace(insert_text=self._insert, press=self._press)

    async def _click(self, x, y):
        self.did.append(("click", x, y))

    async def _wheel(self, dx, dy):
        self.did.append(("scroll", dy))

    async def _insert(self, text):
        self.did.append(("type", text))

    async def _press(self, key):
        self.did.append(("key", key))

    async def title(self):
        return "Sign in"

    async def screenshot(self, **kw):
        return b"\xff\xd8jpeg"


@pytest.fixture
def page(monkeypatch):
    p = _Page()
    browser_mod._BROWSER_SESSIONS["s1"] = SimpleNamespace(is_alive=True, _page=p)
    browser_live.note_session("s1")
    yield p
    browser_mod._BROWSER_SESSIONS.pop("s1", None)


async def test_the_person_watches_takes_control_and_hands_back(page, monkeypatch):
    frame = await browser_live.frame()
    assert frame["alive"] and frame["url"].endswith("/login") and frame["frame"] and not frame["held"]
    assert (await browser_live.person_input("click", x=10, y=20))["ok"] is False  # not theirs yet
    browser_live.take("Ana")
    assert browser_live.status()["held"] and browser_live.status()["by"] == "Ana"
    assert (await browser_live.person_input("click", x=10, y=20))["ok"]
    assert (await browser_live.person_input("type", text="ana@example.test"))["ok"]
    assert (await browser_live.person_input("key", key="Enter"))["ok"]
    assert (await browser_live.person_input("key", key="F12"))["ok"] is False
    assert page.did == [("click", 10.0, 20.0), ("type", "ana@example.test"), ("key", "Enter")]
    # The agent's next browser step waits until they hand it back.
    waiting = asyncio.create_task(browser_live.wait_while_held())
    await asyncio.sleep(0.05)
    assert not waiting.done()
    await browser_live.give_back()
    assert await asyncio.wait_for(waiting, 1) == ""
    note = browser_live.pop_note()
    assert note.startswith("Ana had control of the browser for") and "Sign in — https://portal" in note
    assert browser_live.pop_note() == ""


async def test_held_too_long_the_agent_is_told_to_wait(page):
    browser_live.take("Ana")
    why = await browser_live.wait_while_held(max_s=0.05)
    assert "still has control" in why
    await browser_live.give_back()


async def test_flight_deck_passes_it_through_for_the_owner_only(deck, monkeypatch):
    from captain_claw.flight_deck import browser_live_routes as blr
    from captain_claw.flight_deck import server

    monkeypatch.setattr(server, "_load_process_registry",
                        lambda: {"helper": {"owner": deck.user["id"], "web_port": 1, "web_auth": "t"}})
    monkeypatch.setattr(server, "_process_is_alive", lambda slug: True)
    calls: list[tuple] = []

    async def call(entry, method, path, body=None):
        calls.append((method, path, body))
        return {"alive": False, "held": False}

    monkeypatch.setattr(blr, "_call", call)
    app = FastAPI()
    app.include_router(blr.router)
    who = {"user": deck.user}
    app.dependency_overrides[get_current_user] = lambda: who["user"]
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://deck.test") as c:
        assert (await c.get("/fd/agents/helper/browser/live")).status_code == 200
        assert (await c.post("/fd/agents/helper/browser/control", json={"take": True})).status_code == 200
        assert calls[-1] == ("POST", "/api/browser/control", {"take": True, "by": deck.user["email"]})
        assert (await c.post("/fd/agents/helper/browser/input", json={"kind": "drag"})).status_code == 422
        who["user"] = deck.other
        assert (await c.get("/fd/agents/helper/browser/live")).status_code == 404
        who["user"] = deck.admin
        assert (await c.get("/fd/agents/helper/browser/live")).status_code == 200
