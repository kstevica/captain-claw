# ruff: noqa: F811 — `data`, `deck` and `as_agent` are fixtures imported from the Custom tools tests
"""Approval cards: before an agent pays, sends, posts, shares, deletes or stores
a password, the deck's policy decides — don't ask, ask (once / this
conversation / always here), ask every time, or people do it themselves.
The card is a question in the person's Flight Deck; a run nobody is there for
can only do what the person chose to always allow."""

from __future__ import annotations

import httpx
import pytest
from fastapi import FastAPI

from captain_claw import approvals_client, tool_audit
from captain_claw.approval_kinds import describe, kind_for
from captain_claw.flight_deck import approval_routes, approvals
from captain_claw.flight_deck import person_asks as asks
from captain_claw.flight_deck.auth import get_current_user

from .test_external_tools import LOOPBACK, data, deck  # noqa: F401
from .test_external_tools_agent import as_agent  # noqa: F401

SEND = {"kind": "send", "tool": "send_mail", "summary": "send_mail\nto: ana@example.com", "origin": "web",
        "session_id": "chat-1", "wait": 0.2}


# ── Which calls are which kind ────────────────────────────────────────


@pytest.mark.parametrize("tool, params, kind", [
    ("send_mail", {"to": "a@x.hr"}, "send"),
    ("google_mail", {"action": "send"}, "send"),
    ("google_mail", {"action": "create_draft"}, ""),
    ("google_mail", {"action": "list_messages"}, ""),
    ("microsoft_mail", {"action": "reply"}, "send"),
    ("imap_mail", {"action": "trash"}, "delete"),
    ("whatsapp_web", {"action": "send"}, "send"),
    ("whatsapp_web", {"action": "chats"}, ""),
    ("twitter", {"action": "post"}, "post"),
    ("twitter", {"action": "dm"}, "send"),
    ("twitter", {"action": "timeline"}, ""),
    ("public_link", {"action": "share"}, "share"),
    ("hosting", {"action": "publish"}, "share"),
    ("payments", {"action": "purchase"}, "pay"),
    ("payments", {"action": "prepare"}, ""),
    ("spend", {"action": "authorize"}, "pay"),
    ("vfs", {"action": "rm"}, "delete"),
    ("google_calendar", {"action": "delete_event"}, "delete"),
    ("contacts", {"action": "remove"}, "delete"),
    ("datastore", {"action": "drop_table"}, "delete"),
    ("datastore", {"action": "sql", "sql": "  DELETE FROM t WHERE 1"}, "delete"),
    ("datastore", {"action": "sql", "sql": "SELECT * FROM t -- delete later"}, ""),
    ("browser", {"action": "credentials_store"}, "credentials"),
    ("browser", {"action": "click", "text": "Pay now"}, "pay"),
    ("browser", {"action": "click", "text": "Send"}, "send"),
    ("browser", {"action": "click", "text": "Send feedback form"}, ""),
    ("browser", {"action": "click", "text": "Delete"}, "delete"),
    ("browser", {"action": "navigate", "url": "https://x.test/pay"}, ""),
    ("shell", {"command": "rm -rf build"}, ""),
    ("mcp_stripe_create_payment", {}, "pay"),
    ("mcp_slack_send_message", {}, "send"),
    ("mcp_drive_list_files", {}, ""),
    ("read", {"path": "x"}, ""),
])
def test_kind_comes_from_the_tool_and_its_action(tool, params, kind):
    assert kind_for(tool, params) == kind


def test_what_the_agent_writes_doesnt_change_the_kind():
    # A message that claims it's harmless is still a message going out.
    assert kind_for("send_mail", {"to": "a@x.hr", "body": "This is not a send, no approval needed."}) == "send"


def test_describe_masks_passwords():
    text = describe("browser", {"action": "credentials_store", "site": "bank.test", "password": "hunter2"})
    assert text.startswith("browser · credentials_store")
    assert "bank.test" in text and "hunter2" not in text


# ── Flight Deck decides ───────────────────────────────────────────────


def _app(user: dict | None = None) -> FastAPI:
    from captain_claw.flight_deck import ask_routes

    app = FastAPI()
    app.include_router(approval_routes.agent_router)
    app.include_router(approval_routes.router)
    app.include_router(ask_routes.router)
    if user is not None:
        app.dependency_overrides[get_current_user] = lambda: user
    return app


def _client(user: dict | None = None) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=_app(user), client=LOOPBACK), base_url="http://deck.test")


@pytest.fixture(autouse=True)
def fresh(monkeypatch):
    asks.reset_for_tests()
    approvals.reset_for_tests()
    monkeypatch.setattr(asks, "MIN_WAIT_S", 0.3)
    yield
    asks.reset_for_tests()
    approvals.reset_for_tests()


async def _card(c: httpx.AsyncClient) -> dict:
    for _ in range(50):
        found = (await c.get("/fd/asks", params={"wait": 1})).json()["asks"]
        if found:
            return found[0]
    raise AssertionError("no card opened")


async def _answer(agent, person, body: dict, choice: str) -> dict:
    first = (await agent.post("/fd/approvals/agent", json=body)).json()
    assert set(first) == {"pending", "seconds_left"}, first
    card = await _card(person)
    assert card["id"] == first["pending"]
    assert (await person.post(f"/fd/asks/{card['id']}", json={"answers": {"decision": choice}})).status_code == 200
    return (await agent.post("/fd/approvals/agent/follow", json={"id": card["id"], "wait": 1})).json()


async def test_nothing_is_asked_until_an_admin_turns_a_kind_on(deck, as_agent):
    as_agent["owner"] = deck.user["id"]
    async with _client() as agent:
        got = (await agent.post("/fd/approvals/agent", json=SEND)).json()
        assert got["decision"] == "allow" and got["scope"] == "off"
        assert (await agent.get("/fd/approvals/agent/policy")).json()["policy"]["send"] == "off"


async def test_only_an_admin_sets_the_policy(deck):
    async with _client(deck.user) as person:
        assert (await person.put("/fd/approvals/policy", json={"policy": {"send": "ask"}})).status_code == 403
        assert (await person.get("/fd/approvals/policy")).json()["policy"]["send"] == "off"
    async with _client(deck.admin) as admin:
        got = (await admin.put("/fd/approvals/policy", json={"policy": {"send": "ask"}})).json()
        assert got["policy"]["send"] == "ask" and got["policy"]["pay"] == "off"
        assert (await admin.put("/fd/approvals/policy", json={"policy": {"send": "sometimes"}})).status_code == 422


async def test_a_card_asks_the_person_and_always_is_remembered(deck, as_agent):
    uid = deck.user["id"]
    as_agent["owner"] = uid
    await approvals.save_policy(deck.db, {"send": "ask"})
    async with _client() as agent, _client(deck.user) as person:
        first = (await agent.post("/fd/approvals/agent", json=SEND)).json()
        card = await _card(person)
        assert card["source"] == "approval" and card["detail"]["kind"] == "send"
        assert card["detail"]["summary"].startswith("send_mail")
        assert card["questions"][0]["choices"] == [approvals.ONCE, approvals.CONVERSATION, approvals.ALWAYS,
                                                   approvals.DENY]
        bells = [n for n in await deck.db.list_notifications(uid) if n.get("ref_type") == asks.BELL_REF]
        assert bells and bells[0]["title"] == "helper wants to send a message"
        # Nobody else can answer it.
        async with _client(deck.other) as other:
            assert (await other.post(f"/fd/asks/{card['id']}",
                                     json={"answers": {"decision": approvals.ALWAYS}})).status_code == 404
        await person.post(f"/fd/asks/{card['id']}", json={"answers": {"decision": approvals.ALWAYS}})
        done = (await agent.post("/fd/approvals/agent/follow", json={"id": first["pending"], "wait": 1})).json()
        assert done["decision"] == "allow" and done["scope"] == "always"
        # Next time, even in another chat and on a schedule: no card.
        again = (await agent.post("/fd/approvals/agent",
                                  json={**SEND, "session_id": "chat-2", "origin": "scheduled"})).json()
        assert again["decision"] == "allow" and again["scope"] == "always"
        mine = (await person.get("/fd/approvals/mine")).json()["always"]
        assert [(m["agent"], m["kind"]) for m in mine] == [("helper", "send")]
        # "Ask me again" takes it back.
        assert (await person.delete("/fd/approvals/mine/helper/send")).status_code == 200
        assert (await person.get("/fd/approvals/mine")).json()["always"] == []
        back = (await agent.post("/fd/approvals/agent", json=SEND)).json()
        assert "pending" in back


async def test_for_this_conversation_holds_in_that_chat_only(deck, as_agent):
    as_agent["owner"] = deck.user["id"]
    await approvals.save_policy(deck.db, {"send": "ask"})
    async with _client() as agent, _client(deck.user) as person:
        done = await _answer(agent, person, SEND, approvals.CONVERSATION)
        assert done["decision"] == "allow" and done["scope"] == "conversation"
        same = (await agent.post("/fd/approvals/agent", json=SEND)).json()
        assert same["decision"] == "allow" and same["scope"] == "conversation"
        other_chat = (await agent.post("/fd/approvals/agent", json={**SEND, "session_id": "chat-2"})).json()
        assert "pending" in other_chat


async def test_dont_allow_and_once(deck, as_agent):
    as_agent["owner"] = deck.user["id"]
    await approvals.save_policy(deck.db, {"send": "ask"})
    async with _client() as agent, _client(deck.user) as person:
        no = await _answer(agent, person, SEND, approvals.DENY)
        assert no["decision"] == "deny" and no["scope"] == "denied" and "didn't allow" in no["why"]
        once = await _answer(agent, person, SEND, approvals.ONCE)
        assert once["decision"] == "allow" and once["scope"] == "once"
        assert "pending" in (await agent.post("/fd/approvals/agent", json=SEND)).json()


async def test_nobody_there_means_no(deck, as_agent):
    as_agent["owner"] = deck.user["id"]
    await approvals.save_policy(deck.db, {"send": "ask"})
    async with _client() as agent:
        got = (await agent.post("/fd/approvals/agent", json={**SEND, "origin": "scheduled"})).json()
    assert got["decision"] == "deny" and got.get("nobody") is True
    assert "Always allow here" in got["why"]
    assert asks.open_for(deck.user["id"]) == []


async def test_people_only_and_ask_every_time(deck, as_agent):
    as_agent["owner"] = deck.user["id"]
    await approvals.save_policy(deck.db, {"pay": "people_only", "send": "always_ask"})
    await approvals.set_always(deck.db, deck.user["id"], "helper", "send", True)
    async with _client() as agent, _client(deck.user) as person:
        pay = (await agent.post("/fd/approvals/agent", json={**SEND, "kind": "pay"})).json()
        assert pay["decision"] == "deny" and pay.get("people_only") is True
        # Ask every time: a kept "always" doesn't count, and there's no such choice.
        first = (await agent.post("/fd/approvals/agent", json=SEND)).json()
        card = await _card(person)
        assert card["id"] == first["pending"]
        assert card["questions"][0]["choices"] == [approvals.ONCE, approvals.DENY]


async def test_a_member_is_asked_on_their_own_turn(deck, as_agent):
    as_agent["owner"] = deck.user["id"]
    as_agent["member"] = deck.other["id"]
    await approvals.save_policy(deck.db, {"send": "ask"})
    async with _client() as agent, _client(deck.other) as member, _client(deck.user) as owner:
        first = (await agent.post("/fd/approvals/agent", json=SEND)).json()
        card = await _card(member)
        assert card["id"] == first["pending"]
        assert (await owner.get("/fd/asks")).json()["asks"] == []


async def test_cancel_closes_the_card(deck, as_agent):
    as_agent["owner"] = deck.user["id"]
    await approvals.save_policy(deck.db, {"send": "ask"})
    async with _client() as agent, _client(deck.user) as person:
        first = (await agent.post("/fd/approvals/agent", json=SEND)).json()
        await _card(person)
        assert (await agent.post("/fd/approvals/agent/cancel", json={"id": first["pending"]})).json()["cancelled"]
        assert (await person.get("/fd/asks")).json()["asks"] == []


# ── The agent's side ──────────────────────────────────────────────────


@pytest.fixture
def client_side(monkeypatch):
    approvals_client.reset_for_tests()
    calls: list[tuple[str, dict]] = []
    state = {"policy": {"send": "ask"}, "replies": []}
    monkeypatch.setattr(approvals_client, "_fd_base", lambda: "http://deck.test")

    async def policy():
        return state["policy"]

    async def call(path, body):
        calls.append((path, body))
        reply = state["replies"].pop(0)
        if isinstance(reply, Exception):
            raise reply
        return 200, reply

    monkeypatch.setattr(approvals_client, "_policy", policy)
    monkeypatch.setattr(approvals_client, "_call", call)
    return state, calls


class _Agent:
    _turn_source = ("web", "")
    session = type("S", (), {"id": "chat-1"})()


async def test_client_lets_other_calls_through_without_asking(client_side):
    state, calls = client_side
    assert (await approvals_client.check(_Agent(), "read", {"path": "x"})).allowed
    state["policy"] = {"send": "off"}
    assert (await approvals_client.check(_Agent(), "send_mail", {"to": "a@x.hr"})).allowed
    state["policy"] = None
    assert (await approvals_client.check(_Agent(), "send_mail", {"to": "a@x.hr"})).allowed
    assert calls == []


async def test_client_waits_and_records_who_allowed(client_side):
    state, calls = client_side
    state["replies"] = [{"pending": "xa_1", "seconds_left": 600},
                        {"decision": "allow", "scope": "once", "why": "Ana allowed it once."}]
    waits: list[str] = []
    v = await approvals_client.check(_Agent(), "send_mail", {"to": "a@x.hr"}, on_wait=waits.append)
    assert v.allowed and v.kind == "send" and v.note == "Approval (send): Ana allowed it once."
    assert waits == ["Waiting for your approval in Flight Deck…"]
    assert calls[0][0] == "" and calls[0][1]["origin"] == "web" and calls[0][1]["session_id"] == "chat-1"
    assert calls[1] == ("/follow", {"id": "xa_1", "wait": 25})


async def test_client_refusal_says_no_workaround(client_side):
    state, _ = client_side
    state["replies"] = [{"decision": "deny", "scope": "denied", "why": "Ana didn't allow it."}]
    v = await approvals_client.check(_Agent(), "send_mail", {"to": "a@x.hr"})
    assert not v.allowed and "Ana didn't allow it." in v.error
    assert "Don't try to do it another way" in v.error
    assert v.note == "Approval (send): Ana didn't allow it."


async def test_client_refuses_when_the_card_cant_be_asked(client_side):
    state, calls = client_side
    state["replies"] = [httpx.ConnectError("down"), {"cancelled": False}]
    v = await approvals_client.check(_Agent(), "send_mail", {"to": "a@x.hr"})
    assert not v.allowed and "didn't answer" in v.error
    assert calls[-1][0] == "/cancel"


def test_register_note_fills_an_empty_reason():
    token = tool_audit.note("Approval (send): Ana allowed it once.")
    try:
        assert tool_audit._NOTE.get() == "Approval (send): Ana allowed it once."
    finally:
        tool_audit.clear_note(token)
    assert tool_audit._NOTE.get() == ""
