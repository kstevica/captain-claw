# ruff: noqa: F811 — `deck` and `as_agent` are Crew's route fixtures, imported from test_crew_routes
"""Custom tools, Phase 5 on the deck: an agent's proposal waits for its owner,
who builds it (a Crew task as that agent) or dismiss it. One bell per owner."""

from __future__ import annotations

import httpx
import pytest
from fastapi import FastAPI

from captain_claw.flight_deck import admin_routes, crew_routes, crew_store, speaker_grants
from captain_claw.flight_deck import external_tool_proposals as tpr
from captain_claw.flight_deck import external_tools as xt
from captain_claw.flight_deck import external_tools_routes as xr
from captain_claw.flight_deck.auth import get_current_user
from captain_claw.tools.custom_tools import CustomToolsTool

from .test_crew_routes import LOOPBACK, _approve, as_agent, deck  # noqa: F401

WHAT = "Read a CSV of bank lines and return the total per category. Input: the file. Example: two rows of 5 give 10."


def _client(user: dict | None) -> httpx.AsyncClient:
    app = FastAPI()
    app.include_router(xr.agent_router)
    app.include_router(xr.catalog_router)
    app.include_router(tpr.agent_router)
    app.include_router(tpr.router)
    app.include_router(xr.router)
    app.include_router(crew_routes.router)
    app.include_router(admin_routes.router)
    if user is not None:
        app.dependency_overrides[get_current_user] = lambda: user
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app, client=LOOPBACK), base_url="http://deck.test")


@pytest.fixture
async def on(deck, as_agent):
    xt.reset_for_tests()
    async with _client(deck.admin) as c:
        await c.put("/fd/admin/config", json={"external_tools_enabled": True})
    await _approve(deck, deck.user)
    async with _client(deck.user) as c:
        await c.put("/fd/crew/logins/claude", json={"secret": "sk-ant-oat01-abc"})
    as_agent["owner"] = deck.user["id"]
    yield deck
    xt.reset_for_tests()


async def _propose(name="bank-totals", what=WHAT, **over):
    async with _client(None) as agent:
        return await agent.post("/fd/external-tools/agent/propose",
                                json={"name": name, "title": "Bank totals", "what": what, "why": "Sums slip.", **over})


async def _bells(deck):
    return [n for n in await deck.db.list_notifications(deck.user["id"]) if n.get("ref_type") == tpr.BELL_REF]


async def test_a_proposal_waits_for_its_owner_with_one_bell(on):
    deck = on
    first = await _propose()
    assert first.status_code == 200 and first.json()["kept"] is True
    assert (await _propose()).json() == {"ok": True, "kept": False, "why": "waiting"}
    second = await _propose("other-tool", "Turn a list of addresses into a CSV with street, city and postcode columns.")
    assert second.json()["kept"] is True
    assert len(await _bells(deck)) == 1  # one unread bell, however many wait
    async with _client(deck.user) as c:
        waiting = (await c.get("/fd/external-tools/proposals")).json()["waiting"]
    assert {p["name"] for p in waiting} == {"bank-totals", "other-tool"}
    assert waiting[0]["agent"] == "helper" and "said" in waiting[0]
    assert (await _propose("agent")).status_code == 400
    assert (await _propose("x-tool", "short")).status_code == 422


async def test_building_one_starts_crew_as_the_agent_that_proposed_it(on):
    deck = on
    pid = (await _propose()).json()["id"]
    async with _client(deck.user) as c:
        built = await c.post(f"/fd/external-tools/proposals/{pid}/build", json={"harness": "claude", "title": "Bank totals"})
        assert built.status_code == 200, built.text
        body = built.json()
        assert body["proposal"]["state"] == "built" and body["tool"]["slug"] == "bank-totals"
        task = crew_store.get_store().get(deck.user["id"], body["task"]["id"])
        assert task["repo_kind"] == "tool" and task["agent_slug"] == "helper" and WHAT in task["brief"]
        assert (await c.post(f"/fd/external-tools/proposals/{pid}/build", json={"harness": "claude"})).status_code == 409
        assert (await c.get("/fd/external-tools/proposals")).json()["waiting"] == []
    # Nothing waits: the bell is settled.
    assert all(b.get("resolved") for b in await _bells(deck))
    # A tool by that name exists now, so the same idea isn't proposed again.
    assert (await _propose()).json()["why"] == "exists"


async def test_a_dismissed_idea_isnt_proposed_again(on):
    deck = on
    pid = (await _propose()).json()["id"]
    async with _client(deck.user) as c:
        assert (await c.post(f"/fd/external-tools/proposals/{pid}/dismiss")).json()["proposal"]["state"] == "dismissed"
    assert (await _propose("bank-sums", WHAT)).json()["why"] == "dismissed"


async def test_only_an_owner_who_can_build_gets_proposals(on, as_agent, monkeypatch):
    deck = on
    as_agent["owner"] = deck.other["id"]  # not approved for Crew
    assert (await _propose()).status_code == 403
    as_agent["owner"] = deck.user["id"]
    monkeypatch.setattr(speaker_grants, "member_request", lambda request: True)

    async def member(request, **_):
        return speaker_grants.ActingMember(user_id=deck.other["id"], owner=deck.user["id"], agent_ref="process:helper",
                                           lane="A", turn="t", slug="helper", name="Helper")

    monkeypatch.setattr(speaker_grants, "acting_member", member)
    assert (await _propose()).status_code == 403


async def test_the_tool_proposes_and_says_why_not(monkeypatch):
    from captain_claw import custom_tools_client

    answers = [(200, {"ok": True, "kept": True, "id": "p1"}), (200, {"ok": True, "kept": False, "why": "dismissed"})]
    sent = []

    async def call(method, path, body=None):
        sent.append((path, body))
        return answers.pop(0)

    monkeypatch.setattr(custom_tools_client, "_fd_base", lambda: "http://deck.test")
    monkeypatch.setattr(custom_tools_client, "call", call)
    tool = CustomToolsTool()
    assert not (await tool.execute(action="propose", tool="bank-totals", what="short")).success
    made = await tool.execute(action="propose", tool="bank-totals", what=WHAT, why="Sums slip.")
    assert made.success and "Custom tools page" in made.content and sent[0][1]["source"] == "asked"
    again = await tool.execute(action="propose", tool="bank-totals", what=WHAT)
    assert "dismissed a proposal like it" in again.content
