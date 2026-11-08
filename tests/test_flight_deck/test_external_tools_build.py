# ruff: noqa: F811 — `deck` and `as_agent` are Crew's route fixtures, imported from test_crew_routes
"""Custom tools, Phase 2 over HTTP: a person or their agent asks for a tool to be
made or changed, which makes its repository and starts a Crew task there."""

from __future__ import annotations

import httpx
import pytest
from fastapi import FastAPI

from captain_claw.flight_deck import admin_routes, crew_routes, crew_store, speaker_grants
from captain_claw.flight_deck import external_tools as xt
from captain_claw.flight_deck import external_tools_routes as xr
from captain_claw.flight_deck.auth import get_current_user
from captain_claw.tools.custom_tools import CustomToolsTool

from .test_crew_routes import LOOPBACK, _approve, as_agent, deck  # noqa: F401

WHAT = "Read a CSV file and add up one column. Input: the file and the column's name. Returns rows and total."


def _client(user: dict | None) -> httpx.AsyncClient:
    app = FastAPI()
    app.include_router(xr.agent_router)
    app.include_router(xr.router)
    app.include_router(crew_routes.router)
    app.include_router(admin_routes.router)
    if user is not None:
        app.dependency_overrides[get_current_user] = lambda: user
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app, client=LOOPBACK), base_url="http://deck.test")


@pytest.fixture
async def on(deck):
    xt.reset_for_tests()
    async with _client(deck.admin) as c:
        assert (await c.put("/fd/admin/config", json={"external_tools_enabled": True})).status_code == 200
    yield deck
    xt.reset_for_tests()


async def _ready(deck, user) -> None:
    await _approve(deck, user)
    async with _client(user) as c:
        await c.put("/fd/crew/logins/claude", json={"secret": "sk-ant-oat01-abc"})


async def test_a_person_has_a_tool_built(on):
    deck = on
    async with _client(deck.user) as c:
        resp = await c.post("/fd/external-tools/build", json={"slug": "csv-total", "what": WHAT, "harness": "claude"})
        assert resp.status_code == 403 and "Crew" in resp.json()["detail"]
    await _ready(deck, deck.user)
    async with _client(deck.user) as c:
        resp = await c.post("/fd/external-tools/build", json={"slug": "csv-total", "title": "Total a CSV column",
                                                              "what": WHAT, "harness": "claude"})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["tool"]["slug"] == "csv-total" and body["tool"]["title"] == "Total a CSV column"
        assert body["tool"]["description"].startswith("Read a CSV file and add up one column.")
        task = crew_store.get_store().get(deck.user["id"], body["task"]["id"])
        assert task["repo_kind"] == "tool" and task["repo"] == str(xt.repo_dir(deck.user["id"], "csv-total"))
        assert "Build the custom tool “csv-total”" in task["brief"] and WHAT in task["brief"]
        again = await c.post("/fd/external-tools/build", json={"slug": "csv-total", "what": WHAT, "harness": "claude"})
        assert again.status_code == 409
        short = await c.post("/fd/external-tools/build", json={"slug": "other", "what": "x", "harness": "claude"})
        assert short.status_code == 422
        improved = await c.post("/fd/external-tools/csv-total/improve", json={"what": "Also return the average.",
                                                                              "harness": "claude"})
        assert improved.status_code == 200, improved.text
        assert (await c.post("/fd/external-tools/nope/improve", json={"what": "Also return the average.",
                                                                      "harness": "claude"})).status_code == 404


async def test_a_task_that_cant_start_leaves_no_tool_behind(on):
    deck = on
    await _ready(deck, deck.user)
    async with _client(deck.user) as c:
        resp = await c.post("/fd/external-tools/build", json={"slug": "lost", "what": WHAT, "harness": "codex"})
        assert resp.status_code in (400, 403), resp.text
    assert not xt.exists(deck.user["id"], "lost")


async def test_an_agent_has_a_tool_built_for_its_owner(on, as_agent, monkeypatch):
    deck = on
    await _ready(deck, deck.user)
    as_agent["owner"] = deck.user["id"]
    async with _client(None) as agent:
        resp = await agent.post("/fd/external-tools/agent/build", json={"tool": "csv-total", "what": WHAT, "harness": "claude"})
        assert resp.status_code == 200, resp.text
        task = crew_store.get_store().get(deck.user["id"], resp.json()["task"]["id"])
        assert task["agent_slug"] == "helper" and task["repo_kind"] == "tool"
        resp = await agent.post("/fd/external-tools/agent/improve", json={"tool": "csv-total", "what": "Round to cents.",
                                                                          "harness": "claude"})
        assert resp.status_code == 200, resp.text
        monkeypatch.setattr(speaker_grants, "member_request", lambda request: True)

        async def member(request, **_):
            return speaker_grants.ActingMember(user_id=deck.other["id"], owner=deck.user["id"], agent_ref="process:helper",
                                               lane="A", turn="t", slug="helper", name="Helper")

        monkeypatch.setattr(speaker_grants, "acting_member", member)
        resp = await agent.post("/fd/external-tools/agent/build", json={"tool": "x-tool", "what": WHAT, "harness": "claude"})
        assert resp.status_code == 403 and "owner" in resp.json()["detail"]


async def test_an_unapproved_owners_agent_cant_build(on, as_agent):
    as_agent["owner"] = on.user["id"]
    async with _client(None) as agent:
        resp = await agent.post("/fd/external-tools/agent/build", json={"tool": "t-one", "what": WHAT, "harness": "claude"})
        assert resp.status_code == 403 and "Crew" in resp.json()["detail"]
    assert not xt.exists(on.user["id"], "t-one")


async def test_the_tool_asks_for_what_a_build_needs(monkeypatch):
    from captain_claw import custom_tools_client

    sent = []

    async def call(method, path, body=None):
        sent.append((method, path, body))
        return 200, {"tool": {"slug": "csv-total"}, "task": {"id": "crw_1", "harness_label": "Claude Code"}}

    monkeypatch.setattr(custom_tools_client, "_fd_base", lambda: "http://deck.test")
    monkeypatch.setattr(custom_tools_client, "call", call)
    tool = CustomToolsTool()
    missing = await tool.execute(action="build", tool="csv-total")
    assert not missing.success and "harness" in missing.error and sent == []
    made = await tool.execute(action="build", tool="csv-total", what=WHAT, harness="claude", title="Total")
    assert made.success and "crw_1" in made.content and "request_merge task_id=crw_1" in made.content
    assert sent[0][:2] == ("POST", "build") and sent[0][2]["what"] == WHAT and sent[0][2]["origin_platform"] == "web"
    monkeypatch.setattr(custom_tools_client, "member_call", lambda: True)
    refused = await tool.execute(action="improve", tool="csv-total", what="x", harness="claude")
    assert not refused.success and "owner" in refused.error


def test_the_builder_is_told_the_limits_publishing_checks():
    # Seen on a demo deck: the coding agent wrote a description of over 600 characters into
    # tool.json, and publishing refused it ("'description' must be 1–600 characters").
    brief = xr.build_brief("working-days", "Working days", "Count working days between two dates.")
    assert "600" in brief and "80" in brief
    readme = xt._starter("working-days", "Working days", "Counts working days.")["README.md"]
    assert "1–600" in readme and "1–80" in readme
