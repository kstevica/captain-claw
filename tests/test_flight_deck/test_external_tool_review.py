# ruff: noqa: F811 — `deck` and `as_agent` are Crew's route fixtures, imported from test_crew_routes
"""Custom tools, New tool step two: the person's model checks a spec with what
Captain is and the person's tools, and answers with a fuller spec and why.
Nothing is kept; a check that fails still lets the person build theirs."""

from __future__ import annotations

import json

import httpx
import pytest
from fastapi import FastAPI

from captain_claw.flight_deck import admin_routes, crew_routes, file_library
from captain_claw.flight_deck import external_tool_review as xtr
from captain_claw.flight_deck import external_tools as xt
from captain_claw.flight_deck import external_tools_routes as xr
from captain_claw.flight_deck.auth import get_current_user
from captain_claw.flight_deck.file_library import ModelReply

from .test_crew_routes import LOOPBACK, _approve, as_agent, deck  # noqa: F401

WHAT = "Read a CSV file and add up one column. Input: the file and the column's name. Returns rows and total."
TIER = {"provider": "anthropic", "model": "claude-sonnet-5-5"}

ANSWER = {
    "spec": WHAT + "\n\nInput: file (a file field), column (string).\nExample: 5, 7.5 and 2.5 give rows 3, total 15.",
    "title": "Total a CSV column",
    "suggestions": [{"change": "Made the file a file field.", "why": "Agents pass vfs paths; Flight Deck turns them into files."},
                    {"change": "", "why": "an empty change is dropped"},
                    "not an object"],
    "overlap": "",
}


def _client(user: dict | None) -> httpx.AsyncClient:
    app = FastAPI()
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


@pytest.fixture
def model(monkeypatch):
    """The person's model: records what it was asked and answers ``reply``."""
    seen: dict = {"reply": "```json\n" + json.dumps(ANSWER) + "\n```", "tier": TIER}

    async def resolve(uid, prefer=("fast", "balanced")):
        seen["prefer"] = prefer
        return seen["tier"]

    async def reply(tier, system, user, owner_id, **kw):
        seen.update(system=system, user=user, owner=owner_id, kw=kw)
        if isinstance(seen["reply"], Exception):
            raise seen["reply"]
        return ModelReply(seen["reply"], 100, finish_reason=seen.get("finish", "stop"))

    monkeypatch.setattr(file_library, "_resolve_tier", resolve)
    monkeypatch.setattr(file_library, "model_reply", reply)
    return seen


def test_the_check_knows_how_custom_tools_run_and_what_agents_already_do():
    assert xt.WORKER_CONTRACT in xtr.SYSTEM
    assert "$XT_OUT" in xtr.SYSTEM and "nothing is kept between calls" in xtr.SYSTEM
    assert str(xt.MAX_TIMEOUT_S) in xtr.SYSTEM and "vfs:custom-tools/" in xtr.SYSTEM
    assert "Excel" in xtr.SYSTEM and "web" in xtr.SYSTEM.lower()


def test_the_prompt_names_the_persons_tools_and_quotes_their_spec():
    own = [{"slug": "csv-total", "title": "Total a CSV", "description": "Adds a column."}]
    added = [{"slug": "pdf-split", "title": "Split a PDF", "description": "One file a page.",
              "owner_person": {"name": "Ana"}}]
    text = xtr.prompt("csv-sum", "", WHAT, own, added)
    assert "- csv-total: Total a CSV. Adds a column." in text
    assert "- pdf-split: Split a PDF (offered by Ana)." in text
    assert "The new tool's name: csv-sum" in text and "Its title: (none yet)" in text
    assert f"<<<\n{WHAT}\n>>>" in text
    assert "(none yet)" in xtr.tools_block([], [])


def test_the_answer_is_clipped_and_cleaned():
    out = xtr.parse(json.dumps(ANSWER), what=WHAT, title="")
    assert out["title"] == "Total a CSV column" and out["overlap"] == "" and out["unchanged"] is False
    assert out["suggestions"] == [ANSWER["suggestions"][0]]
    many = {**ANSWER, "spec": "x" * 9000, "suggestions": [{"change": f"c{i}", "why": "w"} for i in range(30)]}
    out = xtr.parse(json.dumps(many), what=WHAT, title="Mine")
    assert len(out["spec"]) == xtr.MAX_SPEC and len(out["suggestions"]) == xtr.MAX_SUGGESTIONS
    same = xtr.parse(json.dumps({"spec": WHAT, "title": "", "suggestions": []}), what=WHAT + "\n", title="Mine")
    assert same["unchanged"] is True and same["title"] == "Mine"
    with pytest.raises(ValueError):
        xtr.parse(json.dumps({"spec": "", "suggestions": []}), what=WHAT, title="")
    with pytest.raises(ValueError):
        xtr.parse("I think it is fine.", what=WHAT, title="")


async def test_a_builder_gets_a_checked_spec(on, model):
    deck = on
    async with _client(deck.user) as c:
        refused = await c.post("/fd/external-tools/review", json={"slug": "csv-sum", "what": WHAT})
        assert refused.status_code == 403 and "Crew" in refused.json()["detail"]
    await _ready(deck, deck.user)
    async with _client(deck.user) as c:
        resp = await c.post("/fd/external-tools/review", json={"slug": "csv-sum", "title": "", "what": WHAT})
        assert resp.status_code == 200, resp.text
        body = resp.json()
    assert body["spec"] == ANSWER["spec"] and body["title"] == "Total a CSV column"
    assert body["model"] == "claude-sonnet-5-5" and len(body["suggestions"]) == 1
    assert model["prefer"] == xtr.TIERS and model["owner"] == deck.user["id"] and model["system"] == xtr.SYSTEM
    assert WHAT in model["user"] and "The new tool's name: csv-sum" in model["user"]
    # Nothing was made: the check keeps nothing.
    assert not xt.exists(deck.user["id"], "csv-sum")


async def test_the_check_sees_the_persons_existing_tools_and_refuses_their_names(on, model):
    deck = on
    await _ready(deck, deck.user)
    async with _client(deck.user) as c:
        made = await c.post("/fd/external-tools/build", json={"slug": "csv-total", "title": "Total a CSV column",
                                                              "what": WHAT, "harness": "claude"})
        assert made.status_code == 200, made.text
        assert (await c.post("/fd/external-tools/review", json={"slug": "csv-total", "what": WHAT})).status_code == 409
        assert (await c.post("/fd/external-tools/review", json={"slug": "review", "what": WHAT})).status_code == 400
        assert (await c.post("/fd/external-tools/review", json={"slug": "Bad Name", "what": WHAT})).status_code == 400
        assert (await c.post("/fd/external-tools/review", json={"slug": "csv-avg", "what": "short"})).status_code == 422
        ok = await c.post("/fd/external-tools/review", json={"slug": "csv-avg", "what": WHAT})
        assert ok.status_code == 200, ok.text
    assert "- csv-total: Total a CSV column." in model["user"]


async def test_a_check_that_fails_says_why_and_the_spec_can_still_be_built(on, model):
    deck = on
    await _ready(deck, deck.user)
    async with _client(deck.user) as c:
        model["tier"] = None
        none = (await c.post("/fd/external-tools/review", json={"slug": "csv-sum", "what": WHAT})).json()
        assert "no model" in none["error"] and "spec" not in none
        model["tier"] = TIER
        model["reply"] = RuntimeError("429 rate limited")
        failed = (await c.post("/fd/external-tools/review", json={"slug": "csv-sum", "what": WHAT})).json()
        assert "claude-sonnet-5-5 couldn't check it (429 rate limited)" in failed["error"]
        model["reply"] = '{"spec": "Read a CSV'
        model["finish"] = "max_tokens"
        cut = (await c.post("/fd/external-tools/review", json={"slug": "csv-sum", "what": WHAT})).json()
        assert "ran out of room" in cut["error"]
        model["reply"], model["finish"] = "Looks fine to me.", "stop"
        prose = (await c.post("/fd/external-tools/review", json={"slug": "csv-sum", "what": WHAT})).json()
        assert "didn't answer with a spec" in prose["error"]
        built = await c.post("/fd/external-tools/build", json={"slug": "csv-sum", "what": WHAT, "harness": "claude"})
        assert built.status_code == 200, built.text


async def test_the_check_is_off_with_custom_tools(deck, model):
    async with _client(deck.user) as c:
        assert (await c.post("/fd/external-tools/review", json={"slug": "csv-sum", "what": WHAT})).status_code == 403
