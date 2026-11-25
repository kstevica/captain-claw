"""Telling the person when their background model stops working (side_call_health.py).

Pinned here:

* why a call failed, in plain words, and whether it fixes itself: a refused
  key, a missing model, no credit are setup problems; a timeout, a busy or
  erroring provider, a server that doesn't answer, an empty reply are not —
  and the provider's own error text (which can carry a key) never shows;
* a setup problem tells the person at once, anything else after 3 in a row;
  one Line item under Needs you (not to phones), naming the model and what
  waits on it — not repeated; a call that gets through clears it and answers
  the item, and a later breakage is a new item; a cancelled call and a call
  without a feature count for nothing; people and models are kept apart;
* both ways the deck calls a person's model say how it went: the File
  library's ``model_reply`` and the owner-tier completion (with fallback, each
  model on its own) — and ``GET /fd/llm/side-call-health`` lists what's wrong.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI

from captain_claw.exceptions import LLMAPIError
from captain_claw.flight_deck import auth as fd_auth
from captain_claw.flight_deck import file_library, line, llm_routes
from captain_claw.flight_deck import side_call_health as sch
from captain_claw.flight_deck.auth import get_current_user, set_auth_db
from captain_claw.flight_deck.db import FlightDeckDB

ALICE, BOB = "user-alice", "user-bob"


@pytest.fixture
async def db(tmp_path, monkeypatch):
    monkeypatch.setattr(line, "_last_prune", 0.0)
    sch.reset_for_tests()
    prev = fd_auth._db
    d = FlightDeckDB(tmp_path / "fd.db")
    await d.init()
    set_auth_db(d)
    try:
        yield d
    finally:
        sch.reset_for_tests()
        await d.close()
        fd_auth._db = prev


async def _notices(db, user=ALICE):
    return [i for i in await db.list_line_items(user) if i["ref_type"] == sch.REF_TYPE]


def test_why_a_call_failed_in_plain_words():
    class LiteLLMishError(Exception):
        status_code = 401

    def wrapped() -> Exception:
        try:
            raise LiteLLMishError("AuthenticationError: Incorrect API key provided: sk-proj-abc123")
        except LiteLLMishError as inner:
            try:
                raise RuntimeError("provider call failed") from inner
            except RuntimeError as outer:
                return outer

    cases = [
        (LLMAPIError("Incorrect API key provided: sk-abc123", status_code=401), sch.CONFIG, "refused the key"),
        (Exception("OpenRouter: Missing Authentication header"), sch.CONFIG, "refused the key"),
        (wrapped(), sch.CONFIG, "refused the key"),
        (LLMAPIError("You exceeded your current quota, check your plan and billing", status_code=429),
         sch.CONFIG, "out of credit"),
        (LLMAPIError("payment required", status_code=402), sch.CONFIG, "out of credit"),
        (Exception("model 'gpt-9' not found"), sch.CONFIG, "doesn't have that model"),
        (LLMAPIError("Rate limit reached for requests", status_code=429), sch.RUNTIME, "busy or rate-limited"),
        (TimeoutError(), sch.RUNTIME, "too long"),
        (LLMAPIError("Internal server error", status_code=500), sch.RUNTIME, "answers with errors"),
        (ConnectionRefusedError("[Errno 61] Connection refused"), sch.RUNTIME, "doesn't answer"),
        (ValueError("something odd"), sch.RUNTIME, "keep failing"),
        (None, sch.RUNTIME, "answered with nothing"),
    ]
    for exc, kind, words in cases:
        got_kind, why = sch.classify(exc)
        assert (got_kind, words in why) == (kind, True), (exc, got_kind, why)
        assert "sk-" not in why and "abc123" not in why, "the provider's own text never shows"


async def test_a_setup_problem_tells_at_once_and_a_success_clears_it(db):
    key_refused = LLMAPIError("Incorrect API key provided", status_code=401)
    await sch.failed(ALICE, sch.FILE_DESCRIPTIONS, "openrouter", "small-model", "", key_refused)
    [item] = await _notices(db)
    assert item["needs_you"] == 1 and item["kind"] == "deck" and item["resolved_at"] == ""
    assert item["title"] == "Background work can't use small-model (openrouter)"
    assert "refused the key" in item["body"] and "File library descriptions" in item["body"]
    assert "Console → Models" in item["body"]
    # More failures, from more features: still one item; what waits on it grows.
    await sch.failed(ALICE, sch.SKILL_CARDS, "openrouter", "small-model", "", key_refused)
    await sch.failed(ALICE, sch.SKILL_CARDS, "openrouter", "small-model", "", key_refused)
    assert len(await _notices(db)) == 1
    [p] = sch.problems(ALICE)
    assert p["model"] == "small-model" and p["kind"] == sch.CONFIG and p["failures"] == 3
    assert p["waiting"] == [sch.SKILL_CARDS, sch.FILE_DESCRIPTIONS], "most recent first"
    assert sch.problems(BOB) == []

    await sch.ok(ALICE, sch.SKILL_CARDS, "openrouter", "small-model")
    [item] = await _notices(db)
    assert item["resolved_at"], "the item is answered once the model works again"
    assert sch.problems(ALICE) == []
    # A later breakage is a new item.
    await sch.failed(ALICE, sch.TASK_TRIAGE, "openrouter", "small-model", "", key_refused)
    items = await _notices(db)
    assert len(items) == 2 and len({i["ref_id"] for i in items}) == 2


async def test_anything_else_tells_after_three_in_a_row(db):
    for _ in range(2):
        await sch.failed(ALICE, sch.TASK_TRIAGE, "ollama", "qwen", "http://box:11434", TimeoutError())
    assert await _notices(db) == [] and sch.problems(ALICE) == []
    await sch.ok(ALICE, sch.TASK_TRIAGE, "ollama", "qwen", "http://box:11434")   # the run starts over
    for _ in range(2):
        await sch.failed(ALICE, sch.TASK_TRIAGE, "ollama", "qwen", "http://box:11434/", TimeoutError())
    assert await _notices(db) == []
    await sch.failed(ALICE, sch.TASK_TRIAGE, "ollama", "qwen", "http://box:11434", None)
    [item] = await _notices(db)
    assert "answered with nothing" in item["body"] and "task triage" in item["body"]


async def test_what_counts_for_nothing_and_what_is_kept_apart(db):
    refused = LLMAPIError("unauthorized", status_code=401)
    await sch.failed(ALICE, "", "openai", "m", "", refused)
    await sch.failed(ALICE, sch.TOOL_REVIEWS, "openai", "m", "", asyncio.CancelledError())
    assert await _notices(db) == [] and sch.problems(ALICE) == []
    await sch.failed(ALICE, sch.TOOL_REVIEWS, "openai", "m", "", refused)
    await sch.failed(BOB, sch.TOOL_REVIEWS, "openai", "m", "", TimeoutError())
    await sch.ok(ALICE, sch.TOOL_REVIEWS, "openai", "other-model")
    assert len(sch.problems(ALICE)) == 1 and sch.problems(BOB) == []
    assert await _notices(db, BOB) == []
    # Auth off: the person is "local", as on the Line.
    await sch.failed("", sch.FIRST_MATE, "openai", "m", "", refused)
    assert len(await _notices(db, "local")) == 1 and len(sch.problems("")) == 1


class _Provider:
    def __init__(self, reply="", error=None):
        self.reply, self.error = reply, error

    async def complete(self, messages, temperature=None, max_tokens=None, **_):
        if self.error is not None:
            raise self.error
        return SimpleNamespace(content=self.reply, usage={}, finish_reason="stop", reasoning_content="")


async def test_the_file_librarys_model_call_says_how_it_went(db, monkeypatch):
    import captain_claw.llm as llm

    made = {"p": _Provider(error=LLMAPIError("Incorrect API key", status_code=401))}
    monkeypatch.setattr(llm, "create_provider", lambda **kw: made["p"])
    tier = {"provider": "openai", "model": "mini", "api_key": "k"}
    with pytest.raises(LLMAPIError):
        await file_library.model_reply(tier, "s", "u", ALICE, feature=sch.DATA_DESCRIPTIONS)
    [item] = await _notices(db)
    assert "mini (openai)" in item["title"] and "Data library descriptions" in item["body"]
    made["p"] = _Provider(reply="A title")
    reply = await file_library.model_reply(tier, "s", "u", ALICE, feature=sch.DATA_DESCRIPTIONS)
    assert reply.text == "A title" and (await _notices(db))[0]["resolved_at"]
    # A call nobody named a feature for (one the person waits on) isn't counted.
    made["p"] = _Provider(error=LLMAPIError("Incorrect API key", status_code=401))
    with pytest.raises(LLMAPIError):
        await file_library.call_model(tier, "s", "u", ALICE)
    assert sch.problems(ALICE) == []


async def test_owner_tier_completions_say_how_each_model_went(db, monkeypatch):
    import captain_claw.llm as llm

    creds = {"balanced": {"provider": "openai", "model": "broken-model", "api_key": "k"},
             "reason": {"provider": "anthropic", "model": "good-model", "api_key": "k"}}

    async def _resolver(_db, _owner):
        return lambda tier: creds.get(tier, {})

    monkeypatch.setattr(llm_routes, "_creds_resolver", _resolver)
    monkeypatch.setattr(llm, "create_provider", lambda **kw: _Provider(
        error=LLMAPIError("invalid api key", status_code=401)) if kw["model"] == "broken-model"
        else _Provider(reply="SEND"))
    out = await llm_routes.complete_with_fallback(db, ALICE, ("balanced", "reason"), "s", "u", max_tokens=50,
                                                  temperature=0, feature=sch.EMAIL_AUTOREPLY)
    assert out.content == "SEND" and out.model == "good-model"
    [p] = sch.problems(ALICE)
    assert p["model"] == "broken-model" and p["waiting"] == [sch.EMAIL_AUTOREPLY], \
        "the broken tier is told about even though the other one answered"

    app = FastAPI()
    app.include_router(llm_routes.router)
    app.dependency_overrides[get_current_user] = lambda: {"id": ALICE}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://fd.test") as c:
        got = (await c.get("/fd/llm/side-call-health")).json()
    assert [x["model"] for x in got["problems"]] == ["broken-model"]
    assert "refused the key" in got["problems"][0]["why"]

    # The route a person calls themselves names no feature: nothing is counted.
    sch.reset_for_tests()
    with pytest.raises(llm_routes.OwnerTierError):
        await llm_routes.complete_for_owner(db, ALICE, "balanced", "s", "u", max_tokens=50, temperature=0)
    assert sch.problems(ALICE) == []


async def test_the_bell_shows_it_while_it_lasts(db, tmp_path, monkeypatch):
    """What the bell reads (``/fd/line``, Needs you, and its badge): the notice
    waits there, and stops waiting once the model works again."""
    from captain_claw.flight_deck import autonomy, line_routes

    monkeypatch.setattr(autonomy, "_STORE", autonomy.AutonomyStore(tmp_path / "autonomy.db"))
    monkeypatch.setattr(line_routes, "_line_agent", lambda user_id: ("", ""))
    monkeypatch.setattr(line, "user_agents", lambda user_id: [])
    app = FastAPI()
    app.include_router(line_routes.router)
    app.dependency_overrides[get_current_user] = lambda: {"id": ALICE}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://fd.test") as c:
        await sch.failed(ALICE, sch.SKILL_CARDS, "openai", "mini", "", LLMAPIError("bad key", status_code=401))
        assert (await c.get("/fd/line/unread")).json()["waiting"] == 1
        [item] = (await c.get("/fd/line", params={"needs_you": "true"})).json()["items"]
        assert item["title"] == "Background work can't use mini (openai)" and item["needs_you"]
        await sch.ok(ALICE, sch.SKILL_CARDS, "openai", "mini")
        assert (await c.get("/fd/line/unread")).json()["waiting"] == 0
