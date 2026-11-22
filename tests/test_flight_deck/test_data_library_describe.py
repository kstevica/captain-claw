"""A data table's description is asked with thinking off.

A thinking model counts its thinking against max_tokens: on the description's
small budget it can think through all of it and never write the JSON. When
an answer is cut off anyway, the error says so rather than that the model
didn't answer with JSON.
"""

from __future__ import annotations

import json

import pytest

from captain_claw.flight_deck import data_library as dl
from captain_claw.flight_deck.file_library import ModelReply
from captain_claw.llm import reasoning

TIER = {"provider": "openai", "model": "Qwen3.8-flash-next", "base_url": "http://127.0.0.1:8080/v1",
        "api_key": "local"}
REPLY = {"title": "Open deals", "summary": "One row per deal.", "category": "leads",
         "tags": ["sales"], "insights": ["3 deals"], "columns": {"name": "Deal name"}}


def _table() -> dict:
    return {"table_name": "deals", "agent_slug": "desk-bot", "row_count": 3,
            "updated_at": "2026-11-22T10:00:00", "columns": [{"name": "name"}],
            "profile": [{"name": "name", "type": "TEXT", "kind": "text", "filled": 3, "distinct": 3}],
            "sample": [{"name": "Alpha"}, {"name": "Beta"}, {"name": "Gamma"}]}


class _Model:
    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.replies: list[ModelReply] = []


@pytest.fixture
def model(monkeypatch: pytest.MonkeyPatch) -> _Model:
    state = _Model()

    async def model_reply(tier, system, user, owner_id, max_tokens=700, request_timeout=None, images=None):
        state.calls.append({"effort": reasoning.requested_effort(), "max_tokens": max_tokens, "user": user})
        return state.replies.pop(0)

    monkeypatch.setattr(dl, "model_reply", model_reply)
    return state


async def test_a_table_is_described_with_thinking_off(model: _Model):
    model.replies.append(ModelReply(json.dumps(REPLY), 900, 700, "stop"))
    out = await dl.summarize_table(_table(), TIER, "u1", "Desk")

    [call] = model.calls
    assert call["effort"] == reasoning.QUIET and call["max_tokens"] == dl._MAX_TOKENS
    assert "Table: deals" in call["user"]
    assert reasoning.requested_effort() is None
    assert out["status"] == "done" and out["title"] == "Open deals" and out["tokens"] == 900
    assert out["category"] == "leads" and out["column_notes"] == {"name": "Deal name"}


async def test_an_answer_cut_off_before_its_json_says_so(model: _Model):
    model.replies.append(ModelReply("", 4400, 3200, "length", "Let me look at the columns first…"))
    with pytest.raises(ValueError, match="cut off at 1200 tokens"):
        await dl.summarize_table(_table(), TIER, "u1", "Desk")


async def test_an_answer_without_json_that_ended_by_itself_keeps_the_plain_error(model: _Model):
    model.replies.append(ModelReply("I can't describe this table.", 800, 700, "stop"))
    with pytest.raises(ValueError, match="did not answer with JSON"):
        await dl.summarize_table(_table(), TIER, "u1", "Desk")


async def test_an_empty_table_asks_no_model(model: _Model):
    out = await dl.summarize_table({**_table(), "row_count": 0}, TIER, "u1", "Desk")
    assert out["status"] == "skipped" and model.calls == []
