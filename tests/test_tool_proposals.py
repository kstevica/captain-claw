"""The agent proposes a custom tool when a job looked like code's: a long chain of
tool calls, or a mechanical job in the owner's words. Drafted by its own model,
handed to Flight Deck for the owner to build or dismiss
(captain_claw/tool_proposals.py). Most turns stop at the cheap gate."""

from __future__ import annotations

import types

import pytest

from captain_claw import config as config_mod
from captain_claw import tool_proposals as tp
from captain_claw.config import ToolProposalsConfig

DRAFT = ('{"name": "invoice-totals", "title": "Total invoices", "what": "Read every invoice PDF given (input: files) '
         'and return the supplier, date and total of each, plus the grand total. Example: two invoices of 10 and 5 '
         'give total 15.", "why": "Adding them up by hand slipped twice."}')


def _calls(n: int, name: str = "read") -> list[dict]:
    out = []
    for i in range(n):
        out.append({"role": "assistant", "content": "", "tool_calls": [{"id": str(i), "function": {"name": name}}]})
        out.append({"role": "tool", "content": "ok", "name": name})
    return out


def _turn(said: str, n: int, name: str = "read") -> list[dict]:
    return [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "Hello"},
            {"role": "user", "content": said}, *_calls(n, name), {"role": "assistant", "content": "Done: the total is 15."}]


class _Agent:
    def __init__(self, messages, reply=DRAFT):
        self.session = types.SimpleNamespace(id="s", messages=messages)
        self.instructions = types.SimpleNamespace(load=lambda name: f"SYSTEM:{name}")
        self.reply = reply
        self.calls: list[dict] = []

    async def _complete_with_guards(self, **kw):
        self.calls.append(kw)
        return types.SimpleNamespace(content=self.reply)


@pytest.fixture
def deck(monkeypatch):
    sent: list[dict] = []

    async def send(proposal):
        sent.append(proposal)
        return True, ""

    cfg = types.SimpleNamespace(tool_proposals=ToolProposalsConfig())
    monkeypatch.setattr(config_mod, "get_config", lambda: cfg)
    monkeypatch.setattr(tp, "_ready", lambda: True)
    monkeypatch.setattr(tp, "send", send)
    from captain_claw import custom_tools_client

    monkeypatch.setattr(custom_tools_client, "owner_tools", lambda: [{"ref": "echo", "title": "Echo", "description": "Says it back."}])
    return types.SimpleNamespace(sent=sent, cfg=cfg)


def test_the_cue_words_read_like_a_repeated_mechanical_job():
    for said in ("Add up these invoices again", "I do this every month", "convert the CSV to JSON",
                 "Svaki mjesec zbroji račune", "reconcile the statements"):
        assert tp.has_cue(said), said
    for said in ("What's the weather?", "Thanks!", "Write a poem", "/new"):
        assert not tp.has_cue(said), said


def test_the_last_turn_is_what_was_asked_answered_and_called():
    said, after, tools = tp.last_turn(_turn("Total these", 3, "read"))
    assert said == "Total these" and after == "Done: the total is 15." and tools == ["read"] * 3
    only_results = [{"role": "user", "content": "x"}, {"role": "tool", "content": "a", "name": "grep"}]
    assert tp.last_turn(only_results)[2] == ["grep"]
    assert tp.last_turn([]) == ("", "", [])


def test_the_gate():
    assert not tp.worth_asking("What's in the file?", ["read"] * 3, 8)
    assert tp.worth_asking("What's in the file?", ["read"] * 8, 8)
    assert tp.worth_asking("add these up again", ["read"] * 3, 8)
    assert not tp.worth_asking("add these up again", ["read"] * 2, 8)
    assert not tp.worth_asking("add these up again", ["custom_tools"] * 9, 8)


def test_the_draft_is_read_and_cleaned():
    got = tp.parse(DRAFT)
    assert got["name"] == "invoice-totals" and got["title"] == "Total invoices" and got["why"].startswith("Adding")
    assert tp.parse('```json\n' + DRAFT + '\n```')["name"] == "invoice-totals"
    assert tp.parse('{"name": "Invoice Totals!", "what": "' + "x" * 30 + '"}')["name"] == "invoice-totals"
    assert tp.parse('{"name": ""}') is None
    assert tp.parse('{"name": "short", "what": "too short"}') is None
    assert tp.parse("No tool would help.") is None


async def test_a_long_hand_made_job_becomes_a_proposal(deck):
    agent = _Agent(_turn("Give me the totals of these invoices", 9))
    got = await tp.maybe_propose_tool(agent, channel="web")
    assert deck.sent == [got] and got["name"] == "invoice-totals" and got["source"] == "noticed"
    [call] = agent.calls
    assert call["interaction_label"] == "tool_proposal" and call["tools"] is None
    assert call["messages"][0].content == f"SYSTEM:{tp.PROMPT}"
    prompt = call["messages"][1].content
    assert "Give me the totals" in prompt and "(9 calls)" in prompt and "- echo: Echo." in prompt


async def test_most_turns_cost_nothing(deck):
    agent = _Agent(_turn("What's in this file?", 2))
    assert await tp.maybe_propose_tool(agent) is None and agent.calls == [] and deck.sent == []


async def test_the_model_can_say_no_tool_would_help(deck):
    agent = _Agent(_turn("add these up again", 4), reply='{"name": ""}')
    assert await tp.maybe_propose_tool(agent) is None and len(agent.calls) == 1 and deck.sent == []


async def test_a_cooldown_and_a_daily_cap(deck):
    agent = _Agent(_turn("add these up again", 4))
    assert await tp.maybe_propose_tool(agent)
    assert await tp.maybe_propose_tool(agent) is None  # cooldown
    deck.cfg.tool_proposals.cooldown_seconds = 0
    deck.cfg.tool_proposals.max_per_day = 2
    assert await tp.maybe_propose_tool(agent)
    assert await tp.maybe_propose_tool(agent) is None  # the day's cap
    assert len(agent.calls) == 2


async def test_off_or_without_a_deck_that_can_build(deck, monkeypatch):
    agent = _Agent(_turn("add these up again", 9))
    deck.cfg.tool_proposals.enabled = False
    assert await tp.maybe_propose_tool(agent) is None
    deck.cfg.tool_proposals.enabled = True
    monkeypatch.setattr(tp, "_ready", lambda: False)
    assert await tp.maybe_propose_tool(agent) is None and agent.calls == []


async def test_a_failing_model_leaves_it_ready_for_next_time(deck):
    class Broken(_Agent):
        async def _complete_with_guards(self, **kw):
            raise RuntimeError("model down")

    agent = Broken(_turn("add these up again", 9))
    assert await tp.maybe_propose_tool(agent) is None
    assert getattr(agent, tp._ATTR_RUNNING) is False
