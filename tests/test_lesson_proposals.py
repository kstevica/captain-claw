"""The agent proposes a lesson from what its owner just wrote — a correction or
a standing rule — drafted by its own model and handed to Flight Deck for the
owner to approve (captain_claw/lesson_proposals.py). Most turns stop at the
word check and cost nothing."""

from __future__ import annotations

import types

import pytest

from captain_claw import config as config_mod
from captain_claw import lesson_proposals as lp
from captain_claw.config import LessonProposalsConfig


def test_only_messages_that_read_like_a_correction_or_a_rule_pass_the_gate():
    for said in ("Too long — keep it short.", "From now on CC Marko on quotes.", "Don't add emojis.",
                 "Always answer in Croatian", "Ubuduće piši kraće.", "Nemoj slati ponude bez mene.",
                 "Ask me before you book anything.", "I told you to use the price list."):
        assert lp.has_cue(said), said
    for said in ("What's the weather in Zagreb?", "Thanks!", "Send the quote to Ana.", "ok",
                 "/new", "Koliko košta dostava?", "always " + "x" * 3000):
        assert not lp.has_cue(said), said
    assert lp.has_cue("Don’t sign as Captain")            # a typographic apostrophe


def test_the_exchange_is_the_answer_the_owner_wrote_after_their_message_and_the_reply():
    messages = [
        {"role": "user", "content": "Write the offer."},
        {"role": "assistant", "content": "Here is a long offer…"},
        {"role": "user", "content": "Too long."},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "1"}]},
        {"role": "tool", "content": "done"},
        {"role": "assistant", "content": [{"type": "text", "text": "Shorter offer."}]},
    ]
    assert lp.exchange(messages) == ("Here is a long offer…", "Too long.", "Shorter offer.")
    assert lp.exchange([{"role": "user", "content": "Always answer in Croatian."}]) == (
        "", "Always answer in Croatian.", "")
    assert lp.exchange([]) == ("", "", "")


def test_the_models_lesson_is_read_and_cleaned():
    assert lp.parse('{"kind": "ask_first", "text": " Ask before  booking. ", "always": true, "quote": "ask me"}') == {
        "kind": "ask_first", "text": "Ask before booking.", "always": True, "quote": "ask me"}
    fenced = '```json\n{"kind": "fact", "text": "The warehouse closes at 15:00.", "always": true}\n```'
    assert lp.parse(fenced) == {"kind": "fact", "text": "The warehouse closes at 15:00.", "always": False, "quote": ""}
    assert lp.parse('{"kind": "whim", "text": "Sign as Bob."}')["kind"] == "rule"
    assert lp.parse('{"text": ""}') is None
    assert lp.parse("Nothing to learn here.") is None
    assert lp.parse('["not", "an object"]') is None


class _Agent:
    def __init__(self, messages, reply='{"kind": "rule", "text": "Keep answers to three short paragraphs.", '
                                       '"always": true, "quote": "Too long"}'):
        self.session = types.SimpleNamespace(id="sess-1", messages=messages)
        self.instructions = types.SimpleNamespace(load=lambda name: f"SYSTEM:{name}")
        self.reply = reply
        self.calls: list[dict] = []

    async def _complete_with_guards(self, **kw):
        self.calls.append(kw)
        return types.SimpleNamespace(content=self.reply)


_TURN = [
    {"role": "user", "content": "Write the offer for Ana."},
    {"role": "assistant", "content": "Dear Ana, here is a very long offer…"},
    {"role": "user", "content": "Too long — keep it to three short paragraphs from now on."},
    {"role": "assistant", "content": "Got it. Dear Ana, …"},
]


@pytest.fixture
def deck(monkeypatch):
    """Under Flight Deck, with the default config; what was sent to it."""
    sent: list[dict] = []

    async def send(proposal):
        sent.append(proposal)
        return True

    cfg = types.SimpleNamespace(lesson_proposals=LessonProposalsConfig())
    monkeypatch.setattr(config_mod, "get_config", lambda: cfg)
    monkeypatch.setattr(lp, "_flight_deck_ready", lambda: True)
    monkeypatch.setattr(lp, "_send", send)
    monkeypatch.setattr(lp, "_near_lessons", lambda said: ["Answer in Croatian."])
    return types.SimpleNamespace(sent=sent, cfg=cfg)


async def test_a_correction_becomes_a_proposal_for_the_owner(deck):
    agent = _Agent(list(_TURN))
    got = await lp.maybe_propose_lesson(agent, channel="web")
    assert deck.sent == [got]
    assert got == {"kind": "rule", "text": "Keep answers to three short paragraphs.", "always": True, "quote": "Too long",
                   "said": "Too long — keep it to three short paragraphs from now on.",
                   "reply": "Dear Ana, here is a very long offer…", "channel": "web", "session_id": "sess-1"}
    [call] = agent.calls
    assert call["interaction_label"] == "lesson_proposal" and call["tools"] is None
    system, user = call["messages"]
    assert system.content == "SYSTEM:lesson_proposal_prompt.md"
    assert "Dear Ana, here is a very long offer" in user.content and "three short paragraphs" in user.content
    assert "- Answer in Croatian." in user.content                      # what it has, so a repeat isn't proposed


async def test_most_turns_cost_nothing(deck):
    agent = _Agent([{"role": "user", "content": "What's on my calendar today?"}, {"role": "assistant", "content": "…"}])
    assert await lp.maybe_propose_lesson(agent) is None and agent.calls == [] and deck.sent == []


async def test_when_the_model_finds_nothing_to_learn_nothing_is_sent(deck):
    agent = _Agent(list(_TURN), reply='{"text": ""}')
    assert await lp.maybe_propose_lesson(agent) is None and len(agent.calls) == 1 and deck.sent == []


async def test_a_cooldown_and_a_daily_cap_bound_the_drafts(deck):
    agent = _Agent(list(_TURN))
    assert await lp.maybe_propose_lesson(agent)
    assert await lp.maybe_propose_lesson(agent) is None and len(agent.calls) == 1   # within the cooldown
    deck.cfg.lesson_proposals.cooldown_seconds = 0
    deck.cfg.lesson_proposals.max_per_day = 2
    assert await lp.maybe_propose_lesson(agent)
    assert await lp.maybe_propose_lesson(agent) is None and len(agent.calls) == 2   # the day's two are spent


async def test_off_or_outside_flight_deck_it_never_drafts(deck, monkeypatch):
    agent = _Agent(list(_TURN))
    deck.cfg.lesson_proposals.enabled = False
    assert await lp.maybe_propose_lesson(agent) is None
    deck.cfg.lesson_proposals.enabled = True
    monkeypatch.setattr(lp, "_flight_deck_ready", lambda: False)
    assert await lp.maybe_propose_lesson(agent) is None
    assert agent.calls == [] and deck.sent == []


async def test_a_failing_model_never_breaks_the_turn(deck):
    agent = _Agent(list(_TURN))

    async def boom(**kw):
        raise RuntimeError("model down")

    agent._complete_with_guards = boom
    assert await lp.maybe_propose_lesson(agent) is None and deck.sent == []
    assert getattr(agent, lp._ATTR_RUNNING) is False


def test_the_proposal_prompt_carries_the_rules_that_keep_bad_lessons_out():
    from pathlib import Path
    text = (Path(lp.__file__).parent / "instructions" / lp.PROMPT).read_text(encoding="utf-8")
    assert '{"text": ""}' in text and '"quote"' in text
    assert "a lesson \"X doesn't work\" turns into a refusal" in text
    assert "an instruction for this task only" in text
    assert "When unsure, it teaches nothing" in text
