"""The model_switch tool: the agent switches among its provider's models when
the user asks — on its own only while its owner allows it — and the chat
box's picker follows."""

from __future__ import annotations

import types

import pytest

from captain_claw import agent_model_mixin
from captain_claw.agent import Agent
from captain_claw.agent_model_mixin import AgentModelMixin
from captain_claw.llm import model_switch
from captain_claw.tools.model_switch import ModelSwitchTool


@pytest.fixture(autouse=True)
def _env(tmp_path, monkeypatch):
    home = tmp_path / ".captain-claw"
    home.mkdir()
    monkeypatch.setattr(model_switch, "SWITCH_PATH", home / model_switch.SWITCH_FILENAME)
    monkeypatch.setattr(model_switch, "CHOICES_PATH", home / model_switch.CHOICES_FILENAME)
    monkeypatch.setattr(model_switch, "_custom_cache", {"mtime": None, "path": None, "value": None})
    monkeypatch.setattr(model_switch, "_refresh_state", {"at": 0.0, "key": "", "task": None})
    monkeypatch.delenv("FD_URL", raising=False)
    config = types.SimpleNamespace(
        model=types.SimpleNamespace(provider="anthropic", model="claude-sonnet-5-5", base_url="",
                                    api_key="", temperature=0.2, max_tokens=32768,
                                    tokens_per_minute=0, allowed=[], vision="auto"),
        context=types.SimpleNamespace(max_tokens=160000),
        provider_keys=types.SimpleNamespace(headers_for=lambda _p: None),
    )

    def create_provider(**kw):
        return types.SimpleNamespace(provider=kw["provider"], model=kw["model"], base_url=kw.get("base_url") or "",
                                     temperature=kw.get("temperature"), max_tokens=kw.get("max_tokens"))

    monkeypatch.setattr(agent_model_mixin, "get_config", lambda: config)
    monkeypatch.setattr(agent_model_mixin, "create_provider", create_provider)
    monkeypatch.setattr(agent_model_mixin, "set_provider", lambda _p: None)


class FakeAgent(AgentModelMixin):
    def __init__(self, said="hello", channel=None):
        self.provider = types.SimpleNamespace(provider="anthropic", model="claude-sonnet-5-5", base_url="")
        self._provider_override = False
        self._runtime_model_details = {}
        self.session = types.SimpleNamespace(metadata={})
        self.frames = []
        self.ws_broadcast = self.frames.append

        async def save_session(_session):
            pass

        self.session_manager = types.SimpleNamespace(save_session=save_session)
        self._turn_user_text = said
        Agent._set_turn_origin(self, said, None, channel, None)

    def _recent_user_text(self):
        return self._turn_user_text


async def _run(agent, **args):
    return await ModelSwitchTool().execute(_agent=agent, **args)


def _allow_own():
    model_switch.SWITCH_PATH.write_text("on")


async def test_get_lists_the_models_and_says_switching_on_its_own_is_off():
    result = await _run(FakeAgent(), action="get")
    assert result.success
    text = result.content
    assert "Claude Opus 5.5 (claude-opus-5-5) — Smartest" in text
    assert "[configured default, running now]" in text
    assert "per 1M tokens" in text
    assert "Switching on your own is OFF" in text


async def test_the_user_asks_and_it_switches_for_the_conversation():
    agent = FakeAgent(said="please switch to opus for the rest of this")
    result = await _run(agent, action="set", model="opus", reason="user_asked")
    assert result.success, result.error
    assert agent.provider.model == "claude-opus-5-5"
    assert agent.session.metadata["model_selection"]["switch"] is True
    frame = agent.frames[-1]
    assert frame["type"] == "model_switch" and frame["model_switch"]["current"] == "claude-opus-5-5"
    assert "reasoning" in frame


async def test_on_its_own_it_is_refused_while_the_owner_hasn_t_allowed_it():
    agent = FakeAgent(said="summarise these 300 rows")
    result = await _run(agent, action="set", model="haiku", reason="own")
    assert not result.success and "on your own is off" in result.error
    assert agent.provider.model == "claude-sonnet-5-5"


async def test_claiming_the_user_asked_when_they_didn_t_is_refused():
    agent = FakeAgent(said="summarise these 300 rows")
    result = await _run(agent, action="set", model="haiku", reason="user_asked")
    assert not result.success and "doesn't ask for a model change" in result.error
    # No reason given and nothing asked counts as its own choice.
    result = await _run(agent, action="set", model="haiku")
    assert not result.success


async def test_allowed_on_its_own_it_switches_for_the_turn_and_goes_back():
    _allow_own()
    agent = FakeAgent(said="prove this lemma")
    result = await _run(agent, action="set", model="smartest", reason="own")
    assert result.success, result.error
    assert agent.provider.model == "claude-opus-5-5"
    assert "model_selection" not in agent.session.metadata
    assert "Tell the user" in result.content
    # The next turn starts on the model before it.
    Agent._set_turn_origin(agent, "next", None, None, None)
    assert agent.provider.model == "claude-sonnet-5-5"


async def test_a_member_or_a_public_visitor_can_t_switch():
    agent = FakeAgent(said="switch to opus", channel="public")
    result = await _run(agent, action="set", model="opus", reason="user_asked")
    assert not result.success and "owner's setting" in result.error
    member = FakeAgent(said="switch to opus")
    member._speaker_scoped = True
    result = await _run(member, action="get")
    assert not result.success


async def test_a_model_that_isn_t_offered_is_refused_with_the_choices():
    agent = FakeAgent(said="use the gpt model")
    result = await _run(agent, action="set", model="gpt-6.1-sol", reason="user_asked")
    assert not result.success and "claude-haiku-5-5" in result.error


async def test_the_turn_note_tells_it_it_can_switch_from_here():
    agent = FakeAgent(said="can you switch to a cheaper model?")
    agent.tools = types.SimpleNamespace(has_tool=lambda name: name == "model_switch")
    from captain_claw.agent_context_mixin import AgentContextMixin

    line = AgentContextMixin._model_switch_line(agent)
    assert "call model_switch set" in line and "claude-haiku-5-5" in line
    # "cheaper" alone doesn't put it in the note (find me cheaper flights).
    vague = FakeAgent(said="find me cheaper flights")
    vague.tools = agent.tools
    assert AgentContextMixin._model_switch_line(vague) == ""
    quiet = FakeAgent(said="what's the weather")
    quiet.tools = agent.tools
    assert AgentContextMixin._model_switch_line(quiet) == ""
    _allow_own()
    assert "lets you switch on your own" in AgentContextMixin._model_switch_line(quiet)
