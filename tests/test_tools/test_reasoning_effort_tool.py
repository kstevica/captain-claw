"""The reasoning_effort tool: the agent changes how hard it thinks the way a
person does in Flight Deck — the chat box's picker for a web chat turn, the
per-source settings for everything else — and the change applies at once."""

from __future__ import annotations

import json
import time
import types

import pytest

from captain_claw import msg_origin, turn_source
from captain_claw.agent import Agent
from captain_claw.agent_guard_mixin import AgentGuardMixin
from captain_claw.llm import reasoning
from captain_claw.tools.reasoning_effort import ReasoningEffortTool

# claude-sonnet-4-6 offers off · low · medium · high · max and runs at off.
SONNET = ("anthropic", "claude-sonnet-4-6")


@pytest.fixture(autouse=True)
def _settings(tmp_path, monkeypatch):
    path = tmp_path / ".captain-claw" / "reasoning_effort.json"
    monkeypatch.setattr(reasoning, "SETTINGS_PATH", path)
    monkeypatch.setattr(reasoning, "_settings_cache", {"mtime": None, "value": None})
    monkeypatch.setitem(reasoning._openrouter_state, "fetched_at", time.time())
    monkeypatch.setattr(reasoning, "_REFUSED", set())
    return path


def _agent(model=SONNET, broadcast=True, **hint):
    """An object carrying what the tool and the guard read, mid-turn."""
    obj = types.SimpleNamespace()
    if hint:
        msg_origin.hint_turn_provenance(obj, **hint)
    Agent._set_turn_origin(obj, "hello", None, None, None)
    obj.provider = types.SimpleNamespace(provider=model[0], model=model[1], base_url=None)
    obj.frames = []
    if broadcast:
        obj.ws_broadcast = obj.frames.append
    return obj


async def _run(agent, **args):
    return await ReasoningEffortTool().execute(_agent=agent, **args)


def _effort(agent):
    return AgentGuardMixin._turn_reasoning_effort(agent)


def _saved(path):
    return json.loads(path.read_text())


# ── settings rows ────────────────────────────────────────────────────


@pytest.mark.parametrize("source,key", [
    ("whatsapp", "whatsapp"),
    ("device:sticks3", "device:sticks3"),
    ("app:android", "app"),
    ("glasses:meta-display", "glasses"),
    ("device:newkind", "device:newkind"),
    ("web", "web"),
    ("", "default"),
])
def test_settings_key_is_the_row_a_person_changes(source, key):
    assert turn_source.settings_key(source) == key


def test_governing_key_most_specific_first():
    cfg = {"default": "low", "channels": {"device": "high", "device:papercolor": "off"}}
    assert reasoning.governing_key("device:papercolor", cfg) == "device:papercolor"
    assert reasoning.governing_key("device:sticks3", cfg) == "device"
    assert reasoning.governing_key("whatsapp", cfg) == "default"
    assert reasoning.governing_key("web", cfg) == "web"


# ── set, where the message came from ─────────────────────────────────


async def test_whatsapp_turn_changes_the_whatsapp_setting_now(_settings):
    agent = _agent(channel="whatsapp")
    assert _effort(agent) is None
    r = await _run(agent, action="set", level="high")
    assert r.success, r.error
    assert _saved(_settings) == {"channels": {"whatsapp": "high"}}
    assert _effort(agent) == "high"  # the rest of this turn
    assert agent.frames == []  # the chat box isn't involved
    assert "WhatsApp" in r.content
    # The next WhatsApp turn reads the setting.
    assert _effort(_agent(channel="whatsapp")) == "high"


async def test_web_turn_moves_the_chat_box_picker(_settings):
    agent = _agent(channel="web")
    with reasoning.turn_effort("low"):  # what the chat box sent
        r = await _run(agent, action="set", level="max")
        assert r.success, r.error
        assert _effort(agent) == "max"  # wins over the frame's pick for the rest of the turn
    assert agent.frames == [{"type": "reasoning_effort", "level": "max"}]
    assert not _settings.exists()  # the web chat's level is Flight Deck's, not the file's


async def test_web_turn_without_a_chat_to_tell(_settings):
    agent = _agent(channel="web", broadcast=False)
    r = await _run(agent, action="set", level="high")
    assert not r.success and "chat box" in r.error
    assert _effort(agent) == "high"  # still this turn


async def test_auto_sends_nothing(_settings):
    agent = _agent(channel="telegram")
    r = await _run(agent, action="set", level="auto")
    assert r.success, r.error
    assert _saved(_settings)["channels"]["telegram"] == "auto"
    assert _effort(agent) is None
    web = _agent(channel="web")
    await _run(web, action="set", level="auto")
    assert web.frames[-1]["level"] == "auto"


async def test_level_the_model_lacks_goes_to_the_nearest(_settings):
    agent = _agent(channel="whatsapp")
    r = await _run(agent, action="set", level="xhigh")
    assert r.success, r.error
    assert _saved(_settings)["channels"]["whatsapp"] == "high"  # tie → the cheaper one
    assert "nearest is High" in r.content


# ── scope and explicit sources ───────────────────────────────────────


async def test_turn_scope_lasts_one_turn(_settings):
    agent = _agent(channel="whatsapp")
    r = await _run(agent, action="set", level="max", scope="turn")
    assert r.success, r.error
    assert _effort(agent) == "max"
    assert not _settings.exists() and agent.frames == []
    # The next turn starts without it.
    Agent._set_turn_origin(agent, "next", None, "whatsapp", None)
    assert _effort(agent) is None


async def test_saving_this_sources_setting_replaces_a_turn_level(_settings):
    agent = _agent(channel="whatsapp")
    await _run(agent, action="set", level="max", scope="turn")
    await _run(agent, action="set", level="low")
    assert _effort(agent) == "low"


async def test_another_source_from_the_web_chat(_settings):
    agent = _agent(channel="web")
    with reasoning.turn_effort("medium"):
        r = await _run(agent, action="set", level="low", source="device:stopwatch")
        assert r.success, r.error
        assert _effort(agent) == "medium"  # this web turn keeps the chat box's pick
    assert _saved(_settings) == {"channels": {"device:stopwatch": "low"}}
    assert agent.frames == []
    assert "next message" in r.content


async def test_default_and_web_by_name(_settings):
    agent = _agent(channel="whatsapp")
    r = await _run(agent, action="set", level="medium", source="default")
    assert r.success, r.error
    assert _saved(_settings)["default"] == "medium"
    assert _effort(agent) == "medium"  # WhatsApp has no level of its own
    r = await _run(agent, action="set", level="high", source="web")
    assert r.success, r.error
    assert agent.frames == [{"type": "reasoning_effort", "level": "high"}]
    assert _effort(agent) == "medium"  # a web chat pick doesn't decide a WhatsApp turn
    r = await _run(agent, action="set", level="high", source="carrier-pigeon")
    assert not r.success and "Unknown source" in r.error


# ── get ──────────────────────────────────────────────────────────────


async def test_get_reports_levels_this_turn_and_settings(_settings):
    _settings.parent.mkdir(parents=True)
    _settings.write_text(json.dumps({"default": "low", "channels": {"whatsapp": "xhigh"}}))
    agent = _agent(channel="whatsapp")
    r = await _run(agent, action="get")
    assert r.success, r.error
    assert "Off, Low, Medium, High, Max" in r.content
    assert "This message (WhatsApp): Extra high (runs at High on this model)" in r.content
    assert "- Default: Low" in r.content
    web = _agent(channel="web")
    with reasoning.turn_effort("medium"):
        assert "Medium — the chat box's pick" in (await _run(web, action="get")).content


# ── refusals ─────────────────────────────────────────────────────────


@pytest.mark.parametrize("hint", [{"channel": "public"}, {"channel": "member"}, {"source": "link"}])
async def test_not_the_owners_turn(_settings, hint):
    agent = _agent(**hint)
    for args in ({"action": "get"}, {"action": "set", "level": "max"}):
        r = await _run(agent, **args)
        assert not r.success and "owner" in r.error
    assert not _settings.exists()


async def test_shared_agent_member_lane(_settings):
    agent = _agent(channel="web")
    agent._speaker_scoped = True
    r = await _run(agent, action="set", level="max")
    assert not r.success and agent.frames == []


async def test_model_without_controls(_settings):
    agent = _agent(model=("openai", "gpt-4o"), channel="whatsapp")
    r = await _run(agent, action="set", level="high")
    assert not r.success and "no reasoning-effort controls" in r.error
    assert "no reasoning-effort controls" in (await _run(agent, action="get")).content


async def test_bad_arguments(_settings):
    agent = _agent(channel="whatsapp")
    assert "Give a level" in (await _run(agent, action="set")).error
    assert "Unknown level" in (await _run(agent, action="set", level="ludicrous")).error
    assert "Unknown scope" in (await _run(agent, action="set", level="low", scope="forever")).error
    assert "Unknown action" in (await _run(agent, action="toggle")).error


# ── every agent has it ───────────────────────────────────────────────


def test_registered_everywhere():
    from captain_claw import member_privacy
    from captain_claw.agent_context_mixin import AgentContextMixin
    from captain_claw.agent_orchestration_mixin import _ECO_CORE_TOOLS
    from captain_claw.config import ToolsConfig
    from captain_claw.flight_deck.tool_risk import risk_for

    assert "reasoning_effort" in ToolsConfig(enabled=["read"]).enabled
    assert AgentContextMixin._BUILTIN_TOOL_MAP["reasoning_effort"] == ["reasoning_effort"]
    assert "reasoning_effort" in _ECO_CORE_TOOLS
    assert member_privacy.CONTENT_TURN_TOOLS["reasoning_effort"] is None
    assert risk_for("reasoning_effort", {"action": "get"}) == "read"
    assert risk_for("reasoning_effort", {"action": "set"}) == "write"


# ── any channel: the model is told it can ────────────────────────────


def _context_agent(text, *, earlier=(), tool=True, channel="whatsapp"):
    from captain_claw.agent_context_mixin import AgentContextMixin

    class _A(AgentContextMixin):
        pass

    obj = object.__new__(_A)
    msg_origin.hint_turn_provenance(obj, channel=channel)
    Agent._set_turn_origin(obj, text, None, None, None)
    obj._turn_user_text = text
    obj.session = types.SimpleNamespace(messages=[
        {"role": "user", "content": e, "turn_input": True} for e in (*earlier, text)])
    obj.tools = types.SimpleNamespace(has_tool=lambda name: tool and name == "reasoning_effort")
    return obj


@pytest.mark.parametrize("text,earlier", [
    ("set your reasoning to off", ()),
    ("think harder about this", ()),
    ("postavi razinu razmišljanja na high", ()),
    ("set it to off here", ("what is your reasoning level?",)),  # the follow-up
])
def test_turn_note_says_it_can_be_changed_from_here(text, earlier):
    line = _context_agent(text, earlier=earlier)._turn_source_line()
    assert "This message came from: WhatsApp." in line
    assert "call reasoning_effort set — it works from WhatsApp" in line
    assert "never tell the user it can only be changed in Flight Deck" in line


@pytest.mark.parametrize("text,tool", [
    ("I think on Monday works", True),
    ("for some reason it failed", True),
    ("set your reasoning to off", False),  # no tool, no hint
])
def test_turn_note_stays_plain_otherwise(text, tool):
    line = _context_agent(text, tool=tool)._turn_source_line()
    assert line.startswith("This message came from: WhatsApp.") and "reasoning_effort" not in line


async def test_texts_say_any_channel(_settings):
    from captain_claw.agent_context_mixin import _TOOL_PROMPT_DESCRIPTIONS

    assert "EVERY channel" in ReasoningEffortTool.description
    assert "only be changed in Flight Deck" in ReasoningEffortTool.description
    assert "EVERY channel" in _TOOL_PROMPT_DESCRIPTIONS["reasoning_effort"]
    report = (await _run(_agent(channel="whatsapp"), action="get")).content
    assert "You can change any of them from any channel, this one included" in report
    assert "Settings for every source" not in report


# ── a level forced for the whole run (benchmark harness) ─────────────


async def test_a_locked_level_beats_every_pick_and_the_tool_cant_change_it(monkeypatch, _settings):
    monkeypatch.setenv(reasoning.LOCK_ENV, "max")
    agent = _agent(channel="api")
    assert reasoning.locked_effort() == "max"
    with reasoning.turn_effort("low"):          # a chat box pick
        assert _effort(agent) == "max"
    agent._turn_effort_override = "low"         # the agent's own turn override
    assert _effort(agent) == "max"

    refused = await _run(agent, action="set", level="low", scope="setting")
    assert not refused.success and "locked at Max" in refused.error
    assert not _settings.exists()                # nothing written
    shown = await _run(agent, action="get")
    assert shown.success and shown.content.startswith("Reasoning effort is locked at Max")


async def test_a_locked_level_the_model_lacks_shows_as_its_nearest(monkeypatch):
    monkeypatch.setenv(reasoning.LOCK_ENV, "xhigh")   # Sonnet 4.6 has no xhigh
    shown = await _run(_agent(channel="api"), action="get")
    assert "locked at" in shown.content and "Extra high" not in shown.content.splitlines()[0]


def test_no_lock_leaves_the_usual_order(monkeypatch):
    monkeypatch.delenv(reasoning.LOCK_ENV, raising=False)
    assert reasoning.locked_effort() is None
    monkeypatch.setenv(reasoning.LOCK_ENV, "auto")    # "auto" forces nothing
    assert reasoning.locked_effort() is None
