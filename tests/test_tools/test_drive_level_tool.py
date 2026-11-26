"""The drive_level tool: the agent changes how far it pushes the way a person
does in Flight Deck — the chat box's Drive picker for a web chat turn, its own
default for everything else — and the change applies at once."""

from __future__ import annotations

import types

import pytest

from captain_claw import drive, msg_origin
from captain_claw.agent import Agent
from captain_claw.tools.drive_level import DriveLevelTool


@pytest.fixture(autouse=True)
def _settings(tmp_path, monkeypatch):
    monkeypatch.setattr(drive, "SETTING_PATH", tmp_path / ".captain-claw" / "drive.txt")
    monkeypatch.setattr(drive, "INITIATIVE_PATH", tmp_path / ".captain-claw" / "drive_initiative.txt")
    monkeypatch.setattr(drive, "mode_suggestion", lambda agent=None: None)
    cfg = types.SimpleNamespace(drive=types.SimpleNamespace(default="", initiative=True))
    monkeypatch.setattr("captain_claw.config.get_config", lambda: cfg)
    return tmp_path / ".captain-claw"


def _agent(broadcast=True, **hint):
    """An object carrying what the tool and the harness read, mid-turn (a web
    chat turn unless another channel or source is given)."""
    obj = types.SimpleNamespace()
    hint = hint or {"channel": "web"}
    if hint:
        msg_origin.hint_turn_provenance(obj, **hint)
    Agent._set_turn_origin(obj, "hello", None, None, None)
    obj.frames = []
    if broadcast:
        obj.ws_broadcast = obj.frames.append
    return obj


async def _run(agent, **args):
    return await DriveLevelTool().execute(_agent=agent, **args)


async def test_get_says_the_level_in_effect_and_why():
    agent = _agent()
    out = await _run(agent, action="get")
    assert out.success and "Diligent — Diligent, the everyday level" in out.content
    assert "Extra work at Ambitious" in out.content and ": on." in out.content
    with drive.turn_drive("persistent"):
        out = await _run(agent, action="get")
    assert "Persistent — the chat box's pick" in out.content


async def test_a_web_turn_moves_the_chat_box_and_applies_to_the_rest_of_the_turn(_settings):
    agent = _agent()
    with drive.turn_drive("relaxed"):
        out = await _run(agent, action="set", level="Ambitious")
        assert out.success and "chat box's picker" in out.content
        assert agent.frames == [{"type": "drive", "level": "ambitious"}]
        assert drive.current_name(agent) == "ambitious"          # at once, over the frame's pick
    assert not (_settings / "drive.txt").exists()                 # the agent's default is untouched


async def test_auto_in_the_chat_box_falls_back_to_the_default(_settings):
    _settings.mkdir(parents=True)
    (_settings / "drive.txt").write_text("persistent")
    agent = _agent()
    with drive.turn_drive("relaxed"):
        out = await _run(agent, action="set", level="auto")
        assert out.success and agent.frames == [{"type": "drive", "level": "auto"}]
        assert drive.current_name(agent) == "persistent"


async def test_another_channel_sets_the_agents_own_default(_settings):
    agent = _agent(channel="whatsapp")
    out = await _run(agent, action="set", level="persistent")
    assert out.success and "default is set to Persistent" in out.content
    assert (_settings / "drive.txt").read_text() == "persistent"
    assert agent.frames == [] and drive.current_name(agent) == "persistent"
    out = await _run(agent, action="set", level="auto")
    assert out.success and (_settings / "drive.txt").read_text() == "auto"
    assert drive.current_name(agent) == "diligent"


async def test_where_names_the_other_setting(_settings):
    agent = _agent()                                              # a web turn
    with drive.turn_drive("relaxed"):
        out = await _run(agent, action="set", level="ambitious", where="default")
        assert out.success and (_settings / "drive.txt").read_text() == "ambitious"
        assert agent.frames == [] and drive.current_name(agent) == "relaxed"   # the chat box's pick stays
    phone = _agent(channel="whatsapp")
    out = await _run(phone, action="set", level="relaxed", where="chat")
    assert out.success and phone.frames == [{"type": "drive", "level": "relaxed"}]
    assert "next web chat message" in out.content
    assert (await _run(phone, action="set", level="relaxed", where="mars")).success is False


async def test_turn_scope_changes_only_this_turn(_settings):
    agent = _agent()
    out = await _run(agent, action="set", level="ambitious", scope="turn")
    assert out.success and drive.current_name(agent) == "ambitious"
    assert agent.frames == [] and not (_settings / "drive.txt").exists()
    Agent._set_turn_origin(agent, "next", None, None, None)        # the next message
    assert agent._turn_drive_override is None and drive.current_name(agent) == "diligent"


async def test_no_chat_connected_is_said_and_still_applies_here():
    agent = _agent(broadcast=False)
    out = await _run(agent, action="set", level="persistent")
    assert out.success is False and "ask the user to pick Persistent" in out.error
    assert drive.current_name(agent) == "persistent"


async def test_the_extra_work_switch_alone_or_with_a_level(_settings):
    agent = _agent()
    out = await _run(agent, action="set", extra_work="off")
    assert out.success and "off" in out.content
    assert (_settings / "drive_initiative.txt").read_text() == "off" and drive.initiative_enabled() is False
    out = await _run(agent, action="set", extra_work="on", level="ambitious", scope="turn")
    assert out.success and drive.initiative_enabled() is True and drive.current_name(agent) == "ambitious"
    assert (await _run(agent, action="set", extra_work="maybe")).success is False


async def test_bad_input_is_refused():
    agent = _agent()
    assert (await _run(agent, action="set")).success is False
    assert (await _run(agent, action="set", level="turbo")).success is False
    assert (await _run(agent, action="set", level="relaxed", scope="week")).success is False
    assert (await _run(agent, action="wipe")).success is False
    assert (await DriveLevelTool().execute(action="get")).success is False


@pytest.mark.parametrize("source", ["public", "member", "link"])
async def test_not_the_owners_turn_cannot_read_or_change_it(source, _settings):
    agent = _agent(source=source)
    for args in ({"action": "get"}, {"action": "set", "level": "ambitious"}, {"action": "set", "extra_work": "off"}):
        out = await _run(agent, **args)
        assert out.success is False and "owner's setting" in out.error
    assert not (_settings / "drive.txt").exists() and not (_settings / "drive_initiative.txt").exists()


def test_the_tool_is_on_every_agent_and_classed():
    from captain_claw.config import Config
    from captain_claw.flight_deck import tool_risk

    assert "drive_level" in Config().tools.enabled
    assert tool_risk.risk_for("drive_level", {"action": "set"}) == "write"


def test_talk_about_drive_is_noticed_but_not_talk_about_files():
    from captain_claw.agent_context_mixin import _DRIVE_TALK_RE

    for text in ("set drive level to ambitious", "be more thorough this time", "go Persistent",
                 "turn extra work off", "budi ambiciozniji", "radi temeljitije"):
        assert _DRIVE_TALK_RE.search(text), text
    for text in ("save it to my drive", "upload the file to Google Drive", "drive folder for the team"):
        assert not _DRIVE_TALK_RE.search(text), text


# ── a level forced for the whole run (benchmark harness) ─────────────


async def test_a_locked_drive_beats_every_pick_and_the_tool_cant_change_it(monkeypatch, _settings):
    monkeypatch.setenv(drive.LOCK_ENV, "ambitious")
    agent = _agent(channel="api")
    with drive.turn_drive("relaxed"):
        assert drive.current_name(agent) == "ambitious"
    agent._turn_drive_override = "relaxed"
    assert drive.current_name(agent) == "ambitious"

    refused = await _run(agent, action="set", level="relaxed")
    assert not refused.success and "locked at Ambitious" in refused.error
    assert not (_settings / "drive.txt").exists()
    shown = await _run(agent, action="get")
    assert shown.success and shown.content.startswith("Drive is locked at Ambitious")


def test_an_unknown_or_empty_lock_forces_nothing(monkeypatch):
    for value in ("", "auto", "frantic"):
        monkeypatch.setenv(drive.LOCK_ENV, value)
        assert drive.locked() is None
        assert drive.current_name(None) == drive.DEFAULT
