"""The system prompt tells the agent when to offer a team (Team up), only
while the `team` tool is offered. Without it the agent saw `team` as one of
dozens of tools and did a two-subject comparison alone."""

from __future__ import annotations

from pathlib import Path

import pytest

import captain_claw.agent_context_mixin as acm
from captain_claw import team_up
from captain_claw.instructions import InstructionLoader
from captain_claw.llm import LLMProvider, LLMResponse

_INSTRUCTIONS = Path(acm.__file__).resolve().parent / "instructions"
HEAD = "Team up — work split among helpers:"


class _Provider(LLMProvider):
    async def complete(self, messages, tools=None, temperature=None, max_tokens=None):
        return LLMResponse(content="ok")

    async def complete_streaming(self, messages, tools=None, temperature=None, max_tokens=None):
        if False:
            yield ""

    def count_tokens(self, text: str) -> int:
        return len(text.split()) or 1


@pytest.fixture(autouse=True)
def _home(monkeypatch, tmp_path):
    home = tmp_path / "home"
    (home / ".captain-claw").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(team_up, "_HOME", home / ".captain-claw")
    monkeypatch.setattr(team_up, "_SERVER", None)


def _prompt(monkeypatch, tmp_path, *, level: str = "normal", member: bool = False) -> str:
    from captain_claw.agent import Agent
    from captain_claw.session import Session
    from captain_claw.tools.team import TeamTool

    monkeypatch.setattr("captain_claw.tools.registry._registry", None)
    agent = Agent(provider=_Provider())
    agent.session = Session(id="s1", name="team")
    agent.instructions = InstructionLoader(
        base_dir=_INSTRUCTIONS, personal_dir=tmp_path / "personal",
        use_micro=level == "micro", use_nano=level == "nano",
    )
    agent._build_skills_system_prompt_section = lambda: ""
    if not agent.tools.has_tool("team"):
        agent.tools.register(TeamTool(), metadata={"requires_team_up": True})
    if member:
        agent._speaker_scoped = True
        agent._speaker_principal = None
        agent._speaker_profile = ("## About the member\nM", "M")
    return agent._build_system_prompt()


@pytest.mark.parametrize("level", ["normal", "micro"])
def test_section_when_team_up_is_offered(monkeypatch, tmp_path, level):
    team_up.bind_server(object())
    prompt = _prompt(monkeypatch, tmp_path, level=level)
    assert prompt.count(HEAD) == 1
    assert "call `team propose` FIRST" in prompt
    assert "even when each part looks quick" in prompt
    assert "{team_block}" not in prompt


def test_no_section_without_a_web_chat(monkeypatch, tmp_path):
    prompt = _prompt(monkeypatch, tmp_path)
    assert HEAD not in prompt
    assert "{team_block}" not in prompt


def test_no_section_when_the_setting_is_off(monkeypatch, tmp_path):
    team_up.bind_server(object())
    (team_up._HOME / team_up.SETTINGS_FILENAME).write_text('{"mode": "off"}')
    assert HEAD not in _prompt(monkeypatch, tmp_path)


def test_no_section_for_a_shared_agent_member(monkeypatch, tmp_path):
    team_up.bind_server(object())
    assert HEAD not in _prompt(monkeypatch, tmp_path, member=True)


def test_tool_description_offers_for_quick_parts_too():
    from captain_claw.tools.team import TeamTool

    text = TeamTool.description
    assert "even quick ones" in text
    assert "ONLY when" not in text and "substantial" not in text

