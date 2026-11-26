"""Direct writes (CLAW_WRITE_DIRECT: Code agents, benchmark runs) put a file
where it is asked for, so the system prompt says the saved/ rules cover
scratch only — otherwise an agent moves a file the task named into saved/."""

from __future__ import annotations

from pathlib import Path

import pytest

import captain_claw.agent_context_mixin as acm
from captain_claw.instructions import InstructionLoader
from captain_claw.llm import LLMProvider, LLMResponse

_INSTRUCTIONS = Path(acm.__file__).resolve().parent / "instructions"
NOTE = "never move, copy or mirror it under saved/"


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
    monkeypatch.delenv("CLAW_WRITE_DIRECT", raising=False)


def _prompt(monkeypatch, tmp_path, level: str = "normal", member: bool = False) -> str:
    from captain_claw.agent import Agent
    from captain_claw.session import Session

    monkeypatch.setattr("captain_claw.tools.registry._registry", None)
    agent = Agent(provider=_Provider())
    agent.session = Session(id="s1", name="direct")
    agent.instructions = InstructionLoader(
        base_dir=_INSTRUCTIONS, personal_dir=tmp_path / "personal",
        use_micro=level == "micro", use_nano=level == "nano",
    )
    agent._build_skills_system_prompt_section = lambda: ""
    if member:
        agent._speaker_scoped = True
        agent._speaker_principal = None
        agent._speaker_profile = ("## About the member\nM", "M")
    return agent._build_system_prompt()


@pytest.mark.parametrize("level", ["normal", "micro", "nano"])
def test_direct_writes_tell_the_agent_named_files_stay_put(monkeypatch, tmp_path, level):
    monkeypatch.setenv("CLAW_WRITE_DIRECT", "1")
    prompt = _prompt(monkeypatch, tmp_path, level)
    assert prompt.count(NOTE) == 1
    assert "{write_direct_block}" not in prompt


@pytest.mark.parametrize("level", ["normal", "micro", "nano"])
def test_the_usual_saved_rules_alone_without_direct_writes(monkeypatch, tmp_path, level):
    prompt = _prompt(monkeypatch, tmp_path, level)
    assert NOTE not in prompt and "{write_direct_block}" not in prompt


def test_a_member_never_gets_it(monkeypatch, tmp_path):
    monkeypatch.setenv("CLAW_WRITE_DIRECT", "1")
    assert NOTE not in _prompt(monkeypatch, tmp_path, member=True)
