"""Course — the agent side.

Flight Deck writes ``course_context.md`` / ``course_context.compact.md`` (the
owner's turns) and ``course_deck.md`` / ``course_deck.compact.md`` (members'
turns) into the agent's ``~/.captain-claw``. The agent inserts them verbatim
before CACHE_SPLIT:

* an owner's turn gets the full course, right after the owner profile;
* a shared agent's member turn gets the deck-wide block only — never the
  owner's course;
* micro/nano templates and orchestrated workers get the compact files;
* public sessions, BotPort dispatch agents and Iskra bodies get neither.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

import captain_claw.agent_context_mixin as acm
import captain_claw.tenant_context as tc
from captain_claw.instructions import InstructionLoader
from captain_claw.llm import LLMProvider, LLMResponse

_INSTRUCTIONS = Path(acm.__file__).resolve().parent / "instructions"
_SPLIT = tc.CACHE_SPLIT_MARKER

OWNER_FULL = "## Your course\nOWNER-COURSE-FULL"
OWNER_COMPACT = "## Your course\nOWNER-COURSE-COMPACT"
DECK_FULL = "## This deck's course\nDECK-COURSE-FULL"
DECK_COMPACT = "## This deck's course\nDECK-COURSE-COMPACT"
PROFILE = "## Your owner\nOWNER-PROFILE-MARKER"


@pytest.fixture
def home(monkeypatch, tmp_path):
    h = tmp_path / "home"
    (h / ".captain-claw").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(h))
    for var in ("CLAW_VFS_PROJECT", "CLAW_BEING_WORKER", "CLAW_BEING_CAPS"):
        monkeypatch.delenv(var, raising=False)
    tc._cache.clear()
    yield h / ".captain-claw"
    tc._cache.clear()


def _write(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


@pytest.fixture
def course(home):
    _write(home / tc.COURSE_FULL_FILENAME, OWNER_FULL)
    _write(home / tc.COURSE_COMPACT_FILENAME, OWNER_COMPACT)
    _write(home / tc.COURSE_DECK_FULL_FILENAME, DECK_FULL)
    _write(home / tc.COURSE_DECK_COMPACT_FILENAME, DECK_COMPACT)
    _write(home / tc.FULL_FILENAME, PROFILE)
    return home


class _Provider(LLMProvider):
    async def complete(self, messages, tools=None, temperature=None, max_tokens=None):
        return LLMResponse(content="ok")

    async def complete_streaming(self, messages, tools=None, temperature=None, max_tokens=None):
        if False:
            yield ""

    def count_tokens(self, text: str) -> int:
        return len(text.split()) or 1


def _agent(monkeypatch, tmp_path, level: str = "normal"):
    from captain_claw.agent import Agent
    from captain_claw.session import Session

    monkeypatch.setattr("captain_claw.tools.registry._registry", None)
    agent = Agent(provider=_Provider())
    agent.session = Session(id="s1", name="course")
    agent.instructions = InstructionLoader(
        base_dir=_INSTRUCTIONS, personal_dir=tmp_path / "personal",
        use_micro=level == "micro", use_nano=level == "nano",
    )
    agent._build_skills_system_prompt_section = lambda: ""
    return agent


def _member(agent):
    agent._speaker_scoped = True
    agent._speaker_principal = None
    agent._speaker_profile = ("## About the member\nMEMBER-PROFILE", "MEMBER-COMPACT")
    return agent


# ── loader ───────────────────────────────────────────────────────────


def test_no_files_no_blocks(home):
    assert tc.load_course_context(compact=False) == ""
    assert tc.load_course_deck(compact=True) == ""


def test_loaders_pick_and_fall_back(home):
    _write(home / tc.COURSE_FULL_FILENAME, OWNER_FULL)
    _write(home / tc.COURSE_DECK_COMPACT_FILENAME, DECK_COMPACT)
    assert tc.load_course_context(compact=False) == OWNER_FULL
    assert tc.load_course_context(compact=True) == OWNER_FULL       # compact missing → full
    assert tc.load_course_deck(compact=True) == DECK_COMPACT
    assert tc.load_course_deck(compact=False) == DECK_COMPACT       # full missing → compact


# ── the prompt ───────────────────────────────────────────────────────


@pytest.mark.parametrize("level,block", [("normal", OWNER_FULL), ("micro", OWNER_COMPACT)])
def test_owner_turn_gets_the_course_after_the_profile(monkeypatch, tmp_path, course, level, block):
    prompt = _agent(monkeypatch, tmp_path, level)._build_system_prompt()
    assert prompt.count(block) == 1
    assert prompt.index(block) < prompt.index(_SPLIT)
    if level == "normal":
        assert prompt.index("OWNER-PROFILE-MARKER") < prompt.index(block)
    assert "DECK-COURSE" not in prompt


def test_nano_appends_the_compact_course(monkeypatch, tmp_path, course):
    prompt = _agent(monkeypatch, tmp_path, "nano")._build_system_prompt()
    assert OWNER_COMPACT in prompt and OWNER_FULL not in prompt


def test_orchestrated_worker_gets_the_compact_course(monkeypatch, tmp_path, course):
    monkeypatch.setenv("CLAW_VFS_PROJECT", "council-run-7")
    prompt = _agent(monkeypatch, tmp_path)._build_system_prompt()
    assert OWNER_COMPACT in prompt and OWNER_FULL not in prompt


def test_member_turn_gets_the_deck_course_only(monkeypatch, tmp_path, course):
    prompt = _member(_agent(monkeypatch, tmp_path))._build_system_prompt()
    assert DECK_FULL in prompt
    assert prompt.index("MEMBER-PROFILE") < prompt.index(DECK_FULL) < prompt.index(_SPLIT)
    assert "OWNER-COURSE" not in prompt and "OWNER-PROFILE-MARKER" not in prompt


def test_member_turn_micro_gets_the_compact_deck_course(monkeypatch, tmp_path, course):
    prompt = _member(_agent(monkeypatch, tmp_path, "micro"))._build_system_prompt()
    assert DECK_COMPACT in prompt and DECK_FULL not in prompt
    assert "OWNER-COURSE" not in prompt


@pytest.mark.parametrize("kind", ["public", "hidden", "being"])
@pytest.mark.parametrize("member", [False, True])
def test_refused_instances_get_no_course(monkeypatch, tmp_path, course, kind, member):
    agent = _agent(monkeypatch, tmp_path)
    if member:
        _member(agent)
    if kind == "public":
        agent._public_scoped = True
    elif kind == "hidden":
        agent._tenant_hidden = True
    else:
        monkeypatch.setenv("CLAW_BEING_WORKER", "1")
    prompt = agent._build_system_prompt()
    assert "OWNER-COURSE" not in prompt and "DECK-COURSE" not in prompt


def test_course_change_applies_on_the_next_build(monkeypatch, tmp_path, course):
    agent = _agent(monkeypatch, tmp_path)
    assert OWNER_FULL in agent._build_system_prompt()
    _write(course / tc.COURSE_FULL_FILENAME, "## Your course\nNEW-COURSE")
    prompt = agent._build_system_prompt()
    assert "NEW-COURSE" in prompt and OWNER_FULL not in prompt
    for name in (tc.COURSE_FULL_FILENAME, tc.COURSE_COMPACT_FILENAME):
        (course / name).unlink()
    assert "## Your course" not in agent._build_system_prompt()
