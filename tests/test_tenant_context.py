"""Owner profile ("tenant context") — the agent side.

Flight Deck writes ``tenant_context.md`` (full) and
``tenant_context.compact.md`` into the agent's ``~/.captain-claw``; the agent
inserts the text verbatim into its system prompt, just before CACHE_SPLIT so
it stays in the cached static part. Micro/nano templates and orchestrated
workers get the compact block. Iskra bodies (at every stage), public-session
agents and BotPort dispatch agents get nothing — the owner's profile is not
theirs to see.

Also covers the micro/nano templates now rendering ``{fleet_instructions_block}``
(new agents default to eco/micro, which used to drop fleet instructions), and
the nano template clipping long fleet instructions.
"""

from __future__ import annotations

import os
import types
from pathlib import Path

import pytest

import captain_claw.agent_context_mixin as acm
import captain_claw.tenant_context as tc
from captain_claw.instructions import InstructionLoader
from captain_claw.llm import LLMProvider, LLMResponse

_INSTRUCTIONS = Path(acm.__file__).resolve().parent / "instructions"
_SPLIT = tc.CACHE_SPLIT_MARKER

FULL = (
    "## Your owner and their company\n"
    "You work for Ana, the Flight Deck user who owns this agent.\n\n"
    "### About your owner\nLikes {braces} kept verbatim."
)
COMPACT = "## Your owner\n- Name: Ana {compact}"


@pytest.fixture
def home(monkeypatch, tmp_path):
    """An isolated HOME with an empty ~/.captain-claw; env knobs cleared."""
    h = tmp_path / "home"
    (h / ".captain-claw").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(h))
    for var in ("CLAW_VFS_PROJECT", "CLAW_BEING_WORKER", "CLAW_BEING_CAPS"):
        monkeypatch.delenv(var, raising=False)
    tc._cache.clear()
    yield h / ".captain-claw"
    tc._cache.clear()


def _write(path: Path, text: str) -> None:
    """Write the way FD does: tmp file + os.replace."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


# ── loader ───────────────────────────────────────────────────────────


def test_no_files_means_no_block(home):
    assert tc.load_tenant_context(compact=False) == ""
    assert tc.load_tenant_context(compact=True) == ""


def test_loader_picks_the_requested_file(home):
    _write(home / tc.FULL_FILENAME, FULL + "\n\n")
    _write(home / tc.COMPACT_FILENAME, COMPACT)
    assert tc.load_tenant_context(compact=False) == FULL
    assert tc.load_tenant_context(compact=True) == COMPACT


@pytest.mark.parametrize("present,text,compact", [
    (tc.FULL_FILENAME, FULL, True),          # compact missing → full
    (tc.COMPACT_FILENAME, COMPACT, False),   # full missing → compact
])
def test_loader_falls_back_to_the_other_file(home, present, text, compact):
    _write(home / present, text)
    assert tc.load_tenant_context(compact=compact) == text


def test_loader_reads_home_at_call_time(home, monkeypatch, tmp_path):
    _write(home / tc.FULL_FILENAME, FULL)
    assert tc.load_tenant_context(compact=False) == FULL
    other = tmp_path / "other-home"
    (other / ".captain-claw").mkdir(parents=True)
    _write(other / ".captain-claw" / tc.FULL_FILENAME, "## Your owner\nBo")
    monkeypatch.setenv("HOME", str(other))
    assert tc.load_tenant_context(compact=False) == "## Your owner\nBo"


def test_loader_is_mtime_cached(home, monkeypatch):
    path = home / tc.FULL_FILENAME
    _write(path, FULL)
    reads = []
    real_read = Path.read_text

    def counting_read(self, *a, **kw):
        reads.append(self.name)
        return real_read(self, *a, **kw)

    monkeypatch.setattr(Path, "read_text", counting_read)
    for _ in range(3):
        assert tc.load_tenant_context(compact=False) == FULL
    assert reads == [tc.FULL_FILENAME]       # one disk read, then cache hits

    # A profile save (tmp + replace) is picked up on the next call.
    _write(path, "## Your owner\nUpdated.")
    assert tc.load_tenant_context(compact=False) == "## Your owner\nUpdated."
    assert len(reads) == 2

    # An emptied profile deletes the files → the block disappears.
    path.unlink()
    assert tc.load_tenant_context(compact=False) == ""


# ── helpers: where the block goes, which file is chosen ──────────────


def test_insert_goes_right_before_the_first_cache_split():
    prompt = f"static part\n\n{_SPLIT}\ndynamic\n{_SPLIT}\nmore"
    out = tc.insert_tenant_block(prompt, FULL)
    assert out == f"static part\n\n{FULL}\n\n{_SPLIT}\ndynamic\n{_SPLIT}\nmore"


def test_insert_appends_without_a_split():
    assert tc.insert_tenant_block("nano prompt\n", COMPACT) == f"nano prompt\n\n{COMPACT}"


def test_insert_is_a_no_op_for_an_empty_block():
    assert tc.insert_tenant_block(f"a\n{_SPLIT}\nb", "  \n") == f"a\n{_SPLIT}\nb"


@pytest.mark.parametrize("micro,nano,vfs,expected", [
    (False, False, "", False),
    (True, False, "", True),
    (False, True, "", True),
    (False, False, "proj-1", True),          # orchestrated worker
    (False, False, "   ", False),
])
def test_compact_choice(monkeypatch, micro, nano, vfs, expected):
    monkeypatch.setenv("CLAW_VFS_PROJECT", vfs)
    assert tc.use_compact_tenant_context(micro=micro, nano=nano) is expected


# ── the agent's system prompt ────────────────────────────────────────


class _Provider(LLMProvider):
    async def complete(self, messages, tools=None, temperature=None, max_tokens=None):
        return LLMResponse(content="ok")

    async def complete_streaming(self, messages, tools=None, temperature=None, max_tokens=None):
        if False:
            yield ""

    def count_tokens(self, text: str) -> int:
        return len(text.split()) or 1


def _agent(monkeypatch, tmp_path, level: str):
    from captain_claw.agent import Agent
    from captain_claw.session import Session

    monkeypatch.setattr("captain_claw.tools.registry._registry", None)
    agent = Agent(provider=_Provider())
    agent.session = Session(id="s1", name="tenant")
    # HOME is already the isolated one, so eco_mode.txt/nano_mode.txt resolve
    # there (absent) and the level is exactly what is passed here.
    agent.instructions = InstructionLoader(
        base_dir=_INSTRUCTIONS, personal_dir=tmp_path / "personal",
        use_micro=level == "micro", use_nano=level == "nano",
    )
    # Skills are appended after everything; keep them out of the picture.
    agent._build_skills_system_prompt_section = lambda: ""
    return agent


@pytest.fixture
def profile(home):
    _write(home / tc.FULL_FILENAME, FULL)
    _write(home / tc.COMPACT_FILENAME, COMPACT)
    return home


@pytest.mark.parametrize("level,block", [("normal", FULL), ("micro", COMPACT)])
def test_block_sits_right_before_cache_split(monkeypatch, tmp_path, profile, level, block):
    prompt = _agent(monkeypatch, tmp_path, level)._build_system_prompt()
    assert prompt.count(block) == 1
    assert prompt.index(block) < prompt.index(_SPLIT)
    between = prompt[prompt.index(block) + len(block):prompt.index(_SPLIT)]
    assert between.strip() == ""
    other = COMPACT if block == FULL else FULL
    assert other not in prompt


def test_nano_appends_the_compact_block(monkeypatch, tmp_path, profile):
    prompt = _agent(monkeypatch, tmp_path, "nano")._build_system_prompt()
    assert _SPLIT not in prompt
    assert prompt.rstrip().endswith(COMPACT)
    assert FULL not in prompt


def test_orchestrated_worker_gets_the_compact_block(monkeypatch, tmp_path, profile):
    monkeypatch.setenv("CLAW_VFS_PROJECT", "council-run-7")
    prompt = _agent(monkeypatch, tmp_path, "normal")._build_system_prompt()
    assert COMPACT in prompt and FULL not in prompt
    assert prompt.index(COMPACT) < prompt.index(_SPLIT)


def test_missing_compact_falls_back_to_full_in_the_prompt(monkeypatch, tmp_path, home):
    _write(home / tc.FULL_FILENAME, FULL)
    prompt = _agent(monkeypatch, tmp_path, "micro")._build_system_prompt()
    assert FULL in prompt


def test_profile_change_applies_on_the_next_build(monkeypatch, tmp_path, profile):
    agent = _agent(monkeypatch, tmp_path, "normal")
    assert FULL in agent._build_system_prompt()
    _write(profile / tc.FULL_FILENAME, "## Your owner and their company\nNew owner text.")
    prompt = agent._build_system_prompt()
    assert "New owner text." in prompt and FULL not in prompt
    for name in (tc.FULL_FILENAME, tc.COMPACT_FILENAME):
        (profile / name).unlink()
    assert "Your owner" not in agent._build_system_prompt()


def _being_caps(stage: str) -> str:
    """CLAW_BEING_CAPS exactly as being_life.spawn_body stamps it."""
    from captain_claw.flight_deck import being_constitution as constitution

    return ",".join(sorted(constitution.capabilities(stage)))


@pytest.mark.parametrize("stage", ["egg", "infant", "child", "adolescent", "adult"])
@pytest.mark.parametrize("level", ["normal", "micro"])
def test_iskra_body_never_sees_the_owner_profile(monkeypatch, tmp_path, profile, stage, level):
    # Adolescent+ bodies hold agent_messaging, so the fleet is visible to
    # them — but the owner profile must still never reach a being's body.
    monkeypatch.setenv("CLAW_BEING_WORKER", "1")
    monkeypatch.setenv("CLAW_BEING_CAPS", _being_caps(stage))
    prompt = _agent(monkeypatch, tmp_path, level)._build_system_prompt()
    assert FULL not in prompt and COMPACT not in prompt


@pytest.mark.parametrize("value,expected", [
    ("1", True), ("true", True), ("YES", True), (" yes ", True),
    ("", False), ("0", False), ("no", False),
])
def test_is_being_body_reads_only_the_worker_marker(monkeypatch, value, expected):
    monkeypatch.setenv("CLAW_BEING_WORKER", value)
    monkeypatch.setenv("CLAW_BEING_CAPS", _being_caps("adult"))
    assert acm._is_being_body() is expected


@pytest.mark.parametrize("stage,hidden", [
    ("infant", True), ("child", True), ("adolescent", False), ("adult", False),
])
def test_fleet_visibility_still_follows_agent_messaging(monkeypatch, stage, hidden):
    # The tenant gate moved to _is_being_body(); fleet identity/peers keep
    # using _iskra_fleet_hidden(), which still keys off agent_messaging.
    monkeypatch.setenv("CLAW_BEING_WORKER", "1")
    monkeypatch.setenv("CLAW_BEING_CAPS", _being_caps(stage))
    assert acm._iskra_fleet_hidden() is hidden


def test_tenant_hidden_agent_never_sees_the_owner_profile(monkeypatch, tmp_path, profile):
    agent = _agent(monkeypatch, tmp_path, "normal")
    agent._tenant_hidden = True
    prompt = agent._build_system_prompt()
    assert FULL not in prompt and COMPACT not in prompt


async def test_botport_dispatch_agent_never_sees_the_owner_profile(monkeypatch, profile):
    from captain_claw.botport_client import BotPortClient
    from captain_claw.config import BotPortClientConfig
    from captain_claw.session import Session

    monkeypatch.setattr("captain_claw.tools.registry._registry", None)

    class _SM:
        async def create_session(self, name):
            return Session(id="bp-1", name=name)

    monkeypatch.setattr("captain_claw.botport_client.get_session_manager", lambda: _SM())
    client = BotPortClient(BotPortClientConfig(), provider=_Provider())
    agent = await client._spawn_dispatch_agent(
        concern_id="c" * 16, task="Summarise the report.", context={},
        from_instance="remote-deck", persona_hint="",
    )
    agent._build_skills_system_prompt_section = lambda: ""
    assert agent._tenant_hidden is True
    prompt = agent._build_system_prompt()
    assert FULL not in prompt and COMPACT not in prompt
    # Without the flag the same agent WOULD get the block — the flag is
    # what keeps a remote instance's task away from the owner's profile.
    agent._tenant_hidden = False
    prompt = agent._build_system_prompt()
    assert FULL in prompt or COMPACT in prompt


def test_public_scoped_agent_never_sees_the_owner_profile(monkeypatch, tmp_path, profile):
    agent = _agent(monkeypatch, tmp_path, "normal")
    agent._public_scoped = True
    prompt = agent._build_system_prompt()
    assert FULL not in prompt and COMPACT not in prompt


async def test_public_session_agents_are_flagged(monkeypatch):
    from captain_claw.web_server import WebServer

    server = WebServer.__new__(WebServer)     # skip __init__'s heavy wiring
    server._public_agents = {}
    server._public_agent_locks = {}
    server._public_active_ws = {}

    async def fake_build(session, send):
        return types.SimpleNamespace(session=session)

    server._build_scoped_agent = fake_build

    class _SM:
        async def load_session(self, sid):
            return types.SimpleNamespace(id=sid, name=f"pub-{sid}")

    monkeypatch.setattr("captain_claw.session.get_session_manager", lambda: _SM())
    agent = await server._get_public_agent("p1")
    assert agent._public_scoped is True


# ── fleet instructions reach the micro and nano templates ────────────


@pytest.mark.parametrize("name", ["micro_system_prompt.md", "nano_system_prompt.md"])
def test_eco_templates_have_the_fleet_instructions_placeholder(name):
    text = (_INSTRUCTIONS / name).read_text(encoding="utf-8")
    assert "{user_context_block}\n{fleet_instructions_block}\n" in text


@pytest.mark.parametrize("level", ["micro", "nano"])
def test_eco_templates_render_fleet_instructions(tmp_path, home, level):
    loader = InstructionLoader(
        base_dir=_INSTRUCTIONS, personal_dir=tmp_path / "personal",
        use_micro=level == "micro", use_nano=level == "nano",
    )
    out = loader.render("system_prompt.md", fleet_instructions_block="<<FLEET>>")
    assert "<<FLEET>>" in out and "{fleet_instructions_block}" not in out


@pytest.mark.parametrize("level", ["normal", "micro", "nano"])
def test_agent_prompt_carries_fleet_instructions(monkeypatch, tmp_path, home, level):
    agent = _agent(monkeypatch, tmp_path, level)
    agent._fleet_instructions = "Always answer in Croatian."
    prompt = agent._build_system_prompt()
    assert "## Fleet-Level Instructions" in prompt
    assert "Always answer in Croatian." in prompt


_CLIP_NOTE = "[truncated: fleet instructions clipped in nano mode]"


def test_nano_clips_long_fleet_instructions(monkeypatch, tmp_path, home):
    agent = _agent(monkeypatch, tmp_path, "nano")
    head = "ZQXHEADZQX "
    agent._fleet_instructions = head + "a" * 4000 + " ZQXTAILZQX"
    prompt = agent._build_system_prompt()
    assert head.strip() in prompt
    assert "ZQXTAILZQX" not in prompt
    assert _CLIP_NOTE in prompt
    limit = acm._NANO_FLEET_INSTRUCTIONS_MAX_CHARS
    body = prompt[prompt.index(head.strip()):prompt.index(_CLIP_NOTE)]
    assert len(body.rstrip("… ")) <= limit


def test_nano_keeps_fleet_instructions_at_the_limit(monkeypatch, tmp_path, home):
    agent = _agent(monkeypatch, tmp_path, "nano")
    text = "ZQXHEADZQX " + "a" * (acm._NANO_FLEET_INSTRUCTIONS_MAX_CHARS - 22) + " ZQXTAILZQX"
    assert len(text) == acm._NANO_FLEET_INSTRUCTIONS_MAX_CHARS
    agent._fleet_instructions = text
    prompt = agent._build_system_prompt()
    assert text in prompt and _CLIP_NOTE not in prompt


@pytest.mark.parametrize("level", ["normal", "micro"])
def test_long_fleet_instructions_are_not_clipped_above_nano(monkeypatch, tmp_path, home, level):
    agent = _agent(monkeypatch, tmp_path, level)
    text = "ZQXHEADZQX " + "a" * 4000 + " ZQXTAILZQX"
    agent._fleet_instructions = text
    prompt = agent._build_system_prompt()
    assert text in prompt and _CLIP_NOTE not in prompt
