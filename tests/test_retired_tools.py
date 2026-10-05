"""Retired tools stay retired — the gws (Google Workspace CLI) tool above all.

Google goes through google_drive / google_calendar / google_mail only: one
identity path, and one Gmail send gate (gws ``raw`` could send mail with the
owner's token around it). An old config.yaml, archetype or prompt that still
names gws must not bring it back — no agent may register, see, be steered to
or shell out to it.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

import captain_claw.agent_context_mixin as acm
from captain_claw.config import RETIRED_TOOLS, Config, ToolsConfig, without_retired_tools
from captain_claw.instructions import InstructionLoader
from captain_claw.llm import LLMProvider, LLMResponse
from captain_claw.tools.registry import Tool, ToolRegistry, ToolResult

_INSTRUCTIONS = Path(acm.__file__).resolve().parent / "instructions"
_GOOGLE_TOOLS = ("google_drive", "google_calendar", "google_mail")
_GWS = re.compile(r"\bgws\b|workspace cli", re.I)


# ── config: the name is stripped wherever a tool list is loaded ──────


def test_gws_is_retired():
    assert "gws" in RETIRED_TOOLS
    assert without_retired_tools(["shell", "gws", "read"]) == ["shell", "read"]


def test_tools_config_strips_gws_and_keeps_order():
    enabled = ToolsConfig(enabled=["shell", "gws", "read", "google_drive"]).enabled
    assert "gws" not in enabled
    assert enabled[:3] == ["shell", "read", "google_drive"]
    assert "gws" not in ToolsConfig().enabled
    assert not hasattr(ToolsConfig(), "gws")


def test_old_yaml_with_a_gws_block_still_loads_without_gws(tmp_path):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(
        "tools:\n"
        "  enabled: [shell, gws, read]\n"
        "  gws:\n"
        "    binary_path: /usr/local/bin/gws\n",
        encoding="utf-8",
    )
    cfg = Config.from_yaml(cfg_file)
    assert "gws" not in cfg.tools.enabled
    assert {"shell", "read"} <= set(cfg.tools.enabled)
    assert not hasattr(cfg.tools, "gws")


# ── the tool itself is gone ──────────────────────────────────────────


@pytest.mark.parametrize("module", [
    "captain_claw.tools.gws",
    "captain_claw.tools._gws_runtime",
    "captain_claw.tools._gws_drive",
    "captain_claw.tools._gws_docs",
    "captain_claw.tools._gws_calendar",
])
def test_gws_modules_are_not_importable(module):
    assert importlib.util.find_spec(module) is None


def test_tools_package_exports_no_gws_tool():
    import captain_claw.tools as tools

    assert not hasattr(tools, "GwsTool")
    assert "GwsTool" not in tools.__all__


def test_builtin_tool_map_has_no_gws():
    builtin = acm.AgentContextMixin._BUILTIN_TOOL_MAP
    assert "gws" not in builtin
    assert not any("gws" in names for names in builtin.values())
    for name in _GOOGLE_TOOLS:
        assert builtin[name] == [name]


class _Recorder:
    """Stands in for the agent: records what _register_default_tools registers."""

    def __init__(self):
        self.tools = ToolRegistry()

    def _register_plugin_tools(self):
        pass


def test_registration_skips_a_retired_name_even_if_the_list_was_mutated(monkeypatch):
    cfg = Config()
    # Bypass validation, as a runtime mutation of the loaded list would.
    cfg.tools.enabled = ["gws", "read", "google_drive"]
    monkeypatch.setattr(acm, "get_config", lambda: cfg)
    agent = _Recorder()
    acm.AgentContextMixin._register_default_tools(agent)
    assert not agent.tools.has_tool("gws")
    assert agent.tools.has_tool("read") and agent.tools.has_tool("google_drive")


# ── prompts: google_* are described, gws is not ──────────────────────


@pytest.mark.parametrize("descs", [
    acm._TOOL_PROMPT_DESCRIPTIONS,
    acm._TOOL_PROMPT_DESCRIPTIONS_MICRO,
])
def test_tool_descriptions_cover_google_tools_and_not_gws(descs):
    assert "gws" not in descs
    for name in _GOOGLE_TOOLS:
        assert descs.get(name), name
        assert not _GWS.search(descs[name]), name


def test_no_instruction_file_mentions_gws_except_the_retired_note():
    hits = []
    for path in sorted(_INSTRUCTIONS.rglob("*")):
        if not path.is_file():
            continue
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not _GWS.search(line):
                continue
            if path.name in ("section_google.md", "micro_section_google.md") and "retired" in line:
                continue
            hits.append(f"{path.name}:{n}: {line.strip()[:120]}")
    assert hits == []
    assert not (_INSTRUCTIONS / "section_gws.md").exists()
    assert not (_INSTRUCTIONS / "micro_section_gws.md").exists()


def test_no_template_has_the_old_placeholder():
    for path in _INSTRUCTIONS.glob("*.md"):
        assert "{gws_block}" not in path.read_text(encoding="utf-8"), path.name
    for name in ("system_prompt.md", "micro_system_prompt.md"):
        assert "{google_block}" in (_INSTRUCTIONS / name).read_text(encoding="utf-8"), name


@pytest.mark.parametrize("micro", [False, True])
def test_system_templates_render_google_block(tmp_path, micro):
    loader = InstructionLoader(
        base_dir=_INSTRUCTIONS, personal_dir=tmp_path, use_micro=micro, use_nano=False,
    )
    out = loader.render("system_prompt.md", google_block="<<GOOGLE>>")
    assert "<<GOOGLE>>" in out
    assert "{google_block}" not in out and "{gws_block}" not in out
    section = loader.load("section_google.md")  # micro resolves micro_section_google.md
    assert "google_drive" in section and "Connections → Google" in section


class _Provider(LLMProvider):
    async def complete(self, messages, tools=None, temperature=None, max_tokens=None):
        return LLMResponse(content="ok")

    async def complete_streaming(self, messages, tools=None, temperature=None, max_tokens=None):
        if False:
            yield ""

    def count_tokens(self, text: str) -> int:
        return len(text.split()) or 1


@pytest.mark.parametrize("micro", [False, True])
def test_agent_prompt_has_the_google_section_only_with_a_google_tool(
    monkeypatch, tmp_path, micro,
):
    from captain_claw.agent import Agent
    from captain_claw.session import Session
    from captain_claw.tools import GoogleDriveTool

    # A fresh global registry: the agent's tools must be only what we add here.
    monkeypatch.setattr("captain_claw.tools.registry._registry", None)
    agent = Agent(provider=_Provider())
    agent.session = Session(id="s1", name="retired-tools")
    agent.instructions = InstructionLoader(
        base_dir=_INSTRUCTIONS, personal_dir=tmp_path, use_micro=micro, use_nano=False,
    )
    marker = "Google (Drive/Docs" if micro else "Google Workspace (Drive"

    without = agent._build_system_prompt()
    assert marker not in without
    assert "{google_block}" not in without and "{gws_block}" not in without

    agent.tools.register(GoogleDriveTool(), metadata={"requires_google": True})
    agent.instructions._cache.clear()
    prompt = agent._build_system_prompt()
    assert marker in prompt
    assert "{google_block}" not in prompt and "{gws_block}" not in prompt
    # The only gws mention left is the note that it is retired.
    for line in prompt.splitlines():
        if _GWS.search(line):
            assert "retired" in line, line


# ── BotPort's decomposer reads CC's raw YAML: filters on its own ─────


def test_botport_decomposer_never_advertises_gws(monkeypatch):
    from botport.swarm import decomposer

    assert "gws" not in decomposer._CC_TOOL_DESCRIPTIONS
    for name in _GOOGLE_TOOLS:
        assert decomposer._CC_TOOL_DESCRIPTIONS.get(name), name
    assert decomposer._CC_RETIRED_TOOLS == RETIRED_TOOLS

    monkeypatch.setattr(
        decomposer, "_load_cc_config",
        lambda: {"tools": {"enabled": ["shell", "gws", "google_drive"]}},
    )
    assert decomposer.get_cc_enabled_tools() == ["shell", "google_drive"]
    ctx = decomposer._build_tools_context()
    assert "google_drive" in ctx and "gws" not in ctx

    monkeypatch.setattr(decomposer, "_load_cc_config", lambda: {})
    assert "gws" not in decomposer.get_cc_enabled_tools()


@pytest.mark.parametrize("registered", [True, False])
def test_read_folders_steer_points_at_google_drive_only_when_it_is_registered(
    monkeypatch, tmp_path, registered,
):
    from captain_claw.agent import Agent
    from captain_claw.config import GDriveFolderEntry, get_config, set_config
    from captain_claw.session import Session
    from captain_claw.tools import GoogleDriveTool

    old_cfg = get_config()
    cfg = old_cfg.model_copy(deep=True)
    cfg.tools.read.gdrive_folders = [GDriveFolderEntry(id="FOLDER123", name="Reports")]
    set_config(cfg)
    try:
        monkeypatch.setattr("captain_claw.tools.registry._registry", None)
        agent = Agent(provider=_Provider())
        agent.session = Session(id="s1", name="retired-tools")
        agent.instructions = InstructionLoader(
            base_dir=_INSTRUCTIONS, personal_dir=tmp_path, use_micro=False, use_nano=False,
        )
        if registered:
            agent.tools.register(GoogleDriveTool(), metadata={"requires_google": True})
        prompt = agent._build_system_prompt()
    finally:
        set_config(old_cfg)

    if registered:
        assert "ALWAYS use the google_drive tool" in prompt
        assert "Reports (folder_id: FOLDER123)" in prompt
    else:
        assert "FOLDER123" not in prompt
    assert "gws tool" not in prompt


# ── loop heuristics and routing name the native tools, never gws ─────


@pytest.mark.parametrize("message,tool", [
    ("check my gmail inbox", "google_mail"),
    ("what is on my calendar tomorrow", "google_calendar"),
    ("list the files in my google drive", "google_drive"),
])
def test_eco_intent_routing_never_selects_gws(message, tool):
    from captain_claw.agent_orchestration_mixin import _eco_select_tools_by_intent

    selected = _eco_select_tools_by_intent(message)
    assert tool in selected and "gws" not in selected


def _set_literal(src: str, name: str) -> str:
    block = src[src.index(f"{name} = "):]
    return block[:block.index("}") + 1]


def test_tool_loop_sets_name_the_google_tools_not_gws():
    import inspect

    from captain_claw import agent_orchestration_mixin, agent_tool_loop_mixin

    loop_src = inspect.getsource(agent_tool_loop_mixin)
    for name in ("_STATEFUL_TOOLS", "_DATA_FETCH_TOOLS"):
        literal = _set_literal(loop_src, name)
        assert '"gws"' not in literal, name
        for tool in _GOOGLE_TOOLS:
            assert f'"{tool}"' in literal, (name, tool)

    orch_src = inspect.getsource(agent_orchestration_mixin)
    script_tools = _set_literal(orch_src, "_SCRIPT_MODE_TOOLS")
    for tool in _GOOGLE_TOOLS:
        assert f'"{tool}"' in script_tools, tool
    # Script-only mode steers Google work to the native tools, never the CLI.
    assert "['gws'" not in orch_src and '"gws"' not in orch_src


# ── duplicate guard: google_* reads get headroom, sends/creates do not ─


class _CountingGoogleTool(Tool):
    """A google_* stand-in that counts how often each action really ran."""

    parameters = {"type": "object", "properties": {}, "required": ["action"]}

    def __init__(self, name: str) -> None:
        self.name = name
        self.description = name
        self.ran: list[str] = []

    async def execute(self, action: str, **kwargs) -> ToolResult:
        self.ran.append(action)
        return ToolResult(success=True, content=f"{action} done")


class _NoSave:
    async def save_session(self, session) -> None:
        return None


@pytest.fixture
def _dup_limit_one():
    """Pin the shipped default (tools.duplicate_call_max = 1)."""
    from captain_claw.config import get_config, set_config

    old_cfg = get_config()
    cfg = old_cfg.model_copy(deep=True)
    cfg.tools.duplicate_call_max = 1
    set_config(cfg)
    yield
    set_config(old_cfg)


def _loop_agent(tmp_path, tool):
    from captain_claw.agent import Agent
    from captain_claw.session import Session

    agent = Agent(provider=_Provider())
    agent._initialized = True
    agent.session = Session(id="s1", name="dup-guard")
    agent.session_manager = _NoSave()
    registry = ToolRegistry(base_path=tmp_path)
    registry.register(tool)
    agent.tools = registry
    return agent


@pytest.mark.parametrize("tool_name,args", [
    ("google_mail", {"action": "send", "to": "a@example.com", "subject": "Hi", "body": "x"}),
    ("google_mail", {"action": "send_draft", "draft_id": "d1"}),
    ("google_calendar", {"action": "create_event", "summary": "Sync", "start": "2026-10-06T10:00"}),
    ("google_drive", {"action": "create", "name": "Notes", "content": "x"}),
])
async def test_identical_google_write_runs_once_per_turn(_dup_limit_one, tmp_path, tool_name, args):
    from captain_claw.llm import ToolCall

    tool = _CountingGoogleTool(tool_name)
    agent = _loop_agent(tmp_path, tool)
    calls = [ToolCall(id=f"c{i}", name=tool_name, arguments=dict(args)) for i in range(3)]

    results = await agent._handle_tool_calls(calls)

    # The second identical send/create is a second email/event/file — blocked.
    assert tool.ran == [args["action"]]
    blocked = [r for r in results if "DUPLICATE CALL BLOCKED" in str(r.get("content", ""))]
    assert len(blocked) == 2
    assert "would send or create it a second time" in blocked[0]["content"]


async def test_identical_google_reads_keep_the_stateful_headroom(_dup_limit_one, tmp_path):
    from captain_claw.llm import ToolCall

    tool = _CountingGoogleTool("google_calendar")
    agent = _loop_agent(tmp_path, tool)
    read = {"action": "list_events", "time_min": "2026-10-06"}
    # list → (write) → identical list to verify is the normal pattern.
    calls = [ToolCall(id=f"c{i}", name="google_calendar", arguments=dict(read)) for i in range(4)]

    results = await agent._handle_tool_calls(calls)

    assert tool.ran == ["list_events"] * 3
    assert "DUPLICATE CALL BLOCKED" in str(results[-1].get("content", ""))


async def test_a_failed_google_send_can_be_retried(_dup_limit_one, tmp_path):
    from captain_claw.llm import ToolCall

    tool = _CountingGoogleTool("google_mail")
    outcomes = iter([False, True])

    async def _flaky(action, **kwargs):
        tool.ran.append(action)
        ok = next(outcomes)
        return ToolResult(success=ok, content="sent" if ok else "", error=None if ok else "503")

    tool.execute = _flaky
    agent = _loop_agent(tmp_path, tool)
    send = {"action": "send", "to": "a@example.com", "subject": "Hi", "body": "x"}

    await agent._handle_tool_calls(
        [ToolCall(id=f"c{i}", name="google_mail", arguments=dict(send)) for i in range(2)],
    )

    # The failure rolled its count back, so the retry was not blocked.
    assert tool.ran == ["send", "send"]
