import asyncio

import pytest

from captain_claw.config import get_config, set_config
from captain_claw.exceptions import ToolBlockedError, ToolExecutionError
from captain_claw.tools.registry import Tool, ToolPolicy, ToolRegistry, ToolResult


class DummyShellTool(Tool):
    name = "shell"
    description = "Dummy shell"
    parameters = {
        "type": "object",
        "properties": {"command": {"type": "string"}},
        "required": ["command"],
    }

    def __init__(self):
        self.commands: list[str] = []

    async def execute(self, **kwargs):
        self.commands.append(str(kwargs.get("command", "")))
        return ToolResult(success=True, content="ok")


class DummyTool(Tool):
    def __init__(self, name: str):
        self.name = name
        self.description = f"Dummy tool {name}"
        self.parameters = {
            "type": "object",
            "properties": {},
            "required": [],
        }

    async def execute(self, **kwargs):
        return ToolResult(success=True, content="ok")


class SlowTool(Tool):
    name = "slow"
    description = "Slow"
    parameters = {
        "type": "object",
        "properties": {},
        "required": [],
    }
    timeout_seconds = 1.0

    async def execute(self, **kwargs):
        await asyncio.sleep(2.0)
        return ToolResult(success=True, content="done")


class CancellableTool(Tool):
    name = "cancellable"
    description = "Cancellable"
    parameters = {
        "type": "object",
        "properties": {},
        "required": [],
    }
    timeout_seconds = 20.0

    def __init__(self):
        self.cancelled = False

    async def execute(self, **kwargs):
        try:
            await asyncio.sleep(10.0)
            return ToolResult(success=True, content="done")
        except asyncio.CancelledError:
            self.cancelled = True
            raise


@pytest.mark.asyncio
async def test_tool_policy_chain_allows_task_level_reenable_after_parent_deny():
    registry = ToolRegistry()
    registry.register(DummyShellTool())
    registry.register(DummyTool("web_search"))
    registry.register(DummyTool("web_fetch"))

    registry.set_global_policy(ToolPolicy(deny=["shell"]))
    session_policy = ToolPolicy(allow=["web_search"])
    task_policy = ToolPolicy(also_allow=["shell"])

    names = set(
        registry.list_tools(
            session_policy=session_policy,
            task_policy=task_policy,
        )
    )
    assert names == {"web_search", "shell"}

    result = await registry.execute(
        "shell",
        {"command": "echo ok"},
        session_policy=session_policy,
        task_policy=task_policy,
    )
    assert result.success is True


@pytest.mark.asyncio
async def test_tool_policy_chain_blocks_execution_when_denied():
    registry = ToolRegistry()
    registry.register(DummyShellTool())

    with pytest.raises(ToolBlockedError, match="Blocked by tool policy chain"):
        await registry.execute(
            "shell",
            {"command": "echo ok"},
            task_policy=ToolPolicy(deny=["shell"]),
        )


@pytest.mark.asyncio
async def test_shell_blocking_uses_parsed_base_command_not_substring_matches():
    old_cfg = get_config().model_copy(deep=True)
    cfg = old_cfg.model_copy(deep=True)
    cfg.tools.shell.blocked = ["rm"]
    cfg.tools.shell.allow_patterns = []
    cfg.tools.shell.deny_patterns = []
    cfg.tools.shell.default_policy = "allow"
    set_config(cfg)
    try:
        registry = ToolRegistry()
        shell = DummyShellTool()
        registry.register(shell)

        allowed = await registry.execute("shell", {"command": "grep format README.md"})
        assert allowed.success is True

        with pytest.raises(ToolBlockedError, match="Command matches blocked pattern: rm"):
            await registry.execute("shell", {"command": "echo ok && rm -rf /tmp/demo"})
    finally:
        set_config(old_cfg)


@pytest.mark.asyncio
async def test_registry_uses_tool_level_timeout_seconds():
    registry = ToolRegistry()
    registry.register(SlowTool())

    with pytest.raises(ToolExecutionError, match="timed out"):
        await registry.execute("slow", {})


@pytest.mark.asyncio
async def test_registry_abort_event_cancels_running_tool_execution():
    registry = ToolRegistry()
    tool = CancellableTool()
    registry.register(tool)

    abort_event = asyncio.Event()
    execution = asyncio.create_task(registry.execute("cancellable", {}, abort_event=abort_event))
    await asyncio.sleep(0.05)
    abort_event.set()

    with pytest.raises(ToolExecutionError, match="aborted"):
        await execution
    assert tool.cancelled is True


@pytest.mark.asyncio
async def test_shell_exec_policy_deny_pattern_blocks_command():
    old_cfg = get_config().model_copy(deep=True)
    cfg = old_cfg.model_copy(deep=True)
    cfg.tools.shell.blocked = []
    cfg.tools.shell.allow_patterns = []
    cfg.tools.shell.deny_patterns = ["echo blocked*"]
    cfg.tools.shell.default_policy = "allow"
    set_config(cfg)
    try:
        registry = ToolRegistry()
        registry.register(DummyShellTool())
        with pytest.raises(ToolBlockedError, match="deny pattern"):
            await registry.execute("shell", {"command": "echo blocked now"})
    finally:
        set_config(old_cfg)


@pytest.mark.asyncio
async def test_shell_exec_policy_allow_pattern_overrides_default_deny():
    old_cfg = get_config().model_copy(deep=True)
    cfg = old_cfg.model_copy(deep=True)
    cfg.tools.shell.blocked = []
    cfg.tools.shell.allow_patterns = ["echo *"]
    cfg.tools.shell.deny_patterns = []
    cfg.tools.shell.default_policy = "deny"
    set_config(cfg)
    try:
        registry = ToolRegistry()
        registry.register(DummyShellTool())
        allowed = await registry.execute("shell", {"command": "echo hello"})
        assert allowed.success is True

        with pytest.raises(ToolBlockedError, match="Default shell execution policy denies command"):
            await registry.execute("shell", {"command": "printf 'nope'"})
    finally:
        set_config(old_cfg)


@pytest.mark.asyncio
async def test_shell_exec_policy_ask_uses_approval_callback():
    old_cfg = get_config().model_copy(deep=True)
    cfg = old_cfg.model_copy(deep=True)
    cfg.tools.shell.blocked = []
    cfg.tools.shell.allow_patterns = []
    cfg.tools.shell.deny_patterns = []
    cfg.tools.shell.default_policy = "ask"
    set_config(cfg)
    try:
        registry = ToolRegistry()
        registry.register(DummyShellTool())
        prompts: list[str] = []

        def deny_callback(question: str) -> bool:
            prompts.append(question)
            return False

        registry.set_approval_callback(deny_callback)
        with pytest.raises(ToolBlockedError, match="approval policy"):
            await registry.execute("shell", {"command": "echo hello"})

        registry.set_approval_callback(lambda _: True)
        result = await registry.execute("shell", {"command": "echo hello"})
        assert result.success is True
        assert prompts
    finally:
        set_config(old_cfg)
