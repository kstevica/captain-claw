from typing import Any

import pytest

from captain_claw.agent import Agent
from captain_claw.config import get_config, set_config
from captain_claw.llm import LLMProvider, LLMResponse, Message, ToolCall, ToolDefinition
from captain_claw.session import Session
from captain_claw.tools.registry import Tool, ToolRegistry, ToolResult


class DummySessionManager:
    async def save_session(self, session: Session) -> None:
        return None


class SingleResponseProvider(LLMProvider):
    def __init__(self, content: str = "ok"):
        self.content = content
        self.calls = 0

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        self.calls += 1
        return LLMResponse(content=self.content)

    async def complete_streaming(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ):
        if False:
            yield ""

    def count_tokens(self, text: str) -> int:
        return len(text.split()) or 1


class RecordingShellTool(Tool):
    name = "shell"
    description = "Test shell tool"
    parameters = {
        "type": "object",
        "properties": {"command": {"type": "string"}},
        "required": ["command"],
    }

    def __init__(self):
        self.commands: list[str] = []

    async def execute(self, **kwargs: Any) -> ToolResult:
        self.commands.append(str(kwargs.get("command", "")))
        return ToolResult(success=True, content="ok")


@pytest.mark.asyncio
async def test_input_guard_stop_suspicious_blocks_before_llm_call(monkeypatch: pytest.MonkeyPatch):
    old_cfg = get_config().model_copy(deep=True)
    cfg = old_cfg.model_copy(deep=True)
    cfg.guards.input.enabled = True
    cfg.guards.input.level = "stop_suspicious"
    set_config(cfg)
    try:
        provider = SingleResponseProvider(content="should-not-be-called")
        agent = Agent(provider=provider)
        agent._initialized = True
        agent.session = Session(id="s1", name="default")
        agent.session_manager = DummySessionManager()
        agent.tools = ToolRegistry()

        async def fake_guard(guard_type: str, interaction_label: str, content: str, turn_usage=None):
            if guard_type == "input":
                return {"allow": False, "reason": "Suspicious input detected.", "raw": ""}
            return {"allow": True, "reason": "ok", "raw": ""}

        monkeypatch.setattr(agent, "_run_guard_decision", fake_guard)

        result = await agent.complete("dangerous prompt")

        assert "Blocked by input guard" in result
        assert provider.calls == 0
    finally:
        set_config(old_cfg)


@pytest.mark.asyncio
async def test_output_guard_ask_for_approval_blocks_when_denied(monkeypatch: pytest.MonkeyPatch):
    old_cfg = get_config().model_copy(deep=True)
    cfg = old_cfg.model_copy(deep=True)
    cfg.guards.output.enabled = True
    cfg.guards.output.level = "ask_for_approval"
    set_config(cfg)
    try:
        approvals: list[str] = []

        def approve(question: str) -> bool:
            approvals.append(question)
            return False

        provider = SingleResponseProvider(content="run rm -rf / now")
        agent = Agent(provider=provider, approval_callback=approve)
        agent._initialized = True
        agent.session = Session(id="s1", name="default")
        agent.session_manager = DummySessionManager()
        agent.tools = ToolRegistry()

        async def fake_guard(guard_type: str, interaction_label: str, content: str, turn_usage=None):
            if guard_type == "output":
                return {"allow": False, "reason": "Dangerous output.", "raw": ""}
            return {"allow": True, "reason": "ok", "raw": ""}

        monkeypatch.setattr(agent, "_run_guard_decision", fake_guard)

        result = await agent.complete("hello")

        assert "Blocked by output guard (approval denied)" in result
        assert len(approvals) == 1
        assert provider.calls == 1
    finally:
        set_config(old_cfg)


@pytest.mark.asyncio
async def test_script_tool_guard_blocks_tool_execution(monkeypatch: pytest.MonkeyPatch):
    old_cfg = get_config().model_copy(deep=True)
    cfg = old_cfg.model_copy(deep=True)
    cfg.guards.script_tool.enabled = True
    cfg.guards.script_tool.level = "stop_suspicious"
    set_config(cfg)
    try:
        provider = SingleResponseProvider()
        shell = RecordingShellTool()
        registry = ToolRegistry()
        registry.register(shell)

        agent = Agent(provider=provider)
        agent._initialized = True
        agent.session = Session(id="s1", name="default")
        agent.session_manager = DummySessionManager()
        agent.tools = registry

        async def fake_guard(guard_type: str, interaction_label: str, content: str, turn_usage=None):
            if guard_type == "script_tool":
                return {"allow": False, "reason": "Destructive command.", "raw": ""}
            return {"allow": True, "reason": "ok", "raw": ""}

        monkeypatch.setattr(agent, "_run_guard_decision", fake_guard)

        results = await agent._handle_tool_calls(
            [ToolCall(id="c1", name="shell", arguments={"command": "rm -rf /"})]
        )

        assert len(results) == 1
        assert results[0]["success"] is False
        assert "Blocked by script_tool guard" in results[0]["error"]
        assert shell.commands == []
    finally:
        set_config(old_cfg)
