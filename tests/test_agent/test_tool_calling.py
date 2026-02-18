from pathlib import Path

import pytest

from captain_claw.agent import Agent
from captain_claw.config import get_config, set_config
from captain_claw.exceptions import LLMAPIError
from captain_claw.llm import LLMProvider, LLMResponse, Message, ToolCall, ToolDefinition
from captain_claw.session import Session
from captain_claw.tools.write import WriteTool
from captain_claw.tools.registry import Tool, ToolRegistry, ToolResult


class DummyProvider(LLMProvider):
    def __init__(self):
        self.calls: list[dict] = []

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        self.calls.append({"messages": messages, "tools": tools})
        return LLMResponse(content="ok")

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
        return len(text)


class FailingProvider(LLMProvider):
    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        raise LLMAPIError("Ollama API error 500: Internal Server Error")

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
        return len(text)


class DummyTool(Tool):
    name = "dummy"
    description = "Dummy test tool"
    parameters = {
        "type": "object",
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
    }

    async def execute(self, **kwargs) -> ToolResult:
        return ToolResult(success=True, content="done")


class DummySessionManager:
    async def save_session(self, session: Session) -> None:
        return None


class TwoTurnProvider(LLMProvider):
    def __init__(self):
        self.call_count = 0
        self.messages_per_call: list[list[Message]] = []

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        self.call_count += 1
        self.messages_per_call.append(messages)
        if self.call_count == 1:
            return LLMResponse(
                content="",
                tool_calls=[ToolCall(id="c1", name="dummy", arguments={"value": "v"})],
            )
        if self.call_count == 2:
            raise LLMAPIError("Ollama API error 500: Internal Server Error")
        if self.call_count == 3:
            return LLMResponse(content="friendly first turn")
        if self.call_count == 4:
            return LLMResponse(content="second turn ok")
        return LLMResponse(content="unexpected")

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
        return len(text)


class SingleToolCallProvider(LLMProvider):
    def __init__(self):
        self.calls = 0
        self.calls_payloads: list[dict] = []

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        self.calls += 1
        self.calls_payloads.append({"messages": messages, "tools": tools})
        if self.calls == 1:
            return LLMResponse(
                content="",
                tool_calls=[ToolCall(id="c1", name="dummy", arguments={"value": "v"})],
            )
        return LLMResponse(content="Friendly: done")

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
        return len(text)


class ToolThenFinalProvider(LLMProvider):
    def __init__(self):
        self.calls = 0

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        self.calls += 1
        if self.calls == 1:
            return LLMResponse(
                content="",
                tool_calls=[ToolCall(id="c1", name="dummy", arguments={"value": "v"})],
            )
        return LLMResponse(content="final")

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
        return len(text)


class DummyWebFetchTool(Tool):
    name = "web_fetch"
    description = "Dummy web fetch"
    parameters = {
        "type": "object",
        "properties": {"url": {"type": "string"}},
        "required": ["url"],
    }

    async def execute(self, **kwargs) -> ToolResult:
        return ToolResult(
            success=True,
            content="[URL: https://www.hr]\n[Status: 200]\n[Mode: text]\n\nhr landing page content",
        )


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

    async def execute(self, **kwargs) -> ToolResult:
        self.commands.append(str(kwargs.get("command", "")))
        return ToolResult(success=True, content="script executed")


class CloudWebFetchThenSummaryProvider(LLMProvider):
    def __init__(self):
        self.calls = 0

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        self.calls += 1
        if self.calls == 1:
            return LLMResponse(
                content="",
                tool_calls=[ToolCall(id="wf1", name="web_fetch", arguments={"url": "https://www.hr"})],
            )
        return LLMResponse(content="Summary: www.hr is currently a minimal landing page.")

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
        return len(text)


class CodeBlockOnlyProvider(LLMProvider):
    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        return LLMResponse(
            content=(
                "Use this script:\n"
                "```python\n"
                "print('hello from generated script')\n"
                "```\n"
            )
        )

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
        return len(text)


@pytest.mark.asyncio
async def test_complete_sends_tool_definitions_to_provider():
    provider = DummyProvider()
    agent = Agent(provider=provider)
    agent._initialized = True
    agent.session = Session(id="s1", name="default")
    agent.session_manager = DummySessionManager()
    registry = ToolRegistry()
    registry.register(DummyTool())
    agent.tools = registry

    result = await agent.complete("say hi")

    assert result == "ok"
    assert provider.calls
    sent_tools = provider.calls[0]["tools"]
    assert sent_tools is not None
    assert sent_tools[0]["name"] == "dummy"


def test_build_messages_keeps_tool_metadata():
    provider = DummyProvider()
    agent = Agent(provider=provider)
    agent.session = Session(id="s1", name="default")
    agent.session.add_message(
        role="tool",
        content="command output",
        tool_call_id="call_123",
        tool_name="shell",
    )

    messages = agent._build_messages()
    tool_messages = [m for m in messages if m.role == "tool"]

    assert len(tool_messages) == 1
    assert tool_messages[0].tool_call_id == "call_123"
    assert tool_messages[0].tool_name == "shell"


@pytest.mark.asyncio
async def test_complete_ignores_historical_tool_messages_on_first_model_call():
    provider = DummyProvider()
    agent = Agent(provider=provider)
    agent._initialized = True
    agent.session = Session(id="s1", name="default")
    agent.session.add_message("user", "old request")
    agent.session.add_message("tool", "old tool output", tool_name="shell")
    agent.session.add_message("assistant", "old final answer")
    agent.session_manager = DummySessionManager()
    agent.tools = ToolRegistry()

    result = await agent.complete("new request")

    assert result == "ok"
    first_call_messages = provider.calls[0]["messages"]
    assert all(msg.role != "tool" for msg in first_call_messages)


@pytest.mark.asyncio
async def test_complete_does_not_return_stale_tool_output_on_fresh_500():
    provider = FailingProvider()
    agent = Agent(provider=provider)
    agent._initialized = True
    agent.session = Session(id="s1", name="default")
    agent.session.add_message("tool", "old tool output", tool_name="shell")
    agent.session_manager = DummySessionManager()
    agent.tools = ToolRegistry()

    with pytest.raises(LLMAPIError):
        await agent.complete("new request")


@pytest.mark.asyncio
async def test_two_turn_flow_does_not_reuse_previous_tool_output_after_500():
    old_cfg = get_config().model_copy(deep=True)
    cfg = old_cfg.model_copy(deep=True)
    cfg.model.provider = "ollama"
    cfg.model.model = "llama3.2"
    set_config(cfg)
    try:
        provider = TwoTurnProvider()
        agent = Agent(provider=provider)
        agent._initialized = True
        agent.session = Session(id="s1", name="default")
        agent.session_manager = DummySessionManager()
        registry = ToolRegistry()
        registry.register(DummyTool())
        agent.tools = registry

        first = await agent.complete("first request")
        second = await agent.complete("second request")

        assert first == "friendly first turn"
        assert second == "second turn ok"
        # Fourth call is the second turn's first model request.
        assert all(msg.role != "tool" for msg in provider.messages_per_call[3])
    finally:
        set_config(old_cfg)


@pytest.mark.asyncio
async def test_tool_output_callback_receives_raw_output():
    provider = ToolThenFinalProvider()
    outputs: list[dict] = []

    def cb(name: str, args: dict, output: str) -> None:
        outputs.append({"name": name, "args": args, "output": output})

    agent = Agent(provider=provider, tool_output_callback=cb)
    agent._initialized = True
    agent.session = Session(id="s1", name="default")
    agent.session_manager = DummySessionManager()
    registry = ToolRegistry()
    registry.register(DummyTool())
    agent.tools = registry

    result = await agent.complete("run tool")

    assert result == "final"
    assert outputs
    assert outputs[0]["name"] == "dummy"
    assert outputs[0]["args"] == {"value": "v"}
    assert outputs[0]["output"] == "done"
    tool_messages = [m for m in agent.session.messages if m.get("role") == "tool"]
    assert tool_messages[0]["tool_arguments"] == {"value": "v"}


@pytest.mark.asyncio
async def test_ollama_cloud_skips_tool_result_followup_call():
    old_cfg = get_config().model_copy(deep=True)
    cfg = old_cfg.model_copy(deep=True)
    cfg.model.provider = "ollama"
    cfg.model.model = "minimax-m2.5:cloud"
    set_config(cfg)
    try:
        provider = SingleToolCallProvider()
        agent = Agent(provider=provider)
        agent._initialized = True
        agent.session = Session(id="s1", name="default")
        agent.session_manager = DummySessionManager()
        registry = ToolRegistry()
        registry.register(DummyTool())
        agent.tools = registry

        result = await agent.complete("run tool")

        assert result == "Friendly: done"
        assert provider.calls == 2
        assert provider.calls_payloads[1]["tools"] is None
    finally:
        set_config(old_cfg)


@pytest.mark.asyncio
async def test_cloud_mode_auto_runs_write_when_user_requested_file_output(tmp_path: Path):
    old_cfg = get_config().model_copy(deep=True)
    cfg = old_cfg.model_copy(deep=True)
    cfg.model.provider = "ollama"
    cfg.model.model = "minimax-m2.5:cloud"
    set_config(cfg)
    try:
        provider = CloudWebFetchThenSummaryProvider()
        outputs: list[str] = []

        def cb(name: str, args: dict, output: str) -> None:
            outputs.append(name)

        agent = Agent(provider=provider, tool_output_callback=cb)
        agent._initialized = True
        agent.session = Session(id="s1", name="default")
        agent.session_manager = DummySessionManager()
        registry = ToolRegistry(base_path=tmp_path)
        registry.register(DummyWebFetchTool())
        registry.register(WriteTool())
        agent.tools = registry

        result = await agent.complete(
            "fetch website www.hr, summarize the content and write it to file wwwhr.txt"
        )

        expected = tmp_path / "saved" / "tmp" / "s1" / "wwwhr.txt"
        assert expected.exists()
        saved_content = expected.read_text(encoding="utf-8")
        assert "Summary: www.hr is currently a minimal landing page." in saved_content
        assert "Written" in result
        assert "write" in outputs
        assert any(
            msg.get("role") == "tool" and msg.get("tool_name") == "write"
            for msg in agent.session.messages
        )
    finally:
        set_config(old_cfg)


@pytest.mark.asyncio
async def test_explicit_generate_script_forces_write_and_run_from_scripts_dir(tmp_path: Path):
    provider = CodeBlockOnlyProvider()
    outputs: list[dict[str, str]] = []

    def cb(name: str, args: dict, output: str) -> None:
        outputs.append({"name": name, "output": output, "args": str(args)})

    agent = Agent(provider=provider, tool_output_callback=cb)
    agent._initialized = True
    agent.session = Session(id="s1", name="default")
    agent.session_manager = DummySessionManager()
    registry = ToolRegistry(base_path=tmp_path)
    shell_tool = DummyShellTool()
    registry.register(WriteTool())
    registry.register(shell_tool)
    agent.tools = registry

    result = await agent.complete("generate script that prints hello")

    scripts_dir = tmp_path / "saved" / "scripts" / "s1"
    created_files = list(scripts_dir.glob("generated_script_*.py"))
    assert created_files
    assert "hello from generated script" in created_files[0].read_text(encoding="utf-8")
    assert any(entry["name"] == "write" for entry in outputs)
    assert any(entry["name"] == "shell" for entry in outputs)
    assert shell_tool.commands
    assert str(scripts_dir) in shell_tool.commands[0]
    assert "&& python3" in shell_tool.commands[0]
    assert "Script saved and executed from" in result
