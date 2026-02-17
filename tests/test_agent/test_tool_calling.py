import pytest

from captain_claw.agent import Agent
from captain_claw.exceptions import LLMAPIError
from captain_claw.llm import LLMProvider, LLMResponse, Message, ToolCall, ToolDefinition
from captain_claw.session import Session
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

    assert first == "Tool executed:\ndone"
    assert second == "second turn ok"
    # Third call is the second turn's first model request.
    assert all(msg.role != "tool" for msg in provider.messages_per_call[2])
