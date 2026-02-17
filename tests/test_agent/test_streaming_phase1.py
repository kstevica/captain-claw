import pytest

from captain_claw.agent import Agent
from captain_claw.llm import LLMProvider, LLMResponse, Message, ToolDefinition
from captain_claw.session import Session
from captain_claw.tools.registry import Tool, ToolRegistry, ToolResult


class DummySessionManager:
    async def save_session(self, session: Session) -> None:
        return None


class DummyTool(Tool):
    name = "dummy"
    description = "Dummy test tool"
    parameters = {"type": "object", "properties": {}, "required": []}

    async def execute(self, **kwargs) -> ToolResult:
        return ToolResult(success=True, content="ok")


class StreamProvider(LLMProvider):
    def __init__(self):
        self.complete_calls = 0
        self.streaming_calls = 0

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        self.complete_calls += 1
        return LLMResponse(content="hello")

    async def complete_streaming(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ):
        self.streaming_calls += 1
        yield "ab"
        yield "cd"

    def count_tokens(self, text: str) -> int:
        return len(text)


@pytest.mark.asyncio
async def test_stream_uses_complete_path_when_tools_registered():
    provider = StreamProvider()
    agent = Agent(provider=provider)
    agent._initialized = True
    agent.session = Session(id="s1", name="default")
    agent.session_manager = DummySessionManager()
    registry = ToolRegistry()
    registry.register(DummyTool())
    agent.tools = registry

    chunks = [chunk async for chunk in agent.stream("hello")]

    assert "".join(chunks) == "hello"
    assert provider.complete_calls == 1
    assert provider.streaming_calls == 0


@pytest.mark.asyncio
async def test_stream_uses_provider_streaming_when_no_tools():
    provider = StreamProvider()
    agent = Agent(provider=provider)
    agent._initialized = True
    agent.session = Session(id="s1", name="default")
    agent.session_manager = DummySessionManager()
    agent.tools = ToolRegistry()

    chunks = [chunk async for chunk in agent.stream("hello")]

    assert "".join(chunks) == "abcd"
    assert provider.complete_calls == 0
    assert provider.streaming_calls == 1
    assert agent.last_usage["total_tokens"] > 0
