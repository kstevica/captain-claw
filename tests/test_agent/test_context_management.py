import pytest

from captain_claw.agent import Agent
from captain_claw.config import get_config, set_config
from captain_claw.llm import LLMProvider, LLMResponse, Message, ToolDefinition
from captain_claw.session import Session
from captain_claw.tools.registry import ToolRegistry


class DummySessionManager:
    async def save_session(self, session: Session) -> None:
        return None


class TokenAwareProvider(LLMProvider):
    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
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
        # Simple deterministic tokenizer for tests.
        return len([part for part in text.split() if part]) or 1


def test_build_messages_prunes_history_to_context_budget():
    old_cfg = get_config().model_copy(deep=True)
    cfg = old_cfg.model_copy(deep=True)
    cfg.context.max_tokens = 25
    set_config(cfg)
    try:
        agent = Agent(provider=TokenAwareProvider())
        agent.session = Session(id="s1", name="default")

        for idx in range(8):
            agent.session.add_message("assistant", f"old message {idx} " + ("noise " * 3))
        agent.session.add_message("user", "latest question")

        messages = agent._build_messages()
        contents = [m.content for m in messages]

        assert messages[0].role == "system"
        assert "latest question" in contents
        assert agent.last_context_window["dropped_messages"] > 0
        assert agent.last_context_window["included_messages"] == 1
        assert agent.last_context_window["over_budget"] in (0, 1)
    finally:
        set_config(old_cfg)


def test_build_messages_populates_message_token_counts():
    agent = Agent(provider=TokenAwareProvider())
    agent.session = Session(id="s1", name="default")
    agent.session.add_message("user", "count me")
    agent.session.messages[-1].pop("token_count", None)

    agent._build_messages()

    assert isinstance(agent.session.messages[-1].get("token_count"), int)
    assert agent.session.messages[-1]["token_count"] > 0


@pytest.mark.asyncio
async def test_complete_persists_token_count_per_message():
    agent = Agent(provider=TokenAwareProvider())
    agent._initialized = True
    agent.session = Session(id="s1", name="default")
    agent.session_manager = DummySessionManager()
    agent.tools = ToolRegistry()

    result = await agent.complete("hello world")

    assert result == "ok"
    assert len(agent.session.messages) == 2
    assert all(isinstance(msg.get("token_count"), int) and msg["token_count"] > 0 for msg in agent.session.messages)
