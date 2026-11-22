"""The front of a request stays the same from turn to turn and restart to restart.

cohort001 (2026-11): the first call of a turn found only ~2,000 of ~100,000
tokens in DeepSeek's prompt cache about half the time, even a minute after the
last one. Two causes, both early in every request:

* the always-on tools were appended to an agent's tool list in a set's order,
  which changes with every process (hash seed) — each agent restart (idle
  sleep, wake for a scheduled job, deploy) reordered the tool definitions and
  the system prompt's tool list;
* the turn's system prompt (which names the connected tools) was built before
  the connection flags were refreshed, so whether it named Google's tools
  depended on how old the last check was.
"""

from __future__ import annotations

import os
import subprocess
import sys
from typing import Any

import pytest

from captain_claw.agent import Agent
from captain_claw.config import ToolsConfig
from captain_claw.llm import LLMProvider, LLMResponse, Message, ToolDefinition
from captain_claw.session import Session
from captain_claw.tools.registry import Tool, ToolRegistry, ToolResult

_ORDER = ("from captain_claw.config import ToolsConfig; "
          "print(','.join(ToolsConfig(enabled=['shell', 'web_search']).enabled))")


def test_always_enabled_tools_append_in_one_order_whatever_the_hash_seed():
    orders = set()
    for seed in ("1", "2", "3", "4"):
        env = {**os.environ, "PYTHONHASHSEED": seed}
        out = subprocess.run([sys.executable, "-c", _ORDER], env=env, capture_output=True,
                             text=True, check=True, cwd=os.getcwd()).stdout.strip().splitlines()[-1]
        orders.add(out)
    assert len(orders) == 1, orders
    first = orders.pop().split(",")
    assert first[:2] == ["shell", "web_search"]              # the agent's own list first
    assert first[2:] == [t for t in ToolsConfig._ALWAYS_ENABLED.default if t not in ("shell", "web_search")]


class _Read(Tool):
    name = "read"
    description = "Read a file."
    parameters = {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}

    async def execute(self, **kwargs: Any) -> ToolResult:
        return ToolResult(success=True, content="x")


class _Provider(LLMProvider):
    async def complete(self, messages: list[Message], tools: list[ToolDefinition] | None = None,
                       temperature: float | None = None, max_tokens: int | None = None) -> LLMResponse:
        return LLMResponse(content="Done.")

    async def complete_streaming(self, *a, **kw):
        if False:
            yield ""

    def count_tokens(self, text: str) -> int:
        return len(text)


class _NoSave:
    async def save_session(self, session: Session) -> None:
        return None


@pytest.mark.asyncio
async def test_connection_flags_are_fresh_before_the_turns_prompt_is_built(monkeypatch):
    from captain_claw import google_oauth_manager

    agent = Agent(provider=_Provider())
    agent._initialized = True
    agent.session = Session(id="pc1", name="default")
    agent.session_manager = _NoSave()
    monkeypatch.setattr(agent, "_supports_tool_result_followup", lambda: True)
    registry = ToolRegistry()
    registry.register(_Read())
    agent.tools = registry
    order: list[str] = []

    async def is_connected(self) -> bool:
        order.append("flags")
        return True

    monkeypatch.setattr(google_oauth_manager.GoogleOAuthManager, "is_connected", is_connected)
    build = agent._build_messages

    def recording_build(*a, **kw):
        order.append("prompt")
        return build(*a, **kw)

    monkeypatch.setattr(agent, "_build_messages", recording_build)
    await agent.complete("hello")
    assert "flags" in order and "prompt" in order
    assert order.index("flags") < order.index("prompt")
