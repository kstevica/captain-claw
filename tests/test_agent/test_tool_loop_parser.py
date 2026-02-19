from typing import AsyncIterator

from captain_claw.agent import Agent
from captain_claw.llm import LLMProvider, LLMResponse, Message, ToolDefinition


class NoopProvider(LLMProvider):
    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        return LLMResponse(content="")

    async def complete_streaming(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        if False:
            yield ""

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)


def test_extract_tool_calls_from_json_pseudo_objects():
    agent = Agent(provider=NoopProvider())
    content = """
I'll search and fetch.
{"query":"Netokracija latest site:netokracija.rs OR site:netokracija.hr","max_results": 10}
{"id":"web_search.1","cursor":0,"cursor_index":0}
{"url":"https://www.netokracija.hr/"}
{"id":"web_fetch.1","cursor":36}
"""
    calls = agent._extract_tool_calls_from_content(content)
    assert calls
    assert any(call.name == "web_search" for call in calls)
    assert any(call.name == "web_fetch" for call in calls)
    web_search_calls = [call for call in calls if call.name == "web_search"]
    web_fetch_calls = [call for call in calls if call.name == "web_fetch"]
    assert web_search_calls[0].arguments.get("query")
    assert int(web_search_calls[0].arguments.get("count")) == 10
    assert str(web_fetch_calls[0].arguments.get("url")).startswith("https://")


def test_extract_tool_calls_deduplicates_and_caps_json_fallback_calls():
    agent = Agent(provider=NoopProvider())
    lines = [
        f'{{"query":"netokracija topic {idx}","max_results":10}}'
        for idx in range(12)
    ]
    lines += ['{"query":"netokracija topic 1","max_results":10}']
    content = "\n".join(lines)
    calls = agent._extract_tool_calls_from_content(content)
    assert len(calls) == 8
    assert len({call.arguments.get("query") for call in calls}) == 8


def test_extract_tool_calls_from_explicit_tool_input_shape():
    agent = Agent(provider=NoopProvider())
    content = '{"tool":"web_search","id":"search1","input":"site:netokracija.com latest netokracija 2026 February"}'
    calls = agent._extract_tool_calls_from_content(content)
    assert len(calls) == 1
    assert calls[0].name == "web_search"
    assert "site:netokracija.com" in str(calls[0].arguments.get("query", ""))


def test_extract_tool_calls_for_pocket_tts_input_mapping():
    agent = Agent(provider=NoopProvider())
    content = '{"tool":"pocket_tts","id":"tts1","input":"Hello from Captain Claw","voice":"af_bella"}'
    calls = agent._extract_tool_calls_from_content(content)
    assert len(calls) == 1
    assert calls[0].name == "pocket_tts"
    assert str(calls[0].arguments.get("text", "")) == "Hello from Captain Claw"
    assert str(calls[0].arguments.get("voice", "")) == "af_bella"
