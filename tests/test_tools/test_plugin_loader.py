from pathlib import Path

from captain_claw.agent import Agent
from captain_claw.llm import LLMProvider, LLMResponse, Message, ToolDefinition
from captain_claw.tools.registry import ToolRegistry


class DummyProvider(LLMProvider):
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
        return len(text)


def test_register_plugin_tools_from_skills_tools_directory(tmp_path: Path):
    plugin_dir = tmp_path / "skills" / "tools"
    plugin_dir.mkdir(parents=True, exist_ok=True)
    plugin_file = plugin_dir / "plugin_echo.py"
    plugin_file.write_text(
        (
            "from captain_claw.tools.registry import Tool, ToolResult\n\n"
            "class PluginEchoTool(Tool):\n"
            "    name = 'plugin_echo'\n"
            "    description = 'Echo value from plugin tool'\n"
            "    parameters = {\n"
            "        'type': 'object',\n"
            "        'properties': {'value': {'type': 'string'}},\n"
            "        'required': ['value'],\n"
            "    }\n\n"
            "    async def execute(self, value: str, **kwargs):\n"
            "        return ToolResult(success=True, content=str(value))\n"
        ),
        encoding="utf-8",
    )

    agent = Agent(provider=DummyProvider())
    agent.workspace_base_path = tmp_path
    registry = ToolRegistry(base_path=tmp_path)
    agent.tools = registry

    agent._register_plugin_tools()

    assert registry.has_tool("plugin_echo") is True
    metadata = registry.get_tool_metadata("plugin_echo")
    assert metadata.get("source") == "plugin"
    assert metadata.get("path") == str(plugin_file.resolve())
