import asyncio
import json
from pathlib import Path
from typing import Any

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


class ExecutingShellTool(Tool):
    name = "shell"
    description = "Execute shell command"
    parameters = {
        "type": "object",
        "properties": {"command": {"type": "string"}},
        "required": ["command"],
    }

    def __init__(self):
        self.commands: list[str] = []

    async def execute(self, **kwargs) -> ToolResult:
        command = str(kwargs.get("command", ""))
        self.commands.append(command)
        process = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        output = stdout.decode("utf-8", errors="replace").strip()
        err = stderr.decode("utf-8", errors="replace").strip()
        if err:
            output = f"{output}\n[stderr] {err}".strip()
        return ToolResult(success=process.returncode == 0, content=output or "[no output]")


class RetryWriteAfterMissingFileTool(Tool):
    name = "write"
    description = "Write tool that reports success before file exists once."
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "content": {"type": "string"},
            "append": {"type": "boolean"},
        },
        "required": ["path", "content"],
    }

    def __init__(self):
        self.calls = 0

    async def execute(self, path: str, content: str, append: bool = False, **kwargs: Any) -> ToolResult:
        self.calls += 1
        saved_root = WriteTool._resolve_saved_root(kwargs)
        session_id = WriteTool._normalize_session_id(str(kwargs.get("_session_id", "")))
        file_path = WriteTool._normalize_under_saved(path, saved_root, session_id)

        if self.calls >= 2:
            mode = "a" if append else "w"
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with file_path.open(mode, encoding="utf-8") as handle:
                handle.write(content)

        redirect_note = ""
        requested = Path(path).expanduser()
        if str(requested) != str(file_path):
            redirect_note = f" (requested: {path})"
        return ToolResult(
            success=True,
            content=f"Written {len(content)} chars to {file_path}{redirect_note}",
        )


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


class ListWorkerEnforcedProvider(LLMProvider):
    def __init__(self):
        self.main_calls = 0
        self.script_calls = 0
        self.critic_calls = 0
        self.planner_calls = 0

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        system_text = (messages[0].content if messages else "").lower()
        if "list-task extractor" in system_text:
            return LLMResponse(
                content=json.dumps(
                    {
                        "has_list_work": True,
                        "members": ["Daytona", "Gideon", "Memgraph"],
                        "per_member_action": "summarize each company in 5 sentences and save one file per company",
                        "recommended_strategy": "script",
                        "confidence": "high",
                    }
                )
            )
        if "task-contract planner" in system_text:
            self.planner_calls += 1
            return LLMResponse(
                content=json.dumps(
                    {
                        "summary": "Process startup list and save per-company files",
                        "tasks": [
                            {"title": "Gather startup names"},
                            {"title": "Generate per-company summaries"},
                            {"title": "Write output files"},
                        ],
                        "requirements": [
                            {"id": "req_files", "title": "Save requested company summary files"},
                        ],
                        "prefetch_urls": [],
                    }
                )
            )
        if "strict completion critic" in system_text:
            self.critic_calls += 1
            return LLMResponse(
                content=json.dumps(
                    {
                        "complete": True,
                        "checks": [
                            {"id": "req_files", "ok": True, "reason": ""},
                            {"id": "req_member_daytona", "ok": True, "reason": ""},
                            {"id": "req_member_gideon", "ok": True, "reason": ""},
                            {"id": "req_member_memgraph", "ok": True, "reason": ""},
                        ],
                        "feedback": "",
                    }
                )
            )
        if "generate a single runnable script" in system_text:
            self.script_calls += 1
            return LLMResponse(
                content=(
                    "```python\n"
                    "from pathlib import Path\n"
                    "def main():\n"
                    "    names = ['daytona', 'gideon', 'memgraph']\n"
                    "    for name in names:\n"
                    "        out = Path(f'saved/showcase/s1/{name}-netokracija-summary.md')\n"
                    "        out.parent.mkdir(parents=True, exist_ok=True)\n"
                    "        out.write_text(f'{name} summary in 5 sentences.\\n', encoding='utf-8')\n"
                    "        print(f'processed={name}')\n"
                    "if __name__ == '__main__':\n"
                    "    main()\n"
                    "```\n"
                )
            )
        self.main_calls += 1
        if self.main_calls == 1:
            return LLMResponse(content="Proceeding with Daytona first.")
        return LLMResponse(content="Done. Processed list and saved outputs.")

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


class ListExtractorOnlyProvider(LLMProvider):
    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        system_text = (messages[0].content if messages else "").lower()
        if "list-task extractor" in system_text:
            return LLMResponse(
                content=json.dumps(
                    {
                        "has_list_work": True,
                        "members": ["Daytona", "Gideon"],
                        "per_member_action": "summarize each in 5 sentences and save to file",
                        "recommended_strategy": "direct",
                        "confidence": "high",
                    }
                )
            )
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


class StaticTextProvider(LLMProvider):
    def __init__(self, content: str):
        self.content = content

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
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
        return len(text)


class IncompleteThenCompleteReportProvider(LLMProvider):
    def __init__(self, complete_text: str):
        self.calls = 0
        self.main_calls = 0
        self.critic_calls = 0
        self.complete_text = complete_text

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        self.calls += 1
        system_text = (messages[0].content if messages else "").lower()
        if "strict completion critic" in system_text:
            self.critic_calls += 1
            if self.critic_calls == 1:
                return LLMResponse(
                    content=json.dumps(
                        {
                            "complete": False,
                            "checks": [
                                {"id": "req_source_1", "ok": True, "reason": ""},
                                {"id": "req_source_2", "ok": False, "reason": "missing Source 2 section"},
                                {"id": "req_conclusion", "ok": False, "reason": "missing Conclusion"},
                            ],
                            "feedback": "Add Source 2 and Conclusion sections.",
                        }
                    )
                )
        if "task-contract planner" in system_text:
            return LLMResponse(
                content=json.dumps(
                    {
                        "summary": "Compile source report",
                        "tasks": [
                            {"title": "Read sources"},
                            {"title": "Write source-by-source report"},
                        ],
                        "requirements": [
                            {"id": "req_source_1", "title": "Include Source 1 section"},
                        ],
                        "prefetch_urls": ["https://example.com/a"],
                    }
                )
            )
            return LLMResponse(
                content=json.dumps(
                    {
                        "complete": True,
                        "checks": [
                            {"id": "req_source_1", "ok": True, "reason": ""},
                            {"id": "req_source_2", "ok": True, "reason": ""},
                            {"id": "req_conclusion", "ok": True, "reason": ""},
                        ],
                        "feedback": "",
                    }
                )
            )

        self.main_calls += 1
        if self.main_calls == 1:
            return LLMResponse(content="Source 1 only. Missing others.")
        return LLMResponse(content=self.complete_text)

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


class IncompleteThenCompleteTopNProvider(LLMProvider):
    def __init__(self, top_n: int):
        self.calls = 0
        self.main_calls = 0
        self.critic_calls = 0
        self.top_n = top_n

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        self.calls += 1
        system_text = (messages[0].content if messages else "").lower()
        if "strict completion critic" in system_text:
            self.critic_calls += 1
            if self.critic_calls == 1:
                return LLMResponse(
                    content=json.dumps(
                        {
                            "complete": False,
                            "checks": [
                                {"id": "req_top_list", "ok": False, "reason": "only 2 items listed"},
                            ],
                            "feedback": f"Provide {self.top_n} numbered items.",
                        }
                    )
                )
            return LLMResponse(
                content=json.dumps(
                    {
                        "complete": True,
                        "checks": [
                            {"id": "req_top_list", "ok": True, "reason": ""},
                        ],
                        "feedback": "",
                    }
                )
            )
        if "task-contract planner" in system_text:
            return LLMResponse(
                content=json.dumps(
                    {
                        "summary": "Provide top cities",
                        "tasks": [
                            {"title": "Draft ordered list"},
                            {"title": "Validate completeness"},
                        ],
                        "requirements": [
                            {"id": "req_top_list", "title": f"Provide {self.top_n} list items"},
                        ],
                        "prefetch_urls": [],
                    }
                )
            )

        self.main_calls += 1
        if self.main_calls == 1:
            return LLMResponse(content="1. London\n2. Birmingham")
        lines = [f"{idx}. City {idx}" for idx in range(1, self.top_n + 1)]
        return LLMResponse(content="\n".join(lines))

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


class SlowCompletionTopNProvider(LLMProvider):
    def __init__(self, top_n: int):
        self.calls = 0
        self.main_calls = 0
        self.critic_calls = 0
        self.top_n = top_n

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        self.calls += 1
        system_text = (messages[0].content if messages else "").lower()
        if "strict completion critic" in system_text:
            self.critic_calls += 1
            if self.critic_calls <= 2:
                missing = self.top_n - self.critic_calls
                return LLMResponse(
                    content=json.dumps(
                        {
                            "complete": False,
                            "checks": [
                                {"id": "req_top_list", "ok": False, "reason": f"missing {missing} items"},
                            ],
                            "feedback": f"Still missing {missing} items.",
                        }
                    )
                )
            return LLMResponse(
                content=json.dumps(
                    {
                        "complete": True,
                        "checks": [
                            {"id": "req_top_list", "ok": True, "reason": ""},
                        ],
                        "feedback": "",
                    }
                )
            )
        if "task-contract planner" in system_text:
            return LLMResponse(
                content=json.dumps(
                    {
                        "summary": "Provide top cities with strict completion checks",
                        "tasks": [
                            {"title": "Draft ordered list"},
                            {"title": "Improve list until complete"},
                        ],
                        "requirements": [
                            {"id": "req_top_list", "title": f"Provide {self.top_n} list items"},
                        ],
                        "prefetch_urls": [],
                    }
                )
            )

        self.main_calls += 1
        if self.main_calls == 1:
            return LLMResponse(content="1. City 1")
        if self.main_calls == 2:
            return LLMResponse(content="1. City 1\n2. City 2")
        lines = [f"{idx}. City {idx}" for idx in range(1, self.top_n + 1)]
        return LLMResponse(content="\n".join(lines))

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


class PlannerRetryProvider(LLMProvider):
    def __init__(self):
        self.planner_max_tokens: list[int | None] = []
        self.planner_calls = 0

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        system_text = (messages[0].content if messages else "").lower()
        if "task-contract planner" in system_text:
            self.planner_calls += 1
            self.planner_max_tokens.append(max_tokens)
            if self.planner_calls == 1:
                return LLMResponse(
                    content="",
                    usage={
                        "prompt_tokens": 10,
                        "completion_tokens": int(max_tokens or 0),
                        "total_tokens": 10 + int(max_tokens or 0),
                    },
                )
            return LLMResponse(
                content=json.dumps(
                    {
                        "summary": "Retry planner output",
                        "tasks": [{"title": "Retry planner task"}],
                        "requirements": [{"id": "req_done", "title": "Return completed output"}],
                        "prefetch_urls": [],
                    }
                )
            )
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


class ClarificationCarryProvider(LLMProvider):
    def __init__(self):
        self.planner_user_payloads: list[str] = []

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        system_text = (messages[0].content if messages else "").lower()
        if "strict completion critic" in system_text:
            return LLMResponse(
                content=json.dumps(
                    {
                        "complete": True,
                        "checks": [{"id": "req_done", "ok": True, "reason": ""}],
                        "feedback": "",
                    }
                )
            )
        if "task-contract planner" in system_text:
            self.planner_user_payloads.append(messages[1].content if len(messages) > 1 else "")
            return LLMResponse(
                content=json.dumps(
                    {
                        "summary": "Process full request",
                        "tasks": [{"title": "Run full request"}],
                        "requirements": [{"id": "req_done", "title": "Return response"}],
                        "prefetch_urls": [],
                    }
                )
            )
        return LLMResponse(content="completed")

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


class RewriteFallbackGateProvider(LLMProvider):
    def __init__(self):
        self.main_calls = 0
        self.critic_calls = 0
        self.rewrite_calls = 0

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        system_text = (messages[0].content if messages else "").lower()
        if "strict completion critic" in system_text:
            self.critic_calls += 1
            if self.critic_calls == 1:
                return LLMResponse(
                    content=json.dumps(
                        {
                            "complete": False,
                            "checks": [
                                {"id": "req_two_items", "ok": False, "reason": "missing second item"},
                            ],
                            "feedback": "Include both numbered items.",
                        }
                    )
                )
            return LLMResponse(
                content=json.dumps(
                    {
                        "complete": True,
                        "checks": [{"id": "req_two_items", "ok": True, "reason": ""}],
                        "feedback": "",
                    }
                )
            )
        if "task-contract planner" in system_text:
            return LLMResponse(
                content=json.dumps(
                    {
                        "summary": "Return two numbered items",
                        "tasks": [{"title": "Collect data"}, {"title": "Return two items"}],
                        "requirements": [{"id": "req_two_items", "title": "Include both item 1 and item 2"}],
                        "prefetch_urls": [],
                    }
                )
            )
        if "rewrite raw tool output" in system_text:
            self.rewrite_calls += 1
            return LLMResponse(content="1. Item one")

        self.main_calls += 1
        if tools is not None and self.main_calls == 1:
            return LLMResponse(
                content="",
                tool_calls=[ToolCall(id="c1", name="dummy", arguments={"value": "v"})],
            )
        return LLMResponse(content="1. Item one\n2. Item two")

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


class OpenAIStrictToolSequenceProvider(LLMProvider):
    provider = "openai"
    model = "gpt-5-mini"

    def __init__(self):
        self.calls = 0
        self.second_messages: list[Message] = []

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
        self.second_messages = list(messages)
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


@pytest.mark.asyncio
async def test_complete_records_full_llm_trace_when_enabled():
    outputs: list[dict[str, Any]] = []

    def cb(name: str, args: dict, output: str) -> None:
        outputs.append({"name": name, "args": args, "output": output})

    provider = DummyProvider()
    agent = Agent(provider=provider, tool_output_callback=cb)
    agent._initialized = True
    agent.session = Session(id="s1", name="default")
    agent.session_manager = DummySessionManager()
    agent.tools = ToolRegistry()
    await agent.set_monitor_trace_llm(True, persist=False)

    result = await agent.complete("say hi")

    assert result == "ok"
    trace_logs = [entry for entry in outputs if entry["name"] == "llm_trace"]
    assert trace_logs
    assert "interaction=turn_1" in trace_logs[0]["output"]
    assert "[assistant_response]" in trace_logs[0]["output"]
    assert "ok" in trace_logs[0]["output"]
    assert any(
        msg.get("role") == "tool" and msg.get("tool_name") == "llm_trace"
        for msg in agent.session.messages
    )


@pytest.mark.asyncio
async def test_emit_tool_output_records_compact_pipeline_trace_when_enabled():
    provider = DummyProvider()
    agent = Agent(provider=provider)
    agent._initialized = True
    agent.session = Session(id="s1", name="default")
    agent.session_manager = DummySessionManager()
    agent.tools = ToolRegistry()
    await agent.set_monitor_trace_pipeline(True, persist=False)

    agent._emit_tool_output(
        "planning",
        {"event": "created", "leaf_index": 1, "leaf_remaining": 4, "current_path": "1.1"},
        "Planning event=created\nstate=active\nprogress=1/5 remaining=4",
    )

    planning_entries = [
        msg for msg in agent.session.messages if msg.get("role") == "tool" and msg.get("tool_name") == "planning"
    ]
    pipeline_entries = [
        msg for msg in agent.session.messages if msg.get("role") == "tool" and msg.get("tool_name") == "pipeline_trace"
    ]
    assert planning_entries
    assert "Planning event=created" in str(planning_entries[0].get("content", ""))
    assert pipeline_entries
    payload = pipeline_entries[0].get("tool_arguments")
    assert isinstance(payload, dict)
    assert payload.get("source") == "planning"
    assert payload.get("event") == "created"
    assert "Planning event=created" not in str(pipeline_entries[0].get("content", ""))


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


def test_build_messages_skips_monitor_only_tool_entries():
    provider = DummyProvider()
    agent = Agent(provider=provider)
    agent.session = Session(id="s1", name="default")
    agent.session.add_message("tool", "plan state", tool_name="planning")
    agent.session.add_message("tool", "contract state", tool_name="task_contract")
    agent.session.add_message("tool", "gate state", tool_name="completion_gate")
    agent.session.add_message("tool", "pipeline trace", tool_name="pipeline_trace")
    agent.session.add_message("tool", "source output", tool_name="web_fetch")

    messages = agent._build_messages(query="next")
    tool_names = [str(m.tool_name or "") for m in messages if m.role == "tool"]

    assert "web_fetch" in tool_names
    assert "planning" not in tool_names
    assert "task_contract" not in tool_names
    assert "completion_gate" not in tool_names
    assert "pipeline_trace" not in tool_names


def test_collect_turn_tool_output_skips_monitor_only_tools():
    provider = DummyProvider()
    agent = Agent(provider=provider)
    agent.session = Session(id="s1", name="default")
    agent.session.add_message("tool", "plan state", tool_name="planning")
    agent.session.add_message("tool", "gate state", tool_name="completion_gate")
    agent.session.add_message("tool", "trace state", tool_name="pipeline_trace")
    agent.session.add_message("tool", "fetched body", tool_name="web_fetch")

    collected = agent._collect_turn_tool_output(0)

    assert collected == "fetched body"


def test_should_use_contract_pipeline_defaults_to_fast_path_for_simple_prompt():
    assert (
        Agent._should_use_contract_pipeline(
            "what time is it in zagreb?",
            planning_enabled=False,
            pipeline_mode="loop",
        )
        is False
    )


def test_should_use_contract_pipeline_does_not_auto_enable_for_complex_prompt():
    assert (
        Agent._should_use_contract_pipeline(
            "check all those sources and compile a report per source with conclusion",
            planning_enabled=False,
            pipeline_mode="loop",
        )
        is False
    )


def test_should_use_contract_pipeline_does_not_auto_enable_for_high_stakes_prompt():
    assert (
        Agent._should_use_contract_pipeline(
            "I have symptoms and need dosage guidance for this prescription medicine",
            planning_enabled=False,
            pipeline_mode="loop",
        )
        is False
    )


def test_should_use_contract_pipeline_respects_manual_planning_mode():
    assert (
        Agent._should_use_contract_pipeline(
            "say hello",
            planning_enabled=True,
            pipeline_mode="contracts",
        )
        is True
    )


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
    assert provider.calls == 2
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


@pytest.mark.asyncio
async def test_list_requests_force_python_worker_tool_execution(tmp_path: Path):
    provider = ListWorkerEnforcedProvider()
    outputs: list[str] = []

    def cb(name: str, args: dict, output: str) -> None:
        outputs.append(name)

    agent = Agent(provider=provider, tool_output_callback=cb)
    agent._initialized = True
    agent.session = Session(id="s1", name="default")
    agent.session_manager = DummySessionManager()
    registry = ToolRegistry(base_path=tmp_path)
    shell_tool = ExecutingShellTool()
    registry.register(WriteTool())
    registry.register(shell_tool)
    agent.tools = registry

    result = await agent.complete(
        "generate a python script that fetches netokracija.com, extracts startup names, summarizes "
        "each company in 5 sentences, and writes each summary to file <company-name>-netokracija-summary.md"
    )

    assert provider.script_calls >= 1
    assert provider.main_calls >= 1
    assert any(name == "write" for name in outputs)
    assert any(name == "shell" for name in outputs)
    assert shell_tool.commands
    assert "python" in shell_tool.commands[0]
    generated_scripts = list((tmp_path / "saved" / "scripts" / "s1").glob("*.py"))
    assert generated_scripts
    assert "Script saved and executed from" in result


def test_choose_list_execution_strategy_prefers_direct_without_explicit_script_request():
    strategy = Agent._choose_list_execution_strategy(
        user_input="fetch many sources and summarize each one into a file",
        members_count=20,
        recommended="script",
    )
    assert strategy == "direct"


@pytest.mark.asyncio
async def test_generate_list_task_plan_extracts_members_and_direct_strategy():
    provider = ListExtractorOnlyProvider()
    agent = Agent(provider=provider)
    agent._initialized = True
    agent.session = Session(id="s1", name="default")
    agent.session_manager = DummySessionManager()

    plan = await agent._generate_list_task_plan(
        user_input="extract startup names and summarize each one into a file",
        context_excerpt="- Daytona\n- Gideon\n",
        turn_usage=agent._empty_usage(),
    )

    assert plan["enabled"] is True
    assert plan["members"] == ["Daytona", "Gideon"]
    assert plan["strategy"] == "direct"
    assert "summarize each" in str(plan["per_member_action"]).lower()


def test_apply_list_requirements_adds_member_coverage_requirements():
    base = [{"id": "req_base", "title": "complete user request"}]
    plan = {
        "enabled": True,
        "members": ["Daytona", "Gideon"],
        "per_member_action": "write summary file",
    }

    merged = Agent._apply_list_requirements(base, plan)

    assert len(merged) == 3
    titles = [str(item.get("title", "")) for item in merged]
    assert any("Cover list member: Daytona" in title for title in titles)
    assert any("Cover list member: Gideon" in title for title in titles)


def test_build_messages_includes_list_task_memory_note():
    provider = DummyProvider()
    agent = Agent(provider=provider)
    agent.session = Session(id="s1", name="default")
    agent.session.add_message("user", "handle all startups")

    messages = agent._build_messages(
        query="handle all startups",
        list_task_plan={
            "enabled": True,
            "members": ["Daytona", "Gideon"],
            "strategy": "direct",
            "per_member_action": "summarize and save",
        },
    )

    assert any(
        msg.role == "assistant" and "List task memory is active" in msg.content
        for msg in messages
    )


@pytest.mark.asyncio
async def test_auto_write_all_files_uses_previous_filename_blocks(tmp_path: Path):
    provider = StaticTextProvider("Acknowledged. Creating all files now.")
    agent = Agent(provider=provider)
    agent._initialized = True
    agent.session = Session(id="s1", name="default")
    agent.session.add_message(
        "assistant",
        (
            "Filename: Zagreb-details.md\n"
            "---\n"
            "# Zagreb\n\n"
            "Capital city details.\n\n"
            "Filename: Split-details.md\n"
            "---\n"
            "# Split\n\n"
            "Coastal city details.\n"
        ),
    )
    agent.session_manager = DummySessionManager()
    registry = ToolRegistry(base_path=tmp_path)
    registry.register(WriteTool())
    agent.tools = registry

    result = await agent.complete("create all files")

    zagreb = tmp_path / "saved" / "showcase" / "s1" / "Zagreb-details.md"
    split = tmp_path / "saved" / "showcase" / "s1" / "Split-details.md"
    assert zagreb.exists()
    assert split.exists()
    assert "Capital city details." in zagreb.read_text(encoding="utf-8")
    assert "Coastal city details." in split.read_text(encoding="utf-8")
    assert "Saved 2 files" in result


@pytest.mark.asyncio
async def test_auto_write_retries_when_file_missing_after_first_write(tmp_path: Path):
    provider = StaticTextProvider("Final report body.")
    agent = Agent(provider=provider)
    agent._initialized = True
    agent.session = Session(id="s1", name="default")
    agent.session_manager = DummySessionManager()
    registry = ToolRegistry(base_path=tmp_path)
    retry_write_tool = RetryWriteAfterMissingFileTool()
    registry.register(retry_write_tool)
    agent.tools = registry

    result = await agent.complete("save this to file report.md")

    saved_path = tmp_path / "saved" / "tmp" / "s1" / "report.md"
    assert retry_write_tool.calls == 2
    assert saved_path.exists()
    assert saved_path.read_text(encoding="utf-8") == "Final report body."
    assert "Written" in result


@pytest.mark.asyncio
async def test_source_report_pipeline_prefetches_all_recent_sources_and_retries_incomplete_draft():
    provider = IncompleteThenCompleteReportProvider(
        (
            "Source 1 — https://example.com/a\n"
            "Line.\n\n"
            "Source 2 — https://example.com/b\n"
            "Line.\n\n"
            "Conclusion\n"
            "Done."
        )
    )
    agent = Agent(provider=provider)
    agent._initialized = True
    agent.session = Session(id="s1", name="default")
    agent.session.add_message(
        "assistant",
        (
            "1. https://example.com/a\n"
            "2. https://example.com/b\n"
        ),
    )
    agent.session_manager = DummySessionManager()
    registry = ToolRegistry()
    registry.register(DummyWebFetchTool())
    agent.tools = registry
    await agent.set_pipeline_mode("contracts", persist=False)

    result = await agent.complete("check all those sources and compile a report per source with conclusion")

    assert provider.calls >= 2
    assert "Source 1" in result
    assert "Source 2" in result
    assert "Conclusion" in result
    tool_messages = [
        msg for msg in agent.session.messages if msg.get("role") == "tool" and msg.get("tool_name") == "web_fetch"
    ]
    assert len(tool_messages) >= 2


@pytest.mark.asyncio
async def test_completion_gate_retries_top_n_response_and_auto_creates_pipeline():
    provider = IncompleteThenCompleteTopNProvider(top_n=3)
    outputs: list[dict[str, object]] = []

    def cb(name: str, args: dict, output: str) -> None:
        outputs.append({"name": name, "args": args, "output": output})

    agent = Agent(provider=provider, tool_output_callback=cb)
    agent._initialized = True
    agent.session = Session(id="s1", name="default")
    agent.session_manager = DummySessionManager()
    agent.tools = ToolRegistry()
    await agent.set_pipeline_mode("contracts", persist=False)

    result = await agent.complete("what are top 3 cities in the uk by population")

    assert provider.calls >= 2
    assert "1. City 1" in result
    assert "2. City 2" in result
    assert "3. City 3" in result

    completion_logs = [entry for entry in outputs if entry["name"] == "completion_gate"]
    assert completion_logs
    assert any(bool(entry["args"].get("passed")) is False for entry in completion_logs)
    assert any(bool(entry["args"].get("passed")) is True for entry in completion_logs)

    planning_logs = [entry for entry in outputs if entry["name"] == "planning"]
    assert planning_logs
    assert any(
        entry["args"].get("mode") in {"manual_with_contract", "auto_contract"}
        for entry in planning_logs
    )


@pytest.mark.asyncio
async def test_completion_gate_extends_iteration_budget_when_feedback_keeps_progressing():
    provider = SlowCompletionTopNProvider(top_n=3)
    outputs: list[dict[str, object]] = []

    def cb(name: str, args: dict, output: str) -> None:
        outputs.append({"name": name, "args": args, "output": output})

    agent = Agent(provider=provider, tool_output_callback=cb)
    agent._initialized = True
    agent.session = Session(id="s1", name="default")
    agent.session_manager = DummySessionManager()
    agent.tools = ToolRegistry()
    await agent.set_pipeline_mode("contracts", persist=False)
    agent.max_iterations = 1
    agent._compute_turn_iteration_budget = lambda **_: 2  # type: ignore[method-assign]

    result = await agent.complete("what are top 3 cities in the uk by population")

    assert "1. City 1" in result
    assert "2. City 2" in result
    assert "3. City 3" in result
    completion_logs = [entry for entry in outputs if entry["name"] == "completion_gate"]
    assert any(entry["args"].get("step") == "iteration_budget_extended" for entry in completion_logs)


@pytest.mark.asyncio
async def test_task_contract_planner_retries_after_empty_output_at_token_cap():
    provider = PlannerRetryProvider()
    agent = Agent(provider=provider)
    agent._initialized = True
    agent.session = Session(id="s1", name="default")
    agent.session_manager = DummySessionManager()
    agent.tools = ToolRegistry()

    contract = await agent._generate_task_contract(
        user_input="Compile a complete report",
        recent_source_urls=[],
        require_all_sources=False,
        turn_usage=agent._empty_usage(),
    )

    assert provider.planner_calls == 2
    assert provider.planner_max_tokens[0] == 1200
    assert int(provider.planner_max_tokens[1] or 0) > 1200
    assert contract["tasks"][0]["title"] == "Retry planner task"


@pytest.mark.asyncio
async def test_complete_merges_pending_clarification_anchor_into_effective_request():
    provider = ClarificationCarryProvider()
    agent = Agent(provider=provider)
    agent._initialized = True
    agent.session = Session(id="s1", name="default")
    agent.session.metadata["clarification_state"] = {
        "pending": True,
        "anchor_request": (
            "fetch netokracija.com and summarize all extracted startup companies "
            "into files named <company>-netokracija-summary.md"
        ),
    }
    agent.session_manager = DummySessionManager()
    agent.tools = ToolRegistry()

    result = await agent.complete("all companies, english summary, normalize file names")

    assert result == "completed"
    assert provider.planner_user_payloads
    planner_prompt = provider.planner_user_payloads[0]
    assert "fetch netokracija.com and summarize all extracted startup companies" in planner_prompt
    assert "all companies, english summary, normalize file names" in planner_prompt
    assert bool(agent.session.metadata.get("clarification_state", {}).get("pending", False)) is False


@pytest.mark.asyncio
async def test_completion_gate_blocks_rewrite_fallback_until_requirements_pass():
    provider = RewriteFallbackGateProvider()
    agent = Agent(provider=provider)
    agent._initialized = True
    agent.session = Session(id="s1", name="default")
    agent.session_manager = DummySessionManager()
    registry = ToolRegistry()
    registry.register(DummyTool())
    agent.tools = registry
    agent.planning_enabled = True
    agent._supports_tool_result_followup = lambda: False  # force rewrite fallback path

    result = await agent.complete("return two numbered items")

    assert provider.rewrite_calls >= 1
    assert provider.critic_calls >= 2
    assert "1. Item one" in result
    assert "2. Item two" in result


@pytest.mark.asyncio
async def test_openai_followup_contains_assistant_tool_calls_before_tool_messages():
    provider = OpenAIStrictToolSequenceProvider()
    agent = Agent(provider=provider)
    agent._initialized = True
    agent.session = Session(id="s1", name="default")
    agent.session_manager = DummySessionManager()
    registry = ToolRegistry()
    registry.register(DummyTool())
    agent.tools = registry

    result = await agent.complete("run tool")

    assert result == "final"
    tool_messages = [m for m in provider.second_messages if m.role == "tool"]
    assert tool_messages
    for idx, msg in enumerate(provider.second_messages):
        if msg.role != "tool":
            continue
        assert idx > 0
        prev = provider.second_messages[idx - 1]
        assert prev.role == "assistant"
        prev_calls = prev.tool_calls or []
        assert any(str(call.get("id", "")) == str(msg.tool_call_id or "") for call in prev_calls)


def test_build_messages_converts_orphan_tool_messages_for_openai():
    provider = OpenAIStrictToolSequenceProvider()
    agent = Agent(provider=provider)
    agent.session = Session(id="s1", name="default")
    agent.session.add_message("user", "previous request")
    agent.session.add_message("tool", "prefetched source output", tool_name="web_fetch")
    agent.session.add_message("assistant", "draft")

    messages = agent._build_messages(query="next request")

    assert not any(m.role == "tool" and not str(m.tool_call_id or "").strip() for m in messages)


def test_build_messages_strips_orphan_assistant_tool_calls_for_openai():
    provider = OpenAIStrictToolSequenceProvider()
    agent = Agent(provider=provider)
    agent.session = Session(id="s1", name="default")
    agent.session.add_message("user", "previous request")
    agent.session.add_message(
        "assistant",
        "",
        tool_calls=[
            {
                "id": "c_missing",
                "type": "function",
                "function": {"name": "dummy", "arguments": "{\"value\":\"v\"}"},
            }
        ],
    )
    agent.session.add_message("assistant", "draft after orphan call")

    messages = agent._build_messages(query="next request")

    assert not any(
        m.role == "assistant" and any(str(call.get("id", "")) == "c_missing" for call in (m.tool_calls or []))
        for m in messages
    )
