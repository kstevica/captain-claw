import pytest
from typing import Any

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


def test_cloud_mode_builds_memory_note_from_filtered_historical_tool_output():
    old_cfg = get_config().model_copy(deep=True)
    cfg = old_cfg.model_copy(deep=True)
    cfg.model.provider = "ollama"
    cfg.model.model = "minimax-m2.5:cloud"
    set_config(cfg)
    try:
        agent = Agent(provider=TokenAwareProvider())
        agent.session = Session(id="s1", name="default")
        agent.session.add_message("user", "extract titles from techcrunch.com")
        agent.session.add_message(
            "tool",
            "Top titles: LaunchPad 3.0, AI Browser X. Link: https://techcrunch.com/story-123",
            tool_name="web_fetch",
        )
        turn_start_idx = len(agent.session.messages)
        agent.session.add_message("user", "Tell me more about LaunchPad 3.0")

        messages = agent._build_messages(
            tool_messages_from_index=turn_start_idx,
            query="Tell me more about LaunchPad 3.0",
        )

        assert all(m.role != "tool" for m in messages[:-1])
        memory_notes = [m.content for m in messages if "Continuity note from earlier tool outputs" in m.content]
        assert memory_notes
        assert "LaunchPad 3.0" in memory_notes[0]
        assert "https://techcrunch.com/story-123" in memory_notes[0]
    finally:
        set_config(old_cfg)


def test_non_cloud_uses_memory_note_for_followup_without_tool_role_history():
    old_cfg = get_config().model_copy(deep=True)
    cfg = old_cfg.model_copy(deep=True)
    cfg.model.provider = "ollama"
    cfg.model.model = "llama3.2"
    set_config(cfg)
    try:
        agent = Agent(provider=TokenAwareProvider())
        agent.session = Session(id="s1", name="default")
        agent.session.add_message("user", "extract titles")
        agent.session.add_message("tool", "Title: Alpha", tool_name="web_fetch")
        turn_start_idx = len(agent.session.messages)
        agent.session.add_message("user", "details for Alpha")

        messages = agent._build_messages(
            tool_messages_from_index=turn_start_idx,
            query="details for Alpha",
        )

        assert all(m.role != "tool" for m in messages[:-1])
        memory_notes = [m.content for m in messages if "Continuity note from earlier tool outputs" in m.content]
        assert memory_notes
        assert "Title: Alpha" in memory_notes[0]
    finally:
        set_config(old_cfg)


def test_memory_selection_debug_emits_links_to_monitor_callback():
    old_cfg = get_config().model_copy(deep=True)
    cfg = old_cfg.model_copy(deep=True)
    cfg.model.provider = "ollama"
    cfg.model.model = "llama3.2"
    set_config(cfg)
    try:
        outputs: list[dict[str, str]] = []

        def cb(name: str, args: dict, output: str) -> None:
            outputs.append({"name": name, "output": output, "query": str(args.get("query", ""))})

        agent = Agent(provider=TokenAwareProvider(), tool_output_callback=cb)
        agent.session = Session(id="s1", name="default")
        agent.session.add_message("user", "extract titles")
        agent.session.add_message(
            "tool",
            "Title: LaunchPad 3.0",
            tool_name="web_fetch",
            tool_arguments={"url": "https://techcrunch.com/story-123"},
        )
        turn_start_idx = len(agent.session.messages)
        agent.session.add_message("user", "tell me more about launchpad")

        _ = agent._build_messages(
            tool_messages_from_index=turn_start_idx,
            query="tell me more about launchpad",
        )

        memory_logs = [entry for entry in outputs if entry["name"] == "memory_select"]
        assert memory_logs
        assert "https://techcrunch.com/story-123" in memory_logs[0]["output"]
        assert "selection_mode=" in memory_logs[0]["output"]
        assert "reason=term_overlap" in memory_logs[0]["output"]
        assert "message_index=" in memory_logs[0]["output"]
    finally:
        set_config(old_cfg)


def test_system_prompt_includes_workspace_folder_policy_with_session_subfolders():
    agent = Agent(provider=TokenAwareProvider())
    agent.session = Session(id="s1", name="phase 3")

    prompt = agent._build_system_prompt()

    assert "Workspace folder policy:" in prompt
    assert "All tool-generated files must be written inside:" in prompt
    assert 'Current session subfolder name: "phase-3".' in prompt
    assert "saved/scripts/phase-3/" in prompt
    assert "saved/tools/phase-3/" in prompt
    assert "saved/downloads/phase-3/" in prompt
    assert "saved/media/phase-3/" in prompt
    assert "saved/showcase/phase-3/" in prompt
    assert "saved/skills/phase-3/" in prompt
    assert "saved/tmp/phase-3/" in prompt


def test_build_messages_includes_planning_pipeline_note():
    agent = Agent(provider=TokenAwareProvider())
    agent.session = Session(id="s1", name="default")
    agent.session.add_message("user", "help me plan deployment")
    pipeline = {
        "tasks": [
            {"id": "task_1", "title": "Gather deployment constraints", "status": "in_progress"},
            {"id": "task_2", "title": "Prepare rollout steps", "status": "pending"},
        ],
        "current_index": 0,
        "state": "active",
    }

    messages = agent._build_messages(query="deployment", planning_pipeline=pipeline)

    notes = [m.content for m in messages if "Planning mode is ON" in m.content]
    assert notes
    assert "[IN_PROGRESS] Gather deployment constraints" in notes[0]


@pytest.mark.asyncio
async def test_complete_emits_planning_pipeline_updates_when_enabled():
    outputs: list[dict[str, Any]] = []

    def cb(name: str, args: dict, output: str) -> None:
        outputs.append({"name": name, "args": args, "output": output})

    agent = Agent(provider=TokenAwareProvider(), tool_output_callback=cb)
    agent._initialized = True
    agent.session = Session(id="s1", name="default")
    agent.session_manager = DummySessionManager()
    agent.tools = ToolRegistry()
    agent.planning_enabled = True

    result = await agent.complete("plan and answer this request")

    assert result == "ok"
    planning_logs = [entry for entry in outputs if entry["name"] == "planning"]
    assert planning_logs
    assert any("event" in entry["args"] and entry["args"]["event"] == "created" for entry in planning_logs)
    assert any("event" in entry["args"] and entry["args"]["event"] == "completed" for entry in planning_logs)


@pytest.mark.asyncio
async def test_compact_session_summarizes_old_messages_and_updates_metadata():
    old_cfg = get_config().model_copy(deep=True)
    cfg = old_cfg.model_copy(deep=True)
    cfg.context.max_tokens = 40
    cfg.context.compaction_threshold = 0.5
    cfg.context.compaction_ratio = 0.2
    set_config(cfg)
    try:
        agent = Agent(provider=TokenAwareProvider())
        agent.session = Session(id="s1", name="default")
        agent.session_manager = DummySessionManager()

        for idx in range(10):
            agent.session.add_message("user", f"message {idx} alpha beta gamma delta")

        compacted, stats = await agent.compact_session(force=False, trigger="auto")

        assert compacted is True
        assert int(stats["compacted_messages"]) > 0
        assert int(stats["after_tokens"]) < int(stats["before_tokens"])
        assert agent.session.messages[0]["tool_name"] == "compaction_summary"
        assert "Conversation summary of earlier messages" in agent.session.messages[0]["content"]
        assert agent.session.metadata["compaction"]["count"] >= 1
        assert agent.session.metadata["compaction"]["auto_count"] >= 1
    finally:
        set_config(old_cfg)


@pytest.mark.asyncio
async def test_complete_triggers_auto_compaction_when_over_threshold():
    old_cfg = get_config().model_copy(deep=True)
    cfg = old_cfg.model_copy(deep=True)
    cfg.context.max_tokens = 60
    cfg.context.compaction_threshold = 0.5
    cfg.context.compaction_ratio = 0.2
    set_config(cfg)
    try:
        agent = Agent(provider=TokenAwareProvider())
        agent._initialized = True
        agent.session = Session(id="s1", name="default")
        agent.session_manager = DummySessionManager()
        agent.tools = ToolRegistry()

        for idx in range(10):
            agent.session.add_message("assistant", f"history {idx} alpha beta gamma delta epsilon")

        result = await agent.complete("new question")

        assert result == "ok"
        assert any(msg.get("tool_name") == "compaction_summary" for msg in agent.session.messages)
        assert agent.session.metadata["compaction"]["auto_count"] >= 1
    finally:
        set_config(old_cfg)
