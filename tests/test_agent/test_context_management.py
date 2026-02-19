import copy
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from captain_claw.agent import Agent
from captain_claw.config import get_config, set_config
from captain_claw.llm import LLMProvider, LLMResponse, Message, ToolDefinition
from captain_claw.session import Session
from captain_claw.tools.registry import ToolRegistry


class DummySessionManager:
    async def save_session(self, session: Session) -> None:
        return None


class ProcreateSessionManager:
    def __init__(self):
        self._counter = 0

    async def create_session(self, name: str, metadata: dict[str, Any] | None = None) -> Session:
        self._counter += 1
        return Session(
            id=f"child-{self._counter}",
            name=name,
            metadata=metadata or {},
        )

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


class DescriptionProvider(LLMProvider):
    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        return LLMResponse(
            content=(
                "Primary goal is API integration hardening. "
                "The session includes debugging auth failures and retries. "
                "It uses shell and file operations to validate fixes. "
                "Current focus is stabilizing edge-case error handling. "
                "Pending work includes regression verification. "
                "This sentence should be trimmed."
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
        return len([part for part in text.split() if part]) or 1


def _write_skill(skill_dir: Path, name: str, description: str, extra_frontmatter: str = "") -> None:
    skill_dir.mkdir(parents=True, exist_ok=True)
    frontmatter = (
        "---\n"
        f"name: {name}\n"
        f"description: {description}\n"
        f"{extra_frontmatter}"
        "---\n\n"
        f"# {name}\n\n"
        "Use this skill.\n"
    )
    (skill_dir / "SKILL.md").write_text(frontmatter, encoding="utf-8")


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
    assert "Workspace root path:" in prompt
    assert "All tool-generated files must be written inside:" in prompt
    assert 'Current session subfolder id: "s1".' in prompt
    assert "saved/scripts/s1/" in prompt
    assert "saved/tools/s1/" in prompt
    assert "saved/downloads/s1/" in prompt
    assert "saved/media/s1/" in prompt
    assert "saved/showcase/s1/" in prompt
    assert "saved/skills/s1/" in prompt
    assert "saved/tmp/s1/" in prompt
    assert "Script/tool generation workflow:" in prompt


def test_system_prompt_includes_available_skills_block(tmp_path: Path):
    old_cfg = get_config().model_copy(deep=True)
    cfg = old_cfg.model_copy(deep=True)
    cfg.workspace.path = str(tmp_path)
    set_config(cfg)
    try:
        _write_skill(
            tmp_path / "skills" / "demo-skill",
            "demo-skill",
            "Demo skill for testing prompt injection.",
        )
        agent = Agent(provider=TokenAwareProvider())
        agent.session = Session(id="s1", name="default")

        prompt = agent._build_system_prompt()

        assert "## Skills (mandatory)" in prompt
        assert "<available_skills>" in prompt
        assert "<name>demo-skill</name>" in prompt
    finally:
        set_config(old_cfg)


def test_disable_model_invocation_skill_hidden_from_prompt_but_still_invocable(tmp_path: Path):
    old_cfg = get_config().model_copy(deep=True)
    cfg = old_cfg.model_copy(deep=True)
    cfg.workspace.path = str(tmp_path)
    set_config(cfg)
    try:
        _write_skill(
            tmp_path / "skills" / "hidden-skill",
            "hidden-skill",
            "Hidden from model prompt but manually invocable.",
            extra_frontmatter="disable-model-invocation: true\nuser-invocable: true\n",
        )
        agent = Agent(provider=TokenAwareProvider())
        agent.session = Session(id="s1", name="default")

        prompt = agent._build_system_prompt()
        assert "<name>hidden-skill</name>" not in prompt

        commands = agent.list_user_invocable_skills()
        assert any(cmd.skill_name == "hidden-skill" for cmd in commands)
    finally:
        set_config(old_cfg)


@pytest.mark.asyncio
async def test_invoke_skill_command_rewrites_prompt(tmp_path: Path):
    old_cfg = get_config().model_copy(deep=True)
    cfg = old_cfg.model_copy(deep=True)
    cfg.workspace.path = str(tmp_path)
    set_config(cfg)
    try:
        _write_skill(
            tmp_path / "skills" / "docs",
            "docs",
            "Documentation writing support.",
        )
        agent = Agent(provider=TokenAwareProvider())
        agent._initialized = True
        agent.session = Session(id="s1", name="default")
        agent.session_manager = DummySessionManager()

        invocation = await agent.invoke_skill_command("docs", args="draft release notes")

        assert invocation["ok"] is True
        assert invocation["mode"] == "rewrite"
        assert 'Use the "docs" skill for this request.' in str(invocation["prompt"])
        assert "draft release notes" in str(invocation["prompt"])
    finally:
        set_config(old_cfg)


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


def test_normalize_contract_tasks_supports_nested_children():
    raw_tasks = [
        {
            "title": "Phase 1",
            "children": [
                {"title": "Subtask A"},
                {"title": "Subtask B"},
            ],
        },
        {"title": "Phase 2"},
    ]

    normalized = Agent._normalize_contract_tasks(raw_tasks, max_tasks=8)

    assert len(normalized) == 2
    assert normalized[0]["title"] == "Phase 1"
    assert isinstance(normalized[0].get("children"), list)
    assert len(normalized[0]["children"]) == 2
    assert normalized[0]["children"][0]["title"] == "Subtask A"
    assert normalized[0]["children"][1]["title"] == "Subtask B"


def test_nested_pipeline_progress_tracks_leaf_order_and_parent_rollup():
    agent = Agent(provider=TokenAwareProvider())
    pipeline = agent._build_task_pipeline(
        "do nested work",
        tasks_override=[
            {
                "title": "Phase 1",
                "children": [
                    {"title": "Collect"},
                    {"title": "Analyze"},
                ],
            },
            {"title": "Phase 2"},
        ],
    )

    task_order = list(pipeline.get("task_order", []))
    assert len(task_order) == 3
    assert pipeline.get("current_task_id") == task_order[0]

    agent._advance_pipeline(pipeline, event="test_advance_1")
    assert pipeline.get("current_task_id") == task_order[1]
    assert pipeline["tasks"][0]["status"] == "in_progress"
    assert pipeline["tasks"][0]["children"][0]["status"] == "completed"
    assert pipeline["tasks"][0]["children"][1]["status"] == "in_progress"

    agent._advance_pipeline(pipeline, event="test_advance_2")
    assert pipeline.get("current_task_id") == task_order[2]
    assert pipeline["tasks"][0]["status"] == "completed"
    assert pipeline["tasks"][1]["status"] == "in_progress"


def test_pipeline_dag_dependencies_block_until_parent_leaf_completes():
    agent = Agent(provider=TokenAwareProvider())
    pipeline = agent._build_task_pipeline(
        "dag dependency test",
        tasks_override=[
            {"id": "task_a", "title": "First"},
            {"id": "task_b", "title": "Second", "depends_on": ["task_a"]},
            {"id": "task_c", "title": "Independent"},
        ],
    )

    blocked = set(pipeline.get("blocked_task_ids", []))
    assert "task_b" in blocked

    # Advance once: first active task completes, dependency should unlock.
    agent._advance_pipeline(pipeline, event="dependency_advance")
    status_b = pipeline["task_graph"]["task_b"]["status"]
    assert status_b in {"ready", "in_progress", "completed"}


def test_pipeline_runtime_timeout_retries_then_fails_task():
    agent = Agent(provider=TokenAwareProvider())
    pipeline = agent._build_task_pipeline(
        "timeout retry test",
        tasks_override=[
            {
                "id": "task_timeout",
                "title": "May timeout",
                "timeout_seconds": 0.001,
                "max_retries": 1,
            }
        ],
    )
    node = pipeline["task_graph"]["task_timeout"]
    node["status"] = "in_progress"
    node["started_at"] = (datetime.now(UTC) - timedelta(seconds=2)).isoformat()
    pipeline["active_task_ids"] = ["task_timeout"]
    pipeline["ready_task_ids"] = []

    first_tick = agent._tick_pipeline_runtime(pipeline, event="timeout_tick_1")
    assert first_tick["changed"] is True
    assert "task_timeout" in first_tick["timed_out"]
    assert "task_timeout" in first_tick["retried"]
    assert int(node.get("retries", 0)) == 1

    node["status"] = "in_progress"
    node["started_at"] = (datetime.now(UTC) - timedelta(seconds=2)).isoformat()
    pipeline["active_task_ids"] = ["task_timeout"]
    pipeline["ready_task_ids"] = []

    second_tick = agent._tick_pipeline_runtime(pipeline, event="timeout_tick_2")
    assert second_tick["changed"] is True
    assert "task_timeout" in second_tick["timed_out"]
    assert "task_timeout" in second_tick["failed"]
    assert pipeline["task_graph"]["task_timeout"]["status"] == "failed"


def test_pipeline_progress_details_include_nested_scope_remaining_counts():
    agent = Agent(provider=TokenAwareProvider())
    pipeline = agent._build_task_pipeline(
        "do nested work",
        tasks_override=[
            {
                "id": "task_1",
                "title": "Phase 1",
                "children": [
                    {"id": "task_1_1", "title": "Collect"},
                    {
                        "id": "task_1_2",
                        "title": "Analyze",
                        "children": [
                            {"id": "task_1_2_1", "title": "Model"},
                            {"id": "task_1_2_2", "title": "Validate"},
                        ],
                    },
                ],
            },
            {"id": "task_2", "title": "Phase 2"},
        ],
    )
    agent._set_pipeline_progress(pipeline, current_index=1, current_status="in_progress")
    pipeline["created_at"] = (datetime.now(UTC) - timedelta(seconds=120)).isoformat()

    progress = Agent._build_pipeline_progress_details(pipeline)

    assert progress["leaf_index"] == 2
    assert progress["leaf_total"] == 4
    assert progress["leaf_remaining"] == 2
    assert progress["current_path"] == "1.2.1"
    scopes = progress["scope_progress"]
    assert len(scopes) == 3
    assert scopes[0]["path"] == "1"
    assert scopes[0]["scope_leaf_total"] == 4
    assert scopes[0]["scope_leaf_remaining"] == 2
    assert scopes[1]["path"] == "1.2"
    assert scopes[1]["scope_leaf_total"] == 3
    assert scopes[1]["scope_leaf_remaining"] == 1
    assert scopes[2]["path"] == "1.2.1"
    assert scopes[2]["scope_leaf_total"] == 2
    assert scopes[2]["scope_leaf_remaining"] == 1
    assert progress["eta_seconds"] is not None
    assert float(progress["eta_seconds"]) > 0
    assert progress["eta_text"] != "unknown"
    assert scopes[0]["eta_text"] != "unknown"


def test_build_pipeline_note_renders_nested_numbering():
    agent = Agent(provider=TokenAwareProvider())
    pipeline = {
        "tasks": [
            {
                "id": "task_1",
                "title": "Phase 1",
                "status": "in_progress",
                "children": [
                    {"id": "task_2", "title": "Collect data", "status": "completed"},
                    {"id": "task_3", "title": "Analyze data", "status": "in_progress"},
                ],
            },
            {"id": "task_4", "title": "Phase 2", "status": "pending"},
        ],
        "current_index": 1,
        "state": "active",
    }

    note = agent._build_pipeline_note(pipeline)

    assert "1. [IN_PROGRESS] Phase 1" in note
    assert "1.1. [COMPLETED] Collect data" in note
    assert "1.2. [IN_PROGRESS] Analyze data" in note
    assert "2. [PENDING] Phase 2" in note


def test_build_messages_excludes_llm_trace_tool_messages():
    agent = Agent(provider=TokenAwareProvider())
    agent.session = Session(id="s1", name="default")
    agent.session.add_message("user", "previous request")
    agent.session.add_message(
        role="tool",
        content="interaction=turn_1\n[assistant_response]\ninternal draft",
        tool_name="llm_trace",
        tool_arguments={"interaction": "turn_1"},
    )
    agent.session.add_message("assistant", "previous answer")

    messages = agent._build_messages(query="new request")

    assert all((m.tool_name or "") != "llm_trace" for m in messages)


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
    created = next(entry for entry in planning_logs if entry["args"].get("event") == "created")
    assert "leaf_index" in created["args"]
    assert "leaf_remaining" in created["args"]
    assert "eta_seconds" in created["args"]
    assert "eta_text" in created["args"]
    assert "scope_progress" in created["args"]
    assert "progress=" in created["output"]
    assert "eta=" in created["output"]
    assert "scope_progress:" in created["output"]


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


@pytest.mark.asyncio
async def test_generate_session_description_uses_context_and_limits_to_five_sentences():
    agent = Agent(provider=DescriptionProvider())
    agent.session = Session(id="s1", name="debug-thread")
    agent.session.add_message("user", "Fix auth timeout failures and flaky retries.")
    agent.session.add_message("tool", "stack trace output", tool_name="shell")
    agent.session.add_message("assistant", "I will patch retry behavior and run tests.")

    description = await agent.generate_session_description(max_sentences=5)

    assert description
    # 5 sentences max.
    sentence_count = len([s for s in description.split(". ") if s.strip()])
    assert sentence_count <= 5
    assert "API integration hardening" in description
    assert "auth failures" in description


def test_sanitize_session_description_limits_sentences():
    agent = Agent(provider=TokenAwareProvider())
    raw = "One. Two. Three. Four. Five. Six."
    cleaned = agent.sanitize_session_description(raw, max_sentences=5)
    assert cleaned.endswith("Five.")
    assert "Six." not in cleaned


@pytest.mark.asyncio
async def test_set_session_model_uses_allowed_models_and_updates_metadata():
    old_cfg = get_config().model_copy(deep=True)
    cfg = old_cfg.model_copy(deep=True)
    cfg.model.provider = "ollama"
    cfg.model.model = "minimax-m2.5:cloud"
    cfg.model.base_url = "http://localhost:11434"
    cfg.model.allowed = [
        {
            "id": "ollama-cloud",
            "provider": "ollama",
            "model": "minimax-m2.5:cloud",
        },
        {
            "id": "chatgpt-fast",
            "provider": "openai",
            "model": "gpt-4o-mini",
        },
    ]
    set_config(cfg)
    try:
        agent = Agent(provider=TokenAwareProvider())
        agent.session = Session(id="s1", name="default")
        agent.session_manager = DummySessionManager()

        ok, message = await agent.set_session_model("chatgpt-fast", persist=True)
        assert ok is True
        assert "openai" in message
        selection = agent.session.metadata.get("model_selection")
        assert isinstance(selection, dict)
        assert selection.get("id") == "chatgpt-fast"
        assert str(selection.get("provider", "")).startswith("openai")
        assert "gpt-4o-mini" in str(selection.get("model", ""))

        details = agent.get_runtime_model_details()
        assert str(details.get("provider", "")).startswith("openai")
        assert "gpt-4o-mini" in str(details.get("model", ""))
        assert str(details.get("base_url", "")).strip() == ""
        assert str(selection.get("base_url", "")).strip() == ""

        ok_default, message_default = await agent.set_session_model("default", persist=True)
        assert ok_default is True
        assert "default config" in message_default
        assert "model_selection" not in agent.session.metadata
    finally:
        set_config(old_cfg)


def test_supports_tool_followup_uses_runtime_model_details():
    old_cfg = get_config().model_copy(deep=True)
    cfg = old_cfg.model_copy(deep=True)
    cfg.model.provider = "openai"
    cfg.model.model = "gpt-4o-mini"
    set_config(cfg)
    try:
        agent = Agent(provider=TokenAwareProvider())
        agent._runtime_model_details = {"provider": "ollama", "model": "minimax-m2.5:cloud"}
        assert agent._supports_tool_result_followup() is False
    finally:
        set_config(old_cfg)


@pytest.mark.asyncio
async def test_set_session_memory_protection_updates_metadata():
    agent = Agent(provider=TokenAwareProvider())
    agent.session = Session(id="s1", name="default")
    agent.session_manager = DummySessionManager()

    assert agent.is_session_memory_protected() is False

    ok_on, message_on = await agent.set_session_memory_protection(True, persist=True)
    assert ok_on is True
    assert "enabled" in message_on
    assert agent.is_session_memory_protected() is True

    ok_off, message_off = await agent.set_session_memory_protection(False, persist=True)
    assert ok_off is True
    assert "disabled" in message_off
    assert agent.is_session_memory_protected() is False


@pytest.mark.asyncio
async def test_procreate_sessions_compacts_and_keeps_parent_memory_unchanged():
    old_cfg = get_config().model_copy(deep=True)
    cfg = old_cfg.model_copy(deep=True)
    cfg.context.max_tokens = 40
    cfg.context.compaction_ratio = 0.2
    set_config(cfg)
    try:
        monitor_outputs: list[dict[str, Any]] = []

        def cb(name: str, args: dict, output: str) -> None:
            monitor_outputs.append({"name": name, "args": args, "output": output})

        agent = Agent(provider=TokenAwareProvider(), tool_output_callback=cb)
        agent.session_manager = ProcreateSessionManager()

        parent_one = Session(id="parent-1", name="alpha")
        parent_two = Session(id="parent-2", name="beta")
        for idx in range(10):
            parent_one.add_message("user", f"alpha msg {idx} one two three four")
            parent_two.add_message("assistant", f"beta msg {idx} one two three four")

        parent_one_before = copy.deepcopy(parent_one.messages)
        parent_two_before = copy.deepcopy(parent_two.messages)

        child, stats = await agent.procreate_sessions(
            parent_one=parent_one,
            parent_two=parent_two,
            new_name="child memory",
            persist=True,
        )

        assert child.name == "child memory"
        assert stats["merged_messages"] == len(child.messages)
        assert stats["parent_one_compacted"] > 0
        assert stats["parent_two_compacted"] > 0
        assert any(msg.get("tool_name") == "compaction_summary" for msg in child.messages)
        assert parent_one.messages == parent_one_before
        assert parent_two.messages == parent_two_before
        procreate_logs = [entry for entry in monitor_outputs if entry["name"] == "session_procreate"]
        assert procreate_logs
        steps = [str(entry.get("args", {}).get("step", "")) for entry in procreate_logs]
        assert "start" in steps
        assert "done" in steps
        assert any("compact_parent_one" in step for step in steps)
        assert any("compact_parent_two" in step for step in steps)
    finally:
        set_config(old_cfg)
