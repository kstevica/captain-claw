from pathlib import Path

from captain_claw.agent import Agent
from captain_claw.llm import LLMProvider, LLMResponse, Message, ToolDefinition
from captain_claw.memory import LayeredMemory, WorkingMemory
from captain_claw.semantic_memory import SemanticMemoryIndex
from captain_claw.session import Session


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
        return len(text.split()) or 1


def test_working_memory_compaction_keeps_recent_messages():
    memory = WorkingMemory(max_tokens=120)
    for idx in range(10):
        memory.add_message("user", f"old-to-new message {idx}")

    memory.compact(keep_recent_ratio=0.3)
    snapshot = memory.snapshot()
    messages = snapshot.messages

    # Includes one summary + most recent 3 detailed messages.
    assert len(messages) == 4
    assert "Conversation summary from older context" in messages[0]["content"]
    assert "message 7" in messages[1]["content"]
    assert "message 8" in messages[2]["content"]
    assert "message 9" in messages[3]["content"]


def test_semantic_memory_indexes_and_searches_keyword_hits(tmp_path: Path):
    db_path = tmp_path / "memory.db"
    session_db = tmp_path / "sessions.db"
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    session_db.touch()

    index = SemanticMemoryIndex(
        db_path=db_path,
        session_db_path=session_db,
        workspace_path=workspace,
        index_workspace=False,
        index_sessions=False,
        min_score=0.0,
        temporal_decay_enabled=False,
    )
    try:
        index.upsert_text(
            source="session",
            reference="session-1",
            path="sessions/analysis.txt",
            text="CSV analysis from yesterday found a spike in conversion rates.",
        )
        index.set_active_session("session-1")
        results = index.search("csv analysis yesterday", max_results=3)
    finally:
        index.close()

    assert results
    assert results[0].source == "session"
    assert "CSV analysis" in results[0].snippet


def test_semantic_memory_scopes_session_results_to_active_session_by_default(tmp_path: Path):
    db_path = tmp_path / "memory.db"
    session_db = tmp_path / "sessions.db"
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    session_db.touch()

    index = SemanticMemoryIndex(
        db_path=db_path,
        session_db_path=session_db,
        workspace_path=workspace,
        index_workspace=False,
        index_sessions=False,
        min_score=0.0,
        temporal_decay_enabled=False,
    )
    try:
        index.upsert_text(
            source="session",
            reference="session-old",
            path="sessions/old.txt",
            text="hello from prior session context",
        )
        index.upsert_text(
            source="session",
            reference="session-new",
            path="sessions/new.txt",
            text="hello from active session context",
        )
        index.set_active_session("session-new")
        results = index.search("hello", max_results=10)
    finally:
        index.close()

    session_refs = {r.reference for r in results if r.source == "session"}
    assert "session-new" in session_refs
    assert "session-old" not in session_refs


def test_semantic_memory_can_opt_in_to_cross_session_retrieval(tmp_path: Path):
    db_path = tmp_path / "memory.db"
    session_db = tmp_path / "sessions.db"
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    session_db.touch()

    index = SemanticMemoryIndex(
        db_path=db_path,
        session_db_path=session_db,
        workspace_path=workspace,
        index_workspace=False,
        index_sessions=False,
        cross_session_retrieval=True,
        min_score=0.0,
        temporal_decay_enabled=False,
    )
    try:
        index.upsert_text(
            source="session",
            reference="session-old",
            path="sessions/old.txt",
            text="hello from prior session context",
        )
        index.upsert_text(
            source="session",
            reference="session-new",
            path="sessions/new.txt",
            text="hello from active session context",
        )
        index.set_active_session("session-new")
        results = index.search("hello", max_results=10)
    finally:
        index.close()

    session_refs = {r.reference for r in results if r.source == "session"}
    assert "session-new" in session_refs
    assert "session-old" in session_refs


def test_build_messages_includes_semantic_memory_note(tmp_path: Path):
    db_path = tmp_path / "memory.db"
    session_db = tmp_path / "sessions.db"
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    session_db.touch()

    semantic = SemanticMemoryIndex(
        db_path=db_path,
        session_db_path=session_db,
        workspace_path=workspace,
        index_workspace=False,
        index_sessions=False,
        min_score=0.0,
        temporal_decay_enabled=False,
    )
    semantic.upsert_text(
        source="workspace",
        reference="reports/conversions.md",
        path="reports/conversions.md",
        text="Yesterday CSV analysis: conversion rate increased by 15 percent for paid traffic.",
    )

    agent = Agent(provider=DummyProvider())
    agent.session = Session(id="s1", name="default")
    agent.session.add_message("user", "Need a quick follow-up.")
    agent.memory = LayeredMemory(working_memory=WorkingMemory(), semantic_memory=semantic)

    try:
        messages = agent._build_messages(query="that csv analysis from yesterday")
    finally:
        semantic.close()

    semantic_notes = [msg.content for msg in messages if "Semantic memory matches" in msg.content]
    assert semantic_notes
    assert "reports/conversions.md" in semantic_notes[0]


def _make_index(tmp_path: Path, **kwargs) -> SemanticMemoryIndex:
    """Helper to create a SemanticMemoryIndex with sensible test defaults."""
    db_path = tmp_path / "memory.db"
    session_db = tmp_path / "sessions.db"
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    session_db.touch()
    defaults = dict(
        db_path=db_path,
        session_db_path=session_db,
        workspace_path=workspace,
        index_workspace=False,
        index_sessions=False,
        min_score=0.0,
        temporal_decay_enabled=False,
    )
    defaults.update(kwargs)
    return SemanticMemoryIndex(**defaults)


def test_layered_summaries_stored_and_returned_in_search(tmp_path: Path):
    """When a summarizer is set, L1/L2 texts are stored and returned by search."""
    index = _make_index(tmp_path, layered_summaries=True)

    def fake_summarizer(text: str) -> tuple[str, str]:
        return "L1: headline", "L2: a short summary of the chunk"

    index.set_summarizer(fake_summarizer)
    try:
        index.upsert_text(
            source="workspace",
            reference="doc1.md",
            path="doc1.md",
            text="Long document about machine learning algorithms and their applications.",
        )
        results = index.search("machine learning", max_results=3)
    finally:
        index.close()

    assert results
    assert results[0].text_l1 == "L1: headline"
    assert results[0].text_l2 == "L2: a short summary of the chunk"
    assert "machine learning" in results[0].snippet.lower()


def test_build_context_note_uses_layer_parameter(tmp_path: Path):
    """build_context_note returns different text depending on the layer."""
    index = _make_index(tmp_path, layered_summaries=True)

    def fake_summarizer(text: str) -> tuple[str, str]:
        return "one-liner headline", "medium summary of the content"

    index.set_summarizer(fake_summarizer)
    try:
        index.upsert_text(
            source="workspace",
            reference="doc.md",
            path="doc.md",
            text="Detailed text about database sharding strategies for horizontal scaling.",
        )
        note_l1, _ = index.build_context_note("sharding", layer="l1")
        note_l2, _ = index.build_context_note("sharding", layer="l2")
        note_l3, _ = index.build_context_note("sharding", layer="l3")
    finally:
        index.close()

    assert "one-liner headline" in note_l1
    assert "medium summary" in note_l2
    assert "database sharding" in note_l3.lower()


def test_promote_returns_requested_layer(tmp_path: Path):
    """promote() fetches specific chunks at the requested detail level."""
    index = _make_index(tmp_path, layered_summaries=True)

    def fake_summarizer(text: str) -> tuple[str, str]:
        return "L1 headline", "L2 summary text"

    index.set_summarizer(fake_summarizer)
    try:
        index.upsert_text(
            source="workspace",
            reference="doc.md",
            path="doc.md",
            text="Full content about API rate limiting and throttling mechanisms.",
        )
        results = index.search("rate limiting", max_results=1)
        assert results
        chunk_id = results[0].chunk_id

        promoted_l1 = index.promote([chunk_id], layer="l1")
        promoted_l3 = index.promote([chunk_id], layer="l3")
    finally:
        index.close()

    assert promoted_l1
    assert promoted_l1[0].snippet == "L1 headline"
    assert promoted_l3
    assert "rate limiting" in promoted_l3[0].snippet.lower()


def test_no_summarizer_falls_back_gracefully(tmp_path: Path):
    """Without a summarizer, L1/L2 are empty and search still works."""
    index = _make_index(tmp_path, layered_summaries=True)
    # No summarizer set
    try:
        index.upsert_text(
            source="workspace",
            reference="doc.md",
            path="doc.md",
            text="Content about caching strategies in distributed systems.",
        )
        results = index.search("caching", max_results=3)
    finally:
        index.close()

    assert results
    assert results[0].text_l1 == ""
    assert results[0].text_l2 == ""
    assert "caching" in results[0].snippet.lower()


def test_build_context_note_l1_falls_back_to_l3_when_no_summaries(tmp_path: Path):
    """When L1/L2 are empty, build_context_note at layer=l1 falls back to L3 text."""
    index = _make_index(tmp_path, layered_summaries=False)
    try:
        index.upsert_text(
            source="workspace",
            reference="doc.md",
            path="doc.md",
            text="Kubernetes pod scheduling and resource allocation details.",
        )
        note, _ = index.build_context_note("kubernetes", layer="l1")
    finally:
        index.close()

    # Should fall back to L3 text since no L1 exists.
    assert "Kubernetes" in note or "kubernetes" in note.lower()
