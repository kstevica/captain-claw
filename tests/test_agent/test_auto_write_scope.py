"""The auto-save of a reply only follows the person's own words in THIS message.

Seen 2026-11-07 while recording the user tutorial: the agent saved
``saved/where-we-stand.md`` (10 lines), its reply ended with an offer
("Would you like me to check …?"), the person clicked the suggested next step,
and the model answered in plain text. That follow-up was merged with the
previous reply (a pinned clarification), the reply named the note's path, and
the turn ended with "Written 142 chars (1 lines) to where-we-stand.md …
Overwriting existing file": the note became the one-line answer. A second
agent did the same with research forwarded to it ("--- Context from … ---").
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from captain_claw.agent import Agent
from captain_claw.agent_file_ops_mixin import AgentFileOpsMixin, own_request_words
from captain_claw.llm import LLMProvider, LLMResponse, Message, ToolCall, ToolDefinition
from captain_claw.session import Session
from captain_claw.tools.registry import ToolRegistry
from captain_claw.tools.write import KEEP_EXISTING_ERROR, WriteTool

NOTE = (
    "# Where we stand\n\n"
    "- Our monthly price: 24.99\n"
    "- Dance: 22.99, 12-month minimum term\n"
    "- Beat: 26.49, no minimum term\n\n"
    "We sit in the middle of the market.\n"
    "Dance is cheaper but locks people in.\n"
    "Beat costs more and lets people leave.\n"
    "Next: watch Dance's minimum term.\n"
)
# A reply that names the note's path next to a save word ("export"), and ends
# with an offer, which pins a clarification for the next message.
OFFER = (
    "I charted our monthly price against Dance and Beat and wrote a one-page "
    "note to saved/where-we-stand.md, ready to export or share. Would you like "
    "me to check whether Dance changed its minimum term this month?"
)
ANSWER = "Dance kept its 12-month minimum term this month; only its price changed."


class DummySessionManager:
    async def save_session(self, session: Session) -> None:
        return None


def _text(content) -> str:
    if isinstance(content, list):
        return " ".join(str(p.get("text", "")) for p in content if isinstance(p, dict))
    return str(content or "")


class ScriptedProvider(LLMProvider):
    """Answers by the person's newest message, like the tutorial's fake model:
    the note request writes the note with the write tool, then replies with an
    offer that ends in a question; anything else gets a plain-text answer."""

    def __init__(self, write_path: str = "saved/where-we-stand.md"):
        self.write_path = write_path

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        system_text = _text(messages[0].content if messages else "").lower()
        if "strict completion critic" in system_text:
            return LLMResponse(content=json.dumps(
                {"complete": True, "checks": [{"id": "req_done", "ok": True, "reason": ""}],
                 "feedback": ""}))
        if "task-contract planner" in system_text:
            return LLMResponse(content=json.dumps(
                {"summary": "Answer", "tasks": [{"title": "Answer"}],
                 "requirements": [{"id": "req_done", "title": "Return response"}],
                 "prefetch_urls": []}))
        said, at = "", -1
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].role == "user":
                said, at = _text(messages[i].content), i
                break
        results = sum(1 for m in messages[at + 1:] if m.role == "tool")
        if "chart our monthly price" in said.lower():
            if results == 0 and tools:
                return LLMResponse(content="", tool_calls=[ToolCall(
                    id="call_note", name="write",
                    arguments={"path": self.write_path, "content": NOTE})])
            return LLMResponse(content=OFFER)
        return LLMResponse(content=ANSWER)

    async def complete_streaming(self, messages, tools=None, temperature=None, max_tokens=None):
        if False:
            yield ""

    def count_tokens(self, text: str) -> int:
        return len(text)


def _agent(tmp_path: Path, monkeypatch, session_id: str = "s1") -> Agent:
    # The agent anchors its workspace (./workspace/saved/…) on the working
    # directory: keep every file this test makes under tmp_path.
    monkeypatch.chdir(tmp_path)
    agent = Agent(provider=ScriptedProvider())
    agent._initialized = True
    agent.session = Session(id=session_id, name="default")
    agent.session_manager = DummySessionManager()
    registry = ToolRegistry(base_path=tmp_path)
    registry.register(WriteTool())
    agent.tools = registry
    return agent


def _notes(root: Path) -> list[Path]:
    return sorted(root.rglob("where-we-stand.md"))


def _writes_since(agent: Agent, idx: int) -> list[dict]:
    return [m for m in agent.session.messages[idx:]
            if m.get("role") == "tool" and m.get("tool_name") == "write"]


# ── own_request_words ────────────────────────────────────────────────


def test_plain_message_is_kept_whole():
    msg = "Save a short summary to file summary.md"
    assert own_request_words(msg) == msg
    assert AgentFileOpsMixin._extract_requested_write_paths(own_request_words(msg)) == ["summary.md"]


def test_kiosk_forward_keeps_only_the_task_after_the_context():
    msg = (
        '--- Context from "Research" ---\n\n'
        f"**Assistant:**\n{OFFER}\n\n"
        "--- End of context ---\n\n"
        "Review the context above and provide your analysis."
    )
    words = own_request_words(msg)
    assert words == "Review the context above and provide your analysis."
    assert AgentFileOpsMixin._extract_requested_write_paths(words) == []
    # The raw text would have asked for the note's path.
    assert AgentFileOpsMixin._extract_requested_write_paths(msg) == ["saved/where-we-stand.md"]


def test_transferred_conversation_keeps_a_path_the_task_names():
    msg = (
        '--- Context transferred from another AI agent "Research" '
        "(last 2 messages from their conversation) ---\n\n"
        "**User:**\nSave a one-page note to saved/where-we-stand.md\n\n"
        "**Assistant:**\nDone.\n\n"
        "--- End of context ---\n\n"
        "Save your own take to file our-take.md"
    )
    words = own_request_words(msg)
    assert AgentFileOpsMixin._extract_requested_write_paths(words) == ["our-take.md"]


def test_clarification_merge_drops_the_previous_reply():
    merged = (
        "Check whether Dance changed its minimum term this month\n\n"
        "Context from the previous assistant response:\n"
        f"{OFFER}\n\n"
        "Execute the user's follow-up request using the context above."
    )
    words = own_request_words(merged)
    assert words == "Check whether Dance changed its minimum term this month"
    assert AgentFileOpsMixin._extract_requested_write_paths(words) == []


def test_clarification_fallback_keeps_only_the_new_words():
    merged = (
        "Write the report and save it to report.md\n\n"
        "Clarifications/preferences from user:\n"
        "the second one\n\n"
        "Execute the full original request using these clarification details."
    )
    assert own_request_words(merged) == "the second one"


def test_pipeline_output_internal_context_and_quotes_are_not_own_words():
    pipeline = (
        '--- Pipeline: "Weekly" — Output from "Research" ---\n\n'
        "The following is output from another agent. Save it to file out.md"
    )
    assert own_request_words(pipeline) == ""
    internal = "[INTERNAL CONTEXT — todos]\nsave the list to todo.md\n[END INTERNAL CONTEXT]\nWhat's next?"
    assert own_request_words(internal) == "What's next?"
    quoted = "> please save this to file draft.md\nWhat does this email want?"
    assert AgentFileOpsMixin._extract_requested_write_paths(own_request_words(quoted)) == []
    fenced = "What does this do?\n```\nexport the table to file rows.csv\n```"
    assert AgentFileOpsMixin._extract_requested_write_paths(own_request_words(fenced)) == []


def test_bare_rules_and_front_matter_are_not_frames():
    msg = "Intro\n---\nSave the list to file list.md\n-------\nThanks"
    assert "Save the list to file list.md" in own_request_words(msg)


# ── the write tool's keep-existing refusal ───────────────────────────


@pytest.mark.asyncio
async def test_write_tool_keeps_an_existing_file_when_asked(tmp_path: Path):
    tool = WriteTool()
    kw = {"_runtime_base_path": tmp_path, "_saved_base_path": tmp_path / "saved", "_session_id": "s1"}
    made = await tool.execute(path="note.md", content=NOTE, **kw)
    assert made.success
    target = next((tmp_path / "saved").rglob("note.md"))

    refused = await tool.execute(path="note.md", content=ANSWER, _keep_existing=True, **kw)
    assert not refused.success
    assert refused.error == KEEP_EXISTING_ERROR
    assert target.read_text(encoding="utf-8") == NOTE

    # A new file is still made with the flag on.
    fresh = await tool.execute(path="fresh.md", content=ANSWER, _keep_existing=True, **kw)
    assert fresh.success


# ── whole turns ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_follow_up_turn_does_not_overwrite_the_earlier_note(tmp_path: Path, monkeypatch):
    agent = _agent(tmp_path, monkeypatch)

    first = await agent.complete(
        "Chart our monthly price against Dance and Beat, and save a one-page "
        "note on where we stand")
    notes = _notes(tmp_path)
    assert len(notes) == 1, first
    note = notes[0]
    assert note.read_text(encoding="utf-8") == NOTE
    # The offer pinned a clarification, so the next message is merged with it.
    assert agent.session.metadata["clarification_state"]["pending"] is True

    turn_start = len(agent.session.messages)
    reply = await agent.complete("Check whether Dance changed its minimum term this month")

    assert note.read_text(encoding="utf-8") == NOTE
    assert _notes(tmp_path) == [note]
    assert _writes_since(agent, turn_start) == []
    assert ANSWER in reply
    assert "Written" not in reply and "Overwr" not in reply


@pytest.mark.asyncio
async def test_forwarded_context_naming_a_path_does_not_steer_the_auto_save(
    tmp_path: Path, monkeypatch,
):
    agent = _agent(tmp_path, monkeypatch, session_id="s2")
    forwarded = (
        '--- Context from "Research" ---\n\n'
        f"**Assistant:**\n{OFFER}\n\n"
        "--- End of context ---\n\n"
        "Review the context above and provide your analysis."
    )

    reply = await agent.complete(forwarded)

    assert _notes(tmp_path) == []
    assert _writes_since(agent, 0) == []
    assert ANSWER in reply
    assert "Written" not in reply


@pytest.mark.asyncio
async def test_asking_again_for_the_same_file_keeps_the_earlier_one(tmp_path: Path, monkeypatch):
    agent = _agent(tmp_path, monkeypatch)
    await agent.complete(
        "Chart our monthly price against Dance and Beat, and save a one-page "
        "note on where we stand")
    note = _notes(tmp_path)[0]
    # A plain request, so no clarification merge is in play.
    agent.session.metadata.pop("clarification_state", None)

    turn_start = len(agent.session.messages)
    reply = await agent.complete(
        "Look up Dance's minimum term and save the answer to saved/where-we-stand.md")

    assert note.read_text(encoding="utf-8") == NOTE
    assert "already exists" in reply
    assert "Written" not in reply
    writes = _writes_since(agent, turn_start)
    assert len(writes) == 1
    # Recorded as a failed write: no turn gate counts it as a file made.
    assert str(writes[0]["content"]).startswith("Error:")
    assert not agent._turn_has_successful_tool(turn_start, "write")


@pytest.mark.asyncio
async def test_own_words_request_for_a_new_file_still_auto_saves(tmp_path: Path, monkeypatch):
    agent = _agent(tmp_path, monkeypatch)

    reply = await agent.complete("Check Dance's minimum term and save the answer to file dance.md")

    saved = sorted(tmp_path.rglob("dance.md"))
    assert len(saved) == 1
    assert saved[0].read_text(encoding="utf-8") == ANSWER
    assert "Written" in reply
