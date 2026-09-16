"""Increment 2 — bounded auto-correction of malformed / cut-off / guard-refused tool calls."""

import pytest

from captain_claw.agent import Agent
from captain_claw.llm import LLMProvider, LLMResponse, Message, ToolCall, ToolDefinition
from captain_claw.session import Session
from captain_claw.tools.registry import ToolRegistry
from captain_claw.tools.write import WriteTool


class _Dummy(LLMProvider):
    async def complete(self, messages, tools=None, temperature=None, max_tokens=None):
        return LLMResponse(content="ok")

    async def complete_streaming(self, messages, tools=None, temperature=None, max_tokens=None):
        if False:
            yield ""

    def count_tokens(self, text: str) -> int:
        return len(text)


class _DummySM:
    async def save_session(self, session) -> None:
        return None


def _agent(tmp_path):
    agent = Agent(provider=_Dummy())
    agent._initialized = True
    agent.session = Session(id="s1", name="default")
    agent.session_manager = _DummySM()
    registry = ToolRegistry(base_path=tmp_path)
    registry.register(WriteTool())
    agent.tools = registry
    agent._malformed_retry_count = 0
    agent._blind_write_paths = set()
    return agent


@pytest.mark.asyncio
async def test_malformed_json_write_is_corrected_not_executed(tmp_path, monkeypatch):
    monkeypatch.setenv("CLAW_MALFORMED_CALL_RETRIES", "2")
    agent = _agent(tmp_path)
    # truncated JSON string arguments (provider couldn't finish the object)
    tc = ToolCall(id="c1", name="write",
                  arguments='{"path": "vfs:p/ch6.md", "content": "The archive was')
    results = await agent._handle_tool_calls([tc])
    assert len(results) == 1
    msg = results[0]["content"]
    assert "truncated or invalid JSON" in msg
    assert "append=true" in msg
    assert 'ch6.md' in msg  # best-effort path recovery for the message
    assert agent._malformed_retry_count == 1
    assert getattr(agent.provider, "_tool_choice_override", None) == "required"
    # nothing was written
    assert not list(tmp_path.rglob("ch6.md"))


@pytest.mark.asyncio
async def test_malformed_json_retries_off_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv("CLAW_MALFORMED_CALL_RETRIES", raising=False)
    agent = _agent(tmp_path)
    tc = ToolCall(id="c1", name="write",
                  arguments='{"path": "vfs:p/ch6.md", "content": "The archive was')
    results = await agent._handle_tool_calls([tc])
    # today's behaviour: malformed branch does NOT fire; the {"raw":…} call is
    # executed and errors normally.
    assert agent._malformed_retry_count == 0
    assert getattr(agent.provider, "_tool_choice_override", None) != "required"
    assert "truncated or invalid JSON" not in str(results)
    assert not list(tmp_path.rglob("ch6.md"))


@pytest.mark.asyncio
async def test_guard_refused_write_shows_guidance_and_arms_retry(tmp_path, monkeypatch):
    monkeypatch.setenv("CLAW_MALFORMED_CALL_RETRIES", "2")
    agent = _agent(tmp_path)
    marker = "[written to disk: a.md, 1 lines, 0.0KB — use read tool to view]"
    tc = ToolCall(id="c1", name="write", arguments={"path": "a.md", "content": marker})
    await agent._handle_tool_calls([tc])
    # the helpful guidance (not just "Error: placeholder_content_rejected") is what
    # the model actually reads — it lands in the tool SESSION message.
    tool_msgs = [m for m in agent.session.messages if m.get("role") == "tool"]
    assert tool_msgs, "expected a tool result message"
    guidance = str(tool_msgs[-1].get("content") or "")
    assert "acknowledgement" in guidance or "❌" in guidance
    assert agent._malformed_retry_count == 1
    assert getattr(agent.provider, "_tool_choice_override", None) == "required"


@pytest.mark.asyncio
async def test_placeholder_then_full_write_succeeds(tmp_path, monkeypatch):
    # The incident sequence: the FIRST write to a path is the placeholder
    # (refused, never added to the blind set), then the corrected full write
    # to the same path must go through.
    monkeypatch.setenv("CLAW_MALFORMED_CALL_RETRIES", "2")
    agent = _agent(tmp_path)
    marker = "[written to disk: story.md, 1 lines, 0.0KB — use read tool to view]"
    refused = ToolCall(id="c1", name="write", arguments={"path": "story.md", "content": marker})
    await agent._handle_tool_calls([refused])
    assert not list(tmp_path.rglob("story.md"))  # placeholder not persisted
    good = "# Chapter One\n\n" + "prose line\n" * 40
    fixed = ToolCall(id="c2", name="write", arguments={"path": "story.md", "content": good})
    results = await agent._handle_tool_calls([fixed])
    # the corrected re-issue is not blind-blocked and succeeds
    assert "BLIND REWRITE BLOCKED" not in str(results)
    assert results[0].get("success") is True


class _FakeResp:
    def __init__(self, finish_reason="stop", tool_calls=None, content=""):
        self.finish_reason = finish_reason
        self.tool_calls = tool_calls or []
        self.content = content


def test_length_truncation_corrective_fires_for_write(tmp_path, monkeypatch):
    monkeypatch.setenv("CLAW_MALFORMED_CALL_RETRIES", "2")
    agent = _agent(tmp_path)
    resp = _FakeResp(finish_reason="length",
                     tool_calls=[ToolCall(id="t1", name="write", arguments={"path": "a.md", "content": "cut"})])
    msg = agent._length_truncation_corrective(resp)
    assert msg is not None and "append=true" in msg
    assert agent._malformed_retry_count == 1
    assert getattr(agent.provider, "_tool_choice_override", None) == "required"


def test_length_truncation_corrective_ignores_normal_finish(tmp_path, monkeypatch):
    monkeypatch.setenv("CLAW_MALFORMED_CALL_RETRIES", "2")
    agent = _agent(tmp_path)
    # normal finish → no corrective
    resp = _FakeResp(finish_reason="stop",
                     tool_calls=[ToolCall(id="t1", name="write", arguments={"path": "a.md", "content": "ok"})])
    assert agent._length_truncation_corrective(resp) is None
    # length finish but no write/edit call → no corrective
    resp2 = _FakeResp(finish_reason="length",
                      tool_calls=[ToolCall(id="t2", name="read", arguments={"path": "a.md"})])
    assert agent._length_truncation_corrective(resp2) is None


def test_length_truncation_corrective_off_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv("CLAW_MALFORMED_CALL_RETRIES", raising=False)
    agent = _agent(tmp_path)
    resp = _FakeResp(finish_reason="length",
                     tool_calls=[ToolCall(id="t1", name="write", arguments={"path": "a.md", "content": "cut"})])
    assert agent._length_truncation_corrective(resp) is None
    assert agent._malformed_retry_count == 0


def test_length_truncation_corrective_bounded(tmp_path, monkeypatch):
    monkeypatch.setenv("CLAW_MALFORMED_CALL_RETRIES", "2")
    agent = _agent(tmp_path)
    resp = _FakeResp(finish_reason="length",
                     tool_calls=[ToolCall(id="t1", name="write", arguments={"path": "a.md", "content": "cut"})])
    assert agent._length_truncation_corrective(resp) is not None  # 1
    assert agent._length_truncation_corrective(resp) is not None  # 2
    assert agent._length_truncation_corrective(resp) is None      # budget spent
    assert agent._malformed_retry_count == 2
