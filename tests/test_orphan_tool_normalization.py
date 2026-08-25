"""An orphan tool message (its parent assistant tool_call dropped from the
retained context, or the chain interrupted) must be retained as a USER context
message, never as a reasoning-less ASSISTANT turn.

Minting an assistant turn here trips thinking-mode servers that require
reasoning_content on every assistant message — DeepSeek V4 thinking via an
OpenAI-compatible endpoint 400s with "The reasoning_content in the thinking
mode must be passed back to the API". This is the exact path memory_semantic_
select telemetry and post-write nudges exercised.
"""

from __future__ import annotations

from captain_claw.agent_context_mixin import AgentContextMixin


class _StubAgent(AgentContextMixin):
    """Minimal carrier for the normalizer under test."""

    def __init__(self, provider: str = "openai"):
        self._provider = provider

    def get_runtime_model_details(self) -> dict:
        return {"provider": self._provider, "model": "deepseek-v4-pro"}


def test_orphan_tool_message_becomes_user_not_assistant():
    agent = _StubAgent(provider="openai")
    # A tool telemetry message with no matching pending assistant tool_call —
    # exactly what memory_semantic_select emits.
    selected = [
        {"role": "user", "content": "research this"},
        {"role": "tool", "tool_name": "memory_semantic_select",
         "content": "selection_mode=term_overlap ...", "tool_call_id": ""},
    ]
    out = agent._normalize_selected_messages_for_provider(selected)
    orphan = out[-1]
    assert orphan["role"] == "user"
    assert orphan["content"].startswith("[tool_context:memory_semantic_select]")
    assert "reasoning_content" not in orphan
    # No assistant message was minted at all.
    assert all(m["role"] != "assistant" for m in out)


def test_matched_tool_chain_is_preserved_as_tool():
    agent = _StubAgent(provider="openai")
    selected = [
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": "", "reasoning_content": "think",
         "tool_calls": [{"id": "c1", "type": "function",
                         "function": {"name": "write", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "tool_name": "write", "content": "ok"},
    ]
    out = agent._normalize_selected_messages_for_provider(selected)
    roles = [m["role"] for m in out]
    assert roles == ["user", "assistant", "tool"]
    # The real assistant tool-call turn keeps its reasoning for the round-trip.
    assert out[1]["reasoning_content"] == "think"


def test_non_strict_provider_left_untouched():
    agent = _StubAgent(provider="gemini")  # not in the strict-order set
    selected = [
        {"role": "tool", "tool_name": "memory_semantic_select", "content": "x",
         "tool_call_id": ""},
    ]
    out = agent._normalize_selected_messages_for_provider(selected)
    assert out is selected  # returned as-is, no normalization
