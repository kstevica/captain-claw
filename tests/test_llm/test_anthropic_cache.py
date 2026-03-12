"""Tests for Anthropic prompt caching injection.

Validates that ``_inject_anthropic_cache_control`` correctly splits system
messages on the CACHE_SPLIT marker, adds cache_control breakpoints, and
handles conversation history cache breakpoints.
"""

import copy

import pytest

from captain_claw.llm import LiteLLMProvider, Message
from captain_claw.llm import _inject_anthropic_cache_control, _CACHE_SPLIT_MARKER


# ── Helpers ──────────────────────────────────────────────────────────────────

def _sys(content: str) -> dict:
    return {"role": "system", "content": content}


def _user(content: str) -> dict:
    return {"role": "user", "content": content}


def _assistant(content: str) -> dict:
    return {"role": "assistant", "content": content}


def _tool(content: str, tool_call_id: str = "call_1") -> dict:
    return {"role": "tool", "content": content, "tool_call_id": tool_call_id}


# ── System message split tests ──────────────────────────────────────────────

class TestSystemMessageSplit:
    """Test that the CACHE_SPLIT marker correctly splits system messages."""

    def test_splits_system_message_on_marker(self):
        """Static part gets cache_control; dynamic part does not."""
        static = "You are a helpful assistant.\n\nInstructions here."
        dynamic = "System info: 2026-03-12 14:30\nDisk: 50GB free"
        content = f"{static}\n\n{_CACHE_SPLIT_MARKER}\n{dynamic}"

        result = _inject_anthropic_cache_control([_sys(content)])

        assert len(result) == 1
        blocks = result[0]["content"]
        assert isinstance(blocks, list)
        assert len(blocks) == 2

        # First block: static, cached.
        assert blocks[0]["type"] == "text"
        assert blocks[0]["text"] == static
        assert blocks[0]["cache_control"] == {"type": "ephemeral"}

        # Second block: dynamic, NOT cached.
        assert blocks[1]["type"] == "text"
        assert blocks[1]["text"] == dynamic
        assert "cache_control" not in blocks[1]

    def test_no_marker_caches_whole_system_message(self):
        """Without marker, entire system message gets cache_control."""
        content = "You are a helpful assistant."
        result = _inject_anthropic_cache_control([_sys(content)])

        blocks = result[0]["content"]
        assert len(blocks) == 1
        assert blocks[0]["cache_control"] == {"type": "ephemeral"}
        assert blocks[0]["text"] == content

    def test_empty_dynamic_part_produces_single_block(self):
        """Marker at the very end with no dynamic content → single cached block."""
        static = "Static instructions"
        content = f"{static}\n\n{_CACHE_SPLIT_MARKER}\n"

        result = _inject_anthropic_cache_control([_sys(content)])

        blocks = result[0]["content"]
        # Dynamic part is empty after strip → only static block.
        assert len(blocks) == 1
        assert blocks[0]["text"] == static
        assert blocks[0]["cache_control"] == {"type": "ephemeral"}

    def test_list_content_adds_cache_control_to_last_block(self):
        """If system content is already a list, cache_control goes on last block."""
        blocks_in = [
            {"type": "text", "text": "block one"},
            {"type": "text", "text": "block two"},
        ]
        result = _inject_anthropic_cache_control([{"role": "system", "content": blocks_in}])

        blocks = result[0]["content"]
        assert len(blocks) == 2
        assert "cache_control" not in blocks[0]
        assert blocks[1]["cache_control"] == {"type": "ephemeral"}

    def test_empty_content_passes_through(self):
        """Empty system message content is not modified."""
        result = _inject_anthropic_cache_control([_sys("")])
        assert result[0]["content"] == ""


# ── Conversation history breakpoint tests ────────────────────────────────────

class TestHistoryBreakpoint:
    """Test that the last user/assistant message gets a cache breakpoint."""

    def test_last_user_message_gets_cache_control(self):
        """Last user message in the conversation gets cache_control."""
        msgs = [
            _sys("instructions"),
            _user("hello"),
            _assistant("hi there"),
            _user("what time is it?"),
        ]
        result = _inject_anthropic_cache_control(msgs)

        # Last message is the last user message.
        last = result[-1]
        assert last["role"] == "user"
        assert isinstance(last["content"], list)
        assert last["content"][0]["cache_control"] == {"type": "ephemeral"}

    def test_last_assistant_message_gets_cache_control(self):
        """If the last user/assistant is assistant, it gets cache_control."""
        msgs = [
            _sys("instructions"),
            _user("hello"),
            _assistant("response text"),
        ]
        result = _inject_anthropic_cache_control(msgs)

        last = result[-1]
        assert last["role"] == "assistant"
        assert isinstance(last["content"], list)
        assert last["content"][0]["cache_control"] == {"type": "ephemeral"}

    def test_tool_messages_skipped_for_breakpoint(self):
        """Tool messages are not candidates for cache breakpoint."""
        msgs = [
            _sys("instructions"),
            _user("hello"),
            _assistant("using tool"),
            _tool("tool result"),
        ]
        result = _inject_anthropic_cache_control(msgs)

        # Tool message should not get cache_control.
        tool_msg = result[-1]
        assert tool_msg["role"] == "tool"
        assert isinstance(tool_msg["content"], str)  # unchanged

        # The assistant message before it should get it.
        asst_msg = result[-2]
        assert asst_msg["role"] == "assistant"
        assert isinstance(asst_msg["content"], list)
        assert asst_msg["content"][0]["cache_control"] == {"type": "ephemeral"}

    def test_no_user_or_assistant_no_history_breakpoint(self):
        """If only system message, no history breakpoint is added."""
        msgs = [_sys("instructions")]
        result = _inject_anthropic_cache_control(msgs)

        # Only the system message, no crash.
        assert len(result) == 1
        assert result[0]["role"] == "system"


# ── No mutation tests ────────────────────────────────────────────────────────

class TestNoMutation:
    """Verify that input messages are not mutated."""

    def test_original_messages_unchanged(self):
        content = f"Static\n{_CACHE_SPLIT_MARKER}\nDynamic"
        original = [_sys(content), _user("hello")]
        snapshot = copy.deepcopy(original)

        _inject_anthropic_cache_control(original)

        assert original == snapshot, "Input messages should not be mutated"


# ── Non-system messages passthrough tests ────────────────────────────────────

class TestPassthrough:
    """Verify non-system messages pass through unchanged."""

    def test_user_message_not_modified_when_not_last(self):
        """User messages that aren't the last user/asst should not be modified."""
        msgs = [
            _sys("instructions"),
            _user("first message"),
            _assistant("response"),
            _user("last message"),
        ]
        result = _inject_anthropic_cache_control(msgs)

        # First user message should stay as string.
        assert isinstance(result[1]["content"], str)
        assert result[1]["content"] == "first message"


# ── Integration with _request_kwargs ─────────────────────────────────────────

class TestRequestKwargsIntegration:
    """Test that _request_kwargs correctly applies cache injection for Anthropic."""

    def test_anthropic_provider_applies_cache_control(self):
        provider = LiteLLMProvider(
            provider="anthropic",
            model="claude-haiku-4-5-20251001",
            temperature=0.7,
            max_tokens=4096,
        )
        static = "You are helpful."
        dynamic = "Time: 14:30"
        sys_content = f"{static}\n\n{_CACHE_SPLIT_MARKER}\n{dynamic}"
        kwargs = provider._request_kwargs(
            messages=[
                Message(role="system", content=sys_content),
                Message(role="user", content="hello"),
            ],
            stream=False,
        )

        sys_msg = kwargs["messages"][0]
        blocks = sys_msg["content"]
        assert isinstance(blocks, list)
        assert len(blocks) == 2
        assert blocks[0]["cache_control"] == {"type": "ephemeral"}
        assert "cache_control" not in blocks[1]

    def test_non_anthropic_provider_strips_marker(self):
        provider = LiteLLMProvider(
            provider="openai",
            model="gpt-4o-mini",
            temperature=0.7,
            max_tokens=4096,
        )
        sys_content = f"Instructions\n\n{_CACHE_SPLIT_MARKER}\nDynamic"
        kwargs = provider._request_kwargs(
            messages=[
                Message(role="system", content=sys_content),
                Message(role="user", content="hello"),
            ],
            stream=False,
        )

        sys_msg = kwargs["messages"][0]
        # Should be a plain string with marker removed.
        assert isinstance(sys_msg["content"], str)
        assert _CACHE_SPLIT_MARKER not in sys_msg["content"]
        assert "Instructions" in sys_msg["content"]
        assert "Dynamic" in sys_msg["content"]


# ── Static part stability test ───────────────────────────────────────────────

class TestStaticPartStability:
    """Verify that the static part is identical across calls with different dynamic content."""

    def test_static_part_byte_identical(self):
        """Static block text should be identical regardless of dynamic content."""
        static = "Large static instructions block with many tokens..."

        content1 = f"{static}\n\n{_CACHE_SPLIT_MARKER}\nTime: 14:30 | Disk: 50GB"
        content2 = f"{static}\n\n{_CACHE_SPLIT_MARKER}\nTime: 14:35 | Disk: 49GB"

        result1 = _inject_anthropic_cache_control([_sys(content1)])
        result2 = _inject_anthropic_cache_control([_sys(content2)])

        # Static blocks should be byte-identical.
        assert result1[0]["content"][0]["text"] == result2[0]["content"][0]["text"]
        # Dynamic blocks should differ.
        assert result1[0]["content"][1]["text"] != result2[0]["content"][1]["text"]
