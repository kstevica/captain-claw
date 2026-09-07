"""A thinking/reasoning model served over an OpenAI-compatible endpoint 400s
when the orchestration loop forces ``tool_choice="required"`` on a stall retry:
"Thinking mode does not support this tool_choice". On the streaming paths that
400 surfaces *during stream iteration* — not on the ``acompletion()`` call — so
``_acompletion_tolerant``'s own retry never sees it and the turn crashed with
"openai streaming callback failed". The streaming methods now catch that 400,
drop the forced constraint, flag the provider, and retry once.
"""

from __future__ import annotations

import pytest

import captain_claw.llm as llm_mod
from captain_claw.llm import (
    LiteLLMProvider,
    Message,
    ToolDefinition,
    _is_tool_choice_unsupported_error,
)


# --- Classifier -------------------------------------------------------------


def test_tool_choice_unsupported_matches_the_server_message():
    m = ("litellm.badrequesterror: openaiexception - thinking mode does not "
         "support this tool_choice")
    assert _is_tool_choice_unsupported_error(m)


def test_tool_choice_unsupported_matches_not_allowed_phrasing():
    assert _is_tool_choice_unsupported_error(
        "tool_choice is not allowed when reasoning is enabled"
    )


def test_tool_choice_unsupported_ignores_unrelated_400s():
    assert not _is_tool_choice_unsupported_error("invalid temperature value")
    assert not _is_tool_choice_unsupported_error("context length exceeded")
    # A support-related 400 that isn't about tool_choice must not match.
    assert not _is_tool_choice_unsupported_error(
        "this model does not support image inputs"
    )


# --- Streaming self-heal ----------------------------------------------------


class _FakeBadRequest(Exception):
    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


_TOOL_CHOICE_400 = _FakeBadRequest(
    "litellm.BadRequestError: OpenAIException - Thinking mode does not support "
    "this tool_choice"
)


async def _raising_stream(exc: Exception):
    """Async iterator that raises on first consumption — a pre-flight 400 that
    surfaces only when the stream is read (nothing streamed yet)."""
    if False:  # make this an async generator
        yield  # pragma: no cover
    raise exc


async def _content_stream():
    yield {"choices": [{"delta": {"content": "hello"}, "finish_reason": None}]}
    yield {
        "choices": [{"delta": {}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
    }


def _forcing_provider() -> LiteLLMProvider:
    provider = LiteLLMProvider(provider="openai", model="deepseek-thinker", api_key="x")
    # The orchestration loop's strongest lever: force a tool call this turn.
    provider._tool_choice_override = "required"
    return provider


_TOOLS = [ToolDefinition(name="write", description="write a file", parameters={})]


async def test_complete_with_callback_self_heals_tool_choice(monkeypatch):
    provider = _forcing_provider()
    calls = {"n": 0}

    async def fake_tolerant(kwargs, prov=None):
        calls["n"] += 1
        if calls["n"] == 1:
            # First attempt carries the forced constraint...
            assert kwargs.get("tool_choice") == "required"
            return _raising_stream(_TOOL_CHOICE_400)
        # ...retry drops it.
        assert "tool_choice" not in kwargs
        return _content_stream()

    monkeypatch.setattr(llm_mod, "_acompletion_tolerant", fake_tolerant)

    seen: list[str] = []
    resp = await provider.complete_with_callback(
        [Message(role="user", content="hi")], tools=_TOOLS, on_chunk=seen.append
    )

    assert calls["n"] == 2
    assert provider._tool_choice_unsupported is True
    assert resp.content == "hello"
    # No content was streamed on the failed attempt, so the callback fires once.
    assert seen == ["hello"]


async def test_complete_with_callback_does_not_retry_unrelated_stream_error(monkeypatch):
    # An unrelated mid-stream error is NOT a tool_choice rejection, so the
    # existing "preserve partial content" behavior stands: no retry, no
    # provider flag flipped, and the collector returns what it had (empty here).
    provider = _forcing_provider()
    calls = {"n": 0}

    async def fake_tolerant(kwargs, prov=None):
        calls["n"] += 1
        return _raising_stream(_FakeBadRequest("context length exceeded"))

    monkeypatch.setattr(llm_mod, "_acompletion_tolerant", fake_tolerant)

    resp = await provider.complete_with_callback(
        [Message(role="user", content="hi")], tools=_TOOLS, on_chunk=lambda _c: None
    )
    assert calls["n"] == 1
    assert resp.content == ""
    assert getattr(provider, "_tool_choice_unsupported", False) is False


async def test_complete_streaming_self_heals_tool_choice(monkeypatch):
    provider = _forcing_provider()
    calls = {"n": 0}

    async def fake_tolerant(kwargs, prov=None):
        calls["n"] += 1
        if calls["n"] == 1:
            assert kwargs.get("tool_choice") == "required"
            return _raising_stream(_TOOL_CHOICE_400)
        assert "tool_choice" not in kwargs
        return _content_stream()

    monkeypatch.setattr(llm_mod, "_acompletion_tolerant", fake_tolerant)

    chunks: list[str] = []
    async for piece in provider.complete_streaming(
        [Message(role="user", content="hi")], tools=_TOOLS
    ):
        chunks.append(piece)

    assert calls["n"] == 2
    assert provider._tool_choice_unsupported is True
    assert "".join(chunks) == "hello"
