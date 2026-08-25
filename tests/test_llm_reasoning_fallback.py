"""When a reasoning model returns empty `content` with a populated
`reasoning_content`, the LLM layer recovers a usable answer — preferring a JSON
object/array (many internal callers demand strict JSON) over the last prose
paragraph. This is what lets deepseek-style models pass FD's JSON-strict
quality checks (claim/consistency/coverage/contract, judges, routers).
"""

from __future__ import annotations

import json

import pytest

from captain_claw.llm import (
    LiteLLMProvider,
    Message,
    _REASONING_BACKFILL_PLACEHOLDER,
    _backfill_reasoning_content,
    _convert_messages_for_openai_style,
    _extract_json_blob,
    _is_reasoning_content_required_error,
    _recover_inline_reasoning,
    _reasoning_content_fallback,
)


def test_extract_json_blob_prefers_fenced():
    text = 'Let me think… here it is:\n```json\n{"verdict": "pass"}\n```\nDone.'
    assert json.loads(_extract_json_blob(text)) == {"verdict": "pass"}


def test_extract_json_blob_bare_trailing_object():
    text = 'Reasoning about the grants. Final answer: {"id": "c1", "verdict": "unclear"}'
    assert json.loads(_extract_json_blob(text)) == {"id": "c1", "verdict": "unclear"}


def test_extract_json_blob_trailing_array_wins_over_inline_example():
    text = ('For example the shape is {"x": 1}. After analysis the result is '
            '[{"id": "a", "ok": true}, {"id": "b", "ok": false}]')
    out = json.loads(_extract_json_blob(text))
    assert isinstance(out, list) and out[1]["id"] == "b"


def test_extract_json_blob_none_when_no_json():
    assert _extract_json_blob("just prose, no json here") is None


def test_fallback_recovers_json_from_reasoning():
    reasoning = ('I need to score each claim. Claim 1 is verifiable, claim 2 is '
                 'not. So:\n{"claims_checked": 2, "claims_confirmed": 1}')
    out = _reasoning_content_fallback(reasoning)
    assert json.loads(out) == {"claims_checked": 2, "claims_confirmed": 1}


def test_fallback_uses_last_paragraph_without_json():
    reasoning = "First I consider the scope.\n\nThe final conclusion is: proceed."
    assert _reasoning_content_fallback(reasoning) == "The final conclusion is: proceed."


def test_fallback_empty_reasoning():
    assert _reasoning_content_fallback("") == ""


# --- Inline <think> reasoning preserved for the thinking-mode round-trip ----
#
# Some servers (NVIDIA Nemotron via an OpenAI-compatible MLX/vLLM endpoint)
# stream the chain-of-thought INLINE as a <think>…</think> block in the content
# rather than as a separate reasoning_content field, yet require that reasoning
# echoed back as reasoning_content on the next turn — otherwise they 400 with
# "The reasoning_content in the thinking mode must be passed back to the API".
# We strip <think> from the user-visible content but must keep a copy so it can
# round-trip.


def test_recover_inline_reasoning_paired_block():
    text = "<think>rename each file to name + seconds</think>Here is the plan."
    assert _recover_inline_reasoning(text) == "rename each file to name + seconds"


def test_recover_inline_reasoning_orphan_closing_tag():
    # Chat template consumed the opening <think>; only the closer survives.
    text = "first I list the files, then rename</think>Done."
    assert _recover_inline_reasoning(text) == "first I list the files, then rename"


def test_recover_inline_reasoning_unterminated_block():
    text = "<think>the response was cut off mid-thought"
    assert _recover_inline_reasoning(text) == "the response was cut off mid-thought"


def test_recover_inline_reasoning_none_when_no_thinking():
    assert _recover_inline_reasoning("Just a plain answer, no reasoning.") == ""
    assert _recover_inline_reasoning("") == ""


def test_convert_messages_round_trips_reasoning_on_assistant():
    # The emit side must echo reasoning_content back on assistant messages so
    # the strict server accepts the follow-up request.
    msgs = [
        Message(role="user", content="rename the files"),
        Message(
            role="assistant",
            content="",
            reasoning_content="plan the rename by capture time",
            tool_calls=[{"id": "call_1", "type": "function",
                         "function": {"name": "write", "arguments": "{}"}}],
        ),
    ]
    out = _convert_messages_for_openai_style(msgs)
    assistant = out[-1]
    assert assistant["role"] == "assistant"
    assert assistant["reasoning_content"] == "plan the rename by capture time"


def test_convert_messages_omits_empty_reasoning():
    # Never emit a stray empty reasoning_content — some gateways reject it.
    msgs = [Message(role="assistant", content="hi", reasoning_content="")]
    assert "reasoning_content" not in _convert_messages_for_openai_style(msgs)[0]


def _chunk(**delta):
    fr = delta.pop("_finish", None)
    return {"choices": [{"delta": delta, "finish_reason": fr}]}


async def _fake_stream(chunks):
    for c in chunks:
        yield c


async def test_streaming_collect_preserves_inline_think_with_tool_call():
    # The exact failing shape: model streams <think>…</think> inline, then a
    # tool call. The collected message must carry the reasoning (so it can be
    # round-tripped) while the user-visible content stays clean.
    provider = LiteLLMProvider(provider="openai", model="nemotron", api_key="x")
    chunks = [
        _chunk(content="<think>read EXIF, then rename with seconds</think>"),
        _chunk(tool_calls=[{"index": 0, "id": "call_1",
                            "function": {"name": "write", "arguments": '{"path":"x"}'}}],
               _finish="tool_calls"),
        {"usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}},
    ]
    seen: list[str] = []
    collected = await provider._collect_streaming_response(
        _fake_stream(chunks), on_chunk=seen.append
    )
    message = collected["choices"][0]["message"]
    assert message["reasoning_content"] == "read EXIF, then rename with seconds"
    assert "<think>" not in message["content"]
    assert message["content"].strip() == ""
    assert message["tool_calls"][0]["function"]["name"] == "write"


async def test_streaming_collect_keeps_separate_reasoning_field():
    # When the server DOES send a separate reasoning_content delta, we use it
    # verbatim and don't double-count inline text.
    provider = LiteLLMProvider(provider="openai", model="nemotron", api_key="x")
    chunks = [
        _chunk(reasoning_content="separate field reasoning"),
        _chunk(content="The answer.", _finish="stop"),
    ]
    collected = await provider._collect_streaming_response(_fake_stream(chunks))
    message = collected["choices"][0]["message"]
    assert message["reasoning_content"] == "separate field reasoning"
    assert message["content"] == "The answer."


# --- Self-heal: reasoning_content required-on-round-trip 400 -----------------
#
# DeepSeek V4 thinking (via an OpenAI-compatible endpoint) 400s when an
# assistant message reaches the payload without reasoning_content — e.g. an
# orphan tool result normalized into an assistant turn, or a synthesized
# context/nudge message. _acompletion_tolerant backfills a placeholder on the
# retry so the turn survives.


def test_reasoning_required_error_matches_the_server_message():
    m = ("litellm.badrequesterror: openaiexception - the `reasoning_content` in "
         "the thinking mode must be passed back to the api.")
    assert _is_reasoning_content_required_error(m)


def test_reasoning_required_error_ignores_unrelated_400s():
    assert not _is_reasoning_content_required_error("invalid temperature value")
    assert not _is_reasoning_content_required_error("context length exceeded")
    # Merely mentioning the field without the round-trip demand is not a match.
    assert not _is_reasoning_content_required_error("reasoning_content was truncated")


def test_backfill_only_patches_reasoningless_assistant_messages():
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
        # assistant with a real reasoning — must be left untouched
        {"role": "assistant", "content": "", "reasoning_content": "real chain",
         "tool_calls": [{"id": "c1", "type": "function",
                         "function": {"name": "write", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "ok"},
        # orphan-tool-turned-assistant: no reasoning -> gets the placeholder
        {"role": "assistant", "content": "[tool_context:memory_semantic_select] …"},
    ]
    patched = _backfill_reasoning_content(messages)
    assert patched is True
    assert messages[2]["reasoning_content"] == "real chain"          # preserved
    assert messages[4]["reasoning_content"] == _REASONING_BACKFILL_PLACEHOLDER
    # Non-assistant roles are never given reasoning_content.
    assert "reasoning_content" not in messages[0]
    assert "reasoning_content" not in messages[1]
    assert "reasoning_content" not in messages[3]


def test_backfill_noop_when_all_assistants_have_reasoning():
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "answer", "reasoning_content": "why"},
    ]
    assert _backfill_reasoning_content(messages) is False


def test_backfill_treats_whitespace_reasoning_as_missing():
    messages = [{"role": "assistant", "content": "x", "reasoning_content": "   "}]
    assert _backfill_reasoning_content(messages) is True
    assert messages[0]["reasoning_content"] == _REASONING_BACKFILL_PLACEHOLDER
