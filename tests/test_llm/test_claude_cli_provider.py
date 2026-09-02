"""Tests for :class:`ClaudeCLIProvider` — the Anthropic *subscription*
(Pro/Max) provider that shells out to the ``claude`` CLI.

All CLI interaction is mocked; these tests never spawn ``claude`` and never
touch the network or the user's subscription.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from captain_claw.exceptions import LLMAPIError
from captain_claw.llm import (
    ClaudeCLIProvider,
    LiteLLMProvider,
    Message,
    _normalize_provider_name,
    create_provider,
)


# ── factory / normalization ──────────────────────────────────────────────
@pytest.mark.parametrize(
    "alias",
    [
        "claude-cli",
        "claude_cli",
        "claude-code",
        "claude-subscription",
        "claude-sub",
        "claude-max",
        "anthropic-cli",
        "anthropic-subscription",
    ],
)
def test_aliases_normalize_to_claude_cli(alias):
    assert _normalize_provider_name(alias) == "claude-cli"


def test_plain_claude_and_anthropic_still_route_to_metered_api():
    # Guard against the subscription provider ever silently shadowing the
    # pay-per-token path.
    assert _normalize_provider_name("claude") == "anthropic"
    assert _normalize_provider_name("anthropic") == "anthropic"


def test_factory_returns_claude_cli_provider():
    prov = create_provider(provider="claude-subscription", model="claude-opus-4-8")
    assert isinstance(prov, ClaudeCLIProvider)
    assert prov.supports_tools is False


def test_factory_metered_path_unchanged():
    prov = create_provider(provider="anthropic", model="claude-opus-4-8", api_key="k")
    assert isinstance(prov, LiteLLMProvider)


# ── pure helpers ─────────────────────────────────────────────────────────
def test_map_model():
    assert ClaudeCLIProvider._map_model("anthropic/claude-opus-4-8") == "claude-opus-4-8"
    assert ClaudeCLIProvider._map_model("claude") == "sonnet"  # bare family word
    assert ClaudeCLIProvider._map_model("opus") == "opus"      # passthrough
    assert ClaudeCLIProvider._map_model("") == "sonnet"


def test_child_env_scrubs_api_key_and_injects_token(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-should-be-scrubbed")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://proxy.local")
    prov = ClaudeCLIProvider(model="sonnet", oauth_token="sk-ant-oat01-xyz")
    env = prov._child_env()
    assert "ANTHROPIC_API_KEY" not in env       # the mis-billing guard
    assert "ANTHROPIC_BASE_URL" not in env
    assert env["CLAUDE_CODE_OAUTH_TOKEN"] == "sk-ant-oat01-xyz"


def test_split_messages_single_and_multi_turn():
    sysp, body = ClaudeCLIProvider._split_messages(
        [Message(role="system", content="You are terse."), Message(role="user", content="Hi")]
    )
    assert sysp == "You are terse."
    assert body == "Hi"

    sysp2, body2 = ClaudeCLIProvider._split_messages(
        [
            Message(role="system", content="S1"),
            Message(role="user", content="Q1"),
            Message(role="assistant", content="A1"),
            Message(role="user", content="Q2"),
        ]
    )
    assert sysp2 == "S1"
    assert body2 == "User: Q1\n\nAssistant: A1\n\nUser: Q2"


def test_parse_result_json_single_and_multiline():
    d = ClaudeCLIProvider._parse_result_json(
        '{"type":"result","is_error":false,"result":"hi"}'
    )
    assert d["result"] == "hi"
    d2 = ClaudeCLIProvider._parse_result_json(
        '{"type":"system"}\n{"type":"result","result":"x"}\n'
    )
    assert d2["result"] == "x"

    with pytest.raises(LLMAPIError):
        ClaudeCLIProvider._parse_result_json("")


def test_auth_hint_points_to_setup_token():
    prov = ClaudeCLIProvider(model="sonnet")
    hint = prov._auth_hint("Failed to authenticate: OAuth session expired")
    assert "setup-token" in hint and "ANTHROPIC_API_KEY" in hint


def test_build_argv_streaming_requires_verbose():
    prov = ClaudeCLIProvider(model="sonnet")
    stream = prov._build_argv("sys", None, streaming=True)
    assert "--verbose" in stream
    assert stream[stream.index("--output-format") + 1] == "stream-json"
    plain = prov._build_argv("sys", None, streaming=False)
    assert "--verbose" not in plain
    assert plain[plain.index("--output-format") + 1] == "json"
    assert "--system-prompt" in plain


# ── complete() with a mocked subprocess ──────────────────────────────────
async def test_complete_success_parses_usage(monkeypatch):
    prov = ClaudeCLIProvider(model="sonnet")
    payload = json.dumps(
        {
            "type": "result",
            "is_error": False,
            "subtype": "success",
            "result": "The answer is 42.",
            "model": "claude-sonnet-4-5",
            "stop_reason": "end_turn",
            "total_cost_usd": 0.012,
            "usage": {
                "input_tokens": 120,
                "cache_read_input_tokens": 30,
                "cache_creation_input_tokens": 0,
                "output_tokens": 8,
            },
        }
    )

    async def fake_spawn(argv, prompt):
        return payload

    monkeypatch.setattr(prov, "_spawn_and_read", fake_spawn)
    resp = await prov.complete([Message(role="user", content="q")], tools=None)
    assert resp.content == "The answer is 42."
    assert resp.usage["prompt_tokens"] == 150   # input + cache_read + cache_creation
    assert resp.usage["completion_tokens"] == 8
    assert resp.usage["total_tokens"] == 158
    assert resp.finish_reason == "end_turn"
    assert resp.tool_calls == []


async def test_complete_auth_error_raises_with_hint(monkeypatch):
    prov = ClaudeCLIProvider(model="sonnet")

    async def fake_spawn(argv, prompt):
        return json.dumps(
            {"type": "result", "is_error": True,
             "result": "Failed to authenticate: OAuth session expired"}
        )

    monkeypatch.setattr(prov, "_spawn_and_read", fake_spawn)
    with pytest.raises(LLMAPIError) as exc:
        await prov.complete([Message(role="user", content="q")])
    assert "setup-token" in str(exc.value)


# ── streaming with a mocked subprocess ───────────────────────────────────
class _FakeStdout:
    def __init__(self, lines):
        self._lines = [l.encode() for l in lines]

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._lines:
            raise StopAsyncIteration
        return self._lines.pop(0)


class _FakeStdin:
    def write(self, b):  # noqa: D401
        pass

    async def drain(self):
        pass

    def close(self):
        pass


class _FakeProc:
    def __init__(self, lines):
        self.stdin = _FakeStdin()
        self.stdout = _FakeStdout(lines)
        self.returncode = 0

    async def wait(self):
        return 0

    def kill(self):
        pass


def _patch_exec(monkeypatch, lines):
    async def _factory(*a, **k):
        return _FakeProc(lines)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _factory)


async def _collect(prov):
    out = []
    async for chunk in prov.complete_streaming([Message(role="user", content="hi")]):
        out.append(chunk)
    return "".join(out)


async def test_streaming_partial_deltas(monkeypatch):
    prov = ClaudeCLIProvider(model="sonnet")
    lines = [
        json.dumps({"type": "system", "subtype": "init"}) + "\n",
        json.dumps({"type": "stream_event", "event": {"type": "content_block_delta",
                    "delta": {"type": "text_delta", "text": "Hel"}}}) + "\n",
        json.dumps({"type": "stream_event", "event": {"type": "content_block_delta",
                    "delta": {"type": "text_delta", "text": "lo!"}}}) + "\n",
        # aggregate assistant message must NOT be re-emitted after deltas
        json.dumps({"type": "assistant", "message": {"model": "claude-sonnet-4-5",
                    "content": [{"type": "text", "text": "Hello!"}]}}) + "\n",
        json.dumps({"type": "result", "is_error": False, "result": "Hello!"}) + "\n",
    ]
    _patch_exec(monkeypatch, lines)
    assert await _collect(prov) == "Hello!"


async def test_streaming_whole_block_fallback(monkeypatch):
    prov = ClaudeCLIProvider(model="sonnet")
    lines = [
        json.dumps({"type": "assistant", "message": {"model": "claude-sonnet-4-5",
                    "content": [{"type": "text", "text": "Whole answer."}]}}) + "\n",
        json.dumps({"type": "result", "is_error": False, "result": "Whole answer."}) + "\n",
    ]
    _patch_exec(monkeypatch, lines)
    assert await _collect(prov) == "Whole answer."


async def test_streaming_synthetic_auth_error_raises(monkeypatch):
    prov = ClaudeCLIProvider(model="sonnet")
    lines = [
        json.dumps({"type": "assistant", "error": "authentication_failed",
                    "message": {"model": "<synthetic>",
                                "content": [{"type": "text",
                                             "text": "Failed to authenticate: OAuth session expired"}]}}) + "\n",
    ]
    _patch_exec(monkeypatch, lines)
    with pytest.raises(LLMAPIError) as exc:
        await _collect(prov)
    assert "setup-token" in str(exc.value)
