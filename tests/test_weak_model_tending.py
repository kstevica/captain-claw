"""Guards that keep a weak local model from failing silently.

Three failure modes, all observed live on an ollama-hosted 9B running the
Deep Researcher archetype:

1. A frontier-sized ``num_ctx`` inherited from the spawn tier blows up the
   local KV cache, throughput collapses, and generations come back empty.
2. The model announces its next step instead of taking it — but prefixes a
   short observation, so the stall detector's anchored regex misses it.
3. The model returns zero-length content, which sails through every
   completion gate and is reported to the user as a successful turn.

The floor for (3) has to be BOUNDED, not budget-bounded: an unbounded
re-roll turns a three-call failure into an eighty-call one and hangs the
turn. That regression is covered here too.

Every guard here must be inert for hosted/frontier models.
"""

from __future__ import annotations

import json

import pytest

from captain_claw.agent_orchestration_mixin import (
    _closing_sentence_stalls,
    _looks_like_stall,
)
from captain_claw.llm import (
    _OLLAMA_DEFAULT_MAX_NUM_CTX,
    Message,
    OllamaProvider,
    _clamp_ollama_num_ctx,
    _forced_tool_call_schema,
)


class _FakeResponse:
    is_success = True
    status_code = 200
    text = ""

    def __init__(self, payload: dict):
        self._payload = payload

    def json(self) -> dict:
        return self._payload


class _CapturingClient:
    """Stands in for the provider's httpx client; records the request body."""

    def __init__(self, payload: dict):
        self._payload = payload
        self.last_body: dict | None = None

    async def post(self, url, json=None, headers=None):  # noqa: A002
        self.last_body = json
        return _FakeResponse(self._payload)


def _ollama_provider(payload: dict) -> tuple[OllamaProvider, _CapturingClient]:
    provider = OllamaProvider(model="ornith:9b", num_ctx=8192)
    client = _CapturingClient(payload)
    provider.client = client  # type: ignore[assignment]
    return provider, client


TOOLS = [
    {"name": "web_search", "description": "search", "parameters": {}},
    {"name": "read", "description": "read", "parameters": {}},
]


# ── num_ctx clamp ───────────────────────────────────────────────────


def test_frontier_num_ctx_is_clamped_for_local_models():
    """A `reason`-tier 200k budget must not reach a local 9B."""
    assert _clamp_ollama_num_ctx(200_000, "ornith:9b") == _OLLAMA_DEFAULT_MAX_NUM_CTX


def test_num_ctx_under_the_ceiling_is_untouched():
    assert _clamp_ollama_num_ctx(8192, "qwen3.5:4b") == 8192
    assert _clamp_ollama_num_ctx(_OLLAMA_DEFAULT_MAX_NUM_CTX, "gemma4:e4b") == (
        _OLLAMA_DEFAULT_MAX_NUM_CTX
    )


def test_cloud_models_are_exempt():
    """`:cloud` runs on Ollama's hardware — no local memory pressure."""
    assert _clamp_ollama_num_ctx(200_000, "minimax-m2.7:cloud") == 200_000


def test_ceiling_is_overridable(monkeypatch):
    monkeypatch.setenv("CLAW_OLLAMA_MAX_NUM_CTX", "131072")
    assert _clamp_ollama_num_ctx(200_000, "ornith:9b") == 131_072


def test_garbage_override_falls_back_to_the_default(monkeypatch):
    monkeypatch.setenv("CLAW_OLLAMA_MAX_NUM_CTX", "not-a-number")
    assert _clamp_ollama_num_ctx(200_000, "ornith:9b") == _OLLAMA_DEFAULT_MAX_NUM_CTX


# ── forced tool call schema ─────────────────────────────────────────


def test_forced_schema_pins_the_tool_name_to_an_enum():
    schema = _forced_tool_call_schema([{"name": "web_search"}, {"name": "read"}])
    assert schema["properties"]["tool"]["enum"] == ["web_search", "read"]
    assert schema["required"] == ["tool", "arguments"]
    # Flat object, not a oneOf union — grammar engines and small models
    # both handle the flat+enum shape far more reliably.
    assert schema["type"] == "object"
    assert "oneOf" not in schema


def test_forced_schema_is_none_without_tools():
    """No tools → nothing to force; caller must fall back to a plain call."""
    assert _forced_tool_call_schema([]) is None
    assert _forced_tool_call_schema([{"name": "   "}]) is None


# ── stall detection ─────────────────────────────────────────────────


@pytest.mark.parametrize(
    "text",
    [
        "",
        "   ",
        "Let me look that up.",
        "I'll fetch the file now.",
        # The shape that escaped the anchored regex live: an observation
        # about what just happened, then the announcement, then silence.
        "Great data from GitHub. Let me grab a few more key sources for depth.",
        "Found the config. Let me check the other one.",
        "Okay, that worked. I'll run the tests now.",
    ],
)
def test_stalls_are_detected(text):
    assert _looks_like_stall(text) is True


@pytest.mark.parametrize(
    "text",
    [
        # Handing control back to the user is how a FINISHED turn signs
        # off. Re-rolling these would discard completed work.
        "Let me know if you need anything else.",
        "Done — the report is saved to report.md. Let me know if you want more detail.",
        "I'll be here if you want to dig further.",
        # "Let me explain…" is an answer, not an announcement.
        (
            "Let me explain why this happens. The KV cache is allocated up front, "
            "so a 200k window on a 9B model needs tens of gigabytes, which spills "
            "to swap and collapses throughput."
        ),
        "The answer is 42.",
        "I fixed the bug in agent.py:120 by clamping num_ctx. Tests pass.",
        "The build failed because the lockfile is stale. Run npm install, then retry.",
    ],
)
def test_real_answers_are_not_stalls(text):
    assert _looks_like_stall(text) is False


def test_long_messages_ending_on_an_intent_phrase_are_not_stalls():
    """The closing-sentence check is length-bounded on purpose.

    A substantial answer that happens to end on "I'll …" is delivering
    work, not deferring it.
    """
    text = (
        "The clamp lands in three places: the provider constructor, the context "
        "budget, and the spawn-time tier resolution. Each is independently safe "
        "for hosted models because none of them expose a num_ctx attribute at "
        "all. I'll note the remaining gap in the plan doc."
    )
    assert len(text) > 160
    assert _looks_like_stall(text) is False


def test_closing_sentence_check_needs_two_sentences():
    """A one-sentence stall is the anchored check's job, not this one."""
    assert _closing_sentence_stalls("Let me look that up.") is False


# ── forced tool call, end to end ────────────────────────────────────


@pytest.mark.asyncio
async def test_forced_tool_call_round_trip():
    """`tool_choice=required` → grammar → a real ToolCall, no JSON leak."""
    provider, client = _ollama_provider({
        "message": {
            "role": "assistant",
            "content": json.dumps({
                "tool": "web_search",
                "arguments": {"query": "Stevica Kuharski"},
            }),
        },
        "model": "ornith:9b",
    })
    provider._tool_choice_override = "required"

    resp = await provider.complete([Message(role="user", content="go")], tools=TOOLS)

    # The grammar was applied and thinking was switched off.
    assert client.last_body["format"]["properties"]["tool"]["enum"] == [
        "web_search",
        "read",
    ]
    assert "think" not in client.last_body
    # Tool definitions stay in the body — the model still needs the
    # argument schemas to fill `arguments`.
    assert client.last_body["tools"]

    # The JSON became a real tool call and did NOT leak into the reply.
    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0].name == "web_search"
    assert resp.tool_calls[0].arguments == {"query": "Stevica Kuharski"}
    assert resp.content == ""


@pytest.mark.asyncio
async def test_override_is_consumed_once():
    """Same contract as the LiteLLM/ChatGPT providers: one shot, then reset."""
    provider, client = _ollama_provider({
        "message": {"role": "assistant", "content": "{\"tool\": \"read\", \"arguments\": {}}"},
        "model": "ornith:9b",
    })
    provider._tool_choice_override = "required"

    await provider.complete([Message(role="user", content="go")], tools=TOOLS)
    assert provider._tool_choice_override is None

    await provider.complete([Message(role="user", content="again")], tools=TOOLS)
    assert "format" not in client.last_body


@pytest.mark.asyncio
async def test_no_override_leaves_the_request_untouched():
    """The off-path request must be exactly what it was before."""
    provider, client = _ollama_provider({
        "message": {"role": "assistant", "content": "hello"},
        "model": "ornith:9b",
    })
    resp = await provider.complete([Message(role="user", content="hi")], tools=TOOLS)
    assert "format" not in client.last_body
    assert resp.content == "hello"
    assert resp.tool_calls == []


# ── empty-answer floor is bounded ───────────────────────────────────


@pytest.mark.asyncio
async def test_empty_answer_floor_gives_up_after_two_rerolls():
    """A persistently-empty model must fail in 3 calls, not 80.

    Regression: the floor originally retried while the *iteration budget*
    allowed, so a model that returns empty every time re-rolled all the way
    to the hard ceiling — turning a fast honest failure into a hang.
    """
    from captain_claw.agent_completion_mixin import (
        MAX_EMPTY_ANSWER_RETRIES,
        AgentCompletionMixin,
    )
    from captain_claw.agent_stuck import MSG_EMPTY_RESPONSE

    class _Stub(AgentCompletionMixin):
        _scale_progress = None
        session = None

        def __init__(self):
            self._empty_answer_retries = 0

        async def _maybe_auto_write_requested_output(self, **kw):
            return ""

        def _collect_turn_tool_output(self, _idx):
            return ""

        def _active_task_tool_policy_payload(self, _p):
            return None

        def _turn_has_unexecuted_script(self, _idx):
            return False, ""

        def _emit_tool_output(self, *a, **kw):
            pass

        def __getattr__(self, name):
            # Every other gate this function consults is a `_turn_has_*` /
            # `_verify_*` predicate. Default them all to "nothing happened"
            # so the empty-answer floor is the only gate under test.
            if name.startswith(("_turn_", "_verify_", "_scale_", "_maybe_")):
                return lambda *a, **kw: False
            raise AttributeError(name)

    stub = _Stub()

    async def _finalize():
        return await stub._attempt_finalize_response(
            output_text="",
            iteration=0,
            hard_turn_iterations=80,
            finish_success=True,
            effective_user_input="go",
            user_input="go",
            turn_start_idx=0,
            turn_usage={},
            session_tool_policy=None,
            planning_pipeline=None,
            list_task_plan={},
            task_contract=None,
            completion_requirements=[],
            completion_feedback="",
            enforce_python_worker_mode=False,
            python_worker_attempted=False,
        )

    # The permitted re-rolls block finalization and ask for a real answer.
    for _ in range(MAX_EMPTY_ANSWER_RETRIES):
        finalized, text, _success, feedback, _pw = await _finalize()
        assert finalized is False
        assert text == ""
        assert "completely empty" in feedback

    # The next one gives up honestly instead of re-rolling to the ceiling.
    finalized, text, success, _feedback, _pw = await _finalize()
    assert finalized is True
    assert text == MSG_EMPTY_RESPONSE
    assert success is False


@pytest.mark.asyncio
async def test_forced_call_that_does_not_parse_is_not_fabricated():
    """A model that ignores the grammar must not yield a bogus tool call."""
    provider, _ = _ollama_provider({
        "message": {"role": "assistant", "content": "I cannot do that"},
        "model": "ornith:9b",
    })
    provider._tool_choice_override = "required"
    resp = await provider.complete([Message(role="user", content="go")], tools=TOOLS)
    assert resp.tool_calls == []
    assert resp.content == "I cannot do that"
