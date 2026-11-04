"""When a turn runs out of steps, Jev judges whether the agent is still
making progress — and a working agent gets more steps.

The check (``captain_claw/jev_progress.py``) asks Jev one yes/no question
about the turn's steps; the main loop (``agent_orchestration_mixin``) asks it
only when its own rule would stop the turn, and extends the budget on a
confident yes (never past the hard ceiling). Never in the way: no key, an
error or a timeout means the usual stop.
"""

from __future__ import annotations

import asyncio
import types

import pytest

from captain_claw import jev_client as jc
from captain_claw import jev_progress as jp
from captain_claw.agent import Agent
from captain_claw.config import get_config
from captain_claw.llm import LLMProvider, LLMResponse, Message, ToolCall, ToolDefinition
from captain_claw.session import Session
from captain_claw.tools.registry import Tool, ToolRegistry, ToolResult


def _answer(p: float) -> dict:
    return {"answers": {jp.QUESTION_NAME: {"type": "noul", "noul": p}}, "model": "jev-latest"}


@pytest.fixture
def jev(monkeypatch):
    """Standalone with a key; JevAPI.ask scripted (the last answer repeats)."""
    from captain_claw.google_oauth_manager import GoogleOAuthManager

    monkeypatch.setattr(GoogleOAuthManager, "_flight_deck_base", staticmethod(lambda: ""))
    monkeypatch.setattr(GoogleOAuthManager, "_member_call", staticmethod(lambda: False))
    monkeypatch.setenv("JEV_API_KEY", "jev-test")
    monkeypatch.setattr(get_config().tools.jev, "api_key", "")
    monkeypatch.setattr(get_config().tools.jev, "check_progress", True)
    state = types.SimpleNamespace(calls=[], answers=[_answer(0.9)])

    async def ask(self, st, questions, model=None):
        state.calls.append((st, questions))
        jc.build_request(st, questions, model)  # what would leave must validate
        ans = state.answers[0] if len(state.answers) == 1 else state.answers.pop(0)
        if isinstance(ans, BaseException):
            raise ans
        return ans

    monkeypatch.setattr(jc.JevAPI, "ask", ask)
    return state


TURN = [
    {"role": "user", "content": "find the open tenders over 1M"},
    {"role": "assistant", "content": "Let me read the grid's JS module."},
    {"role": "tool", "tool_name": "shell", "tool_arguments": {"command": "curl grid.js"},
     "content": "DevExtreme DataGrid ... pageSize=20"},
    {"role": "tool", "tool_name": "completion_gate", "tool_arguments": {}, "content": "step=x"},
    {"role": "assistant", "content": "made-up claim", "origin": "rejected"},
    {"role": "tool", "tool_name": "web_fetch", "tool_arguments": {"url": "https://x.test/api"},
     "content": "Error: 403 Forbidden"},
    {"role": "user", "content": "[SYSTEM: keep going]"},
]


# ── The question and the state ─────────────────────────────────────────


def test_the_question_is_a_valid_jev_noul():
    q = jc.normalize_questions(jp.QUESTIONS)[jp.QUESTION_NAME]
    assert q["type"] == "noul" and set(q["criteria"]) == {"true", "false"}


def test_state_lists_the_steps_and_leaves_out_noise():
    st = jp.build_state("find the open tenders over 1M", TURN, steps_used=40, step_budget=40)
    assert st["request"] == "find the open tenders over 1M"
    assert st["budget"] == "40 of 40 steps used"
    steps = st["recent_steps"].splitlines()
    assert steps == [
        "agent: Let me read the grid's JS module.",
        'shell({"command": "curl grid.js"}) → ok: DevExtreme DataGrid ... pageSize=20',
        'web_fetch({"url": "https://x.test/api"}) → FAILED: Error: 403 Forbidden',
        "[note to the agent: [SYSTEM: keep going]]",
    ]
    # The request opens the turn; it is not repeated as a step.
    assert "find the open tenders" not in st["recent_steps"]
    jc.build_request(st, jp.QUESTIONS)


def test_state_keeps_the_most_recent_steps_and_drops_the_surface_block():
    block = ("[SYSTEM CONTEXT — do not echo, quote, or acknowledge this block in your reply.]\n"
             "rules\n---\nUSER MESSAGE:\n")
    turn = [{"role": "user", "content": "go"}] + [
        {"role": "tool", "tool_name": "web_fetch", "tool_arguments": {"url": f"https://x.test/{i}"},
         "content": "z" * 400}
        for i in range(100)
    ]
    st = jp.build_state(block + "go", turn, steps_used=3, step_budget=3)
    assert st["request"] == "go"
    steps = st["recent_steps"]
    assert steps.startswith("[… ") and "earlier steps cut" in steps
    assert steps.endswith("…") and "https://x.test/99" in steps and "https://x.test/0\"" not in steps
    assert len(steps) < 9200
    assert jp.build_state("go", [], steps_used=1, step_budget=1)["recent_steps"] == "(no steps recorded)"


# ── progress_probability ───────────────────────────────────────────────


async def test_returns_jevs_probability(jev):
    assert await jp.progress_probability("go", TURN, steps_used=4, step_budget=4) == 0.9
    st, questions = jev.calls[0]
    assert st["budget"] == "4 of 4 steps used" and jp.QUESTION_NAME in questions


@pytest.mark.parametrize("why", ["off", "no_key", "member"])
async def test_no_call_when_it_cannot_run(jev, monkeypatch, why):
    if why == "off":
        monkeypatch.setattr(get_config().tools.jev, "check_progress", False)
    elif why == "no_key":
        monkeypatch.delenv("JEV_API_KEY")
        monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    else:
        from captain_claw.google_oauth_manager import GoogleOAuthManager

        monkeypatch.setattr(GoogleOAuthManager, "_member_call", staticmethod(lambda: True))
    assert await jp.progress_probability("go", TURN, steps_used=4, step_budget=4) is None
    assert jev.calls == []


@pytest.mark.parametrize("failure", [
    jc.JevError(429, "slow down"), jc.JevError(0, "unreachable"), RuntimeError("boom"),
    {"answers": {}},
])
async def test_a_jev_failure_or_odd_answer_is_no_verdict(jev, failure):
    jev.answers = [failure]
    assert await jp.progress_probability("go", TURN, steps_used=4, step_budget=4) is None


async def test_a_slow_jev_times_out_without_a_verdict(jev, monkeypatch):
    async def slow(self, st, questions, model=None):
        await asyncio.sleep(5)
        return _answer(0.99)

    monkeypatch.setattr(jc.JevAPI, "ask", slow)
    monkeypatch.setattr(jp, "TIMEOUT_SECONDS", -2.9)  # wait_for gets 0.1 s
    assert await jp.progress_probability("go", TURN, steps_used=4, step_budget=4) is None


async def test_under_flight_deck_the_owner_key_is_asked_with_a_short_timeout(monkeypatch):
    from captain_claw.google_oauth_manager import GoogleOAuthManager

    monkeypatch.setattr(GoogleOAuthManager, "_flight_deck_base", staticmethod(lambda: "http://fd.test"))
    monkeypatch.setattr(GoogleOAuthManager, "_member_call", staticmethod(lambda: False))
    monkeypatch.setattr(get_config().tools.jev, "check_progress", True)

    async def connected(force=False):
        return True

    seen = {}

    async def via_fd(state, questions, model=None, *, timeout):
        seen.update(state=state, timeout=timeout)
        return _answer(0.7)

    monkeypatch.setattr(jc, "refresh_connected", connected)
    monkeypatch.setattr(jc, "ask_via_flight_deck", via_fd)
    assert await jp.progress_probability("go", TURN, steps_used=4, step_budget=4) == 0.7
    assert seen["timeout"] == jp.TIMEOUT_SECONDS and "web_fetch" in seen["state"]["recent_steps"]


# ── The main loop ──────────────────────────────────────────────────────


class ProbeTool(Tool):
    name = "probe"
    description = "Probe one more source"
    parameters = {
        "type": "object",
        "properties": {"source": {"type": "string"}},
        "required": ["source"],
    }

    async def execute(self, **kwargs) -> ToolResult:
        return ToolResult(success=True, content=f"source {kwargs.get('source')}: 3 rows")


class WorkingProvider(LLMProvider):
    """Probes a new source on every step, then answers after *steps* probes —
    an agent that is plainly making progress, with no plan behind it."""

    def __init__(self, steps: int):
        self.steps = steps
        self.probes = 0

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        if tools and self.probes < self.steps:
            self.probes += 1
            return LLMResponse(
                content=f"Source {self.probes} narrowed it down; probing the next one.",
                tool_calls=[ToolCall(id=f"c{self.probes}", name="probe",
                                     arguments={"source": f"s{self.probes}"})],
            )
        return LLMResponse(content="Found it: 7 open tenders over 1M across the probed sources.")

    async def complete_streaming(self, messages, tools=None, temperature=None, max_tokens=None):
        if False:
            yield ""

    def count_tokens(self, text: str) -> int:
        return len(text)


class _NoSave:
    async def save_session(self, session: Session) -> None:
        return None


def _agent(provider: LLMProvider, outputs: list[dict]) -> Agent:
    agent = Agent(
        provider=provider,
        tool_output_callback=lambda name, args, out: outputs.append({"name": name, "args": args}),
    )
    agent._initialized = True
    agent.session = Session(id="s1", name="default")
    agent.session_manager = _NoSave()
    registry = ToolRegistry()
    registry.register(ProbeTool())
    agent.tools = registry
    agent.max_iterations = 3  # soft limit 3, hard ceiling 6
    # Whatever model the home config names, the loop feeds tool results back.
    agent._supports_tool_result_followup = lambda: True  # type: ignore[method-assign]
    agent.thinking_callback = lambda text, tool, phase: outputs.append(
        {"name": "thinking", "args": {"text": text, "tool": tool}})
    return agent


def _asked_jev_status(outputs: list[dict]) -> bool:
    return any(o["name"] == "thinking" and "asking Jev" in o["args"]["text"] for o in outputs)


def _gate_steps(outputs: list[dict]) -> list[dict]:
    return [o["args"] for o in outputs if o["name"] == "completion_gate"
            and str(o["args"].get("step", "")).startswith("iteration_budget")]


async def test_progress_alone_gives_a_working_agent_more_steps(jev):
    # Up to twice the planned budget a turn that keeps making progress is
    # extended with no judge — no Jev call, no "budget exhausted".
    outputs: list[dict] = []
    agent = _agent(WorkingProvider(steps=5), outputs)
    result = await agent.complete("find the open tenders over 1M")
    assert result.endswith("Found it: 7 open tenders over 1M across the probed sources.")
    assert "budget exhausted" not in result
    assert jev.calls == []
    ext = _gate_steps(outputs)
    assert ext[0]["step"] == "iteration_budget_extended" and ext[0]["judge"] == "progress"
    assert ext[0]["previous_limit"] == 3 and ext[0]["new_limit"] == 9


async def test_jev_yes_gives_a_long_running_agent_more_steps(jev):
    # Past twice the planned budget Jev judges each extension.
    outputs: list[dict] = []
    agent = _agent(WorkingProvider(steps=10), outputs)
    result = await agent.complete("find the open tenders over 1M")
    assert result.endswith("Found it: 7 open tenders over 1M across the probed sources.")
    assert len(jev.calls) == 1
    st, _q = jev.calls[0]
    assert st["request"] == "find the open tenders over 1M"
    assert 'probe({"source": "s9"})' in st["recent_steps"]
    ext = _gate_steps(outputs)
    assert [e["judge"] for e in ext] == ["progress", "jev"]
    assert ext[1]["p_progressing"] == 0.9 and ext[1]["new_limit"] == 12
    assert _asked_jev_status(outputs)


async def test_jev_no_stops_a_long_turn_and_says_how_to_go_on(jev):
    jev.answers = [_answer(0.2)]
    outputs: list[dict] = []
    agent = _agent(WorkingProvider(steps=10), outputs)
    result = await agent.complete("find the open tenders over 1M")
    assert "iteration budget exhausted" in result and "step limit" in result
    assert "continue" in result and "simplify the task" not in result
    # The landing is written fresh from what the turn gathered, not an old
    # mid-run narration.
    assert "Found it: 7 open tenders" in result
    assert len(jev.calls) == 1
    stop = _gate_steps(outputs)[-1]
    assert stop["step"] == "iteration_budget_exhausted"
    assert stop["judge"] == "jev" and stop["p_progressing"] == 0.2


async def test_without_jev_progress_runs_to_the_hard_ceiling(jev, monkeypatch):
    monkeypatch.setattr(get_config().tools.jev, "check_progress", False)
    outputs: list[dict] = []
    agent = _agent(WorkingProvider(steps=10), outputs)
    result = await agent.complete("find the open tenders over 1M")
    # Most decks have no Jev: a working turn is not cut off at the soft limit.
    assert result.endswith("Found it: 7 open tenders over 1M across the probed sources.")
    assert jev.calls == []
    assert all(e["judge"] == "progress" for e in _gate_steps(outputs))
    # No "asking Jev" status on a deck where Jev isn't asked.
    assert not _asked_jev_status(outputs)


async def test_the_hard_ceiling_still_holds_and_ships_what_the_turn_has(jev):
    outputs: list[dict] = []
    agent = _agent(WorkingProvider(steps=50), outputs)
    result = await agent.complete("find the open tenders over 1M")
    # Progress, then Jev, extend to 12 = 4x the planned 3; then the turn ends
    # with a written landing instead of a bare "Max iterations".
    assert len(jev.calls) == 1
    assert "iteration budget exhausted" in result and "(12 steps)" in result
    assert "Found it: 7 open tenders" in result
    assert "Max iterations reached" not in result


class IdleProvider(WorkingProvider):
    """Repeats the same probe: no new progress after the first call."""

    async def complete(self, messages, tools=None, temperature=None, max_tokens=None):
        if tools:
            return LLMResponse(content="Checking again.", tool_calls=[
                ToolCall(id=f"c{len(messages)}", name="probe", arguments={"source": "s1"})])
        return LLMResponse(content="Found it: 7 open tenders over 1M across the probed sources.")


async def test_without_progress_and_without_jev_the_turn_stops_at_the_soft_limit(jev, monkeypatch):
    monkeypatch.setattr(get_config().tools.jev, "check_progress", False)
    outputs: list[dict] = []
    agent = _agent(IdleProvider(steps=50), outputs)
    agent.max_iterations = 8
    await agent.complete("find the open tenders over 1M")
    # Repeating the same call is not progress: no extra steps (the stuck
    # detector or the soft limit ends it).
    assert not any(e.get("step") == "iteration_budget_extended" for e in _gate_steps(outputs))


def test_budget_extension_rule():
    from captain_claw.agent_orchestration_mixin import _budget_extends as ext

    # No progress: never free steps.
    assert not ext(False, True, 3, 6, judge_available=False)
    # Progress below twice the plan: always.
    assert ext(True, False, 3, 6, judge_available=True)
    # Beyond it: open work, or no judge on the deck.
    assert ext(True, True, 9, 6, judge_available=True)
    assert ext(True, False, 9, 6, judge_available=False)
    assert not ext(True, False, 9, 6, judge_available=True)  # Jev decides
