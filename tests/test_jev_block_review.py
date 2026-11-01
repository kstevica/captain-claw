"""When a guard refuses a repeated call, Jev picks what the agent does next.

The review (``captain_claw/jev_block_review.py``) asks Jev one choice question
about the refused call; the tool loop (``agent_tool_loop_mixin``) swaps the
guard's usual message for the picked instruction, or — for a read-only call
only — lets the repeat run once. Never in the way: no key, an error or an
unsure pick means the usual message.
"""

from __future__ import annotations

import types

import pytest

from captain_claw import jev_block_review as jb
from captain_claw import jev_client as jc
from captain_claw.agent import Agent
from captain_claw.config import get_config
from captain_claw.llm import LLMProvider, LLMResponse, ToolCall
from captain_claw.session import Session
from captain_claw.tools.registry import Tool, ToolRegistry, ToolResult
from captain_claw.tools.write import WriteTool


def _answer(choice: str, p: float) -> dict:
    return {
        "answers": {jb.QUESTION_NAME: {
            "type": "choice", "choice": choice, "confidence": p,
            "probabilities": {choice: p},
        }},
        "model": "jev-latest",
    }


@pytest.fixture
def jev(monkeypatch):
    """Standalone with a key; JevAPI.ask scripted (the last answer repeats)."""
    from captain_claw.google_oauth_manager import GoogleOAuthManager

    monkeypatch.setattr(GoogleOAuthManager, "_flight_deck_base", staticmethod(lambda: ""))
    monkeypatch.setattr(GoogleOAuthManager, "_member_call", staticmethod(lambda: False))
    monkeypatch.setenv("JEV_API_KEY", "jev-test")
    monkeypatch.setattr(get_config().tools.jev, "api_key", "")
    monkeypatch.setattr(get_config().tools.jev, "review_blocks", True)
    monkeypatch.setattr(get_config().tools, "duplicate_call_max", 1)
    state = types.SimpleNamespace(calls=[], answers=[_answer("continue", 0.9)])

    async def ask(self, st, questions, model=None):
        state.calls.append((st, questions))
        jc.build_request(st, questions, model)  # what would leave must validate
        ans = state.answers[0] if len(state.answers) == 1 else state.answers.pop(0)
        if isinstance(ans, BaseException):
            raise ans
        return ans

    monkeypatch.setattr(jc.JevAPI, "ask", ask)
    return state


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


class _Cron(Tool):
    """A store that changes between reads: `list` after `remove` differs."""

    name = "cron"
    description = "jobs"
    parameters = {"type": "object", "properties": {"action": {"type": "string"}}}

    def __init__(self) -> None:
        self.ran: list[str] = []

    async def execute(self, action: str = "", **kwargs) -> ToolResult:
        self.ran.append(action)
        return ToolResult(success=True, content=f"{action} ok ({len(self.ran)})")


def _agent(tmp_path):
    agent = Agent(provider=_Dummy())
    agent._initialized = True
    agent.session = Session(id="s1", name="default")
    agent.session_manager = _DummySM()
    registry = ToolRegistry(base_path=tmp_path)
    registry.register(WriteTool())
    cron = _Cron()
    registry.register(cron)
    agent.tools = registry
    agent._malformed_retry_count = 0
    agent._blind_write_paths = set()
    agent._turn_tool_call_counts = {}
    agent._jev_block_reviews = 0
    agent._jev_released_sigs = set()
    agent._turn_user_text = "Write the battery report, share it and email the link."
    agent.session.add_message("user", agent._turn_user_text)
    agent.last_turn_start_idx = 0
    return agent, cron


REPORT = "# Battery report\n\n" + "| 18:10 | 53% | 3.72 V |\n" * 20


# ── The question ───────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("kind", "target", "releasable", "expected"),
    [
        (jb.BLIND_REWRITE, "r.md", False, ["verify", "continue", "finish", "ask_user"]),
        (jb.DUPLICATE, "r.md", False, ["verify", "continue", "finish", "ask_user"]),
        (jb.DUPLICATE, "", True, ["continue", "finish", "ask_user", jb.RELEASE]),
        (jb.DUPLICATE, "", False, ["continue", "finish", "ask_user"]),
    ],
)
def test_options_fit_the_refusal(kind, target, releasable, expected):
    options = jb.options_for(kind, target=target, releasable=releasable)
    assert options == expected
    q = jc.normalize_questions(jb.questions_for(options))[jb.QUESTION_NAME]
    assert q["type"] == "choice" and list(q["criteria"]) == expected


@pytest.mark.parametrize(
    ("tool", "action", "ok"),
    [
        ("read", "", True),
        ("web_fetch", "", True),
        ("cron", "list", True),
        ("google_mail", "search", True),
        ("cron", "remove", False),
        ("google_mail", "send", False),
        ("write", "", False),
        ("shell", "", False),
    ],
)
def test_only_lookups_can_be_released(tool, action, ok):
    assert jb.can_release(tool, action) is ok


def test_the_state_carries_the_refused_call_and_why():
    st = jb.build_state(
        "share it", [{"role": "user", "content": "share it"}],
        kind=jb.DUPLICATE, tool="cron", arguments={"action": "list"}, count=1,
    )
    assert st["request"] == "share it"
    assert st["blocked_call"] == 'cron({"action": "list"})'
    assert "1 time(s)" in st["why_blocked"]


# ── In the tool loop ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_blind_rewrite_gets_the_picked_next_step(jev, tmp_path):
    jev.answers = [_answer("continue", 0.9)]
    agent, _ = _agent(tmp_path)
    await agent._handle_tool_calls([ToolCall(id="w1", name="write",
                                             arguments={"path": "r.md", "content": REPORT})])
    results = await agent._handle_tool_calls([ToolCall(
        id="w2", name="write", arguments={"path": "r.md", "content": REPORT + "\nagain"})])
    msg = results[0]["content"]
    assert msg.startswith("BLIND REWRITE BLOCKED")  # the loop's other checks key on it
    assert "go on with the rest of the request" in msg
    assert len(jev.calls) == 1
    state, questions = jev.calls[0]
    assert "write(" in state["blocked_call"]
    assert "verify" in questions[jb.QUESTION_NAME]["options"]  # the file can be checked


@pytest.mark.asyncio
async def test_an_unsure_pick_keeps_the_usual_message(jev, tmp_path):
    jev.answers = [_answer("finish", 0.4)]
    agent, _ = _agent(tmp_path)
    await agent._handle_tool_calls([ToolCall(id="w1", name="write",
                                             arguments={"path": "r.md", "content": REPORT})])
    results = await agent._handle_tool_calls([ToolCall(
        id="w2", name="write", arguments={"path": "r.md", "content": REPORT + "\nagain"})])
    msg = results[0]["content"]
    assert "If you doubt what the file holds, read it first." in msg  # the guard's own text
    assert "request looks done" not in msg


@pytest.mark.asyncio
async def test_a_legitimate_read_only_repeat_runs_once(jev, tmp_path):
    jev.answers = [_answer(jb.RELEASE, 0.9)]
    agent, cron = _agent(tmp_path)
    for cid in ("c1", "c2"):
        await agent._handle_tool_calls([ToolCall(id=cid, name="cron", arguments={"action": "list"})])
    assert cron.ran == ["list", "list"]  # Jev let the second one through
    # A third identical list is not offered for release again.
    results = await agent._handle_tool_calls(
        [ToolCall(id="c3", name="cron", arguments={"action": "list"})])
    assert cron.ran == ["list", "list"]
    assert results[0]["content"].startswith("DUPLICATE CALL BLOCKED")
    assert jb.RELEASE not in jev.calls[-1][1][jb.QUESTION_NAME]["options"]


@pytest.mark.asyncio
async def test_a_call_that_changes_something_is_never_released(jev, tmp_path):
    jev.answers = [_answer(jb.RELEASE, 0.99)]
    agent, cron = _agent(tmp_path)
    for cid in ("c1", "c2"):
        await agent._handle_tool_calls([ToolCall(id=cid, name="cron", arguments={"action": "remove"})])
    assert cron.ran == ["remove"]
    # Not offered, so Jev's answer is out of range and the usual message stays.
    assert jb.RELEASE not in jev.calls[0][1][jb.QUESTION_NAME]["options"]


@pytest.mark.asyncio
async def test_no_key_means_no_review(monkeypatch, tmp_path):
    from captain_claw.google_oauth_manager import GoogleOAuthManager

    monkeypatch.setattr(GoogleOAuthManager, "_flight_deck_base", staticmethod(lambda: ""))
    monkeypatch.setattr(GoogleOAuthManager, "_member_call", staticmethod(lambda: False))
    monkeypatch.delenv("JEV_API_KEY", raising=False)
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.setattr(get_config().tools.jev, "api_key", "")
    monkeypatch.setattr(get_config().tools, "duplicate_call_max", 1)
    agent, cron = _agent(tmp_path)
    for cid in ("c1", "c2"):
        await agent._handle_tool_calls([ToolCall(id=cid, name="cron", arguments={"action": "list"})])
    assert cron.ran == ["list"]
    # Not ready: the rest of the turn skips the check instead of re-asking.
    assert agent._jev_block_reviews == jb.MAX_REVIEWS_PER_TURN


@pytest.mark.asyncio
async def test_reviews_stop_after_the_per_turn_cap(jev, tmp_path):
    jev.answers = [_answer("continue", 0.9)]
    agent, _ = _agent(tmp_path)
    for i in range(jb.MAX_REVIEWS_PER_TURN + 2):
        await agent._handle_tool_calls(
            [ToolCall(id=f"c{i}a", name="cron", arguments={"action": "remove", "n": i})])
        await agent._handle_tool_calls(
            [ToolCall(id=f"c{i}b", name="cron", arguments={"action": "remove", "n": i})])
    assert len(jev.calls) == jb.MAX_REVIEWS_PER_TURN


@pytest.mark.asyncio
async def test_a_jev_error_keeps_the_usual_message(jev, tmp_path):
    jev.answers = [jc.JevError(500, "boom")]
    agent, cron = _agent(tmp_path)
    for cid in ("c1", "c2"):
        results = await agent._handle_tool_calls(
            [ToolCall(id=cid, name="cron", arguments={"action": "list"})])
    assert cron.ran == ["list"]
    assert "Use the data you already have" in results[0]["content"]


def test_review_is_off_by_config(monkeypatch):
    monkeypatch.setattr(get_config().tools.jev, "review_blocks", False)
    assert jb.enabled() is False


class _Crew(_Cron):
    name = "crew"
    description = "workers"


@pytest.mark.asyncio
async def test_crew_can_be_asked_again_but_not_told_twice(jev, tmp_path):
    """A task moves on between reads: a blocked `list` once left an agent reporting a
    task that had started as one that never had. A repeated dispatch is a second worker."""
    agent, _ = _agent(tmp_path)
    crew = _Crew()
    agent.tools.register(crew)
    for cid in ("c1", "c2", "c3"):
        results = await agent._handle_tool_calls([ToolCall(id=cid, name="crew", arguments={"action": "list"})])
        assert "BLOCKED" not in results[0]["content"]
    assert crew.ran == ["list", "list", "list"]
    made = {"action": "dispatch", "harness": "opencode", "brief": "b"}
    await agent._handle_tool_calls([ToolCall(id="d1", name="crew", arguments=made)])
    results = await agent._handle_tool_calls([ToolCall(id="d2", name="crew", arguments=made)])
    assert results[0]["content"].startswith("DUPLICATE CALL BLOCKED")
    assert crew.ran.count("dispatch") == 1
