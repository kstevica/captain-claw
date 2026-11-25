"""A turn stopped by the plan's usage limit carries on by itself after the reset.

An agent on the person's ChatGPT subscription that hits the plan's usage
limit used to retry four times over ~75 s (a 429 read as "busy"), then show
the provider's raw refusal. Now:

- the limit is told apart from an ordinary rate limit, and from a refused
  key, billing or an empty balance, which never reset by themselves;
- the turn isn't retried: it ends with a plain message saying when it carries
  on, never the provider's text;
- exactly one one-shot job in the agent's own scheduled jobs carries on at
  the reset plus a margin — in the chat the limit stopped — and it's listed
  where Flight Deck shows scheduled work and can be removed there;
- a later turn the limit stops moves that job instead of adding another.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from captain_claw import config as cfg_mod
from captain_claw import cron_dispatch as cd
from captain_claw import limit_carry_on as lc
from captain_claw import mail_authority, turn_register
from captain_claw.cron import parse_iso
from captain_claw.exceptions import LLMAPIError, LLMError
from captain_claw.llm import LLMProvider, LLMResponse, Message, ToolDefinition
from captain_claw.plan_limit import LIMIT_LONGEST_S, PlanLimit, from_error, plan_limit, reset_said
from captain_claw.session import SessionManager

NOW = 1_764_000_000.0  # a fixed "now" (epoch seconds)


def chatgpt_429(**error: object) -> LLMAPIError:
    body = json.dumps({"error": {"type": "usage_limit_reached", "message": "The usage limit has been reached",
                                 "plan_type": "plus", **error}})
    return LLMAPIError(f"ChatGPT Responses API error 429: {body}", status_code=429)


# ── recognising the limit ─────────────────────────────────────────────


def test_chatgpt_refusal_with_reset_time():
    got = from_error(chatgpt_429(resets_at=int(NOW) + 7200), now=NOW)
    assert got == PlanLimit(reset_at=NOW + 7200, plan="ChatGPT")


def test_chatgpt_refusal_with_seconds_to_wait():
    got = from_error(chatgpt_429(resets_in_seconds=5400), now=NOW)
    assert got is not None and got.reset_at == NOW + 5400 and got.plan == "ChatGPT"


def test_absolute_reset_wins_over_seconds():
    got = from_error(chatgpt_429(resets_at=int(NOW) + 600, resets_in_seconds=99999), now=NOW)
    assert got.reset_at == NOW + 600


def test_limit_said_mid_stream_without_a_status():
    err = LLMAPIError('ChatGPT Responses API returned an error event: {"type": "response.failed", '
                      '"response": {"error": {"code": "usage_limit_reached"}}}')
    got = from_error(err, now=NOW)
    assert got is not None and got.reset_at is None and got.plan == "ChatGPT"


def test_limit_raised_from_another_error():
    try:
        try:
            raise chatgpt_429(resets_in_seconds=60)
        except LLMAPIError as inner:
            raise LLMError("call failed") from inner
    except LLMError as outer:
        got = from_error(outer, now=NOW)
    assert got is not None and got.reset_at == NOW + 60


def test_claude_plan_limits():
    epoch = int(NOW) + 3600
    got = from_error(LLMError(f"Claude AI usage limit reached|{epoch}"), now=NOW)
    assert got == PlanLimit(reset_at=float(epoch), plan="Claude")
    at = reset_said("You've hit your limit · resets 5pm (UTC)", NOW)
    assert at is not None and datetime.fromtimestamp(at, UTC).strftime("%H:%M") == "17:00"
    assert plan_limit("5-hour limit reached ∙ resets 3pm")


def test_reset_said_in_words():
    assert reset_said("try again in 2 hours 15 minutes", NOW) == NOW + 2 * 3600 + 15 * 60
    assert reset_said("usage limit reached", NOW) is None


@pytest.mark.parametrize("err", [
    # An API key's per-minute rate limit: a few seconds' wait cures it.
    LLMAPIError("Rate limit reached for gpt-4o in organization org-x on tokens per min (TPM): "
                "Limit 30000, Used 30000, Requested 900. Please try again in 1.2s.", status_code=429),
    LLMAPIError("ChatGPT Responses API error 429: Too Many Requests", status_code=429),
    LLMAPIError("Anthropic API error 529: overloaded", status_code=529),
    # Billing, an empty balance, a refused key: they never reset by themselves.
    LLMAPIError("You exceeded your current quota, please check your plan and billing details. "
                "usage limit", status_code=429),
    LLMAPIError("Your credit balance is too low to access the API (usage limit)", status_code=400),
    LLMAPIError("ChatGPT Responses API error 429: " + json.dumps(
        {"error": {"type": "usage_not_included", "message": "Usage limit: not included in your plan"}}),
        status_code=429),
    LLMAPIError("Incorrect API key provided; usage limit", status_code=401),
    # The words, but not a 429: never a plan limit.
    LLMAPIError("usage limit reached while parsing", status_code=500),
])
def test_not_a_plan_limit(err):
    assert from_error(err, now=NOW) is None


def test_reset_too_far_off_is_not_believed():
    got = from_error(chatgpt_429(resets_at=int(NOW + LIMIT_LONGEST_S + 3600)), now=NOW)
    assert got is not None and got.reset_at is None


def test_reset_already_past_means_now():
    assert from_error(chatgpt_429(resets_at=int(NOW) - 30), now=NOW).reset_at == NOW


# ── the message ───────────────────────────────────────────────────────


def test_message_says_when_it_carries_on_and_how_to_cancel():
    limit = PlanLimit(reset_at=NOW + 7710, plan="ChatGPT")
    carry = lc.CarryOn(job_id="j1", at=NOW + 7800, guessed=False)
    text = lc.message(limit, carry, NOW, zone=UTC)
    clock = datetime.fromtimestamp(NOW + 7800, UTC).strftime("%H:%M")
    assert text == (f"The ChatGPT plan's usage limit is used up; I'll carry on by myself at "
                    f"{clock} UTC (in 2 h 10 min). To cancel, remove it under Scheduled tasks.")


def test_message_never_shows_the_provider_text():
    err = chatgpt_429(resets_in_seconds=3600)
    limit = from_error(err, now=NOW)
    text = lc.message(limit, lc.CarryOn("j1", NOW + 3690, False), NOW, zone=UTC)
    assert "429" not in text and "usage_limit_reached" not in text and "{" not in text


def test_message_when_reset_unknown_or_not_carried_on():
    guessed = lc.message(PlanLimit(None, "ChatGPT"), lc.CarryOn("j1", NOW + 1800, True), NOW, zone=UTC)
    assert "didn't say when it resets" in guessed and "try again by myself" in guessed and "(in 30 min)" in guessed
    plain = lc.message(PlanLimit(NOW + 3600, ""), None, NOW, zone=UTC)
    assert plain.startswith("The model plan's usage limit is used up until ")
    assert "carry on by myself" not in plain and "Send me a message" in plain


def test_clock_and_span():
    assert lc.span(NOW + 45 * 60, NOW) == "45 min"
    assert lc.span(NOW + 2 * 3600, NOW) == "2 h"
    assert lc.span(NOW + 3 * 86400 + 4 * 3600, NOW) == "3 d 4 h"
    # Another day names the day; rounded up, never earlier than the moment.
    later = lc.clock(NOW + 3 * 86400 + 10, NOW, zone=UTC)
    assert later.split()[0] in {"Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"}
    assert lc.clock(NOW + 61, NOW, zone=UTC).endswith("UTC")


# ── the agent loop ────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _cfg(monkeypatch):
    monkeypatch.setattr(cfg_mod, "_config", cfg_mod.Config())
    monkeypatch.setattr(lc, "_zone", lambda: UTC)


class RefusingProvider(LLMProvider):
    """A model whose every call is refused with *err*."""

    def __init__(self, err: Exception):
        self.err = err
        self.calls = 0

    async def complete(self, messages: list[Message], tools: list[ToolDefinition] | None = None,
                       temperature: float | None = None, max_tokens: int | None = None) -> LLMResponse:
        self.calls += 1
        raise self.err

    async def complete_streaming(self, messages, tools=None, temperature=None, max_tokens=None):
        if False:
            yield ""

    def count_tokens(self, text: str) -> int:
        return len(text)


def run(tmp_path, body):
    async def main():
        sm = SessionManager(tmp_path / "sessions.db")
        try:
            await body(sm)
        finally:
            await sm.close()

    asyncio.run(main())


async def make_agent(sm, provider, runs_cron=True):
    from captain_claw.agent import Agent
    from captain_claw.tools.registry import ToolRegistry

    agent = Agent(provider=provider)
    agent._initialized = True
    agent.session = await sm.create_session(name="chat")
    agent.session_manager = sm
    agent.tools = ToolRegistry()
    if runs_cron:
        agent._runtime_context = object()   # the agent the cron loop runs with
    return agent


@pytest.fixture
def sleeps(monkeypatch):
    """The loop's retry waits, recorded instead of slept."""
    import captain_claw.agent_orchestration_mixin as mixin

    waited: list[float] = []
    real_sleep = asyncio.sleep

    async def fake_sleep(delay, *a, **k):
        waited.append(delay)
        await real_sleep(0)

    monkeypatch.setattr(mixin.asyncio, "sleep", fake_sleep)
    return waited


async def carry_ons(sm):
    return [j for j in await sm.list_cron_jobs(active_only=True) if lc.marker(j.payload)]


def test_plan_limit_is_not_retried_and_one_carry_on_is_scheduled(tmp_path, sleeps):
    async def body(sm):
        provider = RefusingProvider(chatgpt_429(resets_in_seconds=7200))
        agent = await make_agent(sm, provider)
        before = datetime.now(UTC).timestamp()
        reply = await agent.complete("write the weekly report")

        assert provider.calls == 1
        assert not [d for d in sleeps if d >= 5]   # none of the 5/10/20/40 s retry waits
        assert reply.startswith("The ChatGPT plan's usage limit is used up; I'll carry on by myself at ")
        assert "(in 2 h 2 min)" in reply and "Scheduled tasks" in reply and "429" not in reply
        assert agent.session.messages[-1]["role"] == "assistant"
        assert agent.session.messages[-1]["content"] == reply

        jobs = await carry_ons(sm)
        assert len(jobs) == 1
        job = jobs[0]
        assert job.kind == "prompt" and job.enabled and job.session_id == agent.session.id
        assert job.schedule["type"] == "once"
        due = parse_iso(job.next_run_at).timestamp()
        assert parse_iso(job.schedule["target_iso"]).timestamp() == due
        assert before + 7200 + 90 - 5 <= due <= before + 7200 + 90 + 30
        assert lc.run_session(job.payload) == agent.session.id
        assert job.payload["text"] == lc.CARRY_ON_TEXT

    run(tmp_path, body)


def test_a_later_turn_moves_the_carry_on_instead_of_adding_one(tmp_path, sleeps):
    async def body(sm):
        provider = RefusingProvider(chatgpt_429(resets_in_seconds=3600))
        agent = await make_agent(sm, provider)
        await agent.complete("write the weekly report")
        first = (await carry_ons(sm))[0]

        provider.err = chatgpt_429(resets_in_seconds=4 * 3600)   # the person writes again
        await agent.complete("are you there?")
        jobs = await carry_ons(sm)
        assert [j.id for j in jobs] == [first.id]
        assert parse_iso(jobs[0].next_run_at) > parse_iso(first.next_run_at)
        assert jobs[0].schedule["target_iso"] == jobs[0].next_run_at

    run(tmp_path, body)


def test_the_carry_on_stopped_again_leaves_exactly_one_for_later(tmp_path, sleeps):
    """The carry-on's own run hits the limit: its job turns itself off after
    the run (a one-shot), so the next one is a new job — one in all."""
    async def body(sm):
        provider = RefusingProvider(chatgpt_429(resets_in_seconds=600))
        agent = await make_agent(sm, provider)
        await agent.complete("write the weekly report")
        running = (await carry_ons(sm))[0]

        tok = turn_register.bind_automation("cron", running.id, lc.CARRY_ON_TEXT)
        try:
            await agent.complete(lc.CARRY_ON_TEXT)
        finally:
            turn_register.reset_automation(tok)
        await sm.update_cron_job(running.id, enabled=False, last_status="completed")  # as execute_cron_job does

        jobs = await carry_ons(sm)
        assert len(jobs) == 1 and jobs[0].id != running.id

    run(tmp_path, body)


def test_ordinary_rate_limit_is_still_retried(tmp_path, sleeps):
    async def body(sm):
        provider = RefusingProvider(LLMAPIError(
            "Rate limit reached for gpt-4o on tokens per min. Please try again in 1.2s.", status_code=429))
        agent = await make_agent(sm, provider)
        with pytest.raises(LLMAPIError):
            await agent.complete("write the weekly report")
        assert provider.calls == 5
        assert [d for d in sleeps if d >= 5] == [5, 10, 20, 40]
        assert await carry_ons(sm) == []

    run(tmp_path, body)


def test_a_turn_another_automation_started_gets_no_carry_on(tmp_path, sleeps):
    async def body(sm):
        provider = RefusingProvider(chatgpt_429(resets_in_seconds=3600))
        agent = await make_agent(sm, provider)
        tok = turn_register.bind_automation("fd_scheduler", "task-1")
        try:
            reply = await agent.complete("daily digest")
        finally:
            turn_register.reset_automation(tok)
        assert provider.calls == 1
        assert "usage limit is used up until" in reply and "carry on by myself" not in reply
        assert await carry_ons(sm) == []

    run(tmp_path, body)


def test_a_worker_agent_gets_no_carry_on(tmp_path, sleeps):
    """An orchestrator worker or a sister session shares the sessions file
    but not the cron loop: told plainly, nothing scheduled for its session."""
    async def body(sm):
        provider = RefusingProvider(chatgpt_429(resets_in_seconds=3600))
        agent = await make_agent(sm, provider, runs_cron=False)
        reply = await agent.complete("summarise part 3")
        assert provider.calls == 1
        assert "usage limit is used up until" in reply and "carry on by myself" not in reply
        assert await carry_ons(sm) == []

    run(tmp_path, body)


def test_no_reset_time_tries_again_a_few_times_then_waits_for_the_person(tmp_path):
    async def body(sm):
        agent = SimpleNamespace(session=await sm.create_session(name="chat"), session_manager=sm,
                                _runtime_context=object())
        limit = PlanLimit(reset_at=None, plan="ChatGPT")
        for n in range(lc.GUESSES):
            jobs = await carry_ons(sm)
            tok = turn_register.bind_automation("cron", jobs[0].id) if jobs else None
            try:
                got = await lc.schedule(agent, limit, now=NOW)
            finally:
                turn_register.reset_automation(tok)
            assert got is not None and got.guessed and got.at == NOW + lc.GUESS_S
            for job in jobs:
                await sm.update_cron_job(job.id, enabled=False)
            assert len(await carry_ons(sm)) == 1
            assert lc.marker((await carry_ons(sm))[0].payload)["guesses"] == n + 1
        last = (await carry_ons(sm))[0]
        tok = turn_register.bind_automation("cron", last.id)
        try:
            assert await lc.schedule(agent, limit, now=NOW) is None
        finally:
            turn_register.reset_automation(tok)

    run(tmp_path, body)


# ── where it runs, and where the person sees it ──────────────────────


def test_carry_on_runs_back_in_the_chat_the_limit_stopped(tmp_path, monkeypatch):
    import captain_claw.prompt_execution as pe

    sent: list[dict] = []

    async def fake_dispatch(ctx_, **kwargs):
        sent.append(kwargs)
        return "executed"

    monkeypatch.setattr(pe, "dispatch_prompt_in_session", fake_dispatch)

    async def body(sm):
        chat = await sm.create_session(name="chat")
        agent = SimpleNamespace(session=chat, session_manager=sm, _runtime_context=object())
        got = await lc.schedule(agent, PlanLimit(NOW, "ChatGPT"), now=NOW)
        ctx = SimpleNamespace(
            agent=SimpleNamespace(session_manager=sm, session=None), cron_running_job_ids=set(),
            ui=SimpleNamespace(print_error=lambda *a, **k: None, print_message=lambda *a, **k: None,
                               append_tool_output=lambda *a, **k: None),
            on_cron_output=None,
        )
        await cd.execute_cron_job(ctx, await sm.load_cron_job(got.job_id), trigger="scheduled")

        assert len(sent) == 1
        assert sent[0]["session_id"] == chat.id and sent[0]["prompt_text"] == lc.CARRY_ON_TEXT
        assert sent[0]["cron_job_id"] == got.job_id
        job = await sm.load_cron_job(got.job_id)
        assert job.enabled is False                      # a one-shot: done after its run
        assert not job.run_session_id                    # no job session of its own

    run(tmp_path, body)


def test_carry_on_is_listed_in_flight_deck_and_can_be_removed(tmp_path, monkeypatch):
    from captain_claw.flight_deck import agent_cron_routes as acr

    db = tmp_path / "sessions.db"
    monkeypatch.setattr(acr, "_db_path", lambda slug: db)

    async def body(sm):
        agent = SimpleNamespace(session=await sm.create_session(name="chat"), session_manager=sm,
                                _runtime_context=object())
        got = await lc.schedule(agent, PlanLimit(NOW + 3600, "ChatGPT"), now=NOW)

        rows = acr._read_rows("helper")
        shown = [acr._public_job("helper", "Helper", r, "on", NOW) for r in rows]
        assert len(shown) == 1 and shown[0]["id"] == got.job_id
        assert shown[0]["task"] == lc.CARRY_ON_TEXT and shown[0]["schedule_type"] == "once"
        assert shown[0]["enabled"] is True and shown[0]["next_run_at"] == NOW + 3600 + 90

        assert await sm.delete_cron_job(got.job_id)
        assert await carry_ons(sm) == []
        conn = sqlite3.connect(db)
        try:
            assert conn.execute("SELECT count(*) FROM cron_jobs WHERE enabled = 1").fetchone()[0] == 0
        finally:
            conn.close()

    run(tmp_path, body)


def test_carry_on_mail_authority_is_the_interrupted_request():
    payload = {"text": lc.CARRY_ON_TEXT, "author": lc.AUTHOR,
               "mail_intent_text": "email the weekly report to the team"}
    assert mail_authority.cron_job_text(payload) == "email the weekly report to the team"
    assert mail_authority.cron_job_text({**payload, "mail_intent_text": ""}) == ""
    # A job the agent set up is still judged on the narrower of both texts.
    agent_job = {"text": "check the inbox", "author": "agent", "mail_intent_text": "email the team"}
    assert mail_authority.cron_job_text(agent_job) == mail_authority.narrower_intent(
        "check the inbox", "email the team")
