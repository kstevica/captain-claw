"""Tests for the Basna/Vatra Horizon worker (Phase 1).

All spawn/send/stop seams and the critic provider are injected, so these exercise
the pool + engine wiring without spawning real processes or calling a real model.
Focus: independent rollouts (distinct pool agents per vote sample), the critics
gate, pool teardown, budget honoring, and graceful degradation.
"""

from __future__ import annotations

import asyncio

from captain_claw.flight_deck.horizon_worker import (
    HorizonConfig,
    _with_heartbeat,
    run_horizon_closer,
    run_worker_horizon,
)
from captain_claw.llm import LLMResponse

# ── Stubs ────────────────────────────────────────────────────────────

def make_spawn(n_fail: int = 0):
    """Spawn stub → (port, token, slug). The first ``n_fail`` calls raise."""
    state = {"i": 0}
    calls: list[str] = []

    async def spawn(name_suffix: str):
        idx = state["i"]
        state["i"] += 1
        calls.append(name_suffix)
        if idx < n_fail:
            raise RuntimeError(f"spawn boom {idx}")
        port = 1000 + idx
        return port, f"tok-{port}", f"slug-{idx}"

    spawn.calls = calls  # type: ignore[attr-defined]
    return spawn


def make_send(reply_for=None, emit_action: bool = False):
    """Send stub → reply text. Records every (port, prompt); optional tool action.

    ``reply_for(port, prompt) -> str`` customizes the reply; default agrees on 42.
    """
    seen: list[dict] = []

    async def send(port, token, prompt, timeout, *, fleet_instructions="",
                   agent_name="", on_action=None, on_usage=None):
        seen.append({"port": port, "prompt": prompt})
        if emit_action and on_action is not None:
            on_action({"tool": "search", "detail": "q"})
        if on_usage is not None:
            on_usage(10, 5, 15)
        return reply_for(port, prompt) if reply_for else "reasoning…\nAnswer: 42"

    send.seen = seen  # type: ignore[attr-defined]
    return send


def make_stop():
    stopped: list[str] = []

    async def stop(slug: str):
        stopped.append(slug)

    stop.stopped = stopped  # type: ignore[attr-defined]
    return stop


class StubProvider:
    """A critic provider: every ``complete`` returns the same canned verdict line."""

    def __init__(self, reply: str):
        self.reply = reply
        self.calls = 0

    async def complete(self, messages, tools=None, temperature=None, max_tokens=None):
        self.calls += 1
        return LLMResponse(content=self.reply, finish_reason="stop")


def cfg(**kw) -> HorizonConfig:
    base = dict(samples=3, fix_attempts=1, agreement_threshold=0.6)
    base.update(kw)
    return HorizonConfig.from_dict(base)


# ── Tests ────────────────────────────────────────────────────────────

async def test_independent_rollouts_use_distinct_pool_agents():
    """The N-sample vote must hit N distinct spawned agents, not one re-prompted."""
    spawn, send, stop = make_spawn(), make_send(), make_stop()
    spawned_slugs: list[str] = []
    res = await run_worker_horizon(
        spawn=spawn, tier="fast", prompt="Q", cfg=cfg(samples=3),
        critic_provider=None, send=send, stop=stop,
        on_spawn=lambda s: spawned_slugs.extend(s),
    )
    # Pool of 3 spawned, all 3 distinct ports exercised across the run.
    assert len(spawn.calls) == 3
    assert spawned_slugs == ["slug-0", "slug-1", "slug-2"]
    ports = {s["port"] for s in send.seen}
    assert ports == {1000, 1001, 1002}
    # Agreement is unanimous → passes at the vote rung, best answer returned.
    assert res["ok"] and res["passed"]
    assert "42" in res["output"]


async def test_pool_is_disposed():
    spawn, send, stop = make_spawn(), make_send(), make_stop()
    await run_worker_horizon(spawn=spawn, tier="fast", prompt="Q", cfg=cfg(),
                             critic_provider=None, send=send, stop=stop)
    assert set(stop.stopped) == {"slug-0", "slug-1", "slug-2"}


async def test_critics_skipped_when_agreement_high():
    """Unanimous answers clear the cheap agreement gate — critics never fire."""
    critic = StubProvider("REFUTED: nope")  # would fail the run if ever called
    spawn, send, stop = make_spawn(), make_send(), make_stop()
    res = await run_worker_horizon(spawn=spawn, tier="fast", prompt="Q", cfg=cfg(),
                                   critic_provider=critic, send=send, stop=stop)
    assert critic.calls == 0
    assert res["passed"]


async def test_high_stakes_runs_critics_and_a_refutation_blocks_pass():
    """stakes=high forces critics even on agreement; a refuting panel fails the gate."""
    critic = StubProvider("REFUTED: flawed")
    spawn, send, stop = make_spawn(), make_send(), make_stop()
    res = await run_worker_horizon(
        spawn=spawn, tier="fast", prompt="Q", cfg=cfg(stakes="high", fix_attempts=1),
        critic_provider=critic, send=send, stop=stop)
    assert critic.calls > 0          # critics ran
    assert res["passed"] is False    # majority refuted → not verified
    assert res["ok"] and res["output"]  # but best-so-far still returned (no truncation)


async def test_no_pool_spawned_degrades_gracefully():
    spawn, send, stop = make_spawn(n_fail=3), make_send(), make_stop()
    on_spawn_got: list[list[str]] = []
    res = await run_worker_horizon(
        spawn=spawn, tier="fast", prompt="Q", cfg=cfg(samples=3),
        critic_provider=None, send=send, stop=stop,
        on_spawn=lambda s: on_spawn_got.append(list(s)))
    assert res["ok"] is False
    assert res["error"] == "no pool agent spawned"
    assert send.seen == []           # nothing dispatched
    assert stop.stopped == []        # nothing to dispose
    assert on_spawn_got == [[]]


async def test_partial_spawn_failure_uses_survivors():
    """One pool spawn fails; the run proceeds on the agents that came up."""
    spawn, send, stop = make_spawn(n_fail=1), make_send(), make_stop()
    res = await run_worker_horizon(spawn=spawn, tier="fast", prompt="Q",
                                   cfg=cfg(samples=3), critic_provider=None,
                                   send=send, stop=stop)
    # idx 0 failed; survivors are slug-1, slug-2.
    assert set(stop.stopped) == {"slug-1", "slug-2"}
    assert res["ok"]


async def test_single_sample_still_returns_an_answer():
    spawn, send, stop = make_spawn(), make_send(), make_stop()
    res = await run_worker_horizon(spawn=spawn, tier="fast", prompt="Q",
                                   cfg=cfg(samples=1), critic_provider=None,
                                   send=send, stop=stop)
    assert len(spawn.calls) == 1
    assert res["ok"] and "42" in res["output"]


async def test_budget_is_honored_and_stops_cleanly():
    """A tight compute budget stops the run; cost stays bounded, answer preserved."""
    spawn, send, stop = make_spawn(), make_send(), make_stop()
    res = await run_worker_horizon(
        spawn=spawn, tier="fast", prompt="Q",
        cfg=cfg(samples=3, compute_budget=2.0), critic_provider=None,
        send=send, stop=stop)
    assert res["cost_spent"] <= 2.0
    assert res["stopped_reason"] == "budget"
    assert set(stop.stopped) == {"slug-0", "slug-1", "slug-2"}  # still disposed


async def test_callbacks_stream_with_sample_tags():
    """on_action carries the rollout index; on_event fires per ladder attempt."""
    spawn, send, stop = make_spawn(), make_send(emit_action=True), make_stop()
    actions: list[dict] = []
    events: list[dict] = []
    res = await run_worker_horizon(
        spawn=spawn, tier="fast", prompt="Q", cfg=cfg(),
        critic_provider=None, send=send, stop=stop,
        on_action=actions.append, on_event=events.append)
    assert actions and all("sample" in a for a in actions)
    assert {a["sample"] for a in actions} <= {0, 1, 2}
    assert events and any(e.get("kind") == "vote" for e in events)
    assert res["actions"]  # accumulated on the result too


# ── Closer (Lever B) ─────────────────────────────────────────────────

class SeqProvider:
    """Returns canned replies in order, one per ``complete`` call (deterministic)."""

    def __init__(self, replies: list[str]):
        self.replies = replies
        self.i = 0

    async def complete(self, messages, tools=None, temperature=None, max_tokens=None):
        r = self.replies[min(self.i, len(self.replies) - 1)]
        self.i += 1
        return LLMResponse(content=r, finish_reason="stop")


async def test_closer_keeps_answer_when_critics_hold():
    critic = StubProvider("SOUND: it holds")
    revise = StubProvider("REVISED")
    res = await run_horizon_closer(
        question="Q", answer="original answer", critic_provider=critic,
        revise_provider=revise, critics=["phrygian", "aeolian", "locrian"])
    assert res["revised"] is False
    assert res["answer"] == "original answer"
    assert res["survived"] == 3 and res["total"] == 3
    assert revise.calls == 0          # no revision when the panel holds


async def test_closer_revises_when_majority_refute():
    critic = StubProvider("REFUTED: wrong premise")
    revise = StubProvider("the corrected answer")
    events: list[dict] = []
    res = await run_horizon_closer(
        question="Q", answer="bad answer", critic_provider=critic,
        revise_provider=revise, critics=["phrygian", "aeolian", "locrian"],
        on_event=events.append)
    assert res["revised"] is True
    assert res["answer"] == "the corrected answer"
    assert res["survived"] == 0 and res["total"] == 3
    assert "wrong premise" in res["feedback"]
    assert revise.calls == 1
    stages = [e["stage"] for e in events]
    # Incremental: a start, one event per critic, the verdict, then the revise.
    assert stages[0] == "verify_start"
    assert stages.count("critic") == 3
    assert stages[-2:] == ["verify", "revise"]


async def test_closer_holds_on_minority_refutation():
    """1 of 3 refuting is not a majority — the answer stands, no revision."""
    critic = SeqProvider(["REFUTED: nit", "SOUND: ok", "SOUND: ok"])
    revise = StubProvider("REVISED")
    res = await run_horizon_closer(
        question="Q", answer="solid answer", critic_provider=critic,
        revise_provider=revise, critics=["phrygian", "aeolian", "locrian"])
    assert res["revised"] is False
    assert res["answer"] == "solid answer"
    assert res["survived"] == 2 and revise.calls == 0


async def test_closer_rejects_a_collapsed_revision():
    """A revision that collapses a substantial answer to a fragment (e.g. a reasoning
    model returning only a reasoning tail) must be rejected — keep the original."""
    critic = StubProvider("REFUTED: weak")
    revise = StubProvider("tiny fragment")          # far shorter than the original
    long_answer = "Detailed evaluation point. " * 60  # > 800 chars
    events: list[dict] = []
    res = await run_horizon_closer(
        question="Q", answer=long_answer, critic_provider=critic, revise_provider=revise,
        critics=["phrygian", "aeolian", "locrian"], on_event=events.append)
    assert res["revised"] is False
    assert res["answer"] == long_answer            # original preserved, not nuked
    assert any(e["stage"] == "revise_rejected" for e in events)


async def test_closer_rejects_an_empty_revision():
    critic = StubProvider("REFUTED: weak")
    revise = StubProvider("   ")                     # whitespace-only
    res = await run_horizon_closer(
        question="Q", answer="a concrete answer", critic_provider=critic,
        revise_provider=revise, critics=["phrygian", "aeolian", "locrian"])
    assert res["revised"] is False and res["answer"] == "a concrete answer"


async def test_heartbeat_emits_while_a_slow_call_runs():
    """The closer must show it's alive during a slow critic/revise call."""
    events: list[dict] = []

    async def slow():
        await asyncio.sleep(0.05)
        return "done"

    r = await _with_heartbeat(slow(), on_event=events.append, phase="verify", interval=0.01)
    assert r == "done"
    beats = [e for e in events if e["stage"] == "heartbeat"]
    assert len(beats) >= 2 and all(e["phase"] == "verify" for e in beats)


async def test_heartbeat_is_a_noop_without_on_event():
    r = await _with_heartbeat(asyncio.sleep(0, result="x"), on_event=None, phase="verify")
    assert r == "x"


async def test_closer_is_noop_without_a_critic_provider():
    res = await run_horizon_closer(question="Q", answer="x", critic_provider=None)
    assert res == {"answer": "x", "revised": False, "survived": 0, "total": 0, "feedback": ""}


async def test_closer_is_noop_on_empty_answer():
    critic = StubProvider("REFUTED: n/a")
    res = await run_horizon_closer(question="Q", answer="   ", critic_provider=critic)
    assert res["revised"] is False and critic.calls == 0
