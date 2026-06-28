"""Tests for Dubina run-targets (intent track): agent + archetype adapters.

All spawn/send/stop seams are injected, so these exercise the dispatch framework
without spawning real processes or touching the server module.
"""

from __future__ import annotations

import pytest

from captain_claw.flight_deck import dubina_routes as dr
from captain_claw.flight_deck.dubina_agents import (
    ArchetypeRunner,
    DispatchProvider,
    _prompt_from_messages,
    make_agent_factory,
)
from captain_claw.flight_deck.dubina_store import DubinaStore
from captain_claw.llm import Message

TIERS_MAP = {"t1": {"provider": "openai", "model": "m1"},
             "t2": {"provider": "openai", "model": "m2"}}


@pytest.fixture
async def store(tmp_path):
    s = DubinaStore(tmp_path / "dubina.db")
    await s.init()
    yield s
    await s.close()


# ── DispatchProvider ─────────────────────────────────────────────────

async def test_dispatch_provider_wraps_reply():
    async def dispatch(prompt: str) -> str:
        return f"got: {prompt}"
    p = DispatchProvider(dispatch)
    resp = await p.complete([Message(role="system", content="sys"),
                             Message(role="user", content="hello")])
    assert resp.content == "got: sys\n\nhello"


def test_prompt_from_messages_joins_nonempty():
    msgs = [Message(role="system", content="A"), Message(role="user", content=""),
            Message(role="user", content="B")]
    assert _prompt_from_messages(msgs) == "A\n\nB"


# ── Agent target ─────────────────────────────────────────────────────

async def test_agent_factory_dispatches_to_port():
    seen = {}

    async def send(port, token, prompt, timeout, fleet_instructions="", agent_name="",
                   on_action=None, on_usage=None):
        seen.update(port=port, token=token, prompt=prompt)
        return "agent reply"

    factory = make_agent_factory(9100, "tok", send=send)
    provider = factory("t1")           # tier ignored for a live agent
    resp = await provider.complete([Message(role="user", content="do it")])
    assert resp.content == "agent reply"
    assert seen["port"] == 9100 and seen["token"] == "tok" and "do it" in seen["prompt"]


# ── Archetype target ─────────────────────────────────────────────────

async def test_archetype_runner_spawns_per_tier_and_disposes():
    spawns: list[str] = []
    stops: list[str] = []

    async def spawn(arch, tier, tcfg, request, user, name_suffix=""):
        spawns.append(tier)
        return (9000 + len(spawns), f"tok-{tier}", f"slug-{tier}")

    async def send(port, token, prompt, timeout, fleet_instructions="", agent_name="",
                   on_action=None, on_usage=None):
        return f"{token}:{prompt}"

    async def stop(slug):
        stops.append(slug)

    runner = ArchetypeRunner({"id": "researcher", "role": "Researcher"},
                             request=None, user={"id": "u1"}, tiers_map=TIERS_MAP,
                             spawn=spawn, send=send, stop=stop)
    factory = runner.provider_for_tier()

    # Two dispatches at the same tier → spawn once (cached).
    await factory("t1").complete([Message(role="user", content="q1")])
    await factory("t1").complete([Message(role="user", content="q2")])
    # A new tier → a second spawn.
    r = await factory("t2").complete([Message(role="user", content="q3")])
    assert spawns == ["t1", "t2"]
    assert r.content.startswith("tok-t2:")

    await runner.dispose()
    assert sorted(stops) == ["slug-t1", "slug-t2"]


# ── execute_intent end to end ────────────────────────────────────────

class _StubProvider:
    def __init__(self, content):
        self.content = content

    async def complete(self, messages, tools=None, temperature=None, max_tokens=None):
        from captain_claw.llm import LLMResponse
        return LLMResponse(content=self.content, finish_reason="stop")


async def test_execute_intent_passes_on_agreement_and_disposes(store):
    disposed = {"v": False}

    async def dispose():
        disposed["v"] = True

    req = dr.IntentRequest(
        task="what is 2+2?", target="agent:worker", base_tier="t1", max_tier="t2",
        tiers=["t1", "t2"], critic_modes=[], agreement_threshold=0.6, max_step_samples=3,
    )
    run_id = await store.create_run("intent", "u1", req.task, "t1", "t2", 0.0)
    result = await dr.execute_intent(
        store, run_id, req,
        provider_factory=lambda tier: _StubProvider("reasoning...\nAnswer: 42"),
        critic_provider=None, dispose=dispose, allowed={"t1", "t2"},
    )
    assert result.passed
    assert disposed["v"] is True

    run = await store.get_run("intent", run_id)
    assert run["status"] == "passed"
    assert run["result"]["answer"].endswith("Answer: 42")
