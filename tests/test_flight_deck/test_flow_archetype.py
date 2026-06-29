"""Tests for the Flow engine's `agent on archetype:<id>` selector.

The archetype spawn/dispose seams are injected, so these exercise the runner's
ephemeral-agent lifecycle (resolve → spawn-once-per-run → reuse → dispose) without
spawning real processes or touching the server module.
"""

from __future__ import annotations

from captain_claw.flight_deck import flow_dsl
from captain_claw.flight_deck.flow_runner import FlowRunner, _Root


def _root(dry: bool = True) -> _Root:
    return _Root(run_id="r", control=None, dry=dry, budget={"steps_left": 50}, depth_cap=4)


def _runner(**seams) -> FlowRunner:
    return FlowRunner(
        store=None,
        get_agents=lambda: [],
        resolve_auth=lambda p: "",
        fd_self_base="http://localhost:1",
        **seams,
    )


async def _ready_true(*_a, **_k) -> bool:
    """Stub readiness probe — fake-port spawns never actually listen."""
    return True


# ── progress labels + guards ─────────────────────────────────────────

def test_who_strips_archetype_prefix():
    assert FlowRunner._who({"name": "archetype:deep-researcher"}) == "deep-researcher"
    assert FlowRunner._who({"name": "DeepSeek V4"}) == "DeepSeek V4"
    assert FlowRunner._who({}) == "agent"


def test_usage_detail_renders_token_line():
    out = FlowRunner._usage_detail(
        {"last": {"model": "claude-opus", "input_tokens": 1200, "output_tokens": 340},
         "total": {"total_tokens": 1540}})
    assert out == "claude-opus · 1200→340 tok · 1540 total"
    # tolerate the prompt/completion field shape, no total
    assert FlowRunner._usage_detail({"model": "gpt", "prompt_tokens": 10, "completion_tokens": 5}) == "gpt · 10→5 tok"
    assert FlowRunner._usage_detail({}) == ""


async def test_push_progress_skips_whatsapp_and_scheduler():
    posted: list = []

    fr = _runner()
    # If these early-returns failed, _select_agent would run and (with no agents)
    # return None — still no post — so assert via a sentinel get_agents that records.
    fr.get_agents = lambda: (_ for _ in ()).throw(AssertionError("should not resolve origin"))

    # WhatsApp payload → skipped before any agent resolution
    await fr._push_progress({"waid": "385999"}, "thinking", {"text": "x"})
    # Scheduler payload (no origin) → skipped
    await fr._push_progress({"channel": "scheduler"}, "thinking", {"text": "x"})
    assert posted == []  # nothing attempted


# ── selector parsing ─────────────────────────────────────────────────

def test_parse_archetype_selector():
    assert FlowRunner._parse_archetype_selector("archetype:fact-checker") == ("fact-checker", "")
    assert FlowRunner._parse_archetype_selector("archetype:deep-researcher@reason") == ("deep-researcher", "reason")
    assert FlowRunner._parse_archetype_selector("archetype: fact-checker @fast ") == ("fact-checker", "fast")


# ── graceful degradation when no spawn seam is configured ────────────

async def test_archetype_unavailable_without_seams():
    fr = _runner()  # no load/spawn seams
    agent, err = await fr._ensure_archetype_agent("archetype:fact-checker", _root(), {})
    assert agent is None
    assert "not available" in err


async def test_unknown_archetype_id_fails_cleanly():
    async def load(payload, aid):
        return None  # registry has no such id

    async def spawn(arch, tier, tcfg, payload):  # pragma: no cover - must not be called
        raise AssertionError("spawn should not run for an unknown id")

    fr = _runner(load_archetype=load, spawn_archetype=spawn, stop_archetype=lambda s: None)
    agent, err = await fr._ensure_archetype_agent("archetype:nope", _root(), {})
    assert agent is None
    assert "no archetype 'nope'" in err


# ── spawn-once-per-run + reuse + dispose ─────────────────────────────

async def test_spawn_cached_per_run_and_disposed():
    spawned: list[tuple[str, str]] = []
    stopped: list[str] = []

    async def load(payload, aid):
        return {"id": aid, "role": "Checker", "fleet_instructions": "SOP-TEXT", "tier": "fast"}

    async def spawn(arch, tier, tcfg, payload):
        spawned.append((arch["id"], tier))
        return (9000 + len(spawned), "tok", f"slug-{len(spawned)}")

    async def stop(slug):
        stopped.append(slug)

    fr = _runner(load_archetype=load, spawn_archetype=spawn, stop_archetype=stop)
    fr._wait_agent_ready = _ready_true  # fake ports won't listen; skip the probe
    root = _root()

    a1, e1 = await fr._ensure_archetype_agent("archetype:fact-checker", root, {})
    a2, e2 = await fr._ensure_archetype_agent("archetype:fact-checker", root, {})
    assert e1 == "" and e2 == ""
    assert a1 is a2                       # same cached agent, reused
    assert len(spawned) == 1             # spawned exactly once
    assert a1["name"] == "archetype:fact-checker"
    assert a1["fleet_instructions"] == "SOP-TEXT"

    # a different tier is a distinct ephemeral agent
    a3, _ = await fr._ensure_archetype_agent("archetype:fact-checker@reason", root, {})
    assert a3 is not a1 and len(spawned) == 2

    await fr._dispose_archetypes(root)
    assert sorted(stopped) == ["slug-1", "slug-2"]
    assert root.arch_agents == {} and root.arch_slugs == []


# ── end-to-end: run() disposes spawned agents in its finally ──────────

async def test_run_disposes_archetypes_on_completion():
    spawned: list[str] = []
    stopped: list[str] = []

    async def load(payload, aid):
        return {"id": aid, "role": "R", "fleet_instructions": "", "tier": "fast"}

    async def spawn(arch, tier, tcfg, payload):
        spawned.append(arch["id"])
        return (9100, "tok", f"slug-{arch['id']}")

    async def stop(slug):
        stopped.append(slug)

    fr = _runner(load_archetype=load, spawn_archetype=spawn, stop_archetype=stop)
    fr._wait_agent_ready = _ready_true

    # Patch the agent step to exercise the REAL selector resolution but skip HTTP.
    async def fake_agent_step(step, ctx, payload, root):
        agent, err = await fr._resolve_step_agent(str(step.get("on")), root, payload)
        return (agent["name"] if agent else f"ERR:{err}"), (agent or {}).get("name", "")

    fr._run_agent_step = fake_agent_step

    flow = flow_dsl.compile_dsl(
        'flow "Pipe"\n'
        'trigger any always\n'
        'step a:\n'
        '  agent on archetype:fact-checker\n'
        '  prompt: "check {{trigger.text}}"\n'
        'step out:\n'
        '  emit "{{steps.a.output}}"\n'
        'output -> log\n'
    )

    result = await fr.run(flow, {"text": "hi"}, dry=True)
    assert "archetype:fact-checker" in result["output"]
    assert spawned == ["fact-checker"]
    assert stopped == ["slug-fact-checker"]   # disposed in run()'s finally


async def test_spawn_never_ready_fails_and_disposes():
    stopped: list[str] = []

    async def load(payload, aid):
        return {"id": aid, "role": "R", "fleet_instructions": "", "tier": "fast"}

    async def spawn(arch, tier, tcfg, payload):
        return (9200, "tok", "slug-stuck")

    async def stop(slug):
        stopped.append(slug)

    fr = _runner(load_archetype=load, spawn_archetype=spawn, stop_archetype=stop)

    async def never_ready(*_a, **_k):
        return False

    fr._wait_agent_ready = never_ready
    root = _root()

    agent, err = await fr._ensure_archetype_agent("archetype:fact-checker", root, {})
    assert agent is None
    assert "did not become reachable" in err
    # the slug was tracked before the readiness wait, so dispose still cleans it
    assert root.arch_slugs == ["slug-stuck"]
    await fr._dispose_archetypes(root)
    assert stopped == ["slug-stuck"]
