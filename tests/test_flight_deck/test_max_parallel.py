"""Max-parallel-agents gate: a per-run semaphore that limits how many agent
dispatches run at once (mainly for memory-capped local models)."""

from __future__ import annotations

import asyncio

from captain_claw.flight_deck import basna_routes as b


# ── the gate factory ─────────────────────────────────────────────────

def test_gate_is_none_when_unlimited_or_larger_than_team():
    assert b._make_gate(0, 5) is None       # 0 = unlimited
    assert b._make_gate(5, 5) is None       # cap == team → nothing to gate
    assert b._make_gate(9, 5) is None       # cap > team
    assert b._make_gate(-3, 5) is None      # nonsense → unlimited


def test_gate_is_a_semaphore_when_capping():
    g = b._make_gate(2, 5)
    assert g is not None and g._value == 2


# ── it actually limits concurrency ───────────────────────────────────

def test_dispatch_respects_the_gate(monkeypatch):
    live = 0
    peak = 0

    async def fake_collect(port, token, prompt, timeout, *, usage_sink=None, error_sink=None, **kw):
        nonlocal live, peak
        live += 1
        peak = max(peak, live)
        await asyncio.sleep(0.02)   # hold the slot so overlap is observable
        live -= 1
        return "done", []

    monkeypatch.setattr(b, "_send_chat_and_collect", fake_collect)

    async def run(cap: int) -> int:
        # Set the gate in THIS task's context; child tasks inherit it.
        b._run_gate.set(b._make_gate(cap, 6))
        await asyncio.gather(*[b._dispatch_one(1, "t", "p", 5.0) for _ in range(6)])
        return peak

    # Capped at 2 → never more than 2 dispatches in flight at once.
    peak = 0
    assert asyncio.run(run(2)) == 2


def test_unlimited_gate_lets_all_run_at_once(monkeypatch):
    live = 0
    peak = 0

    async def fake_collect(port, token, prompt, timeout, *, usage_sink=None, error_sink=None, **kw):
        nonlocal live, peak
        live += 1
        peak = max(peak, live)
        await asyncio.sleep(0.02)
        live -= 1
        return "done", []

    monkeypatch.setattr(b, "_send_chat_and_collect", fake_collect)

    async def run() -> int:
        b._run_gate.set(None)  # unlimited
        await asyncio.gather(*[b._dispatch_one(1, "t", "p", 5.0) for _ in range(6)])
        return peak

    assert asyncio.run(run()) == 6  # all six overlap
