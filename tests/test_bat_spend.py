"""Bat Phase 6 — capped real-money spend with pre-authorization.

Covers the cap decision, the persisted ledger (survives restart), the authorize/
settle endpoints (auto-approve under the per-item limit, owner-approve above it,
deny over the run cap or when disabled), idempotent re-declare, and the spend-
approval resume. The Playwright checkout interception is deck-verified; its
pre-click heuristic is tested here.
"""

from __future__ import annotations

import uuid

import pytest

from captain_claw.flight_deck import bat_loop, bat_routes, human_ask
from captain_claw.flight_deck.bat_routes import (
    _SpendReq, _spend_policy, looks_like_purchase, spend_authorize, spend_decision,
    spend_settle, spend_status,
)
from captain_claw.flight_deck.bat_store import BatStore


# ── pure cap logic ─────────────────────────────────────────────────────

def test_spend_decision():
    off = {"enabled": False, "per_item_usd": 50, "run_usd_cap": 100}
    assert spend_decision(off, 0, 10)[0] == "denied"             # kill-switch / not granted

    pol = {"enabled": True, "per_item_usd": 20, "run_usd_cap": 100}
    assert spend_decision(pol, 0, 0)[0] == "denied"              # non-positive
    assert spend_decision(pol, 0, 15)[0] == "approved"           # at/below per-item
    assert spend_decision(pol, 0, 20)[0] == "approved"           # exactly per-item
    assert spend_decision(pol, 0, 50)[0] == "needs_human"        # above per-item, within cap
    assert spend_decision(pol, 60, 50)[0] == "denied"            # 60+50 > 100 cap
    assert spend_decision(pol, 80, 20)[0] == "approved"          # 80+20 == 100, fits


def test_spend_policy_requires_switch_grant_and_cap(monkeypatch):
    run = {"config": {"spend_allowed": True, "real_usd_cap": 100, "per_item_usd": 10}}
    monkeypatch.delenv("FD_BAT_SPEND", raising=False)
    assert _spend_policy(run)["enabled"] is False                # deck switch off
    monkeypatch.setenv("FD_BAT_SPEND", "1")
    assert _spend_policy(run)["enabled"] is True
    assert _spend_policy({"config": {"real_usd_cap": 100}})["enabled"] is False       # not granted
    assert _spend_policy({"config": {"spend_allowed": True}})["enabled"] is False     # no cap


def test_looks_like_purchase():
    for t in ("Place order", "Pay now", "Complete purchase", "Subscribe", "Start free trial", "Checkout"):
        assert looks_like_purchase(t), t
    for t in ("Add to cart", "Next", "Learn more", "Sign in"):
        assert not looks_like_purchase(t), t


# ── ledger (persisted, restart-safe) ───────────────────────────────────

@pytest.fixture
async def store(tmp_path):
    s = BatStore(tmp_path / "bat.db")
    await s.init()
    yield s
    await s.close()


async def test_committed_counts_requested_approved_consumed_not_denied(store):
    await store.create_run(run_id="r1", owner_id="u1", title="t", task="x")
    await store.create_spend(spend_id="s1", run_id="r1", owner_id="u1", amount_usd_max=10, status="approved")
    await store.create_spend(spend_id="s2", run_id="r1", owner_id="u1", amount_usd_max=20, status="requested")
    await store.create_spend(spend_id="s3", run_id="r1", owner_id="u1", amount_usd_max=5, status="denied")
    await store.create_spend(spend_id="s4", run_id="r1", owner_id="u1", amount_usd_max=99, status="consumed")
    await store.set_spend_status("s4", "consumed", actual_usd=7)  # actual wins over ceiling
    assert await store.committed_usd("r1") == pytest.approx(10 + 20 + 7)  # denied excluded


async def test_find_live_spend_matches_merchant_amount(store):
    await store.create_run(run_id="r1", owner_id="u1", title="t", task="x")
    await store.create_spend(spend_id="s1", run_id="r1", owner_id="u1",
                             merchant_domain="x.com", amount_usd_max=30, status="approved")
    hit = await store.find_live_spend("r1", "x.com", 30.0)
    assert hit and hit["id"] == "s1"
    assert await store.find_live_spend("r1", "x.com", 31.0) is None  # amount differs


# ── endpoints ──────────────────────────────────────────────────────────

@pytest.fixture
async def wired(store, monkeypatch):
    """Spend enabled, owner resolution stubbed, a granted run with a $100 cap /
    $20 per-item. Returns the run id."""
    monkeypatch.setenv("FD_BAT_SPEND", "1")
    bat_loop.set_store(store)
    monkeypatch.setattr(bat_routes, "_resolve", lambda body: "u1")

    async def rec(ask):
        pass
    saved = human_ask._NOTIFY
    human_ask.set_notifier(rec)
    rid = f"bat_{uuid.uuid4().hex[:8]}"
    await store.create_run(run_id=rid, owner_id="u1", title="t", task="buy stuff",
                           config={"spend_allowed": True, "real_usd_cap": 100, "per_item_usd": 20},
                           status="running")
    yield rid
    human_ask.set_notifier(saved)


def _req(rid, **kw):
    return _SpendReq(owner_id="u1", session_id=rid, **kw)


async def test_authorize_auto_approves_under_limit(store, wired):
    rid = wired
    d = await spend_authorize(_req(rid, merchant="Acme", merchant_domain="acme.com",
                                   amount_usd=10, description="a domain"))
    assert d["status"] == "approved"
    assert await store.committed_usd(rid) == pytest.approx(10)


async def test_authorize_over_limit_needs_human_and_pauses_run(store, wired):
    rid = wired
    d = await spend_authorize(_req(rid, merchant="Big", merchant_domain="big.com", amount_usd=50))
    assert d["status"] == "requested"
    assert (await store.get_run(rid))["status"] == "awaiting_human"
    ask = await store.latest_open_ask(rid)
    assert ask and ask["kind"] == "spend_approval" and ask["step_key"] == f"spend:{d['id']}"
    assert await store.committed_usd(rid) == pytest.approx(50)  # reserved while pending


async def test_authorize_over_cap_is_denied(store, wired):
    rid = wired
    await spend_authorize(_req(rid, merchant_domain="a.com", amount_usd=20))   # approved, 20 committed
    d = await spend_authorize(_req(rid, merchant_domain="b.com", amount_usd=90))  # 20+90 > 100
    assert d["status"] == "denied" and "cap" in d["reason"]


async def test_authorize_is_idempotent_per_merchant_amount(store, wired):
    rid = wired
    d1 = await spend_authorize(_req(rid, merchant_domain="x.com", amount_usd=10))
    d2 = await spend_authorize(_req(rid, merchant_domain="x.com", amount_usd=10))
    assert d1["id"] == d2["id"] and d2["status"] == "approved"
    assert await store.committed_usd(rid) == pytest.approx(10)  # not double-counted


async def test_denied_when_switch_off(store, wired, monkeypatch):
    monkeypatch.delenv("FD_BAT_SPEND", raising=False)
    d = await spend_authorize(_req(wired, merchant_domain="x.com", amount_usd=5))
    assert d["status"] == "denied"


async def test_settle_marks_consumed_with_actual(store, wired):
    rid = wired
    d = await spend_authorize(_req(rid, merchant_domain="x.com", amount_usd=20))
    res = await spend_settle(_req(rid, spend_id=d["id"], actual_usd=18.5, order_ref="ORD-1"))
    assert res["ok"]
    row = await store.get_spend(d["id"])
    assert row["status"] == "consumed" and row["actual_usd"] == pytest.approx(18.5)
    assert await store.committed_usd(rid) == pytest.approx(18.5)  # actual, not the ceiling


async def test_settle_refuses_unapproved(store, wired):
    rid = wired
    d = await spend_authorize(_req(rid, merchant_domain="big.com", amount_usd=50))  # requested, not approved
    res = await spend_settle(_req(rid, spend_id=d["id"], actual_usd=50))
    assert not res["ok"]


async def test_spend_status_reports_budget(store, wired):
    rid = wired
    await spend_authorize(_req(rid, merchant_domain="x.com", amount_usd=10))
    st = await spend_status(_req(rid))
    assert st["enabled"] and st["run_usd_cap"] == 100 and st["committed_usd"] == pytest.approx(10)


# ── spend-approval resume (gate_check) ─────────────────────────────────

async def test_spend_approval_resume_sets_row(store, wired):
    rid = wired
    d = await spend_authorize(_req(rid, merchant_domain="big.com", amount_usd=50))
    ask = await store.latest_open_ask(rid)
    # owner approves the spend ask; gate_check resume flips the ledger row
    await human_ask.answer(store, ask["id"], "approve")
    run = await store.get_run(rid)
    decision = await bat_routes._gate_check(store, run)
    assert decision == "proceed"
    assert (await store.get_spend(d["id"]))["status"] == "approved"


async def test_spend_approval_decline_denies_row(store, wired):
    rid = wired
    d = await spend_authorize(_req(rid, merchant_domain="big.com", amount_usd=50))
    ask = await store.latest_open_ask(rid)
    await human_ask.answer(store, ask["id"], "cancel")
    await bat_routes._gate_check(store, await store.get_run(rid))
    assert (await store.get_spend(d["id"]))["status"] == "denied"


# ── the worker-side tool ───────────────────────────────────────────────

async def test_spend_tool_refuses_outside_a_bat_run(monkeypatch):
    from captain_claw.tools.bat_spend import SpendTool
    monkeypatch.delenv("CLAW_BAT_SESSION", raising=False)
    res = await SpendTool().execute(action="authorize", amount_usd=5)
    assert not res.success and "only available inside a Bat run" in res.error

