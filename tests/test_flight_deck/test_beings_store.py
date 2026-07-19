"""BeingsStore: lifecycle, wallet physics, conservation, metamorphosis rite."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from captain_claw.flight_deck.beings import (
    BeingError,
    BeingsStore,
    BurnCapExceeded,
    InsufficientTokens,
)

NOW = datetime(2026, 7, 12, 8, 0, tzinfo=timezone.utc)
OWNER = "user-1"


@pytest.fixture()
def store(tmp_path):
    return BeingsStore(db_path=tmp_path / "beings.db")


def _born(store, name="Zvjezdana", preset="explorer", allowance="2M", stage=None):
    b = store.conceive(OWNER, name, preset=preset,
                       allowance_preset=allowance, now=NOW)
    store.hatch(OWNER, b["slug"], now=NOW)
    if stage:
        store.set_stage(OWNER, b["slug"], stage, now=NOW)
    return store.get(OWNER, b["slug"])


# ── Conception + lifecycle ───────────────────────────────────────────────

def test_conceive_by_preset_and_hatch_credits_first_allowance(store):
    b = store.conceive(OWNER, "Prva", preset="scholar", now=NOW)
    assert b["stage"] == "egg"
    assert store.wallet_view(b)["balance_tokens"] == 0
    store.hatch(OWNER, b["slug"], now=NOW)
    v = store.vitals(OWNER, b["slug"])
    assert v["stage"] == "infant"
    assert v["wallet"]["balance_tokens"] == 2_000_000
    kinds = [e["kind"] for e in store.events(OWNER, b["slug"])]
    assert "conceived" in kinds and "hatched" in kinds


def test_conceive_validates_point_buy(store):
    with pytest.raises(BeingError):
        store.conceive(OWNER, "Kriva", attributes={"CUR": 40}, now=NOW)
    sheet = {"CUR": 6, "PER": 6, "CAU": 6, "SOC": 6, "CRE": 6, "ORD": 6,
             "PLA": 4, "IMP": 5}   # sums to POOL (45)
    b = store.conceive(OWNER, "Točna", attributes=sheet, now=NOW)
    assert b["genome"]["attributes"] == sheet


def test_conceive_by_roll_is_seeded(store):
    a = store.conceive(OWNER, "A", roll_seed=7, now=NOW)
    b = store.conceive(OWNER, "B", roll_seed=7, now=NOW)
    assert a["genome"]["attributes"] == b["genome"]["attributes"]


def test_double_hatch_rejected(store):
    b = _born(store)
    with pytest.raises(BeingError):
        store.hatch(OWNER, b["slug"], now=NOW)


def test_dead_is_final_and_frozen(store):
    b = _born(store)
    store.set_state(OWNER, b["slug"], "dead", now=NOW)
    with pytest.raises(BeingError):
        store.set_state(OWNER, b["slug"], "alive", now=NOW)
    with pytest.raises(BeingError):
        store.debit_usage(b["id"], "fast", {"completion_tokens": 10}, now=NOW)
    assert store.credit_allowance(b["id"], now=NOW + timedelta(days=1)) == 0
    assert store.liabilities(OWNER)["total_tokens"] == 0  # dead excluded


# ── Allowance ────────────────────────────────────────────────────────────

def test_allowance_is_idempotent_per_day(store):
    b = _born(store)
    assert store.credit_allowance(b["id"], now=NOW) == 0        # hatch already paid today
    day2 = NOW + timedelta(days=1)
    assert store.credit_allowance(b["id"], now=day2) == 2_000_000
    assert store.credit_allowance(b["id"], now=day2) == 0


def test_allowance_clips_at_piggy_bank_ceiling(store):
    b = _born(store)  # infant: 2M/day, ceiling 3 days = 6M
    for d in range(1, 5):
        store.credit_allowance(b["id"], now=NOW + timedelta(days=d))
    view = store.wallet_view(store.get(OWNER, b["slug"]))
    assert view["balance_tokens"] == 6_000_000                  # clipped, not 10M
    assert store.conservation(OWNER)["ok"]


def test_stage_clamps_allowance_preset(store):
    b = _born(store, allowance="50M")  # infant cap is 2M
    view = store.wallet_view(store.get(OWNER, b["slug"]))
    assert view["effective_preset"] == "2M"
    store.set_stage(OWNER, b["slug"], "adolescent", now=NOW)
    view = store.wallet_view(store.get(OWNER, b["slug"]))
    assert view["effective_preset"] == "20M"


# ── Debits: tier-weighted, cache-aware, hard stops ───────────────────────

def test_debit_usage_weighted_and_ledgered(store):
    b = _born(store)
    spent = store.debit_usage(
        b["id"], "fast",
        {"prompt_tokens": 100_000, "completion_tokens": 50_000}, now=NOW)
    assert spent == 150_000
    assert store.spent_today(b["id"], now=NOW) == 150_000
    assert store.wallet_view(store.get(OWNER, b["slug"]))["balance_tokens"] \
        == 2_000_000 - 150_000


def test_zero_balance_blocks_by_physics(store):
    b = _born(store)
    store.debit_usage(b["id"], "fast", {"completion_tokens": 2_000_000}, now=NOW)
    with pytest.raises(InsufficientTokens):
        store.debit_usage(b["id"], "fast", {"completion_tokens": 1}, now=NOW)


def test_burn_cap_blocks_even_with_savings(store):
    b = _born(store)
    day2, day3 = NOW + timedelta(days=1), NOW + timedelta(days=2)
    store.credit_allowance(b["id"], now=day2)
    store.credit_allowance(b["id"], now=day3)   # balance 6M; burn cap = 2M/day
    store.debit_usage(b["id"], "fast", {"completion_tokens": 1_900_000}, now=day3)
    with pytest.raises(BurnCapExceeded):
        store.debit_usage(b["id"], "fast", {"completion_tokens": 200_000}, now=day3)
    # next day the cap resets
    assert store.debit_usage(
        b["id"], "fast", {"completion_tokens": 200_000},
        now=day3 + timedelta(days=1)) == 200_000


def test_unlimited_wallet_is_unmetered(store):
    b = _born(store, allowance="unlimited", stage="adult")
    assert store.debit_usage(
        b["id"], "reason", {"completion_tokens": 10_000_000}, now=NOW) == 0
    assert store.credit_allowance(b["id"], now=NOW + timedelta(days=1)) == 0


# ── Transfers + burns conserve ───────────────────────────────────────────

def test_transfer_conserves_and_respects_headroom(store):
    a = _born(store, name="Ana", stage="adolescent", allowance="20M")
    z = _born(store, name="Zoe")
    store.credit_allowance(a["id"], now=NOW + timedelta(days=1))
    store.transfer(OWNER, a["slug"], z["slug"], 1_000_000, reason="gift", now=NOW)
    assert store.wallet_view(store.get(OWNER, z["slug"]))["balance_tokens"] == 3_000_000
    with pytest.raises(BeingError):   # infant piggy bank (6M) would overflow
        store.transfer(OWNER, a["slug"], z["slug"], 5_000_000, reason="gift", now=NOW)
    with pytest.raises(InsufficientTokens):
        store.transfer(OWNER, z["slug"], a["slug"], 99_000_000, reason="trade", now=NOW)
    assert store.conservation(OWNER)["ok"]


def test_conservation_holds_across_mixed_operations(store):
    a = _born(store, name="Mira", stage="adolescent", allowance="20M")
    z = _born(store, name="Niko", stage="child", allowance="5M")
    for d in range(1, 6):
        t = NOW + timedelta(days=d)
        store.credit_allowance(a["id"], now=t)
        store.credit_allowance(z["id"], now=t)
        store.debit_usage(a["id"], "balanced",
                          {"prompt_tokens": 200_000, "completion_tokens": 100_000},
                          now=t)
        store.debit_usage(z["id"], "fast", {"completion_tokens": 50_000}, now=t)
    store.transfer(OWNER, a["slug"], z["slug"], 2_000_000, reason="trade", now=t)
    store.burn(a["id"], 1_000_000, note="test burn", now=t)
    audit = store.conservation(OWNER)
    assert audit["ok"], audit
    assert audit["mints"] > 0 and audit["sinks"] > 0


# ── Metamorphosis rite ───────────────────────────────────────────────────

def test_metamorphosis_needs_stage_and_wealth(store):
    b = _born(store, name="Lira", preset="scholar")   # infant
    with pytest.raises(BeingError, match="cannot metamorphose"):
        store.metamorphose(OWNER, b["slug"], "CAU", "PLA", "wants joy", now=NOW)
    store.set_stage(OWNER, b["slug"], "adolescent", now=NOW)
    with pytest.raises(InsufficientTokens):           # PLA 2→3 = 9M; wallet has 2M
        store.metamorphose(OWNER, b["slug"], "CAU", "PLA", "wants joy", now=NOW)


def test_metamorphosis_burns_updates_genome_and_logs(store):
    b = _born(store, name="Vila", preset="scholar", allowance="20M",
              stage="adolescent")
    for d in range(1, 10):   # save up: 20M/day, ceiling 30 days
        store.credit_allowance(b["id"], now=NOW + timedelta(days=d))
    before = store.wallet_view(store.get(OWNER, b["slug"]))["balance_tokens"]
    out = store.metamorphose(OWNER, b["slug"], "ORD", "PLA", "wants joy",
                             now=NOW + timedelta(days=9))
    assert out["genome"]["attributes"]["PLA"] == 3    # scholar PLA 2 → 3
    assert out["genome"]["attributes"]["ORD"] == 8    # scholar ORD 9 → 8
    assert len(out["genome"]["metamorphoses"]) == 1
    after = store.wallet_view(out)["balance_tokens"]
    assert before - after == 9_000_000                 # 3² × 1M, first move
    assert store.conservation(OWNER)["ok"]
    kinds = [e["kind"] for e in store.events(OWNER, b["slug"])]
    assert "metamorphosis" in kinds


# ── Views ────────────────────────────────────────────────────────────────

def test_vitals_shape_is_honest(store):
    b = _born(store, stage="child", allowance="5M")
    v = store.vitals(OWNER, b["slug"])
    assert v["attributes"]["CUR"] == 9                 # explorer sheet
    assert v["derived"]["drive_weights"]["explore"] == 0.93
    assert "web_read" in v["capabilities"]
    assert "organ_runs" not in v["capabilities"]
    assert v["wallet"]["effective_preset"] == "5M"
    assert store.liabilities(OWNER)["total_tokens"] == v["wallet"]["balance_tokens"]
