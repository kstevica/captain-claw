"""Iskra genome: point-buy, derivation, inheritance, metamorphosis pricing."""

from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone

from captain_claw.flight_deck import being_genome as g

NOW = datetime(2026, 7, 12, 8, 0, tzinfo=timezone.utc)


def _sheet(**over):
    base = {"CUR": 6, "PER": 6, "CAU": 6, "SOC": 6, "CRE": 6, "ORD": 6,
            "PLA": 4, "IMP": 5}  # sums to POOL (45)
    base.update(over)
    return base


# ── Point-buy ────────────────────────────────────────────────────────────

def test_valid_sheet_passes():
    assert g.validate_point_buy(_sheet()) == []


def test_pool_must_be_exact():
    errs = g.validate_point_buy(_sheet(PLA=5))  # 46 points
    assert any("exactly 45" in e for e in errs)


def test_range_and_type_enforced():
    assert any("between" in e for e in g.validate_point_buy(_sheet(CUR=11, PLA=3)))
    assert any("between" in e for e in g.validate_point_buy(_sheet(CUR=0, PLA=10)))
    assert any("integer" in e for e in g.validate_point_buy(_sheet(CUR=6.0)))
    assert any("missing" in e for e in g.validate_point_buy({"CUR": 40}))


def test_all_presets_sum_to_pool_and_validate():
    for name, sheet in g.PRESETS.items():
        assert sum(sheet.values()) == g.POOL, name
        assert g.validate_point_buy(sheet) == [], name


def test_roll_respects_constraints_and_seed():
    a = g.roll(random.Random(7))
    b = g.roll(random.Random(7))
    assert a == b                       # seeded determinism
    assert sum(a.values()) == g.POOL
    assert all(g.ATTR_MIN <= v <= g.ATTR_MAX for v in a.values())


# ── Derivation ───────────────────────────────────────────────────────────

def test_derive_wires_attributes_to_mechanisms():
    d = g.derive(_sheet(CUR=10, CAU=1, PER=8))
    assert d["drive_weights"]["explore"] == 1.0        # 0.30 + 0.07*10
    assert d["reserve_fraction"] == 0.07               # 0.05 + 0.02*1
    assert d["goal_hysteresis_ticks"] == 10            # 2 + 8
    assert d["risk_appetite"] == 0.95                  # (10-1+10)/20
    timid = g.derive(_sheet(CUR=1, CAU=10, PLA=5, PER=5))
    assert timid["risk_appetite"] < d["risk_appetite"]
    assert timid["reserve_fraction"] > d["reserve_fraction"]


def test_epigenetics_overlay_applies_clamped():
    genome = g.new_genome(_sheet(CUR=10, PLA=4))
    genome["epigenetics"] = {"CUR": 1, "PLA": 1}
    eff = g.effective_attributes(genome)
    assert eff["CUR"] == 10   # clamped at max
    assert eff["PLA"] == 5


# ── Inheritance ──────────────────────────────────────────────────────────

def test_crossover_stays_in_band_and_range():
    rng = random.Random(42)
    a, b = g.PRESETS["explorer"], g.PRESETS["caretaker"]
    for _ in range(50):
        child = g.crossover(a, b, rng)
        assert g.BAND_MIN <= sum(child.values()) <= g.BAND_MAX
        assert all(g.ATTR_MIN <= v <= g.ATTR_MAX for v in child.values())


def test_budding_always_mutates_and_stays_in_band():
    rng = random.Random(1)
    parent = g.PRESETS["scholar"]
    diffs = 0
    for _ in range(50):
        child = g.budding(parent, rng)
        assert g.BAND_MIN <= sum(child.values()) <= g.BAND_MAX
        if child != parent:
            diffs += 1
    assert diffs == 50  # one mutation is guaranteed per budding


# ── Metamorphosis ────────────────────────────────────────────────────────

def test_price_is_quadratic_and_doubles_per_lifetime_move():
    assert g.metamorphosis_price(5, 0) == 25_000_000
    assert g.metamorphosis_price(8, 1) == 128_000_000
    assert g.metamorphosis_price(10, 0) == 100_000_000


def test_plan_zero_sum_move():
    genome = g.new_genome(_sheet())
    plan = g.plan_metamorphosis(genome, "CAU", "PLA", NOW)
    assert plan["ok"]
    assert plan["price"] == 25_000_000   # PLA reaches 5, first move
    assert plan["new_attributes"]["CAU"] == 5
    assert plan["new_attributes"]["PLA"] == 5
    assert sum(plan["new_attributes"].values()) == g.POOL


def test_plan_rejects_bounds_and_same_attr():
    genome = g.new_genome(_sheet(CAU=1, CUR=10, PLA=5))
    assert not g.plan_metamorphosis(genome, "CAU", "PLA", NOW)["ok"]   # CAU at min
    assert not g.plan_metamorphosis(genome, "PLA", "CUR", NOW)["ok"]   # CUR at max
    assert not g.plan_metamorphosis(genome, "PLA", "PLA", NOW)["ok"]


def test_cooldown_blocks_then_releases():
    genome = g.new_genome(_sheet())
    genome = g.apply_metamorphosis(genome, "CAU", "PLA", "braver", 25_000_000, NOW)
    soon = NOW + timedelta(days=10)
    blocked = g.plan_metamorphosis(genome, "ORD", "CUR", soon)
    assert not blocked["ok"] and any("cooldown" in e for e in blocked["errors"])
    later = NOW + timedelta(days=31)
    released = g.plan_metamorphosis(genome, "ORD", "CUR", later)
    assert released["ok"]
    # second lifetime move doubles: CUR reaches 7 → 49M × 2
    assert released["price"] == 98_000_000


def test_apply_logs_the_move():
    genome = g.new_genome(_sheet())
    out = g.apply_metamorphosis(genome, "CAU", "PLA", "wants joy", 25_000_000, NOW)
    assert out["attributes"]["PLA"] == 5
    assert len(out["metamorphoses"]) == 1
    assert out["metamorphoses"][0]["reason"] == "wants joy"
    assert genome["attributes"]["PLA"] == 4   # input untouched
