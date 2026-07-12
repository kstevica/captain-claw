"""Iskra genome — the DNA of a living being (docs/living-beings-plan.md §2.1).

A genome's ``attributes`` block is the single source of truth: seven RPG-style
attributes (1–10), allocated from a 40-point pool at conception (Generation 1
point-buy) or inherited via crossover/budding. Everything behavioral — drive
weights, wallet reserve, goal hysteresis — is *derived deterministically* from
the attributes here, so the character sheet and the behavior can never drift
apart (design rule #1: no theater).

Metamorphosis (§2.1.2) is the paid self-change rite: zero-sum (move one point
between attributes), priced quadratically on the target value and doubling per
lifetime move, gated by a 30-day cooldown. The *payment* (wallet burn) is the
store's job; this module only prices, validates, and applies the genome edit.

Everything here is pure — no I/O, no clock reads (callers pass ``now``) — so
it is fully unit-testable and safe to reuse from workers.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta

SPECIES = "iskra"

ATTRS = ("CUR", "PER", "CAU", "SOC", "CRE", "ORD", "PLA")
ATTR_NAMES = {
    "CUR": "Curiosity",
    "PER": "Persistence",
    "CAU": "Caution",
    "SOC": "Sociability",
    "CRE": "Creativity",
    "ORD": "Order",
    "PLA": "Playfulness",
}

POOL = 40
ATTR_MIN = 1
ATTR_MAX = 10
# Lineage total band: offspring totals are clamped here so attribute inflation
# can't creep in across generations (tradeoffs stay real).
BAND_MIN = 36
BAND_MAX = 44

# All presets must sum to exactly POOL (enforced by tests).
PRESETS: dict[str, dict[str, int]] = {
    "explorer":  {"CUR": 9, "PER": 6, "CAU": 2, "SOC": 5, "CRE": 7, "ORD": 4, "PLA": 7},
    "scholar":   {"CUR": 8, "PER": 8, "CAU": 6, "SOC": 3, "CRE": 5, "ORD": 8, "PLA": 2},
    "artist":    {"CUR": 7, "PER": 5, "CAU": 2, "SOC": 4, "CRE": 9, "ORD": 3, "PLA": 10},
    "caretaker": {"CUR": 4, "PER": 7, "CAU": 7, "SOC": 9, "CRE": 3, "ORD": 8, "PLA": 2},
}

METAMORPHOSIS_BASE_TOKENS = 1_000_000
METAMORPHOSIS_COOLDOWN_DAYS = 30

CROSSOVER_MUTATION_CHANCE = 0.15   # per-attribute ±1..2 after parent mix
BUDDING_MUTATION_CHANCE = 0.25     # per-attribute ±1 on a single-parent copy


# ── Point-buy (Generation 1) ─────────────────────────────────────────────

def validate_point_buy(attributes: dict) -> list[str]:
    """Errors for a conception sheet; empty list = valid (sum == POOL, 1..10)."""
    errors: list[str] = []
    if not isinstance(attributes, dict):
        return ["attributes must be an object"]
    for a in ATTRS:
        if a not in attributes:
            errors.append(f"missing attribute {a}")
    for k in attributes:
        if k not in ATTRS:
            errors.append(f"unknown attribute {k}")
    if errors:
        return errors
    for a in ATTRS:
        v = attributes[a]
        if not isinstance(v, int) or isinstance(v, bool):
            errors.append(f"{a} must be an integer")
        elif not (ATTR_MIN <= v <= ATTR_MAX):
            errors.append(f"{a} must be between {ATTR_MIN} and {ATTR_MAX}")
    if errors:
        return errors
    total = sum(attributes[a] for a in ATTRS)
    if total != POOL:
        errors.append(f"points must total exactly {POOL} (got {total})")
    return errors


def roll(rng: random.Random | None = None) -> dict[str, int]:
    """Random split of the POOL (min 1, max 10 each) — let nature decide."""
    rng = rng or random.Random()
    attrs = {a: ATTR_MIN for a in ATTRS}
    remaining = POOL - ATTR_MIN * len(ATTRS)
    while remaining > 0:
        open_attrs = [a for a in ATTRS if attrs[a] < ATTR_MAX]
        attrs[rng.choice(open_attrs)] += 1
        remaining -= 1
    return attrs


# ── Derivation: attributes → behavior parameters ─────────────────────────

def effective_attributes(genome: dict) -> dict[str, int]:
    """Genome attributes + epigenetic overlay (coming-of-age +1), clamped."""
    attrs = dict(genome.get("attributes") or {})
    for k, delta in (genome.get("epigenetics") or {}).items():
        if k in attrs:
            attrs[k] = max(ATTR_MIN, min(ATTR_MAX, int(attrs[k]) + int(delta)))
    return {a: int(attrs.get(a, ATTR_MIN)) for a in ATTRS}


def derive(attributes: dict[str, int]) -> dict:
    """Behavior parameters from the sheet (docs §2.1.1 mechanism column)."""
    cur, per, cau = attributes["CUR"], attributes["PER"], attributes["CAU"]
    soc, cre, ord_, pla = (attributes["SOC"], attributes["CRE"],
                           attributes["ORD"], attributes["PLA"])
    return {
        "drive_weights": {
            "survive": 1.0,
            "grow": round(0.40 + 0.04 * per, 3),
            "explore": round(0.30 + 0.07 * cur, 3),
            "connect": round(0.20 + 0.06 * soc, 3),
            "create": round(0.20 + 0.06 * cre, 3),
        },
        "reserve_fraction": round(0.05 + 0.02 * cau, 3),
        "goal_hysteresis_ticks": 2 + per,
        "routine_strength": round(ord_ / 10, 2),
        "whimsy": round(pla / 10, 2),
        "thrift": round((cau + ord_) / 20, 2),
        "industriousness": round((per + cre) / 20, 2),
        "generosity": round((soc + pla) / 20, 2),
        "risk_appetite": round((cur - cau + 10) / 20, 2),
    }


# ── Genome factory ───────────────────────────────────────────────────────

def new_genome(
    attributes: dict[str, int],
    *,
    voice_seed: str = "",
    interest_seeds: list[str] | None = None,
    generation: int = 1,
    lineage: list[str] | None = None,
    inherited_skills: list[str] | None = None,
) -> dict:
    return {
        "species": SPECIES,
        "generation": generation,
        "lineage": list(lineage or []),
        "attributes": {a: int(attributes[a]) for a in ATTRS},
        "epigenetics": {},
        "metamorphoses": [],
        "voice_seed": voice_seed,
        "interest_seeds": list(interest_seeds or []),
        "inherited_skills": list(inherited_skills or []),
    }


# ── Inheritance (offspring never point-buy) ──────────────────────────────

def _clamp_band(attrs: dict[str, int], rng: random.Random) -> dict[str, int]:
    """Pull an offspring total back into [BAND_MIN, BAND_MAX] with random ±1s."""
    attrs = dict(attrs)
    while sum(attrs.values()) > BAND_MAX:
        candidates = [a for a in ATTRS if attrs[a] > ATTR_MIN]
        attrs[rng.choice(candidates)] -= 1
    while sum(attrs.values()) < BAND_MIN:
        candidates = [a for a in ATTRS if attrs[a] < ATTR_MAX]
        attrs[rng.choice(candidates)] += 1
    return attrs


def crossover(a: dict[str, int], b: dict[str, int],
              rng: random.Random | None = None) -> dict[str, int]:
    """Two-parent mix: rounded average ± occasional mutation, band-clamped."""
    rng = rng or random.Random()
    child: dict[str, int] = {}
    for attr in ATTRS:
        base = round((int(a[attr]) + int(b[attr])) / 2 + rng.uniform(-0.5, 0.5))
        if rng.random() < CROSSOVER_MUTATION_CHANCE:
            base += rng.choice((-2, -1, 1, 2))
        child[attr] = max(ATTR_MIN, min(ATTR_MAX, base))
    return _clamp_band(child, rng)


def budding(parent: dict[str, int], rng: random.Random | None = None) -> dict[str, int]:
    """Single-parent copy with stronger drift and one guaranteed mutation."""
    rng = rng or random.Random()
    child = {a: int(parent[a]) for a in ATTRS}
    mutated = False
    for attr in ATTRS:
        if rng.random() < BUDDING_MUTATION_CHANCE:
            child[attr] = max(ATTR_MIN, min(ATTR_MAX, child[attr] + rng.choice((-1, 1))))
            mutated = True
    if not mutated:
        attr = rng.choice(ATTRS)
        child[attr] = max(ATTR_MIN, min(ATTR_MAX, child[attr] + rng.choice((-1, 1))))
    return _clamp_band(child, rng)


# ── Metamorphosis: buying a new self (§2.1.2) ────────────────────────────

def metamorphosis_price(target_value: int, lifetime_moves: int) -> int:
    """Tokens to move one point so an attribute *reaches* ``target_value``.

    Quadratic in the target, doubling per lifetime move — a life project,
    not a settings toggle.
    """
    return int(target_value) ** 2 * METAMORPHOSIS_BASE_TOKENS * (2 ** int(lifetime_moves))


def plan_metamorphosis(
    genome: dict, from_attr: str, to_attr: str, now: datetime,
) -> dict:
    """Validate + price a zero-sum move. Returns {ok, errors, price, new_attributes}.

    Does not mutate the genome; the caller burns the price from the wallet
    first, then calls :func:`apply_metamorphosis`.
    """
    errors: list[str] = []
    attrs = dict(genome.get("attributes") or {})
    moves = list(genome.get("metamorphoses") or [])
    if from_attr not in ATTRS:
        errors.append(f"unknown attribute {from_attr}")
    if to_attr not in ATTRS:
        errors.append(f"unknown attribute {to_attr}")
    if not errors and from_attr == to_attr:
        errors.append("from and to must differ (zero-sum move)")
    if not errors:
        if int(attrs[from_attr]) - 1 < ATTR_MIN:
            errors.append(f"{from_attr} is already at minimum")
        if int(attrs[to_attr]) + 1 > ATTR_MAX:
            errors.append(f"{to_attr} is already at maximum")
    if moves:
        last = moves[-1].get("at", "")
        try:
            last_at = datetime.fromisoformat(last)
            ready_at = last_at + timedelta(days=METAMORPHOSIS_COOLDOWN_DAYS)
            if now < ready_at:
                errors.append(f"cooldown until {ready_at.isoformat()}")
        except ValueError:
            pass
    if errors:
        return {"ok": False, "errors": errors, "price": None, "new_attributes": None}
    target = int(attrs[to_attr]) + 1
    price = metamorphosis_price(target, len(moves))
    new_attrs = dict(attrs)
    new_attrs[from_attr] = int(new_attrs[from_attr]) - 1
    new_attrs[to_attr] = target
    return {"ok": True, "errors": [], "price": price, "new_attributes": new_attrs}


def apply_metamorphosis(
    genome: dict, from_attr: str, to_attr: str, reason: str,
    price: int, now: datetime,
) -> dict:
    """Return a new genome with the move applied and logged (payment already made)."""
    out = dict(genome)
    plan = plan_metamorphosis(genome, from_attr, to_attr, now)
    if not plan["ok"]:
        raise ValueError("; ".join(plan["errors"]))
    out["attributes"] = plan["new_attributes"]
    out["metamorphoses"] = list(genome.get("metamorphoses") or []) + [{
        "from": from_attr, "to": to_attr, "reason": reason,
        "price_tokens": int(price), "at": now.isoformat(),
    }]
    return out
