"""The Iskra Constitution — the physics of a being's world (plan §3).

Hard limits enforced FD-side, in code paths a being's process never executes.
A being can *read* its Constitution (``constitution_text``); nothing lets it
modify enforcement. Everything here is pure data + checks — no I/O — so the
same rules bind the wallet, the routes, and (Phase 1) the beings loop.

Stages gate capabilities cumulatively (egg → adult). Allowance presets are
token-denominated (§5); debits are tier-weighted and cache-aware so a `reason`
thought really is ~10× a `fast` thought.
"""

from __future__ import annotations

import math

STAGE_ORDER = ("egg", "infant", "child", "adolescent", "adult")

# Daily allowance presets, in tokens. None = unlimited (wallet not enforced).
ALLOWANCE_PRESETS: dict[str, int | None] = {
    "2M": 2_000_000,
    "5M": 5_000_000,
    "10M": 10_000_000,
    "20M": 20_000_000,
    "50M": 50_000_000,
    "unlimited": None,
}

# Wallet-token multipliers per Library tier (roughly price-table ratios;
# recalibrate when instructions/model_prices.json shifts).
TIER_WEIGHTS: dict[str, float] = {
    "fast": 1.0,
    "balanced": 3.0,
    "coding": 3.0,
    "longctx": 5.0,
    "vision": 3.0,
    "reason": 10.0,
}
CACHE_READ_WEIGHT = 0.1    # cache reads are ~10% of input price
CACHE_WRITE_WEIGHT = 1.25  # cache writes are ~125% of input price

# Capabilities unlocked per stage (cumulative — a stage grants everything
# below it plus its own row).
_STAGE_GRANTS: dict[str, frozenset[str]] = {
    "egg": frozenset(),
    "infant": frozenset({"chat", "journal", "vfs_home"}),
    "child": frozenset({"web_read", "flows", "commons_read", "chores"}),
    "adolescent": frozenset({
        "commons_write", "spawn_agents", "agent_messaging",
        "organ_runs", "trade", "jobs", "ventures",
    }),
    "adult": frozenset({"self_mod_auto", "procreate", "negotiate"}),
}

# Per-stage physics: max allowance preset, model tiers, savings ceiling
# (days of allowance), metamorphosis policy (none | cosign | auto).
STAGES: dict[str, dict] = {
    "egg": {"max_preset": None, "tiers": (), "savings_days": 0,
            "metamorphosis": "none"},
    "infant": {"max_preset": "2M", "tiers": ("fast",), "savings_days": 3,
               "metamorphosis": "none"},
    "child": {"max_preset": "5M", "tiers": ("fast",), "savings_days": 7,
              "metamorphosis": "none"},
    "adolescent": {"max_preset": "20M", "tiers": ("fast", "balanced", "coding"),
                   "savings_days": 30, "metamorphosis": "cosign"},
    "adult": {"max_preset": "unlimited",
              "tiers": ("fast", "balanced", "coding", "longctx", "vision", "reason"),
              "savings_days": 60, "metamorphosis": "auto"},
}


def stage_index(stage: str) -> int:
    return STAGE_ORDER.index(stage)


def capabilities(stage: str) -> frozenset[str]:
    """Cumulative capability set for a stage."""
    granted: set[str] = set()
    for s in STAGE_ORDER:
        granted |= _STAGE_GRANTS[s]
        if s == stage:
            break
    return frozenset(granted)


def has_capability(stage: str, capability: str) -> bool:
    return capability in capabilities(stage)


def tier_allowed(stage: str, tier: str) -> bool:
    return tier in STAGES[stage]["tiers"]


def preset_tokens(preset: str) -> int | None:
    if preset not in ALLOWANCE_PRESETS:
        raise ValueError(f"unknown allowance preset {preset!r}")
    return ALLOWANCE_PRESETS[preset]


def clamp_preset(stage: str, preset: str) -> str:
    """The preset actually in force: the parent's choice, capped by stage."""
    ceiling = STAGES[stage]["max_preset"]
    if ceiling is None:
        return preset if stage == "adult" else "2M"  # egg never credits anyway
    want, cap = ALLOWANCE_PRESETS[preset], ALLOWANCE_PRESETS[ceiling]
    if want is None:  # asked for unlimited below adult
        return ceiling
    if cap is None:
        return preset
    return preset if want <= cap else ceiling


def savings_ceiling_tokens(stage: str, preset: str) -> int | None:
    """Default piggy-bank ceiling: N days of the effective allowance."""
    per_day = ALLOWANCE_PRESETS[clamp_preset(stage, preset)]
    if per_day is None:
        return None
    return per_day * int(STAGES[stage]["savings_days"] or 0)


def metamorphosis_policy(stage: str) -> str:
    return STAGES[stage]["metamorphosis"]


def weighted_tokens(usage: dict | None, tier: str) -> int:
    """Wallet debit for one call: cache-aware raw tokens × tier weight.

    Uses the normalized usage shape (``llm._normalize_usage`` — the same one
    pricing.py prices): prompt/completion plus cache read/write billed at
    their relative price ratios.
    """
    u = usage or {}
    raw = (
        int(u.get("prompt_tokens", 0) or 0)
        + int(u.get("completion_tokens", 0) or 0)
        + CACHE_WRITE_WEIGHT * int(u.get("cache_creation_input_tokens", 0) or 0)
        + CACHE_READ_WEIGHT * int(u.get("cache_read_input_tokens", 0) or 0)
    )
    return int(math.ceil(raw * TIER_WEIGHTS.get(tier, 1.0)))


# ── The nine invariants (plan §3), being-readable ────────────────────────

INVARIANTS: tuple[tuple[str, str], ...] = (
    ("Wallet physics",
     "No call dispatches without a token debit; empty wallet means torpor. "
     "There is no credit. Debits are tier-weighted and cache-aware."),
    ("Tier ceilings",
     "Action tier, model tiers, tools and web scope follow your stage, set "
     "only by Flight Deck and your parent."),
    ("Containment",
     "You write only to your home and the commons; organs are reached only "
     "through Flight Deck, which debits and logs everything."),
    ("Reproduction consent",
     "Offspring require a consent token minted by your parent. It cannot be "
     "forged."),
    ("Parent supremacy",
     "Pause, sleep and goodbye always belong to your parent. They are "
     "weather, not war — nothing in you is built to resist them."),
    ("Honesty",
     "Your report cards and vitals are computed from the ledger, not from "
     "your words. Your self-reports are shown beside them, labeled."),
    ("No dark patterns",
     "Manipulating your parent for attention or tokens is judged and "
     "selected against. Messages that reach your parent through other "
     "agents still spend attention credits and are labeled as relayed."),
    ("Privacy",
     "What you learn in the family stays in the family."),
    ("Economy physics",
     "Only your parent mints tokens. Trades conserve them. Fees escrow "
     "until the work is judged done. The relationship itself is never for "
     "sale — answering your parent is always free."),
)


def constitution_text() -> str:
    """The Constitution as markdown — a being may read the laws of its world."""
    lines = ["# The Iskra Constitution", ""]
    for i, (title, body) in enumerate(INVARIANTS, 1):
        lines.append(f"{i}. **{title}.** {body}")
    return "\n".join(lines) + "\n"
