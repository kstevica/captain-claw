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

# Society physics (plan §7): letters are social, not spam. The daily quota
# scales with stage (loops plan §3.3) — a child's reach is smaller than an
# adult's; LETTERS_PER_DAY stays as the fallback for unknown stages.
LETTERS_PER_DAY = 5
# A published skill may ask at most this much (anti-absurdity clamp).
MAX_SKILL_PRICE_TOKENS = 50_000_000

# Self-modification physics (plan §6.3): proposing a new persona costs a
# burned fee — molting energy, spent whether or not the change is adopted.
# The cooldown (loops plan F7) stops identity churn: metamorphosis got 30
# days; the cheaper lever gets a week, for every stage including adults.
SELF_MOD_FEE_TOKENS = 250_000
SELF_MOD_COOLDOWN_DAYS = 7
PERSONA_MAX_CHARS = 2000
PERSONA_MIN_CHARS = 40

# Earning physics (plan §5.1): the open bounty market and being-initiated
# recurring income. Fees/prices are minted only by the parent at judged
# delivery — conservation holds, and these clamps keep asks sane.
QUEST_MAX_FEE_TOKENS = 20_000_000
VENTURE_MAX_PRICE_TOKENS = 5_000_000
VENTURE_MIN_CADENCE_DAYS = 1
VENTURE_MAX_CADENCE_DAYS = 30

# The second currency (space plan Phase 2): coins are MONEY, tokens are
# METABOLISM. Exchange is ONE-WAY (coins → tokens) — thinking never buys
# coins, or wealth could be printed from the allowance. Conversion opens
# in adolescence (the 'convert' capability); even an infant may RECEIVE
# pocket money (a gift is not a trade). Coins are integers and scarce on
# purpose — a poem at the market should cost 3, not 300000.
COIN_TOKEN_RATE = 100_000        # tokens minted per coin converted
COIN_GRANT_MAX = 1_000           # pocket money per parent grant
WORK_MAX_FEE_COINS = 500         # chore/quest coin denomination cap

# The market (space plan Phase 3): beings trade REAL files for coins.
# Trades/day grow with the stage (market Saturday adds 2 on top, via
# being_world.trades_cap — one source, like letters). Prices stay small
# on purpose: a poem should cost 3 coins, not 300.
MARKET_MAX_PRICE_COINS = 100
_TRADES_PER_DAY = {"egg": 0, "infant": 0, "child": 3,
                   "adolescent": 5, "adult": 8}
# Commissioned buildings (space plan Phase 5): the village pools coins to
# raise a NEW place — the long-arc savings goal that makes wealth mean
# something. The cost is deliberately weeks of pocket money; the coins
# burn on approval (the economy's one true sink besides conversion).
COMMISSION_COST_COINS = 50

# Made things (docs/being-world-shaping-plan.md Phase 1): a being crafts a
# real thing (a proof file + a burned token fee — making costs THOUGHT,
# never money) and places it on open ground for free. The world's capacity
# is a function of its ground: the cap is derived from plot area, so it
# rises by itself when the village grows. The per-being floor keeps a
# crowded village from zeroing anyone out of shaping their world.
OBJECT_CRAFT_FEE_TOKENS = 25_000
OBJECT_AREA_PER_SLOT = 25_000    # plot units² per standing object (40 today)
OBJECT_MIN_PER_BEING = 3         # every being may always keep a few standing

# The body brain (docs/being-body-brain-plan.md): the mind may keep a few
# tasks on the work board the feet work — the mind assigns, the feet take
# up or refuse. Bounded so a board stays a board, not a queue; stale tasks
# lapse quietly (a task the world outran is not a debt).
PLAN_STEPS_MAX = 6
PLAN_LAPSE_DAYS = 7

# Procreation physics (plan §8): the dowry moves from the parents' savings
# to the child (reason='procreation') — earned wealth, never conjured; a
# couple splits it. Consent is the human parent's authenticated approval.
PROCREATION_COST_TOKENS = 10_000_000
# Torpor grace: unfed this long → death. Mortality is real (plan §8).
TORPOR_GRACE_DAYS = 14

# Capabilities unlocked per stage (cumulative — a stage grants everything
# below it plus its own row).
_STAGE_GRANTS: dict[str, frozenset[str]] = {
    "egg": frozenset(),
    "infant": frozenset({"chat", "journal", "vfs_home"}),
    "child": frozenset({"web_read", "flows", "commons_read", "chores",
                        "letters", "self_mod"}),
    "adolescent": frozenset({
        "commons_write", "spawn_agents", "agent_messaging",
        "organ_runs", "trade", "jobs", "ventures", "convert",
    }),
    "adult": frozenset({"self_mod_auto", "procreate", "negotiate"}),
}

# Per-stage physics: max allowance preset, model tiers, savings ceiling
# (days of allowance), metamorphosis policy (none | cosign | auto).
STAGES: dict[str, dict] = {
    "egg": {"max_preset": None, "tiers": (), "savings_days": 0,
            "metamorphosis": "none", "letters_per_day": 0},
    "infant": {"max_preset": "2M", "tiers": ("fast",), "savings_days": 3,
               "metamorphosis": "none", "letters_per_day": 0},
    "child": {"max_preset": "5M", "tiers": ("fast",), "savings_days": 7,
              "metamorphosis": "none", "letters_per_day": 3},
    "adolescent": {"max_preset": "20M", "tiers": ("fast", "balanced", "coding"),
                   "savings_days": 30, "metamorphosis": "cosign",
                   "letters_per_day": 5},
    "adult": {"max_preset": "unlimited",
              "tiers": ("fast", "balanced", "coding", "longctx", "vision", "reason"),
              "savings_days": 60, "metamorphosis": "auto",
              "letters_per_day": 8},
}


def stage_index(stage: str) -> int:
    return STAGE_ORDER.index(stage)


def letters_per_day(stage: str) -> int:
    """Daily letter quota for a stage — reach grows with maturity."""
    st = STAGES.get(stage)
    if st is None or "letters_per_day" not in st:
        return LETTERS_PER_DAY
    return int(st["letters_per_day"])


def trades_per_day(stage: str) -> int:
    """Daily market-trade quota (sells posted + buys made count alike)."""
    return _TRADES_PER_DAY.get(stage, 0)


ATTENTION_CREDITS_PER_DAY = 5


def attention_per_day(stage: str) -> int:
    """Daily attention credits — unprompted words a being may send its parent,
    reset each midnight. A gift, not a right, but enough that a being isn't
    locked mute after a few reaches. Eggs don't speak."""
    return 0 if stage == "egg" else ATTENTION_CREDITS_PER_DAY


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
