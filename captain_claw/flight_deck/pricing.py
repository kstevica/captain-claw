"""Model pricing + token→dollar cost for Flight Deck runs.

Turns the token usage a run already produces into a dollar cost and an effective
hourly rate, so a run's cost is directly comparable to human wages.

Rates are curated per-million-token prices in
``captain_claw/instructions/model_prices.json`` (input / output / cache-read /
cache-write) — a single-file reprice, matching the model-tier pattern. A Library
tier may pass its own ``price`` override (e.g. a negotiated rate, or a
compute-cost estimate for a self-hosted model). Everything here is pure — no
network, no model calls — so it's fully unit-testable.

Cache accounting: :func:`cost_from_usage` prices cache *reads* and *writes* at
their own (much cheaper / slightly dearer) rates, using the normalized usage
shape from ``llm._normalize_usage`` — ``prompt_tokens`` is the fresh, uncached
input and ``cache_creation_input_tokens`` / ``cache_read_input_tokens`` are
billed on top (the Anthropic convention). A model with no known price yields
``priced=False`` (tokens still counted; dollars shown as unknown) rather than a
wrong number.
"""

from __future__ import annotations

import functools
import json
from pathlib import Path

from captain_claw.logging import get_logger

log = get_logger(__name__)

_PRICES_FILE = Path(__file__).resolve().parent.parent / "instructions" / "model_prices.json"

# Rate components, all in USD per 1,000,000 tokens.
_RATE_KEYS = ("input", "output", "cache_read", "cache_write")

# The token fields we aggregate (the normalized usage shape).
_TOKEN_KEYS = (
    "prompt_tokens", "completion_tokens", "total_tokens",
    "cache_creation_input_tokens", "cache_read_input_tokens",
)

_MILLION = 1_000_000


@functools.lru_cache(maxsize=1)
def _table() -> dict:
    """The curated price table (model id → per-million rates). Best-effort."""
    try:
        raw = json.loads(_PRICES_FILE.read_text())
    except Exception as e:  # noqa: BLE001 — pricing is best-effort; missing table → unpriced
        log.warning("model price table unreadable; runs will show tokens only", error=str(e))
        return {}
    return {k: v for k, v in raw.items() if not k.startswith("_") and isinstance(v, dict)}


def _norm_id(model: str) -> str:
    m = (model or "").strip().lower()
    if "/" in m:  # "anthropic/claude-opus-4-8" / "gemini/gemini-2.5-pro" → bare model
        m = m.split("/")[-1]
    return m


def _coerce(entry: dict) -> dict:
    return {k: float(entry.get(k, 0) or 0) for k in _RATE_KEYS}


def rate_for(model: str, override: dict | None = None) -> dict | None:
    """Per-million rates for ``model``, or ``None`` if unknown.

    An ``override`` carrying any rate key wins outright (Library-tier price). Else
    the table is matched exact-first, then by the longest id prefix, so dated
    variants ("claude-opus-4-8-20260601") price off their family ("claude-opus-4-8").
    """
    if override and any(k in override for k in _RATE_KEYS):
        return _coerce(override)
    mid = _norm_id(model)
    if not mid:
        return None
    table = _table()
    if mid in table:
        return _coerce(table[mid])
    best: str | None = None
    for key in table:
        if mid.startswith(key) and (best is None or len(key) > len(best)):
            best = key
    return _coerce(table[best]) if best else None


def cost_from_usage(model: str, usage: dict | None, override: dict | None = None) -> dict:
    """Dollar cost for one usage dict, pricing cache reads/writes separately.

    Returns a breakdown ``{usd, input_usd, output_usd, cache_read_usd,
    cache_write_usd, priced, model}``. ``usd`` is ``None`` and ``priced`` is
    ``False`` when the model has no known rate (so callers show "—", never a wrong
    number). Cache read/write fall back to the input rate if the table leaves them 0.
    """
    u = usage or {}
    pt = int(u.get("prompt_tokens", 0) or 0)
    ct = int(u.get("completion_tokens", 0) or 0)
    cw = int(u.get("cache_creation_input_tokens", 0) or 0)
    cr = int(u.get("cache_read_input_tokens", 0) or 0)
    rate = rate_for(model, override)
    if not rate:
        return {"usd": None, "input_usd": 0.0, "output_usd": 0.0,
                "cache_read_usd": 0.0, "cache_write_usd": 0.0,
                "priced": False, "model": model}
    input_usd = pt / _MILLION * rate["input"]
    output_usd = ct / _MILLION * rate["output"]
    cache_write_usd = cw / _MILLION * (rate["cache_write"] or rate["input"])
    cache_read_usd = cr / _MILLION * (rate["cache_read"] or rate["input"])
    usd = input_usd + output_usd + cache_write_usd + cache_read_usd
    return {
        "usd": round(usd, 6),
        "input_usd": round(input_usd, 6),
        "output_usd": round(output_usd, 6),
        "cache_read_usd": round(cache_read_usd, 6),
        "cache_write_usd": round(cache_write_usd, 6),
        "priced": True,
        "model": model,
    }


def hourly_rate(usd: float | None, elapsed_seconds: float | None) -> float | None:
    """Effective $/hour = spend ÷ wall-clock hours — the number to compare to a wage.

    ``None`` when cost is unknown or the elapsed time is missing/zero.
    """
    if usd is None or not elapsed_seconds or elapsed_seconds <= 0:
        return None
    return round(usd / (elapsed_seconds / 3600.0), 4)


def summarize(agents: list[dict], elapsed_seconds: float | None = None) -> dict:
    """Roll a run's per-agent usage into one cost block for the UI.

    ``agents`` is a list of ``{model, usage, price?}`` (``price`` is an optional
    per-agent rate override). Returns aggregate tokens, total ``usd`` (``None`` if
    NOTHING could be priced), a per-model breakdown, the run ``elapsed_seconds`` and
    the effective ``hourly_usd``. The human-wage comparison is applied client-side
    from a user-set wage, so it isn't baked in here.
    """
    totals = {k: 0 for k in _TOKEN_KEYS}
    per_model: dict[str, dict] = {}
    usd = 0.0
    any_priced = False
    for a in agents or []:
        u = a.get("usage") or {}
        for k in _TOKEN_KEYS:
            totals[k] += int(u.get(k, 0) or 0)
        c = cost_from_usage(a.get("model", ""), u, a.get("price"))
        mid = a.get("model") or "?"
        pm = per_model.setdefault(mid, {
            "usd": 0.0, "prompt_tokens": 0, "completion_tokens": 0,
            "cache_read_input_tokens": 0, "priced": False, "calls": 0})
        pm["calls"] += 1
        pm["prompt_tokens"] += int(u.get("prompt_tokens", 0) or 0)
        pm["completion_tokens"] += int(u.get("completion_tokens", 0) or 0)
        pm["cache_read_input_tokens"] += int(u.get("cache_read_input_tokens", 0) or 0)
        if c["priced"]:
            any_priced = True
            usd += c["usd"] or 0.0
            pm["usd"] = round(pm["usd"] + (c["usd"] or 0.0), 6)
            pm["priced"] = True
    out: dict = {
        "tokens": totals,
        "usd": round(usd, 6) if any_priced else None,
        "priced": any_priced,
        "per_model": per_model,
        "elapsed_seconds": round(elapsed_seconds, 2) if elapsed_seconds else None,
        "hourly_usd": hourly_rate(usd if any_priced else None, elapsed_seconds),
    }
    return out
