"""Generic single-shot LLM completion on the caller's configured tier.

FD's ``/fd/archetypes/generate`` and ``/forge`` already do one-off completions,
but the caller must pass provider/model/key. This endpoint instead resolves the
OWNER's saved Library tier server-side (the same ``_load_owner_tiers`` +
``_resolve_creds`` path the Vatra/Basna runs use) and runs one completion — so
an API caller (or FD's own UI) can get a single structured generation on the
user's own model without ever handling raw credentials.

Generic: it knows nothing about any product. Not a decomposition, not a run —
one prompt in, one completion out. Use it for structured one-shot generations
(config blobs, drafts) where a multi-agent run would be the wrong tool.

FD services call the same path directly: :func:`complete_for_owner` is the
route without HTTP, and :func:`complete_with_fallback` / :func:`first_owner_tier`
are its unattended form — for a decision no human sees before it acts (the
Gmail auto-reply classifier and judge), which runs on the owner's ``reason`` or
``balanced`` tier only, never ``micro`` or ``fast`` (weak models fail silently).
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterable
from typing import Any, NamedTuple

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from captain_claw.flight_deck.auth import get_current_user, get_db
from captain_claw.logging import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/fd/llm", tags=["llm"])

_VALID_TIERS = ("reason", "balanced", "fast", "longctx", "micro")

# Tiers an unattended decision may run on, and the order each caller tries
# them in. The classifier prefers the cheaper model, the judge the stronger.
UNATTENDED_TIERS = frozenset({"reason", "balanced"})
CLASSIFIER_TIER_ORDER = ("balanced", "reason")
JUDGE_TIER_ORDER = ("reason", "balanced")


class OwnerTierError(Exception):
    """An owner-tier completion produced nothing.

    ``status`` is what the route answers with: 400 the tier has no model, 500
    the archetype registry is missing or the owner's tiers can't be read, 502
    the LLM call failed (message included), 504 it outran ``timeout``.
    """

    def __init__(self, status: int, detail: Any) -> None:
        super().__init__(detail)
        self.status = status
        self.detail = detail


class OwnerCompletion(NamedTuple):
    """One completion: unpacks as ``(content, finish_reason, provider, model)``."""

    content: str
    finish_reason: str        # "" when the provider doesn't say
    provider: str | None      # as the tier names it (the route echoes it)
    model: str | None


async def _creds_resolver(db, owner: str) -> Callable[[str], dict]:
    """A ``tier -> creds`` resolver over *owner*'s saved Library tiers (the
    team default when they set none, the registry for a tier neither has).
    Raises :class:`OwnerTierError` (500) when the registry is missing."""
    # Lazy imports: these modules are heavy and import each other; importing at
    # call time keeps this route free of the import cycle.
    from captain_claw.flight_deck.basna_routes import _load_owner_tiers, _load_registry
    from captain_claw.flight_deck.vatra_routes import _resolve_creds

    tiers, _env = await _load_owner_tiers(db, owner)
    try:
        registry = _load_registry()
    except HTTPException as e:
        raise OwnerTierError(e.status_code, e.detail) from None
    return lambda tier: _resolve_creds(registry, tiers, "", tier)


async def _complete(
    creds: dict, system: str, user: str, *, max_tokens: int, temperature: float,
    timeout: float | None, usage_out: dict | None, owner: str = "", feature: str = "",
) -> OwnerCompletion:
    """Run one completion on resolved *creds*. Raises :class:`OwnerTierError`
    (502 on any LLM error, 504 past *timeout*) and nothing else. A call for a
    background *feature* says how it went (side_call_health)."""
    from captain_claw.flight_deck import side_call_health

    who = (owner, feature, str(creds.get("provider") or ""), str(creds.get("model") or ""),
           str(creds.get("base_url") or ""))
    try:
        from captain_claw.llm import Message, create_provider
        provider = create_provider(
            provider=creds.get("provider") or "anthropic",
            model=creds["model"],
            api_key=creds.get("api_key") or None,
            base_url=creds.get("base_url") or None,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        messages = []
        if system.strip():
            messages.append(Message(role="system", content=system))
        messages.append(Message(role="user", content=user))
        call = provider.complete(
            messages=messages, temperature=temperature,
            max_tokens=max_tokens)
        resp = await (call if timeout is None else asyncio.wait_for(call, timeout))
    except Exception as e:  # noqa: BLE001 — surface the real LLM error to the caller
        await side_call_health.failed(*who, error=e)
        if timeout is not None and isinstance(e, TimeoutError):
            log.warning("llm complete timed out", timeout=timeout)
            raise OwnerTierError(504, f"LLM call timed out after {timeout:g}s") from None
        log.error("llm complete failed", exc_info=True)
        raise OwnerTierError(502, f"LLM call failed: {e}") from e

    if (resp.content or "").strip():
        await side_call_health.ok(*who)
    else:
        await side_call_health.failed(*who, error=None)
    usage = getattr(resp, "usage", None)
    if usage_out is not None and isinstance(usage, dict):
        usage_out.update(usage)
    # getattr: not every provider (nor every test fake) reports a finish reason.
    return OwnerCompletion((resp.content or "").strip(),
                           str(getattr(resp, "finish_reason", "") or ""),
                           creds.get("provider"), creds.get("model"))


async def complete_for_owner(
    db, owner: str, tier: str, system: str, user: str, *,
    max_tokens: int, temperature: float,
    timeout: float | None = None, usage_out: dict | None = None, feature: str = "",
) -> OwnerCompletion:
    """One completion on *owner*'s ``tier`` — ``POST /fd/llm/complete`` without
    the HTTP. *user* is the user message; an empty *system* sends none.
    *feature*: the background work it's for, when it is (side_call_health).

    Returns ``(content, finish_reason, provider, model)`` with ``content``
    stripped. Raises :class:`OwnerTierError` for every outcome the route turns
    into an error status. *timeout* (seconds) bounds the LLM call; *usage_out*,
    when given, is filled with the provider's token counts. Any tier name is
    accepted — use :func:`complete_with_fallback` for an unattended decision.
    """
    creds = (await _creds_resolver(db, owner))(tier)
    if not creds.get("model"):
        raise OwnerTierError(400, f"no model configured for the '{tier}' tier")
    return await _complete(creds, system, user, max_tokens=max_tokens,
                           temperature=temperature, timeout=timeout,
                           usage_out=usage_out, owner=owner, feature=feature)


def _unattended(order: Iterable[str]) -> list[str]:
    """*order* without duplicates and without any tier outside
    :data:`UNATTENDED_TIERS` — a stray ``micro`` or ``fast`` is dropped."""
    return [t for t in dict.fromkeys(order) if t in UNATTENDED_TIERS]


async def owner_tier_model(db, owner: str, tier: str) -> tuple[str, str] | None:
    """``(provider, model)`` *owner*'s ``tier`` would run on, or None when it
    has no model or can't be resolved. Never raises.

    A model is not proof of a working key: a tier left on the registry default
    still resolves, and its first call may fail."""
    try:
        creds = (await _creds_resolver(db, owner))(tier)
    except Exception as e:  # noqa: BLE001 — "doesn't resolve" is the answer
        log.debug("owner tier unresolved", tier=tier, error=type(e).__name__)
        return None
    model = str(creds.get("model") or "")
    return (str(creds.get("provider") or ""), model) if model else None


async def first_owner_tier(
    db, owner: str, order: Iterable[str],
) -> tuple[str, str, str] | None:
    """``(tier, provider, model)`` for the first tier in *order* that resolves
    for *owner* — the one :func:`complete_with_fallback` tries first. Only
    ``reason`` and ``balanced`` are considered; None when neither resolves.
    Never raises. For enable checks and status views."""
    for tier in _unattended(order):
        hit = await owner_tier_model(db, owner, tier)
        if hit:
            return (tier, *hit)
    return None


async def complete_with_fallback(
    db, owner: str, order: Iterable[str], system: str, user: str, *,
    max_tokens: int, temperature: float,
    timeout: float | None = None, usage_out: dict | None = None, feature: str = "",
) -> OwnerCompletion:
    """:func:`complete_for_owner` for an unattended decision: tries the tiers
    of *order* (:data:`CLASSIFIER_TIER_ORDER`, :data:`JUDGE_TIER_ORDER`) in
    turn and returns the first completion.

    A tier outside :data:`UNATTENDED_TIERS` is never tried. A tier moves on to
    the next when it has no model or its call fails or times out (*timeout*
    applies per attempt); one that resolves to a model already tried is
    skipped, since that would only repeat the failure. A completion is a
    success here — the caller judges ``finish_reason`` and the content.

    Raises :class:`OwnerTierError` (the last failure) and nothing else.
    """
    tiers = _unattended(order)
    if not tiers:
        raise OwnerTierError(400, "only the 'reason' or 'balanced' tier may decide unattended")
    try:
        resolve = await _creds_resolver(db, owner)
    except OwnerTierError:
        raise
    except Exception as e:  # noqa: BLE001 — callers get one error type
        log.warning("owner tiers unreadable", error=type(e).__name__)
        raise OwnerTierError(500, "could not read the owner's tiers") from None

    last = OwnerTierError(400, "no model configured for the "
                          + " or ".join(f"'{t}'" for t in tiers) + " tier")
    tried: set[tuple] = set()
    for tier in tiers:
        try:
            creds = resolve(tier)
        except Exception as e:  # noqa: BLE001 — a broken tier is one not to run on
            log.warning("owner tier unresolved", tier=tier, error=type(e).__name__)
            continue
        if not creds.get("model"):
            continue
        same = (creds.get("provider"), creds.get("model"), creds.get("base_url"))
        if same in tried:
            continue
        tried.add(same)
        try:
            return await _complete(creds, system, user, max_tokens=max_tokens,
                                   temperature=temperature, timeout=timeout,
                                   usage_out=usage_out, owner=owner, feature=feature)
        except OwnerTierError as e:
            last = e
    raise last


@router.get("/side-call-health")
async def side_call_health_now(user: dict = Depends(get_current_user)) -> dict:
    """The caller's background models that aren't working now — what their
    Line told them about (side_call_health): model, why, what waits on it."""
    from captain_claw.flight_deck import side_call_health

    return {"problems": side_call_health.problems(user.get("id"))}


class CompleteRequest(BaseModel):
    prompt: str = Field(min_length=1)
    system: str = ""
    tier: str = "balanced"
    max_tokens: int = Field(default=4096, ge=64, le=32768)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)


@router.post("/complete")
async def complete(body: CompleteRequest, user: dict = Depends(get_current_user)):
    """One LLM completion on the owner's configured ``tier``. Returns
    ``{content, provider, model}``. 400 if that tier has no model; 502 on an
    LLM error (message included)."""
    tier = body.tier if body.tier in _VALID_TIERS else "balanced"
    db = get_db()
    try:
        content, _finish, provider, model = await complete_for_owner(
            db, user["id"], tier, body.system, body.prompt,
            max_tokens=body.max_tokens, temperature=body.temperature)
    except OwnerTierError as e:
        raise HTTPException(e.status, e.detail)

    return {"content": content, "provider": provider, "model": model}
