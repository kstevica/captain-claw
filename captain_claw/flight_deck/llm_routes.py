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
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from captain_claw.flight_deck.auth import get_current_user, get_db
from captain_claw.logging import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/fd/llm", tags=["llm"])

_VALID_TIERS = ("reason", "balanced", "fast", "longctx", "micro")


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
    # Lazy imports: these modules are heavy and import each other; importing at
    # call time keeps this route free of the import cycle.
    from captain_claw.flight_deck.basna_routes import _load_owner_tiers, _load_registry
    from captain_claw.flight_deck.vatra_routes import _resolve_creds

    tiers, _env = await _load_owner_tiers(db, user["id"])
    creds = _resolve_creds(_load_registry(), tiers, "", tier)
    if not creds.get("model"):
        raise HTTPException(400, f"no model configured for the '{tier}' tier")

    try:
        from captain_claw.llm import Message, create_provider
        provider = create_provider(
            provider=creds.get("provider") or "anthropic",
            model=creds["model"],
            api_key=creds.get("api_key") or None,
            base_url=creds.get("base_url") or None,
            temperature=body.temperature,
            max_tokens=body.max_tokens,
        )
        messages = []
        if body.system.strip():
            messages.append(Message(role="system", content=body.system))
        messages.append(Message(role="user", content=body.prompt))
        resp = await provider.complete(
            messages=messages, temperature=body.temperature,
            max_tokens=body.max_tokens)
    except Exception as e:  # noqa: BLE001 — surface the real LLM error to the caller
        log.error("llm complete failed", exc_info=True)
        raise HTTPException(502, f"LLM call failed: {e}")

    return {"content": (resp.content or "").strip(),
            "provider": creds.get("provider"), "model": creds.get("model")}
