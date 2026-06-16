"""Per-tenant agent archetype endpoints for Flight Deck.

The base archetype set lives in ``instructions/archetypes.json``. These routes
let each user (tenant) add their own archetypes — manually or generated from a
prompt — stored in the ``user_archetypes`` table. ``GET /fd/archetypes`` returns
the merged registry (base + the caller's own, the latter shadowing a base entry
when ids match); the Forge generator and Basna router/executor read the same
merged set via :mod:`captain_claw.flight_deck.archetypes`.
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from captain_claw.flight_deck import auth as _auth
from captain_claw.flight_deck.archetypes import load_base_registry, merged_registry
from captain_claw.flight_deck.auth import get_current_user, get_db, get_optional_user
from captain_claw.logging import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/fd/archetypes", tags=["archetypes"])

_INSTRUCTIONS_DIR = Path(__file__).parent.parent / "instructions"
_VALID_TIERS = {"reason", "balanced", "fast", "longctx"}
_VALID_MODES = {"ionian", "dorian", "phrygian", "lydian", "mixolydian",
                "aeolian", "locrian", "neutra"}


async def _optional_user(user: dict | None = Depends(get_optional_user)) -> dict | None:
    """Resolve the caller, tolerating unauthenticated and auth-disabled modes.

    Authenticated → that user. Unauthenticated internal call (auth enabled) →
    None (base-only registry). Auth-disabled desktop → the synthetic local user,
    so a single-user install still sees its own custom archetypes.
    """
    if user is None and not _auth._fd_auth_enabled():
        return dict(_auth._LOCAL_USER)
    return user


# ── Models ───────────────────────────────────────────────────────────

class ArchetypeBody(BaseModel):
    archetype_id: str = ""
    role: str = ""
    family: str = "Custom"
    description: str = ""
    cognitive_mode: str = "neutra"
    tier: str = "balanced"
    tools: list[str] = []
    fleet_instructions: str = ""
    keywords: list[str] = []
    lead: bool = False
    reliability_seed: float = 0.7


class GenerateRequest(BaseModel):
    prompt: str = ""
    provider: str = ""
    model: str = ""
    api_key: str = ""
    base_url: str = ""
    max_tokens: int = Field(default=0)


def _slugify(s: str) -> str:
    out = "".join(c if c.isalnum() else "-" for c in (s or "").strip().lower())
    while "--" in out:
        out = out.replace("--", "-")
    return out.strip("-")


def _validate(body: ArchetypeBody) -> dict:
    """Validate and normalize an archetype into the stored ``data`` shape."""
    aid = _slugify(body.archetype_id or body.role)
    if not aid:
        raise HTTPException(400, "archetype_id (or role) is required")
    if body.tier not in _VALID_TIERS:
        raise HTTPException(400, f"tier must be one of {sorted(_VALID_TIERS)}")
    mode = body.cognitive_mode or "neutra"
    if mode not in _VALID_MODES:
        raise HTTPException(400, f"cognitive_mode must be one of {sorted(_VALID_MODES)}")
    if not body.role.strip():
        raise HTTPException(400, "role is required")
    return {
        "id": aid,
        "role": body.role.strip(),
        "family": (body.family or "Custom").strip(),
        "description": body.description.strip(),
        "cognitive_mode": mode,
        "tier": body.tier,
        "tools": body.tools,
        "fleet_instructions": body.fleet_instructions,
        "keywords": [k.strip() for k in body.keywords if k.strip()],
        "lead": bool(body.lead),
        "reliability_seed": float(body.reliability_seed),
    }


# ── Routes ───────────────────────────────────────────────────────────

@router.get("")
async def list_archetypes(user: dict | None = Depends(_optional_user)):
    """Merged archetype registry (base + the caller's own)."""
    # Tolerate a not-yet-initialized DB (e.g. auth-disabled standalone): fall
    # back to the base set, matching the pre-merge behavior of this endpoint.
    try:
        db = get_db()
    except AssertionError:
        db = None
    uid = user["id"] if (user and db is not None) else None
    try:
        if db is None:
            reg = load_base_registry()
            reg["archetypes"] = [{**a, "source": "base"} for a in reg.get("archetypes", [])]
            return reg
        return await merged_registry(db, uid)
    except FileNotFoundError:
        raise HTTPException(500, "Archetype registry not found")
    except json.JSONDecodeError as e:
        raise HTTPException(500, f"Archetype registry is invalid JSON: {e}")


@router.get("/mine")
async def list_mine(user: dict = Depends(get_current_user)):
    """Just the caller's own custom archetypes, parsed."""
    db = get_db()
    rows = await db.list_user_archetypes(user["id"])
    out = []
    for r in rows:
        try:
            data = json.loads(r.get("data") or "{}")
        except json.JSONDecodeError:
            data = {}
        out.append({**data, "id": r["archetype_id"], "source": "user",
                    "updated_at": r.get("updated_at")})
    return out


@router.post("")
async def create_archetype(body: ArchetypeBody, user: dict = Depends(get_current_user)):
    db = get_db()
    data = _validate(body)
    if await db.get_user_archetype(user["id"], data["id"]):
        raise HTTPException(409, f"archetype '{data['id']}' already exists")
    return await db.create_user_archetype(user["id"], data["id"], json.dumps(data))


@router.put("/{archetype_id}")
async def update_archetype(
    archetype_id: str, body: ArchetypeBody, user: dict = Depends(get_current_user),
):
    db = get_db()
    data = _validate(body)
    # The path id is authoritative for which row we touch; keep the slug stable.
    data["id"] = archetype_id
    existing = await db.get_user_archetype(user["id"], archetype_id)
    if existing:
        return await db.update_user_archetype(user["id"], archetype_id, json.dumps(data))
    # Upsert: allow PUT to create (e.g. saving an override of a base archetype).
    return await db.create_user_archetype(user["id"], archetype_id, json.dumps(data))


@router.delete("/{archetype_id}")
async def delete_archetype(archetype_id: str, user: dict = Depends(get_current_user)):
    db = get_db()
    ok = await db.delete_user_archetype(user["id"], archetype_id)
    if not ok:
        raise HTTPException(404, "Archetype not found")
    return {"ok": True}


@router.post("/generate")
async def generate_archetype(body: GenerateRequest, user: dict = Depends(get_current_user)):
    """Draft a single archetype from a natural-language prompt (not persisted).

    Returns a JSON object the UI loads into the edit form for review before the
    user saves it via ``POST /fd/archetypes``.
    """
    if not body.prompt.strip():
        raise HTTPException(400, "prompt is required")
    system_prompt_file = _INSTRUCTIONS_DIR / "archetype_generate_system_prompt.md"
    if not system_prompt_file.is_file():
        raise HTTPException(500, "Archetype generation prompt not found")
    system_prompt = system_prompt_file.read_text()

    try:
        from captain_claw.llm import create_provider, Message
        max_tokens = body.max_tokens if body.max_tokens > 0 else 8192
        provider = create_provider(
            provider=body.provider,
            model=body.model,
            api_key=body.api_key or None,
            base_url=body.base_url or None,
            temperature=0.7,
            max_tokens=max_tokens,
        )
        response = await provider.complete(
            messages=[
                Message(role="system", content=system_prompt),
                Message(role="user", content=body.prompt.strip()),
            ],
            temperature=0.7,
            max_tokens=max_tokens,
        )
    except Exception as e:
        log.error("Archetype generation LLM call failed", exc_info=True)
        raise HTTPException(502, f"LLM call failed: {e}")

    content = response.content.strip()
    if content.startswith("```"):
        lines = [l for l in content.split("\n") if not l.strip().startswith("```")]
        content = "\n".join(lines)
    try:
        result = json.loads(content)
    except json.JSONDecodeError:
        raise HTTPException(502, f"LLM returned invalid JSON: {content[:500]}")

    # Normalize through the same validator so the draft is form-ready and safe.
    # Coerce out-of-vocabulary tier/mode to safe defaults — a draft should never
    # 400; the user can adjust before saving.
    tier = result.get("tier", "balanced")
    mode = result.get("cognitive_mode", "neutra")
    draft = _validate(ArchetypeBody(
        archetype_id=result.get("id") or result.get("archetype_id") or result.get("role", ""),
        role=result.get("role", ""),
        family=result.get("family", "Custom"),
        description=result.get("description", ""),
        cognitive_mode=mode if mode in _VALID_MODES else "neutra",
        tier=tier if tier in _VALID_TIERS else "balanced",
        tools=result.get("tools", []) or [],
        fleet_instructions=result.get("fleet_instructions", ""),
        keywords=result.get("keywords", []) or [],
        lead=bool(result.get("lead", False)),
        reliability_seed=float(result.get("reliability_seed", 0.7)),
    ))
    draft["source"] = "user"
    return draft
