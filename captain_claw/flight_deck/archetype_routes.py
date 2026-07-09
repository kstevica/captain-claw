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

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
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
        max_tokens = body.max_tokens if body.max_tokens > 0 else 16384
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


@router.post("/forge")
async def forge_archetypes(
    instructions: str = Form(""),
    provider: str = Form(""),
    model: str = Form(""),
    api_key: str = Form(""),
    base_url: str = Form(""),
    max_tokens: int = Form(0),
    count: int = Form(0),
    files: list[UploadFile] = File(default=[]),
    user: dict = Depends(get_current_user),
):
    """Forge MULTIPLE archetype drafts from instructions + optional documents.

    Uploaded documents (pdf/docx/xlsx/pptx/txt/md/…) are extracted to text
    server-side and folded into the prompt as reference context. Returns a list
    of drafts (NOT persisted); the UI reviews and saves the selected ones via
    ``POST /fd/archetypes``. Mirrors the single ``/generate`` route but returns a
    batch and accepts reference material.
    """
    instructions = (instructions or "").strip()

    # ── Extract text from any uploaded documents (best-effort per file) ──
    _MAX_PER_FILE = 40000
    _MAX_TOTAL = 160000
    doc_sections: list[str] = []
    total = 0
    if files:
        import os
        import tempfile
        from captain_claw.tools.summarize_files import SummarizeFilesTool
        for uf in files:
            if not uf or not uf.filename:
                continue
            try:
                raw = await uf.read()
            except Exception:
                continue
            if not raw:
                continue
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=(Path(uf.filename).suffix or ".txt"),
                ) as tmp:
                    tmp.write(raw)
                    tmp_path = Path(tmp.name)
                text, err = SummarizeFilesTool._read_file_content(tmp_path)
            except Exception as exc:
                text, err = None, str(exc)
            finally:
                if tmp_path is not None:
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass
            if not text:
                log.info("Archetype forge: could not extract file",
                         file=uf.filename, error=err)
                continue
            remaining = _MAX_TOTAL - total
            if remaining <= 0:
                break
            text = text[:min(_MAX_PER_FILE, remaining)]
            total += len(text)
            doc_sections.append(f"### {uf.filename}\n{text}")

    if not instructions and not doc_sections:
        raise HTTPException(400, "instructions or at least one document is required")

    system_prompt_file = _INSTRUCTIONS_DIR / "archetype_forge_system_prompt.md"
    if not system_prompt_file.is_file():
        raise HTTPException(500, "Archetype forge prompt not found")
    system_prompt = system_prompt_file.read_text()

    user_parts: list[str] = []
    if instructions:
        user_parts.append(f"## Instructions\n{instructions}")
    if count and count > 0:
        user_parts.append(f"\nDesign approximately {count} archetypes.")
    if doc_sections:
        user_parts.append("\n## Reference documents\n" + "\n\n".join(doc_sections))
    user_prompt = "\n".join(user_parts).strip()

    from captain_claw.llm import Message, create_provider

    _FORGE_CEIL = 64000
    forge_max_tokens = max(max_tokens if max_tokens > 0 else 32768, 8192)

    def _truncated(resp) -> bool:
        return str(getattr(resp, "finish_reason", "") or "").lower() in {"length", "max_tokens"}

    async def _run(mt: int):
        prov = create_provider(
            provider=provider or "anthropic", model=model or "",
            api_key=api_key or None, base_url=base_url or None,
            temperature=0.7, max_tokens=mt,
        )
        return await prov.complete(
            messages=[
                Message(role="system", content=system_prompt),
                Message(role="user", content=user_prompt),
            ],
            temperature=0.7, max_tokens=mt,
        )

    try:
        response = await _run(forge_max_tokens)
        if _truncated(response) and forge_max_tokens < _FORGE_CEIL:
            retry = min(forge_max_tokens * 2, _FORGE_CEIL)
            try:
                response = await _run(retry)
                forge_max_tokens = retry
            except Exception:
                log.warning("Archetype forge retry failed; keeping first", exc_info=True)
    except Exception as e:
        log.error("Archetype forge LLM call failed", exc_info=True)
        raise HTTPException(502, f"LLM call failed: {e}")

    content = response.content.strip()
    if content.startswith("```"):
        content = "\n".join(l for l in content.split("\n") if not l.strip().startswith("```"))
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        if _truncated(response):
            raise HTTPException(
                502,
                "The archetype set was cut off before it finished (hit the output "
                "token limit). Raise the forge tier's output budget, or ask for fewer "
                "archetypes.",
            )
        raise HTTPException(502, f"LLM returned invalid JSON: {content[:500]}")

    raw_list = parsed.get("archetypes") if isinstance(parsed, dict) else parsed
    if not isinstance(raw_list, list):
        raise HTTPException(502, "LLM did not return an 'archetypes' array")

    drafts: list[dict] = []
    seen_ids: set[str] = set()
    for item in raw_list:
        if not isinstance(item, dict):
            continue
        tier = item.get("tier", "balanced")
        mode = item.get("cognitive_mode", "neutra")
        try:
            draft = _validate(ArchetypeBody(
                archetype_id=item.get("id") or item.get("archetype_id") or item.get("role", ""),
                role=item.get("role", ""),
                family=item.get("family", "Custom"),
                description=item.get("description", ""),
                cognitive_mode=mode if mode in _VALID_MODES else "neutra",
                tier=tier if tier in _VALID_TIERS else "balanced",
                tools=item.get("tools", []) or [],
                fleet_instructions=item.get("fleet_instructions", ""),
                keywords=item.get("keywords", []) or [],
                lead=bool(item.get("lead", False)),
                reliability_seed=float(item.get("reliability_seed", 0.7) or 0.7),
            ))
        except Exception:
            # Skip a malformed item (e.g. missing role) rather than failing the batch.
            continue
        # De-dupe ids within the batch so the review list has stable, distinct ids.
        base_id = draft["id"]
        uid, n = base_id, 2
        while uid in seen_ids:
            uid = f"{base_id}-{n}"
            n += 1
        draft["id"] = uid
        seen_ids.add(uid)
        draft["source"] = "user"
        drafts.append(draft)

    if not drafts:
        raise HTTPException(502, "No valid archetypes were produced")
    return {"archetypes": drafts}
