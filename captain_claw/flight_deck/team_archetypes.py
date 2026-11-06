"""Archetypes an admin generates and publishes for the whole deck.

An admin describes what their people need. Captain drafts a group (a shelf under
Company, apart from the built-in Generalists and Specialists), a family inside it
and an outline of the archetypes; the admin changes what they like; then each archetype's instructions are
written in a call of its own, so one that fails can be tried again without losing the
others, and a large set never hits a single reply's output limit. Publishing makes
them available to everyone on the deck, read-only for all but admins.

It all lives in one system setting (``archetypes.TEAM_SETTING``), so a publish lands
as one write. An agent copies its instructions when it's created, so changing or
removing a team archetype never touches agents already made from it.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from captain_claw.config import without_retired_tools
from captain_claw.flight_deck import agent_forge as af
from captain_claw.flight_deck import vatra_routes as vr
from captain_claw.flight_deck.admin_routes import require_admin
from captain_claw.flight_deck.archetype_routes import (
    _VALID_MODES,
    _VALID_TIERS,
    ArchetypeBody,
    _slugify,
    _validate,
    read_documents,
)
from captain_claw.flight_deck.archetypes import (
    TEAM_GROUP_PREFIX,
    TEAM_SETTING,
    load_base_registry,
    place_safety_rules,
    team_catalog,
)
from captain_claw.flight_deck.auth import get_db
from captain_claw.logging import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/fd/archetypes/team", tags=["archetypes"])

MAX_PACK = 12
MAX_TEAM_ARCHETYPES = 300
_MAX_REFERENCE = 60_000
_MIN_INSTRUCTIONS = 800
_DISCLAIMERS = ("legal", "health", "clinical")

# What a generated family or archetype may look like. The deck's own archetypeLook.ts
# draws exactly these: a colour needs its classes written out, and an icon its import.
COLORS = ("blue", "teal", "cyan", "amber", "violet", "sky", "rose", "red", "indigo", "pink",
          "purple", "orange", "emerald", "green", "lime", "yellow", "slate", "fuchsia")
ICONS = ("Scale", "Calculator", "Briefcase", "Landmark", "HeartPulse", "Building2", "Megaphone", "Handshake",
         "Compass", "GraduationCap", "Users", "Package", "HandHeart", "ShieldCheck", "Truck", "Radio",
         "Construction", "Sparkles", "Sprout", "Heart", "BookOpen", "House", "PiggyBank", "Bot", "Telescope",
         "PenLine", "Code2", "BarChart3", "ClipboardList", "Eye", "Target", "Microscope", "Stethoscope", "Leaf",
         "Ship", "Store", "Globe", "Database", "FileText", "Wrench", "Rocket", "Palette")


def _known_tools(registry: dict) -> tuple[list[str], set[str]]:
    """The tools every archetype has, and every tool a generated one may ask for."""
    base = list(registry.get("base_tools") or [])
    known = set(base) | {"knowledge"} | {t for a in registry.get("archetypes", []) for t in a.get("tools") or []}
    known |= {"docx", "xlsx", "pptx", "visualize", "datastore", "summarize_files", "browser", "cron", "image_ocr",
              "ocr_pdf", "ocr_image",
              "image_vision", "image_gen", "image_manipulation", "audio", "google_mail", "microsoft_mail",
              "imap_mail", "google_calendar", "microsoft_calendar"}
    return base, known


def _clean_tools(raw: Any, registry: dict) -> list[str]:
    """The basics, plus what was asked for that exists. A model can't invent a tool."""
    base, known = _known_tools(registry)
    asked = [str(t) for t in raw if str(t) in known] if isinstance(raw, list) else []
    return without_retired_tools(list(dict.fromkeys([*base, *asked])))


def _clean_icon(value: Any, fallback: str = "") -> str:
    return str(value) if str(value) in ICONS else fallback


def coerce_outline(raw: dict, registry: dict) -> dict:
    """What a model drafted, made safe to show: anything out of range becomes a default."""
    group = raw.get("group") if isinstance(raw.get("group"), dict) else {}
    family = raw.get("family") if isinstance(raw.get("family"), dict) else {}
    archetypes = []
    for item in raw.get("archetypes") or []:
        if not isinstance(item, dict) or not str(item.get("role") or "").strip():
            continue
        tier, mode = item.get("tier"), item.get("cognitive_mode")
        disclaimer = str(item.get("disclaimer") or "")
        archetypes.append({
            "role": str(item["role"]).strip()[:80],
            "description": str(item.get("description") or "").strip()[:200],
            "tier": tier if tier in _VALID_TIERS and tier != "micro" else "balanced",
            "cognitive_mode": mode if mode in _VALID_MODES else "neutra",
            "icon": _clean_icon(item.get("icon")),
            "keywords": [str(k).strip().lower() for k in item.get("keywords") or [] if str(k).strip()][:12],
            "tools": _clean_tools(item.get("tools"), registry),
            "disclaimer": disclaimer if disclaimer in _DISCLAIMERS else "",
        })
        if len(archetypes) >= MAX_PACK:
            break
    return {
        "group": {"label": str(group.get("label") or "").strip()[:40], "hint": str(group.get("hint") or "").strip()[:140]},
        "family": {"name": str(family.get("name") or "").strip()[:40],
                   "icon": _clean_icon(family.get("icon"), "Sparkles"),
                   "color": str(family.get("color")) if str(family.get("color")) in COLORS else "violet"},
        "archetypes": archetypes,
    }


async def _complete(user: dict, system: str, prompt: str, *, temperature: float, default_max: int) -> tuple[str, bool]:
    """One model call on the admin's Reasoning tier, resolved here: no key crosses the browser."""
    from captain_claw.llm import Message

    db = get_db()
    tiers, _env = await vr._load_owner_tiers(db, user["id"])
    creds = vr._resolve_creds(vr._load_registry(), tiers, "", "reason")
    limit = vr._decompose_timeout()
    try:
        provider, max_tokens = vr._provider_call(creds, temperature=temperature, default_max=default_max,
                                                 cap=vr._LEAD_OUTPUT_CAP, request_timeout=limit)
        response = await asyncio.wait_for(provider.complete(messages=[
            Message(role="system", content=system), Message(role="user", content=prompt),
        ], temperature=temperature, max_tokens=max_tokens), limit)
    except asyncio.TimeoutError as exc:
        raise HTTPException(504, "The model took too long. Try again.") from exc
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001 — the admin sees why, and can try again
        log.warning("Team archetype model call failed", error=str(exc))
        raise HTTPException(502, f"The model call failed: {exc}") from exc
    cut = str(getattr(response, "finish_reason", "") or "").lower() in ("length", "max_tokens")
    return str(response.content or "").strip(), cut


@router.post("/outline")
async def draft_outline(instructions: str = Form(""), count: int = Form(0), files: list[UploadFile] = File(default=[]),
                        user: dict = Depends(require_admin)) -> dict[str, Any]:
    """A group, a family and the archetypes in them, for the admin to change. Nothing is saved."""
    brief = instructions.strip()
    sections = await read_documents(files)
    if not brief and not sections:
        raise HTTPException(400, "Say what these agents are for, or attach a document.")
    await af._refuse_when_spent(user["id"])
    registry = load_base_registry()
    reference = "\n\n".join(sections)[:_MAX_REFERENCE]
    system = (vr._INSTRUCTIONS_DIR / "archetype_pack" / "outline.md").read_text()
    _base, known = _known_tools(registry)
    parts = [
        f"## What the admin asked for\n{brief or '(see the documents)'}",
        f"## Icons you can use\n{', '.join(ICONS)}",
        f"## Colours you can use\n{', '.join(COLORS)}",
        f"## Tools you can list\n{', '.join(sorted(known))}",
        f"## Families that already exist, to avoid repeating\n{', '.join(sorted({a.get('family', '') for a in registry.get('archetypes', [])}))}",
    ]
    wanted = max(0, min(int(count or 0), MAX_PACK))
    parts.append(f"Design exactly {wanted} archetypes." if wanted else f"Design at most {MAX_PACK} archetypes.")
    if reference:
        parts.append(f"## Documents from the admin\n{reference}")
    text, cut = await _complete(user, system, "\n\n".join(parts), temperature=0.4, default_max=8192)
    try:
        outline = coerce_outline(af._json_object(text), registry)
    except (ValueError, TypeError) as exc:
        reason = "It was cut off at the model's output limit." if cut else "It came back unreadable."
        raise HTTPException(502, f"The outline didn't come through. {reason} Try again.") from exc
    if not outline["archetypes"]:
        raise HTTPException(502, "The outline came back with no archetypes. Try again, or say more about the work.")
    return {**outline, "reference": reference, "limits": {"max_pack": MAX_PACK, "icons": ICONS, "colors": COLORS}}


class WriteBody(BaseModel):
    brief: str = Field(default="", max_length=20_000)
    reference: str = Field(default="", max_length=_MAX_REFERENCE)
    group: str = ""
    family: str = ""
    role: str
    description: str = ""
    keywords: list[str] = []
    tools: list[str] = []
    disclaimer: str = ""
    # The rest of the set: who to hand off to, and what not to overlap with.
    others: list[dict] = []


@router.post("/write")
async def write_instructions(body: WriteBody, user: dict = Depends(require_admin)) -> dict[str, str]:
    """One archetype's instructions. Nothing is saved."""
    if not body.role.strip():
        raise HTTPException(400, "It needs a role.")
    await af._refuse_when_spent(user["id"])
    registry = load_base_registry()
    _base, known = _known_tools(registry)
    system = (vr._INSTRUCTIONS_DIR / "archetype_pack" / "write.md").read_text()
    others = "\n".join(f"- `{_slugify(str(o.get('role') or ''))}`: {o.get('role')}. {o.get('description') or ''}"
                       for o in body.others if isinstance(o, dict) and o.get("role"))
    parts = [
        f"## This archetype\nRole: {body.role.strip()}\nDescription: {body.description.strip()}\n"
        f"Group: {body.group.strip()}\nFamily: {body.family.strip()}\n"
        f"Its extra tools: {', '.join(t for t in body.tools if t in known) or 'none beyond the basics'}\n"
        f"Keywords: {', '.join(body.keywords)}",
        f"## What the admin asked for\n{body.brief.strip() or '(see the documents)'}",
    ]
    if others:
        parts.append(f"## The rest of the set\n{others}")
    if body.reference.strip():
        parts.append(f"## Documents from the admin\n{body.reference.strip()}")
    text, cut = await _complete(user, system, "\n\n".join(parts), temperature=0.5, default_max=8192)
    if text.startswith("```"):
        text = "\n".join(line for line in text.split("\n") if not line.strip().startswith("```")).strip()
    if cut or len(text) < _MIN_INSTRUCTIONS or "## Hard limits" not in text:
        reason = "It was cut off at the model's output limit." if cut else "It came back incomplete."
        raise HTTPException(502, f"The instructions didn't come through. {reason} Try again.")
    disclaimer = body.disclaimer if body.disclaimer in _DISCLAIMERS else ""
    return {"fleet_instructions": place_safety_rules(text, disclaimer)}


class PackGroup(BaseModel):
    id: str = ""  # an existing shelf; empty makes a new one from the label
    label: str = ""
    hint: str = ""


class PackFamily(BaseModel):
    name: str = ""
    icon: str = ""
    color: str = ""


class PublishBody(BaseModel):
    group: PackGroup
    family: PackFamily
    archetypes: list[ArchetypeBody] = Field(max_length=MAX_PACK)


def _resolve_group(catalog: dict, group: PackGroup) -> str:
    """One of the company's own groups, never a built-in shelf: what an admin
    publishes stays apart from the generalists and specialists Captain comes with."""
    wanted = group.id.strip()
    existing = next((g for g in catalog["groups"] if g["id"] == wanted), None)
    if existing:
        return existing["id"]
    label = group.label.strip()[:40]
    if not label:
        raise HTTPException(400, "The group needs a name.")
    same = next((g for g in catalog["groups"] if g["label"].casefold() == label.casefold()), None)
    if same:
        return same["id"]
    slug = _slugify(label)[:40]
    if not slug:
        raise HTTPException(400, "The group's name needs some letters or numbers.")
    gid = f"{TEAM_GROUP_PREFIX}{slug}"
    catalog["groups"].append({"id": gid, "label": label, "hint": group.hint.strip()[:140]})
    return gid


def _remember_family(catalog: dict, family: PackFamily, registry: dict) -> str:
    name = family.name.strip()[:40]
    if not name:
        raise HTTPException(400, "The family needs a name.")
    built_in = {a.get("family") for a in registry.get("archetypes", [])}
    if name in built_in:
        return name  # it already has its look
    look = {"name": name, "icon": _clean_icon(family.icon, "Sparkles"),
            "color": family.color if family.color in COLORS else "violet"}
    catalog["families"] = [f for f in catalog["families"] if f["name"] != name] + [look]
    return name


def _prune(catalog: dict) -> None:
    """A group or family with nothing left in it goes with its last archetype."""
    groups = {a.get("category") for a in catalog["archetypes"]}
    families = {a.get("family") for a in catalog["archetypes"]}
    catalog["groups"] = [g for g in catalog["groups"] if g["id"] in groups]
    catalog["families"] = [f for f in catalog["families"] if f["name"] in families]


async def _store(catalog: dict) -> None:
    await get_db().set_system_setting(TEAM_SETTING, json.dumps(catalog))


@router.post("", status_code=201)
async def publish(body: PublishBody, user: dict = Depends(require_admin)) -> dict[str, Any]:
    """Make a reviewed set available to everyone on the deck, as one write."""
    if not body.archetypes:
        raise HTTPException(400, "There's nothing to publish.")
    catalog = await team_catalog(get_db())
    if len(catalog["archetypes"]) + len(body.archetypes) > MAX_TEAM_ARCHETYPES:
        raise HTTPException(409, f"The deck can hold {MAX_TEAM_ARCHETYPES} team archetypes. Remove some first.")
    registry = load_base_registry()
    taken = {a.get("id") for a in registry.get("archetypes", [])} | {a.get("id") for a in catalog["archetypes"]}
    made = []
    for item in body.archetypes:
        if len(item.fleet_instructions.strip()) < _MIN_INSTRUCTIONS:
            raise HTTPException(400, f"{item.role or 'One archetype'} has no instructions yet.")
    # After the checks, so a refused publish leaves no group or family behind.
    group_id = _resolve_group(catalog, body.group)
    family = _remember_family(catalog, body.family, registry)
    for item in body.archetypes:
        base_id = _slugify(item.archetype_id or item.role)
        if not base_id:
            raise HTTPException(400, "Every archetype needs a role.")
        aid, n = base_id, 2
        while aid in taken:
            aid, n = f"{base_id}-{n}", n + 1
        taken.add(aid)
        data = _validate(item.model_copy(update={
            "archetype_id": aid, "category": group_id, "family": family, "tools": _clean_tools(item.tools, registry),
            # The admin may have changed the disclaimer after these were written.
            "fleet_instructions": place_safety_rules(item.fleet_instructions, item.disclaimer),
        }), company=True)
        data["id"] = aid
        data["icon"] = _clean_icon(item.icon)
        made.append(data)
    catalog["archetypes"].extend(made)
    await _store(catalog)
    log.info("Team archetypes published", admin=user["id"], group=group_id, family=family, count=len(made))
    return {"published": len(made), "group": group_id, "family": family, "ids": [a["id"] for a in made]}


@router.put("/{archetype_id}")
async def update_one(archetype_id: str, body: ArchetypeBody, user: dict = Depends(require_admin)) -> dict[str, Any]:
    catalog = await team_catalog(get_db())
    at = next((i for i, a in enumerate(catalog["archetypes"]) if a.get("id") == archetype_id), None)
    if at is None:
        raise HTTPException(404, "There's no team archetype by that name.")
    # It moves between the company's groups, and stays in one.
    groups = {g["id"] for g in catalog["groups"]}
    group = body.category if body.category in groups else catalog["archetypes"][at].get("category")
    data = _validate(body.model_copy(update={
        "category": group,
        "tools": _clean_tools(body.tools, load_base_registry()),
        "fleet_instructions": place_safety_rules(body.fleet_instructions, body.disclaimer),
    }), company=True)
    data["id"] = archetype_id
    data["icon"] = _clean_icon(body.icon, str(catalog["archetypes"][at].get("icon") or ""))
    catalog["archetypes"][at] = data
    _prune(catalog)
    await _store(catalog)
    return data


@router.delete("/groups/{group_id}")
async def delete_group(group_id: str, user: dict = Depends(require_admin)) -> dict[str, int]:
    """A team group and everything in it. Agents already made from them keep working."""
    catalog = await team_catalog(get_db())
    if not any(g["id"] == group_id for g in catalog["groups"]):
        raise HTTPException(404, "There's no team group by that name.")
    before = len(catalog["archetypes"])
    catalog["archetypes"] = [a for a in catalog["archetypes"] if a.get("category") != group_id]
    _prune(catalog)
    await _store(catalog)
    return {"removed": before - len(catalog["archetypes"])}


@router.delete("/{archetype_id}")
async def delete_one(archetype_id: str, user: dict = Depends(require_admin)) -> dict[str, bool]:
    catalog = await team_catalog(get_db())
    kept = [a for a in catalog["archetypes"] if a.get("id") != archetype_id]
    if len(kept) == len(catalog["archetypes"]):
        raise HTTPException(404, "There's no team archetype by that name.")
    catalog["archetypes"] = kept
    _prune(catalog)
    await _store(catalog)
    return {"ok": True}
