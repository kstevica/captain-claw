"""Owner profile routes — what a user's agents know about them.

* ``GET``/``PUT /fd/profile`` — the signed-in user's own profile (about me,
  company, standing preferences), the deck defaults it merges with, the caps
  and a preview of what the agents receive. Plain `get_current_user`: an
  admin's ``X-FD-Act-As`` does not reach another user's profile.
* ``GET``/``PUT /fd/admin/profile-defaults`` — the deck-wide company and
  instructions (admin only; on an auth-off deck the local user is the admin).
* ``GET``/``PUT /fd/profile/sources``, ``POST /fd/profile/documents``,
  ``DELETE /fd/profile/documents/{id}`` — the material behind the profile:
  the person's links, their company's name and links, their documents (a CV,
  a bio). See :mod:`captain_claw.flight_deck.profile_sources`.
* ``POST``/``GET``/``DELETE /fd/profile/research`` — the researcher that
  drafts the three texts from that material and the web, on the team's model.
  See :mod:`captain_claw.flight_deck.profile_research`.

A save rewrites the agents' context files (`tenant_profile.refresh_agents`) and
says how many agents it updated. Text that can't be encoded as UTF-8 (a lone
surrogate) is refused with 400, like an over-cap or non-string field. Storage,
merge and rendering live in :mod:`captain_claw.flight_deck.tenant_profile`.
"""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Body, Depends, File, HTTPException, UploadFile

from captain_claw.flight_deck import profile_research as pr
from captain_claw.flight_deck import profile_sources as ps
from captain_claw.flight_deck import tenant_profile as tp
from captain_claw.flight_deck.admin_routes import require_admin
from captain_claw.flight_deck.auth import get_current_user, get_db

router = APIRouter(tags=["profile"])


def _validated(body: Any, fields: tuple[str, ...]) -> dict[str, str]:
    """The fields ``body`` sets (absent / null = unchanged), each a string of
    valid text within its cap — 400 otherwise."""
    if not isinstance(body, dict):
        raise HTTPException(400, "Expected a JSON object")
    out: dict[str, str] = {}
    for field in fields:
        value = body.get(field)
        if value is None:
            continue
        if not isinstance(value, str):
            raise HTTPException(400, f"{field} must be a string")
        try:
            # JSON lets a lone surrogate (\ud800) through; it can't be written
            # to the agents' UTF-8 files, so it never gets stored either.
            value.encode("utf-8")
        except UnicodeEncodeError:
            raise HTTPException(400, f"{field} contains characters that aren't valid text")
        value = value.strip()
        if len(value) > tp.CAPS[field]:
            raise HTTPException(
                400, f"{field} is too long ({len(value)} characters, at most {tp.CAPS[field]})")
        out[field] = value
    return out


async def _profile_payload(db, user_id: str) -> dict:
    full, compact = await tp.compose_for_owner(db, user_id)
    return {
        "profile": await tp.load_profile(db, user_id),
        "deck": await tp.load_deck(db),
        "caps": dict(tp.CAPS),
        "preview": {"full": full, "compact": compact},
    }


@router.get("/fd/profile")
async def get_profile(user: dict = Depends(get_current_user)):
    return await _profile_payload(get_db(), str(user["id"]))


@router.put("/fd/profile")
async def put_profile(body: Any = Body(...), user: dict = Depends(get_current_user)):
    updates = _validated(body, tp.PROFILE_FIELDS)
    db = get_db()
    uid = str(user["id"])
    profile = await tp.load_profile(db, uid)
    profile.update(updates)
    await tp.save_profile(db, uid, profile)
    updated = await tp.refresh_agents(db, uid)
    # A profile shared on agents (context packs) is rewritten there too — not
    # counted in agents_updated, which is about the user's own agents.
    try:
        from captain_claw.flight_deck import context_packs

        await context_packs.refresh_for_user(db, uid)
    except Exception:
        pass
    return {**await _profile_payload(db, uid), "agents_updated": updated}


@router.get("/fd/admin/profile-defaults")
async def get_profile_defaults(admin: dict = Depends(require_admin)):
    deck = await tp.load_deck(get_db())
    return {**deck, "caps": {f: tp.CAPS[f] for f in tp.DECK_FIELDS}}


@router.put("/fd/admin/profile-defaults")
async def put_profile_defaults(body: Any = Body(...), admin: dict = Depends(require_admin)):
    updates = _validated(body, tp.DECK_FIELDS)
    db = get_db()
    deck = await tp.load_deck(db)
    deck.update(updates)
    await tp.save_deck(db, deck)
    updated = await tp.refresh_agents(db, None)
    try:
        await db.log_usage(str(admin.get("id") or ""), "profile_defaults_update", json.dumps({
            "fields": sorted(updates),
            "company_chars": len(deck["company"]),
            "instructions_chars": len(deck["instructions"]),
            "agents_updated": updated,
        }))
    except Exception:  # never fail the save over its own audit row
        pass
    return {"company": deck["company"], "instructions": deck["instructions"],
            "agents_updated": updated}


# ── The material behind the profile ─────────────────────────────────────────


def _owner(user: dict) -> str:
    return str(user["id"])


async def _sources_payload(db, user: dict) -> dict:
    uid = _owner(user)
    sources = await ps.load(db, uid)
    local = uid == str(tp._local_id())
    return {
        "name": str(user.get("display_name") or ""),
        "email": "" if local else str(user.get("email") or ""),
        # The auth-off deck's local user has no account to rename.
        "can_rename": tp._auth_enabled() and not local,
        **sources,
        "caps": dict(ps.CAPS),
        "kinds": list(ps.DOCUMENT_KINDS),
    }


@router.get("/fd/profile/sources")
async def get_profile_sources(user: dict = Depends(get_current_user)):
    return await _sources_payload(get_db(), user)


@router.put("/fd/profile/sources")
async def put_profile_sources(body: Any = Body(...), user: dict = Depends(get_current_user)):
    """Partial: only the fields given change (links, company_name,
    company_links). Documents change through their own routes."""
    if not isinstance(body, dict):
        raise HTTPException(400, "Expected a JSON object")
    db = get_db()
    uid = _owner(user)
    sources = await ps.load(db, uid)
    try:
        if "links" in body and body["links"] is not None:
            sources["links"] = ps.clean_links(body["links"], ps.MAX_LINKS, "links")
        if "company_links" in body and body["company_links"] is not None:
            sources["company_links"] = ps.clean_links(body["company_links"], ps.MAX_COMPANY_LINKS, "company links")
        if "company_name" in body and body["company_name"] is not None:
            sources["company_name"] = ps.clean_company_name(body["company_name"])
    except ps.SourceError as exc:
        raise HTTPException(400, str(exc))
    await ps.save(db, uid, sources)
    return await _sources_payload(db, user)


@router.post("/fd/profile/documents")
async def add_profile_documents(files: list[UploadFile] = File(default=[]),
                                user: dict = Depends(get_current_user)):
    """Upload documents (multipart ``files``): each is read in full and kept.
    One that can't be read is reported in ``warnings``, the rest are kept."""
    db = get_db()
    uid = _owner(user)
    sources = await ps.load(db, uid)
    warnings: list[str] = []
    for uf in files:
        name = uf.filename or "document"
        if len(sources["documents"]) >= ps.MAX_DOCUMENTS:
            warnings.append(f"{name}: at most {ps.MAX_DOCUMENTS} documents — remove one first.")
            continue
        try:
            raw = await uf.read()
        except Exception:
            raw = b""
        row, why = await ps.add_document(uid, name, raw)
        if row is None:
            warnings.append(why)
            continue
        sources["documents"].append(row)
    await ps.save(db, uid, sources)
    return {**await _sources_payload(db, user), "warnings": warnings}


@router.delete("/fd/profile/documents/{doc_id}")
async def remove_profile_document(doc_id: str, user: dict = Depends(get_current_user)):
    db = get_db()
    uid = _owner(user)
    sources = await ps.load(db, uid)
    before = len(sources["documents"])
    sources["documents"] = [d for d in sources["documents"] if d["id"] != doc_id]
    if len(sources["documents"]) == before:
        raise HTTPException(404, "No such document")
    ps.remove_document(uid, doc_id)
    await ps.save(db, uid, sources)
    return await _sources_payload(db, user)


# ── The researcher ──────────────────────────────────────────────────────────


@router.post("/fd/profile/research", status_code=202)
async def start_profile_research(user: dict = Depends(get_current_user)):
    try:
        job = await pr.start(get_db(), _owner(user))
    except pr.ResearchError as exc:
        live = pr.current(_owner(user))
        raise HTTPException(409 if live is not None and live.state == "running" else 400, str(exc))
    return pr.public(job)


@router.get("/fd/profile/research")
async def get_profile_research(user: dict = Depends(get_current_user)):
    return await pr.status(get_db(), _owner(user))


@router.delete("/fd/profile/research")
async def stop_profile_research(user: dict = Depends(get_current_user)):
    return {"stopped": pr.stop(_owner(user))}
