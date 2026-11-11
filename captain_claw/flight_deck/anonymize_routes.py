"""Anonymization routes — ``/fd/anonymize/*``.

The engine (:mod:`captain_claw.anonymize`) runs here, next to the vaults
(:mod:`anonymize_vault`), so agents and the person get the same tokens for the
same case and the real values never leave Flight Deck except back to the
person who owns them.

Agent routes (an agent's ``X-Agent-Auth``; the vault's person is the agent's
owner, or — on a shared agent's member turn — the member, never the owner):

    POST /agent/run       {content, format, vault, mode, kinds, keep, extra, filename, source}
    POST /agent/restore   {content, format, vault} → the restored text, for the tool to write
                          to a file (the tool never hands it to the model)
    GET  /agent/vaults    names and counts — never values

Person routes (signed in):

    GET    /kinds
    POST   /run, /restore           same bodies as the agent's
    GET    /vaults                  the person's vaults
    GET    /vaults/{vault}          with every real value (logged as a view)
    PATCH  /vaults/{vault}          {name?, expires_at? (unix seconds or null)}
    DELETE /vaults/{vault}
    GET    /vaults/{vault}/events

Request bodies are read by hand: a malformed body must never reach FastAPI's
validation handler, which logs and echoes the raw input — private data.
"""

from __future__ import annotations

import asyncio
import re
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request

from captain_claw.anonymize import engine
from captain_claw.anonymize import formats
from captain_claw.anonymize.detectors import KINDS
from captain_claw.flight_deck import anonymize_vault as av
from captain_claw.flight_deck import fd_crypto
from captain_claw.flight_deck.auth import _LOCAL_USER, _fd_auth_enabled, get_current_user
from captain_claw.logging import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/fd/anonymize", tags=["anonymize"])

MAX_CHARS = 2_000_000
MAX_SEGMENTS = 200_000
MAX_TERMS = 500
_CUSTOM_KIND_RE = re.compile(r"^[A-Z][A-Z0-9_]{1,30}$")

KIND_LABELS = {
    "PERSON": "People's names", "ORG": "Companies and organisations", "EMAIL": "Email addresses",
    "PHONE": "Phone numbers", "ADDRESS": "Street addresses", "DOB": "Dates of birth",
    "IBAN": "Bank accounts (IBAN)", "CARD": "Payment card numbers", "CVV": "Card security codes",
    "CARD_EXPIRY": "Card expiry dates", "OIB": "OIB", "JMBG": "JMBG", "ID_NUMBER": "ID, passport, tax numbers",
    "SSN": "US social security numbers", "IP": "IP addresses", "PROFILE_URL": "Social profile links",
    "PLATE": "Licence plates",
}


async def _body(request: Request) -> dict[str, Any]:
    try:
        data = await request.json()
    except Exception:
        data = None
    return data if isinstance(data, dict) else {}


def _str_list(value: Any, what: str, max_len: int = 300) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise HTTPException(status_code=422, detail=f"{what} must be a list")
    out = [str(v).strip() for v in value if str(v or "").strip()]
    if len(out) > MAX_TERMS or any(len(v) > max_len for v in out):
        raise HTTPException(status_code=422, detail=f"{what}: at most {MAX_TERMS} values of {max_len} characters")
    return out


def _options(data: dict[str, Any]) -> engine.Options:
    kinds_in = _str_list(data.get("kinds"), "kinds", 40)
    unknown = [k for k in kinds_in if k.upper() not in KINDS]
    if unknown:
        raise HTTPException(status_code=422, detail=f"unknown kinds: {', '.join(unknown)} (known: {', '.join(KINDS)})")
    extra: list[tuple[str, str]] = []
    raw_extra = data.get("extra")
    if raw_extra is not None:
        if not isinstance(raw_extra, list) or len(raw_extra) > MAX_TERMS:
            raise HTTPException(status_code=422, detail=f"extra must be a list of at most {MAX_TERMS} items")
        for item in raw_extra:
            if isinstance(item, str):
                text, kind = item, "PERSON"
            elif isinstance(item, dict):
                text, kind = str(item.get("text") or ""), str(item.get("kind") or "PERSON")
            else:
                continue
            kind = kind.strip().upper()
            if not _CUSTOM_KIND_RE.match(kind):
                raise HTTPException(status_code=422, detail="an extra term's kind is upper-case letters, digits and _")
            if text.strip() and len(text) <= 300:
                extra.append((text.strip(), kind))
    kinds = frozenset(k.upper() for k in kinds_in) | frozenset(k for _t, k in extra) if kinds_in else None
    cc = str(data.get("default_country_code") or "385").lstrip("+")
    if not re.fullmatch(r"\d{1,3}", cc):
        raise HTTPException(status_code=422, detail="default_country_code is 1–3 digits")
    return engine.Options(kinds=kinds, keep=tuple(_str_list(data.get("keep"), "keep")), extra=tuple(extra),
                          default_cc=cc)


def _content(data: dict[str, Any]) -> tuple[str, str]:
    content = data.get("content")
    if not isinstance(content, str):
        raise HTTPException(status_code=422, detail="content (text) is required")
    if len(content) > MAX_CHARS:
        raise HTTPException(status_code=413, detail=f"content is over {MAX_CHARS:,} characters — split it")
    fmt = str(data.get("format") or "text").lower()
    return content, fmt if fmt in formats.TEXT_FORMATS else "text"


def _prepare(content: str, fmt: str) -> formats.Prepared:
    try:
        prepared = formats.prepare(content, fmt)
    except formats.FormatError as exc:
        raise HTTPException(status_code=422, detail=f"{fmt}: {exc}") from None
    if len(prepared.segments) > MAX_SEGMENTS:
        raise HTTPException(status_code=413, detail=f"over {MAX_SEGMENTS:,} cells / values — split it")
    return prepared


def _store_error(exc: Exception) -> HTTPException:
    if isinstance(exc, fd_crypto.CryptoUnavailable):
        return HTTPException(status_code=503, detail=f"Vaults can't be used: {exc}. mode 'one_way' still works.")
    return HTTPException(status_code=422, detail=str(exc))


# ── the work, shared by both route sets ─────────────────────────────


async def run_for(person: str, data: dict[str, Any], *, actor: str) -> dict[str, Any]:
    content, fmt = _content(data)
    options = _options(data)
    mode = str(data.get("mode") or "reversible")
    if mode not in ("reversible", "one_way"):
        raise HTTPException(status_code=422, detail="mode is 'reversible' or 'one_way'")
    prepared = _prepare(content, fmt)
    stem = str(data.get("filename") or "")[:255]
    source = str(data.get("source") or "")
    store = av.store()
    vault_row: dict[str, Any] | None = None

    def work(state: engine.VaultState) -> tuple[engine.RunResult, str]:
        result = engine.run(prepared.segments, state, options)
        name = engine.anonymize_filename(stem, state, options) if stem else ""
        return result, name

    try:
        if mode == "one_way":
            result, out_name = await asyncio.to_thread(work, engine.VaultState())
        else:
            vault_row = await store.find_or_create(person, str(data.get("vault") or "default"), created_by=actor)
            async with store.lock(vault_row["id"]):
                state = await store.load_state(vault_row)
                result, out_name = await asyncio.to_thread(work, state)
                await store.save_run(vault_row, state)
    except engine.VerifyFailed as exc:
        log.warning("Anonymization refused its own output", kinds=exc.kinds, actor=actor)
        raise HTTPException(status_code=422, detail=str(exc)) from None
    except (av.VaultError, fd_crypto.CryptoUnavailable) as exc:
        raise _store_error(exc) from None

    output = prepared.rebuild(result.outputs)
    counts = engine.summary(result.used)
    if vault_row is not None:
        await store.log(person, vault_row["id"], actor, "anonymize", {
            "format": prepared.format, "source": source, "entities": counts,
            "new": sum(1 for e in result.used if e.new), "fixed": result.fixed,
        })
    return {
        "vault": {"id": vault_row["id"], "name": vault_row["name"]} if vault_row else None,
        "mode": mode,
        "format": prepared.format,
        "output": output,
        "filename": out_name,
        "entities": [{"token": e.token, "kind": e.kind, "count": e.count, "new": e.new} for e in result.used],
        "summary": counts,
        "verify": {"ok": True, "rounds": result.rounds, "fixed": result.fixed},
        "lanes": {"deterministic": "ok", "llm": "not available yet"},
        "already_exposed": source == "text",
    }


async def restore_for(person: str, data: dict[str, Any], *, actor: str) -> dict[str, Any]:
    content, fmt = _content(data)
    name = str(data.get("vault") or "").strip()
    if not name:
        raise HTTPException(status_code=422, detail="vault is required — the one the text was anonymized into")
    store = av.store()
    try:
        row = await store.get(person, name)
    except fd_crypto.CryptoUnavailable as exc:
        raise _store_error(exc) from None
    if row is None:
        raise HTTPException(status_code=404, detail=f"No vault '{name}'")
    prepared = _prepare(content, fmt)
    try:
        state = await store.load_state(row)
    except fd_crypto.CryptoUnavailable as exc:
        raise _store_error(exc) from None
    total = 0
    unknown: list[str] = []
    outs = []
    for seg in prepared.segments:
        text, n, unk = engine.restore(seg.text, state)
        outs.append(text)
        total += n
        unknown.extend(u for u in unk if u not in unknown)
    await store.touch(row)
    await store.log(person, row["id"], actor, "restore", {"restored": total, "unknown": len(unknown)})
    return {"vault": {"id": row["id"], "name": row["name"]}, "output": prepared.rebuild(outs),
            "restored": total, "unknown": unknown}


# ── agent routes ────────────────────────────────────────────────────


def _agent_slug(request: Request) -> str:
    try:
        from captain_claw.flight_deck import server as srv

        matched, _owner, slug = srv._find_agent_by_auth(request.headers.get("X-Agent-Auth", ""))
        return str(slug) if matched and slug else ""
    except Exception:
        return ""


async def _agent_person(request: Request) -> tuple[str, str]:
    """``(person, actor)``: whose vaults this agent call uses — the member on a
    shared agent's member turn, else the agent's owner — resolved from this
    deck's own records, never from the agent's word."""
    from captain_claw.flight_deck import speaker_grants

    acting = await speaker_grants.acting_member(request)
    if acting is not None:
        return acting.user_id, f"agent:{acting.slug} (member)"
    if _fd_auth_enabled():
        from captain_claw.flight_deck import google_oauth_routes as _g

        _g._authorize_agent_call(request)
        owner = await _g._agent_owner(request, service="Anonymization")
        return owner, f"agent:{_agent_slug(request)}"
    from captain_claw.flight_deck.agent_identity import require_agent_caller

    caller = require_agent_caller(request, what="Anonymization")
    return _LOCAL_USER["id"], f"agent:{_agent_slug(request) or caller.slug}"


@router.post("/agent/run")
async def anonymize_agent_run(request: Request) -> dict[str, Any]:
    person, actor = await _agent_person(request)
    return await run_for(person, await _body(request), actor=actor)


@router.post("/agent/restore")
async def anonymize_agent_restore(request: Request) -> dict[str, Any]:
    person, actor = await _agent_person(request)
    return await restore_for(person, await _body(request), actor=actor)


@router.get("/agent/vaults")
async def anonymize_agent_vaults(request: Request) -> dict[str, Any]:
    person, _actor = await _agent_person(request)
    return {"vaults": await av.store().list_vaults(person)}


# ── person routes ───────────────────────────────────────────────────


@router.get("/kinds")
async def anonymize_kinds(_user: dict = Depends(get_current_user)) -> dict[str, Any]:
    return {"kinds": [{"kind": k, "label": KIND_LABELS.get(k, k)} for k in KINDS], "formats": list(formats.TEXT_FORMATS)}


@router.post("/run")
async def anonymize_run(request: Request, user: dict = Depends(get_current_user)) -> dict[str, Any]:
    return await run_for(user["id"], await _body(request), actor="person")


@router.post("/restore")
async def anonymize_restore(request: Request, user: dict = Depends(get_current_user)) -> dict[str, Any]:
    return await restore_for(user["id"], await _body(request), actor="person")


@router.get("/vaults")
async def anonymize_vaults(user: dict = Depends(get_current_user)) -> dict[str, Any]:
    return {"vaults": await av.store().list_vaults(user["id"])}


async def _own(user: dict, vault: str) -> dict[str, Any]:
    try:
        row = await av.store().get(user["id"], vault)
    except fd_crypto.CryptoUnavailable as exc:
        raise _store_error(exc) from None
    if row is None:
        raise HTTPException(status_code=404, detail="No such vault")
    return row


@router.get("/vaults/{vault}")
async def anonymize_vault_detail(vault: str, user: dict = Depends(get_current_user)) -> dict[str, Any]:
    row = await _own(user, vault)
    store = av.store()
    try:
        entries = await store.entries(row)
    except fd_crypto.CryptoUnavailable as exc:
        raise _store_error(exc) from None
    await store.log(user["id"], row["id"], "person", "view", {"entities": len(entries)})
    return {"vault": av._public(row), "entries": entries}


@router.patch("/vaults/{vault}")
async def anonymize_vault_update(vault: str, request: Request, user: dict = Depends(get_current_user)) -> dict[str, Any]:
    row = await _own(user, vault)
    data = await _body(request)
    expires: float | None | str = "keep"
    if "expires_at" in data:
        value = data["expires_at"]
        if value is not None and not isinstance(value, (int, float)):
            raise HTTPException(status_code=422, detail="expires_at is unix seconds, or null for never")
        expires = None if value is None else float(value)
    try:
        updated = await av.store().update(user["id"], row["id"], name=data.get("name"), expires_at=expires)
    except av.VaultError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    if updated is None:
        raise HTTPException(status_code=404, detail="No such vault")
    await av.store().log(user["id"], row["id"], "person", "update",
                         {k: True for k in ("name", "expires_at") if k in data})
    return {"vault": av._public(updated)}


@router.delete("/vaults/{vault}")
async def anonymize_vault_delete(vault: str, user: dict = Depends(get_current_user)) -> dict[str, Any]:
    row = await _own(user, vault)
    await av.store().delete(user["id"], row["id"], actor="person")
    return {"deleted": row["id"]}


@router.get("/vaults/{vault}/events")
async def anonymize_vault_events(vault: str, user: dict = Depends(get_current_user)) -> dict[str, Any]:
    row = await _own(user, vault)
    return {"events": await av.store().events(user["id"], row["id"])}
