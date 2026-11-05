"""Quick start tier sets: pick a provider, paste its key, get its current models.

The catalog is ``captain_claw/instructions/tier_presets.json`` — per provider,
the model, context and output each tier runs on. A model release is an edit
there. The console's New tier set and the Library's setup wizard build a tier
set from it in the browser (the set is the person's, like every other set);
this module only serves the catalog and checks a key before the set is made.

* ``GET  /fd/tier-presets``        the catalog, without the check URLs.
* ``POST /fd/tier-presets/check``  asks the preset's provider which models the
  key reaches. ``ok`` with the preset's models it doesn't list (``missing``),
  ``refused`` when the provider turns the key down, ``unreachable`` when it
  couldn't be asked. The URL comes from the catalog, never from the caller,
  and the key is not kept.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from captain_claw.flight_deck.auth import get_current_user
from captain_claw.logging import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/fd/tier-presets", tags=["tier-presets"])

PRESETS_FILE = Path(__file__).resolve().parent.parent / "instructions" / "tier_presets.json"
_TIMEOUT = httpx.Timeout(10.0, connect=5.0)
_AUTH_STYLES = ("bearer", "x-api-key", "x-goog-api-key")
# Fields of a tier a preset may set; anything else in the file is ignored.
_TIER_FIELDS = ("model", "input_ctx", "output_ctx", "vision", "weak")


def load_presets() -> list[dict[str, Any]]:
    """The catalog's presets, each with its check endpoint. Unreadable → none."""
    try:
        data = json.loads(PRESETS_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        log.warning("Tier presets unreadable", path=str(PRESETS_FILE), error=str(exc))
        return []
    out: list[dict[str, Any]] = []
    for p in data.get("presets") or []:
        if not isinstance(p, dict) or not p.get("id") or not p.get("provider"):
            continue
        tiers = {
            name: {k: t[k] for k in _TIER_FIELDS if k in t}
            for name, t in (p.get("tiers") or {}).items()
            if isinstance(t, dict) and str(t.get("model") or "").strip()
        }
        if not tiers:
            continue
        out.append({**p, "tiers": tiers})
    return out


def _public(p: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in p.items() if k != "check"}


def checked_on() -> str:
    try:
        return str(json.loads(PRESETS_FILE.read_text(encoding="utf-8")).get("_meta", {}).get("checked") or "")
    except (OSError, ValueError):
        return ""


@router.get("")
async def list_presets(_user: dict = Depends(get_current_user)) -> dict:
    return {"presets": [_public(p) for p in load_presets()], "checked": checked_on()}


# ── key check ───────────────────────────────────────────────────────


def _headers(auth: str, key: str) -> dict[str, str]:
    if auth == "x-api-key":
        return {"x-api-key": key, "anthropic-version": "2023-06-01"}
    if auth == "x-goog-api-key":
        return {"x-goog-api-key": key}
    return {"Authorization": f"Bearer {key}"}


def _listed_ids(data: Any) -> list[str]:
    """Model ids from an OpenAI-style ``data`` list or Gemini's ``models``."""
    if not isinstance(data, dict):
        return []
    rows = data.get("data") if isinstance(data.get("data"), list) else data.get("models")
    ids: list[str] = []
    for r in rows or []:
        if isinstance(r, dict):
            mid = str(r.get("id") or r.get("name") or "").strip()
            if mid:
                ids.append(mid.removeprefix("models/"))
    return ids


def _offered(model: str, ids: list[str]) -> bool:
    """A listed id is the model, or a dated / versioned snapshot of it."""
    m = model.lower()
    return any(i.lower() == m or i.lower().startswith((f"{m}-", f"{m}@")) for i in ids)


def missing_models(preset: dict[str, Any], ids: list[str]) -> list[str]:
    """The preset's models the provider's list doesn't have. An empty list
    (nothing parsed) proves nothing, so nothing is reported missing."""
    if not ids:
        return []
    wanted = dict.fromkeys(str(t["model"]) for t in preset["tiers"].values())
    return [m for m in wanted if not _offered(m, ids)]


class CheckBody(BaseModel):
    preset: str = Field(min_length=1, max_length=64)
    api_key: str = Field(min_length=1, max_length=512)


@router.post("/check")
async def check_key(body: CheckBody, _user: dict = Depends(get_current_user)) -> dict:
    preset = next((p for p in load_presets() if p["id"] == body.preset), None)
    if preset is None:
        raise HTTPException(404, "No such quick start")
    check = preset.get("check") if isinstance(preset.get("check"), dict) else {}
    url = str(check.get("url") or "")
    auth = str(check.get("auth") or "bearer")
    if not url.startswith("https://") or auth not in _AUTH_STYLES:
        return {"ok": True, "checked": False, "missing": []}
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.get(url, headers=_headers(auth, body.api_key.strip()))
    except httpx.HTTPError as exc:
        log.info("Tier preset key check unreachable", preset=preset["id"], error=type(exc).__name__)
        return {"ok": False, "unreachable": True}
    if resp.status_code in (400, 401, 403):
        return {"ok": False, "refused": True, "status": resp.status_code}
    if resp.status_code >= 400:
        return {"ok": False, "unreachable": True, "status": resp.status_code}
    try:
        ids = _listed_ids(resp.json())
    except ValueError:
        ids = []
    return {"ok": True, "checked": True, "missing": missing_models(preset, ids)}
