"""Models agents can switch to on an endpoint no catalog knows.

An agent switches among its own provider's models — same key, same endpoint
(``captain_claw/llm/model_switch.py``). For OpenAI, Anthropic, Gemini, Grok and
DeepSeek the tier presets catalog says which models those are. For any other
endpoint — a team gateway, a local server, OpenRouter — an admin lists them
here; nothing about tier sets changes.

* ``GET  /fd/model-switch/endpoints``     the admin's list (admin only).
* ``PUT  /fd/model-switch/endpoints``     replace it (admin only).
* ``GET  /fd/model-switch/agent/models``  for an agent: the models listed for
  the endpoint it names (``provider`` + ``base_url``), matched by the shared
  "same endpoint" rule. Only this deck's agents are answered.

Stored as the system setting ``fd:model-switch-endpoints``:
``{"endpoints": [{"provider", "base_url", "models": [{"model", "note"}]}]}``.
Model names and endpoint addresses only — never a key.
"""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from captain_claw.flight_deck.admin_routes import require_admin
from captain_claw.flight_deck.auth import get_db
from captain_claw.flight_deck.endpoints import endpoint_of, same_endpoint
from captain_claw.logging import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/fd/model-switch", tags=["model-switch"])

SETTING = "fd:model-switch-endpoints"
MAX_ENDPOINTS = 50
MAX_MODELS = 60
_MAX_MODEL_LEN = 200
_MAX_NOTE_LEN = 200
_MAX_URL_LEN = 500


class SwitchModel(BaseModel):
    model: str = ""
    note: str = ""


class SwitchEndpoint(BaseModel):
    provider: str = ""
    base_url: str = ""
    models: list[SwitchModel] = Field(default_factory=list)


class SwitchEndpoints(BaseModel):
    endpoints: list[SwitchEndpoint] = Field(default_factory=list)


def _norm_provider(provider: str) -> str:
    return endpoint_of(provider, "")[0]


def clean_endpoints(raw: Any) -> list[dict[str, Any]]:
    """The stored list, tidied: known providers, unique endpoints (by the
    "same endpoint" rule; a repeat's models join the first), unique models,
    capped sizes. Unusable rows are dropped."""
    from captain_claw.llm import is_supported_provider

    rows = raw if isinstance(raw, list) else []
    out: list[dict[str, Any]] = []
    for row in rows:
        if isinstance(row, BaseModel):
            row = row.model_dump()
        if not isinstance(row, dict):
            continue
        provider = _norm_provider(str(row.get("provider") or ""))
        base_url = str(row.get("base_url") or "").strip()[:_MAX_URL_LEN]
        if not provider or not is_supported_provider(provider):
            continue
        target = next((e for e in out if same_endpoint(e["provider"], e["base_url"], provider, base_url)), None)
        if target is None:
            if len(out) >= MAX_ENDPOINTS:
                continue
            target = {"provider": provider, "base_url": base_url, "models": []}
            out.append(target)
        seen = {m["model"].lower() for m in target["models"]}
        for item in row.get("models") or []:
            if isinstance(item, BaseModel):
                item = item.model_dump()
            if isinstance(item, str):
                item = {"model": item}
            if not isinstance(item, dict):
                continue
            model = str(item.get("model") or "").strip()[:_MAX_MODEL_LEN]
            if not model or model.lower() in seen or len(target["models"]) >= MAX_MODELS:
                continue
            seen.add(model.lower())
            target["models"].append({"model": model, "note": str(item.get("note") or "").strip()[:_MAX_NOTE_LEN]})
    return [e for e in out if e["models"]]


async def load_endpoints() -> list[dict[str, Any]]:
    """The admin's list ([] when none, or no database)."""
    try:
        raw = await get_db().get_system_setting(SETTING)
    except Exception as exc:
        log.debug("Model switch endpoints unreadable", error=str(exc))
        return []
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except (TypeError, ValueError):
        return []
    return clean_endpoints(data.get("endpoints") if isinstance(data, dict) else None)


def models_for(endpoints: list[dict[str, Any]], provider: str, base_url: str) -> list[dict[str, str]]:
    """The models listed for the endpoint ``provider`` + ``base_url`` names."""
    for row in endpoints:
        if same_endpoint(row.get("provider"), row.get("base_url"), provider, base_url):
            return list(row.get("models") or [])
    return []


@router.get("/endpoints")
async def get_endpoints(admin: dict = Depends(require_admin)) -> dict:
    """The models agents can switch to, per endpoint."""
    return {"endpoints": await load_endpoints()}


@router.put("/endpoints")
async def put_endpoints(body: SwitchEndpoints, admin: dict = Depends(require_admin)) -> dict:
    """Replace the list. Agents pick it up within minutes (at their next turn
    after ten minutes, or when they look) — no restart."""
    endpoints = clean_endpoints(body.endpoints)
    try:
        await get_db().set_system_setting(SETTING, json.dumps({"endpoints": endpoints}))
    except Exception as exc:
        raise HTTPException(503, f"Couldn't save the model list: {exc}") from exc
    return {"ok": True, "endpoints": endpoints}


@router.get("/agent/models")
async def agent_models(request: Request, provider: str = "", base_url: str = "") -> dict:
    """For one of this deck's agents: the models an admin listed for the
    endpoint it runs on."""
    from captain_claw.flight_deck import google_oauth_routes as _g
    from captain_claw.flight_deck.server import _find_agent_by_auth

    _g._authorize_agent_call(request)
    matched, _owner, _slug = _find_agent_by_auth(request.headers.get("X-Agent-Auth", ""))
    if not matched:
        raise HTTPException(403, "This Flight Deck doesn't know this agent")
    if not str(provider or "").strip():
        return {"models": []}
    return {"models": models_for(await load_endpoints(), provider, base_url)}
