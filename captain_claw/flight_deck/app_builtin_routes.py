"""Built-in tool catalogue for agent apps with no external MCP server.

When a manifest's ``agent.mcp_server`` is the reserved value
``__framework__`` (or empty/missing), the app runtime dispatches tool
calls here instead of through the MCP proxy. The framework provides a
small CRUD-shaped catalogue backed by :mod:`app_entities` and
:mod:`app_files`, which is enough to let a generated app store and
display data without forcing the user to provision an MCP server.

The response shape matches the MCP user_call endpoint so the renderer's
``extractRows`` and friends work unchanged.
"""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Body, Depends, HTTPException

from captain_claw.flight_deck import app_entities, app_files, app_manifests
from captain_claw.flight_deck.auth import get_current_user


router = APIRouter(prefix="/fd/apps", tags=["apps"])


FRAMEWORK_SERVER_NAME = "__framework__"


def _require_app(agent_id: str) -> None:
    if app_manifests.get(agent_id) is None:
        raise HTTPException(status_code=404, detail=f"No app '{agent_id}'")


def _wrap(payload: Any) -> dict[str, Any]:
    """Shape data like an MCP tool result so the renderer can extract it."""
    text = json.dumps(payload, default=str)
    return {
        "content": [{"type": "text", "text": text}],
        "structuredContent": payload,
    }


def _str_arg(args: dict, key: str, *, required: bool = True) -> str:
    val = args.get(key)
    if val is None or val == "":
        if required:
            raise HTTPException(status_code=400, detail=f"{key} is required")
        return ""
    return str(val)


# ── tool handlers ─────────────────────────────────────────────────────


def _entities_list(agent_id: str, args: dict) -> dict:
    entity = _str_arg(args, "entity")
    rows = app_entities.get_store().list(agent_id, entity)
    return _wrap({"items": rows})


def _entities_get(agent_id: str, args: dict) -> dict:
    entity = _str_arg(args, "entity")
    record_id = _str_arg(args, "id")
    rec = app_entities.get_store().get(agent_id, entity, record_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="No such record")
    return _wrap(rec)


def _entities_create(agent_id: str, args: dict) -> dict:
    entity = _str_arg(args, "entity")
    data = args.get("data")
    if not isinstance(data, dict):
        # Allow flat-style calls where the record fields are passed
        # alongside ``entity`` — convenient for the LLM-generated args.
        data = {k: v for k, v in args.items() if k != "entity"}
    rec = app_entities.get_store().create(agent_id, entity, data)
    return _wrap(rec)


def _entities_update(agent_id: str, args: dict) -> dict:
    entity = _str_arg(args, "entity")
    record_id = _str_arg(args, "id")
    data = args.get("data")
    if not isinstance(data, dict):
        data = {k: v for k, v in args.items() if k not in ("entity", "id")}
    rec = app_entities.get_store().update(agent_id, entity, record_id, data)
    if rec is None:
        raise HTTPException(status_code=404, detail="No such record")
    return _wrap(rec)


def _entities_delete(agent_id: str, args: dict) -> dict:
    entity = _str_arg(args, "entity")
    record_id = _str_arg(args, "id")
    ok = app_entities.get_store().delete(agent_id, entity, record_id)
    if not ok:
        raise HTTPException(status_code=404, detail="No such record")
    return _wrap({"ok": True, "id": record_id})


def _files_list(agent_id: str, _args: dict) -> dict:
    metas = app_files.get_store().list(agent_id)
    return _wrap({"items": [m.model_dump() for m in metas]})


_HANDLERS = {
    "entities_list": _entities_list,
    "entities_get": _entities_get,
    "entities_create": _entities_create,
    "entities_update": _entities_update,
    "entities_delete": _entities_delete,
    "files_list": _files_list,
}


# ── routes ────────────────────────────────────────────────────────────


@router.post("/{agent_id}/builtin/call")
async def builtin_call(
    agent_id: str,
    payload: dict = Body(...),
    _user: dict = Depends(get_current_user),
) -> dict[str, Any]:
    """Dispatch a built-in tool call for an app.

    Body shape mirrors MCP's user_call:
        { "tool": "entities_list", "arguments": { "entity": "post" } }
    """
    _require_app(agent_id)
    tool_name = str(payload.get("tool") or payload.get("name") or "").strip()
    if not tool_name:
        raise HTTPException(status_code=400, detail="payload.tool is required")
    arguments = payload.get("arguments") or {}
    if not isinstance(arguments, dict):
        raise HTTPException(status_code=400, detail="arguments must be an object")

    handler = _HANDLERS.get(tool_name)
    if handler is None:
        raise HTTPException(
            status_code=404,
            detail=f"No built-in tool '{tool_name}'. Available: {sorted(_HANDLERS)}",
        )
    result = handler(agent_id, arguments)
    return {"server": FRAMEWORK_SERVER_NAME, "tool": tool_name, "result": result}


@router.get("/{agent_id}/builtin/tools")
async def builtin_list_tools(
    agent_id: str,
    _user: dict = Depends(get_current_user),
) -> dict[str, Any]:
    """Catalogue of built-in tools — useful for the authoring prompt."""
    _require_app(agent_id)
    return {"server": FRAMEWORK_SERVER_NAME, "tools": sorted(_HANDLERS)}
