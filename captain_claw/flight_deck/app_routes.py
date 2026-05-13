"""HTTP routes for agent-app manifests.

The Flight-Deck app runtime in the browser fetches manifests from
here. Manifests are owned by Captain Claw (the framework), not by the
renderer — the renderer is just one consumer.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Body, Depends, HTTPException

from captain_claw.flight_deck import app_manifest_authoring, app_manifests
from captain_claw.flight_deck.auth import get_current_user


router = APIRouter(prefix="/fd/apps", tags=["apps"])


@router.get("")
async def list_apps(_user: dict = Depends(get_current_user)) -> dict[str, Any]:
    """Return ``[{id, name, tagline}]`` for every registered manifest."""
    return {"apps": app_manifests.list_summaries()}


@router.get("/{agent_id}")
async def get_app(
    agent_id: str,
    _user: dict = Depends(get_current_user),
) -> dict[str, Any]:
    """Return the full manifest for ``agent_id``."""
    m = app_manifests.get(agent_id)
    if m is None:
        raise HTTPException(status_code=404, detail=f"No manifest for '{agent_id}'")
    return {"manifest": m.model_dump(exclude_none=False)}


# ── authoring: NL → manifest ──────────────────────────────────────────


@router.post("/generate")
async def generate_app(
    payload: dict = Body(...),
    _user: dict = Depends(get_current_user),
) -> dict[str, Any]:
    """Generate (or revise) a manifest from a natural-language description.

    Body::

        {
          "description": str,
          "mcp_server"?: str,
          "base_agent_id"?: str,
          "agent"?: { "host": str, "port": int, "auth": str, "name"?: str }
        }

    When ``agent`` is given, the manifest LLM call is routed through that
    Captain Claw agent's ``/api/llm/complete`` endpoint (so it uses the
    agent's configured model + credentials). Without it, we fall back to
    a direct litellm call which needs vendor keys in the environment.

    Returns ``{manifest: <object|null>, errors: list[str]}``. The
    manifest is returned even when ``errors`` is non-empty so the UI
    can show what the LLM produced and let the user iterate.
    """
    description = str(payload.get("description") or "").strip()
    if not description:
        raise HTTPException(status_code=400, detail="description is required")
    mcp_server = payload.get("mcp_server")
    if mcp_server is not None and not isinstance(mcp_server, str):
        raise HTTPException(status_code=400, detail="mcp_server must be a string")

    agent_target: app_manifest_authoring.AgentTarget | None = None
    raw_agent = payload.get("agent")
    if raw_agent is not None:
        if not isinstance(raw_agent, dict):
            raise HTTPException(status_code=400, detail="agent must be an object")
        try:
            agent_target = app_manifest_authoring.AgentTarget.model_validate(raw_agent)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"invalid agent: {exc}")

    current: dict[str, Any] | None = None
    base_id = payload.get("base_agent_id")
    if base_id:
        existing = app_manifests.get(str(base_id))
        if existing is not None:
            current = existing.model_dump(exclude_none=False)

    return await app_manifest_authoring.generate(
        description,
        mcp_server=mcp_server or None,
        current_manifest=current,
        agent=agent_target,
    )


@router.post("/save")
async def save_app(
    payload: dict = Body(...),
    _user: dict = Depends(get_current_user),
) -> dict[str, Any]:
    """Validate and persist a manifest dict. Body: ``{manifest: <dict>}``."""
    manifest = payload.get("manifest")
    if not isinstance(manifest, dict):
        raise HTTPException(status_code=400, detail="manifest must be an object")
    return app_manifest_authoring.save_validated(manifest)


@router.delete("/{agent_id}")
async def delete_app(
    agent_id: str,
    _user: dict = Depends(get_current_user),
) -> dict[str, Any]:
    """Delete a manifest from disk."""
    try:
        removed = app_manifests.delete(agent_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    if not removed:
        raise HTTPException(status_code=404, detail=f"No manifest for '{agent_id}'")
    return {"ok": True}
