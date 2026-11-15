"""An agent's own browser in Flight Deck: watch it, take control, hand back.

The agent does the work (``tools/browser_live.py`` behind its
``/api/browser/live|control|input``); these routes pass the person's view and
input through, for the agent's owner (or an admin) only:

- ``GET  /fd/agents/{slug}/browser/live``    — the page now (a JPEG, base64)
- ``POST /fd/agents/{slug}/browser/control`` — ``{"take": true|false}``
- ``POST /fd/agents/{slug}/browser/input``   — a click, typing, a key, a scroll

Process agents only (their port and token are in the registry).
"""

from __future__ import annotations

from typing import Any, Literal

import httpx
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from captain_claw.flight_deck.auth import get_current_user

router = APIRouter(prefix="/fd/agents", tags=["browser-live"])


def _srv():
    from captain_claw.flight_deck import server

    return server


def _agent(slug: str, user: dict) -> dict:
    """The caller's own running agent ``slug`` (any, for an admin), or 404."""
    entry = _srv()._load_process_registry().get(slug)
    uid = str(user.get("id") or "")
    if entry is None or (str(entry.get("owner") or "") != uid and user.get("role") != "admin"):
        raise HTTPException(404, "agent not found")
    if not _srv()._process_is_alive(slug):
        raise HTTPException(409, "The agent is off: its browser isn't open.")
    return entry


async def _call(entry: dict, method: str, path: str, body: dict | None = None) -> dict[str, Any]:
    port = int(entry.get("web_port") or 0)
    if not port:
        raise HTTPException(503, "The agent has no port to reach it on.")
    auth = str(entry.get("web_auth") or "")
    try:
        async with httpx.AsyncClient(timeout=15.0, trust_env=False) as client:
            resp = await client.request(method, f"http://127.0.0.1:{port}{path}",
                                        params={"token": auth} if auth else {}, json=body)
    except httpx.HTTPError as exc:
        raise HTTPException(502, f"Couldn't reach the agent: {type(exc).__name__}") from exc
    try:
        data = resp.json()
    except ValueError:
        data = {"error": resp.text[:300]}
    if resp.status_code == 404:
        raise HTTPException(501, "This agent's version can't show its browser yet: restart it to update.")
    if resp.status_code >= 400 and not isinstance(data, dict):
        raise HTTPException(502, f"The agent answered {resp.status_code}.")
    return data if isinstance(data, dict) else {}


@router.get("/{slug}/browser/live")
async def live(slug: str, user: dict = Depends(get_current_user)) -> dict[str, Any]:
    return await _call(_agent(slug, user), "GET", "/api/browser/live")


class ControlBody(BaseModel):
    take: bool


@router.post("/{slug}/browser/control")
async def control(slug: str, body: ControlBody, user: dict = Depends(get_current_user)) -> dict[str, Any]:
    by = str(user.get("display_name") or user.get("email") or "the person")
    return await _call(_agent(slug, user), "POST", "/api/browser/control", {"take": body.take, "by": by})


class InputBody(BaseModel):
    kind: Literal["click", "type", "key", "scroll"]
    x: float | None = Field(None, ge=0, le=10000)
    y: float | None = Field(None, ge=0, le=10000)
    text: str = Field("", max_length=2000)
    key: str = Field("", max_length=20)
    dy: float = Field(0, ge=-5000, le=5000)


@router.post("/{slug}/browser/input")
async def person_input(slug: str, body: InputBody, user: dict = Depends(get_current_user)) -> dict[str, Any]:
    return await _call(_agent(slug, user), "POST", "/api/browser/input", body.model_dump())
