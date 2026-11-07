"""What a person calls an agent.

The name on an agent's card is the person's own: renaming it there keeps the
new name in their settings (``fd:process-names``, a ``{slug: name}`` map synced
from the browser), not in the registry, whose ``name`` is often just the slug
an archetype spawn gave it ("deep-researcher-ze90"). Anything the deck shows a
person about an agent — a question it asks, the agents to run a tool as —
uses this, so it reads the way their own agent list does.
"""

from __future__ import annotations

import json

from captain_claw.logging import get_logger

log = get_logger(__name__)

NAMES_SETTING = "fd:process-names"


async def _renames(user_id: str) -> dict[str, str]:
    from captain_claw.flight_deck.auth import get_db

    try:
        raw = await get_db().get_setting(user_id, NAMES_SETTING)
        data = json.loads(raw) if raw else {}
    except Exception as exc:  # noqa: BLE001 — a name to show, never worth failing over
        log.debug("Couldn't read agent names", user=user_id, error=str(exc))
        return {}
    return {str(k): str(v).strip() for k, v in data.items() if str(v).strip()} if isinstance(data, dict) else {}


def _entry(slug: str) -> dict:
    from captain_claw.flight_deck import server as srv

    try:
        entry = srv._load_process_registry().get(slug)
    except Exception:  # noqa: BLE001
        entry = None
    return entry if isinstance(entry, dict) else {}


async def display_name(slug: str, viewer_id: str = "") -> str:
    """*slug*'s name as *viewer_id* knows it: their own rename, else its owner's
    (what a member of a shared agent sees it called), else the registry's
    name, else the slug."""
    if not slug:
        return ""
    entry = _entry(slug)
    for uid in dict.fromkeys(u for u in (viewer_id, str(entry.get("owner") or "")) if u):
        name = (await _renames(uid)).get(slug)
        if name:
            return name
    return str(entry.get("name") or slug)


async def display_names(viewer_id: str, slugs: list[str]) -> dict[str, str]:
    """:func:`display_name` for several of *viewer_id*'s own agents at once."""
    mine = await _renames(viewer_id) if viewer_id else {}
    return {s: mine.get(s) or str(_entry(s).get("name") or s) for s in slugs}
