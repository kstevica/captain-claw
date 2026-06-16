"""Shared access to the agent archetype registry.

The *base* set lives in ``instructions/archetypes.json`` (tiers, base_tools, and
a curated ``archetypes`` list). Each user (tenant) may add their own archetypes,
stored per-user in the ``user_archetypes`` table; when a user archetype's
``id`` matches a base one it shadows the base for that user.

Every consumer that needs the archetype list — the Library/Forge gallery
(``GET /fd/archetypes``), the Forge generator, and the Basna router/executor —
goes through :func:`merged_registry` / :func:`merged_archetypes` so the base
read and the per-user overlay live in one place.

``tiers`` and ``base_tools`` are intentionally base-only: user archetypes
reference the existing tier names; defining new tiers is out of scope.
"""

from __future__ import annotations

import json
from pathlib import Path

from captain_claw.logging import get_logger

log = get_logger(__name__)

_REGISTRY_FILE = Path(__file__).parent.parent / "instructions" / "archetypes.json"


def load_base_registry() -> dict:
    """Load and parse the base archetype registry from disk.

    Raises ``FileNotFoundError`` / ``json.JSONDecodeError`` on a missing or
    invalid file — callers that must not fail (Forge catalog injection) catch
    these; the route handler surfaces them as HTTP 500.
    """
    return json.loads(_REGISTRY_FILE.read_text())


def merge_archetypes(base: list[dict], user_rows: list[dict]) -> list[dict]:
    """Overlay a user's archetypes on top of the base list.

    Each base entry is tagged ``source="base"``. Each user row (a DB row with a
    JSON ``data`` blob) is tagged ``source="user"`` and either replaces a base
    entry in place when their ``id`` matches, or is appended. Returns a new list;
    inputs are not mutated.
    """
    merged: list[dict] = []
    index: dict[str, int] = {}
    for a in base:
        entry = {**a, "source": "base"}
        index[entry.get("id", "")] = len(merged)
        merged.append(entry)

    for row in user_rows:
        try:
            data = json.loads(row.get("data") or "{}")
        except json.JSONDecodeError:
            log.warning("Skipping user archetype with invalid JSON",
                        archetype_id=row.get("archetype_id"))
            continue
        aid = row.get("archetype_id") or data.get("id") or ""
        entry = {**data, "id": aid, "source": "user"}
        if aid in index:
            entry["overrides"] = True
            merged[index[aid]] = entry
        else:
            index[aid] = len(merged)
            merged.append(entry)
    return merged


async def merged_archetypes(db, user_id: str | None) -> list[dict]:
    """Return the merged archetype list for ``user_id`` (base only if None)."""
    base = load_base_registry().get("archetypes", [])
    if not user_id:
        return [{**a, "source": "base"} for a in base]
    rows = await db.list_user_archetypes(user_id)
    return merge_archetypes(base, rows)


async def merged_registry(db, user_id: str | None) -> dict:
    """Return the full registry dict with ``archetypes`` merged for ``user_id``.

    ``tiers`` and ``base_tools`` come from the base registry unchanged.
    """
    registry = load_base_registry()
    base = registry.get("archetypes", [])
    if not user_id:
        registry["archetypes"] = [{**a, "source": "base"} for a in base]
        return registry
    rows = await db.list_user_archetypes(user_id)
    registry["archetypes"] = merge_archetypes(base, rows)
    return registry
