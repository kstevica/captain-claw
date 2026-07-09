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


def merge_archetypes(
    base: list[dict], user_rows: list[dict],
    shared_rows: list[dict] | None = None,
) -> list[dict]:
    """Overlay a user's archetypes (and ones shared to them) on the base list.

    Each base entry is tagged ``source="base"``. Each owned user row is tagged
    ``source="user"`` and replaces a base entry in place when ids match, else is
    appended. Rows shared *to* the user are then overlaid tagged
    ``source="shared"`` (carrying the owner's id/email + permission) — but never
    over the user's own archetype of the same id (owner-of-the-run always wins
    for their own slugs). Returns a new list; inputs are not mutated.
    """
    merged: list[dict] = []
    index: dict[str, int] = {}
    for a in base:
        entry = {**a, "source": "base"}
        index[entry.get("id", "")] = len(merged)
        merged.append(entry)

    def _data(row: dict) -> dict | None:
        try:
            return json.loads(row.get("data") or "{}")
        except json.JSONDecodeError:
            log.warning("Skipping archetype with invalid JSON",
                        archetype_id=row.get("archetype_id"))
            return None

    for row in user_rows:
        data = _data(row)
        if data is None:
            continue
        aid = row.get("archetype_id") or data.get("id") or ""
        entry = {**data, "id": aid, "source": "user"}
        if aid in index:
            entry["overrides"] = True
            merged[index[aid]] = entry
        else:
            index[aid] = len(merged)
            merged.append(entry)

    for row in shared_rows or []:
        data = _data(row)
        if data is None:
            continue
        aid = row.get("archetype_id") or data.get("id") or ""
        # A grantee's own archetype of the same id always wins.
        if aid in index and merged[index[aid]].get("source") == "user":
            continue
        entry = {
            **data,
            "id": aid,
            "source": "shared",
            "shared_owner": row.get("shared_owner"),
            "shared_owner_email": row.get("shared_owner_email"),
            "shared_owner_name": row.get("shared_owner_name"),
            "shared_permission": row.get("shared_permission") or "view",
        }
        if aid in index:
            entry["overrides"] = True
            merged[index[aid]] = entry
        else:
            index[aid] = len(merged)
            merged.append(entry)
    return merged


async def _owned_and_shared(db, user_id: str) -> tuple[list[dict], list[dict]]:
    """Fetch a user's own archetype rows plus rows shared to them (best-effort)."""
    rows = await db.list_user_archetypes(user_id)
    shared: list[dict] = []
    lister = getattr(db, "list_shared_archetypes", None)
    if lister is not None:
        try:
            shared = await lister(user_id)
        except Exception:  # a share table not present shouldn't break archetypes
            shared = []
    return rows, shared


async def merged_archetypes(db, user_id: str | None) -> list[dict]:
    """Return the merged archetype list for ``user_id`` (base only if None)."""
    base = load_base_registry().get("archetypes", [])
    if not user_id:
        return [{**a, "source": "base"} for a in base]
    rows, shared = await _owned_and_shared(db, user_id)
    return merge_archetypes(base, rows, shared)


async def merged_registry(db, user_id: str | None) -> dict:
    """Return the full registry dict with ``archetypes`` merged for ``user_id``.

    ``tiers`` and ``base_tools`` come from the base registry unchanged.
    """
    registry = load_base_registry()
    base = registry.get("archetypes", [])
    if not user_id:
        registry["archetypes"] = [{**a, "source": "base"} for a in base]
        return registry
    rows, shared = await _owned_and_shared(db, user_id)
    registry["archetypes"] = merge_archetypes(base, rows, shared)
    return registry
