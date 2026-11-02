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
import re
from collections.abc import Iterable
from pathlib import Path

from captain_claw.logging import get_logger

log = get_logger(__name__)

_REGISTRY_FILE = Path(__file__).parent.parent / "instructions" / "archetypes.json"

# The gallery's three shelves, above the families. An archetype without one (a
# person's own, or a row stored before there were any) counts as focused: it was
# made for a job.
CATEGORIES = ("generalist", "focused", "personal")
DEFAULT_CATEGORY = "focused"

# What an agent on legal or health work says about itself, in chat and in its
# own words: the text lives in the registry's `disclaimers`, under these keys.
_DISCLAIMER_RULE = (
    'Disclaimer: start your first reply in a conversation, and any note or report for the owner, '
    'with this line, in their language: "{text}" Keep it out of documents meant for others '
    "(contracts, letters, filings, minutes): say it in your covering message. Once per conversation "
    "is enough, and it never replaces real help."
)

_disclaimers_seen: tuple[float, dict] = (0.0, {})

# How many specialists a planner's prompt may carry on top of the generalists.
_SHORTLIST_SPECIALISTS = 15
_WORD = re.compile(r"[^\W\d_]{4,}")
# Words in a role's name that say nothing about what it's for.
_ROLE_NOISE = {
    "analyst", "assistant", "coach", "helper", "manager", "officer", "partner", "planner",
    "processor", "reviewer", "senior", "specialist", "writer", "with",
}


def disclaimer_text(key: str) -> str:
    """The line a disclaimer key stands for; empty when there is none."""
    global _disclaimers_seen
    if not key:
        return ""
    # Every listing of a person's agents asks: read the catalog only when it changes.
    try:
        mtime = _REGISTRY_FILE.stat().st_mtime
        if mtime != _disclaimers_seen[0]:
            _disclaimers_seen = (mtime, load_base_registry().get("disclaimers") or {})
    except (OSError, json.JSONDecodeError):
        return ""
    return str(_disclaimers_seen[1].get(key) or "")


def disclaimer_rule(text: str) -> str:
    """The instruction that makes an agent say ``text`` itself."""
    return _DISCLAIMER_RULE.format(text=text)


def with_disclaimer_rule(instructions: str, key: str) -> str:
    """``instructions`` with the disclaimer's rule, which the built-in archetypes carry
    under Hard limits: an edited copy or a person's own archetype gets it on top."""
    text = disclaimer_text(key)
    if not text or text in instructions:
        return instructions
    return f"{disclaimer_rule(text)}\n\n{instructions}".strip()


def archetype_category(archetype: dict) -> str:
    """The shelf an archetype sits on."""
    category = str((archetype or {}).get("category") or "")
    return category if category in CATEGORIES else DEFAULT_CATEGORY


def _stems(text: str) -> set[str]:
    """A word's first five letters: crude, and enough to match reconcile with reconciliation."""
    return {w[:5] for w in _WORD.findall(text.lower())}


def _relevance(low_intent: str, intent_stems: set[str], archetype: dict) -> int:
    score = 0
    for keyword in archetype.get("keywords") or []:
        low = str(keyword).lower()
        if low in low_intent:
            score += 3
        else:
            words = _stems(low)
            if words and words <= intent_stems:
                score += 2
    role_words = {w for w in _WORD.findall(str(archetype.get("role") or "").lower()) if w not in _ROLE_NOISE}
    score += 2 * len({w[:5] for w in role_words} & intent_stems)
    score += len(_stems(str(archetype.get("family") or "")) & intent_stems)
    return score


def shortlist_archetypes(intent: str, archetypes: list[dict], keep: Iterable[str] = ()) -> list[dict]:
    """The archetypes worth showing a planner for this request.

    Planners paste the catalog into their prompt, so a hundred specialists would
    cost thousands of tokens on every plan and bury the few that matter. They get
    every generalist, everything the person made or was given, whatever ``keep``
    names, and only the specialists this request is about. A request with no
    match (another language, say) falls back to the generalists alone, and any
    id stays valid for a route: this narrows what's shown, not what's allowed.
    """
    keep_ids = set(keep or ())
    low = (intent or "").lower()
    stems = _stems(low)
    scored: dict[str, int] = {}
    for a in archetypes:
        if a.get("source") in (None, "base") and archetype_category(a) != "generalist" and a.get("id") not in keep_ids:
            score = _relevance(low, stems, a)
            if score >= 2:
                scored[a["id"]] = score
    chosen = set(sorted(scored, key=lambda aid: scored[aid], reverse=True)[:_SHORTLIST_SPECIALISTS])
    return [
        a for a in archetypes
        if a.get("id") in chosen or a.get("id") in keep_ids
        or a.get("source") not in (None, "base") or archetype_category(a) == "generalist"
    ]


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
