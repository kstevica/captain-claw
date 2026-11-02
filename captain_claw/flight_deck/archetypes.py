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
# An admin's own shelves, published for the whole deck (team_archetypes.py), carry
# this prefix, so a mistyped category can never pass for one.
TEAM_GROUP_PREFIX = "group-"
TEAM_SETTING = "fd:team-archetypes"

CRISIS_RULE = (
    "If someone says they may hurt themselves or someone else, or that they are not safe: stop the current "
    "task, respond with care, urge them to contact local emergency services now (112 in the EU, 999 or 112 in "
    "the UK, 911 in the US) or a crisis line (988 in the US, 116 123 Samaritans in the UK and Ireland, "
    "findahelpline.com elsewhere), and to reach a person they trust. Do not try to handle it yourself."
)

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
_SHORTLISTED_SOURCES = (None, "base", "team")
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


def place_safety_rules(instructions: str, disclaimer: str) -> str:
    """Generated instructions with the rules a model doesn't get to word: first under
    Hard limits, as in the built-in catalog, the crisis rule before the disclaimer."""
    rules = []
    if disclaimer == "health" and CRISIS_RULE not in instructions:
        rules.append(CRISIS_RULE)
    text = disclaimer_text(disclaimer)
    if text and text not in instructions:
        rules.append(disclaimer_rule(text))
    if not rules:
        return instructions
    bullets = "".join(f"- {rule}\n" for rule in rules)
    head = "## Hard limits\n"
    at = instructions.find(head)
    if at < 0:
        return f"{bullets}\n{instructions}".strip()
    at += len(head)
    return instructions[:at] + bullets + instructions[at:]


def archetype_category(archetype: dict) -> str:
    """The shelf an archetype sits on."""
    category = str((archetype or {}).get("category") or "")
    return category if category in CATEGORIES or category.startswith(TEAM_GROUP_PREFIX) else DEFAULT_CATEGORY


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
    names, and only the specialists this request is about, which includes what an
    admin published for the deck. A request with no
    match (another language, say) falls back to the generalists alone, and any
    id stays valid for a route: this narrows what's shown, not what's allowed.
    """
    keep_ids = set(keep or ())
    low = (intent or "").lower()
    stems = _stems(low)
    scored: dict[str, int] = {}
    for a in archetypes:
        if a.get("source") in _SHORTLISTED_SOURCES and archetype_category(a) != "generalist" and a.get("id") not in keep_ids:
            score = _relevance(low, stems, a)
            if score >= 2:
                scored[a["id"]] = score
    chosen = set(sorted(scored, key=lambda aid: scored[aid], reverse=True)[:_SHORTLIST_SPECIALISTS])
    return [
        a for a in archetypes
        if a.get("id") in chosen or a.get("id") in keep_ids
        or a.get("source") not in _SHORTLISTED_SOURCES or archetype_category(a) == "generalist"
    ]


def load_base_registry() -> dict:
    """Load and parse the base archetype registry from disk.

    Raises ``FileNotFoundError`` / ``json.JSONDecodeError`` on a missing or
    invalid file — callers that must not fail (Forge catalog injection) catch
    these; the route handler surfaces them as HTTP 500.
    """
    return json.loads(_REGISTRY_FILE.read_text())


async def team_catalog(db) -> dict:
    """What an admin published for the whole deck: its groups, families and archetypes."""
    empty: dict = {"groups": [], "families": [], "archetypes": []}
    reader = getattr(db, "get_system_setting", None)
    if reader is None:
        return empty
    try:
        stored = json.loads(await reader(TEAM_SETTING) or "{}")
    except (json.JSONDecodeError, TypeError):
        log.warning("Team archetypes are unreadable; carrying on without them")
        return empty
    return {**empty, **stored} if isinstance(stored, dict) else empty


def merge_archetypes(
    base: list[dict], user_rows: list[dict],
    shared_rows: list[dict] | None = None,
    team: list[dict] | None = None,
) -> list[dict]:
    """Overlay what the deck published, a user's archetypes and ones shared to them on the
    base list.

    Each base entry is tagged ``source="base"``, and each deck-wide one ``source="team"``,
    which a person's own archetype of the same id replaces for them. Each owned user row is tagged
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

    for a in team or []:
        entry = {**a, "source": "team"}
        if entry.get("id", "") in index:
            merged[index[entry["id"]]] = entry
        else:
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
    team = (await team_catalog(db))["archetypes"]
    if not user_id:
        return merge_archetypes(base, [], None, team)
    rows, shared = await _owned_and_shared(db, user_id)
    return merge_archetypes(base, rows, shared, team)


async def merged_registry(db, user_id: str | None) -> dict:
    """Return the full registry dict with ``archetypes`` merged for ``user_id``.

    ``tiers`` and ``base_tools`` come from the base registry unchanged.
    """
    registry = load_base_registry()
    base = registry.get("archetypes", [])
    team = await team_catalog(db)
    registry["groups"] = team["groups"]
    registry["family_looks"] = team["families"]
    if not user_id:
        registry["archetypes"] = merge_archetypes(base, [], None, team["archetypes"])
        return registry
    rows, shared = await _owned_and_shared(db, user_id)
    registry["archetypes"] = merge_archetypes(base, rows, shared, team["archetypes"])
    return registry
