"""Vatra execution groups — ordered phases A→B→C→D for the grouped run mode.

Each archetype has a preset group (its execution phase): research/architecture/
planning first (A), build/write/analyse in the middle (B), review/debug/assemble
last (C). Groups run in ascending order with a barrier between them, so a later
group already has everything earlier groups posted.

Assignment rule (locked): the archetype preset is the FLOOR; the Lead may push a
subtask to a LATER group but never earlier. An archetype may override its preset
with an explicit ``group`` field (``"A".."D"``) in archetypes.json.

Pure + data-only so it's fully unit-testable — no I/O, no model calls.
"""

from __future__ import annotations

# Ordinals: A=1 (earliest) … D=4 (latest). We run the DISTINCT groups a team
# actually uses, ascending — so {A, C} is two phases (A then C), not four.
_LETTERS = ("A", "B", "C", "D")
_ORD = {letter: i + 1 for i, letter in enumerate(_LETTERS)}
_MIN_ORD, _MAX_ORD = 1, len(_LETTERS)
_DEFAULT_ORD = 2  # untagged archetypes land in the middle (B)


def group_label(ord_: int) -> str:
    """'A'..'D' for an ordinal (clamped to range; non-numeric → middle)."""
    try:
        n = int(ord_)
    except (TypeError, ValueError):
        n = _DEFAULT_ORD
    return _LETTERS[max(_MIN_ORD, min(_MAX_ORD, n)) - 1]


def _parse_group(value) -> int | None:
    """A group value ('A'..'D' or 1..4) → ordinal, or None if unset/invalid."""
    if value is None:
        return None
    s = str(value).strip().upper()
    if s in _ORD:
        return _ORD[s]
    try:
        n = int(s)
    except (TypeError, ValueError):
        return None
    return n if _MIN_ORD <= n <= _MAX_ORD else None


# Substrings (matched against archetype id + role) that pin the earliest / latest
# phases. Everything else defaults to the middle. Kept as heuristics so new
# archetypes get a sensible phase without hand-tagging; override with `group`.
_FIRST_HINTS = (
    "research", "scanner", "fact-check", "architect", "planner", "cartograph",
    "extractor", "screener", "triage",
)
_LAST_HINTS = (
    "reviewer", "debugger", "qa-", "qa ", "security", "report-builder", "reporter",
    "git-operator", "simplifier", "watchdog", "monitor",
)


def archetype_group(arch: dict) -> int:
    """The preset execution group (ordinal) for an archetype.

    Explicit ``group`` field wins; else role/family heuristics (research/design →
    A, review/assemble → C); else the middle (B).
    """
    explicit = _parse_group((arch or {}).get("group"))
    if explicit is not None:
        return explicit
    hay = f"{(arch or {}).get('id', '')} {(arch or {}).get('role', '')}".lower()
    fam = str((arch or {}).get("family", "")).lower()
    if fam.startswith("research") or any(h in hay for h in _FIRST_HINTS):
        return _ORD["A"]
    if any(h in hay for h in _LAST_HINTS):
        return _ORD["C"]
    return _DEFAULT_ORD


def effective_group(subtask: dict, arch: dict) -> int:
    """The group a subtask actually runs in: the archetype FLOOR, raised (never
    lowered) by an optional Lead-assigned ``group`` on the subtask."""
    floor = archetype_group(arch)
    lead = _parse_group((subtask or {}).get("group"))
    return max(floor, lead) if lead is not None else floor


def clamp_lead_group(subtask_group, floor: int) -> int | None:
    """Normalise a Lead-provided subtask group against the archetype floor:
    returns the clamped ordinal (never below the floor), or None if unset."""
    lead = _parse_group(subtask_group)
    if lead is None:
        return None
    return max(int(floor), lead)


def order_groups(ordinals) -> list[int]:
    """The distinct groups a team uses, ascending — the phases to run in order."""
    return sorted({int(o) for o in ordinals})
