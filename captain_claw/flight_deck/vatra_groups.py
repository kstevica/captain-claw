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

import json
import re

# Default cap on approved clarification loop-backs per run (global ceiling).
CLARIFY_CAP = 2

# Cap on later-group owners that may be CALLED FORWARD per run (pull-forward).
PULL_CAP = 2


def pull_decision(*, already_running: bool, used: int, deps: list[str] | None,
                  have: set[str], cap: int = PULL_CAP,
                  max_parallel: int = 0, pulls_in_flight: int = 0) -> str:
    """Whether a later-group owner may be called forward to run NOW because a
    current-phase teammate needs its output:

    * ``"joined"``      — it was already called forward; the waiter just waits again.
    * ``"proceed"``     — start it: pulls remain and every input it depends on
      already exists (pulling an owner whose own inputs are missing would just
      move the hole one hop).
    * ``"no_capacity"`` — the run's concurrency cap can't fit the pull: the
      waiter holds a dispatch slot for as long as it waits, and each in-flight
      pull implies another slot-holding waiter, so with
      ``pulls_in_flight + 1 >= max_parallel`` the pulled owner could never
      acquire a slot — the waiter would only burn its wait budget. Refuse
      instantly instead. ``max_parallel=0`` = uncapped, never refuses on this.
    * ``"refuse"``      — cap spent or inputs missing; the waiter proceeds
      without it.
    """
    if already_running:
        return "joined"  # joining costs no slot, so capacity can't block it
    if max_parallel > 0 and pulls_in_flight + 1 >= max_parallel:
        return "no_capacity"
    if used >= cap:
        return "refuse"
    if any(d not in have for d in (deps or [])):
        return "refuse"
    return "proceed"

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
    """The group a subtask actually runs in.

    A ``group_lock`` (the user's explicit choice in the Group 0 coordination gate)
    is ABSOLUTE — it overrides the archetype floor AND dependency repair, so the
    agent runs exactly where the user put it (the live board/wait covers any
    ordering the user creates). Else a ``group_resolved`` pin (set once by
    :func:`resolve_groups`, after dependency repair) wins. Else the archetype
    FLOOR, raised (never lowered) by an optional Lead-assigned ``group``."""
    locked = _parse_group((subtask or {}).get("group_lock"))
    if locked is not None:
        return locked
    pinned = _parse_group((subtask or {}).get("group_resolved"))
    if pinned is not None:
        return pinned
    floor = archetype_group(arch)
    lead = _parse_group((subtask or {}).get("group"))
    return max(floor, lead) if lead is not None else floor


def resolve_groups(subtasks: list[dict], arch_by_id: dict) -> list[str]:
    """Pin every subtask's final execution group, repairing dependency inversions.

    The observed failure: the Lead pushes a piece to a late group (fact-checker
    → D) while an EARLIER piece records ``depends_on`` it — the dependent then
    waits at runtime for output that is scheduled to be produced after it, which
    can never arrive. Dependencies are the ground truth about data flow, so they
    out-rank archetype floors and Lead pushes: a violated dependency is pulled
    into its dependent's group (same wave — the live board/wait covers a
    same-group hand-off). Ordinals only ever move down, so this terminates even
    on a (nonsensical) dependency cycle; the pass cap is belt-and-braces.

    Mutates each subtask, setting ``group_resolved`` ('A'..'D').
    Returns human-readable notes describing any repairs, for the run log."""
    eff: dict[str, int] = {}
    by_id: dict[str, dict] = {}
    for s in subtasks or []:
        arch = (arch_by_id or {}).get(str(s.get("owner_archetype_id") or ""), {})
        s.pop("group_resolved", None)  # re-resolve from scratch (idempotent)
        eff[s["id"]] = effective_group(s, arch)
        by_id[s["id"]] = s
    notes: list[str] = []
    for _ in range(len(subtasks or []) + 1):
        moved = False
        for s in subtasks or []:
            for dep in s.get("depends_on") or []:
                if dep in eff and eff[dep] > eff[s["id"]]:
                    # A user-locked dependency is absolute — the user chose to run it
                    # later than a dependent; honor it (the board/wait bridges the gap)
                    # instead of pulling it back.
                    dep_s = by_id.get(dep)
                    if dep_s is not None and _parse_group(dep_s.get("group_lock")) is not None:
                        notes.append(
                            f"{dep} kept at {group_label(eff[dep])} (user-locked) though "
                            f"{s['id']} depends on it")
                        continue
                    notes.append(
                        f"{dep} pulled {group_label(eff[dep])}→{group_label(eff[s['id']])} "
                        f"— {s['id']} depends on its output")
                    eff[dep] = eff[s["id"]]
                    moved = True
        if not moved:
            break
    for s in subtasks or []:
        s["group_resolved"] = group_label(eff[s["id"]])
    return notes


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


# ── Schedule awareness (workers must know who has run and who hasn't) ──
# The observed failure: a group-B worker spends its wait budget on a group-D
# teammate's output — which cannot arrive, because D runs after B. Workers get
# the schedule in their brief, and the wait endpoint refuses instantly when the
# target provably hasn't started.

def schedule_block(subtask_id: str, schedule: dict) -> str:
    """The worker-facing schedule block for a grouped run: who already finished,
    who runs WITH you, who runs AFTER you (never wait on those).

    ``schedule`` = {"current": ordinal, "done": set[subtask_id],
    "owners": [{"subtask", "arch", "role", "title", "group"}]}."""
    owners = [o for o in (schedule or {}).get("owners") or []
              if o.get("subtask") != subtask_id]
    if not owners:
        return ""
    current = int((schedule or {}).get("current") or 0)
    done_ids = (schedule or {}).get("done") or set()

    def _name(o: dict) -> str:
        label = o.get("role") or o.get("arch") or "teammate"
        title = str(o.get("title") or "")[:60]
        return f"{label} ({title})" if title else str(label)

    finished = [o for o in owners if o["subtask"] in done_ids]
    with_you = [o for o in owners
                if o["subtask"] not in done_ids and int(o["group"]) <= current]
    later = [o for o in owners
             if o["subtask"] not in done_ids and int(o["group"]) > current]
    lines = ["\n\nTEAM SCHEDULE — the team runs in ordered phases:"]
    if finished:
        lines.append("- Already FINISHED (their work is on the board / in the shared "
                     "folder — search it): " + "; ".join(_name(o) for o in finished))
    if with_you:
        lines.append("- Running WITH you now (their in-flight artifacts arrive on the "
                     "board — `vatra wait` works for these): "
                     + "; ".join(_name(o) for o in with_you))
    if later:
        lines.append("- Runs AFTER you — group " +
                     ", ".join(f"{group_label(int(o['group']))}: {_name(o)}" for o in later)
                     + ". Their output does NOT exist yet and CANNOT arrive while you "
                       "run. NEVER wait for it — produce your part with what exists and "
                       "mark anything they would have provided or verified as "
                       "(unverified) so a later phase can settle it.")
    return "\n".join(lines)


def match_later_owner(query: str, schedule: dict) -> dict | None:
    """The not-yet-started later-phase owner a wait query is about, or None.

    Heuristic by design: workers name the role they're waiting on ("fact-checker
    verified company data …"), so we match each later owner's archetype id and
    role as a normalized phrase against the normalized query. Conservative — no
    match means the wait proceeds normally."""
    if not (query or "").strip() or not schedule:
        return None
    q = " " + re.sub(r"[^\w]+", " ", query.casefold()).strip() + " "
    current = int(schedule.get("current") or 0)
    done_ids = schedule.get("done") or set()
    for o in schedule.get("owners") or []:
        if int(o.get("group") or 0) <= current or o.get("subtask") in done_ids:
            continue
        phrases = {str(o.get("arch") or ""), str(o.get("role") or "")}
        for p in phrases:
            p = re.sub(r"[^\w]+", " ", p.casefold()).strip()
            if p and f" {p} " in q:
                return o
    return None


# ── Clarification loop (later-phase owner asks an earlier one for more) ──

# Injected into an owner that runs AFTER the first phase: it may ask the Lead to
# have an earlier-phase teammate provide missing data. Bounded by CLARIFY_CAP.
REQUEST_DIRECTIVE = (
    "\n\nBLOCKED ON A TEAMMATE? You run in a LATER phase — the earlier specialists' "
    "work is already on the shared board (search it first with the `vatra` tool). If "
    "you are genuinely blocked because a specific earlier teammate's output is missing "
    "or unclear, and you cannot do your part well without it, end your reply with ONE "
    "final line:\n"
    "REQUEST: <that teammate's role> — <exactly what you need>\n"
    "The Lead decides whether to have them provide it — this is limited, so use it only "
    "for real blockers, never nice-to-haves. Otherwise just do your part with the board."
)

_REQUEST_RE = re.compile(r"(?mi)^\s*REQUEST\s*[:\-]\s*(.+?)\s*$")


def parse_request(output: str | None) -> str | None:
    """The text after a `REQUEST:` line in an owner's output, or None."""
    m = _REQUEST_RE.search(output or "")
    if not m:
        return None
    text = m.group(1).strip()
    return text[:400] or None


def clarify_prompt(requester_role: str, request_text: str, roster: list[dict],
                   board_digest: str = "") -> str:
    """Ask the Lead what to do about a blocked owner's request: point it at an
    existing answer, route a finished teammate to provide it, or deny."""
    listed = "\n".join(
        f"- id={o.get('id', '')} · {o.get('role', '')} — {str(o.get('title', ''))[:80]}"
        for o in roster) or "(none)"
    board = (
        "\nRecent team board posts (the request may ALREADY be answered here):\n"
        f"{board_digest}\n" if (board_digest or "").strip() else ""
    )
    return (
        "You are the Lead of a team running in ordered phases. A specialist is "
        "blocked and wants a teammate to provide missing data or clarification. "
        "Decide what happens.\n\n"
        f"Requester: {requester_role}\n"
        f"Its request: {request_text}\n"
        f"{board}\n"
        "Teammates who have ALREADY FINISHED and could provide it:\n"
        f"{listed}\n\n"
        "Decision rules, in order:\n"
        "1. If the requested data ALREADY EXISTS in a board post above, reply "
        "already_available=true with a pointer to it and NO provider — "
        "re-producing existing data wastes a teammate's turn.\n"
        "2. Otherwise approve ONLY if this is a genuine blocker AND one listed "
        "teammate can provide it — prefer the teammate the requester NAMED when "
        "it is on the list.\n"
        "3. DENY vague asks, nice-to-haves, or anything the requester could get "
        "itself from the shared board.\n"
        "Reply with ONLY this JSON — no prose:\n"
        '{"approve": true|false, "already_available": true|false, '
        '"pointer": "<the board post / file that answers it, or empty>", '
        '"provider": "<the id of the teammate to ask, or empty>", '
        '"instruction": "<one concrete sentence telling that teammate exactly what to produce '
        'and post to the board>"}'
    )


def parse_clarify(output: str | None) -> dict:
    """Parse the Lead's decision. Defaults to DENY on any parse trouble."""
    deny = {"approve": False, "already_available": False, "pointer": "",
            "provider": "", "instruction": ""}
    if not output:
        return deny
    text = output.strip()
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return deny
    try:
        raw = json.loads(m.group(0))
    except (ValueError, TypeError):
        return deny
    if not isinstance(raw, dict):
        return deny
    return {
        "approve": bool(raw.get("approve")),
        "already_available": bool(raw.get("already_available")),
        "pointer": str(raw.get("pointer") or "").strip()[:300],
        "provider": str(raw.get("provider") or "").strip(),
        "instruction": str(raw.get("instruction") or "").strip()[:600],
    }
