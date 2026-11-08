"""Course proposals (Phase 3) — goals the agents and the deck suggest.

Every suggestion is a goal row with ``status = proposed`` (see
:meth:`course.CourseStore.propose` for the rules: suggestions on, the weekly
limit, room among the waiting ones, no duplicates, the 90-day decline memory)
plus one Line item that waits for the scope's owner (``ref_type``
``course_goal``). Accepting or declining it — on the Course page or from the
Line item — answers that item.

One exception, the owner's own choice (``auto_accept_agent_short``): an
agent's short-term goal for itself goes straight onto its course, and the Line
just says so.
"""

from __future__ import annotations

from typing import Any

from captain_claw.flight_deck import course as c
from captain_claw.logging import get_logger

log = get_logger(__name__)

PROPOSAL_REF = "course_goal"
DECK_NAME = "Flight Deck"
_WHERE = {"user": "your Course", "agent": "an agent's course"}


def _flat(text: Any, cap: int = 400) -> str:
    return " ".join(str(text or "").split())[:cap]


def proposal_text(goal: dict[str, Any], proposer: str, old: dict[str, Any] | None = None) -> tuple[str, str]:
    """(title, body) of a proposal's Line item."""
    change = old is not None
    title = f"{'Change proposed' if change else 'Goal proposed'}: {_flat(goal['title'], 160)}"
    lines = [f"{proposer} suggests {'a change to a goal on' if change else 'a goal for'} "
             f"{_WHERE.get(goal['scope'], 'your Course')}."]
    lines.append(f"{c.HORIZON_LABELS.get(goal['horizon'], goal['horizon'])} goal: {_flat(goal['title'], 160)}")
    if change:
        what = [k.replace("_", " ") for k in c.CHANGEABLE if (goal.get(k) or "") != (old.get(k) or "")]
        if what:
            lines.append(f"It changes: {', '.join(what)} (was: {_flat(old['title'], 160)}).")
    if goal.get("success"):
        lines.append(f"Done when: {_flat(goal['success'])}")
    if goal.get("rationale"):
        lines.append(f"Why: {_flat(goal['rationale'])}")
    lines.append("Accept it, edit it or decline it on the Course page.")
    return title[:200], "\n".join(lines)


async def announce(goal: dict[str, Any], *, proposer: str, trust: str) -> str:
    """The proposal's Line item, for the scope's owner. Its id ("" when it
    couldn't be written)."""
    from captain_claw.flight_deck import line

    old = c.get_store().get(goal["supersedes_id"]) if goal.get("supersedes_id") else None
    title, body = proposal_text(goal, proposer, old)
    return await line.post(c.norm_user(goal["owner_id"]), "proposal", title, body,
                           bell_type="course_proposal", needs_you=True, source=proposer,
                           ref_type=PROPOSAL_REF, ref_id=goal["id"], trust=trust)


async def resolve(db, goal: dict[str, Any]) -> int:
    """A proposal was answered: its Line item no longer waits. Never raises."""
    try:
        return await db.resolve_line_items(c.norm_user(goal["owner_id"]), PROPOSAL_REF, [goal["id"]])
    except Exception as exc:  # noqa: BLE001
        log.debug("course: proposal %s not resolved on the Line: %s", goal.get("id"), exc)
        return 0


async def submit(*, owner_id: str, origin: str, proposed_by: str, proposer: str,
                 change_of: str = "", changes: dict[str, Any] | None = None,
                 **fields: Any) -> dict[str, Any]:
    """Make the suggestion (a new goal, or with ``change_of`` a change to one),
    then either put it on the course at once (the owner's auto-accept for an
    agent's own short-term goal) or ask the owner on the Line.
    ``{"goal", "auto_accepted"}``; :class:`course.CourseError` when the rules
    refuse it."""
    store = c.get_store()
    trust = "agent" if origin == "agent" else "deck"
    if change_of:
        goal = store.propose_change(change_of, changes or {}, origin=origin,
                                    proposed_by=proposed_by, rationale=fields.get("rationale", ""))
    else:
        goal = store.propose(owner_id=owner_id, origin=origin, proposed_by=proposed_by, **fields)
    auto = (origin == "agent" and goal["scope"] == "agent" and goal["horizon"] == "short"
            and not goal.get("supersedes_id") and goal["agent_ref"] == proposed_by
            and store.prefs(goal["owner_id"])["auto_accept_agent_short"])
    if auto:
        try:
            goal = store.accept(goal["id"], actor=f"auto:{proposed_by}")
        except c.CourseError:
            auto = False   # no room for it, say: it waits like any other
    if auto:
        from captain_claw.flight_deck import course_carrier, line
        from captain_claw.flight_deck.auth import get_db

        await line.post(c.norm_user(goal["owner_id"]), "agent",
                        f"Added to {proposer}'s course: {_flat(goal['title'], 160)}"[:200],
                        f"{proposer} added a short-term goal for itself (you let agents do that).\n"
                        f"Done when: {_flat(goal['success'])}\n\nChange or drop it on the Course page.",
                        bell_type="course_auto_added", source=proposer, trust=trust)
        try:
            await course_carrier.refresh_for_goal(get_db(), goal)
        except Exception:  # noqa: BLE001 — the files catch up on the next change
            pass
        from captain_claw.flight_deck import course_conflicts

        course_conflicts.check_soon(get_db(), goal, thinker=c.norm_user(goal["owner_id"]))
    else:
        await announce(goal, proposer=proposer, trust=trust)
    return {"goal": goal, "auto_accepted": auto}
