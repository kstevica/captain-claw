"""First Mate's proposals: changes it suggests, which the owner accepts from the Line.

See docs/first-mate-plan.md §5 and §11 (Phase 5, piece 4). A proposal is a row
in the deck's action ledger (``autonomy``) of kind ``factory``, so it waits on
the Line and the Taskboard with Approve and Reject like the Arbiter's own.
It never runs without a person's Approve (``fd_dispatch.should_auto_dispatch``
never fires it), and Approve does exactly one thing (``op``):

* ``pause``: pause a factory in trouble;
* ``share``: give a factory that used its whole share a bigger one (share is a
  weight of the weekly budget, at most :data:`SHARE_MAX`);
* ``draft_for_goal``: draft a factory for a Course goal nothing has moved.
  Approving it only drafts: the charter still waits for its own approval.
* ``improve``: start an improvement run of a factory whose satisfaction has
  been under its target for two weeks (Captain Swarm on its own cases,
  ``first_mate_improve``; admins only, like Captain Swarm). Its best comes
  back as a draft the owner approves.

Where they come from:

* **the weekly review**, with Monday's report (:func:`weekly`): rules over the
  week that ended, at most :data:`WEEKLY_MAX`, for an owner with at least one
  approved factory;
* **the Arbiter** (``draft_for_goal`` only), when one of the owner's stalled
  goals needs recurring work.

One waiting proposal per op and target: a second is never added beside it.
"""

from __future__ import annotations

import logging
from typing import Any

from captain_claw.flight_deck import first_mate_store as fms
from captain_claw.flight_deck.first_mate_store import FirstMateError

log = logging.getLogger(__name__)

KIND = "factory"
OPS = ("pause", "share", "draft_for_goal", "improve")
TWO_WEEKS_SIGNALS = 5  # at least this many signals over the two weeks, two or more in each
WEEKLY_MAX = 3
SHARE_MAX = 10
SOURCE = "first_mate"


async def available(owner_id: str) -> bool:
    """Whether First Mate is on this deck for this owner."""
    from captain_claw.flight_deck import crew_runner
    from captain_claw.flight_deck.first_mate_routes import access
    user = await crew_runner._user_of(fms.norm_user(owner_id))
    return bool(user) and access(user)[0]


def serves(store: fms.FirstMateStore, owner_id: str, goal_id: str) -> bool:
    """Whether one of the owner's factories already serves this goal."""
    return any(f["goal_id"] == goal_id for f in store.list_factories(owner_id))


def below_target(store: fms.FirstMateStore, factory: dict[str, Any], now: Any) -> tuple[float, float] | None:
    """``(last week's satisfaction, the target)`` when both of the last two
    weeks were under the factory's target, with enough signals to say so."""
    from datetime import datetime, timedelta
    live = store.live(factory["id"]) or {}
    target = float((live.get("checks") or {}).get("satisfaction_target") or fms.DEFAULT_CHECKS["satisfaction_target"])
    weeks: list[list[float]] = [[], []]
    for s in store.signals(factory["id"], limit=500):
        age = now - datetime.fromisoformat(s["at"])
        if timedelta(0) <= age < timedelta(days=14):
            weeks[0 if age >= timedelta(days=7) else 1].append(float(s["value"]))
    if min(len(w) for w in weeks) < 2 or sum(len(w) for w in weeks) < TWO_WEEKS_SIGNALS:
        return None
    means = [sum(w) / len(w) for w in weeks]
    return (round(means[1], 3), target) if all(m < target for m in means) else None


def _target(payload: dict[str, Any]) -> str:
    return str(payload.get("factory_id") or payload.get("goal_id") or "")


def waiting(owner_id: str) -> list[dict[str, Any]]:
    from captain_claw.flight_deck.autonomy import get_store
    return [a for a in get_store().waiting_actions(owner_id) if a.get("kind") == KIND]


async def propose(owner_id: str, *, op: str, title: str, rationale: str, payload: dict[str, Any],
                  goal_id: str = "", source: str = SOURCE, score: float = 0.7) -> dict[str, Any] | None:
    """A proposal on the Line, unless the same one already waits. None then."""
    if op not in OPS:
        raise FirstMateError(f"a First Mate proposal is one of: {', '.join(OPS)}")
    payload = {**payload, "op": op}
    for a in waiting(owner_id):
        p = a.get("payload") or {}
        if p.get("op") == op and _target(p) == _target(payload):
            return None
    from captain_claw.flight_deck.autonomy import get_store
    from captain_claw.flight_deck.owner_notify import notify_proposal
    row = get_store().add_action(fms.norm_user(owner_id), kind=KIND, title=title[:200], rationale=rationale[:500],
                                 source=source, risk="normal", domain="first_mate", score=score,
                                 status="awaiting_approval", payload=payload, goal_id=goal_id)
    await notify_proposal(fms.norm_user(owner_id), row)
    return row


async def weekly(store: fms.FirstMateStore, owner_id: str, report: dict[str, Any]) -> list[dict[str, Any]]:
    """The weekly review's proposals for the week the report covers."""
    if not report.get("factories") or not await available(owner_id):
        return []
    made: list[dict[str, Any]] = []

    async def add(**kw: Any) -> None:
        if len(made) < WEEKLY_MAX:
            row = await propose(owner_id, **kw)
            if row is not None:
                made.append(row)

    for x in report["factories"]:
        f = store.get(x["id"])
        if f is None:
            continue
        if x["health"] == "red" and f["state"] == "running":
            await add(op="pause", title=f"Pause {f['title']}",
                      rationale=f"It's in trouble: {x['health_why'].rstrip('.')}. Pausing stops it starting "
                                "on its own until you look at it.",
                      payload={"factory_id": f["id"]})
        used_all = x["share"] and x["tokens"] >= 0.95 * x["share"]
        if (f["state"] == "paused_budget" or used_all) and f["share"] < SHARE_MAX and x["health"] != "red":
            await add(op="share", title=f"Give {f['title']} a bigger share",
                      rationale=f"It used its whole share last week ({x['tokens']:,} tokens). A share of "
                                f"{f['share'] + 1} instead of {f['share']} gives it more of your weekly budget.",
                      payload={"factory_id": f["id"], "share": f["share"] + 1})
    from datetime import datetime

    from captain_claw.flight_deck import crew_runner, first_mate_improve
    now = datetime.fromisoformat(report["end"]) if report.get("end") else fms._utcnow()
    user = await crew_runner._user_of(fms.norm_user(owner_id))
    for x in report["factories"]:
        f = store.get(x["id"])
        low = below_target(store, f, now) if f and f["state"] == "running" else None
        if low and user and (await first_mate_improve.can_improve(user, f["id"]))[0]:
            await add(op="improve", title=f"Improve {f['title']}",
                      rationale=f"Satisfaction has been under its target for two weeks ({low[0]:.0%} last week, "
                                f"target {low[1]:.0%}). An improvement run tries better charters on its own cases; "
                                "the best comes back as a draft you approve.",
                      payload={"factory_id": f["id"]})
    try:
        from captain_claw.flight_deck.course_progress import stalled_goals
        stalled = stalled_goals(fms.norm_user(owner_id))
    except Exception:  # noqa: BLE001 — the review never fails on Course
        stalled = []
    for goal, days in stalled:
        if not serves(store, owner_id, goal["id"]):
            await add(op="draft_for_goal", title=f"Start a factory for: {goal['title']}"[:200],
                      rationale=f"Nothing has moved this goal in {days} days. A standing job could keep it "
                                "going. Approving drafts one; its charter still waits for your approval.",
                      payload={"goal_id": goal["id"]}, goal_id=goal["id"])
    return made


async def dispatch(user_id: str, action: dict[str, Any], *, approved_by_human: bool) -> dict[str, Any]:
    """What a person's Approve on a First Mate proposal does: its one op. A
    proposal no person approved never runs."""
    from captain_claw.flight_deck.autonomy import get_store as ledger
    if not approved_by_human:
        return {"ok": False, "target": "", "note": "a First Mate proposal waits for your Approve"}
    payload = action.get("payload") or {}
    op = str(payload.get("op") or "")
    owner = fms.norm_user(user_id)
    store = fms.get_store()
    target = _target(payload)
    try:
        if not await available(owner):
            raise FirstMateError("First Mate isn't on for you on this deck")
        if op == "improve":
            from fastapi import HTTPException

            from captain_claw.flight_deck import crew_runner, first_mate_improve
            from captain_claw.flight_deck.captain_swarm_objective import ObjectiveError
            user = await crew_runner._user_of(owner)
            f = store.get_owned(str(payload.get("factory_id") or ""), owner)
            if f is None or not user:
                raise FirstMateError("that factory is gone")
            try:
                run = await first_mate_improve.start(user, f["id"], started_by="the owner, approving First Mate")
            except ObjectiveError as exc:
                raise FirstMateError(str(exc)) from None
            except HTTPException as exc:
                raise FirstMateError(str(exc.detail)) from None
            note = (f"Started an improvement run of {f['title']} ({run['id']}). You'll hear when it ends; its best "
                    "comes back as a draft for you to approve.")
        elif op in ("pause", "share"):
            f = store.get_owned(str(payload.get("factory_id") or ""), owner)
            if f is None:
                raise FirstMateError("that factory is gone")
            if op == "pause":
                store.set_state(f["id"], "paused", "owner", "paused on your Approve of First Mate's proposal")
                note = f"Paused {f['title']}."
            else:
                share = min(SHARE_MAX, int(payload.get("share") or f["share"] + 1))
                store.update_factory(f["id"], "owner", share=share)
                note = f"{f['title']} now has a share of {share}."
        elif op == "draft_for_goal":
            from captain_claw.flight_deck import course, first_mate_drafting
            goal = course.get_store().get(str(payload.get("goal_id") or ""))
            if goal is None or not (goal["scope"] == "instance" or goal["owner_id"] == owner):
                raise FirstMateError("that goal is gone")
            if serves(store, owner, goal["id"]):
                raise FirstMateError("a factory already serves that goal")
            made = await first_mate_drafting.draft(store, owner, "", goal=goal)
            note = f"Drafted {made['title']}. Its charter waits for your approval on the Line."
        else:
            raise FirstMateError("this proposal doesn't say what to do")
    except FirstMateError as exc:
        ledger().update_status(action["id"], "failed", outcome="fail", outcome_note=str(exc)[:500])
        return {"ok": True, "target": target, "note": str(exc)}
    ledger().update_status(action["id"], "done", outcome="success", outcome_note=note)
    ledger().log(owner, "first mate", note)
    return {"ok": True, "target": target, "note": note}
