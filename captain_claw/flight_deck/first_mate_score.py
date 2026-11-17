"""First Mate's scorecard: how one factory is doing, and its health light.

See docs/first-mate-plan.md §8 (Phase 4). Everything here is read from the
store, with no model:

* this week: runs, how many finished done, how many the reviewers passed,
  tokens per run, the median run time;
* the last two weeks: carry-outs, and how many the proof doubted;
* satisfaction: the average worth of the owner's last 20 signals (👍/👎,
  approvals, what became of the drafts), with how many there are;
* drafts: waiting, sent as written, sent edited, deleted.

Health:

* ``red``: paused by a brake (a failure streak, a rehearsal that reached
  outside, two doubtful carry-outs in two weeks), or two doubts in two weeks;
* ``new``: fewer than three finished runs, too soon to say;
* ``amber``: its latest check of the live version scored 10 points or more
  under the one before (counted before ``new``: a check is evidence on its
  own); over budget, one doubtful carry-out in two weeks, or satisfaction
  under its target with at least five signals;
* ``green``: otherwise.
"""

from __future__ import annotations

import statistics
from datetime import datetime, timedelta
from typing import Any

from captain_claw.flight_deck import first_mate_store as fms

FORTNIGHT = timedelta(days=14)
MIN_SIGNALS = 5
MIN_RUNS = 3


def _at(iso: str | None) -> datetime | None:
    return datetime.fromisoformat(iso) if iso else None


def health_of(factory: dict[str, Any], *, satisfaction: float | None, signals: int, target: float,
              doubts: int, finished: int, drop: tuple[float, float] | None = None) -> tuple[str, str]:
    from captain_claw.flight_deck.first_mate_loop import STREAK_REASON
    from captain_claw.flight_deck.first_mate_proof import TWO_DOUBTS_REASON
    from captain_claw.flight_deck.first_mate_runs import LEAK_REASON
    if factory["state"] == "paused" and factory["state_reason"] in (STREAK_REASON, LEAK_REASON, TWO_DOUBTS_REASON):
        return "red", f"Paused: {factory['state_reason']}."
    if doubts >= 2:
        return "red", f"{doubts} doubtful carry-outs in the last two weeks."
    if drop is not None:  # a check is evidence of its own, however few runs there were
        return "amber", f"It scored {drop[0]:.0%} on its cases, down from {drop[1]:.0%}."
    if finished < MIN_RUNS:
        return "new", "Not enough runs yet to say."
    if factory["state"] == "paused_budget":
        return "amber", "It used its share of this week's tokens."
    if doubts:
        return "amber", "A doubtful carry-out in the last two weeks."
    if satisfaction is not None and signals >= MIN_SIGNALS and satisfaction < target:
        return "amber", f"Satisfaction {satisfaction:.0%}, under its target of {target:.0%}."
    return "green", ""


def scorecard(store: fms.FirstMateStore, factory: dict[str, Any], now: datetime) -> dict[str, Any]:
    from captain_claw.flight_deck.first_mate_loop import week_start
    week = week_start(now)
    runs = store.list_runs(factory["id"], limit=500, kinds=fms.WORK_KINDS)
    finished = [r for r in runs if r["status"] in fms.TERMINAL_RUN and r["status"] != "cancelled"]
    this_week = [r for r in finished if _at(r["created_at"]) >= week]
    done = [r for r in this_week if r["status"] == "done"]
    judged = [r for r in this_week if r["verdict"] in ("pass", "fail", "leak")]
    fortnight = now - FORTNIGHT
    carries = [r for r in finished if r["kind"] == "carry_out" and _at(r["created_at"]) >= fortnight]
    doubts = [r for r in carries if (r.get("proof") or {}).get("state") == "doubt"]
    durations = [(_at(r["ended_at"]) - _at(r["started_at"])).total_seconds() / 60
                 for r in done if r.get("started_at") and r.get("ended_at")]
    recent = store.signals(factory["id"], limit=fms.SATISFACTION_WINDOW)
    satisfaction = store.satisfaction(factory["id"])
    live = store.live(factory["id"]) or {}
    target = float((live.get("checks") or {}).get("satisfaction_target") or fms.DEFAULT_CHECKS["satisfaction_target"])
    drafts = store.drafts(factory["id"], limit=500)
    from captain_claw.flight_deck import first_mate_cases
    checks = store.checks(factory["id"], limit=20)
    scored = [c for c in checks if c["status"] == "done" and c["score"] is not None
              and c["version"] == factory["version"]]
    drop = None
    if len(scored) >= 2 and scored[0]["score"] <= scored[1]["score"] - first_mate_cases.DRIFT:
        drop = (scored[0]["score"], scored[1]["score"])
    health, why = health_of(factory, satisfaction=satisfaction, signals=len(recent), target=target,
                            doubts=len(doubts), finished=len(finished), drop=drop)
    latest = checks[0] if checks else None
    return {
        "health": health, "health_why": why,
        "week": {"runs": len(this_week), "done": len(done), "judged": len(judged),
                 "passed": sum(1 for r in judged if r["verdict"] == "pass"),
                 "tokens_per_run": int(sum(r["tokens"] for r in this_week) / len(this_week)) if this_week else 0,
                 "median_minutes": round(statistics.median(durations), 1) if durations else None},
        "fortnight": {"carry_outs": len(carries), "doubts": len(doubts)},
        "satisfaction": satisfaction, "signals": len(recent), "target": target,
        "drafts": {s: sum(1 for d in drafts if d["status"] == s) for s in fms.FirstMateStore.DRAFT_STATUSES},
        "cases": len(first_mate_cases.cases(store, factory["id"])),
        "check": {k: latest[k] for k in ("id", "purpose", "version", "version_sha", "status", "score", "note",
                                         "created_at", "ended_at")} if latest else None,
    }
