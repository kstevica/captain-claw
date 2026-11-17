"""First Mate's supervisor: the loop that runs factories on their own.

See docs/first-mate-plan.md §5, §9, §10 and §13 (Phase 2). Every
:data:`TICK_SECONDS`, one scan at a time. A scan landing on a running scan
returns at once: overlapping scans once gave a Bat run three drivers.

1. Every going run is brought up to date (:func:`first_mate_runs.refresh`).
2. With First Mate off for the deck, nothing else happens.
3. For each owner with a factory:
   * **failure streak**: a running factory whose last :data:`FAILURE_STREAK`
     finished runs since it last started running all failed is paused, with why.
     Stopped runs don't count either way;
   * **budget**: the owner's weekly tokens are split between their approved
     factories by share. A running factory that spent its share goes to
     ``paused_budget``. One with tokens left again (a new week, a bigger
     budget) runs again. A run's cap is never above what is left;
   * **the owner's switch** (``prefs.on``): off, no run starts on its own;
   * **schedules**: each running factory's schedule triggers that are due
     start a run, in order of priority, then of how long they've waited, while
     the runner has slots.

Schedules. A trigger first seen waits for its first due time. It never fires
the moment it's approved. When due:

* a factory with a run still going skips this one (no pile-up);
* no free slot: it stays due and tries again next scan;
* once fired or skipped, its next time is worked out from now.

One due more than :data:`LATE_GRACE` ago (the deck was down, the factory or
First Mate was off) is skipped and noted rather than run late. Each trigger's
next time lives in the store (``fm_trigger_state``), so a restart neither
repeats nor loses one.

Week: Monday 00:00 UTC.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
from datetime import UTC, datetime, timedelta
from typing import Any

from captain_claw.flight_deck import first_mate_events as fme
from captain_claw.flight_deck import first_mate_line as fml
from captain_claw.flight_deck import first_mate_runs as runs
from captain_claw.flight_deck import first_mate_store as fms
from captain_claw.flight_deck.first_mate_store import FirstMateError

log = logging.getLogger(__name__)


def _env_seconds(name: str, default: float) -> float:
    try:
        return max(15.0, float(os.environ.get(name, "") or default))
    except ValueError:
        return default


TICK_SECONDS = _env_seconds("FD_FIRST_MATE_TICK_SECONDS", 60.0)
FAILURE_STREAK = 3
LATE_GRACE = timedelta(hours=6)

SWITCH_REASON = "First Mate is off"
BUDGET_REASON = "used its share of this week's tokens"
STREAK_REASON = f"{FAILURE_STREAK} failed runs in a row"
ACTOR = "first_mate"


# ── pure helpers ───────────────────────────────────────────────────────

def week_start(now: datetime) -> datetime:
    """Monday 00:00 UTC of ``now``'s week."""
    d = now.astimezone(UTC)
    return (d - timedelta(days=d.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)


def trigger_key(trigger: dict[str, Any]) -> str:
    return hashlib.sha1(json.dumps(trigger, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def shares(prefs: dict[str, Any], factories: list[dict[str, Any]]) -> dict[str, int]:
    """Each approved factory's tokens for the week: the owner's weekly budget
    split by share; UNLIMITED for each when the week has no limit."""
    eligible = [f for f in factories if f["version"] and f["state"] != "retired"]
    total = sum(int(f["share"]) for f in eligible)
    if not total:
        return {}
    if fms.unlimited(prefs["weekly_tokens"]):
        return {f["id"]: fms.UNLIMITED for f in eligible}
    return {f["id"]: int(prefs["weekly_tokens"] * int(f["share"]) / total) for f in eligible}


def _lower_to(left: int) -> int | None:
    """What a run's cap is lowered to: what the factory has left this week, or
    nothing when the week has no limit."""
    return None if not left or fms.unlimited(left) else int(left)


def next_fire(schedule: str, tz: str, now: datetime) -> float | None:
    """The next time ``schedule`` is due after ``now`` (epoch), or None when it
    has none (a one-shot that's done, or a schedule that no longer parses)."""
    from captain_claw.flight_deck.fd_scheduler import ScheduleError, compute_next_run
    # Without a zone the scheduler counts in the deck's local time, from a
    # naive base; with one, it converts an aware base itself.
    base = now if tz else datetime.fromtimestamp(now.timestamp())
    try:
        return compute_next_run(schedule, base=base, tz=tz)
    except ScheduleError:
        return None


# ── the supervisor ─────────────────────────────────────────────────────

class Supervisor:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()

    async def tick(self, now: datetime | None = None) -> dict[str, int]:
        """One scan. Returns what it did, for logs and tests."""
        if self._lock.locked():
            return {"busy": 1}
        async with self._lock:
            return await self._scan(now or fms._utcnow())

    async def _scan(self, now: datetime) -> dict[str, int]:
        out = {"refreshed": 0, "started": 0, "paused": 0, "resumed": 0, "skipped": 0, "waiting": 0, "trusted": 0}
        if fms._STORE is None and not fms._db_path().exists():
            return out  # First Mate never used on this deck: nothing to do, and no file made
        store = fms.get_store()
        for run in store.active_runs():
            try:
                await runs.refresh(run["id"], now=now)
                out["refreshed"] += 1
            except Exception:  # noqa: BLE001 — one bad run never stops the scan
                log.warning("first mate: refreshing a run failed", extra={"run": run["id"]}, exc_info=True)
        if not store.settings()["enabled"]:
            return out
        week = week_start(now)
        try:
            from captain_claw.flight_deck import first_mate_drafts
            await first_mate_drafts.check_due(store, now)  # what became of the drafts factories left
        except Exception:  # noqa: BLE001
            log.warning("first mate: checking drafts failed", exc_info=True)
        from captain_claw.flight_deck import first_mate_cases
        await first_mate_cases.drive(store, now)  # each running check one step on
        for owner in store.owners():
            try:
                await self._owner(store, owner, now, week, out)
            except Exception:  # noqa: BLE001 — one owner's trouble never stops the others
                log.warning("first mate: scanning an owner failed", extra={"owner": owner}, exc_info=True)
        return out

    async def _owner(self, store: fms.FirstMateStore, owner: str, now: datetime, week: datetime,
                     out: dict[str, int]) -> None:
        prefs = store.prefs(owner)
        factories = store.list_factories(owner)
        share = shares(prefs, factories)
        try:
            from captain_claw.flight_deck import first_mate_trust
            out["trusted"] += await first_mate_trust.consider_owner(store, owner, now)  # trust raised by the record
        except Exception:  # noqa: BLE001
            log.warning("first mate: weighing trust failed", extra={"owner": owner}, exc_info=True)
        try:
            from captain_claw.flight_deck import first_mate_report
            await first_mate_report.maybe_send(store, owner, now)  # Monday: the week that ended
        except Exception:  # noqa: BLE001
            log.warning("first mate: the weekly report failed", extra={"owner": owner}, exc_info=True)
        if prefs["on"]:
            from captain_claw.flight_deck import first_mate_cases
            for f in factories:
                if first_mate_cases.weekly_due(store, f, now):
                    try:
                        first_mate_cases.start_check(store, f["id"], version=f["version"], purpose="weekly")
                    except FirstMateError:
                        pass
        due: list[dict[str, Any]] = []
        lefts: dict[str, int] = {}
        for f in factories:
            await self._streak(store, f, out)
            left = lefts[f["id"]] = await self._budget(store, store.get(f["id"]), share.get(f["id"], 0), week, out)
            f = store.get(f["id"])
            if not prefs["on"] or f["state"] != "running":
                continue
            live = store.live(f["id"])
            if live is not None:
                due += self._due(store, f, live, now, out, left)
        # Events: claim the new ones, give back what no factory listens for any
        # more, then each factory's batch once its window has passed.
        fme.release_orphans(store, owner)
        if prefs["on"]:
            fme.route(owner)
            for b in fme.batches(store, owner, now):
                f = store.get(b["factory_id"])
                if f is None or f["state"] != "running":
                    continue  # they wait for it to run again
                due.append({"kind": "events", "factory": f, "live": store.live(f["id"]), "events": b["events"],
                            "next_at": b["oldest"].timestamp(), "left": lefts.get(f["id"], 0)})
        due.sort(key=lambda d: (d["factory"]["priority"], d["next_at"]))
        started: set[str] = set()
        for d in due:
            await self._fire(store, owner, d, now, out, started)

    # ── a failure streak ───────────────────────────────────────────────

    async def _streak(self, store: fms.FirstMateStore, f: dict[str, Any], out: dict[str, int]) -> None:
        if f["state"] != "running":
            return
        finished = [r for r in store.runs_since(f["id"], store.resumed_at(f["id"]),
                                                kinds=("run", "rehearsal", "carry_out"))
                    if r["status"] in ("done", "failed")]
        last = finished[:FAILURE_STREAK]
        if len(last) == FAILURE_STREAK and all(r["status"] == "failed" for r in last):
            store.set_state(f["id"], "paused", ACTOR, STREAK_REASON)
            out["paused"] += 1
            await fml.factory_paused(f, STREAK_REASON, needs_you=True)

    # ── the budget ─────────────────────────────────────────────────────

    async def _budget(self, store: fms.FirstMateStore, f: dict[str, Any], share: int, week: datetime,
                      out: dict[str, int]) -> int:
        """Tokens the factory has left this week; parks or frees it. UNLIMITED
        when the week has no limit."""
        if not f["version"] or f["state"] == "retired":
            return 0
        if fms.unlimited(share):
            if f["state"] == "paused_budget":
                store.set_state(f["id"], "running", ACTOR, "the week has no token limit now")
                out["resumed"] += 1
                await fml.factory_resumed(f["owner_id"], f["id"])
            return fms.UNLIMITED
        left = max(0, share - store.tokens_since(f["id"], week))
        if f["state"] == "running" and left <= 0:
            store.set_state(f["id"], "paused_budget", ACTOR, BUDGET_REASON)
            out["paused"] += 1
            await fml.factory_paused(f, BUDGET_REASON, needs_you=False)
        elif f["state"] == "paused_budget" and left > 0:
            store.set_state(f["id"], "running", ACTOR, "it has tokens left this week")
            out["resumed"] += 1
            await fml.factory_resumed(f["owner_id"], f["id"])
        return left

    # ── schedules ──────────────────────────────────────────────────────

    def _due(self, store: fms.FirstMateStore, f: dict[str, Any], live: dict[str, Any], now: datetime,
             out: dict[str, int], left: int) -> list[dict[str, Any]]:
        state = store.trigger_state(f["id"])
        keys: list[str] = []
        due: list[dict[str, Any]] = []
        ts = now.timestamp()
        for t in live["triggers"]:
            if t.get("kind") != "schedule":
                continue
            key = trigger_key(t)
            keys.append(key)
            st = state.get(key)
            if st is None:
                store.set_trigger(f["id"], key, next_at=next_fire(t["schedule"], t.get("tz", ""), now))
                continue
            nxt = st["next_at"]
            if nxt is None or ts < nxt:
                continue
            if ts - nxt > LATE_GRACE.total_seconds():
                store.note(f["id"], ACTOR, "schedule_skipped",
                           {"schedule": t["schedule"], "reason": "it was due more than 6 hours ago"})
                store.set_trigger(f["id"], key, next_at=next_fire(t["schedule"], t.get("tz", ""), now))
                out["skipped"] += 1
                continue
            due.append({"kind": "schedule", "factory": f, "live": live, "key": key, "trigger": t,
                        "next_at": nxt, "left": left})
        store.keep_triggers(f["id"], keys)
        return due

    async def _fire(self, store: fms.FirstMateStore, owner: str, d: dict[str, Any], now: datetime,
                    out: dict[str, int], started: set[str]) -> None:
        if d["kind"] == "events":
            await self._fire_batch(store, owner, d, out, started)
            return
        f, t = d["factory"], d["trigger"]

        def advance(fired: bool, why: str = "") -> None:
            store.set_trigger(f["id"], d["key"], next_at=next_fire(t["schedule"], t.get("tz", ""), now),
                              last_at=now.timestamp() if fired else None)
            if why:
                store.note(f["id"], ACTOR, "schedule_skipped", {"schedule": t["schedule"], "reason": why})
                out["skipped"] += 1

        if f["id"] in started:
            advance(False)  # two of its schedules due at once: one run covers both
            return
        if any(r["factory_id"] == f["id"] for r in store.active_runs(owner)):
            advance(False, "a run was still going")
            return
        try:
            runner = runs.runner_for(d["live"]["runner"])
        except FirstMateError as exc:  # a runner that isn't wired yet
            advance(False, str(exc))
            return
        try:
            free = await runner.capacity(f["owner_id"])
        except FirstMateError:
            out["waiting"] += 1  # Bat not running right now: try again next scan
            return
        if free <= 0:
            out["waiting"] += 1  # no free slot: it stays due
            return
        try:
            await runs.start(f["id"], owner, trigger="schedule", watch=False, max_tokens=_lower_to(d["left"]))
        except FirstMateError as exc:
            advance(False, str(exc))
            return
        advance(True)
        started.add(f["id"])
        out["started"] += 1

    async def _fire_batch(self, store: fms.FirstMateStore, owner: str, d: dict[str, Any],
                          out: dict[str, int], started: set[str]) -> None:
        """One run for a factory's batch of events. A batch is never skipped:
        when it can't start now (a run going, no slot, Bat down, the day's cap)
        its events stay claimed and wait. A runner that isn't wired gives them
        back to the Arbiter."""
        f = d["factory"]
        ids = [e["id"] for e in d["events"]]
        if f["id"] in started or any(r["factory_id"] == f["id"] for r in store.active_runs(owner)):
            out["waiting"] += 1
            return
        try:
            runner = runs.runner_for(d["live"]["runner"])
        except FirstMateError:
            from captain_claw.flight_deck import events
            events.get_store().unclaim(ids)
            return
        try:
            free = await runner.capacity(f["owner_id"])
        except FirstMateError:
            free = 0
        if free <= 0:
            out["waiting"] += 1
            return
        try:
            run = await runs.start(f["id"], owner, trigger="events", watch=False,
                                   context=[fme.context_of(e) for e in d["events"]],
                                   max_tokens=_lower_to(d["left"]))
        except FirstMateError:
            out["waiting"] += 1
            return
        fme.settle(ids, run["id"])
        started.add(f["id"])
        out["started"] += 1


_SUPERVISOR = Supervisor()


def supervisor() -> Supervisor:
    return _SUPERVISOR


async def kick() -> dict[str, int]:
    """A scan now (a page, a test), sharing the one lock."""
    return await _SUPERVISOR.tick()


async def supervisor_loop(stop: asyncio.Event) -> None:
    """The deck's background task: a scan every TICK_SECONDS until ``stop``."""
    while not stop.is_set():
        try:
            await _SUPERVISOR.tick()
        except Exception:  # noqa: BLE001 — the loop never dies of one scan
            log.warning("first mate: a scan failed", exc_info=True)
        try:
            await asyncio.wait_for(stop.wait(), timeout=TICK_SECONDS)
        except TimeoutError:
            pass


# ── the owner's switch ─────────────────────────────────────────────────

async def switch_off(owner_id: str) -> int:
    """First Mate off for one owner: every running factory pauses (with
    :data:`SWITCH_REASON`) and every going run stops. Returns how many paused."""
    store = fms.get_store()
    paused = 0
    for f in store.list_factories(owner_id):
        if f["state"] == "running":
            store.set_state(f["id"], "paused", ACTOR, SWITCH_REASON)
            paused += 1
    for run in store.active_runs(owner_id):
        try:
            await runs.stop(run["id"], owner_id)
        except FirstMateError:
            pass
    return paused


async def switch_on(owner_id: str) -> int:
    """First Mate on again: the factories the switch paused run again. Ones the
    owner or a brake paused stay paused."""
    store = fms.get_store()
    resumed = 0
    for f in store.list_factories(owner_id):
        if f["state"] == "paused" and f["state_reason"] == SWITCH_REASON:
            store.set_state(f["id"], "running", ACTOR, "First Mate is on")
            resumed += 1
    return resumed
