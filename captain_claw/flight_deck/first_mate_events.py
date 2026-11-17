"""First Mate's event router: new mail and calendar changes start factory runs.

See docs/first-mate-plan.md §9 (Phase 2). Events come from the deck's event
store (:mod:`events`: the Gmail and Calendar pollers, webhooks).

* :func:`match`: does an ``events`` trigger match an event? These are rules
  only: the source, the event type, who it's from, a label, words.
* :func:`route`: each ``new`` event of an owner that matches an events trigger
  of one of their running factories (with their First Mate switch on) is
  claimed for ONE factory. It gets status ``claimed`` and ``fm_factory`` /
  ``fm_trigger`` in its metadata. The Arbiter reads only ``new`` events, so it
  never sees a claimed one. :mod:`arbiter` calls this right before it reads
  events, and the supervisor at every scan.
* When several factories match, the one with the highest priority (lowest
  number) takes it, then the oldest. The plan's model pass for this waits for
  First Mate's voice agent (Phase 5); until then it's this rule.
* :func:`batches`: per factory, its claimed events once the oldest claim has
  waited the trigger's batch window (``batch_minutes``, else the owner's
  default). The supervisor starts one run with them: 20 emails make one run,
  not 20. Their text reaches the run fenced as information
  (:func:`first_mate_runs.compose_brief`).
* After the run starts, :func:`settle` moves them to ``acted`` with the run in
  their metadata. If a run can't start, they stay claimed and wait.
* Events claimed for a factory that is gone, retired, or whose live version no
  longer has that trigger go back to ``new`` (:func:`release_orphans`).
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any

from captain_claw.flight_deck import first_mate_store as fms

log = logging.getLogger(__name__)

BATCH_EVENTS_MAX = 50   # what one run takes; the rest wait for the next batch
BODY_MAX = 2_000


def _events_store():
    """The deck's event store, or None where there is none yet (never made one)."""
    from captain_claw.flight_deck import events
    if events._STORE is None and not events._db_path().exists():
        return None
    return events.get_store()


def _key(trigger: dict[str, Any]) -> str:
    from captain_claw.flight_deck.first_mate_loop import trigger_key
    return trigger_key(trigger)


# ── matching (pure) ────────────────────────────────────────────────────

def match(trigger: dict[str, Any], event: dict[str, Any]) -> bool:
    """Rules only. Every filter given must hold; within one, any value will do.
    ``from`` looks at the sender, reply-to and organizer; ``label`` at the
    event's labels (a source that gives none never matches a label filter);
    ``words`` at the summary, body, subject and snippet. Case doesn't matter."""
    if trigger.get("kind") != "events" or event.get("source") != trigger.get("source"):
        return False
    if trigger.get("event_type") and event.get("event_type") != trigger["event_type"]:
        return False
    m = trigger.get("match") or {}
    meta = event.get("metadata") or {}
    if m.get("from"):
        hay = " ".join(str(meta.get(k) or "") for k in ("from", "reply_to", "organizer")).lower()
        if not any(n.lower() in hay for n in m["from"]):
            return False
    if m.get("label"):
        labels = {str(x).lower() for x in (meta.get("labels") or [])}
        if not any(n.lower() in labels for n in m["label"]):
            return False
    if m.get("words"):
        hay = " ".join(str(x or "") for x in (event.get("summary"), event.get("body"),
                                              meta.get("subject"), meta.get("snippet"))).lower()
        if not any(n.lower() in hay for n in m["words"]):
            return False
    return True


def context_of(event: dict[str, Any]) -> dict[str, Any]:
    """An event as a run's context item (fenced in the brief as information)."""
    meta = event.get("metadata") or {}
    body = event.get("body") or "\n".join(
        f"{k.replace('_', ' ').capitalize()}: {meta[k]}" for k in ("from", "subject", "snippet", "start")
        if meta.get(k))
    return {"id": event["id"], "source": event.get("source") or "outside",
            "summary": event.get("summary") or "", "body": str(body)[:BODY_MAX]}


# ── routing ────────────────────────────────────────────────────────────

def _listeners(store: fms.FirstMateStore, owner: str) -> list[tuple[dict[str, Any], list[dict[str, Any]]]]:
    """The owner's running factories with events triggers, highest priority first."""
    out = []
    for f in store.list_factories(owner):
        if f["state"] != "running":
            continue
        live = store.live(f["id"])
        triggers = [t for t in (live or {}).get("triggers", []) if t.get("kind") == "events"]
        if triggers:
            out.append((f, triggers))
    out.sort(key=lambda ft: (ft[0]["priority"], ft[0]["created_at"]))
    return out


def route(owner_id: str) -> int:
    """Claim the owner's new events that their running factories listen for.
    Returns how many were claimed. Never raises: the Arbiter calls it."""
    try:
        store = fms.get_store() if (fms._STORE is not None or fms._db_path().exists()) else None
        evs = _events_store()
        if store is None or evs is None or not store.settings()["enabled"]:
            return 0
        owner = fms.norm_user(owner_id)
        if not store.prefs(owner)["on"]:
            return 0
        listeners = _listeners(store, owner)
        if not listeners:
            return 0
        claimed = 0
        for ev in evs.list_new(owner, limit=100):
            hits = [(f, t) for f, triggers in listeners for t in triggers if match(t, ev)]
            if not hits:
                continue
            f, t = hits[0]  # listeners are in priority order
            others = sorted({h[0]["id"] for h in hits[1:]} - {f["id"]})
            if evs.claim_for(ev["id"], {"fm_factory": f["id"], "fm_trigger": _key(t), "fm_also": others}):
                claimed += 1
        return claimed
    except Exception:  # noqa: BLE001 — routing never breaks the Arbiter or a scan
        log.warning("first mate: routing events failed", extra={"owner": owner_id}, exc_info=True)
        return 0


def release_orphans(store: fms.FirstMateStore, owner: str) -> int:
    """Claimed events whose factory is gone or retired, or whose live version
    no longer has the trigger that claimed them, go back to ``new``."""
    evs = _events_store()
    if evs is None:
        return 0
    orphans = []
    for ev in evs.list_claimed(owner):
        meta = ev.get("metadata") or {}
        f = store.get(str(meta.get("fm_factory") or ""))
        live = store.live(f["id"]) if f and f["state"] != "retired" else None
        keys = {_key(t) for t in (live or {}).get("triggers", []) if t.get("kind") == "events"}
        if meta.get("fm_trigger") not in keys:
            orphans.append(ev["id"])
    return evs.unclaim(orphans)


def batches(store: fms.FirstMateStore, owner: str, now: datetime) -> list[dict[str, Any]]:
    """Each factory's claimed events whose batch window has passed:
    ``{factory_id, trigger, events, oldest}``, oldest batch first."""
    evs = _events_store()
    if evs is None:
        return []
    default = store.prefs(owner)["batch_minutes"]
    by_factory: dict[str, list[dict[str, Any]]] = {}
    for ev in evs.list_claimed(owner):
        by_factory.setdefault(str((ev.get("metadata") or {}).get("fm_factory") or ""), []).append(ev)
    out = []
    for fid, items in by_factory.items():
        live = store.live(fid) if fid else None
        if live is None:
            continue
        triggers = {_key(t): t for t in live.get("triggers", []) if t.get("kind") == "events"}
        trigger = triggers.get(str((items[0].get("metadata") or {}).get("fm_trigger") or ""))
        if trigger is None:
            continue  # release_orphans gives these back
        window = timedelta(minutes=trigger.get("batch_minutes") or default)
        oldest = datetime.fromisoformat(items[0]["processed_at"])
        if now - oldest < window:
            continue
        out.append({"factory_id": fid, "trigger": trigger, "events": items[:BATCH_EVENTS_MAX], "oldest": oldest})
    out.sort(key=lambda b: b["oldest"])
    return out


def settle(event_ids: list[str], run_id: str) -> int:
    evs = _events_store()
    return evs.settle_claimed(event_ids, {"fm_run": run_id}) if evs is not None else 0
