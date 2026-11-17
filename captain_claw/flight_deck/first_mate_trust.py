"""First Mate's trust ladder: calibration against the owner, and trust raised by the record.

See docs/first-mate-plan.md §6 and §8 (Phase 4, piece 3).

**Calibration.** Every finished run gets the deck's own verdict, made with no
word from the owner: the run failed, its rehearsal reached outside, or its
carry-out's proof is doubtful → bad; it finished → good. Where the owner said
something about the same run, that is their label: 👍/👎 first, then approving
or rejecting what a rehearsal would do, then what became of its drafts (sent,
mostly kept → good; deleted or mostly rewritten → bad). ``label_eval``'s
Cohen's kappa over the owner's most recent labelled runs, across their
factories, says whether the deck's verdict can stand in for the owner's.
Calibrated: kappa ≥ :data:`KAPPA_BAR` on at least :data:`LABELS_BAR` labels,
the same bar Captain Swarm uses. When the owner and the deck called every run
good, kappa can't be worked out: nothing shows yet that the deck would catch
a bad one, so that isn't calibrated either.

**The record.** What the factory did or proposed at its current trust level,
since trust last moved (any move: the owner's, a doubt, a version that added
a grant), newest first:

* a carry-out: what the owner did with its result (👍/👎, its drafts), else
  the deck's verdict; a proposal the owner approved only in part counts by the
  share approved;
* a proposal the owner rejected outright: 0.

The factory moves up one level when its last :data:`RECORD_RUNS` items average
at least :data:`RECORD_VALUE`, none had a doubtful proof, the owner is
calibrated, the factory is running and its level isn't pinned. ``direct`` stays
out of reach while the live version may send mail, spend or open accounts.
The owner is told on the Line, and may set it back and pin it.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from captain_claw.flight_deck import first_mate_store as fms
from captain_claw.flight_deck import label_eval
from captain_claw.flight_deck.first_mate_store import FirstMateError

log = logging.getLogger(__name__)

LABELS_BAR = 30
KAPPA_BAR = 0.6
LABEL_WINDOW = 200
RECORD_RUNS = 10
RECORD_VALUE = 0.9
KEPT = 0.7  # a draft sent with at least this much kept is a good result
_NEXT = {"propose": "commit", "commit": "direct"}
_THUMBS = ("thumbs_up", "thumbs_down")
_DECISIONS = ("approved", "rejected")
_DRAFTS = ("sent_as_written", "sent_edited", "draft_deleted")
_OUTCOMES = _THUMBS + _DRAFTS
_RUNS_SCANNED = 300


# ── labels ─────────────────────────────────────────────────────────────

def deck_verdict(run: dict[str, Any]) -> int | None:
    """The deck's own verdict on a finished run: 1 good, 0 bad, None unfinished."""
    if run["status"] not in ("done", "failed"):
        return None
    if run["status"] == "failed" or run.get("verdict") in ("fail", "leak"):
        return 0
    if run["kind"] == "carry_out" and (run.get("proof") or {}).get("state") == "doubt":
        return 0
    return 1


def owner_label(signals: list[dict[str, Any]]) -> int | None:
    """The owner's label for one run from its signals (newest first): 👍/👎,
    then approve/reject, then what became of its drafts."""
    for kinds in (_THUMBS, _DECISIONS):
        for s in signals:
            if s["kind"] in kinds:
                return 1 if float(s["value"]) >= 0.5 else 0
    drafts = [float(s["value"]) for s in signals if s["kind"] in _DRAFTS]
    if drafts:
        return 1 if sum(drafts) / len(drafts) >= KEPT else 0
    return None


def _by_run(store: fms.FirstMateStore, factory_id: str) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for s in store.signals(factory_id, limit=500):  # newest first
        if s.get("run_id"):
            out.setdefault(s["run_id"], []).append(s)
    return out


def labelled(store: fms.FirstMateStore, factory_id: str) -> list[dict[str, Any]]:
    """The factory's finished runs that have both labels, newest first."""
    signals = _by_run(store, factory_id)
    out = []
    for run in store.list_runs(factory_id, limit=_RUNS_SCANNED, kinds=fms.WORK_KINDS):
        deck, owner = deck_verdict(run), owner_label(signals.get(run["id"], []))
        if deck is not None and owner is not None:
            out.append({"run_id": run["id"], "factory_id": factory_id, "at": run["created_at"],
                        "deck": deck, "owner": owner})
    return out


def calibration(store: fms.FirstMateStore, owner_id: str) -> dict[str, Any]:
    """How well the deck's verdict agrees with the owner's labels."""
    rows: list[dict[str, Any]] = []
    for f in store.list_factories(owner_id, include_retired=True):
        rows += labelled(store, f["id"])
    rows.sort(key=lambda r: r["at"], reverse=True)
    rows = rows[:LABEL_WINDOW]
    agree = label_eval.agreement((r["deck"], r["owner"]) for r in rows)
    n, kappa = agree["n"], agree["kappa"]
    ready = n >= LABELS_BAR and kappa is not None and kappa >= KAPPA_BAR
    if ready:
        why = ""
    elif n < LABELS_BAR:
        why = f"{n} of the {LABELS_BAR} labels it needs so far"
    elif kappa is None:
        why = "your labels and its checks called every result good, so it can't tell yet whether it would catch a bad one"
    else:
        why = f"its checks agree with you only so far (kappa {kappa:.2f}, it needs {KAPPA_BAR})"
    return {"n": n, "kappa": kappa, "agreement": agree["agreement"], "confusion": agree["confusion"],
            "ready": ready, "why": why, "labels_bar": LABELS_BAR, "kappa_bar": KAPPA_BAR}


# ── the record ─────────────────────────────────────────────────────────

def since_trust_moved(store: fms.FirstMateStore, factory_id: str) -> str:
    """When the factory's trust level last changed (ISO), or "" for never."""
    for e in store.events(factory_id, limit=500):
        if e["event"] == "trust" or (e["event"] == "approved" and "trust" in (e.get("detail") or {})):
            return e["at"]
    return ""


def _share(store: fms.FirstMateStore, rehearsal_id: str) -> float:
    """How much of a proposal the owner (or the grants) let through."""
    intents = store.intents(rehearsal_id) if rehearsal_id else []
    let = sum(1 for i in intents if i["status"] in ("approved", "carried_out"))
    said_no = sum(1 for i in intents if i["status"] == "rejected")
    return let / (let + said_no) if let + said_no else 1.0


def record(store: fms.FirstMateStore, factory: dict[str, Any], calibrated: dict[str, Any] | None = None
           ) -> dict[str, Any]:
    """The factory's record at its trust level, and whether it moves up."""
    level = factory["trust"]
    nxt = _NEXT.get(level)
    since = since_trust_moved(store, factory["id"])
    signals = _by_run(store, factory["id"])
    items: list[dict[str, Any]] = []
    for run in store.list_runs(factory["id"], limit=_RUNS_SCANNED, kinds=fms.WORK_KINDS):
        if len(items) >= RECORD_RUNS or (since and run["created_at"] <= since):
            break
        if run["status"] not in ("done", "failed") or run["kind"] == "run":
            continue  # a direct run is already at the top
        sig = signals.get(run["id"], [])
        if run["kind"] == "rehearsal":
            intents = store.intents(run["id"])
            if any(i["status"] == "rejected" for i in intents) and not any(
                    i["status"] in ("approved", "carried_out") for i in intents):
                items.append({"run_id": run["id"], "kind": "rejected", "value": 0.0, "doubt": False,
                              "from": "you"})
            continue  # otherwise its carry-out is the item, or it needed nothing outside
        outcomes = [float(s["value"]) for s in sig if s["kind"] in _OUTCOMES]
        doubt = (run.get("proof") or {}).get("state") == "doubt"
        base = sum(outcomes) / len(outcomes) if outcomes else float(deck_verdict(run) or 0)
        value = 0.0 if doubt else base * _share(store, run["parent_run"])
        items.append({"run_id": run["id"], "kind": run["kind"], "value": round(value, 3), "doubt": doubt,
                      "from": "you" if outcomes else "its checks"})
    value = round(sum(i["value"] for i in items) / len(items), 3) if items else None
    doubts = sum(1 for i in items if i["doubt"])
    grants = (store.live(factory["id"]) or {}).get("grants") or {}
    cal = calibrated
    if nxt is None:
        why = "It's at the top: it acts directly."
    elif nxt == "direct" and fms.outward(grants):
        why = "It may send mail, spend or open accounts, so it always rehearses first."
        nxt = None
    elif factory["trust_pinned"]:
        why = "You pinned its trust level: its record won't raise it."
    elif factory["state"] != "running":
        why = "It isn't running."
    elif len(items) < RECORD_RUNS:
        why = f"{len(items)} of the {RECORD_RUNS} settled results it needs at this level so far."
    elif doubts:
        why = "A doubtful carry-out among its last results."
    elif value is not None and value < RECORD_VALUE:
        why = f"Its last {RECORD_RUNS} results scored {value:.0%}; it needs {RECORD_VALUE:.0%}."
    elif cal is not None and not cal["ready"]:
        why = f"Its record is good enough, but trust moves on its own only once its checks agree with you: {cal['why']}."
    else:
        why = ""
    eligible = bool(nxt) and not why and cal is not None and cal["ready"]
    return {"level": level, "next": nxt, "since": since, "items": items, "counted": len(items),
            "needed": RECORD_RUNS, "value": value, "bar": RECORD_VALUE, "doubts": doubts,
            "pinned": factory["trust_pinned"], "eligible": eligible, "why": why}


async def consider(store: fms.FirstMateStore, factory: dict[str, Any], calibrated: dict[str, Any]) -> bool:
    """Raise the factory one level when its record says so. True when it moved."""
    rec = record(store, factory, calibrated)
    if not rec["eligible"]:
        return False
    try:
        moved = store.set_trust(factory["id"], rec["next"], "record")
    except FirstMateError as exc:
        log.info("first mate: the record couldn't raise trust", extra={"factory": factory["id"], "why": str(exc)})
        return False
    from captain_claw.flight_deck import first_mate_line
    await first_mate_line.trust_raised(moved, rec, calibrated)
    return True


async def consider_owner(store: fms.FirstMateStore, owner_id: str, now: datetime | None = None) -> int:
    """Every running factory of one owner, once their labels are calibrated."""
    factories = [f for f in store.list_factories(owner_id) if f["state"] == "running" and f["trust"] in _NEXT
                 and not f["trust_pinned"]]
    if not factories:
        return 0
    cal = calibration(store, owner_id)
    if not cal["ready"]:
        return 0
    moved = 0
    for f in factories:
        moved += int(await consider(store, f, cal))
    return moved
