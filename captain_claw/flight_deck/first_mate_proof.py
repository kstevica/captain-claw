"""First Mate's proof: what a carry-out really did, against what was approved.

See docs/first-mate-plan.md §6 (Phase 3, piece 3). A carry-out's reply is only
its word. When one ends, its workers' calls are read from the tool-call
register: kept by each agent's own process, never written by its model. Gmail
and Calendar are then asked whether they have what the register says went
through (:mod:`taskboard_proof`). The verdict, :func:`judge`, has no model in
it:

* **gone**: on record, but the user's Google account doesn't have it;
* **more mail than approved**: more sends than the approved mail intents, or
  than the charter's ``sends_per_run``;
* **something not approved**: an outward call (the rehearsal guard's own rules
  tell outward from the run's own files) of a kind no approved intent was;
* **a claim the record doesn't bear out**: the reply says it sent or changed
  something that isn't on record (``taskboard_proof.check``).

``confirmed``: it did outward things, all of them approved and on record.
``nothing``: it did nothing outward. ``doubt``: any of the above, with why.
``unknown``: the register couldn't be read; the run stands on its contract
and reviewers alone.

A doubt lowers the factory's trust one level and asks the owner
(:func:`prove`). It never raises trust: that's the owner's, by hand, until
Phase 4's calibration.

:func:`covered` is what trust ``commit`` may carry out by itself: intents
within the charter's mail and organize grants, at most ``sends_per_run``
emails. Everything else waits for the owner.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any

from captain_claw.flight_deck import first_mate_store as fms

log = logging.getLogger(__name__)

CONFIRMED, NOTHING, DOUBT, UNKNOWN = "confirmed", "nothing", "doubt", "unknown"
_MAIL_TOOLS = ("google_mail", "microsoft_mail", "imap_mail")
_SENDS = frozenset({"send", "send_draft", "reply", "forward", "create_draft", "update_draft"})
_LOWER = {"direct": "commit", "commit": "propose"}
ACTIONS_KEPT = 20


def _params(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    try:
        data = json.loads(raw or "{}")
    except (TypeError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def grant_of(tool: str, action: str) -> str | None:
    """The charter grant a call needs: ``email`` (writing or sending mail),
    ``organize`` (changing the mailbox), or None (no grant covers it here)."""
    from captain_claw.mail_authority import MAILBOX_CHANGE_ACTIONS
    if tool == "send_mail":
        return "email"
    if tool in _MAIL_TOOLS:
        if action in _SENDS:
            return "email"
        if action in MAILBOX_CHANGE_ACTIONS:
            return "organize"
    return None


def _kind_key(tool: str, action: str) -> str:
    """What two calls must share to be "the same kind": the grant, else the tool."""
    return grant_of(tool, action) or str(tool)


def _intent_action(intent: dict[str, Any]) -> str:
    return str((intent.get("args") or {}).get("action") or "")


def covered(intents: list[dict[str, Any]], grants: dict[str, Any]) -> list[int]:
    """The open intents trust ``commit`` may carry out by itself."""
    granted = {g for g in ("email", "organize") if grants.get(g)}
    cap = int(grants.get("sends_per_run") or 0)
    ids, mails = [], 0
    for i in intents:
        if i.get("status") != "open" or i.get("kind") != "intent":
            continue  # a browser commit point always waits for the owner
        g = grant_of(str(i.get("tool") or ""), _intent_action(i))
        if g not in granted:
            continue
        if g == "email":
            if mails >= cap:
                continue
            mails += 1
        ids.append(int(i["id"]))
    return ids


def judge(rows: list[dict[str, Any]], actions: list[dict[str, Any]], approved: list[dict[str, Any]],
          grants: dict[str, Any], *, result: str, status: str) -> dict[str, Any]:
    """The verdict on one carry-out, from its register rows, the actions read
    from them (and checked at the source), and the intents it was approved for."""
    from captain_claw import rehearsal_turn
    from captain_claw.flight_deck import taskboard_proof
    outward = [r for r in rows if str(r.get("outcome") or "") == "ok"
               and rehearsal_turn.classify(str(r.get("tool") or ""), _params(r.get("params")))[0] != "allow"]
    allowed = {_kind_key(str(i.get("tool") or ""), _intent_action(i)) for i in approved}
    sent = [a for a in actions if a["kind"] == "email" and a["outcome"] == "ok"]
    approved_mail = sum(1 for i in approved if grant_of(str(i.get("tool") or ""), _intent_action(i)) == "email")
    cap = int(grants.get("sends_per_run") or 0)
    limit = min(approved_mail, cap) if cap else approved_mail
    notes: list[str] = []
    gone = next((a for a in actions if a.get("seen") == taskboard_proof.GONE), None)
    if gone:
        notes.append(f"It is on record, but your Google account doesn't have it: {gone['label']}.")
    if len(sent) > limit:
        notes.append(f"It sent {len(sent)} {'email' if len(sent) == 1 else 'emails'}; you approved {limit}.")
    for r in outward:
        params = _params(r.get("params"))
        if _kind_key(str(r.get("tool") or ""), str(params.get("action") or "")) not in allowed:
            what = rehearsal_turn.classify(str(r.get("tool") or ""), params)[1] or str(r.get("tool") or "")
            notes.append(f"It did something you didn't approve: {what}.")
            break
    tp = taskboard_proof.check(actions, reply=result or "", words="", outcome="ok")
    if tp["state"] == taskboard_proof.NOT_DONE and tp["note"] and tp["note"] not in notes:
        notes.append(tp["note"])
    state = DOUBT if notes else (CONFIRMED if outward else NOTHING)
    return {"state": state, "note": notes[0] if notes else "", "notes": notes, "sent": len(sent),
            "approved": len(approved), "outward": len(outward), "status": status,
            "actions": [a["label"] for a in actions][:ACTIONS_KEPT]}


async def _rows(run: dict[str, Any]) -> list[dict[str, Any]] | None:
    """The carry-out's workers' calls from the register, or None when it can't be read."""
    if not run.get("child_id") or not run.get("started_at"):
        return []
    try:
        from captain_claw.flight_deck import tool_audit_routes
        from captain_claw.flight_deck.captain_swarm_outcome import worker_prefix
        audit = tool_audit_routes.store()
        prefix = worker_prefix(run["child_id"])
        since = datetime.fromisoformat(run["started_at"]).timestamp() - 5
        return [r async for r in audit.iterate({"since": since}) if str(r.get("agent_slug") or "").startswith(prefix)]
    except Exception:  # noqa: BLE001
        log.warning("first mate: couldn't read the register for a carry-out", extra={"run": run.get("id")},
                    exc_info=True)
        return None


async def prove(store: fms.FirstMateStore, run: dict[str, Any]) -> dict[str, Any]:
    """Prove a finished carry-out, keep the proof on the run, and on a doubt lower
    the factory's trust one level and ask the owner. Never raises."""
    try:
        rows = await _rows(run)
        if rows is None:
            proof = {"state": UNKNOWN, "note": "The tool-call register couldn't be read, so this carry-out "
                                               "rests on its own checks and reviewers."}
        else:
            from captain_claw.flight_deck import taskboard_proof
            actions = [a for a in (taskboard_proof._action(r) for r in rows) if a]
            try:
                await taskboard_proof.confirm(run["owner_id"], actions)
            except Exception:  # noqa: BLE001 — the record stands
                log.info("first mate: couldn't ask Google about a carry-out", exc_info=True)
            ids = {int(x) for x in run.get("trigger_ref") or [] if str(x).isdigit()}
            approved = [i for i in store.intents(run.get("parent_run") or "") if i["id"] in ids]
            version = store.version(run["factory_id"], run["version"]) or {}
            proof = judge(rows, actions, approved, version.get("grants") or {},
                          result=run.get("result", ""), status=run["status"])
        store.set_proof(run["id"], proof)
        if proof["state"] == DOUBT:
            factory = store.get(run["factory_id"])
            if factory:
                lower = _LOWER.get(factory["trust"])
                if lower:
                    store.set_trust(factory["id"], lower, "deck")
                from captain_claw.flight_deck import first_mate_line
                await first_mate_line.proof_doubt(store.get(factory["id"]), run, proof["note"])
        return proof
    except Exception:  # noqa: BLE001 — a proof never breaks a refresh
        log.warning("first mate: proving a carry-out failed", extra={"run": run.get("id")}, exc_info=True)
        return {"state": UNKNOWN, "note": "the proof couldn't be made"}
