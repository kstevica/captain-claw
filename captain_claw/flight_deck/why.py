""""Why did this happen?" for one row of the tool-call register.

Re-runs the deck's real deciders on the facts the row recorded — what the
call was (tool, action, its parameters as recorded), who or what started it,
how it ended — with no model call, and gives one short verdict per layer:

- what it was: the risk class then, and today;
- who started it: a person in a chat, or a run that started on its own;
- approval: what happened then (the register's reason), and what the deck's
  approval rules would do with the same call today (``approvals.preview``);
- guard: whether the agent's own guard stopped it, and whether today's
  blast-radius rules match it (``blast_radius.classify_tool``; whether that
  gate is on lives in each agent's settings, so it says "if");
- outcome: how it ended.

A layer that would decide differently today says so (``changed``), so an
admin can see what a new setting changes before the next call. Parameters a
member's call keeps from the agent's owner stay hidden here too: those
layers say they judged the tool alone.
"""

from __future__ import annotations

from typing import Any

from captain_claw.approval_kinds import KIND_LABELS, KIND_VERBS, kind_for
from captain_claw.flight_deck import approvals
from captain_claw.flight_deck import person_asks as asks
from captain_claw.flight_deck.tool_risk import RISK_LABELS, risk_for

OUTCOME_WORDS = {
    "ok": "It ran",
    "error": "It ran and failed",
    "blocked": "It was stopped before it ran",
    "timeout": "It ran out of time",
    "aborted": "It was stopped while it ran",
    "interrupted": "The agent stopped while it ran (a restart or a crash)",
}


def _layer(key: str, title: str, then: str, today: str = "", changed: bool = False) -> dict[str, Any]:
    return {"layer": key, "title": title, "then": then, "today": today, "changed": changed}


def _source_words(row: dict[str, Any]) -> tuple[str, bool]:
    from captain_claw import turn_source

    source = str(row.get("source") or "")
    label = str(row.get("source_label") or "") or turn_source.label_for(source)
    who = ""
    speaker = row.get("speaker") or {}
    if speaker.get("label") or speaker.get("id"):
        who = "you" if speaker.get("you") else str(speaker.get("label") or "a member")
    elif (row.get("owner") or {}).get("name"):
        who = "you" if row["owner"].get("you") else str(row["owner"]["name"])
    attended = asks.attended(source)
    if not source:
        return "Nothing recorded what started it.", True
    if attended:
        place = f" ({label})" if label else ""
        return f"A turn {who + ' started' if who else 'a person started'}{place}.", True
    return f"It started on its own: {label or source}. Nobody was there to answer a question.", False


def _approval_then(reason: str, outcome: str) -> tuple[str, str]:
    """(sentence, class) for what happened then: allowed / refused / not asked."""
    if reason.startswith("Approval ("):
        text = reason.split("): ", 1)[1] if "): " in reason else reason
        return text, "refused" if outcome == "blocked" else "allowed"
    # Stopped by something else (a guard, a policy): approval wasn't what stopped it.
    return "No approval was recorded for it.", "not asked"


def _today_class(decided: dict[str, Any]) -> str:
    if decided["decision"] == "deny":
        return "refused"
    if decided["decision"] == "ask":
        return "asked"
    return "not asked" if decided.get("scope") == "off" else "allowed"


async def explain(db: Any, row: dict[str, Any]) -> dict[str, Any]:
    """The verdicts for one shaped register row (``tool_audit_routes._shape``)."""
    tool = str(row.get("tool") or "")
    hidden = bool(row.get("hidden"))
    params = row.get("params") if isinstance(row.get("params"), dict) else {}
    action = str(params.get("action") or "") if params else ""
    outcome = str(row.get("outcome") or "")
    reason = str(row.get("reason") or "")
    layers: list[dict[str, Any]] = []
    hidden_note = " Its details are kept from you, so this judges the tool alone." if hidden else ""

    # What it was.
    risk_then = str(row.get("risk") or "other")
    risk_now = risk_for(tool, params)
    what = f"{tool}{' · ' + action if action else ''} — {RISK_LABELS.get(risk_then, risk_then)}."
    layers.append(_layer("what", "What it was", what,
                         f"Today it would be classed {RISK_LABELS.get(risk_now, risk_now)}."
                         if risk_now != risk_then and not hidden else "", risk_now != risk_then and not hidden))

    # Who started it.
    who_text, attended = _source_words(row)
    layers.append(_layer("who", "Who started it", who_text))

    # Approval.
    kind = kind_for(tool, params)
    then_text, then_class = _approval_then(reason, outcome)
    if not kind:
        layers.append(_layer("approval", "Approval",
                             then_text if then_class != "not asked" else
                             "Not a call that needs approval (paying, sending, posting, sharing, deleting, "
                             "passwords)." + hidden_note))
    else:
        person = str((row.get("speaker") or {}).get("id") or (row.get("owner") or {}).get("id") or "")
        agent = str((row.get("agent") or {}).get("slug") or "")
        decided = await approvals.preview(db, user_id=person, agent=agent, kind=kind,
                                          origin=str(row.get("source") or ""),
                                          session_id=str(row.get("session_id") or ""))
        today_class = _today_class(decided)
        verb = {"refused": "be refused", "asked": "ask first", "allowed": "go ahead",
                "not asked": "go ahead without asking"}[today_class]
        today = f"Today it would {verb}: {decided.get('why') or ''}".strip()
        # Different only when today's rules settle it the other way on their
        # own; "ask first" depends on the person, so it never counts.
        ran_then = then_class != "refused"
        ran_today = {"refused": False, "asked": None}.get(today_class, True)
        changed = ran_today is not None and ran_today != ran_then
        layers.append(_layer("approval", f"Approval · {KIND_LABELS[kind].lower()}",
                             then_text + hidden_note, today, changed))

    # The agent's own guard.
    from captain_claw.flight_deck.blast_radius import classify_tool

    hit, why_hit = classify_tool(tool, params)
    stopped = reason.startswith(("Blast radius:", "Guard:"))
    guard_then = (f"The agent's guard stopped it: {reason}" if stopped
                  else "The agent's guards didn't stop it.")
    guard_today = (f"Today's blast-radius rules match it ({why_hit}): an agent with that gate on stops it "
                   "or asks first." if hit else "Today's blast-radius rules don't match it.")
    # Changed only when its rules no longer match a call they stopped; whether
    # a gate that matches now is on is each agent's setting, so not judged.
    layers.append(_layer("guard", "Guard", guard_then, guard_today, stopped and not hit))

    # How it ended.
    ended = OUTCOME_WORDS.get(outcome, "Still running, or its end wasn't recorded")
    if reason and not reason.startswith(("Approval (",)) and outcome != "ok":
        ended += f": {reason}"
    if row.get("duration_ms") is not None and outcome in {"ok", "error", "timeout"}:
        ended += f" (took {max(0, int(row['duration_ms'])) / 1000:.1f} s)"
    layers.append(_layer("outcome", "How it ended", ended + "."))

    return {"id": row.get("id"), "summary": _summary(outcome, reason, kind, attended), "layers": layers,
            "changed": any(layer["changed"] for layer in layers)}


def _summary(outcome: str, reason: str, kind: str, attended: bool) -> str:
    if outcome == "blocked":
        if reason.startswith("Approval ("):
            return "It didn't run: it needed approval and didn't get it."
        if reason.startswith(("Blast radius:", "Guard:")):
            return "It didn't run: the agent's guard stopped it."
        return "It didn't run: a rule stopped it before it started."
    if reason.startswith("Approval ("):
        return "It ran because it was allowed."
    if kind:
        verb = KIND_VERBS.get(kind, kind)
        return (f"It ran without asking: when it ran, this deck didn't ask before agents {verb}."
                if attended else f"It ran on its own: this deck didn't ask before agents {verb}.")
    if outcome == "ok":
        return "It ran: nothing on this deck asks about this kind of call."
    return OUTCOME_WORDS.get(outcome, "Its end wasn't recorded") + "."
