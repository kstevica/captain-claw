"""Did the agent really do what it was handed?

An agent's reply is only its word. When a task the user handed over
(``taskboard_handover``) has had its run, triage checks that word against what
a model can't talk into existing:

* the tool-call register: what the tools answered in that turn, kept by the
  agent's process rather than written by its model (:func:`recorded`);
* the source itself, read with the user's own token: Gmail has the message
  among what was sent, Google Calendar has the event (:func:`confirm`). Only who
  and what are kept. Outlook, chats and the rest aren't read back, so their
  record stands alone.

:func:`check` says, with no model, what those show against the reply: an email,
a calendar change or a WhatsApp message it says it made with none on record
(only a draft, say), something tried that never got through, something the
source doesn't have. Whether that is all the user asked for, in whatever words
and for whatever kind of work, is Jev's to say from the same record
(:func:`judged`).

The verdict is ``flags.proof``. A doubtful one keeps the task with the user: it
isn't closed, "probably done" or waiting on anyone, whatever the reply says
(``taskboard_triage._with_proof``). Nothing here sends anything or asks the agent
to try again: a second copy of an email is worse than a false "Sent", and whether
to try again is the user's to say.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from typing import Any

from captain_claw import fabrication_guard
from captain_claw.logging import get_logger
from captain_claw.tool_audit import clean_refs

log = get_logger(__name__)

PROVEN = "proven"        # all that was asked is on record, by Jev's reading
RECORDED = "recorded"    # something is on record, and nobody compared it with what was asked
ANSWERED = "answered"    # nothing is on record, and the reply claims nothing it should show
UNSURE = "unsure"        # on record, but Jev isn't sure it is all, or another agent did part of it
NOT_DONE = "not_done"    # what it says, or what was asked, isn't on record or at the source
FAILED = "failed"        # it was tried and never got through
NO_RECORD = "no_record"  # the agent reported no turn, so there is nothing to check it against
DOUBTS = (UNSURE, NOT_DONE, FAILED)

NO_RECORD_NOTE = ("The agent didn't report what it did, so this can't be checked. Restart it to turn "
                  "that on, and check the result yourself.")
_NOT_ASKED = "Jev doesn't find what you asked for in what the agent did."
_NOT_SURE = "Jev isn't sure all you asked for was done."
_PASSED_ON = "It passed this to another agent, and what that agent did isn't checked here."
_ENDED_EARLY = "Its run ended before it finished."

GONE = "gone"             # ``seen``: the source was asked and doesn't have it
_SEEN_GMAIL = "gmail"
_SEEN_CALENDAR = "calendar"
_DID_YES, _DID_NO = 0.8, 0.35
_LATE_CALLS_S = 300.0     # a turn's calls the deck hasn't got this long after it ended aren't waited for
_LOOK_AGAIN_S = 30.0
_CALLS_MAX = 200
_ACTIONS_KEPT = 12
_CONFIRMS_MAX = 6
_TIMEOUT_S = 10.0
_GMAIL = "https://gmail.googleapis.com/gmail/v1/users/me/messages"
_CALENDAR = "https://www.googleapis.com/calendar/v3/calendars"

_MAIL_TOOLS = ("google_mail", "microsoft_mail")
_MAIL_SENDS = ("send", "send_draft", "reply", "forward")
_MAIL_DRAFTS = ("create_draft", "update_draft")
_CHAT_TOOLS = ("whatsapp_web", "whatsapp_send_file")
_DELEGATES = ("consult_peer", "botport", "flight_deck")
# The agent's own housekeeping: no part of what it was asked to do out there.
_HOUSEKEEPING = ("insights", "facts", "intentions", "personality", "project_memory", "topics", "playbooks",
                 "reasoning_effort", "clipboard", "todo", "jev", "ask_human")
_CHECKED_KINDS = {"email": "The email", "draft": "The draft", "calendar": "The calendar change", "message": "The message"}
_CALENDAR_LABELS = {"create_event": "Calendar event created", "update_event": "Calendar event updated",
                    "delete_event": "Calendar event deleted"}


def _json(raw: Any) -> dict:
    if isinstance(raw, dict):
        return raw
    try:
        loaded = json.loads(raw or "{}")
    except (TypeError, ValueError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _text(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        value = ", ".join(str(v) for v in value)
    return " ".join(str(value or "").split())


def _kind(tool: str, action: str, risk: str) -> str:
    """What a call is to the user: ``email``, ``draft``, ``calendar``, ``message``,
    ``delegated`` or ``other``; "" for what only looks, or the agent's housekeeping."""
    if risk == "read" or tool in _HOUSEKEEPING:
        return ""
    if tool in _DELEGATES:
        return "delegated"
    if tool == "send_mail":
        return "email"
    if tool in _MAIL_TOOLS:
        return "email" if action in _MAIL_SENDS else "draft" if action in _MAIL_DRAFTS else "other"
    if fabrication_guard.is_calendar_tool(tool):
        return "calendar"
    return "message" if tool in _CHAT_TOOLS else "other"


def _label(kind: str, tool: str, action: str, params: dict) -> str:
    """The call in a few words, from what it was given (masked by the register)."""
    if kind in ("email", "draft"):
        to, subject = _text(params.get("to")), _text(params.get("subject"))
        head = {"email": "Forwarded to" if action == "forward" else "Email to", "draft": "Draft to"}[kind]
        if to:
            return f"{head} {to}" + (f": {subject}" if subject else "")
        return ("Email reply" if kind == "email" else "Draft") + (f": {subject}" if subject else "")
    if kind == "calendar":
        title = _text(params.get("summary"))
        return _CALENDAR_LABELS.get(action, "Calendar change") + (f": {title}" if title else "")
    if kind == "message":
        chat = _text(params.get("chat") or params.get("to"))
        return f"WhatsApp to {chat}" if chat else "WhatsApp message"
    if kind == "delegated":
        return f"Passed to another agent ({tool})"
    return f"{tool} {action}".strip()


def _action(row: dict) -> dict | None:
    params = _json(row.get("params"))
    action = params.get("action") if isinstance(params.get("action"), str) else ""
    tool = str(row.get("tool") or "")
    kind = _kind(tool, action, str(row.get("risk") or ""))
    if not kind:
        return None
    refs = clean_refs(_json(row.get("refs")))
    return {"kind": kind, "tool": tool, "action": action, "label": _label(kind, tool, action, params)[:200],
            "outcome": str(row.get("outcome") or ""), "reason": _text(row.get("reason"))[:200], "seen": "",
            "ref": refs.get("gmail_message_id") or refs.get("calendar_event_id") or "",
            "calendar": _text(params.get("calendar_id"))[:200]}


def recorded(turn: dict) -> tuple[list[dict], bool]:
    """``(the actions the register has of *turn*, whether that is all of them)``:
    fewer calls ended than its close counted means the rest are on their way."""
    from captain_claw.flight_deck import taskboard_ingest

    expected = int(turn.get("tool_calls") or 0)
    path = taskboard_ingest._audit_path()
    if not path.is_file() or "turn_id" not in {r["name"] for r in taskboard_ingest._rows(path, "PRAGMA table_info(tool_calls)")}:
        return [], not expected
    rows = taskboard_ingest._rows(path, "SELECT * FROM tool_calls WHERE turn_id = ? AND agent_slug = ? ORDER BY id LIMIT ?",
                                  (turn["turn_id"], turn["agent_slug"], _CALLS_MAX))
    ended = [r for r in rows if r.get("outcome") != "running"]
    return [a for a in (_action(r) for r in ended) if a], len(ended) >= expected


# ── The source ────────────────────────────────────────────────────────


async def _token(uid: str) -> str | None:
    from captain_claw.flight_deck.google_oauth_routes import get_valid_google_access_token

    return await get_valid_google_access_token(uid)


async def _fetch(token: str, url: str, params: list[tuple[str, str]] | None = None) -> tuple[int, dict]:
    """``(HTTP status, its JSON)``; ``(0, {})`` when it couldn't be asked."""
    import httpx

    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT_S) as client:
            r = await client.get(url, params=params, headers={"Authorization": f"Bearer {token}"})
        return r.status_code, r.json() if r.content else {}
    except (httpx.HTTPError, ValueError) as exc:
        log.info("Taskboard proof: source not read", error=type(exc).__name__)
        return 0, {}


def _when(start: dict) -> str:
    raw = str(start.get("dateTime") or start.get("date") or "")
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return raw[:40]
    return parsed.strftime("%Y-%m-%d %H:%M") if "T" in raw else parsed.strftime("%Y-%m-%d")


async def _confirm_mail(token: str, a: dict) -> None:
    status, body = await _fetch(token, f"{_GMAIL}/{a['ref']}", [
        ("format", "metadata"), ("metadataHeaders", "To"), ("metadataHeaders", "Subject")])
    if status == 404 or (status == 200 and "SENT" not in (body.get("labelIds") or [])):
        a["seen"] = GONE
    elif status == 200:
        heads = {str(h.get("name")).lower(): _text(h.get("value")) for h in (body.get("payload") or {}).get("headers") or []}
        a["seen"] = _SEEN_GMAIL
        if heads.get("to"):
            a["label"] = f"Email to {heads['to']}" + (f": {heads['subject']}" if heads.get("subject") else "")


async def _confirm_event(token: str, a: dict) -> None:
    status, body = await _fetch(token, f"{_CALENDAR}/{a['calendar'] or 'primary'}/events/{a['ref']}")
    if status in (404, 410) or (status == 200 and body.get("status") == "cancelled"):
        a["seen"] = GONE
    elif status == 200:
        a["seen"] = _SEEN_CALENDAR
        title = _text(body.get("summary"))
        people = [_text(p.get("email")) for p in body.get("attendees") or [] if not p.get("self")]
        a["label"] = (_CALENDAR_LABELS.get(a["action"], "Calendar event") + (f": {title}" if title else "")
                      + (f", {_when(body.get('start') or {})}" if body.get("start") else "")
                      + (f", with {', '.join(people[:4])}" if people else ""))[:200]


async def confirm(uid: str, actions: list[dict]) -> None:
    """Ask Google about what the register says went through, in place: ``seen``
    says the source has it, or doesn't (:data:`GONE`). Left as it was when it
    can't be asked (no token, no scope, an outage): the record stands then."""
    mine = [a for a in actions if a["outcome"] == "ok" and a["ref"]
            and ((a["tool"] == "google_mail" and a["kind"] == "email")
                 or (a["tool"] == "google_calendar" and a["action"] in ("create_event", "update_event")))][:_CONFIRMS_MAX]
    if not mine:
        return
    try:
        token = await _token(uid)
    except Exception as exc:  # noqa: BLE001 — the record stands
        log.info("Taskboard proof: no token", user=uid, error=type(exc).__name__)
        return
    if not token:
        return
    for a in mine:
        await (_confirm_mail(token, a) if a["tool"] == "google_mail" else _confirm_event(token, a))


# ── The verdict ───────────────────────────────────────────────────────


def _doubt(actions: list[dict], reply: str, words: str, outcome: str) -> tuple[str, str]:
    """``(state, why)`` when what is on record doesn't bear the reply out, else ``("", "")``."""
    gone = next((a for a in actions if a["seen"] == GONE), None)
    if gone:
        return NOT_DONE, f"It is on record, but your Google account doesn't have it: {gone['label']}."[:300]
    ok = {a["kind"] for a in actions if a["outcome"] == "ok"}
    said = ""
    if fabrication_guard.claims_mail_sent(reply, request=words) and "email" not in ok:
        said = ("It says it sent the email, but only a draft was made." if "draft" in ok
                else "It says it sent the email, but no send is on record.")
    elif fabrication_guard.claims_calendar_change(reply, request=words) and "calendar" not in ok:
        said = "It says it changed your calendar, but no change is on record."
    elif fabrication_guard.claims_whatsapp_sent(reply) and "message" not in ok:
        said = "It says it sent the WhatsApp message, but no send is on record."
    if said and "delegated" not in ok:
        return NOT_DONE, said
    for kind, word in _CHECKED_KINDS.items():
        failed = next((a for a in actions if a["kind"] == kind and a["outcome"] != "ok"), None)
        if failed and kind not in ok:
            why = failed["reason"] or failed["outcome"] or "no reason given"
            return FAILED, f"{word} didn't go through: {why}"[:300]
    if "delegated" in ok:
        return UNSURE, _PASSED_ON
    return (FAILED, _ENDED_EARLY) if outcome != "ok" else ("", "")


def check(actions: list[dict], *, reply: str, words: str, outcome: str = "ok", turn_id: str = "") -> dict:
    """What the record and the source show of a hand-over's run, with no model."""
    state, note = _doubt(actions, reply, words, outcome)
    if not state:
        state = RECORDED if any(a["outcome"] == "ok" for a in actions) else ANSWERED
    return {"state": state, "note": note, "actions": actions[:_ACTIONS_KEPT], "more": max(0, len(actions) - _ACTIONS_KEPT),
            "calls": len(actions), "turn": turn_id}


def judged(proof: dict, did: float | None) -> dict:
    """*proof* with Jev's word on whether what the user asked was done, from the
    record alone. It never overrules what the record itself shows is wrong."""
    if did is None or proof["state"] in DOUBTS:
        return proof
    if did >= _DID_YES:
        return {**proof, "state": PROVEN, "note": ""}
    return {**proof, "state": NOT_DONE if did < _DID_NO else UNSURE, "note": _NOT_ASKED if did < _DID_NO else _NOT_SURE}


async def for_triage(uid: str, view: dict, f: dict, words: str, now: float | None = None) -> tuple[dict | None, float | None]:
    """A hand-over's run as triage reads it: ``(proof, None)``, or ``(None, when to
    look again)`` while its calls are still on their way."""
    turn = f["turn"]
    actions, complete = recorded(turn)
    now = time.time() if now is None else now
    if not complete and now - float(f.get("ended_at") or f.get("started_at") or now) < _LATE_CALLS_S:
        return None, now + _LOOK_AGAIN_S
    flags = _json(view["task"].get("flags"))
    prior = flags.get("proof") if isinstance(flags.get("proof"), dict) else {}
    if prior.get("turn") == turn["turn_id"] and prior.get("calls") == len(actions) and not prior.get("more"):
        actions = prior.get("actions") or []  # read from the source already
    else:
        await confirm(uid, actions)
    return check(actions, reply=str(turn.get("reply_head") or ""), words=words, outcome=f["outcome"],
                 turn_id=turn["turn_id"]), None
