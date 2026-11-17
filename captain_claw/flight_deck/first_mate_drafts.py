"""First Mate's draft watcher: what the owner did with a draft a factory left.

See docs/first-mate-plan.md §8 (Phase 4). Many factories leave drafts for the
owner to send (inbox drafts, follow-ups). What happens to a draft is the
plainest signal there is:

* **sent as written**: the sent message's text is the draft's (similarity ≥ 0.9);
* **sent after edits**: it was sent, changed; the signal's value is how much
  of the draft stayed (0..1);
* **deleted**: the draft is gone and nothing was sent in its thread since;
* **silence**: a draft still sitting there after two weeks is ``expired``, and
  counts for nothing. Silence is never a signal.

:func:`capture` picks up the drafts a finished run made, from the tool-call
register (the ids an ok ``create_draft`` named, and the text it was given; a
text over the register's 2,000 characters keeps only its start, so only that
start is compared). :func:`check_due` looks at waiting drafts from the
supervisor's scan: each one 15 minutes after it was made at the soonest, then
at most once an hour, through the owner's own Google token. A draft that
can't be checked now (no token, an outage) waits for the next look.
"""

from __future__ import annotations

import base64
import difflib
import json
import logging
import re
from datetime import datetime, timedelta
from typing import Any

from captain_claw.flight_deck import first_mate_store as fms

log = logging.getLogger(__name__)

AS_WRITTEN = 0.9
OLDER_THAN = timedelta(minutes=15)
EVERY = timedelta(hours=1)
EXPIRE = timedelta(days=14)
PER_SCAN = 20
_DRAFTS = "https://gmail.googleapis.com/gmail/v1/users/me/drafts"
_THREADS = "https://gmail.googleapis.com/gmail/v1/users/me/threads"
_QUOTE = re.compile(r"^(>|On .{1,200} wrote:\s*$|-{2,} ?Original Message ?-{2,})", re.MULTILINE)


def _params(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    try:
        data = json.loads(raw or "{}")
    except (TypeError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


# ── text (pure) ────────────────────────────────────────────────────────

def own_text(text: str) -> str:
    """The part of a message its writer wrote: up to the quoted reply, with the
    whitespace flattened and case folded."""
    m = _QUOTE.search(text or "")
    head = (text or "")[:m.start()] if m else (text or "")
    return " ".join(head.split()).casefold()


def similarity(draft: str, sent: str, *, partial: bool = False) -> float:
    a, b = own_text(draft), own_text(sent)
    if partial:
        b = b[:len(a)]
    if not a and not b:
        return 1.0
    return round(difflib.SequenceMatcher(None, a, b).ratio(), 3)


def message_text(message: dict[str, Any]) -> str:
    """A Gmail API message's plain text (the first text/plain part, else its snippet)."""
    def walk(part: dict[str, Any]) -> str:
        if str(part.get("mimeType") or "").startswith("text/plain"):
            data = (part.get("body") or {}).get("data") or ""
            try:
                return base64.urlsafe_b64decode(data + "=" * (-len(data) % 4)).decode("utf-8", "replace")
            except (ValueError, TypeError):
                return ""
        for sub in part.get("parts") or []:
            found = walk(sub)
            if found:
                return found
        return ""
    return walk(message.get("payload") or {}) or str(message.get("snippet") or "")


def verdict(draft: dict[str, Any], thread: dict[str, Any]) -> tuple[str, float | None]:
    """What became of a draft that is no longer a draft, from its thread:
    ``("sent", s)``, ``("edited", s)`` or ``("deleted", None)``."""
    made = datetime.fromisoformat(draft["created_at"]).timestamp()
    sent = [m for m in thread.get("messages") or []
            if "SENT" in (m.get("labelIds") or []) and "DRAFT" not in (m.get("labelIds") or [])
            and int(m.get("internalDate") or 0) / 1000 >= made - 60]
    if not sent:
        return "deleted", None
    best = max(similarity(draft["body"], message_text(m), partial=bool(draft["partial"])) for m in sent)
    return ("sent" if best >= AS_WRITTEN else "edited"), best


# ── capture ────────────────────────────────────────────────────────────

def drafts_in(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The drafts among a run's register rows: an ok create/update with a draft id."""
    out = []
    for r in rows:
        if str(r.get("tool") or "") != "google_mail" or str(r.get("outcome") or "") != "ok":
            continue
        params = _params(r.get("params"))
        if params.get("action") not in ("create_draft", "update_draft"):
            continue
        refs = _params(r.get("refs"))
        if not refs.get("gmail_draft_id"):
            continue
        body = params.get("body") or params.get("text") or ""
        partial = False
        if isinstance(body, dict):  # the register keeps only the start of a long text
            body, partial = str(body.get("_head") or ""), True
        out.append({"draft_id": str(refs["gmail_draft_id"]), "message_id": str(refs.get("gmail_message_id") or ""),
                    "thread_id": str(refs.get("gmail_thread_id") or ""), "subject": str(params.get("subject") or ""),
                    "body": str(body), "partial": partial})
    return out


async def capture(store: fms.FirstMateStore, run: dict[str, Any], rows: list[dict[str, Any]] | None = None) -> int:
    """Watch the drafts a finished run made. Never raises; returns how many are new."""
    try:
        if rows is None:
            from captain_claw.flight_deck import first_mate_proof
            rows = await first_mate_proof._rows(run) or []
        added = 0
        for d in drafts_in(rows):
            added += store.add_draft(run["factory_id"], run["owner_id"], run_id=run["id"], **d)
        return added
    except Exception:  # noqa: BLE001
        log.warning("first mate: couldn't pick up a run's drafts", extra={"run": run.get("id")}, exc_info=True)
        return 0


# ── checking ───────────────────────────────────────────────────────────

_SIGNAL = {"sent": "sent_as_written", "edited": "sent_edited", "deleted": "draft_deleted"}


def _later(store: fms.FirstMateStore, d: dict[str, Any], now: datetime) -> None:
    """Couldn't tell now: look again later, unless it's past its two weeks."""
    old = now - datetime.fromisoformat(d["created_at"]) > EXPIRE
    store.set_draft(d["id"], "expired" if old else "waiting", now=now)


async def check_due(store: fms.FirstMateStore, now: datetime) -> int:
    """Look at the drafts due a look. Returns how many got an answer."""
    from captain_claw.flight_deck import taskboard_proof
    answered = 0
    tokens: dict[str, str | None] = {}
    for d in store.drafts_to_check(now, older_than=OLDER_THAN, every=EVERY, limit=PER_SCAN):
        try:
            owner = d["owner_id"]
            if owner not in tokens:
                try:
                    tokens[owner] = await taskboard_proof._token(owner)
                except Exception:  # noqa: BLE001
                    tokens[owner] = None
            token = tokens[owner]
            if not token:
                _later(store, d, now)  # no token now
                continue
            status, _ = await taskboard_proof._fetch(token, f"{_DRAFTS}/{d['draft_id']}", [("format", "minimal")])
            if status == 200:
                _later(store, d, now)  # still a draft
                continue
            if status != 404:
                _later(store, d, now)  # an outage or a scope
                continue
            if not d["thread_id"]:
                store.set_draft(d["id"], "expired", now=now)  # gone, and nowhere to look
                continue
            st, thread = await taskboard_proof._fetch(token, f"{_THREADS}/{d['thread_id']}", [("format", "full")])
            if st != 200:
                _later(store, d, now)
                continue
            outcome, sim = verdict(d, thread)
            store.set_draft(d["id"], outcome, similarity=sim, now=now)
            store.add_signal(d["factory_id"], _SIGNAL[outcome], value=sim if outcome == "edited" else None,
                             run_id=d["run_id"], source="gmail", ref=d["draft_id"])
            answered += 1
        except Exception:  # noqa: BLE001 — one draft never stops the others
            log.warning("first mate: checking a draft failed", extra={"draft": d.get("id")}, exc_info=True)
    return answered
