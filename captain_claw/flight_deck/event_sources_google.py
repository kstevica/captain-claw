"""Google Calendar + Gmail source adapters for the event spine (#2).

FD-side: fetch directly from the Google APIs with the polled user's OWN OAuth
token (``get_valid_google_access_token(user_id)``) — never a deployment-wide
one, which put the primary owner's inbox in every user's feed. Calendar
surfaces new/changed events AND soon-starting ones; Gmail surfaces
important+unread inbox messages. Per-user enable via the autonomy config
(``event_calendar_enabled`` / ``event_gmail_enabled``), and gated on that user
having connected Google (``requires_google``); each poll also fetches the token
itself, so it no-ops cleanly when the user isn't connected.
"""

from __future__ import annotations

import html
import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx

from captain_claw.config import get_config
from captain_claw.flight_deck.event_sources import Adapter, register

_log = logging.getLogger(__name__)
_CAL = "https://www.googleapis.com/calendar/v3"
_GMAIL = "https://gmail.googleapis.com/gmail/v1"


async def _token(user_id: str) -> str | None:
    from captain_claw.flight_deck.google_oauth_routes import get_valid_google_access_token
    return await get_valid_google_access_token(user_id)


def _evt_start(ev: dict[str, Any]) -> str:
    s = ev.get("start") or {}
    return s.get("dateTime") or s.get("date") or ""


async def poll_calendar(user_id: str, cursor: str) -> tuple[list[dict[str, Any]], str]:
    """Surface new/changed events (incremental via updatedMin) AND events starting
    in the next 24h. Dedup: changed by id+updated, upcoming once per id+start."""
    token = await _token(user_id)
    if not token:
        return [], cursor
    headers = {"Authorization": f"Bearer {token}"}
    now = datetime.now(timezone.utc)
    out: list[dict[str, Any]] = []
    async with httpx.AsyncClient(timeout=20.0) as client:
        # (a) New/changed in the next 7 days — only once a cursor exists, so the
        # first poll doesn't dump every existing event as "changed". The first run
        # just establishes the cursor (+ surfaces upcoming below).
        if cursor:
            try:
                changed_params: dict[str, Any] = {
                    "timeMin": now.isoformat(), "timeMax": (now + timedelta(days=7)).isoformat(),
                    "singleEvents": "true", "orderBy": "updated", "maxResults": 25,
                    "updatedMin": cursor,
                }
                r = await client.get(f"{_CAL}/calendars/primary/events", params=changed_params, headers=headers)
                if r.status_code == 200:
                    for ev in r.json().get("items", []):
                        if ev.get("status") == "cancelled":
                            continue
                        out.append({
                            "source": "calendar", "event_type": "event_changed",
                            "summary": f"Calendar: '{ev.get('summary') or '(no title)'}' at {_evt_start(ev)}",
                            "dedup_key": f"cal:{ev.get('id')}:{ev.get('updated', '')}",
                            "metadata": {"event_id": ev.get("id"), "start": _evt_start(ev)},
                        })
            except Exception as exc:
                _log.warning("calendar changed poll failed: %s", exc)
        # (b) Upcoming in the next 24h.
        up_params = {
            "timeMin": now.isoformat(), "timeMax": (now + timedelta(hours=24)).isoformat(),
            "singleEvents": "true", "orderBy": "startTime", "maxResults": 10,
        }
        try:
            r = await client.get(f"{_CAL}/calendars/primary/events", params=up_params, headers=headers)
            if r.status_code == 200:
                for ev in r.json().get("items", []):
                    if ev.get("status") == "cancelled":
                        continue
                    out.append({
                        "source": "calendar", "event_type": "upcoming",
                        "summary": f"Upcoming: '{ev.get('summary') or '(no title)'}' at {_evt_start(ev)}",
                        "dedup_key": f"cal_up:{ev.get('id')}:{_evt_start(ev)}",
                        "metadata": {"event_id": ev.get("id"), "start": _evt_start(ev)},
                    })
        except Exception as exc:
            _log.warning("calendar upcoming poll failed: %s", exc)
    return out, now.isoformat()


# Senders that are machines, not people — their mail is notification noise, never
# something the user needs to act on. Matched as a substring of the From header.
_AUTOMATED_SENDER_BITS: tuple[str, ...] = (
    "noreply", "no-reply", "no_reply", "donotreply", "do-not-reply",
    "notifications@", "notification@", "comments-noreply", "mailer-daemon",
    "postmaster@", "automated@", "auto-confirm", "bounce", "@docs.google.com",
)


def _is_automated_sender(frm: str) -> bool:
    f = (frm or "").lower()
    return any(bit in f for bit in _AUTOMATED_SENDER_BITS)


def _gmail_max_age_hours(user_id: str) -> int:
    """The per-user Gmail event age cutoff in hours (0 = off). Best-effort."""
    try:
        from captain_claw.flight_deck.autonomy import resolve_config
        return int(resolve_config(user_id).get("gmail_event_max_age_hours", 48))
    except Exception:
        return 48


def _internal_date_iso(raw: Any) -> str:
    """Gmail ``internalDate`` (ms since the epoch, as a string) → ISO UTC, or ""."""
    try:
        ms = int(str(raw or "").strip())
    except (TypeError, ValueError):
        return ""
    if ms <= 0:
        return ""
    try:
        return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc).isoformat()
    except (OverflowError, OSError, ValueError):
        return ""


def _header_addr(value: str) -> str:
    """The bare address of a ``"Name" <addr>`` header value (first one), or ""."""
    v = str(value or "").strip()
    if not v:
        return ""
    m = re.search(r"<([^>]+)>", v)
    addr = (m.group(1) if m else v.split(",")[0]).strip().strip('"').strip()
    return addr if "@" in addr else ""


async def poll_gmail(user_id: str, cursor: str) -> tuple[list[dict[str, Any]], str]:
    """Surface important + unread inbox messages. Dedup by message id (unread ones
    persist, so re-listing them is a no-op once ingested).

    A message that arrived more than ``gmail_event_max_age_hours`` ago (Gmail
    ``internalDate``; 0 = no cutoff) is skipped — an old email never becomes an
    event. One poll yields at most ONE event per thread: Gmail lists newest
    first, so the newest message of each thread stands for it (per-thread dedup
    across polls lives in the arbiter). Each event records when the email
    arrived (``received_at``), its Reply-To address and Gmail's short snippet."""
    token = await _token(user_id)
    if not token:
        return [], cursor
    headers = {"Authorization": f"Bearer {token}"}
    max_age: int | None = None   # resolved on first need (a dated message)
    now = datetime.now(timezone.utc)
    out: list[dict[str, Any]] = []
    seen_threads: set[str] = set()
    async with httpx.AsyncClient(timeout=20.0) as client:
        try:
            r = await client.get(
                f"{_GMAIL}/users/me/messages",
                params={"q": "is:important is:unread in:inbox", "maxResults": 10},
                headers=headers,
            )
            if r.status_code != 200:
                return [], cursor
            msgs = r.json().get("messages", []) or []
        except Exception as exc:
            _log.warning("gmail list failed: %s", exc)
            return [], cursor
        for m in msgs[:10]:
            mid = m.get("id")
            if not mid:
                continue
            tid = m.get("threadId") or mid  # for get_thread; falls back to message id
            if tid in seen_threads:
                continue  # an older message of a thread whose newest one came first
            seen_threads.add(tid)
            frm, subj, reply_to, snippet, received_at = "?", "(no subject)", "", "", ""
            try:
                rm = await client.get(
                    f"{_GMAIL}/users/me/messages/{mid}",
                    params={"format": "metadata",
                            "metadataHeaders": ["From", "Subject", "Reply-To"]},
                    headers=headers,
                )
                if rm.status_code == 200:
                    body = rm.json()
                    hdrs = {h.get("name"): h.get("value") for h in body.get("payload", {}).get("headers", [])}
                    frm = hdrs.get("From", frm)
                    subj = hdrs.get("Subject", subj)
                    reply_to = _header_addr(hdrs.get("Reply-To") or "")
                    snippet = html.unescape(body.get("snippet") or "")[:200]
                    received_at = _internal_date_iso(body.get("internalDate"))
            except Exception:
                pass
            if received_at and max_age is None:
                max_age = _gmail_max_age_hours(user_id)
            if received_at and max_age and max_age > 0:
                try:
                    age_h = (now - datetime.fromisoformat(received_at)).total_seconds() / 3600.0
                except ValueError:
                    age_h = 0.0
                if age_h > max_age:
                    continue  # too old to surface (J3)
            if _is_automated_sender(frm):
                continue  # no-reply / notification mail — not a real person needing the user
            out.append({
                "source": "gmail", "event_type": "new_email",
                "summary": f"Email from {frm}: {subj}",
                "dedup_key": f"gmail:{mid}",
                "metadata": {"message_id": mid, "thread_id": tid, "from": frm, "subject": subj,
                             "received_at": received_at, "reply_to": reply_to,
                             "snippet": snippet},
            })
    return out, cursor


def _cfg_flag(user_id: str, key: str) -> bool:
    try:
        from captain_claw.flight_deck.autonomy import resolve_config
        return bool(resolve_config(user_id).get(key))
    except Exception:
        return False


register(Adapter(
    name="calendar",
    interval_seconds=float(get_config().events.calendar_interval_seconds),
    poll=poll_calendar,
    enabled=lambda uid: _cfg_flag(uid, "event_calendar_enabled"),
    requires_google=True,
))
register(Adapter(
    name="gmail",
    interval_seconds=float(get_config().events.gmail_interval_seconds),
    poll=poll_gmail,
    enabled=lambda uid: _cfg_flag(uid, "event_gmail_enabled"),
    requires_google=True,
))
