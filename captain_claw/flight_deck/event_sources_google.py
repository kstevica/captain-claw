"""Google Calendar + Gmail source adapters for the event spine (#2).

FD-side: fetch directly from the Google APIs with FD's OAuth token
(``get_valid_google_access_token``). Calendar surfaces new/changed events AND
soon-starting ones; Gmail surfaces important+unread inbox messages. Per-user
enable via the autonomy config (``event_calendar_enabled`` / ``event_gmail_enabled``).
Each poll fetches the token itself, so it no-ops cleanly when Google isn't connected.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx

from captain_claw.config import get_config
from captain_claw.flight_deck.event_sources import Adapter, register

_log = logging.getLogger(__name__)
_CAL = "https://www.googleapis.com/calendar/v3"
_GMAIL = "https://gmail.googleapis.com/gmail/v1"


async def _token() -> str | None:
    from captain_claw.flight_deck.google_oauth_routes import get_valid_google_access_token
    return await get_valid_google_access_token()


def _evt_start(ev: dict[str, Any]) -> str:
    s = ev.get("start") or {}
    return s.get("dateTime") or s.get("date") or ""


async def poll_calendar(user_id: str, cursor: str) -> tuple[list[dict[str, Any]], str]:
    """Surface new/changed events (incremental via updatedMin) AND events starting
    in the next 24h. Dedup: changed by id+updated, upcoming once per id+start."""
    token = await _token()
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


async def poll_gmail(user_id: str, cursor: str) -> tuple[list[dict[str, Any]], str]:
    """Surface important + unread inbox messages. Dedup by message id (unread ones
    persist, so re-listing them is a no-op once ingested)."""
    token = await _token()
    if not token:
        return [], cursor
    headers = {"Authorization": f"Bearer {token}"}
    out: list[dict[str, Any]] = []
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
            frm, subj = "?", "(no subject)"
            try:
                rm = await client.get(
                    f"{_GMAIL}/users/me/messages/{mid}",
                    params={"format": "metadata", "metadataHeaders": ["From", "Subject"]},
                    headers=headers,
                )
                if rm.status_code == 200:
                    hdrs = {h.get("name"): h.get("value") for h in rm.json().get("payload", {}).get("headers", [])}
                    frm = hdrs.get("From", frm)
                    subj = hdrs.get("Subject", subj)
            except Exception:
                pass
            out.append({
                "source": "gmail", "event_type": "new_email",
                "summary": f"Email from {frm}: {subj}",
                "dedup_key": f"gmail:{mid}",
                "metadata": {"message_id": mid, "from": frm, "subject": subj},
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
))
register(Adapter(
    name="gmail",
    interval_seconds=float(get_config().events.gmail_interval_seconds),
    poll=poll_gmail,
    enabled=lambda uid: _cfg_flag(uid, "event_gmail_enabled"),
))
