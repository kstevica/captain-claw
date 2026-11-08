"""Activity: who on the deck is active, and whether the admin wrote to them.

The Admin console's Activity page (decks with accounts only). Per user:

* **Typed** — the messages they sent their agents per UTC day, from any
  channel (web, kiosk, Telegram, WhatsApp…), as the daily allowance counted
  them (``daily_usage``, with what a reset zeroed added back: ``typed_before``).
* **Agents** — the days their agents spent tokens, scheduled work included
  (the spend ledger, ``usage_spend``), so someone whose agents run a weekly
  scan without them typing doesn't look gone.
* **Status** — ``active`` (typed in the last 7 days), ``quiet`` (typed before
  that, not since) or ``never`` (never typed a message).
* **You wrote** — every time an admin got in touch, from ``user_contacts``:
  the deck's emails (Ask for feedback, a message from this page) and the ones
  an admin noted they sent another way (WhatsApp, a call). And, for the admin
  looking, their own Gmail: the newest email they sent each person and the
  newest from them — only dates and subjects, read when asked, kept in memory
  for ``MAILBOX_TTL_S`` (10 minutes) and never saved.

    GET    /fd/admin/activity                    admin  every user's activity + contacts
    GET    /fd/admin/activity/mailbox            admin  the admin's Gmail, per user
    GET    /fd/admin/activity/{id}/message       admin  a blank email to them + how to send it
    POST   /fd/admin/activity/{id}/message       admin  {subject, body, send_via} → send it
    POST   /fd/admin/activity/{id}/contacts      admin  {channel, note} → "I wrote to them"
    DELETE /fd/admin/activity/contacts/{cid}     admin  take a note back (sent emails stay)
"""

from __future__ import annotations

import asyncio
import html
import json
import re
import time
from datetime import UTC, date, datetime, timedelta
from typing import Any

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from captain_claw import gmail_compose
from captain_claw.flight_deck import google_oauth_routes as _google
from captain_claw.flight_deck import usage_spend
from captain_claw.flight_deck.admin_routes import require_admin
from captain_claw.flight_deck.auth import _fd_auth_enabled, get_db
from captain_claw.flight_deck.auth_routes import _clean_text, _valid_email
from captain_claw.flight_deck.invite_routes import (
    _admin_name,
    _clean_note,
    mail_options,
    send_from_admin,
)
from captain_claw.flight_deck.usage_quota import today
from captain_claw.logging import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/fd/admin/activity", tags=["admin-activity"])

ACTIVE_DAYS = 7          # typed within this many days (today included) = active
DAYS_DEFAULT = 30
DAYS_MAX = 90
CONTACTS_SHOWN = 20      # per user, newest first
SUBJECT_MAX = 200
BODY_MAX = 5000
CHANNEL_MAX = 40
NOTE_MAX = 500
SEND_VIA = ("google", "microsoft")

# Reading the admin's Gmail: users.messages.list with a search needs one of
# these (gmail.metadata can't search; send/compose can't read).
_G = "https://www.googleapis.com/auth/"
GMAIL_READ_SCOPES = (_G + "gmail.readonly", _G + "gmail.modify", "https://mail.google.com/")
MAILBOX_TTL_S = 600.0    # a read is kept this long unless the admin asks again
MAILBOX_PARALLEL = 5
MAILBOX_TIMEOUT_S = 45.0
# Only plain addresses go into a Gmail search (no quotes, braces or spaces).
_SEARCHABLE = re.compile(r"^[A-Za-z0-9._%+'-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$")

_mailbox_cache: dict[str, tuple[float, dict]] = {}


def _http() -> httpx.AsyncClient:
    """The client the Gmail reads use (tests swap in a mock transport)."""
    return httpx.AsyncClient(timeout=20)


def _require_auth_deck() -> None:
    if not _fd_auth_enabled():
        raise HTTPException(403, "Activity needs accounts: this Flight Deck runs without them "
                                 "(FD_AUTH_ENABLED=false).")


# ── Who is active ─────────────────────────────────────────────────────


def _day_age(day: str, now: str) -> int:
    return (date.fromisoformat(now) - date.fromisoformat(day)).days


def status_for(last_typed: str, now: str) -> str:
    """``active`` (typed within ACTIVE_DAYS), ``quiet`` or ``never``."""
    if not last_typed:
        return "never"
    return "active" if _day_age(last_typed, now) < ACTIVE_DAYS else "quiet"


def _kiosk(meta: str) -> bool:
    try:
        return json.loads(meta or "{}").get("kiosk") is True
    except (TypeError, ValueError):
        return False


async def activity(db, viewer_id: str, days: int) -> dict:
    """Every user but *viewer_id*: what they and their agents did in the last
    *days* days, all-time bounds, their status and who wrote to them."""
    now = today()
    since = (date.fromisoformat(now) - timedelta(days=days - 1)).isoformat()
    week = (date.fromisoformat(now) - timedelta(days=ACTIVE_DAYS - 1)).isoformat()

    per_day: dict[str, dict[str, dict]] = {}
    for r in await db.list_message_days(since):
        per_day.setdefault(r["user_id"], {}).setdefault(r["day"], {"day": r["day"], "messages": 0, "tokens": 0})["messages"] = int(r["messages"])
    for r in await db.list_agent_work_days(since):
        per_day.setdefault(r["user_id"], {}).setdefault(r["day"], {"day": r["day"], "messages": 0, "tokens": 0})["tokens"] = r["tokens"]
    bounds = await db.message_day_bounds()
    agents_last = await db.agent_work_last_days()
    contacts: dict[str, list[dict]] = {}
    for c in await db.list_user_contacts():
        contacts.setdefault(c["user_id"], []).append({
            k: c[k] for k in ("id", "kind", "channel", "subject", "note", "created_at",
                              "admin_id", "admin_name")})

    users = []
    for u in await db.list_users(limit=100_000):
        uid = u["id"]
        if uid == viewer_id:
            continue
        days_list = sorted(per_day.get(uid, {}).values(), key=lambda d: d["day"])
        b = bounds.get(uid, {})
        mine = contacts.get(uid, [])
        users.append({
            "id": uid, "email": u["email"], "display_name": u.get("display_name") or "",
            "role": u.get("role") or "user", "kiosk": _kiosk(u.get("metadata") or ""),
            "created_at": u.get("created_at") or "", "last_login_at": u.get("last_login_at") or "",
            "status": status_for(b.get("last_day") or "", now),
            "typed": {
                "first_day": b.get("first_day") or "", "last_day": b.get("last_day") or "",
                "days_all": b.get("days", 0), "messages_all": b.get("messages", 0),
                "messages_week": sum(d["messages"] for d in days_list if d["day"] >= week),
                "messages": sum(d["messages"] for d in days_list),
                "days": sum(1 for d in days_list if d["messages"]),
            },
            "agents": {
                "last_day": agents_last.get(uid, ""),
                "tokens": sum(d["tokens"] for d in days_list),
                "days": sum(1 for d in days_list if d["tokens"]),
            },
            "days": days_list,
            "contacts": mine[:CONTACTS_SHOWN],
            "contacts_total": len(mine),
        })
    return {"today": now, "since": since, "days": days, "active_days": ACTIVE_DAYS, "users": users}


@router.get("")
async def get_activity(days: int = Query(DAYS_DEFAULT, ge=7, le=DAYS_MAX),
                       admin: dict = Depends(require_admin)) -> dict:
    """Every user's activity and contacts (admin only, decks with accounts)."""
    _require_auth_deck()
    await usage_spend._catch_up()   # the agents' latest calls, as the Users page does
    return await activity(get_db(), str(admin["id"]), days)


# ── The admin's own Gmail ─────────────────────────────────────────────


class _MailboxRefusedError(Exception):
    """Gmail turned the whole read down; the message says why, for the admin."""


async def _gmail_access(db, admin_id: str) -> tuple[str, str, str]:
    """``(token, address, reason)``: a token only when this admin's Google can
    search their mail, else the reason it can't."""
    if not await _google.is_google_connected(admin_id):
        return "", "", "Connect your Google account in Connections → Google to see your emails here."
    tokens = await _google._load_tokens(db, admin_id)
    address = str((await _google._load_user(db, admin_id) or {}).get("email") or "")
    if not set((tokens.scope if tokens else "").split()) & set(GMAIL_READ_SCOPES):
        return "", address, ("Your Google connection can't read mail: the deck's Google client needs "
                             "the gmail.readonly or gmail.modify scope (Connections → Google), then reconnect.")
    token = await _google.get_valid_google_access_token(admin_id)
    if not token:
        return "", address, "Google didn't give a working sign-in — reconnect it in Connections → Google."
    return token, address, ""


def _gmail_error(resp: httpx.Response) -> str:
    try:
        err = resp.json().get("error")
        return str(err.get("message") if isinstance(err, dict) else err or "")[:200]
    except Exception:
        return resp.text[:200]


async def _newest(client: httpx.AsyncClient, token: str, query: str) -> dict | None:
    """The newest message matching *query*: ``{at, subject}``, or None."""
    auth = {"Authorization": f"Bearer {token}"}
    resp = await client.get(f"{gmail_compose.GMAIL_API}/users/me/messages", headers=auth,
                            params={"q": query, "maxResults": 1})
    if resp.status_code == 401:
        raise _MailboxRefusedError("Google turned the deck's access down — reconnect Google in Connections → Google.")
    if resp.status_code == 403:
        raise _MailboxRefusedError(f"Gmail refused the search: {_gmail_error(resp)}")
    resp.raise_for_status()
    found = resp.json().get("messages") or []
    if not found:
        return None
    resp = await client.get(f"{gmail_compose.GMAIL_API}/users/me/messages/{found[0]['id']}",
                            headers=auth, params={"format": "metadata", "metadataHeaders": "Subject"})
    resp.raise_for_status()
    msg = resp.json()
    subject = next((h.get("value", "") for h in (msg.get("payload") or {}).get("headers") or []
                    if str(h.get("name", "")).lower() == "subject"), "")
    try:
        at = datetime.fromtimestamp(int(msg.get("internalDate") or 0) / 1000, UTC).isoformat()
    except (TypeError, ValueError):
        at = ""
    return {"at": at, "subject": _clean_text(subject, SUBJECT_MAX)}


async def read_mailbox(db, admin: dict) -> dict:
    """The newest email the admin sent each user and the newest from them, in
    the admin's own Gmail. Never raises: what couldn't be read says why."""
    admin_id = str(admin["id"])
    out: dict[str, Any] = {"provider": "google", "address": "", "available": False, "reason": "",
                           "partial": False, "checked_at": datetime.now(UTC).isoformat(), "users": {}}
    token, address, reason = await _gmail_access(db, admin_id)
    out["address"] = address
    if not token:
        out["reason"] = reason
        return out
    out["available"] = True
    people = [u for u in await db.list_users(limit=100_000)
              if u["id"] != admin_id and _SEARCHABLE.match(u["email"] or "")
              and u["email"].lower() != address.lower()]
    gate = asyncio.Semaphore(MAILBOX_PARALLEL)
    refused: list[str] = []

    async with _http() as client:
        async def one(user: dict) -> None:
            email = user["email"]
            async with gate:
                if refused:
                    return
                try:
                    sent = await _newest(client, token, f"in:sent {{to:{email} cc:{email} bcc:{email}}}")
                    received = await _newest(client, token, f"from:{email}")
                except _MailboxRefusedError as exc:
                    refused.append(str(exc))
                    return
                except (httpx.HTTPError, ValueError) as exc:
                    log.debug("Activity: mailbox not read for a user", error=type(exc).__name__)
                    out["partial"] = True
                    return
            out["users"][user["id"]] = {"sent": sent, "received": received}

        try:
            await asyncio.wait_for(asyncio.gather(*(one(u) for u in people)), MAILBOX_TIMEOUT_S)
        except TimeoutError:
            out["partial"] = True
    if refused:
        out.update(available=False, reason=refused[0], users={})
    elif len(out["users"]) < len(people):
        out["partial"] = True
    return out


@router.get("/mailbox")
async def get_mailbox(refresh: bool = False, admin: dict = Depends(require_admin)) -> dict:
    """This admin's Gmail, per user (kept MAILBOX_TTL_S unless *refresh*)."""
    _require_auth_deck()
    key = str(admin["id"])
    held = _mailbox_cache.get(key)
    if held and not refresh and time.monotonic() - held[0] < MAILBOX_TTL_S:
        return held[1]
    result = await read_mailbox(get_db(), admin)
    if result["available"] and not result["partial"]:
        _mailbox_cache[key] = (time.monotonic(), result)
    else:
        _mailbox_cache.pop(key, None)
    return result


# ── Writing to them ───────────────────────────────────────────────────


class SendMessage(BaseModel):
    subject: str
    body: str
    send_via: str


class NoteContact(BaseModel):
    channel: str
    note: str = ""


async def _user(db, user_id: str) -> dict:
    user = await db.get_user_by_id(user_id)
    if not user:
        raise HTTPException(404, "User not found")
    return user


def compose_message(*, admin_name: str, user_name: str) -> str:
    """The blank email the admin starts from, to the person by first name."""
    first = user_name.split()[0] if user_name.split() else ""
    return "\n".join([f"Hi {first}," if first else "Hi,", "", "", "", admin_name])


def _html(text: str) -> str:
    return (
        '<div style="font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Helvetica,Arial,'
        'sans-serif;font-size:15px;line-height:1.5;color:#1a1a1a;max-width:520px;white-space:pre-wrap">'
        f"{html.escape(text)}</div>"
    )


@router.get("/{user_id}/message")
async def draft_message(user_id: str, admin: dict = Depends(require_admin)) -> dict:
    """A blank email to the user and how this admin can send it."""
    _require_auth_deck()
    db = get_db()
    user = await _user(db, user_id)
    return {"to": user["email"], "subject": "",
            "body": compose_message(admin_name=_admin_name(admin), user_name=user.get("display_name") or ""),
            "mail": await mail_options(db, admin["id"])}


@router.post("/{user_id}/message")
async def send_message(user_id: str, body: SendMessage, admin: dict = Depends(require_admin)) -> dict:
    """Email the user from the admin's own account. A failed email comes back
    as ``ok: false`` with the reason and isn't recorded."""
    _require_auth_deck()
    if body.send_via not in SEND_VIA:
        raise HTTPException(400, "send_via must be one of: google, microsoft")
    subject = _clean_text(body.subject, SUBJECT_MAX)
    text = _clean_note(body.body, BODY_MAX)
    if not subject or not text:
        raise HTTPException(400, "Write a subject and a message")
    db = get_db()
    user = await _user(db, user_id)
    if not _valid_email(user["email"]):
        raise HTTPException(400, "This account has no email address to write to")
    mail = await send_from_admin(db, admin, body.send_via, user["email"], subject, text, _html(text))
    if not mail["sent"]:
        log.warning("Message to a user failed", user=user_id, via=body.send_via, error=mail["error"][:200])
        return {"ok": False, "email": mail}
    _mailbox_cache.pop(str(admin["id"]), None)
    log.info("Message emailed to a user", user=user_id, via=body.send_via)
    # It went out: a record that can't be kept (the user removed meanwhile)
    # doesn't turn that into a failure.
    try:
        contact = await db.add_user_contact(user_id, str(admin["id"]), "message", body.send_via,
                                            subject=subject)
    except Exception as exc:
        log.warning("Message not recorded for Activity", user=user_id, error=str(exc))
        return {"ok": True, "email": mail, "contact": None}
    return {"ok": True, "email": mail, "contact": {**contact, "admin_name": _admin_name(admin)}}


@router.post("/{user_id}/contacts")
async def note_contact(user_id: str, body: NoteContact, admin: dict = Depends(require_admin)) -> dict:
    """The admin wrote to the user some other way (WhatsApp, a call…)."""
    _require_auth_deck()
    channel = _clean_text(body.channel, CHANNEL_MAX)
    if not channel:
        raise HTTPException(400, "Say how you got in touch")
    db = get_db()
    await _user(db, user_id)
    contact = await db.add_user_contact(user_id, str(admin["id"]), "noted", channel,
                                        note=_clean_note(body.note, NOTE_MAX))
    return {"ok": True, "contact": {**contact, "admin_name": _admin_name(admin)}}


@router.delete("/contacts/{contact_id}")
async def delete_contact(contact_id: str, admin: dict = Depends(require_admin)) -> dict:
    """Take back a note. Emails the deck sent stay on record: they went out."""
    _require_auth_deck()
    db = get_db()
    contact = await db.get_user_contact(contact_id)
    if not contact:
        raise HTTPException(404, "Not found")
    if contact["kind"] != "noted":
        raise HTTPException(400, "Emails the deck sent stay on record")
    await db.delete_user_contact(contact_id)
    return {"ok": True}
