"""Gmail sending through Flight Deck — the one gate, audit trail and notifier.

Under Flight Deck an agent's ``google_mail`` tool never sends Gmail itself: it
asks ``POST /fd/google/gmail/send`` and Flight Deck decides, by the agent
OWNER's per-user policy (user_settings ``google_oauth:gmail_send`` — the
``google_oauth:`` prefix is server-owned, so ``/fd/settings`` can't write it,
and a Google disconnect leaves it in place):

* ``enabled`` — OFF by default: nothing changes until the user opts in
  (Connections → Google → Email sending). On means automatic — there is no
  per-send confirmation;
* ``allowed_recipients`` — empty = anyone; else exact addresses and
  ``@domain`` entries (see :mod:`captain_claw.gmail_compose`);
* ``daily_limit`` — emails per rolling 24 hours (default 50, 1..500).

The deck kill switch ``FD_GMAIL_SEND=off`` (also ``0`` / ``false`` / ``no``)
refuses every send regardless. A send that passes goes out with the owner's
token, lands in ``gmail_sends`` (the user's history, the daily count and the
duplicate check) and rings the owner's bell. So does a send whose outcome is
unknown (Gmail answered 5xx, or never answered after the request went out) —
as ``status = 'unknown'``, so a blind retry is caught as a duplicate and the
user is told to check their Sent folder.

The gws Workspace CLI tool is retired (``config.RETIRED_TOOLS``: never
registered, stripped from every tools list, refused by the shell tool), so
google_mail → this route is the only tool path to a Gmail send under Flight
Deck.

The user routes — ``GET``/``PUT /fd/google/gmail-send`` and ``GET
/fd/google/gmail-sends`` — are the signed-in user's own policy and history; no
admin involved. Like all of /fd/google, only on an auth-enabled deck
(``_require_auth_deck``).
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from captain_claw import gmail_compose
from captain_claw.flight_deck import google_oauth_routes as _google
from captain_claw.flight_deck.auth import get_current_user, get_db
from captain_claw.flight_deck.db import FlightDeckDB
from captain_claw.flight_deck.google_oauth_routes import _require_auth_deck
from captain_claw.logging import get_logger

log = get_logger(__name__)


router = APIRouter(
    prefix="/fd/google",
    tags=["google-gmail-send"],
    dependencies=[Depends(_require_auth_deck)],
)


_K_GMAIL_SEND = "google_oauth:gmail_send"  # per-user, server-owned

DEFAULT_DAILY_LIMIT = 50
MAX_DAILY_LIMIT = 500
MAX_ALLOWED_RECIPIENTS = 200

_DAY = timedelta(hours=24)
_DUPLICATE_WINDOW = timedelta(minutes=10)

_ENABLE_WHERE = "Flight Deck → Connections → Google → Email sending"
_RECONNECT = "Flight Deck → Connections → Google"

# The message fields of a composed send (the other shape is {"draft_id"}).
_MESSAGE_FIELDS = ("to", "cc", "bcc", "subject", "body", "html_body", "reply_to_message_id")

# Gmail message / draft ids go into the URL path — never let one walk it.
_GMAIL_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,128}$")

_OFF_DETAIL = (
    "Email sending is turned off for this user, so nothing was sent. Create a "
    "draft with google_mail action=create_draft instead, and tell the user they "
    f"can turn sending on in {_ENABLE_WHERE}."
)
_DECK_OFF_DETAIL = (
    "Email sending is disabled on this Flight Deck by its administrator "
    "(FD_GMAIL_SEND), so nothing was sent. Create a draft with google_mail "
    "action=create_draft instead."
)

# Per user: policy read-modify-write, and the count → duplicate check → send →
# audit sequence (so two concurrent sends can't both slip under the limit or
# both pass the duplicate check).
_policy_locks: dict[str, asyncio.Lock] = {}
_send_locks: dict[str, asyncio.Lock] = {}


def _lock(locks: dict[str, asyncio.Lock], key: str) -> asyncio.Lock:
    lock = locks.get(key)
    if lock is None:
        lock = locks[key] = asyncio.Lock()
    return lock


# ── policy ──────────────────────────────────────────────────────────


def deck_send_disabled() -> bool:
    """The deck kill switch: ``FD_GMAIL_SEND`` = off / 0 / false / no."""
    return os.environ.get("FD_GMAIL_SEND", "").strip().lower() in ("off", "0", "false", "no")


def _default_policy() -> dict[str, Any]:
    return {"enabled": False, "allowed_recipients": [], "daily_limit": DEFAULT_DAILY_LIMIT}


def _valid_daily_limit(value: Any) -> bool:
    return (isinstance(value, int) and not isinstance(value, bool)
            and 1 <= value <= MAX_DAILY_LIMIT)


async def load_gmail_send_policy(db: FlightDeckDB, user_id: str) -> dict[str, Any]:
    """*user_id*'s sending policy — the defaults (sending OFF) when none is
    stored or the record is unreadable: a corrupt record never turns sending on
    or widens the allowlist."""
    raw = await db.get_setting(user_id, _K_GMAIL_SEND)
    if not raw:
        return _default_policy()
    try:
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("not an object")
        enabled = data.get("enabled", False)
        daily_limit = data.get("daily_limit", DEFAULT_DAILY_LIMIT)
        allowed = gmail_compose.normalize_allowlist(data.get("allowed_recipients") or [])
        if (not isinstance(enabled, bool) or not _valid_daily_limit(daily_limit)
                or len(allowed) > MAX_ALLOWED_RECIPIENTS):
            raise ValueError("field out of range")
    except Exception as exc:
        log.warning("Unreadable Gmail send policy for a user; using the defaults (off): %s",
                    type(exc).__name__)
        return _default_policy()
    return {"enabled": enabled, "allowed_recipients": allowed, "daily_limit": daily_limit}


async def _sent_last_24h(db: FlightDeckDB, user_id: str) -> int:
    since = (datetime.now(timezone.utc) - _DAY).isoformat()
    return await db.count_gmail_sends_since(user_id, since)


async def _policy_view(db: FlightDeckDB, user_id: str, policy: dict[str, Any]) -> dict[str, Any]:
    return {
        "enabled": bool(policy["enabled"]),
        "allowed_recipients": list(policy["allowed_recipients"]),
        "daily_limit": int(policy["daily_limit"]),
        "deck_disabled": deck_send_disabled(),
        "sent_last_24h": await _sent_last_24h(db, user_id),
    }


class GmailSendPolicyUpdate(BaseModel):
    # Every field optional; absent / null leaves it unchanged. Typed loosely and
    # checked in the handler so any bad value is a 400 with a readable detail.
    enabled: Any = None
    allowed_recipients: Any = None
    daily_limit: Any = None


# ── user routes: the signed-in user's own policy + history ──────────


@router.get("/gmail-send")
async def gmail_send_policy_get(
    _user: dict = Depends(get_current_user),
) -> dict[str, Any]:
    """This user's Email sending policy (+ the deck switch and today's count)."""
    db = get_db()
    uid = _google._effective_owner(_user)
    return await _policy_view(db, uid, await load_gmail_send_policy(db, uid))


@router.put("/gmail-send")
async def gmail_send_policy_put(
    body: GmailSendPolicyUpdate,
    _user: dict = Depends(get_current_user),
) -> dict[str, Any]:
    """Change this user's Email sending policy (any subset of the fields)."""
    db = get_db()
    uid = _google._effective_owner(_user)
    updates: dict[str, Any] = {}
    if body.enabled is not None:
        if not isinstance(body.enabled, bool):
            raise HTTPException(status_code=400, detail="enabled must be true or false")
        updates["enabled"] = body.enabled
    if body.daily_limit is not None:
        if not _valid_daily_limit(body.daily_limit):
            raise HTTPException(
                status_code=400,
                detail=f"daily_limit must be a whole number from 1 to {MAX_DAILY_LIMIT}",
            )
        updates["daily_limit"] = body.daily_limit
    if body.allowed_recipients is not None:
        try:
            allowed = gmail_compose.normalize_allowlist(body.allowed_recipients)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if len(allowed) > MAX_ALLOWED_RECIPIENTS:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"allowed_recipients can hold at most {MAX_ALLOWED_RECIPIENTS} "
                    f"entries (got {len(allowed)})"
                ),
            )
        updates["allowed_recipients"] = allowed

    async with _lock(_policy_locks, uid):
        policy = await load_gmail_send_policy(db, uid)
        policy.update(updates)
        await db.set_settings(uid, {_K_GMAIL_SEND: json.dumps(policy, ensure_ascii=True)})
    return await _policy_view(db, uid, policy)


@router.get("/gmail-sends")
async def gmail_sends_list(
    limit: int = 20,
    _user: dict = Depends(get_current_user),
) -> dict[str, Any]:
    """This user's emails sent by their agents, newest first."""
    db = get_db()
    uid = _google._effective_owner(_user)
    rows = await db.list_gmail_sends(uid, max(1, min(int(limit), 100)))
    return {"sends": [
        {
            "id": r["id"],
            # "sent", or "unknown": Gmail never confirmed it — it may have gone out.
            "status": r.get("status") or "sent",
            "agent": r["agent"],
            "to": r["to_addrs"],
            "cc": r["cc_addrs"],
            "bcc": r["bcc_addrs"],
            "subject": r["subject"],
            "gmail_message_id": r["gmail_message_id"],
            "thread_id": r["thread_id"],
            "draft_id": r["draft_id"],
            "created_at": r["created_at"],
        }
        for r in rows
    ]}


# ── agent route ─────────────────────────────────────────────────────


def _refused(detail: str, reason: str) -> HTTPException:
    """403 for "sending is off" — the header tells the agent's tool to point
    the model at create_draft (other 403s don't carry it)."""
    return HTTPException(
        status_code=403, detail=detail,
        headers={gmail_compose.SEND_REFUSED_HEADER: reason},
    )


async def _read_payload(request: Request) -> dict[str, str]:
    try:
        data = await request.json()
    except Exception:
        data = None
    if not isinstance(data, dict):
        raise HTTPException(status_code=400, detail="The body must be a JSON object")
    out: dict[str, str] = {}
    for key in _MESSAGE_FIELDS + ("draft_id",):
        value = data.get(key)
        if value is None:
            value = ""
        if not isinstance(value, str):
            raise HTTPException(status_code=400, detail=f"{key} must be a string")
        out[key] = value
    return out


def _gmail_message(exc: httpx.HTTPStatusError) -> str:
    try:
        message = (exc.response.json().get("error") or {}).get("message") or ""
    except Exception:
        message = ""
    message = str(message or exc.response.text[:300] or f"HTTP {exc.response.status_code}")
    return message.strip().rstrip(".")


_SEND_SCOPES_HINT = "gmail.compose or gmail.send"
_DRAFT_SCOPES_HINT = "gmail.compose (gmail.send alone can't send drafts)"
_REPLY_SCOPES_HINT = "gmail.readonly (to read the email being replied to)"


_CHECK_SENT = (
    "Check the Sent folder (google_mail action=search query='in:sent ...') "
    "before trying again."
)


def _send_outcome_unknown(exc: httpx.HTTPError) -> bool:
    """For a failed send call (messages.send / drafts.send): may Gmail still
    have sent it? Yes on a 5xx, or when the connection broke after the request
    went out — anything but a failure to connect."""
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code >= 500
    return not isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout))


def _outcome_unknown(detail: str) -> HTTPException:
    return HTTPException(
        status_code=502, detail=detail,
        headers={gmail_compose.SEND_OUTCOME_HEADER: "unknown"},
    )


def _gmail_error(
    exc: httpx.HTTPStatusError, *, scopes_hint: str = _SEND_SCOPES_HINT,
    attempted_send: bool = False,
) -> HTTPException:
    """Map a Gmail API error to an answer the agent can act on. *scopes_hint*
    names the scope a 403 "insufficient scopes" means is missing.
    *attempted_send* is True only for the send call itself — an error on a
    lookup before it (drafts.get, the reply's messages.get) means not sent."""
    status = exc.response.status_code
    message = _gmail_message(exc)
    if attempted_send and _send_outcome_unknown(exc):
        return _outcome_unknown(
            f"Gmail API error ({status}): {message} — the email may or may not "
            f"have been sent. {_CHECK_SENT}"
        )
    not_sent = " The email was not sent."
    if status == 401:
        return HTTPException(
            status_code=502,
            detail=f"Google authentication expired — reconnect Google in {_RECONNECT}.{not_sent}",
        )
    if status == 403:
        if "insufficient" in message.lower() or "scope" in message.lower():
            return HTTPException(
                status_code=403,
                detail=(
                    "This deck's Google connection is missing a Gmail scope: the deck "
                    f"admin must include {scopes_hint} in {_RECONNECT} → Scopes, then "
                    f"the user reconnects Google.{not_sent}"
                ),
            )
        return HTTPException(status_code=403, detail=f"Gmail refused: {message}.{not_sent}")
    if status == 429:
        return HTTPException(
            status_code=429,
            detail=f"Gmail rate limit exceeded — try again later.{not_sent}",
        )
    return HTTPException(status_code=502, detail=f"Gmail API error ({status}): {message}.{not_sent}")


def _transport_error(exc: httpx.HTTPError, *, attempted_send: bool = False) -> HTTPException:
    """Map a network failure talking to Gmail. Only the send call itself
    (*attempted_send*) can leave the outcome unknown."""
    if attempted_send and _send_outcome_unknown(exc):
        return _outcome_unknown(
            f"Gmail did not answer ({type(exc).__name__}) — the email may or may not "
            f"have been sent. {_CHECK_SENT}"
        )
    return HTTPException(
        status_code=502,
        detail=f"Could not reach Gmail ({type(exc).__name__}). The email was not sent.",
    )


def _check_gmail_id(value: str, name: str) -> None:
    if not _GMAIL_ID_RE.match(value):
        raise HTTPException(status_code=400, detail=f"{name} is not a valid Gmail id: {value[:80]!r}")


async def _draft_headers(client: httpx.AsyncClient, token: str, draft_id: str) -> dict[str, str]:
    """To / Cc / Bcc / Subject (+ thread) of a draft, before sending it.
    (drafts.get takes only ``format`` — metadata returns every header.)"""
    try:
        resp = await client.get(
            f"{gmail_compose.GMAIL_API}/users/me/drafts/{draft_id}",
            params={"format": "metadata"},
            headers={"Authorization": f"Bearer {token}"},
        )
        resp.raise_for_status()
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code in (400, 404):
            raise HTTPException(
                status_code=404,
                detail=(
                    f"Draft {draft_id} not found — it may have been sent or deleted; "
                    "google_mail action=list_drafts shows the current Draft IDs. "
                    "Nothing was sent."
                ),
            ) from exc
        raise _gmail_error(exc, scopes_hint=_DRAFT_SCOPES_HINT) from exc
    except httpx.HTTPError as exc:
        raise _transport_error(exc) from exc
    message = resp.json().get("message") or {}
    headers: dict[str, str] = {}
    for h in (message.get("payload") or {}).get("headers", []):
        name = (h.get("name") or "").lower()
        if name in ("to", "cc", "bcc", "subject"):
            headers[name] = h.get("value", "") or ""
    headers["thread_id"] = message.get("threadId", "") or ""
    return headers


def _check_recipients(to: str, cc: str, bcc: str, allowlist: list[str]) -> None:
    """400 for unusable / too many recipients, 403 for any off the allowlist."""
    bad = gmail_compose.invalid_recipients(to, cc, bcc)
    if bad:
        raise HTTPException(status_code=400, detail=gmail_compose.invalid_recipients_message(bad))
    addresses = gmail_compose.parse_addresses(to, cc, bcc)
    if not addresses:
        raise HTTPException(
            status_code=400,
            detail="No recipient — set to (or cc / bcc). Nothing was sent.",
        )
    if len(addresses) > gmail_compose.MAX_RECIPIENTS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Too many recipients ({len(addresses)}) — at most "
                f"{gmail_compose.MAX_RECIPIENTS} per email across to/cc/bcc. Nothing was sent."
            ),
        )
    disallowed = gmail_compose.recipients_allowed(addresses, allowlist)
    if disallowed:
        raise HTTPException(
            status_code=403,
            detail=(
                "Not on the user's allowed-recipients list: "
                f"{', '.join(disallowed)}. Nothing was sent. Create a draft with "
                "google_mail action=create_draft instead if the user wants this email; "
                f"they manage the list in {_ENABLE_WHERE}."
            ),
        )


def _agent_label(request: Request) -> str:
    """Best-effort name of the calling agent for the audit row and the bell —
    its Flight Deck slug, else "An agent". (The token is never logged.)"""
    try:
        from captain_claw.flight_deck.server import _find_agent_by_auth

        matched, _owner, slug = _find_agent_by_auth(request.headers.get("X-Agent-Auth", ""))
    except Exception:
        return "An agent"
    return str(slug) if matched and slug else "An agent"


def _clip(text: str, limit: int) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


async def _audit_send(
    db: FlightDeckDB, owner: str, *, status: str, agent: str, to: str, cc: str,
    bcc: str, subject: str, message_id: str, thread_id: str, draft_id: str,
    content_hash: str,
) -> None:
    """The ``gmail_sends`` row — *status* ``sent``, or ``unknown`` when Gmail
    never confirmed. A failed insert is logged, never raised: the send (or
    its error) is what the agent must hear about."""
    try:
        await db.add_gmail_send(
            owner, agent=agent, to_addrs=to, cc_addrs=cc, bcc_addrs=bcc,
            subject=subject, gmail_message_id=message_id, thread_id=thread_id,
            draft_id=draft_id, content_hash=content_hash, status=status,
        )
    except Exception as exc:
        log.error("Gmail send audit insert failed (%s, message %s): %s",
                  status, message_id or "-", type(exc).__name__)


async def _notify_send(
    db: FlightDeckDB, owner: str, *, status: str, agent: str, to: str,
    subject: str, message_id: str,
) -> None:
    """The owner's bell for a send (*status* ``sent`` / ``unknown``)."""
    if status == "unknown":
        title = f"{agent} may have sent an email"
        body = (f"Gmail did not confirm it — check your Sent folder. "
                f"To: {to} - Subject: {subject or '(no subject)'}")
    else:
        title = f"{agent} sent an email"
        body = f"To: {to} - Subject: {subject or '(no subject)'}"
    try:
        await db.add_notification(
            owner, "email_sent", _clip(title, 120), _clip(body, 300),
            "gmail_message" if message_id else "", message_id,
        )
    except Exception as exc:
        log.warning("Gmail send notification failed: %s", type(exc).__name__)


@router.post("/gmail/send")
async def gmail_send(request: Request) -> dict[str, Any]:
    """Send an email as the calling agent's OWNER, if their policy allows.

    Body: the message fields ``to``, ``cc``, ``bcc``, ``subject``, ``body``,
    ``html_body``, ``reply_to_message_id`` (threading and the to / subject
    defaults as create_draft does them) — or ``{"draft_id"}`` to send an
    existing draft as it is now. Same agent gate as ``/access_token``.
    """
    _google._authorize_agent_call(request)
    owner = await _google._agent_owner(request)

    # The enforcement point. It stops model mistakes and prompt injection (an
    # email, page or file telling the agent to "send this to …"). It is NOT a
    # tenant boundary against a shell-capable agent: that agent can already
    # fetch its owner's Gmail token (/access_token — gmail.compose sends) and
    # call Gmail directly. Shell agents are not a tenant boundary anywhere.
    if deck_send_disabled():
        raise _refused(_DECK_OFF_DETAIL, "deck-disabled")
    db = get_db()
    policy = await load_gmail_send_policy(db, owner)
    if not policy["enabled"]:
        raise _refused(_OFF_DETAIL, "off")
    token = await _google.get_valid_google_access_token(owner)
    if not token:
        raise HTTPException(
            status_code=404,
            detail=(
                "Google not connected for this user (or its token could not be "
                f"refreshed) — connect Google in {_RECONNECT}. Nothing was sent."
            ),
        )

    payload = await _read_payload(request)
    draft_id = payload["draft_id"].strip()
    reply_to = payload["reply_to_message_id"].strip()
    body, html_body = payload["body"], payload["html_body"]
    async with httpx.AsyncClient(timeout=60) as client:
        if draft_id:
            if any(payload[k].strip() for k in _MESSAGE_FIELDS):
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Send either draft_id alone or the message fields "
                        "(to/cc/bcc/subject/body/html_body/reply_to_message_id) — not both."
                    ),
                )
            _check_gmail_id(draft_id, "draft_id")
            draft = await _draft_headers(client, token, draft_id)
            to, cc, bcc = draft.get("to", ""), draft.get("cc", ""), draft.get("bcc", "")
            subject, thread_id = draft.get("subject", ""), draft["thread_id"]
            _check_recipients(to, cc, bcc, policy["allowed_recipients"])
            raw = ""
        else:
            to, cc, bcc, subject = payload["to"], payload["cc"], payload["bcc"], payload["subject"]
            thread_id = in_reply_to = references = ""
            if reply_to:
                _check_gmail_id(reply_to, "reply_to_message_id")
                try:
                    ctx = await gmail_compose.fetch_reply_context(client, token, reply_to)
                except httpx.HTTPStatusError as exc:
                    if exc.response.status_code in (400, 404):
                        raise HTTPException(
                            status_code=404,
                            detail=(
                                f"reply_to_message_id {reply_to} was not found in this "
                                "Gmail account. Nothing was sent."
                            ),
                        ) from exc
                    raise _gmail_error(exc, scopes_hint=_REPLY_SCOPES_HINT) from exc
                except httpx.HTTPError as exc:
                    raise _transport_error(exc) from exc
                thread_id = ctx["thread_id"]
                in_reply_to, references = ctx["in_reply_to"], ctx["references"]
                if not to and ctx["reply_to_default"]:
                    to = ctx["reply_to_default"]
                if not subject and ctx["subject_default"]:
                    subject = ctx["subject_default"]
            if not subject.strip() or not (body.strip() or html_body.strip()):
                raise HTTPException(
                    status_code=400,
                    detail="A send needs a subject and a body (body or html_body). Nothing was sent.",
                )
            _check_recipients(to, cc, bcc, policy["allowed_recipients"])
            try:
                # Put exactly the checked addresses on the wire.
                to, cc, bcc = (gmail_compose.format_recipients(f) for f in (to, cc, bcc))
                raw = gmail_compose.build_raw_message(
                    to=to, cc=cc, bcc=bcc, subject=subject,
                    body=body, html_body=html_body,
                    in_reply_to=in_reply_to, references=references,
                )
            except ValueError as exc:  # e.g. a line break in the subject
                raise HTTPException(
                    status_code=400,
                    detail=f"Could not build the email: {exc}. Nothing was sent.",
                ) from exc

        daily_limit = int(policy["daily_limit"])
        async with _lock(_send_locks, owner):
            now = datetime.now(timezone.utc)
            used = await db.count_gmail_sends_since(owner, (now - _DAY).isoformat())
            if used >= daily_limit:
                raise HTTPException(
                    status_code=429,
                    detail=(
                        f"Daily send limit reached: {used} of {daily_limit} emails in the "
                        "last 24 hours. Nothing was sent. Create a draft with "
                        "google_mail action=create_draft instead, or ask the user to raise "
                        f"the limit in {_ENABLE_WHERE}."
                    ),
                )
            chash = ""
            if not draft_id:  # a draft is one server-side object; Gmail consumes it
                chash = gmail_compose.content_hash(
                    to, cc, bcc, subject, body, html_body, reply_to=reply_to,
                )
                dup = await db.find_gmail_send_by_hash(
                    owner, chash, (now - _DUPLICATE_WINDOW).isoformat(),
                )
                if dup and dup.get("status") == "unknown":
                    raise HTTPException(
                        status_code=409,
                        detail=(
                            "Duplicate: this exact email (same recipients, subject and "
                            f"body) was already attempted at {dup['created_at']} and Gmail "
                            "never confirmed it — it may have gone out. Not sent again. "
                            f"{_CHECK_SENT}"
                        ),
                    )
                if dup:
                    raise HTTPException(
                        status_code=409,
                        detail=(
                            "Duplicate: this exact email (same recipients, subject and "
                            f"body) was already sent at {dup['created_at']} (Gmail message "
                            f"{dup['gmail_message_id'] or '?'}). Not sent again."
                        ),
                    )

            if draft_id:
                url, send_body = (f"{gmail_compose.GMAIL_API}/users/me/drafts/send",
                                  {"id": draft_id})
            else:
                url = f"{gmail_compose.GMAIL_API}/users/me/messages/send"
                send_body = {"raw": raw}
                if thread_id:
                    send_body["threadId"] = thread_id
            agent = _agent_label(request)
            try:
                resp = await client.post(
                    url, json=send_body, headers={"Authorization": f"Bearer {token}"},
                )
                resp.raise_for_status()
            except httpx.HTTPError as exc:
                if isinstance(exc, httpx.HTTPStatusError):
                    err = _gmail_error(
                        exc, scopes_hint=_DRAFT_SCOPES_HINT if draft_id else _SEND_SCOPES_HINT,
                        attempted_send=True,
                    )
                else:
                    err = _transport_error(exc, attempted_send=True)
                if _send_outcome_unknown(exc):
                    # Gmail may have sent it: record it like a send (daily count,
                    # duplicate check, history, bell) so a retry can't double it.
                    await _audit_send(
                        db, owner, status="unknown", agent=agent, to=to, cc=cc, bcc=bcc,
                        subject=subject, message_id="", thread_id=thread_id,
                        draft_id=draft_id, content_hash=chash,
                    )
                    await _notify_send(db, owner, status="unknown", agent=agent, to=to,
                                       subject=subject, message_id="")
                raise err from exc
            sent = resp.json()
            message_id = str(sent.get("id") or "")
            sent_thread = str(sent.get("threadId") or thread_id or "")

            await _audit_send(
                db, owner, status="sent", agent=agent, to=to, cc=cc, bcc=bcc,
                subject=subject, message_id=message_id, thread_id=sent_thread,
                draft_id=draft_id, content_hash=chash,
            )

    await _notify_send(db, owner, status="sent", agent=agent, to=to,
                       subject=subject, message_id=message_id)

    return {
        "ok": True,
        "message_id": message_id,
        "thread_id": sent_thread,
        "to": to,
        "cc": cc,
        "bcc": bcc,
        "subject": subject,
        "sent_last_24h": used + 1,
        "daily_limit": daily_limit,
    }
