"""Member invitations — an admin invites someone by email; they choose their
own password.

Users → Invitations: the admin names the email (plus a name, the role, the
kiosk flag and an optional note) and how to send it — from their OWN connected
Google (Gmail) or Microsoft (Outlook) account, or no email at all (they copy
the link and pass it on). The email carries a one-time link to
``<deck>/invite#<token>``: the token rides in the fragment, so it never reaches
a server log or a Referer, and only its sha256 is stored.

Opening the link shows who invited them and the email the account will use;
they choose a password (and confirm their name) and the account is created
there and then, signed in. No password travels by email and nobody else ever
knows it, so the account doesn't need ``must_change_password``.

    GET    /fd/admin/invites               admin   pending (?status=accepted|all) + how this admin can email
    POST   /fd/admin/invites               admin   create (+ send)  → invite, url, email outcome
    POST   /fd/admin/invites/{id}/resend   admin   a new link + expiry (+ send); the old link stops working
    DELETE /fd/admin/invites/{id}          admin   remove; a pending link stops working
    POST   /fd/auth/invite/lookup          public  {token} → who invited whom
    POST   /fd/auth/invite/accept          public  {token, password, display_name} → a session

Sending is the admin's own explicit act, so the agents' Gmail sending policy
(gmail_send_routes, an opt-in for agents) isn't consulted; the deck's kill
switch ``FD_GMAIL_SEND`` is. Microsoft needs the Mail.Send scope, which is
opt-in on the deck's Microsoft client (Connections → Microsoft → Scopes). An
email that fails leaves the invitation in place, and the admin gets the link
to pass on themselves.

Approving a waitlist application (admin_routes.approve_waitlist) emails the
applicant the same way — from the approving admin's account — that their
account is ready, with the deck's sign-in address.

The accept routes live under /fd/auth so the refresh cookie (path=/fd/auth)
they set is the one /fd/auth/refresh reads. Public, so limited per client
network like the waitlist and device pairing.
"""

from __future__ import annotations

import html
import secrets
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from pydantic import BaseModel

from captain_claw import gmail_compose
from captain_claw.flight_deck import google_oauth_routes as _google
from captain_claw.flight_deck import microsoft_oauth_routes as _microsoft
from captain_claw.flight_deck.admin_routes import (
    SYSTEM_CONFIG_DEFAULTS,
    _load_system_config,
    new_user_metadata,
    require_admin,
)
from captain_claw.flight_deck.auth import _fd_auth_enabled, get_db, hash_password, hash_token
from captain_claw.flight_deck.auth_routes import (
    BCRYPT_MAX_BYTES,
    TokenResponse,
    _clean_email,
    _clean_text,
    _client_ip,
    _client_network,
    _issue_session,
    _PairLimiter,
    _rate_limit,
    _valid_email,
)
from captain_claw.logging import get_logger
from captain_claw.microsoft_oauth import granted_scopes

log = get_logger(__name__)

admin_router = APIRouter(prefix="/fd/admin/invites", tags=["admin-invites"])
public_router = APIRouter(prefix="/fd/auth/invite", tags=["auth-invites"])

INVITE_TTL = timedelta(days=7)
INVITE_PATH = "/invite"
NAME_MAX = 100
NOTE_MAX = 1000
TOKEN_MAX = 200
MIN_PASSWORD = 6
SEND_VIA = ("google", "microsoft", "link")

# Public calls, per client network: (count, window seconds).
INVITE_LOOKUP_LIMIT = (30, 600.0)
INVITE_ACCEPT_LIMIT = (20, 600.0)
_lookup_limiter = _PairLimiter()
_accept_limiter = _PairLimiter()

GRAPH_API = "https://graph.microsoft.com/v1.0"

_GOOGLE_WHERE = "Connections → Google"
_MICROSOFT_WHERE = "Connections → Microsoft"


# ── Models ──

class CreateInviteRequest(BaseModel):
    email: str
    display_name: str = ""
    role: str = "user"
    kiosk: bool | None = None  # None = the deck's "new users start in kiosk"
    note: str = ""             # a personal line in the email
    send_via: str = "link"     # google | microsoft | link (no email)


class ResendInviteRequest(BaseModel):
    send_via: str = "link"


class InviteTokenRequest(BaseModel):
    token: str


class AcceptInviteRequest(BaseModel):
    token: str
    password: str
    display_name: str = ""


# ── Helpers ──

def _require_auth_deck() -> None:
    if not _fd_auth_enabled():
        raise HTTPException(
            status_code=403,
            detail="Invitations are unavailable: this Flight Deck runs without accounts (FD_AUTH_ENABLED=false).",
        )


def _now() -> datetime:
    """Wall clock for invitation expiry (monkeypatched by tests)."""
    return datetime.now(timezone.utc)


def _expired(invite: dict) -> bool:
    try:
        return datetime.fromisoformat(invite["expires_at"]) <= _now()
    except (KeyError, TypeError, ValueError):
        return True


def _new_token() -> tuple[str, str, str]:
    """(token, its hash, the expiry) for a new link."""
    token = secrets.token_urlsafe(32)
    return token, hash_token(token), (_now() + INVITE_TTL).isoformat()


def _clean_note(value: str | None, cap: int = NOTE_MAX) -> str:
    """The admin's note: plain text, line breaks kept, capped."""
    text = str(value or "").replace("\r\n", "\n").replace("\r", "\n")
    text = "".join(ch for ch in text if ch == "\n" or ch == "\t" or ch.isprintable())
    return text.strip()[:cap]


def _check_send_via(value: str) -> str:
    via = (value or "link").strip().lower()
    if via not in SEND_VIA:
        raise HTTPException(400, "send_via must be one of: google, microsoft, link")
    return via


async def _names(db, ids: set[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for uid in ids:
        if not uid:
            continue
        u = await db.get_user_by_id(uid)
        if u:
            out[uid] = u.get("display_name") or u.get("email") or ""
    return out


def _view(invite: dict, names: dict[str, str] | None = None) -> dict:
    """An invitation as the Admin page sees it."""
    kiosk = invite.get("kiosk")
    return {
        **invite,
        "kiosk": None if kiosk is None else bool(kiosk),
        "expired": invite.get("status") == "pending" and _expired(invite),
        "invited_by_name": (names or {}).get(invite.get("invited_by") or "", ""),
    }


def _link_base(request: Request) -> str:
    """Where the invitee reaches this deck: FD_PUBLIC_URL, else the address
    the admin's browser is on (the Origin of their request, if this deck
    answers to it), else this request's own base URL."""
    from captain_claw.flight_deck import origin_guard
    from captain_claw.flight_deck.public_file_shares import _parse_origin, _public_base

    if _public_base():
        return _public_base()
    parsed = _parse_origin(request.headers.get("origin", ""))
    if parsed and origin_guard.host_allowed(parsed[1]):
        return parsed[0]
    return str(request.base_url).rstrip("/")


async def _branding(db) -> tuple[str, str]:
    cfg = _load_system_config(await db.get_system_setting("fd:system-config"))
    name = str(cfg.get("instance_name") or "").strip() or SYSTEM_CONFIG_DEFAULTS["instance_name"]
    return name, str(cfg.get("instance_description") or "").strip()


# ── The email ──

def _long_date(iso: str) -> str:
    try:
        d = datetime.fromisoformat(iso)
    except (TypeError, ValueError):
        return iso
    return f"{d.day} {d:%B %Y}"


def compose_invite(
    *, instance_name: str, instance_description: str, inviter: str, invitee: str,
    email: str, url: str, expires_at: str, note: str,
) -> tuple[str, str, str]:
    """(subject, plain text, HTML) of an invitation email."""
    subject = f"{inviter} invited you to {instance_name}"
    joined = instance_name + (f" — {instance_description}" if instance_description else "")
    until = _long_date(expires_at)
    greeting = f"Hi {invitee}," if invitee else "Hi,"

    lines = [greeting, "", f"{inviter} invited you to join {joined}.", ""]
    if note:
        lines += [f"{inviter} wrote:", *(f"  {ln}" for ln in note.split("\n")), ""]
    lines += [
        "Choose your password to get started:",
        url,
        "",
        f"You'll sign in with {email}. The link works once and expires on {until}.",
        "",
        "If you weren't expecting this invitation, you can ignore this email.",
    ]
    text = "\n".join(lines)

    e = html.escape
    note_html = ""
    if note:
        note_html = (
            f'<p style="margin:0 0 6px;color:#555;font-size:13px">{e(inviter)} wrote:</p>'
            '<blockquote style="margin:0 0 20px;padding:10px 14px;border-left:3px solid #7c3aed;'
            f'background:#f6f3ff;color:#333;white-space:pre-wrap">{e(note)}</blockquote>'
        )
    html_body = (
        '<div style="font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Helvetica,Arial,'
        'sans-serif;font-size:15px;line-height:1.5;color:#1a1a1a;max-width:520px">'
        f'<p style="margin:0 0 16px">{e(greeting)}</p>'
        f'<p style="margin:0 0 20px">{e(inviter)} invited you to join <strong>{e(instance_name)}</strong>'
        + (f" — {e(instance_description)}" if instance_description else "")
        + ".</p>"
        + note_html
        + '<p style="margin:0 0 24px"><a href="' + e(url, quote=True) + '" style="display:inline-block;'
        'padding:11px 20px;border-radius:8px;background:#7c3aed;color:#ffffff;text-decoration:none;'
        'font-weight:600">Choose your password</a></p>'
        f'<p style="margin:0 0 8px;color:#555;font-size:13px">You\'ll sign in with {e(email)}. '
        f"The link works once and expires on {e(until)}.</p>"
        '<p style="margin:0 0 8px;color:#555;font-size:13px">If the button doesn\'t work, open this '
        f'address: <a href="{e(url, quote=True)}" style="color:#7c3aed;word-break:break-all">{e(url)}</a></p>'
        '<p style="margin:16px 0 0;color:#888;font-size:12px">If you weren\'t expecting this '
        "invitation, you can ignore this email.</p>"
        "</div>"
    )
    return subject, text, html_body


def compose_waitlist_approved(
    *, instance_name: str, instance_description: str, approver: str, applicant: str,
    email: str, cohort: str, url: str,
) -> tuple[str, str, str]:
    """(subject, plain text, HTML) of the email telling a waitlist applicant
    they're in. Their account already exists with the password they chose
    when applying; a reply reaches the admin who approved them."""
    subject = f"You're in — your {instance_name} account is ready"
    joined = instance_name + (f" — {instance_description}" if instance_description else "")
    # The cohort defaults to the instance name; only name it when it adds something.
    for_cohort = f" for {cohort}" if cohort and cohort != instance_name else ""
    greeting = f"Hi {applicant}," if applicant else "Hi,"

    text = "\n".join([
        greeting,
        "",
        f"{approver} approved your application{for_cohort} to join {joined}.",
        "",
        "Sign in with your email and the password you chose when you applied:",
        url,
        "",
        f"Your email: {email}",
        "",
        "Forgot that password? Reply to this email and we'll sort it out.",
    ])

    e = html.escape
    html_body = (
        '<div style="font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Helvetica,Arial,'
        'sans-serif;font-size:15px;line-height:1.5;color:#1a1a1a;max-width:520px">'
        f'<p style="margin:0 0 16px">{e(greeting)}</p>'
        f'<p style="margin:0 0 20px">{e(approver)} approved your application{e(for_cohort)} to join '
        f'<strong>{e(instance_name)}</strong>'
        + (f" — {e(instance_description)}" if instance_description else "")
        + ".</p>"
        '<p style="margin:0 0 12px">Sign in with your email and the password you chose when you applied.</p>'
        '<p style="margin:0 0 24px"><a href="' + e(url, quote=True) + '" style="display:inline-block;'
        'padding:11px 20px;border-radius:8px;background:#7c3aed;color:#ffffff;text-decoration:none;'
        'font-weight:600">Sign in</a></p>'
        f'<p style="margin:0 0 8px;color:#555;font-size:13px">Your email: {e(email)}</p>'
        '<p style="margin:0 0 8px;color:#555;font-size:13px">If the button doesn\'t work, open this '
        f'address: <a href="{e(url, quote=True)}" style="color:#7c3aed;word-break:break-all">{e(url)}</a></p>'
        '<p style="margin:16px 0 0;color:#888;font-size:12px">Forgot that password? Reply to this '
        "email and we'll sort it out.</p>"
        "</div>"
    )
    return subject, text, html_body


# ── Sending, from the admin's own account ──

def _http() -> httpx.AsyncClient:
    """The client the sends use (tests swap in a mock transport)."""
    return httpx.AsyncClient(timeout=30)


class _MailError(Exception):
    """An email that didn't go out; the message says why, for the admin."""


async def mail_options(db, user_id: str) -> dict[str, dict[str, Any]]:
    """How *user_id* can email an invitation: per provider ``available``,
    ``from`` (their connected address) and ``reason`` when not available.
    No network: stored tokens and scopes only."""
    from captain_claw.flight_deck.gmail_send_routes import deck_send_disabled

    google: dict[str, Any] = {"available": False, "from": "", "reason": ""}
    if not await _google.is_google_connected(user_id):
        google["reason"] = f"Connect your Google account in {_GOOGLE_WHERE} to send from Gmail."
    else:
        tokens = await _google._load_tokens(db, user_id)
        scopes = set((tokens.scope if tokens else "").split())
        google["from"] = str((await _google._load_user(db, user_id) or {}).get("email") or "")
        if not scopes & set(gmail_compose.SEND_SCOPES):
            google["reason"] = ("Your Google connection can't send mail: the deck's Google client needs "
                                f"the gmail.compose or gmail.send scope ({_GOOGLE_WHERE}), then reconnect.")
        elif deck_send_disabled():
            google["reason"] = "Gmail sending is turned off on this deck (FD_GMAIL_SEND)."
        else:
            google["available"] = True

    microsoft: dict[str, Any] = {"available": False, "from": "", "reason": ""}
    if not await _microsoft.is_microsoft_connected(user_id):
        microsoft["reason"] = f"Connect your Microsoft account in {_MICROSOFT_WHERE} to send from Outlook."
    else:
        tokens = await _microsoft._load_tokens(db, user_id)
        microsoft["from"] = str((await _microsoft._load_user(db, user_id) or {}).get("email") or "")
        if "mail.send" not in granted_scopes(tokens.scope if tokens else ""):
            microsoft["reason"] = ("Your Microsoft connection can't send mail: add the Mail (send) scope "
                                   f"in {_MICROSOFT_WHERE} → Scopes, then reconnect.")
        else:
            microsoft["available"] = True
    return {"google": google, "microsoft": microsoft}


def _api_error(resp: httpx.Response) -> str:
    try:
        err = resp.json().get("error")
        msg = err.get("message") if isinstance(err, dict) else str(err or "")
    except Exception:
        msg = ""
    return (msg or resp.text or f"HTTP {resp.status_code}")[:300]


def _unsure(service: str, exc: httpx.HTTPError) -> _MailError:
    return _MailError(
        f"{service} didn't answer ({type(exc).__name__}). The email may still have gone out — "
        "check your Sent folder before sending again."
    )


async def _send_google(user_id: str, to: str, subject: str, text: str, html_body: str) -> None:
    token = await _google.get_valid_google_access_token(user_id)
    if not token:
        raise _MailError(f"Google isn't connected for your account — reconnect it in {_GOOGLE_WHERE}.")
    raw = gmail_compose.build_raw_message(to=to, subject=subject, body=text, html_body=html_body)
    try:
        async with _http() as client:
            resp = await client.post(
                f"{gmail_compose.GMAIL_API}/users/me/messages/send",
                headers={"Authorization": f"Bearer {token}"}, json={"raw": raw},
            )
    except httpx.ConnectError as exc:
        raise _MailError(f"Couldn't reach Gmail ({exc}). Nothing was sent.") from exc
    except httpx.HTTPError as exc:
        raise _unsure("Gmail", exc) from exc
    if resp.status_code >= 400:
        raise _MailError(f"Gmail refused the email ({resp.status_code}): {_api_error(resp)}")


async def _send_microsoft(user_id: str, to: str, subject: str, html_body: str) -> None:
    token = await _microsoft.get_valid_microsoft_access_token(user_id)
    if not token:
        raise _MailError(f"Microsoft isn't connected for your account — reconnect it in {_MICROSOFT_WHERE}.")
    message = {
        "subject": subject,
        "body": {"contentType": "HTML", "content": html_body},
        "toRecipients": [{"emailAddress": {"address": to}}],
    }
    try:
        async with _http() as client:
            resp = await client.post(
                f"{GRAPH_API}/me/sendMail",
                headers={"Authorization": f"Bearer {token}"},
                json={"message": message, "saveToSentItems": True},
            )
    except httpx.ConnectError as exc:
        raise _MailError(f"Couldn't reach Microsoft ({exc}). Nothing was sent.") from exc
    except httpx.HTTPError as exc:
        raise _unsure("Microsoft", exc) from exc
    if resp.status_code >= 400:
        raise _MailError(f"Outlook refused the email ({resp.status_code}): {_api_error(resp)}")


def _admin_name(admin: dict) -> str:
    return admin.get("display_name") or admin.get("email") or "An administrator"


async def send_from_admin(db, admin: dict, via: str, to: str, subject: str, text: str,
                          html_body: str) -> dict[str, Any]:
    """Email *to* from the admin's own account (*via* google | microsoft).
    Never raises: the outcome is ``sent``, ``via``, ``from`` and ``error``."""
    out: dict[str, Any] = {"sent": False, "via": via, "from": "", "error": ""}
    options = (await mail_options(db, admin["id"]))[via]
    out["from"] = options["from"]
    if not options["available"]:
        out["error"] = options["reason"]
        return out
    try:
        if via == "google":
            await _send_google(admin["id"], to, subject, text, html_body)
        else:
            await _send_microsoft(admin["id"], to, subject, html_body)
    except _MailError as exc:
        out["error"] = str(exc)
        return out
    out["sent"] = True
    return out


async def _deliver(db, admin: dict, invite: dict, url: str, via: str) -> dict[str, Any]:
    """Email *invite* the way the admin chose. Never raises: the outcome is
    ``sent``, ``via``, ``from`` and ``error``."""
    if via == "link":
        return {"sent": False, "via": via, "from": "", "error": ""}
    instance_name, description = await _branding(db)
    subject, text, html_body = compose_invite(
        instance_name=instance_name, instance_description=description,
        inviter=_admin_name(admin),
        invitee=invite.get("display_name") or "", email=invite["email"], url=url,
        expires_at=invite["expires_at"], note=invite.get("note") or "",
    )
    out = await send_from_admin(db, admin, via, invite["email"], subject, text, html_body)
    if not out["sent"]:
        log.warning("Invitation email failed", invite=invite["id"], via=via, error=out["error"][:200])
        return out
    await db.mark_invite_sent(invite["id"], via, out["from"])
    log.info("Invitation emailed", invite=invite["id"], via=via)
    return out


async def deliver_waitlist_approval(db, admin: dict, entry: dict, display_name: str,
                                    request: Request, via: str) -> dict[str, Any]:
    """Tell an approved waitlist applicant they can sign in, emailed the way
    the admin chose (``none`` = no email). Never raises; the outcome carries
    ``url``, the deck's sign-in address, for the admin to pass on when no
    email went out."""
    url = f"{_link_base(request)}/"
    if via == "none":
        return {"sent": False, "via": via, "from": "", "error": "", "url": url}
    instance_name, description = await _branding(db)
    subject, text, html_body = compose_waitlist_approved(
        instance_name=instance_name, instance_description=description,
        approver=_admin_name(admin), applicant=display_name, email=entry["email"],
        cohort=entry.get("cohort") or "", url=url,
    )
    out = await send_from_admin(db, admin, via, entry["email"], subject, text, html_body)
    if out["sent"]:
        log.info("Waitlist approval emailed", entry=entry["id"], via=via)
    else:
        log.warning("Waitlist approval email failed", entry=entry["id"], via=via, error=out["error"][:200])
    return {**out, "url": url}


# ── Admin ──

@admin_router.get("")
async def list_invites(status: str = Query("pending"), admin: dict = Depends(require_admin)):
    """Invitations, newest first, and how this admin can email one.
    *status* is pending | accepted | all."""
    _require_auth_deck()
    if status not in ("pending", "accepted", "all"):
        raise HTTPException(400, "status must be one of: pending, accepted, all")
    db = get_db()
    rows = await db.list_invites(None if status == "all" else status)
    names = await _names(db, {r.get("invited_by") or "" for r in rows})
    return {
        "invites": [_view(r, names) for r in rows],
        "mail": await mail_options(db, admin["id"]),
        "ttl_days": INVITE_TTL.days,
    }


@admin_router.post("", status_code=201)
async def create_invite(body: CreateInviteRequest, request: Request,
                        admin: dict = Depends(require_admin)):
    """Invite *email*: a pending invitation and its link, emailed when
    ``send_via`` names a provider. An email that fails still leaves the
    invitation — the response carries the link and the error."""
    _require_auth_deck()
    db = get_db()
    email = _clean_email(body.email)
    if not _valid_email(email):
        raise HTTPException(400, "Enter a valid email address")
    if body.role not in ("user", "admin"):
        raise HTTPException(400, "Role must be 'user' or 'admin'")
    via = _check_send_via(body.send_via)
    if await db.get_user_by_email(email):
        raise HTTPException(409, "This email already has an account")
    existing = await db.get_pending_invite(email)
    if existing:
        if not _expired(existing):
            raise HTTPException(409, "This email already has a pending invitation — resend it from the list")
        await db.delete_invite(existing["id"])  # a dead link: replaced by this one

    token, token_hash, expires_at = _new_token()
    try:
        invite = await db.create_invite(
            email=email, display_name=_clean_text(body.display_name, NAME_MAX),
            role=body.role, kiosk=None if body.role == "admin" else body.kiosk,
            note=_clean_note(body.note), token_hash=token_hash,
            invited_by=admin["id"], expires_at=expires_at,
        )
    except sqlite3.IntegrityError:  # another admin invited them a moment ago
        raise HTTPException(409, "This email already has a pending invitation — resend it from the list")
    url = f"{_link_base(request)}{INVITE_PATH}#{token}"
    log.info("Invitation created", invite=invite["id"], via=via)
    mail = await _deliver(db, admin, invite, url, via)
    names = await _names(db, {admin["id"]})
    return {"ok": True, "invite": _view(await db.get_invite(invite["id"]) or invite, names),
            "url": url, "email": mail}


@admin_router.post("/{invite_id}/resend")
async def resend_invite(invite_id: str, request: Request, body: ResendInviteRequest | None = None,
                        admin: dict = Depends(require_admin)):
    """A new link and a fresh expiry for a pending invitation (the old link
    stops working), emailed when ``send_via`` names a provider."""
    _require_auth_deck()
    db = get_db()
    via = _check_send_via(body.send_via if body else "link")
    invite = await db.get_invite(invite_id)
    if not invite:
        raise HTTPException(404, "Invitation not found")
    if invite["status"] != "pending":
        raise HTTPException(409, "This invitation was already accepted")
    if await db.get_user_by_email(invite["email"]):
        raise HTTPException(409, "This email already has an account — remove the invitation")
    token, token_hash, expires_at = _new_token()
    if not await db.renew_invite(invite_id, token_hash, expires_at):
        raise HTTPException(409, "This invitation was already accepted")
    invite = await db.get_invite(invite_id) or invite
    url = f"{_link_base(request)}{INVITE_PATH}#{token}"
    mail = await _deliver(db, admin, invite, url, via)
    names = await _names(db, {invite.get("invited_by") or ""})
    return {"ok": True, "invite": _view(await db.get_invite(invite_id) or invite, names),
            "url": url, "email": mail}


@admin_router.delete("/{invite_id}")
async def delete_invite(invite_id: str, admin: dict = Depends(require_admin)):
    """Remove an invitation outright; a pending link stops working."""
    _require_auth_deck()
    if not await get_db().delete_invite(invite_id):
        raise HTTPException(404, "Invitation not found")
    return {"ok": True}


# ── The invitee (public) ──

async def _usable_invite(db, token: str) -> dict:
    """The pending, unexpired invitation *token* opens — else the reason."""
    token = (token or "").strip()
    invite = (await db.get_invite_by_token_hash(hash_token(token))
              if token and len(token) <= TOKEN_MAX else None)
    if not invite:
        raise HTTPException(
            404, "This invitation link isn't valid — a newer one may have replaced it. "
                 "Ask whoever invited you for a new invitation.")
    if invite["status"] != "pending":
        raise HTTPException(409, "This invitation was already used — sign in with your email and password.")
    if _expired(invite):
        raise HTTPException(410, "This invitation has expired — ask whoever invited you for a new one.")
    return invite


def _network_key(request: Request) -> str:
    return f"net:{_client_network(_client_ip(request))}"


@public_router.post("/lookup")
async def lookup_invite(body: InviteTokenRequest, request: Request):
    """Who invited whom, for the accept page."""
    _require_auth_deck()
    _rate_limit(_lookup_limiter, _network_key(request), INVITE_LOOKUP_LIMIT)
    db = get_db()
    invite = await _usable_invite(db, body.token)
    inviter = await db.get_user_by_id(invite.get("invited_by") or "")
    return {
        "email": invite["email"],
        "display_name": invite.get("display_name") or "",
        "inviter": (inviter or {}).get("display_name") or (inviter or {}).get("email") or "",
        "expires_at": invite["expires_at"],
    }


@public_router.post("/accept", response_model=TokenResponse)
async def accept_invite(body: AcceptInviteRequest, request: Request, response: Response):
    """Create the invited account with the password the invitee chose, and
    sign them in."""
    _require_auth_deck()
    _rate_limit(_accept_limiter, _network_key(request), INVITE_ACCEPT_LIMIT)
    db = get_db()
    invite = await _usable_invite(db, body.token)
    if len(body.password) < MIN_PASSWORD:
        raise HTTPException(400, f"Password must be at least {MIN_PASSWORD} characters")
    if len(body.password.encode()) > BCRYPT_MAX_BYTES:
        raise HTTPException(400, "Password is too long (72 bytes at most)")
    email = invite["email"]
    if await db.get_user_by_email(email):
        raise HTTPException(409, "This email already has an account — sign in instead.")
    name = (_clean_text(body.display_name, NAME_MAX)
            or invite.get("display_name") or email.split("@")[0])

    if not await db.claim_invite(invite["id"]):  # the same link, a moment ago
        raise HTTPException(409, "This invitation was already used — sign in with your email and password.")
    kiosk = invite.get("kiosk")
    meta = await new_user_metadata(db, invite["role"], None if kiosk is None else bool(kiosk))
    try:
        created = await db.create_user(
            email=email, password_hash=hash_password(body.password),
            display_name=name, role=invite["role"], metadata=meta,
        )
    except sqlite3.IntegrityError:  # an admin made the account a moment ago
        try:
            await db.release_invite(invite["id"])
        except sqlite3.IntegrityError:
            pass
        raise HTTPException(409, "This email already has an account — sign in instead.")
    await db.finish_invite(invite["id"], created["id"])

    # Their waitlist application, if any, is settled by the invitation.
    entry = await db.get_pending_waitlist_entry(email)
    if entry:
        await db.decide_waitlist_entry(entry["id"], "approved", invite.get("invited_by") or "",
                                       user_id=created["id"])
    if invite.get("invited_by"):
        try:
            await db.add_notification(
                invite["invited_by"], "invite_accepted", f"{name} accepted your invitation",
                body=f"{email} joined and chose their password.", ref_type="user", ref_id=created["id"],
            )
        except Exception:
            pass
    log.info("Invitation accepted", invite=invite["id"], user=created["id"])
    user = await db.get_user_by_id(created["id"]) or created
    return await _issue_session(db, user, response)
