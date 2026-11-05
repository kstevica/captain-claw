"""Email inboxes over IMAP/SMTP with a username and password — Flight Deck's side.

Each Flight Deck user can connect any number of inboxes (up to
:data:`MAX_ACCOUNTS`): a company mail server, Fastmail, iCloud, Yahoo, a Gmail
account with an app password, Proton Mail through Bridge … The list lives in
the user's ``user_settings`` under ``imap_mail:accounts`` (server-owned: never
readable or writable through ``/fd/settings``), with every password sealed by
:mod:`fd_crypto` (the deck's data key, kept apart from the database).

**The password never leaves Flight Deck.** Agents don't log in to IMAP
themselves: their ``imap_mail`` tool asks ``POST /fd/imap/agent/<action>`` and
the deck does the IMAP / SMTP work (:mod:`captain_claw.imap_mail`, in a worker
thread) for the agent's OWNER — resolved from the ``X-Agent-Auth`` token this
deck minted (``google_oauth_routes._agent_owner``), never from the agent's word.

Per inbox the user chooses:

* ``agents`` — which of their agents may use it (slugs; empty = all of them);
* ``sending`` — ``enabled`` (OFF by default: drafts only until the user opts
  in), ``allowed_recipients`` (empty = anyone; addresses and ``@domain``) and
  ``daily_limit`` (sends per rolling 24 hours, default 50);
* ``signature`` — plain text Flight Deck adds to the end of every draft and
  send it composes from that inbox.

A send goes through the one gate (:func:`_send`): the deck kill switch
``FD_IMAP_SEND=off``, the inbox's policy, the repeat check (the same email
already in Drafts, or in Sent within ``FD_IMAP_REPEAT_CHECK_DAYS``, default 14,
unless the agent passes ``allow_repeat``), then under the inbox's lock the daily
limit and the 10-minute duplicate check, the SMTP delivery, the ``imap_sends``
audit row and the owner's bell. A send the SMTP server took without confirming
is recorded as ``unknown`` — counted and caught as a duplicate, never retried.

Like Google and Microsoft, only on an auth-enabled deck or the desktop app's
own deck. A shared-agent member's request is refused by the speaker-grant
middleware (``/fd/imap`` is not grant-aware) and by the agent tool itself.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from captain_claw import gmail_compose, imap_mail, mail_attachments, mail_signature
from captain_claw.flight_deck import fd_crypto
from captain_claw.flight_deck import google_oauth_routes as _g
from captain_claw.flight_deck.auth import get_current_user, get_db
from captain_claw.flight_deck.db import FlightDeckDB
from captain_claw.imap_mail import Account, MailError, SendOutcomeUnknown
from captain_claw.logging import get_logger

log = get_logger(__name__)

_K_ACCOUNTS = "imap_mail:accounts"  # per user, server-owned

MAX_ACCOUNTS = 20
DEFAULT_DAILY_LIMIT = 50
MAX_DAILY_LIMIT = 500
MAX_ALLOWED_RECIPIENTS = 200
DEFAULT_REPEAT_CHECK_DAYS = 14
MAX_TOTAL_ATTACHMENT_BYTES = 25 * 1024 * 1024
# Concurrent IMAP connections per inbox (servers cap them, often near 10).
_PER_ACCOUNT_CONNECTIONS = 3

_DAY = timedelta(hours=24)
_DUPLICATE_WINDOW = timedelta(minutes=10)
_WHERE = "Flight Deck → Connections → Email (IMAP)"
_AUTH_OFF_DETAIL = (
    "Email (IMAP) via Flight Deck isn't available on this deck: Flight Deck auth is "
    "disabled (FD_AUTH_ENABLED=false). Enable auth to connect inboxes."
)
_AGENT_PREFIX = "/fd/imap/agent"

# The connection fields — changing any of them (or a password) re-tests the inbox.
_CONNECTION_FIELDS = ("email", "username", "imap_host", "imap_port", "imap_security", "smtp_host",
                      "smtp_port", "smtp_security", "smtp_username")
_PLAIN_FIELDS = _CONNECTION_FIELDS + ("label", "display_name", "save_sent")


def _require_auth_deck(request: Request) -> None:
    if _g._connections_available():
        return
    agent = request.url.path.startswith((_AGENT_PREFIX, "/fd/imap/agent_status"))
    raise HTTPException(status_code=403 if agent else 503, detail=_AUTH_OFF_DETAIL)


router = APIRouter(prefix="/fd/imap", tags=["imap-mail"], dependencies=[Depends(_require_auth_deck)])

_locks: dict[str, asyncio.Lock] = {}
_send_locks: dict[str, asyncio.Lock] = {}
_conn_limits: dict[str, asyncio.Semaphore] = {}


def _lock(store: dict[str, asyncio.Lock], key: str) -> asyncio.Lock:
    lock = store.get(key)
    if lock is None:
        lock = store[key] = asyncio.Lock()
    return lock


def deck_send_disabled() -> bool:
    return os.environ.get("FD_IMAP_SEND", "").strip().lower() in ("off", "0", "false", "no")


def repeat_check_days() -> int:
    raw = os.environ.get("FD_IMAP_REPEAT_CHECK_DAYS", "").strip()
    try:
        days = int(raw) if raw else DEFAULT_REPEAT_CHECK_DAYS
    except ValueError:
        return DEFAULT_REPEAT_CHECK_DAYS
    return days if days >= 0 else DEFAULT_REPEAT_CHECK_DAYS


def _now() -> str:
    return datetime.now(UTC).isoformat()


# ── storage ─────────────────────────────────────────────────────────


def _default_sending() -> dict[str, Any]:
    return {"enabled": False, "allowed_recipients": [], "daily_limit": DEFAULT_DAILY_LIMIT}


def _clean_sending(raw: Any) -> dict[str, Any]:
    """A stored sending policy; the defaults (OFF) for anything unreadable."""
    try:
        if not isinstance(raw, dict):
            raise ValueError
        enabled = raw.get("enabled", False)
        limit = raw.get("daily_limit", DEFAULT_DAILY_LIMIT)
        allowed = gmail_compose.normalize_allowlist(raw.get("allowed_recipients") or [])
        if (not isinstance(enabled, bool) or not isinstance(limit, int) or isinstance(limit, bool)
                or not 1 <= limit <= MAX_DAILY_LIMIT or len(allowed) > MAX_ALLOWED_RECIPIENTS):
            raise ValueError
        return {"enabled": enabled, "allowed_recipients": allowed, "daily_limit": limit}
    except Exception:
        return _default_sending()


async def _load(db: FlightDeckDB, user_id: str) -> list[dict[str, Any]]:
    raw = await db.get_setting(user_id, _K_ACCOUNTS)
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except Exception:
        log.warning("Unreadable IMAP account list for a user; treating it as empty")
        return []
    out = []
    for rec in data if isinstance(data, list) else []:
        if isinstance(rec, dict) and imap_mail.ACCOUNT_ID_RE.match(str(rec.get("id") or "")):
            rec["sending"] = _clean_sending(rec.get("sending"))
            rec["agents"] = [str(a) for a in rec.get("agents") or [] if isinstance(a, str)][:200]
            out.append(rec)
    return out


async def _save(db: FlightDeckDB, user_id: str, records: list[dict[str, Any]]) -> None:
    await db.set_settings(user_id, {_K_ACCOUNTS: json.dumps(records, ensure_ascii=True)})


def _account(rec: dict[str, Any]) -> Account:
    """The record with its passwords unsealed; MailError when the deck can't
    open them (its data key was lost or replaced)."""
    try:
        password = fd_crypto.unseal(rec.get("password") or "")
        smtp_password = fd_crypto.unseal(rec.get("smtp_password") or "") if rec.get("smtp_password") else ""
    except fd_crypto.CryptoUnavailable as exc:
        raise MailError(f"Flight Deck can't open stored passwords: {exc}.", 503) from None
    if not isinstance(password, str) or not password:
        raise MailError(
            f"The password for {rec.get('email')} can't be read on this Flight Deck any more "
            f"(its data key changed) — the user re-enters it in {_WHERE}.", 409)
    return Account(
        id=rec["id"], email=rec.get("email", ""), username=rec.get("username", ""), password=password,
        imap_host=rec.get("imap_host", ""), imap_port=int(rec.get("imap_port") or 993),
        imap_security=rec.get("imap_security", "ssl"), smtp_host=rec.get("smtp_host", ""),
        smtp_port=int(rec.get("smtp_port") or 465), smtp_security=rec.get("smtp_security", "ssl"),
        smtp_username=rec.get("smtp_username", ""),
        smtp_password=smtp_password if isinstance(smtp_password, str) else "",
        label=rec.get("label", ""), display_name=rec.get("display_name", ""),
        save_sent=bool(rec.get("save_sent", True)),
    )


def _new_id(base: str, taken: set[str]) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", base.lower()).strip("-")[:20] or "inbox"
    if not slug[0].isalnum():
        slug = "inbox"
    candidate, n = slug, 2
    while candidate in taken:
        candidate = f"{slug[:20]}-{n}"
        n += 1
    return candidate


async def _sent_24h(db: FlightDeckDB, owner: str, account_id: str) -> int:
    return await db.count_imap_sends_since(owner, account_id, (datetime.now(UTC) - _DAY).isoformat())


async def _view(db: FlightDeckDB, owner: str, rec: dict[str, Any]) -> dict[str, Any]:
    """What the browser sees of a record — never a password."""
    out = {k: rec.get(k) for k in ("id",) + _PLAIN_FIELDS}
    out.update(
        can_send=bool(rec.get("smtp_host")),
        separate_smtp_login=bool(rec.get("smtp_username")),
        agents=list(rec.get("agents") or []),
        sending=dict(rec["sending"]),
        signature=str(rec.get("signature") or ""),
        sent_last_24h=await _sent_24h(db, owner, rec["id"]),
        created_at=rec.get("created_at", ""), updated_at=rec.get("updated_at", ""),
        last_check=rec.get("last_check") or {},
        provider=rec.get("provider", ""),
    )
    return out


# ── running the core in a thread ────────────────────────────────────


async def _run(account: Account, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    sem = _conn_limits.get(account.email + "|" + account.imap_host)
    if sem is None:
        sem = _conn_limits[account.email + "|" + account.imap_host] = asyncio.Semaphore(_PER_ACCOUNT_CONNECTIONS)
    async with sem:
        return await asyncio.to_thread(fn, *args, **kwargs)


def _http(exc: MailError) -> HTTPException:
    return HTTPException(status_code=exc.status, detail=exc.message)


# ── user routes ─────────────────────────────────────────────────────


class AccountBody(BaseModel):
    label: Any = None
    email: Any = None
    username: Any = None
    password: Any = None
    display_name: Any = None
    imap_host: Any = None
    imap_port: Any = None
    imap_security: Any = None
    smtp_host: Any = None
    smtp_port: Any = None
    smtp_security: Any = None
    smtp_username: Any = None
    smtp_password: Any = None
    save_sent: Any = None
    agents: Any = None
    sending: Any = None
    signature: Any = None
    provider: Any = None
    # POST /test only: re-use this stored inbox's passwords when none is typed.
    account_id: Any = None


def _text(v: Any) -> str:
    return v.strip() if isinstance(v, str) else ("" if v is None else str(v).strip())


def _settings_of(body: AccountBody, base: dict[str, Any] | None = None) -> dict[str, Any]:
    out = dict(base or {})
    for k in _PLAIN_FIELDS:
        v = getattr(body, k)
        if v is not None:
            out[k] = v
    return out


def _check_agents(value: Any) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(a, str) for a in value):
        raise HTTPException(status_code=400, detail="agents must be a list of agent slugs ([] = all)")
    return sorted({a.strip() for a in value if a.strip()})[:200]


def _check_sending(value: Any, current: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise HTTPException(status_code=400, detail="sending must be an object")
    out = dict(current)
    if "enabled" in value and value["enabled"] is not None:
        if not isinstance(value["enabled"], bool):
            raise HTTPException(status_code=400, detail="sending.enabled must be true or false")
        out["enabled"] = value["enabled"]
    if "daily_limit" in value and value["daily_limit"] is not None:
        lim = value["daily_limit"]
        if not isinstance(lim, int) or isinstance(lim, bool) or not 1 <= lim <= MAX_DAILY_LIMIT:
            raise HTTPException(status_code=400,
                                detail=f"daily_limit must be a whole number from 1 to {MAX_DAILY_LIMIT}")
        out["daily_limit"] = lim
    if "allowed_recipients" in value and value["allowed_recipients"] is not None:
        try:
            allowed = gmail_compose.normalize_allowlist(value["allowed_recipients"])
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if len(allowed) > MAX_ALLOWED_RECIPIENTS:
            raise HTTPException(status_code=400,
                                detail=f"allowed_recipients can hold at most {MAX_ALLOWED_RECIPIENTS} entries")
        out["allowed_recipients"] = allowed
    return out


def _require_crypto() -> None:
    if not fd_crypto.available():
        raise HTTPException(status_code=503, detail=(
            "This Flight Deck can't store passwords encrypted: the 'cryptography' package isn't "
            "installed (pip install cryptography, then restart Flight Deck)."))


@router.get("/accounts")
async def imap_accounts(_user: dict = Depends(get_current_user)) -> dict[str, Any]:
    db = get_db()
    uid = _g._effective_owner(_user)
    recs = await _load(db, uid)
    return {
        "accounts": [await _view(db, uid, r) for r in recs],
        "max_accounts": MAX_ACCOUNTS,
        "deck_send_disabled": deck_send_disabled(),
        "crypto_available": fd_crypto.available(),
    }


_ISPDB = "https://autoconfig.thunderbird.net/v1.1/"


@router.get("/suggest")
async def imap_suggest(email: str = "", _user: dict = Depends(get_current_user)) -> dict[str, Any]:
    """Server settings to start from for *email*: a built-in preset, else
    Thunderbird's provider database (only the domain is sent), else the usual
    imap./smtp. guess."""
    addr = email.strip().lower()
    out = imap_mail.suggest_settings(addr)
    if out["source"] != "guess" or "@" not in addr:
        return out
    domain = addr.rsplit("@", 1)[1]
    try:
        imap_mail.clean_host(domain, "domain")
    except ValueError:
        return out
    try:
        async with httpx.AsyncClient(timeout=6, follow_redirects=False) as client:
            resp = await client.get(_ISPDB + domain)
        if resp.status_code == 200 and len(resp.content) < 200_000:
            found = imap_mail.parse_autoconfig(resp.text, addr)
            if found:
                return found
    except httpx.HTTPError:
        pass
    return out


def _test_account(settings: dict[str, Any], password: str, smtp_password: str, account_id: str) -> Account:
    try:
        return imap_mail.account_from_settings(settings, account_id=account_id, password=password,
                                               smtp_password=smtp_password)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


async def _stored_passwords(db: FlightDeckDB, uid: str, account_id: Any) -> tuple[str, str]:
    if not account_id:
        return "", ""
    rec = next((r for r in await _load(db, uid) if r["id"] == str(account_id)), None)
    if rec is None:
        raise HTTPException(status_code=404, detail="No such inbox")
    try:
        acct = _account(rec)
    except MailError as exc:
        raise _http(exc) from exc
    return acct.password, acct.smtp_password


@router.post("/test")
async def imap_test(body: AccountBody, _user: dict = Depends(get_current_user)) -> dict[str, Any]:
    """Log in to IMAP and SMTP with these settings — nothing is saved, sent or changed."""
    db = get_db()
    uid = _g._effective_owner(_user)
    stored_pw, stored_smtp_pw = await _stored_passwords(db, uid, body.account_id)
    base: dict[str, Any] = {}
    if body.account_id:
        base = next((r for r in await _load(db, uid) if r["id"] == str(body.account_id)), {})
    acct = _test_account(_settings_of(body, base), _text(body.password) or stored_pw,
                         _text(body.smtp_password) or stored_smtp_pw, "test")
    return await _run(acct, imap_mail.test_account, acct)


def _check_result(result: dict[str, Any]) -> None:
    if not result["imap"]["ok"]:
        raise HTTPException(status_code=400, detail=f"IMAP: {result['imap']['error']}")
    if not result["smtp"]["ok"]:
        raise HTTPException(status_code=400, detail=(
            f"SMTP: {result['smtp']['error']} (to connect it receive-only, leave the SMTP server empty)"))


@router.post("/accounts")
async def imap_account_add(body: AccountBody, _user: dict = Depends(get_current_user)) -> dict[str, Any]:
    """Connect another inbox — logged in to first, so only working settings are kept."""
    _require_crypto()
    db = get_db()
    uid = _g._effective_owner(_user)
    password = body.password if isinstance(body.password, str) else ""
    smtp_password = body.smtp_password if isinstance(body.smtp_password, str) else ""
    settings = _settings_of(body)
    acct = _test_account(settings, password, smtp_password, "new")
    result = await _run(acct, imap_mail.test_account, acct)
    # A typed port this deck can't reach, with the other standard one working
    # (a host blocking outgoing 465 …): keep the one that works, and say so.
    acct, adjusted = imap_mail.apply_alternatives(acct, result)
    _check_result(result)
    async with _lock(_locks, uid):
        recs = await _load(db, uid)
        if len(recs) >= MAX_ACCOUNTS:
            raise HTTPException(status_code=400, detail=f"At most {MAX_ACCOUNTS} inboxes per user.")
        if any(r.get("email", "").lower() == acct.email.lower() and r.get("username") == acct.username
               and r.get("imap_host") == acct.imap_host for r in recs):
            raise HTTPException(status_code=409, detail=f"{acct.email} is already connected.")
        new_id = _new_id(acct.label or acct.email.split("@", 1)[0], {r["id"] for r in recs})
        rec: dict[str, Any] = {
            "id": new_id, "label": acct.label, "email": acct.email, "username": acct.username,
            "display_name": acct.display_name, "imap_host": acct.imap_host, "imap_port": acct.imap_port,
            "imap_security": acct.imap_security, "smtp_host": acct.smtp_host, "smtp_port": acct.smtp_port,
            "smtp_security": acct.smtp_security, "smtp_username": acct.smtp_username,
            "save_sent": acct.save_sent, "password": fd_crypto.seal(acct.password),
            "smtp_password": fd_crypto.seal(acct.smtp_password) if acct.smtp_username else "",
            "agents": _check_agents(body.agents) if body.agents is not None else [],
            "sending": _check_sending(body.sending, _default_sending()) if body.sending is not None
            else _default_sending(),
            "signature": "", "provider": _text(body.provider)[:60],
            "created_at": _now(), "updated_at": _now(),
            "last_check": {"ok": True, "at": _now(), "roles": result["imap"].get("roles", [])},
        }
        if body.signature is not None:
            try:
                rec["signature"] = mail_signature.normalize(body.signature)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
        recs.append(rec)
        await _save(db, uid, recs)
    log.info("IMAP inbox connected", account=new_id, host=acct.imap_host)
    return {"account": await _view(db, uid, rec), "test": result, "adjusted": adjusted}


@router.put("/accounts/{account_id}")
async def imap_account_update(account_id: str, body: AccountBody,
                              _user: dict = Depends(get_current_user)) -> dict[str, Any]:
    """Change an inbox: any subset of the fields. New connection settings or a
    new password are logged in to first; a blank password keeps the stored one."""
    _require_crypto()
    db = get_db()
    uid = _g._effective_owner(_user)
    async with _lock(_locks, uid):
        recs = await _load(db, uid)
        rec = next((r for r in recs if r["id"] == account_id), None)
        if rec is None:
            raise HTTPException(status_code=404, detail="No such inbox")
        new_pw = body.password if isinstance(body.password, str) and body.password else ""
        new_smtp_pw = body.smtp_password if isinstance(body.smtp_password, str) and body.smtp_password else ""
        settings = _settings_of(body, {k: rec.get(k) for k in _PLAIN_FIELDS})
        changed = bool(new_pw or new_smtp_pw) or any(
            getattr(body, k) is not None and str(getattr(body, k)).strip() != str(rec.get(k) or "").strip()
            for k in _CONNECTION_FIELDS)
        result = None
        adjusted: list[str] = []
        if changed or any(getattr(body, k) is not None for k in ("label", "display_name", "save_sent")):
            old: Account | None = None
            try:
                old = _account(rec)
            except MailError:
                if not new_pw:
                    raise
            keep_smtp_pw = new_smtp_pw or (old.smtp_password if old else "")
            if not _text(settings.get("smtp_username")):
                keep_smtp_pw = ""
            acct = _test_account(settings, new_pw or (old.password if old else ""), keep_smtp_pw, account_id)
            if changed:
                result = await _run(acct, imap_mail.test_account, acct)
                acct, adjusted = imap_mail.apply_alternatives(acct, result)
                _check_result(result)
                rec["last_check"] = {"ok": True, "at": _now(), "roles": result["imap"].get("roles", [])}
            rec.update(
                label=acct.label, email=acct.email, username=acct.username, display_name=acct.display_name,
                imap_host=acct.imap_host, imap_port=acct.imap_port, imap_security=acct.imap_security,
                smtp_host=acct.smtp_host, smtp_port=acct.smtp_port, smtp_security=acct.smtp_security,
                smtp_username=acct.smtp_username, save_sent=acct.save_sent,
                password=fd_crypto.seal(acct.password),
                smtp_password=fd_crypto.seal(acct.smtp_password) if acct.smtp_username else "",
            )
        if body.agents is not None:
            rec["agents"] = _check_agents(body.agents)
        if body.sending is not None:
            rec["sending"] = _check_sending(body.sending, rec["sending"])
        if body.signature is not None:
            try:
                rec["signature"] = mail_signature.normalize(body.signature)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
        rec["updated_at"] = _now()
        await _save(db, uid, recs)
    out: dict[str, Any] = {"account": await _view(db, uid, rec)}
    if result is not None:
        out["test"] = result
        out["adjusted"] = adjusted
    return out


@router.delete("/accounts/{account_id}")
async def imap_account_delete(account_id: str, _user: dict = Depends(get_current_user)) -> dict[str, Any]:
    """Forget an inbox and its password (its send history stays)."""
    db = get_db()
    uid = _g._effective_owner(_user)
    async with _lock(_locks, uid):
        recs = await _load(db, uid)
        left = [r for r in recs if r["id"] != account_id]
        if len(left) == len(recs):
            raise HTTPException(status_code=404, detail="No such inbox")
        await _save(db, uid, left)
    return {"deleted": account_id}


@router.get("/sends")
async def imap_sends(account_id: str = "", limit: int = 20,
                     _user: dict = Depends(get_current_user)) -> dict[str, Any]:
    db = get_db()
    uid = _g._effective_owner(_user)
    rows = await db.list_imap_sends(uid, max(1, min(int(limit), 100)), account_id or None)
    return {"sends": [
        {"id": r["id"], "status": r.get("status") or "sent", "account_id": r["account_id"],
         "account_email": r["account_email"], "agent": r["agent"], "to": r["to_addrs"],
         "cc": r["cc_addrs"], "bcc": r["bcc_addrs"], "subject": r["subject"],
         "message_id": r["message_id"], "draft_id": r["draft_id"],
         "attachments": _names(r.get("attachments")), "created_at": r["created_at"]}
        for r in rows
    ]}


def _names(raw: Any) -> list[str]:
    try:
        data = json.loads(raw) if raw else []
        return [str(x) for x in data] if isinstance(data, list) else []
    except Exception:
        return []


# ── agent routes ────────────────────────────────────────────────────


def _agent_slug(request: Request) -> str:
    try:
        from captain_claw.flight_deck.server import _find_agent_by_auth

        matched, _owner, slug = _find_agent_by_auth(request.headers.get("X-Agent-Auth", ""))
    except Exception:
        return ""
    return str(slug) if matched and slug else ""


async def _agent_records(request: Request) -> tuple[str, str, list[dict[str, Any]]]:
    """``(owner, agent slug, the owner's inboxes this agent may use)``."""
    _g._authorize_agent_call(request)
    owner = await _g._agent_owner(request, service="email (IMAP)")
    slug = _agent_slug(request)
    recs = await _load(get_db(), owner)
    usable = [r for r in recs if not r.get("agents") or slug in r["agents"]]
    return owner, slug, usable


def _brief(rec: dict[str, Any]) -> dict[str, Any]:
    return {"id": rec["id"], "label": rec.get("label", ""), "email": rec.get("email", ""),
            "can_send": bool(rec.get("smtp_host")),
            "sending_enabled": bool(rec.get("smtp_host")) and rec["sending"]["enabled"]
            and not deck_send_disabled(),
            "allowed_recipients": list(rec["sending"]["allowed_recipients"]),
            "daily_limit": rec["sending"]["daily_limit"]}


@router.get("/agent_status")
async def imap_agent_status(request: Request) -> dict[str, Any]:
    """The calling agent's inboxes (no network): what the imap_mail tool shows."""
    _owner, _slug, recs = await _agent_records(request)
    return {"connected": bool(recs), "accounts": [_brief(r) for r in recs]}


# Actions that read one inbox, many with account=all.
_MULTI = ("list_messages", "search", "list_drafts", "list_folders")
_ID_ACTIONS = {"read_message": "message_id", "get_attachment": "message_id", "update_draft": "draft_id",
               "send_draft": "draft_id", "get_thread": "thread_id"}


def _pick(recs: list[dict[str, Any]], spec: str) -> dict[str, Any] | None:
    s = spec.strip().casefold()
    for key in ("id", "email", "label"):
        hit = [r for r in recs if str(r.get(key) or "").casefold() == s]
        if len(hit) == 1:
            return hit[0]
    return None


def _which(recs: list[dict[str, Any]]) -> str:
    return "; ".join(f"{r['id']} ({r.get('email')}{', ' + r['label'] if r.get('label') else ''})" for r in recs)


def _choose(action: str, body: dict[str, Any], recs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The inbox(es) this call acts on."""
    if not recs:
        raise HTTPException(status_code=404, detail=(
            f"No email inbox is connected for this agent. The user connects one in {_WHERE}."))
    spec = _text(body.get("account"))
    from_id = ""
    for key in ("message_id", "draft_id", "thread_id", "reply_to_message_id"):
        if body.get(key):
            from_id = imap_mail.account_of(body[key])
            break
    for key in ("message_ids", "thread_ids"):
        vals = body.get(key)
        if not from_id and isinstance(vals, list) and vals:
            heads = {imap_mail.account_of(v) for v in vals}
            if len(heads) > 1:
                raise HTTPException(status_code=400, detail=(
                    "These IDs belong to different inboxes — call once per inbox. Nothing was changed."))
            from_id = heads.pop()
    if from_id:
        rec = next((r for r in recs if r["id"] == from_id), None)
        if rec is None:
            raise HTTPException(status_code=404, detail=(
                f"That ID belongs to inbox {from_id!r}, which this agent can't use (connected: "
                f"{_which(recs)})."))
        if spec and spec.casefold() not in ("all", rec["id"], str(rec.get("email", "")).casefold(),
                                            str(rec.get("label", "")).casefold()):
            raise HTTPException(status_code=400, detail=(
                f"account={spec!r} doesn't match the inbox the ID belongs to ({rec['id']})."))
        return [rec]
    if spec.casefold() in ("all", "*", "every"):
        if action not in _MULTI:
            raise HTTPException(status_code=400, detail=f"{action} works on one inbox — pass account=<id>.")
        return recs
    if spec:
        rec = _pick(recs, spec)
        if rec is None:
            raise HTTPException(status_code=404, detail=f"No inbox {spec!r}. Inboxes: {_which(recs)}")
        return [rec]
    if len(recs) == 1:
        return recs
    if action in _MULTI:
        return recs
    raise HTTPException(status_code=400, detail=(
        f"Which inbox? This user has {len(recs)}: {_which(recs)}. Pass account=<id> "
        "(for a reply, the message's ID already says which). Nothing was changed."))


def _int(v: Any, default: int, maximum: int) -> int:
    try:
        n = int(float(v)) if v not in (None, "") else default
    except (TypeError, ValueError):
        n = default
    return max(1, min(n, maximum))


def _flag(v: Any) -> bool:
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "on")
    return bool(v)


def _str_list(v: Any) -> list[str]:
    if v is None or v == "":
        return []
    if isinstance(v, str):
        v = [p for p in re.split(r"[\s,]+", v) if p]
    if not isinstance(v, list):
        return []
    out: list[str] = []
    for x in v:
        x = str(x).strip().strip("[]'\"")
        if x and x not in out:
            out.append(x)
    return out


def _recips(v: Any) -> str:
    if isinstance(v, (list, tuple)):
        return ", ".join(str(x) for x in v if x)
    return _text(v)


@router.post("/agent/{action}")
async def imap_agent_call(action: str, request: Request) -> dict[str, Any]:
    owner, slug, recs = await _agent_records(request)
    try:
        body = await request.json()
    except Exception:
        body = {}
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="The request body must be a JSON object")
    if action == "list_accounts":
        return {"accounts": [_brief(r) for r in recs]}
    handler = _HANDLERS.get(action)
    if handler is None:
        raise HTTPException(status_code=404, detail=f"Unknown action {action!r}")
    chosen = _choose(action, body, recs)
    try:
        if action in _MULTI:
            return await _many(action, chosen, body, handler, owner)
        rec = chosen[0]
        return await handler(owner, slug, rec, _account(rec), body)
    except MailError as exc:
        raise _http(exc) from exc


async def _many(action: str, chosen: list[dict[str, Any]], body: dict[str, Any],
                handler: Callable[..., Awaitable[dict[str, Any]]], owner: str) -> dict[str, Any]:
    async def one(rec: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any] | None, str]:
        try:
            return rec, await handler(owner, "", rec, _account(rec), body), ""
        except MailError as exc:
            return rec, None, exc.message
        except HTTPException as exc:
            return rec, None, str(exc.detail)

    results = await asyncio.gather(*(one(r) for r in chosen))
    if len(chosen) == 1:
        rec, res, err = results[0]
        if res is None:
            raise HTTPException(status_code=502, detail=err)
        return {**res, "accounts": [rec["id"]]}
    merged: dict[str, Any] = {"accounts": [r["id"] for r in chosen], "errors": []}
    want = _int(body.get("max_results"), 10, imap_mail.MAX_RESULTS)
    if action == "list_folders":
        merged["folders"] = []
        for rec, res, err in results:
            if res is None:
                merged["errors"].append({"account": rec["id"], "error": err})
            else:
                merged["folders"] += [{**f, "account": rec["id"]} for f in res["folders"]]
        return merged
    msgs: list[dict[str, Any]] = []
    total = 0
    for rec, res, err in results:
        if res is None:
            merged["errors"].append({"account": rec["id"], "error": err})
            continue
        total += res.get("total", 0)
        msgs += res.get("messages", [])
    msgs.sort(key=lambda m: m.get("ts", 0), reverse=True)
    merged.update(folder="every inbox", total=total, messages=msgs[:want])
    return merged


# ── read handlers ───────────────────────────────────────────────────


async def _h_list_folders(owner, slug, rec, acct, body):
    return {"folders": await _run(acct, imap_mail.list_folders, acct)}


async def _h_list_messages(owner, slug, rec, acct, body):
    return await _run(acct, imap_mail.list_messages, acct, folder=_text(body.get("folder")),
                      unread_only=_flag(body.get("unread_only")),
                      max_results=_int(body.get("max_results"), 10, imap_mail.MAX_RESULTS),
                      include_body=_flag(body.get("include_body")))


async def _h_search(owner, slug, rec, acct, body):
    query = _text(body.get("query"))
    if not query:
        raise MailError("search needs a query (from:ana subject:report since:2026-01-01 …).", 400)
    return await _run(acct, imap_mail.search, acct, query=query, folder=_text(body.get("folder")),
                      max_results=_int(body.get("max_results"), 10, imap_mail.MAX_RESULTS),
                      include_body=_flag(body.get("include_body")))


async def _h_list_drafts(owner, slug, rec, acct, body):
    return await _run(acct, imap_mail.list_drafts, acct, query=_text(body.get("query")),
                      max_results=_int(body.get("max_results"), 10, imap_mail.MAX_RESULTS))


async def _h_read_message(owner, slug, rec, acct, body):
    ref = imap_mail.parse_ref(body.get("message_id"))
    return {"message": await _run(acct, imap_mail.read_message, acct, ref)}


async def _h_get_thread(owner, slug, rec, acct, body):
    _acct, root = imap_mail.parse_thread_id(body.get("thread_id"))
    return await _run(acct, imap_mail.get_thread, acct, root)


async def _h_get_attachment(owner, slug, rec, acct, body):
    ref = imap_mail.parse_ref(body.get("message_id"))
    files = await _run(acct, imap_mail.get_attachments, acct, ref, _text(body.get("attachment")))
    total = sum(len(f.data) for f in files)
    if total > imap_mail.MAX_MESSAGE_BYTES:
        raise MailError("These attachments are too big to fetch together — ask for one at a time.", 413)
    return {"files": mail_attachments.to_wire(files)}


# ── compose / draft / send ──────────────────────────────────────────


def _payload_atts(body: dict[str, Any]) -> list[mail_attachments.Attachment]:
    try:
        return mail_attachments.from_wire(body.get("attachments"), max_total_bytes=MAX_TOTAL_ATTACHMENT_BYTES)
    except mail_attachments.AttachmentError as exc:
        raise MailError(f"{exc} Nothing was sent or saved.", 400) from None


def _check_addresses(to: str, cc: str, bcc: str, *, allowlist: list[str] | None, what: str) -> None:
    bad = gmail_compose.invalid_recipients(to, cc, bcc)
    if bad:
        raise MailError(gmail_compose.invalid_recipients_message(bad).replace("Nothing was sent.", what), 400)
    addrs = gmail_compose.parse_addresses(to, cc, bcc)
    if len(addrs) > gmail_compose.MAX_RECIPIENTS:
        raise MailError(f"Too many recipients ({len(addrs)}) — at most {gmail_compose.MAX_RECIPIENTS} "
                        f"per email. {what}", 400)
    if allowlist is not None:
        if not addrs:
            raise MailError(f"No recipient — set to (or cc / bcc). {what}", 400)
        off = gmail_compose.recipients_allowed(addrs, allowlist)
        if off:
            raise MailError(
                f"Not on this inbox's allowed-recipients list: {', '.join(off)}. {what} Create a draft "
                f"with imap_mail action=create_draft instead; the user manages the list in {_WHERE}.", 403)


class _Composed:
    def __init__(self, data: bytes, message_id: str, to: str, cc: str, bcc: str, subject: str,
                 body: str, html: str, reply: imap_mail.ReplyContext | None,
                 atts: list[mail_attachments.Attachment]) -> None:
        self.data, self.message_id = data, message_id
        self.to, self.cc, self.bcc, self.subject = to, cc, bcc, subject
        self.body, self.html, self.reply, self.atts = body, html, reply, atts


def _compose_work(acct: Account, body: dict[str, Any], signature: str, atts: list[mail_attachments.Attachment],
                  *, for_send: bool, allowlist: list[str] | None, save: bool) -> tuple[_Composed, str]:
    """In a worker thread: the reply's defaults, checks, signature, build,
    the repeat check; for a draft (*save*) the APPEND to Drafts. Returns the
    composed email and the new draft's id ("" for a send)."""
    what = "Nothing was sent." if for_send else "No draft was created."
    to, cc, bcc = _recips(body.get("to")), _recips(body.get("cc")), _recips(body.get("bcc"))
    subject, text, html = _text(body.get("subject")), str(body.get("body") or ""), str(body.get("html_body") or "")
    reply_to = _text(body.get("reply_to_message_id"))
    with imap_mail.Session(acct) as s:
        ctx = None
        if reply_to:
            ctx = imap_mail.reply_context(s, imap_mail.parse_ref(reply_to, "reply_to_message_id"),
                                          reply_all=_flag(body.get("reply_all")))
            to = to or ctx.to_default
            cc = cc or ctx.cc_all
            subject = subject or ctx.subject_default
        if not (text.strip() or html.strip()):
            raise MailError(f"A body is required (body or html_body). {what}", 400)
        if for_send and not subject:
            raise MailError(f"A send needs a subject. {what}", 400)
        if not gmail_compose.parse_addresses(to, cc, bcc):
            raise MailError(f"No recipient — set to (or cc / bcc). {what}", 400)
        _check_addresses(to, cc, bcc, allowlist=allowlist, what=what)
        to, cc, bcc = (gmail_compose.format_recipients(f) for f in (to, cc, bcc))
        text, html = mail_signature.apply(text, html, signature)
        try:
            data, mid = imap_mail.build_message(
                acct, to=to, cc=cc, bcc=bcc, subject=subject, body=text, html_body=html,
                in_reply_to=ctx.message_id if ctx else "", references=ctx.references if ctx else "",
                attachments=atts or None)
        except ValueError as exc:
            raise MailError(f"Could not build the email: {exc}. {what}", 400) from None
        if len(data) > mail_attachments.GMAIL_MAX_MESSAGE_BYTES:
            raise MailError(f"This email would be {mail_attachments.human_size(len(data))} once encoded — "
                            f"at most 25 MB. {what}", 400)
        if not _flag(body.get("allow_repeat")):
            matches = imap_mail.find_repeats(
                s, to=to, cc=cc, bcc=bcc, subject=subject,
                reply_msgid=ctx.message_id if ctx else "", days=repeat_check_days())
            if matches:
                raise MailError(imap_mail.repeat_refusal(
                    matches, lead="Not sent" if for_send else "Not created", reply=bool(ctx)), 409)
        draft_id = imap_mail.save_draft(s, data, mid) if save else ""
    return _Composed(data, mid, to, cc, bcc, subject, text, html, ctx, atts), draft_id


async def _h_create_draft(owner, slug, rec, acct, body):
    atts = _payload_atts(body)
    composed, draft_id = await _run(acct, _compose_work, acct, body, rec.get("signature") or "", atts,
                                    for_send=False, allowlist=None, save=True)
    return {"draft_id": draft_id, "to": composed.to, "cc": composed.cc, "bcc": composed.bcc,
            "subject": composed.subject, "reply": bool(composed.reply),
            "attachments": [{"filename": a.filename, "size": len(a.data)} for a in atts],
            "account": rec["id"]}


def _rewrite_draft(acct: Account, body: dict[str, Any], signature: str,
                   new_atts: list[mail_attachments.Attachment]) -> dict[str, Any]:
    ref = imap_mail.parse_ref(body.get("draft_id"), "draft_id")
    remove = _str_list(body.get("remove_attachments"))
    with imap_mail.Session(acct) as s:
        folder = s.open_ref(ref)
        if folder.role != "drafts":
            raise MailError(f"{ref} isn't in the Drafts folder — update_draft changes drafts only. "
                            "Nothing was changed.", 400)
        raw, _meta = s.fetch_raw(ref.uid)
        old = mail_attachments.parse_message(raw)
        hdr = email_headers(raw)
        files = mail_attachments.message_files(old)
        keep: list[mail_attachments.Attachment] = []
        dropped: list[str] = []
        for i, fp in enumerate(files, 1):
            if "all" in [r.casefold() for r in remove] or str(i) in remove or fp.name in remove:
                dropped.append(fp.name)
                continue
            keep.append(mail_attachments.Attachment(fp.name, fp.part.get_content_type(),
                                                    fp.part.get_payload(decode=True) or b""))
        new_names = {a.filename for a in new_atts}
        atts = [a for a in keep if a.filename not in new_names] + new_atts
        to = _recips(body.get("to")) if body.get("to") is not None else hdr["to"]
        cc = _recips(body.get("cc")) if body.get("cc") is not None else hdr["cc"]
        bcc = _recips(body.get("bcc")) if body.get("bcc") is not None else hdr["bcc"]
        subject = _text(body.get("subject")) if body.get("subject") is not None else hdr["subject"]
        text, html = str(body.get("body") or ""), str(body.get("html_body") or "")
        if not (text.strip() or html.strip()):
            text, _ = imap_mail.body_text(old)
        _check_addresses(to, cc, bcc, allowlist=None, what="Nothing was changed.")
        to, cc, bcc = (gmail_compose.format_recipients(f) for f in (to, cc, bcc))
        text, html = mail_signature.apply(text, html, signature)
        data, mid = imap_mail.build_message(
            acct, to=to, cc=cc, bcc=bcc, subject=subject, body=text, html_body=html,
            in_reply_to=hdr["in_reply_to"], references=hdr["references"], attachments=atts or None)
        new_id = imap_mail.save_draft(s, data, mid)
        note = ""
        try:
            imap_mail.delete_message(s, ref)
        except MailError as exc:
            note = f"the old version is still in Drafts ({exc.message})"
    return {"draft_id": new_id, "old_draft_id": str(ref), "to": to, "cc": cc, "bcc": bcc,
            "subject": subject, "removed": dropped, "note": note,
            "attachments": [{"filename": a.filename, "size": len(a.data)} for a in atts]}


def email_headers(raw: bytes) -> dict[str, str]:
    import email as _email

    h = _email.message_from_bytes(raw)
    d = imap_mail.decode_header
    return {"to": d(h.get("To")), "cc": d(h.get("Cc")), "bcc": d(h.get("Bcc")), "subject": d(h.get("Subject")),
            "in_reply_to": d(h.get("In-Reply-To")), "references": d(h.get("References")),
            "from": d(h.get("From")), "message_id": d(h.get("Message-ID"))}


async def _h_update_draft(owner, slug, rec, acct, body):
    atts = _payload_atts(body)
    out = await _run(acct, _rewrite_draft, acct, body, rec.get("signature") or "", atts)
    out["account"] = rec["id"]
    return out


def _agent_name(slug: str) -> str:
    return slug or "An agent"


async def _audit(db: FlightDeckDB, owner: str, rec: dict[str, Any], *, status: str, agent: str,
                 c: _Composed, draft_id: str, chash: str) -> None:
    try:
        await db.add_imap_send(
            owner, account_id=rec["id"], account_email=rec.get("email", ""), agent=agent,
            to_addrs=c.to, cc_addrs=c.cc, bcc_addrs=c.bcc, subject=c.subject, message_id=c.message_id,
            in_reply_to=c.reply.message_id if c.reply else "", draft_id=draft_id, content_hash=chash,
            status=status,
            attachments=json.dumps([a.filename for a in c.atts], ensure_ascii=False) if c.atts else "")
    except Exception as exc:
        log.error("IMAP send audit insert failed (%s): %s", status, type(exc).__name__)


async def _bell(db: FlightDeckDB, owner: str, rec: dict[str, Any], *, status: str, agent: str,
                c: _Composed) -> None:
    files = f" - Attached: {', '.join(a.filename for a in c.atts)}" if c.atts else ""
    who = rec.get("email", "")
    if status == "unknown":
        title = f"{agent} may have sent an email from {who}"
        body = f"The mail server never confirmed it — check Sent. To: {c.to} - Subject: {c.subject}{files}"
    else:
        title = f"{agent} sent an email from {who}"
        body = f"To: {c.to} - Subject: {c.subject or '(no subject)'}{files}"
    try:
        await db.add_notification(owner, "email_sent", title[:120], body[:300], "", "")
    except Exception as exc:
        log.warning("IMAP send notification failed: %s", type(exc).__name__)


def _send_refusal(rec: dict[str, Any]) -> str | None:
    if not rec.get("smtp_host"):
        return (f"{rec.get('email')} is connected receive-only (no SMTP server), so it can't send. "
                "Nothing was sent. Create a draft with action=create_draft instead and tell the user.")
    if deck_send_disabled():
        return ("Email sending is disabled on this Flight Deck by its administrator (FD_IMAP_SEND), "
                "so nothing was sent. Create a draft with action=create_draft instead.")
    if not rec["sending"]["enabled"]:
        return (f"Sending is turned off for {rec.get('email')}, so nothing was sent. Create a draft "
                "with imap_mail action=create_draft instead, and tell the user they can turn sending "
                f"on in {_WHERE}.")
    return None


async def _deliver(owner: str, slug: str, rec: dict[str, Any], acct: Account, c: _Composed, *,
                   draft_ref: imap_mail.Ref | None) -> dict[str, Any]:
    """Under the inbox's send lock: limit, duplicate check, SMTP, audit; then
    the Sent copy / \\Answered / draft removal and the bell."""
    db = get_db()
    agent = _agent_name(slug)
    limit = int(rec["sending"]["daily_limit"])
    async with _lock(_send_locks, f"{owner}|{rec['id']}"):
        now = datetime.now(UTC)
        used = await db.count_imap_sends_since(owner, rec["id"], (now - _DAY).isoformat())
        if used >= limit:
            raise MailError(
                f"Daily send limit reached for {rec.get('email')}: {used} of {limit} emails in the last "
                f"24 hours. Nothing was sent. Create a draft instead, or ask the user to raise the limit "
                f"in {_WHERE}.", 429)
        chash = gmail_compose.content_hash(
            c.to, c.cc, c.bcc, c.subject, c.body, c.html,
            reply_to=c.reply.message_id if c.reply else "",
            attachments_digest=mail_attachments.digest(c.atts) if c.atts else "")
        dup = await db.find_imap_send_by_hash(owner, rec["id"], chash, (now - _DUPLICATE_WINDOW).isoformat())
        if dup:
            more = (" and the mail server never confirmed it — it may have gone out. Check Sent."
                    if dup.get("status") == "unknown" else ".")
            raise MailError(f"Duplicate: this exact email was already sent at {dup['created_at']}{more} "
                            "Not sent again.", 409)
        rcpts = gmail_compose.parse_addresses(c.to, c.cc, c.bcc)
        try:
            server_answer = await _run(acct, imap_mail.smtp_deliver, acct, rcpts, imap_mail.without_bcc(c.data))
        except SendOutcomeUnknown:
            await _audit(db, owner, rec, status="unknown", agent=agent, c=c,
                         draft_id=str(draft_ref or ""), chash=chash)
            await _bell(db, owner, rec, status="unknown", agent=agent, c=c)
            raise
        await _audit(db, owner, rec, status="sent", agent=agent, c=c, draft_id=str(draft_ref or ""), chash=chash)
    notes = await _run(acct, imap_mail.after_send, acct, c.data,
                       answered=c.reply.ref if c.reply else None, draft=draft_ref)
    await _bell(db, owner, rec, status="sent", agent=agent, c=c)
    return {"ok": True, "account": rec["id"], "from": acct.email, "to": c.to, "cc": c.cc, "bcc": c.bcc,
            "subject": c.subject, "message_id": c.message_id, "server": server_answer,
            "attachments": [{"filename": a.filename, "size": len(a.data)} for a in c.atts],
            "sent_last_24h": used + 1, "daily_limit": limit, "notes": notes}


async def _h_send(owner, slug, rec, acct, body):
    refusal = _send_refusal(rec)
    if refusal:
        raise MailError(refusal, 403)
    atts = _payload_atts(body)
    composed, _ = await _run(acct, _compose_work, acct, body, rec.get("signature") or "", atts,
                             for_send=True, allowlist=rec["sending"]["allowed_recipients"], save=False)
    return await _deliver(owner, slug, rec, acct, composed, draft_ref=None)


def _load_draft_for_send(acct: Account, draft_id: Any) -> tuple[_Composed, imap_mail.Ref]:
    import email.utils as eu

    ref = imap_mail.parse_ref(draft_id, "draft_id")
    with imap_mail.Session(acct) as s:
        folder = s.open_ref(ref)
        if folder.role != "drafts":
            raise MailError(f"{ref} isn't in the Drafts folder — send_draft sends drafts only. "
                            "Nothing was sent.", 400)
        raw, _meta = s.fetch_raw(ref.uid)
    msg = mail_attachments.parse_message(raw)
    hdr = email_headers(raw)
    # Sent from this inbox, dated now, with a Message-ID — whatever the draft said.
    for name in ("From", "Sender", "Date"):
        del msg[name]
    msg["From"] = acct.from_header()
    msg["Date"] = eu.formatdate(localtime=True)
    mid = hdr["message_id"]
    if not imap_mail.msgids(mid):
        del msg["Message-ID"]
        mid = eu.make_msgid(domain=acct.email.rsplit("@", 1)[-1])
        msg["Message-ID"] = mid
    text, _ = imap_mail.body_text(msg)
    files = mail_attachments.message_files(msg)
    atts = [mail_attachments.Attachment(f.name, f.part.get_content_type(), f.part.get_payload(decode=True) or b"")
            for f in files]
    reply = imap_mail.ReplyContext(message_id=(imap_mail.msgids(hdr["in_reply_to"]) or [""])[0])
    c = _Composed(msg.as_bytes(), mid, hdr["to"], hdr["cc"], hdr["bcc"], hdr["subject"], text, "",
                  reply if reply.message_id else None, atts)
    return c, ref


async def _h_send_draft(owner, slug, rec, acct, body):
    refusal = _send_refusal(rec)
    if refusal:
        raise MailError(refusal, 403)
    composed, ref = await _run(acct, _load_draft_for_send, acct, body.get("draft_id"))
    if not composed.subject.strip():
        raise MailError("The draft has no subject — add one with update_draft first. Nothing was sent.", 400)
    _check_addresses(composed.to, composed.cc, composed.bcc,
                     allowlist=rec["sending"]["allowed_recipients"], what="Nothing was sent.")
    return await _deliver(owner, slug, rec, acct, composed, draft_ref=ref)


# ── organize ────────────────────────────────────────────────────────


def _targets(body: dict[str, Any], acct: Account) -> tuple[list[imap_mail.Ref], list[str]]:
    refs = [imap_mail.parse_ref(x) for x in _str_list(body.get("message_ids")) + _str_list(body.get("message_id"))]
    roots = []
    for t in _str_list(body.get("thread_ids")) + _str_list(body.get("thread_id")):
        _a, root = imap_mail.parse_thread_id(t)
        roots.append(root)
    if not refs and not roots:
        raise MailError("Which emails? Pass message_ids and/or thread_ids (find them with list_messages "
                        "or search first). Nothing was changed.", 400)
    if len(refs) + len(roots) > imap_mail.MAX_REFS:
        raise MailError(f"At most {imap_mail.MAX_REFS} emails per call — call again for the rest. "
                        "Nothing was changed.", 400)
    for r in refs:
        if r.account != acct.id:
            raise MailError("These IDs belong to different inboxes — call once per inbox. Nothing was changed.", 400)
    return refs, roots


async def _h_archive(owner, slug, rec, acct, body):
    refs, roots = _targets(body, acct)
    return await _run(acct, imap_mail.move, acct, refs, roots, "", archive=True)


async def _h_move(owner, slug, rec, acct, body):
    dest = _text(body.get("folder"))
    if not dest:
        raise MailError("move needs folder=<destination>. Nothing was changed.", 400)
    refs, roots = _targets(body, acct)
    return await _run(acct, imap_mail.move, acct, refs, roots, dest)


async def _h_mark_read(owner, slug, rec, acct, body):
    refs, roots = _targets(body, acct)
    return await _run(acct, imap_mail.set_seen, acct, refs, roots, True)


async def _h_mark_unread(owner, slug, rec, acct, body):
    refs, roots = _targets(body, acct)
    return await _run(acct, imap_mail.set_seen, acct, refs, roots, False)


async def _h_create_folder(owner, slug, rec, acct, body):
    name = _text(body.get("folder"))
    if not name:
        raise MailError("create_folder needs folder=<name> ('Receipts' or 'Clients/Acme').", 400)
    return await _run(acct, imap_mail.create_folder, acct, name)


_HANDLERS: dict[str, Callable[..., Awaitable[dict[str, Any]]]] = {
    "list_folders": _h_list_folders, "list_messages": _h_list_messages, "search": _h_search,
    "list_drafts": _h_list_drafts, "read_message": _h_read_message, "get_thread": _h_get_thread,
    "get_attachment": _h_get_attachment, "create_draft": _h_create_draft,
    "update_draft": _h_update_draft, "send": _h_send, "send_draft": _h_send_draft,
    "archive": _h_archive, "move": _h_move, "mark_read": _h_mark_read, "mark_unread": _h_mark_unread,
    "create_folder": _h_create_folder,
}
