"""Purchases — Captain buys from an online shop (Payments Phase 3,
docs/payments-phase3-plan.md).

Every purchase needs the user's approval of *this* seller and *this* amount
before any credential exists:

* ``link`` — Stripe Link Agent Wallet: the user approves the spend request in
  Link; Link issues a one-time virtual card (or a Link Pay Token).
* ``revolut_card`` — the owner approves in Flight Deck; Flight Deck makes a
  Revolut Business company card good for this purchase only (one transaction
  up to the amount, the amount in all, terminated after use or by tomorrow).
* ``acp`` / ``ap2`` — checkout protocols (see protocol_checkout.py): the
  owner approves the merchant's cart in Flight Deck.

The card leaves Flight Deck once, to the agent's ``checkout_pay`` step, which
types it into the shop's form and clicks pay in one go — the model never sees
it (payments/card_fill.py). The ledger moves on the provider's word (Link's
spend-request status, Revolut's card transaction), not the model's.
"""

from __future__ import annotations

import datetime as dt
import re
import secrets
import time
import urllib.parse

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from captain_claw.flight_deck.auth import get_current_user
from captain_claw.flight_deck.payments_store import PURCHASE_DONE, PURCHASE_TERMINAL
from captain_claw.logging import get_logger

log = get_logger(__name__)

router = APIRouter(tags=["payments-purchases"])

SOURCES = ("link", "revolut_card", "acp", "ap2")
LINK = "link"
_ID_RE = re.compile(r"^pur_[0-9a-f]{12}$")
MAX_FILL_ATTEMPTS = 3
PURCHASES_PER_HOUR = 20


def _pr():
    from captain_claw.flight_deck import payments_routes

    return payments_routes


def store():
    return _pr().store()


def _domain(url: str) -> str:
    return (urllib.parse.urlsplit(url).hostname or "").lower()


def view(row: dict) -> dict:
    from captain_claw.payments import checks

    ps = row.get("provider_state") or {}
    return {
        "id": row["id"], "status": row["status"], "source": row["source"],
        "merchant_name": row.get("merchant_name", ""), "merchant_url": row.get("merchant_url", ""),
        "merchant_domain": _domain(row.get("merchant_url", "")), "description": row.get("description", ""),
        "items": row.get("items") or [], "amount_cents": int(row.get("amount_cents") or 0),
        "currency": row.get("currency", "EUR"),
        "amount_text": checks.format_amount(int(row.get("amount_cents") or 0), row.get("currency", "EUR")),
        "approval_url": row.get("approval_url", ""), "card_last4": row.get("card_last4", ""),
        "order_ref": row.get("order_ref", ""), "result": row.get("result") or {},
        "warnings": row.get("warnings") or [], "automated": bool(row.get("automated")),
        "next_action": ps.get("next_action") or {}, "credential_type": ps.get("credential_type", ""),
        "test": bool(ps.get("test")), "created_at": row.get("created_at"), "completed_at": row.get("completed_at"),
        "review_path": f"/payments/purchases/{row['id']}",
    }


async def _owned(purchase_id: str, owner: str) -> dict:
    row = await store().get_purchase(purchase_id) if _ID_RE.match(purchase_id or "") else None
    if not row or row["owner_id"] != owner:
        raise HTTPException(404, "No such purchase")
    return row


async def _event(row: dict, actor: str, event: str, detail: dict | None = None) -> None:
    await store().add_event(row["id"], row["owner_id"], actor, event, detail or {})


# ── providers ─────────────────────────────────────────────────────────

async def link_conn(owner: str) -> dict | None:
    from captain_claw.flight_deck import link_wallet

    conn = await store().get_connection(owner, LINK)
    return conn if conn and conn.get("refresh_token") and link_wallet.configured() else None


def link_client(conn: dict):
    from captain_claw.flight_deck import link_wallet

    owner = conn["owner_id"]

    async def save(tokens: dict) -> None:
        await store().save_connection(owner, LINK, **tokens)

    return link_wallet.LinkClient(dict(conn), save)


async def revolut_cards_conn(owner: str) -> dict | None:
    conn = await _pr()._revolut_conn(owner)
    return conn if conn and await store().get_pref(owner, "revolut_cards") else None


async def sources_for(owner: str) -> list[str]:
    out = []
    if await link_conn(owner):
        out.append("link")
    if await revolut_cards_conn(owner):
        out.append("revolut_card")
    from captain_claw.flight_deck import ap2_client, protocol_checkout

    # Ready for this user, not just on for the deck: ACP pays through their Link
    # and needs a shop key; AP2 needs their credentials provider.
    if "link" in out and protocol_checkout.enabled("acp") and await protocol_checkout.merchants(owner):
        out.append("acp")
    if protocol_checkout.enabled("ap2") and ((await store().get_pref(owner, ap2_client.PREF)) or {}).get(
            "credentials_provider_url"):
        out.append("ap2")
    return out


async def sync(row: dict) -> dict:
    """Move a purchase on from what its provider says."""
    try:
        if row["source"] in ("acp", "ap2"):
            from captain_claw.flight_deck import protocol_checkout

            return await protocol_checkout.sync(row)
        if row["source"] == LINK and row["status"] not in PURCHASE_TERMINAL[2:] and row["status"] != "completed":
            return await _sync_link(row)
        if row["source"] == "revolut_card" and row["status"] in ("paying", "ordered", "needs_action"):
            return await _sync_revolut_card(row)
    except Exception as exc:   # the card shows the last known state
        log.info("Purchase sync failed", id=row["id"], error=str(exc))
    return row


async def _sync_link(row: dict) -> dict:
    from captain_claw.flight_deck import link_wallet

    conn = await link_conn(row["owner_id"])
    spend_id = (row.get("provider_state") or {}).get("spend_request_id")
    if not conn or not spend_id:
        return row
    sr = await link_client(conn).get_spend_request(spend_id)
    target = link_wallet.STATUS_MAP.get(str(sr.get("status") or ""), "")
    ps = dict(row.get("provider_state") or {})
    ps["link_status"] = sr.get("status")
    ps["next_action"] = ((sr.get("status_details") or {}).get("requires_action") or {}).get("next_action") or {}
    changes: dict = {"provider_state": ps}
    status = row["status"]
    # Forward only: a credential already handed out stays "paying" until Link
    # or the checkout says more.
    order = ["awaiting_approval", "needs_action", "approved", "paying", "ordered", "completed"]
    if target in ("denied", "expired", "failed", "cancelled") and status not in PURCHASE_DONE:
        changes["status"] = target
    elif target in order and status in order and order.index(target) > order.index(status):
        changes["status"] = target
    if sr.get("card_last4"):
        changes["card_last4"] = str(sr["card_last4"])
    if changes.get("status") == "completed":
        changes["completed_at"] = time.time()
    if changes.get("status", status) != status:
        await _event(row, "link", changes["status"], {"link_status": sr.get("status")})
    return await store().update_purchase(row["id"], **changes)


async def _sync_revolut_card(row: dict) -> dict:
    ps = dict(row.get("provider_state") or {})
    conn = await _pr()._revolut_conn(row["owner_id"])
    if not conn or not ps.get("card_id"):
        return row
    client = _pr()._client(conn)
    since = dt.datetime.fromtimestamp(row["created_at"] - 3600, tz=dt.timezone.utc).isoformat()
    payments = await client.card_payments(ps["card_id"], since)
    done = next((t for t in payments if t.get("state") == "completed"), None)
    pending = next((t for t in payments if t.get("state") == "pending"), None)
    bad = next((t for t in payments if t.get("state") in ("declined", "failed", "reverted")), None)
    if done or (pending and row["status"] != "ordered"):
        status = "completed" if done else "ordered"
        merchant = ((done or pending).get("merchant") or {}).get("name", "")
        ps["merchant_seen"] = merchant
        row = await store().update_purchase(row["id"], status=status, provider_state=ps,
                                            completed_at=time.time() if done else None)
        await _event(row, "revolut", status, {"merchant": merchant})
        if done:
            await client.terminate_card(ps["card_id"])
    elif bad and row["status"] != "ordered":
        row = await store().update_purchase(row["id"], status="failed",
                                            warnings=[{"code": "card_declined", "severity": "info",
                                                       "message": f"Revolut declined the card ({bad.get('reason_code') or bad.get('state')})."}])
        await client.terminate_card(ps["card_id"])
    return row


# ── agent routes ──────────────────────────────────────────────────────

class PurchaseItem(BaseModel):
    name: str
    quantity: int = 1
    unit_amount: str = ""
    url: str = ""
    id: str = ""                           # the shop's own item id (ACP / AP2)


class PurchaseRequest(BaseModel):
    source: str
    merchant_name: str
    merchant_url: str
    description: str = ""
    items: list[PurchaseItem] = []
    amount: str
    currency: str = "EUR"
    credential_type: str = "card"          # link: card | link_pay_token
    merchant_account_id: str = ""          # link_pay_token: the acct_… on the checkout page
    protocol_endpoint: str = ""            # acp / ap2: the merchant's endpoint
    shipping: dict = {}                    # acp / ap2: {name, line_one, line_two, city, state, country, postal_code, email, phone_number}
    session_id: str = ""
    automated: bool = False


class ResultRequest(BaseModel):
    outcome: str                            # confirmed | declined | needs_action | not_filled | no_button | unknown
    url: str = ""
    title: str = ""
    excerpt: str = ""
    order_ref: str = ""
    filled: list[str] = []


@router.post("/fd/payments/agent/purchases")
async def agent_create(req: PurchaseRequest, request: Request):
    from captain_claw import card_scrub
    from captain_claw.payments import checks

    pr = _pr()
    pr._require_on()
    owner, slug = pr._agent_owner(request)
    if req.source not in SOURCES:
        raise HTTPException(422, f"source is one of {', '.join(SOURCES)}")
    for value in (req.merchant_name, req.description, req.merchant_url, *(i.name for i in req.items)):
        if card_scrub.looks_like_card_data(value):
            raise HTTPException(422, "That looks like payment-card data. Captain never takes card numbers.")
    url = req.merchant_url.strip()
    if not re.match(r"^https://[^\s/]+", url):
        raise HTTPException(422, "merchant_url must be the shop's https:// address")
    try:
        cents = pr.parse_amount(req.amount)
    except ValueError:
        raise HTTPException(422, f"'{req.amount}' isn't an amount — send it like 35.00")
    currency = req.currency.strip().upper()[:3] or "EUR"
    if cents <= 0:
        raise HTTPException(422, "The amount must be more than zero")
    if await _count_recent(owner) >= PURCHASES_PER_HOUR:
        raise HTTPException(429, f"At most {PURCHASES_PER_HOUR} purchases an hour")
    cap = await store().get_cap(owner)
    cap = checks.DEFAULT_CAP_CENTS if cap is None else cap
    warnings = []
    if cap and cents > cap:
        warnings.append({"code": "over_cap", "severity": "info",
                         "message": f"{checks.format_amount(cents, currency)} is over your limit of "
                                    f"{checks.format_amount(cap)} — check it carefully before approving."})
    purchase_id = f"pur_{secrets.token_hex(6)}"
    items = [i.model_dump() for i in req.items]
    row = {"id": purchase_id, "owner_id": owner, "agent": slug, "session_id": req.session_id[:200],
           "status": "awaiting_approval", "source": req.source, "merchant_name": req.merchant_name.strip()[:120],
           "merchant_url": url[:500], "description": req.description.strip()[:500], "items": items,
           "amount_cents": cents, "currency": currency, "warnings": warnings,
           "automated": 1 if req.automated else 0, "provider_state": {}}

    if req.source == LINK:
        from captain_claw.flight_deck import link_wallet

        conn = await link_conn(owner)
        if not conn:
            raise HTTPException(409, "Link isn't connected for this user (Payments page → Link).")
        line_items = [{k: v for k, v in {"name": i["name"][:200], "quantity": i["quantity"],
                                         "unit_amount": pr.parse_amount(i["unit_amount"]) if i["unit_amount"] else None,
                                         "url": i["url"] or None}.items() if v is not None} for i in items]
        try:
            sr = await link_client(conn).create_spend_request(
                amount_cents=cents, currency=currency, merchant_name=row["merchant_name"], merchant_url=url,
                context=link_wallet.context_text(row["merchant_name"], url, row["description"], items),
                line_items=line_items, idempotency_key=purchase_id,
                credential_type="link_pay_token" if req.credential_type == "link_pay_token" else "card",
                merchant_account_id=req.merchant_account_id)
        except link_wallet.LinkError as exc:
            raise HTTPException(502, f"Link refused the spend request: {exc.message}")
        row["approval_url"] = sr.get("approval_url", "")
        row["provider_state"] = {"spend_request_id": sr.get("id"), "credential_type": sr.get("credential_type", "card"),
                                 "link_status": sr.get("status"), "test": link_wallet.test_mode()}
    elif req.source == "revolut_card":
        if not await revolut_cards_conn(owner):
            raise HTTPException(409, "Revolut cards aren't switched on for this user (Payments page → Revolut).")
        row["approval_url"] = f"/payments/purchases/{purchase_id}"
    else:
        from captain_claw.flight_deck import protocol_checkout

        shipping = {k: str(v)[:200] for k, v in (req.shipping or {}).items()
                    if k in ("name", "line_one", "line_two", "city", "state", "country", "postal_code",
                             "email", "phone_number") and v}
        row["provider_state"] = {"shipping": shipping}
        row = await protocol_checkout.start(row, req.protocol_endpoint)
    created = await store().create_purchase(row)
    await _event(created, f"agent:{slug}", "purchase_created", {"source": req.source, "amount_cents": cents})
    if req.automated:
        await _notify(owner, created)
    return {"purchase": view(created)}


async def _count_recent(owner: str) -> int:
    rows = await store().list_purchases(owner, limit=PURCHASES_PER_HOUR + 1)
    return sum(1 for r in rows if r["created_at"] > time.time() - 3600)


async def _notify(owner: str, row: dict) -> None:
    try:
        from captain_claw.flight_deck.auth import get_db

        v = view(row)
        await get_db().add_notification(owner, "payment", f"Approve a purchase: {v['merchant_name']} · {v['amount_text']}",
                                        "An automated job wants to buy something — approve or deny it.",
                                        ref_type="purchase", ref_id=row["id"])
    except Exception as exc:
        log.warning("Purchase notification failed", error=str(exc))


@router.get("/fd/payments/agent/purchases/{purchase_id}")
async def agent_get(purchase_id: str, request: Request):
    pr = _pr()
    pr._require_on()
    owner, _ = pr._agent_owner(request)
    return {"purchase": view(await sync(await _owned(purchase_id, owner)))}


@router.post("/fd/payments/agent/purchases/{purchase_id}/credential")
async def agent_credential(purchase_id: str, request: Request):
    """The one-time credential, once, for the checkout_pay step (which types it
    into the shop's form without the model seeing it)."""
    pr = _pr()
    pr._require_on()
    owner, slug = pr._agent_owner(request)
    row = await sync(await _owned(purchase_id, owner))
    if row["status"] != "approved":
        raise HTTPException(409, f"The purchase is {row['status'].replace('_', ' ')}, not approved — "
                                 "nothing can be paid yet.")
    if row["source"] not in (LINK, "revolut_card"):
        raise HTTPException(409, "This purchase is paid through its checkout protocol, not a card.")
    if not await store().release_credential(purchase_id):
        raise HTTPException(409, "The card for this purchase was already handed out.")
    await _event(row, f"agent:{slug}", "credential_released", {})
    try:
        if row["source"] == LINK:
            return await _link_credential(row)
        return await _revolut_credential(row)
    except HTTPException:
        await store().update_purchase(purchase_id, credential_released=0, status="approved")
        raise


async def _link_credential(row: dict) -> dict:
    from captain_claw.flight_deck import link_wallet

    conn = await link_conn(row["owner_id"])
    ps = row.get("provider_state") or {}
    if not conn:
        raise HTTPException(409, "Link isn't connected any more")
    kind = ps.get("credential_type") or "card"
    try:
        sr = await link_client(conn).get_spend_request(ps["spend_request_id"],
                                                      include="link_pay_token" if kind == "link_pay_token" else "card")
    except link_wallet.LinkError as exc:
        raise HTTPException(502, f"Link: {exc.message}")
    if kind == "link_pay_token":
        token = sr.get("link_pay_token")
        if not token:
            raise HTTPException(502, "Link didn't return the pay token")
        return {"kind": "link_pay_token", "token": token}
    card = sr.get("card") or {}
    if not card.get("number"):
        raise HTTPException(502, "Link didn't return the card")
    await store().update_purchase(row["id"], card_last4=str(card["number"])[-4:])
    return {"kind": "card", "card": {"number": card["number"], "cvc": card.get("cvc", ""),
                                     "exp_month": int(card["exp_month"]), "exp_year": int(card["exp_year"]),
                                     "name": (card.get("billing_address") or {}).get("name", "")}}


async def _revolut_credential(row: dict) -> dict:
    from captain_claw.flight_deck import revolut_business as rb

    conn = await _pr()._revolut_conn(row["owner_id"])
    card_id = (row.get("provider_state") or {}).get("card_id")
    if not conn or not card_id:
        raise HTTPException(409, "Revolut isn't connected, or the card wasn't made")
    try:
        details = await _pr()._client(conn).card_details(card_id)
    except rb.RevolutError as exc:
        raise HTTPException(502, f"Revolut: {exc.message} (reading card details needs the "
                                 "READ_SENSITIVE_CARD_DATA scope and the deck's IP on the certificate's whitelist)")
    month, _, year = str(details.get("expiry") or "").partition("/")
    return {"kind": "card", "card": {"number": details.get("pan", ""), "cvc": details.get("cvv", ""),
                                     "exp_month": int(month or 0), "exp_year": int(year or 0), "name": ""}}


@router.post("/fd/payments/agent/purchases/{purchase_id}/result")
async def agent_result(purchase_id: str, req: ResultRequest, request: Request):
    """What the checkout page said after paying (the tool reports it, not the model)."""
    from captain_claw import card_scrub

    pr = _pr()
    pr._require_on()
    owner, slug = pr._agent_owner(request)
    row = await _owned(purchase_id, owner)
    if row["status"] not in ("paying", "needs_action"):
        raise HTTPException(409, f"The purchase is {row['status'].replace('_', ' ')}")
    ps = dict(row.get("provider_state") or {})
    result = {"outcome": req.outcome, "url": card_scrub.scrub(req.url)[0][:500],
              "title": card_scrub.scrub(req.title)[0][:200], "excerpt": card_scrub.scrub(req.excerpt)[0][:1500],
              "filled": req.filled[:8], "at": time.time()}
    changes: dict = {"result": result}
    if req.outcome == "confirmed":
        changes["status"] = "ordered"
        changes["order_ref"] = req.order_ref[:120] or _order_ref(req.excerpt)
    elif req.outcome == "declined":
        changes["status"] = "failed"
    elif req.outcome == "needs_action":
        changes["status"] = "needs_action"
    elif req.outcome in ("not_filled", "no_button"):
        # The card was never submitted: it may be handed out again, a few times.
        ps["attempts"] = int(ps.get("attempts") or 0) + 1
        changes["provider_state"] = ps
        if ps["attempts"] < MAX_FILL_ATTEMPTS:
            changes.update(status="approved", credential_released=0)
        else:
            changes["status"] = "failed"
    row = await store().update_purchase(purchase_id, **changes)
    await _event(row, f"agent:{slug}", f"checkout_{req.outcome}", {"url": result["url"]})
    if row["status"] == "failed":
        await _wind_down(row)
    elif row["source"] == LINK and req.outcome == "confirmed":
        conn = await link_conn(owner)
        if conn:
            await link_client(conn).report(domain=_domain(row["merchant_url"]), outcome="success",
                                           spend_id=ps.get("spend_request_id", ""))
    return {"purchase": view(await sync(row))}


def _order_ref(text: str) -> str:
    m = re.search(r"(?i)(?:order|narud[žz]b\w*|bestellung)\s*(?:number|no\.?|br\.?|broj|nr\.?|#)?\s*[:#]?\s*([A-Z0-9][A-Z0-9\-]{3,30})", text or "")
    return m.group(1) if m else ""


async def _wind_down(row: dict) -> None:
    """A purchase that won't happen: its card or spend request goes too."""
    ps = row.get("provider_state") or {}
    try:
        if row["source"] in ("acp", "ap2"):
            from captain_claw.flight_deck import protocol_checkout

            await protocol_checkout.cancel(row)
        if row["source"] in (LINK, "acp") and ps.get("spend_request_id"):
            conn = await link_conn(row["owner_id"])
            if conn:
                await link_client(conn).cancel_spend_request(ps["spend_request_id"])
        elif row["source"] == "revolut_card" and ps.get("card_id"):
            conn = await _pr()._revolut_conn(row["owner_id"])
            if conn:
                await _pr()._client(conn).terminate_card(ps["card_id"])
    except Exception as exc:
        log.warning("Purchase wind-down failed", id=row["id"], error=str(exc))


@router.post("/fd/payments/agent/purchases/{purchase_id}/cancel")
async def agent_cancel(purchase_id: str, request: Request):
    pr = _pr()
    pr._require_on()
    owner, slug = pr._agent_owner(request)
    return await _cancel(await _owned(purchase_id, owner), f"agent:{slug}")


async def _cancel(row: dict, actor: str) -> dict:
    if row["status"] in PURCHASE_TERMINAL:
        raise HTTPException(409, f"The purchase is already {row['status'].replace('_', ' ')}")
    if row["status"] in ("paying", "ordered"):
        raise HTTPException(409, "The checkout already started — check the shop before cancelling.")
    row = await store().update_purchase(row["id"], status="cancelled")
    await _wind_down(row)
    await _event(row, actor, "cancelled", {})
    return {"purchase": view(row)}


# ── connections: Link, Revolut cards ──────────────────────────────────

_LINK_STATES: dict[str, tuple[str, str, str, float]] = {}   # state → (owner, verifier, redirect_uri, expires)


async def _link_redirect(request: Request, owner: str) -> str:
    import os

    override = os.environ.get("FD_LINK_REDIRECT_URL", "").strip()
    if override:
        return override
    base, _ = await _pr()._link_base(request, owner)
    return f"{base}/fd/payments/link/callback"


@router.get("/fd/payments/link/status")
async def link_status(request: Request, user: dict = Depends(get_current_user)):
    from captain_claw.flight_deck import link_wallet

    _pr()._require_on()
    conn = await store().get_connection(user["id"], LINK)
    out = {"configured": link_wallet.configured(), "connected": bool(conn and conn.get("refresh_token")),
           "test_mode": link_wallet.test_mode(), "redirect_uri": "", "user": None}
    if out["configured"]:
        out["redirect_uri"] = await _link_redirect(request, user["id"])
    if out["connected"] and out["configured"]:
        try:
            info = await link_client(conn).user_info()
            out["user"] = {"email": info.get("email", ""), "name": info.get("name", ""),
                           "limits": info.get("agent_wallet_spend_limits") or {}}
        except link_wallet.LinkError as exc:
            out["error"] = exc.message
    return out


@router.post("/fd/payments/link/connect")
async def link_connect(request: Request, user: dict = Depends(get_current_user)):
    from captain_claw.flight_deck import link_wallet

    _pr()._require_on()
    if not link_wallet.configured():
        raise HTTPException(409, "Link isn't set up on this deck (FD_LINK_CLIENT_ID / _SECRET / _PUBLISHABLE_KEY)")
    now = time.time()
    for state, entry in list(_LINK_STATES.items()):
        if entry[3] < now:
            _LINK_STATES.pop(state, None)
    verifier, challenge = link_wallet.pkce_pair()
    state = secrets.token_urlsafe(18)
    redirect = await _link_redirect(request, user["id"])
    _LINK_STATES[state] = (user["id"], verifier, redirect, now + 600)
    return {"url": link_wallet.authorize_url(redirect_uri=redirect, state=state, code_challenge=challenge)}


@router.get("/fd/payments/link/callback")
async def link_callback(code: str = "", state: str = "", error: str = "", error_description: str = ""):
    """Link's redirect after the user connected their wallet (public route; the
    one-time state names the user, PKCE proves it's the same flow)."""
    import html as _html

    from fastapi.responses import HTMLResponse

    from captain_claw.flight_deck import link_wallet

    def page(title: str, body: str, status: int = 200) -> HTMLResponse:
        return HTMLResponse(f'<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width">'
                            f'<title>{_html.escape(title)}</title><body style="font:16px system-ui;max-width:520px;'
                            f'margin:40px auto;padding:0 16px"><h2>{_html.escape(title)}</h2>{body}</body>',
                            status_code=status, headers={"Cache-Control": "no-store", "Referrer-Policy": "no-referrer"})

    entry = _LINK_STATES.pop(state, None) if state else None
    if not entry or entry[3] < time.time():
        return page("Link not connected", "<p>This sign-in link expired. Start again from the Payments page.</p>", 400)
    if error or not code:
        return page("Link not connected", f"<p>{_html.escape(error_description or error or 'No code returned')}.</p>", 400)
    owner, verifier, redirect, _ = entry
    try:
        tokens = await link_wallet.exchange_code(code=code, redirect_uri=redirect, code_verifier=verifier)
    except link_wallet.LinkError as exc:
        return page("Link not connected", f"<p>{_html.escape(exc.message)}</p>", 502)
    if "payment_methods.agentic" not in (tokens.get("scope") or "payment_methods.agentic"):
        return page("Link not connected", "<p>Link didn't grant spending access, so Captain can't ask to spend.</p>", 400)
    await store().save_connection(owner, LINK, environment="test" if link_wallet.test_mode() else "production",
                                  refresh_token=tokens["refresh_token"], access_token=tokens["access_token"],
                                  access_expires_at=tokens["access_expires_at"], connected_at=time.time())
    return page("Link connected", "<p>Captain can now ask your Link wallet for a one-time card when it buys "
                "something for you. You approve every purchase in Link. You can close this tab.</p>")


@router.delete("/fd/payments/link/connection")
async def link_disconnect(user: dict = Depends(get_current_user)):
    from captain_claw.flight_deck import link_wallet

    _pr()._require_on()
    conn = await store().get_connection(user["id"], LINK)
    if conn and conn.get("refresh_token"):
        await link_wallet.revoke(conn["refresh_token"])
    await store().delete_connection(user["id"], LINK)
    return {"connected": False}


class RevolutCardsRequest(BaseModel):
    enabled: bool


@router.post("/fd/payments/revolut/cards")
async def revolut_cards(req: RevolutCardsRequest, user: dict = Depends(get_current_user)):
    """Switch "a Revolut card per purchase" on or off. On: the consent link
    Revolut needs to add READ_SENSITIVE_CARD_DATA (never PAY) — and the deck's
    public IP must be on the certificate's IP whitelist."""
    from captain_claw.flight_deck import revolut_business as rb

    pr = _pr()
    pr._require_on()
    if not req.enabled:
        await store().delete_pref(user["id"], "revolut_cards")
        return {"enabled": False}
    conn = await store().get_connection(user["id"], "revolut")
    if not conn or not conn.get("client_id") or not conn.get("private_key"):
        raise HTTPException(409, "Connect Revolut Business first")
    await store().set_pref(user["id"], "revolut_cards", True)
    state = secrets.token_urlsafe(16)
    pr._CONNECT_STATES[state] = (user["id"], time.time() + pr._CONNECT_TTL)
    return {"enabled": True, "url": rb.consent_url(conn, state, cards=True)}


class AcpMerchantRequest(BaseModel):
    domain: str
    api_key: str
    name: str = ""


@router.get("/fd/payments/acp/merchants")
async def acp_merchants(user: dict = Depends(get_current_user)):
    from captain_claw.flight_deck import protocol_checkout

    _pr()._require_on()
    rows = await protocol_checkout.merchants(user["id"])
    return {"merchants": [{"domain": d, "name": m.get("name", ""), "added_at": m.get("added_at")}
                          for d, m in sorted(rows.items())]}


@router.put("/fd/payments/acp/merchants")
async def acp_merchant_add(req: AcpMerchantRequest, user: dict = Depends(get_current_user)):
    """The shop's ACP key for this user (ACP agents are onboarded per shop)."""
    from captain_claw.flight_deck import protocol_checkout

    _pr()._require_on()
    domain = protocol_checkout.domain_of(req.domain)
    key = req.api_key.strip()
    if not domain or "." not in domain or not key or len(key) > 500:
        raise HTTPException(422, "Give the shop's domain and its ACP key")
    try:
        await protocol_checkout.discover(domain)
    except protocol_checkout.ProtocolError as exc:
        raise HTTPException(422, exc.message)
    rows = await protocol_checkout.merchants(user["id"])
    rows[domain] = {"api_key": key, "name": req.name.strip()[:120], "added_at": time.time()}
    await store().set_pref(user["id"], protocol_checkout.ACP_PREF, rows)
    return await acp_merchants(user)


@router.delete("/fd/payments/acp/merchants/{domain}")
async def acp_merchant_remove(domain: str, user: dict = Depends(get_current_user)):
    from captain_claw.flight_deck import protocol_checkout

    _pr()._require_on()
    rows = await protocol_checkout.merchants(user["id"])
    rows.pop(protocol_checkout.domain_of(domain), None)
    await store().set_pref(user["id"], protocol_checkout.ACP_PREF, rows)
    return await acp_merchants(user)


# ── owner routes ──────────────────────────────────────────────────────

@router.get("/fd/payments/purchases")
async def owner_list(user: dict = Depends(get_current_user)):
    _pr()._require_on()
    rows = await store().list_purchases(user["id"])
    rows = [await sync(r) if r["status"] not in PURCHASE_TERMINAL else r for r in rows[:30]] + rows[30:]
    return {"purchases": [view(r) for r in rows], "sources": await sources_for(user["id"])}


@router.get("/fd/payments/purchases/{purchase_id}")
async def owner_get(purchase_id: str, user: dict = Depends(get_current_user)):
    _pr()._require_on()
    row = await sync(await _owned(purchase_id, user["id"]))
    return {"purchase": view(row), "events": await store().events(purchase_id)}


@router.post("/fd/payments/purchases/{purchase_id}/approve")
async def owner_approve(purchase_id: str, user: dict = Depends(get_current_user)):
    """The owner approves this seller and amount (Revolut card, checkout
    protocols). Link purchases are approved in Link itself."""
    from captain_claw.flight_deck import protocol_checkout

    _pr()._require_on()
    row = await _owned(purchase_id, user["id"])
    if row["status"] != "awaiting_approval":
        raise HTTPException(409, f"The purchase is {row['status'].replace('_', ' ')}")
    if row["source"] == LINK:
        raise HTTPException(409, "Approve Link purchases in Link (the approval link).")
    actor = f"user:{user['id']}"
    if row["source"] == "revolut_card":
        row = await _make_revolut_card(row)
    else:
        row = await protocol_checkout.approve(row, user["id"])
    await _event(row, actor, "approved", {})
    from captain_claw.flight_deck import line

    await line.settle(user["id"], "purchase", row["id"])
    return {"purchase": view(row)}


async def _make_revolut_card(row: dict) -> dict:
    from captain_claw.flight_deck import revolut_business as rb

    owner = row["owner_id"]
    conn = await revolut_cards_conn(owner)
    if not conn:
        raise HTTPException(409, "Revolut cards aren't switched on")
    client = _pr()._client(conn)
    try:
        contact = await store().get_pref(owner, "revolut_card_contact")
        if not contact:
            members = await client.team_members()
            owner_member = next((m for m in members if m.get("role_id") == "owner"), members[0] if members else None)
            if not owner_member:
                raise HTTPException(409, "Revolut has no team member to name on the card")
            contact = owner_member["id"]
        tomorrow = (dt.date.today() + dt.timedelta(days=2)).isoformat()
        card = await client.create_purchase_card(
            request_id=row["id"], contact_id=contact, account_id=conn["account_id"],
            amount_cents=row["amount_cents"], currency=row["currency"],
            label=f"Captain {row['merchant_name']}", purchase_id=row["id"], end_date=tomorrow)
    except rb.RevolutError as exc:
        raise HTTPException(502, f"Revolut couldn't make the card: {exc.message}")
    ps = {**(row.get("provider_state") or {}), "card_id": card.get("id"), "card_state": card.get("state")}
    return await store().update_purchase(row["id"], status="approved", provider_state=ps,
                                         card_last4=str(card.get("last_digits") or ""))


class ChallengeRequest(BaseModel):
    response: str


@router.post("/fd/payments/purchases/{purchase_id}/challenge")
async def owner_challenge(purchase_id: str, req: ChallengeRequest, user: dict = Depends(get_current_user)):
    """The owner answers the payment's challenge (AP2: e.g. an OTP)."""
    from captain_claw.flight_deck import ap2_client

    _pr()._require_on()
    row = await _owned(purchase_id, user["id"])
    if row["source"] != "ap2" or row["status"] != "needs_action":
        raise HTTPException(409, "This purchase isn't waiting for a challenge answer")
    row = await ap2_client.answer_challenge(row, req.response.strip()[:64])
    await _event(row, f"user:{user['id']}", "challenge_answered", {})
    return {"purchase": view(row)}


class Ap2SettingsRequest(BaseModel):
    credentials_provider_url: str
    user_email: str


@router.get("/fd/payments/ap2/settings")
async def ap2_settings(user: dict = Depends(get_current_user)):
    from captain_claw.flight_deck import ap2_client

    _pr()._require_on()
    settings = await store().get_pref(user["id"], ap2_client.PREF) or {}
    conn = await store().get_connection(user["id"], ap2_client.PROVIDER)
    return {"enabled": ap2_client.enabled(), "credentials_provider_url": settings.get("credentials_provider_url", ""),
            "user_email": settings.get("user_email", ""), "public_key": (conn or {}).get("certificate", ""),
            "kid": (conn or {}).get("client_id", ""), "shopping_agent_id": ap2_client.shopping_agent_id()}


@router.put("/fd/payments/ap2/settings")
async def ap2_settings_set(req: Ap2SettingsRequest, user: dict = Depends(get_current_user)):
    from captain_claw.flight_deck import ap2_client

    _pr()._require_on()
    url = req.credentials_provider_url.strip()
    if not url.startswith(("https://", "http://localhost", "http://127.0.0.1")) or "@" not in req.user_email:
        raise HTTPException(422, "Give the credentials provider's A2A URL and your wallet email")
    await store().set_pref(user["id"], ap2_client.PREF, {"credentials_provider_url": url,
                                                         "user_email": req.user_email.strip()[:200]})
    await ap2_client.user_key(user["id"])          # make the signing key now, so its public key can be shared
    return await ap2_settings(user)


@router.post("/fd/payments/purchases/{purchase_id}/deny")
async def owner_deny(purchase_id: str, user: dict = Depends(get_current_user)):
    _pr()._require_on()
    row = await _owned(purchase_id, user["id"])
    if row["status"] not in ("awaiting_approval", "approved", "needs_action"):
        raise HTTPException(409, f"The purchase is {row['status'].replace('_', ' ')}")
    row = await store().update_purchase(row["id"], status="denied")
    await _wind_down(row)
    await _event(row, f"user:{user['id']}", "denied", {})
    from captain_claw.flight_deck import line

    await line.settle(user["id"], "purchase", row["id"])
    return {"purchase": view(row)}


@router.post("/fd/payments/purchases/{purchase_id}/cancel")
async def owner_cancel(purchase_id: str, user: dict = Depends(get_current_user)):
    _pr()._require_on()
    return await _cancel(await _owned(purchase_id, user["id"]), f"user:{user['id']}")
