"""Revolut Business API client — payment drafts for Payments Phase 1
(docs/payments-phase1-plan.md).

Captain asks Revolut for scope ``READ,WRITE`` only: WRITE creates
counterparties and payment drafts, PAY (initiating transfers) is never asked
for, so Revolut itself refuses any payment from Captain's token. The user sends
the draft in the Revolut Business app.

Auth (per developer.revolut.com): an RS256 client-assertion JWT signed with a
key whose certificate the user uploaded in Revolut Business; the consent page
returns a code (valid 2 minutes) that ``/auth/token`` exchanges for an access
token (40 minutes) and a refresh token (doesn't expire). A refresh invalidates
the previous access token.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import time
import urllib.parse
from typing import Any, Awaitable, Callable

import httpx

from captain_claw.logging import get_logger

log = get_logger(__name__)

SCOPE = "READ,WRITE"            # never PAY: Captain's token can't move money
# Phase 3 (one card per purchase) also reads the card it made: Revolut allows
# that scope only with an IP whitelist on the certificate. Still never PAY.
SCOPE_CARDS = "READ,WRITE,READ_SENSITIVE_CARD_DATA"
AUDIENCE = "https://revolut.com"
ASSERTION_TYPE = "urn:ietf:params:oauth:client-assertion-type:jwt-bearer"
_API = {"production": "https://b2b.revolut.com/api/1.0", "sandbox": "https://sandbox-b2b.revolut.com/api/1.0"}
_CONSENT = {"production": "https://business.revolut.com/app-confirm",
            "sandbox": "https://sandbox-business.revolut.com/app-confirm"}
DRAFT_STATES_DONE = ("COMPLETED",)
DRAFT_STATES_FAILED = ("DECLINED", "FAILED", "REVERTED")
DRAFT_STATES_GONE = ("CANCELLED", "DELETED")

# Tests swap in an httpx.MockTransport.
transport: httpx.AsyncBaseTransport | None = None
_REFRESH_LOCKS: dict[str, asyncio.Lock] = {}


class RevolutError(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


def environment(value: str) -> str:
    return "sandbox" if str(value or "").strip().lower() == "sandbox" else "production"


def generate_key_and_certificate(common_name: str = "Captain payments") -> tuple[str, str]:
    """(private key PEM, self-signed X.509 certificate PEM) — the certificate is
    what the user pastes into Revolut Business → APIs → Business API."""
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, common_name[:64]),
                      x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Captain")])
    now = dt.datetime.now(dt.timezone.utc)
    cert = (x509.CertificateBuilder()
            .subject_name(name).issuer_name(name).public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - dt.timedelta(days=1))
            .not_valid_after(now + dt.timedelta(days=1825))
            .sign(key, hashes.SHA256()))
    key_pem = key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                serialization.NoEncryption()).decode()
    return key_pem, cert.public_bytes(serialization.Encoding.PEM).decode()


def issuer(redirect_uri: str) -> str:
    """The JWT ``iss``: the redirect URI's domain, without the scheme."""
    return (urllib.parse.urlsplit(redirect_uri).hostname or "").lower()


def client_assertion(conn: dict, *, lifetime: int = 300) -> str:
    """A short-lived signed JWT (Revolut recommends just long enough to refresh)."""
    import jwt

    claims = {"iss": issuer(conn["redirect_uri"]), "sub": conn["client_id"], "aud": AUDIENCE,
              "exp": int(time.time()) + lifetime}
    return jwt.encode(claims, conn["private_key"], algorithm="RS256", headers={"typ": "JWT"})


def consent_url(conn: dict, state: str = "", *, cards: bool = False) -> str:
    query = {"client_id": conn["client_id"], "redirect_uri": conn["redirect_uri"],
             "response_type": "code", "scope": SCOPE_CARDS if cards else SCOPE}
    if state:
        query["state"] = state
    return f"{_CONSENT[environment(conn.get('environment'))]}?{urllib.parse.urlencode(query, safe=',:/')}"


def _api(conn: dict) -> str:
    return _API[environment(conn.get("environment"))]


def _message(resp: httpx.Response) -> str:
    try:
        body = resp.json()
    except Exception:  # noqa: BLE001
        return resp.text[:300] or f"HTTP {resp.status_code}"
    if isinstance(body, dict):
        return str(body.get("message") or body.get("error_description") or body.get("error") or body)[:300]
    return str(body)[:300]


def _json(resp: httpx.Response) -> Any:
    try:
        return resp.json()
    except ValueError:
        raise RevolutError(502, "Revolut's answer isn't JSON")


async def _token_request(conn: dict, form: dict) -> dict:
    form = {**form, "client_assertion_type": ASSERTION_TYPE, "client_assertion": client_assertion(conn)}
    try:
        async with httpx.AsyncClient(timeout=30, transport=transport) as client:
            resp = await client.post(f"{_api(conn)}/auth/token", data=form)
    except httpx.HTTPError as exc:
        raise RevolutError(502, f"Couldn't reach Revolut ({type(exc).__name__})")
    if resp.status_code != 200:
        raise RevolutError(resp.status_code, _message(resp))
    data = _json(resp)
    out = {"access_token": data["access_token"],
           "access_expires_at": time.time() + float(data.get("expires_in") or 2399)}
    if data.get("refresh_token"):
        out["refresh_token"] = data["refresh_token"]
    return out


async def exchange_code(conn: dict, code: str) -> dict:
    """Consent code → {access_token, access_expires_at, refresh_token}."""
    return await _token_request(conn, {"grant_type": "authorization_code", "code": code})


class RevolutClient:
    """Calls for one owner's connection; *save* persists refreshed tokens."""

    def __init__(self, conn: dict, save: Callable[[dict], Awaitable[None]]) -> None:
        self.conn = conn
        self._save = save

    async def _token(self, force: bool = False) -> str:
        lock = _REFRESH_LOCKS.setdefault(self.conn.get("owner_id", ""), asyncio.Lock())
        async with lock:
            fresh = float(self.conn.get("access_expires_at") or 0) > time.time() + 60
            if self.conn.get("access_token") and fresh and not force:
                return self.conn["access_token"]
            if not self.conn.get("refresh_token"):
                raise RevolutError(401, "Revolut Business isn't connected")
            tokens = await _token_request(self.conn, {"grant_type": "refresh_token",
                                                      "refresh_token": self.conn["refresh_token"]})
            self.conn.update(tokens)
            await self._save(tokens)
            return self.conn["access_token"]

    async def request(self, method: str, path: str, *, json: Any = None, params: dict | None = None) -> Any:
        for attempt in (0, 1):
            token = await self._token(force=attempt == 1)
            try:
                async with httpx.AsyncClient(timeout=30, transport=transport) as client:
                    resp = await client.request(method, f"{_api(self.conn)}{path}", json=json, params=params,
                                                headers={"Authorization": f"Bearer {token}"})
            except httpx.HTTPError as exc:
                raise RevolutError(502, f"Couldn't reach Revolut ({type(exc).__name__})")
            if resp.status_code == 401 and attempt == 0:
                continue
            if resp.status_code >= 400:
                raise RevolutError(resp.status_code, _message(resp))
            if resp.status_code == 204 or not resp.content:
                return None
            return _json(resp)
        raise RevolutError(401, "Revolut refused the token")

    async def accounts(self) -> list[dict]:
        return await self.request("GET", "/accounts") or []

    async def find_counterparty(self, iban: str) -> dict | None:
        rows = await self.request("GET", "/counterparties", params={"iban": iban}) or []
        return rows[0] if rows else None

    async def create_counterparty(self, *, name: str, iban: str, individual: bool) -> dict:
        body: dict[str, Any] = {"bank_country": iban[:2], "currency": "EUR", "iban": iban}
        if individual:
            first, _, last = name.partition(" ")
            body["individual_name"] = {"first_name": first, "last_name": last or first}
        else:
            body["company_name"] = name
        return await self.request("POST", "/counterparty", json=body)

    async def validate_name(self, *, name: str, iban: str, individual: bool) -> dict:
        body: dict[str, Any] = {"iban": iban, "recipient_country": iban[:2], "recipient_currency": "EUR"}
        if individual:
            first, _, last = name.partition(" ")
            body["individual_name"] = {"first_name": first, "last_name": last or first}
        else:
            body["company_name"] = name
        return await self.request("POST", "/account-name-validation", json=body)

    async def create_draft(self, *, account_id: str, counterparty_id: str, counterparty_account_id: str,
                           amount_cents: int, reference: str, title: str, schedule_for: str = "") -> str:
        receiver = {"counterparty_id": counterparty_id}
        if counterparty_account_id:
            receiver["account_id"] = counterparty_account_id
        body = {"title": title[:100], "payments": [{
            "account_id": account_id, "receiver": receiver, "amount": round(amount_cents / 100, 2),
            "currency": "EUR", "reference": reference[:140]}]}
        if schedule_for:
            body["schedule_for"] = schedule_for          # a date: the draft is paid then, once approved
        data = await self.request("POST", "/payment-drafts", json=body)
        return str((data or {}).get("id") or "")

    async def get_draft(self, draft_id: str) -> dict | None:
        try:
            return await self.request("GET", f"/payment-drafts/{urllib.parse.quote(draft_id)}")
        except RevolutError as exc:
            if exc.status == 404:
                return None
            raise

    # ── one card per purchase (Phase 3) ─────────────────────────────

    async def team_members(self) -> list[dict]:
        return await self.request("GET", "/team-members") or []

    async def create_purchase_card(self, *, request_id: str, contact_id: str, account_id: str,
                                   amount_cents: int, currency: str, label: str, purchase_id: str,
                                   end_date: str) -> dict:
        """A company virtual card good for this purchase only: one transaction up
        to the amount, the amount in all, terminated by Revolut after *end_date*."""
        amount = round(int(amount_cents) / 100, 2)
        limit = {"amount": amount, "currency": currency.upper()}
        body = {
            "request_id": request_id, "virtual": True, "contact_ids": [contact_id], "label": label[:30],
            "accounts": [account_id], "references": [{"name": "captain_purchase", "value": purchase_id}],
            "spending_limits": {"single": limit, "all_time": limit},
            "spending_period": {"end_date": end_date, "end_date_action": "terminate"},
        }
        return await self.request("POST", "/cards", json=body)

    async def card_details(self, card_id: str) -> dict:
        """{pan, cvv, expiry "MM/YYYY"} — needs READ_SENSITIVE_CARD_DATA."""
        return await self.request("GET", f"/cards/{urllib.parse.quote(card_id)}/sensitive-details") or {}

    async def terminate_card(self, card_id: str) -> None:
        try:
            await self.request("DELETE", f"/cards/{urllib.parse.quote(card_id)}")
        except RevolutError as exc:
            if exc.status not in (403, 404):
                raise

    async def card_payments(self, card_id: str, since_iso: str) -> list[dict]:
        """This card's card payments since *since_iso* (Revolut has no card filter)."""
        rows = await self.request("GET", "/transactions",
                                  params={"from": since_iso, "type": "card_payment", "count": 1000}) or []
        return [t for t in rows if (t.get("card") or {}).get("id") == card_id]

    async def delete_draft(self, draft_id: str) -> None:
        try:
            await self.request("DELETE", f"/payment-drafts/{urllib.parse.quote(draft_id)}")
        except RevolutError as exc:
            if exc.status != 404:
                raise
