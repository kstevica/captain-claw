"""Authentication REST endpoints for Flight Deck."""

from __future__ import annotations

import hashlib
import ipaddress
import logging
import os
import secrets
import time
import unicodedata
from collections import OrderedDict
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, status
from pydantic import BaseModel, EmailStr

from captain_claw.flight_deck.auth import (
    REFRESH_COOKIE,
    REFRESH_TOKEN_TTL,
    _fd_auth_enabled,
    create_access_token,
    create_refresh_token,
    get_current_user,
    get_db,
    hash_password,
    hash_token,
    verify_password,
)
from captain_claw.flight_deck.rate_limiter import _SlidingWindow

router = APIRouter(prefix="/fd/auth", tags=["auth"])

log = logging.getLogger("flight_deck.auth")


# ── Request / response models ───────────────────────────────────────

class RegisterRequest(BaseModel):
    email: str
    password: str
    display_name: str = ""


class LoginRequest(BaseModel):
    email: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: dict


class UserResponse(BaseModel):
    id: str
    email: str
    display_name: str
    role: str


class UpdateProfileRequest(BaseModel):
    display_name: str | None = None
    password: str | None = None
    current_password: str | None = None


# ── Helpers ──────────────────────────────────────────────────────────

def _cookie_secure() -> bool:
    """Whether the refresh cookie is marked Secure (HTTPS-only).

    ``FD_COOKIE_SECURE`` wins when set. Otherwise default to Secure whenever the
    deployment is locked down (i.e. reachable beyond the owner's machine, behind
    a TLS proxy) — so a team deployment gets a Secure cookie automatically while
    local http development keeps working.
    """
    v = os.environ.get("FD_COOKIE_SECURE", "").lower()
    if v in ("true", "1", "yes"):
        return True
    if v in ("false", "0", "no"):
        return False
    return os.environ.get("FD_LOCKDOWN", "").lower() in ("true", "1", "yes")


def _set_refresh_cookie(response: Response, refresh_token: str) -> None:
    response.set_cookie(
        key=REFRESH_COOKIE,
        value=refresh_token,
        max_age=int(REFRESH_TOKEN_TTL.total_seconds()),
        httponly=True,
        samesite="lax",
        path="/fd/auth",
        secure=_cookie_secure(),
    )


def _clear_refresh_cookie(response: Response) -> None:
    response.delete_cookie(key=REFRESH_COOKIE, path="/fd/auth")


# ── Endpoints ────────────────────────────────────────────────────────

@router.post("/register", response_model=TokenResponse)
async def register(body: RegisterRequest, response: Response):
    # An auth-disabled deck (desktop build) has no accounts, but it does open
    # its DB for connector settings. Registering there would let whoever
    # reaches the port first plant the first — admin — account, which silently
    # becomes real the day FD_AUTH_ENABLED is switched on.
    if not _fd_auth_enabled():
        raise HTTPException(
            status_code=403,
            detail="Registration is disabled: this Flight Deck runs without accounts (FD_AUTH_ENABLED=false).",
        )
    db = get_db()
    if not body.email or not body.password:
        raise HTTPException(status_code=400, detail="Email and password required")
    if len(body.password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters")

    existing = await db.get_user_by_email(body.email)
    if existing:
        raise HTTPException(status_code=409, detail="Email already registered")

    # First user becomes admin (bootstrap). Self-registration of ADDITIONAL
    # users is closed by default for team deployments — an admin creates
    # accounts from the Admin page. An operator can re-open it with
    # FD_REGISTRATION_OPEN=1; FD_REGISTRATION_DISABLED=1 forces it closed.
    user_count = await db.count_users()
    if user_count > 0:
        disabled = os.environ.get("FD_REGISTRATION_DISABLED", "").lower() in ("true", "1", "yes")
        open_reg = os.environ.get("FD_REGISTRATION_OPEN", "").lower() in ("true", "1", "yes")
        if disabled or not open_reg:
            raise HTTPException(
                status_code=403,
                detail="Registration is closed. Ask an administrator to create your account.",
            )

    pw_hash = hash_password(body.password)
    display = body.display_name or body.email.split("@")[0]

    role = "admin" if user_count == 0 else "user"

    user = await db.create_user(
        email=body.email, password_hash=pw_hash,
        display_name=display, role=role,
    )

    access_token = create_access_token(user["id"], role=role)
    refresh_token = create_refresh_token()

    expires_at = (datetime.now(timezone.utc) + REFRESH_TOKEN_TTL).isoformat()
    await db.create_refresh_session(user["id"], hash_token(refresh_token), expires_at)
    _set_refresh_cookie(response, refresh_token)

    return TokenResponse(
        access_token=access_token,
        user={"id": user["id"], "email": user["email"],
              "display_name": display, "role": role},
    )


@router.post("/login", response_model=TokenResponse)
async def login(body: LoginRequest, response: Response):
    db = get_db()
    user = await db.get_user_by_email(body.email)
    if not user or not verify_password(body.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    return await _issue_session(db, user, response)


async def _issue_session(db, user: dict, response: Response) -> TokenResponse:
    """Sign *user* in on this response: a fresh access token, a new refresh
    session row and the refresh cookie. Shared by login and device pairing."""
    access_token = create_access_token(user["id"], role=user["role"])
    refresh_token = create_refresh_token()

    expires_at = (datetime.now(timezone.utc) + REFRESH_TOKEN_TTL).isoformat()
    await db.create_refresh_session(user["id"], hash_token(refresh_token), expires_at)
    _set_refresh_cookie(response, refresh_token)

    return TokenResponse(
        access_token=access_token,
        user={"id": user["id"], "email": user["email"],
              "display_name": user["display_name"], "role": user["role"]},
    )


@router.post("/refresh")
async def refresh(request: Request, response: Response):
    refresh_token = request.cookies.get(REFRESH_COOKIE)
    if not refresh_token:
        raise HTTPException(status_code=401, detail="No refresh token")

    db = get_db()
    token_hash = hash_token(refresh_token)

    # Find the session matching this refresh token
    # We need to search by hash since we don't store the session_id in the cookie
    assert db._db is not None
    async with db._db.execute(
        "SELECT * FROM user_sessions WHERE refresh_token_hash = ?", (token_hash,)
    ) as cur:
        session = await cur.fetchone()

    if not session:
        _clear_refresh_cookie(response)
        raise HTTPException(status_code=401, detail="Invalid refresh token")

    session = dict(session)
    now = datetime.now(timezone.utc)
    expires = datetime.fromisoformat(session["expires_at"])
    if now > expires:
        await db.delete_refresh_session(session["id"])
        _clear_refresh_cookie(response)
        raise HTTPException(status_code=401, detail="Refresh token expired")

    user = await db.get_user_by_id(session["user_id"])
    if not user:
        await db.delete_refresh_session(session["id"])
        _clear_refresh_cookie(response)
        raise HTTPException(status_code=401, detail="User not found")

    # Rotate: delete old session, create new tokens
    await db.delete_refresh_session(session["id"])
    new_access = create_access_token(user["id"], role=user["role"])
    new_refresh = create_refresh_token()
    new_expires = (now + REFRESH_TOKEN_TTL).isoformat()
    await db.create_refresh_session(user["id"], hash_token(new_refresh), new_expires)
    _set_refresh_cookie(response, new_refresh)

    return {
        "access_token": new_access,
        "token_type": "bearer",
        "user": {"id": user["id"], "email": user["email"],
                 "display_name": user["display_name"], "role": user["role"]},
    }


@router.post("/logout")
async def logout(request: Request, response: Response):
    refresh_token = request.cookies.get(REFRESH_COOKIE)
    if refresh_token:
        db = get_db()
        token_hash = hash_token(refresh_token)
        assert db._db is not None
        await db._db.execute(
            "DELETE FROM user_sessions WHERE refresh_token_hash = ?", (token_hash,)
        )
        await db._db.commit()
    _clear_refresh_cookie(response)
    return {"ok": True}


@router.get("/me")
async def get_me(user: dict = Depends(get_current_user)):
    return {
        "id": user["id"], "email": user["email"],
        "display_name": user["display_name"], "role": user["role"],
    }


async def _refresh_owner_profile(db, user_id: str) -> None:
    """The owner profile names its owner: rewrite their agents' copies after a
    display-name change. Best-effort — never fails the update."""
    try:
        from captain_claw.flight_deck import tenant_profile

        await tenant_profile.refresh_agents(db, user_id)
    except Exception:
        pass
    # …and the shared-context labels that name them (context packs).
    try:
        from captain_claw.flight_deck import context_packs

        await context_packs.refresh_for_user(db, user_id)
    except Exception:
        pass


@router.put("/me")
async def update_me(body: UpdateProfileRequest, user: dict = Depends(get_current_user)):
    db = get_db()
    updates: dict = {}

    if body.display_name is not None:
        updates["display_name"] = body.display_name

    if body.password is not None:
        if not body.current_password:
            raise HTTPException(status_code=400, detail="Current password required")
        full_user = await db.get_user_by_email(user["email"])
        if not full_user or not verify_password(body.current_password, full_user["password_hash"]):
            raise HTTPException(status_code=400, detail="Current password is incorrect")
        if len(body.password) < 6:
            raise HTTPException(status_code=400, detail="Password must be at least 6 characters")
        updates["password_hash"] = hash_password(body.password)

    if updates:
        await db.update_user(user["id"], **updates)
        if "display_name" in updates and updates["display_name"] != (user.get("display_name") or ""):
            await _refresh_owner_profile(db, str(user["id"]))

    updated = await db.get_user_by_id(user["id"])
    return {
        "id": updated["id"], "email": updated["email"],
        "display_name": updated["display_name"], "role": updated["role"],
    }


# ── Device pairing (RFC 8628-style device authorization) ─────────────
# A device that can't comfortably type a password (smart glasses) signs in by
# showing a short user code; the wearer approves it from a phone/desktop where
# they are already signed in. The device then receives a normal session —
# access token + the fd_refresh cookie, exactly as login() issues them — so
# refresh, logout and /fd/agent-ws work unchanged. Nothing secret is typed or
# spoken on the device.
#
#   POST /fd/auth/pair/start    public   {label?}             → device_code, user_code, …
#   POST /fd/auth/pair/poll     public   {device_code}        → pending|denied|expired|approved(+tokens)
#   GET  /fd/auth/pair/lookup   signed-in ?code=XXXX-XXXX     → which device is asking
#   POST /fd/auth/pair/approve  signed-in {user_code, approve} → approved|denied
#
# The store is in-memory: Flight Deck runs as a single uvicorn process, and a
# pairing lives for minutes, so a restart only forces the device to show a
# new code. Only sha256(device_code) is kept. Codes are single-use: a poll that
# reports approved/denied deletes the pairing. The poll lives under /fd/auth so
# the refresh cookie (path=/fd/auth) set on its response is the one refresh reads.
#
# Approving mints a lasting session for another device, so lookup/approve want
# more than an access token (which also travels in ?fd_token= URLs and logs):
# the Authorization header plus the approver's own refresh cookie — which the
# browser sends to /fd/auth/pair/* — for a live session of the same user.
#
# Fairness: the public calls are limited per client network (an IPv4 address,
# an IPv6 /64 — one host can own a whole /64), and one network holds at most
# PAIR_MAX_PER_NETWORK pending codes (a new code replaces its oldest), so
# nobody can fill the table and lock every wearer out of pairing.

PAIR_ALPHABET = "BCDFGHJKLMNPQRSTVWXZ"  # 20 consonants: no vowels (no words), no 0/O/1/I
PAIR_CODE_LEN = 8                       # 20^8 ≈ 2.6e10 codes
PAIR_TTL_S = 600
PAIR_INTERVAL_S = 3
PAIR_MAX_PENDING = 10_000              # memory backstop; the per-network cap binds first
PAIR_MAX_PER_NETWORK = 3                # pending codes one client network can hold
PAIR_VERIFICATION_PATH = "/hud/pair"
PAIR_LABEL_MAX = 60
PAIR_UA_MAX = 300

# Rate limits: (count, window seconds).
PAIR_START_LIMIT = (10, 600.0)     # per client network
PAIR_POLL_LIMIT = (120, 60.0)      # per client network (a device polls every 3 s)
PAIR_APPROVER_LIMIT = (30, 60.0)   # per signed-in user, for lookup and approve each


@dataclass
class _Pairing:
    user_code: str          # normalized, 8 chars of PAIR_ALPHABET
    device_hash: str        # sha256(device_code)
    label: str
    user_agent: str
    ip: str
    network: str            # _client_network(ip): the fairness key
    created_at: float       # epoch seconds (_now)
    expires_at: float
    status: str = "pending"  # pending | approved | denied
    approved_by: str = ""    # user id of the approver


# user_code → pairing. Oldest first: one TTL for all, so this is expiry order.
_pairings: OrderedDict[str, _Pairing] = OrderedDict()
_pairings_by_device: dict[str, str] = {}  # sha256(device_code) → user_code
_pairings_by_network: dict[str, list[str]] = {}  # client network → its user codes, oldest first


def _now() -> float:
    """Wall clock for pairing expiry (monkeypatched by tests)."""
    return time.time()


_LIMITER_GC_AT = 5000          # distinct keys before idle ones are swept…
_LIMITER_SWEEP_EVERY_S = 30.0  # …at most this often


class _PairLimiter(_SlidingWindow):
    """A _SlidingWindow keyed by what clients send (networks, user ids). Idle
    keys are swept so a spray of sources can't grow it without bound, but the
    sweep is amortised — only above _LIMITER_GC_AT keys and at most once per
    _LIMITER_SWEEP_EVERY_S — so a flood of live keys never turns every
    request into a scan of all of them."""

    def __init__(self) -> None:
        super().__init__()
        self.swept_at = float("-inf")

    def maybe_sweep(self, window: float) -> None:
        now = time.monotonic()
        if len(self._requests) <= _LIMITER_GC_AT or now - self.swept_at < _LIMITER_SWEEP_EVERY_S:
            return
        self.swept_at = now
        self.sweep(now - window)

    def sweep(self, cutoff: float) -> None:
        for k in [k for k, ts in self._requests.items() if not ts or ts[-1] <= cutoff]:
            del self._requests[k]


_pair_start_limiter = _PairLimiter()
_pair_poll_limiter = _PairLimiter()
_pair_approver_limiter = _PairLimiter()


def _rate_limit(limiter: _PairLimiter, key: str, limit: tuple[int, float]) -> None:
    limiter.maybe_sweep(limit[1])
    if not limiter.check(key, limit[0], limit[1]):
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                            detail="Too many requests — try again in a moment.")


def _require_auth_enabled() -> None:
    if not _fd_auth_enabled():
        raise HTTPException(
            status_code=400,
            detail="Device pairing is unavailable: this Flight Deck runs without accounts (FD_AUTH_ENABLED=false).",
        )


def _client_ip(request: Request) -> str:
    # uvicorn already resolves X-Forwarded-For for trusted local proxies
    # (FORWARDED_ALLOW_IPS); never trust the raw header here.
    return request.client.host if request.client else ""


def _client_network(ip: str) -> str:
    """The fairness key for a client address: an IPv4 address as is (also when
    IPv4-mapped), an IPv6 address's /64 — keying on the full address would give
    a host that owns a /64 2^64 budgets."""
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return ip
    if isinstance(addr, ipaddress.IPv6Address):
        if addr.ipv4_mapped is not None:
            return str(addr.ipv4_mapped)
        return str(ipaddress.IPv6Network((int(addr) >> 64 << 64, 64)))
    return str(addr)


def _clean_text(value: str | None, cap: int) -> str:
    """Drop control/format characters (incl. bidi overrides), collapse
    whitespace, cap the length — the approver reads this to decide."""
    raw = str(value or "")[: cap * 4]
    kept = "".join(" " if ch.isspace() else ch for ch in raw
                   if ch.isspace() or not unicodedata.category(ch).startswith("C"))
    return " ".join(kept.split())[:cap]


def normalize_user_code(code: str | None) -> str:
    """Uppercase and keep only alphabet letters: 'bcdf-ghjk' → 'BCDFGHJK'."""
    return "".join(ch for ch in str(code or "").upper()[:64] if ch in PAIR_ALPHABET)


def format_user_code(code: str) -> str:
    return f"{code[:4]}-{code[4:]}" if len(code) == PAIR_CODE_LEN else code


def _hash_device_code(device_code: str) -> str:
    return hashlib.sha256(device_code.encode("utf-8")).hexdigest()


def _drop_pairing(p: _Pairing) -> None:
    _pairings.pop(p.user_code, None)
    _pairings_by_device.pop(p.device_hash, None)
    codes = _pairings_by_network.get(p.network)
    if codes is not None:
        with suppress(ValueError):
            codes.remove(p.user_code)
        if not codes:
            del _pairings_by_network[p.network]


def _purge_expired(now: float) -> None:
    # Oldest first, so stop at the first live one: no scan of every pairing.
    while _pairings:
        p = next(iter(_pairings.values()))
        if p.expires_at > now:
            break
        _drop_pairing(p)


def _to_replace(network: str) -> list[_Pairing]:
    """The pending codes *network* gives up for a new one: its oldest, so it
    keeps at most PAIR_MAX_PER_NETWORK (approved/denied ones are claimed
    within seconds and don't count)."""
    codes = _pairings_by_network.get(network, ())
    pending = [_pairings[c] for c in codes if _pairings[c].status == "pending"]
    return pending[: max(0, len(pending) - PAIR_MAX_PER_NETWORK + 1)]


def _pending(code: str | None, now: float) -> _Pairing | None:
    p = _pairings.get(normalize_user_code(code))
    if p is None or p.status != "pending" or p.expires_at <= now:
        return None
    return p


_FULL_SIGN_IN_NEEDED = ("Approving a device needs a full sign-in in this browser — "
                        "sign out, sign in again and retry.")


async def _require_browser_session(request: Request, user: dict) -> None:
    """403 unless the request carries the Authorization header (not the
    ?fd_token= fallback) and this browser's refresh cookie for a live session
    of the same *user* — proof of a full sign-in, not just an access token."""
    refresh_token = request.cookies.get(REFRESH_COOKIE)
    if not refresh_token or not request.headers.get("authorization", "").lower().startswith("bearer "):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=_FULL_SIGN_IN_NEEDED)
    db = get_db()
    assert db._db is not None
    async with db._db.execute(
        "SELECT user_id, expires_at FROM user_sessions WHERE refresh_token_hash = ?",
        (hash_token(refresh_token),),
    ) as cur:
        session = await cur.fetchone()
    if (session is None or str(session["user_id"]) != str(user["id"])
            or datetime.fromisoformat(session["expires_at"]) <= datetime.now(UTC)):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=_FULL_SIGN_IN_NEEDED)


class PairStartRequest(BaseModel):
    label: str | None = None


class PairPollRequest(BaseModel):
    device_code: str


class PairApproveRequest(BaseModel):
    user_code: str
    approve: bool


@router.post("/pair/start")
async def pair_start(request: Request, body: PairStartRequest | None = None):
    _require_auth_enabled()
    now = _now()
    _purge_expired(now)
    ip = _client_ip(request)
    network = _client_network(ip)
    _rate_limit(_pair_start_limiter, f"net:{network}", PAIR_START_LIMIT)
    replaced = _to_replace(network)
    if len(_pairings) - len(replaced) >= PAIR_MAX_PENDING:
        # (A refused start keeps the codes it would have replaced.)
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                            detail="Too many pairings in progress — try again in a few minutes.")
    for p in replaced:
        _drop_pairing(p)
    for _ in range(20):
        user_code = "".join(secrets.choice(PAIR_ALPHABET) for _ in range(PAIR_CODE_LEN))
        if user_code not in _pairings:
            break
    else:  # pragma: no cover — 10k live codes out of 2.6e10
        raise HTTPException(status_code=503, detail="Could not allocate a pairing code")
    device_code = secrets.token_urlsafe(32)
    pairing = _Pairing(
        user_code=user_code,
        device_hash=_hash_device_code(device_code),
        label=_clean_text(body.label if body else "", PAIR_LABEL_MAX),
        user_agent=_clean_text(request.headers.get("user-agent", ""), PAIR_UA_MAX),
        ip=ip,
        network=network,
        created_at=now,
        expires_at=now + PAIR_TTL_S,
    )
    _pairings[user_code] = pairing
    _pairings_by_device[pairing.device_hash] = user_code
    _pairings_by_network.setdefault(network, []).append(user_code)
    return {
        "device_code": device_code,
        "user_code": format_user_code(user_code),
        "expires_in": PAIR_TTL_S,
        "interval": PAIR_INTERVAL_S,
        "verification_path": PAIR_VERIFICATION_PATH,
    }


@router.post("/pair/poll")
async def pair_poll(body: PairPollRequest, request: Request, response: Response):
    _require_auth_enabled()
    now = _now()
    _purge_expired(now)
    _rate_limit(_pair_poll_limiter, f"net:{_client_network(_client_ip(request))}", PAIR_POLL_LIMIT)
    code = _pairings_by_device.get(_hash_device_code(body.device_code[:256]))
    p = _pairings.get(code) if code else None
    if p is None or p.expires_at <= now:  # (the purge stops early if the clock stepped back)
        return {"status": "expired"}
    if p.status == "pending":
        return {"status": "pending"}
    # approved / denied: single use. Drop it before any await so two racing
    # polls can't both claim the session.
    _drop_pairing(p)
    if p.status == "denied":
        return {"status": "denied"}
    db = get_db()
    user = await db.get_user_by_id(p.approved_by)
    if not user:
        return {"status": "denied"}
    issued = await _issue_session(db, user, response)
    log.info("device pairing claimed: user=%s label=%r ip=%s", user["id"], p.label, p.ip)
    return {"status": "approved", **issued.model_dump()}


@router.get("/pair/lookup")
async def pair_lookup(request: Request, code: str = Query("", max_length=64),
                      user: dict = Depends(get_current_user)):
    _require_auth_enabled()
    now = _now()
    _purge_expired(now)
    _rate_limit(_pair_approver_limiter, f"lookup:{user['id']}", PAIR_APPROVER_LIMIT)
    await _require_browser_session(request, user)
    p = _pending(code, now)
    if p is None:
        raise HTTPException(status_code=404, detail="Unknown or expired code")
    return {
        "user_code": format_user_code(p.user_code),
        "label": p.label,
        "user_agent": p.user_agent,
        "ip": p.ip,
        "created_at": datetime.fromtimestamp(p.created_at, UTC).isoformat(),
        "expires_in": max(0, int(p.expires_at - now)),
    }


@router.post("/pair/approve")
async def pair_approve(body: PairApproveRequest, request: Request,
                       user: dict = Depends(get_current_user)):
    _require_auth_enabled()
    now = _now()
    _purge_expired(now)
    _rate_limit(_pair_approver_limiter, f"approve:{user['id']}", PAIR_APPROVER_LIMIT)
    await _require_browser_session(request, user)
    p = _pending(body.user_code, now)
    if p is None:
        raise HTTPException(status_code=404, detail="Unknown or expired code")
    if body.approve:
        p.status = "approved"
        p.approved_by = str(user["id"])
    else:
        p.status = "denied"
    log.info("device pairing %s by user=%s label=%r ip=%s", p.status, user["id"], p.label, p.ip)
    return {"ok": True, "status": p.status}
