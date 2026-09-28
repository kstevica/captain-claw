"""Google OAuth endpoints hosted by the Flight Deck backend.

Flight Deck is the single source of truth for Google OAuth credentials and
tokens. It performs the authorization flow, stores the deployment's
``client_id`` / ``client_secret`` in its SQLite ``system_settings`` and each
user's refresh token in that user's ``user_settings``, and exposes
``GET /fd/google/access_token`` so the captain-claw agents THIS deck spawned
(potentially on different ports or hosts) can pull a fresh access token for
their OWNER's account without duplicating the OAuth dance.

Only one OAuth callback URL needs to be registered with Google — this
one — regardless of how many agents consume the tokens.

Only on an auth-enabled deck: with ``FD_AUTH_ENABLED=false`` Flight Deck keeps
no DB, and every route here answers "not available" (``_require_auth_deck``).
"""

from __future__ import annotations

import html
import json
import os
import secrets
import time
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from pydantic import BaseModel

from captain_claw.flight_deck.auth import (
    _LOCAL_USER,
    _fd_auth_enabled,
    get_current_user,
    get_db,
)
from captain_claw.flight_deck.admin_routes import require_admin
from captain_claw.flight_deck.db import FlightDeckDB
from captain_claw.google_oauth import (
    DEFAULT_SCOPES,
    SCOPE_CATALOG,
    GoogleOAuthTokens,
    build_authorization_url,
    exchange_code_for_tokens,
    fetch_user_info,
    generate_pkce_pair,
    refresh_access_token,
    revoke_token,
    sanitize_scopes,
)
from captain_claw.logging import get_logger

log = get_logger(__name__)


_AUTH_OFF_DETAIL = (
    "Google via Flight Deck isn't available on this deck: Flight Deck auth is "
    "disabled (FD_AUTH_ENABLED=false). Enable auth to connect Google."
)
# Machine-readable twin of _AUTH_OFF_DETAIL: an agent that sees it knows it is
# on a single-tenant deck and may keep its own Google credentials (gws), as on
# main — unlike every other refusal, which must fail closed.
AUTH_OFF_HEADER = "X-FD-Google-Unavailable"
AUTH_OFF_VALUE = "auth-disabled"


def _require_auth_deck(request: Request) -> None:
    """Router-wide gate: Google via Flight Deck needs an auth-enabled deck.

    An auth-disabled deck keeps no DB (``get_db()`` would assert) and can't tell
    its users — or a web page reaching its loopback — apart, so every route says
    so instead of failing with a 500. The agent endpoints answer 403, which the
    agent's Google tools surface verbatim (google_oauth_manager); the UI and
    popup routes 503.
    """
    if _fd_auth_enabled():
        return
    agent_route = request.url.path.rsplit("/", 1)[-1] in ("access_token", "credentials")
    raise HTTPException(
        status_code=403 if agent_route else 503,
        detail=_AUTH_OFF_DETAIL,
        headers={AUTH_OFF_HEADER: AUTH_OFF_VALUE},
    )


router = APIRouter(
    prefix="/fd/google",
    tags=["google-oauth"],
    dependencies=[Depends(_require_auth_deck)],
)


# ── OAuth client: user-supplied only ────────────────────────────────
#
# Captain Claw no longer ships with baked-in Google OAuth credentials.
# Every deployment MUST configure its own ``client_id`` /
# ``client_secret`` in Flight Deck's Connections page (or via the
# ``/fd/google/config`` endpoint). This avoids bundling secrets in the
# source tree / distribution and lets each user own their own Google
# Cloud project, consent screen, and scope verification posture.
#
# Default scopes (granted during the OAuth flow when present on the
# consent screen) are defined in ``captain_claw.google_oauth.DEFAULT_SCOPES``
# and include ``cloud-platform`` so Vertex AI / Gemini is available
# whenever the user also supplies a ``project_id``.


# ── system_settings keys ────────────────────────────────────────────

_K_CLIENT_ID = "google_oauth:client_id"
_K_CLIENT_SECRET = "google_oauth:client_secret"
_K_PROJECT_ID = "google_oauth:project_id"
_K_LOCATION = "google_oauth:location"
_K_SCOPES = "google_oauth:scopes"
_K_TOKENS = "google_oauth:tokens"
_K_USER = "google_oauth:user"
# Legacy / reserved — kept for backwards compatibility when reading
# older databases. Only "custom" is supported going forward.
_K_TOKEN_MODE = "google_oauth:token_mode"


_GMAIL_SCOPE_LABELS = {s["scope"]: s["label"] for s in SCOPE_CATALOG}


def _label_scopes(granted: list[str]) -> list[dict[str, str]]:
    return [{"scope": s, "label": _GMAIL_SCOPE_LABELS.get(s, s)} for s in granted]


# ── in-memory PKCE state ────────────────────────────────────────────
#
# PKCE verifiers are short-lived (≤ 10 min) and only needed between the
# login-redirect and the callback on the same Flight Deck process, so a
# plain dict is fine. No need to persist them.
#
# Each pending flow is also bound to the browser that started it: /login sets
# a short-lived HttpOnly cookie carrying a nonce stored in the entry, and
# /callback refuses a state whose cookie doesn't match. Without it the state is
# a bearer ticket — user B could start /login, forward the Google URL to user A,
# and A's consent would land A's Google account in B's FD account (login-CSRF;
# PKCE doesn't help, the verifier stays server-side).
#
# /login itself is bound the same way. WHO is connecting comes from a connect
# ticket, not from a JWT in the URL: the SPA mints one with an authenticated
# POST /connect-ticket, which also sets it as an HttpOnly SameSite=Strict
# cookie, and /login takes it only from the browser holding that cookie. A
# forwarded /login link (B's ticket, or formerly B's ?fd_token=) therefore
# can't start a flow in A's browser that binds A's consent to B's account.

_pending_oauth: dict[str, dict[str, Any]] = {}

_PENDING_TTL = 600  # seconds — also the state cookie's max-age
_STATE_COOKIE = "fd_google_oauth"
_STATE_COOKIE_PATH = "/fd/google"  # both cookies: /connect-ticket, /login, /callback

_connect_tickets: dict[str, dict[str, Any]] = {}  # ticket → {owner, ts}
_TICKET_TTL = 120  # seconds — also the ticket cookie's max-age
_TICKET_COOKIE = "fd_google_ticket"


def _purge_stale_pending() -> None:
    cutoff = time.time() - _PENDING_TTL
    stale = [k for k, v in _pending_oauth.items() if v.get("ts", 0) < cutoff]
    for k in stale:
        _pending_oauth.pop(k, None)


def _purge_stale_tickets() -> None:
    cutoff = time.time() - _TICKET_TTL
    for k in [k for k, v in _connect_tickets.items() if v.get("ts", 0) < cutoff]:
        _connect_tickets.pop(k, None)


def _same_secret(expected: str, got: str) -> bool:
    """Constant-time equality of two non-empty strings (any characters — a
    cookie or query value is caller-chosen)."""
    return bool(expected and got) and secrets.compare_digest(
        expected.encode("utf-8"), got.encode("utf-8")
    )


# ── storage helpers ─────────────────────────────────────────────────


async def _load_user_config(db: FlightDeckDB) -> dict[str, Any]:
    """Return the user-supplied (custom) OAuth config from system_settings."""
    raw_scopes = (await db.get_system_setting(_K_SCOPES)) or ""
    scopes: list[str]
    if raw_scopes:
        try:
            loaded = json.loads(raw_scopes)
            if isinstance(loaded, list):
                scopes = sanitize_scopes([str(s) for s in loaded])
            else:
                scopes = list(DEFAULT_SCOPES)
        except Exception:
            scopes = list(DEFAULT_SCOPES)
    else:
        scopes = list(DEFAULT_SCOPES)
    return {
        "client_id": (await db.get_system_setting(_K_CLIENT_ID)) or "",
        "client_secret": (await db.get_system_setting(_K_CLIENT_SECRET)) or "",
        "project_id": (await db.get_system_setting(_K_PROJECT_ID)) or "",
        "location": (await db.get_system_setting(_K_LOCATION)) or "us-central1",
        "scopes": scopes,
    }


async def _effective_oauth(db: FlightDeckDB) -> dict[str, Any] | None:
    """Resolve the *active* OAuth client + scopes.

    Returns ``None`` when the user hasn't saved a ``client_id`` /
    ``client_secret`` pair yet — there is no bundled fallback. Callers
    must surface a "not configured" error to the user when this
    happens.
    """
    user = await _load_user_config(db)
    if not (user["client_id"] and user["client_secret"]):
        return None
    scopes = sanitize_scopes(user.get("scopes"))
    has_cloud = "https://www.googleapis.com/auth/cloud-platform" in scopes
    return {
        "mode": "custom",
        "client_id": user["client_id"],
        "client_secret": user["client_secret"],
        "project_id": user["project_id"],
        "location": user["location"],
        "scopes": scopes,
        "supports_vertex": bool(user["project_id"]) and has_cloud,
    }


# ── per-user identity ───────────────────────────────────────────────
#
# The OAuth *client* (client_id/secret/project/scopes) is one per deployment
# and stays in system_settings. The *tokens* — who is actually signed in — are
# per Flight Deck user, in user_settings. A single Google account shared across
# every FD user was the multi-tenant hole this closes.

_primary_owner_cache: dict[str, Any] = {"id": None, "at": 0.0}
_PRIMARY_OWNER_TTL = 60.0  # seconds — an admin can be demoted/deleted


async def _primary_owner(db: FlightDeckDB) -> str:
    """The deployment's primary owner (single user, or oldest admin).

    Only the account that transparently inherits any legacy global tokens —
    NEVER a fallback identity for a caller that can't be pinned to a user (that
    handed the admin's Google to whoever asked). With auth disabled the deck has
    one tenant, the local user, so that is the owner. Cached briefly.
    """
    if not _fd_auth_enabled():
        return _LOCAL_USER["id"]
    now = time.time()
    if (_primary_owner_cache["id"] is None
            or now - float(_primary_owner_cache.get("at") or 0.0) > _PRIMARY_OWNER_TTL):
        try:
            from captain_claw.flight_deck.server import _resolve_primary_owner

            owner = (await _resolve_primary_owner(db)) or _LOCAL_USER["id"]
        except Exception:
            owner = _LOCAL_USER["id"]
        _primary_owner_cache.update(id=owner, at=now)
    return _primary_owner_cache["id"]


def _effective_owner(user: dict | None) -> str:
    """Owner for a dashboard call: the logged-in user, else the local user
    (auth-disabled deployments have no login)."""
    return str((user or {}).get("id") or _LOCAL_USER["id"])


def _tenant(user_id: str) -> str:
    """The Google owner for *user_id*. With auth disabled the deck would have
    exactly one tenant, the local user — dormant: Google is refused on such a
    deck (``_require_auth_deck``)."""
    return user_id if _fd_auth_enabled() else _LOCAL_USER["id"]


def _is_local_tenant(user_id: str) -> bool:
    return not _fd_auth_enabled() and user_id == _LOCAL_USER["id"]


async def _load_tokens(db: FlightDeckDB, user_id: str) -> GoogleOAuthTokens | None:
    """This user's stored tokens.

    Falls back to the legacy deployment-wide key for the primary owner only, so
    a deployment that connected Google before per-user storage keeps working
    with no migration step — the fallback is retired the moment that user
    reconnects or a refresh rewrites the tokens per-user.
    """
    raw = await db.get_setting(user_id, _K_TOKENS)
    if not raw and user_id == await _primary_owner(db):
        raw = await db.get_system_setting(_K_TOKENS)
    if not raw:
        return None
    try:
        return GoogleOAuthTokens.from_dict(json.loads(raw))
    except Exception as exc:
        log.warning("Failed to deserialize Google OAuth tokens: %s", exc)
        return None


async def get_valid_google_access_token(user_id: str) -> str | None:
    """FD-side: a currently-valid Google access token for *user_id* (refreshed if
    needed), or None when that user hasn't connected Google. Used by the
    event-spine pollers (#2) and Drive export to call Google directly from Flight
    Deck. *user_id* is required and there is deliberately no default owner: an
    unattributed call gets nothing rather than the primary owner's account."""
    if not user_id or not _fd_auth_enabled():  # see _require_auth_deck
        return None
    try:
        db = get_db()
        client = await _token_client(db)
        if not client:
            return None
        uid = _tenant(user_id)
        tokens = await _load_tokens(db, uid)
        if not tokens or not tokens.refresh_token:
            return None
        tokens = await _refresh_if_needed(db, uid, client, tokens)
        return tokens.access_token if tokens else None
    except Exception:
        return None


async def is_google_connected(user_id: str) -> bool:
    """FD-side, no network: has *user_id* connected their own Google account on
    this deck? Gates per-user pollers that need Google."""
    if not user_id or not _fd_auth_enabled():  # see _require_auth_deck
        return False
    try:
        db = get_db()
        if not await _token_client(db):
            return False
        tokens = await _load_tokens(db, _tenant(user_id))
        return bool(tokens and tokens.refresh_token)
    except Exception:
        return False


async def _store_tokens(db: FlightDeckDB, user_id: str, tokens: GoogleOAuthTokens) -> None:
    blob = json.dumps(tokens.to_dict(), ensure_ascii=True)
    if _is_local_tenant(user_id):
        # Auth disabled: the synthetic local user has no users row (user_settings
        # has an FK to it), so the deck's single tenant keeps its connection in
        # the deployment-wide keys — the ones _load_tokens falls back to for it.
        await db.set_system_setting(_K_TOKENS, blob)
        return
    await db.set_settings(user_id, {_K_TOKENS: blob})


async def _load_user(db: FlightDeckDB, user_id: str) -> dict[str, Any] | None:
    raw = await db.get_setting(user_id, _K_USER)
    if not raw and user_id == await _primary_owner(db):
        raw = await db.get_system_setting(_K_USER)
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception:
        return None


async def _store_user(db: FlightDeckDB, user_id: str, user: dict[str, Any]) -> None:
    blob = json.dumps(user, ensure_ascii=True)
    if _is_local_tenant(user_id):
        await db.set_system_setting(_K_USER, blob)  # see _store_tokens
        return
    await db.set_settings(user_id, {_K_USER: blob})


def _same_google_account(a: dict[str, Any] | None, b: dict[str, Any] | None) -> bool:
    """Whether two stored userinfo blobs are the same Google account — by the
    stable ``sub`` only, which both must have. /callback always stores Google's
    userinfo (which carries ``sub``); an email is just a label, so matching on
    it would let a blob without ``sub`` claim someone else's account."""
    sub_a = str((a or {}).get("sub") or "")
    sub_b = str((b or {}).get("sub") or "")
    return bool(sub_a and sub_b) and sub_a == sub_b


async def _account_held_by_another_user(db: FlightDeckDB, user_id: str) -> bool:
    """Does another user on this deck still hold tokens for the same Google
    account as *user_id*? Unknown identity (no ``sub``) → False (revoke as
    before)."""
    mine = await _load_user(db, user_id)
    if not mine or not mine.get("sub"):
        return False
    try:
        users = await db.list_users(limit=10000)
    except Exception:
        users = []
    for u in users:
        other = str(u.get("id") or "")
        if not other or other == user_id:
            continue
        tokens = await _load_tokens(db, other)
        if tokens and tokens.refresh_token and _same_google_account(
            mine, await _load_user(db, other)
        ):
            return True
    return False


async def _token_client(db: FlightDeckDB) -> dict[str, Any] | None:
    """Resolve the OAuth client that minted the currently-stored tokens.

    Refreshing a token requires the same client_id/secret pair that
    issued it. With bundled credentials gone, the only valid source is
    the user-supplied config.
    """
    user_cfg = await _load_user_config(db)
    if not user_cfg["client_id"] or not user_cfg["client_secret"]:
        return None
    scopes = sanitize_scopes(user_cfg.get("scopes"))
    has_cloud = "https://www.googleapis.com/auth/cloud-platform" in scopes
    return {
        "mode": "custom",
        "client_id": user_cfg["client_id"],
        "client_secret": user_cfg["client_secret"],
        "project_id": user_cfg["project_id"],
        "location": user_cfg["location"] or "us-central1",
        "scopes": scopes,
        "supports_vertex": bool(user_cfg["project_id"]) and has_cloud,
    }


async def _clear_oauth_state(db: FlightDeckDB, user_id: str) -> None:
    """Disconnect this user. Clears their per-user tokens, and — when they are
    the primary owner — the legacy global keys too, so the fallback can't
    silently re-connect them after a logout."""
    await db.delete_setting(user_id, _K_TOKENS)
    await db.delete_setting(user_id, _K_USER)
    if user_id == await _primary_owner(db):
        await db.set_system_setting(_K_TOKENS, "")
        await db.set_system_setting(_K_USER, "")
        await db.set_system_setting(_K_TOKEN_MODE, "")


async def _clear_all_oauth_state(db: FlightDeckDB) -> None:
    """Disconnect EVERY user. Rotating the deployment's OAuth client or changing
    its scopes invalidates all tokens — a refresh token is bound to the client
    that minted it, and a scope change needs fresh consent — so this can't be
    per-user."""
    try:
        users = await db.list_users(limit=10000)
    except Exception:
        users = []
    for u in users:
        uid = str(u.get("id", ""))
        if uid:
            await db.delete_setting(uid, _K_TOKENS)
            await db.delete_setting(uid, _K_USER)
    await db.set_system_setting(_K_TOKENS, "")
    await db.set_system_setting(_K_USER, "")
    await db.set_system_setting(_K_TOKEN_MODE, "")


# ── redirect URI ────────────────────────────────────────────────────


def _redirect_uri(request: Request) -> str:
    """Build the redirect URI that Google will bounce the user back to.

    Uses the request's own host so it works for dev (localhost:25080),
    packaged desktop app, or behind a reverse proxy. This must exactly
    match the URI registered in Google Cloud Console.
    """
    scheme = request.url.scheme
    host = request.headers.get("host") or request.url.netloc
    # If FD_PUBLIC_URL is set, prefer it (handles proxied deployments).
    public = _public_url()
    if public:
        return f"{public}/fd/google/callback"
    return f"{scheme}://{host}/fd/google/callback"


def _public_url() -> str:
    return os.environ.get("FD_PUBLIC_URL", "").strip().rstrip("/")


def _browser_scheme(request: Request) -> str:
    """The scheme the browser used for THIS request (a TLS proxy's
    X-Forwarded-Proto, else the request's own)."""
    proto = request.headers.get("x-forwarded-proto", "").split(",")[0].strip().lower()
    return proto or str(request.url.scheme).lower()


def _state_cookie_secure(request: Request) -> bool:
    """Secure exactly when the browser reached this request over https (the
    state cookie and the connect-ticket cookie alike).

    Decided by the scheme the browser actually used — never FD_PUBLIC_URL's or
    the deck's cookie policy — because a Secure cookie set over plain http is
    dropped and the callback could then never match it. Set over https without
    a proxy's X-Forwarded-Proto it goes out non-Secure, which still works
    (Secure is hardening only; both cookies are HttpOnly, path-scoped and live
    minutes).
    """
    return _browser_scheme(request) == "https"


# ── models ──────────────────────────────────────────────────────────


class GoogleConfigUpdate(BaseModel):
    # Set ``clear=True`` to wipe all stored credentials + tokens. Other
    # fields: any field that is ``None`` is left unchanged; any non-empty
    # string overwrites the existing value. ``mode`` is accepted for
    # backwards compatibility but ignored.
    mode: str | None = None
    clear: bool | None = None
    client_id: str | None = None
    client_secret: str | None = None
    project_id: str | None = None
    location: str | None = None
    # ``scopes`` is the *full* desired list — pass the current list with
    # items added/removed. Unknown or duplicate entries are silently
    # dropped server-side. ``None`` leaves the stored list unchanged.
    scopes: list[str] | None = None


# ── status / config endpoints ───────────────────────────────────────


@router.get("/status")
async def google_status(
    request: Request,
    _user: dict = Depends(get_current_user),
) -> dict[str, Any]:
    """Return Google OAuth connection status for the UI (this user)."""
    db = get_db()
    uid = _effective_owner(_user)
    eff = await _effective_oauth(db)

    tokens = await _load_tokens(db, uid)
    connected = bool(eff and tokens and tokens.refresh_token)
    user = await _load_user(db, uid) if connected else None
    scopes = tokens.scope.split() if tokens and tokens.scope else []

    return {
        "configured": eff is not None,
        "mode": "custom",
        "supports_vertex": bool(eff and eff["supports_vertex"]),
        "connected": connected,
        "user": user,
        "granted_scopes": _label_scopes(scopes),
        "redirect_uri": _redirect_uri(request),
    }


@router.get("/probe")
async def google_probe(
    _user: dict = Depends(get_current_user),
) -> dict[str, Any]:  # noqa: C901
    """Live read-only health probe of the Google connection.

    Unlike ``/status`` (which only checks that a refresh token is stored),
    this refreshes the access token if needed and makes a real read-only
    Google API call (userinfo). Drives the Connections traffic light:

    - ``configured`` false / no tokens  → not connected (red)
    - tokens present but the call fails  → connected-but-failing (yellow)
    - userinfo succeeds                  → healthy (green)
    """
    db = get_db()
    uid = _effective_owner(_user)
    client = await _token_client(db)
    if not client:
        return {"configured": False, "connected": False, "ok": False,
                "error": "Google OAuth not configured"}
    tokens = await _load_tokens(db, uid)
    if not tokens or not tokens.refresh_token:
        return {"configured": True, "connected": False, "ok": False,
                "error": "Not connected — no stored tokens"}
    try:
        tokens = await _refresh_if_needed(db, uid, client, tokens)
    except Exception as exc:  # defensive — _refresh_if_needed swallows most
        return {"configured": True, "connected": True, "ok": False,
                "error": f"Token refresh failed: {exc}"}
    if not tokens:
        return {"configured": True, "connected": True, "ok": False,
                "error": "Could not refresh access token"}
    try:
        info = await fetch_user_info(tokens.access_token)
    except Exception as exc:
        return {"configured": True, "connected": True, "ok": False,
                "error": f"Read-only API call failed: {exc}"}
    return {
        "configured": True,
        "connected": True,
        "ok": True,
        "email": info.get("email", ""),
    }


@router.get("/config")
async def google_config_get(
    request: Request,
    _user: dict = Depends(get_current_user),
) -> dict[str, Any]:
    """Return the stored Google OAuth configuration (never returns the secret)."""
    db = get_db()
    user = await _load_user_config(db)
    return {
        "mode": "custom",
        "client_id": user["client_id"],
        "client_id_set": bool(user["client_id"]),
        "client_secret_set": bool(user["client_secret"]),
        "project_id": user["project_id"],
        "location": user["location"] or "us-central1",
        "scopes": user["scopes"],
        "default_scopes": list(DEFAULT_SCOPES),
        "scope_catalog": SCOPE_CATALOG,
        "redirect_uri": _redirect_uri(request),
    }


@router.get("/scope_catalog")
async def google_scope_catalog(
    _user: dict = Depends(get_current_user),
) -> dict[str, Any]:
    """Return the catalogue of selectable OAuth scopes for the UI."""
    return {"scope_catalog": SCOPE_CATALOG, "default_scopes": list(DEFAULT_SCOPES)}


@router.post("/config")
async def google_config_post(
    body: GoogleConfigUpdate,
    _admin: dict = Depends(require_admin),
) -> dict[str, Any]:
    """Save Google OAuth credentials (admin only).

    This writes the deployment-wide OAuth *client* registration and, with
    ``clear=True``, wipes every user's stored Google tokens — so it is gated to
    admins. Previously any authenticated user could rotate the client or wipe
    all connections.

    - ``clear=True``: wipe saved credentials AND any stored tokens. The
      Google connection becomes "not configured" until the user enters
      a new ``client_id`` / ``client_secret``.
    - Otherwise: any field that is ``None`` is left unchanged. Any field
      that is a non-empty string overwrites the existing value. If the
      ``client_id`` changes, stored tokens are invalidated (the refresh
      token is bound to the OAuth client that minted it).
    """
    db = get_db()

    if body.clear:
        await db.set_system_setting(_K_CLIENT_ID, "")
        await db.set_system_setting(_K_CLIENT_SECRET, "")
        await db.set_system_setting(_K_PROJECT_ID, "")
        await db.set_system_setting(_K_SCOPES, "")
        await _clear_all_oauth_state(db)
        return {"ok": True, "cleared": True}

    def _clean(v: str | None) -> str | None:
        if v is None:
            return None
        s = v.strip()
        return s if s else None

    client_id = _clean(body.client_id)
    client_secret = _clean(body.client_secret)
    project_id = _clean(body.project_id)
    location = _clean(body.location)

    # If the client_id rotates, wipe stored tokens — the refresh token
    # was minted by the previous OAuth client and will no longer work.
    eff_before = await _effective_oauth(db)
    prev_client_id = eff_before["client_id"] if eff_before else ""
    will_change_client = client_id is not None and client_id != prev_client_id

    if client_id is not None:
        await db.set_system_setting(_K_CLIENT_ID, client_id)
    if client_secret is not None:
        await db.set_system_setting(_K_CLIENT_SECRET, client_secret)
    if project_id is not None:
        await db.set_system_setting(_K_PROJECT_ID, project_id)
    if location is not None:
        await db.set_system_setting(_K_LOCATION, location)

    scopes_changed = False
    if body.scopes is not None:
        new_scopes = sanitize_scopes(body.scopes)
        prev_scopes = eff_before["scopes"] if eff_before else list(DEFAULT_SCOPES)
        if set(new_scopes) != set(prev_scopes):
            scopes_changed = True
        await db.set_system_setting(_K_SCOPES, json.dumps(new_scopes))

    # Both rotating the client and changing the scope set invalidate
    # the stored tokens — the refresh token is bound to the client that
    # minted it, and scope changes require a fresh consent.
    if will_change_client or scopes_changed:
        await _clear_all_oauth_state(db)

    return {"ok": True, "mode": "custom", "scopes_changed": scopes_changed}


# ── login / callback / logout ───────────────────────────────────────


@router.post("/connect-ticket")
async def google_connect_ticket(
    request: Request,
    _user: dict = Depends(get_current_user),
) -> JSONResponse:
    """Mint a single-use ticket that lets THIS browser start /login for the
    calling user (see ``_pending_oauth``).

    The SPA calls this with its session, then points the popup at
    ``/login?ticket=<ticket>``. The ticket is also set as an HttpOnly,
    SameSite=Strict cookie, and /login wants both — so a /login link forwarded
    to someone else carries a ticket their browser doesn't hold. It lives
    ``_TICKET_TTL`` seconds, and a newer ticket replaces the user's older ones
    (the cookie only ever holds the newest anyway).
    """
    owner = str(_user.get("id") or "")
    if not owner:
        raise HTTPException(status_code=401, detail="Not authenticated")
    _purge_stale_tickets()
    for k in [k for k, v in _connect_tickets.items() if v.get("owner") == owner]:
        _connect_tickets.pop(k, None)
    ticket = secrets.token_urlsafe(32)
    _connect_tickets[ticket] = {"owner": owner, "ts": time.time()}
    resp = JSONResponse({"ticket": ticket}, headers={"Cache-Control": "no-store"})
    resp.set_cookie(
        key=_TICKET_COOKIE,
        value=ticket,
        max_age=_TICKET_TTL,
        httponly=True,
        samesite="strict",  # the popup's /login is navigated by FD's own tab
        path=_STATE_COOKIE_PATH,
        secure=_state_cookie_secure(request),
    )
    return resp


def _take_connect_ticket(request: Request) -> str:
    """Consume ``?ticket=`` and return its user id — or "" when it is unknown,
    expired, already used, or not the ticket this browser's cookie holds. Any
    ticket presented is spent, match or not."""
    ticket = request.query_params.get("ticket", "")
    entry = _connect_tickets.pop(ticket, None) if ticket else None
    if not entry or time.time() - float(entry.get("ts") or 0) > _TICKET_TTL:
        return ""
    if not _same_secret(ticket, request.cookies.get(_TICKET_COOKIE, "")):
        log.warning("Google connect refused: ticket not held by this browser")
        return ""
    return str(entry.get("owner") or "")


@router.get("/login")
async def google_login(request: Request) -> Response:
    """Start the OAuth flow by redirecting to Google's consent screen.

    Opened as a popup from the Flight Deck UI, at ``?ticket=<ticket>`` from
    ``POST /connect-ticket``. A popup can't carry an ``Authorization`` header;
    the ticket says *which* user is connecting, and only in the browser the
    ticket was issued to (its cookie). That user id is stashed in the PKCE
    state so ``/callback`` stores the tokens against the right account — the
    crux of per-user connections. A JWT in the URL is no longer accepted.

    A missing / expired / used / foreign ticket is refused (401) — never bound
    to the primary owner. No origin pre-check: the state cookie lands on the
    host the BROWSER is on (whatever Host a proxy forwards), and a flow started
    off FD_PUBLIC_URL's origin simply can't complete — /callback's "not
    started in this browser" page names the address to use.
    """
    db = get_db()
    eff = await _effective_oauth(db)
    if not eff:
        return HTMLResponse(
            "<h3>Google OAuth not configured</h3>"
            "<p>Enter your Client ID and Client Secret on the Connections "
            "page first, then click Connect Google again.</p>",
            status_code=400,
        )

    owner = _take_connect_ticket(request)
    fd_user = await db.get_user_by_id(owner) if owner else None
    if not fd_user:
        return _callback_html(
            ok=False,
            title="Google sign-in link expired",
            detail=(
                "This link is expired, already used, or was opened outside the "
                "Flight Deck tab that created it. Reload Flight Deck and click "
                "Connect again."
            ),
            status_code=401,
        )

    _purge_stale_pending()

    state = secrets.token_urlsafe(32)
    browser = secrets.token_urlsafe(32)
    verifier, challenge = generate_pkce_pair()
    _pending_oauth[state] = {
        "verifier": verifier,
        "ts": time.time(),
        "owner": owner,
        # Named on the callback page, so a mis-bind is visible.
        "fd_user": str(fd_user.get("email") or fd_user.get("display_name") or owner),
        "browser": browser,
    }

    auth_url = build_authorization_url(
        client_id=eff["client_id"],
        redirect_uri=_redirect_uri(request),
        scopes=eff["scopes"],
        state=state,
        code_challenge=challenge,
    )
    resp = RedirectResponse(auth_url, status_code=302)
    secure = _state_cookie_secure(request)
    resp.set_cookie(
        key=_STATE_COOKIE,
        value=browser,
        max_age=_PENDING_TTL,
        httponly=True,
        samesite="lax",  # sent on Google's top-level redirect back to /callback
        path=_STATE_COOKIE_PATH,
        secure=secure,
    )
    resp.delete_cookie(  # spent
        _TICKET_COOKIE, path=_STATE_COOKIE_PATH, secure=secure, httponly=True,
        samesite="strict",
    )
    return resp


def _started_in_this_browser(request: Request, pending: dict[str, Any]) -> bool:
    return _same_secret(str(pending.get("browser") or ""), request.cookies.get(_STATE_COOKIE, ""))


@router.get("/callback")
async def google_callback(request: Request) -> HTMLResponse:
    """Handle Google's redirect, exchange the code, store tokens.

    The state is single-use, expires with the cookie, and must come back in the
    browser that started it (the cookie /login set) — see ``_pending_oauth``.
    """
    resp = await _complete_callback(request)
    resp.delete_cookie(
        _STATE_COOKIE,
        path=_STATE_COOKIE_PATH,
        secure=_state_cookie_secure(request),
        httponly=True,
        samesite="lax",
    )
    return resp


async def _complete_callback(request: Request) -> HTMLResponse:
    state = request.query_params.get("state", "")
    pending = _pending_oauth.pop(state, None) if state else None
    if pending and time.time() - float(pending.get("ts") or 0) > _PENDING_TTL:
        pending = None

    error = request.query_params.get("error")
    if error:
        desc = request.query_params.get("error_description", error)
        return _callback_html(ok=False, title="OAuth error", detail=desc)

    code = request.query_params.get("code", "")
    if not code or not state:
        return _callback_html(ok=False, title="Missing code or state")

    if not pending:
        return _callback_html(ok=False, title="Invalid or expired state, please try again.")

    if not _started_in_this_browser(request, pending):
        log.warning("Google OAuth callback refused: state not started in this browser")
        return _callback_html(
            ok=False,
            title="Sign-in not started in this browser",
            detail=(
                "This Google sign-in was started from a different browser (or "
                "its sign-in cookie was blocked), so nothing was saved. Open "
                f"Flight Deck at {_redirect_uri(request).rsplit('/fd/', 1)[0]} "
                "in this browser and click Connect Google again."
            ),
            status_code=400,
        )

    owner = str(pending.get("owner") or "")
    if not owner:  # /login always records one; never guess an owner here
        return _callback_html(ok=False, title="Invalid or expired state, please try again.")

    db = get_db()
    user_cfg = await _load_user_config(db)
    if not user_cfg["client_id"] or not user_cfg["client_secret"]:
        return _callback_html(
            ok=False,
            title="Google OAuth credentials were removed mid-flow.",
        )
    client_id = user_cfg["client_id"]
    client_secret = user_cfg["client_secret"]

    try:
        tokens = await exchange_code_for_tokens(
            code=code,
            client_id=client_id,
            client_secret=client_secret,
            redirect_uri=_redirect_uri(request),
            code_verifier=pending["verifier"],
        )
    except Exception as exc:
        log.error("Google OAuth token exchange failed: %s", exc)
        return _callback_html(ok=False, title="Token exchange failed", detail=str(exc))

    try:
        user = await fetch_user_info(tokens.access_token)
    except Exception as exc:
        log.warning("Failed to fetch Google user info: %s", exc)
        user = {}

    await _store_tokens(db, owner, tokens)
    await db.set_system_setting(_K_TOKEN_MODE, "custom")
    if user:
        await _store_user(db, owner, user)

    return _callback_html(
        ok=True,
        title="Connected",
        detail=(
            f"Linked {user.get('email') or 'your Google account'} to Flight Deck "
            f"user {pending.get('fd_user') or owner}"
        ),
        email=user.get("email", ""),
    )


@router.post("/logout")
async def google_logout(_user: dict = Depends(get_current_user)) -> dict[str, Any]:
    """Revoke this user's tokens and clear their stored Google OAuth state.

    Google revokes a whole (account, client) grant, so when another user on
    this deck connected the SAME Google account under the shared client, the
    Google-side revoke is skipped — it would kill their connection too. This
    user's local state is cleared either way.
    """
    db = get_db()
    uid = _effective_owner(_user)
    tokens = await _load_tokens(db, uid)
    if tokens and not await _account_held_by_another_user(db, uid):
        if tokens.refresh_token:
            await revoke_token(tokens.refresh_token)
        elif tokens.access_token:
            await revoke_token(tokens.access_token)
    await _clear_oauth_state(db, uid)
    return {"disconnected": True}


# ── agent-facing endpoint ───────────────────────────────────────────


def _authorize_agent_call(request: Request) -> None:
    """Gate /access_token and /credentials for captain-claw agents.

    The same transport rule as the agent-route guard in ``server``
    (``_agent_caller_ok``): this deck's own agent secret in ``X-Agent-Secret``
    (``FD_AGENT_SHARED_SECRET`` or the per-deck ``agent_secret`` file), or a
    loopback caller — except under ``FD_LOCKDOWN``, where the secret is
    mandatory even from loopback (a same-host TLS proxy would otherwise launder
    remote callers into "loopback").

    Loopback stays trusted otherwise: FD-spawned agents send the secret only
    when ``FD_AGENT_SHARED_SECRET`` is set in their (inherited) env. This only
    proves the caller is *an* agent — never *which* one; that's
    :func:`_agent_owner`, which fails closed.
    """
    from captain_claw.flight_deck.server import _agent_caller_ok

    if not _agent_caller_ok(request):
        raise HTTPException(
            status_code=401,
            detail=(
                "Unauthorized agent call — this Flight Deck requires its agent "
                "secret (X-Agent-Secret) from this caller"
            ),
        )


def _is_browser_request(request: Request) -> bool:
    """Browsers stamp every fetch with ``Origin`` (cross-origin) and/or the
    ``Sec-Fetch-*`` metadata headers; captain-claw agents (httpx) send none."""
    names = {str(k).lower() for k in request.headers.keys()}
    return "origin" in names or any(n.startswith("sec-fetch-") for n in names)


async def _sole_user(db: FlightDeckDB) -> str:
    """The deck's only user when exactly one exists, else ""."""
    try:
        if await db.count_users() != 1:
            return ""
        users = await db.list_users(limit=1)
    except Exception:
        return ""
    return str(users[0].get("id") or "") if users else ""


async def _agent_owner(request: Request) -> str:
    """Which user's Google connection an agent call should use, or 403.

    Resolved ONLY from this deck's own records, never from the agent's word:

    * Auth disabled → 403 (``_require_auth_deck`` refuses first; this keeps a
      direct call from ever handing out a desktop deck's tokens).
    * A browser request (``Origin`` / ``Sec-Fetch-*``) is refused outright — a
      web page must never read a token, refresh token or client secret.
    * ``X-Agent-Auth`` — the per-agent ``web_auth`` token FD minted at spawn —
      must be one THIS deck issued (its process registry, or a container with
      its deck label; see ``server._resolve_agent_identity_by_auth``). Missing
      or unknown → 403: loopback and the shared secret prove *an* agent, not
      whose, and another deck's agent on this host presents a token this deck
      never issued. (No source-port rung: the request's client port is the
      caller's ephemeral outbound port.)
    * A recorded owner must still be a user of this deck. A token this deck
      issued with NO recorded owner (or the auth-off deck's synthetic ``local``
      one; pre-owner records) maps to the only user of a single-user deck, else
      a distinct 403 telling the user to respawn — never the primary owner.
    """
    if not _fd_auth_enabled():
        raise HTTPException(status_code=403, detail=_AUTH_OFF_DETAIL)
    if _is_browser_request(request):
        raise HTTPException(
            status_code=403,
            detail="This endpoint is for Flight Deck agents, not browsers",
        )
    token = request.headers.get("X-Agent-Auth", "")
    if not token:
        raise HTTPException(
            status_code=403,
            detail=(
                "Flight Deck can't identify this agent (no X-Agent-Auth) — only "
                "agents spawned by this Flight Deck can use its Google "
                "connections; respawn it from Flight Deck"
            ),
        )
    try:
        from captain_claw.flight_deck.server import _resolve_agent_identity_by_auth

        matched, owner = _resolve_agent_identity_by_auth(token)
    except Exception as exc:  # fail closed; never log the token
        log.warning("Agent identity lookup failed: %s", type(exc).__name__)
        matched, owner = False, ""
    if not matched:
        raise HTTPException(
            status_code=403,
            detail="Unknown agent — it was not spawned by this Flight Deck",
        )

    db = get_db()
    owner = str(owner or "")
    if owner and await db.get_user_by_id(owner):
        return owner
    # "local" is the synthetic tenant an auth-disabled deck records — no owner
    # as far as this (now auth-enabled) deck's users go.
    if owner and owner != _LOCAL_USER["id"]:
        raise HTTPException(
            status_code=403,
            detail="This agent's owner is no longer a user of this Flight Deck",
        )
    sole = await _sole_user(db)
    if sole:
        return sole
    raise HTTPException(
        status_code=403,
        detail=(
            "Flight Deck can't attribute this agent to a user — respawn it "
            "from Flight Deck"
        ),
    )


# Surfaced verbatim by the agent's Google tools (google_oauth_manager), so it
# says what the user can do about it.
_REFRESH_FAILED = (
    "Could not refresh the Google access token — if this persists, reconnect "
    "Google in Flight Deck → Connections → Google"
)


async def _refresh_if_needed(
    db: FlightDeckDB,
    user_id: str,
    client: dict[str, Any],
    tokens: GoogleOAuthTokens,
) -> GoogleOAuthTokens | None:
    """Refresh *tokens* when near expiry; persist the new pair for *user_id*.

    *client* must be the OAuth client that originally minted the
    refresh token (use :func:`_token_client`).
    """
    if not tokens.is_expired():
        return tokens
    if not tokens.refresh_token:
        return None
    try:
        fresh = await refresh_access_token(
            refresh_token=tokens.refresh_token,
            client_id=client["client_id"],
            client_secret=client["client_secret"],
        )
    except Exception as exc:
        log.warning("Google token refresh failed: %s", exc)
        return None
    await _store_tokens(db, user_id, fresh)
    return fresh


@router.get("/access_token")
async def google_access_token(request: Request) -> dict[str, Any]:
    """Return a currently-valid access token for the calling agent's owner."""
    _authorize_agent_call(request)
    db = get_db()
    owner = await _agent_owner(request)
    client = await _token_client(db)
    if not client:
        raise HTTPException(status_code=404, detail="Google OAuth not configured")
    tokens = await _load_tokens(db, owner)
    if not tokens:
        raise HTTPException(status_code=404, detail="No stored Google OAuth tokens")
    tokens = await _refresh_if_needed(db, owner, client, tokens)
    if not tokens:
        raise HTTPException(status_code=401, detail=_REFRESH_FAILED)
    return {
        "access_token": tokens.access_token,
        "token_type": tokens.token_type,
        "expires_at": tokens.expires_at,
        "scope": tokens.scope,
    }


@router.get("/credentials")
async def google_credentials(request: Request) -> dict[str, Any]:
    """Return full ``authorized_user`` credentials for LiteLLM / Vertex.

    Intended for captain-claw agents wiring up the Gemini provider via
    LiteLLM's Vertex AI path. Requires that the user has supplied a
    ``project_id`` alongside their ``client_id`` / ``client_secret``.
    """
    _authorize_agent_call(request)
    db = get_db()
    owner = await _agent_owner(request)
    client = await _token_client(db)
    if not client:
        raise HTTPException(status_code=404, detail="Google OAuth not configured")
    if not client["supports_vertex"]:
        raise HTTPException(
            status_code=409,
            detail=(
                "Vertex AI requires a Google Cloud project_id. Add one to "
                "your Google OAuth credentials on the Connections page."
            ),
        )
    tokens = await _load_tokens(db, owner)
    if not tokens:
        raise HTTPException(status_code=404, detail="No stored Google OAuth tokens")
    tokens = await _refresh_if_needed(db, owner, client, tokens)
    if not tokens:
        raise HTTPException(status_code=401, detail=_REFRESH_FAILED)
    return {
        "credentials": tokens.to_vertex_credentials_json(
            client_id=client["client_id"],
            client_secret=client["client_secret"],
        ),
        "project_id": client["project_id"],
        "location": client["location"],
    }


# ── callback HTML ───────────────────────────────────────────────────


def _callback_html(
    *,
    ok: bool,
    title: str,
    detail: str = "",
    email: str = "",
    status_code: int = 200,
) -> HTMLResponse:
    """Render a popup-aware confirmation page.

    Tries to ``postMessage`` the result to a same-origin ``window.opener`` (the
    Flight Deck tab) and close itself. Falls back to a redirect to ``/`` for
    the full-tab flow.

    *detail* can be attacker-chosen (``/callback?error_description=…`` is
    unauthenticated), so everything is HTML-escaped for the markup and
    JSON-encoded for the script with ``<``, ``>`` and ``&`` as ``\\u`` escapes —
    no value can close the ``<script>`` or open a comment. A nonce CSP blocks
    any script that isn't ours, and the result is only posted to this page's
    own origin, never ``'*'``.
    """
    status_word = "success" if ok else "error"
    payload = json.dumps(
        {
            "type": "captain-claw-google-oauth",
            "status": status_word,
            "title": title,
            "detail": detail,
            "email": email,
        },
        ensure_ascii=True,
    )
    payload = payload.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    h_title = html.escape(title)
    h_detail = html.escape(detail)
    nonce = secrets.token_urlsafe(16)

    page = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{h_title}</title>
<style>
body{{background:#0d1117;color:#e6edf3;font-family:-apple-system,system-ui,sans-serif;
display:flex;align-items:center;justify-content:center;min-height:100vh;margin:0;}}
.box{{text-align:center;padding:2rem;}}
.box h2{{margin:0 0 0.5rem;font-weight:600;}}
.box p{{color:#8b949e;margin:0;}}
.ok{{color:#3fb950;}}
.err{{color:#f85149;}}
</style>
</head>
<body>
<div class="box">
<h2 class="{'ok' if ok else 'err'}">{h_title}</h2>
<p>{h_detail}</p>
<p style="margin-top:1rem;font-size:0.8rem;">You can close this window.</p>
</div>
<script nonce="{nonce}">
(function() {{
  var payload = {payload};
  var origin = window.location.origin;
  var opener = window.opener;
  if (opener && !opener.closed) {{
    try {{
      // Throws for a cross-origin opener: then leave this page up.
      if (opener.location.origin === origin) {{
        opener.postMessage(payload, origin);
        setTimeout(function(){{ window.close(); }}, 400);
      }}
    }} catch (e) {{}}
    return;
  }}
  setTimeout(function(){{ window.location.href = '/'; }}, 1500);
}})();
</script>
</body>
</html>"""
    return HTMLResponse(
        content=page,
        status_code=status_code,
        headers={
            "Content-Security-Policy": (
                "default-src 'none'; style-src 'unsafe-inline'; "
                f"script-src 'nonce-{nonce}'; base-uri 'none'; "
                "form-action 'none'; frame-ancestors 'none'"
            ),
            "Cache-Control": "no-store",
            "Referrer-Policy": "no-referrer",
        },
    )
