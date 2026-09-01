"""JWT authentication for Flight Deck."""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

import bcrypt
import jwt
from fastapi import Depends, HTTPException, Request, WebSocket, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

if TYPE_CHECKING:
    from captain_claw.flight_deck.db import FlightDeckDB

# ── Configuration ────────────────────────────────────────────────────

ACCESS_TOKEN_TTL = timedelta(minutes=15)
REFRESH_TOKEN_TTL = timedelta(days=7)
ALGORITHM = "HS256"
REFRESH_COOKIE = "fd_refresh"

# OAuth 2.1 for the inbound MCP server (claude.ai custom connectors).
MCP_ACCESS_TTL = timedelta(hours=1)      # mcp-purpose access token
OAUTH_REQUEST_TTL = timedelta(minutes=10)  # signed authorize-request blob
OAUTH_CODE_TTL = timedelta(seconds=60)   # single-use authorization code

_jwt_secret: str = ""


def get_jwt_secret() -> str:
    global _jwt_secret
    if not _jwt_secret:
        _jwt_secret = os.environ.get("FD_JWT_SECRET", "")
        if not _jwt_secret:
            _jwt_secret = secrets.token_hex(32)
    return _jwt_secret


# ── Password hashing ────────────────────────────────────────────────

def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode(), hashed.encode())


# ── Token helpers ────────────────────────────────────────────────────

def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def create_access_token(user_id: str, role: str = "user") -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": user_id,
        "role": role,
        "iat": now,
        "exp": now + ACCESS_TOKEN_TTL,
        "type": "access",
    }
    return jwt.encode(payload, get_jwt_secret(), algorithm=ALGORITHM)


def create_refresh_token() -> str:
    return secrets.token_urlsafe(48)


def decode_access_token(token: str) -> dict:
    """Decode and validate an access token. Raises on failure."""
    try:
        payload = jwt.decode(token, get_jwt_secret(), algorithms=[ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    if payload.get("type") != "access":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token type")
    return payload


# ── OAuth 2.1 / MCP tokens ───────────────────────────────────────────
# Same HS256 secret as the session JWT, but a distinct ``type`` so an
# mcp-purpose token can never be replayed as a dashboard session (and vice
# versa) — decode_access_token above rejects anything whose type != "access".

def new_oauth_client_id() -> str:
    return "cc_oc_" + secrets.token_hex(16)


def new_oauth_code() -> str:
    return "cc_ac_" + secrets.token_urlsafe(24)


def new_oauth_refresh() -> str:
    return "cc_rt_" + secrets.token_urlsafe(24)


def create_mcp_access_token(user_id: str, client_id: str, scope: str = "mcp") -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": user_id, "type": "mcp", "client_id": client_id, "scope": scope,
        "iat": now, "exp": now + MCP_ACCESS_TTL,
    }
    return jwt.encode(payload, get_jwt_secret(), algorithm=ALGORITHM)


def decode_mcp_access_token(token: str) -> dict | None:
    """Decode an mcp-purpose access token, or None if invalid/wrong type."""
    try:
        payload = jwt.decode(token, get_jwt_secret(), algorithms=[ALGORITHM])
    except jwt.InvalidTokenError:
        return None
    return payload if payload.get("type") == "mcp" else None


def make_oauth_request(params: dict) -> str:
    """Sign the validated /authorize params into a short-lived blob carried
    through the consent page (so POST /authorize can trust them)."""
    now = datetime.now(timezone.utc)
    payload = {**params, "type": "oauth_request", "iat": now, "exp": now + OAUTH_REQUEST_TTL}
    return jwt.encode(payload, get_jwt_secret(), algorithm=ALGORITHM)


def decode_oauth_request(blob: str) -> dict | None:
    try:
        payload = jwt.decode(blob, get_jwt_secret(), algorithms=[ALGORITHM])
    except jwt.InvalidTokenError:
        return None
    return payload if payload.get("type") == "oauth_request" else None


# ── FastAPI dependencies ─────────────────────────────────────────────

_bearer_scheme = HTTPBearer(auto_error=False)

# Global reference to DB — set by server.py on startup
_db: FlightDeckDB | None = None


def set_auth_db(db: FlightDeckDB) -> None:
    global _db
    _db = db


def get_db() -> FlightDeckDB:
    assert _db is not None, "Flight Deck DB not initialized"
    return _db


def _fd_auth_enabled() -> bool:
    """Whether Flight Deck JWT auth is enforced (mirrors server.AUTH_ENABLED)."""
    return os.environ.get("FD_AUTH_ENABLED", "true").lower() in ("true", "1", "yes")


# Synthetic user returned when auth is disabled (standalone desktop / local).
_LOCAL_USER: dict = {
    "id": "local",
    "email": "local",
    "display_name": "Local",
    "role": "admin",
}


async def get_current_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> dict:
    """FastAPI dependency — extracts and validates JWT, returns user dict.
    Falls back to ?fd_token= query param for direct-URL access (file downloads).
    """
    # When auth is disabled (e.g. the standalone desktop app, FD_AUTH_ENABLED=
    # false) the frontend never logs in, so there is no token. Routers that
    # depend on get_current_user directly (connector config: Google / Codex /
    # MCP) would 401. Return a synthetic local admin so they work — consistent
    # with the server's _no_user bypass for endpoints using _required_user_dep.
    if not _fd_auth_enabled():
        request.state.user_id = _LOCAL_USER["id"]
        request.state.user_role = _LOCAL_USER["role"]
        return dict(_LOCAL_USER)
    token_str: str | None = None
    if credentials is not None:
        token_str = credentials.credentials
    else:
        # Fallback: check for fd_token query parameter (used by file download/view URLs)
        token_str = request.query_params.get("fd_token")
    if not token_str:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    payload = decode_access_token(token_str)
    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    db = get_db()
    user = await db.get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    request.state.user_id = user_id
    request.state.user_role = user.get("role", "user")
    return user


async def get_optional_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> dict | None:
    """Like get_current_user but returns None instead of 401 when no token is provided.
    Used for endpoints that should work both authenticated and from internal agent calls."""
    token_str: str | None = None
    if credentials is not None:
        token_str = credentials.credentials
    else:
        token_str = request.query_params.get("fd_token")
    if not token_str:
        return None
    try:
        payload = decode_access_token(token_str)
        user_id = payload.get("sub")
        if not user_id:
            return None
        db = get_db()
        user = await db.get_user_by_id(user_id)
        if not user:
            return None
        request.state.user_id = user_id
        request.state.user_role = user.get("role", "user")
        return user
    except HTTPException:
        return None


# Personal access tokens for headless clients (MCP). A raw token looks like
# ``cc_pat_<43 url-safe chars>``; only its sha256 hash is stored.
PAT_PREFIX = "cc_pat_"


def new_pat() -> str:
    """Mint a new raw personal access token (shown to the user once)."""
    return PAT_PREFIX + secrets.token_urlsafe(32)


async def get_mcp_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> dict:
    """Authenticate an inbound MCP/API caller.

    Accepts, in order: a long-lived Personal Access Token (``cc_pat_…``, for the
    Claude Code CLI), an OAuth ``mcp``-purpose access token (claude.ai custom
    connectors), or a normal access JWT (browser). A 401 carries the RFC 9728
    ``WWW-Authenticate`` pointer so a connector can discover the OAuth server.
    """
    if not _fd_auth_enabled():
        request.state.user_id = _LOCAL_USER["id"]
        request.state.user_role = _LOCAL_USER["role"]
        return dict(_LOCAL_USER)

    def _unauth(detail: str) -> HTTPException:
        base = str(request.base_url).rstrip("/")
        prm = f"{base}/.well-known/oauth-protected-resource"
        return HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail=detail,
            headers={"WWW-Authenticate": f'Bearer resource_metadata="{prm}"'},
        )

    token_str = credentials.credentials if credentials else (
        request.query_params.get("fd_token") or request.query_params.get("token") or ""
    )
    if not token_str:
        raise _unauth("Not authenticated")
    db = get_db()

    # (1) Personal access token (CLI).
    if token_str.startswith(PAT_PREFIX):
        row = await db.get_pat_by_hash(hash_token(token_str))
        if not row:
            raise _unauth("Invalid access token")
        user = await db.get_user_by_id(row["user_id"])
        if not user:
            raise _unauth("User not found")
        try:
            await db.touch_pat(row["id"])
        except Exception:
            pass
        request.state.user_id = user["id"]
        request.state.user_role = user.get("role", "user")
        return user

    # (2) OAuth mcp-purpose access token (claude.ai custom connector).
    mcp_payload = decode_mcp_access_token(token_str)
    if mcp_payload is not None:
        user_id = mcp_payload.get("sub")
        user = await db.get_user_by_id(user_id) if user_id else None
        if not user:
            raise _unauth("User not found")
        request.state.user_id = user["id"]
        request.state.user_role = user.get("role", "user")
        return user

    # (3) Fallback: a normal access JWT (browser).
    try:
        payload = decode_access_token(token_str)
    except HTTPException:
        raise _unauth("Invalid token")
    user_id = payload.get("sub")
    if not user_id:
        raise _unauth("Invalid token")
    user = await db.get_user_by_id(user_id)
    if not user:
        raise _unauth("User not found")
    request.state.user_id = user_id
    request.state.user_role = user.get("role", "user")
    return user


async def get_ws_user(websocket: WebSocket) -> dict:
    """Validate JWT from WebSocket query param."""
    token = websocket.query_params.get("token", "")
    if not token:
        await websocket.close(code=4001, reason="Missing token")
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
    payload = decode_access_token(token)
    user_id = payload.get("sub")
    if not user_id:
        await websocket.close(code=4001, reason="Invalid token")
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
    db = get_db()
    user = await db.get_user_by_id(user_id)
    if not user:
        await websocket.close(code=4001, reason="User not found")
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
    return user
