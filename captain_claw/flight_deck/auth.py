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

    Accepts a long-lived Personal Access Token (``cc_pat_…``) as the primary
    credential — a headless MCP client holds a static string and can't drive the
    cookie-based JWT refresh. Falls back to a normal access JWT so a browser or
    short-lived token also works. Returns the user dict or raises 401.
    """
    if not _fd_auth_enabled():
        request.state.user_id = _LOCAL_USER["id"]
        request.state.user_role = _LOCAL_USER["role"]
        return dict(_LOCAL_USER)
    token_str = credentials.credentials if credentials else (
        request.query_params.get("fd_token") or request.query_params.get("token") or ""
    )
    if not token_str:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    db = get_db()
    if token_str.startswith(PAT_PREFIX):
        row = await db.get_pat_by_hash(hash_token(token_str))
        if not row:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid access token")
        user = await db.get_user_by_id(row["user_id"])
        if not user:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
        try:
            await db.touch_pat(row["id"])
        except Exception:
            pass
        request.state.user_id = user["id"]
        request.state.user_role = user.get("role", "user")
        return user
    # Fallback: a normal access JWT.
    payload = decode_access_token(token_str)
    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    user = await db.get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
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
