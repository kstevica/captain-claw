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


async def get_current_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> dict:
    """FastAPI dependency — extracts and validates JWT, returns user dict.
    Falls back to ?fd_token= query param for direct-URL access (file downloads).
    """
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
