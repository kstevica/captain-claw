"""Public session management for public-run mode.

Each public visitor gets their own isolated session identified by a 6-character
access code.  The code is stored in ``session.metadata["public_code"]`` and
bound to the visitor via a signed cookie.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import string
import time
from typing import TYPE_CHECKING, Any

from aiohttp import web

from captain_claw.logging import get_logger
from captain_claw.session import Session, get_session_manager

if TYPE_CHECKING:
    from captain_claw.web_server import WebServer

log = get_logger(__name__)

COOKIE_NAME = "claw_public"
# Alphabet without ambiguous characters (no 0/O, 1/I/L)
_ALPHABET = "23456789ABCDEFGHJKMNPQRSTUVWXYZ"
_CODE_LENGTH = 6
# HMAC key derived from the first session id + a fixed salt.  In practice we
# use the server's auth_token when available, otherwise a per-process secret.
_process_secret: str = secrets.token_hex(16)


def _hmac_key(config_token: str) -> str:
    return config_token if config_token else _process_secret


def generate_code() -> str:
    """Generate a 6-character access code."""
    return "".join(secrets.choice(_ALPHABET) for _ in range(_CODE_LENGTH))


def _make_cookie(session_id: str, code: str, key: str) -> str:
    """Create ``session_id:timestamp:hmac`` cookie value."""
    ts = str(int(time.time()))
    payload = f"{session_id}:{code}:{ts}"
    sig = hmac.new(key.encode(), payload.encode(), hashlib.sha256).hexdigest()
    return f"{payload}:{sig}"


def _validate_cookie(value: str, key: str, max_age_days: int = 90) -> tuple[str, str] | None:
    """Return ``(session_id, code)`` if the cookie is valid, else *None*."""
    parts = value.split(":")
    if len(parts) != 4:
        return None
    session_id, code, ts_str, sig = parts
    try:
        ts = int(ts_str)
    except ValueError:
        return None
    age = time.time() - ts
    if age < 0 or age > max_age_days * 86400:
        return None
    expected = hmac.new(
        key.encode(), f"{session_id}:{code}:{ts_str}".encode(), hashlib.sha256
    ).hexdigest()
    if not hmac.compare_digest(sig, expected):
        return None
    return session_id, code


def set_public_cookie(
    response: web.StreamResponse,
    session_id: str,
    code: str,
    auth_token: str,
    *,
    secure: bool = False,
) -> None:
    """Set the public session cookie on *response*."""
    key = _hmac_key(auth_token)
    cookie_val = _make_cookie(session_id, code, key)
    response.set_cookie(
        COOKIE_NAME,
        cookie_val,
        max_age=90 * 86400,
        httponly=True,
        samesite="Lax",
        path="/",
        secure=secure,
    )


def read_public_cookie(request: web.Request, auth_token: str) -> tuple[str, str] | None:
    """Read and validate the public cookie.  Returns ``(session_id, code)`` or *None*."""
    raw = request.cookies.get(COOKIE_NAME, "")
    if not raw:
        return None
    return _validate_cookie(raw, _hmac_key(auth_token))


async def create_public_session(server: "WebServer") -> tuple[Session, str]:
    """Create a new session for a public visitor.

    Returns ``(session, code)``.
    """
    sm = get_session_manager()
    code = generate_code()
    session = await sm.create_session(
        name=f"Public {code}",
        metadata={"public_code": code},
    )
    log.info("Public session created", session_id=session.id, code=code)
    return session, code


async def load_session_by_code(code: str) -> Session | None:
    """Find a session whose ``metadata.public_code`` matches *code*."""
    sm = get_session_manager()
    await sm._ensure_db()
    assert sm._db is not None
    async with sm._db.execute(
        "SELECT id, name, messages, created_at, updated_at, metadata FROM sessions ORDER BY updated_at DESC"
    ) as cursor:
        import json
        async for row in cursor:
            try:
                meta = json.loads(row[5]) if row[5] else {}
            except (json.JSONDecodeError, TypeError):
                meta = {}
            if meta.get("public_code", "").upper() == code.upper():
                messages = json.loads(row[2]) if row[2] else []
                return Session(
                    id=row[0],
                    name=row[1],
                    messages=messages,
                    created_at=row[3],
                    updated_at=row[4],
                    metadata=meta,
                )
    return None
