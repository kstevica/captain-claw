"""Google OAuth2 for Gemini API access via Vertex AI.

Implements the Authorization Code flow with PKCE against standard Google
OAuth2 endpoints.  The resulting ``authorized_user`` credentials JSON is
consumed by LiteLLM's Vertex AI provider, which passes it to
``google-auth`` for automatic token refresh.
"""

from __future__ import annotations

import base64
import hashlib
import secrets
import time
from dataclasses import asdict, dataclass
from typing import Any
from urllib.parse import urlencode

import httpx

from captain_claw.logging import get_logger

log = get_logger(__name__)

# ── Google OAuth2 endpoints ──────────────────────────────────────────

GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_URL = "https://www.googleapis.com/oauth2/v3/userinfo"
GOOGLE_REVOKE_URL = "https://oauth2.googleapis.com/revoke"

# Non-sensitive scopes that Google lets *any* OAuth client request,
# even when the app is unverified and still in Testing mode. We use
# this as the default scope list so a fresh install of captain-claw
# can connect a Google account without hitting the "sensitive scopes
# without verification" block.
#
# Everything beyond this set (Gmail, Drive full access, Calendar,
# Vertex AI / cloud-platform) is either "sensitive" or "restricted"
# per Google's policy and requires either:
#   * Adding the user as an explicit *Test user* on the consent screen
#     while the app is in Testing mode, OR
#   * Going through Google's OAuth verification process.
#
# Users opt into those by ticking them in the Connections page UI,
# which persists the chosen list in ``system_settings`` and feeds it
# back through :func:`build_authorization_url`.
DEFAULT_SCOPES = [
    "openid",
    "email",
    "profile",
    # Per-file Drive access — non-sensitive, unlike the full-drive scope.
    "https://www.googleapis.com/auth/drive.file",
]


# Catalogue of scopes the Connections UI knows how to request.
# ``sensitivity`` is one of:
#   * ``none``       — freely usable by unverified apps
#   * ``sensitive``  — Google flags these; unverified apps need test users
#   * ``restricted`` — requires security review + verification in prod
#
# The UI shows a badge for each tier so users understand what they're
# signing up for before ticking a box.
SCOPE_CATALOG: list[dict[str, str]] = [
    {
        "scope": "openid",
        "label": "OpenID",
        "description": "Sign in with Google (required).",
        "sensitivity": "none",
        "group": "identity",
    },
    {
        "scope": "email",
        "label": "Email address",
        "description": "Your Google account email.",
        "sensitivity": "none",
        "group": "identity",
    },
    {
        "scope": "profile",
        "label": "Basic profile",
        "description": "Your name and profile picture.",
        "sensitivity": "none",
        "group": "identity",
    },
    {
        "scope": "https://www.googleapis.com/auth/drive.file",
        "label": "Drive (per-file)",
        "description": (
            "Read/write Drive files the app created or the user explicitly "
            "picks — no access to anything else in Drive."
        ),
        "sensitivity": "none",
        "group": "drive",
    },
    {
        "scope": "https://www.googleapis.com/auth/drive.readonly",
        "label": "Drive (read all)",
        "description": "Read all files in Drive. Restricted scope.",
        "sensitivity": "restricted",
        "group": "drive",
    },
    {
        "scope": "https://www.googleapis.com/auth/drive",
        "label": "Drive (full access)",
        "description": "Full read/write access to all of Drive. Restricted scope.",
        "sensitivity": "restricted",
        "group": "drive",
    },
    {
        "scope": "https://www.googleapis.com/auth/gmail.readonly",
        "label": "Gmail (read)",
        "description": "List, search, and read messages. Restricted scope.",
        "sensitivity": "restricted",
        "group": "gmail",
    },
    {
        "scope": "https://www.googleapis.com/auth/gmail.compose",
        "label": "Gmail (drafts)",
        "description": "Create draft replies. Sensitive scope.",
        "sensitivity": "sensitive",
        "group": "gmail",
    },
    {
        "scope": "https://www.googleapis.com/auth/gmail.modify",
        "label": "Gmail (read + modify)",
        "description": "Read + label / archive / trash. Restricted scope.",
        "sensitivity": "restricted",
        "group": "gmail",
    },
    {
        "scope": "https://www.googleapis.com/auth/gmail.send",
        "label": "Gmail (send)",
        "description": "Send mail on your behalf. Sensitive scope.",
        "sensitivity": "sensitive",
        "group": "gmail",
    },
    {
        "scope": "https://www.googleapis.com/auth/calendar.readonly",
        "label": "Calendar (read)",
        "description": "Read calendars and events. Sensitive scope.",
        "sensitivity": "sensitive",
        "group": "calendar",
    },
    {
        "scope": "https://www.googleapis.com/auth/calendar",
        "label": "Calendar (read + write)",
        "description": "Read and manage events. Sensitive scope.",
        "sensitivity": "sensitive",
        "group": "calendar",
    },
    {
        "scope": "https://www.googleapis.com/auth/cloud-platform",
        "label": "Vertex AI / Gemini",
        "description": (
            "Access Google Cloud APIs (Vertex AI / Gemini). Sensitive, "
            "and only useful when you also configure a project_id."
        ),
        "sensitivity": "sensitive",
        "group": "cloud",
    },
]


_SCOPE_SET: frozenset[str] = frozenset(s["scope"] for s in SCOPE_CATALOG)


def sanitize_scopes(scopes: list[str] | None) -> list[str]:
    """Return *scopes* filtered to known entries, always including the
    identity triplet (openid/email/profile) so Google can identify the
    account. Falls back to :data:`DEFAULT_SCOPES` when *scopes* is
    empty or ``None``."""
    if not scopes:
        return list(DEFAULT_SCOPES)
    seen: set[str] = set()
    out: list[str] = []
    for s in scopes:
        if not isinstance(s, str):
            continue
        s = s.strip()
        if not s or s in seen:
            continue
        if s not in _SCOPE_SET:
            # Unknown scopes are dropped rather than passed through —
            # prevents typos / stale values from breaking the OAuth URL.
            continue
        seen.add(s)
        out.append(s)
    # Always include openid so the ``/userinfo`` endpoint works.
    for required in ("openid", "email"):
        if required not in seen:
            out.insert(0, required)
            seen.add(required)
    return out

# App-state keys used by GoogleOAuthManager for persistent storage.
STATE_KEY_TOKENS = "google_oauth_tokens"
STATE_KEY_USER = "google_oauth_user"


# ── Data ─────────────────────────────────────────────────────────────


@dataclass
class GoogleOAuthTokens:
    """Holds tokens returned by Google's token endpoint."""

    access_token: str
    refresh_token: str
    token_type: str = "Bearer"
    expires_at: float = 0.0  # Unix timestamp
    scope: str = ""
    id_token: str = ""

    # ── helpers ──────────────────────────────────────────

    def is_expired(self, buffer_seconds: int = 300) -> bool:
        """Return *True* if the access token expires within *buffer_seconds*."""
        return time.time() >= (self.expires_at - buffer_seconds)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GoogleOAuthTokens:
        known = {
            "access_token", "refresh_token", "token_type",
            "expires_at", "scope", "id_token",
        }
        return cls(**{k: v for k, v in data.items() if k in known})

    def to_vertex_credentials_json(
        self,
        client_id: str,
        client_secret: str,
    ) -> dict[str, Any]:
        """Build an ``authorized_user`` credentials dict for LiteLLM.

        LiteLLM's ``VertexBase.load_auth()`` recognises this format and
        creates proper ``google.oauth2.credentials.Credentials`` that
        auto-refresh via the embedded refresh-token.
        """
        return {
            "type": "authorized_user",
            "client_id": client_id,
            "client_secret": client_secret,
            "refresh_token": self.refresh_token,
            "token": self.access_token,
            "token_uri": GOOGLE_TOKEN_URL,
        }


# ── PKCE ─────────────────────────────────────────────────────────────


def generate_pkce_pair() -> tuple[str, str]:
    """Return ``(code_verifier, code_challenge)`` for S256 PKCE."""
    verifier = secrets.token_urlsafe(64)
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return verifier, challenge


# ── URL builders ─────────────────────────────────────────────────────


def build_authorization_url(
    client_id: str,
    redirect_uri: str,
    scopes: list[str] | None = None,
    state: str | None = None,
    code_challenge: str | None = None,
) -> str:
    """Build the Google OAuth2 authorization URL."""
    params: dict[str, str] = {
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": " ".join(scopes or DEFAULT_SCOPES),
        "access_type": "offline",
        "prompt": "consent",
    }
    if state:
        params["state"] = state
    if code_challenge:
        params["code_challenge"] = code_challenge
        params["code_challenge_method"] = "S256"
    return f"{GOOGLE_AUTH_URL}?{urlencode(params)}"


# ── Token exchange ───────────────────────────────────────────────────


async def exchange_code_for_tokens(
    code: str,
    client_id: str,
    client_secret: str,
    redirect_uri: str,
    code_verifier: str | None = None,
) -> GoogleOAuthTokens:
    """Exchange an authorization code for access + refresh tokens."""
    payload: dict[str, str] = {
        "code": code,
        "client_id": client_id,
        "client_secret": client_secret,
        "redirect_uri": redirect_uri,
        "grant_type": "authorization_code",
    }
    if code_verifier:
        payload["code_verifier"] = code_verifier

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(GOOGLE_TOKEN_URL, data=payload)
        resp.raise_for_status()
        data = resp.json()

    return GoogleOAuthTokens(
        access_token=data["access_token"],
        refresh_token=data.get("refresh_token", ""),
        token_type=data.get("token_type", "Bearer"),
        expires_at=time.time() + int(data.get("expires_in", 3600)),
        scope=data.get("scope", ""),
        id_token=data.get("id_token", ""),
    )


async def refresh_access_token(
    refresh_token: str,
    client_id: str,
    client_secret: str,
) -> GoogleOAuthTokens:
    """Use a refresh token to obtain a fresh access token."""
    payload = {
        "refresh_token": refresh_token,
        "client_id": client_id,
        "client_secret": client_secret,
        "grant_type": "refresh_token",
    }

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(GOOGLE_TOKEN_URL, data=payload)
        resp.raise_for_status()
        data = resp.json()

    return GoogleOAuthTokens(
        access_token=data["access_token"],
        refresh_token=refresh_token,  # Google doesn't return a new one
        token_type=data.get("token_type", "Bearer"),
        expires_at=time.time() + int(data.get("expires_in", 3600)),
        scope=data.get("scope", ""),
        id_token=data.get("id_token", ""),
    )


# ── User info ────────────────────────────────────────────────────────


async def fetch_user_info(access_token: str) -> dict[str, Any]:
    """Fetch the authenticated user's Google profile."""
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(
            GOOGLE_USERINFO_URL,
            headers={"Authorization": f"Bearer {access_token}"},
        )
        resp.raise_for_status()
        return resp.json()


# ── Revocation ───────────────────────────────────────────────────────


async def revoke_token(token: str) -> bool:
    """Revoke an access or refresh token with Google.

    Returns *True* on success, *False* on failure (best effort).
    """
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(
                GOOGLE_REVOKE_URL,
                params={"token": token},
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            return resp.status_code == 200
    except Exception as exc:
        log.warning("Google token revocation failed: %s", exc)
        return False
