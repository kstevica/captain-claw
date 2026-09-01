"""OAuth 2.1 authorization server for the inbound MCP endpoint.

This is what lets a browser-based MCP client — the **claude.ai "custom
connector"** flow (Claude Desktop / web) — sign a user in to their Flight Deck
account and obtain a bearer token for ``/fd/mcp-server``, instead of pasting a
static ``cc_pat_`` token (which the Claude Code CLI supports but the connector
UI does not).

Discovery + flow, all at the hub root (un-prefixed, public, mirroring
toolchestrator/backend/app/routers/oauth.py):

* ``GET /.well-known/oauth-protected-resource[/…]`` — RFC 9728: points the client
  from the resource (``/fd/mcp-server``) at this authorization server.
* ``GET /.well-known/oauth-authorization-server`` — RFC 8414: authorize / token /
  register endpoints, PKCE S256, public-client.
* ``POST /register`` — RFC 7591 Dynamic Client Registration (public/PKCE only).
* ``GET /authorize`` → sign-in + consent page; ``POST /authorize`` verifies the
  password and issues a single-use code.
* ``POST /token`` — code (+PKCE) → short-lived mcp JWT + rotating refresh token.

PKCE S256 is mandatory; codes are single-use, 60s, bound to
client+redirect_uri+PKCE+user; redirect_uris match exactly (no open redirect);
the access token is a purpose-scoped JWT usable only at the MCP endpoint.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import html
import json
import logging
import os
import secrets
from datetime import datetime, timezone
from urllib.parse import parse_qs, urlencode, urlsplit, urlunsplit

from fastapi import APIRouter, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel, Field

from captain_claw.flight_deck import auth as A
from captain_claw.flight_deck.auth import get_db

log = logging.getLogger(__name__)
router = APIRouter(tags=["mcp-oauth"])

MAX_REDIRECT_URIS = 10
MAX_REDIRECT_URI_LEN = 2048
RESOURCE_PATH = "/fd/mcp-server"


def _public_base(request: Request) -> str:
    env = os.environ.get("FD_PUBLIC_URL", "").strip()
    if env:
        return env.rstrip("/")
    # Honor a reverse proxy's forwarded scheme/host when present.
    proto = request.headers.get("x-forwarded-proto", "").split(",")[0].strip()
    host = request.headers.get("x-forwarded-host", "").split(",")[0].strip()
    if proto and host:
        return f"{proto}://{host}".rstrip("/")
    return str(request.base_url).rstrip("/")


def _json_no_store(data: dict, status_code: int = 200) -> JSONResponse:
    return JSONResponse(data, status_code=status_code,
                        headers={"Cache-Control": "no-store", "Pragma": "no-cache"})


# ── Discovery (RFC 9728 / RFC 8414) ──────────────────────────────────

def _protected_resource(request: Request) -> dict:
    base = _public_base(request)
    return {
        "resource": f"{base}{RESOURCE_PATH}",
        "authorization_servers": [base],
        "scopes_supported": ["mcp"],
        "bearer_methods_supported": ["header"],
    }


@router.get("/.well-known/oauth-protected-resource")
def protected_resource(request: Request) -> JSONResponse:
    return _json_no_store(_protected_resource(request))


@router.get("/.well-known/oauth-protected-resource/{rest:path}")
def protected_resource_variant(rest: str, request: Request) -> JSONResponse:
    # RFC 9728 clients insert the well-known path BEFORE the resource path, e.g.
    # /.well-known/oauth-protected-resource/fd/mcp-server — answer the same.
    return _json_no_store(_protected_resource(request))


@router.get("/.well-known/oauth-authorization-server")
def authorization_server(request: Request) -> JSONResponse:
    base = _public_base(request)
    return _json_no_store({
        "issuer": base,
        "authorization_endpoint": f"{base}/authorize",
        "token_endpoint": f"{base}/token",
        "registration_endpoint": f"{base}/register",
        "response_types_supported": ["code"],
        "grant_types_supported": ["authorization_code", "refresh_token"],
        "code_challenge_methods_supported": ["S256"],
        "token_endpoint_auth_methods_supported": ["none"],
        "scopes_supported": ["mcp"],
    })


# ── Dynamic Client Registration (RFC 7591) ───────────────────────────

class ClientRegistration(BaseModel):
    redirect_uris: list[str] = Field(min_length=1)
    client_name: str = ""
    # Accepted and ignored (public/PKCE-only server):
    grant_types: list[str] | None = None
    response_types: list[str] | None = None
    token_endpoint_auth_method: str | None = None
    scope: str | None = None


def _valid_redirect_uri(uri: str) -> bool:
    if not uri or len(uri) > MAX_REDIRECT_URI_LEN:
        return False
    parts = urlsplit(uri)
    if parts.scheme not in ("http", "https") or not parts.netloc:
        return False
    if parts.scheme == "http":
        host = parts.hostname or ""
        if host not in ("localhost", "127.0.0.1", "::1"):
            return False
    return True


@router.post("/register")
async def register_client(body: ClientRegistration, request: Request) -> JSONResponse:
    db = get_db()
    await db.prune_oauth()
    uris = list(body.redirect_uris or [])
    if len(uris) > MAX_REDIRECT_URIS:
        return _json_no_store({"error": "invalid_redirect_uri",
                               "error_description": "too many redirect_uris"}, 400)
    for u in uris:
        if not _valid_redirect_uri(u):
            return _json_no_store({"error": "invalid_redirect_uri",
                                   "error_description": f"invalid redirect_uri: {u}"}, 400)
    client_id = A.new_oauth_client_id()
    name = (body.client_name or "").strip()[:120]
    await db.create_oauth_client(client_id, name, uris)
    return _json_no_store({
        "client_id": client_id,
        "client_id_issued_at": int(datetime.now(timezone.utc).timestamp()),
        "client_name": name,
        "redirect_uris": uris,
        "grant_types": ["authorization_code", "refresh_token"],
        "response_types": ["code"],
        "token_endpoint_auth_method": "none",
        "scope": "mcp",
    }, 201)


# ── Authorize (sign-in + consent) ────────────────────────────────────

_PAGE_CSS = (
    "body{margin:0;background:#09090b;color:#e4e4e7;font:15px/1.5 -apple-system,BlinkMacSystemFont,"
    "'Segoe UI',sans-serif;display:flex;min-height:100vh;align-items:center;justify-content:center}"
    ".card{width:360px;max-width:90vw;background:#18181b;border:1px solid #27272a;border-radius:12px;padding:28px}"
    "h1{font-size:18px;margin:0 0 4px}p{color:#a1a1aa;font-size:13px;margin:0 0 18px}"
    "label{display:block;font-size:12px;color:#a1a1aa;margin:12px 0 4px}"
    "input{width:100%;box-sizing:border-box;background:#09090b;border:1px solid #3f3f46;border-radius:8px;"
    "padding:9px 11px;color:#e4e4e7;font-size:14px}input:focus{outline:none;border-color:#8b5cf6}"
    ".row{display:flex;gap:8px;margin-top:18px}button{flex:1;border-radius:8px;padding:9px;font-size:14px;"
    "font-weight:500;border:1px solid #3f3f46;cursor:pointer}"
    ".approve{background:#7c3aed;border-color:#7c3aed;color:#fff}.deny{background:transparent;color:#a1a1aa}"
    ".err{background:rgba(239,68,68,.1);border:1px solid rgba(239,68,68,.4);color:#fca5a5;"
    "border-radius:8px;padding:8px 10px;font-size:12px;margin-bottom:14px}"
)


def _page(inner: str, status_code: int = 200) -> HTMLResponse:
    body = f"<!doctype html><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>" \
           f"<style>{_PAGE_CSS}</style><div class=card>{inner}</div>"
    return HTMLResponse(body, status_code=status_code, headers={
        "X-Frame-Options": "DENY",
        "Content-Security-Policy": "frame-ancestors 'none'",
        "Cache-Control": "no-store",
    })


def _error_page(title: str, detail: str, status_code: int = 400) -> HTMLResponse:
    return _page(f"<h1>{html.escape(title)}</h1><p>{html.escape(detail)}</p>", status_code)


def _consent_page(client_name: str, blob: str, error: str = "") -> HTMLResponse:
    name = html.escape(client_name or "an application")
    err = f"<div class=err>{html.escape(error)}</div>" if error else ""
    inner = (
        f"<h1>Connect to Flight Deck</h1>"
        f"<p>{name} wants to access your Flight Deck agents (list, send tasks, read results).</p>"
        f"{err}"
        f"<form method=post action='/authorize'>"
        f"<input type=hidden name=request value='{html.escape(blob)}'>"
        f"<label>Email</label><input name=email type=email autocomplete=username autofocus>"
        f"<label>Password</label><input name=password type=password autocomplete=current-password>"
        f"<div class=row>"
        f"<button class=deny name=decision value=deny>Deny</button>"
        f"<button class=approve name=decision value=approve>Approve</button>"
        f"</div></form>"
    )
    return _page(inner)


def _append_query(uri: str, params: dict) -> str:
    parts = urlsplit(uri)
    existing = parse_qs(parts.query)
    merged = {k: v[0] for k, v in existing.items()}
    merged.update({k: v for k, v in params.items() if v is not None})
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(merged), parts.fragment))


@router.get("/authorize")
async def authorize_get(
    request: Request,
    response_type: str = Query(default=""),
    client_id: str = Query(default=""),
    redirect_uri: str = Query(default=""),
    code_challenge: str = Query(default=""),
    code_challenge_method: str = Query(default=""),
    state: str = Query(default=""),
    scope: str = Query(default="mcp"),
    resource: str = Query(default=""),
):
    db = get_db()
    client = await db.get_oauth_client(client_id) if client_id else None
    if not client:
        return _error_page("Unknown application", "This client is not registered.", 400)
    if redirect_uri not in (client.get("redirect_uris") or []):
        return _error_page("Redirect mismatch", "redirect_uri is not registered for this client.", 400)
    # Only after client + redirect_uri are trusted may we redirect errors back.
    if response_type != "code":
        return RedirectResponse(_append_query(redirect_uri, {"error": "unsupported_response_type", "state": state}), 302)
    if not code_challenge or code_challenge_method != "S256":
        return RedirectResponse(_append_query(redirect_uri, {"error": "invalid_request", "error_description": "PKCE S256 required", "state": state}), 302)
    blob = A.make_oauth_request({
        "client_id": client_id, "redirect_uri": redirect_uri,
        "code_challenge": code_challenge, "state": state,
        "scope": scope or "mcp", "resource": resource or None,
    })
    return _consent_page(client.get("client_name") or "", blob)


async def _read_form(request: Request) -> dict:
    """Parse an x-www-form-urlencoded OR application/json body by hand (no
    python-multipart dependency)."""
    raw = (await request.body()).decode("utf-8", "replace")
    ctype = request.headers.get("content-type", "")
    if "application/json" in ctype:
        try:
            data = json.loads(raw or "{}")
            return {k: v for k, v in data.items()} if isinstance(data, dict) else {}
        except Exception:
            return {}
    return {k: v[0] for k, v in parse_qs(raw).items()}


@router.post("/authorize")
async def authorize_post(request: Request):
    form = await _read_form(request)
    blob = form.get("request", "")
    req = A.decode_oauth_request(blob) if blob else None
    if not req:
        return _error_page("Session expired", "Please start the connection again.", 400)

    db = get_db()
    client_id = req.get("client_id", "")
    redirect_uri = req.get("redirect_uri", "")
    state = req.get("state", "")
    client = await db.get_oauth_client(client_id)
    if not client or redirect_uri not in (client.get("redirect_uris") or []):
        return _error_page("Invalid request", "Client or redirect_uri is no longer valid.", 400)

    if form.get("decision") != "approve":
        return RedirectResponse(_append_query(redirect_uri, {"error": "access_denied", "state": state}), 302)

    email = (form.get("email") or "").strip().lower()
    password = form.get("password") or ""
    user = await db.get_user_by_email(email) if email else None
    ok = False
    if user and user.get("password_hash"):
        try:
            ok = A.verify_password(password, user["password_hash"])
        except Exception:
            ok = False
    if not user or not ok:
        # Re-render the consent page with the SAME signed blob (still valid).
        return _consent_page(client.get("client_name") or "", blob, "Incorrect email or password")

    raw_code = A.new_oauth_code()
    expires_at = (datetime.now(timezone.utc) + A.OAUTH_CODE_TTL).isoformat()
    await db.create_oauth_code(
        A.hash_token(raw_code), client_id, user["id"], redirect_uri,
        req.get("code_challenge", ""), req.get("scope", "mcp"),
        req.get("resource"), expires_at,
    )
    return RedirectResponse(_append_query(redirect_uri, {"code": raw_code, "state": state}), 302)


# ── Token endpoint ───────────────────────────────────────────────────

def _pkce_ok(verifier: str, challenge: str) -> bool:
    if not verifier or not challenge:
        return False
    digest = hashlib.sha256(verifier.encode()).digest()
    calc = base64.urlsafe_b64encode(digest).decode().rstrip("=")
    return hmac.compare_digest(calc, challenge)


def _token_error(error: str, desc: str = "", status_code: int = 400) -> JSONResponse:
    body = {"error": error}
    if desc:
        body["error_description"] = desc
    return _json_no_store(body, status_code)


async def _issue_tokens(db, user_id: str, client_id: str, client_name: str, scope: str) -> JSONResponse:
    access = A.create_mcp_access_token(user_id, client_id, scope)
    raw_refresh = A.new_oauth_refresh()
    await db.create_oauth_refresh(A.hash_token(raw_refresh), client_id, client_name, user_id, scope)
    return _json_no_store({
        "access_token": access,
        "token_type": "Bearer",
        "expires_in": int(A.MCP_ACCESS_TTL.total_seconds()),
        "refresh_token": raw_refresh,
        "scope": scope,
    })


@router.post("/token")
async def token(request: Request):
    db = get_db()
    await db.prune_oauth()
    form = await _read_form(request)
    grant = form.get("grant_type", "")

    if grant == "authorization_code":
        code = form.get("code", "")
        client_id = form.get("client_id", "")
        redirect_uri = form.get("redirect_uri", "")
        verifier = form.get("code_verifier", "")
        if not code:
            return _token_error("invalid_request", "code required")
        row = await db.get_oauth_code_by_hash(A.hash_token(code))
        if not row or row.get("used_at"):
            return _token_error("invalid_grant", "invalid or used code")
        try:
            expired = datetime.now(timezone.utc) > datetime.fromisoformat(row["expires_at"])
        except Exception:
            expired = True
        if expired:
            return _token_error("invalid_grant", "code expired")
        if row["client_id"] != client_id or row["redirect_uri"] != redirect_uri:
            return _token_error("invalid_grant", "client_id/redirect_uri mismatch")
        if not _pkce_ok(verifier, row["code_challenge"]):
            return _token_error("invalid_grant", "PKCE verification failed")
        if not await db.burn_oauth_code(row["id"]):
            return _token_error("invalid_grant", "authorization code already used")
        client = await db.get_oauth_client(client_id)
        cname = (client or {}).get("client_name", "") if client else ""
        return await _issue_tokens(db, row["user_id"], client_id, cname, row.get("scope", "mcp"))

    if grant == "refresh_token":
        rt = form.get("refresh_token", "")
        client_id = form.get("client_id", "")
        if not rt:
            return _token_error("invalid_request", "refresh_token required")
        row = await db.get_oauth_refresh_by_hash(A.hash_token(rt))
        if not row:
            return _token_error("invalid_grant", "invalid refresh token")
        if client_id and row["client_id"] != client_id:
            return _token_error("invalid_grant", "client mismatch")
        if not await db.rotate_oauth_refresh(row["id"]):
            return _token_error("invalid_grant", "refresh token already used")
        return await _issue_tokens(db, row["user_id"], row["client_id"],
                                   row.get("client_name", ""), row.get("scope", "mcp"))

    return _token_error("unsupported_grant_type", f"unsupported grant_type: {grant}")
