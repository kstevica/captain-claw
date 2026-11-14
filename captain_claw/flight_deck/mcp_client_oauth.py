"""OAuth sign-in to remote MCP servers — the MCP authorization spec, client side.

Flight Deck connects OUT to MCP servers that protect themselves with OAuth 2.1
and let a client register itself (RFC 7591 Dynamic Client Registration): hosted
connectors such as Linear, Notion, Atlassian or Cloudflare, another deck's
``/fd/mcp-server``, anything built on the MCP SDKs' auth. A server record with
``auth: "oauth"`` goes through:

1. **Discovery.** An unauthenticated ``initialize`` draws a 401 whose
   ``WWW-Authenticate: Bearer resource_metadata="…"`` names the Protected
   Resource Metadata (RFC 9728); without one the well-known PRM paths are
   tried. Its ``authorization_servers[0]`` is the issuer, whose RFC 8414
   metadata (or OIDC discovery) names the authorize / token / register
   endpoints. A server with neither (the 2025-03-26 revision) gets that spec's
   fallback: the server's origin + ``/authorize``, ``/token``, ``/register``.
2. **Registration.** One POST to the registration endpoint per (issuer,
   redirect URI), reused by later sign-ins. A record that carries its own
   ``client_id`` skips it (a client registered by hand).
3. **Authorization.** The user's browser goes to the authorize endpoint with
   PKCE S256, ``state`` and ``resource`` (RFC 8707 — the token is bound to
   THIS MCP server, never to whatever resource a hostile PRM names).
4. **Tokens.** The callback swaps the code for access + refresh tokens. The
   HTTP transport asks :func:`access_token` before every request; it refreshes
   ahead of expiry, and once more after a 401 — one refresh per server at a
   time, because refresh tokens commonly rotate (single use).

The grant belongs to the SERVER record, and a record is private to the user
who added it (see :mod:`mcp_storage`): only that user signs in, and only their
agents use the account — unless an admin made the server deck-wide.

Grants live beside the servers file (``<servers>.oauth.json``, mode 0600), not
in the record, so a token refresh never looks like a config change to the
manager (which rebuilds a server's session when its record changes). A grant
remembers the URL it was made for; one for another URL is ignored.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import re
import secrets
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlencode, urlsplit, urlunsplit

import httpx

from captain_claw.flight_deck import mcp_storage
from captain_claw.logging import get_logger

log = get_logger(__name__)

CLIENT_NAME = "Captain Flight Deck"
_SOFTWARE_ID = "captain-claw-flight-deck"
_HTTP_TIMEOUT = 20.0
# Refresh this long before the access token's stated expiry.
_EXPIRY_SKEW_SECONDS = 60.0
# Same version the manager's initialize sends.
_PROTOCOL_VERSION = "2025-03-26"

_LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1"}

_NOT_SIGNED_IN = (
    "Not signed in to MCP server '{name}' — save it, then click Sign in on "
    "Connections → MCP servers"
)
_SIGN_IN_AGAIN = (
    "The sign-in to MCP server '{name}' is no longer valid ({why}) — sign in "
    "again on Connections → MCP servers"
)


class MCPOAuthError(RuntimeError):
    """Discovery, registration or a token call failed; the message is for the user."""


class MCPOAuthRequired(MCPOAuthError):
    """No usable grant: the server needs a (new) sign-in."""


def is_oauth(record: dict[str, Any] | None) -> bool:
    return bool(record) and str(record.get("auth") or "").strip().lower() == "oauth"


# ── grant storage ───────────────────────────────────────────────────


def _grants_path() -> Path:
    servers = mcp_storage._storage_path()
    return servers.with_name(servers.stem + ".oauth.json")


_write_lock = asyncio.Lock()


def _load_all() -> dict[str, dict[str, Any]]:
    path = _grants_path()
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        log.warning("Failed to read MCP OAuth grants file %s: %s", path, exc)
        return {}
    if not isinstance(raw, dict):
        return {}
    return {str(k): v for k, v in raw.items() if isinstance(v, dict)}


def _write_all(grants: dict[str, dict[str, Any]]) -> None:
    path = _grants_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(grants, fh, indent=2, sort_keys=True)
    os.replace(tmp, path)


async def _update(name: str, fn) -> dict[str, Any] | None:
    """Read-modify-write one server's grant under the file lock. ``fn`` gets
    the current grant (or None) and returns the new one (None deletes)."""
    async with _write_lock:
        grants = _load_all()
        new = fn(grants.get(name))
        if new is None:
            grants.pop(name, None)
        else:
            grants[name] = new
        _write_all(grants)
        return new


def _grant_for(record: dict[str, Any]) -> dict[str, Any] | None:
    """The stored grant for ``record`` — only when it was made for this URL."""
    grant = _load_all().get(str(record.get("name") or ""))
    if not grant:
        return None
    if grant.get("server_url") != str(record.get("url") or ""):
        return None
    return grant


def public_status(record: dict[str, Any]) -> dict[str, Any]:
    """What the UI shows about a server's sign-in — never a token."""
    grant = _grant_for(record) or {}
    return {
        "connected": bool(grant.get("access_token")),
        "connected_by": str(grant.get("connected_by") or ""),
        "connected_at": float(grant.get("connected_at") or 0.0),
        "expires_at": float(grant.get("expires_at") or 0.0),
        "issuer": str(grant.get("issuer") or ""),
        "scope": str(grant.get("scope") or ""),
        "registered": bool(grant.get("registration")),
    }


async def forget(name: str) -> None:
    """Drop everything stored for ``name`` (the server was removed)."""
    await _update(name, lambda _g: None)


def signer(record: dict[str, Any]) -> tuple[str, str]:
    """``(user_id, label)`` of whoever signed ``record``'s server in. Grants
    made before ``connected_by_id`` existed have only the label (an email)."""
    grant = _grant_for(record) or {}
    return str(grant.get("connected_by_id") or ""), str(grant.get("connected_by") or "")


_TOKEN_FIELDS = (
    "access_token", "refresh_token", "token_type", "expires_at", "scope",
    "connected_at", "connected_by", "connected_by_id", "client",
)


def _without_tokens(grant: dict[str, Any] | None) -> dict[str, Any] | None:
    if not grant:
        return None
    kept = {k: v for k, v in grant.items() if k not in _TOKEN_FIELDS}
    return kept or None


# ── URLs ────────────────────────────────────────────────────────────


def _origin(url: str) -> str:
    p = urlsplit(url)
    return f"{p.scheme}://{p.netloc}"


def canonical_resource(url: str) -> str:
    """The MCP server's URL as an RFC 8707 resource: lower-case scheme and
    host, no fragment, no trailing slash."""
    p = urlsplit(url.strip())
    return urlunsplit((p.scheme.lower(), p.netloc.lower(), p.path.rstrip("/"), p.query, ""))


def _covers(resource: str, server_url: str) -> bool:
    """True when a PRM's ``resource`` is this server's URL or a parent of it."""
    r = urlsplit(canonical_resource(resource))
    s = urlsplit(canonical_resource(server_url))
    if (r.scheme, r.netloc) != (s.scheme, s.netloc):
        return False
    return not r.path or s.path == r.path or s.path.startswith(r.path + "/")


def _check_endpoint(url: str, what: str, server_url: str) -> str:
    """An endpoint the deck (or the user's browser) is sent to must be https —
    plain http only on loopback, or on the very host the server was configured
    on over http (an admin's own choice, e.g. a LAN box)."""
    p = urlsplit(url or "")
    host = (p.hostname or "").lower()
    if p.scheme == "https" and host:
        return url
    if p.scheme == "http" and host:
        if host in _LOOPBACK_HOSTS or host.endswith(".localhost"):
            return url
        if host == (urlsplit(server_url).hostname or "").lower():
            return url
    raise MCPOAuthError(f"The server's {what} is not an https URL ({url!r}) — refusing to use it")


# ── WWW-Authenticate ────────────────────────────────────────────────


_AUTH_PARAM = re.compile(r'([A-Za-z_][A-Za-z0-9_\-]*)\s*=\s*(?:"((?:[^"\\]|\\.)*)"|([^\s,]+))')


def parse_www_authenticate(header: str) -> dict[str, str]:
    """The Bearer challenge's parameters (``resource_metadata``, ``scope``,
    ``error``…) — ``{}`` when the header has no Bearer challenge."""
    m = re.search(r"\bBearer\b", header or "", re.IGNORECASE)
    if not m:
        return {}
    out: dict[str, str] = {}
    for key, quoted, token in _AUTH_PARAM.findall(header[m.end():]):
        out.setdefault(key.lower(), quoted.replace('\\"', '"') if quoted else token)
    return out


# ── HTTP helpers ────────────────────────────────────────────────────


def _client() -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=_HTTP_TIMEOUT)


def _error_text(resp: httpx.Response) -> str:
    try:
        data = resp.json()
    except ValueError:
        return (resp.text or "").strip()[:300] or resp.reason_phrase
    if isinstance(data, dict) and (data.get("error") or data.get("error_description")):
        err = str(data.get("error") or "")
        desc = str(data.get("error_description") or "")
        return f"{err}: {desc}" if err and desc else (err or desc)
    return json.dumps(data)[:300]


async def _get_json(client: httpx.AsyncClient, url: str) -> dict[str, Any] | None:
    try:
        resp = await client.get(
            url,
            headers={"Accept": "application/json", "MCP-Protocol-Version": _PROTOCOL_VERSION},
            follow_redirects=True,
        )
    except httpx.HTTPError as exc:
        log.debug("MCP OAuth discovery fetch failed", url=url, error=str(exc))
        return None
    if resp.status_code != 200:
        return None
    try:
        data = resp.json()
    except ValueError:
        return None
    return data if isinstance(data, dict) else None


async def _challenge(
    client: httpx.AsyncClient, server_url: str, headers: dict[str, str],
) -> dict[str, str]:
    """POST an unauthenticated ``initialize`` and return the 401's Bearer
    challenge (``{}`` when the server let us in or sent none)."""
    try:
        resp = await client.post(
            server_url,
            json={
                "jsonrpc": "2.0",
                "id": 0,
                "method": "initialize",
                "params": {
                    "protocolVersion": _PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": {"name": _SOFTWARE_ID, "version": "0"},
                },
            },
            headers={
                **headers,
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream",
            },
        )
    except httpx.HTTPError as exc:
        raise MCPOAuthError(f"Could not reach the MCP server at {server_url}: {exc}") from exc
    if resp.status_code in (401, 403):
        return parse_www_authenticate(resp.headers.get("www-authenticate", ""))
    return {}


# ── discovery ───────────────────────────────────────────────────────


async def _authorization_server_metadata(
    client: httpx.AsyncClient, issuer: str,
) -> dict[str, Any] | None:
    p = urlsplit(issuer)
    origin = f"{p.scheme}://{p.netloc}"
    path = p.path.rstrip("/")
    if path:
        urls = [
            f"{origin}/.well-known/oauth-authorization-server{path}",
            f"{origin}/.well-known/openid-configuration{path}",
            f"{origin}{path}/.well-known/openid-configuration",
        ]
    else:
        urls = [
            f"{origin}/.well-known/oauth-authorization-server",
            f"{origin}/.well-known/openid-configuration",
        ]
    for url in urls:
        data = await _get_json(client, url)
        if data and data.get("authorization_endpoint") and data.get("token_endpoint"):
            return data
    return None


async def discover(
    server_url: str,
    *,
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Find the authorization server that guards ``server_url`` and return
    what a sign-in needs: issuer, endpoints, the ``resource`` to bind tokens
    to, and the scope to ask for (the challenge's, else the PRM's list)."""
    async with _client() as client:
        challenge = await _challenge(client, server_url, dict(headers or {}))
        origin = _origin(server_url)
        path = urlsplit(server_url).path.rstrip("/")
        candidates = []
        if challenge.get("resource_metadata"):
            candidates.append(challenge["resource_metadata"])
        if path:
            candidates.append(f"{origin}/.well-known/oauth-protected-resource{path}")
        candidates.append(f"{origin}/.well-known/oauth-protected-resource")
        prm: dict[str, Any] | None = None
        for url in candidates:
            data = await _get_json(client, url)
            if data and isinstance(data.get("authorization_servers"), list) and data["authorization_servers"]:
                prm = data
                break

        resource = canonical_resource(server_url)
        if prm:
            issuer = str(prm["authorization_servers"][0]).strip()
            claimed = str(prm.get("resource") or "").strip()
            if claimed and _covers(claimed, server_url):
                resource = claimed
            elif claimed:
                # Never ask for a token for some OTHER resource on a hostile
                # server's say-so — bind it to the URL we were configured with.
                log.warning(
                    "MCP OAuth: protected resource metadata names another resource",
                    server_url=server_url, claimed=claimed,
                )
        else:
            issuer = origin

        meta = await _authorization_server_metadata(client, issuer)
        fallback = meta is None
        if meta is None:
            base = _origin(issuer)
            meta = {
                "issuer": issuer,
                "authorization_endpoint": f"{base}/authorize",
                "token_endpoint": f"{base}/token",
                "registration_endpoint": f"{base}/register",
            }

    scope = challenge.get("scope") or ""
    if not scope and prm and isinstance(prm.get("scopes_supported"), list):
        scope = " ".join(str(s) for s in prm["scopes_supported"] if str(s).strip())
    registration = str(meta.get("registration_endpoint") or "")
    revocation = str(meta.get("revocation_endpoint") or "")
    try:
        revocation = _check_endpoint(revocation, "revocation endpoint", server_url) if revocation else ""
    except MCPOAuthError:
        revocation = ""  # optional — sign out just won't revoke
    methods = meta.get("code_challenge_methods_supported")
    if isinstance(methods, list) and "S256" not in methods:
        raise MCPOAuthError(
            "The server's authorization server does not support PKCE S256 — "
            "Flight Deck won't sign in without it"
        )
    return {
        "issuer": str(meta.get("issuer") or issuer),
        "authorization_endpoint": _check_endpoint(
            str(meta["authorization_endpoint"]), "authorization endpoint", server_url),
        "token_endpoint": _check_endpoint(str(meta["token_endpoint"]), "token endpoint", server_url),
        "registration_endpoint": (
            _check_endpoint(registration, "registration endpoint", server_url) if registration else ""
        ),
        "revocation_endpoint": revocation,
        "token_endpoint_auth_methods_supported": [
            str(m) for m in (meta.get("token_endpoint_auth_methods_supported") or [])
        ],
        "resource": resource,
        "scope": scope,
        "fallback": fallback,
    }


# ── client registration (RFC 7591) ──────────────────────────────────


def _auth_method(supported: list[str]) -> str:
    if "none" in supported:
        return "none"
    if "client_secret_basic" in supported and "client_secret_post" not in supported:
        return "client_secret_basic"
    return "client_secret_post"


async def register_client(discovered: dict[str, Any], redirect_uri: str) -> dict[str, Any]:
    """Register Flight Deck with the authorization server; return the client."""
    endpoint = discovered.get("registration_endpoint") or ""
    if not endpoint:
        raise MCPOAuthError(
            "This server's authorization server has no dynamic client registration "
            "(no registration_endpoint). Register Flight Deck with it by hand — "
            f"redirect URI {redirect_uri} — and enter the client ID (and secret) "
            "on the server's form."
        )
    method = _auth_method(discovered.get("token_endpoint_auth_methods_supported") or [])
    body: dict[str, Any] = {
        "client_name": CLIENT_NAME,
        "redirect_uris": [redirect_uri],
        "grant_types": ["authorization_code", "refresh_token"],
        "response_types": ["code"],
        "token_endpoint_auth_method": method,
        "software_id": _SOFTWARE_ID,
    }
    if discovered.get("scope"):
        body["scope"] = discovered["scope"]
    async with _client() as client:
        try:
            resp = await client.post(endpoint, json=body, headers={"Accept": "application/json"})
        except httpx.HTTPError as exc:
            raise MCPOAuthError(f"Client registration failed: {exc}") from exc
    if resp.status_code not in (200, 201):
        raise MCPOAuthError(
            f"Client registration was refused ({resp.status_code}): {_error_text(resp)}"
        )
    try:
        data = resp.json()
    except ValueError:
        data = None
    client_id = str((data or {}).get("client_id") or "") if isinstance(data, dict) else ""
    if not client_id:
        raise MCPOAuthError("Client registration returned no client_id")
    log.info("MCP OAuth client registered", issuer=discovered.get("issuer"), client_id=client_id)
    return {
        "issuer": discovered.get("issuer", ""),
        "redirect_uri": redirect_uri,
        "client_id": client_id,
        "client_secret": str(data.get("client_secret") or ""),
        "client_secret_expires_at": float(data.get("client_secret_expires_at") or 0.0),
        "token_endpoint_auth_method": str(data.get("token_endpoint_auth_method") or method),
    }


def _registration_usable(reg: dict[str, Any] | None, issuer: str, redirect_uri: str) -> bool:
    if not reg or not reg.get("client_id"):
        return False
    if reg.get("issuer") != issuer or reg.get("redirect_uri") != redirect_uri:
        return False
    expires = float(reg.get("client_secret_expires_at") or 0.0)
    return not expires or expires > time.time() + 300


def _client_auth(form: dict[str, str], headers: dict[str, str], client: dict[str, Any]) -> None:
    """Authenticate a token-endpoint request the way the client was registered."""
    client_id = str(client.get("client_id") or "")
    secret = str(client.get("client_secret") or "")
    method = str(client.get("token_endpoint_auth_method") or "")
    if secret and method == "client_secret_basic":
        pair = f"{quote(client_id, safe='')}:{quote(secret, safe='')}"
        headers["Authorization"] = "Basic " + base64.b64encode(pair.encode()).decode()
        return
    form["client_id"] = client_id
    if secret:
        form["client_secret"] = secret


# ── sign-in ─────────────────────────────────────────────────────────


def _pkce_pair() -> tuple[str, str]:
    verifier = secrets.token_urlsafe(64)
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    return verifier, base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


async def begin_sign_in(
    record: dict[str, Any], redirect_uri: str, state: str,
) -> tuple[str, dict[str, Any]]:
    """Discover, register if needed, and build the authorization URL.

    Returns ``(authorization_url, pending)``; the caller keeps ``pending``
    (it holds the PKCE verifier) until the callback hands it to
    :func:`complete_sign_in`.
    """
    name = str(record["name"])
    server_url = str(record["url"])
    discovered = await discover(server_url, headers=record.get("headers") or {})

    if record.get("client_id"):
        # A client registered by hand on the server's form.
        secret = str(record.get("client_secret") or "")
        client = {
            "client_id": str(record["client_id"]),
            "client_secret": secret,
            "token_endpoint_auth_method": "client_secret_post" if secret else "none",
        }
    else:
        stored = _grant_for(record) or {}
        reg = stored.get("registration")
        if not _registration_usable(reg, discovered["issuer"], redirect_uri):
            reg = await register_client(discovered, redirect_uri)

            def _keep_registration(old: dict[str, Any] | None) -> dict[str, Any]:
                base = dict(old or {}) if (old or {}).get("server_url") == server_url else {}
                base["server_url"] = server_url
                base["registration"] = reg
                return base

            # Kept now, not at the callback, so an abandoned sign-in doesn't
            # leave the next one to register yet another client.
            await _update(name, _keep_registration)
        client = {
            "client_id": reg["client_id"],
            "client_secret": reg.get("client_secret", ""),
            "token_endpoint_auth_method": reg.get("token_endpoint_auth_method", ""),
        }

    scope = str(record.get("scope") or "").strip() or discovered.get("scope") or ""
    verifier, challenge = _pkce_pair()
    params = {
        "response_type": "code",
        "client_id": client["client_id"],
        "redirect_uri": redirect_uri,
        "state": state,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "resource": discovered["resource"],
    }
    if scope:
        params["scope"] = scope
    endpoint = discovered["authorization_endpoint"]
    url = endpoint + ("&" if "?" in endpoint else "?") + urlencode(params)
    pending = {
        "server": name,
        "server_url": server_url,
        "discovered": discovered,
        "client": client,
        "verifier": verifier,
        "scope": scope,
        "redirect_uri": redirect_uri,
    }
    return url, pending


async def _token_request(
    endpoint: str, form: dict[str, str], headers: dict[str, str],
) -> dict[str, Any]:
    async with _client() as client:
        try:
            resp = await client.post(
                endpoint, data=form, headers={"Accept": "application/json", **headers},
            )
        except httpx.HTTPError as exc:
            raise MCPOAuthError(f"Token request failed: {exc}") from exc
    if resp.status_code != 200:
        text = _error_text(resp)
        if resp.status_code in (400, 401) and text.startswith(("invalid_grant", "invalid_client")):
            raise MCPOAuthRequired(text)
        raise MCPOAuthError(f"Token endpoint returned {resp.status_code}: {text}")
    try:
        data = resp.json()
    except ValueError:
        data = None
    access = str((data or {}).get("access_token") or "") if isinstance(data, dict) else ""
    if not access:
        raise MCPOAuthError("Token response had no access_token")
    expires_in = data.get("expires_in")
    try:
        expires_at = time.time() + float(expires_in) if expires_in else 0.0
    except (TypeError, ValueError):
        expires_at = 0.0
    return {
        "access_token": access,
        "refresh_token": str(data.get("refresh_token") or ""),
        "token_type": str(data.get("token_type") or "Bearer"),
        "expires_at": expires_at,
        "scope": str(data.get("scope") or ""),
    }


async def complete_sign_in(
    pending: dict[str, Any], code: str, *, connected_by: str, connected_by_id: str = "",
    issuer: str = "",
) -> dict[str, Any]:
    """Swap the callback's code for tokens and store the grant."""
    discovered = pending["discovered"]
    if issuer and discovered.get("issuer") and issuer != discovered["issuer"]:
        # RFC 9207: the response came from a different authorization server.
        raise MCPOAuthError("The sign-in answer came from an unexpected authorization server")
    form = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": pending["redirect_uri"],
        "code_verifier": pending["verifier"],
        "resource": discovered["resource"],
    }
    headers: dict[str, str] = {}
    _client_auth(form, headers, pending["client"])
    tokens = await _token_request(discovered["token_endpoint"], form, headers)
    server_url = pending["server_url"]

    def _store(old: dict[str, Any] | None) -> dict[str, Any]:
        base = dict(old or {}) if (old or {}).get("server_url") == server_url else {}
        base.update(tokens)
        base.update({
            "server_url": server_url,
            "issuer": discovered.get("issuer", ""),
            "token_endpoint": discovered["token_endpoint"],
            "revocation_endpoint": discovered.get("revocation_endpoint", ""),
            "resource": discovered["resource"],
            "scope": tokens["scope"] or pending.get("scope", ""),
            "client": pending["client"],
            "connected_at": time.time(),
            "connected_by": connected_by,
            "connected_by_id": connected_by_id,
        })
        return base

    grant = await _update(pending["server"], _store)
    log.info("MCP OAuth sign-in complete", server=pending["server"], issuer=discovered.get("issuer"))
    return grant or {}


# ── tokens for the transport ────────────────────────────────────────


_refresh_locks: dict[str, asyncio.Lock] = {}


def _expiring(grant: dict[str, Any]) -> bool:
    expires = float(grant.get("expires_at") or 0.0)
    return bool(expires) and time.time() > expires - _EXPIRY_SKEW_SECONDS


async def _refresh(name: str, grant: dict[str, Any]) -> dict[str, Any]:
    form = {
        "grant_type": "refresh_token",
        "refresh_token": str(grant["refresh_token"]),
        "resource": str(grant.get("resource") or ""),
    }
    if not form["resource"]:
        form.pop("resource")
    headers: dict[str, str] = {}
    _client_auth(form, headers, grant.get("client") or {})
    try:
        tokens = await _token_request(str(grant["token_endpoint"]), form, headers)
    except MCPOAuthRequired as exc:
        # The refresh token is dead: forget the tokens (keep the registration)
        # so the UI says "Not signed in" instead of a stale "Connected".
        await _update(name, _without_tokens)
        raise MCPOAuthRequired(_SIGN_IN_AGAIN.format(name=name, why=exc)) from exc
    server_url = grant.get("server_url")

    def _store(old: dict[str, Any] | None) -> dict[str, Any] | None:
        if not old or old.get("server_url") != server_url:
            return old  # signed out or re-pointed meanwhile — don't resurrect
        new = dict(old)
        new["access_token"] = tokens["access_token"]
        new["expires_at"] = tokens["expires_at"]
        new["token_type"] = tokens["token_type"]
        if tokens["refresh_token"]:
            new["refresh_token"] = tokens["refresh_token"]
        if tokens["scope"]:
            new["scope"] = tokens["scope"]
        return new

    return (await _update(name, _store)) or {}


async def access_token(
    record: dict[str, Any], *, force_refresh: bool = False, stale_token: str = "",
) -> str:
    """A usable access token for ``record``'s server, refreshed when due.

    ``force_refresh`` (after the server answered 401) refreshes even if the
    token looks fresh — unless a concurrent call already replaced
    ``stale_token``. Raises :class:`MCPOAuthRequired` when only a new
    sign-in will help.
    """
    name = str(record["name"])
    grant = _grant_for(record)
    if not grant or not grant.get("access_token"):
        raise MCPOAuthRequired(_NOT_SIGNED_IN.format(name=name))
    if not force_refresh and not _expiring(grant):
        return str(grant["access_token"])
    lock = _refresh_locks.setdefault(name, asyncio.Lock())
    async with lock:
        # Another call may have refreshed (or signed out) while this one waited.
        grant = _grant_for(record)
        if not grant or not grant.get("access_token"):
            raise MCPOAuthRequired(_NOT_SIGNED_IN.format(name=name))
        if force_refresh and stale_token and grant["access_token"] != stale_token:
            return str(grant["access_token"])
        if not force_refresh and not _expiring(grant):
            return str(grant["access_token"])
        if not grant.get("refresh_token") or not grant.get("token_endpoint"):
            expires = float(grant.get("expires_at") or 0.0)
            if force_refresh or (expires and time.time() > expires):
                raise MCPOAuthRequired(_SIGN_IN_AGAIN.format(name=name, why="no refresh token"))
            return str(grant["access_token"])
        fresh = await _refresh(name, grant)
        if not fresh.get("access_token"):
            raise MCPOAuthRequired(_NOT_SIGNED_IN.format(name=name))
        return str(fresh["access_token"])


async def sign_out(record: dict[str, Any]) -> None:
    """Forget the server's tokens (the registration stays for the next
    sign-in) and, when the server offers RFC 7009 revocation, revoke the
    refresh token — best-effort, a failure never keeps the tokens."""
    grant = _grant_for(record)
    await _update(str(record["name"]), _without_tokens)
    if not grant or not grant.get("revocation_endpoint"):
        return
    token = str(grant.get("refresh_token") or grant.get("access_token") or "")
    if not token:
        return
    form = {
        "token": token,
        "token_type_hint": "refresh_token" if grant.get("refresh_token") else "access_token",
    }
    headers: dict[str, str] = {}
    _client_auth(form, headers, grant.get("client") or {})
    try:
        async with _client() as client:
            await client.post(str(grant["revocation_endpoint"]), data=form, headers=headers)
    except httpx.HTTPError as exc:
        log.debug("MCP OAuth revocation failed (best-effort)", server=record.get("name"), error=str(exc))
