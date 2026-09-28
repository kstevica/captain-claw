"""Browser-facing request guard for Flight Deck: Host allowlist + origin checks.

Flight Deck trusts loopback in many places (agent routes, connector token
endpoints) and, with ``FD_AUTH_ENABLED=false`` (the desktop build), trusts
every caller. Two browser tricks turn "trusts loopback" into "trusts any web
page the user visits":

* **Cross-site requests.** A page on ``https://evil.example`` can ``fetch()``
  ``http://localhost:25080/...`` — or open a WebSocket to it, which CORS never
  covers — from the user's own browser, i.e. from loopback.
* **DNS rebinding.** ``evil.example`` re-resolves to ``127.0.0.1``; the page is
  then *same-origin* with Flight Deck as far as the browser is concerned and
  sends no ``Origin`` at all. Only the ``Host`` header still says
  ``evil.example``.

:class:`BrowserGuardMiddleware` closes both for every HTTP and WebSocket route:

1. **Host allowlist.** ``Host`` must be a name this deck answers to:
   ``localhost`` / ``*.localhost``, any IP literal (an IP can't be rebound),
   ``host.docker.internal`` (containerised agents), this machine's own
   hostname(s), ``FD_PUBLIC_URL``'s host, the hosts in ``FD_CORS_ORIGINS``, and
   ``FD_ALLOWED_HOSTS`` (comma-separated; ``.example.com`` / ``*.example.com``
   also allow subdomains; ``*`` switches the check off). LAN access by IP or by
   this machine's name keeps working with no configuration; a deck reached
   through its own domain (reverse proxy) needs that domain in
   ``FD_ALLOWED_HOSTS`` or ``FD_PUBLIC_URL``.
2. **Origin check.** A request carrying ``Origin`` (every browser WebSocket,
   every cross-origin fetch, every POST) must come from one of this deck's own
   origins — the ``Host`` (or a proxy's ``X-Forwarded-Host``) it was sent to,
   ``FD_PUBLIC_URL``, a configured host above, ``FD_CORS_ORIGINS``, or the Vite
   dev server. Without ``Origin``, a ``Sec-Fetch-Site: cross-site`` request is
   refused unless it is a plain top-level navigation (links, OAuth redirects).

Non-browser callers — agents, the Lupa BFF, webhooks, curl — send neither
``Origin`` nor ``Sec-Fetch-*`` and only meet the Host check. Every refusal is
logged (rate-limited) with the setting that would allow it.
"""

from __future__ import annotations

import ipaddress
import json
import logging
import os
import re
import socket
import time
from typing import NamedTuple
from urllib.parse import urlsplit

from starlette.datastructures import Headers
from starlette.middleware.cors import CORSMiddleware
from starlette.types import ASGIApp, Receive, Scope, Send

log = logging.getLogger("flight_deck.origin_guard")

# Containerised agents reach the host's FD as host.docker.internal:<port>.
_DOCKER_HOST_NAMES = frozenset({"host.docker.internal"})

# The Vite dev server (flight-deck/vite.config.ts) proxies /fd to FD with
# changeOrigin — Host is rewritten, Origin is not.
_DEV_ORIGINS = frozenset({("http", "localhost", 5173), ("http", "127.0.0.1", 5173)})

# A Host header is a DNS name or IP literal with an optional port — nothing
# else (userinfo, paths, whitespace) is ever sent by a browser.
_HOST_HEADER_RE = re.compile(
    r"^(?:[A-Za-z0-9_](?:[A-Za-z0-9_.\-]*[A-Za-z0-9_.])?|\[[0-9A-Fa-f:.%A-Za-z]+\])(?::\d{1,5})?$"
)


class _Origin(NamedTuple):
    scheme: str
    host: str
    port: int


# ── env ──────────────────────────────────────────────────────────────


def _split_env(name: str) -> list[str]:
    return [p.strip() for p in os.environ.get(name, "").split(",") if p.strip()]


def _norm_host(host: str) -> str:
    h = (host or "").strip().lower().rstrip(".")
    if h.startswith("[") and h.endswith("]"):
        h = h[1:-1]
    return h


def split_host_header(value: str) -> tuple[str, int | None] | None:
    """``"Example.com:8080"`` → ``("example.com", 8080)``; ``"[::1]:25080"`` →
    ``("::1", 25080)``. None when the value isn't a well-formed Host."""
    v = (value or "").strip()
    if not v or not _HOST_HEADER_RE.match(v):
        return None
    try:
        parts = urlsplit("//" + v)
        host, port = parts.hostname, parts.port
    except ValueError:
        return None
    if not host:
        return None
    return _norm_host(host), port


def _parse_origin(value: str) -> _Origin | None:
    """Parse an ``Origin`` (or any URL) to (scheme, host, port) with the default
    port filled in. ``null``, non-http(s) and malformed values → None."""
    v = (value or "").strip()
    if not v or v == "null":
        return None
    try:
        parts = urlsplit(v)
        scheme = parts.scheme.lower()
        host = _norm_host(parts.hostname or "")
        port = parts.port
    except ValueError:
        return None
    if scheme not in ("http", "https") or not host:
        return None
    return _Origin(scheme, host, port if port is not None else (443 if scheme == "https" else 80))


def _public_url() -> str:
    return os.environ.get("FD_PUBLIC_URL", "").strip()


def _cors_origins() -> list[str]:
    return _split_env("FD_CORS_ORIGINS")


def _configured_host_patterns() -> list[str]:
    """Host names an operator declared as this deck's own: FD_ALLOWED_HOSTS
    entries (``*`` excluded — that only switches the Host check off),
    FD_PUBLIC_URL's host and the hosts of FD_CORS_ORIGINS."""
    pats: list[str] = []
    for entry in _split_env("FD_ALLOWED_HOSTS"):
        if entry == "*":
            continue
        if entry.startswith("*.") or entry.startswith("."):
            pats.append(entry.lower().rstrip("."))
            continue
        parsed = split_host_header(entry)
        if parsed:
            pats.append(parsed[0])
    pub = _parse_origin(_public_url())
    if pub:
        pats.append(pub.host)
    for o in _cors_origins():
        parsed_o = _parse_origin(o)
        if parsed_o:
            pats.append(parsed_o.host)
    return pats


def _host_matches(host: str, pattern: str) -> bool:
    if pattern.startswith("*."):
        pattern = pattern[1:]
    if pattern.startswith("."):
        return host == pattern[1:] or host.endswith(pattern)
    return host == pattern


_machine_hosts_cache: frozenset[str] | None = None


def _machine_hosts() -> frozenset[str]:
    """This machine's own names (``mymac``, ``mymac.local``, ``mymac.lan``…), so
    LAN access by hostname works without configuration. gethostname() only —
    getfqdn() can block on DNS."""
    global _machine_hosts_cache
    if _machine_hosts_cache is None:
        names: set[str] = set()
        try:
            hn = _norm_host(socket.gethostname())
        except OSError:
            hn = ""
        if hn:
            short = hn.split(".", 1)[0]
            names.update({hn, short, f"{short}.local"})
        _machine_hosts_cache = frozenset(n for n in names if n)
    return _machine_hosts_cache


def _is_ip_literal(host: str) -> bool:
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        return False


# ── policy ───────────────────────────────────────────────────────────


def host_check_disabled() -> bool:
    return "*" in _split_env("FD_ALLOWED_HOSTS")


def host_allowed(host: str) -> bool:
    """Is *host* (a bare, normalised hostname) a name this deck answers to?"""
    if not host:
        return False
    if host == "localhost" or host.endswith(".localhost"):
        return True
    if _is_ip_literal(host):  # an IP literal can't be DNS-rebound
        return True
    if host in _DOCKER_HOST_NAMES or host in _machine_hosts():
        return True
    return any(_host_matches(host, p) for p in _configured_host_patterns())


def _origin_matches_host(o: _Origin, host_value: str) -> bool:
    parsed = split_host_header(host_value)
    if not parsed:
        return False
    host, port = parsed
    if host != o.host:
        return False
    if port is None:  # a proxy that drops the default port
        return o.port in (80, 443)
    return port == o.port


def _first_forwarded_host(headers: Headers) -> str:
    return (headers.get("x-forwarded-host") or "").split(",")[0].strip()


def origin_trusted(
    origin: str,
    *,
    host_header: str = "",
    forwarded_host: str = "",
    allow_wildcard: bool = True,
) -> bool:
    """Is *origin* one of this deck's own origins?

    Same-origin with the request (``Host``, or a proxy's ``X-Forwarded-Host`` —
    a browser can't set that header cross-origin without a preflight, which is
    itself refused), FD_PUBLIC_URL, a configured host (FD_ALLOWED_HOSTS /
    FD_PUBLIC_URL / FD_CORS_ORIGINS hosts, any scheme or port), an entry of
    FD_CORS_ORIGINS, or the Vite dev server. ``FD_CORS_ORIGINS=*`` trusts every
    origin unless *allow_wildcard* is False.
    """
    configured = _cors_origins()
    if allow_wildcard and "*" in configured:
        return True
    o = _parse_origin(origin)
    if o is None:
        return False
    if host_header and _origin_matches_host(o, host_header):
        return True
    if forwarded_host and _origin_matches_host(o, forwarded_host):
        return True
    if any(_parse_origin(c) == o for c in configured if c != "*"):
        return True
    if _parse_origin(_public_url()) == o:
        return True
    if any(_host_matches(o.host, p) for p in _configured_host_patterns()):
        return True
    return (o.scheme, o.host, o.port) in _DEV_ORIGINS


def cors_origin_allowed(origin: str) -> bool:
    """CORS allowlist: this deck's own origins (no Host to compare against —
    a same-origin request needs no CORS headers anyway)."""
    return origin_trusted(origin)


def browser_refusal(method: str, headers: Headers) -> str | None:
    """Why a browser-originated request must be refused, or None."""
    origin = headers.get("origin")
    if origin is not None:
        if origin_trusted(
            origin,
            host_header=headers.get("host", ""),
            forwarded_host=_first_forwarded_host(headers),
        ):
            return None
        return "origin"
    site = (headers.get("sec-fetch-site") or "").lower()
    if site == "cross-site":
        mode = (headers.get("sec-fetch-mode") or "").lower()
        dest = (headers.get("sec-fetch-dest") or "").lower()
        if method in ("GET", "HEAD") and mode == "navigate" and dest not in ("object", "embed"):
            return None  # a link / redirect landing on FD — no page can read it
        return "cross-site"
    return None


def may_expose_agent_secrets(headers: Headers) -> bool:
    """Whether a response may carry agents' secrets (``web_auth`` tokens, env).

    Yes for non-browser callers and for this deck's own pages (same-origin, or an
    explicitly configured origin). No for anything else a browser sends — even
    when ``FD_CORS_ORIGINS=*`` opened CORS up, another site never gets tokens
    that let it drive an agent directly.
    """
    origin = headers.get("origin")
    if origin is not None:
        return origin_trusted(
            origin,
            host_header=headers.get("host", ""),
            forwarded_host=_first_forwarded_host(headers),
            allow_wildcard=False,
        )
    site = (headers.get("sec-fetch-site") or "").lower()
    return site in ("", "same-origin", "none")


def describe_policy() -> str:
    """One line for the startup log."""
    if host_check_disabled():
        hosts = "DISABLED (FD_ALLOWED_HOSTS=*)"
    else:
        extra = sorted(set(_configured_host_patterns()))
        hosts = ("localhost, IP addresses, host.docker.internal, "
                 + ", ".join(sorted(_machine_hosts()))
                 + (f", {', '.join(extra)}" if extra else ""))
    cors = _cors_origins()
    origins = ("ANY (FD_CORS_ORIGINS=*)" if "*" in cors
               else "this deck's own" + (f" + {', '.join(cors)}" if cors else ""))
    return f"host allowlist: {hosts}; browser origins: {origins}"


# ── refusal logging ──────────────────────────────────────────────────

_LOG_EVERY_S = 60.0
_last_logged: dict[tuple[str, str], tuple[float, int]] = {}


def _log_refusal(kind: str, value: str, method: str, path: str) -> None:
    """Warn about a refusal — at most once a minute per (kind, value), with a
    count of the ones suppressed in between."""
    key = (kind, value)
    now = time.monotonic()
    last, suppressed = _last_logged.get(key, (0.0, 0))
    if last and now - last < _LOG_EVERY_S:
        _last_logged[key] = (last, suppressed + 1)
        return
    if len(_last_logged) > 1000:
        _last_logged.clear()
    _last_logged[key] = (now, 0)
    more = f" (+{suppressed} more since last report)" if suppressed else ""
    if kind == "host":
        log.warning(
            "Flight Deck refused %s %s: Host %r is not a name this deck answers to "
            "(DNS-rebinding guard)%s. If it is a legitimate name for this Flight Deck, "
            "add it to FD_ALLOWED_HOSTS (comma-separated; '.example.com' allows "
            "subdomains) or set FD_PUBLIC_URL.", method, path, value, more)
    elif kind == "origin":
        log.warning(
            "Flight Deck refused %s %s from browser origin %r: not one of this deck's "
            "own origins (cross-site request guard)%s. If that site is a legitimate "
            "Flight Deck frontend, add it to FD_CORS_ORIGINS.", method, path, value, more)
    else:
        log.warning(
            "Flight Deck refused cross-site %s %s (Sec-Fetch-Site: cross-site, no "
            "Origin; cross-site request guard)%s.", method, path, more)


_DETAILS = {
    "host": ("Host not allowed for this Flight Deck — add it to FD_ALLOWED_HOSTS "
             "(or set FD_PUBLIC_URL) on the Flight Deck server"),
    "origin": ("Cross-origin request refused — this Flight Deck only accepts "
               "browser requests from its own origin (see FD_CORS_ORIGINS)"),
    "cross-site": "Cross-site request refused by Flight Deck",
}


# ── ASGI middleware ──────────────────────────────────────────────────


class BrowserGuardMiddleware:
    """Refuse requests whose Host isn't this deck's, or that a browser sent from
    another site (HTTP and WebSocket alike). Install outermost."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return
        headers = Headers(scope=scope)
        method = scope.get("method", "GET") if scope["type"] == "http" else "GET"
        kind = ""
        value = ""
        host_value = headers.get("host", "")
        if host_value and not host_check_disabled():
            parsed = split_host_header(host_value)
            if parsed is None or not host_allowed(parsed[0]):
                kind, value = "host", host_value
        if not kind:
            reason = browser_refusal(method, headers)
            if reason:
                kind, value = reason, headers.get("origin") or ""
        if not kind:
            await self.app(scope, receive, send)
            return

        _log_refusal(kind, value, method, scope.get("path", ""))
        detail = _DETAILS[kind]
        if scope["type"] == "websocket":
            # Closing before accept makes the server answer the handshake 403.
            await send({"type": "websocket.close", "code": 4403, "reason": detail[:120]})
            return
        body = json.dumps({"detail": detail}).encode("utf-8")
        await send({
            "type": "http.response.start",
            "status": 403,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode("ascii")),
            ],
        })
        await send({"type": "http.response.body", "body": body})


class FDCORSMiddleware(CORSMiddleware):
    """Starlette's CORS, with the allowlist resolved per request by
    :func:`cors_origin_allowed` (this deck's own origins + FD_CORS_ORIGINS)
    instead of a fixed list — so FD_PUBLIC_URL / FD_ALLOWED_HOSTS set after
    import still apply, and nothing ever mirrors an arbitrary origin unless the
    operator opted into ``FD_CORS_ORIGINS=*``."""

    def is_allowed_origin(self, origin: str) -> bool:
        return cors_origin_allowed(origin)
