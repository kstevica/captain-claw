"""Persistent storage for Flight-Deck-managed MCP server configurations.

Phase 1 keeps things deliberately simple: a flat JSON file under
``~/.captain-claw-fd/mcp_servers.json`` containing a list of server
records. Anything fancier (SQLite, encryption-at-rest, per-tenant
scoping) is deferred until we have a clearer picture of what users
actually want.

Schema (one element per configured server)::

    {
        "name":          "fricmcp",                       # unique key
        "transport":     "http",                          # "http" (default) or "stdio"
        "url":           "https://example.com/mcp",       # http: streamable-http endpoint
        "command":       "uvx",                           # stdio: executable to launch
        "args":          ["mcp-server-foo", "--flag"],    # stdio: argv after command
        "env":           {"PATH": "/usr/bin"},            # stdio: extra env vars
        "client_id":     "abc",                           # optional OAuth2 client_id (http only)
        "client_secret": "xyz",                           # optional OAuth2 client_secret (http only)
        "token_endpoint": "/oauth/token",                 # optional, abs or rel (http only)
        "headers":       {"X-Foo": "bar"},                # optional extra HTTP headers (http only)
        "enabled":       true,                            # disabled servers are skipped
        "allowed_agents": ["nano-man", "everyday"],       # empty list = all agents allowed
        "added_at":      1715000000.0                     # unix seconds
    }

The file is rewritten atomically (write to ``*.tmp`` then ``rename``)
so a crashed FD never leaves callers with a half-written JSON.

The module exposes a tiny CRUD API so the routes layer doesn't need to
know about file locations or atomic writes:

* :func:`load_servers` — return the list (empty when no file)
* :func:`save_servers` — persist a list, overwriting prior contents
* :func:`get_server` / :func:`upsert_server` / :func:`delete_server` —
  by-name conveniences

A process-wide :class:`asyncio.Lock` serialises writes so concurrent
admin requests can't clobber each other.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any

from captain_claw.logging import get_logger

log = get_logger(__name__)


_DEFAULT_DIR = Path.home() / ".captain-claw-fd"
_DEFAULT_PATH = _DEFAULT_DIR / "mcp_servers.json"


def _storage_path() -> Path:
    override = os.environ.get("CAPTAIN_CLAW_FD_MCP_PATH", "").strip()
    if override:
        return Path(override).expanduser()
    return _DEFAULT_PATH


_write_lock = asyncio.Lock()


# ── helpers ─────────────────────────────────────────────────────────


def _coerce_record(raw: Any) -> dict[str, Any] | None:
    """Validate a record loaded from disk; drop entries we can't parse.

    Two transport flavours are accepted:

    * ``transport == "http"`` (default) — requires ``url``.
    * ``transport == "stdio"`` — requires ``command``; ``url`` is ignored.

    Records without the matching transport-specific field are dropped.
    """
    if not isinstance(raw, dict):
        return None
    name = str(raw.get("name") or "").strip()
    if not name:
        return None
    transport = str(raw.get("transport") or "http").strip().lower()
    if transport not in ("http", "stdio"):
        transport = "http"

    url = str(raw.get("url") or "").strip()
    command = str(raw.get("command") or "").strip()

    if transport == "http" and not url:
        return None
    if transport == "stdio" and not command:
        return None

    headers = raw.get("headers") or {}
    if not isinstance(headers, dict):
        headers = {}
    args = raw.get("args") or []
    if not isinstance(args, list):
        args = []
    env = raw.get("env") or {}
    if not isinstance(env, dict):
        env = {}
    allowed = raw.get("allowed_agents") or []
    if not isinstance(allowed, list):
        allowed = []
    allowed = [str(s).strip() for s in allowed if str(s).strip()]

    return {
        "name": name,
        "transport": transport,
        "url": url,
        "command": command,
        "args": [str(a) for a in args],
        "env": {str(k): str(v) for k, v in env.items()},
        "client_id": str(raw.get("client_id") or ""),
        "client_secret": str(raw.get("client_secret") or ""),
        "token_endpoint": str(raw.get("token_endpoint") or ""),
        "headers": {str(k): str(v) for k, v in headers.items()},
        "enabled": bool(raw.get("enabled", True)),
        "allowed_agents": allowed,
        "added_at": float(raw.get("added_at") or 0.0),
    }


def _public_view(record: dict[str, Any]) -> dict[str, Any]:
    """Return a record with secrets masked for UI consumption."""
    out = dict(record)
    secret = out.get("client_secret") or ""
    if secret:
        out["client_secret"] = "•" * 8  # placeholder
        out["client_secret_set"] = True
    else:
        out["client_secret"] = ""
        out["client_secret_set"] = False
    return out


# ── load / save ─────────────────────────────────────────────────────


def load_servers() -> list[dict[str, Any]]:
    path = _storage_path()
    if not path.exists():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        log.warning("Failed to read MCP servers file %s: %s", path, exc)
        return []
    if not isinstance(raw, list):
        log.warning("MCP servers file %s did not contain a list; ignoring", path)
        return []
    out: list[dict[str, Any]] = []
    for entry in raw:
        rec = _coerce_record(entry)
        if rec is not None:
            out.append(rec)
    return out


async def save_servers(records: list[dict[str, Any]]) -> None:
    """Atomically persist ``records``. Caller is expected to validate
    each record (it must already be in canonical shape)."""
    path = _storage_path()
    async with _write_lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        body = json.dumps(records, indent=2, sort_keys=True)
        tmp.write_text(body, encoding="utf-8")
        os.replace(tmp, path)


# ── by-name conveniences ────────────────────────────────────────────


def get_server(name: str) -> dict[str, Any] | None:
    name = name.strip()
    if not name:
        return None
    for rec in load_servers():
        if rec["name"] == name:
            return rec
    return None


def is_agent_allowed(record: dict[str, Any], agent_slug: str) -> bool:
    """Return ``True`` when ``agent_slug`` is allowed to use ``record``.

    The allow-list is opt-in: an empty ``allowed_agents`` list means
    "every agent is allowed" (the Phase 1 behaviour). Once any slug is
    listed, only those slugs may use the server.

    A ``""`` slug (caller couldn't determine its identity) is allowed
    only when ``allowed_agents`` is empty — gating on identity requires
    *having* an identity.
    """
    allowed = record.get("allowed_agents") or []
    if not allowed:
        return True
    if not agent_slug:
        return False
    return agent_slug in allowed


async def upsert_server(record: dict[str, Any]) -> dict[str, Any]:
    """Insert-or-replace by ``name``. Returns the canonicalised record."""
    canonical = _coerce_record(record)
    if canonical is None:
        raise ValueError(
            "MCP server record requires 'name' plus either 'url' (http) or 'command' (stdio)"
        )
    if not canonical["added_at"]:
        canonical["added_at"] = time.time()

    existing = load_servers()
    replaced = False
    out: list[dict[str, Any]] = []
    for rec in existing:
        if rec["name"] == canonical["name"]:
            # Preserve previous added_at on update
            if rec.get("added_at"):
                canonical["added_at"] = rec["added_at"]
            out.append(canonical)
            replaced = True
        else:
            out.append(rec)
    if not replaced:
        out.append(canonical)
    await save_servers(out)
    return canonical


async def delete_server(name: str) -> bool:
    name = name.strip()
    if not name:
        return False
    existing = load_servers()
    kept = [rec for rec in existing if rec["name"] != name]
    if len(kept) == len(existing):
        return False
    await save_servers(kept)
    return True


# ── public-view helpers (used by routes) ────────────────────────────


def list_servers_public() -> list[dict[str, Any]]:
    """Return all configured servers with secrets masked."""
    return [_public_view(rec) for rec in load_servers()]


def get_server_public(name: str) -> dict[str, Any] | None:
    rec = get_server(name)
    if rec is None:
        return None
    return _public_view(rec)
