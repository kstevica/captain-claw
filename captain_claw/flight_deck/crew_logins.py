"""Crew — whose account a worker runs on. Each user's own, never the deck owner's.

Saved logins are sealed in ``user_settings`` as ``crew:logins`` (``fd_crypto``);
``crew:meta`` keeps a hint and the date for the page. No route returns a login.

* **claude** — a token from ``claude setup-token`` (billed to their plan), or an
  Anthropic API key.
* **codex** — an OpenAI API key, or the contents of their ``~/.codex/auth.json``.
* **github** — a token. Flight Deck uses it to clone, push and open pull
  requests; the worker never sees it, since a worker never pushes.
* **anthropic, openai, gemini, openrouter, xai** — provider keys for OpenCode and
  Pi. Keys saved under "Your own AI keys" (``byok``) count as well.

A worker's environment is built from nothing (:data:`_BASE_ENV` is all it takes
from Flight Deck's), so no deck secret and no deck key reaches it. A non-admin's
worker gets a home folder of its own, which keeps the deck user's logins out of
its reach through the CLIs' normal lookups. An admin's uses this computer's home
and the logins already there, unless they saved their own here.
"""

from __future__ import annotations

import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import HTTPException

from captain_claw.flight_deck import byok, crew_harness, crew_store, fd_crypto
from captain_claw.flight_deck.crew_access import Grant
from captain_claw.logging import get_logger

log = get_logger(__name__)

_K_LOGINS = "crew:logins"
_K_META = "crew:meta"
_CLAUDE_TOKEN_PREFIX = "sk-ant-oat"
PROVIDER_ENV = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "xai": "XAI_API_KEY",
}
LOGIN_NAMES = ("claude", "codex", "github", *PROVIDER_ENV)
# OpenSSH finds keys through the passwd home, not $HOME: without this, a plain
# `git push` over SSH from someone else's worker would sign in as the deck's user.
NO_SSH_IDENTITY = "ssh -F /dev/null -o BatchMode=yes -o IdentitiesOnly=yes -o IdentityFile=/dev/null"
_MAX_SECRET = 64_000
_BASE_ENV = (
    "PATH", "LANG", "LANGUAGE", "LC_ALL", "LC_CTYPE", "TERM", "TMPDIR", "TZ", "USER", "LOGNAME", "SHELL",
    "SSL_CERT_FILE", "SSL_CERT_DIR", "NODE_EXTRA_CA_CERTS", "REQUESTS_CA_BUNDLE",
    "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "http_proxy", "https_proxy", "no_proxy",
    "NVM_DIR", "NVM_BIN",
)


def hint(secret: str) -> str:
    return f"…{secret[-4:]}" if len(secret) > 8 and not secret.lstrip().startswith("{") else "saved"


async def read_logins(db, uid: str) -> dict[str, str]:
    token = await db.get_setting(uid, _K_LOGINS) or ""
    if not token:
        return {}
    try:
        logins = fd_crypto.unseal(token)
    except fd_crypto.CryptoUnavailable:
        log.warning("Crew logins can't be read: encryption unavailable", user_id=uid)
        return {}
    return {str(k): str(v) for k, v in logins.items() if k in LOGIN_NAMES and v} if isinstance(logins, dict) else {}


async def read_meta(db, uid: str) -> dict[str, Any]:
    try:
        meta = json.loads(await db.get_setting(uid, _K_META) or "{}")
    except (TypeError, ValueError):
        meta = {}
    return meta if isinstance(meta, dict) else {}


async def save_login(db, uid: str, name: str, secret: str) -> dict[str, Any]:
    if name not in LOGIN_NAMES:
        raise HTTPException(404, "Unknown login")
    secret = (secret or "").strip()
    if not secret:
        raise HTTPException(400, "Paste the token or key to save.")
    if len(secret) > _MAX_SECRET:
        raise HTTPException(400, "That is too long to be a token or key.")
    logins = await read_logins(db, uid)
    logins[name] = secret
    try:
        sealed = fd_crypto.seal(logins)
    except fd_crypto.CryptoUnavailable as exc:
        raise HTTPException(503, f"Logins can't be stored safely on this deck: {exc}") from None
    meta = await read_meta(db, uid)
    meta[name] = {"hint": hint(secret), "saved_at": datetime.now(UTC).isoformat()}
    await db.set_settings(uid, {_K_LOGINS: sealed, _K_META: json.dumps(meta, ensure_ascii=True)})
    if name == "codex":
        # The old account's file goes: the next turn writes this one.
        (_codex_home(uid) / "auth.json").unlink(missing_ok=True)
    return meta[name]


async def delete_login(db, uid: str, name: str) -> None:
    logins = await read_logins(db, uid)
    logins.pop(name, None)
    meta = await read_meta(db, uid)
    meta.pop(name, None)
    if logins:
        await db.set_settings(uid, {_K_LOGINS: fd_crypto.seal(logins)})
    else:
        await db.delete_setting(uid, _K_LOGINS)
    await db.set_settings(uid, {_K_META: json.dumps(meta, ensure_ascii=True)})
    if name == "codex":
        (_codex_home(uid) / "auth.json").unlink(missing_ok=True)


async def _provider_keys(db, uid: str, logins: dict[str, str]) -> dict[str, str]:
    """Keys by provider: what they saved for Crew, then their own AI keys."""
    keys = {p: k for p, k in (await byok.read_keys(db, uid)).items() if p in PROVIDER_ENV}
    keys.update({p: logins[p] for p in PROVIDER_ENV if logins.get(p)})
    return keys


def worker_home(uid: str) -> Path:
    home = crew_store.user_dir(uid) / "home"
    home.mkdir(exist_ok=True)
    return home


def _codex_home(uid: str) -> Path:
    path = worker_home(uid) / ".codex"
    path.mkdir(exist_ok=True)
    return path


def _codex_login_is_file(secret: str) -> bool:
    return secret.lstrip().startswith("{")


async def login_state(db, user: dict, grant: Grant, harness: str) -> dict[str, Any]:
    """Whether a worker on *harness* has an account to run on, and which."""
    uid = user["id"]
    logins = await read_logins(db, uid)
    keys = await _provider_keys(db, uid, logins)
    if harness == "claude" and logins.get("claude"):
        return {"ok": True, "how": "your saved login"}
    if harness == "codex" and logins.get("codex"):
        return {"ok": True, "how": "your saved login"}
    needed = {"claude": ("anthropic",), "codex": ("openai",)}.get(harness, tuple(PROVIDER_ENV))
    found = [p for p in needed if keys.get(p)]
    if found:
        names = ", ".join(byok.PROVIDERS.get(p, p) for p in found)
        return {"ok": True, "how": f"your {names} key"}
    if grant.trusted:
        return {"ok": True, "how": "this computer's login"}
    return {"ok": False, "how": ""}


def _base_env(binary: str) -> dict[str, str]:
    env = {name: os.environ[name] for name in _BASE_ENV if os.environ.get(name)}
    folder = str(Path(binary).parent)
    env["PATH"] = os.pathsep.join(p for p in (folder, env.get("PATH", "")) if p)
    env.setdefault("LANG", "en_US.UTF-8")
    env["NO_COLOR"] = "1"
    env["CAPTAIN_CREW"] = "1"
    return env


async def worker_env(db, user: dict, grant: Grant, harness: str, binary: str) -> dict[str, str]:
    """The whole environment of a worker's turn. Raises when it has no account
    to run on: never falls through to the deck owner's."""
    uid = user["id"]
    state = await login_state(db, user, grant, harness)
    if not state["ok"]:
        label = crew_harness.HARNESSES[harness].label
        raise HTTPException(400, f"Add your {label} login in Crew → Logins first.")
    logins = await read_logins(db, uid)
    keys = await _provider_keys(db, uid, logins)
    env = _base_env(binary)
    env.update(crew_harness.HARNESSES[harness].env)
    if grant.trusted:
        env["HOME"] = os.environ.get("HOME") or str(Path.home())
    else:
        home = worker_home(uid)
        env["HOME"] = str(home)
        for name, sub in (("XDG_CONFIG_HOME", ".config"), ("XDG_DATA_HOME", ".local/share"),
                          ("XDG_STATE_HOME", ".local/state"), ("XDG_CACHE_HOME", ".cache")):
            env[name] = str(home / sub)
        who = (user.get("display_name") or user.get("email") or "Captain Crew").strip()
        mail = (user.get("email") or "crew@captain-claw.local").strip()
        env.update(GIT_AUTHOR_NAME=who, GIT_AUTHOR_EMAIL=mail, GIT_COMMITTER_NAME=who, GIT_COMMITTER_EMAIL=mail,
                   GIT_CONFIG_NOSYSTEM="1", GIT_SSH_COMMAND=NO_SSH_IDENTITY)
    if harness == "claude":
        secret = logins.get("claude", "")
        if secret.startswith(_CLAUDE_TOKEN_PREFIX):
            env["CLAUDE_CODE_OAUTH_TOKEN"] = secret
        elif secret or keys.get("anthropic"):
            env["ANTHROPIC_API_KEY"] = secret or keys["anthropic"]
    elif harness == "codex":
        secret = logins.get("codex", "")
        if not grant.trusted or secret:
            env["CODEX_HOME"] = str(_codex_home(uid))
        if _codex_login_is_file(secret):
            auth = _codex_home(uid) / "auth.json"
            # Codex refreshes its tokens in this file: written once, then it's Codex's.
            if not auth.exists():
                auth.write_text(secret, encoding="utf-8")
                os.chmod(auth, 0o600)
        elif secret or keys.get("openai"):
            env["CODEX_API_KEY"] = env["OPENAI_API_KEY"] = secret or keys["openai"]
    else:
        env.update({PROVIDER_ENV[p]: k for p, k in keys.items()})
    return env


async def github_token(db, user: dict, grant: Grant) -> str:
    """Their token; on this computer's own account, ``gh``'s login."""
    saved = (await read_logins(db, user["id"])).get("github", "")
    if saved or not grant.trusted:
        return saved
    try:
        probe = subprocess.run(["gh", "auth", "token"], capture_output=True, text=True, timeout=10,
                               stdin=subprocess.DEVNULL)
    except (OSError, subprocess.SubprocessError):
        return ""
    return probe.stdout.strip() if probe.returncode == 0 else ""
