"""Crew — whose account a worker runs on. Each user's own, never the deck owner's.

Saved logins are sealed in ``user_settings`` as ``crew:logins`` (``fd_crypto``);
``crew:meta`` keeps a hint and the date for the page. No route returns a login.

* **claude** — a token from ``claude setup-token`` (billed to their plan), or an
  Anthropic API key.
* **codex** — an OpenAI API key, or the contents of their ``~/.codex/auth.json``.
* **github** — a token. Flight Deck uses it to clone, push and open pull
  requests; the worker never sees it, since a worker never pushes.
* **anthropic, openai, gemini, openrouter, xai** — provider keys for OpenCode and
  Pi. Keys saved under "Your own AI keys" (``byok``) count as well. OpenCode also
  has free models that need no key; Pi can't run without one.

``crew:models`` holds the model each of their CLIs uses when a task names none: a
model picked in a CLI's own screen is in that person's home, which their worker
doesn't share.

``crew:own_model`` is a model server of their own (sealed, for its key): OpenCode
and Pi run on it as ``own/<model>``. The key reaches a worker only in the environment
of its turn, so it's in no file: OpenCode's config comes through
``OPENCODE_CONFIG_CONTENT``, and Pi's ``models.json`` names the variable.

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
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
from fastapi import HTTPException

from captain_claw.flight_deck import byok, crew_harness, crew_store, fd_crypto
from captain_claw.flight_deck.crew_access import Grant
from captain_claw.logging import get_logger

log = get_logger(__name__)

_K_LOGINS = "crew:logins"
_K_META = "crew:meta"
_K_MODELS = "crew:models"
_K_OWN = "crew:own_model"
OWN_PROVIDER = "own"
OWN_KEY_ENV = "CREW_OWN_MODEL_KEY"
OWN_HARNESSES = ("opencode", "pi")
_OWN_MODELS_MAX = 20
_MODEL_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@+\-]{0,119}$")
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


def _clean_url(value: str) -> str:
    """An OpenAI-compatible address. It may well be on a private network: that's the
    point, and a worker reaches what it likes anyway. No braces: OpenCode expands
    {env:…} and {file:…} in its config."""
    url = (value or "").strip().rstrip("/")
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username or parsed.password:
        raise HTTPException(400, "Give its address as http://host:port/v1.")
    if re.search(r"[\s{}\"'\\]", url):
        raise HTTPException(400, "That address has characters an address can't have.")
    return url if parsed.path else f"{url}/v1"


def _clean_models(value: Any) -> list[str]:
    names = value if isinstance(value, list) else re.split(r"[,\s]+", str(value or ""))
    out = list(dict.fromkeys(str(n).strip() for n in names if str(n).strip()))
    if not out:
        raise HTTPException(400, "Name at least one model it serves.")
    if len(out) > _OWN_MODELS_MAX or not all(_MODEL_ID_RE.match(n) for n in out):
        raise HTTPException(400, f"Give up to {_OWN_MODELS_MAX} model names, as the server calls them.")
    return out


async def read_own_model(db, uid: str) -> dict[str, Any]:
    token = await db.get_setting(uid, _K_OWN) or ""
    if not token:
        return {}
    try:
        own = fd_crypto.unseal(token)
    except fd_crypto.CryptoUnavailable:
        log.warning("Crew's own model can't be read: encryption unavailable", user_id=uid)
        return {}
    return own if isinstance(own, dict) else {}


def own_model_names(own: dict[str, Any]) -> list[str]:
    return [f"{OWN_PROVIDER}/{m}" for m in own.get("models") or []]


def own_model_view(own: dict[str, Any]) -> dict[str, Any] | None:
    """What the page may see of it: never the key."""
    if not own:
        return None
    return {"url": own.get("url") or "", "models": own.get("models") or [], "names": own_model_names(own),
            "context": own.get("context") or 0, "has_key": bool(own.get("key")), "saved_at": own.get("saved_at") or ""}


async def save_own_model(db, uid: str, url: str, key: str, models: Any, context: Any) -> dict[str, Any]:
    clean = _clean_url(url)
    names = _clean_models(models)
    old = await read_own_model(db, uid)
    # A blank key keeps the saved one, but only for the server it was saved for.
    secret = (key or "").strip() or (old.get("key", "") if old.get("url") == clean else "")
    if len(secret) > _MAX_SECRET:
        raise HTTPException(400, "That is too long to be a key.")
    try:
        size = int(context or 0)
    except (TypeError, ValueError):
        raise HTTPException(400, "Its context size is a number of tokens.") from None
    if size and not 1024 <= size <= 100_000_000:
        raise HTTPException(400, "Its context size is a number of tokens.")
    own = {"url": clean, "key": secret, "models": names, "context": size, "saved_at": datetime.now(UTC).isoformat()}
    try:
        sealed = fd_crypto.seal(own)
    except fd_crypto.CryptoUnavailable as exc:
        raise HTTPException(503, f"It can't be stored safely on this deck: {exc}") from None
    await db.set_settings(uid, {_K_OWN: sealed})
    return own


async def delete_own_model(db, uid: str) -> None:
    await db.delete_setting(uid, _K_OWN)


async def check_own_model(db, uid: str, url: str, key: str) -> dict[str, Any]:
    """Can Flight Deck reach it, and what does it serve? A saved key is only ever sent
    to the server it was saved for. Only ids and a reason come back, never the reply."""
    clean = _clean_url(url)
    secret = (key or "").strip()
    if not secret:
        old = await read_own_model(db, uid)
        secret = old.get("key", "") if old.get("url") == clean else ""
    headers = {"Authorization": f"Bearer {secret}"} if secret else {}
    try:
        async with httpx.AsyncClient(timeout=8, trust_env=False, follow_redirects=False) as client:
            resp = await client.get(f"{clean}/models", headers=headers)
    except httpx.HTTPError as exc:
        return {"ok": False, "url": clean, "models": [], "error": f"Can't reach it from this server: {type(exc).__name__}."}
    if resp.status_code in (401, 403):
        return {"ok": False, "url": clean, "models": [], "error": "It's there, but it refused that key."}
    if resp.status_code >= 300:
        return {"ok": False, "url": clean, "models": [], "error": f"It answered {resp.status_code}. Is /v1 the right path?"}
    try:
        data = resp.json().get("data")
        ids = [str(m["id"]) for m in data if isinstance(m, dict) and m.get("id")]
    except (ValueError, TypeError, AttributeError, KeyError):
        ids = []
    return {"ok": True, "url": clean, "models": [m for m in ids if _MODEL_ID_RE.match(m)][:_OWN_MODELS_MAX], "error": ""}


def opencode_config(own: dict[str, Any]) -> dict[str, Any]:
    options: dict[str, Any] = {"baseURL": own["url"]}
    if own.get("key"):
        options["apiKey"] = "{env:" + OWN_KEY_ENV + "}"
    models: dict[str, Any] = {}
    for name in own.get("models") or []:
        model: dict[str, Any] = {"name": name}
        if own.get("context"):
            model["limit"] = {"context": own["context"], "output": min(own["context"] // 4, 65536)}
        models[name] = model
    return {"provider": {OWN_PROVIDER: {"npm": "@ai-sdk/openai-compatible", "name": "Your own model", "options": options,
                                        "models": models}}}


def _sync_pi_models(home: Path, own: dict[str, Any]) -> None:
    """Pi reads its models from a file, which holds only the variable's name. Whatever
    else is in that file is theirs, and stays."""
    path = home / ".pi" / "agent" / "models.json"
    try:
        current = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except (OSError, ValueError):
        return
    if not isinstance(current, dict):
        return
    providers = current.get("providers") if isinstance(current.get("providers"), dict) else {}
    wanted = dict(providers)
    wanted.pop(OWN_PROVIDER, None)
    if own:
        entries = []
        for name in own.get("models") or []:
            entry: dict[str, Any] = {"id": name}
            if own.get("context"):
                entry.update(contextWindow=own["context"], maxTokens=min(own["context"] // 4, 65536))
            entries.append(entry)
        wanted[OWN_PROVIDER] = {"baseUrl": own["url"], "api": "openai-completions",
                                "apiKey": f"${OWN_KEY_ENV}" if own.get("key") else "none", "models": entries}
    if wanted != providers:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({**current, "providers": wanted}, indent=2), encoding="utf-8")


def effective_model(defaults: dict[str, str], own: dict[str, Any], harness: str) -> str:
    """What a task runs on when it names nothing: their choice, else their own server."""
    if defaults.get(harness):
        return defaults[harness]
    names = own_model_names(own) if harness in OWN_HARNESSES else []
    return names[0] if names else ""


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
    if harness in OWN_HARNESSES and await read_own_model(db, uid):
        return {"ok": True, "how": "your own model"}
    if grant.trusted:
        return {"ok": True, "how": "this computer's login"}
    if harness == "opencode":
        # Nothing of anyone's is spent on these, so there's nothing to log in to.
        return {"ok": True, "how": "OpenCode's free models", "free": True}
    return {"ok": False, "how": ""}


async def read_models(db, uid: str) -> dict[str, str]:
    try:
        models = json.loads(await db.get_setting(uid, _K_MODELS) or "{}")
    except (TypeError, ValueError):
        models = {}
    return {str(k): str(v) for k, v in models.items() if v} if isinstance(models, dict) else {}


async def save_model(db, uid: str, harness: str, model: str) -> None:
    models = await read_models(db, uid)
    if model:
        models[harness] = model
    else:
        models.pop(harness, None)
    await db.set_settings(uid, {_K_MODELS: json.dumps(models, ensure_ascii=True)})


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
    own = await read_own_model(db, uid) if harness in OWN_HARNESSES else {}
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
        if own.get("key"):
            env[OWN_KEY_ENV] = own["key"]
        if harness == "opencode" and own:
            env["OPENCODE_CONFIG_CONTENT"] = json.dumps(opencode_config(own))
        if harness == "pi":
            _sync_pi_models(Path(env["HOME"]), own)
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
