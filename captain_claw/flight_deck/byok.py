"""Your own AI keys (BYOK): a user runs their agents on their own provider
keys, and while they do, the deck's daily plan limits don't apply to them.

* **Keys** — one per provider (:data:`PROVIDERS`), checked with the provider
  before they are saved, in ``user_settings`` as ``byok:keys``: one JSON map
  sealed with the deck's data key (``fd_crypto``). ``byok:meta`` holds what
  Flight Deck knows about them (a hint, when they were checked, the provider's
  model list), the switch (``enabled``) and which model each tier runs on
  (``tiers``). ``byok:`` is server-owned: ``/fd/settings`` never returns it.
  No route ever returns a key — only its hint.
* **Models** — while the switch is on, :func:`overlay` replaces every tier of
  the user's tier set (their own, or the team default) with one on their own
  key, in ``basna_routes.owner_tier_set``. Everything that runs on a user's
  tiers follows: new agents, and Flight Deck's own calls for them (skill
  cards, the file library). A tier on a provider they have a key for keeps
  its model; any other moves to one of their providers (the user can pick
  the model per tier). Local tiers (Ollama, LiteRT) stay as they are: the
  deck pays nothing for them. A tier the user can't cover keeps the switch
  off — the deck never pays while it is on.
* **Existing agents** — a key lives in an agent's ``.env`` from spawn, so
  turning the switch on moves each of the user's process agents onto its
  tier's own-key model, and off puts the model it had back (the deck's key
  then returns through ``server.heal_keyless_agents``'s path). Running ones
  restart. That is the serving module's work (:data:`agent_switcher`, set at
  start-up — ``python -m …server`` runs it as ``__main__``, so importing
  ``server`` here would get a second copy with its own process table).
* **Limits** — ``usage_quota.status`` treats a user with the switch on as
  exempt; tokens their agents spend on their keys are kept apart
  (``daily_usage.byok_tokens``), so turning it off mid-day doesn't spend the
  day's allowance on them.

An admin can turn the feature off for the deck (``fd:byok-allowed``); that
switches everyone back to the deck's models.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from typing import Any, Awaitable, Callable

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from captain_claw.flight_deck import fd_crypto
from captain_claw.flight_deck.admin_routes import require_admin
from captain_claw.flight_deck.auth import _fd_auth_enabled, get_current_user, get_db
from captain_claw.flight_deck.rate_limiter import STANDALONE_MODE
from captain_claw.logging import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/fd", tags=["byok"])

_K_KEYS = "byok:keys"
_K_META = "byok:meta"
ALLOWED_SETTING = "fd:byok-allowed"

# The providers a key can be brought for, in the order a tier that needs a
# new home tries them.
PROVIDERS: dict[str, str] = {
    "anthropic": "Anthropic",
    "openai": "OpenAI",
    "gemini": "Google Gemini",
    "openrouter": "OpenRouter",
    "xai": "xAI",
}

# Models that cost the deck nothing: they run on its own hardware.
LOCAL_PROVIDERS = frozenset({"ollama", "litert"})

# What a tier is for, by name — a tier the registry doesn't know runs like
# "balanced".
_TIER_KIND = {"reason": "reason", "balanced": "balanced", "fast": "fast",
              "longctx": "longctx", "coding": "coding", "vision": "vision"}

# The model a tier prefers on each provider: model families, best first. A
# family matches its own ids and their versions ("claude-sonnet" →
# "claude-sonnet-4-6"), never a sibling ("gpt-4.1" ≠ "gpt-4.1-mini"); the
# newest version wins.
_PREFER: dict[str, dict[str, list[str]]] = {
    "anthropic": {
        "reason": ["claude-opus"], "coding": ["claude-opus", "claude-sonnet"],
        "balanced": ["claude-sonnet"], "longctx": ["claude-sonnet"], "vision": ["claude-sonnet"],
        "fast": ["claude-haiku"],
    },
    # Never the GPT-5 / Codex family: those sign in through ChatGPT, not a key.
    "openai": {
        "reason": ["gpt-4.1"], "coding": ["gpt-4.1"], "balanced": ["gpt-4.1"],
        "longctx": ["gpt-4.1"], "vision": ["gpt-4.1", "gpt-4o"], "fast": ["gpt-4.1-mini", "gpt-4o-mini"],
    },
    "gemini": {
        "reason": ["gemini-2.5-pro"], "coding": ["gemini-2.5-pro"], "longctx": ["gemini-2.5-pro"],
        "balanced": ["gemini-2.5-flash"], "vision": ["gemini-2.5-flash"],
        "fast": ["gemini-2.5-flash-lite", "gemini-2.0-flash"],
    },
    "openrouter": {
        "reason": ["anthropic/claude-opus", "google/gemini-2.5-pro", "openai/gpt-4.1"],
        "coding": ["anthropic/claude-opus", "anthropic/claude-sonnet"],
        "balanced": ["anthropic/claude-sonnet", "google/gemini-2.5-flash", "openai/gpt-4.1"],
        "longctx": ["google/gemini-2.5-pro", "anthropic/claude-sonnet"],
        "vision": ["anthropic/claude-sonnet", "google/gemini-2.5-flash"],
        "fast": ["anthropic/claude-haiku", "google/gemini-2.5-flash-lite", "openai/gpt-4.1-mini"],
    },
    "xai": {
        "reason": ["grok-4"], "coding": ["grok-code-fast", "grok-4"], "balanced": ["grok-4"],
        "longctx": ["grok-4"], "vision": ["grok-4"], "fast": ["grok-3-mini", "grok-4-fast"],
    },
}

# With no model list (the provider couldn't be asked), a tier's model is this.
_FALLBACK: dict[str, dict[str, str]] = {
    "anthropic": {"reason": "claude-opus-4-8", "coding": "claude-opus-4-8", "fast": "claude-haiku-4-5",
                  "_": "claude-sonnet-4-6"},
    "openai": {"fast": "gpt-4.1-mini", "_": "gpt-4.1"},
    "gemini": {"reason": "gemini-2.5-pro", "coding": "gemini-2.5-pro", "longctx": "gemini-2.5-pro",
               "fast": "gemini-2.5-flash-lite", "_": "gemini-2.5-flash"},
    "openrouter": {"fast": "anthropic/claude-haiku-4.5", "_": "anthropic/claude-sonnet-4.5"},
    "xai": {"fast": "grok-3-mini", "_": "grok-4"},
}

_MAX_MODELS = 800
_TIMEOUT = httpx.Timeout(15.0, connect=10.0)

# Set at start-up by the serving module (see the module docstring):
#   agent_switcher(owner_id, tiers or None, keys) -> {"switched": n, "restarted": n}
#   owner_agents(owner_id) -> [{"slug", "running", "byok"}]
agent_switcher: Callable[[str, dict | None, dict], Awaitable[dict]] | None = None
owner_agents: Callable[[str], list[dict]] | None = None


def _now() -> str:
    return datetime.now(UTC).isoformat()


def key_hint(key: str) -> str:
    key = key or ""
    return f"…{key[-4:]}" if len(key) > 8 else "set"


# ── storage ─────────────────────────────────────────────────────────


def offered() -> bool:
    """Can BYOK mean anything here? Not without sign-in or on the standalone
    app: nobody has daily limits there, and the deck's keys are its owner's."""
    return _fd_auth_enabled() and not STANDALONE_MODE


async def allowed(db) -> bool:
    """Has the deck's admin left BYOK on (it is, unless turned off)?"""
    if not offered():
        return False
    try:
        return (await db.get_system_setting(ALLOWED_SETTING) or "1").strip() != "0"
    except Exception:
        return True


async def read_meta(db, uid: str) -> dict:
    try:
        meta = json.loads(await db.get_setting(uid, _K_META) or "{}")
    except (TypeError, ValueError):
        meta = {}
    if not isinstance(meta, dict):
        meta = {}
    meta.setdefault("enabled", False)
    if not isinstance(meta.get("providers"), dict):
        meta["providers"] = {}
    if not isinstance(meta.get("tiers"), dict):
        meta["tiers"] = {}
    return meta


async def _write_meta(db, uid: str, meta: dict) -> None:
    meta["updated_at"] = _now()
    await db.set_settings(uid, {_K_META: json.dumps(meta, ensure_ascii=True)})


async def read_keys(db, uid: str) -> dict[str, str]:
    """The user's keys by provider ({} when none, or the data key can't open them)."""
    token = await db.get_setting(uid, _K_KEYS) or ""
    if not token:
        return {}
    try:
        keys = fd_crypto.unseal(token)
    except fd_crypto.CryptoUnavailable:
        log.warning("Own AI keys can't be read: encryption unavailable", user_id=uid)
        return {}
    if not isinstance(keys, dict):
        return {}
    return {str(p): str(k) for p, k in keys.items() if p in PROVIDERS and k}


async def _write_keys(db, uid: str, keys: dict[str, str]) -> None:
    if not keys:
        await db.delete_setting(uid, _K_KEYS)
        return
    try:
        token = fd_crypto.seal(keys)
    except fd_crypto.CryptoUnavailable as exc:
        raise HTTPException(503, f"Own keys can't be stored safely on this deck: {exc}") from None
    await db.set_settings(uid, {_K_KEYS: token})


async def active(db, uid: str) -> bool:
    """Is *uid* running on their own keys right now? Never raises."""
    if not uid:
        return False
    try:
        if not await allowed(db):
            return False
        if not (await read_meta(db, uid)).get("enabled"):
            return False
        return bool(await read_keys(db, uid))
    except Exception:
        return False


# ── models ──────────────────────────────────────────────────────────


def chatgpt_only(provider: str, model: str) -> bool:
    """A model that signs in through ChatGPT, not an API key (OpenAI GPT-5 / Codex)."""
    if provider != "openai":
        return False
    try:
        from captain_claw.llm import _is_codex_family_model

        return bool(_is_codex_family_model(model or ""))
    except Exception:
        return False


_DATE = re.compile(r"\d{4}-?\d{2}-?\d{2}|\b\d{2}-\d{2}\b|\b\d{4}\b")


def _rank(rest: str) -> tuple:
    """How good a family member is by what follows the family name: a
    release before a preview, the newest version, the alias before a dated
    snapshot of it ("gpt-4.1" before "gpt-4.1-2025-04-14")."""
    undated = _DATE.sub("", rest)
    version = tuple(int(n) for n in re.findall(r"\d+", undated))
    return ("preview" not in rest, version, undated == rest, -len(rest))


def _family_match(model: str, family: str) -> bool:
    if model == family:
        return True
    if not model.startswith(family + "-"):
        return False
    rest = model[len(family) + 1:]
    return rest[:1].isdigit() or rest.startswith("preview")


def default_model(provider: str, tier: str, models: list[str] | None) -> str:
    """The model *tier* runs on with a *provider* key: the newest of its
    preferred families the key can reach, else a known default."""
    kind = _TIER_KIND.get(tier, "balanced")
    usable = [m for m in (models or []) if not chatgpt_only(provider, m)]
    for family in (_PREFER.get(provider) or {}).get(kind, []):
        hits = [m for m in usable if _family_match(m, family)]
        if hits:
            return max(hits, key=lambda m: _rank(m[len(family):]))
    fallback = _FALLBACK.get(provider) or {}
    return fallback.get(kind) or fallback.get("_") or (usable[0] if usable else "")


def _registry_tiers() -> dict:
    try:
        from captain_claw.flight_deck.basna_routes import _load_registry

        tiers = _load_registry().get("tiers") or {}
    except Exception:
        tiers = {}
    return tiers if isinstance(tiers, dict) else {}


def plan(base_tiers: dict, keys: dict[str, str], meta: dict) -> tuple[dict[str, dict], list[str]]:
    """Which own-key model each deck-paid tier runs on: ``({tier: {provider,
    model}}, tiers that can't be covered)``. Local tiers are left out — they
    stay as they are."""
    registry = _registry_tiers()
    chosen = meta.get("tiers") or {}
    known = meta.get("providers") or {}
    order = [p for p in PROVIDERS if p in keys]
    out: dict[str, dict] = {}
    missing: list[str] = []
    for name in [*registry, *(t for t in (base_tiers or {}) if t not in registry)]:
        base = (base_tiers or {}).get(name)
        if not (isinstance(base, dict) and base.get("model")):
            base = registry.get(name) or {}
        if str(base.get("provider") or "") in LOCAL_PROVIDERS:
            continue
        pick = chosen.get(name) if isinstance(chosen.get(name), dict) else {}
        provider, model = str(pick.get("provider") or ""), str(pick.get("model") or "").strip()
        if provider in keys and model and not chatgpt_only(provider, model):
            out[name] = {"provider": provider, "model": model}
            continue
        base_provider, base_model = str(base.get("provider") or ""), str(base.get("model") or "")
        if base_provider in keys and base_model and not str(base.get("base_url") or "").strip() \
                and not chatgpt_only(base_provider, base_model):
            out[name] = {"provider": base_provider, "model": base_model}
            continue
        if not order:
            missing.append(name)
            continue
        provider = order[0]
        model = default_model(provider, name, (known.get(provider) or {}).get("models"))
        if model:
            out[name] = {"provider": provider, "model": model}
        else:
            missing.append(name)
    return out, missing


def overlay_tiers(base_tiers: dict, keys: dict[str, str], meta: dict) -> dict:
    """*base_tiers* with every deck-paid tier on the user's own key (see
    :func:`plan`). Each such tier carries ``byok: True``; the keys are raw —
    never send these tiers on."""
    registry = _registry_tiers()
    planned, _missing = plan(base_tiers, keys, meta)
    out = {name: t for name, t in (base_tiers or {}).items() if name not in planned}
    for name, pick in planned.items():
        base = (base_tiers or {}).get(name) if isinstance((base_tiers or {}).get(name), dict) else {}
        same = (base.get("provider"), base.get("model")) == (pick["provider"], pick["model"])
        sizes = base if same else (registry.get(name) or {})
        tier = {
            "provider": pick["provider"],
            "model": pick["model"],
            "api_key": keys[pick["provider"]],
            "base_url": "",
            "input_ctx": sizes.get("input_ctx") or 0,
            "output_ctx": sizes.get("output_ctx") or 0,
            "byok": True,
        }
        if same and base.get("vision") is not None:
            tier["vision"] = base["vision"]
        out[name] = tier
    return out


async def overlay(db, uid: str, resolved: dict) -> dict:
    """``owner_tier_set``'s answer with the user's own keys over it when
    their switch is on (``source: "byok"``); as it was otherwise."""
    if not await active(db, uid):
        return resolved
    keys = await read_keys(db, uid)
    meta = await read_meta(db, uid)
    return {**resolved, "source": "byok", "base_source": resolved.get("source"),
            "name": "Your own keys",
            "tiers": overlay_tiers(resolved.get("tiers") or {}, keys, meta)}


async def _base_tiers(db, uid: str) -> dict:
    from captain_claw.flight_deck.basna_routes import _owner_base_tier_set

    return (await _owner_base_tier_set(db, uid)).get("tiers") or {}


# ── provider checks ─────────────────────────────────────────────────


class KeyRefused(Exception):
    pass


def _model_ids(provider: str, data: Any) -> list[str]:
    if provider == "gemini":
        rows = data.get("models") if isinstance(data, dict) else None
        ids = [str(r.get("name") or "").removeprefix("models/") for r in rows or []
               if isinstance(r, dict) and "generateContent" in (r.get("supportedGenerationMethods") or [])]
    else:
        rows = data.get("data") if isinstance(data, dict) else None
        ids = [str(r.get("id") or "") for r in rows or [] if isinstance(r, dict)]
    skip = ("embed", "tts", "whisper", "dall-e", "image", "imagen", "audio", "realtime",
            "transcribe", "moderation", "search", "live")
    keep = [m for m in ids if m and not any(s in m.lower() for s in skip)]
    if provider == "openai":
        keep = [m for m in keep if m.startswith(("gpt-4", "o1", "o3", "o4")) and not chatgpt_only("openai", m)]
    elif provider == "gemini":
        keep = [m for m in keep if m.startswith("gemini")]
    return sorted(set(keep))[:_MAX_MODELS]


async def check_key(provider: str, key: str) -> list[str]:
    """Ask *provider* about *key*: the models it can reach. Raises
    :class:`KeyRefused` when the provider says no, ``httpx.HTTPError`` when it
    couldn't be asked."""
    headers: dict[str, str] = {}
    params: dict[str, str] = {}
    if provider == "anthropic":
        url = "https://api.anthropic.com/v1/models"
        headers = {"x-api-key": key, "anthropic-version": "2023-06-01"}
        params = {"limit": "1000"}
    elif provider == "openai":
        url = "https://api.openai.com/v1/models"
        headers = {"Authorization": f"Bearer {key}"}
    elif provider == "gemini":
        url = "https://generativelanguage.googleapis.com/v1beta/models"
        headers = {"x-goog-api-key": key}
        params = {"pageSize": "1000"}
    elif provider == "xai":
        url = "https://api.x.ai/v1/models"
        headers = {"Authorization": f"Bearer {key}"}
    elif provider == "openrouter":
        url = "https://openrouter.ai/api/v1/key"
        headers = {"Authorization": f"Bearer {key}"}
    else:
        raise KeyRefused("unknown provider")
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        resp = await client.get(url, headers=headers, params=params)
        if resp.status_code in (400, 401, 403):
            raise KeyRefused(f"HTTP {resp.status_code}")
        resp.raise_for_status()
        if provider != "openrouter":
            return _model_ids(provider, resp.json())
        # OpenRouter's model list is public; the key check above is the test.
        listed = await client.get("https://openrouter.ai/api/v1/models")
        listed.raise_for_status()
        return _model_ids(provider, listed.json())


# ── status ──────────────────────────────────────────────────────────


async def status_for(db, uid: str) -> dict[str, Any]:
    """What the user's "Your own AI keys" dialog shows. Never a key."""
    meta = await read_meta(db, uid)
    keys = await read_keys(db, uid)
    is_allowed = await allowed(db)
    base = await _base_tiers(db, uid)
    planned, missing = plan(base, keys, meta)
    registry = _registry_tiers()
    tiers = []
    for name in [*registry, *(t for t in base if t not in registry)]:
        team = base.get(name) if isinstance(base.get(name), dict) and base[name].get("model") else registry.get(name) or {}
        local = str(team.get("provider") or "") in LOCAL_PROVIDERS
        tiers.append({
            "id": name,
            "label": str((registry.get(name) or {}).get("label") or name),
            "use": str((registry.get(name) or {}).get("use") or ""),
            "team": {"provider": str(team.get("provider") or ""), "model": str(team.get("model") or "")},
            "mine": planned.get(name),
            "chosen": bool(isinstance(meta["tiers"].get(name), dict) and name in planned
                           and meta["tiers"][name] == planned[name]),
            "local": local,
        })
    providers = []
    for pid, label in PROVIDERS.items():
        info = meta["providers"].get(pid) if isinstance(meta["providers"].get(pid), dict) else {}
        providers.append({
            "id": pid, "label": label, "connected": pid in keys,
            "hint": info.get("hint") if pid in keys else None,
            "verified": info.get("verified") if pid in keys else None,
            "checked_at": info.get("checked_at") if pid in keys else None,
            "models": (info.get("models") or []) if pid in keys else [],
        })
    agents = owner_agents(uid) if owner_agents else []
    return {
        "offered": offered(),
        "allowed": is_allowed,
        "encryption": fd_crypto.available(),
        "enabled": bool(meta.get("enabled")) and is_allowed and bool(keys),
        "providers": providers,
        "tiers": tiers,
        "missing": missing,
        "agents": {"total": len(agents), "running": sum(1 for a in agents if a.get("running")),
                   "on_own_keys": sum(1 for a in agents if a.get("byok"))},
        "switched_at": meta.get("switched_at"),
    }


async def _apply(db, uid: str, on: bool) -> dict:
    """Move the user's agents onto their keys (or back); {} without the hook."""
    if agent_switcher is None:
        return {}
    keys = await read_keys(db, uid)
    tiers = overlay_tiers(await _base_tiers(db, uid), keys, await read_meta(db, uid)) if on else None
    try:
        return await agent_switcher(uid, tiers, keys)
    except Exception as exc:
        log.warning("Moving agents to/from own keys failed", user_id=uid, error=str(exc))
        return {"error": str(exc)}


async def _log(db, uid: str, event: str, detail: dict) -> None:
    try:
        await db.log_usage(uid, event, json.dumps(detail))
    except Exception:
        pass


# ── user routes ─────────────────────────────────────────────────────


def _require_offered() -> None:
    if not offered():
        raise HTTPException(404, "Own keys aren't used on this deck: it has no daily limits")


def _provider(provider: str) -> str:
    p = (provider or "").strip().lower()
    if p not in PROVIDERS:
        raise HTTPException(404, f"Own keys work for {', '.join(PROVIDERS.values())}")
    return p


async def _key_from_body(request: Request, label: str) -> str:
    """``api_key`` from a ``{"api_key": "..."}`` body, read by hand: a
    malformed body would otherwise reach FD's validation handler, which echoes
    the raw input — the key."""
    try:
        data = await request.json()
    except Exception:
        data = None
    raw = data.get("api_key") if isinstance(data, dict) else None
    key = raw.strip() if isinstance(raw, str) else ""
    if not key:
        raise HTTPException(422, f"Paste your {label} API key")
    if not (key.isascii() and key.isprintable()) or any(c.isspace() for c in key) or len(key) > 512:
        raise HTTPException(422, (
            f"That doesn't look like a {label} API key — it has characters a key can't have "
            "(hidden spaces or curly quotes from copying). Copy it again. Nothing was saved."))
    return key


@router.get("/byok")
async def my_byok(user: dict = Depends(get_current_user)) -> dict:
    """The caller's own keys (hints only), the switch, and each tier's model."""
    _require_offered()
    return await status_for(get_db(), str(user["id"]))


@router.put("/byok/key/{provider}")
async def put_key(provider: str, request: Request, user: dict = Depends(get_current_user)) -> dict:
    """Save a key after checking it with the provider. One the provider
    refuses is not saved (400 — never 401, which the UI reads as a signed-out
    session); one it couldn't be asked about is saved unchecked."""
    _require_offered()
    pid = _provider(provider)
    label = PROVIDERS[pid]
    key = await _key_from_body(request, label)
    db, uid = get_db(), str(user["id"])
    warning = ""
    info: dict[str, Any] = {"hint": key_hint(key), "checked_at": _now()}
    try:
        info.update(verified=True, models=await check_key(pid, key))
    except KeyRefused as exc:
        raise HTTPException(400, f"{label} refused this key ({exc}) — check that it is correct "
                                 "and has credit. Nothing was saved.") from None
    except httpx.HTTPError as exc:
        info.update(verified=False, models=[])
        warning = f"Saved, but {label} couldn't be asked about it ({type(exc).__name__}). " \
                  "It's checked again when you save it next."
    keys = await read_keys(db, uid)
    keys[pid] = key
    await _write_keys(db, uid, keys)
    meta = await read_meta(db, uid)
    meta["providers"][pid] = info
    await _write_meta(db, uid, meta)
    await _log(db, uid, "byok_key_saved", {"provider": pid, "verified": info["verified"]})
    moved = await _apply(db, uid, True) if await active(db, uid) else {}
    return {**await status_for(db, uid), "warning": warning, "moved": moved}


@router.delete("/byok/key/{provider}")
async def delete_key(provider: str, user: dict = Depends(get_current_user)) -> dict:
    """Forget a key. While the switch is on, its tiers move to another of the
    user's keys; the last one can't go until the switch is off."""
    _require_offered()
    pid = _provider(provider)
    db, uid = get_db(), str(user["id"])
    keys = await read_keys(db, uid)
    on = await active(db, uid)
    if on and set(keys) <= {pid}:
        raise HTTPException(409, "This is your last key — turn your own keys off first.")
    keys.pop(pid, None)
    await _write_keys(db, uid, keys)
    meta = await read_meta(db, uid)
    meta["providers"].pop(pid, None)
    meta["tiers"] = {t: c for t, c in meta["tiers"].items()
                     if not (isinstance(c, dict) and c.get("provider") == pid)}
    await _write_meta(db, uid, meta)
    await _log(db, uid, "byok_key_removed", {"provider": pid})
    moved = await _apply(db, uid, True) if on else {}
    return {**await status_for(db, uid), "moved": moved}


class TierPick(BaseModel):
    provider: str = Field("", max_length=40)
    model: str = Field("", max_length=200)


class TierPicks(BaseModel):
    # tier → its model; an empty pick goes back to the default.
    tiers: dict[str, TierPick] = Field(default_factory=dict, max_length=40)


@router.put("/byok/models")
async def put_models(body: TierPicks, user: dict = Depends(get_current_user)) -> dict:
    """Choose the model a tier runs on with the user's own keys."""
    _require_offered()
    db, uid = get_db(), str(user["id"])
    keys = await read_keys(db, uid)
    meta = await read_meta(db, uid)
    for name, pick in body.tiers.items():
        name = name.strip()[:40]
        provider, model = pick.provider.strip().lower(), " ".join(pick.model.split())
        if not provider and not model:
            meta["tiers"].pop(name, None)
            continue
        if provider not in keys:
            raise HTTPException(400, f"Add a {PROVIDERS.get(provider, provider or 'provider')} key first")
        if not model:
            raise HTTPException(400, f"Pick a model for {name}")
        if chatgpt_only(provider, model):
            raise HTTPException(400, f"{model} signs in through ChatGPT, not an API key — "
                                     "pick another OpenAI model (e.g. gpt-4.1)")
        meta["tiers"][name] = {"provider": provider, "model": model}
    await _write_meta(db, uid, meta)
    moved = await _apply(db, uid, True) if await active(db, uid) else {}
    return {**await status_for(db, uid), "moved": moved}


class Switch(BaseModel):
    enabled: bool


@router.put("/byok/enabled")
async def put_enabled(body: Switch, user: dict = Depends(get_current_user)) -> dict:
    """Turn the user's own keys on or off — and move their agents with it."""
    _require_offered()
    db, uid = get_db(), str(user["id"])
    meta = await read_meta(db, uid)
    if body.enabled:
        if not await allowed(db):
            raise HTTPException(403, "Your admin has turned own keys off on this deck")
        keys = await read_keys(db, uid)
        if not keys:
            raise HTTPException(409, "Add at least one API key first")
        _planned, missing = plan(await _base_tiers(db, uid), keys, meta)
        if missing:
            raise HTTPException(409, f"No model for: {', '.join(missing)}. Pick one for each.")
    if bool(meta.get("enabled")) == body.enabled:
        return {**await status_for(db, uid), "moved": {}}
    meta["enabled"] = body.enabled
    meta["switched_at"] = _now()
    await _write_meta(db, uid, meta)
    await _log(db, uid, "byok_on" if body.enabled else "byok_off", {})
    moved = await _apply(db, uid, body.enabled)
    return {**await status_for(db, uid), "moved": moved}


# ── admin ───────────────────────────────────────────────────────────


@router.get("/admin/byok")
async def admin_byok(_admin: dict = Depends(require_admin)) -> dict:
    """Is BYOK allowed on this deck, and who is on their own keys."""
    db = get_db()
    on = [uid for uid in await db.users_with_setting(_K_META) if await active(db, uid)]
    return {"offered": offered(), "allowed": await allowed(db), "users_on": on}


class Allow(BaseModel):
    allowed: bool


@router.put("/admin/byok")
async def admin_put_byok(body: Allow, admin: dict = Depends(require_admin)) -> dict:
    """Allow own keys on the deck, or turn them off — which switches everyone
    who uses them back to the deck's models."""
    db = get_db()
    moved = 0
    if not body.allowed:
        for uid in await db.users_with_setting(_K_META):
            meta = await read_meta(db, uid)
            if not meta.get("enabled"):
                continue
            meta["enabled"] = False
            meta["switched_at"] = _now()
            await _write_meta(db, uid, meta)
            await _apply(db, uid, False)
            await _log(db, uid, "byok_off", {"by_admin": str(admin.get("id") or "")})
            moved += 1
    await db.set_system_setting(ALLOWED_SETTING, "1" if body.allowed else "0")
    return {"ok": True, "allowed": body.allowed, "switched_back": moved}
