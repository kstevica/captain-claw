"""Which other models an agent can switch to, on its own provider.

An agent runs on the model its config names (``model.*`` in config.yaml). It
can switch — for the rest of a conversation, or for one turn — to another
model of the SAME provider at the SAME endpoint: the key and the address stay
as they are, only the model name changes. Tier sets are a different thing and
are not touched. The models it can pick:

* the big providers' current models — the tier presets catalog
  (``instructions/tier_presets.json``, the list Quick start builds tier sets
  from and that is re-checked monthly). Its tiers say what each model is for:
  the reasoning tier's model is the smartest, the fast tier's the cheapest;
* the models a Flight Deck admin listed for the agent's endpoint (Console →
  Models → "Models agents can switch to") — a gateway, a local server,
  OpenRouter …, whose models no catalog knows. They are fetched from Flight
  Deck (``/fd/model-switch/agent/models``) and kept in
  ``~/.captain-claw/model_choices.json``;
* the agent's own ``model.allowed`` entries on that endpoint.

Whether the agent may switch on its own — to a cheaper model for simple work,
a smarter one for a hard task — is ``~/.captain-claw/model_switch.txt``
("on"/"off", off by default), which Flight Deck's agent settings write. When
the user asks, it always may.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from pathlib import Path
from typing import Any, Iterable

from captain_claw.flight_deck.endpoints import endpoint_of, same_endpoint
from captain_claw.logging import get_logger

log = get_logger(__name__)

SWITCH_FILENAME = "model_switch.txt"
CHOICES_FILENAME = "model_choices.json"
SWITCH_PATH = Path("~/.captain-claw").expanduser() / SWITCH_FILENAME
CHOICES_PATH = Path("~/.captain-claw").expanduser() / CHOICES_FILENAME
PRESETS_FILE = Path(__file__).resolve().parent.parent / "instructions" / "tier_presets.json"

# How often the admin's list is asked for again (at most; turns schedule it).
REFRESH_EVERY = 600.0
_FETCH_TIMEOUT = 6.0

# What a catalog model is for, from the first tier it sits on in this order.
ROLE_ORDER = ("reason", "balanced", "fast")
ROLE_LABELS = {
    "reason": "Smartest",
    "balanced": "Everyday",
    "fast": "Fastest and cheapest",
}
# Tiers whose context / output say nothing about the model's everyday limits:
# longctx reads past the provider's price step on purpose, micro is tiny.
_LIMIT_SKIP_TIERS = frozenset({"longctx", "micro"})

_ON = {"on", "true", "1", "yes"}
_OFF = {"off", "false", "0", "no"}

DEFAULT_WORDS = frozenset({"default", "config", "configured", "usual", "normal", "original", "back"})
_ROLE_WORDS = {
    "reason": frozenset({"smartest", "smarter", "strongest", "stronger", "best", "most capable",
                         "biggest", "bigger", "reasoning", "pametniji", "najpametniji", "jači", "najjači"}),
    "fast": frozenset({"cheapest", "cheaper", "fastest", "faster", "lightest", "smallest", "smaller",
                       "jeftiniji", "najjeftiniji", "brži", "najbrži", "manji"}),
    "balanced": frozenset({"everyday", "balanced", "middle", "medium"}),
}
# Words of a model id that don't tell one model of a provider from another.
_GENERIC_TOKENS = frozenset({
    "claude", "gemini", "gpt", "grok", "deepseek", "openai", "anthropic", "google", "meta",
    "llama", "qwen", "mistral", "non", "reasoning", "preview", "latest", "instruct", "chat",
    "model", "beta", "exp", "experimental", "thinking",
})
# A message about which model the agent runs on (EN / HR: model, modela, modelu …).
_MODEL_TALK_RE = re.compile(r"\bmodel\w*|\bllm\w*", re.I)
# …or, looser, asking for a smarter or cheaper one without the word.
_SWITCH_ASK_RE = re.compile(r"\b(?:smarter|cheaper|stronger)\b|pametnij\w*|jeftinij\w*|\bjač\w*|\bbrž\w*", re.I)

_presets_cache: dict[str, Any] = {"mtime": None, "value": []}
_custom_cache: dict[str, Any] = {"mtime": None, "path": None, "value": None}
_refresh_state: dict[str, Any] = {"at": 0.0, "key": "", "task": None}


# ── the self-switch flag ──────────────────────────────────────────────

def parse_switch(text: object) -> bool | None:
    """"on"/"off" (and the usual spellings) as True/False; anything else None."""
    value = str(text or "").strip().lower()
    if value in _ON:
        return True
    if value in _OFF:
        return False
    return None


def render_switch(enabled: bool) -> str:
    """What the switch file holds for *enabled*."""
    return "on" if enabled else "off"


def self_switch_enabled() -> bool:
    """May the agent switch models on its own (not only when asked)? Off unless
    its owner turned it on."""
    try:
        return bool(parse_switch(SWITCH_PATH.read_text(encoding="utf-8")))
    except OSError:
        return False


# ── names ─────────────────────────────────────────────────────────────

def bare_model(model: str) -> str:
    """The model without a routing prefix ("anthropic/claude-x" → "claude-x")."""
    return str(model or "").strip().split("/")[-1]


def pretty_name(model: str) -> str:
    """A model id as a person reads it: "claude-opus-5-5" → "Claude Opus 5.5",
    "gpt-6.1-sol" → "GPT 6.1 Sol"."""
    raw = bare_model(model)
    if not raw:
        return ""
    parts = [p for p in re.split(r"[-_\s]+", raw) if p]
    out: list[str] = []
    for part in parts:
        short_number = part.isdigit() and len(part) <= 2
        if short_number and out and re.fullmatch(r"\d+(?:\.\d+)*", out[-1]) and "." not in out[-1]:
            out[-1] = f"{out[-1]}.{part}"
            continue
        if part.lower() in {"gpt", "glm", "llm", "ai", "oss"}:
            out.append(part.upper())
        elif part[:1].isalpha():
            out.append(part[:1].upper() + part[1:])
        else:
            out.append(part)
    return " ".join(out)


def _same_model(a: str, b: str) -> bool:
    x, y = str(a or "").strip().lower(), str(b or "").strip().lower()
    if not x or not y:
        return False
    return x == y or x.endswith("/" + y) or y.endswith("/" + x) or bare_model(x) == bare_model(y)


def is_model(running: str, model: str) -> bool:
    """Is *running* (what a provider reports, maybe with a routing prefix) *model*?"""
    return _same_model(running, model)


# ── the catalog (big providers) ───────────────────────────────────────

def _load_presets() -> list[dict[str, Any]]:
    try:
        mtime = PRESETS_FILE.stat().st_mtime
    except OSError:
        return []
    if _presets_cache["mtime"] == mtime:
        return _presets_cache["value"]
    try:
        data = json.loads(PRESETS_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        log.warning("Tier presets unreadable for model switching", error=str(exc))
        data = {}
    presets = [p for p in (data.get("presets") or []) if isinstance(p, dict) and p.get("provider")]
    _presets_cache.update(mtime=mtime, value=presets)
    return presets


def _codex_family(model: str) -> bool:
    """A model that signs in through ChatGPT, not with an API key."""
    try:
        from captain_claw.llm import _is_codex_family_model

        return bool(_is_codex_family_model(model or ""))
    except Exception:
        return False


def catalog_models(provider: str, base_url: str) -> list[dict[str, Any]]:
    """The catalog's models for this endpoint, smartest first:
    ``{model, role, max_context, max_output}``. Empty for an endpoint the
    catalog doesn't cover (a gateway, a local server)."""
    preset = next((p for p in _load_presets()
                   if same_endpoint(p.get("provider"), p.get("base_url"), provider, base_url)), None)
    if preset is None:
        return []
    tiers = preset.get("tiers") if isinstance(preset.get("tiers"), dict) else {}
    found: dict[str, dict[str, Any]] = {}
    order = [*ROLE_ORDER, *(t for t in tiers if t not in ROLE_ORDER)]
    for tier in order:
        spec = tiers.get(tier)
        if not isinstance(spec, dict):
            continue
        model = str(spec.get("model") or "").strip()
        if not model:
            continue
        entry = found.setdefault(model.lower(), {
            "model": model, "role": tier if tier in ROLE_ORDER else "",
            "max_context": 0, "max_output": 0,
        })
        if tier in _LIMIT_SKIP_TIERS:
            continue
        entry["max_context"] = max(entry["max_context"], int(spec.get("input_ctx") or 0))
        entry["max_output"] = max(entry["max_output"], int(spec.get("output_ctx") or 0))
    return list(found.values())


# ── the admin's list (custom endpoints) ───────────────────────────────

def _endpoint_key(provider: str, base_url: str) -> str:
    name, url = endpoint_of(provider, base_url)
    return f"{name}|{url}"


def _read_custom() -> dict[str, Any] | None:
    path = CHOICES_PATH
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return None
    if _custom_cache["mtime"] == mtime and _custom_cache["path"] == str(path):
        return _custom_cache["value"]
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        data = None
    value = data if isinstance(data, dict) else None
    _custom_cache.update(mtime=mtime, path=str(path), value=value)
    return value


def _clean_models(raw: Any) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in raw if isinstance(raw, list) else []:
        if isinstance(item, str):
            item = {"model": item}
        if not isinstance(item, dict):
            continue
        model = str(item.get("model") or "").strip()
        if not model or model.lower() in seen:
            continue
        seen.add(model.lower())
        out.append({"model": model, "note": str(item.get("note") or "").strip()[:200]})
    return out


def custom_models(provider: str, base_url: str) -> list[dict[str, str]]:
    """The models the admin listed for this endpoint (as last fetched)."""
    data = _read_custom()
    if not data or not same_endpoint(data.get("provider"), data.get("base_url"), provider, base_url):
        return []
    return _clean_models(data.get("models"))


def save_custom(provider: str, base_url: str, models: Any) -> None:
    """Keep the admin's list for this endpoint (survives a restart while
    Flight Deck can't be reached)."""
    CHOICES_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {"provider": provider, "base_url": base_url, "models": _clean_models(models),
               "fetched_at": time.time()}
    tmp = CHOICES_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(CHOICES_PATH)


async def refresh_custom(provider: str, base_url: str) -> bool:
    """Ask Flight Deck for the admin's list for this endpoint and keep it.
    False when there's no Flight Deck to ask or it didn't answer (the kept
    list stays as it was)."""
    from captain_claw import fd_client

    base = fd_client.flight_deck_base()
    if not base:
        return False
    _refresh_state.update(at=time.monotonic(), key=_endpoint_key(provider, base_url))
    try:
        import httpx

        async with httpx.AsyncClient(timeout=_FETCH_TIMEOUT) as client:
            resp = await client.get(
                f"{base}/fd/model-switch/agent/models",
                params={"provider": provider, "base_url": base_url},
                headers=fd_client.flight_deck_headers(),
            )
    except Exception as exc:
        log.debug("Model choices: Flight Deck unreachable", error=str(exc))
        return False
    if resp.status_code != 200:
        # 404: a deck from before model switching — nothing to keep.
        log.debug("Model choices: Flight Deck refused", status=resp.status_code)
        return False
    try:
        data = resp.json()
    except ValueError:
        return False
    models = data.get("models") if isinstance(data, dict) else None
    try:
        save_custom(provider, base_url, models if isinstance(models, list) else [])
    except OSError as exc:
        log.warning("Model choices: couldn't keep the list", error=str(exc))
        return False
    return True


def maybe_refresh(provider: str, base_url: str) -> None:
    """Fetch the admin's list in the background when it is old (or for another
    endpoint) — called at the start of a turn; never blocks it."""
    from captain_claw import fd_client

    if not fd_client.flight_deck_base():
        return
    key = _endpoint_key(provider, base_url)
    task = _refresh_state.get("task")
    if task is not None and not task.done():
        return
    fresh = time.monotonic() - float(_refresh_state.get("at") or 0.0) < REFRESH_EVERY
    if fresh and _refresh_state.get("key") == key:
        return
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    _refresh_state.update(at=time.monotonic(), key=key)
    _refresh_state["task"] = loop.create_task(refresh_custom(provider, base_url))


# ── every choice ──────────────────────────────────────────────────────

def _price(model: str) -> dict[str, float] | None:
    try:
        from captain_claw.flight_deck.pricing import rate_for

        rate = rate_for(model)
    except Exception:
        return None
    if not rate or not (rate.get("input") or rate.get("output")):
        return None
    return {"input": float(rate.get("input") or 0), "output": float(rate.get("output") or 0)}


def choices(provider: str, model: str, base_url: str,
            allowed: Iterable[tuple[str, str, str, str]] = ()) -> list[dict[str, Any]]:
    """Every model the agent can run on at its endpoint, the catalog's first
    (smartest → cheapest), then the admin's, then its own ``model.allowed``
    ones — each ``{model, name, role, label, note, source, default, price,
    max_context, max_output}``. Its configured model is always there (marked
    ``default``)."""
    out: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(entry: dict[str, Any]) -> None:
        key = bare_model(entry["model"]).lower()
        if not key or key in seen:
            return
        seen.add(key)
        name = entry["model"]
        out.append({
            "model": name,
            "name": pretty_name(name),
            "role": entry.get("role", ""),
            "label": ROLE_LABELS.get(entry.get("role", ""), ""),
            "note": entry.get("note", ""),
            "source": entry.get("source", ""),
            "default": _same_model(name, model),
            "price": _price(name),
            "max_context": int(entry.get("max_context") or 0),
            "max_output": int(entry.get("max_output") or 0),
        })

    codex = _codex_family(model)
    for entry in catalog_models(provider, base_url):
        # A ChatGPT sign-in model and the API-key models are different places.
        if _codex_family(entry["model"]) == codex:
            add({**entry, "source": "catalog"})
    for entry in custom_models(provider, base_url):
        add({**entry, "source": "admin"})
    for a_provider, a_model, a_base, a_note in allowed:
        if a_model and same_endpoint(a_provider, a_base, provider, base_url):
            add({"model": str(a_model).strip(), "note": str(a_note or "").strip(), "source": "config"})
    if model and not any(c["default"] for c in out):
        out.insert(0, {
            "model": model, "name": pretty_name(model), "role": "", "label": "", "note": "",
            "source": "config", "default": True, "price": _price(model),
            "max_context": 0, "max_output": 0,
        })
    return out


def resolve(options: list[dict[str, Any]], selector: str) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """The choice *selector* names — an id, a name, part of one ("opus"),
    "default", or what it is for ("smartest", "cheapest") — else None with the
    choices it could mean (none when it names nothing)."""
    text = " ".join(str(selector or "").strip().lower().split())
    if not text or not options:
        return None, []
    for opt in options:
        if text in {opt["model"].lower(), bare_model(opt["model"]).lower(), opt["name"].lower()}:
            return opt, []
    if text in DEFAULT_WORDS or text in {"default model", "the default"}:
        found = next((o for o in options if o.get("default")), None)
        return found, []
    for role, words in _ROLE_WORDS.items():
        if text in words or text.removesuffix(" model") in words:
            found = next((o for o in options if o.get("role") == role), None)
            if found is not None:
                return found, []
            if role != "balanced":
                priced = [o for o in options if o.get("price")]
                if len(priced) >= 2:
                    cost = sorted(priced, key=lambda o: (o["price"]["input"] + o["price"]["output"]))
                    return (cost[-1] if role == "reason" else cost[0]), []
    if len(text) >= 3:
        hits = [o for o in options if text in o["model"].lower() or text in o["name"].lower()]
        if len(hits) == 1:
            return hits[0], []
        if hits:
            return None, hits
    return None, []


def distinct_tokens(options: list[dict[str, Any]]) -> set[str]:
    """Words that name one of the choices in a message ("opus", "haiku", "luna")."""
    out: set[str] = set()
    for opt in options:
        out.add(bare_model(opt["model"]).lower())
        for token in re.split(r"[-_./\s]+", bare_model(opt["model"]).lower()):
            if len(token) >= 4 and token.isalpha() and token not in _GENERIC_TOKENS:
                out.add(token)
    return out


def talks_about_models(text: str, options: list[dict[str, Any]] | None = None, *, loose: bool = False) -> bool:
    """Does *text* talk about the agent's model — the word, or one of the
    choices by name? *loose* also counts asking for a smarter or cheaper one
    ("use something cheaper"), which says too little for anything but
    checking a switch the agent says was asked for."""
    body = str(text or "")
    if not body.strip():
        return False
    if _MODEL_TALK_RE.search(body) or (loose and _SWITCH_ASK_RE.search(body)):
        return True
    lowered = body.lower()
    words = set(re.findall(r"[\w.\-]+", lowered))
    return any(tok in words or (("-" in tok or "." in tok) and tok in lowered)
               for tok in distinct_tokens(options or []))


def price_text(price: dict[str, float] | None) -> str:
    """"$3 in / $15 out per 1M tokens", or "" when unknown."""
    if not price:
        return ""

    def fmt(v: float) -> str:
        return f"${v:g}" if v >= 1 else f"${v:.2f}".rstrip("0").rstrip(".")

    return f"{fmt(price['input'])} in / {fmt(price['output'])} out per 1M tokens"
