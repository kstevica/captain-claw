"""Reasoning effort — one scale for every model's thinking controls.

The kiosk chat box offers a reasoning-effort picker. Every vendor spells
"think harder" differently (``reasoning_effort``, ``thinking.budget_tokens``,
``output_config.effort``, ``thinkingConfig.thinkingLevel``, ``enable_thinking``,
``think`` …), accepts a different set of values, and some refuse the field
outright. This module owns that translation so the rest of the code deals in
one vocabulary:

    off · minimal · low · medium · high · xhigh · max      (and "on")

* :func:`profile_for` says which of those levels a model offers and what it
  runs at when nothing is sent — the chat box shows exactly those.
* :func:`plan_for` turns a chosen level into the request fields for the
  transport the agent uses (the vendor's own API, OpenRouter, a local
  vLLM/SGLang/MLX server, Ollama).

"on" is the single "thinking on" step of a model that only toggles thinking
(Kimi, GLM, a hybrid Qwen on a local server). A level the model doesn't offer
is clamped to the nearest one it does, so switching an agent's model never
sends a value the new model rejects.

The level reaches the provider through a context variable: the chat frame
sets the turn's level (:func:`turn_effort`) and the agent's guarded LLM call
opens it for the main conversation calls only (:func:`call_effort`), so
housekeeping calls (titles, rephrase, compaction, reflections) keep the
model's default and don't pay for deep thinking.

Turns from anywhere but the web chat (WhatsApp, Telegram, the app, a device,
a scheduled task …) take the agent's own settings instead — a default and a
level per source, kept in ``~/.captain-claw/reasoning_effort.json`` and read
afresh when it changes (:func:`configured_effort`).

The agent changes them too, with its ``reasoning_effort`` tool: the same
settings, so what it sets is what a person sees (and could set) in Flight Deck.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator
from urllib.parse import urlparse

LEVELS: tuple[str, ...] = ("off", "minimal", "low", "medium", "high", "xhigh", "max")
ON = "on"
_RANK: dict[str, float] = {lvl: float(i) for i, lvl in enumerate(LEVELS)}
_RANK[ON] = _RANK["medium"]

_ALIASES = {
    "none": "off", "disable": "off", "disabled": "off", "false": "off", "no": "off", "0": "off",
    "enabled": ON, "true": ON, "yes": ON, "1": ON, "think": ON, "thinking": ON,
    "x-high": "xhigh", "extra-high": "xhigh", "extra_high": "xhigh", "extrahigh": "xhigh",
    "very-high": "xhigh", "maximum": "max", "min": "minimal",
}


def normalize_level(value: Any) -> str | None:
    """A level from the wire, or ``None`` for "auto" (leave the model alone)."""
    text = str(value or "").strip().lower()
    if not text or text in ("auto", "default"):
        return None
    text = _ALIASES.get(text, text)
    return text if text in _RANK else None


# ── Turn / call scope ─────────────────────────────────────────────────

_TURN: ContextVar[str | None] = ContextVar("cc_reasoning_turn", default=None)
_CALL: ContextVar[str | None] = ContextVar("cc_reasoning_call", default=None)


@contextmanager
def turn_effort(level: Any) -> Iterator[str | None]:
    """The user's pick for this turn. Tasks created inside inherit it."""
    token = _TURN.set(normalize_level(level))
    try:
        yield _TURN.get()
    finally:
        _TURN.reset(token)


@contextmanager
def call_effort(level: Any) -> Iterator[str | None]:
    """Apply *level* to the LLM call made inside (None: the model's default —
    a housekeeping call, or a turn nobody picked a level for)."""
    token = _CALL.set(normalize_level(level))
    try:
        yield _CALL.get()
    finally:
        _CALL.reset(token)


# A housekeeping call: a task name, next-step pills, a reflection, insight or
# lesson extraction. These carry small output budgets, and a thinking model
# counts its thinking against them — Qwen 3.8 on llama.cpp thought through
# all 25 tokens of a task name and all 1500 of a reflection and never wrote
# the answer. So thinking goes off where the model can switch it off; where it
# can't, or already doesn't think unless asked, the request says nothing.
QUIET = "quiet"


@contextmanager
def quiet_call() -> Iterator[None]:
    """Run the LLM call made inside as a housekeeping call (see ``QUIET``)."""
    token = _CALL.set(QUIET)
    try:
        yield
    finally:
        _CALL.reset(token)


def current_turn_effort() -> str | None:
    return _TURN.get()


# A level forced from outside for the agent's whole run — a benchmark harness
# (captain-claw-once --lock). It beats every other pick for the main
# conversation calls, and the reasoning_effort tool refuses to change it.
# Housekeeping calls keep the model's default, as always.
LOCK_ENV = "CLAW_LOCK_REASONING_EFFORT"


def locked_effort() -> str | None:
    """The level forced for this run (:data:`LOCK_ENV`), or ``None``."""
    return normalize_level(os.environ.get(LOCK_ENV, ""))


def requested_effort() -> str | None:
    """The level the provider should apply to the request it is building."""
    return _CALL.get()


# ── The agent's own settings (every source but the web chat) ──────────

SETTINGS_PATH = Path("~/.captain-claw/reasoning_effort.json").expanduser()
AUTO = "auto"
_settings_cache: dict[str, Any] = {"mtime": None, "value": None}


def clean_settings(raw: Any) -> dict[str, Any]:
    """``{"default": level|"auto", "channels": {source: level|"auto"}}`` with
    anything malformed dropped. A missing default or source falls through
    (source → its channel → the default); "auto" sends nothing."""
    from captain_claw import turn_source

    raw = raw if isinstance(raw, dict) else {}
    out: dict[str, Any] = {"channels": {}}
    default = _setting_value(raw.get("default"))
    if default:
        out["default"] = default
    channels = raw.get("channels") if isinstance(raw.get("channels"), dict) else {}
    for key, value in channels.items():
        source, level = turn_source.normalize(key), _setting_value(value)
        if source and source != "web" and level:
            out["channels"][source] = level
    return out


def _setting_value(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text == AUTO:
        return AUTO
    return normalize_level(text) or ""


def load_settings(path: Path | None = None) -> dict[str, Any]:
    """The settings file, cleaned; re-read only when it changes."""
    target = path or SETTINGS_PATH
    try:
        mtime = target.stat().st_mtime
    except OSError:
        return {"channels": {}}
    if path is None and _settings_cache["mtime"] == mtime and _settings_cache["value"] is not None:
        return _settings_cache["value"]
    try:
        value = clean_settings(json.loads(target.read_text(encoding="utf-8")))
    except (OSError, ValueError):
        value = {"channels": {}}
    if path is None:
        _settings_cache.update(mtime=mtime, value=value)
    return value


def save_settings(settings: Any, path: Path | None = None) -> dict[str, Any]:
    """Write *settings* (cleaned, as Flight Deck's Settings write them); the
    next lookup sees them. Returns what was written."""
    target = path or SETTINGS_PATH
    cleaned = clean_settings(settings)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    tmp.write_text(json.dumps(cleaned, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, target)
    if path is None:
        _settings_cache.update(mtime=target.stat().st_mtime, value=cleaned)
    return cleaned


def governing_key(source: str, settings: dict[str, Any] | None = None) -> str:
    """Which setting decides a turn from *source*: "web" (the chat box's
    picker), the source's own key, its channel's, else "default"."""
    from captain_claw import turn_source

    if turn_source.channel_of(source) == "web":
        return "web"
    cfg = settings if settings is not None else load_settings()
    channels = cfg.get("channels") or {}
    for key in (source, turn_source.channel_of(source)):
        if key and key in channels:
            return key
    return "default"


def configured_effort(source: str, settings: dict[str, Any] | None = None) -> str | None:
    """The level the agent's settings give a turn from *source*, or None
    (send nothing). The web chat has the chat box's picker instead; an
    unknown source takes the default."""
    cfg = settings if settings is not None else load_settings()
    key = governing_key(source, cfg)
    if key == "web":
        return None
    value = cfg.get("default") if key == "default" else (cfg.get("channels") or {}).get(key)
    return None if value in (None, "", AUTO) else value


# ── Profiles ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Profile:
    """What the picker offers for one model."""

    family: str
    levels: tuple[str, ...]
    # The level the model runs at when the request says nothing ("off" when
    # it doesn't think unless asked).
    default: str
    note: str = ""

    def to_wire(self) -> dict[str, Any]:
        out: dict[str, Any] = {"family": self.family, "levels": list(self.levels), "default": self.default}
        if self.note:
            out["note"] = self.note
        return out


def clamp(level: str | None, profile: Profile | None) -> str | None:
    """*level* if the model offers it, else the nearest level it does (a tie
    goes to the cheaper one). ``None`` stays ``None``."""
    if level is None or profile is None or not profile.levels:
        return None
    levels = profile.levels
    if level in levels:
        return level
    if level == ON:
        if profile.default in levels and profile.default != "off":
            return profile.default
        on_levels = [lv for lv in levels if lv != "off"]
        return on_levels[len(on_levels) // 2] if on_levels else None
    if ON in levels:
        return "off" if level == "off" and "off" in levels else ON
    rank = _RANK[level]
    return min(levels, key=lambda lv: (abs(_RANK[lv] - rank), _RANK[lv]))


# ── Model identification ──────────────────────────────────────────────


def model_key(model: str) -> str:
    """The bare model id, lowercased: routing prefixes, org paths and tags
    dropped ("openrouter/moonshotai/kimi-k2.6:free" → "kimi-k2.6",
    "mlx-community/Qwen3.6-35B-A3B-4bit" → "qwen3.6-35b-a3b-4bit"). Ollama
    size tags ("qwen3:8b") are dropped too — the family is in the name."""
    text = str(model or "").strip().lower()
    text = text.rsplit("/", 1)[-1]
    text = text.split(":", 1)[0]
    text = text.split("@", 1)[0]
    return text


def _version(text: str) -> tuple[int, int]:
    """"4-6" / "4.6" / "4" → (4, 6) / (4, 6) / (4, 0)."""
    parts = re.split(r"[.\-_]", text)
    try:
        major = int(parts[0])
    except (ValueError, IndexError):
        return (0, 0)
    minor = 0
    if len(parts) > 1 and parts[1].isdigit() and len(parts[1]) <= 2:
        minor = int(parts[1])
    return (major, minor)


# ── Transport ─────────────────────────────────────────────────────────

# Who receives the request decides the field names: the vendor's own API,
# OpenRouter's unified `reasoning`, a self-hosted server's chat template, or
# Ollama's native `think`.
T_OPENAI = "openai"
T_ANTHROPIC = "anthropic"
T_GEMINI = "gemini"
T_GEMINI_COMPAT = "gemini-compat"
T_XAI = "xai"
T_DEEPSEEK = "deepseek"
T_MOONSHOT = "moonshot"
T_ZAI = "zai"
T_DASHSCOPE = "dashscope"
T_MINIMAX = "minimax"
T_MISTRAL = "mistral"
T_OPENROUTER = "openrouter"
T_OLLAMA = "ollama"
T_CODEX = "codex"
T_LOCAL = "local"  # vLLM, SGLang, MLX, LM Studio, llama.cpp, lithos …

_HOST_TRANSPORTS: tuple[tuple[str, str], ...] = (
    ("api.openai.com", T_OPENAI),
    ("openai.azure.com", T_OPENAI),
    ("chatgpt.com", T_CODEX),
    ("anthropic.com", T_ANTHROPIC),
    ("generativelanguage.googleapis.com", T_GEMINI),
    ("api.x.ai", T_XAI),
    ("deepseek.com", T_DEEPSEEK),
    ("moonshot.ai", T_MOONSHOT),
    ("moonshot.cn", T_MOONSHOT),
    ("z.ai", T_ZAI),
    ("bigmodel.cn", T_ZAI),
    ("dashscope", T_DASHSCOPE),
    ("aliyuncs.com", T_DASHSCOPE),
    ("minimax.io", T_MINIMAX),
    ("minimaxi.com", T_MINIMAX),
    ("minimax.chat", T_MINIMAX),
    ("mistral.ai", T_MISTRAL),
    ("openrouter.ai", T_OPENROUTER),
)

_PROVIDER_TRANSPORTS = {
    "anthropic": T_ANTHROPIC,
    "gemini": T_GEMINI,
    "xai": T_XAI,
    "deepseek": T_DEEPSEEK,
    "openrouter": T_OPENROUTER,
    "ollama": T_OLLAMA,
}


def transport_for(provider: str, base_url: str | None = None) -> str:
    """Which API shape a request from *provider* at *base_url* lands on."""
    p = str(provider or "").strip().lower()
    host = ""
    if base_url:
        try:
            host = (urlparse(base_url if "://" in base_url else f"http://{base_url}").hostname or "").lower()
        except ValueError:
            host = ""
    if host:
        for needle, transport in _HOST_TRANSPORTS:
            if needle in host:
                # Google's and Anthropic's OpenAI-compatible endpoints speak
                # the OpenAI shape, not their native one.
                if transport == T_GEMINI and p != "gemini":
                    return T_GEMINI_COMPAT
                if transport == T_ANTHROPIC and p != "anthropic":
                    return T_LOCAL
                return transport
        if p == "ollama" or host.endswith("ollama.com") or ":11434" in str(base_url):
            return T_OLLAMA
        if p in ("openai", "openrouter", "deepseek", "xai", ""):
            return T_LOCAL
    if p in _PROVIDER_TRANSPORTS:
        return _PROVIDER_TRANSPORTS[p]
    if p == "openai":
        return T_OPENAI
    return T_LOCAL if host else p


# ── Plans ─────────────────────────────────────────────────────────────


@dataclass
class Plan:
    """The request changes one level needs on one transport."""

    level: str
    # LiteLLM keyword arguments (validated / translated by LiteLLM).
    kwargs: dict[str, Any] = field(default_factory=dict)
    # Fields merged straight into the JSON body (LiteLLM ``extra_body``).
    body: dict[str, Any] = field(default_factory=dict)
    # Request keys to remove (e.g. ``temperature`` while thinking).
    drop: tuple[str, ...] = ()
    # Ollama's native ``think`` (None: leave the provider's own choice).
    ollama_think: bool | str | None = None
    # Responses API ``reasoning.effort`` (ChatGPT/Codex backend).
    responses_effort: str | None = None
    # Thinking on with a fixed budget (Anthropic legacy): must not be sent on
    # a continuation of a tool call whose thinking block we can't echo.
    needs_thinking_echo: bool = False
    budget_tokens: int | None = None


# ── Model families ────────────────────────────────────────────────────
#
# Each resolver below knows one vendor's models: the levels they offer on the
# vendor's own API and the fields that set them, and — for open-weight models
# — how a self-hosted server's chat template and Ollama switch thinking.
# Sources (checked 2026-10-24): the vendors' API references
# (platform.claude.com, developers.openai.com, docs.x.ai, ai.google.dev,
# api-docs.deepseek.com, platform.kimi.ai, docs.z.ai, platform.minimax.io,
# Alibaba Model Studio, docs.mistral.ai), docs.ollama.com and the vLLM
# recipes. OpenRouter publishes each model's levels itself (see below).

Build = Callable[[str, "int | None"], Plan]

FULL_EFFORT = ("low", "medium", "high", "xhigh", "max")
ALWAYS_ON = "Thinking is always on for this model — pick how deep it goes."
TOGGLE = ("off", ON)


@dataclass(frozen=True)
class _Way:
    """The levels one transport offers for a model, and how to send them."""

    levels: tuple[str, ...]
    default: str
    build: Build
    note: str = ""


@dataclass(frozen=True)
class _Model:
    family: str
    # The vendor's own API (and the transports that speak it).
    api: _Way | None = None
    vendors: frozenset[str] = frozenset()
    # Open weights on a self-hosted server (vLLM, SGLang, MLX …).
    local: _Way | None = None
    # Ollama's native `think`: "bool" (on/off) or "levels" (low/medium/high).
    ollama: str | None = None
    ollama_default: str = ON


def _plan(level: str, **fields: Any) -> Plan:
    return Plan(level=level, **fields)


def _template_toggle(kwargs: Callable[[bool], dict[str, Any]], default: str = ON) -> _Way:
    """A self-hosted model whose chat template switches thinking on or off."""
    return _Way(TOGGLE, default, lambda level, _max: _plan(
        level, body={"chat_template_kwargs": kwargs(level != "off")}))


def _thinking_type(on: str = "enabled", off: str = "disabled", drop: tuple[str, ...] = ()) -> Build:
    """``thinking: {type: …}`` on/off (Kimi K2.x, GLM, MiniMax M3)."""
    return lambda level, _max: _plan(
        level, body={"thinking": {"type": off if level == "off" else on}}, drop=drop)


# Anthropic ─────────────────────────────────────────────────────────────

_CLAUDE_RE = re.compile(r"^claude-(opus|sonnet|haiku|fable|mythos)-(\d+)(?:[-.](\d{1,2}))?(?=$|[-.@])")
_CLAUDE_OLD_RE = re.compile(r"^claude-(\d)(?:[-.](\d))?-(opus|sonnet|haiku)")
# Fixed thinking budgets for the models that still take ``budget_tokens``
# (at least 1024, and under max_tokens with room left for the answer).
_CLAUDE_BUDGETS = {"minimal": 1024, "low": 2048, "medium": 8192, "high": 16384, "max": 32000}
_CLAUDE_ANSWER_ROOM = 4096
_SAMPLING = ("temperature", "top_p", "top_k")


def _claude_effort(off_mode: str | None, adaptive_explicit: bool) -> Build:
    """Adaptive-thinking Claude: depth through ``output_config.effort``.

    *off_mode*: how "off" is sent — ``"disabled"``, ``"between_tools"``
    (Sonnet 5.5, where ``disabled`` is a 400) or None (thinking is off
    already when the request says nothing). *adaptive_explicit*: thinking
    must be switched on (Opus 4.6–4.8, Sonnet 4.6); newer models think
    unless told otherwise.
    """
    def build(level: str, _max: int | None) -> Plan:
        if level == "off":
            return _plan(level, body={"thinking": {"type": off_mode}} if off_mode else {})
        body: dict[str, Any] = {"output_config": {"effort": level}}
        if adaptive_explicit:
            body["thinking"] = {"type": "adaptive"}
        return _plan(level, body=body, drop=_SAMPLING)
    return build


def _claude_budget(level: str, max_tokens: int | None) -> Plan:
    if level == "off":
        return _plan(level)
    budget = _CLAUDE_BUDGETS.get(level, _CLAUDE_BUDGETS["medium"])
    if max_tokens:
        budget = min(budget, int(max_tokens) - _CLAUDE_ANSWER_ROOM)
    if budget < 1024:
        return _plan(level)  # no room to think beside the answer
    return _plan(
        level,
        body={"thinking": {"type": "enabled", "budget_tokens": budget}},
        drop=_SAMPLING,
        needs_thinking_echo=True,
        budget_tokens=budget,
    )


def _claude(key: str) -> _Model | None:
    m = _CLAUDE_RE.match(key)
    if m:
        tier, ver = m.group(1), (int(m.group(2)), int(m.group(3) or 0))
    else:
        m = _CLAUDE_OLD_RE.match(key)
        if not m:
            return None
        tier, ver = m.group(3), (int(m.group(1)), int(m.group(2) or 0))

    def model(levels: tuple[str, ...], default: str, build: Build, note: str = "") -> _Model:
        return _Model("claude", _Way(levels, default, build, note), frozenset({T_ANTHROPIC}))

    def effort(levels: tuple[str, ...], default: str, off_mode: str | None,
               explicit: bool = False, note: str = "") -> _Model:
        return model(levels, default, _claude_effort(off_mode, explicit), note)

    # Fable / Mythos and Opus 5.5+: thinking can't be switched off.
    if tier in ("fable", "mythos"):
        return effort(FULL_EFFORT, "high", None, note=ALWAYS_ON)
    if tier == "opus" and ver >= (5, 5):
        return effort(FULL_EFFORT, "medium", None, note=ALWAYS_ON)
    if tier == "opus" and ver >= (5, 0):
        return effort(("off",) + FULL_EFFORT, "high", "disabled")
    if tier == "sonnet" and ver >= (5, 5):
        return effort(("off",) + FULL_EFFORT, "high", "between_tools")
    if tier == "sonnet" and ver >= (5, 0):
        return effort(("off",) + FULL_EFFORT, "high", "disabled")
    if tier == "haiku" and ver >= (5, 0):
        return effort(("off",) + FULL_EFFORT, "medium", "disabled")
    if tier == "opus" and ver >= (4, 7):
        return effort(("off",) + FULL_EFFORT, "off", None, explicit=True)
    if ver == (4, 6) and tier in ("opus", "sonnet"):
        return effort(("off", "low", "medium", "high", "max"), "off", None, explicit=True)
    # Opus 4.5, Sonnet/Haiku 4.5, Opus/Sonnet 4.x, Sonnet 3.7: fixed budgets.
    if ver >= (4, 0) or (ver == (3, 7) and tier == "sonnet"):
        return model(("off", "low", "medium", "high", "max"), "off", _claude_budget)
    return None


# OpenAI ────────────────────────────────────────────────────────────────

_GPT_RE = re.compile(r"^gpt-(\d+)(?:\.(\d+))?(.*)$")
_O_SERIES_RE = re.compile(r"^o(\d+)(-mini|-pro|-preview)?(?=$|-)")
_TOOLS_NEED_NONE = (
    "OpenAI's Chat Completions runs this model without reasoning on calls "
    "that use tools; the level applies to the replies that don't."
)


def _openai_effort(level: str, _max: int | None) -> Plan:
    effort = "none" if level == "off" else level
    drop = () if level == "off" else ("temperature", "top_p", "logprobs", "top_logprobs")
    # In the body rather than LiteLLM's reasoning_effort: its model map lags
    # new models and refuses xhigh/max for the ones it doesn't know.
    return _plan(level, body={"reasoning_effort": effort}, drop=drop, responses_effort=effort)


def _openai(key: str) -> _Model | None:
    def model(levels: tuple[str, ...], default: str, note: str = "") -> _Model:
        return _Model("openai", _Way(levels, default, _openai_effort, note), frozenset({T_OPENAI, T_CODEX}))

    m = _O_SERIES_RE.match(key)
    if m:
        if m.group(1) == "1" and m.group(2) in ("-mini", "-preview"):
            return None
        return model(("low", "medium", "high"), "medium")
    if key.startswith("codex-mini"):
        return model(("low", "medium", "high"), "medium")
    m = _GPT_RE.match(key)
    if not m:
        return None
    major, minor, rest = int(m.group(1)), int(m.group(2) or 0), m.group(3)
    if major < 5:
        return None
    if major >= 7:
        return model(FULL_EFFORT, "medium")
    if major == 6:
        if "astra" in rest or minor >= 1:
            return model(FULL_EFFORT, "medium", _TOOLS_NEED_NONE)
        return model(("off",) + FULL_EFFORT, "medium", _TOOLS_NEED_NONE)
    # GPT-5.x
    if rest.startswith("-chat") and minor == 0:
        return None  # gpt-5-chat: not a reasoning model
    if "-pro" in rest:
        if minor == 0:
            return None  # gpt-5-pro: high only
        return model(("medium", "high", "xhigh"), "high" if minor >= 5 else "medium")
    if "codex" in rest:
        if minor >= 2 or "max" in rest:
            return model(("low", "medium", "high", "xhigh"), "medium")
        return model(("low", "medium", "high"), "medium")
    if minor >= 6:
        return model(("off",) + FULL_EFFORT, "medium")
    if minor == 5:
        return model(("off", "low", "medium", "high", "xhigh"), "medium")
    if minor >= 2:
        return model(("off", "low", "medium", "high", "xhigh"), "off")
    if minor == 1:
        return model(("off", "low", "medium", "high"), "off")
    return model(("minimal", "low", "medium", "high"), "medium")


def _gpt_oss(key: str) -> _Model | None:
    if not key.startswith("gpt-oss"):
        return None
    # Open weights: vLLM/SGLang read a top-level reasoning_effort, Ollama a
    # `think` level. Reasoning can't be switched off.
    return _Model(
        "gpt-oss",
        local=_Way(("low", "medium", "high"), "medium",
                   lambda level, _max: _plan(level, body={"reasoning_effort": level})),
        ollama="levels", ollama_default="medium",
    )


# xAI ───────────────────────────────────────────────────────────────────

_GROK_RE = re.compile(r"^grok-(\d+)(?:\.(\d)(?!\d))?")


def _xai(key: str) -> _Model | None:
    if key.startswith("grok-3-mini"):
        levels, default = ("low", "high"), "low"
    else:
        m = _GROK_RE.match(key)
        if not m or any(w in key for w in ("fast", "code", "build", "multi-agent")):
            return None  # retired fast models, grok-code/build: no effort control
        major, minor = int(m.group(1)), m.group(2)
        if major == 4 and minor is None:
            return None  # grok-4-0709 (retired), grok-4.20: reasoning is the model id
        ver = (major, int(minor or 0))
        if ver >= (4, 6):
            levels, default = ("low", "medium", "high", "xhigh"), "high"
        elif ver == (4, 5):
            levels, default = ("low", "medium", "high"), "high"
        elif ver == (4, 3):
            levels, default = ("off", "low", "medium", "high"), "low"
        else:
            return None
    return _Model("grok", _Way(levels, default, lambda level, _max: _plan(
        level, body={"reasoning_effort": "none" if level == "off" else level})), frozenset({T_XAI}))


# Google ────────────────────────────────────────────────────────────────

# thinkingBudget per level on Gemini 2.5 (Pro: 128–32768, can't be off).
_GEMINI_BUDGETS = {"off": 0, "minimal": 512, "low": 1024, "medium": 8192, "high": 16384, "max": 24576}


def _gemini_level(level: str, _max: int | None) -> Plan:
    # A plain kwarg: LiteLLM passes thinkingConfig into generationConfig
    # untouched (its reasoning_effort mapping is stale for Gemini 3.6+/4).
    return _plan(level, kwargs={"thinkingConfig": {"thinkingLevel": level}})


def _gemini_budget(pro: bool) -> Build:
    def build(level: str, _max: int | None) -> Plan:
        budget = _GEMINI_BUDGETS.get(level, 8192)
        if pro:
            budget = 32768 if level == "max" else max(128, budget)
        return _plan(level, kwargs={"thinkingConfig": {"thinkingBudget": budget}})
    return build


def _gemini(key: str) -> _Model | None:
    if not key.startswith("gemini-") or any(w in key for w in ("image", "embedding", "tts", "live")):
        return None
    m = re.match(r"^gemini-(\d+)(?:\.(\d+))?-(.*)$", key)
    if not m:
        return None
    ver, variant = (int(m.group(1)), int(m.group(2) or 0)), m.group(3)
    pro, lite, flash = variant.startswith("pro"), "lite" in variant, variant.startswith("flash")

    def model(levels: tuple[str, ...], default: str, build: Build = _gemini_level, note: str = "") -> _Model:
        return _Model("gemini", _Way(levels, default, build, note), frozenset({T_GEMINI}))

    if ver < (2, 5):
        return None
    if ver < (3, 0):
        if pro:
            return model(("low", "medium", "high", "max"), "medium", _gemini_budget(True),
                         "Thinking can't be turned off on this model.")
        return model(("off", "low", "medium", "high", "max"), "off" if lite else "medium", _gemini_budget(False))
    if ver >= (4, 0):
        return model(("low", "medium", "high"), "medium")
    if pro:
        return model(("low", "high") if ver == (3, 0) else ("low", "medium", "high"), "high")
    if flash:
        if ver >= (3, 7):
            return model(("low", "medium", "high"), "medium")
        default = "minimal" if lite else ("high" if ver == (3, 0) else "medium")
        return model(("minimal", "low", "medium", "high"), default,
                     note="This model can't turn thinking off; minimal thinks very little.")
    return None


def _gemma(key: str) -> _Model | None:
    if not (key.startswith("gemma-4") or key.startswith("gemma4")):
        return None
    # Thinking on/off only, off by default.
    return _Model(
        "gemma",
        _Way(TOGGLE, "off", lambda level, _max: _plan(
            level, kwargs={"thinkingConfig": {"thinkingLevel": "minimal" if level == "off" else "high"}})),
        frozenset({T_GEMINI}),
        local=_template_toggle(lambda on: {"enable_thinking": on}, "off"),
        ollama="bool", ollama_default="off",
    )


# DeepSeek ──────────────────────────────────────────────────────────────


def _deepseek_v4(level: str, _max: int | None) -> Plan:
    if level == "off":
        return _plan(level, body={"thinking": {"type": "disabled"}})
    return _plan(level, body={"thinking": {"type": "enabled"}, "reasoning_effort": level})


def _deepseek_v4_local(level: str, _max: int | None) -> Plan:
    on = level != "off"
    kwargs: dict[str, Any] = {"thinking": on, "enable_thinking": on}
    if on:
        kwargs["reasoning_effort"] = level
    return _plan(level, body={"chat_template_kwargs": kwargs})


def _deepseek(key: str) -> _Model | None:
    if not key.startswith("deepseek"):
        return None
    m = re.match(r"^deepseek-v(\d+)(?:\.(\d+))?", key)
    ver = (int(m.group(1)), int(m.group(2) or 0)) if m else None
    if key.startswith(("deepseek-flash", "deepseek-pro")) or (ver and ver >= (4, 0)):
        # V4 / V4.1: low|high|max (minimal/medium/xhigh are aliases), off via
        # thinking.type. Thinking mode refuses a forced tool_choice (healed).
        return _Model(
            "deepseek",
            _Way(("off", "low", "high", "max"), "high", _deepseek_v4), frozenset({T_DEEPSEEK}),
            local=_Way(("off", "high", "max"), "off", _deepseek_v4_local),
            ollama="bool",
        )
    if ver and ver >= (3, 1):
        # V3.1/V3.2 open weights: hybrid, off unless the template is told.
        return _Model("deepseek", local=_template_toggle(lambda on: {"thinking": on}, "off"), ollama="bool")
    return None  # deepseek-chat / -reasoner (retired), R1: thinking set by the model


# Moonshot Kimi ─────────────────────────────────────────────────────────


def _kimi(key: str) -> _Model | None:
    m = re.match(r"^kimi-k(\d+)(?:\.(\d+))?(.*)$", key)
    if not m:
        return None
    ver, rest = (int(m.group(1)), int(m.group(2) or 0)), m.group(3)
    if ver >= (3, 0):
        # K3 family: always thinking, low|high|max; temperature is fixed.
        return _Model("kimi", _Way(("low", "high", "max"), "max", lambda level, _max: _plan(
            level, body={"reasoning_effort": level}, drop=("temperature", "top_p")), ALWAYS_ON),
            frozenset({T_MOONSHOT}))
    if "code" in rest or "thinking" in rest or ver < (2, 5):
        return None  # k2.7-code / k2-thinking always think; k2 never does
    return _Model(
        "kimi", _Way(TOGGLE, ON, _thinking_type(drop=("temperature", "top_p"))), frozenset({T_MOONSHOT}),
        local=_template_toggle(lambda on: {"thinking": on}),
        ollama="bool",
    )


# Zhipu GLM ─────────────────────────────────────────────────────────────


def _glm_effort(level: str, _max: int | None) -> Plan:
    if level == "off":
        return _plan(level, body={"thinking": {"type": "disabled"}})
    return _plan(level, body={"thinking": {"type": "enabled"}, "reasoning_effort": level})


def _glm(key: str) -> _Model | None:
    m = re.match(r"^glm-(\d+)(?:\.(\d+))?", key)
    if not m:
        return None
    ver = (int(m.group(1)), int(m.group(2) or 0))
    api = frozenset({T_ZAI})
    local = _template_toggle(lambda on: {"enable_thinking": on})
    if ver >= (5, 3):
        # 5.3: always thinking, low|high|max.
        return _Model("glm", _Way(("low", "high", "max"), "max", _glm_effort, ALWAYS_ON), api, ollama="bool")
    if ver == (5, 2):
        # 5.2: low/medium act as high, xhigh as max.
        return _Model("glm", _Way(("off", "high", "max"), "max", _glm_effort), api, local=local, ollama="bool")
    if ver >= (4, 5):
        return _Model("glm", _Way(TOGGLE, ON, _thinking_type()), api, local=local, ollama="bool")
    return None


# MiniMax ───────────────────────────────────────────────────────────────


def _minimax(key: str) -> _Model | None:
    m = re.match(r"^minimax-m(\d+)(?:\.(\d+))?(.*)$", key)
    if not m:
        return None
    ver, rest = (int(m.group(1)), int(m.group(2) or 0)), m.group(3)
    api = frozenset({T_MINIMAX})
    if ver < (3, 0):
        return None  # M2.x always think and ignore the switch
    if "flash" in rest:
        return _Model("minimax", _Way(FULL_EFFORT, "max", lambda level, _max: _plan(
            level, body={"reasoning_effort": level}), ALWAYS_ON), api)
    return _Model(
        "minimax", _Way(TOGGLE, ON, _thinking_type(on="adaptive")), api,
        local=_template_toggle(lambda on: {"thinking_mode": "enabled" if on else "disabled"}),
        ollama="bool",
    )


# Alibaba Qwen ──────────────────────────────────────────────────────────

# Model Studio thinking_budget per level ("max": the model's own ceiling).
_QWEN_BUDGETS = {"minimal": 1024, "low": 2048, "medium": 8192, "high": 32768}
_QWEN38_LEVELS = ("low", "medium", "xhigh")


def _qwen_budget(level: str, _max: int | None) -> Plan:
    if level == "off":
        return _plan(level, body={"enable_thinking": False})
    body: dict[str, Any] = {"enable_thinking": True}
    if level in _QWEN_BUDGETS:
        body["thinking_budget"] = _QWEN_BUDGETS[level]
    return _plan(level, body=body)


def _qwen38(level: str, _max: int | None) -> Plan:
    # Qwen3.8 reads reasoning_effort (low|medium|xhigh); never beside a
    # thinking_budget, which Model Studio refuses.
    if level == "off":
        return _plan(level, body={"enable_thinking": False})
    return _plan(level, body={"reasoning_effort": level})


def _qwen38_local(level: str, _max: int | None) -> Plan:
    # The 3.8 chat template raises on any effort but low|medium|xhigh.
    kwargs: dict[str, Any] = {"enable_thinking": level != "off"}
    if level != "off":
        kwargs["reasoning_effort"] = level
    return _plan(level, body={"chat_template_kwargs": kwargs})


def _qwen(key: str) -> _Model | None:
    if not key.startswith("qwen"):
        return None  # QwQ always thinks
    if any(w in key for w in ("thinking", "instruct", "coder", "qwen-max", "-vl-", "tts", "embedding")):
        return None  # 2507/Next splits: Thinking always, Instruct never; qwen-max: no thinking
    api = frozenset({T_DASHSCOPE})
    budget_levels = ("off", "low", "medium", "high", "max")
    if key.startswith(("qwen-plus", "qwen-flash", "qwen-turbo", "qwen3-max")):
        # Commercial hybrids: thinking off unless asked.
        return _Model("qwen", _Way(budget_levels, "off", _qwen_budget), api)
    m = re.match(r"^qwen(\d+)(?:\.(\d+))?", key)
    if not m:
        return None
    ver = (int(m.group(1)), int(m.group(2) or 0))
    if ver < (3, 0):
        return None
    local_toggle = _template_toggle(lambda on: {"enable_thinking": on})
    if ver >= (3, 8):
        if "2.4t" in key:  # thinking-only
            return _Model("qwen", _Way(_QWEN38_LEVELS, "xhigh", _qwen38, ALWAYS_ON), api)
        return _Model(
            "qwen", _Way(("off",) + _QWEN38_LEVELS, "xhigh", _qwen38), api,
            local=_Way(("off",) + _QWEN38_LEVELS, "xhigh", _qwen38_local),
            ollama="bool",
        )
    if key.startswith("qwen3.7-max-preview") or "max-2026-05-17" in key:  # thinking-only
        return _Model("qwen", _Way(budget_levels[1:], "max", _qwen_budget, ALWAYS_ON), api)
    # Qwen3 hybrids, 3.5, 3.6, 3.7: thinking on by default.
    return _Model("qwen", _Way(budget_levels, "max", _qwen_budget), api, local=local_toggle, ollama="bool")


# Mistral ───────────────────────────────────────────────────────────────


def _mistral(key: str) -> _Model | None:
    if not re.match(r"^mistral-(small-4|small-26|small-latest|medium-3[.-]5|medium-latest|large-4|large-latest)", key):
        return None  # Magistral (retired) always reasoned; older models don't
    return _Model("mistral", _Way(("off", "high"), "high", lambda level, _max: _plan(
        level, body={"reasoning_effort": "none" if level == "off" else "high"})), frozenset({T_MISTRAL}))


# NVIDIA Nemotron ───────────────────────────────────────────────────────


def _nemotron(key: str) -> _Model | None:
    if "nemotron" not in key or "safety" in key:
        return None
    return _Model("nemotron", local=_template_toggle(lambda on: {"enable_thinking": on}), ollama="bool")


# ── OpenRouter ────────────────────────────────────────────────────────
#
# OpenRouter publishes each model's reasoning controls on its public model
# list (`reasoning.supported_efforts`, `default_effort`, `default_enabled`,
# `mandatory`) and translates its unified `reasoning` field for every
# upstream. The list is fetched once per process in the background; until it
# arrives (or if it can't be fetched) the family tables above stand in.

_OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
_OPENROUTER_TTL = 24 * 3600
_openrouter_meta: dict[str, dict[str, Any]] = {}
_openrouter_state = {"fetched_at": 0.0, "fetching": False}
_openrouter_lock = threading.Lock()


def _openrouter_fetch() -> None:
    try:
        import httpx

        resp = httpx.get(_OPENROUTER_MODELS_URL, timeout=15.0)
        resp.raise_for_status()
        meta: dict[str, dict[str, Any]] = {}
        for item in resp.json().get("data") or []:
            mid = str(item.get("id") or "").lower()
            if not mid:
                continue
            params = item.get("supported_parameters") or []
            reasoning = item.get("reasoning")
            meta[mid] = reasoning if isinstance(reasoning, dict) and "reasoning" in params else {"none": True}
        if meta:
            _openrouter_meta.clear()
            _openrouter_meta.update(meta)
            _openrouter_state["fetched_at"] = time.time()
    except Exception:
        pass
    finally:
        _openrouter_state["fetching"] = False


def _openrouter_info(model: str) -> dict[str, Any] | None:
    """OpenRouter's reasoning block for *model* (None: not known yet)."""
    with _openrouter_lock:
        stale = time.time() - _openrouter_state["fetched_at"] > _OPENROUTER_TTL
        if stale and not _openrouter_state["fetching"]:
            _openrouter_state["fetching"] = True
            threading.Thread(target=_openrouter_fetch, name="openrouter-models", daemon=True).start()
    mid = str(model or "").strip().lower()
    if mid.startswith("openrouter/"):
        mid = mid[len("openrouter/"):]
    for candidate in (mid, mid.split(":", 1)[0]):
        if candidate in _openrouter_meta:
            return _openrouter_meta[candidate]
    return None


def _openrouter_build(level: str, _max: int | None) -> Plan:
    if level == ON:
        reasoning: dict[str, Any] = {"enabled": True}
    else:
        reasoning = {"effort": "none" if level == "off" else level}
    return _plan(level, body={"reasoning": reasoning})


def _openrouter_way(model: str, found: _Model | None) -> _Way | None:
    info = _openrouter_info(model)
    if info is not None:
        if info.get("none"):
            return None  # OpenRouter takes no reasoning field for it
        efforts = [str(e).lower() for e in info.get("supported_efforts") or []]
        levels = tuple(lv for lv in LEVELS if lv in {"off" if e == "none" else e for e in efforts})
        enabled = info.get("default_enabled")
        if levels:
            default = str(info.get("default_effort") or "").lower()
            if enabled is False and "off" in levels:
                default = "off"
            elif default not in levels:
                default = levels[len(levels) // 2]
            return _Way(levels, default, _openrouter_build, ALWAYS_ON if info.get("mandatory") else "")
        if info.get("mandatory"):
            return None  # always thinks, nothing to pick
        # Upstreams OpenRouter maps effort for even when it lists no levels.
        if found is not None and found.api is not None and found.family in ("claude", "gemini", "openai", "grok"):
            return _Way(found.api.levels, found.api.default, _openrouter_build, found.api.note)
        return _Way(TOGGLE, "off" if enabled is False else ON, _openrouter_build)
    if found is None:
        return None
    way = found.api or found.local
    if way is None:
        return _Way(TOGGLE, ON, _openrouter_build)
    if ON in way.levels or way.levels == TOGGLE:
        return _Way(way.levels, way.default, _openrouter_build)
    return _Way(way.levels, way.default, _openrouter_build, way.note)


# ── Assembly ──────────────────────────────────────────────────────────

_RESOLVERS: tuple[Callable[[str], _Model | None], ...] = (
    _claude, _gpt_oss, _openai, _xai, _gemini, _gemma, _deepseek,
    _kimi, _glm, _minimax, _qwen, _mistral, _nemotron,
)

# Models whose server refused the reasoning fields this run: they keep the
# model's default from then on (see llm._heal_request_kwargs).
_REFUSED: set[str] = set()


def remember_refused(model: str) -> None:
    key = model_key(model)
    if key:
        _REFUSED.add(key)


def was_refused(model: str) -> bool:
    return model_key(model) in _REFUSED


def _model_for(model: str) -> _Model | None:
    key = model_key(model)
    for resolve in _RESOLVERS:
        found = resolve(key)
        if found is not None:
            return found
    return None


def _way_for(provider: str, model: str, base_url: str | None) -> tuple[str, _Way] | None:
    """(family, way) for *model* reached through *provider* at *base_url*,
    or None when there is nothing to pick."""
    transport = transport_for(provider, base_url)
    found = _model_for(model)
    if transport == T_OPENROUTER:
        way = _openrouter_way(model, found)
        return (found.family if found else "openrouter", way) if way else None
    if found is None:
        return None
    if found.api is not None and transport in found.vendors:
        way = found.api
        if transport == T_CODEX and way.note == _TOOLS_NEED_NONE:
            way = _Way(way.levels, way.default, way.build)  # the Responses API takes both
        return found.family, way
    if transport == T_GEMINI_COMPAT and found.family == "gemini" and found.api is not None:
        # Google's OpenAI-compatible endpoint: reasoning_effort minimal..high,
        # "none" only where thinking can be turned off.
        levels = tuple(lv for lv in found.api.levels if lv != "max")
        return found.family, _Way(levels, found.api.default if found.api.default in levels else levels[-1],
                                  lambda level, _max: _plan(level, body={
                                      "reasoning_effort": "none" if level == "off" else level}),
                                  found.api.note)
    if transport == T_OLLAMA and found.ollama:
        if found.ollama == "levels":
            return found.family, _Way(("low", "medium", "high"), found.ollama_default,
                                      lambda level, _max: _plan(level, ollama_think=level))
        return found.family, _Way(TOGGLE, found.ollama_default,
                                  lambda level, _max: _plan(level, ollama_think=level != "off"))
    if transport == T_LOCAL and found.local is not None:
        return found.family, found.local
    return None


def profile_for(provider: str, model: str, base_url: str | None = None) -> Profile | None:
    """The levels the picker offers for this model, or None (no picker)."""
    routed = _way_for(provider, model, base_url)
    if routed is None:
        return None
    family, way = routed
    return Profile(family, way.levels, way.default, way.note)


def plan_for(provider: str, model: str, base_url: str | None, level: Any,
             *, max_tokens: int | None = None) -> Plan | None:
    """The request changes for *level* on this model, or None (send nothing:
    "auto", no controls, or the server refused them earlier this run)."""
    wanted = normalize_level(level)
    if wanted is None or model_key(model) in _REFUSED:
        return None
    routed = _way_for(provider, model, base_url)
    if routed is None:
        return None
    family, way = routed
    chosen = clamp(wanted, Profile(family, way.levels, way.default))
    return way.build(chosen, max_tokens) if chosen else None


def quiet_plan(provider: str, model: str, base_url: str | None,
               *, max_tokens: int | None = None) -> Plan | None:
    """Thinking off for a housekeeping call (see ``QUIET``), or None: the
    model can't switch thinking off, doesn't think unless asked, isn't known,
    or its server refused the controls earlier this run."""
    if model_key(model) in _REFUSED:
        return None
    routed = _way_for(provider, model, base_url)
    if routed is None:
        return None
    _family, way = routed
    if "off" not in way.levels or way.default == "off":
        return None
    return way.build("off", max_tokens)


_SHAPED_PROVIDERS = frozenset({"openai", "anthropic", "gemini", "xai", "deepseek", "openrouter", "ollama"})


def _provider_parts(provider_obj: Any) -> tuple[str, str, str | None]:
    """(provider, model, base_url) of a live LLM provider object."""
    provider = str(getattr(provider_obj, "provider", "") or "")
    model = str(getattr(provider_obj, "model", "") or "")
    base_url = getattr(provider_obj, "base_url", None) or None
    if type(provider_obj).__name__ == "ChatGPTResponsesProvider":
        base_url = base_url or "https://chatgpt.com/backend-api/codex/responses"
    return provider, model, base_url


def profile_for_provider(provider_obj: Any) -> Profile | None:
    """The picker for an agent's live provider (claude-cli, antigravity,
    litert and the browser provider build no request to shape)."""
    if provider_obj is None:
        return None
    provider, model, base_url = _provider_parts(provider_obj)
    if provider not in _SHAPED_PROVIDERS:
        return None
    return profile_for(provider, model, base_url)


def profile_for_config(provider: str, model: str, base_url: str | None = None) -> Profile | None:
    """The picker for an agent's configured model (its config.yaml), as the
    agent will build it once it runs that model."""
    provider = str(provider or "").strip().lower()
    if provider not in _SHAPED_PROVIDERS and provider not in ("claude", "chatgpt", "grok", "google"):
        return None
    from captain_claw.llm import _normalize_provider_name

    return profile_for(_normalize_provider_name(provider), model, base_url)


def plan_for_provider(provider_obj: Any, *, max_tokens: int | None = None) -> Plan | None:
    """The plan for the level the current call asked for (see call_effort)."""
    level = requested_effort()
    if level is None or provider_obj is None:
        return None
    provider, model, base_url = _provider_parts(provider_obj)
    if provider not in _SHAPED_PROVIDERS:
        return None
    if level == QUIET:
        return quiet_plan(provider, model, base_url, max_tokens=max_tokens)
    return plan_for(provider, model, base_url, level, max_tokens=max_tokens)
