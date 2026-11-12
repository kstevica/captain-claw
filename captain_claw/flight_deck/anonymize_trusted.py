"""The trusted model behind anonymization's LLM lane — chosen by an admin.

"Trusted" means the admin trusts it with the raw document, not "the agent's
own model": an agent on a cloud model must not anonymize by sending the very
document to that cloud. So this is deliberately apart from tiers and from the
rule that work for an agent runs on that agent's model.

* **Local** models are trusted as they are: ``ollama`` / ``litert`` on this
  machine or a private network (not an Ollama ``-cloud`` model, which runs at
  Ollama), or an OpenAI-compatible server (vLLM, LM Studio, llama.cpp …) at a
  private address.
* Anything else is trusted only when the admin **attests** it (zero
  retention, a data-processing agreement, a region they accept).

Where it's set: ``system_settings["anonymize:trusted_model"]`` (the API key
sealed with :mod:`fd_crypto`), else the deck's environment —
``FD_ANONYMIZE_PROVIDER``, ``FD_ANONYMIZE_MODEL``, ``FD_ANONYMIZE_BASE_URL``,
``FD_ANONYMIZE_API_KEY``, ``FD_ANONYMIZE_ATTESTED=1``,
``FD_ANONYMIZE_ON_FAILURE`` — so a self-hosted deck can set it without a page.

The model is called here, not through ``llm_routes``: its errors are logged by
type only, since a provider's error can quote the request — the document.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any

from captain_claw.flight_deck import fd_crypto
from captain_claw.logging import get_logger

log = get_logger(__name__)

SETTING_KEY = "anonymize:trusted_model"
PROVIDERS = ("ollama", "litert", "openai", "anthropic", "gemini", "xai", "openrouter")
LOCAL_PROVIDERS = ("ollama", "litert")
ON_FAILURE = ("refuse", "deterministic")
# One call to the trusted model (one chunk of ~6,000 characters). A local
# model that thinks can write 10k+ tokens for a chunk: minutes, not seconds.
# FD_ANONYMIZE_MODEL_TIMEOUT (seconds) changes it.
CALL_TIMEOUT = float(os.environ.get("FD_ANONYMIZE_MODEL_TIMEOUT") or 600.0)
# Room for the answer and for thinking: models that think by default
# (Claude Haiku 5.5, local thinking models) count it against max_tokens. An
# answer cut at the limit is asked again in smaller pieces (llm_lane.AnswerCut).
# FD_ANONYMIZE_MODEL_MAX_TOKENS changes it.
MAX_TOKENS = int(os.environ.get("FD_ANONYMIZE_MODEL_MAX_TOKENS") or 32000)
# Thinking on the trusted model's calls: off by default — finding names in a
# text needs none of it, and a local model that thought without end ran to
# the answer limit twice on a ~1,000-token chunk (16k, then 32k tokens).
# FD_ANONYMIZE_MODEL_THINK: "off" (default), "low"/"medium"/"high", "on", or
# "model" (the model's own). It goes through the reasoning-effort plans
# (llm/reasoning.py) — ``chat_template_kwargs.enable_thinking`` for a Qwen on
# a local OpenAI-compatible server, ``thinking`` for Claude, ``reasoning_effort``
# for OpenAI — and as Ollama's own ``think`` for a model those don't know.
_THINK = {"off": False, "false": False, "0": False, "no": False, "on": True, "true": True, "1": True,
          "yes": True, "low": "low", "medium": "medium", "high": "high", "model": None}


def model_think() -> bool | str | None:
    raw = os.environ.get("FD_ANONYMIZE_MODEL_THINK", "").strip().lower()
    return _THINK.get(raw, False) if raw else False


class TrustError(ValueError):
    """A trusted-model setting that can't be saved."""


@dataclass
class TrustedModel:
    provider: str
    model: str
    base_url: str = ""
    api_key: str = ""
    attested: bool = False
    on_failure: str = "refuse"
    enabled: bool = True
    source: str = "settings"   # "settings" | "env"
    basis: str = ""            # "local" | "attested" | "" — not trusted (yet)
    updated_by: str = ""
    updated_at: float = 0.0

    @property
    def label(self) -> str:
        return f"{self.provider}/{self.model}"

    @property
    def usable(self) -> bool:
        return self.enabled and bool(self.basis)

    def public(self) -> dict[str, Any]:
        """For the admin page and status: never the key."""
        return {
            "provider": self.provider, "model": self.model, "base_url": self.base_url,
            "has_key": bool(self.api_key), "attested": self.attested, "on_failure": self.on_failure,
            "enabled": self.enabled, "source": self.source, "basis": self.basis, "label": self.label,
            "usable": self.usable, "updated_by": self.updated_by, "updated_at": self.updated_at,
        }


def is_local(provider: str, model: str, base_url: str) -> bool:
    """Whether the model runs on this machine or a private network (resolves
    DNS — call it off the event loop)."""
    from captain_claw import deck_confine

    provider = provider.strip().lower()
    base_url = base_url.strip()
    if provider in LOCAL_PROVIDERS:
        if re.search(r"[-:]cloud$", model.strip().lower()):
            return False  # Ollama runs "-cloud" models on its own servers
        return not base_url or not deck_confine.url_public(base_url)
    if provider == "openai":
        return bool(base_url) and not deck_confine.url_public(base_url)
    return False


async def _with_basis(m: TrustedModel) -> TrustedModel:
    local = await asyncio.to_thread(is_local, m.provider, m.model, m.base_url)
    m.basis = "local" if local else ("attested" if m.attested else "")
    return m


def _from_env() -> TrustedModel | None:
    provider = (os.environ.get("FD_ANONYMIZE_PROVIDER") or "").strip().lower()
    model = (os.environ.get("FD_ANONYMIZE_MODEL") or "").strip()
    if not provider or not model or provider not in PROVIDERS:
        return None
    on_failure = (os.environ.get("FD_ANONYMIZE_ON_FAILURE") or "refuse").strip().lower()
    return TrustedModel(
        provider=provider, model=model, base_url=(os.environ.get("FD_ANONYMIZE_BASE_URL") or "").strip(),
        api_key=(os.environ.get("FD_ANONYMIZE_API_KEY") or "").strip(),
        attested=(os.environ.get("FD_ANONYMIZE_ATTESTED") or "").strip().lower() in ("1", "true", "yes"),
        on_failure=on_failure if on_failure in ON_FAILURE else "refuse", source="env",
    )


async def configured(db) -> TrustedModel | None:
    """The admin's setting, else the environment's; None when neither."""
    raw = await db.get_system_setting(SETTING_KEY)
    m: TrustedModel | None = None
    if raw:
        try:
            data = json.loads(raw)
        except ValueError:
            data = None
        if isinstance(data, dict) and data.get("provider") and data.get("model"):
            try:
                key = fd_crypto.unseal(data.get("api_key_sealed") or "") or ""
            except fd_crypto.CryptoUnavailable:
                key = ""
            m = TrustedModel(
                provider=str(data["provider"]), model=str(data["model"]), base_url=str(data.get("base_url") or ""),
                api_key=str(key), attested=bool(data.get("attested")),
                on_failure=str(data.get("on_failure") or "refuse"), enabled=bool(data.get("enabled", True)),
                updated_by=str(data.get("updated_by") or ""), updated_at=float(data.get("updated_at") or 0),
            )
    if m is None:
        m = _from_env()
    return await _with_basis(m) if m is not None else None


async def usable(db) -> TrustedModel | None:
    m = await configured(db)
    return m if m is not None and m.usable else None


async def save(db, data: dict[str, Any], *, admin: str) -> TrustedModel:
    provider = str(data.get("provider") or "").strip().lower()
    model = str(data.get("model") or "").strip()
    base_url = str(data.get("base_url") or "").strip()
    if provider not in PROVIDERS:
        raise TrustError(f"provider is one of: {', '.join(PROVIDERS)}")
    if not model or len(model) > 200:
        raise TrustError("model is required (at most 200 characters)")
    if base_url and (len(base_url) > 500 or not re.match(r"^https?://[^\s]+$", base_url)):
        raise TrustError("base_url is an http(s) address")
    on_failure = str(data.get("on_failure") or "refuse")
    if on_failure not in ON_FAILURE:
        raise TrustError("on_failure is 'refuse' or 'deterministic'")
    previous_raw = await db.get_system_setting(SETTING_KEY)
    previous = {}
    if previous_raw:
        try:
            previous = json.loads(previous_raw) or {}
        except ValueError:
            previous = {}
    if "api_key" in data:
        key = str(data.get("api_key") or "").strip()
        if len(key) > 500:
            raise TrustError("api_key is at most 500 characters")
        sealed = fd_crypto.seal(key) if key else ""
    else:
        same_target = previous.get("provider") == provider and (previous.get("base_url") or "") == base_url
        sealed = str(previous.get("api_key_sealed") or "") if same_target else ""
    attested = bool(data.get("attested"))
    local = await asyncio.to_thread(is_local, provider, model, base_url)
    if not local and not attested:
        raise TrustError(
            f"{provider}/{model} doesn't run on this machine or a private network, so it would receive the raw "
            "documents. Set attested=true only if you trust that provider with them (zero retention, a "
            "data-processing agreement, a region you accept); otherwise use a local model.")
    record = {
        "provider": provider, "model": model, "base_url": base_url, "api_key_sealed": sealed,
        "attested": attested, "on_failure": on_failure, "enabled": bool(data.get("enabled", True)),
        "updated_by": admin, "updated_at": time.time(),
    }
    await db.set_system_setting(SETTING_KEY, json.dumps(record))
    m = await configured(db)
    assert m is not None
    return m


async def clear(db) -> None:
    await db.set_system_setting(SETTING_KEY, "")


def _tokens(usage: Any) -> int:
    if not isinstance(usage, dict):
        return 0
    total = usage.get("total_tokens")
    if total:
        return int(total)
    return int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0) + int(
        usage.get("completion_tokens") or usage.get("output_tokens") or 0)


def completer(m: TrustedModel):
    """``complete(system, user) -> (answer, tokens)`` on the trusted model."""
    from captain_claw.llm import Message, create_provider

    async def ask(system: str, user: str, think: bool | str | None) -> Any:
        from captain_claw.llm import reasoning

        level = "off" if think is False else ("on" if think is True else think)  # None: the model's own
        try:
            provider = create_provider(
                provider=m.provider, model=m.model, api_key=m.api_key or None, base_url=m.base_url or None,
                temperature=0.0, max_tokens=MAX_TOKENS, think=think)
            # The provider's own per-request timeout (Ollama 120 s, LiteLLM
            # 180 s) would cut a long answer short of the cap below.
            provider.request_timeout = CALL_TIMEOUT
            provider.log_content = False  # its answer is the private data it found
            with reasoning.call_effort(level):  # the provider shapes it for this model and server
                return await asyncio.wait_for(
                    provider.complete(messages=[Message(role="system", content=system),
                                                Message(role="user", content=user)],
                                      temperature=0.0, max_tokens=MAX_TOKENS),
                    CALL_TIMEOUT)
        except Exception as exc:  # noqa: BLE001 — the error may quote the document: type only
            log.warning("Anonymization trusted model call failed", model=m.label, error=type(exc).__name__)
            raise RuntimeError(type(exc).__name__) from None

    def cut(resp: Any) -> bool:
        return str(getattr(resp, "finish_reason", "") or "").lower() in ("length", "max_tokens")

    async def call(system: str, user: str) -> tuple[str, int]:
        from captain_claw.anonymize.llm_lane import AnswerCut

        think = model_think()
        resp = await ask(system, user, think)
        if cut(resp) and think is not False:
            # What runs on is thinking: the same piece, without it.
            log.info("Anonymization trusted model hit its answer limit; asking again without thinking",
                     model=m.label, limit=MAX_TOKENS)
            resp = await ask(system, user, False)
        if cut(resp):
            log.info("Anonymization trusted model hit its answer limit", model=m.label, limit=MAX_TOKENS)
            raise AnswerCut(MAX_TOKENS)
        return str(getattr(resp, "content", "") or ""), _tokens(getattr(resp, "usage", None))

    return call
