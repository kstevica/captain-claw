"""LLM providers (Ollama + LiteLLM-backed providers)."""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import tempfile
import time
import uuid
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable

import httpx

from captain_claw.exceptions import LLMAPIError, LLMError
from captain_claw.logging import get_logger

log = get_logger(__name__)


OLLAMA_NATIVE_BASE_URL = "http://127.0.0.1:11434"


@dataclass
class Message:
    """A message in the conversation."""

    role: str  # "system", "user", "assistant", "tool"
    content: str
    tool_call_id: str | None = None
    tool_name: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    # Provider-side chain-of-thought returned with assistant messages
    # by thinking-mode models (DeepSeek's ``reasoning_content``,
    # Anthropic's ``thinking``, etc.). Stored on the assistant
    # message so it can be echoed back on the next turn — DeepSeek
    # strictly requires this round-trip ("The reasoning_content in
    # the thinking mode must be passed back to the API"). Ignored
    # by providers that don't recognize it.
    reasoning_content: str | None = None


@dataclass
class ToolCall:
    """A tool call from the LLM."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMResponse:
    """Response from the LLM."""

    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    model: str = ""
    usage: dict[str, int] = field(default_factory=dict)
    finish_reason: str = ""
    # Thinking-mode chain-of-thought as returned by the provider.
    # Empty string when the provider didn't emit one. Propagated so
    # the orchestration layer can persist it onto the assistant
    # message and replay it on subsequent turns. See ``Message``.
    reasoning_content: str = ""


@dataclass
class ToolDefinition:
    """Definition of a tool for the LLM."""

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema


class TokenRateLimiter:
    """Sliding-window token rate limiter (tokens per minute).

    Tracks token consumption over a rolling 60-second window and blocks
    callers via ``acquire()`` until capacity is available.  After the API
    call completes, ``record_actual()`` corrects the estimate with the
    real usage reported by the provider.
    """

    def __init__(self, tokens_per_minute: int) -> None:
        self.tpm = max(0, tokens_per_minute)
        self._lock = asyncio.Lock()
        # Each entry: (monotonic_timestamp, token_count)
        self._log: deque[tuple[float, int]] = deque()

    @property
    def enabled(self) -> bool:
        return self.tpm > 0

    def _purge_old(self, now: float) -> int:
        """Remove entries older than 60 s and return current window total."""
        cutoff = now - 60.0
        while self._log and self._log[0][0] < cutoff:
            self._log.popleft()
        return sum(t for _, t in self._log)

    async def acquire(self, estimated_tokens: int) -> None:
        """Wait until the window has room for *estimated_tokens*."""
        if not self.enabled:
            return
        # A single request larger than the full per-minute budget can
        # never fit — waiting would loop forever. Record it and pass
        # through; the provider will reject it if it's truly over.
        if estimated_tokens > self.tpm:
            async with self._lock:
                self._log.append((time.monotonic(), estimated_tokens))
            log.warning(
                "rate-limiter: request %d tokens exceeds %d TPM budget; passing through",
                estimated_tokens,
                self.tpm,
            )
            return
        last_log = 0.0
        while True:
            async with self._lock:
                now = time.monotonic()
                used = self._purge_old(now)
                if used + estimated_tokens <= self.tpm:
                    self._log.append((now, estimated_tokens))
                    return
                # Calculate how long until enough capacity frees up.
                deficit = (used + estimated_tokens) - self.tpm
                wait = 0.0
                freed = 0
                for ts, tok in self._log:
                    freed += tok
                    if freed >= deficit:
                        wait = (ts + 60.0) - now
                        break
            wait = max(0.5, wait)
            if now - last_log >= 10.0:
                log.info(
                    "rate-limiter: waiting %.1fs (used %d/%d TPM)",
                    wait,
                    used,
                    self.tpm,
                )
                last_log = now
            await asyncio.sleep(wait)

    def record_actual(self, actual_tokens: int, estimated_tokens: int) -> None:
        """Correct the most-recent log entry with real usage."""
        diff = actual_tokens - estimated_tokens
        if diff == 0 or not self._log:
            return
        ts, tok = self._log[-1]
        self._log[-1] = (ts, max(0, tok + diff))


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    rate_limiter: TokenRateLimiter | None = None

    @abstractmethod
    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        pass

    @abstractmethod
    async def complete_streaming(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        pass

    async def complete_with_callback(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        on_chunk: Callable[[str], None] | None = None,
    ) -> LLMResponse:
        """Complete with optional streaming callback.

        Streams text chunks to *on_chunk* in real time while still returning
        the full ``LLMResponse`` (including tool_calls and usage).
        Default implementation delegates to ``complete()`` and sends the
        full content as one chunk.  Subclasses may override with true
        token-level streaming.
        """
        response = await self.complete(messages, tools, temperature, max_tokens)
        if on_chunk and response.content:
            try:
                on_chunk(response.content)
            except Exception:
                pass
        return response

    async def complete_structured(
        self,
        messages: list[Message],
        response_schema: dict[str, Any],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Completion constrained to a JSON schema, where the backend supports it.

        Providers with native constrained decoding (Ollama ``format``,
        browser xgrammar) override this. The default falls back to a plain
        completion — safe because callers must describe the expected JSON
        shape in the prompt anyway (grammar never injects the schema into
        the prompt), and callers validate + retry regardless.
        """
        return await self.complete(messages, None, temperature, max_tokens)

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        pass

    def _estimate_request_tokens(
        self,
        messages: list[Message],
        max_tokens: int | None = None,
    ) -> int:
        """Rough estimate of total tokens a request will consume."""
        prompt_tokens = 0
        for msg in messages:
            text = msg.content if isinstance(msg, Message) else str(msg.get("content", ""))
            prompt_tokens += self.count_tokens(text) if text else 0
        # Add a conservative estimate for the completion side.
        completion_budget = max_tokens or 4096
        return prompt_tokens + completion_budget


def _obj_get(obj: Any, key: str, default: Any = None) -> Any:
    """Read key from object or dict."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _safe_json_loads(raw: str) -> dict[str, Any]:
    """Parse JSON dict safely; return empty dict on errors."""
    text = (raw or "").strip()
    if not text:
        return {}
    try:
        value = json.loads(text)
        if isinstance(value, dict):
            return value
        return {"value": value}
    except Exception:
        return {"raw": raw}


def _is_temperature_rejected_error(msg: str) -> bool:
    """Whether *msg* (lowercased) is a 400 rejecting the ``temperature`` value.

    All of these are recovered the same way — drop ``temperature`` and let the
    model fall back to its own default:

    * Anthropic's "``temperature`` is deprecated for this model".
    * Generic "temperature is not supported / unsupported".
    * OpenAI-compatible endpoints that pin temperature to 1 and reject anything
      else — e.g. kimi-k3 served over an OpenAI base path returns
      "invalid temperature: only 1 is allowed for this model", and OpenAI's own
      o-series returns "... does not support 0.7 ... Only the default (1) value
      is supported".

    Every branch also requires the word "temperature", staying narrow enough
    not to swallow unrelated 400s that merely mention it.
    """
    if "temperature" not in msg:
        return False
    return (
        "deprecat" in msg
        or "not support" in msg
        or "unsupported" in msg
        or "invalid temperature" in msg
        or "only 1 is allowed" in msg
        or "only the default" in msg
    )


# Minimal, non-empty stand-in used to satisfy thinking-mode servers that
# demand a reasoning_content on assistant messages we can't supply a real one
# for (orphan tool results normalized into assistant turns, synthesized
# context/nudge messages). Only ever sent on the self-heal retry below, and
# only to a server that already rejected the request for the missing field.
_REASONING_BACKFILL_PLACEHOLDER = "(prior reasoning unavailable)"


def is_reasoning_backfill_placeholder(value: Any) -> bool:
    """True if *value* is the internal reasoning-backfill sentinel.

    That sentinel is plumbing injected to satisfy strict thinking-mode servers;
    it must never be persisted onto a turn or shown to a user. Callers that stash
    a model's ``reasoning_content`` use this to drop it.
    """
    return isinstance(value, str) and value.strip() == _REASONING_BACKFILL_PLACEHOLDER


def _is_reasoning_content_required_error(msg: str) -> bool:
    """True for the thinking-mode 400 that demands reasoning_content be echoed
    back on assistant messages — e.g. DeepSeek V4 thinking served via an
    OpenAI-compatible endpoint: "The reasoning_content in the thinking mode
    must be passed back to the API"."""
    return "reasoning_content" in msg and (
        "passed back" in msg or "thinking mode" in msg
    )


def _backfill_reasoning_content(messages: Any) -> bool:
    """Ensure every assistant message carries a non-empty reasoning_content,
    injecting :data:`_REASONING_BACKFILL_PLACEHOLDER` where one is missing.

    Returns True if any message was patched. Mutates the list in place with
    shallow-copied dicts so the caller's originals are untouched.
    """
    if not isinstance(messages, list):
        return False
    patched = False
    for i, m in enumerate(messages):
        if not isinstance(m, dict) or m.get("role") != "assistant":
            continue
        rc = m.get("reasoning_content")
        if not (isinstance(rc, str) and rc.strip()):
            m = dict(m)
            m["reasoning_content"] = _REASONING_BACKFILL_PLACEHOLDER
            messages[i] = m
            patched = True
    return patched


async def _acompletion_tolerant(kwargs: dict[str, Any], provider: Any = None) -> Any:
    """Call ``acompletion``; on a parameter-rejection 400, strip the offending
    parameter and retry once.

    Two model-specific request-shape rejections are handled transparently so a
    turn never crashes on a parameter the model simply won't accept:

    * **tool_choice** — thinking/reasoning models (e.g. DeepSeek thinking mode
      served via an OpenAI-compatible endpoint) return HTTP 400 "does not
      support this tool_choice". The orchestration layer forces
      ``tool_choice="required"`` on stall retries — rather than crash the turn,
      we drop the constraint and let the model answer normally. We also flag the
      provider so subsequent calls this session skip forcing tool_choice.

    * **temperature** — some models reject the temperature VALUE with a 400:
      newer Anthropic models (Opus 4.8, Sonnet 5, the Fable family) deprecate it
      outright, and OpenAI-compatible endpoints may pin it to 1 (e.g. kimi-k3:
      "only 1 is allowed for this model"). We retry without it — the model then
      uses its own default — and remember it globally so later calls omit the
      parameter up front.

    Each offending parameter costs at most one wasted call.
    """
    from litellm import acompletion

    try:
        return await acompletion(**kwargs)
    except Exception as e:
        msg = str(e).lower()
        retry_kwargs = dict(kwargs)
        stripped: list[str] = []

        # Model rejects tool_choice (thinking-mode models).
        if (
            kwargs.get("tool_choice") is not None
            and "tool_choice" in msg
            and "support" in msg
        ):
            retry_kwargs.pop("tool_choice", None)
            stripped.append("tool_choice")
            if provider is not None:
                try:
                    provider._tool_choice_unsupported = True
                except Exception:
                    pass

        # Model rejects the temperature VALUE — either deprecated outright
        # (Anthropic Opus 4.8 / Sonnet 5 / Fable) or pinned to 1 by an
        # OpenAI-compatible endpoint (e.g. kimi-k3: "only 1 is allowed for this
        # model"). Dropping it lets the model use its default; remembering the
        # model omits it up front on every later call.
        if kwargs.get("temperature") is not None and _is_temperature_rejected_error(msg):
            retry_kwargs.pop("temperature", None)
            stripped.append("temperature")
            _remember_temperature_unsupported(kwargs.get("model", ""))

        # Thinking-mode server demands reasoning_content round-tripped on
        # assistant messages (DeepSeek V4 thinking via an OpenAI-compatible
        # endpoint). An assistant message can reach the payload without it —
        # an orphan tool result normalized into an assistant turn, or a
        # synthesized context/nudge message. Backfill a placeholder on any
        # assistant message that lacks reasoning and retry once. Scoped to the
        # exact 400 so no other provider ever sees the placeholder.
        reasoning_backfilled = False
        if _is_reasoning_content_required_error(msg):
            reasoning_backfilled = _backfill_reasoning_content(
                retry_kwargs.get("messages")
            )

        if stripped or reasoning_backfilled:
            log.warning(
                "Model rejected request; retrying with fix",
                model=kwargs.get("model"),
                stripped=",".join(stripped) or None,
                reasoning_backfilled=reasoning_backfilled,
            )
            return await acompletion(**retry_kwargs)
        raise


# Matches <think>...</think> blocks (and common variants like <thinking>,
# <reasoning>) emitted by some models as raw text in the content field
# instead of in a separate reasoning_content field. We strip these so the
# user-visible chat content never contains the model's private chain of
# thought.
_THINK_BLOCK_RE = re.compile(
    r"<\s*(think|thinking|reasoning|reflection)\s*>.*?<\s*/\s*\1\s*>",
    re.IGNORECASE | re.DOTALL,
)
# Matches an unterminated leading think/reasoning block (model started a
# <think> tag but never closed it before the answer began).
_THINK_OPEN_RE = re.compile(
    r"^\s*<\s*(think|thinking|reasoning|reflection)\s*>.*?(?=<\s*/\s*\1\s*>|$)",
    re.IGNORECASE | re.DOTALL,
)


def _strip_reasoning_artifacts(text: str, *, recover_answer: bool = True) -> str:
    """Remove <think>...</think> style reasoning blocks from model output.

    Some providers (notably xAI Grok and certain DeepSeek/Qwen variants)
    leak their internal reasoning into the ``content`` field rather than
    keeping it in a separate ``reasoning_content`` field. This helper
    strips those blocks so they don't end up in the chat UI.

    Edge case: small thinking models sometimes emit ONLY a thinking block
    with no answer afterwards. If stripping produces empty output but the
    raw text had content, fall back to the last paragraph inside the
    thinking block — that's usually where the model's conclusion lives.

    ``recover_answer=False`` disables that fallback — pass it when the turn
    also carries tool calls, where an empty content is correct (the "answer"
    is the tool call) and surfacing the reasoning as content would both leak
    the private chain-of-thought into the chat and duplicate what we already
    keep as ``reasoning_content``.
    """
    if not text:
        return text
    cleaned = _THINK_BLOCK_RE.sub("", text)
    cleaned = _THINK_OPEN_RE.sub("", cleaned)
    # Orphan CLOSING tag: some chat templates (Ollama with glm/kimi thinking
    # variants) consume the opening <think> themselves, so the content
    # arrives as "…leaked reasoning…</think>the real answer". Everything
    # before the last closing tag is reasoning — keep only what follows.
    _orphan = re.search(
        r"<\s*/\s*(?:think|thinking|reasoning|reflection)\s*>(?!.*<\s*/\s*(?:think|thinking|reasoning|reflection)\s*>)",
        cleaned, flags=re.IGNORECASE | re.DOTALL)
    if _orphan:
        _after = cleaned[_orphan.end():].lstrip()
        if _after:
            cleaned = _after
        else:
            # Tag at the very end — the "answer" is the pre-tag text.
            cleaned = cleaned[:_orphan.start()]
    cleaned = cleaned.lstrip()
    if cleaned or not recover_answer:
        return cleaned
    inner = _extract_thinking_inner(text)
    if not inner:
        return cleaned
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", inner) if p.strip()]
    if paragraphs:
        log.warning(
            "LLM returned only reasoning — surfacing last paragraph",
            raw_chars=len(text),
            inner_chars=len(inner),
        )
        return paragraphs[-1]
    return inner.strip()


def _extract_thinking_inner(text: str) -> str:
    """Extract the inner text of <think>/<thinking>/<reasoning> blocks."""
    matches = re.findall(
        r"<\s*(?:think|thinking|reasoning|reflection)\s*>(.*?)(?:<\s*/\s*(?:think|thinking|reasoning|reflection)\s*>|$)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return "\n\n".join(m.strip() for m in matches if m.strip())


def _recover_inline_reasoning(text: str) -> str:
    """Recover chain-of-thought a model emitted INLINE in its ``content``
    (as a ``<think>…</think>`` block) rather than in a separate
    ``reasoning_content`` field, returning the reasoning inner text (no tags)
    or ``""`` when there is none.

    This exists so thinking-mode servers that stream reasoning inline but
    require it echoed back as ``reasoning_content`` on the next turn can be
    satisfied. Notably NVIDIA Nemotron served via an OpenAI-compatible
    MLX/vLLM endpoint 400s with "The reasoning_content in the thinking mode
    must be passed back to the API" when the round-tripped assistant message
    omits it. We strip the ``<think>`` block from the user-visible content
    elsewhere; this keeps a copy so :func:`_convert_messages_for_openai_style`
    can replay it.

    Handles both the paired form (``<think>…</think>answer``) and the
    orphan-closing-tag form (``…leaked reasoning…</think>answer``) that some
    chat templates produce when they consume the opening ``<think>``
    themselves — matching :func:`_strip_reasoning_artifacts`.
    """
    if not text:
        return ""
    inner = _extract_thinking_inner(text)
    if inner:
        return inner
    # Orphan closing tag with no opener: everything before the last closing
    # tag is the reasoning.
    orphan = re.search(
        r"<\s*/\s*(?:think|thinking|reasoning|reflection)\s*>"
        r"(?!.*<\s*/\s*(?:think|thinking|reasoning|reflection)\s*>)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if orphan:
        return text[: orphan.start()].strip()
    return ""


def _extract_json_blob(text: str) -> str | None:
    """The last complete top-level JSON object/array in ``text``, or None.

    Prefers a ```json fenced block; otherwise scans for opening brackets and
    returns the last one that fully decodes (skipping brackets nested inside an
    already-captured value), so a trailing JSON answer wins over an inline
    example earlier in the text.
    """
    if not text:
        return None
    dec = json.JSONDecoder()
    for frag in reversed(re.findall(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)):
        frag = frag.strip()
        if frag:
            try:
                dec.raw_decode(frag)
                return frag
            except ValueError:
                pass
    best: str | None = None
    best_end = -1
    for m in re.finditer(r"[\[{]", text):
        i = m.start()
        if i < best_end:  # nested inside a value we already captured
            continue
        try:
            _obj, rel_end = dec.raw_decode(text[i:])
        except ValueError:
            continue
        end = i + rel_end
        if end > best_end:
            best, best_end = text[i:end], end
    return best


def _reasoning_content_fallback(reasoning: str) -> str:
    """Recover a usable answer from ``reasoning_content`` when a model returned
    empty ``content`` (some reasoning models put everything in the reasoning
    field). Prefer a JSON object/array — many internal callers ask for STRICT
    JSON (quality checks, judges, routers) and the model often emitted it inside
    its reasoning — else the last non-empty paragraph (the conclusion)."""
    rc = str(reasoning or "")
    # The reasoning-backfill sentinel is internal plumbing, never an answer. A
    # thinking model can echo the placeholder we injected into a prior turn's
    # reasoning_content straight back as its own — surfacing it as content would
    # print "(prior reasoning unavailable)" to the user. Treat it as empty so the
    # empty-answer floor re-prompts for a real reply instead.
    if rc.strip() == _REASONING_BACKFILL_PLACEHOLDER:
        return ""
    blob = _extract_json_blob(rc)
    if blob is not None:
        return blob
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", rc) if p.strip()]
    result = paragraphs[-1] if paragraphs else rc.strip()
    return "" if result == _REASONING_BACKFILL_PLACEHOLDER else result


def _normalize_provider_name(provider: str) -> str:
    """Normalize provider aliases."""
    key = (provider or "").strip().lower()
    aliases = {
        "chatgpt": "openai",
        "openai": "openai",
        "claude": "anthropic",
        "anthropic": "anthropic",
        # Anthropic *subscription* (Pro/Max) via the ``claude`` CLI, as
        # opposed to the pay-per-token ``anthropic`` API path above. Keep
        # these distinct from ``claude``/``anthropic`` so selecting the
        # subscription can never silently fall back to metered billing.
        "claude-cli": "claude-cli",
        "claude_cli": "claude-cli",
        "claude-code": "claude-cli",
        "claude-subscription": "claude-cli",
        "claude-sub": "claude-cli",
        "claude-max": "claude-cli",
        "anthropic-cli": "claude-cli",
        "anthropic-subscription": "claude-cli",
        "gemini": "gemini",
        "google": "gemini",
        "googleai": "gemini",
        "grok": "xai",
        "xai": "xai",
        "ollama": "ollama",
        "openrouter": "openrouter",
        "litert": "litert",
        "litert-lm": "litert",
        "litertlm": "litert",
        "gemma-local": "litert",
        # DeepSeek aliases — LiteLLM uses the ``deepseek/`` prefix.
        # Thinking-mode models (e.g. deepseek-reasoner) require us
        # to round-trip their ``reasoning_content`` field on each
        # turn; see :func:`_convert_messages_for_openai_style`.
        "deepseek": "deepseek",
        "deep-seek": "deepseek",
        "deepseek-ai": "deepseek",
    }
    return aliases.get(key, key)


def _provider_model_name(provider: str, model: str) -> str:
    """Ensure model name includes provider prefix for LiteLLM."""
    cleaned = (model or "").strip()
    if not cleaned:
        return cleaned
    # OpenRouter and OpenAI-compatible model IDs may contain a slash
    # (e.g. nvidia/model-name, mlx-community/NVIDIA-...), but LiteLLM
    # still needs the provider prefix to route correctly.
    if provider in ("openrouter", "openai"):
        if cleaned.startswith(f"{provider}/"):
            return cleaned
        return f"{provider}/{cleaned}"
    if "/" in cleaned:
        return cleaned
    return f"{provider}/{cleaned}"


def _base_model_name(model: str) -> str:
    """Return provider-agnostic model name (without provider prefix)."""
    cleaned = (model or "").strip()
    if "/" in cleaned:
        return cleaned.split("/", 1)[1]
    return cleaned


def _is_openai_gpt5_family(provider: str, model: str) -> bool:
    """Whether model is in OpenAI GPT-5 family."""
    normalized_provider = _normalize_provider_name(provider)
    if normalized_provider != "openai":
        return False
    base = _base_model_name(model).lower()
    return base.startswith("gpt-5")


# Model base-names discovered at runtime to reject the ``temperature`` value.
# Populated by _acompletion_tolerant when a request 400s because temperature is
# deprecated, unsupported, or pinned to 1; consulted by
# _is_temperature_unsupported_model so every subsequent call (even from a
# freshly-built provider instance) omits the parameter instead of paying another
# failed round-trip. Covers newer Anthropic models (Opus 4.8, Sonnet 5, …) that
# deprecated temperature like the Fable family, and OpenAI-compatible endpoints
# that only accept temperature=1 (e.g. kimi-k3), for which omitting it — and so
# letting the model use its default of 1 — is exactly right.
_TEMPERATURE_UNSUPPORTED_MODELS: set[str] = set()


def _remember_temperature_unsupported(model: str) -> None:
    """Record that *model* rejects ``temperature`` so we stop sending it."""
    base = str(model or "").split("/")[-1].lower()
    if base:
        _TEMPERATURE_UNSUPPORTED_MODELS.add(base)


def _is_temperature_unsupported_model(provider: str, model: str) -> bool:
    """True for models that reject the temperature parameter entirely.

    Anthropic's Fable family deprecated temperature — sending it (even
    temperature=1) returns a 400 ``temperature is deprecated for this
    model``, so it must be omitted from the request, not just clamped.
    Models discovered at runtime to reject it (see
    :data:`_TEMPERATURE_UNSUPPORTED_MODELS`) are treated the same way.
    """
    base = str(model or "").split("/")[-1].lower()
    if "fable" in base:
        return True
    return base in _TEMPERATURE_UNSUPPORTED_MODELS


def _normalize_temperature_for_model(provider: str, model: str, temperature: float | None) -> float | None:
    """Adjust temperature for provider/model constraints.

    Returns None when the parameter must be omitted entirely (callers that
    build request payloads must drop the key when this is None).
    """
    # Models that reject temperature outright — omit it regardless of input.
    if _is_temperature_unsupported_model(provider, model):
        return None
    if temperature is None:
        return None
    # OpenAI GPT-5 family only accepts temperature=1.
    if _is_openai_gpt5_family(provider, model):
        return 1.0
    return temperature


def _resolve_api_key(provider: str, explicit_api_key: str | None) -> str | None:
    """Resolve provider API key from explicit value, environment, or provider_keys."""
    if explicit_api_key:
        return explicit_api_key
    if provider == "openai":
        val = os.getenv("OPENAI_API_KEY") or None
        if val:
            return val
    elif provider == "anthropic":
        val = os.getenv("ANTHROPIC_API_KEY") or None
        if val:
            return val
    elif provider == "gemini":
        val = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or None
        if val:
            return val
    elif provider == "xai":
        val = os.getenv("XAI_API_KEY") or None
        if val:
            return val
    elif provider == "openrouter":
        val = os.getenv("OPENROUTER_API_KEY") or None
        if val:
            return val
    # Fallback: provider_keys from config (settings UI).
    try:
        from captain_claw.config import get_config
        pk = get_config().provider_keys
        pk_map = {"openai": pk.openai, "anthropic": pk.anthropic, "gemini": pk.gemini, "xai": pk.xai, "openrouter": pk.openrouter}
        pk_val = str(pk_map.get(provider, "") or "").strip()
        if pk_val:
            return pk_val
    except Exception:
        pass
    return None


_CACHE_SPLIT_MARKER = "<!-- CACHE_SPLIT -->"


def _inject_anthropic_cache_control(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Add ``cache_control`` to system and history messages for Anthropic prompt caching.

    Anthropic caches the prompt prefix up to each ``cache_control`` breakpoint.
    We use up to 2 breakpoints (out of 4 allowed):

    1. **System message (static part)** — the large instruction text that is
       identical across turns.  The system prompt template contains a
       ``<!-- CACHE_SPLIT -->`` marker that separates static instructions
       (above) from dynamic context (below, e.g. timestamp, file trees).
       Only the static block gets ``cache_control``; the dynamic block is
       sent as a separate content block without it so that changes to
       timestamps/env info don't bust the cache.

    2. **Last user or assistant message before the current turn** — during a
       tool-use loop the same conversation prefix is sent multiple times.
       Marking the last historical message lets Anthropic cache the entire
       prefix (system + history) across tool-loop iterations within a turn.

    LiteLLM passes ``cache_control`` through to Anthropic when the message
    content is a list of content blocks (not a plain string).
    """
    result: list[dict[str, Any]] = []
    for msg in messages:
        if msg.get("role") == "system":
            content = msg.get("content", "")
            if isinstance(content, str) and content:
                if _CACHE_SPLIT_MARKER in content:
                    # Split into static (cached) and dynamic (uncached) blocks.
                    static_part, dynamic_part = content.split(_CACHE_SPLIT_MARKER, 1)
                    static_part = static_part.rstrip()
                    dynamic_part = dynamic_part.strip()
                    blocks: list[dict[str, Any]] = [
                        {
                            "type": "text",
                            "text": static_part,
                            "cache_control": {"type": "ephemeral"},
                        },
                    ]
                    if dynamic_part:
                        blocks.append({"type": "text", "text": dynamic_part})
                    msg = {**msg, "content": blocks}
                else:
                    # No marker — cache the whole thing.
                    msg = {**msg, "content": [
                        {
                            "type": "text",
                            "text": content,
                            "cache_control": {"type": "ephemeral"},
                        },
                    ]}
            elif isinstance(content, list):
                blocks = [dict(b) if isinstance(b, dict) else b for b in content]
                if blocks and isinstance(blocks[-1], dict):
                    blocks[-1] = {**blocks[-1], "cache_control": {"type": "ephemeral"}}
                msg = {**msg, "content": blocks}
        result.append(msg)

    # Breakpoint 2: mark the last user or assistant message in the
    # conversation history so tool-loop iterations cache the prefix.
    # Walk backwards to find the last user/assistant message.
    for i in range(len(result) - 1, -1, -1):
        role = result[i].get("role", "")
        if role in ("user", "assistant"):
            content = result[i].get("content", "")
            if isinstance(content, str) and content:
                result[i] = {**result[i], "content": [
                    {
                        "type": "text",
                        "text": content,
                        "cache_control": {"type": "ephemeral"},
                    },
                ]}
            elif isinstance(content, list):
                blocks = [dict(b) if isinstance(b, dict) else b for b in content]
                if blocks and isinstance(blocks[-1], dict):
                    blocks[-1] = {**blocks[-1], "cache_control": {"type": "ephemeral"}}
                result[i] = {**result[i], "content": blocks}
            break

    return result


def _convert_messages_for_openai_style(messages: list[Message]) -> list[dict[str, Any]]:
    """Convert messages to OpenAI-style payload."""
    result: list[dict[str, Any]] = []
    for msg in messages:
        if isinstance(msg, dict):
            role = str(msg.get("role", ""))
            content = str(msg.get("content", ""))
            tool_call_id = msg.get("tool_call_id")
            tool_name = msg.get("tool_name")
            tool_calls = msg.get("tool_calls")
            reasoning_content = msg.get("reasoning_content")
        else:
            role = str(getattr(msg, "role", ""))
            content = str(getattr(msg, "content", ""))
            tool_call_id = getattr(msg, "tool_call_id", None)
            tool_name = getattr(msg, "tool_name", None)
            tool_calls = getattr(msg, "tool_calls", None)
            reasoning_content = getattr(msg, "reasoning_content", None)

        if role not in {"system", "user", "assistant", "tool"}:
            continue
        # Anthropic rejects empty text content blocks.
        if not content and role == "tool":
            content = "[empty tool response]"
        elif not content and role not in {"assistant"}:
            content = " "
        entry: dict[str, Any] = {"role": role, "content": content}
        # DeepSeek thinking mode strictly requires the original
        # ``reasoning_content`` to round-trip on the assistant
        # message — otherwise the API errors with "The
        # reasoning_content in the thinking mode must be passed
        # back to the API". Other providers (OpenAI, Anthropic via
        # LiteLLM, Gemini, etc.) ignore unknown fields, so it's
        # safe to always emit when we have one. Only echo non-empty
        # values to avoid sending a stray ``""`` that some strict
        # gateways might reject.
        # Never echo the backfill sentinel back to the model: a thinking model
        # can adopt it as its own reasoning and hand it back, which then leaks
        # into visible content. Suppressing it to absent is safe — the next 400
        # (if any) re-backfills it transiently. This breaks the echo loop.
        if (role == "assistant" and isinstance(reasoning_content, str)
                and reasoning_content
                and reasoning_content.strip() != _REASONING_BACKFILL_PLACEHOLDER):
            entry["reasoning_content"] = reasoning_content
        if role == "assistant" and isinstance(tool_calls, list) and tool_calls:
            normalized_calls: list[dict[str, Any]] = []
            for idx, raw in enumerate(tool_calls, start=1):
                if not isinstance(raw, dict):
                    continue
                call_id = str(_obj_get(raw, "id", "") or f"call_{idx}")
                call_type = str(_obj_get(raw, "type", "") or "function")
                function_obj = _obj_get(raw, "function", None)
                if isinstance(function_obj, dict):
                    fn_name = str(_obj_get(function_obj, "name", "") or "")
                    fn_args = _obj_get(function_obj, "arguments", "")
                else:
                    fn_name = str(_obj_get(raw, "name", "") or "")
                    fn_args = _obj_get(raw, "arguments", {})
                if not fn_name:
                    continue
                if isinstance(fn_args, str):
                    fn_args_text = fn_args
                elif isinstance(fn_args, dict):
                    fn_args_text = json.dumps(fn_args, ensure_ascii=True)
                else:
                    fn_args_text = "{}"
                normalized_calls.append({
                    "id": call_id,
                    "type": call_type,
                    "function": {
                        "name": fn_name,
                        "arguments": fn_args_text,
                    },
                })
            if normalized_calls:
                entry["tool_calls"] = normalized_calls
        if role == "tool":
            # tool_call_id is required by many providers (OpenRouter, OpenAI, etc.)
            entry["tool_call_id"] = tool_call_id or f"call_{len(result)}"
        if role == "tool" and tool_name:
            entry["name"] = tool_name
        result.append(entry)
    return result


# Matches the attachment marker the chat pipeline injects for images.
_ATTACHED_IMAGE_RE = re.compile(r"\[Attached image:\s*([^\]\n]+?)\s*\]")

# Internal-context block injected into the prompt (todos, fleet info, …) that
# small models sometimes echo verbatim into their reply. Strip it so it never
# reaches the user.
_INTERNAL_CTX_RE = re.compile(r"\[INTERNAL CONTEXT.*?\[END INTERNAL CONTEXT\]\s*", re.S | re.I)


def _strip_internal_context(text: str) -> str:
    """Remove any echoed internal-context block from a model reply."""
    if not text or "[INTERNAL CONTEXT" not in text:
        return text
    cleaned = _INTERNAL_CTX_RE.sub("", text)
    # A dangling, unterminated "[INTERNAL CONTEXT …" means the model echoed the
    # injected block — cut from it (if at the very start, this yields "" and the
    # caller's empty-content fallback surfaces the reasoning tail instead).
    idx = cleaned.find("[INTERNAL CONTEXT")
    if idx != -1:
        cleaned = cleaned[:idx]
    return cleaned.strip()


def _encode_ollama_image(path: str) -> str | None:
    """Read, resize, and base64-encode an image for Ollama's ``images`` array.

    Multimodal Ollama models (e.g. minimax-m3, llava, qwen-vl) accept images
    inline per-message. Resized first (longest edge ~1568px) to bound tokens.
    Returns the raw base64 string (no data: prefix), or None on failure.
    """
    import base64
    from pathlib import Path

    p = Path(path.strip())
    if not p.is_file():
        return None
    try:
        data = p.read_bytes()
        try:
            from captain_claw.tools.image_ocr import _maybe_resize_image
            data, _mime = _maybe_resize_image(data, 1568, 85)
        except Exception:
            pass  # resize is best-effort; send original if it fails
        return base64.b64encode(data).decode("ascii")
    except Exception:
        return None


def _convert_messages_for_ollama(messages: list[Message]) -> list[dict[str, Any]]:
    """Convert messages to Ollama API format.

    Attached images are sent INLINE to multimodal models via Ollama's
    per-message ``images`` array — but only on the most recent user message
    that references one, so historical images aren't re-encoded every call.
    """
    def _content_of(m: Any) -> str:
        return str(m.get("content", "")) if isinstance(m, dict) else str(getattr(m, "content", ""))

    def _role_of(m: Any) -> str:
        return str(m.get("role", "")) if isinstance(m, dict) else str(getattr(m, "role", ""))

    last_img_idx = -1
    for i, msg in enumerate(messages):
        if _role_of(msg) == "user" and _ATTACHED_IMAGE_RE.search(_content_of(msg)):
            last_img_idx = i

    result: list[dict[str, Any]] = []
    for i, msg in enumerate(messages):
        if isinstance(msg, dict):
            role = str(msg.get("role", ""))
            content = str(msg.get("content", ""))
            tool_name = msg.get("tool_name")
        else:
            role = str(getattr(msg, "role", ""))
            content = str(getattr(msg, "content", ""))
            tool_name = getattr(msg, "tool_name", None)

        if role == "tool":
            entry = {"role": "tool", "content": content}
            if tool_name:
                entry["tool_name"] = str(tool_name)
            result.append(entry)
            continue

        if role in {"system", "user", "assistant"}:
            entry = {"role": role, "content": content}
            if i == last_img_idx:
                # Cap to the last few markers so a reflection/insight prompt that
                # concatenates the whole history (many [Attached image:] markers)
                # doesn't re-encode dozens of images and blow up tokens/latency.
                paths = _ATTACHED_IMAGE_RE.findall(content)[-2:]
                images = [b64 for p in paths if (b64 := _encode_ollama_image(p))]
                if images:
                    entry["images"] = images
            result.append(entry)
    return result


def _convert_tools_for_openai_style(tools: list[ToolDefinition]) -> list[dict[str, Any]]:
    """Convert tool schema to OpenAI function-tool format."""
    result: list[dict[str, Any]] = []
    for tool in tools:
        if isinstance(tool, dict):
            name = tool.get("name")
            description = tool.get("description", "")
            parameters = tool.get("parameters", {})
        else:
            name = getattr(tool, "name", None)
            description = getattr(tool, "description", "") or ""
            parameters = getattr(tool, "parameters", None) or {}

        if not name:
            continue
        result.append({
            "type": "function",
            "function": {
                "name": str(name),
                "description": str(description),
                "parameters": parameters if isinstance(parameters, dict) else {},
            },
        })
    return result


def _extract_usage(usage_obj: Any) -> dict[str, int]:
    """Extract usage token counts from provider response usage object."""
    if usage_obj is None:
        return {
            "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
            "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0,
        }

    prompt_tokens = int(_obj_get(usage_obj, "prompt_tokens", 0) or 0)
    completion_tokens = int(_obj_get(usage_obj, "completion_tokens", 0) or 0)
    total_tokens = int(_obj_get(usage_obj, "total_tokens", 0) or 0)
    if total_tokens <= 0:
        total_tokens = prompt_tokens + completion_tokens

    # Anthropic cache tokens (passed through by LiteLLM): here cache_read is a
    # SEPARATE bucket from prompt_tokens (input = the non-cached portion).
    cache_creation = int(_obj_get(usage_obj, "cache_creation_input_tokens", 0) or 0)
    cache_read = int(_obj_get(usage_obj, "cache_read_input_tokens", 0) or 0)

    # OpenAI-compatible caching (OpenAI, and openai-served local models like the
    # beings' MLX server) reports cached prompt tokens under
    # ``prompt_tokens_details.cached_tokens`` — and there they are INCLUDED in
    # prompt_tokens. Normalise to the Anthropic shape (input = non-cached,
    # cache_read = cached) so pricing/costing counts them once, at the cache
    # rate, instead of missing the discount entirely.
    if not cache_read:
        details = _obj_get(usage_obj, "prompt_tokens_details", None)
        oa_cached = int(_obj_get(details, "cached_tokens", 0) or 0) if details else 0
        if oa_cached > 0:
            cache_read = oa_cached
            prompt_tokens = max(0, prompt_tokens - oa_cached)

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "cache_creation_input_tokens": cache_creation,
        "cache_read_input_tokens": cache_read,
    }


# ── ChatGPT Responses API helpers ─────────────────────────────────────────


# Models the ChatGPT "Sign in with ChatGPT" / Codex backend actually
# accepts. The endpoint at chatgpt.com/backend-api/codex/responses is
# *not* the same as api.openai.com/v1 — it only serves Codex-family
# models tied to a ChatGPT plan. Anything else either 400s or, worse,
# returns an empty body that looks like a successful but blank response.
_CODEX_BACKEND_SUPPORTED_MODELS: frozenset[str] = frozenset({
    "gpt-5",
    "gpt-5-codex",
    "gpt-5.1",
    "gpt-5.1-codex",
    "gpt-5.1-codex-mini",
    "gpt-5.1-codex-max",
    "gpt-5.2",
    "gpt-5.2-codex",
    "gpt-5.3-codex",
    "codex-mini-latest",
})


_REASONING_EFFORT_VALUES: tuple[str, ...] = ("minimal", "low", "medium", "high", "xhigh")


def _extract_reasoning_effort(name: str) -> tuple[str, str | None]:
    """Split an effort suffix off a model name.

    ``"gpt-5-high"`` → ``("gpt-5", "high")``;
    ``"gpt-5"`` → ``("gpt-5", None)``.
    """
    base = (name or "").strip()
    if not base:
        return base, None
    lowered = base.lower()
    for sep in ("-", "_"):
        for effort in _REASONING_EFFORT_VALUES:
            suffix = f"{sep}{effort}"
            if lowered.endswith(suffix):
                return base[: -len(suffix)], effort
    return base, None


def _normalize_chatgpt_model(name: str) -> str:
    """Normalize ChatGPT / Codex model name aliases."""
    base = (name or "").strip()
    if not base:
        return "gpt-5"
    # Strip effort suffixes (e.g. gpt-5-high → gpt-5).
    base, _ = _extract_reasoning_effort(base)
    if not base:
        return "gpt-5"
    mapping: dict[str, str] = {
        "gpt5": "gpt-5",
        "gpt-5-latest": "gpt-5",
        "gpt-5": "gpt-5",
        "gpt-5.1": "gpt-5.1",
        "gpt5.2": "gpt-5.2",
        "gpt-5.2": "gpt-5.2",
        "gpt-5.2-latest": "gpt-5.2",
        "gpt5.3-codex": "gpt-5.3-codex",
        "gpt-5.3-codex": "gpt-5.3-codex",
        "gpt-5.3-codex-latest": "gpt-5.3-codex",
        "gpt5.2-codex": "gpt-5.2-codex",
        "gpt-5.2-codex": "gpt-5.2-codex",
        "gpt-5.2-codex-latest": "gpt-5.2-codex",
        "gpt5-codex": "gpt-5-codex",
        "gpt-5-codex": "gpt-5-codex",
        "gpt-5-codex-latest": "gpt-5-codex",
        "gpt-5.1-codex": "gpt-5.1-codex",
        "gpt-5.1-codex-max": "gpt-5.1-codex-max",
        "codex": "codex-mini-latest",
        "codex-mini": "codex-mini-latest",
        "codex-mini-latest": "codex-mini-latest",
        "gpt-5.1-codex-mini": "gpt-5.1-codex-mini",
        # Common non-codex aliases users pick from generic OpenAI model
        # lists. The Codex backend doesn't serve these, so remap them to
        # the closest Codex-family equivalent rather than 400ing.
        "gpt-5-mini": "gpt-5.1-codex-mini",
        "gpt-5.1-mini": "gpt-5.1-codex-mini",
        "gpt-5.2-mini": "gpt-5.1-codex-mini",
        "gpt-5.3-mini": "gpt-5.1-codex-mini",
        "gpt-5.4-mini": "gpt-5.1-codex-mini",
    }
    return mapping.get(base.lower(), base)


def _is_codex_family_model(name: str) -> bool:
    """Return True if *name* is a GPT-5 / Codex family model that can
    only be served by the ChatGPT Responses endpoint (never by the
    regular ``api.openai.com/v1`` chat completions API)."""
    normalized = _normalize_chatgpt_model(name)
    if not normalized:
        return False
    lowered = normalized.lower()
    if "codex" in lowered:
        return True
    if lowered.startswith("gpt-5"):
        return True
    return False


def _convert_messages_for_responses_api(
    messages: list[Message],
) -> tuple[str, list[dict[str, Any]]]:
    """Convert internal messages to ChatGPT Responses API format.

    Returns ``(instructions, input_items)`` where *instructions* is the
    concatenated system prompt and *input_items* is a list of Responses
    API input objects.
    """
    system_parts: list[str] = []
    items: list[dict[str, Any]] = []

    for msg in messages:
        if isinstance(msg, dict):
            role = str(msg.get("role", ""))
            content = str(msg.get("content", ""))
            tool_call_id = msg.get("tool_call_id")
            tool_calls = msg.get("tool_calls")
        else:
            role = str(getattr(msg, "role", ""))
            content = str(getattr(msg, "content", ""))
            tool_call_id = getattr(msg, "tool_call_id", None)
            tool_calls = getattr(msg, "tool_calls", None)

        if role == "system":
            if content:
                system_parts.append(content)
            continue

        if role == "user":
            items.append({
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": content}],
            })
            continue

        if role == "assistant":
            # Emit text content as a message item (if any).
            if content:
                items.append({
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": content}],
                })
            # Emit each tool call as a separate function_call item.
            if isinstance(tool_calls, list):
                for idx, raw in enumerate(tool_calls, start=1):
                    if not isinstance(raw, dict):
                        continue
                    fn = raw.get("function") if isinstance(raw.get("function"), dict) else raw
                    fn_name = str(fn.get("name", "") or "")
                    fn_args = fn.get("arguments", {})
                    if isinstance(fn_args, dict):
                        fn_args = json.dumps(fn_args, ensure_ascii=False)
                    elif not isinstance(fn_args, str):
                        fn_args = "{}"
                    call_id = str(raw.get("id", "") or f"call_{idx}")
                    items.append({
                        "type": "function_call",
                        "call_id": call_id,
                        "name": fn_name,
                        "arguments": fn_args,
                    })
            continue

        if role == "tool":
            items.append({
                "type": "function_call_output",
                "call_id": str(tool_call_id or f"call_{len(items)}"),
                "output": content,
            })
            continue

    instructions = "\n\n".join(system_parts)
    return instructions, items


def _convert_tools_for_responses_api(
    tools: list[ToolDefinition],
) -> list[dict[str, Any]]:
    """Convert tool definitions to Responses API format.

    Responses API uses a flat structure (``name`` at top level) unlike
    Chat Completions which nests under a ``function`` key.
    """
    result: list[dict[str, Any]] = []
    for tool in tools:
        if isinstance(tool, dict):
            name = tool.get("name")
            description = tool.get("description", "")
            parameters = tool.get("parameters", {})
        else:
            name = getattr(tool, "name", None)
            description = getattr(tool, "description", "") or ""
            parameters = getattr(tool, "parameters", None) or {}
        if not name:
            continue
        result.append({
            "type": "function",
            "name": str(name),
            "description": str(description),
            "parameters": parameters if isinstance(parameters, dict) else {},
        })
    return result


class _CodexAuthExpired(Exception):
    """Internal sentinel — raised when the Responses API returns 401.

    Caught inside :class:`ChatGPTResponsesProvider` to trigger a
    one-shot forced refresh from the Codex auth manager. Never
    surfaces to callers.

    Carries the upstream response body so the *second* 401 (after the
    forced refresh) can be re-raised as an :class:`LLMError` whose
    message includes what chatgpt.com actually said — otherwise we'd
    swallow the diagnostic and surface an empty "ChatGPT Responses
    API call failed:" string.
    """

    def __init__(self, body: str = "") -> None:
        super().__init__(body or "401 Unauthorized")
        self.body = body


class ChatGPTResponsesProvider(LLMProvider):
    """Direct connection to the ChatGPT Responses API.

    Used when the OpenAI provider has ``extra_headers`` configured
    (e.g. ``Authorization``, ``chatgpt-account-id``, ``OpenAI-Beta``).
    Bypasses LiteLLM entirely and speaks the Responses API protocol
    (SSE streaming, ``input`` items instead of ``messages``).
    """

    def __init__(
        self,
        model: str = "gpt-5",
        base_url: str = "https://chatgpt.com/backend-api/codex/responses",
        extra_headers: dict[str, str] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 32000,
        tokens_per_minute: int = 0,
        use_codex_auth_manager: bool = True,
    ):
        import uuid

        self.provider = "openai"
        # Pull off any reasoning-effort suffix (gpt-5-high → effort=high)
        # before alias normalization, then remember it for the request.
        _stripped, _effort = _extract_reasoning_effort(model)
        self.reasoning_effort = _effort
        self.model = _normalize_chatgpt_model(model)
        if self.model != (model or "").strip():
            log.info(
                "ChatGPT Codex: remapped model alias",
                requested=model,
                using=self.model,
                reasoning_effort=self.reasoning_effort or "default",
            )
        if self.model not in _CODEX_BACKEND_SUPPORTED_MODELS:
            log.warning(
                "ChatGPT Codex: model not in known-supported set — "
                "the chatgpt.com Codex backend will likely reject it. "
                "Pick one of: %s",
                ", ".join(sorted(_CODEX_BACKEND_SUPPORTED_MODELS)),
                extra={"requested_model": model, "normalized": self.model},
            )
        self.base_url = base_url.rstrip("/")
        self.extra_headers = dict(extra_headers or {})
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.session_id = uuid.uuid4().hex
        self.client = httpx.AsyncClient(timeout=600.0, follow_redirects=True)
        self.rate_limiter = (
            TokenRateLimiter(tokens_per_minute) if tokens_per_minute > 0 else None
        )

        # Codex tokens (Authorization + chatgpt-account-id) expire roughly
        # every 24 hours and are refreshed in the background by the Codex
        # CLI (or, when running under Flight Deck, by FD re-reading
        # ``~/.codex/auth.json`` on demand). When enabled, we hand the
        # auth problem off to :class:`CodexAuthManager`, which sources
        # tokens from Flight Deck (preferred) or the local auth.json,
        # and refreshes them pre-request when stale or on any 401.
        self._codex_auth: "CodexAuthManager | None" = None
        if use_codex_auth_manager:
            try:
                from captain_claw.codex_auth_manager import CodexAuthManager
                self._codex_auth = CodexAuthManager()
            except Exception as exc:  # pragma: no cover — import safety net
                log.debug("CodexAuthManager unavailable: %s", exc)
                self._codex_auth = None

    # ── helpers ────────────────────────────────────────────────────────

    def _build_headers(self) -> dict[str, str]:
        """Merge provider extra_headers with required defaults."""
        headers: dict[str, str] = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "session_id": self.session_id,
        }
        headers.update(self.extra_headers)
        return headers

    async def _ensure_fresh_auth(self, *, force: bool = False) -> None:
        """Refresh ``self.extra_headers`` from the Codex auth manager.

        Called before each request (fast path: returns immediately when
        the cached token isn't stale) and again after any 401 response
        with ``force=True``.
        """
        if self._codex_auth is None:
            return
        try:
            fresh = await self._codex_auth.get_headers(force_refresh=force)
        except Exception as exc:
            log.debug("Codex auth refresh failed: %s", exc)
            return
        if not fresh:
            return
        # Replace just the auth-related headers so any caller-supplied
        # extras (e.g. OpenAI-Beta) are preserved.
        for key in ("Authorization", "chatgpt-account-id"):
            if key in fresh:
                self.extra_headers[key] = fresh[key]

    def _build_payload(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        instructions, input_items = _convert_messages_for_responses_api(messages)
        api_tools = _convert_tools_for_responses_api(tools) if tools else []
        # One-shot tool_choice override, set by the orchestration loop
        # when it wants to force tool use on a retry (e.g. after a
        # stall). Consumed and reset here so the very next call returns
        # to the default ``"auto" if api_tools else "none"`` behavior.
        override = getattr(self, "_tool_choice_override", None)
        if override is not None:
            try:
                self._tool_choice_override = None
            except Exception:
                pass
        if override and api_tools:
            resolved_tool_choice = override
        else:
            resolved_tool_choice = "auto" if api_tools else "none"
        payload: dict[str, Any] = {
            "model": self.model,
            "input": input_items,
            "tools": api_tools,
            "tool_choice": resolved_tool_choice,
            "parallel_tool_calls": False,
            "store": False,
            "stream": True,
            "prompt_cache_key": self.session_id,
        }
        payload["instructions"] = instructions or "Follow the user's instructions."
        if self.reasoning_effort:
            payload["reasoning"] = {"effort": self.reasoning_effort}
        return payload

    @staticmethod
    def _parse_sse_events(lines: list[str]) -> list[dict[str, Any]]:
        """Parse raw SSE lines into a list of JSON event dicts."""
        events: list[dict[str, Any]] = []
        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith("event:"):
                continue
            if stripped.startswith("data: "):
                data_str = stripped[6:]
            elif stripped.startswith("data:"):
                data_str = stripped[5:]
            else:
                continue
            try:
                events.append(json.loads(data_str))
            except json.JSONDecodeError:
                continue
        return events

    def _parse_response_output(
        self,
        completed_event: dict[str, Any],
    ) -> LLMResponse:
        """Extract content and tool calls from a ``response.completed`` event."""
        response_data = completed_event.get("response") or completed_event
        output_items = response_data.get("output", []) or []

        content_parts: list[str] = []
        tool_calls: list[ToolCall] = []

        def _harvest_text(node: Any) -> None:
            """Recursively pull any ``text`` strings out of a Responses API
            output node. The Codex backend's shape varies between
            ``message`` items with ``content[].text``, ``output_text``
            items with a top-level ``text``, and (for reasoning models)
            nested ``content`` arrays — so we just walk everything."""
            if isinstance(node, dict):
                t = node.get("type", "")
                if isinstance(t, str) and ("output_text" in t or t == "text"):
                    text = node.get("text")
                    if isinstance(text, str) and text:
                        content_parts.append(text)
                # Some shapes nest text in {"text": {"value": "..."}}
                txt_field = node.get("text")
                if isinstance(txt_field, dict):
                    val = txt_field.get("value")
                    if isinstance(val, str) and val:
                        content_parts.append(val)
                for key in ("content", "parts", "items"):
                    child = node.get(key)
                    if child is not None:
                        _harvest_text(child)
            elif isinstance(node, list):
                for child in node:
                    _harvest_text(child)

        for item in output_items:
            item_type = item.get("type", "")

            if item_type == "message":
                _harvest_text(item.get("content", []) or [])

            elif item_type in ("output_text", "text"):
                _harvest_text(item)

            elif item_type == "function_call":
                call_id = str(item.get("call_id", "") or item.get("id", ""))
                name = str(item.get("name", ""))
                raw_args = item.get("arguments", "{}")
                args = _safe_json_loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
                tool_calls.append(ToolCall(id=call_id, name=name, arguments=args))

            else:
                # Unknown item shape — try to harvest any text fields
                # rather than silently dropping the turn.
                _harvest_text(item)

        usage_raw = response_data.get("usage", {}) or {}
        usage = {
            "prompt_tokens": int(usage_raw.get("input_tokens", 0) or 0),
            "completion_tokens": int(usage_raw.get("output_tokens", 0) or 0),
            "total_tokens": int(usage_raw.get("total_tokens", 0) or 0),
        }
        if usage["total_tokens"] <= 0:
            usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]

        return LLMResponse(
            content="".join(content_parts),
            tool_calls=tool_calls,
            model=str(response_data.get("model", self.model) or self.model),
            usage=usage,
            finish_reason="tool_calls" if tool_calls else "stop",
        )

    # ── LLMProvider interface ──────────────────────────────────────────

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        estimated = 0
        if self.rate_limiter:
            estimated = self._estimate_request_tokens(messages, max_tokens or self.max_tokens)
            await self.rate_limiter.acquire(estimated)

        payload = self._build_payload(messages, tools, temperature, max_tokens)

        # Diagnostic: surface exactly which tools we're sending to the
        # Codex Responses API.  Helps debug "model won't invoke tool X"
        # — if X isn't in this list, the payload is stripping it; if it
        # is, the model itself is refusing.
        try:
            _payload_tool_names = [t.get("name") for t in payload.get("tools") or []]
            _mcp_in_payload = [n for n in _payload_tool_names if isinstance(n, str) and n.startswith("mcp_")]
            log.info(
                "ChatGPT Responses API payload: tools",
                total=len(_payload_tool_names),
                mcp=len(_mcp_in_payload),
                tool_choice=payload.get("tool_choice"),
                sample=_payload_tool_names[:25],
            )
        except Exception:
            pass

        # Refresh from Codex auth manager pre-request (cheap no-op when
        # cached token isn't stale). On 401 we'll retry once with a
        # forced refresh.
        await self._ensure_fresh_auth()

        async def _do_request() -> list[str]:
            collected: list[str] = []
            headers = self._build_headers()
            async with self.client.stream(
                "POST", self.base_url, json=payload, headers=headers,
            ) as response:
                if response.status_code == 401:
                    body_bytes = await response.aread()
                    body_text = body_bytes.decode(errors="replace")[:500]
                    log.warning(
                        "ChatGPT Responses API 401 body",
                        body=body_text,
                        model=self.model,
                        url=self.base_url,
                        actual_url=str(response.request.url),
                        host=response.request.url.host,
                        port=response.request.url.port,
                        auth_present=bool(headers.get("Authorization")),
                        account_id_present=bool(headers.get("chatgpt-account-id")),
                    )
                    raise _CodexAuthExpired(body_text)
                if not response.is_success:
                    error_text = await response.aread()
                    raise LLMAPIError(
                        f"ChatGPT Responses API error {response.status_code}: {error_text.decode(errors='replace')}",
                        status_code=response.status_code,
                    )
                log.debug(
                    "ChatGPT Responses API stream opened",
                    status=response.status_code,
                    content_type=response.headers.get("content-type", ""),
                )
                async for line in response.aiter_lines():
                    collected.append(line)
            return collected

        try:
            try:
                collected_lines = await _do_request()
            except _CodexAuthExpired:
                log.info("ChatGPT Responses API 401 — force-refreshing Codex auth and retrying once.")
                await self._ensure_fresh_auth(force=True)
                collected_lines = await _do_request()

            events = self._parse_sse_events(collected_lines)

            log.info(
                "ChatGPT Responses API stream parsed",
                model=self.model,
                raw_lines=len(collected_lines),
                events=len(events),
                event_types=sorted({str(e.get("type", "")) for e in events})[:20],
            )

            # Find the completed event.
            completed: dict[str, Any] | None = None
            for evt in events:
                if evt.get("type") == "response.completed":
                    completed = evt
                    break

            if completed is None:
                # Fallback: try last event or collect text deltas.
                content_parts: list[str] = []
                tc_items: list[dict[str, Any]] = []
                for evt in events:
                    etype = evt.get("type", "")
                    if etype == "response.output_text.delta":
                        content_parts.append(str(evt.get("delta", "")))
                    elif etype == "response.output_item.done":
                        item = evt.get("item", {})
                        if item.get("type") == "function_call":
                            tc_items.append(item)
                tool_calls = []
                for tc in tc_items:
                    call_id = str(tc.get("call_id", "") or tc.get("id", ""))
                    name = str(tc.get("name", ""))
                    raw_args = tc.get("arguments", "{}")
                    args = _safe_json_loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
                    tool_calls.append(ToolCall(id=call_id, name=name, arguments=args))
                result = LLMResponse(
                    content="".join(content_parts),
                    tool_calls=tool_calls,
                    model=self.model,
                    usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                    finish_reason="tool_calls" if tool_calls else "stop",
                )
            else:
                result = self._parse_response_output(completed)

            # If parsing the completed event produced no content/tools,
            # reconstruct directly from the streamed events. The Codex
            # backend reliably emits text via ``response.output_text.delta``
            # and tool calls via ``response.output_item.done`` (with the
            # function_call item) plus per-arg ``response.function_call_arguments.delta``
            # chunks. Either source is enough to recover a valid turn.
            if not result.content and not result.tool_calls:
                delta_parts: list[str] = []
                # call_id → {"id", "name", "arguments_str"}
                fc_by_id: dict[str, dict[str, Any]] = {}
                fc_order: list[str] = []

                def _ingest_fc_item(item: dict[str, Any]) -> None:
                    if not isinstance(item, dict):
                        return
                    if item.get("type") != "function_call":
                        return
                    cid = str(item.get("call_id", "") or item.get("id", ""))
                    if not cid:
                        return
                    slot = fc_by_id.setdefault(
                        cid, {"id": cid, "name": "", "arguments_str": ""}
                    )
                    if cid not in fc_order:
                        fc_order.append(cid)
                    nm = item.get("name")
                    if isinstance(nm, str) and nm:
                        slot["name"] = nm
                    args = item.get("arguments")
                    if isinstance(args, str) and args:
                        slot["arguments_str"] = args
                    elif isinstance(args, dict):
                        slot["arguments_str"] = json.dumps(args)

                for evt in events:
                    et = evt.get("type", "")
                    if et == "response.output_text.delta":
                        d = evt.get("delta", "")
                        if isinstance(d, str) and d:
                            delta_parts.append(d)
                    elif et == "response.output_item.added":
                        _ingest_fc_item(evt.get("item", {}) or {})
                    elif et == "response.output_item.done":
                        _ingest_fc_item(evt.get("item", {}) or {})
                    elif et == "response.function_call_arguments.delta":
                        cid = str(evt.get("call_id", "") or evt.get("item_id", ""))
                        if cid:
                            slot = fc_by_id.setdefault(
                                cid, {"id": cid, "name": "", "arguments_str": ""}
                            )
                            if cid not in fc_order:
                                fc_order.append(cid)
                            d = evt.get("delta", "")
                            if isinstance(d, str):
                                # Only append deltas if we don't already have
                                # the full arguments string from output_item.done
                                # (which would otherwise duplicate).
                                if not slot["arguments_str"]:
                                    slot["arguments_str"] += d
                    elif et == "response.function_call_arguments.done":
                        cid = str(evt.get("call_id", "") or evt.get("item_id", ""))
                        full = evt.get("arguments", "")
                        if cid and isinstance(full, str) and full:
                            slot = fc_by_id.setdefault(
                                cid, {"id": cid, "name": "", "arguments_str": ""}
                            )
                            if cid not in fc_order:
                                fc_order.append(cid)
                            slot["arguments_str"] = full

                recovered_tool_calls: list[ToolCall] = []
                for cid in fc_order:
                    slot = fc_by_id[cid]
                    if not slot.get("name"):
                        continue
                    raw_args = slot.get("arguments_str") or "{}"
                    args = _safe_json_loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
                    recovered_tool_calls.append(
                        ToolCall(id=slot["id"], name=slot["name"], arguments=args)
                    )

                if delta_parts or recovered_tool_calls:
                    result = LLMResponse(
                        content="".join(delta_parts),
                        tool_calls=recovered_tool_calls or result.tool_calls,
                        model=result.model,
                        usage=result.usage,
                        finish_reason=(
                            "tool_calls" if recovered_tool_calls else result.finish_reason
                        ),
                    )

            if not result.content and not result.tool_calls:
                # The Codex backend sometimes returns 200 with a body
                # that has no message / no tool_calls — usually because
                # the model isn't actually served on this account, or
                # because the request was rejected mid-stream with an
                # error event we didn't recognise. Surface a useful
                # diagnostic instead of letting the orchestrator finish
                # silently with an empty turn.
                preview_lines = [ln for ln in collected_lines if ln.strip()][:20]
                log.warning(
                    "ChatGPT Responses API returned an empty turn",
                    model=self.model,
                    raw_lines=len(collected_lines),
                    events=len(events),
                    completed_seen=completed is not None,
                    preview="\n".join(preview_lines)[:2000],
                )
                # Look for a Codex error event the parser may have missed.
                err_detail = ""
                for evt in events:
                    et = str(evt.get("type", ""))
                    if "error" in et.lower():
                        err_detail = json.dumps(evt)[:500]
                        break
                if err_detail:
                    raise LLMAPIError(
                        f"ChatGPT Responses API returned an error event: {err_detail}"
                    )

            if self.rate_limiter:
                self.rate_limiter.record_actual(result.usage.get("total_tokens", 0), estimated)
            return result

        except LLMAPIError:
            raise
        except httpx.HTTPError as exc:
            raise LLMAPIError(f"ChatGPT Responses API HTTP error: {exc}")
        except Exception as exc:
            raise LLMError(f"ChatGPT Responses API call failed: {exc}")

    async def complete_streaming(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        if self.rate_limiter:
            estimated = self._estimate_request_tokens(messages, max_tokens or self.max_tokens)
            await self.rate_limiter.acquire(estimated)

        payload = self._build_payload(messages, tools, temperature, max_tokens)

        await self._ensure_fresh_auth()

        # For streaming we retry on 401 by re-entering the stream once.
        # The retry is only safe before we've yielded any delta to the
        # caller; we check the response status code first and bail out
        # of the pre-stream block via ``continue`` when we see a 401.
        for _attempt in (1, 2):
            headers = self._build_headers()
            try:
                async with self.client.stream(
                    "POST", self.base_url, json=payload, headers=headers,
                ) as response:
                    if response.status_code == 401 and _attempt == 1:
                        await response.aread()
                        log.info("ChatGPT Responses API 401 (streaming) — force-refreshing Codex auth and retrying once.")
                        await self._ensure_fresh_auth(force=True)
                        continue
                    if not response.is_success:
                        error_text = await response.aread()
                        raise LLMAPIError(
                            f"ChatGPT Responses API error {response.status_code}: {error_text.decode(errors='replace')}",
                            status_code=response.status_code,
                        )
                    async for line in response.aiter_lines():
                        stripped = line.strip()
                        if not stripped or stripped.startswith("event:"):
                            continue
                        if stripped.startswith("data: "):
                            data_str = stripped[6:]
                        elif stripped.startswith("data:"):
                            data_str = stripped[5:]
                        else:
                            continue
                        try:
                            evt = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue
                        if evt.get("type") == "response.output_text.delta":
                            delta = evt.get("delta", "")
                            if delta:
                                yield str(delta)
                    return  # successful stream completed
            except LLMAPIError:
                raise
            except httpx.HTTPError as exc:
                raise LLMAPIError(f"ChatGPT Responses API streaming error: {exc}")
            except Exception as exc:
                raise LLMError(f"ChatGPT Responses API streaming failed: {exc}")

    def count_tokens(self, text: str) -> int:
        return len(text) // 4

    async def close(self) -> None:
        await self.client.aclose()
        if self._codex_auth is not None:
            await self._codex_auth.close()


# Model families served via Ollama (local or Ollama cloud) that support —
# and materially benefit from — native thinking. Sending `think: false`
# to these EXPLICITLY DISABLES their reasoning, which is why "DeepSeek via
# API reasons well but via Ollama it's bad": the API serves the reasoner
# natively while our request was switching it off.
_OLLAMA_THINKING_MODEL_RE = re.compile(
    r"(?i)deepseek|reasoner|qwq|qwen3|glm|kimi|gpt-oss|magistral|cogito|"
    r"exaone-deep|smallthinker|phi-?4.*reason|-r1\b|r1[:\-]|thinking"
)


def _resolve_ollama_think(raw: bool | str | None, model: str) -> bool | str:
    """Decide the `think` value for an Ollama request.

    Precedence: CLAW_OLLAMA_THINK env (1/0 or low/medium/high) → explicit
    constructor value → auto-detect by model family. Auto-on for known
    thinking families; a model that rejects it falls back at request time.
    """
    env = os.getenv("CLAW_OLLAMA_THINK", "").strip().lower()
    if env in ("1", "true", "yes", "on"):
        return True
    if env in ("0", "false", "no", "off"):
        return False
    if env in ("low", "medium", "high"):
        return env
    if raw is not None:
        return raw
    return bool(_OLLAMA_THINKING_MODEL_RE.search(model or ""))


# Ceiling on `num_ctx` for LOCALLY-hosted Ollama models. Ollama allocates
# the KV cache for the full num_ctx up front, so a frontier-sized context
# budget handed to a small local model is not merely wasteful — it is
# fatal. A 9B model asked for 200k ctx wants tens of GB of KV cache; on a
# workstation that spills to swap, throughput collapses (measured: 0.2
# tok/s), and the starved generation comes back empty.
#
# This bites because `context.max_tokens` is written once at spawn from
# the archetype's tier (`reason` = 200000) and then outlives the model it
# was sized for — swapping an agent onto a local model leaves the frontier
# budget in place. Clamping here catches every such agent regardless of
# how its config was produced.
#
# `:cloud` models are exempt: those run on Ollama's hosted hardware, where
# a large context is both legitimate and free of local memory pressure.
# Raise the ceiling with CLAW_OLLAMA_MAX_NUM_CTX if you have the memory.
_OLLAMA_DEFAULT_MAX_NUM_CTX = 32768


def _clamp_ollama_num_ctx(num_ctx: int, model: str) -> int:
    """Clamp a requested context window to what a local Ollama box can hold."""
    requested = max(1, int(num_ctx))
    if str(model or "").strip().endswith(":cloud"):
        return requested
    try:
        ceiling = int(
            os.getenv("CLAW_OLLAMA_MAX_NUM_CTX", "") or _OLLAMA_DEFAULT_MAX_NUM_CTX
        )
    except ValueError:
        ceiling = _OLLAMA_DEFAULT_MAX_NUM_CTX
    if ceiling <= 0 or requested <= ceiling:
        return requested
    log.warning(
        "Clamping Ollama num_ctx — requested window would blow up the local KV cache",
        model=model,
        requested=requested,
        clamped_to=ceiling,
        override_env="CLAW_OLLAMA_MAX_NUM_CTX",
    )
    return ceiling


def _forced_tool_call_schema(tools: list[ToolDefinition]) -> dict[str, Any] | None:
    """JSON schema that forces the model to emit exactly one tool call.

    Ollama's native ``/api/chat`` has no ``tool_choice``, so the agent
    loop's ``_tool_choice_override = "required"`` (its strongest lever
    against a stalling model) was a silent no-op on precisely the local
    models that stall most. Grammar-constrained decoding via ``format``
    is the equivalent Ollama *does* provide.

    Shape is a flat object with an enum discriminator rather than a
    ``oneOf`` union over per-tool argument schemas — same reasoning as
    ``mrav.protocol.ACT_RESPONSE_SCHEMA``: flat + enum is far more
    robust for small models and for grammar engines.
    """
    names = [
        str(t.get("name", "")).strip()
        for t in (tools or [])
        if str(t.get("name", "")).strip()
    ]
    if not names:
        return None
    return {
        "type": "object",
        "properties": {
            "tool": {"type": "string", "enum": names},
            "arguments": {"type": "object"},
        },
        "required": ["tool", "arguments"],
    }


# Captain Claw model ids that need remapping to a ``claude --model`` value.
# Bare family names (opus/sonnet/haiku/fable) and full dated ids the CLI
# already accepts pass straight through, so this map stays intentionally
# small — extend it only when the CLI rejects an id Captain Claw uses.
_CLAUDE_CLI_MODEL_ALIASES = {
    "claude": "sonnet",
}


class ClaudeCLIProvider(LLMProvider):
    """Anthropic **subscription** (Pro/Max) via the ``claude`` CLI.

    Subscription usage is only billable through the Claude Code CLI or the
    Agent SDK — never ``api.anthropic.com`` with a key (always metered
    pay-per-token), and never an OAuth-token HTTP client pointed at the
    Messages API (a ToS-violating spoof). So this provider shells out to
    ``claude -p`` and reads the subscription-billed result back.

    **Scope: text generation only.** The CLI is an *agent*, not a chat
    endpoint — it *executes* tools in its own loop instead of returning a
    tool_call for Captain Claw's loop to run. So tool/function calling is
    deliberately unimplemented (``supports_tools = False``); route only
    tool-free work here (council prose, narration, research synthesis,
    reflections). Tool-using agent turns stay on the ``anthropic`` API
    provider.

    **Runtime dependency:** a *valid* OAuth login for the standalone
    ``claude`` binary — ``claude login`` (interactive) or
    ``claude setup-token`` (a long-lived ``sk-ant-oat01-…`` token; pass it
    as ``oauth_token`` or the ``CLAUDE_CODE_OAUTH_TOKEN`` env var for
    headless/daemon use). ``ANTHROPIC_API_KEY`` in the environment silently
    wins over OAuth and bills the API, so it — and any ``ANTHROPIC_BASE_URL``
    proxy override that would misroute off the subscription endpoint — is
    scrubbed from the child environment.

    **Note:** the CLI exposes no ``temperature`` / ``max_tokens`` knobs, so
    those arguments are accepted for interface parity but ignored. Spawning
    the CLI also fires the user's Claude Code hooks (e.g. ``SessionStart``);
    their output is discarded.
    """

    #: Advertised so orchestration never routes tool-calls to this provider.
    supports_tools = False

    def __init__(
        self,
        model: str = "sonnet",
        base_url: str | None = None,      # optional explicit ``claude`` binary path
        temperature: float = 0.7,         # accepted for parity; the CLI has no knob
        max_tokens: int = 32000,          # accepted for parity; the CLI has no knob
        tokens_per_minute: int = 0,
        oauth_token: str | None = None,
        cli_path: str | None = None,
        cwd: str | None = None,
        timeout: float = 600.0,
        scrub_base_url: bool = True,
    ):
        self.provider = "claude-cli"
        self.model = self._map_model(model)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.scrub_base_url = scrub_base_url
        # Long-lived token (``claude setup-token``) for headless use; falls
        # back to the ambient login in ``~/.claude/.credentials.json``.
        self.oauth_token = oauth_token or os.getenv("CLAUDE_CODE_OAUTH_TOKEN") or None
        self.cli_path = self._resolve_cli(cli_path or base_url)
        # Neutral working dir so the CLI doesn't scrape the project's
        # CLAUDE.md / files into the generation context.
        self.cwd = cwd or tempfile.gettempdir()
        self.rate_limiter = (
            TokenRateLimiter(tokens_per_minute) if tokens_per_minute > 0 else None
        )
        self._warned_tools = False
        log.info(
            "ClaudeCLIProvider ready (Anthropic subscription via claude CLI)",
            model=self.model,
            cli_path=self.cli_path,
            has_oauth_token=bool(self.oauth_token),
        )

    # ── helpers ──────────────────────────────────────────────────────────
    @staticmethod
    def _resolve_cli(explicit: str | None) -> str:
        for cand in (
            explicit,
            os.getenv("CLAUDE_CLI_PATH"),
            shutil.which("claude"),
            "~/.local/bin/claude",
        ):
            if not cand:
                continue
            path = os.path.expanduser(cand)
            if os.path.exists(path):
                return path
        # Last resort: bare name, let PATH resolution fail loudly at spawn.
        return "claude"

    @staticmethod
    def _map_model(model: str) -> str:
        m = (model or "").strip()
        if "/" in m:  # strip a provider prefix (``anthropic/claude-…``)
            m = m.split("/", 1)[1]
        return _CLAUDE_CLI_MODEL_ALIASES.get(m.lower(), m) or "sonnet"

    def _child_env(self) -> dict[str, str]:
        env = dict(os.environ)
        # The single most common way this path gets mis-billed: an
        # ANTHROPIC_API_KEY anywhere in the environment silently overrides
        # OAuth and bills pay-per-token. Drop it (and the token variant).
        env.pop("ANTHROPIC_API_KEY", None)
        env.pop("ANTHROPIC_AUTH_TOKEN", None)
        if self.scrub_base_url:
            env.pop("ANTHROPIC_BASE_URL", None)
        if self.oauth_token:
            env["CLAUDE_CODE_OAUTH_TOKEN"] = self.oauth_token
        return env

    @staticmethod
    def _split_messages(messages: list[Message]) -> tuple[str, str]:
        """Flatten a conversation into ``(system_prompt, prompt_body)``.

        The CLI takes one system prompt and one prompt body, so system
        messages are concatenated and the remaining turns are rendered as a
        labelled transcript. A lone turn is passed through verbatim.
        """
        system_parts: list[str] = []
        turns: list[tuple[str, str]] = []
        for m in messages:
            role = str(_obj_get(m, "role", "") or "")
            content = str(_obj_get(m, "content", "") or "")
            if role == "system":
                if content.strip():
                    system_parts.append(content)
            elif role == "tool":
                name = _obj_get(m, "tool_name", None)
                label = f"Tool result ({name})" if name else "Tool result"
                turns.append(("tool", f"{label}:\n{content}"))
            else:
                turns.append((role or "user", content))
        system_prompt = "\n\n".join(system_parts).strip()
        if not turns:
            return system_prompt, ""
        if len(turns) == 1:
            return system_prompt, turns[0][1]
        lines: list[str] = []
        for role, content in turns:
            if role == "assistant":
                lines.append(f"Assistant: {content}")
            elif role == "tool":
                lines.append(content)
            else:
                lines.append(f"User: {content}")
        return system_prompt, "\n\n".join(lines)

    def _build_argv(
        self, system_prompt: str, sp_file: str | None, streaming: bool
    ) -> list[str]:
        argv = [
            self.cli_path, "-p",
            "--input-format", "text",
            "--output-format", "stream-json" if streaming else "json",
            "--model", self.model,
            # No tools: this is pure text generation. Print mode cannot
            # answer a permission prompt, so nothing executes regardless.
            "--allowedTools", "",
        ]
        if streaming:
            # stream-json in --print mode is rejected without --verbose.
            argv += ["--verbose", "--include-partial-messages"]
        if sp_file:
            argv += ["--system-prompt-file", sp_file,
                     "--exclude-dynamic-system-prompt-sections"]
        elif system_prompt:
            argv += ["--system-prompt", system_prompt,
                     "--exclude-dynamic-system-prompt-sections"]
        return argv

    @staticmethod
    def _maybe_spill_system(system_prompt: str) -> str | None:
        """Write an oversized system prompt to a temp file (arg-length guard)."""
        if len(system_prompt) <= 200_000:
            return None
        fd, path = tempfile.mkstemp(prefix="claw-sysprompt-", suffix=".txt")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(system_prompt)
        except Exception:
            try:
                os.unlink(path)
            except OSError:
                pass
            return None
        return path

    @staticmethod
    def _text_of(msg: dict[str, Any]) -> str:
        parts: list[str] = []
        for block in (msg.get("content") or []):
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)

    def _auth_hint(self, text: str) -> str:
        t = (text or "").strip()
        low = t.lower()
        if any(k in low for k in ("authenticate", "oauth", "expired", "log in", "login")):
            return (
                "claude CLI is not authenticated for the subscription "
                f"({t[:200]}). Run `claude setup-token` (headless, long-lived) "
                "or `claude login`, and make sure ANTHROPIC_API_KEY is unset."
            )
        return f"claude CLI error: {t[:300]}"

    @staticmethod
    def _parse_result_json(raw: str) -> dict[str, Any]:
        raw = (raw or "").strip()
        if not raw:
            raise LLMAPIError("claude CLI returned no output")
        try:
            obj = json.loads(raw)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
        # Fall back: scan lines for the final ``result`` object.
        result: dict[str, Any] | None = None
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(o, dict) and (o.get("type") == "result" or "result" in o):
                result = o
        if result is None:
            raise LLMAPIError(f"claude CLI: unparseable output: {raw[:300]}")
        return result

    async def _spawn_and_read(self, argv: list[str], prompt: str) -> str:
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.cwd,
                env=self._child_env(),
            )
        except FileNotFoundError:
            raise LLMAPIError(
                f"claude CLI not found at '{self.cli_path}'. Install it "
                "(`npm i -g @anthropic-ai/claude-code`) or set CLAUDE_CLI_PATH."
            )
        try:
            out, err = await asyncio.wait_for(
                proc.communicate(prompt.encode("utf-8")), timeout=self.timeout
            )
        except asyncio.TimeoutError:
            try:
                proc.kill()
                await proc.wait()
            except ProcessLookupError:
                pass
            raise LLMAPIError(f"claude CLI timed out after {self.timeout:.0f}s")
        text = (out or b"").decode("utf-8", "replace")
        if proc.returncode not in (0, None) and not text.strip():
            detail = (err or b"").decode("utf-8", "replace")[:500]
            raise LLMAPIError(f"claude CLI exited {proc.returncode}: {detail}")
        return text

    # ── public API ───────────────────────────────────────────────────────
    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        if tools and not self._warned_tools:
            self._warned_tools = True
            log.warning(
                "ClaudeCLIProvider ignores tools — it is text-generation only; "
                "route tool-using turns to the API-backed 'anthropic' provider",
                tool_count=len(tools),
            )
        system_prompt, prompt = self._split_messages(messages)
        if not prompt.strip():
            # The CLI needs a non-empty prompt body; fold system into it.
            prompt, system_prompt = (system_prompt or " "), ""
        sp_file = self._maybe_spill_system(system_prompt)

        estimated = 0
        if self.rate_limiter:
            estimated = self._estimate_request_tokens(messages, max_tokens or self.max_tokens)
            await self.rate_limiter.acquire(estimated)

        argv = self._build_argv(system_prompt, sp_file, streaming=False)
        try:
            raw = await self._spawn_and_read(argv, prompt)
        finally:
            if sp_file:
                try:
                    os.unlink(sp_file)
                except OSError:
                    pass

        data = self._parse_result_json(raw)
        result_text = str(data.get("result", "") or "")
        if data.get("is_error"):
            raise LLMAPIError(self._auth_hint(result_text))

        u = data.get("usage") or {}
        prompt_toks = (
            int(u.get("input_tokens", 0) or 0)
            + int(u.get("cache_read_input_tokens", 0) or 0)
            + int(u.get("cache_creation_input_tokens", 0) or 0)
        )
        completion_toks = int(u.get("output_tokens", 0) or 0)
        usage = {
            "prompt_tokens": prompt_toks,
            "completion_tokens": completion_toks,
            "total_tokens": prompt_toks + completion_toks,
        }
        cost = data.get("total_cost_usd")
        log.info(
            "claude CLI (subscription) call complete",
            model=self.model,
            cost_usd=cost,
            **usage,
        )
        if self.rate_limiter:
            self.rate_limiter.record_actual(usage["total_tokens"], estimated)
        return LLMResponse(
            content=result_text,
            tool_calls=[],
            model=str(data.get("model") or self.model),
            usage=usage,
            finish_reason=str(data.get("stop_reason") or ""),
        )

    async def complete_streaming(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        if tools and not self._warned_tools:
            self._warned_tools = True
            log.warning(
                "ClaudeCLIProvider ignores tools — text-generation only",
                tool_count=len(tools),
            )
        system_prompt, prompt = self._split_messages(messages)
        if not prompt.strip():
            prompt, system_prompt = (system_prompt or " "), ""
        sp_file = self._maybe_spill_system(system_prompt)

        if self.rate_limiter:
            estimated = self._estimate_request_tokens(messages, max_tokens or self.max_tokens)
            await self.rate_limiter.acquire(estimated)

        argv = self._build_argv(system_prompt, sp_file, streaming=True)
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.cwd,
                env=self._child_env(),
            )
        except FileNotFoundError:
            raise LLMAPIError(
                f"claude CLI not found at '{self.cli_path}'. Install it "
                "(`npm i -g @anthropic-ai/claude-code`) or set CLAUDE_CLI_PATH."
            )
        assert proc.stdin is not None and proc.stdout is not None
        try:
            proc.stdin.write(prompt.encode("utf-8"))
            await proc.stdin.drain()
            proc.stdin.close()
        except (BrokenPipeError, ConnectionResetError):
            pass

        streamed = False
        try:
            async for raw_line in proc.stdout:
                line = raw_line.decode("utf-8", "replace").strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue
                etype = ev.get("type")
                if etype == "stream_event":
                    inner = ev.get("event") or {}
                    if inner.get("type") == "content_block_delta":
                        delta = inner.get("delta") or {}
                        if delta.get("type") == "text_delta" and delta.get("text"):
                            streamed = True
                            yield str(delta["text"])
                elif etype == "assistant":
                    msg = ev.get("message") or {}
                    if ev.get("error") or msg.get("model") == "<synthetic>":
                        raise LLMAPIError(self._auth_hint(self._text_of(msg)))
                    # No partial deltas seen (older CLI / non-partial mode):
                    # emit the whole text block once so callers still get output.
                    if not streamed:
                        whole = self._text_of(msg)
                        if whole:
                            streamed = True
                            yield whole
                elif etype == "result":
                    if ev.get("is_error"):
                        raise LLMAPIError(self._auth_hint(str(ev.get("result", ""))))
                    break
        finally:
            if sp_file:
                try:
                    os.unlink(sp_file)
                except OSError:
                    pass
            if proc.returncode is None:
                try:
                    proc.kill()
                    await proc.wait()
                except ProcessLookupError:
                    pass

    def count_tokens(self, text: str) -> int:
        return len(text) // 4

    async def close(self) -> None:  # symmetry with other providers
        return None


class OllamaProvider(LLMProvider):
    """Direct Ollama API provider."""

    def __init__(
        self,
        model: str = "llama3.2",
        base_url: str = OLLAMA_NATIVE_BASE_URL,
        temperature: float = 0.7,
        max_tokens: int = 32000,
        num_ctx: int = 160000,
        api_key: str | None = None,
        tokens_per_minute: int = 0,
        think: bool | str | None = None,
    ):
        self.provider = "ollama"
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.num_ctx = _clamp_ollama_num_ctx(num_ctx, model)
        self.api_key = api_key
        self.think = _resolve_ollama_think(think, model)
        if self.think:
            log.info("Ollama thinking enabled", model=model, think=self.think)
        self.client = httpx.AsyncClient(timeout=120.0, follow_redirects=True)
        self.rate_limiter = TokenRateLimiter(tokens_per_minute) if tokens_per_minute > 0 else None

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_schema: dict[str, Any] | None = None,
    ) -> LLMResponse:
        log.info(
            "OllamaProvider.complete entry",
            model=self.model,
            base_url=self.base_url,
            message_count=len(messages),
            has_tools=bool(tools),
        )
        try:
            from captain_claw.vastai.wake import maybe_wake_instance
            await maybe_wake_instance(self.base_url)
        except Exception:
            pass

        url = f"{self.base_url}/api/chat"
        ollama_messages = _convert_messages_for_ollama(messages)
        ollama_tools = _convert_tools_for_openai_style(tools) if tools else None

        options: dict[str, Any] = {
            "num_ctx": self.num_ctx,
            "temperature": self.temperature if temperature is None else temperature,
        }
        if max_tokens or self.max_tokens:
            options["num_predict"] = max_tokens or self.max_tokens

        body: dict[str, Any] = {
            "model": self.model,
            "messages": ollama_messages,
            "stream": False,
            "options": options,
        }
        if self.think:
            body["think"] = self.think
        if ollama_tools:
            body["tools"] = ollama_tools
        if response_schema:
            # Ollama structured outputs: `format` takes a full JSON schema
            # and constrains decoding via grammar (GBNF-backed).
            body["format"] = response_schema

        # ── Forced tool call (stall recovery) ─────────────────────────
        # Honor the agent loop's `_tool_choice_override = "required"`.
        # Consumed once, then reset — same contract as the LiteLLM and
        # ChatGPT providers. `tools` stays in the body so the prompt
        # template still renders the argument schemas; `think` is
        # disabled because a reasoning block plus a decoding grammar is
        # a known-bad combination on small models (mrav forces the same).
        _forced_tool = False
        _tc_override = getattr(self, "_tool_choice_override", None)
        if _tc_override:
            self._tool_choice_override = None
            if str(_tc_override) == "required" and ollama_tools and not response_schema:
                _forced_schema = _forced_tool_call_schema(tools or [])
                if _forced_schema:
                    body["format"] = _forced_schema
                    body.pop("think", None)
                    _forced_tool = True
                    log.info(
                        "Ollama forced tool call via grammar",
                        model=self.model,
                        tool_count=len(ollama_tools),
                    )

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        estimated = 0
        if self.rate_limiter:
            estimated = self._estimate_request_tokens(messages, max_tokens or self.max_tokens)
            await self.rate_limiter.acquire(estimated)

        max_retries = 2
        last_exc: Exception | None = None
        for attempt in range(1, max_retries + 2):
            try:
                response = await self.client.post(url, json=body, headers=headers)
                if not response.is_success:
                    # Model doesn't support thinking → retry once without it
                    # (auto-detection is a heuristic; this is the safety net).
                    if (body.get("think") and response.status_code == 400
                            and "think" in response.text.lower()):
                        log.warning("Ollama model rejected think — retrying without",
                                    model=self.model)
                        self.think = False
                        body.pop("think", None)
                        continue
                    raise LLMAPIError(
                        f"Ollama API error {response.status_code}: {response.text}",
                        status_code=response.status_code,
                    )
                data = response.json()
                msg_obj = _obj_get(data, "message", {})
                raw_content = _obj_get(msg_obj, "content", "") or ""
                _rc = (
                    _obj_get(msg_obj, "reasoning_content", None)
                    or _obj_get(msg_obj, "thinking", None)
                    or _obj_get(msg_obj, "reasoning", None)
                )
                try:
                    if hasattr(msg_obj, "keys"):
                        _msg_keys_str = ",".join(str(k) for k in list(msg_obj.keys())[:20])
                    else:
                        _msg_keys_str = f"<no-keys: {type(msg_obj).__name__}>"
                except Exception as _ke:
                    _msg_keys_str = f"<error: {_ke}>"
                log.info(
                    "Ollama msg fields",
                    keys=_msg_keys_str,
                    msg_type=type(msg_obj).__name__,
                    raw_content_len=len(str(raw_content)),
                    reasoning_len=len(str(_rc)) if _rc else 0,
                )
                content = _strip_internal_context(_strip_reasoning_artifacts(str(raw_content)))
                # Only surface the reasoning tail as content when the model has
                # NO tool calls. With tool calls the reasoning is "I should call
                # X" — leaking it as content pollutes the reply (and made vision
                # agents narrate "I should call image_vision").
                _has_tool_calls = bool(_obj_get(msg_obj, "tool_calls", []) or [])
                if not content.strip() and _rc and not _has_tool_calls:
                    rc_str = str(_rc)
                    fallback = _reasoning_content_fallback(rc_str)
                    log.warning(
                        "Ollama content empty - recovering answer from reasoning",
                        reasoning_chars=len(rc_str),
                        fallback_chars=len(fallback),
                        recovered_json=fallback.startswith(("{", "[")),
                    )
                    content = fallback
                log.info(
                    "Ollama final content",
                    content_len=len(content),
                    preview=content[:200].replace("\n", " "),
                )

                tool_calls: list[ToolCall] = []
                raw_calls = _obj_get(msg_obj, "tool_calls", []) or []
                for idx, raw_call in enumerate(raw_calls, start=1):
                    function = _obj_get(raw_call, "function", {}) or {}
                    call_name = str(_obj_get(function, "name", "") or "")
                    args = _obj_get(function, "arguments", {})
                    if isinstance(args, str):
                        args = _safe_json_loads(args)
                    if not isinstance(args, dict):
                        args = {}
                    tool_calls.append(ToolCall(
                        id=str(_obj_get(raw_call, "id", f"ollama_call_{idx}") or f"ollama_call_{idx}"),
                        name=call_name,
                        arguments=args,
                    ))

                # Grammar-forced tool call: the model emitted our
                # {"tool": ..., "arguments": {...}} object as *content*,
                # not as a native tool_call. Convert it back so the agent
                # loop sees a normal tool call and the JSON never leaks
                # into the reply.
                if _forced_tool and not tool_calls:
                    _forced_obj = _safe_json_loads(content)
                    _forced_name = ""
                    if isinstance(_forced_obj, dict):
                        _forced_name = str(_forced_obj.get("tool", "") or "").strip()
                    if _forced_name:
                        _forced_args = _forced_obj.get("arguments", {})
                        if not isinstance(_forced_args, dict):
                            _forced_args = {}
                        tool_calls.append(ToolCall(
                            id="ollama_forced_1",
                            name=_forced_name,
                            arguments=_forced_args,
                        ))
                        content = ""
                        log.info(
                            "Ollama forced tool call parsed",
                            tool=_forced_name,
                            arg_keys=sorted(_forced_args)[:10],
                        )
                    else:
                        log.warning(
                            "Ollama forced tool call did not parse",
                            preview=content[:160].replace("\n", " "),
                        )

                usage = {
                    "prompt_tokens": int(_obj_get(data, "prompt_eval_count", 0) or 0),
                    "completion_tokens": int(_obj_get(data, "eval_count", 0) or 0),
                    "total_tokens": int(_obj_get(data, "prompt_eval_count", 0) or 0)
                    + int(_obj_get(data, "eval_count", 0) or 0),
                }
                if self.rate_limiter:
                    self.rate_limiter.record_actual(usage.get("total_tokens", 0), estimated)
                finish_reason = str(_obj_get(data, "done_reason", "") or "")
                return LLMResponse(
                    content=str(content),
                    tool_calls=tool_calls,
                    model=str(_obj_get(data, "model", self.model) or self.model),
                    usage=usage,
                    finish_reason=finish_reason,
                )
            except LLMAPIError:
                raise
            except httpx.HTTPError as e:
                last_exc = e
                if attempt <= max_retries:
                    await asyncio.sleep(1.0 * attempt)
                    continue
                err_detail = str(e) or type(e).__name__
                raise LLMAPIError(f"Ollama HTTP error: {err_detail}")
            except json.JSONDecodeError as e:
                raise LLMError(f"Ollama response decode error: {e}")
            except Exception as e:
                raise LLMError(f"Ollama call failed: {e}")
        # Should not reach here, but just in case:
        err_detail = str(last_exc) or type(last_exc).__name__ if last_exc else "unknown"
        raise LLMAPIError(f"Ollama HTTP error after retries: {err_detail}")

    async def complete_structured(
        self,
        messages: list[Message],
        response_schema: dict[str, Any],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Native structured outputs via Ollama's grammar-backed `format`."""
        return await self.complete(
            messages,
            None,
            temperature,
            max_tokens,
            response_schema=response_schema,
        )

    async def complete_streaming(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        # Auto-wake vast.ai instance if needed.
        try:
            from captain_claw.vastai.wake import maybe_wake_instance
            await maybe_wake_instance(self.base_url)
        except Exception:
            pass

        url = f"{self.base_url}/api/chat"
        ollama_messages = _convert_messages_for_ollama(messages)
        ollama_tools = _convert_tools_for_openai_style(tools) if tools else None

        options: dict[str, Any] = {
            "num_ctx": self.num_ctx,
            "temperature": self.temperature if temperature is None else temperature,
        }
        if max_tokens or self.max_tokens:
            options["num_predict"] = max_tokens or self.max_tokens

        body: dict[str, Any] = {
            "model": self.model,
            "messages": ollama_messages,
            "stream": True,
            "options": options,
        }
        if self.think:
            body["think"] = self.think
        if ollama_tools:
            body["tools"] = ollama_tools

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        if self.rate_limiter:
            estimated = self._estimate_request_tokens(messages, max_tokens or self.max_tokens)
            await self.rate_limiter.acquire(estimated)

        max_retries = 2
        for attempt in range(1, max_retries + 2):
            try:
                async with self.client.stream("POST", url, json=body, headers=headers) as response:
                    if not response.is_success:
                        error_text = await response.atext()
                        if (body.get("think") and response.status_code == 400
                                and "think" in error_text.lower()):
                            log.warning("Ollama model rejected think — retrying without",
                                        model=self.model)
                            self.think = False
                            body.pop("think", None)
                            continue
                        raise LLMAPIError(
                            f"Ollama API error {response.status_code}: {error_text}",
                            status_code=response.status_code,
                        )
                    async for line in response.aiter_lines():
                        if not line.strip():
                            continue
                        try:
                            chunk = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        content = _obj_get(_obj_get(chunk, "message", {}), "content", "") or ""
                        if content:
                            yield str(content)
                        if _obj_get(chunk, "done", False):
                            break
                return  # successful stream completed
            except LLMAPIError:
                raise
            except httpx.HTTPError as e:
                if attempt <= max_retries:
                    await asyncio.sleep(1.0 * attempt)
                    continue
                err_detail = str(e) or type(e).__name__
                raise LLMAPIError(f"Ollama streaming error: {err_detail}")
            except Exception as e:
                raise LLMError(f"Ollama stream failed: {e}")

    def count_tokens(self, text: str) -> int:
        return len(text) // 4

    async def close(self) -> None:
        await self.client.aclose()


class LiteLLMProvider(LLMProvider):
    """LiteLLM-backed provider for OpenAI, Anthropic (Claude), and Gemini."""

    def __init__(
        self,
        provider: str,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 32000,
        tokens_per_minute: int = 0,
        extra_headers: dict[str, str] | None = None,
    ):
        self.provider = _normalize_provider_name(provider)
        # Pull any reasoning-effort suffix off the model name (gpt-5-high
        # → effort=high, model=gpt-5). Meaningful for OpenAI's
        # reasoning models AND DeepSeek thinking-mode models — both
        # accept ``reasoning_effort``. We still strip the suffix for
        # every provider so the LiteLLM call never sees a fake one.
        _stripped, _effort = _extract_reasoning_effort(model)
        self.reasoning_effort = (
            _effort if self.provider in ("openai", "deepseek") else None
        )
        self.model = _provider_model_name(self.provider, _stripped or model)
        self.api_key = _resolve_api_key(self.provider, api_key)
        self.base_url = (base_url or "").strip() or None
        self.temperature = _normalize_temperature_for_model(
            self.provider,
            self.model,
            temperature,
        )
        self.max_tokens = max_tokens
        self.rate_limiter = TokenRateLimiter(tokens_per_minute) if tokens_per_minute > 0 else None
        self.extra_headers = extra_headers or None

        # Google OAuth / Vertex AI credentials (set via set_vertex_credentials).
        self._vertex_credentials: dict[str, Any] | None = None
        self._vertex_project: str = ""
        self._vertex_location: str = "us-central1"

    def set_vertex_credentials(
        self,
        credentials: dict[str, Any],
        project: str,
        location: str = "us-central1",
    ) -> None:
        """Inject Google OAuth ``authorized_user`` credentials for Vertex AI.

        When set and the provider is ``gemini``, requests are routed
        through LiteLLM's ``vertex_ai/`` prefix instead of ``gemini/``,
        using OAuth tokens rather than API keys.
        """
        self._vertex_credentials = credentials
        self._vertex_project = project
        self._vertex_location = location

    def clear_vertex_credentials(self) -> None:
        """Remove injected Vertex AI credentials, reverting to API key auth."""
        self._vertex_credentials = None
        self._vertex_project = ""
        self._vertex_location = "us-central1"

    def _request_kwargs(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stream: bool = False,
    ) -> dict[str, Any]:
        resolved_temperature = _normalize_temperature_for_model(
            self.provider,
            self.model,
            self.temperature if temperature is None else temperature,
        )
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": _convert_messages_for_openai_style(messages),
            "max_tokens": max_tokens or self.max_tokens,
            "stream": stream,
            "timeout": 180,
        }
        # Omit temperature when the model doesn't accept it (e.g. Anthropic
        # Fable/Opus 4.8/Sonnet 5 reject it with a 400 — some known up front,
        # others learned at runtime via _remember_temperature_unsupported).
        # _normalize_* returns None in that case.
        if resolved_temperature is not None:
            kwargs["temperature"] = resolved_temperature
        if tools:
            kwargs["tools"] = _convert_tools_for_openai_style(tools)

        # One-shot tool_choice override (set by the orchestration loop
        # on stall retries to force tool use). Only honored when tools
        # are actually being sent. Consumed and reset so subsequent
        # calls return to the model's default behavior.
        override = getattr(self, "_tool_choice_override", None)
        if override is not None:
            try:
                self._tool_choice_override = None
            except Exception:
                pass
            # Skip when this model has already rejected tool_choice this
            # session (thinking-mode models) — avoids repeating doomed calls.
            if kwargs.get("tools") and not getattr(self, "_tool_choice_unsupported", False):
                kwargs["tool_choice"] = override

        # Anthropic prompt caching: split system message on CACHE_SPLIT marker
        # into static (cached) + dynamic (uncached) blocks, and add a cache
        # breakpoint on the last conversation message for tool-loop caching.
        if self.provider == "anthropic":
            kwargs["messages"] = _inject_anthropic_cache_control(kwargs["messages"])
        else:
            # Strip the cache-split marker for non-Anthropic providers.
            for msg in kwargs["messages"]:
                if msg.get("role") == "system" and isinstance(msg.get("content"), str):
                    msg["content"] = msg["content"].replace(_CACHE_SPLIT_MARKER, "")

        # Force text-only output for Gemini so it uses function tools
        # (e.g. image_gen) instead of native image generation, which
        # returns image bytes that our response parser cannot handle.
        if self.provider == "gemini":
            kwargs["modalities"] = ["text"]

        # Always use the API key for Gemini (Google AI Studio).
        # Vertex AI routing is disabled — use api_key directly.
        if self.api_key:
            kwargs["api_key"] = self.api_key

        if self.base_url:
            kwargs["api_base"] = self.base_url
        if self.extra_headers:
            kwargs["extra_headers"] = self.extra_headers

        # Reasoning-effort suffix on OpenAI reasoning-capable models
        # (gpt-5*, o-series) and DeepSeek thinking-mode models.
        # LiteLLM forwards ``reasoning_effort`` to OpenAI as
        # ``reasoning.effort`` and to DeepSeek as the same field.
        if self.reasoning_effort and self.provider in ("openai", "deepseek"):
            kwargs["reasoning_effort"] = self.reasoning_effort

        # DeepSeek thinking-mode opt-in. Only the dedicated reasoning
        # models actually honor it (``deepseek-reasoner``,
        # ``deepseek-v4-pro``, etc.); other DeepSeek models ignore
        # the field. We default to *enabled* when an effort level was
        # provided via the model-name suffix (``deepseek-reasoner-high``)
        # since the user clearly wanted to think. An ops override
        # via ``FD_DEEPSEEK_THINKING=off`` disables this if it ever
        # causes trouble.
        if self.provider == "deepseek":
            thinking_on = self.reasoning_effort is not None
            if os.environ.get("FD_DEEPSEEK_THINKING", "").strip().lower() in ("off", "0", "false"):
                thinking_on = False
            if thinking_on:
                extra = dict(kwargs.get("extra_body") or {})
                extra.setdefault("thinking", {"type": "enabled"})
                kwargs["extra_body"] = extra
        return kwargs

    async def _collect_streaming_response(
        self,
        stream: Any,
        on_chunk: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        """Collect an async streaming response into a unified response dict.

        Some providers (notably Gemini via LiteLLM) may return a streaming
        object even when ``stream=False``.  This helper iterates the stream
        and reassembles the chunks into the standard non-streaming format
        expected by :meth:`complete`.

        When *on_chunk* is provided, each text delta is forwarded to the
        callback in real time (for UI streaming).
        """
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        collected_tool_calls: dict[int, dict[str, Any]] = {}
        usage_obj: Any = None
        finish_reason = ""
        model = self.model

        try:
            async for chunk in stream:
                choices = _obj_get(chunk, "choices", [])
                if not choices:
                    continue
                first = choices[0]
                delta = _obj_get(first, "delta", {})

                _rc = (
                    _obj_get(delta, "reasoning_content", None)
                    or _obj_get(delta, "thinking", None)
                    or _obj_get(delta, "reasoning", None)
                )
                if _rc:
                    reasoning_parts.append(str(_rc))

                c = _obj_get(delta, "content", "")
                if c:
                    if isinstance(c, str):
                        content_parts.append(c)
                        if on_chunk:
                            try:
                                on_chunk(c)
                            except Exception:
                                pass
                    elif isinstance(c, list):
                        for part in c:
                            text = _obj_get(part, "text", "")
                            if text:
                                content_parts.append(str(text))
                                if on_chunk:
                                    try:
                                        on_chunk(str(text))
                                    except Exception:
                                        pass
                    else:
                        content_parts.append(str(c))
                        if on_chunk:
                            try:
                                on_chunk(str(c))
                            except Exception:
                                pass

                # Finish reason (last chunk wins)
                fr = _obj_get(first, "finish_reason", "")
                if fr:
                    finish_reason = str(fr)

                # Streamed tool calls — accumulated by index
                tc_list = _obj_get(delta, "tool_calls", []) or []
                for tc in tc_list:
                    idx = int(_obj_get(tc, "index", 0) or 0)
                    fn = _obj_get(tc, "function", {}) or {}
                    if idx not in collected_tool_calls:
                        collected_tool_calls[idx] = {
                            "id": str(_obj_get(tc, "id", f"call_{idx}") or f"call_{idx}"),
                            "function": {"name": "", "arguments": ""},
                        }
                    name = str(_obj_get(fn, "name", "") or "")
                    if name:
                        collected_tool_calls[idx]["function"]["name"] = name
                    args = str(_obj_get(fn, "arguments", "") or "")
                    if args:
                        collected_tool_calls[idx]["function"]["arguments"] += args
                        # Stream tool call argument deltas too — this
                        # covers long write/code-generation tool calls
                        # where the model spends most of its tokens on
                        # the tool arguments rather than text content.
                        if on_chunk:
                            try:
                                on_chunk(args)
                            except Exception:
                                pass

                # Usage (typically on the final chunk)
                u = _obj_get(chunk, "usage", None)
                if u is not None:
                    usage_obj = u

                m = str(_obj_get(chunk, "model", "") or "")
                if m:
                    model = m
        except Exception as e:
            # Preserve whatever we collected so far rather than losing
            # the entire response.  Log so we can diagnose.
            log.warning(
                "Stream collection interrupted, returning partial content",
                error=str(e),
                content_parts_collected=len(content_parts),
                provider=self.provider,
                model=self.model,
            )

        tc_out = (
            [collected_tool_calls[i] for i in sorted(collected_tool_calls)]
            if collected_tool_calls
            else None
        )
        joined_content = "".join(content_parts)
        # On a tool-call turn the visible answer is the tool call itself, so an
        # empty content is correct — don't let the reasoning-recovery fallback
        # surface the <think> block as content (it would leak the private chain-
        # of-thought and duplicate the reasoning_content we keep below).
        final_content = _strip_internal_context(
            _strip_reasoning_artifacts(joined_content, recover_answer=not bool(tc_out))
        )
        joined_reasoning = "".join(reasoning_parts).strip()
        # When the server streams the chain-of-thought inline in the content
        # (a <think>…</think> block) instead of as a separate reasoning_content
        # delta, we've stripped it from final_content above but must still keep
        # it so thinking-mode servers can round-trip it on the next turn (e.g.
        # NVIDIA Nemotron via an OpenAI-compatible endpoint, which otherwise
        # 400s with "reasoning_content ... must be passed back to the API").
        if not joined_reasoning:
            joined_reasoning = _recover_inline_reasoning(joined_content)
        if not final_content.strip() and joined_reasoning and not tc_out:
            fallback = _reasoning_content_fallback(joined_reasoning)
            log.warning(
                "Streamed content empty — recovering answer from reasoning",
                provider=self.provider,
                model=self.model,
                reasoning_chars=len(joined_reasoning),
                fallback_chars=len(fallback),
                recovered_json=fallback.startswith(("{", "[")),
            )
            final_content = fallback
        log.info(
            "Streamed LLM response",
            provider=self.provider,
            model=self.model,
            content_len=len(final_content),
            content_preview=final_content[:300],
            raw_content_len=len(joined_content),
            reasoning_len=len(joined_reasoning),
            tool_calls=len(collected_tool_calls),
            finish_reason=finish_reason,
        )
        return {
            "choices": [{
                "message": {
                    "content": final_content,
                    "tool_calls": tc_out,
                    # Preserved verbatim from the provider so the
                    # orchestration layer can replay it on the next
                    # turn. DeepSeek's strict API requires this.
                    "reasoning_content": joined_reasoning,
                },
                "finish_reason": finish_reason,
            }],
            "usage": usage_obj,
            "model": model,
        }

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        try:
            from litellm import acompletion
        except Exception as e:
            raise LLMError(f"LiteLLM is required for provider '{self.provider}': {e}")

        estimated = 0
        if self.rate_limiter:
            estimated = self._estimate_request_tokens(messages, max_tokens or self.max_tokens)
            await self.rate_limiter.acquire(estimated)

        try:
            kwargs = self._request_kwargs(
                messages=messages,
                tools=tools,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False,
            )

            if self.provider == "gemini":
                # Gemini via LiteLLM's acompletion() has persistent
                # issues: truncated content, unexpected streaming objects.
                # Use the synchronous completion() in a thread to ensure
                # we block until the full response is available.
                from litellm import completion as sync_completion
                import asyncio as _asyncio
                loop = _asyncio.get_event_loop()
                _gemini_timeout = 180  # seconds
                try:
                    response = await _asyncio.wait_for(
                        loop.run_in_executor(
                            None, lambda: sync_completion(**kwargs)
                        ),
                        timeout=_gemini_timeout,
                    )
                except _asyncio.TimeoutError:
                    raise LLMError(
                        f"Gemini completion timed out after {_gemini_timeout}s "
                        f"(model={self.model}). The model may be overloaded or "
                        f"the request may be too large."
                    )
            else:
                response = await _acompletion_tolerant(kwargs, self)

            # Safety fallback: if any provider still returns a streaming
            # object despite stream=False, collect it.
            if hasattr(response, "__aiter__"):
                log.warning(
                    "Provider returned streaming response with stream=False, "
                    "collecting full output",
                    provider=self.provider, model=self.model,
                )
                response = await self._collect_streaming_response(response)

            choices = _obj_get(response, "choices", [{}])
            if not choices:
                raise LLMError(
                    f"{self.provider} returned empty choices array "
                    f"(model={self.model})"
                )
            first_choice = choices[0]
            choice = _obj_get(first_choice, "message", {})
            try:
                _msg_dump = choice if isinstance(choice, dict) else getattr(choice, "model_dump", lambda: None)()
                if _msg_dump is None and hasattr(choice, "__dict__"):
                    _msg_dump = {k: v for k, v in vars(choice).items() if not k.startswith("_")}
                _msg_keys = list(_msg_dump.keys()) if isinstance(_msg_dump, dict) else []
            except Exception:
                _msg_dump = None
                _msg_keys = []
            log.info(
                "LLM raw message keys",
                provider=self.provider,
                model=self.model,
                keys=_msg_keys[:30],
                has_thinking=bool(_obj_get(choice, "thinking", None)),
                has_reasoning=bool(_obj_get(choice, "reasoning_content", None) or _obj_get(choice, "reasoning", None)),
            )
            raw_content = _obj_get(choice, "content", "") or ""
            if isinstance(raw_content, list):
                text_parts = []
                for part in raw_content:
                    t = _obj_get(part, "text", "")
                    if t:
                        text_parts.append(str(t))
                content = "".join(text_parts)
            else:
                content = str(raw_content) if raw_content else ""
            reasoning_content = (
                _obj_get(choice, "reasoning_content", None)
                or _obj_get(choice, "thinking", None)
                or _obj_get(choice, "reasoning", None)
            )
            _content_before_strip = content
            # A tool-call turn has no visible answer to recover — keep content
            # empty rather than surfacing the <think> block (see the streaming
            # collector for the same guard).
            _has_tool_calls = bool(_obj_get(choice, "tool_calls", []) or [])
            content = _strip_reasoning_artifacts(
                content, recover_answer=not _has_tool_calls
            )
            # Preserve inline <think>…</think> reasoning (emitted in the content
            # field rather than a separate reasoning_content field) so thinking-
            # mode servers can round-trip it on the next turn. See
            # _recover_inline_reasoning for the full rationale.
            if not (reasoning_content and str(reasoning_content).strip()):
                _inline_reasoning = _recover_inline_reasoning(_content_before_strip)
                if _inline_reasoning:
                    reasoning_content = _inline_reasoning
            if not content.strip() and reasoning_content:
                rc_str = str(reasoning_content)
                fallback = _reasoning_content_fallback(rc_str)
                log.warning(
                    "LLM content empty — recovering answer from reasoning_content",
                    provider=self.provider,
                    model=self.model,
                    reasoning_chars=len(rc_str),
                    fallback_chars=len(fallback),
                    recovered_json=fallback.startswith(("{", "[")),
                )
                content = fallback
            elif reasoning_content:
                log.debug(
                    "Discarding reasoning_content from LLM response",
                    provider=self.provider,
                    model=self.model,
                    reasoning_chars=len(str(reasoning_content)),
                )
            log.info(
                "LLM response content",
                provider=self.provider,
                model=self.model,
                content_len=len(content or ""),
                content_preview=str(content or "")[:300],
                raw_content_len=len(str(raw_content) if raw_content else ""),
                reasoning_len=len(str(reasoning_content)) if reasoning_content else 0,
            )
            finish_reason = str(_obj_get(first_choice, "finish_reason", "") or "")

            tool_calls: list[ToolCall] = []
            raw_calls = _obj_get(choice, "tool_calls", []) or []
            for idx, raw_call in enumerate(raw_calls, start=1):
                function = _obj_get(raw_call, "function", {}) or {}
                name = str(_obj_get(function, "name", "") or "")
                raw_args = _obj_get(function, "arguments", {})
                if isinstance(raw_args, str):
                    arguments = _safe_json_loads(raw_args)
                elif isinstance(raw_args, dict):
                    arguments = raw_args
                else:
                    arguments = {}
                tool_calls.append(ToolCall(
                    id=str(_obj_get(raw_call, "id", f"call_{idx}") or f"call_{idx}"),
                    name=name,
                    arguments=arguments,
                ))

            usage = _extract_usage(_obj_get(response, "usage", None))
            if self.rate_limiter:
                self.rate_limiter.record_actual(usage.get("total_tokens", 0), estimated)
            return LLMResponse(
                content=str(content),
                tool_calls=tool_calls,
                model=str(_obj_get(response, "model", self.model) or self.model),
                usage=usage,
                finish_reason=finish_reason,
                reasoning_content=str(reasoning_content or ""),
            )
        except Exception as e:
            status_code = _obj_get(e, "status_code", None)
            message = f"{self.provider} API call failed: {e}"
            # Log details for debugging vague provider errors.
            msg_count = len(kwargs.get("messages", []))
            total_chars = sum(len(str(m.get("content", ""))) for m in kwargs.get("messages", []))
            tool_count = len(kwargs.get("tools", []) or [])
            log.error(
                "LLM call failed",
                error=str(e),
                model=kwargs.get("model"),
                msg_count=msg_count,
                total_chars=total_chars,
                tool_count=tool_count,
                temperature=kwargs.get("temperature"),
            )
            if status_code is not None:
                raise LLMAPIError(message, status_code=int(status_code))
            raise LLMError(message)

    async def complete_streaming(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        try:
            from litellm import acompletion
        except Exception as e:
            raise LLMError(f"LiteLLM is required for provider '{self.provider}': {e}")

        if self.rate_limiter:
            estimated = self._estimate_request_tokens(messages, max_tokens or self.max_tokens)
            await self.rate_limiter.acquire(estimated)

        # Reasoning deltas don't get yielded as visible chunks (the
        # caller expects user-visible text), but we still capture
        # them so DeepSeek thinking mode can replay them on the next
        # turn. Stashed on the provider; callers read it after the
        # generator is exhausted via :attr:`last_reasoning_content`.
        _reasoning_parts: list[str] = []
        self.last_reasoning_content = ""

        try:
            stream = await _acompletion_tolerant(
                self._request_kwargs(
                    messages=messages,
                    tools=tools,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=True,
                ),
                self,
            )
            async for chunk in stream:
                delta_obj = _obj_get(_obj_get(chunk, "choices", [{}])[0], "delta", {})
                # Accumulate reasoning_content (Grok, DeepSeek, etc.)
                # without yielding it as user-visible text. Replayed
                # on the next turn so DeepSeek doesn't 400.
                _rc = (
                    _obj_get(delta_obj, "reasoning_content", None)
                    or _obj_get(delta_obj, "thinking", None)
                    or _obj_get(delta_obj, "reasoning", None)
                )
                if _rc:
                    _reasoning_parts.append(str(_rc))
                    continue
                delta = _obj_get(delta_obj, "content", "")
                if not delta:
                    continue
                if isinstance(delta, str):
                    yield delta
                    continue
                if isinstance(delta, list):
                    text_parts: list[str] = []
                    for part in delta:
                        text = _obj_get(part, "text", "")
                        if text:
                            text_parts.append(str(text))
                    if text_parts:
                        yield "".join(text_parts)
                    continue
                yield str(delta)
            # Persist the accumulated reasoning so the caller (e.g.
            # ``stream()`` in the orchestration layer) can stash it
            # on the just-produced assistant message.
            self.last_reasoning_content = "".join(_reasoning_parts).strip()
        except Exception as e:
            status_code = _obj_get(e, "status_code", None)
            message = f"{self.provider} streaming failed: {e}"
            if status_code is not None:
                raise LLMAPIError(message, status_code=int(status_code))
            raise LLMError(message)

    async def complete_with_callback(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        on_chunk: Callable[[str], None] | None = None,
    ) -> LLMResponse:
        """Stream completion with real-time text callback.

        Uses ``stream=True`` so text chunks are forwarded to *on_chunk*
        as they arrive, while still returning the full ``LLMResponse``
        with tool_calls, usage, and finish_reason.
        """
        if not on_chunk:
            return await self.complete(messages, tools, temperature, max_tokens)

        try:
            from litellm import acompletion
        except Exception as e:
            raise LLMError(f"LiteLLM is required for provider '{self.provider}': {e}")

        estimated = 0
        if self.rate_limiter:
            estimated = self._estimate_request_tokens(messages, max_tokens or self.max_tokens)
            await self.rate_limiter.acquire(estimated)

        try:
            kwargs = self._request_kwargs(
                messages=messages,
                tools=tools,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )
            # Request usage in the final stream chunk.
            kwargs["stream_options"] = {"include_usage": True}

            stream = await _acompletion_tolerant(kwargs, self)
            collected = await self._collect_streaming_response(stream, on_chunk=on_chunk)

            # Parse the collected dict into an LLMResponse (same as complete()).
            choices = _obj_get(collected, "choices", [{}])
            if not choices:
                raise LLMError(f"{self.provider} returned empty choices (streaming)")
            first_choice = choices[0]
            choice = _obj_get(first_choice, "message", {})
            raw_content = _obj_get(choice, "content", "") or ""
            if isinstance(raw_content, list):
                text_parts = []
                for part in raw_content:
                    t = _obj_get(part, "text", "")
                    if t:
                        text_parts.append(str(t))
                content = "".join(text_parts)
            else:
                content = str(raw_content) if raw_content else ""
            finish_reason = str(_obj_get(first_choice, "finish_reason", "") or "")

            tool_calls: list[ToolCall] = []
            raw_calls = _obj_get(choice, "tool_calls", []) or []
            for idx, raw_call in enumerate(raw_calls, start=1):
                function = _obj_get(raw_call, "function", {}) or {}
                name = str(_obj_get(function, "name", "") or "")
                raw_args = _obj_get(function, "arguments", {})
                if isinstance(raw_args, str):
                    arguments = _safe_json_loads(raw_args)
                elif isinstance(raw_args, dict):
                    arguments = raw_args
                else:
                    arguments = {}
                tool_calls.append(ToolCall(
                    id=str(_obj_get(raw_call, "id", f"call_{idx}") or f"call_{idx}"),
                    name=name,
                    arguments=arguments,
                ))

            usage = _extract_usage(_obj_get(collected, "usage", None))
            if self.rate_limiter:
                self.rate_limiter.record_actual(usage.get("total_tokens", 0), estimated)
            # Pull reasoning_content out of the collected message so
            # the caller can persist it for the next turn (required
            # by DeepSeek thinking mode, ignored by others).
            reasoning_collected = str(
                _obj_get(choice, "reasoning_content", "") or ""
            )
            return LLMResponse(
                content=str(content),
                tool_calls=tool_calls,
                model=str(_obj_get(collected, "model", self.model) or self.model),
                usage=usage,
                finish_reason=finish_reason,
                reasoning_content=reasoning_collected,
            )
        except Exception as e:
            status_code = _obj_get(e, "status_code", None)
            message = f"{self.provider} streaming callback failed: {e}"
            if status_code is not None:
                raise LLMAPIError(message, status_code=int(status_code))
            raise LLMError(message)

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        try:
            from litellm import token_counter

            return int(token_counter(model=self.model, messages=[{"role": "user", "content": text}]))
        except Exception:
            return len(text) // 4


def _litert_tool_fields(tool: Any) -> tuple[str, str, dict[str, Any]]:
    """Extract ``(name, description, parameters)`` from a tool of any shape.

    Captain Claw passes tools to providers in several shapes depending on
    the call site:
      - ``ToolDefinition`` dataclass instances
      - OpenAI-style dicts: ``{"type": "function", "function": {...}}``
      - Flat dicts: ``{"name": ..., "description": ..., "parameters": ...}``
    """
    if isinstance(tool, ToolDefinition):
        return tool.name, tool.description or "", tool.parameters or {}
    if isinstance(tool, dict):
        if isinstance(tool.get("function"), dict):
            fn = tool["function"]
            return (
                str(fn.get("name") or ""),
                str(fn.get("description") or ""),
                fn.get("parameters") or {},
            )
        return (
            str(tool.get("name") or ""),
            str(tool.get("description") or ""),
            tool.get("parameters") or {},
        )
    # Last-resort attribute lookup.
    return (
        str(getattr(tool, "name", "") or ""),
        str(getattr(tool, "description", "") or ""),
        getattr(tool, "parameters", {}) or {},
    )


def _litert_build_tool_manifest(tools: list[Any]) -> str:
    """Render a compact text manifest of tools for the system prompt.

    Local Gemma via litert-lm has no structured function-calling bridge,
    so we teach the model to emit calls as inline text using a fixed
    fence that we can parse out of the reply afterwards.
    """
    lines: list[str] = [
        "You can call tools. To call a tool, write EXACTLY this on its own line:",
        "<tool_call>{\"name\": \"tool_name\", \"arguments\": {\"arg\": \"value\"}}</tool_call>",
        "Use ONE call per turn. After the tool result is returned to you,",
        "continue the conversation. Do not invent tools — only use the ones listed.",
        "",
        "Available tools:",
    ]
    for t in tools:
        name, description, params = _litert_tool_fields(t)
        if not name:
            continue
        props = params.get("properties") if isinstance(params, dict) else None
        required = params.get("required") if isinstance(params, dict) else None
        arg_summary = ""
        if isinstance(props, dict) and props:
            arg_bits = []
            for arg_name, arg_schema in props.items():
                ty = ""
                if isinstance(arg_schema, dict):
                    ty = str(arg_schema.get("type") or "")
                req = (
                    isinstance(required, list) and arg_name in required
                )
                arg_bits.append(
                    f"{arg_name}{':' + ty if ty else ''}{'' if req else '?'}"
                )
            arg_summary = "(" + ", ".join(arg_bits) + ")"
        desc = description.strip().replace("\n", " ")
        if len(desc) > 200:
            desc = desc[:197] + "..."
        lines.append(f"- {name}{arg_summary} — {desc}")
    return "\n".join(lines)


# Patterns we accept for inline tool calls coming back from local Gemma:
#   1. Our preferred fence:  <tool_call>{...json...}</tool_call>
#   2. The model's habit:    <execute_tool_call> name(arg='value') </execute_tool_call>
#                            <execute_tool_call> web_tool_call: name(arg='value') </execute_tool_call>
#   3. Bare JSON-in-fence:   ```tool_call\n{...}\n```
#   4. Gemma-4 native template:
#         <|tool_call>call:name{key:<|"|>value<|"|>,n:42}<tool_call|>
#      where strings are wrapped in the literal delimiter ``<|"|>``.
_LITERT_TOOL_CALL_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL | re.IGNORECASE),
        "json",
    ),
    (
        re.compile(r"```tool_call\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE),
        "json",
    ),
    (
        re.compile(
            r"<execute_tool_call>\s*(?:web_tool_call:\s*)?([a-zA-Z_][\w\.]*)\s*\((.*?)\)\s*</execute_tool_call>",
            re.DOTALL | re.IGNORECASE,
        ),
        "pyish",
    ),
    (
        re.compile(
            r"<\|tool_call>\s*call:\s*([a-zA-Z_][\w\.]*)\s*\{(.*?)\}\s*<tool_call\|>",
            re.DOTALL,
        ),
        "gemma",
    ),
]


def _litert_parse_gemma_args(arg_str: str) -> dict[str, Any]:
    """Parse Gemma-4 native ``key:<|"|>value<|"|>,n:42`` argument strings.

    Strings are wrapped in the literal three-char delimiter ``<|"|>`` on
    both sides. Other scalars (ints, floats, bools, null) are bare.
    Nested objects/arrays use ``{}`` / ``[]`` and are kept as raw strings
    if encountered (best-effort).
    """
    out: dict[str, Any] = {}
    if not arg_str.strip():
        return out

    QUOTE = "<|\"|>"

    # Tokenise on top-level commas, respecting nested {}/[] and <|"|>...<|"|>.
    parts: list[str] = []
    depth = 0
    in_str = False
    i = 0
    buf: list[str] = []
    n = len(arg_str)
    while i < n:
        # Detect the literal quote delimiter <|"|>
        if arg_str.startswith(QUOTE, i):
            in_str = not in_str
            buf.append(QUOTE)
            i += len(QUOTE)
            continue
        ch = arg_str[i]
        if in_str:
            buf.append(ch)
            i += 1
            continue
        if ch in "{[":
            depth += 1
        elif ch in "}]":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(buf))
            buf = []
            i += 1
            continue
        buf.append(ch)
        i += 1
    if buf:
        parts.append("".join(buf))

    for part in parts:
        if ":" not in part:
            continue
        key, _, raw = part.partition(":")
        key = key.strip()
        raw = raw.strip()
        if not key:
            continue
        # String form: <|"|>...<|"|>
        if raw.startswith(QUOTE) and raw.endswith(QUOTE) and len(raw) >= 2 * len(QUOTE):
            out[key] = raw[len(QUOTE) : -len(QUOTE)]
            continue
        low = raw.lower()
        if low == "true":
            out[key] = True
            continue
        if low == "false":
            out[key] = False
            continue
        if low in ("null", "none"):
            out[key] = None
            continue
        try:
            out[key] = int(raw)
            continue
        except ValueError:
            pass
        try:
            out[key] = float(raw)
            continue
        except ValueError:
            pass
        # Strip any leftover Gemma quote markers we might have missed.
        out[key] = raw.replace(QUOTE, "")
    return out


def _litert_parse_pyish_args(arg_str: str) -> dict[str, Any]:
    """Parse ``key='value', key2=123`` style argument lists into a dict.

    Best-effort — we accept single or double quoted strings, bare ints
    and floats, and ``true``/``false``/``null``. Unparseable values fall
    back to the raw string.
    """
    out: dict[str, Any] = {}
    if not arg_str.strip():
        return out
    # Tokenise on top-level commas (don't split inside quotes/brackets).
    parts: list[str] = []
    depth = 0
    quote: str | None = None
    buf: list[str] = []
    for ch in arg_str:
        if quote:
            buf.append(ch)
            if ch == quote and (len(buf) < 2 or buf[-2] != "\\"):
                quote = None
            continue
        if ch in ("'", '"'):
            quote = ch
            buf.append(ch)
            continue
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(buf))
            buf = []
            continue
        buf.append(ch)
    if buf:
        parts.append("".join(buf))

    for part in parts:
        if "=" not in part:
            continue
        key, _, raw = part.partition("=")
        key = key.strip()
        raw = raw.strip()
        if not key:
            continue
        # Strip matching quotes.
        if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in ("'", '"'):
            out[key] = raw[1:-1]
            continue
        low = raw.lower()
        if low == "true":
            out[key] = True
            continue
        if low == "false":
            out[key] = False
            continue
        if low in ("null", "none"):
            out[key] = None
            continue
        try:
            out[key] = int(raw)
            continue
        except ValueError:
            pass
        try:
            out[key] = float(raw)
            continue
        except ValueError:
            pass
        out[key] = raw
    return out


def _litert_extract_tool_calls(text: str) -> tuple[str, list[ToolCall]]:
    """Scan ``text`` for inline tool-call fences and return ``(stripped_text, calls)``.

    The returned text has all matched fences removed so the user only
    sees the natural-language portion of the model's reply.
    """
    if not text:
        return text, []
    calls: list[ToolCall] = []
    cleaned = text
    counter = 0

    for pattern, kind in _LITERT_TOOL_CALL_PATTERNS:
        while True:
            m = pattern.search(cleaned)
            if not m:
                break
            try:
                if kind == "json":
                    payload = json.loads(m.group(1))
                    if isinstance(payload, dict):
                        name = str(payload.get("name") or payload.get("tool") or "").strip()
                        args = payload.get("arguments") or payload.get("args") or {}
                        if isinstance(args, str):
                            try:
                                args = json.loads(args)
                            except Exception:  # pylint: disable=broad-exception-caught
                                args = {"_raw": args}
                        if name:
                            counter += 1
                            calls.append(
                                ToolCall(
                                    id=f"litert_{counter}",
                                    name=name,
                                    arguments=args if isinstance(args, dict) else {},
                                )
                            )
                elif kind == "pyish":
                    name = m.group(1).strip()
                    args = _litert_parse_pyish_args(m.group(2) or "")
                    if name:
                        counter += 1
                        calls.append(
                            ToolCall(
                                id=f"litert_{counter}",
                                name=name,
                                arguments=args,
                            )
                        )
                elif kind == "gemma":
                    name = m.group(1).strip()
                    args = _litert_parse_gemma_args(m.group(2) or "")
                    if name:
                        counter += 1
                        calls.append(
                            ToolCall(
                                id=f"litert_{counter}",
                                name=name,
                                arguments=args,
                            )
                        )
            except Exception as exc:  # pylint: disable=broad-exception-caught
                log.warning("litert tool-call parse failed: %s", exc)
            # Drop this fence from the visible text either way.
            cleaned = cleaned[: m.start()] + cleaned[m.end():]

    return cleaned.strip(), calls


class LiteRTProvider(LLMProvider):
    """Local provider backed by Google's litert-lm runtime.

    Loads a ``.litertlm`` model file once at construction and serves chat
    requests in-process. Each request creates a fresh ``Conversation`` with
    the prior history as preface and sends only the latest user message —
    Captain Claw is stateless from the model's perspective (it always
    passes the full message list), and litert-lm's ``send_message`` API
    expects exactly one message at a time.

    **Tool calling is not exposed.** litert-lm's ``Conversation`` API does
    accept Python tool functions, but Captain Claw passes JSON-Schema
    ``ToolDefinition`` objects which would need an inverse bridge. For
    now any ``tools=`` argument is logged and ignored. Local Gemma models
    are best used as council members or chat companions where tools
    aren't needed.

    **Concurrency.** The underlying engine is single-threaded, so an
    ``asyncio.Lock`` serializes ``complete()`` and ``complete_streaming()``
    across the process. Streaming runs the sync iterator in a worker
    thread and pipes chunks back through an ``asyncio.Queue``.

    **Model resolution.** ``model_path`` may be either an absolute path to
    a ``.litertlm`` file or a model id like
    ``litert-community/gemma-4-E4B-it-litert-lm`` — in the latter case the
    model must already be present at
    ``~/.litert-lm/models/<repo>--<name>/model.litertlm`` (run
    ``litert-lm import <id>`` once first, or run the upstream
    ``litert-lm run --from-huggingface-repo=<id>`` once to download).
    """

    def __init__(
        self,
        model: str,
        model_path: str | None = None,
        backend: str = "gpu",
        temperature: float = 0.7,  # noqa: ARG002 — kept for API parity
        max_tokens: int = 4096,
        max_num_tokens: int = 16384,
        tokens_per_minute: int = 0,
    ):
        # The provider only validates the model path here. The actual
        # ``litert_lm.Engine`` is owned by a dedicated subprocess that
        # is spawned lazily on the first ``complete()`` call by the
        # shared worker client. Running the engine out-of-process
        # isolates two C++-side failure modes that previously took the
        # whole agent down: KV-cache overflow on long conversations
        # and Metal/GPU context exhaustion. See
        # ``captain_claw/llm/litert_worker.py``.

        self.provider = "litert"
        self.model = model
        # Expose the original reference (HF id or absolute path) as
        # ``base_url`` so the agent's "is current provider already this
        # one?" comparison can short-circuit and skip rebuilding the
        # engine on session-load.
        self.base_url = (model_path or "").strip()
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Resolve the model file. Order of precedence:
        #   1. If `ref` is an existing file path, use it directly.
        #   2. Try the litert-lm CLI layout (~/.litert-lm/models/<id>/model.litertlm).
        #   3. Try the Hugging Face hub cache populated by
        #      `litert-lm run --from-huggingface-repo=<id>` — that path is
        #      ~/.cache/huggingface/hub/models--<owner>--<repo>/snapshots/<rev>/<file>.litertlm.
        ref = model_path or model
        resolved_path = self._resolve_model_file(ref)
        if not resolved_path:
            raise LLMError(
                f"litert model file not found for reference '{ref}'. "
                "Pass an absolute path to a .litertlm file in `base_url`, or run "
                f"`litert-lm run --from-huggingface-repo={ref} <file>.litertlm` "
                "once to download it first."
            )
        self.model_path = resolved_path

        self._backend = (backend or "gpu").strip().lower()
        # Honour the caller's chosen window verbatim. ``create_provider``
        # is responsible for capping this at the .litertlm file's real
        # KV limit (see ``LITERT_MAX_NUM_TOKENS``); the previous
        # ``max(max_num_tokens, max_tokens, 4096)`` floor was leaking
        # the global 160k context budget into the engine and either
        # getting silently clamped or hanging on VRAM pressure.
        self._max_num_tokens = max(int(max_num_tokens or 0), 1024)

        # Acquire (or build) the shared worker client for this
        # (path, backend, max_num_tokens) triple. This does NOT spawn
        # the child yet — that happens on the first send_message call,
        # so importing this provider is cheap and side-effect free.
        from captain_claw.llm.litert_worker import get_or_create_litert_worker

        self._client = get_or_create_litert_worker(
            model_path=self.model_path,
            backend=self._backend,
            max_num_tokens=self._max_num_tokens,
            recycle_after_each=True,
        )

        log.info(
            "LiteRTProvider bound to worker client",
            model=self.model,
            path=self.model_path,
            backend=self._backend,
            max_num_tokens=self._max_num_tokens,
        )

        self.rate_limiter = (
            TokenRateLimiter(tokens_per_minute) if tokens_per_minute > 0 else None
        )

    @staticmethod
    def _resolve_model_file(ref: str) -> str | None:
        """Resolve a model reference to an absolute .litertlm file path."""
        if not ref:
            return None

        # 1. Direct file path.
        if os.path.isfile(ref):
            return os.path.abspath(ref)

        # 2. litert-lm CLI imported-models layout.
        try:
            from litert_lm_cli.model import Model as _CliModel  # type: ignore[import-not-found]

            cli_model = _CliModel.from_model_reference(ref)
            if os.path.isfile(cli_model.model_path):
                return cli_model.model_path
        except ImportError:
            pass

        # 3. Hugging Face hub cache. The reference is expected to look
        # like "owner/repo" (with optional ":filename" suffix). The cache
        # directory follows the form
        # ~/.cache/huggingface/hub/models--<owner>--<repo>/snapshots/<rev>/.
        # IMPORTANT: we return the SYMLINK path inside snapshots/, not
        # the resolved blob — litert-lm relies on the .litertlm suffix
        # for format detection, and the blob filename is a bare sha256
        # without an extension.
        if "/" in ref:
            spec, _, hint_filename = ref.partition(":")
            owner_repo = spec.replace("/", "--")
            # Flight Deck overrides $HOME for process agents, so
            # expanduser("~") may point to the agent sandbox rather
            # than the real user home.  Check both the $HOME-derived
            # path AND the real home (from /etc/passwd / pw_dir) so
            # we find models downloaded by the user on the host.
            _candidate_homes: list[str] = []
            _env_hf = os.environ.get("HF_HOME")
            if _env_hf:
                _candidate_homes.append(_env_hf)
            _candidate_homes.append(
                os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
            )
            try:
                import pwd
                _real_home = pwd.getpwuid(os.getuid()).pw_dir
                _real_hf = os.path.join(_real_home, ".cache", "huggingface")
                if _real_hf not in _candidate_homes:
                    _candidate_homes.append(_real_hf)
            except Exception:
                pass
            try:
                for hf_home in _candidate_homes:
                    hub_dir = os.path.join(hf_home, "hub", f"models--{owner_repo}", "snapshots")
                    if not os.path.isdir(hub_dir):
                        continue
                    candidates: list[str] = []
                    for snap in sorted(os.listdir(hub_dir)):
                        snap_dir = os.path.join(hub_dir, snap)
                        if not os.path.isdir(snap_dir):
                            continue
                        for fname in sorted(os.listdir(snap_dir)):
                            if not fname.endswith(".litertlm"):
                                continue
                            if hint_filename and fname != hint_filename:
                                continue
                            full = os.path.join(snap_dir, fname)
                            # `os.path.isfile` follows the symlink for us;
                            # we keep the symlink path so the .litertlm
                            # suffix is preserved when litert-lm opens it.
                            if os.path.isfile(full):
                                candidates.append(full)
                    if candidates:
                        # Prefer the lexicographically last (newest snapshot
                        # rev sorted, last file alphabetically).
                        return candidates[-1]
            except OSError:
                pass

        return None

    @staticmethod
    def _to_litert_msg(msg: Message) -> dict[str, Any]:
        """Convert a Captain Claw Message to litert-lm's expected dict shape."""
        role = msg.role or "user"
        # litert-lm's chat template expects content as a list of typed parts.
        return {
            "role": role,
            "content": [{"type": "text", "text": msg.content or ""}],
        }

    def _split_history(self, messages: list[Message]) -> tuple[list[dict[str, Any]], str]:
        """Return ``(preface, last_user_text)`` for the conversation.

        The preface is everything up to (but not including) the trailing
        user turn; the trailing user turn's text becomes the
        ``send_message`` argument. Anything after the last user turn (e.g.
        a stale assistant draft) is dropped — the new response replaces it.
        """
        if not messages:
            return [], ""
        last_user_idx = -1
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].role == "user":
                last_user_idx = i
                break
        if last_user_idx < 0:
            # No user turn at all — feed everything as preface and send
            # an empty prompt. Unusual, but the model will produce
            # something deterministic.
            return [self._to_litert_msg(m) for m in messages], ""
        preface = [self._to_litert_msg(m) for m in messages[:last_user_idx]]
        last_text = messages[last_user_idx].content or ""
        return preface, last_text

    def _build_preface(
        self,
        messages: list[Message],
        tool_manifest: str = "",
    ) -> tuple[list[dict[str, Any]], str]:
        """Split history into (preface, last_user) and inject the manifest.

        The actual ``send_message`` call is performed by the worker
        subprocess via ``self._client.send_message(...)``.
        """
        preface, last_user = self._split_history(messages)
        if tool_manifest:
            # Inject the manifest as a synthetic system turn at the very
            # top of the preface so it's always in scope.
            preface = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": tool_manifest}],
                }
            ] + preface
        return preface, last_user

    # ------------------------------------------------------------------
    # CLI fallback (current default)
    # ------------------------------------------------------------------
    #
    # The Python ``litert_lm.Engine`` path still exists above as the
    # subprocess worker client, but right now we deliberately bypass it
    # and shell out to the upstream ``litert-lm run`` CLI for every
    # call. Each invocation pays the full model load cost (~5–15s on
    # warm OS page cache) but starts from a completely fresh process,
    # which sidesteps every state-leak / GPU-context / KV-cache issue
    # we've been chasing. The trade-off is latency, not correctness.
    #
    # The CLI command we run mirrors what the user verified manually:
    #
    #     litert-lm run \
    #         --from-huggingface-repo=<repo> <file> \
    #         --backend=gpu --prompt="<text>"
    #
    # ``LITERT_HF_REPO`` / ``LITERT_HF_FILE`` / ``LITERT_CLI_BIN`` /
    # ``LITERT_CLI_TIMEOUT_SECONDS`` override the defaults.

    def _build_cli_prompt(
        self,
        messages: list[Message],
        manifest: str = "",
    ) -> str:
        """Flatten the conversation into a single string for ``--prompt``.

        Format is intentionally minimal — role label + content per turn,
        blank line between turns. The CLI applies its own chat template
        on top of this so we don't need Gemma-specific markers.
        """
        parts: list[str] = []
        if manifest:
            parts.append(f"System:\n{manifest}")
        for msg in messages:
            role = (msg.role or "user").strip().lower()
            text = (msg.content or "").strip()
            if not text:
                continue
            label = {
                "system": "System",
                "user": "User",
                "assistant": "Assistant",
                "tool": "Tool",
            }.get(role, role.capitalize())
            parts.append(f"{label}:\n{text}")
        return "\n\n".join(parts)

    async def _complete_via_cli(
        self,
        messages: list[Message],
        manifest: str = "",
    ) -> str:
        """Run ``litert-lm run`` as a one-shot subprocess and return stdout.

        Returns the raw stdout text. Caller is responsible for
        post-processing (reasoning-artifact stripping, tool-call
        extraction, etc.). Raises :class:`LLMError` on non-zero exit
        or timeout.
        """
        prompt = self._build_cli_prompt(messages, manifest)

        # ── Pre-flight prune ───────────────────────────────────────
        # The .litertlm file we ship has an 8k KV cache, and the CLI
        # segfaults (SIGSEGV, not a clean error) on prompts much over
        # ~25 KB. Enforce a hard character budget here by dropping
        # middle conversation turns (oldest first, keeping the system
        # preamble + the most recent turns) until we fit. This is a
        # last-line safety net; ideally the agent's context manager
        # wouldn't send us 50 KB in the first place.
        try:
            budget_chars = int(
                os.getenv("LITERT_PROMPT_BUDGET_CHARS", "24000") or 24000
            )
        except ValueError:
            budget_chars = 24000

        if budget_chars > 0 and len(prompt) > budget_chars:
            # Split messages into system (kept) vs. conversation (prunable).
            keep_system: list[Message] = []
            convo: list[Message] = []
            for m in messages:
                if (m.role or "").strip().lower() == "system" and not convo:
                    keep_system.append(m)
                else:
                    convo.append(m)

            before_len = len(prompt)
            before_msgs = len(messages)
            dropped = 0
            # Drop oldest conversation turns one at a time, always
            # preserving the most recent user/assistant turn so the
            # model still has something to respond to.
            while convo and len(prompt) > budget_chars and len(convo) > 1:
                convo.pop(0)
                dropped += 1
                prompt = self._build_cli_prompt(
                    keep_system + convo, manifest
                )

            log.warning(
                "litert prompt pruned",
                before_len=before_len,
                after_len=len(prompt),
                before_msgs=before_msgs,
                after_msgs=len(keep_system) + len(convo),
                dropped=dropped,
                budget_chars=budget_chars,
            )

            # If we're *still* over budget after dropping everything
            # prunable, the system preamble alone is too big. Log
            # loudly and let the CLI try — it may segfault, but at
            # least we'll see it in the logs.
            if len(prompt) > budget_chars:
                log.error(
                    "litert prompt still over budget after pruning",
                    final_len=len(prompt),
                    budget_chars=budget_chars,
                    hint=(
                        "System preamble + manifest exceed the budget. "
                        "Reduce tool manifest or system prompt."
                    ),
                )

        # Dump the exact prompt to disk so we can re-run the failing
        # call manually with ``litert-lm run … --prompt="$(cat …)"``
        # without having to reconstruct it. Overwritten on every call;
        # the most recent prompt always wins.
        try:
            with open("/tmp/big_prompt.txt", "w", encoding="utf-8") as _f:
                _f.write(prompt)
        except Exception as _dump_err:  # pylint: disable=broad-exception-caught
            log.warning(
                "Failed to dump litert prompt to /tmp/big_prompt.txt",
                error=str(_dump_err),
            )

        # Default to the ``litert-lm`` script next to the running
        # Python interpreter. When Captain Claw runs from a venv,
        # ``sys.executable`` is ``<venv>/bin/python`` and the CLI is
        # ``<venv>/bin/litert-lm`` — looking it up via PATH would miss
        # it because nothing puts the venv bin on PATH globally.
        import sys as _sys

        default_cli = os.path.join(
            os.path.dirname(_sys.executable), "litert-lm"
        )
        if not os.path.isfile(default_cli):
            default_cli = "litert-lm"
        cli_bin = os.getenv("LITERT_CLI_BIN", default_cli)
        hf_repo = os.getenv(
            "LITERT_HF_REPO",
            "litert-community/gemma-4-E4B-it-litert-lm",
        )
        hf_file = os.getenv(
            "LITERT_HF_FILE",
            os.path.basename(self.model_path) or "gemma-4-E4B-it.litertlm",
        )
        timeout_s = float(
            os.getenv("LITERT_CLI_TIMEOUT_SECONDS", "300") or 300
        )

        cmd = [
            cli_bin,
            "run",
            f"--from-huggingface-repo={hf_repo}",
            hf_file,
            f"--backend={self._backend}",
            f"--prompt={prompt}",
        ]

        log.info(
            "Running litert-lm CLI",
            bin=cli_bin,
            backend=self._backend,
            hf_repo=hf_repo,
            hf_file=hf_file,
            prompt_len=len(prompt),
            timeout_s=timeout_s,
        )

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError as e:
            raise LLMError(
                f"litert-lm CLI not found at '{cli_bin}'. Set LITERT_CLI_BIN "
                f"or install litert-lm. ({e})"
            ) from e

        # Stream stdout/stderr line-by-line so the parent's log shows
        # progress in real time instead of waiting for the subprocess
        # to exit. This is the only way to know the CLI is alive when
        # generation is taking 30+ seconds.
        stdout_chunks: list[str] = []
        stderr_chunks: list[str] = []

        async def _drain(
            stream: asyncio.StreamReader | None,
            sink: list[str],
            label: str,
        ) -> None:
            if stream is None:
                return
            while True:
                try:
                    line = await stream.readline()
                except Exception as e:  # pylint: disable=broad-exception-caught
                    log.warning(
                        "litert-lm CLI stream read failed",
                        stream=label,
                        error=str(e),
                    )
                    break
                if not line:
                    break
                text = line.decode(errors="replace")
                sink.append(text)
                stripped = text.rstrip("\n")
                if stripped:
                    log.info(
                        "litert-lm CLI",
                        stream=label,
                        line=stripped,
                    )

        drain_out = asyncio.create_task(_drain(proc.stdout, stdout_chunks, "stdout"))
        drain_err = asyncio.create_task(_drain(proc.stderr, stderr_chunks, "stderr"))

        try:
            await asyncio.wait_for(proc.wait(), timeout=timeout_s)
        except asyncio.TimeoutError as e:
            log.error(
                "litert-lm CLI timed out — killing subprocess",
                timeout_s=timeout_s,
                stdout_so_far=len("".join(stdout_chunks)),
                stderr_so_far=len("".join(stderr_chunks)),
            )
            try:
                proc.kill()
            except Exception:  # pylint: disable=broad-exception-caught
                pass
            try:
                await asyncio.wait_for(proc.wait(), timeout=5)
            except Exception:  # pylint: disable=broad-exception-caught
                pass
            # Cancel the drain tasks so they don't leak.
            for t in (drain_out, drain_err):
                if not t.done():
                    t.cancel()
            raise LLMError(
                f"litert-lm CLI timed out after {timeout_s}s"
            ) from e

        # Make sure the drain tasks finish reading whatever is left in
        # the pipes before we return.
        try:
            await asyncio.wait_for(
                asyncio.gather(drain_out, drain_err, return_exceptions=True),
                timeout=5,
            )
        except Exception:  # pylint: disable=broad-exception-caught
            for t in (drain_out, drain_err):
                if not t.done():
                    t.cancel()

        stdout_text = "".join(stdout_chunks)
        stderr_text = "".join(stderr_chunks)

        log.info(
            "litert-lm CLI finished",
            returncode=proc.returncode,
            stdout_len=len(stdout_text),
            stderr_len=len(stderr_text),
        )

        if proc.returncode != 0:
            tail = stderr_text[-800:] if stderr_text else "(no stderr)"
            raise LLMError(
                f"litert-lm CLI exited with code {proc.returncode}: {tail}"
            )

        return stdout_text

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        temperature: float | None = None,  # noqa: ARG002
        max_tokens: int | None = None,
    ) -> LLMResponse:
        manifest = _litert_build_tool_manifest(list(tools)) if tools else ""
        if self.rate_limiter:
            estimated = self._estimate_request_tokens(
                messages, max_tokens or self.max_tokens
            )
            await self.rate_limiter.acquire(estimated)

        preface, last_user = self._build_preface(messages, manifest)
        try:
            content = await self._client.send_message(preface, last_user)
        except Exception as e:  # pylint: disable=broad-exception-caught
            err_text = str(e)
            err_type = type(e).__name__
            log.error(
                "litert-lm complete() failed; returning graceful response",
                error=err_text,
                error_type=err_type,
            )
            friendly = self._friendly_litert_error(err_text, err_type)
            return LLMResponse(
                content=friendly,
                tool_calls=[],
                model=self.model,
                usage={
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
                finish_reason="length",
            )

        content = _strip_reasoning_artifacts(content)
        tool_calls: list[ToolCall] = []
        if tools:
            content, tool_calls = _litert_extract_tool_calls(content)
        completion_tokens = self.count_tokens(content)
        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            model=self.model,
            usage={
                "prompt_tokens": 0,
                "completion_tokens": completion_tokens,
                "total_tokens": completion_tokens,
            },
            finish_reason="tool_calls" if tool_calls else "stop",
        )

    async def complete_streaming(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        temperature: float | None = None,  # noqa: ARG002
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        manifest = _litert_build_tool_manifest(list(tools)) if tools else ""
        if self.rate_limiter:
            estimated = self._estimate_request_tokens(
                messages, max_tokens or self.max_tokens
            )
            await self.rate_limiter.acquire(estimated)

        # The worker client doesn't expose true token streaming yet, so
        # we just run one blocking ``send_message`` and yield the full
        # answer as a single chunk. ``_complete_via_cli`` is still below
        # as an emergency fallback but is no longer used.
        preface, last_user = self._build_preface(messages, manifest)
        try:
            content = await self._client.send_message(preface, last_user)
        except Exception as e:  # pylint: disable=broad-exception-caught
            err_text = str(e)
            err_type = type(e).__name__
            log.error(
                "litert-lm complete_streaming() failed; yielding graceful response",
                error=err_text,
                error_type=err_type,
            )
            yield self._friendly_litert_error(err_text, err_type)
            return

        content = _strip_reasoning_artifacts(content)
        if tools:
            # Strip the inline tool-call fences before streaming the
            # remainder to the user — the agent loop will run the actual
            # call via complete() on its own (it always re-issues with
            # tools when finish_reason == "tool_calls").
            content, _ = _litert_extract_tool_calls(content)
        if content:
            yield content

    def count_tokens(self, text: str) -> int:
        return len(text) // 4

    @staticmethod
    def _friendly_litert_error(err_text: str, err_type: str) -> str:
        """Turn a worker-client error into a user-visible message.

        Recognises three buckets:
          1. **Timeout / crash** — the worker hung or died (most common
             cause: KV-cache overflow that aborts the C++ side). The
             worker has already been killed; the next call will respawn
             a fresh child.
          2. **Overflow signalled cleanly** — the worker raised a
             readable error mentioning ``cache``/``length``/etc.
          3. **Anything else** — surfaced verbatim.
        """
        lower = (err_text or "").lower()

        # Bucket 1: timeout / crash from the worker client.
        if (
            "did not respond" in lower
            or "did not boot" in lower
            or "worker boot failed" in lower
            or "unexpected response from worker" in lower
            or "failed to spawn worker" in lower
            or "failed to enqueue request" in lower
        ):
            return (
                "[litert] Local model worker crashed or hung "
                "(usually KV-cache overflow on long conversations — "
                "Gemma-3n previews are capped at ~8192 tokens by the "
                ".litertlm file). A fresh worker will be spawned on "
                "the next message; use /clear or trim the conversation "
                "to stay under the limit."
            )

        # Bucket 2: overflow surfaced cleanly by the worker.
        looks_like_overflow = any(
            tok in lower
            for tok in (
                "kv",
                "cache",
                "max_num_tokens",
                "max_seq",
                "out of range",
                "exceed",
                "too long",
                "length",
                "capacity",
            )
        )
        if looks_like_overflow:
            return (
                "[litert] Local model ran out of context window "
                "(KV cache exhausted — Gemma-3n previews are capped at "
                "~8192 tokens by the .litertlm file). Use /clear or trim "
                "the conversation and try again."
            )

        return (
            f"[litert] Local model call failed ({err_type}): {err_text}. "
            "The conversation was preserved — try again, or /clear if it "
            "keeps failing."
        )

    async def close(self) -> None:
        # The worker client is shared across providers via the registry
        # and the child is daemonized, so it will die with the parent.
        # We deliberately do NOT call ``self._client.shutdown()`` here:
        # another provider in the same process may still be using it.
        return


class BrowserProvider(LLMProvider):
    """Completions served by a browser tab registered as an inference worker.

    The loop (and every tool) runs server-side; the tab only turns
    (messages, schema) into tokens via WebLLM/WebGPU, brokered by Flight
    Deck's ``/fd/infer/complete`` (see flight_deck/infer_broker.py). The
    prod host needs no GPU — whoever has the Flight Deck tab open with
    Local inference enabled donates the compute.

    ``tools`` are ignored by design: the only current caller is the Mrav
    micro runtime, which speaks a text protocol with grammar-constrained
    JSON (``response_schema`` → xgrammar in the tab). A stable per-instance
    ``session_key`` keeps one agent pinned to one tab so WebLLM's KV
    delta-prefill turns 10-40s cold prefills into ~1-3s steps.
    """

    def __init__(
        self,
        model: str = "browser-auto",
        base_url: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        tokens_per_minute: int = 0,
    ):
        from captain_claw.fd_client import flight_deck_base

        self.provider = "browser"
        self.model = model
        self.base_url = (base_url or flight_deck_base() or "http://127.0.0.1:25080").rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.session_key = uuid.uuid4().hex[:16]
        # Read timeout must survive a cold in-tab prefill (10-40s on good
        # GPUs, worse on iGPUs) plus generation; the broker's own job
        # timeout (240s) is the real ceiling.
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=10.0, read=330.0, write=30.0, pool=30.0),
            follow_redirects=True,
        )
        self.rate_limiter = TokenRateLimiter(tokens_per_minute) if tokens_per_minute > 0 else None

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_schema: dict[str, Any] | None = None,
    ) -> LLMResponse:
        from captain_claw.fd_client import flight_deck_headers

        body: dict[str, Any] = {
            "messages": [
                {"role": m.role, "content": m.content or ""} for m in messages
            ],
            "max_tokens": max_tokens or self.max_tokens,
            "temperature": self.temperature if temperature is None else temperature,
            "session_key": self.session_key,
        }
        if response_schema:
            body["response_schema"] = response_schema

        try:
            response = await self.client.post(
                f"{self.base_url}/fd/infer/complete",
                json=body,
                headers=flight_deck_headers(),
            )
        except httpx.HTTPError as exc:
            raise LLMAPIError(f"Browser inference broker unreachable: {exc}")
        if response.status_code == 503:
            raise LLMAPIError(
                "No browser inference worker online — open Flight Deck and "
                "enable Local inference in a tab, or switch this agent's "
                "provider to ollama.",
                status_code=503,
            )
        if not response.is_success:
            # Name the host we actually called — the classic footgun is a
            # stale tier Base URL (e.g. a leftover cloud-API endpoint), and
            # a foreign 401/404 is unreadable without it.
            raise LLMAPIError(
                f"Browser inference error {response.status_code} from "
                f"{self.base_url}: {response.text[:300]} — the broker URL "
                "must be the agent's Flight Deck (leave the tier's Base URL "
                "empty unless FD runs elsewhere)",
                status_code=response.status_code,
            )
        data = response.json()
        return LLMResponse(
            content=str(data.get("content") or ""),
            model=str(data.get("model") or self.model),
            usage=data.get("usage") or {},
            finish_reason=str(data.get("finish_reason") or ""),
        )

    async def complete_structured(
        self,
        messages: list[Message],
        response_schema: dict[str, Any],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Grammar-enforced JSON via xgrammar inside the tab's WebLLM engine."""
        return await self.complete(
            messages, None, temperature, max_tokens, response_schema=response_schema
        )

    async def complete_streaming(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        # v1 brokers whole completions; callers chunk for display themselves.
        response = await self.complete(messages, tools, temperature, max_tokens)
        if response.content:
            yield response.content

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)


def create_provider(
    provider: str = "ollama",
    model: str = "llama3.2",
    api_key: str | None = None,
    base_url: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 32000,
    num_ctx: int = 160000,
    tokens_per_minute: int = 0,
    extra_headers: dict[str, str] | None = None,
    think: bool | str | None = None,
) -> LLMProvider:
    """Create an LLM provider.

    Supported providers:
    - `ollama`
    - `openai` / `chatgpt`
    - `anthropic` / `claude` — pay-per-token via the Messages API.
    - `claude-cli` / `claude-subscription` — Anthropic **subscription**
      (Pro/Max) billed through the local ``claude`` CLI. Text generation
      only (no tool calling); requires ``claude login`` /
      ``claude setup-token``. See :class:`ClaudeCLIProvider`.
    - `gemini` / `google`
    - `grok` / `xai`
    - `openrouter`
    - `litert` / `litert-lm` — local Gemma via Google's litert-lm runtime.
      Pass the model id (e.g. ``litert-community/gemma-4-E4B-it-litert-lm``)
      or an absolute path to a ``.litertlm`` file in ``base_url``. The
      model must already be present at
      ``~/.litert-lm/models/<repo>--<name>/model.litertlm``. Set
      ``LITERT_BACKEND=cpu`` env var to force CPU; defaults to GPU.
    """
    normalized = _normalize_provider_name(provider)

    if normalized == "browser":
        # Browser-tab inference worker via the Flight Deck broker; base_url
        # defaults to FD_URL (agents spawned by FD have it injected).
        return BrowserProvider(
            model=model,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            tokens_per_minute=tokens_per_minute,
        )

    if normalized == "ollama":
        return OllamaProvider(
            model=model,
            base_url=(base_url or os.getenv("OLLAMA_BASE_URL") or OLLAMA_NATIVE_BASE_URL),
            temperature=temperature,
            max_tokens=max_tokens,
            num_ctx=num_ctx,
            api_key=api_key,
            tokens_per_minute=tokens_per_minute,
            think=think,
        )

    # ChatGPT Responses API path — activated when the OpenAI provider
    # has explicit ``extra_headers`` configured OR the selected model
    # is a Codex-family model (which can only be served by the ChatGPT
    # Responses endpoint, never by the regular api.openai.com/v1).
    # In the second case the ``CodexAuthManager`` will resolve the
    # Authorization + chatgpt-account-id headers at call time from
    # Flight Deck (``/fd/codex/access_token``) or ``~/.codex/auth.json``.
    if normalized == "openai" and (bool(extra_headers) or _is_codex_family_model(model)):
        return ChatGPTResponsesProvider(
            model=model,
            base_url=base_url or "https://chatgpt.com/backend-api/codex/responses",
            extra_headers=extra_headers,
            temperature=temperature,
            max_tokens=max_tokens,
            tokens_per_minute=tokens_per_minute,
        )

    if normalized == "claude-cli":
        # Anthropic **subscription** (Pro/Max) via the ``claude`` CLI
        # subprocess — text generation only, no metered API key. Auth comes
        # from the ambient ``claude login`` (or CLAUDE_CODE_OAUTH_TOKEN).
        return ClaudeCLIProvider(
            model=model,
            base_url=base_url,          # optional explicit ``claude`` binary path
            temperature=temperature,
            max_tokens=max_tokens,
            tokens_per_minute=tokens_per_minute,
        )

    if normalized in {"openai", "anthropic", "gemini", "xai", "openrouter"}:
        return LiteLLMProvider(
            provider=normalized,
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            tokens_per_minute=tokens_per_minute,
            extra_headers=extra_headers,
        )

    if normalized == "litert":
        # `base_url` doubles as the model file path / model id resolver
        # for the local litert-lm runtime, since this provider has no
        # network endpoint. Falls back to `model` if base_url is empty.
        #
        # IMPORTANT: ``max_num_tokens`` is NOT the same thing as
        # Captain Claw's global ``num_ctx``. It's the *combined*
        # prompt+output KV-cache working window allocated at engine
        # construction time. The upstream Gemma-4 E4B .litertlm file
        # supports up to 32k tokens (per the HF model card:
        # "The model can support up to 32k context length."), but the
        # ``Engine(max_num_tokens=...)`` default is only 4096 — if we
        # don't override it, the engine silently clamps at 4k and then
        # segfaults the moment we feed it a larger prompt.
        #
        # We previously capped this at 8192 while chasing a different
        # bug where Captain Claw's 160k global context budget was
        # leaking into the engine. That cap was the wrong fix: 8k is
        # too small for real multi-round council sessions, and when a
        # ~12k prompt hit the engine with only 8k allocated it blew
        # past the buffer → SIGSEGV. Use the full 32k the model
        # supports and let the agent's own context manager handle
        # higher-level pruning. Override via ``LITERT_MAX_NUM_TOKENS``
        # if you're running a differently-built file.
        litert_max_num_tokens = int(
            os.getenv("LITERT_MAX_NUM_TOKENS", "32768") or 32768
        )
        return LiteRTProvider(
            model=model,
            model_path=(base_url or None),
            backend=os.getenv("LITERT_BACKEND", "gpu"),
            temperature=temperature,
            max_tokens=max_tokens,
            max_num_tokens=litert_max_num_tokens,
            tokens_per_minute=tokens_per_minute,
        )

    raise ValueError(
        f"Provider '{provider}' not supported. "
        "Use one of: ollama, openai/chatgpt, anthropic/claude, "
        "claude-cli/claude-subscription, gemini/google, "
        "grok/xai, openrouter, litert."
    )


# Global provider instance
_provider: LLMProvider | None = None


def get_provider() -> LLMProvider:
    """Get the global LLM provider instance."""
    global _provider
    if _provider is None:
        from captain_claw.config import get_config

        cfg = get_config()
        normalized = _normalize_provider_name(cfg.model.provider)
        headers = cfg.provider_keys.headers_for(normalized) or None
        api_key = None if headers else (cfg.model.api_key or None)
        _provider = create_provider(
            provider=cfg.model.provider,
            model=cfg.model.model,
            temperature=cfg.model.temperature,
            max_tokens=cfg.model.max_tokens,
            num_ctx=cfg.context.max_tokens,
            api_key=api_key,
            base_url=cfg.model.base_url or None,
            tokens_per_minute=cfg.model.tokens_per_minute,
            extra_headers=headers,
        )
    return _provider


def set_provider(provider: LLMProvider) -> None:
    """Set the global LLM provider instance."""
    global _provider
    _provider = provider
