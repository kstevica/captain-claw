"""The agent's side of the privacy gate.

Flight Deck keeps the person's setting and decides whether it applies to this
agent on the model it runs on now (:mod:`captain_claw.flight_deck.anonymize_gate`);
this module applies it, from the agent's one tool-call path
(``AgentGuardMixin._execute_tool_with_guard``):

* **Results in.** A tool from a covered source (mail, chats, files …): its
  result is anonymized into the person's "Privacy gate" vault *before* it is
  returned — before the model, the session history or anything derived from
  them sees it. If that can't be done, the result isn't shown (fail closed).
* **Messages out.** A tool that sends to people (mail, WhatsApp): tokens in its
  arguments are put back just before it runs, so the recipient reads real
  names, and its own result (which may echo them) is anonymized again.

With the gate covering **the whole chat**, every tool's result is anonymized
(and every tool's arguments get the real values back, so files the agent
writes and searches it runs use them), the person's message is anonymized as
it comes in (``Agent.complete``), and whatever else reaches the model — the
system prompt, the turn's notes, anything put in the history directly — goes
through :func:`anonymize_messages` just before each model call
(``AgentGuardMixin._complete_with_guards``). Each text is anonymized once and
the result cached, so the same text always reads the same to the model (its
prompt cache holds), and the history keeps the tokens.

The decision is asked once a minute per model (and per shared-agent member);
when Flight Deck can't be reached, the last answer stands — never a silent
"off" for a person who turned the gate on.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

import httpx

from captain_claw.anonymize import gate as G
from captain_claw.anonymize.engine import TOKEN_RE
from captain_claw.logging import get_logger

log = get_logger(__name__)

TTL = 60.0
DECIDE_TIMEOUT = 10.0
# One check on Flight Deck — with a trusted model that looks too, every chunk
# of what goes to the model (the system prompt alone is several): it can take
# minutes on a local model. CLAW_ANONYMIZE_GATE_TIMEOUT (seconds) changes it.
CALL_TIMEOUT = float(os.environ.get("CLAW_ANONYMIZE_GATE_TIMEOUT") or 1800.0)
NOTE = ("[Privacy gate: names, addresses and numbers in this result are tokens like ⟦PERSON_1⟧. Use them as "
        "they are; they are put back when you send mail or a message.]")
CHAT_NOTE = ("[Privacy gate: names, addresses and numbers in the person's messages, in what your tools return and "
             "in your memory notes are tokens like ⟦PERSON_1⟧. Use them exactly as they are (in replies, tool "
             "calls, files); the real values are put back for the person and in what you send or write.]")
BATCH_CHARS = 1_500_000  # under Flight Deck's 2M-character limit per call
CACHE_MAX = 20_000


class GateFailed(RuntimeError):
    pass


@dataclass
class Decision:
    active: bool
    groups: frozenset[str] = frozenset()
    folders: list[str] = field(default_factory=list)
    vault: str = G.GATE_VAULT
    lane: str = "deterministic"
    chat: bool = False  # the whole chat, not chosen sources
    model: str = ""  # the trusted model that looks too (lane "both"), for the status line

    def covers(self, name: str, arguments: dict[str, Any]) -> bool:
        if self.active and self.chat:
            return True
        if not self.active or not G.gated(name, self.groups):
            return False
        # Folders narrow local and VFS file reads only; Drive and OneDrive have
        # no local path, so they stay covered whenever "files" is.
        if (name in G.GROUPS["files"] and name not in ("google_drive", "microsoft_onedrive") and self.folders
                and not any(name in G.GROUPS[g] for g in self.groups - {"files"})):
            return G.under_folders(G.path_values(arguments), self.folders)
        return True

    def egress(self, name: str) -> bool:
        """Put the real values back into this tool's arguments before it runs?"""
        if self.active and self.chat:
            return name not in G.KEEP_TOKEN_ARGS
        return self.active and name in G.EGRESS_TOOLS


def why(exc: BaseException) -> str:
    """What went wrong, in a few words (a timeout's own text is empty)."""
    if isinstance(exc, httpx.TimeoutException):
        return f"it took longer than {CALL_TIMEOUT / 60:g} minutes"
    if isinstance(exc, httpx.ConnectError):
        return "Flight Deck couldn't be reached"
    return str(exc) or type(exc).__name__


def working(d: Decision, what: str) -> str:
    """The live status while the gate looks at *what* — so a slow trusted
    model doesn't read as the agent thinking."""
    who = f"the trusted model ({d.model})" if d.lane == "both" and d.model else "code"
    return f"🛡️ Anonymizing {what} — {who} is looking for private data…"


def restoring(name: str) -> str:
    return f"🛡️ Putting the real values back into {name}…"


def fd_facing(label: str) -> bool:
    """A tool call whose result goes to Flight Deck, not to the model."""
    label = str(label or "")
    return label in G.FD_FACING_LABELS or label.startswith(G.FD_FACING_PREFIXES)


_CACHE: dict[tuple[str, ...], tuple[float, Decision]] = {}


def _model(agent: Any) -> tuple[str, str, str]:
    p = getattr(agent, "provider", None)
    return (str(getattr(p, "provider", "") or ""), str(getattr(p, "model", "") or ""),
            str(getattr(p, "base_url", "") or getattr(p, "api_base", "") or ""))


def _transport() -> tuple[str, dict[str, str], dict[str, str]] | None:
    """(deck base, headers, params) for this turn — the member's on a shared
    agent's member turn — or None when this agent has no deck."""
    from captain_claw import speaker
    from captain_claw.google_oauth_manager import GoogleOAuthManager

    base = GoogleOAuthManager._flight_deck_base()
    if not base:
        return None
    headers = GoogleOAuthManager._flight_deck_headers(as_speaker=True)
    if not headers.get("X-Agent-Auth"):
        return None
    return base, headers, speaker.grant_params()


async def decision(agent: Any) -> Decision | None:
    """The gate's answer for this agent now; None when there is no deck or
    the gate doesn't apply."""
    try:
        t = _transport()
    except Exception:  # a member turn without a grant: the deck can't be asked for them
        t = None
    if t is None:
        return None
    base, headers, params = t
    key = (headers.get("X-FD-Speaker-Grant", "owner"), *_model(agent))
    hit = _CACHE.get(key)
    if hit and time.monotonic() - hit[0] < TTL:
        return hit[1] if hit[1].active else None
    provider, model, base_url = _model(agent)
    try:
        async with httpx.AsyncClient(timeout=DECIDE_TIMEOUT) as client:
            resp = await client.post(f"{base}/fd/anonymize/agent/gate", headers=headers, params=params,
                                     json={"provider": provider, "model": model, "base_url": base_url})
        if resp.status_code == 404:  # a deck without the gate
            d = Decision(False)
        else:
            resp.raise_for_status()
            data = resp.json()
            d = Decision(bool(data.get("active")), frozenset(data.get("groups") or ()),
                         list(data.get("folders") or []), str(data.get("vault") or G.GATE_VAULT),
                         str(data.get("lane") or "deterministic"), bool(data.get("chat")),
                         str(data.get("model") or ""))
    except Exception as exc:  # noqa: BLE001 — keep the last answer
        log.debug("Privacy gate: decision unavailable", error=type(exc).__name__)
        return hit[1] if hit and hit[1].active else None
    _CACHE[key] = (time.monotonic(), d)
    return d if d.active else None


def cached_decision(agent: Any) -> Decision | None:
    """Sync: the gate's last answer for this agent's current model, when it
    applied (the owner's, or any member's — the stricter reading)."""
    provider, model, base_url = _model(agent)
    for (_who, *rest), (_at, d) in _CACHE.items():
        if tuple(rest) == (provider, model, base_url) and d.active:
            return d
    return None


def cached_active(agent: Any) -> bool:
    """Sync: did the gate apply to this agent at the last look (for the
    anonymize tool's default vault)?"""
    return cached_decision(agent) is not None


async def _post(path: str, body: dict[str, Any]) -> dict[str, Any]:
    t = _transport()
    if t is None:
        raise GateFailed("this agent has no Flight Deck")
    base, headers, params = t
    async with httpx.AsyncClient(timeout=CALL_TIMEOUT) as client:
        resp = await client.post(f"{base}{path}", headers=headers, params=params, json=body)
    if resp.status_code != 200:
        try:
            detail = str(resp.json().get("detail") or "")
        except Exception:  # noqa: BLE001
            detail = ""
        raise GateFailed(detail or f"Flight Deck answered {resp.status_code}")
    return resp.json()


async def anonymize(d: Decision, text: str) -> str:
    data = await _post("/fd/anonymize/agent/run", {
        "content": text, "format": "text", "vault": d.vault, "lane": d.lane, "mode": "reversible", "source": "gate"})
    return str(data.get("output") or "")


async def gate_result(d: Decision, name: str, result: Any) -> Any:
    """*result* (a ToolResult) with its text anonymized; a failed check means
    no result at all."""
    from captain_claw.tools.registry import ToolResult

    try:
        content = await anonymize(d, result.content) if (result.content or "").strip() else result.content
        error = await anonymize(d, result.error) if (result.error or "").strip() else result.error
    except Exception as exc:  # noqa: BLE001 — fail closed
        log.warning("Privacy gate: result withheld", tool=name, error=type(exc).__name__)
        return ToolResult(success=False, error=(
            f"The privacy gate couldn't check what {name} returned, so it isn't shown ({why(exc)}). "
            "Try again in a moment; if it keeps failing, tell the person."))
    if content and TOKEN_RE.search(content) and not d.chat:  # the whole chat: said once, in the prompt
        content = f"{content}\n\n{NOTE}"
    remember(content or "")
    return result.model_copy(update={"content": content, "error": error})


async def restore_args(d: Decision, arguments: dict[str, Any]) -> tuple[dict[str, Any], int]:
    """*arguments* with tokens put back in every string the agent wrote (not
    the runtime's own ``_…`` entries); and how many strings changed."""
    changed = 0

    async def fix(value: Any) -> Any:
        nonlocal changed
        if isinstance(value, str):
            if not TOKEN_RE.search(value):
                return value
            data = await _post("/fd/anonymize/agent/restore", {"content": value, "format": "text", "vault": d.vault})
            out = str(data.get("output") or value)
            if out != value:
                changed += 1
            return out
        if isinstance(value, list):
            return [await fix(v) for v in value]
        if isinstance(value, dict):
            return {k: (await fix(v)) for k, v in value.items()}
        return value

    fixed = {k: (v if k.startswith("_") else await fix(v)) for k, v in arguments.items()}
    return fixed, changed


# ── the whole chat: what reaches the model ──────────────────────────

# sha256(text) → its anonymized text, for texts already anonymized (both the
# original's hash and the result's), so each is sent to Flight Deck once and
# always reads the same.
_CLEAN: OrderedDict[str, str] = OrderedDict()


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()


def remember(anonymized: str, original: str | None = None) -> None:
    """*anonymized* is clean (and is what *original* becomes)."""
    if not anonymized:
        return
    _CLEAN[_sha(anonymized)] = anonymized
    if original is not None:
        _CLEAN[_sha(original)] = anonymized
    while len(_CLEAN) > CACHE_MAX:
        _CLEAN.popitem(last=False)


def known(text: str) -> str | None:
    """The anonymized form of *text*, when it has been seen (None otherwise)."""
    hit = _CLEAN.get(_sha(text))
    if hit is not None:
        _CLEAN.move_to_end(_sha(text))
    return hit


async def anonymize_texts(d: Decision, texts: list[str]) -> list[str]:
    """Several texts in one call (a JSON list): one engine run, same tokens."""
    out: list[str] = []
    batch: list[str] = []
    size = 0

    async def flush() -> None:
        nonlocal batch, size
        if not batch:
            return
        data = await _post("/fd/anonymize/agent/run", {
            "content": json.dumps(batch, ensure_ascii=False), "format": "json", "vault": d.vault,
            "lane": d.lane, "mode": "reversible", "source": "gate-chat"})
        got = json.loads(str(data.get("output") or "[]"))
        if not isinstance(got, list) or len(got) != len(batch):
            raise GateFailed("Flight Deck returned a different number of texts")
        out.extend(str(x) for x in got)
        batch, size = [], 0

    for text in texts:
        if batch and size + len(text) > BATCH_CHARS:
            await flush()
        batch.append(text)
        size += len(text)
    await flush()
    return out


async def anonymize_user_text(d: Decision, text: str) -> str:
    """The person's message, before anything reads it (the whole chat)."""
    if not text.strip():
        return text
    hit = known(text)
    if hit is not None:
        return hit
    (out,) = await anonymize_texts(d, [text])
    remember(out, text)
    return out


async def anonymize_messages(agent: Any, d: Decision, messages: list[Any],
                             on_work: Any = None) -> list[Any]:
    """*messages* (llm ``Message``) as the model may see them: every text not
    seen yet goes to Flight Deck (in one call); the session's history is
    rewritten to the tokens too, so it keeps them. Pictures pass as they are.
    The system prompt (role "system") is left as it is: it is the agent's own
    setup, not what the person or their memory brought in — and reading it
    through a trusted model on every change cost minutes per turn (decided
    2026-11-12). The memory notes ride in a user message, so they are covered.
    *on_work(n)*: called before Flight Deck is asked, with how many texts."""
    todo: dict[str, str] = {}
    for m in messages:
        text = getattr(m, "content", None)
        if (getattr(m, "role", "") != "system" and isinstance(text, str) and text.strip()
                and known(text) is None):
            todo.setdefault(_sha(text), text)
    if todo:
        if on_work is not None:
            on_work(len(todo))
        originals = list(todo.values())
        for original, anonymized in zip(originals, await anonymize_texts(d, originals), strict=True):
            remember(anonymized, original)
    out = []
    for m in messages:
        text = getattr(m, "content", None)
        if getattr(m, "role", "") == "system":
            out.append(m)
            continue
        clean = known(text) if isinstance(text, str) and text.strip() else None
        out.append(dataclasses.replace(m, content=clean) if clean is not None and clean != text else m)
    _rewrite_history(agent)
    return out


def _rewrite_history(agent: Any) -> None:
    """The session's messages that were anonymized on the way out keep the
    tokens (the next call reads the same, and nothing raw is left to derive
    notes or summaries from)."""
    session = getattr(agent, "session", None)
    for msg in getattr(session, "messages", None) or []:
        if not isinstance(msg, dict):
            continue
        text = msg.get("content")
        if isinstance(text, str) and text.strip():
            clean = known(text)
            if clean is not None and clean != text:
                msg["content"] = clean


async def restore_text(d: Decision, text: str) -> str:
    """*text* with the real values back (for the person: their Telegram chat)."""
    if not TOKEN_RE.search(text or ""):
        return text
    data = await _post("/fd/anonymize/agent/restore", {"content": text, "format": "text", "vault": d.vault})
    return str(data.get("output") or text)


def reset_for_tests() -> None:
    _CACHE.clear()
    _CLEAN.clear()
