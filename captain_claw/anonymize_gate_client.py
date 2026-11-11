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

The decision is asked once a minute per model (and per shared-agent member);
when Flight Deck can't be reached, the last answer stands — never a silent
"off" for a person who turned the gate on.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from captain_claw.anonymize import gate as G
from captain_claw.anonymize.engine import TOKEN_RE
from captain_claw.logging import get_logger

log = get_logger(__name__)

TTL = 60.0
DECIDE_TIMEOUT = 10.0
CALL_TIMEOUT = 180.0
NOTE = ("[Privacy gate: names, addresses and numbers in this result are tokens like ⟦PERSON_1⟧. Use them as "
        "they are; they are put back when you send mail or a message.]")


class GateFailed(RuntimeError):
    pass


@dataclass
class Decision:
    active: bool
    groups: frozenset[str] = frozenset()
    folders: list[str] = field(default_factory=list)
    vault: str = G.GATE_VAULT
    lane: str = "deterministic"

    def covers(self, name: str, arguments: dict[str, Any]) -> bool:
        if not self.active or not G.gated(name, self.groups):
            return False
        # Folders narrow local and VFS file reads only; Drive and OneDrive have
        # no local path, so they stay covered whenever "files" is.
        if (name in G.GROUPS["files"] and name not in ("google_drive", "microsoft_onedrive") and self.folders
                and not any(name in G.GROUPS[g] for g in self.groups - {"files"})):
            return G.under_folders(G.path_values(arguments), self.folders)
        return True

    def egress(self, name: str) -> bool:
        return self.active and name in G.EGRESS_TOOLS


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
                         str(data.get("lane") or "deterministic"))
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
            f"The privacy gate couldn't check what {name} returned, so it isn't shown ({exc}). "
            "Try again in a moment; if it keeps failing, tell the person."))
    if content and TOKEN_RE.search(content):
        content = f"{content}\n\n{NOTE}"
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


def reset_for_tests() -> None:
    _CACHE.clear()
