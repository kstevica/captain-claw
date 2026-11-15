"""Before a call of a kind that may need approval, ask Flight Deck.

The agent's guard (``agent_guard_mixin._execute_tool_with_guard``) calls
:func:`check` right before a tool runs. A call that is none of the kinds
(``approval_kinds``), an agent without Flight Deck, and a kind the deck
doesn't ask about all go ahead at once (the deck's policy is cached for a
minute). Otherwise Flight Deck decides: allowed already (this conversation,
or always for this agent), refused (people do it themselves, or nobody is
there to allow it), or a card in the person's Flight Deck that the agent
waits on, following it in short requests as ``ask_user`` does.
"""

from __future__ import annotations

import secrets
import time
from dataclasses import dataclass
from typing import Any

import httpx

from captain_claw.approval_kinds import KIND_LABELS, describe, kind_for
from captain_claw.logging import get_logger

log = get_logger(__name__)

POLICY_TTL_S = 60.0
_TIMEOUT = 45.0
_POLICY: tuple[float, dict[str, str]] | None = None


@dataclass
class Verdict:
    allowed: bool
    kind: str = ""
    note: str = ""      # for the tool-call register: who allowed it, how
    error: str = ""     # for the model, when it isn't allowed


GO = Verdict(True)


def reset_for_tests() -> None:
    global _POLICY
    _POLICY = None


def _fd_base() -> str:
    from captain_claw.fd_client import flight_deck_base

    return flight_deck_base()


async def _policy() -> dict[str, str] | None:
    """The deck's modes per kind (cached); None when Flight Deck can't say."""
    global _POLICY
    now = time.monotonic()
    if _POLICY is not None and now - _POLICY[0] < POLICY_TTL_S:
        return _POLICY[1]
    from captain_claw.fd_client import flight_deck_headers

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{_fd_base()}/fd/approvals/agent/policy", headers=flight_deck_headers())
        if resp.status_code == 404:  # a Flight Deck from before approvals: nothing asks
            _POLICY = (now, {})
            return {}
        resp.raise_for_status()
        policy = resp.json().get("policy") or {}
        _POLICY = (now, {str(k): str(v) for k, v in policy.items()})
        return _POLICY[1]
    except Exception as exc:  # noqa: BLE001
        log.debug("Approvals policy unavailable", error=str(exc))
        return _POLICY[1] if _POLICY is not None else None


async def _call(path: str, body: dict[str, Any]) -> tuple[int, Any]:
    from captain_claw import speaker
    from captain_claw.fd_client import flight_deck_headers

    headers = flight_deck_headers()
    try:
        headers.update(speaker.grant_headers())
        params = speaker.grant_params()
    except speaker.SpeakerGrantMissing:
        return 403, {"detail": "this member's turn has no grant, so the card can't reach them"}
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        resp = await client.post(f"{_fd_base()}/fd/approvals/agent{path}", headers=headers, params=params,
                                 json=body)
    try:
        return resp.status_code, resp.json()
    except ValueError:
        return resp.status_code, {"detail": resp.text[:500]}


def refusal(kind: str, why: str) -> str:
    """What the model hears when the call wasn't allowed: no workaround."""
    return (
        f"Not done — this needs a person's approval ({KIND_LABELS.get(kind, kind).lower()}) and it wasn't given: "
        f"{why} Don't try to do it another way (another tool, the browser, a script or a colleague). Tell the "
        "person what you were about to do and leave it to them; try again only if they ask you to."
    )


def _origin(agent: Any) -> str:
    src = getattr(agent, "_turn_source", None)
    return str(src[0]) if isinstance(src, tuple) and src else ""


def _session_id(agent: Any) -> str:
    session = getattr(agent, "session", None)
    return str(getattr(session, "id", "") or "")


async def check(agent: Any, name: str, arguments: dict[str, Any], *, on_wait: Any = None) -> Verdict:
    """May *name* run with *arguments* now? Waits while the person decides."""
    kind = kind_for(name, arguments)
    if not kind or not _fd_base():
        return GO
    policy = await _policy()
    if policy is None:
        # Never heard from Flight Deck: every kind starts as "don't ask", so a
        # deck that can't say hasn't turned any on. (Once it has said, a kind
        # it asks about is refused below when the card can't be shown.)
        return GO
    mode = policy.get(kind, "off")
    if mode == "off":
        return GO
    ask_id = "xa_" + secrets.token_hex(8)
    body = {"kind": kind, "tool": name, "summary": describe(name, arguments), "origin": _origin(agent),
            "session_id": _session_id(agent), "wait": 20, "ask_id": ask_id}
    finished = False
    try:
        status, data = await _call("", body)
        told = False
        while status == 200 and isinstance(data, dict) and data.get("pending"):
            if not told and callable(on_wait):
                told = True
                try:
                    on_wait("Waiting for your approval in Flight Deck…")
                except Exception:  # noqa: BLE001
                    pass
            status, data = await _call("/follow", {"id": data["pending"], "wait": 25})
        finished = True
    except httpx.HTTPError as exc:
        return Verdict(False, kind, error=refusal(kind, f"Flight Deck didn't answer ({type(exc).__name__})."))
    finally:
        if not finished:  # the turn was stopped while it waited: the card closes
            try:
                await _call("/cancel", {"id": ask_id})
            except Exception:  # noqa: BLE001
                pass
    if status != 200 or not isinstance(data, dict):
        detail = data.get("detail") if isinstance(data, dict) else data
        return Verdict(False, kind, error=refusal(kind, f"The card couldn't be shown ({detail})."))
    why = str(data.get("why") or "")
    if data.get("decision") == "allow":
        scope = str(data.get("scope") or "")
        note = "" if scope == "off" else f"Approval ({kind}): {why or 'allowed'}"
        return Verdict(True, kind, note=note)
    return Verdict(False, kind, note=f"Approval ({kind}): {why or 'not allowed'}",
                   error=refusal(kind, why or "It wasn't allowed."))
