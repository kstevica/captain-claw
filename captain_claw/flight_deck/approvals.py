"""Approval cards: a person allows, or doesn't, an agent's risky tool call.

The kinds (paying, sending, posting, sharing, deleting, passwords) come from
``captain_claw.approval_kinds``. For each kind an admin sets what it needs on
this deck:

- ``off``          — don't ask (the default: nothing changes until an admin
                     turns a kind on);
- ``ask``          — ask the person the turn is for; they may allow it once,
                     for this conversation, or always for this agent;
- ``always_ask``   — ask every time (only "Allow once" or "Don't allow");
- ``people_only``  — agents never do it; people do it themselves.

The card is a question in the person's Flight Deck (``person_asks``, source
``approval``): every tab of theirs and the bell. "For this conversation" is
kept in memory for the chat (lost when Flight Deck restarts); "Always allow
here" is kept in the person's settings until they remove it. A run nobody is
there for (a schedule, autonomous work) can only do an ``ask`` kind the person
already chose to always allow for that agent.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

from captain_claw.approval_kinds import KIND_LABELS, KIND_VERBS, KINDS
from captain_claw.flight_deck import person_asks as asks
from captain_claw.logging import get_logger

log = get_logger(__name__)

MODES = ("off", "ask", "always_ask", "people_only")
MODE_LABELS = {
    "off": "Don't ask",
    "ask": "Ask",
    "always_ask": "Ask every time",
    "people_only": "People do it themselves",
}
POLICY_KEY = "fd:approvals"          # system setting: {kind: mode}
ALWAYS_KEY = "fd:approvals-always"   # user setting: {agent_slug: [kind, ...]}
CONVERSATION_KEEP_S = 12 * 3600.0
DEFAULT_WAIT_S = 600.0

ONCE = "Allow once"
CONVERSATION = "Allow for this conversation"
ALWAYS = "Always allow here"
DENY = "Don't allow"

# (user, agent, session, kind) → when it was allowed
_CONVERSATION: dict[tuple[str, str, str, str], float] = {}


@dataclass
class Pending:
    """An open card's facts, to turn the person's answer into a decision."""
    user_id: str
    agent: str
    session_id: str
    kind: str
    by: str


_PENDING: dict[str, Pending] = {}
_DECIDED: dict[str, tuple[float, dict[str, Any]]] = {}


def reset_for_tests() -> None:
    _CONVERSATION.clear()
    _PENDING.clear()
    _DECIDED.clear()


# ── The deck's policy ─────────────────────────────────────────────────


def clean_policy(raw: Any) -> dict[str, str]:
    """Every kind with a known mode (``off`` for anything missing or odd)."""
    src = raw if isinstance(raw, dict) else {}
    return {k: (str(src.get(k)) if str(src.get(k)) in MODES else "off") for k in KINDS}


async def load_policy(db: Any) -> dict[str, str]:
    try:
        raw = await db.get_system_setting(POLICY_KEY)
        return clean_policy(json.loads(raw) if raw else {})
    except Exception:  # noqa: BLE001 — a broken setting asks for nothing, as before it existed
        log.warning("Approvals policy unreadable: nothing asks", exc_info=True)
        return clean_policy({})


async def save_policy(db: Any, policy: dict[str, Any]) -> dict[str, str]:
    for kind, mode in (policy or {}).items():
        if kind not in KINDS or mode not in MODES:
            raise ValueError(f"Unknown kind or mode: {kind}={mode}")
    merged = {**(await load_policy(db)), **policy}
    clean = clean_policy(merged)
    await db.set_system_setting(POLICY_KEY, json.dumps(clean))
    return clean


# ── What a person already allowed ─────────────────────────────────────


async def always_allowed(db: Any, user_id: str) -> dict[str, list[str]]:
    try:
        raw = await db.get_setting(user_id, ALWAYS_KEY)
        data = json.loads(raw) if raw else {}
    except Exception:  # noqa: BLE001
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(a): sorted({k for k in kinds if k in KINDS})
            for a, kinds in data.items() if isinstance(kinds, list)}


async def set_always(db: Any, user_id: str, agent: str, kind: str, allowed: bool) -> dict[str, list[str]]:
    data = await always_allowed(db, user_id)
    kinds = set(data.get(agent, []))
    if allowed:
        kinds.add(kind)
    else:
        kinds.discard(kind)
    if kinds:
        data[agent] = sorted(kinds)
    else:
        data.pop(agent, None)
    if data:
        await db.set_settings(user_id, {ALWAYS_KEY: json.dumps(data)})
    else:
        await db.delete_setting(user_id, ALWAYS_KEY)
    return data


def _conversation_allows(user_id: str, agent: str, session_id: str, kind: str) -> bool:
    if not session_id:
        return False
    key = (user_id, agent, session_id, kind)
    at = _CONVERSATION.get(key)
    if at is None:
        return False
    if time.time() - at > CONVERSATION_KEEP_S:
        _CONVERSATION.pop(key, None)
        return False
    return True


def _allow_conversation(user_id: str, agent: str, session_id: str, kind: str) -> None:
    now = time.time()
    for key in [k for k, at in _CONVERSATION.items() if now - at > CONVERSATION_KEEP_S]:
        _CONVERSATION.pop(key, None)
    if session_id:
        _CONVERSATION[(user_id, agent, session_id, kind)] = now


# ── Deciding ──────────────────────────────────────────────────────────


def _allow(scope: str, by: str = "", why: str = "") -> dict[str, Any]:
    return {"decision": "allow", "scope": scope, "by": by, "why": why}


def _deny(why: str, *, scope: str = "", by: str = "", **extra: Any) -> dict[str, Any]:
    return {"decision": "deny", "scope": scope, "by": by, "why": why, **extra}


async def person_name(db: Any, user_id: str) -> str:
    try:
        user = await db.get_user_by_id(user_id)
    except Exception:  # noqa: BLE001
        user = None
    if not user:
        return "the person"
    return str(user.get("display_name") or user.get("email") or "the person")


async def request(db: Any, *, user_id: str, agent: str, agent_name: str, kind: str, summary: str,
                  origin: str, session_id: str, ask_id: str = "", wait_s: float = DEFAULT_WAIT_S,
                  ) -> tuple[dict[str, Any] | None, asks.Ask | None]:
    """Decide a call of *kind* for the person *user_id* on *agent*: the decision
    at once, or ``(None, ask)`` while the card waits for them."""
    if kind not in KINDS:
        return _allow("off"), None
    mode = (await load_policy(db))[kind]
    label = KIND_LABELS[kind].lower()
    if mode == "off":
        return _allow("off"), None
    if mode == "people_only":
        return _deny(f"On this deck agents don't {KIND_VERBS[kind]}: people do that themselves.",
                     scope="people_only", people_only=True), None
    by = await person_name(db, user_id)
    if mode == "ask":
        if kind in (await always_allowed(db, user_id)).get(agent, []):
            return _allow("always", by, f"{by} always allows this agent to {KIND_VERBS[kind]}."), None
        if _conversation_allows(user_id, agent, session_id, kind):
            return _allow("conversation", by, f"{by} allowed it for this conversation."), None
    if not asks.attended(origin):
        return _deny(
            f"Nobody was there to allow it: this run started on its own. {by} can choose “{ALWAYS}” "
            f"when the agent asks in a chat, or an admin can stop asking about “{label}”.",
            scope="unattended", nobody=True), None
    choices = [ONCE, CONVERSATION, ALWAYS, DENY] if mode == "ask" else [ONCE, DENY]
    question = {"id": "decision", "question": f"Let {agent_name} {KIND_VERBS[kind]}?",
                "help": summary, "choices": choices}
    detail = {"kind": kind, "kind_label": KIND_LABELS[kind], "summary": summary, "mode": mode}
    ring = (f"{agent_name} wants to {KIND_VERBS[kind]}"[:200],
            "Allow it or not in Flight Deck: the card is open there.")
    a = await asks.open_ask(user_id, source="approval", title=agent_name, agent_name=agent_name,
                            questions=asks.check_questions([question]), wait_s=wait_s,
                            ask_id=ask_id, ring=ring, detail=detail)
    _PENDING[a.id] = Pending(user_id, agent, session_id, kind, by)
    return None, a


async def outcome(db: Any, ask_id: str, done: dict[str, Any]) -> dict[str, Any]:
    """The decision from a finished card's outcome; remembers "for this
    conversation" and "always". Asked again, it gives the same answer."""
    kept = _DECIDED.get(ask_id)
    if kept is not None:
        return kept[1]
    p = _PENDING.pop(ask_id, None)
    if p is None:
        return _deny("That approval card is gone.")
    answer = (done.get("answers") or {}).get("decision") if done.get("ok") else None
    if answer == ONCE:
        result = _allow("once", p.by, f"{p.by} allowed it once.")
    elif answer == CONVERSATION:
        _allow_conversation(p.user_id, p.agent, p.session_id, p.kind)
        result = _allow("conversation", p.by, f"{p.by} allowed it for this conversation.")
    elif answer == ALWAYS:
        await set_always(db, p.user_id, p.agent, p.kind, True)
        result = _allow("always", p.by, f"{p.by} chose to always allow this agent to {KIND_VERBS[p.kind]}.")
    elif answer == DENY or done.get("declined"):
        result = _deny(f"{p.by} didn't allow it.", scope="denied", by=p.by)
    else:
        result = _deny(f"{p.by} didn't answer in time.", scope="no_answer", by=p.by)
    now = time.time()
    for old in [k for k, (at, _r) in _DECIDED.items() if now - at > asks.DONE_KEEP_S]:
        _DECIDED.pop(old, None)
    _DECIDED[ask_id] = (now, result)
    return result
