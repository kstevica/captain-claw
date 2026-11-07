"""Custom tools the agent proposes when a job looks like code's: for the owner to approve.

A custom tool (flight_deck/external_tools.py) is a small program Crew writes and
Flight Deck runs: it does the same job the same way every time, cheaply. The
owner can ask for one, but often the agent is the one who sees the need: it
just walked through a long chain of reads, sums and rewrites by hand, or the
owner said they do this every week.

After an owner's turn, :func:`maybe_propose_tool` checks two cheap signals
before any model call: how many tool calls the turn took, and whether the
owner's words read like a repeated or mechanical job (:func:`has_cue`). Most
turns stop there. For the rest, the agent's own model reads the exchange
(``instructions/tool_proposal_prompt.md``) and either drafts ONE tool (a name,
a title, what it should do, why) or says a tool wouldn't help. A draft goes to
Flight Deck (``POST /fd/external-tools/agent/propose``), where it waits on the
owner's Custom tools page; nothing is built without their click, because every
build spends Crew time and their coding agents' limits.

Only under Flight Deck, only when the deck has custom tools on and the owner
may build them (Crew), only for the owner's own turns (the caller skips
members, public chats, automations, exams and private turns), one draft at a
time, a cooldown and a daily cap (``tool_proposals`` in the config).
"""

from __future__ import annotations

import json
import re
import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from captain_claw.logging import get_logger

if TYPE_CHECKING:
    from captain_claw.agent import Agent

log = get_logger(__name__)

PROPOSE_PATH = "/fd/external-tools/agent/propose"
PROMPT = "tool_proposal_prompt.md"
SAID_MAX = 3000
WHAT_MAX = 2000
WHY_MAX = 400
TITLE_MAX = 80
_SLUG_RE = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,46}[a-z0-9])?$")

_ATTR_RUNNING = "_tool_proposal_running"
_ATTR_LAST = "_tool_proposal_last"
_ATTR_DAY = "_tool_proposal_day"     # (UTC day, drafts that day)

# Words that make a request read like a repeated or mechanical job, the kind a
# program does better. Only a gate in front of the model.
_CUES = re.compile(
    r"\b("
    # English
    r"every (day|week|month|morning|monday|friday|time|quarter|year)|each (day|week|month|time)|weekly|monthly|daily|"
    r"again|repeat\w*|regularly|routinely|automate\w*|script|batch|bulk|all (the|these|of the) \w+|"
    r"convert|parse|extract|reconcile|calculate|total|sum up|add up|compare|merge|dedupe|deduplicate|"
    r"csv|excel|spreadsheet|xlsx|pdfs?|json|xml|invoices?|statements?|receipts?|"
    # Croatian
    r"svaki (dan|tjedan|mjesec|put|ponedjeljak|petak)|tjedno|mjesečno|mjesecno|dnevno|opet|ponovno|redovito|"
    r"automatiz\w*|skript\w*|pretvori|izvuci|izvući|zbroji|izračunaj|izracunaj|usporedi|spoji|tablic\w*|"
    r"račun\w*|racun\w*|izvod\w*"
    r")\b",
    re.IGNORECASE,
)


def has_cue(said: str) -> bool:
    text = " ".join(str(said or "").split())
    if len(text) < 4 or len(text) > SAID_MAX or text.startswith("/"):
        return False
    return bool(_CUES.search(text.replace("’", "'")))


def _content(m: dict[str, Any]) -> str:
    c = m.get("content")
    if isinstance(c, list):
        c = " ".join(str(b.get("text") or "") for b in c if isinstance(b, dict))
    return str(c or "").strip()


def last_turn(messages: list[dict[str, Any]]) -> tuple[str, str, list[str]]:
    """(the owner's last message, the agent's answer to it, the tools the turn
    called, in order) — "" / [] where there is none."""
    user_at = next((i for i in range(len(messages) - 1, -1, -1) if messages[i].get("role") == "user"), -1)
    if user_at < 0:
        return "", "", []
    said = _content(messages[user_at])
    after = next((_content(m) for m in reversed(messages[user_at + 1:])
                  if m.get("role") == "assistant" and _content(m)), "")
    turn = messages[user_at + 1:]
    tools: list[str] = []
    for m in turn:
        for call in m.get("tool_calls") or []:
            if isinstance(call, dict):
                name = (call.get("function") or {}).get("name") or call.get("name")
                if name:
                    tools.append(str(name))
    if not tools:  # a session that keeps only the results
        tools = [str(m.get("name") or m.get("tool_name") or "tool") for m in turn if m.get("role") == "tool"]
    return said, after, tools


def worth_asking(said: str, tools: list[str], min_tool_calls: int) -> bool:
    """The cheap gate: a long hand-made chain, or a mechanical job in words with
    some tool work behind it. A tool proposing tools is never a reason."""
    work = [t for t in tools if t != "custom_tools"]
    return len(work) >= max(1, min_tool_calls) or (has_cue(said) and len(work) >= 3)


def parse(raw: str) -> dict[str, Any] | None:
    """The model's draft, cleaned — None when a tool wouldn't help or it can't be read."""
    text = str(raw or "").strip()
    if text.startswith("```"):
        text = "\n".join(text.split("\n")[1:]).split("\n```")[0]
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        got = json.loads(text[start:end + 1])
    except ValueError:
        return None
    if not isinstance(got, dict):
        return None
    name = str(got.get("name") or "").strip().lower()
    name = re.sub(r"[^a-z0-9-]+", "-", name).strip("-")[:48].strip("-")
    what = " ".join(str(got.get("what") or "").split())[:WHAT_MAX]
    if not name or not _SLUG_RE.match(name) or len(what) < 20:
        return None
    return {"name": name, "title": " ".join(str(got.get("title") or name).split())[:TITLE_MAX],
            "what": what, "why": " ".join(str(got.get("why") or "").split())[:WHY_MAX]}


def build_prompt(said: str, after: str, tools: list[str], existing: list[dict[str, str]]) -> str:
    chain = ", ".join(tools[:40]) + (f" … ({len(tools)} calls in all)" if len(tools) > 40 else f" ({len(tools)} calls)")
    parts = [f"## What the owner asked\n{said[:SAID_MAX]}",
             f"## The tool calls the agent made for it\n{chain if tools else '(none)'}"]
    if after:
        parts.append(f"## The agent's answer\n{after[:2000]}")
    parts.append("## Custom tools the owner already has\n"
                 + ("\n".join(f"- {t.get('ref')}: {t.get('title')}. {t.get('description')}" for t in existing[:30])
                    if existing else "(none)"))
    return "\n\n".join(parts)


def _within_limits(agent: Agent, cfg: Any) -> bool:
    now = time.time()
    if now - float(getattr(agent, _ATTR_LAST, 0.0) or 0.0) < max(0, int(cfg.cooldown_seconds)):
        return False
    day = datetime.now(UTC).date().isoformat()
    seen_day, count = getattr(agent, _ATTR_DAY, ("", 0)) or ("", 0)
    return not (seen_day == day and count >= max(0, int(cfg.max_per_day)))


def _count(agent: Agent) -> None:
    day = datetime.now(UTC).date().isoformat()
    seen_day, count = getattr(agent, _ATTR_DAY, ("", 0)) or ("", 0)
    setattr(agent, _ATTR_DAY, (day, (count if seen_day == day else 0) + 1))
    setattr(agent, _ATTR_LAST, time.time())


def _ready() -> bool:
    """Under Flight Deck, with custom tools on and the owner able to build them."""
    from captain_claw import custom_tools_client
    from captain_claw.fd_client import flight_deck_base, flight_deck_headers, flight_deck_slug

    return bool(flight_deck_base() and flight_deck_slug() and flight_deck_headers().get("X-Agent-Auth")
                and custom_tools_client.can_build())


async def send(proposal: dict[str, Any]) -> tuple[bool, str]:
    """``(kept, why)`` from Flight Deck; never raises."""
    from captain_claw.fd_client import FDClient

    client = FDClient(timeout=10.0)
    try:
        resp = await client.post(PROPOSE_PATH, json=proposal)
        if resp.status_code != 200:
            try:
                detail = str(resp.json().get("detail") or "")
            except Exception:
                detail = ""
            return False, detail or f"Flight Deck answered {resp.status_code}"
        data = resp.json()
        return bool(data.get("kept")), str(data.get("why") or "")
    except Exception as exc:
        return False, type(exc).__name__
    finally:
        await client.close()


async def maybe_propose_tool(agent: Agent, *, channel: str = "") -> dict[str, Any] | None:
    """After an owner's turn: when the job looks like code's, draft one custom tool
    and hand it to Flight Deck for the owner to approve. Returns what was
    proposed (None when nothing was). Never raises."""
    try:
        from captain_claw.config import get_config

        cfg = get_config().tool_proposals
        if not cfg.enabled or getattr(agent, _ATTR_RUNNING, False) or not getattr(agent, "session", None):
            return None
        said, after, tools = last_turn(list(agent.session.messages))
        if not said or not worth_asking(said, tools, cfg.min_tool_calls):
            return None
        if not _within_limits(agent, cfg) or not _ready():
            return None
        setattr(agent, _ATTR_RUNNING, True)
        try:
            _count(agent)
            from captain_claw import custom_tools_client
            from captain_claw.llm import Message

            response = await agent._complete_with_guards(
                messages=[Message(role="system", content=agent.instructions.load(PROMPT)),
                          Message(role="user", content=build_prompt(said, after, tools, custom_tools_client.owner_tools()))],
                tools=None,
                interaction_label="tool_proposal",
                max_tokens=600,
            )
            draft = parse(getattr(response, "content", "") or "")
            if draft is None:
                return None
            proposal = {**draft, "said": said[:SAID_MAX], "channel": str(channel or "")[:20], "source": "noticed"}
            kept, why = await send(proposal)
            if not kept:
                log.debug("Tool proposal not kept", why=why)
                return None
            log.info("Custom tool proposed to the owner", name=draft["name"])
            return proposal
        finally:
            setattr(agent, _ATTR_RUNNING, False)
    except Exception as exc:
        log.warning("Tool proposal failed (non-fatal)", error=str(exc))
        return None
