"""Taskboard ingest — every turn an agent takes, and the tool calls in it, on
the board of the person it was for (v1).

  agent  POST /fd/taskboard/agent/turns   an agent's turn register (turn_register.py)

An agent reports each turn twice: when it opens (where it came from, who
spoke, the first words) and when it closes (how it ended, the reply, the files
it saved). Both are kept in ``turns`` for 30 days; then a gate — no model —
decides what the turn is to the board:

* a person's turn (``msg_origin``: a turn opener, human input, no code-made
  opening such as a corrective, a nudge or a life tick; not an onboarding
  exam) is a task of its own, ``turn:<turn_id>`` — every quick question too,
  never one task per session (decision #3) — unless the user's ``task_rules``
  say it isn't, or it came by Gmail and the mail didn't pass
  ``gmail_inbound.prefilter``;
* a scheduled task's, an Autonomous Work action's or a linked system's turn
  is told on the task v0 keeps for it (by its session, action or message),
  never a task of its own; when the sweep hasn't read that run yet, the turn
  waits for it (six hours at most);
* a worker Flight Deck runs itself (a run's worker, a being's body) makes
  nothing.

Whose board: the agent's owner's — or, on a shared agent, the member's who
spoke, and only theirs (decision #4). Who the agent and its owner are comes
from Flight Deck's own records (the ``X-Agent-Auth`` token, exactly as the
tool-call register identifies a caller), never from the body; a member counts
only while they are a member of that agent, and a member's turn that names no
such member is dropped, never shown to the owner. Texts are masked again here.

The tool-call register's rows name the turn they were made in: the sweep
reads them (:func:`tool_calls`) onto that turn's task — the tool, its action
and how it ended, never its parameters — and an ok call that sends something
out is recorded evidence. A call that names its Gmail thread links a
person's request to it, so what v0 reads about the thread lands on the same
task (an automation's standing task never takes the threads it mails).

A new event on a turn's open task puts it back in the triage queue
(``taskboard_triage``).
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from captain_claw import turn_source
from captain_claw.flight_deck import taskboard_sources, taskboard_store
from captain_claw.flight_deck.auth import get_db
from captain_claw.flight_deck.taskboard_sources import _head, _json, _rows
from captain_claw.flight_deck.taskboard_store import CHANNEL_LABELS, TaskboardStore
from captain_claw.logging import get_logger
from captain_claw.outbound_guard import _fold

log = get_logger(__name__)

router = APIRouter(tags=["taskboard"])

MAX_BATCH = 200
GROUP_SPEAKER_PREFIX = "tg-group:"   # speaker.GROUP_SPEAKER_PREFIX
OUTCOMES = ("ok", "error", "blocked", "cancelled")
AUTOMATIONS = ("scheduled", "autonomy", "link")
TURNS_KEPT_S = taskboard_sources.TURNS_KEPT_S

_HEAD_MAX = 1000          # text_head / reply_head kept on a turn
_SMALL_JSON = 2000        # origin_detail / ext_refs kept, as JSON
_TURN_WAIT_S = 6 * 3600  # an automation turn waits this long for its task
_CALL_WAIT_S = 3600       # a tool call waits this long for its turn to arrive
_CALLS_WAITING_KEPT = 1000
_CLOCK_SKEW_S = 300.0
_PRUNE_EVERY_S = 3600.0   # turns past 30 days are forgotten this often as agents post (the sweep may be off)
_pruned_at = 0.0

# Turns from these sources (turn_source) are automated whatever they say.
_AUTOMATED_SOURCES = ("scheduled", "flow", "agent", "autonomy", "exam", "link")
# Channels only the board user speaks on (taskboard_triage.OWN_CHANNELS), and
# the sources where someone else writes in.
_OWN_CHANNELS = ("web", "member", "glasses", "device", "api", "mcp")
_OUTSIDE_SOURCES = ("whatsapp", "telegram", "messenger", "discord", "public")
# A session's stored origin kind → the turn channel it is an address on.
_ORIGIN_CHANNELS = {"whatsapp": "whatsapp", "waid": "whatsapp", "telegram": "telegram", "tg": "telegram",
                    "email": "gmail", "gmail": "gmail", "waweb": "waweb", "whatsapp_web": "waweb",
                    "wa_jid": "waweb"}
# The most specific match first; at the same, the effect that keeps it off first.
_RULE_DIMS = ("sender", "domain", "agent", "channel", "source")
_RULE_EFFECTS = ("not_task", "mute", "always_task")
# Sends out by its risk class, but reaches nobody a task is about.
_NOT_DELIVERIES = frozenset({"jev"})
# Where a send went, by its tool.
_SEND_CHANNELS = {"google_mail": "gmail", "send_mail": "email", "microsoft_mail": "email",
                  "whatsapp_send_file": "whatsapp", "whatsapp_web": "waweb", "linked_system": "link"}
_CALL_OUTCOMES = {"ok": "ok", "error": "failed", "blocked": "blocked", "timeout": "timed out",
                  "aborted": "stopped", "interrupted": "interrupted"}
_ACTION_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,40}$")
_END_SUMMARY = {"error": "The turn ended in an error", "blocked": "The turn was blocked",
                "cancelled": "Stopped before it finished"}


# ── The turn register ─────────────────────────────────────────────────


class TurnRecord(BaseModel):
    seq: int = Field(ge=1, le=10**12)
    phase: Literal["open", "close"]
    turn_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    at: float
    # open
    session_id: str = Field("", max_length=128)
    session_name: str = Field("", max_length=500)
    origin: str = Field("", max_length=64)
    origin_detail: dict[str, Any] = Field(default_factory=dict)
    channel: str = Field("", max_length=64)
    source: str = Field("", max_length=64)
    source_label: str = Field("", max_length=200)
    speaker_id: str = Field("", max_length=160)
    speaker_label: str = Field("", max_length=200)
    member: bool = False
    private: bool = False
    opener: bool = False
    human: bool = False
    literal: str = Field("", max_length=64)
    text_head: str = Field("", max_length=8000)
    ext_refs: dict[str, Any] = Field(default_factory=dict)
    automation: dict[str, Any] = Field(default_factory=dict)
    # close
    outcome: str = Field("", max_length=16)
    reply_head: str = Field("", max_length=8000)
    deliverables: list[str] = Field(default_factory=list, max_length=200)
    tool_calls: int = Field(0, ge=0, le=10**9)
    failed_tool_calls: int = Field(0, ge=0, le=10**9)


class TurnBatch(BaseModel):
    boot: str = Field(pattern=r"^[0-9A-Za-z_-]{1,64}$")
    records: list[TurnRecord] = Field(max_length=MAX_BATCH)


async def _user(uid: str, cache: dict[str, Any]) -> dict | None:
    key = f"user:{uid}"
    if key not in cache:
        cache[key] = await get_db().get_user_by_id(uid)
    return cache[key]


async def _is_member(sid: str, who: dict[str, str], token: str, cache: dict[str, Any]) -> bool:
    """Is *sid* a member of this agent now (``agent_sharing.member_check``,
    as the tool-call register checks a speaker)? Fails closed."""
    key = f"member:{sid}"
    if key not in cache:
        ok = False
        try:
            from captain_claw.flight_deck import agent_sharing
            from captain_claw.flight_deck.context_packs import agent_ref_for_auth

            if "ref" not in cache:
                cache["ref"] = await asyncio.to_thread(agent_ref_for_auth, token)
            if cache["ref"] and who["owner"]:
                ok = await agent_sharing.member_check(get_db(), cache["ref"], who["owner"], sid)
        except Exception as exc:
            log.warning("Taskboard turns: member check failed", error=type(exc).__name__)
        cache[key] = ok
    return bool(cache[key])


def _name(user: dict | None) -> str:
    return str((user or {}).get("display_name") or (user or {}).get("email") or "")


async def _board(rec: TurnRecord, who: dict[str, str], token: str,
                 cache: dict[str, Any]) -> tuple[str | None, str, str]:
    """``(board user, speaker id, speaker name)`` for an open record: the
    owner's board — or the member's who spoke, on a shared agent. The board
    is None when the turn is nobody's: an agent without a (living) owner, or
    a member's turn that names no member of this agent."""
    label = " ".join(rec.speaker_label.split())[:120]
    # Nobody's name stands in for someone else's words (a stranger on Messenger).
    own = _own_channel(rec.channel.strip().lower(), turn_source.normalize(rec.source) or rec.source.strip())
    if not taskboard_sources._auth_enabled():
        return "local", "local" if own else "", label or ("You" if own else "")
    owner = who["owner"]
    if not owner or owner == "local":
        return None, "", ""
    owner_user = await _user(owner, cache)
    if owner_user is None:
        return None, "", ""
    sid = rec.speaker_id.strip()
    if sid.startswith(GROUP_SPEAKER_PREFIX):
        sid = ""  # a Telegram group: the owner's agent, other people speaking
    if sid == owner:
        return owner, owner, label or _name(owner_user)
    if sid and await _is_member(sid, who, token, cache):
        return sid, sid, label or _name(await _user(sid, cache)) or "Member"
    if rec.member:
        return None, "", ""  # a member's words, and no member to show them to
    return owner, "", label or (_name(owner_user) if own else "")


@router.post("/fd/taskboard/agent/turns")
async def agent_turns(body: TurnBatch, request: Request) -> dict:
    """An agent reports turns it opened and closed."""
    from captain_claw.flight_deck.tool_audit_routes import _identify

    who = await _identify(request)
    token = request.headers.get("X-Agent-Auth", "") or ""
    cache: dict[str, Any] = {}
    items: list[tuple[TurnRecord, tuple[str | None, str, str]]] = []
    for rec in body.records:
        items.append((rec, await _board(rec, who, token, cache) if rec.phase == "open" else (None, "", "")))
    touched = await asyncio.to_thread(_take, body.boot, who, items)
    for uid in touched:
        _kick(uid)
    return {"ok": True, "accepted": len(body.records)}


def _small(value: dict, limit: int = _SMALL_JSON) -> dict:
    """*value* masked, keeping its keys while the whole fits in *limit*
    characters of JSON."""
    from captain_claw.tool_audit import mask_tree

    out: dict[str, Any] = {}
    size = 2
    for k, v in mask_tree(value).items():
        n = len(json.dumps({str(k)[:64]: v}, ensure_ascii=False, default=str))
        if size + n <= limit:
            out[str(k)[:64]] = v
            size += n
    return out


def _files(paths: list[str]) -> list[str]:
    """The files a turn saved, as relative paths (at most 20)."""
    out = []
    for p in paths:
        p = str(p or "").strip().replace("\\", "/")
        if p and not p.startswith("/") and ".." not in p.split("/") and len(p) <= 300 and p not in out:
            out.append(p)
    return out[:20]


def _take(boot: str, who: dict[str, str], items: list[tuple[TurnRecord, tuple[str | None, str, str]]]) -> set[str]:
    """Keep each record once and put what it adds on the board. The boards
    with a task now in the triage queue."""
    from captain_claw.tool_audit import mask_text

    global _pruned_at
    store = taskboard_store.get_store()
    managed = _managed(who["slug"])
    agents = taskboard_sources._agent_index()
    now = time.time()
    touched: set[str] = set()
    if now - _pruned_at >= _PRUNE_EVERY_S:
        # The 30 days hold with the sweep switched off too (FD_TASKBOARD_DISABLED).
        _pruned_at = now
        try:
            taskboard_sources.prune_turns(store, now)
        except Exception as exc:
            log.warning("Taskboard turns prune failed", error=str(exc))
    for rec, (board, speaker_id, speaker_name) in items:
        if rec.at < now - TURNS_KEPT_S:
            continue  # older than the board keeps turns
        at = rec.at if rec.at <= now + _CLOCK_SKEW_S else now
        agent = {"agent_owner_id": who["owner"], "agent_slug": who["slug"]}
        ref = f"{boot}:{rec.seq}"
        if rec.phase == "open":
            if board is None:
                # Nobody's to see: its close (before or after) isn't kept either.
                store.drop_turn(rec.turn_id, who["slug"], who["owner"], ref)
                continue
            kind = str(rec.automation.get("kind") or "").strip().lower()[:32]
            member = board not in (who["owner"], "local")
            fields = {
                **agent, "user_id": board, "session_id": rec.session_id, "session_name": _head(rec.session_name, 200),
                "origin": rec.origin.strip(), "origin_detail": _small(rec.origin_detail),
                "channel": rec.channel.strip().lower(), "source": turn_source.normalize(rec.source) or rec.source.strip(),
                "source_label": _head(rec.source_label, 80), "speaker_id": speaker_id, "speaker_label": speaker_name,
                "member": member, "private": rec.private or member,  # a member's words are theirs
                "opener": rec.opener, "human": rec.human, "literal": rec.literal.strip().lower(),
                "text_head": mask_text(rec.text_head)[:_HEAD_MAX], "ext_refs": _small(rec.ext_refs),
                "automation": {"kind": kind, "id": str(rec.automation.get("id") or "").strip()[:128]} if kind else {},
                "started_at": at,
            }
        else:
            outcome = rec.outcome if rec.outcome in OUTCOMES else "error"
            fields = {**agent, "ended_at": at, "outcome": outcome, "reply_head": mask_text(rec.reply_head)[:_HEAD_MAX],
                      "deliverables": _files(rec.deliverables), "tool_calls": rec.tool_calls,
                      "failed_tool_calls": min(rec.failed_tool_calls, rec.tool_calls)}
        if store.save_turn(rec.turn_id, rec.phase, ref, fields) != "new":
            continue
        uid = settle(store, rec.turn_id, agents=agents, managed=managed)
        if uid:
            touched.add(uid)
    return touched


def _managed(slug: str) -> bool:
    """A worker Flight Deck runs itself (``agent_sharing.is_managed_agent``)."""
    from captain_claw.flight_deck.agent_sharing import is_managed_agent

    try:
        from captain_claw.flight_deck.server import _load_process_registry

        entry = _load_process_registry().get(slug)
    except Exception:
        entry = None
    return is_managed_agent(slug, str(entry.get("description") or "") if isinstance(entry, dict) else "")


def _kick(uid: str) -> None:
    """Wake the triage worker for *uid*'s board (never in the way)."""
    try:
        from captain_claw.flight_deck import taskboard_triage

        taskboard_triage.kick(uid)
    except Exception:
        pass


# ── The gate ──────────────────────────────────────────────────────────


def settle(store: TaskboardStore, turn_id: str, *, agents: dict[str, dict] | None = None,
           managed: bool = False, now: float | None = None) -> str | None:
    """Put what is known of a turn on the board: once its open is here, the
    gate's verdict (once — a waiting automation turn is tried again); once
    its close is, the reply on the task it is on. The board user when that
    put one of their tasks in the triage queue."""
    now = time.time() if now is None else now
    turn = store.get_turn(turn_id)
    if turn is None or turn["started_at"] is None or not turn["user_id"]:
        return None  # the close came first: the open says what the turn is
    agents = taskboard_sources._agent_index() if agents is None else agents
    queued = False
    if turn["gate"] in ("", "waiting"):
        gate, task_id, key = _gate(store, turn, agents, managed)
        if gate == "waiting" and now - float(turn["received_at"] or now) > _TURN_WAIT_S:
            gate = "skipped:no_task"
        store.set_turn(turn_id, gate=gate, task_id=task_id, task_key=key)
        turn.update(gate=gate, task_id=task_id, task_key=key)
        task = store.get_task(turn["user_id"], task_id) if gate == "task" else None
        queued = task is not None and task["triage_state"] == "pending"
    if turn["gate"] in ("task", "attached") and turn["ended_at"] is not None:
        queued |= _reply(store, turn, agents, now)
    return turn["user_id"] if queued else None


def _gate(store: TaskboardStore, turn: dict, agents: dict[str, dict], managed: bool) -> tuple[str, str, str]:
    """``(verdict, task id, task key)``: ``task`` (a task of its own),
    ``attached`` (on its automation's task), ``waiting`` (that task isn't on
    the board yet) or ``skipped:<why>``."""
    if managed:
        return "skipped:managed", "", ""
    kind = str(turn["automation"].get("kind") or "")
    if kind:
        if kind not in AUTOMATIONS:
            return "skipped:automated", "", ""
        keys = _automation_keys(turn)
        if not keys:
            return "skipped:no_task", "", ""  # nothing names its task
        held = store.tasks_for_keys(turn["user_id"], keys)
        key = next((k for k in keys if k in held), "")
        return ("attached", held[key], key) if key else ("waiting", "", "")
    why = _not_a_persons(turn)
    if why:
        return f"skipped:{why}", "", ""
    gmail = turn["channel"] == "gmail" or bool(_ref(turn["ext_refs"], "gmail", "message_id"))
    blocked, sender = _gmail_verdict(turn) if gmail else ("", None)
    requester = sender or _requester(turn)
    rule = _rule(store, turn, requester)
    effect = rule["effect"] if rule else ""
    if rule:
        store.rule_hit(turn["user_id"], rule["id"])
    if effect == "not_task":
        return "skipped:rule", "", ""
    if blocked and effect != "always_task":
        return f"skipped:{blocked}", "", ""
    return "task", _admit(store, turn, requester, effect, agents), f"turn:{turn['turn_id']}"


def _not_a_persons(turn: dict) -> str:
    """Why a turn isn't a person's ask ("" when it is). Every kind
    ``msg_origin.detect_literal`` names is a code-made opening (a corrective,
    a nudge, a life tick, a scheduler prompt…), so any keeps it off."""
    if not turn["opener"]:
        return "not_opener"
    if not turn["human"]:
        return "not_human"
    if turn["literal"]:
        return f"literal:{turn['literal']}"
    if turn_source.channel_of(turn["source"]) in _AUTOMATED_SOURCES:
        return "automated"
    return ""


def _ref(ext: dict, group: str, key: str) -> str:
    """``ext_refs[group][key]``, or the flat ``ext_refs[f"{group}_{key}"]``."""
    nested = ext.get(group)
    value = nested.get(key) if isinstance(nested, dict) else ext.get(f"{group}_{key}")
    return str(value).strip()[:200] if isinstance(value, (str, int)) and not isinstance(value, bool) else ""


def _automation_keys(turn: dict) -> list[str]:
    """The keys of the task v0 keeps for an automation turn: a scheduled
    task's by its session (or its run, or its job), an action's by its id, a
    linked system's message by its link and message id."""
    slug, auto, ext = turn["agent_slug"], turn["automation"], turn["ext_refs"]
    kind, aid = str(auto.get("kind") or ""), str(auto.get("id") or "")
    keys: list[str] = []
    if kind == "scheduled":
        if turn["session_id"]:
            keys.append(f"session:{slug}:{turn['session_id']}")
        if aid:
            keys += [f"sched:run:{aid}", f"sched:job:{aid}"]
    elif kind == "autonomy" and aid:
        keys.append(f"autonomy:{aid}")
    elif kind == "link":
        link_id, msg_id = _ref(ext, "link", "id"), _ref(ext, "link", "msg_id")
        if link_id and msg_id:
            keys.append(f"link:{link_id}:{msg_id}")
        if ":" in aid:
            keys.append(f"link:{aid}")
    return keys


def _gmail_verdict(turn: dict) -> tuple[str, dict | None]:
    """Did a Gmail turn's email pass ``gmail_inbound.prefilter``? The
    auto-reply engine runs it on every email it reads and keeps the verdict
    (``skipped`` with the check that failed). ``(why not, sender)``: "" and
    the sender (a header address: verified) when it passed; an email with no
    verdict fails closed."""
    from captain_claw.gmail_inbound import CHECK_ERROR, PREFILTER_CODES

    mid = _ref(turn["ext_refs"], "gmail", "message_id")
    if not mid:
        return "gmail_unchecked", None
    mine, args = ("", []) if not taskboard_sources._auth_enabled() else (" AND owner_id = ?", [turn["agent_owner_id"]])
    try:
        rows = _rows(taskboard_sources._fd_db_path(), "SELECT state, reason, from_addr FROM gmail_autoreplies "
                     f"WHERE inbound_message_id = ?{mine} ORDER BY updated_at DESC LIMIT 1", (mid, *args))
    except Exception:
        rows = []
    if not rows or rows[0]["state"] == "new":
        return "gmail_unchecked", None
    if rows[0]["state"] == "skipped" and rows[0]["reason"] in (*PREFILTER_CODES, CHECK_ERROR):
        return f"gmail:{rows[0]['reason']}", None
    sender = str(rows[0]["from_addr"] or "").strip().lower()
    return "", ({"role": "requester", "kind": "email", "address": sender,
                 "display": turn["speaker_label"] or sender, "verified": True} if "@" in sender else None)


def _requester(turn: dict) -> dict:
    """Who asked: the outside address the turn came in from — verified for a
    WhatsApp number and a Telegram id (the bridges', not the model's) — taken
    only from what this turn's own channel names (the session's stored origin
    is a WhatsApp contact's even when the owner types in its web chat); else
    the person on this board who spoke (an FD user), but only on a channel of
    the board user's own or as a confirmed member. Anyone else — a stranger
    on Messenger, a visitor on the public link, Discord, a bridge turn whose
    id didn't come — is a name, unverified: never the board user's own words."""
    ext, name, channel = turn["ext_refs"], turn["speaker_label"], turn["channel"]
    origin = ext.get("origin") if isinstance(ext.get("origin"), dict) else {}
    okind = _ORIGIN_CHANNELS.get(str(origin.get("kind") or "").strip().lower(), "")
    address = str(origin.get("address") or "").strip()[:320] if okind and okind == channel else ""
    party = {"role": "requester", "display": name}
    if channel == "whatsapp":
        waid = re.sub(r"\D", "", str(ext.get("waid") or "")) or re.sub(r"\D", "", address)
        if waid:
            return {**party, "kind": "waid", "address": waid, "display": name or f"+{waid}", "verified": True}
    if channel == "telegram":
        chat = _ref(ext, "telegram", "chat_id") or address
        if chat:
            return {**party, "kind": "tg_id", "address": chat, "display": name or chat,
                    "verified": bool(re.fullmatch(r"\d{1,20}", chat))}  # a group's id is negative: not a person
    if address and okind == "gmail":
        return {**party, "kind": "email", "address": address.lower(), "display": name or address, "verified": False}
    if address and okind == "waweb":
        return {**party, "kind": "wa_jid", "address": address, "display": name or address, "verified": False}
    if _own_turn(turn):
        return {**party, "kind": "fd_user", "address": turn["speaker_id"] or turn["user_id"],
                "display": name or "You", "verified": True}
    label = name or turn["source_label"] or CHANNEL_LABELS.get(channel.partition(":")[0], "") or (
        turn_source.label_for(turn["source"]) or "Someone")
    return {**party, "kind": "name", "address": _fold(label)[:320] or "someone", "display": label,
            "verified": False}


def _own_turn(turn: dict) -> bool:
    """Whether the words are the board user's own: a confirmed member's on
    a shared agent (the board is theirs), or said where only the board user
    speaks (:func:`_own_channel`)."""
    return bool(turn["member"] and turn["speaker_id"]) or _own_channel(turn["channel"], turn["source"])


def _own_channel(channel: str, source: str) -> bool:
    """A channel only the board user speaks on: their chat, the app, their
    glasses and devices, their API and MCP clients — and, with no channel, a
    source that isn't someone else's (the command line)."""
    base = channel.partition(":")[0]
    if base:
        return base in _OWN_CHANNELS
    return turn_source.channel_of(source) not in _OUTSIDE_SOURCES


def _rule(store: TaskboardStore, turn: dict, requester: dict) -> dict | None:
    """The user's task rule that decides this turn: of those whose every
    field matches, the most specific (sender, then domain, agent, channel,
    source) — at the same, not a task before mute before always a task.
    A sender or domain matches a verified outside address only."""
    facts = {"agent": turn["agent_slug"].lower(), "channel": turn["channel"], "source": turn["source"]}
    if requester.get("verified") and requester["kind"] != "fd_user":
        facts["sender"] = requester["address"].lower()
        if requester["kind"] == "email":
            facts["domain"] = requester["address"].rpartition("@")[2]
    best, rank = None, None
    for rule in store.rules(turn["user_id"]):
        match, effect = rule["match"], rule["effect"]
        if effect not in _RULE_EFFECTS or not match or not all(_matches(d, v, facts) for d, v in match.items()):
            continue
        r = (min(_RULE_DIMS.index(d) for d in match), _RULE_EFFECTS.index(effect))
        if rank is None or r < rank:
            best, rank = rule, r
    return best


def _matches(dim: str, value: Any, facts: dict[str, str]) -> bool:
    want = str(value or "").strip().lower()
    have = facts.get(dim)
    if dim not in _RULE_DIMS or not want or not have:
        return False
    if dim in ("channel", "source"):
        return have == want or have.partition(":")[0] == want
    if dim == "domain":
        return have == want or have.endswith("." + want)
    return have == want


def _admit(store: TaskboardStore, turn: dict, requester: dict, effect: str, agents: dict[str, dict]) -> str:
    """A person's turn as a task of its own: what they asked (the first line
    its title until extraction writes a better one), owed by the agent and
    queued for triage — or, muted by a rule, kept quietly (nobody owes it,
    never triaged)."""
    tid, text = turn["turn_id"], turn["text_head"]
    first = next((line.strip() for line in text.splitlines() if line.strip()), "")
    channel = turn["channel"]
    label = CHANNEL_LABELS.get(channel.partition(":")[0], channel)
    muted = effect == "mute"
    info = agents.get(turn["agent_slug"]) or {}
    create = {
        "title": first[:160] or (f"Message on {label}" if label else "Message"), "ask": _head(text, 600),
        "agent_slug": turn["agent_slug"], "agent_name": info.get("name") or turn["agent_slug"],
        "agent_ref": info.get("ref") or "", "agent_owner_id": turn["agent_owner_id"],
        "kind": "request", "direction": "received", "channel": channel, "status": "open",
        "owes_next": "nobody" if muted else "agent", "triage_state": "skipped" if muted else "pending",
        "triage_after": time.time(), "private": int(turn["private"]), "source": "agent_turns", "source_ref": tid,
        "flags": {"rule": effect} if effect else {},
    }
    task_id, _ = store.ingest(
        turn["user_id"], keys=[f"turn:{tid}"], create=create, parties=[requester],
        event={"source": "agent_turns", "source_ref": f"{tid}:open", "at": turn["started_at"], "kind": "received",
               "channel": channel, "actor_kind": "me" if requester["kind"] == "fd_user" else "party",
               "actor": requester["display"], "summary": _head(text, 500) or "(no text)",
               "private": turn["private"], "refs": {"turn_id": tid, "session_id": turn["session_id"]}})
    return task_id or ""


def _requeue(store: TaskboardStore, uid: str, task_id: str, now: float) -> dict | None:
    """The patch that puts a turn's task back in the triage queue — only a
    task a turn made (kind request) that is still open, and never one a
    mute rule keeps quiet (never triaged)."""
    task = store.get_task(uid, task_id) if task_id else None
    if task is None or task["kind"] != "request" or task["status"] not in ("open", "maybe"):
        return None
    if task["triage_state"] == "skipped" or _json(task.get("flags"), {}).get("rule") == "mute":
        return None
    return {"triage_state": "pending", "triage_after": now}


def _reply(store: TaskboardStore, turn: dict, agents: dict[str, dict], now: float) -> bool:
    """A turn's close on its task: the reply, how the turn ended and the
    files it saved (as refs) — once. True when that queued the task for
    triage again."""
    uid, tid, outcome = turn["user_id"], turn["turn_id"], turn["outcome"]
    info = agents.get(turn["agent_slug"]) or {}
    patch = _requeue(store, uid, turn["task_id"], now)
    _, new = store.ingest(
        uid, keys=[turn["task_key"]], patch=patch,
        event={"source": "agent_turns", "source_ref": f"{tid}:close", "at": turn["ended_at"], "kind": "reply",
               "channel": turn["channel"], "actor_kind": "agent", "actor": info.get("name") or turn["agent_slug"],
               "summary": _head(turn["reply_head"], 500) or _END_SUMMARY.get(outcome, "Answered"),
               "outcome": outcome if outcome in ("ok", "error", "blocked") else "", "private": turn["private"],
               "refs": {"turn_id": tid, "outcome": outcome, "deliverables": turn["deliverables"],
                        "tool_calls": turn["tool_calls"], "failed_tool_calls": turn["failed_tool_calls"]}})
    return new and patch is not None


# ── Sweep adapters (taskboard_sources.ADAPTERS) ───────────────────────


def waiting_turns(p: taskboard_sources._Pass) -> None:
    """Automation turns whose task the sweep hadn't read yet: on it now (or,
    six hours on, given up). Their tasks are v0's own, never triaged."""
    for turn in p.store.waiting_turns(user_id=p.user_id if p.auth else None):
        settle(p.store, turn["turn_id"], agents=p.agents)
        if (p.store.get_turn(turn["turn_id"]) or {}).get("gate") == "attached":
            p.new_events["agent_turns"] = p.new_events.get("agent_turns", 0) + 1


def _audit_path() -> Path:
    from captain_claw.flight_deck import tool_audit_routes

    if tool_audit_routes._store is not None:
        return Path(tool_audit_routes._store.db_path)
    return taskboard_store.data_dir() / "tool_audit.db"  # the register's, beside the board


def _call_board(p: taskboard_sources._Pass, row: dict) -> str | None:
    """The board a call belongs on, by the rule a turn's goes by: the member's
    whose turn it was (the register verified them), else the owner's."""
    if not p.auth:
        return "local"
    if row.get("member"):
        return str(row.get("speaker_id") or "") or None
    return str(row.get("owner_id") or "") or None


def tool_calls(p: taskboard_sources._Pass) -> None:
    """The tool-call register's calls made in a turn, onto that turn's task
    (an automation turn's: its v0 task) — once each call has ended. A call
    whose turn isn't here yet, or waits for its task, is read again on the
    next passes (``waiting`` in the cursor); one that never finds a task is
    let go."""
    path = _audit_path()
    if not path.is_file():
        return
    if "turn_id" not in {r["name"] for r in _rows(path, "PRAGMA table_info(tool_calls)")}:
        return  # calls from before v1 name no turn
    cur = _json(p.cursor("tool_audit"), {})
    after = int(cur.get("id") or 0)
    waiting = [int(i) for i in cur.get("waiting") or []][-_CALLS_WAITING_KEPT:]
    mine, mine_args = ("", []) if p.user_id is None or not p.auth else (
        " AND (owner_id = ? OR speaker_id = ?)", [p.user_id, p.user_id])
    rows = _rows(path, "SELECT * FROM tool_calls WHERE turn_id != '' "
                       f"AND (id > ? OR id IN ({','.join('?' * len(waiting)) or 'NULL'})){mine} ORDER BY id LIMIT ?",
                 (after, *waiting, *mine_args, taskboard_sources._BATCH))
    now = time.time()
    still: list[int] = []
    kicked: set[str] = set()
    for row in rows:
        verdict = _call(p, row, now)
        if verdict == "wait":
            still.append(int(row["id"]))
        elif verdict:
            kicked.add(verdict)
    if rows or waiting:
        p.save_cursor("tool_audit", json.dumps({"id": max([after, *(int(r["id"]) for r in rows)]),
                                                "waiting": still[-_CALLS_WAITING_KEPT:]}))
    for uid in kicked:
        _kick(uid)


def _call(p: taskboard_sources._Pass, row: dict, now: float) -> str:
    """One call onto its turn's task: ``wait`` (read it again), the board
    user when it was new on a task triage reads, or "" (done with it)."""
    from captain_claw.tool_audit import clean_refs

    age = now - float(row.get("received_at") or now)
    if row.get("outcome") == "running":
        return "wait" if age < _TURN_WAIT_S else ""
    turn = p.store.get_turn(str(row.get("turn_id") or ""))
    if turn is None or (turn["started_at"] is None and not turn["gate"]):
        return "wait" if age < _CALL_WAIT_S else ""  # its turn's open hasn't come yet
    if turn["agent_slug"] != row.get("agent_slug") or turn["agent_owner_id"] != str(row.get("owner_id") or ""):
        return ""  # another agent's turn: never on its task
    if turn["gate"] == "waiting":
        return "wait" if now - float(turn["received_at"] or now) < _TURN_WAIT_S else ""
    if turn["gate"] not in ("task", "attached") or not turn["task_key"]:
        return ""
    uid = _call_board(p, row)
    if uid is None or uid != turn["user_id"]:
        return ""  # a member's call never on the owner's board, nor the owner's on a member's
    if p.user_id is not None and p.auth and uid != p.user_id:
        return ""
    tool = str(row.get("tool") or "") or "(unknown)"
    params = _json(row.get("params"), {})
    action = params.get("action") if isinstance(params.get("action"), str) else ""
    action = action if _ACTION_RE.match(action or "") else ""
    outcome = str(row.get("outcome") or "")
    sent = outcome == "ok" and row.get("risk") == "send" and tool not in _NOT_DELIVERIES
    refs = clean_refs(_json(row.get("refs"), {}))  # the result ids the register keeps
    # A person's request is the task its thread is about; an automation's
    # task (a scheduled job's one standing task) would gather every thread
    # the job ever mailed, and a reply on one would land there.
    thread = refs.get("gmail_thread_id", "") if turn["gate"] == "task" else ""
    patch = _requeue(p.store, uid, turn["task_id"], now)
    _, new = p.ingest(
        "tool_audit", uid, keys=[turn["task_key"]], patch=patch,
        extra_keys=[f"gmail:thread:{thread}"] if thread else [],
        event={"source": "tool_audit", "source_ref": str(row["id"]), "at": float(row.get("started_at") or now),
               "kind": "sent" if sent else "tool_call", "channel": _SEND_CHANNELS.get(tool, "") if sent else "",
               "actor_kind": "agent", "actor": str(row.get("agent_name") or row.get("agent_slug") or ""),
               "summary": f"{tool} {action}".strip() + f": {_CALL_OUTCOMES.get(outcome, outcome)}",
               "outcome": {"ok": "ok", "blocked": "blocked"}.get(outcome, "error"),
               "evidence": "recorded" if sent else "", "private": bool(row.get("private")),
               "refs": {"tool_call_id": row["id"], "turn_id": turn["turn_id"], "tool": tool, "action": action,
                        "risk": str(row.get("risk") or ""), **refs}})
    return uid if new and patch else ""
