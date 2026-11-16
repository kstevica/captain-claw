"""Which Autonomous Work nudges reach the user's phone (WhatsApp, Telegram).

Every nudge still lands in the agent's chat, the Line and the bell. The phone
gets only the ones worth a ping. On a busy day the arbiter made a dozen nudges
in a few hours, and most were noise on a phone: the same photo shoot twice in
nine minutes, an echo of what the user had just asked on WhatsApp, or "I can't
see the memo from here". Proposals waiting for approval are not gated: they
always ring (owner_notify).

The decision, in order:

1. **Rules that need no model** (each one keeps the nudge off the phone):

   * *couldn't check*: the message opens with the assistant saying it can't see,
     open or check something (``couldnt_check_lead``);
   * *repeat*: a nudge about the same email thread, calendar event or
     follow-up already reached the phone within ``phone_repeat_hours``;
   * *chatting*: the user wrote on WhatsApp or Telegram within
     ``phone_hold_while_chatting_min`` — the nudge waits until they go quiet
     (at most ``HOLD_MAX_S``), then all of this runs again, so a nudge that
     only echoed the conversation is caught by then.

2. **Jev first** (the owner's key, ``jev_routes.ask_for_owner``): three yes/no
   questions over the message, what reached the phone recently and the recent
   conversation on the phone channels. Sent when ``needs_user_soon`` is likely
   and ``already_told`` / ``couldnt_check`` are not.

3. **Rules when Jev can't answer** (not connected, an error, a timeout): the
   phone gets a nudge about an email someone sent (it has a Gmail thread), a
   calendar event starting within the day, or one whose title or reason says
   today / tonight / urgent. Anything else waits for the app.

Each decision is written on the action (``payload.phone``) and in the
Autonomous Work log. ``phone_filter: false`` sends every nudge, as before.

The recent conversation is what the bridges hand in (``note_inbound`` /
``note_outbound``), kept in memory for a few hours: a restart forgets it, and
a nudge held while the user was chatting is then not sent (it stays in the
chat and on the Line).
"""

from __future__ import annotations

import asyncio
import re
import time
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from captain_claw.logging import get_logger

log = get_logger(__name__)

JEV_SOURCE = "phone_gate"
JEV_TIMEOUT_S = 8.0
# Jev's thresholds: sending a phone message is easy to live with either way,
# so these are looser than the "consequential action" 0.8 / 0.2.
SEND_MIN = 0.6          # needs_user_soon at or above → may go
TOLD_MAX = 0.5          # already_told above → stays
CANT_MAX = 0.5          # couldnt_check above → stays

HOLD_MAX_S = 45 * 60    # longest a nudge waits for the user to go quiet
HOLD_POLL_S = 60.0
EVENT_SOON_H = 24       # rules: a calendar event starting within this many hours
RECENT_SENT_H = 12      # what reached the phone that Jev reads (at least)
RECENT_SENT_MAX = 8
CONVO_KEEP = 12         # messages per channel
CONVO_MAX_AGE_S = 3 * 3600
CONVO_READ_S = 2 * 3600  # how far back Jev reads the conversation
CONVO_TEXT_MAX = 500
MESSAGE_MAX = 3000
SENT_TEXT_MAX = 300

# Why a nudge stayed off the phone (``payload.phone.why``) or went.
COULDNT_CHECK = "couldnt_check"
REPEAT = "repeat"
CHATTING = "chatting"
JEV_SEND = "jev:send"
JEV_NOT_URGENT = "jev:can_wait"
JEV_ALREADY_TOLD = "jev:already_told"
JEV_COULDNT_CHECK = "jev:couldnt_check"
RULES_EMAIL = "rules:email"
RULES_EVENT_SOON = "rules:event_soon"
RULES_TODAY = "rules:today"
RULES_CAN_WAIT = "rules:can_wait"
FILTER_OFF = "filter_off"


@dataclass
class Verdict:
    """``send``: ring the phones now. ``held``: waiting for the user to go
    quiet (a task decides later). ``why`` is one of the codes above."""

    send: bool
    why: str
    via: str = "rules"           # rules | jev | off
    held: bool = False
    p: dict[str, float] = field(default_factory=dict)


# ── the conversation on the phone channels (bridges hand it in) ─────────

_convo: dict[str, deque[tuple[float, str, str]]] = {}
_last_in: dict[str, float] = {}


def wa_key(waid: str) -> str:
    return f"wa:{str(waid or '').lstrip('+').strip()}"


def tg_key(chat: Any) -> str:
    return f"tg:{str(chat or '').strip()}"


def _push(key: str, who: str, text: str, *, replace_last: bool = False) -> None:
    text = " ".join(str(text or "").split())[:CONVO_TEXT_MAX]
    if not key or not text:
        return
    ring = _convo.setdefault(key, deque(maxlen=CONVO_KEEP))
    now = time.time()
    while ring and now - ring[0][0] > CONVO_MAX_AGE_S:
        ring.popleft()
    if replace_last and ring and ring[-1][1] == who:
        ring[-1] = (now, who, text)
    else:
        ring.append((now, who, text))


def note_inbound(key: str, text: str = "") -> None:
    """The user wrote on a phone channel (``wa_key`` / ``tg_key``). Never raises."""
    try:
        if key:
            _last_in[key] = time.time()
            _push(key, "user", text)
    except Exception:  # noqa: BLE001
        pass


def note_outbound(key: str, text: str, *, replace_last: bool = False) -> None:
    """The assistant answered on a phone channel; ``replace_last`` for an edit
    of the message it sent last (Telegram's live narration). Never raises."""
    try:
        _push(key, "assistant", text, replace_last=replace_last)
    except Exception:  # noqa: BLE001
        pass


def last_inbound(keys: list[str]) -> float:
    return max((_last_in.get(k, 0.0) for k in keys), default=0.0)


def recent_conversation(keys: list[str], since_s: float = CONVO_READ_S) -> list[dict[str, str]]:
    cutoff = time.time() - since_s
    rows = sorted(
        (row for k in keys for row in _convo.get(k, ()) if row[0] >= cutoff),
        key=lambda r: r[0],
    )[-CONVO_KEEP:]
    return [{"at": _iso(ts), "from": who, "text": text} for ts, who, text in rows]


# ── deterministic rules ─────────────────────────────────────────────────

_MARKUP = re.compile(r"[*_`#>]+")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+|\n+")
_CANT_LEAD = re.compile(
    r"^(?:sorry,?\s+)?i\s*(?:can(?:no|['’])t|could(?:n['’]t| not)|(?:am|['’]m)\s+(?:unable|not able)"
    r"|do(?:n['’]t| not) have|have(?:n['’]t| not) (?:been able|opened|seen|found|got))\b",
    re.I,
)
_SESSION_BLIND = re.compile(
    r"\b(?:in|from) this (?:session|chat|turn)\b|\bfrom here\b|\bnot visible\b", re.I)
_NEGATION = re.compile(r"\b(?:can(?:no|['’])t|isn['’]t|aren['’]t|not|don['’]t|haven['’]t|no)\b", re.I)
_TODAY = re.compile(
    r"\b(?:today|tonight|this (?:morning|afternoon|evening)|in an hour|within the hour"
    r"|in \d+ ?(?:min|mins|minutes|hours?)|urgent|asap|overdue)\b", re.I)


def lead_sentence(text: str) -> str:
    """The message's first sentence, markdown stripped."""
    plain = _MARKUP.sub("", str(text or "")).strip()
    plain = re.sub(r"^[-•\d.)\s]+", "", plain)
    return (_SENTENCE_END.split(plain, maxsplit=1)[0] if plain else "").strip()


def couldnt_check_lead(text: str) -> bool:
    """The message OPENS with the assistant saying it can't see / open / check
    something ("I can't see the memo from here", "The memo isn't in this
    session, so I can't check it") — nothing found, nothing for the phone."""
    lead = lead_sentence(text)
    if not lead:
        return False
    if _CANT_LEAD.search(lead):
        return True
    return bool(_SESSION_BLIND.search(lead) and _NEGATION.search(lead))


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, UTC).isoformat(timespec="minutes")


def _dt(value: Any) -> datetime | None:
    s = str(value or "").strip()
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


@dataclass
class _Ground:
    """What the nudge is about, as far as the data says."""

    subject: str = ""            # "gmail:<thread>" | "cal:<event id>" | "fu:<id>" | ""
    kind: str = ""               # email | event | follow_up | ""
    start: datetime | None = None  # a calendar event's start
    what: str = "nothing specific"


def ground_of(action: dict[str, Any]) -> _Ground:
    """The email thread, calendar event or follow-up a nudge is about — the
    key that tells two differently worded nudges apart from one repeated."""
    payload = action.get("payload") if isinstance(action.get("payload"), dict) else {}
    g = _Ground()
    ev: dict[str, Any] | None = None
    ref = str(payload.get("event_ref") or "").strip()
    if ref:
        try:
            from captain_claw.flight_deck.events import get_store as events_store

            ev = events_store().get_event(ref)
        except Exception as exc:  # noqa: BLE001
            log.debug("phone gate: event lookup failed", error=type(exc).__name__)
    thread = str(payload.get("thread_key") or "").strip()
    if not thread and ev and ev.get("source") == "gmail":
        from captain_claw.flight_deck.arbiter import _thread_key

        thread = _thread_key(ev)
    if thread:
        g.subject, g.kind, g.what = thread, "email", "an email someone sent the user"
        return g
    if ev and ev.get("source") == "calendar":
        md = ev.get("metadata") if isinstance(ev.get("metadata"), dict) else {}
        event_id = str(md.get("event_id") or "").strip()
        g.start = _dt(md.get("start"))
        g.kind = "event"
        g.subject = f"cal:{event_id}" if event_id else ""
        g.what = "a calendar event" + (f" starting {g.start.isoformat(timespec='minutes')}"
                                       if g.start else "")
        return g
    fu = str(payload.get("follow_up_id") or "").strip()
    if fu:
        g.subject, g.kind, g.what = f"fu:{fu}", "follow_up", "a follow-up the user is tracking"
    return g


def _phoned(store: Any, user_id: str, hours: float, exclude: str) -> list[dict[str, Any]]:
    """Nudges that reached the phone within *hours*, newest first."""
    cutoff = datetime.now(UTC) - timedelta(hours=hours)
    out: list[dict[str, Any]] = []
    for a in store.list_actions(user_id, limit=150):
        if a.get("id") == exclude or a.get("kind") != "nudge":
            continue
        ph = (a.get("payload") or {}).get("phone")
        if not isinstance(ph, dict) or not ph.get("sent"):
            continue
        at = _dt(ph.get("at"))
        if at is None or at < cutoff:
            continue
        out.append({"at": ph.get("at"), "subject": str(ph.get("subject") or ""),
                    "title": str(a.get("title") or ""),
                    "text": str(ph.get("text") or a.get("outcome_note") or "")[:SENT_TEXT_MAX]})
    return out


def rules_verdict(action: dict[str, Any], ground: _Ground, now: datetime | None = None) -> Verdict:
    """Jev couldn't answer: send only what plainly can't wait."""
    now = now or datetime.now(UTC)
    if ground.kind == "email":
        return Verdict(True, RULES_EMAIL)
    if ground.kind == "event" and ground.start is not None:
        if now - timedelta(hours=1) <= ground.start <= now + timedelta(hours=EVENT_SOON_H):
            return Verdict(True, RULES_EVENT_SOON)
    if _TODAY.search(f"{action.get('title') or ''} {action.get('rationale') or ''}"):
        return Verdict(True, RULES_TODAY)
    return Verdict(False, RULES_CAN_WAIT)


# ── Jev ─────────────────────────────────────────────────────────────────

QUESTIONS: dict[str, dict[str, Any]] = {
    "needs_user_soon": {
        "type": "noul",
        "instructions": (
            "The assistant wants to send `message` to the user's phone, unasked. Does it need "
            "the user's attention within the next few hours?"
        ),
        "yes_means": (
            "Someone is waiting on the user's reply, the user must decide or approve something "
            "soon, or a deadline or an event is today or early tomorrow and needs something "
            "from the user."
        ),
        "no_means": (
            "Information with nothing to do, general advice or a reminder that can wait until "
            "the user next opens the app."
        ),
    },
    "already_told": {
        "type": "noul",
        "instructions": (
            "Was the user already told what `message` says — is it about the same matter as an "
            "item in `sent_to_phone_recently`, or something the user and the assistant just "
            "discussed in `recent_conversation`?"
        ),
        "yes_means": "Same matter, even in other words; nothing important is new.",
        "no_means": "A new matter, or a real change the user hasn't heard about.",
    },
    "couldnt_check": {
        "type": "noul",
        "instructions": (
            "Is `message` mainly the assistant saying it couldn't see, open, find or check "
            "something, rather than reporting something it found?"
        ),
    },
}


def jev_state(action: dict[str, Any], text: str, ground: _Ground,
              sent: list[dict[str, Any]], convo: list[dict[str, str]],
              now: datetime | None = None) -> dict[str, Any]:
    now = now or datetime.now(UTC)
    return {
        "now_utc": now.strftime("%a %d %b %Y, %H:%M"),
        "message": str(text or "")[:MESSAGE_MAX],
        "about": str(action.get("title") or "")[:300],
        "why_the_assistant_wants_to_send_it": str(action.get("rationale") or "")[:500],
        "grounded_on": ground.what,
        "sent_to_phone_recently": [
            {"at": s["at"], "message": s["text"] or s["title"]} for s in sent[:RECENT_SENT_MAX]
        ],
        "recent_conversation": convo,
    }


async def ask_jev(user_id: str, state: dict[str, Any]) -> dict[str, float] | None:
    """Jev's three probabilities, or None when it can't answer."""
    from captain_claw import jev_client
    from captain_claw.flight_deck import jev_routes

    try:
        out = await asyncio.wait_for(
            jev_routes.ask_for_owner(user_id, state, QUESTIONS, source=JEV_SOURCE),
            JEV_TIMEOUT_S)
    except jev_routes.JevNotConnected:
        return None
    except (TimeoutError, jev_client.JevError) as exc:
        log.info("phone gate: Jev did not answer", error=type(exc).__name__)
        return None
    except Exception as exc:  # noqa: BLE001 — anything else: the rules decide
        log.warning("phone gate: Jev call failed", error=type(exc).__name__)
        return None
    values = out.get("values") if isinstance(out, dict) else None
    if not isinstance(values, dict):
        return None
    probs: dict[str, float] = {}
    for name in QUESTIONS:
        v = values.get(name)
        if not isinstance(v, (int, float)):
            return None
        probs[name] = float(v)
    return probs


def jev_verdict(p: dict[str, float]) -> Verdict:
    if p["couldnt_check"] > CANT_MAX:
        return Verdict(False, JEV_COULDNT_CHECK, "jev", p=p)
    if p["already_told"] > TOLD_MAX:
        return Verdict(False, JEV_ALREADY_TOLD, "jev", p=p)
    if p["needs_user_soon"] >= SEND_MIN:
        return Verdict(True, JEV_SEND, "jev", p=p)
    return Verdict(False, JEV_NOT_URGENT, "jev", p=p)


# ── the decision ────────────────────────────────────────────────────────


def channel_keys(user_id: str, cfg: dict[str, Any]) -> list[str]:
    """The user's phone channels: their WhatsApp number(s) and their Telegram
    chat with the deck's bot — where their conversation and typing show."""
    keys: list[str] = []
    try:
        from captain_claw.flight_deck.fd_dispatch import _nudge_waids

        keys += [wa_key(w) for w in _nudge_waids(cfg)[0]]
    except Exception:  # noqa: BLE001
        pass
    try:
        from captain_claw.flight_deck.fd_scheduler import _owner_telegram_chat

        chat = _owner_telegram_chat(user_id)
        if chat:
            keys.append(tg_key(chat))
    except Exception:  # noqa: BLE001
        pass
    return keys


def phones_on(user_id: str, cfg: dict[str, Any]) -> bool:
    """Would a nudge reach any phone at all? (No: nothing to decide.)"""
    try:
        from captain_claw.flight_deck.fd_dispatch import _nudge_waids

        if cfg.get("nudge_to_whatsapp", True) and _nudge_waids(cfg)[0]:
            return True
    except Exception:  # noqa: BLE001
        pass
    if cfg.get("notify_telegram", False):
        from captain_claw.flight_deck.fd_scheduler import _owner_telegram_chat

        return bool(_owner_telegram_chat(user_id))
    return False


async def judge(user_id: str, action: dict[str, Any], text: str, cfg: dict[str, Any],
                *, keys: list[str] | None = None, wait_for_quiet: bool = True) -> Verdict:
    """Decide one nudge. ``wait_for_quiet=False`` skips the chatting rule (the
    hold already waited)."""
    from captain_claw.flight_deck.autonomy import get_store

    if not cfg.get("phone_filter", True):
        return Verdict(True, FILTER_OFF, "off")
    if couldnt_check_lead(text):
        return Verdict(False, COULDNT_CHECK)
    ground = ground_of(action)
    store = get_store()
    repeat_h = max(0.0, float(cfg.get("phone_repeat_hours", 6) or 0))
    sent = _phoned(store, user_id, max(repeat_h, RECENT_SENT_H), str(action.get("id") or ""))
    if ground.subject and repeat_h > 0:
        cutoff = datetime.now(UTC) - timedelta(hours=repeat_h)
        for s in sent:
            at = _dt(s["at"])
            if s["subject"] == ground.subject and at is not None and at >= cutoff:
                return Verdict(False, REPEAT)
    keys = channel_keys(user_id, cfg) if keys is None else keys
    hold_s = max(0, int(cfg.get("phone_hold_while_chatting_min", 10) or 0)) * 60
    if wait_for_quiet and hold_s and keys and time.time() - last_inbound(keys) < hold_s:
        return Verdict(False, CHATTING, held=True)
    p = await ask_jev(user_id, jev_state(action, text, ground, sent, recent_conversation(keys)))
    if p is not None:
        return jev_verdict(p)
    return rules_verdict(action, ground)


_held: dict[str, asyncio.Task] = {}


def _record(user_id: str, action: dict[str, Any], text: str, v: Verdict) -> None:
    from captain_claw.flight_deck.autonomy import get_store

    store = get_store()
    aid = str(action.get("id") or "")
    phone = {"sent": v.send, "held": v.held, "why": v.why, "via": v.via,
             "at": datetime.now(UTC).isoformat(), "subject": ground_of(action).subject}
    if v.p:
        phone["p"] = {k: round(x, 3) for k, x in v.p.items()}
    if v.send:
        phone["text"] = str(text or "")[:SENT_TEXT_MAX]
    try:
        if aid:
            store.update_payload(aid, {"phone": phone})
        if v.why != FILTER_OFF:
            event = ("nudge → phone" if v.send else
                     "nudge → phone held" if v.held else "nudge kept off the phone")
            p = (" · " + ", ".join(f"{k} {x:.2f}" for k, x in v.p.items())) if v.p else ""
            store.log(user_id, event, f"{v.why}{p} — {str(action.get('title') or '')[:120]}")
    except Exception as exc:  # noqa: BLE001
        log.debug("phone gate: record failed", error=type(exc).__name__)


async def decide(user_id: str, action: dict[str, Any], text: str, cfg: dict[str, Any],
                 ring_later: Callable[[], Awaitable[Any]]) -> Verdict:
    """Decide whether this nudge rings the phones now. When the user is
    chatting, the nudge is held: a task waits until they go quiet, decides
    again and calls ``ring_later`` if it is still worth a ping. Never raises."""
    try:
        if not phones_on(user_id, cfg):
            # Nothing to decide (and no Jev call): notify_owner goes on as
            # always, logging why WhatsApp was skipped.
            return Verdict(True, "no_phone", "off")
        keys = channel_keys(user_id, cfg)
        v = await judge(user_id, action, text, cfg, keys=keys)
    except Exception as exc:  # noqa: BLE001 — the gate never loses a nudge to a bug
        log.warning("phone gate failed; sending as before", error=type(exc).__name__)
        return Verdict(True, "gate_error", "off")
    _record(user_id, action, text, v)
    if v.held:
        aid = str(action.get("id") or id(action))
        if aid not in _held:
            task = asyncio.create_task(
                _release_when_quiet(user_id, action, text, cfg, keys, ring_later))
            _held[aid] = task
            task.add_done_callback(lambda _t, k=aid: _held.pop(k, None))
    return v


async def _release_when_quiet(user_id: str, action: dict[str, Any], text: str,
                              cfg: dict[str, Any], keys: list[str],
                              ring_later: Callable[[], Awaitable[Any]]) -> None:
    hold_s = max(0, int(cfg.get("phone_hold_while_chatting_min", 10) or 0)) * 60
    give_up = time.time() + HOLD_MAX_S
    try:
        while time.time() < give_up:
            wait = last_inbound(keys) + hold_s - time.time()
            if wait <= 0:
                break
            await asyncio.sleep(min(wait, HOLD_POLL_S))
        v = await judge(user_id, action, text, cfg, keys=keys, wait_for_quiet=False)
        _record(user_id, action, text, v)
        if v.send:
            await ring_later()
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001
        log.warning("phone gate: held nudge not released", error=type(exc).__name__)
