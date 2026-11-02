"""A client on the glasses channel bus that talks to one of its owner's agents.

The part a handheld device (device_bridge.py) and a connected system
(agent_link.py) share: one socket per channel, bound to an agent through the
glasses channel bus (glasses_bridge.py), sending turns to it and getting its
answer back — kept across a dropped socket, so a reconnect still gets it.

What each kind of client does with the answer (speak it, show it on a tiny
screen, hand it to a program) is the subclass's: the hooks below.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import re
import secrets
import time
from typing import Any

from fastapi import WebSocket

from captain_claw.flight_deck import glasses_bridge as gb
from captain_claw.flight_deck import usage_quota

TURN_KEEP_S = 600.0              # a turn (its answer) waits this long for its client to come back
HEARD_AFTER_S = 2.0              # a client message this long after an answer: it got there
POWER_CHECK_S = 5.0              # the agent's link down: how often its client hears asleep / stopped again

_conn_serials = itertools.count(1)

# Emoji and symbol marks the device font can't draw (agent activity uses a few).
_NO_GLYPH = re.compile("[\U0001F000-\U0001FFFF\u2600-\u27BF\uFE0F\u200D]")
# Runtime statuses that are turn bookkeeping, not activity worth showing.
_QUIET_STATUSES = {"", "thinking", "ready", "idle", "done", "completed", "agent_connected",
                   "agent_reconnecting", "subscribed"}


def activity_text(text: str, limit: int = 120) -> str:
    """An activity line as the device shows it: one line, no emoji, "…"."""
    out = _NO_GLYPH.sub("", str(text or "")).replace("\n", " ")
    out = re.sub(r"\.\.\.$", "…", re.sub(r"\s{2,}", " ", out).strip())
    return out[:limit]


def running_agents(user_id: str) -> list[dict]:
    """The owner's agents that are up and can take chat — asleep ones too
    (agent_idle): the first message wakes them."""
    from captain_claw.flight_deck.mcp_server_routes import _user_agents

    return [
        a for a in _user_agents(user_id)
        if a.get("port") and str(a.get("status", "")).lower() in ("running", "up", "asleep")
    ]


def agent_power(agent: dict | None) -> str:
    """Whether a process agent is "running", "asleep" (put to sleep for being
    idle: the next word wakes it) or "stopped" (switched off, or gone); ""
    for one Flight Deck can't tell about (a container)."""
    slug = str((agent or {}).get("slug") or "")
    if not slug or (agent or {}).get("kind") == "docker":
        return ""
    from captain_claw.flight_deck import server as srv

    try:
        entry = srv._load_process_registry().get(slug)
        if entry is None:
            return "stopped"
        # Marked asleep before it's stopped (agent_idle): the link that drops
        # as it goes down already reads asleep.
        if entry.get("asleep") and not entry.get("stopped"):
            return "asleep"
        return "running" if srv._process_is_alive(slug) else "stopped"
    except Exception:
        return ""


class BusClient:
    """One client socket on a channel of its own (``channel_id``), bound to
    one of its owner's agents. A turn is open from the send until the agent
    says "ready"; its answer is kept in the subclass's ``turns()`` until the
    client is heard from after it."""

    MSG_ID_PREFIX = "bus_"   # the chat frame's client_msg_id
    TAKES_PROACTIVE = True   # the agent's own messages (a cron result) come here too

    def __init__(self, ws: WebSocket, user: dict) -> None:
        self.ws = ws
        self.user = user
        self.user_id = str(user.get("id", ""))
        self.ch: gb._ChannelState | None = None
        self.agent: dict | None = None
        self.serial = next(_conn_serials)
        self.closed = False  # the link is gone; a turn still open keeps listening
        self.opened_at = self.last_heard_at = time.monotonic()
        self._send_lock = asyncio.Lock()
        # A turn is open from our send until the agent says "ready".
        self.pending = False
        self.last_reply_at = 0.0
        self._last_activity = ""
        self._bg: set[asyncio.Task] = set()
        # The agent's power as last told ("running" / "asleep" / "stopped"),
        # looked at again while its link is down.
        self._power = ""
        self._power_watch: asyncio.Task | None = None

    # ── what each kind of client says for itself ──
    @property
    def channel_id(self) -> str:
        raise NotImplementedError

    def turns(self) -> dict[str, dict]:
        """The open (or last) turn per channel of this kind, kept across reconnects."""
        raise NotImplementedError

    def pick_agent(self, selector: str) -> dict | None:
        raise NotImplementedError

    def no_agent_text(self, selector: str) -> str:
        return "No running agent. Start one in Flight Deck."

    async def _no_agent(self) -> None:
        """There's no agent to bind (before no_agent_text goes)."""

    def chat_fields(self, ch: Any, text: str, ctx: Any, extra_notes: list[str] | None) -> dict:
        """The chat frame's fields besides its type and id: content, surface, source, notes."""
        raise NotImplementedError

    def _turn_started(self, text: str, turn: dict) -> None:
        """A turn just went to the agent."""

    def _command_answer(self, command: str, text: str) -> tuple[str | None, bool]:
        """What to show for a slash command's answer: (text, closes the turn);
        None drops it."""
        return text, False

    async def _show_reply(self, text: str, proactive: bool, turn: dict | None = None) -> None:
        raise NotImplementedError

    async def _on_delta(self, text: str) -> None:
        """The answer's text as the model writes it."""

    async def _reset_stream(self) -> None:
        """Text the model wrote before a tool call wasn't the answer."""

    def _forget_stream(self) -> None:
        """A new answer starts."""

    async def _on_media(self, ev: dict) -> None:
        """Pictures / sounds the turn made (turn_media)."""

    async def stop_output(self) -> None:
        """Stop what is being played out to the client."""

    async def _resume_ready(self) -> bool:
        """Whether a reconnect should get the kept answer now (False: the
        client has moved on)."""
        return True

    async def _close_output(self) -> None:
        """The socket is gone: stop sending to it."""

    def _turn_state(self, turn: dict, state: str) -> dict:
        """The state frame a reconnect gets for a kept *turn* (its own, not the running one's)."""
        return {"type": "state", "state": state}

    # ── wire ──
    async def send_json(self, obj: dict) -> None:
        if self.closed:
            return
        async with self._send_lock:
            await self.ws.send_text(json.dumps(obj, default=str))

    async def send_bin(self, kind: int, payload: bytes) -> None:
        if self.closed:
            return
        async with self._send_lock:
            await self.ws.send_bytes(bytes([kind]) + payload)

    def _spawn(self, coro) -> asyncio.Task:
        task = asyncio.create_task(coro)
        self._bg.add(task)
        task.add_done_callback(self._bg.discard)
        return task

    # ── the agent's power ──
    async def _send_offline(self) -> None:
        """The agent's link is down: that, with whether it's asleep or stopped."""
        self._power = agent_power(self.agent)
        frame = {"type": "state", "state": "agent_offline"}
        if self._power:
            frame["power"] = self._power
        await self.send_json(frame)
        self._start_power_watch()

    def _start_power_watch(self) -> None:
        if self.ch is not None and self.agent is not None and (
                self._power_watch is None or self._power_watch.done()):
            self._power_watch = self._spawn(self._watch_power())

    async def _watch_power(self) -> None:
        """While the link is down: an agent that falls asleep, is switched off
        or goes is told as it happens (its link coming back says "connected")."""
        while not self.closed:
            await asyncio.sleep(POWER_CHECK_S)
            ch = self.ch
            if self.closed or ch is None or ch.agent_ws is not None or self.agent is None:
                return
            power = agent_power(self.agent)
            if power and power != self._power:
                self._power = power
                await self.send_json({"type": "state", "state": "agent_offline", "power": power})

    # ── binding ──
    async def bind(self, selector: str) -> bool:
        agent = self.pick_agent(selector)
        if agent is None:
            await self._no_agent()
            await self.send_json({"type": "error", "text": self.no_agent_text(selector)})
            return False
        ch = await gb._get_or_create_channel(self.channel_id)
        if self.ch is not ch:
            if self.ch is not None and self.on_bus in self.ch.callback_subscribers:
                self.ch.callback_subscribers.remove(self.on_bus)
            self.ch = ch
        # One client socket per channel listens (a reconnect replaces the old).
        ch.callback_subscribers[:] = [cb for cb in ch.callback_subscribers
                                      if getattr(cb, "__self__", None) is None
                                      or not isinstance(cb.__self__, BusClient)]
        ch.callback_subscribers.append(self.on_bus)
        await gb._ensure_agent_binding(ch, str(agent.get("host") or "localhost"), int(agent["port"]))
        self.agent = agent
        linked = ch.agent_ws is not None
        frame = {"type": "bound", "agent": agent.get("name", ""), "slug": agent.get("slug", ""), "linked": linked}
        self._power = "running" if linked else agent_power(agent)
        if self._power:
            frame["power"] = self._power
        await self.send_json(frame)
        if not linked:
            self._start_power_watch()  # (a pump already backing off says nothing new)
        return True

    def unbind(self) -> None:
        if self.ch is not None and self.on_bus in self.ch.callback_subscribers:
            self.ch.callback_subscribers.remove(self.on_bus)

    # ── agent bus → client ──
    async def on_bus(self, ev: dict) -> None:
        etype = ev.get("type")
        if etype == "agent":
            proactive = self.TAKES_PROACTIVE and bool(ev.get("proactive")) and not ev.get("fd_delivers")
            if not (self.pending or proactive):
                return  # a turn from another surface on the same agent
            turn = self.turns().get(self.channel_id) if self.pending else None
            text = str(ev.get("text", "") or "")
            closes = False
            if ev.get("command"):
                text, closes = self._command_answer(str(ev["command"]), text)
                if text is None:
                    return
            if not text.strip():
                return
            await self._show_reply(text, proactive, turn)
            if closes:
                await self.send_json({"type": "state", "state": "ready"})
        elif etype == "narration":
            if self.pending and ev.get("text"):
                await self.send_json({"type": "status", "text": str(ev["text"])[:200]})
        elif etype == "agent_delta":
            if self.pending and not self.closed:
                await self._on_delta(str(ev.get("text", "") or ""))
        elif etype == "activity":
            if self.pending and ev.get("phase") != "done":
                tool = str(ev.get("tool", "") or "")
                if tool and tool != "progress":
                    await self._reset_stream()  # what the model wrote before a tool call isn't the answer
                await self._send_activity(str(ev.get("text", "")), tool)
        elif etype == "status":
            status = str(ev.get("status", ""))
            # "Using <tool>..." duplicates the tool's activity line (which says more).
            tool_start = status.startswith("Using ") and status.endswith("...")
            if self.pending and status.strip().lower() not in _QUIET_STATUSES and not tool_start:
                await self._send_activity(status, "")
            if status == "thinking" and self.pending:
                await self.send_json({"type": "state", "state": "thinking"})
            elif status == "ready" and self.pending:
                self._end_turn()
                await self.send_json({"type": "state", "state": "ready"})
            elif status == "agent_connected":
                self._power = "running"
                await self.send_json({"type": "state", "state": "connected", "power": "running"})
            elif status == "agent_reconnecting":
                await self._send_offline()
                await self.send_json({"type": "status", "text": "Reconnecting to agent…"})
        elif etype == "error":
            if self.pending:
                self._end_turn()
                text = ("The agent is busy — try again in a moment." if ev.get("retryable")
                        else str(ev.get("text", "") or "Agent error"))
                await self.send_json({"type": "error", "text": text[:300]})
        elif etype == "media":
            await self._on_media(ev)

    async def _send_activity(self, text: str, tool: str) -> None:
        line = activity_text(text)
        if not line or line == self._last_activity:
            return
        self._last_activity = line
        await self.send_json({"type": "activity", "text": line, "tool": tool})

    async def stop_agent(self) -> None:
        """Stop while the agent works: the agent stops after its current step
        (its "ready" ends the turn); what was queued to go out is dropped."""
        ch = self.ch
        if not (self.pending and ch is not None and ch.agent_ws is not None):
            return
        await self.stop_output()
        self._forget_stream()
        try:
            await ch.agent_ws.send(json.dumps({"type": "cancel"}))
        except Exception as exc:
            await self.send_json({"type": "error", "text": f"Couldn't stop: {exc}"[:200]})
            return
        await self.send_json({"type": "status", "text": "Stopping…"})

    # ── a turn across reconnects ──
    def _end_turn(self) -> None:
        self.pending = False
        turn = self.turns().get(self.channel_id)
        if turn is not None:
            turn["open"] = False
            turn["ended"].set()
        if self.closed:
            self.unbind()  # nothing more to keep for the reconnect

    def _kept_turn(self) -> dict | None:
        turn = self.turns().get(self.channel_id)
        if turn and time.monotonic() - max(turn["at"], turn["sent_at"]) <= TURN_KEEP_S:
            return turn
        return None

    def _keep_answer(self, turn: dict | None, text: str, proactive: bool) -> None:
        """A turn's answer is kept until the client is heard from after it
        (a dropped link loses it)."""
        if turn is not None:
            turn.update(answer={"text": text, "proactive": proactive}, via=self.serial,
                        sent_at=time.monotonic(), heard=False)

    def take_over(self) -> None:
        """A client back after a dropped link: the turn still running on its
        channel is this connection's now, so its answer comes here."""
        turn = self._kept_turn()
        if turn and turn["open"]:
            self.pending = True

    async def resume(self) -> None:
        """After take_over: the answer the client may have missed comes
        again, or a turn still running shows as thinking again."""
        turn = self._kept_turn()
        if turn is None:
            return
        if not await self._resume_ready() or self.turns().get(self.channel_id) is not turn:
            return  # they're asking something new
        answer = turn.get("answer")
        if answer and not turn["heard"] and turn["via"] != self.serial:
            await self._show_reply(answer["text"], answer["proactive"], turn)
            if not turn["open"]:
                await self.send_json(self._turn_state(turn, "ready"))
        elif turn["open"] and not answer:
            await self.send_json(self._turn_state(turn, "thinking"))

    def heard(self) -> None:
        """Something came from the client: the answer sent to it a moment
        before got there."""
        self.last_heard_at = time.monotonic()
        turn = self.turns().get(self.channel_id)
        if (turn and turn.get("answer") and turn["via"] == self.serial
                and time.monotonic() - turn["sent_at"] >= HEARD_AFTER_S):
            turn["heard"] = True

    # ── client → agent ──
    async def before_send(self) -> None:
        """Anything a client needs to put right before a turn goes to the agent."""

    async def first_frame(self, ch: Any) -> dict | None:
        """A frame the agent needs ahead of this turn, or None."""
        return None

    async def send_to_agent(self, text: str, ctx: Any = None, extra_notes: list[str] | None = None) -> bool:
        """The words to the agent as this client's turn; False when it didn't
        go (the client has been told why)."""
        await self.before_send()
        ch = self.ch
        if ch is None:
            await self.send_json({"type": "error", "text": "No agent bound."})
            return False
        # The first send after binding waits for the link (a sleeping agent wakes).
        if not await gb.await_agent_link(ch):
            await self.send_json({"type": "error", "text": "Agent isn't reachable."})
            return False
        allowed, quota = await usage_quota.admit_message(self.user_id, text)
        if not allowed:
            await self.send_json({"type": "error", "text": usage_quota.short_limit_text(quota)})
            await self.send_json({"type": "state", "state": "ready"})
            return False
        first = await self.first_frame(ch)
        async with ch.send_lock:
            payload = {"type": "chat", "client_msg_id": self.MSG_ID_PREFIX + secrets.token_hex(6),
                       **self.chat_fields(ch, text, ctx, extra_notes)}
            try:
                if first:
                    await ch.agent_ws.send(json.dumps(first))
                await ch.agent_ws.send(json.dumps(payload))
                ch.context_sent = True
            except Exception as exc:
                await self.send_json({"type": "error", "text": f"Send failed: {exc}"})
                return False
        self.pending = True
        turn = {"at": time.monotonic(), "open": True, "answer": None,
                "via": 0, "sent_at": 0.0, "heard": False, "ended": asyncio.Event()}
        self.turns()[self.channel_id] = turn
        self._turn_started(text, turn)
        self._last_activity = ""
        try:
            await self.send_json({"type": "state", "state": "thinking"})
        except Exception:
            pass  # the client went away: the agent has the turn all the same
        return True

    async def close(self) -> None:
        self.closed = True
        if not self.pending:
            self.unbind()  # (a turn still running keeps listening: its answer waits for the reconnect)
        await self._close_output()
        for task in list(self._bg):
            task.cancel()
