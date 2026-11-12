"""Real values back on the way to the person — the privacy gate's whole chat.

With the gate covering the whole chat, an agent's model only ever sees tokens
(⟦PERSON_1⟧) and writes them in its narration and replies. Flight Deck puts
the real values back in the frames it relays to the person — their own chat
with their own agent, or a member's chat with a shared agent (the member's
vault) — never for anyone else (an admin looking at someone else's agent, a
public page, a channel other people can reach).

Restoring never fails the person: anything that goes wrong shows the tokens.

What the model wrote and was put back — a reply, a narration line, its
reasoning, a question — goes into each token's history ("back", channel
"chat", the frame's type), with the context it came in, once per text: the
same text again (a final frame repeating a narration line) is not the model
sending it back again. Frames the model didn't write (the prompt echoed in a
trace, a tool's output, the person's own message, a reconnect's replay) are
restored for the person but not noted; the tool calls it wrote are noted
where they are put back (the gate's restore, "tool:<name>").
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections import OrderedDict
from typing import Any

from captain_claw.anonymize import engine, usage
from captain_claw.anonymize import gate as G  # noqa: N812
from captain_claw.logging import get_logger

log = get_logger(__name__)

SETTING_TTL = 15.0
# Frames left as they are: streamed deltas (a token can be split across
# them; the chat doesn't draw them) and the agent's own bookkeeping.
SKIP_TYPES = frozenset({"response_stream", "usage", "turn_usage", "status", "session_info", "welcome", "pong"})

# Frames whose text the model wrote (a chat_message only from the assistant).
MODEL_FRAMES = frozenset({"thinking", "narration", "question"})
SEEN_MAX = 20_000

_STATE: dict[str, tuple[tuple[Any, ...], engine.VaultState]] = {}
_SEEN: OrderedDict[str, None] = OrderedDict()
_TASKS: set[asyncio.Task[Any]] = set()


def may_have_tokens(text: str) -> bool:
    """A token's bracket, as it is or as JSON escapes it (the agent's frames
    are ``json.dumps``-ed with non-ASCII escaped: ``\\u27e6``)."""
    return "⟦" in text or "\\u27e6" in text or "\\u27E6" in text


async def vault_state(person: str) -> engine.VaultState | None:
    """The person's Chat vault, loaded once per change (None when there is none)."""
    from captain_claw.flight_deck import anonymize_vault as av

    store = av.store()
    row = await store.get(person, G.GATE_VAULT)
    if row is None:
        return None
    key = (row["id"], row.get("updated_at"), row.get("entity_count"))
    hit = _STATE.get(person)
    if hit and hit[0] == key:
        return hit[1]
    state = await store.load_state(row)
    _STATE[person] = (key, state)
    return state


def restore_value(value: Any, state: engine.VaultState) -> Any:
    """Every string in *value* (a frame's JSON) with its tokens put back."""
    if isinstance(value, str):
        return engine.restore(value, state)[0] if "⟦" in value else value
    if isinstance(value, list):
        return [restore_value(v, state) for v in value]
    if isinstance(value, dict):
        return {k: restore_value(v, state) for k, v in value.items()}
    return value


async def restore_text(person: str, text: str) -> str:
    """*text* with the person's Chat vault's tokens put back (as it is when
    there's nothing to do or anything fails)."""
    if "⟦" not in text:
        return text
    try:
        state = await vault_state(person)
        return engine.restore(text, state)[0] if state is not None else text
    except Exception as exc:  # noqa: BLE001 — tokens, never a failed message
        log.debug("Chat restore skipped", error=type(exc).__name__)
        return text


def _fresh(person: str, token: str, context: str) -> bool:
    """First time this token came back in this context (for this person)?"""
    key = hashlib.sha256(f"{person}\0{token}\0{context}".encode("utf-8", "surrogatepass")).hexdigest()
    if key in _SEEN:
        _SEEN.move_to_end(key)
        return False
    _SEEN[key] = None
    while len(_SEEN) > SEEN_MAX:
        _SEEN.popitem(last=False)
    return True


async def _record_back(person: str, contexts: dict[str, tuple[int, str]], *, actor: str, label: str) -> None:
    from captain_claw.flight_deck import anonymize_vault as av

    try:
        store = av.store()
        row = await store.get(person, G.GATE_VAULT)
        if row is not None:
            await store.record_uses(row, [av.TokenUse(t, "back", n, c) for t, (n, c) in contexts.items()],
                                    channel="chat", actor=actor, label=label)
    except Exception as exc:  # noqa: BLE001 — the history never holds up the chat
        log.debug("Chat restore: history not recorded", error=type(exc).__name__)


def note_restored(person: str, data: Any, state: engine.VaultState, *, actor: str = "", label: str = "") -> None:
    """Record, in the background, the tokens *data* (a frame, already parsed)
    had put back — each new (token, context) once."""
    found = usage.token_contexts(usage.strings(data), set(state.by_token))
    fresh = {t: v for t, v in found.items() if _fresh(person, t, v[1])}
    if not fresh:
        return
    task = asyncio.ensure_future(_record_back(person, fresh, actor=actor, label=label))
    _TASKS.add(task)
    task.add_done_callback(_TASKS.discard)


def model_wrote(data: Any) -> bool:
    if not isinstance(data, dict):
        return False
    kind = data.get("type")
    if kind == "chat_message":
        return str(data.get("role") or "assistant") == "assistant"
    return kind in MODEL_FRAMES


class ChatRestorer:
    """One per socket: restores the frames an agent sends its person, while
    their gate covers the whole chat. *agent*: the agent's slug, for the
    tokens' history."""

    def __init__(self, person: str, agent: str = "") -> None:
        self.person = person
        self.actor = f"agent:{agent}" if agent else ""
        self._on = False
        self._checked = 0.0

    async def active(self) -> bool:
        now = time.monotonic()
        if now - self._checked > SETTING_TTL:
            from captain_claw.flight_deck import anonymize_gate

            self._checked = now
            try:
                self._on = await anonymize_gate.chat_restores(self.person)
            except Exception as exc:  # noqa: BLE001
                log.debug("Chat restore: setting unavailable", error=type(exc).__name__)
        return self._on

    async def frame(self, text: str) -> str:
        if not may_have_tokens(text) or not self.person or not await self.active():
            return text
        try:
            data = json.loads(text)
        except ValueError:
            return text
        if isinstance(data, dict) and data.get("type") in SKIP_TYPES:
            return text
        try:
            state = await vault_state(self.person)
            if state is None:
                return text
            restored = json.dumps(restore_value(data, state), ensure_ascii=False)
            if model_wrote(data):
                note_restored(self.person, data, state, actor=self.actor, label=str(data.get("type") or ""))
            return restored
        except Exception as exc:  # noqa: BLE001 — tokens, never a broken chat
            log.debug("Chat restore skipped", error=type(exc).__name__)
            return text


def reset_for_tests() -> None:
    _STATE.clear()
    _SEEN.clear()
