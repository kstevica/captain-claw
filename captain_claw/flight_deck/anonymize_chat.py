"""Real values back on the way to the person — the privacy gate's whole chat.

With the gate covering the whole chat, an agent's model only ever sees tokens
(⟦PERSON_1⟧) and writes them in its narration and replies. Flight Deck puts
the real values back in the frames it relays to the person — their own chat
with their own agent, or a member's chat with a shared agent (the member's
vault) — never for anyone else (an admin looking at someone else's agent, a
public page, a channel other people can reach).

Restoring never fails the person: anything that goes wrong shows the tokens.
"""

from __future__ import annotations

import json
import time
from typing import Any

from captain_claw.anonymize import engine
from captain_claw.anonymize import gate as G  # noqa: N812
from captain_claw.logging import get_logger

log = get_logger(__name__)

SETTING_TTL = 15.0
# Frames left as they are: streamed deltas (a token can be split across
# them; the chat doesn't draw them) and the agent's own bookkeeping.
SKIP_TYPES = frozenset({"response_stream", "usage", "turn_usage", "status", "session_info", "welcome", "pong"})

_STATE: dict[str, tuple[tuple[Any, ...], engine.VaultState]] = {}


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


class ChatRestorer:
    """One per socket: restores the frames an agent sends its person, while
    their gate covers the whole chat."""

    def __init__(self, person: str) -> None:
        self.person = person
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
            return json.dumps(restore_value(data, state), ensure_ascii=False)
        except Exception as exc:  # noqa: BLE001 — tokens, never a broken chat
            log.debug("Chat restore skipped", error=type(exc).__name__)
            return text


def reset_for_tests() -> None:
    _STATE.clear()
