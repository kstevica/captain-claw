"""Tick loop driver.

`run_tick` collects one intent per character (from each seat), orders
them deterministically, hands them to the engine, writes the log entry,
and returns the new state plus the events.

The web layer calls this once per HTTP request to advance one tick.

Since M3 the function is async — AgentSeat.decide() makes an LLM call.
Sync seats (Scripted, Human) still work: run_tick checks for an async
`decide` coroutine before falling back to the sync `submit_intent`.
"""

from __future__ import annotations

import random
from typing import Any

from captain_claw.games.engine import order_intents, resolve
from captain_claw.games.intent import Intent
from captain_claw.games.log import GameLog
from captain_claw.games.seats import SeatTable
from captain_claw.games.world import State


async def run_tick(
    state: State,
    seats: SeatTable,
    log: GameLog,
    rng: random.Random,
) -> tuple[State, list[Intent], list[dict[str, Any]]]:
    """Advance the game by one tick. Returns (new_state, intents, events)."""
    if state.terminal:
        return state, [], []

    raw_intents: list[Intent] = []
    for char_id in seats.all_chars():
        seat = seats.get(char_id)
        # AgentSeat has an async `decide` method; all others use sync `submit_intent`
        if hasattr(seat, "decide"):
            intent = await seat.decide(state, char_id)
        else:
            intent = seat.submit_intent(state, char_id)
        raw_intents.append(intent)

    ordered = order_intents(raw_intents)
    new_state, events = resolve(state, ordered, rng)
    log.append(tick=new_state.tick, intents=ordered, events=events)
    return new_state, ordered, events
