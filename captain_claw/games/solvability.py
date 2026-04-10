"""Solvability checker — confirms a generated world is winnable.

For M1 we use the dumbest possible thing that proves the win condition
is reachable: spin up scripted seats, run the engine forward up to a
budget, and check whether the win flag flips. The ScriptedSeat already
does BFS toward the win room when `kind == "all_in_room"`, so this is
a tight upper bound on a solver.

This is not a proof of solvability for adversarial worlds — it's a
sanity check that says "the world isn't broken / unreachable / missing
the goal room". M1b regenerates the failing pipeline stage when this
returns False.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from captain_claw.games.engine import order_intents, resolve
from captain_claw.games.seats import ScriptedSeat, SeatTable
from captain_claw.games.world import World, initial_state


@dataclass
class SolvabilityReport:
    solvable: bool
    ticks_to_win: int | None
    reason: str


def check_solvable(world: World, max_ticks: int = 60, seed: int = 0) -> SolvabilityReport:
    if not world.characters:
        return SolvabilityReport(False, None, "world has no characters")
    if not world.win_condition:
        return SolvabilityReport(False, None, "world has no win_condition")

    # Validate the win room exists if win condition references one
    cond = world.win_condition or {}
    if cond.get("kind") == "all_in_room":
        target = cond.get("room")
        if target not in world.rooms:
            return SolvabilityReport(False, None, f"win room '{target}' missing from world.rooms")

    # Validate every character starts in a real room
    for c in world.characters.values():
        if c.start_room not in world.rooms:
            return SolvabilityReport(
                False, None, f"character {c.id} starts in unknown room '{c.start_room}'"
            )

    seats = SeatTable()
    for cid in world.characters:
        seats.assign(cid, ScriptedSeat())

    state = initial_state(world)
    rng = random.Random(seed)
    for _ in range(max_ticks):
        if state.terminal:
            break
        intents = []
        for cid in seats.all_chars():
            intents.append(seats.get(cid).submit_intent(state, cid))
        state, _ = resolve(state, order_intents(intents), rng)

    if state.terminal and state.win:
        return SolvabilityReport(True, state.tick, "scripted seats reached win")
    if state.terminal:
        return SolvabilityReport(False, state.tick, "game ended without win")
    return SolvabilityReport(False, None, f"no win within {max_ticks} ticks")
