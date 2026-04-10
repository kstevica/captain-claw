"""Hardcoded demo worlds for M0/M2.

These exist solely to prove the engine, view projection, renderer, log,
and replay loop are correct end-to-end. M1 replaces this with the
LLM-driven generator.
"""

from __future__ import annotations

from captain_claw.games.world import Character, Entity, Interaction, Room, World


def lighthouse_demo() -> World:
    """A 4-room demo with puzzles: foyer → hall → (locked) lantern_room → exit.

    Two players must find a brass key, unlock the eastern door, examine
    a mysterious journal for clues, and reach the cliffside to win.
    """
    rooms = {
        "foyer": Room(
            id="foyer",
            name="The Foyer",
            description="A dim entryway. Salt-bleached floorboards, a stuck door to the north.",
            exits={"north": "hall"},
            initial_entities=("rusty_lantern",),
            ascii_tile=(
                "+--------+",
                "|        |",
                "|   ()   |",
                "|        |",
                "+---  ---+",
            ),
        ),
        "hall": Room(
            id="hall",
            name="The Long Hall",
            description=(
                "A narrow hall that smells of old kelp. A heavy oak door "
                "stands east, secured with a brass lock. Stairs lead south."
            ),
            exits={"south": "foyer", "east": "lantern_room"},
            initial_entities=("brass_key", "old_journal"),
            ascii_tile=(
                "+--------+",
                "|        |",
                "|        |",
                "|  k  j  |",
                "+---  ---+",
            ),
            locked_exits={"east": "door_unlocked"},
        ),
        "lantern_room": Room(
            id="lantern_room",
            name="The Lantern Room",
            description=(
                "The old lighthouse lantern sits cold and dark. A door to "
                "the north opens onto the cliff path. The hall is west."
            ),
            exits={"west": "hall", "north": "exit"},
            initial_entities=("dusty_mirror",),
            ascii_tile=(
                "+--------+",
                "|  ****  |",
                "|  *  *  |",
                "|  ****  |",
                "+---  ---+",
            ),
        ),
        "exit": Room(
            id="exit",
            name="The Cliffside",
            description="Cold open air. Waves boom below. The lighthouse stands at your back.",
            exits={"south": "lantern_room"},
            initial_entities=(),
            ascii_tile=(
                "+--------+",
                "| ~~~~~~ |",
                "|~~~~~~~ |",
                "|        |",
                "+--------+",
            ),
        ),
    }

    entities = {
        "rusty_lantern": Entity(
            id="rusty_lantern",
            name="rusty lantern",
            description="A storm lantern with half a candle stub. Surprisingly heavy.",
            glyph="(",
            takeable=True,
            examinable=True,
            examine_text=(
                "Scratched into the base you find the letters 'J.H. 1887'. "
                "The glass is cracked but intact."
            ),
        ),
        "brass_key": Entity(
            id="brass_key",
            name="brass key",
            description="A small brass key, green with verdigris.",
            glyph="k",
            takeable=True,
        ),
        "old_journal": Entity(
            id="old_journal",
            name="old journal",
            description="A water-stained journal, its pages curled with damp.",
            glyph="j",
            takeable=True,
            examinable=True,
            examine_text=(
                "Most pages are illegible, but one entry reads: "
                "'The brass key opens the eastern passage. Beyond lies the lantern room, "
                "and from there, the cliff path to freedom.'"
            ),
        ),
        "dusty_mirror": Entity(
            id="dusty_mirror",
            name="dusty mirror",
            description="A large wall mirror, coated in years of grime.",
            glyph="m",
            takeable=False,
            examinable=True,
            examine_text=(
                "You wipe the dust away and see your reflection — and behind you, "
                "scratched into the wall: 'The way out is always through.'"
            ),
        ),
    }

    characters = {
        "char_ada": Character(
            id="char_ada",
            name="Ada",
            description="A surveyor with a keen eye for old masonry.",
            glyph="A",
            objective=(
                "Find the brass key, unlock the eastern door in the hall, "
                "and reach the cliffside. Examine anything interesting along the way."
            ),
            start_room="foyer",
        ),
        "char_ben": Character(
            id="char_ben",
            name="Ben",
            description="A coastguard cadet, nervous in dim places.",
            glyph="B",
            objective=(
                "Stay close to Ada and reach the cliffside together. "
                "Pick up the lantern for light — it might reveal something useful."
            ),
            start_room="foyer",
        ),
    }

    interactions = (
        Interaction(
            item_id="brass_key",
            target_id="hall",
            message=(
                "You insert the brass key into the lock on the eastern door. "
                "It turns with a satisfying click — the door swings open!"
            ),
            sets_flag="door_unlocked",
            consumes_item=True,
            unlocks_exit="hall:east",
        ),
    )

    return World(
        id="lighthouse_demo",
        title="The Lighthouse Keeper",
        summary=(
            "A two-player escape from a derelict lighthouse. Find the key, "
            "unlock the passage, and reach the cliffside."
        ),
        rooms=rooms,
        entities=entities,
        characters=characters,
        win_condition={"kind": "all_in_room", "room": "exit"},
        interactions=interactions,
    )


# Registry of available demo worlds keyed by world id.
DEMO_WORLDS = {
    "lighthouse_demo": lighthouse_demo,
}
