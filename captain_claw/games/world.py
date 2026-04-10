"""Frozen world definitions and mutable game state.

The `World` is the immutable spec produced by generation (or hardcoded in
M0). The `State` is the mutable per-tick game state. The engine never
mutates `World`; it produces a new `State` every tick.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Entity:
    """A thing in the world: item, prop, NPC body, etc."""
    id: str
    name: str
    description: str
    glyph: str            # single ASCII char shown on the map
    takeable: bool = False
    talkable: bool = False  # NPC: scripted-dialogue (v1) / mini-agent (v2)
    examinable: bool = False  # can be examined for extra detail
    examine_text: str = ""    # text revealed on examine


@dataclass(frozen=True)
class Interaction:
    """A use-item-on-target rule.

    When a character uses `item_id` on `target_id` (entity or room feature),
    the engine sets the specified flag and emits a message. If `consumes_item`
    is True the item is removed from inventory after use.

    Optional `unlocks_exit` opens a locked exit in a room.
    Optional `reveals_entity` makes a hidden entity appear in a room.
    """
    item_id: str           # entity the player must be holding
    target_id: str         # entity (in room or inventory) or room_id to use it on
    message: str           # feedback text shown to the player
    sets_flag: str = ""    # flag key to set True on success
    consumes_item: bool = False
    unlocks_exit: str = ""    # "room_id:direction" — unlocks a locked exit
    reveals_entity: str = ""  # entity_id to spawn in the current room


@dataclass(frozen=True)
class Room:
    """A node in the location graph."""
    id: str
    name: str
    description: str
    exits: dict[str, str]              # direction ("north"/"east"/...) -> room_id
    initial_entities: tuple[str, ...]  # entity ids placed here at game start
    ascii_tile: tuple[str, ...]        # multi-line ASCII art (rows)
    locked_exits: dict[str, str] = field(default_factory=dict)
    # locked_exits: direction -> required_flag to pass (empty = no locked exits)


@dataclass(frozen=True)
class Character:
    """A player-controllable body. Owned by a seat at runtime."""
    id: str
    name: str
    description: str
    glyph: str             # single ASCII char shown on the map
    objective: str         # PRIVATE — only this character's view ever sees it
    start_room: str


@dataclass(frozen=True)
class World:
    id: str
    title: str
    summary: str
    rooms: dict[str, Room]
    entities: dict[str, Entity]
    characters: dict[str, Character]
    win_condition: dict[str, Any] = field(default_factory=dict)
    interactions: tuple[Interaction, ...] = ()
    # win_condition kinds:
    #   {"kind": "all_in_room", "room": "exit"}
    #   {"kind": "collect_items", "character": "char_id", "items": ["id1", "id2"]}
    #   {"kind": "flags_set", "flags": ["flag1", "flag2"]}


@dataclass
class State:
    """Mutable per-tick game state. Serializable via `to_dict`."""
    world: World
    tick: int
    char_room: dict[str, str]              # char_id -> room_id
    char_inventory: dict[str, list[str]]   # char_id -> [entity_id]
    room_entities: dict[str, list[str]]    # room_id -> [entity_id]
    visited: dict[str, list[str]]          # char_id -> visited room_ids (insertion order)
    flags: dict[str, bool] = field(default_factory=dict)
    public_say: list[dict[str, Any]] = field(default_factory=list)  # tick-scoped chat log
    terminal: bool = False
    win: bool = False
    examined: dict[str, set[str]] = field(default_factory=dict)
    # examined: char_id -> set of entity_ids they have examined
    unlocked_exits: set[str] = field(default_factory=set)
    # unlocked_exits: set of "room_id:direction" strings for exits unlocked at runtime
    action_results: list[dict[str, Any]] = field(default_factory=list)
    # action_results: tick-scoped feedback messages (examine text, use messages, etc.)
    discoveries: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    # discoveries: char_id -> [{tick, kind, text, entity_name, ...}] — cumulative knowledge log

    # ── serialization (state only — world is referenced by id elsewhere) ──

    def to_dict(self) -> dict[str, Any]:
        return {
            "tick": self.tick,
            "char_room": dict(self.char_room),
            "char_inventory": {k: list(v) for k, v in self.char_inventory.items()},
            "room_entities": {k: list(v) for k, v in self.room_entities.items()},
            "visited": {k: list(v) for k, v in self.visited.items()},
            "flags": dict(self.flags),
            "public_say": list(self.public_say),
            "terminal": self.terminal,
            "win": self.win,
            "examined": {k: list(v) for k, v in self.examined.items()},
            "unlocked_exits": list(self.unlocked_exits),
            "action_results": list(self.action_results),
            "discoveries": {k: list(v) for k, v in self.discoveries.items()},
        }

    @classmethod
    def from_dict(cls, world: World, data: dict[str, Any]) -> "State":
        return cls(
            world=world,
            tick=int(data["tick"]),
            char_room=dict(data["char_room"]),
            char_inventory={k: list(v) for k, v in data["char_inventory"].items()},
            room_entities={k: list(v) for k, v in data["room_entities"].items()},
            visited={k: list(v) for k, v in data["visited"].items()},
            flags=dict(data.get("flags", {})),
            public_say=list(data.get("public_say", [])),
            terminal=bool(data.get("terminal", False)),
            win=bool(data.get("win", False)),
            examined={k: set(v) for k, v in data.get("examined", {}).items()},
            unlocked_exits=set(data.get("unlocked_exits", [])),
            action_results=list(data.get("action_results", [])),
            discoveries={k: list(v) for k, v in data.get("discoveries", {}).items()},
        )


def initial_state(world: World) -> State:
    """Build the starting State for a freshly-loaded world."""
    char_room = {c.id: c.start_room for c in world.characters.values()}
    room_entities: dict[str, list[str]] = {r.id: list(r.initial_entities) for r in world.rooms.values()}
    char_inventory: dict[str, list[str]] = {c.id: [] for c in world.characters.values()}
    visited: dict[str, list[str]] = {cid: [room] for cid, room in char_room.items()}
    return State(
        world=world,
        tick=0,
        char_room=char_room,
        char_inventory=char_inventory,
        room_entities=room_entities,
        visited=visited,
    )
