"""The pure rules engine.

`resolve(state, intents, rng)` is the only function that mutates game
state. It is a pure function: same input → same output, every time.
This is the foundation of replay and determinism.

Events are emitted as a structured list. Narration (prose) consumes events
later, but the engine itself is text-free.
"""

from __future__ import annotations

import copy
import random
from typing import Any

from captain_claw.games.intent import Intent
from captain_claw.games.world import State


# ── Event helpers ────────────────────────────────────────────────────


def _evt(actor: str, kind: str, **payload: Any) -> dict[str, Any]:
    return {"actor": actor, "kind": kind, **payload}


# ── Verb resolvers ───────────────────────────────────────────────────


def _resolve_wait(state: State, intent: Intent) -> list[dict[str, Any]]:
    return [_evt(intent.actor, "waited")]


def _resolve_look(state: State, intent: Intent) -> list[dict[str, Any]]:
    room_id = state.char_room.get(intent.actor)
    return [_evt(intent.actor, "looked", room=room_id)]


def _resolve_move(state: State, intent: Intent) -> list[dict[str, Any]]:
    actor = intent.actor
    here = state.char_room.get(actor)
    if here is None:
        return [_evt(actor, "error", message="character has no room")]
    room = state.world.rooms[here]

    target_room: str | None = None
    direction = intent.args.get("direction")
    room_id_arg = intent.args.get("room_id")

    if direction:
        direction = str(direction).lower()
        target_room = room.exits.get(direction)
        if target_room is None:
            return [_evt(actor, "error", message=f"no exit '{direction}' from {room.name}")]
        # Check locked exits
        lock_key = f"{here}:{direction}"
        if direction in room.locked_exits and lock_key not in state.unlocked_exits:
            required_flag = room.locked_exits[direction]
            return [_evt(actor, "error",
                         message=f"The exit {direction} is locked. You need to find a way to unlock it.")]
    elif room_id_arg:
        if room_id_arg in room.exits.values():
            target_room = str(room_id_arg)
            # Check locked exits for room_id based moves
            for d, r in room.exits.items():
                if r == room_id_arg:
                    lock_key = f"{here}:{d}"
                    if d in room.locked_exits and lock_key not in state.unlocked_exits:
                        return [_evt(actor, "error",
                                     message=f"The way to {room_id_arg} is locked.")]
                    break
        else:
            return [_evt(actor, "error", message=f"cannot reach {room_id_arg} directly")]
    else:
        return [_evt(actor, "error", message="move requires direction or room_id")]

    state.char_room[actor] = target_room
    if target_room not in state.visited[actor]:
        state.visited[actor].append(target_room)
    return [_evt(actor, "moved", from_room=here, to_room=target_room)]


def _fuzzy_entity(state: State, query: str, candidates: list[str]) -> str | None:
    """Resolve a fuzzy entity name to an exact entity ID.

    Tries: exact match → case-insensitive ID → substring of name.
    Returns the entity_id or None.
    """
    if query in candidates:
        return query
    q = query.lower().replace(" ", "_")
    # Case-insensitive ID match
    for eid in candidates:
        if eid.lower() == q:
            return eid
    # Substring match on entity name
    matches = []
    for eid in candidates:
        ent = state.world.entities.get(eid)
        if ent and q.replace("_", " ") in ent.name.lower():
            matches.append(eid)
    if len(matches) == 1:
        return matches[0]
    return None


def _resolve_take(state: State, intent: Intent) -> list[dict[str, Any]]:
    actor = intent.actor
    raw_id = intent.args.get("entity_id")
    if not raw_id:
        return [_evt(actor, "error", message="take requires entity_id")]
    here = state.char_room[actor]
    contents = state.room_entities.get(here, [])
    entity_id = _fuzzy_entity(state, raw_id, contents)
    if entity_id is None:
        return [_evt(actor, "error", message=f"no '{raw_id}' here")]
    entity = state.world.entities.get(entity_id)
    if entity is None:
        return [_evt(actor, "error", message=f"unknown entity {entity_id}")]
    if not entity.takeable:
        return [_evt(actor, "error", message=f"{entity.name} cannot be taken")]
    contents.remove(entity_id)
    state.char_inventory.setdefault(actor, []).append(entity_id)
    return [_evt(actor, "took", entity_id=entity_id, from_room=here)]


def _resolve_drop(state: State, intent: Intent) -> list[dict[str, Any]]:
    actor = intent.actor
    raw_id = intent.args.get("entity_id")
    if not raw_id:
        return [_evt(actor, "error", message="drop requires entity_id")]
    inv = state.char_inventory.get(actor, [])
    entity_id = _fuzzy_entity(state, raw_id, inv)
    if entity_id is None:
        return [_evt(actor, "error", message=f"you do not have {raw_id}")]
    inv.remove(entity_id)
    here = state.char_room[actor]
    state.room_entities.setdefault(here, []).append(entity_id)
    return [_evt(actor, "dropped", entity_id=entity_id, in_room=here)]


def _resolve_say(state: State, intent: Intent) -> list[dict[str, Any]]:
    actor = intent.actor
    text = str(intent.args.get("text", "")).strip()
    if not text:
        return [_evt(actor, "error", message="say requires text")]
    if len(text) > 280:
        text = text[:280]
    here = state.char_room[actor]
    # Everyone in the same room hears you
    audience = [cid for cid, room in state.char_room.items() if room == here]
    # Magical comms: if the speech mentions another character by name,
    # they hear it regardless of room (walkie-talkie / telepathic link)
    text_lower = text.lower()
    for cid, char in state.world.characters.items():
        if cid not in audience and char.name.lower() in text_lower:
            audience.append(cid)
    record = {"tick": state.tick, "actor": actor, "room": here, "text": text, "audience": audience}
    state.public_say.append(record)
    return [_evt(actor, "said", text=text, room=here, audience=audience)]


def _resolve_talk(state: State, intent: Intent) -> list[dict[str, Any]]:
    """Private direct message — always crosses rooms, only the target hears it."""
    actor = intent.actor
    text = str(intent.args.get("text", "")).strip()
    if not text:
        return [_evt(actor, "error", message="talk requires text")]
    if len(text) > 280:
        text = text[:280]

    raw_target = str(intent.args.get("target", "")).strip()
    if not raw_target:
        return [_evt(actor, "error", message="talk requires a target character")]

    # Resolve target: try exact char ID, then fuzzy name match
    target_id: str | None = None
    if raw_target in state.world.characters:
        target_id = raw_target
    else:
        q = raw_target.lower()
        for cid, char in state.world.characters.items():
            if cid != actor and (q in char.name.lower() or char.name.lower() in q):
                target_id = cid
                break
    if target_id is None or target_id == actor:
        return [_evt(actor, "error", message=f"unknown character '{raw_target}'")]

    here = state.char_room[actor]
    # Private: only the target hears it (plus actor for echo)
    audience = [actor, target_id]
    record = {
        "tick": state.tick, "actor": actor, "room": here,
        "text": text, "audience": audience, "kind": "talk", "target": target_id,
    }
    state.public_say.append(record)
    return [_evt(actor, "talked", text=text, target=target_id, audience=audience)]


def _resolve_use_target(state: State, raw_target: str, item_id: str, here: str, inv: list[str]) -> str | None:
    """Resolve a use-target through multiple strategies.

    1. Fuzzy entity match (room contents + inventory)
    2. Exact room ID
    3. Fuzzy room name match
    4. Direction / exit reference → resolve to room ID (for locked-exit interactions)
    5. Scan interaction table: if item has only one interaction, use that target
    """
    room_contents = state.room_entities.get(here, [])

    # 1) fuzzy entity
    eid = _fuzzy_entity(state, raw_target, room_contents + inv)
    if eid is not None:
        return eid

    # 2) exact room ID
    if raw_target in state.world.rooms:
        return raw_target

    # 3) fuzzy room name — match "the long hall" or "hall" to room id
    q = raw_target.lower().replace("_", " ")
    for rid, room in state.world.rooms.items():
        if rid.lower() == q or q in room.name.lower():
            return rid

    # 4) direction / exit reference — "east door", "east", "eastern door", "locked door"
    room_obj = state.world.rooms.get(here)
    if room_obj:
        target_lower = raw_target.lower()
        # Strip common suffixes: "east door" → "east", "eastern passage" → "east"
        direction_words = target_lower.replace(" door", "").replace(" gate", "").replace(" passage", "").replace(" exit", "").strip()
        # Map "eastern" → "east", "northern" → "north", etc.
        dir_map = {"northern": "north", "southern": "south", "eastern": "east", "western": "west"}
        direction_words = dir_map.get(direction_words, direction_words)

        if direction_words in room_obj.exits:
            # The agent is referring to an exit direction — resolve to current room
            # (interactions that unlock exits target the room containing the locked exit)
            return here

        # "locked door" — if there's exactly one locked exit, target this room
        if "lock" in target_lower or "door" in target_lower:
            locked_dirs = [d for d in room_obj.locked_exits if f"{here}:{d}" not in state.unlocked_exits]
            if len(locked_dirs) >= 1:
                return here

    # 5) scan interaction table — if item has exactly one interaction, auto-resolve
    matching = [ix for ix in state.world.interactions if ix.item_id == item_id]
    if len(matching) == 1:
        return matching[0].target_id

    return None


def _resolve_use(state: State, intent: Intent) -> list[dict[str, Any]]:
    """Use an item on a target. Checks the world's interaction table."""
    actor = intent.actor
    raw_item = intent.args.get("item_id")
    raw_target = intent.args.get("target_id")
    if not raw_item or not raw_target:
        return [_evt(actor, "error", message="use requires item_id and target_id")]

    inv = state.char_inventory.get(actor, [])
    item_id = _fuzzy_entity(state, raw_item, inv)
    if item_id is None:
        return [_evt(actor, "error", message=f"you are not holding '{raw_item}'")]

    here = state.char_room[actor]
    target_id = _resolve_use_target(state, raw_target, item_id, here, inv)
    if target_id is None:
        return [_evt(actor, "error", message=f"no '{raw_target}' here to use that on")]

    # Find matching interaction
    for interaction in state.world.interactions:
        if interaction.item_id == item_id and interaction.target_id == target_id:
            events: list[dict[str, Any]] = []

            # Set flag
            if interaction.sets_flag:
                state.flags[interaction.sets_flag] = True

            # Consume item
            if interaction.consumes_item and item_id in inv:
                inv.remove(item_id)

            # Unlock exit
            if interaction.unlocks_exit:
                state.unlocked_exits.add(interaction.unlocks_exit)

            # Reveal entity
            if interaction.reveals_entity:
                ent = state.world.entities.get(interaction.reveals_entity)
                if ent:
                    state.room_entities.setdefault(here, []).append(interaction.reveals_entity)

            # Store result so the view can surface it (per-tick)
            result = {
                "actor": actor, "kind": "use_result",
                "item_id": item_id, "target_id": target_id,
                "text": interaction.message,
            }
            state.action_results.append(result)
            # Persist in discoveries (cumulative)
            disc = state.discoveries.setdefault(actor, [])
            disc.append({"tick": state.tick, "kind": "use_result",
                          "item_id": item_id, "target_id": target_id,
                          "text": interaction.message})
            events.append(_evt(actor, "used", item_id=item_id, target_id=target_id,
                               message=interaction.message))
            return events

    return [_evt(actor, "error", message=f"Using {item_id} on {target_id} does nothing.")]


def _resolve_examine(state: State, intent: Intent) -> list[dict[str, Any]]:
    """Examine an entity for extra detail."""
    actor = intent.actor
    raw_id = intent.args.get("entity_id")
    if not raw_id:
        return [_evt(actor, "error", message="examine requires entity_id")]

    here = state.char_room[actor]
    room_contents = state.room_entities.get(here, [])
    inv = state.char_inventory.get(actor, [])
    entity_id = _fuzzy_entity(state, raw_id, room_contents + inv)
    if entity_id is None:
        return [_evt(actor, "error", message=f"no '{raw_id}' here to examine")]

    entity = state.world.entities.get(entity_id)
    if entity is None:
        return [_evt(actor, "error", message=f"unknown entity {entity_id}")]

    if not entity.examinable:
        text = entity.description
    else:
        # Track examination
        if actor not in state.examined:
            state.examined[actor] = set()
        state.examined[actor].add(entity_id)
        text = entity.examine_text if entity.examine_text else entity.description

    # Store result so the view can surface it (per-tick)
    result = {
        "actor": actor, "kind": "examine_result",
        "entity_id": entity_id, "entity_name": entity.name, "text": text,
    }
    state.action_results.append(result)
    # Persist in discoveries (cumulative) — avoid duplicates
    disc = state.discoveries.setdefault(actor, [])
    if not any(d.get("entity_id") == entity_id and d.get("kind") == "examine_result" for d in disc):
        disc.append({"tick": state.tick, "kind": "examine_result",
                      "entity_id": entity_id, "entity_name": entity.name, "text": text})
    return [_evt(actor, "examined", entity_id=entity_id, text=text)]


def _resolve_give(state: State, intent: Intent) -> list[dict[str, Any]]:
    """Give an item from inventory to another character in the same room."""
    actor = intent.actor
    raw_entity = intent.args.get("entity_id")
    raw_target = intent.args.get("target_id")
    if not raw_entity or not raw_target:
        return [_evt(actor, "error", message="give requires entity_id and target_id")]

    inv = state.char_inventory.get(actor, [])
    entity_id = _fuzzy_entity(state, raw_entity, inv)
    if entity_id is None:
        return [_evt(actor, "error", message=f"you are not holding '{raw_entity}'")]

    # Find target character in same room
    here = state.char_room[actor]
    target_char = None
    for cid, room_id in state.char_room.items():
        if cid == actor or room_id != here:
            continue
        char = state.world.characters.get(cid)
        if char and (cid == raw_target or char.name.lower() == raw_target.lower()):
            target_char = cid
            break

    if target_char is None:
        return [_evt(actor, "error", message=f"'{raw_target}' is not here to give items to")]

    # Transfer
    inv.remove(entity_id)
    state.char_inventory.setdefault(target_char, []).append(entity_id)
    return [_evt(actor, "gave", entity_id=entity_id, to=target_char, in_room=here)]


_RESOLVERS = {
    "wait": _resolve_wait,
    "look": _resolve_look,
    "move": _resolve_move,
    "take": _resolve_take,
    "drop": _resolve_drop,
    "say": _resolve_say,
    "talk": _resolve_talk,
    "use": _resolve_use,
    "examine": _resolve_examine,
    "give": _resolve_give,
}


# ── Public entry point ───────────────────────────────────────────────


def resolve(
    state: State,
    intents: list[Intent],
    rng: random.Random,
) -> tuple[State, list[dict[str, Any]]]:
    """Apply all intents to `state` and return `(new_state, events)`.

    Pure: deepcopies the input state, never mutates the caller's copy.
    Intents are ordered deterministically by (character_id, original index)
    *before* this function is called — see `tick.order_intents`.
    """
    new_state = State(
        world=state.world,
        tick=state.tick + 1,
        char_room=dict(state.char_room),
        char_inventory={k: list(v) for k, v in state.char_inventory.items()},
        room_entities={k: list(v) for k, v in state.room_entities.items()},
        visited={k: list(v) for k, v in state.visited.items()},
        flags=dict(state.flags),
        public_say=[],  # public_say is per-tick, not cumulative
        terminal=state.terminal,
        win=state.win,
        examined={k: set(v) for k, v in state.examined.items()},
        unlocked_exits=set(state.unlocked_exits),
        action_results=[],  # per-tick, not cumulative
        discoveries={k: list(v) for k, v in state.discoveries.items()},  # cumulative
    )

    events: list[dict[str, Any]] = []
    for intent in intents:
        resolver = _RESOLVERS.get(intent.verb)
        if resolver is None:
            events.append(_evt(intent.actor, "error", message=f"unknown verb {intent.verb}"))
            continue
        try:
            events.extend(resolver(new_state, intent))
        except Exception as exc:  # noqa: BLE001 — defensive: never crash the loop
            events.append(_evt(intent.actor, "error", message=f"resolver crashed: {exc}"))

    # Surface error events as action_results so the view/UI can display them
    for evt in events:
        if evt.get("kind") == "error":
            new_state.action_results.append({
                "actor": evt.get("actor", ""),
                "kind": "error",
                "text": evt.get("message", ""),
            })

    _evaluate_win_condition(new_state, events)
    # rng is reserved for future probabilistic verbs (combat, lockpicking)
    _ = rng
    return new_state, events


def _evaluate_win_condition(state: State, events: list[dict[str, Any]]) -> None:
    cond = state.world.win_condition or {}
    kind = cond.get("kind")
    if kind == "all_in_room":
        room = cond.get("room")
        if room and all(r == room for r in state.char_room.values()):
            state.terminal = True
            state.win = True
            events.append(_evt("__engine__", "game_won", room=room))
    elif kind == "collect_items":
        char_id = cond.get("character")
        required_items = cond.get("items", [])
        if char_id and required_items:
            inv = state.char_inventory.get(char_id, [])
            if all(item in inv for item in required_items):
                state.terminal = True
                state.win = True
                events.append(_evt("__engine__", "game_won",
                                   character=char_id, items=required_items))
    elif kind == "flags_set":
        required_flags = cond.get("flags", [])
        if required_flags and all(state.flags.get(f) for f in required_flags):
            state.terminal = True
            state.win = True
            events.append(_evt("__engine__", "game_won", flags=required_flags))


def order_intents(intents: list[Intent]) -> list[Intent]:
    """Deterministic intent ordering: by (actor_id, submission_order)."""
    indexed = list(enumerate(intents))
    indexed.sort(key=lambda x: (x[1].actor, x[0]))
    return [i for _, i in indexed]


def clone_state(state: State) -> State:
    """Helper for tests / replay sandboxes."""
    return copy.deepcopy(state)
