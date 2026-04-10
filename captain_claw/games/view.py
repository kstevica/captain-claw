"""Per-character view projection — the fog-of-war boundary.

`project_view` is the ONLY function that reads the full game state on
behalf of a player. It must never return a reference to the underlying
mutable structures, and must never include information the character
cannot legitimately know.

Cheat detection (§14.5 of the design doc) compares an intent against the
view this function produced for the same character on the previous tick.
"""

from __future__ import annotations

from typing import Any

from captain_claw.games.world import State


def project_view(state: State, char_id: str) -> dict[str, Any]:
    """Build a serializable view of `state` from `char_id`'s perspective.

    Returns a plain dict (no references to mutable state), safe to send
    over the wire to an agent or human.
    """
    if char_id not in state.world.characters:
        return {"error": f"unknown character {char_id}"}

    character = state.world.characters[char_id]
    here_id = state.char_room.get(char_id)
    here = state.world.rooms.get(here_id) if here_id else None

    # ── current room ────────────────────────────────────────────────
    if here is not None:
        visible_entities = []
        for ent_id in state.room_entities.get(here.id, []):
            ent = state.world.entities.get(ent_id)
            if ent is None:
                continue
            examined_set = state.examined.get(char_id, set())
            visible_entities.append({
                "id": ent.id,
                "name": ent.name,
                "description": ent.description,
                "glyph": ent.glyph,
                "takeable": ent.takeable,
                "talkable": ent.talkable,
                "examinable": ent.examinable,
                "examined": ent.id in examined_set,
            })
        # other characters in the same room (visible to each other)
        others_here = []
        for cid, room_id in state.char_room.items():
            if cid == char_id or room_id != here.id:
                continue
            other = state.world.characters[cid]
            others_here.append({
                "id": other.id,
                "name": other.name,
                "glyph": other.glyph,
            })
        # Compute locked exits visible to this character
        locked_exits_info: dict[str, bool] = {}
        for d in here.exits:
            lock_key = f"{here.id}:{d}"
            if d in here.locked_exits:
                locked_exits_info[d] = lock_key not in state.unlocked_exits  # True = still locked
        current_room = {
            "id": here.id,
            "name": here.name,
            "description": here.description,
            "exits": dict(here.exits),
            "ascii_tile": list(here.ascii_tile),
            "entities": visible_entities,
            "others_here": others_here,
            "locked_exits": locked_exits_info,
        }
    else:
        current_room = None

    # ── inventory & character sheet ────────────────────────────────
    inventory = []
    for ent_id in state.char_inventory.get(char_id, []):
        ent = state.world.entities.get(ent_id)
        if ent is None:
            continue
        inventory.append({
            "id": ent.id,
            "name": ent.name,
            "description": ent.description,
            "glyph": ent.glyph,
        })

    # ── visited rooms (last-known names only, not live contents) ────
    visited = []
    for room_id in state.visited.get(char_id, []):
        room = state.world.rooms.get(room_id)
        if room is None:
            continue
        visited.append({"id": room.id, "name": room.name})

    # ── public say / talk events from THIS tick that this character heard ──
    heard = []
    for record in state.public_say:
        if record["actor"] == char_id:
            continue  # don't echo your own speech back to you
        if char_id in record.get("audience", []):
            speaker = state.world.characters.get(record["actor"])
            entry: dict[str, Any] = {
                "actor": record["actor"],
                "actor_name": speaker.name if speaker else record["actor"],
                "text": record["text"],
                "kind": record.get("kind", "say"),  # "say" or "talk"
            }
            heard.append(entry)

    # ── action results from THIS tick for this character ──
    action_results = []
    for result in state.action_results:
        if result.get("actor") == char_id:
            action_results.append({
                "kind": result.get("kind", ""),
                "text": result.get("text", ""),
                "entity_id": result.get("entity_id", ""),
                "entity_name": result.get("entity_name", ""),
            })

    # ── cumulative discoveries for this character ──
    discoveries = list(state.discoveries.get(char_id, []))

    return {
        "tick": state.tick,
        "terminal": state.terminal,
        "win": state.win,
        "character": {
            "id": character.id,
            "name": character.name,
            "description": character.description,
            "glyph": character.glyph,
            "objective": character.objective,  # PRIVATE — only this view sees it
        },
        "current_room": current_room,
        "inventory": inventory,
        "visited": visited,
        "heard": heard,
        "action_results": action_results,
        "discoveries": discoveries,
    }


def collect_visible_ids(view: dict[str, Any]) -> set[str]:
    """Return the set of entity/character/room ids referenced in `view`.

    Used by cheat detection: an intent that names something not in this
    set may be acting on knowledge the character could not have.
    """
    ids: set[str] = set()
    if view.get("character"):
        ids.add(view["character"]["id"])
    cr = view.get("current_room")
    if cr:
        ids.add(cr["id"])
        for d, r in cr.get("exits", {}).items():
            ids.add(r)
            ids.add(d)
        for e in cr.get("entities", []):
            ids.add(e["id"])
        for o in cr.get("others_here", []):
            ids.add(o["id"])
    for e in view.get("inventory", []):
        ids.add(e["id"])
    for r in view.get("visited", []):
        ids.add(r["id"])
    return ids
