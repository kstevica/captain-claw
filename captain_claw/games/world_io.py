"""Serialize and deserialize a `World` to JSON.

Used by the generator (writes `world.json` after generation) and by the
registry hydration path (reads it on agent restart).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from captain_claw.games.world import Character, Entity, Interaction, Room, World


def world_to_dict(world: World) -> dict[str, Any]:
    return {
        "id": world.id,
        "title": world.title,
        "summary": world.summary,
        "rooms": {
            rid: {
                "id": r.id,
                "name": r.name,
                "description": r.description,
                "exits": dict(r.exits),
                "initial_entities": list(r.initial_entities),
                "ascii_tile": list(r.ascii_tile),
                "locked_exits": dict(r.locked_exits),
            }
            for rid, r in world.rooms.items()
        },
        "entities": {
            eid: {
                "id": e.id,
                "name": e.name,
                "description": e.description,
                "glyph": e.glyph,
                "takeable": e.takeable,
                "talkable": e.talkable,
                "examinable": e.examinable,
                "examine_text": e.examine_text,
            }
            for eid, e in world.entities.items()
        },
        "characters": {
            cid: {
                "id": c.id,
                "name": c.name,
                "description": c.description,
                "glyph": c.glyph,
                "objective": c.objective,
                "start_room": c.start_room,
            }
            for cid, c in world.characters.items()
        },
        "win_condition": dict(world.win_condition or {}),
        "interactions": [
            {
                "item_id": ix.item_id,
                "target_id": ix.target_id,
                "message": ix.message,
                "sets_flag": ix.sets_flag,
                "consumes_item": ix.consumes_item,
                "unlocks_exit": ix.unlocks_exit,
                "reveals_entity": ix.reveals_entity,
            }
            for ix in world.interactions
        ],
    }


def world_from_dict(data: dict[str, Any]) -> World:
    rooms = {
        rid: Room(
            id=r["id"],
            name=r["name"],
            description=r["description"],
            exits=dict(r.get("exits", {})),
            initial_entities=tuple(r.get("initial_entities", [])),
            ascii_tile=tuple(r.get("ascii_tile", [])),
            locked_exits=dict(r.get("locked_exits", {})),
        )
        for rid, r in (data.get("rooms") or {}).items()
    }
    entities = {
        eid: Entity(
            id=e["id"],
            name=e["name"],
            description=e["description"],
            glyph=e.get("glyph", "?"),
            takeable=bool(e.get("takeable", False)),
            talkable=bool(e.get("talkable", False)),
            examinable=bool(e.get("examinable", False)),
            examine_text=str(e.get("examine_text", "")),
        )
        for eid, e in (data.get("entities") or {}).items()
    }
    characters = {
        cid: Character(
            id=c["id"],
            name=c["name"],
            description=c["description"],
            glyph=c.get("glyph", "@"),
            objective=c.get("objective", ""),
            start_room=c["start_room"],
        )
        for cid, c in (data.get("characters") or {}).items()
    }
    interactions = tuple(
        Interaction(
            item_id=str(ix.get("item_id", "")),
            target_id=str(ix.get("target_id", "")),
            message=str(ix.get("message", "")),
            sets_flag=str(ix.get("sets_flag", "")),
            consumes_item=bool(ix.get("consumes_item", False)),
            unlocks_exit=str(ix.get("unlocks_exit", "")),
            reveals_entity=str(ix.get("reveals_entity", "")),
        )
        for ix in (data.get("interactions") or [])
    )
    return World(
        id=data["id"],
        title=data.get("title", data["id"]),
        summary=data.get("summary", ""),
        rooms=rooms,
        entities=entities,
        characters=characters,
        win_condition=dict(data.get("win_condition") or {}),
        interactions=interactions,
    )


def save_world(world: World, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(world_to_dict(world), indent=2), encoding="utf-8")


def load_world(path: Path) -> World:
    data = json.loads(path.read_text(encoding="utf-8"))
    return world_from_dict(data)
