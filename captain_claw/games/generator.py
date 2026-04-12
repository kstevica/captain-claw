"""WorldSpec → World generator (M1a fast + M1b pipeline).

Two modes:

  - **fast** (M1a): one LLM call returns the entire world as a JSON
    blob. Fast and gets us to "describe a world, play it" in one round
    trip. Lower quality and no solvability rescue.

  - **pipeline** (M1b): the 8-stage pipeline from §5 of the design doc.
    Each stage is a separate constrained LLM call. After all stages run,
    `solvability.check_solvable` runs; if it fails, the failing stage is
    regenerated with feedback up to `max_repairs` times.

Both modes record every LLM call to a `generation.log` JSONL file in
the game directory. Replay never re-calls the LLM — it reads from disk.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from captain_claw.games.solvability import SolvabilityReport, check_solvable
from captain_claw.games.spec import BATCH_ENTITIES, BATCH_ROOMS, SIZE_HINTS, WorldSpec
from captain_claw.games.world import Character, Entity, Interaction, Room, World
from captain_claw.games.world_io import save_world
from captain_claw.llm import LLMProvider, Message
from captain_claw.logging import get_logger

log = get_logger(__name__)


# ── Generation transcript ──────────────────────────────────────────


@dataclass
class GenerationTranscript:
    """Append-only audit trail of every LLM call during generation."""
    entries: list[dict[str, Any]] = field(default_factory=list)

    def record(self, stage: str, prompt: str, response: str, parsed: Any, ok: bool) -> None:
        self.entries.append({
            "ts": time.time(),
            "stage": stage,
            "prompt": prompt,
            "response": response,
            "parsed": parsed,
            "ok": ok,
        })

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fp:
            for entry in self.entries:
                fp.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ── JSON extraction ────────────────────────────────────────────────


_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.+?)\s*```", re.DOTALL)


def _extract_json(raw: str) -> dict[str, Any] | None:
    """Pull a JSON object out of a model response.

    Tries: bare JSON, ```json fenced```, then the first `{...}` block.
    """
    text = (raw or "").strip()
    if not text:
        return None
    # 1) bare JSON
    if text.startswith("{"):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
    # 2) fenced
    m = _JSON_FENCE_RE.search(text)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    # 3) first {...}
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass
    return None


# ── LLM helper ─────────────────────────────────────────────────────


async def _call(
    provider: LLMProvider,
    system: str,
    user: str,
    *,
    temperature: float = 0.7,
    max_tokens: int = 4096,
) -> str:
    import time as _time
    from captain_claw.games.llm_usage import fire_and_forget_usage

    messages = [
        Message(role="system", content=system),
        Message(role="user", content=user),
    ]
    t0 = _time.monotonic()
    resp = await provider.complete(messages, temperature=temperature, max_tokens=max_tokens)
    fire_and_forget_usage(
        interaction="game_generate",
        messages=messages, response=resp,
        provider=provider, max_tokens=max_tokens,
        latency_ms=int((_time.monotonic() - t0) * 1000),
    )
    return resp.content or ""


# ── World assembly from a parsed JSON dict ──────────────────────────


def _slug(name: str, prefix: str = "") -> str:
    s = re.sub(r"[^a-z0-9_]+", "_", name.lower()).strip("_")
    if not s:
        s = uuid.uuid4().hex[:6]
    return f"{prefix}{s}" if prefix else s


def _build_world_from_blob(spec: WorldSpec, blob: dict[str, Any]) -> World:
    """Best-effort coercion of an LLM JSON blob into a `World` object.

    Tolerant: missing fields get filled in with defaults so a slightly
    sloppy LLM response still produces a runnable game. The solvability
    check is the real correctness gate.
    """
    world_id = _slug(spec.title) or uuid.uuid4().hex[:8]

    # ── rooms ──
    rooms_in = blob.get("rooms") or []
    if isinstance(rooms_in, dict):
        rooms_in = list(rooms_in.values())
    rooms: dict[str, Room] = {}
    for r in rooms_in:
        rid = str(r.get("id") or _slug(r.get("name", "room")))
        rooms[rid] = Room(
            id=rid,
            name=str(r.get("name", rid.title())),
            description=str(r.get("description", "")),
            exits={str(k).lower(): str(v) for k, v in (r.get("exits") or {}).items()},
            initial_entities=tuple(str(e) for e in (r.get("initial_entities") or r.get("entities") or [])),
            ascii_tile=tuple(
                str(line) for line in (r.get("ascii_tile") or r.get("ascii") or _default_tile())
            ),
            locked_exits={str(k).lower(): str(v) for k, v in (r.get("locked_exits") or {}).items()},
        )

    # ── entities ──
    entities_in = blob.get("entities") or []
    if isinstance(entities_in, dict):
        entities_in = list(entities_in.values())
    entities: dict[str, Entity] = {}
    for e in entities_in:
        eid = str(e.get("id") or _slug(e.get("name", "thing")))
        glyph = str(e.get("glyph", "?"))[:1] or "?"
        entities[eid] = Entity(
            id=eid,
            name=str(e.get("name", eid)),
            description=str(e.get("description", "")),
            glyph=glyph,
            takeable=bool(e.get("takeable", True)),
            talkable=bool(e.get("talkable", False)),
            examinable=bool(e.get("examinable", False)),
            examine_text=str(e.get("examine_text", "")),
        )

    # ── characters ──
    chars_in = blob.get("characters") or []
    if isinstance(chars_in, dict):
        chars_in = list(chars_in.values())
    characters: dict[str, Character] = {}
    fallback_room = next(iter(rooms), None)
    for c in chars_in:
        cid = str(c.get("id") or _slug(c.get("name", "char"), prefix="char_"))
        glyph = str(c.get("glyph", "@"))[:1] or "@"
        start = str(c.get("start_room", fallback_room or ""))
        if start not in rooms and fallback_room:
            start = fallback_room
        characters[cid] = Character(
            id=cid,
            name=str(c.get("name", cid)),
            description=str(c.get("description", "")),
            glyph=glyph,
            objective=str(c.get("objective", spec.goal)),
            start_room=start,
        )

    # ── interactions ──
    interactions_in = blob.get("interactions") or []
    if isinstance(interactions_in, dict):
        interactions_in = list(interactions_in.values())
    interactions: list[Interaction] = []
    for ix in interactions_in:
        try:
            interactions.append(Interaction(
                item_id=str(ix.get("item_id", "")),
                target_id=str(ix.get("target_id", "")),
                message=str(ix.get("message", "")),
                sets_flag=str(ix.get("sets_flag", "")),
                consumes_item=bool(ix.get("consumes_item", False)),
                unlocks_exit=str(ix.get("unlocks_exit", "")),
                reveals_entity=str(ix.get("reveals_entity", "")),
            ))
        except Exception:
            continue

    # ── win condition ──
    win = blob.get("win_condition") or {}
    if not isinstance(win, dict):
        win = {}
    if not win:
        # Default: get everyone to a room called something exit-y, else last room
        target = None
        for rid, r in rooms.items():
            if any(k in r.name.lower() for k in ("exit", "outside", "escape", "shore", "freedom")):
                target = rid
                break
        if target is None and rooms:
            target = list(rooms.keys())[-1]
        if target:
            win = {"kind": "all_in_room", "room": target}

    return World(
        id=world_id,
        title=spec.title,
        summary=spec.summary or blob.get("summary", ""),
        rooms=rooms,
        entities=entities,
        characters=characters,
        win_condition=win,
        interactions=tuple(interactions),
    )


def _default_tile() -> tuple[str, ...]:
    return (
        "+--------+",
        "|        |",
        "|        |",
        "|        |",
        "+--------+",
    )


# ── Mode 1: fast / single-call (M1a) ───────────────────────────────


from captain_claw.games.game_instructions import load_game_instruction


def _fast_user_prompt(spec: WorldSpec) -> str:
    hints = SIZE_HINTS[spec.size]
    return f"""\
Generate a world matching this spec. Return ONLY the JSON object.

title: {spec.title}
goal: {spec.goal}
genre: {spec.genre}
tone: {spec.tone}
size: {spec.size}  (target ~{hints['rooms']} rooms, ~{hints['entities']} entities)
seats: {spec.seats}  (generate exactly this many characters)
seat_mode: {spec.seat_mode}
constraints: {spec.constraints or '(none)'}
summary hint: {spec.summary or '(none)'}
"""


async def generate_fast(provider: LLMProvider, spec: WorldSpec) -> tuple[World, GenerationTranscript]:
    transcript = GenerationTranscript()
    user = _fast_user_prompt(spec)
    raw = await _call(provider, load_game_instruction("generator_fast_system.md"), user, temperature=0.8, max_tokens=4096)
    blob = _extract_json(raw)
    transcript.record("fast", user, raw, blob, blob is not None)
    if blob is None:
        raise ValueError("generator: model did not return parseable JSON")
    world = _build_world_from_blob(spec, blob)
    return world, transcript


# ── Mode 2: 8-stage pipeline (M1b) ─────────────────────────────────



# Stage prompts. Each one builds on the previous stages' parsed output.
def _stage_premise(spec: WorldSpec) -> str:
    return f"""\
STAGE 1 — PREMISE.
Spec:
  title: {spec.title}
  goal: {spec.goal}
  genre: {spec.genre}
  tone: {spec.tone}
  seats: {spec.seats}
  constraints: {spec.constraints or '(none)'}

Return JSON: {{
  "premise": "2-3 sentence setup",
  "secret_truth": "the underlying fact or twist that the players must discover or overcome",
  "act1": "what happens at the start",
  "act2": "the complication",
  "act3": "the resolution path"
}}
"""


def _stage_rooms(spec: WorldSpec, premise: dict[str, Any], *, count: int | None = None, existing_rooms: list[dict[str, Any]] | None = None, batch_label: str = "") -> str:
    hints = SIZE_HINTS[spec.size]
    target = count or hints['rooms']
    context = ""
    if existing_rooms:
        context = f"""
You already generated these rooms (connect new rooms to them via exits):
{json.dumps([{{'id': r['id'], 'name': r['name'], 'exits': r.get('exits', {{}})}} for r in existing_rooms])}

DO NOT re-generate these rooms. Only generate NEW rooms that connect to them.
"""
    return f"""\
STAGE 2 — ROOMS{batch_label}.
Premise: {json.dumps(premise)}
Generate exactly {target} NEW rooms for this world. Every room reachable from every other room.
{context}
Return JSON: {{
  "rooms": [
    {{
      "id": "snake_case",
      "name": "Title Case",
      "description": "1-2 sentences fitting the tone",
      "exits": {{"north": "other_room_id"}},
      "ascii_tile": ["+--------+","|        |","|        |","|        |","+--------+"]
    }}
  ]
}}

Rules:
- Exits MUST be bidirectional and reference real room ids.
- ascii_tile is 5 lines of exactly 10 chars.
- One room should feel like the natural goal location (an exit, a destination, etc.).
"""


def _stage_cast(spec: WorldSpec, premise: dict[str, Any], rooms: dict[str, Any]) -> str:
    return f"""\
STAGE 3 — CAST.
Premise: {json.dumps(premise)}
Rooms (ids only): {[r['id'] for r in rooms['rooms']]}

Generate exactly {spec.seats} player characters. Each gets a different start_room and a private objective hinting at the secret_truth.

Return JSON: {{
  "characters": [
    {{
      "id": "char_xxx",
      "name": "Name",
      "description": "1 sentence",
      "glyph": "A",
      "objective": "private goal — different per character",
      "start_room": "room_id"
    }}
  ]
}}
"""


def _stage_inventory(spec: WorldSpec, rooms: dict[str, Any], *, count: int | None = None, existing_entities: list[dict[str, Any]] | None = None, batch_label: str = "") -> str:
    hints = SIZE_HINTS[spec.size]
    target = count or hints['entities']
    room_ids = [r['id'] for r in rooms['rooms']]
    context = ""
    if existing_entities:
        context = f"""
You already generated these items (DO NOT re-generate them):
{json.dumps([{{'id': e['id'], 'name': e['name'], 'room': e.get('room', '')}} for e in existing_entities])}

Generate only NEW items with unique ids.
"""
    return f"""\
STAGE 4 — INVENTORY{batch_label}.
Rooms: {room_ids}
Generate exactly {target} NEW items distributed across the rooms. Items can be evidence, tools, decoration, or props.
{context}
Return JSON: {{
  "entities": [
    {{
      "id": "snake_case",
      "name": "...",
      "description": "1 sentence",
      "glyph": "k",
      "takeable": true,
      "room": "room_id"
    }}
  ]
}}
"""


def _stage_clues(premise: dict[str, Any], inventory: dict[str, Any]) -> str:
    return f"""\
STAGE 5 — CLUE WEB.
Premise secret_truth: {premise.get('secret_truth')}
Items: {[e['id'] for e in inventory['entities']]}

Pick a subset of items that, when examined or held together, point to the secret_truth.
This stage has no schema enforcement at the engine level yet — output is informational
and recorded in the transcript.

Return JSON: {{
  "clues": [
    {{ "item_id": "...", "fact": "what this item reveals" }}
  ]
}}
"""


def _stage_sheets(spec: WorldSpec, characters: dict[str, Any], premise: dict[str, Any]) -> str:
    return f"""\
STAGE 6 — CHARACTER SHEETS.
Characters: {characters['characters']}
Secret truth: {premise.get('secret_truth')}

Refine each character with starting knowledge that is asymmetric (each knows something the others don't).

Return JSON: {{
  "characters": [
    {{
      "id": "char_xxx",
      "objective": "private goal — refined",
      "starting_knowledge": "what this character already knows"
    }}
  ]
}}
"""


def _stage_ascii(rooms: dict[str, Any]) -> str:
    return f"""\
STAGE 7 — ASCII TILES.
You may refine each room's ASCII tile if it can be more evocative. 5 lines of 10 chars each, exactly.
Current rooms: {[{'id': r['id'], 'name': r['name'], 'ascii_tile': r['ascii_tile']} for r in rooms['rooms']]}

Return JSON: {{
  "rooms": [
    {{ "id": "...", "ascii_tile": ["+--------+", "...", "...", "...", "+--------+"] }}
  ]
}}
"""


def _stage_winconditon(spec: WorldSpec, rooms: dict[str, Any]) -> str:
    return f"""\
STAGE 8 — WIN CONDITION.
Goal: {spec.goal}
Rooms: {[r['id'] for r in rooms['rooms']]}

Pick the win room. For now only "all_in_room" is supported by the engine.

Return JSON: {{ "win_condition": {{ "kind": "all_in_room", "room": "room_id" }} }}
"""


def _assemble_pipeline(
    spec: WorldSpec,
    premise: dict[str, Any],
    rooms: dict[str, Any],
    characters: dict[str, Any],
    inventory: dict[str, Any],
    win: dict[str, Any],
) -> dict[str, Any]:
    """Merge pipeline stage outputs into a single blob `_build_world_from_blob` can consume."""
    # Attach inventory to rooms via initial_entities
    items_by_room: dict[str, list[str]] = {}
    for e in inventory.get("entities", []):
        items_by_room.setdefault(e.get("room", ""), []).append(e["id"])
    for r in rooms.get("rooms", []):
        r["initial_entities"] = items_by_room.get(r["id"], [])

    return {
        "summary": premise.get("premise", ""),
        "rooms": rooms.get("rooms", []),
        "entities": [
            {k: v for k, v in e.items() if k != "room"}
            for e in inventory.get("entities", [])
        ],
        "characters": characters.get("characters", []),
        "win_condition": win.get("win_condition", {}),
    }


async def _stage_call(
    provider: LLMProvider,
    transcript: GenerationTranscript,
    stage: str,
    user: str,
    *,
    max_retries: int = 1,
    max_tokens: int = 4096,
) -> dict[str, Any]:
    system = load_game_instruction("generator_pipeline_system.md")
    raw = await _call(provider, system, user, temperature=0.7, max_tokens=max_tokens)
    blob = _extract_json(raw)
    transcript.record(stage, user, raw, blob, blob is not None)
    if blob is not None:
        return blob

    # Retry: feed the broken response back with a repair prompt
    for attempt in range(max_retries):
        log.warning(
            "Pipeline stage returned unparseable JSON, retrying",
            stage=stage, attempt=attempt + 1,
            raw_preview=raw[:300] if raw else "(empty)",
        )
        repair_user = (
            f"{user}\n\n"
            f"YOUR PREVIOUS RESPONSE WAS NOT VALID JSON. Here is what you returned:\n"
            f"---\n{raw[:1000]}\n---\n\n"
            f"Please return ONLY a valid JSON object. No prose, no markdown fences, no commentary."
        )
        raw = await _call(provider, system, repair_user, temperature=0.3, max_tokens=max_tokens)
        blob = _extract_json(raw)
        transcript.record(f"{stage}_repair_{attempt + 1}", repair_user, raw, blob, blob is not None)
        if blob is not None:
            return blob

    raise ValueError(f"pipeline stage '{stage}' returned no parseable JSON after {max_retries + 1} attempts")


async def _batched_rooms(
    provider: LLMProvider,
    transcript: GenerationTranscript,
    spec: WorldSpec,
    premise: dict[str, Any],
) -> dict[str, Any]:
    """Generate rooms, batching into multiple calls if the target exceeds BATCH_ROOMS."""
    hints = SIZE_HINTS[spec.size]
    total = hints["rooms"]
    if total <= BATCH_ROOMS:
        return await _stage_call(provider, transcript, "rooms", _stage_rooms(spec, premise), max_tokens=8192)

    all_rooms: list[dict[str, Any]] = []
    remaining = total
    batch_num = 0
    while remaining > 0:
        batch_num += 1
        count = min(remaining, BATCH_ROOMS)
        label = f" (batch {batch_num}, {count} rooms)"
        prompt = _stage_rooms(
            spec, premise,
            count=count,
            existing_rooms=all_rooms if all_rooms else None,
            batch_label=label,
        )
        result = await _stage_call(
            provider, transcript, f"rooms_batch_{batch_num}", prompt, max_tokens=8192,
        )
        new_rooms = result.get("rooms", [])
        all_rooms.extend(new_rooms)
        remaining -= len(new_rooms)
        log.info("Room batch complete", batch=batch_num, generated=len(new_rooms), total=len(all_rooms), target=total)
        if len(new_rooms) == 0:
            log.warning("Room batch returned 0 rooms, stopping", batch=batch_num)
            break
    return {"rooms": all_rooms}


async def _batched_inventory(
    provider: LLMProvider,
    transcript: GenerationTranscript,
    spec: WorldSpec,
    rooms: dict[str, Any],
) -> dict[str, Any]:
    """Generate entities, batching into multiple calls if the target exceeds BATCH_ENTITIES."""
    hints = SIZE_HINTS[spec.size]
    total = hints["entities"]
    if total <= BATCH_ENTITIES:
        return await _stage_call(provider, transcript, "inventory", _stage_inventory(spec, rooms), max_tokens=8192)

    all_entities: list[dict[str, Any]] = []
    remaining = total
    batch_num = 0
    while remaining > 0:
        batch_num += 1
        count = min(remaining, BATCH_ENTITIES)
        label = f" (batch {batch_num}, {count} items)"
        prompt = _stage_inventory(
            spec, rooms,
            count=count,
            existing_entities=all_entities if all_entities else None,
            batch_label=label,
        )
        result = await _stage_call(
            provider, transcript, f"inventory_batch_{batch_num}", prompt, max_tokens=8192,
        )
        new_entities = result.get("entities", [])
        all_entities.extend(new_entities)
        remaining -= len(new_entities)
        log.info("Entity batch complete", batch=batch_num, generated=len(new_entities), total=len(all_entities), target=total)
        if len(new_entities) == 0:
            log.warning("Entity batch returned 0 entities, stopping", batch=batch_num)
            break
    return {"entities": all_entities}


async def generate_pipeline(
    provider: LLMProvider,
    spec: WorldSpec,
    *,
    max_repairs: int = 2,
) -> tuple[World, GenerationTranscript]:
    transcript = GenerationTranscript()

    premise = await _stage_call(provider, transcript, "premise", _stage_premise(spec))
    rooms = await _batched_rooms(provider, transcript, spec, premise)
    characters = await _stage_call(
        provider, transcript, "cast", _stage_cast(spec, premise, rooms)
    )
    inventory = await _batched_inventory(provider, transcript, spec, rooms)
    # Clue web is recorded in the transcript but not consumed by the engine yet (M2 territory)
    try:
        await _stage_call(provider, transcript, "clues", _stage_clues(premise, inventory))
    except Exception as exc:  # noqa: BLE001
        log.warning("Clue stage failed (non-fatal)", error=str(exc))
    try:
        await _stage_call(provider, transcript, "sheets", _stage_sheets(spec, characters, premise))
    except Exception as exc:  # noqa: BLE001
        log.warning("Sheets stage failed (non-fatal)", error=str(exc))
    # ASCII refinement — skip for very large worlds (too many rooms)
    if SIZE_HINTS[spec.size]["rooms"] <= BATCH_ROOMS:
        try:
            ascii_refined = await _stage_call(provider, transcript, "ascii", _stage_ascii(rooms))
            refined_by_id = {r["id"]: r.get("ascii_tile") for r in ascii_refined.get("rooms", [])}
            for r in rooms["rooms"]:
                if r["id"] in refined_by_id and refined_by_id[r["id"]]:
                    r["ascii_tile"] = refined_by_id[r["id"]]
        except Exception as exc:  # noqa: BLE001
            log.warning("ASCII stage failed (non-fatal)", error=str(exc))
    else:
        log.info("Skipping ASCII refinement for large world", rooms=len(rooms.get("rooms", [])))

    win = await _stage_call(provider, transcript, "win", _stage_winconditon(spec, rooms))
    blob = _assemble_pipeline(spec, premise, rooms, characters, inventory, win)
    world = _build_world_from_blob(spec, blob)

    # ── Solvability check + repair loop ──
    report = check_solvable(world)
    repairs = 0
    while not report.solvable and repairs < max_repairs:
        repairs += 1
        log.info("Solvability failed, repairing", reason=report.reason, attempt=repairs)
        # Re-roll the win condition stage with feedback. Cheaper than rebuilding the whole world,
        # and the most common failure mode is "win room id not present" or "unreachable goal".
        repair_user = (
            _stage_winconditon(spec, rooms)
            + f"\n\nPREVIOUS ATTEMPT FAILED: {report.reason}. Pick a different room id from the list."
        )
        try:
            win = await _stage_call(provider, transcript, f"win_repair_{repairs}", repair_user)
        except Exception as exc:  # noqa: BLE001
            log.warning("Repair stage failed", error=str(exc))
            break
        blob = _assemble_pipeline(spec, premise, rooms, characters, inventory, win)
        world = _build_world_from_blob(spec, blob)
        report = check_solvable(world)

    transcript.record(
        "solvability",
        prompt="",
        response="",
        parsed={"solvable": report.solvable, "ticks_to_win": report.ticks_to_win, "reason": report.reason},
        ok=report.solvable,
    )
    if not report.solvable:
        raise ValueError(f"world unsolvable after {repairs} repairs: {report.reason}")
    return world, transcript


# ── Public entry point ──────────────────────────────────────────────


async def generate_world(
    provider: LLMProvider,
    spec: WorldSpec,
    *,
    mode: str = "fast",
) -> tuple[World, GenerationTranscript, SolvabilityReport]:
    """Generate a `World` from a `WorldSpec`. Returns (world, transcript, report).

    `mode`:
        "fast"     → single LLM call (M1a)
        "pipeline" → 8-stage with solvability repair (M1b)
    """
    err = spec.validate()
    if err:
        raise ValueError(err)
    # Force pipeline mode for large worlds — fast mode can't fit them in one call
    if spec.size in ("xl", "huge", "epic") and mode == "fast":
        log.info("Forcing pipeline mode for large world", size=spec.size)
        mode = "pipeline"
    if mode == "fast":
        world, transcript = await generate_fast(provider, spec)
    elif mode == "pipeline":
        world, transcript = await generate_pipeline(provider, spec)
    else:
        raise ValueError(f"unknown generator mode '{mode}'")
    report = check_solvable(world)
    return world, transcript, report


def persist_generated_game(
    game_dir: Path,
    spec: WorldSpec,
    world: World,
    transcript: GenerationTranscript,
    seed: int,
    seat_kinds: dict[str, str],
) -> None:
    game_dir.mkdir(parents=True, exist_ok=True)
    save_world(world, game_dir / "world.json")
    (game_dir / "spec.json").write_text(json.dumps(spec.to_dict(), indent=2), encoding="utf-8")
    transcript.write(game_dir / "generation.log")
    (game_dir / "meta.json").write_text(
        json.dumps(
            {"seed": seed, "world_id": world.id, "seat_kinds": seat_kinds, "kind": "generated"},
            indent=2,
        ),
        encoding="utf-8",
    )
