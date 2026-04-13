"""Flight Deck game routes — hosts CCG directly instead of proxying to agents.

Games run inside the Flight Deck process. Agent LLM providers are accessed
via ``RemoteLLMProvider`` over ``POST /api/llm/complete`` on each agent.
"""

from __future__ import annotations

import asyncio
import json
import random
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from captain_claw.games.registry import GameSession, get_registry
from captain_claw.games.remote_provider import RemoteLLMProvider
from captain_claw.games.seats import AgentSeat
from captain_claw.games.tick import run_tick
from captain_claw.games.view import project_view
from captain_claw.games.render import render_view
from captain_claw.games.engine import order_intents, resolve
from captain_claw.games.intent import Intent, validate_shape, VERBS
from captain_claw.games.world import initial_state
from captain_claw.logging import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/fd/games", tags=["games"])


# ── Models ──────────────────────────────────────────────────────────


class AgentTarget(BaseModel):
    host: str = "localhost"
    port: int
    auth: str = ""
    name: str = ""


class CreateGameRequest(BaseModel):
    world_id: str
    seats: dict[str, str] = Field(default_factory=dict)
    agent_targets: dict[str, AgentTarget] | None = None
    seed: int | None = None


class GenerateGameRequest(BaseModel):
    spec: dict[str, Any]
    mode: str = "fast"
    seats: dict[str, str] = Field(default_factory=dict)
    agent_targets: dict[str, AgentTarget] | None = None
    seed: int | None = None
    provider: AgentTarget  # which agent to use for world generation


class ReassignSeatsRequest(BaseModel):
    seats: dict[str, str]
    agent_targets: dict[str, AgentTarget] | None = None


class IntentRequest(BaseModel):
    actor: str
    verb: str
    args: dict[str, str] = Field(default_factory=dict)


class NaturalRequest(BaseModel):
    actor: str
    text: str
    provider: AgentTarget | None = None  # which agent to use for NL parsing


class RandomIdeaRequest(BaseModel):
    provider: AgentTarget  # which agent to use for idea generation


class ImportGameRequest(BaseModel):
    bundle: dict[str, Any]


# ── Helpers ─────────────────────────────────────────────────────────


def _make_provider(target: AgentTarget) -> RemoteLLMProvider:
    return RemoteLLMProvider(host=target.host, port=target.port, auth=target.auth)


def _session_payload(session: GameSession) -> dict[str, Any]:
    """Full game payload — summary + per-character views + ASCII renders."""
    from captain_claw.games.image_service import has_image

    views: dict[str, Any] = {}
    rendered: dict[str, str] = {}
    for cid in session.world.characters:
        view = project_view(session.state, cid)
        views[cid] = view
        rendered[cid] = render_view(view)
    agent_thoughts: dict[str, list[dict[str, Any]]] = {}
    for cid in session.world.characters:
        try:
            seat = session.seats.get(cid)
        except KeyError:
            continue
        if isinstance(seat, AgentSeat):
            agent_thoughts[cid] = seat.thought_log

    game_images: dict[str, bool] = {
        "banner": has_image(session.dir, "banner"),
    }
    for rid in session.world.rooms:
        game_images[f"room:{rid}"] = has_image(session.dir, "room", rid)
    for cid in session.world.characters:
        game_images[f"char:{cid}"] = has_image(session.dir, "char", cid)
    for eid in session.world.entities:
        game_images[f"entity:{eid}"] = has_image(session.dir, "entity", eid)

    return {
        **session.to_summary(),
        "views": views,
        "rendered": rendered,
        "agent_thoughts": agent_thoughts,
        "conversation_log": session.conversation_log,
        "game_images": game_images,
    }


def _get_session(game_id: str) -> GameSession:
    session = get_registry().get(game_id)
    if session is None:
        raise HTTPException(404, "game not found")
    return session


def _background_generate_images(session: GameSession) -> None:
    """Fire-and-forget background task to generate all game images."""
    from captain_claw.games.image_service import generate_all_images, get_image_provider

    provider = get_image_provider()
    if provider is None:
        return

    async def _run():
        try:
            await generate_all_images(
                game_dir=session.dir,
                world=session.world,
                seed=session.seed,
                provider=provider,
            )
        except Exception as exc:
            log.warning("Background image generation failed", game_id=session.game_id, error=str(exc))

    asyncio.create_task(_run())


def _build_agent_targets_dict(
    targets: dict[str, AgentTarget] | None,
) -> dict[str, dict[str, Any]] | None:
    if not targets:
        return None
    return {
        cid: {"host": t.host, "port": t.port, "auth": t.auth, "name": t.name}
        for cid, t in targets.items()
    }


# ── Image provider routes ────────────────────────────────────────


@router.get("/image-providers")
async def get_image_providers():
    """List available image generation providers."""
    from captain_claw.games.image_service import list_providers, get_provider_id
    return {"providers": list_providers(), "active": get_provider_id()}


@router.post("/image-providers/{provider_id}")
async def set_image_provider_route(provider_id: str):
    """Switch the active image generation provider."""
    from captain_claw.games.image_service import switch_provider, get_provider_id
    if provider_id == "none":
        switch_provider("none")
        return {"ok": True, "active": "none", "label": "None"}
    provider = switch_provider(provider_id)
    if provider is None:
        raise HTTPException(400, f"Provider '{provider_id}' not available")
    return {"ok": True, "active": get_provider_id(), "label": provider.label}


# ── Routes ──────────────────────────────────────────────────────────


@router.get("/worlds")
async def list_worlds():
    return {"worlds": get_registry().available_worlds()}


@router.get("/cognitive-modes")
async def list_cognitive_modes():
    from captain_claw.cognitive_mode import list_modes, mode_to_dict
    return {"modes": [mode_to_dict(m) for m in list_modes()]}


@router.get("")
async def list_games():
    return {"games": get_registry().list()}


@router.post("")
async def create_game(body: CreateGameRequest):
    try:
        session = get_registry().create_from_demo(
            body.world_id, dict(body.seats), body.seed,
        )
    except ValueError as exc:
        raise HTTPException(404, str(exc))

    # If agent_targets provided, reassign to wire up remote providers
    at = _build_agent_targets_dict(body.agent_targets)
    if at:
        session.reassign_seats(dict(body.seats), provider=None, agent_targets=at)

    log.info("Game created", game_id=session.game_id, world_id=body.world_id)
    _background_generate_images(session)
    return {"ok": True, "game": _session_payload(session)}


@router.post("/generate")
async def generate_game(body: GenerateGameRequest):
    from captain_claw.games.spec import WorldSpec

    try:
        spec = WorldSpec.from_dict(body.spec)
    except Exception as exc:
        raise HTTPException(400, f"bad spec: {exc}")
    err = spec.validate()
    if err:
        raise HTTPException(400, err)
    if body.mode not in ("fast", "pipeline"):
        raise HTTPException(400, "mode must be fast | pipeline")

    provider = _make_provider(body.provider)
    try:
        session = await get_registry().create_from_spec(
            provider, spec,
            mode=body.mode,
            seat_assignments=dict(body.seats),
            seed=body.seed,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except Exception as exc:
        log.exception("Generation failed")
        raise HTTPException(500, f"generation failed: {exc}")

    # Wire up remote agent targets for seats
    at = _build_agent_targets_dict(body.agent_targets)
    if at:
        session.reassign_seats(dict(body.seats), provider=None, agent_targets=at)

    log.info("Game generated", game_id=session.game_id, mode=body.mode)
    _background_generate_images(session)
    return {"ok": True, "game": _session_payload(session)}


@router.post("/random-idea")
async def random_world_idea(body: RandomIdeaRequest):
    provider = _make_provider(body.provider)

    from captain_claw.llm import Message

    seed_words = [
        "clockwork", "underwater", "miniature", "fungal", "orbital", "frozen",
        "volcanic", "dream", "microscopic", "desert", "arctic", "jungle",
        "subterranean", "floating", "mirror", "ancient", "neon", "abandoned",
        "haunted", "coral", "crystalline", "mechanical", "overgrown", "buried",
        "inverted", "temporal", "acoustic", "magnetic", "bioluminescent", "nomadic",
        "petrified", "hollow", "woven", "fermented", "tidal", "spectral",
    ]
    seed = random.choice(seed_words)
    prompt = (
        "Generate a creative, original text-adventure game idea. Be imaginative and varied — "
        "avoid generic fantasy quests. Draw from sci-fi, horror, mystery, historical, surreal, "
        "comedic, noir, post-apocalyptic, underwater, space, micro-scale, time-travel, or any "
        "other unexpected genre. Surprise me.\n\n"
        "Return ONLY a JSON object with these fields:\n"
        '- "title": a short evocative title (3-6 words)\n'
        '- "goal": a one-sentence objective for the players\n'
        '- "description": a 2-3 sentence description of the world\n\n'
        "JSON only, no markdown fences:\n\n"
        f"Inspiration seed (use loosely): {seed}"
    )

    try:
        resp = await provider.complete(
            [Message(role="user", content=prompt)],
            temperature=1.0,
            max_tokens=300,
        )
        raw = (resp.content or "").strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3].strip()
        idea = json.loads(raw)
        return {
            "ok": True,
            "title": str(idea.get("title", "")),
            "goal": str(idea.get("goal", "")),
            "description": str(idea.get("description", "")),
        }
    except Exception as exc:
        log.warning("Random idea generation failed", error=str(exc))
        raise HTTPException(500, str(exc))


@router.post("/import")
async def import_game(body: ImportGameRequest):
    try:
        session = get_registry().import_game(body.bundle)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except Exception as exc:
        log.exception("Import failed")
        raise HTTPException(500, f"import failed: {exc}")
    _background_generate_images(session)
    return {"ok": True, "game": _session_payload(session)}


@router.get("/{game_id}")
async def get_game(game_id: str):
    session = _get_session(game_id)
    session.ensure_providers()
    return {"ok": True, "game": _session_payload(session)}


@router.post("/{game_id}/seats")
async def reassign_seats(game_id: str, body: ReassignSeatsRequest):
    session = _get_session(game_id)
    at = _build_agent_targets_dict(body.agent_targets)
    ok, err = session.reassign_seats(dict(body.seats), provider=None, agent_targets=at)
    if not ok:
        raise HTTPException(400, err)
    return {"ok": True, "game": _session_payload(session)}


@router.post("/{game_id}/tick")
async def tick_game(game_id: str):
    session = _get_session(game_id)
    if session.state.terminal:
        raise HTTPException(409, "game already terminal")

    # Ensure agent seat providers are wired from persisted config
    session.ensure_providers()

    new_state, intents, events = await run_tick(
        session.state, session.seats, session.log, session.rng,
    )
    session.state = new_state
    session.record_conversations()
    session.persist_thoughts()

    return {
        "ok": True,
        "tick": new_state.tick,
        "intents": [i.to_dict() for i in intents],
        "events": events,
        "game": _session_payload(session),
    }


@router.post("/{game_id}/intent")
async def submit_intent(game_id: str, body: IntentRequest):
    session = _get_session(game_id)
    intent, err = validate_shape({"actor": body.actor, "verb": body.verb, "args": body.args})
    if err or intent is None:
        raise HTTPException(400, err or "invalid intent")
    ok, err = session.queue_human_intent(intent)
    if not ok:
        raise HTTPException(400, err)
    return {"ok": True, "queued": intent.to_dict()}


@router.post("/{game_id}/natural")
async def submit_natural(game_id: str, body: NaturalRequest):
    session = _get_session(game_id)

    if body.actor not in session.world.characters:
        raise HTTPException(400, f"unknown character {body.actor}")

    # Need a provider for NL parsing — use the one specified, or find one from agent seats
    provider = None
    if body.provider:
        provider = _make_provider(body.provider)
    else:
        # Try to find a provider from any agent seat
        for cid in session.seats.all_chars():
            seat = session.seats.get(cid)
            if isinstance(seat, AgentSeat) and seat.provider is not None:
                provider = seat.provider
                break
    if provider is None:
        raise HTTPException(503, "no LLM provider available — specify a provider in the request")

    from captain_claw.games.game_instructions import load_game_instruction
    from captain_claw.llm import Message
    import re as _re

    view = project_view(session.state, body.actor)
    cr = view.get("current_room", {})
    inv = view.get("inventory", [])
    context_lines = [f"Character: {view['character']['name']}"]
    if cr:
        context_lines.append(f"Room: {cr.get('name', '?')}")
        if cr.get("entities"):
            items = ", ".join(f"{e['name']} (id: {e['id']})" for e in cr["entities"])
            context_lines.append(f"Items here: {items}")
        if cr.get("others_here"):
            others = ", ".join(o["name"] for o in cr["others_here"])
            context_lines.append(f"Others here: {others}")
        if cr.get("exits"):
            exits = ", ".join(f"{d} → {r}" for d, r in sorted(cr["exits"].items()))
            context_lines.append(f"Exits: {exits}")
    if inv:
        inv_str = ", ".join(f"{e['name']} (id: {e['id']})" for e in inv)
        context_lines.append(f"Inventory: {inv_str}")
    context = "\n".join(context_lines)

    messages = [
        Message(role="system", content=load_game_instruction("nl_parser_system.md")),
        Message(role="user", content=f"Game state:\n{context}\n\nPlayer says: {body.text}"),
    ]
    try:
        resp = await provider.complete(messages, temperature=0.2, max_tokens=128)
        raw = resp.content or ""
    except Exception as exc:
        log.warning("NL parse failed", error=str(exc))
        raise HTTPException(500, f"LLM parse failed: {exc}")

    parsed = None
    try:
        parsed = json.loads(raw.strip())
    except json.JSONDecodeError:
        m = _re.search(r"\{[^{}]*\}", raw, _re.DOTALL)
        if m:
            try:
                parsed = json.loads(m.group(0))
            except json.JSONDecodeError:
                pass

    if not parsed or "verb" not in parsed:
        parsed = {"verb": "say", "args": {"text": body.text}}

    verb = str(parsed.get("verb", "say")).lower()
    args = parsed.get("args", {})
    if not isinstance(args, dict):
        args = {}

    from captain_claw.games.seats import _normalize_args
    if verb not in VERBS:
        verb = "say"
        args = {"text": body.text}

    args = _normalize_args(verb, args)
    intent = Intent(actor=body.actor, verb=verb, args={str(k): str(v) for k, v in args.items()})
    ok, err = session.queue_human_intent(intent)
    if not ok:
        raise HTTPException(400, err)

    return {"ok": True, "queued": intent.to_dict(), "parsed_from": body.text}


@router.post("/{game_id}/export")
async def export_game(game_id: str):
    bundle = get_registry().export_game(game_id)
    if bundle is None:
        raise HTTPException(404, "game not found")
    return {"ok": True, "bundle": bundle}


@router.post("/{game_id}/replay")
async def replay_game(game_id: str):
    session = _get_session(game_id)
    records = session.log.read_all()
    state = initial_state(session.world)
    rng = random.Random(session.seed)

    for record in records:
        intents = [Intent.from_dict(d) for d in record["intents"]]
        ordered = order_intents(intents)
        state, _ = resolve(state, ordered, rng)

    matches = state.to_dict() == session.state.to_dict() and state.tick == session.state.tick
    return {
        "ok": True,
        "ticks_replayed": len(records),
        "matches_live": matches,
        "final_tick": state.tick,
    }


@router.post("/{game_id}/restart")
async def restart_game(game_id: str):
    session = _get_session(game_id)
    session.restart()
    log.info("Game restarted", game_id=game_id)
    return {"ok": True, "game": _session_payload(session)}


@router.delete("/{game_id}")
async def delete_game(game_id: str):
    ok = get_registry().delete(game_id)
    return {"ok": ok}


# ── Game Images ───────────────────────────────────────────────────


@router.get("/{game_id}/images/{kind}/{obj_id}")
async def get_game_image(game_id: str, kind: str, obj_id: str):
    """Return a cached image (room, char, entity), or 404."""
    from captain_claw.games.image_service import has_image, image_path

    session = _get_session(game_id)
    if not has_image(session.dir, kind, obj_id):
        raise HTTPException(404, "image not generated yet")

    return FileResponse(image_path(session.dir, kind, obj_id), media_type="image/png")


@router.get("/{game_id}/images/banner")
async def get_banner_image(game_id: str):
    """Return the world banner image."""
    from captain_claw.games.image_service import has_image, image_path

    session = _get_session(game_id)
    if not has_image(session.dir, "banner"):
        raise HTTPException(404, "banner not generated yet")

    return FileResponse(image_path(session.dir, "banner", ""), media_type="image/png")


@router.post("/{game_id}/images/{kind}/{obj_id}")
async def generate_single_image(game_id: str, kind: str, obj_id: str):
    """Generate (or return cached) a single image by kind."""
    from captain_claw.games.image_service import (
        has_image, get_image_provider,
        generate_room_image, generate_character_image, generate_entity_image,
    )

    session = _get_session(game_id)
    provider = get_image_provider()
    if provider is None:
        raise HTTPException(503, "no image provider configured")

    key = f"{kind}:{obj_id}"
    if has_image(session.dir, kind, obj_id):
        return {"ok": True, "cached": True, "key": key}

    path = None
    w = session.world
    if kind == "room":
        room = w.rooms.get(obj_id)
        if room is None:
            raise HTTPException(404, "unknown room")
        path = await generate_room_image(
            session.dir, obj_id, room.name, room.description,
            w.title, w.summary, seed=session.seed, provider=provider,
        )
    elif kind == "char":
        char = w.characters.get(obj_id)
        if char is None:
            raise HTTPException(404, "unknown character")
        path = await generate_character_image(
            session.dir, obj_id, char.name, char.description,
            w.title, w.summary, seed=session.seed, provider=provider,
        )
    elif kind == "entity":
        entity = w.entities.get(obj_id)
        if entity is None:
            raise HTTPException(404, "unknown entity")
        path = await generate_entity_image(
            session.dir, obj_id, entity.name, entity.description,
            w.title, w.summary, seed=session.seed, provider=provider,
        )
    else:
        raise HTTPException(400, f"unknown image kind: {kind}")

    if path is None:
        raise HTTPException(500, "image generation failed")
    return {"ok": True, "cached": False, "key": key}


@router.post("/{game_id}/generate-images")
async def generate_all_images_route(game_id: str):
    """Batch-generate all images (banner, characters, entities, rooms)."""
    from captain_claw.games.image_service import (
        generate_all_images as _gen_all, get_image_provider,
    )

    session = _get_session(game_id)
    provider = get_image_provider()
    if provider is None:
        raise HTTPException(503, "no image provider configured")

    results = await _gen_all(
        game_dir=session.dir, world=session.world,
        seed=session.seed, provider=provider,
    )
    return {"ok": True, "images": results}


@router.get("/{game_id}/image-status")
async def image_status(game_id: str):
    """Return which images have been generated."""
    from captain_claw.games.image_service import has_image

    session = _get_session(game_id)
    status: dict[str, bool] = {"banner": has_image(session.dir, "banner")}
    for rid in session.world.rooms:
        status[f"room:{rid}"] = has_image(session.dir, "room", rid)
    for cid in session.world.characters:
        status[f"char:{cid}"] = has_image(session.dir, "char", cid)
    for eid in session.world.entities:
        status[f"entity:{eid}"] = has_image(session.dir, "entity", eid)
    return {"ok": True, "images": status}
