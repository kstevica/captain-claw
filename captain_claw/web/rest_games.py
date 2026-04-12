"""REST handlers for Captain Claw Game (CCG).

Endpoints (mounted in `web_server.py`):

  GET    /api/games/worlds                 → list available demo worlds
  GET    /api/games                        → list active game sessions
  POST   /api/games                        → create a new game from a world
  GET    /api/games/{game_id}              → full game state + per-character views
  POST   /api/games/{game_id}/tick         → advance one tick (collects all seats)
  POST   /api/games/{game_id}/intent       → queue a human intent for next tick
  POST   /api/games/{game_id}/replay       → re-run from intent log, return final state
  POST   /api/games/{game_id}/restart      → reset game to tick 0
  DELETE /api/games/{game_id}              → drop the in-memory session

The agent's web server is the only home for game state. Flight Deck talks
to it through the existing per-agent proxy pattern (`/fd/agent-games/...`).
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Any

from aiohttp import web

from captain_claw.games.engine import order_intents, resolve
from captain_claw.games.intent import Intent, validate_shape
from captain_claw.games.registry import GameSession, get_registry
from captain_claw.games.seats import AgentSeat
from captain_claw.games.render import render_view
from captain_claw.games.tick import run_tick
from captain_claw.games.view import project_view
from captain_claw.games.world import initial_state
from captain_claw.logging import get_logger

if TYPE_CHECKING:
    from captain_claw.web_server import WebServer

log = get_logger(__name__)


# ── Serialization helpers ──────────────────────────────────────────


def _session_payload(session: GameSession) -> dict[str, Any]:
    """Full game payload — summary + per-character views + ASCII renders."""
    views: dict[str, Any] = {}
    rendered: dict[str, str] = {}
    for cid in session.world.characters:
        view = project_view(session.state, cid)
        views[cid] = view
        rendered[cid] = render_view(view)
    # Collect agent thought logs
    agent_thoughts: dict[str, list[dict[str, Any]]] = {}
    for cid in session.world.characters:
        try:
            seat = session.seats.get(cid)
        except KeyError:
            continue
        if isinstance(seat, AgentSeat):
            agent_thoughts[cid] = seat.thought_log

    return {
        **session.to_summary(),
        "views": views,
        "rendered": rendered,
        "agent_thoughts": agent_thoughts,
        "conversation_log": session.conversation_log,
    }


# ── Handlers ───────────────────────────────────────────────────────


async def list_worlds(server: "WebServer", request: web.Request) -> web.Response:
    return web.json_response({"worlds": get_registry().available_worlds()})


async def list_cognitive_modes(server: "WebServer", request: web.Request) -> web.Response:
    """Return all available cognitive modes for game character agents."""
    from captain_claw.cognitive_mode import list_modes, mode_to_dict
    modes = [mode_to_dict(m) for m in list_modes()]
    return web.json_response({"modes": modes})


_RANDOM_IDEA_PROMPT = """\
Generate a creative, original text-adventure game idea. Be imaginative and varied — \
avoid generic fantasy quests. Draw from sci-fi, horror, mystery, historical, surreal, \
comedic, noir, post-apocalyptic, underwater, space, micro-scale, time-travel, or any \
other unexpected genre. Surprise me.

Return ONLY a JSON object with these fields:
- "title": a short evocative title (3-6 words)
- "goal": a one-sentence objective for the players (what they must accomplish)
- "description": a 2-3 sentence description of the world, its atmosphere, and backstory

Be specific and vivid. Not "explore a dungeon" but "navigate the ventilation shafts \
of a derelict space station before the reactor melts down". Not "find the treasure" \
but "recover the stolen painting from a 1920s speakeasy before the cops raid at midnight".

JSON only, no markdown fences:"""


async def random_world_idea(server: "WebServer", request: web.Request) -> web.Response:
    """Use the agent's LLM to generate a random world idea."""
    provider = getattr(getattr(server, "agent", None), "provider", None)
    if provider is None:
        return web.json_response({"ok": False, "error": "no LLM provider"}, status=503)

    import json as _json
    from captain_claw.llm import Message

    try:
        # Add a random seed word to push the LLM away from repetitive ideas
        seed_words = [
            "clockwork", "underwater", "miniature", "fungal", "orbital", "frozen",
            "volcanic", "dream", "microscopic", "desert", "arctic", "jungle",
            "subterranean", "floating", "mirror", "ancient", "neon", "abandoned",
            "haunted", "coral", "crystalline", "mechanical", "overgrown", "buried",
            "inverted", "temporal", "acoustic", "magnetic", "bioluminescent", "nomadic",
            "petrified", "hollow", "woven", "fermented", "tidal", "spectral",
        ]
        seed = random.choice(seed_words)
        prompt = f"{_RANDOM_IDEA_PROMPT}\n\nInspiration seed (use loosely, don't force it): {seed}"

        resp = await provider.complete(
            [Message(role="user", content=prompt)],
            temperature=1.0,
            max_tokens=300,
        )
        raw = (resp.content or "").strip()
        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3].strip()

        idea = _json.loads(raw)
        return web.json_response({
            "ok": True,
            "title": str(idea.get("title", "")),
            "goal": str(idea.get("goal", "")),
            "description": str(idea.get("description", "")),
        })
    except Exception as exc:
        log.warning("Random idea generation failed", error=str(exc))
        return web.json_response({"ok": False, "error": str(exc)}, status=500)


async def list_games(server: "WebServer", request: web.Request) -> web.Response:
    return web.json_response({"games": get_registry().list()})


async def create_game(server: "WebServer", request: web.Request) -> web.Response:
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"ok": False, "error": "Invalid JSON"}, status=400)

    world_id = str(body.get("world_id", "")).strip()
    if not world_id:
        return web.json_response({"ok": False, "error": "Missing world_id"}, status=400)

    raw_seats = body.get("seats") or {}
    if not isinstance(raw_seats, dict):
        return web.json_response({"ok": False, "error": "seats must be an object"}, status=400)
    seat_assignments = {str(k): str(v) for k, v in raw_seats.items()}

    seed = body.get("seed")
    if seed is not None:
        try:
            seed = int(seed)
        except (TypeError, ValueError):
            return web.json_response({"ok": False, "error": "seed must be int"}, status=400)

    try:
        session = get_registry().create_from_demo(world_id, seat_assignments, seed)
    except ValueError as exc:
        return web.json_response({"ok": False, "error": str(exc)}, status=404)

    # If agent_targets are provided, reassign seats to wire up remote providers
    raw_targets = body.get("agent_targets") or {}
    if raw_targets:
        agent_targets = {str(k): dict(v) for k, v in raw_targets.items()}
        provider = getattr(getattr(server, "agent", None), "provider", None)
        session.reassign_seats(seat_assignments, provider=provider, agent_targets=agent_targets)

    log.info("Game created", game_id=session.game_id, world_id=world_id, seed=session.seed)
    return web.json_response({"ok": True, "game": _session_payload(session)})


async def get_game(server: "WebServer", request: web.Request) -> web.Response:
    game_id = request.match_info["game_id"]
    session = get_registry().get(game_id)
    if session is None:
        return web.json_response({"ok": False, "error": "game not found"}, status=404)
    return web.json_response({"ok": True, "game": _session_payload(session)})


async def tick_game(server: "WebServer", request: web.Request) -> web.Response:
    game_id = request.match_info["game_id"]
    session = get_registry().get(game_id)
    if session is None:
        return web.json_response({"ok": False, "error": "game not found"}, status=404)
    if session.state.terminal:
        return web.json_response({"ok": False, "error": "game already terminal"}, status=409)

    new_state, intents, events = await run_tick(
        session.state, session.seats, session.log, session.rng,
    )
    session.state = new_state
    session.record_conversations()
    session.persist_thoughts()

    return web.json_response({
        "ok": True,
        "tick": new_state.tick,
        "intents": [i.to_dict() for i in intents],
        "events": events,
        "game": _session_payload(session),
    })


async def submit_intent(server: "WebServer", request: web.Request) -> web.Response:
    game_id = request.match_info["game_id"]
    session = get_registry().get(game_id)
    if session is None:
        return web.json_response({"ok": False, "error": "game not found"}, status=404)

    try:
        body = await request.json()
    except Exception:
        return web.json_response({"ok": False, "error": "Invalid JSON"}, status=400)

    intent, err = validate_shape(body)
    if err or intent is None:
        return web.json_response({"ok": False, "error": err or "invalid intent"}, status=400)

    ok, err = session.queue_human_intent(intent)
    if not ok:
        return web.json_response({"ok": False, "error": err}, status=400)

    return web.json_response({"ok": True, "queued": intent.to_dict()})


import time as _time

from captain_claw.games.game_instructions import load_game_instruction
from captain_claw.games.llm_usage import fire_and_forget_usage


async def submit_natural(server: "WebServer", request: web.Request) -> web.Response:
    """Parse natural language input via LLM and queue as a structured intent."""
    game_id = request.match_info["game_id"]
    session = get_registry().get(game_id)
    if session is None:
        return web.json_response({"ok": False, "error": "game not found"}, status=404)

    try:
        body = await request.json()
    except Exception:
        return web.json_response({"ok": False, "error": "Invalid JSON"}, status=400)

    actor = str(body.get("actor", "")).strip()
    text = str(body.get("text", "")).strip()
    if not actor or not text:
        return web.json_response({"ok": False, "error": "actor and text required"}, status=400)

    if actor not in session.world.characters:
        return web.json_response({"ok": False, "error": f"unknown character {actor}"}, status=400)

    provider = getattr(getattr(server, "agent", None), "provider", None)
    if provider is None:
        return web.json_response({"ok": False, "error": "no LLM provider"}, status=503)

    # Build context from current view
    from captain_claw.games.view import project_view
    view = project_view(session.state, actor)
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

    from captain_claw.llm import Message
    import json as _json
    import re as _re
    messages = [
        Message(role="system", content=load_game_instruction("nl_parser_system.md")),
        Message(role="user", content=f"Game state:\n{context}\n\nPlayer says: {text}"),
    ]
    try:
        t0 = _time.monotonic()
        resp = await provider.complete(messages, temperature=0.2, max_tokens=128)
        raw = resp.content or ""
        fire_and_forget_usage(
            interaction="game_nl_parse",
            messages=messages, response=resp,
            provider=provider, max_tokens=128,
            latency_ms=int((_time.monotonic() - t0) * 1000),
        )
    except Exception as exc:
        log.warning("NL parse failed", error=str(exc))
        return web.json_response({"ok": False, "error": f"LLM parse failed: {exc}"}, status=500)

    # Extract JSON from response
    parsed = None
    try:
        parsed = _json.loads(raw.strip())
    except _json.JSONDecodeError:
        m = _re.search(r"\{[^{}]*\}", raw, _re.DOTALL)
        if m:
            try:
                parsed = _json.loads(m.group(0))
            except _json.JSONDecodeError:
                pass

    if not parsed or "verb" not in parsed:
        # Fallback: treat as say
        parsed = {"verb": "say", "args": {"text": text}}

    verb = str(parsed.get("verb", "say")).lower()
    args = parsed.get("args", {})
    if not isinstance(args, dict):
        args = {}

    from captain_claw.games.intent import Intent, VERBS
    from captain_claw.games.seats import _normalize_args
    if verb not in VERBS:
        parsed = {"verb": "say", "args": {"text": text}}
        verb = "say"
        args = {"text": text}

    args = _normalize_args(verb, args)
    intent = Intent(actor=actor, verb=verb, args={str(k): str(v) for k, v in args.items()})
    ok, err = session.queue_human_intent(intent)
    if not ok:
        return web.json_response({"ok": False, "error": err}, status=400)

    return web.json_response({
        "ok": True,
        "queued": intent.to_dict(),
        "parsed_from": text,
    })


async def replay_game(server: "WebServer", request: web.Request) -> web.Response:
    """Re-run a game deterministically from its intent log.

    This is the proof of M0 determinism: starting from the same seed and
    replaying the same intents must produce byte-identical state.
    """
    game_id = request.match_info["game_id"]
    session = get_registry().get(game_id)
    if session is None:
        return web.json_response({"ok": False, "error": "game not found"}, status=404)

    records = session.log.read_all()
    state = initial_state(session.world)
    rng = random.Random(session.seed)

    replayed_events: list[list[dict[str, Any]]] = []
    for record in records:
        intents = [Intent.from_dict(d) for d in record["intents"]]
        ordered = order_intents(intents)
        state, events = resolve(state, ordered, rng)
        replayed_events.append(events)

    matches = (
        state.to_dict() == session.state.to_dict()
        and state.tick == session.state.tick
    )

    return web.json_response({
        "ok": True,
        "ticks_replayed": len(records),
        "matches_live": matches,
        "final_tick": state.tick,
        "final_state": state.to_dict(),
        "live_state": session.state.to_dict(),
    })


async def restart_game(server: "WebServer", request: web.Request) -> web.Response:
    """Reset a game back to tick 0."""
    game_id = request.match_info["game_id"]
    session = get_registry().get(game_id)
    if session is None:
        return web.json_response({"ok": False, "error": "game not found"}, status=404)
    session.restart()
    log.info("Game restarted", game_id=game_id)
    return web.json_response({"ok": True, "game": _session_payload(session)})


async def reassign_seats(server: "WebServer", request: web.Request) -> web.Response:
    """Change seat assignments for a game (only at tick 0)."""
    game_id = request.match_info["game_id"]
    session = get_registry().get(game_id)
    if session is None:
        return web.json_response({"ok": False, "error": "game not found"}, status=404)

    try:
        body = await request.json()
    except Exception:
        return web.json_response({"ok": False, "error": "Invalid JSON"}, status=400)

    raw_seats = body.get("seats") or {}
    if not isinstance(raw_seats, dict):
        return web.json_response({"ok": False, "error": "seats must be an object"}, status=400)
    seat_assignments = {str(k): str(v) for k, v in raw_seats.items()}

    # Optional per-character agent targets: {char_id: {host, port, auth}}
    raw_targets = body.get("agent_targets") or {}
    agent_targets = {str(k): dict(v) for k, v in raw_targets.items()} if raw_targets else None

    provider = getattr(getattr(server, "agent", None), "provider", None)
    ok, err = session.reassign_seats(seat_assignments, provider=provider, agent_targets=agent_targets)
    if not ok:
        return web.json_response({"ok": False, "error": err}, status=400)

    return web.json_response({"ok": True, "game": _session_payload(session)})


async def generate_game(server: "WebServer", request: web.Request) -> web.Response:
    """Generate a world from a WorldSpec and start a game session."""
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"ok": False, "error": "Invalid JSON"}, status=400)

    spec_data = body.get("spec") or {}
    if not isinstance(spec_data, dict):
        return web.json_response({"ok": False, "error": "spec must be an object"}, status=400)

    from captain_claw.games.spec import WorldSpec
    try:
        spec = WorldSpec.from_dict(spec_data)
    except Exception as exc:  # noqa: BLE001
        return web.json_response({"ok": False, "error": f"bad spec: {exc}"}, status=400)
    err = spec.validate()
    if err:
        return web.json_response({"ok": False, "error": err}, status=400)

    mode = str(body.get("mode", "fast")).strip() or "fast"
    if mode not in {"fast", "pipeline"}:
        return web.json_response({"ok": False, "error": "mode must be fast | pipeline"}, status=400)

    raw_seats = body.get("seats") or {}
    if not isinstance(raw_seats, dict):
        return web.json_response({"ok": False, "error": "seats must be an object"}, status=400)
    seat_assignments = {str(k): str(v) for k, v in raw_seats.items()}

    seed = body.get("seed")
    if seed is not None:
        try:
            seed = int(seed)
        except (TypeError, ValueError):
            return web.json_response({"ok": False, "error": "seed must be int"}, status=400)

    provider = getattr(getattr(server, "agent", None), "provider", None)
    if provider is None:
        return web.json_response({"ok": False, "error": "no LLM provider available"}, status=503)

    try:
        session = await get_registry().create_from_spec(
            provider, spec, mode=mode, seat_assignments=seat_assignments, seed=seed,
        )
    except ValueError as exc:
        return web.json_response({"ok": False, "error": str(exc)}, status=400)
    except Exception as exc:  # noqa: BLE001
        log.exception("Generation failed")
        return web.json_response({"ok": False, "error": f"generation failed: {exc}"}, status=500)

    log.info("Game generated", game_id=session.game_id, mode=mode, seed=session.seed)
    return web.json_response({"ok": True, "game": _session_payload(session)})


async def export_game(server: "WebServer", request: web.Request) -> web.Response:
    """Export a game as a portable JSON bundle for transfer to another agent."""
    game_id = request.match_info["game_id"]
    bundle = get_registry().export_game(game_id)
    if bundle is None:
        return web.json_response({"ok": False, "error": "game not found"}, status=404)
    return web.json_response({"ok": True, "bundle": bundle})


async def import_game(server: "WebServer", request: web.Request) -> web.Response:
    """Import a game from a portable JSON bundle (transferred from another agent)."""
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"ok": False, "error": "Invalid JSON"}, status=400)

    bundle = body.get("bundle")
    if not bundle or not isinstance(bundle, dict):
        return web.json_response({"ok": False, "error": "missing bundle object"}, status=400)

    try:
        session = get_registry().import_game(bundle)
    except ValueError as exc:
        return web.json_response({"ok": False, "error": str(exc)}, status=400)
    except Exception as exc:
        log.exception("Import failed")
        return web.json_response({"ok": False, "error": f"import failed: {exc}"}, status=500)

    return web.json_response({"ok": True, "game": _session_payload(session)})


async def delete_game(server: "WebServer", request: web.Request) -> web.Response:
    game_id = request.match_info["game_id"]
    ok = get_registry().delete(game_id)
    return web.json_response({"ok": ok})
