"""Captain Claw Game — multiplayer text-adventure framework.

See `docs/captain-claw-game-design.md` for the design rationale.

This package is structured around a hard determinism boundary:
- `engine.resolve(state, intents, rng)` is a pure function.
- `view.project_view(state, char_id)` is the only place that touches the
  full world state — it returns a fog-of-war view per character.
- `seats.*` decides *who* (agent | human | scripted) supplies an intent
  for a given character on a given tick.
- LLM-generated content is never called from inside the engine; in M0 the
  whole thing runs without any LLM calls at all.
"""

from captain_claw.games.engine import resolve
from captain_claw.games.intent import Intent, VERBS
from captain_claw.games.registry import GameRegistry, get_registry
from captain_claw.games.view import project_view
from captain_claw.games.world import Character, Entity, Room, State, World

__all__ = [
    "Character",
    "Entity",
    "GameRegistry",
    "Intent",
    "Room",
    "State",
    "VERBS",
    "World",
    "get_registry",
    "project_view",
    "resolve",
]
