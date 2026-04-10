"""Load game instruction templates from the instructions/games/ directory.

Provides the same caching/loading pattern as the agent's Instructions
class, but standalone — game code doesn't require an agent instance.
"""

from __future__ import annotations

from pathlib import Path
from functools import lru_cache

_GAMES_DIR = (Path(__file__).resolve().parent.parent / "instructions" / "games").resolve()


@lru_cache(maxsize=32)
def load_game_instruction(name: str) -> str:
    """Load a game instruction template by filename.

    Files live in ``captain_claw/instructions/games/<name>``.
    Results are cached for the process lifetime.
    """
    path = _GAMES_DIR / name
    if not path.is_file():
        raise FileNotFoundError(
            f"Game instruction template not found: {path}. "
            "Add the file under captain_claw/instructions/games/."
        )
    return path.read_text(encoding="utf-8").strip()


def render_game_instruction(name: str, **variables: object) -> str:
    """Load and render a game instruction template with ``str.format``."""
    template = load_game_instruction(name)
    return template.format(**{k: str(v) for k, v in variables.items()})
