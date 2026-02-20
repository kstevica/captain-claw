"""Load and render LLM instruction templates from disk.

Supports a two-layer override system:
  1. Personal overrides in ``~/.captain-claw/instructions/`` (highest priority)
  2. System defaults in ``<project>/instructions/``

When loading a template, the personal directory is checked first.  Edits via
the web UI are always saved to the personal directory, keeping the system
defaults intact.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping


_PERSONAL_DIR = Path("~/.captain-claw/instructions").expanduser()


class _SafeFormatDict(dict[str, str]):
    """Leave unknown placeholders untouched instead of raising KeyError."""

    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


class InstructionLoader:
    """Read and render instruction templates with personal-override support.

    Resolution order for every template:
      1. ``personal_dir / name``  (``~/.captain-claw/instructions/``)
      2. ``base_dir / name``      (system project ``instructions/``)
    """

    def __init__(
        self,
        base_dir: Path | str | None = None,
        personal_dir: Path | str | None = None,
    ):
        self.base_dir = self._resolve_base_dir(base_dir)
        self.personal_dir: Path = (
            Path(personal_dir).expanduser().resolve()
            if personal_dir is not None
            else _PERSONAL_DIR.resolve()
        )
        self._cache: dict[str, str] = {}

    @staticmethod
    def _resolve_base_dir(base_dir: Path | str | None) -> Path:
        if base_dir is not None:
            return Path(base_dir).expanduser().resolve()
        env_dir = os.getenv("CAPTAIN_CLAW_INSTRUCTIONS_DIR")
        if env_dir:
            return Path(env_dir).expanduser().resolve()
        return (Path(__file__).resolve().parent.parent / "instructions").resolve()

    def _path(self, name: str) -> Path:
        """Return the effective file path, preferring the personal override."""
        personal = self.personal_dir / name
        if personal.is_file():
            return personal
        return self.base_dir / name

    def is_overridden(self, name: str) -> bool:
        """Return ``True`` if a personal override exists for *name*."""
        return (self.personal_dir / name).is_file()

    def load(self, name: str) -> str:
        """Load instruction template content by filename."""
        cached = self._cache.get(name)
        if cached is not None:
            return cached

        path = self._path(name)
        if not path.is_file():
            raise FileNotFoundError(
                f"Instruction template not found: {path}. "
                "Add the file under the instructions folder."
            )
        content = path.read_text(encoding="utf-8").strip()
        self._cache[name] = content
        return content

    def render(self, name: str, **variables: object) -> str:
        """Render template with simple ``str.format`` placeholder substitution."""
        template = self.load(name)
        values: Mapping[str, str] = {k: str(v) for k, v in variables.items()}
        return template.format_map(_SafeFormatDict(values))
