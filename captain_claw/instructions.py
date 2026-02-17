"""Load and render LLM instruction templates from disk."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping


class _SafeFormatDict(dict[str, str]):
    """Leave unknown placeholders untouched instead of raising KeyError."""

    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


class InstructionLoader:
    """Read and render instruction templates from the project instructions folder."""

    def __init__(self, base_dir: Path | str | None = None):
        self.base_dir = self._resolve_base_dir(base_dir)
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
        return self.base_dir / name

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
