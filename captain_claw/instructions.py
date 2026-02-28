"""Load and render LLM instruction templates from disk.

Supports a two-layer override system:
  1. Personal overrides in ``~/.captain-claw/instructions/`` (highest priority)
  2. System defaults in ``<project>/instructions/``

When loading a template, the personal directory is checked first.  Edits via
the web UI are always saved to the personal directory, keeping the system
defaults intact.

When ``use_micro`` is enabled (via ``context.micro_instructions`` config),
the loader transparently resolves ``micro_<name>`` first, falling back to
the standard template when no micro variant exists.
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

    Resolution order for every template (when ``use_micro=True``):
      1. ``personal_dir / micro_<name>``
      2. ``base_dir / micro_<name>``
      3. ``personal_dir / <name>``  (fallback to standard)
      4. ``base_dir / <name>``

    When ``use_micro=False`` the standard two-layer resolution applies.
    """

    def __init__(
        self,
        base_dir: Path | str | None = None,
        personal_dir: Path | str | None = None,
        *,
        use_micro: bool | None = None,
    ):
        self.base_dir = self._resolve_base_dir(base_dir)
        self.personal_dir: Path = (
            Path(personal_dir).expanduser().resolve()
            if personal_dir is not None
            else _PERSONAL_DIR.resolve()
        )
        self._cache: dict[str, str] = {}
        # Track recently loaded files for LLM session logging.
        # Cleared on each drain via ``drain_recent_files()``.
        self._recent_files: list[str] = []

        # Auto-detect micro mode from config when not explicitly passed.
        if use_micro is None:
            use_micro = self._detect_micro_from_config()
        self.use_micro: bool = use_micro

    @staticmethod
    def _detect_micro_from_config() -> bool:
        """Read ``context.micro_instructions`` from the global config."""
        try:
            from captain_claw.config import get_config  # noqa: lazy import
            return bool(get_config().context.micro_instructions)
        except Exception:
            return False

    @staticmethod
    def _resolve_base_dir(base_dir: Path | str | None) -> Path:
        if base_dir is not None:
            return Path(base_dir).expanduser().resolve()
        env_dir = os.getenv("CAPTAIN_CLAW_INSTRUCTIONS_DIR")
        if env_dir:
            return Path(env_dir).expanduser().resolve()
        return (Path(__file__).resolve().parent / "instructions").resolve()

    def _path(self, name: str) -> Path:
        """Return the effective file path.

        When ``use_micro`` is active, tries ``micro_<name>`` first in both
        personal and base directories before falling back to the standard name.
        """
        if self.use_micro:
            micro_name = f"micro_{name}"
            personal_micro = self.personal_dir / micro_name
            if personal_micro.is_file():
                return personal_micro
            base_micro = self.base_dir / micro_name
            if base_micro.is_file():
                return base_micro
            # Fall back to standard template when no micro variant exists.

        personal = self.personal_dir / name
        if personal.is_file():
            return personal
        return self.base_dir / name

    def is_overridden(self, name: str) -> bool:
        """Return ``True`` if a personal override exists for *name*."""
        return (self.personal_dir / name).is_file()

    def load(self, name: str) -> str:
        """Load instruction template content by filename."""
        self._recent_files.append(name)
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

    def drain_recent_files(self) -> list[str]:
        """Return and clear the list of recently loaded instruction files."""
        files = list(dict.fromkeys(self._recent_files))  # dedupe, preserve order
        self._recent_files.clear()
        return files

    def render(self, name: str, **variables: object) -> str:
        """Render template with simple ``str.format`` placeholder substitution."""
        template = self.load(name)
        values: Mapping[str, str] = {k: str(v) for k, v in variables.items()}
        return template.format_map(_SafeFormatDict(values))
