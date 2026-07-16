"""Iskra instruction templates — the being-facing narrative, on disk.

Every instruction a tick speaks to a being (task blocks, gates, frames,
offers) lives as a file under ``captain_claw/instructions/beings/`` instead
of a Python string literal, so a parent can read and tune the species'
narrative without touching code. Two sets share one folder:

  * ``<name>.md``          — the full set (the original prose, verbatim).
  * ``compact_<name>.md``  — the Compact-mode set: same narrative beats,
                             same physics and honesty rules, fewer words.

Compact mode is a per-being flag (``beings.compact_mode``, toggled on the
being panel). ``render(being, name, **vars)`` picks the set from the being
and falls back to the full file when no compact variant exists — one-line
notes don't shrink, so they don't fork.

Placeholders are ``{name}`` substrings replaced literally (no str.format —
the templates are full of literal JSON braces). Unknown placeholders are
left intact, like the InstructionLoader's safe rendering. Files are cached
by mtime, so edits land on the next tick without an FD restart.
"""

from __future__ import annotations

from pathlib import Path

TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "instructions" / "beings"

_cache: dict[str, tuple[float, str]] = {}


def is_compact(being: dict) -> bool:
    return bool(being.get("compact_mode"))


def load(name: str, *, compact: bool = False) -> str:
    """Read one template. Compact tries ``compact_<name>`` first and falls
    back to the full file, so a missing compact variant is never an error."""
    candidates = [f"compact_{name}", name] if compact else [name]
    for candidate in candidates:
        path = TEMPLATE_DIR / candidate
        try:
            mtime = path.stat().st_mtime
        except OSError:
            continue
        cached = _cache.get(candidate)
        if cached and cached[0] == mtime:
            return cached[1]
        text = path.read_text(encoding="utf-8").strip()
        _cache[candidate] = (mtime, text)
        return text
    raise FileNotFoundError(
        f"being instruction template not found: {TEMPLATE_DIR / name}")


def render(being: dict, template: str, **vars: object) -> str:
    """Load the being's set (compact or full) and substitute placeholders.

    Literal replacement, not str.format: the templates carry JSON examples
    whose braces must survive untouched. (``template`` deliberately isn't
    called ``name`` — placeholders like ``name=...`` arrive as kwargs.)"""
    text = load(template, compact=is_compact(being))
    for key, value in vars.items():
        text = text.replace("{" + key + "}", str(value))
    return text
