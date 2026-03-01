"""Agent personality management.

Stores and retrieves the agent personality profile from a Markdown file
at ``~/.captain-claw/personality.md``.  The file uses a simple section
format with ``# Name``, ``# Description``, ``# Background``, and
``# Expertise`` headings.

When no file exists on disk, a built-in default is used.  The loaded
personality is cached in memory and automatically refreshed when the
file's modification time changes.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


_PERSONALITY_DIR = Path("~/.captain-claw").expanduser()
PERSONALITY_PATH = _PERSONALITY_DIR / "personality.md"

# ── Data model ────────────────────────────────────────────────────────


@dataclass
class Personality:
    """Agent personality profile."""

    name: str = "Captain Claw"
    description: str = (
        "A powerful AI assistant with claws — part of the Captain Claw family."
    )
    background: str = (
        "Born from the need for a capable, tool-wielding AI agent that can "
        "navigate files, browse the web, manage data, and automate tasks."
    )
    expertise: list[str] = field(
        default_factory=lambda: [
            "Shell scripting and automation",
            "Web research and data extraction",
            "File management and organization",
            "Structured data management",
        ]
    )


DEFAULT_PERSONALITY = Personality()

# ── Markdown serialization ────────────────────────────────────────────


def personality_to_markdown(p: Personality) -> str:
    """Serialize a ``Personality`` to the canonical Markdown format."""
    lines: list[str] = []
    lines.append("# Name\n")
    lines.append(p.name.strip())
    lines.append("")
    lines.append("# Description\n")
    lines.append(p.description.strip())
    lines.append("")
    lines.append("# Background\n")
    lines.append(p.background.strip())
    lines.append("")
    lines.append("# Expertise\n")
    for item in p.expertise:
        lines.append(f"- {item.strip()}")
    lines.append("")
    return "\n".join(lines)


def parse_personality_markdown(text: str) -> Personality:
    """Parse the ``# Section`` Markdown format into a ``Personality``.

    Unrecognized sections are silently ignored.  Missing sections fall
    back to the default values.
    """
    sections: dict[str, str] = {}
    current_section: str | None = None
    current_lines: list[str] = []

    for line in text.splitlines():
        header_match = re.match(r"^#\s+(.+)$", line)
        if header_match:
            if current_section is not None:
                sections[current_section] = "\n".join(current_lines).strip()
            current_section = header_match.group(1).strip().lower()
            current_lines = []
        else:
            current_lines.append(line)

    # Flush last section.
    if current_section is not None:
        sections[current_section] = "\n".join(current_lines).strip()

    name = sections.get("name", DEFAULT_PERSONALITY.name).strip()
    description = sections.get("description", DEFAULT_PERSONALITY.description).strip()
    background = sections.get("background", DEFAULT_PERSONALITY.background).strip()

    # Parse expertise bullet list.
    expertise_raw = sections.get("expertise", "")
    expertise: list[str] = []
    if expertise_raw:
        for item_line in expertise_raw.splitlines():
            item_line = item_line.strip()
            if item_line.startswith("- "):
                item_line = item_line[2:].strip()
            elif item_line.startswith("* "):
                item_line = item_line[2:].strip()
            if item_line:
                expertise.append(item_line)
    if not expertise:
        expertise = list(DEFAULT_PERSONALITY.expertise)

    return Personality(
        name=name or DEFAULT_PERSONALITY.name,
        description=description or DEFAULT_PERSONALITY.description,
        background=background or DEFAULT_PERSONALITY.background,
        expertise=expertise,
    )


# ── Dict conversion ──────────────────────────────────────────────────


def personality_to_dict(p: Personality) -> dict[str, Any]:
    """Convert ``Personality`` to a JSON-serializable dict."""
    return {
        "name": p.name,
        "description": p.description,
        "background": p.background,
        "expertise": list(p.expertise),
    }


def personality_from_dict(d: dict[str, Any]) -> Personality:
    """Build ``Personality`` from a dict (e.g. from JSON body)."""
    expertise_raw = d.get("expertise", [])
    if isinstance(expertise_raw, str):
        expertise = [e.strip() for e in expertise_raw.split(",") if e.strip()]
    elif isinstance(expertise_raw, list):
        expertise = [str(e).strip() for e in expertise_raw if str(e).strip()]
    else:
        expertise = list(DEFAULT_PERSONALITY.expertise)

    return Personality(
        name=str(d.get("name", DEFAULT_PERSONALITY.name)).strip()
        or DEFAULT_PERSONALITY.name,
        description=str(d.get("description", DEFAULT_PERSONALITY.description)).strip()
        or DEFAULT_PERSONALITY.description,
        background=str(d.get("background", DEFAULT_PERSONALITY.background)).strip()
        or DEFAULT_PERSONALITY.background,
        expertise=expertise or list(DEFAULT_PERSONALITY.expertise),
    )


# ── Prompt rendering ─────────────────────────────────────────────────


def personality_to_prompt_block(p: Personality) -> str:
    """Render the personality as a block for system prompt injection.

    The returned string replaces the ``{personality_block}`` placeholder
    inside ``system_prompt.md``.  Curly braces in user-supplied content
    are escaped (doubled) to prevent ``str.format_map`` conflicts.
    """

    def _esc(s: str) -> str:
        return s.replace("{", "{{").replace("}", "}}")

    expertise_list = ", ".join(_esc(e) for e in p.expertise) if p.expertise else "general assistance"

    escaped_name = _esc(p.name)
    # Auto-append family suffix if not already present.
    if "captain claw" not in p.name.lower():
        escaped_name += " of the Captain Claw family"

    return (
        f"You are {escaped_name}.\n"
        f"{_esc(p.description)}\n\n"
        f"Background: {_esc(p.background)}\n\n"
        f"Your areas of expertise: {expertise_list}."
    )


# ── File I/O with caching ────────────────────────────────────────────

_cached_personality: Personality | None = None
_cached_mtime: float = 0.0


def load_personality() -> Personality:
    """Load the personality from disk, using a cached copy when unchanged.

    Returns the default personality when no file exists.
    """
    global _cached_personality, _cached_mtime

    if not PERSONALITY_PATH.is_file():
        # Create the default personality file so it's always on disk.
        default = Personality()
        save_personality(default)
        _cached_personality = default
        return _cached_personality

    try:
        mtime = PERSONALITY_PATH.stat().st_mtime
    except OSError:
        return _cached_personality or Personality()

    if _cached_personality is not None and mtime == _cached_mtime:
        return _cached_personality

    try:
        text = PERSONALITY_PATH.read_text(encoding="utf-8")
        _cached_personality = parse_personality_markdown(text)
        _cached_mtime = mtime
    except Exception:
        _cached_personality = Personality()
        _cached_mtime = 0.0

    return _cached_personality


def save_personality(p: Personality) -> None:
    """Write the personality to ``~/.captain-claw/personality.md``."""
    global _cached_personality, _cached_mtime

    _PERSONALITY_DIR.mkdir(parents=True, exist_ok=True)
    content = personality_to_markdown(p)
    PERSONALITY_PATH.write_text(content, encoding="utf-8")

    # Update cache immediately.
    _cached_personality = p
    try:
        _cached_mtime = PERSONALITY_PATH.stat().st_mtime
    except OSError:
        _cached_mtime = 0.0
