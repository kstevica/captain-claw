"""Agent personality management.

Stores and retrieves the agent personality profile from a Markdown file
at ``~/.captain-claw/personality.md``.  The file uses a simple section
format with ``# Name``, ``# Description``, ``# Background``, and
``# Expertise`` headings.

Per-user (Telegram) personalities are stored as individual ``.md`` files
under ``~/.captain-claw/personalities/{user_id}.md``.

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
_USER_PERSONALITIES_DIR = _PERSONALITY_DIR / "personalities"

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


def user_context_to_prompt_block(p: Personality) -> str:
    """Render a user personality as context about the *user* (not the agent).

    This tells the LLM who it is talking to — the user's name, expertise,
    and background — so it can tailor responses to that perspective.
    The returned string replaces the ``{{user_context_block}}`` placeholder.
    """

    def _esc(s: str) -> str:
        return s.replace("{", "{{").replace("}", "}}")

    expertise_list = ", ".join(_esc(e) for e in p.expertise) if p.expertise else "various topics"

    return (
        f"\nThe user you are talking to is {_esc(p.name)}.\n"
        f"{_esc(p.description)}\n"
        f"Background: {_esc(p.background)}\n"
        f"Areas of expertise: {expertise_list}.\n"
        f"Tailor your responses to their level of expertise and perspective."
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


# ── Per-user (Telegram) personalities ────────────────────────────────

_user_cache: dict[str, tuple[Personality, float]] = {}  # user_id → (personality, mtime)


def _user_personality_path(user_id: str) -> Path:
    """Return the file path for a user's personality."""
    # Sanitize user_id to prevent path traversal.
    safe_id = re.sub(r"[^a-zA-Z0-9_-]", "", user_id)
    return _USER_PERSONALITIES_DIR / f"{safe_id}.md"


def load_user_personality(user_id: str) -> Personality | None:
    """Load a user personality from disk, returning ``None`` if not set."""
    path = _user_personality_path(user_id)
    if not path.is_file():
        _user_cache.pop(user_id, None)
        return None

    try:
        mtime = path.stat().st_mtime
    except OSError:
        return _user_cache.get(user_id, (None, 0.0))[0]

    cached = _user_cache.get(user_id)
    if cached is not None and cached[1] == mtime:
        return cached[0]

    try:
        text = path.read_text(encoding="utf-8")
        p = parse_personality_markdown(text)
        _user_cache[user_id] = (p, mtime)
        return p
    except Exception:
        return None


def save_user_personality(user_id: str, p: Personality) -> None:
    """Write a user personality to ``~/.captain-claw/personalities/{user_id}.md``."""
    _USER_PERSONALITIES_DIR.mkdir(parents=True, exist_ok=True)
    path = _user_personality_path(user_id)
    content = personality_to_markdown(p)
    path.write_text(content, encoding="utf-8")

    try:
        mtime = path.stat().st_mtime
    except OSError:
        mtime = 0.0
    _user_cache[user_id] = (p, mtime)


def delete_user_personality(user_id: str) -> bool:
    """Delete a user personality file.  Returns ``True`` if removed."""
    path = _user_personality_path(user_id)
    _user_cache.pop(user_id, None)
    if path.is_file():
        path.unlink()
        return True
    return False


def list_user_personalities() -> list[dict[str, Any]]:
    """Return a list of all user personalities with their user IDs.

    Each entry is a dict: ``{"user_id": ..., "name": ..., ...}``.
    """
    result: list[dict[str, Any]] = []
    if not _USER_PERSONALITIES_DIR.is_dir():
        return result

    for md_file in sorted(_USER_PERSONALITIES_DIR.glob("*.md")):
        user_id = md_file.stem
        try:
            text = md_file.read_text(encoding="utf-8")
            p = parse_personality_markdown(text)
            entry = personality_to_dict(p)
            entry["user_id"] = user_id
            result.append(entry)
        except Exception:
            continue
    return result


def load_effective_personality(user_id: str | None = None) -> Personality:
    """Load the effective personality for a request.

    If *user_id* is provided and that user has a custom personality,
    it takes precedence.  Otherwise the global personality is returned.
    """
    if user_id:
        user_p = load_user_personality(user_id)
        if user_p is not None:
            return user_p
    return load_personality()
