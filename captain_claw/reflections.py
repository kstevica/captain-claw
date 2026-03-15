"""Self-reflection system for Captain Claw.

Periodically generates self-improvement instructions by reviewing recent
interactions, memory, and task history.  Reflections are persisted as
timestamped Markdown files in ``~/.captain-claw/reflections/``.

The latest reflection is injected into the system prompt via the
``{reflection_block}`` placeholder.  Only the newest file matters so
the prompt stays lean.

Follows the personality.py pattern: dataclass -> markdown file -> mtime
caching -> prompt block with ``_esc()``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from captain_claw.logging import get_logger

if TYPE_CHECKING:
    from captain_claw.agent import Agent

log = get_logger(__name__)

REFLECTIONS_DIR = Path("~/.captain-claw/reflections").expanduser()

# Auto-reflection cooldown in seconds (4 hours).
AUTO_REFLECT_COOLDOWN_SECONDS = 4 * 60 * 60

# Minimum messages since last reflection to trigger auto-reflect.
AUTO_REFLECT_MIN_MESSAGES = 10


# ── Data model ────────────────────────────────────────────────────────


@dataclass
class Reflection:
    """A single self-reflection record."""

    timestamp: str = ""  # ISO 8601
    summary: str = ""  # The self-improvement instructions
    topics_reviewed: list[str] = field(default_factory=list)
    token_usage: dict[str, int] = field(default_factory=dict)


# ── Markdown serialization ────────────────────────────────────────────


def reflection_to_markdown(r: Reflection) -> str:
    """Serialize a ``Reflection`` to the canonical Markdown format."""
    lines: list[str] = []
    lines.append("# Reflection\n")
    lines.append("## Timestamp\n")
    lines.append(r.timestamp)
    lines.append("")
    lines.append("## Summary\n")
    lines.append(r.summary.strip())
    lines.append("")
    lines.append("## Topics Reviewed\n")
    for topic in r.topics_reviewed:
        lines.append(f"- {topic.strip()}")
    lines.append("")
    if r.token_usage:
        lines.append("## Token Usage\n")
        for key, value in r.token_usage.items():
            lines.append(f"- {key}: {value}")
        lines.append("")
    return "\n".join(lines)


def markdown_to_reflection(text: str) -> Reflection:
    """Parse a Markdown reflection file back to a ``Reflection`` dataclass.

    Only our own known ``## Section`` headers are treated as delimiters.
    Any markdown headers inside the LLM-generated summary are preserved
    as content.
    """
    _KNOWN_SECTIONS = {"timestamp", "summary", "topics reviewed", "token usage"}

    sections: dict[str, str] = {}
    current_section: str | None = None
    current_lines: list[str] = []

    for line in text.splitlines():
        header_match = re.match(r"^##\s+(.+)$", line)
        if header_match:
            section_name = header_match.group(1).strip().lower()
            if section_name in _KNOWN_SECTIONS:
                # Flush previous section.
                if current_section is not None:
                    sections[current_section] = "\n".join(current_lines).strip()
                current_section = section_name
                current_lines = []
                continue
        if re.match(r"^#\s+Reflection\s*$", line):
            # Skip the top-level ``# Reflection`` heading only.
            continue
        current_lines.append(line)

    if current_section is not None:
        sections[current_section] = "\n".join(current_lines).strip()

    timestamp = sections.get("timestamp", "").strip()
    summary = sections.get("summary", "").strip()

    # Parse topics bullet list.
    topics_raw = sections.get("topics reviewed", "")
    topics: list[str] = []
    for item_line in topics_raw.splitlines():
        item_line = item_line.strip()
        if item_line.startswith("- "):
            item_line = item_line[2:].strip()
        elif item_line.startswith("* "):
            item_line = item_line[2:].strip()
        if item_line:
            topics.append(item_line)

    # Parse token usage.
    token_usage: dict[str, int] = {}
    usage_raw = sections.get("token usage", "")
    for item_line in usage_raw.splitlines():
        item_line = item_line.strip()
        if item_line.startswith("- "):
            item_line = item_line[2:].strip()
        if ": " in item_line:
            key, val = item_line.split(": ", 1)
            try:
                token_usage[key.strip()] = int(val.strip())
            except ValueError:
                pass

    return Reflection(
        timestamp=timestamp,
        summary=summary,
        topics_reviewed=topics,
        token_usage=token_usage,
    )


# ── Dict conversion ──────────────────────────────────────────────────


def reflection_to_dict(r: Reflection) -> dict[str, Any]:
    """Convert a ``Reflection`` to a JSON-serializable dict."""
    return {
        "timestamp": r.timestamp,
        "summary": r.summary,
        "topics_reviewed": list(r.topics_reviewed),
        "token_usage": dict(r.token_usage),
    }


# ── File I/O ──────────────────────────────────────────────────────────


def _safe_filename(timestamp: str) -> str:
    """Turn an ISO timestamp into a safe filename."""
    return timestamp.replace(":", "-").replace(" ", "_") + ".md"


def save_reflection(r: Reflection) -> Path:
    """Write a reflection to disk and return the file path."""
    global _cached_reflection, _cached_mtime

    REFLECTIONS_DIR.mkdir(parents=True, exist_ok=True)
    filename = _safe_filename(r.timestamp)
    path = REFLECTIONS_DIR / filename
    path.write_text(reflection_to_markdown(r), encoding="utf-8")

    # Invalidate cache so next load picks up the new file.
    _cached_reflection = r
    try:
        _cached_mtime = path.stat().st_mtime
    except OSError:
        _cached_mtime = 0.0

    return path


def list_reflections(limit: int = 50) -> list[Reflection]:
    """Return all reflections sorted newest-first, up to *limit*."""
    if not REFLECTIONS_DIR.is_dir():
        return []

    files = sorted(REFLECTIONS_DIR.glob("*.md"), reverse=True)
    result: list[Reflection] = []
    for f in files[:limit]:
        try:
            text = f.read_text(encoding="utf-8")
            result.append(markdown_to_reflection(text))
        except Exception:
            continue
    return result


def delete_reflection(timestamp: str) -> bool:
    """Delete a reflection by its timestamp.  Returns True if found."""
    global _cached_reflection, _cached_mtime

    if not REFLECTIONS_DIR.is_dir():
        return False

    filename = _safe_filename(timestamp)
    path = REFLECTIONS_DIR / filename
    if path.is_file():
        path.unlink()
        # Invalidate cache.
        _cached_reflection = None
        _cached_mtime = 0.0
        return True
    return False


# ── Cached loading (mtime-based) ─────────────────────────────────────

_cached_reflection: Reflection | None = None
_cached_mtime: float = 0.0


def load_latest_reflection() -> Reflection | None:
    """Load the newest reflection from disk, using cache when unchanged."""
    global _cached_reflection, _cached_mtime

    if not REFLECTIONS_DIR.is_dir():
        return None

    files = sorted(REFLECTIONS_DIR.glob("*.md"), reverse=True)
    if not files:
        return None

    newest = files[0]
    try:
        mtime = newest.stat().st_mtime
    except OSError:
        return _cached_reflection

    if _cached_reflection is not None and mtime == _cached_mtime:
        return _cached_reflection

    try:
        text = newest.read_text(encoding="utf-8")
        _cached_reflection = markdown_to_reflection(text)
        _cached_mtime = mtime
    except Exception:
        _cached_reflection = None
        _cached_mtime = 0.0

    return _cached_reflection


# ── Prompt block rendering ────────────────────────────────────────────


def _esc(s: str) -> str:
    """Double curly braces for ``str.format_map`` safety."""
    return s.replace("{", "{{").replace("}", "}}")


def reflection_to_prompt_block(r: Reflection | None) -> str:
    """Render the latest reflection as a block for system prompt injection.

    Returns an empty string when no reflection exists (the template
    placeholder collapses cleanly).
    """
    if r is None or not r.summary.strip():
        return ""

    return (
        "\nSelf-reflection (latest self-assessment and improvement directives):\n"
        + _esc(r.summary.strip())
        + "\n"
    )


# ── LLM-based reflection generation ──────────────────────────────────


async def generate_reflection(agent: Agent) -> Reflection:
    """Generate a new self-reflection using the LLM.

    Gathers recent context (session messages, memory, previous reflection)
    and produces self-improvement instructions.
    """
    from captain_claw.config import get_config
    from captain_claw.llm import LLMResponse, Message
    from captain_claw.session import get_session_manager

    # 1. Load previous reflection.
    prev = load_latest_reflection()
    previous_summary = prev.summary.strip() if prev else "No previous reflection."

    # 2. Gather recent session messages.
    recent_messages_text = ""
    topics: list[str] = []
    if agent.session and agent.session.messages:
        messages = agent.session.messages[-20:]  # Last 20 messages
        lines: list[str] = []
        for m in messages:
            role = m.get("role", "unknown")
            content = str(m.get("content", ""))[:500]
            lines.append(f"[{role}] {content}")
        recent_messages_text = "\n".join(lines)
        topics.append("recent conversation messages")

    # 3. Gather memory facts.
    memory_text = ""
    try:
        if agent.session and hasattr(agent, "session_manager"):
            sm = agent.session_manager
            if hasattr(sm, "get_memory_facts"):
                facts = await sm.get_memory_facts(limit=30)
                if facts:
                    memory_text = "\n".join(f"- {f}" for f in facts)
                    topics.append("memory facts")
    except Exception:
        pass

    # 4. Gather completed tasks/cron since last reflection.
    tasks_text = ""
    try:
        if hasattr(agent, "session_manager"):
            sm = agent.session_manager
            if hasattr(sm, "get_recent_cron_history"):
                since = prev.timestamp if prev else None
                history = await sm.get_recent_cron_history(since=since, limit=20)
                if history:
                    lines = [f"- {h.get('task_name', 'unknown')}: {h.get('status', '')}" for h in history]
                    tasks_text = "\n".join(lines)
                    topics.append("completed tasks/cron jobs")
    except Exception:
        pass

    if not topics:
        topics.append("general self-assessment")

    # 5. Build LLM messages.
    system_prompt = agent.instructions.load("reflection_system_prompt.md")
    user_prompt = agent.instructions.render(
        "reflection_user_prompt.md",
        previous_reflection=previous_summary,
        recent_messages=recent_messages_text or "(No recent messages available.)",
        memory_facts=memory_text or "(No memory facts available.)",
        tasks_summary=tasks_text or "(No task history available.)",
    )

    messages = [
        Message(role="system", content=system_prompt),
        Message(role="user", content=user_prompt),
    ]

    # 6. Call the LLM.
    cfg = get_config()
    max_tokens = min(1500, int(cfg.model.max_tokens))

    import time as _time
    t0 = _time.monotonic()

    response: LLMResponse = await agent._complete_with_guards(
        messages=messages,
        tools=None,
        interaction_label="reflection",
        max_tokens=max_tokens,
    )

    latency_ms = int((_time.monotonic() - t0) * 1000)
    summary = (response.content or "").strip()
    usage = response.usage or {}

    # 7. Build and save reflection.
    now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")
    reflection = Reflection(
        timestamp=now_iso,
        summary=summary,
        topics_reviewed=topics,
        token_usage={
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        },
    )

    save_reflection(reflection)

    # 8. Log token usage.
    try:
        sm = get_session_manager()
        await sm.record_llm_usage(
            session_id=agent.session.id if agent.session else None,
            interaction="reflection",
            provider=str(getattr(agent.provider, "provider", "") or getattr(agent.provider, "provider_name", "") or ""),
            model=str(getattr(agent.provider, "model", "") or ""),
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
            max_tokens=max_tokens,
            latency_ms=latency_ms,
            task_name="self_reflection",
        )
    except Exception as exc:
        log.warning("Failed to record reflection LLM usage", error=str(exc))

    log.info(
        "Self-reflection generated",
        timestamp=now_iso,
        prompt_tokens=usage.get("prompt_tokens", 0),
        completion_tokens=usage.get("completion_tokens", 0),
    )

    return reflection


# ── Auto-reflection trigger ───────────────────────────────────────────


async def maybe_auto_reflect(agent: Agent) -> Reflection | None:
    """Conditionally trigger a self-reflection after an agent turn.

    Runs only when:
      - At least ``AUTO_REFLECT_COOLDOWN_SECONDS`` since last reflection
      - At least ``AUTO_REFLECT_MIN_MESSAGES`` in the current session

    Returns the new Reflection or None if skipped.
    """
    try:
        # Check session has enough messages.
        if not agent.session or len(agent.session.messages) < AUTO_REFLECT_MIN_MESSAGES:
            return None

        # Check cooldown.
        prev = load_latest_reflection()
        if prev and prev.timestamp:
            try:
                last_dt = datetime.fromisoformat(prev.timestamp)
                now = datetime.now(timezone.utc)
                elapsed = (now - last_dt).total_seconds()
                if elapsed < AUTO_REFLECT_COOLDOWN_SECONDS:
                    return None
            except (ValueError, TypeError):
                pass  # Can't parse timestamp — allow reflection.

        log.info("Auto-reflection triggered")
        return await generate_reflection(agent)

    except Exception as exc:
        log.warning("Auto-reflection failed (non-fatal)", error=str(exc))
        return None
