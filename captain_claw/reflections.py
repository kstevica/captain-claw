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


def import_reflections_archive(
    items: list[dict[str, Any]],
    *,
    source_label: str = "imported",
) -> dict[str, int]:
    """Stage externally-supplied reflections under an ``imported/<source>/`` subdir.

    Critically, files are written to a subdirectory so ``load_latest_reflection()``
    (which only globs the top-level ``REFLECTIONS_DIR``) ignores them — the
    receiving agent's *active* personality is never overwritten silently.

    To promote one to active, the user must explicitly run a merge flow.
    Returns counters: stored, skipped_invalid, skipped_duplicate.
    """
    stats = {"stored": 0, "skipped_invalid": 0, "skipped_duplicate": 0}
    safe_label = re.sub(r"[^A-Za-z0-9_.-]", "_", source_label or "imported") or "imported"
    target_dir = REFLECTIONS_DIR / "imported" / safe_label
    target_dir.mkdir(parents=True, exist_ok=True)

    for item in items:
        if not isinstance(item, dict):
            stats["skipped_invalid"] += 1
            continue
        timestamp = str(item.get("timestamp", "")).strip()
        summary = str(item.get("summary", "")).strip()
        if not timestamp or not summary:
            stats["skipped_invalid"] += 1
            continue
        topics = item.get("topics_reviewed") or []
        if not isinstance(topics, list):
            topics = []
        token_usage = item.get("token_usage") or {}
        if not isinstance(token_usage, dict):
            token_usage = {}

        r = Reflection(
            timestamp=timestamp,
            summary=summary,
            topics_reviewed=[str(t) for t in topics],
            token_usage={k: int(v) for k, v in token_usage.items() if isinstance(v, (int, float))},
        )
        path = target_dir / _safe_filename(timestamp)
        if path.is_file():
            stats["skipped_duplicate"] += 1
            continue
        path.write_text(reflection_to_markdown(r), encoding="utf-8")
        stats["stored"] += 1

    return stats


def list_imported_reflections() -> list[dict[str, Any]]:
    """List every reflection staged under ``reflections/imported/<label>/``.

    Returns one entry per imported subdir with the parsed reflection objects
    inside, newest-first. Used by the Flight Deck merge UI and the
    ``/reflection merge`` slash command to let the user pick which imported
    personality to promote.
    """
    root = REFLECTIONS_DIR / "imported"
    if not root.is_dir():
        return []

    sources: list[dict[str, Any]] = []
    for label_dir in sorted(root.iterdir()):
        if not label_dir.is_dir():
            continue
        files = sorted(label_dir.glob("*.md"), reverse=True)
        entries: list[dict[str, Any]] = []
        for f in files:
            try:
                text = f.read_text(encoding="utf-8")
                r = markdown_to_reflection(text)
            except Exception:
                continue
            entries.append(
                {
                    "filename": f.name,
                    "path": f"{label_dir.name}/{f.name}",
                    "timestamp": r.timestamp,
                    "summary": r.summary,
                    "topics_reviewed": list(r.topics_reviewed),
                }
            )
        if entries:
            latest_mtime = 0.0
            try:
                latest_mtime = max(f.stat().st_mtime for f in files)
            except Exception:
                pass
            sources.append(
                {
                    "label": label_dir.name,
                    "count": len(entries),
                    "latest_mtime": latest_mtime,
                    "reflections": entries,
                }
            )
    sources.sort(key=lambda s: s.get("latest_mtime", 0.0), reverse=True)
    return sources


def load_imported_reflection(label: str, filename: str) -> Reflection | None:
    """Load a single imported reflection by ``<label>/<filename>``."""
    safe_label = re.sub(r"[^A-Za-z0-9_.-]", "_", label or "")
    # Allow '+' in filenames so ISO timestamps with timezone offsets
    # (e.g. 2026-04-09T05-01-59+00-00.md) round-trip correctly.
    safe_filename = re.sub(r"[^A-Za-z0-9_.+-]", "_", filename or "")
    if not safe_label or not safe_filename:
        return None
    path = REFLECTIONS_DIR / "imported" / safe_label / safe_filename
    if not path.is_file():
        return None
    try:
        return markdown_to_reflection(path.read_text(encoding="utf-8"))
    except Exception:
        return None


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
            byok=bool(getattr(agent, "_byok_active", False)),
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


# ── Personality-preserving merge ──────────────────────────────────────


async def merge_reflection_with_import(
    agent: Agent,
    *,
    label: str,
    filename: str | None = None,
) -> Reflection:
    """Merge an imported reflection into the active personality via LLM.

    Loads the current top-level reflection plus an imported reflection from
    ``reflections/imported/<label>/[<filename>]`` (newest in the subdir if
    ``filename`` is omitted) and asks the LLM to synthesize a new reflection
    that preserves the agent's current personality while absorbing new
    knowledge/directives from the imported one. The result is saved as a new
    top-level reflection (becoming the active personality).

    The imported file is left in place so the merge is reversible — the user
    can re-run it or pick a different imported bundle.
    """
    from captain_claw.config import get_config
    from captain_claw.llm import LLMResponse, Message
    from captain_claw.session import get_session_manager

    safe_label = re.sub(r"[^A-Za-z0-9_.-]", "_", label or "")
    if not safe_label:
        raise ValueError("label required")

    # 1. Locate the imported reflection.
    source_dir = REFLECTIONS_DIR / "imported" / safe_label
    if not source_dir.is_dir():
        raise FileNotFoundError(f"No imported reflections under label '{safe_label}'")

    if filename:
        # Preserve '+' so ISO timezone offsets in filenames match the real files.
        safe_filename = re.sub(r"[^A-Za-z0-9_.+-]", "_", filename)
        candidate = source_dir / safe_filename
        if not candidate.is_file():
            raise FileNotFoundError(f"Imported reflection not found: {safe_label}/{safe_filename}")
        source_path = candidate
    else:
        md_files = sorted(source_dir.glob("*.md"), reverse=True)
        if not md_files:
            raise FileNotFoundError(f"No .md files under imported/{safe_label}")
        source_path = md_files[0]

    try:
        imported = markdown_to_reflection(source_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"Failed to parse imported reflection: {exc}") from exc

    if not imported.summary.strip():
        raise ValueError("Imported reflection has an empty summary")

    # 2. Load the current active reflection (may be None on a fresh agent).
    current = load_latest_reflection()
    current_summary = current.summary.strip() if current else "(No existing reflection — this is the first personality directive.)"
    current_topics = ", ".join(current.topics_reviewed) if (current and current.topics_reviewed) else "(none)"
    imported_topics = ", ".join(imported.topics_reviewed) if imported.topics_reviewed else "(none)"

    # 3. Build merge prompt. Try a dedicated template first; fall back to
    #    an inline prompt if ops haven't added the template yet.
    try:
        system_prompt = agent.instructions.load("reflection_merge_system_prompt.md")
    except Exception:
        system_prompt = (
            "You are the agent's reflection synthesizer. You preserve the "
            "agent's existing personality, values, and working style while "
            "selectively absorbing useful knowledge, directives, and lessons "
            "from an imported reflection (from a peer agent on another "
            "machine). Your output is a new self-reflection that will replace "
            "the active personality. Output ONLY the merged reflection body "
            "in the same voice and format as the current reflection — no "
            "preamble, no meta-commentary, no headings beyond what the "
            "current reflection already uses."
        )

    try:
        user_prompt = agent.instructions.render(
            "reflection_merge_user_prompt.md",
            current_reflection=current_summary,
            current_topics=current_topics,
            imported_reflection=imported.summary.strip(),
            imported_topics=imported_topics,
            imported_label=safe_label,
            imported_timestamp=imported.timestamp or "(unknown)",
        )
    except Exception:
        user_prompt = (
            f"## Current reflection (active personality — preserve this voice)\n\n"
            f"{current_summary}\n\n"
            f"Current topics reviewed: {current_topics}\n\n"
            f"## Imported reflection (from peer agent '{safe_label}', "
            f"timestamp {imported.timestamp or 'unknown'})\n\n"
            f"{imported.summary.strip()}\n\n"
            f"Imported topics reviewed: {imported_topics}\n\n"
            f"## Task\n\n"
            f"Produce a NEW reflection that:\n"
            f"1. Keeps the active voice, tone, and core commitments of the "
            f"current reflection verbatim where they conflict with the import.\n"
            f"2. Absorbs any NEW factual knowledge, workflow improvements, "
            f"safety lessons, or explicit directives from the imported one "
            f"that don't contradict the current personality.\n"
            f"3. Explicitly rejects any import content that conflicts with "
            f"the active personality (do not mention the rejection in the "
            f"output — just drop it silently).\n"
            f"4. Is no longer than the current reflection + 30%.\n\n"
            f"Output only the merged reflection body."
        )

    messages = [
        Message(role="system", content=system_prompt),
        Message(role="user", content=user_prompt),
    ]

    # 4. Call the LLM (reuses the same guards as generate_reflection).
    cfg = get_config()
    max_tokens = min(2000, int(cfg.model.max_tokens))

    import time as _time
    t0 = _time.monotonic()

    response: LLMResponse = await agent._complete_with_guards(
        messages=messages,
        tools=None,
        interaction_label="reflection_merge",
        max_tokens=max_tokens,
    )

    latency_ms = int((_time.monotonic() - t0) * 1000)
    summary = (response.content or "").strip()
    usage = response.usage or {}

    if not summary:
        raise RuntimeError("LLM returned empty merge result")

    # 5. Save as a new top-level reflection.
    now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")
    merged_topics: list[str] = ["merged from imported personality", f"source: {safe_label}"]
    if current and current.topics_reviewed:
        merged_topics.extend(current.topics_reviewed)
    for t in imported.topics_reviewed:
        if t not in merged_topics:
            merged_topics.append(t)

    merged = Reflection(
        timestamp=now_iso,
        summary=summary,
        topics_reviewed=merged_topics,
        token_usage={
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        },
    )
    save_reflection(merged)

    # 6. Record usage.
    try:
        sm = get_session_manager()
        await sm.record_llm_usage(
            session_id=agent.session.id if agent.session else None,
            interaction="reflection_merge",
            provider=str(getattr(agent.provider, "provider", "") or getattr(agent.provider, "provider_name", "") or ""),
            model=str(getattr(agent.provider, "model", "") or ""),
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
            max_tokens=max_tokens,
            latency_ms=latency_ms,
            task_name="reflection_merge",
            byok=bool(getattr(agent, "_byok_active", False)),
        )
    except Exception as exc:
        log.warning("Failed to record reflection_merge LLM usage", error=str(exc))

    log.info(
        "Reflection merged from imported source",
        source=safe_label,
        source_file=source_path.name,
        timestamp=now_iso,
        prompt_tokens=usage.get("prompt_tokens", 0),
        completion_tokens=usage.get("completion_tokens", 0),
    )

    return merged


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
