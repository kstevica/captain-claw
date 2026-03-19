"""Visualization style profile management.

Stores and retrieves a visualization style profile from a Markdown file
at ``~/.captain-claw/visualization_style.md``.  The file uses a simple
section format with ``# Name``, ``# Color Palette``, ``# Font Primary``,
etc.

When no file exists on disk, an empty default is used (no style
constraints applied).  The loaded profile is cached in memory and
automatically refreshed when the file's modification time changes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


try:
    _STYLE_DIR = Path("~/.captain-claw").expanduser()
except RuntimeError:
    _STYLE_DIR = Path("/tmp/.captain-claw")
STYLE_PATH = _STYLE_DIR / "visualization_style.md"

# ── Data model ────────────────────────────────────────────────────────


@dataclass
class VisualizationStyle:
    """Visualization style profile."""

    name: str = "Default"
    color_palette: list[str] = field(default_factory=list)
    font_primary: str = ""
    font_headings: str = ""
    font_mono: str = ""
    background_style: str = ""
    chart_style: str = ""
    layout_notes: str = ""
    additional_rules: str = ""
    source_description: str = ""


DEFAULT_STYLE = VisualizationStyle()

# ── Markdown serialization ────────────────────────────────────────────


def visualization_style_to_markdown(s: VisualizationStyle) -> str:
    """Serialize a ``VisualizationStyle`` to the canonical Markdown format."""
    lines: list[str] = []
    lines.append("# Name\n")
    lines.append(s.name.strip())
    lines.append("")
    lines.append("# Color Palette\n")
    for color in s.color_palette:
        lines.append(f"- {color.strip()}")
    lines.append("")
    lines.append("# Font Primary\n")
    lines.append(s.font_primary.strip())
    lines.append("")
    lines.append("# Font Headings\n")
    lines.append(s.font_headings.strip())
    lines.append("")
    lines.append("# Font Mono\n")
    lines.append(s.font_mono.strip())
    lines.append("")
    lines.append("# Background Style\n")
    lines.append(s.background_style.strip())
    lines.append("")
    lines.append("# Chart Style\n")
    lines.append(s.chart_style.strip())
    lines.append("")
    lines.append("# Layout Notes\n")
    lines.append(s.layout_notes.strip())
    lines.append("")
    lines.append("# Additional Rules\n")
    lines.append(s.additional_rules.strip())
    lines.append("")
    if s.source_description.strip():
        lines.append("# Source Description\n")
        lines.append(s.source_description.strip())
        lines.append("")
    return "\n".join(lines)


def parse_visualization_style_markdown(text: str) -> VisualizationStyle:
    """Parse the ``# Section`` Markdown format into a ``VisualizationStyle``.

    Unrecognized sections are silently ignored.
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

    if current_section is not None:
        sections[current_section] = "\n".join(current_lines).strip()

    name = sections.get("name", "Default").strip() or "Default"

    # Parse color palette bullet list.
    palette_raw = sections.get("color palette", "")
    color_palette: list[str] = []
    if palette_raw:
        for item_line in palette_raw.splitlines():
            item_line = item_line.strip()
            if item_line.startswith("- "):
                item_line = item_line[2:].strip()
            elif item_line.startswith("* "):
                item_line = item_line[2:].strip()
            if item_line:
                color_palette.append(item_line)

    return VisualizationStyle(
        name=name,
        color_palette=color_palette,
        font_primary=sections.get("font primary", "").strip(),
        font_headings=sections.get("font headings", "").strip(),
        font_mono=sections.get("font mono", "").strip(),
        background_style=sections.get("background style", "").strip(),
        chart_style=sections.get("chart style", "").strip(),
        layout_notes=sections.get("layout notes", "").strip(),
        additional_rules=sections.get("additional rules", "").strip(),
        source_description=sections.get("source description", "").strip(),
    )


# ── Dict conversion ──────────────────────────────────────────────────


def visualization_style_to_dict(s: VisualizationStyle) -> dict[str, Any]:
    """Convert ``VisualizationStyle`` to a JSON-serializable dict."""
    return {
        "name": s.name,
        "color_palette": list(s.color_palette),
        "font_primary": s.font_primary,
        "font_headings": s.font_headings,
        "font_mono": s.font_mono,
        "background_style": s.background_style,
        "chart_style": s.chart_style,
        "layout_notes": s.layout_notes,
        "additional_rules": s.additional_rules,
        "source_description": s.source_description,
    }


def visualization_style_from_dict(d: dict[str, Any]) -> VisualizationStyle:
    """Build ``VisualizationStyle`` from a dict (e.g. from JSON body)."""
    palette_raw = d.get("color_palette", [])
    if isinstance(palette_raw, str):
        color_palette = [c.strip() for c in palette_raw.split(",") if c.strip()]
    elif isinstance(palette_raw, list):
        color_palette = [str(c).strip() for c in palette_raw if str(c).strip()]
    else:
        color_palette = []

    return VisualizationStyle(
        name=str(d.get("name", "Default")).strip() or "Default",
        color_palette=color_palette,
        font_primary=str(d.get("font_primary", "")).strip(),
        font_headings=str(d.get("font_headings", "")).strip(),
        font_mono=str(d.get("font_mono", "")).strip(),
        background_style=str(d.get("background_style", "")).strip(),
        chart_style=str(d.get("chart_style", "")).strip(),
        layout_notes=str(d.get("layout_notes", "")).strip(),
        additional_rules=str(d.get("additional_rules", "")).strip(),
        source_description=str(d.get("source_description", "")).strip(),
    )


# ── Prompt rendering ─────────────────────────────────────────────────


def visualization_style_to_prompt_block(s: VisualizationStyle) -> str:
    """Render the style as a block for system prompt injection.

    Returns an empty string when no meaningful style is configured,
    so the system prompt stays clean.  Curly braces in user-supplied
    content are escaped (doubled) to prevent ``str.format_map`` conflicts.
    """

    def _esc(val: str) -> str:
        return val.replace("{", "{{").replace("}", "}}")

    # Check if any meaningful field is set.
    has_content = (
        s.color_palette
        or s.font_primary
        or s.font_headings
        or s.font_mono
        or s.background_style
        or s.chart_style
        or s.layout_notes
        or s.additional_rules
    )
    if not has_content:
        return ""

    parts: list[str] = [
        "\nVisualization and report style guide (apply to ALL generated output "
        "— HTML charts, dashboards, visualizations, styled reports, DOCX, and "
        "PPTX documents. Use these colors, fonts, and design rules consistently "
        "across all output formats):"
    ]

    if s.color_palette:
        colors = ", ".join(_esc(c) for c in s.color_palette)
        parts.append(f"- Color palette: {colors}")

    if s.font_primary:
        parts.append(f"- Primary font: {_esc(s.font_primary)}")

    if s.font_headings:
        parts.append(f"- Headings font: {_esc(s.font_headings)}")

    if s.font_mono:
        parts.append(f"- Monospace font: {_esc(s.font_mono)}")

    if s.background_style:
        parts.append(f"- Background style: {_esc(s.background_style)}")

    if s.chart_style:
        parts.append(f"- Chart style: {_esc(s.chart_style)}")

    if s.layout_notes:
        parts.append(f"- Layout: {_esc(s.layout_notes)}")

    if s.additional_rules:
        parts.append(f"- Additional rules: {_esc(s.additional_rules)}")

    return "\n".join(parts)


# ── File I/O with caching ────────────────────────────────────────────

_cached_style: VisualizationStyle | None = None
_cached_mtime: float = 0.0


def load_visualization_style() -> VisualizationStyle:
    """Load the visualization style from disk, using a cached copy when unchanged.

    Returns the default (empty) style when no file exists.
    """
    global _cached_style, _cached_mtime

    if not STYLE_PATH.is_file():
        return _cached_style or VisualizationStyle()

    try:
        mtime = STYLE_PATH.stat().st_mtime
    except OSError:
        return _cached_style or VisualizationStyle()

    if _cached_style is not None and mtime == _cached_mtime:
        return _cached_style

    try:
        text = STYLE_PATH.read_text(encoding="utf-8")
        _cached_style = parse_visualization_style_markdown(text)
        _cached_mtime = mtime
    except Exception:
        _cached_style = VisualizationStyle()
        _cached_mtime = 0.0

    return _cached_style


def save_visualization_style(s: VisualizationStyle) -> None:
    """Write the visualization style to ``~/.captain-claw/visualization_style.md``."""
    global _cached_style, _cached_mtime

    _STYLE_DIR.mkdir(parents=True, exist_ok=True)
    content = visualization_style_to_markdown(s)
    STYLE_PATH.write_text(content, encoding="utf-8")

    _cached_style = s
    try:
        _cached_mtime = STYLE_PATH.stat().st_mtime
    except OSError:
        _cached_mtime = 0.0
