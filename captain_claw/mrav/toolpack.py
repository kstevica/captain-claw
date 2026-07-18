"""Mrav toolpack — compact core tool schemas + one-line index + paging.

The full registry's schemas are ~26k tokens; Mrav's whole prompt is 8k.
So: a small core is always visible as hand-tight compact text (~1.5-2k
tokens), everything else is a one-line index entry, and ``open_tool`` pins a
compacted schema on demand (LRU). Compaction derives from the REAL schema at
runtime, so it cannot drift; the test suite additionally asserts compact
params are always a subset of the real ones.
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass, field
from typing import Any

# Ordered — the prompt renders these first, in this order, every step
# (byte-stable prefix → prefix-cache/KV hits).
CORE_TOOLS: tuple[str, ...] = (
    "read",
    "write",
    "glob",
    "grep",
    "shell",
    "web_search",
    "web_fetch",
    "todo",
)

# Hand-tightened one-line descriptions for the core pack. Anything not here
# falls back to the first sentence of the registry description.
CORE_DESCRIPTIONS: dict[str, str] = {
    "read": "Read a text file (whole or a line range)",
    "write": "Write/overwrite a file with the given content",
    "glob": "List files matching a glob pattern",
    "grep": "Search file contents for a regex pattern",
    "shell": "Run a shell command in the workspace",
    "web_search": "Search the web; returns titles, URLs, snippets",
    "web_fetch": "Fetch a URL and return its readable text",
    "todo": "Manage the task todo list (add/list/done)",
}

_TYPE_SHORT = {
    "string": "str",
    "integer": "int",
    "number": "num",
    "boolean": "bool",
    "array": "list",
    "object": "obj",
}

_MAX_PARAMS = 6
_MAX_DESC_CHARS = 110
_MAX_INDEX_DESC_CHARS = 64


def _first_sentence(text: str, limit: int) -> str:
    text = " ".join((text or "").split())
    for sep in (". ", "! ", "? ", "\n"):
        idx = text.find(sep)
        if 0 < idx < limit:
            return text[: idx + 1].strip().rstrip(".")
    return textwrap.shorten(text, width=limit, placeholder="…") if text else ""


def _param_token(name: str, spec: dict[str, Any], required: bool) -> str:
    ptype = _TYPE_SHORT.get(str(spec.get("type", "")), str(spec.get("type", "")) or "any")
    enum = spec.get("enum")
    if isinstance(enum, list) and 0 < len(enum) <= 6:
        ptype = "|".join(str(v) for v in enum)
    return f"{name}{'*' if required else ''}:{ptype}"


@dataclass
class CompactTool:
    """A budget-friendly rendering of one tool definition."""

    name: str
    description: str
    param_line: str
    param_names: list[str] = field(default_factory=list)

    def render(self) -> str:
        if self.param_line:
            return f"{self.name} — {self.description}\n  args: {self.param_line}"
        return f"{self.name} — {self.description}\n  args: (none)"


def compact_definition(definition: dict[str, Any], description_override: str | None = None) -> CompactTool:
    """Compact an OpenAI-style tool definition to ~2 lines.

    Keeps every required param; optional params only while the total stays
    within ``_MAX_PARAMS`` (required-first ordering).
    """
    name = str(definition.get("name", "")).strip()
    desc = description_override or _first_sentence(str(definition.get("description", "")), _MAX_DESC_CHARS)
    params = definition.get("parameters") or {}
    properties = params.get("properties") or {}
    required = [p for p in (params.get("required") or []) if p in properties]

    ordered = list(required) + [p for p in properties if p not in required]
    kept: list[str] = []
    tokens: list[str] = []
    for pname in ordered:
        if pname not in required and len(kept) >= _MAX_PARAMS:
            continue
        spec = properties.get(pname) or {}
        if not isinstance(spec, dict):
            spec = {}
        tokens.append(_param_token(pname, spec, pname in required))
        kept.append(pname)
    return CompactTool(name=name, description=desc, param_line=", ".join(tokens), param_names=kept)


@dataclass
class ToolpackContext:
    """What one ACT step can see: visible schemas + the index of the rest."""

    visible: dict[str, CompactTool]
    index_names: list[str]
    defs_text: str
    index_text: str

    @property
    def visible_names(self) -> set[str]:
        return set(self.visible.keys())

    @property
    def all_names(self) -> set[str]:
        return set(self.visible.keys()) | set(self.index_names)


def build_toolpack(
    definitions: list[dict[str, Any]],
    pinned: list[str] | None = None,
    core: tuple[str, ...] = CORE_TOOLS,
) -> ToolpackContext:
    """Build the step's tool view from live registry definitions.

    ``definitions`` come from ``ToolRegistry.get_definitions(...)`` so any
    session/task tool policy has already been applied upstream.
    """
    by_name = {str(d.get("name", "")).strip(): d for d in definitions if d.get("name")}
    pinned = [p for p in (pinned or []) if p in by_name]

    visible: dict[str, CompactTool] = {}
    for name in core:
        if name in by_name:
            visible[name] = compact_definition(by_name[name], CORE_DESCRIPTIONS.get(name))
    for name in pinned:
        if name not in visible:
            visible[name] = compact_definition(by_name[name])

    index_names = sorted(n for n in by_name if n not in visible)
    index_lines = []
    for name in index_names:
        desc = _first_sentence(str(by_name[name].get("description", "")), _MAX_INDEX_DESC_CHARS)
        index_lines.append(f"{name} — {desc}" if desc else name)

    defs_text = "\n".join(visible[n].render() for n in visible)
    index_text = "\n".join(index_lines)
    return ToolpackContext(
        visible=visible,
        index_names=index_names,
        defs_text=defs_text,
        index_text=index_text,
    )
