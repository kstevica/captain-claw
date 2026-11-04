"""DeepSeek's own tool-call markup ("DSML") written into a reply's text.

DeepSeek models sometimes answer with their native tool-call syntax as plain
content instead of a structured tool call — the provider passes it through
untouched, and it reached the user's chat as raw markup (cohort001, four
times in one chat)::

    <｜DSML｜function_calls>
    <｜DSML｜invoke name="read">
    <｜DSML｜parameter name="path" string="true">vfs:x/notes.md</｜DSML｜parameter>
    </｜DSML｜invoke>
    </｜DSML｜function_calls>

The bars are fullwidth (U+FF5C), sometimes doubled, and the wrapper tag is
``function_calls`` or ``calls``. :func:`dsml_calls` reads the calls so the
agent can run them; :func:`strip_dsml` takes the markup out of what is shown.
"""

from __future__ import annotations

import json
import re
from typing import Any

_BAR = r"[|｜]+"
_OPEN = r"<\s*" + _BAR + r"\s*DSML\s*" + _BAR + r"\s*"
_CLOSE = r"<\s*/\s*" + _BAR + r"\s*DSML\s*" + _BAR + r"\s*"

_INVOKE_RE = re.compile(
    _OPEN + r'invoke\s+name\s*=\s*"([^"]+)"\s*>(.*?)' + _CLOSE + r"invoke\s*>",
    re.DOTALL | re.IGNORECASE,
)
_PARAM_RE = re.compile(
    _OPEN + r'parameter\s+name\s*=\s*"([^"]+)"(?:\s+string\s*=\s*"(true|false)")?\s*>(.*?)'
    + _CLOSE + r"parameter\s*>",
    re.DOTALL | re.IGNORECASE,
)
# Any DSML tag at all, and a whole calls block (closed, or cut off at the end).
_ANY_TAG_RE = re.compile(r"</?\s*" + _BAR + r"\s*DSML\s*" + _BAR + r"[^>\n]*>", re.IGNORECASE)
_BLOCK_RE = re.compile(
    _OPEN + r"(?:function_)?calls\s*>.*?(?:" + _CLOSE + r"(?:function_)?calls\s*>|\Z)",
    re.DOTALL | re.IGNORECASE,
)
_LONE_INVOKE_RE = re.compile(
    _OPEN + r"invoke\b.*?(?:" + _CLOSE + r"invoke\s*>|\Z)", re.DOTALL | re.IGNORECASE)


def has_dsml(text: str) -> bool:
    return bool(text) and bool(_ANY_TAG_RE.search(text))


def _value(raw: str, is_string: str | None) -> Any:
    text = raw.strip()
    if is_string and is_string.lower() == "true":
        return raw.strip("\n")
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return raw.strip("\n")


def dsml_calls(text: str) -> list[tuple[str, dict[str, Any]]]:
    """``(tool name, arguments)`` for every complete DSML invoke in *text*."""
    if not has_dsml(text):
        return []
    calls: list[tuple[str, dict[str, Any]]] = []
    for invoke in _INVOKE_RE.finditer(text):
        name = invoke.group(1).strip()
        args = {m.group(1).strip(): _value(m.group(3), m.group(2))
                for m in _PARAM_RE.finditer(invoke.group(2))}
        if name:
            calls.append((name, args))
    return calls


def strip_dsml(text: str) -> str:
    """*text* without DSML call blocks or stray DSML tags."""
    if not has_dsml(text):
        return text
    out = _BLOCK_RE.sub("", text)
    out = _LONE_INVOKE_RE.sub("", out)
    out = _ANY_TAG_RE.sub("", out)
    return re.sub(r"\n{3,}", "\n\n", out).strip()
