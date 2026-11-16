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

The markup is often broken, and a strict reader then finds no call at all, so
the reply ships as an answer and the tool silently never runs (cohort001: 24
replies in one afternoon on DeepSeek Flash). Seen in the wild:

- ``</invoke>`` left out — the next ``invoke`` or the ``calls`` tag ends it;
- a stray fullwidth ``＜`` in a closer: ``</＜｜DSML｜｜ invoke>``;
- the block "closed" with an opening ``<｜｜DSML｜｜ calls>``;
- single-quoted attributes: ``invoke name='insights'``.

So the tags are walked in order rather than matched in pairs. An invoke runs
when a DSML tag ends it; one still open when the text ends (cut off, or quoted
in prose) does not run.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

_LT = r"[<＜]"
_GT = r"[>＞]"
_BAR = r"[|｜]+"
# Any DSML tag: optional "/", optional stray "＜" in place of a bar, the bars,
# DSML, the bars, then the tag's kind and attributes.
_TAG_RE = re.compile(
    _LT + r"\s*(/)?\s*" + _LT + r"?\s*" + _BAR + r"\s*DSML\s*" + _BAR + r"\s*"
    r"((?:function_|tool_)?calls|invoke|parameter)\b([^>＞\n]*)" + _GT,
    re.IGNORECASE,
)
# A tag cut off at the very end of the text (no closing ">").
_CUT_TAG_RE = re.compile(
    r"\A\s*" + _LT + r"\s*/?\s*" + _LT + r"?\s*" + _BAR + r"[^>＞]*\Z")
_NAME_RE = re.compile(r"""\bname\s*=\s*(["'])(.*?)\1""", re.IGNORECASE | re.DOTALL)
_STRING_RE = re.compile(r"""\bstring\s*=\s*(["'])(true|false)\1""", re.IGNORECASE)


def has_dsml(text: str) -> bool:
    return bool(text) and bool(_TAG_RE.search(text))


def _value(raw: str, is_string: str | None) -> Any:
    text = raw.strip()
    if is_string and is_string.lower() == "true":
        return raw.strip("\n")
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return raw.strip("\n")


@dataclass
class _Walk:
    calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    spans: list[tuple[int, int]] = field(default_factory=list)


def _walk(text: str) -> _Walk:
    """Read the calls and the spans of markup in *text*, tag by tag."""
    out = _Walk()
    region: int | None = None              # start of the markup being read
    invoke: tuple[str, dict[str, Any]] | None = None
    param: tuple[str, str | None, int] | None = None   # name, string flag, value start
    prev_end = 0

    def close_param(at: int) -> None:
        nonlocal param
        if param is not None and invoke is not None:
            name, flag, start = param
            invoke[1][name] = _value(text[start:at], flag)
        param = None

    def end_invoke(at: int) -> None:
        nonlocal invoke
        close_param(at)
        if invoke is not None and invoke[0]:
            out.calls.append(invoke)
        invoke = None

    for tag in _TAG_RE.finditer(text):
        gap = text[prev_end:tag.start()]
        if region is not None and param is None and gap.strip():
            # Prose between tags: the markup ended at the previous tag, and an
            # invoke no DSML tag ended doesn't run.
            out.spans.append((region, prev_end))
            region, invoke = None, None
        if region is None:
            region = tag.start()
        closing = bool(tag.group(1))
        kind = tag.group(2).lower()
        attrs = tag.group(3) or ""
        if kind == "parameter":
            if closing:
                close_param(tag.start())
            elif invoke is not None:
                close_param(tag.start())
                name = _NAME_RE.search(attrs)
                flag = _STRING_RE.search(attrs)
                if name and name.group(2).strip():
                    param = (name.group(2).strip(), flag.group(2) if flag else None, tag.end())
        elif kind == "invoke":
            end_invoke(tag.start())
            if not closing:
                name = _NAME_RE.search(attrs)
                invoke = (name.group(2).strip() if name else "", {})
        else:  # calls / function_calls, opening or closing: either ends an invoke
            end_invoke(tag.start())
        prev_end = tag.end()

    if region is not None:
        tail = text[prev_end:]
        if param is not None or not tail.strip() or _CUT_TAG_RE.match(tail):
            # Cut off mid-call (or nothing after it): all of it is markup, and
            # the unfinished invoke doesn't run.
            out.spans.append((region, len(text)))
        else:
            out.spans.append((region, prev_end))
    return out


def dsml_calls(text: str) -> list[tuple[str, dict[str, Any]]]:
    """``(tool name, arguments)`` for every DSML invoke a DSML tag ended."""
    if not has_dsml(text):
        return []
    return _walk(text).calls


def strip_dsml(text: str) -> str:
    """*text* without DSML call blocks or stray DSML tags."""
    if not has_dsml(text):
        return text
    parts: list[str] = []
    pos = 0
    for start, end in _walk(text).spans:
        parts.append(text[pos:start])
        pos = end
    parts.append(text[pos:])
    out = _TAG_RE.sub("", "".join(parts))
    return re.sub(r"\n{3,}", "\n\n", out).strip()
