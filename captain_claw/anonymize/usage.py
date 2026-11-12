"""Where a token went: the short context it was sent out or came back in.

A context is cut from text that already holds tokens — what went to a model,
or what a model wrote back — never from the original, so it carries no real
value of its own. The vault keeps it encrypted anyway, next to the values.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator
from typing import Any

from captain_claw.anonymize.engine import TOKEN_RE, make_token

CONTEXT_CHARS = 80   # on each side of the token
MAX_CONTEXT = 300


def snippet(text: str, start: int, end: int, width: int = CONTEXT_CHARS) -> str:
    """The words around ``text[start:end]``, whitespace collapsed, never a
    token cut in half, "…" where it was cut."""
    a, b = max(0, start - width), min(len(text), end + width)
    for m in TOKEN_RE.finditer(text, max(0, a - 40), min(len(text), b + 40)):
        if m.start() < a < m.end():
            a = m.start()
        if m.start() < b < m.end():
            b = m.end()
    if a > 0:
        space = text.find(" ", a, start)
        a = space + 1 if space != -1 else a
    if b < len(text):
        space = text.rfind(" ", end, b)
        b = space if space != -1 else b
    body = re.sub(r"\s+", " ", text[a:b]).strip()
    return (("…" if a > 0 else "") + body + ("…" if b < len(text) else ""))[:MAX_CONTEXT]


def token_contexts(texts: Iterable[str], wanted: set[str] | None = None) -> dict[str, tuple[int, str]]:
    """Every token in *texts* (only those in *wanted*, when given): how many
    times it appears, and the context of its first appearance."""
    found: dict[str, tuple[int, str]] = {}
    for text in texts:
        if not text or ("⟦" not in text and "[[" not in text):
            continue
        for m in TOKEN_RE.finditer(text):
            token = make_token(m.group(1) or m.group(3), int(m.group(2) or m.group(4)))
            if wanted is not None and token not in wanted:
                continue
            count, context = found.get(token, (0, ""))
            found[token] = (count + 1, context or snippet(text, m.start(), m.end()))
    return found


def strings(value: Any) -> Iterator[str]:
    """Every string in a JSON value (a chat frame)."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for v in value:
            yield from strings(v)
    elif isinstance(value, dict):
        for v in value.values():
            yield from strings(v)
