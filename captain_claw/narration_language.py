"""Progress lines in the user's language.

The short text a model writes next to its tool calls goes to the chat live
(``Agent._emit_narration``) — and models write it in English by habit even
when the person writes Croatian and the answer comes in Croatian: "Starting
to collect…" in the middle of a Croatian chat. The system prompt asks for the
user's language; this is the guard for a model that doesn't listen: a line
that reads as English, in a chat whose person clearly doesn't write English,
is not shown (the tool activity still is, and the answer is unchanged).

English is told by its function words that are not words in the languages
the deck's people write (so "i", "a", "to", "on", "me", "data" — Croatian
words too — don't count). Short texts are never judged: too little to tell.

Stdlib only.
"""

from __future__ import annotations

import re
from typing import Any

_ENGLISH = frozenset("""
the and of with is are was were will would i'll i'm i've i'd let let's now next then this
that these those from for first going into onto your my our we it's its which what there
here have has had been being should could can just also all some each about after before
while using use check checking checked start starting started collect collecting look looking
find finding found get getting write writing build building create creating make making
read reading search searching fetch fetching file files results result page pages
""".split())

_WORD_RE = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)?", re.UNICODE)

# Fewer words than this and the text can't be judged.
_MIN_WORDS = 4
# A progress line this English (share of its words) reads as English…
_ENGLISH_LINE = 0.3
# …and a person's message under this share isn't English.
_NOT_ENGLISH = 0.08


def _words(text: str) -> list[str]:
    return [w.casefold().replace("’", "'") for w in _WORD_RE.findall(str(text or ""))]


def english_share(text: str) -> tuple[float, int]:
    """(share of English function words, word count) of *text*."""
    words = _words(text)
    if not words:
        return 0.0, 0
    return sum(1 for w in words if w in _ENGLISH) / len(words), len(words)


def reads_english(text: str) -> bool:
    share, count = english_share(text)
    return count >= _MIN_WORDS and share >= _ENGLISH_LINE


def clearly_not_english(text: str) -> bool:
    share, count = english_share(text)
    return count >= _MIN_WORDS and share < _NOT_ENGLISH


def last_human_text(session: Any, *, scan: int = 60) -> str:
    """The person's latest message in *session* (a scheduler prompt or a
    relayed result is not theirs), as they wrote it — without a surface's
    rules block or a stale clock. "" when there is none in the last *scan*
    messages."""
    from captain_claw import msg_origin

    messages = getattr(session, "messages", None) or []
    for msg in reversed(messages[-scan:]):
        if msg_origin.is_human_input(msg):
            return msg_origin.model_view_text(msg).strip()
    return ""


def out_of_language(narration: str, user_text: str) -> bool:
    """A progress line in English for a person who writes another language."""
    return reads_english(narration) and clearly_not_english(user_text)
