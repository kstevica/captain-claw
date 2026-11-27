"""Caveats that take back an answer's own result.

A model can find the deciding fact and still headline the looser reading,
leaving the right one in a caveat further down:

    "All 7 conditions met (at the product level)." … "If both have to be on
    the same model, no model meets every condition."

    "1,000 − 820 = 180. (Read strictly as 'more than 1,000', it would be 181.)"

The reader takes the headline; the caveat was the answer. The check before
delivering (agent_completion_mixin) finds such sentences here by their
wording — no model call — and sends the answer back once to settle on the
reading its evidence supports.

The wording list is general (English, Chinese, Croatian, German), but it was
written after two answers that failed this way (WildClawBench Search 3 and 6,
2026-11-27), so it is partly drawn from them. A sentence it flags wrongly
costs one round: the model keeps its headline and says why. Plain "depends
on" wordings are left out: in answers they mostly state facts (a deal that
depends on an audit), not a second reading of the result.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

# How many flagged sentences the feedback quotes, and how much of each.
MAX_QUOTED = 4
MAX_SENTENCE_CHARS = 300

# Documents the check reads: prose a person reads, not code or data.
PROSE_SUFFIXES = frozenset({"", ".md", ".markdown", ".txt", ".html", ".htm", ".rst", ".adoc", ".org"})

_PATTERNS = [re.compile(p, re.IGNORECASE) for p in (
    # English
    r"\bstrict(?:ly)?\s+(?:speaking|read(?:ing)?|interpret\w*|sense|terms|definition|wording)\b",
    r"\bread(?:ing)?\b[^.;]{0,60}\bstrictly\b",
    r"\b(?:strict|stricter|loose|looser|literal|narrow|narrower|broad|broader)\b[^.;]{0,40}?\b(?:reading|interpretation|definition)\b",
    r"\bif\b[^.;]{0,80}\b(?:is|are|were)\s+(?:a\s+)?(?:hard|strict|firm|must-have)\s+"
    r"(?:requirement|constraint|condition|rule)s?\b",
    r"\btechnically\b",
    r"\((?:at|on)\s+the\s+[\w -]{2,30}\s+level\)",
    r"\bif\s+(?:you|one|we)\s+(?:count|read|interpret|take|treat)\b",
    # Chinese
    r"严格(?:来说|来讲|意义|地说|按|解读|理解|口径)",
    r"(?:如果|若|假如)[^。；]{0,40}(?:硬性|硬|严格|必须)(?:要求|条件|限制)",
    r"在[^。；]{0,12}层面",
    r"(?:字面|狭义|广义)(?:上|理解|解读)",
    # Croatian
    r"\bstrogo\s+(?:gledano|uzevši|govoreći)\b",
    r"\bako\s+je\b[^.;]{0,60}\b(?:nužan|nužno|obavezan|obavezno|strog|tvrd)\w*\s+(?:uvjet|zahtjev)\w*",
    r"\btehnički\s+gledano\b",
    # German
    r"\bstreng\s+genommen\b",
    r"\btechnisch\s+gesehen\b",
)]

_FENCE = re.compile(r"```.*?```", re.DOTALL)
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+|(?<=[。！？；])")
_MARKUP = re.compile(r"[*_`#>|]+")


def _sentences(text: str) -> Iterable[str]:
    for line in _FENCE.sub(" ", text).splitlines():
        if set(line.strip()) <= set("|-: "):        # table rules, blank lines
            continue
        for part in _SENTENCE_END.split(line):
            sentence = " ".join(_MARKUP.sub(" ", part).split())
            if sentence:
                yield sentence


def find_caveats(texts: Iterable[str]) -> list[str]:
    """The sentences in these texts (the reply, the documents the turn wrote)
    that qualify the answer's own result, each once, at most MAX_QUOTED."""
    found: list[str] = []
    for text in texts:
        for sentence in _sentences(str(text or "")):
            if sentence in found or not any(p.search(sentence) for p in _PATTERNS):
                continue
            found.append(sentence)
            if len(found) >= MAX_QUOTED:
                return [s[:MAX_SENTENCE_CHARS] for s in found]
    return [s[:MAX_SENTENCE_CHARS] for s in found]


def feedback(caveats: list[str]) -> str:
    """What the model is told once, when its answer qualifies its own result."""
    quoted = "\n".join(f'- "{c}"' for c in caveats)
    return (
        "[Check before delivering] Your answer qualifies its own result here:\n"
        f"{quoted}\n\n"
        "A caveat like this leaves two readings standing, and the reader takes the headline. "
        "Decide which reading your evidence and the exact wording of the request support. If it "
        "is the caveat's reading, the headline, verdict, number or table must say so, and so must "
        "every file you wrote: fix them with your tools. If your headline is already the right "
        "reading, keep it and say in one line why the other does not apply. Then give your "
        "complete answer again. Do not ask the person a question, and do not mention this check."
    )
