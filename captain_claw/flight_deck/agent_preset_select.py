"""What a budgeted read reads (see agent_preset) — token-free.

Material bigger than the person's budget isn't read front to back. It is cut
into its sections (``units``: each heading's own text, from the outline the
section maps use; a long one, and a document without headings, in pieces),
each scored for how much it tells an agent to do (``instructive``: rule
words, numbered steps, definitions, formulas, numbers with a unit) and —
when the person said what to focus on — for how close it is to that
(``focus_scores``, by the local model2vec embeddings). ``select`` takes the
best of them up to the budget, every document getting a share, and keeps
them in document order, so each section reads in context. The rest stays
whole in the agent's knowledge, searchable; nothing here calls a model.
"""

from __future__ import annotations

import re
from typing import Any

from captain_claw.flight_deck import agent_preset_outline as poutline
from captain_claw.logging import get_logger

log = get_logger(__name__)

# How much material a read takes, in tokens: the person picks one.
BUDGETS = {"quick": 60_000, "standard": 250_000, "deep": 1_000_000}
DEFAULT_BUDGET = "standard"
UNIT_MAX_CHARS = 16_000      # a section longer than this is read in pieces
MIN_UNIT_CHARS = 200         # a heading with less under it has nothing to read
FOCUS_WEIGHT = 0.6           # with a focus: how much closeness counts, against instructiveness
_FOCUS_CHARS = 800           # of a section, what its closeness to the focus is judged on
_USAGE_WEIGHT = {"rules": 1.25, "procedures": 1.25, "templates": 1.15, "reference": 0.8}

_FENCE = re.compile(r"^\s*(```|~~~)")
# Words that make a sentence a rule — English and Croatian.
_RULE_WORDS = re.compile(
    r"\b(must|never|always|should|shall|required|requires?|mandatory|do not|does not|cannot|may not|ensure|"
    r"avoid|only if|unless|at least|at most|no (?:more|later|less|fewer) than|deadline|forbidden|"
    r"prohibited|make sure|important|"
    r"mora(?:ju|mo|te|š)?|nikad(?:a)?|uvijek|obavez\w*|obvez\w*|zabranjen\w*|potrebno|treba(?:ju)?|"
    r"ne smije\w*|najmanje|najviše|rok)\b", re.I)
_STEP_LINE = re.compile(r"^\s*(?:\d{1,2}[.)]|step \d+|korak \d+)\s+\S", re.I | re.M)
_DEFINITION = re.compile(
    r"\b(?:means|is defined as|refers to|is called|stands for|označava|znači|podrazumijeva)\b"
    r"|^\s*(?:\*\*[^*\n]{2,60}\*\*|[A-ZČĆŠŽĐ][\w /-]{1,40})\s*[:—–]\s+\S", re.M)
_FORMULA = re.compile(r"(?:^|\s)[A-Za-z_][\w()/]{0,20}\s?=\s?[^=\n]{2,}|\$[^$\n]+\$|\\frac|∑|√|\bformula\b", re.I | re.M)
_AMOUNT = re.compile(
    r"\b\d+(?:[.,]\d+)?\s?(?:%|eur\b|€|\$|usd\b|days?\b|dana\b|hours?\b|sati\b|weeks?\b|tjed\w*|months?\b|"
    r"mjesec\w*|years?\b|godin\w*|minutes?\b|minut\w*)", re.I)
# Lines that are links, pictures or table rules, not prose.
_NOISE_LINE = re.compile(r"^\s*(?:\[[^\]]*\]:\s*\S|\[?!\[|<img|<a\s|\|?\s*:?-{3,}|https?://\S+\s*$)", re.I)
# Boilerplate that reads like rules but isn't the owner's: a changelog's entries ("## [1.4.2] -
# 2024-03-01", "v2.0"), licenses, copyright and warranty notices.
_BOILERPLATE_TITLE = re.compile(
    r"^\[?v?\d+\.\d+(?:\.\d+)?\]?|^\d{4}-\d{2}-\d{2}|licen[cs]e|copyright|warranty|disclaimer|"
    r"change ?log|release notes|third[- ]party notices", re.I)


def _cuts(text: str, start: int, end: int, limit: int) -> list[tuple[int, int]]:
    """``[start, end)`` in spans of at most *limit* characters, cut at a
    paragraph break, else a line break, else where it must."""
    out: list[tuple[int, int]] = []
    a = start
    while end - a > limit:
        window = text[a:a + limit]
        cut = window.rfind("\n\n")
        if cut < limit // 2:
            cut = window.rfind("\n")
        if cut < limit // 2:
            cut = limit
        out.append((a, a + cut))
        a += cut
    out.append((a, end))
    return [(x, y) for x, y in out if text[x:y].strip()]


def units(text: str, max_chars: int = UNIT_MAX_CHARS) -> list[dict[str, Any]]:
    """The document as reading units, in order and covering all of it:
    ``start``/``end`` (offsets), ``title`` and ``path`` (its parents'
    titles). Each heading's own text, up to its first subsection, is one —
    the text before the first heading too; a long one is cut into pieces of
    at most *max_chars*, and so is a text without headings."""
    sections = poutline.outline(text)
    spans: list[tuple[int, int, str, list[str]]] = []
    if not sections:
        spans.append((0, len(text), "", []))
    else:
        if text[:sections[0]["start"]].strip():
            spans.append((0, sections[0]["start"], "Its opening", []))
        parents: list[tuple[int, str]] = []
        for s in sections:
            while parents and parents[-1][0] >= s["level"]:
                parents.pop()
            spans.append((s["start"], s["own"], s["title"], [t for _, t in parents]))
            parents.append((s["level"], s["title"]))
    out: list[dict[str, Any]] = []
    for start, end, title, path in spans:
        pieces = _cuts(text, start, end, max(1000, max_chars))
        for n, (a, b) in enumerate(pieces, 1):
            name = (f"{title} ({n}/{len(pieces)})" if len(pieces) > 1 else title) if title \
                else f"Part {n} of {len(pieces)}"
            out.append({"start": a, "end": b, "path": path, "title": name})
    return out


def _prose(body: str) -> tuple[str, float]:
    """*body* without fenced code, and the share of its lines that are
    links, pictures, tables or code."""
    kept, noise, total, fenced = [], 0, 0, False
    for line in body.splitlines():
        if not line.strip():
            continue
        total += 1
        if _FENCE.match(line):
            fenced = not fenced
            noise += 1
        elif fenced or _NOISE_LINE.match(line) or line.lstrip().startswith("|"):
            noise += 1
        else:
            kept.append(line)
    return "\n".join(kept), (noise / total if total else 1.0)


def instructive(body: str, title: str = "") -> float:
    """How much *body* tells an agent what to do or know exactly — rule
    words, numbered steps, definitions, formulas, amounts with a unit — per
    thousand characters of its prose, so a long section doesn't win by
    size. Code, tables, link lists and changelog entries count for little;
    so does a section with hardly anything in it."""
    prose, noise = _prose(body)
    size = len(prose)
    if size < MIN_UNIT_CHARS // 2:
        return 0.0
    hits = (2.0 * len(_RULE_WORDS.findall(prose)) + 2.0 * len(_STEP_LINE.findall(prose))
            + 1.5 * len(_DEFINITION.findall(prose)) + 1.5 * len(_FORMULA.findall(prose))
            + 1.0 * len(_AMOUNT.findall(prose)))
    score = hits * 1000.0 / size * (1.0 - 0.8 * noise) * min(1.0, size / 1500.0)
    if _BOILERPLATE_TITLE.search(title.strip()):
        score *= 0.2
    return score


def focus_scores(texts: list[str], focus: str) -> list[float]:
    """How close each of *texts* is to *focus* (0..1): by the local
    embeddings, else by the share of its words they hold."""
    if not texts or not focus.strip():
        return [0.0] * len(texts)
    try:
        vecs = poutline._encode([focus, *texts])
        return [max(0.0, float(s)) for s in vecs[1:] @ vecs[0]]
    except Exception:
        log.warning("agent preset: no local embeddings for the focus — matching its words instead", exc_info=True)
    words = set(re.findall(r"\w{4,}", focus.lower()))
    if not words:
        return [0.0] * len(texts)
    return [len(words & set(re.findall(r"\w{4,}", t.lower()))) / len(words) for t in texts]


def _ranks(values: list[float]) -> list[float]:
    """Each of *values* as its rank among them, 0 (the lowest) to 1 — so two
    scores of different kinds can be weighed together, and an outlier
    flattens nothing. Equal values share a rank."""
    if len(values) < 2:
        return [1.0] * len(values)
    order = sorted(set(values))
    at = {v: n / max(1, len(order) - 1) for n, v in enumerate(order)}
    return [at[v] for v in values]


def select(docs: list[dict], budget_chars: int, focus: str = "",
           max_chars: int = UNIT_MAX_CHARS) -> tuple[list[dict], list[dict]]:
    """What to read of *docs* (each ``text``, ``name``, ``usage``) within
    *budget_chars*: their sections scored, every document first given its
    best ones up to a share of the budget, then the best of all the rest.
    Returns the chosen units — each with ``doc`` (its index in *docs*) and
    ``score`` — in document order, and per document what that is:
    ``sections`` read of ``of``, and its ``chars``."""
    every: list[dict] = []
    for i, d in enumerate(docs):
        for u in units(d["text"], max_chars):
            body = d["text"][u["start"]:u["end"]]
            if len(body.strip()) >= MIN_UNIT_CHARS:
                every.append({**u, "doc": i, "chars": u["end"] - u["start"],
                              "raw": instructive(body, u["title"]) * _USAGE_WEIGHT.get(d.get("usage") or "", 1.0)})
    scores = _ranks([u["raw"] for u in every])
    if focus.strip():
        near = _ranks(focus_scores([" › ".join([*u["path"], u["title"]]) + "\n"
                                     + docs[u["doc"]]["text"][u["start"]:min(u["end"], u["start"] + _FOCUS_CHARS)]
                                     for u in every], focus))
        scores = [(1 - FOCUS_WEIGHT) * s + FOCUS_WEIGHT * n for s, n in zip(scores, near)]
    for u, s in zip(every, scores):
        u["score"] = s
    ranked = sorted(every, key=lambda u: (-u["score"], u["doc"], u["start"]))

    chosen: set[int] = set()
    left = budget_chars
    share = budget_chars // (2 * max(1, len(docs)))
    for i in range(len(docs)):
        room = share
        for u in ranked:
            # A document's best section that fits goes in, even when it is more than its share.
            if u["doc"] == i and u["chars"] <= left and (room == share or u["chars"] <= room):
                chosen.add(id(u))
                room -= u["chars"]
                left -= u["chars"]
    for u in ranked:
        if id(u) not in chosen and u["chars"] <= left:
            chosen.add(id(u))
            left -= u["chars"]
    picked = sorted((u for u in every if id(u) in chosen), key=lambda u: (u["doc"], u["start"]))
    summary = [{"sections": sum(1 for u in picked if u["doc"] == i),
                "of": sum(1 for u in every if u["doc"] == i),
                "chars": sum(u["chars"] for u in picked if u["doc"] == i)} for i in range(len(docs))]
    for u in picked:
        u.pop("raw", None)
    return picked, summary
