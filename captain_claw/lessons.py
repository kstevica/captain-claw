"""What the agent's owner taught it — its lessons — brought in with the
messages they bear on.

Flight Deck writes them (flight_deck/agent_lessons.py) to
``~/.captain-claw/lessons.json``: rules, things to ask the owner first, and
facts, each with where it came from. Only those marked ``always`` are in
the agent's instructions; the rest would be lost in a long list, so each
turn gets the few its message is about (:meth:`LessonBook.relevant`): by its
words (FTS, as insights are matched) and by meaning (the agent's local
embedding model, when it has one), fused by rank. A lesson already in the
instructions is never repeated — an agent set up before lessons keeps its
rules there, and gains the ones its instructions had no room for.

The file is read again when it changes; the agent never writes it.
"""

from __future__ import annotations

import json
import math
import sqlite3
import threading
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from captain_claw import retrieval
from captain_claw.logging import get_logger

log = get_logger(__name__)

CAPABILITY = "owner_lessons"      # /api/version: Flight Deck writes this agent's rules as lessons
PATH = Path("~/.captain-claw/lessons.json")
MAX_ITEMS = 6                     # lessons a turn gets, at most
MAX_CHARS = 1800                  # … and their characters
_CANDIDATES = 30                  # per ranking leg, before fusion
_COSINE_FLOOR = 0.35              # a meaning match counts only above this cosine …
_COSINE_REACH = 0.8               # … and within this share of the best one
_LABELS = {"rule": "rule", "ask_first": "ask your owner first", "fact": "fact"}

Embed = Callable[[list[str]], Sequence[Sequence[float]]]


def _flat(text: str) -> str:
    return " ".join(str(text or "").split()).casefold()


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


class LessonBook:
    """The lessons file, read when it changes, with a word index and the
    lessons' vectors (made on first need, kept by lesson)."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = (path or PATH).expanduser()
        self.lessons: list[dict[str, Any]] = []
        self._sig: tuple[int, int] | None = None
        self._db: sqlite3.Connection | None = None
        self._vectors: dict[str, Sequence[float]] = {}
        self._lock = threading.Lock()

    def _load(self) -> None:
        try:
            st = self.path.stat()
        except OSError:
            self.lessons, self._sig, self._db = [], None, None
            return
        sig = (st.st_mtime_ns, st.st_size)
        if sig == self._sig:
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            items = data.get("lessons") if isinstance(data, dict) else None
        except (OSError, ValueError):
            log.warning("Lessons unreadable", path=str(self.path))
            items = None
        self.lessons = [x for x in items or [] if isinstance(x, dict) and str(x.get("text") or "").strip()
                        and x.get("id")]
        db = sqlite3.connect(":memory:", check_same_thread=False)
        db.execute("CREATE VIRTUAL TABLE lx USING fts5(text, section, doc)")
        db.executemany("INSERT INTO lx(rowid, text, section, doc) VALUES (?, ?, ?, ?)",
                       [(n, x["text"], str((x.get("source") or {}).get("section") or ""),
                         str((x.get("source") or {}).get("doc") or ""))
                        for n, x in enumerate(self.lessons)])
        self._db = db
        self._vectors = {k: v for k, v in self._vectors.items() if k in {x["id"] for x in self.lessons}}
        self._sig = sig
        log.info("Lessons loaded", lessons=len(self.lessons))

    def _by_words(self, query: str) -> list[int]:
        terms = retrieval.query_terms(query, max_terms=12)
        if not terms or self._db is None:
            return []
        try:
            rows = self._db.execute(
                "SELECT rowid, text, section FROM lx WHERE lx MATCH ? ORDER BY bm25(lx) LIMIT ?",
                ("{text section} : (" + retrieval.fts_match(terms) + ")", _CANDIDATES)).fetchall()
        except sqlite3.Error as exc:
            log.debug("Lessons word search failed", error=str(exc))
            return []
        rare = max(1, int(len(self.lessons) * 0.02))
        out: list[int] = []
        for rowid, text, section in rows:
            hits = retrieval.matched_terms(terms, text, section)
            # Two words of the message, its only word, or one word few lessons have.
            if len(hits) >= 2 or (len(terms) == 1 and hits) or (len(hits) == 1 and self._rows_with(hits[0]) <= rare):
                out.append(int(rowid))
        return out

    def _rows_with(self, term: str) -> int:
        word = retrieval.fold(term)
        return sum(1 for x in self.lessons if word in retrieval.fold(x["text"]))

    def _by_meaning(self, query: str, embed: Embed | None) -> list[int]:
        if embed is None or not self.lessons:
            return []
        try:
            todo = [x for x in self.lessons if x["id"] not in self._vectors]
            if todo:
                for x, vec in zip(todo, embed([x["text"] for x in todo]), strict=True):
                    self._vectors[x["id"]] = vec
            (qv,) = embed([query])
        except Exception as exc:
            log.debug("Lessons meaning search unavailable", error=str(exc))
            return []
        scored = sorted(((_cosine(qv, self._vectors[x["id"]]), n) for n, x in enumerate(self.lessons)),
                        reverse=True)[:_CANDIDATES]
        best = scored[0][0] if scored else 0.0
        return [n for score, n in scored if score >= _COSINE_FLOOR and score >= best * _COSINE_REACH]

    def relevant(self, query: str, *, instructions: str = "", embed: Embed | None = None,
                 limit: int = MAX_ITEMS, budget: int = MAX_CHARS) -> list[dict[str, Any]]:
        """The lessons *query* bears on, best first — none already in
        *instructions* — within *limit* items and *budget* characters."""
        with self._lock:
            self._load()
            if not self.lessons or not query.strip():
                return []
            ranks = retrieval.rrf([[str(n) for n in self._by_words(query)],
                                   [str(n) for n in self._by_meaning(query, embed)]])
            given = _flat(instructions)
            out: list[dict[str, Any]] = []
            used = 0
            for key, _score in sorted(ranks.items(), key=lambda kv: -kv[1]):
                x = self.lessons[int(key)]
                if given and _flat(x["text"]) in given:
                    continue
                if used + len(x["text"]) > budget and out:
                    break
                out.append(x)
                used += len(x["text"])
                if len(out) >= limit:
                    break
            return out


def note(lessons: list[dict[str, Any]]) -> str:
    """The lessons as a turn's context note."""
    if not lessons:
        return ""
    lines = ["Your owner's lessons that bear on this message — rules they set and facts they taught you. "
             "Follow them as you follow your instructions; where one names its document and section, read that "
             "section (knowledge tool) when the details matter:"]
    for x in lessons:
        src = x.get("source") or {}
        where = f" (from {src['doc']}{' — ' + src['section'] if src.get('section') else ''})" if src.get("doc") else ""
        lines.append(f"- [{_LABELS.get(str(x.get('kind')), 'rule')}] {x['text']}{where}")
    return "\n".join(lines)


_BOOK: LessonBook | None = None
_BOOK_LOCK = threading.Lock()


def get_book() -> LessonBook:
    global _BOOK
    with _BOOK_LOCK:
        if _BOOK is None:
            _BOOK = LessonBook()
        return _BOOK

