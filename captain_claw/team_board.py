"""The Team up board — how helpers talk while they work (the Vatra protocol, in memory).

docs/team-up-plan.md, P1. One board per team (team_up.Team.board), in this
process, no Flight Deck round trip:

* **post** — a finding, a decision or a draft others may build on. A post
  that starts with "remember:" is a note for the agent's memory; the lead
  decides on it after the team (helpers never write memory themselves).
* **ask** — a request to one helper, to ``anyone`` or to the ``lead``. It
  doesn't block: the asker goes on working. An ask to a helper shows in that
  helper's next step ("New for you"); an ask to the lead (or to a helper that
  has finished) is answered by the lead at once, quietly, from what it knows.
* **answer** — the reply to an ask, shown to the asker at its next step.
* **inbox / wait** — the asker can block a while for answers, or for another
  helper's result or post when its part truly depends on it.

Entries are small dicts, so they go to the chat as they are.
"""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass, field
from typing import Any

KINDS = ("post", "ask", "answer", "remember")
MAX_ENTRIES = 300
MAX_ASKS = 12
MAX_POSTS_PER_AUTHOR = 40
TEXT_CHARS = 4000
TITLE_CHARS = 120
WAIT_MAX = 120
INBOX_WAIT_MAX = 30
LEAD = "lead"
ANYONE = "anyone"

_REMEMBER_RE = re.compile(r"^\s*remember\s*:\s*", re.IGNORECASE)
_WORD_RE = re.compile(r"[\w€$£%.-]+", re.UNICODE)


def _clean(text: Any, limit: int) -> str:
    value = str(text or "").strip()
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value[:limit].rstrip()


@dataclass
class Board:
    entries: list[dict[str, Any]] = field(default_factory=list)
    next_id: int = 1
    asks: int = 0
    _event: asyncio.Event | None = None

    # ── changes ──────────────────────────────────────────────────────

    def _pulse(self) -> None:
        """Wake everyone waiting on the board."""
        if self._event is not None:
            self._event.set()
        self._event = None

    async def changed(self, timeout: float) -> bool:
        """Wait until the board changes (True) or *timeout* passes (False)."""
        if self._event is None:
            self._event = asyncio.Event()
        try:
            await asyncio.wait_for(self._event.wait(), timeout=max(0.0, timeout))
            return True
        except asyncio.TimeoutError:
            return False

    def add(self, kind: str, who: str, who_name: str, text: str, *, title: str = "", to: str = "",
            to_name: str = "", ask_id: int = 0) -> dict[str, Any]:
        entry = {
            "id": self.next_id,
            "t": round(time.time(), 2),
            "kind": kind,
            "who": who,
            "who_name": who_name,
            "title": _clean(title, TITLE_CHARS),
            "text": _clean(text, TEXT_CHARS),
            **({"to": to, "to_name": to_name} if to else {}),
            **({"ask_id": ask_id} if ask_id else {}),
            **({"answered": False} if kind == "ask" else {}),
        }
        self.next_id += 1
        self.entries.append(entry)
        if kind == "ask":
            self.asks += 1
        if ask_id:
            for e in self.entries:
                if e["id"] == ask_id and e["kind"] == "ask":
                    e["answered"] = True
                    break
        if len(self.entries) > MAX_ENTRIES:
            del self.entries[: len(self.entries) - MAX_ENTRIES]
        self._pulse()
        return entry

    # ── reading ──────────────────────────────────────────────────────

    def get(self, entry_id: int) -> dict[str, Any] | None:
        for e in self.entries:
            if e["id"] == entry_id:
                return e
        return None

    def posts_by(self, who: str) -> int:
        return sum(1 for e in self.entries if e["who"] == who and e["kind"] in ("post", "remember"))

    def recent(self, *, who: str = "", kind: str = "", limit: int = 15) -> list[dict[str, Any]]:
        out = [e for e in self.entries
               if (not who or e["who"] == who) and (not kind or e["kind"] == kind)]
        return out[-limit:]

    def search(self, query: str, *, limit: int = 8, exclude_who: str = "") -> list[dict[str, Any]]:
        terms = [t for t in _WORD_RE.findall(str(query or "").lower()) if len(t) > 1]
        if not terms:
            return []
        scored: list[tuple[int, int, dict[str, Any]]] = []
        for e in self.entries:
            if exclude_who and e["who"] == exclude_who:
                continue
            hay = f"{e.get('title', '')} {e.get('text', '')}".lower()
            score = sum(hay.count(t) for t in terms) + 3 * sum(1 for t in terms if t in e.get("title", "").lower())
            if score:
                scored.append((score, e["id"], e))
        scored.sort(key=lambda x: (-x[0], -x[1]))
        return [e for _s, _i, e in scored[:limit]]

    def open_asks_for(self, helper_id: str) -> list[dict[str, Any]]:
        """Unanswered asks this helper should see: to it, or to anyone (not its own)."""
        return [e for e in self.entries if e["kind"] == "ask" and not e.get("answered")
                and e["who"] != helper_id and e.get("to") in (helper_id, ANYONE)]

    def asks_by(self, helper_id: str) -> list[dict[str, Any]]:
        return [e for e in self.entries if e["kind"] == "ask" and e["who"] == helper_id]

    def answers_to(self, ask_id: int) -> list[dict[str, Any]]:
        return [e for e in self.entries if e["kind"] == "answer" and e.get("ask_id") == ask_id]

    def remember_notes(self) -> list[dict[str, Any]]:
        return [e for e in self.entries if e["kind"] == "remember"]

    def same_open_ask(self, who: str, to: str, text: str) -> dict[str, Any] | None:
        key = re.sub(r"\W+", " ", text.lower()).strip()
        for e in self.entries:
            if (e["kind"] == "ask" and not e.get("answered") and e["who"] == who and e.get("to") == to
                    and re.sub(r"\W+", " ", e["text"].lower()).strip() == key):
                return e
        return None

    def snapshot(self, limit: int = 120) -> list[dict[str, Any]]:
        return [dict(e) for e in self.entries[-limit:]]


def is_remember(text: str) -> bool:
    return bool(_REMEMBER_RE.match(str(text or "")))


def strip_remember(text: str) -> str:
    return _REMEMBER_RE.sub("", str(text or ""), count=1).strip()


def fmt_entry(e: dict[str, Any], *, full: bool = True, chars: int = 600) -> str:
    """One entry as a helper or the lead reads it."""
    text = e.get("text", "")
    if not full and len(text) > chars:
        text = text[:chars].rstrip() + " …"
    who = e.get("who_name") or e.get("who")
    kind = e.get("kind")
    if kind == "ask":
        state = "answered" if e.get("answered") else "open"
        return f"#{e['id']} {who} asks {e.get('to_name') or e.get('to')} ({state}): {text}"
    if kind == "answer":
        return f"#{e['id']} {who} answers #{e.get('ask_id')}: {text}"
    if kind == "remember":
        return f"#{e['id']} {who} — remember: {text}"
    title = f" «{e['title']}»" if e.get("title") else ""
    return f"#{e['id']} {who} posted{title}: {text}"


def digest(board: Board, *, limit: int = 12, chars: int = 300) -> str:
    rows = [fmt_entry(e, full=False, chars=chars) for e in board.entries[-limit:]]
    if not rows:
        return "The board is empty."
    more = len(board.entries) - len(rows)
    head = f"(… {more} earlier)\n" if more > 0 else ""
    return head + "\n".join(rows)
