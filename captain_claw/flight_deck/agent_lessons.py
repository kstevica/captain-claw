"""An agent's lessons — what its owner taught it, one item each.

Onboarding's unit: a rule ("Always answer in Croatian"), something to ask
the owner first ("Ask before refunding more than 200 EUR"), or a fact ("The
refund window is 14 days"), each with where it came from (a document and
section) and who taught it (a setup, an exam's "Teach it this"). The
agent's instructions hold only the few marked ``always`` (they apply to
every conversation); the rest reach it with the messages they bear on.

Flight Deck keeps them, in the agent's setup record (``.owner-setup.json``,
agent_preset): rules and when-to-ask as the setup's groups, facts and taught
items under ``lessons``. It writes them for the agent to read
(``lessons.json`` in its home — beside its skills), which it can't change,
and which is there while it's off: the agent (captain_claw/lessons.py)
picks those a message bears on, by its words and by meaning, and skips any
already in its instructions — so an agent set up before lessons gains the
rules its instructions had no room for, without them being rewritten.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from captain_claw.logging import get_logger

log = get_logger(__name__)

FILE = "lessons.json"
CAPABILITY = "owner_lessons"     # /api/version: the agent reads lessons.json (captain_claw/lessons.py)
KINDS = ("rule", "ask_first", "fact")
_RECORD = ".owner-setup.json"
_TEXT_MAX = 600


def path(agent_dir: Path) -> Path:
    """Where the agent reads its lessons: ``~/.captain-claw`` under its HOME."""
    return agent_dir / "data" / "home-config-parent" / ".captain-claw" / FILE


def lesson_id(kind: str, text: str) -> str:
    return "l" + hashlib.sha1(f"{kind}|{' '.join(text.split()).casefold()}".encode()).hexdigest()[:12]


def _source(raw: Any) -> dict:
    src = raw if isinstance(raw, dict) else {}
    doc = "" if src.get("doc") in (None, "instructions") else str(src.get("doc") or "")
    return {"doc": doc, "section": str(src.get("section") or "") if doc else ""}


def _lesson(kind: str, text: Any, source: Any = None, *, always: bool = False, taught_by: str = "setup",
            importance: int = 5, added: float = 0.0) -> dict | None:
    body = " ".join(str(text or "").split())[:_TEXT_MAX]
    if not body:
        return None
    return {"id": lesson_id(kind, body), "kind": kind, "text": body, "source": _source(source),
            "always": bool(always), "taught_by": taught_by, "importance": int(importance), "added": added}


def from_record(record: dict) -> list[dict]:
    """Every lesson a setup record holds: its rules and when-to-ask, then
    its facts and what was taught since — each once."""
    out: list[dict] = []
    seen: set[str] = set()

    def keep(item: dict | None) -> None:
        if item and item["id"] not in seen:
            seen.add(item["id"])
            out.append(item)

    for group, kind in (("rules", "rule"), ("escalation", "ask_first")):
        for r in record.get(group) or []:
            if isinstance(r, dict):
                keep(_lesson(kind, r.get("text"), r.get("source"),
                             always=r.get("always") is True if kind == "rule" else r.get("always") is not False,
                             taught_by=str(r.get("taught_by") or "setup"), added=float(r.get("added") or 0.0)))
    for x in record.get("lessons") or []:
        if isinstance(x, dict) and x.get("kind") in KINDS:
            keep(_lesson(x["kind"], x.get("text"), x.get("source"), always=x.get("always") is True,
                         taught_by=str(x.get("taught_by") or "setup"), importance=int(x.get("importance") or 5),
                         added=float(x.get("added") or 0.0)))
    return out


def write(agent_dir: Path, record: dict) -> int:
    """Write the agent's lessons from its setup record; how many."""
    lessons = from_record(record)
    target = path(agent_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{FILE}.tmp")
    tmp.write_text(json.dumps({"version": 1, "updated": time.time(), "lessons": lessons}, ensure_ascii=False),
                   encoding="utf-8")
    tmp.replace(target)
    return len(lessons)


def ensure(agent_dir: Path) -> bool:
    """The agent's lessons written, when its setup record is newer than them
    (an agent set up before lessons, a record changed while Flight Deck was
    down). True when written."""
    record_path, target = agent_dir / _RECORD, path(agent_dir)
    try:
        if not record_path.is_file():
            return False
        if target.is_file() and target.stat().st_mtime >= record_path.stat().st_mtime:
            return False
        record = json.loads(record_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if not isinstance(record, dict):
        return False
    n = write(agent_dir, record)
    log.info("agent lessons written", agent=agent_dir.name, lessons=n)
    return True


def ensure_all(data_dir: Path) -> int:
    """Every agent's lessons up to date with its setup record; how many were written."""
    written = 0
    for agent_dir in data_dir.iterdir() if data_dir.is_dir() else []:
        try:
            written += bool(agent_dir.is_dir() and ensure(agent_dir))
        except Exception:
            log.warning("agent lessons: couldn't write them", agent=agent_dir.name, exc_info=True)
    return written


def taught(record: dict, kind: str, text: str, source: Any = None, *, taught_by: str,
           always: bool = False, importance: int = 8) -> tuple[dict, bool]:
    """*record* with one lesson more under ``lessons`` (a fact a setup read,
    an answer an exam taught); False when it already has it."""
    item = _lesson(kind, text, source, always=always, taught_by=taught_by, importance=importance, added=time.time())
    if item is None or item["id"] in {x["id"] for x in from_record(record)}:
        return record, False
    return {**record, "lessons": [*(record.get("lessons") or []), item]}, True


def facts_as_lessons(facts: list[dict]) -> list[dict]:
    """A setup's facts as lessons (they used to become the agent's insights)."""
    now = time.time()
    return [x for x in (_lesson("fact", f.get("content"), f.get("source"), importance=int(f.get("importance") or 5),
                                added=now) for f in facts if isinstance(f, dict)) if x]


def counts(record: dict) -> dict[str, int]:
    lessons = from_record(record)
    out = {k: sum(1 for x in lessons if x["kind"] == k) for k in KINDS}
    out["always"] = sum(1 for x in lessons if x["always"])
    return out


async def capabilities(slug: str) -> list[str] | None:
    """What a running agent says it can do (``/api/version``); None when it
    isn't running or doesn't answer."""
    import httpx

    from captain_claw.flight_deck import server as srv

    entry = srv._load_process_registry().get(slug) or {}
    port, token = int(entry.get("web_port") or 0), str(entry.get("web_auth") or "")
    if not port or not srv._process_is_alive(slug):
        return None
    try:
        async with httpx.AsyncClient(timeout=10.0, trust_env=False) as client:
            resp = await client.get(f"http://127.0.0.1:{port}/api/version", params={"token": token} if token else {})
        caps = resp.json().get("capabilities") if resp.status_code == 200 else None
    except (httpx.HTTPError, ValueError, AttributeError):
        return None
    return [str(c) for c in caps] if isinstance(caps, list) else None
