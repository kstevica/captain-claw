"""Captain Swarm — Flight Deck's side of a rehearsal: intents kept, and proof afterwards.

docs/captain-swarm-plan.md §4.1 and §4.6. A rehearsal worker never acts on the
world (``captain_claw/rehearsal_turn.py``): what it would have done reaches
Flight Deck as an *intent* (an outward call recorded instead of made) or a
*commit point* (the browser stopped before submitting, buying, sending…).
They are kept with the experiment, so the winner can be carried out once,
later, through Bat's gate.

Trusting the guard isn't enough. After an attempt, :func:`leaks` reads the
tool-call register — every call Flight Deck's agents report, HMAC-chained —
for that attempt's workers, and re-judges each call with the guard's own rules:
one that should have been stopped but ran ``ok`` is a leak, and the attempt is
disqualified. A broken guard is a bug to fix, never a result.
"""

from __future__ import annotations

import json
import re
from typing import Any

from captain_claw.logging import get_logger

log = get_logger(__name__)

TAG_RE = re.compile(r"^(cs_[0-9a-f]{6,32}):(exp_[0-9a-f]{6,32})$")
_MAX_PER_ATTEMPT = 300


def parse_tag(tag: str) -> tuple[str, str] | None:
    m = TAG_RE.match(str(tag or "").strip())
    return (m.group(1), m.group(2)) if m else None


def make_tag(run_id: str, exp_id: str) -> str:
    tag = f"{run_id}:{exp_id}"
    if not TAG_RE.match(tag):
        raise ValueError("not a rehearsal tag")
    return tag


async def record(store: Any, owner: str, *, tag: str, kind: str, tool: str, what: str, args: dict,
                 agent: str = "") -> dict:
    """Keep one intent for the attempt *tag* names, if it is *owner*'s and under way."""
    parsed = parse_tag(tag)
    if parsed is None or kind not in ("intent", "commit"):
        return {"ok": False, "why": "not a rehearsal"}
    run_id, exp_id = parsed
    run = await store.get_run(run_id)
    exp = await store.get_experiment(exp_id)
    if not run or run["owner_id"] != owner or not exp or exp["run_id"] != run_id:
        return {"ok": False, "why": "no such attempt"}
    if exp["status"] not in ("proposed", "running"):
        return {"ok": False, "why": "that attempt is over"}
    if await store.count_intents(run_id, exp_id) >= _MAX_PER_ATTEMPT:
        return {"ok": False, "why": "too many intents for one attempt"}
    iid = await store.add_intent(run_id=run_id, exp_id=exp_id, kind=kind, tool=str(tool)[:80], what=str(what)[:500],
                                 args=args if isinstance(args, dict) else {}, agent=agent)
    return {"ok": True, "id": iid}


def _params(raw: Any) -> dict:
    if isinstance(raw, dict):
        return raw
    try:
        data = json.loads(raw or "{}")
    except (TypeError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def judge_rows(rows: list[dict]) -> list[dict]:
    """The calls among *rows* that a rehearsal must not have made, but that ran."""
    from captain_claw import rehearsal_turn
    out = []
    for r in rows:
        if str(r.get("outcome") or "") != "ok":
            continue
        verdict, what = rehearsal_turn.classify(str(r.get("tool") or ""), _params(r.get("params")))
        if verdict != "allow":
            out.append({"agent": r.get("agent_slug"), "tool": r.get("tool"), "what": what or r.get("tool"),
                        "verdict": verdict, "at": r.get("started_at")})
    return out


async def leaks(slug_prefix: str, since: float, until: float | None = None) -> list[dict]:
    """Outward calls the attempt's workers made anyway between *since* and
    *until*, by the tool-call register. Workers are found by name: a Bat run's
    are all ``bat-<its id>…``, and no one else's are."""
    from captain_claw.flight_deck import tool_audit_routes
    if not slug_prefix:
        raise ValueError("whose calls?")
    try:
        audit = tool_audit_routes.store()
    except Exception:
        log.warning("captain swarm: the tool-call register isn't open; can't check a rehearsal")
        raise
    filters: dict[str, Any] = {"since": since}
    if until:
        filters["until"] = until
    rows = [r async for r in audit.iterate(filters) if str(r.get("agent_slug") or "").startswith(slug_prefix)]
    return judge_rows(rows)
