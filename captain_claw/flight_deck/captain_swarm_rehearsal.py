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

One attempt per site and per account (§4.4, Phase 4). Attempts that run at
once must not look like a crowd of bots to one website, and two sign-ins to
one account set off its alarms. Before a worker's browser opens a page, or
signs in with stored credentials, the guard asks :func:`take`: a site or an
account another attempt is using in its browser is refused, deck-wide, until
that attempt ends or leaves it alone for ten minutes. Plain page reads
(``web_fetch``) aren't held: they are what any visitor does.
"""

from __future__ import annotations

import json
import re
import time
import urllib.parse
from typing import Any

from captain_claw.logging import get_logger

log = get_logger(__name__)

TAG_RE = re.compile(r"^(cs_[0-9a-f]{6,32}):(exp_[0-9a-f]{6,32})$")
_MAX_PER_ATTEMPT = 300
_SITE_IDLE_S = 600.0
# Second-level labels under which a registrable name has three parts (shop.co.uk).
_TWO_PART = frozenset({"co", "com", "org", "net", "ac", "gov", "edu", "ne", "or", "gv"})
# Who is on which site or account in their browser: key → {"<run>:<exp>": last use}.
# In memory on purpose: a restart forgets, and the next look takes the site again.
_SITES: dict[str, dict[str, float]] = {}


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


def per_site() -> int:
    """Attempts that may use one site or account at once: FD_CAPTAIN_SWARM_PER_SITE, 1."""
    from captain_claw.flight_deck.captain_swarm_objective import env_int
    return env_int("FD_CAPTAIN_SWARM_PER_SITE", 1, 1, 4)


def site_of(raw: str) -> str:
    """The site a URL or a domain belongs to — ``shop.berrybase.de`` and
    ``berrybase.de`` are one — or "" for anything that isn't a public name."""
    s = str(raw or "").strip().lower()
    if not s:
        return ""
    try:
        host = urllib.parse.urlsplit(s if "://" in s else f"http://{s}").hostname or ""
    except ValueError:
        return ""
    labels = [x for x in host.split(".") if x]
    if len(labels) < 2 or not all(re.fullmatch(r"[a-z0-9-]{1,63}", x) for x in labels) or labels[-1].isdigit():
        return ""
    if len(labels) >= 3 and labels[-2] in _TWO_PART and len(labels[-1]) == 2:
        return ".".join(labels[-3:])
    return ".".join(labels[-2:])


def take_site(tag: str, key: str, *, now: float | None = None) -> str:
    """Let attempt *tag* use *key* (a site, or ``account:<name>``): "" when it
    may, else the attempt that has it."""
    now = time.time() if now is None else now
    holders = {h: t for h, t in (_SITES.get(key) or {}).items() if now - t < _SITE_IDLE_S}
    if tag in holders or len(holders) < per_site():
        holders[tag] = now
        _SITES[key] = holders
        return ""
    _SITES[key] = holders
    return next(iter(holders))


def release_sites(run_id: str, exp_id: str = "") -> None:
    """The attempt (or the whole run) is over: its sites and accounts are free."""
    for key in list(_SITES):
        left = {h: t for h, t in _SITES[key].items()
                if not (h == f"{run_id}:{exp_id}" if exp_id else h.startswith(f"{run_id}:"))}
        if left:
            _SITES[key] = left
        else:
            del _SITES[key]


def sites_held(run_id: str, *, now: float | None = None) -> dict[str, list[str]]:
    """Experiment → the sites and accounts it is on now, for the page."""
    now = time.time() if now is None else now
    out: dict[str, list[str]] = {}
    for key, holders in _SITES.items():
        for h, t in holders.items():
            rid, _, exp_id = h.partition(":")
            if rid == run_id and now - t < _SITE_IDLE_S:
                out.setdefault(exp_id, []).append(key)
    return {k: sorted(v) for k, v in out.items()}


async def take(store: Any, owner: str, *, tag: str, url: str = "", account: str = "") -> dict:
    """A rehearsal worker asks before its browser opens *url* or signs in to
    *account*. Refused when another attempt is on that site or account now, or
    when the attempt isn't *owner*'s and under way."""
    parsed = parse_tag(tag)
    if parsed is None:
        return {"ok": False, "why": "not a rehearsal"}
    run_id, exp_id = parsed
    run = await store.get_run(run_id)
    exp = await store.get_experiment(exp_id)
    if not run or run["owner_id"] != owner or not exp or exp["run_id"] != run_id:
        return {"ok": False, "why": "no such attempt"}
    if exp["status"] not in ("proposed", "running"):
        return {"ok": False, "why": "that attempt is over"}
    name = " ".join(str(account or "").lower().split())[:80]
    key = f"account:{name}" if name else site_of(url)
    if not key:
        return {"ok": True, "site": ""}      # about:blank, a local file: no one else's site
    holder = take_site(tag, key)
    if not holder:
        return {"ok": True, "site": key}
    other = await store.get_experiment(holder.partition(":")[2])
    who = (f"attempt #{other['round']}.{other['n']}" if other and other["run_id"] == run_id
           else "an attempt of another run")
    what = f"the account {name!r}" if name else key
    return {"ok": False, "site": key, "why": f"{who} is using {what} in its browser right now"}


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
