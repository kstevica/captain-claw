"""Agent family: who made which agent, and where it sits now.

An agent that another agent made is that agent's *child*; agents under the
same parent are *siblings*. The owner can also move one of their agents under
another by hand. See docs/agent-family-plan.md.

The fields live on the process registry entry (server.PROCESS_REGISTRY_FILE):

``parent``         slug of the agent it sits under now; "" = one of its owner's own
``made_by``        slug of the agent that created it, ``first_mate``, or "" (its owner)
``made_for``       one line on why it exists
``role``           ``standing`` (stays) or ``worker`` (a run's, torn down after it)
``made_at``        when it was first made (ISO, UTC)
``parent_set_by``  ``spawn`` (its maker) or ``person`` (moved by hand)
``run``            a worker's run: ``bat:<id>``, ``vatra:<id>``
``factory``        a First Mate factory worker's factory id

``made_by`` never changes; ``parent`` is where the agent sits now. Entries made
before this existed have none of the fields: they are top-level, standing
unless their slug is a run worker's (``agent_sharing.is_managed_agent``).

Docker agents are not part of a family yet (a running container's labels
can't change): they stay top-level.

The helpers here are pure over a registry dict; the caller loads it, holds
the spawn lock while it changes it, and saves it.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

STANDING = "standing"
WORKER = "worker"
FIRST_MATE = "first_mate"

# Levels below an owner's own (top-level) agent: child 1, grandchild 2, …
MAX_DEPTH = 3
# Standing agents under one parent (Forge's MAX_TEAM).
MAX_CHILDREN = 12
MADE_FOR_MAX = 200

FIELDS = ("parent", "made_by", "made_for", "role", "made_at", "parent_set_by", "run", "factory")

# The env every runner stamps on its workers (bat_routes / vatra_routes /
# council_routes _WORKER_MARKER, Basna's, Code mode's).
_WORKER_ENV = frozenset({
    "CLAW_BAT_WORKER", "CLAW_VATRA_WORKER", "CLAW_BASNA_WORKER", "CLAW_COUNCIL_WORKER",
    "CLAW_CODE_AGENT",
})
_RUN_ENV = (("CLAW_BAT_SESSION", "bat"), ("CLAW_VATRA_SESSION", "vatra"))


class FamilyError(Exception):
    """A family change that isn't allowed; ``reason`` is said to the person."""

    def __init__(self, reason: str, status: int = 409) -> None:
        super().__init__(reason)
        self.reason = reason
        self.status = status


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def clean_made_for(text: Any) -> str:
    return " ".join(str(text or "").split())[:MADE_FOR_MAX]


def _entry(registry: dict, slug: str) -> dict | None:
    e = registry.get(slug) if slug else None
    return e if isinstance(e, dict) else None


def _name(registry: dict, slug: str) -> str:
    e = _entry(registry, slug) or {}
    return str(e.get("name") or slug)


def _same_owner(a: dict, b: dict) -> bool:
    """One owner's agents. A deck with sign-in off has a single person, whose
    agents may carry "", "local" or an id from different times."""
    if str(a.get("owner") or "") == str(b.get("owner") or ""):
        return True
    from captain_claw.flight_deck.auth import _fd_auth_enabled

    return not _fd_auth_enabled()


def role_of(slug: str, entry: dict) -> str:
    role = entry.get("role")
    if role in (STANDING, WORKER):
        return role
    from captain_claw.flight_deck.agent_sharing import is_managed_agent

    return WORKER if is_managed_agent(slug, str(entry.get("description") or "")) else STANDING


def parent_of(registry: dict, slug: str) -> str:
    """Where ``slug`` sits now: its parent's slug, or "" when it is top-level —
    also when the recorded parent is gone or belongs to someone else."""
    e = _entry(registry, slug)
    if e is None:
        return ""
    p = str(e.get("parent") or "")
    pe = _entry(registry, p)
    if not p or p == slug or pe is None:
        return ""
    if not _same_owner(pe, e):
        return ""
    return p


def children(registry: dict, slug: str, *, workers: bool = False) -> list[str]:
    """The agents right under ``slug`` (standing ones, plus workers if asked)."""
    out = []
    for s, e in registry.items():
        if s == slug or not isinstance(e, dict) or parent_of(registry, s) != slug:
            continue
        if workers or role_of(s, e) == STANDING:
            out.append(s)
    return out


def ancestors(registry: dict, slug: str) -> list[str]:
    """Parent, grandparent, … (nearest first). Stops at a loop, should one exist."""
    out: list[str] = []
    seen = {slug}
    cur = parent_of(registry, slug)
    while cur and cur not in seen and len(out) < 64:
        out.append(cur)
        seen.add(cur)
        cur = parent_of(registry, cur)
    return out


def depth(registry: dict, slug: str) -> int:
    """0 for a top-level agent, 1 for its child, …"""
    return len(ancestors(registry, slug))


def descendants(registry: dict, slug: str) -> set[str]:
    out: set[str] = set()
    todo = [slug]
    while todo:
        for c in children(registry, todo.pop(), workers=True):
            if c not in out and c != slug:
                out.add(c)
                todo.append(c)
    return out


def height(registry: dict, slug: str) -> int:
    """How many levels of standing agents sit below ``slug`` (0 = none)."""
    level, best, seen = [slug], 0, {slug}
    while True:
        nxt = [c for s in level for c in children(registry, s) if c not in seen]
        if not nxt:
            return best
        seen.update(nxt)
        best += 1
        level = nxt


def view(registry: dict, slug: str) -> dict[str, str]:
    """The family fields an API returns for ``slug``."""
    e = _entry(registry, slug) or {}
    return {
        "parent": parent_of(registry, slug),
        "made_by": str(e.get("made_by") or ""),
        "made_for": str(e.get("made_for") or ""),
        "role": role_of(slug, e),
        "run": str(e.get("run") or ""),
        "factory": str(e.get("factory") or ""),
    }


# ── checks ──────────────────────────────────────────────────────────


def check_child(registry: dict, parent: str) -> None:
    """May ``parent`` get one more standing agent under it (an agent spawning one)?"""
    if depth(registry, parent) + 1 > MAX_DEPTH:
        raise FamilyError(f"{_name(registry, parent)} is already {MAX_DEPTH} levels down: "
                          "an agent there can't make agents of its own.")
    if len(children(registry, parent)) >= MAX_CHILDREN:
        raise FamilyError(f"{_name(registry, parent)} already has {MAX_CHILDREN} agents under it.")


def check_move(registry: dict, slug: str, parent: str) -> None:
    """May ``slug`` (with everything under it) be put under ``parent``
    ("" = make it one of its owner's own)?"""
    e = _entry(registry, slug)
    if e is None:
        raise FamilyError("That agent isn't on this deck.", 404)
    if role_of(slug, e) == WORKER:
        raise FamilyError("A run's worker can't be moved: it goes away when its run ends.")
    if not parent:
        return
    if parent == slug:
        raise FamilyError("An agent can't sit under itself.")
    pe = _entry(registry, parent)
    if pe is None:
        raise FamilyError("That agent isn't on this deck.", 404)
    if role_of(parent, pe) == WORKER:
        raise FamilyError("A run's worker can't have agents under it.")
    if not _same_owner(pe, e):
        raise FamilyError("Both agents have to be yours.", 403)
    if parent in descendants(registry, slug):
        raise FamilyError(f"{_name(registry, parent)} sits under {_name(registry, slug)}: "
                          "that would make a loop.")
    if depth(registry, parent) + 1 + height(registry, slug) > MAX_DEPTH:
        raise FamilyError(f"That would go past {MAX_DEPTH} levels.")
    if len([c for c in children(registry, parent) if c != slug]) >= MAX_CHILDREN:
        raise FamilyError(f"{_name(registry, parent)} already has {MAX_CHILDREN} agents under it.")


# ── changes (mutate the registry dict; the caller saves it) ─────────


def move(registry: dict, slug: str, parent: str, *, made_for: Any = None) -> dict[str, str]:
    """Put ``slug`` under ``parent`` ("" = top-level) by the owner's hand."""
    check_move(registry, slug, parent)
    e = registry[slug]
    if parent:
        e["parent"] = parent
    else:
        e.pop("parent", None)
    e["parent_set_by"] = "person"
    e.setdefault("role", STANDING)
    if made_for is not None:
        text = clean_made_for(made_for)
        if text:
            e["made_for"] = text
        else:
            e.pop("made_for", None)
    return view(registry, slug)


def on_removed(registry: dict, slug: str) -> list[str]:
    """``slug`` is leaving the deck: what sat under it moves up one level.
    Call before the entry is dropped. Returns the slugs that moved."""
    up = parent_of(registry, slug)
    moved = children(registry, slug, workers=True)
    for c in moved:
        if up:
            registry[c]["parent"] = up
        else:
            registry[c].pop("parent", None)
    return moved


def label(registry: dict, slug: str, fields: dict[str, Any]) -> bool:
    """Set family ``fields`` on an existing entry (a run's worker, right after
    its spawn). Unknown keys are ignored; "" clears a field. False when the
    entry is gone."""
    e = _entry(registry, slug)
    if e is None:
        return False
    for k in FIELDS:
        if k not in fields:
            continue
        v = clean_made_for(fields[k]) if k == "made_for" else str(fields[k] or "")
        if v:
            e[k] = v
        else:
            e.pop(k, None)
    return True


# ── at spawn ────────────────────────────────────────────────────────


def _env_map(config: Any) -> dict[str, str]:
    out: dict[str, str] = {}
    for ev in getattr(config, "env_vars", None) or []:
        if isinstance(ev, dict) and ev.get("key"):
            out[str(ev["key"])] = str(ev.get("value", ""))
    return out


def for_spawn(config: Any, *, prior: dict | None, caller: str) -> dict[str, str]:
    """The family fields of a process agent being (re)spawned, in order of say:

    1. what FD code asked for (``config._family``);
    2. a new agent another agent made (``caller``): that agent's child — a
       standing one, whatever env its maker handed it;
    3. a run's worker (the env its runner stamps on it);
    4. a re-spawn of an existing agent keeps its own (an agent restarting
       another agent never adopts it — and so can never make a loop);
    5. otherwise one of its owner's own.

    ``config.made_for`` (the maker's one line) fills ``made_for`` in every case.
    Empty values are left out.
    """
    explicit = dict(getattr(config, "_family", None) or {})
    env = _env_map(config)
    prior = prior if isinstance(prior, dict) else None
    made_at = str((prior or {}).get("made_at") or "") or now_iso()
    if explicit:
        out = {k: str(explicit.get(k) or "") for k in FIELDS}
        out["role"] = out["role"] if out["role"] in (STANDING, WORKER) else STANDING
    elif caller and prior is None:
        out = {"role": STANDING, "parent": caller, "made_by": caller, "parent_set_by": "spawn"}
    elif _WORKER_ENV & env.keys():
        run = next((f"{kind}:{env[k]}" for k, kind in _RUN_ENV if env.get(k)), "")
        out = {"role": WORKER, "run": run}
    elif prior is not None:
        out = {k: str(prior.get(k) or "") for k in FIELDS}
        out["role"] = out["role"] or STANDING
    else:
        out = {"role": STANDING}
    made_for = clean_made_for(getattr(config, "made_for", "") or out.get("made_for"))
    out["made_for"] = made_for
    out["made_at"] = made_at
    return {k: v for k, v in out.items() if k in FIELDS and v}
