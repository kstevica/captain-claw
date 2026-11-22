"""Family and groups → the agents: who made whom, and which of the owner's
agent groups each agent is in, in each agent's prompt.

Flight Deck writes ``family_context.md`` into a process agent's config home,
next to ``course_context.md`` (see :mod:`course_carrier` / :mod:`tenant_profile`
for the path and the atomic write). The agent inserts it verbatim on its
owner's turns (``tenant_context.load_family_context``), in the cached part of
the prompt: the block holds only who is who (names, what each is for, the
groups and who is in them), never live state, so it changes only when the
family or the groups do.

Standing agents that have a parent or children get "Your family"; standing
agents of an owner with agent groups get "Your groups" (the groups they are in,
with the others in each, and the names of the rest). Everyone else's file is
removed (fail closed). A run's workers get none: their run tells them what
they need.

The groups are the owner's ``fd:agent-groups`` setting (the agent column's
folders), kept here per owner (``set_groups``) because ``refresh`` is sync. Only
a deck with sign-in keeps them; without it they live in the browser and no
agent hears of them (as ``to='group:<name>'`` can't reach them there).

When it is written: after a spawn, a move, a removal and a rename (every agent
of that owner, unchanged files untouched), after the owner saves their groups,
and once at FD startup (``reconcile``).

Names and ``made_for`` lines come from people and agents: they are flattened
onto one line each, quoted, and capped, so none can pass for the block's own
words or start a heading.

Also here: what to call a sender to its receiver (``relation_words``), for the
peer and hand-over messages.
"""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Any

from captain_claw.flight_deck import agent_family as fam
from captain_claw.flight_deck import tenant_profile as tp
from captain_claw.logging import get_logger

log = get_logger(__name__)

FILE = "family_context.md"
# Siblings (and a group's members) listed by name before "and N more".
MAX_LISTED = 12
# The owner's other groups named before "and N more".
MAX_GROUPS = 20
GROUPS_SETTING = "fd:agent-groups"

# owner → their stored agent groups (decks with sign-in only).
_groups: dict[str, Any] = {}


def set_groups(owner: str, raw: Any) -> None:
    """Keep ``owner``'s agent groups as stored (a JSON string, a list, or None
    when they were removed)."""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError:
            raw = None
    if isinstance(raw, list):
        _groups[str(owner or "")] = raw
    else:
        _groups.pop(str(owner or ""), None)


def groups_for(owner: Any) -> Any:
    return _groups.get(str(owner or ""))


def _name(entry: dict, slug: str) -> str:
    text = re.sub(r"[\s*_`#>\[\]<]+", " ", str(entry.get("name") or slug)).strip()
    return (text or slug)[:60]


def _for(entry: dict) -> str:
    made_for = fam.clean_made_for(entry.get("made_for")).replace('"', "'")
    return f' — for: "{made_for}"' if made_for else ""


def _line(reg: dict, slug: str) -> str:
    e = reg.get(slug) or {}
    return f"- **{_name(e, slug)}**{_for(e)}"


def compose(reg: dict, slug: str, groups: Any = None) -> str:
    """The block for ``slug`` ("Your family", "Your groups"), or "" when it has
    neither (or is a worker). ``groups`` is its owner's stored agent groups."""
    me = reg.get(slug)
    if not isinstance(me, dict) or fam.role_of(slug, me) != fam.STANDING:
        return ""
    parts = [p for p in (_family(reg, slug, me), _groups_block(reg, slug, groups)) if p]
    return "\n\n".join(parts) + "\n" if parts else ""


def _names(reg: dict, slugs: list[str]) -> str:
    shown = sorted((_name(reg.get(s) or {}, s) for s in slugs), key=str.lower)
    text = ", ".join(f"**{n}**" for n in shown[:MAX_LISTED])
    return text + (f" and {len(shown) - MAX_LISTED} more" if len(shown) > MAX_LISTED else "")


def _groups_block(reg: dict, slug: str, groups: Any) -> str:
    """"Your groups": the owner's agent groups ``slug`` is in, with who else is
    in each, and the names of the others. "" when the owner has none."""
    view = fam.groups_of(reg, slug, groups)
    if not view:
        return ""
    mine = sorted((g for g in view if g["mine"]), key=lambda g: g["name"].lower())
    other = sorted((g["name"] for g in view if not g["mine"]), key=str.lower)
    out = ["## Your groups",
           "Your owner sorts their agents into groups. When they say \"the group\" or \"your group\", "
           "they mean one of these."]
    if mine:
        out += ["", "You are in:"]
        for g in mine:
            out.append(f"- **{g['name']}**, with {_names(reg, g['members'])}" if g["members"]
                       else f"- **{g['name']}** (only you so far)")
    else:
        out += ["", "You are in none of them."]
    if other:
        label = "Your owner's other groups" if mine else "Your owner's groups"
        listed = ", ".join(f"**{n}**" for n in other[:MAX_GROUPS])
        more = f" and {len(other) - MAX_GROUPS} more" if len(other) > MAX_GROUPS else ""
        out += ["", f"{label}: {listed}{more}."]
    example = mine[0]["name"] if mine else other[0]
    out += ["", "To ask a whole group something or give it a task, all at once: the `flight_deck` tool's `ask` "
            f"(waits for their answers) or `task` (their results come back later), with `to` set to "
            f"group:<name> (e.g. group:{example}). The `colleagues` tool's `list` shows the groups too."]
    return "\n".join(out)


def _family(reg: dict, slug: str, me: dict) -> str:
    """"Your family": parent, siblings, children ("" when none)."""
    parent = fam.parent_of(reg, slug)
    kids = sorted(fam.children(reg, slug), key=lambda s: _name(reg.get(s) or {}, s).lower())
    if not parent and not kids:
        return ""
    out = [
        "## Your family",
        "On this deck your owner's agents are related by who made whom. This says who is who: you can still "
        "consult, delegate to or hand over to any of your owner's agents, and messages from them say how they "
        "are related to you.",
    ]
    if parent:
        pe = reg.get(parent) or {}
        made = str(me.get("made_by") or "") == parent and me.get("parent_set_by") != "person"
        how = "made you" if made else "is the agent your owner put you under"
        out += ["", f"Your parent: **{_name(pe, parent)}** — it {how}{_for(me)}.",
                f"When you finish work {_name(pe, parent)} asked for, or it should know something, report to it: "
                "the `flight_deck` tool's `report` action (if you have that tool) reaches it even when it's asleep."]
        sibs = sorted((s for s in fam.children(reg, parent) if s != slug),
                      key=lambda s: _name(reg.get(s) or {}, s).lower())
        if sibs:
            out += ["", f"Your siblings, also under {_name(pe, parent)}:"]
            out += [_line(reg, s) for s in sibs[:MAX_LISTED]]
            if len(sibs) > MAX_LISTED:
                out.append(f"- and {len(sibs) - MAX_LISTED} more")
    if kids:
        out += ["", "Your children, agents you made or that your owner put under you:"]
        out += [_line(reg, s) for s in kids]
    out += ["", "To ask them something or give them a task, all at once: the `flight_deck` tool's `ask` (waits for "
            "their answers) or `task` (their results come back later), with `to` set to parent, siblings, children "
            "or family."]
    return "\n".join(out)


def relation_words(reg: dict, sender: str, receiver: str) -> str:
    """What ``sender`` is to ``receiver``, in words for the receiver ("" when
    they aren't family)."""
    if not sender or not receiver or sender == receiver:
        return ""
    if fam.parent_of(reg, receiver) == sender:
        return "your parent, the agent you sit under"
    if fam.parent_of(reg, sender) == receiver:
        return "your child, an agent that sits under you"
    p = fam.parent_of(reg, sender)
    if p and p == fam.parent_of(reg, receiver):
        return f"your sibling, also under {_name(reg.get(p) or {}, p)}"
    return ""


def sender_line(reg: dict, sender: str, receiver: str) -> str:
    """The first line of a peer message: who sent it, and what they are to you."""
    e = reg.get(sender)
    if not isinstance(e, dict):
        return ""
    rel = relation_words(reg, sender, receiver)
    return f"[From {_name(e, sender)}" + (f", {rel}" if rel else ", another of your owner's agents") + "]"


# ── files ───────────────────────────────────────────────────────────


def _write(agent_dir: Path, text: str) -> bool:
    """Write (or, for "", remove) the file. True when something changed."""
    target = tp.context_dir(agent_dir, "process") / FILE
    try:
        old = target.read_text(encoding="utf-8")
    except OSError:
        old = None
    if not text:
        if old is None:
            return False
        target.unlink(missing_ok=True)
        return True
    if old == text:
        return False
    tp._atomic_write(target, text)
    return True


def refresh(owner: str | None = None, *, registry: dict | None = None) -> int:
    """Bring the family files of ``owner``'s process agents (every agent's for
    None) in line with the registry. Never raises; returns how many changed."""
    from captain_claw.flight_deck import server as srv

    changed = 0
    try:
        reg = registry if registry is not None else srv._load_process_registry()
    except Exception as exc:  # noqa: BLE001
        log.warning("family: couldn't read the registry", error=str(exc))
        return 0
    for slug, entry in reg.items():
        if not isinstance(entry, dict):
            continue
        if owner is not None and not fam._same_owner(entry, {"owner": owner}):
            continue
        agent_dir = srv.DATA_DIR / slug
        if not agent_dir.is_dir():
            continue
        try:
            text = compose(reg, slug, groups_for(entry.get("owner")))
        except Exception as exc:  # noqa: BLE001 — fail closed: no block
            log.warning("family: couldn't compose", agent=slug, error=str(exc))
            text = ""
        try:
            changed += _write(agent_dir, text)
        except Exception as exc:  # noqa: BLE001
            log.warning("family: couldn't write", agent=slug, error=str(exc))
    return changed


def refresh_quietly(owner: Any) -> None:
    """``refresh`` for a hook that must never fail its caller."""
    try:
        refresh(str(owner or ""))
    except Exception:  # noqa: BLE001
        log.debug("family: refresh failed", exc_info=True)


async def groups_saved(owner: str, raw: Any) -> None:
    """The owner saved (or removed: None) their agent groups: their agents'
    blocks follow. A deck without sign-in keeps groups in the browser: nothing
    to do. Never raises."""
    from captain_claw.flight_deck.auth import _fd_auth_enabled

    try:
        if not _fd_auth_enabled():
            return
        set_groups(owner, raw)
        await asyncio.to_thread(refresh_quietly, owner)
    except Exception:  # noqa: BLE001
        log.debug("family: groups refresh failed", exc_info=True)


async def reconcile(db: Any) -> int:
    """At FD startup: read every owner's agent groups (decks with sign-in),
    then bring every agent's block in line. Never raises."""
    from captain_claw.flight_deck.auth import _fd_auth_enabled

    try:
        if _fd_auth_enabled() and db is not None:
            for owner, raw in (await db.users_with_setting(GROUPS_SETTING)).items():
                set_groups(owner, raw)
    except Exception as exc:  # noqa: BLE001 — the family part still goes out
        log.warning("family: couldn't read the agent groups", error=str(exc)[:200])
    return await asyncio.to_thread(refresh)
