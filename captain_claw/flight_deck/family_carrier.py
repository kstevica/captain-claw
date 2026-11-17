"""Family → the agents: who made whom, in each agent's prompt.

Flight Deck writes ``family_context.md`` into a process agent's config home,
next to ``course_context.md`` (see :mod:`course_carrier` / :mod:`tenant_profile`
for the path and the atomic write). The agent inserts it verbatim on its
owner's turns (``tenant_context.load_family_context``), in the cached part of
the prompt: the block holds only who is who (names, what each is for), never
live state, so it changes only when the family does.

Only standing agents that have a parent or children get one; everyone else's
file is removed (fail closed). A run's workers get none: their run tells them
what they need.

When it is written: after a spawn, a move, a removal and a rename (every agent
of that owner, unchanged files untouched), and once at FD startup.

Names and ``made_for`` lines come from people and agents: they are flattened
onto one line each, quoted, and capped, so none can pass for the block's own
words or start a heading.

Also here: what to call a sender to its receiver (``relation_words``), for the
peer and hand-over messages.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from captain_claw.flight_deck import agent_family as fam
from captain_claw.flight_deck import tenant_profile as tp
from captain_claw.logging import get_logger

log = get_logger(__name__)

FILE = "family_context.md"
# Siblings listed by name before "and N more".
MAX_LISTED = 12


def _name(entry: dict, slug: str) -> str:
    text = re.sub(r"[\s*_`#>\[\]<]+", " ", str(entry.get("name") or slug)).strip()
    return (text or slug)[:60]


def _for(entry: dict) -> str:
    made_for = fam.clean_made_for(entry.get("made_for")).replace('"', "'")
    return f' — for: "{made_for}"' if made_for else ""


def _line(reg: dict, slug: str) -> str:
    e = reg.get(slug) or {}
    return f"- **{_name(e, slug)}**{_for(e)}"


def compose(reg: dict, slug: str) -> str:
    """The block for ``slug``, or "" when it has no family (or is a worker)."""
    me = reg.get(slug)
    if not isinstance(me, dict) or fam.role_of(slug, me) != fam.STANDING:
        return ""
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
    return "\n".join(out) + "\n"


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
            text = compose(reg, slug)
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
