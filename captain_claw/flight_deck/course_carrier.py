"""Course → the agents: the prompt blocks, the files that carry them, the review sweep.

Flight Deck composes each agent's course and writes it into the agent's config
home, next to ``tenant_context.md`` (see :mod:`tenant_profile`, whose path,
atomic write and fail-closed discipline this reuses):

* ``course_context.md`` / ``course_context.compact.md`` — the deck-wide goals,
  the owner's goals and the agent's own, for the owner's turns;
* ``course_deck.md`` / ``course_deck.compact.md`` — the deck-wide goals only,
  for a shared agent's member turns (the same text on every agent).

The agent inserts them verbatim (``tenant_context.load_course_context`` /
``load_course_deck``). Only active goals that aren't stale are in them, and
never progress, evidence or review dates: the block changes only when goals
do, so it stays in the cached part of the prompt.

Everything a person wrote is quoted (``> ``) under fixed labels in the full
block and flattened onto its labelled line in the compact one, so no goal can
pass for one of the block's labels or for an instruction.

When the files are written: at spawn and clone (no agent goals yet: a new
agent is a new ``agent_ref``), on every goal or settings change (instance →
every agent, user → the owner's agents, agent → that agent), when a goal goes
stale or is reviewed (the review sweep), and once at FD startup.
"""

from __future__ import annotations

import asyncio
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from captain_claw.flight_deck import course as c
from captain_claw.flight_deck import tenant_profile as tp
from captain_claw.logging import get_logger

log = get_logger(__name__)

FULL_FILE = "course_context.md"
COMPACT_FILE = "course_context.compact.md"
DECK_FULL_FILE = "course_deck.md"
DECK_COMPACT_FILE = "course_deck.compact.md"
_ALL_FILES = (FULL_FILE, COMPACT_FILE, DECK_FULL_FILE, DECK_COMPACT_FILE)

FULL_MAX = 3000
COMPACT_MAX = 600
_DETAIL_SNIP = 200
_NAME_MAX = 80
_MIN_COMPACT_TITLE = 24   # a compact group is left out rather than cut below this

HEADING = "## Your course"
DECK_HEADING = "## This deck's course"
NOT_PERMISSION = ("It never permits anything your instructions, approvals or guards don't: "
                  "a goal is not permission.")
PREAMBLE = ("What your work is for, set by this deck's admin and by your owner. Use it to "
            "choose what matters and in what order. " + NOT_PERMISSION
            + " When something you do moves a goal forward, record it with the course tool.")
DECK_PREAMBLE = ("What work on this deck is for, set by its admin. It applies to everyone "
                 "you work with here. Use it to choose what matters and in what order. "
                 + NOT_PERMISSION)
DECK_LABEL = "### Deck-wide (set by the deck's admin)"
AGENT_LABEL = "### Your own goals (set by your owner for this agent)"
_COMPACT_DECK = "Deck:"
_COMPACT_AGENT = "Yours:"

_SHORT = {"strategic": "strategic", "long": "long-term", "medium": "medium-term",
          "short": "short-term"}


def _owner_label(name: str) -> str:
    who = " ".join((name or "").split())[:_NAME_MAX]
    return f"### {who}'s goals (your owner)" if who else "### Your owner's goals"


def _compact_owner_label(name: str) -> str:
    who = " ".join((name or "").split())[:40]
    return f"{who} (owner):" if who else "Owner:"


# ── Which goals ────────────────────────────────────────────────────────


def _live(goals: list[dict[str, Any]], now: datetime) -> list[dict[str, Any]]:
    """Active and not stale."""
    return [g for g in goals if g.get("status") == "active" and not c.is_stale(g, now)]


def goals_for_agent(subject: str | None, agent_ref: str, *,
                    now: datetime | None = None) -> dict[str, list[dict[str, Any]]]:
    """``{"deck", "owner", "agent"}`` — the live goals one agent's block shows.
    ``subject`` is whose user goals it carries (``tenant_profile.profile_subject``;
    None for an unowned agent or a deleted owner: deck-wide goals only)."""
    store = c.get_store()
    now = now or datetime.now(UTC)
    out: dict[str, list[dict[str, Any]]] = {
        "deck": _live(store.list_goals(scope="instance", statuses=("active",)), now),
        "owner": [],
        "agent": [],
    }
    if subject:
        out["owner"] = _live(store.list_goals(scope="user", owner_id=subject,
                                              statuses=("active",)), now)
        if agent_ref:
            # Only goals its current owner set: an agent passed to someone else
            # never carries what the previous owner wrote for it.
            out["agent"] = _live(store.list_goals(scope="agent", owner_id=subject,
                                                  agent_refs=[agent_ref],
                                                  statuses=("active",)), now)
    return out


# ── Composing ──────────────────────────────────────────────────────────


def _head(goal: dict[str, Any], shown: set[str]) -> str:
    """The fixed line naming a goal: its id, horizon, due date and parent."""
    bits = [f"- G:{goal['id']}", _SHORT.get(goal["horizon"], goal["horizon"])]
    due = c._parse_iso(goal.get("due_at"))
    if due is not None:
        bits.append(f"due {due.date().isoformat()}")
    parent = goal.get("parent_id") or ""
    if parent and parent in shown:
        bits.append(f"serves G:{parent}")
    return " · ".join(bits)


def _sections(groups: dict[str, list[dict[str, Any]]], owner_name: str) -> list[tuple[str, list]]:
    out = []
    if groups.get("deck"):
        out.append((DECK_LABEL, groups["deck"]))
    if groups.get("owner"):
        out.append((_owner_label(owner_name), groups["owner"]))
    if groups.get("agent"):
        out.append((AGENT_LABEL, groups["agent"]))
    return out


def _render_full(heading: str, preamble: str, sections: list[tuple[str, list]],
                 *, details: bool) -> tuple[list[str], list[str]]:
    """(fixed lines, person-written texts) in order; a person-written text is
    rendered as ``"  > " + text`` in place of a ``None`` marker in the fixed list."""
    shown = {g["id"] for _, goals in sections for g in goals}
    fixed: list[str | None] = [heading, preamble]
    texts: list[str] = []
    for label, goals in sections:
        fixed += ["", label]
        for g in goals:
            fixed.append(_head(g, shown))
            fixed.append(None)
            texts.append(tp._flat(g["title"]))
            if g.get("success"):
                fixed.append(None)
                texts.append("Done when: " + tp._flat(g["success"]))
            if details and g.get("detail"):
                fixed.append(None)
                texts.append("Why: " + tp._snip(tp._flat(g["detail"]), _DETAIL_SNIP))
    return fixed, texts  # type: ignore[return-value]


def _join(fixed: list[str | None], texts: list[str]) -> str:
    it = iter(texts)
    return "\n".join(line if line is not None else "  > " + next(it) for line in fixed)


def _compose_full(heading: str, preamble: str, sections: list[tuple[str, list]]) -> str:
    if not sections:
        return ""
    for details in (True, False):
        fixed, texts = _render_full(heading, preamble, sections, details=details)
        text = _join(fixed, texts)
        if len(text) <= FULL_MAX:
            return text
    # Still too long: keep every goal, share the room between their texts.
    fixed_len = sum(len(line) for line in fixed if line is not None)
    fixed_len += len(fixed) - 1 + 4 * len(texts)       # newlines + "  > "
    sizes = tp._allot([len(t) for t in texts], FULL_MAX - fixed_len)
    return tp._snip(_join(fixed, [tp._snip(t, n) for t, n in zip(texts, sizes)]), FULL_MAX)


def _compose_compact(heading: str, preamble: str,
                     groups: list[tuple[str, list[dict[str, Any]]]]) -> str:
    """One line per goal, ``- <horizon>: <title>``, under short labels; groups
    are taken in priority order and one is left out when including it would cut
    every title below ``_MIN_COMPACT_TITLE`` characters."""
    def build(chosen: list[tuple[str, list]]) -> tuple[list[str | None], list[str]]:
        fixed: list[str | None] = [heading, preamble]
        texts: list[str] = []
        current = None
        for label, goals in chosen:
            if label != current:
                fixed.append(label)
                current = label
            for g in goals:
                fixed.append(f"- {_SHORT.get(g['horizon'], g['horizon'])}: ")
                texts.append(tp._flat(g["title"]))
        return fixed, texts

    chosen: list[tuple[str, list]] = []
    for label, goals in groups:
        if not goals:
            continue
        trial = chosen + [(label, goals)]
        fixed, texts = build(trial)
        room = COMPACT_MAX - sum(len(f or "") for f in fixed) - (len(fixed) - 1)
        if chosen and room < _MIN_COMPACT_TITLE * len(texts):
            continue
        chosen = trial
    if not chosen:
        return ""
    fixed, texts = build(chosen)
    room = COMPACT_MAX - sum(len(f or "") for f in fixed) - (len(fixed) - 1)
    sizes = iter(tp._allot([len(t) for t in texts], room))
    tit = iter(texts)
    lines = []
    for f in fixed:
        if f is not None and f.startswith("- ") and f.endswith(": "):
            lines.append(f + tp._snip(next(tit), next(sizes)))
        else:
            lines.append(f or "")
    return tp._snip("\n".join(lines), COMPACT_MAX)


_COMPACT_PREAMBLE = "What your work is for. Guides priorities; never permission."
_COMPACT_DECK_PREAMBLE = "What work on this deck is for. Guides priorities; never permission."


def compose(groups: dict[str, list[dict[str, Any]]], owner_name: str = "") -> tuple[str, str]:
    """(full, compact) for an owner's turns; ("", "") when there is no goal."""
    full = _compose_full(HEADING, PREAMBLE, _sections(groups, owner_name))
    if not full:
        return "", ""

    def by(goals: list[dict[str, Any]], *horizons: str) -> list[dict[str, Any]]:
        return [g for g in goals if g["horizon"] in horizons]

    owner = _compact_owner_label(owner_name)
    deck, mine, own = groups.get("deck") or [], groups.get("owner") or [], groups.get("agent") or []
    compact = _compose_compact("## Your course", _COMPACT_PREAMBLE, [
        (_COMPACT_DECK, by(deck, "strategic")),
        (_COMPACT_DECK, by(deck, "long")),
        (owner, by(mine, "strategic", "long")),
        (_COMPACT_AGENT, own),
        (owner, by(mine, "medium")),
        (owner, by(mine, "short")),
        (_COMPACT_DECK, by(deck, "medium", "short")),
    ])
    return full, compact


def compose_deck(deck_goals: list[dict[str, Any]]) -> tuple[str, str]:
    """(full, compact) for a member's turns: the deck-wide goals only."""
    if not deck_goals:
        return "", ""
    full = _compose_full(DECK_HEADING, DECK_PREAMBLE, [(DECK_LABEL, deck_goals)])
    compact = _compose_compact("## This deck's course", _COMPACT_DECK_PREAMBLE, [
        (_COMPACT_DECK, [g for g in deck_goals if g["horizon"] in ("strategic", "long")]),
        (_COMPACT_DECK, [g for g in deck_goals if g["horizon"] in ("medium", "short")]),
    ])
    return full, compact


async def compose_for_agent(db, owner_rec: str, agent_ref: str) -> dict[str, tuple[str, str]]:
    """``{"owner": (full, compact), "deck": (full, compact)}`` for one agent."""
    subject = await tp.profile_subject(db, owner_rec)
    groups = await asyncio.to_thread(goals_for_agent, subject, agent_ref)
    name = await tp.owner_name(db, subject) if subject else ""
    return {"owner": compose(groups, name), "deck": compose_deck(groups["deck"])}


# ── Files ──────────────────────────────────────────────────────────────


def write_files(agent_dir: Path, runtime: str, texts: dict[str, tuple[str, str]]) -> None:
    """Write (or, for an empty block, remove) the four files where an agent of
    ``runtime`` reads them."""
    d = tp.context_dir(agent_dir, runtime)
    pairs = ((FULL_FILE, COMPACT_FILE, texts.get("owner") or ("", "")),
             (DECK_FULL_FILE, DECK_COMPACT_FILE, texts.get("deck") or ("", "")))
    for full_name, compact_name, (full, compact) in pairs:
        for name, text in ((full_name, full), (compact_name, compact or full)):
            target = d / name
            if full.strip():
                tp._atomic_write(target, text)
            else:
                target.unlink(missing_ok=True)


def remove_files(agent_dir: Path, runtime: str) -> None:
    """Fail closed: an agent whose course could not be written gets none."""
    try:
        d = tp.context_dir(agent_dir, runtime)
        for name in _ALL_FILES:
            (d / name).unlink(missing_ok=True)
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not remove a stale course", agent=Path(agent_dir).name,
                    runtime=runtime, error=str(exc))


_LOCK: asyncio.Lock | None = None
_LOCK_LOOP = None


def _lock() -> asyncio.Lock:
    global _LOCK, _LOCK_LOOP
    loop = asyncio.get_running_loop()
    if _LOCK is None or _LOCK_LOOP is not loop:
        _LOCK, _LOCK_LOOP = asyncio.Lock(), loop
    return _LOCK


def list_agents(owner: str | None) -> list[dict[str, Any]]:
    """This deck's agents (every one, or only ``owner``'s): ``{runtime, slug,
    dir, owner, ref, name}``, from the process registry and this deck's Docker
    containers. Keyed by (runtime, slug), as ``tenant_profile._list_agents``."""
    from captain_claw.flight_deck import agent_sharing as sharing
    from captain_claw.flight_deck import server as srv

    found: dict[tuple[str, str], dict[str, Any]] = {}
    try:
        registry = srv._load_process_registry()
    except Exception:  # noqa: BLE001
        registry = {}
    for slug, entry in (registry or {}).items():
        entry = entry or {}
        rec = str(entry.get("owner") or "")
        if owner is not None and rec != owner:
            continue
        try:
            ref = sharing.process_ref(slug, entry)
        except Exception:  # noqa: BLE001
            ref = ""
        found.setdefault(("process", slug), {
            "runtime": "process", "slug": slug, "owner": rec, "ref": ref,
            "name": str(entry.get("name") or slug)})
    try:
        containers = srv._deck_containers(all=True)
    except Exception:  # noqa: BLE001 — Docker not available
        containers = []
    for cont in containers:
        labels = cont.labels or {}
        rec = str(labels.get(srv.OWNER_LABEL, "") or "")
        if owner is not None and rec != owner:
            continue
        try:
            ref = sharing.docker_ref(cont)
        except Exception:  # noqa: BLE001
            ref = ""
        found.setdefault(("docker", cont.name), {
            "runtime": "docker", "slug": cont.name, "owner": rec, "ref": ref,
            "name": str(labels.get("flight-deck.agent-name") or cont.name)})
    out = []
    for a in found.values():
        a["dir"] = srv.DATA_DIR / a["slug"]
        if a["dir"].is_dir():
            out.append(a)
    return sorted(out, key=lambda a: a["name"].lower())


def _auth_enabled() -> bool:
    return tp._auth_enabled()


async def write_on_spawn(agent_dir: Path, owner_id: str, runtime: str) -> None:
    """A new (or cloned) agent's course: the deck's and its owner's goals. Never
    fails a spawn; fails closed (no files) when it can't be written."""
    try:
        async with _lock():
            try:
                from captain_claw.flight_deck.auth import get_db

                texts = await compose_for_agent(get_db(), owner_id, "")
                write_files(agent_dir, runtime, texts)
            except Exception as exc:  # noqa: BLE001
                log.warning("Could not write the course for a new agent",
                            agent=Path(agent_dir).name, runtime=runtime, error=str(exc))
                remove_files(agent_dir, runtime)
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not write the course for a new agent",
                    agent=Path(agent_dir).name, runtime=runtime, error=str(exc))


async def refresh_agents(db, owner_id: str | None = None, *, agent_ref: str = "") -> int:
    """Rewrite (or remove) the course files of ``owner_id``'s agents — every
    agent when None (a deck-wide change, startup) or on an auth-off deck —
    or of the one agent ``agent_ref``. An agent that fails gets no files and
    isn't counted. How many agents were updated."""
    owner = owner_id if (_auth_enabled() and owner_id is not None) else None
    updated = 0
    async with _lock():
        agents = await asyncio.to_thread(list_agents, owner)
        if agent_ref:
            agents = [a for a in agents if a["ref"] == agent_ref]
        for a in agents:
            try:
                texts = await compose_for_agent(db, a["owner"], a["ref"])
                write_files(a["dir"], a["runtime"], texts)
                updated += 1
            except Exception as exc:  # noqa: BLE001
                log.warning("Could not update an agent's course", agent=a["slug"],
                            runtime=a["runtime"], error=str(exc))
                remove_files(a["dir"], a["runtime"])
    return updated


async def refresh_for_goal(db, goal: dict[str, Any]) -> int:
    """Rewrite the files of every agent ``goal`` reaches."""
    scope = goal.get("scope")
    if scope == "instance":
        return await refresh_agents(db, None)
    if scope == "user":
        return await refresh_agents(db, str(goal.get("owner_id") or ""))
    if scope == "agent":
        return await refresh_agents(db, str(goal.get("owner_id") or ""),
                                    agent_ref=str(goal.get("agent_ref") or ""))
    return 0


async def reconcile(db) -> int:
    """Bring every agent's course files in line with the store — once at FD
    startup. Best-effort: never raises."""
    try:
        updated = await refresh_agents(db, None)
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not reconcile the agents' courses", error=str(exc))
        return 0
    log.info("Courses reconciled", agents=updated)
    return updated


# ── The review sweep ───────────────────────────────────────────────────

REVIEW_REF = "course_review"


def review_tick_s() -> float:
    try:
        return max(5.0, float(os.environ.get("FD_COURSE_REVIEW_TICK_S", "1800")))
    except ValueError:
        return 1800.0


def review_ref(goal: dict[str, Any]) -> str:
    """One Line item per goal per review cycle."""
    return f"{goal['id']}@{goal.get('review_at') or ''}"


async def reviewers(db, goal: dict[str, Any]) -> list[str]:
    """Who is asked to review ``goal``: its owner; the admins for a deck-wide one."""
    if goal.get("scope") != "instance":
        return [c.norm_user(goal.get("owner_id"))]
    if not _auth_enabled():
        return [tp._local_id()]
    try:
        return await db.list_admin_ids()
    except Exception:  # noqa: BLE001
        return []


async def resolve_reviews(db, goal: dict[str, Any]) -> int:
    """A goal was reviewed or left active: its waiting "Still on course?" items
    are answered, for everyone they went to. Never raises."""
    done = 0
    prefix = f"{goal['id']}@"
    try:
        for uid in await reviewers(db, goal):
            refs = [r for r in await db.unresolved_line_refs(uid, REVIEW_REF)
                    if r.startswith(prefix)]
            if refs:
                done += await db.resolve_line_items(uid, REVIEW_REF, refs)
    except Exception as exc:  # noqa: BLE001
        log.debug("course: reviews of %s not resolved: %s", goal.get("id"), exc)
    return done


def _review_body(goal: dict[str, Any], stale: bool) -> str:
    when = ("It has been out of your agents' prompts since its review was missed. "
            if stale else
            f"If nobody reviews it within {c.grace_days(int(goal['review_every_days']))} days "
            "of its review date, it leaves your agents' prompts until someone does. ")
    return (f"{c.HORIZON_LABELS.get(goal['horizon'], goal['horizon'])} goal: "
            f"{tp._flat(goal['title'])}\n\n{when}"
            "Keep it, change it, pause it, mark it done or drop it on the Course page.")


_last_stale: set[str] | None = None


async def review_round(db, now: datetime | None = None) -> dict[str, int]:
    """One sweep: ask about goals that came due, and about goals whose measure
    reached its target; rewrite the agents whose block changed because a goal
    went stale. Never raises."""
    global _last_stale
    now = now or datetime.now(UTC)
    posted = refreshed = 0
    try:
        from captain_claw.flight_deck import line

        due = await asyncio.to_thread(c.get_store().due_for_review, now)
        for goal in due:
            stale = c.is_stale(goal, now)
            for uid in await reviewers(db, goal):
                item = await line.post_once(
                    uid, "ask", f"Still on course? {tp._flat(goal['title'])}"[:200],
                    _review_body(goal, stale), bell_type="course_review", needs_you=True,
                    ref_type=REVIEW_REF, ref_id=review_ref(goal), source="Course")
                posted += 1 if item else 0
        # Phase 4: a measure that reached its target asks whether the goal is done.
        from captain_claw.flight_deck import course_progress

        await course_progress.target_round(db)
        stale_now = {g["id"] for g in due if c.is_stale(g, now)}
        if _last_stale is None:
            _last_stale = stale_now          # startup's reconcile wrote the files
        elif stale_now != _last_stale:
            gone = [g for g in due if g["id"] in stale_now - _last_stale]
            _last_stale = stale_now
            if any(g["scope"] == "instance" for g in gone):
                refreshed += await refresh_agents(db, None)
            else:
                for owner in sorted({str(g["owner_id"]) for g in gone}):
                    refreshed += await refresh_agents(db, owner)
    except Exception as exc:  # noqa: BLE001
        log.warning("course: review sweep failed", error=str(exc))
    return {"posted": posted, "refreshed": refreshed}


async def review_loop(db, stop: asyncio.Event) -> None:
    """The review sweep, every ``FD_COURSE_REVIEW_TICK_S`` (default 30 min)."""
    while not stop.is_set():
        await review_round(db)
        try:
            await asyncio.wait_for(stop.wait(), timeout=review_tick_s())
        except TimeoutError:
            pass
