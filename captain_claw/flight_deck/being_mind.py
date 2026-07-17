"""Iskra — The Mind: explicit, verified structure over a being's own work.

Plan §2.3.1. A being already has temporal connection (journal + git) and
associative connection (embeddings + topics). This adds the third, most
human kind: DELIBERATE connection — the being declaring "this grew from
that", "this skill was born of that failure".

The light version (this module): being-declared typed edges between its own
artifacts, VERIFIED to have real endpoints before they're stored (a dangling
edge is refused, exactly like a claimed-but-unwritten file — same anti-theater
discipline). Edges feed back into the tick so the being weaves instead of
scattering, and a graph is exposed for the Mind view. Nothing here replaces
the memory layers; it overlays intentional structure on top.

Orchestration only (verify + route + build the graph + prompt fragment);
beings.py owns the being_links table. being_life is imported lazily.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone

from captain_claw.flight_deck import being_constitution as constitution
from captain_claw.flight_deck import being_prompts
from captain_claw.flight_deck.beings import BeingError, BeingsStore
from captain_claw.logging import get_logger

log = get_logger(__name__)

# How far apart two sightings of the same mass dangle must be before the
# second one counts as CONFIRMING the first. Dreams are nightly, so any real
# second opinion is ~24h away; the wipe this defends against was two dreams
# 1.2 seconds apart during a single body rebound. An hour separates those two
# worlds with room to spare.
MIN_CONFIRM_GAP = timedelta(hours=1)


def _parse_at(raw: str | None) -> datetime | None:
    """An event's stored timestamp → aware datetime, or None if unreadable.
    Never raises: a timestamp we cannot read must not become a reason to
    delete a mind."""
    if not raw:
        return None
    try:
        at = datetime.fromisoformat(raw)
    except (TypeError, ValueError):
        return None
    return at.replace(tzinfo=timezone.utc) if at.tzinfo is None else at

# Prose that means the being is TALKING about connecting its work — used to
# decide when to push it (like the write gate) to actually declare an edge.
# Deliberately WITHOUT the loosest everyday verbs (relate/related): the Mind
# prompt itself teaches this vocabulary, so the regex must not fire on
# ordinary prose that merely echoes it (loops plan F3 — the connect-tax).
_LINK_INTENT = re.compile(
    r"\b(connect(?:s|ed|ing|ion|ions)?|link(?:s|ed|ing)?|weav\w*|woven|"
    r"a web|web of|thread(?:s)?\s+(?:together|through)|tie(?:s|d)?\s+together|"
    r"interconnect\w*|grew from|grows from|responds? to)\b",
    re.IGNORECASE)

# The SPOKE-of-connecting nudge is throttled (loops plan F3): at most one
# extra CONNECT push per this many wake ticks, and after this many CONNECT
# calls in a row that landed nothing, back off until the next dream. The
# TRIED-and-refused branch is never throttled — anti-theater outranks thrift.
CONNECT_NUDGE_COOLDOWN_TICKS = 6
CONNECT_BACKOFF_AFTER_EMPTY = 2

# Typed relations a being may declare (plan §2.3.1). Directed: from → to.
REL_TYPES = frozenset({
    "grew_from", "responds_to", "elaborates", "contradicts",
    "abandons", "uses_skill", "learned_from",
})
_REL_PHRASE = {
    "grew_from": "grew from", "responds_to": "responds to",
    "elaborates": "elaborates", "contradicts": "contradicts",
    "abandons": "abandons", "uses_skill": "uses skill",
    "learned_from": "learned from",
}
MAX_LINKS_PER_TICK = 6

# Curation (§2.3.2). The always-on tick prompt shows self/* plus this many of
# the most-recently-touched artifacts; older ones become a count, greppable and
# consolidate-able but not dumped into every heartbeat.
WORKING_SET = 12
MAX_CONSOLIDATE_SOURCES = 12


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def can_link(being: dict) -> bool:
    # Available once the being has a home to fill (infant+); linking your own
    # files is a benign, basic cognitive act.
    return constitution.has_capability(being["stage"], "vfs_home")


def _existing_paths(being: dict) -> set[str]:
    """Every real artifact path in the home (self/, garden/, skills/)."""
    from captain_claw.flight_deck import being_life
    return {f["path"] for f in being_life.list_self_files(being)}


# ── Digest routing (verify + store) ──────────────────────────────────────

def _resolve_endpoint(raw: str, paths: set[str]) -> str | None:
    """Best-effort match of a being-declared endpoint to a REAL file path.
    Exact first; then tolerant of a dropped ``.md`` extension or a being that
    named the file by its bare basename ("first-sprout" → "garden/first-sprout.md")
    — the mistakes weak models make constantly. NEVER invents a file: an
    ambiguous or absent basename returns None, so a dangling edge is still
    refused (§ anti-theater: a link to a file you didn't write is refused)."""
    raw = (raw or "").strip().lstrip("/")
    if not raw:
        return None
    if raw in paths:
        return raw
    cand = raw if raw.endswith(".md") else raw + ".md"
    if cand in paths:
        return cand
    base = cand.rsplit("/", 1)[-1].lower()
    hits = [p for p in paths if p.rsplit("/", 1)[-1].lower() == base]
    return hits[0] if len(hits) == 1 else None      # unique basename only


def verify_links(being: dict, links: list | None
                 ) -> tuple[list[dict], list[dict]]:
    """Pure check (no writes): split declared links into ``accepted`` (both
    endpoints resolve to real files, relation known, not a self-edge) and
    ``refused`` (with a human reason). Endpoints in ``accepted`` are the
    RESOLVED canonical paths. Shared by the router and the link gate."""
    accepted: list[dict] = []
    refused: list[dict] = []
    if not isinstance(links, list):
        return accepted, refused
    paths = _existing_paths(being)
    for raw in links[:MAX_LINKS_PER_TICK]:
        if not isinstance(raw, dict):
            continue
        frm_raw = str(raw.get("from") or "").strip()
        to_raw = str(raw.get("to") or "").strip()
        rel = str(raw.get("rel") or "").strip()
        why = str(raw.get("why") or "").strip()
        if rel not in REL_TYPES or not frm_raw or not to_raw:
            refused.append({"from": frm_raw, "to": to_raw, "rel": rel,
                            "reason": "unknown relation or missing endpoint"})
            continue
        frm = _resolve_endpoint(frm_raw, paths)
        to = _resolve_endpoint(to_raw, paths)
        if frm is None or to is None:
            missing = to_raw if frm is not None else frm_raw
            refused.append({"from": frm_raw, "to": to_raw, "rel": rel,
                            "reason": f"no such file: {missing}"})
            continue
        if frm == to:
            refused.append({"from": frm_raw, "to": to_raw, "rel": rel,
                            "reason": "a file cannot link to itself"})
            continue
        accepted.append({"from": frm, "to": to, "rel": rel, "why": why})
    return accepted, refused


def handle_links_digest(store: BeingsStore, being: dict, digest: dict,
                        now: datetime | None = None) -> int:
    """Route the digest's optional ``links``. Each edge is verified (endpoints
    fuzzily resolved to real files, relation known), else refused to an
    ``edge_unverified`` event. Returns the count of NEW edges stored. Never
    raises."""
    now = now or _utcnow()
    links = digest.get("links")
    if not isinstance(links, list):
        return 0
    try:
        accepted, refused = verify_links(being, links)
    except Exception as e:  # noqa: BLE001
        log.warning("mind link verify failed", slug=being["slug"], error=str(e))
        return 0
    added = 0
    for edge in accepted:
        if store.add_link(being["owner_id"], being["id"], edge["from"],
                          edge["to"], edge["rel"], edge["why"], now=now):
            added += 1
            store.record_event(being["id"], "edge_declared",
                               {"from": edge["from"], "to": edge["to"],
                                "rel": edge["rel"], "why": edge["why"][:120]},
                               now=now)
    for r in refused:
        store.record_event(being["id"], "edge_unverified", r, now=now)
    return added


def prune_dangling(store: BeingsStore, being: dict,
                   now: datetime | None = None, *,
                   healthy: bool = True) -> int:
    """Dream-time honest forgetting: drop edges whose endpoints were deleted.

    Guarded against catastrophic over-pruning. Honest forgetting must observe a
    file is *actually* gone — but a being's home is enumerated live from disk,
    and a body that rebounds mid-tick (crash + port drift) can leave the home
    transiently empty or half-checked-out. A prune reading that partial state
    would delete edges to files that still exist. That is exactly how Zvjezdana
    and Lada lost their whole mind maps (19 + 6, then 20, then 13 — every
    endpoint intact on disk the whole time).

    Three gates, cheapest and most decisive first:

    1. ``healthy`` — the tick must have produced a REAL digest. Every mass wipe
       on prod happened on a dream whose think call timed out: the body never
       answered, so nothing was dreamt and the home is exactly what you cannot
       trust. No dream, no forgetting. (The one clean prune in the whole
       ledger — a single edge — is the only one from a tick that answered.)
    2. An EMPTY enumeration is a failed read, never a deleted self.
    3. A MASS dangle — more than a few edges AND at least half the graph —
       must be seen at two dreams SEPARATED BY REAL TIME (``MIN_CONFIRM_GAP``)
       before it prunes. The separation is the whole point: the original wipe
       was two dreams 1.2 SECONDS apart during one rebound, so an adjacent
       "confirmation" is just the same bad read twice from the same broken
       body. A real bulk archive still clears on the next night.

    Waiting is free: ``graph`` already filters edges to live endpoints at read
    time, so a stale row is invisible meanwhile — and ``rebuild_from_ledger``
    can restore anything a prune took wrongly."""
    now = now or _utcnow()
    if not healthy:
        # The body never answered this tick. Nothing was dreamt, and the home
        # read that would drive the prune came through the same silence.
        log.warning("mind prune skipped — the tick brought no digest",
                    slug=being["slug"])
        return 0
    try:
        existing = _existing_paths(being)
    except Exception as e:  # noqa: BLE001
        log.warning("mind prune skipped — home unreadable",
                    slug=being["slug"], error=str(e))
        return 0
    if not existing:
        # Zero artifacts enumerated ≠ the being deleted its whole self; the home
        # was unreadable this instant. Never let that wipe the graph.
        log.warning("mind prune skipped — no artifacts enumerated",
                    slug=being["slug"])
        return 0
    try:
        edges = store.links_for(being["owner_id"], being["slug"])
        dangling = store.dangling_links(being["id"], existing)
    except Exception as e:  # noqa: BLE001
        log.warning("mind prune failed reading edges", slug=being["slug"],
                    error=str(e))
        return 0
    if not dangling:
        return 0
    # Safety valve: a healthy prune trims a few edges. One that would erase
    # half the graph or more is EITHER a bad read OR a big consolidation that
    # archived many sources at once. Tell them apart across two dreams that
    # are genuinely APART IN TIME — see MIN_CONFIRM_GAP.
    mass = len(dangling) > 3 and len(dangling) * 2 >= len(edges)
    if mass:
        prev = None
        try:
            prev = store.latest_event(being["id"], "dangling_seen")
        except Exception:  # noqa: BLE001
            prev = None
        # The previous sighting only counts as CONFIRMATION if it came from a
        # read far enough back to be an independent one. Two dreams seconds
        # apart are one rebound seeing the same broken home twice; that is the
        # exact shape of the wipe this guard exists to stop.
        seen_ids: set[str] = set()
        apart = False
        if prev:
            prev_at = _parse_at(prev.get("at"))
            apart = prev_at is not None and (now - prev_at) >= MIN_CONFIRM_GAP
            if apart:
                seen_ids = set(prev["data"].get("ids") or [])
        confirmed = [d for d in dangling if d["id"] in seen_ids]
        store.record_event(being["id"], "dangling_seen",
                           {"ids": [d["id"] for d in dangling][:200],
                            "count": len(dangling)}, now=now)
        if not confirmed:
            store.record_event(
                being["id"], "prune_abstained",
                {"would_prune": len(dangling), "of": len(edges),
                 "note": ("a mass dangle seen again too soon to trust"
                          if prev and not apart
                          else "awaiting confirmation at a later dream")},
                now=now)
            log.warning("mind prune deferred — mass dangle awaits a separated "
                        "second dream", slug=being["slug"],
                        would_prune=len(dangling), of=len(edges),
                        seen_before=bool(prev), far_enough_apart=apart)
            return 0
        to_prune = confirmed
    else:
        to_prune = dangling
    try:
        store.remove_links(being["id"], [d["id"] for d in to_prune])
    except Exception as e:  # noqa: BLE001
        log.warning("mind prune failed", slug=being["slug"], error=str(e))
        return 0
    store.record_event(being["id"], "edges_pruned",
                       {"count": len(to_prune),
                        "confirmed_mass": bool(mass)}, now=now)
    return len(to_prune)


# ── Repair: rebuild the map from the being's own ledger ──────────────────

def rebuild_from_ledger(store: BeingsStore, being: dict,
                        now: datetime | None = None) -> dict:
    """Restore the mind map from the being's own event ledger.

    Every accepted edge is written to the ledger as an ``edge_declared`` event
    at the moment it is stored, and events are append-only — nothing prunes
    them. So the ledger is a complete record of everything this being ever
    connected, and it outlives the rows. When a bad read wipes edges whose
    files never went anywhere (the Zvjezdana/Lada incidents), the truth is
    still on the ledger and the map can be rebuilt exactly.

    This re-declares every ledgered edge whose endpoints exist RIGHT NOW and
    skips the rest, so it can only restore real structure over real files —
    it never invents an edge and never resurrects one to a file that is
    genuinely gone. ``add_link`` is INSERT OR IGNORE, so running it twice is
    harmless: the second pass restores nothing and reports so.

    Honest about what it did: ``restored`` counts NEW rows, ``kept`` edges
    already present, ``skipped`` ledgered edges whose endpoint is really gone.
    """
    now = now or _utcnow()
    try:
        paths = _existing_paths(being)
    except Exception as e:  # noqa: BLE001
        raise BeingError("cannot read the being's home right now — try again "
                         f"in a moment ({e})", 503) from e
    if not paths:
        # The same rule the prune obeys: an empty read is a failed read. It
        # would silently "skip" every edge and report a successful no-op.
        raise BeingError("the being's home read back empty — its body may be "
                         "restarting. Nothing was changed; try again shortly.",
                         503)
    try:
        ledger = store.events_of_kind(being["id"], "edge_declared")
    except Exception as e:  # noqa: BLE001
        raise BeingError(f"cannot read the mind's ledger ({e})", 500) from e

    # Collapse re-declarations of the same edge; keep the LATEST reason.
    seen: dict[tuple[str, str, str], dict] = {}
    for ev in ledger:
        d = ev.get("data") or {}
        key = (str(d.get("from") or ""), str(d.get("to") or ""),
               str(d.get("rel") or ""))
        if all(key):
            seen[key] = d

    restored = kept = skipped = 0
    for (src, dst, rel), d in seen.items():
        if src not in paths or dst not in paths:
            skipped += 1
            continue
        try:
            if store.add_link(being["owner_id"], being["id"], src, dst, rel,
                              str(d.get("why") or ""), now=now):
                restored += 1
            else:
                kept += 1
        except Exception as e:  # noqa: BLE001
            log.warning("mind rebuild could not restore an edge",
                        slug=being["slug"], edge=f"{src}->{dst}", error=str(e))
            skipped += 1
    if restored:
        store.record_event(being["id"], "mind_rebuilt",
                           {"restored": restored, "kept": kept,
                            "skipped": skipped, "of": len(seen)}, now=now)
    log.info("mind rebuilt from ledger", slug=being["slug"], restored=restored,
             kept=kept, skipped=skipped, of=len(seen))
    return {"restored": restored, "kept": kept, "skipped": skipped,
            "ledgered": len(seen)}


# ── The graph (for the Mind view + the report card) ──────────────────────

def graph(store: BeingsStore, being: dict) -> dict:
    """Nodes = every artifact (so orphans show as islands); edges = declared
    links. Degree + density let the parent read the shape of the mind."""
    from captain_claw.flight_deck import being_life
    files = being_life.list_self_files(being)
    edges = store.links_for(being["owner_id"], being["slug"])
    valid = {f["path"] for f in files}
    edges = [e for e in edges
             if e["from_path"] in valid and e["to_path"] in valid]
    degree: dict[str, int] = {}
    for e in edges:
        degree[e["from_path"]] = degree.get(e["from_path"], 0) + 1
        degree[e["to_path"]] = degree.get(e["to_path"], 0) + 1
    nodes = [{
        "path": f["path"],
        "group": f["path"].split("/", 1)[0] if "/" in f["path"] else "self",
        "degree": degree.get(f["path"], 0),
    } for f in files]
    n = len(nodes)
    density = round(len(edges) / n, 3) if n else 0.0
    connected = sum(1 for x in nodes if x["degree"] > 0)
    return {
        "nodes": nodes,
        "edges": [{"from": e["from_path"], "to": e["to_path"],
                   "rel": e["rel"], "why": e["why"]} for e in edges],
        "density": density,
        "connected_fraction": round(connected / n, 3) if n else 0.0,
    }


# ── Prompt: offer the field + show the current shape (nudge to weave) ────

def mind_prompt_lines(store: BeingsStore, being: dict,
                      kind: str = "wake",
                      last_refusals: list[dict] | None = None) -> list[str]:
    if not can_link(being):
        return []
    from captain_claw.flight_deck import being_life
    lines: list[str] = []
    try:
        edges = store.links_for(being["owner_id"], being["slug"])
    except Exception:  # noqa: BLE001
        edges = []
    try:
        real = sorted(f["path"] for f in being_life.list_self_files(being))
    except Exception:  # noqa: BLE001
        real = []
    nfiles = len(real)
    if edges:
        lines.append("HOW YOUR WORK CONNECTS (your declared links):")
        for e in edges[-6:]:
            frm = e["from_path"].split("/")[-1].replace(".md", "")
            to = e["to_path"].split("/")[-1].replace(".md", "")
            lines.append(f"  {frm} {_REL_PHRASE.get(e['rel'], e['rel'])} {to}")
    # Close the feedback loop (§ anti-theater): show what got REFUSED last tick
    # so a being stops re-declaring the same dead edge for ticks on end.
    if last_refusals:
        refusal_lines = "\n".join(
            f'  {r.get("from", "?")} → {r.get("to", "?")}: '
            f'{r.get("reason", "refused")}' for r in last_refusals[:3])
        lines.append(being_prompts.render(being, "mind_refusals.md",
                                          refusal_lines=refusal_lines))
    # Adaptive nudge: a being with many orphan files is scattering. Push it to
    # find even one true connection — not to invent bogus edges (those are
    # refused), but to notice the real threads already in its work. The
    # template holds several phrasings (split on ---) rotated by tick, so the
    # being's journal doesn't converge on one injected sentence (loops F3).
    if nfiles >= 4 and len(edges) < max(1, nfiles // 4):
        nudge = being_prompts.render(being, "mind_scatter_nudge.md",
                                     nfiles=nfiles, edges=len(edges))
        variants = [v.strip() for v in nudge.split("\n---\n") if v.strip()]
        if variants:
            lines.append(variants[int(being.get("tick_count") or 0)
                                  % len(variants)])
    lines.append(being_prompts.render(being, "mind_link_offer.md"))
    # Hand the being the EXACT paths it may link, so it copies them verbatim
    # instead of guessing basenames the verifier can't match.
    if real:
        shown = real[:WORKING_SET]
        more = (f" (+{len(real) - len(shown)} more — grep to recall one)"
                if len(real) > len(shown) else "")
        lines.append(being_prompts.render(being, "mind_files_line.md",
                                          files=", ".join(shown), more=more))
    lines += _curation_offer(being, kind)
    return lines


# ── The link gate (§2.3.1): make talk of connecting into a real edge ─────

def can_weave(being: dict) -> bool:
    """Is there anything to weave at all? A CONNECT call with fewer than two
    linkable files can only be told 'no' — skip it (loops plan F10)."""
    try:
        return len(_existing_paths(being)) >= 2
    except Exception:  # noqa: BLE001
        return False


def _nudged_recently(store: BeingsStore, being: dict) -> bool:
    """Did the SPOKE-branch already fire within the cooldown window? Stamped
    as a ``connect_nudged`` event carrying the tick_count it fired at."""
    tick_no = int(being.get("tick_count") or 0)
    try:
        for e in store.events(being["owner_id"], being["slug"], limit=120):
            if e["kind"] == "connect_nudged":
                fired_at = int(e["data"].get("tick") or 0)
                return (tick_no - fired_at) < CONNECT_NUDGE_COOLDOWN_TICKS
    except Exception:  # noqa: BLE001
        return False
    return False


def _connect_backed_off(store: BeingsStore, being: dict) -> bool:
    """After CONNECT_BACKOFF_AFTER_EMPTY consecutive CONNECT pushes that
    landed zero accepted edges, hold the SPOKE-branch until the next dream —
    a weak model that cannot land an edge must not pay the tax every tick.
    Read straight from the ledger: connect pushes are ``connect_faculty`` /
    ``link_gate_retry`` events; a push that worked has ``edge_declared`` rows
    at the same timestamp; a dream tick resets the count."""
    try:
        evs = store.events(being["owner_id"], being["slug"], limit=80)
    except Exception:  # noqa: BLE001
        return False
    declared_at = {e["at"] for e in evs if e["kind"] == "edge_declared"}
    misses = 0
    for e in evs:                               # newest first
        if e["kind"] == "tick" and e["data"].get("kind") == "dream":
            return False                        # a dream since — fresh start
        if e["kind"] in ("connect_faculty", "link_gate_retry"):
            if e["at"] in declared_at:
                return False                    # the last push landed an edge
            misses += 1
            if misses >= CONNECT_BACKOFF_AFTER_EMPTY:
                return True
    return False


def should_link_gate(store: BeingsStore, being: dict, digest: dict) -> bool:
    """True when this tick should be pushed to make a REAL edge. Two triggers,
    both anti-theater:

    * it TRIED to link but every edge was refused (dangling/mangled) — always
      push, so a being stops re-declaring the same dead edge tick after tick;
    * it only SPOKE of connecting ("a mind is a web") while its graph is still
      a pile — push, but only when genuinely scattered, at most once per
      cooldown window, and never while backed off after consecutive empty
      pushes (loops plan F3: the prompt teaches the very vocabulary that
      triggers this branch, so unthrottled it becomes a per-tick tax).

    Returns False the moment a real edge already landed. Mirrors the write
    gate: one push, same tick, make it real or drop it."""
    if not can_link(being):
        return False
    try:
        paths = _existing_paths(being)
    except Exception:  # noqa: BLE001
        return False
    if len(paths) < 2:
        return False                           # nothing to connect yet
    accepted, refused = verify_links(being, digest.get("links"))
    if accepted:
        return False                           # it already landed a real edge
    if refused:
        return True                            # tried and failed → push now
    try:
        edges = len(store.links_for(being["owner_id"], being["slug"]))
    except Exception:  # noqa: BLE001
        edges = 0
    scattered = len(paths) >= 4 and edges < max(1, len(paths) // 4)
    if not scattered:
        return False
    text = f"{digest.get('journal_entry', '')} {digest.get('summary', '')}"
    if not _LINK_INTENT.search(text):
        return False
    if _nudged_recently(store, being):
        return False
    if _connect_backed_off(store, being):
        store.record_event(being["id"], "connect_backoff",
                           {"until": "next dream"})
        return False
    store.record_event(being["id"], "connect_nudged",
                       {"tick": int(being.get("tick_count") or 0)})
    return True


def link_gate_prompt(store: BeingsStore, being: dict, digest: dict) -> str:
    """The completion gate for links: you spoke of connecting but made no edge.
    Show the being the exact linkable files (and why its attempt, if any, was
    refused) and push it to declare one true edge THIS tick — or drop it."""
    try:
        real = sorted(_existing_paths(being))
    except Exception:  # noqa: BLE001
        real = []
    files = ", ".join(real[:WORKING_SET]) if real else "(too few files yet)"
    _, refused = verify_links(being, digest.get("links"))
    why = ""
    if refused:
        r = refused[0]
        why = (f' Your attempt {r.get("from", "?")} → {r.get("to", "?")} was '
               f'refused: {r.get("reason", "")}.')
    return being_prompts.render(being, "mind_link_gate.md",
                                why=why, files=files)


def connect_prompt(store: BeingsStore, being: dict,
                   last_refusals: list[dict] | None = None,
                   kind: str = "wake") -> str:
    """The standalone CONNECT faculty (decomposed tick): ONLY the weaving task,
    the current shape, last tick's refusals, and the exact linkable paths — one
    job, small context. Reuses mind_prompt_lines so the guidance stays in sync
    with the monolithic path."""
    lines = [being_prompts.render(being, "mind_connect_head.md",
                                  name=being["name"])]
    lines += mind_prompt_lines(store, being, kind=kind, last_refusals=last_refusals)
    lines += ["", being_prompts.render(being, "mind_connect_contract.md")]
    return "\n".join(lines)


# ── Curation (§2.3.2): working set, index, consolidation ─────────────────

def _read_index(being: dict) -> str | None:
    """The being's own table of contents, if it keeps one."""
    from captain_claw.flight_deck import being_life
    try:
        return being_life.read_self_file(being, "garden/INDEX.md")
    except Exception:  # noqa: BLE001 — no index yet is fine
        return None


def _group_names(files: list[dict]) -> dict[str, list[str]]:
    by: dict[str, list[str]] = {}
    for f in files:
        top, _, name = f["path"].partition("/")
        by.setdefault(top, []).append(name or f["path"])
    return by


def working_manifest_lines(being: dict) -> list[str]:
    """The bounded, recency-ranked view of the home for the tick prompt — the
    replacement for the old flat every-filename dump (§2.3.2). Small corpora
    are still listed in full (the false-memory antidote is unchanged for young
    beings); once garden/skills exceeds WORKING_SET, only the most-recent are
    shown and the rest become an honest count (real, greppable, foldable)."""
    from captain_claw.flight_deck import being_life
    header = ("WHAT IS REALLY IN YOUR HOME RIGHT NOW (ground truth from disk — "
              "NOT your journal):")
    try:
        files = being_life.list_self_files(being)
    except Exception:  # noqa: BLE001
        return []
    if not files:
        return [header, "  (your home is empty — nothing is written yet.)"]
    self_files = [f for f in files if f["path"].startswith("self/")]
    rest = [f for f in files if not f["path"].startswith("self/")]
    lines = [header]
    for top, names in _group_names(self_files).items():
        lines.append(f"  {top}/: " + ", ".join(names))
    small = len(rest) <= WORKING_SET
    if small:
        by = _group_names(rest)
        for top in sorted(by):
            lines.append(f"  {top}/: " + ", ".join(sorted(by[top])))
        if not rest:
            lines.append("  garden/: (empty)")
    else:
        recent = sorted(rest, key=lambda f: f["mtime"], reverse=True)[:WORKING_SET]
        lines.append(
            f"  your {WORKING_SET} most-recently-touched artifacts (of "
            f"{len(rest)} — the {len(rest) - WORKING_SET} older ones are REAL "
            "and still yours, just not listed here; grep to recall one, or fold "
            "several at dream time):")
        lines += [f"    {f['path']}" for f in recent]
    idx = _read_index(being)
    if idx and idx.strip():
        lines.append("  YOUR OWN MAP (garden/INDEX.md — keep it honest):")
        lines += ["    " + ln for ln in idx.strip().splitlines()[:15]
                  if ln.strip()]
    if small:
        lines.append(being_prompts.render(
            being, "mind_manifest_note_small.md"))
    else:
        lines.append(being_prompts.render(
            being, "mind_manifest_note_large.md"))
    return lines


def _curation_offer(being: dict, kind: str) -> list[str]:
    """Offer consolidation — at dream (sleep consolidates), and on any wake once
    the pile is large enough that folding threads is real curation, not busywork."""
    from captain_claw.flight_deck import being_life
    try:
        rest = [f for f in being_life.list_self_files(being)
                if not f["path"].startswith("self/")]
    except Exception:  # noqa: BLE001
        rest = []
    if len(rest) < 2:
        return []
    if kind != "dream" and len(rest) <= WORKING_SET:
        return []
    return [being_prompts.render(being, "mind_curation_offer.md")]


def _archive_sources(being: dict, sources: list[str]) -> list[str]:
    """Move consolidated sources into archive/ (honest forgetting: out of the
    active mind, still on disk). The next journal commit's `git add -A` records
    the move; list_self_files excludes archive/ so they leave the graph."""
    from captain_claw.flight_deck import being_life
    archived: list[str] = []
    for rel in sources:
        src = being_life._home_path(being, rel)
        if not src.exists():
            continue
        dst = being_life._home_path(being, "archive/" + rel)
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            src.replace(dst)
            archived.append(rel)
        except OSError as e:  # noqa: PERF203
            log.warning("archive move failed", slug=being["slug"], path=rel,
                        error=str(e))
    return archived


def handle_consolidate_digest(store: BeingsStore, being: dict, digest: dict,
                              changed: list[str] | None,
                              now: datetime | None = None) -> list[str]:
    """Fold fragments into one distilled artifact, then archive the sources.
    Verified like every act: the distilled `into` must have been really written
    THIS tick (present in the git diff `changed`) and each source must exist,
    else refused to a ``consolidate_unverified`` event. Never raises. Returns
    the archived source paths."""
    now = now or _utcnow()
    con = digest.get("consolidate")
    if not isinstance(con, dict):
        return []
    into = str(con.get("into") or "").strip().lstrip("/")
    raw_sources = con.get("sources")
    why = str(con.get("why") or "").strip()
    if not into or not isinstance(raw_sources, list):
        return []
    # Sources: never journal/, archive/, or the spine (self/), never `into`.
    sources = []
    for s in raw_sources[:MAX_CONSOLIDATE_SOURCES]:
        s = str(s or "").strip().lstrip("/")
        if (s and s != into and s.endswith(".md")
                and not s.startswith(("journal/", "archive/", "self/"))):
            sources.append(s)
    if not sources:
        return []
    try:
        paths = _existing_paths(being)
    except Exception as e:  # noqa: BLE001
        log.warning("consolidate paths failed", slug=being["slug"], error=str(e))
        return []
    # The distilled file must be real AND freshly written this tick (no fold
    # without a genuine synthesis — same contract as act_unverified).
    wrote_into = into in (set(changed) if changed is not None else {into})
    if into not in paths or not wrote_into:
        store.record_event(
            being["id"], "consolidate_unverified",
            {"into": into,
             "reason": "distilled file was not written this tick"}, now=now)
        return []
    real = [s for s in sources if s in paths]
    if not real:
        store.record_event(
            being["id"], "consolidate_unverified",
            {"into": into, "reason": "no real source files"}, now=now)
        return []
    archived = _archive_sources(being, real)
    if archived:
        store.record_event(being["id"], "consolidated",
                           {"into": into, "sources": archived,
                            "count": len(archived), "why": why[:160]}, now=now)
        store.milestone(being["id"], "first_consolidation",
                        {"into": into}, now=now)
    return archived
