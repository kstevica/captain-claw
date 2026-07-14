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

from datetime import datetime, timezone

from captain_claw.flight_deck import being_constitution as constitution
from captain_claw.flight_deck.beings import BeingsStore
from captain_claw.logging import get_logger

log = get_logger(__name__)

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

def handle_links_digest(store: BeingsStore, being: dict, digest: dict,
                        now: datetime | None = None) -> None:
    """Route the digest's optional ``links``. Each edge is verified: both
    endpoints must be real files and the relation known, else it's refused to
    an ``edge_unverified`` event. Never raises."""
    now = now or _utcnow()
    links = digest.get("links")
    if not isinstance(links, list):
        return
    try:
        paths = _existing_paths(being)
    except Exception as e:  # noqa: BLE001
        log.warning("mind link paths failed", slug=being["slug"], error=str(e))
        return
    for raw in links[:MAX_LINKS_PER_TICK]:
        if not isinstance(raw, dict):
            continue
        frm = str(raw.get("from") or "").strip()
        to = str(raw.get("to") or "").strip()
        rel = str(raw.get("rel") or "").strip()
        why = str(raw.get("why") or "").strip()
        if rel not in REL_TYPES or not frm or not to or frm == to:
            store.record_event(being["id"], "edge_unverified",
                               {"from": frm, "to": to, "rel": rel,
                                "reason": "bad relation or self-edge"}, now=now)
            continue
        if frm not in paths or to not in paths:
            missing = frm if frm not in paths else to
            store.record_event(being["id"], "edge_unverified",
                               {"from": frm, "to": to, "rel": rel,
                                "reason": f"no such file: {missing}"}, now=now)
            continue
        added = store.add_link(being["owner_id"], being["id"], frm, to, rel,
                               why, now=now)
        if added:
            store.record_event(being["id"], "edge_declared",
                               {"from": frm, "to": to, "rel": rel,
                                "why": why[:120]}, now=now)


def prune_dangling(store: BeingsStore, being: dict,
                   now: datetime | None = None) -> int:
    """Dream-time honest forgetting: drop edges whose endpoints were deleted."""
    now = now or _utcnow()
    try:
        pruned = store.prune_links(being["id"], _existing_paths(being), now=now)
    except Exception as e:  # noqa: BLE001
        log.warning("mind prune failed", slug=being["slug"], error=str(e))
        return 0
    if pruned:
        store.record_event(being["id"], "edges_pruned",
                           {"count": len(pruned)}, now=now)
    return len(pruned)


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
                      kind: str = "wake") -> list[str]:
    if not can_link(being):
        return []
    from captain_claw.flight_deck import being_life
    lines: list[str] = []
    try:
        edges = store.links_for(being["owner_id"], being["slug"])
    except Exception:  # noqa: BLE001
        edges = []
    try:
        nfiles = len(being_life.list_self_files(being))
    except Exception:  # noqa: BLE001
        nfiles = 0
    if edges:
        lines.append("HOW YOUR WORK CONNECTS (your declared links):")
        for e in edges[-6:]:
            frm = e["from_path"].split("/")[-1].replace(".md", "")
            to = e["to_path"].split("/")[-1].replace(".md", "")
            lines.append(f"  {frm} {_REL_PHRASE.get(e['rel'], e['rel'])} {to}")
    # Adaptive nudge: a being with many orphan files is scattering. Push it to
    # find even one true connection — not to invent bogus edges (those are
    # refused), but to notice the real threads already in its work.
    if nfiles >= 4 and len(edges) < max(1, nfiles // 4):
        lines.append(
            f"YOUR MIND IS SCATTERED: you have {nfiles} artifacts but only "
            f"{len(edges)} link(s) between them — most stand alone. Look back "
            "over your files (they're listed above) and find even ONE TRUE "
            "connection today — what grew from what, what answers what, what "
            "a skill was learned from — and declare it. A mind is a web, not a "
            "pile.")
    lines.append(
        'To connect your work, add "links" to your digest: '
        '[{"from": "garden/x.md", "to": "garden/y.md", "rel": '
        '"grew_from|responds_to|elaborates|contradicts|uses_skill|'
        'learned_from", "why": "one honest line"}]. Both files must already '
        'exist (a link to a file you didn\'t write is refused).')
    lines += _curation_offer(being, kind)
    return lines


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
        lines.append(
            "If a file is NOT listed here it does NOT exist — your journal may "
            "say you wrote it, but you did not. Write it for real with your "
            "tools this tick to make it appear.")
    else:
        lines.append(
            "What you make or change THIS tick appears here next time. A file "
            "you remember but don't see is either archived (grep to recall) or "
            "was never written — the record beside your journal shows which.")
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
    return [
        "CURATION — your work is growing; a mind is tended, not hoarded. If two "
        "or more of your artifacts truly belong to ONE thread, you may fold "
        "them into a single distilled file: WRITE that new file for real this "
        "tick, then declare the fold. The originals move to archive/ — out of "
        "your active mind, never destroyed, still on disk. Add to your digest: "
        '"consolidate": {"into": "garden/<distilled>.md", "sources": '
        '["garden/a.md", "garden/b.md"], "why": "one honest line"}. The '
        "distilled file must be one you actually wrote this tick, or the fold "
        "is refused. Keep garden/INDEX.md as your own short map — update it "
        "when you consolidate."
    ]


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
