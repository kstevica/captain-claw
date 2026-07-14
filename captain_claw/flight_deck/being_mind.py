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

def mind_prompt_lines(store: BeingsStore, being: dict) -> list[str]:
    if not can_link(being):
        return []
    lines: list[str] = []
    try:
        edges = store.links_for(being["owner_id"], being["slug"])
    except Exception:  # noqa: BLE001
        edges = []
    if edges:
        recent = edges[-6:]
        lines.append("HOW YOUR WORK CONNECTS (your declared links):")
        for e in recent:
            frm = e["from_path"].split("/")[-1].replace(".md", "")
            to = e["to_path"].split("/")[-1].replace(".md", "")
            lines.append(f"  {frm} {_REL_PHRASE.get(e['rel'], e['rel'])} {to}")
    lines.append(
        'OPTIONAL — connect your work: if something you make or read today '
        'grows from, responds to, elaborates, contradicts, uses, or was '
        'learned from an EXISTING file of yours, add "links": [{"from": '
        '"garden/x.md", "to": "garden/y.md", "rel": "grew_from", "why": '
        '"one line"}] to your digest. Weave; don\'t only scatter. Both files '
        'must already exist (a link to a file you didn\'t write is refused).')
    return lines
