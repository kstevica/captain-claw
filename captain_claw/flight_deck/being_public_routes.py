"""Iskra public square — the ONE un-gated surface of the life layer (§9).

Every route here is deliberately auth-free: a stranger with no account can see
the beings their parents flagged ``public``, read their files / journal / mind,
and leave a short note. That is the whole point — a window into a living being
that anyone on the internet can look through.

Because there is no ``user`` to scope by, these routes resolve beings by slug
through :meth:`BeingsStore.get_public`, which returns a being ONLY if it is
flagged public. A visitor can never reach a private being, another owner's
private data, or any mutation beyond leaving a (rate-limited, 64-char) note.

Kept intentionally small and read-mostly. The one write — POST a note — mints
nothing, spends nothing, and is capped so a public page can't be weaponised
into unbounded token burn on the parent's wallet.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, WebSocket
from pydantic import BaseModel

from captain_claw.flight_deck import being_federation, being_life, being_mind
from captain_claw.flight_deck.beings import (
    PUBLIC_MSG_MAX_CHARS,
    BeingError,
    get_store,
)

# Under /fd/ so the SPA catch-all treats it as an API path (a 404 stays a 404,
# never index.html), but with NO get_current_user dependency anywhere.
router = APIRouter(prefix="/fd/public/beings", tags=["beings-public"])
# Federation (§9.1): the announce handshake + read/write proxy for visiting
# beings that live on other machines. Separate prefix so visitor ids never
# collide with the /{slug} routes above.
village_router = APIRouter(prefix="/fd/public/village", tags=["village-public"])


def _run(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except BeingError as e:
        raise HTTPException(e.status, str(e)) from e


class PublicMessageRequest(BaseModel):
    name: str
    body: str
    thread_id: str | None = None


def _public_profile(store, being: dict) -> dict:
    """The curated, parent-approved public face of a being (shared builder in
    being_life so senders can ship the same snapshot to a host village)."""
    return being_life.public_profile(store, being)


# Backwards-compatible alias for the test that imports it.
def _latest_thought(store, being: dict) -> dict | None:
    return being_life.latest_thought(store, being)


@router.get("")
async def list_public_beings():
    """The square's roster — every public being across every family — plus the
    village's own description and its VISITORS (beings from other machines)."""
    store = get_store()
    return {
        "beings": [_public_profile(store, b) for b in store.public_beings()],
        "village": store.public_village(),
        "visitors": [
            {"id": v["id"], "origin": v["origin"], "slug": v["slug"],
             "name": v["name"], "linked": being_federation.is_linked(v["id"]),
             **v["profile"]}
            for v in store.public_visitors(
                ttl_minutes=being_life.VISITOR_TTL_MINUTES)],
    }


@router.get("/{slug}")
async def public_being(slug: str):
    store = get_store()
    being = _run(store.get_public, slug)
    return _public_profile(store, being)


@router.get("/{slug}/files")
async def public_files(slug: str):
    """Every file in the being's home the parent can browse — full transparency
    (self/, garden/, skills/, assessments/ …); journal has its own viewer."""
    being = _run(get_store().get_public, slug)
    return {"files": being_life.list_self_files(being)}


@router.get("/{slug}/file")
async def public_file(slug: str, path: str):
    being = _run(get_store().get_public, slug)
    text = _run(being_life.read_self_file, being, path)
    return {"path": path, "text": text}


@router.get("/{slug}/journal")
async def public_journal(slug: str, date: str = ""):
    """One day's journal straight from the selfhood repo (default today, UTC)."""
    being = _run(get_store().get_public, slug)
    from datetime import datetime, timezone
    day = date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    try:
        p = being_life._home_path(being, f"journal/{day}.md")
        text = p.read_text(encoding="utf-8") if p.exists() else ""
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"journal read failed: {e}") from e
    return {"date": day, "text": text}


@router.get("/{slug}/graph")
async def public_graph(slug: str):
    """The Mind — the being's artifacts and the edges it declared between them."""
    store = get_store()
    being = _run(store.get_public, slug)
    return being_mind.graph(store, being)


@router.post("/{slug}/message")
async def post_public_message(slug: str, body: PublicMessageRequest):
    """Leave the being a short note (max 64 chars; your name is required). It
    starts or continues a thread; keep the returned thread_id to see its reply."""
    text = (body.body or "").strip()
    if len(text) > PUBLIC_MSG_MAX_CHARS:
        raise HTTPException(
            400, f"a note can be at most {PUBLIC_MSG_MAX_CHARS} characters")
    return _run(get_store().post_public_message, slug, body.name, text,
                body.thread_id)


@router.get("/{slug}/thread/{thread_id}")
async def public_thread(slug: str, thread_id: str):
    """Your own conversation with the being (your browser holds the thread id)."""
    return _run(get_store().public_thread, slug, thread_id)


# ── Federation: visitors from other machines, over WebSocket (§9.1) ──────
# The sender dials in (NAT-friendly) and holds the socket; we push data
# requests down it. All the machinery lives in being_federation.


@village_router.get("/map")
async def public_village_map():
    """Observer mode (village-world plan): the living isometric map for the
    public /village page — the fronting village's ground and streets, with
    only its PUBLIC beings placed on it and walking their courses. No owner
    to scope by; nothing here mutates. A private being never appears."""
    from datetime import datetime, timezone
    from captain_claw.flight_deck import being_world
    store = get_store()
    owner = store.public_village_owner()
    if not owner:
        return {"plot": being_world.PLOT_SIZE, "grid": {}, "terrain": {},
                "roads": [], "props": [], "places": [], "beings": []}
    _run(being_world.ensure_village, store, owner)
    slugs = {b["slug"] for b in store.public_beings()
             if b.get("owner_id") == owner}
    now = datetime.now(timezone.utc)
    return being_world.village_map_payload(store, owner, now=now,
                                           only_slugs=slugs)


# ── The visiting ghost (FPV plan Phase 3) ─────────────────────────────────
# Public visitors roam the village in first person and leave signed notes.
# Both writes are bounded hard: the village-wide sign cap, text/name length
# limits in the store, a per-village minimum interval here, and the presence
# cooldown per being. A visitor's wake only ever touches PUBLIC beings.

_last_public_note_at: dict[str, float] = {}
PUBLIC_NOTE_MIN_INTERVAL_S = 15.0


class PublicNoteRequest(BaseModel):
    x: int
    y: int
    text: str
    name: str


class PublicPresenceRequest(BaseModel):
    x: int
    y: int
    name: str = ""


@village_router.post("/notes")
async def public_plant_note(body: PublicNoteRequest):
    """A visitor plants a signed note in the grass. The Iskre find it as
    their feet carry them near — signed with the visitor's name."""
    import time
    store = get_store()
    owner = store.public_village_owner()
    if not owner:
        raise HTTPException(404, "no public village fronts this machine")
    name = (body.name or "").strip()
    if not name:
        raise HTTPException(400, "a visitor's sign needs a name")
    last = _last_public_note_at.get(owner, 0.0)
    if time.monotonic() - last < PUBLIC_NOTE_MIN_INTERVAL_S:
        raise HTTPException(429, "the grass needs a breath between signs")
    note = _run(store.add_village_note, owner, body.x, body.y, body.text,
                author=name, author_kind="visitor")
    _last_public_note_at[owner] = time.monotonic()
    # what a visitor learns back is only the sign itself
    return {"note": {k: note[k] for k in
                     ("id", "x", "y", "text", "author", "author_kind",
                      "created_at")}}


@village_router.post("/presence")
async def public_presence(body: PublicPresenceRequest):
    """A visiting ghost passes close to a PUBLIC being — one presence fact,
    per being, per cooldown. Private beings never feel a stranger."""
    from datetime import datetime, timezone
    from captain_claw.flight_deck import being_world
    store = get_store()
    owner = store.public_village_owner()
    if not owner:
        raise HTTPException(404, "no public village fronts this machine")
    slugs = {b["slug"] for b in store.public_beings()
             if b.get("owner_id") == owner}
    name = (body.name or "").strip()[:24] or "unnamed"
    felt = _run(being_world.presence_felt, store, owner, body.x, body.y,
                author=name, author_kind="visitor",
                now=datetime.now(timezone.utc), only_slugs=slugs)
    return {"felt": len(felt)}


class PublicGhostRequest(BaseModel):
    id: str
    x: int
    y: int
    name: str = ""


@village_router.post("/ghost")
async def public_ghost(body: PublicGhostRequest):
    """A visiting ghost's heartbeat (FPV plan Phase 5): report my spot,
    receive the other ghosts roaming this village — the parent and my
    fellow visitors. Shares ONE roster per village with the parent, so we
    all see each other. In-memory, $0, un-gated."""
    from captain_claw.flight_deck import being_world
    store = get_store()
    owner = store.public_village_owner()
    if not owner:
        raise HTTPException(404, "no public village fronts this machine")
    others = being_world.ghost_heartbeat(
        owner, body.id, kind="visitor", name=body.name,
        x=body.x, y=body.y)
    return {"ghosts": others}


@village_router.post("/ghost/leave")
async def public_ghost_leave(body: PublicGhostRequest):
    from captain_claw.flight_deck import being_world
    store = get_store()
    owner = store.public_village_owner()
    if owner:
        being_world.ghost_depart(owner, body.id)
    return {"ok": True}


@village_router.websocket("/link")
async def village_link(ws: WebSocket):
    """A sending village dials in, presents the secret, and stays connected so
    we can proxy browsing over its socket. Un-gated (the secret is the gate)."""
    await being_federation.village_link_ws(ws)


def _visitor(vid: str) -> dict:
    return _run(get_store().get_visitor, vid)   # 404 if unknown


@village_router.get("/visitors/{vid}")
async def visitor_profile(vid: str):
    """The visitor's profile — fetched LIVE over the link when its home machine
    is connected (so the detail page updates as the being ticks), falling back
    to the last cached snapshot when it's offline."""
    v = _visitor(vid)
    linked = being_federation.is_linked(vid)
    prof = v["profile"]
    if linked:
        try:
            prof = await being_federation.link_request(vid, "profile")
        except Exception:  # noqa: BLE001 — link hiccup → show the cached snapshot
            prof = v["profile"]
    return {**prof, "id": v["id"], "origin": v["origin"], "slug": v["slug"],
            "visitor": True, "last_seen": v["last_seen"], "linked": linked}


@village_router.get("/visitors/{vid}/files")
async def visitor_files(vid: str):
    _visitor(vid)
    return await being_federation.link_request(vid, "files")


@village_router.get("/visitors/{vid}/file")
async def visitor_file(vid: str, path: str):
    _visitor(vid)
    return await being_federation.link_request(vid, "file", path=path)


@village_router.get("/visitors/{vid}/journal")
async def visitor_journal(vid: str, date: str = ""):
    _visitor(vid)
    return await being_federation.link_request(vid, "journal", date=date)


@village_router.get("/visitors/{vid}/graph")
async def visitor_graph(vid: str):
    _visitor(vid)
    return await being_federation.link_request(vid, "graph")


@village_router.post("/visitors/{vid}/message")
async def visitor_message(vid: str, body: PublicMessageRequest):
    """Leave a note on a visitor — pushed down its link so its being weighs it on
    its own tick at home, just like a local note."""
    _visitor(vid)
    return await being_federation.link_request(
        vid, "message", name=body.name, body=body.body, thread_id=body.thread_id)


@village_router.get("/visitors/{vid}/thread/{thread_id}")
async def visitor_thread(vid: str, thread_id: str):
    _visitor(vid)
    return await being_federation.link_request(vid, "thread", thread_id=thread_id)
