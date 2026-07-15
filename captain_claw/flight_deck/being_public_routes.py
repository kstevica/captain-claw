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

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from captain_claw.flight_deck import being_life, being_mind
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
             "name": v["name"], **v["profile"]}
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


# ── Federation: visitors from other machines (§9.1) ──────────────────────

class AnnounceRequest(BaseModel):
    secret: str
    origin: str          # the sender machine's public base URL
    slug: str
    name: str = ""
    profile: dict = {}


@village_router.post("/announce")
async def announce_visit(body: AnnounceRequest):
    """A remote machine announces one of its beings as a visitor here. Gated by
    this village's secret; refreshes the cached snapshot + heartbeat. The being
    is never copied — only where to fetch it live (origin+slug) is kept."""
    store = get_store()
    owner = store.owner_by_secret(body.secret)
    if not owner:
        raise HTTPException(403, "invalid or missing village secret")
    origin = (body.origin or "").strip().rstrip("/")
    if not (origin.startswith("http://") or origin.startswith("https://")):
        raise HTTPException(400, "origin must be an http(s) URL")
    if not (body.slug or "").strip():
        raise HTTPException(400, "a visiting being needs a slug")
    v = _run(store.upsert_visitor, owner, origin, body.slug.strip(),
             body.name or body.slug, body.profile or {})
    return {"ok": True, "visitor_id": v["id"]}


def _visitor_origin(vid: str) -> tuple[dict, str]:
    v = _run(get_store().get_visitor, vid)
    origin = (v["origin"] or "").rstrip("/")
    if not (origin.startswith("http://") or origin.startswith("https://")):
        raise HTTPException(502, "visitor has an invalid origin")
    return v, origin


async def _relay(method: str, url: str, *, params=None, json=None):
    """Proxy one request to a visitor's home machine (only the fixed public
    being paths are ever built here, so the origin can't be steered elsewhere)."""
    import httpx
    try:
        async with httpx.AsyncClient(timeout=12, follow_redirects=False) as c:
            r = await c.request(method, url, params=params, json=json)
    except Exception:  # noqa: BLE001
        raise HTTPException(502, "the visitor's home village didn't answer")
    if r.status_code >= 400:
        try:
            detail = r.json().get("detail", "")
        except Exception:  # noqa: BLE001
            detail = ""
        raise HTTPException(502 if r.status_code >= 500 else r.status_code,
                            detail or "the visitor's home village returned an error")
    try:
        return r.json()
    except Exception:  # noqa: BLE001
        raise HTTPException(502, "the visitor sent back something unreadable")


@village_router.get("/visitors/{vid}")
async def visitor_profile(vid: str):
    """The visitor's cached profile snapshot (fast; refreshed by heartbeats)."""
    v, origin = _visitor_origin(vid)
    return {**v["profile"], "id": v["id"], "origin": origin, "slug": v["slug"],
            "visitor": True, "last_seen": v["last_seen"]}


@village_router.get("/visitors/{vid}/files")
async def visitor_files(vid: str):
    v, origin = _visitor_origin(vid)
    return await _relay("GET", f"{origin}/fd/public/beings/{v['slug']}/files")


@village_router.get("/visitors/{vid}/file")
async def visitor_file(vid: str, path: str):
    v, origin = _visitor_origin(vid)
    return await _relay("GET", f"{origin}/fd/public/beings/{v['slug']}/file",
                        params={"path": path})


@village_router.get("/visitors/{vid}/journal")
async def visitor_journal(vid: str, date: str = ""):
    v, origin = _visitor_origin(vid)
    return await _relay("GET", f"{origin}/fd/public/beings/{v['slug']}/journal",
                        params={"date": date} if date else None)


@village_router.get("/visitors/{vid}/graph")
async def visitor_graph(vid: str):
    v, origin = _visitor_origin(vid)
    return await _relay("GET", f"{origin}/fd/public/beings/{v['slug']}/graph")


@village_router.post("/visitors/{vid}/message")
async def visitor_message(vid: str, body: PublicMessageRequest):
    """Leave a note on a visitor — forwarded to its home machine (proxied write),
    so the being weighs it on its own tick just like a local note."""
    v, origin = _visitor_origin(vid)
    return await _relay("POST", f"{origin}/fd/public/beings/{v['slug']}/message",
                        json={"name": body.name, "body": body.body,
                              "thread_id": body.thread_id})


@village_router.get("/visitors/{vid}/thread/{thread_id}")
async def visitor_thread(vid: str, thread_id: str):
    v, origin = _visitor_origin(vid)
    return await _relay(
        "GET", f"{origin}/fd/public/beings/{v['slug']}/thread/{thread_id}")
