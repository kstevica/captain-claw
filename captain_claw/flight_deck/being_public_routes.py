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

from captain_claw.flight_deck import being_genome as genome_mod
from captain_claw.flight_deck import being_life, being_mind
from captain_claw.flight_deck.beings import (
    PUBLIC_MSG_MAX_CHARS,
    BeingError,
    get_store,
)

# Under /fd/ so the SPA catch-all treats it as an API path (a 404 stays a 404,
# never index.html), but with NO get_current_user dependency anywhere.
router = APIRouter(prefix="/fd/public/beings", tags=["beings-public"])


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
    """The curated, parent-approved public face of a being. Deliberately omits
    the wallet, ledger, allowance, drives arithmetic and owner — the square
    shows who a being IS and how it's doing, not its finances."""
    g = being.get("genome") or {}
    attrs = genome_mod.effective_attributes(g)
    affect = being.get("affect") or {}
    return {
        "slug": being["slug"],
        "name": being["name"],
        "stage": being["stage"],
        "state": being["state"],
        "generation": g.get("generation", 1),
        "born_at": being["born_at"],
        "hatched_at": being["hatched_at"],
        "died_at": being["died_at"],
        "voice": g.get("voice_seed", ""),
        "interests": g.get("interest_seeds", []),
        "temperament": attrs,
        "mood": affect.get("mood", ""),
        "tick_interval_minutes": being.get("tick_interval_minutes"),
        "stats": store.public_stats(being["id"]),
    }


@router.get("")
async def list_public_beings():
    """The square's roster — every public being across every family — plus the
    village's own description (the parent's words, set on the Beings page)."""
    store = get_store()
    return {"beings": [_public_profile(store, b) for b in store.public_beings()],
            "village": store.public_village()}


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
