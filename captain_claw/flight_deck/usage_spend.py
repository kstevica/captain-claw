"""What each user's agents spent on LLMs: this month and all time.

Neither of Flight Deck's own counters can answer that. ``daily_usage``
(usage_quota) holds tokens against the day's allowance and a free reset wipes
the day; ``cost_ledger`` holds only Basna, Vatra, Code and being runs. The full
record is each agent's own ``llm_usage`` table: every call, with its model and
its cache split.

So this reads those tables — read only, past the last row taken from each agent
(``usage_spend_sync``) — into one permanent ledger per user, day and model
(``usage_spend``). The first pass brings in the history the agents still hold,
and what was taken outlives the agent. Docker agents are read the same way:
their sessions folder is bind-mounted from ``DATA_DIR/<slug>``. Nothing changes
on the agents, so none needs a restart.

Dollars are worked out when read, from the curated price table (``pricing``):
list prices, so what a subscription or a negotiated rate really costs isn't
known here. A price added later also prices the past, and a model with no price
(a local one, usually) stays tokens, never a wrong number.

The ledger bills an agent's owner, a shared agent's member turns included:
``llm_usage`` doesn't say whose turn a call was. Calls on someone's own key
(BYOK) cost the deck nothing and are kept apart. An agent no owner can be named
for is left out, and so are LLM calls Flight Deck makes itself, outside any
agent.

    GET /fd/admin/usage/spend   admin   every user's spend, month and all time
"""

from __future__ import annotations

import asyncio
import sqlite3
import time
from pathlib import Path
from urllib.parse import quote

from fastapi import APIRouter, Depends

from captain_claw.flight_deck import pricing
from captain_claw.flight_deck.admin_routes import require_admin
from captain_claw.flight_deck.auth import get_db
from captain_claw.flight_deck.usage_quota import today
from captain_claw.llm import input_tokens_total
from captain_claw.logging import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/fd/admin/usage", tags=["usage-spend"])

SYNC_EVERY_S = 900.0     # the background pass; an agent removed between two loses that much
_FRESH_S = 60.0          # opening the Users page reads the agents again after this long
_READ_WAIT_S = 5.0       # the page waits this long for a pass; the first can take longer
_OPEN_TIMEOUT_S = 2.0    # an agent writing its sessions.db holds a read back this long at most
_UNPRICED_NAMED = 5      # how many models with no price a user's totals name

_sync_lock = asyncio.Lock()
_sync_task: asyncio.Task | None = None
_last_sync = 0.0


def _agents() -> dict[str, tuple[str, Path]]:
    """``{slug: (owner, its sessions.db)}`` for every agent folder on the deck."""
    from captain_claw.flight_deck import server as srv

    registry = srv._load_process_registry()
    out: dict[str, tuple[str, Path]] = {}
    for path in srv.DATA_DIR.glob("*/data/sessions/sessions.db"):
        slug = path.parents[2].name
        entry = registry.get(slug)
        owner = str(entry.get("owner") or "") if isinstance(entry, dict) else ""
        owner = owner or srv._folder_owner(slug)
        if owner:
            out[slug] = (owner, path)
    return out


def _read_calls(path: Path, after: int) -> tuple[list[dict], int]:
    """An agent's calls past rowid *after*, summed per day, model and byok flag,
    and the rowid read up to. Rowids only grow, so nothing is taken twice or
    skipped; one lower than *after* is a new database under the same name."""
    conn = sqlite3.connect(f"file:{quote(str(path))}?mode=ro", uri=True, timeout=_OPEN_TIMEOUT_S)
    try:
        newest = conn.execute("SELECT MAX(rowid) FROM llm_usage").fetchone()[0] or 0
        if newest < after:
            after = 0
        if newest == after:
            return [], newest
        # An agent that last ran before own keys existed has no such column.
        columns = {row[1] for row in conn.execute("PRAGMA table_info(llm_usage)")}
        byok_column = "byok" if "byok" in columns else "0"
        calls = conn.execute(
            f"SELECT substr(created_at, 1, 10), provider, model, {byok_column}, SUM(prompt_tokens),"
            " SUM(completion_tokens), SUM(cache_creation_input_tokens),"
            " SUM(cache_read_input_tokens), COUNT(*)"
            " FROM llm_usage WHERE rowid > ? AND rowid <= ? GROUP BY 1, 2, 3, 4",
            (after, newest),
        ).fetchall()
    finally:
        conn.close()
    merged: dict[tuple[str, str, int], dict] = {}
    for day, provider, model, byok, prompt, output, created, cached, count in calls:
        created, cached = created or 0, cached or 0
        # Whether prompt_tokens already holds the cache depends on the provider.
        fresh = input_tokens_total(provider, {
            "prompt_tokens": prompt, "cache_creation_input_tokens": created,
            "cache_read_input_tokens": cached,
        }) - created - cached
        row = merged.setdefault((day, model or "", int(byok or 0)), {
            "day": day, "model": model or "", "byok": int(byok or 0), "input_tokens": 0,
            "output_tokens": 0, "cache_write_tokens": 0, "cache_read_tokens": 0, "calls": 0,
        })
        row["input_tokens"] += max(0, fresh)
        row["output_tokens"] += output or 0
        row["cache_write_tokens"] += created
        row["cache_read_tokens"] += cached
        row["calls"] += count
    return list(merged.values()), newest


async def sync() -> int:
    """Take every agent's new LLM calls into the ledger; returns how many. An
    agent that can't be read now (busy, or no such table) waits for the next pass."""
    global _last_sync
    async with _sync_lock:
        db = get_db()
        marks = await db.get_usage_spend_marks()
        taken = 0
        for slug, (owner, path) in sorted((await asyncio.to_thread(_agents)).items()):
            mark = marks.get(slug, 0)
            try:
                rows, newest = await asyncio.to_thread(_read_calls, path, mark)
            except sqlite3.Error as exc:
                log.debug("Spend: agent not read this pass", slug=slug, error=type(exc).__name__)
                continue
            if newest != mark:
                await db.add_usage_spend(slug, owner, rows, newest)
                taken += sum(r["calls"] for r in rows)
        _last_sync = time.monotonic()
    if taken:
        log.info("Spend ledger read the agents' calls", calls=taken)
    return taken


async def sync_loop(stop: asyncio.Event) -> None:
    """A pass a minute after start, then every SYNC_EVERY_S."""
    delay = 60.0
    while not stop.is_set():
        try:
            await asyncio.wait_for(stop.wait(), timeout=delay)
            return
        except TimeoutError:
            pass
        delay = SYNC_EVERY_S
        try:
            await sync()
        except Exception as exc:
            log.warning("Spend ledger pass failed", error=str(exc))


async def _catch_up() -> None:
    """Read the agents again if the last pass is old, without holding the page
    up for long: a pass that takes longer goes on in the background."""
    global _sync_task
    if time.monotonic() - _last_sync < _FRESH_S:
        return
    if _sync_task is None or _sync_task.done():
        _sync_task = asyncio.create_task(sync())
    try:
        await asyncio.wait_for(asyncio.shield(_sync_task), timeout=_READ_WAIT_S)
    except TimeoutError:
        pass
    except Exception as exc:
        log.warning("Spend ledger pass failed", error=str(exc))


def _blank() -> dict:
    return {"usd": 0.0, "tokens": 0, "unpriced_tokens": 0, "own_usd": 0.0, "own_tokens": 0}


async def totals(db) -> dict:
    """Each user's spend this UTC month and all time, priced now. ``usd`` and
    ``tokens`` are on the deck's keys; ``own_*`` on their own; ``tokens`` are
    what missed the prompt cache, as the plan's allowance counts them."""
    month = today()[:7]
    users: dict[str, dict] = {}
    unpriced: dict[str, dict[str, int]] = {}
    first = ""
    for r in await db.list_usage_spend(f"{month}-01"):
        tokens = r["input_tokens"] + r["output_tokens"] + r["cache_write_tokens"]
        usd = pricing.cost_from_usage(r["model"], {
            "prompt_tokens": r["input_tokens"], "completion_tokens": r["output_tokens"],
            "cache_creation_input_tokens": r["cache_write_tokens"],
            "cache_read_input_tokens": r["cache_read_tokens"],
        })["usd"]
        first = min(first or r["first_day"], r["first_day"])
        user = users.setdefault(r["user_id"], {"month": _blank(), "all": _blank()})
        for period in (user["all"], user["month"]) if r["in_month"] else (user["all"],):
            if r["byok"]:
                period["own_usd"] += usd or 0.0
                period["own_tokens"] += tokens
            elif usd is None:
                period["unpriced_tokens"] += tokens
            else:
                period["usd"] += usd
            if not r["byok"]:
                period["tokens"] += tokens
        if usd is None and tokens:
            held = unpriced.setdefault(r["user_id"], {})
            name = r["model"] or "unnamed model"
            held[name] = held.get(name, 0) + tokens
    for uid, user in users.items():
        for period in (user["month"], user["all"]):
            period["usd"] = round(period["usd"], 6)
            period["own_usd"] = round(period["own_usd"], 6)
        held = unpriced.get(uid, {})
        user["unpriced_models"] = sorted(held, key=held.__getitem__, reverse=True)[:_UNPRICED_NAMED]
    return {"month": month, "since": first, "users": users}


@router.get("/spend")
async def get_spend(_admin: dict = Depends(require_admin)) -> dict:
    """Every user's spend on LLMs, this month and all time (admin only)."""
    await _catch_up()
    return await totals(get_db())
