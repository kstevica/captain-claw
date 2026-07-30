"""Run-cost query endpoints — the read side of the ``cost_ledger``.

Every Basna/Vatra/Code run logs its priced usage at completion
(``db.log_run_cost``, the ``pricing.summarize`` output), but the table was
write-only over HTTP: the only way a client ever saw cost was the transient
``cost`` progress event or the synchronous execute response. This router
exposes the ledger per authenticated user so any client (Flight Deck or an
external app) can render cost history and totals.
"""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends, Query

from captain_claw.flight_deck.auth import get_current_user, get_db
from captain_claw.logging import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/fd/costs", tags=["costs"])


@router.get("")
async def list_costs(
    run_kind: str = Query("", description="filter by run kind (basna, vatra, code, …)"),
    ref: str = Query("", description="filter by owner_ref (e.g. a being id)"),
    since: str = Query("", description="ISO timestamp lower bound (inclusive)"),
    limit: int = Query(200, ge=1, le=1000),
    user: dict = Depends(get_current_user),
):
    """The caller's run costs, newest first.

    ``since``/``ref`` filter within the newest ``limit`` rows (the ledger is
    read newest-first, so raising ``limit`` widens the window). ``total_usd``
    sums only priced rows; ``priced`` says how many rows that was — a run on
    an unpriced (e.g. local) model logs usage with ``usd`` NULL.
    """
    db = get_db()
    rows = await db.list_run_costs(user["id"], limit=limit,
                                   run_kind=run_kind or None)
    out: list[dict] = []
    total_usd = 0.0
    priced = 0
    for r in rows:
        if since and str(r.get("at") or "") < since:
            continue
        if ref and str(r.get("owner_ref") or "") != ref:
            continue
        try:
            r["usage"] = json.loads(r.get("usage") or "{}")
        except (ValueError, TypeError):
            r["usage"] = {}
        if r.get("usd") is not None:
            total_usd += float(r["usd"] or 0.0)
            priced += 1
        out.append(r)
    return {"costs": out, "count": len(out), "priced": priced,
            "total_usd": round(total_usd, 6) if priced else None}
