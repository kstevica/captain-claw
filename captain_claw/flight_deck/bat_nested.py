"""Bat — the runs a Bat worker starts under its run.

A Bat worker has the whole toolset, so a step can start a Basna or Vatra team, a
Code session or a Crew task. Each of those keeps its own accounts, so without this
module their spend never reached the Bat run's cap, nothing waited for their
results, and they kept working after the run had ended.

The start routes call :func:`adopt` with what Flight Deck already knows about the
caller (its port or slug), so a worker can't opt out and a model can't name another
run. From then on the driver's watch calls :func:`poll` every few seconds: it adds
what each child has spent to the run's persisted total, and when a child finishes,
closes it and hands its result to the step that started it. :func:`stop` ends the
open ones when the run stops.

Everything a child is asked lives behind one small reader per kind, so the routes
stay untouched beyond the adopt call.
"""

from __future__ import annotations

import time
from typing import Any

import structlog

from captain_claw.flight_deck import bat_loop

log = structlog.get_logger(__name__)

# A child that hasn't finished after this long is stopped, so a stuck one can't
# hold the judge back forever. Crew's own turn limit is 6 hours by default.
_CHILD_WAIT_S = 6 * 3600
_RESULT_CHARS = 6000

KINDS = ("basna", "vatra", "code", "crew")

# Who's working for which run, by what the start routes can see. In memory on
# purpose: a worker doesn't outlive the deck that spawned it.
_WORKERS: dict[str, tuple[str, str]] = {}
# What a worker's current turn has spent so far, not yet in the run's total:
# dollars, and tokens (a run can be capped in either).
_LIVE: dict[tuple[str, str], float] = {}
_LIVE_TOKENS: dict[tuple[str, str], int] = {}


def worker_started(run_id: str, step_key: str, *, port: int = 0, slug: str = "") -> None:
    if port:
        _WORKERS[f"port:{int(port)}"] = (run_id, step_key)
    if slug:
        _WORKERS[f"slug:{slug}"] = (run_id, step_key)


def worker_ended(run_id: str, step_key: str, *, port: int = 0, slug: str = "") -> None:
    _WORKERS.pop(f"port:{int(port)}", None)
    _WORKERS.pop(f"slug:{slug}", None)
    _LIVE.pop((run_id, step_key), None)
    _LIVE_TOKENS.pop((run_id, step_key), None)


def worker_of(*, port: int = 0, slug: str = "") -> tuple[str, str] | None:
    return (_WORKERS.get(f"slug:{slug}") if slug else None) or (_WORKERS.get(f"port:{int(port)}") if port else None)


def set_live(run_id: str, step_key: str, usd: float, tokens: int = 0) -> None:
    _LIVE[(run_id, step_key)] = float(usd or 0.0)
    _LIVE_TOKENS[(run_id, step_key)] = int(tokens or 0)


def live_usd(run_id: str) -> float:
    return sum(v for (rid, _), v in _LIVE.items() if rid == run_id)


def live_tokens(run_id: str) -> int:
    return sum(v for (rid, _), v in _LIVE_TOKENS.items() if rid == run_id)


def token_cap(run: dict) -> int:
    """A run's cap in tokens (``config.token_cap``; 0 = none), beside its dollar cap."""
    try:
        return max(0, int((run.get("config") or {}).get("token_cap") or 0))
    except (TypeError, ValueError):
        return 0


async def refusal(owner: str, *, port: int = 0, slug: str = "") -> str:
    """Why a Bat worker may not start another run now, or "". A worker of a run
    that has spent its budget gets told, instead of starting something the watch
    would stop seconds later."""
    who = worker_of(port=port, slug=slug)
    store = bat_loop.get_store()
    if not who or store is None:
        return ""
    run = await store.get_run(who[0])
    if not run or run["owner_id"] != owner:
        return ""
    cap = float(run.get("llm_usd_cap") or 0.0)
    if cap > 0 and float(run.get("cumulative_usd") or 0.0) + live_usd(run["id"]) >= cap:
        return f"This Bat run has spent its LLM budget (${cap:.2f}), so it can't start another run."
    tokens = token_cap(run)
    if tokens and int(run.get("cumulative_tokens") or 0) + live_tokens(run["id"]) >= tokens:
        return f"This Bat run has used its {tokens:,} tokens, so it can't start another run."
    return ""


async def adopt(kind: str, child_id: str, owner: str, *, port: int = 0, slug: str = "") -> None:
    """Tie a just-started run to the Bat run whose worker asked for it. A no-op for
    anyone else, and it never gets in the way of the start itself."""
    who = worker_of(port=port, slug=slug)
    store = bat_loop.get_store()
    if not who or store is None:
        return
    try:
        run = await store.get_run(who[0])
        if not run or run["owner_id"] != owner:
            return
        await store.add_child(run["id"], child_id, kind, who[1])
        await store.append_event(run["id"], "note", f"{kind} run started under this one: {child_id}", agent=who[1])
    except Exception as e:  # noqa: BLE001
        log.warning("bat adopt failed", child=child_id, error=str(e))


# ── what each kind of child can tell us ───────────────────────────────

def _usage_of(cost: dict | None) -> tuple[float, int]:
    cost = cost or {}
    return float(cost.get("usd") or 0.0), int((cost.get("tokens") or {}).get("total_tokens", 0) or 0)


async def _read_team(child_id: str, owner: str) -> dict[str, Any]:
    """Basna and Vatra share their session table and their usage accumulator."""
    from captain_claw.flight_deck import pricing
    from captain_claw.flight_deck.auth import get_db
    from captain_claw.flight_deck.basna_routes import _RUN_USAGE, _active_agent_runs

    db = get_db()
    entries = _RUN_USAGE.get(child_id)
    usd, tokens = _usage_of(pricing.summarize(entries)) if entries else await db.run_cost_total(owner, child_id)
    is_open = child_id in _active_agent_runs.get(owner, ())
    result = ""
    if not is_open:
        sess = await db.get_basna_session(child_id, owner) or {}
        result = (sess.get("truth") or "").strip() or f"It ended with no result (status: {sess.get('status') or 'unknown'})."
    return {"usd": usd, "tokens": tokens, "open": is_open, "result": result}


async def _read_code(child_id: str, owner: str) -> dict[str, Any]:
    from captain_claw.flight_deck import code_routes
    from captain_claw.flight_deck.auth import get_db

    cost = code_routes._run_cost(child_id)
    usd, tokens = _usage_of(cost) if cost else await get_db().run_cost_total(owner, child_id)
    is_open = child_id in code_routes._agent_code_active.get(owner, ())
    result = ""
    if not is_open:
        project, _, session_id = child_id.partition("/")
        try:
            _sess, _repo, sdir, _pk = code_routes._sctx(owner, project, session_id)
            said = [m for m in code_routes._read_chat(sdir) if m.get("role") == "assistant"]
            said = [m for m in said if m.get("kind") not in ("note", "approval")] or said
            result = (said[-1].get("text", "") if said else "").strip()
        except Exception as e:  # noqa: BLE001
            log.warning("bat read code child failed", child=child_id, error=str(e))
        result = result or "It ended with no result."
    return {"usd": usd, "tokens": tokens, "open": is_open, "result": result}


async def _read_crew(child_id: str, owner: str) -> dict[str, Any]:
    from captain_claw.flight_deck import crew_runner, crew_store

    task = crew_runner._store().get(owner, child_id) or {}
    is_open = task.get("status") in crew_store.ACTIVE
    result = ""
    if not is_open:
        report = (task.get("results") or {}).get("files_path") or ""
        result = "\n".join(p for p in ((task.get("final") or "").strip(), (task.get("note") or "").strip(),
                                       f"Its files are in {report}" if report else "") if p)
        result = result or f"It ended with no result (status: {task.get('status') or 'unknown'})."
    return {"usd": float(task.get("cost_usd") or 0.0),
            "tokens": int(task.get("tokens_in") or 0) + int(task.get("tokens_out") or 0),
            "open": is_open, "result": result}


async def _read(kind: str, child_id: str, owner: str) -> dict[str, Any]:
    if kind == "code":
        return await _read_code(child_id, owner)
    if kind == "crew":
        return await _read_crew(child_id, owner)
    return await _read_team(child_id, owner)


async def _stop_child(kind: str, child_id: str, owner: str) -> None:
    if kind == "code":
        from captain_claw.flight_deck.code_routes import CodeAgentSessionReq, agent_code_cancel
        project, _, session_id = child_id.partition("/")
        await agent_code_cancel(CodeAgentSessionReq(project=project, session_id=session_id),
                                auth_user={"id": owner})
    elif kind == "crew":
        from captain_claw.flight_deck import crew_runner
        await crew_runner.stop(owner, child_id, note="Stopped: the Bat run that started this one has stopped.")
    else:
        from captain_claw.flight_deck.basna_routes import _cancel_basna_run
        await _cancel_basna_run(child_id, owner)


# ── the driver's watch ────────────────────────────────────────────────

async def _count(store, run_id: str, child: dict, state: dict) -> None:
    """Add what the child spent since the last look. Counts only go up: a finished
    run's accumulator is gone, and its ledger row may not be there yet."""
    usd = max(0.0, state["usd"] - float(child["counted_usd"]))
    tokens = max(0, state["tokens"] - int(child["counted_tokens"]))
    if usd or tokens:
        await store.bump_cost(run_id, usd, tokens)
        await store.update_child(run_id, child["child_id"],
                                 counted_usd=float(child["counted_usd"]) + usd,
                                 counted_tokens=int(child["counted_tokens"]) + tokens)


async def _hand_result(store, run_id: str, child: dict, result: str) -> None:
    """The child's result joins the output of the step that started it: that's what
    later steps and the judge read. The worker that asked is long gone."""
    block = f"\n\n## Result of the {child['kind']} run {child['child_id']}\n{result[:_RESULT_CHARS]}"
    for s in await store.list_steps(run_id):
        if s["step_key"] == child["step_key"]:
            kept = (s.get("output") or "")[:max(0, bat_loop._OUTPUT_CAP - len(block))]
            await store.upsert_step(run_id, s["step_key"], output=kept + block)
            return


async def poll(store, run: dict) -> float:
    """Bring the run's spend up to date; returns what's in flight beyond that."""
    run_id, owner = run["id"], run["owner_id"]
    for child in await store.list_children(run_id, open_only=True):
        try:
            state = await _read(child["kind"], child["child_id"], owner)
        except Exception as e:  # noqa: BLE001 — one bad child can't blind the watch
            log.warning("bat read child failed", child=child["child_id"], error=str(e))
            continue
        await _count(store, run_id, child, state)
        if state["open"]:
            if time.time() - float(child["created_at"]) > _CHILD_WAIT_S:
                await store.append_event(run_id, "note", f"{child['kind']} run {child['child_id']} took too long: stopping it")
                await _stop_child(child["kind"], child["child_id"], owner)
                await _hand_result(store, run_id, child, "It was stopped for taking too long.")
                await store.update_child(run_id, child["child_id"], status="closed")
            continue
        await _hand_result(store, run_id, child, state["result"])
        await store.update_child(run_id, child["child_id"], status="closed")
        await store.append_event(run_id, "note", f"{child['kind']} run {child['child_id']} finished", agent=child["step_key"])
    return live_usd(run_id)


async def stop(store, run: dict) -> None:
    """End every open child, counting what they spent up to now."""
    run_id, owner = run["id"], run.get("owner_id", "")
    for child in await store.list_children(run_id, open_only=True):
        try:
            await _count(store, run_id, child, await _read(child["kind"], child["child_id"], owner))
        except Exception as e:  # noqa: BLE001
            log.warning("bat count child failed", child=child["child_id"], error=str(e))
        try:
            await _stop_child(child["kind"], child["child_id"], owner)
        except Exception as e:  # noqa: BLE001
            log.warning("bat stop child failed", child=child["child_id"], error=str(e))
        await store.update_child(run_id, child["child_id"], status="closed")
        await store.append_event(run_id, "note", f"stopped the {child['kind']} run {child['child_id']}", agent=child["step_key"])


bat_loop.set_spend_watch(poll, stop)
