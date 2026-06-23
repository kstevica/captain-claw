"""Efferent dispatch (Topic 2) + execution-outcome judge (Topic 3, auto side).

The Arbiter decides *what*; this module does it and then learns from how it went.

Dispatch hands an action to the user's strongest running agent — reusing Basna's
proven ``_dispatch_one`` (send an instruction, stream the turn to completion,
collect the final output + tool actions). It runs as a background task so the
heartbeat / approve request returns immediately; the action moves to ``dispatched``
now and to ``done`` when the turn finishes.

When it finishes, an LLM judge decides whether the result accomplished the intent
(gated by ``judge_mode`` ∈ auto/both and ``learning_enabled``) and records the
outcome into reliability — the same signal a human Approve/Reject produces, so
auto-fired actions learn too. The ledger always reflects what happened; reliability
only moves when the dials allow it.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Any

from captain_claw.flight_deck.autonomy import get_store, resolve_config

_log = logging.getLogger(__name__)

# Keep references to in-flight judge tasks so they aren't garbage-collected.
_BG_TASKS: set[asyncio.Task] = set()

_JUDGE_SYSTEM = (
    "You are a strict evaluator. Given an action the assistant was asked to carry "
    "out and the result it produced, decide whether the action was accomplished. "
    'Reply with ONLY JSON: {"success": true|false, "why": "one short sentence"}. '
    "Be honest: partial, refused, or error results are failures."
)


def _strongest_agent(user_id: str) -> dict[str, Any] | None:
    """The user's most capable running agent, or None if none are up."""
    try:
        from captain_claw.flight_deck.consciousness import _agent_rank, _user_agents

        agents = _user_agents(user_id)
        if not agents:
            return None
        return sorted(agents, key=_agent_rank, reverse=True)[0]
    except Exception:
        return None


def _grounding_suffix(action: dict[str, Any]) -> str:
    """If the action is about a surfaced event, hand the agent the REAL handle so
    it fetches by id instead of re-searching (and falsely reporting the item
    absent). Resolves ``payload.event_ref`` → the event's stored handle."""
    ref = str((action.get("payload") or {}).get("event_ref") or "")
    if not ref:
        return ""
    try:
        from captain_claw.flight_deck.events import get_store as _events_store
        ev = _events_store().get_event(ref)
    except Exception:
        ev = None
    if not ev:
        return ""
    md = ev.get("metadata") or {}
    src = str(ev.get("source") or "")
    # Built-in sources keep a precise, tool-named line. Each entry resolves the
    # item's id + the tool that opens it — the same shape a custom source's fetch
    # contract (_fetch_tool/_handle_id) provides, so the two paths converge below.
    if src == "gmail":
        handle_id = md.get("thread_id") or md.get("message_id") or ""
        fetch_tool = "your Gmail tool (get_thread)"
        what = f"a REAL email already in the inbox: from {md.get('from') or '?'}, " \
               f"subject \"{md.get('subject') or ev.get('summary') or ''}\""
        id_label = "thread id"
    elif src == "calendar":
        handle_id = md.get("event_id") or ""
        fetch_tool = "your Calendar tool (get_event)"
        what = "a REAL calendar event"
        id_label = "event id"
    else:
        # Custom source: use the fetch contract stamped at ingest (Theme A).
        handle_id = md.get("_handle_id") or ""
        ft = md.get("_fetch_tool") or ""
        if not handle_id or not ft:
            return (f"\n\nGROUNDING — this is about a real item the system already "
                    f"fetched: {ev.get('summary', '')}. Treat it as existing; do not "
                    f"deny it or claim you can't find it.")
        fetch_tool = f"the '{ft}' tool"
        what = f"a REAL item: {ev.get('summary', '')}"
        id_label = "id"
    if not handle_id:
        return (f"\n\nGROUNDING — this is about {what}. It exists; treat it as real "
                f"and do not deny it.")
    return (
        f"\n\nGROUNDING — this is about {what} ({id_label}={handle_id}). It exists. "
        f"If you need its content, open it with {fetch_tool} BY ID ({handle_id}) — "
        f"do not search and do not scan recent items. If the tool cannot open that "
        f"id, say you couldn't open it; never tell the user it doesn't exist."
    )


def _instruction_for(action: dict[str, Any]) -> str:
    """Render an action into a concrete instruction the agent can act on."""
    kind = str(action.get("kind") or "nudge")
    title = str(action.get("title") or "").strip()
    rationale = str(action.get("rationale") or "").strip()
    ground = _grounding_suffix(action)
    if kind == "nudge":
        return (f"[Autonomous nudge] Proactively reach out to the user now: {title}. "
                f"{rationale} Keep it brief and in their language.{ground}")
    if kind == "basna":
        return f"Run a Basna on: {title}"
    if kind == "materialize_schedule":
        return (f"[Autonomous task] Set up a scheduled task: {title}. {rationale} "
                f"Use your scheduling tool.")
    # run_prompt and anything else: treat as a task prompt.
    return f"[Autonomous task] {title}\n\n{rationale}{ground}".strip()


def should_auto_dispatch(cfg: dict[str, Any], action: dict[str, Any]) -> bool:
    """Whether this action may fire WITHOUT human approval, per the dials.

    - ``act_low_risk``: only kinds in ``low_risk_kinds`` AND risk == 'low'.
    - ``act``: any non-high-risk action (high-risk still gated unless approval is
      not required).
    - otherwise (off / propose): never.
    """
    # stop_run is a destructive safety action — always human-approved, never auto.
    if str(action.get("kind")) == "stop_run":
        return False
    # track is internal bookkeeping (an open loop, no external side effect): always
    # record it without approval, even under the 'propose' ceiling — otherwise every
    # soft request would need a click just to be remembered, defeating the point.
    if str(action.get("kind")) == "track":
        return True
    # tool_action auto-fires only when the user has GRANTED that action (or its
    # grant category) AND it's reversible + low-risk + not human-only — the
    # explicit trust hook. Everything else stays propose→approve.
    if str(action.get("kind")) == "tool_action":
        if not cfg.get("allow_auto_dispatch"):
            return False
        if str(cfg.get("autonomy_level") or "off") not in ("act_low_risk", "act"):
            return False
        from captain_claw.flight_deck.action_catalog import get_action
        payload = action.get("payload") or {}
        action_id = str(payload.get("action_id") or "")
        spec = get_action(action_id, str(action.get("user_id") or ""))
        if (not spec or spec.get("human_only")
                or spec.get("reversibility") not in ("read_only", "reversible")
                or spec.get("risk") != "low"):
            return False
        # Manual override: an explicit grant trusts it outright.
        granted = set(cfg.get("granted_actions") or [])
        if action_id in granted or spec.get("grant") in granted:
            return True
        # Earned trust (#3): auto-promote once this specific action's learned
        # reliability clears the bar over enough runs. Demotes automatically when
        # the weight falls back below threshold (fails count double in the weight).
        try:
            rel = get_store().reliability_for(str(action.get("user_id") or ""), "tool_action", action_id)
            if rel and float(rel.get("weight", 0)) >= float(cfg.get("trust_threshold", 0.85)) \
                    and int(rel.get("runs", 0)) >= int(cfg.get("trust_min_runs", 3)):
                return True
        except Exception:
            pass
        return False
    if not cfg.get("allow_auto_dispatch"):
        return False
    level = str(cfg.get("autonomy_level") or "off")
    risk = str(action.get("risk") or "normal")
    kind = str(action.get("kind") or "")
    if level == "act_low_risk":
        return kind in (cfg.get("low_risk_kinds") or []) and risk == "low"
    if level == "act":
        if risk == "high":
            return not cfg.get("high_risk_requires_approval", True)
        return True
    return False


async def _judge_outcome(
    user_id: str, action: dict[str, Any], output: str,
) -> dict[str, Any]:
    """LLM verdict on whether ``output`` accomplished ``action``. Returns
    ``{"success": bool, "why": str}``; defaults to failure if unparseable."""
    agent = _strongest_agent(user_id)
    if not agent:
        return {"success": False, "why": "no agent to judge with"}
    try:
        from captain_claw.games.remote_provider import RemoteLLMProvider
        from captain_claw.llm import Message

        provider = RemoteLLMProvider(
            host=agent["host"], port=agent["port"], auth=agent["auth"],
            name=agent.get("name", ""),
        )
        user_prompt = (
            f"Intended action ({action.get('kind')}): {action.get('title')}\n"
            f"Why: {action.get('rationale')}\n\n"
            f"Result the assistant produced:\n{(output or '(no output)')[:4000]}\n\n"
            "Did it accomplish the intent?"
        )
        resp = await provider.complete(
            messages=[
                Message(role="system", content=_JUDGE_SYSTEM),
                Message(role="user", content=user_prompt),
            ],
            temperature=0.0,
            max_tokens=200,
        )
    except Exception as exc:
        return {"success": False, "why": f"judge error: {exc}"}

    txt = (resp.content or "").strip()
    try:
        data = json.loads(txt)
    except (ValueError, TypeError):
        m = re.search(r"\{.*\}", txt, re.S)
        data = {}
        if m:
            try:
                data = json.loads(m.group(0))
            except (ValueError, TypeError):
                data = {}
    return {
        "success": bool(data.get("success", False)),
        "why": str(data.get("why", "") or "")[:500],
    }


async def _execute_and_judge(user_id: str, action: dict[str, Any], agent: dict[str, Any]) -> None:
    """Background: run the action on ``agent``, judge the result, learn, mark done.
    Never raises — it's fire-and-forget off the heartbeat / approve request."""
    store = get_store()
    aid = action["id"]
    store.log(user_id, "executing", f"{action.get('kind')} · {action.get('title')} → {agent.get('slug','?')}")
    try:
        from captain_claw.flight_deck.basna_routes import _dispatch_one

        res = await _dispatch_one(
            int(agent.get("port") or 0), str(agent.get("auth", "")),
            _instruction_for(action), 180.0,
        )
        output = str(res.get("output") or "").strip()
        cfg = resolve_config(user_id)
        learn = bool(cfg.get("learning_enabled")) and \
            str(cfg.get("judge_mode") or "both") in ("auto", "both")

        kind = str(action.get("kind") or "")
        if not res.get("ok"):
            success: bool | None = False
            note = str(res.get("error") or "execution failed")[:500]
        elif kind == "nudge":
            # A nudge SUCCEEDS by being delivered — its whole job is to reach the
            # user. LLM-judging "did it accomplish reaching out" wrongly fails good
            # nudges and would suppress them via reliability. Delivery == success.
            success = True
            note = output[:300] or "delivered"
            # Also push the nudge to the user's WhatsApp (the agent turn already
            # surfaces it in web chat; this fans it out to WhatsApp too).
            if output and cfg.get("nudge_to_whatsapp", True):
                try:
                    from captain_claw.flight_deck.whatsapp_bridge import _allowed_waids, push_to_waid
                    waids = list(_allowed_waids())
                    sent = 0
                    for waid in waids:
                        if await push_to_waid(waid, output):
                            sent += 1
                    if waids:
                        store.log(user_id, "nudge → whatsapp", f"sent to {sent}/{len(waids)} recipient(s)")
                except Exception as exc:
                    _log.debug("nudge whatsapp delivery failed: %s", exc)
            # If this nudge was reminding about a tracked follow-up, re-arm it
            # (escalate: sooner each time) — or retire it after enough nudges.
            fu_id = str((action.get("payload") or {}).get("follow_up_id") or "")
            if fu_id:
                _escalate_follow_up(store, user_id, fu_id, cfg)
        elif learn:
            verdict = await _judge_outcome(user_id, action, output)
            success = bool(verdict["success"])
            note = verdict["why"] or output[:300]
        else:
            success = None  # nothing judged it; record the result, no verdict
            note = output[:500]

        if learn and success is not None:
            store.record_outcome(
                user_id, str(action.get("kind") or "nudge"),
                str(action.get("domain") or "general"), bool(success),
                seed=float(cfg.get("reliability_seed", 0.6)),
            )
        outcome = None if success is None else ("success" if success else "fail")
        store.update_status(aid, "done", outcome=outcome, outcome_note=note)
        store.log(user_id, f"done: {outcome or 'unjudged'}",
                  f"{action.get('title')} — {note[:160]}",
                  "warn" if outcome == "fail" else "info")
        _log.info("dispatch: action %s done (outcome=%s)", aid, outcome)
    except Exception as exc:
        _log.warning("dispatch execute/judge failed for %s: %s", aid, exc)
        store.log(user_id, "error: dispatch crashed", f"{action.get('title')}: {exc}", "error")
        store.update_status(aid, "done", outcome="fail", outcome_note=f"dispatch error: {exc}"[:500])


def _escalate_follow_up(store: Any, user_id: str, fu_id: str, cfg: dict[str, Any]) -> None:
    """A reminder nudge for this follow-up just went out. Re-arm it to come due
    again — sooner each time (escalation) — or mark it 'stale' once it has been
    nudged ``followup_max_nudges`` times with no resolution."""
    try:
        from captain_claw.flight_deck.events import get_store as _events_store
        es = _events_store()
        fu = es.get_follow_up(fu_id)
        if not fu or fu.get("status") != "open":
            return
        new_count = int(fu.get("nudged_count", 0)) + 1  # this nudge
        max_nudges = int(cfg.get("followup_max_nudges", 4))
        if new_count >= max_nudges:
            es.touch_follow_up(fu_id, nudged=True)
            es.mark_follow_up(fu_id, "stale")
            store.log(user_id, "follow-up went stale",
                      f"{fu.get('summary', '')[:80]} — nudged {new_count}× with no resolution", "warn")
            return
        base = int(cfg.get("followup_default_days", 3))
        days = max(1, base - new_count)  # escalate: closer each time
        next_at = (datetime.now(timezone.utc) + timedelta(days=days)).isoformat()
        es.touch_follow_up(fu_id, follow_up_at=next_at, nudged=True)
        store.log(user_id, "follow-up re-armed", f"{fu.get('summary', '')[:80]} — next in {days}d")
    except Exception as exc:
        _log.debug("escalate follow-up failed: %s", exc)


async def _dispatch_track(user_id: str, action: dict[str, Any]) -> dict[str, Any]:
    """Record (or re-snooze) a tracked open loop — a soft reminder/request the
    arbiter chose to TRACK rather than act on. No agent, no external side effect:
    it just writes to the follow-ups list, due in ``follow_up_days``."""
    store = get_store()
    payload = action.get("payload") or {}
    cfg = resolve_config(user_id)
    days = payload.get("follow_up_days")
    if not isinstance(days, int) or days <= 0:
        days = int(cfg.get("followup_default_days", 3))
    next_at = (datetime.now(timezone.utc) + timedelta(days=days)).isoformat()
    from captain_claw.flight_deck.events import get_store as _events_store
    es = _events_store()
    fu_id = str(payload.get("follow_up_id") or "")
    target = fu_id
    if fu_id:
        fu = es.get_follow_up(fu_id)
        if fu and fu.get("status") == "open":
            es.touch_follow_up(fu_id, follow_up_at=next_at)
            note = f"re-snoozed '{fu.get('summary', '')[:60]}' · due in {days}d"
        else:
            note = "follow-up not found or not open"
    else:
        fu = es.add_follow_up(
            user_id,
            summary=str(payload.get("summary") or action.get("title") or "follow-up"),
            detail=str(payload.get("detail") or ""),
            source=str(payload.get("source") or ""),
            follow_up_at=next_at,
        )
        target = (fu or {}).get("id", "")
        note = (f"tracking '{str(payload.get('summary') or '')[:60]}' · due in {days}d"
                if fu else "already tracked")
    store.update_status(action["id"], "done", outcome="success", outcome_note=note[:500])
    store.log(user_id, "tracked follow-up", note)
    return {"ok": True, "target": target, "note": note}


async def _dispatch_stop_run(user_id: str, action: dict[str, Any]) -> dict[str, Any]:
    """Carry out an approved stop_run: hard-stop the targeted run in-process
    (no agent involved). Idempotent — stopping an already-finished run is a
    success (nothing left to stop)."""
    store = get_store()
    payload = action.get("payload") or {}
    system = str(payload.get("system") or "basna").lower()
    target = str(payload.get("target") or "").strip()
    if not target:
        store.update_status(action["id"], "rejected", outcome="fail",
                            outcome_note="stop_run had no target")
        return {"ok": False, "target": "", "note": "stop_run missing target"}
    try:
        if system == "council":
            from captain_claw.flight_deck.auth import get_db
            await get_db().update_council_session(target, user_id, status="cancelled")
            note = "council deliberation cancelled"
        else:
            from captain_claw.flight_deck.basna_routes import _cancel_basna_run
            res = await _cancel_basna_run(target, user_id)
            note = f"stopped {res.get('stopped_workers', 0)} worker(s)"
        store.update_status(action["id"], "done", outcome="success", outcome_note=note)
        store.log(user_id, f"stop_run: {system}", f"{target} — {note}")
        return {"ok": True, "target": target, "note": note}
    except Exception as exc:
        store.update_status(action["id"], "done", outcome="fail", outcome_note=str(exc)[:300])
        store.log(user_id, "error: stop_run failed", f"{system} {target}: {exc}", "error")
        return {"ok": False, "target": target, "note": str(exc)}


def _enrich_args_from_event(action_id: str, args: dict[str, Any], event_ref: str) -> dict[str, Any]:
    """Fill an action's args from the surfaced event's real handle so it acts on
    the actual item, not the arbiter's guess (Theme B). Today: a reply draft
    targets the email's true sender + threads under its subject."""
    if action_id != "mail.draft" or not event_ref:
        return args
    try:
        from captain_claw.flight_deck.events import get_store as _events_store
        ev = _events_store().get_event(event_ref)
    except Exception:
        ev = None
    if not ev or str(ev.get("source")) != "gmail":
        return args
    md = ev.get("metadata") or {}
    out = dict(args)
    # Recipient: the real sender (parse "Name <addr>"); never overwrite an explicit to.
    if not str(out.get("to") or "").strip():
        frm = str(md.get("from") or "")
        m = re.search(r"<([^>]+)>", frm)
        addr = (m.group(1) if m else frm).strip()
        if addr:
            out["to"] = addr
    # Subject: reply form of the original, if the arbiter didn't set one.
    if not str(out.get("subject") or "").strip():
        subj = str(md.get("subject") or "").strip()
        if subj:
            out["subject"] = subj if subj.lower().startswith("re:") else f"Re: {subj}"
    return out


async def _dispatch_tool_action(user_id: str, action: dict[str, Any]) -> dict[str, Any]:
    """Run a catalog action via the deterministic rail and record the grounded
    outcome (the ToolResult success — no LLM judge needed). The reverse handle for
    undo is captured in Phase 2 (this turn's follow-up)."""
    store = get_store()
    payload = action.get("payload") or {}
    action_id = str(payload.get("action_id") or "")
    args = dict(payload.get("args")) if isinstance(payload.get("args"), dict) else {}
    # Theme B: ground a reply draft on the real event so it targets the actual
    # sender/thread instead of whatever the arbiter guessed.
    args = _enrich_args_from_event(action_id, args, str(payload.get("event_ref") or ""))
    store.update_status(action["id"], "dispatched")
    from captain_claw.flight_deck.actions import run_action

    res = await run_action(user_id, action_id, args)
    ok = bool(res.get("ok"))
    note = str(res.get("content") or res.get("error") or "")[:500]

    # Capture the reverse handle (for one-tap undo) from the real result.
    if ok:
        from captain_claw.flight_deck import action_catalog
        spec = action_catalog.get_action(action_id, user_id)
        reverse = action_catalog.build_reverse(spec, res.get("content", "")) if spec else None
        if reverse:
            store.update_payload(action["id"], {"reverse": reverse})

    cfg = resolve_config(user_id)
    if cfg.get("learning_enabled") and str(cfg.get("judge_mode") or "both") in ("auto", "both"):
        from captain_claw.flight_deck.autonomy import reliability_key
        rk, rd = reliability_key(action)  # per-action-id trust bucket
        store.record_outcome(user_id, rk, rd, ok, seed=float(cfg.get("reliability_seed", 0.6)))
    store.update_status(action["id"], "done",
                        outcome="success" if ok else "fail", outcome_note=note)
    store.log(user_id, f"tool_action: {action_id}", note, "info" if ok else "warn")
    # ok=True means "executed" so the approve route doesn't re-queue; the ledger
    # outcome carries whether the tool itself succeeded.
    return {"ok": True, "target": action_id, "note": note}


async def dispatch_action(user_id: str, action: dict[str, Any]) -> dict[str, Any]:
    """Execute one action by handing it to the strongest agent, in the background.
    Marks the action ``dispatched`` and spawns the run+judge task. Returns
    ``{ok, target, note}`` — ok=False (note set) when no agent is reachable, and
    the action keeps its prior status for the caller to resolve."""
    if str(action.get("kind")) == "stop_run":
        return await _dispatch_stop_run(user_id, action)
    if str(action.get("kind")) == "tool_action":
        return await _dispatch_tool_action(user_id, action)
    if str(action.get("kind")) == "track":
        return await _dispatch_track(user_id, action)
    agent = _strongest_agent(user_id)
    if not agent:
        return {"ok": False, "target": "", "note": "no running agent to dispatch to"}
    store = get_store()
    store.update_status(action["id"], "dispatched", ref_id=agent.get("slug", ""))
    try:
        task = asyncio.create_task(_execute_and_judge(user_id, dict(action), agent))
        _BG_TASKS.add(task)
        task.add_done_callback(_BG_TASKS.discard)
    except RuntimeError:
        # No running loop (e.g. a sync test harness): execute inline best-effort.
        await _execute_and_judge(user_id, dict(action), agent)
    return {"ok": True, "target": agent.get("slug", ""), "note": ""}
