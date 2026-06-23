"""The Arbiter — the decider that closes the autonomy loop (Topic 1 + Topic 4).

It runs inside the consciousness heartbeat, right after a reflection. Given the
candidate goals the system has surfaced to itself — the reflection's standing
intentions, its current thought, and the latest *agent* self-reflection bullets
(Topic 4: reflections → proposed work) — it ranks them into one concrete next
action and writes it to the action ledger.

Phase 2 runs in **propose** mode only (the shipped ceiling): every proposal lands
as ``awaiting_approval`` and waits for the human on the Autonomous Work page.
Dispatch (Topic 2) and learning feedback into selection (Topic 3) come later;
this module already *reads* learned reliability to suppress losing action kinds.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Any

from captain_claw.flight_deck.autonomy import get_store, resolve_config

_log = logging.getLogger(__name__)

# Action kinds the arbiter may propose (must match the ledger / later dispatch).
_KINDS = ("nudge", "run_prompt", "basna", "materialize_schedule", "stop_run", "tool_action", "track")
_RISKS = ("low", "normal", "high")

_SYSTEM_PROMPT = (
    "You are the Arbiter: you turn the assistant's own reflections and standing "
    "intentions into its single next concrete action on the user's behalf.\n\n"
    "From the candidate goals below, pick the ONE most useful thing to do now and "
    "express it as a concrete action. Lean toward proposing one action whenever a "
    "candidate suggests something genuinely helpful — a check-in or nudge to the "
    "user, a small piece of research, a draft, a reminder, a recurring task. Return "
    "an empty array [] ONLY if every candidate is pure internal musing or automated "
    "noise (e.g. an automated notification) with no value to the user.\n\n"
    "THREE ways to handle a candidate — choose deliberately:\n"
    "1. ACT NOW (kind nudge/run_prompt/tool_action/…): it's timely and warrants "
    "doing something this moment.\n"
    "2. TRACK (kind=\"track\"): it's a soft reminder, a soft request, or a "
    "'waiting on you' item that should NOT be forgotten but doesn't need action "
    "this instant (e.g. 'kindly reminding you of our offer — let us know your "
    "comments'). Record it as an open loop to revisit. Give \"follow_up_days\" = how "
    "many days until it should resurface (default 3). USE THIS for soft "
    "reminders/requests instead of dropping them — they must be tracked.\n"
    "3. DROP ([]): pure internal musing or automated noise with no user value.\n\n"
    "Reply with ONLY a JSON array of 0 or 1 objects:\n"
    '{"kind": one of ["nudge","run_prompt","basna","materialize_schedule","stop_run","tool_action","track"], '
    '"title": short imperative, "rationale": one sentence on why now, '
    '"risk": "low" | "normal" | "high", "domain": short slug e.g. "ops"/"research", '
    '"score": 0.0-1.0 how valuable/timely, '
    '"target": run session id (stop_run only), "system": "basna" (stop_run only), '
    '"action_id": catalog id (tool_action only), "args": {…} (tool_action only), '
    '"follow_up_days": int days until it resurfaces (track only), '
    '"follow_up_id": the FU:<id> of a due follow-up this addresses (when acting on '
    'or re-snoozing a "follow-up due" candidate — copy the id after "FU:"), '
    '"event_ref": the EV:<id> of the surfaced event this action is about (copy the '
    'id after "EV:" from the candidate — set it on ANY action derived from an '
    'event so the assistant can open the real item)}\n\n'
    "Kind/risk mapping (use these exact kinds): a proactive message to the user is "
    'kind="nudge", risk="low". Running a task with the agent\'s tools is '
    'kind="run_prompt" (risk "normal", or "high" if it sends/changes external data). '
    'A multi-agent research run is kind="basna". Setting up a recurring/scheduled job '
    'is kind="materialize_schedule". If a candidate maps cleanly onto one of the '
    'concrete actions in the "action catalog" below, prefer kind="tool_action" with '
    'its "action_id" and an "args" object filling that action\'s required fields '
    "(verbatim from the candidate — never invent facts like emails, times, or names). "
    'If a candidate shows a run is stuck, looping, or runaway AND it matches an '
    '"active run" below, propose kind="stop_run" with that run\'s session id as '
    '"target" (risk "normal"). stop_run and tool_action are held for approval.\n'
    "An '(event · … · EV:<id>)' candidate is a REAL item the system already "
    "fetched from the user's world (an email, a calendar entry). Treat its "
    "existence as a confirmed fact — set \"event_ref\" to its EV id and write the "
    "action as if it is true (it is). Do NOT ask the assistant to re-verify or "
    "search for it; the assistant will be handed the exact id to open.\n"
    "Some candidates are marked '(follow-up due …)' — these are open loops you "
    "TRACKed earlier that have come due. For one of these, either propose a "
    'kind="nudge" reminding the user (set "follow_up_id" to its FU:<id>; make the '
    "reminder MORE insistent the older it is / the more times it has been surfaced), "
    'or re-snooze it with kind="track" + "follow_up_id" + a new "follow_up_days".\n'
    "Score honestly: a genuinely useful action is ~0.7-0.9; score low only if you "
    "doubt it helps. A track is cheap and worth doing — score it ~0.6+. Don't invent "
    "busywork, and never duplicate the 'already proposed' list.\n\n"
    "Output ONLY the JSON array — no preamble, no reasoning, no markdown fences. "
    "Your reply must start with '[' and end with ']'."
)


def _in_quiet_hours(start: int, end: int) -> bool:
    """Is the current UTC hour within the quiet window (which may wrap midnight)?"""
    try:
        h = datetime.now(timezone.utc).hour
    except Exception:
        return False
    if start == end:
        return False
    if start < end:
        return start <= h < end
    return h >= start or h < end  # wraps midnight


def _gather_candidates(reflection: dict[str, Any], *, include_reflections: bool = True) -> list[str]:
    """Candidate goals from the reflection plus, when enabled (Topic 4), the
    latest agent self-reflection bullets. De-duped, trimmed, capped — just the
    raw material for ranking."""
    out: list[str] = []
    for i in reflection.get("intentions") or []:
        s = str(i).strip()
        if s:
            out.append(s)
    thought = str(reflection.get("thought") or "").strip()
    if thought:
        out.append(f"(current thought) {thought}")
    if include_reflections:
        try:
            from captain_claw.reflections import load_latest_reflection

            refl = load_latest_reflection()
            if refl and refl.summary:
                for line in str(refl.summary).splitlines():
                    line = line.strip().lstrip("-*0123456789. ").strip()
                    if len(line) > 8:
                        out.append(f"(self-reflection) {line}")
        except Exception:
            pass
    # De-dupe preserving order, cap the pool.
    seen: set[str] = set()
    uniq: list[str] = []
    for s in out:
        k = s.lower()
        if k not in seen:
            seen.add(k)
            uniq.append(s)
    return uniq[:20]


def _parse_actions(text: str) -> list[dict[str, Any]]:
    """Defensively pull action objects out of the LLM reply — tolerates a prose
    preamble, markdown fences, a bare array, or a single bare object."""
    txt = (text or "").strip()
    # Strip ```json … ``` fences if present.
    if txt.startswith("```"):
        txt = re.sub(r"^```[a-zA-Z]*\n?", "", txt)
        txt = re.sub(r"\n?```$", "", txt).strip()

    data: Any = None
    try:
        data = json.loads(txt)
    except (ValueError, TypeError):
        # Prose-then-JSON: grab the first array, else the first object.
        m = re.search(r"\[.*\]", txt, re.S)
        if m:
            try:
                data = json.loads(m.group(0))
            except (ValueError, TypeError):
                data = None
        if data is None:
            m2 = re.search(r"\{.*\}", txt, re.S)
            if m2:
                try:
                    data = json.loads(m2.group(0))
                except (ValueError, TypeError):
                    data = None
    if data is None:
        return []
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        return []
    out: list[dict[str, Any]] = []
    for raw in data:
        if not isinstance(raw, dict):
            continue
        kind = str(raw.get("kind") or "").strip()
        title = str(raw.get("title") or "").strip()
        if kind not in _KINDS or not title:
            continue
        risk = str(raw.get("risk") or "normal").strip()
        if risk not in _RISKS:
            risk = "normal"
        try:
            score = max(0.0, min(1.0, float(raw.get("score", 0.0))))
        except (ValueError, TypeError):
            score = 0.0
        out.append({
            "kind": kind,
            "title": title[:200],
            "rationale": str(raw.get("rationale") or "").strip()[:500],
            "risk": risk,
            "domain": (str(raw.get("domain") or "general").strip() or "general")[:40],
            "score": score,
            # stop_run carries the run to halt.
            "target": str(raw.get("target") or "").strip()[:80],
            "system": (str(raw.get("system") or "basna").strip().lower() or "basna"),
            # tool_action carries the catalog action + its args.
            "action_id": str(raw.get("action_id") or "").strip()[:64],
            "args": raw.get("args") if isinstance(raw.get("args"), dict) else {},
            # track carries a follow-up horizon; track/nudge may reference an
            # existing due follow-up by id.
            "follow_up_days": _coerce_int(raw.get("follow_up_days")),
            "follow_up_id": str(raw.get("follow_up_id") or "").strip().removeprefix("FU:")[:40],
            # event_ref ties an action back to the surfaced event it acts on, so
            # dispatch can ground the agent with the real handle (see EV:<id>).
            "event_ref": str(raw.get("event_ref") or "").strip().removeprefix("EV:")[:48],
        })
    return out


def _coerce_int(v: Any) -> int | None:
    try:
        return int(v)
    except (ValueError, TypeError):
        return None


def _parse_iso(s: Any, default: datetime) -> datetime:
    try:
        return datetime.fromisoformat(str(s))
    except (ValueError, TypeError):
        return default


async def _gather_active_runs(user_id: str) -> list[dict[str, Any]]:
    """Currently-running Basna runs the arbiter could stop (owner-scoped). Sourced
    from the live worker/task registries so it only ever lists genuinely-running,
    stoppable runs."""
    try:
        from captain_claw.flight_deck.auth import get_db
        from captain_claw.flight_deck.basna_routes import (
            _active_agent_runs,
            _run_workers,
        )
    except Exception:
        return []
    db = get_db()
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    candidates = set(_active_agent_runs.get(user_id, set())) | set(_run_workers.keys())
    for sid in candidates:
        if sid in seen:
            continue
        seen.add(sid)
        try:
            sess = await db.get_basna_session(sid, user_id)  # None if not owner's
        except Exception:
            sess = None
        if sess and str(sess.get("status") or "") not in ("done", "cancelled", "error"):
            out.append({
                "system": "basna", "session_id": sid,
                "title": (sess.get("title") or sess.get("intent") or "")[:60],
            })
    return out


async def maybe_run_arbiter(
    user_id: str,
    reflection: dict[str, Any],
    author: dict[str, Any],
    agent_slugs: list[str],
    *,
    trigger: str = "pulse",
) -> dict[str, Any] | None:
    """One arbiter pass for ``user_id`` after a reflection. Returns a small summary
    (or None when the loop is off). Never raises — and it writes a trace to the
    autonomy log so nothing is swallowed. ``trigger='manual'`` (a nudge) logs every
    decision; routine pulses log only when something happens or errors, to stay quiet."""
    cfg = resolve_config(user_id)
    if not cfg.get("enabled") or not cfg.get("arbiter_on_pulse"):
        return None
    level = str(cfg.get("autonomy_level") or "off")
    if level == "off":
        return None

    store = get_store()

    def emit(event: str, detail: str = "", level_: str = "info", routine: bool = False) -> None:
        # Routine skips are noisy on the 180s pulse — only surface them on a
        # manual nudge. Outcomes and errors always log.
        if routine and trigger != "manual":
            return
        store.log(user_id, event, detail, level_)

    try:
        if _in_quiet_hours(int(cfg.get("quiet_hours_start", 22)), int(cfg.get("quiet_hours_end", 8))):
            emit("skipped: quiet hours", f"{cfg.get('quiet_hours_start')}–{cfg.get('quiet_hours_end')} UTC", routine=True)
            return {"ran": False, "reason": "quiet-hours"}

        cutoff = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
        if store.count_since(user_id, cutoff) >= int(cfg.get("max_actions_per_day", 6)):
            emit("skipped: daily cap reached", f"max={cfg.get('max_actions_per_day')}", routine=True)
            return {"ran": False, "reason": "daily-cap"}

        open_actions = store.open_actions(user_id)
        if len(open_actions) >= int(cfg.get("max_concurrent_actions", 2)):
            emit("skipped: concurrency cap", f"{len(open_actions)} in flight, max={cfg.get('max_concurrent_actions')}", routine=True)
            return {"ran": False, "reason": "concurrent-cap"}
        look_cutoff = (
            datetime.now(timezone.utc)
            - timedelta(hours=int(cfg.get("candidate_lookback_hours", 24)))
        ).isoformat()
        dedup_titles = store.recent_titles(user_id, look_cutoff)
        dedup_titles |= {a["title"].strip().lower() for a in open_actions}

        candidates = _gather_candidates(
            reflection, include_reflections=bool(cfg.get("reflection_to_intention")),
        )
        # External-world events (#2): surface new events as candidates so the loop
        # reacts to the user's world. We do NOT mark them surfaced here — only once
        # a pass actually produces an action (below). A pass that yields nothing
        # leaves them reconsiderable (bounded by event_max_surface_attempts) so a
        # single whiffed beat — or a manual "Run arbiter now" — gets another shot.
        event_ids: list[str] = []
        evstore = None
        try:
            from captain_claw.flight_deck.events import get_store as _events_store
            evstore = _events_store()
            new_events = evstore.list_new(user_id, limit=5)
            for ev in new_events:
                # Tag with EV:<id> so an action ABOUT this event can reference it
                # ("event_ref"); dispatch then resolves the real handle (gmail
                # thread/message id, calendar event id) and hands it to the agent
                # to fetch by id — instead of the agent re-searching and missing it.
                candidates.append(f"(event · {ev['source']} · EV:{ev['id']}) {ev['summary']}")
                event_ids.append(ev["id"])
        except Exception as exc:
            _log.debug("event intake failed (non-fatal): %s", exc)

        max_attempts = int(cfg.get("event_max_surface_attempts", 4))

        def _settle_events(*, produced: bool) -> None:
            """Resolve the events fed into this pass. produced=True → they got
            their shot, mark surfaced. produced=False → defer for a later pass,
            giving up only after event_max_surface_attempts."""
            if not evstore or not event_ids:
                return
            try:
                if produced:
                    evstore.mark(event_ids, "surfaced")
                else:
                    spent = evstore.defer(event_ids, max_attempts=max_attempts)
                    if spent:
                        emit("events given up", f"{len(spent)} reconsidered "
                             f"{max_attempts}× with no action → ignored", "warn", routine=True)
            except Exception as exc:
                _log.debug("event settle failed (non-fatal): %s", exc)

        # Due follow-ups (tracked open loops whose time has come): re-feed them as
        # candidates so the arbiter can nudge / re-snooze. Cool each one down right
        # away so it isn't re-fed every 180s pulse; a nudge re-arms it on dispatch.
        fu_count = 0
        if evstore is not None:
            try:
                now_dt = datetime.now(timezone.utc)
                cooldown = timedelta(hours=int(cfg.get("followup_resurface_cooldown_hours", 12)))
                for fu in evstore.list_due_follow_ups(user_id, now_dt.isoformat(), limit=5):
                    age_days = max(0, (now_dt - _parse_iso(fu.get("created_at"), now_dt)).days)
                    candidates.append(
                        f"(follow-up due · FU:{fu['id']} · {age_days}d old · "
                        f"surfaced {fu.get('surfaced_count', 0)}×) {fu['summary']}"
                        + (f" — {fu['detail']}" if fu.get("detail") else "")
                    )
                    evstore.touch_follow_up(
                        fu["id"], follow_up_at=(now_dt + cooldown).isoformat(), surfaced=True)
                    fu_count += 1
            except Exception as exc:
                _log.debug("follow-up intake failed (non-fatal): %s", exc)

        emit("arbiter pass", f"trigger={trigger}, level={level}, {len(candidates)} candidate(s)"
             + (f", {len(event_ids)} event(s)" if event_ids else "")
             + (f", {fu_count} follow-up(s) due" if fu_count else ""))
        if not candidates:
            emit("skipped: no candidates", "reflection produced no intentions/thought to act on", "warn")
            return {"ran": False, "reason": "no-candidates"}

        reliability = store.list_reliability(user_id)
        rel_by_kind: dict[str, float] = {}
        rel_by_pair: dict[tuple[str, str], float] = {}   # (kind, domain) — per-action for tool_action
        for r in reliability:
            rel_by_kind[r["kind"]] = min(rel_by_kind.get(r["kind"], 1.0), float(r["weight"]))
            rel_by_pair[(r["kind"], r["domain"])] = float(r["weight"])

        def _weight_for(a: dict[str, Any]) -> float:
            # tool_action trust/suppression is per concrete action_id; others per kind.
            if a["kind"] == "tool_action" and a.get("action_id"):
                return rel_by_pair.get(("tool_action", a["action_id"]), 1.0)
            return rel_by_kind.get(a["kind"], 1.0)

        rel_hint = ""
        if reliability:
            rel_hint = "\n\nLearned reliability of action kinds (favour higher): " + ", ".join(
                f"{r['kind']}={r['weight']:.2f}" for r in reliability[:8]
            )
        dup_hint = ""
        if dedup_titles:
            dup_hint = "\n\nAlready proposed (do not repeat): " + "; ".join(
                a["title"] for a in open_actions[:8]
            )
        # Recently COMPLETED actions — so the arbiter doesn't re-propose reworded
        # variants of work it already did (auto-fired actions go straight to done,
        # leaving the "open" list, so they must be surfaced separately).
        done_recent = [
            a["title"] for a in store.list_actions(user_id, limit=40)
            if a.get("status") in ("done", "dispatched", "undone")
            and a.get("created_at", "") >= look_cutoff
        ]
        if done_recent:
            dup_hint += ("\n\nAlready DONE recently (do NOT re-propose these or any "
                         "reworded variant — move on to something else): "
                         + "; ".join(done_recent[:10]))
        # Active runs the arbiter may target with stop_run (owner-scoped, live).
        active_runs = await _gather_active_runs(user_id)
        valid_targets = {r["session_id"] for r in active_runs}
        run_hint = ""
        if active_runs:
            run_hint = "\n\nActive runs (stop_run target = the session id):\n" + "\n".join(
                f"- {r['session_id']} · {r['title']}" for r in active_runs[:8]
            )
        # Concrete actions the arbiter may PROPOSE via tool_action — any
        # non-human-only catalog action. Whether one auto-fires vs awaits approval
        # is decided downstream by grants + reversibility (should_auto_dispatch).
        from captain_claw.flight_deck.action_catalog import list_catalog
        catalog = [a for a in list_catalog(user_id=user_id) if not a["human_only"]]
        cat_hint = ""
        if catalog:
            cat_hint = "\n\nAction catalog (tool_action — action_id + args filling required):\n" + "\n".join(
                f"- {a['id']}: {a['label']} · required args: {a['required']}" for a in catalog
            )
        user_prompt = (
            "Candidate goals the assistant has surfaced to itself:\n"
            + "\n".join(f"- {c}" for c in candidates)
            + rel_hint + dup_hint + run_hint + cat_hint
        )

        # Think through the same agent that authored the reflection.
        try:
            from captain_claw.games.remote_provider import RemoteLLMProvider
            from captain_claw.llm import Message

            provider = RemoteLLMProvider(
                host=author["host"], port=author["port"], auth=author["auth"],
                name=author.get("name", ""),
            )
            resp = await provider.complete(
                messages=[
                    Message(role="system", content=_SYSTEM_PROMPT),
                    Message(role="user", content=user_prompt),
                ],
                temperature=0.3,
                max_tokens=1500,
            )
        except Exception as exc:
            _log.warning("arbiter: no agent could rank: %s", exc)
            emit("error: ranking LLM failed", f"{author.get('name','?')}: {exc}", "error")
            _settle_events(produced=False)  # transient — let the next pass retry
            return {"ran": False, "reason": "no-thinker", "error": str(exc)}

        actions = _parse_actions(resp.content)
        min_score = float(cfg.get("arbiter_min_score", 0.6))
        suppress_below = float(cfg.get("suppress_below_weight", 0.25))
        emit("ranked", f"{len(actions)} action(s) returned: " +
             ("; ".join(f"{a['kind']}:{a['title']}={a['score']:.2f}" for a in actions[:5]) or "(none)"))
        if not actions:
            # Show the raw reply so we can tell "model chose to wait ([])" from
            # "model answered in a shape we rejected (e.g. kind outside the enum)".
            raw = (resp.content or "").strip().replace("\n", " ")
            emit("ranker raw reply", raw[:600] or "(empty)", "warn")

        # Filter: threshold, learned-loser suppression, dedup. Keep the best one.
        viable = []
        for a in actions:
            # 'track' is cheap bookkeeping (an open loop, no external effect) and is
            # the whole point for low-urgency soft requests — exempt it from min_score.
            if a["kind"] != "track" and a["score"] < min_score:
                emit("dropped: below min score", f"{a['title']} ({a['score']:.2f} < {min_score})", routine=True)
                continue
            if a["title"].strip().lower() in dedup_titles:
                emit("dropped: already proposed", a["title"], routine=True)
                continue
            if _weight_for(a) < suppress_below:
                emit("dropped: suppressed (low reliability)",
                     f"{a['kind']}{':' + a['action_id'] if a.get('action_id') else ''} "
                     f"weight {_weight_for(a):.2f} < {suppress_below}", routine=True)
                continue
            if a["kind"] == "stop_run" and a.get("target") not in valid_targets:
                # Don't let it stop a run it can't see (or hallucinate a session id).
                emit("dropped: stop_run unknown target", str(a.get("target")), "warn")
                continue
            if a["kind"] == "tool_action":
                # Resolve against the catalog; risk/reversibility come from the
                # catalog, never the LLM. Drop unknown/human-only/invalid-arg actions.
                from captain_claw.flight_deck import action_catalog
                spec = action_catalog.get_action(a.get("action_id"), user_id)
                if not spec or spec.get("human_only"):
                    emit("dropped: tool_action not allowed", str(a.get("action_id")), "warn")
                    continue
                ok_args, arg_err = action_catalog.validate_args(spec, a.get("args") or {})
                if not ok_args:
                    emit("dropped: tool_action bad args", f"{a.get('action_id')}: {arg_err}", "warn")
                    continue
                a["risk"] = spec["risk"]
                a["reversibility"] = spec["reversibility"]
            viable.append(a)
        viable.sort(key=lambda a: a["score"], reverse=True)

        if not viable:
            emit("nothing viable", f"{len(actions)} considered, all filtered out", "warn")
            _settle_events(produced=False)  # reconsider next pass / manual run
            return {"ran": True, "proposed": 0, "reason": "nothing-viable",
                    "considered": len(actions)}

        chosen = viable[0]
        _settle_events(produced=True)  # a pass produced an action — events got their shot
        _payload = None
        if chosen["kind"] == "stop_run":
            _payload = {"system": chosen.get("system") or "basna", "target": chosen.get("target")}
        elif chosen["kind"] == "tool_action":
            _payload = {"action_id": chosen.get("action_id"), "args": chosen.get("args") or {}}
        elif chosen["kind"] == "track":
            days = chosen.get("follow_up_days")
            if not isinstance(days, int) or days <= 0:
                days = int(cfg.get("followup_default_days", 3))
            _payload = {
                "summary": chosen["title"], "detail": chosen.get("rationale") or "",
                "source": chosen.get("domain") or "reflection",
                "follow_up_days": days,
                "follow_up_id": chosen.get("follow_up_id") or "",  # set ⇒ re-snooze
            }
        elif chosen["kind"] == "nudge" and chosen.get("follow_up_id"):
            # A reminder nudge that addresses a due follow-up — dispatch re-arms it.
            _payload = {"follow_up_id": chosen.get("follow_up_id")}
        # Grounding: if this action is about a surfaced event, carry its EV ref so
        # dispatch can hand the agent the real handle (fetch by id, never search).
        if chosen.get("event_ref") and chosen["kind"] in ("nudge", "run_prompt", "tool_action"):
            _payload = {**(_payload or {}), "event_ref": chosen["event_ref"]}
        row = store.add_action(
            user_id,
            kind=chosen["kind"], title=chosen["title"], rationale=chosen["rationale"],
            source="reflection", risk=chosen["risk"], domain=chosen["domain"],
            score=chosen["score"], status="awaiting_approval", payload=_payload,
        )

        from captain_claw.flight_deck.fd_dispatch import dispatch_action, should_auto_dispatch

        dispatched = False
        if should_auto_dispatch(cfg, row):
            disp = await dispatch_action(user_id, row)
            dispatched = disp["ok"]
            if not disp["ok"]:
                emit("dispatch deferred", f"{chosen['title']}: {disp['note']}", "warn")

        emit("dispatched" if dispatched else "proposed",
             f"{chosen['kind']} · {chosen['title']} (score {chosen['score']:.2f}, risk {chosen['risk']})")
        _log.info("arbiter: %s %r for %s",
                  "dispatched" if dispatched else "proposed", chosen["title"], user_id)
        return {"ran": True, "proposed": 1, "dispatched": dispatched,
                "action_id": row.get("id"), "title": chosen["title"],
                "considered": len(actions)}
    except Exception as exc:
        import traceback
        _log.warning("arbiter pass crashed: %s", exc)
        store.log(user_id, "error: arbiter crashed",
                  f"{exc}\n{traceback.format_exc()[-800:]}", "error")
        return {"ran": False, "reason": "error", "error": str(exc)}
