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
_KINDS = ("nudge", "run_prompt", "basna", "materialize_schedule")
_RISKS = ("low", "normal", "high")

_SYSTEM_PROMPT = (
    "You are the Arbiter: the part of an autonomous assistant that decides what, "
    "if anything, is worth doing next on the user's behalf. You are given the "
    "assistant's own recent reflection and standing intentions. Choose the SINGLE "
    "most valuable concrete next action — or none, if nothing rises above noise.\n\n"
    "Reply with ONLY a JSON array (0 or 1 objects), each:\n"
    '{"kind": one of ["nudge","run_prompt","basna","materialize_schedule"], '
    '"title": short imperative, "rationale": one sentence on why now, '
    '"risk": one of ["low","normal","high"] (low=read-only/internal, '
    'high=sends or changes external data), "domain": short slug e.g. "ops"/"research", '
    '"score": 0.0-1.0 value/urgency}\n\n'
    "Prefer low-risk, genuinely useful actions. Do not invent work to look busy. "
    "Never duplicate something already proposed (a list is given). Return [] when "
    "the best move is to wait."
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
    """Defensively pull a JSON array of action objects out of the LLM reply."""
    txt = (text or "").strip()
    try:
        data = json.loads(txt)
    except (ValueError, TypeError):
        m = re.search(r"\[.*\]", txt, re.S)
        if not m:
            return []
        try:
            data = json.loads(m.group(0))
        except (ValueError, TypeError):
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
        emit("arbiter pass", f"trigger={trigger}, level={level}, {len(candidates)} candidate(s)")
        if not candidates:
            emit("skipped: no candidates", "reflection produced no intentions/thought to act on", "warn")
            return {"ran": False, "reason": "no-candidates"}

        reliability = store.list_reliability(user_id)
        rel_by_kind: dict[str, float] = {}
        for r in reliability:
            rel_by_kind[r["kind"]] = min(rel_by_kind.get(r["kind"], 1.0), float(r["weight"]))

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
        user_prompt = (
            "Candidate goals the assistant has surfaced to itself:\n"
            + "\n".join(f"- {c}" for c in candidates)
            + rel_hint + dup_hint
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
                max_tokens=600,
            )
        except Exception as exc:
            _log.warning("arbiter: no agent could rank: %s", exc)
            emit("error: ranking LLM failed", f"{author.get('name','?')}: {exc}", "error")
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
            if a["score"] < min_score:
                emit("dropped: below min score", f"{a['title']} ({a['score']:.2f} < {min_score})", routine=True)
                continue
            if a["title"].strip().lower() in dedup_titles:
                emit("dropped: already proposed", a["title"], routine=True)
                continue
            if rel_by_kind.get(a["kind"], 1.0) < suppress_below:
                emit("dropped: kind suppressed", f"{a['kind']} weight {rel_by_kind.get(a['kind']):.2f} < {suppress_below}", routine=True)
                continue
            viable.append(a)
        viable.sort(key=lambda a: a["score"], reverse=True)

        if not viable:
            emit("nothing viable", f"{len(actions)} considered, all filtered out", "warn")
            return {"ran": True, "proposed": 0, "reason": "nothing-viable",
                    "considered": len(actions)}

        chosen = viable[0]
        row = store.add_action(
            user_id,
            kind=chosen["kind"], title=chosen["title"], rationale=chosen["rationale"],
            source="reflection", risk=chosen["risk"], domain=chosen["domain"],
            score=chosen["score"], status="awaiting_approval",
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
