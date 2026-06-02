"""Phase 3: proactive intentions generator.

A cooldown-gated, quiet-hours-aware, fire-and-forget background pass (mirrors
``reflections``) that reviews recent activity and *proposes* agent intentions —
announced if low-risk, asked otherwise. Guardrails keep it from being noisy:

  * cooldown (``intentions.interval_hours``) between passes
  * per-day cap (``intentions.max_per_day``)
  * quiet hours (``intentions.quiet_hours_start/end``)
  * dedup against active + recently-declined intentions
  * proactivity dial (conservative | balanced | eager) sizes each pass

Opt-in via ``intentions.auto_generate`` (off by default).
"""

from __future__ import annotations

import json
import re
import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from captain_claw.config import get_config
from captain_claw.intentions import (
    OPEN_STATUSES,
    create_proposal,
    get_intentions_manager,
)
from captain_claw.logging import get_logger

if TYPE_CHECKING:
    from captain_claw.agent import Agent

log = get_logger(__name__)

_ATTR_RUNNING = "_intentions_gen_running"

_PROACTIVITY_CAP = {"conservative": 1, "balanced": 2, "eager": 4}


def _in_quiet_hours(hour: int, start: int, end: int) -> bool:
    if start == end:
        return False
    if start < end:
        return start <= hour < end
    return hour >= start or hour < end  # wraps midnight


def _norm(s: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", (s or "").lower()) if len(t) > 2}


def _is_dup(title: str, existing: list[str]) -> bool:
    """Fuzzy title dedup: substring or high token overlap with an existing one."""
    t = (title or "").strip().lower()
    if not t:
        return True
    tt = _norm(t)
    for ex in existing:
        e = ex.strip().lower()
        if not e:
            continue
        if t in e or e in t:
            return True
        et = _norm(e)
        if tt and et:
            overlap = len(tt & et) / len(tt | et)
            if overlap >= 0.6:
                return True
    return False


def _extract_json_array(text: str) -> list[dict[str, Any]]:
    """Pull a JSON array of proposals out of an LLM response (tolerant)."""
    s = (text or "").strip()
    if not s:
        return []
    # Strip code fences.
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.MULTILINE).strip()
    start, end = s.find("["), s.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return []
    try:
        data = json.loads(s[start : end + 1])
    except (json.JSONDecodeError, ValueError):
        return []
    return [d for d in data if isinstance(d, dict)] if isinstance(data, list) else []


async def maybe_auto_propose(agent: "Agent", *, trigger: str = "periodic") -> None:
    """Entry point — run a proposal pass if all guardrails allow it."""
    cfg = get_config().intentions
    if not cfg.auto_generate:
        return
    if getattr(agent, _ATTR_RUNNING, False):
        return
    session = getattr(agent, "session", None)
    if session is None or len(getattr(session, "messages", []) or []) < cfg.min_messages:
        return

    now = datetime.now(UTC)
    if _in_quiet_hours(now.hour, cfg.quiet_hours_start, cfg.quiet_hours_end):
        return

    setattr(agent, _ATTR_RUNNING, True)
    try:
        mgr = get_intentions_manager()
        state = await mgr.get_generator_state()
        today = now.date().isoformat()
        count_today = state["count"] if state.get("day") == today else 0
        if count_today >= cfg.max_per_day:
            return
        # Cooldown.
        last = state.get("last_run_at")
        if last:
            try:
                elapsed = (now - datetime.fromisoformat(last)).total_seconds()
                if elapsed < cfg.interval_hours * 3600:
                    return
            except (ValueError, TypeError):
                pass

        remaining = cfg.max_per_day - count_today
        cap = min(_PROACTIVITY_CAP.get(cfg.proactivity, 2), remaining)
        created = await _generate(agent, cap)

        await mgr.set_generator_state(
            last_run_at=now.isoformat(),
            day=today,
            count=count_today + created,
        )
        if created:
            log.info("Intentions generator proposed", count=created, trigger=trigger)
    except Exception as exc:
        log.debug("Intentions generator skipped: %s", exc)
    finally:
        setattr(agent, _ATTR_RUNNING, False)


async def _generate(agent: "Agent", cap: int) -> int:
    """Run the LLM proposal pass; create up to ``cap`` new intentions. Returns count."""
    if cap <= 0:
        return 0
    from captain_claw.llm import LLMResponse, Message

    mgr = get_intentions_manager()

    # ── Signals ──────────────────────────────────────────────────────
    recent = ""
    try:
        msgs = (agent.session.messages or [])[-15:]
        lines = []
        for m in msgs:
            role = m.get("role", "")
            if role not in ("user", "assistant"):
                continue
            content = str(m.get("content", "") or "").strip()
            if content:
                lines.append(f"[{role}] {content[:400]}")
        recent = "\n".join(lines)
    except Exception:
        pass

    insights_text = ""
    try:
        from captain_claw.insights import get_insights_manager
        recent_ins = await get_insights_manager().list_recent(limit=15)
        insights_text = "\n".join(
            f"- [{i.get('category')}] {i.get('content')}" for i in recent_ins
        )
    except Exception:
        pass

    # Existing titles → both the prompt (avoid) and the post-filter (dedup).
    existing_titles: list[str] = []
    try:
        for it in await mgr.list(statuses=list(OPEN_STATUSES), limit=50):
            existing_titles.append(it.get("title", ""))
        for it in await mgr.list(status="declined", limit=30):
            existing_titles.append(it.get("title", ""))
    except Exception:
        pass

    if not recent and not insights_text:
        return 0

    avoid = "\n".join(f"- {t}" for t in existing_titles if t) or "(none)"
    system_prompt = (
        "You are the proactive-assistant module. Review the user's recent activity "
        "and propose at most {cap} NEW intentions worth acting on — automations, "
        "follow-ups, or reminders the user would genuinely value. Be useful, not "
        "noisy: propose only high-signal items, or none.\n"
        "Return ONLY a JSON array (no prose). Each item: "
        '{{"title": short imperative, "why": one sentence, '
        '"risk": "low" for read-only/no-send (will be announced) or "normal" for '
        'anything that sends/changes data (will ask first), '
        '"repeat": optional schedule like "weekly mon 09:00" or omit, '
        '"action_prompt": optional instruction to run when it fires}}.\n'
        "Do NOT propose anything similar to the AVOID list. If nothing is worth "
        "proposing, return []."
    ).format(cap=cap)
    user_prompt = (
        f"RECENT CONVERSATION:\n{recent or '(none)'}\n\n"
        f"KNOWN FACTS / INSIGHTS:\n{insights_text or '(none)'}\n\n"
        f"AVOID (already active or previously declined):\n{avoid}\n\n"
        f"Propose up to {cap} new intentions as a JSON array."
    )

    try:
        cfg = get_config()
        resp: LLMResponse = await agent._complete_with_guards(
            messages=[
                Message(role="system", content=system_prompt),
                Message(role="user", content=user_prompt),
            ],
            tools=None,
            interaction_label="intentions_generator",
            max_tokens=min(800, int(cfg.model.max_tokens)),
        )
    except Exception as exc:
        log.debug("Intentions generator LLM call failed: %s", exc)
        return 0

    proposals = _extract_json_array(resp.content or "")
    if not proposals:
        return 0

    waid = ""
    md = getattr(agent.session, "metadata", None)
    if isinstance(md, dict):
        waid = str(md.get("whatsapp_waid") or "").strip()
    push = bool(get_config().intentions.push_to_whatsapp) and bool(waid)

    created = 0
    for p in proposals:
        if created >= cap:
            break
        title = str(p.get("title") or "").strip()
        if not title or _is_dup(title, existing_titles):
            continue
        risk = str(p.get("risk") or "normal").strip().lower()
        risk = risk if risk in ("low", "normal", "high") else "normal"
        repeat = str(p.get("repeat") or "").strip() or None
        action_prompt = str(p.get("action_prompt") or "").strip() or None
        try:
            result = await create_proposal(
                title=title,
                why=str(p.get("why") or "").strip(),
                risk=risk,
                repeat=repeat,
                action_prompt=action_prompt,
                source_session=str(getattr(agent.session, "id", "") or ""),
                waid=waid,
            )
        except Exception as exc:
            log.debug("create_proposal failed: %s", exc)
            continue
        existing_titles.append(title)
        created += 1
        if push and result.get("question"):
            await _push_whatsapp(result["question"], waid)
    return created


async def _push_whatsapp(question: str, waid: str) -> None:
    """Best-effort proactive delivery of a proposal to the current WhatsApp chat."""
    import os

    from captain_claw.fd_client import FDClient, flight_deck_base

    if not flight_deck_base() or not waid:
        return
    headers = {}
    tok = (os.environ.get("FD_GLASSES_BRIDGE_TOKEN") or "").strip()
    if tok:
        headers["x-glasses-token"] = tok
    fd = FDClient(timeout=15.0)
    try:
        await fd.post("/whatsapp/push", json={"to": waid, "text": f"💡 {question}"}, headers=headers)
    except Exception as exc:
        log.debug("Intention push failed: %s", exc)
    finally:
        await fd.close()
