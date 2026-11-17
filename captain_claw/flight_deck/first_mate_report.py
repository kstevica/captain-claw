"""First Mate's week: one Line item a week with how each factory did.

See docs/first-mate-plan.md §11 (Phase 4, piece 4). Everything is read from
the store, with no model, so the report never costs tokens.

* **The week** is Monday 00:00 to Monday 00:00 UTC, the same week the budget
  uses. The report on a week goes out on the first scan after
  :data:`REPORT_HOUR` UTC on the Monday that ends it, once (a mark per owner,
  and the Line item is keyed by the week).
* **Totals:** runs and how many finished done, carry-outs and doubtful ones,
  tokens against the weekly budget, what became of its drafts.
* **A line per factory** (each one approved by the end of the week): health,
  trust, runs, tokens against its share, satisfaction from the week's signals,
  its cases' latest score, and what moved: trust raised or lowered, a version
  that went live or was held, a pause.
* **Worth a look:** what waits for the owner or is wrong: a red or paused
  factory, proposals waiting, a held draft, a drop on its cases, a record
  good enough for more trust that waits on calibration, a budget nearly used.
  Proposals the owner accepts from the item come with Phase 5.

A week with no run and nothing worth a look sends nothing.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any

from captain_claw.flight_deck import first_mate_store as fms

log = logging.getLogger(__name__)

REPORT_HOUR = 7
WEEK = timedelta(days=7)
REF_TYPE = "first_mate_report"
_TRUST_WORDS = {"propose": "asks first", "commit": "acts within its grants", "direct": "acts directly"}
_HEALTH_WORDS = {"green": "Healthy", "amber": "Needs a look", "red": "In trouble", "new": "Too new to say"}
_DRAFT_SIGNALS = {"sent_as_written": "sent", "sent_edited": "edited", "draft_deleted": "deleted"}


def tokens_text(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}".rstrip("0").rstrip(".") + "M"
    if n >= 1_000:
        return f"{round(n / 1_000)}k"
    return str(n)


def period(now: datetime, which: str = "last") -> tuple[datetime, datetime]:
    """``last``: the week that ended at this Monday; ``this``: this week so far."""
    from captain_claw.flight_deck.first_mate_loop import week_start
    start = week_start(now)
    return (start, now) if which == "this" else (start - WEEK, start)


def _within(iso: str | None, start: datetime, end: datetime) -> bool:
    if not iso:
        return False
    at = datetime.fromisoformat(iso)
    return start <= at < end


def factory_week(store: fms.FirstMateStore, f: dict[str, Any], start: datetime, end: datetime, now: datetime,
                 share: int, calibration: dict[str, Any]) -> dict[str, Any]:
    """One factory's week, and what about it is worth a look."""
    from captain_claw.flight_deck import first_mate_score, first_mate_trust
    runs = [r for r in store.runs_since(f["id"], fms._iso(start), kinds=fms.WORK_KINDS)
            if datetime.fromisoformat(r["created_at"]) < end]
    finished = [r for r in runs if r["status"] in fms.TERMINAL_RUN and r["status"] != "cancelled"]
    carries = [r for r in finished if r["kind"] == "carry_out"]
    doubts = [r for r in carries if (r.get("proof") or {}).get("state") == "doubt"]
    tokens = max(0, store.tokens_since(f["id"], start) - store.tokens_since(f["id"], end))
    signals = [s for s in store.signals(f["id"], limit=500) if _within(s["at"], start, end)]
    drafts = {v: sum(1 for s in signals if _DRAFT_SIGNALS.get(s["kind"]) == v) for v in _DRAFT_SIGNALS.values()}
    card = first_mate_score.scorecard(store, f, now)
    checks = [c for c in store.checks(f["id"], limit=50) if c["status"] == "done" and _within(c["ended_at"], start, end)]
    moved: list[str] = []
    for e in reversed(store.events(f["id"], limit=500)):  # oldest first
        if not _within(e["at"], start, end):
            continue
        d = e.get("detail") or {}
        if e["event"] == "trust" and d.get("from") != d.get("to"):
            who = {"record": "Its record raised", "deck": "The deck lowered", "owner": "You set"}.get(e["actor"], "")
            moved.append(f"{who} its trust to: {_TRUST_WORDS.get(d.get('to'), d.get('to'))}.")
        elif e["event"] == "approved" and int(d.get("version") or 0) > 1:
            moved.append(f"Version {d.get('version')} went live.")
        elif e["event"] == "check_finished" and d.get("purpose") == "approve" and d.get("status") == "done" \
                and "stays a draft" in str(d.get("note") or ""):
            moved.append(f"Version {d.get('version')} didn't go live: it scored lower on its cases.")
        elif e["event"] == "state" and d.get("to") in ("paused", "paused_budget", "stopped"):
            what = {"paused": "Paused", "paused_budget": "Paused over budget", "stopped": "Stopped"}[d["to"]]
            reason = str(d.get("reason") or "").strip().rstrip(".")
            moved.append(f"{what}: {reason}." if reason else f"{what}.")
        elif e["event"] == "state" and d.get("to") == "running" and d.get("from") in ("paused", "paused_budget",
                                                                                         "stopped"):
            moved.append("Resumed.")
    look: list[str] = []
    title = f["title"]
    if f["state"] in ("paused", "paused_budget", "stopped"):
        what = {"paused": "paused", "paused_budget": "paused over budget", "stopped": "stopped"}[f["state"]]
        reason = (f["state_reason"] or "").strip().rstrip(".")
        look.append(f"{title} is {what}: {reason}." if reason else f"{title} is {what}.")
    elif card["health"] in ("red", "amber") and card["health_why"]:
        look.append(f"{title}: {card['health_why'].rstrip('.')}.")
    waiting = len(store.open_proposals(f["id"]))
    if waiting:
        look.append(f"{title} has {waiting} {'proposal' if waiting == 1 else 'proposals'} waiting for you.")
    latest = card.get("check")
    draft = (store.view(f["id"]) or {}).get("draft")
    if latest and draft and latest["purpose"] == "approve" and latest["status"] == "done" \
            and latest["version"] == draft["version"] and latest.get("version_sha") == draft["sha"]:
        look.append(f"{title}: version {draft['version']} waits for you after scoring {latest['score']:.0%} "
                    "on its cases.")
    rec = first_mate_trust.record(store, f, calibration)
    if rec["next"] and not calibration["ready"] and "agree with you" in rec["why"]:
        look.append(f"{title} has earned more trust, but trust moves on its own only once its checks agree "
                    f"with your labels ({calibration['why']}).")
    return {
        "id": f["id"], "title": title, "state": f["state"], "trust": f["trust"],
        "health": card["health"], "health_why": card["health_why"],
        "runs": len(runs), "done": sum(1 for r in finished if r["status"] == "done"),
        "carry_outs": len(carries), "doubts": len(doubts), "tokens": tokens, "share": share,
        "signals": len(signals),
        "satisfaction": round(sum(float(s["value"]) for s in signals) / len(signals), 3) if signals else None,
        "drafts": drafts, "cases_score": checks[0]["score"] if checks else None,
        "moved": moved, "look": look,
    }


def build(store: fms.FirstMateStore, owner_id: str, start: datetime, end: datetime,
          now: datetime | None = None) -> dict[str, Any]:
    """The owner's week from ``start`` to ``end``."""
    from captain_claw.flight_deck import first_mate_loop, first_mate_trust
    now = now or fms._utcnow()
    owner = fms.norm_user(owner_id)
    prefs = store.prefs(owner)
    factories = [f for f in store.list_factories(owner) if f["version"]
                 and datetime.fromisoformat(f["created_at"]) < end]
    share = first_mate_loop.shares(prefs, factories)
    cal = first_mate_trust.calibration(store, owner)
    lines = [factory_week(store, f, start, end, now, share.get(f["id"], 0), cal) for f in factories]
    used = sum(x["tokens"] for x in lines)
    budget = int(prefs["weekly_tokens"])
    look = [item for x in lines for item in x["look"]]
    if budget and used >= 0.9 * budget:
        look.append(f"Your factories used {used / budget:.0%} of the week's {tokens_text(budget)} tokens.")
    return {
        "start": start.isoformat(), "end": end.isoformat(), "owner": owner,
        "totals": {"runs": sum(x["runs"] for x in lines), "done": sum(x["done"] for x in lines),
                   "carry_outs": sum(x["carry_outs"] for x in lines), "doubts": sum(x["doubts"] for x in lines),
                   "tokens": used, "budget": budget,
                   "drafts": {k: sum(x["drafts"][k] for x in lines) for k in _DRAFT_SIGNALS.values()}},
        "factories": lines, "look": look,
        "calibration": {k: cal[k] for k in ("n", "kappa", "ready", "why")},
    }


def _dates(start: datetime, end: datetime) -> str:
    last = end - timedelta(seconds=1)  # a week ends at Monday 00:00; this week so far, today
    if start.month == last.month:
        return f"{start.day}–{last.day} {last.strftime('%b')}"
    return f"{start.day} {start.strftime('%b')} – {last.day} {last.strftime('%b')}"


def text(report: dict[str, Any]) -> tuple[str, str]:
    """The Line item's title and body (Markdown, short lines for phones)."""
    start, end = datetime.fromisoformat(report["start"]), datetime.fromisoformat(report["end"])
    t = report["totals"]
    parts = [f"{t['runs']} {'run' if t['runs'] == 1 else 'runs'}, {t['done']} done"
             + (f" · {t['carry_outs']} carried out" + (f", {t['doubts']} doubtful" if t["doubts"] else "")
                if t["carry_outs"] else "")
             + f" · {tokens_text(t['tokens'])} of {tokens_text(t['budget'])} tokens"]
    d = t["drafts"]
    if any(d.values()):
        parts.append(f"Drafts: {d['sent']} sent as written, {d['edited']} after edits, {d['deleted']} deleted.")
    for x in report["factories"]:
        parts += ["", f"**{x['title']}** · {_HEALTH_WORDS.get(x['health'], x['health'])} · "
                      f"{_TRUST_WORDS.get(x['trust'], x['trust'])}"]
        bits = [f"{x['runs']} {'run' if x['runs'] == 1 else 'runs'}, {x['done']} done",
                f"{tokens_text(x['tokens'])} of its {tokens_text(x['share'])} share"]
        parts.append("- " + " · ".join(bits))
        sat = []
        if x["satisfaction"] is not None:
            sat.append(f"{x['satisfaction']:.0%} satisfied ({x['signals']} "
                       f"{'signal' if x['signals'] == 1 else 'signals'})")
        if x["cases_score"] is not None:
            sat.append(f"{x['cases_score']:.0%} on its cases")
        if sat:
            parts.append("- " + " · ".join(sat))
        parts += [f"- {m}" for m in x["moved"]]
    if report["look"]:
        parts += ["", "**Worth a look**"] + [f"- {item}" for item in report["look"]]
    if report.get("proposals"):
        parts += ["", "**First Mate proposes** (approve or reject each on the Line)"]
        parts += [f"- {p}" for p in report["proposals"]]
    parts += ["", "Open First Mate for each factory's record."]
    body = "\n".join(parts)
    if len(body) > 3900:
        body = body[:3880].rsplit("\n", 1)[0] + "\n…"
    return f"First Mate's week: {_dates(start, end)}", body


def _mark_key(owner: str) -> str:
    return f"report:{fms.norm_user(owner)}"


def last_sent(store: fms.FirstMateStore, owner_id: str) -> dict[str, Any]:
    return store._setting(_mark_key(owner_id)) or {}


async def maybe_send(store: fms.FirstMateStore, owner_id: str, now: datetime) -> bool:
    """Send the report on the week that just ended, once, after REPORT_HOUR on Monday."""
    from captain_claw.flight_deck.first_mate_loop import week_start
    this_week = week_start(now)
    if now < this_week + timedelta(hours=REPORT_HOUR):
        return False
    week = this_week.date().isoformat()
    if last_sent(store, owner_id).get("week") == week:
        return False
    start, end = period(now, "last")
    report = build(store, owner_id, start, end, now)
    sent = bool(report["factories"]) and bool(report["totals"]["runs"] or report["look"])
    # Proposals come with a report that goes out: a quiet week proposes nothing.
    store._put_setting(_mark_key(owner_id), {"week": week, "at": fms._iso(now), "sent": sent})
    if not sent:
        return False
    try:
        from captain_claw.flight_deck import first_mate_review
        report["proposals"] = [r["title"] for r in await first_mate_review.weekly(store, owner_id, report)]
    except Exception:  # noqa: BLE001 — the report goes out without them
        log.warning("first mate: the weekly proposals failed", extra={"owner": owner_id}, exc_info=True)
    title, body = text(report)
    try:
        from captain_claw.flight_deck import line
        from captain_claw.flight_deck.first_mate_line import SOURCE
        await line.post_once(owner_id, "result", title, body, ref_type=REF_TYPE, ref_id=start.date().isoformat(),
                             bell_type="first_mate_report", source=SOURCE, needs_you=False,
                             to_phones=bool(report["look"]))
    except Exception:  # noqa: BLE001
        log.warning("first mate: the weekly report wasn't posted", extra={"owner": owner_id}, exc_info=True)
        return False
    return True
