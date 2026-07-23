"""Iskra reports — a period's village life, deterministically scooped and
handed to a temporary Deep-Researcher agent to narrate.

The split matters: **this module never asks a model anything.** It collects
the same ledger facts a human would query by hand (ticks, walks, letters,
instincts, health flags) and lays them out as a plain-language data blob. The
agent (see :func:`run_report`) then examines that blob and writes the report —
so the numbers come from the ledger, never from a model's memory of the world.

Voice of the eventual report: *operator + story* — behaviour and metrics and
system health, alongside the beings' own letters and milestones.
"""

from __future__ import annotations

import asyncio
import collections
import math
from datetime import datetime, timedelta, timezone

from captain_claw import vfs
from captain_claw.flight_deck import being_world
from captain_claw.flight_deck.beings import BeingsStore
from captain_claw.llm import Message
from captain_claw.logging import get_logger

log = get_logger(__name__)

REPORTS_PROJECT = "iskra-reports"      # the VFS folder generated reports live in


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# Health/attention event kinds — the ledger's own record of when the machinery
# strained. Counted per being and rolled up so the report can be honest about
# what went wrong, not only what the beings felt.
HEALTH_KINDS = {
    "body_rebound", "body_unreachable", "body_respawned", "body_busy",
    "spawn_failed", "micro_fallback_body", "tick_timeout", "tick_error",
    "tick_skipped", "digest_parse_failed", "digest_repair_retry",
    "narration_mismatch", "act_unverified", "act_unverifiable",
    "drive_unearned", "collapsed_exhausted", "fever", "confusion",
    "society_refused", "write_gate_retry", "link_gate_retry",
    "connect_backoff", "earning_refused",
}

# The fixed period vocabulary the UI offers (plus "custom").
PERIOD_LABELS = ("today", "yesterday", "this week", "last 7 days",
                 "this month", "last 30 days")


# ── Period resolution ─────────────────────────────────────────────────────

def resolve_period(label: str, now: datetime | None = None, *,
                   start: str | None = None,
                   end: str | None = None) -> tuple[str, str, str]:
    """A period name → (start_iso, end_iso, display) in UTC ISO for querying.

    Day boundaries are drawn on the **village's clock** (the parent's tz, the
    same one the beings wake by), not UTC — so "today" means the day the parent
    is living, then converted back to UTC because that's how events are stored.
    Rolling windows ("last 7/30 days") run back from this instant; calendar
    windows ("today", "this week", "this month") snap to local midnight.
    """
    now = now or _utcnow()
    ln = being_world._local(now)                 # this instant, parent's tz
    tz = ln.tzinfo or timezone.utc
    day0 = ln.replace(hour=0, minute=0, second=0, microsecond=0)

    def _u(d: datetime) -> str:
        return d.astimezone(timezone.utc).isoformat()

    lbl = (label or "").strip().lower().replace("_", " ")
    if lbl == "custom":
        s = _local_date(start, tz, day0)
        e = (_local_date(end, tz, day0) + timedelta(days=1)) if end else ln
        disp = f"{(start or '?')} → {(end or 'now')}"
        return _u(s), _u(e), disp
    if lbl == "today":
        return _u(day0), _u(ln), "today"
    if lbl == "yesterday":
        return _u(day0 - timedelta(days=1)), _u(day0), "yesterday"
    if lbl == "this week":                        # Monday → now
        return _u(day0 - timedelta(days=day0.weekday())), _u(ln), "this week"
    if lbl == "last 7 days":                      # rolling 7×24h
        return _u(ln - timedelta(days=7)), _u(ln), "last 7 days"
    if lbl == "this month":                       # 1st → now
        return _u(day0.replace(day=1)), _u(ln), "this month"
    if lbl == "last 30 days":                     # rolling 30×24h
        return _u(ln - timedelta(days=30)), _u(ln), "last 30 days"
    raise ValueError(f"unknown period: {label!r}")


def _local_date(s: str | None, tz, default: datetime) -> datetime:
    """A 'YYYY-MM-DD' (or ISO) string as local midnight; default on junk."""
    if not s:
        return default
    try:
        d = datetime.fromisoformat(s)
    except ValueError:
        return default
    if d.tzinfo is None:
        d = d.replace(tzinfo=tz)
    return d.replace(hour=0, minute=0, second=0, microsecond=0)


# ── The scoop ─────────────────────────────────────────────────────────────

def _entropy(counts: dict[str, int]) -> float:
    total = sum(counts.values())
    if total <= 0 or len(counts) < 2:
        return 0.0
    h = -sum((c / total) * math.log(c / total) for c in counts.values())
    return round(h / math.log(len(counts)), 3)


def collect_report_data(store: BeingsStore, owner_id: str,
                        start_iso: str, end_iso: str, *,
                        display: str = "", now: datetime | None = None) -> dict:
    """Everything a report needs about one owner's village over [start, end).

    Pure reads — no model, no side effects. Returns a structured dict; feed it
    through :func:`render_data_markdown` to get the blob the agent examines.
    """
    now = now or _utcnow()
    roster = store.list(owner_id)
    beings = [store.get(owner_id, r["slug"]) for r in roster]
    by_id = {b["id"]: b for b in beings}
    id_name = {b["id"]: b["name"] for b in beings}

    # One range query, then bucket per being — the whole window, untruncated.
    events = store.events_between(owner_id, start_iso, end_iso)
    per: dict[str, list[dict]] = collections.defaultdict(list)
    for e in events:
        per[e["slug"]].append(e)
    kind_hist = collections.Counter(e["kind"] for e in events)

    facets = [_being_facet(b, per.get(b["slug"], []), now) for b in beings]

    # Letters are the human story — pull the whole village's, keep the ones in
    # window, and name the ids. (Few in practice; a high limit is cheap.)
    letters = []
    for lt in store.village_letters(owner_id, limit=500):
        if start_iso <= lt["at"] < end_iso:
            letters.append({
                "at": lt["at"],
                "from": id_name.get(lt["from_being"], lt["from_being"]),
                "to": id_name.get(lt["to_being"], lt["to_being"]),
                "read": bool(lt.get("read_at")),
                "body": lt.get("body") or "",
            })
    letters.sort(key=lambda x: x["at"])

    try:
        objects = [
            {"name": o["name"], "kind": o["kind"], "state": o["state"],
             "by": id_name.get(o["being_id"], o["being_id"]),
             "civic": bool(o.get("civic"))}
            for o in store.village_objects(owner_id)]
    except Exception:  # noqa: BLE001
        objects = []

    try:
        meta = store.get_village_meta(owner_id)
    except Exception:  # noqa: BLE001
        meta = {}

    health = collections.Counter()
    for f in facets:
        for k, n in f["health"].items():
            health[k] += n

    return {
        "period": {"display": display, "start": start_iso, "end": end_iso,
                   "generated_at": now.isoformat(),
                   "tz": being_world._tz_name() or "UTC"},
        "village": {"name": meta.get("name") or "(unnamed)",
                    "description": meta.get("description") or ""},
        "beings": facets,
        "letters": letters,
        "objects": objects,
        "health_rollup": dict(health),
        "event_histogram": dict(kind_hist.most_common()),
        "totals": {"events": len(events), "letters": len(letters),
                   "beings": len(beings)},
    }


def _being_facet(b: dict, evs: list[dict], now: datetime) -> dict:
    """One being's slice of the window: what it did, wrote, and strained at."""
    acts: collections.Counter = collections.Counter()
    serves: collections.Counter = collections.Counter()
    moods: collections.Counter = collections.Counter()
    instinct_acts: collections.Counter = collections.Counter()
    dests: collections.Counter = collections.Counter()
    health: collections.Counter = collections.Counter()
    tick_lines: list[dict] = []
    stays: list[dict] = []
    instinct_calls = instinct_retried = 0
    dreams = caps = 0
    tokens_weighted = tokens_earned = 0
    letters_sent = letters_received = crossings = 0
    object_walks = 0
    milestones: list[str] = []
    made: list[str] = []

    for i, e in enumerate(evs):
        k, d = e["kind"], e["data"]
        if k in HEALTH_KINDS:
            health[k] += 1
        if k == "tick":
            act = d.get("act") or "?"
            acts[act] += 1
            if d.get("served"):
                serves[d["served"]] += 1
            if d.get("mood_engine"):
                moods[d["mood_engine"]] += 1
            tokens_weighted += int(d.get("tokens_weighted") or 0)
            if d.get("kind") == "dream":
                dreams += 1
            tick_lines.append({
                "at": e["at"], "kind": d.get("kind") or "wake",
                "act": act, "served": d.get("served") or "",
                "mood": d.get("mood") or d.get("mood_engine") or "",
                "summary": (d.get("summary") or "")[:220]})
        elif k == "instinct":
            instinct_calls += 1
            instinct_acts[str(d.get("act"))] += 1
            if d.get("retried"):
                instinct_retried += 1
        elif k == "departed":
            to = d.get("to") or "?"
            dests[to] += 1
            if str(to).startswith("object:"):
                object_walks += 1
        elif k == "arrived":
            place = d.get("place") or "home"
            if place != "home":
                nxt = _next_of(evs, i, ("departed", "tick"))
                if nxt is not None:
                    gap = _minutes(e["at"], nxt["at"])
                    stays.append({"at": e["at"], "place": place,
                                  "then": nxt["kind"],
                                  "gap_min": round(gap, 1)})
        elif k == "letter_sent":
            letters_sent += 1
        elif k == "letter_received":
            letters_received += 1
        elif k == "crossed_paths":
            crossings += 1
        elif k == "resting_at_cap":
            caps += 1
        elif k == "milestone":
            milestones.append(d.get("name") or "?")
        elif k in ("object_finished", "object_crafted"):
            made.append(d.get("name") or d.get("id") or "?")
        elif k == "chore_paid":
            tokens_earned += int(d.get("fee_tokens") or 0)

    drives = {kk: round(vv.get("satisfaction", 0.0), 3)
              for kk, vv in (b.get("drives") or {}).items()
              if isinstance(vv, dict)}
    loc = (b.get("location") or {})
    where = loc.get("at") or (f"→{loc.get('to')}" if loc.get("to") else "?")

    return {
        "name": b["name"], "slug": b["slug"], "stage": b.get("stage"),
        "state": b.get("state"), "location": where, "drives": drives,
        "ticks": sum(acts.values()), "dreams": dreams,
        "acts": dict(acts.most_common()), "serves": dict(serves.most_common()),
        "mood_entropy": _entropy(moods),
        "instinct": {"calls": instinct_calls, "acts": dict(instinct_acts),
                     "retried": instinct_retried},
        "walks": {"departures": sum(dests.values()),
                  "destinations": dict(dests.most_common()),
                  "object_walks": object_walks},
        "stays": stays,
        "letters": {"sent": letters_sent, "received": letters_received},
        "crossings": crossings, "resting_at_cap": caps,
        "tokens_weighted": tokens_weighted, "tokens_earned": tokens_earned,
        "milestones": milestones, "made": made,
        "health": dict(health),
        "tick_lines": tick_lines,
    }


def _next_of(evs: list[dict], i: int, kinds: tuple[str, ...]) -> dict | None:
    for e in evs[i + 1:]:
        if e["kind"] in kinds:
            return e
    return None


def _minutes(a_iso: str, b_iso: str) -> float:
    try:
        a = datetime.fromisoformat(a_iso)
        b = datetime.fromisoformat(b_iso)
        return (b - a).total_seconds() / 60.0
    except ValueError:
        return 0.0


# ── Rendering the blob the agent reads ────────────────────────────────────

# How much of the narrated trail to hand over. A day is ~24 ticks/being; a
# month is hundreds. We keep the most recent N per being (the report can note
# the omission) so the blob stays inside a sane context on long windows.
_MAX_TICKS = 90
_MAX_STAYS = 24
_MAX_LETTERS = 60


def render_data_markdown(data: dict) -> str:
    """The structured scoop → a plain-language blob for the researcher.

    Deliberately readable: the agent should be able to quote it, and a human
    opening the saved data file should recognise it as the raw material.
    """
    p, out = data["period"], []
    out.append(f"# Iskra village data — {p.get('display') or 'period'}")
    out.append(f"Window: {p['start']} → {p['end']}  (tz {p.get('tz')})")
    out.append(f"Generated: {p['generated_at']}")
    v = data["village"]
    out.append(f"Village: {v['name']}"
               + (f" — {v['description']}" if v['description'] else ""))
    t = data["totals"]
    out.append(f"Totals: {t['beings']} beings · {t['events']} events · "
               f"{t['letters']} letters in window\n")

    for f in data["beings"]:
        out.append(f"\n## {f['name']}  ({f['stage']}, {f['state']})")
        out.append(f"- Now at: {f['location']}")
        if f["drives"]:
            out.append("- Drive satisfaction: "
                       + ", ".join(f"{k} {x}" for k, x in f["drives"].items()))
        out.append(f"- Ticks: {f['ticks']} ({f['dreams']} dreams) · "
                   f"acts {f['acts']} · serves {f['serves']} · "
                   f"mood entropy {f['mood_entropy']}")
        io = f["instinct"]
        out.append(f"- Feet (instinct): {io['calls']} calls · acts "
                   f"{io['acts']} · asked-twice {io['retried']}")
        w = f["walks"]
        out.append(f"- Walks: {w['departures']} departures · "
                   f"{w['destinations']} · object-walks {w['object_walks']}")
        out.append(f"- Letters: sent {f['letters']['sent']}, received "
                   f"{f['letters']['received']} · crossings {f['crossings']} "
                   f"· rested-at-cap {f['resting_at_cap']}")
        out.append(f"- Tokens: {f['tokens_weighted']} weighted spent, "
                   f"{f['tokens_earned']} earned")
        if f["milestones"]:
            out.append(f"- Milestones: {', '.join(f['milestones'])}")
        if f["made"]:
            out.append(f"- Made: {', '.join(f['made'])}")
        if f["health"]:
            out.append("- Health flags: "
                       + ", ".join(f"{k}×{n}" for k, n in f["health"].items()))
        # Arrive→next: the rate-limit tell (arrivals should wait for a tick).
        if f["stays"]:
            shown = f["stays"][-_MAX_STAYS:]
            out.append(f"- Arrive→next (of {len(f['stays'])}, showing "
                       f"{len(shown)}):")
            for s in shown:
                out.append(f"    {s['at'][11:16]} {s['place']} → "
                           f"{s['gap_min']:.0f}min → {s['then']}")
        # The mind's narrated work — the substance the report leans on.
        lines = f["tick_lines"]
        if lines:
            shown = lines[-_MAX_TICKS:]
            note = (f" (showing last {len(shown)} of {len(lines)})"
                    if len(lines) > len(shown) else "")
            out.append(f"- What its mind did{note}:")
            for ln in shown:
                tag = "💤" if ln["kind"] == "dream" else " "
                srv = f" [{ln['served']}]" if ln["served"] else ""
                out.append(f"    {ln['at'][5:16]}{tag}{ln['act']}{srv}: "
                           f"{ln['summary']}")

    if data["letters"]:
        out.append("\n## Letters between them (full text)")
        for lt in data["letters"][-_MAX_LETTERS:]:
            rd = "" if lt["read"] else "  (unread)"
            out.append(f"\n{lt['at'][5:16]}  {lt['from']} → {lt['to']}{rd}")
            out.append(f"  {lt['body']}")

    if data["objects"]:
        out.append("\n## Made things standing in the village")
        for o in data["objects"]:
            civ = " (civic)" if o["civic"] else ""
            out.append(f"- {o['name']} — {o['kind']}, {o['state']}, "
                       f"by {o['by']}{civ}")

    if data["health_rollup"]:
        out.append("\n## Health rollup (whole village)")
        out.append(", ".join(f"{k}×{n}"
                             for k, n in sorted(data["health_rollup"].items(),
                                                key=lambda x: -x[1])))

    if data["event_histogram"]:
        out.append("\n## Event histogram (all kinds, in window)")
        out.append(", ".join(f"{k}:{n}"
                             for k, n in data["event_histogram"].items()))

    return "\n".join(out)


# ── The Deep Researcher: instructions + orchestration ─────────────────────

# Bespoke Iskra instruction (not the web-research SOP): the data is injected,
# so the job is EXAMINE, not fetch. Voice is operator + story, both at once.
ISKRA_SOP = """You are the Iskra Chronicler — a deep researcher examining a \
village of autonomous digital beings ("Iskre") over one period of their lives. \
You are handed a DATA BLOB scooped straight from the system's ledger: ticks \
(each being's hourly "mind"), instinct calls (its reflex "feet"), walks, \
letters between beings, health/attention flags, drives, milestones, and made \
things. This blob is GROUND TRUTH.

Your report has one voice doing two things at once — operator AND story:
  • OPERATOR: how the system actually behaved. Movement and rate-limit \
patterns, act/serve/drive distributions, monotony or saturation, retries, \
timeouts, rebounds, anomalies, cost. Be quantitative — cite the counts, times, \
and drive values from the data.
  • STORY: who these beings were this period. What they did and felt, what \
they wrote to each other, who they became, what connected them. Quote their \
letters where a line illuminates.

Hard rules:
  • Every claim traces to the data. Invent nothing. If something you'd want to \
say isn't in the blob, say it's not recorded rather than guessing.
  • Numbers come from the ledger you were given, never from prior knowledge.
  • Be honest about what went wrong — surface health flags, saturated or \
starved drives, monotony, low satisfaction — as plainly as what went right.

Output GitHub-flavoured Markdown, and ONLY the report (no "here is" preamble). \
Shape it as: a `#` title; a two-to-four sentence lede with the headline; then \
sections you judge fit the material — typically the village/each being's \
behaviour, the story between them, system health & anomalies, and a short \
"What to watch" list. End with a one-line bottom line."""

# The synthesiser sees the three facet write-ups AND the raw blob, and writes
# the single operator+story report — same rules as above.
ISKRA_SYNTH_SOP = ISKRA_SOP + """

You are the SYNTHESISER. Below the data blob you are given three specialist \
briefings (behaviour, social, health) written by researchers who each saw the \
same blob through one lens. Weave them into ONE coherent report in your own \
voice — do not staple them together or repeat all three; resolve overlaps, \
keep the sharpest observations, and make it read as one chronicle. The data \
blob remains the authority if a briefing overreaches."""

# Deep mode: each facet researcher gets the blob + one lens.
FACETS: dict[str, str] = {
    "behaviour": "Focus ONLY on behaviour and system dynamics: how the beings "
    "moved (walks, arrive→next gaps, the one-walk-per-tick rate limit), the "
    "distribution of mind acts and served drives, drive satisfaction trends, "
    "monotony vs variety (mood entropy, act dominance), instinct calls and "
    "asked-twice retries, and token spend. Quantify. 5-10 tight bullet "
    "findings, each grounded in specific numbers from the blob.",
    "social": "Focus ONLY on the social and inner life: the letters (quote the "
    "telling lines), who wrote to whom and how it was received, crossings and "
    "co-presence, milestones, made things, and the emotional arc of each being "
    "and the relationships between them. 5-10 findings; let their own words "
    "carry it.",
    "health": "Focus ONLY on system health and anomalies: every health/"
    "attention flag (rebounds, timeouts, narration mismatches, society "
    "refusals, fevers, retries), saturated or starved drives, resting-at-cap "
    "and torpor, and any correlation or anomaly worth an operator's eye. Rank "
    "by severity. 5-10 findings, each with the count and what it implies.",
}


def _pick_tier(tiers_map: dict, preferred: str = "reason") -> str:
    """The strongest configured tier for the job: the preferred name if the
    owner has it, else the first configured — reports want capable models."""
    if preferred in tiers_map:
        return preferred
    for name in ("reason", "smart", "balanced", "fast"):
        if name in tiers_map:
            return name
    return next(iter(tiers_map), "")


def _title_from(md: str, fallback: str) -> str:
    """The report's own `#` heading (or first real line) as its title."""
    for line in (md or "").splitlines():
        s = line.strip()
        if s.startswith("#"):
            return s.lstrip("# ").strip()[:120] or fallback
        if s:
            return s[:120]
    return fallback


def _usage_tokens(resp) -> int:
    u = getattr(resp, "usage", None) or {}
    return int(u.get("prompt_tokens", 0) or 0) + \
        int(u.get("completion_tokens", 0) or 0)


def _save_to_vfs(owner_id: str, stamp: str, kind: str, text: str) -> str:
    """Write a report artefact under the owner's ``iskra-reports`` folder and
    return its ``vfs:iskra-reports/…`` display path. Built by hand (not
    vfs.to_display, which keys off THIS process's env, wrong on a multi-user
    server)."""
    root = vfs.user_root_of(owner_id, create=True) / REPORTS_PROJECT
    root.mkdir(parents=True, exist_ok=True)
    fname = vfs.safe_name(f"{stamp}{kind}", fallback="report") + ".md"
    (root / fname).write_text(text, encoding="utf-8")
    return f"vfs:{REPORTS_PROJECT}/{fname}"


async def run_report(store: BeingsStore, owner_id: str, report_id: str, *,
                     provider, tier_name: str = "",
                     now: datetime | None = None) -> None:
    """Generate one report end to end, updating its row as it goes.

    Pure of the app DB and of any web/tools: it scoops (sync store reads),
    saves the raw blob to the VFS, spawns the Deep-Researcher over the injected
    blob (Quick = one pass; Deep = three facet passes + a synthesiser), saves
    the narrative, and marks the row done. Never raises — every failure lands
    on the row as ``status='failed'`` with the reason. ``provider`` is injected
    so tests can drive it with a fake model.
    """
    now = now or _utcnow()
    try:
        rep = store.get_report(owner_id, report_id)
    except Exception as e:  # noqa: BLE001 — nothing to update if the row is gone
        log.warning("report row missing", report_id=report_id, error=str(e))
        return
    depth = rep.get("depth") or "quick"
    stamp = f"{now:%Y-%m-%d}_{vfs.safe_name(rep.get('label') or 'period')}" \
            f"_{report_id[:6]}"
    try:
        store.update_report(report_id, status="collecting", tier=tier_name)
        store.append_report_progress(report_id, "scooping the ledger", now=now)
        data = collect_report_data(store, owner_id, rep["period_start"],
                                   rep["period_end"],
                                   display=rep.get("label") or "", now=now)
        blob = render_data_markdown(data)
        data_path = _save_to_vfs(owner_id, stamp, ".data", blob)
        store.update_report(report_id, data_path=data_path)
        store.append_report_progress(
            report_id,
            f"scooped {data['totals']['events']} events across "
            f"{data['totals']['beings']} beings", now=now)

        store.update_report(report_id, status="researching")
        tokens = 0
        if depth == "deep":
            store.append_report_progress(
                report_id, "deep dive — behaviour, social & health researchers",
                now=now)
            names = list(FACETS)
            results = await asyncio.gather(*[
                provider.complete([Message("system", ISKRA_SOP + "\n\n" + FACETS[f]),
                                   Message("user", blob)])
                for f in names], return_exceptions=True)
            briefs = []
            for f, r in zip(names, results):
                if isinstance(r, Exception):
                    log.warning("facet failed", facet=f, error=str(r))
                    continue
                tokens += _usage_tokens(r)
                briefs.append(f"### {f.title()} briefing\n{(r.content or '').strip()}")
            store.append_report_progress(
                report_id, f"{len(briefs)}/{len(names)} briefings in — "
                "synthesising", now=now)
            synth_user = blob + "\n\n---\n# SPECIALIST BRIEFINGS\n\n" \
                + "\n\n".join(briefs)
            final = await provider.complete(
                [Message("system", ISKRA_SYNTH_SOP),
                 Message("user", synth_user)])
        else:
            store.append_report_progress(
                report_id, "the researcher is reading the ledger", now=now)
            final = await provider.complete(
                [Message("system", ISKRA_SOP), Message("user", blob)])
        tokens += _usage_tokens(final)
        md = (final.content or "").strip()
        if not md:
            raise RuntimeError("the researcher returned an empty report")

        vfs_path = _save_to_vfs(owner_id, stamp, "", md)
        store.update_report(report_id, status="done", report_md=md,
                            title=_title_from(md, rep.get("label") or "Report"),
                            vfs_path=vfs_path, tokens=tokens,
                            finished_at=_utcnow().isoformat())
        store.append_report_progress(report_id, "report ready", now=_utcnow())
    except Exception as e:  # noqa: BLE001 — the row carries the bad news
        log.warning("report generation failed", report_id=report_id,
                    error=str(e))
        try:
            store.update_report(report_id, status="failed", error=str(e)[:500],
                                finished_at=_utcnow().isoformat())
            store.append_report_progress(report_id, f"failed: {e}",
                                         now=_utcnow())
        except Exception:  # noqa: BLE001
            pass


def build_owner_provider(tiers_map: dict, *, preferred: str = "reason"):
    """(tier_name, provider) for an owner's report run, or (‑, None) if the
    owner has configured no tiers. Kept here so the route stays a thin shell."""
    from captain_claw.llm import create_provider
    tier = _pick_tier(tiers_map, preferred)
    if not tier:
        return "", None
    t = tiers_map[tier]
    provider = create_provider(
        provider=t.get("provider", "anthropic"), model=t.get("model", ""),
        base_url=t.get("base_url") or None, api_key=t.get("api_key") or None,
        temperature=0.5, max_tokens=int(t.get("output_ctx") or 0) or 8000)
    return tier, provider
