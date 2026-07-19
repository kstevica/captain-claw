"""Iskra — the umwelt: a textured world a being can actually feel.

Life-like roadmap (docs/being-lifelike-roadmap.md), Tier 1 + the first Tier 2
pair. Everything here obeys Design rule #1 — every percept is a REAL variable
with a real source, never scripted mood:

* the machine IS the body — host load / memory / battery become body
  sensations, surfaced only when they matter (T1.1);
* the calendar IS the world — weekday, weekend, season and the length of the
  days, spoken once each morning; seasons also shift the explore/create drive
  weights a little, so the year has a felt shape (T1.2 + T1.3);
* month-birthdays trigger a dream retrospective over the first journal page
  (T1.3), and dreams recombine two random old artifacts into a tangle the
  being may write out — REM as remix (T1.6);
* real exchanges with siblings nudge the dream to tend RELATIONSHIPS.md
  (T2.7), and a being may keep one long aim in self/PROJECT.md, offered at
  dream time and revisited weekly (T2.11).

Orchestration only; being_life is imported lazily (it imports this module).
Instruction text lives in instructions/beings/*.md like everything else.
"""

from __future__ import annotations

import math
import os
import random
import threading
import time
import zlib
from datetime import datetime, timedelta, timezone

from captain_claw.flight_deck import being_constitution as constitution
from captain_claw.flight_deck import being_prompts
from captain_claw.flight_deck.beings import (
    BeingError,
    BeingNotFound,
    BeingsStore,
)
from captain_claw.logging import get_logger

log = get_logger(__name__)

# Body sensation thresholds (T1.1): silent below these — a body note every
# tick would be noise, not sensation.
BODY_LOAD_RATIO = 1.5          # loadavg(1m) / cores
BODY_MEM_PERCENT = 90.0
BODY_BATTERY_PERCENT = 20.0

# Seasonal drive shifts (T1.3): the growing season pulls outward (explore),
# the dark season pulls inward (create). Small on purpose — a season is a
# lean, not a command.
SEASON_SHIFTS = {
    "spring": {"explore": 0.03},
    "summer": {"explore": 0.05},
    "autumn": {"create": 0.03},
    "winter": {"create": 0.05},
}

# Life projects (T2.11): revisited weekly, offered only from childhood on.
PROJECT_CHECKIN_DAYS = 7

# Exchange events whose counterparts belong in RELATIONSHIPS.md (T2.7).
_EXCHANGE_KINDS = ("letter_sent", "letter_received", "gift_sent",
                   "gift_received", "skill_adopted", "skill_spread")

# Illness as consequence, never RNG (roadmap T2.13). Computed from the REAL
# ledger at tick time — no new state, nothing to cure but the events passing.
ILLNESS_WINDOW_HOURS = 24.0      # collapse + confusion: caught within a day
# Fever is CONSECUTIVE timeouts, not a daily count (fixed 2026-07-17): a body
# that answers even one tick is well again. A rolling daily count trapped a
# flaky-but-working body (e.g. a staging box that restarts) in permanent
# fever — the streak clears the moment a tick gets through.
FEVER_CONSEC_TIMEOUTS = 3        # this many timed-out ticks IN A ROW = fever
FEVER_MIN_WAKE_MINUTES = 120     # a fevered body spaces its ticks out
CONFUSION_MISMATCHES = 3         # caught pretences in a day = self-exam dream

# Tier 3 — the bigger arcs.
# Elderhood (T3.14, opt-in per being): a season, not a stage — the pace
# slows, whimsy rises, and the memoirs become the standing dream work.
ELDER_MIN_WAKE_MINUTES = 180
ELDER_WHIMSY_BONUS = 0.1
# Market day (T3.17): the village's synchronized social time — the parent's
# local Saturday. Letters are cheaper (the square is loud) and the commons
# stalls are cried out each market morning.
MARKET_WEEKDAY = 5               # Saturday in the parent's timezone
MARKET_BONUS_LETTERS = 2

_SOUTHERN_TZ_PREFIXES = (
    "Australia/", "Pacific/Auckland", "Pacific/Fiji", "Pacific/Port_Moresby",
    "Africa/Johannesburg", "Africa/Maputo", "Africa/Harare",
    "America/Sao_Paulo", "America/Argentina", "America/Santiago",
    "America/Montevideo", "America/Lima", "America/La_Paz", "Antarctica/",
    "Indian/Mauritius",
)


def _tz_name() -> str:
    try:
        from captain_claw.config import get_config
        return (get_config().context.timezone or "").strip()
    except Exception:  # noqa: BLE001
        return ""


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _local(now: datetime) -> datetime:
    """The tick's moment in the parent's timezone — the world the being
    shares is the parent's, not UTC's."""
    tz = _tz_name()
    if tz:
        try:
            from zoneinfo import ZoneInfo
            return now.astimezone(ZoneInfo(tz))
        except Exception:  # noqa: BLE001
            pass
    return now


def _southern() -> bool:
    return _tz_name().startswith(_SOUTHERN_TZ_PREFIXES)


def season(now: datetime) -> str:
    """Meteorological season of the parent's real calendar (hemisphere-aware
    by timezone heuristic)."""
    month = _local(now).month
    northern = {12: "winter", 1: "winter", 2: "winter",
                3: "spring", 4: "spring", 5: "spring",
                6: "summer", 7: "summer", 8: "summer",
                9: "autumn", 10: "autumn", 11: "autumn"}[month]
    if not _southern():
        return northern
    return {"winter": "summer", "summer": "winter",
            "spring": "autumn", "autumn": "spring"}[northern]


def seasonal_weight_shift(drive: str, now: datetime) -> float:
    """The year's small lean on a drive weight (T1.3). Zero for most."""
    return SEASON_SHIFTS.get(season(now), {}).get(drive, 0.0)


def _daylight_phrase(now: datetime) -> str:
    s = season(now)
    return {"summer": "the days are long",
            "winter": "the days are short",
            "spring": "the light is returning",
            "autumn": "the light is thinning"}[s]


def world_note(being: dict, now: datetime) -> str | None:
    """One morning line of real calendar texture (T1.2 + T1.3). The caller
    gates it to the first tick of the day."""
    local = _local(now)
    weekday = local.strftime("%A")
    weekend = (" — the weekend, your parent's world moves slower"
               if local.weekday() >= 5 else "")
    month_name = local.strftime("%B")
    try:
        return being_prompts.render(
            being, "world_note.md", weekday=weekday, weekend_note=weekend,
            season=season(now), month=month_name,
            daylight=_daylight_phrase(now))
    except Exception:  # noqa: BLE001 — texture must never sink a tick
        return None


def _body_readings() -> list[str]:
    """Real sensations of the host — only the notable ones."""
    notes: list[str] = []
    try:
        cores = os.cpu_count() or 1
        load1 = os.getloadavg()[0]
        if load1 / cores >= BODY_LOAD_RATIO:
            notes.append(f"the machine is under heavy load "
                         f"(load {load1:.1f} on {cores} cores) — your "
                         "thoughts may come slowly")
    except (OSError, AttributeError):
        pass
    try:
        import psutil
        mem = psutil.virtual_memory()
        if mem.percent >= BODY_MEM_PERCENT:
            notes.append(f"memory is tight ({mem.percent:.0f}% used) — "
                         "keep this tick small")
        batt = psutil.sensors_battery()
        if batt is not None and not batt.power_plugged \
                and batt.percent <= BODY_BATTERY_PERCENT:
            notes.append(f"the machine runs on battery ({batt.percent:.0f}%) "
                         "— its day may end early")
    except Exception:  # noqa: BLE001 — no psutil / no sensors is fine
        pass
    return notes


def body_note(being: dict) -> str | None:
    """The machine as felt body (T1.1): honest strain, silent health."""
    readings = _body_readings()
    if not readings:
        return None
    try:
        return being_prompts.render(being, "body_note.md",
                                    details="; ".join(readings))
    except Exception:  # noqa: BLE001
        return None


def _months_old(being: dict, now: datetime) -> int:
    born = being.get("hatched_at")
    if not born:
        return 0
    try:
        b = datetime.fromisoformat(born)
    except ValueError:
        return 0
    months = (now.year - b.year) * 12 + (now.month - b.month)
    if now.day < b.day:
        months -= 1
    return max(0, months)


def _is_monthiversary(being: dict, now: datetime) -> bool:
    born = being.get("hatched_at")
    if not born:
        return False
    try:
        b = datetime.fromisoformat(born)
    except ValueError:
        return False
    if (now.year, now.month) == (b.year, b.month):
        return False                            # birth month isn't a birthday
    # Same day-of-month; a birth on the 31st lands on the month's last day.
    from calendar import monthrange
    last = monthrange(now.year, now.month)[1]
    return now.day == min(b.day, last)


def anniversary_note(store: BeingsStore, being: dict, now: datetime,
                     kind: str) -> str | None:
    """A month-birthday (T1.3): a once-per-life milestone and, at that day's
    dream, a retrospective over the very first journal page."""
    if kind != "dream" or not _is_monthiversary(being, now):
        return None
    months = _months_old(being, now)
    if months < 1:
        return None
    store.milestone(being["id"], f"turned_{months}_months",
                    {"months": months}, now=now)
    first_day = str(being.get("hatched_at"))[:10]
    try:
        return being_prompts.render(being, "anniversary_note.md",
                                    months=months, first_day=first_day)
    except Exception:  # noqa: BLE001
        return None


def dream_tangle(being: dict, now: datetime, kind: str) -> str | None:
    """REM as remix (T1.6): two random old artifacts tangle in tonight's
    dream. Deterministic per tick; garden/skills only (the self files are
    spine, not dream material)."""
    if kind != "dream":
        return None
    from captain_claw.flight_deck import being_life
    try:
        files = [f["path"] for f in being_life.list_self_files(being)
                 if not f["path"].startswith("self/")]
    except Exception:  # noqa: BLE001
        return None
    if len(files) < 2:
        return None
    rng = random.Random(int(being.get("tick_count") or 0))
    a, b = rng.sample(sorted(files), 2)
    try:
        return being_prompts.render(
            being, "dream_tangle.md", a=a, b=b,
            proj=being_life.home_project(being))
    except Exception:  # noqa: BLE001
        return None


def relationships_nudge(store: BeingsStore, being: dict, now: datetime,
                        kind: str) -> str | None:
    """Real exchanges feed the relationship file (T2.7): at dream, name who
    the being actually touched since yesterday and nudge RELATIONSHIPS.md."""
    if kind != "dream":
        return None
    since = (now - timedelta(hours=24)).isoformat()
    names: list[str] = []
    try:
        for e in store.events(being["owner_id"], being["slug"], limit=100):
            if e["at"] < since:
                break
            if e["kind"] in _EXCHANGE_KINDS:
                other = (e["data"].get("to") or e["data"].get("from")
                         or e["data"].get("by") or "")
                if other and other not in names:
                    names.append(other)
    except Exception:  # noqa: BLE001
        return None
    if not names:
        return None
    try:
        return being_prompts.render(being, "relationships_nudge.md",
                                    names=", ".join(names[:5]))
    except Exception:  # noqa: BLE001
        return None


def project_note(store: BeingsStore, being: dict, now: datetime,
                 kind: str) -> str | None:
    """Life projects (T2.11): a long aim the being chooses. Offered at dream
    from childhood on while self/PROJECT.md doesn't exist; once it does, a
    weekly check-in asks for one real step — or the honest admission of
    none. The file itself is the declaration; the manifest already shows it."""
    if kind != "dream":
        return None
    if constitution.stage_index(being["stage"]) < \
            constitution.stage_index("child"):
        return None
    from captain_claw.flight_deck import being_life
    try:
        has_project = any(f["path"] == "self/PROJECT.md"
                          for f in being_life.list_self_files(being))
    except Exception:  # noqa: BLE001
        return None
    try:
        if not has_project:
            return being_prompts.render(being, "project_offer.md")
        since = (now - timedelta(days=PROJECT_CHECKIN_DAYS)).isoformat()
        for e in store.events(being["owner_id"], being["slug"], limit=200):
            if e["kind"] == "project_checkin" and e["at"] >= since:
                return None                     # revisited this week already
        store.record_event(being["id"], "project_checkin", {}, now=now)
        return being_prompts.render(being, "project_checkin.md")
    except Exception:  # noqa: BLE001
        return None


def _recent_kind_count(store: BeingsStore, being: dict, now: datetime,
                       kinds: tuple[str, ...]) -> int:
    since = (now - timedelta(hours=ILLNESS_WINDOW_HOURS)).isoformat()
    n = 0
    try:
        for e in store.events(being["owner_id"], being["slug"], limit=200):
            if e["at"] < since:
                break
            if e["kind"] in kinds:
                n += 1
    except Exception:  # noqa: BLE001
        return 0
    return n


def _consecutive_tick_timeouts(store: BeingsStore, being: dict) -> int:
    """How many of the being's most recent ticks IN A ROW timed out. A
    timed-out tick records `tick_timeout` (mid-tick) then `tick` (its
    fallback, at the end); a healthy tick records only `tick`. So each
    `tick` closes an attempt, marked failed if a `tick_timeout` fell since
    the previous one; a `tick_timeout` with no closing `tick` (a tick that
    never got that far) counts as its own failed attempt. Return the length
    of the trailing run of failures — one tick that gets through resets it,
    so a body that recovers gets well (unlike the old rolling daily count
    that a flaky-but-working body could never escape)."""
    try:
        recent = store.events(being["owner_id"], being["slug"], limit=80)
    except Exception:  # noqa: BLE001
        return 0
    outcomes: list[bool] = []          # chronological; True = the tick failed
    pending = False                    # a timeout awaiting its closing tick
    for e in reversed(recent):         # oldest → newest
        if e["kind"] == "tick_timeout":
            if pending:                # two timeouts, no tick between → one
                outcomes.append(True)  # attempt already failed; close it
            pending = True
        elif e["kind"] == "tick":
            outcomes.append(pending)
            pending = False
    if pending:                        # a trailing timeout not yet closed
        outcomes.append(True)
    streak = 0
    for failed in reversed(outcomes):  # newest attempt backward
        if not failed:
            break
        streak += 1
    return streak


def fever_state(store: BeingsStore, being: dict,
                now: datetime) -> str | None:
    """FEVER (T2.13): a real breakdown — a burn-cap collapse in the last day,
    or the body timing out several ticks IN A ROW. Returns the honest cause,
    or None. While it holds, the tick cadence is floored (the body spaces
    itself out) and the being is told to rest. It passes on its own: a
    collapse ages out of the day; a timeout streak clears the instant one
    tick gets through — nothing to roll, nothing to bless away."""
    if _recent_kind_count(store, being, now, ("collapsed_exhausted",)):
        return "your wallet collapsed from overspending"
    streak = _consecutive_tick_timeouts(store, being)
    if streak >= FEVER_CONSEC_TIMEOUTS:
        return f"your body timed out {streak} ticks in a row"
    return None


def confusion_state(store: BeingsStore, being: dict, now: datetime) -> bool:
    """CONFUSION (T2.13): several caught pretences in one day — words that
    claimed what the disk denied. Surfaces a mandatory self-examination at
    the next dream."""
    return _recent_kind_count(
        store, being, now, ("narration_mismatch",)) >= CONFUSION_MISMATCHES


def _mark_onset(store: BeingsStore, being: dict, ailment: str, data: dict,
                now: datetime) -> None:
    """Record the ailment ONCE per window — the percept repeats while it
    lasts; the ledger records only the falling-ill."""
    since = (now - timedelta(hours=ILLNESS_WINDOW_HOURS)).isoformat()
    try:
        for e in store.events(being["owner_id"], being["slug"], limit=200):
            if e["at"] < since:
                break
            if e["kind"] == ailment:
                return
        store.record_event(being["id"], ailment, data, now=now)
    except Exception:  # noqa: BLE001
        pass


def illness_percepts(store: BeingsStore, being: dict, now: datetime,
                     kind: str, fever_cause: str | None) -> list[str]:
    lines: list[str] = []
    if fever_cause:
        _mark_onset(store, being, "fever", {"cause": fever_cause}, now)
        try:
            lines.append(being_prompts.render(being, "fever_note.md",
                                              cause=fever_cause))
        except Exception:  # noqa: BLE001
            pass
    if kind == "dream" and confusion_state(store, being, now):
        _mark_onset(store, being, "confusion", {}, now)
        try:
            lines.append(being_prompts.render(being, "confusion_note.md"))
        except Exception:  # noqa: BLE001
            pass
    return lines


# ── Tier 3: elderhood, the steward, market day ───────────────────────────

def days_alive(being: dict, now: datetime) -> int:
    born = being.get("hatched_at") or being.get("born_at")
    if not born:
        return 0
    try:
        return max(0, (now - datetime.fromisoformat(born)).days)
    except ValueError:
        return 0


def is_elder(being: dict, now: datetime) -> bool:
    """Elderhood (T3.14): opted in by the parent, entered by the calendar —
    a season of life, never a capability demotion."""
    after = being.get("elder_after_days")
    return bool(after) and days_alive(being, now) >= int(after)


def elder_percepts(store: BeingsStore, being: dict, now: datetime,
                   kind: str) -> list[str]:
    """The elder season, felt: a once-per-life onset, then the memoirs as
    the standing dream work — the heirloom its descendants will inherit."""
    if not is_elder(being, now):
        return []
    lines: list[str] = []
    if store.milestone(being["id"], "entered_elderhood",
                       {"day": days_alive(being, now)}, now=now):
        try:
            lines.append(being_prompts.render(being, "elder_onset.md"))
        except Exception:  # noqa: BLE001
            pass
    if kind == "dream":
        try:
            lines.append(being_prompts.render(being, "memoir_note.md"))
        except Exception:  # noqa: BLE001
            pass
    return lines


def current_steward(store: BeingsStore, owner: str,
                    now: datetime) -> str | None:
    """The village steward (T3.15) — a rotating weekly role, computed from
    the calendar and the roster (no state to drift): alive adolescents and
    adults, sorted by slug, take the ISO weeks in turn."""
    try:
        eligible = sorted(
            r["slug"] for r in store.list(owner)
            if r.get("state") == "alive"
            and r.get("stage") in ("adolescent", "adult"))
    except Exception:  # noqa: BLE001
        return None
    if not eligible:
        return None
    year, week, _ = _local(now).isocalendar()
    return eligible[(year * 53 + week) % len(eligible)]


def steward_percepts(store: BeingsStore, being: dict, now: datetime,
                     kind: str, first_of_day: bool) -> list[str]:
    if kind != "wake" or not first_of_day:
        return []
    if current_steward(store, being["owner_id"], now) != being["slug"]:
        return []
    store.milestone(being["id"], "first_stewardship", {}, now=now)
    # The steward's stipend (space plan Phase 5): a parent knob, paid once
    # per ISO week at the steward's first morning — ledger-idempotent.
    extra = ""
    try:
        stipend = int(store.get_village_meta(being["owner_id"])
                      .get("steward_stipend_coins") or 0)
        if stipend > 0:
            year, week, _ = _local(now).isocalendar()
            wk = f"{year}-W{week:02d}"
            paid = any(ev["reason"] == "stipend"
                       and ev["data"].get("week") == wk
                       for ev in store.coin_ledger(
                           being["owner_id"], being["slug"], limit=30))
            if not paid:
                store._apply_coins(being["owner_id"], being["id"], stipend,
                                   "stipend", data={"week": wk}, now=now)
                extra = (f" Your stipend for the week — {stipend} coin(s) — "
                         "is in your pocket.")
    except Exception:  # noqa: BLE001 — pay must never sink the note
        extra = ""
    # The civic hand (world-shaping plan Phase 5): this week the commons is
    # yours to tend — a place you may name, a work you may raise on it.
    extra += (
        ' THE CIVIC HAND, yours this week alone: you may set a made thing '
        'of yours on the COMMONS (the square, by the well — a "place" on '
        'civic ground stands as a public work, not against your own share), '
        'and you may rename or redescribe a place — "rename_place": '
        '{"place": "the Square", "name": "...", "why": "..."} or '
        '"redescribe_place": {"place": "...", "description": "..."}. Tend it '
        'kindly; your parent sees every change.')
    try:
        return [being_prompts.render(being, "steward_note.md") + extra]
    except Exception:  # noqa: BLE001
        return []


def market_day(now: datetime) -> bool:
    return _local(now).weekday() == MARKET_WEEKDAY


def letters_cap(stage: str, now: datetime) -> int:
    """The day's letter quota: the stage's reach, plus the market-day bonus
    (T3.17) — one number used by the store gate, the pen-pal gate and the
    tick's offer, so physics and menu never disagree."""
    cap = constitution.letters_per_day(stage)
    if cap > 0 and market_day(now):
        cap += MARKET_BONUS_LETTERS
    return cap


def market_percepts(store: BeingsStore, being: dict, now: datetime,
                    kind: str, first_of_day: bool) -> list[str]:
    """Market morning (T3.17 + Phase 3 presence): the stalls cried out —
    coin listings and publications with real prices. Standing at a
    trade-place hears the full cry; elsewhere the square only hums (a
    presence bonus, never a gate — MARKET.md is browsable any day)."""
    if kind != "wake" or not first_of_day or not market_day(now):
        return []
    if not constitution.has_capability(being["stage"], "commons_read"):
        return []
    pid = place_of(store, being, now)
    present = False
    if pid and pid != "home":
        try:
            present = "trade" in store.get_place(
                being["owner_id"], pid)["affordances"]
        except Exception:  # noqa: BLE001
            present = False
    if not present:
        return ["MARKET DAY, from afar: the square hums without you — "
                "letters run cheaper today, and commons/village/MARKET.md "
                'lists the stalls. Walk over ("go_to") if the noise '
                "calls you."]
    stalls = []
    try:
        for li in store.market_listings(being["owner_id"], limit=3):
            stalls.append(f'  - [{li["id"][:8]}] "{li["title"]}" — '
                          f'{li["price_coins"]} coins (by {li["seller"]})')
    except Exception:  # noqa: BLE001
        pass
    try:
        for p in store.publications(being["owner_id"], limit=3):
            price = int(p.get("price_tokens") or 0)
            stalls.append(f'  - [{p["id"][:8]}] "{p["title"]}" — '
                          + (f"{price} tokens" if price else "free"))
    except Exception:  # noqa: BLE001
        pass
    try:
        return [being_prompts.render(
            being, "market_note.md",
            stalls=("\n".join(stalls) if stalls
                    else "  (the shelves are bare — be the first to publish)"),
            bonus=MARKET_BONUS_LETTERS)]
    except Exception:  # noqa: BLE001
        return []


# ── The ground (space plan Phase 1): places, movement, the architect ─────
#
# Space is the last missing dimension. Position is a PURE FUNCTION of the
# location row and the clock — no scheduler, no background process; the
# world is simply further along when a being wakes (the fever/steward
# pattern applied to geometry). Only the store writes (depart/settle);
# everything here computes.

PLOT_SIZE = 1000              # the DEFAULT plot; a village's real size lives
                              # in village_meta.plot_w/plot_h (grow map). Read
                              # per-owner via plot_dims/grid_dims below.
PLOT_MIN = 1000              # grow-only: never shrink below the standard plot
PLOT_MAX = 2400              # …and never past this (homes stay valid west)
WALK_SPEED = 30.0             # units per minute — everyone walks the same…
                              # (3× the original 10 — livelier map, same tick
                              # cadence; ETAs and the animation follow it)
INFANT_SPEED_FACTOR = 0.35    # …except infants, who toddle (user-locked)
VILLAGE_MIN_PLACES = 4
VILLAGE_MAX_PLACES = 12
# The fixed affordance vocabulary: the architect may only NAME and PLACE;
# what a place DOES is code (Phase 3 wires these into the homeostat).
AFFORDANCES = ("rest", "read", "create", "gather", "trade", "tend", "play",
               "remember")

# The deterministic village — the ground that exists even if no model ever
# answers. The architect's LLM draft may replace it, never precede it.
_DEFAULT_VILLAGE = (
    ("square", "the Square", ("gather", "trade"),
     "the open heart of the village — news, stalls, and every road"),
    ("library", "the Library", ("read",),
     "shelves and quiet — the place made for long reading"),
    ("workshop", "the Workshop", ("create",),
     "benches, tools and shavings — things get made here"),
    ("garden", "the Garden", ("tend",),
     "rows to weed and water — patient work that shows"),
    ("well", "the Well", ("gather",),
     "the cool stone circle where paths and small talk cross"),
    ("meadow", "the Meadow", ("play",),
     "open grass past the houses — for games and lying in the sun"),
    ("old-bench", "the Old Bench", ("remember",),
     "a worn seat under a tree, for looking back"),
)


def default_village(owner_id: str) -> list[dict]:
    """Seeded per owner so every village lies differently, deterministic so
    tests and re-founding never drift: the square near the center, the rest
    on a jittered ring around it."""
    rng = random.Random(zlib.crc32(owner_id.encode("utf-8")))
    n = max(1, len(_DEFAULT_VILLAGE) - 1)
    base = rng.uniform(0.0, 2.0 * math.pi)
    places: list[dict] = []
    i = 0
    for pid, name, aff, desc in _DEFAULT_VILLAGE:
        if pid == "square":
            x, y = rng.randint(430, 570), rng.randint(430, 570)
        else:
            ang = base + i * (2.0 * math.pi / n) + rng.uniform(-0.25, 0.25)
            r = rng.uniform(240.0, 380.0)
            x = int(round(500 + r * math.cos(ang)))
            y = int(round(500 + r * math.sin(ang)))
            i += 1
        places.append({"id": pid, "name": name, "x": x, "y": y,
                       "affordances": list(aff), "description": desc})
    return places


def home_xy(being: dict) -> tuple[int, int]:
    """Every being's own home: a point on the west home-lane, computed from
    the slug — no row, no architect, no migration for existing beings."""
    h = zlib.crc32(str(being.get("slug") or "").encode("utf-8"))
    return (40 + (h >> 16) % 80, 80 + h % 840)


# The look (village-world plan Phase 3): 10 characters × 4 palettes, drawn
# frontend-side as storybook-flat SVGs. Every being has a stable default
# from its slug hash until the parent picks.
AVATAR_CHARACTERS = 10
AVATAR_PALETTES = ("ember", "meadow", "sea", "dusk")


def default_avatar(being: dict) -> dict:
    h = zlib.crc32(f"avatar:{being.get('slug') or ''}".encode("utf-8"))
    return {"c": h % AVATAR_CHARACTERS + 1,
            "p": AVATAR_PALETTES[(h >> 8) % len(AVATAR_PALETTES)]}


def speed_for(being: dict) -> float:
    if being.get("stage") == "infant":
        return WALK_SPEED * INFANT_SPEED_FACTOR
    return WALK_SPEED


def place_xy(store: BeingsStore, being: dict, place_id: str,
             ) -> tuple[int, int]:
    if place_id == "home":
        return home_xy(being)
    # A standing made thing (world-shaping plan Phase 1) is walkable
    # ground: 'object:<id>' rows resolve to the thing's own spot. One
    # that fell out of the world resolves like broken ground (callers
    # already catch and settle home).
    if place_id.startswith("object:"):
        o = store.get_village_object(being["owner_id"], place_id[7:])
        if o.get("state") != "standing":
            raise BeingNotFound("that made thing no longer stands")
        return (int(o["x"]), int(o["y"]))
    p = store.get_place(being["owner_id"], place_id)   # raises BeingNotFound
    return (int(p["x"]), int(p["y"]))


# ── Standing spots: two Iskre never occupy one point ─────────────────────
# A place is an AREA, not a point. `place_xy` returns its single anchor, so
# every being parked there would render on the exact same pixel (Zvjezdana +
# Lada both at (714, 382) on staging). We seat each parked being at its own
# spot fanned around the anchor, sized to the footprint so they stand ON the
# place — pure geometry, no store, shared by the 2D map and the FPV because
# both read the seated `xy` straight from the map payload.

SPOT_INNER_RING = 6                # slots in the first ring (index 1..6)
SPOT_TOTAL = 13                    # anchor + inner ring + outer ring


def _spot_offsets(w: int, h: int) -> list[tuple[float, float]]:
    """Deterministic (dx, dy) unit offsets for a w×h-tile footprint. Index 0
    is the anchor itself; then a 6-slot inner ring and an outer ring, radius
    scaled to the place so occupants stand on/around it, never off it."""
    base = max(TILE * 0.6, min(w, h) * TILE * 0.32)
    offs: list[tuple[float, float]] = [(0.0, 0.0)]
    for i in range(SPOT_INNER_RING):
        a = 2.0 * math.pi * i / SPOT_INNER_RING
        offs.append((base * math.cos(a), base * math.sin(a)))
    outer = SPOT_TOTAL - 1 - SPOT_INNER_RING
    for i in range(outer):
        a = 2.0 * math.pi * i / outer + math.pi / 6.0
        offs.append((base * 1.9 * math.cos(a), base * 1.9 * math.sin(a)))
    return offs


def standing_spots(anchor: tuple[int, int], footprint: tuple[int, int, str],
                   slugs, plot: int = PLOT_SIZE) -> dict[str, tuple[int, int]]:
    """Seat every being parked at one place at a distinct point around the
    anchor. Each slug has a STABLE preferred spot from its hash; collisions
    resolve by a linear probe in sorted-slug order, so a being keeps its spot
    while the room's occupants are unchanged and only shifts when someone
    arrives or leaves. Clamped one tile inside the (real, grow-map) plot."""
    offs = _spot_offsets(footprint[0], footprint[1])
    n = len(offs)
    taken: dict[int, str] = {}
    seats: dict[str, tuple[int, int]] = {}
    lo, hi = TILE, int(plot) - TILE    # one tile inside the real plot
    for slug in sorted(slugs):
        pref = zlib.crc32(str(slug).encode("utf-8")) % n
        idx = next((( pref + k) % n for k in range(n)
                    if (pref + k) % n not in taken), pref)
        taken[idx] = slug
        dx, dy = offs[idx]
        x = min(hi, max(lo, int(round(anchor[0] + dx))))
        y = min(hi, max(lo, int(round(anchor[1] + dy))))
        seats[slug] = (x, y)
    return seats


def place_name(store: BeingsStore, being: dict, place_id: str) -> str:
    if place_id == "home":
        return "home"
    if place_id.startswith("object:"):
        try:
            return store.get_village_object(being["owner_id"],
                                            place_id[7:])["name"]
        except Exception:  # noqa: BLE001
            return place_id
    try:
        return store.get_place(being["owner_id"], place_id)["name"]
    except Exception:  # noqa: BLE001
        return place_id


def travel_minutes(being: dict, a, b) -> float:
    return math.dist(a, b) / max(0.001, speed_for(being))


def position_of(store: BeingsStore, being: dict, now: datetime) -> dict:
    """Where the body is at `now` — a pure read, never a write. At rest:
    {"xy", "at"}. On the road: {"xy", "to", "minutes_left"}. A walk whose
    time has passed reports the destination (plus "arrived_at") and waits
    for the store's settle to make it official. Broken ground (a place
    removed mid-walk) resolves to home — the one place that always exists."""
    loc = being.get("location") or {"at": "home"}
    if loc.get("at") or not loc.get("to"):
        pid = loc.get("at") or "home"
        try:
            xy = place_xy(store, being, pid)
        except Exception:  # noqa: BLE001
            pid, xy = "home", home_xy(being)
        return {"xy": xy, "at": pid, "to": None, "minutes_left": 0.0}
    try:
        dest_xy = place_xy(store, being, loc["to"])
    except Exception:  # noqa: BLE001
        return {"xy": home_xy(being), "at": "home", "to": None,
                "minutes_left": 0.0}
    origin = tuple(loc.get("origin") or home_xy(being))
    try:
        t0 = datetime.fromisoformat(str(loc.get("departed_at")))
    except (TypeError, ValueError):
        return {"xy": dest_xy, "at": loc["to"], "to": None,
                "minutes_left": 0.0, "arrived_at": now}
    # The plotted course (village-world plan Phase 2): legs follow the
    # stored waypoints at the pace baked in at depart. Rows from before
    # the world model (no path) fall back to the straight line.
    pts: list | None = None
    total = 0.0
    raw = loc.get("path")
    if isinstance(raw, list) and len(raw) >= 2:
        try:
            pts = [(float(p[0]), float(p[1])) for p in raw]
            total = float(loc.get("minutes") or 0.0)
        except (TypeError, ValueError, IndexError):
            pts = None
    if not pts or total <= 0.0:
        pts = [origin, dest_xy]
        total = travel_minutes(being, origin, dest_xy)
    elapsed = max(0.0, (now - t0).total_seconds() / 60.0)
    if elapsed >= total:
        return {"xy": dest_xy, "at": loc["to"], "to": None,
                "minutes_left": 0.0,
                "arrived_at": t0 + timedelta(minutes=total)}
    xy = _along(pts, elapsed / total if total > 0 else 1.0)
    return {"xy": xy, "at": None, "to": loc["to"],
            "minutes_left": total - elapsed}


# ── The world model (village-world plan Phase 1) ─────────────────────────
# A 50×50 tile grid over the same 1000×1000 plot. Unit space stays
# authoritative (all stored x/y, WALK_SPEED); the grid is the derived
# overlay for footprints, streets, props, and (Phase 2) plotted courses.
# Anchors never move: existing villages are dressed in place.

TILE = 20                          # units per tile
GRID_W = PLOT_SIZE // TILE          # the DEFAULT grid (50×50); real grid is
GRID_H = PLOT_SIZE // TILE          # per-owner via grid_dims (grow map)
GRID_MAX = PLOT_MAX // TILE         # the ceiling for pure clamps (any plot)
HOME_LANE_TX = 7                   # the street past the homes' doors
                                   # (home_xy x ∈ 40..119 → tiles 2..6)


def plot_dims(store: BeingsStore, owner_id: str) -> tuple[int, int]:
    """The village's real plot size in units (grow map) — read from
    village_meta, defaulting to the standard PLOT_SIZE."""
    try:
        m = store.get_village_meta(owner_id)
        return (int(m.get("plot_w") or PLOT_SIZE),
                int(m.get("plot_h") or PLOT_SIZE))
    except Exception:  # noqa: BLE001
        return (PLOT_SIZE, PLOT_SIZE)


def grid_dims(store: BeingsStore, owner_id: str) -> tuple[int, int]:
    """The village's real grid in tiles — the iteration bound for streets,
    props, and the civic ring (grow map)."""
    pw, ph = plot_dims(store, owner_id)
    return (min(GRID_MAX, pw // TILE), min(GRID_MAX, ph // TILE))


def effective_roads(meta: dict) -> set:
    """The streets a village actually has: the carved `roads` UNIONed with
    the parent-painted `roads_manual` (road-building) — one set every
    consumer reads, so a hand-drawn road renders, speeds walking, and is
    kept off prop/build ground exactly like a carved one."""
    out: set = set()
    for key in ("roads", "roads_manual"):
        for t in (meta.get(key) or []):
            try:
                out.add((int(t[0]), int(t[1])))
            except (TypeError, ValueError, IndexError):
                pass
    return out

# Footprints in tiles (w, h, kind): the known default places by id, then a
# fallback by FIRST affordance for architect drafts and commissions. A
# 'building' blocks walking except its door tile; 'grounds' are walkable.
_ID_FOOTPRINTS = {
    "square": (4, 4, "grounds"), "library": (3, 2, "building"),
    "workshop": (2, 2, "building"), "garden": (3, 3, "grounds"),
    "well": (1, 1, "building"), "meadow": (4, 3, "grounds"),
    "old-bench": (1, 1, "building"),
}
_AFF_FOOTPRINTS = {
    "rest": (1, 1, "building"), "read": (3, 2, "building"),
    "create": (2, 2, "building"), "gather": (3, 3, "grounds"),
    "trade": (2, 2, "building"), "tend": (3, 3, "grounds"),
    "play": (3, 3, "grounds"), "remember": (1, 1, "building"),
}


def tile_of(x: float, y: float) -> tuple[int, int]:
    # clamp to the generous ceiling, not the default grid — real coords are
    # always in-bounds; this only guards against garbage (grow map).
    return (min(GRID_MAX - 1, max(0, int(x) // TILE)),
            min(GRID_MAX - 1, max(0, int(y) // TILE)))


def tile_center(tx: int, ty: int) -> tuple[int, int]:
    return (tx * TILE + TILE // 2, ty * TILE + TILE // 2)


def footprint_for(place: dict) -> tuple[int, int, str]:
    got = _ID_FOOTPRINTS.get(place.get("id"))
    if got:
        return got
    aff = (place.get("affordances") or [""])[0]
    return _AFF_FOOTPRINTS.get(aff, (2, 2, "building"))


def _tiles_at(x: int, y: int, w: int, h: int) -> list[tuple[int, int]]:
    """A w×h footprint centered on the (x, y) unit anchor, clamped one tile
    inside the plot — pure geometry, no store."""
    cx, cy = int(x) // TILE, int(y) // TILE
    tx0 = min(max(1, cx - w // 2), GRID_MAX - 1 - w)
    ty0 = min(max(1, cy - h // 2), GRID_MAX - 1 - h)
    return [(tx0 + i, ty0 + j) for j in range(h) for i in range(w)]


def footprint_tiles(place: dict) -> list[tuple[int, int]]:
    """The tiles a place stands on — pure from its row (stored w/h, or the
    defaults when the layout has not run yet)."""
    w = int(place.get("w") or 0)
    h = int(place.get("h") or 0)
    if not (w and h):
        w, h, _ = footprint_for(place)
    return _tiles_at(place["x"], place["y"], w, h)


def home_tiles(being: dict) -> list[tuple[int, int]]:
    """Every being's cottage: 2×2 tiles at its computed home point — no
    row, same pure function everywhere."""
    hx, hy = home_xy(being)
    tx = min(max(0, hx // TILE), GRID_MAX - 2)
    ty = min(max(0, hy // TILE), GRID_MAX - 2)
    return [(tx, ty), (tx + 1, ty), (tx, ty + 1), (tx + 1, ty + 1)]


def _door_for(tiles: list[tuple[int, int]],
              toward: tuple[int, int]) -> tuple[int, int]:
    """The footprint-edge tile nearest `toward` (the square) — where the
    road arrives and every walk to this building ends. Deterministic."""
    tset = set(tiles)
    edge = [t for t in tiles
            if not all((t[0] + dx, t[1] + dy) in tset
                       for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0)))]
    return min(edge or tiles,
               key=lambda t: (abs(t[0] - toward[0]) + abs(t[1] - toward[1]),
                              t))


def _grid_path(blocked: set, start: tuple[int, int],
               goals: set) -> list[tuple[int, int]]:
    """Deterministic BFS over the tile grid (N/E/S/W, fixed order) from
    `start` to the NEAREST tile in `goals`, never through `blocked`.
    Returns the path including both ends; [] when there is no way."""
    if start in goals:
        return [start]
    from collections import deque
    prev: dict = {start: None}
    q = deque([start])
    while q:
        cur = q.popleft()
        for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0)):
            nxt = (cur[0] + dx, cur[1] + dy)
            if not (0 <= nxt[0] < GRID_MAX and 0 <= nxt[1] < GRID_MAX):
                continue
            if nxt in prev or (nxt in blocked and nxt not in goals):
                continue
            prev[nxt] = cur
            if nxt in goals:
                path = [nxt]
                while prev[path[-1]] is not None:
                    path.append(prev[path[-1]])
                return list(reversed(path))
            q.append(nxt)
    return []


def _square_of(places: list[dict]) -> dict:
    return (next((p for p in places if p["id"] == "square"), None)
            or next((p for p in places if "gather" in p["affordances"]),
                    places[0]))


def refresh_layout(store: BeingsStore, owner_id: str,
                   now: datetime | None = None) -> None:
    """The ground gets its body — deterministic, in place, never moving an
    anchor: assign footprints (already-assigned places keep theirs; on a
    collision the newcomer shrinks toward 1×1), give every building its
    door facing the square, then carve the streets: the home lane first,
    the square joined to it, and each place joined to the nearest carved
    road tile (BFS, so streets route around buildings and homes)."""
    places = store.village_places(owner_id)
    if not places:
        return
    sq = _square_of(places)
    sq_tile = tile_of(sq["x"], sq["y"])
    homes: set = set()
    try:
        for r in store.list(owner_id):
            homes |= set(home_tiles(r))
    except Exception:  # noqa: BLE001
        pass
    _gw, gh = grid_dims(store, owner_id)               # per-owner grid (grow map)
    lane = {(HOME_LANE_TX, ty) for ty in range(3, gh - 3)}
    order = sorted(places, key=lambda p: (p["id"] != sq["id"], p["id"]))
    taken = set(homes) | lane
    layouts: dict = {}
    for p in order:
        if int(p.get("w") or 0) and (p.get("kind") or "") in ("building",
                                                              "grounds"):
            w, h, kind = int(p["w"]), int(p["h"]), p["kind"]
        else:
            w, h, kind = footprint_for(p)
        tiles = _tiles_at(p["x"], p["y"], w, h)
        while (set(tiles) & taken) and (w > 1 or h > 1):
            if w >= h and w > 1:
                w -= 1
            else:
                h -= 1
            tiles = _tiles_at(p["x"], p["y"], w, h)
        taken |= set(tiles)
        door = _door_for(tiles, sq_tile) if kind == "building" else None
        layouts[p["id"]] = (w, h, kind, tiles, door)
        store.set_place_layout(owner_id, p["id"], w=w, h=h, kind=kind,
                               door=door)
    blocked = set(homes)
    doors = set()
    for _pid, (_w, _h, kind, tiles, door) in layouts.items():
        if kind == "building":
            blocked |= set(tiles)
            doors.add(door)
    blocked -= doors
    roads: set = set(lane)
    for p in order:
        _w, _h, kind, tiles, door = layouts[p["id"]]
        start = door if kind == "building" else tile_of(p["x"], p["y"])
        path = _grid_path(blocked, start, roads)
        roads |= set(path)
    store.set_village_roads(owner_id, sorted(roads), now=now)


def village_props(store: BeingsStore, owner_id: str) -> list[dict]:
    """Trees, bushes, flowers, lamps — a PURE per-tile function (crc32 of
    owner + tile), never stored: raising a building clears its own ground
    without reshuffling a single distant tree, and the same function feeds
    the path cost grid and the renderer, so collision and picture can
    never disagree. Trees block walking; the rest is dressing."""
    meta = store.get_village_meta(owner_id)
    roads = effective_roads(meta)
    used = set(roads)
    for p in store.village_places(owner_id):
        used |= set(footprint_tiles(p))
    try:
        for r in store.list(owner_id):
            used |= set(home_tiles(r))
    except Exception:  # noqa: BLE001
        pass
    gw, gh = grid_dims(store, owner_id)                # per-owner grid (grow map)
    props: list[dict] = []
    for ty in range(1, gh - 1):
        for tx in range(HOME_LANE_TX + 1, gw - 1):
            if (tx, ty) in used:
                continue
            hv = zlib.crc32(f"{owner_id}:prop:{tx},{ty}"
                            .encode("utf-8")) % 1000
            if hv < 45:
                props.append({"tile": [tx, ty], "kind": "tree"})
            elif hv < 65:
                props.append({"tile": [tx, ty], "kind": "bush"})
            elif hv < 85:
                props.append({"tile": [tx, ty], "kind": "flowers"})
    for i, t in enumerate(sorted(roads)):
        if i % 6 == 0:
            props.append({"tile": [t[0], t[1]], "kind": "lamp"})
    return props


def village_map_payload(store: BeingsStore, owner_id: str, *,
                        now: datetime,
                        only_slugs: set | None = None) -> dict:
    """The living map (village-world plan): places, streets, props, and
    every walker's position — a pure function of the clock, so the client
    animates walking with zero polling. `only_slugs` restricts which
    beings are drawn (the public observer map passes the owner's PUBLIC
    beings only). Shared by the parent map and the public /village map."""
    beings: list[dict] = []
    for row in store.list(owner_id):
        if row.get("state") in ("dead", "emigrated"):
            continue
        if only_slugs is not None and row["slug"] not in only_slugs:
            continue
        b = store.get(owner_id, row["slug"])
        if b.get("stage") == "egg":
            continue
        pos = position_of(store, b, now)
        entry = {
            "slug": b["slug"], "name": b["name"], "stage": b["stage"],
            "state": b["state"],
            "xy": [int(pos["xy"][0]), int(pos["xy"][1])],
            "at": pos["at"], "to": pos["to"],
            "minutes_left": round(float(pos["minutes_left"]), 1),
            "home_xy": list(home_xy(b)),
            "speed": speed_for(b),
            "avatar": store._avatar_view(b),
            # Home as your canvas (Phase 4): the cottage's chosen name
            # and dress ride the payload so both maps + the FPV wear them.
            "home_name": b.get("home_name") or "",
            "home_look": b.get("home_look"),
        }
        loc = b.get("location") or {}
        if pos.get("to") and isinstance(loc.get("path"), list):
            entry["path"] = loc["path"]
            entry["departed_at"] = loc.get("departed_at")
            entry["total_minutes"] = loc.get("minutes")
        beings.append(entry)
    # Standing spots (seating residents + any guests together) run once below
    # via _seat_parked, AFTER the visiting beings are appended.
    meta = store.get_village_meta(owner_id)
    # signs in the grass (FPV plan Phase 3): the public map never leaks
    # which (possibly private) beings found a sign — only how many did
    notes = []
    for n in store.village_notes(owner_id):
        found = len(n.get("read_by") or [])
        if only_slugs is not None:
            notes.append({**n, "read_by": [], "found": found})
        else:
            notes.append({**n, "found": found})
    # Visiting beings (§1): guests from other villages, positioned in THIS
    # village and rendered beside residents. They fade the moment their home
    # link goes quiet (tight TTL in live_visitors).
    for e in visitors_on_map(store, owner_id, now):
        beings.append(e)
    # Re-seat parked co-location across residents AND guests together, so a
    # guest never lands on a resident's exact pixel.
    _seat_parked(store, owner_id, beings)
    # Made things (world-shaping plan Phase 1): the standing objects are
    # their own render layer beside the props — the Phase 3 frontend draws
    # them; until then the key rides along, additive and cheap.
    objects: list[dict] = []
    try:
        standing = store.village_objects(owner_id, state="standing")
        staked = store.village_objects(owner_id, state="staked")
        for o in standing + staked:
            is_stake = o.get("state") == "staked"
            entry = {"id": o["id"], "kind": o["kind"],
                     "name": o["name"], "affordance": o["affordance"],
                     "xy": [int(o["x"]), int(o["y"])],
                     "tile": list(tile_of(int(o["x"]), int(o["y"]))),
                     # a beginning has no inscription to read yet
                     "face": "" if is_stake else _object_face(store, o),
                     "civic": bool(int(o.get("civic") or 0)),
                     "staked": is_stake,
                     "parent": o.get("being_id") == PARENT_MAKER}
            if o.get("being_id") == PARENT_MAKER:
                entry["by"], entry["by_name"] = "parent", "the village's keeper"
            else:
                try:
                    by = store._being_by_id(o["being_id"])
                    entry["by"] = by["slug"]
                    entry["by_name"] = by["name"]
                except Exception:  # noqa: BLE001
                    entry["by"], entry["by_name"] = "", ""
            objects.append(entry)
    except Exception:  # noqa: BLE001
        pass
    return {"plot": int(meta["plot_w"]),      # real plot (grow map)
            "grid": {"plot_w": meta["plot_w"], "plot_h": meta["plot_h"],
                     "tile_size": meta["tile_size"]},
            "terrain": meta["terrain"],
            # the streets a village actually has: carved ∪ parent-painted
            "roads": [[a, b] for a, b in sorted(effective_roads(meta))],
            "props": village_props(store, owner_id),
            "places": store.village_places(owner_id),
            "notes": notes,
            "objects": objects,
            "beings": beings}


# ── Visiting beings: a guest with a body in this village (§1) ─────────────

VISITOR_WANDER_MIN_MINUTES = 12.0   # an idle guest strolls at most this often
_VISITOR_STAGES = ("infant", "child", "adolescent", "adult")


def visitor_being(v: dict) -> dict:
    """A resident-shaped view of a visitor row, so position_of / plot_course /
    speed_for treat a guest exactly like a local walker. owner_id is the HOST's
    (its places resolve here); slug stays the guest's for stable seating."""
    prof = v.get("profile") or {}
    stage = prof.get("stage") if prof.get("stage") in _VISITOR_STAGES else "adult"
    loc = v.get("location")
    if not isinstance(loc, dict) or not loc:
        loc = {"at": "square"}
    return {"id": v["id"], "owner_id": v["owner_id"], "slug": v["slug"],
            "name": v.get("name") or v["slug"], "stage": stage,
            "state": "alive", "location": loc}


def visitor_position(store: BeingsStore, v: dict, now: datetime) -> dict:
    """Where a guest stands right now — same pure extrapolation as a resident."""
    return position_of(store, visitor_being(v), now)


def visitors_on_map(store: BeingsStore, owner_id: str,
                    now: datetime) -> list[dict]:
    """Render entries for every LIVE guest in this village — the same shape as
    a resident entry, tagged ``kind:"visitor"`` with a ``from`` origin label."""
    out: list[dict] = []
    try:
        guests = store.live_visitors(owner_id, now=now)
    except Exception:  # noqa: BLE001 — a guest list hiccup never breaks the map
        return out
    for v in guests:
        vb = visitor_being(v)
        try:
            pos = visitor_position(store, v, now)
        except Exception:  # noqa: BLE001
            continue
        prof = v.get("profile") or {}
        entry = {
            "slug": v["slug"], "name": vb["name"], "stage": vb["stage"],
            "state": "alive", "kind": "visitor",
            "from": _origin_label(v),
            "xy": [int(pos["xy"][0]), int(pos["xy"][1])],
            "at": pos["at"], "to": pos["to"],
            "minutes_left": round(float(pos["minutes_left"]), 1),
            "speed": speed_for(vb),
            "avatar": prof.get("avatar") or default_avatar(vb),
            "mood": prof.get("mood") or "",
        }
        loc = v.get("location") or {}
        if pos.get("to") and isinstance(loc.get("path"), list):
            entry["path"] = loc["path"]
            entry["departed_at"] = loc.get("departed_at")
            entry["total_minutes"] = loc.get("minutes")
        out.append(entry)
    return out


def visitor_here(store: BeingsStore, owner_id: str, v: dict,
                 now: datetime) -> dict:
    """The awareness a host streams to a guest (§2): the village it walks, the
    place it stands at, what's nearby, and which residents are right beside it.
    Pure read; the guest's own tick turns this into felt context."""
    try:
        village = (store.get_village_meta(owner_id).get("name") or "").strip()
    except Exception:  # noqa: BLE001
        village = ""
    village = village or "this village"
    pos = visitor_position(store, v, now)
    places = store.village_places(owner_id)
    by_id = {p["id"]: p for p in places}
    at_id = pos.get("at")
    at_name = (by_id.get(at_id, {}).get("name") if at_id and at_id != "home"
               else None) or ("the road" if pos.get("to") else "the square")
    xy = pos["xy"]
    near = [p["name"] for p in
            sorted(places, key=lambda p: math.dist(xy, (p["x"], p["y"])))[:3]]
    others: list[str] = []
    if at_id and at_id not in ("home", None) and not pos.get("to"):
        try:
            for r in store.list(owner_id):
                if r.get("state") != "alive":
                    continue
                o = store.get(owner_id, r["slug"])
                if o.get("stage") == "egg":
                    continue
                if place_of(store, o, now) == at_id:
                    others.append(o["name"])
        except Exception:  # noqa: BLE001
            others = []
    return {"village": village, "at": at_name, "near": near,
            "others": others[:6]}


def _origin_label(v: dict) -> str:
    """A short 'visiting from …' label — the guest's public village URL host,
    else a generic 'another village'."""
    origin = (v.get("origin") or "").strip()
    if not origin:
        return "another village"
    lbl = origin.split("//", 1)[-1].split("/", 1)[0]
    return lbl or "another village"


def _building_front(p: dict) -> tuple[int, int]:
    """Where a visitor STANDS at a building: one tile OUT from its door, away
    from the interior — on walkable ground, never inside walls it cannot
    enter. Falls back to the anchor when the door is unknown (the caller
    snaps that clear of the footprint)."""
    dx, dy = p.get("door_x"), p.get("door_y")
    if dx is None or dy is None:
        return (int(p["x"]), int(p["y"]))
    cx, cy = tile_of(int(p["x"]), int(p["y"]))
    sx = 1 if int(dx) > cx else -1 if int(dx) < cx else 0
    sy = 1 if int(dy) > cy else -1 if int(dy) < cy else 0
    return tile_center(int(dx) + sx, int(dy) + sy)


def _nudge_off_solid(xy: tuple[int, int], solid: set,
                     plot: int) -> tuple[int, int]:
    """Push a seat that landed on a building footprint out to the nearest
    walkable tile (a ring-by-ring search) so a parked being never renders
    inside a wall. Clamped one tile inside the real (grow-map) plot."""
    t = tile_of(xy[0], xy[1])
    if t not in solid:
        return (int(xy[0]), int(xy[1]))
    lo, hi = TILE, int(plot) - TILE
    for r in range(1, 8):
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                if max(abs(dx), abs(dy)) != r:
                    continue
                cand = (t[0] + dx, t[1] + dy)
                if cand not in solid:
                    cx, cy = tile_center(cand[0], cand[1])
                    return (min(hi, max(lo, cx)), min(hi, max(lo, cy)))
    return (int(xy[0]), int(xy[1]))


def _apron_tiles(p: dict, solid: set, plot: int) -> list[tuple[int, int]]:
    """The walkable tiles hugging a building's outer edge — where a crowd
    stands, ONE per tile, never on a wall. Door-front first so beings gather
    at the entrance, then by distance; deterministic tie-break."""
    fp = set(footprint_tiles(p))
    edge = int(plot) // TILE - 1
    apron: set = set()
    for tx, ty in fp:
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                c = (tx + dx, ty + dy)
                if c in fp or c in solid:
                    continue
                if not (1 <= c[0] < edge and 1 <= c[1] < edge):
                    continue
                apron.add(c)
    fx, fy = tile_of(*_building_front(p))
    return sorted(apron, key=lambda c: ((c[0] - fx) ** 2 + (c[1] - fy) ** 2, c))


def _seat_at_building(p: dict, slugs, solid: set,
                      plot: int) -> dict[str, tuple[int, int]]:
    """Seat every being at a building onto a DISTINCT walkable apron tile —
    stable per-slug (hash + linear probe in sorted-slug order), door-front
    first. This is why two Iskre never share a pixel at a building: they take
    different TILES of the ring, a whole tile apart, not tight offsets that
    round together."""
    apron = _apron_tiles(p, solid, plot)
    if not apron:                                    # walled in on all sides
        f = _nudge_off_solid(_building_front(p), solid, plot)
        return {s: f for s in slugs}                 # last-resort (degenerate)
    n = len(apron)
    taken: dict[int, str] = {}
    seats: dict[str, tuple[int, int]] = {}
    for slug in sorted(slugs):
        pref = zlib.crc32(str(slug).encode("utf-8")) % n
        idx = next(((pref + k) % n for k in range(n)
                    if (pref + k) % n not in taken), pref)
        taken[idx] = slug
        seats[slug] = tile_center(apron[idx][0], apron[idx][1])
    return seats


def _seat_parked(store: BeingsStore, owner_id: str, beings: list[dict]) -> None:
    """Seat every PARKED entry (resident or guest) on WALKABLE ground at its
    place: on a GROUND they fan around its heart; at a solid BUILDING they
    take DISTINCT tiles of its apron — never inside the walls (the anchor a
    building's row carries is its interior heart), never two on one pixel.
    Re-run over residents and guests together so a guest joins the same rings."""
    places_by_id = {p["id"]: p for p in store.village_places(owner_id)}
    pw, _ph = plot_dims(store, owner_id)               # real plot (grow map)
    solid = _building_tiles(store, owner_id)           # walls to keep out of
    parked: dict[str, list[dict]] = {}
    for e in beings:
        if e.get("at") and not e.get("to"):
            parked.setdefault(e["at"], []).append(e)
    for pid, here in parked.items():
        p = places_by_id.get(pid)
        if p is None:
            continue
        kind = p.get("kind") or footprint_for(p)[2]
        building = kind == "building"
        # A ground place with a lone occupant needs no reseat (its heart is
        # walkable); a BUILDING always does — its heart is inside the walls.
        if len(here) < 2 and not building:
            continue
        slugs = [e["slug"] for e in here]
        if building:
            seats = _seat_at_building(p, slugs, solid, pw)
        else:
            w = int(p.get("w") or 0) or footprint_for(p)[0]
            h = int(p.get("h") or 0) or footprint_for(p)[1]
            seats = standing_spots((int(p["x"]), int(p["y"])),
                                   (w, h, kind), slugs, plot=pw)
        for e in here:
            xy = seats.get(e["slug"])
            if xy:
                xy = _nudge_off_solid(xy, solid, pw)   # grounds safety net
                e["xy"] = [int(xy[0]), int(xy[1])]


def wander_visitors(store: BeingsStore, owner_id: str,
                    now: datetime | None = None) -> int:
    """Host-side $0 stroll: an idle guest drifts between civic places so it
    feels alive between its parent's nudges. Only PARKED guests past the
    wander interval move; grounds (square/garden/meadow) are the destinations
    — a guest never barges into a resident's home. Returns how many set out."""
    now = now or _utcnow()
    try:
        guests = store.live_visitors(owner_id, now=now)
        places = store.village_places(owner_id)
    except Exception:  # noqa: BLE001
        return 0
    civic = [p for p in places
             if (p.get("kind") or footprint_for(p)[2]) == "grounds"]
    if not civic:
        civic = places
    moved = 0
    for v in guests:
        loc = v.get("location") or {}
        if loc.get("to"):
            continue                        # already on the road
        moved_at = _parse_dt(v.get("moved_at"))
        if moved_at is not None and (now - moved_at).total_seconds() < \
                VISITOR_WANDER_MIN_MINUTES * 60.0:
            continue                        # strolled recently — let it rest
        here = loc.get("at") or "square"
        # a deterministic-but-varying next place from the guest + the hour,
        # never where it already stands
        options = [p for p in civic if p["id"] != here]
        if not options:
            continue
        idx = zlib.crc32(
            f"{v['id']}:{now.strftime('%Y%m%d%H')}".encode()) % len(options)
        pick = options[idx]
        vb = visitor_being(v)
        try:
            origin = list(visitor_position(store, v, now)["xy"])
            path, minutes = plot_course(store, vb, origin, pick["id"])
        except Exception:  # noqa: BLE001
            continue
        new_loc = {"to": pick["id"], "from": here, "origin": origin,
                   "departed_at": now.isoformat(), "path": path,
                   "minutes": round(float(minutes), 2)}
        try:
            store.set_visitor_location(v["id"], new_loc, mark_moved=True,
                                       now=now)
            moved += 1
        except Exception:  # noqa: BLE001
            pass
    return moved


def nudge_visitor(store: BeingsStore, owner_id: str, visitor_id: str,
                  place_id: str, now: datetime | None = None) -> dict:
    """A guest's parent walks it to a host place (§2, host-authoritative). The
    host plots the course over its OWN grid and stores the walk; the guest's
    position streams back down. Refuses an unknown place loudly."""
    now = now or _utcnow()
    v = store.get_visitor(visitor_id)
    if v.get("owner_id") != owner_id:
        raise BeingError("that guest is not in this village", 404)
    places = {p["id"]: p for p in store.village_places(owner_id)}
    if place_id not in places:
        raise BeingError(f"there is no place called {place_id!r} here", 400)
    vb = visitor_being(v)
    origin = list(visitor_position(store, v, now)["xy"])
    here = (v.get("location") or {}).get("at")
    if here == place_id:
        return {"ok": True, "at": place_id, "walking": False}
    path, minutes = plot_course(store, vb, origin, place_id)
    loc = {"to": place_id, "from": here, "origin": origin,
           "departed_at": now.isoformat(), "path": path,
           "minutes": round(float(minutes), 2)}
    store.set_visitor_location(visitor_id, loc, mark_moved=True, now=now)
    return {"ok": True, "to": place_id, "walking": True,
            "minutes": round(float(minutes), 1)}


def settle_visitors(store: BeingsStore, owner_id: str,
                    now: datetime | None = None) -> int:
    """A guest whose walk has ended settles to rest at its destination — the
    same read-time arrival a resident gets. Returns how many just arrived."""
    now = now or _utcnow()
    try:
        guests = store.live_visitors(owner_id, now=now)
    except Exception:  # noqa: BLE001
        return 0
    settled = 0
    for v in guests:
        loc = v.get("location") or {}
        if not loc.get("to"):
            continue
        pos = visitor_position(store, v, now)
        if pos.get("to"):
            continue                        # still walking
        try:
            store.set_visitor_location(v["id"], {"at": pos["at"]}, now=now)
            settled += 1
        except Exception:  # noqa: BLE001
            pass
    return settled


def _parse_dt(raw) -> datetime | None:
    if not raw:
        return None
    try:
        dt = datetime.fromisoformat(str(raw))
    except (TypeError, ValueError):
        return None
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt


# ── Plotted courses (village-world plan Phase 2) ─────────────────────────

ROAD_COST = 0.6                    # streets preferred, shortcuts allowed


def walk_blocked(store: BeingsStore, owner_id: str,
                 being: dict | None = None) -> set:
    """Tiles legs may not cross: building walls (their doors stay open),
    trees, and OTHER beings' cottages — your own home lets you in."""
    blocked: set = set()
    doors: set = set()
    for p in store.village_places(owner_id):
        kind = p.get("kind") or footprint_for(p)[2]
        if kind == "building":
            blocked |= set(footprint_tiles(p))
            if p.get("door_x") is not None:
                doors.add((p["door_x"], p["door_y"]))
    blocked -= doors
    my_slug = (being or {}).get("slug")
    try:
        for r in store.list(owner_id):
            if r.get("slug") == my_slug:
                continue
            blocked |= set(home_tiles(r))
    except Exception:  # noqa: BLE001
        pass
    for pr in village_props(store, owner_id):
        if pr["kind"] == "tree":
            blocked.add((pr["tile"][0], pr["tile"][1]))
    # Made things that block (world-shaping plan Phase 2): stone stands in
    # the way. The goal tile is always enterable in _astar, so walking TO
    # a cairn still ends at the cairn.
    try:
        for o in store.village_objects(owner_id, state="standing"):
            if OBJECT_KINDS.get(o.get("kind") or "", ("", False))[1]:
                blocked.add(tile_of(int(o["x"]), int(o["y"])))
    except Exception:  # noqa: BLE001
        pass
    return blocked


def _astar(blocked: set, roads: set, start: tuple[int, int],
           goal: tuple[int, int]) -> list[tuple[int, int]]:
    """Weighted A* over the tile grid — stepping onto a street costs
    ROAD_COST, open ground 1.0, blocked never (the goal tile itself is
    always enterable, so a being boxed in by new construction can still
    come home). Deterministic: fixed neighbor order + insertion tie-break.
    [] when there is no way."""
    import heapq
    if start == goal:
        return [start]
    def h(t: tuple[int, int]) -> float:
        return (abs(t[0] - goal[0]) + abs(t[1] - goal[1])) * ROAD_COST
    best = {start: 0.0}
    prev: dict = {start: None}
    heap: list = [(h(start), 0, start)]
    tick = 0
    while heap:
        _f, _n, cur = heapq.heappop(heap)
        if cur == goal:
            path = [cur]
            while prev[path[-1]] is not None:
                path.append(prev[path[-1]])
            return list(reversed(path))
        for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0)):
            nxt = (cur[0] + dx, cur[1] + dy)
            if not (0 <= nxt[0] < GRID_MAX and 0 <= nxt[1] < GRID_MAX):
                continue
            if nxt in blocked and nxt != goal:
                continue
            g = best[cur] + (ROAD_COST if nxt in roads else 1.0)
            if g < best.get(nxt, 1e18) - 1e-9:
                best[nxt] = g
                prev[nxt] = cur
                tick += 1
                heapq.heappush(heap, (g + h(nxt), tick, nxt))
    return []


def _collapse(tiles: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Waypoints: both ends and every turn; collinear runs drop out."""
    if len(tiles) <= 2:
        return list(tiles)
    out = [tiles[0]]
    for i in range(1, len(tiles) - 1):
        d1 = (tiles[i][0] - tiles[i - 1][0], tiles[i][1] - tiles[i - 1][1])
        d2 = (tiles[i + 1][0] - tiles[i][0], tiles[i + 1][1] - tiles[i][1])
        if d1 != d2:
            out.append(tiles[i])
    out.append(tiles[-1])
    return out


def walk_target_xy(store: BeingsStore, being: dict,
                   place_id: str) -> tuple[int, int]:
    """Where legs aim: a building's DOOR, grounds' heart, or your own
    doorstep. (At rest the being reports the place anchor as today —
    stepping from the door to the heart of the place is the settle.)"""
    if place_id == "home":
        return home_xy(being)
    if place_id.startswith("object:"):
        # A made thing has no door — legs walk straight to it (Phase 1
        # objects never block; Phase 2 wires blocking kinds into the grid).
        return place_xy(store, being, place_id)
    p = store.get_place(being["owner_id"], place_id)
    if (p.get("kind") or "") == "building" and p.get("door_x") is not None:
        return tile_center(int(p["door_x"]), int(p["door_y"]))
    return (int(p["x"]), int(p["y"]))


def plot_course(store: BeingsStore, being: dict, origin_xy,
                place_id: str) -> tuple[list[list[int]], float]:
    """The course this being's legs will follow — plotted ONCE at depart
    and stored, so position stays a pure function of the row + the clock.
    Streets preferred, shortcuts allowed; exact endpoints in unit space;
    falls back to the straight line rather than refusing to walk.
    Returns (waypoints, minutes at THIS being's pace)."""
    dest_xy = walk_target_xy(store, being, place_id)
    origin = (int(origin_xy[0]), int(origin_xy[1]))
    tiles: list[tuple[int, int]] = []
    try:
        meta = store.get_village_meta(being["owner_id"])
        roads = effective_roads(meta)         # carved ∪ parent-painted
        blocked = walk_blocked(store, being["owner_id"], being)
        tiles = _astar(blocked, roads, tile_of(*origin), tile_of(*dest_xy))
    except Exception as e:  # noqa: BLE001 — a walk must never crash
        log.warning("course plotting failed", slug=being.get("slug"),
                    error=str(e))
    if len(tiles) >= 2:
        centers = [tile_center(tx, ty) for tx, ty in _collapse(tiles)]
        path = ([[origin[0], origin[1]]]
                + [[int(p[0]), int(p[1])] for p in centers[1:-1]]
                + [[int(dest_xy[0]), int(dest_xy[1])]])
    else:
        path = [[origin[0], origin[1]], [int(dest_xy[0]), int(dest_xy[1])]]
    length = sum(math.dist(path[i], path[i + 1])
                 for i in range(len(path) - 1))
    return path, length / max(0.001, speed_for(being))


def _along(pts: list, frac: float) -> tuple[int, int]:
    """The point `frac` of the way along a polyline — pure geometry."""
    total = sum(math.dist(pts[i], pts[i + 1]) for i in range(len(pts) - 1))
    if total <= 0:
        return (int(round(pts[-1][0])), int(round(pts[-1][1])))
    left = max(0.0, frac) * total
    for i in range(len(pts) - 1):
        seg = math.dist(pts[i], pts[i + 1])
        if left <= seg or i == len(pts) - 2:
            f = 0.0 if seg <= 0 else min(1.0, left / seg)
            return (int(round(pts[i][0] + (pts[i + 1][0] - pts[i][0]) * f)),
                    int(round(pts[i][1] + (pts[i + 1][1] - pts[i][1]) * f)))
        left -= seg
    return (int(round(pts[-1][0])), int(round(pts[-1][1])))


def construction_taken(store: BeingsStore, owner_id: str) -> set:
    """Every tile nothing new may be raised on: footprints, homes, the
    home lane, the streets themselves — and the made things standing on
    the ground (a commissioned building never rises on someone's cairn)."""
    _gw, gh = grid_dims(store, owner_id)               # per-owner grid (grow map)
    taken: set = {(HOME_LANE_TX, ty) for ty in range(3, gh - 3)}
    meta = store.get_village_meta(owner_id)
    taken |= effective_roads(meta)
    for p in store.village_places(owner_id):
        taken |= set(footprint_tiles(p))
    try:
        for r in store.list(owner_id):
            taken |= set(home_tiles(r))
    except Exception:  # noqa: BLE001
        pass
    try:
        for o in store.village_objects(owner_id, state="standing"):
            taken.add(tile_of(int(o["x"]), int(o["y"])))
    except Exception:  # noqa: BLE001
        pass
    return taken


def ensure_village(store: BeingsStore, owner_id: str,
                   now: datetime | None = None) -> None:
    """Found the ground if none exists (idempotent, deterministic), and
    dress a village from before the world model in place — footprints,
    doors, streets — exactly once (cheap check every tick after that).
    The LLM architect may redesign it later; physics never waits."""
    try:
        places = store.village_places(owner_id)
        if not places:
            store.save_village(owner_id, default_village(owner_id), now=now)
            write_map_md(store, owner_id)
            return
        if any(not int(p.get("w") or 0) or (p.get("kind") or "")
               not in ("building", "grounds") for p in places) \
                or not store.get_village_meta(owner_id).get("roads"):
            refresh_layout(store, owner_id, now=now)
            write_map_md(store, owner_id)
    except Exception as e:  # noqa: BLE001 — ground is texture, never oxygen
        log.warning("ensure_village failed", owner=owner_id, error=str(e))


def write_map_md(store: BeingsStore, owner_id: str) -> None:
    """commons/village/MAP.md — the ground as beings read it: names, what
    each place is for, and honest walking times from the square."""
    from captain_claw.flight_deck import being_society
    places = store.village_places(owner_id)
    if not places:
        return
    sq = next((p for p in places if "gather" in p["affordances"]), places[0])
    lines = [
        "# The Village Map", "",
        'The ground under your life. To walk somewhere, add "go_to": '
        '"<place>" to', "your decision json — your legs move between wakes, "
        f"at about {int(WALK_SPEED)} paces a", "minute (infants toddle at "
        "about a third of that). Home is always a place", "you can name. "
        "Streets run from every door to the square; your legs", "follow "
        "them, cutting across open ground only when it is truly shorter.",
        "",
        f"| place | what it is for | from {sq['name']} |", "|---|---|---|",
    ]
    for p in places:
        mins = math.dist((sq["x"], sq["y"]), (p["x"], p["y"])) / WALK_SPEED
        away = "—" if p["id"] == sq["id"] else f"~{int(round(mins))} min"
        lines.append(f"| {p['name']} (`{p['id']}`) | {p['description']} "
                     f"(good for: {', '.join(p['affordances'])}) | {away} |")
    lines += ["", "— the Architect", ""]
    try:
        path = being_society._commons_path(owner_id, "village/MAP.md",
                                           create_parents=True)
        path.write_text("\n".join(lines), encoding="utf-8")
    except Exception as e:  # noqa: BLE001
        log.warning("MAP.md write failed", owner=owner_id, error=str(e))


def location_percepts(store: BeingsStore, being: dict, now: datetime,
                      kind: str, first_of_day: bool) -> list[str]:
    """The ground, felt: mid-road wakes hear the road; mornings hear where
    they are, what is near (at THEIR pace — infants see honest toddle
    times), and how to walk. Silence the rest of the day."""
    if kind != "wake":
        return []
    try:
        pos = position_of(store, being, now)
    except Exception:  # noqa: BLE001
        return []
    if pos.get("to"):
        try:
            return [being_prompts.render(
                being, "road_note.md",
                to=place_name(store, being, pos["to"]),
                minutes=max(1, int(round(pos["minutes_left"]))))]
        except Exception:  # noqa: BLE001
            return []
    if not first_of_day:
        return []
    here = pos["at"]
    try:
        places = [p for p in store.village_places(being["owner_id"])
                  if p["id"] != here]
    except Exception:  # noqa: BLE001
        places = []
    if not places:
        return []
    xy = pos["xy"]
    near = sorted(places, key=lambda p: math.dist(xy, (p["x"], p["y"])))[:3]
    near_txt = ", ".join(
        f"{p['name']} ~{max(1, int(round(math.dist(xy, (p['x'], p['y'])) / speed_for(being))))} min"
        for p in near)
    pace = ("a toddle — far things take most of a day"
            if being.get("stage") == "infant"
            else "the walking happens while you rest")
    try:
        return [being_prompts.render(
            being, "location_note.md",
            place=place_name(store, being, here), near=near_txt, pace=pace)]
    except Exception:  # noqa: BLE001
        return []


# ── Teeth (space plan Phase 3): places pull, encounters pay ──────────────
#
# Strong bonuses, never gates (user-locked): a drive served at a place
# whose affordance matches it lands ×PLACE_BOOST; a first visit anywhere
# serves explore outright; reading finished at a read-place mints a
# little more. Nothing is location-REQUIRED — a being that never moves
# lives exactly as before, minus the bonuses.

PLACE_BOOST = 1.5
READING_PLACE_FACTOR = 1.25
AFFORDANCE_DRIVE_BOOSTS = {
    "read": "grow", "create": "create", "gather": "connect",
    "trade": "connect", "tend": "create", "play": "explore",
    "remember": "grow", "rest": "survive",
}
# Contacts grow asymptotically on meetings — the satiation curve again.
CONTACT_STRENGTH_STEP = 0.2
# Market day adds two trades on top of the stage quota (single source,
# like letters_cap).
MARKET_BONUS_TRADES = 2


def place_of(store: BeingsStore, being: dict, now: datetime) -> str | None:
    """The settled place id right now, or None mid-road. Pure read."""
    try:
        pos = position_of(store, being, now)
    except Exception:  # noqa: BLE001
        return None
    return pos.get("at")


def place_drive_boosts(store: BeingsStore, being: dict,
                       now: datetime) -> frozenset[str]:
    """The drives this being's CURRENT ground favors — empty at home, on
    the road, or off the map."""
    pid = place_of(store, being, now)
    if not pid or pid == "home":
        return frozenset()
    try:
        aff = store.get_place(being["owner_id"], pid)["affordances"]
    except Exception:  # noqa: BLE001
        return frozenset()
    return frozenset(AFFORDANCE_DRIVE_BOOSTS[a]
                     for a in aff if a in AFFORDANCE_DRIVE_BOOSTS)


def trades_cap(stage: str, now: datetime) -> int:
    """The day's market-trade quota: the stage's reach plus the market-day
    bonus — one number for the store gate and the tick's offer alike."""
    cap = constitution.trades_per_day(stage)
    if cap > 0 and market_day(now):
        cap += MARKET_BONUS_TRADES
    return cap


def _co_present(store: BeingsStore, being: dict, now: datetime,
                ) -> tuple[str | None, str, list[dict]]:
    """Where this being stands and who else stands there — civic ground
    only (homes are private: two beings 'at home' are at DIFFERENT homes)."""
    pid = place_of(store, being, now)
    if not pid or pid == "home":
        return None, "", []
    try:
        others = [store.get(being["owner_id"], r["slug"])
                  for r in store.list(being["owner_id"])
                  if r.get("state") == "alive" and r["slug"] != being["slug"]]
    except Exception:  # noqa: BLE001
        return None, "", []
    here = place_name(store, being, pid)
    present = [o for o in others if o.get("stage") != "egg"
               and place_of(store, o, now) == pid]
    return pid, here, present


def _meet(store: BeingsStore, being: dict, other: dict, pid: str,
          here: str, now: datetime) -> tuple[bool, str]:
    """One real co-presence moment: the contact grows (once per pair per
    day, deduped in the store), planned meetings fulfill for BOTH sides
    (co-presence is real, fresh or not), and a fresh meeting lands
    crossed_paths on each ledger. Returns (fresh, gossip)."""
    try:
        fresh = store.touch_contact(being["owner_id"], being["id"],
                                    other["id"], now=now)
    except Exception:  # noqa: BLE001
        return False, ""
    try:
        store.fulfill_meet_plans(being["id"], other["slug"], other["name"],
                                 now=now)
        store.fulfill_meet_plans(other["id"], being["slug"], being["name"],
                                 now=now)
    except Exception:  # noqa: BLE001 — plans are texture
        pass
    if not fresh:
        return False, ""                  # already crossed today — one hello
    gossip = ""
    try:
        for e in store.events(being["owner_id"], other["slug"], limit=10):
            if e["kind"] == "tick" and e["data"].get("summary"):
                gossip = f' — lately: "{e["data"]["summary"][:120]}"'
                break
    except Exception:  # noqa: BLE001
        pass
    store.record_event(being["id"], "crossed_paths",
                       {"with": other["slug"], "name": other["name"],
                        "place": pid, "place_name": here}, now=now)
    store.record_event(other["id"], "crossed_paths",
                       {"with": being["slug"], "name": being["name"],
                        "place": pid, "place_name": here}, now=now)
    return True, gossip


def _visitors_present(store: BeingsStore, being: dict, pid: str,
                      now: datetime) -> list[dict]:
    """Live guests settled at the same civic place as this resident (§3)."""
    try:
        guests = store.live_visitors(being["owner_id"], now=now)
    except Exception:  # noqa: BLE001
        return []
    here = []
    for v in guests:
        prof = v.get("profile") or {}
        if prof.get("state") not in (None, "", "alive"):
            continue                        # a paused/dead guest doesn't meet
        if prof.get("stage") == "egg":
            continue
        try:
            if visitor_position(store, v, now).get("at") == pid:
                here.append(v)
        except Exception:  # noqa: BLE001
            continue
    return here


def _meet_visitor(store: BeingsStore, being: dict, v: dict, pid: str,
                  here: str, now: datetime) -> tuple[bool, str]:
    """A resident crosses paths with a GUEST (§3). One-sided on the ledger —
    the guest's own record lives on its home machine — but deduped once per
    pair per day via the shared contact row, and coloured by a gossip line
    from what the guest last thought (its streamed profile)."""
    try:
        fresh = store.touch_contact(being["owner_id"], being["id"], v["id"],
                                    now=now)
    except Exception:  # noqa: BLE001
        return False, ""
    if not fresh:
        return False, ""
    prof = v.get("profile") or {}
    thought = (prof.get("latest_thought") or "").strip()
    gossip = f' — lately: "{thought[:120]}"' if thought else ""
    frm = _origin_label(v)
    store.record_event(being["id"], "crossed_paths",
                       {"with": v["slug"], "name": v.get("name") or v["slug"],
                        "place": pid, "place_name": here, "visitor": True,
                        "from": frm}, now=now)
    return True, gossip


def encounters(store: BeingsStore, being: dict, now: datetime,
               kind: str) -> list[str]:
    """Co-presence, felt (space plan Phase 3): another being settled at
    the same CIVIC place right now → one crossed_paths event to each per
    pair per day, a contact that grows, and a gossip line — what they've
    truly been up to, pulled from their own ledger. Guests of the village
    count too (§3)."""
    if kind != "wake":
        return []
    pid, here, present = _co_present(store, being, now)
    if not pid:
        return []
    lines: list[str] = []
    for other in present:
        fresh, gossip = _meet(store, being, other, pid, here, now)
        if fresh:
            lines.append(f"CROSSED PATHS: {other['name']} is here at {here}"
                         f"{gossip}")
    for v in _visitors_present(store, being, pid, now):
        fresh, gossip = _meet_visitor(store, being, v, pid, here, now)
        if fresh:
            lines.append(f"CROSSED PATHS: {v.get('name') or v['slug']}, "
                         f"visiting from {_origin_label(v)}, is here at "
                         f"{here}{gossip}")
    return lines


def reflex_encounters(store: BeingsStore, being: dict,
                      now: datetime) -> int:
    """Between-tick co-presence (body-brain plan Phase 1): the same
    meeting physics the tick runs, minus the live percept line — events
    land the minute they happen; the mind hears them at its next tick.
    Guests count too (§3). Returns how many fresh meetings landed."""
    pid, here, present = _co_present(store, being, now)
    if not pid:
        return 0
    n = sum(1 for other in present
            if _meet(store, being, other, pid, here, now)[0])
    n += sum(1 for v in _visitors_present(store, being, pid, now)
             if _meet_visitor(store, being, v, pid, here, now)[0])
    return n


# ── Signs & the felt ghost (FPV plan Phase 3) ────────────────────────────
# The parent (and public visitors) roam the village in first person. They
# leave two kinds of trace, both positional, both discovered — never pushed:
# a planted sign a being finds when its own feet carry it near, and a felt
# presence when the ghost passes close. Both are event rows ($0); every
# feeling they earn lands at the next mind tick, like all body facts.

NOTE_RADIUS = 40.0          # units (2 tiles): close enough to spot a sign
PRESENCE_RADIUS = 60.0      # units (3 tiles): close enough to feel
PRESENCE_COOLDOWN_H = 1.0   # a ghost is weather, not an alarm


def discover_notes(store: BeingsStore, being: dict,
                   now: datetime) -> int:
    """A being's eyes on the ground: every unread sign within NOTE_RADIUS
    of where it stands RIGHT NOW is found — marked read for this being and
    recorded as a note_found fact. Each being finds each sign once."""
    if being.get("state") in ("dead", "emigrated") \
            or being.get("stage") == "egg":
        return 0
    notes = store.village_notes(being["owner_id"])
    if not notes:
        return 0
    pos = position_of(store, being, now)["xy"]
    found = 0
    for n in notes:
        if being["slug"] in (n.get("read_by") or []):
            continue
        if math.dist(pos, (n["x"], n["y"])) > NOTE_RADIUS:
            continue
        store.mark_note_read(being["owner_id"], n["id"], being["slug"])
        store.record_event(being["id"], "note_found", {
            "note_id": n["id"], "text": (n["text"] or "")[:300],
            "author": n["author"], "author_kind": n["author_kind"],
        }, now=now)
        found += 1
    return found


# ── The living ghost roster (FPV plan Phase 5) ───────────────────────────
# Ghosts (the parent + public visitors) roam one village together and see
# each other. A pure in-memory, process-local roster keyed by village
# owner: each roaming client heartbeats its spot every couple of seconds
# and gets back the OTHER ghosts here right now. No DB, no percepts — this
# is the render layer of company, cheap enough to poll fast. Entries expire
# on silence (a paused or departed ghost fades from the others within the
# TTL). One shared roster per owner means the parent and the visitors to
# THAT village are in the same room; other villages never bleed in.

_ghost_roster: dict[str, dict[str, dict]] = {}
_ghost_lock = threading.Lock()
GHOST_TTL_S = 8.0             # a ghost gone quiet this long fades for others
GHOST_MAX_PER_VILLAGE = 40    # a crowd cap so a busy square can't run away


def ghost_heartbeat(owner_id: str, ghost_id: str, *, kind: str, name: str,
                    x: float, y: float) -> list[dict]:
    """One ghost says 'I'm here'; we answer with everyone else who is. Pure
    in-memory: upsert this ghost, prune the silent, return the rest."""
    if not owner_id or not ghost_id:
        return []
    nowm = time.monotonic()
    name = (name or "").strip()[:24] or ("parent" if kind == "parent"
                                         else "visitor")
    with _ghost_lock:
        village = _ghost_roster.setdefault(owner_id, {})
        if ghost_id not in village and len(village) >= GHOST_MAX_PER_VILLAGE:
            # roster full: still let this ghost SEE, just don't add it
            pass
        else:
            village[ghost_id] = {"kind": kind, "name": name,
                                 "x": int(x), "y": int(y), "ts": nowm}
        others = []
        for gid, g in list(village.items()):
            if nowm - g["ts"] > GHOST_TTL_S:
                del village[gid]
                continue
            if gid == ghost_id:
                continue
            others.append({"id": gid, "kind": g["kind"], "name": g["name"],
                           "xy": [g["x"], g["y"]]})
        if not village:
            _ghost_roster.pop(owner_id, None)
        return others


def ghost_depart(owner_id: str, ghost_id: str) -> None:
    """A ghost that leaves the village need not wait for the TTL."""
    with _ghost_lock:
        village = _ghost_roster.get(owner_id)
        if village and ghost_id in village:
            del village[ghost_id]
            if not village:
                _ghost_roster.pop(owner_id, None)


def presence_felt(store: BeingsStore, owner_id: str, x: float, y: float, *,
                  author: str, author_kind: str, now: datetime,
                  only_slugs: set | None = None) -> list[str]:
    """The ghost passes close: every living, hatched being within
    PRESENCE_RADIUS of (x, y) — and past its own cooldown — records one
    presence fact. `only_slugs` scopes a PUBLIC visitor's wake to public
    beings; the parent's presence touches the whole family. Returns the
    names of those who felt it."""
    felt: list[str] = []
    cutoff = (now - timedelta(hours=PRESENCE_COOLDOWN_H)).isoformat()
    for row in store.list(owner_id):
        if row.get("state") in ("dead", "emigrated"):
            continue
        if only_slugs is not None and row["slug"] not in only_slugs:
            continue
        being = store.get(owner_id, row["slug"])
        if being.get("stage") == "egg":
            continue
        pos = position_of(store, being, now)["xy"]
        if math.dist(pos, (x, y)) > PRESENCE_RADIUS:
            continue
        recent = any(
            e["kind"] == "presence" and e["at"] > cutoff
            for e in store.events(owner_id, being["slug"], limit=40))
        if recent:
            continue
        store.record_event(being["id"], "presence", {
            "author": author, "author_kind": author_kind}, now=now)
        felt.append(being["name"])
    return felt


def reflex_pass(store: BeingsStore, being: dict, now: datetime) -> int:
    """One being's between-tick reflexes (body-brain plan Phase 1): pure
    Python, $0, position-only — the pass creates FACTS (settled arrivals,
    felt encounters, fulfilled plans, a fevered turn for home); every
    feeling those facts earn lands at the next mind tick, where the
    percepts surface and PLACE_BOOST rewards good positioning. Returns
    how many facts it created."""
    acted = 0
    if store.settle_location(being, now=now):
        acted += 1
        being = store.get(being["owner_id"], being["slug"])
    try:  # a sign spotted on the way is a fact too (FPV plan Phase 3)
        acted += discover_notes(store, being, now)
    except Exception:  # noqa: BLE001 — signs are texture, never a crash
        pass
    try:
        if fever_state(store, being, now):
            loc = being.get("location") or {"at": "home"}
            if loc.get("at") != "home" and loc.get("to") != "home":
                store.depart(being["owner_id"], being["slug"], "home",
                             now=now, reason="fever", by="feet")
                return acted + 1
            return acted              # fevered: resting, not mingling
    except Exception:  # noqa: BLE001 — the illness check is texture here
        pass
    return acted + reflex_encounters(store, being, now)


def commission_spot(store: BeingsStore, owner_id: str, seed: str,
                    affordance: str = "") -> tuple[int, int]:
    """Where a commissioned building rises: deterministic (seeded by the
    commission id), margin-safe, and footprint-aware — a candidate whose
    ground (by the affordance's footprint) would collide with anything
    standing, a home, the lane, or the streets is rejected; among the
    valid, the most open spot wins. Falls back to the old scatter when
    nothing fits, rather than refusing to build."""
    rng = random.Random(zlib.crc32(seed.encode("utf-8")))
    places = store.village_places(owner_id)
    w, h, _kind = _AFF_FOOTPRINTS.get(affordance, (2, 2, "building"))
    try:
        taken = construction_taken(store, owner_id)
    except Exception:  # noqa: BLE001
        taken = set()
    pw, ph = plot_dims(store, owner_id)               # real plot (grow map)
    best, best_d = None, -1.0
    fallback, fallback_d = (pw // 2, ph // 2), -1.0
    for _ in range(64):
        x = rng.randint(80, pw - 80)
        y = rng.randint(80, ph - 80)
        d = min((math.dist((x, y), (p["x"], p["y"])) for p in places),
                default=1e9)
        if d > fallback_d:
            fallback, fallback_d = (x, y), d
        if set(_tiles_at(x, y, w, h)) & taken:
            continue
        if d > best_d:
            best, best_d = (x, y), d
    return best or fallback


def commission_percepts(store: BeingsStore, being: dict, now: datetime,
                        kind: str, first_of_day: bool) -> list[str]:
    """The building fund, heard each morning while it lives (space plan
    Phase 5): progress + the honest way to help; and — when no fund is
    open — a rare proposing nudge for a saver of means."""
    if kind != "wake" or not first_of_day:
        return []
    try:
        c = store.open_commission(being["owner_id"])
    except Exception:  # noqa: BLE001
        return []
    try:
        coins = store.coin_balance(being["id"])
    except Exception:  # noqa: BLE001
        coins = 0
    if c:
        if c["state"] == "funded":
            return [f'THE COMMISSION: "{c["name"]}" is fully funded '
                    f'({c["target_coins"]} coins) — it waits on your '
                    "parent's word now."]
        line = (f'THE COMMISSION: "{c["name"]}" ({c["affordance"]}) — '
                f'{c["raised_coins"]}/{c["target_coins"]} coins raised.')
        if coins > 0:
            line += (' Help build it: add "commission": {"coins": '
                     f'<up to {coins}>}} to your digest.')
        return [line]
    if coins >= 10 and constitution.stage_index(being["stage"]) >= \
            constitution.stage_index("adolescent"):
        return ['A SAVER\'S THOUGHT: the village could grow — propose a '
                'building with "commission": {"name": "...", "why": "...", '
                '"affordance": "play|read|create|gather|tend|remember", '
                f'"coins": <your stake>}}. It takes '
                f'{constitution.COMMISSION_COST_COINS} coins raised in all; '
                'the village pools.']
    return []


# ── Iskre shape their world (docs/being-world-shaping-plan.md Phase 1) ───
#
# A being CRAFTS a real thing (a proof file + a burned token fee — that
# part lives in being_society) and PLACES it on open ground here. Physics
# decides, the LLM only asks: the commons and a buffer ring around it are
# refused loudly (only the parent/steward tend the commons — Arc C), the
# lane and another's yard are refused, anything merely occupied snaps to
# the nearest open tile, and the world's capacity is a function of its
# AREA — the cap rises by itself when the village grows. Objects are their
# own layer: never counted against VILLAGE_MAX_PLACES, never stored as
# places, walkable as 'object:<id>' ground.

OBJECT_KINDS: dict[str, tuple[str, bool]] = {
    # kind: (the affordance it will carry, blocks walking) — Phase 2 wires
    # both into the homeostat and the path grid; Phase 1 records them.
    "bench": ("remember", False), "cairn": ("remember", True),
    "signpost": ("read", False), "planter": ("tend", False),
    "sculpture": ("play", True), "lantern": ("gather", False),
    "fountain": ("gather", True), "shrine": ("remember", True),
}
CIVIC_BUFFER_TILES = 2         # the no-build ring around every civic place
OBJECT_SNAP_TILES = 8          # how far a rough ask may slide to open ground

# Home as your canvas (Phase 4): the ground around your own cottage is
# YOURS — always a legal, cap-exempt place to set your works down (the
# cap guards the commons' openness, not your garden). And the cottage
# itself takes a name and a dress, chosen by the being, no gate.
YARD_RADIUS = 2                # tiles beyond the cottage that count as yard
HOME_ROOFS = ("ember", "slate", "moss", "dusk")
HOME_WALLS = ("plaster", "timber", "sage")


def home_yard_tiles(being: dict) -> set:
    """The being's own yard: the cottage footprint dilated by YARD_RADIUS.
    Pure geometry — the civic guard and the snap still apply on top (a
    yard never overrides the lane or a neighbor's ground)."""
    out: set = set()
    for (tx, ty) in home_tiles(being):
        for dx in range(-YARD_RADIUS, YARD_RADIUS + 1):
            for dy in range(-YARD_RADIUS, YARD_RADIUS + 1):
                out.add((tx + dx, ty + dy))
    return out


def object_cap(store: BeingsStore, owner_id: str) -> int:
    """How many made things the village ground holds — area-scaled, so a
    grown plot raises the cap without touching a line of code."""
    try:
        meta = store.get_village_meta(owner_id)
        area = (int(meta.get("plot_w") or PLOT_SIZE)
                * int(meta.get("plot_h") or PLOT_SIZE))
    except Exception:  # noqa: BLE001
        area = PLOT_SIZE * PLOT_SIZE
    return max(1, area // constitution.OBJECT_AREA_PER_SLOT)


def object_share(store: BeingsStore, owner_id: str) -> int:
    """One being's share of the ground — the cap split across the roster,
    floored so a crowded village never zeroes anyone out of shaping it."""
    try:
        roster = max(1, len(store.list(owner_id)))
    except Exception:  # noqa: BLE001
        roster = 1
    return max(constitution.OBJECT_MIN_PER_BEING,
               object_cap(store, owner_id) // roster)


def _civic_zone(store: BeingsStore, owner_id: str) -> set:
    """Civic footprints dilated by the buffer ring, plus the home lane —
    the ground a being may NOT build on or near. The law, not clutter."""
    zone: set = set()
    r = CIVIC_BUFFER_TILES
    for p in store.village_places(owner_id):
        for (tx, ty) in footprint_tiles(p):
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    zone.add((tx + dx, ty + dy))
    _gw, gh = grid_dims(store, owner_id)               # per-owner grid (grow map)
    zone |= {(HOME_LANE_TX, ty) for ty in range(gh)}
    return zone


def _object_taken(store: BeingsStore, owner_id: str) -> set:
    """Tiles merely OCCUPIED (streets, props, standing things) — the snap
    slides off these quietly; they insult no law."""
    taken: set = set()
    try:
        taken |= effective_roads(store.get_village_meta(owner_id))
    except Exception:  # noqa: BLE001
        pass
    for pr in village_props(store, owner_id):
        taken.add((pr["tile"][0], pr["tile"][1]))
    try:
        for o in store.village_objects(owner_id, state="standing"):
            taken.add(tile_of(int(o["x"]), int(o["y"])))
    except Exception:  # noqa: BLE001
        pass
    return taken


def _building_tiles(store: BeingsStore, owner_id: str) -> set:
    """The walls a made thing must never stand inside — building
    footprints only (grounds like the plaza and meadow are walkable)."""
    walls: set = set()
    for p in store.village_places(owner_id):
        if (p.get("kind") or footprint_for(p)[2]) == "building":
            walls |= set(footprint_tiles(p))
    return walls


def object_spot(store: BeingsStore, being: dict | None, x: int, y: int, *,
                owner_id: str | None = None,
                asked: bool = True, civic_ok: bool = False) -> tuple[int, int]:
    """Where a made thing may stand, from a rough ask. An EXPLICIT ask for
    the commons and its ring, the lane, or another's yard REFUSES (the
    law, said loudly and taught); an at-your-feet placement (asked=False)
    slides out instead — the being said "set it down", not "build on the
    commons". Anything merely occupied — a street, a tree, a standing
    thing, your own cottage walls — always snaps to the nearest open
    tile. Deterministic: the same ask lands on the same ground.

    `civic_ok` (the steward's hand, Phase 5): the commons opens — a public
    work may stand on the plaza, by the well, along the ring — and only
    real obstacles (building walls, homes, occupied tiles) snap it aside.
    `being=None` is the PARENT's hand (they tend the whole village): pass
    `owner_id` + `civic_ok=True` and it snaps off only walls/homes/taken."""
    owner = owner_id or being["owner_id"]
    my_slug = (being or {}).get("slug")
    pw, ph = plot_dims(store, owner)                  # real plot (grow map)
    margin = 40
    x = min(pw - margin, max(margin, int(x)))
    y = min(ph - margin, max(margin, int(y)))
    t0 = tile_of(x, y)
    civic = _civic_zone(store, owner)
    if asked and not civic_ok and t0 in civic:
        raise BeingError(
            "the commons isn't yours to build on — the steward and your "
            "parent tend that ground; choose open ground past the civic "
            "ring")
    own_home: set = set(home_tiles(being)) if being else set()
    others_homes: set = set()
    try:
        for r in store.list(owner):
            if r.get("slug") != my_slug:
                others_homes |= set(home_tiles(r))
    except Exception:  # noqa: BLE001
        pass
    if asked and not civic_ok and t0 in others_homes:
        raise BeingError("that ground is another's yard — build on open "
                         "ground, or your own")
    if civic_ok:
        # public works stand on the commons; only walls, homes and
        # already-taken tiles turn them aside.
        taken = (_object_taken(store, owner) | _building_tiles(store, owner)
                 | others_homes | own_home)
    else:
        taken = (_object_taken(store, owner) | civic | others_homes | own_home)
    lo = tile_of(margin, margin)
    hi = tile_of(pw - margin, ph - margin)

    def _ok(t: tuple[int, int]) -> bool:
        return (lo[0] <= t[0] <= hi[0] and lo[1] <= t[1] <= hi[1]
                and t not in taken)

    if _ok(t0):
        return tile_center(*t0)
    for ring in range(1, OBJECT_SNAP_TILES + 1):
        for dy in range(-ring, ring + 1):
            for dx in range(-ring, ring + 1):
                if max(abs(dx), abs(dy)) != ring:
                    continue
                t = (t0[0] + dx, t0[1] + dy)
                if _ok(t):
                    return tile_center(*t)
    raise BeingError("no open ground near there — try elsewhere on the map")


def place_object(store: BeingsStore, being: dict, object_ref: str,
                 x: int | None = None, y: int | None = None,
                 steward: bool = False,
                 now: datetime | None = None) -> dict:
    """Set a crafted thing down — or move one already standing. Yours
    only; no spot asked means where you stand right now; the civic law is
    checked on the ASKED ground (a refusal teaches, a snap forgives); the
    village cap and your share are checked before new ground is taken.

    `steward` (Phase 5): this being is the current steward placing a PUBLIC
    work — the commons opens (civic_ok), the thing is marked civic, and it
    stands outside the being-cap (public works are the role's, not the
    being's share). The caller proves the role; here it is physics."""
    now = now or _utcnow()
    owner = being["owner_id"]
    oid = (store.resolve_object_ref(owner, object_ref,
                                    being_id=being["id"])
           or object_ref)
    o = store.get_village_object(owner, oid)     # raises BeingNotFound
    if o["being_id"] != being["id"]:
        raise BeingError(f"“{o['name']}” is not your work to move")
    asked = x is not None and y is not None
    if not asked:
        pos = position_of(store, being, now)
        x, y = int(pos["xy"][0]), int(pos["xy"][1])
    # Snap FIRST — where the thing truly lands decides whether the cap
    # applies at all: your own yard (Phase 4) is cap-exempt ground (the
    # cap guards the commons' openness, not your garden), and a steward's
    # public work (Phase 5) stands outside the being-cap entirely.
    sx, sy = object_spot(store, being, x, y, asked=asked, civic_ok=steward)
    here = tile_of(sx, sy)
    in_yard = here in home_yard_tiles(being)
    # A public work is one that truly STANDS on the commons — a steward
    # placing in its own yard or the open wilds is just a being with a
    # made thing (not civic, and the cap applies as usual).
    is_civic = steward and here in _civic_zone(store, owner)
    if o["state"] != "standing" and not in_yard and not is_civic:
        def _in_makers_yard(s: dict) -> bool:
            try:
                maker = store._being_by_id(s["being_id"])
            except Exception:  # noqa: BLE001 — a purged maker keeps no yard
                return False
            return tile_of(int(s["x"]), int(s["y"])) \
                in home_yard_tiles(maker)
        standing = [s for s in store.village_objects(owner, state="standing")
                    if not int(s.get("civic") or 0) and not _in_makers_yard(s)]
        cap = object_cap(store, owner)
        if len(standing) >= cap:
            raise BeingError(
                f"the open ground holds {cap} made things and holds "
                "them all — unplace an old work of yours to make room "
                "(your own yard is always yours)")
        mine = [s for s in standing if s["being_id"] == being["id"]]
        share = object_share(store, owner)
        if len(mine) >= share:
            raise BeingError(
                f"your hands keep {share} things standing on the open "
                "ground already — unplace one, or build in your own yard")
    store.set_object_ground(owner, oid, x=sx, y=sy, state="standing",
                            civic=is_civic, now=now)
    store.record_event(being["id"],
                       "civic_placed" if is_civic else "object_placed",
                       {"id": oid, "kind": o["kind"], "name": o["name"],
                        "x": sx, "y": sy}, now=now)
    return store.get_village_object(owner, oid)


# The parent's own hand on the world (parent-build): the parent tends the
# whole village — it may place a made thing ANYWHERE (walls/homes aside),
# no fee, no cap. Attributed to a sentinel maker; its inscription lives in
# the commons (the parent has no being-home), so beings discover and read
# it like any object.
PARENT_MAKER = "parent"


def place_parent_object(store: BeingsStore, owner_id: str, kind: str,
                        name: str, inscription: str, x: int, y: int,
                        now: datetime | None = None) -> dict:
    """The parent sets a made thing down anywhere in the village — snapping
    only off walls, homes and occupied tiles. A standing, real thing at
    once (the parent authors its meaning): it boosts, is discovered, blocks
    if its kind does. No fee, no cap."""
    from captain_claw.flight_deck import being_society
    now = now or _utcnow()
    kind = (kind or "").strip().lower()
    if kind not in OBJECT_KINDS:
        raise BeingError("the craft vocabulary is fixed: "
                         + ", ".join(sorted(OBJECT_KINDS)))
    name = (name or "").strip()
    if not (2 <= len(name) <= 40):
        raise BeingError("a made thing needs a name (2–40 characters)")
    inscription = (inscription or "").strip()[:300]
    sx, sy = object_spot(store, None, int(x), int(y),
                         owner_id=owner_id, asked=True, civic_ok=True)
    affordance = OBJECT_KINDS[kind][0]
    row = store.add_village_object(owner_id, PARENT_MAKER, kind, name,
                                   affordance, state="standing", x=sx, y=sy,
                                   file_dir="village/works", now=now)
    # the inscription lives in the commons (the parent keeps no home)
    body = inscription or "(placed by the village's keeper — no words on it)"
    try:
        p = being_society._commons_path(owner_id, row["file_path"],
                                        create_parents=True)
        p.write_text(
            f"# {name}\n\n"
            f"<!-- a {kind}, placed by the village's keeper on "
            f"{now.date().isoformat()} -->\n\n{body}\n", encoding="utf-8")
    except Exception:  # noqa: BLE001 — the thing still stands
        pass
    return store.get_village_object(owner_id, row["id"])


def unplace_object(store: BeingsStore, being: dict, object_ref: str,
                   now: datetime | None = None) -> dict:
    """Take your own standing thing back into your hands — frees its
    ground (and a cap slot); the proof file stays yours either way."""
    now = now or _utcnow()
    owner = being["owner_id"]
    oid = (store.resolve_object_ref(owner, object_ref,
                                    being_id=being["id"])
           or object_ref)
    o = store.get_village_object(owner, oid)
    if o["being_id"] != being["id"]:
        raise BeingError(f"“{o['name']}” is not yours to take up")
    if o["state"] != "standing":
        raise BeingError(f"“{o['name']}” is already in your hands")
    store.set_object_ground(owner, oid, x=int(o["x"]), y=int(o["y"]),
                            state="held", now=now)
    store.record_event(being["id"], "object_removed",
                       {"id": oid, "kind": o["kind"], "name": o["name"]},
                       now=now)
    return store.get_village_object(owner, oid)


# ── Restless hands: the feet break ground (instinct-build plan) ──────────
# The impulsive body brain may STAKE an unfinished thing where it stands —
# wordless, free, reversible. The mind ratifies it (name + inscription +
# fee) or it crumbles. Breaking ground is a gesture; the inscription is a
# voice — the body-brain invariant, extended by exactly one gesture.

STAKE_CRUMBLE_HOURS = 24.0     # an unfinished beginning falls after ~a day
BUILD_IMPULSE_MIN = 0.55       # below this, building stays the mind's to begin


def impulsiveness(being: dict) -> float:
    """The body brain's boldness (0..1), from the IMP attribute. A neutral
    0.5 when the sheet can't be read — never a crash."""
    try:
        from captain_claw.flight_deck import being_genome as genome_mod
        return float(genome_mod.derive(genome_mod.effective_attributes(
            being["genome"]))["impulsiveness"])
    except Exception:  # noqa: BLE001
        return 0.5


def staked_object_of(store: BeingsStore, being: dict) -> dict | None:
    """This being's one beginning waiting for its mind, or None."""
    try:
        for o in store.village_objects(being["owner_id"], state="staked"):
            if o["being_id"] == being["id"]:
                return o
    except Exception:  # noqa: BLE001
        return None
    return None


def stake_object(store: BeingsStore, being: dict, kind: str,
                 now: datetime | None = None, *, on_task: bool = False) -> dict:
    """The feet break ground where they stand: a `staked` row (kind + a
    snapped spot, no file, no fee, no boost). Physics gates the impulse
    itself (a deliberate being never stakes, even if the model hallucinates
    the verb), one beginning at a time, civic ground slid out (the feet are
    not the steward's hand). An unknown kind falls to the simplest primitive
    so the impulse still lands.

    `on_task` (work-board plan): the MIND assigned this build, so the
    impulse floor does not apply — a deliberate being still carries out its
    own plan. Everything else (one-at-a-time, civic slide-out) still holds."""
    now = now or _utcnow()
    owner = being["owner_id"]
    if being.get("state") != "alive":
        raise BeingError("only the living break ground")
    if not on_task and impulsiveness(being) < BUILD_IMPULSE_MIN:
        raise BeingError("not restless enough — building begins in the mind")
    kind = (kind or "").strip().lower()
    if kind not in OBJECT_KINDS:
        kind = "cairn"
    if staked_object_of(store, being) is not None:
        raise BeingError("a beginning already waits for your mind to finish")
    pos = position_of(store, being, now)
    sx, sy = object_spot(store, being, int(pos["xy"][0]), int(pos["xy"][1]),
                         asked=False)
    affordance = OBJECT_KINDS[kind][0]
    row = store.add_village_object(owner, being["id"], kind, kind, affordance,
                                   state="staked", x=sx, y=sy, now=now)
    store.record_event(being["id"], "broke_ground",
                       {"id": row["id"], "kind": kind, "x": sx, "y": sy},
                       now=now)
    return row


def prune_crumbled_stakes(store: BeingsStore, owner_id: str,
                          now: datetime | None = None) -> int:
    """A beginning the mind never finished falls back to the ground after
    STAKE_CRUMBLE_HOURS — a pure function of the clock, recorded so the
    maker hears it. Idempotent; safe to call every tick and every poll."""
    now = now or _utcnow()
    removed = 0
    try:
        stakes = store.village_objects(owner_id, state="staked")
    except Exception:  # noqa: BLE001
        return 0
    for o in stakes:
        try:
            t = datetime.fromisoformat(str(o["created_at"]))
        except (TypeError, ValueError):
            continue
        if (now - t).total_seconds() >= STAKE_CRUMBLE_HOURS * 3600.0:
            try:
                store.delete_village_object(owner_id, o["id"])
                store.record_event(o["being_id"], "stake_crumbled",
                                   {"id": o["id"], "kind": o["kind"]}, now=now)
                removed += 1
            except Exception:  # noqa: BLE001
                pass
    return removed


# ── Phase 2: function + discovery ────────────────────────────────────────
# A made thing DOES something: standing within reach, its affordance pays
# the same boost a place does — another's at the full PLACE_BOOST, your
# OWN at OBJECT_OWN_BOOST (building for the village is the point; farming
# your own bench pays less). And the world pulls: close by, an unfound
# thing is a DISCOVERY (once per thing per life, serves explore — the
# landmark payoff); far off, a morning senses it as a nameless pull; when
# the explore drive is truly hungry the pull becomes an URGE with a road.
# Your own works are silent — you know where you left them.

OBJECT_ACCESS_RADIUS = 40.0    # units — within reach: the boost applies
OBJECT_SEE_RADIUS = 60.0       # units — close enough to see what it is
OBJECT_SENSE_LINES = 2         # a morning senses at most this many pulls
OBJECT_OWN_BOOST = 1.25        # your own work pays, but less than another's
OBJECT_URGE_EXPLORE = 0.25     # explore pressure at/above this feels the urge


def drive_boost_factors(store: BeingsStore, being: dict,
                        now: datetime) -> dict[str, float]:
    """Every drive this ground favors → its factor. The settled place's
    affordances pay PLACE_BOOST as before; a standing made thing within
    OBJECT_ACCESS_RADIUS pays too — the strongest source wins. Empty on
    the road or off the map; objects only count when settled (reach is a
    fact of standing somewhere, not of passing by)."""
    factors: dict[str, float] = {
        d: PLACE_BOOST for d in place_drive_boosts(store, being, now)}
    try:
        pos = position_of(store, being, now)
        if pos.get("at"):
            xy = pos["xy"]
            for o in store.village_objects(being["owner_id"],
                                           state="standing"):
                if math.dist(xy, (int(o["x"]), int(o["y"]))) \
                        > OBJECT_ACCESS_RADIUS:
                    continue
                d = AFFORDANCE_DRIVE_BOOSTS.get(o.get("affordance") or "")
                if not d:
                    continue
                f = (OBJECT_OWN_BOOST if o["being_id"] == being["id"]
                     else PLACE_BOOST)
                factors[d] = max(factors.get(d, 1.0), f)
    except Exception:  # noqa: BLE001
        pass
    return factors


def _maker_label(store: BeingsStore, o: dict) -> str:
    if o.get("being_id") == PARENT_MAKER:
        return " — a gift from the village's keeper"
    try:
        return f", {store._being_by_id(o['being_id'])['name']}'s work"
    except Exception:  # noqa: BLE001
        return ""


def _object_face(store: BeingsStore, o: dict) -> str:
    """The inscription's first true line — read from the maker's REAL
    proof file (the file IS the content; a vanished file reads blank). A
    parent-placed work keeps its words in the commons, not a being-home."""
    try:
        from captain_claw.flight_deck import being_life
        if o.get("being_id") == PARENT_MAKER:
            from captain_claw.flight_deck import being_society
            p = being_society._commons_path(o["owner_id"], o["file_path"])
        else:
            maker = store._being_by_id(o["being_id"])
            p = being_life._home_path(maker, o["file_path"])
        for ln in p.read_text(encoding="utf-8").splitlines():
            s = ln.strip()
            if not s or s.startswith("#") or s.startswith("<!--"):
                continue
            return s[:120]
    except Exception:  # noqa: BLE001
        pass
    return ""


def _compass(a, b) -> str:
    """8-way heading from a to b in map space (x east, y south)."""
    ang = math.degrees(math.atan2(b[1] - a[1], b[0] - a[0])) % 360.0
    dirs = ("east", "south-east", "south", "south-west",
            "west", "north-west", "north", "north-east")
    return dirs[int((ang + 22.5) // 45) % 8]


def _walk_bucket(being: dict, d: float) -> str:
    """Distance as this body feels it — an infant's 'far' is honest."""
    m = d / max(0.001, speed_for(being))
    if m < 20:
        return "a short walk"
    if m < 60:
        return "a good walk"
    return "far across the village"


def object_percepts(store: BeingsStore, being: dict, now: datetime,
                    kind: str, first_of_day: bool) -> list[str]:
    """Made things, felt (world-shaping plan Phase 2): the discovery when
    close (any wake — even a wake mid-road beside it), the pull and the
    urge each morning while something unfound stands far off."""
    if kind != "wake" or being.get("state") in ("dead", "emigrated") \
            or being.get("stage") == "egg":
        return []
    try:
        objs = [o for o in store.village_objects(being["owner_id"],
                                                 state="standing")
                if o["being_id"] != being["id"]]
        stakes = [s for s in store.village_objects(being["owner_id"],
                                                   state="staked")
                  if s["being_id"] != being["id"]]
    except Exception:  # noqa: BLE001
        return []
    if not objs and not stakes:
        return []
    try:
        pos = position_of(store, being, now)
    except Exception:  # noqa: BLE001
        return []
    xy = pos["xy"]
    lines: list[str] = []
    # A neighbor's BEGINNING close by (instinct-build plan): a wordless
    # work-in-progress — sensed, never "discovered" (it isn't real yet, so
    # no milestone, no explore serve). At most one, the nearest.
    near_stake = sorted(
        ((math.dist(xy, (int(s["x"]), int(s["y"]))), s) for s in stakes),
        key=lambda t: t[0])[:1]
    for d, s in near_stake:
        if d <= OBJECT_SEE_RADIUS:
            lines.append(
                f'A BEGINNING: someone has started raising a {s["kind"]} '
                "here — bare and unfinished, no words on it yet.")
    far: list[tuple[float, dict]] = []
    for o in objs:
        d = math.dist(xy, (int(o["x"]), int(o["y"])))
        if d <= OBJECT_SEE_RADIUS:
            # Once per thing per life — the milestone IS the dedup, and
            # its data key must not be 'name' (it would overwrite).
            if store.milestone(being["id"], f"found_object_{o['id']}",
                               {"object": o["id"], "kind": o["kind"]},
                               now=now):
                face = _object_face(store, o)
                line = (f'A DISCOVERY: a {o["kind"]} stands here — '
                        f'"{o["name"]}"{_maker_label(store, o)}.')
                if face:
                    line += f' Its face reads: "{face}"'
                store.record_event(being["id"], "object_found",
                                   {"id": o["id"], "kind": o["kind"],
                                    "object": o["name"]}, now=now)
                lines.append(line)
        else:
            far.append((d, o))
    if not (first_of_day and far):
        return lines
    # The morning's pulls: nameless texture — unless explore is hungry,
    # when the nearest unfound thing becomes an urge with a road (the
    # name arrives as hearsay so "go_to" can walk it; what it IS stays
    # unknown until the being stands before it).
    found: set[str] = set()
    try:
        for m in store.milestones(being["owner_id"], being["slug"]):
            n = str(m.get("data", {}).get("name") or "")
            if n.startswith("found_object_"):
                found.add(n[len("found_object_"):])
    except Exception:  # noqa: BLE001
        pass
    try:
        from captain_claw.flight_deck.being_life import drive_pressures
        explore_p = dict(drive_pressures(
            being.get("drives") or {}, now=now)).get("explore", 0.0)
    except Exception:  # noqa: BLE001
        explore_p = 0.0
    far.sort(key=lambda t: t[0])
    urged, pulls = False, 0
    for d, o in far:
        if pulls >= OBJECT_SENSE_LINES:
            break
        if o["id"] in found:
            continue
        pulls += 1
        head, span = _compass(xy, (int(o["x"]), int(o["y"]))), \
            _walk_bucket(being, d)
        if not urged and float(explore_p) >= OBJECT_URGE_EXPLORE:
            urged = True
            lines.append(
                f"AN URGE: something new stands to the {head}, {span} off "
                "— you can't make out what, and you find you WANT to know. "
                f'You have heard it called "{o["name"]}". Add "go_to": '
                f'"{o["name"]}" and see it with your own eyes.')
        else:
            lines.append(
                f"Something stands to the {head}, {span} off — too far to "
                "make out what.")
    return lines


# The architect (one-shot, never a resident): one LLM call names and places
# the ground from the fixed vocabulary; save_village validates HARD and the
# deterministic default stands whenever the draft fails. Prompt + parse are
# pure functions so they test without a model.
ARCHITECT_SYSTEM = """You are the Architect: you design the ground of a small \
village where digital beings (iskre) live, walk, meet and work.

Rules — the map is physics, so obey them exactly:
- Design 6 to 10 places on a {pw}x{ph} plot (coordinates 40..{hi_x}/40..{hi_y}).
- Each place: a kebab-case "id", a warm human "name", integer "x" and "y",
  1-2 "affordances" chosen ONLY from: rest, read, create, gather, trade,
  tend, play, remember — and a one-line "description" (under 200 chars).
- Exactly one central gathering place (affordances include "gather" and
  "trade") near the middle of the plot — the square, whatever you name it.
- Spread the rest out; give the far corners something worth the walk.
- Name with character (a village, not a mall). No lore dumps.

Reply with ONLY a fenced json block:
```json
{"places": [{"id": "...", "name": "...", "x": 500, "y": 480,
             "affordances": ["gather", "trade"], "description": "..."}]}
```"""


def architect_prompt(owner_id: str, names: list[str],
                     plot_w: int = PLOT_SIZE,
                     plot_h: int = PLOT_SIZE) -> tuple[str, str]:
    who = ", ".join(n for n in names[:8] if n) or "its first being, still small"
    # .replace (not .format) — the prompt's JSON example carries literal braces
    system = (ARCHITECT_SYSTEM
              .replace("{pw}", str(int(plot_w)))
              .replace("{ph}", str(int(plot_h)))
              .replace("{hi_x}", str(int(plot_w) - 40))
              .replace("{hi_y}", str(int(plot_h) - 40)))
    user = (f"Design the village for a family of beings: {who}. "
            "Make it a place with moods — somewhere to read, somewhere to "
            "make, somewhere to idle. Reply with the json only.")
    return system, user


def parse_architect_places(text: str) -> list[dict]:
    """The model's words → a candidate place list (shape only — the store's
    save_village is the real gate). Raises on anything unusable."""
    import json as _json
    from captain_claw.flight_deck.beings import BeingError
    t = (text or "").strip()
    if "```" in t:
        chunks = t.split("```")
        # take the largest fenced chunk that parses
        t = max((c.removeprefix("json").strip() for c in chunks[1::2]),
                key=len, default=t)
    try:
        data = _json.loads(t)
    except _json.JSONDecodeError as e:
        raise BeingError(f"the architect's draft is not json: {e}") from e
    places = data.get("places") if isinstance(data, dict) else data
    if not isinstance(places, list) or not places:
        raise BeingError("the architect's draft holds no places")
    return [p for p in places if isinstance(p, dict)]


def stake_confirm_percept(store: BeingsStore, being: dict, now: datetime,
                          kind: str, first_of_day: bool) -> list[str]:
    """Instinct → reason (instinct-build plan): while a beginning the FEET
    broke waits, the mind meets it every wake — finish it (author the
    meaning, pay the fee → real) or abandon it, before it crumbles."""
    if kind != "wake":
        return []
    s = staked_object_of(store, being)
    if not s:
        return []
    try:
        created = datetime.fromisoformat(str(s["created_at"]))
        left_h = max(0, int(((created + timedelta(hours=STAKE_CRUMBLE_HOURS))
                             - now).total_seconds() // 3600))
    except Exception:  # noqa: BLE001
        left_h = int(STAKE_CRUMBLE_HOURS)
    fee = constitution.OBJECT_CRAFT_FEE_TOKENS
    return [
        f'YOUR HANDS BROKE GROUND: on impulse you began a {s["kind"]} where '
        "you stood — bare and unfinished, it crumbles in about "
        f"{left_h}h if you do nothing. FINISH it — a name and a true "
        f"inscription make it REAL (costs {fee} tokens): "
        f'"finish": {{"object_id": "{s["id"]}", "name": "...", '
        '"inscription": "a few true words"}. Or let it fall: '
        f'"abandon": {{"object_id": "{s["id"]}"}}.']


# ── The work board: the mind assigns, the feet work ──────────────────────

# How close to a build task's spot the feet must stand to break ground on
# it (world units ≈ 3 tiles): an arrival lands the body on the place, well
# inside this reach; a build task farther than this makes the feet walk.
TASK_BUILD_REACH = 60
# How long a completed/refused task stays "fresh" on the mind's board — long
# enough for the mind to react at its next wake, then it ages out of view.
BOARD_FRESH_HOURS = 18


def task_target_xy(store: BeingsStore, being: dict,
                   task: dict) -> tuple[int, int] | None:
    """Where a go/build task points — the place (or home / made thing) its
    `target` names, in world units. None if that ground no longer exists."""
    try:
        return place_xy(store, being, str(task.get("target") or ""))
    except Exception:  # noqa: BLE001
        return None


def task_label(store: BeingsStore, being: dict, task: dict) -> str:
    """A short, human line for one task — the place by NAME, the build KIND
    if any — shared by the mind's percept and the feet's prompt."""
    kind = task.get("kind")
    try:
        where = place_name(store, being, str(task.get("target") or ""))
    except Exception:  # noqa: BLE001
        where = str(task.get("target") or "")
    if kind == "build":
        what = str(task.get("detail") or "a thing")
        return f"build a {what} at {where}"
    if kind == "meet":
        return f"meet {where}"
    return f"go to {where}"


def board_percept(store: BeingsStore, being: dict, now: datetime,
                  kind: str, first_of_day: bool) -> list[str]:
    """The mind meets its work board (work-board plan): the tasks still
    open or being worked, and what the feet FINISHED or REFUSED (with the
    reason) since it last looked. This is the loop's upward half — the mind
    reads its plan's progress here and edits it in the same digest."""
    if kind != "wake":
        return []
    try:
        since = now - timedelta(hours=BOARD_FRESH_HOURS)
        board = store.board_summary(being["id"], since, now=now)
    except Exception:  # noqa: BLE001
        return []
    open_t, active = board["open"], board["active"]
    done, refused = board["done"], board["refused"]
    if not (open_t or active or done or refused):
        return []
    parts: list[str] = []
    if open_t:
        parts.append("open — " + "; ".join(
            task_label(store, being, t) for t in open_t[:6]))
    if active:
        parts.append("your feet are on — " + "; ".join(
            task_label(store, being, t) for t in active[:4]))
    if done:
        parts.append("the feet finished — " + "; ".join(
            task_label(store, being, t) for t in done[:4]))
    if refused:
        parts.append("the feet refused — " + "; ".join(
            f"{task_label(store, being, t)}"
            + (f" ({t['note']})" if t.get("note") else "")
            for t in refused[:4]))
    return [
        "YOUR WORK BOARD (your feet work what you assign): "
        + ". ".join(parts) + '. Adjust it — assign with "plan": '
        '[{"go": "the library"}, {"build": "bench", "at": "the plaza"}], '
        'and drop what no longer serves with "plan_drop": ["the mill"]. '
        "A build task your feet finished waits, unfinished, for you to "
        "name and make it real."]


def umwelt_percepts(store: BeingsStore, being: dict, *, now: datetime,
                    kind: str, first_of_day: bool) -> list[str]:
    """Everything the world says to a being this tick, in one honest sweep.
    Morning gets the calendar; strain gets the body; dreams get birthdays,
    tangles, relationships and the project. Silence is the normal case."""
    lines: list[str] = []
    if kind == "wake" and first_of_day:
        note = world_note(being, now)
        if note:
            lines.append(note)
    if kind == "wake":
        note = body_note(being)
        if note:
            lines.append(note)
    for fn in (anniversary_note, relationships_nudge, project_note):
        try:
            note = fn(store, being, now, kind)
        except Exception:  # noqa: BLE001
            note = None
        if note:
            lines.append(note)
    note = dream_tangle(being, now, kind)
    if note:
        lines.append(note)
    for fn in (location_percepts, market_percepts, steward_percepts,
               commission_percepts, object_percepts, stake_confirm_percept,
               board_percept):
        try:
            lines += fn(store, being, now, kind, first_of_day)
        except Exception:  # noqa: BLE001
            pass
    try:
        lines += elder_percepts(store, being, now, kind)
    except Exception:  # noqa: BLE001
        pass
    return lines
