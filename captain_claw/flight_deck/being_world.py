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

import os
import random
from datetime import datetime, timedelta

from captain_claw.flight_deck import being_constitution as constitution
from captain_claw.flight_deck import being_prompts
from captain_claw.flight_deck.beings import BeingsStore
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

# Illness as consequence, never RNG (roadmap T2.13). Both ailments are
# computed from the last 24h of the REAL ledger at tick time — no new state,
# nothing to cure but the underlying events aging out.
ILLNESS_WINDOW_HOURS = 24.0
FEVER_TIMEOUTS = 3               # this many timed-out ticks in a day = fever
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


def fever_state(store: BeingsStore, being: dict,
                now: datetime) -> str | None:
    """FEVER (T2.13): a real breakdown in the last day — a burn-cap collapse
    or repeated timed-out ticks. Returns the honest cause, or None. While it
    holds, the tick cadence is floored (the body spaces itself out) and the
    being is told to rest. It passes when the events age out — nothing to
    roll, nothing to bless away."""
    if _recent_kind_count(store, being, now, ("collapsed_exhausted",)):
        return "your wallet collapsed from overspending"
    timeouts = _recent_kind_count(store, being, now, ("tick_timeout",))
    if timeouts >= FEVER_TIMEOUTS:
        return f"your body timed out {timeouts} times today"
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
    try:
        return [being_prompts.render(being, "steward_note.md")]
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
    """Market morning (T3.17): the stalls cried out — real publications with
    real prices, plus the reminder that letters run cheaper today."""
    if kind != "wake" or not first_of_day or not market_day(now):
        return []
    if not constitution.has_capability(being["stage"], "commons_read"):
        return []
    stalls = []
    try:
        for p in store.publications(being["owner_id"], limit=5):
            price = int(p.get("price_tokens") or 0)
            stalls.append(f'  - [{p["id"][:8]}] "{p["title"]}" — '
                          + (f"{price} tokens" if price else "free"))
    except Exception:  # noqa: BLE001
        stalls = []
    try:
        return [being_prompts.render(
            being, "market_note.md",
            stalls=("\n".join(stalls) if stalls
                    else "  (the shelves are bare — be the first to publish)"),
            bonus=MARKET_BONUS_LETTERS)]
    except Exception:  # noqa: BLE001
        return []


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
    for fn in (market_percepts, steward_percepts):
        try:
            lines += fn(store, being, now, kind, first_of_day)
        except Exception:  # noqa: BLE001
            pass
    try:
        lines += elder_percepts(store, being, now, kind)
    except Exception:  # noqa: BLE001
        pass
    return lines
