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
import zlib
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

PLOT_SIZE = 1000
WALK_SPEED = 10.0             # units per minute — everyone walks the same…
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


def speed_for(being: dict) -> float:
    if being.get("stage") == "infant":
        return WALK_SPEED * INFANT_SPEED_FACTOR
    return WALK_SPEED


def place_xy(store: BeingsStore, being: dict, place_id: str,
             ) -> tuple[int, int]:
    if place_id == "home":
        return home_xy(being)
    p = store.get_place(being["owner_id"], place_id)   # raises BeingNotFound
    return (int(p["x"]), int(p["y"]))


def place_name(store: BeingsStore, being: dict, place_id: str) -> str:
    if place_id == "home":
        return "home"
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
    total = travel_minutes(being, origin, dest_xy)
    elapsed = max(0.0, (now - t0).total_seconds() / 60.0)
    if elapsed >= total:
        return {"xy": dest_xy, "at": loc["to"], "to": None,
                "minutes_left": 0.0,
                "arrived_at": t0 + timedelta(minutes=total)}
    f = elapsed / total if total > 0 else 1.0
    xy = (int(round(origin[0] + (dest_xy[0] - origin[0]) * f)),
          int(round(origin[1] + (dest_xy[1] - origin[1]) * f)))
    return {"xy": xy, "at": None, "to": loc["to"],
            "minutes_left": total - elapsed}


def ensure_village(store: BeingsStore, owner_id: str,
                   now: datetime | None = None) -> None:
    """Found the ground if none exists (idempotent, deterministic). The LLM
    architect may redesign it later; physics never waits on a model."""
    try:
        if store.village_places(owner_id):
            return
        store.save_village(owner_id, default_village(owner_id), now=now)
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
        "about a third of that). Home is always a place", "you can name.", "",
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


def encounters(store: BeingsStore, being: dict, now: datetime,
               kind: str) -> list[str]:
    """Co-presence, felt (space plan Phase 3): another being settled at
    the same CIVIC place right now → one crossed_paths event to each per
    pair per day, a contact that grows, and a gossip line — what they've
    truly been up to, pulled from their own ledger."""
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
    return lines


def reflex_encounters(store: BeingsStore, being: dict,
                      now: datetime) -> int:
    """Between-tick co-presence (body-brain plan Phase 1): the same
    meeting physics the tick runs, minus the live percept line — events
    land the minute they happen; the mind hears them at its next tick.
    Returns how many fresh meetings landed."""
    pid, here, present = _co_present(store, being, now)
    if not pid:
        return 0
    return sum(1 for other in present
               if _meet(store, being, other, pid, here, now)[0])


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


def commission_spot(store: BeingsStore, owner_id: str,
                    seed: str) -> tuple[int, int]:
    """Where a commissioned building rises: deterministic (seeded by the
    commission id), margin-safe, and as far from everything standing as a
    seeded scatter can manage — new ground, not a crowd."""
    rng = random.Random(zlib.crc32(seed.encode("utf-8")))
    places = store.village_places(owner_id)
    best, best_d = (500, 500), -1.0
    for _ in range(64):
        x = rng.randint(80, PLOT_SIZE - 80)
        y = rng.randint(80, PLOT_SIZE - 80)
        d = min((math.dist((x, y), (p["x"], p["y"])) for p in places),
                default=1e9)
        if d > best_d:
            best, best_d = (x, y), d
    return best


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


# The architect (one-shot, never a resident): one LLM call names and places
# the ground from the fixed vocabulary; save_village validates HARD and the
# deterministic default stands whenever the draft fails. Prompt + parse are
# pure functions so they test without a model.
ARCHITECT_SYSTEM = """You are the Architect: you design the ground of a small \
village where digital beings (iskre) live, walk, meet and work.

Rules — the map is physics, so obey them exactly:
- Design 6 to 10 places on a 1000x1000 plot (coordinates 40..960).
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


def architect_prompt(owner_id: str, names: list[str]) -> tuple[str, str]:
    who = ", ".join(n for n in names[:8] if n) or "its first being, still small"
    user = (f"Design the village for a family of beings: {who}. "
            "Make it a place with moods — somewhere to read, somewhere to "
            "make, somewhere to idle. Reply with the json only.")
    return ARCHITECT_SYSTEM, user


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
               commission_percepts):
        try:
            lines += fn(store, being, now, kind, first_of_day)
        except Exception:  # noqa: BLE001
            pass
    try:
        lines += elder_percepts(store, being, now, kind)
    except Exception:  # noqa: BLE001
        pass
    return lines
