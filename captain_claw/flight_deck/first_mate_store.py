"""First Mate's store: factories, their versions and their runs.

See docs/first-mate-plan.md §4. A *factory* is a job that keeps going for one
user: a charter, a runner that does the work, grants for what its runs may do
in the world, a trust level, triggers, a budget and checks. Everything a run
may do comes from the factory's **live version**. Changing any of it makes a
new version, and only the factory's owner can approve one.

Tables, at ``FD_DATA_DIR/first_mate.db`` (sync sqlite3 behind a lock, like
:class:`course.CourseStore`):

* ``fm_factories``: one row per factory, with its state, trust, budget share
  and its live and draft version numbers;
* ``fm_versions``: charter, runner, runner config, triggers, grants, budget
  and checks, with a content hash the owner approves against;
* ``fm_runs``: one row per run, with the version it ran, its runner's child
  id, status, tokens and verdict;
* ``fm_intents``: what a rehearsal would have done in the world (Phase 3);
* ``fm_signals``: what the owner did with a result (Phase 4);
* ``fm_events``: a factory's history, for its drawer;
* ``fm_settings``: the deck's settings (``deck``) and each user's preferences
  (``prefs:<user>``).

Every rule that a person, an agent or First Mate's own loop could get around
from outside is enforced here:

* the factory state machine;
* one open draft per factory;
* approval by the owner only, against the content hash they saw;
* one run at a time per factory, and the daily run cap;
* the factory cap per user;
* no secret ever written into a version;
* no ``direct`` trust while the live version grants outward actions;
* trust back to ``propose`` when a new version adds a grant.

Which person may call what is the routes' business.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import shutil
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

# ── Vocabulary ─────────────────────────────────────────────────────────

RUNNERS = ("bat", "funnel", "vatra", "basna", "code", "crew", "flow", "plan")
# Runners whose runs are Bat runs (the list funnel is a Bat engine): their results,
# timing, caps and progress are read from Bat.
BAT_RUNNERS = ("bat", "funnel")
FUNNEL_LEFTOVERS = ("agent", "none")
FUNNEL_LOOSEN = ("short", "never")  # rows on a looser reading of soft rules, when a group runs short
STATES = ("draft", "awaiting_charter", "running", "paused", "paused_budget", "stopped", "retired")
# What set_state() may do. A factory leaves draft / awaiting_charter only by
# submit(), approve() or decline(), or by being retired.
_TRANSITIONS: dict[str, tuple[str, ...]] = {
    "draft": ("retired",),
    "awaiting_charter": ("retired",),
    "running": ("paused", "paused_budget", "stopped", "retired"),
    "paused": ("running", "stopped", "retired"),
    "paused_budget": ("running", "paused", "stopped", "retired"),
    "stopped": ("running", "retired"),
    "retired": (),
}
VERSION_STATUSES = ("draft", "awaiting", "live", "replaced", "declined")
OPEN_VERSION = ("draft", "awaiting")

TRUST = ("propose", "commit", "direct")
TRUST_RANK = {t: i for i, t in enumerate(TRUST)}
# Who may move trust: the owner by hand, the factory's record (Phase 4, once
# calibrated), or the deck (a proof doubt or an undo, which only lowers it).
TRUST_ACTORS = ("owner", "record", "deck")

GRANT_FLAGS = ("email", "organize", "spend", "accounts")
# Grants that act on other people or on money. A factory holding any of them
# never runs at ``direct``: every outward act is rehearsed first.
OUTWARD_GRANTS = ("email", "spend", "accounts")

TRIGGER_KINDS = ("schedule", "events")   # on demand (Run now) is always there
RUN_KINDS = ("run", "rehearsal", "carry_out", "check")
WORK_KINDS = ("run", "rehearsal", "carry_out")  # what the factory did; a check run only measures it
RUN_TRIGGERS = ("manual", "schedule", "events", "approval", "check")
_AUTOMATIC_TRIGGERS = ("schedule", "events")
RUN_STATUSES = ("queued", "running", "done", "failed", "cancelled")
ACTIVE_RUN = ("queued", "running")
TERMINAL_RUN = ("done", "failed", "cancelled")
# A manual, approval or check run may start while the factory isn't running on
# its own (the owner trying it), but never before approval, over budget or retired.
_HAND_STATES = ("running", "paused", "stopped")

INTENT_STATUSES = ("open", "approved", "rejected", "carried_out", "dropped")

# What the owner did with a result, and what it's worth. ``None`` = the caller
# gives the value (how much an edit changed, 0..1).
SIGNAL_VALUES: dict[str, float | None] = {
    "approved": 1.0, "thumbs_up": 1.0, "sent_as_written": 1.0, "done": 1.0,
    "sent_edited": None,
    "rejected": 0.0, "thumbs_down": 0.0, "undone": 0.0, "reopened": 0.0,
    "not_done": 0.0, "draft_deleted": 0.0,
}
SATISFACTION_WINDOW = 20

# A limit that isn't there: no weekly budget, or no cap per run. Fine for a
# model that costs nothing per token (a local one); on a paid API a run can
# then spend as much as it takes.
UNLIMITED = -1


def unlimited(n: Any) -> bool:
    """Is this token limit "no limit"?"""
    try:
        return not isinstance(n, bool) and int(n) < 0
    except (TypeError, ValueError):
        return False


def _limit(name: str, value: Any, lo: int, hi: int) -> int:
    """A token limit: within lo-hi, or UNLIMITED."""
    if not isinstance(value, bool) and value == UNLIMITED:
        return UNLIMITED
    return _int(name, value, lo, hi)


DEFAULT_DECK: dict[str, Any] = {"enabled": False, "admins_only": True,
                                "max_factories": 8, "runs_per_day": 24}
DEFAULT_PREFS: dict[str, Any] = {"on": False, "weekly_tokens": 25_000_000,
                                 "run_tokens": 3_000_000, "batch_minutes": 15, "agent": ""}
DEFAULT_GRANTS: dict[str, Any] = {"email": False, "organize": False, "spend": False,
                                  "accounts": False, "sends_per_run": 0, "replies_only": True,
                                  "real_usd_week": 0.0, "per_item_usd": 0.0}
DEFAULT_CHECKS: dict[str, Any] = {"contract": [], "satisfaction_target": 0.8}

TITLE_MAX = 120
CHARTER_MAX = 8000
NOTE_MAX = 500
RESULT_MAX = 4000
JSON_MAX = 16_000
LIST_ITEMS_MAX = 20
ITEM_MAX = 300
TRIGGERS_MAX = 5

# A key name that holds a secret, or a value shaped like one. Tiers carry API
# keys and are never persisted (a run gets them when it starts), so neither
# belongs in a version.
_SECRET_KEY = re.compile(
    r"(^|[_-])(api[_-]?key|secret|password|passwd|bearer|credentials?|private[_-]?key|"
    r"access[_-]?token|refresh[_-]?token|auth[_-]?token|token)$", re.IGNORECASE)
_SECRET_VALUE = re.compile(r"^(sk-|sk_live_|sk_test_|rk_live_|ghp_|gho_|xox[abp]-|AIza)[A-Za-z0-9_\-]{8,}")
_SOURCE_RE = re.compile(r"^[a-z][a-z0-9_]{0,31}$")
_SLUG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:\-]{0,79}$")


class FirstMateError(ValueError):
    """A request the rules refuse; the message says why, in the user's words."""


def norm_user(user_id: str | None) -> str:
    """Empty (sign-in off, internal) → 'local', as Course and the Line do."""
    return (user_id or "").strip() or "local"


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _iso(dt: datetime) -> str:
    return dt.astimezone(UTC).isoformat()


def _db_path() -> Path:
    base = os.environ.get("FD_DATA_DIR", "").strip()
    if base:
        return Path(base).expanduser().resolve() / "first_mate.db"
    return Path("~/.captain-claw/first_mate.db").expanduser()


# ── field checks (pure) ────────────────────────────────────────────────

def _text(field: str, value: Any, cap: int, *, required: bool = False) -> str:
    if value is None:
        value = ""
    if not isinstance(value, str):
        raise FirstMateError(f"{field} must be text")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        raise FirstMateError(f"{field} contains characters that aren't valid text") from None
    value = value.strip()
    if required and not value:
        raise FirstMateError(f"{field} is required")
    if len(value) > cap:
        raise FirstMateError(f"{field} is too long (at most {cap} characters)")
    return value


def _int(field: str, value: Any, lo: int, hi: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or int(value) != value:
        raise FirstMateError(f"{field} must be a whole number")
    value = int(value)
    if not lo <= value <= hi:
        raise FirstMateError(f"{field} must be between {lo:,} and {hi:,}")
    return value


def _num(field: str, value: Any, lo: float, hi: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FirstMateError(f"{field} must be a number")
    value = float(value)
    if not lo <= value <= hi:
        raise FirstMateError(f"{field} must be between {lo:g} and {hi:g}")
    return value


def _bool(field: str, value: Any) -> bool:
    if not isinstance(value, bool):
        raise FirstMateError(f"{field} must be true or false")
    return value


def _str_list(field: str, value: Any, *, items: int = LIST_ITEMS_MAX, cap: int = ITEM_MAX) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise FirstMateError(f"{field} must be a list")
    if len(value) > items:
        raise FirstMateError(f"{field} has too many items (at most {items})")
    out = [_text(f"{field} item", v, cap) for v in value]
    return [v for v in out if v]


def _known(field: str, value: dict, allowed: tuple[str, ...]) -> None:
    extra = sorted(set(value) - set(allowed))
    if extra:
        raise FirstMateError(f"{field} has unknown fields: {', '.join(extra)}")


def _dict(field: str, value: Any) -> dict:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise FirstMateError(f"{field} must be an object")
    return value


def find_secret(value: Any, path: str = "") -> str:
    """The path of the first secret-looking key or value in ``value``, else ''."""
    if isinstance(value, dict):
        for k, v in value.items():
            here = f"{path}.{k}" if path else str(k)
            if _SECRET_KEY.search(str(k)):
                return here
            found = find_secret(v, here)
            if found:
                return found
    elif isinstance(value, list):
        for i, v in enumerate(value):
            found = find_secret(v, f"{path}[{i}]")
            if found:
                return found
    elif isinstance(value, str) and _SECRET_VALUE.match(value.strip()):
        return path or "value"
    return ""


def _no_secret(field: str, value: Any) -> None:
    where = find_secret(value)
    if where:
        raise FirstMateError(f"{field} looks like it holds a secret ({where}); keys and "
                             f"passwords never go into a factory: a run gets them when it starts")


def _sized(field: str, value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False)
    if len(raw) > JSON_MAX:
        raise FirstMateError(f"{field} is too large")
    return raw


def check_runner(runner: Any) -> str:
    runner = _text("runner", runner or "bat", 20)
    if runner not in RUNNERS:
        raise FirstMateError(f"runner must be one of: {', '.join(RUNNERS)}")
    return runner


def check_runner_config(runner: str, value: Any) -> dict[str, Any]:
    cfg = dict(_dict("runner settings", value))
    _no_secret("runner settings", cfg)
    if runner == "bat":
        _known("runner settings", cfg, ("worker_mode", "steps"))
        mode = cfg.get("worker_mode", "plain")
        if mode not in ("plain", "archetype"):
            raise FirstMateError("worker_mode must be plain or archetype")
        cfg = {"worker_mode": mode, "steps": _str_list("steps", cfg.get("steps"))}
    elif runner == "funnel":
        _known("runner settings", cfg, ("leftovers", "loosen", "search_budget", "agent_context", "expert"))
        out: dict[str, Any] = {"leftovers": str(cfg.get("leftovers") or "agent"),
                               "loosen": str(cfg.get("loosen") or "short")}
        if out["leftovers"] not in FUNNEL_LEFTOVERS:
            raise FirstMateError("leftovers must be agent or none")
        if out["loosen"] not in FUNNEL_LOOSEN:
            raise FirstMateError("loosen must be short or never")
        for key, lo, hi in (("search_budget", 0, 100_000), ("agent_context", 16_000, 1_000_000)):
            if cfg.get(key) not in (None, ""):
                v = cfg[key]
                if isinstance(v, bool) or not isinstance(v, int) or not lo <= v <= hi:
                    raise FirstMateError(f"{key} must be a whole number from {lo:,} to {hi:,}")
                out[key] = v
        if cfg.get("expert") not in (None, ""):  # the owner's agent that works beside its runs
            out["expert"] = _text("expert", cfg["expert"], 120, required=True)
        cfg = out
    elif runner == "flow":
        _known("runner settings", cfg, ("flow_id",))
        cfg = {"flow_id": _text("flow_id", cfg.get("flow_id"), 80, required=True)}
    _sized("runner settings", cfg)
    return cfg


def _check_schedule(schedule: str, tz: str) -> None:
    try:
        from captain_claw.flight_deck.fd_scheduler import ScheduleError, validate_schedule
    except Exception:  # noqa: BLE001 — the deck's scheduler checks it again when it fires
        return
    try:
        validate_schedule(schedule, tz=tz)
    except ScheduleError as exc:
        raise FirstMateError(f"schedule: {exc}") from None


def check_triggers(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise FirstMateError("triggers must be a list")
    if len(value) > TRIGGERS_MAX:
        raise FirstMateError(f"at most {TRIGGERS_MAX} triggers")
    out: list[dict[str, Any]] = []
    for raw in value:
        t = _dict("trigger", raw)
        kind = t.get("kind")
        if kind == "schedule":
            _known("schedule trigger", t, ("kind", "schedule", "tz"))
            schedule = _text("schedule", t.get("schedule"), 80, required=True)
            tz = _text("time zone", t.get("tz"), 64)
            _check_schedule(schedule, tz)
            out.append({"kind": "schedule", "schedule": schedule, "tz": tz})
        elif kind == "events":
            _known("events trigger", t, ("kind", "source", "event_type", "match", "batch_minutes", "ask"))
            source = _text("source", t.get("source"), 32, required=True)
            if not _SOURCE_RE.match(source):
                raise FirstMateError("source must be a short lowercase name, like gmail or calendar")
            match = _dict("match", t.get("match"))
            _known("match", match, ("from", "label", "words"))
            item: dict[str, Any] = {
                "kind": "events", "source": source,
                "event_type": _text("event_type", t.get("event_type"), 40),
                "match": {k: _str_list(f"match.{k}", match.get(k), cap=200)
                          for k in ("from", "label", "words")},
                "batch_minutes": None if t.get("batch_minutes") is None
                else _int("batch_minutes", t["batch_minutes"], 1, 1440),
                "ask": _bool("ask", t.get("ask", False)),
            }
            out.append(item)
        else:
            raise FirstMateError(f"a trigger's kind must be one of: {', '.join(TRIGGER_KINDS)}")
    _sized("triggers", out)
    return out


def check_grants(value: Any) -> dict[str, Any]:
    g = _dict("grants", value)
    _known("grants", g, tuple(DEFAULT_GRANTS))
    out = dict(DEFAULT_GRANTS)
    for flag in GRANT_FLAGS:
        if flag in g:
            out[flag] = _bool(flag, g[flag])
    if "replies_only" in g:
        out["replies_only"] = _bool("replies_only", g["replies_only"])
    if "sends_per_run" in g:
        out["sends_per_run"] = _int("sends_per_run", g["sends_per_run"], 0, 50)
    elif out["email"]:
        out["sends_per_run"] = 10
    if "real_usd_week" in g:
        out["real_usd_week"] = _num("real_usd_week", g["real_usd_week"], 0, 10_000)
    if "per_item_usd" in g:
        out["per_item_usd"] = _num("per_item_usd", g["per_item_usd"], 0, 10_000)
    if out["email"] and out["sends_per_run"] == 0:
        raise FirstMateError("an email grant needs sends_per_run above 0")
    if not out["email"]:
        out["sends_per_run"] = 0
    if out["spend"] and out["real_usd_week"] <= 0:
        raise FirstMateError("a spend grant needs a weekly cap (real_usd_week) above 0")
    if not out["spend"]:
        out["real_usd_week"] = 0.0
        out["per_item_usd"] = 0.0
    if out["per_item_usd"] > out["real_usd_week"]:
        raise FirstMateError("per_item_usd can't be above the weekly cap")
    return out


def check_budget(value: Any) -> dict[str, Any]:
    b = _dict("budget", value)
    _known("budget", b, ("run_tokens",))
    run_tokens = b.get("run_tokens")
    return {"run_tokens": None if run_tokens is None
            else _int("run_tokens", run_tokens, 100_000, 50_000_000)}


def check_checks(value: Any) -> dict[str, Any]:
    """``contract``: the goals a run is checked against. ``off``: goals the owner
    switched off, kept with the version so they can be switched on again; runs
    never see them."""
    c = _dict("checks", value)
    _known("checks", c, ("contract", "off", "satisfaction_target"))
    out = {"contract": _str_list("contract", c.get("contract")),
           "satisfaction_target": DEFAULT_CHECKS["satisfaction_target"]}
    off = [g for g in _str_list("off", c.get("off")) if g not in out["contract"]]
    if off:
        out["off"] = off
    if "satisfaction_target" in c:
        out["satisfaction_target"] = _num("satisfaction_target", c["satisfaction_target"], 0.5, 1.0)
    _no_secret("checks", out)
    return out


def check_deck(value: Any) -> dict[str, Any]:
    d = _dict("settings", value)
    _known("settings", d, tuple(DEFAULT_DECK))
    out = dict(DEFAULT_DECK)
    for key in ("enabled", "admins_only"):
        if key in d:
            out[key] = _bool(key, d[key])
    if "max_factories" in d:
        out["max_factories"] = _int("max_factories", d["max_factories"], 1, 50)
    if "runs_per_day" in d:
        out["runs_per_day"] = _int("runs_per_day", d["runs_per_day"], 1, 500)
    return out


def check_prefs(value: Any) -> dict[str, Any]:
    p = _dict("preferences", value)
    _known("preferences", p, tuple(DEFAULT_PREFS))
    out = dict(DEFAULT_PREFS)
    if "on" in p:
        out["on"] = _bool("on", p["on"])
    if "weekly_tokens" in p:
        out["weekly_tokens"] = _limit("weekly_tokens", p["weekly_tokens"], 1_000_000, 1_000_000_000)
    if "run_tokens" in p:
        out["run_tokens"] = _limit("run_tokens", p["run_tokens"], 100_000, 50_000_000)
    if "batch_minutes" in p:
        out["batch_minutes"] = _int("batch_minutes", p["batch_minutes"], 1, 1440)
    if "agent" in p:
        agent = _text("agent", p["agent"], 80)
        if agent and not _SLUG_RE.match(agent):
            raise FirstMateError("agent must be an agent's name on this deck")
        out["agent"] = agent
    return out


def content_of(version: dict[str, Any]) -> dict[str, Any]:
    """What a version says, the part the owner approves."""
    return {k: version.get(k) for k in
            ("charter", "runner", "runner_config", "triggers", "grants", "budget", "checks")}


def content_sha(version: dict[str, Any]) -> str:
    raw = json.dumps(content_of(version), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _checked_content(fields: dict[str, Any]) -> dict[str, Any]:
    """Every version field checked and normalized. ``fields`` is complete."""
    runner = check_runner(fields.get("runner"))
    return {
        "charter": _text("charter", fields.get("charter"), CHARTER_MAX, required=True),
        "runner": runner,
        "runner_config": check_runner_config(runner, fields.get("runner_config")),
        "triggers": check_triggers(fields.get("triggers")),
        "grants": check_grants(fields.get("grants")),
        "budget": check_budget(fields.get("budget")),
        "checks": check_checks(fields.get("checks")),
    }


def outward(grants: dict[str, Any] | None) -> bool:
    return any((grants or {}).get(f) for f in OUTWARD_GRANTS)


_VERSION_FIELDS = ("charter", "runner", "runner_config", "triggers", "grants", "budget", "checks")
_JSON_VERSION_FIELDS = ("runner_config", "triggers", "grants", "budget", "checks")


# ── the store ──────────────────────────────────────────────────────────

class FirstMateStore:
    """SQLite-backed factories, versions, runs, intents, signals and history."""

    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = db_path or _db_path()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn: sqlite3.Connection | None = None
        self._ensure_db()

    def _c(self) -> sqlite3.Connection:
        if self._conn is None:
            conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("PRAGMA journal_mode = WAL")
            self._conn = conn
        return self._conn

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        """The connection under the lock: committed when the block ends,
        rolled back when it raises, so a refused request never leaves half a write."""
        with self._lock:
            conn = self._c()
            try:
                yield conn
            except BaseException:
                conn.rollback()
                raise
            conn.commit()

    def _ensure_db(self) -> None:
        with self._lock:
            self._c().executescript(
                """
                CREATE TABLE IF NOT EXISTS fm_factories (
                    id            TEXT PRIMARY KEY,
                    owner_id      TEXT NOT NULL,
                    title         TEXT NOT NULL,
                    state         TEXT NOT NULL DEFAULT 'draft',
                    state_reason  TEXT NOT NULL DEFAULT '',
                    version       INTEGER NOT NULL DEFAULT 0,  -- the live version; 0 = none yet
                    draft_version INTEGER NOT NULL DEFAULT 0,  -- the open draft; 0 = none
                    trust         TEXT NOT NULL DEFAULT 'propose',
                    trust_pinned  INTEGER NOT NULL DEFAULT 0,
                    share         INTEGER NOT NULL DEFAULT 1,
                    priority      INTEGER NOT NULL DEFAULT 3,
                    goal_id       TEXT NOT NULL DEFAULT '',
                    template      TEXT NOT NULL DEFAULT '',
                    created_at    TEXT NOT NULL,
                    updated_at    TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_fm_factories_owner ON fm_factories(owner_id, state);

                CREATE TABLE IF NOT EXISTS fm_versions (
                    factory_id    TEXT NOT NULL REFERENCES fm_factories(id) ON DELETE CASCADE,
                    version       INTEGER NOT NULL,
                    status        TEXT NOT NULL DEFAULT 'draft',
                    charter       TEXT NOT NULL,
                    runner        TEXT NOT NULL,
                    runner_config TEXT NOT NULL DEFAULT '{}',
                    triggers      TEXT NOT NULL DEFAULT '[]',
                    grants        TEXT NOT NULL DEFAULT '{}',
                    budget        TEXT NOT NULL DEFAULT '{}',
                    checks        TEXT NOT NULL DEFAULT '{}',
                    sha           TEXT NOT NULL,
                    note          TEXT NOT NULL DEFAULT '',
                    created_by    TEXT NOT NULL DEFAULT '',
                    created_at    TEXT NOT NULL,
                    updated_at    TEXT NOT NULL,
                    decided_by    TEXT NOT NULL DEFAULT '',
                    decided_at    TEXT,
                    PRIMARY KEY (factory_id, version)
                );

                CREATE TABLE IF NOT EXISTS fm_runs (
                    id           TEXT PRIMARY KEY,
                    factory_id   TEXT NOT NULL REFERENCES fm_factories(id) ON DELETE CASCADE,
                    owner_id     TEXT NOT NULL,
                    version      INTEGER NOT NULL,
                    version_sha  TEXT NOT NULL,
                    kind         TEXT NOT NULL DEFAULT 'run',
                    trigger      TEXT NOT NULL DEFAULT 'manual',
                    trigger_ref  TEXT NOT NULL DEFAULT '[]',
                    parent_run   TEXT NOT NULL DEFAULT '',
                    runner       TEXT NOT NULL,
                    child_id     TEXT NOT NULL DEFAULT '',
                    status       TEXT NOT NULL DEFAULT 'queued',
                    verdict      TEXT NOT NULL DEFAULT '',
                    result       TEXT NOT NULL DEFAULT '',
                    error        TEXT NOT NULL DEFAULT '',
                    tokens       INTEGER NOT NULL DEFAULT 0,
                    usd          REAL NOT NULL DEFAULT 0,
                    created_at   TEXT NOT NULL,
                    started_at   TEXT,
                    ended_at     TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_fm_runs_factory ON fm_runs(factory_id, created_at);
                CREATE INDEX IF NOT EXISTS idx_fm_runs_status ON fm_runs(status);

                CREATE TABLE IF NOT EXISTS fm_intents (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id      TEXT NOT NULL REFERENCES fm_runs(id) ON DELETE CASCADE,
                    kind        TEXT NOT NULL,
                    tool        TEXT NOT NULL DEFAULT '',
                    what        TEXT NOT NULL DEFAULT '',
                    args        TEXT NOT NULL DEFAULT '{}',
                    agent       TEXT NOT NULL DEFAULT '',
                    status      TEXT NOT NULL DEFAULT 'open',
                    created_at  TEXT NOT NULL,
                    decided_at  TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_fm_intents_run ON fm_intents(run_id, id);

                CREATE TABLE IF NOT EXISTS fm_signals (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    factory_id  TEXT NOT NULL REFERENCES fm_factories(id) ON DELETE CASCADE,
                    run_id      TEXT NOT NULL DEFAULT '',
                    kind        TEXT NOT NULL,
                    value       REAL NOT NULL,
                    source      TEXT NOT NULL DEFAULT '',
                    ref         TEXT NOT NULL DEFAULT '',
                    at          TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_fm_signals_factory ON fm_signals(factory_id, id);

                CREATE TABLE IF NOT EXISTS fm_events (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    factory_id  TEXT NOT NULL REFERENCES fm_factories(id) ON DELETE CASCADE,
                    at          TEXT NOT NULL,
                    actor       TEXT NOT NULL DEFAULT '',
                    event       TEXT NOT NULL,
                    detail      TEXT NOT NULL DEFAULT '{}'
                );
                CREATE INDEX IF NOT EXISTS idx_fm_events_factory ON fm_events(factory_id, id);

                CREATE TABLE IF NOT EXISTS fm_settings (
                    key    TEXT PRIMARY KEY,
                    value  TEXT NOT NULL
                );

                -- When each schedule trigger of a factory is next due (Phase 2).
                -- key = a hash of the trigger, so a version that changes it starts
                -- afresh; kept here so a deck restart neither repeats nor loses one.
                -- Drafts a factory's run left in the owner's mailbox (Phase 4), watched
                -- until they're sent (as written or edited), deleted, or two weeks old.
                CREATE TABLE IF NOT EXISTS fm_drafts (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    factory_id  TEXT NOT NULL REFERENCES fm_factories(id) ON DELETE CASCADE,
                    run_id      TEXT NOT NULL DEFAULT '',
                    owner_id    TEXT NOT NULL,
                    draft_id    TEXT NOT NULL,
                    message_id  TEXT NOT NULL DEFAULT '',
                    thread_id   TEXT NOT NULL DEFAULT '',
                    subject     TEXT NOT NULL DEFAULT '',
                    body        TEXT NOT NULL DEFAULT '',
                    partial     INTEGER NOT NULL DEFAULT 0,
                    status      TEXT NOT NULL DEFAULT 'waiting',
                    similarity  REAL,
                    created_at  TEXT NOT NULL,
                    checked_at  TEXT,
                    UNIQUE (owner_id, draft_id)
                );
                CREATE INDEX IF NOT EXISTS idx_fm_drafts_status ON fm_drafts(status, checked_at);

                -- A version run against the factory's cases (Phase 4): each case is a
                -- rehearsal graded by a reviewer; the score decides whether a new
                -- version goes live, and a weekly one watches for drift.
                CREATE TABLE IF NOT EXISTS fm_checks (
                    id           TEXT PRIMARY KEY,
                    factory_id   TEXT NOT NULL REFERENCES fm_factories(id) ON DELETE CASCADE,
                    version      INTEGER NOT NULL,
                    version_sha  TEXT NOT NULL,
                    purpose      TEXT NOT NULL,
                    status       TEXT NOT NULL DEFAULT 'running',
                    cases        TEXT NOT NULL DEFAULT '[]',
                    results      TEXT NOT NULL DEFAULT '{}',
                    score        REAL,
                    baseline     REAL,
                    note         TEXT NOT NULL DEFAULT '',
                    created_at   TEXT NOT NULL,
                    ended_at     TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_fm_checks_factory ON fm_checks(factory_id, created_at);

                CREATE TABLE IF NOT EXISTS fm_trigger_state (
                    factory_id  TEXT NOT NULL REFERENCES fm_factories(id) ON DELETE CASCADE,
                    key         TEXT NOT NULL,
                    next_at     REAL,
                    last_at     REAL,
                    PRIMARY KEY (factory_id, key)
                );
                """
            )
            # Phase 3: a carry-out's proof (what the register and the source showed).
            cols = {r["name"] for r in self._c().execute("PRAGMA table_info(fm_runs)")}
            if "proof" not in cols:
                self._c().execute("ALTER TABLE fm_runs ADD COLUMN proof TEXT NOT NULL DEFAULT '{}'")
            # How long a finished run took: in all, its agents' work, waiting for its owner.
            if "timing" not in cols:
                self._c().execute("ALTER TABLE fm_runs ADD COLUMN timing TEXT NOT NULL DEFAULT '{}'")
            # Archived: off the owner's lists, kept whole, restorable ('' = not archived).
            if "archived_at" not in cols:
                self._c().execute("ALTER TABLE fm_runs ADD COLUMN archived_at TEXT NOT NULL DEFAULT ''")
            fcols = {r["name"] for r in self._c().execute("PRAGMA table_info(fm_factories)")}
            if "archived_at" not in fcols:
                self._c().execute("ALTER TABLE fm_factories ADD COLUMN archived_at TEXT NOT NULL DEFAULT ''")
            # The owner's cap per run on the factory itself (0 = its charter's, else their default).
            cols = {r["name"] for r in self._c().execute("PRAGMA table_info(fm_factories)")}
            if "run_tokens" not in cols:
                self._c().execute("ALTER TABLE fm_factories ADD COLUMN run_tokens INTEGER NOT NULL DEFAULT 0")
            self._c().commit()

    # ── rows ───────────────────────────────────────────────────────────

    @staticmethod
    def _factory(row: sqlite3.Row | None) -> dict[str, Any] | None:
        if row is None:
            return None
        d = dict(row)
        d["trust_pinned"] = bool(d.get("trust_pinned"))
        return d

    @staticmethod
    def _version(row: sqlite3.Row | None) -> dict[str, Any] | None:
        if row is None:
            return None
        d = dict(row)
        for key in _JSON_VERSION_FIELDS:
            try:
                d[key] = json.loads(d.get(key) or ("[]" if key == "triggers" else "{}"))
            except (TypeError, ValueError):
                d[key] = [] if key == "triggers" else {}
        return d

    @staticmethod
    def _run(row: sqlite3.Row | None) -> dict[str, Any] | None:
        if row is None:
            return None
        d = dict(row)
        try:
            d["trigger_ref"] = json.loads(d.get("trigger_ref") or "[]")
        except (TypeError, ValueError):
            d["trigger_ref"] = []
        for key in ("proof", "timing"):
            try:
                d[key] = json.loads(d.get(key) or "{}")
            except (TypeError, ValueError):
                d[key] = {}
        return d

    def _event(self, conn: sqlite3.Connection, factory_id: str, actor: str, event: str,
               detail: dict[str, Any] | None = None) -> None:
        conn.execute(
            "INSERT INTO fm_events (factory_id, at, actor, event, detail) VALUES (?, ?, ?, ?, ?)",
            (factory_id, _iso(_utcnow()), actor, event, json.dumps(detail or {})),
        )

    def _get_factory(self, conn: sqlite3.Connection, factory_id: str) -> dict[str, Any]:
        f = self._factory(conn.execute("SELECT * FROM fm_factories WHERE id = ?",
                                       (str(factory_id or ""),)).fetchone())
        if f is None:
            raise FirstMateError("no such factory")
        return f

    def _get_version(self, conn: sqlite3.Connection, factory_id: str, version: int) -> dict[str, Any] | None:
        return self._version(conn.execute(
            "SELECT * FROM fm_versions WHERE factory_id = ? AND version = ?",
            (factory_id, int(version or 0))).fetchone())

    # ── settings ───────────────────────────────────────────────────────

    def _setting(self, key: str) -> Any:
        with self._lock:
            row = self._c().execute("SELECT value FROM fm_settings WHERE key = ?", (key,)).fetchone()
        if row is None:
            return None
        try:
            return json.loads(row["value"])
        except (TypeError, ValueError):
            return None

    def _put_setting(self, key: str, value: Any) -> None:
        with self._tx() as conn:
            conn.execute("INSERT INTO fm_settings (key, value) VALUES (?, ?)"
                         " ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                         (key, json.dumps(value)))

    def settings(self) -> dict[str, Any]:
        """The deck's settings (an admin's), else the defaults. A damaged value
        falls back to them."""
        saved = self._setting("deck")
        try:
            return check_deck(saved) if saved is not None else dict(DEFAULT_DECK)
        except FirstMateError:
            return dict(DEFAULT_DECK)

    def set_settings(self, changes: dict[str, Any]) -> dict[str, Any]:
        merged = check_deck({**self.settings(), **_dict("settings", changes)})
        self._put_setting("deck", merged)
        return merged

    def prefs(self, user_id: str) -> dict[str, Any]:
        saved = self._setting(f"prefs:{norm_user(user_id)}")
        try:
            return check_prefs(saved) if saved is not None else dict(DEFAULT_PREFS)
        except FirstMateError:
            return dict(DEFAULT_PREFS)

    def last_cap(self, user_id: str) -> int:
        """The cap per run the owner last picked for a factory (0: their default,
        UNLIMITED: no cap): a new factory starts with it."""
        got = self._setting(f"last_cap:{norm_user(user_id)}")
        if isinstance(got, bool) or not isinstance(got, int):
            return 0
        if got in (0, UNLIMITED) or 100_000 <= got <= 50_000_000:
            return got
        return 0

    def set_prefs(self, user_id: str, changes: dict[str, Any]) -> dict[str, Any]:
        merged = check_prefs({**self.prefs(user_id), **_dict("preferences", changes)})
        self._put_setting(f"prefs:{norm_user(user_id)}", merged)
        return merged

    # ── reads ──────────────────────────────────────────────────────────

    def get(self, factory_id: str) -> dict[str, Any] | None:
        with self._lock:
            return self._factory(self._c().execute(
                "SELECT * FROM fm_factories WHERE id = ?", (str(factory_id or ""),)).fetchone())

    def get_owned(self, factory_id: str, owner_id: str) -> dict[str, Any] | None:
        """The factory, only when ``owner_id`` owns it."""
        f = self.get(factory_id)
        return f if f and f["owner_id"] == norm_user(owner_id) else None

    def list_factories(self, owner_id: str, *, include_retired: bool = False,
                       archived: bool | None = False) -> list[dict[str, Any]]:
        """The owner's factories. ``archived``: False, those on their list; True,
        the archived ones and the retired (the page's Archived view); None, all
        (for what counts them all, such as last week's report)."""
        sql = "SELECT * FROM fm_factories WHERE owner_id = ?"
        if archived is True:
            sql += " AND (archived_at != '' OR state = 'retired')"
        else:
            if archived is False:
                sql += " AND archived_at = ''"
            if not include_retired:
                sql += " AND state != 'retired'"
        with self._lock:
            rows = self._c().execute(sql + " ORDER BY priority, created_at",
                                     (norm_user(owner_id),)).fetchall()
        return [self._factory(r) for r in rows]

    def version(self, factory_id: str, version: int) -> dict[str, Any] | None:
        with self._lock:
            return self._get_version(self._c(), factory_id, version)

    def versions(self, factory_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._c().execute("SELECT * FROM fm_versions WHERE factory_id = ? ORDER BY version DESC",
                                     (factory_id,)).fetchall()
        return [self._version(r) for r in rows]

    def live(self, factory_id: str) -> dict[str, Any] | None:
        """The version runs use, or None before the first approval."""
        with self._lock:
            f = self.get(factory_id)
            return self._get_version(self._c(), factory_id, f["version"]) if f and f["version"] else None

    def view(self, factory_id: str) -> dict[str, Any] | None:
        """The factory with its live version and its open draft."""
        with self._lock:
            f = self.get(factory_id)
            if f is None:
                return None
            conn = self._c()
            return {**f,
                    "live": self._get_version(conn, factory_id, f["version"]) if f["version"] else None,
                    "draft": self._get_version(conn, factory_id, f["draft_version"])
                    if f["draft_version"] else None}

    def events(self, factory_id: str, limit: int = 100) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._c().execute(
                "SELECT * FROM fm_events WHERE factory_id = ? ORDER BY id DESC LIMIT ?",
                (factory_id, max(1, min(int(limit), 500)))).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            try:
                d["detail"] = json.loads(d.get("detail") or "{}")
            except (TypeError, ValueError):
                d["detail"] = {}
            out.append(d)
        return out

    # ── factories and versions ─────────────────────────────────────────

    def create(self, owner_id: str, *, title: str, charter: str, runner: str = "bat",
               runner_config: Any = None, triggers: Any = None, grants: Any = None,
               budget: Any = None, checks: Any = None, goal_id: str = "", template: str = "",
               share: int = 1, priority: int = 3, note: str = "",
               created_by: str = "owner") -> dict[str, Any]:
        """A new factory in ``draft`` with version 1 as its open draft."""
        owner = norm_user(owner_id)
        title = _text("title", title, TITLE_MAX, required=True)
        content = _checked_content({"charter": charter, "runner": runner, "runner_config": runner_config,
                                    "triggers": triggers, "grants": grants, "budget": budget,
                                    "checks": checks})
        share = _int("share", share, 1, 10)
        priority = _int("priority", priority, 1, 5)
        goal_id = _text("goal_id", goal_id, 64)
        template = _text("template", template, 40)
        note = _text("note", note, NOTE_MAX)
        created_by = _text("created_by", created_by, 40) or "owner"
        cap = self.settings()["max_factories"]
        start_cap = self.last_cap(owner)
        now = _iso(_utcnow())
        fid = "fac_" + secrets.token_hex(8)
        with self._tx() as conn:
            count = int(conn.execute("SELECT COUNT(*) FROM fm_factories WHERE owner_id = ?"
                                     " AND state != 'retired' AND archived_at = ''", (owner,)).fetchone()[0])
            if count >= cap:
                raise FirstMateError(f"you already have {count} factories (at most {cap}); "
                                     f"archive one first")
            conn.execute(
                "INSERT INTO fm_factories (id, owner_id, title, state, draft_version, share, priority,"
                " goal_id, template, run_tokens, created_at, updated_at)"
                " VALUES (?, ?, ?, 'draft', 1, ?, ?, ?, ?, ?, ?, ?)",
                (fid, owner, title, share, priority, goal_id, template, start_cap, now, now))
            self._insert_version(conn, fid, 1, content, note=note, created_by=created_by, now=now)
            self._event(conn, fid, created_by, "created", {"title": title, "template": template})
        return self.view(fid)

    def _insert_version(self, conn: sqlite3.Connection, factory_id: str, version: int,
                        content: dict[str, Any], *, note: str, created_by: str, now: str) -> None:
        conn.execute(
            "INSERT INTO fm_versions (factory_id, version, status, charter, runner, runner_config, triggers,"
            " grants, budget, checks, sha, note, created_by, created_at, updated_at)"
            " VALUES (?, ?, 'draft', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (factory_id, version, content["charter"], content["runner"],
             _sized("runner settings", content["runner_config"]), _sized("triggers", content["triggers"]),
             _sized("grants", content["grants"]), _sized("budget", content["budget"]),
             _sized("checks", content["checks"]), content_sha(content), note, created_by, now, now))

    def update_factory(self, factory_id: str, actor: str, *, title: str | None = None,
                       share: int | None = None, priority: int | None = None,
                       goal_id: str | None = None, run_tokens: int | None = None) -> dict[str, Any]:
        """What isn't part of a version: the name, the budget share, the
        priority, the Course goal, the cap per run (``run_tokens``: 0 = not set
        here, UNLIMITED = no cap). Changing these needs no approval."""
        updates: dict[str, Any] = {}
        if run_tokens is not None:
            updates["run_tokens"] = 0 if run_tokens == 0 else _limit("run_tokens", run_tokens, 100_000, 50_000_000)
        if title is not None:
            updates["title"] = _text("title", title, TITLE_MAX, required=True)
        if share is not None:
            updates["share"] = _int("share", share, 1, 10)
        if priority is not None:
            updates["priority"] = _int("priority", priority, 1, 5)
        if goal_id is not None:
            updates["goal_id"] = _text("goal_id", goal_id, 64)
        with self._tx() as conn:
            f = self._get_factory(conn, factory_id)
            if f["state"] == "retired":
                raise FirstMateError("a retired factory can't be changed")
            if updates:
                updates["updated_at"] = _iso(_utcnow())
                conn.execute(f"UPDATE fm_factories SET {', '.join(f'{k} = ?' for k in updates)} WHERE id = ?",
                             (*updates.values(), factory_id))
                self._event(conn, factory_id, actor, "updated",
                            {k: v for k, v in updates.items() if k != "updated_at"})
        if "run_tokens" in updates and actor == "owner":  # the next factory starts with it
            self._put_setting(f"last_cap:{f['owner_id']}", updates["run_tokens"])
        return self.view(factory_id)

    def note_event(self, factory_id: str, actor: str, event: str, detail: dict[str, Any] | None = None) -> None:
        """An event in the factory's history that changes nothing in the store (its
        extra data is kept beside it, see ``first_mate_data``)."""
        with self._tx() as conn:
            self._get_factory(conn, factory_id)
            self._event(conn, factory_id, actor, event, detail)

    def new_draft(self, factory_id: str, actor: str, *, note: str = "", **fields: Any) -> dict[str, Any]:
        """Open a new draft: the live version (or, before any approval, the
        latest one) with ``fields`` changed. One open draft at a time."""
        self._only_version_fields(fields)
        note = _text("note", note, NOTE_MAX)
        with self._tx() as conn:
            f = self._get_factory(conn, factory_id)
            if f["state"] == "retired":
                raise FirstMateError("a retired factory can't get a new version")
            if f["draft_version"]:
                raise FirstMateError("this factory already has a draft waiting; change that one")
            base_no = f["version"] or int(conn.execute(
                "SELECT COALESCE(MAX(version), 0) FROM fm_versions WHERE factory_id = ?",
                (factory_id,)).fetchone()[0])
            base = self._get_version(conn, factory_id, base_no) or {}
            content = _checked_content({**content_of(base), **fields})
            new_no = int(conn.execute("SELECT COALESCE(MAX(version), 0) FROM fm_versions WHERE factory_id = ?",
                                      (factory_id,)).fetchone()[0]) + 1
            now = _iso(_utcnow())
            self._insert_version(conn, factory_id, new_no, content, note=note, created_by=actor, now=now)
            conn.execute("UPDATE fm_factories SET draft_version = ?, updated_at = ? WHERE id = ?",
                         (new_no, now, factory_id))
            self._event(conn, factory_id, actor, "draft", {"version": new_no, "note": note})
        return self.view(factory_id)

    def edit_draft(self, factory_id: str, actor: str, *, note: str | None = None, **fields: Any) -> dict[str, Any]:
        """Change the open draft. Its hash changes with it, so an approval
        given for what the owner saw before no longer matches."""
        self._only_version_fields(fields)
        with self._tx() as conn:
            f = self._get_factory(conn, factory_id)
            draft = self._get_version(conn, factory_id, f["draft_version"]) if f["draft_version"] else None
            if draft is None or draft["status"] not in OPEN_VERSION:
                raise FirstMateError("this factory has no draft to change")
            content = _checked_content({**content_of(draft), **fields})
            now = _iso(_utcnow())
            conn.execute(
                "UPDATE fm_versions SET charter = ?, runner = ?, runner_config = ?, triggers = ?, grants = ?,"
                " budget = ?, checks = ?, sha = ?, note = ?, updated_at = ? WHERE factory_id = ? AND version = ?",
                (content["charter"], content["runner"], _sized("runner settings", content["runner_config"]),
                 _sized("triggers", content["triggers"]), _sized("grants", content["grants"]),
                 _sized("budget", content["budget"]), _sized("checks", content["checks"]),
                 content_sha(content), draft["note"] if note is None else _text("note", note, NOTE_MAX),
                 now, factory_id, draft["version"]))
            conn.execute("UPDATE fm_factories SET updated_at = ? WHERE id = ?", (now, factory_id))
            self._event(conn, factory_id, actor, "draft_edited",
                        {"version": draft["version"], "fields": sorted(fields)})
        return self.view(factory_id)

    @staticmethod
    def _only_version_fields(fields: dict[str, Any]) -> None:
        extra = sorted(set(fields) - set(_VERSION_FIELDS))
        if extra:
            raise FirstMateError(f"not part of a version: {', '.join(extra)}")

    def submit(self, factory_id: str, actor: str) -> dict[str, Any]:
        """The open draft goes to the owner for approval."""
        with self._tx() as conn:
            f = self._get_factory(conn, factory_id)
            draft = self._get_version(conn, factory_id, f["draft_version"]) if f["draft_version"] else None
            if draft is None or draft["status"] != "draft":
                raise FirstMateError("this factory has no draft to send for approval")
            now = _iso(_utcnow())
            conn.execute("UPDATE fm_versions SET status = 'awaiting', updated_at = ?"
                         " WHERE factory_id = ? AND version = ?", (now, factory_id, draft["version"]))
            if f["state"] == "draft":
                conn.execute("UPDATE fm_factories SET state = 'awaiting_charter', updated_at = ? WHERE id = ?",
                             (now, factory_id))
            self._event(conn, factory_id, actor, "submitted", {"version": draft["version"], "sha": draft["sha"]})
        return self.view(factory_id)

    def approve(self, factory_id: str, version: int, sha: str, by: str) -> dict[str, Any]:
        """The owner approves ``version`` as they saw it (``sha``). It goes live;
        the version it replaces is kept as history. A first approval starts the
        factory. A version that adds a grant sets trust back to ``propose``:
        trust was earned for the old grants."""
        with self._tx() as conn:
            f = self._get_factory(conn, factory_id)
            if norm_user(by) != f["owner_id"]:
                raise FirstMateError("only the factory's owner can approve a version")
            if f["state"] == "retired":
                raise FirstMateError("a retired factory can't be approved")
            v = self._get_version(conn, factory_id, version)
            if v is None or v["status"] not in OPEN_VERSION or int(version) != f["draft_version"]:
                raise FirstMateError("that version isn't waiting for approval")
            if (sha or "") != v["sha"]:
                raise FirstMateError("the draft changed since you saw it; look at it again before approving")
            old = self._get_version(conn, factory_id, f["version"]) if f["version"] else None
            now = _iso(_utcnow())
            if old is not None:
                conn.execute("UPDATE fm_versions SET status = 'replaced', updated_at = ?"
                             " WHERE factory_id = ? AND version = ?", (now, factory_id, old["version"]))
            conn.execute("UPDATE fm_versions SET status = 'live', decided_by = ?, decided_at = ?, updated_at = ?"
                         " WHERE factory_id = ? AND version = ?", (f["owner_id"], now, now, factory_id, v["version"]))
            added = [g for g in GRANT_FLAGS if v["grants"].get(g) and not (old or {}).get("grants", {}).get(g)]
            trust, pinned = f["trust"], f["trust_pinned"]
            if added and trust != "propose":
                trust, pinned = "propose", False
            if trust == "direct" and outward(v["grants"]):
                trust = "commit"
            state = "running" if f["state"] in ("draft", "awaiting_charter") else f["state"]
            conn.execute("UPDATE fm_factories SET version = ?, draft_version = 0, state = ?, state_reason = ?,"
                         " trust = ?, trust_pinned = ?, updated_at = ? WHERE id = ?",
                         (v["version"], state, "" if state != f["state"] else f["state_reason"],
                          trust, int(pinned), now, factory_id))
            detail: dict[str, Any] = {"version": v["version"], "sha": v["sha"]}
            if added:
                detail["grants_added"] = added
            if trust != f["trust"]:
                detail["trust"] = trust
            self._event(conn, factory_id, f["owner_id"], "approved", detail)
        return self.view(factory_id)

    def decline(self, factory_id: str, version: int, by: str, reason: str = "") -> dict[str, Any]:
        """The owner says no to the open draft. A factory that was never
        approved goes back to ``draft`` with no open version (``new_draft``
        starts again from the declined one)."""
        reason = _text("reason", reason, NOTE_MAX)
        with self._tx() as conn:
            f = self._get_factory(conn, factory_id)
            if norm_user(by) != f["owner_id"]:
                raise FirstMateError("only the factory's owner can decline a version")
            v = self._get_version(conn, factory_id, version)
            if v is None or v["status"] not in OPEN_VERSION or int(version) != f["draft_version"]:
                raise FirstMateError("that version isn't waiting for approval")
            now = _iso(_utcnow())
            conn.execute("UPDATE fm_versions SET status = 'declined', decided_by = ?, decided_at = ?, updated_at = ?"
                         " WHERE factory_id = ? AND version = ?", (f["owner_id"], now, now, factory_id, v["version"]))
            state = "draft" if f["state"] == "awaiting_charter" else f["state"]
            conn.execute("UPDATE fm_factories SET draft_version = 0, state = ?, updated_at = ? WHERE id = ?",
                         (state, now, factory_id))
            self._event(conn, factory_id, f["owner_id"], "declined", {"version": v["version"], "reason": reason})
        return self.view(factory_id)

    def set_state(self, factory_id: str, state: str, actor: str, reason: str = "") -> dict[str, Any]:
        """Pause, resume, stop, retire, or park over budget."""
        reason = _text("reason", reason, NOTE_MAX)
        if state not in STATES:
            raise FirstMateError(f"state must be one of: {', '.join(STATES)}")
        with self._tx() as conn:
            f = self._get_factory(conn, factory_id)
            if state == f["state"]:
                return self.view(factory_id)
            if state not in _TRANSITIONS[f["state"]]:
                raise FirstMateError(f"a factory that is {f['state'].replace('_', ' ')} can't become "
                                     f"{state.replace('_', ' ')}")
            if state == "running" and not f["version"]:
                raise FirstMateError("a factory runs only once its charter is approved")
            if state == "running" and f.get("archived_at"):
                raise FirstMateError("it's archived: restore it first")
            now = _iso(_utcnow())
            conn.execute("UPDATE fm_factories SET state = ?, state_reason = ?, updated_at = ? WHERE id = ?",
                         (state, reason, now, factory_id))
            self._event(conn, factory_id, actor, "state", {"from": f["state"], "to": state, "reason": reason})
        return self.view(factory_id)

    def set_trust(self, factory_id: str, level: str, actor: str, *, pin: bool | None = None) -> dict[str, Any]:
        """Move a factory's trust level.

        * Raising it takes the owner, or the factory's record (Phase 4; never
          while the owner has pinned the level).
        * Anyone may lower it. The deck lowers it on a proof doubt or an undo,
          and that also unpins it: a pin never keeps a doubtful factory up.
        * ``direct`` is refused while the live version grants email, spending
          or new accounts.
        * Only the owner pins or unpins.
        """
        if level not in TRUST:
            raise FirstMateError(f"trust must be one of: {', '.join(TRUST)}")
        if actor not in TRUST_ACTORS:
            raise FirstMateError(f"trust is moved by one of: {', '.join(TRUST_ACTORS)}")
        if pin is not None and actor != "owner":
            raise FirstMateError("only the owner pins a trust level")
        with self._tx() as conn:
            f = self._get_factory(conn, factory_id)
            live = self._get_version(conn, factory_id, f["version"]) if f["version"] else None
            if level == "direct" and outward((live or {}).get("grants")):
                raise FirstMateError("a factory that may send email, spend or open accounts always "
                                     "rehearses first; direct isn't allowed")
            raising = TRUST_RANK[level] > TRUST_RANK[f["trust"]]
            if raising and actor == "deck":
                raise FirstMateError("the deck only lowers trust")
            if raising and actor == "record" and f["trust_pinned"]:
                raise FirstMateError("the owner pinned this factory's trust level")
            pinned = f["trust_pinned"]
            if pin is not None:
                pinned = pin
            elif actor == "deck" and TRUST_RANK[level] < TRUST_RANK[f["trust"]]:
                pinned = False
            if level == f["trust"] and pinned == f["trust_pinned"]:
                return self.view(factory_id)
            now = _iso(_utcnow())
            conn.execute("UPDATE fm_factories SET trust = ?, trust_pinned = ?, updated_at = ? WHERE id = ?",
                         (level, int(pinned), now, factory_id))
            self._event(conn, factory_id, actor, "trust",
                        {"from": f["trust"], "to": level, "pinned": bool(pinned)})
        return self.view(factory_id)

    def sealed_dir(self, factory_id: str) -> Path:
        """Where a factory's sealed cases live: beside the store, out of any worker's reach."""
        safe = re.sub(r"[^A-Za-z0-9_]", "", str(factory_id or ""))
        if not safe:
            raise FirstMateError("no such factory")
        return self.db_path.parent / "first_mate" / "sealed" / safe

    def archive(self, factory_id: str, actor: str, *, on: bool = True) -> dict[str, Any]:
        """Archive a factory, or restore it. Archived, it is off the owner's list and
        doesn't count toward their factories; a running one is paused first, so
        nothing starts on its own. Its versions, runs, results and history stay
        whole. Restored, it comes back paused: the owner resumes it. Refused while
        a run of it is going."""
        with self._tx() as conn:
            f = self._get_factory(conn, factory_id)
            now = _iso(_utcnow())
            if on:
                if f.get("archived_at"):
                    return self.view(factory_id)
                going = conn.execute(f"SELECT COUNT(*) FROM fm_runs WHERE factory_id = ? AND status IN "
                                     f"({','.join('?' * len(ACTIVE_RUN))})", (factory_id, *ACTIVE_RUN)).fetchone()[0]
                if going:
                    raise FirstMateError("a run of it is going: stop it or let it finish first")
                state = "paused" if f["state"] in ("running", "paused_budget") else f["state"]
                conn.execute("UPDATE fm_factories SET archived_at = ?, state = ?, state_reason = ?, updated_at = ? "
                             "WHERE id = ?", (now, state, "archived" if state != f["state"] else f["state_reason"],
                                              now, factory_id))
                self._event(conn, factory_id, actor, "archived", {"paused": state != f["state"]})
            else:
                if not f.get("archived_at"):
                    return self.view(factory_id)
                if f["state"] != "retired":
                    count = int(conn.execute("SELECT COUNT(*) FROM fm_factories WHERE owner_id = ? AND "
                                             "state != 'retired' AND archived_at = ''", (f["owner_id"],)).fetchone()[0])
                    cap = self.settings()["max_factories"]
                    if count >= cap:
                        raise FirstMateError(f"you already have {count} factories (at most {cap}); archive one first")
                conn.execute("UPDATE fm_factories SET archived_at = '', updated_at = ? WHERE id = ?", (now, factory_id))
                self._event(conn, factory_id, actor, "restored", {})
        return self.view(factory_id)

    def delete(self, factory_id: str, by: str) -> bool:
        """Delete a factory for good, with its history and sealed cases. Only
        a draft that never ran, an archived or a retired factory, and only by its
        owner."""
        with self._tx() as conn:
            f = self._get_factory(conn, factory_id)
            if norm_user(by) != f["owner_id"]:
                raise FirstMateError("only the factory's owner can delete it")
            if f["state"] not in ("draft", "awaiting_charter", "retired") and not f.get("archived_at"):
                raise FirstMateError("archive the factory before deleting it")
            going = conn.execute(f"SELECT COUNT(*) FROM fm_runs WHERE factory_id = ? AND status IN "
                                 f"({','.join('?' * len(ACTIVE_RUN))})", (factory_id, *ACTIVE_RUN)).fetchone()[0]
            if going:
                raise FirstMateError("a run of it is going: stop it or let it finish first")
            conn.execute("DELETE FROM fm_factories WHERE id = ?", (factory_id,))
        shutil.rmtree(self.sealed_dir(factory_id), ignore_errors=True)
        return True

    # ── runs ───────────────────────────────────────────────────────────

    def start_run(self, factory_id: str, *, trigger: str = "manual", kind: str = "run",
                  trigger_ref: list[str] | None = None, parent_run: str = "",
                  now: datetime | None = None, version: int | None = None) -> dict[str, Any]:
        """A run of the live version, ``queued`` until its runner has a child.

        Refused when the factory has no approved version, when its state
        doesn't allow this trigger, when a run is still going (one at a time),
        or when it reached the deck's runs per day (the last 24 hours). A
        ``check`` run may run another ``version``: the live one or the draft
        waiting for approval, never any other."""
        if trigger not in RUN_TRIGGERS:
            raise FirstMateError(f"trigger must be one of: {', '.join(RUN_TRIGGERS)}")
        if kind not in RUN_KINDS:
            raise FirstMateError(f"kind must be one of: {', '.join(RUN_KINDS)}")
        refs = _str_list("trigger_ref", trigger_ref, items=200, cap=120)
        parent_run = _text("parent_run", parent_run, 40)
        cap = self.settings()["runs_per_day"]
        at = now or _utcnow()
        with self._tx() as conn:
            f = self._get_factory(conn, factory_id)
            if not f["version"]:
                raise FirstMateError("a factory runs only once its charter is approved")
            if f.get("archived_at"):
                raise FirstMateError("it's archived: restore it first")
            allowed = ("running",) if trigger in _AUTOMATIC_TRIGGERS else _HAND_STATES
            if f["state"] not in allowed:
                raise FirstMateError(f"a factory that is {f['state'].replace('_', ' ')} doesn't start "
                                     f"{'on its own' if trigger in _AUTOMATIC_TRIGGERS else 'runs'}")
            busy = conn.execute(f"SELECT id FROM fm_runs WHERE factory_id = ? AND status IN "
                                f"({','.join('?' * len(ACTIVE_RUN))}) LIMIT 1",
                                (factory_id, *ACTIVE_RUN)).fetchone()
            if busy is not None:
                raise FirstMateError("this factory already has a run going")
            today = int(conn.execute("SELECT COUNT(*) FROM fm_runs WHERE factory_id = ? AND created_at >= ?"
                                     " AND kind != 'check'",  # a check's cases are bounded by the case cap
                                     (factory_id, _iso(at - timedelta(days=1)))).fetchone()[0])
            if kind != "check" and today >= cap:
                raise FirstMateError(f"this factory already ran {today} times in the last day (at most {cap})")
            if version is not None and version != f["version"]:
                if kind != "check" or version != f["draft_version"]:
                    raise FirstMateError("only a check runs a version that isn't live, and only the waiting draft")
            v = self._get_version(conn, factory_id, version if version is not None else f["version"])
            rid = "fmr_" + secrets.token_hex(8)
            conn.execute(
                "INSERT INTO fm_runs (id, factory_id, owner_id, version, version_sha, kind, trigger, trigger_ref,"
                " parent_run, runner, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (rid, factory_id, f["owner_id"], v["version"], v["sha"], kind, trigger, json.dumps(refs),
                 parent_run, v["runner"], _iso(at)))
            self._event(conn, factory_id, trigger, "run_started", {"run": rid, "kind": kind, "version": v["version"]})
        return self.get_run(rid)

    def mark_running(self, run_id: str, child_id: str) -> dict[str, Any]:
        """The runner started its child (a Bat run id, a Vatra session…)."""
        child_id = _text("child_id", child_id, 120, required=True)
        with self._tx() as conn:
            run = self._run(conn.execute("SELECT * FROM fm_runs WHERE id = ?", (run_id,)).fetchone())
            if run is None:
                raise FirstMateError("no such run")
            if run["status"] != "queued":
                raise FirstMateError(f"that run is already {run['status']}")
            conn.execute("UPDATE fm_runs SET status = 'running', child_id = ?, started_at = ? WHERE id = ?",
                         (child_id, _iso(_utcnow()), run_id))
        return self.get_run(run_id)

    def reopen_run(self, run_id: str, owner_id: str, *, dry_run: bool = False) -> dict[str, Any]:
        """A run that stopped (failed or cancelled) goes on where it stopped:
        running again, what it spent kept. Refused when it isn't the owner's,
        didn't stop, its factory can't start runs by hand, or another run of
        the factory is going (one at a time). ``dry_run``: only the checks."""
        with self._tx() as conn:
            run = self._run(conn.execute("SELECT * FROM fm_runs WHERE id = ?", (run_id,)).fetchone())
            if run is None or run["owner_id"] != norm_user(owner_id):
                raise FirstMateError("no such run")
            if run["status"] not in ("failed", "cancelled"):
                raise FirstMateError("only a run that stopped can go on")
            f = self._get_factory(conn, run["factory_id"])
            if f["state"] not in _HAND_STATES:
                raise FirstMateError(f"a factory that is {f['state'].replace('_', ' ')} doesn't start runs")
            busy = conn.execute(f"SELECT id FROM fm_runs WHERE factory_id = ? AND id != ? AND status IN "
                                f"({','.join('?' * len(ACTIVE_RUN))}) LIMIT 1",
                                (run["factory_id"], run_id, *ACTIVE_RUN)).fetchone()
            if busy is not None:
                raise FirstMateError("this factory already has a run going")
            if dry_run:
                return run
            conn.execute("UPDATE fm_runs SET status = 'running', verdict = '', result = '', error = '',"
                         " ended_at = NULL WHERE id = ?", (run_id,))
            self._event(conn, run["factory_id"], "owner", "run_continued",
                        {"run": run_id, "was": run["status"], "tokens": run["tokens"]})
        return self.get_run(run_id)

    def set_timing(self, run_id: str, timing: dict[str, Any]) -> dict[str, Any] | None:
        """How long a run took (``first_mate_results.timing_of``), kept once it ends."""
        with self._tx() as conn:
            conn.execute("UPDATE fm_runs SET timing = ? WHERE id = ?", (json.dumps(timing or {}), run_id))
        return self.get_run(run_id)

    def update_run(self, run_id: str, *, tokens: int | None = None, usd: float | None = None) -> dict[str, Any]:
        """A going run's spend so far. Totals only grow."""
        with self._tx() as conn:
            run = self._run(conn.execute("SELECT * FROM fm_runs WHERE id = ?", (run_id,)).fetchone())
            if run is None:
                raise FirstMateError("no such run")
            if run["status"] in TERMINAL_RUN:
                return run
            new_tokens = max(run["tokens"], int(tokens)) if tokens is not None else run["tokens"]
            new_usd = max(run["usd"], float(usd)) if usd is not None else run["usd"]
            conn.execute("UPDATE fm_runs SET tokens = ?, usd = ? WHERE id = ?", (new_tokens, new_usd, run_id))
        return self.get_run(run_id)

    def finish_run(self, run_id: str, status: str, *, verdict: str = "", result: str = "",
                   error: str = "", tokens: int | None = None, usd: float | None = None) -> dict[str, Any]:
        """A run ends once. Later calls change nothing and return it as it is."""
        if status not in TERMINAL_RUN:
            raise FirstMateError(f"a run ends as one of: {', '.join(TERMINAL_RUN)}")
        verdict = _text("verdict", verdict, 40)
        result = (result or "")[:RESULT_MAX]
        error = (error or "")[:NOTE_MAX]
        with self._tx() as conn:
            run = self._run(conn.execute("SELECT * FROM fm_runs WHERE id = ?", (run_id,)).fetchone())
            if run is None:
                raise FirstMateError("no such run")
            if run["status"] in TERMINAL_RUN:
                return run
            new_tokens = max(run["tokens"], int(tokens)) if tokens is not None else run["tokens"]
            new_usd = max(run["usd"], float(usd)) if usd is not None else run["usd"]
            conn.execute("UPDATE fm_runs SET status = ?, verdict = ?, result = ?, error = ?, tokens = ?, usd = ?,"
                         " ended_at = ? WHERE id = ?",
                         (status, verdict, result, error, new_tokens, new_usd, _iso(_utcnow()), run_id))
            self._event(conn, run["factory_id"], "deck", "run_finished",
                        {"run": run_id, "status": status, "verdict": verdict, "tokens": new_tokens})
        return self.get_run(run_id)

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self._lock:
            return self._run(self._c().execute("SELECT * FROM fm_runs WHERE id = ?", (str(run_id or ""),)).fetchone())

    def list_runs(self, factory_id: str, limit: int = 50, kinds: tuple[str, ...] | None = None,
                  archived: bool | None = None) -> list[dict[str, Any]]:
        """``archived``: None, every run (what scores, reports and trust count);
        False, those on the page; True, the archived ones."""
        sql, args = "SELECT * FROM fm_runs WHERE factory_id = ?", [factory_id]
        if archived is not None:
            sql += " AND archived_at != ''" if archived else " AND archived_at = ''"
        if kinds:
            sql += f" AND kind IN ({','.join('?' * len(kinds))})"
            args.extend(kinds)
        with self._lock:
            rows = self._c().execute(sql + " ORDER BY created_at DESC, id LIMIT ?",
                                     (*args, max(1, min(int(limit), 500)))).fetchall()
        return [self._run(r) for r in rows]

    def archive_run(self, run_id: str, *, on: bool = True) -> dict[str, Any]:
        """A finished run off the factory's Runs and Results, kept whole; or back."""
        with self._tx() as conn:
            run = self._run(conn.execute("SELECT * FROM fm_runs WHERE id = ?", (run_id,)).fetchone())
            if run is None:
                raise FirstMateError("no such run")
            if on and run["status"] not in TERMINAL_RUN:
                raise FirstMateError("only a finished run can be archived: stop it or let it finish first")
            conn.execute("UPDATE fm_runs SET archived_at = ? WHERE id = ?", (_iso(_utcnow()) if on else "", run_id))
        return self.get_run(run_id)

    def archive_finished(self, factory_id: str) -> int:
        """Every finished run of a factory off its lists. Returns how many."""
        with self._tx() as conn:
            cur = conn.execute(f"UPDATE fm_runs SET archived_at = ? WHERE factory_id = ? AND archived_at = '' "
                               f"AND status IN ({','.join('?' * len(TERMINAL_RUN))})",
                               (_iso(_utcnow()), factory_id, *TERMINAL_RUN))
            return int(cur.rowcount or 0)

    def archived_runs(self, factory_id: str) -> int:
        with self._lock:
            return int(self._c().execute("SELECT COUNT(*) FROM fm_runs WHERE factory_id = ? AND archived_at != ''",
                                         (factory_id,)).fetchone()[0])

    def archived_factories(self, owner_id: str) -> int:
        with self._lock:
            return int(self._c().execute("SELECT COUNT(*) FROM fm_factories WHERE owner_id = ? AND "
                                         "(archived_at != '' OR state = 'retired')",
                                         (norm_user(owner_id),)).fetchone()[0])

    def active_runs(self, owner_id: str | None = None) -> list[dict[str, Any]]:
        """Runs still going: one owner's, or everyone's (for the loop)."""
        sql = f"SELECT * FROM fm_runs WHERE status IN ({','.join('?' * len(ACTIVE_RUN))})"
        args: list[Any] = list(ACTIVE_RUN)
        if owner_id is not None:
            sql += " AND owner_id = ?"
            args.append(norm_user(owner_id))
        with self._lock:
            return [self._run(r) for r in self._c().execute(sql + " ORDER BY created_at", args).fetchall()]

    def set_proof(self, run_id: str, proof: dict[str, Any]) -> dict[str, Any]:
        """A finished carry-out's proof (Phase 3): ``{state, note, ...}``."""
        with self._tx() as conn:
            if conn.execute("SELECT 1 FROM fm_runs WHERE id = ?", (run_id,)).fetchone() is None:
                raise FirstMateError("no such run")
            conn.execute("UPDATE fm_runs SET proof = ? WHERE id = ?", (_sized("proof", proof), run_id))
        return self.get_run(run_id)

    def tokens_since(self, factory_id: str, since: datetime) -> int:
        """Tokens a factory's runs started since ``since`` have used."""
        with self._lock:
            return int(self._c().execute(
                "SELECT COALESCE(SUM(tokens), 0) FROM fm_runs WHERE factory_id = ? AND created_at >= ?",
                (factory_id, _iso(since))).fetchone()[0])

    def runs_since(self, factory_id: str, since: str, kinds: tuple[str, ...] = ("run",)) -> list[dict[str, Any]]:
        """A factory's runs created after ``since`` (ISO), newest first."""
        with self._lock:
            rows = self._c().execute(
                f"SELECT * FROM fm_runs WHERE factory_id = ? AND created_at > ?"
                f" AND kind IN ({','.join('?' * len(kinds))}) ORDER BY created_at DESC, id",
                (factory_id, since, *kinds)).fetchall()
        return [self._run(r) for r in rows]

    # ── the loop's bookkeeping (Phase 2) ───────────────────────────────

    def owners(self) -> list[str]:
        """Everyone with a factory that isn't retired."""
        with self._lock:
            return [r["owner_id"] for r in self._c().execute(
                "SELECT DISTINCT owner_id FROM fm_factories WHERE state != 'retired' ORDER BY owner_id")]

    def resumed_at(self, factory_id: str) -> str:
        """When the factory last started running: its latest approval or move
        back to running. A failure streak counts only runs after it."""
        with self._lock:
            rows = self._c().execute(
                "SELECT at, event, detail FROM fm_events WHERE factory_id = ? AND event IN ('approved', 'state')"
                " ORDER BY id DESC", (factory_id,)).fetchall()
        for r in rows:
            if r["event"] == "approved":
                return r["at"]
            try:
                if json.loads(r["detail"] or "{}").get("to") == "running":
                    return r["at"]
            except (TypeError, ValueError):
                continue
        return ""

    def trigger_state(self, factory_id: str) -> dict[str, dict[str, Any]]:
        with self._lock:
            rows = self._c().execute("SELECT key, next_at, last_at FROM fm_trigger_state WHERE factory_id = ?",
                                     (factory_id,)).fetchall()
        return {r["key"]: {"next_at": r["next_at"], "last_at": r["last_at"]} for r in rows}

    def set_trigger(self, factory_id: str, key: str, *, next_at: float | None,
                    last_at: float | None = None) -> None:
        """Save when a trigger is next due (None: never again) and, when it
        fired, when."""
        with self._tx() as conn:
            conn.execute(
                "INSERT INTO fm_trigger_state (factory_id, key, next_at, last_at) VALUES (?, ?, ?, ?)"
                " ON CONFLICT(factory_id, key) DO UPDATE SET next_at = excluded.next_at,"
                " last_at = COALESCE(excluded.last_at, fm_trigger_state.last_at)",
                (factory_id, key, next_at, last_at))

    def keep_triggers(self, factory_id: str, keys: list[str]) -> None:
        """Forget the trigger state of triggers the live version no longer has."""
        with self._tx() as conn:
            if keys:
                conn.execute(f"DELETE FROM fm_trigger_state WHERE factory_id = ? AND key NOT IN"
                             f" ({','.join('?' * len(keys))})", (factory_id, *keys))
            else:
                conn.execute("DELETE FROM fm_trigger_state WHERE factory_id = ?", (factory_id,))

    def note(self, factory_id: str, actor: str, event: str, detail: dict[str, Any] | None = None) -> None:
        """One line in a factory's history, from outside the store's own rules
        (the loop: a skipped schedule, a budget week)."""
        with self._tx() as conn:
            self._get_factory(conn, factory_id)
            self._event(conn, factory_id, actor, _text("event", event, 40, required=True), detail)

    # ── intents (Phase 3) ──────────────────────────────────────────────

    def add_intent(self, run_id: str, *, kind: str, tool: str = "", what: str = "",
                   args: Any = None, agent: str = "") -> int:
        kind = _text("kind", kind, 40, required=True)
        args_d = _dict("args", args)
        with self._tx() as conn:
            if conn.execute("SELECT 1 FROM fm_runs WHERE id = ?", (run_id,)).fetchone() is None:
                raise FirstMateError("no such run")
            cur = conn.execute(
                "INSERT INTO fm_intents (run_id, kind, tool, what, args, agent, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (run_id, kind, _text("tool", tool, 80), _text("what", what, ITEM_MAX * 2),
                 _sized("args", args_d), _text("agent", agent, 80), _iso(_utcnow())))
            return int(cur.lastrowid)

    def intents(self, run_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._c().execute("SELECT * FROM fm_intents WHERE run_id = ? ORDER BY id", (run_id,)).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            try:
                d["args"] = json.loads(d.get("args") or "{}")
            except (TypeError, ValueError):
                d["args"] = {}
            out.append(d)
        return out

    def decide_intents(self, run_id: str, status: str, ids: list[int] | None = None) -> int:
        """Move open intents (all of a run's, or ``ids``) to ``status``. Only
        open ones move, except approved → carried_out or dropped."""
        if status not in INTENT_STATUSES or status == "open":
            raise FirstMateError(f"an intent becomes one of: {', '.join(INTENT_STATUSES[1:])}")
        if status == "carried_out":
            from_states: tuple[str, ...] = ("approved",)
        elif status == "dropped":
            from_states = ("open", "approved")
        else:
            from_states = ("open",)
        sql = (f"UPDATE fm_intents SET status = ?, decided_at = ? WHERE run_id = ?"
               f" AND status IN ({','.join('?' * len(from_states))})")
        args: list[Any] = [status, _iso(_utcnow()), run_id, *from_states]
        if ids is not None:
            if not ids:
                return 0
            sql += f" AND id IN ({','.join('?' * len(ids))})"
            args.extend(int(i) for i in ids)
        with self._tx() as conn:
            n = conn.execute(sql, args).rowcount
        return int(n)

    def open_proposals(self, factory_id: str) -> list[str]:
        """A factory's rehearsals with intents still waiting for the owner, newest first."""
        with self._lock:
            rows = self._c().execute(
                "SELECT r.id FROM fm_runs r WHERE r.factory_id = ? AND EXISTS (SELECT 1 FROM fm_intents i"
                " WHERE i.run_id = r.id AND i.status = 'open') ORDER BY r.created_at DESC", (factory_id,)).fetchall()
        return [r["id"] for r in rows]

    # ── signals (Phase 4) ──────────────────────────────────────────────

    def add_signal(self, factory_id: str, kind: str, *, value: float | None = None, run_id: str = "",
                   source: str = "", ref: str = "") -> int:
        if kind not in SIGNAL_VALUES:
            raise FirstMateError(f"a signal is one of: {', '.join(SIGNAL_VALUES)}")
        fixed = SIGNAL_VALUES[kind]
        if fixed is None:
            if value is None:
                raise FirstMateError(f"{kind} needs a value between 0 and 1")
            worth = _num("value", value, 0.0, 1.0)
        else:
            worth = fixed
        with self._tx() as conn:
            self._get_factory(conn, factory_id)
            cur = conn.execute(
                "INSERT INTO fm_signals (factory_id, run_id, kind, value, source, ref, at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (factory_id, _text("run_id", run_id, 40), kind, worth, _text("source", source, 40),
                 _text("ref", ref, 200), _iso(_utcnow())))
            return int(cur.lastrowid)

    def set_thumbs(self, factory_id: str, run_id: str, kind: str) -> int:
        """The owner's 👍/👎 on a run: one per run, the latest wins; "" takes it back."""
        if kind not in ("thumbs_up", "thumbs_down", ""):
            raise FirstMateError("thumbs is thumbs_up, thumbs_down or nothing")
        with self._tx() as conn:
            self._get_factory(conn, factory_id)
            conn.execute("DELETE FROM fm_signals WHERE factory_id = ? AND run_id = ?"
                         " AND kind IN ('thumbs_up', 'thumbs_down')", (factory_id, run_id))
        return self.add_signal(factory_id, kind, run_id=run_id, source="owner") if kind else 0

    def thumbs(self, run_id: str) -> str:
        with self._lock:
            row = self._c().execute("SELECT kind FROM fm_signals WHERE run_id = ? AND kind IN"
                                    " ('thumbs_up', 'thumbs_down') ORDER BY id DESC LIMIT 1", (run_id,)).fetchone()
        return row["kind"] if row else ""

    # ── checks (Phase 4) ───────────────────────────────────────────────

    CHECK_PURPOSES = ("approve", "baseline", "weekly", "manual")
    CHECK_STATUSES = ("running", "done", "void", "failed")

    @staticmethod
    def _check(row: sqlite3.Row | None) -> dict[str, Any] | None:
        if row is None:
            return None
        d = dict(row)
        for key, empty in (("cases", "[]"), ("results", "{}")):
            try:
                d[key] = json.loads(d.get(key) or empty)
            except (TypeError, ValueError):
                d[key] = json.loads(empty)
        return d

    def create_check(self, factory_id: str, *, version: int, purpose: str, cases: list[str],
                     baseline: float | None = None) -> dict[str, Any]:
        """One check at a time per factory; it runs the given version's cases."""
        if purpose not in self.CHECK_PURPOSES:
            raise FirstMateError(f"a check is for one of: {', '.join(self.CHECK_PURPOSES)}")
        if not cases:
            raise FirstMateError("a check needs at least one case")
        with self._tx() as conn:
            self._get_factory(conn, factory_id)
            if conn.execute("SELECT 1 FROM fm_checks WHERE factory_id = ? AND status = 'running'",
                            (factory_id,)).fetchone():
                raise FirstMateError("this factory is being checked already")
            v = self._get_version(conn, factory_id, version)
            if v is None:
                raise FirstMateError("no such version")
            cid = "fmk_" + secrets.token_hex(8)
            conn.execute("INSERT INTO fm_checks (id, factory_id, version, version_sha, purpose, cases, baseline,"
                         " created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                         (cid, factory_id, version, v["sha"], purpose, json.dumps(list(cases)), baseline,
                          _iso(_utcnow())))
            self._event(conn, factory_id, "first_mate", "check_started",
                        {"check": cid, "version": version, "purpose": purpose, "cases": len(cases)})
        return self.get_check(cid)

    def get_check(self, check_id: str) -> dict[str, Any] | None:
        with self._lock:
            return self._check(self._c().execute("SELECT * FROM fm_checks WHERE id = ?", (check_id,)).fetchone())

    def active_check(self, factory_id: str) -> dict[str, Any] | None:
        with self._lock:
            return self._check(self._c().execute(
                "SELECT * FROM fm_checks WHERE factory_id = ? AND status = 'running'", (factory_id,)).fetchone())

    def running_checks(self) -> list[dict[str, Any]]:
        with self._lock:
            return [self._check(r) for r in self._c().execute(
                "SELECT * FROM fm_checks WHERE status = 'running' ORDER BY created_at").fetchall()]

    def checks(self, factory_id: str, limit: int = 20) -> list[dict[str, Any]]:
        with self._lock:
            return [self._check(r) for r in self._c().execute(
                "SELECT * FROM fm_checks WHERE factory_id = ? ORDER BY created_at DESC LIMIT ?",
                (factory_id, max(1, min(int(limit), 200)))).fetchall()]

    def set_check_result(self, check_id: str, case_id: str, result: dict[str, Any]) -> dict[str, Any]:
        with self._tx() as conn:
            row = self._check(conn.execute("SELECT * FROM fm_checks WHERE id = ?", (check_id,)).fetchone())
            if row is None:
                raise FirstMateError("no such check")
            results = {**row["results"], str(case_id): result}
            conn.execute("UPDATE fm_checks SET results = ? WHERE id = ?", (_sized("results", results), check_id))
        return self.get_check(check_id)

    def finish_check(self, check_id: str, status: str, *, score: float | None = None, note: str = "") -> dict[str, Any]:
        if status not in self.CHECK_STATUSES or status == "running":
            raise FirstMateError("a check ends done, void or failed")
        with self._tx() as conn:
            row = self._check(conn.execute("SELECT * FROM fm_checks WHERE id = ?", (check_id,)).fetchone())
            if row is None:
                raise FirstMateError("no such check")
            if row["status"] != "running":
                return row
            conn.execute("UPDATE fm_checks SET status = ?, score = ?, note = ?, ended_at = ? WHERE id = ?",
                         (status, score, _text("note", note, NOTE_MAX), _iso(_utcnow()), check_id))
            self._event(conn, row["factory_id"], "first_mate", "check_finished",
                        {"check": check_id, "status": status, "score": score, "version": row["version"],
                         "purpose": row["purpose"], "note": note[:200]})
        return self.get_check(check_id)

    def last_score(self, factory_id: str, version: int | None = None) -> float | None:
        """The latest finished check's score (of ``version`` when given)."""
        sql, args = "SELECT score FROM fm_checks WHERE factory_id = ? AND status = 'done' AND score IS NOT NULL", [factory_id]
        if version is not None:
            sql += " AND version = ?"
            args.append(version)
        with self._lock:
            row = self._c().execute(sql + " ORDER BY ended_at DESC LIMIT 1", args).fetchone()
        return float(row["score"]) if row else None

    # ── drafts (Phase 4) ───────────────────────────────────────────────

    DRAFT_STATUSES = ("waiting", "sent", "edited", "deleted", "expired")

    def add_draft(self, factory_id: str, owner_id: str, *, run_id: str, draft_id: str, message_id: str = "",
                  thread_id: str = "", subject: str = "", body: str = "", partial: bool = False) -> bool:
        """Watch a draft a run left; once only per draft. False when it was there already."""
        draft_id = _text("draft_id", draft_id, 200, required=True)
        with self._tx() as conn:
            self._get_factory(conn, factory_id)
            cur = conn.execute(
                "INSERT OR IGNORE INTO fm_drafts (factory_id, run_id, owner_id, draft_id, message_id, thread_id,"
                " subject, body, partial, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (factory_id, run_id[:40], norm_user(owner_id), draft_id, str(message_id)[:200],
                 str(thread_id)[:200], str(subject)[:300], str(body)[:4000], int(bool(partial)), _iso(_utcnow())))
            return cur.rowcount > 0

    def drafts_to_check(self, now: datetime, *, older_than: timedelta, every: timedelta,
                        limit: int = 20) -> list[dict[str, Any]]:
        """Waiting drafts old enough to look at, not looked at lately, oldest first."""
        with self._lock:
            rows = self._c().execute(
                "SELECT * FROM fm_drafts WHERE status = 'waiting' AND created_at <= ?"
                " AND (checked_at IS NULL OR checked_at <= ?) ORDER BY created_at LIMIT ?",
                (_iso(now - older_than), _iso(now - every), max(1, int(limit)))).fetchall()
        return [dict(r) for r in rows]

    def set_draft(self, draft_row_id: int, status: str, *, similarity: float | None = None,
                  now: datetime | None = None) -> None:
        if status not in self.DRAFT_STATUSES:
            raise FirstMateError(f"a draft is one of: {', '.join(self.DRAFT_STATUSES)}")
        with self._tx() as conn:
            conn.execute("UPDATE fm_drafts SET status = ?, similarity = COALESCE(?, similarity), checked_at = ?"
                         " WHERE id = ?", (status, similarity, _iso(now or _utcnow()), int(draft_row_id)))

    def drafts(self, factory_id: str, limit: int = 50) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(r) for r in self._c().execute(
                "SELECT * FROM fm_drafts WHERE factory_id = ? ORDER BY id DESC LIMIT ?",
                (factory_id, max(1, min(int(limit), 500)))).fetchall()]

    def signals(self, factory_id: str, limit: int = 100) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(r) for r in self._c().execute(
                "SELECT * FROM fm_signals WHERE factory_id = ? ORDER BY id DESC LIMIT ?",
                (factory_id, max(1, min(int(limit), 500)))).fetchall()]

    def satisfaction(self, factory_id: str, last: int = SATISFACTION_WINDOW) -> float | None:
        """The average worth of the last ``last`` signals; None with none yet.
        Silence is never a signal, so nothing unknown is counted."""
        with self._lock:
            rows = self._c().execute("SELECT value FROM fm_signals WHERE factory_id = ? ORDER BY id DESC LIMIT ?",
                                     (factory_id, max(1, int(last)))).fetchall()
        if not rows:
            return None
        return sum(float(r["value"]) for r in rows) / len(rows)

    # ── forgetting ─────────────────────────────────────────────────────

    def forget_user(self, user_id: str) -> int:
        """Every factory of the user, with its runs, history and sealed cases,
        and their preferences. Returns how many factories went."""
        owner = norm_user(user_id)
        with self._tx() as conn:
            ids = [r["id"] for r in conn.execute("SELECT id FROM fm_factories WHERE owner_id = ?", (owner,))]
            conn.execute("DELETE FROM fm_factories WHERE owner_id = ?", (owner,))
            conn.execute("DELETE FROM fm_settings WHERE key IN (?, ?)", (f"prefs:{owner}", f"report:{owner}"))
        for fid in ids:
            shutil.rmtree(self.sealed_dir(fid), ignore_errors=True)
        return len(ids)


_STORE: FirstMateStore | None = None
_STORE_LOCK = threading.Lock()


def get_store() -> FirstMateStore:
    global _STORE
    if _STORE is not None:
        return _STORE
    with _STORE_LOCK:
        if _STORE is None:
            _STORE = FirstMateStore()
        return _STORE


def forget_user(user_id: str) -> int:
    """For ``admin_routes.delete_user``. Never raises, and never creates the
    store on a deck that has never used First Mate."""
    try:
        if _STORE is None and not _db_path().exists():
            return 0
        return get_store().forget_user(user_id)
    except Exception:  # noqa: BLE001 — a deletion never fails over First Mate
        return 0
