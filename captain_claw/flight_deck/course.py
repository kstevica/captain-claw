"""Course — what the work is for (see docs/course-plan.md).

Goals at three scopes, each with a horizon, arranged as a tree:

* ``instance`` — the whole deck, set by an admin;
* ``user``     — one user space, set by that user;
* ``agent``    — one agent, set by its owner (keyed by ``agent_ref``, so a
  removed-and-recreated agent with the same slug never inherits them).

Which horizons each scope uses, and how many active goals a horizon may hold,
are the admin's settings (``course_settings``). A goal *serves* a parent in its
own scope or a wider one, never the other way round, and never a shorter
horizon than its own.

This module is the store and its rules: ``CourseStore`` (sync sqlite3 behind a
lock, mirroring ``AutonomyStore``) at ``FD_DATA_DIR/course.db``. Who may call
what is decided in ``course_routes``; how goals reach the agents is in
``course_carrier``. Every rule a person or an agent could get around from the
browser is enforced here, not in the UI.
"""

from __future__ import annotations

import json
import os
import re
import secrets
import sqlite3
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

# ── Vocabulary ─────────────────────────────────────────────────────────

SCOPES = ("instance", "user", "agent")
# Longest first: a parent's horizon must come no later in this order than its child's.
HORIZONS = ("strategic", "long", "medium", "short")
HORIZON_RANK = {h: i for i, h in enumerate(HORIZONS)}
HORIZON_LABELS = {"strategic": "Strategic", "long": "Long-term", "medium": "Medium-term",
                  "short": "Short-term"}

STATUSES = ("proposed", "active", "paused", "achieved", "dropped", "superseded", "declined")
ORIGINS = ("admin", "user", "agent", "deck")
LINK_KINDS = ("autonomy_action", "plan", "project", "task", "schedule", "swarm_run",
              "session", "line_item")

# Which status may follow which. A goal is never deleted by a person: it is
# achieved or dropped, and can be reopened.
_TRANSITIONS: dict[str, tuple[str, ...]] = {
    "proposed": ("declined",),       # becoming active is accept()'s: it may replace a goal
    "active": ("paused", "achieved", "dropped"),
    "paused": ("active", "achieved", "dropped"),
    "achieved": ("active",),
    "dropped": ("active",),
    "superseded": (),
    "declined": (),
}

# The scopes a goal of each scope may serve.
_PARENT_SCOPES = {
    "instance": ("instance",),
    "user": ("user", "instance"),
    "agent": ("agent", "user", "instance"),
}

DEFAULT_HORIZONS: dict[str, list[str]] = {
    "instance": ["strategic", "long"],
    "user": list(HORIZONS),
    "agent": ["medium", "short"],
}
DEFAULT_CAPS: dict[str, int] = {"strategic": 3, "long": 5, "medium": 7, "short": 3}
CAP_RANGE = (1, 20)

# Phase 3: goals the agents and the deck suggest. Every one waits for a person.
DEFAULT_PROPOSALS: dict[str, Any] = {"recommendations": True, "agent_per_week": 2, "deck_per_week": 2}
PER_WEEK_RANGE = (0, 20)
WAITING_MAX = 5              # waiting proposals per scope (one user's, one agent's)
DECLINE_MEMORY_DAYS = 90     # a declined title isn't suggested again in that scope for this long
PROPOSAL_ORIGINS = ("agent", "deck")
DEFAULT_PREFS: dict[str, Any] = {"auto_accept_agent_short": False}
CHANGEABLE = ("horizon", "title", "success", "detail", "parent_id", "due_at")

REVIEW_DAYS: dict[str, int] = {"short": 7, "medium": 30, "long": 90, "strategic": 365}
REVIEW_RANGE = (1, 730)
GRACE_MIN_DAYS = 3

TEXT_CAPS: dict[str, int] = {"title": 120, "detail": 1000, "success": 500, "rationale": 500,
                             "progress_note": 500, "note": 300, "unit": 30, "reason": 300}
_REF_MAX = 200

# Phase 4: progress and off course.
# A goal's metric: how its progress is measured. A deck-wide goal's work is
# everyone's, and each user sees only their own, so it takes only a value the
# admin enters.
METRIC_KINDS = ("manual", "taskboard_done", "link_count")
METRIC_KIND_LABELS = {"manual": "A number you enter", "taskboard_done": "Tasks done for it",
                      "link_count": "Pieces of work linked to it"}
METRIC_MAX = 1_000_000_000
# What Flight Deck does on its own (the admin's switches).
DEFAULT_AUTOMATION: dict[str, bool] = {
    "weekly_estimate": True,           # estimate progress once a week from new evidence
    "conflict_check": True,            # check a new or accepted goal against the others
    "goal_driven_candidates": False,   # stalled goals become Arbiter candidates
}
ESTIMATE_EVERY_DAYS = 7
OFF_COURSE_MIN_DAYS = 14   # a goal is judged off course once active this long (or its review period)
CONFLICT_STATUSES = ("open", "dismissed")
_SCOPE_RANK = {"instance": 0, "user": 1, "agent": 2}   # wider first


class CourseError(ValueError):
    """A request the rules refuse; the message says why, in the user's words."""


class CourseNotFoundError(LookupError):
    """No such goal (or not one the caller may see — the routes decide)."""


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _iso(dt: datetime) -> str:
    return dt.astimezone(UTC).isoformat()


def _parse_iso(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


def norm_user(user_id: str | None) -> str:
    """Empty (auth off / internal) → 'local', as Autonomous Work and the Line do."""
    return (user_id or "").strip() or "local"


def _db_path() -> Path:
    base = os.environ.get("FD_DATA_DIR", "").strip()
    if base:
        return Path(base).expanduser().resolve() / "course.db"
    return Path("~/.captain-claw/course.db").expanduser()


def _text(field: str, value: Any, *, required: bool = False) -> str:
    """A person's (or an agent's) text for ``field``: a string within its cap."""
    if value is None:
        value = ""
    if not isinstance(value, str):
        raise CourseError(f"{field} must be text")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        raise CourseError(f"{field} contains characters that aren't valid text") from None
    value = value.strip()
    cap = TEXT_CAPS[field]
    if len(value) > cap:
        raise CourseError(f"{field} is too long ({len(value)} characters, at most {cap})")
    if required and not value:
        raise CourseError(f"{field} is required")
    return value


def grace_days(review_every_days: int) -> int:
    """How long a goal past its review stays in the agents' prompts."""
    return max(GRACE_MIN_DAYS, int(review_every_days) // 4)


def is_due(goal: dict[str, Any], now: datetime | None = None) -> bool:
    """Active and past its review date."""
    if goal.get("status") != "active":
        return False
    at = _parse_iso(goal.get("review_at"))
    return bool(at and at <= (now or _utcnow()))


def is_stale(goal: dict[str, Any], now: datetime | None = None) -> bool:
    """Active and past its review date by more than the grace period: it leaves
    the agents' prompts until someone reviews it. Computed, never stored."""
    if goal.get("status") != "active":
        return False
    at = _parse_iso(goal.get("review_at"))
    if at is None:
        return False
    days = grace_days(int(goal.get("review_every_days") or REVIEW_DAYS["medium"]))
    return at + timedelta(days=days) <= (now or _utcnow())


def scope_key(goal: dict[str, Any]) -> tuple[str, str, str]:
    return (str(goal.get("scope") or ""), str(goal.get("owner_id") or ""),
            str(goal.get("agent_ref") or ""))


def _check_horizons(raw: Any) -> dict[str, list[str]]:
    if not isinstance(raw, dict):
        raise CourseError("horizons must be an object of scope → list of horizons")
    out: dict[str, list[str]] = {}
    for scope in SCOPES:
        value = raw.get(scope, DEFAULT_HORIZONS[scope])
        if not isinstance(value, list) or not all(isinstance(h, str) for h in value):
            raise CourseError(f"horizons.{scope} must be a list of horizons")
        bad = [h for h in value if h not in HORIZONS]
        if bad:
            raise CourseError(f"horizons.{scope}: unknown horizon {bad[0]!r}")
        out[scope] = [h for h in HORIZONS if h in value]  # canonical order, deduped
    return out


def _check_proposals(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise CourseError("proposals must be an object")
    out: dict[str, Any] = {}
    rec = raw.get("recommendations", DEFAULT_PROPOSALS["recommendations"])
    if not isinstance(rec, bool):
        raise CourseError("proposals.recommendations must be true or false")
    out["recommendations"] = rec
    lo, hi = PER_WEEK_RANGE
    for key in ("agent_per_week", "deck_per_week"):
        value = raw.get(key, DEFAULT_PROPOSALS[key])
        if isinstance(value, bool) or not isinstance(value, int) or not lo <= value <= hi:
            raise CourseError(f"proposals.{key} must be a whole number from {lo} to {hi}")
        out[key] = value
    return out


def _check_prefs(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise CourseError("prefs must be an object")
    value = raw.get("auto_accept_agent_short", DEFAULT_PREFS["auto_accept_agent_short"])
    if not isinstance(value, bool):
        raise CourseError("auto_accept_agent_short must be true or false")
    return {"auto_accept_agent_short": value}


def _check_automation(raw: Any) -> dict[str, bool]:
    if not isinstance(raw, dict):
        raise CourseError("automation must be an object")
    out: dict[str, bool] = {}
    for key, default in DEFAULT_AUTOMATION.items():
        value = raw.get(key, default)
        if not isinstance(value, bool):
            raise CourseError(f"automation.{key} must be true or false")
        out[key] = value
    return out


def _number(field: str, value: Any, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CourseError(f"{field} must be a number")
    value = float(value)
    if value != value or not minimum <= value <= METRIC_MAX:   # NaN, or out of range
        raise CourseError(f"{field} must be a number from {minimum:g} to {METRIC_MAX:,}")
    return value


def check_metric(raw: Any, scope: str, old: dict[str, Any] | None = None) -> dict[str, Any]:
    """A goal's metric as stored: ``{}`` (none), or ``{kind, target, unit}``
    plus, for ``manual``, the last ``value`` entered (kept across an edit of
    the target or unit; a new kind starts empty)."""
    if raw in (None, "") or (isinstance(raw, dict) and not raw.get("kind")):
        return {}
    if not isinstance(raw, dict):
        raise CourseError("metric must be an object")
    kind = raw.get("kind")
    if kind not in METRIC_KINDS:
        raise CourseError(f"metric.kind must be one of {', '.join(METRIC_KINDS)}")
    if scope == "instance" and kind != "manual":
        raise CourseError("A deck-wide goal's work is everyone's: measure it with a number "
                          "the admin enters")
    unknown = [k for k in raw if k not in ("kind", "target", "unit", "value", "value_at")]
    if unknown:
        raise CourseError(f"metric.{unknown[0]} is unknown")
    target = raw.get("target")
    out: dict[str, Any] = {
        "kind": kind,
        "target": None if target in (None, "") else _number("metric.target", target, minimum=0.0),
        "unit": _text("unit", raw.get("unit")),
    }
    if out["target"] == 0:
        raise CourseError("metric.target must be more than 0")
    if kind == "manual":
        old = old or {}
        same = old.get("kind") == "manual"
        out["value"] = old.get("value") if same else None
        out["value_at"] = old.get("value_at") if same else None
    return out


def norm_title(text: Any) -> str:
    """A title as compared for duplicates: lower case, letters and digits only."""
    return " ".join(re.sub(r"[^\w]+", " ", str(text or "").lower()).split())


def _check_caps(raw: Any) -> dict[str, int]:
    if not isinstance(raw, dict):
        raise CourseError("caps must be an object of horizon → number")
    out: dict[str, int] = {}
    lo, hi = CAP_RANGE
    for h in HORIZONS:
        value = raw.get(h, DEFAULT_CAPS[h])
        if isinstance(value, bool) or not isinstance(value, int):
            raise CourseError(f"caps.{h} must be a whole number")
        if not lo <= value <= hi:
            raise CourseError(f"caps.{h} must be between {lo} and {hi}")
        out[h] = value
    return out


# ── Store ──────────────────────────────────────────────────────────────

class CourseStore:
    """SQLite-backed goals, their evidence links, their history and the admin's
    settings. Sync sqlite3 + a lock, mirroring ``AutonomyStore``."""

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
            self._conn = conn
        return self._conn

    def _ensure_db(self) -> None:
        with self._lock:
            self._c().executescript(
                """
                CREATE TABLE IF NOT EXISTS course_goals (
                    id                TEXT PRIMARY KEY,
                    scope             TEXT NOT NULL,             -- instance|user|agent
                    owner_id          TEXT NOT NULL DEFAULT '',  -- '' for instance
                    agent_ref         TEXT NOT NULL DEFAULT '',  -- '' unless scope=agent
                    horizon           TEXT NOT NULL,             -- strategic|long|medium|short
                    parent_id         TEXT NOT NULL DEFAULT '',
                    title             TEXT NOT NULL,
                    detail            TEXT NOT NULL DEFAULT '',
                    success           TEXT NOT NULL DEFAULT '',
                    metric            TEXT NOT NULL DEFAULT '{}',
                    status            TEXT NOT NULL DEFAULT 'active',
                    origin            TEXT NOT NULL DEFAULT 'user',
                    proposed_by       TEXT NOT NULL DEFAULT '',
                    rationale         TEXT NOT NULL DEFAULT '',
                    supersedes_id     TEXT NOT NULL DEFAULT '',
                    review_every_days INTEGER NOT NULL DEFAULT 30,
                    review_at         TEXT,
                    due_at            TEXT,
                    progress          REAL,
                    progress_note     TEXT NOT NULL DEFAULT '',
                    sort              INTEGER NOT NULL DEFAULT 0,
                    created_at        TEXT NOT NULL,
                    updated_at        TEXT NOT NULL,
                    decided_at        TEXT,
                    decided_by        TEXT NOT NULL DEFAULT ''
                );
                CREATE INDEX IF NOT EXISTS idx_course_scope
                    ON course_goals(scope, owner_id, agent_ref, status);
                CREATE INDEX IF NOT EXISTS idx_course_parent ON course_goals(parent_id);

                -- The evidence: what moved a goal forward.
                CREATE TABLE IF NOT EXISTS course_links (
                    goal_id   TEXT NOT NULL REFERENCES course_goals(id) ON DELETE CASCADE,
                    kind      TEXT NOT NULL,
                    ref_id    TEXT NOT NULL,
                    note      TEXT NOT NULL DEFAULT '',
                    added_by  TEXT NOT NULL DEFAULT '',
                    at        TEXT NOT NULL,
                    PRIMARY KEY (goal_id, kind, ref_id)
                );

                -- The history shown in a goal's drawer.
                CREATE TABLE IF NOT EXISTS course_events (
                    id       INTEGER PRIMARY KEY AUTOINCREMENT,
                    goal_id  TEXT NOT NULL REFERENCES course_goals(id) ON DELETE CASCADE,
                    at       TEXT NOT NULL,
                    actor    TEXT NOT NULL DEFAULT '',
                    event    TEXT NOT NULL,
                    detail   TEXT NOT NULL DEFAULT '{}'
                );
                CREATE INDEX IF NOT EXISTS idx_course_events ON course_events(goal_id, id);

                -- The admin's settings: horizons per scope, caps per horizon.
                CREATE TABLE IF NOT EXISTS course_settings (
                    key    TEXT PRIMARY KEY,
                    value  TEXT NOT NULL
                );

                -- Phase 4: goals that likely pull against each other. The deck flags
                -- them; a person settles them. goal_id is the narrower of the two.
                CREATE TABLE IF NOT EXISTS course_conflicts (
                    id          TEXT PRIMARY KEY,
                    goal_id     TEXT NOT NULL REFERENCES course_goals(id) ON DELETE CASCADE,
                    other_id    TEXT NOT NULL REFERENCES course_goals(id) ON DELETE CASCADE,
                    owner_id    TEXT NOT NULL DEFAULT '',   -- who is told; '' = the deck's admins
                    reason      TEXT NOT NULL DEFAULT '',
                    status      TEXT NOT NULL DEFAULT 'open',
                    at          TEXT NOT NULL,
                    decided_at  TEXT,
                    decided_by  TEXT NOT NULL DEFAULT ''
                );
                CREATE INDEX IF NOT EXISTS idx_course_conflicts ON course_conflicts(owner_id, status);
                """
            )
            self._c().commit()
            have = {r["name"] for r in self._c().execute("PRAGMA table_info(course_links)")}
            if "owner_id" not in have:
                # Whose work a link is: a deck-wide goal shows each user only their own.
                self._c().execute(
                    "ALTER TABLE course_links ADD COLUMN owner_id TEXT NOT NULL DEFAULT ''")
                self._c().commit()
            have = {r["name"] for r in self._c().execute("PRAGMA table_info(course_goals)")}
            if "estimated_at" not in have:
                # When the weekly estimate last looked at the goal (Phase 4).
                self._c().execute("ALTER TABLE course_goals ADD COLUMN estimated_at TEXT")
                self._c().commit()

    # ── rows ───────────────────────────────────────────────────────────

    @staticmethod
    def _row(row: sqlite3.Row | None) -> dict[str, Any] | None:
        if row is None:
            return None
        d = dict(row)
        try:
            d["metric"] = json.loads(d.get("metric") or "{}")
        except (TypeError, ValueError):
            d["metric"] = {}
        return d

    def _event(self, conn: sqlite3.Connection, goal_id: str, actor: str, event: str,
               detail: dict[str, Any] | None = None) -> None:
        conn.execute(
            "INSERT INTO course_events (goal_id, at, actor, event, detail) VALUES (?, ?, ?, ?, ?)",
            (goal_id, _iso(_utcnow()), actor, event, json.dumps(detail or {})),
        )

    # ── settings ───────────────────────────────────────────────────────

    def settings(self) -> dict[str, Any]:
        """``{"horizons": {scope: [...]}, "caps": {horizon: n}}`` — the admin's
        choice, else the defaults (a damaged value falls back to them)."""
        with self._lock:
            rows = {r["key"]: r["value"] for r in
                    self._c().execute("SELECT key, value FROM course_settings").fetchall()}
        out: dict[str, Any] = {}
        for key, check, default in (("horizons", _check_horizons, DEFAULT_HORIZONS),
                                    ("caps", _check_caps, DEFAULT_CAPS),
                                    ("proposals", _check_proposals, DEFAULT_PROPOSALS),
                                    ("automation", _check_automation, DEFAULT_AUTOMATION)):
            try:
                out[key] = check(json.loads(rows[key])) if key in rows else check(default)
            except (CourseError, TypeError, ValueError):
                out[key] = check(default)
        return out

    def set_settings(self, *, horizons: Any = None, caps: Any = None,
                     proposals: Any = None, automation: Any = None) -> dict[str, Any]:
        """Save what is given (None = unchanged). Goals already at a horizon a
        scope no longer uses are kept; only new ones are refused."""
        updates: dict[str, Any] = {}
        if horizons is not None:
            updates["horizons"] = _check_horizons(horizons)
        if caps is not None:
            updates["caps"] = _check_caps(caps)
        if proposals is not None:
            updates["proposals"] = _check_proposals({**self.settings()["proposals"], **proposals}
                                                    if isinstance(proposals, dict) else proposals)
        if automation is not None:
            updates["automation"] = _check_automation(
                {**self.settings()["automation"], **automation}
                if isinstance(automation, dict) else automation)
        if updates:
            with self._lock:
                conn = self._c()
                for key, value in updates.items():
                    conn.execute(
                        "INSERT INTO course_settings (key, value) VALUES (?, ?)"
                        " ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                        (key, json.dumps(value)),
                    )
                conn.commit()
        return self.settings()

    def prefs(self, user_id: str) -> dict[str, Any]:
        """One user's Course preferences (the defaults when none were saved)."""
        with self._lock:
            row = self._c().execute("SELECT value FROM course_settings WHERE key = ?",
                                    (f"prefs:{norm_user(user_id)}",)).fetchone()
        try:
            return _check_prefs(json.loads(row["value"])) if row else dict(DEFAULT_PREFS)
        except (CourseError, TypeError, ValueError):
            return dict(DEFAULT_PREFS)

    def set_prefs(self, user_id: str, prefs: Any) -> dict[str, Any]:
        merged = _check_prefs({**self.prefs(user_id), **prefs} if isinstance(prefs, dict) else prefs)
        with self._lock:
            conn = self._c()
            conn.execute("INSERT INTO course_settings (key, value) VALUES (?, ?)"
                         " ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                         (f"prefs:{norm_user(user_id)}", json.dumps(merged)))
            conn.commit()
        return merged

    # ── reads ──────────────────────────────────────────────────────────

    def get(self, goal_id: str) -> dict[str, Any] | None:
        with self._lock:
            return self._row(self._c().execute(
                "SELECT * FROM course_goals WHERE id = ?", (str(goal_id or ""),)).fetchone())

    def list_goals(
        self,
        *,
        scope: str | None = None,
        owner_id: str | None = None,
        agent_refs: list[str] | None = None,
        statuses: tuple[str, ...] | list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Goals matching every filter given, longest horizon first, then sort, then age."""
        where, args = [], []
        if scope is not None:
            where.append("scope = ?")
            args.append(scope)
        if owner_id is not None:
            where.append("owner_id = ?")
            args.append(owner_id)
        if agent_refs is not None:
            if not agent_refs:
                return []
            where.append(f"agent_ref IN ({','.join('?' * len(agent_refs))})")
            args.extend(agent_refs)
        if statuses is not None:
            if not statuses:
                return []
            where.append(f"status IN ({','.join('?' * len(statuses))})")
            args.extend(statuses)
        sql = "SELECT * FROM course_goals"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += (" ORDER BY CASE horizon WHEN 'strategic' THEN 0 WHEN 'long' THEN 1"
                " WHEN 'medium' THEN 2 ELSE 3 END, sort, created_at")
        with self._lock:
            return [self._row(r) for r in self._c().execute(sql, args).fetchall()]

    def links(self, goal_id: str, *, owner_id: str | None = None) -> list[dict[str, Any]]:
        """A goal's evidence, newest first — only ``owner_id``'s when given."""
        sql, args = "SELECT * FROM course_links WHERE goal_id = ?", [goal_id]
        if owner_id is not None:
            sql += " AND owner_id = ?"
            args.append(norm_user(owner_id))
        with self._lock:
            return [dict(r) for r in self._c().execute(sql + " ORDER BY at DESC", args).fetchall()]

    def links_for(self, kind: str, ref_id: str) -> list[dict[str, Any]]:
        """The goals one item (a plan, a project, a scheduled task…) is linked to."""
        with self._lock:
            return [dict(r) for r in self._c().execute(
                "SELECT * FROM course_links WHERE kind = ? AND ref_id = ? ORDER BY at DESC",
                (kind, str(ref_id or ""))).fetchall()]

    def events(self, goal_id: str, limit: int = 100) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._c().execute(
                "SELECT * FROM course_events WHERE goal_id = ? ORDER BY id DESC LIMIT ?",
                (goal_id, max(1, min(int(limit), 500))),
            ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            try:
                d["detail"] = json.loads(d.get("detail") or "{}")
            except (TypeError, ValueError):
                d["detail"] = {}
            out.append(d)
        return out

    def due_for_review(self, now: datetime | None = None) -> list[dict[str, Any]]:
        """Active goals whose review date has passed (stale ones included)."""
        with self._lock:
            rows = self._c().execute(
                "SELECT * FROM course_goals WHERE status = 'active' AND review_at IS NOT NULL"
                " AND review_at <= ?", (_iso(now or _utcnow()),)
            ).fetchall()
        return [self._row(r) for r in rows]

    # ── rules ──────────────────────────────────────────────────────────

    def _active_count(self, conn: sqlite3.Connection, key: tuple[str, str, str],
                      horizon: str, exclude_id: str = "") -> int:
        return int(conn.execute(
            "SELECT COUNT(*) FROM course_goals WHERE scope = ? AND owner_id = ? AND agent_ref = ?"
            " AND horizon = ? AND status = 'active' AND id != ?",
            (*key, horizon, exclude_id),
        ).fetchone()[0])

    def _check_cap(self, conn: sqlite3.Connection, key: tuple[str, str, str], horizon: str,
                   exclude_id: str = "") -> None:
        cap = self.settings()["caps"][horizon]
        if self._active_count(conn, key, horizon, exclude_id) >= cap:
            raise CourseError(
                f"This level already has {cap} active {HORIZON_LABELS[horizon].lower()} "
                f"goal{'s' if cap != 1 else ''}, the most it may hold. Finish, pause or drop "
                "one first.")

    def _check_parent(self, conn: sqlite3.Connection, child: dict[str, Any],
                      parent_id: str) -> None:
        """Raise unless ``parent_id`` (non-empty) may be ``child``'s parent."""
        row = conn.execute("SELECT * FROM course_goals WHERE id = ?", (parent_id,)).fetchone()
        parent = self._row(row)
        if parent is None:
            raise CourseError("The goal it serves doesn't exist")
        scope = child["scope"]
        if parent["scope"] not in _PARENT_SCOPES[scope]:
            raise CourseError(f"A {scope} goal can't serve a {parent['scope']} goal")
        if parent["scope"] == "user" and parent["owner_id"] != child["owner_id"]:
            raise CourseError("The goal it serves doesn't exist")
        if parent["scope"] == "agent" and parent["agent_ref"] != child["agent_ref"]:
            raise CourseError("The goal it serves doesn't exist")
        if HORIZON_RANK[parent["horizon"]] > HORIZON_RANK[child["horizon"]]:
            raise CourseError(
                f"A {HORIZON_LABELS[child['horizon']].lower()} goal can't serve a "
                f"{HORIZON_LABELS[parent['horizon']].lower()} one: it can only serve a goal "
                "of the same horizon or a longer one")
        if parent["status"] not in ("active", "proposed"):
            raise CourseError("The goal it serves isn't active")
        # No cycle: walking up from the parent must never reach the child.
        seen = {child.get("id") or ""}
        cur: dict[str, Any] | None = parent
        while cur is not None:
            if cur["id"] in seen:
                raise CourseError("A goal can't serve itself, directly or through others")
            seen.add(cur["id"])
            nxt = cur.get("parent_id") or ""
            cur = self._row(conn.execute(
                "SELECT * FROM course_goals WHERE id = ?", (nxt,)).fetchone()) if nxt else None

    def _review_days(self, value: Any, horizon: str) -> int:
        if value is None or value == "":
            return REVIEW_DAYS[horizon]
        if isinstance(value, bool) or not isinstance(value, int):
            raise CourseError("review_every_days must be a whole number of days")
        lo, hi = REVIEW_RANGE
        if not lo <= value <= hi:
            raise CourseError(f"review_every_days must be between {lo} and {hi}")
        return value

    @staticmethod
    def _due_at(value: Any) -> str | None:
        if value in (None, ""):
            return None
        dt = _parse_iso(value)
        if dt is None:
            raise CourseError("due_at must be a date")
        return _iso(dt)

    # ── writes ─────────────────────────────────────────────────────────

    def create(
        self,
        *,
        scope: str,
        horizon: str,
        title: Any,
        owner_id: str = "",
        agent_ref: str = "",
        detail: Any = "",
        success: Any = "",
        parent_id: Any = "",
        due_at: Any = None,
        review_every_days: Any = None,
        status: str = "active",
        origin: str = "user",
        proposed_by: str = "",
        rationale: Any = "",
        actor: str = "",
        supersedes_id: str = "",
        metric: Any = None,
    ) -> dict[str, Any]:
        """A new goal. ``status`` is ``active`` (a person writing their own) or
        ``proposed`` (Phase 3: an agent or the deck suggesting one)."""
        if scope not in SCOPES:
            raise CourseError(f"Unknown scope {scope!r}")
        if horizon not in HORIZONS:
            raise CourseError(f"Unknown horizon {horizon!r}")
        if status not in ("active", "proposed"):
            raise CourseError("A new goal is active or proposed")
        if origin not in ORIGINS:
            raise CourseError(f"Unknown origin {origin!r}")
        if scope == "instance":
            owner_id, agent_ref = "", ""
        else:
            owner_id = norm_user(owner_id)
            agent_ref = str(agent_ref or "").strip() if scope == "agent" else ""
            if scope == "agent" and not agent_ref:
                raise CourseError("An agent goal needs its agent")
            if len(agent_ref) > _REF_MAX:
                raise CourseError("Unknown agent")
        settings = self.settings()
        if horizon not in settings["horizons"][scope]:
            raise CourseError(
                f"{HORIZON_LABELS[horizon]} goals aren't used at the {scope} level on this deck")
        clean = {
            "title": _text("title", title, required=True),
            "detail": _text("detail", detail),
            "success": _text("success", success),
            "rationale": _text("rationale", rationale),
        }
        if horizon != "strategic" and not clean["success"]:
            raise CourseError("Say how you'll know it's done (success)")
        parent_id = str(parent_id or "").strip()
        days = self._review_days(review_every_days, horizon)
        due = self._due_at(due_at)
        measure = check_metric(metric, scope)
        now = _utcnow()
        goal = {"id": "goal_" + secrets.token_hex(8), "scope": scope, "owner_id": owner_id,
                "agent_ref": agent_ref, "horizon": horizon}
        with self._lock:
            conn = self._c()
            if parent_id:
                self._check_parent(conn, goal, parent_id)
            if status == "active":
                self._check_cap(conn, (scope, owner_id, agent_ref), horizon)
            conn.execute(
                """
                INSERT INTO course_goals
                    (id, scope, owner_id, agent_ref, horizon, parent_id, title, detail, success,
                     status, origin, proposed_by, rationale, review_every_days, review_at,
                     due_at, created_at, updated_at, decided_at, decided_by, supersedes_id,
                     metric)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (goal["id"], scope, owner_id, agent_ref, horizon, parent_id, clean["title"],
                 clean["detail"], clean["success"], status, origin, proposed_by,
                 clean["rationale"], days,
                 _iso(now + timedelta(days=days)) if status == "active" else None,
                 due, _iso(now), _iso(now),
                 _iso(now) if status == "active" else None,
                 actor if status == "active" else "", str(supersedes_id or ""),
                 json.dumps(measure)),
            )
            self._event(conn, goal["id"], actor, "created" if status == "active" else "proposed",
                        {"horizon": horizon, "parent_id": parent_id})
            conn.commit()
        return self.get(goal["id"])  # type: ignore[return-value]

    _EDITABLE = ("title", "detail", "success", "horizon", "parent_id", "due_at",
                 "review_every_days", "metric")

    def update(self, goal_id: str, fields: dict[str, Any], *, actor: str = "") -> dict[str, Any]:
        """Edit a goal in place (a person editing their own). Only the keys
        present in ``fields`` change; unknown keys are refused."""
        if not isinstance(fields, dict):
            raise CourseError("Expected an object of fields")
        unknown = [k for k in fields if k not in self._EDITABLE]
        if unknown:
            raise CourseError(f"{unknown[0]} can't be edited")
        with self._lock:
            conn = self._c()
            goal = self._row(conn.execute(
                "SELECT * FROM course_goals WHERE id = ?", (goal_id,)).fetchone())
            if goal is None:
                raise CourseNotFoundError(goal_id)
            if goal["status"] in ("superseded", "declined"):
                raise CourseError(f"A {goal['status']} goal can't be edited")
            new = dict(goal)
            for field in ("title", "detail", "success"):
                if field in fields:
                    new[field] = _text(field, fields[field], required=(field == "title"))
            if "horizon" in fields:
                h = fields["horizon"]
                if h not in HORIZONS:
                    raise CourseError(f"Unknown horizon {h!r}")
                if h != goal["horizon"] and h not in self.settings()["horizons"][goal["scope"]]:
                    raise CourseError(
                        f"{HORIZON_LABELS[h]} goals aren't used at the {goal['scope']} level "
                        "on this deck")
                new["horizon"] = h
            if "parent_id" in fields:
                new["parent_id"] = str(fields["parent_id"] or "").strip()
            if "due_at" in fields:
                new["due_at"] = self._due_at(fields["due_at"])
            if "review_every_days" in fields:
                new["review_every_days"] = self._review_days(
                    fields["review_every_days"], new["horizon"])
            if "metric" in fields:
                new["metric"] = check_metric(fields["metric"], goal["scope"], goal["metric"])
            if new["horizon"] != "strategic" and not new["success"]:
                raise CourseError("Say how you'll know it's done (success)")
            if new["parent_id"] and (new["parent_id"] != goal["parent_id"]
                                     or new["horizon"] != goal["horizon"]):
                self._check_parent(conn, new, new["parent_id"])
            if new["horizon"] != goal["horizon"]:
                # Children must not end up serving a shorter horizon than theirs.
                for child in conn.execute(
                        "SELECT horizon FROM course_goals WHERE parent_id = ?"
                        " AND status NOT IN ('superseded','declined')", (goal_id,)).fetchall():
                    if HORIZON_RANK[new["horizon"]] > HORIZON_RANK[child["horizon"]]:
                        raise CourseError(
                            "Goals that serve this one have a longer horizon than "
                            f"{HORIZON_LABELS[new['horizon']].lower()}; move them first")
                if goal["status"] == "active":
                    self._check_cap(conn, scope_key(goal), new["horizon"], exclude_id=goal_id)
            changed = [f for f in self._EDITABLE if new.get(f) != goal.get(f)]
            if not changed:
                return goal
            review_at = goal["review_at"]
            if "review_every_days" in changed and goal["status"] == "active":
                review_at = _iso(_utcnow() + timedelta(days=new["review_every_days"]))
            conn.execute(
                """
                UPDATE course_goals SET title = ?, detail = ?, success = ?, horizon = ?,
                    parent_id = ?, due_at = ?, review_every_days = ?, review_at = ?,
                    metric = ?, updated_at = ?
                WHERE id = ?
                """,
                (new["title"], new["detail"], new["success"], new["horizon"], new["parent_id"],
                 new["due_at"], new["review_every_days"], review_at, json.dumps(new["metric"]),
                 _iso(_utcnow()), goal_id),
            )
            self._event(conn, goal_id, actor, "edited", {"fields": changed})
            conn.commit()
        return self.get(goal_id)  # type: ignore[return-value]

    # ── proposals (Phase 3) ─────────────────────────────────────────────

    def _proposal_rules(self, conn: sqlite3.Connection, *, scope: str, owner_id: str,
                        agent_ref: str, title: str, origin: str, proposed_by: str,
                        supersedes_id: str = "") -> None:
        """Raise unless this suggestion may be made now: suggestions on, under the
        proposer's weekly limit, room among the waiting ones, not a title already
        waiting or on the course, not one declined in the last 90 days."""
        cfg = self.settings()["proposals"]
        if not cfg["recommendations"]:
            raise CourseError("Goal suggestions are switched off on this deck")
        now = _utcnow()
        week = _iso(now - timedelta(days=7))
        limit = cfg["agent_per_week" if origin == "agent" else "deck_per_week"]
        made = int(conn.execute(
            "SELECT COUNT(*) FROM course_goals WHERE origin = ? AND proposed_by = ? AND owner_id = ?"
            " AND created_at >= ?", (origin, proposed_by, owner_id, week)).fetchone()[0])
        if made >= limit:
            raise CourseError(f"The weekly limit of {limit} suggestion{'s' if limit != 1 else ''} "
                              "is reached; try again next week")
        key = (scope, owner_id, agent_ref)
        rows = [self._row(r) for r in conn.execute(
            "SELECT * FROM course_goals WHERE scope = ? AND owner_id = ? AND agent_ref = ?", key)]
        if sum(1 for r in rows if r["status"] == "proposed") >= WAITING_MAX:
            raise CourseError(f"{WAITING_MAX} suggestions are already waiting here; "
                              "they need an answer first")
        want = norm_title(title)
        cutoff = _iso(now - timedelta(days=DECLINE_MEMORY_DAYS))
        for r in rows:
            if norm_title(r["title"]) != want or r["id"] == supersedes_id:
                continue
            if r["status"] == "proposed":
                raise CourseError("The same goal is already waiting for an answer")
            if r["status"] in ("active", "paused") and not supersedes_id:
                raise CourseError("That goal is already on the course")
            if r["status"] == "declined" and (r["decided_at"] or "") >= cutoff:
                raise CourseError("That goal was declined recently; it isn't suggested again "
                                  f"within {DECLINE_MEMORY_DAYS} days")

    def propose(self, *, scope: str, horizon: str, title: Any, owner_id: str, origin: str,
                proposed_by: str, agent_ref: str = "", success: Any = "", detail: Any = "",
                parent_id: Any = "", due_at: Any = None, rationale: Any = "") -> dict[str, Any]:
        """A goal an agent (``origin="agent"``, ``proposed_by`` its agent_ref) or the
        deck (``origin="deck"``) suggests: stored ``proposed``, waiting for the
        scope's owner. Instance goals are an admin's alone."""
        if origin not in PROPOSAL_ORIGINS:
            raise CourseError(f"Unknown origin {origin!r}")
        if scope not in ("user", "agent"):
            raise CourseError("Only a person sets the deck's goals")
        owner_id = norm_user(owner_id)
        agent_ref = str(agent_ref or "").strip() if scope == "agent" else ""
        with self._lock:
            conn = self._c()
            self._proposal_rules(conn, scope=scope, owner_id=owner_id, agent_ref=agent_ref,
                                 title=title, origin=origin, proposed_by=proposed_by)
            return self.create(scope=scope, horizon=horizon, title=title, owner_id=owner_id,
                               agent_ref=agent_ref, detail=detail, success=success,
                               parent_id=parent_id, due_at=due_at, status="proposed",
                               origin=origin, proposed_by=proposed_by, rationale=rationale,
                               actor=f"{origin}:{proposed_by}")

    def propose_change(self, goal_id: str, changes: Any, *, origin: str, proposed_by: str,
                       rationale: Any = "") -> dict[str, Any]:
        """A changed version of a user or agent goal, waiting for its owner: a new
        ``proposed`` row with ``supersedes_id``. Accepting it replaces the goal."""
        if origin not in PROPOSAL_ORIGINS:
            raise CourseError(f"Unknown origin {origin!r}")
        if not isinstance(changes, dict) or not changes:
            raise CourseError("Say what should change")
        unknown = [k for k in changes if k not in CHANGEABLE]
        if unknown:
            raise CourseError(f"{unknown[0]} can't be changed")
        with self._lock:
            conn = self._c()
            goal = self._row(conn.execute("SELECT * FROM course_goals WHERE id = ?",
                                          (goal_id,)).fetchone())
            if goal is None:
                raise CourseNotFoundError(goal_id)
            if goal["scope"] == "instance":
                raise CourseError("Only an admin changes the deck's goals")
            if goal["status"] not in ("active", "paused"):
                raise CourseError(f"A {goal['status']} goal can't be changed")
            if conn.execute("SELECT 1 FROM course_goals WHERE supersedes_id = ? AND status = 'proposed'",
                            (goal_id,)).fetchone():
                raise CourseError("A change to this goal is already waiting for an answer")
            new = {k: goal[k] for k in CHANGEABLE}
            new.update(changes)
            if all(new[k] == goal[k] for k in CHANGEABLE):
                raise CourseError("That changes nothing")
            if new["horizon"] != goal["horizon"]:
                for child in conn.execute(
                        "SELECT horizon FROM course_goals WHERE parent_id = ?"
                        " AND status NOT IN ('superseded','declined')", (goal_id,)).fetchall():
                    if new["horizon"] in HORIZON_RANK and \
                            HORIZON_RANK[new["horizon"]] > HORIZON_RANK[child["horizon"]]:
                        raise CourseError("Goals that serve this one have a longer horizon")
            self._proposal_rules(conn, scope=goal["scope"], owner_id=goal["owner_id"],
                                 agent_ref=goal["agent_ref"], title=new["title"], origin=origin,
                                 proposed_by=proposed_by, supersedes_id=goal_id)
            if new["parent_id"] == goal_id:
                raise CourseError("A goal can't serve itself, directly or through others")
            return self.create(scope=goal["scope"], horizon=new["horizon"], title=new["title"],
                               owner_id=goal["owner_id"], agent_ref=goal["agent_ref"],
                               detail=new["detail"], success=new["success"],
                               parent_id=new["parent_id"], due_at=new["due_at"],
                               review_every_days=goal["review_every_days"], status="proposed",
                               origin=origin, proposed_by=proposed_by, rationale=rationale,
                               actor=f"{origin}:{proposed_by}", supersedes_id=goal_id)

    def accept(self, goal_id: str, *, actor: str = "") -> dict[str, Any]:
        """A proposal becomes an active goal. A change replaces the goal it
        changes: that one is ``superseded``, and its children, evidence and
        progress move to the new one."""
        with self._lock:
            conn = self._c()
            goal = self._row(conn.execute("SELECT * FROM course_goals WHERE id = ?",
                                          (goal_id,)).fetchone())
            if goal is None:
                raise CourseNotFoundError(goal_id)
            if goal["status"] != "proposed":
                raise CourseError(f"This goal is {goal['status']}, not a proposal")
            if goal["horizon"] not in self.settings()["horizons"][goal["scope"]]:
                raise CourseError(f"{HORIZON_LABELS[goal['horizon']]} goals aren't used at the "
                                  f"{goal['scope']} level any more; change its horizon first")
            old = None
            if goal["supersedes_id"]:
                old = self._row(conn.execute("SELECT * FROM course_goals WHERE id = ?",
                                             (goal["supersedes_id"],)).fetchone())
                if old is None or old["status"] not in ("active", "paused"):
                    raise CourseError("The goal this would change is no longer active; decline "
                                      "this and add it again if you still want it")
            self._check_cap(conn, scope_key(goal), goal["horizon"],
                            exclude_id=old["id"] if old else "")
            now = _utcnow()
            conn.execute(
                "UPDATE course_goals SET status = 'active', review_at = ?, updated_at = ?,"
                " decided_at = ?, decided_by = ?, progress = ?, progress_note = ?, metric = ?"
                " WHERE id = ?",
                (_iso(now + timedelta(days=int(goal["review_every_days"]))), _iso(now), _iso(now),
                 actor, old["progress"] if old else goal["progress"],
                 old["progress_note"] if old else goal["progress_note"],
                 json.dumps(goal["metric"] or (old["metric"] if old else {})), goal_id))
            self._event(conn, goal_id, actor, "accepted",
                        {"supersedes_id": old["id"]} if old else {})
            if old is not None:
                conn.execute(
                    "UPDATE course_goals SET status = 'superseded', updated_at = ?, decided_at = ?,"
                    " decided_by = ? WHERE id = ?", (_iso(now), _iso(now), actor, old["id"]))
                self._event(conn, old["id"], actor, "status",
                            {"from": old["status"], "to": "superseded", "by": goal_id})
                rank = HORIZON_RANK[goal["horizon"]]
                for child in conn.execute(
                        "SELECT id, horizon FROM course_goals WHERE parent_id = ?"
                        " AND status NOT IN ('superseded','declined')", (old["id"],)).fetchall():
                    if HORIZON_RANK[child["horizon"]] >= rank:
                        conn.execute("UPDATE course_goals SET parent_id = ? WHERE id = ?",
                                     (goal_id, child["id"]))
                conn.execute("UPDATE OR IGNORE course_links SET goal_id = ? WHERE goal_id = ?",
                             (goal_id, old["id"]))
                if old["status"] == "paused":
                    conn.execute("UPDATE course_goals SET status = 'paused' WHERE id = ?",
                                 (goal_id,))
            conn.commit()
        return self.get(goal_id)  # type: ignore[return-value]

    def set_status(self, goal_id: str, status: str, *, actor: str = "") -> dict[str, Any]:
        """Move a goal along its lifecycle. Becoming active again checks the cap
        and restarts its review clock."""
        if status not in STATUSES:
            raise CourseError(f"Unknown status {status!r}")
        with self._lock:
            conn = self._c()
            goal = self._row(conn.execute(
                "SELECT * FROM course_goals WHERE id = ?", (goal_id,)).fetchone())
            if goal is None:
                raise CourseNotFoundError(goal_id)
            if status == goal["status"]:
                return goal
            if status not in _TRANSITIONS[goal["status"]]:
                raise CourseError(f"A {goal['status']} goal can't become {status}")
            now = _utcnow()
            review_at = goal["review_at"]
            if status == "active":
                if goal["horizon"] not in self.settings()["horizons"][goal["scope"]]:
                    raise CourseError(
                        f"{HORIZON_LABELS[goal['horizon']]} goals aren't used at the "
                        f"{goal['scope']} level any more; change its horizon first")
                self._check_cap(conn, scope_key(goal), goal["horizon"], exclude_id=goal_id)
                review_at = _iso(now + timedelta(days=int(goal["review_every_days"])))
            conn.execute(
                "UPDATE course_goals SET status = ?, review_at = ?, updated_at = ?,"
                " decided_at = ?, decided_by = ? WHERE id = ?",
                (status, review_at, _iso(now), _iso(now), actor, goal_id),
            )
            self._event(conn, goal_id, actor, "status", {"from": goal["status"], "to": status})
            conn.commit()
        return self.get(goal_id)  # type: ignore[return-value]

    def review(self, goal_id: str, *, actor: str = "") -> dict[str, Any]:
        """"Still on course": restart the goal's review clock."""
        with self._lock:
            conn = self._c()
            goal = self._row(conn.execute(
                "SELECT * FROM course_goals WHERE id = ?", (goal_id,)).fetchone())
            if goal is None:
                raise CourseNotFoundError(goal_id)
            if goal["status"] != "active":
                raise CourseError("Only an active goal is reviewed")
            now = _utcnow()
            conn.execute(
                "UPDATE course_goals SET review_at = ?, updated_at = ? WHERE id = ?",
                (_iso(now + timedelta(days=int(goal["review_every_days"]))), _iso(now), goal_id),
            )
            self._event(conn, goal_id, actor, "reviewed")
            conn.commit()
        return self.get(goal_id)  # type: ignore[return-value]

    def add_link(self, goal_id: str, kind: str, ref_id: Any, *, note: Any = "",
                 actor: str = "", owner_id: str = "") -> dict[str, Any]:
        if kind not in LINK_KINDS:
            raise CourseError(f"Unknown link kind {kind!r}")
        ref = str(ref_id or "").strip()
        if not ref or len(ref) > _REF_MAX:
            raise CourseError("ref_id is required")
        note = _text("note", note)
        with self._lock:
            conn = self._c()
            if conn.execute("SELECT 1 FROM course_goals WHERE id = ?", (goal_id,)).fetchone() is None:
                raise CourseNotFoundError(goal_id)
            now = _iso(_utcnow())
            owner = norm_user(owner_id)
            conn.execute(
                "INSERT INTO course_links (goal_id, kind, ref_id, note, added_by, at, owner_id)"
                " VALUES (?, ?, ?, ?, ?, ?, ?) ON CONFLICT(goal_id, kind, ref_id)"
                " DO UPDATE SET note = excluded.note, at = excluded.at",
                (goal_id, kind, ref, note, actor, now, owner),
            )
            self._event(conn, goal_id, actor, "linked",
                        {"kind": kind, "ref_id": ref, "owner_id": owner})
            conn.commit()
        return {"goal_id": goal_id, "kind": kind, "ref_id": ref, "note": note,
                "added_by": actor, "at": now, "owner_id": owner}

    def set_progress(self, goal_id: str, progress: Any, note: Any = "", *,
                     actor: str = "") -> dict[str, Any]:
        """An estimate of how far along a goal is (0..1), with a note. Never
        sets ``achieved`` — only a person does."""
        if isinstance(progress, bool) or not isinstance(progress, (int, float)):
            raise CourseError("progress must be a number from 0 to 1")
        value = float(progress)
        if not 0.0 <= value <= 1.0:
            raise CourseError("progress must be a number from 0 to 1")
        note = _text("progress_note", note)
        with self._lock:
            conn = self._c()
            goal = self._row(conn.execute(
                "SELECT * FROM course_goals WHERE id = ?", (goal_id,)).fetchone())
            if goal is None:
                raise CourseNotFoundError(goal_id)
            if goal["status"] not in ("active", "paused"):
                raise CourseError(f"A {goal['status']} goal takes no progress report")
            conn.execute(
                "UPDATE course_goals SET progress = ?, progress_note = ?, updated_at = ? WHERE id = ?",
                (value, note, _iso(_utcnow()), goal_id))
            self._event(conn, goal_id, actor, "progress", {"progress": value, "note": note})
            conn.commit()
        return self.get(goal_id)  # type: ignore[return-value]

    def set_metric_value(self, goal_id: str, value: Any, *, actor: str = "") -> dict[str, Any]:
        """A person enters the current value of a ``manual`` metric."""
        number = _number("value", value)
        with self._lock:
            conn = self._c()
            goal = self._row(conn.execute(
                "SELECT * FROM course_goals WHERE id = ?", (goal_id,)).fetchone())
            if goal is None:
                raise CourseNotFoundError(goal_id)
            if (goal["metric"] or {}).get("kind") != "manual":
                raise CourseError("This goal isn't measured by a number you enter")
            if goal["status"] not in ("active", "paused"):
                raise CourseError(f"A {goal['status']} goal takes no new value")
            metric = {**goal["metric"], "value": number, "value_at": _iso(_utcnow())}
            conn.execute("UPDATE course_goals SET metric = ?, updated_at = ? WHERE id = ?",
                         (json.dumps(metric), _iso(_utcnow()), goal_id))
            self._event(conn, goal_id, actor, "measured", {"value": number})
            conn.commit()
        return self.get(goal_id)  # type: ignore[return-value]

    def link_count(self, goal_id: str, *, owner_id: str | None = None) -> int:
        sql, args = "SELECT COUNT(*) FROM course_links WHERE goal_id = ?", [goal_id]
        if owner_id is not None:
            sql += " AND owner_id = ?"
            args.append(norm_user(owner_id))
        with self._lock:
            return int(self._c().execute(sql, args).fetchone()[0])

    # ── the weekly estimate (Phase 4) ──────────────────────────────────

    def to_estimate(self, owner_id: str, *, now: datetime | None = None,
                    limit: int = 3) -> list[dict[str, Any]]:
        """``owner_id``'s active goals (their own and their agents') with
        evidence newer than their last estimate, and no estimate in the last
        week — the oldest estimate first. A deck-wide goal is everyone's: no
        one user's work estimates it."""
        now = now or _utcnow()
        week = _iso(now - timedelta(days=ESTIMATE_EVERY_DAYS))
        with self._lock:
            rows = self._c().execute(
                """
                SELECT g.* FROM course_goals g
                WHERE g.scope IN ('user', 'agent') AND g.owner_id = ? AND g.status = 'active'
                  AND (g.estimated_at IS NULL OR g.estimated_at <= ?)
                  AND EXISTS (SELECT 1 FROM course_links l WHERE l.goal_id = g.id
                              AND l.at > COALESCE(g.estimated_at, ''))
                ORDER BY COALESCE(g.estimated_at, '') LIMIT ?
                """, (norm_user(owner_id), week, max(1, int(limit)))).fetchall()
        return [self._row(r) for r in rows]  # type: ignore[misc]

    def mark_estimated(self, goal_id: str, *, now: datetime | None = None) -> None:
        with self._lock:
            conn = self._c()
            conn.execute("UPDATE course_goals SET estimated_at = ? WHERE id = ?",
                         (_iso(now or _utcnow()), goal_id))
            conn.commit()

    # ── conflicts (Phase 4) ────────────────────────────────────────────

    def add_conflict(self, goal_id: str, other_id: str, reason: Any = "") -> dict[str, Any] | None:
        """Flag two goals as likely pulling against each other. The narrower
        goal's owner is told (the admins for two deck-wide goals); a pair
        flagged before — answered or not — is never flagged again. None when
        nothing new was flagged."""
        reason = " ".join(str(reason or "").split())[:TEXT_CAPS["reason"]]
        if not goal_id or not other_id or goal_id == other_id:
            return None
        with self._lock:
            conn = self._c()
            a = self._row(conn.execute("SELECT * FROM course_goals WHERE id = ?", (goal_id,)).fetchone())
            b = self._row(conn.execute("SELECT * FROM course_goals WHERE id = ?", (other_id,)).fetchone())
            if a is None or b is None:
                return None
            if any(g["status"] not in ("active", "paused") for g in (a, b)):
                return None
            if conn.execute(
                    "SELECT 1 FROM course_conflicts WHERE (goal_id = ? AND other_id = ?)"
                    " OR (goal_id = ? AND other_id = ?)", (a["id"], b["id"], b["id"], a["id"])).fetchone():
                return None
            if _SCOPE_RANK[b["scope"]] > _SCOPE_RANK[a["scope"]]:
                a, b = b, a                       # goal_id is the narrower one
            if a["scope"] != "instance" and b["scope"] != "instance" and a["owner_id"] != b["owner_id"]:
                return None                       # never across two people's courses
            row = {"id": "cf_" + secrets.token_hex(8), "goal_id": a["id"], "other_id": b["id"],
                   "owner_id": "" if a["scope"] == "instance" else a["owner_id"],
                   "reason": reason, "status": "open", "at": _iso(_utcnow())}
            conn.execute(
                "INSERT INTO course_conflicts (id, goal_id, other_id, owner_id, reason, status, at)"
                " VALUES (:id, :goal_id, :other_id, :owner_id, :reason, :status, :at)", row)
            self._event(conn, a["id"], "deck:conflict", "conflict",
                        {"other_id": b["id"], "conflict_id": row["id"]})
            if b["scope"] != "instance" or a["scope"] == "instance":
                # A deck-wide goal's history is everyone's: it never names a user's goal.
                self._event(conn, b["id"], "deck:conflict", "conflict",
                            {"other_id": a["id"], "conflict_id": row["id"]})
            conn.commit()
        return {**row, "decided_at": None, "decided_by": ""}

    def get_conflict(self, conflict_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._c().execute("SELECT * FROM course_conflicts WHERE id = ?",
                                    (str(conflict_id or ""),)).fetchone()
        return dict(row) if row else None

    def conflicts(self, owner_id: str, *, admin: bool = False,
                  status: str = "open") -> list[dict[str, Any]]:
        """The conflicts ``owner_id`` is told about (an admin also those between
        deck-wide goals), while both goals are still active or paused."""
        sql = ("SELECT c.* FROM course_conflicts c"
               " JOIN course_goals a ON a.id = c.goal_id JOIN course_goals b ON b.id = c.other_id"
               " WHERE c.status = ? AND a.status IN ('active','paused')"
               " AND b.status IN ('active','paused') AND (c.owner_id = ?")
        args: list[Any] = [status, norm_user(owner_id)]
        sql += " OR c.owner_id = '')" if admin else ")"
        with self._lock:
            return [dict(r) for r in self._c().execute(sql + " ORDER BY c.at DESC", args).fetchall()]

    def conflicts_of(self, goal_id: str) -> list[dict[str, Any]]:
        """Every open conflict a goal is part of, whoever was told."""
        with self._lock:
            return [dict(r) for r in self._c().execute(
                "SELECT * FROM course_conflicts WHERE status = 'open' AND (goal_id = ? OR other_id = ?)",
                (goal_id, goal_id)).fetchall()]

    def dismiss_conflict(self, conflict_id: str, *, actor: str = "") -> dict[str, Any]:
        """"Not a conflict": the pair is never flagged again."""
        with self._lock:
            conn = self._c()
            row = conn.execute("SELECT * FROM course_conflicts WHERE id = ?",
                               (conflict_id,)).fetchone()
            if row is None:
                raise CourseNotFoundError(conflict_id)
            if row["status"] != "open":
                return dict(row)
            now = _iso(_utcnow())
            conn.execute("UPDATE course_conflicts SET status = 'dismissed', decided_at = ?,"
                         " decided_by = ? WHERE id = ?", (now, actor, conflict_id))
            conn.commit()
        return {**dict(row), "status": "dismissed", "decided_at": now, "decided_by": actor}

    def last_moved(self, goal_ids: list[str], *, owner_id: str | None = None) -> dict[str, str]:
        """The newest evidence time per goal (only ``owner_id``'s links when given)."""
        if not goal_ids:
            return {}
        sql = (f"SELECT goal_id, MAX(at) AS at FROM course_links"
               f" WHERE goal_id IN ({','.join('?' * len(goal_ids))})")
        args: list[Any] = list(goal_ids)
        if owner_id is not None:
            sql += " AND owner_id = ?"
            args.append(norm_user(owner_id))
        with self._lock:
            return {r["goal_id"]: r["at"] for r in
                    self._c().execute(sql + " GROUP BY goal_id", args).fetchall()}

    def remove_link(self, goal_id: str, kind: str, ref_id: str, *,
                    owner_id: str | None = None) -> bool:
        """Remove a link (only ``owner_id``'s when given)."""
        sql = "DELETE FROM course_links WHERE goal_id = ? AND kind = ? AND ref_id = ?"
        args: list[Any] = [goal_id, kind, ref_id]
        if owner_id is not None:
            sql += " AND owner_id = ?"
            args.append(norm_user(owner_id))
        with self._lock:
            conn = self._c()
            cur = conn.execute(sql, args)
            conn.commit()
            return cur.rowcount > 0

    # ── forgetting ─────────────────────────────────────────────────────

    def forget_user(self, user_id: str) -> int:
        """A deleted user: their user goals and their agents' goals, with their
        links and history. How many goals went."""
        uid = norm_user(user_id)
        with self._lock:
            conn = self._c()
            conn.execute("DELETE FROM course_conflicts WHERE owner_id = ?", (uid,))
            cur = conn.execute(
                "DELETE FROM course_goals WHERE scope IN ('user', 'agent') AND owner_id = ?",
                (uid,))
            conn.commit()
            return cur.rowcount

    def forget_agent(self, agent_ref: str) -> int:
        """A removed agent: its own goals (nothing else can serve them)."""
        ref = str(agent_ref or "").strip()
        if not ref:
            return 0
        with self._lock:
            conn = self._c()
            cur = conn.execute(
                "DELETE FROM course_goals WHERE scope = 'agent' AND agent_ref = ?", (ref,))
            conn.commit()
            return cur.rowcount


_STORE: CourseStore | None = None
_STORE_LOCK = threading.Lock()


def get_store() -> CourseStore:
    global _STORE
    if _STORE is not None:
        return _STORE
    with _STORE_LOCK:
        if _STORE is None:
            _STORE = CourseStore()
        return _STORE


def forget_user(user_id: str) -> int:
    """For ``admin_routes.delete_user``. Never raises."""
    try:
        return get_store().forget_user(user_id)
    except Exception:  # noqa: BLE001 — a deletion never fails over Course
        return 0


def forget_agent(agent_ref: str) -> int:
    """For ``server._forget_shared_agent``. Never raises."""
    try:
        return get_store().forget_agent(agent_ref)
    except Exception:  # noqa: BLE001 — a removal never fails over Course
        return 0


# ── For the rest of the deck (Phase 2: the Arbiter, the Taskboard, plans) ──

_SHORT = {"strategic": "strategic", "long": "long-term", "medium": "medium-term",
          "short": "short-term"}


def live_goals(user_id: str, *, agent_ref: str = "", include_stale: bool = False,
               now: datetime | None = None) -> list[dict[str, Any]]:
    """The goals that steer work for ``user_id`` right now: the deck's and the
    user's own (and, with ``agent_ref``, that agent's) — active and not stale
    (stale ones too with ``include_stale``). Never raises ([] on any error)."""
    try:
        store = get_store()
        uid = norm_user(user_id)
        now = now or _utcnow()
        goals = store.list_goals(scope="instance", statuses=("active",))
        goals += store.list_goals(scope="user", owner_id=uid, statuses=("active",))
        if agent_ref:
            goals += store.list_goals(scope="agent", owner_id=uid, agent_refs=[agent_ref],
                                      statuses=("active",))
        return [g for g in goals if include_stale or not is_stale(g, now)]
    except Exception:  # noqa: BLE001 — Course never breaks its callers
        return []


def prompt_lines(goals: list[dict[str, Any]], *, max_chars: int = 1500) -> str:
    """The goals as one line each, ``- G:<id> (<horizon>): <title> — done when: <success>``,
    for a deck-internal model prompt (the Arbiter, Taskboard triage). People's
    text is flattened onto its line; lines past ``max_chars`` are left out."""
    lines: list[str] = []
    used = 0
    for g in goals:
        title = " ".join(str(g.get("title") or "").split())[:120]
        line = f"- G:{g['id']} ({_SHORT.get(g.get('horizon', ''), g.get('horizon', ''))}): {title}"
        success = " ".join(str(g.get("success") or "").split())[:160]
        if success:
            line += f" — done when: {success}"
        if used + len(line) + 1 > max_chars:
            break
        lines.append(line)
        used += len(line) + 1
    return "\n".join(lines)


def goal_ref(value: Any) -> str:
    """A model's ``goal_id`` / ``serves_goal`` answer as a bare goal id ("" for none)."""
    text = str(value or "").strip().removeprefix("G:").strip()
    if not text or text.lower() in ("none", "null", "-"):
        return ""
    return text[:40]


def usable_goal(goal_id: str, user_id: str, *, agent_ref: str = "",
                include_stale: bool = False) -> dict[str, Any] | None:
    """``goal_id`` when it is one of ``user_id``'s live goals (see :func:`live_goals`),
    else None — so a model can only ever point at a goal it was shown."""
    gid = goal_ref(goal_id)
    if not gid:
        return None
    for g in live_goals(user_id, agent_ref=agent_ref, include_stale=include_stale):
        if g["id"] == gid:
            return g
    return None


def link_quietly(goal_id: str, kind: str, ref_id: str, *, note: str = "",
                 actor: str = "", owner_id: str = "") -> bool:
    """Record evidence for a goal — ``owner_id``'s work; False (never an
    exception) when the goal is gone or the link can't be written. The note is
    clipped to its cap."""
    gid = goal_ref(goal_id)
    if not gid or not ref_id:
        return False
    try:
        get_store().add_link(gid, kind, str(ref_id)[:_REF_MAX],
                             note=" ".join(str(note or "").split())[:TEXT_CAPS["note"]],
                             actor=actor, owner_id=owner_id)
        return True
    except (CourseError, CourseNotFoundError):
        return False
    except Exception:  # noqa: BLE001 — evidence never breaks the work it records
        return False


def _clip_title(text: str) -> str:
    """A long text cut to a goal title's cap, at a word where it can."""
    text = " ".join(str(text or "").split())
    cap = TEXT_CAPS["title"]
    if len(text) <= cap:
        return text
    cut = text[: cap - 1].rsplit(" ", 1)[0]
    return (cut if len(cut) > cap // 2 else text[: cap - 1]).rstrip(" ,.;:") + "…"


def goal_title(goal_id: str) -> str:
    """A goal's title ("" when there is none)."""
    try:
        g = get_store().get(goal_ref(goal_id))
    except Exception:  # noqa: BLE001
        return ""
    return str(g["title"]) if g else ""



# ── Progress and off course (Phase 4) ──────────────────────────────────


def automation() -> dict[str, bool]:
    """The admin's switches for what Flight Deck does on its own (the
    defaults on any error)."""
    try:
        return get_store().settings()["automation"]
    except Exception:  # noqa: BLE001
        return dict(DEFAULT_AUTOMATION)


def _tasks_done(goal: dict[str, Any]) -> int | None:
    since = _parse_iso(goal.get("decided_at") or goal.get("created_at"))
    try:
        from captain_claw.flight_deck import taskboard_store

        return taskboard_store.get_store().count_goal_done(
            norm_user(goal.get("owner_id")), goal["id"], since.timestamp() if since else 0.0)
    except Exception:  # noqa: BLE001 — no board: no count
        return None


def measure(goal: dict[str, Any]) -> dict[str, Any] | None:
    """A goal's metric as it stands: ``{kind, label, value, target, unit,
    progress}`` (``progress`` 0..1 when there is a target and a value), or None
    when the goal has no metric. Counts are the goal owner's own work."""
    m = goal.get("metric") or {}
    kind = m.get("kind")
    if kind not in METRIC_KINDS:
        return None
    value: float | int | None = None
    if kind == "manual":
        value = m.get("value")
    elif goal.get("scope") != "instance":
        if kind == "link_count":
            try:
                value = get_store().link_count(goal["id"])
            except Exception:  # noqa: BLE001
                value = None
        elif kind == "taskboard_done":
            value = _tasks_done(goal)
    target = m.get("target")
    progress = (min(1.0, max(0.0, float(value) / float(target)))
                if value is not None and target else None)
    return {"kind": kind, "label": METRIC_KIND_LABELS[kind], "value": value, "target": target,
            "unit": m.get("unit") or "", "value_at": m.get("value_at"), "progress": progress}


def off_course(goals: list[dict[str, Any]], moved: dict[str, str], *,
               now: datetime | None = None) -> dict[str, str]:
    """Which of one user's goals are off course, and why (goal id → reason).

    ``goals`` is everything the user sees (the deck's, theirs, their agents');
    ``moved`` their last-moved times. Off course:

    * an active user goal no active agent goal serves (anywhere below it), with
      no work on it or below it within its review period — judged once it has
      been active for that period or two weeks, whichever is shorter;
    * an active agent goal that serves none of the user's goals or the deck's.
    """
    now = now or _utcnow()
    by_id = {g["id"]: g for g in goals}
    children: dict[str, list[dict[str, Any]]] = {}
    for g in goals:
        if g.get("parent_id") and g["status"] in ("active", "paused"):
            children.setdefault(g["parent_id"], []).append(g)

    def below(gid: str) -> list[dict[str, Any]]:
        out, todo, seen = [], list(children.get(gid, [])), {gid}
        while todo:
            g = todo.pop()
            if g["id"] in seen:
                continue
            seen.add(g["id"])
            out.append(g)
            todo.extend(children.get(g["id"], []))
        return out

    reasons: dict[str, str] = {}
    for g in goals:
        if g["status"] != "active":
            continue
        if g["scope"] == "user":
            days = int(g.get("review_every_days") or REVIEW_DAYS[g["horizon"]])
            since = _parse_iso(g.get("decided_at") or g.get("created_at")) or now
            if now - since < timedelta(days=min(days, OFF_COURSE_MIN_DAYS)):
                continue
            under = below(g["id"])
            if any(x["scope"] == "agent" and x["status"] == "active" for x in under):
                continue
            times = [t for t in (_parse_iso(moved.get(x["id"])) for x in [g, *under]) if t]
            if times and max(times) > now - timedelta(days=days):
                continue
            reasons[g["id"]] = (f"No agent goal serves it, and no work moved it in the last "
                                f"{days} days")
        elif g["scope"] == "agent":
            cur, seen, reached = g, set(), False
            while cur is not None and cur["id"] not in seen:
                seen.add(cur["id"])
                parent = by_id.get(cur.get("parent_id") or "")
                if parent is not None and parent["status"] in ("active", "paused") \
                        and parent["scope"] in ("user", "instance"):
                    reached = True
                    break
                cur = parent if parent is not None and parent["status"] in ("active", "paused") else None
            if not reached:
                reasons[g["id"]] = "Serves none of your goals or the deck’s"
    return reasons


async def ask_looks_achieved(goal: dict[str, Any], *, source: str, trust: str, body: str,
                             key: str | None = None, to: list[str] | None = None) -> bool:
    """Ask whether a goal is achieved on the Line of ``to`` (default: the goal's
    owner) — once per ``key`` (default: today, so at most once a day whoever
    asks). It never marks the goal done. True when an item was posted. Never
    raises."""
    try:
        from captain_claw.flight_deck import line

        key = key or _utcnow().date().isoformat()
        posted = False
        for uid in to if to is not None else [norm_user(goal.get("owner_id"))]:
            item = await line.post_once(
                uid, "ask", f"Looks achieved? {' '.join(str(goal['title']).split())}"[:200],
                body + "\n\nMark it done, or keep it going, on the Course page.",
                bell_type="course_looks_achieved", needs_you=True, source=source, trust=trust,
                ref_type="course_review", ref_id=f"{goal['id']}@achieved:{key}")
            posted = posted or bool(item)
        return posted
    except Exception:  # noqa: BLE001 — a question never breaks its caller
        return False
