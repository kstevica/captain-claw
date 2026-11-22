"""Scheduled tasks pause when their owner has been away for a week (idle_pause).

On a cohort deck one owner's two daily jobs ran for sixteen days after their
last sign-in, and another's twice-a-day task wrote "no change" thirty times to
a Line nobody read.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest

from captain_claw.flight_deck import agent_cron_routes as acr
from captain_claw.flight_deck import fd_scheduler, idle_pause

NOW = datetime(2026, 11, 19, 12, 0, tzinfo=UTC)


def _iso(days_ago: float) -> str:
    return (NOW - timedelta(days=days_ago)).isoformat()


def _user(uid: str, *, login: float | None, created: float = 60) -> dict:
    return {"id": uid, "role": "user", "created_at": _iso(created),
            "last_login_at": _iso(login) if login is not None else ""}


class _Settings:
    """Flight Deck's db as idle_pause reads the admin's setting from it."""

    def __init__(self) -> None:
        self.config: dict = {}

    async def get_system_setting(self, key: str):
        return json.dumps(self.config) if key == "fd:system-config" and self.config else None


class _Db:
    def __init__(self, users: list[dict], typed: dict[str, str] | None = None) -> None:
        self.users = users
        self.typed = typed or {}

    async def list_users(self, limit: int = 100, offset: int = 0) -> list[dict]:
        return self.users

    async def message_day_bounds(self) -> dict[str, dict]:
        return {uid: {"last_day": day} for uid, day in self.typed.items()}


@pytest.fixture
def deck(tmp_path, monkeypatch):
    """Sign-in on, a scheduler store of its own, one agent per owner whose
    jobs live in a stand-in file, and a Line that records what it was told."""
    monkeypatch.setattr("captain_claw.flight_deck.auth._fd_auth_enabled", lambda: True)
    settings = _Settings()
    monkeypatch.setattr("captain_claw.flight_deck.auth.get_db", lambda: settings)
    store = fd_scheduler.SchedulerStore(tmp_path / "scheduler.db")
    monkeypatch.setattr(fd_scheduler, "get_store", lambda: store)

    agent_rows: dict[str, list[dict]] = {}
    monkeypatch.setattr(idle_pause, "_owned_agents",
                        lambda uid: {f"agent-{uid}": {"name": f"Agent {uid}", "owner": uid}})
    monkeypatch.setattr(acr, "_read_rows", lambda slug: agent_rows.get(slug, []))

    def edit(slug, job_id, action):
        for row in agent_rows.get(slug, []):
            if row["id"] == job_id and action == "pause":
                row["enabled"] = 0
                return True
        return False

    monkeypatch.setattr(acr, "_edit_file", edit)
    from captain_claw.flight_deck import server

    monkeypatch.setattr(server, "_process_is_alive", lambda slug: False)
    told: list[tuple] = []

    async def post(user_id, kind, title, body="", **kw):
        told.append((user_id, title, body, kw))
        return "item"

    from captain_claw.flight_deck import line

    monkeypatch.setattr(line, "post", post)
    return {"store": store, "agent_rows": agent_rows, "told": told, "settings": settings}


def _task(store, owner: str, name: str, enabled: bool = True, schedule: str = "daily 07:30") -> dict:
    return store.create(owner_id=owner, name=name, schedule=schedule, agent_slug=f"agent-{owner}",
                        prompt="Check the tenders", delivery_kind="agent",
                        delivery={"chat": True}, enabled=enabled)


def _agent_job(deck, owner: str, job_id: str, text: str, enabled: int = 1,
               schedule: dict | None = None) -> None:
    deck["agent_rows"].setdefault(f"agent-{owner}", []).append(
        {"id": job_id, "enabled": enabled, "kind": "prompt", "payload": json.dumps({"text": text}),
         "schedule": json.dumps(schedule or {"type": "daily", "hour": 7, "minute": 30})})


# ── who counts as away ────────────────────────────────────────────────


def test_activity_is_the_later_of_signing_in_typing_and_the_account():
    user = _user("u", login=20, created=60)
    assert idle_pause.last_activity(user) == pytest.approx((NOW - timedelta(days=20)).timestamp())
    # Typed on a day: that whole day counts.
    day = (NOW - timedelta(days=2)).date().isoformat()
    assert idle_pause.last_activity(user, day) > (NOW - timedelta(days=2)).timestamp()
    # Never signed in: the account's own age.
    assert idle_pause.last_activity(_user("n", login=None, created=3)) == pytest.approx(
        (NOW - timedelta(days=3)).timestamp())


async def test_an_away_owner_has_every_task_paused_and_is_told_once(deck):
    _task(deck["store"], "roko", "Tender watch")
    _task(deck["store"], "roko", "Old one", enabled=False)
    _agent_job(deck, "roko", "j1", "Screen carriers in Zagreb")
    db = _Db([_user("roko", login=16)])

    assert await idle_pause.sweep(db, now=NOW.timestamp()) == 2
    assert not any(j["enabled"] for j in deck["store"].list(owner_id="roko"))
    assert deck["agent_rows"]["agent-roko"][0]["enabled"] == 0
    assert len(deck["told"]) == 1
    uid, title, body, kw = deck["told"][0]
    assert uid == "roko" and title == "Scheduled tasks paused (2)"
    assert "Tender watch" in body and "Agent roko: Screen carriers in Zagreb" in body
    assert "Old one" not in body and "Scheduled tasks" in body
    assert kw["needs_you"] is True

    # Nothing left running: the next pass pauses nothing and says nothing.
    assert await idle_pause.sweep(db, now=NOW.timestamp() + 3600) == 0
    assert len(deck["told"]) == 1


@pytest.mark.parametrize("user,typed", [
    (_user("ana", login=2), {}),                                        # signed in this week
    (_user("ana", login=30), {"ana": (NOW - timedelta(days=1)).date().isoformat()}),  # typed on WhatsApp
    (_user("ana", login=None, created=3), {}),                          # a new account
])
async def test_someone_active_keeps_their_tasks(deck, user, typed):
    _task(deck["store"], "ana", "Morning brief")
    assert await idle_pause.sweep(_Db([user], typed), now=NOW.timestamp()) == 0
    assert all(j["enabled"] for j in deck["store"].list(owner_id="ana"))
    assert deck["told"] == []


async def test_it_can_be_turned_off_and_is_off_without_accounts(deck, monkeypatch):
    _task(deck["store"], "roko", "Tender watch")
    db = _Db([_user("roko", login=40)])
    deck["settings"].config = {"scheduled_idle_pause_days": 0}     # the admin turned it off
    assert await idle_pause.sweep(db, now=NOW.timestamp()) == 0
    deck["settings"].config = {}
    monkeypatch.setattr("captain_claw.flight_deck.auth._fd_auth_enabled", lambda: False)
    assert await idle_pause.sweep(db, now=NOW.timestamp()) == 0
    assert deck["store"].list(owner_id="roko")[0]["enabled"]


async def test_the_day_count_comes_from_the_deck(deck, monkeypatch):
    _task(deck["store"], "vedran", "Weekly scan")
    db = _Db([_user("vedran", login=4)])
    deck["settings"].config = {"scheduled_idle_pause_days": 3}
    assert await idle_pause.idle_days() == 3
    assert await idle_pause.sweep(db, now=NOW.timestamp()) == 1
    assert "for 4 days" in deck["told"][0][2]      # how long they have been away


async def test_a_running_agent_is_asked_through_its_own_rest(deck, monkeypatch):
    from captain_claw.flight_deck import server

    _agent_job(deck, "roko", "j9", "Daily screen")
    monkeypatch.setattr(server, "_process_is_alive", lambda slug: True)
    calls: list[tuple] = []

    async def rest(entry, method, path, name):
        calls.append((method, path))
        return {}

    monkeypatch.setattr(acr, "_agent_rest", rest)
    assert await idle_pause.sweep(_Db([_user("roko", login=10)]), now=NOW.timestamp()) == 1
    assert calls == [("POST", "/api/cron/jobs/j9/pause")]


async def test_one_owner_failing_doesnt_stop_the_others(deck, monkeypatch):
    _task(deck["store"], "a", "A's task")
    _task(deck["store"], "b", "B's task")
    real = idle_pause._pause_agent_jobs

    async def boom(uid, *args):
        if uid == "a":
            raise RuntimeError("agent file locked")
        return await real(uid, *args)

    monkeypatch.setattr(idle_pause, "_pause_agent_jobs", boom)
    db = _Db([_user("a", login=9), _user("b", login=9)])
    assert await idle_pause.sweep(db, now=NOW.timestamp()) == 1
    assert [t[0] for t in deck["told"]] == ["b"]


async def test_on_by_default_with_accounts_and_off_by_setting(deck, monkeypatch):
    assert await idle_pause.pause_days() == 7                      # nothing stored: the default
    deck["settings"].config = {"scheduled_idle_pause_days": 0}
    assert await idle_pause.pause_days() == 0
    deck["settings"].config = {"scheduled_idle_pause_days": 14}
    assert await idle_pause.pause_days() == 14
    monkeypatch.setattr("captain_claw.flight_deck.auth._fd_auth_enabled", lambda: False)
    assert await idle_pause.pause_days() == 0                      # no accounts to be away from


async def test_an_unreadable_setting_pauses_nothing(deck, monkeypatch):
    class _Broken:
        async def get_system_setting(self, key):
            raise RuntimeError("database is locked")

    monkeypatch.setattr("captain_claw.flight_deck.auth.get_db", lambda: _Broken())
    _task(deck["store"], "roko", "Tender watch")
    assert await idle_pause.sweep(_Db([_user("roko", login=40)]), now=NOW.timestamp()) == 0
    assert deck["store"].list(owner_id="roko")[0]["enabled"]


async def test_the_admin_setting_is_bounded_and_saved(monkeypatch):
    from fastapi import HTTPException

    from captain_claw.flight_deck import admin_routes

    assert admin_routes.SYSTEM_CONFIG_DEFAULTS["scheduled_idle_pause_days"] == 7

    class _DB:
        stored = None

        async def get_system_setting(self, key):
            return self.stored

        async def set_system_setting(self, key, value):
            self.stored = value

    db = _DB()
    monkeypatch.setattr(admin_routes, "get_db", lambda: db)
    req = admin_routes.UpdateConfigRequest
    for bad in (-1, idle_pause.MAX_DAYS + 1):
        with pytest.raises(HTTPException):
            await admin_routes.update_config(req(scheduled_idle_pause_days=bad), admin={})
    out = await admin_routes.update_config(req(scheduled_idle_pause_days=14), admin={})
    assert out["scheduled_idle_pause_days"] == 14
    assert json.loads(db.stored) == {"scheduled_idle_pause_days": 14}
    off = await admin_routes.update_config(req(scheduled_idle_pause_days=0), admin={})
    assert off["scheduled_idle_pause_days"] == 0


# ── A job that runs less often waits longer (cohort001, 2026-11-20) ──────────
#
# With the setting at 3 days, a weekly Monday scan was paused on the Thursday
# after the Monday its owner last came to read it.


@pytest.mark.parametrize(("schedule", "days"), [
    ("daily 07:30", 1.0), ("weekdays 09:00", 1.0), ("every 2d", 2.0), ("every 6h", 0.25),
    ("weekly mon 08:00", 7.0), ("weekly mon,thu 08:00", 3.5), ("in 3d", 0.0),
    ("once 2026-12-01T09:00", 0.0), ("nonsense", 0.0), ("", 0.0),
    ({"type": "weekly", "weekday": 0, "hour": 8, "minute": 0}, 7.0),
    ({"type": "daily", "hour": 9, "minute": 0}, 1.0),
    ({"type": "interval", "unit": "hours", "interval": 48}, 2.0),
    ({"type": "interval", "unit": "minutes", "interval": 15}, 15 / 1440),
    ({"type": "once", "target_iso": "2026-12-01T09:00:00+00:00"}, 0.0),
])
def test_period_days(schedule, days):
    assert idle_pause.period_days(schedule) == pytest.approx(days)


async def test_a_weekly_job_waits_two_of_its_weeks(deck):
    _task(deck["store"], "vedran", "Daily check")
    _task(deck["store"], "vedran", "Weekly scan", schedule="weekly mon 08:00")
    _agent_job(deck, "vedran", "w1", "Competitor scan",
               schedule={"type": "weekly", "weekday": 0, "day": "mon", "hour": 8, "minute": 0})
    deck["settings"].config = {"scheduled_idle_pause_days": 3}

    assert await idle_pause.sweep(_Db([_user("vedran", login=4)]), now=NOW.timestamp()) == 1
    by_name = {t["name"]: t["enabled"] for t in deck["store"].list(owner_id="vedran")}
    assert by_name == {"Daily check": 0, "Weekly scan": 1}
    assert deck["agent_rows"]["agent-vedran"][0]["enabled"] == 1

    # Two weeks away: the weekly ones go too, told in one more Line item.
    assert await idle_pause.sweep(_Db([_user("vedran", login=14)]), now=NOW.timestamp()) == 2
    assert not any(t["enabled"] for t in deck["store"].list(owner_id="vedran"))
    assert deck["agent_rows"]["agent-vedran"][0]["enabled"] == 0
    assert [t[1] for t in deck["told"]] == ["Scheduled tasks paused (1)", "Scheduled tasks paused (2)"]
