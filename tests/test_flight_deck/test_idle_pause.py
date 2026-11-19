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
    monkeypatch.delenv("FD_IDLE_PAUSE_DAYS", raising=False)
    monkeypatch.setattr("captain_claw.flight_deck.auth._fd_auth_enabled", lambda: True)
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
    return {"store": store, "agent_rows": agent_rows, "told": told}


def _task(store, owner: str, name: str, enabled: bool = True) -> dict:
    return store.create(owner_id=owner, name=name, schedule="daily 07:30", agent_slug=f"agent-{owner}",
                        prompt="Check the tenders", delivery_kind="agent",
                        delivery={"chat": True}, enabled=enabled)


def _agent_job(deck, owner: str, job_id: str, text: str, enabled: int = 1) -> None:
    deck["agent_rows"].setdefault(f"agent-{owner}", []).append(
        {"id": job_id, "enabled": enabled, "kind": "prompt", "payload": json.dumps({"text": text})})


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
    monkeypatch.setenv("FD_IDLE_PAUSE_DAYS", "0")
    assert await idle_pause.sweep(db, now=NOW.timestamp()) == 0
    monkeypatch.delenv("FD_IDLE_PAUSE_DAYS")
    monkeypatch.setattr("captain_claw.flight_deck.auth._fd_auth_enabled", lambda: False)
    assert await idle_pause.sweep(db, now=NOW.timestamp()) == 0
    assert deck["store"].list(owner_id="roko")[0]["enabled"]


async def test_the_day_count_comes_from_the_deck(deck, monkeypatch):
    _task(deck["store"], "vedran", "Weekly scan")
    db = _Db([_user("vedran", login=4)])
    monkeypatch.setenv("FD_IDLE_PAUSE_DAYS", "3")
    assert idle_pause.idle_days() == 3
    assert await idle_pause.sweep(db, now=NOW.timestamp()) == 1
    assert "3 days" in deck["told"][0][2]


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

    async def boom(uid):
        if uid == "a":
            raise RuntimeError("agent file locked")
        return await real(uid)

    monkeypatch.setattr(idle_pause, "_pause_agent_jobs", boom)
    db = _Db([_user("a", login=9), _user("b", login=9)])
    assert await idle_pause.sweep(db, now=NOW.timestamp()) == 1
    assert [t[0] for t in deck["told"]] == ["b"]


def test_on_by_default_with_accounts_and_off_by_setting(monkeypatch):
    monkeypatch.delenv("FD_IDLE_PAUSE_DAYS", raising=False)
    monkeypatch.setattr("captain_claw.flight_deck.auth._fd_auth_enabled", lambda: True)
    assert idle_pause.active() and idle_pause.idle_days() == 7
    monkeypatch.setenv("FD_IDLE_PAUSE_DAYS", "off")
    assert not idle_pause.active()
