"""Tests for the FD scheduler: schedule parsing, next-run computation, and
the SQLite store CRUD + one-shot lifecycle.

These cover everything that doesn't need a live agent — the execution path
(inject prompt → capture reply → deliver) is integration-tested manually
since it depends on a running captain-claw agent + WhatsApp.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from captain_claw.flight_deck import fd_scheduler as sched


# ── Schedule parsing ──────────────────────────────────────────────────


BASE = datetime(2026, 5, 28, 12, 0, 0)  # a Thursday, noon


def test_every_interval():
    nxt = sched.compute_next_run("every 30m", base=BASE)
    assert nxt == (BASE.replace(minute=30)).timestamp()


def test_every_hours():
    nxt = sched.compute_next_run("every 2h", base=BASE)
    assert nxt == datetime(2026, 5, 28, 14, 0, 0).timestamp()


def test_daily_later_today():
    nxt = sched.compute_next_run("daily 18:30", base=BASE)
    assert nxt == datetime(2026, 5, 28, 18, 30, 0).timestamp()


def test_daily_already_passed_rolls_to_tomorrow():
    nxt = sched.compute_next_run("daily 08:00", base=BASE)
    assert nxt == datetime(2026, 5, 29, 8, 0, 0).timestamp()


def test_weekly_future_day():
    # BASE is Thursday (weekday 3). Next Monday (0) is 4 days ahead.
    nxt = sched.compute_next_run("weekly mon 09:00", base=BASE)
    assert nxt == datetime(2026, 6, 1, 9, 0, 0).timestamp()


def test_weekly_same_day_later_time():
    # Thursday 18:00 from Thursday noon → today.
    nxt = sched.compute_next_run("weekly thu 18:00", base=BASE)
    assert nxt == datetime(2026, 5, 28, 18, 0, 0).timestamp()


def test_weekly_same_day_passed_time_rolls_a_week():
    nxt = sched.compute_next_run("weekly thu 09:00", base=BASE)
    assert nxt == datetime(2026, 6, 4, 9, 0, 0).timestamp()


def test_in_oneshot():
    nxt = sched.compute_next_run("in 10m", base=BASE)
    assert nxt == datetime(2026, 5, 28, 12, 10, 0).timestamp()


def test_once_iso():
    nxt = sched.compute_next_run("once 2026-12-25T09:00:00", base=BASE)
    assert nxt == datetime(2026, 12, 25, 9, 0, 0).timestamp()


def test_is_one_shot():
    assert sched.is_one_shot("in 5m")
    assert sched.is_one_shot("once 2026-01-01T00:00:00")
    assert not sched.is_one_shot("every 5m")
    assert not sched.is_one_shot("daily 09:00")
    assert not sched.is_one_shot("weekly mon 09:00")


@pytest.mark.parametrize("bad", [
    "", "every", "every 0m", "daily 25:00", "daily 9:99",
    "weekly funday 09:00", "once not-a-date", "nonsense",
])
def test_invalid_schedules_raise(bad):
    with pytest.raises(sched.ScheduleError):
        sched.validate_schedule(bad)


# ── Quiet hours ───────────────────────────────────────────────────────


def test_quiet_hours_wrap_midnight(monkeypatch):
    monkeypatch.setenv("FD_SCHEDULER_QUIET_HOURS", "22-08")
    assert sched._in_quiet_hours(datetime(2026, 5, 28, 23, 0))  # 23:00 quiet
    assert sched._in_quiet_hours(datetime(2026, 5, 28, 3, 0))   # 03:00 quiet
    assert not sched._in_quiet_hours(datetime(2026, 5, 28, 12, 0))  # noon awake


def test_quiet_hours_same_day_window(monkeypatch):
    monkeypatch.setenv("FD_SCHEDULER_QUIET_HOURS", "09-17")
    assert sched._in_quiet_hours(datetime(2026, 5, 28, 12, 0))
    assert not sched._in_quiet_hours(datetime(2026, 5, 28, 20, 0))


def test_quiet_hours_unset(monkeypatch):
    monkeypatch.delenv("FD_SCHEDULER_QUIET_HOURS", raising=False)
    assert not sched._in_quiet_hours(datetime(2026, 5, 28, 3, 0))


# ── Store CRUD ────────────────────────────────────────────────────────


@pytest.fixture
def store(tmp_path):
    return sched.SchedulerStore(db_path=tmp_path / "scheduler.db")


def test_create_and_get(store):
    row = store.create(
        name="Morning briefing",
        schedule="daily 08:00",
        agent_slug="deepseek-v4-flash-glasses",
        prompt="Compile my morning briefing",
        delivery_kind="whatsapp",
        delivery_target="+385976707736",
    )
    assert row["id"].startswith("job_")
    assert row["enabled"] == 1
    assert row["next_run_at"] is not None
    # whatsapp target normalized (leading + stripped)
    assert row["delivery_target"] == "385976707736"

    fetched = store.get(row["id"])
    assert fetched["name"] == "Morning briefing"
    assert fetched["schedule"] == "daily 08:00"


def test_create_rejects_bad_schedule(store):
    with pytest.raises(sched.ScheduleError):
        store.create(
            schedule="every 0m", agent_slug="x", prompt="hi",
            delivery_kind="channel", delivery_target="c",
        )


def test_create_rejects_bad_delivery_kind(store):
    with pytest.raises(ValueError, match="delivery_kind"):
        store.create(
            schedule="daily 08:00", agent_slug="x", prompt="hi",
            delivery_kind="carrier-pigeon", delivery_target="c",
        )


def test_create_requires_prompt_and_target(store):
    with pytest.raises(ValueError, match="prompt"):
        store.create(schedule="daily 08:00", agent_slug="x", prompt="",
                     delivery_kind="channel", delivery_target="c")
    with pytest.raises(ValueError, match="delivery_target"):
        store.create(schedule="daily 08:00", agent_slug="x", prompt="hi",
                     delivery_kind="channel", delivery_target="")


def test_update_disable_clears_next_run(store):
    row = store.create(
        schedule="every 15m", agent_slug="x", prompt="hi",
        delivery_kind="channel", delivery_target="c",
    )
    assert row["next_run_at"] is not None
    updated = store.update(row["id"], enabled=False)
    assert updated["enabled"] == 0
    assert updated["next_run_at"] is None
    # re-enable recomputes
    re_en = store.update(row["id"], enabled=True)
    assert re_en["enabled"] == 1
    assert re_en["next_run_at"] is not None


def test_update_schedule_recomputes_next_run(store):
    row = store.create(
        schedule="daily 08:00", agent_slug="x", prompt="hi",
        delivery_kind="channel", delivery_target="c",
    )
    old_next = row["next_run_at"]
    updated = store.update(row["id"], schedule="daily 09:00")
    assert updated["schedule"] == "daily 09:00"
    assert updated["next_run_at"] != old_next


def test_list_due_filters(store):
    import time
    # Due job: next_run in the past
    due = store.create(
        schedule="every 15m", agent_slug="x", prompt="hi",
        delivery_kind="channel", delivery_target="c",
    )
    store.mark_run(due["id"], status="ok", result="r",
                   last_run_at=time.time() - 100, next_run_at=time.time() - 10)
    # Not-due job: next_run in the future
    notdue = store.create(
        schedule="every 15m", agent_slug="x", prompt="hi",
        delivery_kind="channel", delivery_target="c",
    )
    store.mark_run(notdue["id"], status="ok", result="r",
                   last_run_at=time.time(), next_run_at=time.time() + 9999)
    # Disabled job: never due
    disabled = store.create(
        schedule="every 15m", agent_slug="x", prompt="hi",
        delivery_kind="channel", delivery_target="c", enabled=False,
    )

    due_ids = {j["id"] for j in store.list_due(time.time())}
    assert due["id"] in due_ids
    assert notdue["id"] not in due_ids
    assert disabled["id"] not in due_ids


def test_delete(store):
    row = store.create(
        schedule="daily 08:00", agent_slug="x", prompt="hi",
        delivery_kind="channel", delivery_target="c",
    )
    assert store.delete(row["id"]) is True
    assert store.get(row["id"]) is None
    assert store.delete("job_nonexistent") is False


def test_mark_run_one_shot_disable(store):
    """Simulate the loop disabling a one-shot after it fires."""
    import time
    row = store.create(
        schedule="in 5m", agent_slug="x", prompt="hi",
        delivery_kind="channel", delivery_target="c",
    )
    assert row["enabled"] == 1
    store.mark_run(row["id"], status="ok", result="done",
                   last_run_at=time.time(), next_run_at=None, enabled=0)
    after = store.get(row["id"])
    assert after["enabled"] == 0
    assert after["next_run_at"] is None
    assert after["last_status"] == "ok"
    assert after["last_result"] == "done"
