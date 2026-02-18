from datetime import timedelta

import pytest

from captain_claw.cron import compute_next_run, now_utc, parse_schedule_tokens, schedule_to_text, to_utc_iso
from captain_claw.session import SessionManager


def test_parse_schedule_tokens_supports_interval_daily_weekly():
    schedule_interval, consumed_interval = parse_schedule_tokens(["every", "15m", "check", "x"])
    assert consumed_interval == 2
    assert schedule_interval["type"] == "interval"
    assert schedule_to_text(schedule_interval) == "every 15m"

    schedule_daily, consumed_daily = parse_schedule_tokens(["daily", "09:30", "check"])
    assert consumed_daily == 2
    assert schedule_daily["type"] == "daily"
    assert schedule_to_text(schedule_daily) == "daily 09:30"

    schedule_weekly, consumed_weekly = parse_schedule_tokens(["weekly", "mon", "14:05", "check"])
    assert consumed_weekly == 3
    assert schedule_weekly["type"] == "weekly"
    assert schedule_to_text(schedule_weekly) == "weekly mon 14:05"


def test_compute_next_run_is_in_future():
    now = now_utc()
    interval_next = compute_next_run({"type": "interval", "unit": "minutes", "interval": 5}, now=now)
    daily_next = compute_next_run({"type": "daily", "hour": now.hour, "minute": now.minute}, now=now)
    weekly_next = compute_next_run({"type": "weekly", "weekday": now.weekday(), "day": "mon", "hour": now.hour, "minute": now.minute}, now=now)

    assert interval_next > now
    assert daily_next > now
    assert weekly_next > now


@pytest.mark.asyncio
async def test_cron_job_crud_and_due_queries(tmp_path):
    manager = SessionManager(db_path=tmp_path / "sessions.db")
    try:
        now = now_utc()
        schedule = {"type": "interval", "unit": "minutes", "interval": 10, "_text": "every 10m"}
        due_at = to_utc_iso(now + timedelta(seconds=1))
        job = await manager.create_cron_job(
            kind="prompt",
            payload={"text": "health check"},
            schedule=schedule,
            session_id="session-1",
            next_run_at=due_at,
            enabled=True,
        )

        loaded = await manager.load_cron_job(job.id)
        assert loaded is not None
        assert loaded.kind == "prompt"
        assert loaded.session_id == "session-1"
        assert loaded.chat_history == []
        assert loaded.monitor_history == []
        assert (await manager.select_cron_job(job.id)) is not None

        jobs = await manager.list_cron_jobs(limit=10, active_only=False)
        assert any(item.id == job.id for item in jobs)

        paused_job = await manager.create_cron_job(
            kind="prompt",
            payload={"text": "skip"},
            schedule=schedule,
            session_id="session-1",
            next_run_at=to_utc_iso(now + timedelta(hours=1)),
            enabled=False,
        )
        active_jobs = await manager.list_cron_jobs(limit=10, active_only=True)
        assert any(item.id == job.id for item in active_jobs)
        assert all(item.id != paused_job.id for item in active_jobs)
        by_index = await manager.select_cron_job("#1", active_only=True)
        assert by_index is not None
        assert by_index.id == job.id
        assert await manager.select_cron_job("#2", active_only=True) is None

        due_jobs = await manager.get_due_cron_jobs(to_utc_iso(now + timedelta(seconds=2)), limit=10)
        assert any(item.id == job.id for item in due_jobs)

        assert await manager.append_cron_job_history(
            job.id,
            chat_event={"timestamp": to_utc_iso(now), "role": "assistant", "content": "ok"},
            monitor_event={"timestamp": to_utc_iso(now), "step": "job_done"},
        ) is True
        with_history = await manager.load_cron_job(job.id)
        assert with_history is not None
        assert len(with_history.chat_history) == 1
        assert len(with_history.monitor_history) == 1

        assert await manager.update_cron_job(job.id, enabled=False, last_status="paused") is True
        paused = await manager.load_cron_job(job.id)
        assert paused is not None
        assert paused.enabled is False
        assert paused.last_status == "paused"

        assert await manager.delete_cron_job(job.id) is True
        assert await manager.load_cron_job(job.id) is None
    finally:
        await manager.close()
