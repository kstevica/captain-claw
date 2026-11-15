"""Routine safety for an agent's own scheduled jobs (cron_dispatch).

A job that runs the model repeats no more often than the floor, an agent
holds a capped number of active jobs, a run that only announces its work is
recorded as "did nothing", a job is paused after N bad runs in a row, a late
daily/weekly run is skipped, and a list-shaped task asks for one closing line
per part.
"""

from __future__ import annotations

import asyncio
import sqlite3
from datetime import timedelta
from types import SimpleNamespace

import pytest

from captain_claw import config as cfg_mod
from captain_claw import cron_dispatch as cd
from captain_claw.cron import compute_next_run, now_utc, to_utc_iso
from captain_claw.session import SessionManager


# ── helpers ───────────────────────────────────────────────────────────


@pytest.fixture
def cron_cfg(monkeypatch):
    """A config whose cron limits a test can change."""
    cfg = cfg_mod.Config()
    monkeypatch.setattr(cfg_mod, "_config", cfg)
    return cfg.cron


def with_sm(tmp_path, body):
    """Run ``body(manager)`` in one event loop and close the file after."""

    async def main():
        manager = SessionManager(tmp_path / "sessions.db")
        try:
            await body(manager)
        finally:
            await manager.close()

    asyncio.run(main())


class _Ctx:
    """The RuntimeContext surface cron_dispatch touches."""

    def __init__(self, manager=None):
        self.agent = SimpleNamespace(session_manager=manager)
        self.cron_running_job_ids: set[str] = set()
        self.sent: list[tuple[str, str]] = []
        self.ui = SimpleNamespace(
            append_tool_output=lambda *a, **k: None,
            print_error=lambda *a, **k: None,
            print_message=lambda *a, **k: None,
        )

    def bind(self, manager):
        self.agent.session_manager = manager

    async def on_cron_output(self, session_id, text):
        self.sent.append((session_id, text))


async def _job(manager, *, schedule, kind="prompt", payload=None, next_run_at=None):
    return await manager.create_cron_job(
        kind=kind,
        payload=payload if payload is not None else {"text": "check the inbox"},
        schedule=schedule,
        session_id="s1",
        next_run_at=next_run_at or to_utc_iso(compute_next_run(schedule)),
    )


DAILY = {"type": "daily", "hour": 9, "minute": 0}


# ── what counts as doing nothing ──────────────────────────────────────


def test_tool_use_is_never_doing_nothing():
    assert cd.run_did_nothing("Running the reply-watch — checking the live inbox now.", 1) is False


@pytest.mark.parametrize("reply", [
    "",
    "   ",
    "Running the reply-watch — checking the live inbox now.",
    "Let me check the inbox.",
])
def test_announcing_without_a_tool_is_doing_nothing(reply):
    assert cd.run_did_nothing(reply, 0) is True


@pytest.mark.parametrize("reply", [
    "Time to stretch! Stand up and move for five minutes.",
    "Good morning — today is Monday, 16 November.",
])
def test_a_plain_answer_without_tools_did_its_job(reply):
    assert cd.run_did_nothing(reply, 0) is False


def test_turn_tool_calls_counts_every_call():
    agent = SimpleNamespace(_turn_tool_call_counts={"read|{}": 2, "shell|{}": 1})
    assert cd.turn_tool_calls(agent) == 3
    assert cd.turn_tool_calls(SimpleNamespace()) == 0


def test_completion_message_says_did_nothing():
    msg = cd.format_cron_completion("abcdef1234", "check the inbox", "Checking now.", did_nothing=True)
    assert msg.startswith("⚠️ Cron abcdef12 did nothing — check the inbox")
    assert "nothing was checked, sent or saved" in msg
    assert cd.format_cron_completion("abcdef1234", "x", "done").startswith("✅ Cron abcdef12 finished")


# ── checklist ─────────────────────────────────────────────────────────


def test_checklist_needs_two_list_items():
    assert cd.checklist_parts("check the inbox") == 0
    assert cd.checklist_parts("- check the inbox") == 0
    assert cd.checklist_parts("- check the inbox\n- reply to Ana\n* file the receipts") == 3
    assert cd.checklist_parts("1. check\n2) reply") == 2


def test_wrap_asks_for_one_line_per_part_only_for_lists():
    plain = cd.wrap_cron_prompt("job12345", "check the inbox")
    assert "parts" not in plain
    assert "call them in this turn" in plain
    listed = cd.wrap_cron_prompt("job12345", "- check the inbox\n- reply to Ana")
    assert "The task has 2 parts" in listed
    assert "Done, Blocked or Skipped" in listed
    assert listed.rstrip().endswith("- reply to Ana")


def test_parts_accounted():
    reply = (
        "Here is the run.\n\n"
        "1. Done — inbox checked, 2 new mails\n"
        "- ✅ Blocked — no login for the portal\n"
        "**Skipped** — nothing to file today\n"
        "Reply to Ana: done, sent at 09:02\n"
        "I'm done for now."
    )
    assert cd.parts_accounted(reply) == 4


# ── limits on new jobs ────────────────────────────────────────────────


def test_interval_minutes():
    assert cd.interval_minutes({"type": "interval", "unit": "minutes", "interval": 5}) == 5
    assert cd.interval_minutes({"type": "interval", "unit": "hours", "interval": 2}) == 120
    assert cd.interval_minutes(DAILY) is None


def test_floor_applies_to_model_runs_only(cron_cfg, tmp_path):
    every5 = {"type": "interval", "unit": "minutes", "interval": 5}

    async def run(sm):
        refusal = await cd.check_new_cron_job(sm, "prompt", every5)
        assert "every 15 minutes" in refusal and "every 5m" in refusal
        assert await cd.check_new_cron_job(sm, "orchestrate", every5)
        assert await cd.check_new_cron_job(sm, "script", every5) == ""
        assert await cd.check_new_cron_job(
            sm, "prompt", {"type": "interval", "unit": "minutes", "interval": 15}) == ""

    with_sm(tmp_path, run)


def test_cap_on_active_jobs(cron_cfg, tmp_path):
    cron_cfg.max_active_jobs = 2

    async def run(sm):
        a = await _job(sm, schedule=DAILY)
        await _job(sm, schedule=DAILY)
        refusal = await cd.check_new_cron_job(sm, "prompt", DAILY)
        assert "already has 2 active scheduled jobs" in refusal
        await sm.update_cron_job(a.id, enabled=False)  # a paused job doesn't count
        assert await cd.check_new_cron_job(sm, "prompt", DAILY) == ""

    with_sm(tmp_path, run)


def test_cron_tool_refuses_a_too_frequent_job(cron_cfg, tmp_path):
    from captain_claw.tools.cron_tool import CronTool

    async def run(sm):
        tool = CronTool.__new__(CronTool)
        tool._get_session_manager = lambda: sm  # type: ignore[method-assign]
        tool._get_session_id = lambda: "s1"  # type: ignore[method-assign]
        tool._agent = None
        result = await tool._create("every 5m", "check the inbox")
        assert result.success is False
        assert "every 15 minutes" in (result.error or "")
        assert await sm.list_cron_jobs() == []

    with_sm(tmp_path, run)


# ── bad runs in a row ─────────────────────────────────────────────────


def test_streak_counts_bad_runs_and_resets_on_a_good_one(cron_cfg, tmp_path):
    ctx = _Ctx()

    async def run(sm):
        ctx.bind(sm)
        job = await _job(sm, schedule=DAILY)
        await cd.note_run_outcome(ctx, job.id, "no_action")
        r = await cd.note_run_outcome(ctx, job.id, "failed", detail="boom")
        assert r == {"streak": 2, "paused": False}
        got = await sm.load_cron_job(job.id)
        assert (got.bad_streak, got.last_status, got.last_error) == (2, "failed", "boom")
        await cd.note_run_outcome(ctx, job.id, "ok")
        got = await sm.load_cron_job(job.id)
        assert (got.bad_streak, got.last_status, got.last_error) == (0, "ok", None)

    with_sm(tmp_path, run)


def test_paused_after_the_limit_and_told(cron_cfg, tmp_path):
    cron_cfg.pause_after_bad_runs = 3
    ctx = _Ctx()

    async def run(sm):
        ctx.bind(sm)
        job = await _job(sm, schedule=DAILY)
        for _ in range(2):
            assert (await cd.note_run_outcome(ctx, job.id, "no_action"))["paused"] is False
        r = await cd.note_run_outcome(ctx, job.id, "no_action")
        assert r == {"streak": 3, "paused": True}
        got = await sm.load_cron_job(job.id)
        assert got.enabled is False and got.last_status == "paused"
        assert got.last_error.startswith("Paused after 3 runs in a row")
        assert ctx.sent and ctx.sent[0][0] == "s1" and "paused" in ctx.sent[0][1]
        assert any("[CRON] job paused" in ev["content"] for ev in got.chat_history)
        # Turning it on again gives it its full allowance back.
        await sm.update_cron_job(job.id, enabled=True)
        assert (await sm.load_cron_job(job.id)).bad_streak == 0

    with_sm(tmp_path, run)


def test_zero_limit_never_pauses(cron_cfg, tmp_path):
    cron_cfg.pause_after_bad_runs = 0
    ctx = _Ctx()

    async def run(sm):
        ctx.bind(sm)
        job = await _job(sm, schedule=DAILY)
        for _ in range(12):
            await cd.note_run_outcome(ctx, job.id, "failed")
        got = await sm.load_cron_job(job.id)
        assert got.enabled is True and got.bad_streak == 12

    with_sm(tmp_path, run)


def test_failing_run_is_counted_once_and_a_runaway_one_shot_stops(cron_cfg, tmp_path):
    """An empty prompt fails before the turn; a one-shot whose time is past
    would otherwise fire on every poll forever."""
    cron_cfg.pause_after_bad_runs = 3
    ctx = _Ctx()

    async def run(sm):
        ctx.bind(sm)
        past = to_utc_iso(now_utc() - timedelta(minutes=1))
        job = await _job(sm, schedule={"type": "once", "target_iso": past},
                         payload={"text": ""}, next_run_at=past)
        for n in range(1, 4):
            await cd.execute_cron_job(ctx, await sm.load_cron_job(job.id))
            assert (await sm.load_cron_job(job.id)).bad_streak == n
        got = await sm.load_cron_job(job.id)
        assert got.enabled is False and got.last_status == "paused"

    with_sm(tmp_path, run)


def test_a_turn_that_counted_its_failure_is_not_counted_again(cron_cfg, tmp_path, monkeypatch):
    ctx = _Ctx()

    async def failing_dispatch(ctx_, **kwargs):
        await cd.note_run_outcome(ctx_, kwargs["cron_job_id"], "failed", detail="model down")
        err = RuntimeError("model down")
        err._cron_outcome_noted = True  # type: ignore[attr-defined]
        raise err

    import captain_claw.prompt_execution as pe

    monkeypatch.setattr(pe, "dispatch_prompt_in_session", failing_dispatch)

    async def run(sm):
        ctx.bind(sm)
        job = await _job(sm, schedule=DAILY)
        await cd.execute_cron_job(ctx, job, trigger="manual")
        got = await sm.load_cron_job(job.id)
        assert got.bad_streak == 1 and got.last_status == "failed"
        assert got.last_run_at

    with_sm(tmp_path, run)


def test_prompt_run_status_comes_from_its_turn(cron_cfg, tmp_path, monkeypatch):
    """execute_cron_job doesn't paint a run the turn called "did nothing" as ok."""
    ctx = _Ctx()

    async def stalled_dispatch(ctx_, **kwargs):
        await cd.note_run_outcome(ctx_, kwargs["cron_job_id"], "no_action")
        return "executed"

    import captain_claw.prompt_execution as pe

    monkeypatch.setattr(pe, "dispatch_prompt_in_session", stalled_dispatch)

    async def run(sm):
        ctx.bind(sm)
        job = await _job(sm, schedule=DAILY)
        await cd.execute_cron_job(ctx, job, trigger="manual")
        got = await sm.load_cron_job(job.id)
        assert got.last_status == "no_action" and got.bad_streak == 1

        once = await _job(sm, schedule={"type": "once", "unit": "minutes", "amount": 5})
        await cd.execute_cron_job(ctx, once, trigger="manual")
        got = await sm.load_cron_job(once.id)
        assert got.enabled is False and got.last_status == "no_action"

    with_sm(tmp_path, run)


# ── missed slots ──────────────────────────────────────────────────────


def _late(minutes, schedule=DAILY):
    return SimpleNamespace(schedule=schedule,
                           next_run_at=to_utc_iso(now_utc() - timedelta(minutes=minutes)))


def test_missed_slot(cron_cfg):
    assert cd.missed_slot(_late(30)) == ""
    why = cd.missed_slot(_late(300))
    assert why.startswith("missed its ") and why.endswith("by 5h")
    assert cd.missed_slot(_late(300, {"type": "interval", "unit": "minutes", "interval": 30})) == ""
    assert cd.missed_slot(_late(300, {"type": "once", "target_iso": "2026-01-01T00:00:00+00:00"})) == ""
    cron_cfg.skip_missed_after_minutes = 0
    assert cd.missed_slot(_late(3000)) == ""


def test_late_daily_run_is_skipped_not_run(cron_cfg, tmp_path, monkeypatch):
    ctx = _Ctx()
    ran: list[str] = []

    async def dispatch(ctx_, **kwargs):
        ran.append(kwargs["cron_job_id"])
        return "executed"

    import captain_claw.prompt_execution as pe

    monkeypatch.setattr(pe, "dispatch_prompt_in_session", dispatch)

    async def run(sm):
        ctx.bind(sm)
        job = await _job(sm, schedule=DAILY,
                         next_run_at=to_utc_iso(now_utc() - timedelta(hours=5)))
        await cd.execute_cron_job(ctx, job, trigger="scheduled")
        assert ran == []
        got = await sm.load_cron_job(job.id)
        assert got.last_status == "skipped" and got.enabled is True
        assert got.next_run_at > to_utc_iso(now_utc())
        assert any(ev["content"].startswith("[CRON] job skipped: missed its")
                   for ev in got.chat_history)
        assert ctx.sent and "skipped" in ctx.sent[0][1]
        # Run now is the person's choice: never skipped.
        await sm.update_cron_job(job.id, next_run_at=to_utc_iso(now_utc() - timedelta(hours=5)))
        await cd.execute_cron_job(ctx, await sm.load_cron_job(job.id), trigger="manual")
        assert ran == [job.id]

    with_sm(tmp_path, run)


# ── storage ───────────────────────────────────────────────────────────


def test_old_file_gets_the_bad_streak_column(tmp_path):
    path = tmp_path / "old.db"
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE cron_jobs (id TEXT PRIMARY KEY, kind TEXT NOT NULL, payload TEXT NOT NULL,"
        " schedule TEXT NOT NULL, session_id TEXT NOT NULL, enabled INTEGER NOT NULL DEFAULT 1,"
        " created_at TEXT NOT NULL, updated_at TEXT NOT NULL, last_run_at TEXT,"
        " next_run_at TEXT NOT NULL, last_status TEXT NOT NULL DEFAULT 'pending', last_error TEXT)"
    )
    conn.execute(
        "INSERT INTO cron_jobs (id, kind, payload, schedule, session_id, created_at, updated_at,"
        " next_run_at) VALUES ('j1', 'prompt', '{\"text\": \"x\"}', '{\"type\": \"daily\"}', 's1',"
        " '2026-01-01', '2026-01-01', '2026-01-02')"
    )
    conn.commit()
    conn.close()

    async def run():
        manager = SessionManager(path)
        try:
            job = await manager.load_cron_job("j1")
            assert job is not None and job.bad_streak == 0
            await manager.update_cron_job("j1", bad_streak=4)
            assert (await manager.load_cron_job("j1")).bad_streak == 4
        finally:
            await manager.close()

    asyncio.run(run())


# ── Flight Deck's reading of the runs ─────────────────────────────────


def test_fd_reads_did_nothing_and_skipped_runs():
    from captain_claw.flight_deck import agent_cron_routes as acr

    t = "2026-11-15T09:00:0{}+00:00"
    history = [
        {"role": "system", "content": "[CRON] job skipped: missed its 2026-11-14 09:00 UTC slot by 5h",
         "timestamp": t.format(0)},
        {"role": "system", "content": "[CRON] job start trigger=scheduled kind=prompt session=s1",
         "timestamp": t.format(1), "trigger": "scheduled"},
        {"role": "assistant", "content": "Checking the inbox now.", "timestamp": t.format(2)},
        {"role": "system", "content": f"[CRON] run did nothing: {cd.NO_ACTION_TEXT}",
         "timestamp": t.format(3)},
        {"role": "system", "content": "[CRON] job done trigger=scheduled kind=prompt session=s1",
         "timestamp": t.format(4)},
    ]
    runs = acr.runs_from_history(history, alive=True)
    assert [r["status"] for r in runs] == ["no_action", "skipped"]
    assert runs[0]["error"] == cd.NO_ACTION_TEXT
    assert runs[1]["error"].startswith("missed its 2026-11-14 09:00 UTC slot")
