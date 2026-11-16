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


# ── a prompt job runs in a session of its own ─────────────────────────


class _AnyUi:
    def can_capture_escape(self):
        return False

    def __getattr__(self, name):
        return lambda *a, **kw: None


class _TurnAgent:
    """What a cron prompt turn touches; records what the model was given."""

    def __init__(self, manager, chat, reply="Checked the inbox: 2 new mails.", on_turn=None):
        self.session_manager = manager
        self.session = chat
        self.pipeline_mode = "contracts"
        self.planning_enabled = True
        self._turn_is_automated = False
        self.tools = SimpleNamespace(list_tools=lambda: ["read", "google_mail"])
        self._reply = reply
        self._on_turn = on_turn
        self.turns: list[dict] = []

    def refresh_session_runtime_flags(self):
        return None

    def _count_tokens(self, text):
        return len(str(text or "").split())

    def _record_timing_event(self, kind):
        return None

    async def _turn(self, prompt):
        from captain_claw import mail_authority

        s = self.session
        self.turns.append({
            "session_id": s.id, "name": s.name, "metadata": dict(s.metadata),
            "history": [str(m.get("content") or "") for m in s.messages],
            "job_text": mail_authority.current().job_text,
        })
        if self._on_turn:
            self._on_turn(self)
        # The turn's own messages, as the agent keeps them.
        s.add_message("user", prompt)
        s.add_message("assistant", self._reply)
        self._turn_tool_call_counts = {"google_mail|{}": 1}
        return self._reply

    async def complete(self, prompt):
        return await self._turn(prompt)

    async def stream(self, prompt):
        yield await self._turn(prompt)


def _turn_ctx(agent):
    from captain_claw.execution_queue import CommandQueueManager, FollowupQueueManager

    ctx = _Ctx(agent.session_manager)
    ctx.agent = agent
    ctx.ui = _AnyUi()
    ctx.command_queue = CommandQueueManager()
    ctx.followup_queue = FollowupQueueManager()
    ctx.last_next_steps = []
    return ctx


CHAT_HISTORY = [
    "Can you look at my inbox every morning?",
    "Done — a daily inbox check at 09:00.",
    "Earlier run: inbox checked, nothing new — confirmed.",
]
INBOX_JOB = "check the inbox with google_mail and list new mails"


async def _chat(sm, *, session_id=None, **meta):
    from captain_claw.session import Session

    chat = Session(id=session_id, name="main chat", metadata=dict(meta)) if session_id else \
        await sm.create_session(name="main chat", metadata=dict(meta))
    for i, text in enumerate(CHAT_HISTORY):
        chat.add_message("user" if i % 2 == 0 else "assistant", text)
    await sm.save_session(chat)
    return chat


async def _chat_job(sm, chat, text=INBOX_JOB, payload=None):
    return await sm.create_cron_job(
        kind="prompt", payload=payload or {"text": text}, schedule=DAILY, session_id=chat.id,
        next_run_at=to_utc_iso(compute_next_run(DAILY)),
    )


def _mentions_chat(history):
    return any(line in text for text in history for line in CHAT_HISTORY)


def test_a_run_sees_only_its_jobs_session_and_reports_to_the_chat(cron_cfg, tmp_path):
    from captain_claw import msg_origin
    from captain_claw.flight_deck import agent_cron_routes as acr

    async def run(sm):
        chat = await _chat(sm)
        agent = _TurnAgent(sm, chat)
        ctx = _turn_ctx(agent)
        job = await _chat_job(sm, chat)
        await cd.execute_cron_job(ctx, job, trigger="manual")

        # The model turn ran in the job's session, without the chat's history.
        (turn,) = agent.turns
        assert turn["session_id"] != chat.id
        assert turn["name"] == f"⏰ {INBOX_JOB}"
        assert turn["metadata"][cd.JOB_SESSION_KEY] == job.id
        assert not _mentions_chat(turn["history"])
        # The agent is back on the chat — the very object it was on.
        assert agent.session is chat

        # The result reached the chat: live, and kept in its history as a note.
        assert [sid for sid, _ in ctx.sent] == [chat.id]
        assert ctx.sent[0][1].startswith(f"✅ Cron {job.id[:8]} finished")
        stored = await sm.load_session(chat.id)
        assert [m["content"] for m in stored.messages[:3]] == CHAT_HISTORY
        assert len(stored.messages) == 4  # nothing else of the run is in the chat
        note = stored.messages[-1]
        assert note["content"] == ctx.sent[0][1]
        assert (note["origin"], note["origin_detail"], note["cron_job_id"]) == (
            "system_note", "cron_result", job.id)
        assert msg_origin.is_lane_note(note)

        # The job knows its session; the run is there, and the job's record says ok.
        got = await sm.load_cron_job(job.id)
        assert got.run_session_id == turn["session_id"]
        assert got.last_status == "ok" and got.bad_streak == 0
        own = await sm.load_session(got.run_session_id)
        assert any("Checked the inbox" in str(m.get("content")) for m in own.messages)
        assert own.metadata[cd.JOB_CHAT_KEY] == chat.id

        # Flight Deck's Scheduled view still reads the run from the job's record.
        runs = acr.runs_from_history(got.chat_history, alive=True)
        assert [r["status"] for r in runs] == ["ok"]
        assert "Checked the inbox" in runs[0]["result"]

        # The next run: the same session, its own earlier run in view, the chat not.
        await cd.execute_cron_job(ctx, got, trigger="manual")
        second = agent.turns[1]
        assert second["session_id"] == turn["session_id"]
        assert any("Checked the inbox" in text for text in second["history"])
        assert not _mentions_chat(second["history"])
        assert len((await sm.load_session(chat.id)).messages) == 5

    with_sm(tmp_path, run)


def test_an_existing_job_gets_its_session_on_its_first_run(cron_cfg, tmp_path):
    """A job and file from before sessions per job: no column, no session.
    The first run after the update makes the job's session and records it."""
    path = tmp_path / "old.db"
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE cron_jobs (id TEXT PRIMARY KEY, kind TEXT NOT NULL, payload TEXT NOT NULL,"
        " schedule TEXT NOT NULL, session_id TEXT NOT NULL, enabled INTEGER NOT NULL DEFAULT 1,"
        " created_at TEXT NOT NULL, updated_at TEXT NOT NULL, last_run_at TEXT,"
        " next_run_at TEXT NOT NULL, last_status TEXT NOT NULL DEFAULT 'pending', last_error TEXT,"
        " chat_history TEXT NOT NULL DEFAULT '[]', monitor_history TEXT NOT NULL DEFAULT '[]',"
        " bad_streak INTEGER NOT NULL DEFAULT 0)"
    )
    conn.execute(
        "INSERT INTO cron_jobs (id, kind, payload, schedule, session_id, created_at, updated_at,"
        " next_run_at) VALUES ('old-job', 'prompt', ?, '{\"type\": \"daily\", \"hour\": 9}',"
        " 'old-chat', '2026-01-01', '2026-01-01', '2026-01-02')",
        ('{"text": "' + INBOX_JOB + '"}',),
    )
    conn.commit()
    conn.close()

    async def run():
        manager = SessionManager(path)
        try:
            chat = await _chat(manager, session_id="old-chat")
            job = await manager.load_cron_job("old-job")
            assert job is not None and job.run_session_id == ""
            agent = _TurnAgent(manager, chat)
            ctx = _turn_ctx(agent)
            await cd.execute_cron_job(ctx, job, trigger="manual")
            got = await manager.load_cron_job("old-job")
            assert got.run_session_id and got.run_session_id != "old-chat"
            assert agent.turns[0]["session_id"] == got.run_session_id
            assert not _mentions_chat(agent.turns[0]["history"])
            assert ctx.sent and ctx.sent[0][0] == "old-chat"
            assert got.last_status == "ok"
        finally:
            await manager.close()

    asyncio.run(run())


def test_the_jobs_session_is_only_ever_its_own(tmp_path):
    async def run(sm):
        chat = await _chat(sm)
        ctx = _Ctx(sm)
        ctx.agent.session = chat
        job = await _chat_job(sm, chat)
        # An id that points at another session (here the chat) is never adopted.
        await sm.update_cron_job(job.id, run_session_id=chat.id)
        own = await cd.job_run_session(ctx, await sm.load_cron_job(job.id))
        assert own.id != chat.id and cd.is_job_session(own, job.id)
        assert (await sm.load_cron_job(job.id)).run_session_id == own.id
        assert (await cd.job_run_session(ctx, await sm.load_cron_job(job.id))).id == own.id
        # Deleted: a new one, recorded.
        await sm.delete_session(own.id)
        fresh = await cd.job_run_session(ctx, await sm.load_cron_job(job.id))
        assert fresh.id != own.id
        assert (await sm.load_cron_job(job.id)).run_session_id == fresh.id
        # Another job with the same task gets a session of its own, not this one.
        twin = await _chat_job(sm, chat)
        twin_session = await cd.job_run_session(ctx, twin)
        assert twin_session.name == fresh.name and twin_session.id != fresh.id
        assert cd.is_job_session(twin_session, twin.id)

    with_sm(tmp_path, run)


def test_the_jobs_session_follows_the_chats_settings(tmp_path):
    async def run(sm):
        chat = await _chat(
            sm, model_selection={"id": "model-a"}, origin={"kind": "whatsapp", "address": "chat-a"},
            session_instructions="Answer briefly.", queue={"mode": "collect"},
        )
        ctx = _Ctx(sm)
        ctx.agent.session = chat
        job = await _chat_job(sm, chat)
        meta = (await cd.job_run_session(ctx, job)).metadata
        assert meta["model_selection"] == {"id": "model-a"}
        assert meta["origin"] == {"kind": "whatsapp", "address": "chat-a"}
        assert meta["session_instructions"] == "Answer briefly."
        assert "queue" not in meta  # only the settings a run follows
        assert meta[cd.JOB_CHAT_KEY] == chat.id and meta[cd.FILES_SESSION_KEY] == chat.id
        # A setting dropped in the chat is dropped from the job's next run too.
        del chat.metadata["model_selection"]
        meta = (await cd.job_run_session(ctx, await sm.load_cron_job(job.id))).metadata
        assert "model_selection" not in meta

    with_sm(tmp_path, run)


def test_a_jobs_session_keeps_its_files_in_the_chats_folder():
    from captain_claw.agent_file_ops_mixin import AgentFileOpsMixin

    def slug(session):
        agent = SimpleNamespace(session=session,
                                _normalize_session_slug=AgentFileOpsMixin._normalize_session_slug)
        return AgentFileOpsMixin._current_session_slug(agent)

    assert slug(SimpleNamespace(id="job-session", metadata={cd.FILES_SESSION_KEY: "chat-1"})) == "chat-1"
    assert slug(SimpleNamespace(id="chat-1", metadata={})) == "chat-1"
    assert slug(None) == "default"


def test_a_job_set_up_during_a_run_reports_to_the_chat():
    from captain_claw.tools.cron_tool import CronTool

    tool = CronTool.__new__(CronTool)
    tool._agent = SimpleNamespace(session=SimpleNamespace(
        id="job-session", metadata={cd.JOB_SESSION_KEY: "j1", cd.JOB_CHAT_KEY: "chat-1"}))
    assert tool._get_session_id() == "chat-1"
    tool._agent.session = SimpleNamespace(id="chat-1", metadata={})
    assert tool._get_session_id() == "chat-1"


def test_mail_authority_still_comes_from_the_jobs_own_text(cron_cfg, tmp_path):
    from captain_claw import mail_authority

    payload = {"text": "email the inbox summary to the team", "author": "agent",
               "mail_intent_text": "check my inbox every morning"}

    async def run(sm):
        chat = await _chat(sm)
        agent = _TurnAgent(sm, chat)
        job = await _chat_job(sm, chat, payload=payload)
        await cd.execute_cron_job(_turn_ctx(agent), job, trigger="manual")
        assert agent.turns[0]["job_text"] == mail_authority.cron_job_text(payload)

    with_sm(tmp_path, run)


def test_the_web_main_chat_is_held_while_the_agent_is_on_the_jobs_session(cron_cfg, tmp_path):
    import contextlib

    events: list = []

    async def run(sm):
        chat = await _chat(sm)
        agent = _TurnAgent(sm, chat, on_turn=lambda a: events.append(("turn", a.session is chat)))
        ctx = _turn_ctx(agent)

        @contextlib.asynccontextmanager
        async def hold():
            events.append("hold")
            yield
            events.append(("release", agent.session is chat))

        ctx.hold_main_lane = hold
        await cd.execute_cron_job(ctx, await _chat_job(sm, chat), trigger="manual")

    with_sm(tmp_path, run)
    assert events == ["hold", ("turn", False), ("release", True)]


def test_results_mirrored_into_the_chat_mid_run_go_back_to_it(cron_cfg, tmp_path):
    """Flight Deck mirrors a result into ``agent.session`` — the job's while it
    runs; it belongs to the chat."""

    def mirror(agent):
        agent.session.metadata.setdefault("automation_results", []).append(
            {"label": "Scheduled task", "text": "Report ready."})

    async def run(sm):
        chat = await _chat(sm)
        agent = _TurnAgent(sm, chat, on_turn=mirror)
        job = await _chat_job(sm, chat)
        await cd.execute_cron_job(_turn_ctx(agent), job, trigger="manual")
        assert [r["text"] for r in chat.metadata["automation_results"]] == ["Report ready."]
        own = await sm.load_session((await sm.load_cron_job(job.id)).run_session_id)
        assert "automation_results" not in own.metadata

    with_sm(tmp_path, run)


def test_a_queued_run_still_reports_to_the_chat(monkeypatch):
    import captain_claw.prompt_execution as pe
    from captain_claw.execution_queue import FollowupRun

    seen: dict = {}

    async def fake_run(ctx, **kwargs):
        seen.update(kwargs)

    monkeypatch.setattr(pe, "run_prompt_in_session", fake_run)
    run = FollowupRun(prompt="check the inbox", enqueued_at_ms=0, metadata={
        "session_id": "job-session", "cron_job_id": "j1", "reply_session_id": "chat-1"})
    asyncio.run(pe.run_queued_followup_prompt(SimpleNamespace(), run))
    assert seen["session_id"] == "job-session" and seen["reply_session_id"] == "chat-1"


def test_a_run_that_gave_up_on_a_busy_main_chat_is_skipped_not_bad(cron_cfg, tmp_path):
    import contextlib

    from captain_claw.flight_deck import agent_cron_routes as acr

    async def run(sm):
        chat = await _chat(sm)
        agent = _TurnAgent(sm, chat)
        ctx = _turn_ctx(agent)

        @contextlib.asynccontextmanager
        async def busy():
            raise cd.MainChatBusyError("the main conversation stayed busy for 15 minutes")
            yield  # pragma: no cover

        ctx.hold_main_lane = busy
        job = await _chat_job(sm, chat)
        await cd.execute_cron_job(ctx, job, trigger="scheduled")
        assert agent.turns == [] and agent.session is chat
        got = await sm.load_cron_job(job.id)
        assert got.last_status == "skipped" and got.bad_streak == 0 and got.enabled is True
        assert got.last_error.startswith("the main conversation stayed busy")
        assert got.next_run_at > to_utc_iso(now_utc())
        # One run in the Scheduled view: skipped, with the reason.
        runs = acr.runs_from_history(got.chat_history, alive=True)
        assert [r["status"] for r in runs] == ["skipped"]
        assert runs[0]["error"].startswith("the main conversation stayed busy")

    with_sm(tmp_path, run)
