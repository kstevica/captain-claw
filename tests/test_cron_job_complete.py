"""Scheduled jobs that burned tokens after their work was done.

Three 15-minute jobs shared one chat. Two filled their sheet by midnight and
then read one cell every run for eight hours, each reply saying "done, remove
this job"; our own list-task block, built from the URL in the job text, rode
on every call and the model reported it as prompt injection; and once a run's
context hit the ceiling, every call dropped a different number of old
messages, so the provider's prompt cache missed the whole history each time.

- a run that checked with its tools and ends with ``[JOB COMPLETE]`` stops
  its job (resumable, status ``completed``) and the line isn't shown;
- a cron turn never starts the list/scale machinery, and a message with one
  or two links isn't a list;
- the packer without the governor cuts once and holds the cut.
"""

from __future__ import annotations

import asyncio
import types

import pytest

from captain_claw import config as cfg_mod
from captain_claw import cron_dispatch as cd
from captain_claw.agent_context_governor_mixin import AgentContextGovernorMixin
from captain_claw.agent_scale_detection_mixin import AgentScaleDetectionMixin
from captain_claw.cron import compute_next_run, to_utc_iso
from captain_claw.session import SessionManager

EVERY_15 = {"type": "interval", "unit": "minutes", "interval": 15}


@pytest.fixture(autouse=True)
def _cfg(monkeypatch):
    monkeypatch.setattr(cfg_mod, "_config", cfg_mod.Config())


def with_sm(tmp_path, body):
    async def main():
        manager = SessionManager(tmp_path / "sessions.db")
        try:
            await body(manager)
        finally:
            await manager.close()

    asyncio.run(main())


async def _job(manager, text="Fill rows C:F, 150 per run, until B1 > 500."):
    return await manager.create_cron_job(
        kind="prompt", payload={"text": text}, schedule=EVERY_15, session_id="s1",
        next_run_at=to_utc_iso(compute_next_run(EVERY_15)),
    )


class _Ui:
    def can_capture_escape(self):
        return False

    def __getattr__(self, name):
        return lambda *a, **kw: None


# ── the line ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("reply", [
    "B1 = 501 — every row is filled.\n[JOB COMPLETE]",
    "Gotovo.\n\n**[JOB COMPLETE]**",
    "Done\n[job complete].",
    "Done\n  [JOB_COMPLETE]  ",
])
def test_the_line_is_found_in_any_dress(reply):
    assert cd.job_complete_claimed(reply)
    assert "COMPLETE" not in cd.strip_job_complete(reply).upper()


@pytest.mark.parametrize("reply", [
    "",
    "Batch 3 of 9 written; B1 = 151.",
    "When B1 passes 500 I'll answer [JOB COMPLETE] on the last line.",
])
def test_no_line_no_claim(reply):
    assert not cd.job_complete_claimed(reply)


def test_wrap_tells_the_model_about_the_line():
    wrapped = cd.wrap_cron_prompt("job-1234", "Fill rows until B1 > 500.")
    assert cd.JOB_COMPLETE_MARKER in wrapped
    assert "Never for a job meant to repeat" in wrapped
    assert wrapped.rstrip().endswith("Fill rows until B1 > 500.")


# ── stopping the job ──────────────────────────────────────────────────


class _Ctx:
    def __init__(self, manager):
        self.agent = types.SimpleNamespace(session_manager=manager)
        self.ui = _Ui()
        self.sent: list[tuple[str, str]] = []

    async def on_cron_output(self, session_id, text):
        self.sent.append((session_id, text))


def test_stop_completed_job_disables_and_tells(tmp_path):
    async def run(sm):
        ctx = _Ctx(sm)
        job = await _job(sm)
        assert await cd.stop_completed_job(ctx, job.id) is True
        got = await sm.load_cron_job(job.id)
        assert got.enabled is False and got.last_status == "completed"
        assert ctx.sent and ctx.sent[0][0] == "s1" and "stopped" in ctx.sent[0][1]
        assert any("[CRON] job stopped" in ev["content"] for ev in got.chat_history)
        # Already off: nothing more to do, nothing more said.
        assert await cd.stop_completed_job(ctx, job.id) is False
        assert len(ctx.sent) == 1

    with_sm(tmp_path, run)


class _Agent:
    """What run_prompt_in_active_session touches on a cron turn."""

    def __init__(self, manager, reply: str, tool_calls: dict[str, int]):
        self.session_manager = manager
        self.session = types.SimpleNamespace(id="s1", messages=[])
        self._turn_is_automated = False
        self._reply = reply
        self._calls = tool_calls

    def _record_timing_event(self, kind):
        return None

    async def complete(self, prompt):
        self._turn_tool_call_counts = dict(self._calls)
        return self._reply

    async def stream(self, prompt):
        self._turn_tool_call_counts = dict(self._calls)
        yield self._reply


def _run_turn(sm, job_id, reply, tool_calls):
    from captain_claw.prompt_execution import run_prompt_in_active_session

    agent = _Agent(sm, reply, tool_calls)
    sent: list[tuple[str, str]] = []

    async def on_cron_output(session_id, text):
        sent.append((session_id, text))

    ctx = types.SimpleNamespace(agent=agent, ui=_Ui(), on_cron_output=on_cron_output,
                                last_next_steps=[])
    return run_prompt_in_active_session(ctx, "Fill rows until B1 > 500.", queue=False,
                                        cron_job_id=job_id), sent


def test_a_checked_finish_stops_the_job(tmp_path):
    async def run(sm):
        job = await _job(sm)
        coro, sent = _run_turn(sm, job.id, "B1 = 501, every row filled.\n[JOB COMPLETE]",
                               {"google_drive": 1})
        await coro
        got = await sm.load_cron_job(job.id)
        assert got.enabled is False and got.last_status == "completed"
        assert sent and all("JOB COMPLETE" not in text for _, text in sent)
        assert any("stopped" in text for _, text in sent)

    with_sm(tmp_path, run)


def test_a_finish_claimed_without_tools_keeps_the_job(tmp_path):
    async def run(sm):
        job = await _job(sm)
        coro, sent = _run_turn(sm, job.id, "All done.\n[JOB COMPLETE]", {})
        await coro
        got = await sm.load_cron_job(job.id)
        assert got.enabled is True
        assert all("JOB COMPLETE" not in text for _, text in sent)

    with_sm(tmp_path, run)


def test_an_ordinary_run_keeps_the_job(tmp_path):
    async def run(sm):
        job = await _job(sm)
        coro, _ = _run_turn(sm, job.id, "Rows 101–150 written; B1 = 151.", {"google_drive": 3})
        await coro
        got = await sm.load_cron_job(job.id)
        assert got.enabled is True and got.last_status == "ok"

    with_sm(tmp_path, run)


# ── a reply that skipped the tools its job names did nothing ─────────

TOOLS = ["read", "write", "browser", "google_drive", "web_fetch"]
SHEET_JOB = "1) google_drive sheet_read range='B1' → START. If START > 500, report done."


def test_copied_confirmation_without_a_tool_did_nothing():
    reply = "B1 = **501** — confirmed by reading. The job is done."
    assert cd.run_did_nothing(reply, 0, task_text=SHEET_JOB, tool_names=TOOLS)
    # The same reply after really reading the cell is a run that did its job.
    assert not cd.run_did_nothing(reply, 1, task_text=SHEET_JOB, tool_names=TOOLS)


@pytest.mark.parametrize("task", [
    "Remind me to read a book.",
    "Open the browser and stretch.",
    "Say good morning.",
])
def test_plain_word_tool_names_dont_count(task):
    assert not cd.task_names_a_tool(task, TOOLS)
    assert not cd.run_did_nothing("Time to stretch!", 0, task_text=task, tool_names=TOOLS)


def test_tool_names_match_whole_words_only():
    assert cd.task_names_a_tool("use web_fetch on it", TOOLS)
    assert not cd.task_names_a_tool("my_google_drive_notes.txt", TOOLS)


def test_a_cron_turn_that_skipped_its_tool_is_counted_bad(tmp_path):
    async def run(sm):
        job = await _job(sm, text=SHEET_JOB)
        from captain_claw.prompt_execution import run_prompt_in_active_session

        agent = _Agent(sm, "B1 = 501 — confirmed by reading.", {})
        agent.tools = types.SimpleNamespace(list_tools=lambda: TOOLS)
        ctx = types.SimpleNamespace(agent=agent, ui=_Ui(), on_cron_output=None, last_next_steps=[])
        await run_prompt_in_active_session(ctx, SHEET_JOB, queue=False, cron_job_id=job.id)
        got = await sm.load_cron_job(job.id)
        assert got.last_status == "no_action" and got.bad_streak == 1

    with_sm(tmp_path, run)


# ── list/scale machinery stays out of cron turns ─────────────────────


@pytest.mark.parametrize("origin,off", [
    (("cron", "cron", None), True),
    (("human", "", "web"), False),
    (None, False),
])
def test_scale_system_is_off_for_cron_turns(monkeypatch, origin, off):
    monkeypatch.delenv("CLAW_NO_SCALE", raising=False)
    monkeypatch.delenv("CLAW_CODE_AGENT", raising=False)
    agent = types.SimpleNamespace(_turn_origin=origin)
    assert AgentScaleDetectionMixin._scale_system_disabled(agent) is off
    # Per turn, never latched: the same agent serves people next.
    assert not getattr(agent, "_skip_deferred_scale", False)


def test_one_or_two_links_are_not_a_list():
    from captain_claw import agent_orchestration_mixin as aom

    assert aom._MIN_INPUT_URLS_FOR_LIST == 3


# ── eco mode's tool list holds across calls ───────────────────────────


def test_eco_tool_set_only_grows_and_lives_in_the_session():
    from captain_claw.agent_orchestration_mixin import _ECO_STATE_KEY, AgentOrchestrationMixin

    agent = types.SimpleNamespace(session=types.SimpleNamespace(metadata={}))
    kept = AgentOrchestrationMixin._eco_kept_tools
    first = kept(agent, {"read", "browser", "google_drive"})
    assert first == {"read", "browser", "google_drive"}
    # A later call whose wording points elsewhere keeps what was sent before.
    second = kept(agent, {"read", "pdf_extract"})
    assert second == {"read", "browser", "google_drive", "pdf_extract"}
    third = kept(agent, {"read"})
    assert third == second
    assert agent.session.metadata[_ECO_STATE_KEY] == sorted(second)


def test_eco_tool_set_without_a_session_is_kept_on_the_agent():
    from captain_claw.agent_orchestration_mixin import AgentOrchestrationMixin

    agent = types.SimpleNamespace(session=None)
    AgentOrchestrationMixin._eco_kept_tools(agent, {"read", "browser"})
    assert AgentOrchestrationMixin._eco_kept_tools(agent, {"read"}) == {"read", "browser"}


# ── cut and hold without the governor ─────────────────────────────────


class _Packer(AgentContextGovernorMixin):
    def __init__(self):
        self._turn_window = None


def _history(n: int, size: int = 10):
    """user/assistant/tool triples, each message *size* tokens."""
    msgs, sidx = [], []
    for i in range(n):
        role = ("user", "assistant", "tool")[i % 3]
        msgs.append({"role": role, "content": f"m{i}", "_t": size})
        sidx.append(i)
    return msgs, sidx


def _fill(packer, msgs, sidx, budget):
    selected: set[int] = set()
    used, dropped = packer._plain_fill(msgs, sidx, selected, 0, budget, None,
                                       lambda p: msgs[p]["_t"])
    return sorted(selected), used, dropped


def test_everything_that_fits_is_sent():
    msgs, sidx = _history(9)
    kept, used, dropped = _fill(_Packer(), msgs, sidx, 1000)
    assert kept == list(range(9)) and used == 90 and dropped == 0


def test_a_cut_drops_to_low_water_then_holds():
    packer = _Packer()
    msgs, sidx = _history(30)                     # 300 tokens
    kept, used, _ = _fill(packer, msgs, sidx, 200)
    assert used <= int(packer._GOVERNOR_LOW * 200)
    assert kept == list(range(kept[0], 30))        # one contiguous tail
    assert msgs[kept[0]]["role"] != "tool"         # never opens on an orphaned result
    floor = kept[0]
    # The next calls only append: the first kept message stays put.
    for extra in range(1, 4):
        more, more_sidx = _history(30 + extra)
        kept2, _, _ = _fill(packer, more, more_sidx, 200)
        assert kept2[0] == floor and kept2[-1] == 29 + extra


def test_a_held_cut_moves_once_the_tail_outgrows_the_budget():
    packer = _Packer()
    msgs, sidx = _history(30)
    first = _fill(packer, msgs, sidx, 200)[0][0]
    grown, grown_sidx = _history(60)
    kept, used, _ = _fill(packer, grown, grown_sidx, 200)
    assert kept[0] > first and used <= 200
    assert kept == list(range(kept[0], 60))


def test_live_notes_go_in_first():
    packer = _Packer()
    msgs, sidx = _history(30)
    msgs.append({"role": "user", "content": "state", "_t": 15, "_live_notes": [("x", "state")]})
    sidx.append(None)
    kept, used, _ = _fill(packer, msgs, sidx, 200)
    assert 30 in kept and used <= 200
