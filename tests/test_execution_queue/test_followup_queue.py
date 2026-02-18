import asyncio

import pytest

from captain_claw.execution_queue import FollowupQueueManager, FollowupRun, QueueSettings


@pytest.mark.asyncio
async def test_collect_mode_batches_multiple_prompts() -> None:
    manager = FollowupQueueManager()
    key = "session-collect"
    settings = QueueSettings(mode="collect", debounce_ms=0, cap=20, drop_policy="summarize")
    ran: list[str] = []
    done = asyncio.Event()

    manager.enqueue_followup(
        key,
        FollowupRun(prompt="first task", enqueued_at_ms=1, metadata={"session_id": key}),
        settings,
        dedupe_mode="none",
    )
    manager.enqueue_followup(
        key,
        FollowupRun(prompt="second task", enqueued_at_ms=2, metadata={"session_id": key}),
        settings,
        dedupe_mode="none",
    )

    async def _run_followup(run: FollowupRun) -> None:
        ran.append(run.prompt)
        done.set()

    manager.schedule_drain(key, _run_followup)
    await asyncio.wait_for(done.wait(), timeout=1.0)

    assert len(ran) == 1
    assert "Queued #1" in ran[0]
    assert "first task" in ran[0]
    assert "Queued #2" in ran[0]
    assert "second task" in ran[0]


@pytest.mark.asyncio
async def test_steer_mode_keeps_latest_prompt_only() -> None:
    manager = FollowupQueueManager()
    key = "session-steer"
    settings = QueueSettings(mode="steer", debounce_ms=0, cap=20, drop_policy="summarize")
    ran: list[str] = []
    done = asyncio.Event()

    manager.enqueue_followup(
        key,
        FollowupRun(prompt="old prompt", enqueued_at_ms=1, metadata={"session_id": key}),
        settings,
        dedupe_mode="none",
    )
    manager.enqueue_followup(
        key,
        FollowupRun(prompt="new prompt", enqueued_at_ms=2, metadata={"session_id": key}),
        settings,
        dedupe_mode="none",
    )

    async def _run_followup(run: FollowupRun) -> None:
        ran.append(run.prompt)
        done.set()

    manager.schedule_drain(key, _run_followup)
    await asyncio.wait_for(done.wait(), timeout=1.0)

    assert ran == ["new prompt"]


@pytest.mark.asyncio
async def test_followup_drop_summarize_emits_summary_then_latest() -> None:
    manager = FollowupQueueManager()
    key = "session-summary"
    settings = QueueSettings(mode="followup", debounce_ms=0, cap=1, drop_policy="summarize")
    ran: list[str] = []
    done = asyncio.Event()

    manager.enqueue_followup(
        key,
        FollowupRun(prompt="first", enqueued_at_ms=1, summary_line="first", metadata={"session_id": key}),
        settings,
        dedupe_mode="none",
    )
    manager.enqueue_followup(
        key,
        FollowupRun(prompt="second", enqueued_at_ms=2, summary_line="second", metadata={"session_id": key}),
        settings,
        dedupe_mode="none",
    )
    manager.enqueue_followup(
        key,
        FollowupRun(prompt="third", enqueued_at_ms=3, summary_line="third", metadata={"session_id": key}),
        settings,
        dedupe_mode="none",
    )

    async def _run_followup(run: FollowupRun) -> None:
        ran.append(run.prompt)
        if len(ran) >= 2:
            done.set()

    manager.schedule_drain(key, _run_followup)
    await asyncio.wait_for(done.wait(), timeout=1.0)

    assert len(ran) == 2
    assert "Queue summary" in ran[0]
    assert "2 queued follow-ups were summarized" in ran[0]
    assert ran[1] == "third"
