import asyncio

import pytest

from captain_claw.execution_queue import CommandLaneClearedError, CommandQueueManager
from captain_claw.execution_queue import CommandLaneStaleTaskError


@pytest.mark.asyncio
async def test_lane_is_serial_by_default() -> None:
    queue = CommandQueueManager()
    release = asyncio.Event()
    first_started = asyncio.Event()
    seen: list[str] = []

    async def first() -> str:
        seen.append("first_start")
        first_started.set()
        await release.wait()
        seen.append("first_end")
        return "first"

    async def second() -> str:
        seen.append("second")
        return "second"

    first_task = asyncio.create_task(queue.enqueue_in_lane("lane-a", first))
    await first_started.wait()
    second_task = asyncio.create_task(queue.enqueue_in_lane("lane-a", second))
    await asyncio.sleep(0)
    assert seen == ["first_start"]

    release.set()
    assert await first_task == "first"
    assert await second_task == "second"
    assert seen == ["first_start", "first_end", "second"]


@pytest.mark.asyncio
async def test_different_lanes_can_run_concurrently() -> None:
    queue = CommandQueueManager()
    lane_one_started = asyncio.Event()
    lane_two_started = asyncio.Event()
    release = asyncio.Event()

    async def lane_one() -> str:
        lane_one_started.set()
        await release.wait()
        return "lane-one"

    async def lane_two() -> str:
        lane_two_started.set()
        return "lane-two"

    lane_one_task = asyncio.create_task(queue.enqueue_in_lane("lane-1", lane_one))
    await lane_one_started.wait()
    lane_two_task = asyncio.create_task(queue.enqueue_in_lane("lane-2", lane_two))
    await asyncio.wait_for(lane_two_started.wait(), timeout=1.0)

    release.set()
    assert await lane_one_task == "lane-one"
    assert await lane_two_task == "lane-two"


@pytest.mark.asyncio
async def test_clear_lane_rejects_queued_entries() -> None:
    queue = CommandQueueManager()
    release = asyncio.Event()

    async def first() -> str:
        await release.wait()
        return "first"

    async def second() -> str:
        return "second"

    first_task = asyncio.create_task(queue.enqueue_in_lane("lane-clear", first))
    await asyncio.sleep(0)
    second_task = asyncio.create_task(queue.enqueue_in_lane("lane-clear", second))
    await asyncio.sleep(0)

    removed = queue.clear_lane("lane-clear")
    assert removed == 1

    release.set()
    assert await first_task == "first"
    with pytest.raises(CommandLaneClearedError):
        await second_task


@pytest.mark.asyncio
async def test_reset_all_lanes_rejects_stale_active_completion() -> None:
    queue = CommandQueueManager()
    release = asyncio.Event()
    started = asyncio.Event()

    async def stale_task() -> str:
        started.set()
        await release.wait()
        return "stale"

    task = asyncio.create_task(queue.enqueue_in_lane("lane-reset", stale_task))
    await asyncio.wait_for(started.wait(), timeout=1.0)

    queue.reset_all_lanes()
    release.set()

    with pytest.raises(CommandLaneStaleTaskError):
        await task
