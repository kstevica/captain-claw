# ruff: noqa: F811 — `deck` is Crew's runner fixture, imported from test_crew_runner
"""Custom tools, Phase 2: Crew works in a tool's own repository, its worker hears
the tool contract, and a merge publishes the tool only once its tests and
examples pass."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from fastapi import HTTPException

from captain_claw.flight_deck import crew_runner
from captain_claw.flight_deck import external_tools as xt

from .test_crew_runner import ADMIN_GRANT, _run, _seen, _settle, deck  # noqa: F401


@pytest.fixture
def told_published(deck, monkeypatch):
    xt.reset_for_tests()
    told: list = []

    async def tell(task, slug, summary):
        told.append((slug, summary))

    monkeypatch.setattr(crew_runner, "_tell_published", tell)
    yield told
    xt.reset_for_tests()


async def _jobs() -> None:
    while crew_runner._TASKS:
        await asyncio.gather(*list(crew_runner._TASKS), return_exceptions=True)


async def _tool_task(deck, slug="echo", brief="SCENARIO:work") -> dict:
    uid = deck["admin"]["id"]
    if not xt.exists(uid, slug):
        await xt.create_tool(uid, slug, title=slug.title(), description="Says it back.")
    return await crew_runner.dispatch(deck["admin"], ADMIN_GRANT, {"harness": "claude", "brief": brief, "tool": slug})


async def test_a_tool_task_works_in_the_tools_repository_and_its_merge_publishes_it(deck, told_published):
    uid = deck["admin"]["id"]
    task = await _tool_task(deck)
    assert task["repo_kind"] == "tool" and task["repo"] == str(xt.repo_dir(uid, "echo"))
    assert crew_runner.tool_of(task) == "echo"
    done = await _settle(task)
    assert done["status"] == "ready" and done["evidence"]["commits"] >= 1
    assert "This repository is a Captain custom tool" in _seen(done)
    assert crew_runner.next_steps(done) == [crew_runner._TOOL_MERGE_STEP]
    assert "pull request" not in " ".join(crew_runner.next_steps(done)).replace("There's no pull request", "")
    merged = await crew_runner.merge(deck["admin"], ADMIN_GRANT, done["id"], expect_head=done["evidence"]["head"])
    assert merged["status"] == "delivered"
    assert "publishing this custom tool now" in crew_runner.notice_text(merged, for_agent=True)
    await _jobs()
    live = xt.read_live(uid, "echo")
    assert live and live["sha"] == merged["delivery"]["sha"]
    fresh = crew_runner._get(uid, done["id"])
    assert fresh["delivery"]["publish"]["state"] == "passed"
    assert f"is live (version {live['version']})" in crew_runner.notice_text(fresh, for_agent=True)
    assert told_published == [("echo", fresh["delivery"]["publish"])]
    # The worker's file is in the published version, from git, with no .git.
    vdir = xt.versions_dir(uid, "echo") / live["version"]
    assert (vdir / "work.txt").exists() and not (vdir / ".git").exists()


async def test_a_merge_that_breaks_the_tool_keeps_the_live_version(deck, told_published):
    uid = deck["admin"]["id"]
    await xt.create_tool(uid, "echo", title="Echo", description="Says it back.")
    first = await xt.publish(uid, "echo", by=uid)
    assert first["state"] == "passed"
    done = await _settle(await _tool_task(deck))
    wt = Path(done["worktree"])
    manifest = json.loads((wt / "tool.json").read_text())
    manifest["examples"] = [{"input": {"text": "hi"}, "expect": {"text": "something else"}}]
    (wt / "tool.json").write_text(json.dumps(manifest))
    _run(wt, "commit", "-qam", "a wrong example")
    merged = await crew_runner.merge(deck["admin"], ADMIN_GRANT, done["id"])
    await _jobs()
    assert xt.read_live(uid, "echo")["version"] == first["version"]
    publish = crew_runner._get(uid, done["id"])["delivery"]["publish"]
    assert publish["state"] == "failed" and publish["step"] == "examples"
    assert "doesn't contain what 'expect' says" in publish["detail"]
    assert told_published[0][1]["state"] == "failed"
    assert merged["delivery"]["kind"] == "merge"


async def test_where_a_tool_task_may_work(deck, told_published):
    with pytest.raises(HTTPException) as err:
        await crew_runner.dispatch(deck["admin"], ADMIN_GRANT, {"harness": "claude", "brief": "x", "tool": "nope"})
    assert err.value.status_code == 404
    with pytest.raises(HTTPException) as err:
        await crew_runner.dispatch(deck["admin"], ADMIN_GRANT, {"harness": "claude", "brief": "x", "tool": "../up"})
    assert err.value.status_code == 400
    await xt.create_tool(deck["admin"]["id"], "mine", title="Mine", description="d")
    with pytest.raises(HTTPException) as err:
        await crew_runner.dispatch(deck["admin"], ADMIN_GRANT,
                                   {"harness": "claude", "brief": "x", "tool": "mine", "project": "p"})
    assert err.value.status_code == 400 and "not project and tool" in err.value.detail
    # Someone else's tool is under their folder, so it isn't found for this one.
    await xt.create_tool("someone-else", "theirs", title="T", description="d")
    with pytest.raises(HTTPException) as err:
        await crew_runner.dispatch(deck["admin"], ADMIN_GRANT, {"harness": "claude", "brief": "x", "tool": "theirs"})
    assert err.value.status_code == 404


def test_only_a_tool_task_hears_the_contract():
    base = {"kind": "ship", "worktree": "/w", "branch": "crew/x", "brief": "do it"}
    assert xt.WORKER_CONTRACT not in crew_runner._first_prompt(base)
    assert xt.WORKER_CONTRACT in crew_runner._first_prompt({**base, "repo_kind": "tool", "repo": "/d/u/t/repo"})
    assert crew_runner._first_prompt({**base, "repo_kind": "tool", "repo": "/d/u/t/repo"}).endswith("do it")
