"""Bat Phase 3 — the agent `bat` tool: recursion guard, validation, routing."""

from __future__ import annotations

import pytest

from captain_claw.tools.bat import BatTool, _in_worker


@pytest.mark.parametrize("marker", ["CLAW_BAT_WORKER", "CLAW_VATRA_WORKER",
                                    "CLAW_BASNA_WORKER", "CLAW_CODE_AGENT"])
async def test_start_refused_inside_any_worker(monkeypatch, marker):
    monkeypatch.setenv(marker, "1")
    assert _in_worker() is True
    res = await BatTool()._start("http://localhost:1", task="do it")
    assert not res.success and "recursion" in res.error.lower()


async def test_start_requires_task(monkeypatch):
    for m in ("CLAW_BAT_WORKER", "CLAW_VATRA_WORKER", "CLAW_BASNA_WORKER",
              "CLAW_COUNCIL_WORKER", "CLAW_CODE_AGENT"):
        monkeypatch.delenv(m, raising=False)
    res = await BatTool()._start("http://localhost:1", task="   ")
    assert not res.success and "task" in res.error.lower()


async def test_unknown_action(monkeypatch):
    tool = BatTool()
    monkeypatch.setattr(tool, "_get_fd_url", lambda **kw: "http://localhost:1")
    res = await tool.execute(action="bogus")
    assert not res.success and "unknown action" in res.error.lower()


async def test_no_fd_url_is_a_clear_error(monkeypatch):
    tool = BatTool()
    monkeypatch.setattr(tool, "_get_fd_url", lambda **kw: "")
    res = await tool.execute(action="list")
    assert not res.success and "flight deck url" in res.error.lower()


def test_tool_schema_actions():
    actions = BatTool().parameters["properties"]["action"]["enum"]
    assert set(actions) == {"start", "status", "get", "list", "cancel"}
