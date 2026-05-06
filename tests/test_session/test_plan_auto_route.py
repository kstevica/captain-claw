"""Tests for plan-mode auto-routing (`/planning on` step 8)."""

from __future__ import annotations

from typing import Any

import pytest


class _StubAgent:
    """Minimal agent stub: only the attribute used by the auto-route path."""

    def __init__(self, plan_mode_auto: bool = False):
        self.plan_mode_auto = plan_mode_auto


class _StubWS:
    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []


class _StubServer:
    def __init__(self, agent: _StubAgent):
        self.agent = agent
        self.broadcasts: list[dict[str, Any]] = []
        self.sent: list[dict[str, Any]] = []

    def _broadcast(self, msg: dict[str, Any]) -> None:
        self.broadcasts.append(msg)

    async def _send(self, ws: _StubWS, msg: dict[str, Any]) -> None:
        self.sent.append(msg)
        ws.sent.append(msg)


@pytest.mark.asyncio
async def test_auto_route_calls_plan_then_execute(monkeypatch: pytest.MonkeyPatch):
    """The handler should call /plan with the user's content, then /plan-execute."""
    from captain_claw.web import plan_auto_route

    plan_calls: list[tuple[Any, str]] = []
    execute_calls: list[tuple[Any, str]] = []

    async def fake_plan(server: Any, request: str) -> str:
        plan_calls.append((server, request))
        return "plan body"

    async def fake_execute(server: Any, arg: str) -> str:
        execute_calls.append((server, arg))
        return "execute body"

    # Patch into the plan_commands module — handle_plan_auto_route imports
    # them lazily inside the function body, so monkeypatching the source
    # module is sufficient.
    monkeypatch.setattr(
        "captain_claw.web.plan_commands.handle_plan_command", fake_plan,
    )
    monkeypatch.setattr(
        "captain_claw.web.plan_commands.handle_plan_execute_command", fake_execute,
    )

    server = _StubServer(_StubAgent(plan_mode_auto=True))
    ws = _StubWS()
    await plan_auto_route.handle_plan_auto_route(server, ws, "make me a summary")

    assert len(plan_calls) == 1
    assert plan_calls[0][1] == "make me a summary"
    assert len(execute_calls) == 1
    assert execute_calls[0][1] == ""

    # User message is echoed and command results are sent.
    user_echo = [b for b in server.broadcasts if b.get("type") == "chat_message"]
    assert len(user_echo) == 1
    assert user_echo[0]["role"] == "user"
    assert user_echo[0]["content"] == "make me a summary"

    cmd_results = [m for m in server.sent if m.get("type") == "command_result"]
    assert [r["command"] for r in cmd_results] == [
        "/plan make me a summary",
        "/plan-execute",
    ]
    assert [r["content"] for r in cmd_results] == ["plan body", "execute body"]


@pytest.mark.asyncio
async def test_auto_route_reports_failure(monkeypatch: pytest.MonkeyPatch):
    """When /plan raises, the handler must surface a single error result."""
    from captain_claw.web import plan_auto_route

    async def boom(server: Any, request: str) -> str:
        raise RuntimeError("planner offline")

    async def never(server: Any, arg: str) -> str:  # pragma: no cover
        raise AssertionError("execute should not run after /plan failure")

    monkeypatch.setattr(
        "captain_claw.web.plan_commands.handle_plan_command", boom,
    )
    monkeypatch.setattr(
        "captain_claw.web.plan_commands.handle_plan_execute_command", never,
    )

    server = _StubServer(_StubAgent(plan_mode_auto=True))
    ws = _StubWS()
    await plan_auto_route.handle_plan_auto_route(server, ws, "anything")

    cmd_results = [m for m in server.sent if m.get("type") == "command_result"]
    assert len(cmd_results) == 1
    assert "planner offline" in cmd_results[0]["content"]
    # Status returns to ready even on error.
    statuses = [b for b in server.broadcasts if b.get("type") == "status"]
    assert statuses[-1]["status"] == "ready"
