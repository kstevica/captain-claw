"""MCP tool calls an agent follows in short requests.

A tool call can now wait for a person (the server asks them first, on a card:
``mcp_elicitation``), so an agent no longer holds one request open for the
whole call. It starts the call with a little waiting, then follows it
(``mcp_routes``: ``/fd/mcp/agent/calls``, ``…/follow``, ``…/cancel``), as it
does with ``ask_user`` and approval cards. Flight Deck keeps the call running
in between and bounds it:

- a call with no card open and no sign of life (start, progress, a card
  opening or closing) for :data:`QUIET_LIMIT_S` is stopped;
- a call nobody has followed for :data:`ORPHAN_S` (the agent went away) is
  stopped; stopping one tells the server, which takes its question back.

A finished call's outcome is kept :data:`DONE_KEEP_S` for a late follow.
"""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass, field
from typing import Any

from captain_claw.flight_deck.mcp_elicitation import Asker
from captain_claw.logging import get_logger

log = get_logger(__name__)

CALL_ID_RE = re.compile(r"^xm_[0-9a-f]{16}$")
QUIET_LIMIT_S = 600.0
ORPHAN_S = 90.0
DONE_KEEP_S = 600.0
MAX_RUNS_PER_AGENT = 16


class CallError(RuntimeError):
    """A call that ended without a result; the message says why."""


@dataclass(eq=False)
class Run:
    id: str
    caller: tuple[str, str]        # (agent slug, agent owner) that started it
    server: str
    tool: str
    asker: Asker
    task: asyncio.Task[dict[str, Any]]
    started: float = field(default_factory=time.monotonic)
    last_follow: float = field(default_factory=time.monotonic)
    stopped: str = ""              # why Flight Deck stopped it, if it did
    finished_at: float = 0.0


_RUNS: dict[str, Run] = {}


def reset_for_tests() -> None:
    for run in _RUNS.values():
        if not run.task.done():
            run.task.cancel()
    _RUNS.clear()


def _stop(run: Run, why: str) -> None:
    if not run.task.done():
        run.stopped = why
        run.task.cancel()
        log.info("MCP call stopped", server=run.server, tool=run.tool, why=why)


def _sweep() -> None:
    now = time.monotonic()
    for run in list(_RUNS.values()):
        if run.task.done():
            if not run.finished_at:
                run.finished_at = now
            elif now - run.finished_at > DONE_KEEP_S:
                _RUNS.pop(run.id, None)
        elif now - run.last_follow > ORPHAN_S:
            _stop(run, "nobody was waiting for it any more")


def start(*, call_id: str, caller: tuple[str, str], server: str, tool: str,
          arguments: dict[str, Any], asker: Asker | None, call: Any) -> Run:
    """Start *call* (``MCPManager.call_tool``) as a followed run. Raises
    ValueError for a bad or taken id, or too many runs for one agent."""
    _sweep()
    if not CALL_ID_RE.match(call_id) or call_id in _RUNS:
        raise ValueError("That call id is taken.")
    if sum(1 for r in _RUNS.values() if r.caller == caller and not r.task.done()) >= MAX_RUNS_PER_AGENT:
        raise ValueError(f"This agent already has {MAX_RUNS_PER_AGENT} MCP calls running.")
    # A call for nobody we may ask still runs; its server's questions are
    # cancelled. Its Asker only keeps the time of its last sign of life.
    life = asker or Asker(user_id="")
    task = asyncio.create_task(
        call(server, tool, arguments, on_progress=lambda _p: life.touch(), asker=asker),
        name=f"mcp-call[{server}/{tool}]",
    )
    # An outcome nobody followed is still read, so asyncio doesn't warn.
    task.add_done_callback(lambda t: t.cancelled() or t.exception())
    run = Run(id=call_id, caller=caller, server=server, tool=tool, asker=life, task=task)
    _RUNS[call_id] = run
    return run


def get(call_id: str, caller: tuple[str, str]) -> Run:
    """The caller's run; KeyError for another agent's or one long gone."""
    run = _RUNS.get(call_id)
    if run is None or run.caller != caller:
        raise KeyError(call_id)
    return run


async def follow(run: Run, wait: float) -> dict[str, Any]:
    """Wait up to *wait* seconds: ``{"result": …}`` once it's done, else
    ``{"pending": id, "asking": bool, …}``. Raises :class:`CallError` (or the
    manager's error) when it ended without a result."""
    run.last_follow = time.monotonic()
    if not run.task.done():
        await asyncio.wait({run.task}, timeout=max(0.0, wait))
    run.last_follow = time.monotonic()
    if not run.task.done():
        quiet = time.monotonic() - run.asker.last_activity
        if not run.asker.asking and quiet > QUIET_LIMIT_S:
            _stop(run, f"no answer from the server for {int(QUIET_LIMIT_S // 60)} minutes")
            await asyncio.wait({run.task}, timeout=5.0)
        else:
            return {"pending": run.id, "asking": run.asker.asking,
                    "seconds": int(time.monotonic() - run.started)}
    if not run.task.done():  # cancelled but still unwinding
        raise CallError(f"Stopped: {run.stopped}.")
    if run.task.cancelled():
        raise CallError(f"Stopped: {run.stopped or 'the call was cancelled'}.")
    return {"result": run.task.result()}


def cancel(run: Run) -> bool:
    if run.task.done():
        return False
    _stop(run, "the agent stopped waiting")
    return True
