"""Bat — the stubborn-finisher supervisor + per-run driver.

The driver is the loop that makes Bat *stubborn*: it plans a task into steps,
runs each step through an injected ``attempt_runner``, checkpoints every step
before and after it runs, retries failed steps, and keeps going round until an
injected ``judge`` says the task is genuinely done — or a serious condition
stops it (owner cancel, LLM-spend cap reached, lost lease, no actionable steps
left, round ceiling).

Durability (Phase 2's point): all state lives in ``BatStore`` (bat.db). The
supervisor claims a per-run lease before driving, so exactly one FD drives a
run; a crashed driver's run goes stale and is re-adopted on the next tick (or
after an FD restart), its ``running`` steps demoted back to ``pending`` so it
resumes from the last checkpoint instead of redoing finished work.

Seams (filled in Phase 3 with the real ``_dispatch_one`` worker runner, the LLM
planner and the fail-closed done-judge). Until a real ``attempt_runner`` is
registered, ``bat_loop`` stays idle — it never drives a run with the default
placeholder runner. Everything here is pure orchestration and unit-testable with
injected handlers.
"""

from __future__ import annotations

import asyncio
import os
import socket
from typing import Any, Awaitable, Callable

import structlog

from captain_claw.flight_deck.bat_store import BatStore, TERMINAL_STATES

log = structlog.get_logger(__name__)

# run dict, step dict -> {ok, output, usd, tokens, produced_file, error}
AttemptRunner = Callable[[dict, dict], Awaitable[dict]]
# run dict -> [{step_key, title, seq}]
Planner = Callable[[dict], Awaitable[list[dict]]]
# run dict, steps -> {done: bool, reason: str}
Judge = Callable[[dict, list[dict]], Awaitable[dict]]

_OUTPUT_CAP = 20_000
_DEFAULT_MAX_STEP_ATTEMPTS = 3
_DEFAULT_MAX_ROUNDS = 8
_LOOP_INTERVAL_S = 5.0


# ── default (placeholder) handlers — replaced in Phase 3 ──────────────

async def _default_attempt_runner(run: dict, step: dict) -> dict:
    raise RuntimeError("Bat attempt_runner is not configured yet (Phase 3 wires the real worker dispatch)")


async def _default_planner(run: dict) -> list[dict]:
    """Phase-2 planner: take the step list straight from the run config. The
    Phase-3 planner derives steps from the task with an LLM."""
    steps = (run.get("config") or {}).get("steps") or []
    out: list[dict] = []
    for i, s in enumerate(steps):
        if isinstance(s, str):
            out.append({"step_key": s, "title": s, "seq": i})
        elif isinstance(s, dict) and s.get("step_key"):
            out.append({"step_key": str(s["step_key"]), "title": str(s.get("title", "")), "seq": int(s.get("seq", i))})
    return out


async def _default_judge(run: dict, steps: list[dict]) -> dict:
    """Phase-2 judge: done when every step is done. The Phase-3 judge is the
    fail-closed, evidence-reading, independent-panel verdict (Invariant B)."""
    if not steps:
        return {"done": False, "reason": "no steps"}
    pending = [s for s in steps if s["status"] != "done"]
    if pending:
        return {"done": False, "reason": f"{len(pending)} step(s) not done"}
    return {"done": True, "reason": "all steps done"}


_ATTEMPT_RUNNER: AttemptRunner = _default_attempt_runner
_PLANNER: Planner = _default_planner
_JUDGE: Judge = _default_judge
_STORE: BatStore | None = None
# Human-in-the-loop gate (Phase 4). Called once per drive, after planning:
#   (store, run) -> 'proceed' | 'wait' | 'cancelled' | 'error'
# It owns the start-gate (plan approval for mail/$/account runs) and the resume
# of an awaiting_plan/awaiting_human run once the human answers. None (default)
# = no gate, so every run proceeds (keeps the Phase-2 tests unchanged).
_GATE_CHECK: Callable[[BatStore, dict], Awaitable[str]] | None = None
# Called once with the final run dict when a run reaches a terminal state
# (done/error/cancelled). Phase 3 sets this to deliver the result back to the
# caller / origin channel. Optional.
_ON_FINISH: Callable[[dict], Awaitable[None]] | None = None
# The live supervisor, published while the lifespan loop runs so a route can
# kick a freshly created run immediately instead of waiting for the next tick.
_SUPERVISOR: "BatSupervisor | None" = None


def set_attempt_runner(fn: AttemptRunner) -> None:
    global _ATTEMPT_RUNNER
    _ATTEMPT_RUNNER = fn


def set_planner(fn: Planner) -> None:
    global _PLANNER
    _PLANNER = fn


def set_judge(fn: Judge) -> None:
    global _JUDGE
    _JUDGE = fn


def set_gate_check(fn: "Callable[[BatStore, dict], Awaitable[str]] | None") -> None:
    global _GATE_CHECK
    _GATE_CHECK = fn


def set_store(store: BatStore) -> None:
    global _STORE
    _STORE = store


def get_store() -> BatStore | None:
    return _STORE


def set_on_finish(fn: Callable[[dict], Awaitable[None]] | None) -> None:
    global _ON_FINISH
    _ON_FINISH = fn


async def kick() -> int:
    """Ask the live supervisor to scan now (so a just-created run starts without
    waiting for the periodic tick). No-op if the loop isn't running yet."""
    if _SUPERVISOR is None:
        return 0
    return await _SUPERVISOR.tick()


def handlers_ready() -> bool:
    """True once a real attempt_runner has been registered. The supervisor only
    drives runs when this holds, so the placeholder runner never touches a run."""
    return _ATTEMPT_RUNNER is not _default_attempt_runner


async def cancel_run(store: BatStore, run_id: str) -> bool:
    """Request a stop. The driver checks the run status between steps and stands
    down; this is also what the Phase-3 cancel route calls."""
    run = await store.get_run(run_id)
    if not run or run["status"] in TERMINAL_STATES:
        return False
    await store.set_status(run_id, "cancelled", stopped_reason="owner_cancel")
    await store.append_event(run_id, "cancelled", "stop requested by owner")
    return True


def _assemble_truth(steps: list[dict]) -> str:
    done = [s for s in steps if s["status"] == "done" and (s.get("output") or "").strip()]
    if not done:
        return ""
    parts = []
    for s in done:
        head = s.get("title") or s.get("step_key") or ""
        parts.append(f"## {head}\n\n{s['output'].strip()}" if head else s["output"].strip())
    return "\n\n".join(parts)


class BatDriver:
    """Drives one run to a terminal state. One instance can drive many runs."""

    def __init__(
        self,
        store: BatStore,
        *,
        attempt_runner: AttemptRunner | None = None,
        planner: Planner | None = None,
        judge: Judge | None = None,
        max_step_attempts: int = _DEFAULT_MAX_STEP_ATTEMPTS,
        max_rounds: int = _DEFAULT_MAX_ROUNDS,
    ) -> None:
        self.store = store
        self._runner = attempt_runner
        self._planner = planner
        self._judge = judge
        self.max_step_attempts = max_step_attempts
        self.max_rounds = max_rounds
        self._pid = os.getpid()
        try:
            self._host = socket.gethostname()
        except Exception:
            self._host = "localhost"

    # resolve handlers lazily so module-level set_* (Phase 3) is honoured even
    # for a driver built earlier (e.g. by the lifespan).
    @property
    def runner(self) -> AttemptRunner:
        return self._runner or _ATTEMPT_RUNNER

    @property
    def planner(self) -> Planner:
        return self._planner or _PLANNER

    @property
    def judge(self) -> Judge:
        return self._judge or _JUDGE

    async def _finish(self, run_id: str, status: str, *, truth: str = "",
                      reason: str = "", analysis: dict | None = None) -> None:
        await self.store.set_status(
            run_id, status, truth=truth, analysis=analysis or {}, stopped_reason=reason,
        )

    async def _ensure_plan(self, run: dict) -> list[dict]:
        run_id = run["id"]
        steps = await self.store.list_steps(run_id)
        if steps:
            return steps
        await self.store.set_status(run_id, "planning")
        try:
            plan = await self.planner(run)
        except Exception as e:  # noqa: BLE001
            await self.store.append_event(run_id, "error", f"planning failed: {e}")
            await self._finish(run_id, "error", reason=f"planning_failed: {e}")
            return []
        if not plan:
            await self.store.append_event(run_id, "error", "planner produced no steps")
            await self._finish(run_id, "error", reason="empty_plan")
            return []
        await self.store.seed_steps(run_id, plan)
        await self.store.append_event(
            run_id, "plan", f"{len(plan)} step(s) planned",
            steps=[p["step_key"] for p in plan],
        )
        return await self.store.list_steps(run_id)

    async def _should_stop(self, run_id: str) -> tuple[bool, str]:
        """Between-step checks: owner cancel, lost lease, LLM cap. Returns
        (stop, reason)."""
        run = await self.store.get_run(run_id)
        if not run:
            return True, "gone"
        if run["status"] == "cancelled":
            return True, "cancelled"
        if not await self.store.heartbeat_lease(run_id, pid=self._pid, host=self._host):
            return True, "lost_lease"
        cap = float(run.get("llm_usd_cap") or 0.0)
        if cap > 0 and float(run.get("cumulative_usd") or 0.0) >= cap:
            return True, "llm_usd_cap_reached"
        return False, ""

    async def drive(self, run_id: str) -> str:
        """Drive the run to a terminal state. Returns the final status.

        Wrapped so any unhandled error lands the run in 'error' (never stuck on
        'running'), mirroring Vatra's execute_vatra crash guard."""
        try:
            return await self._drive_inner(run_id)
        except asyncio.CancelledError:
            # The process is going down mid-drive: leave the run resumable. The
            # lease goes stale and the next tick / restart re-adopts it.
            raise
        except Exception as e:  # noqa: BLE001
            log.warning("bat.drive crashed", run_id=run_id, error=str(e))
            try:
                await self.store.append_event(run_id, "error", f"driver crashed: {e}")
                await self._finish(run_id, "error", reason=f"driver_crashed: {e}")
            except Exception:
                pass
            return "error"

    async def _drive_inner(self, run_id: str) -> str:
        run = await self.store.get_run(run_id)
        if not run:
            return "gone"
        if run["status"] in TERMINAL_STATES:
            return run["status"]

        # Own the lease for the duration of the drive. Idempotent with the
        # supervisor's pre-claim; refused only if another live driver holds it.
        if not await self.store.claim_lease(run_id, pid=self._pid, host=self._host):
            return run["status"]

        # On adopt, any step left 'running' was orphaned by a crash (no driver
        # is mid-step at entry) — demote it so this drive re-runs it from the
        # last checkpoint rather than skipping it forever.
        demoted = await self.store.demote_running_steps(run_id)
        if demoted:
            await self.store.append_event(run_id, "note", f"resumed: re-queued {demoted} orphaned step(s)")

        steps = await self._ensure_plan(run)
        if not steps:
            fresh = await self.store.get_run(run_id)
            return fresh["status"] if fresh else "error"

        # Human-in-the-loop gate (start-gate for mail/$/account runs, and resume
        # of an awaiting_* run once the human answered). Owns awaiting_* status.
        if _GATE_CHECK is not None:
            run = await self.store.get_run(run_id)
            decision = await _GATE_CHECK(self.store, run)
            if decision in ("wait", "cancelled", "error"):
                fresh = await self.store.get_run(run_id)
                return fresh["status"] if fresh else "error"
            # 'proceed' falls through

        await self.store.set_status(run_id, "running")
        await self.store.append_event(run_id, "phase", "running")

        rounds = 0
        while True:
            rounds += 1
            steps = await self.store.list_steps(run_id)
            actionable = self._actionable(steps)
            for step in actionable:
                stop, reason = await self._should_stop(run_id)
                if stop:
                    return await self._stop(run_id, reason, rounds)
                stood_down = await self._run_step(run_id, step)
                if stood_down:
                    # the step needs a human (awaiting_human) — stand down; the
                    # answer re-kicks the run and _GATE_CHECK resumes it.
                    fresh = await self.store.get_run(run_id)
                    return fresh["status"] if fresh else "error"

            steps = await self.store.list_steps(run_id)
            run = await self.store.get_run(run_id)
            try:
                verdict = await self.judge(run, steps)
            except Exception as e:  # noqa: BLE001
                verdict = {"done": False, "reason": f"judge_error: {e}"}

            if verdict.get("done"):
                await self.store.append_event(run_id, "done", verdict.get("reason", "done"))
                await self._finish(
                    run_id, "done", truth=_assemble_truth(steps),
                    analysis={"rounds": rounds, "verdict": verdict},
                )
                return "done"

            if not self._actionable(steps) or rounds >= self.max_rounds:
                reason = verdict.get("reason", "not_done")
                await self.store.append_event(
                    run_id, "error", f"stopped: {reason} (round {rounds})")
                await self._finish(
                    run_id, "error", truth=_assemble_truth(steps),
                    reason=reason, analysis={"rounds": rounds, "verdict": verdict},
                )
                return "error"
            # otherwise: failed steps with attempts left remain actionable → loop

    def _actionable(self, steps: list[dict]) -> list[dict]:
        out = []
        for s in steps:
            if s["status"] == "pending":
                out.append(s)
            elif s["status"] == "failed" and int(s["attempt"]) < self.max_step_attempts:
                out.append(s)
        return out

    async def _run_step(self, run_id: str, step: dict) -> bool:
        """Run one step attempt. Returns True if the run stood down to wait for a
        human (the step asked for input); False otherwise."""
        key = step["step_key"]
        attempt = int(step["attempt"]) + 1
        await self.store.upsert_step(run_id, key, status="running", attempt=attempt)
        await self.store.append_event(
            run_id, "step", step.get("title") or key, agent=key, attempt=attempt)
        run = await self.store.get_run(run_id)
        try:
            res = await self.runner(run, step)
        except Exception as e:  # noqa: BLE001
            res = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        # The step needs something only the human can provide (a code, a
        # credential, a CAPTCHA). Raise a durable ask and stand down; the step
        # returns to 'pending' so it re-runs once answered.
        if isinstance(res.get("ask"), dict):
            from captain_claw.flight_deck import human_ask
            a = res["ask"]
            await human_ask.raise_ask(
                self.store, run_id=run_id, owner=run["owner_id"],
                kind=str(a.get("kind") or "input"), question=str(a.get("question") or ""),
                options=a.get("options") or [], step_key=key,
                secret=bool(a.get("secret")), expires_at=float(a.get("expires_at") or 0.0),
            )
            await self.store.upsert_step(run_id, key, status="pending")
            await self.store.set_status(run_id, "awaiting_human")
            await self.store.append_event(
                run_id, "awaiting_human",
                "awaiting a private value" if a.get("secret") else (a.get("question") or "awaiting input"),
                agent=key)
            return True
        usd = float(res.get("usd") or 0.0)
        tokens = int(res.get("tokens") or 0)
        if usd or tokens:
            await self.store.bump_cost(run_id, usd, tokens)
        if res.get("ok"):
            await self.store.upsert_step(
                run_id, key, status="done",
                output=str(res.get("output", ""))[:_OUTPUT_CAP],
                produced_file=str(res.get("produced_file", "")), error="",
            )
            await self.store.append_event(run_id, "step", step.get("title") or key,
                                          agent=key, ok=True, usd=usd)
        else:
            await self.store.upsert_step(
                run_id, key, status="failed", error=str(res.get("error", ""))[:_OUTPUT_CAP])
            await self.store.append_event(run_id, "step", step.get("title") or key,
                                          agent=key, ok=False,
                                          detail=str(res.get("error", ""))[:200])
        return False

    async def _stop(self, run_id: str, reason: str, rounds: int) -> str:
        if reason == "cancelled":
            await self.store.append_event(run_id, "cancelled", "stopped by owner")
            # status already 'cancelled'
            return "cancelled"
        if reason == "lost_lease":
            await self.store.append_event(run_id, "note", "lost lease — standing down")
            return "running"  # left non-terminal for re-adoption
        if reason == "gone":
            return "gone"
        # llm_usd_cap_reached (serious problem): stop, keep work, resumable
        steps = await self.store.list_steps(run_id)
        await self.store.append_event(run_id, "budget", f"stopped: {reason} (round {rounds})")
        await self._finish(run_id, "error", truth=_assemble_truth(steps),
                           reason=reason, analysis={"rounds": rounds})
        return "error"


class BatSupervisor:
    """Scans the store for adoptable runs, leases each, and drives it in its own
    task so one long run never blocks another."""

    def __init__(self, store: BatStore, driver: BatDriver | None = None) -> None:
        self.store = store
        self.driver = driver or BatDriver(store)
        self._running: dict[str, asyncio.Task] = {}

    async def tick(self) -> int:
        """One scan. Returns how many new drive tasks were launched.

        First expire overdue human-asks (a timed-out ask fails its run on the
        next adopt). Then adopt each lease-free non-terminal run — but skip one
        that is still waiting on an OPEN ask, so a run parked on the human does
        not spin; it is re-driven only once the ask is answered (via kick()) or
        expired (no longer open)."""
        try:
            await self.store.expire_asks()
        except Exception as e:  # noqa: BLE001
            log.warning("bat expire_asks failed", error=str(e))
        launched = 0
        for run in await self.store.adoptable_runs():
            rid = run["id"]
            t = self._running.get(rid)
            if t is not None and not t.done():
                continue
            if run["status"] in ("awaiting_plan", "awaiting_human"):
                open_ask = await self.store.latest_open_ask(rid)
                if open_ask is not None:
                    continue  # still waiting on the human — don't spin
            if not await self.store.claim_lease(rid):
                continue
            task = asyncio.create_task(self._drive_and_release(rid))
            self._running[rid] = task
            launched += 1
        return launched

    async def _drive_and_release(self, run_id: str) -> None:
        status = "error"
        try:
            status = await self.driver.drive(run_id)
        finally:
            try:
                await self.store.release_lease(run_id)
            except Exception:
                pass
            if self._running.get(run_id) is asyncio.current_task():
                self._running.pop(run_id, None)
        # Deliver the result once, after the lease is freed, on a terminal
        # outcome. ('running' is returned only when the driver stood down after
        # losing its lease — not terminal, so no delivery.)
        if _ON_FINISH is not None and status in TERMINAL_STATES:
            try:
                run = await self.store.get_run(run_id)
                if run:
                    await _ON_FINISH(run)
            except Exception as e:  # noqa: BLE001 — delivery must never crash the loop
                log.warning("bat on_finish failed", run_id=run_id, error=str(e))

    async def drain(self, timeout: float = 5.0) -> None:
        tasks = [t for t in self._running.values() if not t.done()]
        if tasks:
            await asyncio.wait(tasks, timeout=timeout)


async def bat_loop(store: BatStore, stop_event: asyncio.Event) -> None:
    """Supervisor loop, started from the FD lifespan. Idle until a real
    attempt_runner is registered (Phase 3)."""
    global _SUPERVISOR
    set_store(store)
    sup = BatSupervisor(store)
    _SUPERVISOR = sup
    log.info("bat supervisor loop started")
    while not stop_event.is_set():
        if handlers_ready():
            try:
                await sup.tick()
            except Exception as e:  # noqa: BLE001
                log.warning("bat supervisor tick failed", error=str(e))
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=_LOOP_INTERVAL_S)
        except asyncio.TimeoutError:
            pass
    await sup.drain()
    _SUPERVISOR = None
    log.info("bat supervisor loop stopped")
