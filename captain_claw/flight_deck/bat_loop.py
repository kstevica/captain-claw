"""Bat — the stubborn-finisher supervisor + per-run driver.

The driver is the loop that makes Bat *stubborn*: it plans a task into steps,
runs each step through an injected ``attempt_runner`` — several at once where the
plan's dependencies allow — checkpoints every step before and after it runs,
retries failed steps, and keeps going round until an injected ``judge`` says the
task is genuinely done. A refusal from the judge is new work: an injected
``replanner`` turns its findings into fix steps for the next round. Only a serious
condition stops it (owner cancel, LLM-spend cap reached, lost lease, the same
failure surviving fix rounds, no new work to try, round ceiling).

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
import time
import os
import re
import socket
from typing import Any, Awaitable, Callable

import structlog

from captain_claw.flight_deck.bat_store import LEASE_STALE_SECONDS, BatStore, TERMINAL_STATES

log = structlog.get_logger(__name__)

# run dict, step dict -> {ok, output, usd, tokens, produced_file, error}
AttemptRunner = Callable[[dict, dict], Awaitable[dict]]
# run dict -> [{step_key, title, seq}]
Planner = Callable[[dict], Awaitable[list[dict]]]
# run dict, steps -> {done: bool, reason: str}
Judge = Callable[[dict, list[dict]], Awaitable[dict]]
# run dict, steps, verdict, round number -> [{title, brief, after}] (the new work)
Replanner = Callable[[dict, list[dict], dict, int], Awaitable[list[dict]]]

_OUTPUT_CAP = 20_000
_DEFAULT_MAX_STEP_ATTEMPTS = 3
_DEFAULT_MAX_ROUNDS = 8
_DEFAULT_MAX_PARALLEL = 3
_LOOP_INTERVAL_S = 5.0
# How often a driver with work in flight looks up: owner cancel, the lease, spend.
_WATCH_INTERVAL_S = 10.0
# The same deterministic failure surviving this many fix rounds is no progress.
_STALL_FIX_ROUNDS = 2


# The deck's own setting (Console → Settings → Bat), when an admin chose one.
_PARALLEL: int | None = None


def set_max_parallel(n: int | None) -> None:
    """How many steps of a run work at the same time on this deck (None: the
    FD_BAT_MAX_PARALLEL default)."""
    global _PARALLEL
    _PARALLEL = max(1, int(n)) if n else None


def env_max_parallel() -> int:
    try:
        return max(1, int(os.environ.get("FD_BAT_MAX_PARALLEL", _DEFAULT_MAX_PARALLEL)))
    except ValueError:
        return _DEFAULT_MAX_PARALLEL


def _max_parallel() -> int:
    return _PARALLEL or env_max_parallel()


# ── the plan's shape (pure) ───────────────────────────────────────────

def effective_deps(steps: list[dict]) -> dict[str, list[str]]:
    """What each step waits for. A step with no recorded dependencies (a plain
    ordered list, or a run from before dependencies existed) waits for every step
    before it, which is how Bat always ran them."""
    known = {s["step_key"] for s in steps}
    out: dict[str, list[str]] = {}
    for s in steps:
        deps = s.get("depends_on")
        if deps is None:
            deps = [o["step_key"] for o in steps if int(o.get("seq", 0)) < int(s.get("seq", 0))]
        # A step the plan never had can't be waited for: it would block forever.
        out[s["step_key"]] = [d for d in deps if d in known]
    return out


def ready_steps(steps: list[dict], max_attempts: int, holding: set[str] | None = None) -> list[dict]:
    """Steps that can start now: pending or retryable, everything they wait for
    finished. A step behind a failure stays put until a fix round replaces it, and
    so does one behind a step in ``holding``: a step whose own worker is finished
    but whose result is still coming from a run it started."""
    deps = effective_deps(steps)
    finished = {s["step_key"] for s in steps if s["status"] in ("done", "skipped")} - (holding or set())
    out = []
    for s in steps:
        retryable = s["status"] == "failed" and int(s.get("attempt", 0)) < max_attempts
        if (s["status"] == "pending" or retryable) and all(d in finished for d in deps[s["step_key"]]):
            out.append(s)
    return out


def fix_steps(plan: list[dict], round_no: int, start_seq: int) -> list[dict]:
    """Keys and order for a fix round's new steps. ``after`` holds 1-based places in
    this same list; anything else in it is ignored, so a fix round can't wait on a
    step that's gone or on itself."""
    keys = [f"fix-{round_no}-{i + 1}" for i in range(len(plan))]
    out = []
    for i, p in enumerate(plan):
        after = [keys[a - 1] for a in (p.get("after") or []) if isinstance(a, int) and 1 <= a <= i]
        out.append({"step_key": keys[i], "title": str(p.get("title", ""))[:400],
                    "brief": str(p.get("brief", "")), "seq": start_seq + i,
                    "round": round_no, "depends_on": after})
    return out


# A failure that says the step was too big for one worker's turn: it ran out of
# time, of context or of steps.
_SIZE_FAILURE_RE = re.compile(
    r"timed? ?out|time budget|context (?:window|length|size)|maximum context|too (?:long|large|big)"
    r"|exceeds? the (?:available )?context|(?:iteration|step) budget",
    re.IGNORECASE)


def size_failure(error: str | None) -> bool:
    """Did a step fail because it was too big for one worker's turn?"""
    return bool(_SIZE_FAILURE_RE.search(error or ""))


# A worker's chat manners around its report (agent_stuck's step-limit notice, the
# rating hint). Nobody can "reply continue" to a Bat worker, and the fix round, which
# reads the first few hundred characters of each report, often saw only the notice.
_STEP_LIMIT_RE = re.compile(
    r"(?:⏸️?\s*)?I reached this message's step limit[^\n]*?pick up where I left off\.?", re.I)
_RATING_RE = re.compile(r"\n*(?:-{3,}\s*\n)?\U0001f4a1[^\n]*\brate (?:good|bad)\b[^\n]*", re.I)
RAN_OUT = "(Its worker ran out of steps before it finished; this is where it got to.)"


def step_report(text: str | None) -> str:
    """A step's report as Bat keeps it: the step-limit notice said in Bat's words,
    the rating hint dropped."""
    out = _STEP_LIMIT_RE.sub(RAN_OUT, str(text or ""))
    return _RATING_RE.sub("", out).strip()


def rewire_failed(steps: list[dict], new_keys: list[str]) -> tuple[list[str], dict[str, list[str]]]:
    """A fix round takes over from the steps that ran out of attempts. Returns those
    steps (to skip) and the new dependencies of every pending step that waited on
    one, so the rest of the plan carries on once the fix is in."""
    failed = {s["step_key"] for s in steps if s["status"] == "failed"}
    if not failed:
        return [], {}
    deps = effective_deps(steps)
    rewired: dict[str, list[str]] = {}
    for s in steps:
        waits = deps[s["step_key"]]
        if s["status"] == "pending" and failed.intersection(waits):
            rewired[s["step_key"]] = [d for d in waits if d not in failed] + list(new_keys)
    return sorted(failed), rewired


def failure_key(verdict: dict) -> str:
    """What failed deterministically, without the changing details of how."""
    names = verdict.get("failing") or verdict.get("det_criticals") or []
    return "\n".join(sorted(str(n) for n in names))


def progress_mark(verdict: dict) -> int | None:
    """How much the work holds of what's asked in amounts (rows, words), summed;
    None when the judge measured nothing."""
    measured = verdict.get("measured") or {}
    vals = [int(v) for v in measured.values() if isinstance(v, (int, float))]
    return sum(vals) if vals else None


def still_growing(marks: list[int | None]) -> bool:
    """Whether the amounts grew across these rounds: the last measured is more than
    the first. A rule that keeps failing while rows pile up is work in progress, not
    a stall."""
    known = [m for m in marks if m is not None]
    return len(known) >= 2 and known[-1] > known[0]


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
    pending = [s for s in steps if s["status"] not in ("done", "skipped")]
    if pending:
        return {"done": False, "reason": f"{len(pending)} step(s) not done"}
    return {"done": True, "reason": "all steps done"}


_ATTEMPT_RUNNER: AttemptRunner = _default_attempt_runner
_PLANNER: Planner = _default_planner
_JUDGE: Judge = _default_judge
# None = no fix rounds: a refusal with nothing left to retry ends the run.
_REPLANNER: Replanner | None = None
# Runs a Bat worker started under this run. Poll counts what they've spent into
# the run, closes the finished ones, and returns spend still in flight that isn't
# in the run yet (a worker mid-turn). Stop ends the open ones. None = no children.
_POLL_SPEND: Callable[[BatStore, dict], Awaitable[float]] | None = None
_STOP_CHILDREN: Callable[[BatStore, dict], Awaitable[None]] | None = None
# Before a run stops on its token cap: has its owner changed the cap meanwhile?
# (First Mate's factories follow theirs.) True when the run's cap moved.
_CAP_REFRESH: Callable[[dict], Awaitable[bool]] | None = None


def set_cap_refresh(fn: "Callable[[dict], Awaitable[bool]] | None") -> None:
    global _CAP_REFRESH
    _CAP_REFRESH = fn


# Checks a finished step against the run's goal before it counts (bat_course):
# (run, step, started_at, last_try) -> {"redo": bool, "why": str} or None.
_STEP_CHECK: Callable[[dict, dict, float, bool], Awaitable[dict | None]] | None = None


def set_step_check(fn: "Callable[[dict, dict, float, bool], Awaitable[dict | None]] | None") -> None:
    global _STEP_CHECK
    _STEP_CHECK = fn
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


def set_replanner(fn: Replanner | None) -> None:
    global _REPLANNER
    _REPLANNER = fn


def set_spend_watch(poll: "Callable[[BatStore, dict], Awaitable[float]] | None",
                    stop: "Callable[[BatStore, dict], Awaitable[None]] | None") -> None:
    global _POLL_SPEND, _STOP_CHILDREN
    _POLL_SPEND, _STOP_CHILDREN = poll, stop


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
    # A run waiting on the human has no driver to do this.
    if _STOP_CHILDREN is not None:
        try:
            await _STOP_CHILDREN(store, run)
        except Exception as e:  # noqa: BLE001
            log.warning("bat stop children failed", run_id=run_id, error=str(e))
    return True


# Why a stopped run may go on where it stopped: its cap ran out, or its owner
# stopped it. A run that stopped on its own (a step kept failing, the reviewers
# wouldn't pass it) would stop again: that one starts over.
CONTINUABLE = ("llm_token_cap_reached", "llm_usd_cap_reached")


async def why_not_reopen(store: BatStore, run_id: str) -> str:
    """Why this run can't go on where it stopped, or "" when it can."""
    run = await store.get_run(run_id)
    if run is None:
        return "its Bat run is gone"
    if run["status"] not in ("error", "cancelled"):
        return "it didn't stop" if run["status"] != "done" else "it finished"
    if run["status"] == "error" and (run.get("stopped_reason") or "") not in CONTINUABLE:
        return f"it stopped on its own ({run.get('stopped_reason') or 'an error'}), so it would stop again"
    steps = await store.list_steps(run_id)
    if not any(s["status"] not in ("done", "skipped") for s in steps):
        return "it has no step left to do"
    import time
    if run.get("lease_pid") and time.time() - float(run.get("lease_heartbeat_at") or 0.0) < LEASE_STALE_SECONDS:
        return "it is still winding down; try again in a minute"
    return ""


async def reopen_run(store: BatStore, run_id: str, *, token_cap: int | None = None) -> dict:
    """Let a run its cap or its owner stopped go on where it stopped: its finished
    steps stay finished, the step it was on runs again, and ``token_cap`` (the
    run's whole cap in tokens, what it used included; 0 = no cap) replaces its
    old one; None keeps it.
    Raises ValueError with why when it can't (:func:`why_not_reopen`)."""
    why = await why_not_reopen(store, run_id)
    if why:
        raise ValueError(why)
    run = await store.get_run(run_id)
    cfg = dict(run.get("config") or {})
    if token_cap is not None:
        cfg["token_cap"] = max(0, int(token_cap))
    await store.set_config(run_id, cfg)
    demoted = await store.demote_running_steps(run_id)
    await store.set_status(run_id, "running", stopped_reason="")
    note = "continued where it stopped"
    if token_cap is not None:
        note += f", with a cap of {int(token_cap):,} tokens" if token_cap > 0 else ", with no token cap"
    await store.append_event(run_id, "note", note + (f" (re-queued {demoted} step(s))" if demoted else ""))
    try:
        await kick()
    except Exception as e:  # noqa: BLE001 — it starts at the next tick
        log.warning("bat kick after reopen failed", run_id=run_id, error=str(e))
    return await store.get_run(run_id)


def run_parallel(run: dict | None) -> int:
    """A run's own limit on steps at once (``config.max_parallel``), 0 = the deck's."""
    try:
        return max(0, int(((run or {}).get("config") or {}).get("max_parallel") or 0))
    except (TypeError, ValueError):
        return 0


def dissent(verdict: dict) -> str:
    """"1 of 3 reviewers objected: <why>", for a run the panel passed by a majority."""
    votes = (verdict.get("panel") or {}).get("votes") or []
    no = [v for v in votes if v.get("vote") == "disagree" and str(v.get("reason") or "").strip()]
    if not no:
        return ""
    why = " / ".join(str(v["reason"]).strip() for v in no)[:900]
    return f"{len(no)} of {len(votes)} reviewer{'' if len(votes) == 1 else 's'} objected: {why}"


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

    async def _poll_spend(self, run: dict) -> float:
        if _POLL_SPEND is None:
            return 0.0
        try:
            return float(await _POLL_SPEND(self.store, run) or 0.0)
        except Exception as e:  # noqa: BLE001 — a bad count must not stop the run
            log.warning("bat spend poll failed", run_id=run.get("id"), error=str(e))
            return 0.0

    async def _should_stop(self, run_id: str) -> tuple[bool, str]:
        """Owner cancel, lost lease, LLM cap (counting what workers are spending
        right now and the runs they started). Returns (stop, reason)."""
        run = await self.store.get_run(run_id)
        if not run:
            return True, "gone"
        if run["status"] == "cancelled":
            return True, "cancelled"
        if not await self.store.heartbeat_lease(run_id, pid=self._pid, host=self._host):
            return True, "lost_lease"
        live = await self._poll_spend(run)
        run = await self.store.get_run(run_id) or run
        cap = float(run.get("llm_usd_cap") or 0.0)
        if cap > 0 and float(run.get("cumulative_usd") or 0.0) + live >= cap:
            return True, "llm_usd_cap_reached"
        # A cap in tokens (config.token_cap), against what's done and what running turns are using.
        from captain_claw.flight_deck import bat_nested
        tokens = bat_nested.token_cap(run)
        if tokens and int(run.get("cumulative_tokens") or 0) + bat_nested.live_tokens(run_id) >= tokens:
            # Read the cap once more before stopping: the owner may have raised it.
            if _CAP_REFRESH is not None and await _CAP_REFRESH(run):
                run = await self.store.get_run(run_id) or run
                tokens = bat_nested.token_cap(run)
            if tokens and int(run.get("cumulative_tokens") or 0) + bat_nested.live_tokens(run_id) >= tokens:
                return True, "llm_token_cap_reached"
        return False, ""

    async def watch_parked(self, run: dict) -> None:
        """A run waiting on the human has no driver, but runs it started keep
        spending: count them, and end them once the cap is gone. The run itself
        stops at the cap when the answer brings a driver back."""
        await self._poll_spend(run)
        fresh = await self.store.get_run(run["id"]) or run
        cap = float(fresh.get("llm_usd_cap") or 0.0)
        from captain_claw.flight_deck import bat_nested
        tokens = bat_nested.token_cap(fresh)
        if (cap > 0 and float(fresh.get("cumulative_usd") or 0.0) >= cap) or (
                tokens and int(fresh.get("cumulative_tokens") or 0) >= tokens):
            await self._stop_children(fresh)

    async def _stop_children(self, run: dict) -> None:
        if _STOP_CHILDREN is None:
            return
        try:
            await _STOP_CHILDREN(self.store, run)
        except Exception as e:  # noqa: BLE001
            log.warning("bat stop children failed", run_id=run.get("id"), error=str(e))

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

        # A round is one judgement of the work so far. The count survives a restart,
        # so a run that keeps getting restarted still hits the ceiling.
        rounds = int(((await self.store.get_run(run_id)) or {}).get("config", {}).get("rounds") or 0)
        while True:
            rounds += 1
            outcome, reason = await self._run_wave(run_id)
            if outcome == "stop":
                return await self._stop(run_id, reason, rounds)
            if outcome == "parked":
                # A step needs a human. The answer re-kicks the run and _GATE_CHECK
                # resumes it.
                fresh = await self.store.get_run(run_id)
                return fresh["status"] if fresh else "error"

            steps = await self.store.list_steps(run_id)
            run = await self.store.get_run(run_id)
            try:
                verdict = await self.judge(run, steps)
            except Exception as e:  # noqa: BLE001
                verdict = {"done": False, "reason": f"judge_error: {e}"}

            # The judge takes a while, and a cancel that came meanwhile wins.
            cur = await self.store.get_run(run_id)
            if cur and cur["status"] == "cancelled":
                return await self._stop(run_id, "cancelled", rounds)

            if verdict.get("parked"):
                # The judge asked the owner to settle it. The answer resumes the run.
                fresh = await self.store.get_run(run_id)
                return fresh["status"] if fresh else "error"

            if verdict.get("done"):
                await self.store.append_event(run_id, "done", verdict.get("reason", "done"))
                truth = _assemble_truth(steps)
                if verdict.get("overridden"):
                    truth = f"> Reviewers still objected when this run ended: {verdict.get('reason', '')}\n\n{truth}"
                else:
                    said = dissent(verdict)
                    if said:  # passed by a majority: the one who said no is heard, not dropped
                        truth = f"> {said}\n\n{truth}"
                await self._finish(
                    run_id, "done", truth=truth,
                    analysis={"rounds": rounds, "verdict": verdict},
                )
                return "done"

            reason = verdict.get("reason", "not_done")
            await self.store.append_event(run_id, "judge", f"not done yet (round {rounds}): {reason}", ok=False)
            stop_why = await self._replan(run, steps, verdict, rounds)
            if stop_why:
                await self.store.append_event(run_id, "error", f"stopped: {stop_why} (round {rounds})")
                await self._finish(
                    run_id, "error", truth=_assemble_truth(steps),
                    reason=stop_why, analysis={"rounds": rounds, "verdict": verdict},
                )
                return "error"
            # New work is in the plan. Work the plan didn't have may need the owner.
            if _GATE_CHECK is not None:
                decision = await _GATE_CHECK(self.store, await self.store.get_run(run_id))
                if decision in ("wait", "cancelled", "error"):
                    fresh = await self.store.get_run(run_id)
                    return fresh["status"] if fresh else "error"

    async def _replan(self, run: dict, steps: list[dict], verdict: dict, rounds: int) -> str:
        """Turn a refusal into the next round's steps. Returns why the run has to
        stop instead, or "" when there's new work."""
        run_id = run["id"]
        reason = verdict.get("reason", "not_done")
        if _REPLANNER is None:
            return reason
        if rounds >= self.max_rounds:
            return f"{reason} (no more rounds: {self.max_rounds})"
        cfg = dict(run.get("config") or {})
        seen = list(cfg.get("failure_keys") or [])
        key = failure_key(verdict)
        seen.append(key)
        # One mark per judged round, beside its key (a run from before marks has none).
        marks = list(cfg.get("progress_marks") or [])
        marks = [None] * max(0, len(seen) - 1 - len(marks)) + marks[-(len(seen) - 1):] if len(seen) > 1 else []
        marks.append(progress_mark(verdict))
        window = _STALL_FIX_ROUNDS + 1
        if key and seen[-window:] == [key] * window and not still_growing(marks[-window:]):
            return f"no progress: the same checks failed after {_STALL_FIX_ROUNDS} fix rounds: {reason}"
        try:
            plan = await _REPLANNER(run, steps, verdict, rounds)
        except Exception as e:  # noqa: BLE001
            log.warning("bat replan failed", run_id=run_id, error=str(e))
            plan = []
        if not plan:
            return reason
        # The round is on record before its steps, so a crash between the two can't
        # reuse a step key. Read fresh: the judge writes to the config too.
        cfg = dict((await self.store.get_run(run_id) or run).get("config") or {})
        cfg["rounds"] = rounds
        cfg["failure_keys"] = seen
        cfg["progress_marks"] = marks
        await self.store.set_config(run_id, cfg)
        start = max((int(s.get("seq", 0)) for s in steps), default=-1) + 1
        new = fix_steps(plan, rounds, start)
        skip, rewired = rewire_failed(steps, [s["step_key"] for s in new])
        await self.store.seed_steps(run_id, new)
        for k in skip:
            await self.store.upsert_step(run_id, k, status="skipped")
        for k, deps in rewired.items():
            await self.store.upsert_step(run_id, k, depends_on=deps)
        await self.store.append_event(
            run_id, "plan", f"round {rounds + 1}: {len(new)} new step(s) to fix what was found",
            steps=[s["step_key"] for s in new])
        return ""

    async def _run_wave(self, run_id: str) -> tuple[str, str]:
        """Run everything that can run, up to the parallel limit, until nothing is
        running and nothing more can start. Looks up every few seconds, so a cancel
        or a spent budget ends work in flight instead of waiting for it. Returns
        ("idle", ""), ("parked", "") when a step needs a human, or ("stop", reason)."""
        running: dict[asyncio.Task, str] = {}
        parked = False
        held: int | None = None  # what this deck had room for when a worker couldn't start
        try:
            while True:
                stop, reason = await self._should_stop(run_id)
                if stop:
                    await self._cancel_running(run_id, running)
                    return "stop", reason
                cur = await self.store.get_run(run_id)
                parked = parked or bool(cur and cur["status"] in ("awaiting_human", "awaiting_plan"))
                if not parked:
                    busy = set(running.values())
                    # Read each time round: an admin's change applies to a run going now.
                    limit = _max_parallel() if held is None else min(held, _max_parallel())
                    own = run_parallel(cur)
                    if own:  # this run's own limit (one model: one step at a time)
                        limit = min(limit, own)
                    room = limit - len(running)
                    holding = ({c["step_key"] for c in await self.store.list_children(run_id, open_only=True)}
                               if _POLL_SPEND is not None else set())
                    for step in ready_steps(await self.store.list_steps(run_id), self.max_step_attempts, holding):
                        if room <= 0:
                            break
                        if step["step_key"] not in busy:
                            running[asyncio.create_task(self._run_step(run_id, step))] = step["step_key"]
                            room -= 1
                if not running:
                    if parked:
                        return "parked", ""
                    # A run a worker started still counts as this round's work: the
                    # judge should see its result, not a promise of one.
                    if _POLL_SPEND is None or not await self.store.list_children(run_id, open_only=True):
                        return "idle", ""
                    await asyncio.sleep(_WATCH_INTERVAL_S)
                    continue
                done, _ = await asyncio.wait(running, timeout=_WATCH_INTERVAL_S,
                                             return_when=asyncio.FIRST_COMPLETED)
                for t in done:
                    running.pop(t, None)
                    outcome = t.result()
                    parked = parked or outcome == "parked"
                    if outcome == "held":
                        held = max(1, len(running))  # this deck has no room for that many
        except BaseException:
            # Going down (a restart, or a bug): no worker runs on with nobody watching.
            await self._cancel_running(run_id, running, demote=False)
            raise

    async def _cancel_running(self, run_id: str, running: dict[asyncio.Task, str],
                              *, demote: bool = True) -> None:
        for t in running:
            t.cancel()
        if running:
            await asyncio.gather(*running, return_exceptions=True)
            if demote:
                await self.store.demote_running_steps(run_id)
        running.clear()

    async def _ask_is_for(self, run_id: str, key: str) -> bool:
        """Whether the run is parked on a question from this step. A worker asks
        through a tool and then ends its turn, which reads as a finished attempt."""
        ask = await self.store.latest_open_ask(run_id)
        if not ask:
            return False
        asked_by = ask.get("step_key") or ""
        if asked_by.startswith("spend:"):
            spend = await self.store.get_spend(asked_by.split("spend:", 1)[-1])
            asked_by = (spend or {}).get("agent") or ""
        return asked_by in ("", key)

    async def _run_step(self, run_id: str, step: dict) -> str:
        """Run one step attempt. Returns "parked" if the run stood down to wait for
        a human (the step asked for input), "held" if the step never got a worker
        while others are running and should wait its turn, "" otherwise."""
        key = step["step_key"]
        attempt = int(step["attempt"]) + 1
        started_at = time.time()
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
            return "parked"
        if res.get("spawn_refused") and any(
                s["status"] == "running" and s["step_key"] != key for s in await self.store.list_steps(run_id)):
            # Not this step's failure: the deck had no worker for it right now.
            await self.store.upsert_step(run_id, key, status="pending", attempt=attempt - 1)
            await self.store.append_event(run_id, "note", "waiting for room for another worker", agent=key)
            return "held"
        usd = float(res.get("usd") or 0.0)
        tokens = int(res.get("tokens") or 0)
        if usd or tokens:
            await self.store.bump_cost(run_id, usd, tokens)
        cur = await self.store.get_run(run_id)
        if cur and cur["status"] in ("awaiting_human", "awaiting_plan") and await self._ask_is_for(run_id, key):
            # The step isn't finished, it's waiting: it runs again with the answer.
            await self.store.upsert_step(run_id, key, status="pending")
            return "parked"
        if res.get("ok") and _STEP_CHECK is not None:
            # Before it counts: is what it made on course for the goal? A broken rule
            # sends it round again with why (the last try is kept, with the finding).
            try:
                check = await _STEP_CHECK(await self.store.get_run(run_id) or run,
                                          {**step, "attempt": attempt, "output": str(res.get("output", ""))},
                                          started_at, attempt >= self.max_step_attempts)
            except Exception as e:  # noqa: BLE001 — a check that breaks never fails a step
                log.warning("bat step check failed", run_id=run_id, step=key, error=str(e))
                check = None
            if check and check.get("redo"):
                res = {"ok": False, "error": check.get("error")
                       or f"Captain checked this step's work and it isn't on course: {check.get('why', '')}"}
        if res.get("ok"):
            await self.store.upsert_step(
                run_id, key, status="done",
                output=str(res.get("output", ""))[:_OUTPUT_CAP],
                produced_file=str(res.get("produced_file", "")), error="",
            )
            await self.store.append_event(run_id, "step", step.get("title") or key,
                                          agent=key, ok=True, usd=usd)
        else:
            error = str(res.get("error", ""))
            fields: dict[str, Any] = {"status": "failed", "error": error[:_OUTPUT_CAP]}
            if size_failure(error) and size_failure(step.get("error")):
                # Too big for one turn twice in a row: a third try would go the same
                # way. The fix round takes it over and splits what's left of it.
                fields["attempt"] = max(self.max_step_attempts, attempt)
                await self.store.append_event(run_id, "note", "too big for one turn twice: the fix round "
                                              "splits what's left of it into smaller steps", agent=key)
            await self.store.upsert_step(run_id, key, **fields)
            await self.store.append_event(run_id, "step", step.get("title") or key,
                                          agent=key, ok=False,
                                          detail=str(res.get("error", ""))[:200])
        return ""

    async def _stop(self, run_id: str, reason: str, rounds: int) -> str:
        if reason not in ("lost_lease", "gone"):
            await self._stop_children(await self.store.get_run(run_id) or {"id": run_id})
        if reason == "cancelled":
            await self.store.append_event(run_id, "cancelled", "stopped by owner")
            # status already 'cancelled'
            return "cancelled"
        if reason == "lost_lease":
            await self.store.append_event(run_id, "note", "lost lease — standing down")
            return "running"  # left non-terminal for re-adoption
        if reason == "gone":
            return "gone"
        # llm_usd_cap_reached / llm_token_cap_reached (serious problem): stop, keep work, resumable
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
        # One scan at a time. kick() scans at once, and runs started together
        # (a Captain Swarm wave) kick together: overlapping scans would each see a
        # run as undriven, and this process's own lease doesn't stop it twice.
        self._scan = asyncio.Lock()

    async def tick(self) -> int:
        """One scan. Returns how many new drive tasks were launched.

        First expire overdue human-asks (a timed-out ask fails its run on the
        next adopt). Then adopt each lease-free non-terminal run — but skip one
        that is still waiting on an OPEN ask, so a run parked on the human does
        not spin; it is re-driven only once the ask is answered (via kick()) or
        expired (no longer open)."""
        async with self._scan:
            return await self._tick()

    async def _tick(self) -> int:
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
                    await self.driver.watch_parked(run)
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
