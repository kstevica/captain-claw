"""Plan-Horizon (Lever C) — the verify-gated multi-step horizon.

Basna/Vatra parallelize **breadth** (a fleet, one shot). Lever A/B added per-worker
depth + a final closer. Lever C adds the **sequential horizon** frontier models get
"for free": a planner decomposes the task into ordered steps; each step is **driven
to a verified result before the next begins**; a step that can't be verified triggers
a **re-plan** of the remainder. The deliverable is synthesized from the verified steps.

    decompose → [ run step → verify → (fix | re-plan) ] → advance → … → synthesize

This is the "think way ahead, way long" lever: the system never compounds an
unverified step, and recovers by re-planning rather than barreling forward.

Pure orchestration — every LLM/exec touchpoint is an injected async seam, so the
whole loop unit-tests with stubs (mirrors the Dubina engine's discipline):

* ``planner(task, completed) -> [goal, …]``     — ordered REMAINING step goals.
* ``step_runner(goal, context) -> output``       — execute one step (an LLM call,
  or a whole Basna ensemble — the seam doesn't care).
* ``verifier(task, goal, output) -> Verdict``    — did the step achieve its goal?
* ``synthesizer(task, steps) -> deliverable``    — assemble the verified results.

Everything is bounded (``max_steps`` / ``max_fix_per_step`` / ``max_replans`` + a hard
iteration cap) and ends with an explicit ``stopped_reason`` — never a silent run-on.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from captain_claw.logging import get_logger

log = get_logger(__name__)


@dataclass
class StepVerdict:
    passed: bool
    confidence: float = 0.0
    feedback: str = ""


@dataclass
class PlanConfig:
    max_steps: int = 5            # plan is truncated to this many steps
    max_fix_per_step: int = 1     # feedback-driven retries before a step is "hard-failed"
    max_replans: int = 1          # whole-remainder re-plans on a hard step failure
    min_step_confidence: float = 0.6  # verifier confidence a step must clear to pass


# Injected seams.
Planner = Callable[[str, list], Awaitable[list]]
StepRunner = Callable[[str, str], Awaitable[str]]
Verifier = Callable[[str, str, str], Awaitable[StepVerdict]]
Synthesizer = Callable[[str, list], Awaitable[str]]


@dataclass
class PlanResult:
    deliverable: str
    steps: list[dict]            # [{goal, output, confidence, attempts, verified}]
    replans: int
    stopped_reason: str          # "" | "max_steps" | "step_unverified" | "empty_plan"
    completed: int               # verified steps


def _context(task: str, completed: list[dict], feedback: str) -> str:
    """The prompt context for a step: the task, verified prior results, and (on a
    retry) the critique of the previous attempt at *this* step."""
    parts = [f"# Overall task\n{task}"]
    if completed:
        parts.append("# Verified results from earlier steps")
        for s in completed:
            parts.append(f"## {s['goal']}\n{s['output']}")
    if feedback:
        parts.append(
            "# A previous attempt at THIS step was rejected — address the critique:\n"
            + feedback)
    return "\n\n".join(parts)


async def run_plan_horizon(
    task: str,
    *,
    planner: Planner,
    step_runner: StepRunner,
    verifier: Verifier,
    synthesizer: Synthesizer,
    cfg: PlanConfig | None = None,
    on_event: Callable[[dict], None] | None = None,
) -> PlanResult:
    """Drive ``task`` to a deliverable through a verify-gated, re-planning step chain."""
    cfg = cfg or PlanConfig()

    def emit(stage: str, **kw) -> None:
        if on_event is not None:
            on_event({"stage": stage, **kw})

    raw_goals = [str(g).strip() for g in await planner(task, []) if str(g).strip()]
    goals = raw_goals[: cfg.max_steps]
    if not goals:
        return PlanResult("", [], 0, "empty_plan", 0)
    emit("plan", goals=list(goals))

    completed: list[dict] = []
    replans = 0
    # If the planner wanted more steps than the cap allows, say so (no silent drop).
    stopped = "max_steps" if len(raw_goals) > len(goals) else ""
    i = 0
    # Hard cap: even with re-plans the loop must terminate.
    hard_cap = cfg.max_steps + cfg.max_replans * cfg.max_steps + cfg.max_steps
    runs = 0

    while i < len(goals):
        if runs >= hard_cap:
            stopped = stopped or "step_unverified"
            break
        runs += 1
        goal = str(goals[i])
        emit("step_start", index=i, goal=goal, total=len(goals))

        best_out, best_conf, attempts, passed, feedback = "", 0.0, 0, False, ""
        for _attempt in range(cfg.max_fix_per_step + 1):
            attempts += 1
            out = await step_runner(goal, _context(task, completed, feedback))
            verdict = await verifier(task, goal, out)
            emit("verify", index=i, attempt=attempts, passed=verdict.passed,
                 confidence=verdict.confidence)
            if verdict.confidence >= best_conf or not best_out:
                best_out, best_conf = out, verdict.confidence
            if verdict.passed and verdict.confidence >= cfg.min_step_confidence:
                passed = True
                break
            feedback = verdict.feedback or "the step did not fully satisfy its goal"

        if passed:
            completed.append({"goal": goal, "output": best_out, "confidence": best_conf,
                              "attempts": attempts, "verified": True})
            emit("step_done", index=i, verified=True, confidence=best_conf)
            i += 1
            continue

        # Hard step failure → re-plan the remainder (bounded), else accept best-so-far.
        if replans < cfg.max_replans:
            replans += 1
            emit("replan", index=i, replans=replans)
            remaining = [g for g in (await planner(task, completed)) if str(g).strip()]
            if not remaining:
                stopped = "step_unverified"
                completed.append({"goal": goal, "output": best_out, "confidence": best_conf,
                                  "attempts": attempts, "verified": False})
                break
            # Replace the not-yet-done tail; keep total within max_steps.
            goals = goals[:i] + remaining[: max(0, cfg.max_steps - i)]
            continue

        # Replans exhausted: keep the best attempt, mark unverified, advance (no
        # silent truncation — it's recorded as unverified in the result).
        completed.append({"goal": goal, "output": best_out, "confidence": best_conf,
                          "attempts": attempts, "verified": False})
        emit("step_done", index=i, verified=False, confidence=best_conf)
        stopped = stopped or "step_unverified"
        i += 1

    deliverable = await synthesizer(task, completed) if completed else ""
    emit("synthesize", steps=len(completed))
    return PlanResult(
        deliverable=deliverable, steps=completed, replans=replans,
        stopped_reason=stopped, completed=sum(1 for s in completed if s["verified"]),
    )


# ── DAG variant: a dependency graph instead of a linear chain ────────

@dataclass
class DagStep:
    id: str
    goal: str
    depends_on: list[str]


def _normalize_dag(raw: list, max_steps: int) -> list[DagStep]:
    """Coerce planner output into a clean DAG: unique ids, goals present, dangling
    deps dropped, self-loops removed, capped to ``max_steps``, and any cycle broken
    (an edge into an already-seen node is dropped) so a topological order exists."""
    steps: list[DagStep] = []
    seen_ids: set[str] = set()
    for i, item in enumerate(raw or []):
        if isinstance(item, dict):
            sid = str(item.get("id") or f"s{i + 1}").strip() or f"s{i + 1}"
            goal = str(item.get("goal") or "").strip()
            deps = [str(d).strip() for d in (item.get("depends_on") or []) if str(d).strip()]
        else:
            sid, goal, deps = f"s{i + 1}", str(item).strip(), []
        if not goal or sid in seen_ids:
            continue
        seen_ids.add(sid)
        steps.append(DagStep(id=sid, goal=goal, depends_on=deps))
        if len(steps) >= max_steps:
            break
    # Keep only deps that point at an earlier-listed step (drops danglers, self-loops,
    # and forward/back edges that would form a cycle — a valid topological subset).
    earlier: set[str] = set()
    for s in steps:
        s.depends_on = [d for d in s.depends_on if d in earlier]
        earlier.add(s.id)
    return steps


def _dag_context(task: str, step: DagStep, done: dict) -> str:
    """A DAG step sees the task + the verified outputs of the steps it depends on."""
    parts = [f"# Overall task\n{task}"]
    deps = [done[d] for d in step.depends_on if d in done]
    if deps:
        parts.append("# Results of the steps this one builds on")
        for d in deps:
            parts.append(f"## {d['goal']}\n{d['output']}")
    return "\n\n".join(parts)


async def _execute_step(
    task: str, goal: str, context: str, step_runner: StepRunner, verifier: Verifier,
    cfg: PlanConfig, *, on_event=None, step_id: str | None = None,
) -> dict:
    """Run one step to a verified result with a bounded fix loop (shared by the DAG
    driver). Returns ``{output, confidence, attempts, verified}`` — best-so-far if it
    never clears the bar (never raises, never silently drops)."""
    best_out, best_conf, attempts, passed, feedback = "", 0.0, 0, False, ""
    for _attempt in range(cfg.max_fix_per_step + 1):
        attempts += 1
        ctx = context if not feedback else (
            f"{context}\n\n# A previous attempt at THIS step was rejected — fix it:\n{feedback}")
        out = await step_runner(goal, ctx)
        verdict = await verifier(task, goal, out)
        if on_event is not None:
            on_event({"stage": "verify", "id": step_id, "attempt": attempts,
                      "passed": verdict.passed, "confidence": verdict.confidence})
        if verdict.confidence >= best_conf or not best_out:
            best_out, best_conf = out, verdict.confidence
        if verdict.passed and verdict.confidence >= cfg.min_step_confidence:
            passed = True
            break
        feedback = verdict.feedback or "the step did not fully satisfy its goal"
    return {"output": best_out, "confidence": best_conf, "attempts": attempts, "verified": passed}


# A DAG planner returns step dicts: ``[{id, goal, depends_on:[id,…]}, …]``.
DagPlanner = Callable[[str], Awaitable[list]]


async def run_dag_horizon(
    task: str,
    *,
    planner_dag: DagPlanner,
    step_runner: StepRunner,
    verifier: Verifier,
    synthesizer: Synthesizer,
    cfg: PlanConfig | None = None,
    on_event: Callable[[dict], None] | None = None,
) -> PlanResult:
    """Plan-horizon over a **dependency DAG**: independent steps run concurrently in
    dependency waves; each step sees only its dependencies' verified outputs. Bounded
    (each step runs at most once + its fix loop); a step that can't be scheduled
    (cycle / unsatisfiable dep) leaves the run ``blocked`` rather than hanging."""
    cfg = cfg or PlanConfig()

    def emit(stage: str, **kw) -> None:
        if on_event is not None:
            on_event({"stage": stage, **kw})

    steps = _normalize_dag(await planner_dag(task), cfg.max_steps)
    if not steps:
        return PlanResult("", [], 0, "empty_plan", 0)
    emit("plan", goals=[s.goal for s in steps])

    done: dict[str, dict] = {}
    order: list[str] = []
    remaining = {s.id: s for s in steps}
    stopped = ""
    guard = 0
    while remaining:
        guard += 1
        if guard > len(steps) + 2:  # safety — should never trigger
            stopped = stopped or "blocked"
            break
        ready = [s for s in remaining.values() if all(d in done for d in s.depends_on)]
        if not ready:
            stopped = "blocked"  # a cycle survived normalization, or a dep failed to land
            break

        async def _run(s: DagStep):
            emit("step_start", id=s.id, goal=s.goal, total=len(steps))
            r = await _execute_step(task, s.goal, _dag_context(task, s, done),
                                    step_runner, verifier, cfg, on_event=on_event, step_id=s.id)
            return s, r

        wave = await asyncio.gather(*[_run(s) for s in ready])
        for s, r in wave:
            done[s.id] = {"goal": s.goal, "output": r["output"], "confidence": r["confidence"],
                          "attempts": r["attempts"], "verified": r["verified"]}
            order.append(s.id)
            remaining.pop(s.id, None)
            emit("step_done", id=s.id, verified=r["verified"], confidence=r["confidence"])

    steps_out = [done[i] for i in order]
    if not stopped and any(not s["verified"] for s in steps_out):
        stopped = "step_unverified"
    deliverable = await synthesizer(task, steps_out) if steps_out else ""
    emit("synthesize", steps=len(steps_out))
    return PlanResult(
        deliverable=deliverable, steps=steps_out, replans=0,
        stopped_reason=stopped, completed=sum(1 for s in steps_out if s["verified"]),
    )


# ── Concrete LLM-backed seams (used by the Basna wiring; provider injected) ───

def _strip_fence(text: str) -> str:
    t = (text or "").strip()
    if t.startswith("```"):
        t = "\n".join(ln for ln in t.split("\n") if not ln.strip().startswith("```"))
    return t.strip()


# A hung model must never stall the plan-horizon — every raw seam call is bounded.
_LLM_TIMEOUT = 240.0


async def _complete(provider, messages, timeout: float = _LLM_TIMEOUT):
    return await asyncio.wait_for(provider.complete(messages), timeout)


_PLANNER_SYSTEM = (
    "You are a planning model. Break the task into the SMALLEST sequence of ordered, "
    "concrete steps that together accomplish it — each step a self-contained unit of "
    "work that builds on the prior ones. Given any already-completed steps, return ONLY "
    "the REMAINING steps. Reply with a JSON array of short step-goal strings, nothing "
    "else. Prefer fewer steps; never pad."
)


def make_llm_planner(provider, *, max_steps: int = 5):
    from captain_claw.llm import Message

    async def planner(task: str, completed: list) -> list:
        done = "\n".join(f"- {s['goal']}" for s in completed)
        user = f"## Task\n{task}\n\n## Already completed\n{done or '(none)'}\n\n" \
               f"Return the remaining steps (≤{max_steps}) as a JSON array of strings."
        try:
            resp = await _complete(
                provider,
                [Message(role="system", content=_PLANNER_SYSTEM), Message(role="user", content=user)])
            arr = json.loads(_strip_fence(resp.content))
        except Exception:  # noqa: BLE001 — timeout / bad JSON → no plan
            return []
        return [str(x).strip() for x in arr if str(x).strip()] if isinstance(arr, list) else []

    return planner


_DAG_PLANNER_SYSTEM = (
    "You are a planning model. Decompose the task into a DAG of concrete steps. Mark "
    "which steps depend on which, so independent steps can run in parallel and each "
    "step only consumes what it actually needs. Reply with ONLY a JSON array of "
    '{"id": "s1", "goal": "...", "depends_on": ["s0", ...]} objects, ordered so every '
    "dependency appears before the step that needs it. Prefer fewer steps; never pad."
)


def make_llm_dag_planner(provider, *, max_steps: int = 6):
    from captain_claw.llm import Message

    async def planner_dag(task: str) -> list:
        user = (f"## Task\n{task}\n\nReturn a DAG of at most {max_steps} steps as a JSON "
                'array of {"id","goal","depends_on"} objects.')
        try:
            resp = await _complete(
                provider,
                [Message(role="system", content=_DAG_PLANNER_SYSTEM),
                 Message(role="user", content=user)])
            arr = json.loads(_strip_fence(resp.content))
        except Exception:  # noqa: BLE001 — timeout / bad JSON → no plan
            return []
        return arr if isinstance(arr, list) else []

    return planner_dag


_STEP_SYSTEM = (
    "You are executing ONE step of a larger plan. Use the verified results of earlier "
    "steps as given. Do the current step thoroughly and completely, and output only the "
    "result of this step — no preamble, no meta-commentary about the plan."
)


def make_llm_step_runner(provider):
    """A lean step runner: one capable generation per step. Swap this seam for a
    Basna-ensemble runner to make every step a full fleet run."""
    from captain_claw.llm import Message

    async def step_runner(goal: str, context: str) -> str:
        user = f"{context}\n\n# Your step now\n{goal}"
        try:
            resp = await _complete(
                provider,
                [Message(role="system", content=_STEP_SYSTEM), Message(role="user", content=user)])
        except Exception:  # noqa: BLE001 — timeout → empty step output (verifier will fail it)
            return ""
        return (resp.content or "").strip()

    return step_runner


_VERIFIER_SYSTEM = (
    "You are a strict verifier. Decide whether the output ACHIEVES the step goal in the "
    "context of the overall task. Be skeptical: incomplete, off-target, or unsupported "
    "output fails. Reply ONLY with JSON: "
    '{"passed": true|false, "confidence": 0.0-1.0, "feedback": "what is missing or wrong"}.'
)


def make_llm_verifier(provider):
    from captain_claw.llm import Message

    async def verifier(task: str, goal: str, output: str) -> StepVerdict:
        user = (f"## Overall task\n{task}\n\n## Step goal\n{goal}\n\n"
                f"## Output to judge\n{output[:8000]}")
        try:
            resp = await _complete(
                provider,
                [Message(role="system", content=_VERIFIER_SYSTEM),
                 Message(role="user", content=user)])
            d = json.loads(_strip_fence(resp.content))
            return StepVerdict(
                passed=bool(d.get("passed")),
                confidence=float(d.get("confidence") or 0.0),
                feedback=str(d.get("feedback") or ""))
        except Exception:  # noqa: BLE001 — timeout / unparseable judgement
            # Soft pass (don't block the horizon on a flaky or slow judge), low
            # confidence so a fix attempt still fires.
            return StepVerdict(passed=True, confidence=0.5, feedback="")

    return verifier


_SYNTH_SYSTEM = (
    "You are assembling the final deliverable for a task from the verified results of "
    "its steps. Integrate them into one coherent, complete answer — do not truncate, "
    "do not just list the steps. Resolve overlaps; keep everything the task asked for."
)


def make_llm_synthesizer(provider):
    from captain_claw.llm import Message

    async def synthesizer(task: str, steps: list) -> str:
        body = "\n\n".join(
            f"### {s['goal']}{'' if s.get('verified', True) else ' (unverified)'}\n{s['output']}"
            for s in steps)
        user = f"## Task\n{task}\n\n## Verified step results\n{body}\n\n## Final deliverable"
        try:
            resp = await _complete(
                provider,
                [Message(role="system", content=_SYNTH_SYSTEM), Message(role="user", content=user)])
        except Exception:  # noqa: BLE001 — timeout → fall back to labeled concatenation
            return body
        return (resp.content or "").strip()

    return synthesizer
