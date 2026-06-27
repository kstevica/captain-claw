"""Dubina coder benchmark — does the scaffolding actually elevate a weak model?

The harness answers the only question that makes "frontier simulation" falsifiable:

    bare cheap tier  <<  cheap tier + Horizon  ≈  the best tier you HAVE
       (baseline)           (treatment)            (ceiling)   at a fraction of its cost

It is **target-agnostic**: the ceiling is whatever tier you point it at today
(`gpt-5.3-codex`, `claude-opus`, …). The day you get Fable 5 / GPT-5.6 max access,
add it to ``model.allowed`` and pass it as the ceiling — nothing else changes.

The coder track benchmarks itself for free: the test suite *is* the objective grader,
so no frontier model, human, or LLM-judge is needed. Each task carries its own files
(tests + any stubs); the harness materializes them into a fresh workspace per run and
scores pass/fail by the test command's exit code.

Run real: ``python -m captain_claw.dubina.benchmark --base gemini-flash
--max gpt-5.3-codex --ceiling gpt-5.3-codex``. Unit-tested with stub providers.
"""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from captain_claw.dubina.coder import (
    SOLUTION_PATH_KEY,
    TEST_COMMAND_KEY,
    WORKSPACE_KEY,
    CoderVerifier,
    Workspace,
    make_coder_generator,
    provider_for_tier_from_config,
    shell_command_runner,
)
from captain_claw.dubina.engine import (
    EngineConfig,
    HorizonEngine,
    Step,
    Tier,
    any_pass_aggregator,
)
from captain_claw.logging import get_logger

log = get_logger(__name__)

# Rough relative cost per tier (cheap draft ≈ 1 … expensive frontier ≈ 8). Shared
# across conditions so the cost column is comparable; Phase 4+ swaps in measured $.
DEFAULT_TIER_COSTS: dict[str, float] = {
    "gemini-flash-lite": 0.5, "gpt-5-nano": 0.5, "gemini-flash": 1.0,
    "gpt-5-mini": 3.0, "claude-haiku": 1.5, "claude-sonnet": 3.0,
    "gpt-5.3-codex": 8.0, "claude-opus": 8.0, "gemini-pro": 6.0,
}


def _cost(tier_id: str, costs: dict[str, float]) -> float:
    return costs.get(tier_id, 1.0)


# ── Tasks & conditions ───────────────────────────────────────────────

@dataclass
class BenchmarkTask:
    id: str
    prompt: str
    files: dict[str, str]              # written into the workspace (tests + stubs)
    test_command: str = "python -m pytest -q"
    solution_path: str = "solution.py"
    test_path: str = ""
    difficulty: str = "medium"         # bucket for the horizon curve


@dataclass
class Condition:
    name: str
    ladder: list[Tier]                 # 1 rung = bare; multi-rung = horizon
    max_step_samples: int = 1
    max_fix_attempts: int = 0


def bare_condition(tier_id: str, costs: dict[str, float] = DEFAULT_TIER_COSTS) -> Condition:
    """No scaffolding: one tier, one shot — the honest baseline / ceiling."""
    return Condition(f"bare:{tier_id}", [Tier(tier_id, _cost(tier_id, costs))], 1, 0)


def horizon_condition(
    ladder_ids: Sequence[str],
    costs: dict[str, float] = DEFAULT_TIER_COSTS,
    *, max_step_samples: int = 3, max_fix_attempts: int = 2, name: str = "horizon",
) -> Condition:
    """Full scaffolding: the escalation ladder with sampling + fix loop."""
    return Condition(name, [Tier(i, _cost(i, costs)) for i in ladder_ids],
                     max_step_samples, max_fix_attempts)


# ── Results ──────────────────────────────────────────────────────────

@dataclass
class TaskResult:
    task_id: str
    condition: str
    passed: bool
    cost: float
    difficulty: str
    rung_reached: int
    tier_used: str | None


@dataclass
class BenchmarkReport:
    results: list[TaskResult]
    conditions: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, dict]:
        """Per condition: pass rate, total/avg cost, n."""
        out: dict[str, dict] = {}
        for cond in self.conditions:
            rows = [r for r in self.results if r.condition == cond]
            n = len(rows)
            passed = sum(1 for r in rows if r.passed)
            total = sum(r.cost for r in rows)
            out[cond] = {
                "n": n, "passed": passed,
                "pass_rate": (passed / n) if n else 0.0,
                "total_cost": total, "avg_cost": (total / n) if n else 0.0,
            }
        return out

    def by_difficulty(self) -> dict[str, dict[str, float]]:
        """Per condition → per difficulty → pass rate (the horizon curve)."""
        diffs = sorted({r.difficulty for r in self.results})
        out: dict[str, dict[str, float]] = {}
        for cond in self.conditions:
            row: dict[str, float] = {}
            for d in diffs:
                rows = [r for r in self.results if r.condition == cond and r.difficulty == d]
                row[d] = (sum(1 for r in rows if r.passed) / len(rows)) if rows else 0.0
            out[cond] = row
        return out

    def render(self) -> str:
        s = self.summary()
        lines = ["## Dubina coder benchmark", "",
                 "| Condition | Pass | Pass rate | Total cost | Avg cost |",
                 "|---|---|---|---|---|"]
        for cond in self.conditions:
            m = s[cond]
            lines.append(
                f"| {cond} | {m['passed']}/{m['n']} | {m['pass_rate']*100:.0f}% | "
                f"{m['total_cost']:.1f} | {m['avg_cost']:.2f} |"
            )
        lines += ["", "### Pass rate by difficulty", "",
                  "| Condition | " + " | ".join(sorted({r.difficulty for r in self.results})) + " |"]
        diffs = sorted({r.difficulty for r in self.results})
        lines.append("|---|" + "|".join("---" for _ in diffs) + "|")
        bd = self.by_difficulty()
        for cond in self.conditions:
            lines.append(f"| {cond} | " + " | ".join(f"{bd[cond][d]*100:.0f}%" for d in diffs) + " |")
        return "\n".join(lines) + "\n" + self._verdict(s)

    def _verdict(self, s: dict[str, dict]) -> str:
        """One-line takeaway: elevation over the cheap baseline + cost vs. ceiling."""
        bare = [c for c in self.conditions if c.startswith("bare:")]
        horizon = [c for c in self.conditions if not c.startswith("bare:")]
        if not horizon or not bare:
            return ""
        h = horizon[0]
        cheap, ceiling = bare[0], bare[-1]
        elev = (s[h]["pass_rate"] - s[cheap]["pass_rate"]) * 100
        ratio = (s[h]["total_cost"] / s[ceiling]["total_cost"]) if s[ceiling]["total_cost"] else 0.0
        return (f"\n**Verdict:** horizon {s[h]['pass_rate']*100:.0f}% vs cheap-bare "
                f"{s[cheap]['pass_rate']*100:.0f}% (+{elev:.0f} pts), ceiling "
                f"{s[ceiling]['pass_rate']*100:.0f}% at {ratio*100:.0f}% of ceiling cost.")


# ── Runner ───────────────────────────────────────────────────────────

async def run_task(
    task: BenchmarkTask, condition: Condition,
    *, provider_factory, runner, workdir_root: Path,
) -> TaskResult:
    base = workdir_root / f"{task.id}__{condition.name.replace(':', '_')}"
    base.mkdir(parents=True, exist_ok=True)
    for rel, content in task.files.items():
        p = base / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)

    config = EngineConfig(
        ladder=condition.ladder, max_step_samples=condition.max_step_samples,
        max_fix_attempts=condition.max_fix_attempts,
    )
    engine = HorizonEngine(
        config, make_coder_generator(provider_factory, Workspace(base)),
        CoderVerifier(runner), aggregator=any_pass_aggregator,
    )
    step = Step(id=task.id, prompt=task.prompt, metadata={
        WORKSPACE_KEY: str(base), TEST_COMMAND_KEY: task.test_command,
        SOLUTION_PATH_KEY: task.solution_path, "test_path": task.test_path,
    })
    result = await engine.run(step)
    out = result.steps[-1] if result.steps else None
    return TaskResult(
        task.id, condition.name, result.passed, result.cost_spent, task.difficulty,
        out.rung_reached if out else -1, out.tier_used if out else None,
    )


async def run_benchmark(
    tasks: Sequence[BenchmarkTask], conditions: Sequence[Condition],
    *, provider_factory, runner, workdir_root: Path,
) -> BenchmarkReport:
    results: list[TaskResult] = []
    for task in tasks:
        for cond in conditions:
            try:
                results.append(await run_task(
                    task, cond, provider_factory=provider_factory,
                    runner=runner, workdir_root=workdir_root,
                ))
            except Exception as e:  # noqa: BLE001 — a broken task shouldn't sink the suite
                log.exception("benchmark task failed", task=task.id, condition=cond.name)
                results.append(TaskResult(task.id, cond.name, False, 0.0,
                                          task.difficulty, -1, None))
                _ = e
    return BenchmarkReport(results, [c.name for c in conditions])


# ── Built-in coder fixtures ──────────────────────────────────────────

CODER_FIXTURES: list[BenchmarkTask] = [
    BenchmarkTask(
        id="add", difficulty="easy", test_path="test_solution.py",
        prompt="Implement add(a, b) returning their sum.",
        files={"test_solution.py":
               "from solution import add\n"
               "def test_add():\n    assert add(2, 3) == 5\n    assert add(-1, 1) == 0\n"},
    ),
    BenchmarkTask(
        id="fizzbuzz", difficulty="medium", test_path="test_solution.py",
        prompt="Implement fizzbuzz(n): list 1..n, 'Fizz' for /3, 'Buzz' for /5, "
               "'FizzBuzz' for both, else the number as a string.",
        files={"test_solution.py":
               "from solution import fizzbuzz\n"
               "def test_fizzbuzz():\n"
               "    r = fizzbuzz(15)\n"
               "    assert r[0] == '1' and r[2] == 'Fizz' and r[4] == 'Buzz' and r[14] == 'FizzBuzz'\n"},
    ),
    BenchmarkTask(
        id="lru", difficulty="hard", test_path="test_solution.py",
        prompt="Implement an LRUCache class with get(key) and put(key, value) and a "
               "fixed capacity, evicting the least-recently-used entry.",
        files={"test_solution.py":
               "from solution import LRUCache\n"
               "def test_lru():\n"
               "    c = LRUCache(2)\n    c.put(1, 1); c.put(2, 2)\n"
               "    assert c.get(1) == 1\n    c.put(3, 3)\n    assert c.get(2) == -1\n"},
    ),
]


# ── CLI ──────────────────────────────────────────────────────────────

async def _main_async(args) -> None:
    import tempfile

    costs = dict(DEFAULT_TIER_COSTS)
    conditions = [
        bare_condition(args.base, costs),
        horizon_condition([args.base, args.max], costs,
                          max_step_samples=args.samples, max_fix_attempts=args.fixes),
        bare_condition(args.ceiling, costs),
    ]
    provider_factory = provider_for_tier_from_config()
    with tempfile.TemporaryDirectory(prefix="dubina-bench-") as tmp:
        report = await run_benchmark(
            CODER_FIXTURES, conditions, provider_factory=provider_factory,
            runner=shell_command_runner, workdir_root=Path(tmp),
        )
    print(report.render())


def main() -> None:
    ap = argparse.ArgumentParser(description="Dubina coder benchmark")
    ap.add_argument("--base", required=True, help="cheap tier (baseline + ladder start)")
    ap.add_argument("--max", required=True, help="ladder ceiling for the horizon condition")
    ap.add_argument("--ceiling", required=True, help="best tier you have (bare ceiling)")
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--fixes", type=int, default=2)
    asyncio.run(_main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
