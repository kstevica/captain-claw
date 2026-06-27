"""Tests for the Dubina coder benchmark harness (Phase 4).

Stub providers + a stub test runner prove the harness mechanics and that it shows
the elevation story: horizon beats cheap-bare, ceiling-bare also passes, costs are
comparable across conditions, and the difficulty buckets aggregate.
"""

from __future__ import annotations

from pathlib import Path

from captain_claw.dubina.benchmark import (
    BenchmarkTask,
    bare_condition,
    horizon_condition,
    run_benchmark,
)
from captain_claw.llm import LLMResponse

COSTS = {"cheap": 1.0, "mid": 3.0, "ceiling": 8.0}


class StubProvider:
    def __init__(self, first: str, on_fix: str | None = None):
        self.first, self.on_fix = first, on_fix

    async def complete(self, messages, tools=None, temperature=None, max_tokens=None):
        user = messages[-1].content
        if self.on_fix is not None and "FAILED" in user:
            return LLMResponse(content=self.on_fix, finish_reason="stop")
        return LLMResponse(content=self.first, finish_reason="stop")


GOOD = "```python\ndef add(a, b): return a + b  # CORRECT\n```"
BAD = "```python\ndef add(a, b): return 0\n```"


def tiered_factory():
    """Ceiling one-shots it; the cheap tier only succeeds once it gets fix feedback."""
    def f(tier: str):
        if tier == "ceiling":
            return StubProvider(first=GOOD)
        return StubProvider(first=BAD, on_fix=GOOD)
    return f


def runner_passes_on(token: str):
    async def run(command: str, cwd: str) -> tuple[bool, str]:
        code = (Path(cwd) / "solution.py").read_text()
        return (token in code), ("1 passed" if token in code else "AssertionError\n1 failed")
    return run


def tasks():
    mk = lambda i, d: BenchmarkTask(  # noqa: E731
        id=i, difficulty=d, prompt="implement add", test_path="test_solution.py",
        files={"test_solution.py": "from solution import add\n"},
    )
    return [mk("add1", "easy"), mk("add2", "hard")]


async def _run(tmp_path):
    conditions = [
        bare_condition("cheap", COSTS),
        horizon_condition(["cheap", "ceiling"], COSTS, max_step_samples=1, max_fix_attempts=2),
        bare_condition("ceiling", COSTS),
    ]
    return await run_benchmark(
        tasks(), conditions,
        provider_factory=tiered_factory(), runner=runner_passes_on("CORRECT"),
        workdir_root=tmp_path,
    )


async def test_horizon_elevates_over_cheap_bare(tmp_path):
    report = await _run(tmp_path)
    s = report.summary()
    assert s["bare:cheap"]["pass_rate"] == 0.0     # one shot, no fix → fails
    assert s["horizon"]["pass_rate"] == 1.0        # fix loop rescues it
    assert s["bare:ceiling"]["pass_rate"] == 1.0   # strong model one-shots it


async def test_cost_columns_are_comparable(tmp_path):
    report = await _run(tmp_path)
    s = report.summary()
    # Horizon stayed on the cheap tier (fix loop, never climbed) → cheaper than ceiling.
    assert s["horizon"]["total_cost"] < s["bare:ceiling"]["total_cost"]
    assert s["bare:cheap"]["total_cost"] > 0


async def test_by_difficulty_buckets(tmp_path):
    report = await _run(tmp_path)
    bd = report.by_difficulty()
    assert set(bd["horizon"].keys()) == {"easy", "hard"}
    assert bd["horizon"]["easy"] == 1.0 and bd["horizon"]["hard"] == 1.0
    assert bd["bare:cheap"]["easy"] == 0.0


async def test_render_includes_verdict(tmp_path):
    report = await _run(tmp_path)
    text = report.render()
    assert "Dubina coder benchmark" in text
    assert "Verdict:" in text
    assert "% of ceiling cost" in text


async def test_broken_task_does_not_sink_suite(tmp_path):
    # A task whose provider raises is recorded as a failed result, not an exception.
    def boom_factory():
        def f(tier):
            raise RuntimeError("provider down")
        return f
    report = await run_benchmark(
        tasks()[:1], [bare_condition("cheap", COSTS)],
        provider_factory=boom_factory(), runner=runner_passes_on("CORRECT"),
        workdir_root=tmp_path,
    )
    assert report.results and report.results[0].passed is False
