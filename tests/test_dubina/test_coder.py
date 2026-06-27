"""Tests for the Dubina coder track (Phase 1).

Cover code extraction, per-attempt workspace isolation, the ground-truth
CoderVerifier, the spec->tests->code path, and an end-to-end engine run where a
stub LLM is elevated against a stub test runner.
"""

from __future__ import annotations

from pathlib import Path

from captain_claw.dubina import (
    CODER_LADDER,
    CoderVerifier,
    EngineConfig,
    HorizonEngine,
    Step,
    Workspace,
    any_pass_aggregator,
    ensure_tests,
    extract_code_blocks,
    make_coder_generator,
    resolve_ladder,
)
from captain_claw.dubina.coder import (
    SOLUTION_PATH_KEY,
    TEST_COMMAND_KEY,
    WORKDIR_KEY,
    WORKSPACE_KEY,
)
from captain_claw.llm import LLMResponse, Message

# ── Stubs ────────────────────────────────────────────────────────────

class StubProvider:
    """Returns canned content; can vary the reply once feedback appears."""

    def __init__(self, first: str, on_fix: str | None = None):
        self.first = first
        self.on_fix = on_fix
        self.calls: list[list[Message]] = []

    async def complete(self, messages, tools=None, temperature=None, max_tokens=None):
        self.calls.append(messages)
        user = messages[-1].content
        if self.on_fix is not None and "FAILED" in user:
            return LLMResponse(content=self.on_fix, finish_reason="stop")
        return LLMResponse(content=self.first, finish_reason="stop")


def provider_const(provider) -> object:
    return lambda tier: provider


def runner_passes_on(token: str, solution="solution.py"):
    """Stub CommandRunner: reads the solution in cwd, passes iff it contains token."""
    seen: list[str] = []

    async def run(command: str, cwd: str) -> tuple[bool, str]:
        seen.append(cwd)
        code = (Path(cwd) / solution).read_text()
        if token in code:
            return True, "1 passed"
        return False, "E   AssertionError: wrong answer\n1 failed"

    run.seen = seen  # type: ignore[attr-defined]
    return run


def coder_step(base: Path, **meta) -> Step:
    md = {
        WORKSPACE_KEY: str(base),
        TEST_COMMAND_KEY: "pytest -q",
        SOLUTION_PATH_KEY: "solution.py",
    }
    md.update(meta)
    return Step(id="add", prompt="implement add(a, b)", metadata=md)


def cfg(base_tier="gemini-flash", max_tier="gpt-5.3-codex", **kw) -> EngineConfig:
    params = dict(ladder=resolve_ladder(CODER_LADDER, base_tier, max_tier),
                  max_step_samples=3, max_fix_attempts=2)
    params.update(kw)
    return EngineConfig(**params)


# ── extract_code_blocks ──────────────────────────────────────────────

def test_extract_prefers_largest_fenced_block():
    text = "blah\n```py\nsmall\n```\nmore\n```python\nthe real code\nline2\n```\n"
    assert extract_code_blocks(text) == "the real code\nline2"


def test_extract_falls_back_to_whole_text():
    assert extract_code_blocks("  def add(a, b): return a + b  ") == "def add(a, b): return a + b"


# ── Workspace isolation ──────────────────────────────────────────────

def test_workspace_isolates_attempts(tmp_path):
    (tmp_path / "test_add.py").write_text("from solution import add\n")
    (tmp_path / ".git").mkdir()
    ws = Workspace(tmp_path)

    a = ws.prepare_attempt("t-0001")
    b = ws.prepare_attempt("t-0002")

    assert a != b
    assert (a / "test_add.py").exists()       # base files copied in
    assert not (a / ".git").exists()          # .git ignored
    (a / "solution.py").write_text("A")
    (b / "solution.py").write_text("B")
    assert (a / "solution.py").read_text() == "A"  # no cross-contamination


# ── CoderVerifier ────────────────────────────────────────────────────

async def test_verifier_passes_on_zero_exit(tmp_path):
    (tmp_path / "solution.py").write_text("def add(a, b): return a + b  # CORRECT")
    from captain_claw.dubina import Candidate

    runner = runner_passes_on("CORRECT")
    verifier = CoderVerifier(runner)
    step = coder_step(tmp_path)
    cand = Candidate("add", "code", "gemini-flash", {WORKDIR_KEY: str(tmp_path)})

    verdict = await verifier.check(step, cand)
    assert verdict.passed and verdict.confidence == 1.0


async def test_verifier_returns_failure_output_as_feedback(tmp_path):
    (tmp_path / "solution.py").write_text("def add(a, b): return 0")
    from captain_claw.dubina import Candidate

    verifier = CoderVerifier(runner_passes_on("CORRECT"))
    cand = Candidate("add", "code", "gemini-flash", {WORKDIR_KEY: str(tmp_path)})
    verdict = await verifier.check(coder_step(tmp_path), cand)

    assert not verdict.passed
    assert "AssertionError" in verdict.feedback


# ── Generator ────────────────────────────────────────────────────────

async def test_generator_writes_solution_into_isolated_dir(tmp_path):
    (tmp_path / "test_add.py").write_text("from solution import add\n")
    provider = StubProvider(first="```python\ndef add(a, b): return a + b\n```")
    gen = make_coder_generator(provider_const(provider), Workspace(tmp_path))

    c1 = await gen(coder_step(tmp_path, test_path="test_add.py"), "gemini-flash", "", 0)
    c2 = await gen(coder_step(tmp_path, test_path="test_add.py"), "gemini-flash", "", 1)

    assert c1.metadata[WORKDIR_KEY] != c2.metadata[WORKDIR_KEY]   # isolated per sample
    sol = Path(c1.metadata[WORKDIR_KEY]) / "solution.py"
    assert sol.read_text() == "def add(a, b): return a + b"
    # The model saw the tests it must satisfy.
    assert "from solution import add" in provider.calls[0][-1].content


# ── End-to-end through the engine ─────────────────────────────────────

async def test_engine_elevates_stub_model_to_passing_code(tmp_path):
    (tmp_path / "test_add.py").write_text("from solution import add\n")
    provider = StubProvider(first="```python\ndef add(a, b): return a + b  # CORRECT\n```")
    engine = HorizonEngine(
        cfg(), make_coder_generator(provider_const(provider), Workspace(tmp_path)),
        CoderVerifier(runner_passes_on("CORRECT")), aggregator=any_pass_aggregator,
    )

    result = await engine.run(coder_step(tmp_path))
    assert result.passed
    assert result.steps[0].rung_reached == 0   # cheap tier was enough


async def test_engine_fix_loop_recovers_from_bad_first_attempt(tmp_path):
    (tmp_path / "test_add.py").write_text("from solution import add\n")
    # First reply is wrong; once the failure feedback arrives, the model fixes it.
    provider = StubProvider(
        first="```python\ndef add(a, b): return 0\n```",
        on_fix="```python\ndef add(a, b): return a + b  # CORRECT\n```",
    )
    engine = HorizonEngine(
        cfg(max_step_samples=1),  # force the win to come from the fix loop
        make_coder_generator(provider_const(provider), Workspace(tmp_path)),
        CoderVerifier(runner_passes_on("CORRECT")),
    )

    result = await engine.run(coder_step(tmp_path))
    assert result.passed
    assert result.steps[0].fix_attempts >= 1


# ── spec -> tests -> code ────────────────────────────────────────────

async def test_ensure_tests_synthesizes_when_missing(tmp_path):
    provider = StubProvider(first="```python\ndef test_add(): assert add(1, 2) == 3\n```")
    created = await ensure_tests("add(a,b) returns a+b", provider, tmp_path, "test_add.py")

    assert created is True
    assert "def test_add" in (tmp_path / "test_add.py").read_text()


async def test_ensure_tests_skips_when_present(tmp_path):
    (tmp_path / "test_add.py").write_text("def test_add(): assert True\n")
    provider = StubProvider(first="should-not-be-used")
    created = await ensure_tests("spec", provider, tmp_path, "test_add.py")

    assert created is False
    assert provider.calls == []
