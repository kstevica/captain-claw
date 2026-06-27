"""Dubina coder track — the ground-truth verifier plug-in.

The coder track is the strongest elevation lever because code has an *infallible*
external checker: the test runner. This turns the weak model's job from "write
correct code" (low p) into "write code + fix against a checker" (much higher p),
which is what breaks the p^n compounding the engine fights.

Pieces:

* ``Workspace``            — per-attempt isolated copy of the project so the
  parallel sample/vote candidates can't clobber each other's files (design's
  worktree/vfs isolation; copytree here, vfs overlay is a later optimization).
* ``make_coder_generator`` — an engine ``Generator``: ask the LLM at a tier for
  code, write it into a fresh attempt dir, return a ``Candidate`` carrying that dir.
* ``CoderVerifier``        — an engine ``Verifier``: run the step's test command in
  the candidate's dir; exit 0 → passed, else the failure output becomes feedback.
* ``ensure_tests``         — the spec→tests→code path: synthesize a test file from a
  spec when the project has none, so a ground-truth verifier always exists.

The LLM provider factory and the command runner are injected, so everything here
is unit-testable with stubs (no real model or subprocess needed).
"""

from __future__ import annotations

import re
import shutil
from collections.abc import Awaitable, Callable
from pathlib import Path

from captain_claw.dubina.engine import Candidate, Step, Verdict
from captain_claw.llm import LLMProvider, Message
from captain_claw.logging import get_logger

log = get_logger(__name__)

# Step.metadata / Candidate.metadata keys the coder track relies on.
WORKSPACE_KEY = "workspace"        # Step: base project dir (str/Path)
TEST_COMMAND_KEY = "test_command"  # Step: e.g. "pytest -q"
SOLUTION_PATH_KEY = "solution_path"  # Step: relative file the LLM must write
WORKDIR_KEY = "workdir"            # Candidate: the isolated attempt dir

_DEFAULT_SOLUTION = "solution.py"
_FEEDBACK_LIMIT = 4000  # chars of test output threaded back into the fix loop


# A command runner: ``async (command, cwd) -> (ok, combined_output)``.
CommandRunner = Callable[[str, str], Awaitable[tuple[bool, str]]]
# Maps a tier id (model.allowed id) to a ready LLM provider.
ProviderForTier = Callable[[str], LLMProvider]


# ── Code extraction ──────────────────────────────────────────────────

_FENCE_RE = re.compile(r"```[^\n]*\n(.*?)```", re.DOTALL)


def extract_code_blocks(text: str) -> str:
    """Pull source out of an LLM reply.

    Returns the largest fenced code block (models sometimes wrap prose around it);
    if there are no fences, returns the whole stripped text.
    """
    blocks = _FENCE_RE.findall(text)
    if blocks:
        return max(blocks, key=len).strip()
    return text.strip()


# ── Isolation ────────────────────────────────────────────────────────

# Not copied into attempt dirs (module-level so it isn't bound as a method).
_WS_IGNORE = shutil.ignore_patterns(".dubina", ".git", "__pycache__", "*.pyc")


class Workspace:
    """Hands out isolated per-attempt copies of a base project directory.

    Each attempt gets its own dir under ``<base>/.dubina/`` so N parallel samples
    (and the fix loop) never collide. ``.dubina`` and ``.git`` are not copied in.
    """

    def __init__(self, base: str | Path):
        self.base = Path(base)
        self._attempts_root = self.base / ".dubina"

    def prepare_attempt(self, label: str) -> Path:
        """Return a fresh copy of the base project, isolated under ``label``."""
        dest = self._attempts_root / label
        if dest.exists():
            shutil.rmtree(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(self.base, dest, ignore=_WS_IGNORE)
        return dest


# ── Generator ────────────────────────────────────────────────────────

_CODER_SYSTEM = (
    "You are a coding agent. Produce a complete, correct implementation that makes "
    "the project's tests pass. Output ONLY the full contents of the target file in a "
    "single fenced code block — no prose, no explanation."
)


def _build_coder_messages(step: Step, feedback: str, tests: str) -> list[Message]:
    parts = [f"Task:\n{step.prompt}"]
    solution_path = step.metadata.get(SOLUTION_PATH_KEY, _DEFAULT_SOLUTION)
    parts.append(f"\nWrite the complete contents of `{solution_path}`.")
    if tests:
        parts.append(f"\nThe code must satisfy these tests:\n```\n{tests}\n```")
    if feedback:
        parts.append(
            f"\nThe previous attempt FAILED its tests. Fix it. Test output:\n"
            f"```\n{feedback}\n```"
        )
    return [
        Message(role="system", content=_CODER_SYSTEM),
        Message(role="user", content="\n".join(parts)),
    ]


def make_coder_generator(
    provider_for_tier: ProviderForTier,
    workspace: Workspace,
    *,
    max_tokens: int = 8000,
):
    """Build an engine ``Generator`` that writes LLM-produced code to isolated dirs.

    On each call it prepares a fresh attempt dir, reads the step's test file (so the
    model sees what it must satisfy), asks the tier's model for the solution file,
    writes it, and returns a ``Candidate`` whose ``metadata[WORKDIR_KEY]`` points at
    the dir the verifier will run in.
    """
    counter = {"n": 0}

    async def generate(step: Step, tier: str, feedback: str, sample: int) -> Candidate:
        counter["n"] += 1
        label = f"{tier}-{counter['n']:04d}"
        workdir = workspace.prepare_attempt(label)

        solution_path = step.metadata.get(SOLUTION_PATH_KEY, _DEFAULT_SOLUTION)
        tests = _read_tests(step, workdir)

        provider = provider_for_tier(tier)
        messages = _build_coder_messages(step, feedback, tests)
        response = await provider.complete(messages, max_tokens=max_tokens)
        code = extract_code_blocks(response.content)

        target = workdir / solution_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(code)

        return Candidate(
            step_id=step.id,
            content=code,
            tier=tier,
            metadata={WORKDIR_KEY: str(workdir)},
        )

    return generate


def _read_tests(step: Step, workdir: Path) -> str:
    """Best-effort read of the step's test file from an attempt dir (for the prompt)."""
    test_file = step.metadata.get("test_path")
    if not test_file:
        return ""
    path = workdir / test_file
    try:
        return path.read_text()
    except OSError:
        return ""


# ── Verifier (ground truth) ──────────────────────────────────────────

class CoderVerifier:
    """Runs the step's test command in the candidate's dir — the ground truth.

    ``confidence`` is binary (1.0 pass / 0.0 fail) because a passing test suite is
    not a statistical signal. On failure the (truncated) test output is returned as
    ``feedback`` so the engine's fix loop can act on it.
    """

    def __init__(self, runner: CommandRunner):
        self._runner = runner

    async def check(self, step: Step, candidate: Candidate) -> Verdict:
        command = step.metadata.get(TEST_COMMAND_KEY)
        if not command:
            raise ValueError(f"step {step.id!r} has no {TEST_COMMAND_KEY!r}")
        workdir = candidate.metadata.get(WORKDIR_KEY)
        if not workdir:
            raise ValueError(f"candidate for {step.id!r} has no {WORKDIR_KEY!r}")

        ok, output = await self._runner(command, workdir)
        if ok:
            return Verdict(passed=True, confidence=1.0, feedback="")
        return Verdict(passed=False, confidence=0.0, feedback=output[:_FEEDBACK_LIMIT])


# ── spec → tests → code ──────────────────────────────────────────────

_TESTS_SYSTEM = (
    "You are a test author. Given a specification, write a focused test file that "
    "pins the required behavior. Output ONLY the test file contents in one fenced "
    "code block."
)


async def ensure_tests(
    spec: str,
    provider: LLMProvider,
    workspace_base: str | Path,
    test_path: str,
    *,
    max_tokens: int = 4000,
) -> bool:
    """Guarantee a ground-truth verifier exists by synthesizing tests from a spec.

    If ``test_path`` already exists and is non-empty in the base project, do nothing
    and return ``False``. Otherwise generate a test file from ``spec`` and write it,
    returning ``True``. The generated tests are then copied into every attempt dir by
    ``Workspace.prepare_attempt`` and become the verifier's ground truth.
    """
    path = Path(workspace_base) / test_path
    if path.exists() and path.read_text().strip():
        return False

    messages = [
        Message(role="system", content=_TESTS_SYSTEM),
        Message(role="user", content=f"Specification:\n{spec}\n\nWrite `{test_path}`."),
    ]
    response = await provider.complete(messages, max_tokens=max_tokens)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(extract_code_blocks(response.content))
    return True


# ── Default wiring (real shell + config-driven providers) ────────────

async def shell_command_runner(command: str, cwd: str) -> tuple[bool, str]:
    """Default ``CommandRunner`` over the project's ShellTool (real subprocess)."""
    from captain_claw.tools.shell import ShellTool

    result = await ShellTool().execute(command, _runtime_base_path=cwd)
    output = result.content or result.error or ""
    return bool(result.success), output


def provider_for_tier_from_config() -> ProviderForTier:
    """Map a tier id to a provider via ``config.model.allowed``.

    Thin default: resolves provider/model/base_url/temperature from the allowed-model
    entry and lets ``create_provider`` pick up API keys from the environment. Full
    per-agent BYOK key resolution (AgentModelMixin) is wired in at the FD layer (Phase 3).
    """
    from captain_claw.config import get_config
    from captain_claw.llm import create_provider

    def factory(tier: str) -> LLMProvider:
        cfg = get_config()
        entry = next((m for m in cfg.model.allowed if m.id == tier), None)
        if entry is None:
            raise ValueError(f"tier {tier!r} not found in config.model.allowed")
        return create_provider(
            provider=entry.provider,
            model=entry.model,
            base_url=entry.base_url or None,
            temperature=entry.temperature if entry.temperature is not None else 0.7,
            max_tokens=entry.max_tokens or 8000,
        )

    return factory
