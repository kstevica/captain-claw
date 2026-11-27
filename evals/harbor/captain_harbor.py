"""Harbor adapter: run Captain on Harbor tasks (ShellBench, Terminal-Bench …).

    harbor run -p <tasks dir> --agent captain_harbor:Captain \
        -m anthropic/claude-haiku-5-5

A Harbor trial runs Captain at reasoning effort medium and Drive ambitious by
default, locked for the whole run.

(with ``evals/harbor`` on PYTHONPATH). Each trial installs Captain in the task
container and runs ``captain-claw-once`` in the task's working directory. That
starts a private Flight Deck inside the container, which spawns a process
agent on the task folder and gets the task through the deck's inbound MCP —
the whole system, not a bare agent. ``--ak standalone=true`` runs a bare
in-process agent instead, for comparison.

The model is ``provider/model`` in Captain's provider names (anthropic,
openai, gemini, openrouter, xai): ``openrouter/deepseek/deepseek-v4-pro``
gives provider ``openrouter`` and model ``deepseek/deepseek-v4-pro``. The key
comes from the host's environment (ANTHROPIC_API_KEY …) or ``--ae``; web
search keys (BRAVE_API_KEY …) are passed along when set.

Options (``--ak name=value``):
  source           what to pip-install: ``captain-claw==0.9.3`` (default:
                   the latest release), a git URL, or a path to a wheel on
                   this machine (uploaded into the container)
  constraints      pip constraints file on this machine pinning Captain's
                   dependencies to the tested versions (``uv export`` of
                   uv.lock) — without it the container gets whatever is
                   newest that day
  extras           Captain extras to install, comma-separated (default
                   ``pdf``: PDFs read with their headings and tables through
                   PyMuPDF; without it the agent reads PDFs page by page, or
                   with pictures). ``none`` installs none
  workspace        folder the agent works in (default: the container's
                   working directory)
  reasoning_effort off · minimal · low · medium · high · xhigh · max · auto
                   (default medium; auto leaves it to Captain)
  drive            relaxed · diligent · persistent · ambitious · auto (how far
                   the harness pushes the turn; default ambitious)
  lock             force the effort and Drive above for the whole run, so the
                   agent can't change them (default true)
  endpoint         the model's server, for a local or self-hosted model on an
                   OpenAI-compatible API: ``-m openai/<model id>`` with
                   ``--ak endpoint=http://host:8080/v1`` and its key in
                   OPENAI_API_KEY
  max_context      the model's context window, in tokens (default 160K)
  max_output       output tokens per model call
  max_iterations   tool-call budget for the turn
  timeout_seconds  stop the task after this long (default: the task's own
                   limit, which ``--timeout-multiplier`` scales)
  archetype        spawn the agent from the deck's agent store, by id
                   (``captain-claw-once --list-store`` lists them); it still
                   runs the model given with -m
  standalone       true: bare agent, no Flight Deck
  python           Python version for Captain's venv (default 3.12)

Logs: ``captain.txt`` (progress), ``captain-result.json`` (answer, usage,
changed files, transcript), ``trajectory.json`` (the run in ATIF, which
graders read the agent's replies from) and the deck's and agent's logs under
``captain-run/`` in the trial's agent log folder.
"""

from __future__ import annotations

import json
import os
import shlex
from pathlib import Path
from typing import Any, Literal, override

from harbor.agents.capabilities import AgentCapabilities
from harbor.agents.installed.base import BaseInstalledAgent, with_prompt_template
from harbor.agents.options import InstalledAgentOptions
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from pydantic import Field

ROOT = "/opt/captain-claw"
VENV = f"{ROOT}/venv"
RUNNER = f"{VENV}/bin/captain-claw-once"
RUN_DIR = "/tmp/captain-run"
RESULT_NAME = "captain-result.json"
# ATIF: what Harbor and its benchmarks' graders read the run from.
TRAJECTORY_NAME = "trajectory.json"

PROVIDER_KEY_ENVS: dict[str, tuple[str, ...]] = {
    "anthropic": ("ANTHROPIC_API_KEY",),
    "openai": ("OPENAI_API_KEY",),
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "openrouter": ("OPENROUTER_API_KEY",),
    "xai": ("XAI_API_KEY",),
}
TOOL_KEY_ENVS = ("BRAVE_API_KEY", "TAVILY_API_KEY", "YDC_API_KEY", "KEENABLE_API_KEY")
# A Harbor trial's levels unless told otherwise, held for the whole run
# (``lock``). Medium, not max: at max, Haiku 5.5 thought 4–6 minutes per step
# and ran out a 15-minute task. Ambitious's unasked extra step never runs on
# these tasks (automated turns), so it adds Persistent's limits and check plus
# "go deeper".
HARBOR_EFFORT = "medium"
HARBOR_DRIVE = "ambitious"
_LEAVE_TO_CAPTAIN = ("auto", "default")
# Captain's optional extras a trial installs unless told otherwise: without
# ``pdf`` (PyMuPDF) a PDF is read page by page, and a small model falls back to
# reading pages as pictures.
HARBOR_EXTRAS = "pdf"
# The runner's own process (not the shell that started it), for stopping it.
RUNNER_PROCESS = f"^{VENV}/bin/python[0-9.]* {RUNNER}"
# How long a stopped runner gets to cancel its task and write its result and
# transcript.
STOP_GRACE_SECONDS = 150


class CaptainOptions(InstalledAgentOptions):
    source: str | None = Field(
        default=None,
        description="pip spec for Captain: a version pin, a git URL, or a local wheel path.",
    )
    constraints: str | None = Field(
        default=None,
        description="A pip constraints file on this machine (uv export of uv.lock) pinning Captain's dependencies.",
    )
    extras: str = Field(
        default=HARBOR_EXTRAS,
        description="Captain extras to install, comma-separated ('none' for none).",
    )
    workspace: str | None = Field(
        default=None, description="Folder the agent works in (default: the working directory)."
    )
    # Not "base_url": Harbor keeps that name for agents it connects itself.
    endpoint: str | None = Field(
        default=None, description="Model server URL (a local or self-hosted OpenAI-compatible API).",
    )
    max_context: int | None = Field(default=None, description="The model's context window, in tokens.")
    max_output: int | None = Field(default=None, description="Output tokens per model call.")
    drive: Literal["relaxed", "diligent", "persistent", "ambitious", "auto"] | None = Field(
        default=None, description=f"Drive: how far the harness pushes the turn (default {HARBOR_DRIVE}).",
    )
    lock: bool = Field(default=True, description="Force effort and Drive for the whole run.")
    max_iterations: int | None = Field(default=None, description="Tool-call budget for the turn.")
    timeout_seconds: float | None = Field(default=None, description="Stop the task after this long.")
    archetype: str | None = Field(default=None, description="Spawn this agent from the deck's agent store, by id.")
    standalone: bool = Field(default=False, description="Bare in-process agent, no Flight Deck.")
    python: str = Field(default="3.12", description="Python version for Captain's venv.")


def split_model(model_name: str) -> tuple[str, str]:
    """``provider/model`` → (provider, model); the model may contain slashes."""
    provider, sep, model = (model_name or "").partition("/")
    if not sep or not provider or not model:
        raise ValueError("Model name must be provider/model, e.g. anthropic/claude-haiku-5-5")
    return provider, model


def with_extras(spec: str, extras: str) -> str:
    """The pip spec with Captain's extras: ``captain-claw[pdf]==0.9.3``, or
    ``captain-claw[pdf] @ file:///tmp/x.whl`` for a wheel, path or URL."""
    names = [e.strip() for e in (extras or "").split(",") if e.strip() and e.strip().lower() != "none"]
    if not names:
        return spec
    wanted = ",".join(names)
    if spec.startswith("/"):
        return f"captain-claw[{wanted}] @ file://{spec}"
    if "://" in spec or spec.endswith(".whl"):
        return f"captain-claw[{wanted}] @ {spec}"
    name, rest = spec, ""
    for i, ch in enumerate(spec):
        if ch in "[=<>!~ ;@":
            name, rest = spec[:i], spec[i:]
            break
    if rest.startswith("["):   # extras given in the spec already: keep both
        given, _, rest = rest[1:].partition("]")
        wanted = ",".join(dict.fromkeys([*given.split(","), *names]))
    return f"{name}[{wanted}]{rest}"


def stop_runner_command(result_path: str) -> str:
    """Shell that stops a runner still going (Harbor's time limit only drops its
    connection) and waits for it to write its result and transcript."""
    pattern = shlex.quote(RUNNER_PROCESS)
    return (
        f"if pgrep -f {pattern} >/dev/null 2>&1; then "
        f"pkill -TERM -f {pattern}; "
        f"for i in $(seq 1 {STOP_GRACE_SECONDS}); do "
        f"pgrep -f {pattern} >/dev/null 2>&1 || break; sleep 1; done; "
        f"pkill -KILL -f {pattern} 2>/dev/null; "
        f"[ -s {shlex.quote(result_path)} ] || echo 'captain: stopped without a result file' >&2; "
        "fi; true"
    )


def benchmark_levels(options: CaptainOptions) -> tuple[str, str]:
    """(effort, drive) for the run: what was asked, else the Harbor defaults;
    "" where ``auto`` leaves the choice to Captain."""
    def pick(asked: str | None, default: str) -> str:
        text = str(asked or "").strip().lower()
        if text in _LEAVE_TO_CAPTAIN:
            return ""
        return text or default

    return pick(options.reasoning_effort, HARBOR_EFFORT), pick(options.drive, HARBOR_DRIVE)


def runner_flags(provider: str, model: str, options: CaptainOptions, logs_dir: str,
                 skills_dir: str | None = None) -> list[str]:
    flags = ["-p", provider, "-m", model, "--run-dir", RUN_DIR,
             "--result-json", f"{logs_dir}/{RESULT_NAME}",
             "--trajectory-json", f"{logs_dir}/{TRAJECTORY_NAME}"]
    effort, drive = benchmark_levels(options)
    if effort:
        flags += ["--effort", effort]
    if drive:
        flags += ["--drive", drive]
    if options.lock and (effort or drive):
        flags += ["--lock"]
    if options.max_iterations:
        flags += ["--max-iterations", str(options.max_iterations)]
    if options.timeout_seconds:
        flags += ["--timeout", str(options.timeout_seconds)]
    if options.endpoint:
        flags += ["--base-url", options.endpoint]
    if options.max_context:
        flags += ["--max-context", str(options.max_context)]
    if options.max_output:
        flags += ["--max-output", str(options.max_output)]
    if skills_dir:
        flags += ["--skills-dir", skills_dir]
    if options.workspace:
        flags += ["--workspace", options.workspace]
    if options.archetype and not options.standalone:
        flags += ["--archetype", options.archetype]
    if options.standalone:
        flags += ["--standalone", "--bench"]
    return flags


def context_from_result(data: dict[str, Any]) -> dict[str, Any]:
    """Harbor ``AgentContext`` fields from ``captain-result.json``."""
    usage = data.get("usage") or {}
    return {
        "n_input_tokens": int(usage.get("prompt_tokens", 0)) or None,
        "n_output_tokens": int(usage.get("completion_tokens", 0)) or None,
        "n_cache_tokens": int(usage.get("cache_read_input_tokens", 0)) or None,
        "cost_usd": data.get("cost_usd_estimate"),
        "metadata": {
            key: data.get(key)
            for key in ("ok", "error", "mode", "captain_version", "effort", "drive", "locked", "store_agent",
                        "models_used", "duration_seconds", "tool_calls", "tools_used", "files", "deck")
        },
    }


class Captain(BaseInstalledAgent):
    """Captain, run through a Flight Deck inside the task container."""

    # skills: a task's skills folder is installed into the agent (captain-claw-once --skills-dir).
    capabilities = AgentCapabilities(reasoning_effort=True, skills=True, atif=True)
    options_model = CaptainOptions
    # uv's installer, Python and Captain's wheels: a few minutes on a busy host.
    DEFAULT_SETUP_TIMEOUT_SEC = 1200.0

    @staticmethod
    @override
    def name() -> str:
        return "captain"

    @override
    def get_version_command(self) -> str | None:
        return f"{VENV}/bin/python -c 'import captain_claw; print(captain_claw.__version__)'"

    async def _pip_spec(self, environment: BaseEnvironment) -> str:
        source = (self.options.source or "").strip()
        if not source:
            return f"captain-claw=={self._version}" if self._version else "captain-claw"
        local = Path(source).expanduser()
        if local.is_file():
            remote = f"/tmp/{local.name}"
            await environment.upload_file(local, remote)
            return remote
        return source

    async def _constraints_flag(self, environment: BaseEnvironment) -> str:
        source = (self.options.constraints or "").strip()
        if not source:
            return ""
        local = Path(source).expanduser()
        if not local.is_file():
            raise FileNotFoundError(f"constraints file not found: {local}")
        remote = "/tmp/captain-constraints.txt"
        await environment.upload_file(local, remote)
        return f"-c {remote} "

    @override
    async def install(self, environment: BaseEnvironment) -> None:
        await self.ensure_system_dependencies(environment, ("curl", "bash"))
        spec = with_extras(await self._pip_spec(environment), self.options.extras)
        constraints = await self._constraints_flag(environment)
        await self.exec_as_root(
            environment,
            command=(
                "set -euo pipefail; "
                f"mkdir -p {ROOT}/bin; "
                f"export UV_PYTHON_INSTALL_DIR={ROOT}/python UV_CACHE_DIR={ROOT}/cache; "
                # Fewer parallel downloads and a longer wait: on a busy or
                # emulated host many connections at once time out.
                "export UV_CONCURRENT_DOWNLOADS=8 UV_HTTP_TIMEOUT=120; "
                f"if [ ! -x {ROOT}/bin/uv ]; then "
                f"curl -LsSf https://astral.sh/uv/install.sh | env UV_UNMANAGED_INSTALL={ROOT}/bin sh; fi; "
                f"{ROOT}/bin/uv venv --python {shlex.quote(self.options.python)} {VENV}; "
                # A dropped download is retried; what finished stays in the cache.
                "for attempt in 1 2 3; do "
                f"{ROOT}/bin/uv pip install --python {VENV}/bin/python {constraints}{shlex.quote(spec)} && break; "
                '[ "$attempt" = 3 ] && exit 1; sleep 5; done; '
                f"rm -rf {ROOT}/cache; chmod -R a+rX {ROOT}; "
                f"{RUNNER} --help >/dev/null"
            ),
        )

    def _agent_env(self, provider: str, instruction: str) -> dict[str, str]:
        env = {"CLAW_INSTRUCTION": instruction}
        for name in (*PROVIDER_KEY_ENVS.get(provider, ()), *TOOL_KEY_ENVS):
            value = self._extra_env.get(name) or os.environ.get(name)
            if value:
                env[name] = value
        return env

    @override
    @with_prompt_template
    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        provider, model = split_model(self.model_name or "")
        logs_dir = str(self.environment_logs_dir)
        flags = " ".join(shlex.quote(f) for f in runner_flags(provider, model, self.options, logs_dir, self.skills_dir))
        try:
            await self.exec_as_agent(
                environment,
                command=f"{RUNNER} {flags} 2>&1 | tee {logs_dir}/captain.txt",
                env=self._agent_env(provider, instruction),
            )
        finally:
            # At its time limit Harbor cancels this call and only drops its
            # exec: the runner would carry on in the container, changing files
            # while the grader reads them, and never write its result or the
            # transcript the graders judge. Stop it and let it write both.
            # (Harbor waits for this block before it raises the timeout.)
            await environment.exec(
                command=stop_runner_command(f"{logs_dir}/{RESULT_NAME}"),
                timeout_sec=STOP_GRACE_SECONDS + 30,
            )
            # The deck's and agent's logs, for a look at what happened.
            await environment.exec(
                command=(
                    f"mkdir -p {logs_dir}/captain-run && cd {RUN_DIR} 2>/dev/null && "
                    f"find . -name '*.log' -size -20M "
                    f"-exec cp --parents {{}} {logs_dir}/captain-run/ \\; 2>/dev/null; true"
                ),
            )

    @override
    def populate_context_post_run(self, context: AgentContext) -> None:
        path = self.logs_dir / RESULT_NAME
        if not path.is_file():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        for key, value in context_from_result(data).items():
            setattr(context, key, value)
