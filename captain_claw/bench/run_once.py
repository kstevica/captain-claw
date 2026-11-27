"""One task, headless — for benchmark harnesses and scripts.

    captain-claw-once -p anthropic -m claude-haiku-5-5 "Total the receipts into output/total.md"
    python -m captain_claw.bench.run_once --instruction-file task.md -p openai -m gpt-6.1-sol

By default the task runs through Flight Deck, the whole system: a private deck
starts on loopback (or ``--deck-url`` names a running one), spawns a process
agent whose workspace is the current folder (or ``--workspace``), hands it the
task through the deck's inbound MCP and waits for the answer — see
``deck_run``. ``--standalone`` runs a bare in-process agent instead, without
the deck (quicker, for debugging the agent alone).

Either way the run writes ``result.json``: the answer, token usage, an
estimated cost, the files the agent created or changed in the workspace, and
the transcript. The answer is also printed to stdout; progress lines go to
stderr.

Nothing touches the person's own Captain data. The private deck has its own
data folder and HOME; a standalone run keeps sessions and memory in the run
folder and restarts itself with HOME there (``--keep-home`` turns that off).
Model keys come from the usual environment variables (ANTHROPIC_API_KEY,
OPENAI_API_KEY, OPENROUTER_API_KEY, GEMINI_API_KEY …).

Writes land where the agent asks (``CLAW_WRITE_DIRECT``): a benchmark grades
the files in the task folder, and the interactive default files every write
under ``saved/tmp/<session>/`` instead. ``--sandboxed-writes`` keeps that.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any

from captain_claw.drive import LEVELS as _DRIVE

DRIVE_LEVELS: tuple[str, ...] = tuple(_DRIVE)   # relaxed, diligent, persistent, ambitious
INSTRUCTION_ENV = "CLAW_INSTRUCTION"
# Set on the restarted process so it doesn't restart again.
_HOME_SET_ENV = "CLAW_ONCE_HOME_SET"
# The caller's own HOME, so a "~/…" argument still means their folder.
_REAL_HOME_ENV = "CLAW_ONCE_REAL_HOME"
# More files than this and the before/after file listing is skipped.
_SNAPSHOT_LIMIT = 20000
# Folders never listed: version control and package caches.
_SNAPSHOT_SKIP = {".git", "node_modules", "__pycache__", ".venv"}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="captain-claw-once",
        description="Run one task through Flight Deck (or a bare agent) and write a result file.",
    )
    p.add_argument("instruction", nargs="?", default="",
                   help=f"What to do. Or --instruction-file, ${INSTRUCTION_ENV}, or stdin ('-').")
    p.add_argument("--instruction-file", default="", help="Read the instruction from this file ('-' = stdin).")
    p.add_argument("-p", "--provider", default="", help="Provider: anthropic, openai, gemini, openrouter, xai, ollama …")
    p.add_argument("-m", "--model", default="", help="Model id for that provider.")
    p.add_argument("--base-url", default="",
                   help="Model endpoint, e.g. a local OpenAI-compatible server (http://host:8080/v1).")
    p.add_argument("--max-context", type=int, default=0,
                   help="The model's context window to plan for, in tokens (default: Captain's 160K).")
    p.add_argument("--max-output", type=int, default=0, help="Output tokens per model call (default: Captain's).")
    p.add_argument("--vision", default="", choices=["", "auto", "on", "off"],
                   help="Whether the model sees pictures itself: on, off, or auto (Captain decides; the default). "
                        "Off, a picture goes to a separate vision call instead.")
    p.add_argument("--effort", default="", help="Reasoning effort: off, minimal, low, medium, high, xhigh, max.")
    p.add_argument("--drive", default="", choices=["", *DRIVE_LEVELS],
                   help="Drive, how far the harness pushes the turn: " + ", ".join(DRIVE_LEVELS) + ".")
    p.add_argument("--lock", action="store_true",
                   help="Force --effort and --drive for the whole run: the agent can't change them.")
    p.add_argument("--workspace", default="", help="Folder the agent works in (default: the current folder).")
    p.add_argument("--skills-dir", default="",
                   help="Skills for the task: a folder of skill folders, each with a SKILL.md.")
    p.add_argument("--run-dir", default="", help="Where the deck, logs and result.json go (default: a new temp folder).")
    p.add_argument("--result-json", default="", help="Result file (default: <run dir>/result.json).")
    p.add_argument("--answer-file", default="", help="Also write the final answer here.")
    p.add_argument("--trajectory-json", default="",
                   help="Also write the run as an ATIF trajectory here (what Harbor graders read).")
    p.add_argument("--max-iterations", type=int, default=0, help="Tool-call budget for the turn (default: Captain's).")
    p.add_argument("--timeout", type=float, default=0.0,
                   help="Stop the task after this many seconds (default: 4 h on a private deck, so the "
                        "harness sets the limit; 30 min on a --deck-url deck).")
    p.add_argument("--sandboxed-writes", action="store_true", help="File writes go under saved/ as in the app.")
    p.add_argument("-q", "--quiet", action="store_true", help="No progress lines on stderr.")
    p.add_argument("-v", "--verbose", action="store_true", help="Status updates on stderr too (standalone).")

    deck = p.add_argument_group("Flight Deck (default)")
    deck.add_argument("--deck-url", default="",
                      help="Use this running deck on this machine instead of starting a private one.")
    deck.add_argument("--deck-token", default="",
                      help="A sign-in token for it (or $CLAW_DECK_TOKEN); none for a sign-in-off deck.")
    deck.add_argument("--deck-port", type=int, default=0, help="Port for the private deck (default: a free one).")
    deck.add_argument("--archetype", default="",
                      help="Spawn this agent from the deck's agent store, by id (default: a plain new agent).")
    deck.add_argument("--list-store", action="store_true", help="List the agent store's ids and exit.")
    deck.add_argument("--keep-agent", action="store_true", help="On a --deck-url deck, leave the agent on its list.")

    alone = p.add_argument_group("standalone")
    alone.add_argument("--standalone", action="store_true", help="A bare in-process agent, no Flight Deck.")
    alone.add_argument("-c", "--config", default="", help="Config YAML (default: ./config.yaml as usual).")
    alone.add_argument("--bench", action="store_true", help="Benchmark profile (memory off, shell allowed).")
    alone.add_argument("--keep-home", action="store_true", help="Use the real HOME (~/.captain-claw).")
    return p


# ── paths ─────────────────────────────────────────────────────────────


def user_path(raw: str) -> Path:
    """*raw* as the caller meant it: "~" is their home even after the restart."""
    text = str(raw or "").strip()
    real_home = os.environ.get(_REAL_HOME_ENV)
    if real_home and (text == "~" or text.startswith("~/")):
        text = real_home + text[1:]
    return Path(text).expanduser().resolve()


def resolve_run_dir(raw: str) -> Path:
    if raw:
        run_dir = user_path(raw)
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir
    return Path(tempfile.mkdtemp(prefix="captain-once-")).resolve()


def restart_with_run_home(argv: list[str], run_dir: Path) -> None:
    """Start this command again with HOME inside the run folder (never returns).

    Many modules fix ``~/.captain-claw`` paths when they are first imported,
    and this module's package already imported the config, so HOME can only
    change for a new process.
    """
    home = run_dir / "home"
    home.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env[_REAL_HOME_ENV] = os.environ.get("HOME", "")
    env["HOME"] = str(home)
    env[_HOME_SET_ENV] = "1"
    child = [sys.executable, "-m", "captain_claw.bench.run_once", *argv, "--run-dir", str(run_dir)]
    sys.stdout.flush()
    sys.stderr.flush()
    os.execve(sys.executable, child, env)


def read_instruction(args: argparse.Namespace) -> str:
    if args.instruction_file:
        if args.instruction_file == "-":
            return sys.stdin.read().strip()
        return user_path(args.instruction_file).read_text(encoding="utf-8").strip()
    if args.instruction == "-":
        return sys.stdin.read().strip()
    if args.instruction:
        return args.instruction.strip()
    return os.environ.get(INSTRUCTION_ENV, "").strip()


# ── what changed in the workspace ─────────────────────────────────────


def snapshot(root: Path) -> dict[str, tuple[int, int]] | None:
    """``{relative path: (size, mtime_ns)}`` for files under *root*, or None
    when there are too many to be worth it."""
    files: dict[str, tuple[int, int]] = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _SNAPSHOT_SKIP]
        for name in filenames:
            path = Path(dirpath) / name
            try:
                st = path.stat()
            except OSError:
                continue
            files[path.relative_to(root).as_posix()] = (st.st_size, st.st_mtime_ns)
            if len(files) > _SNAPSHOT_LIMIT:
                return None
    return files


def diff_snapshots(before: dict | None, after: dict | None) -> dict[str, list[str]] | None:
    if before is None or after is None:
        return None
    return {
        "created": sorted(set(after) - set(before)),
        "changed": sorted(p for p in after.keys() & before.keys() if after[p] != before[p]),
        "deleted": sorted(set(before) - set(after)),
    }


# ── result ────────────────────────────────────────────────────────────


def tool_counts(messages: list[dict[str, Any]]) -> Counter:
    counts: Counter = Counter()
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "tool":
            counts[str(msg.get("tool_name") or "unknown")] += 1
    return counts


def estimate_cost(provider: str, model: str, usage: dict[str, int]) -> float | None:
    """USD for *usage* at litellm's list prices, or None when it can't tell."""
    try:
        import litellm

        from captain_claw.llm import _normalize_provider_name

        prompt_cost, completion_cost = litellm.cost_per_token(
            model=f"{_normalize_provider_name(provider)}/{model}",
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
            cache_read_input_tokens=int(usage.get("cache_read_input_tokens", 0)),
            cache_creation_input_tokens=int(usage.get("cache_creation_input_tokens", 0)),
        )
        return round(float(prompt_cost) + float(completion_cost), 6)
    except Exception:
        return None


def build_result_payload(
    *,
    ok: bool,
    error: str | None,
    answer: str,
    provider: str,
    model: str,
    effort: str,
    workspace: Path,
    run_dir: Path,
    started: float,
    usage: dict[str, int],
    messages: list[dict[str, Any]],
    counts: Counter,
    files: dict[str, list[str]] | None,
    session_id: str | None = None,
    cost: float | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from captain_claw import __version__

    if cost is None and usage:
        cost = estimate_cost(provider, model, usage)

    out: dict[str, Any] = {
        "ok": ok,
        "error": error,
        "answer": answer,
        "captain_version": __version__,
        "provider": provider,
        "model": model,
        "effort": effort or None,
        "workspace": str(workspace),
        "run_dir": str(run_dir),
        "session_id": session_id,
        "duration_seconds": round(time.monotonic() - started, 3),
        "usage": usage,
        "cost_usd_estimate": cost,
        "tool_calls": sum(counts.values()),
        "tools_used": dict(counts.most_common()),
        "files": files,
    }
    out.update(extra or {})
    out["messages"] = messages
    return out


def write_outputs(args: argparse.Namespace, result: dict[str, Any], result_path: Path) -> None:
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2, default=str, ensure_ascii=False), encoding="utf-8")
    if args.answer_file:
        target = user_path(args.answer_file)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(result.get("answer") or "", encoding="utf-8")
    if getattr(args, "trajectory_json", ""):
        from captain_claw.bench.atif import to_atif

        trajectory = to_atif(
            result.get("messages") or [],
            session_id=result.get("session_id") or (result.get("deck") or {}).get("task_id") or None,
            model=f"{result.get('provider')}/{result.get('model')}",
            version=str(result.get("captain_version") or ""),
            usage=result.get("usage") or None,
            cost_usd=result.get("cost_usd_estimate"),
        )
        if trajectory is not None:
            target = user_path(args.trajectory_json)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(trajectory, indent=2, default=str, ensure_ascii=False), encoding="utf-8")


def report(args: argparse.Namespace, result: dict[str, Any], result_path: Path) -> None:
    if not args.quiet:
        usage = result.get("usage") or {}
        state = "done" if result["ok"] else f"failed: {result['error']}"
        _say(
            f"{state} · {result['duration_seconds']:.0f}s · {result['tool_calls']} tool calls · "
            f"{usage.get('prompt_tokens', 0):,} in / {usage.get('completion_tokens', 0):,} out · {result_path}"
        )
    if result.get("answer"):
        print(result["answer"], flush=True)


# ── standalone: a bare agent in this process ──────────────────────────


def apply_bench_profile(cfg: Any) -> None:
    """Captain as shipped, minus what must not carry over or can't work here."""
    cfg.memory.enabled = False          # no recall from other runs, no embedding downloads
    cfg.deep_memory.enabled = False     # Typesense isn't there
    cfg.tools.shell.default_policy = type(cfg.tools.shell).ExecPolicy.ALLOW   # nobody is there to approve


def isolate_paths(cfg: Any, run_dir: Path) -> None:
    """Sessions and memory in the run folder, whatever the config says — a
    copied config often names the person's real databases."""
    cfg.session.path = str(run_dir / "sessions.db")
    cfg.memory.path = str(run_dir / "memory.db")


def build_config(args: argparse.Namespace, run_dir: Path, workspace: Path) -> Any:
    from captain_claw.config import Config

    cfg = Config.from_yaml(user_path(args.config)) if args.config else Config.load()
    if args.provider:
        cfg.model.provider = args.provider
    if args.model:
        cfg.model.model = args.model
    if args.base_url:
        cfg.model.base_url = args.base_url
    if args.max_context > 0:
        cfg.context.max_tokens = args.max_context
    if args.max_output > 0:
        cfg.model.max_tokens = args.max_output
    if args.vision in ("on", "off"):
        cfg.model.vision = args.vision
    cfg.workspace.path = str(workspace)
    if args.skills_dir:
        cfg.skills.load.extra_dirs = [*cfg.skills.load.extra_dirs, str(user_path(args.skills_dir))]
    isolate_paths(cfg, run_dir)
    if args.bench:
        apply_bench_profile(cfg)
    return cfg


def _say(text: str) -> None:
    print(f"[captain] {text}", file=sys.stderr, flush=True)


def _short(value: Any, limit: int = 160) -> str:
    text = value if isinstance(value, str) else json.dumps(value, default=str, ensure_ascii=False)
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


async def run_standalone(args: argparse.Namespace, instruction: str, run_dir: Path) -> int:
    from captain_claw.config import set_config

    workspace = user_path(args.workspace) if args.workspace else Path.cwd().resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    cfg = build_config(args, run_dir, workspace)
    set_config(cfg)

    if not args.sandboxed_writes:
        os.environ["CLAW_WRITE_DIRECT"] = "1"
    if args.max_iterations > 0:
        os.environ["CLAW_MAX_ITERATIONS"] = str(args.max_iterations)
    from captain_claw.bench.deck_run import lock_env

    os.environ.update(lock_env(args))

    from captain_claw.logging import configure_logging, set_system_log_sink

    log_file = open(run_dir / "captain.log", "a", encoding="utf-8")

    def _log_line(line: str) -> None:
        log_file.write(line + "\n")
        log_file.flush()

    set_system_log_sink(_log_line)
    configure_logging()

    from captain_claw import drive, msg_origin
    from captain_claw.agent import Agent
    from captain_claw.llm import reasoning

    def _status(text: str) -> None:
        if args.verbose and text:
            _say(_short(text))

    def _tool_output(name: str, arguments: dict[str, Any], _output: str) -> None:
        if not args.quiet:
            _say(f"tool {name} {_short(arguments)}")

    result_path = user_path(args.result_json) if args.result_json else run_dir / "result.json"
    started = time.monotonic()
    before = snapshot(workspace)
    agent: Any | None = None
    answer, error, ok = "", None, False
    try:
        agent = Agent(
            status_callback=_status,
            tool_output_callback=_tool_output,
            approval_callback=lambda _question: True,
        )
        await agent.initialize()
        if args.max_iterations > 0:
            agent.max_iterations = args.max_iterations
        try:
            asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, agent.cancel_event.set)
        except (NotImplementedError, RuntimeError):
            pass
        if not args.quiet:
            _say(f"standalone · {cfg.model.provider}/{cfg.model.model} · in {workspace}")
        msg_origin.hint_turn_provenance(agent, channel="api")
        with reasoning.turn_effort(args.effort or None), drive.turn_drive(args.drive or None):
            turn = agent.complete(instruction)
            answer = await (asyncio.wait_for(turn, args.timeout) if args.timeout > 0 else turn)
        ok = not getattr(agent, "_last_complete_cancelled", False)
        if not ok:
            error = "stopped"
    except TimeoutError:
        error = f"timed out after {args.timeout:g}s"
    except Exception as exc:  # the result file says what went wrong
        error = f"{type(exc).__name__}: {exc}"
        _log_line(f"run failed: {error}")

    session = getattr(agent, "session", None)
    messages = list(getattr(session, "messages", None) or [])
    result = build_result_payload(
        ok=ok, error=error, answer=answer or "", provider=cfg.model.provider, model=cfg.model.model,
        effort=args.effort, workspace=workspace, run_dir=run_dir, started=started,
        usage=dict(getattr(agent, "total_usage", None) or {}), messages=messages,
        counts=tool_counts(messages), files=diff_snapshots(before, snapshot(workspace)),
        session_id=getattr(session, "id", None),
        extra={"mode": "standalone", "drive": args.drive or None, "locked": bool(lock_env(args))},
    )
    write_outputs(args, result, result_path)
    report(args, result, result_path)
    log_file.close()
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> None:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(raw_argv)
    run_dir = resolve_run_dir(args.run_dir)
    if args.list_store:
        from captain_claw.bench.deck_run import list_store

        sys.exit(asyncio.run(list_store(args, run_dir)))
    if args.standalone and not args.keep_home and not os.environ.get(_HOME_SET_ENV):
        restart_with_run_home(raw_argv, run_dir)

    instruction = read_instruction(args)
    if not instruction:
        print("captain-claw-once: no instruction given", file=sys.stderr)
        sys.exit(2)
    if args.standalone:
        sys.exit(asyncio.run(run_standalone(args, instruction, run_dir)))

    from captain_claw.bench.deck_run import run_on_deck

    sys.exit(asyncio.run(run_on_deck(args, instruction, run_dir)))


if __name__ == "__main__":
    main()
