#!/usr/bin/env python3
"""Mrav eval harness — canned agentic tasks against a real small model.

Runs the REAL end-to-end path (Agent with mrav.enabled → MravRuntime →
tool registry) in a scratch workspace, one fresh agent per task, and prints
a pass/fail + token/time table. This is the gate for blessing models into
the micro roster and the regression net for any prompt change — and, for
Captain Swarm, the first number it improves (docs/captain-swarm-plan.md).

Tasks come from a JSON file (``--tasks``, default ``evals/mrav/dev.json``):
each has a prompt, the files it starts with, and checks in Captain Swarm's
predicate shapes (``captain_swarm_score``). A held-out set lives outside the
repo, so no change can be fitted to it.

``--repeat N`` runs every task N times (the noise is part of the number).
``--isolate`` runs the agent under test in a child process that gets only
the prompt and its workdir, importing ``captain_claw`` from ``--code-root``;
the checks stay here, in this copy of the harness. That is how Captain Swarm
measures a variant's checkout with a harness it pinned. ``--json`` prints
one JSON object (progress goes to stderr).

Usage:
  .venv/bin/python scripts/mrav_eval.py --model gemma4:e2b
  .venv/bin/python scripts/mrav_eval.py --provider ollama --model qwen3.5:4b \
      --base-url http://127.0.0.1:11434 --only write-read,count-files
  .venv/bin/python scripts/mrav_eval.py --model gemma4:e2b --repeat 3 --isolate \
      --code-root /path/to/variant --tasks /path/to/holdout.json --json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

HARNESS_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HARNESS_ROOT))

from captain_claw.flight_deck.captain_swarm_score import (  # noqa: E402
    check_predicates,
    stats,
)

DEFAULT_TASKS = HARNESS_ROOT / "evals" / "mrav" / "dev.json"
RESULT_MARK = "@@MRAV_RESULT@@"


@dataclass
class Task:
    id: str
    prompt: str
    files: dict[str, str] = field(default_factory=dict)
    dirs: list[str] = field(default_factory=list)
    checks: list[dict] = field(default_factory=list)


def load_tasks(path: Path) -> list[Task]:
    """The tasks in *path*. A task with no checks is refused: it would pass
    by doing nothing."""
    data = json.loads(path.read_text(encoding="utf-8"))
    tasks = []
    for raw in data.get("tasks") or []:
        setup = raw.get("setup") or {}
        task = Task(id=str(raw["id"]), prompt=str(raw["prompt"]),
                    files={str(k): str(v) for k, v in (setup.get("files") or {}).items()},
                    dirs=[str(d) for d in setup.get("dirs") or []],
                    checks=[dict(c) for c in raw.get("checks") or []])
        if not task.checks:
            raise SystemExit(f"task {task.id} in {path} has no checks")
        tasks.append(task)
    ids = [t.id for t in tasks]
    if len(set(ids)) != len(ids):
        raise SystemExit(f"duplicate task ids in {path}")
    return tasks


def seed(task: Task, workdir: Path) -> None:
    workdir.mkdir(parents=True, exist_ok=True)
    for d in task.dirs:
        (workdir / d).mkdir(parents=True, exist_ok=True)
    for name, content in task.files.items():
        target = workdir / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")


def judge(task: Task, workdir: Path, reply: str) -> list[dict]:
    """The task's checks. The write tool sandboxes relative paths into
    saved/tmp/<session>/, so files are found anywhere under the workdir."""
    return check_predicates(workdir, reply, task.checks, anywhere=True)


# ── running the agent ────────────────────────────────────────────────


def _configure(spec: dict, workdir: Path) -> None:
    from captain_claw.config import get_config

    cfg = get_config()
    cfg.mrav.enabled = True
    cfg.model.provider = spec["provider"]
    cfg.model.model = spec["model"]
    if spec.get("base_url"):
        cfg.model.base_url = spec["base_url"]
    if hasattr(cfg, "workspace"):
        cfg.workspace.path = str(workdir)
    if hasattr(cfg, "botport"):
        cfg.botport.enabled = False
    if hasattr(cfg, "memory"):
        cfg.memory.enabled = False


def _usage_of(agent) -> tuple[int, dict]:
    runtime = getattr(agent, "_mrav_runtime_cache", None)
    steps = getattr(getattr(runtime, "board", None), "step", 0)
    # Prefer the runtime's own counters — they survive a task timeout,
    # where agent.last_usage was never merged.
    usage = dict(getattr(runtime, "last_usage", None) or getattr(agent, "last_usage", {}) or {})
    return steps, usage


async def run_in_process(task: Task, spec: dict, workdir: Path, timeout: float) -> dict:
    from captain_claw.agent import Agent

    os.chdir(workdir)
    _configure(spec, workdir)
    agent = Agent(approval_callback=lambda _prompt: True)
    reply, error = "", ""
    try:
        reply = await asyncio.wait_for(agent.complete(task.prompt), timeout=timeout) or ""
    except TimeoutError:
        error = "timeout"
    except Exception as exc:  # noqa: BLE001 — eval must report, not crash
        error = f"{type(exc).__name__}: {exc}"
    steps, usage = _usage_of(agent)
    return {"reply": reply, "error": error, "steps": steps, "usage": usage, "imported_from": ""}


# The child of --isolate: this text is the harness's, so a variant can't
# change how it is run. It imports captain_claw from PYTHONPATH (--code-root),
# answers once and leaves without joining the agent's non-daemon thread.
_CHILD = r'''
import sys
if sys.path and sys.path[0] in ("", "."):
    sys.path.pop(0)
import asyncio, json, os
spec = json.loads(sys.stdin.read())
os.chdir(spec["workdir"])
import captain_claw
from captain_claw.config import get_config
cfg = get_config()
cfg.mrav.enabled = True
cfg.model.provider = spec["provider"]
cfg.model.model = spec["model"]
if spec.get("base_url"):
    cfg.model.base_url = spec["base_url"]
if hasattr(cfg, "workspace"):
    cfg.workspace.path = spec["workdir"]
if hasattr(cfg, "botport"):
    cfg.botport.enabled = False
if hasattr(cfg, "memory"):
    cfg.memory.enabled = False
from captain_claw.agent import Agent
async def _main():
    agent = Agent(approval_callback=lambda _prompt: True)
    try:
        return agent, (await agent.complete(spec["prompt"])) or "", ""
    except Exception as exc:
        return agent, "", f"{type(exc).__name__}: {exc}"
agent, reply, error = asyncio.run(_main())
runtime = getattr(agent, "_mrav_runtime_cache", None)
steps = getattr(getattr(runtime, "board", None), "step", 0)
usage = dict(getattr(runtime, "last_usage", None) or getattr(agent, "last_usage", {}) or {})
out = {"reply": reply, "error": error, "steps": steps, "usage": usage,
       "imported_from": os.path.dirname(os.path.dirname(os.path.abspath(captain_claw.__file__)))}
sys.stdout.write("\n''' + RESULT_MARK + r'''" + json.dumps(out, default=str) + "\n")
sys.stdout.flush()
os._exit(0)
'''


async def run_isolated(task: Task, spec: dict, workdir: Path, timeout: float, code_root: Path) -> dict:
    env = {**os.environ, "PYTHONPATH": str(code_root), "PYTHONDONTWRITEBYTECODE": "1"}
    proc = await asyncio.create_subprocess_exec(
        sys.executable, "-P", "-c", _CHILD, cwd=str(workdir), env=env,
        stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    payload = json.dumps({**spec, "prompt": task.prompt, "workdir": str(workdir)}).encode()
    try:
        out, err = await asyncio.wait_for(proc.communicate(payload), timeout=timeout)
    except TimeoutError:
        proc.kill()
        await proc.wait()
        return {"reply": "", "error": "timeout", "steps": 0, "usage": {}, "imported_from": ""}
    text = out.decode("utf-8", errors="replace")
    for line in reversed(text.splitlines()):
        if line.startswith(RESULT_MARK):
            result = json.loads(line[len(RESULT_MARK):])
            imported = Path(result.get("imported_from") or "").resolve()
            if imported != code_root.resolve():
                # Measuring the wrong code is worse than not measuring.
                result["error"] = f"imported captain_claw from {imported}, not {code_root}"
            return result
    tail = err.decode("utf-8", errors="replace").strip()[-400:]
    return {"reply": "", "error": f"the child gave no result (exit {proc.returncode}): {tail}",
            "steps": 0, "usage": {}, "imported_from": ""}


async def run_task(task: Task, args: argparse.Namespace, workdir: Path) -> dict:
    seed(task, workdir)
    spec = {"provider": args.provider, "model": args.model, "base_url": args.base_url}
    started = time.monotonic()
    if args.isolate:
        got = await run_isolated(task, spec, workdir, args.timeout, Path(args.code_root).resolve())
    else:
        got = await run_in_process(task, spec, workdir, args.timeout)
    elapsed = time.monotonic() - started
    reply, error = got.get("reply") or "", got.get("error") or ""
    if error == "timeout":
        outcome, checks = "TIMEOUT", []
    elif error and not reply:
        outcome, checks = "ERROR", []
    else:
        checks = judge(task, workdir, reply)
        outcome = "pass" if all(c["ok"] for c in checks) and not error.startswith("imported") else "FAIL"
    usage = got.get("usage") or {}
    return {
        "task": task.id,
        "outcome": outcome,
        "steps": int(got.get("steps") or 0),
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "completion_tokens": int(usage.get("completion_tokens") or 0),
        "seconds": round(elapsed, 1),
        "reply": reply.replace("\n", " ")[:200],
        "error": error[:300],
        "failed": [c["detail"] for c in checks if not c["ok"]],
    }


# ── the number ───────────────────────────────────────────────────────


def summarize(results: list[dict], repeats: int) -> dict:
    """The pass rate over every run, its spread across repeats, and the
    guardrail numbers (tokens and seconds per task)."""
    total = len(results)
    passed = sum(r["outcome"] == "pass" for r in results)
    per_repeat = []
    for k in range(repeats):
        runs = [r for r in results if r["repeat"] == k]
        if runs:
            per_repeat.append(sum(r["outcome"] == "pass" for r in runs) / len(runs))
    per_task: dict[str, float] = {}
    for tid in dict.fromkeys(r["task"] for r in results):
        runs = [r for r in results if r["task"] == tid]
        per_task[tid] = round(sum(r["outcome"] == "pass" for r in runs) / len(runs), 4)
    spread = stats(per_repeat)
    outcomes: dict[str, int] = {}
    for r in results:
        outcomes[r["outcome"]] = outcomes.get(r["outcome"], 0) + 1
    return {
        "pass_rate": round(passed / total, 4) if total else 0.0,
        "passed": passed,
        "total": total,
        "per_repeat": [round(x, 4) for x in per_repeat],
        "stdev": round(spread.stdev, 4),
        "tokens_per_task": round(sum(r["prompt_tokens"] + r["completion_tokens"] for r in results) / total, 1)
        if total else 0.0,
        "seconds_per_task": round(sum(r["seconds"] for r in results) / total, 1) if total else 0.0,
        "outcomes": outcomes,
        "per_task": per_task,
    }


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--provider", default="ollama")
    parser.add_argument("--model", required=True, help="e.g. gemma4:e2b, qwen3.5:4b")
    parser.add_argument("--base-url", default="")
    parser.add_argument("--tasks", default=str(DEFAULT_TASKS), help="a task file (JSON)")
    parser.add_argument("--only", default="", help="comma-separated task ids")
    parser.add_argument("--repeat", type=int, default=1, help="runs of every task")
    parser.add_argument("--isolate", action="store_true", help="run the agent in a child process")
    parser.add_argument("--code-root", default=str(HARNESS_ROOT),
                        help="with --isolate: where the agent under test imports captain_claw from")
    parser.add_argument("--timeout", type=float, default=300.0, help="seconds per task")
    parser.add_argument("--json", action="store_true", help="print one JSON object")
    parser.add_argument("--keep", action="store_true", help="keep the scratch workspace")
    args = parser.parse_args()
    say = (lambda *a: print(*a, file=sys.stderr, flush=True)) if args.json else \
        (lambda *a: print(*a, flush=True))

    tasks = load_tasks(Path(args.tasks))
    wanted = {t.strip() for t in args.only.split(",") if t.strip()}
    tasks = [t for t in tasks if not wanted or t.id in wanted]
    if not tasks:
        say(f"no tasks match --only={args.only}")
        return 2
    repeats = max(1, args.repeat)

    base_dir = Path(tempfile.mkdtemp(prefix="mrav-eval-"))
    original_cwd = os.getcwd()
    say(f"mrav eval — {args.provider}/{args.model} — {len(tasks)} task(s) × {repeats} — workspace {base_dir}\n")

    results = []
    for k in range(repeats):
        for task in tasks:
            say(f"→ {task.id} (run {k + 1}/{repeats}) …")
            r = await run_task(task, args, base_dir / f"r{k}" / task.id)
            r["repeat"] = k
            results.append(r)
            if r["outcome"] != "pass":
                say(f"  {r['outcome']}: {'; '.join(r['failed']) or r['error']}")
    os.chdir(original_cwd)
    summary = summarize(results, repeats)

    if args.json:
        print(json.dumps({"provider": args.provider, "model": args.model, "tasks": str(args.tasks),
                          "isolate": args.isolate, "code_root": str(Path(args.code_root).resolve()),
                          "repeat": repeats, "summary": summary, "results": results}))
    else:
        header = f"{'task':<16} {'run':>3} {'outcome':<8} {'steps':>5} {'in-tok':>8} {'out-tok':>8} {'sec':>7}  reply"
        print("\n" + header)
        print("-" * len(header))
        for r in results:
            print(f"{r['task']:<16} {r['repeat'] + 1:>3} {r['outcome']:<8} {r['steps']:>5} "
                  f"{r['prompt_tokens']:>8} {r['completion_tokens']:>8} {r['seconds']:>7}  {r['reply'][:100]}")
        print(f"\n{summary['passed']}/{summary['total']} passed "
              f"(pass rate {summary['pass_rate']:.2f}, ± {summary['stdev']:.2f} across runs)")

    if not args.keep:
        shutil.rmtree(base_dir, ignore_errors=True)
    else:
        say(f"workspace kept: {base_dir}")
    return 0 if summary["passed"] == summary["total"] else 1


if __name__ == "__main__":
    code = asyncio.run(main())
    # Agent init spawns a non-daemon thread that blocks interpreter
    # shutdown (pre-existing; full pytest runs of tests/test_agent hit the
    # same wait). The report is printed — leave without joining it.
    sys.stdout.flush()
    os._exit(code)
