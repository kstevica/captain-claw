"""One row per task from Harbor job folders, as CSV and a Markdown table.

    python evals/harbor/summarize_jobs.py <job dir> [<job dir> …] --out <folder> [--label haiku]

Reads each trial's ``result.json`` (Harbor) and ``agent/captain-result.json``
(captain-claw-once). When a task appears in several jobs, the latest finished
trial that got a score wins — a rerun replaces an earlier attempt. Writes
``<label>.csv`` and ``<label>.md`` with the score, run time, tokens, cost and
settings per task, and per-category and overall means.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

FIELDS = ("task", "category", "score", "status", "agent_seconds", "tool_calls", "input_tokens",
          "output_tokens", "cost_usd", "model", "effort", "drive", "store_agent", "job")


def _load(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _seconds(span: dict[str, Any] | None) -> float | None:
    from datetime import datetime

    try:
        start = datetime.fromisoformat(str(span["started_at"]).replace("Z", "+00:00"))
        end = datetime.fromisoformat(str(span["finished_at"]).replace("Z", "+00:00"))
    except (KeyError, TypeError, ValueError):
        return None
    return round((end - start).total_seconds(), 1)


def trial_row(trial_dir: Path, job: str) -> dict[str, Any] | None:
    harbor = _load(trial_dir / "result.json")
    if not harbor.get("finished_at"):
        return None
    captain = _load(trial_dir / "agent" / "captain-result.json")
    rewards = (harbor.get("verifier_result") or {}).get("rewards") or {}
    score = rewards.get("overall_score", rewards.get("reward"))
    exc = (harbor.get("exception_info") or {}).get("exception_type")
    usage = captain.get("usage") or {}
    task = str(harbor.get("task_name") or trial_dir.name.rsplit("__", 1)[0])
    return {
        "task": task,
        "category": task.split("_task_")[0] if "_task_" in task else "",
        "score": score,
        "status": exc or ("ok" if score is not None else "no score"),
        "agent_seconds": _seconds(harbor.get("agent_execution")),
        "tool_calls": captain.get("tool_calls"),
        "input_tokens": usage.get("prompt_tokens"),
        "output_tokens": usage.get("completion_tokens"),
        "cost_usd": captain.get("cost_usd_estimate"),
        "model": "/".join(x for x in (captain.get("provider"), captain.get("model")) if x),
        "effort": captain.get("effort"),
        "drive": captain.get("drive"),
        "store_agent": captain.get("store_agent") or "",
        "job": job,
        "_finished": str(harbor.get("finished_at")),
    }


def collect(job_dirs: list[Path]) -> list[dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for job_dir in job_dirs:
        for result in sorted(job_dir.glob("*/result.json")):
            row = trial_row(result.parent, job_dir.name)
            if row is None:
                continue
            kept = best.get(row["task"])
            scored, kept_scored = row["score"] is not None, kept is not None and kept["score"] is not None
            if kept is None or (scored and not kept_scored) or (
                    scored == kept_scored and row["_finished"] > kept["_finished"]):
                best[row["task"]] = row
    return [best[t] for t in sorted(best)]


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def write(rows: list[dict[str, Any]], out: Path, label: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    with (out / f"{label}.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    by_cat: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row["score"] is not None:
            by_cat[row["category"]].append(float(row["score"]))
    scored = [float(r["score"]) for r in rows if r["score"] is not None]
    lines = [f"# {label}", "",
             f"Tasks: {len(rows)} · scored: {len(scored)} · mean score (unscored count as 0): "
             f"{_mean([float(r['score'] or 0) for r in rows])}", "",
             "| Category | Tasks | Mean |", "|---|---|---|"]
    lines += [f"| {cat} | {len(v)} | {_mean(v)} |" for cat, v in sorted(by_cat.items())]
    lines += ["", "| Task | Score | Status | Agent s | Tools | In tok | Out tok | $ | Store agent |",
              "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['task']} | {r['score']} | {r['status']} | {r['agent_seconds']} | {r['tool_calls']} | "
                     f"{r['input_tokens']} | {r['output_tokens']} | {r['cost_usd']} | {r['store_agent']} |")
    (out / f"{label}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("jobs", nargs="+", help="Harbor job folders (or folders of job folders).")
    p.add_argument("--out", required=True, help="Where to write <label>.csv and <label>.md.")
    p.add_argument("--label", default="results")
    args = p.parse_args(argv)
    job_dirs: list[Path] = []
    for raw in args.jobs:
        path = Path(raw)
        job_dirs += [path] if (path / "config.json").exists() else sorted(d for d in path.iterdir() if d.is_dir())
    rows = collect(job_dirs)
    write(rows, Path(args.out), args.label)
    print(f"{len(rows)} tasks → {Path(args.out) / (args.label + '.md')}")


if __name__ == "__main__":
    main()
