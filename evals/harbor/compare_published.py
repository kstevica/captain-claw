"""Our runs next to WildClawBench's published per-task scores, on the same tasks.

    python evals/harbor/compare_published.py --published <dir of *.jsonl> \
        --run "Captain · Haiku 5.5=results/haiku.csv" [--run "…=…"] --out comparison.md

The published scores are ``data/*.jsonl`` of the HF dataset
``internlm/WildClawBench-Trajectories`` (one file per model, OpenClaw harness).
Only the tasks our runs cover are compared; a task a run has no score for
counts as 0. Writes a Markdown table (overall and per category, best first)
and ``published_subset.csv`` beside it.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

CATEGORIES = {"01": "Productivity", "02": "Code", "03": "Social", "04": "Search",
              "05": "Creative", "06": "Safety"}


def _score(raw) -> float:
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 0.0


def load_published(folder: Path) -> dict[str, dict[str, float]]:
    """``{model: {task: score}}``."""
    out: dict[str, dict[str, float]] = {}
    for path in sorted(folder.glob("*.jsonl")):
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if rows:
            out[f"{rows[0].get('model', path.stem)} (OpenClaw)"] = {
                r["task_id"]: _score(r.get("score")) for r in rows}
    return out


def load_run(path: Path) -> dict[str, float]:
    with path.open(encoding="utf-8") as fh:
        return {r["task"]: _score(r["score"]) for r in csv.DictReader(fh)}


def table(runs: dict[str, dict[str, float]], published: dict[str, dict[str, float]], tasks: list[str]) -> str:
    cats = [c for k, c in CATEGORIES.items() if any(t.startswith(k + "_") for t in tasks)]
    keys = {c: k for k, c in CATEGORIES.items()}

    def mean(scores: dict[str, float], subset: list[str]) -> float:
        return sum(scores.get(t, 0.0) for t in subset) / len(subset) if subset else float("nan")

    rows = []
    for name, scores in {**runs, **published}.items():
        per = {c: mean(scores, [t for t in tasks if t.startswith(keys[c] + "_")]) for c in cats}
        rows.append((name, mean(scores, tasks), per, name in runs))
    rows.sort(key=lambda r: -r[1])
    head = "| Model | All |" + "".join(f" {c} |" for c in cats)
    lines = [f"Tasks compared: {len(tasks)}", "", head, "|---|---|" + "---|" * len(cats)]
    for name, overall, per, ours in rows:
        label = f"**{name}**" if ours else name
        lines.append(f"| {label} | {overall:.3f} |" + "".join(f" {per[c]:.3f} |" for c in cats))
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--published", required=True, help="Folder of the published *.jsonl files.")
    p.add_argument("--run", action="append", required=True, help='"Name=path/to/run.csv" (repeatable).')
    p.add_argument("--out", required=True, help="Markdown file to write.")
    args = p.parse_args(argv)
    runs = {}
    for spec in args.run:
        name, _, path = spec.partition("=")
        runs[name] = load_run(Path(path))
    tasks = sorted(set().union(*(set(r) for r in runs.values())))
    published = load_published(Path(args.published))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(table(runs, published, tasks), encoding="utf-8")
    with (out.parent / "published_subset.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["task", *published])
        for t in tasks:
            writer.writerow([t, *(published[m].get(t, "") for m in published)])
    print(f"{len(tasks)} tasks, {len(published)} published models → {out}")


if __name__ == "__main__":
    main()
