#!/usr/bin/env python3
"""Does Keep on course help? A report from agents' session stores.

Every turn that used tools keeps a record on its task anchor
(``session.metadata["task_course"]["turn_stats"]``, carried into
``task_course_closed`` when the task closes): how it ended, steps, tokens,
Jev course checks, and whether each corrective line was followed (the calls
after it worked on something other than the calls just before it).

Usage:
    python scripts/course_report.py fd-data/<agent>/data/sessions/sessions.db [more.db ...]
    python scripts/course_report.py --json fd-data/*/data/sessions/sessions.db

It reads the stores only (sqlite, read-only), never the conversations
themselves, and prints counts — nothing a person wrote. docs/jev-course-plan.md §5.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

OUTCOMES = ("finished", "budget", "stuck", "cancelled", "failed")


def turn_records(metadata: dict[str, Any]) -> Iterable[dict[str, Any]]:
    """Every per-turn record in one session's metadata (open task + closed)."""
    course = metadata.get("task_course")
    if isinstance(course, dict):
        yield from (t for t in course.get("turn_stats") or [] if isinstance(t, dict))
    for closed in metadata.get("task_course_closed") or []:
        if isinstance(closed, dict):
            yield from (t for t in closed.get("turn_stats") or [] if isinstance(t, dict))


def read_store(path: Path) -> list[dict[str, Any]]:
    """Per-turn records from one sessions.db (opened read-only)."""
    uri = f"file:{path}?mode=ro"
    out: list[dict[str, Any]] = []
    with sqlite3.connect(uri, uri=True) as conn:
        for (raw,) in conn.execute("SELECT metadata FROM sessions"):
            try:
                metadata = json.loads(raw or "{}")
            except json.JSONDecodeError:
                continue
            if isinstance(metadata, dict):
                out.extend(turn_records(metadata))
    return out


def _avg(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 1) if values else None


def summarize(turns: list[dict[str, Any]]) -> dict[str, Any]:
    """The numbers the report prints."""
    def group(name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
        outcomes = Counter(str(t.get("outcome") or "failed") for t in rows)
        finished = [t for t in rows if t.get("outcome") == "finished"]
        return {
            "group": name,
            "turns": len(rows),
            "outcomes": {o: outcomes.get(o, 0) for o in OUTCOMES},
            "finished_rate": round(outcomes.get("finished", 0) / len(rows), 3) if rows else None,
            "avg_tool_steps_finished": _avg([float(t.get("tool_steps") or 0) for t in finished]),
            "avg_tokens_finished": _avg([float(t.get("tokens") or 0) for t in finished]),
        }

    with_jev = [t for t in turns if t.get("jev")]
    without_jev = [t for t in turns if not t.get("jev")]
    with_lines = [t for t in turns if t.get("lines")]
    jev_no_lines = [t for t in with_jev if not t.get("lines")]

    per_line: dict[str, Counter] = defaultdict(Counter)
    for t in turns:
        for ln in t.get("lines") or []:
            if not isinstance(ln, dict):
                continue
            key = str(ln.get("line") or "?")
            f = ln.get("followed")
            per_line[key]["shown"] += 1
            per_line[key]["followed" if f is True else ("unknown" if f is None else "ignored")] += 1
    follow = {
        key: {**dict(c), "follow_rate": round(c["followed"] / (c["shown"] - c["unknown"]), 3)
              if c["shown"] - c["unknown"] else None}
        for key, c in sorted(per_line.items())
    }
    stages = Counter(int(round(t["final_stage"])) for t in turns
                     if isinstance(t.get("final_stage"), (int, float)))
    return {
        "turns": len(turns),
        "workers": sum(1 for t in turns if t.get("worker")),
        "groups": [group("all", turns), group("no Jev check (anchor only)", without_jev),
                   group("Jev checked", with_jev), group("Jev checked, no line", jev_no_lines),
                   group("a corrective line shown", with_lines)],
        "lines": follow,
        "final_stage": {str(k): stages[k] for k in sorted(stages)},
        "avg_checks_jev": _avg([float(t.get("checks") or 0) for t in with_jev]),
    }


def render(summary: dict[str, Any]) -> str:
    lines = [f"Turns with tools: {summary['turns']} (worker turns: {summary['workers']})", ""]
    head = f"{'group':<30} {'turns':>6} {'finished':>9} " + " ".join(f"{o:>9}" for o in OUTCOMES[1:]) \
        + f" {'steps✓':>7} {'tokens✓':>9}"
    lines += [head, "-" * len(head)]
    for g in summary["groups"]:
        rate = "-" if g["finished_rate"] is None else f"{g['finished_rate']:.0%}"
        lines.append(
            f"{g['group']:<30} {g['turns']:>6} {rate:>9} "
            + " ".join(f"{g['outcomes'][o]:>9}" for o in OUTCOMES[1:])
            + f" {g['avg_tool_steps_finished'] or '-':>7} {g['avg_tokens_finished'] or '-':>9}"
        )
    lines.append("")
    if summary["lines"]:
        lines.append("Corrective lines (followed = the next calls worked on something new):")
        for key, c in summary["lines"].items():
            rate = "-" if c.get("follow_rate") is None else f"{c['follow_rate']:.0%}"
            lines.append(f"  {key:<11} shown {c.get('shown', 0):>4} · followed {c.get('followed', 0):>4}"
                         f" · ignored {c.get('ignored', 0):>4} · unknown {c.get('unknown', 0):>4} · rate {rate}")
    else:
        lines.append("No corrective lines shown yet.")
    if summary["final_stage"]:
        lines.append("Last checked stage per turn: " + ", ".join(
            f"{k}: {v}" for k, v in summary["final_stage"].items()))
    if summary["avg_checks_jev"] is not None:
        lines.append(f"Jev checks per checked turn: {summary['avg_checks_jev']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stores", nargs="+", type=Path, help="sessions.db files")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args(argv)
    turns: list[dict[str, Any]] = []
    for path in args.stores:
        if not path.is_file():
            print(f"skipped (not a file): {path}", file=sys.stderr)
            continue
        try:
            turns.extend(read_store(path))
        except sqlite3.Error as exc:
            print(f"skipped ({exc}): {path}", file=sys.stderr)
    summary = summarize(turns)
    print(json.dumps(summary, indent=2) if args.json else render(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
