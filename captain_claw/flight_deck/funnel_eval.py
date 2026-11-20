"""List funnel test sets: make one from a recorded run, score a run against it.

A change to the funnel was measured by running it against the live web and
reading the result, twenty minutes a try, the web different each time. A test
set fixes what the web returned:

* a run with recording on (``config.funnel_record`` or CLAW_FUNNEL_RECORD) keeps
  every search and page it read in ``shared/funnel/tape/``;
* :func:`make_set` turns that run into a set: the tape, the recipe it followed,
  and ``expected.csv`` drafted from what the run decided, for a person to correct
  (the run's own verdicts are a draft, not the truth);
* a run given the set as ``extra/tape/`` replays it (``funnel.TapeTools``): same
  recipe, same searches, same pages, no web;
* :func:`score` compares what a run decided with ``expected.csv``.

``expected.csv``: ``name``, ``verdict`` (keep, close or drop), ``group`` (when
the list has groups), then any of the list's columns whose value is known. An
empty cell isn't checked. Nothing here knows what kind of list it is.

    python -m captain_claw.flight_deck.funnel_eval make  <run folder> <set folder>
    python -m captain_claw.flight_deck.funnel_eval score <run folder> [<expected.csv>]
"""

from __future__ import annotations

import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Any

from captain_claw.flight_deck import funnel
from captain_claw.flight_deck import funnel_extract as fx

VERDICTS = ("keep", "close", "drop")
_OF_STATUS = {"kept": "keep", "maybe": "close", "loose": "close", "handed": "close", "dropped": "drop"}
EXPECTED = "expected.csv"
REPORT = "test-report.md"


def outcomes(folder: Path) -> dict[str, dict[str, str]]:
    """What a run decided for each item, by its name's key: its last word on it (an
    item found again under another name doesn't change what it was)."""
    out: dict[str, dict[str, str]] = {}
    for r in funnel._read_csv(folder / funnel.ITEMS):
        if r.get("stage") == "duplicate":
            continue
        verdict = _OF_STATUS.get(r.get("status", ""))
        if not verdict:
            continue
        for key in {funnel.norm_key(r.get("name", "")), funnel.norm_key(r.get("key", ""))} - {""}:
            out[key] = {"name": r.get("name", ""), "verdict": verdict, "group": r.get("group", ""),
                        "reason": r.get("reason", ""), "stage": r.get("stage", "")}
    return out


def _list_rows(folder: Path, recipe: dict[str, Any]) -> dict[str, dict[str, str]]:
    key = recipe.get("key") or (recipe["columns"][0] if recipe.get("columns") else "")
    return {funnel.norm_key(r.get(key, "")): r for r in funnel._read_csv(folder / funnel.ROWS) if r.get(key)}


def make_set(run_folder: Path, set_folder: Path) -> dict[str, int]:
    """A test set from a recorded run: its tape, its recipe, and a draft of the
    expected answers (one line per item it looked at, in the order looked at)."""
    tape = run_folder / funnel.TAPE_OUT
    recipe = funnel.load_recipe(run_folder)
    if recipe is None or not tape.is_dir():
        raise SystemExit(f"{run_folder} has no recording: run it with recording on (funnel_record)")
    set_folder.mkdir(parents=True, exist_ok=True)
    copied = 0
    for name in ("searches.jsonl", "pages.jsonl"):
        if (tape / name).is_file():
            shutil.copyfile(tape / name, set_folder / name)
            copied += sum(1 for _ in (set_folder / name).open(encoding="utf-8"))
    (set_folder / "recipe.json").write_text(json.dumps(recipe, indent=1, ensure_ascii=False), encoding="utf-8")
    got, rows = outcomes(run_folder), _list_rows(run_folder, recipe)
    cols = ["name", "verdict", "group"] + [c for c in recipe["columns"] if c not in ("name", "verdict", "group")]
    seen: set[str] = set()
    lines = []
    for r in funnel._read_csv(run_folder / funnel.ITEMS):
        k = funnel.norm_key(r.get("name", ""))
        if not k or k in seen or k not in got:
            continue
        seen.add(k)
        row = rows.get(k, {}) if got[k]["verdict"] == "keep" else {}
        lines.append({"name": r.get("name", ""), "verdict": got[k]["verdict"], "group": got[k]["group"],
                      **{c: row.get(c, "") for c in cols[3:]}})
    with (set_folder / EXPECTED).open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(lines)
    return {"recorded": copied, "items": len(lines)}


def score(run_folder: Path, expected: Path | None = None) -> dict[str, Any]:
    """A run against the expected answers: verdicts, groups and cells right, the wrong
    keeps (the costly mistake) and the missed ones by name, and what the tape lacked."""
    expected = expected or run_folder / funnel.TAPE_IN / EXPECTED
    want = [r for r in funnel._read_csv(expected) if (r.get("verdict") or "").strip().lower() in VERDICTS]
    recipe = funnel.load_recipe(run_folder) or {"columns": [], "key": ""}
    got, rows = outcomes(run_folder), _list_rows(run_folder, recipe)
    out: dict[str, Any] = {"expected": len(want), "judged": 0, "verdicts_right": 0, "groups": 0, "groups_right": 0,
                           "cells": 0, "cells_right": 0, "wrong_keeps": [], "missed_keeps": [], "not_looked_at": [],
                           "wrong": [], "turned_away": []}
    for w in want:
        name, verdict = w.get("name", ""), w["verdict"].strip().lower()
        g = got.get(funnel.norm_key(name))
        if g is None:
            out["not_looked_at"].append(name)
            continue
        if g.get("stage") == "full":  # its group had its rows already: not judged, so neither right nor wrong
            out["turned_away"].append(name)
            continue
        out["judged"] += 1
        if g["verdict"] == verdict:
            out["verdicts_right"] += 1
        else:
            out["wrong"].append(f"{name}: {g['verdict']}, expected {verdict} ({g['reason'][:160]})")
            if g["verdict"] == "keep":
                out["wrong_keeps"].append(name)
            elif verdict == "keep":
                out["missed_keeps"].append(name)
        if (w.get("group") or "").strip() and verdict != "drop":
            out["groups"] += 1
            out["groups_right"] += int(g["group"].strip().lower() == w["group"].strip().lower())
        row = rows.get(funnel.norm_key(name))
        if verdict == "keep" and row is not None:
            for col, value in w.items():
                if col in ("name", "verdict", "group") or not (value or "").strip() or col not in recipe["columns"]:
                    continue
                out["cells"] += 1
                have = row.get(col, "")
                out["cells_right"] += int(bool(have) and (fx.in_text(value, have) or fx.in_text(have, value)))
    try:
        misses = json.loads((run_folder / funnel.TAPE_MISSES).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        misses = {}
    out["tape"] = {"asked": misses.get("asked") or {},
                   "missed": {k: len(v) for k, v in (misses.get("missed") or {}).items()}}
    return out


def report(result: dict[str, Any]) -> str:
    def share(a: int, b: int) -> str:
        return f"{a} of {b}" + (f" ({round(100 * a / b)}%)" if b else "")

    lines = ["# Test set report", "",
             f"- Verdicts right: {share(result['verdicts_right'], result['judged'])}",
             f"- Wrong keeps: {len(result['wrong_keeps'])}" + (f" ({', '.join(result['wrong_keeps'][:12])})"
                                                               if result["wrong_keeps"] else ""),
             f"- Missed keeps: {len(result['missed_keeps'])}" + (f" ({', '.join(result['missed_keeps'][:12])})"
                                                                 if result["missed_keeps"] else "")]
    if result["groups"]:
        lines.append(f"- Groups right: {share(result['groups_right'], result['groups'])}")
    if result["cells"]:
        lines.append(f"- Cells right on kept rows: {share(result['cells_right'], result['cells'])}")
    if result.get("turned_away"):
        lines.append(f"- Not judged, their group was full: {len(result['turned_away'])} "
                     f"({', '.join(result['turned_away'][:12])})")
    if result["not_looked_at"]:
        lines.append(f"- Not looked at: {len(result['not_looked_at'])} of {result['expected']} expected "
                     f"({', '.join(result['not_looked_at'][:12])})")
    tape = result.get("tape") or {}
    if tape.get("asked"):
        asked, missed = tape["asked"], tape.get("missed") or {}
        lines.append(f"- Not on the tape: {missed.get('searches', 0)} of {asked.get('searches', 0)} searches, "
                     f"{missed.get('pages', 0)} of {asked.get('pages', 0)} pages")
    if result["wrong"]:
        lines += ["", "## Wrong", ""] + [f"- {w}" for w in result["wrong"][:60]]
    return "\n".join(lines) + "\n"


def write_report(run_folder: Path) -> str:
    """The report for a replayed run, written next to its list; "" when the run has no
    expected answers."""
    if not (run_folder / funnel.TAPE_IN / EXPECTED).is_file():
        return ""
    text = report(score(run_folder))
    (run_folder / REPORT).write_text(text, encoding="utf-8")
    return text


def main(argv: list[str]) -> int:
    if len(argv) >= 3 and argv[0] == "make":
        got = make_set(Path(argv[1]), Path(argv[2]))
        print(f"{argv[2]}: {got['recorded']} searches and pages, {got['items']} items in {EXPECTED} "
              "(a draft from the run's own verdicts: correct it by hand)")
        return 0
    if len(argv) >= 2 and argv[0] == "score":
        print(report(score(Path(argv[1]), Path(argv[2]) if len(argv) > 2 else None)), end="")
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
