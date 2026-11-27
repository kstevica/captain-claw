"""scripts/course_report.py: reads per-turn records from session stores."""

from __future__ import annotations

import importlib.util
import json
import sqlite3
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "course_report", Path(__file__).resolve().parents[1] / "scripts" / "course_report.py")
report = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(report)


def _store(path: Path, metas: list[dict]) -> Path:
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE sessions (id TEXT PRIMARY KEY, name TEXT, messages TEXT, "
                 "created_at TEXT, updated_at TEXT, metadata TEXT)")
    for i, meta in enumerate(metas):
        conn.execute("INSERT INTO sessions VALUES (?, 'default', '[]', '', '', ?)", (f"s{i}", json.dumps(meta)))
    conn.execute("INSERT INTO sessions VALUES ('broken', 'x', '[]', '', '', 'not json')")
    conn.commit()
    conn.close()
    return path


TURNS_OPEN = [
    {"outcome": "finished", "jev": True, "tool_steps": 10, "tokens": 5000, "checks": 2,
     "final_stage": 4.6, "lines": [{"line": "refocus", "at": 3, "followed": True}]},
    {"outcome": "budget", "jev": False, "tool_steps": 40, "tokens": 90000, "lines": []},
]
TURNS_CLOSED = [
    {"outcome": "finished", "jev": True, "tool_steps": 6, "tokens": 3000, "checks": 1, "worker": True,
     "lines": [{"line": "refocus", "at": 2, "followed": False}, {"line": "finish", "at": 6, "followed": None}]},
]


def test_summary_splits_by_jev_and_lines_and_counts_follow_rate(tmp_path):
    db = _store(tmp_path / "sessions.db", [
        {"task_course": {"turn_stats": TURNS_OPEN},
         "task_course_closed": [{"turn_stats": TURNS_CLOSED}]},
        {"other": 1},
    ])
    s = report.summarize(report.read_store(db))
    assert s["turns"] == 3 and s["workers"] == 1
    groups = {g["group"]: g for g in s["groups"]}
    assert groups["all"]["outcomes"]["finished"] == 2 and groups["all"]["outcomes"]["budget"] == 1
    assert groups["Jev checked"]["turns"] == 2 and groups["Jev checked"]["finished_rate"] == 1.0
    assert groups["no Jev check (anchor only)"]["finished_rate"] == 0.0
    assert groups["a corrective line shown"]["avg_tool_steps_finished"] == 8.0
    assert s["lines"]["refocus"] == {"shown": 2, "followed": 1, "ignored": 1, "follow_rate": 0.5}
    assert s["lines"]["finish"]["follow_rate"] is None
    assert s["final_stage"] == {"5": 1}
    text = report.render(s)
    assert "Turns with tools: 3" in text and "refocus" in text and "rate 50%" in text


def test_main_prints_json_and_skips_missing_stores(tmp_path, capsys):
    db = _store(tmp_path / "a.db", [{"task_course": {"turn_stats": TURNS_OPEN}}])
    assert report.main(["--json", str(db), str(tmp_path / "missing.db")]) == 0
    out = capsys.readouterr()
    assert json.loads(out.out)["turns"] == 2
    assert "skipped" in out.err
