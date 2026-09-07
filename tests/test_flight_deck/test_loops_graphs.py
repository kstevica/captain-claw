"""Unit tests for the loops-and-graphs improvements (R6, R8).

Pure-function coverage for the deterministic helpers added by:
  * R6 — return the failing unit with an explicit scope (code_routes)
  * R8 — feed the run's trajectory into the autonomy judge (fd_dispatch)
"""

from __future__ import annotations

from captain_claw.flight_deck import code_routes, fd_dispatch


# ── R6: scope helpers ────────────────────────────────────────────────────
def test_scope_from_findings_dedupes_and_drops_empty():
    findings = [
        {"file": "a.py", "severity": "blocking", "title": "x"},
        {"file": "a.py", "severity": "major", "title": "y"},   # dup file
        {"file": "", "severity": "major", "title": "no file"},  # empty
        {"file": "b.py", "severity": "minor", "title": "z"},
        {"severity": "minor", "title": "missing key"},           # no file key
    ]
    assert code_routes._scope_from_findings(findings) == ["a.py", "b.py"]
    assert code_routes._scope_from_findings([]) == []
    assert code_routes._scope_from_findings(None) == []


def test_fix_prompt_without_findings_matches_pre_r6_shape():
    # Empty findings/scope must reproduce the original prompt content.
    p = code_routes._fix_prompt("build a thing", "fix the bug")
    assert "Failing units" not in p
    assert "SCOPE —" not in p
    assert "fix the bug" in p
    assert "build a thing" in p


def test_fix_prompt_with_findings_adds_units_and_scope():
    findings = [{"file": "auth.py", "severity": "blocking", "title": "redirect wrong"}]
    scope = code_routes._scope_from_findings(findings)
    p = code_routes._fix_prompt("do it", "fix redirect", findings=findings, scope_files=scope)
    assert "Failing units" in p
    assert "auth.py" in p
    assert "SCOPE —" in p
    assert "return the failing unit" in p.lower()


# ── R8: trajectory summary ───────────────────────────────────────────────
def test_trajectory_summary_reports_tools_errors_and_timeout():
    res = {
        "actions": [
            {"tool": "shell"}, {"tool": "shell"}, {"tool": "write", "error": "boom"},
        ],
        "latency_ms": 1234,
        "usage": {"total": 4200},
        "timed_out": True,
    }
    s = fd_dispatch._trajectory_summary(res)
    assert "3 tool call(s)" in s
    assert "shell×2" in s
    assert "1 tool error(s)" in s
    assert "1234ms" in s
    assert "4200 tokens" in s
    assert "time budget" in s


def test_trajectory_summary_text_only_turn():
    s = fd_dispatch._trajectory_summary({"actions": [], "latency_ms": 50})
    assert "no tool calls" in s
    assert "50ms" in s


def test_trajectory_summary_is_defensive():
    assert fd_dispatch._trajectory_summary({}) == "no tool calls (text-only turn)"
    assert fd_dispatch._trajectory_summary(None) == ""
