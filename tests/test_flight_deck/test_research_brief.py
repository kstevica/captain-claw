"""Tests for R12 — the pure part of the intent brief (rephrase) step.

The load-bearing safety property: an empty/absent brief makes ``brief_task``
return the raw intent byte-for-byte, so the lever being off == today's routing
and dispatch. And when a brief IS present, the original intent is always kept and
declared authoritative, so a rephrase can never silently become the source of
truth.
"""

from __future__ import annotations

from captain_claw.flight_deck import research_brief as rb


# ── the safety property ──────────────────────────────────────────────

def test_no_brief_is_exactly_the_raw_intent():
    for empty in ("", None, "   "):
        assert rb.brief_task("do a ROPA for Acme", empty) == "do a ROPA for Acme"


def test_brief_keeps_the_original_and_makes_it_authoritative():
    t = rb.brief_task("do a ROPA for Acme", "**Objective**\nMap Acme's processing.")
    assert "do a ROPA for Acme" in t          # original preserved verbatim
    assert "Map Acme's processing." in t      # brief included
    assert "ORIGINAL" in t                    # original declared to govern on conflict
    # The original comes first, the brief follows as a labelled clarification.
    assert t.index("do a ROPA for Acme") < t.index("Map Acme's processing.")


# ── parsing ──────────────────────────────────────────────────────────

def test_parse_strips_code_fences():
    out = "```markdown\n**Objective**\nX\n```"
    b = rb.parse_brief(out)
    assert b == "**Objective**\nX"


def test_parse_trims_and_caps_length():
    assert rb.parse_brief("   hi   ") == "hi"
    big = "x" * 9000
    assert len(rb.parse_brief(big)) == rb._MAX_BRIEF_CHARS


def test_parse_empty_is_empty():
    assert rb.parse_brief("") == ""
    assert rb.parse_brief(None) == ""  # type: ignore[arg-type]


# ── the derive prompt is faithful-by-construction ────────────────────

def test_derive_prompt_forbids_scope_change_and_carries_the_task():
    p = rb.derive_brief_prompt("do a ROPA for Acme")
    assert "do a ROPA for Acme" in p
    # It must instruct faithfulness, not reinterpretation.
    assert "no new goals" in p.lower() or "add no new" in p.lower()
    assert "scope" in p.lower()
    # It must not answer the task.
    assert "do not answer" in p.lower() or "not a reinterpretation" in p.lower()


# ── the file-aware derive prompt (run-time file examiner) ────────────

def test_file_aware_prompt_extends_the_base_and_names_the_files():
    p = rb.derive_brief_with_files_prompt("analyse the numbers", ["q3.csv", "notes.pdf"])
    # It is a superset of the plain brief prompt (same faithfulness rules).
    assert "analyse the numbers" in p
    assert "no new goals" in p.lower() or "add no new" in p.lower()
    # It names the attached files and tells the agent to open + fold them in.
    assert "q3.csv" in p and "notes.pdf" in p
    assert "open" in p.lower() and "attached" in p.lower()
    # Domain-agnostic + anti-fabrication: never invent file contents.
    assert "invent" in p.lower() or "only what is there" in p.lower()


def test_file_aware_prompt_handles_no_names():
    p = rb.derive_brief_with_files_prompt("do X", [])
    assert "attached file" in p.lower()          # generic fallback phrasing
    p2 = rb.derive_brief_with_files_prompt("do X", None)  # type: ignore[arg-type]
    assert "attached file" in p2.lower()
