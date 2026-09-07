"""R2 tests — the pure constraint_learning module (parse / render / signal)."""

from __future__ import annotations

from captain_claw.flight_deck import constraint_learning as cl


def test_parse_clean_json():
    out = '{"trigger": "when porting adapters", "constraint": "preserve keyword args", "severity": "major", "domain": "API"}'
    r = cl.parse_constraint(out)
    assert r == {"trigger": "when porting adapters",
                 "constraint": "preserve keyword args",
                 "severity": "major", "domain": "api"}


def test_parse_fenced_and_trailing_prose():
    fenced = 'Here you go:\n```json\n{"constraint": "always add a test", "severity": "critical"}\n```\nThanks!'
    r = cl.parse_constraint(fenced)
    assert r["constraint"] == "always add a test"
    assert r["severity"] == "critical"
    assert r["domain"] == "" and r["trigger"] == ""


def test_parse_clamps_and_sanitizes():
    r = cl.parse_constraint('{"constraint": "x", "severity": "whenever", "domain": "Web Apps!"}')
    assert r["severity"] == "major"           # unknown -> major
    assert r["domain"] == "webapps"           # lowercased, non [a-z0-9_-] stripped


def test_parse_rejects_garbage_and_empty():
    assert cl.parse_constraint("") is None
    assert cl.parse_constraint("no json here") is None
    assert cl.parse_constraint('{"constraint": ""}') is None
    assert cl.parse_constraint('[1,2,3]') is None
    assert cl.parse_constraint('{"constraint": "   "}') is None


def test_format_block_empty_and_bounded():
    assert cl.format_constraints_block(None) == ""
    assert cl.format_constraints_block([]) == ""
    rows = [{"constraint": f"rule {i}", "severity": "minor"} for i in range(10)]
    block = cl.format_constraints_block(rows, max_n=3)
    assert block.startswith("\n\n## Learned constraints")
    assert block.count("- [minor]") == 3       # honors max_n


def test_format_block_renders_both_key_shapes():
    db_row = {"constraint_text": "from db", "trigger_text": "on db path", "severity": "critical"}
    parsed = {"constraint": "from parse", "trigger": "on parse path", "severity": "major"}
    block = cl.format_constraints_block([db_row, parsed])
    assert "when on db path: from db" in block
    assert "when on parse path: from parse" in block
    assert "[critical]" in block and "[major]" in block


def test_build_signal_assembles_and_truncates():
    sig = cl.build_signal("A" * 10000, ["gap one", "  ", "gap two"], fixed_notes="had to fix imports")
    assert "had to fix imports" in sig
    assert "- gap: gap one" in sig and "- gap: gap two" in sig
    assert len(sig.split("\n\n")[0]) <= cl._SIGNAL_CAP  # deliverable truncated


def test_distill_prompt_shapes():
    p_a = cl.distill_prompt("do a thing", "signal", "accepted")
    p_f = cl.distill_prompt("do a thing", "signal", "fixed")
    assert '"constraint"' in p_a and '"severity"' in p_a
    assert "was accepted as good" in p_a
    assert "passed only after failures were fixed" in p_f
