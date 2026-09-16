"""Story Integrity P1/P2 — the model judgment passes + validator integration."""

import json

import pytest

from captain_claw.flight_deck import story_passes as sp
from captain_claw.flight_deck import story_state as ss


def test_pass_registry_and_phases():
    assert set(sp.ALL_PASSES) == set(sp.PASSES)
    assert sp.P1_PASSES == ("C", "F", "I", "J")
    assert sp.P2_PASSES == ("CA", "PR")
    # I (economy) is soft-ceilinged; C/F/J/CA/PR can go hard
    assert sp.PASSES["I"][1] == "soft"
    assert sp.PASSES["C"][1] == "hard"


def test_severity_clamp():
    assert sp._clamp("hard", "soft") == "soft"     # economy can never block
    assert sp._clamp("hard", "hard") == "hard"
    assert sp._clamp("major", "hard") == "major"
    assert sp._clamp("bogus", "hard") == "major"   # default major


def test_prompts_carry_their_discipline():
    draft, store = "x", "{}"
    assert "two places at once" in sp._p_physical(draft, store)
    assert "not found" in sp._p_research(draft, store).lower()
    assert "soft" in sp._p_economy(draft, store)
    assert "seeded" in sp._p_fairness(draft, store).lower()
    assert "jurisdiction" in sp._p_claim_attacker(draft, store).lower()
    assert "chain of custody" in sp._p_procedure(draft, store).lower() or \
           "custody" in sp._p_procedure(draft, store).lower()


@pytest.mark.asyncio
async def test_run_model_pass_parses_and_clamps():
    async def model_fn(prompt):
        return ('{"findings": [{"kind": "impossible_time", "severity": "hard", '
                '"scene": "Ch 4", "reason": "24 min is too short", "fix": "use a range", '
                '"quote": "in twenty-four minutes"}]}')
    out = await sp.run_model_pass("C", "draft", "{}", model_fn)
    assert len(out) == 1
    f = out[0]
    assert f["pass"] == "C" and f["severity"] == "hard" and f["scene"] == "Ch 4"
    assert f["quotes"] == ["in twenty-four minutes"] and f["source"] == "story_integrity"


@pytest.mark.asyncio
async def test_economy_pass_clamped_to_soft():
    async def model_fn(prompt):
        # even if the model over-rates it 'hard', economy is clamped to soft
        return '{"findings": [{"kind": "repetition", "severity": "hard", "reason": "dup scene"}]}'
    out = await sp.run_model_pass("I", "draft", "{}", model_fn)
    assert out and out[0]["severity"] == "soft"


@pytest.mark.asyncio
async def test_run_model_pass_tolerant_of_junk():
    async def model_fn(prompt):
        return "sorry, I could not analyze this"
    # junk that parses to no findings = the pass RAN and found nothing → []
    assert await sp.run_model_pass("F", "draft", "{}", model_fn) == []

    async def boom(prompt):
        raise RuntimeError("model down")
    # a model crash = the pass did NOT run → None (distinct from ran-clean [])
    assert await sp.run_model_pass("F", "draft", "{}", boom) is None


@pytest.mark.asyncio
async def test_run_model_passes_merges_all(monkeypatch):
    calls = []

    async def model_fn(prompt):
        # tag the finding with which pass by sniffing the prompt's discipline
        if "PHYSICAL-MECHANISM" in prompt:
            return '{"findings":[{"kind":"x","severity":"hard","reason":"phys"}]}'
        if "NARRATIVE-ECONOMY" in prompt:
            return '{"findings":[{"kind":"y","severity":"hard","reason":"econ"}]}'
        return '{"findings":[]}'
    out, ran = await sp.run_model_passes("draft", "{}", model_fn, passes=("C", "I"))
    kinds = {(f["pass"], f["severity"]) for f in out}
    assert ("C", "hard") in kinds
    assert ("I", "soft") in kinds  # clamped
    assert ran == {"C", "I"}       # both passes actually executed


@pytest.mark.asyncio
async def test_run_model_passes_empty_when_no_draft():
    async def model_fn(prompt):
        return '{"findings":[{"kind":"x","severity":"hard"}]}'
    out, ran = await sp.run_model_passes("", "{}", model_fn, passes=("C",))
    assert out == [] and ran == set()


@pytest.mark.asyncio
async def test_validator_runs_model_passes_and_gates(monkeypatch):
    # deterministic extract yields a clean store; a model pass returns a hard finding
    async def extract_fn(prompt):
        return '{"characters": [], "timeline": [], "story_year": ""}'

    async def model_fn(prompt):
        if "PHYSICAL-MECHANISM" in prompt:
            return ('{"findings":[{"kind":"impossible_mechanism","severity":"hard",'
                    '"scene":"Ch 4","reason":"the seal cannot open while fitted"}]}')
        return '{"findings":[]}'

    res = await ss.run_validator("# Ch4\n\nbody", extract_fn=extract_fn, revise_fn=None,
                                 model_fn=model_fn, model_passes=("C", "F", "I", "J", "CA", "PR"))
    assert any(f["pass"] == "C" and f["severity"] == "hard" for f in res["hard"])
    b = ss.blocking_analysis(res)
    assert b["hard"] and b["hard"][0]["pass"] == "C"


@pytest.mark.asyncio
async def test_validator_model_passes_off_when_no_model_fn():
    async def extract_fn(prompt):
        return '{"characters": [], "timeline": []}'
    res = await ss.run_validator("draft", extract_fn=extract_fn, revise_fn=None)
    assert res["findings"] == []  # deterministic clean, no model passes run


def test_patch_prompt_simplify_escalation():
    findings = [{"pass": "C", "kind": "x", "severity": "hard", "reason": "impossible",
                 "fix": "use a range", "quotes": ["in 24 min"]}]
    normal = ss.patch_prompt("text", findings, simplify=False)
    esc = ss.patch_prompt("text", findings, simplify=True)
    assert "SIMPLIFY the plot as needed" in esc
    assert "SIMPLIFY the plot as needed" not in normal
    assert "suggested fix: use a range" in normal  # the model's fix is surfaced


# ── Fixes from the P1/P2 adversarial verification ─────────────────────

def test_severity_synonyms_map_to_hard():
    assert sp._clamp("critical", "hard") == "hard"
    assert sp._clamp("blocker", "hard") == "hard"
    assert sp._clamp("critical", "soft") == "soft"   # economy still clamps


def test_parse_accepts_bare_array():
    out = sp._parse('[{"kind": "unseeded", "severity": "hard"}]')
    assert len(out) == 1 and out[0]["kind"] == "unseeded"
    # object form still works
    out2 = sp._parse('{"findings": [{"kind": "x", "severity": "major"}]}')
    assert len(out2) == 1


def test_parse_coerces_non_string():
    assert sp._parse(None) == []
    assert sp._parse(12345) == []
    assert sp._parse({"findings": [{"kind": "x"}]}) == []  # a dict has no JSON braces text


@pytest.mark.asyncio
async def test_pass_J_downgraded_to_major_on_truncation():
    long_draft = "# Ch1\n\n" + ("word " * 20000)  # > DRAFT_CAP → windowed/truncated

    async def model_fn(prompt):
        return '{"findings":[{"kind":"unseeded_twist","severity":"hard","reason":"no seeds"}]}'
    out = await sp.run_model_pass("J", long_draft, "{}", model_fn)
    # truncated → J cannot judge seeding → cannot HARD-block
    assert out and out[0]["severity"] == "major"
    # a SHORT draft keeps J's hard ceiling
    out2 = await sp.run_model_pass("J", "short", "{}", model_fn)
    assert out2 and out2[0]["severity"] == "hard"


@pytest.mark.asyncio
async def test_validator_rejects_major_to_deterministic_hard_trade():
    # initial: 0 hard + 2 major (deterministic custody). The revision clears them but the
    # revised store yields a DETERMINISTIC hard (age mismatch) → must be REJECTED, because
    # a deterministic hard is exactly what gates `done`.
    async def extract_fn(prompt):
        if "REVISED" in prompt:
            # revised store: an age mismatch (deterministic G hard), no majors
            return '{"characters": [{"name": "Ana", "age": "34"}, {"name": "Ana", "age": "41"}]}'
        return ('{"evidence": [{"id": "a", "custody_gap": true, "key_proof": true},'
                '{"id": "b", "custody_gap": true, "key_proof": true}]}')

    async def revise_fn(prompt):
        return '[{"find": "orig", "replace": "REVISED and much longer body ' + "x" * 50 + '"}]'

    text = "orig " * 300
    res = await ss.run_validator(text, extract_fn=extract_fn, revise_fn=revise_fn, max_rounds=2)
    # the loop must NOT have kept a revision that introduced a deterministic (gating) hard
    assert res["blocking"] == [], "a major→deterministic-hard trade must be rejected"
    assert res["revised"] is False


@pytest.mark.asyncio
async def test_validator_accepts_major_to_model_hard_trade():
    # A revision that clears 2 deterministic majors but provokes a MODEL-pass hard is
    # ACCEPTED: the model hard is advisory (never gates), the deterministic state improved,
    # and the total finding count dropped. The gating set stays empty either way.
    async def extract_fn(prompt):
        if "REVISED" in prompt:
            return '{"evidence": []}'   # revised store: clean of deterministic findings
        return ('{"evidence": [{"id": "a", "custody_gap": true, "key_proof": true},'
                '{"id": "b", "custody_gap": true, "key_proof": true}]}')

    async def revise_fn(prompt):
        return '[{"find": "orig", "replace": "REVISED and much longer body ' + "x" * 50 + '"}]'

    async def model_fn(prompt):
        if "REVISED" in prompt and "PHYSICAL-MECHANISM" in prompt:
            return '{"findings":[{"kind":"impossible","severity":"hard","reason":"new"}]}'
        return '{"findings":[]}'

    text = "orig " * 300
    res = await ss.run_validator(text, extract_fn=extract_fn, revise_fn=revise_fn,
                                 model_fn=model_fn, model_passes=("C",), max_rounds=2)
    # the model hard is present but does NOT gate; the run remains done-eligible
    assert res["blocking"] == [], "a model-pass hard must never gate"
