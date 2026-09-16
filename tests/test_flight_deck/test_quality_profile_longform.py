"""Increment 3: the long_form preset and the new deliverable flags."""

from captain_claw.flight_deck.quality_profile import QualityProfile, _PRESETS


def test_long_form_preset_membership():
    lf = QualityProfile.from_dict({"profile": "long_form"})
    # free/structural levers ON
    for f in ("write_guard", "derive_manifest", "resolve_pointer_truth",
              "require_deliverable_file", "require_inputs", "wait_heartbeat",
              "strict_deps", "clarify_dep_grant", "push_deps"):
        assert getattr(lf, f) is True, f
    assert lf.synthesis_emit == "concat_then_smooth"
    # balanced levers carried
    assert lf.acted_gate and lf.critic_triage
    # judgment_ledger deliberately dropped from long_form
    assert lf.judgment_ledger is False


def test_paid_levers_off_in_every_preset():
    for name in _PRESETS:
        p = QualityProfile.from_dict({"profile": name})
        assert p.canon_pass is False, name
        assert p.gate_blocks_done is False, name
        assert p.block_on_critical is False, name
        assert p.claim_check is False, name


def test_new_flags_not_in_bool_flags_any_enabled():
    # A manifest/wait flag alone must not trip the quality machinery.
    p = QualityProfile.from_dict({"profile": "off", "derive_manifest": True,
                                  "require_inputs": True, "wait_heartbeat": True})
    assert p.any_enabled is False
    for f in ("write_guard", "derive_manifest", "resolve_pointer_truth",
              "require_deliverable_file", "require_inputs", "wait_heartbeat",
              "strict_deps", "clarify_dep_grant", "canon_pass", "gate_blocks_done"):
        assert f not in QualityProfile._BOOL_FLAGS, f


def test_off_is_byte_identical_defaults():
    off = QualityProfile.from_dict(None)
    assert off.any_enabled is False
    assert off.write_guard is False
    assert off.synthesis_emit == ""
    assert off.deliverable_kind == ""
    assert off.require_deliverable_file is False


def test_explicit_overrides_and_int_fields():
    p = QualityProfile.from_dict({
        "profile": "long_form", "synthesis_emit": "rewrite",
        "deliverable_kind": "fiction", "canon_pass": True, "gate_blocks_done": True,
        "deliverable_min_chars": 60000, "deliverable_min_sections": 12,
        "wait_max_total_s": 900, "clarify_cap": 3, "qa_tier": "reason",
    })
    assert p.synthesis_emit == "rewrite"       # explicit overrides preset value
    assert p.deliverable_kind == "fiction"
    assert p.canon_pass and p.gate_blocks_done
    assert p.deliverable_min_chars == 60000 and p.deliverable_min_sections == 12
    assert p.wait_max_total_s == 900 and p.clarify_cap == 3
    assert p.qa_tier == "reason"


def test_unknown_synthesis_emit_ignored():
    p = QualityProfile.from_dict({"synthesis_emit": "bogus"})
    assert p.synthesis_emit == ""
