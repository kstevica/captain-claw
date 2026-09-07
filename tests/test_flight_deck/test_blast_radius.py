"""R5 tests — the deterministic blast-radius classifier + opt-in flag."""

from __future__ import annotations

from captain_claw.config import AUTONOMY_HARD_EXCLUDE
from captain_claw.flight_deck.blast_radius import (
    _MONEY_OUTWARD,
    classify_plan,
    classify_tool,
)
from captain_claw.flight_deck.quality_profile import QualityProfile


def test_classify_tool_flags_high_blast_commands():
    hits = [
        ("shell", {"command": "git push --force origin main"}),
        ("shell", {"command": "rm -rf build"}),
        ("shell", {"command": "psql -c 'DROP TABLE users'"}),
        ("shell", {"command": "alembic upgrade head"}),
        ("shell", {"command": "rails db:migrate"}),
        ("shell", {"command": "python manage.py migrate"}),
        ("shell", {"command": "kubectl delete pod web-0"}),
        ("shell", {"command": "terraform destroy -auto-approve"}),
    ]
    for name, args in hits:
        hit, reason = classify_tool(name, args)
        assert hit, (name, args)
        assert reason


def test_classify_tool_flags_money_tool_name_and_prose():
    hit, reason = classify_tool("stripe_charge", {})
    assert hit and "stripe" in reason
    hit2, _ = classify_tool("shell", {"command": "initiate a wire transfer of $5000"})
    assert hit2


def test_classify_tool_benign_is_clean():
    for name, args in [
        ("shell", {"command": "ls -la"}),
        ("shell", {"command": "git status"}),
        ("shell", {"command": "cat README.md"}),
        ("shell", {"command": "python test.py"}),
        ("read", {"path": "x"}),
    ]:
        assert classify_tool(name, args) == (False, "")


def test_classify_tool_scans_only_command_fields():
    # A file write whose CONTENT mentions DROP TABLE is not itself a migration.
    assert classify_tool("write", {"content": "DROP TABLE x", "path": "y"}) == (False, "")


def test_classify_plan_prose():
    hit, _ = classify_plan("Then we run the database migration and delete from the old table")
    assert hit
    assert classify_plan("Add a helper function and a unit test") == (False, "")


def test_money_vocab_is_subset_of_autonomy_hard_exclude():
    # Drift guard: the two safety walls share the money/outward vocabulary.
    assert set(_MONEY_OUTWARD) <= set(AUTONOMY_HARD_EXCLUDE)


class _GuardStub:
    """Minimal host for AgentGuardMixin._enforce_blast_radius — exercises the
    guard wiring without get_config()/network (test_guarding needs both)."""

    def __init__(self, enabled, level="stop_suspicious", approve=False):
        self._enabled, self._level, self._approve = enabled, level, approve
        self.emitted: list = []

    def _guard_settings(self, _t):
        return (self._enabled, self._level)

    def _request_guard_approval(self, _q):
        return self._approve

    def _emit_tool_output(self, kind, args, out):
        self.emitted.append((kind, args, out))


def test_enforce_blast_radius_wiring():
    from captain_claw.agent_guard_mixin import AgentGuardMixin
    f = AgentGuardMixin._enforce_blast_radius
    # disabled -> no-op even for a dangerous command (classifier not consulted)
    assert f(_GuardStub(False), "shell", {"command": "rm -rf /"}, "x") == (True, "")
    # enabled + stop_suspicious + hit -> blocked
    ok, err = f(_GuardStub(True, "stop_suspicious"), "shell",
                {"command": "git push --force origin main"}, "x")
    assert ok is False and "blast_radius" in err
    # enabled + ask_for_approval + approved -> allowed
    ok2, _ = f(_GuardStub(True, "ask_for_approval", approve=True), "shell",
               {"command": "rm -rf build"}, "x")
    assert ok2 is True
    # enabled + ask_for_approval + denied -> blocked
    ok3, _ = f(_GuardStub(True, "ask_for_approval", approve=False), "shell",
               {"command": "rm -rf build"}, "x")
    assert ok3 is False
    # enabled + benign -> allowed
    assert f(_GuardStub(True), "shell", {"command": "ls -la"}, "x") == (True, "")


def test_blast_radius_gate_is_explicit_opt_in_and_out_of_bool_flags():
    assert QualityProfile.from_dict(None).blast_radius_gate is False
    for preset in ("off", "balanced", "thorough"):
        assert QualityProfile.from_dict({"profile": preset}).blast_radius_gate is False, preset
    q = QualityProfile.from_dict({"blast_radius_gate": True})
    assert q.blast_radius_gate is True
    # Acts before the build loop, so it must NOT flip any_enabled (in-build work).
    assert "blast_radius_gate" not in QualityProfile._BOOL_FLAGS
    assert q.any_enabled is False
