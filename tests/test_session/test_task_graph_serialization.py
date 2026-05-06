"""Round-trip serialization tests for OrchestratorTask, including plan-mode fields."""

from __future__ import annotations

import json

from captain_claw.task_graph import OrchestratorTask


def test_to_dict_omits_default_plan_fields():
    task = OrchestratorTask(id="t1", title="Task one", description="desc")
    out = task.to_dict()

    assert out["id"] == "t1"
    assert out["title"] == "Task one"
    assert out["description"] == "desc"
    # Plan-mode fields with default values must NOT bloat the persisted JSON.
    assert "step_kind" not in out
    assert "acceptance_criteria" not in out
    assert "revision_of" not in out


def test_to_dict_includes_plan_fields_when_set():
    task = OrchestratorTask(
        id="t2",
        title="Verify auth flow",
        description="Run the verifier prompt",
        step_kind="verify",
        acceptance_criteria="Login round-trips a valid JWT.",
        revision_of="t1",
    )
    out = task.to_dict()

    assert out["step_kind"] == "verify"
    assert out["acceptance_criteria"] == "Login round-trips a valid JWT."
    assert out["revision_of"] == "t1"


def test_round_trip_preserves_plan_fields():
    original = OrchestratorTask(
        id="step-3",
        title="Fan out research",
        description="Dispatch sub-tasks via /orchestrate",
        depends_on=["step-1", "step-2"],
        model_id="claude-opus-4-7",
        skills=["web-search"],
        workspace_inputs=["research-brief"],
        workspace_outputs=["research-summary"],
        output_schema={"type": "object", "properties": {"summary": {"type": "string"}}},
        output_schema_name="ResearchSummary",
        step_kind="orchestrate",
        acceptance_criteria="At least three sources cited with URLs.",
        revision_of="",
    )

    serialized = json.dumps(original.to_dict())
    restored = OrchestratorTask.from_dict(json.loads(serialized))

    assert restored.id == original.id
    assert restored.title == original.title
    assert restored.description == original.description
    assert restored.depends_on == original.depends_on
    assert restored.model_id == original.model_id
    assert restored.skills == original.skills
    assert restored.workspace_inputs == original.workspace_inputs
    assert restored.workspace_outputs == original.workspace_outputs
    assert restored.output_schema == original.output_schema
    assert restored.output_schema_name == original.output_schema_name
    assert restored.step_kind == original.step_kind
    assert restored.acceptance_criteria == original.acceptance_criteria
    assert restored.revision_of == original.revision_of


def test_from_dict_defaults_plan_fields_when_missing():
    """Legacy workflow files (saved before plan-mode) must still load cleanly."""
    legacy = {
        "id": "legacy",
        "title": "Old task",
        "description": "Saved before plan-mode existed",
        "depends_on": [],
    }
    task = OrchestratorTask.from_dict(legacy)

    assert task.step_kind == "atomic"
    assert task.acceptance_criteria == ""
    assert task.verification_status == "unverified"
    assert task.verification_notes == ""
    assert task.revision_of == ""
    assert task.revision_count == 0


def test_from_dict_applies_caller_timeout_and_retries():
    task = OrchestratorTask.from_dict(
        {"id": "t", "title": "x"},
        timeout_seconds=900.0,
        max_retries=5,
    )
    assert task.timeout_seconds == 900.0
    assert task.max_retries == 5
