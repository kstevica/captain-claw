"""Data models for the Swarm orchestration system."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── State constants ───────────────────────────────────────────

SWARM_STATES = frozenset({
    "draft",         # Created, task not yet rephrased/decomposed
    "decomposing",   # LLM is rephrasing/decomposing the task
    "ready",         # Decomposition done, waiting to start
    "running",       # Tasks actively executing
    "paused",        # Manually paused
    "completed",     # All tasks completed
    "failed",        # Swarm failed (unrecoverable)
    "cancelled",     # Manually cancelled
})

SWARM_TERMINAL_STATES = frozenset({"completed", "failed", "cancelled"})

TASK_STATES = frozenset({
    "queued",             # Waiting for dependencies
    "waiting",            # Dependencies met, waiting for concurrency slot
    "pending_approval",   # Ready but requires human approval before launch
    "running",            # Concern dispatched, agent working
    "completed",          # Agent returned result
    "failed",             # Failed after retries exhausted
    "retrying",           # Failed, waiting for retry
    "paused",             # Manually paused
    "skipped",            # Manually skipped
})

ERROR_POLICIES = frozenset({
    "fail_fast",         # Stop swarm on first task failure
    "continue_on_error", # Skip dependents of failed tasks, continue others
    "manual_review",     # Pause swarm on failure for human review
})

TASK_TERMINAL_STATES = frozenset({"completed", "failed", "skipped"})


# ── Models ────────────────────────────────────────────────────

@dataclass
class SwarmProject:
    """Top-level container for swarms."""

    id: str
    name: str
    description: str = ""
    created_at: str = field(default_factory=_utcnow_iso)
    updated_at: str = field(default_factory=_utcnow_iso)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict) -> SwarmProject:
        return cls(
            id=str(data.get("id", "")),
            name=str(data.get("name", "")),
            description=str(data.get("description", "")),
            created_at=str(data.get("created_at", "")),
            updated_at=str(data.get("updated_at", "")),
            metadata=json.loads(data["metadata"]) if isinstance(data.get("metadata"), str) else dict(data.get("metadata") or {}),
        )


@dataclass
class Swarm:
    """A single orchestration run within a project."""

    id: str
    project_id: str
    name: str = ""
    original_task: str = ""
    rephrased_task: str = ""
    status: str = "draft"
    priority: int = 0
    concurrency_limit: int = 5
    error_policy: str = "fail_fast"  # fail_fast | continue_on_error | manual_review
    created_at: str = field(default_factory=_utcnow_iso)
    updated_at: str = field(default_factory=_utcnow_iso)
    started_at: str = ""
    completed_at: str = ""
    template_id: str = ""
    metadata: dict = field(default_factory=dict)

    @property
    def is_terminal(self) -> bool:
        return self.status in SWARM_TERMINAL_STATES

    def touch(self) -> None:
        self.updated_at = _utcnow_iso()

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "project_id": self.project_id,
            "name": self.name,
            "original_task": self.original_task,
            "rephrased_task": self.rephrased_task,
            "status": self.status,
            "priority": self.priority,
            "concurrency_limit": self.concurrency_limit,
            "error_policy": self.error_policy,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "template_id": self.template_id,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict) -> Swarm:
        return cls(
            id=str(data.get("id", "")),
            project_id=str(data.get("project_id", "")),
            name=str(data.get("name", "")),
            original_task=str(data.get("original_task", "")),
            rephrased_task=str(data.get("rephrased_task", "")),
            status=str(data.get("status", "draft")),
            priority=int(data.get("priority", 0)),
            concurrency_limit=int(data.get("concurrency_limit", 5)),
            error_policy=str(data.get("error_policy", "fail_fast")),
            created_at=str(data.get("created_at", "")),
            updated_at=str(data.get("updated_at", "")),
            started_at=str(data.get("started_at", "")),
            completed_at=str(data.get("completed_at", "")),
            template_id=str(data.get("template_id", "")),
            metadata=json.loads(data["metadata"]) if isinstance(data.get("metadata"), str) else dict(data.get("metadata") or {}),
        )


@dataclass
class SwarmTask:
    """An individual subtask in the swarm DAG."""

    id: str
    swarm_id: str
    name: str = ""
    description: str = ""
    status: str = "queued"
    priority: int = 0
    # Agent assignment.
    assigned_instance: str = ""
    assigned_persona: str = ""
    concern_id: str = ""
    # DAG layout position.
    position_x: float = 0.0
    position_y: float = 0.0
    # Retry policy.
    retry_count: int = 0
    max_retries: int = 3
    retry_backoff_seconds: int = 30
    fallback_persona: str = ""
    timeout_seconds: int = 600
    timeout_warn_seconds: int = 0      # warn threshold (0 = disabled)
    timeout_extend_seconds: int = 0    # extra time after warn (0 = no extension)
    # Approval gate.
    requires_approval: bool = False
    approval_status: str = ""          # "" | approved | rejected
    approved_by: str = ""
    # Scheduling.
    is_periodic: bool = False
    cron_expression: str = ""
    next_run_at: str = ""
    # Timestamps.
    created_at: str = field(default_factory=_utcnow_iso)
    updated_at: str = field(default_factory=_utcnow_iso)
    started_at: str = ""
    completed_at: str = ""
    # Data.
    input_data: dict = field(default_factory=dict)
    output_data: dict = field(default_factory=dict)
    error_message: str = ""
    metadata: dict = field(default_factory=dict)

    @property
    def is_terminal(self) -> bool:
        return self.status in TASK_TERMINAL_STATES

    def touch(self) -> None:
        self.updated_at = _utcnow_iso()

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "swarm_id": self.swarm_id,
            "name": self.name,
            "description": self.description,
            "status": self.status,
            "priority": self.priority,
            "assigned_instance": self.assigned_instance,
            "assigned_persona": self.assigned_persona,
            "concern_id": self.concern_id,
            "position_x": self.position_x,
            "position_y": self.position_y,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "retry_backoff_seconds": self.retry_backoff_seconds,
            "fallback_persona": self.fallback_persona,
            "timeout_seconds": self.timeout_seconds,
            "timeout_warn_seconds": self.timeout_warn_seconds,
            "timeout_extend_seconds": self.timeout_extend_seconds,
            "requires_approval": self.requires_approval,
            "approval_status": self.approval_status,
            "approved_by": self.approved_by,
            "is_periodic": self.is_periodic,
            "cron_expression": self.cron_expression,
            "next_run_at": self.next_run_at,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "input_data": dict(self.input_data),
            "output_data": dict(self.output_data),
            "error_message": self.error_message,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict) -> SwarmTask:
        def _json_field(val: object) -> dict:
            if isinstance(val, str):
                return json.loads(val) if val else {}
            return dict(val) if val else {}

        return cls(
            id=str(data.get("id", "")),
            swarm_id=str(data.get("swarm_id", "")),
            name=str(data.get("name", "")),
            description=str(data.get("description", "")),
            status=str(data.get("status", "queued")),
            priority=int(data.get("priority", 0)),
            assigned_instance=str(data.get("assigned_instance", "")),
            assigned_persona=str(data.get("assigned_persona", "")),
            concern_id=str(data.get("concern_id", "")),
            position_x=float(data.get("position_x", 0)),
            position_y=float(data.get("position_y", 0)),
            retry_count=int(data.get("retry_count", 0)),
            max_retries=int(data.get("max_retries", 3)),
            retry_backoff_seconds=int(data.get("retry_backoff_seconds", 30)),
            fallback_persona=str(data.get("fallback_persona", "")),
            timeout_seconds=int(data.get("timeout_seconds", 600)),
            timeout_warn_seconds=int(data.get("timeout_warn_seconds", 0)),
            timeout_extend_seconds=int(data.get("timeout_extend_seconds", 0)),
            requires_approval=bool(data.get("requires_approval", False)),
            approval_status=str(data.get("approval_status", "")),
            approved_by=str(data.get("approved_by", "")),
            is_periodic=bool(data.get("is_periodic", False)),
            cron_expression=str(data.get("cron_expression", "")),
            next_run_at=str(data.get("next_run_at", "")),
            created_at=str(data.get("created_at", "")),
            updated_at=str(data.get("updated_at", "")),
            started_at=str(data.get("started_at", "")),
            completed_at=str(data.get("completed_at", "")),
            input_data=_json_field(data.get("input_data")),
            output_data=_json_field(data.get("output_data")),
            error_message=str(data.get("error_message", "")),
            metadata=_json_field(data.get("metadata")),
        )


@dataclass
class SwarmEdge:
    """A dependency edge between two tasks in the DAG."""

    id: int = 0
    swarm_id: str = ""
    from_task_id: str = ""
    to_task_id: str = ""
    edge_type: str = "dependency"  # dependency | data_flow

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "swarm_id": self.swarm_id,
            "from_task_id": self.from_task_id,
            "to_task_id": self.to_task_id,
            "edge_type": self.edge_type,
        }

    @classmethod
    def from_dict(cls, data: dict) -> SwarmEdge:
        return cls(
            id=int(data.get("id", 0)),
            swarm_id=str(data.get("swarm_id", "")),
            from_task_id=str(data.get("from_task_id", "")),
            to_task_id=str(data.get("to_task_id", "")),
            edge_type=str(data.get("edge_type", "dependency")),
        )


@dataclass
class SwarmArtifact:
    """An output artifact produced by a task."""

    id: str
    task_id: str
    swarm_id: str
    label: str = ""
    content_type: str = "text"  # text | json | file_ref
    content: str = ""
    created_at: str = field(default_factory=_utcnow_iso)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "task_id": self.task_id,
            "swarm_id": self.swarm_id,
            "label": self.label,
            "content_type": self.content_type,
            "content": self.content,
            "created_at": self.created_at,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict) -> SwarmArtifact:
        return cls(
            id=str(data.get("id", "")),
            task_id=str(data.get("task_id", "")),
            swarm_id=str(data.get("swarm_id", "")),
            label=str(data.get("label", "")),
            content_type=str(data.get("content_type", "text")),
            content=str(data.get("content", "")),
            created_at=str(data.get("created_at", "")),
            metadata=json.loads(data["metadata"]) if isinstance(data.get("metadata"), str) else dict(data.get("metadata") or {}),
        )


@dataclass
class SwarmAuditEntry:
    """An audit log entry for a swarm event."""

    id: int = 0
    swarm_id: str = ""
    task_id: str = ""
    event_type: str = ""
    details: dict = field(default_factory=dict)
    actor: str = "system"     # system | user | engine
    severity: str = "info"    # info | warn | error
    created_at: str = field(default_factory=_utcnow_iso)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "swarm_id": self.swarm_id,
            "task_id": self.task_id,
            "event_type": self.event_type,
            "details": dict(self.details),
            "actor": self.actor,
            "severity": self.severity,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> SwarmAuditEntry:
        return cls(
            id=int(data.get("id", 0)),
            swarm_id=str(data.get("swarm_id", "")),
            task_id=str(data.get("task_id", "")),
            event_type=str(data.get("event_type", "")),
            details=json.loads(data["details"]) if isinstance(data.get("details"), str) else dict(data.get("details") or {}),
            actor=str(data.get("actor", "system")),
            severity=str(data.get("severity", "info")),
            created_at=str(data.get("created_at", "")),
        )


@dataclass
class SwarmCheckpoint:
    """A snapshot of swarm state for resume/rollback."""

    id: str
    swarm_id: str
    label: str = ""
    swarm_state: dict = field(default_factory=dict)
    task_states: list = field(default_factory=list)
    edge_states: list = field(default_factory=list)
    created_at: str = field(default_factory=_utcnow_iso)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "swarm_id": self.swarm_id,
            "label": self.label,
            "swarm_state": dict(self.swarm_state),
            "task_states": list(self.task_states),
            "edge_states": list(self.edge_states),
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> SwarmCheckpoint:
        def _json_field(val: object, default: object = None) -> object:
            if default is None:
                default = {}
            if isinstance(val, str):
                return json.loads(val) if val else default
            return val if val is not None else default

        return cls(
            id=str(data.get("id", "")),
            swarm_id=str(data.get("swarm_id", "")),
            label=str(data.get("label", "")),
            swarm_state=_json_field(data.get("swarm_state"), {}),
            task_states=_json_field(data.get("task_states"), []),
            edge_states=_json_field(data.get("edge_states"), []),
            created_at=str(data.get("created_at", "")),
        )


@dataclass
class SwarmTemplate:
    """A reusable DAG template."""

    id: str
    name: str
    description: str = ""
    dag_definition: dict = field(default_factory=dict)
    created_at: str = field(default_factory=_utcnow_iso)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "dag_definition": dict(self.dag_definition),
            "created_at": self.created_at,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict) -> SwarmTemplate:
        return cls(
            id=str(data.get("id", "")),
            name=str(data.get("name", "")),
            description=str(data.get("description", "")),
            dag_definition=json.loads(data["dag_definition"]) if isinstance(data.get("dag_definition"), str) else dict(data.get("dag_definition") or {}),
            created_at=str(data.get("created_at", "")),
            metadata=json.loads(data["metadata"]) if isinstance(data.get("metadata"), str) else dict(data.get("metadata") or {}),
        )
