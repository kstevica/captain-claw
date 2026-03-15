"""SQLite persistence for Swarm orchestration data."""

from __future__ import annotations

import json
from typing import Any

import aiosqlite

from botport.swarm.models import (
    Swarm,
    SwarmArtifact,
    SwarmAuditEntry,
    SwarmCheckpoint,
    SwarmEdge,
    SwarmProject,
    SwarmTask,
    SwarmTemplate,
)


class SwarmStore:
    """Async SQLite store for swarm projects, swarms, tasks, and artifacts.

    Shares the same database connection as BotPortStore (passed in).
    """

    def __init__(self, db: aiosqlite.Connection) -> None:
        self._db = db

    async def create_tables(self) -> None:
        """Create all swarm-related tables."""
        await self._db.executescript("""
            CREATE TABLE IF NOT EXISTS swarm_projects (
                id              TEXT PRIMARY KEY,
                name            TEXT NOT NULL,
                description     TEXT DEFAULT '',
                created_at      TEXT NOT NULL,
                updated_at      TEXT NOT NULL,
                metadata        TEXT DEFAULT '{}'
            );

            CREATE TABLE IF NOT EXISTS swarms (
                id              TEXT PRIMARY KEY,
                project_id      TEXT NOT NULL,
                name            TEXT DEFAULT '',
                original_task   TEXT DEFAULT '',
                rephrased_task  TEXT DEFAULT '',
                status          TEXT DEFAULT 'draft',
                priority        INTEGER DEFAULT 0,
                concurrency_limit INTEGER DEFAULT 5,
                created_at      TEXT NOT NULL,
                updated_at      TEXT NOT NULL,
                started_at      TEXT DEFAULT '',
                completed_at    TEXT DEFAULT '',
                template_id     TEXT DEFAULT '',
                metadata        TEXT DEFAULT '{}',
                FOREIGN KEY (project_id) REFERENCES swarm_projects(id)
            );

            CREATE TABLE IF NOT EXISTS swarm_tasks (
                id              TEXT PRIMARY KEY,
                swarm_id        TEXT NOT NULL,
                name            TEXT DEFAULT '',
                description     TEXT NOT NULL,
                status          TEXT DEFAULT 'queued',
                priority        INTEGER DEFAULT 0,
                assigned_instance TEXT DEFAULT '',
                assigned_persona  TEXT DEFAULT '',
                concern_id      TEXT DEFAULT '',
                position_x      REAL DEFAULT 0,
                position_y      REAL DEFAULT 0,
                retry_count     INTEGER DEFAULT 0,
                max_retries     INTEGER DEFAULT 3,
                retry_backoff_seconds INTEGER DEFAULT 30,
                fallback_persona TEXT DEFAULT '',
                timeout_seconds INTEGER DEFAULT 600,
                is_periodic     INTEGER DEFAULT 0,
                cron_expression TEXT DEFAULT '',
                next_run_at     TEXT DEFAULT '',
                created_at      TEXT NOT NULL,
                updated_at      TEXT NOT NULL,
                started_at      TEXT DEFAULT '',
                completed_at    TEXT DEFAULT '',
                input_data      TEXT DEFAULT '{}',
                output_data     TEXT DEFAULT '{}',
                error_message   TEXT DEFAULT '',
                metadata        TEXT DEFAULT '{}',
                FOREIGN KEY (swarm_id) REFERENCES swarms(id)
            );

            CREATE TABLE IF NOT EXISTS swarm_task_edges (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                swarm_id        TEXT NOT NULL,
                from_task_id    TEXT NOT NULL,
                to_task_id      TEXT NOT NULL,
                edge_type       TEXT DEFAULT 'dependency',
                FOREIGN KEY (swarm_id) REFERENCES swarms(id),
                FOREIGN KEY (from_task_id) REFERENCES swarm_tasks(id),
                FOREIGN KEY (to_task_id) REFERENCES swarm_tasks(id),
                UNIQUE(from_task_id, to_task_id)
            );

            CREATE TABLE IF NOT EXISTS swarm_artifacts (
                id              TEXT PRIMARY KEY,
                task_id         TEXT NOT NULL,
                swarm_id        TEXT NOT NULL,
                label           TEXT DEFAULT '',
                content_type    TEXT DEFAULT 'text',
                content         TEXT DEFAULT '',
                created_at      TEXT NOT NULL,
                metadata        TEXT DEFAULT '{}',
                FOREIGN KEY (task_id) REFERENCES swarm_tasks(id),
                FOREIGN KEY (swarm_id) REFERENCES swarms(id)
            );

            CREATE TABLE IF NOT EXISTS swarm_audit_log (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                swarm_id        TEXT NOT NULL,
                task_id         TEXT DEFAULT '',
                event_type      TEXT NOT NULL,
                details         TEXT DEFAULT '{}',
                created_at      TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS swarm_cost_log (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                swarm_id        TEXT NOT NULL,
                task_id         TEXT DEFAULT '',
                instance_name   TEXT DEFAULT '',
                persona_name    TEXT DEFAULT '',
                tokens_in       INTEGER DEFAULT 0,
                tokens_out      INTEGER DEFAULT 0,
                cost_usd        REAL DEFAULT 0,
                created_at      TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS swarm_templates (
                id              TEXT PRIMARY KEY,
                name            TEXT NOT NULL,
                description     TEXT DEFAULT '',
                dag_definition  TEXT DEFAULT '{}',
                created_at      TEXT NOT NULL,
                metadata        TEXT DEFAULT '{}'
            );

            CREATE TABLE IF NOT EXISTS swarm_task_runs (
                id              TEXT PRIMARY KEY,
                task_id         TEXT NOT NULL,
                swarm_id        TEXT NOT NULL,
                run_number      INTEGER DEFAULT 1,
                status          TEXT DEFAULT 'running',
                concern_id      TEXT DEFAULT '',
                started_at      TEXT NOT NULL,
                completed_at    TEXT DEFAULT '',
                output_data     TEXT DEFAULT '{}',
                error_message   TEXT DEFAULT '',
                FOREIGN KEY (task_id) REFERENCES swarm_tasks(id)
            );

            CREATE TABLE IF NOT EXISTS swarm_checkpoints (
                id              TEXT PRIMARY KEY,
                swarm_id        TEXT NOT NULL,
                label           TEXT DEFAULT '',
                swarm_state     TEXT DEFAULT '{}',
                task_states     TEXT DEFAULT '[]',
                edge_states     TEXT DEFAULT '[]',
                created_at      TEXT NOT NULL,
                FOREIGN KEY (swarm_id) REFERENCES swarms(id)
            );

            CREATE INDEX IF NOT EXISTS idx_swarms_project ON swarms(project_id);
            CREATE INDEX IF NOT EXISTS idx_swarms_status ON swarms(status);
            CREATE INDEX IF NOT EXISTS idx_swarm_tasks_swarm ON swarm_tasks(swarm_id);
            CREATE INDEX IF NOT EXISTS idx_swarm_tasks_status ON swarm_tasks(status);
            CREATE INDEX IF NOT EXISTS idx_swarm_edges_swarm ON swarm_task_edges(swarm_id);
            CREATE INDEX IF NOT EXISTS idx_swarm_artifacts_task ON swarm_artifacts(task_id);
            CREATE INDEX IF NOT EXISTS idx_swarm_audit_swarm ON swarm_audit_log(swarm_id);
            CREATE INDEX IF NOT EXISTS idx_swarm_cost_swarm ON swarm_cost_log(swarm_id);
            CREATE INDEX IF NOT EXISTS idx_swarm_checkpoints_swarm ON swarm_checkpoints(swarm_id);
        """)
        await self._db.commit()

        # Schema migration: add new columns to existing tables.
        migrations = [
            ("swarms", "error_policy", "TEXT DEFAULT 'fail_fast'"),
            ("swarm_tasks", "timeout_warn_seconds", "INTEGER DEFAULT 0"),
            ("swarm_tasks", "timeout_extend_seconds", "INTEGER DEFAULT 0"),
            ("swarm_tasks", "requires_approval", "INTEGER DEFAULT 0"),
            ("swarm_tasks", "approval_status", "TEXT DEFAULT ''"),
            ("swarm_tasks", "approved_by", "TEXT DEFAULT ''"),
            ("swarm_audit_log", "actor", "TEXT DEFAULT 'system'"),
            ("swarm_audit_log", "severity", "TEXT DEFAULT 'info'"),
        ]
        for table, column, col_type in migrations:
            try:
                await self._db.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")
            except Exception:
                pass  # Column already exists.
        await self._db.commit()

    # ── Projects ──────────────────────────────────────────────

    async def save_project(self, project: SwarmProject) -> None:
        await self._db.execute(
            """INSERT INTO swarm_projects (id, name, description, created_at, updated_at, metadata)
               VALUES (?, ?, ?, ?, ?, ?)
               ON CONFLICT(id) DO UPDATE SET
                 name = excluded.name,
                 description = excluded.description,
                 updated_at = excluded.updated_at,
                 metadata = excluded.metadata
            """,
            (project.id, project.name, project.description,
             project.created_at, project.updated_at, json.dumps(project.metadata)),
        )
        await self._db.commit()

    async def get_project(self, project_id: str) -> SwarmProject | None:
        async with self._db.execute(
            "SELECT * FROM swarm_projects WHERE id = ?", (project_id,)
        ) as cursor:
            row = await cursor.fetchone()
        return SwarmProject.from_dict(dict(row)) if row else None

    async def list_projects(self) -> list[SwarmProject]:
        projects: list[SwarmProject] = []
        async with self._db.execute(
            "SELECT * FROM swarm_projects ORDER BY updated_at DESC"
        ) as cursor:
            async for row in cursor:
                projects.append(SwarmProject.from_dict(dict(row)))
        return projects

    async def delete_project(self, project_id: str) -> bool:
        # Check for running swarms.
        async with self._db.execute(
            "SELECT COUNT(*) as cnt FROM swarms WHERE project_id = ? AND status IN ('running', 'decomposing')",
            (project_id,),
        ) as cursor:
            row = await cursor.fetchone()
            if row and row["cnt"] > 0:
                return False

        await self._db.execute("DELETE FROM swarm_projects WHERE id = ?", (project_id,))
        await self._db.commit()
        return True

    # ── Swarms ────────────────────────────────────────────────

    async def save_swarm(self, swarm: Swarm) -> None:
        await self._db.execute(
            """INSERT INTO swarms
               (id, project_id, name, original_task, rephrased_task, status,
                priority, concurrency_limit, error_policy, created_at, updated_at,
                started_at, completed_at, template_id, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(id) DO UPDATE SET
                 name = excluded.name,
                 original_task = excluded.original_task,
                 rephrased_task = excluded.rephrased_task,
                 status = excluded.status,
                 priority = excluded.priority,
                 concurrency_limit = excluded.concurrency_limit,
                 error_policy = excluded.error_policy,
                 updated_at = excluded.updated_at,
                 started_at = excluded.started_at,
                 completed_at = excluded.completed_at,
                 template_id = excluded.template_id,
                 metadata = excluded.metadata
            """,
            (swarm.id, swarm.project_id, swarm.name, swarm.original_task,
             swarm.rephrased_task, swarm.status, swarm.priority,
             swarm.concurrency_limit, swarm.error_policy, swarm.created_at,
             swarm.updated_at, swarm.started_at, swarm.completed_at,
             swarm.template_id, json.dumps(swarm.metadata)),
        )
        await self._db.commit()

    async def get_swarm(self, swarm_id: str) -> Swarm | None:
        async with self._db.execute(
            "SELECT * FROM swarms WHERE id = ?", (swarm_id,)
        ) as cursor:
            row = await cursor.fetchone()
        return Swarm.from_dict(dict(row)) if row else None

    async def list_swarms(self, project_id: str | None = None, limit: int = 100) -> list[Swarm]:
        swarms: list[Swarm] = []
        if project_id:
            query = "SELECT * FROM swarms WHERE project_id = ? ORDER BY updated_at DESC LIMIT ?"
            params: tuple[Any, ...] = (project_id, limit)
        else:
            query = "SELECT * FROM swarms ORDER BY updated_at DESC LIMIT ?"
            params = (limit,)

        async with self._db.execute(query, params) as cursor:
            async for row in cursor:
                swarms.append(Swarm.from_dict(dict(row)))
        return swarms

    async def delete_swarm(self, swarm_id: str) -> bool:
        swarm = await self.get_swarm(swarm_id)
        if swarm and swarm.status in ("running", "decomposing"):
            return False

        # Cascade delete tasks, edges, artifacts, audit, costs, checkpoints.
        for table in ("swarm_task_edges", "swarm_artifacts", "swarm_audit_log",
                       "swarm_cost_log", "swarm_task_runs", "swarm_checkpoints"):
            await self._db.execute(f"DELETE FROM {table} WHERE swarm_id = ?", (swarm_id,))
        await self._db.execute("DELETE FROM swarm_tasks WHERE swarm_id = ?", (swarm_id,))
        await self._db.execute("DELETE FROM swarms WHERE id = ?", (swarm_id,))
        await self._db.commit()
        return True

    async def list_running_swarms(self) -> list[Swarm]:
        """List swarms in running state (for engine startup)."""
        swarms: list[Swarm] = []
        async with self._db.execute(
            "SELECT * FROM swarms WHERE status = 'running'"
        ) as cursor:
            async for row in cursor:
                swarms.append(Swarm.from_dict(dict(row)))
        return swarms

    # ── Tasks ─────────────────────────────────────────────────

    async def save_task(self, task: SwarmTask) -> None:
        await self._db.execute(
            """INSERT INTO swarm_tasks
               (id, swarm_id, name, description, status, priority,
                assigned_instance, assigned_persona, concern_id,
                position_x, position_y,
                retry_count, max_retries, retry_backoff_seconds,
                fallback_persona, timeout_seconds,
                timeout_warn_seconds, timeout_extend_seconds,
                requires_approval, approval_status, approved_by,
                is_periodic, cron_expression, next_run_at,
                created_at, updated_at, started_at, completed_at,
                input_data, output_data, error_message, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(id) DO UPDATE SET
                 name = excluded.name,
                 description = excluded.description,
                 status = excluded.status,
                 priority = excluded.priority,
                 assigned_instance = excluded.assigned_instance,
                 assigned_persona = excluded.assigned_persona,
                 concern_id = excluded.concern_id,
                 position_x = excluded.position_x,
                 position_y = excluded.position_y,
                 retry_count = excluded.retry_count,
                 max_retries = excluded.max_retries,
                 retry_backoff_seconds = excluded.retry_backoff_seconds,
                 fallback_persona = excluded.fallback_persona,
                 timeout_seconds = excluded.timeout_seconds,
                 timeout_warn_seconds = excluded.timeout_warn_seconds,
                 timeout_extend_seconds = excluded.timeout_extend_seconds,
                 requires_approval = excluded.requires_approval,
                 approval_status = excluded.approval_status,
                 approved_by = excluded.approved_by,
                 is_periodic = excluded.is_periodic,
                 cron_expression = excluded.cron_expression,
                 next_run_at = excluded.next_run_at,
                 updated_at = excluded.updated_at,
                 started_at = excluded.started_at,
                 completed_at = excluded.completed_at,
                 input_data = excluded.input_data,
                 output_data = excluded.output_data,
                 error_message = excluded.error_message,
                 metadata = excluded.metadata
            """,
            (task.id, task.swarm_id, task.name, task.description,
             task.status, task.priority,
             task.assigned_instance, task.assigned_persona, task.concern_id,
             task.position_x, task.position_y,
             task.retry_count, task.max_retries, task.retry_backoff_seconds,
             task.fallback_persona, task.timeout_seconds,
             task.timeout_warn_seconds, task.timeout_extend_seconds,
             int(task.requires_approval), task.approval_status, task.approved_by,
             int(task.is_periodic), task.cron_expression, task.next_run_at,
             task.created_at, task.updated_at, task.started_at, task.completed_at,
             json.dumps(task.input_data), json.dumps(task.output_data),
             task.error_message, json.dumps(task.metadata)),
        )
        await self._db.commit()

    async def get_task(self, task_id: str) -> SwarmTask | None:
        async with self._db.execute(
            "SELECT * FROM swarm_tasks WHERE id = ?", (task_id,)
        ) as cursor:
            row = await cursor.fetchone()
        return SwarmTask.from_dict(dict(row)) if row else None

    async def list_tasks(self, swarm_id: str) -> list[SwarmTask]:
        tasks: list[SwarmTask] = []
        async with self._db.execute(
            "SELECT * FROM swarm_tasks WHERE swarm_id = ? ORDER BY priority DESC, name",
            (swarm_id,),
        ) as cursor:
            async for row in cursor:
                tasks.append(SwarmTask.from_dict(dict(row)))
        return tasks

    async def delete_task(self, task_id: str) -> None:
        # Remove edges referencing this task.
        await self._db.execute(
            "DELETE FROM swarm_task_edges WHERE from_task_id = ? OR to_task_id = ?",
            (task_id, task_id),
        )
        await self._db.execute("DELETE FROM swarm_artifacts WHERE task_id = ?", (task_id,))
        await self._db.execute("DELETE FROM swarm_tasks WHERE id = ?", (task_id,))
        await self._db.commit()

    # ── Edges ─────────────────────────────────────────────────

    async def save_edge(self, edge: SwarmEdge) -> int:
        """Insert an edge. Returns the edge ID."""
        cursor = await self._db.execute(
            """INSERT INTO swarm_task_edges (swarm_id, from_task_id, to_task_id, edge_type)
               VALUES (?, ?, ?, ?)""",
            (edge.swarm_id, edge.from_task_id, edge.to_task_id, edge.edge_type),
        )
        await self._db.commit()
        return cursor.lastrowid or 0

    async def list_edges(self, swarm_id: str) -> list[SwarmEdge]:
        edges: list[SwarmEdge] = []
        async with self._db.execute(
            "SELECT * FROM swarm_task_edges WHERE swarm_id = ?", (swarm_id,)
        ) as cursor:
            async for row in cursor:
                edges.append(SwarmEdge.from_dict(dict(row)))
        return edges

    async def delete_edge(self, edge_id: int) -> None:
        await self._db.execute("DELETE FROM swarm_task_edges WHERE id = ?", (edge_id,))
        await self._db.commit()

    # ── Artifacts ─────────────────────────────────────────────

    async def save_artifact(self, artifact: SwarmArtifact) -> None:
        await self._db.execute(
            """INSERT INTO swarm_artifacts
               (id, task_id, swarm_id, label, content_type, content, created_at, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(id) DO UPDATE SET
                 label = excluded.label,
                 content = excluded.content,
                 metadata = excluded.metadata
            """,
            (artifact.id, artifact.task_id, artifact.swarm_id,
             artifact.label, artifact.content_type, artifact.content,
             artifact.created_at, json.dumps(artifact.metadata)),
        )
        await self._db.commit()

    async def list_artifacts(self, swarm_id: str, task_id: str | None = None) -> list[SwarmArtifact]:
        artifacts: list[SwarmArtifact] = []
        if task_id:
            query = "SELECT * FROM swarm_artifacts WHERE swarm_id = ? AND task_id = ? ORDER BY created_at"
            params: tuple[Any, ...] = (swarm_id, task_id)
        else:
            query = "SELECT * FROM swarm_artifacts WHERE swarm_id = ? ORDER BY created_at"
            params = (swarm_id,)

        async with self._db.execute(query, params) as cursor:
            async for row in cursor:
                artifacts.append(SwarmArtifact.from_dict(dict(row)))
        return artifacts

    # ── Audit log ─────────────────────────────────────────────

    async def add_audit_entry(self, entry: SwarmAuditEntry) -> None:
        await self._db.execute(
            """INSERT INTO swarm_audit_log
               (swarm_id, task_id, event_type, details, actor, severity, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (entry.swarm_id, entry.task_id, entry.event_type,
             json.dumps(entry.details), entry.actor, entry.severity,
             entry.created_at),
        )
        await self._db.commit()

    async def list_audit_log(self, swarm_id: str, limit: int = 200) -> list[SwarmAuditEntry]:
        entries: list[SwarmAuditEntry] = []
        async with self._db.execute(
            "SELECT * FROM swarm_audit_log WHERE swarm_id = ? ORDER BY id DESC LIMIT ?",
            (swarm_id, limit),
        ) as cursor:
            async for row in cursor:
                entries.append(SwarmAuditEntry.from_dict(dict(row)))
        return entries

    # ── Cost log ──────────────────────────────────────────────

    async def add_cost_entry(
        self, swarm_id: str, task_id: str = "", instance_name: str = "",
        persona_name: str = "", tokens_in: int = 0, tokens_out: int = 0,
        cost_usd: float = 0.0,
    ) -> None:
        from botport.swarm.models import _utcnow_iso
        await self._db.execute(
            """INSERT INTO swarm_cost_log
               (swarm_id, task_id, instance_name, persona_name,
                tokens_in, tokens_out, cost_usd, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (swarm_id, task_id, instance_name, persona_name,
             tokens_in, tokens_out, cost_usd, _utcnow_iso()),
        )
        await self._db.commit()

    async def get_cost_summary(self, swarm_id: str) -> dict[str, Any]:
        result: dict[str, Any] = {"total_tokens_in": 0, "total_tokens_out": 0, "total_cost_usd": 0.0, "by_task": []}
        async with self._db.execute(
            """SELECT task_id, instance_name, persona_name,
                      SUM(tokens_in) as t_in, SUM(tokens_out) as t_out, SUM(cost_usd) as cost
               FROM swarm_cost_log WHERE swarm_id = ?
               GROUP BY task_id""",
            (swarm_id,),
        ) as cursor:
            async for row in cursor:
                entry = {
                    "task_id": row["task_id"],
                    "instance_name": row["instance_name"],
                    "persona_name": row["persona_name"],
                    "tokens_in": row["t_in"],
                    "tokens_out": row["t_out"],
                    "cost_usd": row["cost"],
                }
                result["by_task"].append(entry)
                result["total_tokens_in"] += row["t_in"]
                result["total_tokens_out"] += row["t_out"]
                result["total_cost_usd"] += row["cost"]
        return result

    # ── Templates ─────────────────────────────────────────────

    async def save_template(self, template: SwarmTemplate) -> None:
        await self._db.execute(
            """INSERT INTO swarm_templates (id, name, description, dag_definition, created_at, metadata)
               VALUES (?, ?, ?, ?, ?, ?)
               ON CONFLICT(id) DO UPDATE SET
                 name = excluded.name,
                 description = excluded.description,
                 dag_definition = excluded.dag_definition,
                 metadata = excluded.metadata
            """,
            (template.id, template.name, template.description,
             json.dumps(template.dag_definition), template.created_at,
             json.dumps(template.metadata)),
        )
        await self._db.commit()

    async def list_templates(self) -> list[SwarmTemplate]:
        templates: list[SwarmTemplate] = []
        async with self._db.execute(
            "SELECT * FROM swarm_templates ORDER BY created_at DESC"
        ) as cursor:
            async for row in cursor:
                templates.append(SwarmTemplate.from_dict(dict(row)))
        return templates

    async def get_template(self, template_id: str) -> SwarmTemplate | None:
        async with self._db.execute(
            "SELECT * FROM swarm_templates WHERE id = ?", (template_id,)
        ) as cursor:
            row = await cursor.fetchone()
        return SwarmTemplate.from_dict(dict(row)) if row else None

    async def delete_template(self, template_id: str) -> None:
        await self._db.execute("DELETE FROM swarm_templates WHERE id = ?", (template_id,))
        await self._db.commit()

    # ── Periodic tasks ───────────────────────────────────────

    async def list_due_periodic_tasks(self, now_iso: str) -> list[SwarmTask]:
        """List periodic tasks whose next_run_at is due (past or equal to now)."""
        tasks: list[SwarmTask] = []
        async with self._db.execute(
            """SELECT * FROM swarm_tasks
               WHERE is_periodic = 1
                 AND next_run_at != ''
                 AND next_run_at <= ?
                 AND status IN ('completed', 'queued')
               ORDER BY next_run_at""",
            (now_iso,),
        ) as cursor:
            async for row in cursor:
                tasks.append(SwarmTask.from_dict(dict(row)))
        return tasks

    async def list_cost_log(self, swarm_id: str) -> list[dict[str, Any]]:
        """List individual cost log entries for a swarm."""
        entries: list[dict[str, Any]] = []
        async with self._db.execute(
            "SELECT * FROM swarm_cost_log WHERE swarm_id = ? ORDER BY created_at DESC",
            (swarm_id,),
        ) as cursor:
            async for row in cursor:
                entries.append(dict(row))
        return entries

    # ── Checkpoints ──────────────────────────────────────────

    async def save_checkpoint(self, checkpoint: SwarmCheckpoint) -> None:
        await self._db.execute(
            """INSERT INTO swarm_checkpoints
               (id, swarm_id, label, swarm_state, task_states, edge_states, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(id) DO UPDATE SET
                 label = excluded.label,
                 swarm_state = excluded.swarm_state,
                 task_states = excluded.task_states,
                 edge_states = excluded.edge_states
            """,
            (checkpoint.id, checkpoint.swarm_id, checkpoint.label,
             json.dumps(checkpoint.swarm_state), json.dumps(checkpoint.task_states),
             json.dumps(checkpoint.edge_states), checkpoint.created_at),
        )
        await self._db.commit()

    async def get_checkpoint(self, checkpoint_id: str) -> SwarmCheckpoint | None:
        async with self._db.execute(
            "SELECT * FROM swarm_checkpoints WHERE id = ?", (checkpoint_id,)
        ) as cursor:
            row = await cursor.fetchone()
        return SwarmCheckpoint.from_dict(dict(row)) if row else None

    async def list_checkpoints(self, swarm_id: str) -> list[SwarmCheckpoint]:
        checkpoints: list[SwarmCheckpoint] = []
        async with self._db.execute(
            "SELECT * FROM swarm_checkpoints WHERE swarm_id = ? ORDER BY created_at DESC",
            (swarm_id,),
        ) as cursor:
            async for row in cursor:
                checkpoints.append(SwarmCheckpoint.from_dict(dict(row)))
        return checkpoints

    async def delete_checkpoint(self, checkpoint_id: str) -> None:
        await self._db.execute("DELETE FROM swarm_checkpoints WHERE id = ?", (checkpoint_id,))
        await self._db.commit()
