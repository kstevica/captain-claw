"""Project management — persistent goal-oriented containers for multi-agent work.

A project accumulates knowledge, coordinates agents, and provides continuity
across sessions, forged teams, and solo work.  It owns scoped memory (semantic
chunks + insights), artifacts, an activity log, and agent membership.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from typing import Any

import aiosqlite

from captain_claw.logging import get_logger

log = get_logger(__name__)


def _utcnow_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


# ── ProjectManager ──────────────────────────────────────────────────


class ProjectManager:
    """Manages the project lifecycle, membership, artifacts, and activity log.

    Shares the session SQLite database so that project tables live alongside
    sessions, todos, contacts, etc.
    """

    def __init__(self, db: aiosqlite.Connection) -> None:
        self._db = db
        self._migrated = False

    # ── schema ──────────────────────────────────────────────────────

    async def ensure_tables(self) -> None:
        """Create project tables if they don't exist (idempotent)."""
        if self._migrated:
            return
        db = self._db

        await db.execute("""
            CREATE TABLE IF NOT EXISTS projects (
                id              TEXT PRIMARY KEY,
                name            TEXT NOT NULL UNIQUE,
                description     TEXT NOT NULL DEFAULT '',
                status          TEXT NOT NULL DEFAULT 'active',
                goals           TEXT NOT NULL DEFAULT '[]',
                config          TEXT NOT NULL DEFAULT '{}',
                lead_agent_id   TEXT NOT NULL DEFAULT '',
                created_at      TEXT NOT NULL,
                updated_at      TEXT NOT NULL,
                completed_at    TEXT,
                metadata        TEXT NOT NULL DEFAULT '{}'
            )
        """)
        await db.execute(
            "CREATE INDEX IF NOT EXISTS idx_projects_status ON projects(status)"
        )
        await db.execute(
            "CREATE INDEX IF NOT EXISTS idx_projects_name ON projects(name COLLATE NOCASE)"
        )

        await db.execute("""
            CREATE TABLE IF NOT EXISTS project_members (
                id              TEXT PRIMARY KEY,
                project_id      TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                agent_id        TEXT NOT NULL,
                agent_name      TEXT NOT NULL DEFAULT '',
                role            TEXT NOT NULL DEFAULT 'contributor',
                expertise_tags  TEXT NOT NULL DEFAULT '[]',
                joined_at       TEXT NOT NULL,
                left_at         TEXT,
                contribution_summary TEXT NOT NULL DEFAULT ''
            )
        """)
        await db.execute(
            "CREATE INDEX IF NOT EXISTS idx_pm_project ON project_members(project_id)"
        )
        await db.execute(
            "CREATE INDEX IF NOT EXISTS idx_pm_agent ON project_members(agent_id)"
        )
        await db.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_pm_unique "
            "ON project_members(project_id, agent_id, joined_at)"
        )

        await db.execute("""
            CREATE TABLE IF NOT EXISTS project_sessions (
                id              TEXT PRIMARY KEY,
                project_id      TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                session_id      TEXT NOT NULL,
                session_name    TEXT NOT NULL DEFAULT '',
                agent_id        TEXT NOT NULL DEFAULT '',
                purpose         TEXT NOT NULL DEFAULT '',
                started_at      TEXT NOT NULL,
                ended_at        TEXT
            )
        """)
        await db.execute(
            "CREATE INDEX IF NOT EXISTS idx_ps_project ON project_sessions(project_id)"
        )
        await db.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_ps_unique "
            "ON project_sessions(project_id, session_id)"
        )

        await db.execute("""
            CREATE TABLE IF NOT EXISTS project_artifacts (
                id              TEXT PRIMARY KEY,
                project_id      TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                kind            TEXT NOT NULL,
                title           TEXT NOT NULL,
                content         TEXT NOT NULL DEFAULT '',
                created_by      TEXT NOT NULL DEFAULT '',
                session_id      TEXT NOT NULL DEFAULT '',
                status          TEXT NOT NULL DEFAULT 'active',
                created_at      TEXT NOT NULL,
                updated_at      TEXT NOT NULL
            )
        """)
        await db.execute(
            "CREATE INDEX IF NOT EXISTS idx_pa_project ON project_artifacts(project_id)"
        )
        await db.execute(
            "CREATE INDEX IF NOT EXISTS idx_pa_kind ON project_artifacts(project_id, kind)"
        )

        await db.execute("""
            CREATE TABLE IF NOT EXISTS project_activity (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id      TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                event_type      TEXT NOT NULL,
                agent_id        TEXT NOT NULL DEFAULT '',
                detail          TEXT NOT NULL DEFAULT '{}',
                created_at      TEXT NOT NULL
            )
        """)
        await db.execute(
            "CREATE INDEX IF NOT EXISTS idx_pact_project ON project_activity(project_id)"
        )
        await db.execute(
            "CREATE INDEX IF NOT EXISTS idx_pact_created ON project_activity(created_at)"
        )

        await db.commit()
        self._migrated = True

    # ── project CRUD ────────────────────────────────────────────────

    async def create(
        self,
        name: str,
        *,
        description: str = "",
        goals: list[dict[str, Any]] | None = None,
        config: dict[str, Any] | None = None,
        lead_agent_id: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create a new project."""
        await self.ensure_tables()
        now = _utcnow_iso()
        project_id = _new_id()
        await self._db.execute(
            """INSERT INTO projects
               (id, name, description, status, goals, config,
                lead_agent_id, created_at, updated_at, metadata)
               VALUES (?, ?, ?, 'active', ?, ?, ?, ?, ?, ?)""",
            (
                project_id,
                name.strip(),
                description.strip(),
                json.dumps(goals or []),
                json.dumps(config or {}),
                lead_agent_id,
                now,
                now,
                json.dumps(metadata or {}),
            ),
        )
        await self._db.commit()
        await self.log_activity(project_id, "project_created", detail={"name": name})
        log.info("Project created", id=project_id, name=name)
        return await self.get(project_id)  # type: ignore[return-value]

    async def get(self, project_id: str) -> dict[str, Any] | None:
        """Get a project by ID."""
        await self.ensure_tables()
        rows = await self._db.execute_fetchall(
            "SELECT * FROM projects WHERE id = ?", (project_id,),
        )
        return self._project_row_to_dict(rows[0]) if rows else None

    async def get_by_name(self, name: str) -> dict[str, Any] | None:
        """Get a project by name (case-insensitive)."""
        await self.ensure_tables()
        rows = await self._db.execute_fetchall(
            "SELECT * FROM projects WHERE name = ? COLLATE NOCASE", (name.strip(),),
        )
        return self._project_row_to_dict(rows[0]) if rows else None

    async def update(self, project_id: str, **fields: Any) -> dict[str, Any] | None:
        """Update project fields."""
        await self.ensure_tables()
        allowed = {
            "name", "description", "status", "goals", "config",
            "lead_agent_id", "completed_at", "metadata",
        }
        updates: dict[str, Any] = {}
        for k, v in fields.items():
            if k not in allowed or v is None:
                continue
            if k in ("goals", "config", "metadata") and isinstance(v, (list, dict)):
                updates[k] = json.dumps(v)
            else:
                updates[k] = v
        if not updates:
            return await self.get(project_id)
        updates["updated_at"] = _utcnow_iso()
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [project_id]
        await self._db.execute(
            f"UPDATE projects SET {set_clause} WHERE id = ?", values,
        )
        await self._db.commit()
        if "status" in fields:
            await self.log_activity(
                project_id, "status_changed",
                detail={"new_status": fields["status"]},
            )
        return await self.get(project_id)

    async def list_projects(
        self,
        *,
        status: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """List projects, optionally filtered by status."""
        await self.ensure_tables()
        if status:
            rows = await self._db.execute_fetchall(
                "SELECT * FROM projects WHERE status = ? ORDER BY updated_at DESC LIMIT ?",
                (status, limit),
            )
        else:
            rows = await self._db.execute_fetchall(
                "SELECT * FROM projects ORDER BY updated_at DESC LIMIT ?",
                (limit,),
            )
        return [self._project_row_to_dict(r) for r in rows]

    async def delete(self, project_id: str) -> bool:
        """Delete a project (cascades to members, sessions, artifacts, activity)."""
        await self.ensure_tables()
        cursor = await self._db.execute(
            "DELETE FROM projects WHERE id = ?", (project_id,),
        )
        await self._db.commit()
        return (cursor.rowcount or 0) > 0

    # ── membership ──────────────────────────────────────────────────

    async def join(
        self,
        project_id: str,
        agent_id: str,
        *,
        agent_name: str = "",
        role: str = "contributor",
        expertise_tags: list[str] | None = None,
    ) -> dict[str, Any]:
        """Add an agent to a project (or re-join if previously left)."""
        await self.ensure_tables()
        now = _utcnow_iso()
        member_id = _new_id()
        await self._db.execute(
            """INSERT INTO project_members
               (id, project_id, agent_id, agent_name, role, expertise_tags, joined_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                member_id,
                project_id,
                agent_id,
                agent_name,
                role,
                json.dumps(expertise_tags or []),
                now,
            ),
        )
        await self._db.commit()
        await self.log_activity(
            project_id, "agent_joined",
            agent_id=agent_id,
            detail={"agent_name": agent_name, "role": role},
        )
        log.info("Agent joined project", project_id=project_id, agent_id=agent_id, role=role)
        return {
            "id": member_id,
            "project_id": project_id,
            "agent_id": agent_id,
            "agent_name": agent_name,
            "role": role,
            "expertise_tags": expertise_tags or [],
            "joined_at": now,
        }

    async def leave(
        self,
        project_id: str,
        agent_id: str,
        *,
        contribution_summary: str = "",
    ) -> bool:
        """Mark an active agent as having left the project."""
        await self.ensure_tables()
        now = _utcnow_iso()
        cursor = await self._db.execute(
            """UPDATE project_members
               SET left_at = ?, contribution_summary = ?
               WHERE project_id = ? AND agent_id = ? AND left_at IS NULL""",
            (now, contribution_summary, project_id, agent_id),
        )
        await self._db.commit()
        if (cursor.rowcount or 0) > 0:
            await self.log_activity(
                project_id, "agent_left",
                agent_id=agent_id,
                detail={"contribution_summary": contribution_summary[:500]},
            )
            log.info("Agent left project", project_id=project_id, agent_id=agent_id)
            return True
        return False

    async def active_members(self, project_id: str) -> list[dict[str, Any]]:
        """List active (not left) members of a project."""
        await self.ensure_tables()
        rows = await self._db.execute_fetchall(
            """SELECT * FROM project_members
               WHERE project_id = ? AND left_at IS NULL
               ORDER BY joined_at""",
            (project_id,),
        )
        return [self._member_row_to_dict(r) for r in rows]

    async def all_members(self, project_id: str) -> list[dict[str, Any]]:
        """List all members (current + past) of a project."""
        await self.ensure_tables()
        rows = await self._db.execute_fetchall(
            "SELECT * FROM project_members WHERE project_id = ? ORDER BY joined_at",
            (project_id,),
        )
        return [self._member_row_to_dict(r) for r in rows]

    async def agent_projects(
        self,
        agent_id: str,
        *,
        active_only: bool = True,
    ) -> list[dict[str, Any]]:
        """Get projects an agent belongs to."""
        await self.ensure_tables()
        if active_only:
            rows = await self._db.execute_fetchall(
                """SELECT p.* FROM projects p
                   JOIN project_members pm ON pm.project_id = p.id
                   WHERE pm.agent_id = ? AND pm.left_at IS NULL
                   ORDER BY p.updated_at DESC""",
                (agent_id,),
            )
        else:
            rows = await self._db.execute_fetchall(
                """SELECT DISTINCT p.* FROM projects p
                   JOIN project_members pm ON pm.project_id = p.id
                   WHERE pm.agent_id = ?
                   ORDER BY p.updated_at DESC""",
                (agent_id,),
            )
        return [self._project_row_to_dict(r) for r in rows]

    async def get_membership(
        self,
        project_id: str,
        agent_id: str,
    ) -> dict[str, Any] | None:
        """Get active membership record for an agent in a project."""
        await self.ensure_tables()
        rows = await self._db.execute_fetchall(
            """SELECT * FROM project_members
               WHERE project_id = ? AND agent_id = ? AND left_at IS NULL
               LIMIT 1""",
            (project_id, agent_id),
        )
        return self._member_row_to_dict(rows[0]) if rows else None

    # ── sessions ────────────────────────────────────────────────────

    async def link_session(
        self,
        project_id: str,
        session_id: str,
        *,
        session_name: str = "",
        agent_id: str = "",
        purpose: str = "",
    ) -> dict[str, Any]:
        """Associate a session with a project."""
        await self.ensure_tables()
        now = _utcnow_iso()
        link_id = _new_id()
        await self._db.execute(
            """INSERT OR IGNORE INTO project_sessions
               (id, project_id, session_id, session_name, agent_id, purpose, started_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (link_id, project_id, session_id, session_name, agent_id, purpose, now),
        )
        await self._db.commit()
        await self.log_activity(
            project_id, "session_started",
            agent_id=agent_id,
            detail={"session_id": session_id, "purpose": purpose},
        )
        return {
            "id": link_id,
            "project_id": project_id,
            "session_id": session_id,
            "session_name": session_name,
            "agent_id": agent_id,
            "purpose": purpose,
            "started_at": now,
        }

    async def end_session(self, project_id: str, session_id: str) -> bool:
        """Mark a project-session link as ended."""
        await self.ensure_tables()
        now = _utcnow_iso()
        cursor = await self._db.execute(
            """UPDATE project_sessions SET ended_at = ?
               WHERE project_id = ? AND session_id = ? AND ended_at IS NULL""",
            (now, project_id, session_id),
        )
        await self._db.commit()
        if (cursor.rowcount or 0) > 0:
            await self.log_activity(
                project_id, "session_ended",
                detail={"session_id": session_id},
            )
            return True
        return False

    async def project_sessions(self, project_id: str) -> list[dict[str, Any]]:
        """List all sessions linked to a project."""
        await self.ensure_tables()
        rows = await self._db.execute_fetchall(
            "SELECT * FROM project_sessions WHERE project_id = ? ORDER BY started_at DESC",
            (project_id,),
        )
        return [self._session_row_to_dict(r) for r in rows]

    async def session_project(self, session_id: str) -> dict[str, Any] | None:
        """Find which project a session belongs to (if any)."""
        await self.ensure_tables()
        rows = await self._db.execute_fetchall(
            """SELECT p.* FROM projects p
               JOIN project_sessions ps ON ps.project_id = p.id
               WHERE ps.session_id = ?
               LIMIT 1""",
            (session_id,),
        )
        return self._project_row_to_dict(rows[0]) if rows else None

    # ── artifacts ───────────────────────────────────────────────────

    async def add_artifact(
        self,
        project_id: str,
        kind: str,
        title: str,
        *,
        content: str = "",
        created_by: str = "",
        session_id: str = "",
        status: str = "active",
    ) -> dict[str, Any]:
        """Add an artifact (decision, milestone, deliverable, note, blocker)."""
        await self.ensure_tables()
        now = _utcnow_iso()
        artifact_id = _new_id()
        await self._db.execute(
            """INSERT INTO project_artifacts
               (id, project_id, kind, title, content, created_by, session_id, status, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (artifact_id, project_id, kind, title, content, created_by, session_id, status, now, now),
        )
        await self._db.commit()
        await self.log_activity(
            project_id, "artifact_created",
            agent_id=created_by,
            detail={"artifact_id": artifact_id, "kind": kind, "title": title},
        )
        return {
            "id": artifact_id,
            "project_id": project_id,
            "kind": kind,
            "title": title,
            "content": content,
            "created_by": created_by,
            "session_id": session_id,
            "status": status,
            "created_at": now,
            "updated_at": now,
        }

    async def list_artifacts(
        self,
        project_id: str,
        *,
        kind: str | None = None,
        status: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """List project artifacts with optional filters."""
        await self.ensure_tables()
        clauses = ["project_id = ?"]
        params: list[Any] = [project_id]
        if kind:
            clauses.append("kind = ?")
            params.append(kind)
        if status:
            clauses.append("status = ?")
            params.append(status)
        params.append(limit)
        where = " AND ".join(clauses)
        rows = await self._db.execute_fetchall(
            f"SELECT * FROM project_artifacts WHERE {where} ORDER BY created_at DESC LIMIT ?",
            params,
        )
        return [self._artifact_row_to_dict(r) for r in rows]

    async def update_artifact(
        self,
        artifact_id: str,
        **fields: Any,
    ) -> dict[str, Any] | None:
        """Update an artifact."""
        await self.ensure_tables()
        allowed = {"title", "content", "status", "kind"}
        updates = {k: v for k, v in fields.items() if k in allowed and v is not None}
        if not updates:
            return None
        updates["updated_at"] = _utcnow_iso()
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [artifact_id]
        await self._db.execute(
            f"UPDATE project_artifacts SET {set_clause} WHERE id = ?", values,
        )
        await self._db.commit()
        rows = await self._db.execute_fetchall(
            "SELECT * FROM project_artifacts WHERE id = ?", (artifact_id,),
        )
        return self._artifact_row_to_dict(rows[0]) if rows else None

    # ── goals ───────────────────────────────────────────────────────

    async def add_goal(
        self,
        project_id: str,
        goal: str,
        *,
        success_criteria: str = "",
        priority: str = "normal",
    ) -> dict[str, Any] | None:
        """Append a goal to the project's goals list."""
        project = await self.get(project_id)
        if not project:
            return None
        goals = project.get("goals", [])
        new_goal = {
            "goal": goal,
            "success_criteria": success_criteria,
            "priority": priority,
            "status": "pending",
        }
        goals.append(new_goal)
        await self.update(project_id, goals=goals)
        await self.log_activity(
            project_id, "goal_added",
            detail={"goal": goal, "priority": priority},
        )
        return new_goal

    async def update_goal(
        self,
        project_id: str,
        goal_index: int,
        **fields: Any,
    ) -> dict[str, Any] | None:
        """Update a specific goal by index."""
        project = await self.get(project_id)
        if not project:
            return None
        goals = project.get("goals", [])
        if goal_index < 0 or goal_index >= len(goals):
            return None
        allowed = {"goal", "success_criteria", "priority", "status", "notes"}
        for k, v in fields.items():
            if k in allowed and v is not None:
                goals[goal_index][k] = v
        await self.update(project_id, goals=goals)
        await self.log_activity(
            project_id, "goal_updated",
            detail={"goal_index": goal_index, **{k: v for k, v in fields.items() if k in allowed}},
        )
        return goals[goal_index]

    # ── activity ────────────────────────────────────────────────────

    async def log_activity(
        self,
        project_id: str,
        event_type: str,
        *,
        agent_id: str = "",
        detail: dict[str, Any] | None = None,
    ) -> None:
        """Append an event to the project activity log."""
        await self.ensure_tables()
        await self._db.execute(
            """INSERT INTO project_activity
               (project_id, event_type, agent_id, detail, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            (project_id, event_type, agent_id, json.dumps(detail or {}), _utcnow_iso()),
        )
        await self._db.commit()

    async def recent_activity(
        self,
        project_id: str,
        *,
        limit: int = 20,
        event_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get recent activity for a project."""
        await self.ensure_tables()
        if event_type:
            rows = await self._db.execute_fetchall(
                """SELECT * FROM project_activity
                   WHERE project_id = ? AND event_type = ?
                   ORDER BY created_at DESC LIMIT ?""",
                (project_id, event_type, limit),
            )
        else:
            rows = await self._db.execute_fetchall(
                """SELECT * FROM project_activity
                   WHERE project_id = ?
                   ORDER BY created_at DESC LIMIT ?""",
                (project_id, limit),
            )
        return [self._activity_row_to_dict(r) for r in rows]

    # ── summary / context ───────────────────────────────────────────

    async def get_project_context(self, project_id: str) -> dict[str, Any] | None:
        """Build a rich context bundle for system prompt injection."""
        project = await self.get(project_id)
        if not project:
            return None
        members = await self.active_members(project_id)
        decisions = await self.list_artifacts(project_id, kind="decision", limit=10)
        blockers = await self.list_artifacts(project_id, kind="blocker", status="active", limit=5)
        milestones = await self.list_artifacts(project_id, kind="milestone", limit=5)
        activity = await self.recent_activity(project_id, limit=10)
        return {
            "project": project,
            "members": members,
            "decisions": decisions,
            "blockers": blockers,
            "milestones": milestones,
            "recent_activity": activity,
        }

    # ── row converters ──────────────────────────────────────────────

    @staticmethod
    def _project_row_to_dict(row: Any) -> dict[str, Any]:
        """Convert a project row tuple to dict."""
        d = {
            "id": row[0],
            "name": row[1],
            "description": row[2],
            "status": row[3],
            "goals": json.loads(row[4]) if row[4] else [],
            "config": json.loads(row[5]) if row[5] else {},
            "lead_agent_id": row[6],
            "created_at": row[7],
            "updated_at": row[8],
            "completed_at": row[9],
            "metadata": json.loads(row[10]) if row[10] else {},
        }
        return d

    @staticmethod
    def _member_row_to_dict(row: Any) -> dict[str, Any]:
        return {
            "id": row[0],
            "project_id": row[1],
            "agent_id": row[2],
            "agent_name": row[3],
            "role": row[4],
            "expertise_tags": json.loads(row[5]) if row[5] else [],
            "joined_at": row[6],
            "left_at": row[7],
            "contribution_summary": row[8],
        }

    @staticmethod
    def _session_row_to_dict(row: Any) -> dict[str, Any]:
        return {
            "id": row[0],
            "project_id": row[1],
            "session_id": row[2],
            "session_name": row[3],
            "agent_id": row[4],
            "purpose": row[5],
            "started_at": row[6],
            "ended_at": row[7],
        }

    @staticmethod
    def _artifact_row_to_dict(row: Any) -> dict[str, Any]:
        return {
            "id": row[0],
            "project_id": row[1],
            "kind": row[2],
            "title": row[3],
            "content": row[4],
            "created_by": row[5],
            "session_id": row[6],
            "status": row[7],
            "created_at": row[8],
            "updated_at": row[9],
        }

    @staticmethod
    def _activity_row_to_dict(row: Any) -> dict[str, Any]:
        return {
            "id": row[0],
            "project_id": row[1],
            "event_type": row[2],
            "agent_id": row[3],
            "detail": json.loads(row[4]) if row[4] else {},
            "created_at": row[5],
        }


# ── Singleton access ────────────────────────────────────────────────

_instance: ProjectManager | None = None


def get_project_manager() -> ProjectManager | None:
    """Return the global ProjectManager instance (or None if not initialized)."""
    return _instance


def set_project_manager(pm: ProjectManager) -> None:
    """Set the global ProjectManager instance."""
    global _instance
    _instance = pm
