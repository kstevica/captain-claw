"""Project memory tool — search and interact with project-scoped knowledge.

Lets agents working inside or outside a project retrieve project memory,
insights, artifacts, status, and contribute findings back to the project.
"""

from __future__ import annotations

import json
from typing import Any

from captain_claw.logging import get_logger
from captain_claw.tools.registry import Tool, ToolResult

log = get_logger(__name__)


class ProjectMemoryTool(Tool):
    """Search and interact with project-scoped knowledge."""

    name = "project_memory"
    description = (
        "Access project knowledge: search project memory and insights, "
        "list artifacts and decisions, check project status and members, "
        "browse project files, or contribute new insights/artifacts back "
        "to a project. Use when you need context from a project you belong to."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "search",
                    "insights",
                    "artifacts",
                    "status",
                    "files",
                    "contribute",
                    "list_projects",
                ],
                "description": (
                    "search=semantic search across project memory, "
                    "insights=search project insights, "
                    "artifacts=list/search project artifacts (decisions, milestones, blockers), "
                    "status=project goals/members/recent activity, "
                    "files=project file registry, "
                    "contribute=push an insight or artifact to the project, "
                    "list_projects=show all projects you belong to"
                ),
            },
            "project": {
                "type": "string",
                "description": "Project name or ID (not required for list_projects).",
            },
            "query": {
                "type": "string",
                "description": "Search query (for search/insights/artifacts actions).",
            },
            "kind": {
                "type": "string",
                "enum": ["decision", "milestone", "deliverable", "note", "blocker"],
                "description": "Artifact kind filter (for artifacts) or kind to create (for contribute).",
            },
            "contribute_type": {
                "type": "string",
                "enum": ["insight", "artifact"],
                "description": "What to contribute (for contribute action).",
            },
            "content": {
                "type": "string",
                "description": "Content text (for contribute action).",
            },
            "title": {
                "type": "string",
                "description": "Artifact title (for contribute action with contribute_type=artifact).",
            },
            "category": {
                "type": "string",
                "enum": [
                    "contact", "decision", "preference", "fact",
                    "deadline", "project", "workflow",
                ],
                "description": "Insight category (for contribute action with contribute_type=insight).",
            },
            "importance": {
                "type": "integer",
                "description": "Importance 1-10 (for contribute insight).",
            },
            "limit": {
                "type": "integer",
                "description": "Max results. Default 10.",
            },
        },
        "required": ["action"],
    }

    async def execute(self, **kwargs: Any) -> ToolResult:
        action = str(kwargs.get("action", "")).strip()
        project_ref = str(kwargs.get("project", "") or "").strip()
        query = str(kwargs.get("query", "") or "").strip()
        kind = str(kwargs.get("kind", "") or "").strip() or None
        limit = int(kwargs.get("limit") or 10)

        session_id = str(kwargs.get("_session_id", "") or "").strip() or None
        agent_id = str(kwargs.get("_agent_id", "") or session_id or "").strip()

        try:
            from captain_claw.projects import get_project_manager

            pm = get_project_manager()
            if pm is None:
                return ToolResult(
                    success=False,
                    error="Project system not initialized. Ensure the session database is active.",
                )

            if action == "list_projects":
                return await self._list_projects(pm, agent_id)

            if not project_ref:
                return ToolResult(success=False, error="'project' is required for this action.")

            project = await pm.get_by_name(project_ref)
            if project is None:
                project = await pm.get(project_ref)
            if project is None:
                return ToolResult(success=False, error=f"Project not found: {project_ref}")

            project_id = project["id"]

            if action == "search":
                return await self._search(pm, project_id, query, limit)
            if action == "insights":
                return await self._insights(project_id, query, kind, limit)
            if action == "artifacts":
                return await self._artifacts(pm, project_id, query, kind, limit)
            if action == "status":
                return await self._status(pm, project_id)
            if action == "files":
                return await self._files(pm, project_id)
            if action == "contribute":
                return await self._contribute(
                    pm, project_id, agent_id, session_id, kwargs,
                )
            return ToolResult(success=False, error=f"Unknown action: {action}")

        except Exception as e:
            log.error("project_memory tool error", action=action, error=str(e))
            return ToolResult(success=False, error=str(e))

    # ── actions ──────────────────────────────────────────────────────

    @staticmethod
    async def _list_projects(pm: Any, agent_id: str) -> ToolResult:
        projects = await pm.agent_projects(agent_id, active_only=False)
        if not projects:
            all_projects = await pm.list_projects()
            if all_projects:
                lines = ["You are not a member of any project. Available projects:"]
                for p in all_projects:
                    lines.append(f"  • {p['name']} ({p['status']}) — {p['description'][:100]}")
                return ToolResult(success=True, content="\n".join(lines))
            return ToolResult(success=True, content="No projects exist yet.")
        lines = ["Your projects:"]
        for p in projects:
            lines.append(
                f"  • {p['name']} ({p['status']}) — {p['description'][:100]}  id={p['id']}"
            )
        return ToolResult(success=True, content="\n".join(lines))

    @staticmethod
    async def _search(pm: Any, project_id: str, query: str, limit: int) -> ToolResult:
        if not query:
            return ToolResult(success=False, error="'query' is required for search.")
        from captain_claw.memory import LayeredMemory

        # Get project session IDs for scoped search.
        sessions = await pm.project_sessions(project_id)
        session_ids = [s["session_id"] for s in sessions]
        if not session_ids:
            return ToolResult(
                success=True,
                content="No sessions linked to this project yet — no memory to search.",
            )

        # Try to use the global semantic memory if available.
        from captain_claw.semantic_memory import SemanticMemoryIndex
        try:
            from captain_claw.config import get_config
            cfg = get_config()
            from pathlib import Path
            db_path = Path(cfg.memory.path).expanduser()
            if not db_path.exists():
                return ToolResult(success=True, content="Semantic memory database not found.")
            import sqlite3
            conn = sqlite3.connect(str(db_path), check_same_thread=False)
            # Direct query against memory_chunks with project session scoping.
            placeholders = ",".join("?" for _ in session_ids)
            rows = conn.execute(
                f"""SELECT chunk_id, source, reference, path, start_line, end_line,
                           text, updated_at, text_l1, text_l2
                    FROM memory_chunks
                    WHERE (source = 'workspace' OR reference IN ({placeholders}))
                    AND text LIKE ?
                    ORDER BY updated_at DESC
                    LIMIT ?""",
                (*session_ids, f"%{query}%", limit),
            ).fetchall()
            conn.close()
            if not rows:
                return ToolResult(success=True, content="No matching memory found in project.")
            lines = [f"Project memory matches ({len(rows)} results):"]
            for r in rows:
                snippet = str(r[6])[:300].replace("\n", " ")
                lines.append(f"  • [{r[1]}] {r[3]}:{r[4]} — {snippet}")
            return ToolResult(success=True, content="\n".join(lines))
        except Exception as e:
            return ToolResult(success=False, error=f"Memory search failed: {e}")

    @staticmethod
    async def _insights(project_id: str, query: str, kind: str | None, limit: int) -> ToolResult:
        from captain_claw.insights import get_insights_manager

        mgr = get_insights_manager()
        if query:
            results = await mgr.search_in_project(query, project_id, limit=limit)
        else:
            results = await mgr.list_project_insights(
                project_id, limit=limit, category=kind,
            )
        if not results:
            return ToolResult(success=True, content="No project insights found.")
        lines = [f"Project insights ({len(results)}):"]
        for i in results:
            imp = i.get("importance", 5)
            cat = i.get("category", "fact")
            tags = f" [{i['tags']}]" if i.get("tags") else ""
            lines.append(f"  • [{cat}] (imp:{imp}) {i['content']}{tags}  id={i['id']}")
        return ToolResult(success=True, content="\n".join(lines))

    @staticmethod
    async def _artifacts(
        pm: Any, project_id: str, query: str, kind: str | None, limit: int,
    ) -> ToolResult:
        artifacts = await pm.list_artifacts(project_id, kind=kind, limit=limit)
        if not artifacts:
            return ToolResult(success=True, content="No project artifacts found.")
        if query:
            query_lower = query.lower()
            artifacts = [
                a for a in artifacts
                if query_lower in a["title"].lower() or query_lower in a["content"].lower()
            ]
        lines = [f"Project artifacts ({len(artifacts)}):"]
        for a in artifacts:
            status = f" [{a['status']}]" if a["status"] != "active" else ""
            content_preview = a["content"][:200].replace("\n", " ") if a["content"] else ""
            lines.append(
                f"  • [{a['kind']}]{status} {a['title']}  id={a['id']}"
            )
            if content_preview:
                lines.append(f"    {content_preview}")
        return ToolResult(success=True, content="\n".join(lines))

    @staticmethod
    async def _status(pm: Any, project_id: str) -> ToolResult:
        ctx = await pm.get_project_context(project_id)
        if not ctx:
            return ToolResult(success=False, error="Failed to load project context.")

        project = ctx["project"]
        lines = [
            f"Project: {project['name']}",
            f"Status: {project['status']}",
            f"Description: {project['description']}",
            f"Created: {project['created_at']}",
            "",
        ]

        goals = project.get("goals", [])
        if goals:
            lines.append("Goals:")
            status_icons = {"done": "✓", "in_progress": "→", "blocked": "✗", "pending": " "}
            for i, g in enumerate(goals):
                icon = status_icons.get(g.get("status", "pending"), " ")
                lines.append(f"  [{icon}] {g['goal']}")
                if g.get("success_criteria"):
                    lines.append(f"      Criteria: {g['success_criteria']}")
            lines.append("")

        members = ctx["members"]
        if members:
            lines.append("Active members:")
            for m in members:
                tags = ", ".join(m.get("expertise_tags", []))
                tag_str = f" ({tags})" if tags else ""
                lines.append(f"  • {m['agent_name'] or m['agent_id']} — {m['role']}{tag_str}")
            lines.append("")

        blockers = ctx["blockers"]
        if blockers:
            lines.append("Active blockers:")
            for b in blockers:
                lines.append(f"  ✗ {b['title']}: {b['content'][:150]}")
            lines.append("")

        decisions = ctx["decisions"]
        if decisions:
            lines.append(f"Recent decisions ({len(decisions)}):")
            for d in decisions[:5]:
                lines.append(f"  • {d['title']}")
            lines.append("")

        activity = ctx["recent_activity"]
        if activity:
            lines.append("Recent activity:")
            for a in activity[:8]:
                detail = a.get("detail", {})
                detail_str = ""
                if isinstance(detail, dict):
                    detail_str = " — " + ", ".join(
                        f"{k}={v}" for k, v in detail.items()
                        if k not in ("agent_name",) and v
                    )[:100]
                lines.append(f"  {a['created_at'][:16]}  {a['event_type']}{detail_str}")

        return ToolResult(success=True, content="\n".join(lines))

    @staticmethod
    async def _files(pm: Any, project_id: str) -> ToolResult:
        sessions = await pm.project_sessions(project_id)
        if not sessions:
            return ToolResult(success=True, content="No sessions linked to this project.")
        lines = [f"Project sessions ({len(sessions)}):"]
        for s in sessions:
            ended = " (ended)" if s.get("ended_at") else " (active)"
            purpose = f" — {s['purpose']}" if s.get("purpose") else ""
            lines.append(
                f"  • {s['session_name'] or s['session_id'][:8]}{ended}{purpose}"
            )
        return ToolResult(success=True, content="\n".join(lines))

    @staticmethod
    async def _contribute(
        pm: Any,
        project_id: str,
        agent_id: str,
        session_id: str | None,
        kwargs: dict[str, Any],
    ) -> ToolResult:
        contribute_type = str(kwargs.get("contribute_type", "")).strip()
        content = str(kwargs.get("content", "")).strip()
        if not content:
            return ToolResult(success=False, error="'content' is required for contribute.")

        if contribute_type == "insight":
            from captain_claw.insights import get_insights_manager

            mgr = get_insights_manager()
            category = str(kwargs.get("category", "fact")).strip()
            importance = int(kwargs.get("importance") or 5)
            insight_id = await mgr.add(
                content=content,
                category=category,
                importance=importance,
                source_tool="project_memory",
                source_session=session_id,
                project_id=project_id,
            )
            if insight_id:
                await pm.log_activity(
                    project_id, "insight_contributed",
                    agent_id=agent_id,
                    detail={"insight_id": insight_id, "category": category},
                )
                return ToolResult(
                    success=True,
                    content=f"Contributed insight {insight_id} to project.",
                )
            return ToolResult(
                success=True,
                content="Insight was deduped (similar one already exists).",
            )

        if contribute_type == "artifact":
            title = str(kwargs.get("title", "")).strip()
            if not title:
                return ToolResult(
                    success=False, error="'title' is required for artifact contribution.",
                )
            kind = str(kwargs.get("kind", "note")).strip()
            artifact = await pm.add_artifact(
                project_id, kind, title,
                content=content,
                created_by=agent_id,
                session_id=session_id or "",
            )
            return ToolResult(
                success=True,
                content=f"Added {kind} artifact '{title}' to project. id={artifact['id']}",
            )

        return ToolResult(
            success=False,
            error="'contribute_type' must be 'insight' or 'artifact'.",
        )
