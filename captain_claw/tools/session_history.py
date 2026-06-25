"""History tool — recall the frozen, verbatim transcript archive.

Before a session is compacted, the raw messages that are about to be dropped
are frozen into append-only ``session_history`` memory (see
``SemanticMemoryIndex.archive_session_history``). The compaction *summary*
keeps the gist but loses specifics — exact models, names, lists, numbers. This
tool lets the agent search that verbatim archive on demand, so "go through your
memory" actually reaches the original words instead of only the summary.

Relevant snapshots also surface passively in the semantic-memory context note;
this tool is the deliberate, explicit recall path (and shows up as a real tool
call for observability).
"""

from typing import Any

from captain_claw.logging import get_logger
from captain_claw.tools.registry import Tool, ToolResult

log = get_logger(__name__)


class SessionHistoryTool(Tool):
    """Search and read the frozen verbatim transcript archive."""

    name = "history"
    description = (
        "Search the verbatim TRANSCRIPT ARCHIVE — raw past messages frozen "
        "before they were compacted away. Compaction summaries keep the gist "
        "but drop specifics (exact model names, lists, numbers, who-said-what); "
        "this reaches the original words. Use it when the user asks you to "
        "recall something specific from earlier ('what did I say the motor "
        "model was', 'the places we listed') and it isn't in current context. "
        "Actions: 'search' (find snapshots by keyword), 'list' (recent "
        "snapshots), 'get' (one snapshot's full verbatim text by id)."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["search", "list", "get"],
                "description": "search = find by keyword; list = recent snapshots; get = one snapshot in full.",
            },
            "query": {"type": "string", "description": "Keyword(s) for 'search'."},
            "history_id": {"type": "string", "description": "Snapshot id for 'get'."},
            "limit": {"type": "integer", "description": "Max results for search/list (default 10)."},
        },
        "required": ["action"],
    }

    async def execute(
        self,
        action: str,
        query: str | None = None,
        history_id: str | None = None,
        limit: int | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        memory = getattr(getattr(self, "_agent", None), "memory", None)
        if memory is None:
            return ToolResult(success=False, error="History memory is not available in this session.")
        try:
            n = int(limit) if limit else 10
            if action == "search":
                if not (query or "").strip():
                    return ToolResult(success=False, error="'query' is required for search.")
                results = memory.search_history(query, max_results=n)
                return ToolResult(success=True, content=_fmt_results(results, query or ""))
            if action == "list":
                rows = memory.list_history(limit=n)
                return ToolResult(success=True, content=_fmt_list(rows))
            if action == "get":
                if not (history_id or "").strip():
                    return ToolResult(success=False, error="'history_id' is required for get.")
                snap = memory.get_history(history_id)
                if not snap:
                    return ToolResult(success=False, error=f"No snapshot found for {history_id!r}.")
                return ToolResult(success=True, content=_fmt_snapshot(snap))
            return ToolResult(success=False, error=f"Unknown action: {action}")
        except Exception as e:
            log.error("History tool error", action=action, error=str(e))
            return ToolResult(success=False, error=str(e))


def _fmt_results(results: list[Any], query: str) -> str:
    if not results:
        return f"Transcript archive — no verbatim matches for {query!r}."
    lines = [f"Transcript archive matches for {query!r} ({len(results)}):"]
    for r in results:
        snippet = " ".join(str(getattr(r, "snippet", "")).split())[:400]
        created = (getattr(r, "updated_at", "") or "").strip() or "unknown"
        lines.append(
            f"- [{getattr(r, 'reference', '')}] (created={created}, score={getattr(r, 'score', 0.0):.3f})\n  {snippet}"
        )
    lines.append("\nUse action='get' with a [id] to read that snapshot's full verbatim text.")
    return "\n".join(lines)


def _fmt_list(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "Transcript archive: (empty — nothing has been compacted yet)."
    lines = [f"Recent transcript snapshots ({len(rows)}):"]
    for r in rows:
        lines.append(
            f"- [{r['history_id']}] {r.get('session_name', '')} "
            f"· {r.get('message_count', 0)} msgs · {r.get('created_at', '')}\n"
            f"  {r.get('preview', '')}"
        )
    lines.append("\nUse action='get' with a [id] to read a snapshot in full, or 'search' to find by keyword.")
    return "\n".join(lines)


def _fmt_snapshot(snap: dict[str, Any]) -> str:
    return (
        f"Snapshot [{snap['history_id']}] — {snap.get('session_name', '')}  "
        f"({snap.get('message_count', 0)} msgs, frozen {snap.get('created_at', '')})\n\n"
        f"{snap.get('text', '')}"
    )
