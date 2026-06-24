"""Topics tool — recall the conversation's auto-clustered topic memory.

The agent uses this to pull a whole thread's context at once ("the Munich trip",
"the Vesna VC deal") instead of re-deriving it from a long transcript. Topics are
built automatically by the periodic classifier in conversation_topics.py.
"""

from typing import Any

from captain_claw.conversation_topics import get_topics_manager
from captain_claw.logging import get_logger
from captain_claw.tools.registry import Tool, ToolResult

log = get_logger(__name__)


class TopicsTool(Tool):
    """Browse and recall auto-clustered conversation topics."""

    name = "topics"
    description = (
        "Recall the conversation's automatically-maintained TOPICS — durable "
        "clusters of past comms (the Munich trip, the Vesna VC deal, the weekly "
        "brief…), each with a summary and recent message excerpts. Use it to pull "
        "the full context of a thread the user returns to, instead of scrolling "
        "history. Actions: 'list' (recent topics overview), 'search' (find topics "
        "by keyword), 'get' (one topic's summary + message excerpts by id or label)."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["list", "search", "get"],
                "description": "list = recent topics; search = find by keyword; get = one topic in full.",
            },
            "query": {"type": "string", "description": "Keyword(s) for 'search'."},
            "topic": {"type": "string", "description": "Topic id or label for 'get'."},
            "limit": {"type": "integer", "description": "Max results for list/search (default 15)."},
        },
        "required": ["action"],
    }

    async def execute(
        self,
        action: str,
        query: str | None = None,
        topic: str | None = None,
        limit: int | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        try:
            mgr = get_topics_manager()
            n = int(limit) if limit else 15
            if action == "list":
                rows = mgr.list_topics(limit=n)
                return ToolResult(success=True, content=_fmt_overview(rows, "Recent topics"))
            if action == "search":
                rows = mgr.search_topics(query or "", limit=n)
                return ToolResult(success=True, content=_fmt_overview(rows, f"Topics matching {query!r}"))
            if action == "get":
                if not topic:
                    return ToolResult(success=False, error="'topic' (id or label) is required for get.")
                t = mgr.get_topic(topic, max_excerpts=n if limit else 40)
                if not t:
                    return ToolResult(success=False, error=f"No topic found for {topic!r}.")
                return ToolResult(success=True, content=_fmt_topic(t))
            return ToolResult(success=False, error=f"Unknown action: {action}")
        except Exception as e:
            log.error("Topics tool error", action=action, error=str(e))
            return ToolResult(success=False, error=str(e))


def _fmt_overview(rows: list[dict[str, Any]], header: str) -> str:
    if not rows:
        return f"{header}: (none yet)"
    lines = [f"{header} ({len(rows)}):"]
    for r in rows:
        kw = (r.get("keywords") or "").replace(",", ", ")
        lines.append(
            f"- [{r['id']}] {r['label']} — {r.get('summary', '')[:160]}"
            + (f"  · {r.get('msg_count', 0)} msgs" if r.get("msg_count") else "")
            + (f"  · tags: {kw}" if kw else "")
        )
    lines.append("\nUse action='get' with the [id] in brackets to see a topic's messages.")
    return "\n".join(lines)


def _fmt_topic(t: dict[str, Any]) -> str:
    lines = [
        f"Topic: {t['label']}  [{t['id']}]",
        f"Summary: {t.get('summary', '') or '(none)'}",
    ]
    if t.get("keywords"):
        lines.append(f"Tags: {t['keywords'].replace(',', ', ')}")
    lines.append(f"Messages ({t.get('msg_count', 0)} total, showing recent):")
    for m in t.get("messages", []):
        ts = str(m.get("ts", ""))[:16].replace("T", " ")
        lines.append(f"  [{ts}] ({m.get('role', '')}) {m.get('excerpt', '')[:280]}")
    return "\n".join(lines)
