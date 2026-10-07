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
            from captain_claw.speaker import principal_for

            # A shared-agent member (None = the owner, unchanged): labels,
            # summaries and keywords are the shared commons; excerpts are
            # transcripts — a member sees only their OWN, and no counts.
            p = principal_for(kwargs.get("_agent"))
            mgr = get_topics_manager()
            n = int(limit) if limit else 15
            if action == "list":
                rows = mgr.list_topics(limit=n)
                return ToolResult(success=True, content=_fmt_overview(
                    rows, "Recent topics", show_counts=p is None))
            if action == "search":
                rows = mgr.search_topics(query or "", limit=n)
                return ToolResult(success=True, content=_fmt_overview(
                    rows, f"Topics matching {query!r}", show_counts=p is None))
            if action == "get":
                if not topic:
                    return ToolResult(success=False, error="'topic' (id or label) is required for get.")
                max_excerpts = n if limit else 40
                if p is None:
                    t = mgr.get_topic(topic, max_excerpts=max_excerpts)
                    if not t:
                        return ToolResult(success=False, error=f"No topic found for {topic!r}.")
                    if any(str(m.get("speaker") or "").strip() for m in t.get("messages", [])):
                        return await _owner_topic_with_members(t, kwargs.get("_agent"))
                    return ToolResult(success=True, content=_fmt_topic(t))
                if not p.speaker_id:
                    # Unverified member: never query speaker='' (the owner's rows).
                    t = mgr.get_topic(topic, max_excerpts=1)
                    if not t:
                        return ToolResult(success=False, error=f"No topic found for {topic!r}.")
                    return ToolResult(success=True, content=_fmt_topic(
                        t, include_messages=False, show_count=False))
                t = mgr.get_topic(topic, max_excerpts=max_excerpts, speaker=p.speaker_id)
                if not t:
                    return ToolResult(success=False, error=f"No topic found for {topic!r}.")
                return ToolResult(success=True, content=_fmt_topic(t, own_messages=True))
            return ToolResult(success=False, error=f"Unknown action: {action}")
        except Exception as e:
            log.error("Topics tool error", action=action, error=str(e))
            return ToolResult(success=False, error=str(e))


async def _owner_topic_with_members(t: dict[str, Any], agent: Any) -> ToolResult:
    """PR D (J19): the owner's ``get`` of a topic holding members' excerpts.

    A current member's excerpts from since their current ``shared_at`` are
    shown and make this a turn that read members' private text (header +
    level); ex-members' — and every member's when Flight Deck can't say who
    the members are — are dropped. The owner's own excerpts always show.
    A member excerpt shows only when its message is one ``read_conversation``
    would show (``shared_usage.shown_message_ids``): never a reply grounded in
    their Google or private deep memory, never the agent's own follow-up
    prompts, never narration, never one whose message can't be found."""
    from captain_claw import member_privacy, shared_usage

    scope, ok = await shared_usage.member_scope(agent)
    candidates: list[tuple[dict[str, Any], Any]] = []
    for m in t.get("messages", []):
        spk = str(m.get("speaker") or "").strip()
        if not spk:
            candidates.append((m, None))
            continue
        member = scope.get(spk) if ok else None
        since = shared_usage.since_of(member) if member is not None else None
        ts = shared_usage._parse_ts(m.get("ts"))
        if since is None or ts is None or ts < since:
            continue
        if m.get("role") not in ("user", "agent") or not str(m.get("msg_id") or ""):
            continue
        candidates.append((m, member))
    shown_ids: set[str] = set()
    members = {mb.user_id: mb for _m, mb in candidates if mb is not None}
    if members:
        shown_ids = await shared_usage.shown_message_ids(list(members.values()))
    kept: list[dict[str, Any]] = []
    members_shown = 0
    for m, member in candidates:
        if member is not None:
            if str(m.get("msg_id") or "") not in shown_ids:
                continue
            members_shown += 1
        kept.append(m)
    text = _fmt_topic({**t, "messages": kept})
    if members_shown:
        member_privacy.mark_private_read(agent, member_privacy.LEVEL_CONTENT)
        text = member_privacy.PRIVATE_HEADER + "\n" + text
    return ToolResult(success=True, content=text)


def _fmt_overview(rows: list[dict[str, Any]], header: str, *, show_counts: bool = True) -> str:
    if not rows:
        return f"{header}: (none yet)"
    lines = [f"{header} ({len(rows)}):"]
    for r in rows:
        kw = (r.get("keywords") or "").replace(",", ", ")
        lines.append(
            f"- [{r['id']}] {r['label']} — {r.get('summary', '')[:160]}"
            + (f"  · {r.get('msg_count', 0)} msgs" if show_counts and r.get("msg_count") else "")
            + (f"  · tags: {kw}" if kw else "")
        )
    lines.append("\nUse action='get' with the [id] in brackets to see a topic's messages.")
    return "\n".join(lines)


def _fmt_topic(t: dict[str, Any], include_messages: bool = True, *,
               show_count: bool = True, own_messages: bool = False) -> str:
    lines = [
        f"Topic: {t['label']}  [{t['id']}]",
        f"Summary: {t.get('summary', '') or '(none)'}",
    ]
    if t.get("keywords"):
        lines.append(f"Tags: {t['keywords'].replace(',', ', ')}")
    if not include_messages:
        if show_count:
            lines.append(f"Messages: {t.get('msg_count', 0)} total (excerpts are private).")
        else:
            lines.append("Messages: excerpts are private.")
        return "\n".join(lines)
    if own_messages:
        # A shared-agent member: only their own excerpts, with their own count.
        lines.append(f"Your messages in this topic ({len(t.get('messages', []))}):")
    else:
        lines.append(f"Messages ({t.get('msg_count', 0)} total, showing recent):")
    for m in t.get("messages", []):
        ts = str(m.get("ts", ""))[:16].replace("T", " ")
        lines.append(f"  [{ts}] ({m.get('role', '')}) {m.get('excerpt', '')[:280]}")
    return "\n".join(lines)
