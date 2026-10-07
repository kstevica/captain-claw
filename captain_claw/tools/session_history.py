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
        agent = kwargs.get("_agent") or getattr(self, "_agent", None)
        try:
            n = int(limit) if limit else 10
            if action == "search":
                if not (query or "").strip():
                    return ToolResult(success=False, error="'query' is required for search.")
                results = memory.search_history(query, max_results=n)
                snaps = {str(getattr(r, "reference", "")): memory.get_history(
                    str(getattr(r, "reference", ""))) for r in results}
                keep, private = await _member_filter(
                    agent, {ref: s for ref, s in snaps.items() if s})
                results = [r for r in results if str(getattr(r, "reference", "")) in keep]
                return ToolResult(success=True, content=_private_prefix(
                    agent, private, _fmt_results(results, query or "")))
            if action == "list":
                rows = memory.list_history(limit=n)
                keep, private = await _member_filter(
                    agent, {str(r.get("history_id")): r for r in rows})
                rows = [r for r in rows if str(r.get("history_id")) in keep]
                return ToolResult(success=True, content=_private_prefix(agent, private, _fmt_list(rows)))
            if action == "get":
                if not (history_id or "").strip():
                    return ToolResult(success=False, error="'history_id' is required for get.")
                snap = memory.get_history(history_id)
                if snap:
                    keep, private = await _member_filter(agent, {str(snap.get("history_id")): snap})
                    if str(snap.get("history_id")) not in keep:
                        snap = None
                if not snap:
                    return ToolResult(success=False, error=f"No snapshot found for {history_id!r}.")
                return ToolResult(success=True, content=_private_prefix(
                    agent, private, _fmt_snapshot(snap)))
            return ToolResult(success=False, error=f"Unknown action: {action}")
        except Exception as e:
            log.error("History tool error", action=action, error=str(e))
            return ToolResult(success=False, error=str(e))


async def _member_filter(agent: Any, snaps: dict[str, dict[str, Any]]) -> tuple[set[str], bool]:
    """PR D (J19): which snapshots (history_id → row with ``session_id`` and
    ``created_at``) the owner's ``history`` may return, and whether any of
    them is a current member's (the turn then reads members' private text).

    The owner's own snapshots are kept unchanged; a member's only while they
    are a current member and it was archived since their current
    ``shared_at``. Any doubt — Flight Deck unreachable, an unreadable session
    store — drops member snapshots (every snapshot when the store fails)."""
    from captain_claw import shared_usage
    from captain_claw.speaker import principal_for

    if principal_for(agent) is not None:
        return set(snaps), False            # members never have this tool
    try:
        speakers = await shared_usage.speakers_of(
            {str(s.get("session_id") or "") for s in snaps.values()})
    except Exception:
        log.warning("History snapshots hidden: session owners unknown")
        return set(), False
    keep: set[str] = set()
    member_snaps: dict[str, str] = {}
    for ref, snap in snaps.items():
        spk = speakers.get(str(snap.get("session_id") or ""), "")
        if spk:
            member_snaps[ref] = spk
        else:
            keep.add(ref)
    if not member_snaps:
        return keep, False
    scope, ok = await shared_usage.member_scope(agent)
    private = False
    if ok:
        # A snapshot is the flattened text of the messages it froze, without
        # their times: one taken after a re-share (leave → re-add, owner
        # change) from a session that began before it may hold messages from
        # the earlier membership. Only sessions begun since the current
        # shared_at qualify (J16, fail closed).
        try:
            began = await shared_usage.sessions_created_at(
                {str(snaps[ref].get("session_id") or "") for ref in member_snaps})
        except Exception:
            log.warning("History snapshots hidden: session start unknown")
            began = {}
        for ref, spk in member_snaps.items():
            m = scope.get(spk)
            since = shared_usage.since_of(m) if m is not None else None
            created = shared_usage._parse_ts(snaps[ref].get("created_at"))
            started = shared_usage._parse_ts(began.get(str(snaps[ref].get("session_id") or "")))
            if (since is not None and created is not None and created >= since
                    and started is not None and started >= since):
                keep.add(ref)
                private = True
    return keep, private


def _private_prefix(agent: Any, private: bool, text: str) -> str:
    if not private:
        return text
    from captain_claw import member_privacy

    member_privacy.mark_private_read(agent, member_privacy.LEVEL_CONTENT)
    return member_privacy.PRIVATE_HEADER + "\n" + text


def _fmt_results(results: list[Any], query: str) -> str:
    if not results:
        return (
            f"Transcript archive — no relevant verbatim matches for {query!r}. "
            "Nothing from earlier sessions matches this; answer from the current "
            "conversation and do not infer from unrelated past sessions."
        )
    lines = [
        f"Transcript archive matches for {query!r} ({len(results)}):",
        "(These are frozen snippets from PAST sessions — possibly unrelated to the "
        "current conversation. Confirm a snippet is actually about what the user asked "
        "before using it, and never present an old session's state as the current one.)",
    ]
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
