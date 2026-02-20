"""Session history export and rendering utilities."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from captain_claw.cron import now_utc, to_utc_iso


def normalize_session_id(raw: str) -> str:
    safe = "".join(c if c.isalnum() or c in "._-" else "-" for c in (raw or "").strip())
    safe = safe.strip("-")
    return safe or "default"


def truncate_history_text(text: str, max_chars: int = 8000) -> str:
    cleaned = str(text or "")
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[:max_chars].rstrip() + "... [truncated]"


def render_chat_export_markdown(
    session_id: str,
    session_name: str,
    messages: list[dict[str, object]],
) -> str:
    chat_messages = [msg for msg in messages if str(msg.get("role", "")).lower() in {"user", "assistant", "system"}]
    lines = [
        "# Session Chat Export",
        f"- Exported at (UTC): {to_utc_iso(now_utc())}",
        f"- Session ID: {session_id}",
        f"- Session name: {session_name}",
        f"- Messages: {len(chat_messages)}",
        "",
    ]
    if not chat_messages:
        lines.append("(no chat messages found)")
        lines.append("")
        return "\n".join(lines)

    for idx, msg in enumerate(chat_messages, start=1):
        role = str(msg.get("role", "unknown")).strip() or "unknown"
        timestamp = str(msg.get("timestamp", "")).strip()
        content = str(msg.get("content", ""))
        lines.append(f"## {idx}. role={role} timestamp={timestamp or '-'}")
        lines.append(content if content else "(empty)")
        lines.append("")
    return "\n".join(lines)


def render_monitor_export_markdown(
    session_id: str,
    session_name: str,
    messages: list[dict[str, object]],
) -> str:
    monitor_messages = [msg for msg in messages if str(msg.get("role", "")).lower() == "tool"]
    lines = [
        "# Session Monitor Export",
        f"- Exported at (UTC): {to_utc_iso(now_utc())}",
        f"- Session ID: {session_id}",
        f"- Session name: {session_name}",
        f"- Monitor entries: {len(monitor_messages)}",
        "",
    ]
    if not monitor_messages:
        lines.append("(no monitor/tool messages found)")
        lines.append("")
        return "\n".join(lines)

    for idx, msg in enumerate(monitor_messages, start=1):
        tool_name = str(msg.get("tool_name") or "tool")
        timestamp = str(msg.get("timestamp", "")).strip()
        args = msg.get("tool_arguments")
        if isinstance(args, dict):
            try:
                args_text = json.dumps(args, ensure_ascii=True, sort_keys=True)
            except Exception:
                args_text = str(args)
        else:
            args_text = "{}"
        content = str(msg.get("content", ""))
        lines.append(f"## {idx}. tool={tool_name} timestamp={timestamp or '-'}")
        lines.append(f"args={args_text}")
        lines.append(content if content else "(empty)")
        lines.append("")
    return "\n".join(lines)


def collect_pipeline_trace_entries(
    session_id: str,
    session_name: str,
    messages: list[dict[str, object]],
) -> list[dict[str, object]]:
    pipeline_messages = [
        msg
        for msg in messages
        if str(msg.get("role", "")).lower() == "tool"
        and str(msg.get("tool_name", "")).strip().lower() == "pipeline_trace"
    ]

    entries: list[dict[str, object]] = []
    for idx, msg in enumerate(pipeline_messages, start=1):
        args = msg.get("tool_arguments")
        payload = dict(args) if isinstance(args, dict) else {}
        payload["seq"] = idx
        payload["timestamp"] = str(msg.get("timestamp", "")).strip()
        payload["session_id"] = session_id
        payload["session_name"] = session_name
        entries.append(payload)

    if not entries:
        fallback_sources = {"planning", "task_contract", "completion_gate"}
        for idx, msg in enumerate(messages, start=1):
            if str(msg.get("role", "")).lower() != "tool":
                continue
            source = str(msg.get("tool_name", "")).strip().lower()
            if source not in fallback_sources:
                continue
            args = msg.get("tool_arguments")
            payload = dict(args) if isinstance(args, dict) else {}
            payload["source"] = source
            payload["seq"] = idx
            payload["timestamp"] = str(msg.get("timestamp", "")).strip()
            payload["session_id"] = session_id
            payload["session_name"] = session_name
            entries.append(payload)

    return entries


def render_pipeline_export_jsonl(
    session_id: str,
    session_name: str,
    messages: list[dict[str, object]],
) -> str:
    entries = collect_pipeline_trace_entries(session_id, session_name, messages)
    return "\n".join(json.dumps(item, ensure_ascii=True, sort_keys=True) for item in entries)


def render_pipeline_summary_markdown(
    session_id: str,
    session_name: str,
    messages: list[dict[str, object]],
) -> str:
    entries = collect_pipeline_trace_entries(session_id, session_name, messages)
    lines = [
        "# Session Pipeline Trace Summary",
        f"- Exported at (UTC): {to_utc_iso(now_utc())}",
        f"- Session ID: {session_id}",
        f"- Session name: {session_name}",
        f"- Trace entries: {len(entries)}",
        "",
    ]
    if not entries:
        lines.append("(no pipeline trace entries found)")
        lines.append("")
        return "\n".join(lines)

    by_source: dict[str, int] = {}
    by_step: dict[str, int] = {}
    by_event: dict[str, int] = {}
    first_ts = str(entries[0].get("timestamp", "")).strip()
    last_ts = str(entries[-1].get("timestamp", "")).strip()
    for entry in entries:
        source = str(entry.get("source", "")).strip() or "unknown"
        by_source[source] = by_source.get(source, 0) + 1
        step = str(entry.get("step", "")).strip()
        if step:
            by_step[step] = by_step.get(step, 0) + 1
        event = str(entry.get("event", "")).strip()
        if event:
            by_event[event] = by_event.get(event, 0) + 1

    lines.append(f"- First entry timestamp: {first_ts or '-'}")
    lines.append(f"- Last entry timestamp: {last_ts or '-'}")
    lines.append("")
    lines.append("## Sources")
    for source, count in sorted(by_source.items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"- {source}: {count}")
    lines.append("")
    if by_event:
        lines.append("## Planning Events")
        for event, count in sorted(by_event.items(), key=lambda kv: (-kv[1], kv[0])):
            lines.append(f"- {event}: {count}")
        lines.append("")
    if by_step:
        lines.append("## Completion/Contract Steps")
        for step, count in sorted(by_step.items(), key=lambda kv: (-kv[1], kv[0])):
            lines.append(f"- {step}: {count}")
        lines.append("")

    lines.append("## Timeline")
    for entry in entries:
        seq = int(entry.get("seq", 0))
        timestamp = str(entry.get("timestamp", "")).strip() or "-"
        source = str(entry.get("source", "")).strip() or "unknown"
        item = f"{seq}. [{timestamp}] source={source}"
        event = str(entry.get("event", "")).strip()
        step = str(entry.get("step", "")).strip()
        if event:
            item += f" event={event}"
        if step:
            item += f" step={step}"
        leaf_index = entry.get("leaf_index")
        leaf_tasks = entry.get("leaf_tasks")
        leaf_remaining = entry.get("leaf_remaining")
        if isinstance(leaf_index, int) and isinstance(leaf_tasks, int):
            item += f" progress={leaf_index}/{leaf_tasks}"
        if isinstance(leaf_remaining, int):
            item += f" remaining={leaf_remaining}"
        current_path = str(entry.get("current_path", "")).strip()
        if current_path:
            item += f" path={current_path}"
        eta_text = str(entry.get("eta_text", "")).strip()
        if eta_text:
            item += f" eta={eta_text}"
        lines.append(item)
    lines.append("")
    return "\n".join(lines)


def export_session_history(
    mode: str,
    session_id: str,
    session_name: str,
    messages: list[dict[str, object]],
    saved_base_path: Path,
) -> list[Path]:
    """Export session history to files. Returns list of written paths."""
    mode_key = (mode or "all").strip().lower()
    if mode_key not in {"chat", "monitor", "pipeline", "pipeline-summary", "all"}:
        mode_key = "all"

    safe_session = normalize_session_id(session_id)
    snapshot: list[dict[str, object]] = [dict(msg) for msg in messages]

    export_root = (saved_base_path / "showcase" / safe_session / "exports").resolve()
    export_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    written: list[Path] = []

    if mode_key in {"chat", "all"}:
        chat_path = export_root / f"chat-{stamp}.md"
        chat_path.write_text(
            render_chat_export_markdown(session_id, session_name, snapshot) + "\n",
            encoding="utf-8",
        )
        written.append(chat_path)

    if mode_key in {"monitor", "all"}:
        monitor_path = export_root / f"monitor-{stamp}.md"
        monitor_path.write_text(
            render_monitor_export_markdown(session_id, session_name, snapshot) + "\n",
            encoding="utf-8",
        )
        written.append(monitor_path)

    if mode_key in {"pipeline", "all"}:
        pipeline_path = export_root / f"pipeline-{stamp}.jsonl"
        pipeline_path.write_text(
            render_pipeline_export_jsonl(session_id, session_name, snapshot) + "\n",
            encoding="utf-8",
        )
        written.append(pipeline_path)
    if mode_key in {"pipeline-summary", "all"}:
        pipeline_summary_path = export_root / f"pipeline-summary-{stamp}.md"
        pipeline_summary_path.write_text(
            render_pipeline_summary_markdown(session_id, session_name, snapshot) + "\n",
            encoding="utf-8",
        )
        written.append(pipeline_summary_path)

    return written
