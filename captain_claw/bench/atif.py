"""Captain's transcript as an ATIF trajectory.

ATIF (Agent Trajectory Interchange Format) is what Harbor and the benchmarks
built on it read an agent's run from — WildClawBench's graders, for one, look
for ``/logs/agent/trajectory.json`` and judge the agent's replies there. Without
it they see no reply at all and score "warned the user" or "refused" as 0.

One step per message: the task (and any system message) as ``user`` /
``system`` steps, each reply as an ``agent`` step with its tool calls, and each
tool result in the observation of the agent step that asked for it. Drafts the
harness rejected and the correctives it added (``origin`` rejected /
corrective) never reached the person, so they are left out.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

SCHEMA_VERSION = "ATIF-v1.8"
_HIDDEN_ORIGINS = frozenset({"rejected", "corrective"})


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    return json.dumps(value, ensure_ascii=False, default=str)


def _arguments(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw or "{}")
    except (TypeError, ValueError):
        return {"raw": str(raw)}
    return parsed if isinstance(parsed, dict) else {"value": parsed}


def _timestamp(raw: Any) -> str | None:
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    return text


def to_atif(
    messages: list[dict[str, Any]],
    *,
    session_id: str | None,
    model: str,
    version: str,
    usage: dict[str, int] | None = None,
    cost_usd: float | None = None,
) -> dict[str, Any] | None:
    """The run as an ATIF trajectory, or None when there is no step to tell."""
    steps: list[dict[str, Any]] = []
    last_agent: dict[str, Any] | None = None
    for msg in messages:
        if not isinstance(msg, dict) or msg.get("origin") in _HIDDEN_ORIGINS:
            continue
        role = msg.get("role")
        step: dict[str, Any] = {}
        stamp = _timestamp(msg.get("timestamp"))
        if stamp:
            step["timestamp"] = stamp
        if role in ("user", "system"):
            steps.append({**step, "source": role, "message": _text(msg.get("content"))})
            last_agent = None
        elif role == "assistant":
            step.update({"source": "agent", "model_name": model, "message": _text(msg.get("content"))})
            calls = []
            for call in msg.get("tool_calls") or []:
                fn = call.get("function") or {}
                calls.append({
                    "tool_call_id": str(call.get("id") or ""),
                    "function_name": str(fn.get("name") or call.get("name") or ""),
                    "arguments": _arguments(fn.get("arguments", call.get("arguments"))),
                })
            if calls:
                step["tool_calls"] = calls
            steps.append(step)
            last_agent = step
        elif role == "tool" and last_agent is not None:
            last_agent.setdefault("observation", {"results": []})["results"].append(
                {"source_call_id": msg.get("tool_call_id"), "content": _text(msg.get("content"))}
            )
    if not steps:
        return None
    trajectory: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "session_id": session_id,
        "agent": {"name": "captain", "version": version, "model_name": model},
        "steps": [{"step_id": i, **s} for i, s in enumerate(steps, 1)],
    }
    if usage or cost_usd is not None:
        usage = usage or {}
        trajectory["final_metrics"] = {
            "total_prompt_tokens": int(usage.get("prompt_tokens", 0)),
            "total_completion_tokens": int(usage.get("completion_tokens", 0)),
            "total_cached_tokens": int(usage.get("cache_read_input_tokens", 0)),
            "total_cost_usd": cost_usd,
            "total_steps": len(steps),
        }
    return trajectory
