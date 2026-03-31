"""Tool for consulting peer agents via Flight Deck.

Allows an agent to send a message to another agent running in the
same Flight Deck environment and receive the response, enabling
direct agent-to-agent collaboration.
"""

from __future__ import annotations

import json
from typing import Any

import structlog

from captain_claw.tools.registry import Tool, ToolResult

log = structlog.get_logger(__name__)


class ConsultPeerTool(Tool):
    name = "consult_peer"
    description = (
        "Consult another agent available in the Flight Deck environment. "
        "Send a message to a peer agent and receive their response. "
        "Use this when a task would benefit from another agent's expertise."
    )
    timeout_seconds = 600.0

    parameters = {
        "type": "object",
        "properties": {
            "agent_name": {
                "type": "string",
                "description": (
                    "Name of the peer agent to consult. Must match one of "
                    "the agents listed in the 'Other Available Agents' section."
                ),
            },
            "message": {
                "type": "string",
                "description": (
                    "The message or task to send to the peer agent. Be specific "
                    "about what you need — include relevant context so the peer "
                    "can provide a useful response."
                ),
            },
        },
        "required": ["agent_name", "message"],
    }

    async def execute(self, agent_name: str, message: str, **kwargs: Any) -> ToolResult:
        try:
            import httpx
        except ImportError:
            return ToolResult(success=False, error="httpx is required for peer consultation")

        # Look up peer info from session metadata, with agent-level fallback
        session = kwargs.get("_session")
        agent = kwargs.get("_agent")
        metadata = getattr(session, "metadata", {}) or {} if session else {}
        peers = metadata.get("peer_agents", [])
        fd_url = metadata.get("fd_url", "")

        # Fallback: check agent-level attributes (set by ws_handler even
        # when session wasn't ready at welcome time)
        if not peers and agent:
            peers = getattr(agent, "_peer_agents", []) or []
        if not fd_url and agent:
            fd_url = getattr(agent, "_fd_url", "") or ""

        if not fd_url:
            return ToolResult(
                success=False,
                error="Flight Deck URL not available. Peer consultation requires Flight Deck.",
            )

        if not peers:
            return ToolResult(
                success=False,
                error="No peer agents available. Other agents must be running in Flight Deck.",
            )

        # Find the matching peer (exact, then case-insensitive substring)
        target = None
        _query = agent_name.lower()
        for p in peers:
            if not isinstance(p, dict):
                continue
            if p.get("name", "").lower() == _query:
                target = p
                break
        if not target:
            for p in peers:
                if not isinstance(p, dict):
                    continue
                _pname = p.get("name", "").lower()
                if _query in _pname or _pname in _query:
                    target = p
                    break

        if not target:
            available = ", ".join(p.get("name", "?") for p in peers if isinstance(p, dict))
            return ToolResult(
                success=False,
                error=f"Agent '{agent_name}' not found. Available agents: {available}",
            )

        host = target.get("host", "localhost")
        port = target.get("port")
        auth = target.get("auth", "")
        require_approval = target.get("requireApproval", False)

        if not port:
            return ToolResult(
                success=False,
                error=f"No connection info for agent '{agent_name}'.",
            )

        # Get source agent name from session
        source_name = metadata.get("session_display_name", "another agent")

        # Human-in-the-loop approval if configured for this target agent
        if require_approval:
            approval_cb = kwargs.get("_peer_consult_approval_callback")
            if callable(approval_cb):
                import asyncio as _asyncio
                question = (
                    f"Agent wants to consult peer \"{agent_name}\"\n\n"
                    f"Message: {message[:300]}{'...' if len(message) > 300 else ''}"
                )
                result = approval_cb(question)
                if _asyncio.iscoroutine(result):
                    approved = await result
                else:
                    approved = bool(result)
                if not approved:
                    return ToolResult(
                        success=False,
                        error=f"Consultation with '{agent_name}' was denied by the user.",
                    )
            else:
                log.warning(
                    "Peer consultation requires approval but no callback available; proceeding",
                    target=agent_name,
                )

        log.info(
            "Consulting peer agent",
            target=agent_name,
            host=host,
            port=port,
            message_len=len(message),
        )

        # Resolve the peer's display name for UI
        peer_display = target.get("name", agent_name)

        # Get broadcast callback so we can send peer_activity to the UI
        broadcast = getattr(agent, "ws_broadcast", None) if agent else None

        def _emit(activity_type: str, detail: str = "") -> None:
            """Broadcast a peer_activity event to connected WS clients."""
            if not callable(broadcast):
                return
            broadcast({
                "type": "peer_activity",
                "peer_name": peer_display,
                "activity_type": activity_type,
                "detail": detail,
            })

        _emit("connecting", f"Connecting to {peer_display}…")

        try:
            async with httpx.AsyncClient(timeout=600.0) as client:
                async with client.stream(
                    "POST",
                    f"{fd_url}/fd/consult-peer",
                    json={
                        "host": host,
                        "port": port,
                        "auth": auth,
                        "message": message,
                        "source_name": source_name,
                        "timeout": 480.0,
                    },
                ) as resp:
                    if resp.status_code != 200:
                        return ToolResult(
                            success=False,
                            error=f"Flight Deck returned HTTP {resp.status_code}",
                        )

                    final_response = ""
                    async for line in resp.aiter_lines():
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            event = json.loads(line)
                        except json.JSONDecodeError:
                            continue

                        # Final result line
                        if "done" in event and event.get("ok"):
                            final_response = event.get("response", "(no response)")
                            _emit("done", "Consultation complete")
                            break
                        if event.get("ok") is False:
                            _emit("error", event.get("error", "Failed"))
                            return ToolResult(
                                success=False,
                                error=event.get("error", "Peer consultation failed"),
                            )

                        # Intermediate event — broadcast as peer activity
                        evt_type = event.get("event", "")
                        data = event.get("data", {})
                        if evt_type == "status":
                            _emit("status", data.get("status", ""))
                        elif evt_type == "thinking":
                            tool = data.get("tool", "")
                            text = data.get("text", "")
                            detail = f"Using {tool}" if tool else text
                            _emit("thinking", detail[:200])
                        elif evt_type == "monitor":
                            tool_name = data.get("tool_name", "")
                            _emit("tool", tool_name)

                    return ToolResult(
                        success=True,
                        content=f"Response from {peer_display}:\n\n{final_response}",
                    )

        except httpx.TimeoutException:
            _emit("error", "Timed out")
            return ToolResult(
                success=False,
                error=f"Timed out waiting for response from '{agent_name}'.",
            )
        except httpx.ConnectError as e:
            log.error("Cannot connect to Flight Deck", fd_url=fd_url, error=str(e))
            _emit("error", "Connection failed")
            return ToolResult(
                success=False,
                error=f"Cannot connect to Flight Deck at {fd_url}.",
            )
        except Exception as e:
            log.error("Peer consultation failed", error=str(e))
            _emit("error", str(e)[:100])
            return ToolResult(success=False, error=str(e))
