"""Tool for discovering and consulting peer agents via Flight Deck fleet API.

Provides live peer discovery by querying the Flight Deck /fd/fleet endpoint,
replacing the static peer list pushed at WebSocket connect time.
"""

from __future__ import annotations

import json
from typing import Any

import structlog

from captain_claw.tools.registry import Tool, ToolResult

log = structlog.get_logger(__name__)


class FlightDeckTool(Tool):
    name = "flight_deck"
    description = (
        "Discover and communicate with peer agents in the Flight Deck environment. "
        "Use 'list_agents' to get a live list of all running agents, or 'consult' "
        "to send a message to a specific peer agent and receive their response."
    )
    timeout_seconds = 600.0

    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["list_agents", "consult"],
                "description": (
                    "'list_agents' — get all agents currently in the fleet. "
                    "'consult' — send a message to a peer agent."
                ),
            },
            "agent_name": {
                "type": "string",
                "description": (
                    "Name of the peer agent to consult (required for 'consult' action). "
                    "Use list_agents first to see available agents."
                ),
            },
            "message": {
                "type": "string",
                "description": (
                    "The message to send to the peer agent (required for 'consult' action)."
                ),
            },
        },
        "required": ["action"],
    }

    def _get_fd_url(self, **kwargs: Any) -> str:
        """Resolve the Flight Deck URL from session metadata or agent attributes."""
        session = kwargs.get("_session")
        agent = kwargs.get("_agent")
        metadata = getattr(session, "metadata", {}) or {} if session else {}
        fd_url = metadata.get("fd_url", "")
        if not fd_url and agent:
            fd_url = getattr(agent, "_fd_url", "") or ""
        return fd_url

    async def _list_agents(self, fd_url: str, **kwargs: Any) -> ToolResult:
        """Query /fd/fleet for live agent list."""
        try:
            import httpx
        except ImportError:
            return ToolResult(success=False, error="httpx is required")

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(f"{fd_url}/fd/fleet")
                if resp.status_code != 200:
                    return ToolResult(
                        success=False,
                        error=f"Flight Deck returned HTTP {resp.status_code}",
                    )
                agents = resp.json()
        except Exception as e:
            return ToolResult(success=False, error=f"Cannot reach Flight Deck: {e}")

        if not agents:
            return ToolResult(success=True, content="No agents are currently running in the fleet.")

        # Filter out self if possible
        session = kwargs.get("_session")
        metadata = getattr(session, "metadata", {}) or {} if session else {}
        my_name = metadata.get("session_display_name", "")

        lines = []
        for a in agents:
            name = a.get("name", "?")
            kind = a.get("kind", "?")
            status = a.get("status", "?")
            port = a.get("port", 0)
            desc = a.get("description", "")
            marker = " (you)" if my_name and name.lower() == my_name.lower() else ""
            line = f"- **{name}**{marker} [{kind}] — status: {status}, port: {port}"
            if desc:
                line += f" — {desc}"
            lines.append(line)

        return ToolResult(
            success=True,
            content=f"Fleet agents ({len(agents)}):\n" + "\n".join(lines),
        )

    async def _consult(self, fd_url: str, agent_name: str, message: str, **kwargs: Any) -> ToolResult:
        """Consult a peer agent via /fd/fleet lookup + /fd/consult-peer."""
        try:
            import httpx
        except ImportError:
            return ToolResult(success=False, error="httpx is required")

        # First, get the live fleet to find the target agent
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(f"{fd_url}/fd/fleet")
                if resp.status_code != 200:
                    return ToolResult(success=False, error=f"Fleet query failed: HTTP {resp.status_code}")
                agents = resp.json()
        except Exception as e:
            return ToolResult(success=False, error=f"Cannot reach Flight Deck: {e}")

        # Find matching agent (exact, then case-insensitive substring)
        target = None
        query = agent_name.lower()
        for a in agents:
            if a.get("name", "").lower() == query:
                target = a
                break
        if not target:
            for a in agents:
                aname = a.get("name", "").lower()
                if query in aname or aname in query:
                    target = a
                    break

        if not target:
            available = ", ".join(a.get("name", "?") for a in agents)
            return ToolResult(
                success=False,
                error=f"Agent '{agent_name}' not found in fleet. Available: {available}",
            )

        if target.get("status") != "running":
            return ToolResult(
                success=False,
                error=f"Agent '{agent_name}' is not running (status: {target.get('status')}).",
            )

        host = target.get("host", "localhost")
        port = target.get("port", 0)
        if not port:
            return ToolResult(success=False, error=f"No port info for agent '{agent_name}'.")

        # Check approval requirement
        agent = kwargs.get("_agent")
        session = kwargs.get("_session")
        metadata = getattr(session, "metadata", {}) or {} if session else {}

        # Get source agent name
        source_name = metadata.get("session_display_name", "another agent")
        peer_display = target.get("name", agent_name)

        # Broadcast peer activity if available
        broadcast = getattr(agent, "ws_broadcast", None) if agent else None

        def _emit(activity_type: str, detail: str = "") -> None:
            if not callable(broadcast):
                return
            broadcast({
                "type": "peer_activity",
                "peer_name": peer_display,
                "activity_type": activity_type,
                "detail": detail,
            })

        _emit("connecting", f"Connecting to {peer_display}...")

        log.info("Consulting peer via fleet", target=peer_display, host=host, port=port)

        try:
            async with httpx.AsyncClient(timeout=600.0) as client:
                async with client.stream(
                    "POST",
                    f"{fd_url}/fd/consult-peer",
                    json={
                        "host": host,
                        "port": port,
                        "auth": "",
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
            return ToolResult(success=False, error=f"Timed out waiting for '{agent_name}'.")
        except httpx.ConnectError as e:
            _emit("error", "Connection failed")
            return ToolResult(success=False, error=f"Cannot connect to Flight Deck at {fd_url}.")
        except Exception as e:
            log.error("Fleet consultation failed", error=str(e))
            _emit("error", str(e)[:100])
            return ToolResult(success=False, error=str(e))

    async def execute(self, action: str, agent_name: str = "", message: str = "", **kwargs: Any) -> ToolResult:
        fd_url = self._get_fd_url(**kwargs)
        if not fd_url:
            return ToolResult(
                success=False,
                error="Flight Deck URL not available. This tool requires Flight Deck.",
            )

        if action == "list_agents":
            return await self._list_agents(fd_url, **kwargs)
        elif action == "consult":
            if not agent_name:
                return ToolResult(success=False, error="agent_name is required for consult action.")
            if not message:
                return ToolResult(success=False, error="message is required for consult action.")
            return await self._consult(fd_url, agent_name, message, **kwargs)
        else:
            return ToolResult(success=False, error=f"Unknown action: {action}. Use 'list_agents' or 'consult'.")
