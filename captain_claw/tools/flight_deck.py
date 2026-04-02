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
        "Discover, communicate with, and spawn peer agents in the Flight Deck environment. "
        "Actions: 'list_agents' to discover peers, 'consult' for quick synchronous Q&A, "
        "'delegate' for tasks the peer should do independently, "
        "'spawn_agent' to create a new agent in the fleet. "
        "IMPORTANT: When the user says 'delegate', or the task is large/long-running "
        "(scraping, research, analysis, file creation), ALWAYS use action='delegate'. "
        "Only use 'consult' for quick questions where you need the answer immediately to continue."
    )
    timeout_seconds = 600.0

    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["list_agents", "consult", "delegate", "spawn_agent"],
                "description": (
                    "'list_agents' — get all agents currently in the fleet. "
                    "'consult' — ONLY for quick questions where you need the answer right now to continue your work (synchronous, blocks until response). "
                    "'delegate' — for ANY task the peer should handle: research, scraping, analysis, summarization, file creation, etc. "
                    "You send the task and immediately free yourself. The peer works independently and delivers results back to you when done. "
                    "'spawn_agent' — create a new agent in the fleet. Provide agent_name (required), and optionally "
                    "spawn_provider, spawn_model, spawn_description via the message field as JSON. "
                    "Uses your own model/key as defaults. "
                    "RULE: If the user says 'delegate' or the task involves work (not just a question), use 'delegate'."
                ),
            },
            "agent_name": {
                "type": "string",
                "description": (
                    "Name of the peer agent to consult or delegate to "
                    "(required for 'consult' and 'delegate' actions). "
                    "Use list_agents first to see available agents."
                ),
            },
            "message": {
                "type": "string",
                "description": (
                    "The message/task to send to the peer agent "
                    "(required for 'consult' and 'delegate' actions)."
                ),
            },
        },
        "required": ["action"],
    }

    def _get_fd_url(self, **kwargs: Any) -> str:
        """Resolve the Flight Deck URL from session metadata, agent attributes, or env."""
        import os
        session = kwargs.get("_session")
        agent = kwargs.get("_agent")
        metadata = getattr(session, "metadata", {}) or {} if session else {}
        fd_url = metadata.get("fd_url", "")
        if not fd_url and agent:
            fd_url = getattr(agent, "_fd_url", "") or ""
        if not fd_url:
            fd_url = os.environ.get("FD_URL", "") or os.environ.get("FD_INTERNAL_URL", "")
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

        target, agents, error = await self._resolve_target(fd_url, agent_name, **kwargs)
        if error:
            return ToolResult(success=False, error=error)

        host = target.get("host", "localhost")
        port = target.get("port", 0)

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
                        if evt_type == "heartbeat":
                            elapsed = data.get("elapsed", 0)
                            timeout = data.get("timeout", 0)
                            _emit("status", f"Still working... ({elapsed}s / {timeout}s)")
                        elif evt_type == "status":
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

    async def _resolve_target(self, fd_url: str, agent_name: str, **kwargs: Any):
        """Look up a target agent from the fleet. Returns (target_dict, agents_list) or raises."""
        import httpx

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(f"{fd_url}/fd/fleet")
                if resp.status_code != 200:
                    return None, None, f"Fleet query failed: HTTP {resp.status_code}"
                agents = resp.json()
        except Exception as e:
            return None, None, f"Cannot reach Flight Deck: {e}"

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
            return None, agents, f"Agent '{agent_name}' not found in fleet. Available: {available}"

        if target.get("status") != "running":
            return None, agents, f"Agent '{agent_name}' is not running (status: {target.get('status')})."

        if not target.get("port"):
            return None, agents, f"No port info for agent '{agent_name}'."

        return target, agents, None

    async def _delegate(self, fd_url: str, agent_name: str, message: str, **kwargs: Any) -> ToolResult:
        """Delegate a task to a peer agent (fire-and-forget). Results are delivered back as a message."""
        try:
            import httpx
        except ImportError:
            return ToolResult(success=False, error="httpx is required")

        target, agents, error = await self._resolve_target(fd_url, agent_name, **kwargs)
        if error:
            return ToolResult(success=False, error=error)

        agent = kwargs.get("_agent")
        session = kwargs.get("_session")
        metadata = getattr(session, "metadata", {}) or {} if session else {}
        source_name = metadata.get("session_display_name", "")
        peer_display = target.get("name", agent_name)

        # Find source agent's port — try multiple strategies
        source_port = 0
        source_host = "localhost"

        # Strategy 1: match by session_display_name in fleet
        if source_name:
            for a in agents:
                if a.get("name", "").lower() == source_name.lower():
                    source_port = a.get("port", 0)
                    source_host = a.get("host", "localhost")
                    break

        # Strategy 2: match by config web port in fleet
        if not source_port:
            try:
                from captain_claw.config import get_config
                cfg_port = get_config().web.port
                if cfg_port:
                    for a in agents:
                        if a.get("port") == cfg_port:
                            source_port = cfg_port
                            source_host = a.get("host", "localhost")
                            if not source_name:
                                source_name = a.get("name", "this agent")
                            break
                # Strategy 3: use config port directly as last resort
                if not source_port:
                    source_port = cfg_port
                    if not source_name:
                        source_name = "this agent"
            except Exception:
                pass

        if not source_port:
            return ToolResult(
                success=False,
                error="Cannot delegate: could not determine own port. Use 'consult' (synchronous) instead.",
            )

        # Detect originating platform (telegram, web, etc.) so delegate results
        # are delivered back to the correct session/channel.
        origin_platform = "web"
        origin_user_id = ""
        origin_chat_id = 0
        if agent and hasattr(agent, "_user_id") and hasattr(agent, "_telegram_chat_id"):
            origin_platform = "telegram"
            origin_user_id = str(getattr(agent, "_user_id", ""))
            origin_chat_id = int(getattr(agent, "_telegram_chat_id", 0))

        log.info("Delegating task to peer", target=peer_display, source=source_name,
                 target_port=target.get("port"), source_port=source_port,
                 origin_platform=origin_platform)

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(
                    f"{fd_url}/fd/delegate-peer",
                    json={
                        "target_host": target.get("host", "localhost"),
                        "target_port": target.get("port"),
                        "target_name": peer_display,
                        "source_host": "localhost",
                        "source_port": source_port,
                        "source_name": source_name,
                        "message": message,
                        "timeout": 600.0,
                        "origin_platform": origin_platform,
                        "origin_user_id": origin_user_id,
                        "origin_chat_id": origin_chat_id,
                    },
                )
                if resp.status_code != 200:
                    return ToolResult(success=False, error=f"Flight Deck returned HTTP {resp.status_code}")
                result = resp.json()
        except Exception as e:
            return ToolResult(success=False, error=f"Failed to delegate: {e}")

        return ToolResult(
            success=True,
            content=(
                f"Task delegated to **{peer_display}**. The task is now {peer_display}'s responsibility.\n\n"
                f"IMPORTANT: Do NOT attempt to do this task yourself. Do NOT use your own tools "
                f"(browser, web_search, shell, etc.) to work on the delegated task. "
                f"{peer_display} will deliver results back to you as a message when finished. "
                f"Tell the user the task has been delegated and move on."
            ),
        )

    async def _spawn_agent(self, fd_url: str, agent_name: str, message: str = "", **kwargs: Any) -> ToolResult:
        """Spawn a new agent in the Flight Deck fleet.

        Uses the caller's own provider/model/api_key as defaults so that
        Old Man can spawn agents that inherit its configuration.
        The *message* field can optionally contain a JSON object with overrides:
        ``{"provider": "...", "model": "...", "description": "..."}``.
        """
        try:
            import httpx
        except ImportError:
            return ToolResult(success=False, error="httpx is required")

        # Resolve caller's own model config as defaults for the new agent.
        agent = kwargs.get("_agent")
        provider_obj = getattr(agent, "provider", None) if agent else None
        default_provider = str(getattr(provider_obj, "provider", "ollama") or "ollama")
        default_model = str(getattr(provider_obj, "model", "") or "")
        default_api_key = str(getattr(provider_obj, "api_key", "") or "")

        # Parse optional overrides from message.
        overrides: dict[str, Any] = {}
        if message.strip().startswith("{"):
            try:
                overrides = json.loads(message)
            except json.JSONDecodeError:
                pass

        spawn_body = {
            "name": agent_name,
            "description": overrides.get("description", f"Agent spawned by Old Man"),
            "provider": overrides.get("provider", default_provider),
            "model": overrides.get("model", default_model),
            "provider_api_key": overrides.get("api_key", default_api_key),
            "tools": overrides.get("tools", [
                "shell", "read", "write", "glob", "edit",
                "web_fetch", "web_search", "browser",
                "pdf_extract", "docx_extract", "xlsx_extract", "pptx_extract",
                "scripts", "playbooks", "personality", "flight_deck",
            ]),
            "web_enabled": True,
            "web_port": overrides.get("web_port", 0),  # 0 = auto-assign
        }

        # Try process spawn first (no Docker needed), fall back to docker.
        for endpoint in ["/fd/spawn-process", "/fd/spawn"]:
            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    resp = await client.post(f"{fd_url}{endpoint}", json=spawn_body)
                    if resp.status_code == 200:
                        result = resp.json()
                        if result.get("ok"):
                            return ToolResult(
                                success=True,
                                content=(
                                    f"Agent **{agent_name}** spawned successfully.\n"
                                    f"Provider: {spawn_body['provider']}, Model: {spawn_body['model']}\n"
                                    f"The agent will be available in the fleet shortly. "
                                    f"Use list_agents to check when it's running."
                                ),
                            )
                    # If process spawn fails with 400 (already exists), don't retry with docker
                    if resp.status_code == 400:
                        detail = resp.json().get("detail", "")
                        return ToolResult(success=False, error=f"Spawn failed: {detail}")
            except Exception:
                continue

        return ToolResult(success=False, error="Failed to spawn agent via Flight Deck. Neither process nor Docker spawn succeeded.")

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
        elif action == "delegate":
            if not agent_name:
                return ToolResult(success=False, error="agent_name is required for delegate action.")
            if not message:
                return ToolResult(success=False, error="message is required for delegate action.")
            return await self._delegate(fd_url, agent_name, message, **kwargs)
        elif action == "spawn_agent":
            if not agent_name:
                return ToolResult(success=False, error="agent_name is required for spawn_agent action.")
            return await self._spawn_agent(fd_url, agent_name, message, **kwargs)
        else:
            return ToolResult(success=False, error=f"Unknown action: {action}. Use 'list_agents', 'consult', 'delegate', or 'spawn_agent'.")
