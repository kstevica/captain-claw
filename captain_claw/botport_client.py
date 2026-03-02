"""BotPort WebSocket client for Captain Claw.

Maintains a persistent WebSocket connection to a BotPort server,
handles registration, heartbeats, and bidirectional concern routing.

When this CC instance is CC-A (originator): sends concerns, receives results.
When this CC instance is CC-B (handler): receives dispatches, runs agents, sends results.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import TYPE_CHECKING, Any, Callable

import aiohttp

from captain_claw.config import BotPortClientConfig, get_config
from captain_claw.instructions import InstructionLoader
from captain_claw.logging import get_logger
from captain_claw.personality import load_personality, personality_to_prompt_block
from captain_claw.session import Session, get_session_manager

if TYPE_CHECKING:
    from captain_claw.agent import Agent
    from captain_claw.llm import LLMProvider

log = get_logger(__name__)

# System prompt fragment injected into dispatched agent sessions.
_DISPATCH_SYSTEM_PROMPT = (
    "You are handling a request from another agent instance named \"{from_instance}\".\n"
    "You are communicating agent-to-agent, not agent-to-human.\n"
    "Be precise, structured, and direct in your responses.\n"
    "The requesting agent will interpret your response and relay it to its user.\n\n"
    "Requesting instance: {from_instance}\n"
    "Task: {task}"
)


class BotPortClient:
    """Persistent WebSocket client connecting a CC instance to BotPort."""

    def __init__(
        self,
        config: BotPortClientConfig,
        provider: LLMProvider | None = None,
        status_callback: Callable[[str], None] | None = None,
        tool_output_callback: Callable[[str, dict[str, Any], str], None] | None = None,
        thinking_callback: Callable[[str, str, str], None] | None = None,
    ) -> None:
        self._config = config
        self._provider = provider
        self._status_callback = status_callback
        self._tool_output_callback = tool_output_callback
        self._thinking_callback = thinking_callback

        # Connection state.
        self._session: aiohttp.ClientSession | None = None
        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._connected = False
        self._instance_id: str = ""
        self._should_run = False

        # Background tasks.
        self._receive_task: asyncio.Task[None] | None = None
        self._heartbeat_task: asyncio.Task[None] | None = None
        self._reconnect_task: asyncio.Task[None] | None = None

        # CC-A state: pending concern results.
        self._pending_results: dict[str, asyncio.Future[dict[str, Any]]] = {}

        # CC-B state: dispatch agents and sessions.
        self._dispatch_agents: dict[str, Any] = {}  # concern_id -> Agent
        self._dispatch_sessions: dict[str, str] = {}  # concern_id -> session_id
        self._dispatch_personas: dict[str, str] = {}  # concern_id -> persona name

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def instance_id(self) -> str:
        return self._instance_id

    # ── Lifecycle ────────────────────────────────────────────────

    async def start(self) -> None:
        """Start the client (connect + background loops)."""
        if self._should_run:
            return
        self._should_run = True
        self._session = aiohttp.ClientSession()
        await self._connect()
        if self._connected:
            self._receive_task = asyncio.create_task(self._receive_loop())
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        else:
            # Start reconnect loop if initial connect failed.
            self._reconnect_task = asyncio.create_task(self._reconnect_loop())

    async def stop(self) -> None:
        """Stop the client and clean up."""
        self._should_run = False

        for task in (self._receive_task, self._heartbeat_task, self._reconnect_task):
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        if self._ws and not self._ws.closed:
            await self._ws.close()
        if self._session and not self._session.closed:
            await self._session.close()

        self._connected = False

        # Resolve any pending futures with errors.
        for future in self._pending_results.values():
            if not future.done():
                future.set_result({"ok": False, "error": "BotPort client stopped"})
        self._pending_results.clear()

        log.info("BotPort client stopped")

    # ── Connection ───────────────────────────────────────────────

    async def _connect(self) -> bool:
        """Establish WebSocket connection and register."""
        url = self._config.url.strip()
        if not url:
            log.warning("BotPort URL not configured")
            return False

        try:
            assert self._session is not None
            self._ws = await self._session.ws_connect(
                url,
                max_msg_size=4 * 1024 * 1024,
                heartbeat=60.0,
            )
        except Exception as exc:
            log.warning("BotPort connection failed: %s", exc)
            self._connected = False
            return False

        # Send registration.
        capabilities = self._build_capabilities()
        register_msg = {
            "type": "register",
            "instance_name": self._config.instance_name,
            "key": self._config.key,
            "secret": self._config.secret,
            "capabilities": capabilities,
        }
        await self._ws.send_str(json.dumps(register_msg))

        # Wait for registration response.
        try:
            raw = await asyncio.wait_for(self._ws.receive_str(), timeout=10.0)
            data = json.loads(raw)
            if data.get("type") == "registered" and data.get("ok"):
                self._instance_id = data.get("instance_id", "")
                self._connected = True
                log.info(
                    "Connected to BotPort as '%s' (id=%s)",
                    self._config.instance_name, self._instance_id[:8],
                )
                return True
            else:
                error = data.get("error", "Unknown registration error")
                log.error("BotPort registration failed: %s", error)
                self._connected = False
                return False
        except (TimeoutError, asyncio.TimeoutError):
            log.error("BotPort registration timed out")
            self._connected = False
            return False

    async def _reconnect_loop(self) -> None:
        """Background loop that attempts reconnection."""
        delay = self._config.reconnect_delay_seconds
        while self._should_run:
            try:
                await asyncio.sleep(delay)
                if self._connected:
                    continue

                log.info("Attempting BotPort reconnection...")
                if self._ws and not self._ws.closed:
                    await self._ws.close()

                if await self._connect():
                    # Restart receive and heartbeat.
                    if self._receive_task and not self._receive_task.done():
                        self._receive_task.cancel()
                    if self._heartbeat_task and not self._heartbeat_task.done():
                        self._heartbeat_task.cancel()
                    self._receive_task = asyncio.create_task(self._receive_loop())
                    self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
                    return  # Stop reconnect loop on success.
                else:
                    delay = min(delay * 1.5, 60.0)  # Exponential backoff up to 60s.

            except asyncio.CancelledError:
                break
            except Exception as exc:
                log.error("Reconnection error: %s", exc)
                await asyncio.sleep(delay)

    # ── Message handling ─────────────────────────────────────────

    async def _receive_loop(self) -> None:
        """Main WebSocket receive loop."""
        assert self._ws is not None
        try:
            async for raw_msg in self._ws:
                if raw_msg.type == aiohttp.WSMsgType.TEXT:
                    try:
                        data = json.loads(raw_msg.data)
                        await self._handle_message(data)
                    except json.JSONDecodeError as exc:
                        log.warning("Invalid JSON from BotPort: %s", exc)
                    except Exception as exc:
                        log.error("BotPort message handler error: %s", exc)

                elif raw_msg.type in (aiohttp.WSMsgType.ERROR, aiohttp.WSMsgType.CLOSED):
                    break

        except asyncio.CancelledError:
            return
        except Exception as exc:
            log.error("BotPort receive loop error: %s", exc)

        # Connection lost.
        self._connected = False
        log.warning("BotPort connection lost")
        if self._should_run:
            self._reconnect_task = asyncio.create_task(self._reconnect_loop())

    async def _handle_message(self, data: dict[str, Any]) -> None:
        msg_type = str(data.get("type", ""))

        if msg_type == "heartbeat_ack":
            pass  # Nothing to do.

        elif msg_type == "concern_ack":
            # Acknowledgment of our submitted concern.
            concern_id = data.get("concern_id", "")
            if not data.get("ok"):
                future = self._pending_results.get(concern_id)
                if future and not future.done():
                    future.set_result({
                        "ok": False,
                        "error": data.get("error", "Concern rejected"),
                    })

        elif msg_type == "concern_result":
            await self._handle_concern_result(data)

        elif msg_type == "dispatch":
            asyncio.create_task(self._handle_dispatch(data))

        elif msg_type == "follow_up":
            asyncio.create_task(self._handle_incoming_follow_up(data))

        elif msg_type == "context_request":
            await self._handle_context_request(data)

        elif msg_type == "context_reply":
            await self._handle_context_reply(data)

        elif msg_type == "concern_closed":
            await self._handle_concern_closed(data)

        elif msg_type == "timeout_notice":
            await self._handle_timeout_notice(data)

        else:
            log.debug("Unknown BotPort message type: %s", msg_type)

    # ── Heartbeat ────────────────────────────────────────────────

    async def _heartbeat_loop(self) -> None:
        """Send periodic heartbeats to BotPort."""
        interval = self._config.heartbeat_interval_seconds
        while self._should_run and self._connected:
            try:
                await asyncio.sleep(interval)
                if self._ws and not self._ws.closed:
                    await self._ws.send_str(json.dumps({
                        "type": "heartbeat",
                        "instance_id": self._instance_id,
                        "active_concerns": len(self._dispatch_agents),
                    }))
            except asyncio.CancelledError:
                break
            except Exception as exc:
                log.warning("Heartbeat error: %s", exc)

    # ── CC-A: Sending concerns ───────────────────────────────────

    async def send_concern(
        self,
        task: str,
        context: dict[str, Any] | None = None,
        expertise_tags: list[str] | None = None,
        session_id: str = "",
        timeout: float = 300.0,
    ) -> dict[str, Any]:
        """Send a concern to BotPort and wait for the result.

        Returns dict with keys: ok, response, error, metadata, concern_id
        """
        if not self._connected or not self._ws:
            return {"ok": False, "error": "Not connected to BotPort"}

        concern_id = str(uuid.uuid4())
        loop = asyncio.get_event_loop()
        future: asyncio.Future[dict[str, Any]] = loop.create_future()
        self._pending_results[concern_id] = future

        try:
            await self._ws.send_str(json.dumps({
                "type": "concern",
                "concern_id": concern_id,
                "task": task,
                "context": context or {},
                "expertise_tags": expertise_tags or [],
                "from_session": session_id,
            }))

            result = await asyncio.wait_for(future, timeout=timeout)
            result["concern_id"] = concern_id
            return result

        except (TimeoutError, asyncio.TimeoutError):
            return {
                "ok": False,
                "error": "Concern timed out waiting for result",
                "concern_id": concern_id,
            }
        except Exception as exc:
            return {
                "ok": False,
                "error": f"Error sending concern: {exc}",
                "concern_id": concern_id,
            }
        finally:
            self._pending_results.pop(concern_id, None)

    async def send_follow_up(
        self,
        concern_id: str,
        message: str,
        additional_context: dict[str, Any] | None = None,
        timeout: float = 300.0,
    ) -> dict[str, Any]:
        """Send a follow-up on an existing concern."""
        if not self._connected or not self._ws:
            return {"ok": False, "error": "Not connected to BotPort"}

        loop = asyncio.get_event_loop()
        future: asyncio.Future[dict[str, Any]] = loop.create_future()
        self._pending_results[concern_id] = future

        try:
            await self._ws.send_str(json.dumps({
                "type": "follow_up",
                "concern_id": concern_id,
                "message": message,
                "additional_context": additional_context or {},
            }))

            result = await asyncio.wait_for(future, timeout=timeout)
            result["concern_id"] = concern_id
            return result

        except (TimeoutError, asyncio.TimeoutError):
            return {"ok": False, "error": "Follow-up timed out", "concern_id": concern_id}
        except Exception as exc:
            return {"ok": False, "error": str(exc), "concern_id": concern_id}
        finally:
            self._pending_results.pop(concern_id, None)

    async def close_concern(self, concern_id: str) -> bool:
        """Close a concern we originated."""
        if not self._connected or not self._ws:
            return False
        try:
            await self._ws.send_str(json.dumps({
                "type": "close_concern",
                "concern_id": concern_id,
            }))
            return True
        except Exception:
            return False

    async def list_agents(self) -> list[dict[str, Any]]:
        """Query BotPort's registry for all connected agents and their capabilities."""
        # Derive HTTP URL from WebSocket URL.
        api_url = self._registry_api_url()
        if api_url and self._session and not self._session.closed:
            try:
                async with self._session.get(api_url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if isinstance(data, list) and data:
                            return data
                        log.debug("Registry API returned empty, falling back to local")
                    else:
                        log.warning("Registry API returned %d", resp.status)
            except Exception as exc:
                log.debug("Registry API call failed: %s, using local capabilities", exc)

        # Fallback: return our own capabilities.
        return [self._build_capabilities()]

    def _registry_api_url(self) -> str:
        """Derive the BotPort HTTP registry URL from the WebSocket URL."""
        url = self._config.url.strip()
        if not url:
            return ""
        # ws://host:port/ws  ->  http://host:port/api/registry
        http_url = url.replace("ws://", "http://").replace("wss://", "https://")
        if http_url.endswith("/ws"):
            http_url = http_url[:-3]
        return http_url.rstrip("/") + "/api/registry"

    # ── CC-A: Receiving results ──────────────────────────────────

    async def _handle_concern_result(self, data: dict[str, Any]) -> None:
        """Handle result relayed from BotPort (we are CC-A)."""
        concern_id = data.get("concern_id", "")
        future = self._pending_results.get(concern_id)
        if future and not future.done():
            future.set_result({
                "ok": data.get("ok", True),
                "response": data.get("response", ""),
                "from_instance_name": data.get("from_instance_name", ""),
                "persona_name": data.get("persona_name", ""),
                "metadata": data.get("metadata", {}),
                "error": data.get("error", ""),
            })

    async def _handle_timeout_notice(self, data: dict[str, Any]) -> None:
        concern_id = data.get("concern_id", "")
        future = self._pending_results.get(concern_id)
        if future and not future.done():
            future.set_result({
                "ok": False,
                "error": f"Concern timed out: {data.get('reason', 'idle_timeout')}",
            })

    async def _handle_context_request(self, data: dict[str, Any]) -> None:
        """BotPort forwarded a context request from CC-B. Auto-reply for now."""
        # In v1, we auto-reply with empty answers. The agent could handle this
        # more intelligently in a future version.
        concern_id = data.get("concern_id", "")
        if self._ws and not self._ws.closed:
            await self._ws.send_str(json.dumps({
                "type": "context_reply",
                "concern_id": concern_id,
                "answers": {},
            }))

    # ── CC-B: Handling dispatches ────────────────────────────────

    async def _handle_dispatch(self, data: dict[str, Any]) -> None:
        """Handle a dispatched concern from BotPort (we are CC-B)."""
        concern_id = data.get("concern_id", "")
        from_instance = data.get("from_instance_name", "unknown")
        task = data.get("task", "")
        context = data.get("context", {})
        persona_hint = data.get("persona_hint", "")

        log.info(
            "Dispatch received: concern=%s from=%s task_len=%d",
            concern_id[:8], from_instance, len(task),
        )

        try:
            agent = await self._spawn_dispatch_agent(
                concern_id, task, context, from_instance, persona_hint,
            )
            self._dispatch_agents[concern_id] = agent
            self._dispatch_personas[concern_id] = persona_hint or self._config.instance_name

            # Build effective input with context.
            effective_input = task
            if context:
                context_str = json.dumps(context, indent=2, default=str)
                effective_input = f"{task}\n\nContext:\n{context_str}"

            # Run the agent.
            response = await agent.complete(effective_input)

            # Send result back, including persona name that handled it.
            if self._ws and not self._ws.closed:
                await self._ws.send_str(json.dumps({
                    "type": "result",
                    "concern_id": concern_id,
                    "response": response or "",
                    "persona_name": persona_hint or self._config.instance_name,
                    "ok": True,
                    "metadata": {
                        "model_used": getattr(agent, "_last_model_used", ""),
                        "tokens": getattr(agent, "last_usage", {}).get("total_tokens", 0),
                    },
                }))

            log.info("Dispatch completed: concern=%s response_len=%d", concern_id[:8], len(response or ""))

        except Exception as exc:
            log.error("Dispatch execution failed: concern=%s error=%s", concern_id[:8], exc)
            if self._ws and not self._ws.closed:
                await self._ws.send_str(json.dumps({
                    "type": "result",
                    "concern_id": concern_id,
                    "ok": False,
                    "error": str(exc),
                }))

    async def _handle_incoming_follow_up(self, data: dict[str, Any]) -> None:
        """Handle a follow-up message for a concern we're handling (CC-B)."""
        concern_id = data.get("concern_id", "")
        message = data.get("message", "")
        agent = self._dispatch_agents.get(concern_id)

        if not agent:
            log.warning("Follow-up for unknown dispatch concern %s", concern_id[:8])
            return

        # Retrieve persona hint from the agent's dispatch context.
        persona_name = self._dispatch_personas.get(concern_id, self._config.instance_name)

        try:
            response = await agent.complete(message)

            if self._ws and not self._ws.closed:
                await self._ws.send_str(json.dumps({
                    "type": "result",
                    "concern_id": concern_id,
                    "response": response or "",
                    "persona_name": persona_name,
                    "ok": True,
                }))
        except Exception as exc:
            log.error("Follow-up execution failed: %s", exc)
            if self._ws and not self._ws.closed:
                await self._ws.send_str(json.dumps({
                    "type": "result",
                    "concern_id": concern_id,
                    "ok": False,
                    "error": str(exc),
                }))

    async def _handle_concern_closed(self, data: dict[str, Any]) -> None:
        """Clean up a dispatch agent when the concern is closed."""
        concern_id = data.get("concern_id", "")
        self._dispatch_agents.pop(concern_id, None)
        self._dispatch_sessions.pop(concern_id, None)
        self._dispatch_personas.pop(concern_id, None)
        log.info("Dispatch concern %s closed, agent cleaned up", concern_id[:8])

    async def _handle_context_reply(self, data: dict[str, Any]) -> None:
        """Handle context reply from CC-A (we are CC-B).

        In v1, context replies are logged but not yet fed into running agents.
        """
        concern_id = data.get("concern_id", "")
        log.debug("Context reply received for concern %s", concern_id[:8])

    # ── Agent spawning ───────────────────────────────────────────

    async def _spawn_dispatch_agent(
        self,
        concern_id: str,
        task: str,
        context: dict[str, Any],
        from_instance: str,
        persona_hint: str,
    ) -> Any:
        """Create an ephemeral agent for handling a dispatched concern.

        Follows the same pattern as Telegram bridge agent creation.
        """
        from captain_claw.agent import Agent as AgentCls

        sm = get_session_manager()
        session = await sm.create_session(name=f"bp-{from_instance[:20]}-{concern_id[:8]}")
        self._dispatch_sessions[concern_id] = session.id

        # Add agent-to-agent system context.
        dispatch_context = _DISPATCH_SYSTEM_PROMPT.format(
            from_instance=from_instance,
            task=task,
        )
        session.add_message("system", dispatch_context)

        agent = AgentCls(
            provider=self._provider,
            status_callback=self._status_callback,
            tool_output_callback=self._tool_output_callback,
            thinking_callback=self._thinking_callback,
        )
        agent.session = session
        agent.session_manager = sm
        agent._sync_runtime_flags_from_session()

        # Register tools (but exclude botport tool to prevent multi-hop in v1).
        agent._register_default_tools()
        try:
            agent.tools.unregister("botport")
        except Exception:
            pass

        agent.instructions = InstructionLoader()
        agent._initialized = True
        agent.max_iterations = 10  # Allow more iterations than a basic worker.

        log.info(
            "Spawned dispatch agent: concern=%s session=%s persona=%s",
            concern_id[:8], session.id[:8], persona_hint or "default",
        )
        return agent

    # ── Capabilities ─────────────────────────────────────────────

    def _build_capabilities(self) -> dict[str, Any]:
        """Build capabilities manifest from current CC configuration."""
        cfg = get_config()
        capabilities: dict[str, Any] = {
            "max_concurrent": self._config.max_concurrent,
        }

        # Personas: global personality + user profiles.
        if self._config.advertise_personas:
            try:
                from captain_claw.personality import list_user_personalities

                personality = load_personality()
                tags = [t.strip().lower() for t in personality.expertise if t.strip()]
                personas: list[dict[str, Any]] = [{
                    "name": personality.name,
                    "description": personality.description,
                    "background": personality.background,
                    "expertise_tags": tags,
                }]

                for up in list_user_personalities():
                    utags = [t.strip().lower() for t in up.get("expertise", []) if t.strip()]
                    personas.append({
                        "name": up.get("name", ""),
                        "description": up.get("description", ""),
                        "background": up.get("background", ""),
                        "expertise_tags": utags,
                    })

                capabilities["personas"] = personas
            except Exception:
                capabilities["personas"] = []

        # Tools.
        if self._config.advertise_tools:
            capabilities["tools"] = list(cfg.tools.enabled)

        # Models.
        if self._config.advertise_models:
            models = [f"{cfg.model.provider}:{cfg.model.model}"]
            for allowed in cfg.model.allowed:
                models.append(f"{allowed.provider}:{allowed.model}")
            capabilities["models"] = models

        return capabilities
