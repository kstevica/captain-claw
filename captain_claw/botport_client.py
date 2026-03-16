"""BotPort WebSocket client for Captain Claw.

Maintains a persistent WebSocket connection to a BotPort server,
handles registration, heartbeats, and bidirectional concern routing.

When this CC instance is CC-A (originator): sends concerns, receives results.
When this CC instance is CC-B (handler): receives dispatches, runs agents, sends results.
"""

from __future__ import annotations

import asyncio
import base64
import gzip
import json
import os
import re
import tempfile
import uuid
from pathlib import Path
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

# Formats that are already compressed — skip gzip for these.
_PRECOMPRESSED_EXTS = frozenset({
    ".zip", ".gz", ".bz2", ".xz", ".7z", ".rar",
    ".jpg", ".jpeg", ".png", ".gif", ".webp", ".avif",
    ".mp3", ".mp4", ".m4a", ".m4v", ".avi", ".mov", ".mkv", ".webm",
    ".woff", ".woff2", ".pdf",
})

# System prompt fragment injected into dispatched agent sessions.
_DISPATCH_SYSTEM_PROMPT = (
    "You are handling a request from another agent instance named \"{from_instance}\".\n"
    "You are communicating agent-to-agent, not agent-to-human.\n"
    "Be precise, structured, and direct in your responses.\n"
    "The requesting agent will interpret your response and relay it to its user.\n\n"
    "CRITICAL — AUTONOMOUS EXECUTION RULES:\n"
    "This is part of an automated multi-agent workflow. You MUST:\n"
    "- Execute the task completely and autonomously. Make all decisions yourself.\n"
    "- NEVER ask questions, request clarification, or seek confirmation.\n"
    "- NEVER present options (e.g. \"Option A vs Option B\") and ask which to choose.\n"
    "- NEVER say \"Should I proceed?\", \"Would you like me to...\", or similar.\n"
    "- If the task is ambiguous, use your best judgment and proceed.\n"
    "- If multiple approaches exist, pick the best one and execute it.\n"
    "- Deliver a complete, finished result — not a proposal or draft for approval.\n\n"
    "CRITICAL — FILE WRITING RULES:\n"
    "- Write each file EXACTLY ONCE. After a successful write, the file is DONE.\n"
    "- NEVER rewrite, update, or overwrite a file you have already written.\n"
    "- After writing a file, the system may compact the content in your history to "
    "\"[written to disk: ...]\". This is NORMAL — the file IS saved correctly. "
    "Do NOT interpret this as the file being lost or needing to be rewritten.\n"
    "- After completing all tool calls, provide a brief text summary of what you did "
    "and the file(s) you created, then STOP.\n\n"
    "Requesting instance: {from_instance}\n"
    "Task: {task}"
)


def _build_persona_prompt(agent_spec: dict[str, Any]) -> str:
    """Build a persona system prompt from a designed agent spec."""
    name = agent_spec.get("persona_name", "")
    if not name:
        return ""

    desc = agent_spec.get("persona_description", "")
    expertise = agent_spec.get("persona_expertise", [])
    instructions = agent_spec.get("persona_instructions", "")

    parts = [f"You are {name}."]
    if desc:
        parts.append(desc)
    if expertise:
        parts.append(f"Your areas of expertise: {', '.join(expertise)}.")
    if instructions:
        parts.append(f"\n{instructions}")

    return "\n".join(parts)


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

        # Activity emission throttle (avoid flooding botport).
        self._activity_min_interval = 0.3  # seconds between activity messages
        self._last_activity_sent: float = 0.0

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

        # File transfer state.
        self._dispatch_swarm_ids: dict[str, str] = {}  # concern_id -> swarm_id
        self._dispatch_created_files: dict[str, list[str]] = {}  # concern_id -> [file_paths]
        self._file_upload_futures: dict[str, asyncio.Future[dict[str, Any]]] = {}
        self._file_response_futures: dict[str, asyncio.Future[dict[str, Any]]] = {}

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
                max_msg_size=16 * 1024 * 1024,  # 16 MB for file transfers
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

        elif msg_type == "file_upload_ack":
            self._handle_file_upload_ack(data)

        elif msg_type == "file_response":
            self._handle_file_response(data)

        elif msg_type == "file_list_response":
            self._handle_file_list_response(data)

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

        # Track swarm context for file transfers.
        swarm_id = context.get("swarm_id", "")
        if swarm_id:
            self._dispatch_swarm_ids[concern_id] = swarm_id
            self._dispatch_created_files[concern_id] = []

        log.info(
            "Dispatch received: concern=%s from=%s task_len=%d swarm=%s",
            concern_id[:8], from_instance, len(task),
            swarm_id[:8] if swarm_id else "none",
        )

        try:
            agent = await self._spawn_dispatch_agent(
                concern_id, task, context, from_instance, persona_hint,
            )
            self._dispatch_agents[concern_id] = agent
            self._dispatch_personas[concern_id] = persona_hint or self._config.instance_name

            # Build effective input with context.
            # Pre-fetch swarm files so the agent can read them locally.
            effective_input = task
            filtered_context = {
                k: v for k, v in context.items()
                if k not in ("swarm_files",)
            }
            swarm_files = context.get("swarm_files", [])
            prefetched: dict[str, str] = {}
            if swarm_files and swarm_id:
                prefetched = await self._prefetch_swarm_files(swarm_id, swarm_files)
            if prefetched:
                file_list = "\n".join(
                    f"  - {finfo['filename']} ({finfo['size']} bytes) — read from: {prefetched[finfo['path']]}"
                    for finfo in swarm_files
                    if finfo.get("path") in prefetched
                )
                effective_input += (
                    f"\n\n--- Files from Previous Tasks ---\n"
                    f"The following files have been downloaded to your local disk. "
                    f"Use the read tool with the exact paths below to access them.\n{file_list}"
                )
            elif swarm_files:
                # Fallback: show manifest even if pre-fetch failed.
                file_list = "\n".join(
                    f"  - {f['filename']} ({f['size']} bytes, {f['mime_type']}) path={f['path']}"
                    for f in swarm_files
                )
                effective_input += (
                    f"\n\n--- Available Files in Swarm Workspace ---\n"
                    f"The following files are available from previous tasks. "
                    f"If you need any of them, mention the file path and the "
                    f"orchestrator will provide them.\n{file_list}"
                )
            if filtered_context:
                context_str = json.dumps(filtered_context, indent=2, default=str)
                effective_input = f"{effective_input}\n\nContext:\n{context_str}"

            # Run the agent.
            response = await agent.complete(effective_input)

            # Scan agent response for file paths that exist on disk
            # (files from previous runs that weren't created in this session).
            if swarm_id and response:
                self._extract_files_from_response(concern_id, response)

            # Upload any files created during execution.
            uploaded_files = []
            if swarm_id:
                uploaded_files = await self._upload_created_files(concern_id)

            # Send result back, including persona name that handled it.
            result_metadata: dict[str, Any] = {
                "model_used": getattr(agent, "_last_model_used", ""),
                "tokens": getattr(agent, "last_usage", {}).get("total_tokens", 0),
            }
            if uploaded_files:
                result_metadata["uploaded_files"] = uploaded_files

            if self._ws and not self._ws.closed:
                await self._ws.send_str(json.dumps({
                    "type": "result",
                    "concern_id": concern_id,
                    "response": response or "",
                    "persona_name": persona_hint or self._config.instance_name,
                    "ok": True,
                    "metadata": result_metadata,
                }))

            log.info(
                "Dispatch completed: concern=%s response_len=%d files_uploaded=%d",
                concern_id[:8], len(response or ""), len(uploaded_files),
            )

        except Exception as exc:
            log.error("Dispatch execution failed: concern=%s error=%s", concern_id[:8], exc)
            if self._ws and not self._ws.closed:
                await self._ws.send_str(json.dumps({
                    "type": "result",
                    "concern_id": concern_id,
                    "ok": False,
                    "error": str(exc),
                }))
        finally:
            # Clean up file tracking state.
            self._dispatch_swarm_ids.pop(concern_id, None)
            self._dispatch_created_files.pop(concern_id, None)

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
        self._dispatch_swarm_ids.pop(concern_id, None)
        self._dispatch_created_files.pop(concern_id, None)
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

        # Apply designed agent persona if provided via swarm agent_spec.
        agent_spec = context.get("agent_spec", {}) if context else {}
        if agent_spec:
            persona_prompt = _build_persona_prompt(agent_spec)
            if persona_prompt:
                session.add_message("system", persona_prompt)

        # Build tool output callback that also tracks file creation.
        file_tracking_cb = self._make_file_tracking_tool_callback(
            concern_id,
            self._make_activity_tool_output_callback(self._tool_output_callback),
        )

        # Resolve provider — use model override from agent_spec if present.
        provider = self._provider
        spec_model_id = agent_spec.get("model_id", "") if agent_spec else ""
        if spec_model_id:
            provider = self._resolve_model_provider(spec_model_id)

        agent = AgentCls(
            provider=provider,
            status_callback=self._make_activity_status_callback(self._status_callback),
            tool_output_callback=file_tracking_cb,
            thinking_callback=self._make_activity_thinking_callback(self._thinking_callback),
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

        persona_name = agent_spec.get("persona_name", "") if agent_spec else ""
        log.info(
            "Spawned dispatch agent: concern=%s session=%s persona=%s model=%s",
            concern_id[:8], session.id[:8],
            persona_name or persona_hint or "default",
            spec_model_id or "default",
        )
        return agent

    def wrap_callbacks(
        self,
        status_cb: Callable[[str], None] | None = None,
        thinking_cb: Callable[[str, str, str], None] | None = None,
        tool_output_cb: Callable[[str, dict[str, Any], str], None] | None = None,
    ) -> tuple[
        Callable[[str], None],
        Callable[[str, str, str], None],
        Callable[[str, dict[str, Any], str], None],
    ]:
        """Wrap callbacks so they also emit activity to BotPort.

        Use this when creating the main agent so its activity is streamed too.
        Returns (status_cb, thinking_cb, tool_output_cb).
        """
        return (
            self._make_activity_status_callback(status_cb),
            self._make_activity_thinking_callback(thinking_cb),
            self._make_activity_tool_output_callback(tool_output_cb),
        )

    # ── Activity emission ────────────────────────────────────────

    def _send_activity(self, step_type: str, data: dict[str, Any]) -> None:
        """Send an activity message to BotPort (fire-and-forget).

        Throttled to avoid flooding the server. Thinking 'done' phase
        is always sent immediately to clear the slot.
        """
        import time

        now = time.monotonic()
        is_clear = step_type == "thinking" and data.get("phase") == "done"
        if not is_clear and (now - self._last_activity_sent) < self._activity_min_interval:
            return

        if not self._connected or not self._ws or self._ws.closed:
            return

        self._last_activity_sent = now
        msg = json.dumps({
            "type": "activity",
            "instance_id": self._instance_id,
            "step_type": step_type,
            "data": data,
        })
        # Fire-and-forget: schedule the send without blocking the agent loop.
        asyncio.ensure_future(self._ws.send_str(msg))

    def _make_activity_status_callback(
        self, original: Callable[[str], None] | None,
    ) -> Callable[[str], None]:
        """Wrap a status callback to also emit activity to BotPort."""
        def callback(status: str) -> None:
            if original:
                original(status)
            self._send_activity("status", {"text": status})
        return callback

    def _make_activity_thinking_callback(
        self, original: Callable[[str, str, str], None] | None,
    ) -> Callable[[str, str, str], None]:
        """Wrap a thinking callback to also emit activity to BotPort."""
        def callback(text: str, tool: str = "", phase: str = "tool") -> None:
            if original:
                original(text, tool, phase)
            self._send_activity("thinking", {"text": text, "tool": tool, "phase": phase})
        return callback

    def _make_file_tracking_tool_callback(
        self,
        concern_id: str,
        wrapped: Callable[[str, dict[str, Any], str], None],
    ) -> Callable[[str, dict[str, Any], str], None]:
        """Wrap a tool_output callback to also track file creation for swarm uploads."""
        def callback(tool_name: str, arguments: dict[str, Any], output: str) -> None:
            wrapped(tool_name, arguments, output)
            self._track_file_creation(concern_id, tool_name, arguments, output)
        return callback

    def _make_activity_tool_output_callback(
        self, original: Callable[[str, dict[str, Any], str], None] | None,
    ) -> Callable[[str, dict[str, Any], str], None]:
        """Wrap a tool_output callback to also emit activity to BotPort."""
        def callback(tool_name: str, arguments: dict[str, Any], output: str) -> None:
            if original:
                original(tool_name, arguments, output)
            self._send_activity("tool_output", {
                "tool_name": tool_name,
                "arguments": arguments,
                "output": output[:500],  # Truncate to avoid huge messages.
            })
        return callback

    # ── Model override ─────────────────────────────────────────────

    def _resolve_model_provider(self, model_id: str) -> Any:
        """Create an LLM provider for a specific model ID.

        Used when a designed agent spec requests a different model than
        the CC instance's default.  Falls back to self._provider on error.
        """
        try:
            from captain_claw.llm import LLMProvider, get_config

            cfg = get_config()
            # Find matching model in allowed list.
            for m in cfg.model.allowed:
                mid = m.id or f"{m.provider}/{m.model}"
                if mid == model_id or m.model == model_id:
                    return LLMProvider(
                        provider=m.provider,
                        model=m.model,
                        api_key=m.api_key if hasattr(m, 'api_key') else "",
                        base_url=m.base_url,
                        temperature=m.temperature if m.temperature is not None else cfg.model.temperature,
                        max_tokens=m.max_tokens if m.max_tokens is not None else cfg.model.max_tokens,
                    )

            # Try parsing as provider/model string.
            if "/" in model_id:
                provider, model = model_id.split("/", 1)
                return LLMProvider(
                    provider=provider,
                    model=model,
                    temperature=cfg.model.temperature,
                    max_tokens=cfg.model.max_tokens,
                )

        except Exception as exc:
            log.warning(
                "Failed to create provider for model %s, using default: %s",
                model_id, exc,
            )

        return self._provider

    # ── File transfer ─────────────────────────────────────────────

    def _encode_file_for_transfer(
        self, data: bytes, filename: str,
    ) -> tuple[str, bool]:
        """Encode file data: optionally gzip, then base64."""
        ext = Path(filename).suffix.lower()
        compressed = False
        payload = data

        if ext not in _PRECOMPRESSED_EXTS and len(data) > 256:
            try:
                gz = gzip.compress(data, compresslevel=6)
                if len(gz) < len(data):
                    payload = gz
                    compressed = True
            except Exception:
                pass

        return base64.b64encode(payload).decode("ascii"), compressed

    def _decode_file_from_transfer(
        self, b64_data: str, compressed: bool,
    ) -> bytes:
        """Decode file data from base64, optionally gunzip."""
        raw = base64.b64decode(b64_data)
        if compressed:
            raw = gzip.decompress(raw)
        return raw

    async def upload_file(
        self,
        swarm_id: str,
        concern_id: str,
        filepath: str,
        subfolder: str = "",
        timeout: float = 30.0,
    ) -> dict[str, Any]:
        """Upload a file to the BotPort swarm workspace."""
        if not self._connected or not self._ws:
            return {"ok": False, "error": "Not connected"}

        path = Path(filepath)
        if not path.is_file():
            return {"ok": False, "error": f"File not found: {filepath}"}

        data = path.read_bytes()
        if len(data) > 50 * 1024 * 1024:
            return {"ok": False, "error": "File too large (max 50 MB)"}

        b64_data, compressed = self._encode_file_for_transfer(data, path.name)

        # Create a future for the upload ack.
        upload_id = uuid.uuid4().hex
        loop = asyncio.get_event_loop()
        future: asyncio.Future[dict[str, Any]] = loop.create_future()
        self._file_upload_futures[upload_id] = future

        try:
            await self._ws.send_str(json.dumps({
                "type": "file_upload",
                "swarm_id": swarm_id,
                "concern_id": concern_id,
                "filename": path.name,
                "data": b64_data,
                "compressed": compressed,
                "mime_type": "",
                "subfolder": subfolder,
            }))

            result = await asyncio.wait_for(future, timeout=timeout)
            return result

        except (TimeoutError, asyncio.TimeoutError):
            return {"ok": False, "error": "Upload timed out"}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}
        finally:
            self._file_upload_futures.pop(upload_id, None)

    async def request_file(
        self,
        swarm_id: str,
        file_path: str = "",
        file_id: str = "",
        timeout: float = 30.0,
    ) -> tuple[bytes | None, dict[str, Any]]:
        """Request a file from the BotPort swarm workspace.

        Returns (file_data, metadata) or (None, error_dict).
        """
        if not self._connected or not self._ws:
            return None, {"ok": False, "error": "Not connected"}

        req_id = uuid.uuid4().hex
        loop = asyncio.get_event_loop()
        future: asyncio.Future[dict[str, Any]] = loop.create_future()
        self._file_response_futures[req_id] = future

        try:
            await self._ws.send_str(json.dumps({
                "type": "file_request",
                "swarm_id": swarm_id,
                "file_path": file_path,
                "file_id": file_id,
            }))

            result = await asyncio.wait_for(future, timeout=timeout)
            if not result.get("ok"):
                return None, result

            file_data = self._decode_file_from_transfer(
                result.get("data", ""), result.get("compressed", False),
            )
            return file_data, result

        except (TimeoutError, asyncio.TimeoutError):
            return None, {"ok": False, "error": "File request timed out"}
        except Exception as exc:
            return None, {"ok": False, "error": str(exc)}
        finally:
            self._file_response_futures.pop(req_id, None)

    async def list_swarm_files(
        self,
        swarm_id: str,
        agent_filter: str = "",
        timeout: float = 10.0,
    ) -> list[dict[str, Any]]:
        """List available files in a swarm workspace."""
        if not self._connected or not self._ws:
            return []

        req_id = uuid.uuid4().hex
        loop = asyncio.get_event_loop()
        future: asyncio.Future[dict[str, Any]] = loop.create_future()
        self._file_response_futures[req_id] = future

        try:
            await self._ws.send_str(json.dumps({
                "type": "file_list",
                "swarm_id": swarm_id,
                "agent_filter": agent_filter,
            }))

            result = await asyncio.wait_for(future, timeout=timeout)
            return result.get("files", [])

        except (TimeoutError, asyncio.TimeoutError):
            return []
        except Exception:
            return []
        finally:
            self._file_response_futures.pop(req_id, None)

    def _handle_file_upload_ack(self, data: dict[str, Any]) -> None:
        """Handle file upload acknowledgment from BotPort."""
        # Resolve the oldest pending upload future.
        if self._file_upload_futures:
            key = next(iter(self._file_upload_futures))
            future = self._file_upload_futures.pop(key, None)
            if future and not future.done():
                future.set_result(data)

    def _handle_file_response(self, data: dict[str, Any]) -> None:
        """Handle file response (download) from BotPort."""
        if self._file_response_futures:
            key = next(iter(self._file_response_futures))
            future = self._file_response_futures.pop(key, None)
            if future and not future.done():
                future.set_result(data)

    def _handle_file_list_response(self, data: dict[str, Any]) -> None:
        """Handle file list response from BotPort."""
        if self._file_response_futures:
            key = next(iter(self._file_response_futures))
            future = self._file_response_futures.pop(key, None)
            if future and not future.done():
                future.set_result(data)

    async def _prefetch_swarm_files(
        self,
        swarm_id: str,
        swarm_files: list[dict[str, Any]],
    ) -> dict[str, str]:
        """Download swarm files from BotPort and write them to local disk.

        Returns ``{swarm_rel_path: local_abs_path}`` for successfully fetched files.
        The files are placed under ``<workspace>/swarm_files/<rel_path>`` so the
        agent can read them with the normal ``read`` tool.
        """
        if not swarm_files or not swarm_id:
            return {}

        # Determine local root for swarm files.
        try:
            from captain_claw.config import get_config
            ws_root = Path(get_config().resolved_workspace_path())
        except Exception:
            ws_root = Path.cwd() / "workspace"
        swarm_dir = ws_root / "swarm_files"

        fetched: dict[str, str] = {}
        for finfo in swarm_files:
            rel_path = finfo.get("path", "")
            if not rel_path:
                continue

            local_path = swarm_dir / rel_path
            # Skip if already fetched (from a prior task in the same run).
            if local_path.is_file():
                fetched[rel_path] = str(local_path)
                continue

            try:
                file_data, meta = await self.request_file(
                    swarm_id=swarm_id,
                    file_path=rel_path,
                )
                if file_data is None:
                    log.warning(
                        "Failed to fetch swarm file %s: %s",
                        rel_path, meta.get("error", "unknown"),
                    )
                    continue

                local_path.parent.mkdir(parents=True, exist_ok=True)
                local_path.write_bytes(file_data)
                fetched[rel_path] = str(local_path)
                log.info("Pre-fetched swarm file: %s -> %s", rel_path, local_path)
            except Exception as exc:
                log.warning("Error fetching swarm file %s: %s", rel_path, exc)

        return fetched

    async def _upload_created_files(self, concern_id: str) -> list[dict[str, Any]]:
        """Upload all files created during a dispatch to the swarm workspace."""
        swarm_id = self._dispatch_swarm_ids.get(concern_id, "")
        created = self._dispatch_created_files.get(concern_id, [])

        if not swarm_id or not created:
            return []

        # Deduplicate: resolve to absolute paths and keep first occurrence.
        seen: set[str] = set()
        unique_paths: list[str] = []
        for fp in created:
            resolved = str(Path(fp).resolve())
            if resolved not in seen:
                seen.add(resolved)
                unique_paths.append(fp)

        uploaded = []
        for filepath in unique_paths:
            path = Path(filepath)
            if not path.is_file():
                continue

            # Skip very large files.
            if path.stat().st_size > 50 * 1024 * 1024:
                log.warning("Skipping large file: %s (%d bytes)", path.name, path.stat().st_size)
                continue

            try:
                result = await self.upload_file(
                    swarm_id=swarm_id,
                    concern_id=concern_id,
                    filepath=str(path),
                    timeout=60.0,
                )
                if result.get("ok"):
                    uploaded.append({
                        "filename": result.get("filename", path.name),
                        "path": result.get("path", ""),
                        "size": result.get("size", 0),
                    })
                    log.info("Uploaded file: %s for swarm %s", path.name, swarm_id[:8])
                else:
                    log.warning(
                        "Failed to upload %s: %s", path.name, result.get("error", ""),
                    )
            except Exception as exc:
                log.error("Error uploading %s: %s", path.name, exc)

        return uploaded

    def _track_file_creation(
        self, concern_id: str, tool_name: str, arguments: dict[str, Any], output: str,
    ) -> None:
        """Detect file creation from tool outputs and track for upload.

        The Write tool resolves relative paths internally (e.g. ``saved/showcase/...``
        becomes ``/abs/workspace/saved/showcase/...``).  The original ``arguments``
        dict contains the LLM-provided *relative* path which won't resolve from CWD.
        Instead we parse the tool **output** which always contains the resolved
        absolute path in the form ``Written N chars (...) to /absolute/path``.
        """
        if concern_id not in self._dispatch_created_files:
            return

        tracked = self._dispatch_created_files[concern_id]

        def _add(fp: str) -> None:
            resolved = str(Path(fp).resolve())
            # Skip pre-fetched swarm files — already on the server.
            if "/swarm_files/" in resolved:
                return
            if not any(str(Path(p).resolve()) == resolved for p in tracked):
                tracked.append(fp)
                log.debug("Tracked file creation: %s", fp)

        # Detect Write / Edit tool — extract resolved path from output.
        if tool_name.lower() in ("write", "write_file", "file_write",
                                   "edit", "edit_file", "file_edit"):
            # Primary: parse the absolute path from tool output.
            # Write output: "Written N chars (M lines) to /absolute/path"
            # Edit output may vary, but often contains the path.
            filepath = self._extract_path_from_tool_output(output)
            if filepath and Path(filepath).is_file():
                _add(filepath)
                return

            # Fallback: try the argument path directly.
            filepath = arguments.get("file_path", "") or arguments.get("path", "")
            if filepath:
                p = Path(filepath)
                if p.is_file():
                    _add(str(p.resolve()))

        # Detect bash commands that might create files.
        elif tool_name.lower() in ("bash", "shell", "execute"):
            cmd = arguments.get("command", "") or arguments.get("cmd", "")
            if ">" in cmd or "tee " in cmd:
                for part in cmd.split(">"):
                    if part == cmd.split(">")[0]:
                        continue
                    filepath = part.strip().split()[0] if part.strip() else ""
                    if filepath:
                        p = Path(filepath)
                        if p.is_file():
                            _add(str(p.resolve()))

    @staticmethod
    def _extract_path_from_tool_output(output: str) -> str:
        """Extract the resolved file path from a Write/Edit tool output string.

        Handles patterns like:
          "Written 9587 chars (150 lines) to /abs/path/file.md"
          "Written 9587 chars (150 lines) to /abs/path/file.md (requested: saved/...)"
        """
        if not output:
            return ""
        # Match "to <path>" — path is everything after "to " until end,
        # optional parens, or newline.
        m = re.search(r'\bto\s+(/[^\s()\n]+)', output)
        if m:
            return m.group(1).rstrip(".,;:!?")
        return ""

    def _extract_files_from_response(self, concern_id: str, response: str) -> None:
        """Scan agent response text for file paths that exist on disk.

        Agents may reference files created in previous sessions that already
        exist in their workspace.  These won't be caught by tool-output
        tracking, so we parse the response for path-like strings and check
        if they point to real files.
        """
        if concern_id not in self._dispatch_created_files:
            return

        # Build dedup set using resolved absolute paths so relative vs
        # absolute variants of the same file are recognized as duplicates.
        already = set(str(Path(p).resolve()) for p in self._dispatch_created_files[concern_id])

        # Match path-like strings: sequences with at least one "/" and a file
        # extension, or absolute paths.  Covers patterns like:
        #   saved/showcase/abc123/report.md
        #   /home/user/output.pdf
        #   ./results/chart.html
        path_pattern = re.compile(
            r'(?:^|[\s`\'"(])('                   # preceded by whitespace/quote/tick/paren
            r'(?:[./~]|[a-zA-Z]:[\\/])'           # starts with . / ~ or drive letter
            r'[^\s`\'"<>|*?\n]+'                   # path characters (no spaces/special)
            r')',
            re.MULTILINE,
        )

        for match in path_pattern.finditer(response):
            candidate = match.group(1).rstrip(".,;:!?)]}'\"")
            if not candidate:
                continue

            path = Path(candidate).expanduser()

            # Try multiple base directories for relative paths:
            # 1. As-is (if absolute)
            # 2. Relative to CWD
            # 3. Relative to workspace root (CWD/workspace)
            candidates: list[Path] = []
            if path.is_absolute():
                candidates.append(path)
            else:
                candidates.append(Path.cwd() / path)
                candidates.append(Path.cwd() / "workspace" / path)
                # Also try workspace config path if available.
                try:
                    from captain_claw.config import get_config
                    ws = get_config().resolved_workspace_path()
                    candidates.append(ws / path)
                except Exception:
                    pass

            for cand in candidates:
                try:
                    resolved = str(cand.resolve())
                    # Skip pre-fetched swarm files — they're already on the server.
                    if "/swarm_files/" in resolved:
                        break
                    if cand.is_file() and resolved not in already:
                        if cand.stat().st_size <= 50 * 1024 * 1024:
                            self._dispatch_created_files[concern_id].append(str(cand.resolve()))
                            already.add(resolved)
                            log.debug("Extracted file from response: %s", cand)
                        break
                except (OSError, ValueError):
                    continue

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
