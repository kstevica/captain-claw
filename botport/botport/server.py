"""BotPort server - agent-to-agent task routing hub."""

from __future__ import annotations

import asyncio
import logging
import signal
import sys
import uuid
from pathlib import Path
from typing import Any

import yaml
from aiohttp import web

from botport import __version__
from botport.auth import validate_credentials
from botport.concern_manager import ConcernManager
from botport.config import BotPortConfig, get_config, set_config
from botport.connection_manager import ConnectionManager
from botport.models import _utcnow_iso
from botport.protocol import (
    ActivityMessage,
    BaseMessage,
    ConcernAckMessage,
    ConcernClosedMessage,
    ConcernResultMessage,
    ConcernSubmitMessage,
    CloseConcernMessage,
    ContextReplyMessage,
    ContextRequestMessage,
    DispatchMessage,
    FileListMessage,
    FileListResponseMessage,
    FileRequestMessage,
    FileResponseMessage,
    FileUploadAckMessage,
    FileUploadMessage,
    FollowUpMessage,
    HeartbeatAckMessage,
    HeartbeatMessage,
    RegisterMessage,
    RegisteredMessage,
    ResultMessage,
    TimeoutNoticeMessage,
    parse_raw,
    serialize_message,
)
from botport.swarm.file_manager import FileManager, decode_file, encode_file
from botport.registry import Registry
from botport.router import RouteResult, Router
from botport.store import BotPortStore

log = logging.getLogger(__name__)


class BotPortServer:
    """Main BotPort server orchestrating all components."""

    def __init__(self, config: BotPortConfig) -> None:
        self.config = config
        self.store = BotPortStore()
        self.connections = ConnectionManager()
        self.registry = Registry(self.connections)
        self.router = Router(self.registry)
        self.concerns = ConcernManager(self.store)
        self.file_manager = FileManager()
        self._swarm_store_initialized = False
        self.swarm_engine: "SwarmEngine | None" = None
        self.swarm_scheduler: "SwarmScheduler | None" = None

    def create_app(self) -> web.Application:
        app = web.Application()

        # WebSocket endpoint.
        app.router.add_get("/ws", self._ws_handler)

        # Dashboard.
        if self.config.server.dashboard_enabled:
            from botport.dashboard.routes import setup_dashboard_routes
            from botport.dashboard.swarm_routes import setup_swarm_routes
            setup_dashboard_routes(app, self)
            setup_swarm_routes(app, self)

        return app

    @property
    def swarm_store(self):
        """Lazy accessor for the swarm store (initialized with DB)."""
        return self.store.swarm

    # ── WebSocket handler ────────────────────────────────────────

    async def _ws_handler(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse(
            heartbeat=60.0,
            max_msg_size=16 * 1024 * 1024,  # 16 MB for file transfers
        )
        await ws.prepare(request)

        instance_id: str | None = None
        try:
            # First message must be a register message.
            async for raw_msg in ws:
                if raw_msg.type == web.WSMsgType.TEXT:
                    try:
                        msg = parse_raw(raw_msg.data)
                    except ValueError as exc:
                        log.warning("Invalid message from new connection: %s", exc)
                        await ws.send_str(serialize_message(
                            RegisteredMessage(ok=False, error=str(exc))
                        ))
                        continue

                    if isinstance(msg, RegisterMessage):
                        instance_id = await self._handle_register(ws, msg)
                        if instance_id:
                            break
                    else:
                        await ws.send_str(serialize_message(
                            RegisteredMessage(ok=False, error="First message must be 'register'")
                        ))

                elif raw_msg.type in (web.WSMsgType.ERROR, web.WSMsgType.CLOSE):
                    break

            if not instance_id:
                return ws

            # Main message loop.
            async for raw_msg in ws:
                if raw_msg.type == web.WSMsgType.TEXT:
                    try:
                        msg = parse_raw(raw_msg.data)
                        await self._handle_message(instance_id, msg)
                    except ValueError as exc:
                        log.warning("Invalid message from %s: %s", instance_id[:8], exc)
                    except Exception as exc:
                        log.error("Handler error for %s: %s", instance_id[:8], exc)

                elif raw_msg.type in (web.WSMsgType.ERROR, web.WSMsgType.CLOSE):
                    break

        except asyncio.CancelledError:
            pass
        except Exception as exc:
            log.error("WebSocket error: %s", exc)
        finally:
            if instance_id:
                await self._handle_disconnect(instance_id)

        return ws

    # ── Registration ─────────────────────────────────────────────

    async def _handle_register(
        self, ws: web.WebSocketResponse, msg: RegisterMessage,
    ) -> str | None:
        """Handle instance registration. Returns instance_id or None."""
        # Validate credentials.
        if not validate_credentials(msg.key, msg.secret):
            await ws.send_str(serialize_message(
                RegisteredMessage(ok=False, error="Authentication failed")
            ))
            log.warning("Auth failed for instance: %s", msg.instance_name)
            return None

        instance_id = str(uuid.uuid4())
        info = await self.connections.register(ws, msg, instance_id)

        await ws.send_str(serialize_message(RegisteredMessage(
            instance_id=instance_id,
            botport_version=__version__,
            ok=True,
        )))

        log.info(
            "Instance registered: %s (id=%s, personas=%d, tools=%d)",
            info.name, instance_id[:8], len(info.personas), len(info.tools),
        )
        return instance_id

    async def _handle_disconnect(self, instance_id: str) -> None:
        """Handle instance disconnection."""
        self.connections.clear_activity(instance_id)
        info = await self.connections.unregister(instance_id)
        name = info.name if info else instance_id[:8]

        # Fail all active concerns assigned to this instance.
        failed = await self.concerns.fail_concerns_for_instance(instance_id)

        # Notify originators of failed concerns.
        for concern_id in failed:
            concern = self.concerns.get_concern(concern_id)
            if concern:
                await self.connections.send_to(
                    concern.from_instance,
                    ConcernResultMessage(
                        concern_id=concern_id,
                        ok=False,
                        error=f"Assigned instance '{name}' disconnected",
                    ),
                )

        if failed:
            log.warning(
                "Instance %s disconnected, failed %d concerns",
                name, len(failed),
            )
        else:
            log.info("Instance %s disconnected", name)

    # ── Message dispatch ─────────────────────────────────────────

    async def _handle_message(self, instance_id: str, msg: BaseMessage) -> None:
        if isinstance(msg, HeartbeatMessage):
            await self._handle_heartbeat(instance_id, msg)
        elif isinstance(msg, ActivityMessage):
            self._handle_activity(instance_id, msg)
        elif isinstance(msg, ConcernSubmitMessage):
            await self._handle_concern(instance_id, msg)
        elif isinstance(msg, ResultMessage):
            await self._handle_result(instance_id, msg)
        elif isinstance(msg, FollowUpMessage):
            await self._handle_follow_up(instance_id, msg)
        elif isinstance(msg, ContextRequestMessage):
            await self._handle_context_request(instance_id, msg)
        elif isinstance(msg, ContextReplyMessage):
            await self._handle_context_reply(instance_id, msg)
        elif isinstance(msg, CloseConcernMessage):
            await self._handle_close_concern(instance_id, msg)
        elif isinstance(msg, FileUploadMessage):
            await self._handle_file_upload(instance_id, msg)
        elif isinstance(msg, FileListMessage):
            await self._handle_file_list(instance_id, msg)
        elif isinstance(msg, FileRequestMessage):
            await self._handle_file_request(instance_id, msg)
        else:
            log.warning("Unhandled message type from %s: %s", instance_id[:8], msg.type)

    # ── Activity ─────────────────────────────────────────────────

    def _handle_activity(self, instance_id: str, msg: ActivityMessage) -> None:
        """Handle live activity update from an agent. Ephemeral, in-memory only."""
        self.connections.update_activity(instance_id, msg.step_type, dict(msg.data))

    # ── Heartbeat ────────────────────────────────────────────────

    async def _handle_heartbeat(self, instance_id: str, msg: HeartbeatMessage) -> None:
        self.connections.update_heartbeat(instance_id, msg.active_concerns)
        await self.connections.send_to(instance_id, HeartbeatAckMessage(
            connected_instances=self.connections.connected_count,
        ))

    # ── Concern flow ─────────────────────────────────────────────

    async def _handle_concern(self, instance_id: str, msg: ConcernSubmitMessage) -> None:
        """CC-A submits a concern -> route to CC-B."""
        from_info = self.connections.get_instance(instance_id)
        from_name = from_info.name if from_info else instance_id[:8]

        # Deduplication: reject if same instance already has an active concern
        # with the exact same task text.
        for existing in self.concerns.get_concerns_from_instance(instance_id):
            if existing.is_active and existing.task == msg.task:
                await self.connections.send_to(instance_id, ConcernAckMessage(
                    concern_id=msg.concern_id or existing.id,
                    ok=False,
                    error="Duplicate concern — identical task already active",
                ))
                return

        concern = await self.concerns.create_concern(
            from_instance=instance_id,
            task=msg.task,
            context=dict(msg.context),
            expertise_tags=list(msg.expertise_tags),
            from_session=msg.from_session,
        )
        concern.from_instance_name = from_name

        # Override concern ID if client provided one (for tracking).
        if msg.concern_id:
            self.concerns._concerns.pop(concern.id, None)
            concern.id = msg.concern_id
            self.concerns._concerns[concern.id] = concern

        await self.store.save_concern(concern)

        # Route to best instance.
        route_result = await self.router.route(concern, exclude_instance=instance_id)

        if route_result is None:
            await self.concerns.fail_concern(concern.id, reason="no_available_instance")
            await self.connections.send_to(instance_id, ConcernAckMessage(
                concern_id=concern.id,
                ok=False,
                error="No available instance to handle this concern",
            ))
            return

        target = route_result.instance
        persona_hint = route_result.persona_name

        # Assign and dispatch.
        concern.assigned_instance_name = target.name
        await self.concerns.assign_concern(concern.id, target.id, target.name)
        self.connections.increment_active(target.id)

        # Send ack to CC-A.
        await self.connections.send_to(instance_id, ConcernAckMessage(
            concern_id=concern.id,
            assigned_to_name=target.name,
            ok=True,
        ))

        # Dispatch to CC-B.
        await self.connections.send_to(target.id, DispatchMessage(
            concern_id=concern.id,
            from_instance_name=from_name,
            task=concern.task,
            context=concern.context,
            persona_hint=persona_hint,
        ))

        log.info(
            "Concern %s routed: %s -> %s (persona: %s, reason: %s)",
            concern.id[:8], from_name, target.name,
            persona_hint or "auto", route_result.reason,
        )

    async def _handle_result(self, instance_id: str, msg: ResultMessage) -> None:
        """CC-B sends result -> relay to CC-A."""
        concern = self.concerns.get_concern(msg.concern_id)
        if not concern:
            log.warning("Result for unknown concern %s", msg.concern_id[:8])
            return

        self.connections.decrement_active(instance_id)

        if not msg.ok:
            await self.concerns.fail_concern(concern.id, reason=msg.error or "agent_error")
            await self.connections.send_to(concern.from_instance, ConcernResultMessage(
                concern_id=concern.id,
                ok=False,
                error=msg.error,
            ))
            return

        # Store persona name in concern metadata for dashboard.
        result_metadata = dict(msg.metadata)
        if msg.persona_name:
            result_metadata["persona_name"] = msg.persona_name
            concern.metadata["persona_name"] = msg.persona_name

        await self.concerns.record_result(
            concern.id, msg.response, result_metadata, from_instance=instance_id,
        )

        from_info = self.connections.get_instance(instance_id)
        from_name = from_info.name if from_info else instance_id[:8]

        await self.connections.send_to(concern.from_instance, ConcernResultMessage(
            concern_id=concern.id,
            response=msg.response,
            from_instance_name=from_name,
            persona_name=msg.persona_name,
            metadata=result_metadata,
            ok=True,
        ))

        log.info(
            "Result relayed: concern=%s persona=%s",
            concern.id[:8], msg.persona_name or "unknown",
        )

    # ── Follow-ups ───────────────────────────────────────────────

    async def _handle_follow_up(self, instance_id: str, msg: FollowUpMessage) -> None:
        """Route follow-up from CC-A to CC-B or vice versa."""
        concern = self.concerns.get_concern(msg.concern_id)
        if not concern or concern.is_terminal:
            log.warning("Follow-up for invalid concern %s", msg.concern_id[:8])
            return

        await self.concerns.add_follow_up(
            concern.id, msg.message, from_instance=instance_id,
            additional_context=dict(msg.additional_context),
        )

        # Determine target: if from CC-A, send to CC-B; if from CC-B, send to CC-A.
        if instance_id == concern.from_instance and concern.assigned_instance:
            target_id = concern.assigned_instance
        elif instance_id == concern.assigned_instance:
            target_id = concern.from_instance
        else:
            log.warning("Follow-up from unrelated instance %s", instance_id[:8])
            return

        await self.connections.send_to(target_id, FollowUpMessage(
            concern_id=concern.id,
            message=msg.message,
            additional_context=msg.additional_context,
        ))

    # ── Context negotiation ──────────────────────────────────────

    async def _handle_context_request(
        self, instance_id: str, msg: ContextRequestMessage,
    ) -> None:
        """CC-B asks for more context -> forward to CC-A."""
        concern = self.concerns.get_concern(msg.concern_id)
        if not concern or concern.is_terminal:
            return

        await self.concerns.add_context_request(
            concern.id, msg.questions, from_instance=instance_id,
        )

        await self.connections.send_to(concern.from_instance, ContextRequestMessage(
            concern_id=concern.id,
            questions=msg.questions,
        ))

    async def _handle_context_reply(
        self, instance_id: str, msg: ContextReplyMessage,
    ) -> None:
        """CC-A replies with context -> forward to CC-B."""
        concern = self.concerns.get_concern(msg.concern_id)
        if not concern or concern.is_terminal or not concern.assigned_instance:
            return

        await self.concerns.add_context_reply(
            concern.id, msg.answers, from_instance=instance_id,
        )

        await self.connections.send_to(concern.assigned_instance, ContextReplyMessage(
            concern_id=concern.id,
            answers=msg.answers,
        ))

    # ── Lifecycle ────────────────────────────────────────────────

    async def _handle_close_concern(
        self, instance_id: str, msg: CloseConcernMessage,
    ) -> None:
        """CC-A closes a concern."""
        concern = self.concerns.get_concern(msg.concern_id)
        if not concern or concern.is_terminal:
            return

        await self.concerns.close_concern(concern.id, reason="closed_by_originator")

        if concern.assigned_instance:
            self.connections.decrement_active(concern.assigned_instance)
            await self.connections.send_to(concern.assigned_instance, ConcernClosedMessage(
                concern_id=concern.id,
                reason="closed_by_originator",
            ))

    # ── File transfer ──────────────────────────────────────────

    async def _handle_file_upload(
        self, instance_id: str, msg: FileUploadMessage,
    ) -> None:
        """Handle file upload from an agent."""
        info = self.connections.get_instance(instance_id)
        agent_name = info.name if info else instance_id[:8]

        if not msg.swarm_id or not msg.filename or not msg.data:
            await self.connections.send_to(instance_id, FileUploadAckMessage(
                ok=False, error="Missing swarm_id, filename, or data",
            ))
            return

        try:
            file_data = decode_file(msg.data, msg.compressed)
            meta = self.file_manager.store_file(
                swarm_id=msg.swarm_id,
                agent_name=agent_name,
                filename=msg.filename,
                data=file_data,
                subfolder=msg.subfolder,
            )

            await self.connections.send_to(instance_id, FileUploadAckMessage(
                file_id=meta["file_id"],
                filename=meta["filename"],
                path=meta["path"],
                size=meta["size"],
                ok=True,
            ))

            log.info(
                "File uploaded: %s (%d bytes) from %s for swarm %s",
                meta["filename"], meta["size"], agent_name, msg.swarm_id[:8],
            )

        except Exception as exc:
            log.error("File upload failed: %s", exc)
            await self.connections.send_to(instance_id, FileUploadAckMessage(
                ok=False, error=str(exc),
            ))

    async def _handle_file_list(
        self, instance_id: str, msg: FileListMessage,
    ) -> None:
        """Handle file list request."""
        if not msg.swarm_id:
            await self.connections.send_to(instance_id, FileListResponseMessage(
                swarm_id="", files=[],
            ))
            return

        files = self.file_manager.list_files(
            msg.swarm_id, agent_name=msg.agent_filter,
        )

        # Strip modified_at (float) for JSON serialization — not needed by client.
        for f in files:
            f.pop("modified_at", None)

        await self.connections.send_to(instance_id, FileListResponseMessage(
            swarm_id=msg.swarm_id,
            files=files,
        ))

    async def _handle_file_request(
        self, instance_id: str, msg: FileRequestMessage,
    ) -> None:
        """Handle file download request."""
        if not msg.swarm_id:
            await self.connections.send_to(instance_id, FileResponseMessage(
                ok=False, error="Missing swarm_id",
            ))
            return

        file_data: bytes | None = None
        filename = ""
        rel_path = ""
        mime = ""

        if msg.file_path:
            file_data = self.file_manager.get_file(msg.swarm_id, msg.file_path)
            rel_path = msg.file_path
            filename = Path(msg.file_path).name
        elif msg.file_id:
            file_data, meta = self.file_manager.get_file_by_id(
                msg.swarm_id, msg.file_id,
            )
            if meta:
                filename = meta.get("filename", "")
                rel_path = meta.get("path", "")
                mime = meta.get("mime_type", "")

        if file_data is None:
            await self.connections.send_to(instance_id, FileResponseMessage(
                swarm_id=msg.swarm_id,
                ok=False,
                error="File not found",
            ))
            return

        from botport.swarm.file_manager import guess_mime_type
        if not mime:
            mime = guess_mime_type(filename)

        b64_data, compressed = encode_file(file_data, filename)

        await self.connections.send_to(instance_id, FileResponseMessage(
            swarm_id=msg.swarm_id,
            filename=filename,
            path=rel_path,
            data=b64_data,
            compressed=compressed,
            mime_type=mime,
            size=len(file_data),
            ok=True,
        ))

        log.info(
            "File served: %s (%d bytes, compressed=%s) for swarm %s",
            filename, len(file_data), compressed, msg.swarm_id[:8],
        )

    async def shutdown(self) -> None:
        """Graceful shutdown."""
        if self.swarm_scheduler:
            self.swarm_scheduler.stop()
            await self.swarm_scheduler.wait_stopped()
        if self.swarm_engine:
            self.swarm_engine.stop()
            await self.swarm_engine.wait_stopped()
        self.concerns.stop_timeout_checker()
        # Close all WebSocket connections so the runner can exit quickly.
        for instance_id in list(self.connections._connections):
            ws = self.connections._connections.get(instance_id)
            if ws is not None and not ws.closed:
                try:
                    await ws.close()
                except Exception:
                    pass
        await self.router.close()
        await self.store.close()
        log.info("BotPort server shut down.")


# ── Logging ──────────────────────────────────────────────────────

# ANSI colors matching Captain Claw's structlog ConsoleRenderer.
_RST = "\033[0m"
_BRIGHT = "\033[1m"
_DIM = "\033[2m"
_RED = "\033[31m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_BLUE = "\033[34m"
_CYAN = "\033[36m"
_MAGENTA = "\033[35m"

_LEVEL_COLORS = {
    "DEBUG": _DIM + _GREEN,
    "INFO": _GREEN,
    "WARNING": _YELLOW,
    "ERROR": _RED,
    "CRITICAL": _RED + _BRIGHT,
}


class _ColorFormatter(logging.Formatter):
    """Colored console formatter à la structlog ConsoleRenderer."""

    def format(self, record: logging.LogRecord) -> str:
        ts = self.formatTime(record, "%Y-%m-%dT%H:%M:%S")
        lvl = record.levelname
        color = _LEVEL_COLORS.get(lvl, "")
        name = record.name.replace("botport.", "")
        msg = record.getMessage()
        line = (
            f"{_DIM}{ts}{_RST} "
            f"[{color}{lvl:<8s}{_RST}] "
            f"{_BLUE}{name}{_RST}  "
            f"{_BRIGHT}{msg}{_RST}"
        )
        if record.exc_info and record.exc_info[1]:
            line += "\n" + self.formatException(record.exc_info)
        return line


def _setup_logging(config: BotPortConfig) -> None:
    """Configure colored logging, suppress noisy aiohttp/ws chatter."""
    log_level = getattr(logging, config.logging.level.upper(), logging.INFO)

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(_ColorFormatter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(log_level)

    # Suppress noisy aiohttp access logs and websocket internals.
    for noisy in ("aiohttp.access", "aiohttp.server", "aiohttp.web", "aiohttp"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


# ── Entry point ──────────────────────────────────────────────────


def main() -> None:
    """Start the BotPort server."""
    import argparse

    parser = argparse.ArgumentParser(description="BotPort - Agent routing hub")
    parser.add_argument("--config", "-c", default="", help="Path to config YAML")
    parser.add_argument("--host", default="", help="Bind host")
    parser.add_argument("--port", "-p", type=int, default=0, help="Bind port")
    args = parser.parse_args()

    # Load config.
    if args.config:
        config = BotPortConfig.from_yaml(args.config)
    else:
        config = BotPortConfig.load()

    if args.host:
        config.server.host = args.host
    if args.port:
        config.server.port = args.port

    set_config(config)

    # Configure logging with colored output.
    _setup_logging(config)

    try:
        asyncio.run(_run_server(config))
    except KeyboardInterrupt:
        print("\nBotPort stopped.")
        sys.exit(0)


async def _run_server(config: BotPortConfig) -> None:
    """Async server lifecycle."""
    server = BotPortServer(config)
    app = server.create_app()
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()

    host = config.server.host
    port = config.server.port

    # Try binding, retry on port conflict.
    site: web.TCPSite | None = None
    for attempt in range(5):
        try:
            site = web.TCPSite(runner, host, port)
            await site.start()
            break
        except OSError:
            port += 1

    if site is None:
        log.error("Failed to bind to any port")
        return

    # Start background tasks.
    server.concerns.start_timeout_checker()

    # Start swarm engine and scheduler (ensure DB is initialized first).
    await server.store._ensure_db()
    from botport.swarm.engine import SwarmEngine
    from botport.swarm.scheduler import SwarmScheduler
    server.swarm_engine = SwarmEngine(server)
    server.swarm_engine.start()
    server.swarm_scheduler = SwarmScheduler(server)
    server.swarm_scheduler.start()

    # Periodic store cleanup.
    async def _cleanup_loop() -> None:
        while True:
            try:
                await asyncio.sleep(3600)  # hourly
                deleted = await server.store.cleanup_old(config.logging.retention_days)
                if deleted:
                    log.info("Cleaned up %d old concerns", deleted)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                log.error("Cleanup error: %s", exc)

    cleanup_task = asyncio.create_task(_cleanup_loop())

    # Purge disconnected instances after 60s.
    async def _purge_loop() -> None:
        while True:
            try:
                await asyncio.sleep(15)
                purged = server.connections.purge_stale(max_age_seconds=60.0)
                for iid in purged:
                    log.info("Purged stale instance %s", iid[:8])
            except asyncio.CancelledError:
                break
            except Exception as exc:
                log.error("Purge error: %s", exc)

    purge_task = asyncio.create_task(_purge_loop())

    print(f"BotPort v{__version__} running on http://{host}:{port}")
    if config.server.dashboard_enabled:
        print(f"  Dashboard: http://{host}:{port}/")
    print(f"  WebSocket: ws://{host}:{port}/ws")
    llm_info = f"LLM routing: {config.llm.model}" if config.llm.enabled else "LLM routing: disabled"
    print(f"  Routing: tags -> {'LLM' if config.llm.enabled else 'skip'} -> fallback  ({llm_info})")
    if config.auth.enabled:
        print(f"  Auth: enabled ({len(config.auth.keys)} keys)")
    else:
        print("  Auth: disabled (open)")

    # Wait for shutdown signal.
    stop_event = asyncio.Event()
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop_event.set)
        except (NotImplementedError, OSError):
            pass

    await stop_event.wait()

    print("\nShutting down BotPort...")
    cleanup_task.cancel()
    purge_task.cancel()
    for t in (cleanup_task, purge_task):
        try:
            await t
        except asyncio.CancelledError:
            pass

    await server.shutdown()
    try:
        await asyncio.wait_for(runner.cleanup(), timeout=3.0)
    except asyncio.TimeoutError:
        pass


if __name__ == "__main__":
    main()
