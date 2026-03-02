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
    BaseMessage,
    ConcernAckMessage,
    ConcernClosedMessage,
    ConcernResultMessage,
    ConcernSubmitMessage,
    CloseConcernMessage,
    ContextReplyMessage,
    ContextRequestMessage,
    DispatchMessage,
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
from botport.registry import Registry
from botport.router import Router
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

    def create_app(self) -> web.Application:
        app = web.Application()

        # WebSocket endpoint.
        app.router.add_get("/ws", self._ws_handler)

        # Dashboard.
        if self.config.server.dashboard_enabled:
            from botport.dashboard.routes import setup_dashboard_routes
            setup_dashboard_routes(app, self)

        return app

    # ── WebSocket handler ────────────────────────────────────────

    async def _ws_handler(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse(
            heartbeat=60.0,
            max_msg_size=4 * 1024 * 1024,
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
            log.info(
                "Instance %s disconnected, failed %d concerns",
                name, len(failed),
            )
        else:
            log.info("Instance %s disconnected", name)

    # ── Message dispatch ─────────────────────────────────────────

    async def _handle_message(self, instance_id: str, msg: BaseMessage) -> None:
        if isinstance(msg, HeartbeatMessage):
            await self._handle_heartbeat(instance_id, msg)
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
        else:
            log.warning("Unhandled message type from %s: %s", instance_id[:8], msg.type)

    # ── Heartbeat ────────────────────────────────────────────────

    async def _handle_heartbeat(self, instance_id: str, msg: HeartbeatMessage) -> None:
        self.connections.update_heartbeat(instance_id, msg.active_concerns)
        await self.connections.send_to(instance_id, HeartbeatAckMessage(
            connected_instances=self.connections.connected_count,
        ))

    # ── Concern flow ─────────────────────────────────────────────

    async def _handle_concern(self, instance_id: str, msg: ConcernSubmitMessage) -> None:
        """CC-A submits a concern -> route to CC-B."""
        concern = await self.concerns.create_concern(
            from_instance=instance_id,
            task=msg.task,
            context=dict(msg.context),
            expertise_tags=list(msg.expertise_tags),
            from_session=msg.from_session,
        )

        # Override concern ID if client provided one (for tracking).
        if msg.concern_id:
            # Re-key in the manager.
            self.concerns._concerns.pop(concern.id, None)
            concern.id = msg.concern_id
            self.concerns._concerns[concern.id] = concern
            await self.store.save_concern(concern)

        # Route to best instance.
        target = await self.router.route(concern, exclude_instance=instance_id)

        if target is None:
            # No instance available.
            await self.concerns.fail_concern(concern.id, reason="no_available_instance")
            await self.connections.send_to(instance_id, ConcernAckMessage(
                concern_id=concern.id,
                ok=False,
                error="No available instance to handle this concern",
            ))
            return

        # Assign and dispatch.
        await self.concerns.assign_concern(concern.id, target.id, target.name)
        self.connections.increment_active(target.id)

        # Determine persona hint (best matching persona name).
        persona_hint = self._pick_persona_hint(target, concern.expertise_tags)

        # Send ack to CC-A.
        await self.connections.send_to(instance_id, ConcernAckMessage(
            concern_id=concern.id,
            assigned_to_name=target.name,
            ok=True,
        ))

        # Dispatch to CC-B.
        from_info = self.connections.get_instance(instance_id)
        from_name = from_info.name if from_info else instance_id[:8]

        await self.connections.send_to(target.id, DispatchMessage(
            concern_id=concern.id,
            from_instance_name=from_name,
            task=concern.task,
            context=concern.context,
            persona_hint=persona_hint,
        ))

        log.info(
            "Concern %s routed: %s -> %s (persona: %s)",
            concern.id[:8], from_name, target.name, persona_hint or "auto",
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

        await self.concerns.record_result(
            concern.id, msg.response, msg.metadata, from_instance=instance_id,
        )

        from_info = self.connections.get_instance(instance_id)
        from_name = from_info.name if from_info else instance_id[:8]

        await self.connections.send_to(concern.from_instance, ConcernResultMessage(
            concern_id=concern.id,
            response=msg.response,
            from_instance_name=from_name,
            metadata=msg.metadata,
            ok=True,
        ))

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

    # ── Helpers ───────────────────────────────────────────────────

    def _pick_persona_hint(
        self, instance: Any, expertise_tags: list[str],
    ) -> str:
        """Pick the best persona name from instance based on expertise overlap."""
        if not expertise_tags or not instance.personas:
            return instance.personas[0].name if instance.personas else ""

        query_tags = {t.lower() for t in expertise_tags}
        best_name = ""
        best_score = 0

        for persona in instance.personas:
            persona_tags = {t.lower() for t in persona.expertise_tags}
            overlap = len(query_tags & persona_tags)
            if overlap > best_score:
                best_score = overlap
                best_name = persona.name

        return best_name or (instance.personas[0].name if instance.personas else "")

    async def shutdown(self) -> None:
        """Graceful shutdown."""
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

    # Configure logging.
    log_level = getattr(logging, config.logging.level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    try:
        asyncio.run(_run_server(config))
    except KeyboardInterrupt:
        print("\nBotPort stopped.")
        sys.exit(0)


async def _run_server(config: BotPortConfig) -> None:
    """Async server lifecycle."""
    server = BotPortServer(config)
    app = server.create_app()
    runner = web.AppRunner(app)
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

    print(f"BotPort v{__version__} running on http://{host}:{port}")
    if config.server.dashboard_enabled:
        print(f"  Dashboard: http://{host}:{port}/")
    print(f"  WebSocket: ws://{host}:{port}/ws")
    print(f"  Routing strategy: {config.routing.strategy}")
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
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass

    await server.shutdown()
    try:
        await asyncio.wait_for(runner.cleanup(), timeout=3.0)
    except asyncio.TimeoutError:
        pass


if __name__ == "__main__":
    main()
