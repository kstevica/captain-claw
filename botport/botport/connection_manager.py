"""WebSocket connection pool for connected CC instances."""

from __future__ import annotations

import json
import logging
from typing import Any

from aiohttp import web

from botport.models import InstanceInfo, PersonaInfo, _utcnow_iso
from botport.protocol import BaseMessage, RegisterMessage, serialize_message

log = logging.getLogger(__name__)


class ConnectionManager:
    """Manages persistent WebSocket connections from CC instances."""

    def __init__(self) -> None:
        self._connections: dict[str, web.WebSocketResponse] = {}  # instance_id -> ws
        self._instances: dict[str, InstanceInfo] = {}  # instance_id -> info
        self._ws_to_instance: dict[int, str] = {}  # id(ws) -> instance_id

    async def register(
        self,
        ws: web.WebSocketResponse,
        msg: RegisterMessage,
        instance_id: str,
    ) -> InstanceInfo:
        """Register a new CC instance connection.

        Returns the created InstanceInfo.
        """
        capabilities = msg.capabilities or {}

        personas: list[PersonaInfo] = []
        for p in capabilities.get("personas", []):
            personas.append(PersonaInfo.from_dict(p) if isinstance(p, dict) else PersonaInfo(name=str(p)))

        info = InstanceInfo(
            id=instance_id,
            name=msg.instance_name or instance_id,
            personas=personas,
            tools=list(capabilities.get("tools", [])),
            models=list(capabilities.get("models", [])),
            max_concurrent=int(capabilities.get("max_concurrent", 5)),
            active_concerns=0,
            status="connected",
            connected_at=_utcnow_iso(),
            last_heartbeat=_utcnow_iso(),
        )

        self._connections[instance_id] = ws
        self._instances[instance_id] = info
        self._ws_to_instance[id(ws)] = instance_id

        log.info("Instance registered: %s (%s)", info.name, instance_id)
        return info

    async def unregister(self, instance_id: str) -> InstanceInfo | None:
        """Unregister an instance (disconnect)."""
        info = self._instances.get(instance_id)
        if info:
            info.status = "disconnected"

        ws = self._connections.pop(instance_id, None)
        if ws is not None:
            self._ws_to_instance.pop(id(ws), None)

        if info:
            log.info("Instance unregistered: %s (%s)", info.name, instance_id)
        return info

    def get_instance_id_for_ws(self, ws: web.WebSocketResponse) -> str | None:
        """Look up instance_id by WebSocket object."""
        return self._ws_to_instance.get(id(ws))

    async def send_to(self, instance_id: str, message: BaseMessage) -> bool:
        """Send a message to a specific instance. Returns False if not connected."""
        ws = self._connections.get(instance_id)
        if ws is None or ws.closed:
            return False
        try:
            await ws.send_str(serialize_message(message))
            return True
        except Exception as exc:
            log.warning("Failed to send to %s: %s", instance_id, exc)
            return False

    async def send_json_to(self, instance_id: str, data: dict[str, Any]) -> bool:
        """Send raw JSON dict to a specific instance."""
        ws = self._connections.get(instance_id)
        if ws is None or ws.closed:
            return False
        try:
            await ws.send_str(json.dumps(data, default=str))
            return True
        except Exception as exc:
            log.warning("Failed to send to %s: %s", instance_id, exc)
            return False

    def get_instance(self, instance_id: str) -> InstanceInfo | None:
        return self._instances.get(instance_id)

    def list_instances(self) -> list[InstanceInfo]:
        return list(self._instances.values())

    def list_connected(self) -> list[InstanceInfo]:
        return [i for i in self._instances.values() if i.status == "connected"]

    def list_available(self, exclude: str | None = None) -> list[InstanceInfo]:
        """List instances that are connected and have capacity."""
        return [
            i for i in self._instances.values()
            if i.has_capacity and (exclude is None or i.id != exclude)
        ]

    def update_heartbeat(self, instance_id: str, active_concerns: int = 0) -> None:
        """Update heartbeat timestamp and load info."""
        info = self._instances.get(instance_id)
        if info:
            info.last_heartbeat = _utcnow_iso()
            info.active_concerns = active_concerns

    @property
    def connected_count(self) -> int:
        return sum(1 for i in self._instances.values() if i.status == "connected")

    def increment_active(self, instance_id: str) -> None:
        info = self._instances.get(instance_id)
        if info:
            info.active_concerns += 1

    def decrement_active(self, instance_id: str) -> None:
        info = self._instances.get(instance_id)
        if info and info.active_concerns > 0:
            info.active_concerns -= 1
