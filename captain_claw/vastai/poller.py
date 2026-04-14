"""Background status poller for vast.ai instances.

Since vast.ai has no webhooks, we poll the API periodically to detect
state changes and run Ollama health checks on running instances.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

import httpx

from captain_claw.vastai.models import VastInstanceState, parse_vast_status

if TYPE_CHECKING:
    from captain_claw.vastai.manager import VastAIManager

log = logging.getLogger(__name__)

# Polling intervals (seconds).
POLL_FAST = 10  # When instances are in transitional states.
POLL_SLOW = 60  # When all instances are stable.
HEALTH_TIMEOUT = 8  # Timeout for Ollama health checks.


class VastAIPoller:
    """Periodically polls vast.ai instance status and Ollama health.

    Uses adaptive intervals: fast when instances are transitioning
    (creating/loading/stopping), slow when everything is stable.
    """

    def __init__(self, manager: VastAIManager):
        self._manager = manager
        self._task: asyncio.Task | None = None
        self._health_client = httpx.AsyncClient(
            timeout=HEALTH_TIMEOUT,
            verify=False,  # vast.ai instances use self-signed certs.
        )

    async def start(self) -> None:
        """Launch the polling loop as a background task."""
        if self._task and not self._task.done():
            log.debug("Poller already running.")
            return
        self._task = asyncio.create_task(self._poll_loop(), name="vastai-poller")
        log.info("vast.ai poller started.")

    async def stop(self) -> None:
        """Cancel the polling task and clean up."""
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None
        await self._health_client.aclose()
        log.info("vast.ai poller stopped.")

    @property
    def is_running(self) -> bool:
        return self._task is not None and not self._task.done()

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def _poll_loop(self) -> None:
        """Periodically sync instance state with vast.ai API."""
        while True:
            try:
                has_transitioning = await self._poll_tick()
                interval = POLL_FAST if has_transitioning else POLL_SLOW
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("vast.ai poller tick failed")
                interval = POLL_FAST  # Retry quickly after errors.
            await asyncio.sleep(interval)

    async def _poll_tick(self) -> bool:
        """Run one polling cycle.

        Returns True if any instances are in a transitional state.
        """
        client = self._manager._client
        if client is None:
            return False

        instances = self._manager._instances
        if not instances:
            return False

        # Fetch all instances from vast.ai in one call.
        try:
            remote_list = await client.list_instances()
        except Exception:
            log.warning("Failed to fetch instance list from vast.ai", exc_info=True)
            return any(inst.is_transitioning for inst in instances.values())

        # Build a lookup by ID.
        remote_by_id: dict[int, dict] = {}
        for raw in remote_list:
            rid = raw.get("id")
            if rid is not None:
                remote_by_id[int(rid)] = raw

        has_transitioning = False

        for inst_id, inst in list(instances.items()):
            if inst.state == VastInstanceState.DESTROYED:
                continue

            remote = remote_by_id.get(inst_id)
            if remote is None:
                # Instance disappeared from vast.ai — likely destroyed externally.
                if inst.state != VastInstanceState.DESTROYED:
                    log.warning("Instance %d disappeared from vast.ai, marking destroyed", inst_id)
                    inst.state = VastInstanceState.DESTROYED
                    inst.ollama_ready = False
                    await self._manager._persist_instance(inst)
                continue

            # Update state from remote.
            new_state = parse_vast_status(str(remote.get("actual_status", remote.get("status_msg", ""))))
            old_state = inst.state

            if new_state != old_state:
                log.info("Instance %d state: %s -> %s", inst_id, old_state, new_state)
                inst.state = new_state

                # Reset ollama_ready when instance stops/errors.
                if new_state in (
                    VastInstanceState.STOPPED,
                    VastInstanceState.EXITED,
                    VastInstanceState.ERROR,
                ):
                    inst.ollama_ready = False

            # Update network info from remote.
            self._update_network_info(inst, remote)

            # Update cost info.
            inst.dph_total = float(remote.get("dph_total", inst.dph_total) or inst.dph_total)

            # Ollama health check for running instances.
            if inst.state == VastInstanceState.RUNNING and not inst.ollama_ready:
                if inst.ollama_base_url:
                    is_healthy = await self._check_ollama_health(inst.ollama_base_url, inst.auth_token)
                    if is_healthy:
                        log.info("Instance %d: Ollama is healthy at %s", inst_id, inst.ollama_base_url)
                        inst.ollama_ready = True
                        # Becoming healthy counts as activity (starts the idle timer).
                        if not inst.last_activity_at:
                            from datetime import datetime, timezone
                            inst.last_activity_at = datetime.now(timezone.utc).isoformat()

            # Auto-stop check: if instance is running, has auto_stop set,
            # and has been idle too long, stop it.
            if (
                inst.state == VastInstanceState.RUNNING
                and inst.auto_stop_minutes > 0
            ):
                from datetime import datetime, timezone
                now = datetime.now(timezone.utc)
                # If no activity recorded yet, set it to now (start the timer).
                if not inst.last_activity_at:
                    inst.last_activity_at = now.isoformat()
                else:
                    try:
                        last = datetime.fromisoformat(inst.last_activity_at)
                        idle_secs = (now - last).total_seconds()
                        if idle_secs > inst.auto_stop_minutes * 60:
                            log.info(
                                "Instance %d idle for %.0fs (limit %dm), auto-stopping.",
                                inst_id, idle_secs, inst.auto_stop_minutes,
                            )
                            try:
                                await self._manager.stop_instance(inst_id)
                            except Exception:
                                log.warning("Auto-stop failed for instance %d", inst_id, exc_info=True)
                    except (ValueError, TypeError):
                        inst.last_activity_at = now.isoformat()

            if inst.is_transitioning:
                has_transitioning = True

            # Persist any changes.
            await self._manager._persist_instance(inst)

        return has_transitioning

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _update_network_info(self, inst: Any, remote: dict) -> None:
        """Extract public IP and port mappings from vast.ai instance data."""
        from captain_claw.vastai.client import extract_port_mapping

        # Public IP.
        public_ip = str(remote.get("public_ipaddr", "") or "")
        if public_ip:
            inst.public_ip = public_ip

        # Port mappings.
        ports = remote.get("ports", {})
        if ports:
            ip, port = extract_port_mapping(ports, 11434)
            if port:
                inst.ollama_port = port
                if ip and ip != "0.0.0.0":
                    inst.public_ip = ip

        # SSH port.
        ssh_port = remote.get("ssh_port", 0)
        if ssh_port:
            inst.ssh_port = int(ssh_port)

    async def _check_ollama_health(self, base_url: str, token: str) -> bool:
        """Probe the Ollama ``/api/version`` endpoint.

        Returns True if the server responds with 200.
        """
        try:
            headers = {}
            if token:
                headers["Authorization"] = f"Bearer {token}"
            resp = await self._health_client.get(
                f"{base_url}/api/version",
                headers=headers,
            )
            return resp.is_success
        except Exception:
            return False
