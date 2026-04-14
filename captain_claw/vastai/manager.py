"""High-level orchestration for vast.ai GPU instances with Ollama.

The manager coordinates the REST client, background poller, DB persistence,
and Ollama model management.  It is the single entry point for all vast.ai
operations from Flight Deck routes.
"""

from __future__ import annotations

import json
import logging
import secrets
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import httpx

from captain_claw.vastai.client import VastAIClient, VastAPIError
from captain_claw.vastai.models import (
    CreateInstanceRequest,
    InstanceConnectionInfo,
    VastAccountInfo,
    VastInstance,
    VastInstanceState,
    VastOffer,
    VastOfferFilter,
)
from captain_claw.vastai.poller import VastAIPoller
from captain_claw.vastai.setup_scripts import env_vars_for_instance, ollama_setup_script

if TYPE_CHECKING:
    from captain_claw.flight_deck.db import FlightDeckDB

log = logging.getLogger(__name__)

# System settings keys.
_KEY_API_KEY = "vastai:api_key"
_KEY_INSTANCE_PREFIX = "vastai:instance:"

# Timeout for Ollama model operations (pull can take a long time).
_MODEL_OP_TIMEOUT = httpx.Timeout(30.0, read=600.0)


class VastAIManager:
    """Manages vast.ai GPU instances with Ollama.

    Lifecycle::

        mgr = VastAIManager(db)
        await mgr.initialize()      # Load persisted state, start poller if configured.
        ...                          # Use via Flight Deck routes.
        await mgr.shutdown()         # Clean up on app shutdown.
    """

    def __init__(self, db: FlightDeckDB):
        self._db = db
        self._client: VastAIClient | None = None
        self._poller: VastAIPoller | None = None
        self._instances: dict[int, VastInstance] = {}
        self._ollama_http = httpx.AsyncClient(
            timeout=_MODEL_OP_TIMEOUT,
            verify=False,  # Self-signed certs on vast.ai instances.
        )

    # ------------------------------------------------------------------
    # Initialization / Shutdown
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Load persisted state and start the poller if an API key exists."""
        api_key = await self._db.get_system_setting(_KEY_API_KEY)
        if api_key:
            await self._init_client(api_key)
        await self._load_instances()

    async def shutdown(self) -> None:
        """Stop the poller and close HTTP clients."""
        if self._poller:
            await self._poller.stop()
        if self._client:
            await self._client.close()
        await self._ollama_http.aclose()

    @property
    def is_configured(self) -> bool:
        """True if a vast.ai API key is set and the client is ready."""
        return self._client is not None

    async def set_api_key(self, api_key: str) -> None:
        """Set or update the vast.ai API key."""
        # Validate the key by hitting the account endpoint.
        test_client = VastAIClient(api_key)
        try:
            await test_client.get_account()
        finally:
            await test_client.close()

        await self._db.set_system_setting(_KEY_API_KEY, api_key)
        await self._init_client(api_key)

    async def remove_api_key(self) -> None:
        """Remove the vast.ai API key and stop the poller."""
        if self._poller:
            await self._poller.stop()
            self._poller = None
        if self._client:
            await self._client.close()
            self._client = None
        await self._db.set_system_setting(_KEY_API_KEY, "")

    async def _init_client(self, api_key: str) -> None:
        """Create the API client and start the poller."""
        if self._client:
            await self._client.close()
        if self._poller:
            await self._poller.stop()

        self._client = VastAIClient(api_key)
        self._poller = VastAIPoller(self)
        await self._poller.start()

    # ------------------------------------------------------------------
    # Offer browsing
    # ------------------------------------------------------------------

    async def search_offers(self, filters: VastOfferFilter | None = None) -> list[VastOffer]:
        """Search available GPU offers on the vast.ai marketplace."""
        self._require_client()
        assert self._client is not None
        return await self._client.search_offers(filters or VastOfferFilter())

    # ------------------------------------------------------------------
    # Instance lifecycle
    # ------------------------------------------------------------------

    async def create_instance(self, req: CreateInstanceRequest) -> VastInstance:
        """Create a new vast.ai instance with Ollama.

        Generates a bearer token, composes the setup script, calls the
        vast.ai API, and starts tracking the instance.
        """
        self._require_client()
        assert self._client is not None

        # Generate a unique bearer token for this instance.
        bearer_token = secrets.token_urlsafe(32)

        # Build env vars and onstart script.
        env = env_vars_for_instance(bearer_token, secure=req.secure_ollama)
        onstart = ollama_setup_script(
            pre_pull_model=req.pre_pull_model,
            secure=req.secure_ollama,
            bearer_token=bearer_token,
        )

        # Create the instance via vast.ai API.
        instance_id, instance_api_key = await self._client.create_instance(
            offer_id=req.offer_id,
            image="ollama/ollama",
            disk_gb=req.disk_gb,
            env=env,
            onstart_cmd=onstart,
            direct=True,
            ssh=True,
            label=req.label or f"captain-claw-{req.offer_id}",
        )

        # Fetch initial instance details to populate GPU info.
        gpu_name = ""
        gpu_ram_gb = 0.0
        num_gpus = 1
        dph_total = 0.0
        try:
            details = await self._client.get_instance(instance_id)
            gpu_name = str(details.get("gpu_name", "") or "")
            gpu_ram_gb = round(float(details.get("gpu_ram", 0) or 0) / 1024, 1)
            num_gpus = int(details.get("num_gpus", 1) or 1)
            dph_total = float(details.get("dph_total", 0) or 0)
        except Exception:
            log.debug("Could not fetch initial details for instance %d", instance_id)

        inst = VastInstance(
            id=instance_id,
            offer_id=req.offer_id,
            gpu_name=gpu_name,
            num_gpus=num_gpus,
            gpu_ram_gb=gpu_ram_gb,
            state=VastInstanceState.CREATING,
            auth_token=bearer_token,
            dph_total=dph_total,
            disk_gb=req.disk_gb,
            created_at=datetime.now(timezone.utc).isoformat(),
            label=req.label,
            auto_stop_minutes=5,
            secure_ollama=req.secure_ollama,
        )

        self._instances[instance_id] = inst
        await self._persist_instance(inst)
        log.info("Created vast.ai instance %d (offer=%d, gpu=%s)", instance_id, req.offer_id, gpu_name)
        return inst

    async def stop_instance(self, instance_id: int) -> VastInstance:
        """Stop a running instance (GPU billing stops, storage continues)."""
        self._require_client()
        assert self._client is not None
        inst = self._get_instance(instance_id)

        await self._client.stop_instance(instance_id)
        inst.state = VastInstanceState.STOPPING
        inst.ollama_ready = False
        await self._persist_instance(inst)
        return inst

    async def start_instance(self, instance_id: int) -> VastInstance:
        """Start a stopped instance."""
        self._require_client()
        assert self._client is not None
        inst = self._get_instance(instance_id)

        await self._client.start_instance(instance_id)
        inst.state = VastInstanceState.LOADING
        inst.ollama_ready = False
        await self._persist_instance(inst)
        return inst

    async def destroy_instance(self, instance_id: int) -> VastInstance:
        """Permanently destroy an instance and all its data."""
        self._require_client()
        assert self._client is not None
        inst = self._get_instance(instance_id)

        await self._client.destroy_instance(instance_id)
        inst.state = VastInstanceState.DESTROYED
        inst.ollama_ready = False
        await self._persist_instance(inst)
        log.info("Destroyed vast.ai instance %d", instance_id)
        return inst

    async def get_instance(self, instance_id: int) -> VastInstance | None:
        """Get a tracked instance by ID."""
        return self._instances.get(instance_id)

    async def list_instances(self) -> list[VastInstance]:
        """List all tracked instances (including stopped/destroyed)."""
        return list(self._instances.values())

    # ------------------------------------------------------------------
    # Auto-stop configuration
    # ------------------------------------------------------------------

    async def set_auto_stop(self, instance_id: int, minutes: int) -> VastInstance:
        """Set auto-stop timer for an instance. 0 = disabled."""
        inst = self._get_instance(instance_id)
        inst.auto_stop_minutes = minutes
        # Reset the activity timer so countdown starts from now.
        if minutes > 0:
            inst.last_activity_at = datetime.now(timezone.utc).isoformat()
        await self._persist_instance(inst)
        log.info("Instance %d auto-stop set to %d min", instance_id, minutes)
        return inst

    def touch_activity(self, instance_id: int) -> None:
        """Record Ollama API activity on an instance (resets inactivity timer)."""
        inst = self._instances.get(instance_id)
        if inst:
            inst.last_activity_at = datetime.now(timezone.utc).isoformat()

    def find_instance_by_url(self, base_url: str) -> VastInstance | None:
        """Find a tracked instance whose Ollama URL matches *base_url*."""
        if not base_url:
            return None
        base_url = base_url.rstrip("/")
        for inst in self._instances.values():
            if inst.ollama_base_url and inst.ollama_base_url.rstrip("/") == base_url:
                return inst
        return None

    # ------------------------------------------------------------------
    # Auto-wake: ensure instance is running before LLM request
    # ------------------------------------------------------------------

    async def ensure_running(self, instance_id: int, timeout: float = 120) -> VastInstance:
        """Ensure an instance is running and Ollama is ready.

        If the instance is stopped/exited, starts it and waits up to
        *timeout* seconds for Ollama to become healthy.

        Returns the instance once ready, or raises VastAPIError on timeout.
        """
        import asyncio

        inst = self._get_instance(instance_id)

        # Already running and ready — fast path.
        if inst.state == VastInstanceState.RUNNING and inst.ollama_ready:
            self.touch_activity(instance_id)
            return inst

        # Need to start it.
        if inst.state in (VastInstanceState.STOPPED, VastInstanceState.EXITED):
            log.info("Auto-waking vast.ai instance %d ...", instance_id)
            await self.start_instance(instance_id)

        # Wait for running + ollama_ready.
        deadline = asyncio.get_event_loop().time() + timeout
        while asyncio.get_event_loop().time() < deadline:
            # Let the poller do its work, but also do our own quick check.
            if inst.state == VastInstanceState.RUNNING and inst.ollama_base_url:
                # Quick health check.
                try:
                    headers = {}
                    if inst.auth_token:
                        headers["Authorization"] = f"Bearer {inst.auth_token}"
                    resp = await self._ollama_http.get(
                        f"{inst.ollama_base_url}/api/version",
                        headers=headers,
                    )
                    if resp.is_success:
                        inst.ollama_ready = True
                        self.touch_activity(instance_id)
                        await self._persist_instance(inst)
                        log.info("Instance %d is awake and Ollama ready.", instance_id)
                        return inst
                except Exception:
                    pass

            if inst.state in (VastInstanceState.ERROR, VastInstanceState.DESTROYED):
                raise VastAPIError(f"Instance {instance_id} is in state {inst.state}, cannot wake.")

            await asyncio.sleep(3)

            # Re-read state (poller may have updated it).
            inst = self._get_instance(instance_id)

        raise VastAPIError(
            f"Timeout waiting for instance {instance_id} to become ready after {timeout}s.",
            status_code=408,
        )

    # ------------------------------------------------------------------
    # Connection info (for OllamaProvider)
    # ------------------------------------------------------------------

    def get_connection_info(self, instance_id: int) -> InstanceConnectionInfo:
        """Get the connection details for using this instance as an Ollama provider."""
        inst = self._get_instance(instance_id)
        return InstanceConnectionInfo(
            provider="ollama",
            base_url=inst.ollama_base_url,
            api_key=inst.auth_token,
            ollama_ready=inst.ollama_ready,
            models=list(inst.models_loaded),
        )

    # ------------------------------------------------------------------
    # Model management (proxied to Ollama on the instance)
    # ------------------------------------------------------------------

    async def list_models(self, instance_id: int) -> list[dict[str, Any]]:
        """List models available on the instance's Ollama server."""
        inst = self._get_instance(instance_id)
        self._require_ollama_ready(inst)

        data = await self._ollama_request("GET", inst, "/api/tags")
        models = data.get("models", [])

        # Update cached model list.
        inst.models_loaded = [
            str(m.get("name", "")) for m in models if m.get("name")
        ]
        await self._persist_instance(inst)
        return models

    async def pull_model(self, instance_id: int, model_tag: str) -> dict[str, Any]:
        """Pull a model on the instance's Ollama server.

        This can take a long time for large models.  The request timeout
        is set generously (10 minutes).
        """
        inst = self._get_instance(instance_id)
        self._require_ollama_ready(inst)

        log.info("Pulling model %r on instance %d", model_tag, instance_id)
        data = await self._ollama_request(
            "POST", inst, "/api/pull",
            json={"name": model_tag, "stream": False},
        )

        # Refresh model list after pull.
        try:
            await self.list_models(instance_id)
        except Exception:
            pass  # Non-critical.

        return data

    async def delete_model(self, instance_id: int, model_tag: str) -> None:
        """Delete a model from the instance's Ollama server."""
        inst = self._get_instance(instance_id)
        self._require_ollama_ready(inst)

        log.info("Deleting model %r on instance %d", model_tag, instance_id)
        await self._ollama_request(
            "DELETE", inst, "/api/delete",
            json={"name": model_tag},
        )

        # Refresh model list.
        try:
            await self.list_models(instance_id)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Account
    # ------------------------------------------------------------------

    async def get_account(self) -> VastAccountInfo:
        """Get vast.ai account info (balance, etc.)."""
        self._require_client()
        assert self._client is not None
        return await self._client.get_account()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    async def _persist_instance(self, inst: VastInstance) -> None:
        """Save instance state to system_settings."""
        key = f"{_KEY_INSTANCE_PREFIX}{inst.id}"
        await self._db.set_system_setting(key, inst.model_dump_json())

    async def _load_instances(self) -> None:
        """Load all persisted instances from system_settings."""
        # We need to query all keys with the prefix.  The DB doesn't have
        # a prefix-search method, so we'll use a raw query.
        try:
            async with self._db._db.execute(
                "SELECT key, value FROM system_settings WHERE key LIKE ?",
                (f"{_KEY_INSTANCE_PREFIX}%",),
            ) as cur:
                rows = await cur.fetchall()
        except Exception:
            log.debug("No persisted vast.ai instances found.")
            return

        for row in rows:
            try:
                inst = VastInstance.model_validate_json(row["value"])
                self._instances[inst.id] = inst
            except Exception:
                log.warning("Failed to load persisted instance: key=%s", row["key"])

        if self._instances:
            log.info("Loaded %d persisted vast.ai instances.", len(self._instances))

    async def _remove_instance(self, instance_id: int) -> None:
        """Remove a persisted instance from system_settings."""
        key = f"{_KEY_INSTANCE_PREFIX}{instance_id}"
        await self._db.set_system_setting(key, "")
        self._instances.pop(instance_id, None)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _require_client(self) -> None:
        if self._client is None:
            raise VastAPIError("vast.ai is not configured. Set an API key first.")

    def _get_instance(self, instance_id: int) -> VastInstance:
        inst = self._instances.get(instance_id)
        if inst is None:
            raise VastAPIError(f"Instance {instance_id} not found.", status_code=404)
        return inst

    def _require_ollama_ready(self, inst: VastInstance) -> None:
        if not inst.ollama_ready:
            raise VastAPIError(
                f"Ollama is not ready on instance {inst.id} "
                f"(state={inst.state}, ollama_ready={inst.ollama_ready}).",
                status_code=409,
            )
        if not inst.ollama_base_url:
            raise VastAPIError(
                f"Instance {inst.id} has no Ollama endpoint yet.",
                status_code=409,
            )

    async def _ollama_request(
        self,
        method: str,
        inst: VastInstance,
        path: str,
        *,
        json: dict | None = None,
    ) -> dict[str, Any]:
        """Make an authenticated request to the Ollama API on an instance."""
        url = f"{inst.ollama_base_url}{path}"
        headers: dict[str, str] = {}
        if inst.auth_token:
            headers["Authorization"] = f"Bearer {inst.auth_token}"

        try:
            resp = await self._ollama_http.request(method, url, json=json, headers=headers)
        except httpx.HTTPError as exc:
            raise VastAPIError(f"Failed to reach Ollama on instance {inst.id}: {exc}") from exc

        if not resp.is_success:
            raise VastAPIError(
                f"Ollama API error on instance {inst.id}: {resp.status_code} {resp.text[:500]}",
                status_code=resp.status_code,
            )
        if not resp.text.strip():
            return {}
        return resp.json()
