"""Flight Deck REST routes for vast.ai GPU cloud management."""

from __future__ import annotations

import os
import secrets as _secrets

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from captain_claw.flight_deck.auth import get_current_user, get_db
from captain_claw.vastai import (
    CreateInstanceRequest,
    PullModelRequest,
    SetAutoStopRequest,
    VastAIManager,
    VastAPIError,
    VastOfferFilter,
)

router = APIRouter(prefix="/fd/vastai", tags=["vastai"])

# ---------------------------------------------------------------------------
# Manager singleton (initialized in server.py lifespan)
# ---------------------------------------------------------------------------

_manager: VastAIManager | None = None


def set_vastai_manager(mgr: VastAIManager) -> None:
    global _manager
    _manager = mgr


def get_vastai_manager() -> VastAIManager:
    if _manager is None:
        raise HTTPException(503, "vast.ai integration not initialized")
    return _manager


def _handle_vast_error(exc: VastAPIError) -> HTTPException:
    """Convert a VastAPIError to an appropriate HTTPException."""
    status = exc.status_code or 502
    if status == 401:
        return HTTPException(401, "Invalid vast.ai API key")
    if status == 404:
        return HTTPException(404, str(exc))
    if status == 409:
        return HTTPException(409, str(exc))
    return HTTPException(status, str(exc))


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class SetAPIKeyRequest(BaseModel):
    api_key: str


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@router.get("/status")
async def vastai_status(user: dict = Depends(get_current_user)):
    """Check if vast.ai is configured and return account balance."""
    mgr = get_vastai_manager()
    result: dict = {"configured": mgr.is_configured}
    if mgr.is_configured:
        try:
            account = await mgr.get_account()
            result["balance"] = account.balance
            result["email"] = account.email
        except VastAPIError:
            result["balance"] = None
            result["error"] = "Failed to fetch account info"
    return result


@router.put("/api-key")
async def set_api_key(body: SetAPIKeyRequest, user: dict = Depends(get_current_user)):
    """Set or update the vast.ai API key. Validates by hitting the API."""
    mgr = get_vastai_manager()
    try:
        await mgr.set_api_key(body.api_key)
    except VastAPIError as exc:
        raise _handle_vast_error(exc)
    return {"ok": True}


@router.delete("/api-key")
async def remove_api_key(user: dict = Depends(get_current_user)):
    """Remove the vast.ai API key."""
    mgr = get_vastai_manager()
    await mgr.remove_api_key()
    return {"ok": True}


# ---------------------------------------------------------------------------
# GPU offer search
# ---------------------------------------------------------------------------


@router.post("/offers/search")
async def search_offers(
    filters: VastOfferFilter | None = None,
    user: dict = Depends(get_current_user),
):
    """Search the vast.ai marketplace for available GPU offers."""
    mgr = get_vastai_manager()
    try:
        offers = await mgr.search_offers(filters)
    except VastAPIError as exc:
        raise _handle_vast_error(exc)
    return {"offers": [o.model_dump() for o in offers]}


# ---------------------------------------------------------------------------
# Instance lifecycle
# ---------------------------------------------------------------------------


@router.get("/instances")
async def list_instances(user: dict = Depends(get_current_user)):
    """List all tracked vast.ai instances."""
    mgr = get_vastai_manager()
    instances = await mgr.list_instances()
    return {"instances": [i.model_dump() for i in instances]}


@router.post("/instances")
async def create_instance(
    body: CreateInstanceRequest,
    user: dict = Depends(get_current_user),
):
    """Create a new vast.ai GPU instance with Ollama."""
    mgr = get_vastai_manager()
    try:
        inst = await mgr.create_instance(body)
    except VastAPIError as exc:
        raise _handle_vast_error(exc)
    return inst.model_dump()


@router.get("/instances/{instance_id}")
async def get_instance(instance_id: int, user: dict = Depends(get_current_user)):
    """Get details for a specific instance."""
    mgr = get_vastai_manager()
    inst = await mgr.get_instance(instance_id)
    if inst is None:
        raise HTTPException(404, f"Instance {instance_id} not found")
    return inst.model_dump()


@router.post("/instances/{instance_id}/stop")
async def stop_instance(instance_id: int, user: dict = Depends(get_current_user)):
    """Stop a running instance (GPU billing stops, storage continues)."""
    mgr = get_vastai_manager()
    try:
        inst = await mgr.stop_instance(instance_id)
    except VastAPIError as exc:
        raise _handle_vast_error(exc)
    return inst.model_dump()


@router.post("/instances/{instance_id}/start")
async def start_instance(instance_id: int, user: dict = Depends(get_current_user)):
    """Start a stopped instance."""
    mgr = get_vastai_manager()
    try:
        inst = await mgr.start_instance(instance_id)
    except VastAPIError as exc:
        raise _handle_vast_error(exc)
    return inst.model_dump()


@router.delete("/instances/{instance_id}")
async def destroy_instance(instance_id: int, user: dict = Depends(get_current_user)):
    """Permanently destroy an instance and all its data."""
    mgr = get_vastai_manager()
    try:
        inst = await mgr.destroy_instance(instance_id)
    except VastAPIError as exc:
        raise _handle_vast_error(exc)
    return inst.model_dump()


# ---------------------------------------------------------------------------
# Auto-stop configuration
# ---------------------------------------------------------------------------


@router.put("/instances/{instance_id}/auto-stop")
async def set_auto_stop(
    instance_id: int,
    body: SetAutoStopRequest,
    user: dict = Depends(get_current_user),
):
    """Set auto-stop timer for an instance. 0 = disabled."""
    mgr = get_vastai_manager()
    try:
        inst = await mgr.set_auto_stop(instance_id, body.auto_stop_minutes)
    except VastAPIError as exc:
        raise _handle_vast_error(exc)
    return inst.model_dump()


# ---------------------------------------------------------------------------
# Instance connection info (for OllamaProvider)
# ---------------------------------------------------------------------------


@router.get("/instances/{instance_id}/connection")
async def get_connection_info(instance_id: int, user: dict = Depends(get_current_user)):
    """Get Ollama provider connection details for this instance.

    Returns the base URL and auth token needed to configure an OllamaProvider.
    """
    mgr = get_vastai_manager()
    try:
        info = mgr.get_connection_info(instance_id)
    except VastAPIError as exc:
        raise _handle_vast_error(exc)
    return info.model_dump()


# ---------------------------------------------------------------------------
# Model management (proxied to Ollama on instance)
# ---------------------------------------------------------------------------


@router.get("/instances/{instance_id}/models")
async def list_models(instance_id: int, user: dict = Depends(get_current_user)):
    """List Ollama models available on the instance."""
    mgr = get_vastai_manager()
    try:
        models = await mgr.list_models(instance_id)
    except VastAPIError as exc:
        raise _handle_vast_error(exc)
    return {"models": models}


@router.post("/instances/{instance_id}/models/pull")
async def pull_model(
    instance_id: int,
    body: PullModelRequest,
    user: dict = Depends(get_current_user),
):
    """Pull a model on the instance's Ollama server.

    This can take several minutes for large models.
    """
    mgr = get_vastai_manager()
    try:
        result = await mgr.pull_model(instance_id, body.model)
    except VastAPIError as exc:
        raise _handle_vast_error(exc)
    return result


@router.delete("/instances/{instance_id}/models/{model_tag:path}")
async def delete_model(
    instance_id: int,
    model_tag: str,
    user: dict = Depends(get_current_user),
):
    """Delete a model from the instance's Ollama server."""
    mgr = get_vastai_manager()
    try:
        await mgr.delete_model(instance_id, model_tag)
    except VastAPIError as exc:
        raise _handle_vast_error(exc)
    return {"ok": True}


# ---------------------------------------------------------------------------
# Account
# ---------------------------------------------------------------------------


@router.get("/account")
async def get_account(user: dict = Depends(get_current_user)):
    """Get vast.ai account info (balance, email)."""
    mgr = get_vastai_manager()
    try:
        account = await mgr.get_account()
    except VastAPIError as exc:
        raise _handle_vast_error(exc)
    return account.model_dump()


# ---------------------------------------------------------------------------
# Internal: auto-wake (called by agent processes, not the UI)
# ---------------------------------------------------------------------------


def _authorize_agent_call(request: Request) -> None:
    """Verify the caller is either on loopback or has the shared secret."""
    secret = os.environ.get("FD_AGENT_SHARED_SECRET", "").strip()
    if secret:
        provided = request.headers.get("X-Agent-Secret", "")
        if provided and _secrets.compare_digest(provided, secret):
            return
    # Fall through to loopback check.
    client_host = request.client.host if request.client else ""
    if client_host in ("127.0.0.1", "::1", "localhost"):
        return
    raise HTTPException(403, "Forbidden")


class WakeRequest(BaseModel):
    base_url: str


@router.post("/wake")
async def wake_instance(body: WakeRequest, request: Request):
    """Start a stopped vast.ai instance and wait for Ollama to be ready.

    This is an internal endpoint called by agent processes (via
    ``captain_claw.vastai.wake``) when they need to talk to a stopped
    instance.  Auth is loopback or ``X-Agent-Secret``.

    Returns 404 if the base_url doesn't match any tracked instance.
    Returns 200 with instance data once Ollama is ready.
    """
    _authorize_agent_call(request)
    mgr = get_vastai_manager()

    inst = mgr.find_instance_by_url(body.base_url)
    if inst is None:
        raise HTTPException(404, "No tracked instance matches this URL")

    try:
        inst = await mgr.ensure_running(inst.id, timeout=120)
    except VastAPIError as exc:
        raise _handle_vast_error(exc)
    return inst.model_dump()
