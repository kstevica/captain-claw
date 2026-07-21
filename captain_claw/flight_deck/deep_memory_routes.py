"""Flight Deck REST surface for deep memory.

Two families of caller, one tenancy chokepoint:

* **Dashboard** — ``Depends(get_current_user)`` + ``_eff_owner()``, mirroring
  ``vfs_routes``, so a share row is what grants cross-user access.
* **Agents** — the ``/agent/*`` routes. An agent never holds a Typesense key;
  it proves it is an FD-spawned process with ``X-Agent-Secret`` and identifies
  *which* agent with its unique ``web_auth`` token, from which FD looks the
  owner up in the process registry. The owner is therefore never something the
  agent asserts.
"""

from __future__ import annotations

import hmac
import json
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from captain_claw.flight_deck import deep_memory_service as svc
from captain_claw.flight_deck.auth import get_current_user, get_db
from captain_claw.logging import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/fd/deep-memory", tags=["deep-memory"])


# ---------------------------------------------------------------------------
# Tenancy
# ---------------------------------------------------------------------------


async def _eff_owner(caller_id: str, project: str, owner: str, *, write: bool) -> str:
    """Resolve whose archive this request acts on. The only place an owner id
    is chosen — mirrors ``vfs_routes._eff_owner``."""
    owner = (owner or "").strip()
    if not owner or owner == caller_id:
        return caller_id
    share = await get_db().get_share_for_grantee("vfs", project, caller_id, owner)
    if not share:
        raise HTTPException(404, "not found")
    if write and str(share.get("permission")) != "edit":
        raise HTTPException(403, "read-only share")
    return owner


def _agent_owner(request: Request) -> str:
    """Resolve the owner behind an agent-tool request, or 403.

    Identity comes from FD's own records, never from the agent's claim: the
    per-agent ``web_auth`` token (unique, stored in the registry and Docker
    label at spawn) first, then the source port. The shared ``X-Agent-Secret``
    only proves the caller is *an* FD-spawned agent — it cannot distinguish
    one agent from another, so it is a gate, not an identity.
    """
    from captain_claw.flight_deck.agent_secret import get_or_create_agent_secret
    from captain_claw.flight_deck.server import (
        _resolve_agent_owner,
        _resolve_agent_owner_by_auth,
    )

    expected = get_or_create_agent_secret()
    provided = request.headers.get("X-Agent-Secret", "")
    client_host = getattr(getattr(request, "client", None), "host", "") or ""
    loopback = client_host in ("127.0.0.1", "::1", "localhost")
    if expected and provided:
        if not hmac.compare_digest(expected, provided):
            raise HTTPException(403, "bad agent secret")
    elif not loopback:
        raise HTTPException(403, "agent secret required")

    owner = ""
    token = request.headers.get("X-Agent-Auth", "")
    if token:
        owner = _resolve_agent_owner_by_auth(token) or ""
    if not owner:
        port = getattr(getattr(request, "client", None), "port", 0) or 0
        try:
            owner = _resolve_agent_owner(int(port)) or ""
        except Exception:
            owner = ""
    if not owner:
        raise HTTPException(403, "could not resolve calling agent's owner")
    return owner


# ---------------------------------------------------------------------------
# Bodies
# ---------------------------------------------------------------------------


class IndexFileBody(BaseModel):
    project: str
    path: str
    owner: str = ""
    summarize: bool = False
    force: bool = False


class IndexProjectBody(BaseModel):
    project: str
    owner: str = ""
    summarize: bool = False
    force: bool = False


class IndexingToggleBody(BaseModel):
    project: str
    enabled: bool
    owner: str = ""


class DropBody(BaseModel):
    project: str
    path: str = ""
    owner: str = ""
    recursive: bool = False


class ConnectionBody(BaseModel):
    enabled: bool = True
    host: str = "localhost"
    port: int = 8108
    protocol: str = "http"
    api_key: str = ""
    collection_name: str = "captain_claw_deep_memory"


class AgentSearchBody(BaseModel):
    query: str
    max_results: int = 10
    filter_by: str = ""


class AgentIndexBody(BaseModel):
    text: str = ""
    reference: str = ""
    source: str = "agent"
    summarize: bool = False


class AgentDeleteBody(BaseModel):
    reference: str = ""
    filter_by: str = ""


# ---------------------------------------------------------------------------
# Connection (Connections → Typesense)
# ---------------------------------------------------------------------------

_MASK = "••••••••"


async def load_connection() -> None:
    """Prime the service's settings cache from the FD database.

    Called at server startup so the first request — which may well be an agent's
    — sees the configured connection rather than the config.yaml fallback.
    """
    try:
        raw = await get_db().get_system_setting(svc.SETTINGS_KEY)
    except Exception as exc:
        log.debug("Deep memory connection not loaded", error=str(exc))
        return
    if raw:
        try:
            svc.apply_settings(json.loads(raw))
        except ValueError:
            log.warning("Stored deep memory connection is not valid JSON")


@router.get("/connection")
async def get_connection(user: dict = Depends(get_current_user)) -> dict[str, Any]:
    """Current connection, with the key masked — never echo a secret back."""
    s = svc.current_settings()
    return {
        "enabled": bool(s.get("enabled")),
        "host": s.get("host", ""),
        "port": s.get("port", 8108),
        "protocol": s.get("protocol", "http"),
        "collection_name": s.get("collection_name", ""),
        "api_key": _MASK if str(s.get("api_key") or "") else "",
        "has_api_key": bool(str(s.get("api_key") or "")),
        "configured": svc.configured(),
    }


@router.put("/connection")
async def put_connection(
    body: ConnectionBody, user: dict = Depends(get_current_user)
) -> dict[str, Any]:
    """Save the connection. Flight Deck owns it; agents never see these values."""
    current = svc.current_settings()
    values = body.model_dump()
    # The GET masks the key, so an unchanged form posts the mask back. Treat
    # that as "leave it alone" rather than overwriting the real key with dots.
    if values.get("api_key") in (_MASK, ""):
        values["api_key"] = current.get("api_key", "")
    await get_db().set_system_setting(svc.SETTINGS_KEY, json.dumps(values))
    svc.apply_settings(values)
    return {"ok": True, "configured": svc.configured(), **(svc.probe() if svc.configured() else {})}


@router.post("/connection/test")
async def test_connection(user: dict = Depends(get_current_user)) -> dict[str, Any]:
    """Reach the configured Typesense and report what came back."""
    return svc.probe()


# ---------------------------------------------------------------------------
# Dashboard routes
# ---------------------------------------------------------------------------


@router.get("/status")
async def status(user: dict = Depends(get_current_user)) -> dict[str, Any]:
    """Enabled state plus the vector health that hid the original bug."""
    index = svc.get_index()
    if index is None:
        return {"enabled": False}
    out: dict[str, Any] = {
        "enabled": True,
        "collection": index.collection_name,
        "vectors_disabled": index._vectors_disabled,  # noqa: SLF001
    }
    try:
        index.ensure_collection()
        out["embedding_dims"] = index._embedding_dims  # noqa: SLF001
        out["provider_dims"] = index._probe_chain_dims()  # noqa: SLF001
        # Pre-tenancy documents: present but invisible to owner-scoped search.
        out["unowned"] = index.unowned_count()
    except Exception as exc:
        out["error"] = str(exc)
    return out


@router.post("/claim-unowned")
async def claim_unowned(user: dict = Depends(get_current_user)) -> dict[str, Any]:
    """Take ownership of documents indexed before tenancy existed."""
    index = svc.get_index()
    if index is None:
        raise HTTPException(503, "deep memory is not configured")
    return {"ok": True, "claimed": index.claim_unowned(user["id"])}


@router.get("/projects")
async def list_projects(
    owner: str = "", user: dict = Depends(get_current_user)
) -> dict[str, Any]:
    oid = await _eff_owner(user["id"], "", owner, write=False)
    return {"projects": svc.read_registry(oid)}


@router.post("/indexing")
async def toggle_indexing(
    body: IndexingToggleBody, user: dict = Depends(get_current_user)
) -> dict[str, Any]:
    """Turn automatic indexing on/off for a project.

    Switching it off leaves already-indexed content in place — use the drop
    endpoint to remove it, so a toggle is never silently destructive.
    """
    oid = await _eff_owner(user["id"], body.project, body.owner, write=True)
    entry = svc.set_indexing(oid, body.project, body.enabled)
    return {"ok": True, "project": body.project, **entry}


@router.post("/index-file")
async def index_file(
    body: IndexFileBody, user: dict = Depends(get_current_user)
) -> dict[str, Any]:
    oid = await _eff_owner(user["id"], body.project, body.owner, write=True)
    return svc.index_file(
        oid, body.project, body.path, summarize=body.summarize, force=body.force
    )


@router.post("/index-project")
async def index_project(
    body: IndexProjectBody, user: dict = Depends(get_current_user)
) -> dict[str, Any]:
    oid = await _eff_owner(user["id"], body.project, body.owner, write=True)
    return svc.index_project(
        oid, body.project, summarize=body.summarize, force=body.force
    )


@router.post("/drop")
async def drop(body: DropBody, user: dict = Depends(get_current_user)) -> dict[str, Any]:
    oid = await _eff_owner(user["id"], body.project, body.owner, write=True)
    if body.recursive or not body.path:
        return svc.drop_prefix(oid, body.project, body.path)
    return svc.drop_file(oid, body.project, body.path)


@router.get("/search")
async def search(
    q: str,
    max_results: int = 10,
    filter_by: str = "",
    owner: str = "",
    user: dict = Depends(get_current_user),
) -> dict[str, Any]:
    oid = await _eff_owner(user["id"], "", owner, write=False)
    return {"results": svc.search(oid, q, max_results=max_results, filter_by=filter_by)}


# ---------------------------------------------------------------------------
# Agent routes — no Typesense key ever leaves Flight Deck
# ---------------------------------------------------------------------------


def _require_connection() -> None:
    """Agents are told Typesense is always available, so a missing connection is
    Flight Deck's problem to explain — not something the agent should have to
    diagnose or configure on its side."""
    if not svc.configured():
        raise HTTPException(
            503,
            "Deep memory is not configured in Flight Deck. "
            "Open Flight Deck → Connections → Typesense and set the connection.",
        )
    if svc.get_index() is None:
        probe = svc.probe()
        raise HTTPException(503, f"Deep memory is unavailable: {probe.get('error', 'unknown error')}")


@router.post("/agent/search")
async def agent_search(body: AgentSearchBody, request: Request) -> dict[str, Any]:
    owner = _agent_owner(request)
    _require_connection()
    return {
        "results": svc.search(
            owner, body.query, max_results=body.max_results, filter_by=body.filter_by
        )
    }


@router.post("/agent/index")
async def agent_index(body: AgentIndexBody, request: Request) -> dict[str, Any]:
    """Index free text an agent produced (not a VFS file — that path is
    automatic). Stamped with the resolved owner, never a claimed one."""
    owner = _agent_owner(request)
    _require_connection()
    index = svc.get_index()
    if not body.text.strip():
        raise HTTPException(400, "text is required")
    import hashlib

    digest = hashlib.sha256(body.text.encode()).hexdigest()
    reference = body.reference or f"agent:{digest[:16]}"
    index.delete_by_reference(reference, owner_id=owner)
    chunks = index.index_document(
        doc_id=hashlib.sha1(f"{owner}:{reference}".encode()).hexdigest()[:16],
        text=body.text.strip(),
        source=body.source or "agent",
        reference=reference,
        path=reference,
        owner_id=owner,
        content_hash=digest,
        summarize=body.summarize,
    )
    return {"ok": True, "reference": reference, "chunks": chunks}


@router.post("/agent/delete")
async def agent_delete(body: AgentDeleteBody, request: Request) -> dict[str, Any]:
    """Delete from the caller's own archive.

    The owner filter is applied inside ``DeepMemoryIndex``, so a caller-supplied
    ``filter_by`` can narrow the deletion but never widen it past its own tenant.
    """
    owner = _agent_owner(request)
    _require_connection()
    index = svc.get_index()
    if body.reference.strip():
        deleted = index.delete_by_reference(body.reference.strip(), owner_id=owner)
    elif body.filter_by.strip():
        scope = f"owner_id:={index.escape_filter_value(owner)}"
        deleted = index.delete_by_filter(f"({body.filter_by.strip()}) && {scope}")
    else:
        raise HTTPException(400, "reference or filter_by is required")
    return {"ok": True, "deleted": deleted}
