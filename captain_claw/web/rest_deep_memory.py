"""REST API endpoints for the Deep Memory (Typesense) dashboard.

Provides browse, search, detail, delete, facet, and manual-index
operations against the Typesense-backed deep memory collection.
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any

import httpx
from aiohttp import web

from captain_claw.config import get_config
from captain_claw.logging import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TS_TIMEOUT = httpx.Timeout(15.0, connect=5.0)


def _get_ts_config(server: Any) -> dict[str, str]:
    """Extract Typesense connection params from agent or config."""
    agent = getattr(server, "agent", None)
    dm = getattr(agent, "_deep_memory", None) if agent else None
    if dm is not None:
        return {
            "base_url": dm._base_url,
            "api_key": dm._api_key,
            "collection": dm._collection_name,
        }
    cfg = get_config()
    dm_cfg = cfg.deep_memory
    return {
        "base_url": f"{dm_cfg.protocol}://{dm_cfg.host}:{dm_cfg.port}",
        "api_key": dm_cfg.api_key,
        "collection": dm_cfg.collection_name or "captain_claw_deep_memory",
    }


def _ts_headers(api_key: str) -> dict[str, str]:
    return {
        "X-TYPESENSE-API-KEY": api_key,
        "Content-Type": "application/json",
    }


def _hash_id(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# GET /api/deep-memory/status
# ---------------------------------------------------------------------------


async def get_status(server: Any, request: web.Request) -> web.Response:
    """Return Typesense connectivity status and collection stats."""
    ts = _get_ts_config(server)
    if not ts["api_key"]:
        return web.json_response(
            {"connected": False, "error": "No API key configured"},
            status=200,
        )
    try:
        async with httpx.AsyncClient(
            timeout=_TS_TIMEOUT, headers=_ts_headers(ts["api_key"])
        ) as client:
            resp = await client.get(
                f"{ts['base_url']}/collections/{ts['collection']}"
            )
            if resp.status_code == 404:
                return web.json_response({
                    "connected": True,
                    "collection": ts["collection"],
                    "num_documents": 0,
                    "num_chunks": 0,
                    "error": "Collection not found (will be created on first index)",
                })
            resp.raise_for_status()
            data = resp.json()
            return web.json_response({
                "connected": True,
                "collection": ts["collection"],
                "num_documents": data.get("num_documents", 0),
                "num_chunks": data.get("num_documents", 0),
                "fields": [f["name"] for f in data.get("fields", [])],
                "created_at": data.get("created_at", 0),
            })
    except Exception as exc:
        log.debug("Deep memory status check failed", error=str(exc))
        return web.json_response(
            {"connected": False, "error": str(exc)},
            status=200,
        )


# ---------------------------------------------------------------------------
# GET /api/deep-memory/documents
# ---------------------------------------------------------------------------


async def list_documents(server: Any, request: web.Request) -> web.Response:
    """List unique documents grouped by doc_id."""
    ts = _get_ts_config(server)
    q = request.query.get("q", "*").strip() or "*"
    source = request.query.get("source", "").strip()
    tag = request.query.get("tag", "").strip()
    page = max(1, int(request.query.get("page", "1")))
    per_page = min(250, max(1, int(request.query.get("per_page", "50"))))

    params: dict[str, Any] = {
        "q": q,
        "query_by": "text",
        "group_by": "doc_id",
        "group_limit": 1,
        "per_page": per_page,
        "page": page,
        "sort_by": "_text_match:desc,updated_at:desc" if q != "*" else "updated_at:desc",
    }

    # Build filter_by
    filters: list[str] = []
    if source:
        filters.append(f"source:={source}")
    if tag:
        filters.append(f"tags:={tag}")
    if filters:
        params["filter_by"] = " && ".join(filters)

    try:
        async with httpx.AsyncClient(
            timeout=_TS_TIMEOUT, headers=_ts_headers(ts["api_key"])
        ) as client:
            resp = await client.get(
                f"{ts['base_url']}/collections/{ts['collection']}/documents/search",
                params=params,
            )
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 404:
            return web.json_response({
                "documents": [], "total": 0, "page": page,
                "per_page": per_page, "found": 0,
            })
        return web.json_response({"error": str(exc)}, status=502)
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=502)

    documents: list[dict[str, Any]] = []
    for group in data.get("grouped_hits", []):
        hits = group.get("hits", [])
        if not hits:
            continue
        first_doc = hits[0].get("document", {})
        chunk_count = group.get("found", len(hits))
        snippet = first_doc.get("text", "")
        if len(snippet) > 200:
            snippet = snippet[:200] + "..."
        documents.append({
            "doc_id": first_doc.get("doc_id", ""),
            "source": first_doc.get("source", ""),
            "reference": first_doc.get("reference", ""),
            "path": first_doc.get("path", ""),
            "tags": first_doc.get("tags", []),
            "updated_at": first_doc.get("updated_at", 0),
            "chunk_count": chunk_count,
            "snippet": snippet,
        })

    return web.json_response({
        "documents": documents,
        "total": data.get("found", 0),
        "page": page,
        "per_page": per_page,
        "found": data.get("found", 0),
    })


# ---------------------------------------------------------------------------
# GET /api/deep-memory/documents/{doc_id}
# ---------------------------------------------------------------------------


async def get_document(server: Any, request: web.Request) -> web.Response:
    """Get all chunks for a specific document."""
    ts = _get_ts_config(server)
    doc_id = request.match_info["doc_id"]

    params: dict[str, Any] = {
        "q": "*",
        "filter_by": f"doc_id:={doc_id}",
        "sort_by": "chunk_index:asc",
        "per_page": 250,
    }

    try:
        async with httpx.AsyncClient(
            timeout=_TS_TIMEOUT, headers=_ts_headers(ts["api_key"])
        ) as client:
            resp = await client.get(
                f"{ts['base_url']}/collections/{ts['collection']}/documents/search",
                params=params,
            )
            resp.raise_for_status()
            data = resp.json()
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=502)

    hits = data.get("hits", [])
    if not hits:
        return web.json_response({"error": "Document not found"}, status=404)

    first_doc = hits[0].get("document", {})
    chunks: list[dict[str, Any]] = []
    for hit in hits:
        doc = hit.get("document", {})
        chunks.append({
            "chunk_index": doc.get("chunk_index", 0),
            "start_line": doc.get("start_line", 0),
            "end_line": doc.get("end_line", 0),
            "text": doc.get("text", ""),
        })

    return web.json_response({
        "doc_id": doc_id,
        "source": first_doc.get("source", ""),
        "reference": first_doc.get("reference", ""),
        "path": first_doc.get("path", ""),
        "tags": first_doc.get("tags", []),
        "updated_at": first_doc.get("updated_at", 0),
        "chunks": chunks,
    })


# ---------------------------------------------------------------------------
# DELETE /api/deep-memory/documents/{doc_id}
# ---------------------------------------------------------------------------


async def delete_document(server: Any, request: web.Request) -> web.Response:
    """Delete all chunks for a document."""
    ts = _get_ts_config(server)
    doc_id = request.match_info["doc_id"]

    try:
        async with httpx.AsyncClient(
            timeout=_TS_TIMEOUT, headers=_ts_headers(ts["api_key"])
        ) as client:
            resp = await client.delete(
                f"{ts['base_url']}/collections/{ts['collection']}/documents",
                params={"filter_by": f"doc_id:={doc_id}"},
            )
            resp.raise_for_status()
            result = resp.json()
            return web.json_response({
                "deleted": True,
                "num_deleted": result.get("num_deleted", 0),
            })
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=502)


# ---------------------------------------------------------------------------
# GET /api/deep-memory/facets
# ---------------------------------------------------------------------------


async def get_facets(server: Any, request: web.Request) -> web.Response:
    """Get facet values for source and tags."""
    ts = _get_ts_config(server)

    params: dict[str, Any] = {
        "q": "*",
        "query_by": "text",
        "facet_by": "source,tags",
        "per_page": 0,
        "max_facet_values": 100,
    }

    try:
        async with httpx.AsyncClient(
            timeout=_TS_TIMEOUT, headers=_ts_headers(ts["api_key"])
        ) as client:
            resp = await client.get(
                f"{ts['base_url']}/collections/{ts['collection']}/documents/search",
                params=params,
            )
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 404:
            return web.json_response({"sources": [], "tags": []})
        return web.json_response({"error": str(exc)}, status=502)
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=502)

    sources: list[dict[str, Any]] = []
    tags: list[dict[str, Any]] = []

    for facet in data.get("facet_counts", []):
        field_name = facet.get("field_name", "")
        values = [
            {"value": v.get("value", ""), "count": v.get("count", 0)}
            for v in facet.get("counts", [])
            if v.get("value")
        ]
        if field_name == "source":
            sources = values
        elif field_name == "tags":
            tags = values

    return web.json_response({"sources": sources, "tags": tags})


# ---------------------------------------------------------------------------
# POST /api/deep-memory/index
# ---------------------------------------------------------------------------


async def index_document(server: Any, request: web.Request) -> web.Response:
    """Index a new document manually."""
    ts = _get_ts_config(server)

    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON body"}, status=400)

    text = str(body.get("text", "")).strip()
    if not text:
        return web.json_response({"error": "text is required"}, status=400)

    source = str(body.get("source", "manual")).strip() or "manual"
    reference = str(body.get("reference", "")).strip()
    tags_raw = str(body.get("tags", "")).strip()
    tags = [t.strip() for t in tags_raw.split(",") if t.strip()] if tags_raw else []

    doc_id = _hash_id(f"{reference}:{text[:128]}")

    # Try using the agent's DeepMemoryIndex if available
    agent = getattr(server, "agent", None)
    dm = getattr(agent, "_deep_memory", None) if agent else None
    if dm is not None:
        try:
            chunk_count = dm.index_document(
                doc_id=doc_id,
                text=text,
                source=source,
                reference=reference,
                tags=tags or None,
            )
            return web.json_response({
                "doc_id": doc_id,
                "chunks_indexed": chunk_count,
            })
        except Exception as exc:
            log.warning("Deep memory index via agent failed, trying direct", error=str(exc))

    # Fallback: direct Typesense import
    from captain_claw.deep_memory import _chunk_text

    chunks = _chunk_text(text)
    if not chunks:
        return web.json_response({"error": "Text produced no chunks"}, status=400)

    now = int(time.time())
    docs: list[dict[str, Any]] = []
    for chunk in chunks:
        cid = _hash_id(f"{doc_id}:{chunk['chunk_index']}:{chunk['text'][:64]}")
        doc: dict[str, Any] = {
            "id": cid,
            "doc_id": doc_id,
            "source": source,
            "reference": reference,
            "path": reference,
            "text": chunk["text"],
            "chunk_index": chunk["chunk_index"],
            "start_line": chunk["start_line"],
            "end_line": chunk["end_line"],
            "updated_at": now,
        }
        if tags:
            doc["tags"] = tags
        docs.append(doc)

    lines = [json.dumps(d, ensure_ascii=False) for d in docs]
    body_str = "\n".join(lines)

    try:
        async with httpx.AsyncClient(
            timeout=_TS_TIMEOUT,
            headers={
                "X-TYPESENSE-API-KEY": ts["api_key"],
                "Content-Type": "text/plain",
            },
        ) as client:
            resp = await client.post(
                f"{ts['base_url']}/collections/{ts['collection']}/documents/import",
                params={"action": "upsert"},
                content=body_str,
            )
            resp.raise_for_status()
            result_lines = resp.text.strip().splitlines()
            ok = sum(
                1
                for line in result_lines
                if line.strip() and json.loads(line).get("success", False)
            )
            return web.json_response({
                "doc_id": doc_id,
                "chunks_indexed": ok,
            })
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=502)
