"""REST API endpoints for the Semantic Memory (SQLite) browser.

Provides status, document listing, chunk browsing with L1/L2/L3 layer
support, search, and promote operations.
"""

from __future__ import annotations

from typing import Any

from aiohttp import web

from captain_claw.logging import get_logger

log = get_logger(__name__)


def _get_semantic(server: Any):
    """Return the SemanticMemoryIndex from the agent, or None."""
    agent = getattr(server, "agent", None)
    memory = getattr(agent, "memory", None) if agent else None
    return getattr(memory, "semantic", None) if memory else None


# ---------------------------------------------------------------------------
# GET /api/semantic-memory/status
# ---------------------------------------------------------------------------

async def get_status(server: Any, request: web.Request) -> web.Response:
    semantic = _get_semantic(server)
    if semantic is None:
        return web.json_response({"enabled": False, "documents": 0, "chunks": 0})
    try:
        conn = semantic._conn_or_raise()
        doc_count = conn.execute("SELECT COUNT(*) FROM memory_documents").fetchone()[0]
        chunk_count = conn.execute("SELECT COUNT(*) FROM memory_chunks").fetchone()[0]
        has_l1 = conn.execute(
            "SELECT COUNT(*) FROM memory_chunks WHERE text_l1 != ''"
        ).fetchone()[0]
        has_l2 = conn.execute(
            "SELECT COUNT(*) FROM memory_chunks WHERE text_l2 != ''"
        ).fetchone()[0]
        return web.json_response({
            "enabled": True,
            "documents": doc_count,
            "chunks": chunk_count,
            "chunks_with_l1": has_l1,
            "chunks_with_l2": has_l2,
            "layered_summaries": semantic.layered_summaries,
        })
    except Exception as e:
        return web.json_response({"enabled": True, "error": str(e)}, status=500)


# ---------------------------------------------------------------------------
# GET /api/semantic-memory/documents
# ---------------------------------------------------------------------------

async def list_documents(server: Any, request: web.Request) -> web.Response:
    semantic = _get_semantic(server)
    if semantic is None:
        return web.json_response({"documents": []})
    try:
        conn = semantic._conn_or_raise()
        source_filter = request.query.get("source", "").strip()
        where = ""
        params: tuple = ()
        if source_filter:
            where = "WHERE d.source = ?"
            params = (source_filter,)
        rows = conn.execute(
            f"""
            SELECT d.doc_id, d.source, d.reference, d.path, d.updated_at,
                   COUNT(c.chunk_id) AS chunk_count
            FROM memory_documents d
            LEFT JOIN memory_chunks c ON c.doc_id = d.doc_id
            {where}
            GROUP BY d.doc_id
            ORDER BY d.updated_at DESC
            LIMIT 200
            """,
            params,
        ).fetchall()
        docs = []
        for row in rows:
            docs.append({
                "doc_id": row[0],
                "source": row[1],
                "reference": row[2],
                "path": row[3],
                "updated_at": row[4],
                "chunk_count": row[5],
            })
        # Distinct sources for filter dropdown.
        sources = [
            r[0] for r in conn.execute(
                "SELECT DISTINCT source FROM memory_documents ORDER BY source"
            ).fetchall()
        ]
        return web.json_response({"documents": docs, "sources": sources})
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)


# ---------------------------------------------------------------------------
# GET /api/semantic-memory/documents/{doc_id}
# ---------------------------------------------------------------------------

async def get_document_chunks(server: Any, request: web.Request) -> web.Response:
    semantic = _get_semantic(server)
    if semantic is None:
        return web.json_response({"error": "Semantic memory not available"}, status=404)
    doc_id = request.match_info["doc_id"]
    try:
        conn = semantic._conn_or_raise()
        doc_row = conn.execute(
            "SELECT doc_id, source, reference, path, updated_at FROM memory_documents WHERE doc_id = ?",
            (doc_id,),
        ).fetchone()
        if not doc_row:
            return web.json_response({"error": "Document not found"}, status=404)
        chunks = conn.execute(
            """
            SELECT chunk_id, chunk_index, start_line, end_line, text, text_l1, text_l2, updated_at
            FROM memory_chunks
            WHERE doc_id = ?
            ORDER BY chunk_index
            """,
            (doc_id,),
        ).fetchall()
        chunk_list = []
        for c in chunks:
            chunk_list.append({
                "chunk_id": c[0],
                "chunk_index": c[1],
                "start_line": c[2],
                "end_line": c[3],
                "text": c[4],
                "text_l1": c[5],
                "text_l2": c[6],
                "updated_at": c[7],
            })
        return web.json_response({
            "document": {
                "doc_id": doc_row[0],
                "source": doc_row[1],
                "reference": doc_row[2],
                "path": doc_row[3],
                "updated_at": doc_row[4],
            },
            "chunks": chunk_list,
        })
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)


# ---------------------------------------------------------------------------
# GET /api/semantic-memory/search?q=...&layer=l1|l2|l3&max=10
# ---------------------------------------------------------------------------

async def search(server: Any, request: web.Request) -> web.Response:
    semantic = _get_semantic(server)
    if semantic is None:
        return web.json_response({"results": []})
    query = request.query.get("q", "").strip()
    if not query:
        return web.json_response({"results": [], "query": ""})
    layer = request.query.get("layer", "l3").strip()
    if layer not in ("l1", "l2", "l3"):
        layer = "l3"
    max_results = min(int(request.query.get("max", "20")), 100)
    try:
        results = semantic.search(query, max_results=max_results)
        items = []
        for r in results:
            snippet = semantic._pick_layer_text(
                layer, text=r.snippet, text_l1=r.text_l1, text_l2=r.text_l2,
            )
            items.append({
                "chunk_id": r.chunk_id,
                "source": r.source,
                "reference": r.reference,
                "path": r.path,
                "start_line": r.start_line,
                "end_line": r.end_line,
                "snippet": snippet,
                "text_l1": r.text_l1,
                "text_l2": r.text_l2,
                "text_l3": r.snippet,
                "score": round(r.score, 4),
                "text_score": round(r.text_score, 4),
                "vector_score": round(r.vector_score, 4),
                "updated_at": r.updated_at,
            })
        return web.json_response({"results": items, "query": query, "layer": layer})
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)


# ---------------------------------------------------------------------------
# GET /api/semantic-memory/promote?ids=id1,id2&layer=l3
# ---------------------------------------------------------------------------

async def promote(server: Any, request: web.Request) -> web.Response:
    semantic = _get_semantic(server)
    if semantic is None:
        return web.json_response({"results": []})
    ids_raw = request.query.get("ids", "").strip()
    if not ids_raw:
        return web.json_response({"results": []})
    chunk_ids = [x.strip() for x in ids_raw.split(",") if x.strip()]
    layer = request.query.get("layer", "l3").strip()
    if layer not in ("l1", "l2", "l3"):
        layer = "l3"
    try:
        results = semantic.promote(chunk_ids, layer=layer)
        items = []
        for r in results:
            items.append({
                "chunk_id": r.chunk_id,
                "source": r.source,
                "path": r.path,
                "start_line": r.start_line,
                "end_line": r.end_line,
                "snippet": r.snippet,
                "text_l1": r.text_l1,
                "text_l2": r.text_l2,
            })
        return web.json_response({"results": items, "layer": layer})
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)
