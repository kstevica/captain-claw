"""Cross-machine memory export / import.

Bundles the *curated* memory layers (insights + reflections) into a single
JSON envelope that can be moved between Captain Claw agents. The raw layers
(working memory, semantic memory, deep memory, datastore) are intentionally
NOT exported — they are machine-local by design.

See ``MEMORY_STRUCTURE.md`` for the full architecture and the rationale.

Endpoints
---------
GET  /api/memory/export   – download the curated memory bundle as JSON
POST /api/memory/import   – merge a bundle into this agent's memory

The export filters insights by minimum importance (default 0 = everything).
The import always:
  * dedupes insights via the existing entity_key + FTS5 logic in ``add()``
  * tags imported insights with ``imported-from:<source_label>``
  * stages imported reflections under ``reflections/imported/<source>/`` so
    the agent's *active* reflection (loaded only from the top-level dir) is
    never silently overwritten
"""

from __future__ import annotations

import json
import socket
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from aiohttp import web

from captain_claw.insights import (
    get_insights_manager,
    get_session_insights_manager,
)
from captain_claw.logging import get_logger
from captain_claw.reflections import (
    import_reflections_archive,
    list_imported_reflections,
    list_reflections,
    merge_reflection_with_import,
    reflection_to_dict,
)

if TYPE_CHECKING:
    from captain_claw.web_server import WebServer

log = get_logger(__name__)

# schema_version history:
#   1 — curated-only bundle (insights + reflections)
#   2 — adds optional `semantic_chunks` list (text + metadata, no vectors;
#        re-embedded by the target on import)
SCHEMA_VERSION = 2
SUPPORTED_SCHEMA_VERSIONS = {1, 2}


def _resolve_insights_manager(request: web.Request) -> Any:
    """Same per-session resolution rule as ``rest_insights.py``."""
    from captain_claw.web.public_auth import get_request_session_id
    is_public, session_id = get_request_session_id(request)
    if is_public and session_id:
        return get_session_insights_manager(session_id)
    return get_insights_manager()


def _get_semantic_index(server: WebServer) -> Any:
    """Return the agent's SemanticMemoryIndex, or None if not wired up."""
    agent = getattr(server, "agent", None)
    memory = getattr(agent, "memory", None) if agent is not None else None
    return getattr(memory, "semantic", None) if memory is not None else None


def _safe_label(value: str) -> str:
    """Sanitize an arbitrary string into a filename/tag-safe label."""
    cleaned = "".join(c if c.isalnum() or c in "._-" else "_" for c in value)
    return cleaned.strip("._-") or "imported"


# ── Export ──────────────────────────────────────────────────────────


async def export_memory(server: WebServer, request: web.Request) -> web.Response:
    """GET /api/memory/export — download curated memory bundle.

    Query params:
      min_importance     — int, default 0; filter insights below this score
      include_expired    — '1' to keep expired insights (default: drop)
      reflection_limit   — int, default 50; max reflections to include
    """
    min_importance = int(request.query.get("min_importance", "0"))
    include_expired = request.query.get("include_expired", "").lower() in ("1", "true", "yes")
    reflection_limit = int(request.query.get("reflection_limit", "50"))
    include_semantic = request.query.get("include_semantic", "").lower() in ("1", "true", "yes")
    semantic_limit = int(request.query.get("semantic_limit", "1000"))
    semantic_min_chars = int(request.query.get("semantic_min_chars", "100"))
    semantic_include_imported = request.query.get(
        "semantic_include_imported", ""
    ).lower() in ("1", "true", "yes")
    # Comma-separated list like "workspace,session,manual" — empty = all.
    semantic_sources_raw = (request.query.get("semantic_sources") or "").strip()
    semantic_sources = (
        [s.strip() for s in semantic_sources_raw.split(",") if s.strip()]
        if semantic_sources_raw
        else None
    )

    mgr = _resolve_insights_manager(request)
    insights = await mgr.export_all(
        min_importance=min_importance,
        include_expired=include_expired,
    )
    reflections = [reflection_to_dict(r) for r in list_reflections(limit=reflection_limit)]

    semantic_chunks: list[dict[str, Any]] = []
    semantic_provider_key = ""
    if include_semantic:
        index = _get_semantic_index(server)
        if index is not None:
            try:
                semantic_chunks = index.export_chunks(
                    min_chars=semantic_min_chars,
                    limit=semantic_limit,
                    include_sources=semantic_sources,
                    include_imported=semantic_include_imported,
                )
                semantic_provider_key = (
                    index.embedding_chain.active_provider_key or ""
                )
            except Exception as exc:
                log.warning("Semantic export failed", error=str(exc))

    bundle: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "exported_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "source": {
            "host": socket.gethostname(),
            "min_importance": min_importance,
            "include_expired": include_expired,
            "semantic_included": include_semantic,
            "semantic_provider_key": semantic_provider_key,
        },
        "insights": insights,
        "reflections": reflections,
        "counts": {
            "insights": len(insights),
            "reflections": len(reflections),
            "semantic_chunks": len(semantic_chunks),
        },
    }
    if include_semantic:
        # Only include the key when the caller asked — avoids confusing
        # older importers that don't know about it.
        bundle["semantic_chunks"] = semantic_chunks

    log.info(
        "Memory bundle exported",
        insights=len(insights),
        reflections=len(reflections),
        semantic_chunks=len(semantic_chunks),
        min_importance=min_importance,
    )

    body = json.dumps(bundle, default=str, indent=2)
    safe_host = _safe_label(socket.gethostname())
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    filename = f"captain-claw-memory-{safe_host}-{timestamp}.json"

    return web.Response(
        body=body,
        content_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ── Import ──────────────────────────────────────────────────────────


async def import_memory(server: WebServer, request: web.Request) -> web.Response:
    """POST /api/memory/import — merge a bundle into this agent.

    Body: the JSON bundle produced by /api/memory/export.

    Optional query params:
      min_importance — int, default 0; drop incoming insights below this score
                        (e.g. dev→prod transfer should use 7)
      source_label   — string, default = bundle.source.host or 'imported';
                        used as the dedup tag and reflection subdir name
    """
    try:
        bundle = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON body"}, status=400)

    if not isinstance(bundle, dict):
        return web.json_response({"error": "Bundle must be a JSON object"}, status=400)

    schema = int(bundle.get("schema_version", 0))
    if schema not in SUPPORTED_SCHEMA_VERSIONS:
        return web.json_response(
            {
                "error": (
                    f"Unsupported schema_version {schema} "
                    f"(supported: {sorted(SUPPORTED_SCHEMA_VERSIONS)})"
                )
            },
            status=400,
        )

    min_importance = int(request.query.get("min_importance", "0"))
    stage_conflicts = request.query.get("stage_conflicts", "").lower() in ("1", "true", "yes")
    # Opt-out knob: callers can drop semantic chunks from an otherwise
    # valid v2 bundle (e.g. a quick curated-only re-merge on an agent
    # that happens to have embeddings disabled).
    skip_semantic = request.query.get("skip_semantic", "").lower() in ("1", "true", "yes")
    default_label = (
        bundle.get("source", {}).get("host")
        if isinstance(bundle.get("source"), dict) else None
    ) or "imported"
    source_label = _safe_label(request.query.get("source_label") or default_label)

    insights_in = bundle.get("insights") or []
    reflections_in = bundle.get("reflections") or []
    semantic_in = bundle.get("semantic_chunks") or []
    if not isinstance(insights_in, list) or not isinstance(reflections_in, list):
        return web.json_response(
            {"error": "Bundle 'insights' and 'reflections' must be lists"},
            status=400,
        )
    if not isinstance(semantic_in, list):
        return web.json_response(
            {"error": "Bundle 'semantic_chunks' must be a list if present"},
            status=400,
        )

    mgr = _resolve_insights_manager(request)
    insights_stats = await mgr.import_items(
        insights_in,
        min_importance=min_importance,
        source_label=source_label,
        stage_conflicts=stage_conflicts,
    )
    reflections_stats = import_reflections_archive(
        reflections_in,
        source_label=source_label,
    )

    semantic_stats: dict[str, Any] = {
        "chunks_inserted": 0,
        "chunks_skipped": 0,
        "docs_upserted": 0,
        "embedded": 0,
        "embedding_skipped": False,
        "available": False,
    }
    if semantic_in and not skip_semantic:
        index = _get_semantic_index(server)
        if index is None:
            semantic_stats["note"] = (
                "Semantic index not available on this agent — chunks skipped."
            )
        else:
            try:
                result = index.import_chunks(
                    semantic_in,
                    source_label=source_label,
                    re_embed=True,
                )
                semantic_stats.update(result)
                semantic_stats["available"] = True
            except Exception as exc:
                log.exception("Semantic import failed")
                semantic_stats["error"] = str(exc)

    log.info(
        "Memory bundle imported",
        source=source_label,
        schema_version=schema,
        insights=insights_stats,
        reflections=reflections_stats,
        semantic=semantic_stats,
    )

    return web.json_response({
        "ok": True,
        "source_label": source_label,
        "schema_version": schema,
        "min_importance": min_importance,
        "stage_conflicts": stage_conflicts,
        "insights": insights_stats,
        "reflections": reflections_stats,
        "semantic": semantic_stats,
        "note": (
            "Reflections were staged under reflections/imported/<source>/ — "
            "the active reflection was not modified. Use /reflection merge "
            "<label> (or the Merge Reflection action) to promote a personality."
        ),
    })


# ── Reflection merge ────────────────────────────────────────────────


async def list_imported(server: WebServer, request: web.Request) -> web.Response:
    """GET /api/memory/reflections/imported — list staged imported reflections.

    Returns one entry per ``reflections/imported/<label>/`` subdir with the
    parsed reflection objects inside, newest first. Used by the Flight Deck
    merge picker and the ``/reflection merge`` slash command.
    """
    sources = list_imported_reflections()
    return web.json_response({"sources": sources, "count": len(sources)})


async def merge_reflection(server: WebServer, request: web.Request) -> web.Response:
    """POST /api/memory/reflections/merge — personality-preserving promotion.

    Body: ``{"label": "<imported-subdir>", "filename": "<optional>"}``

    Runs the agent's reflection merge LLM flow with the current active
    reflection plus the imported one as input, then saves the result as the
    new top-level (active) reflection.
    """
    try:
        payload = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON body"}, status=400)

    if not isinstance(payload, dict):
        return web.json_response({"error": "Body must be a JSON object"}, status=400)

    label = str(payload.get("label") or "").strip()
    filename = payload.get("filename")
    if not label:
        return web.json_response({"error": "Missing 'label'"}, status=400)

    if not server.agent:
        return web.json_response({"error": "Agent not available"}, status=503)

    try:
        merged = await merge_reflection_with_import(
            server.agent,
            label=label,
            filename=str(filename).strip() if filename else None,
        )
    except FileNotFoundError as exc:
        return web.json_response({"error": str(exc)}, status=404)
    except ValueError as exc:
        return web.json_response({"error": str(exc)}, status=400)
    except Exception as exc:
        log.exception("Reflection merge failed")
        return web.json_response({"error": f"Merge failed: {exc}"}, status=500)

    return web.json_response(
        {
            "ok": True,
            "label": label,
            "reflection": reflection_to_dict(merged),
            "note": (
                "The merged reflection is now the active personality. The "
                "imported file was left in place so you can re-run the merge "
                "or pick a different source."
            ),
        }
    )


# ── Semantic import management ──────────────────────────────────────


async def list_semantic_imports(server: WebServer, request: web.Request) -> web.Response:
    """GET /api/memory/semantic/labels — list imported semantic sources."""
    index = _get_semantic_index(server)
    if index is None:
        return web.json_response({"available": False, "labels": []})
    labels = index.list_import_labels()
    provider_key = index.embedding_chain.active_provider_key or ""
    return web.json_response(
        {
            "available": True,
            "provider_key": provider_key,
            "labels": labels,
            "count": len(labels),
        }
    )


async def delete_semantic_import(server: WebServer, request: web.Request) -> web.Response:
    """DELETE /api/memory/semantic/labels/{label} — purge one imported source."""
    index = _get_semantic_index(server)
    if index is None:
        return web.json_response(
            {"error": "Semantic index not available on this agent"}, status=503
        )
    label = request.match_info.get("label") or ""
    if not label.strip():
        return web.json_response({"error": "Missing label"}, status=400)
    try:
        result = index.delete_imported(label)
    except Exception as exc:
        log.exception("Semantic delete failed")
        return web.json_response({"error": str(exc)}, status=500)
    return web.json_response({"ok": True, "label": label, **result})
