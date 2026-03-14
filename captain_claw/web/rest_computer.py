"""REST handlers for the Computer UI.

Provides the /api/computer/visualize endpoint that generates an HTML page
from the user's prompt and the agent's result, styled to match the active
retro theme.  Also provides CRUD endpoints for the exploration tree.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from typing import TYPE_CHECKING

from aiohttp import web

from captain_claw.logging import get_logger

if TYPE_CHECKING:
    from captain_claw.web_server import WebServer

log = get_logger(__name__)

# Visual generation token tiers — selectable from the UI.
VISUAL_TOKEN_TIERS: dict[str, int] = {
    "micro": 4000,
    "minimal": 8000,
    "standard": 16000,
    "generous": 32000,
}
_DEFAULT_TIER = "standard"


async def computer_visualize(
    server: "WebServer", request: web.Request,
) -> web.Response:
    """POST /api/computer/visualize — generate themed HTML from prompt+result."""
    from captain_claw.llm import Message

    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON body"}, status=400)

    prompt = str(body.get("prompt", "")).strip()
    result = str(body.get("result", "")).strip()
    theme_instructions = str(body.get("theme_instructions", "")).strip()

    theme = str(body.get("theme", "")).strip()

    # Token tier selection (micro / minimal / standard / generous).
    tier = str(body.get("token_tier", _DEFAULT_TIER)).strip().lower()
    if tier not in VISUAL_TOKEN_TIERS:
        tier = _DEFAULT_TIER
    max_tokens = VISUAL_TOKEN_TIERS[tier]

    # Optional model override — id from allowed_models list.
    model_override = str(body.get("model", "")).strip() or None

    if not prompt and not result:
        return web.json_response(
            {"error": "Both prompt and result are empty"}, status=400,
        )

    log.info(
        "Visual generation started",
        theme=theme,
        tier=tier,
        max_tokens=max_tokens,
        model_override=model_override or "(default)",
        prompt_len=len(prompt),
        result_len=len(result),
        prompt_preview=prompt[:120] + ("…" if len(prompt) > 120 else ""),
    )

    # Truncate very long results to avoid token blowup.
    max_result_len = max_tokens  # scale truncation to tier
    truncated = len(result) > max_result_len
    if truncated:
        result = result[:max_result_len] + "\n\n[...truncated for visual generation]"
        log.info("Result truncated", original_len=len(result), max_len=max_result_len)

    # Load prompts from externalized instruction files (supports micro + personal overrides).
    instructions = server.agent.instructions if server.agent else None
    if instructions is None:
        from captain_claw.instructions import InstructionLoader
        instructions = InstructionLoader()

    system_content = instructions.load("computer_visualize_system_prompt.md")
    user_content = instructions.render(
        "computer_visualize_user_prompt.md",
        prompt=prompt,
        result=result,
        theme_instructions=theme_instructions or "Use a clean, modern dark theme.",
    )

    # Resolve provider — optionally override model.
    provider = _resolve_provider(server, model_override)

    provider_name = str(getattr(provider, "provider_name", "") or "")
    model_hint = str(getattr(provider, "model", "") or "")
    log.info(
        "Calling LLM for visual generation",
        provider=provider_name,
        model=model_hint,
        system_prompt_len=len(system_content),
        user_prompt_len=len(user_content),
        max_tokens=max_tokens,
        tier=tier,
    )

    messages = [
        Message(role="system", content=system_content),
        Message(role="user", content=user_content),
    ]

    try:
        t0 = time.monotonic()
        response = await asyncio.wait_for(
            provider.complete(messages=messages, tools=None, max_tokens=max_tokens),
            timeout=120.0,
        )
        latency_ms = int((time.monotonic() - t0) * 1000)
        raw = str(getattr(response, "content", "") or "").strip()

        # Log LLM response details.
        usage = getattr(response, "usage", {}) or {}
        response_model = str(getattr(response, "model", "") or "")
        log.info(
            "Visual generation LLM response",
            model=response_model,
            latency_ms=latency_ms,
            raw_len=len(raw),
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
        )

        # Record LLM usage for cost tracking.
        _record_visual_usage(server, messages, response, latency_ms, max_tokens, provider)

        # Strip markdown code fences if the LLM wrapped the HTML.
        html = re.sub(r"^```(?:html)?\s*\n?", "", raw)
        html = re.sub(r"\n?```\s*$", "", html)

        if not html:
            log.warning("LLM returned empty HTML after stripping fences", raw_len=len(raw))
            return web.json_response(
                {"error": "LLM returned empty response"}, status=500,
            )

        log.info(
            "Visual generation complete",
            html_len=len(html),
            latency_ms=latency_ms,
            theme=theme,
        )
        return web.json_response({"html": html})

    except asyncio.TimeoutError:
        latency_ms = int((time.monotonic() - t0) * 1000)
        log.warning("Visual generation timed out", latency_ms=latency_ms, theme=theme)
        return web.json_response(
            {"error": "Visual generation timed out"}, status=504,
        )
    except Exception as exc:
        latency_ms = int((time.monotonic() - t0) * 1000)
        log.error("Visual generation failed", error=str(exc), latency_ms=latency_ms, theme=theme)
        return web.json_response(
            {"error": f"Visual generation failed: {exc}"}, status=500,
        )


def _resolve_provider(server: "WebServer", model_id: str | None):
    """Return an LLM provider, optionally overriding the model.

    If *model_id* matches an entry in the allowed-models list, a new
    provider is created for that model.  Otherwise the session provider
    (or global default) is returned.
    """
    if model_id and server.agent:
        try:
            allowed = server.agent.get_allowed_models()
            for m in allowed:
                if m.get("id") == model_id or m.get("model") == model_id:
                    from captain_claw.llm import create_provider
                    from captain_claw.config import get_config

                    cfg = get_config()
                    provider_name = m.get("provider", cfg.model.provider)
                    headers = cfg.provider_keys.headers_for(provider_name) or None
                    api_key = None if headers else (m.get("api_key") or cfg.model.api_key or None)
                    return create_provider(
                        provider=provider_name,
                        model=m.get("model", cfg.model.model),
                        temperature=m.get("temperature") or cfg.model.temperature,
                        max_tokens=m.get("max_tokens") or cfg.model.max_tokens,
                        api_key=api_key,
                        base_url=m.get("base_url") or cfg.model.base_url or None,
                        extra_headers=headers,
                    )
        except Exception as exc:
            log.warning("Failed to resolve model override", model_id=model_id, error=str(exc))

    # Fall back to session/global provider.
    provider = server.agent.provider if server.agent else None
    if provider is None:
        from captain_claw.llm import get_provider
        provider = get_provider()
    return provider


def _record_visual_usage(
    server: "WebServer",
    messages: list,
    response: object,
    latency_ms: int,
    max_tokens: int = 16000,
    provider: object = None,
) -> None:
    """Fire-and-forget: persist LLM usage for the visual generation call."""
    try:
        from captain_claw.session import get_session_manager

        usage = getattr(response, "usage", {}) or {}
        model_name = str(getattr(response, "model", "") or "")
        # Use the actual provider that was used for this call.
        provider_name = str(getattr(provider, "provider_name", "") or "") if provider else ""
        if not provider_name and server.agent:
            provider_name = str(getattr(server.agent.provider, "provider_name", "") or "")
        content = str(getattr(response, "content", "") or "")
        session_id = server.agent.session.id if server.agent and server.agent.session else None

        input_bytes = 0
        for m in messages:
            c = getattr(m, "content", None) or ""
            if isinstance(c, str):
                input_bytes += len(c.encode("utf-8", errors="replace"))
        output_bytes = len(content.encode("utf-8", errors="replace"))

        sm = get_session_manager()
        loop = asyncio.get_event_loop()
        loop.create_task(sm.record_llm_usage(
            session_id=session_id,
            interaction="computer_visualize",
            provider=provider_name,
            model=model_name,
            prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
            completion_tokens=int(usage.get("completion_tokens", 0) or 0),
            total_tokens=int(usage.get("total_tokens", 0) or 0),
            cache_creation_input_tokens=int(usage.get("cache_creation_input_tokens", 0) or 0),
            cache_read_input_tokens=int(usage.get("cache_read_input_tokens", 0) or 0),
            input_bytes=input_bytes,
            output_bytes=output_bytes,
            max_tokens=max_tokens,
            latency_ms=latency_ms,
            task_name="computer_visual",
        ))
    except Exception as exc:
        log.warning("Failed to record visual generation usage", error=str(exc))


# ═══════════════════════════════════════════════════════════════
# Exploration tree CRUD
# ═══════════════════════════════════════════════════════════════

_EXPLORATION_COLS = (
    "id, session_id, parent_id, edge_label, prompt, answer, "
    "visual_html, theme, source, created_at, metadata"
)


async def _ensure_exploration_table(server: "WebServer") -> None:
    """Create the exploration_nodes table if it doesn't exist."""
    from captain_claw.session import get_session_manager
    sm = get_session_manager()
    await sm._ensure_db()
    assert sm._db is not None
    await sm._db.execute("""
        CREATE TABLE IF NOT EXISTS exploration_nodes (
            id          TEXT PRIMARY KEY,
            session_id  TEXT NOT NULL,
            parent_id   TEXT,
            edge_label  TEXT,
            prompt      TEXT NOT NULL,
            answer      TEXT NOT NULL,
            visual_html TEXT,
            theme       TEXT NOT NULL DEFAULT '',
            source      TEXT NOT NULL DEFAULT 'manual',
            created_at  TEXT NOT NULL,
            metadata    TEXT NOT NULL DEFAULT '{}'
        )
    """)
    await sm._db.execute(
        "CREATE INDEX IF NOT EXISTS idx_exploration_session "
        "ON exploration_nodes(session_id, created_at)"
    )
    await sm._db.execute(
        "CREATE INDEX IF NOT EXISTS idx_exploration_parent "
        "ON exploration_nodes(parent_id)"
    )
    await sm._db.commit()


async def exploration_save(
    server: "WebServer", request: web.Request,
) -> web.Response:
    """POST /api/computer/exploration — save a new exploration node."""
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON"}, status=400)

    required = ("id", "prompt", "answer")
    for key in required:
        if not body.get(key):
            return web.json_response({"error": f"Missing {key}"}, status=400)

    await _ensure_exploration_table(server)

    from captain_claw.session import get_session_manager
    sm = get_session_manager()
    assert sm._db is not None

    await sm._db.execute(
        f"INSERT OR REPLACE INTO exploration_nodes ({_EXPLORATION_COLS}) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            str(body["id"]),
            str(body.get("session_id", "")),
            body.get("parent_id"),
            body.get("edge_label"),
            str(body["prompt"]),
            str(body["answer"]),
            body.get("visual_html"),
            str(body.get("theme", "")),
            str(body.get("source", "manual")),
            str(body.get("created_at", "")),
            json.dumps(body.get("metadata", {})) if isinstance(body.get("metadata"), dict) else str(body.get("metadata", "{}")),
        ),
    )
    await sm._db.commit()
    log.info("Exploration node saved", node_id=body["id"], source=body.get("source", "manual"))
    return web.json_response({"ok": True})


async def exploration_list(
    server: "WebServer", request: web.Request,
) -> web.Response:
    """GET /api/computer/exploration?session_id=... — list exploration nodes."""
    session_id = request.query.get("session_id", "")

    await _ensure_exploration_table(server)

    from captain_claw.session import get_session_manager
    sm = get_session_manager()
    assert sm._db is not None

    query = f"SELECT {_EXPLORATION_COLS} FROM exploration_nodes"
    params: tuple = ()
    if session_id:
        query += " WHERE session_id = ?"
        params = (session_id,)
    query += " ORDER BY created_at ASC"

    async with sm._db.execute(query, params) as cursor:
        rows = await cursor.fetchall()

    cols = [c.strip() for c in _EXPLORATION_COLS.split(",")]
    nodes = []
    for row in rows:
        node = {}
        for i, col in enumerate(cols):
            node[col] = row[i]
        nodes.append(node)

    return web.json_response({"nodes": nodes})


async def exploration_get(
    server: "WebServer", request: web.Request,
) -> web.Response:
    """GET /api/computer/exploration/{id} — get a single exploration node."""
    node_id = request.match_info["id"]

    await _ensure_exploration_table(server)

    from captain_claw.session import get_session_manager
    sm = get_session_manager()
    assert sm._db is not None

    async with sm._db.execute(
        f"SELECT {_EXPLORATION_COLS} FROM exploration_nodes WHERE id = ?",
        (node_id,),
    ) as cursor:
        row = await cursor.fetchone()

    if not row:
        return web.json_response({"error": "Not found"}, status=404)

    cols = [c.strip() for c in _EXPLORATION_COLS.split(",")]
    node = {}
    for i, col in enumerate(cols):
        node[col] = row[i]

    return web.json_response({"node": node})


async def exploration_update_visual(
    server: "WebServer", request: web.Request,
) -> web.Response:
    """PUT /api/computer/exploration/{id}/visual — update visual_html for a node."""
    node_id = request.match_info["id"]

    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON"}, status=400)

    visual_html = body.get("visual_html")

    await _ensure_exploration_table(server)

    from captain_claw.session import get_session_manager
    sm = get_session_manager()
    assert sm._db is not None

    await sm._db.execute(
        "UPDATE exploration_nodes SET visual_html = ? WHERE id = ?",
        (visual_html, node_id),
    )
    await sm._db.commit()

    return web.json_response({"ok": True})


async def exploration_delete(
    server: "WebServer", request: web.Request,
) -> web.Response:
    """DELETE /api/computer/exploration/{id} — delete an exploration node."""
    node_id = request.match_info["id"]

    await _ensure_exploration_table(server)

    from captain_claw.session import get_session_manager
    sm = get_session_manager()
    assert sm._db is not None

    await sm._db.execute(
        "DELETE FROM exploration_nodes WHERE id = ?", (node_id,),
    )
    await sm._db.commit()
    log.info("Exploration node deleted", node_id=node_id)

    return web.json_response({"ok": True})
