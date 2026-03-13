"""REST handlers for the visualization style profile.

Provides endpoints for viewing, updating, and analyzing references
to extract visual style information (colors, fonts, layout rules).
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import mimetypes
import re
from pathlib import Path
from typing import TYPE_CHECKING

from aiohttp import web

from captain_claw.logging import get_logger

if TYPE_CHECKING:
    from captain_claw.web_server import WebServer

log = get_logger(__name__)

_IMAGE_EXTENSIONS: set[str] = {".png", ".jpg", ".jpeg", ".webp"}
_DOCUMENT_EXTENSIONS: set[str] = {".pdf", ".docx"}
_ALL_ALLOWED = _IMAGE_EXTENSIONS | _DOCUMENT_EXTENSIONS


# ── Helpers ──────────────────────────────────────────────────────────


def _merge_style_fields(s: "VisualizationStyle", body: dict) -> None:  # type: ignore[name-defined]
    """Apply body fields to style *s* in-place."""
    if "name" in body:
        name = str(body["name"]).strip()
        if name:
            s.name = name

    if "color_palette" in body:
        raw = body["color_palette"]
        if isinstance(raw, list):
            s.color_palette = [str(c).strip() for c in raw if str(c).strip()]
        elif isinstance(raw, str):
            s.color_palette = [c.strip() for c in raw.split(",") if c.strip()]

    for field in (
        "font_primary", "font_headings", "font_mono",
        "background_style", "chart_style",
        "layout_notes", "additional_rules", "source_description",
    ):
        if field in body:
            setattr(s, field, str(body[field]).strip())


def _clear_instruction_caches(server: "WebServer") -> None:
    """Clear instruction caches so prompts are rebuilt with new style."""
    if server.agent:
        server.agent.instructions._cache.pop("system_prompt.md", None)
        server.agent.instructions._cache.pop("micro_system_prompt.md", None)
    for agent in getattr(server, "_telegram_agents", {}).values():
        if hasattr(agent, "instructions") and hasattr(agent.instructions, "_cache"):
            agent.instructions._cache.pop("system_prompt.md", None)
            agent.instructions._cache.pop("micro_system_prompt.md", None)


# ── GET / PUT ────────────────────────────────────────────────────────


async def get_visualization_style(
    server: WebServer, request: web.Request,
) -> web.Response:
    """GET /api/visualization-style — return the style as a JSON object."""
    from captain_claw.visualization_style import (
        load_visualization_style,
        visualization_style_to_dict,
    )

    s = load_visualization_style()
    return web.json_response(visualization_style_to_dict(s))


async def put_visualization_style(
    server: WebServer, request: web.Request,
) -> web.Response:
    """PUT /api/visualization-style — update style fields from a JSON body.

    Only provided keys are updated; omitted keys are left unchanged.
    """
    from captain_claw.visualization_style import (
        VisualizationStyle,
        load_visualization_style,
        save_visualization_style,
        visualization_style_to_dict,
    )

    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON body"}, status=400)

    s = load_visualization_style()
    _merge_style_fields(s, body)
    save_visualization_style(s)
    _clear_instruction_caches(server)

    return web.json_response(visualization_style_to_dict(s))


# ── Analyze uploaded reference ───────────────────────────────────────

_ANALYZE_PROMPT = """\
Analyze this {source_type} as a brand/design reference for data visualizations, charts, and styled reports.

Extract the following and return ONLY valid JSON (no markdown fences, no explanation):

{{
  "color_palette": ["#hex1", "#hex2", ...],
  "font_primary": "font-family for body text or empty string",
  "font_headings": "font-family for headings or empty string",
  "font_mono": "monospace font-family or empty string",
  "background_style": "dark/light/gradient/custom description",
  "chart_style": "minimal/corporate/playful/modern/custom description",
  "layout_notes": "brief layout and spacing observations",
  "additional_rules": "any other design rules, CSS conventions, or style notes"
}}

Be specific with hex color codes. For fonts, include fallback families \
(e.g. "Inter, system-ui, sans-serif"). For charts, describe gridline style, \
border radius, shadow usage, and spacing conventions you observe."""


async def analyze_visualization_style(
    server: WebServer, request: web.Request,
) -> web.Response:
    """POST /api/visualization-style/analyze — upload image/document, extract style.

    Accepts multipart form with a ``file`` field.
    Returns JSON with extracted style fields.
    """
    try:
        reader = await request.multipart()
        if reader is None:
            return web.json_response(
                {"error": "Multipart body required"}, status=400,
            )

        file_field = None
        while True:
            field = await reader.next()
            if field is None:
                break
            if field.name == "file":
                file_field = field
                break

        if file_field is None:
            return web.json_response(
                {"error": "No file field in upload"}, status=400,
            )

        original_name = file_field.filename or "upload"
        ext = Path(original_name).suffix.lower()

        if ext not in _ALL_ALLOWED:
            return web.json_response(
                {
                    "error": (
                        f"Unsupported file type '{ext}'. "
                        f"Allowed: {', '.join(sorted(_ALL_ALLOWED))}"
                    )
                },
                status=400,
            )

        # Read file data.
        chunks: list[bytes] = []
        while True:
            chunk = await file_field.read_chunk(8192)
            if not chunk:
                break
            chunks.append(chunk)
        file_bytes = b"".join(chunks)

        if not file_bytes:
            return web.json_response({"error": "Empty file"}, status=400)

        log.info(
            "Visualization style analysis requested",
            filename=original_name,
            size=len(file_bytes),
        )

        # Route to image or document analysis.
        if ext in _IMAGE_EXTENSIONS:
            result = await _analyze_image(file_bytes, ext, server)
        else:
            result = await _analyze_document(file_bytes, ext, server)

        result["source_description"] = (
            f"Extracted from uploaded file: {original_name}"
        )

        return web.json_response(result)

    except web.HTTPException:
        raise
    except Exception as exc:
        log.error("Visualization style analysis failed", error=str(exc))
        return web.json_response(
            {"error": f"Analysis failed: {exc}"}, status=500,
        )


async def _analyze_image(
    file_bytes: bytes, ext: str, server: "WebServer",
) -> dict:
    """Analyze an image using a vision-capable LLM."""
    from captain_claw.config import get_config

    cfg = get_config()

    # Find a vision model.
    model_cfg = None
    for model_type in ("vision", "ocr"):
        for m in cfg.model.allowed:
            if getattr(m, "model_type", "llm") == model_type:
                model_cfg = m
                break
        if model_cfg:
            break

    if model_cfg is None:
        # Fall back to the main LLM provider with text-only analysis.
        return await _analyze_with_provider(
            "I uploaded a design reference image but no vision model is configured. "
            "Please return a default style JSON with empty fields.",
            server,
        )

    # Optionally resize.
    try:
        from captain_claw.tools.image_ocr import _maybe_resize_image

        max_px = int(
            getattr(
                getattr(cfg.tools, "image_vision", None), "max_pixels", 1568
            )
            or 1568
        )
        jpg_q = int(
            getattr(
                getattr(cfg.tools, "image_vision", None), "jpeg_quality", 85
            )
            or 85
        )
        file_bytes, resized_mime = await asyncio.to_thread(
            _maybe_resize_image, file_bytes, max_px, jpg_q,
        )
    except Exception:
        resized_mime = ""

    b64_data = base64.b64encode(file_bytes).decode("ascii")
    mime = resized_mime or mimetypes.guess_type(f"file{ext}")[0] or "image/png"

    prompt = _ANALYZE_PROMPT.format(source_type="image")

    try:
        from litellm import completion as litellm_completion

        from captain_claw.llm import (
            _normalize_provider_name,
            _provider_model_name,
            _resolve_api_key,
        )

        provider = _normalize_provider_name(model_cfg.provider)
        model_name = _provider_model_name(provider, model_cfg.model)

        extra_headers: dict[str, str] | None = None
        try:
            extra_headers = cfg.provider_keys.headers_for(provider) or None
        except Exception:
            pass
        api_key = None if extra_headers else _resolve_api_key(provider, None)

        call_kwargs: dict = {
            "model": model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime};base64,{b64_data}",
                            },
                        },
                    ],
                }
            ],
            "timeout": 60,
            "drop_params": True,
        }
        if api_key:
            call_kwargs["api_key"] = api_key
        base_url = str(getattr(model_cfg, "base_url", "") or "").strip()
        if base_url:
            call_kwargs["api_base"] = base_url

        response = await asyncio.to_thread(litellm_completion, **call_kwargs)

        choices = getattr(response, "choices", None)
        if not choices:
            return _empty_result("Vision model returned no choices.")
        message = getattr(choices[0], "message", None)
        raw = getattr(message, "content", "") if message else ""

        return _parse_style_json(raw)

    except Exception as exc:
        log.warning("Vision analysis failed", error=str(exc))
        return _empty_result(f"Vision analysis failed: {exc}")


async def _analyze_document(
    file_bytes: bytes, ext: str, server: "WebServer",
) -> dict:
    """Analyze a document (PDF/DOCX) by extracting text and sending to LLM."""
    import tempfile

    # Write to temp file for extractor functions.
    suffix = ext
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = Path(tmp.name)

    try:
        content = ""
        if ext == ".pdf":
            from captain_claw.tools.document_extract import _extract_pdf_markdown

            text, err = _extract_pdf_markdown(tmp_path, max_pages=50)
            if err:
                return _empty_result(f"PDF extraction error: {err}")
            content = text or ""
        elif ext == ".docx":
            from captain_claw.tools.document_extract import _extract_docx_markdown

            content = _extract_docx_markdown(tmp_path) or ""
    finally:
        try:
            tmp_path.unlink()
        except OSError:
            pass

    if not content.strip():
        return _empty_result("No text content extracted from document.")

    # Truncate if very long.
    if len(content) > 200_000:
        content = content[:200_000] + "\n\n[truncated]"

    prompt = (
        _ANALYZE_PROMPT.format(source_type="document")
        + f"\n\nDOCUMENT CONTENT:\n{content}"
    )
    return await _analyze_with_provider(prompt, server)


async def _analyze_with_provider(
    prompt: str, server: "WebServer",
) -> dict:
    """Send a text prompt to the main LLM provider and parse style JSON."""
    from captain_claw.llm import Message

    provider = server.agent.provider if server.agent else None
    if provider is None:
        from captain_claw.llm import get_provider

        provider = get_provider()

    messages = [
        Message(role="user", content=prompt),
    ]

    try:
        response = await asyncio.wait_for(
            provider.complete(messages=messages, tools=None, max_tokens=2000),
            timeout=60.0,
        )
        raw = str(getattr(response, "content", "") or "").strip()
        return _parse_style_json(raw)
    except Exception as exc:
        log.warning("Style analysis via provider failed", error=str(exc))
        return _empty_result(f"Analysis failed: {exc}")


def _parse_style_json(raw: str) -> dict:
    """Extract a JSON object from LLM response text."""
    # Strip markdown code fences if present.
    cleaned = re.sub(r"^```(?:json)?\s*\n?", "", raw.strip())
    cleaned = re.sub(r"\n?```\s*$", "", cleaned)

    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            # Normalize color_palette to a list of strings.
            palette = data.get("color_palette", [])
            if isinstance(palette, list):
                data["color_palette"] = [str(c).strip() for c in palette if str(c).strip()]
            return data
    except json.JSONDecodeError:
        pass

    # Try to find JSON object in the text.
    match = re.search(r"\{[\s\S]*\}", cleaned)
    if match:
        try:
            data = json.loads(match.group())
            if isinstance(data, dict):
                palette = data.get("color_palette", [])
                if isinstance(palette, list):
                    data["color_palette"] = [str(c).strip() for c in palette if str(c).strip()]
                return data
        except json.JSONDecodeError:
            pass

    return _empty_result("Could not parse style from LLM response.")


def _empty_result(note: str = "") -> dict:
    """Return an empty style dict with an optional error note."""
    result = {
        "color_palette": [],
        "font_primary": "",
        "font_headings": "",
        "font_mono": "",
        "background_style": "",
        "chart_style": "",
        "layout_notes": "",
        "additional_rules": "",
    }
    if note:
        result["_note"] = note
    return result


# ── Rephrase ─────────────────────────────────────────────────────────


async def rephrase_visualization_style(
    server: WebServer, request: web.Request,
) -> web.Response:
    """POST /api/visualization-style/rephrase — rephrase/enrich a style field.

    Body: ``{"field": "layout_notes|additional_rules|...", "text": "..."}``
    Returns ``{"text": "..."}`` with the rephrased content.
    """
    from captain_claw.llm import Message

    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON"}, status=400)

    field_name = str(body.get("field", "")).strip()
    text = str(body.get("text", "")).strip()

    if not text:
        return web.json_response({"error": "Empty text"}, status=400)

    valid_fields = {
        "layout_notes", "additional_rules", "chart_style",
        "background_style", "font_primary", "font_headings", "font_mono",
    }
    if field_name not in valid_fields:
        return web.json_response({"error": "Invalid field"}, status=400)

    system_msg = (
        "You are a design systems expert. The user has a visualization style "
        f"field called '{field_name}'. Rewrite it to be clearer, more specific, "
        "and more actionable for an AI generating HTML charts and dashboards. "
        "Keep it concise but detailed enough to produce consistent results. "
        "Do not add markdown formatting."
    )

    provider = server.agent.provider if server.agent else None
    if provider is None:
        from captain_claw.llm import get_provider

        provider = get_provider()

    messages = [
        Message(role="system", content=system_msg),
        Message(role="user", content=text),
    ]

    try:
        response = await asyncio.wait_for(
            provider.complete(messages=messages, tools=None, max_tokens=1000),
            timeout=30.0,
        )
        result = str(getattr(response, "content", "") or "").strip()
        if not result:
            result = text
    except Exception:
        result = text

    return web.json_response({"text": result})
