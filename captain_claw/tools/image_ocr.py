"""Image OCR tool — extract text from images via a vision-capable LLM."""

import asyncio
import base64
import mimetypes
from typing import Any

from captain_claw.config import get_config
from captain_claw.logging import get_logger
from captain_claw.tools.document_extract import _require_existing_file
from captain_claw.tools.registry import Tool, ToolResult

log = get_logger(__name__)

_IMAGE_EXTENSIONS: set[str] = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}


class ImageOcrTool(Tool):
    """Extract text from images using a vision-capable LLM."""

    name = "image_ocr"
    timeout_seconds = 120.0
    description = (
        "Extract text from an image file using OCR via a vision-capable model. "
        "Provide the path to a local image file (.png, .jpg, .jpeg, .webp, .gif, .bmp)."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the image file to extract text from.",
            },
            "prompt": {
                "type": "string",
                "description": (
                    "Optional instruction for the vision model. "
                    "Default: 'Extract all text from this image.'"
                ),
            },
            "max_chars": {
                "type": "number",
                "description": "Maximum characters to return (default: 120000).",
            },
        },
        "required": ["path"],
    }

    def __init__(self):
        cfg = get_config()
        tool_cfg = getattr(cfg.tools, "image_ocr", None)
        if tool_cfg is not None:
            self.timeout_seconds = float(
                getattr(tool_cfg, "timeout_seconds", 120) or 120
            )

    @staticmethod
    def _find_ocr_model():
        """Find the first allowed model with model_type 'ocr' or 'vision'."""
        cfg = get_config()
        # Prefer explicit 'ocr' type first.
        for m in cfg.model.allowed:
            if getattr(m, "model_type", "llm") == "ocr":
                return m
        # Fall back to 'vision' type.
        for m in cfg.model.allowed:
            if getattr(m, "model_type", "llm") == "vision":
                return m
        return None

    async def execute(
        self,
        path: str,
        prompt: str = "",
        max_chars: int | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        """Read an image and extract text via a vision LLM."""
        path_str = str(path or "").strip()
        if not path_str:
            return ToolResult(success=False, error="Missing required argument: path")

        runtime_base = kwargs.get("_runtime_base_path")
        file_path, error = _require_existing_file(path_str, runtime_base_path=runtime_base)
        if error:
            return ToolResult(success=False, error=error)

        if file_path.suffix.lower() not in _IMAGE_EXTENSIONS:
            return ToolResult(
                success=False,
                error=f"Unsupported image format '{file_path.suffix}'. Supported: {', '.join(sorted(_IMAGE_EXTENSIONS))}",
            )

        model_cfg = self._find_ocr_model()
        if model_cfg is None:
            return ToolResult(
                success=False,
                error=(
                    "No OCR/vision model configured. "
                    "Add a model with model_type: 'ocr' or 'vision' to "
                    "model.allowed in settings."
                ),
            )

        cfg = get_config()
        tool_cfg = getattr(cfg.tools, "image_ocr", None)

        # Resolve prompt.
        default_prompt = (
            getattr(tool_cfg, "default_prompt", "") if tool_cfg else ""
        )
        effective_prompt = (
            str(prompt or "").strip()
            or default_prompt
            or "Extract all text from this image."
        )

        # Resolve max_chars.
        cfg_max = int(getattr(tool_cfg, "max_chars", 120000) if tool_cfg else 120000)
        effective_max = int(max_chars or cfg_max)

        log.info(
            "Image OCR requested",
            provider=model_cfg.provider,
            model=model_cfg.model,
            path=str(file_path),
            prompt=effective_prompt[:80],
        )

        try:
            from litellm import completion as litellm_completion

            from captain_claw.llm import (
                _normalize_provider_name,
                _provider_model_name,
                _resolve_api_key,
            )

            provider = _normalize_provider_name(model_cfg.provider)
            model_name = _provider_model_name(provider, model_cfg.model)
            api_key = _resolve_api_key(provider, None)

            # Read and encode image.
            image_bytes = await asyncio.to_thread(file_path.read_bytes)
            b64_data = base64.b64encode(image_bytes).decode("ascii")
            mime = mimetypes.guess_type(str(file_path))[0] or "image/png"

            call_kwargs: dict[str, Any] = {
                "model": model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": effective_prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime};base64,{b64_data}",
                                },
                            },
                        ],
                    }
                ],
                "timeout": int(self.timeout_seconds),
                "drop_params": True,
            }
            if api_key:
                call_kwargs["api_key"] = api_key
            base_url = str(getattr(model_cfg, "base_url", "") or "").strip()
            if base_url:
                call_kwargs["api_base"] = base_url

            response = await asyncio.to_thread(litellm_completion, **call_kwargs)

            # Extract text from response.
            choices = getattr(response, "choices", None)
            if not choices:
                return ToolResult(success=False, error="Vision model returned no choices.")
            message = getattr(choices[0], "message", None)
            raw_content = getattr(message, "content", "") if message else ""
            if isinstance(raw_content, list):
                text_parts = []
                for part in raw_content:
                    t = getattr(part, "text", "") if not isinstance(part, dict) else part.get("text", "")
                    if t:
                        text_parts.append(t)
                extracted = "\n".join(text_parts)
            else:
                extracted = str(raw_content or "")

            if not extracted.strip():
                return ToolResult(
                    success=True,
                    content="No text was detected in the image.",
                )

            # Truncate if needed.
            if len(extracted) > effective_max:
                extracted = extracted[:effective_max] + "\n\n... [truncated]"

            size_kb = len(image_bytes) / 1024.0
            log.info(
                "Image OCR completed",
                path=str(file_path),
                model=model_cfg.model,
                chars=len(extracted),
            )
            return ToolResult(
                success=True,
                content=(
                    f"OCR results from {file_path.name} "
                    f"({size_kb:.1f} KB, model: {model_cfg.model}):\n\n"
                    f"{extracted}"
                ),
            )

        except Exception as e:
            log.error("Image OCR failed", error=str(e))
            return ToolResult(success=False, error=str(e))
