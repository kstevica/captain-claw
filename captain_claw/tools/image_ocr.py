"""Image analysis tools — OCR and vision via multimodal LLMs."""

import asyncio
import base64
import io
import mimetypes
import shutil
import subprocess
from typing import Any

from captain_claw.config import get_config
from captain_claw.logging import get_logger
from captain_claw.tools.document_extract import _require_existing_file
from captain_claw.tools.registry import Tool, ToolResult

log = get_logger(__name__)

_IMAGE_EXTENSIONS: set[str] = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}

try:
    from PIL import Image as _PILImage  # type: ignore[import-untyped]
    _HAS_PILLOW = True
except ImportError:
    _HAS_PILLOW = False


def _resize_with_pillow(
    image_bytes: bytes,
    max_pixels: int,
    jpeg_quality: int,
) -> tuple[bytes, str] | None:
    """Try resizing with Pillow.  Returns None on failure."""
    if not _HAS_PILLOW:
        return None
    try:
        img = _PILImage.open(io.BytesIO(image_bytes))
        w, h = img.size

        if max(w, h) <= max_pixels:
            return image_bytes, ""  # no resize needed

        if w >= h:
            new_w = max_pixels
            new_h = int(h * (max_pixels / w))
        else:
            new_h = max_pixels
            new_w = int(w * (max_pixels / h))

        img = img.resize((new_w, new_h), _PILImage.LANCZOS)
        if img.mode in ("RGBA", "P", "LA"):
            img = img.convert("RGB")

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=jpeg_quality)
        resized = buf.getvalue()

        if len(resized) == 0:
            log.warning("Pillow produced 0-byte output, falling back")
            return None

        log.info(
            "image resized (pillow)",
            original=f"{w}x{h}", resized=f"{new_w}x{new_h}",
            orig_kb=f"{len(image_bytes)/1024:.0f}",
            new_kb=f"{len(resized)/1024:.0f}",
        )
        return resized, "image/jpeg"
    except Exception as exc:
        log.warning("Pillow resize failed, falling back", error=str(exc))
        return None


def _resize_with_imagemagick(
    image_bytes: bytes,
    max_pixels: int,
    jpeg_quality: int,
) -> tuple[bytes, str] | None:
    """Try resizing with ImageMagick ``convert`` via subprocess (stdin/stdout)."""
    convert_bin = shutil.which("convert") or shutil.which("magick")
    if not convert_bin:
        return None
    try:
        args = [convert_bin]
        if "magick" in convert_bin:
            args.append("convert")
        args += [
            "-",                                    # read from stdin
            "-resize", f"{max_pixels}x{max_pixels}>",  # shrink only
            "-quality", str(jpeg_quality),
            "jpeg:-",                               # write JPEG to stdout
        ]
        proc = subprocess.run(
            args, input=image_bytes,
            capture_output=True, timeout=15,
        )
        if proc.returncode == 0 and len(proc.stdout) > 0:
            log.info(
                "image resized (imagemagick)",
                orig_kb=f"{len(image_bytes)/1024:.0f}",
                new_kb=f"{len(proc.stdout)/1024:.0f}",
            )
            return proc.stdout, "image/jpeg"
    except Exception as exc:
        log.warning("ImageMagick resize failed", error=str(exc))
    return None


def _maybe_resize_image(
    image_bytes: bytes,
    max_pixels: int,
    jpeg_quality: int,
) -> tuple[bytes, str]:
    """Resize an image if its longest edge exceeds *max_pixels*.

    Tries Pillow first, then ImageMagick subprocess, then returns
    original bytes untouched.
    """
    if max_pixels <= 0:
        return image_bytes, ""

    # Strategy 1: Pillow
    result = _resize_with_pillow(image_bytes, max_pixels, jpeg_quality)
    if result is not None:
        return result

    # Strategy 2: ImageMagick CLI (works on Termux with pkg install imagemagick)
    result = _resize_with_imagemagick(image_bytes, max_pixels, jpeg_quality)
    if result is not None:
        return result

    # No resize available — send raw.
    log.info("no image resize backend available, sending raw")
    return image_bytes, ""


class _BaseImageLLMTool(Tool):
    """Shared base for tools that send an image to a vision-capable LLM.

    Subclasses set ``name``, ``description``, ``parameters``, and override
    ``_config_key``, ``_model_types``, and ``_default_prompt``.
    """

    # Subclasses must set these.
    _config_key: str = ""          # e.g. "image_ocr"
    _model_types: list[str] = []   # e.g. ["ocr", "vision"] — tried in order
    _default_prompt: str = ""
    _no_model_error: str = ""

    def __init__(self):
        cfg = get_config()
        tool_cfg = getattr(cfg.tools, self._config_key, None)
        if tool_cfg is not None:
            self.timeout_seconds = float(
                getattr(tool_cfg, "timeout_seconds", 120) or 120
            )

    def _find_model(self):
        """Find the first allowed model matching any of ``_model_types``."""
        cfg = get_config()
        for model_type in self._model_types:
            for m in cfg.model.allowed:
                if getattr(m, "model_type", "llm") == model_type:
                    return m
        return None

    async def execute(
        self,
        path: str,
        prompt: str = "",
        max_chars: int | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        """Send an image to a vision LLM and return the response."""
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

        model_cfg = self._find_model()
        if model_cfg is None:
            return ToolResult(success=False, error=self._no_model_error)

        cfg = get_config()
        tool_cfg = getattr(cfg.tools, self._config_key, None)

        # Resolve prompt.
        default_prompt = (
            getattr(tool_cfg, "default_prompt", "") if tool_cfg else ""
        )
        effective_prompt = (
            str(prompt or "").strip()
            or default_prompt
            or self._default_prompt
        )

        # Resolve max_chars.
        cfg_max = int(getattr(tool_cfg, "max_chars", 120000) if tool_cfg else 120000)
        effective_max = int(max_chars or cfg_max)

        log.info(
            "%s requested", self.name,
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

            # Read, optionally resize, and encode image.
            image_bytes = await asyncio.to_thread(file_path.read_bytes)
            original_size = len(image_bytes)

            max_px = int(getattr(tool_cfg, "max_pixels", 1568) if tool_cfg else 1568)
            jpg_q = int(getattr(tool_cfg, "jpeg_quality", 85) if tool_cfg else 85)
            image_bytes, resized_mime = await asyncio.to_thread(
                _maybe_resize_image, image_bytes, max_px, jpg_q,
            )

            b64_data = base64.b64encode(image_bytes).decode("ascii")
            mime = resized_mime or mimetypes.guess_type(str(file_path))[0] or "image/png"

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
                    content="The model returned no content for this image.",
                )

            # Truncate if needed.
            if len(extracted) > effective_max:
                extracted = extracted[:effective_max] + "\n\n... [truncated]"

            size_kb = len(image_bytes) / 1024.0
            log.info(
                "%s completed", self.name,
                path=str(file_path),
                model=model_cfg.model,
                chars=len(extracted),
            )
            return ToolResult(
                success=True,
                content=(
                    f"{self.name} results from {file_path.name} "
                    f"({size_kb:.1f} KB, model: {model_cfg.model}):\n\n"
                    f"{extracted}"
                ),
            )

        except Exception as e:
            log.error("%s failed", self.name, error=str(e))
            return ToolResult(success=False, error=str(e))


# ── Concrete tools ────────────────────────────────────────────


class ImageOcrTool(_BaseImageLLMTool):
    """Extract text from images using OCR via a vision-capable LLM."""

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

    _config_key = "image_ocr"
    _model_types = ["ocr", "vision"]
    _default_prompt = "Extract all text from this image."
    _no_model_error = (
        "No OCR/vision model configured. "
        "Add a model with model_type: 'ocr' or 'vision' to "
        "model.allowed in settings."
    )


class ImageVisionTool(_BaseImageLLMTool):
    """Analyze and describe images using a vision-capable LLM."""

    name = "image_vision"
    timeout_seconds = 120.0
    description = (
        "Analyze an image using a vision model to describe scenes, identify objects, "
        "read signs, understand charts, or answer questions about visual content. "
        "Provide the path to a local image file (.png, .jpg, .jpeg, .webp, .gif, .bmp)."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the image file to analyze.",
            },
            "prompt": {
                "type": "string",
                "description": (
                    "Question or instruction about the image. "
                    "Default: 'Describe this image in detail.'"
                ),
            },
            "max_chars": {
                "type": "number",
                "description": "Maximum characters to return (default: 120000).",
            },
        },
        "required": ["path"],
    }

    _config_key = "image_vision"
    _model_types = ["vision", "ocr"]
    _default_prompt = "Describe this image in detail."
    _no_model_error = (
        "No vision model configured. "
        "Add a model with model_type: 'vision' to "
        "model.allowed in settings."
    )
