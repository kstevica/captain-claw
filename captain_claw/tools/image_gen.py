"""Image generation tool via LiteLLM."""

import asyncio
import base64
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from captain_claw.config import get_config
from captain_claw.logging import get_logger
from captain_claw.tools.registry import Tool, ToolResult
from captain_claw.tools.write import WriteTool

log = get_logger(__name__)


class ImageGenTool(Tool):
    """Generate images from text prompts using configured image models."""

    name = "image_gen"
    timeout_seconds = 120.0
    description = (
        "Generate an image from a text prompt using an AI image model "
        "(e.g. DALL-E 3, gpt-image-1). Saves the resulting image under "
        "saved/media/ and returns the file path."
    )
    parameters = {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": "Text description of the image to generate.",
            },
            "size": {
                "type": "string",
                "description": (
                    "Image dimensions. Options: '1024x1024' (default), "
                    "'1536x1024' (landscape), '1024x1536' (portrait)."
                ),
            },
            "quality": {
                "type": "string",
                "description": (
                    "Image quality. Options: 'auto' (default), 'high', "
                    "'medium', 'low', 'hd', 'standard'."
                ),
            },
            "output_path": {
                "type": "string",
                "description": (
                    "Optional output path. Normalized under "
                    "saved/media/<session-id> and saved as .png."
                ),
            },
        },
        "required": ["prompt"],
    }

    def __init__(self):
        cfg = get_config()
        tool_cfg = getattr(cfg.tools, "image_gen", None)
        if tool_cfg is not None:
            self.timeout_seconds = float(
                getattr(tool_cfg, "timeout_seconds", 120) or 120
            )

    @staticmethod
    def _find_image_model():
        """Find the first allowed model with model_type == 'image'."""
        cfg = get_config()
        for m in cfg.model.allowed:
            if getattr(m, "model_type", "llm") == "image":
                return m
        return None

    def _resolve_output_path(
        self, output_path: str | None, **kwargs: Any
    ) -> tuple[Path, str]:
        """Resolve safe output location under saved root."""
        saved_root = WriteTool._resolve_saved_root(kwargs)
        session_id = WriteTool._normalize_session_id(
            str(kwargs.get("_session_id", ""))
        )

        requested = str(output_path or "").strip()
        if not requested:
            stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
            requested = f"media/{session_id}/image-gen-{stamp}.png"

        resolved = WriteTool._normalize_under_saved(
            requested, saved_root, session_id
        )
        if resolved.suffix.lower() not in (".png", ".jpg", ".jpeg", ".webp"):
            resolved = resolved.with_suffix(".png")
        return resolved, requested

    async def execute(
        self,
        prompt: str,
        size: str = "",
        quality: str = "",
        output_path: str = "",
        **kwargs: Any,
    ) -> ToolResult:
        """Generate an image and persist it locally."""
        prompt_text = str(prompt or "").strip()
        if not prompt_text:
            return ToolResult(success=False, error="Missing required argument: prompt")

        model_cfg = self._find_image_model()
        if model_cfg is None:
            return ToolResult(
                success=False,
                error=(
                    "No image generation model configured. "
                    "Add a model with model_type: 'image' to "
                    "model.allowed in settings."
                ),
            )

        output_file, requested_path = self._resolve_output_path(
            output_path, **kwargs
        )
        output_file.parent.mkdir(parents=True, exist_ok=True)

        log.info(
            "Image generation requested",
            provider=model_cfg.provider,
            model=model_cfg.model,
            prompt=prompt_text[:100],
        )

        try:
            from litellm import image_generation as litellm_image_generation

            from captain_claw.llm import (
                _normalize_provider_name,
                _provider_model_name,
                _resolve_api_key,
            )

            provider = _normalize_provider_name(model_cfg.provider)
            model_name = _provider_model_name(provider, model_cfg.model)
            api_key = _resolve_api_key(provider, None)

            cfg = get_config()
            tool_cfg = getattr(cfg.tools, "image_gen", None)

            gen_kwargs: dict[str, Any] = {
                "model": model_name,
                "prompt": prompt_text,
                "n": 1,
                "timeout": int(self.timeout_seconds),
                "drop_params": True,
            }
            # Request base64 when possible (avoids URL expiration).
            # Providers that don't support it will have it dropped by LiteLLM.
            if provider in ("openai",):
                gen_kwargs["response_format"] = "b64_json"
            if api_key:
                gen_kwargs["api_key"] = api_key
            base_url = str(getattr(model_cfg, "base_url", "") or "").strip()
            if base_url:
                gen_kwargs["api_base"] = base_url

            default_size = (
                getattr(tool_cfg, "default_size", "1024x1024")
                if tool_cfg
                else "1024x1024"
            )
            effective_size = str(size or "").strip() or default_size
            gen_kwargs["size"] = effective_size

            default_quality = (
                getattr(tool_cfg, "default_quality", "") if tool_cfg else ""
            )
            effective_quality = str(quality or "").strip() or default_quality
            if effective_quality:
                gen_kwargs["quality"] = effective_quality

            response = await asyncio.to_thread(
                litellm_image_generation, **gen_kwargs
            )

            # Extract image data from response
            data_list = getattr(response, "data", None)
            if not data_list and isinstance(response, dict):
                data_list = response.get("data", [])
            if not data_list:
                return ToolResult(
                    success=False,
                    error="Image generation returned no data.",
                )

            first_item = data_list[0]
            b64_data = getattr(first_item, "b64_json", None) or (
                first_item.get("b64_json") if isinstance(first_item, dict) else None
            )
            image_url = getattr(first_item, "url", None) or (
                first_item.get("url") if isinstance(first_item, dict) else None
            )
            revised_prompt = getattr(first_item, "revised_prompt", None) or (
                first_item.get("revised_prompt")
                if isinstance(first_item, dict)
                else None
            )

            if b64_data:
                image_bytes = base64.b64decode(b64_data)
            elif image_url:
                import httpx

                async with httpx.AsyncClient(
                    timeout=30.0, follow_redirects=True
                ) as client:
                    resp = await client.get(str(image_url))
                    resp.raise_for_status()
                    image_bytes = resp.content
            else:
                return ToolResult(
                    success=False,
                    error="Image generation returned neither base64 data nor URL.",
                )

            await asyncio.to_thread(output_file.write_bytes, image_bytes)

            # Register in file registry so the web UI can serve it.
            file_registry = kwargs.get("_file_registry")
            if file_registry is not None:
                try:
                    file_registry.register(
                        logical_path=requested_path,
                        physical_path=str(output_file),
                        task_id=str(kwargs.get("_task_id", "")),
                    )
                except Exception:
                    pass

            redirect_note = ""
            if requested_path != str(output_file):
                redirect_note = f" (requested: {requested_path})"

            size_kb = len(image_bytes) / 1024.0
            content_lines = [
                "Generated image successfully.",
                f"Path: {output_file}{redirect_note}",
                f"Model: {model_cfg.model}",
                f"Provider: {model_cfg.provider}",
                f"Size: {effective_size}",
                f"File size: {size_kb:.1f} KB",
            ]
            if effective_quality:
                content_lines.append(f"Quality: {effective_quality}")
            if revised_prompt:
                content_lines.append(f"Revised prompt: {revised_prompt}")

            log.info(
                "Image generated successfully",
                path=str(output_file),
                model=model_cfg.model,
                size_kb=f"{size_kb:.1f}",
            )
            return ToolResult(success=True, content="\n".join(content_lines))

        except Exception as e:
            log.error("Image generation failed", error=str(e))
            return ToolResult(success=False, error=str(e))
