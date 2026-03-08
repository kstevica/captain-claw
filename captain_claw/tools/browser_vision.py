"""Vision LLM integration for the browser tool.

Sends browser screenshots to a vision-capable LLM for page understanding.
Follows the exact same LiteLLM call pattern as ``image_ocr.py`` —
resize with Pillow, base64 encode, send as ``image_url`` content part.
"""

from __future__ import annotations

import asyncio
import base64
import io
from typing import Any

from captain_claw.config import get_config
from captain_claw.logging import get_logger

log = get_logger(__name__)

# ---------- optional Pillow guard --------------------------------------------

try:
    from PIL import Image as _PILImage  # type: ignore[import-untyped]

    _HAS_PILLOW = True
except ImportError:
    _HAS_PILLOW = False


# ---------- image helpers (mirrors image_ocr.py) -----------------------------


def _resize_screenshot(
    image_bytes: bytes,
    max_pixels: int = 1568,
    jpeg_quality: int = 85,
) -> tuple[bytes, str]:
    """Resize screenshot if needed.  Returns (bytes, mime_type).

    If Pillow is not available, returns the original bytes with an
    empty mime string (the caller should fall back to "image/jpeg").
    """
    if not _HAS_PILLOW or max_pixels <= 0:
        return image_bytes, ""

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

        # Convert to RGB if needed (e.g. RGBA screenshots)
        if img.mode in ("RGBA", "P", "LA"):
            img = img.convert("RGB")

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=jpeg_quality)
        return buf.getvalue(), "image/jpeg"

    except Exception as e:
        log.warning("Screenshot resize failed, sending raw", error=str(e))
        return image_bytes, ""


# ---------- BrowserVision ----------------------------------------------------


class BrowserVision:
    """Analyze browser screenshots using a vision-capable LLM.

    Follows the same pattern as ``_BaseImageLLMTool`` in ``image_ocr.py``
    for model selection and LiteLLM invocation, but operates on in-memory
    screenshot bytes rather than file paths.
    """

    @staticmethod
    def find_vision_model() -> Any | None:
        """Find a configured vision model from ``config.model.allowed``.

        Searches model_type in order: ``vision``, ``ocr``, ``multimodal``.
        Returns the ``AllowedModelConfig`` or ``None``.
        """
        cfg = get_config()
        for model_type in ("vision", "ocr", "multimodal"):
            for m in cfg.model.allowed:
                if getattr(m, "model_type", "llm") == model_type:
                    return m
        return None

    @staticmethod
    async def analyze_screenshot(
        screenshot_bytes: bytes,
        prompt: str = "",
        max_pixels: int = 0,
        jpeg_quality: int = 0,
    ) -> str:
        """Send a screenshot to the vision LLM and return the analysis.

        Args:
            screenshot_bytes: Raw JPEG/PNG bytes from Playwright.
            prompt: Instruction for the vision model.
            max_pixels: Override max pixel dimension (0 = use config).
            jpeg_quality: Override JPEG quality (0 = use config).

        Returns:
            The model's text response describing the page, or an empty
            string if no vision model is configured.
        """
        log.info("BrowserVision.analyze_screenshot called", screenshot_bytes=len(screenshot_bytes), has_prompt=bool(prompt))
        model_cfg = BrowserVision.find_vision_model()
        if model_cfg is None:
            log.info("No vision model configured, skipping visual analysis")
            return ""  # silently skip — accessibility tree is the fallback

        cfg = get_config()
        browser_cfg = cfg.tools.browser

        effective_max_px = max_pixels or browser_cfg.screenshot_max_pixels
        effective_quality = jpeg_quality or browser_cfg.screenshot_jpeg_quality

        effective_prompt = prompt.strip() if prompt else (
            "Describe this web page screenshot. Identify:\n"
            "1. The page layout and main sections\n"
            "2. Navigation elements and menus\n"
            "3. Forms, input fields, and buttons\n"
            "4. Data displayed (tables, lists, cards)\n"
            "5. Any notifications, modals, or overlays\n"
            "Be specific about what each element does and where it is located."
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

            # Resolve auth: prefer extra headers (bearer/OAuth) over API key.
            from captain_claw.config import get_config as _get_config
            _cfg = _get_config()
            extra_headers: dict[str, str] | None = None
            try:
                extra_headers = _cfg.provider_keys.headers_for(provider) or None
            except Exception:
                pass
            api_key = None if extra_headers else _resolve_api_key(provider, None)

            # Resize if needed
            image_bytes, resized_mime = await asyncio.to_thread(
                _resize_screenshot, screenshot_bytes, effective_max_px, effective_quality,
            )

            # Base64 encode
            b64_data = base64.b64encode(image_bytes).decode("ascii")
            mime = resized_mime or "image/jpeg"

            log.info(
                "Vision analysis requested",
                model=model_cfg.model,
                image_bytes=len(image_bytes),
                prompt=effective_prompt[:80],
            )

            # Route: OpenAI + bearer headers → ChatGPT Responses API directly,
            # everything else → litellm Chat Completions.
            if provider == "openai" and extra_headers:
                from captain_claw.tools.image_ocr import _chatgpt_responses_vision

                resp_base_url = (
                    str(getattr(model_cfg, "base_url", "") or "").strip()
                    or "https://chatgpt.com/backend-api/codex/responses"
                )
                extracted = await _chatgpt_responses_vision(
                    headers=extra_headers,
                    model=model_cfg.model,
                    prompt=effective_prompt,
                    b64_image=b64_data,
                    mime=mime,
                    base_url=resp_base_url,
                    timeout=120.0,
                )
            else:
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
                    "timeout": 120,
                    "drop_params": True,
                }

                if api_key:
                    call_kwargs["api_key"] = api_key
                base_url = str(getattr(model_cfg, "base_url", "") or "").strip()
                if base_url:
                    call_kwargs["api_base"] = base_url

                response = await asyncio.to_thread(litellm_completion, **call_kwargs)

                # Extract text from response
                choices = getattr(response, "choices", None)
                if not choices:
                    log.warning("Vision model returned no choices")
                    return ""

                message = getattr(choices[0], "message", None)
                raw_content = getattr(message, "content", "") if message else ""

                if isinstance(raw_content, list):
                    text_parts = []
                    for part in raw_content:
                        t = (
                            getattr(part, "text", "")
                            if not isinstance(part, dict)
                            else part.get("text", "")
                        )
                        if t:
                            text_parts.append(t)
                    extracted = "\n".join(text_parts)
                else:
                    extracted = str(raw_content or "")

            log.info("Vision analysis completed", chars=len(extracted))
            return extracted.strip()

        except Exception as e:
            log.error("Vision analysis failed", error=str(e))
            return f"(vision analysis failed: {e})"
