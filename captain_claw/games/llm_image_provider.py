"""API-based image generation via LiteLLM.

Supports any model that LiteLLM's ``image_generation`` handles:
- Gemini Imagen  (``gemini/imagen-4.0-fast-generate-001``)
- OpenAI DALL-E  (``dall-e-3``)
- etc.

Resolves API keys from environment variables, captain_claw config,
or the Flight Deck provider-keys database.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
from pathlib import Path

from captain_claw.games.image_provider import ImageProvider
from captain_claw.logging import get_logger

_log = get_logger(__name__)

# Provider name -> env var name mapping
_ENV_KEY_MAP: dict[str, list[str]] = {
    "gemini": ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
    "openai": ["OPENAI_API_KEY"],
}

# Provider name -> key in FD provider-keys DB JSON
_FD_KEY_MAP: dict[str, str] = {
    "gemini": "gemini",
    "openai": "openai",
}


def _resolve_key_for_provider(provider: str) -> str | None:
    """Resolve API key from env vars, config, or FD database."""
    # 1. Environment variables
    for env_name in _ENV_KEY_MAP.get(provider, []):
        val = os.environ.get(env_name)
        if val:
            return val

    # 2. Captain Claw config provider_keys
    try:
        from captain_claw.config import get_config
        pk = get_config().provider_keys
        pk_map = {
            "openai": pk.openai, "anthropic": pk.anthropic,
            "gemini": pk.gemini, "xai": pk.xai, "openrouter": pk.openrouter,
        }
        val = str(pk_map.get(provider, "") or "").strip()
        if val:
            return val
    except Exception:
        pass

    # 3. Flight Deck database (provider-keys system setting)
    try:
        from captain_claw.flight_deck.auth import get_db
        db = get_db()
        if db is not None:
            # Run sync query via the underlying sqlite connection
            import asyncio as _aio
            loop = None
            try:
                loop = _aio.get_running_loop()
            except RuntimeError:
                pass

            async def _read():
                raw = await db.get_system_setting("fd:provider-keys")
                if not raw:
                    return None
                keys = json.loads(raw)
                fd_key = _FD_KEY_MAP.get(provider, provider)
                return str(keys.get(fd_key, "") or "").strip() or None

            if loop and loop.is_running():
                # We're in an async context — use a thread to avoid deadlock
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(1) as pool:
                    val = pool.submit(lambda: _aio.run(_read())).result(timeout=5)
            else:
                val = _aio.run(_read())
            if val:
                return val
    except Exception as exc:
        _log.debug("Could not read FD provider keys", error=str(exc))

    return None


class LLMImageProvider(ImageProvider):
    """Generate images via LiteLLM's image_generation API."""

    def __init__(self, model: str = "gemini/imagen-4.0-fast-generate-001"):
        self.model = model

    @property
    def label(self) -> str:
        return f"LLM ({self.model})"

    async def generate(
        self,
        prompt: str,
        output_path: Path,
        width: int = 768,
        height: int = 512,
        seed: int | None = None,
    ) -> Path:
        from litellm import image_generation as litellm_image_generation
        from captain_claw.llm import _normalize_provider_name

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Resolve provider and API key
        provider_raw = self.model.split("/")[0] if "/" in self.model else self.model
        provider = _normalize_provider_name(provider_raw)
        api_key = _resolve_key_for_provider(provider)

        if not api_key:
            raise RuntimeError(
                f"No API key found for provider '{provider}'. "
                f"Set it in Admin > Settings > Provider API Keys."
            )

        # Map dimensions to closest supported size string
        size = f"{width}x{height}"

        gen_kwargs: dict = {
            "model": self.model,
            "prompt": prompt,
            "n": 1,
            "timeout": 120,
            "size": size,
            "drop_params": True,
            "api_key": api_key,
        }

        _log.info("Generating image via LLM", model=self.model)

        # litellm's Gemini image gen validates env var before using api_key param,
        # so we must inject it into the environment
        env_vars = _ENV_KEY_MAP.get(provider, [])
        env_set: list[str] = []
        for env_name in env_vars:
            if not os.environ.get(env_name):
                os.environ[env_name] = api_key
                env_set.append(env_name)

        try:
            response = await asyncio.to_thread(litellm_image_generation, **gen_kwargs)
        finally:
            for env_name in env_set:
                os.environ.pop(env_name, None)

        # Extract image data
        data_list = getattr(response, "data", None)
        if not data_list and isinstance(response, dict):
            data_list = response.get("data", [])
        if not data_list:
            raise RuntimeError("Image generation returned no data")

        first_item = data_list[0]
        b64_data = getattr(first_item, "b64_json", None) or (
            first_item.get("b64_json") if isinstance(first_item, dict) else None
        )
        image_url = getattr(first_item, "url", None) or (
            first_item.get("url") if isinstance(first_item, dict) else None
        )

        if b64_data:
            image_bytes = base64.b64decode(b64_data)
        elif image_url:
            import httpx
            async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
                resp = await client.get(str(image_url))
                resp.raise_for_status()
                image_bytes = resp.content
        else:
            raise RuntimeError("Image generation returned neither base64 data nor URL")

        await asyncio.to_thread(output_path.write_bytes, image_bytes)
        _log.info("Image generated via LLM", model=self.model, path=str(output_path))
        return output_path
