"""Local image generation on Apple Silicon via mflux.

Requires: pip install mflux

mflux runs FLUX models natively on MLX — fast on M-series Macs.
Models are downloaded automatically on first use.

Supported models (mflux 0.17.5):
- schnell        (FLUX.1-schnell, ~12 GB, gated)
- fibo-lite      (Fibo-lite, ~5 GB, ungated)
- flux2-klein-4b (FLUX.2-klein-4B, ~8 GB, gated)
"""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path

from captain_claw.games.image_provider import ImageProvider
from captain_claw.logging import get_logger

_log = get_logger(__name__)

# Cache: model_name -> Flux1 instance
_flux_instances: dict[str, object] = {}
_flux_lock = asyncio.Lock()
_init_lock = threading.Lock()

# Model-specific inference step counts
_STEPS: dict[str, int] = {
    "schnell": 4,
    "fibo-lite": 4,
    "flux2-klein-4b": 4,
}


def _load_mflux_classes():
    """Import Flux1 and ModelConfig from whichever mflux version is installed."""
    # mflux 0.17.x
    try:
        from mflux.models.flux.cli.flux_generate import Flux1, ModelConfig
        return Flux1, ModelConfig, "from_name"
    except ImportError:
        pass
    # mflux 0.2.x
    try:
        from mflux.flux.flux import Flux1
        from mflux.config.model_config import ModelConfig
        return Flux1, ModelConfig, "from_alias"
    except ImportError:
        pass
    # Top-level import (future versions)
    try:
        from mflux import Flux1, ModelConfig
        return Flux1, ModelConfig, "from_alias"
    except ImportError:
        pass
    raise RuntimeError("mflux is not installed. Install with: pip install mflux")


def _get_flux(model_name: str):
    """Return a cached Flux1 instance for the given model (loads on first call)."""
    if model_name in _flux_instances:
        return _flux_instances[model_name]

    with _init_lock:
        if model_name in _flux_instances:
            return _flux_instances[model_name]

        Flux1, ModelConfig, resolver = _load_mflux_classes()

        _log.info("Loading mflux model (first time may download)...", model=model_name)
        if resolver == "from_name":
            model_config = ModelConfig.from_name(model_name)
        else:
            model_config = ModelConfig.from_alias(model_name)

        _flux_instances[model_name] = Flux1(
            model_config=model_config,
            quantize=8,
        )
        _log.info("mflux model loaded", model=model_name)
        return _flux_instances[model_name]


class MfluxImageProvider(ImageProvider):
    """Generate images locally using mflux/MLX on Apple Silicon."""

    def __init__(self, model_name: str = "schnell"):
        self.model_name = model_name

    @property
    def label(self) -> str:
        return f"mflux ({self.model_name})"

    async def generate(
        self,
        prompt: str,
        output_path: Path,
        width: int = 768,
        height: int = 512,
        seed: int | None = None,
    ) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        steps = _STEPS.get(self.model_name, 4)
        model_name = self.model_name

        def _run():
            flux = _get_flux(model_name)
            image = flux.generate_image(
                seed=seed or 42,
                prompt=prompt,
                num_inference_steps=steps,
                width=width,
                height=height,
            )
            image.save(path=str(output_path), export_json_metadata=False, overwrite=True)

        async with _flux_lock:
            await asyncio.to_thread(_run)

        _log.info("Image generated", model=model_name, path=str(output_path))
        return output_path
