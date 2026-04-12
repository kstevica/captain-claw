"""Abstract image provider interface for room image generation.

Concrete implementations:
- MfluxImageProvider  — local generation on Apple Silicon via mflux (FLUX.1-schnell)
- (future) APIImageProvider — remote generation via DALL-E / Stability / etc.
"""

from __future__ import annotations

import abc
from pathlib import Path


class ImageProvider(abc.ABC):
    """Generate a scene image from a text prompt."""

    @abc.abstractmethod
    async def generate(
        self,
        prompt: str,
        output_path: Path,
        width: int = 768,
        height: int = 512,
        seed: int | None = None,
    ) -> Path:
        """Generate an image and save it to *output_path*. Returns the path."""

    @property
    @abc.abstractmethod
    def label(self) -> str:
        """Human-readable name for this provider."""
