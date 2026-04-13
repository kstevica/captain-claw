"""Game image generation service.

Handles prompt construction, caching, provider selection, and batch
generation for rooms, characters, entities, and world banners.
Images are stored in ``games/{game_id}/images/<type>_<id>.png``.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from captain_claw.games.image_provider import ImageProvider
from captain_claw.games.world import World
from captain_claw.logging import get_logger

_log = get_logger(__name__)

# ── Available image provider presets ──────────────────────────────

IMAGE_PROVIDERS: list[dict[str, str]] = [
    {"id": "none",           "label": "None (no image generation)",       "kind": "none"},
    {"id": "gemini-imagen",  "label": "Gemini Imagen 4 Fast (API)",       "kind": "llm"},
    {"id": "fibo-lite",      "label": "Fibo-lite (local, ~5 GB)",        "kind": "mflux"},
    {"id": "schnell",        "label": "FLUX.1-schnell (local, ~12 GB)",  "kind": "mflux"},
    {"id": "flux2-klein-4b", "label": "FLUX.2-klein-4B (local, ~8 GB)", "kind": "mflux"},
]

# ── Active provider state ─────────────────────────────────────────

_provider: ImageProvider | None = None
_provider_id: str = "none"
_mflux_available: bool = False


def _check_mflux() -> bool:
    global _mflux_available
    try:
        import mflux  # noqa: F401
        _mflux_available = True
    except ImportError:
        _mflux_available = False
    return _mflux_available


def create_provider(provider_id: str) -> ImageProvider | None:
    preset = next((p for p in IMAGE_PROVIDERS if p["id"] == provider_id), None)
    if preset is None:
        _log.warning("Unknown image provider", provider_id=provider_id)
        return None
    if preset["kind"] == "mflux":
        if not _mflux_available and not _check_mflux():
            return None
        from captain_claw.games.mflux_provider import MfluxImageProvider
        return MfluxImageProvider(model_name=provider_id)
    if preset["kind"] == "llm":
        from captain_claw.games.llm_image_provider import LLMImageProvider
        return LLMImageProvider(model="gemini/imagen-4.0-fast-generate-001")
    return None


def set_image_provider(provider: ImageProvider | None, provider_id: str = "") -> None:
    global _provider, _provider_id
    _provider = provider
    if provider_id:
        _provider_id = provider_id


def get_image_provider() -> ImageProvider | None:
    return _provider


def get_provider_id() -> str:
    return _provider_id


def switch_provider(provider_id: str) -> ImageProvider | None:
    if provider_id == "none":
        set_image_provider(None, "none")
        _log.info("Image provider switched", provider_id="none", label="None")
        return None
    provider = create_provider(provider_id)
    if provider is not None:
        set_image_provider(provider, provider_id)
        _log.info("Image provider switched", provider_id=provider_id, label=provider.label)
    return provider


def list_providers() -> list[dict[str, Any]]:
    _check_mflux()
    result = []
    for p in IMAGE_PROVIDERS:
        available = True
        if p["kind"] == "mflux" and not _mflux_available:
            available = False
        result.append({**p, "available": available, "active": p["id"] == _provider_id})
    return result


# ── Prompt builders ──────────────────────────────────────────────

_STYLE = "atmospheric digital painting, rich colors, moody lighting, game concept art"


def _room_prompt(room_name: str, room_desc: str, world_title: str, world_summary: str) -> str:
    return (
        f"A detailed scene illustration for a text adventure game. "
        f"Game setting: {world_title} — {world_summary}. "
        f"Location: {room_name}. {room_desc} "
        f"Style: {_STYLE}, no text, no UI elements, no characters."
    )


def _character_prompt(char_name: str, char_desc: str, world_title: str, world_summary: str) -> str:
    return (
        f"Character portrait for a text adventure game. "
        f"Game setting: {world_title} — {world_summary}. "
        f"Character: {char_name}. {char_desc} "
        f"Style: {_STYLE}, portrait composition, shoulders-up view, "
        f"no text, no UI elements."
    )


def _entity_prompt(entity_name: str, entity_desc: str, world_title: str, world_summary: str) -> str:
    return (
        f"Item illustration for a text adventure game. "
        f"Game setting: {world_title} — {world_summary}. "
        f"Item: {entity_name}. {entity_desc} "
        f"Style: {_STYLE}, centered object on subtle background, "
        f"no text, no UI elements, no characters."
    )


def _world_banner_prompt(world_title: str, world_summary: str) -> str:
    return (
        f"Epic wide banner illustration for a text adventure game. "
        f"Game: {world_title}. {world_summary} "
        f"Style: {_STYLE}, cinematic wide-angle vista, panoramic, "
        f"no text, no UI elements, no characters."
    )


# ── Image path helpers ────────────────────────────────────────────

def image_path(game_dir: Path, kind: str, obj_id: str) -> Path:
    """Return the cached image path. kind: 'room', 'char', 'entity', 'banner'."""
    if kind == "banner":
        return game_dir / "images" / "banner.png"
    return game_dir / "images" / f"{kind}_{obj_id}.png"


def has_image(game_dir: Path, kind: str, obj_id: str = "") -> bool:
    return image_path(game_dir, kind, obj_id).exists()


# Legacy helpers (backwards-compat for existing room routes)
def room_image_path(game_dir: Path, room_id: str) -> Path:
    return image_path(game_dir, "room", room_id)


def has_room_image(game_dir: Path, room_id: str) -> bool:
    return has_image(game_dir, "room", room_id)


# ── Single-image generation ──────────────────────────────────────

async def _generate_image(
    prompt: str,
    output: Path,
    seed: int | None = None,
    provider: ImageProvider | None = None,
    width: int = 768,
    height: int = 512,
    label: str = "",
) -> Path | None:
    prov = provider or _provider
    if prov is None:
        return None
    if output.exists():
        return output
    _log.info("Generating image", label=label)
    try:
        return await prov.generate(prompt, output, width=width, height=height, seed=seed)
    except Exception as exc:
        _log.warning("Image generation failed", label=label, error=str(exc))
        return None


async def generate_room_image(
    game_dir: Path, room_id: str, room_name: str, room_desc: str,
    world_title: str, world_summary: str,
    seed: int | None = None, provider: ImageProvider | None = None,
) -> Path | None:
    prompt = _room_prompt(room_name, room_desc, world_title, world_summary)
    out = image_path(game_dir, "room", room_id)
    return await _generate_image(prompt, out, seed=seed, provider=provider, label=f"room:{room_id}")


async def generate_character_image(
    game_dir: Path, char_id: str, char_name: str, char_desc: str,
    world_title: str, world_summary: str,
    seed: int | None = None, provider: ImageProvider | None = None,
) -> Path | None:
    prompt = _character_prompt(char_name, char_desc, world_title, world_summary)
    out = image_path(game_dir, "char", char_id)
    return await _generate_image(
        prompt, out, seed=seed, provider=provider,
        width=512, height=512, label=f"char:{char_id}",
    )


async def generate_entity_image(
    game_dir: Path, entity_id: str, entity_name: str, entity_desc: str,
    world_title: str, world_summary: str,
    seed: int | None = None, provider: ImageProvider | None = None,
) -> Path | None:
    prompt = _entity_prompt(entity_name, entity_desc, world_title, world_summary)
    out = image_path(game_dir, "entity", entity_id)
    return await _generate_image(
        prompt, out, seed=seed, provider=provider,
        width=512, height=512, label=f"entity:{entity_id}",
    )


async def generate_world_banner(
    game_dir: Path, world_title: str, world_summary: str,
    seed: int | None = None, provider: ImageProvider | None = None,
) -> Path | None:
    prompt = _world_banner_prompt(world_title, world_summary)
    out = image_path(game_dir, "banner", "")
    return await _generate_image(
        prompt, out, seed=seed, provider=provider,
        width=1024, height=512, label="banner",
    )


# ── Batch generation ─────────────────────────────────────────────

async def generate_all_images(
    game_dir: Path,
    world: World,
    seed: int | None = None,
    provider: ImageProvider | None = None,
) -> dict[str, bool]:
    """Generate all images for a game world. Sequential to avoid GPU overload.

    Returns a dict of ``"kind:id" -> success`` entries.
    """
    results: dict[str, bool] = {}
    base_seed = seed or 42
    idx = 0

    # World banner
    path = await generate_world_banner(
        game_dir, world.title, world.summary,
        seed=base_seed + idx, provider=provider,
    )
    results["banner"] = path is not None
    idx += 1

    # Characters
    for char_id, char in world.characters.items():
        path = await generate_character_image(
            game_dir, char_id, char.name, char.description,
            world.title, world.summary,
            seed=base_seed + idx, provider=provider,
        )
        results[f"char:{char_id}"] = path is not None
        idx += 1

    # Entities
    for entity_id, entity in world.entities.items():
        path = await generate_entity_image(
            game_dir, entity_id, entity.name, entity.description,
            world.title, world.summary,
            seed=base_seed + idx, provider=provider,
        )
        results[f"entity:{entity_id}"] = path is not None
        idx += 1

    # Rooms
    for room_id, room in world.rooms.items():
        path = await generate_room_image(
            game_dir, room_id, room.name, room.description,
            world.title, world.summary,
            seed=base_seed + idx, provider=provider,
        )
        results[f"room:{room_id}"] = path is not None
        idx += 1

    generated = sum(1 for v in results.values() if v)
    _log.info("All images generated", total=len(results), generated=generated)
    return results


# Legacy alias
async def generate_all_room_images(
    game_dir: Path, world: World,
    seed: int | None = None, provider: ImageProvider | None = None,
) -> dict[str, Path | None]:
    """Generate room images only (backwards compat)."""
    results: dict[str, Path | None] = {}
    base_seed = seed or 42
    for idx, (room_id, room) in enumerate(world.rooms.items()):
        path = await generate_room_image(
            game_dir, room_id, room.name, room.description,
            world.title, world.summary,
            seed=base_seed + idx, provider=provider,
        )
        results[room_id] = path
    return results
