"""Pack loading — the vertical-as-data layer (plan Part II, "Kalup").

Phase 1 scope: packs are directories under ``lupa/packs/``; the active pack is
chosen by the ``LUPA_PACK`` env (default ``research-desk``). The runtime pack
registry (DB-backed, Pack Studio) replaces this loader later — the manifest
shape is the contract, not the storage.

Every vertical-specific surface the SPA renders — name, theme tokens,
vocabulary, intake types, quality profile, onboarding copy — comes from here.
Nothing vertical-specific may be hardcoded in the shell.
"""

from __future__ import annotations

import json
import os
from pathlib import Path


def packs_root() -> Path:
    env = os.environ.get("LUPA_PACKS_DIR", "")
    if env:
        return Path(env)
    # lupa/api/lupa_api/packs.py → lupa/packs
    return Path(__file__).resolve().parents[2] / "packs"


def active_pack_slug() -> str:
    return os.environ.get("LUPA_PACK", "research-desk").strip() or "research-desk"


def load_pack(slug: str | None = None) -> dict:
    slug = slug or active_pack_slug()
    root = packs_root() / slug
    manifest = json.loads((root / "pack.json").read_text(encoding="utf-8"))
    manifest["slug"] = slug
    onboarding = root / "onboarding.md"
    if onboarding.is_file():
        manifest["onboarding_md"] = onboarding.read_text(encoding="utf-8")
    return manifest


def pack_quality(pack: dict) -> dict:
    """The quality dict sent to FD with every commission — the pack's preset."""
    return dict(pack.get("quality") or {})
