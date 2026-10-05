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

import hashlib
import json
import os
from pathlib import Path

# Keys row_manifest stamps on from the registry row — identity, not content.
_REGISTRY_STAMPS = ("slug", "pack_status", "pack_version")


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


def list_seed_packs() -> dict[str, dict]:
    """All repo packs (lupa/packs/*/pack.json) — imported into the registry at
    startup as published system packs. The registry is the runtime authority;
    seeds only fill gaps, they never overwrite runtime edits."""
    out: dict[str, dict] = {}
    root = packs_root()
    if root.is_dir():
        for d in sorted(root.iterdir()):
            if (d / "pack.json").is_file():
                try:
                    out[d.name] = load_pack(d.name)
                except (ValueError, OSError):
                    continue
    return out


def row_manifest(row: dict) -> dict:
    """A registry row's manifest, stamped with its registry identity."""
    try:
        manifest = json.loads(row.get("manifest") or "{}")
    except (ValueError, TypeError):
        manifest = {}
    manifest["slug"] = row["slug"]
    manifest["pack_status"] = row.get("status", "")
    manifest["pack_version"] = row.get("version", 0)
    return manifest


def _canon(v):
    """85.0 → 85: the Studio's JSON round-trip through the browser drops the
    '.0', and an unchanged save must not read as an edit."""
    if isinstance(v, float) and v.is_integer():
        return int(v)
    if isinstance(v, dict):
        return {k: _canon(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_canon(x) for x in v]
    return v


def manifest_hash(manifest: dict) -> str:
    """A content hash of a manifest, ignoring the registry stamps — what the
    ship-gate binds an eval verdict to, so only a verdict for the manifest as it
    is NOW can publish it."""
    body = {k: _canon(v) for k, v in (manifest or {}).items()
            if k not in _REGISTRY_STAMPS}
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
