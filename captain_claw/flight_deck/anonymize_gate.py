"""The privacy gate's setting, per person, and whether it applies to an agent.

Kept in the person's settings (``anonymize:gate``; the deck's system settings
when sign-in is off, where there is one person). Off until the person turns it
on. When on, it applies to an agent whose current model isn't trusted — not
local and not the admin's trusted model — or, with ``when: always``, to every
agent.
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from captain_claw.anonymize import gate as G
from captain_claw.flight_deck import anonymize_trusted as trusted
from captain_claw.flight_deck.auth import _fd_auth_enabled, get_db

SETTING_KEY = "anonymize:gate"
MAX_FOLDERS = 50


class GateError(ValueError):
    pass


@dataclass
class GateSettings:
    enabled: bool = False
    groups: list[str] = field(default_factory=lambda: list(G.DEFAULT_GROUPS))
    folders: list[str] = field(default_factory=list)
    when: str = "untrusted"
    lane: str = "deterministic"
    coverage: str = "sources"  # or "chat": everything that reaches the model

    def public(self) -> dict[str, Any]:
        return asdict(self)


def _clean(data: dict[str, Any]) -> GateSettings:
    groups = data.get("groups", list(G.DEFAULT_GROUPS))
    if not isinstance(groups, list) or any(g not in G.GROUPS for g in groups):
        raise GateError(f"groups are some of: {', '.join(G.GROUPS)}")
    folders = data.get("folders", [])
    if not isinstance(folders, list) or len(folders) > MAX_FOLDERS:
        raise GateError(f"folders is a list of at most {MAX_FOLDERS} paths")
    folders = [str(f).strip() for f in folders if str(f or "").strip()]
    if any(len(f) > 500 for f in folders):
        raise GateError("a folder path is at most 500 characters")
    when = str(data.get("when") or "untrusted")
    lane = str(data.get("lane") or "deterministic")
    if when not in G.WHEN:
        raise GateError("when is 'untrusted' or 'always'")
    if lane not in G.LANES:
        raise GateError("lane is 'deterministic' or 'both'")
    coverage = str(data.get("coverage") or "sources")
    if coverage not in G.COVERAGE:
        raise GateError("coverage is 'sources' or 'chat'")
    return GateSettings(bool(data.get("enabled")), list(dict.fromkeys(groups)), folders, when, lane, coverage)


async def load(person: str) -> GateSettings:
    db = get_db()
    raw = await db.get_system_setting(SETTING_KEY) if not _fd_auth_enabled() else (
        await db.get_setting(person, SETTING_KEY) if person else None)
    if not raw:
        return GateSettings()
    try:
        return _clean(json.loads(raw))
    except (ValueError, GateError):
        return GateSettings()


async def save(person: str, data: dict[str, Any]) -> GateSettings:
    settings = _clean(data)
    db = get_db()
    value = json.dumps(settings.public())
    if not _fd_auth_enabled():
        await db.set_system_setting(SETTING_KEY, value)
    else:
        await db.set_settings(person, {SETTING_KEY: value})
    return settings


def _norm_url(url: str) -> str:
    return re.sub(r"/+$", "", str(url or "").strip().lower())


async def model_trusted(provider: str, model: str, base_url: str) -> bool:
    """An agent's model is trusted when it runs locally, or it is the admin's
    usable trusted model."""
    if await asyncio.to_thread(trusted.is_local, provider, model, base_url):
        return True
    m = await trusted.usable(get_db())
    return bool(m and m.provider == provider.strip().lower() and m.model == model.strip()
                and _norm_url(m.base_url) == _norm_url(base_url))


async def decide(person: str, provider: str, model: str, base_url: str) -> dict[str, Any]:
    """What the gate asks of an agent right now: ``active`` and, when it is,
    what to cover, the vault and the lane."""
    settings = await load(person)
    if not settings.enabled:
        return {"active": False, "reason": "off"}
    if settings.when == "untrusted" and await model_trusted(provider, model, base_url):
        return {"active": False, "reason": "trusted model"}
    lane = settings.lane
    model = await trusted.usable(get_db()) if lane == "both" else None
    if model is None:
        lane = "deterministic"
    chat = settings.coverage == "chat"
    return {"active": True, "chat": chat, "groups": [] if chat else settings.groups,
            "folders": [] if chat else settings.folders, "vault": G.GATE_VAULT, "lane": lane,
            "model": model.label if model else "", "reason": settings.when, "epoch": await _vault_epoch(person)}


async def _vault_epoch(person: str) -> float:
    """When the person's gate vault was last reset (0: never) — an agent that
    sees it change forgets the texts it anonymized before."""
    from captain_claw.flight_deck import anonymize_vault as av

    try:
        row = await av.store().get(person, G.GATE_VAULT)
    except Exception:  # noqa: BLE001 — no store, no vault: nothing to forget
        return 0.0
    return float((row or {}).get("reset_at") or 0.0)


async def chat_restores(person: str) -> bool:
    """Whether the person's own chat shows real values (the gate covers the
    whole chat). Flight Deck restores what their agents send them."""
    settings = await load(person)
    return settings.enabled and settings.coverage == "chat"
