"""The external-memory providers this Flight Deck knows. Add one by writing a
:class:`~captain_claw.flight_deck.external_memory.base.ExternalMemoryProvider`
subclass and listing it here — the routes, the UI's provider picker and the
agent tool all read this table."""

from __future__ import annotations

from typing import Any

from captain_claw.flight_deck.external_memory.base import ConnectionInfo, ExternalMemoryProvider
from captain_claw.flight_deck.external_memory.hindsight import HindsightProvider
from captain_claw.flight_deck.external_memory.membase import MembaseProvider

PROVIDERS: dict[str, type[ExternalMemoryProvider]] = {
    MembaseProvider.name: MembaseProvider,
    HindsightProvider.name: HindsightProvider,
}

# Tests swap a provider's construction (to hand it a mock transport).
_factory_override: dict[str, Any] = {}


def provider_class(name: str) -> type[ExternalMemoryProvider] | None:
    return PROVIDERS.get((name or "").strip().lower())


def build(name: str, credentials: dict[str, str], info: ConnectionInfo | None = None,
          **kwargs: Any) -> ExternalMemoryProvider:
    """A provider instance for one call. Raises ``KeyError`` for an unknown name."""
    key = (name or "").strip().lower()
    if key in _factory_override:
        return _factory_override[key](credentials, info, **kwargs)
    cls = PROVIDERS[key]
    return cls(credentials, info, **kwargs)


def catalog() -> list[dict[str, Any]]:
    return [cls.catalog_entry() for cls in PROVIDERS.values()]
