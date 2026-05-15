"""Shared FD datastore client for agent-coded apps.

A code-app's ``backend.py`` is a normal Python module loaded by the
worker — but it should *not* reach into Captain Claw internals to read
and write data. That couples every agent-generated app to FD's
implementation details and makes the surface impossible to sandbox.

Instead, the worker exposes one tiny, slug-scoped client:

    from captain_claw.flight_deck.app_datastore_client import datastore
    notes = datastore("notes")
    notes.create({"title": "Hello", "body": "World"})
    rows = notes.list()

Under the hood we delegate to the same :class:`LocalEntityStore` that
backs manifest apps (``app_entities.py``), so a code-app and a
manifest app can share a namespace if the agent wires them up that
way. That's the "shared FD datastore" the user signed off on.

Two design points worth flagging:

1. **Scoping is by slug, not by user.** The slug *is* the tenancy
   boundary. If you later need per-user data inside an app, add a
   user_id field on the record — don't try to split the namespace.

2. **No schema validation here.** The agent owns the schema for its
   own app. The datastore is happy to store anything JSON-serializable,
   and the agent is responsible for keeping its own records sane.
   v1 trades safety for friction-free iteration.
"""

from __future__ import annotations

import os
from typing import Any, Protocol

from captain_claw.flight_deck import app_entities


Record = app_entities.Record


# ── per-entity client ─────────────────────────────────────────────────


class _EntityClient:
    """Thin wrapper around ``LocalEntityStore`` that fixes ``(agent_id, entity_id)``."""

    def __init__(self, slug: str, entity_id: str) -> None:
        self._slug = slug
        self._entity_id = entity_id
        self._store = app_entities.get_store()

    # The public API mirrors LocalEntityStore but without the redundant
    # (agent_id, entity_id) on every call.

    def list(self) -> list[Record]:
        """Return all records, newest-created first."""
        return self._store.list(self._slug, self._entity_id)

    def get(self, record_id: str) -> Record | None:
        return self._store.get(self._slug, self._entity_id, record_id)

    def create(self, data: Record) -> Record:
        """Insert a record. ``id``/``created_at``/``updated_at`` are added if missing."""
        return self._store.create(self._slug, self._entity_id, data)

    def update(self, record_id: str, data: Record) -> Record | None:
        """Merge ``data`` into an existing record. Returns the merged record or None."""
        return self._store.update(self._slug, self._entity_id, record_id, data)

    def delete(self, record_id: str) -> bool:
        return self._store.delete(self._slug, self._entity_id, record_id)

    # Convenience: ``len(client)``, ``for r in client``, ``client[id]``.
    def __len__(self) -> int:
        return len(self.list())

    def __iter__(self):
        return iter(self.list())

    def __getitem__(self, record_id: str) -> Record:
        rec = self.get(record_id)
        if rec is None:
            raise KeyError(record_id)
        return rec


# ── top-level entry point ─────────────────────────────────────────────


def _current_slug() -> str:
    """Pick up the slug FD set in the subprocess environment.

    The worker entry point doesn't currently set ``FD_APP_SLUG``; the
    spawner does via ``_build_subprocess_env``. We read it lazily here
    so the constant is set in *one* place.
    """
    slug = os.environ.get("FD_APP_SLUG", "").strip()
    if not slug:
        # Fall back to a marker so misuse is loud rather than silently
        # writing into an empty-string namespace.
        raise RuntimeError(
            "FD_APP_SLUG not set — datastore() is only callable from "
            "inside a Flight Deck app subprocess."
        )
    return slug


def datastore(entity_id: str, *, slug: str | None = None) -> _EntityClient:
    """Return a slug-scoped client for one entity collection.

    Args:
        entity_id: collection name, e.g. ``"notes"`` or ``"tasks"``.
            Becomes a JSON file on disk; allowed chars are alphanumerics
            plus ``-`` and ``_``.
        slug: override the auto-detected app slug. Useful in tests; in
            production agent code should not pass this — leaving it
            ``None`` makes the client pick up the subprocess's slug.

    Example::

        from captain_claw.flight_deck.app_datastore_client import datastore

        notes = datastore("notes")
        notes.create({"title": "First note"})
        for n in notes:
            print(n["title"])
    """
    if not isinstance(entity_id, str) or not entity_id.strip():
        raise ValueError("entity_id must be a non-empty string")
    resolved_slug = (slug or _current_slug()).strip()
    return _EntityClient(resolved_slug, entity_id.strip())


# ── protocol re-export (for type hints in user code) ─────────────────


class EntityClient(Protocol):
    """Public type interface for the per-entity client returned by ``datastore()``."""

    def list(self) -> list[Record]: ...
    def get(self, record_id: str) -> Record | None: ...
    def create(self, data: Record) -> Record: ...
    def update(self, record_id: str, data: Record) -> Record | None: ...
    def delete(self, record_id: str) -> bool: ...


__all__ = ["datastore", "EntityClient", "Record"]
