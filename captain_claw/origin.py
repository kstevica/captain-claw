"""Durable per-session *origin* descriptor.

Records where a session's messages came from — which channel, and the address
on that channel to reply to — so that a result produced *asynchronously* (a
cron / scheduled / deferred task that finishes long after the user's message)
can be routed back to the same place.

It lives on ``session.metadata['origin'] = {'kind': ..., 'address': ...}`` and
is therefore persisted with the session on disk. Unlike the old in-memory
channel↔WAID maps in Flight Deck, this survives restarts and is recoverable
from the session id alone — which is exactly what a deferred job has.

``kind`` is one of the KIND_* constants; ``address`` is the channel-specific
reply target (a WhatsApp WAID, a Telegram chat id, a channel name, …).
"""

from __future__ import annotations

from typing import Any

ORIGIN_KEY = "origin"

KIND_WHATSAPP = "whatsapp"
KIND_TELEGRAM = "telegram"
KIND_GLASSES = "glasses"
KIND_CHANNEL = "channel"
KIND_WEB = "web"

_VALID_KINDS = frozenset({
    KIND_WHATSAPP, KIND_TELEGRAM, KIND_GLASSES, KIND_CHANNEL, KIND_WEB,
})


def set_session_origin(session: Any, kind: str, address: str) -> None:
    """Stamp a session with where it came from. No-op on bad input so callers
    can fire-and-forget on every inbound message."""
    if session is None:
        return
    kind = (kind or "").strip()
    address = str(address or "").strip()
    if not kind or not address:
        return
    try:
        if getattr(session, "metadata", None) is None:
            session.metadata = {}
        session.metadata[ORIGIN_KEY] = {"kind": kind, "address": address}
    except Exception:
        pass


def get_session_origin(session: Any) -> dict[str, str] | None:
    """Return ``{'kind','address'}`` for a session, or None when unknown."""
    try:
        o = (getattr(session, "metadata", None) or {}).get(ORIGIN_KEY)
    except Exception:
        o = None
    if isinstance(o, dict):
        kind = str(o.get("kind") or "").strip()
        address = str(o.get("address") or "").strip()
        if kind and address:
            return {"kind": kind, "address": address}
    return None


def normalize_origin(origin: Any) -> dict[str, str] | None:
    """Validate a raw origin dict from the wire into ``{'kind','address'}``."""
    if not isinstance(origin, dict):
        return None
    kind = str(origin.get("kind") or "").strip()
    address = str(origin.get("address") or "").strip()
    if kind and address:
        return {"kind": kind, "address": address}
    return None
