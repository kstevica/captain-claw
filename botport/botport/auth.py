"""Simple key/secret authentication for BotPort."""

from __future__ import annotations

from botport.config import get_config


def validate_credentials(key: str, secret: str) -> bool:
    """Validate instance credentials against the configured key list.

    Returns ``True`` if auth is disabled or credentials match an entry.
    """
    cfg = get_config()
    if not cfg.auth.enabled:
        return True

    key = key.strip()
    secret = secret.strip()
    if not key:
        return False

    for entry in cfg.auth.keys:
        if entry.key.strip() == key and entry.secret.strip() == secret:
            return True

    return False


def get_instance_name_for_key(key: str) -> str | None:
    """Return the configured instance name for a given key, or None."""
    cfg = get_config()
    for entry in cfg.auth.keys:
        if entry.key.strip() == key.strip():
            return entry.instance or None
    return None
