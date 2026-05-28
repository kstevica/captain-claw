"""Shared helpers for Meta-webhook-based bridges.

Messenger, WhatsApp Cloud API, and Instagram Graph API all share the same
operational scaffolding:

  - GET handshake using ``hub.mode`` / ``hub.verify_token`` / ``hub.challenge``
  - POST signed with HMAC-SHA256 in ``X-Hub-Signature-256``
  - Per-recipient channel binding driven by a ``/c <name>`` slash command
  - Outbound fan-out from the glasses-bridge channel bus to N platform
    recipients via a single per-channel callback subscriber

Each bridge module owns the platform-specific bits (Send API call shape,
attachment fetch flow, env-var names) and imports the helpers below to
avoid drifting copies of the Meta-flavored boilerplate.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import re
from collections.abc import Awaitable, Callable, Iterable
from datetime import datetime, timezone

UTC = timezone.utc


def now_iso() -> str:
    """ISO-8601 UTC timestamp. Bridge events use it on the channel bus."""
    return datetime.now(UTC).isoformat()


# ── Webhook verification ──────────────────────────────────────────────


def verify_signature(body: bytes, signature_header: str, secret: str) -> bool:
    """Validate the ``X-Hub-Signature-256`` header against ``secret``.

    Meta signs every webhook POST body with HMAC-SHA256 keyed by the app
    secret; header format is ``sha256=<hexdigest>``. We refuse requests
    without a valid signature — public webhook endpoints are spam magnets.

    Constant-time compare via :func:`hmac.compare_digest` so we don't leak
    the secret through timing.
    """
    if not secret or not signature_header:
        return False
    if not signature_header.startswith("sha256="):
        return False
    sent = signature_header.split("=", 1)[1].strip()
    expected = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(sent, expected)


def verify_hub_challenge(mode: str, sent_token: str, expected_token: str) -> bool:
    """Constant-time check for the Meta webhook subscribe handshake.

    On webhook subscribe, Meta sends ``GET …?hub.mode=subscribe
    &hub.verify_token=<token>&hub.challenge=<rand>``. We must echo back
    ``hub.challenge`` only when ``mode == "subscribe"`` *and* the
    verify-token matches the one we configured on Meta's side.
    """
    if mode != "subscribe" or not expected_token:
        return False
    return hmac.compare_digest(sent_token or "", expected_token)


# ── Outbound text shaping ─────────────────────────────────────────────


def strip_markdown(s: str) -> str:
    """Quick-and-dirty markdown → plain text for chat threads.

    None of Messenger / WhatsApp / Instagram render markdown — the
    asterisks and underscores would show as literals. Mirrors
    ``stripMarkdownForTts`` in ``glasses_view.html`` (less ambitious; chat
    UIs are more forgiving of stray punctuation than TTS engines).
    """
    s = re.sub(r"`+([^`]+)`+", r"\1", s)
    s = re.sub(r"(\*{2}|_{2})([^*_\n]+)\1", r"\2", s)
    s = re.sub(r"\*([^*\n]+?)\*", r"\1", s)
    s = re.sub(r"(?<![A-Za-z0-9])_([^_\n]+?)_(?![A-Za-z0-9])", r"\1", s)
    s = re.sub(r"\[([^\]\n]+)\]\(([^)\n]+)\)", r"\1", s)
    s = re.sub(r"^\s*[-*+]\s+", "", s, flags=re.M)
    s = re.sub(r"^\s*\d+\.\s+", "", s, flags=re.M)
    s = re.sub(r"^\s*#{1,6}\s+", "", s, flags=re.M)
    return s.strip()


# ── Channel-bus callback fan-out ──────────────────────────────────────


def register_channel_callback(
    *,
    channel_id: str,
    wired_set: set[str],
    recipients_for_channel: Callable[[str], Iterable[str]],
    send_one: Callable[[str, str], Awaitable[None]],
) -> None:
    """Attach a callback subscriber that fans agent replies on ``channel_id``
    out to every platform recipient currently bound to that channel.

    Cross-bridge note
    -----------------
    Each bridge calls this with its **own** ``wired_set`` and lookup, so a
    single channel can have **multiple** callbacks attached (one per
    bridge). When the agent replies, both fire — that's how a Messenger
    user with ``/c lounge`` and a WhatsApp user with ``/c lounge`` get
    the same agent reply on their respective threads.

    Parameters
    ----------
    channel_id
        The channel the recipient is bound to. We install at most one
        callback per (bridge, channel) pair.
    wired_set
        A bridge-owned ``set[str]`` of channel ids it has already wired
        — keeps this call idempotent without us hardcoding "Messenger" /
        "WhatsApp" state here.
    recipients_for_channel
        Function returning the bridge's current recipient ids for the
        given channel (called fresh on every broadcast, so rebinds via
        ``/c`` take effect immediately).
    send_one
        Async function the bridge provides to push one text message to
        one recipient. Failure to send to one recipient must not block
        the others.
    """
    if channel_id in wired_set:
        return

    # Lazy import — this module sits above glasses_bridge in the import
    # graph and we want to keep that direction.
    from captain_claw.flight_deck.glasses_bridge import _channels

    ch = _channels.get(channel_id)
    if ch is None:
        return

    async def _forward(payload: dict) -> None:
        # Surface only outbound content the user would want. Skip ``user``
        # echoes (the user already typed it on their own screen) and
        # ``status`` heartbeats (noise on a chat thread).
        mtype = payload.get("type")
        if mtype not in ("agent", "error"):
            return
        text = str(payload.get("text") or "").strip()
        if not text:
            return
        ids = list(recipients_for_channel(channel_id))
        if not ids:
            return
        plain = strip_markdown(text)
        for rid in ids:
            try:
                await send_one(rid, plain)
            except Exception:
                # One dead recipient must not block the others.
                continue

    ch.callback_subscribers.append(_forward)
    wired_set.add(channel_id)


# ── Agent target resolution ───────────────────────────────────────────


def resolve_agent_target(
    *,
    slug_env: str,
    port_env: str,
    host_env: str = "",
) -> tuple[str, int]:
    """Resolve a (host, port) for the bridge's outbound agent connection.

    Flight Deck reassigns web ports on every agent spawn — a fixed port
    in the bridge's env breaks the moment FD restarts the agent. Binding
    by **agent slug** instead and looking up the current port at message
    receive time survives that.

    Resolution priority
    -------------------
    1. ``<port_env>`` set and parseable as int > 0
       → legacy fixed-port mode. Used when the bridge points at an agent
       FD doesn't manage (rare). ``<host_env>`` honoured here.
    2. ``<slug_env>`` set
       → look up that slug in FD's process registry, return its current
       ``web_port`` if alive. Returns ``(localhost, 0)`` if the named
       slug isn't running — explicit binding means "this agent or
       nothing"; we don't auto-pick a different agent.
    3. Neither set
       → return the first alive agent in the registry that has a
       ``web_port``. Single-agent users get zero-config; multi-agent
       users should set the slug env var.

    Returns ``(host, port)``. ``port == 0`` means "no agent available"
    — caller should bail and tell the user.

    Called per inbound message, not cached. The registry lookup is a
    JSON read off disk — sub-millisecond and fresh every time, so port
    floats between agent restarts are picked up automatically.
    """
    host_default = (os.environ.get(host_env, "").strip() or "localhost") if host_env else "localhost"

    # 1. Legacy fixed port wins if both are set.
    raw_port = os.environ.get(port_env, "").strip()
    if raw_port:
        try:
            port = int(raw_port)
        except ValueError:
            port = 0
        if port > 0:
            return host_default, port

    # 2/3. Need FD's process registry. Lazy import so this module stays
    # importable even when server.py isn't (e.g. in isolated tests).
    try:
        from captain_claw.flight_deck.server import (
            _load_process_registry,
            _process_is_alive,
        )
    except Exception:
        return "localhost", 0

    try:
        registry = _load_process_registry()
    except Exception:
        return "localhost", 0

    slug = os.environ.get(slug_env, "").strip()
    if slug:
        # Explicit slug: this agent or nothing. No fallback — surprise
        # routing to a different agent on misconfig is worse than failing.
        entry = registry.get(slug)
        if not entry or not _process_is_alive(slug):
            return "localhost", 0
        try:
            port = int(entry.get("web_port", 0) or 0)
        except (TypeError, ValueError):
            port = 0
        return "localhost", port

    # 3. Auto-pick first running agent with a web port. Iteration order
    # is dict-insertion order (Python 3.7+) — FD writes the registry
    # alphabetically, so the choice is stable across calls until a new
    # agent is spawned.
    for cand_slug, entry in registry.items():
        if not _process_is_alive(cand_slug):
            continue
        try:
            port = int(entry.get("web_port", 0) or 0)
        except (TypeError, ValueError):
            continue
        if port > 0:
            return "localhost", port

    return "localhost", 0
