"""One linked WhatsApp account's state, and what each WhatsApp event does to it.

Pure apart from :meth:`LinkState.load` / :meth:`LinkState.save`: the engine
feeds :func:`apply_event` the events its client reports and rings a bell for
each :class:`Notice` it gets back. The pairing QR and code are transient —
they are never written to ``state.json``.

States:

* ``pairing`` — waiting for a QR scan or a pairing code
* ``connecting`` — paired (or resuming), not connected yet
* ``connected`` / ``disconnected`` — whatsmeow reconnects on its own
* ``expired`` — nobody scanned in time; the session was discarded
* ``failed`` — the connection could not be set up
* ``logged_out`` — unlinked from the phone, or WhatsApp banned the number; the
  session is gone, the messages stay (:mod:`.kept_history`)
* ``stream_replaced`` — another client took this session over
* ``temporary_ban`` — WhatsApp barred the account for a while
* ``outdated`` — WhatsApp refuses this client version (update neonize)
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

STATES = (
    "pairing", "connecting", "connected", "disconnected", "expired", "failed",
    "logged_out", "stream_replaced", "temporary_ban", "outdated",
)
# Not resumed after a restart: the session is gone, or resuming would fight
# whoever holds it now, or WhatsApp would refuse it again.
NOT_RESUMED = frozenset({"expired", "failed", "logged_out", "stream_replaced", "outdated"})

_JID = re.compile(r"^(?P<user>[^@:]+)(?::\d+)?@(?P<server>.+)$")


def now_iso(ts: float | None = None) -> str:
    dt = datetime.fromtimestamp(ts, UTC) if ts is not None else datetime.now(UTC)
    return dt.isoformat(timespec="seconds")


def bare_jid(jid: str) -> str:
    """``385976707736:56@s.whatsapp.net`` → ``385976707736@s.whatsapp.net``
    (whatsmeow reports the device's JID; the account is the bare one)."""
    m = _JID.match((jid or "").strip())
    return f"{m['user']}@{m['server']}" if m else (jid or "").strip()


def number_of(jid: str) -> str:
    """The phone number of a ``…@s.whatsapp.net`` JID, as ``+385976707736``."""
    m = _JID.match(jid or "")
    if not m or m["server"] != "s.whatsapp.net" or not m["user"].isdigit():
        return ""
    return "+" + m["user"]


def normalize_phone(raw: str) -> str:
    """A phone number for a pairing code: digits only, country code first.

    Accepts ``+385 97 670 7736``, ``00385976707736`` or ``385976707736``.
    Raises ValueError for anything that can't be one (7-15 digits, E.164)."""
    text = (raw or "").strip()
    if text.startswith("00"):
        text = text[2:]
    digits = re.sub(r"[\s\-().+]", "", text)
    if not digits.isdigit() or not 7 <= len(digits) <= 15 or digits.startswith("0"):
        raise ValueError(
            "Enter the number with its country code, e.g. +385 97 123 4567")
    return digits


@dataclass(frozen=True)
class Notice:
    """A bell for the link's owner."""

    title: str
    body: str


@dataclass
class LinkState:
    state: str = "pairing"
    jid: str = ""
    consented_at: str = ""
    created_at: str = ""
    linked_at: str = ""
    connected_at: str = ""
    last_event_at: str = ""
    detail: str = ""
    ban_expires_at: str = ""
    sync: dict[str, Any] = field(default_factory=lambda: {
        "chunks": 0, "messages": 0, "progress": 0, "last_type": "", "last_at": "",
    })
    # Transient: shown while pairing, never saved.
    qr: str = ""
    qr_at: float = 0.0
    pair_code: str = ""

    _TRANSIENT = ("qr", "qr_at", "pair_code")

    # ── persistence ─────────────────────────────────────────────────

    @classmethod
    def load(cls, path: Path) -> LinkState | None:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if not isinstance(raw, dict):
            return None
        known = {k: v for k, v in raw.items() if k in cls.__dataclass_fields__ and k not in cls._TRANSIENT}
        st = cls(**known)
        if st.state not in STATES:
            st.state = "failed"
        return st

    def save(self, path: Path) -> None:
        data = {k: v for k, v in asdict(self).items() if k not in self._TRANSIENT}
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
        os.chmod(tmp, 0o600)
        tmp.replace(path)

    # ── what the card shows ─────────────────────────────────────────

    def view(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "jid": self.jid,
            "number": number_of(self.jid),
            "linked_at": self.linked_at or None,
            "connected_at": self.connected_at or None,
            "last_event_at": self.last_event_at or None,
            "detail": self.detail,
            "ban_expires_at": self.ban_expires_at or None,
            "sync": dict(self.sync),
            "pair_code": self.pair_code if self.state == "pairing" else "",
        }


def apply_event(st: LinkState, name: str, data: dict[str, Any] | None = None,
                *, now: float) -> Notice | None:
    """Move *st* on for one event from the client; return the owner's bell, if
    this event deserves one. Unknown events only touch ``last_event_at``."""
    data = data or {}
    st.last_event_at = now_iso(now)

    if name == "qr":
        st.state = "pairing"
        st.qr = str(data.get("code") or "")
        st.qr_at = now
        return None
    if name == "pair_code":
        st.pair_code = str(data.get("code") or "")
        return None
    if name == "paired":
        st.jid = bare_jid(str(data.get("jid") or ""))
        st.linked_at = now_iso(now)
        st.state = "connecting"
        st.qr, st.qr_at, st.pair_code = "", 0.0, ""
        st.detail = ""
        return None
    if name == "connected":
        st.state = "connected"
        st.connected_at = now_iso(now)
        st.detail = ""
        st.ban_expires_at = ""
        return None
    if name == "disconnected":
        if st.state == "connected":
            st.state = "disconnected"
            st.detail = "Reconnecting…"
        return None
    if name == "history":
        sync = st.sync
        sync["chunks"] = int(sync.get("chunks") or 0) + 1
        sync["messages"] = int(sync.get("messages") or 0) + int(data.get("messages") or 0)
        sync["progress"] = max(int(sync.get("progress") or 0), int(data.get("progress") or 0))
        sync["last_type"] = str(data.get("sync_type") or "")
        sync["last_at"] = now_iso(now)
        return None
    if name == "expired":
        st.state = "expired"
        st.qr, st.qr_at, st.pair_code = "", 0.0, ""
        st.detail = "Nobody scanned the code in time."
        return None
    if name == "connect_failure":
        reason = str(data.get("reason") or "").strip()
        if st.state in ("pairing", "connecting") and not st.linked_at:
            st.state = "failed"
        st.detail = f"WhatsApp refused the connection{': ' + reason if reason else ''}."
        return None
    if name == "logged_out":
        st.state = "logged_out"
        st.detail = "WhatsApp ended this link: it was removed on the phone, or WhatsApp banned the number."
        return Notice(
            "WhatsApp Web was unlinked",
            f"{number_of(st.jid) or 'Your account'} is no longer linked to Flight Deck: it was removed on "
            "the phone, or WhatsApp banned it. Your messages stay, and your agents can still read them. "
            "Link a number or delete them in Connections → WhatsApp Web.")
    if name == "stream_replaced":
        st.state = "stream_replaced"
        st.detail = "Another WhatsApp Web session took this link over."
        return Notice(
            "WhatsApp Web stopped",
            "Another program took over this WhatsApp Web session, so Flight Deck let go "
            "of it. Reconnect in Connections → WhatsApp Web once the other one is gone.")
    if name == "temporary_ban":
        expires = data.get("expires_at")
        st.state = "temporary_ban"
        st.ban_expires_at = now_iso(float(expires)) if isinstance(expires, (int, float)) and expires > 0 else ""
        until = f" until {st.ban_expires_at}" if st.ban_expires_at else ""
        st.detail = f"WhatsApp has temporarily banned this account{until}."
        return Notice("WhatsApp temporarily banned your account",
                      f"WhatsApp reports a temporary ban{until}. Flight Deck sends nothing "
                      "meanwhile. Use WhatsApp on your phone as usual.")
    if name == "outdated":
        st.state = "outdated"
        st.detail = "WhatsApp refuses this client version; the deck needs an update."
        return Notice("WhatsApp Web needs an update",
                      "WhatsApp no longer accepts this deck's WhatsApp Web client. Ask your "
                      "admin to update Captain Claw.")
    return None
