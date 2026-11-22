"""WhatsApp protos → the rows the store keeps. Pure, so every message kind is
tested by building its proto (no WhatsApp, no neonize client).

Every function takes neonize's protobuf objects (or anything shaped like them)
and returns plain dicts. One *op* per message:

* ``message`` — a message worth keeping: text, a caption, or media
* ``edit`` / ``revoke`` — change or blank an earlier message (``target`` = its id)
* ``react`` — a reaction to an earlier message (``emoji`` "" = reaction removed)
* ``skip`` — technical traffic: sender keys, key shares, history notices, empty
  shells. Never stored. A group message that carries a new sender key arrives
  as two events with ONE id (key only, then content): the content's event
  wins because ids are upserted and ``skip`` never writes.

Field names follow neonize's protos, which don't follow one pattern:
``AudioMessage.PTT``, ``MessageInfo.Pushname``, ``MessageKey.ID``,
``DocumentMessage.fileName``.
"""

from __future__ import annotations

from typing import Any

# Content fields → kind, checked in this order (a message has one of them).
_MEDIA = (
    ("imageMessage", "image"),
    ("videoMessage", "video"),
    ("audioMessage", "audio"),
    ("documentMessage", "document"),
    ("stickerMessage", "sticker"),
    ("ptvMessage", "video_note"),
)
_OTHER = (
    ("locationMessage", "location"),
    ("liveLocationMessage", "location"),
    ("contactMessage", "contact"),
    ("contactsArrayMessage", "contacts"),
    ("pollCreationMessage", "poll"),
    ("pollCreationMessageV2", "poll"),
    ("pollCreationMessageV3", "poll"),
    ("eventMessage", "event"),
)
# Wrappers whose ``message`` holds the real one.
_WRAPPERS = (
    "ephemeralMessage", "viewOnceMessage", "viewOnceMessageV2", "viewOnceMessageV2Extension",
    "documentWithCaptionMessage", "editedMessage",
)


def ts_seconds(value: Any) -> int:
    """A WhatsApp timestamp as unix seconds (live events carry milliseconds,
    history seconds)."""
    try:
        v = int(value or 0)
    except (TypeError, ValueError):
        return 0
    return v // 1000 if v > 100_000_000_000 else v


def jid_str(j: Any) -> str:
    """A neonize JID proto → ``user@server`` (no device part)."""
    if isinstance(j, str):
        return j
    user = getattr(j, "User", "")
    return f"{user}@{getattr(j, 'Server', '')}" if user else ""


def _has(m: Any, field: str) -> bool:
    try:
        return bool(m.HasField(field))
    except (ValueError, AttributeError):
        return False


def unwrap(m: Any) -> Any:
    """The message inside ephemeral / view-once / document-with-caption wrappers."""
    for _ in range(4):
        for w in _WRAPPERS:
            if _has(m, w):
                inner = getattr(getattr(m, w), "message", None)
                if inner is not None:
                    m = inner
                    break
        else:
            return m
    return m


def _context_of(m: Any) -> Any:
    for field in ("extendedTextMessage", *(f for f, _ in _MEDIA)):
        if _has(m, field):
            sub = getattr(m, field)
            if _has(sub, "contextInfo"):
                return sub.contextInfo
    return None


def content(m: Any) -> dict[str, Any] | None:
    """A message's kept content — ``{kind, text, media, reply_to}`` — or None
    when it has none (technical traffic)."""
    if m is None:
        return None
    m = unwrap(m)
    ctx = _context_of(m)
    reply_to = str(getattr(ctx, "stanzaID", "") or "") if ctx is not None else ""
    if getattr(m, "conversation", ""):
        return {"kind": "text", "text": m.conversation, "media": None, "reply_to": reply_to}
    if _has(m, "extendedTextMessage"):
        text = m.extendedTextMessage.text
        if text:
            return {"kind": "text", "text": text, "media": None, "reply_to": reply_to}
    for field, kind in _MEDIA:
        if _has(m, field):
            sub = getattr(m, field)
            if field == "audioMessage" and getattr(sub, "PTT", False):
                kind = "voice"
            media = {
                "field": field,
                "mimetype": str(getattr(sub, "mimetype", "") or ""),
                "size": int(getattr(sub, "fileLength", 0) or 0),
                "seconds": int(getattr(sub, "seconds", 0) or 0),
                "file_name": str(getattr(sub, "fileName", "") or ""),
                # Everything a later download needs (media key, direct path …).
                "raw": sub.SerializeToString().hex() if hasattr(sub, "SerializeToString") else "",
            }
            return {"kind": kind, "text": str(getattr(sub, "caption", "") or ""),
                    "media": media, "reply_to": reply_to}
    for field, kind in _OTHER:
        if _has(m, field):
            sub = getattr(m, field)
            text = ""
            if kind == "location":
                lat, lng = getattr(sub, "degreesLatitude", 0), getattr(sub, "degreesLongitude", 0)
                label = getattr(sub, "name", "") or getattr(sub, "address", "") or ""
                text = f"{label} ({lat:.5f}, {lng:.5f})".strip() if (lat or lng) else str(label)
            elif kind == "contact":
                text = str(getattr(sub, "displayName", "") or "")
            elif kind == "poll":
                opts = [getattr(o, "optionName", "") for o in getattr(sub, "options", [])]
                text = f"{getattr(sub, 'name', '')} — options: {', '.join(o for o in opts if o)}"
            elif kind == "event":
                text = str(getattr(sub, "name", "") or "")
            return {"kind": kind, "text": text, "media": None, "reply_to": reply_to}
    return None


def classify(m: Any) -> dict[str, Any]:
    """What one message (an E2E ``Message``) means for the store: the op and,
    for edit / revoke / react, which earlier message it targets."""
    if m is None:
        return {"op": "skip"}
    m = unwrap(m)
    if _has(m, "reactionMessage"):
        r = m.reactionMessage
        return {"op": "react", "target": str(r.key.ID or ""), "emoji": str(r.text or ""),
                "react_ts": ts_seconds(getattr(r, "senderTimestampMS", 0))}
    if _has(m, "protocolMessage"):
        p = m.protocolMessage
        type_name = p.Type.Name(p.type) if hasattr(p, "Type") else ""
        if type_name == "REVOKE":
            return {"op": "revoke", "target": str(p.key.ID or "")}
        if type_name == "MESSAGE_EDIT" and _has(p, "editedMessage"):
            new = content(p.editedMessage)
            return {"op": "edit", "target": str(p.key.ID or ""),
                    "text": (new or {}).get("text", "")}
        return {"op": "skip"}
    found = content(m)
    if found is None:
        return {"op": "skip"}
    return {"op": "message", **found}


def live_row(ev: Any) -> dict[str, Any]:
    """A neonize ``MessageEv`` → one row (see :func:`classify` for ``op``)."""
    info = ev.Info
    src = info.MessageSource
    row = classify(ev.Message if _has(ev, "Message") else None)
    row.update({
        "chat": jid_str(src.Chat),
        "id": str(info.ID or ""),
        "sender": jid_str(src.Sender),
        "from_me": bool(src.IsFromMe),
        "is_group": bool(src.IsGroup),
        "ts": ts_seconds(info.Timestamp),
        "push_name": str(getattr(info, "Pushname", "") or ""),
    })
    return row


# Receipts that say what happened to a message this account sent, by WhatsApp's
# name for them → what the store records. The others (``SENDER``, ``READ_SELF``,
# ``PEER_MSG`` …) are the account's own devices talking among themselves.
RECEIPT_KINDS = {
    "DELIVERED": "delivered", "READ": "read", "PLAYED": "read",
    "RETRY": "retry", "SERVER_ERROR": "error", "INACTIVE": "inactive",
}


def receipt(ev: Any) -> dict[str, Any]:
    """A neonize ``Receipt`` → ``kind`` (a :data:`RECEIPT_KINDS` value, or ""
    for one that doesn't concern a sent message), the message ids and who sent
    the receipt."""
    raw = getattr(ev, "Type", "")
    try:
        name = raw if isinstance(raw, str) else ev.ReceiptType.Name(raw)
    except (AttributeError, ValueError):
        name = ""
    src = getattr(ev, "MessageSource", None)
    from_me = bool(getattr(src, "IsFromMe", False))
    kind = RECEIPT_KINDS.get(str(name).upper(), "")
    if from_me and kind in ("delivered", "read"):
        kind = ""  # one of this account's own devices, not the other side
    return {
        "kind": kind,
        "ids": [str(i) for i in getattr(ev, "MessageIDs", []) if i],
        "chat": jid_str(getattr(src, "Chat", "")),
        "sender": jid_str(getattr(src, "Sender", "")),
        "ts": ts_seconds(getattr(ev, "Timestamp", 0)),
    }


def history_rows(conv: Any) -> list[dict[str, Any]]:
    """One history-sync ``Conversation`` → its rows, the reactions it carries
    as ``react`` rows."""
    chat = str(conv.ID or "")
    is_group = chat.endswith("@g.us")
    out: list[dict[str, Any]] = []
    for item in conv.messages:
        info = item.message
        key = info.key
        msg = info.message if _has(info, "message") else None
        row = classify(msg)
        from_me = bool(key.fromMe)
        sender = str(key.participant or info.participant or "") or ("" if from_me else chat)
        row.update({
            "chat": chat, "id": str(key.ID or ""), "sender": sender, "from_me": from_me,
            "is_group": is_group, "ts": ts_seconds(info.messageTimestamp),
            "push_name": str(info.pushName or ""),
        })
        out.append(row)
        for r in getattr(info, "reactions", []):
            out.append({
                "op": "react", "target": str(key.ID or ""), "emoji": str(r.text or ""),
                "react_ts": ts_seconds(getattr(r, "senderTimestampMS", 0)),
                "chat": chat, "id": "", "is_group": is_group,
                "sender": str(r.key.participant or "") or ("" if r.key.fromMe else chat),
                "from_me": bool(r.key.fromMe), "ts": ts_seconds(getattr(r, "senderTimestampMS", 0)),
                "push_name": "",
            })
    return out


def conversation_meta(conv: Any) -> dict[str, Any]:
    """A history ``Conversation``'s chat details (no messages)."""
    return {
        "jid": str(conv.ID or ""),
        "name": str(getattr(conv, "name", "") or getattr(conv, "displayName", "") or ""),
        "is_group": str(conv.ID or "").endswith("@g.us"),
        "archived": bool(getattr(conv, "archived", False)),
        "pinned": int(getattr(conv, "pinned", 0) or 0) > 0,
        "mute_until": ts_seconds(getattr(conv, "muteEndTime", 0)),
        "unread": int(getattr(conv, "unreadCount", 0) or 0),
        "last_ts": ts_seconds(getattr(conv, "lastMsgTimestamp", 0)),
        "pn": str(getattr(conv, "pnJID", "") or ""),
        "lid": str(getattr(conv, "lidJID", "") or ""),
    }


def history_chunk(hs: Any) -> dict[str, Any]:
    """A decoded ``HistorySync`` → everything the store takes from it."""
    rows: list[dict[str, Any]] = []
    chats: list[dict[str, Any]] = []
    for conv in hs.conversations:
        chats.append(conversation_meta(conv))
        rows.extend(history_rows(conv))
    lid_map = [{"pn": str(m.pnJID or ""), "lid": str(m.lidJID or "")}
               for m in getattr(hs, "phoneNumberToLidMappings", [])]
    for c in chats:
        if c["pn"] and c["lid"]:
            lid_map.append({"pn": c["pn"], "lid": c["lid"]})
    return {
        "sync_type": hs.HistorySyncType.Name(hs.syncType),
        "progress": int(hs.progress),
        "messages": sum(1 for r in rows if r["op"] == "message"),
        "rows": rows,
        "chats": chats,
        "pushnames": [{"jid": str(p.ID or ""), "name": str(p.pushname or "")}
                      for p in getattr(hs, "pushnames", []) if p.pushname],
        "lid_map": [m for m in lid_map if m["pn"] and m["lid"]],
    }
