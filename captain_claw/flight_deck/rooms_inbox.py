"""The Inbox (``docs/data-room-plan.md`` P3): what Claude sends back to a data
room, with the real names put back here, for the room's owner only.

``room_deliver`` (``rooms_mcp``) hands :func:`deliver` a title and markdown in
tokens. Long content comes in parts (part 1 of N …), kept encrypted until the
last one arrives and joined exactly as sent. Then:

* every token of the room's vault is put back (bracketed or bare);
* tokens the room doesn't know stay as written and are listed;
* real vault values Claude wrote itself are found (it was told them, or it
  guessed) — an alarm for the owner, and NEVER mentioned to Claude: telling it
  would confirm a guess;
* both the tokens version and the restored one are stored encrypted with the
  room vault's key; the receipt Claude gets has counts and tokens only.

The owner gets a bell notice with generic text (no title, which could carry
real names once restored): the bell feeds the Line and the owner's channels.

The restored markdown is someone else's writing: shown without pictures or
live links (a picture Claude planted would fetch a URL with the real names in
it), and the Word download takes no pictures (the builder would otherwise read
a local file Claude named).
"""

from __future__ import annotations

import hashlib
import io
import json
import re
from typing import Any

from captain_claw.anonymize import engine, usage
from captain_claw.flight_deck import anonymize_vault as av
from captain_claw.flight_deck import rooms_job as rj
from captain_claw.flight_deck import rooms_store as rs
from captain_claw.logging import get_logger

log = get_logger(__name__)

MAX_TITLE = 300
MAX_PART_CHARS = 200_000
MAX_PARTS = 50
MAX_DELIVERY_CHARS = 2_000_000
FORMATS = ("markdown", "text")
TO_MODEL = "Claude (data-room connector)"
_IMAGE_MD = re.compile(r"!\[([^\]]*)\]\([^)]*\)")


class DeliveryError(ValueError):
    """Told to Claude (no real value in it)."""


def _clean_title(title: str) -> str:
    title = re.sub(r"\s+", " ", str(title or "")).strip()[:MAX_TITLE]
    if not title:
        raise DeliveryError("A title is required.")
    return title


async def _vault(room: dict[str, Any]) -> tuple[dict[str, Any], Any, engine.VaultState]:
    vaults = av.store()
    row = await vaults.get(room["owner_id"], room["vault_id"])
    if row is None:
        raise DeliveryError("This room can't take deliveries right now.")
    async with vaults.lock(row["id"]):
        state = await vaults.load_state(row)
    return row, vaults.cipher(row), state


async def deliver(room: dict[str, Any], client_id: str, *, title: Any, content: Any, fmt: Any = "markdown",
                  part: Any = None, of: Any = None) -> dict[str, Any]:
    """Take a delivery (or one part of it) into the room's Inbox; the receipt for Claude."""
    title = _clean_title(title)
    content = str(content or "")
    fmt = str(fmt or "markdown").lower()
    if fmt not in FORMATS:
        fmt = "markdown"
    if not content.strip():
        raise DeliveryError("content is empty.")
    if len(content) > MAX_PART_CHARS:
        raise DeliveryError(f"One call takes at most {MAX_PART_CHARS:,} characters — send it in parts "
                            "(part 1 of N, part 2 of N …).")
    row, cipher, state = await _vault(room)
    store = rs.store()
    total = 1
    if of not in (None, "", 0, 1, "1"):
        try:
            total, part = int(of), int(part or 0)
        except (TypeError, ValueError):
            raise DeliveryError("part and of are numbers (part 1 of 3 …).") from None
        if not 1 <= total <= MAX_PARTS or not 1 <= part <= total:
            raise DeliveryError(f"part must be between 1 and of, and of at most {MAX_PARTS}.")
    if total > 1:
        key = hashlib.sha256(f"{title}\n{total}".encode()).hexdigest()[:32]
        have = await store.add_part(room["id"], client_id, key, part, total,
                                    cipher.encrypt(content.encode("utf-8")).decode("ascii"))
        missing = [n for n in range(1, total + 1) if n not in have]
        if missing:
            return {"received": f"part {part} of {total}", "waiting_for_parts": missing,
                    "note": "Send the other parts with the same title and of; they are joined exactly as sent, "
                            "in order."}
        parts = await store.take_parts(room["id"], client_id, key, total)
        content = "".join(cipher.decrypt(p.encode("ascii")).decode("utf-8") for p in parts)
        if len(content) > MAX_DELIVERY_CHARS:
            raise DeliveryError(f"The parts add up to more than {MAX_DELIVERY_CHARS:,} characters.")

    restored_title, n_title, unknown_title = engine.restore(title, state)
    restored_body, n_body, unknown_body = engine.restore(content, state)
    not_in_room = sorted(set(unknown_title) | set(unknown_body))
    real_values = rj.leaks(f"{title}\n{content}", rj.needles(state))
    report = {"restored": n_title + n_body, "not_in_room": not_in_room[:200], "real_values": real_values[:200],
              "parts": total, "characters": len(content), "format": fmt}
    payload = json.dumps({"title_tokens": title, "title": restored_title, "body_tokens": content,
                          "body": restored_body, "format": fmt}, ensure_ascii=False)
    seq = await store.add_delivery(room["id"], room["owner_id"], client_id,
                                   cipher.encrypt(payload.encode("utf-8")).decode("ascii"), len(content), report)
    if real_values:
        await store.add_alarm(room["id"], room["owner_id"], client_id, "room_deliver", real_values)
        log.warning("Data room delivery holds real values Claude wrote", room=room["id"], tokens=len(real_values))
    try:
        found = usage.token_contexts([title, content], set(state.by_token))
        if found:
            await av.store().record_uses(
                row, [av.TokenUse(t, "back", n, c, model=TO_MODEL) for t, (n, c) in found.items()],
                channel="data-room:delivery", actor=f"mcp:{client_id}", label=f"delivery #{seq}")
    except Exception as exc:  # noqa: BLE001 — history is a nicety; the delivery is in
        log.warning("Data room delivery: token history not written", error=type(exc).__name__)
    await _tell_owner(room, seq)
    receipt: dict[str, Any] = {"delivered": f"#{seq}", "restored": report["restored"], "characters": len(content),
                               "note": "The person reads it in Flight Deck with the real names. You don't get the "
                                       "restored text."}
    if total > 1:
        receipt["parts_joined"] = total
    if not_in_room:
        receipt["not_in_room"] = not_in_room[:50]
        receipt["note"] += " Tokens not in this room stay as you wrote them — use only tokens from the room."
    return receipt


async def _tell_owner(room: dict[str, Any], seq: int) -> None:
    """A bell notice with generic text: no title, no names (the bell feeds the
    Line and the owner's channels)."""
    try:
        from captain_claw.flight_deck.auth import get_db

        await get_db().add_notification(
            room["owner_id"], "data_room_delivery", "Claude delivered a result",
            f"Delivery #{seq} is in the data room {room['codename']}. Open Data rooms to read it.",
            ref_type="data_room", ref_id=room["id"])
    except Exception as exc:  # noqa: BLE001 — the delivery is in either way
        log.debug("Data room delivery notice failed", error=type(exc).__name__)


def open_payload(cipher: Any, delivery: dict[str, Any]) -> dict[str, Any]:
    return json.loads(cipher.decrypt(delivery["payload"].encode("ascii")).decode("utf-8"))


def safe_filename(title: str, ext: str) -> str:
    stem = re.sub(r"[^\w\-. ]+", "_", title, flags=re.UNICODE).strip(" ._")[:120] or "delivery"
    return f"{stem}.{ext}"


def docx_bytes(title: str, markdown: str) -> bytes:
    """The restored markdown as a Word file, built here — no pictures at all."""
    from captain_claw import docx_ops

    text = _IMAGE_MD.sub(lambda m: f"[picture: {m.group(1)}]" if m.group(1) else "[picture]", markdown)
    doc, _notes = docx_ops.build_document(text, title=title, image_root=lambda _src: None)
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()
