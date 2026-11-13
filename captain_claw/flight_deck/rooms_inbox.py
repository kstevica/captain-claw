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
    return _build_docx(title, markdown)[0]


def _build_docx(title: str, markdown: str, page_numbers: bool = False) -> tuple[bytes, list[str]]:
    from captain_claw import docx_ops

    text = _IMAGE_MD.sub(lambda m: f"[picture: {m.group(1)}]" if m.group(1) else "[picture]", markdown)
    doc, notes = docx_ops.build_document(text, title=title, page_numbers=page_numbers, image_root=lambda _src: None)
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue(), [n for n in notes if "Picture" not in n]


# ── Files built here (P4): a workbook, a deck, a document ───────────
#
# Claude sends the content as a spec, in tokens. The file is built from that
# token spec first — so every error or note Claude gets back holds tokens
# only — and the real names are then put back inside the finished file with
# the anonymizer's own Office restore (cells, sheet names with formulas
# following and Excel's 31-character cap, chart titles, slides, notes, text).
# Pictures never go in: a link would be fetched, a path read from the disk.

FILE_KINDS = {"workbook": ("xlsx", "Excel workbook"), "deck": ("pptx", "PowerPoint deck"),
              "document": ("docx", "Word document")}
MAX_SPEC_CHARS = 600_000
LOOK_KEYS = ("font", "heading_font", "heading_color", "accent_color")
# Formulas that reach outside the workbook: once the names are back, Excel
# would send them out (WEBSERVICE, IMAGE …) or a click would (HYPERLINK).
_OUTSIDE_FORMULA = re.compile(
    r"\b(WEBSERVICE|IMAGE|RTD|DDE|DDEAUTO|HYPERLINK|CALL|REGISTER|EXEC|FILTERXML|STOCKHISTORY)\s*\("
    r"|https?:|ftp:|file:|\\\\|\[[^\]]*\.(xls[xmb]?|csv|ods)\b[^\]]*\]|'?\[\d+\]", re.I)
_MD_LINK = re.compile(r"\[([^\]]*)\]\((?:https?|mailto|ftp|file):[^)]*\)|<(?:https?|mailto):[^>]*>", re.I)


def _strings(value: Any, path: str = "") -> Any:
    """Every (path, string) in a spec."""
    if isinstance(value, str):
        yield path, value
    elif isinstance(value, dict):
        for k, v in value.items():
            yield from _strings(v, f"{path}.{k}" if path else str(k))
    elif isinstance(value, list):
        for i, v in enumerate(value):
            yield from _strings(v, f"{path}[{i}]")


def _check_formulas(spec: dict[str, Any]) -> None:
    for where, text in _strings(spec):
        if (text.lstrip().startswith("=") or where.endswith("refers_to")) and _OUTSIDE_FORMULA.search(text):
            raise DeliveryError(f"{where}: formulas that reach outside the workbook (web addresses, other files, "
                                "WEBSERVICE, IMAGE, HYPERLINK …) aren't allowed in a data-room delivery.")


def _unlink(value: Any, counter: list[int]) -> Any:
    """Links become their text: in a restored file a click would send the names out."""
    if isinstance(value, str):
        def drop(m: re.Match[str]) -> str:
            counter[0] += 1
            return m.group(1) or ""
        return _MD_LINK.sub(drop, value)
    if isinstance(value, dict):
        return {k: _unlink(v, counter) for k, v in value.items()}
    if isinstance(value, list):
        return [_unlink(v, counter) for v in value]
    return value


def _build_workbook(title: str, spec: dict[str, Any]) -> tuple[bytes, list[str]]:
    from captain_claw import xlsx_ops

    sheets = spec.get("sheets")
    if not isinstance(sheets, list) or not sheets:
        raise DeliveryError("sheets is required: a list of {name, rows, header}.")
    edits = spec.get("edits") or []
    if not isinstance(edits, list):
        raise DeliveryError("edits is a list of changes ({op, …}).")
    _check_formulas({"sheets": sheets, "edits": edits})
    try:
        book, report = xlsx_ops.build_workbook(sheets, edits=edits, title=title, author="Captain")
        data = book.to_bytes()
    except xlsx_ops.XlsxError as exc:
        raise DeliveryError(f"The workbook can't be built: {exc}") from None
    notes = list(report.notes)
    notes += [f"{s}!{c}: {code} from {text}" for s, c, code, text in report.new_errors[:20]]
    return data, notes


def _strip_pictures(slide: dict[str, Any], notes: list[str], n: int) -> dict[str, Any]:
    if slide.get("image") or slide.get("images"):
        notes.append(f"Slide {n}: pictures were left out — a data-room delivery takes none.")
    return {k: v for k, v in slide.items() if k not in ("image", "images", "at")}


def _build_deck(title: str, spec: dict[str, Any]) -> tuple[bytes, list[str]]:
    from captain_claw import pptx_ops

    slides = spec.get("slides")
    if not isinstance(slides, list) or not slides:
        raise DeliveryError("slides is required: a list of {layout, title, body, table, chart, notes}.")
    notes: list[str] = []
    clean = [_strip_pictures(s, notes, i) if isinstance(s, dict) else s for i, s in enumerate(slides, 1)]
    look = {k: str(spec[k]) for k in LOOK_KEYS if spec.get(k)}
    try:
        prs, report = pptx_ops.build_deck(clean, title=title, author="Captain", page_numbers=bool(spec.get("page_numbers")),
                                   look=look or None, image_root=None)
        data = pptx_ops.deck_bytes(prs)
    except pptx_ops.PptxError as exc:
        raise DeliveryError(f"The deck can't be built: {exc}") from None
    notes += [f"{i.where}: {i.what}" for i in report.fit if i.level in ("error", "warn")][:30]
    notes += list(report.notes)
    return data, notes


def _build_document(title: str, spec: dict[str, Any]) -> tuple[bytes, list[str]]:
    markdown = str(spec.get("markdown") or "")
    if not markdown.strip():
        raise DeliveryError("markdown is required: the document's text.")
    if _IMAGE_MD.search(markdown):
        notes = ["Pictures were left out — a data-room delivery takes none."]
    else:
        notes = []
    data, more = _build_docx(title, markdown, page_numbers=bool(spec.get("page_numbers")))
    return data, notes + more


_BUILDERS = {"workbook": _build_workbook, "deck": _build_deck, "document": _build_document}


def _restore_file(blob: bytes, ext: str, state: engine.VaultState) -> tuple[bytes, int, list[str], str]:
    """(file with the names back, how many, unknown tokens, the token text of it all)."""
    from captain_claw.anonymize import office

    prep = office.prepare(blob, ext, scrub=False)
    edits, total, unknown = [], 0, []
    for seg in prep.segments:
        e, n, unk = engine.restore_edits(seg.text, state)
        edits.append(e)
        total += n
        unknown += [u for u in unk if u not in unknown]
    return prep.rebuild(edits), total, unknown, "\n".join(s.text for s in prep.segments)


async def deliver_file(room: dict[str, Any], client_id: str, *, kind: str, title: Any,
                       spec: dict[str, Any]) -> dict[str, Any]:
    """Build a file from Claude's token spec, put the names back inside it, and
    take it into the Inbox; the receipt for Claude (tokens and notes only)."""
    import asyncio

    if kind not in FILE_KINDS:
        raise DeliveryError("unknown kind of file")
    title = _clean_title(title)
    if len(json.dumps(spec, ensure_ascii=False, default=str)) > MAX_SPEC_CHARS:
        raise DeliveryError(f"The file's content is over {MAX_SPEC_CHARS:,} characters — split it into several "
                            "deliveries.")
    ext, label = FILE_KINDS[kind]
    link_notes: list[str] = []
    if kind in ("deck", "document"):
        dropped = [0]
        spec = _unlink(spec, dropped)
        if dropped[0]:
            link_notes.append(f"{dropped[0]} link(s) were turned into plain text — a data-room delivery takes no "
                              "links.")
    token_blob, notes = await asyncio.to_thread(_BUILDERS[kind], title, spec)
    notes = link_notes + notes
    row, cipher, state = await _vault(room)
    restored_blob, n_file, unknown_file, token_text = await asyncio.to_thread(_restore_file, token_blob, ext, state)
    restored_title, n_title, unknown_title = engine.restore(title, state)
    not_in_room = sorted(set(unknown_file) | set(unknown_title))
    real_values = rj.leaks(f"{title}\n{token_text}", rj.needles(state))
    name = safe_filename(restored_title, ext)
    view_tokens = await asyncio.to_thread(rj.text_view, f"{title}.{ext}", token_blob)
    view = await asyncio.to_thread(rj.text_view, name, restored_blob)
    report = {"restored": n_file + n_title, "not_in_room": not_in_room[:200], "real_values": real_values[:200],
              "parts": 1, "characters": len(token_text), "format": "markdown", "kind": kind, "file": ext,
              "notes": notes[:30]}
    payload = json.dumps({"title_tokens": title, "title": restored_title, "body_tokens": view_tokens,
                          "body": view, "format": "markdown", "kind": kind, "file": ext}, ensure_ascii=False)
    store = rs.store()
    seq = await store.add_delivery(room["id"], room["owner_id"], client_id,
                                   cipher.encrypt(payload.encode("utf-8")).decode("ascii"), len(token_text), report)
    inbox = store.room_dir(room["id"]) / "inbox"
    inbox.mkdir(parents=True, exist_ok=True)
    await asyncio.to_thread((inbox / f"{seq}.{ext}.enc").write_bytes, cipher.encrypt(restored_blob))
    await asyncio.to_thread((inbox / f"{seq}.tokens.{ext}.enc").write_bytes, cipher.encrypt(token_blob))
    if real_values:
        await store.add_alarm(room["id"], room["owner_id"], client_id, f"room_deliver_{kind}", real_values)
        log.warning("Data room file delivery holds real values Claude wrote", room=room["id"], tokens=len(real_values))
    try:
        found = usage.token_contexts([title, token_text], set(state.by_token))
        if found:
            await av.store().record_uses(
                row, [av.TokenUse(t, "back", n, c, model=TO_MODEL) for t, (n, c) in found.items()],
                channel="data-room:delivery", actor=f"mcp:{client_id}", label=f"delivery #{seq} ({ext})")
    except Exception as exc:  # noqa: BLE001
        log.warning("Data room delivery: token history not written", error=type(exc).__name__)
    await _tell_owner(room, seq)
    receipt: dict[str, Any] = {"delivered": f"#{seq}", "file": f"{label} (.{ext})", "restored": report["restored"],
                               "note": "The person gets the file in Flight Deck with the real names. You don't get "
                                       "the restored file."}
    if notes:
        receipt["notes"] = notes[:30]
    if not_in_room:
        receipt["not_in_room"] = not_in_room[:50]
        receipt["note"] += " Tokens not in this room stay as you wrote them — use only tokens from the room."
    return receipt


def file_path(room_id: str, seq: int, ext: str, *, tokens: bool = False) -> Any:
    folder = rs.store().room_dir(room_id) / "inbox"
    return folder / (f"{seq}.tokens.{ext}.enc" if tokens else f"{seq}.{ext}.enc")
