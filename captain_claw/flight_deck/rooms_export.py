"""“Download for Claude” (``docs/data-room-plan.md`` P6): a released room as a
ZIP to add to a Claude Project, for people who work there instead of (or
besides) the connector.

The ZIP holds, under a folder named after the codename:

* the released clean copies at their anonymized paths — Word, Excel and
  PowerPoint as files, PDFs and scans as the markdown they became; with
  ``text_only`` every Office file is its markdown text view instead (smaller,
  and every Claude surface reads it);
* ``ROLES.md`` — each token's role (checked for real values when saved);
* ``README-for-Claude.md`` — the connector's own rules, and how to give results
  back (the connector's ``room_deliver``, or a file with tokens the person adds
  in Flight Deck).

Nothing goes in unchecked: every piece of text in every file (an Office file
carries more than its text view) is looked through for the vault's real values
once more, and a file with a hit is left out — the owner is told which, by
path, and the room gets an alarm. The ZIP is built on the deck, kept for an
hour for one download, and the export is logged with the room's reads.
"""

from __future__ import annotations

import asyncio
import io
import re
import secrets
import time
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

from captain_claw.flight_deck import anonymize_vault as av
from captain_claw.flight_deck import rooms_job as rj
from captain_claw.flight_deck import rooms_store as rs

EXPORT_TTL = 3600.0
MAX_EXPORT_BYTES = 300 * 1024 * 1024
_OFFICE = ("docx", "xlsx", "pptx")


class ExportError(ValueError):
    """Shown to the person."""


def _exports_dir(room_id: str) -> Path:
    return rs.store().room_dir(room_id) / "exports"


def _prune(room_id: str) -> None:
    folder = _exports_dir(room_id)
    if not folder.exists():
        return
    now = time.time()
    for path in folder.glob("*.zip"):
        try:
            if now - path.stat().st_mtime > EXPORT_TTL:
                path.unlink()
        except OSError:
            pass


def export_path(room_id: str, token: str) -> Path | None:
    if not re.fullmatch(r"[A-Za-z0-9_-]{16,64}", token or ""):
        return None
    _prune(room_id)
    path = _exports_dir(room_id) / f"{token}.zip"
    return path if path.exists() else None


def _all_text(blob: bytes, ext: str) -> str:
    """Every piece of text a file carries, for the last leak check."""
    if ext in _OFFICE:
        from captain_claw.anonymize import office

        return "\n".join(s.text for s in office.prepare(blob, ext, scrub=False).segments)
    return blob.decode("utf-8", errors="replace")


def roles_md(room: dict[str, Any]) -> str:
    roles = room.get("roles") or {}
    lines = [f"# Roles in {room['codename']}", "",
             "Each token stands for a real name you can't see. Its role says what it is, never who.", ""]
    if roles:
        lines += ["| Token | Role |", "|---|---|"]
        lines += [f"| {token} | {role.replace('|', '/')} |" for token, role in sorted(roles.items())]
    else:
        lines.append("_No roles were given for this room._")
    return "\n".join(lines) + "\n"


def readme_md(room: dict[str, Any], files: int, text_only: bool) -> str:
    from captain_claw.flight_deck.rooms_mcp import INSTRUCTIONS

    return "\n".join([
        f"# {room['codename']} — an anonymized data room",
        "",
        f"{files} file(s), exported {time.strftime('%Y-%m-%d')}"
        + (" as text (Office files as markdown)." if text_only else "."),
        "Read ROLES.md first: it says what each token is.",
        "",
        "## Rules",
        "",
        INSTRUCTIONS,
        "",
        "## Giving results back",
        "",
        "If you have the Captain Data Rooms connector, deliver with it (room_deliver and the file tools) — the "
        "person reads the result in Flight Deck with the real names. Without it, write your result with the "
        "tokens kept exactly; the person adds it in Flight Deck → Data rooms → From Claude, and the names are put "
        "back there.",
        "",
    ])


async def build(room: dict[str, Any], *, text_only: bool) -> dict[str, Any]:
    """Build the ZIP; returns {token, files, held_back, size}."""
    store = rs.store()
    files = [f for f in await store.files(room["id"], released_only=True) if f["state"] == "done"]
    if not files:
        raise ExportError("Nothing is released in this room yet.")
    vault = await av.store().get(room["owner_id"], room["vault_id"])
    if vault is None:
        raise ExportError("This room's vault is gone.")
    async with av.store().lock(vault["id"]):
        state = await av.store().load_state(vault)
    needle_list = rj.needles(state)
    root = re.sub(r"[^\w\- ]+", "_", room["codename"]).strip() or "data-room"

    def make() -> tuple[bytes, list[str], int, list[str]]:
        held: list[str] = []
        held_tokens: list[str] = []
        count = 0
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            for f in files:
                ext = PurePosixPath(f["clean_path"]).suffix.lower().lstrip(".")
                clean = store.clean_file(room["id"], f["id"], "." + ext if ext else "")
                if not clean.exists():
                    held.append(f["clean_path"])
                    continue
                blob = clean.read_bytes()
                hits = rj.leaks(_all_text(blob, ext) + "\n" + f["clean_path"], needle_list)
                if hits:
                    held.append(f["clean_path"])
                    held_tokens.extend(t for t in hits if t not in held_tokens)
                    continue
                if text_only and ext in _OFFICE:
                    text_file = store.text_file(room["id"], f["id"])
                    data = text_file.read_bytes() if text_file.exists() else b""
                    name = f"{f['clean_path']}.md"
                else:
                    data, name = blob, f["clean_path"]
                z.writestr(f"{root}/{name}", data)
                count += 1
                if buf.tell() > MAX_EXPORT_BYTES:
                    raise ExportError(f"The ZIP would be over {MAX_EXPORT_BYTES // (1024 * 1024)} MB — try text only.")
            z.writestr(f"{root}/ROLES.md", roles_md(room))
            z.writestr(f"{root}/README-for-Claude.md", readme_md(room, count, text_only))
        return buf.getvalue(), held, count, held_tokens

    data, held, count, held_tokens = await asyncio.to_thread(make)
    if held_tokens:
        await store.add_alarm(room["id"], room["owner_id"], "", "export (held back)", held_tokens)
    token = secrets.token_urlsafe(24)
    folder = _exports_dir(room["id"])
    folder.mkdir(parents=True, exist_ok=True)
    await asyncio.to_thread(_prune, room["id"])
    await asyncio.to_thread((folder / f"{token}.zip").write_bytes, data)
    await store.log_read(room["id"], room["owner_id"], "", "export", chars=len(data))
    return {"token": token, "files": count, "held_back": held, "size": len(data), "text_only": text_only,
            "filename": f"{root} - for Claude.zip"}
